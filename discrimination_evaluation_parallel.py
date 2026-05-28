import argparse
import asyncio
import json
import pickle
import random
import re
import time
from datetime import datetime
from pathlib import Path

import nest_asyncio
import numpy as np
import yaml
from tqdm import tqdm

from data.dataloader import ML_1M, Amazon_Book, Amazon_Beauty
from model.intent_predictor import IntentPredictor
from llama_index.core.prompts import PromptTemplate


INTENT_ID2NAME = {
    0: "stay_profile",
    1: "follow_recent",
    2: "explore",
}

INTENT_DESCRIPTIONS = {
    "stay_profile": (
        "The user's recent taste is consistent with their long-term profile (low drift). "
        "Prefer items that match the long-term genre profile."
    ),
    "follow_recent": (
        "The user's recent taste has shifted from their long-term profile (high drift). "
        "Prefer items that align with the recent genre distribution, even if they differ from the long-term profile."
    ),
    "explore": (
        "The user is seeking novelty — items not well-represented in either the long-term profile or recent history. "
        "Prefer items from underrepresented genres; avoid the dominant genres from both profiles."
    ),
}

# Alphabetically sorted ML-1M genres — matches the genre2idx order used during training
ML1M_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Alphabetically sorted Amazon Book genre labels — L1 categories only, junk/promotional excluded.
# 35 genres. Matches genre2idx from data construction (state_transition.ipynb Cell 20 / Cell 22).
AMAZON_BOOK_GENRES = [
    'Amazon Publishing',
    'Arts & Photography',
    'Audible Books & Originals',
    "Baby's First Library",
    'Biographies & Memoirs',
    'Business & Money',
    "Children's Book Awards",
    "Children's Books",
    'Christian Books & Bibles',
    'Comics & Graphic Novels',
    'Computers & Technology',
    'Cookbooks, Food & Wine',
    'Crafts, Hobbies & Home',
    'Education & Teaching',
    'Engineering & Transportation',
    'Foreign Language Books',
    'Health, Fitness & Dieting',
    'History',
    'Humor & Entertainment',
    'LGBTQ+ Books',
    'Law',
    'Literature & Fiction',
    'Medical Books',
    'Mystery, Thriller & Suspense',
    'Parenting & Relationships',
    'Politics & Social Sciences',
    'Reference',
    'Religion & Spirituality',
    'Romance',
    'Science & Math',
    'Science Fiction & Fantasy',
    'Self-Help',
    'Sports & Outdoors',
    'Teen & Young Adult',
    'Travel',
]

# Active genre names — overridden in main() based on --dataset
_GENRE_NAMES = ML1M_GENRES


class ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_idx", type=int, default=0)
    parser.add_argument("-e", "--end_idx", type=int, default=50)
    parser.add_argument("-r", "--rounds", type=int, default=3, help="Max sim_eval rounds (time steps) to evaluate per user. Default: 3.")
    parser.add_argument("-rep", "--repetition", type=int, default=3)
    parser.add_argument("-c", "--config_file", type=str, default="config/config_general.yaml")
    parser.add_argument("-nn", "--neg_num", type=int, default=5)
    parser.add_argument("-tn", "--test_num", type=int, default=1)
    parser.add_argument("-ts", "--test_sampling", type=str, default="sequence") # in time order
    parser.add_argument("-sf", "--shuffle", action="store_true")
    parser.add_argument("--skip_num", type=int, default=25, help="number of used interactions")
    parser.add_argument("-d", "--data-path", type=str, default=None)
    parser.add_argument("-ckpt","--ckpt-dir", type=str, default=None, help="If set, uses predicted intent else ground-truth.")
    parser.add_argument("--no-intent", action="store_true", help="Profile-only baseline: omit intent and bias from the prompt entirely.")
    parser.add_argument("--output-path", type=str, default="output/simulation/book/discriminate_1_6/", help="Path to save per-sample interaction records as JSONL.")
    parser.add_argument("-p","--profile-path", type=str, default=None, help="Optional JSON/JSONL/pkl/parquet file mapping user_id to profile text (or genre weights), overrides embedded nl_static_profile.")
    parser.add_argument("--emb-dir", type=str, default=None,
                        help="Path to embeddings/ directory (items.npy, profiles.npy). "
                             "When set, reconstructs embedding-space inputs for the intent model "
                             "from pre-computed tables — no embedding model required at eval time.")
    parser.add_argument("--decay-factor", type=float, default=1.0, help="Exponential decay for window item embeddings (1.0 = uniform, matches training default).")
    parser.add_argument("--prompt-yaml", type=str, default="config/prompt.yaml", help="YAML file containing prompt templates.")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="If set, route to profile-only prompt when intent confidence is below this value.")
    parser.add_argument("--random-intent", action="store_true", help="Baseline: randomly assign one of the three intents instead of using ground-truth or predicted intent.")
    parser.add_argument("--recent-items", action="store_true", help="Profile-only mode: show last 5 interacted items as recent behaviour instead of genre vector.")
    parser.add_argument("--user-ids", type=str, default="baseline/amazon_book_sub_userids.txt", help="Path to a txt file with one user ID per line. If provided, only those users are evaluated (then --start_idx/--end_idx slicing is applied).")
    parser.add_argument("--concurrency", type=int, default=8, help="Max number of samples evaluated concurrently (default: 8).")
    parser.add_argument("--dataset", type=str, default="auto",
                        choices=["auto", "ml1m", "amazon_book", "amazon_beauty"],
                        help="Dataset type. 'auto' infers from --data-path.")
    parser.add_argument("--genre-list", type=str, default=None,
                        help="Path to JSON file (list or {name: idx} dict) with ordered genre names. "
                             "Overrides the built-in genre list. Use with datasets built from "
                             "genres_all (unfiltered) to match the correct genre dimensions. "
                             "Generated by the enrich_profiles notebook cell as genre_names_{mode}.json.")
    return parser.parse_args()


def load_sim_eval_for_users(sim_eval_path, target_user_ids, is_string_ids=False):
    """Memory-efficient targeted load: reads only rows whose user_id is in target_user_ids."""
    sim_eval = np.load(sim_eval_path)

    # Load companion metadata pkl (nl_static_profile, etc.) when present.
    meta_pkl_path = Path(sim_eval_path).with_name(Path(sim_eval_path).stem + "_meta.pkl")
    meta_list = None
    if meta_pkl_path.exists():
        with open(meta_pkl_path, "rb") as _f:
            meta_list = pickle.load(_f)

    if "user_id" not in sim_eval.files:
        raise KeyError(
            "sim_eval npz does not contain a user_id array. "
            "Re-generate the data with the updated notebook, or drop --profile_path."
        )

    user_ids_arr = sim_eval["user_id"]
    if is_string_ids:
        # Beauty user IDs are alphanumeric strings stored as -1 in npz; use meta pkl for lookup.
        if meta_list is None:
            print("  No meta pkl found — cannot look up string user IDs.")
            return []
        target_set = set(str(u) for u in target_user_ids)
        indices = np.array([
            i for i, m in enumerate(meta_list)
            if str(m.get("user_id", "")) in target_set
        ], dtype=np.int64)
        if len(indices) == 0:
            sample_ids = [str(m.get("user_id")) for m in meta_list[:10]]
            print(f"  No matching users found. Sample user_ids in meta: {sample_ids}")
            return []
    else:
        target_int_ids = np.array([int(uid) for uid in target_user_ids], dtype=np.int64)
        indices        = np.where(np.isin(user_ids_arr, target_int_ids))[0]
        if len(indices) == 0:
            sample_ids = sorted(set(user_ids_arr[user_ids_arr != -1].tolist()))[:10]
            print(f"  No matching users found. Sample user_ids in sim_eval: {sample_ids}")
            return []

    # bulk-load arrays for matching indices only
    static_profiles = sim_eval["static_profile"][indices]
    state_sequences = sim_eval["state_sequence"][indices]
    intent_labels   = sim_eval["intent_label"][indices]
    time_indices    = sim_eval["time_index"][indices] if "time_index" in sim_eval.files else None
    score_keys      = [k for k in sim_eval.files if k.startswith("score_")]
    scores_data     = {k: sim_eval[k][indices] for k in score_keys}
    window_item_ids = sim_eval["window_item_ids"][indices] if "window_item_ids" in sim_eval.files else None
    profile_emb_idx = sim_eval["profile_emb_idx"][indices] if "profile_emb_idx" in sim_eval.files else None
    recent_vectors  = sim_eval["recent_vector"][indices]   if "recent_vector"   in sim_eval.files else None

    rows = []
    for i, npz_idx in enumerate(indices.tolist()):
        intent_label = int(intent_labels[i])
        row = {
            "sample_idx":     npz_idx,
            "static_profile": static_profiles[i],
            "state_sequence": state_sequences[i],
            "intent_label":   intent_label,
            "intent_name":    INTENT_ID2NAME.get(intent_label, "unknown"),
            "user_id":        (str(meta_list[npz_idx]["user_id"]) if is_string_ids and meta_list
                               else int(user_ids_arr[npz_idx])),
        }
        if time_indices is not None:
            tidx = int(time_indices[i])
            row["time_index"] = tidx if tidx != -1 else None
        if score_keys:
            row["scores"] = {k[len("score_"):]: float(scores_data[k][i]) for k in score_keys}
        if window_item_ids is not None:
            row["window_item_ids"] = window_item_ids[i]   # (T, W) int32
        if profile_emb_idx is not None:
            row["profile_emb_idx"] = int(profile_emb_idx[i])
        if recent_vectors is not None:
            row["recent_vector"] = recent_vectors[i]      # (T, num_genres)
        if meta_list is not None and npz_idx < len(meta_list):
            row["nl_static_profile"] = meta_list[npz_idx].get("nl_static_profile")
        rows.append(row)

    return rows


def _format_recent_items(items):
    if not items:
        return ""
    lines = []
    for i, item in enumerate(items, 1):
        title = next((item[k] for k in ("title", "book_title", "product_title") if k in item), None)
        genre = item.get("genres") or item.get("genre", "")
        rating = item.get("user_rating", "")
        parts = [title] if title else []
        if genre:
            parts.append(f"[{genre}]")
        if rating:
            parts.append(f"rating: {rating}")
        lines.append(f"{i}. {' '.join(str(p) for p in parts)}")
    return "\n".join(lines)


def _sample_base_context(sample):
    latest_state = sample["state_sequence"][-1]
    drift_val = float(latest_state[-1])
    if drift_val > 0.6:
        drift_label = "high"
    elif drift_val > 0.3:
        drift_label = "moderate"
    else:
        drift_label = "low"
    recent_drift = f"Drift from long-term profile: {drift_val:.3f} ({drift_label})"

    static_profile_text = (
        sample.get("nl_static_profile")
        or vector_to_text(sample["static_profile"], _GENRE_NAMES)
    )
    recent_items = sample.get("recent_items")
    # State dim == num_genres + 1 means vector/genre mode; larger means emb/hybrid mode.
    emb_based_state = len(latest_state) > len(_GENRE_NAMES) + 1
    if recent_items:
        recent_history = "Recent interactions (last 5 items):\n" + _format_recent_items(recent_items)
    elif emb_based_state:
        recent_vec = sample.get("recent_vector")
        genre_vec = (
            np.asarray(recent_vec[-1])[:len(_GENRE_NAMES)]
            if recent_vec is not None
            else None
        )
        if genre_vec is not None and genre_vec.sum() > 0:
            recent_history = f"Recent genre preferences: {vector_to_text(genre_vec, _GENRE_NAMES)}\n"
        else:
            recent_history = ""
    else:
        recent_history = (
            f"Recent genre preferences: {vector_to_text(latest_state[:len(_GENRE_NAMES)], _GENRE_NAMES)}\n"
        )

    return {
        "static_profile_text": static_profile_text,
        "recent_history":      recent_history,
        "recent_drift":        recent_drift,
    }


def _explore_avoid_hint(sample):
    """Return a 'avoid these genres' hint for the explore intent."""
    static = np.asarray(sample["static_profile"])
    latest_state = np.asarray(sample["state_sequence"][-1])

    n = len(_GENRE_NAMES)
    static = static[:n]
    top_static = [_GENRE_NAMES[i] for i in np.argsort(static)[::-1] if static[i] > 0][:2]

    emb_based_state = len(latest_state) > n + 1
    if emb_based_state:
        recent_vec = sample.get("recent_vector")
        if recent_vec is not None:
            recent = np.asarray(recent_vec[-1])[:n]
            top_recent = [_GENRE_NAMES[i] for i in np.argsort(recent)[::-1] if recent[i] > 0][:2]
            avoid = list(dict.fromkeys(top_static + top_recent))[:3]
        else:
            avoid = top_static[:3]
    else:
        recent = latest_state[:-1][:n]  # drop drift, then slice to genre dims
        top_recent = [_GENRE_NAMES[i] for i in np.argsort(recent)[::-1] if recent[i] > 0][:2]
        avoid = list(dict.fromkeys(top_static + top_recent))[:3]

    if avoid:
        return f" Genres to avoid (dominant in both profiles): {', '.join(avoid)}."
    return ""


def sim_eval_to_prompt_context(sample, intent_name, bias_vector):
    scores = sample.get("scores", {})
    s_profile = scores.get("profile_similarity")
    s_recent  = scores.get("recent_similarity")
    drift     = scores.get("drift")

    # For explore intent, append a dynamic genre-exclusion hint
    intent_desc = INTENT_DESCRIPTIONS.get(intent_name, "")
    if intent_name == "explore":
        intent_desc += _explore_avoid_hint(sample)

    return {
        **_sample_base_context(sample),
        "predicted_intent": intent_name,
        "intent_description": intent_desc,
        "profile_similarity": f"{s_profile:.2f}" if s_profile is not None else "N/A",
        "recent_similarity":  f"{s_recent:.2f}"  if s_recent  is not None else "N/A",
        "drift":              f"{drift:.2f}"      if drift     is not None else "N/A",
        "profile_weight": f"{bias_vector['profile']:.2f}",
        "recent_weight":  f"{bias_vector['recent']:.2f}",
        "novelty_weight": f"{bias_vector['novelty']:.2f}",
    }


def _build_predictor_inputs(sim_sample, intent_predictor):
    """Return (state_seq, profile) numpy arrays matching the predictor's input_repr.

    concat: cat(genre_arr, emb_arr) along last axis — matches models trained with
            --input-repr concat.
    auto/embedding: pure embedding when available, genre fallback otherwise.
    genre:  always genre histograms.
    """
    has_emb = "emb_state_sequence" in sim_sample
    if has_emb and getattr(intent_predictor, "input_repr", "auto") == "concat":
        # Mirror the training _get_inputs concat branch exactly:
        # L2-normalize the genre profile slice (dims 0:-1, excluding drift) so
        # it has unit norm like the embedding half. Drift (last dim) is kept raw.
        state_arr  = np.asarray(sim_sample["state_sequence"], dtype=np.float32)   # (T, 19)
        genre_s    = state_arr[..., :-1]                                           # (T, 18)
        norms_s    = np.linalg.norm(genre_s, axis=-1, keepdims=True)
        genre_s_l2 = genre_s / np.maximum(norms_s, 1e-8)                          # (T, 18) unit-norm
        drift_s    = state_arr[..., -1:]                                           # (T,  1)
        state_seq  = np.concatenate([genre_s_l2, drift_s,
                                     sim_sample["emb_state_sequence"]], axis=-1)   # (T, 19+D)

        profile_arr = np.asarray(sim_sample["static_profile"], dtype=np.float32)  # (18,)
        norm_p      = np.linalg.norm(profile_arr)
        profile_l2  = profile_arr / max(float(norm_p), 1e-8)                      # (18,) unit-norm
        profile     = np.concatenate([profile_l2,
                                      sim_sample["emb_static_profile"]], axis=-1) # (18+D,)
    else:
        state_seq = sim_sample.get("emb_state_sequence", sim_sample["state_sequence"])
        profile   = sim_sample.get("emb_static_profile", sim_sample["static_profile"])
    return state_seq, profile


def _get_confidence(sim_sample, intent_predictor):
    """Return (intent_name, bias_vector, confidence) for confidence-gated routing.

    For model predictions: delegates to IntentPredictor.predict_with_confidence,
    which returns the softmax max-probability of the predicted class.
    For ground-truth: reads scores["confidence"] written by the robust construction
    module (normalized top-intent share of total evidence); None if the data was
    built without it.
    """
    if intent_predictor is not None:
        state_seq, profile = _build_predictor_inputs(sim_sample, intent_predictor)
        return intent_predictor.predict_with_confidence(state_seq, profile)
    intent_name = sim_sample["intent_name"]
    scores = sim_sample.get("scores", {})
    bias_vector = {
        "profile": scores.get("profile_similarity", 0.33),
        "recent":  scores.get("recent_similarity",  0.33),
        "novelty": scores.get("novelty",             0.33),
    }
    confidence = scores.get("confidence")  # None when not built with robust module
    return intent_name, bias_vector, confidence


def vector_to_text(vector, genre_names=None):
    if genre_names is not None:
        entries = sorted(
            [(genre_names[i], v) for i, v in enumerate(vector) if v > 0],
            key=lambda x: x[1], reverse=True,
        )
        return ", ".join(f"{name} ({v:.1%})" for name, v in entries) or "none"
    return ", ".join(f"{i}={v:.3f}" for i, v in enumerate(vector) if v > 0)


async def discrimination_evaluation(
    llm,
    sim_sample,
    positive_items,
    negative_items,
    shuffle=False,
    repetition=3,
    intent_predictor=None,
    use_intent=True,
    discrimination_template=None,
    profile_only_template=None,
    confidence_threshold=None,
    random_intent=False,
):
    used_intent = None
    used_confidence = None

    if not use_intent:
        prompt_template = profile_only_template
        prompt_mode = "profile_only"
        base_context = _sample_base_context(sim_sample)
        base_context["recent_drift"] = ""
    elif confidence_threshold is not None:
        intent_name, bias_vector, confidence = _get_confidence(sim_sample, intent_predictor)
        used_intent = intent_name
        used_confidence = confidence
        if confidence is not None and confidence >= confidence_threshold:
            prompt_template = discrimination_template
            prompt_mode = "intent"
            base_context = sim_eval_to_prompt_context(sim_sample, intent_name, bias_vector)
        else:
            prompt_template = profile_only_template
            prompt_mode = "profile_only"
            base_context = _sample_base_context(sim_sample)
    elif random_intent:
        prompt_template = discrimination_template
        prompt_mode = "random_intent"
        intent_name = random.choice(list(INTENT_ID2NAME.values()))
        arr = np.random.rand(3)
        arr = arr / arr.sum()
        bias_vector = {
            "profile": arr[0],
            "recent": arr[1],
            "novelty": arr[2],
        }
        bias_vector = {
            "profile": 0.33,
            "recent": 0.33,
            "novelty": 0.33,
        }
        used_intent = intent_name
        base_context = sim_eval_to_prompt_context(sim_sample, intent_name, bias_vector)
    else:
        prompt_template = discrimination_template
        prompt_mode = "intent"
        if intent_predictor is not None:
            
            state_seq, profile = _build_predictor_inputs(sim_sample, intent_predictor)
            intent_name, bias_vector = intent_predictor.predict(state_seq, profile)
        else:
            intent_name = sim_sample["intent_name"]
            scores = sim_sample.get("scores", {})
            bias_vector = {
                "profile": scores.get("profile_similarity", 0.33),
                "recent":  scores.get("recent_similarity",  0.33),
                "novelty": scores.get("novelty",             0.33),
            }
        used_intent = intent_name
        base_context = sim_eval_to_prompt_context(sim_sample, intent_name, bias_vector)

    recommendation_list = positive_items + negative_items
    random.shuffle(recommendation_list)

    scores = []
    token_usage = []
    interactions = []
    for _ in range(repetition):
        if shuffle:
            random.shuffle(recommendation_list)

        positive_indexes = [recommendation_list.index(item) + 1 for item in positive_items]
        messages = prompt_template.format_messages(
            **base_context,
            positive_num=len(positive_items),
            candidate_items="\n".join(recommendation_list_to_prompt(recommendation_list)),
        )
    
        try:
            response = await llm.achat(messages=messages)
        except Exception as exc:
            print(f"Error during LLM call: {exc}")
            continue
        # print(messages)
        # return
    
        raw_response = getattr(response, "raw", None)
        if hasattr(raw_response, "usage"):
            token_usage.append(raw_response.usage)

        predicted_indexes = parse_selected_indexes(
            general_parser(response.message.content),
            max_index=len(recommendation_list),
        )[:len(positive_items)]

        if predicted_indexes:
            scores.append(get_overlap_ratio(predicted_indexes, positive_indexes))
            interactions.append({
                "positive_item": [_extract_item_id(recommendation_list[i - 1]) for i in positive_indexes],
                # "candidates": recommendation_list,
                "selected_item": [_extract_item_id(recommendation_list[i - 1]) for i in predicted_indexes],
            })

    if not scores:
        return "Err", 0.0, token_usage, [], used_intent, prompt_mode, used_confidence
    return round(sum(scores) / len(scores), 2), float(np.std(scores)), token_usage, interactions, used_intent, prompt_mode, used_confidence


def parse_selected_indexes(response_text, max_index=None):
    parsed = parse_json_object(response_text)
    if parsed and "selected_item_index" in parsed:
        value = parsed["selected_item_index"]
        indexes = value if isinstance(value, list) else [value]
        return validate_indexes([int(i) for i in indexes], max_index)

    if "]::" in response_text:
        response_text = response_text.split("]::", 1)[1]
    return validate_indexes([int(x) for x in re.findall(r"\d+", response_text)], max_index)


def validate_indexes(indexes, max_index=None):
    if max_index is None:
        return indexes
    return [idx for idx in indexes if 1 <= idx <= max_index]


def parse_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_config(config_file):
    with open(config_file) as f:
        return ConfigDict(yaml.safe_load(f))


def load_prompts(yaml_path):
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_llm(config):
    import utils.utils as utils
    return utils.load_LLM(config)


def general_parser(response):
    return response.split("</think>")[-1].strip() if response else ""


def get_overlap_ratio(y_pred, y_list):
    return len(set(y_pred).intersection(y_list)) / len(y_list)


def _extract_item_id(item):
    for key in ("title", "movie_title", "book_title", "product_title", "name"):
        if key in item and item[key]:
            return item[key]
    return next((v for v in item.values() if v is not None), None)


def recommendation_list_to_prompt(recommendation_list):
    if not recommendation_list:
        return []

    if "title" in recommendation_list[0]:
        keys = ["title", "genres", "description", "average_rating", "popularity"]
    elif "book_title" in recommendation_list[0]:
        keys = ["book_title", "genre", "brief_description", "average_rating", "price"]
    elif "product_title" in recommendation_list[0]:
        keys = ["product_title", "average_rating", "rating_number", "price", "store", "product_brand"]
    else:
        keys = []

    formatted_items = []
    for idx, item in enumerate(recommendation_list, start=1):
        item_info = f"{idx}. "
        for key, value in item.items():
            if "id" in key:
                continue
            if keys and key not in keys:
                continue
            if value is None or str(value) == "":
                continue
            item_info += f"{key.replace('_', ' ')}: {value}\n"
        formatted_items.append(item_info)
    return formatted_items


def load_candidate_items(dataloader, userid, time_index, args):
    """Return the exact positive item at time_index and sampled negative items.

    time_index is the index into the user's chronologically sorted interactions,
    matching positives[t] from training_data_construction.build_training_data.
    Falls back to the old 80%-cutoff positive when time_index is None.
    """
    if time_index is not None:
        # sample negatives only (pos_num=0, test_num=0)
        _, negative_items = dataloader.get_items_for_profile(
            userid=userid,
            pos_num=0,
            neg_num=args.neg_num,
            test_num=0,
        )

        interacted_df = dataloader.get_interacted_items(userid=userid, mergeitem=True)
        interacted_records = interacted_df.reset_index(drop=True).to_dict(orient="records")

        if time_index >= len(interacted_records):
            raise IndexError(
                f"time_index={time_index} out of range for user {userid} "
                f"(total interactions: {len(interacted_records)})"
            )

        target_item = interacted_records[time_index]
        if negative_items:
            target_item = {k: target_item[k] for k in negative_items[0] if k in target_item}

        return [target_item], negative_items
    else:
        print("No time index")
        # legacy fallback: use the original 80%-cutoff positive item
        _, negative_items, positive_items = dataloader.get_items_for_profile(
            userid=userid,
            pos_num=args.skip_num,
            neg_num=args.neg_num,
            test_sampling=args.test_sampling,
            test_num=args.test_num,
        )
        positive_items = [
            {key: item[key] for key in negative_items[0].keys() if key in item}
            for item in positive_items
        ]
        return positive_items, negative_items

def load_recent_items(dataloader, userid, time_index, n=5):
    """Return the n items interacted with immediately before time_index."""
    interacted_df = dataloader.get_interacted_items(userid=userid, mergeitem=True)
    interacted_records = interacted_df.reset_index(drop=True).to_dict(orient="records")
    start = max(0, time_index - n)
    return interacted_records[start:time_index]


def get_sorted_users(sim_eval_path, is_string_ids=False):
    """Return the sorted list of unique user_ids present in the sim_eval npz."""
    if is_string_ids:
        meta_pkl_path = Path(sim_eval_path).with_name(Path(sim_eval_path).stem + "_meta.pkl")
        if not meta_pkl_path.exists():
            return []
        with open(meta_pkl_path, "rb") as _f:
            meta_list = pickle.load(_f)
        return sorted(set(str(m["user_id"]) for m in meta_list if m.get("user_id") is not None))
    sim_eval = np.load(sim_eval_path)
    if "user_id" not in sim_eval.files:
        return []
    user_ids = sim_eval["user_id"]
    return sorted(set(int(u) for u in user_ids if int(u) != -1))


def apply_emb_inputs(eval_samples, emb_dir, decay_factor=1.0):
    """Replace static_profile and state_sequence with pre-computed embedding vectors.

    Loads shared tables (items.npy, profiles.npy) from emb_dir and reconstructs
    per-sample tensors from the integer indices stored in the npz — identical to
    what TransitionDataset does during training, but in numpy (no model required).

    Samples that lack profile_emb_idx or window_item_ids are left unchanged.
    """
    emb_dir      = Path(emb_dir)
    item_embs    = np.load(emb_dir / "items.npy").astype(np.float32)     # (N_items, D)
    profile_embs = np.load(emb_dir / "profiles.npy").astype(np.float32)  # (N_profiles, D)

    n_applied = 0
    for sample in eval_samples:
        p_idx   = sample.get("profile_emb_idx")
        win_ids = sample.get("window_item_ids")
        if p_idx is None or win_ids is None:
            continue

        # Store under separate keys so genre-based arrays remain intact for
        # prompt rendering (_sample_base_context, vector_to_text, drift label).
        sample["emb_static_profile"] = profile_embs[int(p_idx)]   # (D,)

        # Recent-state embeddings: (T, W) indices → weighted mean → L2-normalised (T, D)
        win_ids = np.asarray(win_ids, dtype=np.int32)              # (T, W)
        W = win_ids.shape[1]
        if decay_factor == 1.0:
            state_emb = item_embs[win_ids].mean(axis=1)            # (T, D)
        else:
            weights = np.array(
                [decay_factor ** (W - 1 - i) for i in range(W)], dtype=np.float32
            )
            weights /= weights.sum()
            state_emb = (item_embs[win_ids] * weights[None, :, None]).sum(axis=1)
        norms = np.linalg.norm(state_emb, axis=-1, keepdims=True)
        sample["emb_state_sequence"] = state_emb / np.maximum(norms, 1e-8)   # (T, D)
        n_applied += 1

    D = profile_embs.shape[1]
    print(f"Applied embedding inputs: embed_dim={D}, {n_applied}/{len(eval_samples)} samples from {emb_dir}")


def load_profiles(profile_path):
    """Load a user_id -> profile_text mapping from JSON, JSONL, pkl, or parquet.

    JSON: {"user_id": "profile text", ...}
    JSONL: one {"user_id": ..., "nl_static_profile": ...} object per line
    pkl: dict {user_id: profile_text}
    parquet: genre-weight table (userid | Action | Adventure | ...) produced by
             genre_profile_generation.save_genre_profiles; each user's genre
             vector is converted to a text description via vector_to_text.
    """
    path = Path(profile_path)
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        # Enriched NL profile parquet (userid | nl_static_profile)
        if "nl_static_profile" in df.columns:
            return {str(uid): text for uid, text in zip(df["userid"], df["nl_static_profile"])}
        # Genre-weight parquet (userid | Action | Adventure | ...)
        genre_cols = [g for g in _GENRE_NAMES if g in df.columns]
        genre_matrix = df[genre_cols].to_numpy(dtype=np.float32)
        profiles = {}
        for i, uid in enumerate(df["userid"]):
            profiles[str(int(uid))] = vector_to_text(genre_matrix[i], genre_cols)
        return profiles
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("{"):
        return json.loads(content)
    profiles = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        uid = obj.get("user_id")
        text = obj.get("nl_static_profile") or obj.get("profile")
        if uid is not None and text:
            profiles[uid] = text
    return profiles


def _resolve_dataset(args) -> str:
    if args.dataset != "auto":
        return args.dataset
    path = (args.data_path or "").lower()
    if "book" in path:
        return "amazon_book"
    if "beauty" in path:
        return "amazon_beauty"
    return "ml1m"


async def main(args):
    global _GENRE_NAMES
    config = build_config(args.config_file)
    fix_seeds(config.seed)

    dataset = _resolve_dataset(args)
    is_string_ids = (dataset == "amazon_beauty")
    if dataset == "amazon_book":
        _GENRE_NAMES = AMAZON_BOOK_GENRES
        print(f"Dataset: amazon_book ({len(_GENRE_NAMES)} genre categories)")
    elif dataset == "amazon_beauty":
        _GENRE_NAMES = []  # dynamic; pass --genre-list genre_names_form.json or rely on nl_static_profile
        print(f"Dataset: amazon_beauty (pass --genre-list to set genre names, or use --profile-path for NL profiles)")
    else:
        _GENRE_NAMES = ML1M_GENRES
        print(f"Dataset: ml1m ({len(_GENRE_NAMES)} genres)")

    if args.genre_list:
        with open(args.genre_list) as f:
            data = json.load(f)
        _GENRE_NAMES = data if isinstance(data, list) else sorted(data, key=data.get)
        print(f"Genre list overridden from {args.genre_list}: {len(_GENRE_NAMES)} genres")

    prompts = load_prompts(args.prompt_yaml)
    # if 'score' in args.data_path:
    #     discrimination_template = PromptTemplate(prompts["score_discrimination"])
    #     print("score_discrimination intent prompt]")
    # else:
    #     discrimination_template = PromptTemplate(prompts["threshold_discrimination"])
    #     print("threshold_discrimination intent prompt]")
        
    discrimination_template = PromptTemplate(prompts["intent_discrimination"])
    profile_only_template = PromptTemplate(prompts["profile_only_discrimination"])
    # set up intent predictor first — needed to decide whether to embed profiles
    print(f"Task: 1:{1+args.neg_num}")
    use_intent = not args.no_intent
    intent_predictor = None
    if not use_intent:
        mode_tag = "profile_only"
        print("Profile-only mode: intent and bias omitted from prompt.")
    elif args.random_intent:
        mode_tag = "random_intent"
        print("Random-intent mode: intent randomly sampled per sample (baseline).")
    elif args.ckpt_dir:
        intent_predictor = IntentPredictor(args.ckpt_dir)
        base_tag = intent_predictor.model_type
        mode_tag = f"{base_tag}_conf{args.confidence_threshold}" if args.confidence_threshold is not None else base_tag
        print(
            f"{intent_predictor.model_type} predictor loaded from {args.ckpt_dir} "
            f"(val_acc={intent_predictor.val_acc:.4f}, epoch={intent_predictor.epoch}, "
            f"input_repr={intent_predictor.input_repr}, decay_factor={intent_predictor.decay_factor})"
        )
        if intent_predictor.input_repr in ("auto", "concat", "embedding") and args.emb_dir is None:
            candidate = Path(args.data_path) / "embeddings"
            if (candidate / "items.npy").exists():
                args.emb_dir = str(candidate)
                print(f"Auto-detected emb-dir: {args.emb_dir}")
            elif intent_predictor.emb_dir and Path(intent_predictor.emb_dir).exists():
                args.emb_dir = intent_predictor.emb_dir
                print(f"Using emb-dir from checkpoint config: {args.emb_dir}")
            else:
                print(
                    f"WARNING: model input_repr={intent_predictor.input_repr!r} requires embeddings "
                    f"but no embeddings/ found under {args.data_path} and --emb-dir not set. "
                    f"Predictions will use wrong-dimension inputs."
                )
    else:
        base_tag = "ground_truth"
        mode_tag = f"ground_truth_conf{args.confidence_threshold}" if args.confidence_threshold is not None else base_tag
        print("Using ground-truth intent labels (no --ckpt-dir provided).")

    if args.confidence_threshold is not None:
        print(f"Confidence-gated mode: threshold={args.confidence_threshold} "
              f"(low-confidence samples → profile_only prompt).")
        
    with open(args.user_ids) as f:
        if is_string_ids:
            user_id_list = [str(line.strip()) for line in f if line.strip()]
        else:
            user_id_list = [int(line.strip()) for line in f if line.strip()]
    user_id_set = set(user_id_list)
    print(f"Loaded {len(user_id_list)} user IDs from {args.user_ids}")

    print(f"Load sim_eval data from {args.data_path}")
    if args.profile_path:
        profiles = load_profiles(args.profile_path)
        print(f"Loaded external profiles from {args.profile_path}: {len(profiles)} users")
        # follow txt-file order: pick profiles in the order they appear in user_id_list
        profile_items = [
            (str(uid), profiles[str(uid)]) for uid in user_id_list if str(uid) in profiles
        ][args.start_idx:args.end_idx]
        print(f"Selected users {args.start_idx}:{args.end_idx} → {len(profile_items)} users")
        matched_rows = load_sim_eval_for_users(args.data_path+"/sim_eval.npz", [uid for uid, _ in profile_items], is_string_ids=is_string_ids)
        user_index = {}
        for row in matched_rows:
            user_index.setdefault(str(row["user_id"]), []).append(row)
        eval_samples = []
        for uid, profile_text in profile_items:
            rows_for_user = user_index.get(str(uid), [])
            if not rows_for_user:
                print(f"  Warning: user_id={uid} not found in sim_eval — skipping")
                continue
            rows_for_user = sorted(rows_for_user, key=lambda r: r.get("time_index") or 0)
            rows_for_user = rows_for_user[:args.rounds]
            for row in rows_for_user:
                sample = dict(row)
                sample["nl_static_profile"] = profile_text
                eval_samples.append(sample)
        
        print(f"Selected users: {[uid for uid, _ in profile_items]}")
        print(f"Profile-driven: {len(profile_items)} users → {len(eval_samples)} sim_eval samples")
        if args.emb_dir is not None and eval_samples:
            # Use decay_factor from the checkpoint when available; fall back to CLI arg.
            eff_decay = (intent_predictor.decay_factor
                         if intent_predictor is not None else args.decay_factor)
            apply_emb_inputs(eval_samples, args.emb_dir, decay_factor=eff_decay)
    else:
        sim_eval_user_set = set(get_sorted_users(args.data_path+"/sim_eval.npz", is_string_ids=is_string_ids))
        all_users = [u for u in user_id_list if u in sim_eval_user_set]
        print(f"Filtered to {len(all_users)} users from --user-ids")
        selected_users = all_users[args.start_idx:args.end_idx]
        print(f"Selected users: {selected_users}")
        print(f"Selected users {args.start_idx}:{args.end_idx} → {len(selected_users)} / {len(all_users)} users")
        eval_samples = load_sim_eval_for_users(args.data_path+"/sim_eval.npz", selected_users, is_string_ids=is_string_ids)
        from collections import defaultdict
        user_buckets = defaultdict(list)
        for row in eval_samples:
            user_buckets[row["user_id"]].append(row)
        eval_samples = []
        for rows_for_user in user_buckets.values():
            rows_for_user.sort(key=lambda r: r.get("time_index") or 0)
            eval_samples.extend(rows_for_user[:args.rounds])
        print(f"Loaded {len(eval_samples)} sim_eval rows")
        if args.emb_dir is not None and eval_samples:
            # Use decay_factor from the checkpoint when available; fall back to CLI arg.
            eff_decay = (intent_predictor.decay_factor
                         if intent_predictor is not None else args.decay_factor)
            apply_emb_inputs(eval_samples, args.emb_dir, decay_factor=eff_decay)
    if not eval_samples:
        return

    if intent_predictor is not None:
        _s0, _p0 = _build_predictor_inputs(eval_samples[0], intent_predictor)
        _sd = np.asarray(_s0).shape[-1]
        _pd = np.asarray(_p0).shape[-1]
        if _sd != intent_predictor.state_dim or _pd != intent_predictor.profile_dim:
            print(
                f"WARNING: dimension mismatch — predictor expects "
                f"state_dim={intent_predictor.state_dim}, profile_dim={intent_predictor.profile_dim}; "
                f"data provides state_dim={_sd}, profile_dim={_pd}. "
                f"Check that --emb-dir and data mode match the training configuration."
            )
        else:
            print(f"Predictor input dims OK: state_dim={_sd}, profile_dim={_pd}")

    _DATALOADER_CLS = {"ml1m": ML_1M, "amazon_book": Amazon_Book, "amazon_beauty": Amazon_Beauty}
    dataloader = _DATALOADER_CLS[dataset](config)
    llm = load_llm(config)

    semaphore = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(eval_samples), desc="Evaluating")

    async def process_one(sample):
        if sample.get("time_index") is None:
            pbar.update(1)
            return None
        try:
            # Deterministic per-sample seed so negative item sampling is reproducible
            # across runs and independent of intent-label configuration.
            uid_int = sample["user_id"] if isinstance(sample["user_id"], int) else abs(hash(str(sample["user_id"]))) % (2**31)
            np.random.seed((uid_int * 10007 + int(sample.get("time_index") or 0)) % (2**32))
            positive_items, negative_items = load_candidate_items(
                dataloader, sample["user_id"], sample.get("time_index"), args
            )
            if args.recent_items:
                sample["recent_items"] = load_recent_items(
                    dataloader, sample["user_id"], sample.get("time_index")
                )
            async with semaphore:
                t0 = time.perf_counter()
                score, score_std, usage, interactions, predicted_intent, prompt_mode, confidence = \
                    await discrimination_evaluation(
                        llm,
                        sample,
                        positive_items,
                        negative_items,
                        shuffle=args.shuffle,
                        repetition=args.repetition,
                        intent_predictor=intent_predictor,
                        use_intent=use_intent,
                        discrimination_template=discrimination_template,
                        profile_only_template=profile_only_template,
                        confidence_threshold=args.confidence_threshold,
                        random_intent=args.random_intent,
                    )
                round_time = time.perf_counter() - t0
        except Exception as exc:
            print(f"Error processing user {sample.get('user_id')} t={sample.get('time_index')}: {exc}")
            pbar.update(1)
            return None
        pbar.update(1)
        if score == "Err":
            return None
        round_tokens = sum(getattr(u, "total_tokens", 0) for u in usage)
        return {
            "user_id": sample.get("user_id"),
            "sample_idx": sample.get("sample_idx"),
            "time_index": sample.get("time_index"),
            "gt_intent_label": sample.get("intent_label"),
            "gt_intent_name": sample.get("intent_name"),
            "predicted_intent_name": predicted_intent,
            "prompt_mode": prompt_mode,
            "confidence": confidence,
            "score": score,
            "score_std": score_std,
            "repetition": interactions,
            "round_time_s": round_time,
            "round_tokens": round_tokens,
            "_usage": usage,
        }

    raw_results = await asyncio.gather(*[process_one(s) for s in eval_samples])
    pbar.close()

    records = [r for r in raw_results if r is not None]
    if not records:
        print("No valid scores.")
        return

    scores           = [r["score"]        for r in records]
    stds             = [r["score_std"]    for r in records]
    round_times      = [r["round_time_s"] for r in records]
    round_token_counts = [r["round_tokens"] for r in records]
    token_usage      = [u for r in records for u in r.pop("_usage")]

    total_tokens = sum(getattr(u, "total_tokens", 0) for u in token_usage)
    avg_time   = np.mean(round_times)
    avg_tokens = np.mean(round_token_counts)
    print(
        f"[{mode_tag}] concurrency[{args.concurrency}] repetition[{args.repetition}] "
        f"sim_eval[{args.start_idx}:{args.end_idx}] "
        f"{len(scores)} samples: {np.mean(scores):.4f}(±{np.mean(stds):.4f})"
    )
    print(f"Avg time per round: {avg_time:.2f}s  Total time: {sum(round_times):.1f}s")
    print(f"Avg tokens per round: {avg_tokens:.1f}  Total Token Usage: {total_tokens}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if args.output_path is None:
        output_dir = Path("output/discrimination")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = str(output_dir / f"{mode_tag}_{args.start_idx}_{args.end_idx}_{timestamp}")
    else:
        stem = args.output_path + f"{mode_tag}_{args.start_idx}_{args.end_idx}_{timestamp}"

    output_path = stem + ".jsonl"
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(records)} interaction records to {output_path}")

    log = {
        "timestamp": timestamp,
        "mode": mode_tag,
        "args": vars(args),
        "intent_predictor": (
            {
                "model_type": intent_predictor.model_type,
                "val_acc": intent_predictor.val_acc,
                "val_macro_f1": getattr(intent_predictor, "val_macro_f1", None),
                "epoch": intent_predictor.epoch,
                "ckpt_dir": str(args.ckpt_dir),
            }
            if intent_predictor is not None else None
        ),
        "results": {
            "n_samples": len(scores),
            "mean_score": float(np.mean(scores)),
            "mean_std": float(np.mean(stds)),
            "total_time_s": float(sum(round_times)),
            "avg_time_per_round_s": float(avg_time),
            "total_tokens": int(total_tokens),
            "avg_tokens_per_round": float(avg_tokens),
        },
        "output_path": output_path,
    }
    log_path = stem + ".log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Saved experiment log to {log_path}")


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main(parse_arguments()))
