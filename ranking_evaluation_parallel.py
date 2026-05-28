import argparse
import asyncio
import json
import pickle
import random
import re
import time
from collections import defaultdict
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
    parser = argparse.ArgumentParser(description="LLM ranking evaluation (NDCG@K / HitRate@K).")
    parser.add_argument("-s",   "--start_idx",   type=int, default=0)
    parser.add_argument("-e",   "--end_idx",     type=int, default=50)
    parser.add_argument("-r",   "--rounds",      type=int, default=3,
                        help="Max sim_eval rounds (time steps) per user.")
    parser.add_argument("-rep", "--repetition",  type=int, default=3)
    parser.add_argument("-c",   "--config_file", type=str, default="config/config_general.yaml")
    parser.add_argument("-nn",  "--neg_num",     type=int, default=9)
    parser.add_argument("-tn",  "--test_num",    type=int, default=1)
    parser.add_argument("-ts",  "--test_sampling", type=str, default="sequence")
    parser.add_argument("-sf",  "--shuffle",     action="store_true",
                        help="Re-shuffle the candidate list each repetition.")
    parser.add_argument("--skip_num", type=int, default=25,
                        help="Number of used interactions (legacy fallback path).")
    parser.add_argument("--ndcg-k",  type=int, default=5,  help="K for NDCG@K metric.")
    parser.add_argument("--hit-k",   type=int, default=3,  help="K for HitRate@K metric.")
    parser.add_argument("-d","--data-path", type=str,
                        default="output/state_transition/ml_1m/robust_hybrid/")
    parser.add_argument("-ckpt", "--ckpt-dir", type=str, default=None,
                        help="Intent model checkpoint dir. Omit to use ground-truth intent.")
    parser.add_argument("--no-intent",    action="store_true",
                        help="Profile-only baseline: omit intent from the prompt entirely.")
    parser.add_argument("--gt-intent",    action="store_true",
                        help="Explicitly use ground-truth intent (ignores --ckpt-dir).")
    parser.add_argument("--random-intent", action="store_true",
                        help="Baseline: randomly assign one of the three intents.")
    parser.add_argument("--recent-items", action="store_true",
                        help="Show last 5 interacted items as recent behaviour instead of genre vector.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Directory to save per-sample JSONL records.")
    parser.add_argument("-p","--profile-path", type=str, default=None,
                        help="JSON/JSONL/pkl/parquet mapping user_id → profile text.")
    parser.add_argument("--emb-dir", type=str, default=None,
                        help="Path to embeddings/ directory (items.npy, profiles.npy). "
                             "When set, reconstructs embedding-space inputs for the intent model.")
    parser.add_argument("--decay-factor", type=float, default=1.0,
                        help="Exponential decay for window item embeddings (1.0 = uniform).")
    parser.add_argument("--embed-model", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--prompt-yaml", type=str, default="config/prompt.yaml")
    parser.add_argument("--confidence", type=float, default=None,
                        help="If set, fall back to profile_only_ranking when confidence < threshold.")
    parser.add_argument("--user-ids", type=str, default="baseline/amazon_book_sub_userids.txt",
                        help="Path to a txt file with one user ID per line.")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max number of samples evaluated concurrently.")
    parser.add_argument("--dataset", type=str, default="auto",
                        choices=["auto", "ml1m", "amazon_book", "amazon_beauty"],
                        help="Dataset type. 'auto' infers from --data-path.")
    parser.add_argument("--genre-list", type=str, default=None,
                        help="Path to JSON file (list or {name: idx} dict) with ordered genre names. "
                             "Overrides the built-in genre list. Use with datasets built from "
                             "genres_all (unfiltered) to match the correct genre dimensions. "
                             "Generated by the enrich_profiles notebook cell as genre_names_{mode}.json.")
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sim_eval_for_users(sim_eval_path, target_user_ids, is_string_ids=False):
    sim_eval = np.load(sim_eval_path)

    meta_pkl_path = Path(sim_eval_path).with_name(Path(sim_eval_path).stem + "_meta.pkl")
    meta_list = None
    if meta_pkl_path.exists():
        with open(meta_pkl_path, "rb") as _f:
            meta_list = pickle.load(_f)

    if "user_id" not in sim_eval.files:
        raise KeyError("sim_eval npz has no user_id array. Re-generate the data.")

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
            print(f"  No matching users. Sample user_ids in meta: {sample_ids}")
            return []
    else:
        target_int_ids = np.array([int(uid) for uid in target_user_ids], dtype=np.int64)
        indices        = np.where(np.isin(user_ids_arr, target_int_ids))[0]
        if len(indices) == 0:
            sample_ids = sorted(set(user_ids_arr[user_ids_arr != -1].tolist()))[:10]
            print(f"  No matching users. Sample user_ids in sim_eval: {sample_ids}")
            return []

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
            row["window_item_ids"] = window_item_ids[i]
        if profile_emb_idx is not None:
            row["profile_emb_idx"] = int(profile_emb_idx[i])
        if recent_vectors is not None:
            row["recent_vector"] = recent_vectors[i]      # (T, num_genres)
        if meta_list is not None and npz_idx < len(meta_list):
            row["nl_static_profile"] = meta_list[npz_idx].get("nl_static_profile")
        rows.append(row)

    return rows


def get_sorted_users(sim_eval_path, is_string_ids=False):
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


def load_candidate_items(dataloader, userid, time_index, args):
    """Return ([positive_item], [negative_items]) for this sample."""
    if time_index is not None:
        _, negative_items = dataloader.get_items_for_profile(
            userid=userid, pos_num=0, neg_num=args.neg_num, test_num=0,
        )
        interacted_df = dataloader.get_interacted_items(userid=userid, mergeitem=True)
        interacted_records = interacted_df.reset_index(drop=True).to_dict(orient="records")

        if time_index >= len(interacted_records):
            raise IndexError(
                f"time_index={time_index} out of range for user {userid} "
                f"({len(interacted_records)} total interactions)"
            )
        target_item = interacted_records[time_index]
        if negative_items:
            target_item = {k: target_item[k] for k in negative_items[0] if k in target_item}
        return [target_item], negative_items
    else:
        print("No time index — using legacy fallback")
        _, negative_items, positive_items = dataloader.get_items_for_profile(
            userid=userid,
            pos_num=args.skip_num,
            neg_num=args.neg_num,
            test_sampling=args.test_sampling,
            test_num=args.test_num,
        )
        positive_items = [
            {k: item[k] for k in negative_items[0] if k in item}
            for item in positive_items
        ]
        return positive_items, negative_items


def load_recent_items(dataloader, userid, time_index, n=5):
    """Return the n items interacted with immediately before time_index."""
    interacted_df = dataloader.get_interacted_items(userid=userid, mergeitem=True)
    interacted_records = interacted_df.reset_index(drop=True).to_dict(orient="records")
    start = max(0, time_index - n)
    return interacted_records[start:time_index]


def apply_emb_inputs(eval_samples, emb_dir, decay_factor=1.0):
    """Reconstruct embedding-space inputs from pre-computed index tables.

    Stores results under separate keys (emb_*) so genre arrays remain intact
    for prompt rendering (vector_to_text, drift label).
    """
    emb_dir      = Path(emb_dir)
    item_embs    = np.load(emb_dir / "items.npy").astype(np.float32)
    profile_embs = np.load(emb_dir / "profiles.npy").astype(np.float32)

    n_applied = 0
    for sample in eval_samples:
        p_idx   = sample.get("profile_emb_idx")
        win_ids = sample.get("window_item_ids")
        if p_idx is None or win_ids is None:
            continue

        sample["emb_static_profile"] = profile_embs[int(p_idx)]

        win_ids = np.asarray(win_ids, dtype=np.int32)
        W = win_ids.shape[1]
        if decay_factor == 1.0:
            state_emb = item_embs[win_ids].mean(axis=1)
        else:
            weights = np.array(
                [decay_factor ** (W - 1 - i) for i in range(W)], dtype=np.float32
            )
            weights /= weights.sum()
            state_emb = (item_embs[win_ids] * weights[None, :, None]).sum(axis=1)
        norms = np.linalg.norm(state_emb, axis=-1, keepdims=True)
        sample["emb_state_sequence"] = state_emb / np.maximum(norms, 1e-8)
        n_applied += 1

    D = profile_embs.shape[1]
    print(f"Applied embedding inputs: embed_dim={D}, {n_applied}/{len(eval_samples)} samples from {emb_dir}")


# ── Profile / context helpers ─────────────────────────────────────────────────

def vector_to_text(vector, genre_names=None):
    if genre_names is not None:
        entries = sorted(
            [(genre_names[i], v) for i, v in enumerate(vector) if v > 0],
            key=lambda x: x[1], reverse=True,
        )
        return ", ".join(f"{name} ({v:.1%})" for name, v in entries) or "none"
    return ", ".join(f"{i}={v:.3f}" for i, v in enumerate(vector) if v > 0)


def _format_recent_items(items):
    lines = []
    for i, item in enumerate(items, 1):
        title  = next((item[k] for k in ("title", "book_title", "product_title") if k in item), None)
        genre  = item.get("genres") or item.get("genre", "")
        rating = item.get("user_rating", "")
        parts  = [title] if title else []
        if genre:
            parts.append(f"[{genre}]")
        if rating:
            parts.append(f"rating: {rating}")
        lines.append(f"{i}. {' '.join(str(p) for p in parts)}")
    return "\n".join(lines)

def _get_drift_label(drift_val):
    if drift_val > 0.6:
        drift_label = "high"
    elif drift_val > 0.3:
        drift_label = "moderate"
    else:
        drift_label = "low"
    return drift_label

def _sample_base_context(sample):
    latest_state = sample["state_sequence"][-1]
    drift_val = float(latest_state[-1])
    drift_label = _get_drift_label(drift_val)
    recent_drift = f"Drift from long-term profile: {drift_val:.3f} ({drift_label})"

    static_profile_text = (
        sample.get("nl_static_profile")
        or vector_to_text(sample["static_profile"], _GENRE_NAMES)
    )

    recent_items = sample.get("recent_items")
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
        recent_history = f"Recent genre preferences: {vector_to_text(latest_state[:len(_GENRE_NAMES)], _GENRE_NAMES)}\n"

    return {
        "static_profile_text": static_profile_text,
        "recent_history":      recent_history,
        "recent_drift":        recent_drift,
    }


def _explore_avoid_hint(sample):
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
        recent = latest_state[:-1][:n]
        top_recent = [_GENRE_NAMES[i] for i in np.argsort(recent)[::-1] if recent[i] > 0][:2]
        avoid = list(dict.fromkeys(top_static + top_recent))[:3]

    if avoid:
        return f" Genres to avoid (dominant in both profiles): {', '.join(avoid)}."
    return ""


def sim_eval_to_prompt_context(sample, intent_name, bias_vector):
    scores    = sample.get("scores", {})
    s_profile = scores.get("profile_similarity")
    s_recent  = scores.get("recent_similarity")
    drift     = scores.get("drift")

    intent_desc = INTENT_DESCRIPTIONS.get(intent_name, "")
    if intent_name == "explore":
        intent_desc += _explore_avoid_hint(sample)

    return {
        **_sample_base_context(sample),
        "predicted_intent":   intent_name,
        "intent_description": intent_desc,
        "profile_similarity": f"{s_profile:.2f}" if s_profile is not None else "N/A",
        "recent_similarity":  f"{s_recent:.2f}"  if s_recent  is not None else "N/A",
        "drift":              f"{drift:.2f}"      if drift     is not None else "N/A",
        "profile_weight":     f"{bias_vector['profile']:.2f}",
        "recent_weight":      f"{bias_vector['recent']:.2f}",
        "novelty_weight":     f"{bias_vector['novelty']:.2f}",
    }


def _build_predictor_inputs(sim_sample, intent_predictor):
    """Return (state_seq, profile) arrays matching the predictor's input_repr.

    concat:         cat(L2-norm(genre), drift, emb) — mirrors training _get_inputs.
    auto/embedding: pure embedding when available, genre fallback otherwise.
    genre:          always genre histograms.
    """
    has_emb = "emb_state_sequence" in sim_sample
    if has_emb and getattr(intent_predictor, "input_repr", "auto") == "concat":
        state_arr  = np.asarray(sim_sample["state_sequence"], dtype=np.float32)
        genre_s    = state_arr[..., :-1]
        norms_s    = np.linalg.norm(genre_s, axis=-1, keepdims=True)
        genre_s_l2 = genre_s / np.maximum(norms_s, 1e-8)
        drift_s    = state_arr[..., -1:]
        state_seq  = np.concatenate([genre_s_l2, drift_s,
                                     sim_sample["emb_state_sequence"]], axis=-1)
        profile_arr = np.asarray(sim_sample["static_profile"], dtype=np.float32)
        norm_p      = np.linalg.norm(profile_arr)
        profile_l2  = profile_arr / max(float(norm_p), 1e-8)
        profile     = np.concatenate([profile_l2,
                                      sim_sample["emb_static_profile"]], axis=-1)
    else:
        state_seq = sim_sample.get("emb_state_sequence", sim_sample["state_sequence"])
        profile   = sim_sample.get("emb_static_profile", sim_sample["static_profile"])
    return state_seq, profile


def _resolve_intent(sim_sample, intent_predictor):
    """Return (intent_name, bias_vector, confidence).

    Routes through _build_predictor_inputs so concat/embedding models receive
    the correct input representation.  Falls back to ground-truth when predictor is None.
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
    confidence = scores.get("confidence")
    return intent_name, bias_vector, confidence


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_indexes, positive_indexes, k):
    """NDCG@k — IDCG = 1.0 (single relevant item)."""
    pos_set = set(positive_indexes)
    for rank, idx in enumerate(ranked_indexes[:k], start=1):
        if idx in pos_set:
            return 1.0 / np.log2(rank + 1)
    return 0.0


def hit_at_k(ranked_indexes, positive_indexes, k):
    pos_set = set(positive_indexes)
    return int(any(idx in pos_set for idx in ranked_indexes[:k]))


# ── Response parsing ──────────────────────────────────────────────────────────

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


def validate_indexes(indexes, max_index=None):
    if max_index is None:
        return indexes
    return [idx for idx in indexes if 1 <= idx <= max_index]


def parse_ranked_indexes(response_text, max_index=None):
    parsed = parse_json_object(response_text)
    if parsed and "ranked_item_indexes" in parsed:
        value = parsed["ranked_item_indexes"]
        indexes = value if isinstance(value, list) else [value]
        return validate_indexes([int(i) for i in indexes], max_index)
    return validate_indexes([int(x) for x in re.findall(r"\d+", response_text)], max_index)


def general_parser(response):
    return response.split("</think>")[-1].strip() if response else ""


# ── Candidate formatting ──────────────────────────────────────────────────────

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


def _extract_item_id(item):
    for key in ("title", "movie_title", "book_title", "product_title", "name"):
        if key in item and item[key]:
            return item[key]
    return next((v for v in item.values() if v is not None), None)


# ── Single repetition ─────────────────────────────────────────────────────────

async def _run_single_rep(
    llm,
    candidate_list,
    positive_item_ids,
    base_context,
    prompt_template,
    num_items,
    ndcg_k,
    hit_k,
    shuffle,
):
    """Issue one LLM ranking call. Returns (ndcg, hit, usage, interaction) or None."""
    if shuffle:
        candidate_list = candidate_list[:]
        random.shuffle(candidate_list)

    positive_indexes = [
        i + 1
        for i, item in enumerate(candidate_list)
        if id(item) in positive_item_ids
    ]

    messages = prompt_template.format_messages(
        **base_context,
        num_items=num_items,
        candidate_items="\n".join(recommendation_list_to_prompt(candidate_list)),
    )
    # print(messages)
    # return
    try:
        response = await llm.achat(messages=messages)
    except Exception as exc:
        print(f"LLM error: {exc}")
        return None

    raw_response = getattr(response, "raw", None)
    usage = raw_response.usage if hasattr(raw_response, "usage") else None

    ranked_indexes = parse_ranked_indexes(
        general_parser(response.message.content),
        max_index=num_items,
    )
    if not ranked_indexes:
        return None

    return (
        ndcg_at_k(ranked_indexes, positive_indexes, ndcg_k),
        hit_at_k(ranked_indexes, positive_indexes, hit_k),
        usage,
        {
            "positive_items": [_extract_item_id(candidate_list[i - 1]) for i in positive_indexes],
            "ranked_items": [
                _extract_item_id(candidate_list[i - 1])
                for i in ranked_indexes if 1 <= i <= num_items
            ],
        },
    )


# ── Core ranking call ─────────────────────────────────────────────────────────

async def ranking_evaluation(
    llm,
    sim_sample,
    positive_items,
    negative_items,
    shuffle=False,
    repetition=3,
    intent_predictor=None,
    use_intent=True,
    ranking_template=None,
    profile_only_template=None,
    confidence_threshold=None,
    random_intent=False,
    ndcg_k=5,
    hit_k=3,
):
    used_intent = None
    used_confidence = None
    prompt_mode = "profile_only"   # default; always overwritten below

    if not use_intent:
        prompt_template = profile_only_template
        prompt_mode = "profile_only"
        base_context = _sample_base_context(sim_sample)
        base_context["recent_drift"] = ""
    elif random_intent:
        prompt_template = ranking_template
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
        drift_val = np.random.rand()
        drift_label = _get_drift_label(drift_val)
        base_context["recent_drift"] = f"Drift from long-term profile: {drift_val} ({drift_label})"
    else:
        intent_name, bias_vector, confidence = _resolve_intent(sim_sample, intent_predictor)
        used_intent = intent_name
        used_confidence = confidence

        if confidence_threshold is not None and (confidence is None or confidence < confidence_threshold):
            prompt_template = profile_only_template
            prompt_mode = "profile_only"
            base_context = _sample_base_context(sim_sample)
        else:
            prompt_template = ranking_template
            prompt_mode = "intent"
            base_context = sim_eval_to_prompt_context(sim_sample, intent_name, bias_vector)

    candidate_list = positive_items + negative_items
    random.shuffle(candidate_list)
    num_items = len(candidate_list)
    positive_item_ids = {id(item) for item in positive_items}

    rep_results = await asyncio.gather(*[
        _run_single_rep(
            llm, candidate_list, positive_item_ids,
            base_context, prompt_template, num_items, ndcg_k, hit_k, shuffle,
        )
        for _ in range(repetition)
    ])
    ndcg_scores  = []
    hit_scores   = []
    token_usage  = []
    interactions = []
    for r in rep_results:
        if r is None:
            continue
        ndcg, hit, usage, interaction = r
        ndcg_scores.append(ndcg)
        hit_scores.append(hit)
        if usage is not None:
            token_usage.append(usage)
        interactions.append(interaction)

    if not ndcg_scores:
        return "Err", 0.0, 0.0, 0.0, token_usage, [], used_intent, prompt_mode, used_confidence

    return (
        round(float(np.mean(ndcg_scores)), 4),
        round(float(np.std(ndcg_scores)),  4),
        round(float(np.mean(hit_scores)),  4),
        round(float(np.std(hit_scores)),   4),
        token_usage,
        interactions,
        used_intent,
        prompt_mode,
        used_confidence,
    )


# ── Setup helpers ─────────────────────────────────────────────────────────────

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


def load_profiles(profile_path):
    """Load user_id → profile_text from JSON, JSONL, pkl, or parquet.

    JSON: {"user_id": "profile text", ...}
    JSONL: one {"user_id": ..., "nl_static_profile": ...} object per line
    pkl: dict {user_id: profile_text}
    parquet (enriched): userid | nl_static_profile — returned as-is.
    parquet (genre-weight): userid | <genre> | ... — converted via vector_to_text.
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
            profiles[str(uid)] = vector_to_text(genre_matrix[i], genre_cols)
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
        uid  = obj.get("user_id")
        text = obj.get("nl_static_profile") or obj.get("profile")
        if uid is not None and text:
            profiles[uid] = text
    return profiles


def embed_profiles(eval_samples, model_name):
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": "cpu"}
    )
    cache = {}
    for sample in eval_samples:
        text = sample.get("nl_static_profile")
        if not text:
            continue
        if text not in cache:
            cache[text] = np.array(embedding_model.embed_query(text), dtype=np.float32)
        sample["static_profile"] = cache[text]
    print(f"Embedded {len(cache)} unique profiles with {model_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

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
    ranking_template      = PromptTemplate(prompts["intent_ranking"])
    profile_only_template = PromptTemplate(prompts["profile_only_ranking"])

    print(f"Task: 1:{1 + args.neg_num}")
    use_intent = not args.no_intent
    intent_predictor = None

    if not use_intent:
        mode_tag = "profile_only"
        print("Profile-only mode: intent omitted from prompt.")
    elif args.random_intent:
        mode_tag = "random_intent"
        print("Random-intent mode: intent randomly sampled per sample (baseline).")
    elif args.gt_intent or args.ckpt_dir is None:
        base_tag = "ground_truth"
        mode_tag = f"{base_tag}_conf{args.confidence}" if args.confidence is not None else base_tag
        print("Using ground-truth intent labels.")
    else:
        intent_predictor = IntentPredictor(args.ckpt_dir)
        base_tag = intent_predictor.model_type
        mode_tag = f"{base_tag}_conf{args.confidence}" if args.confidence is not None else base_tag
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

    if args.confidence is not None:
        print(f"Confidence-gated mode: threshold={args.confidence} "
              f"(low-confidence samples → profile_only prompt).")

    with open(args.user_ids) as f:
        if is_string_ids:
            user_id_list = [str(line.strip()) for line in f if line.strip()]
        else:
            user_id_list = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(user_id_list)} user IDs from {args.user_ids}")

    print(f"Load sim_eval data from {args.data_path}")
    if args.profile_path:
        profiles = load_profiles(args.profile_path)
        print(f"Loaded external profiles from {args.profile_path}: {len(profiles)} users")
        profile_items = [
            (str(uid), profiles[str(uid)]) for uid in user_id_list if str(uid) in profiles
        ][args.start_idx:args.end_idx]
        print(f"Selected users {args.start_idx}:{args.end_idx} → {len(profile_items)} users")
        matched_rows = load_sim_eval_for_users(
            args.data_path + "/sim_eval.npz", [uid for uid, _ in profile_items], is_string_ids=is_string_ids
        )
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
            for row in rows_for_user[:args.rounds]:
                sample = dict(row)
                sample["nl_static_profile"] = profile_text
                eval_samples.append(sample)
        print(f"Selected users: {[uid for uid, _ in profile_items]}")
        print(f"Profile-driven: {len(profile_items)} users → {len(eval_samples)} sim_eval samples")
        if intent_predictor is not None and eval_samples and args.emb_dir is None:
            data_profile_dim = np.asarray(eval_samples[0]["static_profile"]).shape[-1]
            if data_profile_dim != intent_predictor.profile_dim:
                embed_profiles(eval_samples, args.embed_model)
        if args.emb_dir is not None and eval_samples:
            eff_decay = (intent_predictor.decay_factor
                         if intent_predictor is not None else args.decay_factor)
            apply_emb_inputs(eval_samples, args.emb_dir, decay_factor=eff_decay)
    else:
        sim_eval_user_set = set(get_sorted_users(args.data_path + "/sim_eval.npz", is_string_ids=is_string_ids))
        all_users = [u for u in user_id_list if u in sim_eval_user_set]
        print(f"Filtered to {len(all_users)} users from --user-ids")
        selected_users = all_users[args.start_idx:args.end_idx]
        print(f"Selected users {args.start_idx}:{args.end_idx} → {len(selected_users)} users")
        eval_samples = load_sim_eval_for_users(
            args.data_path + "/sim_eval.npz", selected_users, is_string_ids=is_string_ids
        )
        user_buckets = defaultdict(list)
        for row in eval_samples:
            user_buckets[row["user_id"]].append(row)
        eval_samples = []
        for rows in user_buckets.values():
            rows.sort(key=lambda r: r.get("time_index") or 0)
            eval_samples.extend(rows[:args.rounds])
        print(f"Loaded {len(eval_samples)} sim_eval rows")
        if args.emb_dir is not None and eval_samples:
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
            print(f"### Load intent_predictor Successfully ### Predictor input dims OK: state_dim={_sd}, profile_dim={_pd}")

    _DATALOADER_CLS = {"ml1m": ML_1M, "amazon_book": Amazon_Book, "amazon_beauty": Amazon_Beauty}
    dataloader = _DATALOADER_CLS[dataset](config)
    llm = load_llm(config)

    semaphore = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(eval_samples), desc="Ranking eval")

    async def process_one(sample):
        if sample.get("time_index") is None:
            pbar.update(1)
            return None
        try:
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
                result = await ranking_evaluation(
                    llm,
                    sample,
                    positive_items,
                    negative_items,
                    shuffle=args.shuffle,
                    repetition=args.repetition,
                    intent_predictor=intent_predictor,
                    use_intent=use_intent,
                    ranking_template=ranking_template,
                    profile_only_template=profile_only_template,
                    confidence_threshold=args.confidence,
                    random_intent=args.random_intent,
                    ndcg_k=args.ndcg_k,
                    hit_k=args.hit_k,
                )
                round_time = time.perf_counter() - t0
        except Exception as exc:
            print(f"Error processing user {sample.get('user_id')} t={sample.get('time_index')}: {exc}")
            pbar.update(1)
            return None

        pbar.update(1)
        if result[0] == "Err":
            return None

        ndcg_val, ndcg_std, hit_val, hit_std, usage, interactions, predicted_intent, prompt_mode, confidence = result
        round_tokens = sum(getattr(u, "total_tokens", 0) for u in usage)
        return {
            "user_id":               sample.get("user_id"),
            "sample_idx":            sample.get("sample_idx"),
            "time_index":            sample.get("time_index"),
            "gt_intent_label":       sample.get("intent_label"),
            "gt_intent_name":        sample.get("intent_name"),
            "predicted_intent_name": predicted_intent,
            "prompt_mode":           prompt_mode,
            "confidence":            confidence,
            f"ndcg@{args.ndcg_k}":  ndcg_val,
            "ndcg_std":              ndcg_std,
            f"hit@{args.hit_k}":    hit_val,
            "hit_std":               hit_std,
            "repetition":            interactions,
            "round_time_s":          round_time,
            "round_tokens":          round_tokens,
            "_usage":                usage,
        }

    raw_results = await asyncio.gather(*[process_one(s) for s in eval_samples])
    pbar.close()

    records = [r for r in raw_results if r is not None]
    if not records:
        print("No valid scores.")
        return

    ndcg_list         = [r[f"ndcg@{args.ndcg_k}"] for r in records]
    ndcg_std_list     = [r["ndcg_std"]             for r in records]
    hit_list          = [r[f"hit@{args.hit_k}"]    for r in records]
    hit_std_list      = [r["hit_std"]               for r in records]
    round_times       = [r["round_time_s"]          for r in records]
    round_token_counts = [r["round_tokens"]         for r in records]
    token_usage       = [u for r in records for u in r.pop("_usage")]

    total_tokens = sum(getattr(u, "total_tokens", 0) for u in token_usage)
    avg_time     = np.mean(round_times)
    avg_tokens   = np.mean(round_token_counts)

    print(
        f"[{mode_tag}] concurrency[{args.concurrency}] rep={args.repetition} "
        f"sim_eval[{args.start_idx}:{args.end_idx}] {len(ndcg_list)} samples | "
        f"NDCG@{args.ndcg_k}={np.mean(ndcg_list):.4f}(±{np.mean(ndcg_std_list):.4f}) | "
        f"Hit@{args.hit_k}={np.mean(hit_list):.4f}(±{np.mean(hit_std_list):.4f})"
    )
    print(f"Avg time per round: {avg_time:.2f}s  Total time: {sum(round_times):.1f}s")
    print(f"Avg tokens per round: {avg_tokens:.1f}  Total Token Usage: {total_tokens}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_path) if args.output_path else Path("output/ranking")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(output_dir / f"{mode_tag}_{args.start_idx}_{args.end_idx}_{timestamp}")

    output_path = stem + ".jsonl"
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(records)} records to {output_path}")

    log = {
        "timestamp": timestamp,
        "mode": mode_tag,
        "args": vars(args),
        "intent_predictor": (
            {
                "model_type":   intent_predictor.model_type,
                "val_acc":      intent_predictor.val_acc,
                "val_macro_f1": getattr(intent_predictor, "val_macro_f1", None),
                "epoch":        intent_predictor.epoch,
                "ckpt_dir":     str(args.ckpt_dir),
            }
            if intent_predictor is not None else None
        ),
        "results": {
            "n_samples":                  len(ndcg_list),
            f"mean_ndcg@{args.ndcg_k}":  float(np.mean(ndcg_list)),
            f"mean_hit@{args.hit_k}":    float(np.mean(hit_list)),
            "total_time_s":               float(sum(round_times)),
            "avg_time_per_round_s":       float(avg_time),
            "total_tokens":               int(total_tokens),
            "avg_tokens_per_round":       float(avg_tokens),
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
