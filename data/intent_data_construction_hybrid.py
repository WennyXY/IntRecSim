import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data.profile_generation_ml_1m import (
    load_item_genres,
    get_static_profile,
    genre_vector,
    normalize,
    keep_top_k_genres,
)


def load_item_embs(emb_dir):
    """Load pre-computed item embeddings (float16 .npy + items_idx.json).

    Returns ``{item_id: float32_array}``.  Item IDs are cast to int when
    possible; kept as str for string-ID datasets (ASINs etc.).
    """
    emb_dir = Path(emb_dir)
    matrix = np.load(emb_dir / "items.npy").astype(np.float32)
    with (emb_dir / "items_idx.json").open() as f:
        id_strs = json.load(f)
    try:
        item_ids = [int(s) for s in id_strs]
    except ValueError:
        item_ids = id_strs
    print(f"Loaded {len(item_ids)} item embeddings  dim={matrix.shape[1]}  from {emb_dir}")
    return {item_id: matrix[i] for i, item_id in enumerate(item_ids)}


INTENT2ID = {
    "stay_profile": 0,
    "follow_recent": 1,
    "explore": 2,
}
INTENT_NAMES = {v: k for k, v in INTENT2ID.items()}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_training_data(
    ratings_df,
    items_df,
    k=20,
    train_m=20,
    eval_n=5,
    window_size=5,
    state_seq_len=4,
    rating_threshold=4.0,
    min_user_len=30,
    user_col="userId",
    item_col="movieId",
    rating_col="rating",
    time_col="timestamp",
    genres_col="genres",
    static_profile_path=None,
    embedding_model=None,
    item_embs_path=None,
    item_text_cols=None,
    alpha=1.0,
    lam=0.5,
    intent_method="score",
    sim_method="cosine",
    top_genre_k=None,
    start_after_profile=False,
    score_margin=0.05,
    explore_global=False,
    explore_global_top_k=500,
):
    """Build a hybrid dataset combining category histograms and item embeddings.

    Args:
        ratings_df: Interaction dataframe (user, item, rating, timestamp).
        items_df: Item metadata dataframe.
        k: Items used to build the static profile per user.
        train_m: Timesteps per user assigned to the training split.
        eval_n: Timesteps per user assigned to the intent-eval split.
        window_size: Recent items per dynamic state.
        state_seq_len: Consecutive states fed as model input.
        rating_threshold: Minimum rating counted as positive.
        min_user_len: Minimum positive interactions to include a user.
        user_col, item_col, rating_col, time_col, genres_col: Column names.
        static_profile_path: Optional precomputed category profile file
            (.json / .pkl / .csv / .parquet).  Overrides the first-k histogram.
        embedding_model: Encoder with ``encode(texts) -> np.ndarray``.
            Used when ``item_embs_path`` is not provided.
        item_embs_path: Directory produced by ``precompute_item_embs`` in
            ``intent_data_construction_2_emb``.  Takes priority over
            ``embedding_model`` — shared across emb mode, hybrid mode, and
            training/inference.
        item_text_cols: Item columns to concatenate as embedding input
            (e.g. ``["title", "genres"]``).  Only used when embedding
            on-the-fly via ``embedding_model``.
        alpha: Recency decay rate.  ``weight = exp(-alpha * age)`` where
            ``age=0`` for the most recent item.  ``alpha=0`` gives uniform
            weighting; larger values down-weight older items more steeply.
        lam: Blend weight in [0, 1].  ``lam=1`` is pure category; ``lam=0``
            is pure embedding.  Default 0.5 weights both equally.
        intent_method: ``"threshold"`` (default) or ``"score"``.
        sim_method: Similarity metric for the category component
            (``"cosine"`` default; category only — embeddings always use cosine).
        top_genre_k: If set, truncate category profile to top-k genres.
        start_after_profile: If True, sampling starts at t=k; else t=window_size.

    Returns:
        ``(samples, emb_tables)`` where ``samples`` has ``train``,
        ``intent_eval``, and ``sim_eval`` splits.  ``emb_tables`` contains
        ``item_emb_matrix``, ``item_ids``, ``profile_emb_matrix``,
        ``profile_user_ids``.
    """
    if item_embs_path is None and embedding_model is None:
        raise ValueError(
            "Provide either embedding_model (on-the-fly) or "
            "item_embs_path (pre-computed) to build_training_data."
        )

    print(f"Hybrid mode: alpha={alpha}, lam={lam}, intent_method={intent_method}, score_margin={score_margin}")

    # ------------------------------------------------------------------
    # Step 1: Category infrastructure.
    # ------------------------------------------------------------------
    item_genres = load_item_genres(items_df, item_col, genres_col)
    genre2idx = {
        g: i
        for i, g in enumerate(sorted({g for gs in item_genres.values() for g in gs}))
    }
    static_cat_profiles = load_static_profiles(static_profile_path, user_col, genre2idx)

    # ------------------------------------------------------------------
    # Step 2: Obtain item embeddings (load pre-computed or embed on-the-fly).
    # ------------------------------------------------------------------
    if item_embs_path is not None:
        item_embs = load_item_embs(item_embs_path)
    else:
        item_texts = build_item_texts(items_df, item_col, item_text_cols, item_genres)
        item_embs  = embed_items(item_texts, embedding_model)
        print(f"Embedded {len(item_embs)} items (dim={next(iter(item_embs.values())).shape[0]})")

    item_list   = sorted(item_embs.keys(), key=str)
    item_to_idx = {item: i for i, item in enumerate(item_list)}
    emb_dim     = next(iter(item_embs.values())).shape[0]

    if explore_global:
        # Identify the top-K most-interacted items (across all users, all ratings).
        # explore_global = cos(v_target, mean_emb of popular items NOT in H_u).
        item_counts = ratings_df[item_col].value_counts()
        popular_items = [
            item for item in item_counts.index
            if item in item_embs
        ][:explore_global_top_k]
        popular_set  = set(popular_items)
        N_popular    = len(popular_items)
        popular_sum_emb = np.stack([item_embs[iid] for iid in popular_items]).sum(axis=0).astype(np.float64)
        print(f"  explore_global: top-{explore_global_top_k} popular items → {N_popular} with embeddings")

    # ------------------------------------------------------------------
    # Step 3: Per-user profile embeddings (recency-weighted mean of first-k).
    # ------------------------------------------------------------------
    ratings_df = ratings_df.sort_values([user_col, time_col])
    profile_emb_cache: dict = {}
    profile_cat_cache: dict = {}

    for user_id, user_df in ratings_df.groupby(user_col):
        positives = [
            getattr(row, item_col)
            for row in user_df.itertuples(index=False)
            if getattr(row, rating_col) >= rating_threshold
            and getattr(row, item_col) in item_embs
        ]
        if len(positives) < min_user_len:
            continue

        profile_items = positives[:k]

        # Embedding profile: uniform mean over first-k items (no recency decay).
        emb_vecs = [item_embs[item] for item in profile_items]
        profile_emb_cache[user_id] = l2_normalize(
            np.stack(emb_vecs).mean(axis=0).astype(np.float32)
        )

        # Category profile: from precomputed or computed on-the-fly.
        if user_id in static_cat_profiles:
            cat_prof = static_cat_profiles[user_id]
        else:
            cat_items = [i for i in profile_items if i in item_genres]
            cat_prof = get_static_profile(
                user_id, {}, cat_items, item_genres, genre2idx
            )
        if top_genre_k is not None:
            cat_prof = keep_top_k_genres(cat_prof, top_genre_k)
        profile_cat_cache[user_id] = cat_prof

    profile_user_ids = sorted(profile_emb_cache.keys(), key=str)
    profile_idx_lookup = {uid: i for i, uid in enumerate(profile_user_ids)}

    # Per-user mean embedding of popular items NOT in H_u.
    # popular_sum_emb - sum(embs of user's popular items) / (N_popular - n_user_popular)
    uninteracted_emb_cache: dict = {}
    if explore_global:
        for user_id, user_df in ratings_df.groupby(user_col):
            if user_id not in profile_emb_cache:
                continue
            user_popular = [
                getattr(row, item_col)
                for row in user_df.itertuples(index=False)
                if getattr(row, item_col) in popular_set
            ]
            n_user_popular = len(user_popular)
            n_popular_unint = N_popular - n_user_popular
            if n_popular_unint <= 0:
                # user has touched every popular item — fall back to full popular mean
                v_unint = (popular_sum_emb / N_popular).astype(np.float32)
            elif n_user_popular == 0:
                v_unint = (popular_sum_emb / N_popular).astype(np.float32)
            else:
                user_pop_sum = np.stack([item_embs[i] for i in user_popular]).sum(axis=0).astype(np.float64)
                v_unint = ((popular_sum_emb - user_pop_sum) / n_popular_unint).astype(np.float32)
            uninteracted_emb_cache[user_id] = l2_normalize(v_unint)
    print(
        f"Built {len(profile_emb_cache)} profile embeddings and "
        f"{len(profile_cat_cache)} category profiles"
    )

    # ------------------------------------------------------------------
    # Step 4: Generate samples.
    # ------------------------------------------------------------------
    loop_start   = k if start_after_profile else window_size
    split_offset = k if start_after_profile else 0
    samples: dict = {"train": [], "intent_eval": [], "sim_eval": []}

    for user_id, user_df in ratings_df.groupby(user_col):
        positives = [
            getattr(row, item_col)
            for row in user_df.itertuples(index=False)
            if getattr(row, rating_col) >= rating_threshold
            and getattr(row, item_col) in item_embs
        ]
        if len(positives) < min_user_len:
            continue
        profile_emb = profile_emb_cache.get(user_id)
        if profile_emb is None:
            continue
        cat_profile = profile_cat_cache[user_id]
        profile_emb_idx_val = profile_idx_lookup.get(user_id, -1)
        v_uninteracted_norm = uninteracted_emb_cache.get(user_id, np.zeros(emb_dim, dtype=np.float32)) if explore_global else None

        states: list = []
        recent_cats_seq: list = []
        window_ids_seq: list = []
        raw_steps: list = []

        # --- Pass 1: compute blended similarities, collect raw steps ---------
        for t in range(loop_start, len(positives)):
            window = positives[max(0, t - window_size):t]
            target_item = positives[t]

            target_emb = item_embs.get(target_item, np.zeros(emb_dim, dtype=np.float32))
            target_cat = (
                genre_vector(item_genres[target_item], genre2idx)
                if target_item in item_genres
                else np.zeros(len(genre2idx), dtype=np.float32)
            )

            recent_emb = _weighted_mean_emb(window, item_embs, alpha)
            if recent_emb is None:
                recent_emb = np.zeros(emb_dim, dtype=np.float32)
            recent_cat = _weighted_cat_profile(window, item_genres, genre2idx, alpha)

            d_emb = float(1.0 - cosine(recent_emb, profile_emb))
            d_cat = float(1.0 - _cat_sim(recent_cat, cat_profile, sim_method))

            state = np.concatenate(
                [recent_emb, np.array([d_emb], dtype=np.float32)]
            )
            states.append(state)
            recent_cats_seq.append(recent_cat)
            window_ids_seq.append([item_to_idx.get(item, -1) for item in window])

            if len(states) < state_seq_len:
                continue

            s_profile_emb    = float(cosine(target_emb, profile_emb))
            s_recent_emb     = float(cosine(target_emb, recent_emb))
            s_profile_cat    = float(_cat_sim(target_cat, cat_profile, sim_method))
            s_recent_cat     = float(_cat_sim(target_cat, recent_cat,  sim_method))
            # s_explore_global: cos(v_target, v̄_{global\H_u})
            # "Is this item pointing toward content the user has never touched?"
            s_explore_global = float(cosine(target_emb, v_uninteracted_norm)) if explore_global else None

            raw_steps.append({
                "t":               t,
                "target_item":     target_item,
                "s_profile":       lam * s_profile_cat + (1.0 - lam) * s_profile_emb,
                "s_recent":        lam * s_recent_cat  + (1.0 - lam) * s_recent_emb,
                "drift":           lam * d_cat         + (1.0 - lam) * d_emb,
                "s_profile_emb":   s_profile_emb,
                "s_recent_emb":    s_recent_emb,
                "s_profile_cat":   s_profile_cat,
                "s_recent_cat":    s_recent_cat,
                "d_emb":           d_emb,
                "d_cat":           d_cat,
                "s_explore_global": s_explore_global,
                "state_seq":       np.stack(states[-state_seq_len:]),
                "recent_vector":   np.stack(recent_cats_seq[-state_seq_len:]),
                "window_ids":      np.array(window_ids_seq[-state_seq_len:], dtype=np.int32),
            })

        if not raw_steps:
            continue

        # --- Per-user min-max of blended similarities ------------------------
        sp_vals = [s["s_profile"] for s in raw_steps]
        sr_vals = [s["s_recent"]  for s in raw_steps]
        sp_min, sp_max = min(sp_vals), max(sp_vals)
        sr_min, sr_max = min(sr_vals), max(sr_vals)
        sp_range = sp_max - sp_min
        sr_range = sr_max - sr_min

        # --- Pass 2: normalize, score, assign --------------------------------
        for step in raw_steps:
            s_profile_n = (step["s_profile"] - sp_min) / (sp_range + 1e-8)
            s_recent_n  = (step["s_recent"]  - sr_min) / (sr_range + 1e-8)
            drift       = step["drift"]
            novelty     = (1.0 - s_profile_n) * (1.0 - s_recent_n)
            t           = step["t"]
            target_item = step["target_item"]

            intent_scores = compute_intent_scores(s_profile_n, s_recent_n, drift)
            vals  = np.array(list(intent_scores.values()), dtype=np.float64)
            total = vals.sum()
            confidence = float(vals.max() / total) if total > 0 else 1.0 / 3.0
            split = choose_split(t - split_offset, train_m, eval_n)

            if intent_method == "score":
                intent, _, _ = assign_intent_score(
                    s_profile_n, s_recent_n, margin=score_margin, scores=intent_scores
                )
                if intent is None:
                    if split == "train":
                        continue
                    intent = assign_intent_threshold(
                        s_profile_n, s_recent_n, novelty,
                        intent_method=intent_method, margin=score_margin,
                    )
            else:
                intent = assign_intent_threshold(
                    s_profile_n, s_recent_n, novelty,
                    intent_method=intent_method, margin=score_margin,
                )

            scores = {
                "profile_similarity":     s_profile_n,
                "recent_similarity":      s_recent_n,
                "profile_similarity_raw": step["s_profile"],
                "recent_similarity_raw":  step["s_recent"],
                "profile_similarity_emb": step["s_profile_emb"],
                "recent_similarity_emb":  step["s_recent_emb"],
                "profile_similarity_cat": step["s_profile_cat"],
                "recent_similarity_cat":  step["s_recent_cat"],
                "novelty":                float(novelty),
                "drift":                  drift,
                "drift_emb":              step["d_emb"],
                "drift_cat":              step["d_cat"],
                "stay_profile":           intent_scores["stay_profile"],
                "follow_recent":          intent_scores["follow_recent"],
                "explore":                intent_scores["explore"],
                **({"explore_global": step["s_explore_global"]} if explore_global else {}),
                "confidence":             confidence,
            }

            samples[split].append({
                "user_id":           user_id,
                "time_index":        t,
                "static_profile":    cat_profile.tolist(),
                "nl_static_profile": None,
                "state_sequence":    step["state_seq"],
                "recent_vector":     step["recent_vector"],
                "intent_label":      INTENT2ID[intent],
                "intent_name":       intent,
                "scores":            scores,
                "window_item_ids":   step["window_ids"],
                "target_item_id":    item_to_idx.get(target_item, -1),
                "profile_emb_idx":   profile_emb_idx_val,
            })

    emb_tables = {
        "item_emb_matrix":    np.stack([item_embs[i] for i in item_list]).astype(np.float16),
        "item_ids":           [str(i) for i in item_list],
        "profile_emb_matrix": np.stack(
            [profile_emb_cache[uid] for uid in profile_user_ids]
        ).astype(np.float16),
        "profile_user_ids":   [str(uid) for uid in profile_user_ids],
    }
    return samples, emb_tables


# ---------------------------------------------------------------------------
# Confidence / intent filtering
# ---------------------------------------------------------------------------

def filter_by_confidence(samples, threshold=0.10):
    """Filter training samples by confidence; evaluation splits are kept intact."""
    result = {}
    for split, rows in samples.items():
        if split == "train":
            result[split] = [
                s for s in rows if s["scores"].get("confidence", 0.0) >= threshold
            ]
        else:
            result[split] = rows
    return result


# ---------------------------------------------------------------------------
# Intent assignment
# ---------------------------------------------------------------------------


def compute_intent_scores(s_profile, s_recent, drift=0):
    """Evidence scores for each intent.

    Scores are raw similarities: stay_profile = s_profile, follow_recent = s_recent,
    explore = 0.5 * (1 - s_profile) * (1 - s_recent).

    Note: the alternative explore_global = cos(v_target, v̄_{global\H_u}) is computed
    upstream in build_training_data and stored separately in scores["explore_global"].
    """
    s_profile = float(np.clip(s_profile, 0.0, 1.0))
    s_recent  = float(np.clip(s_recent,  0.0, 1.0))
    return {
        "stay_profile":  s_profile,
        "follow_recent": s_recent,
        "explore":       0.5 * (1.0 - s_profile) * (1.0 - s_recent),
    }


def assign_intent_threshold(
    s_profile, s_recent, novelty,
    intent_method="threshold", margin=0.10, novelty_threshold=0.65,
):
    """Hard-rule intent assignment."""
    if "4" in intent_method:
        novelty = 1 - max(s_profile, s_recent)
        if novelty >= novelty_threshold:
            return "explore"
        margin = max(s_profile, s_recent) * 0.2
        if s_recent >= s_profile + margin:
            return "follow_recent"
    elif "3" in intent_method:
        novelty = 1 - max(s_profile, s_recent)
        if s_recent >= s_profile + margin:
            return "follow_recent"
        if novelty >= novelty_threshold:
            return "explore"
    elif "2" in intent_method:
        if s_recent >= s_profile + margin:
            return "follow_recent"
        if novelty >= novelty_threshold:
            return "explore"
    else:
        if novelty >= novelty_threshold:
            return "explore"
        if s_recent >= s_profile + margin:
            return "follow_recent"
    return "stay_profile"


def assign_intent_score(s_profile, s_recent, min_score=0.10, margin=0.05, scores=None):
    """Score-based intent assignment.  Returns ``(label, scores, gap)``."""
    if scores is None:
        scores = compute_intent_scores(s_profile, s_recent)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ranked[0]
    _, second_score = ranked[1]
    gap = best_score - second_score
    if best_score < min_score or gap < margin:
        return None, scores, gap
    return best_label, scores, gap


def choose_split(t, train_m, eval_n):
    if t < train_m:
        return "train"
    if t < train_m + eval_n:
        return "intent_eval"
    return "sim_eval"


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def _cat_sim(a, b, method="cosine"):
    """Similarity between two category vectors."""
    if method == "cosine":
        return cosine(a, b)
    from scipy.spatial.distance import jensenshannon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(1.0 - jensenshannon(a, b, base=2))


def _weighted_cat_profile(items, item_genres, genre2idx, alpha):
    """Recency-weighted L1-normalised genre histogram for a list of items.

    Items without genre info are skipped.  ``alpha=0`` gives uniform weights.
    """
    n = len(items)
    vec = np.zeros(len(genre2idx), dtype=np.float32)
    weight_total = 0.0
    for i, item in enumerate(items):
        if item not in item_genres:
            continue
        age = n - 1 - i   # age=0 for the most recent item
        w = float(np.exp(-alpha * age))
        vec += w * genre_vector(item_genres[item], genre2idx)
        weight_total += w
    if weight_total > 0:
        vec /= weight_total
    return normalize(vec)


def load_static_profiles(path, user_col, genre2idx):
    if path is None:
        return {}
    path = Path(path)
    if path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        return {uid: as_profile_vector(vec, genre2idx) for uid, vec in data.items()}
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            return _profiles_from_df(data, user_col, genre2idx)
        return {uid: as_profile_vector(vec, genre2idx) for uid, vec in data.items()}
    if path.suffix == ".csv":
        return _profiles_from_df(pd.read_csv(path), user_col, genre2idx)
    if path.suffix == ".parquet":
        return _profiles_from_df(pd.read_parquet(path), user_col, genre2idx)
    raise ValueError(f"Unsupported static profile file: {path}")


def _profiles_from_df(df, user_col, genre2idx):
    if "static_profile" in df.columns:
        return {
            uid: as_profile_vector(_parse_profile(val), genre2idx)
            for uid, val in zip(df[user_col], df["static_profile"])
        }
    genre_cols = [g for g in genre2idx if g in df.columns]
    if len(genre_cols) != len(genre2idx):
        raise ValueError("Profile dataframe needs `static_profile` or genre columns.")
    matrix = df[genre_cols].to_numpy(dtype=np.float32)
    return {uid: normalize(matrix[i]) for i, uid in enumerate(df[user_col])}


def _parse_profile(value):
    if isinstance(value, str):
        return json.loads(value) if value.strip().startswith("[") else ast.literal_eval(value)
    return value


def as_profile_vector(value, genre2idx):
    if isinstance(value, dict):
        vec = np.zeros(len(genre2idx), dtype=np.float32)
        for g, s in value.items():
            if g in genre2idx:
                vec[genre2idx[g]] = s
        return normalize(vec)
    vec = np.asarray(value, dtype=np.float32)
    if vec.shape != (len(genre2idx),):
        raise ValueError(f"Profile length {len(vec)} != genre size {len(genre2idx)}.")
    return normalize(vec)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def build_item_texts(items_df, item_col, text_cols, item_genres):
    """Build {item_id: text} for ALL items in items_df.

    Falls back to comma-joined genre names when ``text_cols`` is None or when
    no valid column values are found for a row.
    """
    if not text_cols:
        return {
            item: ", ".join(genres) for item, genres in item_genres.items()
        }
    available_cols = [c for c in text_cols if c in items_df.columns]
    genre_fallback = {item: ", ".join(gs) for item, gs in item_genres.items()}
    result = {}
    for row in items_df[[item_col] + available_cols].itertuples(index=False):
        item_id = getattr(row, item_col)
        parts = []
        for col in available_cols:
            val = getattr(row, col, None)
            s = str(val).strip() if val is not None else ""
            if s and s.lower() not in ("nan", "none", ""):
                parts.append(s)
        result[item_id] = ". ".join(parts) if parts else genre_fallback.get(item_id, str(item_id))
    return result


def embed_items(item_texts, embedder):
    """Embed all items from a pre-built text dict.

    Returns ``{item_id: l2_normalised_float32_array}``.
    """
    items = list(item_texts.keys())
    texts = [item_texts[item] for item in items]
    raw = embedder.encode(texts)
    return {
        item: l2_normalize(np.asarray(emb, dtype=np.float32))
        for item, emb in zip(items, raw)
    }


def _recency_weights(n, alpha):
    """L1-normalised recency weights for n items (index 0=oldest, n-1=most recent).

    ``weight[i] = exp(-alpha * (n-1-i))`` so most recent item has weight 1
    before normalisation.  ``alpha=0`` gives uniform weights.
    """
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)   # [n-1, n-2, ..., 0]
    w = np.exp(-alpha * ages).astype(np.float32)
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / n, dtype=np.float32)


def _weighted_mean_emb(items, item_embs, alpha):
    """Recency-weighted L2-normalised mean embedding for a list of items.

    Items absent from ``item_embs`` are skipped.  Returns ``None`` when none
    of the items are embeddable.
    """
    valid = [(i, item_embs[item]) for i, item in enumerate(items) if item in item_embs]
    if not valid:
        return None
    idxs, embs = zip(*valid)
    n = len(items)
    ages = np.array([n - 1 - i for i in idxs], dtype=np.float32)
    w = np.exp(-alpha * ages).astype(np.float32)
    w /= w.sum()
    return l2_normalize((np.stack(embs) * w[:, None]).sum(0).astype(np.float32))


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)


# ---------------------------------------------------------------------------
# HuggingFace embedder wrapper
# ---------------------------------------------------------------------------

class HFEmbedder:
    """Wraps a HuggingFace AutoModel for mean-pooled sentence embeddings.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
        batch_size: Texts per forward pass.
        max_length: Tokeniser truncation length.
    """

    def __init__(self, model_name_or_path, device="mps", batch_size=512, max_length=512):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, texts):
        import torch
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embs = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)
