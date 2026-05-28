import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_genre_profiles(
    ratings_df,
    items_df,
    k=20,
    rating_threshold=3.0,
    min_user_len=30,
    top_genre_k=None,
    user_col="userid",
    item_col="movieid",
    rating_col="rating",
    time_col="timestamp",
    genres_col="genres",
):
    item_genres = load_item_genres(items_df, item_col, genres_col)
    genre2idx = {
        g: i
        for i, g in enumerate(sorted({g for gs in item_genres.values() for g in gs}))
    }

    profiles = {}
    ratings_df = ratings_df.sort_values([user_col, time_col])

    for user_id, user_df in ratings_df.groupby(user_col):
        positives = [
            getattr(row, item_col)
            for row in user_df.itertuples(index=False)
            if getattr(row, rating_col) >= rating_threshold
            and getattr(row, item_col) in item_genres
        ]
        if len(positives) < min_user_len:
            continue

        vec = genre_distribution(positives[:k], item_genres, genre2idx)
        if top_genre_k is not None:
            vec = keep_top_k_genres(vec, top_genre_k)
        profiles[user_id] = vec

    return profiles, genre2idx


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_genre_profiles(profiles, genre2idx, output_path, user_col="userid"):
    """Save profiles to parquet (default) or CSV depending on the extension.

    The file has one row per user with columns: ``user_col``, then one float32
    column per genre in ``genre2idx`` order.
    """
    output_path = Path(output_path)
    idx2genre = {idx: genre for genre, idx in genre2idx.items()}
    genre_cols = [idx2genre[i] for i in range(len(genre2idx))]

    rows = []
    for user_id, vec in profiles.items():
        row = {user_col: user_id}
        for i, col in enumerate(genre_cols):
            row[col] = float(vec[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".json":
        # {user_id: {genre: weight, ...}} — human-readable
        out = {
            str(uid): {
                g: float(df.loc[df[user_col] == uid, g].iloc[0])
                for g in genre_cols
                if float(df.loc[df[user_col] == uid, g].iloc[0]) > 0
            }
            for uid in df[user_col]
        }
        output_path.write_text(json.dumps(out, indent=2))
    else:
        df.to_parquet(output_path, index=False)

    print(f"Saved {len(profiles)} user profiles → {output_path}")
    return df


def load_genre_profiles(path, user_col="userid"):
    """Load profiles saved by ``save_genre_profiles``.

    Returns ``{user_id: np.ndarray}`` with the same dtype as at save time.
    """
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        genre_cols = sorted({g for d in data.values() for g in d})
        rows = []
        for uid, gdict in data.items():
            row = {user_col: uid}
            row.update({g: gdict.get(g, 0.0) for g in genre_cols})
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = pd.read_parquet(path)

    genre_cols = [c for c in df.columns if c != user_col]
    genre_matrix = df[genre_cols].to_numpy(dtype=np.float32)
    profiles = {}
    for i, uid in enumerate(df[user_col]):
        profiles[uid] = genre_matrix[i]
    return profiles


def load_item_genres(items_df, item_col, genres_col):
    """Build {item_id: [genre, ...]} from the items dataframe."""
    item_genres = {}
    for row in items_df[[item_col, genres_col]].dropna().itertuples(index=False):
        genres = [g for g in str(getattr(row, genres_col)).split("|") if g]
        if genres and genres != ["(no genres listed)"]:
            item_genres[getattr(row, item_col)] = genres
    return item_genres


def get_static_profile(user_id, static_profiles, init_items, item_genres, genre2idx):
    """Return precomputed profile if available, else build from init_items."""
    if user_id in static_profiles:
        return static_profiles[user_id]
    if str(user_id) in static_profiles:
        return static_profiles[str(user_id)]
    return genre_distribution(init_items, item_genres, genre2idx)


def genre_distribution(items, item_genres, genre2idx):
    """L1-normalised genre histogram over a list of items."""
    vec = np.zeros(len(genre2idx), dtype=np.float32)
    for item in items:
        vec += genre_vector(item_genres[item], genre2idx)
    return normalize(vec)


def genre_vector(genres, genre2idx):
    """L1-normalised one-hot genre vector for a single item."""
    vec = np.zeros(len(genre2idx), dtype=np.float32)
    for genre in genres:
        if genre in genre2idx:
            vec[genre2idx[genre]] = 1.0
    return normalize(vec)


def normalize(vec):
    total = vec.sum()
    return vec / total if total else vec


def keep_top_k_genres(vec, k):
    """Filter genre weights by top-k or minimum-weight threshold.

    If k >= 1 (integer mode): keep the k largest entries and re-normalise.
    If 0 < k < 1 (float mode): keep entries with weight >= k; no re-normalisation
        so weights remain their original L1 fractions (e.g. 0.05 keeps all genres
        that represent ≥ 5% of the user's taste history).
    """
    if k >= 1:
        k = int(k)
        if k >= len(vec):
            return vec
        threshold = np.partition(vec, -k)[-k]
        masked = np.where(vec >= threshold, vec, 0.0).astype(np.float32)
        return normalize(masked)
    else:
        return np.where(vec >= k, vec, 0.0).astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="Build genre-based static user profiles.")
    p.add_argument("--ratings",           required=True, help="Path to ratings CSV.")
    p.add_argument("--items",             required=True, help="Path to items CSV.")
    p.add_argument("--output", default="genre_profiles/genre_profiles.json",            required=True, help="Output file (.parquet, .csv, or .json).")
    p.add_argument("--k",                 type=int,   default=10,  help="Profile depth (default 20).")
    p.add_argument("--rating-threshold",  type=float, default=3.0, help="Min rating to count as positive (default 3).")
    p.add_argument("--min-user-len",      type=int,   default=30,  help="Min positive interactions to include a user (default 30).")
    p.add_argument("--top-genre-k",       type=int,   default=None, help="Keep only top-k genres (default: all).")
    p.add_argument("--user-col",          default="userid")
    p.add_argument("--item-col",          default="movieid")
    p.add_argument("--rating-col",        default="rating")
    p.add_argument("--time-col",          default="timestamp")
    p.add_argument("--genres-col",        default="genres")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ratings_df = pd.read_csv(args.ratings)
    items_df   = pd.read_csv(args.items)

    profiles, genre2idx = build_genre_profiles(
        ratings_df, items_df,
        k=args.k,
        rating_threshold=args.rating_threshold,
        min_user_len=args.min_user_len,
        top_genre_k=args.top_genre_k,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        time_col=args.time_col,
        genres_col=args.genres_col,
    )

    print(f"Built profiles for {len(profiles)} users, {len(genre2idx)} genres.")
    save_genre_profiles(profiles, genre2idx, args.output, user_col=args.user_col)
