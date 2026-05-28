from collections import Counter

import numpy as np

_EMBEDDER_CACHE: dict = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and cache a SentenceTransformer model."""
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def embed_texts(
    texts: list,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """L2-normalised embeddings.

    Args:
        texts: List of strings to embed.
        model_name: SentenceTransformer model name or path.
        batch_size: Encoding batch size.

    Returns:
        Float32 array of shape ``(len(texts), embed_dim)``.
        Cosine similarity between rows equals their dot product.
    """
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    model = get_embedder(model_name)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embs.astype(np.float32)


def item_to_text(item_id, item_genres: dict, item_authors: dict) -> str:
    """Short text description of an item for embedding.

    Example: "Genres: Mystery, Thriller. Author: John Grisham"
    """
    genres = item_genres.get(item_id, [])
    author = item_authors.get(item_id)
    parts = []
    if genres:
        parts.append("Genres: " + ", ".join(genres))
    if author:
        parts.append("Author: " + author)
    return ". ".join(parts) if parts else "Unknown item"


def window_to_text(
    window_items: list,
    item_genres: dict,
    item_authors: dict,
    top_k: int = 5,
) -> str:
    """Natural-language summary of a recent interaction window.

    Example: "Recent genres: Mystery (40%), Thriller (30%). Recent authors: A, B"
    """
    genre_counts: Counter = Counter()
    seen_authors: list = []
    for item in window_items:
        for g in item_genres.get(item, []):
            genre_counts[g] += 1
        a = item_authors.get(item)
        if a and a not in seen_authors:
            seen_authors.append(a)
    total = sum(genre_counts.values()) or 1
    top_genres = genre_counts.most_common(top_k)
    genre_str = ", ".join(f"{g} ({c/total:.0%})" for g, c in top_genres)
    text = "Recent genres: " + (genre_str or "unknown")
    if seen_authors:
        text += ". Recent authors: " + ", ".join(seen_authors[:3])
    return text


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D float32 array in-place-free."""
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)


def weighted_mean_emb(
    window_items: list,
    item_embs: dict,
    decay_factor: float = 1.0,
) -> np.ndarray:
    """L2-normalised weighted mean of pre-computed item embeddings.

    Items missing from ``item_embs`` are skipped.  Falls back to a zero
    vector of the correct embedding dimension when the window is empty or
    no items are found.  Uses the same decay scheme as ``make_state_book``:
    weight for position i = ``decay_factor ** (n-1-i)`` so the most recent
    item always has weight 1.

    This is a pure numpy operation — no model calls are needed.
    """
    valid = [(i, item_embs[item]) for i, item in enumerate(window_items) if item in item_embs]
    if not valid:
        emb_dim = next(iter(item_embs.values())).shape[0] if item_embs else 1
        return np.zeros(emb_dim, dtype=np.float32)
    if decay_factor == 1.0:
        return l2_normalize(np.mean([e for _, e in valid], axis=0).astype(np.float32))
    n = len(window_items)
    weights = np.array([decay_factor ** (n - 1 - i) for i, _ in valid], dtype=np.float32)
    embs = np.stack([e for _, e in valid])
    return l2_normalize((embs * weights[:, None]).sum(0) / weights.sum())
