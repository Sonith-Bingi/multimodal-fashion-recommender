"""Retrieval helpers (FAISS, cosine similarity) and ranking metrics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ArtifactStatus:
    catalog: bool
    index: bool
    vectors: bool
    meta: bool
    reviews: bool


@dataclass
class EvalMetrics:
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _try_import_faiss() -> Any | None:
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


def _recall_at_k(retrieved: list[tuple[float, int]], target: int, k: int) -> float:
    return 1.0 if target in [idx for _, idx in retrieved[:k]] else 0.0


def _ndcg_at_k(retrieved: list[tuple[float, int]], target: int, k: int) -> float:
    for rank, (_, idx) in enumerate(retrieved[:k]):
        if idx == target:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def _mrr_at_k(retrieved: list[tuple[float, int]], target: int, k: int) -> float:
    for rank, (_, idx) in enumerate(retrieved[:k]):
        if idx == target:
            return 1.0 / (rank + 1)
    return 0.0


def _reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]], rrf_k: int = 60
) -> list[tuple[float, int]]:
    """Blend multiple ranked candidate lists into one fused ranking.

    Standard technique for combining retrieval signals whose raw scores
    aren't on comparable scales (e.g. a trained model's similarity vs. raw
    cosine similarity in a different embedding space): each item gets
    sum(1 / (rrf_k + rank)) across every list it appears in, so only rank
    position matters, not the underlying score magnitude.
    """
    scores: dict[int, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, item_id in enumerate(ranked_ids):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(((score, item_id) for item_id, score in scores.items()), reverse=True)


def _mmr_rerank(
    vectors: np.ndarray,
    candidates: list[tuple[float, int]],
    k: int,
    lambda_mult: float = 0.5,
) -> list[tuple[float, int]]:
    """Re-rank retrieved candidates for diversity via Maximal Marginal
    Relevance (Carbonell & Goldstein, 1998).

    Without this, top-k can be dominated by near-duplicate items (five
    color variants of the same shirt) because they all score similarly well
    against the query. MMR greedily picks the candidate that best trades off
    relevance against redundancy with what's already been selected:
    argmax [ lambda * relevance(i) - (1 - lambda) * max_sim(i, selected) ].
    Relevance is taken from rank position (not the raw incoming score) so
    this works regardless of whether `candidates` came from cosine
    similarity, FAISS inner product, or RRF fusion scores -- those aren't on
    comparable scales, but rank order always is. lambda_mult=1.0 recovers
    pure relevance ranking; lower values favor diversity more.

    0.5, not the more conservative 0.7 this started at: for a tight-cluster
    query (e.g. history = Winter Coat + Beanie + Gloves), 0.7 still returned
    8 near-identical winter gloves -- the top-scoring candidates are all
    genuinely close in embedding space, so relevance alone doesn't create
    room for anything else. 0.5 (checked manually against real catalog data)
    starts pulling in scarves and hat/scarf sets instead of more gloves,
    without drifting off-topic; 0.35 goes too far and admits irrelevant
    items (a leather belt, welding gloves) purely because they're
    dissimilar. Needs a real candidate pool to work with -- see fetch_k in
    train.py's _retrieve(), bumped alongside this for the same reason.

    Serving-time only: applying this during evaluation would trade away
    recall/ndcg/mrr for diversity and make metrics incomparable to prior
    runs, so callers should keep it off when scoring against a fixed split.
    """
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return candidates

    n = len(candidates)
    cand_idx = [idx for _, idx in candidates]
    cand_vecs = _normalize_rows(vectors[np.array(cand_idx)])
    relevance = [1.0 - rank / n for rank in range(n)]

    selected_pos: list[int] = []
    selected_vecs: list[np.ndarray] = []
    remaining = list(range(n))

    while remaining and len(selected_pos) < k:
        best_pos, best_val = remaining[0], float("-inf")
        for pos in remaining:
            redundancy = (
                max(float(cand_vecs[pos] @ v) for v in selected_vecs) if selected_vecs else 0.0
            )
            val = lambda_mult * relevance[pos] - (1 - lambda_mult) * redundancy
            if val > best_val:
                best_val, best_pos = val, pos
        selected_pos.append(best_pos)
        selected_vecs.append(cand_vecs[best_pos])
        remaining.remove(best_pos)

    return [candidates[pos] for pos in selected_pos]


# Common words that appear across many product titles regardless of type
# ("Women's", "for Men", sizes) -- stripping these keeps _title_keywords()
# focused on words that actually distinguish one product type from another.
_TITLE_STOPWORDS = frozenset(
    {
        "with",
        "womens",
        "mens",
        "women",
        "men",
        "size",
        "black",
        "white",
        "your",
        "this",
        "that",
        "unisex",
        "fashion",
        "casual",
        "classic",
    }
)


def _title_keywords(title: str) -> set[str]:
    """Coarse per-item vocabulary for the redundancy check in
    _diversify_beyond_history() below -- words of 4+ letters, common
    boilerplate stripped. Not a real category taxonomy (the catalog doesn't
    have one -- every item's `categories` field is empty in this dataset),
    just a cheap proxy for "these two titles are about the same kind of
    product," which is exactly the granularity that check needs."""
    return set(re.findall(r"[a-z]{4,}", title.lower())) - _TITLE_STOPWORDS


def _diversify_beyond_history(
    vectors: np.ndarray,
    candidates: list[tuple[float, int]],
    k: int,
    history_titles: list[str],
    item_titles: list[str],
    novel_frac: float = 0.45,
    lambda_mult: float = 0.6,
) -> list[tuple[float, int]]:
    """Split `candidates` into items that share a title keyword with the
    user's history ("more of what they already have") vs. items that
    don't ("a genuinely different next item"), fill roughly novel_frac of
    the k slots from the latter, and MMR-diversify within each bucket so
    picks aren't just whatever happened to rank highest overall.

    Why this exists on top of _mmr_rerank(): MMR alone reduces redundancy
    *between the recommended items*, not redundancy *with what the user
    already has* -- it will happily fill 8 slots with 8 different-looking
    gloves if gloves are what scores best, because none of them are
    similar to each other. Measured on real held-out Amazon purchase
    sequences: the actual next item a real shopper buys shares a title
    keyword with their recent history only ~61% of the time (39% of real
    "next purchases" are a different kind of product entirely), but this
    model's raw top-8 was doing so ~90% of the time -- a real,
    measured over-indexing on same-category re-recommendation, not
    hypothetical. novel_frac=0.45 (checked against the same real-sequence
    benchmark) lands close to that 61/39 real-world split, erring slightly
    toward more variety rather than less.

    `candidates` should be a plain relevance-ranked list (diversify=False),
    not already MMR-diversified -- this needs the full ranked pool to find
    good novel candidates, not just whatever survived an earlier cut.
    """
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return candidates

    history_keywords: set[str] = set()
    for title in history_titles:
        history_keywords |= _title_keywords(title)

    novel_pool: list[tuple[float, int]] = []
    overlap_pool: list[tuple[float, int]] = []
    for score, idx in candidates:
        shares_keyword = bool(_title_keywords(item_titles[idx]) & history_keywords)
        (overlap_pool if shares_keyword else novel_pool).append((score, idx))

    n_novel = min(round(k * novel_frac), len(novel_pool))
    n_overlap = k - n_novel

    picked_novel = _mmr_rerank(vectors, novel_pool, k=n_novel, lambda_mult=lambda_mult)
    picked_overlap = _mmr_rerank(vectors, overlap_pool, k=n_overlap, lambda_mult=lambda_mult)

    combined = picked_novel + picked_overlap
    combined.sort(key=lambda pair: pair[0], reverse=True)
    return combined
