"""Raw data download, k-core filtering, and sequence construction."""

from __future__ import annotations

import hashlib
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature-hashed brand/store embedding vocab size. A fixed small bucket
# count avoids an unbounded, ever-growing embedding table as new brands
# appear (a real product catalog has thousands of distinct stores, most
# seen only once or twice -- one embedding row per brand would mostly be
# untrained noise, same failure mode as item cold-start).
BRAND_HASH_BUCKETS = 256


def _hash_brand(brand: str, n_buckets: int = BRAND_HASH_BUCKETS) -> int:
    """Stable, process-independent hash into a fixed bucket count. Not
    Python's built-in hash(), which is randomized per-process for str by
    default (PYTHONHASHSEED) -- this project's determinism guarantee
    (byte-identical metrics across independent runs, see MODEL_CARD.md)
    would silently break if brand IDs weren't reproducible run to run."""
    digest = hashlib.md5(brand.strip().lower().encode("utf-8")).hexdigest()
    return int(digest, 16) % n_buckets


def _token_fallback_embedding(text: str, dim: int = 512) -> np.ndarray:
    tokens = [t.strip().lower() for t in text.split() if t.strip()]
    if not tokens:
        return np.zeros(dim, dtype=np.float32)
    vecs = []
    for tok in tokens:
        seed = abs(hash(tok)) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        vecs.append(rng.standard_normal(dim).astype(np.float32))
    out = np.mean(np.stack(vecs, axis=0), axis=0)
    out = out / (np.linalg.norm(out) + 1e-12)
    return out.astype(np.float32)


# Time-gap bucket edges in seconds, roughly log-scaled (TiSASRec-style
# discretization of elapsed time between consecutive interactions -- Li et
# al., "Time Interval Aware Self-Attention for Sequential Recommendation",
# WSDM'20). Bucket 0 is reserved for "no real gap available": the first
# position in any history (nothing precedes it) and, via UserTower's
# default, every position when a caller has no real timestamps at all --
# which is always true of the recommend_for_history() serving API, since it
# only ever receives free-text product names, never timestamps. Training
# and its internal validation split (used for early stopping) use real
# gaps; the offline recall/ndcg/mrr evaluation and the serving API
# deliberately do not, so the reported metrics reflect what's actually
# available at serving time rather than an easier setup the model won't
# have in production -- see MODEL_CARD.md.
_TIME_BUCKET_EDGES = (
    60,  # < 1 minute
    3600,  # < 1 hour
    86400,  # < 1 day
    7 * 86400,  # < 1 week
    30 * 86400,  # < 1 month
    90 * 86400,  # < 3 months
    180 * 86400,  # < 6 months
    365 * 86400,  # < 1 year
    2 * 365 * 86400,  # < 2 years
)
N_TIME_BUCKETS = len(_TIME_BUCKET_EDGES) + 2  # +unknown/first bucket, +">= 2 years" bucket


def _time_gap_bucket(gap_seconds: float | None) -> int:
    if gap_seconds is None or gap_seconds < 0:
        return 0
    for i, edge in enumerate(_TIME_BUCKET_EDGES):
        if gap_seconds < edge:
            return i + 1
    return len(_TIME_BUCKET_EDGES) + 1


def _hist_time_buckets(hist_ts: list[int]) -> list[int]:
    """Per-position time-gap bucket for a history: bucket 0 (unknown/first)
    for position 0, then the gap-to-previous-event bucket for every
    position after it."""
    if not hist_ts:
        return []
    buckets = [0]
    for j in range(1, len(hist_ts)):
        buckets.append(_time_gap_bucket(hist_ts[j] - hist_ts[j - 1]))
    return buckets


def _download_from_hub(hf_filename: str, dest: Path) -> None:
    if dest.exists():
        return

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download raw Amazon Fashion files"
        ) from exc

    logger.info("Downloading %s", dest.name)
    cached = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=hf_filename,
        repo_type="dataset",
    )
    shutil.copy(cached, dest)


def _filter_k_core(events_dict: dict[str, list[tuple[int, str]]], k: int) -> tuple[
    dict[str, list[tuple[int, str]]],
    set[str],
    int,
    int,
]:
    records: list[dict[str, Any]] = []
    for uid, events in events_dict.items():
        for ts, asin in events:
            records.append({"user_id": uid, "asin": asin, "ts": ts})

    df_events = pd.DataFrame(records)
    if df_events.empty:
        return {}, set(), 0, 0

    rounds = 0
    while True:
        rounds += 1
        start_len = len(df_events)

        item_counts = df_events["asin"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df_events = df_events[df_events["asin"].isin(valid_items)]

        user_counts = df_events["user_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df_events = df_events[df_events["user_id"].isin(valid_users)]

        if len(df_events) == start_len:
            break

    filtered_events: dict[str, list[tuple[int, str]]] = {}
    for row in df_events.itertuples(index=False):
        filtered_events.setdefault(str(row.user_id), []).append((int(row.ts), str(row.asin)))

    return filtered_events, set(df_events["asin"].unique()), rounds, len(df_events)


def _build_sequences(
    user_events: dict[str, list[tuple[int, str]]],
    raw_user_events_backup: dict[str, list[tuple[int, str]]],
    asin_to_idx: dict[str, int],
    seq_len: int,
    min_seq: int,
    n_catalog: int,
) -> tuple[
    list[tuple[list[int], int]],
    list[tuple[list[int], int]],
    list[tuple[list[int], int]],
    list[tuple[list[int], int]],
    list[list[int]],
]:
    random.seed(42)
    train_seqs: list[tuple[list[int], int]] = []
    train_time_buckets: list[list[int]] = []
    val_seqs: list[tuple[list[int], int]] = []
    val_novel_seqs: list[tuple[list[int], int]] = []

    for _uid, events in user_events.items():
        events = sorted(events, key=lambda x: x[0])
        idxs = [asin_to_idx[a] for _, a in events if a in asin_to_idx]
        timestamps = [ts for ts, a in events if a in asin_to_idx]
        if len(idxs) < min_seq:
            continue

        for i in range(1, len(idxs) - 1):
            hist = idxs[max(0, i - seq_len) : i]
            hist_ts = timestamps[max(0, i - seq_len) : i]
            target = idxs[i]
            train_seqs.append((hist, target))
            train_time_buckets.append(_hist_time_buckets(hist_ts))

        i = len(idxs) - 1
        hist_last = idxs[max(0, i - seq_len) : i]
        target_last = idxs[i]
        val_seqs.append((hist_last, target_last))

        for j in range(len(idxs) - 1, 0, -1):
            hist_j = idxs[max(0, j - seq_len) : j]
            target_j = idxs[j]
            if target_j not in set(hist_j):
                val_novel_seqs.append((hist_j, target_j))
                break

    # Shuffle train_seqs and train_time_buckets together (same permutation)
    # -- they're parallel arrays, index i of one must stay index i of the
    # other after shuffling.
    paired = list(zip(train_seqs, train_time_buckets, strict=True))
    random.shuffle(paired)
    if paired:
        train_seqs, train_time_buckets = (list(t) for t in zip(*paired, strict=True))
    else:
        train_seqs, train_time_buckets = [], []

    dense_user_ids = set(user_events.keys())
    sparse_val_seqs: list[tuple[list[int], int]] = []
    for uid, events in raw_user_events_backup.items():
        if uid in dense_user_ids:
            continue
        events = sorted(events, key=lambda x: x[0])
        idxs = [asin_to_idx[a] for _, a in events if a in asin_to_idx]
        if len(idxs) >= 2:
            hist = idxs[:-1]
            target = idxs[-1]
            if 0 <= target < n_catalog:
                sparse_val_seqs.append((hist[-seq_len:], target))

    return train_seqs, val_seqs, val_novel_seqs, sparse_val_seqs, train_time_buckets
