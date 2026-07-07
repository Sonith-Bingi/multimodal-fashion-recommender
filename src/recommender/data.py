"""Raw data download, k-core filtering, and sequence construction."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _token_fallback_embedding(text: str, dim: int = 768) -> np.ndarray:
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
]:
    random.seed(42)
    train_seqs: list[tuple[list[int], int]] = []
    val_seqs: list[tuple[list[int], int]] = []
    val_novel_seqs: list[tuple[list[int], int]] = []

    for _uid, events in user_events.items():
        events = sorted(events, key=lambda x: x[0])
        idxs = [asin_to_idx[a] for _, a in events if a in asin_to_idx]
        if len(idxs) < min_seq:
            continue

        for i in range(1, len(idxs) - 1):
            hist = idxs[max(0, i - seq_len) : i]
            target = idxs[i]
            train_seqs.append((hist, target))

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

    random.shuffle(train_seqs)

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

    return train_seqs, val_seqs, val_novel_seqs, sparse_val_seqs
