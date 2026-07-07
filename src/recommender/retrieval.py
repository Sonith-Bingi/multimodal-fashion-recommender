"""Retrieval helpers (FAISS, cosine similarity) and ranking metrics."""

from __future__ import annotations

import math
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
