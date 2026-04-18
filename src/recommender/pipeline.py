from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class ArtifactStatus:
    catalog: bool
    index: bool
    vectors: bool


@dataclass
class EvalMetrics:
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _token_vector(token: str, dim: int) -> np.ndarray:
    seed = abs(hash(token)) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)


def _text_embedding(text: str, dim: int) -> np.ndarray:
    tokens = [t.strip().lower() for t in text.replace("&", " ").split() if t.strip()]
    if not tokens:
        return np.zeros(dim, dtype=np.float32)
    vecs = np.stack([_token_vector(t, dim) for t in tokens], axis=0)
    return vecs.mean(axis=0).astype(np.float32)


def _image_embedding(item_id: int, dim: int, seed_offset: int) -> np.ndarray:
    rng = np.random.default_rng(seed_offset + int(item_id))
    return rng.standard_normal(dim).astype(np.float32)


def _build_multimodal_vectors(df: pd.DataFrame, seed: int) -> np.ndarray:
    text_dim = 64
    image_dim = 64
    text = np.stack([_text_embedding(t, text_dim) for t in df["category_name"].astype(str)], axis=0)
    image = np.stack(
        [_image_embedding(i, image_dim, seed_offset=seed) for i in df["id"].astype(int)], axis=0
    )
    fused = np.concatenate([text, image], axis=1).astype(np.float32)
    return _normalize_rows(fused)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    part = np.argpartition(-scores, kth=k - 1)[:k]
    return part[np.argsort(-scores[part])]


def _simulate_sequences(item_ids: np.ndarray, n_users: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    seqs: list[list[int]] = []
    n_items = len(item_ids)
    for _ in range(n_users):
        length = int(rng.integers(4, 10))
        start_idx = int(rng.integers(0, n_items))
        seq = [int(item_ids[start_idx])]
        current = start_idx
        for _step in range(length - 1):
            jump = int(rng.integers(-3, 4))
            current = int(np.clip(current + jump, 0, n_items - 1))
            seq.append(int(item_ids[current]))
        seqs.append(seq)
    return seqs


def _build_transition_matrix(seqs: list[list[int]], id_to_pos: dict[int, int], n_items: int) -> np.ndarray:
    trans = np.zeros((n_items, n_items), dtype=np.float32)
    for seq in seqs:
        for left, right in zip(seq[:-1], seq[1:]):
            li = id_to_pos[left]
            ri = id_to_pos[right]
            trans[li, ri] += 1.0
    row_sum = trans.sum(axis=1, keepdims=True) + 1e-12
    return trans / row_sum


def _save_index(path: Path, ids: np.ndarray) -> None:
    payload = {"ids": ids.astype(int).tolist()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_index(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["ids"], dtype=np.int64)


class RecommenderPipeline:
    """Production, script-first pipeline (no notebook dependency)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_catalog(self) -> pd.DataFrame:
        path = self.settings.catalog_path
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        df = pd.read_csv(path)
        required = {"id", "category_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
        return df[["id", "category_name"]].dropna().reset_index(drop=True)

    def validate_artifacts(self) -> ArtifactStatus:
        status = ArtifactStatus(
            catalog=self.settings.catalog_path.exists(),
            index=self.settings.index_path.exists(),
            vectors=self.settings.vectors_path.exists(),
        )
        logger.info(
            "Artifact status | catalog=%s index=%s vectors=%s",
            status.catalog,
            status.index,
            status.vectors,
        )
        return status

    def summarize_pipeline(self) -> dict[str, object]:
        df = self.load_catalog()
        status = self.validate_artifacts()
        return {
            "catalog_path": str(self.settings.catalog_path),
            "catalog_items": int(len(df)),
            "artifacts_dir": str(self.settings.artifacts_dir),
            "artifact_status": status.__dict__,
            "dense_k": self.settings.dense_k,
            "seq_len": self.settings.seq_len,
            "min_seq": self.settings.min_seq,
            "batch_size": self.settings.batch_size,
        }

    def train(self) -> dict[str, object]:
        ensure_dir(self.settings.artifacts_dir)
        df = self.load_catalog()

        item_ids = df["id"].to_numpy(dtype=np.int64)
        id_to_pos = {int(i): idx for idx, i in enumerate(item_ids.tolist())}

        vectors = _build_multimodal_vectors(df, seed=self.settings.random_seed)
        seqs = _simulate_sequences(item_ids, n_users=max(300, len(df)), seed=self.settings.random_seed)
        transition = _build_transition_matrix(seqs, id_to_pos=id_to_pos, n_items=len(item_ids))

        smoothed = 0.8 * vectors + 0.2 * (transition @ vectors)
        smoothed = _normalize_rows(smoothed.astype(np.float32))

        np.save(self.settings.vectors_path, smoothed)
        _save_index(self.settings.index_path, item_ids)

        logger.info("Saved vectors to %s", self.settings.vectors_path)
        logger.info("Saved index to %s", self.settings.index_path)

        return {
            "items": int(len(item_ids)),
            "vectors_shape": list(smoothed.shape),
            "index_path": str(self.settings.index_path),
            "vectors_path": str(self.settings.vectors_path),
        }

    def evaluate(self) -> EvalMetrics:
        if not self.settings.vectors_path.exists() or not self.settings.index_path.exists():
            self.train()

        vectors = np.load(self.settings.vectors_path).astype(np.float32)
        item_ids = _load_index(self.settings.index_path)
        id_to_pos = {int(i): idx for idx, i in enumerate(item_ids.tolist())}

        seqs = _simulate_sequences(
            item_ids,
            n_users=max(120, len(item_ids) // 2),
            seed=self.settings.random_seed + 7,
        )

        hits5 = 0
        hits10 = 0
        rr_sum = 0.0
        n = 0

        for seq in seqs:
            if len(seq) < 2:
                continue
            history = seq[:-1]
            target = seq[-1]
            hist_pos = [id_to_pos[i] for i in history if i in id_to_pos]
            if not hist_pos or target not in id_to_pos:
                continue

            user_vec = vectors[np.array(hist_pos)].mean(axis=0)
            user_vec /= np.linalg.norm(user_vec) + 1e-12
            scores = vectors @ user_vec

            for h in hist_pos:
                scores[h] = -np.inf

            ranked = _topk_indices(scores, k=10)
            target_pos = id_to_pos[target]
            n += 1

            if target_pos in ranked[:5]:
                hits5 += 1
            if target_pos in ranked:
                hits10 += 1
                rank = int(np.where(ranked == target_pos)[0][0]) + 1
                rr_sum += 1.0 / rank

        if n == 0:
            return EvalMetrics(0.0, 0.0, 0.0)

        return EvalMetrics(
            recall_at_5=hits5 / n,
            recall_at_10=hits10 / n,
            mrr_at_10=rr_sum / n,
        )

    def list_expected_files(self) -> list[Path]:
        return [self.settings.catalog_path, self.settings.index_path, self.settings.vectors_path]


def run_full_training() -> None:
    settings = Settings()
    pipeline = RecommenderPipeline(settings)
    result = pipeline.train()
    print(json.dumps({"status": "ok", "train": result}, indent=2))


def run_full_evaluation() -> None:
    settings = Settings()
    pipeline = RecommenderPipeline(settings)
    metrics = pipeline.evaluate()
    print(
        json.dumps(
            {
                "status": "ok",
                "metrics": {
                    "recall_at_5": round(metrics.recall_at_5, 4),
                    "recall_at_10": round(metrics.recall_at_10, 4),
                    "mrr_at_10": round(metrics.mrr_at_10, 4),
                },
            },
            indent=2,
        )
    )


def recommend_for_history(history: list[str], top_k: int = 5) -> list[dict[str, object]]:
    settings = Settings()
    pipeline = RecommenderPipeline(settings)

    if not settings.vectors_path.exists() or not settings.index_path.exists():
        pipeline.train()

    df = pipeline.load_catalog()
    vectors = np.load(settings.vectors_path).astype(np.float32)
    item_ids = _load_index(settings.index_path)
    id_to_pos = {int(i): idx for idx, i in enumerate(item_ids.tolist())}

    matched_positions: list[int] = []
    for query in history:
        q = query.strip().lower()
        if not q:
            continue
        matches = df[df["category_name"].str.lower().str.contains(q, regex=False)]
        if not matches.empty:
            item_id = int(matches.iloc[0]["id"])
            pos = id_to_pos.get(item_id)
            if pos is not None:
                matched_positions.append(pos)

    if not matched_positions:
        matched_positions = [0]

    user_vec = vectors[np.array(matched_positions)].mean(axis=0)
    user_vec /= np.linalg.norm(user_vec) + 1e-12
    scores = vectors @ user_vec

    for pos in matched_positions:
        scores[pos] = -np.inf

    best = _topk_indices(scores, k=top_k)
    out: list[dict[str, object]] = []
    for rank, pos in enumerate(best, start=1):
        item_id = int(item_ids[pos])
        row = df[df["id"] == item_id].iloc[0]
        out.append(
            {
                "rank": rank,
                "item_id": item_id,
                "category_name": str(row["category_name"]),
                "score": float(scores[pos]),
            }
        )
    return out
