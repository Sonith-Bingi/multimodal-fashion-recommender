from __future__ import annotations

import json
import logging
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class RecommenderPipeline:
    """Production pipeline extracted into source modules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_artifacts(self) -> ArtifactStatus:
        return ArtifactStatus(
            catalog=self.settings.catalog_path.exists() or self.settings.catalog_cache_path.exists(),
            index=self.settings.index_path.exists(),
            vectors=self.settings.vectors_path.exists(),
            meta=self.settings.meta_path.exists(),
            reviews=self.settings.review_path.exists(),
        )

    def _load_fallback_catalog(self) -> pd.DataFrame:
        df = pd.read_csv(self.settings.catalog_path)
        if {"id", "category_name"}.issubset(df.columns):
            out = pd.DataFrame(
                {
                    "asin": df["id"].astype(str),
                    "title": df["category_name"].astype(str),
                    "categories": df["category_name"].astype(str),
                    "price": 0.0,
                    "imgUrl": "",
                }
            )
            out["text"] = out["title"].str.strip() + " [" + out["categories"].str.strip() + "]"
            return out
        raise ValueError("Fallback catalog must contain id and category_name")

    def prepare_data(self) -> tuple[pd.DataFrame, dict[str, list[tuple[int, str]]], dict[str, list[tuple[int, str]]]]:
        ensure_dir(self.settings.artifacts_dir)

        if self.settings.catalog_cache_path.exists():
            fashion_products = pd.read_csv(self.settings.catalog_cache_path)
            if "asin" in fashion_products.columns and "text" in fashion_products.columns:
                logger.info("Loaded cached dense catalog: %s", self.settings.catalog_cache_path)
                return fashion_products, {}, {}

        meta_path = self.settings.meta_path
        review_path = self.settings.review_path

        if not meta_path.exists() or not review_path.exists():
            try:
                _download_from_hub("raw/meta_categories/meta_Amazon_Fashion.jsonl", meta_path)
                _download_from_hub("raw/review_categories/Amazon_Fashion.jsonl", review_path)
            except Exception as exc:
                logger.warning("Falling back to local catalog only: %s", exc)
                return self._load_fallback_catalog(), {}, {}

        logger.info("Loading product metadata from %s", meta_path)
        meta_records: list[dict[str, Any]] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                imgs = d.get("images") or []
                img_url = imgs[0].get("large", "") if isinstance(imgs, list) and imgs else ""
                meta_records.append(
                    {
                        "asin": d.get("parent_asin", ""),
                        "title": d.get("title", ""),
                        "categories": " > ".join(d.get("categories") or []),
                        "price": d.get("price"),
                        "imgUrl": img_url,
                        "store": d.get("store", ""),
                    }
                )

        fashion_products = pd.DataFrame(meta_records)
        fashion_products = fashion_products[fashion_products["asin"].astype(str).str.strip() != ""]
        fashion_products = fashion_products.dropna(subset=["asin", "title"])
        fashion_products = fashion_products[fashion_products["title"].astype(str).str.strip() != ""]
        fashion_products = fashion_products.drop_duplicates(subset="asin").reset_index(drop=True)
        fashion_products["price"] = pd.to_numeric(fashion_products["price"], errors="coerce").fillna(0.0)
        fashion_products["categories"] = fashion_products["categories"].fillna("Amazon Fashion")
        fashion_products["text"] = (
            fashion_products["title"].astype(str).str.strip()
            + " ["
            + fashion_products["categories"].astype(str).str.strip()
            + "]"
        )

        valid_asins = set(fashion_products["asin"].tolist())
        user_events: dict[str, list[tuple[int, str]]] = {}
        with open(review_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                uid = str(d.get("user_id", ""))
                asin = str(d.get("parent_asin", ""))
                ts = int(d.get("timestamp", 0))
                if uid and asin in valid_asins:
                    user_events.setdefault(uid, []).append((ts, asin))

        raw_backup = {uid: list(events) for uid, events in user_events.items()}

        user_events, dense_catalog_asins, rounds, dense_events = _filter_k_core(
            user_events, k=self.settings.dense_k
        )
        fashion_products = fashion_products[
            fashion_products["asin"].isin(dense_catalog_asins)
        ].reset_index(drop=True)
        fashion_products["text"] = (
            fashion_products["title"].astype(str).str.strip()
            + " ["
            + fashion_products["categories"].astype(str).str.strip()
            + "]"
        )

        logger.info(
            "Applied %s-core in %s rounds | dense users=%s dense catalog=%s dense events=%s",
            self.settings.dense_k,
            rounds,
            len(user_events),
            len(fashion_products),
            dense_events,
        )

        fashion_products.to_csv(self.settings.catalog_cache_path, index=False)
        return fashion_products, user_events, raw_backup

    def _build_item_embeddings(self, fashion_products: pd.DataFrame) -> np.ndarray:
        if self.settings.item_embs_path.exists():
            arr = np.load(self.settings.item_embs_path).astype(np.float32)
            if arr.shape[0] == len(fashion_products):
                return arr

        texts = fashion_products["text"].fillna("").astype(str).tolist()

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            item_embs = model.encode(
                texts,
                batch_size=128,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            item_embs = np.array(item_embs, dtype=np.float32)
        except Exception as exc:
            logger.warning("sentence-transformers unavailable, using fallback text encoder: %s", exc)
            item_embs = np.stack([_token_fallback_embedding(t) for t in texts], axis=0)

        np.save(self.settings.item_embs_path, item_embs)
        return item_embs

    def _save_index(self, vectors: np.ndarray) -> None:
        faiss = _try_import_faiss()
        if faiss is None:
            payload = {"shape": list(vectors.shape)}
            self.settings.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype(np.float32))
        faiss.write_index(index, str(self.settings.index_path))

    def summarize_pipeline(self) -> dict[str, object]:
        status = self.validate_artifacts()
        return {
            "dense_k": self.settings.dense_k,
            "seq_len": self.settings.seq_len,
            "min_seq": self.settings.min_seq,
            "catalog_cache": str(self.settings.catalog_cache_path),
            "item_embeddings": str(self.settings.item_embs_path),
            "artifact_status": status.__dict__,
        }

    def train(self) -> dict[str, object]:
        fashion_products, user_events, raw_backup = self.prepare_data()
        item_embs = self._build_item_embeddings(fashion_products)
        item_embs = _normalize_rows(item_embs.astype(np.float32))

        np.save(self.settings.vectors_path, item_embs)
        self._save_index(item_embs)

        asin_to_idx = {a: i for i, a in enumerate(fashion_products["asin"].astype(str).tolist())}
        train_seqs, val_seqs, val_novel_seqs, sparse_val_seqs = _build_sequences(
            user_events=user_events,
            raw_user_events_backup=raw_backup,
            asin_to_idx=asin_to_idx,
            seq_len=self.settings.seq_len,
            min_seq=self.settings.min_seq,
            n_catalog=len(fashion_products),
        )

        splits = {
            "train": train_seqs,
            "val": val_seqs,
            "val_novel": val_novel_seqs,
            "sparse_val": sparse_val_seqs,
        }
        (self.settings.artifacts_dir / "splits.json").write_text(
            json.dumps(splits), encoding="utf-8"
        )

        return {
            "items": int(len(fashion_products)),
            "vectors_shape": [int(item_embs.shape[0]), int(item_embs.shape[1])],
            "train_sequences": len(train_seqs),
            "val_sequences": len(val_seqs),
            "val_novel_sequences": len(val_novel_seqs),
            "index_path": str(self.settings.index_path),
            "vectors_path": str(self.settings.vectors_path),
        }

    def _encode_user_history(self, vectors: np.ndarray, hist_indices: list[int]) -> np.ndarray:
        if not hist_indices:
            q = vectors.mean(axis=0, keepdims=True)
        else:
            q = vectors[np.array(hist_indices)].mean(axis=0, keepdims=True)
        return _normalize_rows(q.astype(np.float32))

    def _retrieve(self, vectors: np.ndarray, hist_indices: list[int], k: int) -> list[tuple[float, int]]:
        q = self._encode_user_history(vectors, hist_indices)
        sims = (q @ vectors.T).squeeze(0)
        seen = set(hist_indices)
        order = np.argsort(-sims)
        results: list[tuple[float, int]] = []
        for idx in order.tolist():
            if idx in seen:
                continue
            results.append((float(sims[idx]), int(idx)))
            if len(results) >= k:
                break
        return results

    def _run_eval(self, vectors: np.ndarray, samples: list[tuple[list[int], int]], k: int = 10) -> EvalMetrics:
        if not samples:
            return EvalMetrics(0.0, 0.0, 0.0)

        subset = random.sample(samples, min(2000, len(samples)))
        r_vals: list[float] = []
        n_vals: list[float] = []
        m_vals: list[float] = []

        for hist, target in subset:
            results = self._retrieve(vectors, hist, k=k)
            r_vals.append(_recall_at_k(results, target, k))
            n_vals.append(_ndcg_at_k(results, target, k))
            m_vals.append(_mrr_at_k(results, target, k))

        return EvalMetrics(
            recall_at_10=float(sum(r_vals) / max(len(r_vals), 1)),
            ndcg_at_10=float(sum(n_vals) / max(len(n_vals), 1)),
            mrr_at_10=float(sum(m_vals) / max(len(m_vals), 1)),
        )

    def evaluate(self) -> EvalMetrics:
        if not self.settings.vectors_path.exists():
            self.train()

        vectors = np.load(self.settings.vectors_path).astype(np.float32)

        split_path = self.settings.artifacts_dir / "splits.json"
        if split_path.exists():
            splits = json.loads(split_path.read_text(encoding="utf-8"))
            val_seqs = [
                (list(map(int, hist)), int(tgt)) for hist, tgt in splits.get("val", [])
            ]
        else:
            fashion_products, user_events, raw_backup = self.prepare_data()
            asin_to_idx = {a: i for i, a in enumerate(fashion_products["asin"].astype(str).tolist())}
            _, val_seqs, _, _ = _build_sequences(
                user_events=user_events,
                raw_user_events_backup=raw_backup,
                asin_to_idx=asin_to_idx,
                seq_len=self.settings.seq_len,
                min_seq=self.settings.min_seq,
                n_catalog=len(fashion_products),
            )

        return self._run_eval(vectors, val_seqs, k=10)

    def list_expected_files(self) -> list[Path]:
        return [
            self.settings.catalog_path,
            self.settings.catalog_cache_path,
            self.settings.item_embs_path,
            self.settings.index_path,
            self.settings.vectors_path,
        ]


def run_full_training() -> None:
    pipeline = RecommenderPipeline(Settings())
    print(json.dumps({"status": "ok", "train": pipeline.train()}, indent=2))


def run_full_evaluation() -> None:
    pipeline = RecommenderPipeline(Settings())
    m = pipeline.evaluate()
    print(
        json.dumps(
            {
                "status": "ok",
                "metrics": {
                    "recall_at_10": round(m.recall_at_10, 4),
                    "ndcg_at_10": round(m.ndcg_at_10, 4),
                    "mrr_at_10": round(m.mrr_at_10, 4),
                },
            },
            indent=2,
        )
    )


def recommend_for_history(history: list[str], top_k: int = 5) -> list[dict[str, object]]:
    settings = Settings()
    pipeline = RecommenderPipeline(settings)

    if not settings.vectors_path.exists():
        pipeline.train()

    vectors = np.load(settings.vectors_path).astype(np.float32)

    if settings.catalog_cache_path.exists():
        catalog = pd.read_csv(settings.catalog_cache_path)
    else:
        catalog = pipeline._load_fallback_catalog()

    if "title" not in catalog.columns:
        catalog["title"] = catalog.get("category_name", "").astype(str)
    if "categories" not in catalog.columns:
        catalog["categories"] = catalog.get("category_name", "").astype(str)

    matched_indices: list[int] = []
    for query in history:
        q = query.strip()
        if not q:
            continue
        match = catalog[catalog["title"].astype(str).str.contains(q, case=False, na=False)]
        if len(match) == 0:
            match = catalog[catalog["categories"].astype(str).str.contains(q, case=False, na=False)]
        if len(match) > 0:
            matched_indices.append(int(match.index[0]))

    if not matched_indices:
        matched_indices = [0]

    retrieved = pipeline._retrieve(vectors, matched_indices, k=top_k)
    out: list[dict[str, object]] = []
    for rank, (score, idx) in enumerate(retrieved, start=1):
        row = catalog.iloc[idx]
        out.append(
            {
                "rank": rank,
                "item_index": idx,
                "title": str(row.get("title", "")),
                "categories": str(row.get("categories", "")),
                "score": score,
            }
        )
    return out
