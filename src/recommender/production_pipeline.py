from __future__ import annotations

import json
import logging
import math
import random
import shutil
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd

from .config import Settings
from .utils import ensure_dir

logger = logging.getLogger(__name__)

IMG_DIM = 512
TOWER_DIM = 256
DROPOUT = 0.15
MAX_HIST_LEN = 64


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


def _try_import_torch() -> Any | None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset

        return {
            "torch": torch,
            "nn": nn,
            "F": F,
            "Dataset": Dataset,
            "DataLoader": DataLoader,
        }
    except Exception:
        return None


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


def _build_torch_model(emb_dim: int, num_catalog_items: int) -> tuple[type, type, type] | None:
    torch_ctx = _try_import_torch()
    if torch_ctx is None:
        return None
    torch = torch_ctx["torch"]
    nn = torch_ctx["nn"]
    F = torch_ctx["F"]

    class ItemTower(nn.Module):
        def __init__(
            self,
            text_dim: int = emb_dim,
            img_dim: int = IMG_DIM,
            id_dim: int = 128,
            out_dim: int = TOWER_DIM,
            num_items: int = num_catalog_items,
            id_dropout: float = 0.25,
        ) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(num_items, id_dim)
            self.id_dropout = nn.Dropout(id_dropout)

            self.text_proj = nn.Linear(text_dim, out_dim, bias=False)
            self.id_proj = nn.Linear(id_dim, out_dim, bias=False)
            nn.init.zeros_(self.id_proj.weight)

            self.img_proj = nn.Linear(img_dim, out_dim, bias=False)
            self.img_gate = nn.Linear(img_dim, 1, bias=True)
            nn.init.zeros_(self.img_gate.weight)
            nn.init.constant_(self.img_gate.bias, -3.0)

            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=8,
                dropout=DROPOUT,
                batch_first=True,
            )
            self.fusion_drop = nn.Dropout(DROPOUT)
            self.fusion_norm = nn.LayerNorm(out_dim)
            self.out_norm = nn.LayerNorm(out_dim)

        def forward(self, text_emb: Any, img_emb: Any, item_ids: Any) -> Any:
            id_vec = self.id_dropout(self.id_emb(item_ids))
            t = self.text_proj(text_emb)
            c = self.id_proj(id_vec)
            v = self.img_proj(img_emb)

            g = torch.sigmoid(self.img_gate(img_emb))
            v = g * v

            tokens = torch.stack([t, v, c], dim=1)
            attn_out, _ = self.fusion_attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.fusion_norm(tokens + self.fusion_drop(attn_out))
            fused = tokens.mean(dim=1)
            return self.out_norm(fused)

    class UserTower(nn.Module):
        def __init__(
            self,
            in_dim: int = emb_dim,
            hidden: int = 512,
            out_dim: int = TOWER_DIM,
            n_layers: int = 2,
            n_heads: int = 8,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )
            self.pos_emb = nn.Embedding(MAX_HIST_LEN, hidden)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 4,
                dropout=DROPOUT,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(hidden, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

        def forward(self, hist_embs: Any, mask: Any) -> Any:
            bsz, length, _ = hist_embs.shape
            x = self.input_proj(hist_embs)
            pos_idx = (torch.arange(length, device=hist_embs.device) % MAX_HIST_LEN).unsqueeze(0).expand(
                bsz, length
            )
            x = x + self.pos_emb(pos_idx)
            x = self.encoder(x, src_key_padding_mask=mask)
            lengths = (~mask).sum(dim=1).clamp(min=1)
            last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(bsz, 1, x.size(-1))
            last_hidden = x.gather(1, last_idx).squeeze(1)
            return self.head(last_hidden)

    class TwoTowerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_tower = UserTower()
            self.item_tower = ItemTower()
            self.log_temp = nn.Parameter(torch.tensor(0.07).log())

        def register_popular_pool(self, text_embs: Any, img_embs: Any, pool_ids: Any) -> None:
            self.register_buffer("_pop_text_embs", text_embs.float())
            self.register_buffer("_pop_img_embs", img_embs.float())
            self.register_buffer("_pop_ids", pool_ids.long())

        def encode_user(self, hist_embs: Any, mask: Any) -> Any:
            return self.user_tower(hist_embs, mask)

        def encode_item(self, text_emb: Any, img_emb: Any, item_ids: Any) -> Any:
            return self.item_tower(text_emb, img_emb, item_ids)

        def forward(self, hist_embs: Any, hist_mask: Any, tgt_pos: Any, hist_pos: Any | None = None) -> Any:
            user_vecs = self.encode_user(hist_embs, hist_mask)
            u = F.normalize(user_vecs, dim=-1)
            temp = self.log_temp.exp().clamp(0.02, 0.40)

            pool_raw = self.encode_item(self._pop_text_embs, self._pop_img_embs, self._pop_ids)
            pool_vecs = F.normalize(pool_raw, dim=-1)
            logits = u @ pool_vecs.T / temp

            if hist_pos is not None:
                bsz, length = hist_pos.shape
                b_idx = torch.arange(bsz, device=logits.device).unsqueeze(1).expand(bsz, length)
                valid_mask = hist_pos >= 0
                b_valid = b_idx[valid_mask]
                p_valid = hist_pos[valid_mask]

                tgt_pos_expanded = tgt_pos.unsqueeze(1).expand(bsz, length)
                not_target_mask = (hist_pos != tgt_pos_expanded)[valid_mask]
                logits[b_valid[not_target_mask], p_valid[not_target_mask]] = -1e4

            return F.cross_entropy(logits, tgt_pos)

    return ItemTower, UserTower, TwoTowerModel


class RecommenderPipeline:
    """Production pipeline extracted into source modules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inference_runtime: dict[str, Any] | None = None

    def _two_tower_state_path(self) -> Path:
        return self.settings.artifacts_dir / "two_tower_model.pt"

    def _clip_img_emb_path(self, top_n: int) -> Path:
        return self.settings.drive_dir / f"train_target_img_embs_clip_kcore{self.settings.dense_k}_top{top_n}.npy"

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

    def _build_interaction_dataloaders(
        self,
        train_seqs: list[tuple[list[int], int]],
        val_seqs: list[tuple[list[int], int]],
        item_embs_with_pad: np.ndarray,
        img_embs_with_pad: np.ndarray,
        popular_items: np.ndarray,
    ) -> tuple[Any, Any, dict[int, int]] | None:
        torch_ctx = _try_import_torch()
        if torch_ctx is None:
            return None

        torch = torch_ctx["torch"]
        Dataset = torch_ctx["Dataset"]
        DataLoader = torch_ctx["DataLoader"]

        pop_pos_lookup = {int(idx): pos for pos, idx in enumerate(popular_items.tolist())}
        pop_train_seqs = list(train_seqs)
        pop_val_seqs = [(h, t) for h, t in val_seqs if t in pop_pos_lookup]

        item_embs_t = torch.tensor(item_embs_with_pad, dtype=torch.float32)
        img_embs_t = torch.tensor(img_embs_with_pad, dtype=torch.float32)

        class InteractionDataset(Dataset):
            def __init__(self, seqs: list[tuple[list[int], int]]) -> None:
                self.seqs = seqs

            def __len__(self) -> int:
                return len(self.seqs)

            def __getitem__(self, index: int) -> tuple[list[int], int]:
                return self.seqs[index]

        def collate_fn(batch: list[tuple[list[int], int]]) -> tuple[Any, Any, Any, Any, Any, Any]:
            hists, targets = zip(*batch)
            max_len = max(len(h) for h in hists)
            padded, masks = [], []
            for hist in hists:
                pad = max_len - len(hist)
                padded.append(list(hist) + [0] * pad)
                masks.append([False] * len(hist) + [True] * pad)

            hist_idx = torch.tensor(padded, dtype=torch.long)
            hist_mask = torch.tensor(masks, dtype=torch.bool)
            tgt_idx = torch.tensor(list(targets), dtype=torch.long)

            hist_embs = item_embs_t[hist_idx]
            tgt_embs = item_embs_t[tgt_idx]
            tgt_img_embs = img_embs_t[tgt_idx]
            return hist_embs, hist_mask, tgt_embs, tgt_img_embs, tgt_idx, hist_idx

        train_dl = DataLoader(
            InteractionDataset(pop_train_seqs),
            batch_size=self.settings.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )
        val_dl = DataLoader(
            InteractionDataset(pop_val_seqs),
            batch_size=self.settings.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )
        return train_dl, val_dl, pop_pos_lookup

    def _build_clip_image_embeddings(
        self,
        fashion_products: pd.DataFrame,
        train_seqs: list[tuple[list[int], int]],
        n_items: int,
    ) -> np.ndarray:
        img_embs_with_pad = np.zeros((n_items + 1, IMG_DIM), dtype=np.float32)
        if len(train_seqs) == 0:
            return img_embs_with_pad

        train_target_counts = Counter(t for _, t in train_seqs)
        ranked_targets = [idx for idx, _ in train_target_counts.most_common()]
        max_clip_items = 5000
        clip_item_ids = np.array(ranked_targets[: min(max_clip_items, len(ranked_targets))], dtype=np.int64)
        if len(clip_item_ids) == 0:
            return img_embs_with_pad

        emb_path = self._clip_img_emb_path(len(clip_item_ids))
        clip_img_embs: np.ndarray | None = None

        if emb_path.exists():
            try:
                clip_img_embs = np.load(str(emb_path)).astype(np.float32)
            except Exception:
                clip_img_embs = None

        if clip_img_embs is None:
            try:
                import requests
                from PIL import Image
                from concurrent.futures import ThreadPoolExecutor
                from transformers import CLIPProcessor, CLIPModel

                clip_img_urls = fashion_products.iloc[clip_item_ids]["imgUrl"].fillna("").tolist()

                def fetch_image(args: tuple[int, str]) -> tuple[int, Any | None]:
                    pos, url = args
                    if not url:
                        return pos, None
                    try:
                        response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
                        if response.status_code == 200:
                            return pos, Image.open(io.BytesIO(response.content)).convert("RGB")
                    except Exception:
                        pass
                    return pos, None

                images: dict[int, Any] = {}
                with ThreadPoolExecutor(max_workers=32) as pool:
                    for pos, img in pool.map(fetch_image, enumerate(clip_img_urls)):
                        if img is not None:
                            images[pos] = img

                torch_ctx = _try_import_torch()
                if torch_ctx is None:
                    return img_embs_with_pad
                torch = torch_ctx["torch"]
                F = torch_ctx["F"]
                device = "cuda" if torch.cuda.is_available() else "cpu"

                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                clip_model.eval()

                clip_img_embs = np.zeros((len(clip_item_ids), IMG_DIM), dtype=np.float32)
                clip_batch = 64
                pos_list = sorted(images.keys())
                for i in range(0, len(pos_list), clip_batch):
                    batch_pos = pos_list[i : i + clip_batch]
                    batch_imgs = [images[p] for p in batch_pos]
                    with torch.no_grad():
                        inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
                        feats = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                        if not isinstance(feats, torch.Tensor):
                            feats = feats.pooler_output
                        feats = F.normalize(feats, dim=-1).cpu().float().numpy()
                    for j, pos in enumerate(batch_pos):
                        clip_img_embs[pos] = feats[j]

                np.save(str(emb_path), clip_img_embs)
            except Exception as exc:
                logger.warning("CLIP image embedding step failed, using text-only items: %s", exc)
                clip_img_embs = None

        if clip_img_embs is not None:
            img_embs_with_pad[clip_item_ids] = clip_img_embs
        return img_embs_with_pad

    def _train_two_tower_item_vectors(
        self,
        item_embs: np.ndarray,
        train_seqs: list[tuple[list[int], int]],
        val_seqs: list[tuple[list[int], int]],
        popular_items: np.ndarray,
    ) -> np.ndarray:
        torch_ctx = _try_import_torch()
        if torch_ctx is None:
            logger.warning("PyTorch is unavailable; using normalized text embeddings only")
            return _normalize_rows(item_embs.astype(np.float32))

        torch = torch_ctx["torch"]
        nn = torch_ctx["nn"]
        F = torch_ctx["F"]

        model_defs = _build_torch_model(emb_dim=item_embs.shape[1], num_catalog_items=len(item_embs) + 1)
        if model_defs is None:
            return _normalize_rows(item_embs.astype(np.float32))
        _item_cls, _user_cls, TwoTowerModel = model_defs

        # Notebook behavior: append one PAD row for text embeddings
        item_embs_with_pad = np.concatenate(
            [item_embs.astype(np.float32), np.zeros((1, item_embs.shape[1]), dtype=np.float32)],
            axis=0,
        )

        # Notebook Phase 5.5 logic: CLIP embeddings on capped train-target subset
        img_embs_with_pad = self._build_clip_image_embeddings(
            fashion_products=pd.read_csv(self.settings.catalog_cache_path)
            if self.settings.catalog_cache_path.exists()
            else self._load_fallback_catalog(),
            train_seqs=train_seqs,
            n_items=len(item_embs),
        )

        loaders = self._build_interaction_dataloaders(
            train_seqs=train_seqs,
            val_seqs=val_seqs,
            item_embs_with_pad=item_embs_with_pad,
            img_embs_with_pad=img_embs_with_pad,
            popular_items=popular_items,
        )
        if loaders is None:
            return _normalize_rows(item_embs.astype(np.float32))
        train_dl, val_dl, pop_pos_lookup = loaders

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TwoTowerModel().to(device)
        popular_idxs = torch.tensor(popular_items, dtype=torch.long)
        item_embs_t = torch.tensor(item_embs_with_pad, dtype=torch.float32)
        img_embs_t = torch.tensor(img_embs_with_pad, dtype=torch.float32)
        model.register_popular_pool(
            text_embs=item_embs_t[popular_idxs],
            img_embs=img_embs_t[popular_idxs],
            pool_ids=popular_idxs,
        )
        model.to(device)

        # Notebook defaults
        epochs = 40
        early_stop = 10
        warmup = 2
        lr = 5e-4

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        item_to_pop_pos = torch.full((item_embs_t.size(0),), -1, dtype=torch.long, device=device)
        for idx, pos in pop_pos_lookup.items():
            item_to_pop_pos[idx] = pos

        def run_epoch(loader: Any, train: bool = True, freeze_temp: bool = False) -> float:
            model.train(train)
            model.log_temp.requires_grad_(not freeze_temp)
            total_loss, n_items = 0.0, 0

            with torch.set_grad_enabled(train):
                for hist_embs, hist_mask, _tgt_embs, _tgt_img_embs, tgt_idx, hist_idx in loader:
                    hist_embs = hist_embs.to(device)
                    hist_mask = hist_mask.to(device)
                    hist_idx = hist_idx.to(device)
                    tgt_idx = tgt_idx.to(device)

                    lookup = item_to_pop_pos.to(hist_idx.device)
                    hist_pos = lookup[hist_idx]
                    tgt_pos = lookup[tgt_idx]
                    hist_pos = hist_pos.masked_fill(hist_mask, -1)

                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        loss = model(hist_embs, hist_mask, tgt_pos, hist_pos=hist_pos)

                    if not torch.isfinite(loss):
                        continue

                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    batch_size = int(hist_embs.size(0))
                    total_loss += float(loss.item()) * batch_size
                    n_items += batch_size

            return total_loss / max(n_items, 1)

        ema_alpha = 0.3
        ema_val: float | None = None
        best_ema_val = float("inf")
        best_state: dict[str, Any] | None = None
        patience = 0

        if len(train_seqs) > 0 and len(val_seqs) > 0 and len(popular_items) > 0:
            for epoch in range(1, epochs + 1):
                freeze = epoch <= warmup
                _tr = run_epoch(train_dl, train=True, freeze_temp=freeze)
                vl = run_epoch(val_dl, train=False, freeze_temp=False)
                scheduler.step()
                ema_val = vl if ema_val is None else ema_alpha * vl + (1 - ema_alpha) * ema_val

                if ema_val < best_ema_val:
                    best_ema_val = ema_val
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stop:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        # Phase 7 notebook logic: project all items through item tower and L2-normalize
        chunk = 256
        all_item_vecs: list[Any] = []
        with torch.no_grad():
            for i in range(0, len(item_embs_t), chunk):
                text_chunk = item_embs_t[i : i + chunk].to(device)
                end_i = i + text_chunk.size(0)
                img_chunk = img_embs_t[i:end_i].to(device)
                id_chunk = torch.arange(i, end_i, dtype=torch.long, device=device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    raw_vecs = model.encode_item(text_chunk, img_chunk, id_chunk)

                all_item_vecs.append(F.normalize(raw_vecs.float(), dim=-1).cpu())

        ensure_dir(self.settings.artifacts_dir)
        torch.save(model.state_dict(), self._two_tower_state_path())

        vecs = torch.cat(all_item_vecs, dim=0).numpy().astype(np.float32)
        return vecs[:-1]  # drop PAD row

    def _load_inference_runtime(self) -> dict[str, Any] | None:
        if self._inference_runtime is not None:
            return self._inference_runtime

        state_path = self._two_tower_state_path()
        if not state_path.exists():
            return None

        torch_ctx = _try_import_torch()
        if torch_ctx is None:
            return None

        torch = torch_ctx["torch"]
        F = torch_ctx["F"]

        if not self.settings.item_embs_path.exists():
            return None

        item_embs = np.load(self.settings.item_embs_path).astype(np.float32)
        item_embs_with_pad = np.concatenate(
            [item_embs, np.zeros((1, item_embs.shape[1]), dtype=np.float32)],
            axis=0,
        )

        model_defs = _build_torch_model(
            emb_dim=item_embs.shape[1],
            num_catalog_items=len(item_embs_with_pad),
        )
        if model_defs is None:
            return None

        _item_cls, _user_cls, TwoTowerModel = model_defs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TwoTowerModel().to(device)

        try:
            state = torch.load(state_path, map_location=device)
            model.load_state_dict(state)
        except Exception as exc:
            logger.warning("Could not load two-tower state for sequence-aware retrieval: %s", exc)
            return None

        model.eval()
        runtime = {
            "torch": torch,
            "F": F,
            "device": device,
            "model": model,
            "item_embs_t": torch.tensor(item_embs_with_pad, dtype=torch.float32),
            "num_items": len(item_embs),
        }
        self._inference_runtime = runtime
        return runtime

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

        asin_to_idx = {a: i for i, a in enumerate(fashion_products["asin"].astype(str).tolist())}
        train_seqs, val_seqs, val_novel_seqs, sparse_val_seqs = _build_sequences(
            user_events=user_events,
            raw_user_events_backup=raw_backup,
            asin_to_idx=asin_to_idx,
            seq_len=self.settings.seq_len,
            min_seq=self.settings.min_seq,
            n_catalog=len(fashion_products),
        )

        train_target_items = np.array(sorted({t for _, t in train_seqs}), dtype=np.int64)
        popular_items = train_target_items

        if len(popular_items) > 0 and len(train_seqs) > 0:
            all_item_vecs = self._train_two_tower_item_vectors(
                item_embs=item_embs,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                popular_items=popular_items,
            )
        else:
            all_item_vecs = item_embs

        np.save(self.settings.vectors_path, all_item_vecs.astype(np.float32))
        self._save_index(all_item_vecs.astype(np.float32))

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
            "vectors_shape": [int(all_item_vecs.shape[0]), int(all_item_vecs.shape[1])],
            "train_sequences": len(train_seqs),
            "val_sequences": len(val_seqs),
            "val_novel_sequences": len(val_novel_seqs),
            "index_path": str(self.settings.index_path),
            "vectors_path": str(self.settings.vectors_path),
        }

    def _encode_user_history(self, vectors: np.ndarray, hist_indices: list[int]) -> np.ndarray:
        runtime = self._load_inference_runtime()
        if runtime is not None:
            torch = runtime["torch"]
            F = runtime["F"]
            device = runtime["device"]
            model = runtime["model"]
            item_embs_t = runtime["item_embs_t"]
            num_items = int(runtime["num_items"])

            clean_hist = [int(i) for i in hist_indices if 0 <= int(i) < num_items]
            if clean_hist:
                with torch.no_grad():
                    hist_idx = torch.tensor(clean_hist, dtype=torch.long).unsqueeze(0)
                    hist_embs = item_embs_t[hist_idx].to(device)
                    hist_mask = torch.zeros(1, len(clean_hist), dtype=torch.bool, device=device)
                    raw = model.encode_user(hist_embs, hist_mask)
                    q = F.normalize(raw.float(), dim=-1).cpu().numpy().astype(np.float32)
                    return q

        if not hist_indices:
            q = vectors.mean(axis=0, keepdims=True)
        else:
            q = vectors[np.array(hist_indices)].mean(axis=0, keepdims=True)
        return _normalize_rows(q.astype(np.float32))

    def _retrieve(self, vectors: np.ndarray, hist_indices: list[int], k: int) -> list[tuple[float, int]]:
        q = self._encode_user_history(vectors, hist_indices)
        seen = set(hist_indices)

        faiss = _try_import_faiss()
        if faiss is not None:
            idx_obj = None
            try:
                if self.settings.index_path.exists():
                    idx_obj = faiss.read_index(str(self.settings.index_path))
            except Exception:
                idx_obj = None

            if idx_obj is None:
                idx_obj = faiss.IndexFlatIP(vectors.shape[1])
                idx_obj.add(vectors.astype(np.float32))

            extra = len(hist_indices)
            scores, indices = idx_obj.search(q.astype(np.float32), k + extra)
            results = [
                (float(s), int(i))
                for s, i in zip(scores[0], indices[0])
                if int(i) not in seen
            ]
            return results[:k]

        sims = (q @ vectors.T).squeeze(0)
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

        rng = random.Random(self.settings.random_seed)
        subset = rng.sample(samples, min(2000, len(samples)))
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
