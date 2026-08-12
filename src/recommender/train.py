"""Training, evaluation, and recommendation orchestration."""

from __future__ import annotations

import io
import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .data import (
    _build_sequences,
    _download_from_hub,
    _filter_k_core,
    _hash_brand,
    _token_fallback_embedding,
)
from .models import (
    IMG_DIM,
    TEXT_EMB_DIM,
    TRAINING_ONLY_STATE_KEYS,
    _build_torch_model,
    _select_device,
    _try_import_torch,
)
from .retrieval import (
    ArtifactStatus,
    EvalMetrics,
    _diversify_beyond_history,
    _mmr_rerank,
    _mrr_at_k,
    _ndcg_at_k,
    _normalize_rows,
    _recall_at_k,
    _reciprocal_rank_fusion,
    _try_import_faiss,
)
from .utils import ensure_dir

# torch and faiss-cpu each bundle their own OpenMP runtime; loading both in one
# process aborts the interpreter ("OMP: Error #15") unless this is set before
# either is imported. Both are only ever imported lazily (via
# _try_import_torch / _try_import_faiss), so setting it here at module import
# time is early enough.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)

# patrickjohncyh/fashion-clip: a CLIP ViT-B/32 checkpoint domain-adapted on
# fashion image-text pairs. One model for both text and image in a single
# joint space. This project tried mpnet+generic-CLIP, this fashion-tuned
# CLIP, and Marqo/marqo-fashionCLIP; an eval-methodology bug in this
# project's own testing scripts (evaluating with use_hybrid=True instead of
# the real default False) initially made both fashion-tuned encoders look
# like regressions. Once fixed, patrickjohncyh/fashion-clip measured a real
# +7.7% recall@10 improvement over mpnet+openai-clip, narrowly ahead of
# Marqo-fashionCLIP too, with a simpler and more reliable integration
# (standard transformers API; Marqo's requires open_clip and hit a
# trust_remote_code/meta-tensor incompatibility with current transformers).
# See MODEL_CARD.md "Resolution: the gap was a testing bug, not a model
# bug" for the full story and numbers.
_FASHION_CLIP_MODEL_ID = "patrickjohncyh/fashion-clip"


class RecommenderPipeline:
    """Production pipeline extracted into source modules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inference_runtime: dict[str, Any] | None = None
        self._text_encoder: Any | None = None

    def _two_tower_state_path(self) -> Path:
        return self.settings.artifacts_dir / "two_tower_model.pt"

    def _clip_img_emb_path(self, top_n: int) -> Path:
        # "fashionclip": see item_embs_path in config.py for why the
        # encoder name is baked into the cache filename.
        name = f"train_target_img_embs_fashionclip_kcore{self.settings.dense_k}_top{top_n}.npy"
        return self.settings.drive_dir / name

    def validate_artifacts(self) -> ArtifactStatus:
        catalog_exists = (
            self.settings.catalog_path.exists() or self.settings.catalog_cache_path.exists()
        )
        return ArtifactStatus(
            catalog=catalog_exists,
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

    def _load_dense_events_cache(self) -> dict[str, list[tuple[int, str]]]:
        user_events: dict[str, list[tuple[int, str]]] = {}
        with self.settings.dense_events_cache_path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                user_events.setdefault(d["user_id"], []).append((d["ts"], d["asin"]))
        return user_events

    def _save_dense_events_cache(self, user_events: dict[str, list[tuple[int, str]]]) -> None:
        with self.settings.dense_events_cache_path.open("w", encoding="utf-8") as f:
            for uid, events in user_events.items():
                for ts, asin in events:
                    f.write(json.dumps({"user_id": uid, "ts": ts, "asin": asin}) + "\n")

    def prepare_data(
        self,
    ) -> tuple[pd.DataFrame, dict[str, list[tuple[int, str]]], dict[str, list[tuple[int, str]]]]:
        ensure_dir(self.settings.artifacts_dir)

        catalog_cached = self.settings.catalog_cache_path.exists()
        events_cached = self.settings.dense_events_cache_path.exists()
        if catalog_cached and events_cached:
            fashion_products = pd.read_csv(self.settings.catalog_cache_path)
            if "asin" in fashion_products.columns and "text" in fashion_products.columns:
                logger.info("Loaded cached dense catalog: %s", self.settings.catalog_cache_path)
                user_events = self._load_dense_events_cache()
                # sparse_val_seqs (built from users outside the dense set) can't be
                # reconstructed without the full pre-filter interaction log, which
                # this cache intentionally doesn't keep; only that diagnostic split
                # is affected on a cache hit, not train_seqs/val_seqs.
                return fashion_products, user_events, dict(user_events)

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
        with open(meta_path, encoding="utf-8") as f:
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
        fashion_products["price"] = pd.to_numeric(
            fashion_products["price"], errors="coerce"
        ).fillna(0.0)
        fashion_products["categories"] = fashion_products["categories"].fillna("Amazon Fashion")
        fashion_products["text"] = (
            fashion_products["title"].astype(str).str.strip()
            + " ["
            + fashion_products["categories"].astype(str).str.strip()
            + "]"
        )

        valid_asins = set(fashion_products["asin"].tolist())
        user_events: dict[str, list[tuple[int, str]]] = {}
        with open(review_path, encoding="utf-8") as f:
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
        self._save_dense_events_cache(user_events)
        return fashion_products, user_events, raw_backup

    def _load_text_encoder(self) -> Any:
        """Cached loader for the shared fashion-clip model/processor so a
        live API request doesn't reload weights on every call, and so the
        image-embedding builder below can reuse the same loaded model
        instead of instantiating a second copy. Mirrors the
        _inference_runtime caching pattern used for the two-tower model.
        Returns the string "fallback" (rather than None) when the real
        encoder is unavailable, so callers can cache that outcome too."""
        if self._text_encoder is not None:
            return self._text_encoder
        try:
            from transformers import CLIPModel, CLIPProcessor

            torch_ctx = _try_import_torch()
            if torch_ctx is None:
                raise RuntimeError("torch unavailable")
            torch = torch_ctx["torch"]
            device = _select_device(torch)
            model = CLIPModel.from_pretrained(_FASHION_CLIP_MODEL_ID).to(device)
            model.eval()
            processor = CLIPProcessor.from_pretrained(_FASHION_CLIP_MODEL_ID)
            self._text_encoder = {
                "torch": torch,
                "model": model,
                "processor": processor,
                "device": device,
            }
        except Exception as exc:
            logger.warning("fashion-clip unavailable, using fallback text encoder: %s", exc)
            self._text_encoder = "fallback"
        return self._text_encoder

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Single text-encoding path shared by catalog embedding (offline)
        and live query encoding (online) so both live in the same space --
        required for the semantic history-matching fallback in
        recommend_for_history() to be meaningful."""
        encoder = self._load_text_encoder()
        if encoder == "fallback":
            return np.stack(
                [_token_fallback_embedding(t, dim=TEXT_EMB_DIM) for t in texts], axis=0
            )

        torch = encoder["torch"]
        model = encoder["model"]
        processor = encoder["processor"]
        device = encoder["device"]

        all_embs: list[np.ndarray] = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    # CLIP's tokenizer doesn't reliably infer this checkpoint's
                    # 77-token position-embedding limit from truncation=True
                    # alone -- long "title [category]" strings without this
                    # raise "Sequence length must be less than
                    # max_position_embeddings" instead of silently truncating.
                    max_length=77,
                ).to(device)
                feats = model.get_text_features(**inputs)
                # transformers >= 5.x returns BaseModelOutputWithPooling (with
                # .pooler_output holding the projected embedding) instead of a
                # plain tensor -- same API shift the image-embedding code
                # below already guards against.
                if not isinstance(feats, torch.Tensor):
                    feats = feats.pooler_output
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_embs.append(feats.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    def _build_item_embeddings(self, fashion_products: pd.DataFrame) -> np.ndarray:
        if self.settings.item_embs_path.exists():
            arr = np.load(self.settings.item_embs_path).astype(np.float32)
            if arr.shape[0] == len(fashion_products):
                return arr

        texts = fashion_products["text"].fillna("").astype(str).tolist()
        item_embs = self._encode_texts(texts)

        np.save(self.settings.item_embs_path, item_embs)
        return item_embs

    def _build_interaction_dataloaders(
        self,
        train_seqs: list[tuple[list[int], int]],
        val_seqs: list[tuple[list[int], int]],
        item_embs_with_pad: np.ndarray,
        img_embs_with_pad: np.ndarray,
        popular_items: np.ndarray,
        train_time_buckets: list[list[int]] | None = None,
    ) -> tuple[Any, Any, dict[int, int]] | None:
        torch_ctx = _try_import_torch()
        if torch_ctx is None:
            return None

        torch = torch_ctx["torch"]
        Dataset = torch_ctx["Dataset"]
        DataLoader = torch_ctx["DataLoader"]

        pop_pos_lookup = {int(idx): pos for pos, idx in enumerate(popular_items.tolist())}
        pop_train_seqs = list(train_seqs)
        if train_time_buckets is None:
            pop_train_time_buckets = [[0] * len(h) for h, _ in pop_train_seqs]
        else:
            pop_train_time_buckets = list(train_time_buckets)
        pop_val_seqs = [(h, t) for h, t in val_seqs if t in pop_pos_lookup]
        # Validation deliberately gets no real time-gap info (all "unknown"
        # bucket 0), mirroring recommend_for_history()'s serving API, which
        # never has real timestamps either -- see _time_gap_bucket in
        # data.py and MODEL_CARD.md.
        pop_val_time_buckets = [[0] * len(h) for h, _ in pop_val_seqs]

        item_embs_t = torch.tensor(item_embs_with_pad, dtype=torch.float32)
        img_embs_t = torch.tensor(img_embs_with_pad, dtype=torch.float32)

        class InteractionDataset(Dataset):
            def __init__(
                self, seqs: list[tuple[list[int], int]], time_buckets: list[list[int]]
            ) -> None:
                self.seqs = seqs
                self.time_buckets = time_buckets

            def __len__(self) -> int:
                return len(self.seqs)

            def __getitem__(self, index: int) -> tuple[list[int], int, list[int]]:
                hist, target = self.seqs[index]
                return hist, target, self.time_buckets[index]

        def collate_fn(
            batch: list[tuple[list[int], int, list[int]]],
        ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
            hists, targets, time_buckets = zip(*batch, strict=True)
            max_len = max(len(h) for h in hists)
            padded, masks, tb_padded = [], [], []
            for hist, tb in zip(hists, time_buckets, strict=True):
                pad = max_len - len(hist)
                padded.append(list(hist) + [0] * pad)
                masks.append([False] * len(hist) + [True] * pad)
                tb_padded.append(list(tb) + [0] * pad)

            hist_idx = torch.tensor(padded, dtype=torch.long)
            hist_mask = torch.tensor(masks, dtype=torch.bool)
            hist_time_buckets = torch.tensor(tb_padded, dtype=torch.long)
            tgt_idx = torch.tensor(list(targets), dtype=torch.long)

            hist_embs = item_embs_t[hist_idx]
            tgt_embs = item_embs_t[tgt_idx]
            tgt_img_embs = img_embs_t[tgt_idx]
            return (
                hist_embs,
                hist_mask,
                tgt_embs,
                tgt_img_embs,
                tgt_idx,
                hist_idx,
                hist_time_buckets,
            )

        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(self.settings.random_seed)
        train_dl = DataLoader(
            InteractionDataset(pop_train_seqs, pop_train_time_buckets),
            batch_size=self.settings.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            generator=shuffle_generator,
        )
        val_dl = DataLoader(
            InteractionDataset(pop_val_seqs, pop_val_time_buckets),
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
        n_clip_items = min(max_clip_items, len(ranked_targets))
        clip_item_ids = np.array(ranked_targets[:n_clip_items], dtype=np.int64)
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
            # Reuses the same loaded fashion-clip model as _encode_texts()
            # -- one model, one joint embedding space for text and image.
            encoder = self._load_text_encoder()
            if encoder == "fallback":
                logger.warning("fashion-clip unavailable, using text-only items (no image signal)")
                return img_embs_with_pad

            try:
                from concurrent.futures import ThreadPoolExecutor

                import requests
                from PIL import Image

                clip_img_urls = fashion_products.iloc[clip_item_ids]["imgUrl"].fillna("").tolist()

                def fetch_image(args: tuple[int, str]) -> tuple[int, Any | None]:
                    pos, url = args
                    if not url:
                        return pos, None
                    try:
                        headers = {"User-Agent": "Mozilla/5.0"}
                        response = requests.get(url, timeout=6, headers=headers)
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

                torch = encoder["torch"]
                clip_model = encoder["model"]
                clip_processor = encoder["processor"]
                device = encoder["device"]

                clip_img_embs = np.zeros((len(clip_item_ids), IMG_DIM), dtype=np.float32)
                clip_batch = 64
                pos_list = sorted(images.keys())
                for i in range(0, len(pos_list), clip_batch):
                    batch_pos = pos_list[i : i + clip_batch]
                    batch_imgs = [images[p] for p in batch_pos]
                    with torch.no_grad():
                        inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
                        feats = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                        # transformers >= 5.x returns BaseModelOutputWithPooling
                        # (with .pooler_output holding the projected embedding)
                        # instead of a plain tensor.
                        if not isinstance(feats, torch.Tensor):
                            feats = feats.pooler_output
                        feats = torch.nn.functional.normalize(feats, dim=-1).cpu().float().numpy()
                    for j, pos in enumerate(batch_pos):
                        clip_img_embs[pos] = feats[j]

                np.save(str(emb_path), clip_img_embs)
            except Exception as exc:
                logger.warning("CLIP image embedding step failed, using text-only items: %s", exc)
                clip_img_embs = None

        if clip_img_embs is not None:
            img_embs_with_pad[clip_item_ids] = clip_img_embs
        return img_embs_with_pad

    def _build_price_brand_arrays(
        self, fashion_products: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Price and brand/store are already parsed by prepare_data() but
        were never fed into the model -- real signal being discarded.
        Price: log1p then z-scored against this catalog's own distribution
        (encode_item() only ever runs at training/index-build time against a
        fixed, fully-known catalog, never per live request, so there's no
        train/serve skew risk in computing these stats fresh each run).
        Brand/store: feature-hashed into a fixed embedding vocab (see
        BRAND_HASH_BUCKETS in data.py) rather than one row per distinct
        brand string, which would mostly be untrained noise for the many
        brands seen only once or twice."""
        price_raw = pd.to_numeric(
            fashion_products["price"] if "price" in fashion_products.columns else 0.0,
            errors="coerce",
        )
        price_raw = price_raw.fillna(0.0) if hasattr(price_raw, "fillna") else pd.Series(
            [0.0] * len(fashion_products)
        )
        log_price = np.log1p(price_raw.to_numpy(dtype=np.float64)).astype(np.float32)
        mean, std = float(log_price.mean()), float(log_price.std())
        price_arr = (log_price - mean) / std if std > 1e-6 else np.zeros_like(log_price)

        if "store" in fashion_products.columns:
            brands = fashion_products["store"].fillna("").astype(str)
        else:
            brands = pd.Series([""] * len(fashion_products))
        brand_arr = np.array(
            [_hash_brand(b) if b else 0 for b in brands.tolist()], dtype=np.int64
        )

        return price_arr.astype(np.float32), brand_arr

    def _train_two_tower_item_vectors(
        self,
        item_embs: np.ndarray,
        train_seqs: list[tuple[list[int], int]],
        val_seqs: list[tuple[list[int], int]],
        popular_items: np.ndarray,
        logq_alpha: float = 0.0,
        id_branch_dropout: float = 0.0,
        neg_sample_size: int | None = None,
        train_time_buckets: list[list[int]] | None = None,
        time_aware: bool = False,
        use_price_brand: bool = False,
    ) -> np.ndarray:
        torch_ctx = _try_import_torch()
        if torch_ctx is None:
            logger.warning("PyTorch is unavailable; using normalized text embeddings only")
            return _normalize_rows(item_embs.astype(np.float32))

        torch = torch_ctx["torch"]
        nn = torch_ctx["nn"]
        F = torch_ctx["F"]

        # Both default False: time-aware position encoding and price/brand
        # fusion are validated-and-available, not validated-and-adopted --
        # single-run real-data testing showed them flat-to-negative on
        # recall@10 (see MODEL_CARD.md "Text / image encoder ablation" and
        # its follow-up), which given this project's own "single run per
        # condition, not statistically bulletproof" caveat could easily be
        # torch.manual_seed() RNG-cascade noise rather than a real effect --
        # multi-seed variance estimation is needed before trusting either
        # direction, so the safe default keeps the exact validated
        # architecture. Opt in explicitly once that's done.
        if not time_aware:
            train_time_buckets = None

        model_defs = _build_torch_model(
            emb_dim=item_embs.shape[1],
            num_catalog_items=len(item_embs) + 1,
            id_branch_dropout=id_branch_dropout,
            use_time_aware=time_aware,
            use_price_brand=use_price_brand,
        )
        if model_defs is None:
            return _normalize_rows(item_embs.astype(np.float32))
        _item_cls, _user_cls, TwoTowerModel = model_defs

        # Notebook behavior: append one PAD row for text embeddings
        item_embs_with_pad = np.concatenate(
            [item_embs.astype(np.float32), np.zeros((1, item_embs.shape[1]), dtype=np.float32)],
            axis=0,
        )

        fashion_products = (
            pd.read_csv(self.settings.catalog_cache_path)
            if self.settings.catalog_cache_path.exists()
            else self._load_fallback_catalog()
        )

        # Notebook Phase 5.5 logic: CLIP embeddings on capped train-target subset
        img_embs_with_pad = self._build_clip_image_embeddings(
            fashion_products=fashion_products,
            train_seqs=train_seqs,
            n_items=len(item_embs),
        )

        price_arr, brand_arr = self._build_price_brand_arrays(fashion_products)
        price_with_pad = np.concatenate([price_arr, np.zeros(1, dtype=np.float32)], axis=0)
        brand_with_pad = np.concatenate([brand_arr, np.zeros(1, dtype=np.int64)], axis=0)

        loaders = self._build_interaction_dataloaders(
            train_seqs=train_seqs,
            val_seqs=val_seqs,
            item_embs_with_pad=item_embs_with_pad,
            img_embs_with_pad=img_embs_with_pad,
            popular_items=popular_items,
            train_time_buckets=train_time_buckets,
        )
        if loaders is None:
            return _normalize_rows(item_embs.astype(np.float32))
        train_dl, val_dl, pop_pos_lookup = loaders

        device = _select_device(torch)

        # Fixes weight init + dropout masks so runs are comparable/reproducible;
        # DataLoader shuffling is separately seeded in _build_interaction_dataloaders.
        torch.manual_seed(self.settings.random_seed)

        model = TwoTowerModel(logq_alpha=logq_alpha, neg_sample_size=neg_sample_size).to(device)
        popular_idxs = torch.tensor(popular_items, dtype=torch.long)
        item_embs_t = torch.tensor(item_embs_with_pad, dtype=torch.float32)
        img_embs_t = torch.tensor(img_embs_with_pad, dtype=torch.float32)
        price_t = torch.tensor(price_with_pad, dtype=torch.float32)
        brand_t = torch.tensor(brand_with_pad, dtype=torch.long)

        # Empirical training-target frequency (raw counts) per pool item, for
        # the logQ popularity-bias correction (see TwoTowerModel.forward()).
        # Every pool item appeared as a training target at least once by
        # construction (popular_items == train_target_items), so counts are
        # always >= 1; the `.get(..., 1)` fallback is defensive only.
        target_counts = Counter(t for _, t in train_seqs)
        pop_freq = torch.tensor(
            [float(target_counts.get(int(idx), 1)) for idx in popular_items],
            dtype=torch.float32,
        )

        model.register_popular_pool(
            text_embs=item_embs_t[popular_idxs],
            img_embs=img_embs_t[popular_idxs],
            pool_ids=popular_idxs,
            pop_freq=pop_freq,
            price=price_t[popular_idxs],
            brand_id=brand_t[popular_idxs],
        )
        model.to(device)

        # Notebook defaults
        epochs = 40
        early_stop = 10
        warmup = 2
        lr = 5e-4

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        item_to_pop_pos = torch.full((item_embs_t.size(0),), -1, dtype=torch.long, device=device)
        for idx, pos in pop_pos_lookup.items():
            item_to_pop_pos[idx] = pos

        def run_epoch(loader: Any, train: bool = True, freeze_temp: bool = False) -> float:
            model.train(train)
            model.log_temp.requires_grad_(not freeze_temp)
            total_loss, n_items = 0.0, 0

            with torch.set_grad_enabled(train):
                for (
                    hist_embs,
                    hist_mask,
                    _tgt_embs,
                    _tgt_img_embs,
                    tgt_idx,
                    hist_idx,
                    hist_time_buckets,
                ) in loader:
                    hist_embs = hist_embs.to(device)
                    hist_mask = hist_mask.to(device)
                    hist_idx = hist_idx.to(device)
                    tgt_idx = tgt_idx.to(device)
                    hist_time_buckets = hist_time_buckets.to(device)

                    lookup = item_to_pop_pos.to(hist_idx.device)
                    hist_pos = lookup[hist_idx]
                    tgt_pos = lookup[tgt_idx]
                    hist_pos = hist_pos.masked_fill(hist_mask, -1)

                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        loss = model(
                            hist_embs,
                            hist_mask,
                            tgt_pos,
                            hist_pos=hist_pos,
                            hist_time_buckets=hist_time_buckets,
                        )

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
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
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
                price_chunk = price_t[i:end_i].to(device)
                brand_chunk = brand_t[i:end_i].to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    raw_vecs = model.encode_item(
                        text_chunk, img_chunk, id_chunk, price_chunk, brand_chunk
                    )

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
        device = _select_device(torch)
        model = TwoTowerModel().to(device)

        try:
            state = torch.load(state_path, map_location=device)
            # _pop_text_embs/_pop_img_embs/_pop_ids are training-only buffers
            # registered by register_popular_pool() for computing the
            # contrastive loss against the popularity pool. encode_user() and
            # encode_item() (the only methods used at inference time) never
            # touch them, and the inference-time model never calls
            # register_popular_pool(), so they're expected to be absent here.
            # A strict load would reject the checkpoint over that alone.
            result = model.load_state_dict(state, strict=False)
            unexpected = set(result.unexpected_keys) - TRAINING_ONLY_STATE_KEYS
            if result.missing_keys or unexpected:
                logger.warning(
                    "Two-tower state_dict mismatch (missing=%s, unexpected=%s); "
                    "falling back to mean-pooled retrieval",
                    result.missing_keys,
                    sorted(unexpected),
                )
                return None
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

    def train(
        self,
        logq_alpha: float = 0.0,
        id_branch_dropout: float = 0.1,
        neg_sample_size: int | None = None,
        time_aware: bool = False,
        use_price_brand: bool = False,
    ) -> dict[str, object]:
        fashion_products, user_events, raw_backup = self.prepare_data()
        item_embs = self._build_item_embeddings(fashion_products)
        item_embs = _normalize_rows(item_embs.astype(np.float32))

        asins = fashion_products["asin"].astype(str).tolist()
        asin_to_idx = {a: i for i, a in enumerate(asins)}
        train_seqs, val_seqs, val_novel_seqs, sparse_val_seqs, train_time_buckets = (
            _build_sequences(
                user_events=user_events,
                raw_user_events_backup=raw_backup,
                asin_to_idx=asin_to_idx,
                seq_len=self.settings.seq_len,
                min_seq=self.settings.min_seq,
                n_catalog=len(fashion_products),
            )
        )

        train_target_items = np.array(sorted({t for _, t in train_seqs}), dtype=np.int64)
        popular_items = train_target_items

        target_freq = Counter(t for _, t in train_seqs)
        popularity_ranking = [idx for idx, _ in target_freq.most_common()]
        self.settings.popular_items_path.write_text(
            json.dumps(popularity_ranking), encoding="utf-8"
        )

        if len(popular_items) > 0 and len(train_seqs) > 0:
            all_item_vecs = self._train_two_tower_item_vectors(
                item_embs=item_embs,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                popular_items=popular_items,
                logq_alpha=logq_alpha,
                id_branch_dropout=id_branch_dropout,
                neg_sample_size=neg_sample_size,
                train_time_buckets=train_time_buckets,
                time_aware=time_aware,
                use_price_brand=use_price_brand,
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

    def _search_vectors(
        self,
        vectors: np.ndarray,
        q: np.ndarray,
        seen: set[int],
        k: int,
        index_path: Path | None = None,
    ) -> list[tuple[float, int]]:
        faiss = _try_import_faiss()
        if faiss is not None:
            idx_obj = None
            if index_path is not None:
                try:
                    if index_path.exists():
                        idx_obj = faiss.read_index(str(index_path))
                except Exception:
                    idx_obj = None

            if idx_obj is None:
                idx_obj = faiss.IndexFlatIP(vectors.shape[1])
                idx_obj.add(vectors.astype(np.float32))

            extra = len(seen)
            n_search = min(k + extra, idx_obj.ntotal)
            scores, indices = idx_obj.search(q.astype(np.float32), n_search)
            results = [
                (float(s), int(i))
                for s, i in zip(scores[0], indices[0], strict=True)
                # FAISS pads with -1 when a query asks for more results than
                # the index holds; without this filter those sentinel rows
                # look like a real (very negative-index) hit.
                if int(i) not in seen and int(i) >= 0
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

    def _retrieve(
        self,
        vectors: np.ndarray,
        hist_indices: list[int],
        k: int,
        text_embs: np.ndarray | None = None,
        diversify: bool = False,
        item_titles: list[str] | None = None,
    ) -> list[tuple[float, int]]:
        q = self._encode_user_history(vectors, hist_indices)
        seen = set(hist_indices)
        # Over-fetch when diversifying so MMR has real alternatives to trade
        # off against the top hit, not just k items with no slack. 50 wasn't
        # enough in practice: for a tight-cluster query (e.g. Winter Coat +
        # Beanie + Gloves) the top 20+ candidates alone are all near-
        # identical gloves, so MMR had nothing but gloves to choose between.
        # Genuine variety (scarves, hat/scarf sets) shows up around rank
        # 20-50, so the pool needs to reach well past there -- see
        # _mmr_rerank in retrieval.py for the matching lambda_mult tuning.
        fetch_k = max(k * 10, 100) if diversify else k

        if text_embs is None:
            results = self._search_vectors(
                vectors, q, seen, fetch_k, index_path=self.settings.index_path
            )
        else:
            # Hybrid retrieval: blend the trained two-tower sequence signal
            # with a content-based signal (cosine similarity over raw,
            # frozen text embeddings). The two-tower's item-ID embeddings are
            # only as good as how many interactions an item had during
            # training (see MODEL_CARD.md "Known limitations"), so sparse
            # items can retrieve arbitrary-looking neighbors. Text embeddings
            # don't have that dependency -- they're the same regardless of
            # interaction count -- so blending in this second signal gives
            # the system a reliability floor the trained model alone doesn't
            # have. Reciprocal rank fusion combines the two without needing
            # to calibrate their score scales.
            n_candidates = max(fetch_k, 50)
            seq_results = self._search_vectors(
                vectors, q, seen, n_candidates, index_path=self.settings.index_path
            )

            valid_hist = [i for i in hist_indices if 0 <= i < len(text_embs)]
            if valid_hist:
                content_q = _normalize_rows(
                    text_embs[np.array(valid_hist)].mean(axis=0, keepdims=True)
                )
            else:
                content_q = _normalize_rows(text_embs.mean(axis=0, keepdims=True))
            content_results = self._search_vectors(text_embs, content_q, seen, n_candidates)

            fused = _reciprocal_rank_fusion(
                [[idx for _, idx in seq_results], [idx for _, idx in content_results]]
            )
            results = [(score, idx) for score, idx in fused[:fetch_k]]

        if diversify:
            if item_titles is not None:
                # Prefer keyword-novelty-aware diversification (doesn't just
                # avoid near-duplicates among the recs, avoids near-
                # duplicates of what's already in history) -- see
                # _diversify_beyond_history in retrieval.py for why this
                # exists on top of plain MMR.
                history_titles = [
                    item_titles[i] for i in hist_indices if 0 <= i < len(item_titles)
                ]
                return _diversify_beyond_history(vectors, results, k, history_titles, item_titles)
            return _mmr_rerank(vectors, results, k)
        return results[:k]

    def _run_eval(
        self,
        vectors: np.ndarray,
        samples: list[tuple[list[int], int]],
        k: int = 10,
        text_embs: np.ndarray | None = None,
    ) -> EvalMetrics:
        if not samples:
            return EvalMetrics(0.0, 0.0, 0.0)

        rng = random.Random(self.settings.random_seed)
        subset = rng.sample(samples, min(2000, len(samples)))
        r_vals: list[float] = []
        n_vals: list[float] = []
        m_vals: list[float] = []

        for hist, target in subset:
            results = self._retrieve(vectors, hist, k=k, text_embs=text_embs)
            r_vals.append(_recall_at_k(results, target, k))
            n_vals.append(_ndcg_at_k(results, target, k))
            m_vals.append(_mrr_at_k(results, target, k))

        return EvalMetrics(
            recall_at_10=float(sum(r_vals) / max(len(r_vals), 1)),
            ndcg_at_10=float(sum(n_vals) / max(len(n_vals), 1)),
            mrr_at_10=float(sum(m_vals) / max(len(m_vals), 1)),
        )

    def evaluate(self, use_hybrid: bool = False) -> EvalMetrics:
        if not self.settings.vectors_path.exists():
            self.train()

        vectors = np.load(self.settings.vectors_path).astype(np.float32)
        text_embs = None
        if use_hybrid and self.settings.item_embs_path.exists():
            text_embs = _normalize_rows(
                np.load(self.settings.item_embs_path).astype(np.float32)
            )

        split_path = self.settings.artifacts_dir / "splits.json"
        if split_path.exists():
            splits = json.loads(split_path.read_text(encoding="utf-8"))
            # _retrieve() excludes every item already in the user's history from
            # the candidate results, so a target that's a repeat of something the
            # user already interacted with can never be retrieved regardless of
            # model quality. splits["val"] always holds out the literal last
            # event, repeat or not; splits["val_novel"] walks back to the most
            # recent genuinely novel target per user, which is the fair
            # comparison given how _retrieve() works.
            val_seqs = [
                (list(map(int, hist)), int(tgt)) for hist, tgt in splits.get("val_novel", [])
            ]
        else:
            fashion_products, user_events, raw_backup = self.prepare_data()
            asins = fashion_products["asin"].astype(str).tolist()
            asin_to_idx = {a: i for i, a in enumerate(asins)}
            _, _, val_seqs, _, _ = _build_sequences(
                user_events=user_events,
                raw_user_events_backup=raw_backup,
                asin_to_idx=asin_to_idx,
                seq_len=self.settings.seq_len,
                min_seq=self.settings.min_seq,
                n_catalog=len(fashion_products),
            )

        return self._run_eval(vectors, val_seqs, k=10, text_embs=text_embs)

    def evaluate_by_item_warmth(
        self, use_hybrid: bool = False, cold_threshold: int = 2
    ) -> dict[str, Any]:
        """Split val_novel targets into warm/cold by training-target frequency
        and evaluate each slice separately. A target that appeared
        <= cold_threshold times as a training target (including never) had
        little to no gradient signal reach its item-ID embedding, so this
        isolates whether a change (e.g. id_branch_dropout) actually helps the
        items an ID-embedding-only model would struggle with -- the headline
        recall@10 alone is dominated by warm items and can mask a cold-item
        regression or improvement.
        """
        if not self.settings.vectors_path.exists():
            self.train()

        vectors = np.load(self.settings.vectors_path).astype(np.float32)
        text_embs = None
        if use_hybrid and self.settings.item_embs_path.exists():
            text_embs = _normalize_rows(np.load(self.settings.item_embs_path).astype(np.float32))

        split_path = self.settings.artifacts_dir / "splits.json"
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        val_seqs = [
            (list(map(int, hist)), int(tgt)) for hist, tgt in splits.get("val_novel", [])
        ]
        target_freq = Counter(int(t) for _, t in splits.get("train", []))

        warm_seqs = [(h, t) for h, t in val_seqs if target_freq.get(t, 0) > cold_threshold]
        cold_seqs = [(h, t) for h, t in val_seqs if target_freq.get(t, 0) <= cold_threshold]

        return {
            "overall": self._run_eval(vectors, val_seqs, k=10, text_embs=text_embs),
            "warm": self._run_eval(vectors, warm_seqs, k=10, text_embs=text_embs),
            "cold": self._run_eval(vectors, cold_seqs, k=10, text_embs=text_embs),
            "n_warm": len(warm_seqs),
            "n_cold": len(cold_seqs),
        }

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


def recommend_for_history(
    history: list[str], top_k: int = 5, pipeline: RecommenderPipeline | None = None
) -> list[dict[str, object]]:
    if pipeline is None:
        pipeline = RecommenderPipeline(Settings())
    settings = pipeline.settings

    if not settings.vectors_path.exists():
        pipeline.train()

    vectors = np.load(settings.vectors_path).astype(np.float32)
    text_embs = None
    if settings.item_embs_path.exists():
        text_embs = _normalize_rows(np.load(settings.item_embs_path).astype(np.float32))

    if settings.catalog_cache_path.exists():
        catalog = pd.read_csv(settings.catalog_cache_path)
    else:
        catalog = pipeline._load_fallback_catalog()

    if "title" not in catalog.columns:
        catalog["title"] = catalog.get("category_name", "").astype(str)
    if "categories" not in catalog.columns:
        catalog["categories"] = catalog.get("category_name", "").astype(str)
    if "imgUrl" not in catalog.columns:
        catalog["imgUrl"] = ""

    matched_indices: list[int] = []
    unmatched_queries: list[str] = []
    for query in history:
        q = query.strip()
        if not q:
            continue
        match = catalog[catalog["title"].astype(str).str.contains(q, case=False, na=False)]
        if len(match) == 0:
            match = catalog[catalog["categories"].astype(str).str.contains(q, case=False, na=False)]
        if len(match) > 0:
            matched_indices.append(int(match.index[0]))
        else:
            unmatched_queries.append(q)

    # Semantic fallback: substring matching is exact and misses typos or
    # paraphrases ("swim trunks" vs. "Swim Trunk", "sunnies" vs.
    # "Sunglasses"). Embed whatever didn't match and take the nearest
    # catalog item by cosine similarity, but only above a similarity floor --
    # otherwise genuinely unrecognizable input (no fashion signal at all)
    # would always resolve to *some* nearest neighbor and this would never
    # fall through to the popularity fallback below, which is the correct
    # behavior for real user cold-start (see MODEL_CARD.md "User cold-start").
    # 0.65, not something lower like 0.3: CLIP-family text embeddings are
    # anisotropic (Ethayarajh 2019) -- unrelated text pairs already sit
    # around cosine 0.55-0.6 in this space, unlike sentence-transformer
    # models tuned for calibrated semantic-textual-similarity scores. This
    # floor is encoder-specific; measured against
    # data_real/fashion_products_kcore3.csv with patrickjohncyh/fashion-clip
    # specifically: genuine queries ("swim trunks", "sunnies", "a comfy
    # hoodie for winter") scored 0.70-0.80 best-match; nonsense queries
    # scored 0.58-0.59 on a 500-item sample. On the full ~5,015-item catalog
    # nonsense queries can spuriously hit up to 0.75, because a handful of
    # items have degenerate 1-2 word titles ("Fashion", "Casmonal",
    # "4 Pairs" -- malformed source metadata, ~0.2% of the catalog) that
    # embed as generic attractors for almost any query. Excluding those from
    # eligibility as a match *target* (they're still fully retrievable as
    # recommendations, just not valid "this is what you meant" matches for a
    # garbage query) removes the false-positive source at its cause instead
    # of chasing an unreliable threshold around it.
    _SEMANTIC_MATCH_FLOOR = 0.65
    _MIN_TITLE_WORDS_FOR_MATCH_TARGET = 3
    if unmatched_queries and text_embs is not None:
        query_vecs = _normalize_rows(pipeline._encode_texts(unmatched_queries))
        sims = query_vecs @ text_embs.T
        title_words = catalog["title"].astype(str).str.split().str.len()
        eligible = title_words >= _MIN_TITLE_WORDS_FOR_MATCH_TARGET
        sims[:, ~eligible.to_numpy()] = -1.0
        best_idx = sims.argmax(axis=1)
        best_sim = sims.max(axis=1)
        for idx, sim in zip(best_idx.tolist(), best_sim.tolist(), strict=True):
            if sim >= _SEMANTIC_MATCH_FLOOR:
                matched_indices.append(int(idx))

    def _clean_str(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    if not matched_indices:
        # Genuine user cold-start: no history to condition on, so there is no
        # sequence to feed the user tower and nothing to search "similar to" --
        # falling back to an arbitrary catalog row (as this used to do) is not
        # a real recommendation. Standard practice for this case is a
        # popularity/trending fallback until the user accrues real signal.
        popular_ranking: list[int] = []
        if settings.popular_items_path.exists():
            try:
                popular_ranking = json.loads(
                    settings.popular_items_path.read_text(encoding="utf-8")
                )
            except Exception:
                popular_ranking = []
        if not popular_ranking:
            popular_ranking = list(range(len(catalog)))

        out = []
        for rank, idx in enumerate(popular_ranking[:top_k], start=1):
            row = catalog.iloc[idx]
            out.append(
                {
                    "rank": rank,
                    "item_index": int(idx),
                    "title": _clean_str(row.get("title", "")),
                    "categories": _clean_str(row.get("categories", "")),
                    "image_url": _clean_str(row.get("imgUrl", "")),
                    "score": 0.0,
                }
            )
        return out

    retrieved = pipeline._retrieve(
        vectors,
        matched_indices,
        k=top_k,
        text_embs=text_embs,
        diversify=True,
        item_titles=catalog["title"].astype(str).tolist(),
    )
    out: list[dict[str, object]] = []
    for rank, (score, idx) in enumerate(retrieved, start=1):
        row = catalog.iloc[idx]
        out.append(
            {
                "rank": rank,
                "item_index": idx,
                "title": _clean_str(row.get("title", "")),
                "categories": _clean_str(row.get("categories", "")),
                "image_url": _clean_str(row.get("imgUrl", "")),
                "score": score,
            }
        )
    return out
