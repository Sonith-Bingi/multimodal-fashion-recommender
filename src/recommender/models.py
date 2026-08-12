"""Two-tower model architecture: user tower, item tower, contrastive loss."""

from __future__ import annotations

from typing import Any

from .data import BRAND_HASH_BUCKETS, N_TIME_BUCKETS

IMG_DIM = 512
# patrickjohncyh/fashion-clip (CLIP ViT-B/32-based) projects both text and
# image into the same 512-dim space -- TEXT_EMB_DIM and IMG_DIM are equal
# here by construction, not coincidence.
TEXT_EMB_DIM = 512
TOWER_DIM = 256
DROPOUT = 0.15
MAX_HIST_LEN = 64

# Buffers registered by TwoTowerModel.register_popular_pool() for the training-
# time contrastive loss. Absent from the plain TwoTowerModel() built for
# inference (which never calls register_popular_pool()), so they're expected
# to show up as "unexpected" when loading a training checkpoint at inference.
# _pop_log_freq is the old (pre-fix) buffer name, kept here so a checkpoint
# saved before the logQ correction fix still loads without a strict-key error.
TRAINING_ONLY_STATE_KEYS = frozenset(
    {
        "_pop_text_embs",
        "_pop_img_embs",
        "_pop_ids",
        "_pop_log_freq",
        "_pop_freq",
        "_pop_price",
        "_pop_brand",
    }
)


def _try_import_torch() -> Any | None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset

        return {
            "torch": torch,
            "nn": nn,
            "F": F,
            "Dataset": Dataset,
            "DataLoader": DataLoader,
        }
    except Exception:
        return None


def _select_device(torch: Any) -> str:
    """CUDA > MPS (Apple Silicon GPU) > CPU. Note torch.cuda.amp autocast/
    GradScaler are CUDA-only -- callers gate those on `device == "cuda"`, so
    MPS runs in full precision (no AMP), which is still a real speedup over
    CPU from GPU parallelism alone."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_torch_model(
    emb_dim: int,
    num_catalog_items: int,
    id_branch_dropout: float = 0.0,
    use_time_aware: bool = False,
    use_price_brand: bool = False,
) -> tuple[type, type, type] | None:
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
            id_branch_dropout: float = id_branch_dropout,
            use_price_brand: bool = use_price_brand,
        ) -> None:
            super().__init__()
            self.id_emb = nn.Embedding(num_items, id_dim)
            self.id_dropout = nn.Dropout(id_dropout)
            # DropoutNet-style cold-start training (Volkovs et al., NeurIPS 2017):
            # id_dropout above zeros individual dims but still lets the ID
            # branch through on expectation every example. A brand-new item at
            # serving time has no such thing -- its id_emb row never received a
            # gradient update, so it's pure noise. Zeroing the *entire* ID
            # branch for a random fraction of training examples forces
            # id_proj/fusion to produce a good item vector from text+image
            # alone, matching what actually happens for cold items at serving
            # time. 0.0 (default) is a no-op, preserving old behavior exactly.
            self.id_branch_dropout = id_branch_dropout

            self.text_proj = nn.Linear(text_dim, out_dim, bias=False)
            self.id_proj = nn.Linear(id_dim, out_dim, bias=False)
            nn.init.zeros_(self.id_proj.weight)

            self.img_proj = nn.Linear(img_dim, out_dim, bias=False)
            self.img_gate = nn.Linear(img_dim, 1, bias=True)
            nn.init.zeros_(self.img_gate.weight)
            nn.init.constant_(self.img_gate.bias, -3.0)

            # Price and brand were already parsed by prepare_data() but never
            # reached the model. Deliberately *not* added as extra tokens
            # into fusion_attn below, unlike text/image/id: fusion_attn is a
            # shared self-attention over all tokens, and its internal Q/K/V
            # linear layers have their own (non-zero-initialized) bias
            # terms -- a "zero" token still produces non-zero query/key/value
            # contributions through those biases, so it perturbs the
            # attention pattern over the *other* tokens even before its own
            # projection has learned anything. Zero-initializing id_proj
            # works because id_proj's token already existed in the original
            # 3-token design; adding a *new* token to a shared attention
            # isn't a safe no-op the same way. Instead, price/brand are
            # fused the way img_gate fuses image (a gated additive term on
            # the pooled output, not a new attention token) so the original
            # text/image/id attention path is untouched when they're near-
            # zero, and the two projections are still zero-initialized so
            # their contribution is exactly zero at the very start.
            #
            # use_price_brand defaults to False, and these submodules are
            # only *constructed* when it's True, not just unused when
            # False: torch.manual_seed() determinism depends on the exact
            # sequence of random draws consumed during __init__, and even a
            # module whose weights get immediately zeroed still consumes
            # RNG state (from its default random init, before zeros_()
            # overwrites it) -- constructing it unconditionally would shift
            # every module initialized afterward relative to the pre-
            # existing architecture, breaking bit-for-bit reproducibility
            # of the validated baseline. See MODEL_CARD.md.
            self.use_price_brand = use_price_brand
            if self.use_price_brand:
                self.price_proj = nn.Linear(1, out_dim, bias=True)
                nn.init.zeros_(self.price_proj.weight)
                nn.init.zeros_(self.price_proj.bias)
                self.brand_emb = nn.Embedding(BRAND_HASH_BUCKETS, out_dim)
                nn.init.zeros_(self.brand_emb.weight)
                self.price_brand_gate = nn.Parameter(torch.tensor(-3.0))

            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=8,
                dropout=DROPOUT,
                batch_first=True,
            )
            self.fusion_drop = nn.Dropout(DROPOUT)
            self.fusion_norm = nn.LayerNorm(out_dim)
            self.out_norm = nn.LayerNorm(out_dim)

        def forward(
            self,
            text_emb: Any,
            img_emb: Any,
            item_ids: Any,
            price: Any | None = None,
            brand_id: Any | None = None,
        ) -> Any:
            id_vec = self.id_dropout(self.id_emb(item_ids))
            if self.training and self.id_branch_dropout > 0:
                keep = (
                    torch.rand(id_vec.size(0), 1, device=id_vec.device)
                    >= self.id_branch_dropout
                ).float()
                id_vec = id_vec * keep / (1 - self.id_branch_dropout)
            t = self.text_proj(text_emb)
            c = self.id_proj(id_vec)
            v = self.img_proj(img_emb)

            g = torch.sigmoid(self.img_gate(img_emb))
            v = g * v

            tokens = torch.stack([t, v, c], dim=1)
            attn_out, _ = self.fusion_attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.fusion_norm(tokens + self.fusion_drop(attn_out))
            fused = tokens.mean(dim=1)

            if self.use_price_brand:
                price_brand = self.price_proj(price.unsqueeze(-1)) + self.brand_emb(brand_id)
                pb_gate = torch.sigmoid(self.price_brand_gate)
                fused = fused + pb_gate * price_brand

            return self.out_norm(fused)

    class UserTower(nn.Module):
        def __init__(
            self,
            in_dim: int = emb_dim,
            hidden: int = 512,
            out_dim: int = TOWER_DIM,
            n_layers: int = 2,
            n_heads: int = 8,
            use_time_aware: bool = use_time_aware,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )
            self.pos_emb = nn.Embedding(MAX_HIST_LEN, hidden)
            # Time-interval-aware position signal (TiSASRec-style, Li et al.
            # WSDM'20): pos_emb above only encodes ordinal position (1st,
            # 2nd, ... item back), not how much real time separates
            # consecutive interactions -- a user who bought two items an
            # hour apart and one a year apart look identical to pos_emb.
            # Bucket 0 is "unknown gap" (first position in a history, or any
            # caller with no real timestamps -- see _time_gap_bucket in
            # data.py); forward() below defaults to all-zeros when the
            # caller passes no hist_time_buckets at all, so this module
            # still runs correctly, just without the extra signal.
            # Zero-initialized (matching id_proj/price_proj/brand_emb) so
            # this starts as a true no-op.
            #
            # use_time_aware defaults to False, and time_gap_emb is only
            # *constructed* when it's True, not just unused when False --
            # see the matching use_price_brand comment in ItemTower above
            # for why: torch.manual_seed() determinism depends on the exact
            # sequence of random draws in __init__, and constructing this
            # module unconditionally (even zero-initialized) would still
            # shift every module initialized afterward, breaking bit-for-
            # bit reproducibility of the validated baseline.
            self.use_time_aware = use_time_aware
            if self.use_time_aware:
                self.time_gap_emb = nn.Embedding(N_TIME_BUCKETS, hidden)
                nn.init.zeros_(self.time_gap_emb.weight)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 4,
                dropout=DROPOUT,
                batch_first=True,
                activation="gelu",
            )
            # enable_nested_tensor's fast path relies on an op not yet
            # implemented for MPS (aten::_nested_tensor_from_mask_left_aligned);
            # it's a padded-batch performance optimization, not required for
            # correctness, so disabling it keeps CUDA/CPU/MPS all working
            # identically rather than special-casing the device here.
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=n_layers, enable_nested_tensor=False
            )
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

        def forward(self, hist_embs: Any, mask: Any, hist_time_buckets: Any | None = None) -> Any:
            bsz, length, _ = hist_embs.shape
            x = self.input_proj(hist_embs)
            pos_idx = torch.arange(length, device=hist_embs.device) % MAX_HIST_LEN
            pos_idx = pos_idx.unsqueeze(0).expand(bsz, length)
            x = x + self.pos_emb(pos_idx)
            if self.use_time_aware:
                if hist_time_buckets is None:
                    hist_time_buckets = torch.zeros(
                        bsz, length, dtype=torch.long, device=x.device
                    )
                x = x + self.time_gap_emb(hist_time_buckets)
            x = self.encoder(x, src_key_padding_mask=mask)
            lengths = (~mask).sum(dim=1).clamp(min=1)
            last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(bsz, 1, x.size(-1))
            last_hidden = x.gather(1, last_idx).squeeze(1)
            return self.head(last_hidden)

    class TwoTowerModel(nn.Module):
        def __init__(
            self, logq_alpha: float = 0.0, neg_sample_size: int | None = None
        ) -> None:
            super().__init__()
            self.user_tower = UserTower()
            self.item_tower = ItemTower(id_branch_dropout=id_branch_dropout)
            self.log_temp = nn.Parameter(torch.tensor(0.07).log())
            # Sampling-bias correction (Yi et al. 2019, "Sampling-Bias-Corrected
            # Neural Modeling for Large Corpus Item Recommendations"): items that
            # appear more often as training targets get an inflated logit purely
            # from frequency, independent of true relevance. Subtracting
            # alpha * log(empirical target frequency) at training time corrects
            # for this at the loss level, rather than patching scores after the
            # (already frequency-biased) embeddings are fixed at retrieval time.
            # alpha=0.0 (default) is a no-op, preserving old behavior exactly.
            self.logq_alpha = logq_alpha
            # Mixed Negative Sampling (Yang et al., WWW'20): rather than
            # scoring every batch against the *entire* registered pool (a
            # dense softmax -- an item-tower forward pass over every pool
            # item, every batch), score against a random subset of the pool
            # mixed with the batch's own targets (which double as in-batch
            # negatives for every other row). None (default) preserves the
            # original dense-pool behavior exactly. At this project's
            # catalog scale (~5,000 items) dense pooling is what sampling
            # approximates and is cheap enough to just do directly, so this
            # is an opt-in lever for faster iteration or for scaling to a
            # much larger catalog, not something expected to beat the dense
            # pool on quality here -- see MODEL_CARD.md for the measured
            # trade-off.
            self.neg_sample_size = neg_sample_size

        def register_popular_pool(
            self,
            text_embs: Any,
            img_embs: Any,
            pool_ids: Any,
            pop_freq: Any | None = None,
            price: Any | None = None,
            brand_id: Any | None = None,
        ) -> None:
            self.register_buffer("_pop_text_embs", text_embs.float())
            self.register_buffer("_pop_img_embs", img_embs.float())
            self.register_buffer("_pop_ids", pool_ids.long())
            if pop_freq is None:
                pop_freq = torch.ones(pool_ids.shape[0])
            self.register_buffer("_pop_freq", pop_freq.float())
            if price is None:
                price = torch.zeros(pool_ids.shape[0])
            self.register_buffer("_pop_price", price.float())
            if brand_id is None:
                brand_id = torch.zeros(pool_ids.shape[0], dtype=torch.long)
            self.register_buffer("_pop_brand", brand_id.long())

        def encode_user(
            self, hist_embs: Any, mask: Any, hist_time_buckets: Any | None = None
        ) -> Any:
            return self.user_tower(hist_embs, mask, hist_time_buckets)

        def encode_item(
            self,
            text_emb: Any,
            img_emb: Any,
            item_ids: Any,
            price: Any | None = None,
            brand_id: Any | None = None,
        ) -> Any:
            return self.item_tower(text_emb, img_emb, item_ids, price, brand_id)

        def _sample_candidates(
            self, tgt_pos: Any, hist_pos: Any | None
        ) -> tuple[Any, Any, Any | None]:
            """Build this batch's reduced candidate pool: every unique target
            in the batch (guarantees each row's positive is present) plus
            enough uniformly-sampled pool items to reach neg_sample_size.
            Returns (candidate pool positions, tgt_pos remapped into that
            pool, hist_pos remapped the same way with -1 for anything not in
            the sampled pool -- "not sampled" is equivalent to "not part of
            this batch's softmax denominator", so no separate handling
            needed for those beyond leaving them out of the mask)."""
            pool_size = self._pop_freq.shape[0]
            device = tgt_pos.device
            unique_targets = torch.unique(tgt_pos)
            n_random = max(int(self.neg_sample_size) - unique_targets.numel(), 0)
            random_negs = torch.randint(0, pool_size, (n_random,), device=device)
            cand_pos = torch.unique(torch.cat([unique_targets, random_negs]))

            # cand_pos is sorted (torch.unique guarantees this), so
            # searchsorted gives each global pool position's local column
            # index within the reduced candidate set.
            tgt_pos_local = torch.searchsorted(cand_pos, tgt_pos)

            hist_pos_local = None
            if hist_pos is not None:
                valid = hist_pos >= 0
                safe_hist_pos = hist_pos.clamp(min=0)
                lookup = torch.searchsorted(cand_pos, safe_hist_pos).clamp(
                    max=cand_pos.numel() - 1
                )
                found = cand_pos[lookup] == safe_hist_pos
                hist_pos_local = torch.where(valid & found, lookup, torch.full_like(hist_pos, -1))

            return cand_pos, tgt_pos_local, hist_pos_local

        def forward(
            self,
            hist_embs: Any,
            hist_mask: Any,
            tgt_pos: Any,
            hist_pos: Any | None = None,
            hist_time_buckets: Any | None = None,
        ) -> Any:
            user_vecs = self.encode_user(hist_embs, hist_mask, hist_time_buckets)
            u = F.normalize(user_vecs, dim=-1)
            temp = self.log_temp.exp().clamp(0.02, 0.40)

            if self.neg_sample_size is not None and self.training:
                cand_pos, tgt_pos, hist_pos = self._sample_candidates(tgt_pos, hist_pos)
                pool_text = self._pop_text_embs[cand_pos]
                pool_img = self._pop_img_embs[cand_pos]
                pool_ids = self._pop_ids[cand_pos]
                pool_freq = self._pop_freq[cand_pos]
                pool_price = self._pop_price[cand_pos]
                pool_brand = self._pop_brand[cand_pos]
            else:
                pool_text, pool_img, pool_ids = (
                    self._pop_text_embs,
                    self._pop_img_embs,
                    self._pop_ids,
                )
                pool_freq = self._pop_freq
                pool_price = self._pop_price
                pool_brand = self._pop_brand

            pool_raw = self.encode_item(pool_text, pool_img, pool_ids, pool_price, pool_brand)
            pool_vecs = F.normalize(pool_raw, dim=-1)
            logits = u @ pool_vecs.T / temp
            if self.logq_alpha != 0.0:
                # Corrected logQ correction (Khrylchenko et al., "Correcting
                # the LogQ Correction", RecSys'25, arXiv:2507.09331). The
                # textbook logQ correction subtracts alpha*log(freq) from
                # every logit in a row, including that row's own positive --
                # but the positive isn't Monte-Carlo sampled from the pool
                # the way the negatives conceptually are, it's always
                # present by construction, so penalizing it by its own
                # popularity is unjustified and was double-counting. Fix:
                # (1) never correct a row's own positive logit, (2)
                # normalize each negative's frequency against the pool total
                # *excluding* that row's positive, not the raw unconditional
                # total, since the positive isn't part of what the negatives
                # are competing against. When neg_sample_size is active,
                # pool_freq/pool_size are already the reduced candidate set,
                # so this correction naturally operates on that subset.
                bsz, pool_size = logits.shape
                pos_freq = pool_freq[tgt_pos].unsqueeze(1)  # (bsz, 1)
                denom = (pool_freq.sum() - pos_freq).clamp(min=1.0)
                neg_correction = self.logq_alpha * torch.log(
                    (pool_freq.unsqueeze(0) / denom).clamp(min=1e-12)
                )
                is_own_positive = (
                    torch.arange(pool_size, device=logits.device).unsqueeze(0)
                    == tgt_pos.unsqueeze(1)
                )
                logits = logits - neg_correction.masked_fill(is_own_positive, 0.0)

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
