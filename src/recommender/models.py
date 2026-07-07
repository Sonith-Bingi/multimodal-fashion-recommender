"""Two-tower model architecture: user tower, item tower, contrastive loss."""

from __future__ import annotations

from typing import Any

IMG_DIM = 512
TOWER_DIM = 256
DROPOUT = 0.15
MAX_HIST_LEN = 64

# Buffers registered by TwoTowerModel.register_popular_pool() for the training-
# time contrastive loss. Absent from the plain TwoTowerModel() built for
# inference (which never calls register_popular_pool()), so they're expected
# to show up as "unexpected" when loading a training checkpoint at inference.
TRAINING_ONLY_STATE_KEYS = frozenset({"_pop_text_embs", "_pop_img_embs", "_pop_ids"})


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
            pos_idx = torch.arange(length, device=hist_embs.device) % MAX_HIST_LEN
            pos_idx = pos_idx.unsqueeze(0).expand(bsz, length)
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

        def forward(
            self, hist_embs: Any, hist_mask: Any, tgt_pos: Any, hist_pos: Any | None = None
        ) -> Any:
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
