from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    """Runtime configuration loaded from env vars and sane defaults."""

    model_config = SettingsConfigDict(env_prefix="RECO_", env_file=".env", extra="ignore")

    drive_dir: Path = Field(default=DEFAULT_DATA_DIR)
    dense_k: int = Field(default=3, ge=2)
    seq_len: int = Field(default=15, ge=1)
    min_seq: int = Field(default=3, ge=2)
    batch_size: int = Field(default=512, ge=1)
    random_seed: int = 42

    @property
    def catalog_path(self) -> Path:
        return self.drive_dir / "amazon_categories.csv"

    @property
    def meta_path(self) -> Path:
        return self.drive_dir / "meta_Amazon_Fashion.jsonl"

    @property
    def review_path(self) -> Path:
        return self.drive_dir / "Amazon_Fashion.jsonl"

    @property
    def catalog_cache_path(self) -> Path:
        return self.drive_dir / f"fashion_products_kcore{self.dense_k}.csv"

    @property
    def dense_events_cache_path(self) -> Path:
        return self.drive_dir / f"dense_events_kcore{self.dense_k}.jsonl"

    @property
    def item_embs_path(self) -> Path:
        # patrickjohncyh/fashion-clip: once a real eval-methodology bug in
        # this project's own testing scripts was fixed (see MODEL_CARD.md
        # "Resolution: the gap was a testing bug, not a model bug"), this
        # fashion-tuned CLIP checkpoint measured a genuine +7.7% recall@10
        # improvement over all-mpnet-base-v2 on this pipeline's downstream
        # task -- narrowly ahead of Marqo/marqo-fashionCLIP too, with a
        # simpler, more reliable integration (standard transformers API,
        # no open_clip/trust_remote_code fragility). Filename carries the
        # encoder name so a future switch can't silently load stale,
        # mismatched cached embeddings.
        return self.drive_dir / f"item_embs_fashionclip_kcore{self.dense_k}.npy"

    @property
    def artifacts_dir(self) -> Path:
        return self.drive_dir / "artifacts"

    @property
    def index_path(self) -> Path:
        return self.drive_dir / "item_index_v11.faiss"

    @property
    def vectors_path(self) -> Path:
        return self.drive_dir / "item_tower_vecs_v11.npy"

    @property
    def popular_items_path(self) -> Path:
        return self.drive_dir / f"popular_items_kcore{self.dense_k}.json"
