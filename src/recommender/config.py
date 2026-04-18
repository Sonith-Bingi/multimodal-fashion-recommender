from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from env vars and sane defaults."""

    model_config = SettingsConfigDict(env_prefix="RECO_", env_file=".env", extra="ignore")

    drive_dir: Path = Field(default=Path("/Users/Patron/Desktop/Recommender"))
    dense_k: int = Field(default=3, ge=2)
    seq_len: int = Field(default=15, ge=1)
    min_seq: int = Field(default=3, ge=2)
    batch_size: int = Field(default=512, ge=1)
    random_seed: int = 42

    @property
    def catalog_path(self) -> Path:
        return self.drive_dir / "amazon_categories.csv"

    @property
    def artifacts_dir(self) -> Path:
        return self.drive_dir / "artifacts"

    @property
    def index_path(self) -> Path:
        return self.drive_dir / "item_index_v11.faiss"

    @property
    def vectors_path(self) -> Path:
        return self.drive_dir / "item_tower_vecs_v11.npy"
