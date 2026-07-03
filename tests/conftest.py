from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommender.config import Settings

N_ITEMS = 6
N_USERS = 6


def _write_synthetic_catalog(tmp_path: Path, settings: Settings) -> None:
    meta_lines = []
    for i in range(N_ITEMS):
        meta_lines.append(
            json.dumps(
                {
                    "parent_asin": f"A{i}",
                    "title": f"Synthetic Product {i}",
                    "categories": ["Fashion", "Test"],
                    "price": 10.0 + i,
                    "images": [{"large": f"https://example.com/{i}.jpg"}],
                    "store": "TestStore",
                }
            )
        )
    settings.meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    review_lines = []
    ts = 1_700_000_000
    for u in range(N_USERS):
        for i in range(N_ITEMS):
            review_lines.append(
                json.dumps({"user_id": f"U{u}", "parent_asin": f"A{i}", "timestamp": ts})
            )
            ts += 1
    settings.review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_settings(tmp_path: Path) -> Settings:
    """Settings pointed at a small, fully offline synthetic dataset.

    Every user interacts with every item, so k-core filtering (default k=3)
    and min_seq keep the whole synthetic catalog dense.
    """
    settings = Settings(drive_dir=tmp_path)
    _write_synthetic_catalog(tmp_path, settings)
    return settings
