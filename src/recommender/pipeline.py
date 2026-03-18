from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ArtifactStatus:
    notebook: bool
    index: bool
    vectors: bool


class RecommenderPipeline:
    """Small production wrapper around notebook-generated artifacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_artifacts(self) -> ArtifactStatus:
        status = ArtifactStatus(
            notebook=self.settings.notebook_path.exists(),
            index=self.settings.index_path.exists(),
            vectors=self.settings.vectors_path.exists(),
        )
        logger.info("Artifact status | notebook=%s index=%s vectors=%s", status.notebook, status.index, status.vectors)
        return status

    def summarize_notebook(self) -> dict[str, object]:
        """Return a minimal summary (cell count + code/markdown counts)."""
        nb_path = self.settings.notebook_path
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")

        raw = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = raw.get("cells", [])
        code = sum(1 for c in cells if c.get("cell_type") == "code")
        markdown = sum(1 for c in cells if c.get("cell_type") == "markdown")
        return {
            "path": str(nb_path),
            "cells_total": len(cells),
            "cells_code": code,
            "cells_markdown": markdown,
            "dense_k": self.settings.dense_k,
            "seq_len": self.settings.seq_len,
        }

    def list_expected_files(self) -> list[Path]:
        return [
            self.settings.notebook_path,
            self.settings.index_path,
            self.settings.vectors_path,
        ]
