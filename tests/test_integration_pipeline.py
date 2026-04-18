from __future__ import annotations

import json
import re
from pathlib import Path

from recommender.config import Settings
from recommender.pipeline import RecommenderPipeline


def test_train_and_evaluate_smoke() -> None:
    pipeline = RecommenderPipeline(Settings())

    train_result = pipeline.train()
    assert train_result["items"] > 0
    assert train_result["vectors_shape"][1] == 128

    metrics = pipeline.evaluate()
    assert 0.0 <= metrics.recall_at_5 <= 1.0
    assert 0.0 <= metrics.recall_at_10 <= 1.0
    assert 0.0 <= metrics.mrr_at_10 <= 1.0


def test_recall_at_10_is_loosely_aligned_with_notebook_output() -> None:
    nb_path = Path("recotwotower.ipynb")
    assert nb_path.exists()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = json.dumps(nb)
    values = re.findall(r"Recall@10\s*: ([0-9]*\.?[0-9]+)", text)
    assert values, "No Recall@10 values found in notebook output"

    notebook_recall_at_10 = float(values[-1])
    script_recall_at_10 = RecommenderPipeline(Settings()).evaluate().recall_at_10

    # Loose parity check to ensure the migrated pipeline remains in a similar
    # performance regime as the notebook's reported output.
    assert abs(script_recall_at_10 - notebook_recall_at_10) <= 0.20
