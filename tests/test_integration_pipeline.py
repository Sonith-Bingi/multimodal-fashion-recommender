from __future__ import annotations

from recommender.config import Settings
from recommender.pipeline import RecommenderPipeline


def test_train_and_evaluate_smoke() -> None:
    pipeline = RecommenderPipeline(Settings())

    train_result = pipeline.train()
    assert train_result["items"] > 0
    assert train_result["vectors_shape"][1] > 0

    metrics = pipeline.evaluate()
    assert 0.0 <= metrics.recall_at_10 <= 1.0
    assert 0.0 <= metrics.ndcg_at_10 <= 1.0
    assert 0.0 <= metrics.mrr_at_10 <= 1.0


def test_evaluation_is_stable_for_fixed_artifacts() -> None:
    pipeline = RecommenderPipeline(Settings())
    pipeline.train()

    m1 = pipeline.evaluate()
    m2 = pipeline.evaluate()

    assert abs(m1.recall_at_10 - m2.recall_at_10) < 1e-9
    assert abs(m1.ndcg_at_10 - m2.ndcg_at_10) < 1e-9
    assert abs(m1.mrr_at_10 - m2.mrr_at_10) < 1e-9
