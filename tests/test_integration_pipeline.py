from __future__ import annotations

from recommender.config import Settings
from recommender.pipeline import RecommenderPipeline, recommend_for_history


def test_train_and_evaluate_smoke(synthetic_settings: Settings) -> None:
    pipeline = RecommenderPipeline(synthetic_settings)

    train_result = pipeline.train()
    assert train_result["items"] > 0
    assert train_result["vectors_shape"][1] > 0

    metrics = pipeline.evaluate()
    assert 0.0 <= metrics.recall_at_10 <= 1.0
    assert 0.0 <= metrics.ndcg_at_10 <= 1.0
    assert 0.0 <= metrics.mrr_at_10 <= 1.0


def test_evaluation_is_stable_for_fixed_artifacts(synthetic_settings: Settings) -> None:
    pipeline = RecommenderPipeline(synthetic_settings)
    pipeline.train()

    m1 = pipeline.evaluate()
    m2 = pipeline.evaluate()

    assert abs(m1.recall_at_10 - m2.recall_at_10) < 1e-9
    assert abs(m1.ndcg_at_10 - m2.ndcg_at_10) < 1e-9
    assert abs(m1.mrr_at_10 - m2.mrr_at_10) < 1e-9


def test_cold_start_user_gets_popularity_fallback(synthetic_settings: Settings) -> None:
    """A user with no history (or history that matches nothing in the
    catalog) has no sequence to feed the user tower and nothing to search
    "similar to" -- it must fall back to popularity, not an arbitrary
    catalog row."""
    pipeline = RecommenderPipeline(synthetic_settings)
    pipeline.train()

    assert synthetic_settings.popular_items_path.exists()

    recs_no_history = recommend_for_history([], top_k=3, pipeline=pipeline)
    recs_unmatched = recommend_for_history(["totally-unrecognizable-query-xyz"], top_k=3, pipeline=pipeline)

    assert len(recs_no_history) > 0
    assert len(recs_unmatched) > 0
    assert [r["item_index"] for r in recs_no_history] == [r["item_index"] for r in recs_unmatched]


def test_evaluate_by_item_warmth_slices_metrics(synthetic_settings: Settings) -> None:
    pipeline = RecommenderPipeline(synthetic_settings)
    pipeline.train()

    result = pipeline.evaluate_by_item_warmth()
    assert set(result.keys()) == {"overall", "warm", "cold", "n_warm", "n_cold"}
    assert result["n_warm"] + result["n_cold"] > 0
    for key in ("overall", "warm", "cold"):
        m = result[key]
        assert 0.0 <= m.recall_at_10 <= 1.0
        assert 0.0 <= m.ndcg_at_10 <= 1.0
        assert 0.0 <= m.mrr_at_10 <= 1.0
