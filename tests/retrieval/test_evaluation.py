from __future__ import annotations

import pytest

from swarmbrain.retrieval.evaluation import (
    RankingCase,
    ann_recall_at_k,
    evaluate_lanes,
    evaluate_rankings,
)


def test_ranking_metrics_cover_recall_rank_gain_and_abstention() -> None:
    metrics = evaluate_rankings(
        (
            RankingCase("multi", frozenset({"a", "b"}), ("a", "x", "b")),
            RankingCase("single", frozenset({"c"}), ("x", "c")),
            RankingCase("no-answer", frozenset(), ()),
        ),
        k=3,
    )

    assert metrics.cases == 3
    assert metrics.answerable_cases == 2
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.mrr_at_k == pytest.approx(0.75)
    assert metrics.ndcg_at_k == pytest.approx(0.7753252713598225)
    assert metrics.no_answer_precision == pytest.approx(1.0)
    assert metrics.no_answer_recall == pytest.approx(1.0)


def test_lane_ablation_and_ann_oracle_metrics_are_deterministic() -> None:
    metrics = evaluate_lanes(
        {"semantic": ("dense-hit",), "literal": ("exact-hit",)},
        {
            "dense": {"semantic": ("dense-hit",), "literal": ()},
            "hybrid": {
                "semantic": ("dense-hit",),
                "literal": ("exact-hit",),
            },
        },
        k=1,
    )

    assert metrics["dense"].recall_at_k == pytest.approx(0.5)
    assert metrics["hybrid"].recall_at_k == pytest.approx(1.0)
    assert ann_recall_at_k(("a", "x", "c"), ("a", "b", "c"), k=3) == pytest.approx(2 / 3)
