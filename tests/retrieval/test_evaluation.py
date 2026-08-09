from __future__ import annotations

import pytest

from swarmbrain.retrieval.evaluation import (
    RankingCase,
    ann_recall_at_k,
    evaluate_bundle,
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


def test_precision_at_k_ignores_lanes_that_abstained_on_a_case() -> None:
    """An empty ranking is an abstention, scored by the no-answer columns.

    Counting it as zero precision would punish a lane twice for the same
    behaviour and make a silent lane look worse than a noisy one.
    """

    metrics = evaluate_rankings(
        (
            RankingCase("returned", frozenset({"a"}), ("a", "x", "y", "z")),
            RankingCase("abstained", frozenset({"b"}), ()),
        ),
        k=4,
    )

    assert metrics.precision_at_k == pytest.approx(0.25)
    assert metrics.recall_at_k == pytest.approx(0.5)


def test_bundle_precision_rises_as_the_relevance_floor_drops_noise() -> None:
    cases = (
        RankingCase("q", frozenset({"a"}), ("a", "noise-1", "noise-2")),
        RankingCase("no-answer", frozenset(), ("noise-3",)),
    )
    relevance = {"q": (0.9, 0.2, 0.1), "no-answer": (0.2,)}

    wide = evaluate_bundle(cases, relevance, k=10, floor=0.0)
    tight = evaluate_bundle(cases, relevance, k=10, floor=0.5)

    assert wide.mean_bundle_size == pytest.approx(3.0)
    assert wide.precision == pytest.approx(1 / 3)
    assert wide.abstained_cases == 0
    # The floor drops both decoys and the whole no-answer bundle.
    assert tight.mean_bundle_size == pytest.approx(1.0)
    assert tight.precision == pytest.approx(1.0)
    assert tight.recall == pytest.approx(1.0)
    assert tight.abstained_cases == 1
    assert tight.no_answer_precision == pytest.approx(1.0)
    assert tight.no_answer_recall == pytest.approx(1.0)


def test_abstaining_on_an_answerable_case_scores_zero_precision() -> None:
    """Giving up on a question that had an answer is a miss, not an exemption."""

    cases = (RankingCase("q", frozenset({"a"}), ("a",)),)
    result = evaluate_bundle(cases, {"q": (0.1,)}, k=10, floor=0.5)

    assert result.mean_bundle_size == pytest.approx(0.0)
    assert result.precision == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)
    assert result.no_answer_precision == pytest.approx(0.0)


def test_a_case_without_recorded_relevance_keeps_its_full_ranking() -> None:
    cases = (RankingCase("q", frozenset({"a"}), ("a", "b")),)
    result = evaluate_bundle(cases, {}, k=10, floor=0.9)

    assert result.mean_bundle_size == pytest.approx(2.0)
    assert result.abstained_cases == 0
