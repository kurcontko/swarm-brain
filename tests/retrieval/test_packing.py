"""Token-budgeted packing and answer-in-context scoring.

Recall@k asks whether the evidence was ranked; these pin the quantity that
survives a context window, which is what a reader actually gets.
"""

from __future__ import annotations

import pytest

from swarmbrain.retrieval import (
    MAX_FACILITY_CANDIDATES,
    PackingFeatures,
    answer_in_context,
    build_packing_features,
    estimate_tokens,
    pack_to_budget,
)


def test_empty_text_costs_nothing_and_any_text_costs_something() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 40) == 10


def test_no_budget_keeps_the_whole_bundle() -> None:
    """The shipped default is unbounded, so packing must be a no-op."""

    packed = pack_to_budget([100, 200, 300], None)
    assert packed.kept_indices == (0, 1, 2)
    assert packed.dropped_indices == ()
    assert packed.used_tokens == 600


def test_greedy_skips_an_oversized_hit_and_keeps_filling() -> None:
    packed = pack_to_budget([10, 500, 5], 20)
    assert packed.kept_indices == (0, 2)
    assert packed.dropped_indices == (1,)
    assert packed.used_tokens == 15


def test_prefix_policy_stops_at_the_first_hit_that_does_not_fit() -> None:
    packed = pack_to_budget([10, 500, 5], 20, policy="prefix")
    assert packed.kept_indices == (0,)
    assert packed.dropped_indices == (1, 2)


def test_a_budget_smaller_than_every_hit_keeps_nothing() -> None:
    packed = pack_to_budget([10, 20], 1)
    assert packed.kept_indices == ()
    assert packed.used_tokens == 0


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="policy"):
        pack_to_budget([1], 10, policy="whatever")


def test_facility_location_selects_complementary_evidence() -> None:
    features = (
        PackingFeatures(
            relevance=0.9,
            query_terms=frozenset({"alpha"}),
            representation_terms=frozenset({"shared", "topic"}),
            diversity_labels=frozenset({"kind:decision"}),
        ),
        PackingFeatures(
            relevance=0.85,
            query_terms=frozenset({"alpha"}),
            representation_terms=frozenset({"shared", "topic"}),
            diversity_labels=frozenset({"kind:decision"}),
        ),
        PackingFeatures(
            relevance=0.8,
            query_terms=frozenset({"beta"}),
            representation_terms=frozenset({"distinct", "evidence"}),
            diversity_labels=frozenset({"kind:warning"}),
        ),
    )

    packed = pack_to_budget(
        [10, 10, 10],
        20,
        policy="facility_location",
        features=features,
        max_items=2,
    )

    # Selection optimizes the set, but delivery preserves retrieval order.
    assert packed.kept_indices == (0, 2)
    assert packed.dropped_indices == (1,)
    assert packed.used_tokens == 20


def test_facility_location_is_deterministic_and_hard_bounded() -> None:
    features = tuple(
        PackingFeatures(
            relevance=0.5,
            representation_terms=frozenset({f"topic-{index}"}),
        )
        for index in range(MAX_FACILITY_CANDIDATES + 2)
    )

    first = pack_to_budget(
        [1] * len(features),
        1_000,
        policy="facility_location",
        features=features,
        max_items=len(features),
    )
    second = pack_to_budget(
        [1] * len(features),
        1_000,
        policy="facility_location",
        features=features,
        max_items=len(features),
    )

    assert first == second
    assert first.kept == MAX_FACILITY_CANDIDATES
    assert first.dropped_indices == (
        MAX_FACILITY_CANDIDATES,
        MAX_FACILITY_CANDIDATES + 1,
    )


def test_facility_location_requires_bounded_features() -> None:
    with pytest.raises(ValueError, match="PackingFeatures"):
        pack_to_budget([1], 1, policy="facility_location")
    with pytest.raises(ValueError, match="per item"):
        pack_to_budget(
            [1, 1],
            1,
            policy="facility_location",
            features=(PackingFeatures(relevance=1.0),),
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        PackingFeatures(relevance=1.1)


def test_feature_projection_is_bounded_and_stable() -> None:
    first = build_packing_features(
        "Alpha shared facts and beta distinct evidence " * 100,
        query_terms=("alpha", "missing"),
        relevance=0.75,
        diversity_labels=("Tag:One", "tag:one", "Kind:Decision"),
    )
    second = build_packing_features(
        "Alpha shared facts and beta distinct evidence " * 100,
        query_terms=("alpha", "missing"),
        relevance=0.75,
        diversity_labels=("Kind:Decision", "Tag:One"),
    )

    assert first == second
    assert first.query_terms == frozenset({"alpha"})
    assert first.diversity_labels == frozenset({"tag:one", "kind:decision"})
    assert len(first.representation_terms) <= 96


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        pack_to_budget([1], -1)


def test_answer_in_context_falls_when_the_budget_evicts_the_gold_hit() -> None:
    """The gold memory is ranked second and large; recall@k cannot see this."""

    cases = [(frozenset({"gold"}), ["noise", "gold"], [100, 100])]

    generous = answer_in_context(cases, budget=None)
    tight = answer_in_context(cases, budget=100)

    assert generous.any_gold == pytest.approx(1.0)
    assert generous.truncated_cases == 0
    assert tight.any_gold == pytest.approx(0.0)
    assert tight.truncated_cases == 1


def test_all_gold_is_stricter_than_any_gold_for_multi_evidence() -> None:
    cases = [(frozenset({"a", "b"}), ["a", "b"], [50, 50])]

    half = answer_in_context(cases, budget=50)
    whole = answer_in_context(cases, budget=100)

    assert half.any_gold == pytest.approx(1.0)
    assert half.all_gold == pytest.approx(0.0)
    assert whole.all_gold == pytest.approx(1.0)


def test_no_answer_cases_are_excluded_from_the_rates() -> None:
    cases = [
        (frozenset({"gold"}), ["gold"], [10]),
        (frozenset(), ["noise"], [10]),
    ]
    result = answer_in_context(cases, budget=None)

    assert result.cases == 2
    assert result.answerable_cases == 1
    assert result.any_gold == pytest.approx(1.0)


def test_a_case_without_recorded_sizes_is_treated_as_untruncated() -> None:
    """Runs recorded before token sizes existed must not read as total loss."""

    cases = [(frozenset({"gold"}), ["gold", "noise"], [])]
    result = answer_in_context(cases, budget=10)

    assert result.any_gold == pytest.approx(1.0)
    assert result.truncated_cases == 0
