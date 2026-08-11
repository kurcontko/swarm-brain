"""Focused safety tests for the not-yet-integrated adaptive policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from swarmbrain.domain.retrieval import RetrievalSignal
from swarmbrain.retrieval.adaptive_policy import (
    DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG,
    AdaptiveQueryFeatures,
    AdaptiveRetrievalDecision,
    AdaptiveRetrievalPolicy,
    AdaptiveRetrievalPolicyConfig,
    GraphDecision,
    GraphHopEvidence,
    LaneRawEvidence,
    ReviewedLaneCoefficients,
)


def _evidence(
    lane: RetrievalSignal,
    raw_evidence: float | None,
    *,
    available: bool = True,
) -> LaneRawEvidence:
    return LaneRawEvidence(
        lane=lane,
        available=available,
        raw_evidence=raw_evidence,
    )


def _multi_evidence_decision() -> tuple[AdaptiveRetrievalPolicy, AdaptiveRetrievalDecision]:
    policy = AdaptiveRetrievalPolicy()
    decision = policy.decide(
        AdaptiveQueryFeatures(requires_multiple_evidence=True, relational_need=0.8),
        (
            _evidence(RetrievalSignal.LEXICAL, 0.78),
            _evidence(RetrievalSignal.DENSE, 0.74),
            _evidence(RetrievalSignal.GRAPH, 0.82),
        ),
    )
    return policy, decision


def test_simple_sufficient_exact_query_skips_graph() -> None:
    policy = AdaptiveRetrievalPolicy()

    decision = policy.decide(
        AdaptiveQueryFeatures(exact_lookup=True),
        (
            _evidence(RetrievalSignal.EXACT, 1.0),
            _evidence(RetrievalSignal.GRAPH, 1.0),
        ),
    )

    assert decision.abstain is False
    assert decision.best_direct_calibrated_evidence == 1.0
    assert decision.graph_decision is GraphDecision.SKIP
    assert decision.max_graph_depth == 0
    assert decision.per_hop_continuation_thresholds == ()
    assert decision.lane_weights[RetrievalSignal.GRAPH.value] == 0.0
    assert decision.reasons == ("graph_skipped:sufficient_exact",)


def test_multi_evidence_query_may_allow_bounded_graph() -> None:
    _policy, decision = _multi_evidence_decision()

    assert decision.abstain is False
    assert decision.graph_decision is GraphDecision.ALLOW
    assert decision.max_graph_depth == 2
    assert decision.per_hop_continuation_thresholds == (0.60, 0.72)
    assert decision.lane_weights[RetrievalSignal.GRAPH.value] > 0.0
    assert decision.reasons == ("graph_allowed:multi_evidence",)


def test_no_raw_direct_evidence_abstains_and_fails_closed() -> None:
    policy = AdaptiveRetrievalPolicy()

    decision = policy.decide(
        AdaptiveQueryFeatures(requires_multiple_evidence=True),
        (
            _evidence(RetrievalSignal.EXACT, None),
            _evidence(RetrievalSignal.LEXICAL, None),
            _evidence(RetrievalSignal.DENSE, None, available=False),
            _evidence(RetrievalSignal.GRAPH, 0.95),
        ),
    )

    assert decision.abstain is True
    assert decision.abstention_reason == "no_calibrated_direct_evidence"
    assert decision.best_direct_calibrated_evidence is None
    assert decision.graph_decision is GraphDecision.SKIP
    assert decision.lane_weights[RetrievalSignal.GRAPH.value] == 0.0
    assert all(weight == 0.0 for weight in decision.lane_weights.values())
    assert decision.reasons == ("graph_skipped:abstained",)


def test_weak_or_missing_hop_stops_graph_expansion() -> None:
    policy, decision = _multi_evidence_decision()

    assert (
        policy.should_continue_graph(
            decision,
            GraphHopEvidence(depth=1, raw_evidence=0.59),
        )
        is False
    )
    assert (
        policy.should_continue_graph(
            decision,
            GraphHopEvidence(depth=1, raw_evidence=0.60),
        )
        is True
    )
    assert (
        policy.should_continue_graph(
            decision,
            GraphHopEvidence(depth=2, raw_evidence=0.71),
        )
        is False
    )
    assert (
        policy.should_continue_graph(
            decision,
            GraphHopEvidence(depth=2, raw_evidence=None),
        )
        is False
    )
    assert (
        policy.should_continue_graph(
            decision,
            GraphHopEvidence(depth=3, raw_evidence=1.0),
        )
        is False
    )


def test_increasing_raw_evidence_is_monotone_for_score_weight_and_abstention() -> None:
    policy = AdaptiveRetrievalPolicy()
    query = AdaptiveQueryFeatures()

    weak = policy.decide(query, (_evidence(RetrievalSignal.LEXICAL, 0.20),))
    strong = policy.decide(query, (_evidence(RetrievalSignal.LEXICAL, 0.70),))

    assert weak.calibrated_evidence[RetrievalSignal.LEXICAL.value] == 0.20
    assert strong.calibrated_evidence[RetrievalSignal.LEXICAL.value] == 0.70
    assert (
        strong.lane_weights[RetrievalSignal.LEXICAL.value]
        >= weak.lane_weights[RetrievalSignal.LEXICAL.value]
    )
    assert weak.abstain is True
    assert strong.abstain is False


def test_config_round_trips_and_fingerprints_deterministically() -> None:
    payload = DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG.model_dump(mode="json")
    restored = AdaptiveRetrievalPolicyConfig.model_validate(payload)

    assert restored == DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG
    assert (
        AdaptiveRetrievalPolicy(restored).config_fingerprint
        == AdaptiveRetrievalPolicy(DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG).config_fingerprint
    )


def test_config_rejects_nonmonotone_calibration_and_hop_thresholds() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ReviewedLaneCoefficients(
            lane=RetrievalSignal.LEXICAL,
            base_weight=3.0,
            calibration_slope=-0.1,
        )

    payload = DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG.model_dump(mode="json")
    payload["per_hop_continuation_thresholds"] = [0.70, 0.60]
    with pytest.raises(ValidationError, match="deeper graph hops"):
        AdaptiveRetrievalPolicyConfig.model_validate(payload)


def test_serialized_decision_cannot_overstate_direct_evidence() -> None:
    _policy, decision = _multi_evidence_decision()
    payload = decision.model_dump(mode="json")
    payload["best_direct_calibrated_evidence"] = 0.99

    with pytest.raises(ValidationError, match="strongest calibrated direct lane"):
        AdaptiveRetrievalDecision.model_validate(payload)


def test_evidence_contract_rejects_rrf_fields_and_unavailable_claims() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LaneRawEvidence.model_validate(
            {
                "lane": "lexical",
                "available": True,
                "raw_evidence": 0.8,
                "rrf_rank": 1,
            }
        )
    with pytest.raises(ValidationError, match="unavailable lane"):
        LaneRawEvidence(
            lane=RetrievalSignal.DENSE,
            available=False,
            raw_evidence=0.8,
        )


def test_duplicate_lane_evidence_is_rejected() -> None:
    policy = AdaptiveRetrievalPolicy()

    with pytest.raises(ValueError, match="duplicate lane evidence: lexical"):
        policy.decide(
            AdaptiveQueryFeatures(),
            (
                _evidence(RetrievalSignal.LEXICAL, 0.5),
                _evidence(RetrievalSignal.LEXICAL, 0.6),
            ),
        )
