"""Pure policy decisions for evidence-aware retrieval and graph expansion.

This module is deliberately not wired into :mod:`planner` or the retrieval
service yet.  It is a deterministic policy boundary that can be evaluated on
held-out traces before it is allowed to change serving behaviour.

The policy consumes *rank-independent* lane evidence in ``[0, 1]``.  It has no
field for RRF rank or fused score, so neither can accidentally become an
abstention signal.  Direct lanes determine answer sufficiency; graph evidence
only says whether a bounded traversal is promising and can never rescue an
otherwise unsupported answer.

Scope, visibility, tenancy, trust, revocation, and temporal eligibility are
hard retrieval constraints.  They intentionally do not appear here: callers
must enforce them before constructing policy evidence, and must enforce them
again when hydrating any selected memory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator

from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.retrieval import RetrievalSignal

ADAPTIVE_POLICY_CONFIG_VERSION = "adaptive-retrieval-policy-v1"
COEFFICIENT_PROVENANCE = "human_reviewed"

UnitScore = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
PolicyWeight = Annotated[FiniteFloat, Field(ge=0.0, le=100.0)]
PositiveMultiplier = Annotated[FiniteFloat, Field(ge=1.0, le=10.0)]


class GraphDecision(StrEnum):
    """Whether the graph lane may execute for this query."""

    SKIP = "skip"
    ALLOW = "allow"


class AdaptiveQueryFeatures(ContractModel):
    """Small, content-free query features supplied by an upstream classifier.

    ``relational_need`` expresses how strongly the query asks for a relation,
    dependency, history, or chain rather than a single fact.  No feature is
    inferred in this module, which keeps the decision replayable.
    """

    exact_lookup: bool = False
    requires_multiple_evidence: bool = False
    relational_need: UnitScore = 0.0


class LaneRawEvidence(ContractModel):
    """Availability and rank-independent raw evidence for one retrieval lane.

    ``available=True, raw_evidence=None`` is meaningful: the lane ran (or
    could run) but supplied no calibrated input.  The policy gives that lane
    zero weight rather than imputing a score.  An unavailable lane cannot
    claim evidence.
    """

    lane: RetrievalSignal
    available: bool
    raw_evidence: UnitScore | None = None

    @model_validator(mode="after")
    def unavailable_lane_has_no_evidence(self) -> Self:
        if not self.available and self.raw_evidence is not None:
            raise ValueError("an unavailable lane cannot supply raw evidence")
        return self


class GraphHopEvidence(ContractModel):
    """Rank-independent evidence observed after one bounded graph hop."""

    depth: int = Field(ge=1, le=4)
    available: bool = True
    raw_evidence: UnitScore | None = None

    @model_validator(mode="after")
    def unavailable_hop_has_no_evidence(self) -> Self:
        if not self.available and self.raw_evidence is not None:
            raise ValueError("an unavailable graph hop cannot supply raw evidence")
        return self


class ReviewedLaneCoefficients(ContractModel):
    """Human-reviewed monotone calibration and weighting for one lane.

    A future learned policy needs a new config version and provenance value;
    this schema intentionally accepts only reviewed coefficients.  A
    nonnegative slope is the load-bearing monotonicity invariant: increasing a
    lane's raw evidence cannot reduce its calibrated evidence or its own
    pre-gate weight.
    """

    lane: RetrievalSignal
    base_weight: Annotated[FiniteFloat, Field(gt=0.0, le=100.0)]
    calibration_intercept: Annotated[FiniteFloat, Field(ge=-1.0, le=1.0)] = 0.0
    calibration_slope: Annotated[FiniteFloat, Field(ge=0.0, le=4.0)] = 1.0

    def calibrate(self, raw_evidence: float) -> float:
        """Apply the reviewed monotone affine calibration and clamp to [0, 1]."""

        return min(
            1.0,
            max(
                0.0,
                float(self.calibration_intercept) + float(self.calibration_slope) * raw_evidence,
            ),
        )


class AdaptiveRetrievalPolicyConfig(ContractModel):
    """Versioned, JSON-serializable policy configuration."""

    policy_version: Literal[ADAPTIVE_POLICY_CONFIG_VERSION] = ADAPTIVE_POLICY_CONFIG_VERSION
    coefficient_provenance: Literal[COEFFICIENT_PROVENANCE] = COEFFICIENT_PROVENANCE
    lane_coefficients: tuple[ReviewedLaneCoefficients, ...]

    abstention_threshold: UnitScore = 0.35
    exact_sufficiency_threshold: UnitScore = 0.90
    graph_seed_threshold: UnitScore = 0.50
    graph_lane_threshold: UnitScore = 0.45
    graph_relation_threshold: UnitScore = 0.50
    deep_graph_relation_threshold: UnitScore = 0.75

    max_graph_depth: int = Field(default=2, ge=1, le=4)
    per_hop_continuation_thresholds: tuple[UnitScore, ...] = (0.60, 0.72)

    exact_query_weight_multiplier: PositiveMultiplier = 1.25
    multi_evidence_graph_weight_multiplier: PositiveMultiplier = 1.25

    @model_validator(mode="after")
    def safety_invariants_hold(self) -> Self:
        lanes = tuple(item.lane for item in self.lane_coefficients)
        if len(lanes) != len(set(lanes)):
            raise ValueError("lane coefficients must be unique")
        if RetrievalSignal.EXACT not in lanes:
            raise ValueError("the exact lane requires reviewed coefficients")
        if RetrievalSignal.GRAPH not in lanes:
            raise ValueError("the graph lane requires reviewed coefficients")
        if not any(lane is not RetrievalSignal.GRAPH for lane in lanes):
            raise ValueError("at least one direct lane requires reviewed coefficients")

        thresholds = self.per_hop_continuation_thresholds
        if len(thresholds) != self.max_graph_depth:
            raise ValueError("one continuation threshold is required for every graph depth")
        if any(later < earlier for earlier, later in zip(thresholds, thresholds[1:], strict=False)):
            raise ValueError("deeper graph hops cannot have a weaker continuation threshold")
        if self.exact_sufficiency_threshold < self.abstention_threshold:
            raise ValueError("exact sufficiency cannot be weaker than answer sufficiency")
        if self.graph_seed_threshold < self.abstention_threshold:
            raise ValueError("graph traversal requires an answerable direct seed")
        if self.deep_graph_relation_threshold < self.graph_relation_threshold:
            raise ValueError("deep graph traversal cannot be easier to trigger than one hop")
        return self


DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG = AdaptiveRetrievalPolicyConfig(
    lane_coefficients=(
        ReviewedLaneCoefficients(lane=RetrievalSignal.EXACT, base_weight=6.0),
        ReviewedLaneCoefficients(lane=RetrievalSignal.LEXICAL, base_weight=3.0),
        ReviewedLaneCoefficients(lane=RetrievalSignal.FUZZY, base_weight=1.0),
        ReviewedLaneCoefficients(lane=RetrievalSignal.DENSE, base_weight=4.0),
        ReviewedLaneCoefficients(lane=RetrievalSignal.TEMPORAL, base_weight=3.0),
        ReviewedLaneCoefficients(lane=RetrievalSignal.SOURCE, base_weight=2.0),
        ReviewedLaneCoefficients(lane=RetrievalSignal.GRAPH, base_weight=1.5),
    )
)


class AdaptiveRetrievalDecision(ContractModel):
    """Auditable output of :class:`AdaptiveRetrievalPolicy`.

    Every known lane appears in each lane mapping, including unavailable and
    unconfigured lanes.  This makes a zero an explicit decision rather than an
    ambiguous omission.
    """

    policy_version: Literal[ADAPTIVE_POLICY_CONFIG_VERSION]
    config_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    lane_availability: dict[str, bool]
    calibrated_evidence: dict[str, UnitScore | None]
    lane_weights: dict[str, PolicyWeight]

    graph_decision: GraphDecision
    max_graph_depth: int = Field(ge=0, le=4)
    per_hop_continuation_thresholds: tuple[UnitScore, ...] = ()

    best_direct_calibrated_evidence: UnitScore | None = None
    abstention_threshold: UnitScore
    abstain: bool
    abstention_reason: (
        Literal[
            "no_calibrated_direct_evidence",
            "below_calibrated_evidence_threshold",
        ]
        | None
    ) = None
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def decision_is_fail_closed_and_fully_explicit(self) -> Self:
        lanes = {lane.value for lane in RetrievalSignal}
        for name, values in (
            ("lane_availability", self.lane_availability),
            ("calibrated_evidence", self.calibrated_evidence),
            ("lane_weights", self.lane_weights),
        ):
            if set(values) != lanes:
                raise ValueError(f"{name} must explicitly cover every retrieval lane")

        for lane in RetrievalSignal:
            score = self.calibrated_evidence[lane.value]
            if not self.lane_availability[lane.value] and score is not None:
                raise ValueError("an unavailable lane cannot have calibrated evidence")
            if score is None and self.lane_weights[lane.value] != 0.0:
                raise ValueError("a lane without calibrated evidence must have zero weight")

        direct_scores = tuple(
            score
            for lane in RetrievalSignal
            if lane is not RetrievalSignal.GRAPH
            and (score := self.calibrated_evidence[lane.value]) is not None
        )
        expected_best_direct = max(direct_scores, default=None)
        if self.best_direct_calibrated_evidence != expected_best_direct:
            raise ValueError("best direct evidence must equal the strongest calibrated direct lane")

        expected_abstention = (
            expected_best_direct is None or expected_best_direct < self.abstention_threshold
        )
        if self.abstain != expected_abstention:
            raise ValueError("abstention must be derived from calibrated direct evidence")
        expected_reason = (
            "no_calibrated_direct_evidence"
            if expected_best_direct is None
            else "below_calibrated_evidence_threshold"
            if expected_abstention
            else None
        )
        if self.abstention_reason != expected_reason:
            raise ValueError("abstention reason must identify the calibrated-evidence failure")

        graph_weight = self.lane_weights[RetrievalSignal.GRAPH.value]
        if self.graph_decision is GraphDecision.SKIP:
            if self.max_graph_depth != 0 or self.per_hop_continuation_thresholds:
                raise ValueError("a skipped graph must have zero depth and no hop thresholds")
            if graph_weight != 0.0:
                raise ValueError("a skipped graph must have zero lane weight")
        else:
            if self.abstain:
                raise ValueError("an abstaining query cannot allow graph traversal")
            if self.max_graph_depth < 1:
                raise ValueError("an allowed graph requires a positive depth")
            if len(self.per_hop_continuation_thresholds) != self.max_graph_depth:
                raise ValueError("an allowed graph requires one threshold per permitted hop")
            if any(
                later < earlier
                for earlier, later in zip(
                    self.per_hop_continuation_thresholds,
                    self.per_hop_continuation_thresholds[1:],
                    strict=False,
                )
            ):
                raise ValueError("deeper graph hops cannot have a weaker continuation threshold")
            if graph_weight <= 0.0:
                raise ValueError("an allowed graph requires a positive lane weight")
            if not self.lane_availability[RetrievalSignal.GRAPH.value]:
                raise ValueError("an allowed graph lane must be available")
            if self.calibrated_evidence[RetrievalSignal.GRAPH.value] is None:
                raise ValueError("an allowed graph lane requires calibrated evidence")
        return self


@dataclass(frozen=True, slots=True)
class AdaptiveRetrievalPolicy:
    """Stateless deterministic policy over immutable configuration."""

    config: AdaptiveRetrievalPolicyConfig = DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG

    @property
    def config_fingerprint(self) -> str:
        payload = json.dumps(
            self.config.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def decide(
        self,
        query: AdaptiveQueryFeatures,
        lane_evidence: Iterable[LaneRawEvidence],
    ) -> AdaptiveRetrievalDecision:
        """Return weights, graph bounds, and calibrated abstention.

        Missing lanes and missing raw evidence receive no calibration and zero
        weight.  The maximum calibrated *direct* evidence is used for
        abstention, matching the conservative "one lane must defend the hit"
        rule rather than allowing several weak lanes to manufacture support.
        """

        observed: dict[RetrievalSignal, LaneRawEvidence] = {}
        for item in lane_evidence:
            if item.lane in observed:
                raise ValueError(f"duplicate lane evidence: {item.lane.value}")
            observed[item.lane] = item

        coefficients = {item.lane: item for item in self.config.lane_coefficients}
        availability: dict[str, bool] = {}
        calibrated: dict[str, float | None] = {}
        weights: dict[str, float] = {}

        for lane in RetrievalSignal:
            evidence = observed.get(lane)
            availability[lane.value] = evidence.available if evidence is not None else False
            coefficient = coefficients.get(lane)
            if (
                evidence is None
                or not evidence.available
                or evidence.raw_evidence is None
                or coefficient is None
            ):
                calibrated[lane.value] = None
                weights[lane.value] = 0.0
                continue

            score = coefficient.calibrate(float(evidence.raw_evidence))
            calibrated[lane.value] = score
            multiplier = 1.0
            if lane is RetrievalSignal.EXACT and query.exact_lookup:
                multiplier = float(self.config.exact_query_weight_multiplier)
            elif lane is RetrievalSignal.GRAPH and query.requires_multiple_evidence:
                multiplier = float(self.config.multi_evidence_graph_weight_multiplier)
            weights[lane.value] = min(100.0, float(coefficient.base_weight) * score * multiplier)

        direct_scores = tuple(
            score
            for lane in RetrievalSignal
            if lane is not RetrievalSignal.GRAPH and (score := calibrated[lane.value]) is not None
        )
        best_direct = max(direct_scores, default=None)
        abstain = best_direct is None or best_direct < self.config.abstention_threshold
        if best_direct is None:
            abstention_reason = "no_calibrated_direct_evidence"
        elif abstain:
            abstention_reason = "below_calibrated_evidence_threshold"
        else:
            abstention_reason = None

        exact_score = calibrated[RetrievalSignal.EXACT.value]
        sufficient_simple_exact = (
            query.exact_lookup
            and not query.requires_multiple_evidence
            and exact_score is not None
            and exact_score >= self.config.exact_sufficiency_threshold
        )
        graph_requested = (
            query.requires_multiple_evidence
            or query.relational_need >= self.config.graph_relation_threshold
        )
        graph_available = availability[RetrievalSignal.GRAPH.value]
        graph_score = calibrated[RetrievalSignal.GRAPH.value]

        reasons: list[str] = []
        allow_graph = False
        if abstain:
            reasons.append("graph_skipped:abstained")
        elif sufficient_simple_exact:
            reasons.append("graph_skipped:sufficient_exact")
        elif not graph_requested:
            reasons.append("graph_skipped:not_requested")
        elif not graph_available:
            reasons.append("graph_skipped:unavailable")
        elif graph_score is None:
            reasons.append("graph_skipped:missing_calibrated_evidence")
        elif graph_score < self.config.graph_lane_threshold:
            reasons.append("graph_skipped:weak_graph_evidence")
        elif best_direct is None or best_direct < self.config.graph_seed_threshold:
            reasons.append("graph_skipped:weak_seed_evidence")
        else:
            allow_graph = True
            reasons.append(
                "graph_allowed:multi_evidence"
                if query.requires_multiple_evidence
                else "graph_allowed:relational_query"
            )

        if allow_graph:
            max_graph_depth = (
                self.config.max_graph_depth
                if query.requires_multiple_evidence
                or query.relational_need >= self.config.deep_graph_relation_threshold
                else 1
            )
            hop_thresholds = self.config.per_hop_continuation_thresholds[:max_graph_depth]
            graph_decision = GraphDecision.ALLOW
        else:
            max_graph_depth = 0
            hop_thresholds = ()
            graph_decision = GraphDecision.SKIP
            weights[RetrievalSignal.GRAPH.value] = 0.0

        return AdaptiveRetrievalDecision(
            policy_version=self.config.policy_version,
            config_fingerprint=self.config_fingerprint,
            lane_availability=availability,
            calibrated_evidence=calibrated,
            lane_weights=weights,
            graph_decision=graph_decision,
            max_graph_depth=max_graph_depth,
            per_hop_continuation_thresholds=hop_thresholds,
            best_direct_calibrated_evidence=best_direct,
            abstention_threshold=self.config.abstention_threshold,
            abstain=abstain,
            abstention_reason=abstention_reason,
            reasons=tuple(reasons),
        )

    def should_continue_graph(
        self,
        decision: AdaptiveRetrievalDecision,
        hop: GraphHopEvidence,
    ) -> bool:
        """Return whether one observed hop is strong enough to continue.

        Missing evidence, an unavailable hop, a config mismatch, a skipped
        graph, and depths outside the allowed bound all stop traversal.  The
        continuation comparison is in calibrated-evidence space.
        """

        if decision.config_fingerprint != self.config_fingerprint:
            raise ValueError("decision was produced by a different policy configuration")
        if decision.graph_decision is not GraphDecision.ALLOW or decision.abstain:
            return False
        if hop.depth > decision.max_graph_depth or not hop.available or hop.raw_evidence is None:
            return False
        coefficient = next(
            (item for item in self.config.lane_coefficients if item.lane is RetrievalSignal.GRAPH),
            None,
        )
        if coefficient is None:
            return False
        calibrated = coefficient.calibrate(float(hop.raw_evidence))
        threshold = decision.per_hop_continuation_thresholds[hop.depth - 1]
        return calibrated >= threshold


__all__ = [
    "ADAPTIVE_POLICY_CONFIG_VERSION",
    "COEFFICIENT_PROVENANCE",
    "DEFAULT_ADAPTIVE_RETRIEVAL_POLICY_CONFIG",
    "AdaptiveQueryFeatures",
    "AdaptiveRetrievalDecision",
    "AdaptiveRetrievalPolicy",
    "AdaptiveRetrievalPolicyConfig",
    "GraphDecision",
    "GraphHopEvidence",
    "LaneRawEvidence",
    "ReviewedLaneCoefficients",
]
