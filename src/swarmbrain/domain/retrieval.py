"""Internal retrieval planning, candidate, and diagnostic contracts.

These models deliberately do not extend the public recall request or response.
The server owns purpose selection, lane budgets, fusion, and full traces.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from .common import (
    ContractModel,
    EvidenceId,
    MemoryId,
    ProjectId,
    RepositoryId,
    RunId,
    SemanticLabel,
    SwarmId,
    TaskId,
    TenantId,
    UUIDString,
)
from .memory import Visibility


class RetrievalPurpose(StrEnum):
    INTERACTIVE_RECALL = "interactive_recall"
    TASK_BOOTSTRAP = "task_bootstrap"
    HANDOFF_RECOVERY = "handoff_recovery"
    PLANNING = "planning"
    CONFLICT_REVIEW = "conflict_review"
    HISTORICAL_AUDIT = "historical_audit"
    REPOSITORY_ORIENTATION = "repository_orientation"


class RetrievalSignal(StrEnum):
    EXACT = "exact"
    LEXICAL = "lexical"
    FUZZY = "fuzzy"
    DENSE = "dense"
    TEMPORAL = "temporal"
    SOURCE = "source"
    GRAPH = "graph"


class RetrievalScope(ContractModel):
    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    task_id: TaskId | None = None
    visibilities: frozenset[Visibility]


class DenseQuery(ContractModel):
    """Internal query embedding or an observable provider degradation.

    Query vectors never become part of the public recall contract or the
    persisted retrieval trace.  The model and dimensions are retained even
    when generation fails so the dense lane can report a useful degraded
    batch while the remaining lanes continue.
    """

    model: SemanticLabel
    dimensions: int = Field(ge=1)
    values: tuple[FiniteFloat, ...] | None = Field(default=None, min_length=1)
    unavailable_reason: SemanticLabel | None = None

    @model_validator(mode="after")
    def vector_or_failure(self) -> Self:
        if (self.values is None) == (self.unavailable_reason is None):
            raise ValueError("dense query requires exactly one of values or unavailable_reason")
        if self.values is not None and len(self.values) != self.dimensions:
            raise ValueError("dense query dimensions must equal the vector length")
        return self


LaneBudget = Annotated[int, Field(ge=1, le=2000)]
LaneWeight = Annotated[FiniteFloat, Field(gt=0.0, le=100.0)]
GraphSeedLimit = Annotated[int, Field(ge=0, le=64)]
GraphFanout = Annotated[int, Field(ge=0, le=64)]
GraphEdgeBudget = Annotated[int, Field(ge=0, le=10000)]
RerankAlpha = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
RerankWindow = Annotated[int, Field(ge=0, le=512)]


class RetrievalPlan(ContractModel):
    purpose: RetrievalPurpose
    intent: SemanticLabel = "general"
    domain_lanes: frozenset[SemanticLabel]
    signal_lanes: frozenset[RetrievalSignal]
    world_at: AwareDatetime | None = None
    recorded_at: AwareDatetime | None = None
    hard_scope: RetrievalScope
    lane_budgets: dict[str, LaneBudget]
    lane_weights: dict[str, LaneWeight]
    seed_memory_ids: tuple[MemoryId, ...] = ()
    max_graph_hops: int = Field(default=0, ge=0, le=4)
    graph_seed_limit: GraphSeedLimit = 0
    graph_max_fanout: GraphFanout = 0
    graph_edge_budget: GraphEdgeBudget = 0
    graph_link_types: frozenset[SemanticLabel] = frozenset()
    rerank: bool = False
    rerank_alpha: RerankAlpha = 0.0
    rerank_window: RerankWindow = 0
    diversify: bool = False
    token_budget: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def graph_lane_is_fully_bounded(self) -> Self:
        enabled = RetrievalSignal.GRAPH in self.signal_lanes
        configured = (
            self.max_graph_hops > 0
            and self.graph_seed_limit > 0
            and self.graph_max_fanout > 0
            and self.graph_edge_budget > 0
            and bool(self.graph_link_types)
        )
        if enabled != configured:
            raise ValueError("graph lane selection and graph bounds must be configured together")
        if enabled and (
            RetrievalSignal.GRAPH.value not in self.lane_budgets
            or RetrievalSignal.GRAPH.value not in self.lane_weights
        ):
            raise ValueError("graph lane requires a candidate budget and fusion weight")
        return self

    @model_validator(mode="after")
    def rerank_is_fully_bounded(self) -> Self:
        """``rerank`` and its two bounds are one switch, not three knobs.

        A plan that asks for reranking without a window would silently reorder
        nothing, and a window without the flag would be dead configuration that
        a later reader could mistake for an active setting.
        """

        configured = self.rerank_alpha > 0.0 and self.rerank_window > 0
        if self.rerank != configured:
            raise ValueError("rerank selection requires a positive alpha and window together")
        return self


class Candidate(ContractModel):
    """Non-authoritative reference emitted by one candidate lane."""

    resource_type: SemanticLabel
    resource_id: MemoryId
    resource_version: int = Field(ge=1)
    canonical_id: MemoryId
    domain_lane: SemanticLabel
    signal: RetrievalSignal
    rank: int = Field(ge=1)
    raw_score: FiniteFloat | None = None
    projection_id: str | None = None
    projection_version: str | None = None
    reasons: tuple[str, ...] = ()
    evidence_ids: tuple[EvidenceId, ...] = ()
    path: tuple[str, ...] = ()


class CandidateBatch(ContractModel):
    lane: RetrievalSignal
    candidates: tuple[Candidate, ...] = ()
    examined_count: int = Field(ge=0)
    latency_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    truncated: bool = False
    degraded: bool = False
    degradation_reason: str | None = None
    projection_watermark: str | None = None


class FusionContribution(ContractModel):
    canonical_id: MemoryId
    lane: RetrievalSignal
    rank: int = Field(ge=1)
    lane_weight: LaneWeight
    raw_score: FiniteFloat | None = None
    rrf_contribution: Annotated[FiniteFloat, Field(gt=0.0)]


class FusedCandidate(ContractModel):
    canonical_id: MemoryId
    raw_rrf: Annotated[FiniteFloat, Field(gt=0.0)]
    normalized_score: Annotated[FiniteFloat, Field(gt=0.0, le=1.0)]
    contributions: tuple[FusionContribution, ...]
    reasons: tuple[str, ...] = ()


RelevanceScore = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]


class CandidateRelevance(ContractModel):
    """Rank-independent relevance evidence for one fused candidate.

    ``relevance`` is the conservative combination (``max``) of the per-lane
    components below, each of which is a similarity in [0, 1] that does not
    depend on where the candidate landed in any lane's ranking.  It is the
    quantity ``RecallQuery.min_score`` is gated on; it is deliberately *not*
    the public ``RecallHit.score``, whose rank-anchored meaning is unchanged.

    See :mod:`swarmbrain.retrieval.relevance` for how each component is derived
    and why the lexical and trigram components are recomputed from canonical
    text instead of read off the lanes' raw scores.
    """

    canonical_id: MemoryId
    relevance: RelevanceScore
    exact_term: RelevanceScore = 0.0
    lexical_coverage: RelevanceScore = 0.0
    trigram_similarity: RelevanceScore = 0.0
    dense_cosine: RelevanceScore = 0.0


class HydrationRejection(ContractModel):
    canonical_id: MemoryId
    reason: SemanticLabel = "not_recallable"


class RetrievalTrace(ContractModel):
    """Auditable internal trace; never serialized in the v1 recall response."""

    trace_id: UUIDString
    plan: RetrievalPlan
    parsed_identifiers: tuple[str, ...] = ()
    batches: tuple[CandidateBatch, ...] = ()
    fusion_version: SemanticLabel = "weighted-rrf-v1"
    fused_candidates: tuple[FusedCandidate, ...] = ()
    relevance_version: SemanticLabel = "lane-max-v1"
    # Only the candidates the service actually evaluated, in fused order.  The
    # service stops once the public limit is satisfied, so this is a prefix of
    # the hydrated candidates rather than a score for every fused candidate.
    candidate_relevance: tuple[CandidateRelevance, ...] = ()
    hydrated_ids: tuple[MemoryId, ...] = ()
    hydration_rejections: tuple[HydrationRejection, ...] = ()
    final_ids: tuple[MemoryId, ...] = ()
    degraded_lanes: frozenset[RetrievalSignal] = frozenset()
    abstained: bool = False
    abstention_reason: str | None = None
    started_at: AwareDatetime
    completed_at: AwareDatetime


__all__ = [
    "Candidate",
    "CandidateBatch",
    "CandidateRelevance",
    "DenseQuery",
    "FusedCandidate",
    "FusionContribution",
    "HydrationRejection",
    "RetrievalPlan",
    "RetrievalPurpose",
    "RetrievalScope",
    "RetrievalSignal",
    "RetrievalTrace",
]
