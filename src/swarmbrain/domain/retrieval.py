"""Internal retrieval planning, candidate, and diagnostic contracts.

These models deliberately do not extend the public recall request or response.
The server owns purpose selection, lane budgets, fusion, and full traces.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, Field, FiniteFloat

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


LaneBudget = Annotated[int, Field(ge=1, le=2000)]
LaneWeight = Annotated[FiniteFloat, Field(gt=0.0, le=100.0)]


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
    rerank: bool = False
    diversify: bool = False
    token_budget: int | None = Field(default=None, ge=1)


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
    "FusedCandidate",
    "FusionContribution",
    "HydrationRejection",
    "RetrievalPlan",
    "RetrievalPurpose",
    "RetrievalScope",
    "RetrievalSignal",
    "RetrievalTrace",
]
