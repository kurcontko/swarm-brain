"""Temporal, source-preserving swarm memory contracts."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from .common import (
    AgentId,
    Confidence,
    ContentText,
    ContractModel,
    JsonObject,
    MemoryId,
    MutationCommand,
    ProjectId,
    RepositoryId,
    RunId,
    ShortText,
    SwarmId,
    TaskId,
    TenantId,
    UUIDString,
    utc_now,
)
from .evidence import EvidenceRef


class MemoryKind(StrEnum):
    OBSERVATION = "observation"
    INVARIANT = "invariant"
    HYPOTHESIS = "hypothesis"
    DECISION = "decision"
    ATTEMPT = "attempt"
    OUTCOME = "outcome"
    PROCEDURE = "procedure"
    WARNING = "warning"
    HANDOFF = "handoff"


class MemoryState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    SUPERSEDED = "superseded"


class Visibility(StrEnum):
    TASK = "task"
    RUN = "run"
    REPOSITORY = "repository"


class MemoryOperation(StrEnum):
    """Conservative operation selected by the memory policy."""

    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"
    NOOP = "noop"


class MemoryReviewDecision(StrEnum):
    CONFIRM = "confirm"
    REFUTE = "refute"


class MemoryLinkKind(StrEnum):
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DUPLICATE_OF = "duplicate_of"
    MERGED_FROM = "merged_from"
    RELATED_TO = "related_to"


class Memory(ContractModel):
    """One append-only version of a swarm memory.

    ``valid_*`` describes when the assertion is true in the world, while
    ``recorded_*`` describes when this database version was believed.  An
    update appends a new row and links both rows; it never replaces content.
    """

    memory_id: MemoryId
    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    task_id: TaskId | None = None
    author_agent_id: AgentId
    kind: MemoryKind
    state: MemoryState = MemoryState.TENTATIVE
    visibility: Visibility = Visibility.RUN
    content: ContentText
    title: str | None = Field(default=None, max_length=500)
    tags: tuple[str, ...] = ()
    confidence: Confidence = 0.5
    evidence: tuple[EvidenceRef, ...] = ()
    valid_from: AwareDatetime
    valid_to: AwareDatetime | None = None
    recorded_from: AwareDatetime = Field(default_factory=utc_now)
    recorded_to: AwareDatetime | None = None
    supersedes_memory_id: MemoryId | None = None
    superseded_by_memory_id: MemoryId | None = None
    version: int = Field(default=1, ge=1)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(item.strip().lower() for item in value if item.strip()))
        return normalized

    @model_validator(mode="after")
    def validate_temporal_lineage(self) -> Self:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be later than valid_from")
        if self.recorded_to is not None and self.recorded_to <= self.recorded_from:
            raise ValueError("recorded_to must be later than recorded_from")
        if self.supersedes_memory_id == self.memory_id:
            raise ValueError("a memory cannot supersede itself")
        if self.superseded_by_memory_id == self.memory_id:
            raise ValueError("a memory cannot be superseded by itself")
        if self.state is MemoryState.SUPERSEDED and self.superseded_by_memory_id is None:
            raise ValueError("superseded memories require superseded_by_memory_id")
        if self.visibility is Visibility.TASK and self.task_id is None:
            raise ValueError("task-visible memory requires task_id")
        return self


class MemoryPolicyDecision(ContractModel):
    operation: MemoryOperation
    reason: ShortText
    confidence: Confidence
    target_memory_ids: tuple[MemoryId, ...] = ()

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        needs_target = self.operation in {
            MemoryOperation.UPDATE,
            MemoryOperation.MERGE,
            MemoryOperation.DELETE,
        }
        if needs_target and not self.target_memory_ids:
            raise ValueError(f"{self.operation.value} requires at least one target memory")
        if self.operation is MemoryOperation.ADD and self.target_memory_ids:
            raise ValueError("add cannot target an existing memory")
        return self


class RememberCommand(MutationCommand):
    """Model-visible memory proposal; actor-owned scope is deliberately absent."""

    kind: MemoryKind
    content: ContentText
    desired_state: MemoryState = MemoryState.TENTATIVE
    visibility: Visibility = Visibility.RUN
    task_id: TaskId | None = None
    title: str | None = Field(default=None, max_length=500)
    tags: tuple[str, ...] = ()
    confidence: Confidence = 0.5
    evidence: tuple[EvidenceRef, ...] = ()
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    supersedes_memory_id: MemoryId | None = None
    related_memory_ids: tuple[MemoryId, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.strip().lower() for item in value if item.strip()))

    @model_validator(mode="after")
    def validate_scope_and_interval(self) -> Self:
        if self.desired_state is MemoryState.SUPERSEDED:
            raise ValueError("a newly appended memory cannot start superseded")
        if self.visibility is Visibility.TASK and self.task_id is None:
            raise ValueError("task-visible memory requires task_id")
        if self.valid_from is None and self.valid_to is not None:
            raise ValueError("valid_to requires valid_from")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            raise ValueError("valid_to must be later than valid_from")
        return self


class RememberResult(ContractModel):
    operation: MemoryOperation
    decision: MemoryPolicyDecision
    memory: Memory | None = None
    affected_memories: tuple[Memory, ...] = ()
    replayed: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.operation is not self.decision.operation:
            raise ValueError("operation must match the policy decision")
        if (
            self.operation
            in {
                MemoryOperation.ADD,
                MemoryOperation.UPDATE,
                MemoryOperation.MERGE,
            }
            and self.memory is None
        ):
            raise ValueError(f"{self.operation.value} result requires a resulting memory")
        return self


class RecallQuery(ContractModel):
    """Current recall by default; historical/refuted data is opt-in."""

    text: ContentText
    task_id: TaskId | None = None
    memory_ids: frozenset[MemoryId] = Field(default_factory=frozenset)
    kinds: frozenset[MemoryKind] = Field(default_factory=frozenset)
    visibilities: frozenset[Visibility] = Field(default_factory=lambda: frozenset(Visibility))
    states: frozenset[MemoryState] | None = None
    include_refuted: bool = False
    include_superseded: bool = False
    world_at: AwareDatetime | None = None
    recorded_at: AwareDatetime | None = None
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    limit: int = Field(default=10, ge=1, le=100)
    include_evidence: bool = True
    include_lineage: bool = False

    @property
    def effective_states(self) -> frozenset[MemoryState]:
        states = set(self.states or {MemoryState.TENTATIVE, MemoryState.CONFIRMED})
        if self.include_refuted:
            states.add(MemoryState.REFUTED)
        if self.include_superseded:
            states.add(MemoryState.SUPERSEDED)
        return frozenset(states)


class RecallHit(ContractModel):
    memory: Memory
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reasons: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()


class RecallBundle(ContractModel):
    query: RecallQuery
    hits: tuple[RecallHit, ...] = ()
    generated_at: AwareDatetime = Field(default_factory=utc_now)
    total_candidates: int = Field(default=0, ge=0)
    truncated: bool = False


class MemoryLink(ContractModel):
    link_id: UUIDString
    source_memory_id: MemoryId
    target_memory_id: MemoryId
    kind: MemoryLinkKind
    evidence: tuple[EvidenceRef, ...] = ()
    reason: str | None = Field(default=None, max_length=4096)
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def no_self_link(self) -> Self:
        if self.source_memory_id == self.target_memory_id:
            raise ValueError("memory lineage links cannot be self-referential")
        return self


class MemoryLineage(ContractModel):
    requested_memory_id: MemoryId
    memories: tuple[Memory, ...]
    links: tuple[MemoryLink, ...] = ()
    current_memory_ids: tuple[MemoryId, ...] = ()

    @model_validator(mode="after")
    def requested_memory_is_present(self) -> Self:
        known = {item.memory_id for item in self.memories}
        if self.requested_memory_id not in known:
            raise ValueError("requested memory must be included in lineage")
        if not set(self.current_memory_ids).issubset(known):
            raise ValueError("current_memory_ids must refer to lineage memories")
        return self


class ReviewMemoryCommand(MutationCommand):
    memory_id: MemoryId
    expected_version: int = Field(ge=1)
    decision: MemoryReviewDecision
    reason: ContentText
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)


class ReviewMemoryResult(ContractModel):
    memory: Memory
    replayed: bool = False


class EmbeddingVector(ContractModel):
    memory_id: MemoryId
    model: ShortText
    dimensions: int = Field(ge=1)
    values: tuple[float, ...] = Field(min_length=1)
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @field_validator("values")
    @classmethod
    def finite_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not isfinite(item) for item in value):
            raise ValueError("embedding values must be finite")
        return value

    @model_validator(mode="after")
    def dimensions_match(self) -> Self:
        if self.dimensions != len(self.values):
            raise ValueError("dimensions must equal the vector length")
        return self


class EmbeddingMatch(ContractModel):
    memory_id: MemoryId
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


__all__ = [
    "EmbeddingMatch",
    "EmbeddingVector",
    "Memory",
    "MemoryKind",
    "MemoryLineage",
    "MemoryLink",
    "MemoryLinkKind",
    "MemoryOperation",
    "MemoryPolicyDecision",
    "MemoryReviewDecision",
    "MemoryState",
    "RecallBundle",
    "RecallHit",
    "RecallQuery",
    "RememberCommand",
    "RememberResult",
    "ReviewMemoryCommand",
    "ReviewMemoryResult",
    "Visibility",
]
