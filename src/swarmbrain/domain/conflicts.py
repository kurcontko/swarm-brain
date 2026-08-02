"""Evidence-backed contradiction reporting and resolution contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from .common import (
    AgentId,
    ConflictId,
    ContentText,
    ContractModel,
    JsonObject,
    MemoryId,
    MutationCommand,
    ProjectId,
    RepositoryId,
    RunId,
    SwarmId,
    TaskId,
    TenantId,
    UUIDString,
    utc_now,
)
from .evidence import EvidenceRef


class ConflictStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ConflictSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictResolutionKind(StrEnum):
    PREFER_EXISTING = "prefer_existing"
    PREFER_NEWER = "prefer_newer"
    SUPERSEDE = "supersede"
    MERGE = "merge"
    REFUTE_ALL = "refute_all"
    DISMISS = "dismiss"


class ConflictResolutionProposal(ContractModel):
    kind: ConflictResolutionKind
    rationale: ContentText
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    supported_memory_ids: tuple[MemoryId, ...] = ()
    refuted_memory_ids: tuple[MemoryId, ...] = ()

    @model_validator(mode="after")
    def has_an_outcome(self) -> Self:
        if self.kind is not ConflictResolutionKind.DISMISS and not (
            self.supported_memory_ids or self.refuted_memory_ids
        ):
            raise ValueError("a resolution must identify supported or refuted memories")
        if set(self.supported_memory_ids) & set(self.refuted_memory_ids):
            raise ValueError("a memory cannot be both supported and refuted")
        return self


class ConflictResolution(ConflictResolutionProposal):
    resolution_id: UUIDString
    resolved_by_agent_id: AgentId
    resolved_at: AwareDatetime = Field(default_factory=utc_now)


class Conflict(ContractModel):
    conflict_id: ConflictId
    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    task_id: TaskId | None = None
    memory_ids: tuple[MemoryId, ...] = Field(min_length=2)
    description: ContentText
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    status: ConflictStatus = ConflictStatus.OPEN
    reported_by_agent_id: AgentId
    reported_evidence: tuple[EvidenceRef, ...] = ()
    resolution: ConflictResolution | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    version: int = Field(default=1, ge=1)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_conflict(self) -> Self:
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("conflict memory_ids must be unique")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.status is ConflictStatus.RESOLVED and self.resolution is None:
            raise ValueError("resolved conflicts require an evidence-backed resolution")
        if self.status is ConflictStatus.OPEN and self.resolution is not None:
            raise ValueError("open conflicts cannot have a resolution")
        return self


class ReportConflictCommand(MutationCommand):
    memory_ids: tuple[MemoryId, ...] = Field(min_length=2)
    description: ContentText
    task_id: TaskId | None = None
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    evidence: tuple[EvidenceRef, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_memories(self) -> Self:
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("conflict memory_ids must be unique")
        return self


class ResolveConflictCommand(MutationCommand):
    conflict_id: ConflictId
    expected_version: int = Field(ge=1)
    resolution: ConflictResolutionProposal


class ReportConflictResult(ContractModel):
    conflict: Conflict
    replayed: bool = False


class ResolveConflictResult(ContractModel):
    conflict: Conflict
    replayed: bool = False


__all__ = [
    "Conflict",
    "ConflictResolution",
    "ConflictResolutionKind",
    "ConflictResolutionProposal",
    "ConflictSeverity",
    "ConflictStatus",
    "ReportConflictCommand",
    "ReportConflictResult",
    "ResolveConflictCommand",
    "ResolveConflictResult",
]
