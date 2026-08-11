"""Durable leased work and effect-once contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import (
    AwareDatetime,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .common import (
    AgentId,
    ContractModel,
    JsonObject,
    MutationCommand,
    ProjectId,
    RepositoryId,
    RunId,
    SourceId,
    SwarmId,
    TaskId,
    TenantId,
    UUIDString,
    utc_now,
)
from .consolidation import ConsolidationReflection, ConsolidationWorkPayload
from .evidence import Sha256
from .extraction import (
    ExtractionCandidate,
    ExtractionProvenance,
    ExtractionRoute,
    ExtractionStage,
)
from .memory import EmbeddingVector, MemoryLinkKind

WorkerId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class WorkKind(StrEnum):
    EXTRACT_SOURCE = "extract_source"
    EMBED_MEMORY = "embed_memory"
    PERSIST_ARTIFACT = "persist_artifact"
    CONSOLIDATE_MEMORY = "consolidate_memory"


def durable_work_id(
    tenant_id: str,
    run_id: str,
    kind: WorkKind,
    dedupe_key: str,
) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            ":".join(
                (
                    "swarmbrain-work",
                    tenant_id,
                    run_id,
                    kind.value,
                    dedupe_key,
                )
            ),
        )
    )


class WorkStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RETRY = "retry"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceExtractionState(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def source_extraction_state(status: WorkStatus) -> SourceExtractionState:
    if status in {WorkStatus.PENDING, WorkStatus.RETRY}:
        return SourceExtractionState.QUEUED
    return SourceExtractionState(status.value)


class WorkEffectKind(StrEnum):
    MEMORY = "memory"


HANDLER_WORK_KINDS = frozenset({WorkKind.EMBED_MEMORY, WorkKind.PERSIST_ARTIFACT})


class EmbedMemoryWorkPayload(ContractModel):
    """Durable textual projection needed to reproduce an embedding request."""

    content: str
    model: str = Field(min_length=1, max_length=255)
    metadata: JsonObject = Field(default_factory=dict)


class PersistArtifactWorkPayload(ContractModel):
    """Durable reference to staged bytes awaiting an external artifact sink."""

    name: str = Field(min_length=1, max_length=4096)
    media_type: str = Field(min_length=1, max_length=255)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    staging_uri: str = Field(min_length=1, max_length=2048)
    metadata: JsonObject = Field(default_factory=dict)


def validate_work_payload(
    kind: WorkKind,
    payload: JsonObject,
) -> EmbedMemoryWorkPayload | PersistArtifactWorkPayload | ConsolidationWorkPayload:
    if kind is WorkKind.EMBED_MEMORY:
        return EmbedMemoryWorkPayload.model_validate(payload)
    if kind is WorkKind.PERSIST_ARTIFACT:
        return PersistArtifactWorkPayload.model_validate(payload)
    if kind is WorkKind.CONSOLIDATE_MEMORY:
        return ConsolidationWorkPayload.model_validate(payload)
    raise ValueError("extract_source work must be created by raw-source ingestion")


class EnqueueWorkCommand(MutationCommand):
    kind: WorkKind
    subject_id: UUIDString
    payload: JsonObject
    dedupe_key: str = Field(min_length=1, max_length=512)
    task_id: TaskId | None = None
    priority: int = Field(default=0, ge=-1000, le=1000)
    max_attempts: int = Field(default=5, ge=1, le=100)
    available_at: AwareDatetime = Field(default_factory=utc_now)

    @field_validator("payload")
    @classmethod
    def normalize_handler_payload(
        cls,
        value: JsonObject,
        info: ValidationInfo,
    ) -> JsonObject:
        kind = info.data.get("kind")
        if not isinstance(kind, WorkKind):
            return value
        return validate_work_payload(kind, value).model_dump(mode="json")


class PreparedWorkEnqueue(ContractModel):
    work_id: UUIDString
    command: EnqueueWorkCommand


class WorkItem(ContractModel):
    work_id: UUIDString
    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    task_id: TaskId | None = None
    requested_by_agent_id: AgentId
    kind: WorkKind
    subject_id: UUIDString
    payload: JsonObject = Field(default_factory=dict)
    dedupe_key: str = Field(min_length=1, max_length=512)
    priority: int = Field(default=0, ge=-1000, le=1000)
    status: WorkStatus = WorkStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=5, ge=1, le=100)
    available_at: AwareDatetime = Field(default_factory=utc_now)
    locked_by: WorkerId | None = None
    lease_token: UUIDString | None = None
    lease_version: int = Field(default=0, ge=0)
    locked_until: AwareDatetime | None = None
    outcome: str | None = Field(default=None, max_length=255)
    result: JsonObject | None = None
    last_error: str | None = Field(default=None, max_length=1024)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    completed_at: AwareDatetime | None = None
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.attempts > self.max_attempts:
            raise ValueError("work attempts cannot exceed max_attempts")
        if self.status is WorkStatus.LEASED and (
            self.locked_by is None or self.lease_token is None or self.locked_until is None
        ):
            raise ValueError("leased work requires owner, token, and locked_until")
        if self.status is WorkStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed work requires completed_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self


class EnqueueWorkResult(ContractModel):
    item: WorkItem
    replayed: bool = False


class WorkLease(ContractModel):
    item: WorkItem
    worker_id: WorkerId
    lease_token: UUIDString
    lease_version: int = Field(ge=1)
    work_version: int = Field(ge=1)
    attempt: int = Field(ge=1)
    locked_until: AwareDatetime

    @model_validator(mode="after")
    def validate_fence(self) -> Self:
        if self.item.status is not WorkStatus.LEASED:
            raise ValueError("work lease requires a leased item")
        if (
            self.item.locked_by != self.worker_id
            or self.item.lease_token != self.lease_token
            or self.item.lease_version != self.lease_version
            or self.item.version != self.work_version
            or self.item.attempts != self.attempt
            or self.item.locked_until != self.locked_until
        ):
            raise ValueError("work lease fence must match the item")
        return self


class WorkLeaseBatch(ContractModel):
    leases: tuple[WorkLease, ...] = ()


class StageConsolidationPlanCommand(ContractModel):
    """Fence a model-generated plan before any governed effects are applied."""

    work_id: UUIDString
    worker_id: WorkerId
    lease_token: UUIDString
    lease_version: int = Field(ge=1)
    expected_work_version: int = Field(ge=1)
    attempt: int = Field(ge=1)
    reflection: ConsolidationReflection


class StageConsolidationPlanResult(ContractModel):
    item: WorkItem
    reflection: ConsolidationReflection
    replayed: bool = False


class SourceExtractionStatus(ContractModel):
    """Scope-safe public view of extraction work; lease secrets are omitted."""

    source_id: SourceId
    work_id: UUIDString
    status: SourceExtractionState
    attempts: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    outcome: str | None = Field(default=None, max_length=255)
    route: ExtractionRoute | None = None
    candidate_count: int | None = Field(default=None, ge=0)
    memory_ids: tuple[UUIDString, ...] = ()
    created_at: AwareDatetime
    updated_at: AwareDatetime
    completed_at: AwareDatetime | None = None


class ClaimWorkCommand(ContractModel):
    worker_id: WorkerId
    kinds: frozenset[WorkKind] = Field(default_factory=lambda: frozenset(WorkKind), min_length=1)
    limit: int = Field(default=1, ge=1, le=100)
    lease_seconds: int = Field(default=60, ge=5, le=3600)
    extractor_name: str | None = Field(default=None, min_length=1, max_length=255)
    extractor_revision: str | None = Field(default=None, min_length=1, max_length=255)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_extractor_profile(self) -> Self:
        if (self.extractor_name is None) != (self.extractor_revision is None):
            raise ValueError("extractor_name and extractor_revision must be configured together")
        return self


class WorkEffect(ContractModel):
    effect_key: str = Field(min_length=1, max_length=512)
    kind: WorkEffectKind = WorkEffectKind.MEMORY
    payload_sha256: Sha256
    candidate: ExtractionCandidate
    candidate_origin: ExtractionStage = ExtractionStage.DETERMINISTIC

    @model_validator(mode="after")
    def validate_payload_hash(self) -> Self:
        payload = json.dumps(
            self.candidate.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if digest != self.payload_sha256:
            raise ValueError("payload_sha256 must identify the exact candidate payload")
        if self.candidate_origin not in {
            ExtractionStage.DETERMINISTIC,
            ExtractionStage.PROVIDER,
        }:
            raise ValueError("candidate_origin must be deterministic or provider")
        return self


class AppliedWorkEffect(ContractModel):
    work_id: UUIDString
    effect_key: str = Field(min_length=1, max_length=512)
    kind: WorkEffectKind
    payload_sha256: Sha256
    resource_id: UUIDString


class ApplyExtractionWorkCommand(ContractModel):
    work_id: UUIDString
    worker_id: WorkerId
    lease_token: UUIDString
    lease_version: int = Field(ge=1)
    expected_work_version: int = Field(ge=1)
    attempt: int = Field(ge=1)
    effects: tuple[WorkEffect, ...] = ()
    provenance: tuple[ExtractionProvenance, ...] = Field(min_length=1)
    outcome: str = Field(default="completed", min_length=1, max_length=255)
    result: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_effect_keys(self) -> Self:
        keys = [effect.effect_key for effect in self.effects]
        if len(keys) != len(set(keys)):
            raise ValueError("effect keys must be unique within one apply command")
        stages = [item.stage for item in self.provenance]
        if len(stages) != len(set(stages)):
            raise ValueError("provenance stages must be unique within one work attempt")
        if any(item.attempt != self.attempt for item in self.provenance):
            raise ValueError("provenance attempt must match the work attempt")
        candidate_keys = [
            effect.candidate.candidate_key
            for effect in self.effects
            if effect.candidate.candidate_key is not None
        ]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("candidate_key values must be unique within one apply command")
        known_candidate_keys = set(candidate_keys)
        allowed_relation_kinds = {
            MemoryLinkKind.DERIVED_FROM,
            MemoryLinkKind.SUPPORTS,
            MemoryLinkKind.CONTRADICTS,
            MemoryLinkKind.RELATED_TO,
        }
        for effect in self.effects:
            for relation in effect.candidate.relations:
                if relation.kind not in allowed_relation_kinds:
                    raise ValueError("candidate relation kind requires a governed lifecycle action")
                if relation.target_candidate_key not in known_candidate_keys:
                    raise ValueError(
                        "candidate relation target is not present in the apply command"
                    )
        return self


class ApplyWorkResult(ContractModel):
    item: WorkItem
    effects: tuple[AppliedWorkEffect, ...] = ()
    replayed: bool = False


class CompleteWorkCommand(ContractModel):
    work_id: UUIDString
    worker_id: WorkerId
    lease_token: UUIDString
    lease_version: int = Field(ge=1)
    expected_work_version: int = Field(ge=1)
    attempt: int = Field(ge=1)
    outcome: str = Field(default="completed", min_length=1, max_length=255)
    result: JsonObject = Field(default_factory=dict)


class ApplyEmbeddingWorkCommand(CompleteWorkCommand):
    """Fence an embedding vector and work completion into one atomic write."""

    vector: EmbeddingVector

    @model_validator(mode="after")
    def bind_result_to_vector(self) -> Self:
        if (
            self.result.get("model") != self.vector.model
            or self.result.get("dimensions") != self.vector.dimensions
            or self.result.get("vector_sha256") != embedding_vector_sha256(self.vector)
        ):
            raise ValueError("embedding result must identify the exact vector")
        return self


class CompleteWorkResult(ContractModel):
    item: WorkItem
    replayed: bool = False


class WorkHandlerResult(ContractModel):
    outcome: str = Field(default="completed", min_length=1, max_length=255)
    result: JsonObject = Field(default_factory=dict)


class EmbeddingWorkHandlerResult(WorkHandlerResult):
    vector: EmbeddingVector

    @model_validator(mode="after")
    def bind_result_to_vector(self) -> Self:
        if (
            self.result.get("model") != self.vector.model
            or self.result.get("dimensions") != self.vector.dimensions
            or self.result.get("vector_sha256") != embedding_vector_sha256(self.vector)
        ):
            raise ValueError("embedding result must identify the exact vector")
        return self


def embedding_vector_sha256(vector: EmbeddingVector) -> str:
    payload = json.dumps(vector.values, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FailWorkCommand(ContractModel):
    work_id: UUIDString
    worker_id: WorkerId
    lease_token: UUIDString
    lease_version: int = Field(ge=1)
    expected_work_version: int = Field(ge=1)
    attempt: int = Field(ge=1)
    error: str = Field(min_length=1, max_length=1024)
    retry_at: AwareDatetime
    provenance: tuple[ExtractionProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        stages = [item.stage for item in self.provenance]
        if len(stages) != len(set(stages)):
            raise ValueError("provenance stages must be unique within one work attempt")
        if any(item.attempt != self.attempt for item in self.provenance):
            raise ValueError("provenance attempt must match the work attempt")
        return self


class WorkLeaseLost(RuntimeError):
    """The work lease expired, changed owner, or failed its fence check."""


class WorkEffectConflict(RuntimeError):
    """An effect key was reused with a different payload hash."""


class WorkCompletionConflict(RuntimeError):
    """Completed work was replayed with a different outcome or result."""


class WorkFailureConflict(RuntimeError):
    """Failed work was replayed with different failure data."""


__all__ = [
    "AppliedWorkEffect",
    "ApplyEmbeddingWorkCommand",
    "ApplyExtractionWorkCommand",
    "ApplyWorkResult",
    "ClaimWorkCommand",
    "CompleteWorkCommand",
    "CompleteWorkResult",
    "EmbedMemoryWorkPayload",
    "EmbeddingWorkHandlerResult",
    "EnqueueWorkCommand",
    "EnqueueWorkResult",
    "FailWorkCommand",
    "HANDLER_WORK_KINDS",
    "PersistArtifactWorkPayload",
    "PreparedWorkEnqueue",
    "SourceExtractionStatus",
    "SourceExtractionState",
    "StageConsolidationPlanCommand",
    "StageConsolidationPlanResult",
    "WorkEffect",
    "WorkEffectConflict",
    "WorkEffectKind",
    "WorkCompletionConflict",
    "WorkFailureConflict",
    "WorkHandlerResult",
    "WorkItem",
    "WorkKind",
    "WorkLease",
    "WorkLeaseBatch",
    "WorkLeaseLost",
    "WorkStatus",
    "WorkerId",
    "durable_work_id",
    "embedding_vector_sha256",
    "source_extraction_state",
    "validate_work_payload",
]
