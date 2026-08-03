"""Source-preserving evidence and artifact contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from .common import (
    AgentId,
    ArtifactId,
    ContentText,
    ContractModel,
    EvidenceId,
    JsonObject,
    MemoryId,
    MutationCommand,
    RunId,
    SemanticLabel,
    ShortText,
    SourceId,
    TaskId,
    utc_now,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class EvidenceKind(StrEnum):
    SOURCE_CODE = "source_code"
    COMMAND_OUTPUT = "command_output"
    TEST_RESULT = "test_result"
    LOG = "log"
    MESSAGE = "message"
    DOCUMENT = "document"
    URL = "url"
    ARTIFACT = "artifact"
    MEMORY = "memory"


EvidenceKindValue = Annotated[
    EvidenceKind | SemanticLabel,
    Field(union_mode="left_to_right"),
]


class SourceTrust(StrEnum):
    UNKNOWN = "unknown"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class SourceReviewState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ArtifactRef(ContractModel):
    artifact_id: ArtifactId
    uri: str = Field(min_length=1, max_length=2048)
    media_type: str = Field(min_length=1, max_length=255)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    metadata: JsonObject = Field(default_factory=dict)


class StoreArtifactCommand(MutationCommand):
    name: ShortText
    media_type: str = Field(min_length=1, max_length=255)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    task_id: TaskId | None = None
    metadata: JsonObject = Field(default_factory=dict)


class EvidenceSource(ContractModel):
    """Immutable source identity with mutable review state represented by version."""

    source_id: SourceId
    run_id: RunId
    task_id: TaskId | None = None
    kind: EvidenceKindValue
    uri: str | None = Field(default=None, max_length=2048)
    content_sha256: Sha256
    occurrence_key: str | None = Field(default=None, min_length=1, max_length=512)
    trust: SourceTrust = SourceTrust.UNKNOWN
    review_state: SourceReviewState = SourceReviewState.PENDING
    observed_at: AwareDatetime
    recorded_at: AwareDatetime = Field(default_factory=utc_now)
    reviewed_by_agent_id: AgentId | None = None
    reviewed_at: AwareDatetime | None = None
    rejection_reason: str | None = Field(default=None, max_length=4096)
    version: int = Field(default=1, ge=1)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        reviewed = self.review_state is not SourceReviewState.PENDING
        if reviewed and (self.reviewed_by_agent_id is None or self.reviewed_at is None):
            raise ValueError("reviewed sources require reviewer and reviewed_at")
        if self.review_state is SourceReviewState.REJECTED and not self.rejection_reason:
            raise ValueError("rejected sources require a reason")
        return self


class EvidenceRef(ContractModel):
    """A precise citation to immutable material from one preserved source."""

    evidence_id: EvidenceId
    source_id: SourceId
    locator: str | None = Field(default=None, max_length=2048)
    excerpt: str | None = Field(default=None, max_length=8192)
    content_sha256: Sha256 | None = None
    artifact: ArtifactRef | None = None
    metadata: JsonObject = Field(default_factory=dict)


class Evidence(ContractModel):
    evidence_id: EvidenceId
    source_id: SourceId
    kind: EvidenceKindValue
    locator: str | None = Field(default=None, max_length=2048)
    excerpt: str | None = Field(default=None, max_length=8192)
    content_sha256: Sha256 | None = None
    artifact: ArtifactRef | None = None
    recorded_at: AwareDatetime = Field(default_factory=utc_now)
    metadata: JsonObject = Field(default_factory=dict)

    def as_ref(self) -> EvidenceRef:
        return EvidenceRef(
            evidence_id=self.evidence_id,
            source_id=self.source_id,
            locator=self.locator,
            excerpt=self.excerpt,
            content_sha256=self.content_sha256,
            artifact=self.artifact,
            metadata=self.metadata,
        )


class RegisterEvidenceSourceCommand(MutationCommand):
    kind: EvidenceKindValue
    content_sha256: Sha256
    observed_at: AwareDatetime
    task_id: TaskId | None = None
    uri: str | None = Field(default=None, max_length=2048)
    occurrence_key: str | None = Field(default=None, min_length=1, max_length=512)
    trust: SourceTrust = SourceTrust.UNKNOWN
    metadata: JsonObject = Field(default_factory=dict)


class AddEvidenceCommand(MutationCommand):
    source_id: SourceId
    kind: EvidenceKindValue
    locator: str | None = Field(default=None, max_length=2048)
    excerpt: str | None = Field(default=None, max_length=8192)
    content_sha256: Sha256 | None = None
    artifact: ArtifactRef | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ReviewSourceCommand(MutationCommand):
    source_id: SourceId
    expected_version: int = Field(ge=1)
    review_state: SourceReviewState
    reason: ContentText

    @model_validator(mode="after")
    def reject_via_explicit_command(self) -> Self:
        if self.review_state is SourceReviewState.REJECTED:
            raise ValueError("use RejectSourceCommand to reject and roll back a source")
        return self


class RejectSourceCommand(MutationCommand):
    source_id: SourceId
    expected_version: int = Field(ge=1)
    reason: ContentText


class SourceRejectionResult(ContractModel):
    source: EvidenceSource
    rolled_back_memory_ids: tuple[MemoryId, ...] = ()
    replayed: bool = False


__all__ = [
    "AddEvidenceCommand",
    "ArtifactRef",
    "Evidence",
    "EvidenceKind",
    "EvidenceKindValue",
    "EvidenceRef",
    "EvidenceSource",
    "RegisterEvidenceSourceCommand",
    "RejectSourceCommand",
    "ReviewSourceCommand",
    "Sha256",
    "SourceRejectionResult",
    "SourceReviewState",
    "SourceTrust",
    "StoreArtifactCommand",
]
