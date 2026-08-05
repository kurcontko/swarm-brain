"""Raw-source ingestion and provider-neutral extraction contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .common import (
    ContractModel,
    JsonObject,
    MemoryContent,
    MutationCommand,
    SemanticLabel,
    SourceId,
    TaskId,
    UUIDString,
    utc_now,
)
from .evidence import EvidenceKind, EvidenceKindValue, EvidenceSource, Sha256, SourceTrust
from .memory import MemoryKindValue

GENERAL_SOURCE_KINDS = frozenset(
    {
        EvidenceKind.COMMAND_OUTPUT,
        EvidenceKind.TEST_RESULT,
        EvidenceKind.LOG,
        EvidenceKind.MESSAGE,
        EvidenceKind.DOCUMENT,
    }
)
EXTRACTABLE_SOURCE_KINDS = frozenset({EvidenceKind.SOURCE_CODE, *GENERAL_SOURCE_KINDS})

ExtractorName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


def _bounded_candidate_content(value: MemoryContent) -> MemoryContent:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > 32_768:
        raise ValueError("candidate content exceeds 32768 encoded bytes")
    return value


CandidateContent = Annotated[MemoryContent, AfterValidator(_bounded_candidate_content)]


class ExtractionRoute(StrEnum):
    CODING = "coding"
    GENERAL = "general"
    SKIP = "skip"


class ExtractionStage(StrEnum):
    DETERMINISTIC = "deterministic"
    PROVIDER = "provider"
    VALIDATION = "validation"
    APPLY = "apply"


class ExtractionOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FALLBACK = "fallback"
    REJECTED = "rejected"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExtractionRunStatus(StrEnum):
    COMPLETED = "completed"
    FALLBACK = "fallback"
    SKIPPED = "skipped"
    FAILED = "failed"


class SourceChunk(ContractModel):
    chunk_id: UUIDString
    source_id: SourceId
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1, max_length=262_144)
    content_sha256: Sha256
    token_count: int = Field(default=0, ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_and_offsets(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be later than char_start")
        if self.char_end - self.char_start != len(self.content):
            raise ValueError("chunk offsets must match content length")
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if digest != self.content_sha256:
            raise ValueError("chunk content_sha256 does not match content")
        return self


class SourceSpan(ContractModel):
    """One exact source quotation, relative to an immutable raw source."""

    chunk_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    excerpt: str = Field(min_length=1, max_length=8192)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be later than char_start")
        if self.char_end - self.char_start != len(self.excerpt):
            raise ValueError("span offsets must match excerpt length")
        return self


class IngestRawSourceCommand(MutationCommand):
    """Preserve exact source bytes before any fallible extraction work."""

    kind: EvidenceKindValue
    content: str = Field(min_length=1, max_length=1_000_000)
    observed_at: AwareDatetime | None = None
    task_id: TaskId | None = None
    uri: str | None = Field(default=None, max_length=2048)
    occurrence_key: str | None = Field(default=None, min_length=1, max_length=512)
    trust: SourceTrust = SourceTrust.UNKNOWN
    use_provider: bool = False
    chunk_chars: int = Field(default=65_536, ge=1024, le=262_144)
    priority: int = Field(default=0, ge=-1000, le=1000)
    metadata: JsonObject = Field(default_factory=dict)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class PreparedSourceIngest(ContractModel):
    """Application-prepared source, chunks, and extraction work identity."""

    source_id: SourceId
    work_id: UUIDString
    command: IngestRawSourceCommand
    chunks: tuple[SourceChunk, ...] = Field(min_length=1, max_length=4096)
    extractor_name: ExtractorName
    extractor_revision: str = Field(min_length=1, max_length=255)
    embedding_model: SemanticLabel | None = None
    work_dedupe_key: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        indexes = [chunk.chunk_index for chunk in self.chunks]
        if indexes != list(range(len(self.chunks))):
            raise ValueError("source chunk indexes must be contiguous and ordered")
        if any(chunk.source_id != self.source_id for chunk in self.chunks):
            raise ValueError("all chunks must belong to source_id")
        expected_start = 0
        for chunk in self.chunks:
            if chunk.char_start != expected_start:
                raise ValueError("source chunk offsets must be contiguous and ordered")
            expected_start = chunk.char_end
        reconstructed = "".join(chunk.content for chunk in self.chunks)
        if reconstructed != self.command.content:
            raise ValueError("source chunks must reconstruct the exact raw content")
        return self


class SourceIngestResult(ContractModel):
    """Lean durable receipt; raw chunk content remains only in source storage."""

    source: EvidenceSource
    work_id: UUIDString
    chunk_count: int = Field(ge=1, le=4096)
    replayed: bool = False

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_chunk_receipt(cls, value: object) -> object:
        """Read pre-v6 idempotency responses without re-emitting raw chunks."""

        if not isinstance(value, Mapping) or "chunk_count" in value or "chunks" not in value:
            return value
        upgraded = dict(value)
        chunks = upgraded.pop("chunks")
        if not isinstance(chunks, (list, tuple)):
            return value
        upgraded["chunk_count"] = len(chunks)
        return upgraded


class ExtractionCandidate(ContractModel):
    """Untrusted proposal; storage-owned identity and lifecycle are impossible to set."""

    kind: MemoryKindValue
    content: CandidateContent
    title: str | None = Field(default=None, max_length=500)
    tags: tuple[str, ...] = Field(default=(), max_length=64)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    spans: tuple[SourceSpan, ...] = Field(default=(), max_length=32)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.strip().lower() for item in value if item.strip()))

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.valid_from is None and self.valid_to is not None:
            raise ValueError("valid_to requires valid_from")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            raise ValueError("valid_to must be later than valid_from")
        return self


class ProviderDescriptor(ContractModel):
    provider: str = Field(min_length=1, max_length=255)
    model: str = Field(min_length=1, max_length=255)
    revision: str | None = Field(default=None, max_length=255)
    prompt_id: str | None = Field(default=None, max_length=255)
    prompt_sha256: Sha256 | None = None


class ExtractionProvenance(ContractModel):
    stage: ExtractionStage
    attempt: int = Field(ge=1)
    outcome: ExtractionOutcome
    input_sha256: Sha256
    output_sha256: Sha256 | None = None
    provider: str | None = Field(default=None, max_length=255)
    model: str | None = Field(default=None, max_length=255)
    revision: str | None = Field(default=None, max_length=255)
    prompt_id: str | None = Field(default=None, max_length=255)
    prompt_sha256: Sha256 | None = None
    candidate_count: int = Field(default=0, ge=0)
    error: str | None = Field(default=None, max_length=1024)
    started_at: AwareDatetime = Field(default_factory=utc_now)
    finished_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        return self


class ExtractionInput(ContractModel):
    work_id: UUIDString
    attempt: int = Field(ge=1)
    source: EvidenceSource
    raw_content: str = Field(min_length=1, max_length=1_000_000)
    chunks: tuple[SourceChunk, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if any(chunk.source_id != self.source.source_id for chunk in self.chunks):
            raise ValueError("input chunks must belong to the source")
        if "".join(chunk.content for chunk in self.chunks) != self.raw_content:
            raise ValueError("input chunks must reconstruct raw_content")
        digest = hashlib.sha256(self.raw_content.encode("utf-8")).hexdigest()
        if digest != self.source.content_sha256:
            raise ValueError("raw_content does not match source content_sha256")
        return self


class ExtractionResult(ContractModel):
    route: ExtractionRoute
    status: ExtractionRunStatus
    candidates: tuple[ExtractionCandidate, ...] = ()
    deterministic_candidates: tuple[ExtractionCandidate, ...] = ()
    provider_candidates: tuple[ExtractionCandidate, ...] = ()
    provenance: tuple[ExtractionProvenance, ...] = Field(min_length=1)
    fallback_reason: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def validate_fallback(self) -> Self:
        if self.status is ExtractionRunStatus.FALLBACK and not self.fallback_reason:
            raise ValueError("fallback extraction requires fallback_reason")
        return self


__all__ = [
    "CandidateContent",
    "EXTRACTABLE_SOURCE_KINDS",
    "ExtractionCandidate",
    "ExtractionInput",
    "ExtractionOutcome",
    "ExtractionProvenance",
    "ExtractionResult",
    "ExtractionRoute",
    "ExtractionRunStatus",
    "ExtractionStage",
    "ExtractorName",
    "GENERAL_SOURCE_KINDS",
    "IngestRawSourceCommand",
    "PreparedSourceIngest",
    "ProviderDescriptor",
    "SourceChunk",
    "SourceIngestResult",
    "SourceSpan",
]
