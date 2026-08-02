"""Shared primitives for Swarm Brain domain contracts.

The public identifiers in Swarm Brain are canonical UUID strings.  Keeping
them as strings makes the HTTP, MCP, and SQL boundaries agree while the
validator still rejects ambiguous or malformed identifiers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, TypeAlias
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for model defaults."""

    return datetime.now(UTC)


def _canonical_uuid_string(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)
    if not isinstance(value, str):
        raise ValueError("identifier must be a UUID string")
    try:
        return str(UUID(value.strip()))
    except (AttributeError, ValueError) as exc:
        raise ValueError("identifier must be a valid UUID string") from exc


UUIDString = Annotated[str, BeforeValidator(_canonical_uuid_string)]
IdempotencyKey = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]
ContentText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
JsonObject: TypeAlias = dict[str, Any]

# Semantic aliases preserve intent in signatures without introducing wrapper
# objects that would serialize differently at API boundaries.
TenantId: TypeAlias = UUIDString
ProjectId: TypeAlias = UUIDString
RepositoryId: TypeAlias = UUIDString
SwarmId: TypeAlias = UUIDString
RunId: TypeAlias = UUIDString
AgentId: TypeAlias = UUIDString
TaskId: TypeAlias = UUIDString
LeaseId: TypeAlias = UUIDString
CheckpointId: TypeAlias = UUIDString
MemoryId: TypeAlias = UUIDString
EvidenceId: TypeAlias = UUIDString
SourceId: TypeAlias = UUIDString
ArtifactId: TypeAlias = UUIDString
ConflictId: TypeAlias = UUIDString
EventId: TypeAlias = UUIDString


class ContractModel(BaseModel):
    """Strict, immutable base class for boundary-safe domain values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        populate_by_name=True,
    )


class MutationCommand(ContractModel):
    """Base for retry-safe mutation requests.

    The caller creates this key before its first attempt and reuses the same
    key for every retry, including after an ambiguous database commit.
    """

    idempotency_key: IdempotencyKey


__all__ = [
    "AgentId",
    "ArtifactId",
    "AwareDatetime",
    "CheckpointId",
    "Confidence",
    "ConflictId",
    "ContentText",
    "ContractModel",
    "EventId",
    "EvidenceId",
    "IdempotencyKey",
    "JsonObject",
    "LeaseId",
    "MemoryId",
    "MutationCommand",
    "ProjectId",
    "RepositoryId",
    "RunId",
    "ShortText",
    "SourceId",
    "SwarmId",
    "TaskId",
    "TenantId",
    "UUIDString",
    "utc_now",
]
