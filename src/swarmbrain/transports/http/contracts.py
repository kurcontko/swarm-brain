"""HTTP body schemas derived from the authoritative domain commands."""

from __future__ import annotations

from copy import deepcopy

from pydantic import AwareDatetime, BaseModel, Field, create_model

from swarmbrain.domain.common import ContractModel, IdempotencyKey, JsonObject, TaskId
from swarmbrain.domain.conflicts import ReportConflictCommand, ResolveConflictCommand
from swarmbrain.domain.evidence import AddEvidenceCommand, EvidenceKindValue, Sha256
from swarmbrain.domain.leases import RenewLeaseCommand
from swarmbrain.domain.memory import RememberCommand
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    ClaimTaskCommand,
    CompleteTaskCommand,
    ReleaseTaskCommand,
)


def _body_schema(
    name: str,
    command: type[BaseModel],
    *,
    path_fields: frozenset[str] = frozenset(),
) -> type[ContractModel]:
    """Make header/path-injected fields optional while retaining constraints."""

    definitions = {}
    optional_fields = {"idempotency_key", *path_fields}
    for field_name, source in command.model_fields.items():
        if field_name in optional_fields:
            # These values are compatibility echoes only. The transport checks
            # them against the authoritative header/path, then the domain
            # command applies the canonical UUID/key validators.
            definitions[field_name] = (str | None, None)
        else:
            field = deepcopy(source)
            definitions[field_name] = (field.annotation, field)
    return create_model(name, __base__=ContractModel, **definitions)


ClaimTaskBody = _body_schema("ClaimTaskBody", ClaimTaskCommand)
RenewLeaseBody = _body_schema(
    "RenewLeaseBody", RenewLeaseCommand, path_fields=frozenset({"lease_id"})
)
RememberBody = _body_schema("RememberBody", RememberCommand)
CheckpointBody = _body_schema(
    "CheckpointBody", CheckpointCommand, path_fields=frozenset({"task_id"})
)
CompleteTaskBody = _body_schema(
    "CompleteTaskBody", CompleteTaskCommand, path_fields=frozenset({"task_id"})
)
ReleaseTaskBody = _body_schema(
    "ReleaseTaskBody", ReleaseTaskCommand, path_fields=frozenset({"task_id"})
)
ReportConflictBody = _body_schema("ReportConflictBody", ReportConflictCommand)
ResolveConflictBody = _body_schema(
    "ResolveConflictBody",
    ResolveConflictCommand,
    path_fields=frozenset({"conflict_id"}),
)


class IngestSourceBody(ContractModel):
    """Public source material only; trust and extraction policy are operator-owned."""

    kind: EvidenceKindValue
    content: str = Field(min_length=1, max_length=1_000_000)
    observed_at: AwareDatetime | None = None
    task_id: TaskId | None = None
    uri: str | None = Field(default=None, max_length=2048)
    occurrence_key: str | None = Field(default=None, min_length=1, max_length=512)
    metadata: JsonObject = Field(default_factory=dict)
    idempotency_key: IdempotencyKey | None = None


class RegisterEvidenceSourceBody(ContractModel):
    """Unreviewed evidence source; callers cannot self-assign trust."""

    kind: EvidenceKindValue
    content_sha256: Sha256
    observed_at: AwareDatetime
    task_id: TaskId | None = None
    uri: str | None = Field(default=None, max_length=2048)
    occurrence_key: str | None = Field(default=None, min_length=1, max_length=512)
    metadata: JsonObject = Field(default_factory=dict)
    idempotency_key: IdempotencyKey | None = None


AddEvidenceBody = _body_schema("AddEvidenceBody", AddEvidenceCommand)


__all__ = [
    "AddEvidenceBody",
    "CheckpointBody",
    "ClaimTaskBody",
    "CompleteTaskBody",
    "IngestSourceBody",
    "ReleaseTaskBody",
    "RememberBody",
    "RegisterEvidenceSourceBody",
    "RenewLeaseBody",
    "ReportConflictBody",
    "ResolveConflictBody",
]
