"""HTTP body schemas derived from the authoritative domain commands."""

from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, create_model

from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.conflicts import ReportConflictCommand, ResolveConflictCommand
from swarmbrain.domain.evidence import AddEvidenceCommand, RegisterEvidenceSourceCommand
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
RegisterEvidenceSourceBody = _body_schema(
    "RegisterEvidenceSourceBody", RegisterEvidenceSourceCommand
)
AddEvidenceBody = _body_schema("AddEvidenceBody", AddEvidenceCommand)
ReportConflictBody = _body_schema("ReportConflictBody", ReportConflictCommand)
ResolveConflictBody = _body_schema(
    "ResolveConflictBody",
    ResolveConflictCommand,
    path_fields=frozenset({"conflict_id"}),
)


__all__ = [
    "AddEvidenceBody",
    "CheckpointBody",
    "ClaimTaskBody",
    "CompleteTaskBody",
    "RegisterEvidenceSourceBody",
    "ReleaseTaskBody",
    "RememberBody",
    "RenewLeaseBody",
    "ReportConflictBody",
    "ResolveConflictBody",
]
