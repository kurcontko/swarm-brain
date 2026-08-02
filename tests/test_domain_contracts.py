from __future__ import annotations

import inspect
from datetime import datetime

import pytest
from pydantic import ValidationError

from conftest import make_actor, new_id
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.domain import conflicts, events, evidence, leases, memory, tasks
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import ContractModel, MutationCommand
from swarmbrain.domain.memory import RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    ClaimTaskCommand,
    CompleteTaskCommand,
)
from swarmbrain.ports.coordination_store import CoordinationStore
from swarmbrain.ports.memory_store import (
    ConflictStore,
    EvidenceStore,
    MemoryReviewStore,
    MemoryStore,
)


def _domain_models() -> list[type[ContractModel]]:
    modules = (conflicts, evidence, events, leases, memory, tasks)
    return sorted(
        {
            value
            for module in modules
            for value in vars(module).values()
            if inspect.isclass(value)
            and issubclass(value, ContractModel)
            and value not in {ContractModel, MutationCommand}
            and value.__module__ == module.__name__
        },
        key=lambda model: model.__name__,
    )


def test_all_domain_models_generate_json_schema() -> None:
    models = _domain_models()
    assert len(models) >= 50
    for model in models:
        assert model.model_json_schema()["title"] == model.__name__


def test_every_mutation_command_requires_idempotency() -> None:
    commands = [model for model in _domain_models() if issubclass(model, MutationCommand)]
    assert len(commands) >= 15
    for command in commands:
        field = command.model_fields["idempotency_key"]
        assert field.is_required()


def test_model_visible_contracts_have_no_auth_owned_identity_fields() -> None:
    forbidden = {
        "tenant_id",
        "project_id",
        "repository_id",
        "repo_id",
        "run_id",
        "agent_id",
        "author_agent_id",
        "provider",
        "model",
        "capabilities",
        "token",
    }
    visible = (
        ClaimTaskCommand,
        RecallQuery,
        RememberCommand,
        CheckpointCommand,
        CompleteTaskCommand,
        conflicts.ReportConflictCommand,
    )
    for contract in visible:
        assert forbidden.isdisjoint(contract.model_fields)


def test_contracts_reject_extra_fields_bad_uuid_and_naive_time(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ActorContext.model_validate({**actor.model_dump(), "spoofed": True})
    with pytest.raises(ValidationError, match="valid UUID string"):
        ClaimTaskCommand(idempotency_key="claim", task_id="not-a-uuid")
    with pytest.raises(ValidationError, match="timezone"):
        RememberCommand(
            idempotency_key="remember",
            kind="observation",
            content="fact",
            valid_from=datetime(2026, 8, 2, 12, 0),
        )
    with pytest.raises(ValidationError, match="task-visible"):
        RememberCommand(
            idempotency_key="remember",
            kind="observation",
            content="fact",
            visibility=Visibility.TASK,
        )


def test_in_memory_kernel_satisfies_all_core_runtime_ports() -> None:
    kernel = InMemoryKernel()
    assert isinstance(kernel, CoordinationStore)
    assert isinstance(kernel, MemoryStore)
    assert isinstance(kernel, MemoryReviewStore)
    assert isinstance(kernel, EvidenceStore)
    assert isinstance(kernel, ConflictStore)


def test_identifiers_are_canonical_uuid_strings() -> None:
    upper = new_id().upper()
    command = ClaimTaskCommand(idempotency_key="claim", task_id=upper)
    assert command.task_id == upper.lower()
