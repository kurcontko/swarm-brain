from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from math import nan

import pytest
from pydantic import ValidationError

from conftest import make_actor, new_id
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.domain import conflicts, events, evidence, leases, memory, tasks
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import ContractModel, MutationCommand
from swarmbrain.domain.evidence import EvidenceRef
from swarmbrain.domain.memory import (
    MemoryKind,
    MemoryLink,
    RecallQuery,
    RememberCommand,
    Visibility,
)
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


def test_occurrence_time_is_provenance_backed_and_query_prior_is_bounded() -> None:
    observed_at = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
    occurred_at = observed_at - timedelta(days=30)
    with pytest.raises(ValidationError, match="occurred_at requires immutable evidence"):
        RememberCommand(
            idempotency_key="unproved-occurrence",
            kind="observation",
            content="retrospective claim",
            occurred_at=occurred_at,
            valid_from=observed_at,
        )

    command = RememberCommand(
        idempotency_key="proved-occurrence",
        kind="observation",
        content="retrospective claim",
        evidence=(EvidenceRef(evidence_id=new_id(), source_id=new_id()),),
        occurred_at=occurred_at,
        valid_from=observed_at,
    )
    assert command.occurred_at == occurred_at
    assert command.valid_from == observed_at

    with pytest.raises(ValidationError, match="must be provided together"):
        RecallQuery(text="when", occurrence_time_prior_from=occurred_at)
    with pytest.raises(ValidationError, match="must be later"):
        RecallQuery(
            text="when",
            occurrence_time_prior_from=observed_at,
            occurrence_time_prior_to=occurred_at,
        )


def test_recall_query_text_is_bounded_unlike_stored_content() -> None:
    # Query text reaches the embedding provider verbatim, so the contract caps
    # it; stored memory content deliberately has no such cap.
    assert RecallQuery(text="q" * 8_192).text == "q" * 8_192
    with pytest.raises(ValidationError, match="at most 8192 characters"):
        RecallQuery(text="q" * 8_193)


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


def test_memory_contract_accepts_json_documents_and_open_semantic_labels() -> None:
    command = RememberCommand(
        idempotency_key="preference-1",
        kind="org.acme/preference",
        content={
            "subject": "editor",
            "preference": {"name": "vim", "strength": 0.8},
            "alternatives": ["helix", "zed"],
        },
    )
    built_in = RememberCommand(
        idempotency_key="observation-1",
        kind="observation",
        content=["ordered", {"value": 2}],
    )
    link = MemoryLink(
        link_id=new_id(),
        source_memory_id=new_id(),
        target_memory_id=new_id(),
        kind="org.acme/caused_by",
    )

    assert command.kind == "org.acme/preference"
    assert command.content["preference"]["name"] == "vim"
    assert built_in.kind is MemoryKind.OBSERVATION
    assert link.kind == "org.acme/caused_by"
    with pytest.raises(ValidationError):
        RememberCommand(idempotency_key="null-1", kind="note", content=None)
    with pytest.raises(ValidationError):
        RememberCommand(
            idempotency_key="nan-1",
            kind="measurement",
            content={"value": nan},
        )
