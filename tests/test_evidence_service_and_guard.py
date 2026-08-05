"""Evidence trust authorization and confirmed-memory supersession guard."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from conftest import make_actor
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.application.errors import Forbidden
from swarmbrain.application.evidence_service import EvidenceService
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.domain.agents import Capability
from swarmbrain.domain.evidence import RegisterEvidenceSourceCommand, SourceTrust
from swarmbrain.domain.memory import MemoryOperation, MemoryState, RememberCommand


@pytest.mark.asyncio
async def test_source_trust_requires_review_capability(scope_ids: dict[str, str]) -> None:
    kernel = InMemoryKernel()
    service = EvidenceService(kernel)
    publisher = make_actor(
        scope_ids,
        capabilities=frozenset({Capability.MEMORY_PUBLISH.value}),
    )
    command = RegisterEvidenceSourceCommand(
        idempotency_key="trusted-source-1",
        kind="org.acme/trace",
        content_sha256="a" * 64,
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
        trust=SourceTrust.TRUSTED,
    )

    with pytest.raises(Forbidden):
        await service.register_source(publisher, command)

    unknown = await service.register_source(
        publisher,
        command.model_copy(
            update={
                "idempotency_key": "unknown-source-1",
                "trust": SourceTrust.UNKNOWN,
            }
        ),
    )
    assert unknown.trust is SourceTrust.UNKNOWN


@pytest.mark.asyncio
async def test_confirmed_memory_is_protected_even_without_existing_evidence(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    memory = MemoryService(kernel, ConservativeMemoryPolicy(), review_store=kernel)
    verifier = make_actor(scope_ids)
    publisher = make_actor(
        scope_ids,
        capabilities=frozenset({Capability.MEMORY_PUBLISH.value}),
    )

    confirmed = await memory.publish(
        verifier,
        RememberCommand(
            idempotency_key="confirmed-without-evidence",
            kind="org.acme/invariant",
            content={"amounts": {"unit": "cents", "representation": "integer"}},
            desired_state=MemoryState.CONFIRMED,
        ),
    )
    assert confirmed.memory is not None

    attempted = await memory.publish(
        publisher,
        RememberCommand(
            idempotency_key="tentative-over-confirmed",
            kind="org.acme/invariant",
            content={"amounts": {"unit": "dollars", "representation": "float"}},
            supersedes_memory_id=confirmed.memory.memory_id,
        ),
    )

    assert attempted.operation is MemoryOperation.NOOP
    assert attempted.memory is None
    assert kernel.memories[confirmed.memory.memory_id].state is MemoryState.CONFIRMED
