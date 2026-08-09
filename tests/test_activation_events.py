from __future__ import annotations

from datetime import UTC, datetime

import pytest

from conftest import make_actor, make_task
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.application.errors import InvalidState
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationReason,
    ActivationTrigger,
    MemoryActivationTelemetry,
    memory_activation_id,
)
from swarmbrain.domain.events import EventType
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
    RejectSourceCommand,
    ReviewSourceCommand,
    SourceReviewState,
)
from swarmbrain.domain.memory import MemoryKind, MemoryState, RememberCommand, Visibility
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import ClaimTaskCommand


@pytest.mark.asyncio
async def test_activation_event_is_content_free_and_idempotent(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    task = await kernel.add_task(make_task(scope_ids))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="activation-event-claim"),
    )
    activation_id = memory_activation_id(
        task.task_id,
        claim.lease.lease_id,
        ActivationTrigger.TASK_CLAIM,
    )
    telemetry = MemoryActivationTelemetry(
        activation_id=activation_id,
        run_id=actor.run_id,
        agent_id=actor.agent_id,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        trigger=ActivationTrigger.TASK_CLAIM,
        decision=ActivationDecision.SKIP,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        reason=ActivationReason.NO_RELEVANT_MEMORY,
        token_budget=2_048,
        min_score=0.4,
    )

    await kernel.record_memory_activation(actor, telemetry)
    await kernel.record_memory_activation(actor, telemetry)

    events = [event for event in kernel.events if event.event_type is EventType.MEMORY_ACTIVATED]
    outbox = [item for item in kernel.outbox if item.event.event_type is EventType.MEMORY_ACTIVATED]
    assert len(events) == len(outbox) == 1
    assert events[0].event_id == activation_id
    assert events[0].payload["activation_id"] == activation_id
    assert events[0].payload["decision"] == ActivationDecision.SKIP.value
    assert "query" not in events[0].model_dump_json()
    assert "content" not in events[0].model_dump_json()
    assert await kernel.get_memory_activation(actor, activation_id) == telemetry

    divergent = MemoryActivationTelemetry.model_validate(
        {
            **telemetry.model_dump(),
            "decision": ActivationDecision.DEFER,
            "reason": ActivationReason.RECALL_UNAVAILABLE,
        }
    )
    with pytest.raises(InvalidState):
        await kernel.record_memory_activation(actor, divergent)


@pytest.mark.asyncio
async def test_activation_event_rejects_an_actor_mismatch(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    other = make_actor(scope_ids)
    task = await kernel.add_task(make_task(scope_ids))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="activation-event-scope-claim"),
    )
    telemetry = MemoryActivationTelemetry(
        activation_id=memory_activation_id(
            task.task_id,
            claim.lease.lease_id,
            ActivationTrigger.TASK_CLAIM,
        ),
        run_id=actor.run_id,
        agent_id=actor.agent_id,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        trigger=ActivationTrigger.TASK_CLAIM,
        decision=ActivationDecision.SKIP,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        reason=ActivationReason.NO_RELEVANT_MEMORY,
        token_budget=2_048,
        min_score=0.4,
    )

    with pytest.raises(InvalidState):
        await kernel.record_memory_activation(other, telemetry)


@pytest.mark.asyncio
async def test_activation_event_revalidates_selected_memory(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    published = await kernel.remember(
        actor,
        RememberCommand(
            idempotency_key="activation-race-memory",
            kind=MemoryKind.INVARIANT,
            content="This memory becomes invalid before activation is committed",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
        ),
        ConservativeMemoryPolicy(),
    )
    assert published.memory is not None
    task = await kernel.add_task(make_task(scope_ids))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="activation-race-claim"),
    )
    kernel.memories[published.memory.memory_id] = published.memory.model_copy(
        update={"state": MemoryState.REFUTED}
    )
    telemetry = MemoryActivationTelemetry(
        activation_id=memory_activation_id(
            task.task_id,
            claim.lease.lease_id,
            ActivationTrigger.TASK_CLAIM,
        ),
        run_id=actor.run_id,
        agent_id=actor.agent_id,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        trigger=ActivationTrigger.TASK_CLAIM,
        decision=ActivationDecision.RECALL,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        reason=ActivationReason.CONTEXT_ACTIVATED,
        memory_ids=(published.memory.memory_id,),
        memory_versions={published.memory.memory_id: published.memory.version},
        token_budget=2_048,
        estimated_tokens=16,
        min_score=0.4,
        candidate_count=1,
    )

    with pytest.raises(InvalidState, match="no longer current and recallable"):
        await kernel.record_memory_activation(actor, telemetry)
    assert not any(event.event_type is EventType.MEMORY_ACTIVATED for event in kernel.events)
    assert not any(item.event.event_type is EventType.MEMORY_ACTIVATED for item in kernel.outbox)


@pytest.mark.asyncio
async def test_activation_version_proof_rejects_partially_revoked_evidence(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    observed_at = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
    sources = []
    evidence = []
    for suffix in ("a", "b"):
        source = await kernel.register_source(
            actor,
            RegisterEvidenceSourceCommand(
                idempotency_key=f"activation-proof-source-{suffix}",
                kind=EvidenceKind.DOCUMENT,
                content_sha256=suffix * 64,
                occurrence_key=f"activation-proof-{suffix}",
                observed_at=observed_at,
            ),
        )
        item = await kernel.add_evidence(
            actor,
            AddEvidenceCommand(
                idempotency_key=f"activation-proof-evidence-{suffix}",
                source_id=source.source_id,
                kind=EvidenceKind.DOCUMENT,
                excerpt=f"trusted evidence {suffix}",
                content_sha256=source.content_sha256,
            ),
        )
        sources.append(source)
        evidence.append(item)

    published = await kernel.remember(
        actor,
        RememberCommand(
            idempotency_key="activation-version-proof-memory",
            kind=MemoryKind.INVARIANT,
            content="Both citations were rendered into the selected memory block",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            evidence=tuple(item.as_ref() for item in evidence),
        ),
        ConservativeMemoryPolicy(),
    )
    assert published.memory is not None
    task = await kernel.add_task(make_task(scope_ids, title="Use both citations"))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="activation-version-proof-claim"),
    )
    telemetry = MemoryActivationTelemetry(
        activation_id=memory_activation_id(
            task.task_id,
            claim.lease.lease_id,
            ActivationTrigger.TASK_CLAIM,
        ),
        run_id=actor.run_id,
        agent_id=actor.agent_id,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        trigger=ActivationTrigger.TASK_CLAIM,
        decision=ActivationDecision.RECALL,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        reason=ActivationReason.CONTEXT_ACTIVATED,
        memory_ids=(published.memory.memory_id,),
        memory_versions={published.memory.memory_id: published.memory.version},
        token_budget=2_048,
        estimated_tokens=32,
        min_score=0.4,
        candidate_count=1,
    )

    await kernel.reject_source(
        actor,
        RejectSourceCommand(
            idempotency_key="activation-proof-reject-a",
            source_id=sources[0].source_id,
            expected_version=sources[0].version,
            reason="citation A was revoked",
        ),
    )
    current = kernel.memories[published.memory.memory_id]
    assert current.state is MemoryState.CONFIRMED
    assert current.version == published.memory.version + 1
    assert current.evidence == (evidence[1].as_ref(),)
    with pytest.raises(InvalidState, match="rejection is terminal"):
        await kernel.review_source(
            actor,
            ReviewSourceCommand(
                idempotency_key="activation-proof-reapprove-a",
                source_id=sources[0].source_id,
                expected_version=sources[0].version + 1,
                review_state=SourceReviewState.APPROVED,
                reason="attempt to restore a rolled-back source",
            ),
        )

    with pytest.raises(InvalidState, match="no longer current and recallable"):
        await kernel.record_memory_activation(actor, telemetry)
    assert not any(event.event_type is EventType.MEMORY_ACTIVATED for event in kernel.events)
