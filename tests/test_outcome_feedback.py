from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_actor, make_task, new_id
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationReason,
    ActivationTrigger,
    MemoryActivationTelemetry,
    memory_activation_id,
)
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
    RejectSourceCommand,
)
from swarmbrain.domain.memory import (
    MemoryKind,
    MemoryState,
    RecallQuery,
    RememberCommand,
    Visibility,
)
from swarmbrain.domain.outcome_feedback import OutcomeAssociationKind
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import (
    ClaimTaskCommand,
    CompleteTaskCommand,
    TaskOutcome,
)


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


async def _publish_confirmed(
    kernel: InMemoryKernel,
    actor: object,
    *,
    key: str,
) -> object:
    result = await kernel.remember(
        actor,
        RememberCommand(
            idempotency_key=key,
            kind=MemoryKind.PROCEDURE,
            content=f"content intentionally absent from outcome telemetry {key}",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
        ),
        ConservativeMemoryPolicy(),
    )
    assert result.memory is not None
    return result.memory


async def _record_activation(
    kernel: InMemoryKernel,
    actor: object,
    *,
    task_id: str,
    lease_id: str,
    memory: object,
    trigger: ActivationTrigger = ActivationTrigger.TASK_CLAIM,
) -> None:
    await kernel.record_memory_activation(
        actor,
        MemoryActivationTelemetry(
            activation_id=memory_activation_id(task_id, lease_id, trigger),
            run_id=actor.run_id,
            agent_id=actor.agent_id,
            task_id=task_id,
            lease_id=lease_id,
            trigger=trigger,
            decision=ActivationDecision.RECALL,
            purpose=RetrievalPurpose.TASK_BOOTSTRAP,
            reason=ActivationReason.CONTEXT_ACTIVATED,
            memory_ids=(memory.memory_id,),
            memory_versions={memory.memory_id: memory.version},
            token_budget=2_048,
            estimated_tokens=16,
            min_score=0.4,
            candidate_count=1,
        ),
    )


@pytest.mark.parametrize("outcome", [TaskOutcome.SUCCEEDED, TaskOutcome.FAILED])
@pytest.mark.asyncio
async def test_completion_records_only_a_proven_observational_association(
    scope_ids: dict[str, str],
    outcome: TaskOutcome,
) -> None:
    clock = MutableClock()
    kernel = InMemoryKernel(clock=clock)
    actor = make_actor(scope_ids)
    memory = await _publish_confirmed(kernel, actor, key=f"publish-{outcome.value}")
    task = await kernel.add_task(make_task(scope_ids, created_at=clock()))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key=f"claim-{outcome.value}", task_id=task.task_id),
    )
    await _record_activation(
        kernel,
        actor,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        memory=memory,
    )
    recall_before = await kernel.recall(actor, RecallQuery(text="outcome telemetry"))

    result = await kernel.complete_task(
        actor,
        CompleteTaskCommand(
            idempotency_key=f"complete-{outcome.value}",
            task_id=task.task_id,
            lease_id=claim.lease.lease_id,
            expected_task_version=claim.task.version,
            expected_lease_version=claim.lease.version,
            outcome=outcome,
            summary="Outcome summary must never be copied into the association",
            memory_ids=(memory.memory_id,),
        ),
    )

    associations = await kernel.list_memory_outcome_associations(actor)
    assert len(associations) == 1
    association = associations[0]
    assert association.kind is OutcomeAssociationKind.OBSERVATIONAL_SILVER
    assert association.outcome is outcome
    assert association.memory_id == memory.memory_id
    assert association.memory_version == memory.version
    assert association.task_id == task.task_id
    assert association.lease_id == claim.lease.lease_id
    assert association.consumer_agent_id == actor.agent_id
    assert association.observed_at == result.completion.completed_at
    serialized = association.model_dump_json()
    for forbidden in ("query", "content", "summary", "causal"):
        assert forbidden not in serialized
    assert kernel.memories[memory.memory_id].state is MemoryState.CONFIRMED
    recall_after = await kernel.recall(actor, RecallQuery(text="outcome telemetry"))
    assert recall_after.hits == recall_before.hits


@pytest.mark.asyncio
async def test_recall_or_citation_alone_and_uncited_activation_are_not_use(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    memory = await _publish_confirmed(kernel, actor, key="publish-not-use")

    cited_task = await kernel.add_task(make_task(scope_ids, title="Citation only"))
    cited_claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="claim-citation-only", task_id=cited_task.task_id),
    )
    await kernel.complete_task(
        actor,
        CompleteTaskCommand(
            idempotency_key="complete-citation-only",
            task_id=cited_task.task_id,
            lease_id=cited_claim.lease.lease_id,
            expected_task_version=cited_claim.task.version,
            expected_lease_version=cited_claim.lease.version,
            summary="Cited without an activation",
            memory_ids=(memory.memory_id,),
        ),
    )

    activated_task = await kernel.add_task(make_task(scope_ids, title="Activation only"))
    activated_claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(
            idempotency_key="claim-activation-only",
            task_id=activated_task.task_id,
        ),
    )
    await _record_activation(
        kernel,
        actor,
        task_id=activated_task.task_id,
        lease_id=activated_claim.lease.lease_id,
        memory=memory,
    )
    await kernel.complete_task(
        actor,
        CompleteTaskCommand(
            idempotency_key="complete-activation-only",
            task_id=activated_task.task_id,
            lease_id=activated_claim.lease.lease_id,
            expected_task_version=activated_claim.task.version,
            expected_lease_version=activated_claim.lease.version,
            summary="Activated but deliberately uncited",
        ),
    )

    assert await kernel.list_memory_outcome_associations(actor) == ()


@pytest.mark.asyncio
async def test_activation_from_an_old_lease_or_other_consumer_is_not_use(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    kernel = InMemoryKernel(clock=clock)
    first = make_actor(scope_ids)
    second = make_actor(scope_ids)
    memory = await _publish_confirmed(kernel, first, key="publish-handoff")
    task = await kernel.add_task(make_task(scope_ids, created_at=clock()))
    first_claim = await kernel.claim_task(
        first,
        ClaimTaskCommand(
            idempotency_key="claim-first-consumer",
            task_id=task.task_id,
            lease_seconds=15,
        ),
    )
    await _record_activation(
        kernel,
        first,
        task_id=task.task_id,
        lease_id=first_claim.lease.lease_id,
        memory=memory,
    )
    clock.advance(seconds=16)
    second_claim = await kernel.claim_task(
        second,
        ClaimTaskCommand(idempotency_key="claim-second-consumer", task_id=task.task_id),
    )

    result = await kernel.complete_task(
        second,
        CompleteTaskCommand(
            idempotency_key="complete-second-consumer",
            task_id=task.task_id,
            lease_id=second_claim.lease.lease_id,
            expected_task_version=second_claim.task.version,
            expected_lease_version=second_claim.lease.version,
            outcome=TaskOutcome.FAILED,
            summary="Old consumer activation cannot prove this consumer used memory",
            memory_ids=(memory.memory_id,),
        ),
    )

    assert result.completion.outcome is TaskOutcome.FAILED
    assert await kernel.list_memory_outcome_associations(second) == ()


@pytest.mark.parametrize("mutation", ["version", "stale", "revoked"])
@pytest.mark.asyncio
async def test_changed_or_revoked_memory_proof_is_omitted_without_breaking_completion(
    scope_ids: dict[str, str],
    mutation: str,
) -> None:
    clock = MutableClock()
    kernel = InMemoryKernel(clock=clock)
    actor = make_actor(scope_ids)
    memory = await _publish_confirmed(kernel, actor, key=f"publish-{mutation}")
    task = await kernel.add_task(make_task(scope_ids, created_at=clock()))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key=f"claim-{mutation}", task_id=task.task_id),
    )
    await _record_activation(
        kernel,
        actor,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        memory=memory,
    )
    clock.advance(seconds=1)
    updates: dict[str, object] = {"version": memory.version + 1}
    if mutation == "stale":
        updates["recorded_to"] = clock()
    elif mutation == "revoked":
        updates["state"] = MemoryState.REFUTED
    kernel.memories[memory.memory_id] = memory.model_copy(update=updates)

    result = await kernel.complete_task(
        actor,
        CompleteTaskCommand(
            idempotency_key=f"complete-{mutation}",
            task_id=task.task_id,
            lease_id=claim.lease.lease_id,
            expected_task_version=claim.task.version,
            expected_lease_version=claim.lease.version,
            summary="Completion remains authoritative; the proof is simply omitted",
            memory_ids=(memory.memory_id,),
        ),
    )

    assert result.completion.outcome is TaskOutcome.SUCCEEDED
    assert await kernel.list_memory_outcome_associations(actor) == ()


@pytest.mark.asyncio
async def test_two_activated_versions_make_an_unversioned_citation_ambiguous(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    memory = await _publish_confirmed(kernel, actor, key="publish-ambiguous")
    task = await kernel.add_task(make_task(scope_ids))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="claim-ambiguous", task_id=task.task_id),
    )
    await _record_activation(
        kernel,
        actor,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        memory=memory,
    )
    current = memory.model_copy(update={"version": memory.version + 1})
    kernel.memories[memory.memory_id] = current
    await _record_activation(
        kernel,
        actor,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        memory=current,
        trigger=ActivationTrigger.EXPLICIT,
    )

    await kernel.complete_task(
        actor,
        CompleteTaskCommand(
            idempotency_key="complete-ambiguous",
            task_id=task.task_id,
            lease_id=claim.lease.lease_id,
            expected_task_version=claim.task.version,
            expected_lease_version=claim.lease.version,
            summary="Citation has no version and cannot identify either rendering",
            memory_ids=(memory.memory_id,),
        ),
    )

    assert await kernel.list_memory_outcome_associations(actor) == ()


@pytest.mark.asyncio
async def test_source_revocation_after_activation_omits_proof_but_not_completion(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    source = await kernel.register_source(
        actor,
        RegisterEvidenceSourceCommand(
            idempotency_key="outcome-source",
            kind=EvidenceKind.DOCUMENT,
            content_sha256="a" * 64,
            occurrence_key="outcome-source-occurrence",
            observed_at=datetime(2026, 8, 9, 12, 0, tzinfo=UTC),
        ),
    )
    evidence = await kernel.add_evidence(
        actor,
        AddEvidenceCommand(
            idempotency_key="outcome-evidence",
            source_id=source.source_id,
            kind=EvidenceKind.DOCUMENT,
            excerpt="Evidence accepted at activation time",
            content_sha256=source.content_sha256,
        ),
    )
    published = await kernel.remember(
        actor,
        RememberCommand(
            idempotency_key="outcome-evidenced-memory",
            kind=MemoryKind.INVARIANT,
            content="A source-backed memory whose source is later revoked",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            evidence=(evidence.as_ref(),),
        ),
        ConservativeMemoryPolicy(),
    )
    assert published.memory is not None
    memory = published.memory
    task = await kernel.add_task(make_task(scope_ids))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="outcome-revocation-claim", task_id=task.task_id),
    )
    await _record_activation(
        kernel,
        actor,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        memory=memory,
    )
    await kernel.reject_source(
        actor,
        RejectSourceCommand(
            idempotency_key="outcome-revoke-source",
            source_id=source.source_id,
            expected_version=source.version,
            reason="The supporting source failed review",
        ),
    )

    result = await kernel.complete_task(
        actor,
        CompleteTaskCommand(
            idempotency_key="outcome-complete-after-revocation",
            task_id=task.task_id,
            lease_id=claim.lease.lease_id,
            expected_task_version=claim.task.version,
            expected_lease_version=claim.lease.version,
            summary="Completion persists while revoked proof is omitted",
            memory_ids=(memory.memory_id,),
        ),
    )

    assert result.completion.outcome is TaskOutcome.SUCCEEDED
    assert await kernel.list_memory_outcome_associations(actor) == ()


@pytest.mark.asyncio
async def test_completion_replay_is_idempotent_and_reads_are_scope_isolated(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    memory = await _publish_confirmed(kernel, actor, key="publish-replay")
    task = await kernel.add_task(make_task(scope_ids))
    claim = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="claim-replay-outcome", task_id=task.task_id),
    )
    await _record_activation(
        kernel,
        actor,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        memory=memory,
    )
    command = CompleteTaskCommand(
        idempotency_key="complete-replay-outcome",
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        expected_task_version=claim.task.version,
        expected_lease_version=claim.lease.version,
        summary="Complete exactly once",
        memory_ids=(memory.memory_id,),
    )

    first = await kernel.complete_task(actor, command)
    replay = await kernel.complete_task(actor, command)

    assert not first.replayed and replay.replayed
    assert len(await kernel.list_memory_outcome_associations(actor)) == 1
    foreign = actor.model_copy(update={"run_id": new_id()})
    assert await kernel.list_memory_outcome_associations(foreign) == ()
    assert await kernel.list_memory_outcome_associations(
        actor,
        task_id=task.task_id,
        memory_id=memory.memory_id,
        limit=1,
    ) == tuple(kernel.memory_outcome_associations.values())
    with pytest.raises(ValueError, match="between 1 and 1000"):
        await kernel.list_memory_outcome_associations(actor, limit=1001)
