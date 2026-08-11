from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_actor, make_task, new_id
from swarmbrain.application.coordination import CoordinationService
from swarmbrain.application.errors import Forbidden
from swarmbrain.domain.activation import ActivationDecision, ActivationTrigger
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.leases import TaskLease
from swarmbrain.domain.memory import Memory, MemoryKind, RecallBundle, RecallHit, RecallQuery
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import (
    ClaimTaskCommand,
    ClaimTaskResult,
    TaskCheckpoint,
    TaskStatus,
)
from swarmbrain.retrieval import estimate_tokens, render_recall_hit


class _ClaimStore:
    def __init__(
        self,
        result: ClaimTaskResult,
        *,
        record_failure: Exception | None = None,
    ) -> None:
        self.result = result
        self.record_failure = record_failure
        self.activation_records: list[tuple[object, object]] = []

    async def claim_task(self, _actor: object, _command: object) -> ClaimTaskResult:
        return self.result

    async def record_memory_activation(self, actor: object, telemetry: object) -> None:
        self.activation_records.append((actor, telemetry))
        if self.record_failure is not None:
            raise self.record_failure


class _CapturingMemory:
    def __init__(
        self,
        *,
        hits: tuple[RecallHit, ...] = (),
        failure: Exception | None = None,
    ) -> None:
        self.hits = hits
        self.failure = failure
        self.query: RecallQuery | None = None
        self.purpose: RetrievalPurpose | None = None
        self.seeds: tuple[str, ...] = ()
        self.token_budget: int | None = None

    async def recall(
        self,
        _actor: object,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose,
        seed_memory_ids: tuple[str, ...],
        token_budget: int | None,
    ) -> RecallBundle:
        self.query = query
        self.purpose = purpose
        self.seeds = seed_memory_ids
        self.token_budget = token_budget
        if self.failure is not None:
            raise self.failure
        return RecallBundle(
            query=query,
            hits=self.hits,
            total_candidates=len(self.hits),
        )


def _claimed_result(
    scope_ids: dict[str, str],
    actor: ActorContext,
    *,
    checkpoint_memory_ids: tuple[str, ...] | None = None,
) -> ClaimTaskResult:
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    task = make_task(scope_ids, title="Repair retry coordinator", created_at=now).model_copy(
        update={
            "description": "Fix SQLSTATE 40001 handling",
            "tags": ("cockroachdb", "retry"),
            "required_capabilities": frozenset({"python"}),
            "status": TaskStatus.CLAIMED,
            "claimed_by_agent_id": actor.agent_id,
            "active_lease_id": new_id(),
        }
    )
    lease = TaskLease(
        lease_id=task.active_lease_id,
        task_id=task.task_id,
        run_id=task.run_id,
        owner_agent_id=actor.agent_id,
        acquired_at=now,
        expires_at=now + timedelta(minutes=2),
    )
    checkpoint = None
    if checkpoint_memory_ids is not None:
        checkpoint = TaskCheckpoint(
            checkpoint_id=new_id(),
            task_id=task.task_id,
            lease_id=lease.lease_id,
            run_id=task.run_id,
            agent_id=actor.agent_id,
            sequence=2,
            summary="Parser repaired, transaction retry remains",
            discoveries=("RetryCoordinator owns backoff",),
            remaining_work=("Run serialization integration test",),
            memory_ids=checkpoint_memory_ids,
            created_at=now,
        )
    return ClaimTaskResult(task=task, lease=lease, checkpoint=checkpoint)


def _memory(actor: ActorContext, content: str) -> Memory:
    return Memory(
        memory_id=new_id(),
        tenant_id=actor.tenant_id,
        project_id=actor.project_id,
        repository_id=actor.repository_id,
        swarm_id=actor.swarm_id,
        run_id=actor.run_id,
        author_agent_id=new_id(),
        kind=MemoryKind.PROCEDURE,
        content=content,
        valid_from=datetime(2026, 8, 5, 11, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_checkpoint_claim_selects_resume_activation_and_checkpoint_seeds(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    memory_id = new_id()
    stored = _claimed_result(scope_ids, actor, checkpoint_memory_ids=(memory_id,))
    store = _ClaimStore(stored)
    capture = _CapturingMemory()
    service = CoordinationService(
        store,  # type: ignore[arg-type]
        memory_service=capture,  # type: ignore[arg-type]
        initial_memory_min_score=0.4,
    )

    result = await service.claim(actor, ClaimTaskCommand(idempotency_key="claim-bootstrap"))

    assert result.memory is not None
    assert result.initial_memory is result.memory
    assert result.activation is not None
    assert result.activation.trigger is ActivationTrigger.CHECKPOINT_RESUME
    assert result.activation.decision is ActivationDecision.SKIP
    assert capture.purpose is RetrievalPurpose.HANDOFF_RECOVERY
    assert capture.seeds == (memory_id,)
    assert capture.token_budget == 2048
    assert capture.query is not None
    assert capture.query.task_id == stored.task.task_id
    assert capture.query.min_score == pytest.approx(0.4)
    # Deep checkpoint recovery broadens candidate retrieval within the same
    # output budget.
    assert capture.query.limit == 24
    assert "Parser repaired" in capture.query.text
    assert "RetryCoordinator owns backoff" in capture.query.text
    assert "serialization integration test" in capture.query.text
    assert store.activation_records == [(actor, result.activation)]
    serialized = result.activation.model_dump_json()
    assert "Parser repaired" not in serialized
    assert "RetryCoordinator" not in serialized


@pytest.mark.asyncio
async def test_new_claim_returns_bounded_bundle_and_content_free_activation_telemetry(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    oversized = _memory(actor, "private oversized procedure " + "x" * 8_000)
    compact = _memory(actor, "private compact procedure")
    oversized_hit = RecallHit(memory=oversized, score=1.0)
    compact_hit = RecallHit(memory=compact, score=0.9)
    token_budget = estimate_tokens("\n\n" + render_recall_hit(compact_hit))
    capture = _CapturingMemory(hits=(oversized_hit, compact_hit))
    stored = _claimed_result(scope_ids, actor)
    store = _ClaimStore(stored)
    service = CoordinationService(
        store,  # type: ignore[arg-type]
        memory_service=capture,  # type: ignore[arg-type]
        initial_memory_limit=5,
        initial_memory_token_budget=token_budget,
        initial_memory_min_score=0.55,
    )

    result = await service.claim(actor, ClaimTaskCommand(idempotency_key="bounded-claim"))

    assert result.memory is not None and result.activation is not None
    assert result.initial_memory is result.memory
    assert [hit.memory.memory_id for hit in result.memory.hits] == [compact.memory_id]
    assert result.activation.trigger is ActivationTrigger.TASK_CLAIM
    assert result.activation.decision is ActivationDecision.RECALL
    assert result.activation.memory_ids == (compact.memory_id,)
    assert result.activation.dropped_memory_ids == (oversized.memory_id,)
    assert result.activation.estimated_tokens <= token_budget
    assert capture.purpose is RetrievalPurpose.TASK_BOOTSTRAP
    assert capture.seeds == ()
    assert capture.token_budget == token_budget
    assert capture.query is not None
    assert capture.query.limit == 5
    assert capture.query.min_score == pytest.approx(0.55)
    assert store.activation_records == [(actor, result.activation)]

    telemetry = result.activation.model_dump_json()
    assert stored.task.title not in telemetry
    assert str(oversized.content) not in telemetry
    assert str(compact.content) not in telemetry


@pytest.mark.asyncio
async def test_claim_after_completed_blocker_uses_dependency_unblocked_trigger(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    prerequisite_id = new_id()
    stored = _claimed_result(scope_ids, actor).model_copy(
        update={"unblocked_by_task_ids": (prerequisite_id,)}
    )
    store = _ClaimStore(stored)
    capture = _CapturingMemory()
    service = CoordinationService(
        store,  # type: ignore[arg-type]
        memory_service=capture,  # type: ignore[arg-type]
    )

    result = await service.claim(actor, ClaimTaskCommand(idempotency_key="dependency-ready"))

    assert result.activation is not None
    assert result.activation.trigger is ActivationTrigger.DEPENDENCY_UNBLOCKED
    assert capture.purpose is RetrievalPurpose.TASK_BOOTSTRAP
    assert capture.query is not None
    # Opaque task UUIDs remain audit metadata, not retrieval terms: adding
    # them dilutes calibrated query-token coverage and can suppress otherwise
    # relevant bootstrap memories at the default relevance floor.
    assert prerequisite_id not in capture.query.text


@pytest.mark.asyncio
async def test_optional_activation_outage_defers_without_hiding_the_committed_claim(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    stored = _claimed_result(scope_ids, actor)
    store = _ClaimStore(stored)
    service = CoordinationService(
        store,  # type: ignore[arg-type]
        memory_service=_CapturingMemory(  # type: ignore[arg-type]
            failure=RuntimeError("database unavailable")
        ),
    )

    result = await service.claim(actor, ClaimTaskCommand(idempotency_key="deferred-claim"))

    assert result.task == stored.task
    assert result.lease == stored.lease
    assert result.memory is None
    assert result.activation is not None
    assert result.activation.decision is ActivationDecision.DEFER
    assert store.activation_records == [(actor, result.activation)]


@pytest.mark.asyncio
async def test_post_claim_activation_authorization_failure_does_not_hide_the_lease(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    stored = _claimed_result(scope_ids, actor)
    store = _ClaimStore(stored)
    service = CoordinationService(
        store,  # type: ignore[arg-type]
        memory_service=_CapturingMemory(  # type: ignore[arg-type]
            failure=Forbidden("memory:recall")
        ),
    )

    result = await service.claim(
        actor,
        ClaimTaskCommand(idempotency_key="forbidden-activation"),
    )

    assert result.task == stored.task
    assert result.lease == stored.lease
    assert result.activation is None
    assert result.memory is None
    assert store.activation_records == []


@pytest.mark.asyncio
async def test_activation_telemetry_failure_prevents_untracked_context_delivery(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    stored = _claimed_result(scope_ids, actor)
    store = _ClaimStore(stored, record_failure=RuntimeError("telemetry unavailable"))
    service = CoordinationService(
        store,  # type: ignore[arg-type]
        memory_service=_CapturingMemory(),  # type: ignore[arg-type]
    )

    result = await service.claim(actor, ClaimTaskCommand(idempotency_key="telemetry-failure"))

    assert result.activation is None
    assert result.activation_context is None
    assert result.memory is None
    assert len(store.activation_records) == 1
