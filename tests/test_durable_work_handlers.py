from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import make_actor, new_id
from swarmbrain.adapters.extraction import InMemoryWorkStore
from swarmbrain.application.errors import IdempotencyConflict
from swarmbrain.application.work import DurableWorkService
from swarmbrain.domain.memory import EmbeddingVector
from swarmbrain.domain.work import (
    ApplyEmbeddingWorkCommand,
    ClaimWorkCommand,
    CompleteWorkCommand,
    EmbeddingWorkHandlerResult,
    EnqueueWorkCommand,
    FailWorkCommand,
    WorkCompletionConflict,
    WorkFailureConflict,
    WorkHandlerResult,
    WorkKind,
    WorkLease,
    WorkLeaseLost,
    WorkStatus,
    embedding_vector_sha256,
)
from swarmbrain.workers import LeasedWorkWorker


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _embedding_command(
    clock: MutableClock,
    *,
    idempotency_key: str = "embed-enqueue-1",
    dedupe_key: str = "memory:model-a",
    subject_id: str | None = None,
    content: str = "A durable memory to embed",
) -> EnqueueWorkCommand:
    return EnqueueWorkCommand(
        idempotency_key=idempotency_key,
        kind=WorkKind.EMBED_MEMORY,
        subject_id=subject_id or new_id(),
        dedupe_key=dedupe_key,
        payload={"content": content, "model": "model-a"},
        available_at=clock.now,
    )


def _artifact_command(
    clock: MutableClock,
    *,
    idempotency_key: str = "artifact-enqueue-1",
    dedupe_key: str = "artifact:report",
    subject_id: str | None = None,
) -> EnqueueWorkCommand:
    content = b"durable artifact bytes"
    return EnqueueWorkCommand(
        idempotency_key=idempotency_key,
        kind=WorkKind.PERSIST_ARTIFACT,
        subject_id=subject_id or new_id(),
        dedupe_key=dedupe_key,
        payload={
            "name": "report.txt",
            "media_type": "text/plain",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "staging_uri": "file:///staging/report.txt",
        },
        available_at=clock.now,
    )


def _embedding_apply_command(
    lease: WorkLease,
    values: tuple[float, ...],
) -> ApplyEmbeddingWorkCommand:
    vector = EmbeddingVector(
        memory_id=lease.item.subject_id,
        model=str(lease.item.payload["model"]),
        dimensions=len(values),
        values=values,
    )
    return ApplyEmbeddingWorkCommand(
        work_id=lease.item.work_id,
        worker_id=lease.worker_id,
        lease_token=lease.lease_token,
        lease_version=lease.lease_version,
        expected_work_version=lease.work_version,
        attempt=lease.attempt,
        vector=vector,
        result={
            "model": vector.model,
            "dimensions": vector.dimensions,
            "vector_sha256": embedding_vector_sha256(vector),
        },
    )


class CountingHandler:
    def __init__(
        self,
        kind: WorkKind,
        *,
        transaction_probe: Any | None = None,
        failures: int = 0,
    ) -> None:
        self.kind = kind
        self.transaction_probe = transaction_probe
        self.failures = failures
        self.calls = 0

    async def handle(
        self,
        lease: WorkLease,
    ) -> WorkHandlerResult | EmbeddingWorkHandlerResult:
        assert lease.item.kind is self.kind
        if self.transaction_probe is not None:
            assert self.transaction_probe.in_transaction is False
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("local handler failure")
        result = {
            "kind": self.kind.value,
            "subject_id": lease.item.subject_id,
            "handler_calls": self.calls,
        }
        if self.kind is not WorkKind.EMBED_MEMORY:
            return WorkHandlerResult(result=result)
        vector = EmbeddingVector(
            memory_id=lease.item.subject_id,
            model=str(lease.item.payload["model"]),
            dimensions=3,
            values=(1.0, 0.0, 0.0),
        )
        return EmbeddingWorkHandlerResult(
            vector=vector,
            result={
                "model": vector.model,
                "dimensions": vector.dimensions,
                "vector_sha256": embedding_vector_sha256(vector),
            },
        )


class RetryingCompletionQueue:
    """Simulate an internal 40001 retry while exposing transaction boundaries."""

    def __init__(self, delegate: InMemoryWorkStore) -> None:
        self.delegate = delegate
        self.in_transaction = False
        self.completion_transaction_attempts = 0

    async def enqueue_work(self, actor: Any, prepared: Any) -> Any:
        self.in_transaction = True
        try:
            return await self.delegate.enqueue_work(actor, prepared)
        finally:
            self.in_transaction = False

    async def claim_work(self, command: Any) -> Any:
        self.in_transaction = True
        try:
            return await self.delegate.claim_work(command)
        finally:
            self.in_transaction = False

    async def apply_extraction(self, command: Any) -> Any:
        self.in_transaction = True
        try:
            return await self.delegate.apply_extraction(command)
        finally:
            self.in_transaction = False

    async def apply_embedding(self, command: Any) -> Any:
        self.in_transaction = True
        try:
            self.completion_transaction_attempts += 1
            try:
                raise SerializationFailure()
            except SerializationFailure as exc:
                assert exc.sqlstate == "40001"
            self.completion_transaction_attempts += 1
            return await self.delegate.apply_embedding(command)
        finally:
            self.in_transaction = False

    async def complete_work(self, command: Any) -> Any:
        self.in_transaction = True
        try:
            self.completion_transaction_attempts += 1
            try:
                raise SerializationFailure()
            except SerializationFailure as exc:
                assert exc.sqlstate == "40001"
            self.completion_transaction_attempts += 1
            return await self.delegate.complete_work(command)
        finally:
            self.in_transaction = False

    async def fail_work(self, command: Any) -> Any:
        self.in_transaction = True
        try:
            return await self.delegate.fail_work(command)
        finally:
            self.in_transaction = False


class SerializationFailure(RuntimeError):
    sqlstate = "40001"


def test_work_payloads_are_typed_and_extract_source_cannot_use_generic_enqueue() -> None:
    with pytest.raises(ValidationError):
        EnqueueWorkCommand(
            idempotency_key="invalid-embed",
            kind=WorkKind.EMBED_MEMORY,
            subject_id=new_id(),
            dedupe_key="missing-model",
            payload={"content": "text"},
        )

    with pytest.raises(ValidationError, match="raw-source ingestion"):
        EnqueueWorkCommand(
            idempotency_key="invalid-extract",
            kind=WorkKind.EXTRACT_SOURCE,
            subject_id=new_id(),
            dedupe_key="extract",
            payload={},
        )


@pytest.mark.asyncio
async def test_enqueue_is_durable_and_dedupes_semantic_work(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = DurableWorkService(store)
    actor = make_actor(scope_ids)
    subject_id = new_id()
    first_command = _embedding_command(clock, subject_id=subject_id)

    first = await service.enqueue(actor, first_command)
    idempotency_replay = await service.enqueue(actor, first_command)
    semantic_replay = await service.enqueue(
        actor,
        _embedding_command(
            clock,
            idempotency_key="embed-enqueue-2",
            subject_id=subject_id,
        ),
    )

    assert first.replayed is False
    assert idempotency_replay.replayed is True
    assert semantic_replay.replayed is True
    assert semantic_replay.item.work_id == first.item.work_id
    assert len(store.items) == 1

    with pytest.raises(IdempotencyConflict):
        await service.enqueue(
            actor,
            _embedding_command(
                clock,
                idempotency_key="embed-enqueue-3",
                subject_id=subject_id,
                content="different semantic payload",
            ),
        )


@pytest.mark.asyncio
async def test_handlers_run_once_outside_retried_completion_transaction(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    queue = RetryingCompletionQueue(store)
    service = DurableWorkService(queue)
    actor = make_actor(scope_ids)
    await service.enqueue(actor, _embedding_command(clock))
    await service.enqueue(actor, _artifact_command(clock))
    embedding = CountingHandler(WorkKind.EMBED_MEMORY, transaction_probe=queue)
    artifact = CountingHandler(WorkKind.PERSIST_ARTIFACT, transaction_probe=queue)
    worker = LeasedWorkWorker(queue, (embedding, artifact), clock=clock)

    completed = await worker.run_once("handler-worker", limit=2)
    empty = await worker.run_once("handler-worker", limit=2)

    assert len(completed) == 2
    assert empty == ()
    assert embedding.calls == 1
    assert artifact.calls == 1
    assert queue.completion_transaction_attempts == 4
    assert {item.item.kind for item in completed} == {
        WorkKind.EMBED_MEMORY,
        WorkKind.PERSIST_ARTIFACT,
    }
    assert all(item.item.status is WorkStatus.COMPLETED for item in completed)


@pytest.mark.asyncio
async def test_failed_handler_is_fenced_then_retried(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = DurableWorkService(store)
    actor = make_actor(scope_ids)
    queued = await service.enqueue(actor, _artifact_command(clock))
    handler = CountingHandler(WorkKind.PERSIST_ARTIFACT, failures=1)
    worker = LeasedWorkWorker(
        store,
        (handler,),
        retry_delay_seconds=5,
        clock=clock,
    )

    first = await worker.run_once("retry-worker")
    assert first == ()
    assert handler.calls == 1
    assert store.items[queued.item.work_id].status is WorkStatus.RETRY
    assert store.items[queued.item.work_id].last_error == "handler_RuntimeError"

    clock.now += timedelta(seconds=6)
    second = await worker.run_once("retry-worker")

    assert len(second) == 1
    assert second[0].item.status is WorkStatus.COMPLETED
    assert second[0].item.attempts == 2
    assert handler.calls == 2


@pytest.mark.asyncio
async def test_expired_final_attempt_is_terminalized_instead_of_staying_leased(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = DurableWorkService(store)
    actor = make_actor(scope_ids)
    queued = await service.enqueue(
        actor,
        _artifact_command(clock).model_copy(update={"max_attempts": 1}),
    )
    lease = (
        await store.claim_work(
            ClaimWorkCommand(
                worker_id="doomed-worker",
                kinds=frozenset({WorkKind.PERSIST_ARTIFACT}),
                lease_seconds=5,
            )
        )
    ).leases[0]
    assert lease.attempt == 1
    clock.now += timedelta(seconds=6)

    reclaimed = await store.claim_work(
        ClaimWorkCommand(
            worker_id="replacement-worker",
            kinds=frozenset({WorkKind.PERSIST_ARTIFACT}),
        )
    )

    assert reclaimed.leases == ()
    terminal = store.items[queued.item.work_id]
    assert terminal.status is WorkStatus.FAILED
    assert terminal.outcome == "failed"
    assert terminal.last_error == "lease_expired"
    assert terminal.locked_by is None
    assert terminal.lease_token is None
    assert terminal.locked_until is None
    assert terminal.completed_at == clock.now


@pytest.mark.asyncio
async def test_generic_completion_rejects_stale_fence_and_replays_once(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = DurableWorkService(store)
    actor = make_actor(scope_ids)
    await service.enqueue(actor, _artifact_command(clock))
    stale = (
        await store.claim_work(
            ClaimWorkCommand(
                worker_id="stale-handler",
                kinds=frozenset({WorkKind.PERSIST_ARTIFACT}),
                lease_seconds=5,
            )
        )
    ).leases[0]
    clock.now += timedelta(seconds=6)
    current = (
        await store.claim_work(
            ClaimWorkCommand(
                worker_id="current-handler",
                kinds=frozenset({WorkKind.PERSIST_ARTIFACT}),
                lease_seconds=30,
            )
        )
    ).leases[0]

    stale_completion = CompleteWorkCommand(
        work_id=stale.item.work_id,
        worker_id=stale.worker_id,
        lease_token=stale.lease_token,
        lease_version=stale.lease_version,
        expected_work_version=stale.work_version,
        attempt=stale.attempt,
        result={"dimensions": 3},
    )
    with pytest.raises(WorkLeaseLost):
        await store.complete_work(stale_completion)

    completion = CompleteWorkCommand(
        work_id=current.item.work_id,
        worker_id=current.worker_id,
        lease_token=current.lease_token,
        lease_version=current.lease_version,
        expected_work_version=current.work_version,
        attempt=current.attempt,
        result={"dimensions": 3},
    )
    first = await store.complete_work(completion)
    replay = await store.complete_work(completion)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.item == first.item

    with pytest.raises(WorkCompletionConflict):
        await store.complete_work(completion.model_copy(update={"result": {"dimensions": 4}}))


@pytest.mark.asyncio
async def test_embedding_apply_is_atomic_and_rejects_a_stale_vector(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = DurableWorkService(store)
    actor = make_actor(scope_ids)
    await service.enqueue(actor, _embedding_command(clock))
    stale = (
        await store.claim_work(
            ClaimWorkCommand(
                worker_id="stale-embedding",
                kinds=frozenset({WorkKind.EMBED_MEMORY}),
                lease_seconds=5,
            )
        )
    ).leases[0]
    clock.now += timedelta(seconds=6)
    current = (
        await store.claim_work(
            ClaimWorkCommand(
                worker_id="current-embedding",
                kinds=frozenset({WorkKind.EMBED_MEMORY}),
                lease_seconds=30,
            )
        )
    ).leases[0]

    with pytest.raises(WorkLeaseLost):
        await store.apply_embedding(_embedding_apply_command(stale, (1.0, 0.0, 0.0)))
    assert store.memory_embeddings == {}

    command = _embedding_apply_command(current, (0.0, 1.0, 0.0))
    first = await store.apply_embedding(command)
    replay = await store.apply_embedding(command)

    assert first.replayed is False
    assert replay.replayed is True
    assert store.memory_embeddings[(current.item.subject_id, "model-a")].vector.values == (
        0.0,
        1.0,
        0.0,
    )

    with pytest.raises(WorkCompletionConflict):
        await store.apply_embedding(_embedding_apply_command(current, (0.0, 0.0, 1.0)))


@pytest.mark.asyncio
async def test_failure_delivery_replays_exactly_and_rejects_changed_data(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = DurableWorkService(store)
    actor = make_actor(scope_ids)
    await service.enqueue(actor, _artifact_command(clock))
    lease = (
        await store.claim_work(
            ClaimWorkCommand(
                worker_id="failure-replay-worker",
                kinds=frozenset({WorkKind.PERSIST_ARTIFACT}),
            )
        )
    ).leases[0]
    failure = FailWorkCommand(
        work_id=lease.item.work_id,
        worker_id=lease.worker_id,
        lease_token=lease.lease_token,
        lease_version=lease.lease_version,
        expected_work_version=lease.work_version,
        attempt=lease.attempt,
        error="handler_LocalFailure",
        retry_at=clock.now + timedelta(seconds=30),
    )

    first = await store.fail_work(failure)
    replay = await store.fail_work(failure)

    assert replay == first
    assert replay.status is WorkStatus.RETRY
    with pytest.raises(WorkFailureConflict):
        await store.fail_work(failure.model_copy(update={"error": "handler_DifferentFailure"}))

    terminal_command = _artifact_command(
        clock,
        idempotency_key="artifact-terminal",
        dedupe_key="artifact:terminal",
    ).model_copy(update={"max_attempts": 1})
    await service.enqueue(actor, terminal_command)
    terminal_lease = (
        await store.claim_work(
            ClaimWorkCommand(
                worker_id="terminal-failure-worker",
                kinds=frozenset({WorkKind.PERSIST_ARTIFACT}),
            )
        )
    ).leases[0]
    terminal = await store.fail_work(
        FailWorkCommand(
            work_id=terminal_lease.item.work_id,
            worker_id=terminal_lease.worker_id,
            lease_token=terminal_lease.lease_token,
            lease_version=terminal_lease.lease_version,
            expected_work_version=terminal_lease.work_version,
            attempt=terminal_lease.attempt,
            error="handler_TerminalFailure",
            retry_at=clock.now + timedelta(seconds=30),
        )
    )
    assert terminal.status is WorkStatus.FAILED
    assert terminal.outcome == "failed"
    assert terminal.completed_at == clock.now
