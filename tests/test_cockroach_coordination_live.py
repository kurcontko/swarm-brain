from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import pytest

from conftest import make_actor, make_task, new_id
from swarmbrain.adapters.cockroach import coordination
from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.application.errors import InvalidState, LeaseLost, NoTaskAvailable
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationReason,
    ActivationTrigger,
    MemoryActivationTelemetry,
    memory_activation_id,
)
from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.events import EventType
from swarmbrain.domain.leases import LeaseStatus, RenewLeaseCommand
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    ClaimTaskCommand,
    CompleteTaskCommand,
    ReleaseTaskCommand,
    TaskDependency,
    TaskStatus,
)

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


class _Ack(ContractModel):
    ok: bool = True


class _EventCounts(ContractModel):
    event_count: int
    outbox_count: int
    event_id: str
    envelope: dict[str, Any]


@pytest.fixture
async def database() -> CockroachDatabase:
    assert DATABASE_URL is not None
    value = CockroachDatabase(DATABASE_URL, min_size=1, max_size=16)
    await value.start()
    try:
        yield value
    finally:
        await value.close()


async def _cleanup(database: CockroachDatabase, scope: dict[str, str]) -> None:
    async def transaction(connection: Any) -> _Ack:
        params = (scope["tenant_id"], scope["run_id"])
        await connection.execute(
            "DELETE FROM outbox_events WHERE tenant_id = %s AND run_id = %s",
            params,
        )
        await connection.execute(
            "DELETE FROM action_attempts WHERE tenant_id = %s AND run_id = %s",
            params,
        )
        await connection.execute(
            "DELETE FROM idempotency_records WHERE tenant_id = %s AND run_id = %s",
            params,
        )
        await connection.execute(
            "DELETE FROM runs WHERE tenant_id = %s AND id = %s",
            params,
        )
        return _Ack()

    await database.run(transaction)


async def test_concurrent_claims_and_duplicate_completion_are_atomic(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    store = CockroachCoordinationStore(database)
    actors = [make_actor(scope_ids) for _ in range(12)]
    try:
        await asyncio.gather(*(store.join_agent(actor) for actor in actors))
        for index in range(4):
            await store.add_task(
                make_task(
                    scope_ids,
                    title=f"Concurrent task {index}",
                    priority=index,
                )
            )

        outcomes = await asyncio.gather(
            *(
                store.claim_task(
                    actor,
                    ClaimTaskCommand(idempotency_key=f"claim-{index}"),
                )
                for index, actor in enumerate(actors)
            ),
            return_exceptions=True,
        )

        claims = [item for item in outcomes if not isinstance(item, BaseException)]
        failures = [item for item in outcomes if isinstance(item, BaseException)]
        assert len(claims) == 4
        assert len({item.task.task_id for item in claims}) == 4
        assert len({item.lease.lease_id for item in claims}) == 4
        assert all(isinstance(item, NoTaskAvailable) for item in failures)

        selected = claims[0]
        selected_actor = next(
            actor for actor in actors if actor.agent_id == selected.lease.owner_agent_id
        )
        command = CompleteTaskCommand(
            idempotency_key="complete-once",
            task_id=selected.task.task_id,
            lease_id=selected.lease.lease_id,
            expected_task_version=selected.task.version,
            expected_lease_version=selected.lease.version,
            summary="durable completion",
        )
        first = await store.complete_task(selected_actor, command)
        replay = await CockroachCoordinationStore(database).complete_task(
            selected_actor,
            command,
        )

        assert replay == first.model_copy(update={"replayed": True})

        async def read_counts(connection: Any) -> _EventCounts:
            cursor = await connection.execute(
                """
                SELECT
                    (SELECT count(*)
                     FROM swarm_events
                     WHERE tenant_id = %s AND run_id = %s AND event_type = %s
                    ) AS event_count,
                    (SELECT count(*)
                     FROM outbox_events
                     WHERE tenant_id = %s AND run_id = %s AND event_type = %s
                    ) AS outbox_count,
                    event.id AS event_id,
                    outbox.payload AS envelope
                FROM swarm_events AS event
                JOIN outbox_events AS outbox ON outbox.event_id = event.id
                WHERE event.tenant_id = %s
                  AND event.run_id = %s
                  AND event.event_type = %s
                """,
                (
                    scope_ids["tenant_id"],
                    scope_ids["run_id"],
                    EventType.TASK_COMPLETED.value,
                    scope_ids["tenant_id"],
                    scope_ids["run_id"],
                    EventType.TASK_COMPLETED.value,
                    scope_ids["tenant_id"],
                    scope_ids["run_id"],
                    EventType.TASK_COMPLETED.value,
                ),
            )
            row = await cursor.fetchone()
            assert row is not None
            return _EventCounts(
                event_count=int(row["event_count"]),
                outbox_count=int(row["outbox_count"]),
                event_id=str(row["event_id"]),
                envelope=row["envelope"],
            )

        counts = await database.run(read_counts)
        assert counts.event_count == 1
        assert counts.outbox_count == 1
        assert counts.envelope["event_id"] == counts.event_id
        assert counts.envelope["event_type"] == EventType.TASK_COMPLETED.value
        assert counts.envelope["task_id"] == selected.task.task_id
    finally:
        await _cleanup(database, scope_ids)


async def test_expired_lease_handoff_restores_checkpoint_and_fences_old_owner(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    store = CockroachCoordinationStore(database)
    first_actor = make_actor(scope_ids, harness="claude-code", provider="anthropic")
    second_actor = make_actor(scope_ids, harness="opencode", provider="local")
    try:
        await store.join_agent(first_actor)
        await store.join_agent(second_actor)
        task = await store.add_task(make_task(scope_ids))
        claimed = await store.claim_task(
            first_actor,
            ClaimTaskCommand(
                idempotency_key="first-claim",
                task_id=task.task_id,
                lease_seconds=15,
            ),
        )
        checkpoint = await store.checkpoint_task(
            first_actor,
            CheckpointCommand(
                idempotency_key="first-checkpoint",
                task_id=task.task_id,
                lease_id=claimed.lease.lease_id,
                expected_task_version=claimed.task.version,
                expected_lease_version=claimed.lease.version,
                summary="serializer remains",
                discoveries=("parser isolated",),
                remaining_work=("patch serializer",),
            ),
        )

        async def expire(connection: Any) -> _Ack:
            await connection.execute(
                """
                UPDATE task_leases
                SET claimed_at = now() - INTERVAL '30 seconds',
                    renewed_at = now() - INTERVAL '30 seconds',
                    expires_at = now() - INTERVAL '1 second'
                WHERE id = %s
                  AND tenant_id = %s
                  AND run_id = %s
                """,
                (
                    claimed.lease.lease_id,
                    first_actor.tenant_id,
                    first_actor.run_id,
                ),
            )
            return _Ack()

        await database.run(expire)
        handoff = await store.claim_task(
            second_actor,
            ClaimTaskCommand(
                idempotency_key="second-claim",
                task_id=task.task_id,
            ),
        )

        assert handoff.lease.owner_agent_id == second_actor.agent_id
        assert handoff.checkpoint is not None
        assert handoff.checkpoint.checkpoint_id == checkpoint.checkpoint.checkpoint_id
        assert handoff.checkpoint.remaining_work == ("patch serializer",)
        metrics = await store.get_run_metrics(second_actor, second_actor.run_id)
        assert metrics.crash_handoffs == 1

        with pytest.raises(LeaseLost):
            await store.checkpoint_task(
                first_actor,
                CheckpointCommand(
                    idempotency_key="late-first-write",
                    task_id=task.task_id,
                    lease_id=claimed.lease.lease_id,
                    expected_task_version=checkpoint.task.version,
                    expected_lease_version=checkpoint.lease.version,
                    summary="stale write",
                ),
            )
    finally:
        await _cleanup(database, scope_ids)


async def test_renew_release_and_reclaim_preserve_lease_fencing(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    store = CockroachCoordinationStore(database)
    first_actor = make_actor(scope_ids)
    second_actor = make_actor(scope_ids, harness="claude-code", provider="anthropic")
    try:
        await store.join_agent(first_actor)
        await store.join_agent(second_actor)
        task = await store.add_task(make_task(scope_ids, title="Renew and release"))
        claim = await store.claim_task(
            first_actor,
            ClaimTaskCommand(idempotency_key="renew-release-claim", task_id=task.task_id),
        )
        renew_command = RenewLeaseCommand(
            idempotency_key="renew-release-renew",
            lease_id=claim.lease.lease_id,
            expected_version=claim.lease.version,
            extension_seconds=180,
        )
        renewed = await store.renew_lease(first_actor, renew_command)
        replay = await CockroachCoordinationStore(database).renew_lease(
            first_actor,
            renew_command,
        )

        assert renewed.lease.version == claim.lease.version + 1
        assert renewed.lease.expires_at >= claim.lease.expires_at
        assert replay == renewed.model_copy(update={"replayed": True})

        released = await store.release_task(
            first_actor,
            ReleaseTaskCommand(
                idempotency_key="renew-release-release",
                task_id=task.task_id,
                lease_id=renewed.lease.lease_id,
                expected_task_version=claim.task.version,
                expected_lease_version=renewed.lease.version,
                reason="intentional handoff",
            ),
        )
        assert released.task.status is TaskStatus.READY
        assert released.lease.status is LeaseStatus.RELEASED

        reclaimed = await store.claim_task(
            second_actor,
            ClaimTaskCommand(idempotency_key="renew-release-reclaim", task_id=task.task_id),
        )
        assert reclaimed.lease.owner_agent_id == second_actor.agent_id
        assert reclaimed.lease.lease_id != claim.lease.lease_id
    finally:
        await _cleanup(database, scope_ids)


async def test_dependency_gating_cycle_guard_keyset_events_and_explain(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    store = CockroachCoordinationStore(database)
    actor = make_actor(scope_ids)
    try:
        await store.join_agent(actor)
        prerequisite = await store.add_task(make_task(scope_ids, title="Prepare fixture"))
        dependent = await store.add_task(make_task(scope_ids, title="Run integration"))
        await store.add_dependency(
            TaskDependency(
                task_id=dependent.task_id,
                depends_on_task_id=prerequisite.task_id,
            )
        )
        with pytest.raises(InvalidState, match="cycle"):
            await store.add_dependency(
                TaskDependency(
                    task_id=prerequisite.task_id,
                    depends_on_task_id=dependent.task_id,
                )
            )
        with pytest.raises(NoTaskAvailable):
            await store.claim_task(
                actor,
                ClaimTaskCommand(idempotency_key="blocked", task_id=dependent.task_id),
            )

        claim = await store.claim_task(
            actor,
            ClaimTaskCommand(idempotency_key="claim-prerequisite", task_id=prerequisite.task_id),
        )
        await store.complete_task(
            actor,
            CompleteTaskCommand(
                idempotency_key="complete-prerequisite",
                task_id=prerequisite.task_id,
                lease_id=claim.lease.lease_id,
                expected_task_version=claim.task.version,
                expected_lease_version=claim.lease.version,
                summary="fixture ready",
            ),
        )
        available = await store.claim_task(
            actor,
            ClaimTaskCommand(idempotency_key="claim-dependent", task_id=dependent.task_id),
        )
        assert available.task.task_id == dependent.task_id

        first_page = await store.list_run_events(actor, actor.run_id, limit=2)
        assert len(first_page.events) == 2
        assert first_page.next_cursor is not None
        second_page = await store.list_run_events(
            actor,
            actor.run_id,
            cursor=first_page.next_cursor,
            limit=2,
        )
        assert second_page.events
        assert not (
            {item.event_id for item in first_page.events}
            & {item.event_id for item in second_page.events}
        )

        async def explain(connection: Any) -> _Ack:
            explain_queries = (
                (
                    coordination.CLAIM_CANDIDATE_SQL,
                    (
                        actor.tenant_id,
                        actor.project_id,
                        actor.repository_id,
                        actor.swarm_id,
                        actor.run_id,
                        None,
                        None,
                        [],
                        [],
                        sorted(actor.capabilities),
                    ),
                ),
                (
                    coordination.EVENT_PAGE_AFTER_SQL,
                    (
                        actor.tenant_id,
                        actor.project_id,
                        actor.repository_id,
                        actor.swarm_id,
                        actor.run_id,
                        datetime.now(UTC),
                        new_id(),
                        101,
                    ),
                ),
                (
                    coordination.RUN_METRICS_SQL,
                    (
                        actor.tenant_id,
                        actor.run_id,
                        actor.project_id,
                        actor.repository_id,
                        actor.swarm_id,
                    ),
                ),
            )
            for sql, params in explain_queries:
                cursor = await connection.execute("EXPLAIN " + sql, params)
                assert await cursor.fetchall()
            return _Ack()

        await database.run(explain)
    finally:
        await _cleanup(database, scope_ids)


async def test_memory_activation_event_is_scoped_content_free_and_idempotent(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    store = CockroachCoordinationStore(database)
    actor = make_actor(scope_ids)
    try:
        await store.join_agent(actor)
        task = await store.add_task(make_task(scope_ids, title="Activation event"))
        claim = await store.claim_task(
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

        await asyncio.gather(*(store.record_memory_activation(actor, telemetry) for _ in range(8)))
        assert await store.get_memory_activation(actor, activation_id) == telemetry

        async def read_activation(connection: Any) -> dict[str, Any]:
            cursor = await connection.execute(
                """
                SELECT event.id, event.payload, count(outbox.id) AS outbox_count
                FROM swarm_events AS event
                LEFT JOIN outbox_events AS outbox ON outbox.event_id = event.id
                WHERE event.id = %s::UUID
                  AND event.tenant_id = %s
                  AND event.run_id = %s
                GROUP BY event.id, event.payload
                """,
                (activation_id, actor.tenant_id, actor.run_id),
            )
            row = await cursor.fetchone()
            assert row is not None
            return dict(row)

        row = await database.run(read_activation)
        assert str(row["id"]) == activation_id
        assert int(row["outbox_count"]) == 1
        assert row["payload"]["decision"] == ActivationDecision.SKIP.value
        serialized = str(row["payload"])
        assert "query" not in serialized
        assert "content" not in serialized
    finally:
        await _cleanup(database, scope_ids)
