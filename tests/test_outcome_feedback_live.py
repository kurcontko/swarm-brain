from __future__ import annotations

import os
from typing import Any

import pytest

from conftest import make_actor, make_task, new_id
from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationReason,
    ActivationTrigger,
    MemoryActivationTelemetry,
    memory_activation_id,
)
from swarmbrain.domain.memory import MemoryKind, MemoryState, RememberCommand, Visibility
from swarmbrain.domain.outcome_feedback import OutcomeAssociationKind
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import ClaimTaskCommand, CompleteTaskCommand, TaskOutcome

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


@pytest.fixture
async def database() -> CockroachDatabase:
    assert DATABASE_URL is not None
    value = CockroachDatabase(DATABASE_URL, min_size=1, max_size=4)
    await value.start()
    try:
        yield value
    finally:
        await value.close()


async def _cleanup(database: CockroachDatabase, actor: object) -> None:
    async def transaction(connection: Any) -> None:
        params = (actor.tenant_id, actor.run_id)
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

    await database.run(transaction)  # type: ignore[arg-type]


async def test_cockroach_completion_atomically_records_one_silver_association(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    coordination = CockroachCoordinationStore(database)
    memory_store = CockroachMemoryStore(database)
    try:
        await coordination.join_agent(actor)
        published = await memory_store.remember(
            actor,
            RememberCommand(
                idempotency_key="live-outcome-memory",
                kind=MemoryKind.PROCEDURE,
                content="Never copy this content into outcome telemetry",
                desired_state=MemoryState.CONFIRMED,
                visibility=Visibility.REPOSITORY,
            ),
            ConservativeMemoryPolicy(),
        )
        assert published.memory is not None
        memory = published.memory
        task = await coordination.add_task(make_task(scope_ids, title="Use proven memory"))
        claim = await coordination.claim_task(
            actor,
            ClaimTaskCommand(idempotency_key="live-outcome-claim", task_id=task.task_id),
        )
        await coordination.record_memory_activation(
            actor,
            MemoryActivationTelemetry(
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
                memory_ids=(memory.memory_id,),
                memory_versions={memory.memory_id: memory.version},
                token_budget=2_048,
                estimated_tokens=16,
                min_score=0.4,
                candidate_count=1,
            ),
        )
        command = CompleteTaskCommand(
            idempotency_key="live-outcome-complete",
            task_id=task.task_id,
            lease_id=claim.lease.lease_id,
            expected_task_version=claim.task.version,
            expected_lease_version=claim.lease.version,
            outcome=TaskOutcome.FAILED,
            summary="Failure is observational and must not refute memory",
            memory_ids=(memory.memory_id,),
        )

        first = await coordination.complete_task(actor, command)
        replay = await coordination.complete_task(actor, command)
        associations = await coordination.list_memory_outcome_associations(actor)

        assert replay == first.model_copy(update={"replayed": True})
        assert len(associations) == 1
        association = associations[0]
        assert association.kind is OutcomeAssociationKind.OBSERVATIONAL_SILVER
        assert association.outcome is TaskOutcome.FAILED
        assert association.memory_id == memory.memory_id
        assert association.memory_version == memory.version
        foreign = actor.model_copy(update={"run_id": new_id()})
        assert await coordination.list_memory_outcome_associations(foreign) == ()

        async def inspect(connection: Any) -> tuple[int, tuple[str, ...]]:
            count_cursor = await connection.execute(
                """
                SELECT count(*) AS total
                FROM memory_outcome_associations
                WHERE tenant_id = %s AND run_id = %s
                """,
                (actor.tenant_id, actor.run_id),
            )
            count_row = await count_cursor.fetchone()
            columns_cursor = await connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'memory_outcome_associations'
                ORDER BY column_name
                """
            )
            return int(count_row["total"]), tuple(
                str(row["column_name"]) for row in await columns_cursor.fetchall()
            )

        total, columns = await database.run(inspect)  # type: ignore[arg-type]
        assert total == 1
        assert not {"query", "content", "summary", "prompt"} & set(columns)
    finally:
        await _cleanup(database, actor)
