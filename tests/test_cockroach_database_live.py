from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import uuid4

import pytest
from psycopg.types.json import Jsonb

from swarmbrain.adapters.cockroach import database as database_module
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.retry import AmbiguousTransactionResult
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import ContractModel, MutationCommand

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


class _Command(MutationCommand):
    value: str


class _Result(ContractModel):
    value: str
    replayed: bool = False


def _id() -> str:
    return str(uuid4())


def _actor(*, run_id: str | None = None, agent_id: str | None = None) -> ActorContext:
    return ActorContext(
        tenant_id=_id(),
        project_id=_id(),
        repository_id=_id(),
        swarm_id=_id(),
        run_id=run_id or _id(),
        agent_id=agent_id or _id(),
        harness="pytest",
        provider="local",
        model="none",
        capabilities=frozenset(),
    )


async def _insert_run(connection: Any, actor: ActorContext) -> None:
    await connection.execute(
        """
        INSERT INTO runs (
            tenant_id, id, project_id, repository_id, swarm_id
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        (
            actor.tenant_id,
            actor.run_id,
            actor.project_id,
            actor.repository_id,
            actor.swarm_id,
        ),
    )


@pytest.fixture
async def database() -> CockroachDatabase:
    assert DATABASE_URL is not None
    value = CockroachDatabase(DATABASE_URL, min_size=1, max_size=4)
    await value.start()
    try:
        yield value
    finally:
        await value.close()


async def test_idempotent_result_is_durable_and_scoped_by_run(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    command = _Command(idempotency_key="same-key", value="first")
    calls = 0

    async def body(connection: Any) -> _Result:
        nonlocal calls
        calls += 1
        await _insert_run(connection, actor)
        return _Result(value=command.value)

    first = await database.run_idempotent(actor, "test.insert-run", command, _Result, body)
    replay = await database.run_idempotent(actor, "test.insert-run", command, _Result, body)

    assert first.replayed is False
    assert replay == first.model_copy(update={"replayed": True})
    assert calls == 1

    other_run = actor.model_copy(update={"run_id": _id()})

    async def other_body(connection: Any) -> _Result:
        nonlocal calls
        calls += 1
        await _insert_run(connection, other_run)
        return _Result(value=command.value)

    other = await database.run_idempotent(
        other_run,
        "test.insert-run",
        command,
        _Result,
        other_body,
    )
    assert other.replayed is False
    assert calls == 2


async def test_ambiguous_ack_resolves_without_executing_body_twice(
    database: CockroachDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _actor()
    command = _Command(idempotency_key="ambiguous-once", value="committed")
    calls = 0
    original = database_module.run_serializable

    async def lose_ack(pool: Any, body: Any, **kwargs: Any) -> Any:
        await original(pool, body, **kwargs)
        raise AmbiguousTransactionResult("injected lost commit acknowledgement")

    async def body(connection: Any) -> _Result:
        nonlocal calls
        calls += 1
        await _insert_run(connection, actor)
        return _Result(value=command.value)

    monkeypatch.setattr(database_module, "run_serializable", lose_ack)
    result = await database.run_idempotent(
        actor,
        "test.ambiguous-insert",
        command,
        _Result,
        body,
    )

    assert result.value == "committed"
    assert result.replayed is True
    assert calls == 1


async def test_business_unique_violation_is_not_misclassified_as_idempotency(
    database: CockroachDatabase,
) -> None:
    actor = _actor()

    async def seed(connection: Any) -> _Result:
        await _insert_run(connection, actor)
        return _Result(value="seeded")

    await database.run(seed)
    command = _Command(idempotency_key="business-unique", value="duplicate")

    async def duplicate(connection: Any) -> _Result:
        await _insert_run(connection, actor)
        return _Result(value="unreachable")

    with pytest.raises(Exception) as caught:
        await database.run_idempotent(
            actor,
            "test.business-unique",
            command,
            _Result,
            duplicate,
        )

    assert getattr(caught.value, "sqlstate", None) == "23505"


async def test_live_serialization_race_retries_the_whole_transaction(
    database: CockroachDatabase,
) -> None:
    actor = _actor()

    async def seed(connection: Any) -> _Result:
        await _insert_run(connection, actor)
        await connection.execute(
            """
            UPDATE runs
            SET metadata = %s
            WHERE tenant_id = %s AND id = %s
            """,
            (Jsonb({"counter": 0}), actor.tenant_id, actor.run_id),
        )
        return _Result(value="seeded")

    await database.run(seed)
    both_read = asyncio.Event()
    first_reads = 0
    first_reads_lock = asyncio.Lock()
    body_calls = 0

    async def increment(connection: Any) -> _Result:
        nonlocal body_calls, first_reads
        body_calls += 1
        cursor = await connection.execute(
            """
            SELECT metadata
            FROM runs
            WHERE tenant_id = %s AND id = %s
            """,
            (actor.tenant_id, actor.run_id),
        )
        row = await cursor.fetchone()
        assert row is not None
        async with first_reads_lock:
            if first_reads < 2:
                first_reads += 1
                if first_reads == 2:
                    both_read.set()
                should_wait = True
            else:
                should_wait = False
        if should_wait:
            await both_read.wait()
        value = int(row["metadata"]["counter"]) + 1
        await connection.execute(
            """
            UPDATE runs
            SET metadata = %s
            WHERE tenant_id = %s AND id = %s
            """,
            (Jsonb({"counter": value}), actor.tenant_id, actor.run_id),
        )
        return _Result(value=str(value))

    await asyncio.gather(database.run(increment), database.run(increment))

    async def read_final(connection: Any) -> _Result:
        cursor = await connection.execute(
            """
            SELECT metadata
            FROM runs
            WHERE tenant_id = %s AND id = %s
            """,
            (actor.tenant_id, actor.run_id),
        )
        row = await cursor.fetchone()
        assert row is not None
        return _Result(value=str(row["metadata"]["counter"]))

    final = await database.run(read_final)
    assert final.value == "2"
    assert body_calls >= 3
