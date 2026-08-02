from __future__ import annotations

from collections.abc import Callable

import pytest

from swarmbrain.adapters.cockroach import (
    AmbiguousTransactionResult,
    RetryPolicy,
    run_serializable,
)


class DatabaseError(RuntimeError):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakeConnection:
    def transaction(self) -> FakeTransaction:
        return FakeTransaction()


class FakeConnectionLease:
    async def __aenter__(self) -> FakeConnection:
        return FakeConnection()

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakePool:
    def connection(self) -> FakeConnectionLease:
        return FakeConnectionLease()


@pytest.mark.asyncio
async def test_serialization_failures_retry_the_entire_transaction() -> None:
    calls = 0
    delays: list[float] = []

    async def body(_connection: FakeConnection) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise DatabaseError("40001")
        return "committed"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    result = await run_serializable(
        FakePool(),
        body,
        policy=RetryPolicy(
            max_attempts=4,
            base_delay_seconds=0.1,
            max_delay_seconds=1.0,
            jitter_seconds=0.0,
        ),
        sleep=fake_sleep,
        jitter=lambda _start, _end: 0.0,
    )

    assert result == "committed"
    assert calls == 3
    assert delays == [0.1, 0.2]


@pytest.mark.asyncio
async def test_ambiguous_commit_is_never_blindly_replayed() -> None:
    calls = 0

    async def body(_connection: FakeConnection) -> None:
        nonlocal calls
        calls += 1
        raise DatabaseError("40003")

    with pytest.raises(AmbiguousTransactionResult, match="idempotency key"):
        await run_serializable(FakePool(), body)

    assert calls == 1


@pytest.mark.asyncio
async def test_non_serialization_errors_are_not_retried() -> None:
    calls = 0

    async def body(_connection: FakeConnection) -> None:
        nonlocal calls
        calls += 1
        raise DatabaseError("23505")

    with pytest.raises(DatabaseError):
        await run_serializable(FakePool(), body)

    assert calls == 1


def test_retry_policy_rejects_invalid_values() -> None:
    invalid_factories: list[Callable[[], RetryPolicy]] = [
        lambda: RetryPolicy(max_attempts=0),
        lambda: RetryPolicy(base_delay_seconds=-0.1),
        lambda: RetryPolicy(max_delay_seconds=-0.1),
        lambda: RetryPolicy(jitter_seconds=-0.1),
    ]

    for factory in invalid_factories:
        with pytest.raises(ValueError):
            factory()
