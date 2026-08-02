from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")
TransactionBody = Callable[[Any], Awaitable[T]]
AttemptHook = Callable[[int, str | None], Awaitable[None]]


class AmbiguousTransactionResult(RuntimeError):
    """Raised when CockroachDB cannot prove whether COMMIT took effect.

    Callers must resolve the operation through its idempotency key instead of
    blindly replaying a mutation.
    """

    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_seconds: float = 0.025
    max_delay_seconds: float = 0.5
    jitter_seconds: float = 0.025

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if min(self.base_delay_seconds, self.max_delay_seconds, self.jitter_seconds) < 0:
            raise ValueError("retry delays must be non-negative")


async def run_serializable(
    pool: Any,
    body: TransactionBody[T],
    *,
    policy: RetryPolicy | None = None,
    on_attempt: AttemptHook | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> T:
    """Run one short SERIALIZABLE unit of work with bounded 40001 retries.

    `body` may perform database work only. Remote calls and other irreversible
    side effects belong after the transaction and must be driven from the
    transactional outbox.
    """

    resolved_policy = policy or RetryPolicy()
    for attempt in range(1, resolved_policy.max_attempts + 1):
        if on_attempt is not None:
            await on_attempt(attempt, None)
        try:
            async with pool.connection() as connection, connection.transaction():
                return await body(connection)
        except Exception as exc:
            sqlstate = getattr(exc, "sqlstate", None)
            if on_attempt is not None:
                await on_attempt(attempt, sqlstate)

            # 40003 means the commit result is indeterminate. The caller must
            # query idempotency_records; replaying here could duplicate effects.
            if sqlstate == "40003":
                raise AmbiguousTransactionResult(
                    "CockroachDB returned an ambiguous transaction result; "
                    "resolve it through the operation idempotency key",
                    sqlstate=sqlstate,
                ) from exc

            if sqlstate != "40001" or attempt >= resolved_policy.max_attempts:
                raise

            exponential = min(
                resolved_policy.base_delay_seconds * (2 ** (attempt - 1)),
                resolved_policy.max_delay_seconds,
            )
            await sleep(exponential + jitter(0.0, resolved_policy.jitter_seconds))

    raise AssertionError("retry loop exhausted without returning or raising")
