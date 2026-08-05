"""Leased workers for embeddings and external artifact persistence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from swarmbrain.domain.common import utc_now
from swarmbrain.domain.work import (
    HANDLER_WORK_KINDS,
    ApplyEmbeddingWorkCommand,
    ClaimWorkCommand,
    CompleteWorkCommand,
    CompleteWorkResult,
    EmbeddingWorkHandlerResult,
    FailWorkCommand,
    WorkHandlerResult,
    WorkKind,
    WorkLease,
    WorkLeaseLost,
    validate_work_payload,
)
from swarmbrain.ports.work_queue import WorkQueueStore


@runtime_checkable
class WorkHandler(Protocol):
    """Potentially remote or blocking side effect, always called outside queue TXs."""

    kind: WorkKind

    async def handle(
        self,
        lease: WorkLease,
    ) -> WorkHandlerResult | EmbeddingWorkHandlerResult: ...


class LeasedWorkWorker:
    """Claim, execute one typed handler, then fence completion or retry."""

    def __init__(
        self,
        queue: WorkQueueStore,
        handlers: Sequence[WorkHandler],
        *,
        retry_delay_seconds: int = 30,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        mapped: dict[WorkKind, WorkHandler] = {}
        for handler in handlers:
            if handler.kind not in HANDLER_WORK_KINDS:
                raise ValueError(f"unsupported handler kind: {handler.kind.value}")
            if handler.kind in mapped:
                raise ValueError(f"duplicate handler kind: {handler.kind.value}")
            mapped[handler.kind] = handler
        if not mapped:
            raise ValueError("at least one work handler is required")
        self.queue = queue
        self.handlers = mapped
        embedding_handler = mapped.get(WorkKind.EMBED_MEMORY)
        embedding_model = getattr(embedding_handler, "claim_model", None)
        if embedding_model is not None and (
            not isinstance(embedding_model, str) or not 1 <= len(embedding_model) <= 255
        ):
            raise ValueError("embedding handler claim_model must be a non-empty string")
        self.embedding_model = embedding_model
        self.retry_delay_seconds = retry_delay_seconds
        self.clock = clock

    async def run_once(
        self,
        worker_id: str,
        *,
        limit: int = 10,
        lease_seconds: int = 60,
    ) -> tuple[CompleteWorkResult, ...]:
        batch = await self.queue.claim_work(
            ClaimWorkCommand(
                worker_id=worker_id,
                kinds=frozenset(self.handlers),
                limit=limit,
                lease_seconds=lease_seconds,
                embedding_model=self.embedding_model,
            )
        )
        completed: list[CompleteWorkResult] = []
        for lease in batch.leases:
            result = await self._process(lease)
            if result is not None:
                completed.append(result)
        return tuple(completed)

    async def _process(self, lease: WorkLease) -> CompleteWorkResult | None:
        handler = self.handlers[lease.item.kind]
        try:
            validate_work_payload(lease.item.kind, lease.item.payload)
            # Handler/provider work deliberately occurs after claim COMMIT and
            # before the short fenced completion transaction.
            raw_result = await handler.handle(lease)
            if lease.item.kind is WorkKind.EMBED_MEMORY:
                handled_embedding = EmbeddingWorkHandlerResult.model_validate(raw_result)
                return await self.queue.apply_embedding(
                    ApplyEmbeddingWorkCommand(
                        work_id=lease.item.work_id,
                        worker_id=lease.worker_id,
                        lease_token=lease.lease_token,
                        lease_version=lease.lease_version,
                        expected_work_version=lease.work_version,
                        attempt=lease.attempt,
                        vector=handled_embedding.vector,
                        outcome=handled_embedding.outcome,
                        result=handled_embedding.result,
                    )
                )
            handled = WorkHandlerResult.model_validate(raw_result)
            return await self.queue.complete_work(
                CompleteWorkCommand(
                    work_id=lease.item.work_id,
                    worker_id=lease.worker_id,
                    lease_token=lease.lease_token,
                    lease_version=lease.lease_version,
                    expected_work_version=lease.work_version,
                    attempt=lease.attempt,
                    outcome=handled.outcome,
                    result=handled.result,
                )
            )
        except WorkLeaseLost:
            return None
        except Exception as exc:
            error = f"handler_{type(exc).__name__}"
            with suppress(WorkLeaseLost):
                await self.queue.fail_work(
                    FailWorkCommand(
                        work_id=lease.item.work_id,
                        worker_id=lease.worker_id,
                        lease_token=lease.lease_token,
                        lease_version=lease.lease_version,
                        expected_work_version=lease.work_version,
                        attempt=lease.attempt,
                        error=error,
                        retry_at=self.clock() + timedelta(seconds=self.retry_delay_seconds),
                    )
                )
            return None


__all__ = ["LeasedWorkWorker", "WorkHandler"]
