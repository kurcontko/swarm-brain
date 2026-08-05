"""Separate-process lifecycle for leased extraction and embedding workers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass

from swarmbrain.application.runtime import SwarmBrainRuntime
from swarmbrain.config import WorkerSettings


@dataclass(frozen=True, slots=True)
class WorkerCycleResult:
    extracted: int = 0
    embedded: int = 0

    @property
    def processed(self) -> int:
        return self.extracted + self.embedded


class WorkerSupervisor:
    """Own runtime startup/shutdown and poll only compatible durable queues."""

    def __init__(self, runtime: SwarmBrainRuntime, settings: WorkerSettings) -> None:
        self.runtime = runtime
        self.settings = settings

    async def run_once(self) -> WorkerCycleResult:
        extraction = self.runtime.extraction_worker(
            retry_delay_seconds=self.settings.retry_delay_seconds
        )
        extracted = await extraction.run_once(
            self.settings.worker_id,
            limit=self.settings.limit,
            lease_seconds=self.settings.lease_seconds,
        )
        embedded_count = 0
        if self.runtime.embeddings is not None:
            embedding = self.runtime.embedding_worker(
                retry_delay_seconds=self.settings.retry_delay_seconds
            )
            embedded = await embedding.run_once(
                self.settings.worker_id,
                limit=self.settings.limit,
                lease_seconds=self.settings.lease_seconds,
            )
            embedded_count = len(embedded)
        return WorkerCycleResult(extracted=len(extracted), embedded=embedded_count)

    async def serve(
        self,
        stop: asyncio.Event,
        *,
        once: bool = False,
    ) -> WorkerCycleResult:
        """Run one cycle or poll until stopped, always closing the owned runtime."""

        total = WorkerCycleResult()
        await self.runtime.start()
        try:
            while not stop.is_set():
                cycle = await self.run_once()
                total = WorkerCycleResult(
                    extracted=total.extracted + cycle.extracted,
                    embedded=total.embedded + cycle.embedded,
                )
                if once:
                    break
                if cycle.processed:
                    continue
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        stop.wait(),
                        timeout=self.settings.poll_interval_seconds,
                    )
            return total
        finally:
            await self.runtime.close()


__all__ = ["WorkerCycleResult", "WorkerSupervisor"]
