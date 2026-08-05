from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from conftest import make_actor
from swarmbrain.adapters.extraction import (
    STRUCTURED_MEMORY_SOURCE_KIND,
    DefaultRuleExtractor,
    InMemoryWorkStore,
)
from swarmbrain.application.extraction import ExtractionService
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.config import WorkerSettings
from swarmbrain.domain.extraction import IngestRawSourceCommand
from swarmbrain.workers.runtime import WorkerSupervisor

SECRET = "0123456789abcdef-local-test-secret"


@pytest.mark.asyncio
async def test_worker_supervisor_runs_bounded_cycle_and_closes_runtime(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    extraction = ExtractionService(store, DefaultRuleExtractor())
    runtime = replace(
        build_in_memory_runtime(SECRET),
        extraction=extraction,
        source_store=store,
        work_queue=store,
    )
    actor = make_actor(scope_ids)
    await extraction.ingest(
        actor,
        IngestRawSourceCommand(
            idempotency_key="supervisor-ingest-1",
            kind=STRUCTURED_MEMORY_SOURCE_KIND,
            content=json.dumps({"memories": [{"kind": "observation", "content": {"ready": True}}]}),
            observed_at=datetime(2026, 8, 3, 10, 0, tzinfo=UTC),
        ),
    )
    supervisor = WorkerSupervisor(
        runtime,
        WorkerSettings(
            worker_id="supervisor-test",
            poll_interval_seconds=0.01,
            lease_seconds=10,
            retry_delay_seconds=1,
        ),
    )

    result = await supervisor.serve(asyncio.Event(), once=True)

    assert result.extracted == 1
    assert result.embedded == 0
    assert result.processed == 1


def test_worker_settings_keep_batch_at_one_until_lease_heartbeat_exists() -> None:
    with pytest.raises(ValueError, match="limit must be 1"):
        WorkerSettings(worker_id="unsafe-batch", limit=2)
