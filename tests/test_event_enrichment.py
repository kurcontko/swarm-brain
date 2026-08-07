"""Events must read like sentences, and both adapters must stamp the same keys.

The console shows agents and tasks by name, so every emitter adds the same
additive payload keys. Nothing here may become a required envelope field: an
old reader must keep parsing a new event, and a new reader must keep rendering
an old one.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from conftest import ALL_CAPABILITIES, make_actor, make_task, new_id
from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.events import (
    EVENT_ENRICHMENT_KEYS,
    EventType,
    SwarmEvent,
    agent_display_label,
    enriched_payload,
    event_enrichment,
)
from swarmbrain.domain.leases import RenewLeaseCommand
from swarmbrain.domain.memory import MemoryKind, RememberCommand
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    ClaimTaskCommand,
    CompleteTaskCommand,
    ReleaseTaskCommand,
)

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")

AGENT_KEYS = frozenset({"agent_harness", "agent_provider", "agent_model", "agent_display"})
TASK_KEYS = AGENT_KEYS | {"task_title"}


class _Ack(ContractModel):
    ok: bool = True


def _enrichment(event: SwarmEvent) -> dict[str, Any]:
    return {key: value for key, value in event.payload.items() if key in EVENT_ENRICHMENT_KEYS}


# ---------------------------------------------------------------------------
# The helper itself


def test_enrichment_is_additive_and_never_overwrites_an_emitters_own_keys() -> None:
    merged = enriched_payload(
        {"outcome": "succeeded", "agent_display": "already mine"},
        harness="claude-code",
        provider="anthropic",
        model="claude-sonnet-5",
        task_title="Trace the duplicate charge",
    )

    assert merged["outcome"] == "succeeded"
    assert merged["agent_display"] == "already mine"
    assert merged["agent_harness"] == "claude-code"
    assert merged["task_title"] == "Trace the duplicate charge"


def test_display_label_degrades_when_a_field_is_missing() -> None:
    assert agent_display_label("claude-code", "anthropic") == "claude-code (anthropic)"
    assert agent_display_label("claude-code", "") == "claude-code"
    assert agent_display_label("", "anthropic") is None
    assert event_enrichment() == {}


def test_events_without_enrichment_still_validate_on_the_wire() -> None:
    """Wire compatibility: no enrichment key became a required envelope field."""

    scope = {name: new_id() for name in ("tenant_id", "project_id", "repository_id")}
    legacy = SwarmEvent(
        event_id=new_id(),
        event_type=EventType.TASK_CLAIMED,
        aggregate_type="task",
        aggregate_id=new_id(),
        **scope,
        swarm_id=new_id(),
        run_id=new_id(),
        aggregate_version=1,
        payload={"lease_id": new_id()},
    )

    assert _enrichment(legacy) == {}
    assert SwarmEvent.model_validate(legacy.model_dump(mode="json")) == legacy


# ---------------------------------------------------------------------------
# In-memory adapter


async def _drive_in_memory(scope: dict[str, str]) -> dict[str, dict[str, Any]]:
    kernel = InMemoryKernel()
    actor = make_actor(scope, harness="claude-code", provider="anthropic", model="claude-sonnet-5")
    await kernel.join_agent(actor)
    task = await kernel.add_task(make_task(scope, title="Trace the duplicate charge"))

    claimed = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="claim", task_id=task.task_id, lease_seconds=120),
    )
    renewed = await kernel.renew_lease(
        actor,
        RenewLeaseCommand(
            idempotency_key="renew",
            lease_id=claimed.lease.lease_id,
            expected_version=claimed.lease.version,
            extension_seconds=120,
        ),
    )
    checkpointed = await kernel.checkpoint_task(
        actor,
        CheckpointCommand(
            idempotency_key="checkpoint",
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            expected_task_version=claimed.task.version,
            expected_lease_version=renewed.lease.version,
            summary="halfway",
        ),
    )
    await kernel.remember(
        actor,
        RememberCommand(
            idempotency_key="remember",
            kind=MemoryKind.OBSERVATION,
            content="the webhook replay is not deduplicated",
            task_id=task.task_id,
        ),
        ConservativeMemoryPolicy(),
    )
    released = await kernel.release_task(
        actor,
        ReleaseTaskCommand(
            idempotency_key="release",
            task_id=task.task_id,
            lease_id=checkpointed.lease.lease_id,
            expected_task_version=checkpointed.task.version,
            expected_lease_version=checkpointed.lease.version,
            reason="handing back",
        ),
    )
    reclaimed = await kernel.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="reclaim", task_id=task.task_id, lease_seconds=120),
    )
    await kernel.complete_task(
        actor,
        CompleteTaskCommand(
            idempotency_key="complete",
            task_id=task.task_id,
            lease_id=reclaimed.lease.lease_id,
            expected_task_version=reclaimed.task.version,
            expected_lease_version=reclaimed.lease.version,
            summary="root cause recorded",
        ),
    )
    del released
    return {str(event.event_type): _enrichment(event) for event in kernel.events}


@pytest.mark.asyncio
async def test_in_memory_events_name_the_agent_and_the_task(
    scope_ids: dict[str, str],
) -> None:
    stamped = await _drive_in_memory(scope_ids)

    for event_type in (
        EventType.AGENT_JOINED,
        EventType.TASK_CLAIMED,
        EventType.TASK_CHECKPOINTED,
        EventType.TASK_COMPLETED,
        EventType.TASK_RELEASED,
        EventType.MEMORY_ADDED,
    ):
        assert str(event_type) in stamped, event_type

    assert stamped[EventType.TASK_CLAIMED.value] == {
        "task_title": "Trace the duplicate charge",
        "agent_harness": "claude-code",
        "agent_provider": "anthropic",
        "agent_model": "claude-sonnet-5",
        "agent_display": "claude-code (anthropic)",
    }
    # Memory and agent events carry the actor, but no task title to carry.
    assert set(stamped[EventType.MEMORY_ADDED.value]) == AGENT_KEYS
    assert set(stamped[EventType.AGENT_JOINED.value]) == AGENT_KEYS
    for event_type in (
        EventType.TASK_CHECKPOINTED,
        EventType.TASK_COMPLETED,
        EventType.TASK_RELEASED,
    ):
        assert set(stamped[str(event_type)]) == TASK_KEYS, event_type


# ---------------------------------------------------------------------------
# Adapter parity


@pytest.mark.asyncio
@pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set")
async def test_both_adapters_stamp_identical_enrichment_for_the_same_operations(
    scope_ids: dict[str, str],
) -> None:
    """The durable adapter must not drift from the local one, key for key."""

    assert DATABASE_URL is not None
    database = CockroachDatabase(DATABASE_URL, min_size=1, max_size=4)
    await database.start()
    store = CockroachCoordinationStore(database)
    actor = make_actor(
        scope_ids,
        harness="claude-code",
        provider="anthropic",
        model="claude-sonnet-5",
        capabilities=ALL_CAPABILITIES,
    )
    try:
        await store.join_agent(actor)
        task = await store.add_task(make_task(scope_ids, title="Trace the duplicate charge"))
        claimed = await store.claim_task(
            actor,
            ClaimTaskCommand(idempotency_key="claim", task_id=task.task_id, lease_seconds=120),
        )
        renewed = await store.renew_lease(
            actor,
            RenewLeaseCommand(
                idempotency_key="renew",
                lease_id=claimed.lease.lease_id,
                expected_version=claimed.lease.version,
                extension_seconds=120,
            ),
        )
        checkpointed = await store.checkpoint_task(
            actor,
            CheckpointCommand(
                idempotency_key="checkpoint",
                task_id=task.task_id,
                lease_id=claimed.lease.lease_id,
                expected_task_version=claimed.task.version,
                expected_lease_version=renewed.lease.version,
                summary="halfway",
            ),
        )
        released = await store.release_task(
            actor,
            ReleaseTaskCommand(
                idempotency_key="release",
                task_id=task.task_id,
                lease_id=checkpointed.lease.lease_id,
                expected_task_version=checkpointed.task.version,
                expected_lease_version=checkpointed.lease.version,
                reason="handing back",
            ),
        )
        del released
        reclaimed = await store.claim_task(
            actor,
            ClaimTaskCommand(idempotency_key="reclaim", task_id=task.task_id, lease_seconds=120),
        )
        await store.complete_task(
            actor,
            CompleteTaskCommand(
                idempotency_key="complete",
                task_id=task.task_id,
                lease_id=reclaimed.lease.lease_id,
                expected_task_version=reclaimed.task.version,
                expected_lease_version=reclaimed.lease.version,
                summary="root cause recorded",
            ),
        )

        # Read the payloads back out of the durable column, not out of memory.
        page = await store.list_run_events(actor, actor.run_id, limit=100)
        durable = {str(event.event_type): _enrichment(event) for event in page.events}
        local = await _drive_in_memory(scope_ids)

        shared = set(durable) & set(local)
        assert {
            EventType.AGENT_JOINED.value,
            EventType.TASK_CLAIMED.value,
            EventType.TASK_CHECKPOINTED.value,
            EventType.TASK_COMPLETED.value,
            EventType.TASK_RELEASED.value,
        } <= shared
        for event_type in sorted(shared):
            assert durable[event_type] == local[event_type], event_type
    finally:

        async def cleanup(connection: Any) -> _Ack:
            params = (scope_ids["tenant_id"], scope_ids["run_id"])
            for table in ("outbox_events", "action_attempts", "idempotency_records"):
                await connection.execute(
                    f"DELETE FROM {table} WHERE tenant_id = %s AND run_id = %s",
                    params,
                )
            await connection.execute(
                "DELETE FROM runs WHERE tenant_id = %s AND id = %s",
                params,
            )
            return _Ack()

        await database.run(cleanup)
        await database.close()
