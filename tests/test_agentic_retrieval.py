from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from conftest import make_actor, make_task, new_id
from swarmbrain.application.errors import ResourceNotFound
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationTrigger,
    MemoryActivationCommand,
)
from swarmbrain.domain.exploration import ReadExpandMemoryRequest
from swarmbrain.domain.memory import (
    Memory,
    MemoryKind,
    MemoryLink,
    MemoryLinkKind,
    MemoryState,
    Visibility,
)
from swarmbrain.domain.tasks import ClaimTaskCommand
from swarmbrain.transports.http import create_app

SECRET = "0123456789abcdef-agentic-retrieval"


def _memory(
    actor: object,
    *,
    task_id: str,
    content: str,
    state: MemoryState = MemoryState.CONFIRMED,
    visibility: Visibility = Visibility.TASK,
) -> Memory:
    now = datetime.now(UTC) - timedelta(minutes=5)
    return Memory(
        memory_id=new_id(),
        tenant_id=actor.tenant_id,  # type: ignore[attr-defined]
        project_id=actor.project_id,  # type: ignore[attr-defined]
        repository_id=actor.repository_id,  # type: ignore[attr-defined]
        swarm_id=actor.swarm_id,  # type: ignore[attr-defined]
        run_id=actor.run_id,  # type: ignore[attr-defined]
        task_id=task_id,
        author_agent_id=new_id(),
        kind=MemoryKind.PROCEDURE,
        state=state,
        visibility=visibility,
        content=content,
        valid_from=now,
        recorded_from=now,
    )


@pytest.mark.parametrize(
    "trigger",
    (
        ActivationTrigger.TOOL_ERROR,
        ActivationTrigger.REPEATED_FAILURE,
        ActivationTrigger.EXPLICIT,
    ),
)
@pytest.mark.asyncio
async def test_owned_lease_wires_model_visible_activation_triggers(
    scope_ids: dict[str, str],
    trigger: ActivationTrigger,
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    task = make_task(scope_ids, title="Repair SQLSTATE 40001 retry")
    await runtime.kernel.add_task(task)
    memory = _memory(
        actor,
        task_id=task.task_id,
        content="SQLSTATE 40001 requires replaying the complete transaction",
        visibility=Visibility.RUN,
    )
    runtime.kernel.memories[memory.memory_id] = memory
    claimed = await runtime.coordination.claim(
        actor,
        ClaimTaskCommand(idempotency_key=f"claim-{trigger.value}", task_id=task.task_id),
    )
    secret_query = "SQLSTATE 40001 private tool diagnostic"

    delivery = await runtime.coordination.activate_memory(
        actor,
        MemoryActivationCommand(
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            trigger=trigger,
            query_text=secret_query,
        ),
    )
    replay = await runtime.coordination.activate_memory(
        actor,
        MemoryActivationCommand(
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            trigger=trigger,
            query_text="a retry may use different ephemeral query text",
        ),
    )

    assert delivery.activation.trigger is trigger
    assert delivery.activation.decision in {
        ActivationDecision.RECALL,
        ActivationDecision.DEEP_RECALL,
    }
    assert delivery.activation_context is not None
    assert memory.memory_id in delivery.activation_context
    assert secret_query not in delivery.model_dump_json()
    assert replay.replayed is True
    assert replay.activation == delivery.activation
    assert replay.activation_context is None


@pytest.mark.asyncio
async def test_explicit_activation_rejects_forged_task_or_lease_before_recall(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    task = make_task(scope_ids)
    await runtime.kernel.add_task(task)
    claimed = await runtime.coordination.claim(
        actor,
        ClaimTaskCommand(idempotency_key="claim-owned", task_id=task.task_id),
    )

    with pytest.raises(ResourceNotFound, match="active activation lease"):
        await runtime.coordination.activate_memory(
            actor,
            MemoryActivationCommand(
                task_id=task.task_id,
                lease_id=new_id(),
                trigger=ActivationTrigger.EXPLICIT,
                query_text="must not evaluate this query",
            ),
        )
    with pytest.raises(ValidationError, match="not caller-selectable"):
        MemoryActivationCommand(
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            trigger=ActivationTrigger.TASK_CLAIM,
            query_text="forged automatic trigger",
        )


@pytest.mark.asyncio
async def test_read_expand_is_canonical_depth_bounded_and_budgeted(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    task = make_task(scope_ids, title="Recover checkout retry")
    other_task = make_task(scope_ids, title="Unrelated private task")
    await runtime.kernel.add_task(task)
    await runtime.kernel.add_task(other_task)

    seed = _memory(actor, task_id=task.task_id, content="checkout retry runbook seed")
    neighbor = _memory(
        actor,
        task_id=task.task_id,
        content="checkout retry neighbor says use an injected clock",
    )
    refuted = _memory(
        actor,
        task_id=task.task_id,
        content="refuted checkout retry advice",
        state=MemoryState.REFUTED,
    )
    other_private = _memory(
        actor,
        task_id=other_task.task_id,
        content="other task private checkout retry secret",
    )
    for memory in (seed, neighbor, refuted, other_private):
        runtime.kernel.memories[memory.memory_id] = memory
    for target in (neighbor, refuted, other_private):
        link = MemoryLink(
            link_id=new_id(),
            source_memory_id=seed.memory_id,
            target_memory_id=target.memory_id,
            kind=MemoryLinkKind.RELATED_TO,
        )
        runtime.kernel.memory_links[link.link_id] = link

    claimed = await runtime.coordination.claim(
        actor,
        ClaimTaskCommand(idempotency_key="claim-read-expand", task_id=task.task_id),
    )
    result = await runtime.coordination.read_expand_memory(
        actor,
        ReadExpandMemoryRequest(
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            query_text="checkout retry injected clock",
            memory_ids=(seed.memory_id,),
            max_depth=1,
            max_fanout=4,
            token_budget=4_096,
        ),
    )

    assert result.memory_ids == (seed.memory_id, neighbor.memory_id)
    assert result.memory_versions == {seed.memory_id: 1, neighbor.memory_id: 1}
    assert result.estimated_tokens <= result.token_budget
    assert "checkout retry runbook seed" in result.context
    assert "injected clock" in result.context
    assert "refuted checkout retry advice" not in result.context
    assert "other task private" not in result.context
    assert result.provenance[seed.memory_id] == ("explicit_read",)
    assert "signal:graph" in result.provenance[neighbor.memory_id]


@pytest.mark.asyncio
async def test_read_expand_drops_oversized_content_without_exceeding_budget(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    task = make_task(scope_ids)
    await runtime.kernel.add_task(task)
    oversized = _memory(actor, task_id=task.task_id, content="x" * 20_000)
    runtime.kernel.memories[oversized.memory_id] = oversized
    claimed = await runtime.coordination.claim(
        actor,
        ClaimTaskCommand(idempotency_key="claim-small-budget", task_id=task.task_id),
    )

    result = await runtime.coordination.read_expand_memory(
        actor,
        ReadExpandMemoryRequest(
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            query_text="oversized",
            memory_ids=(oversized.memory_id,),
            max_depth=0,
            token_budget=32,
        ),
    )

    assert result.context == ""
    assert result.memory_ids == ()
    assert result.dropped_memory_ids == (oversized.memory_id,)
    assert result.estimated_tokens == 0
    assert result.truncated is True


@pytest.mark.asyncio
async def test_http_exposes_lease_bound_activation_and_read_expand_with_same_contracts(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    task = make_task(scope_ids, title="Repair checkout retry")
    await runtime.kernel.add_task(task)
    memory = _memory(
        actor,
        task_id=task.task_id,
        content="checkout retry uses an injected clock",
        visibility=Visibility.RUN,
    )
    runtime.kernel.memories[memory.memory_id] = memory
    token = runtime.tokens.issue(actor)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(runtime)),
        base_url="http://test",
    ) as client:
        claimed = await client.post(
            "/v1/tasks:claim",
            json={"task_id": task.task_id},
            headers={**headers, "Idempotency-Key": "http-agentic-claim"},
        )
        assert claimed.status_code == 200, claimed.text
        lease_id = claimed.json()["lease"]["lease_id"]
        activated = await client.post(
            f"/v1/tasks/{task.task_id}/memories:activate",
            json={
                "lease_id": lease_id,
                "trigger": "tool_error",
                "query_text": "checkout retry private error",
            },
            headers=headers,
        )
        expanded = await client.post(
            f"/v1/tasks/{task.task_id}/memories:read-expand",
            json={
                "lease_id": lease_id,
                "query_text": "checkout retry injected clock",
                "memory_ids": [memory.memory_id],
                "max_depth": 0,
                "token_budget": 1024,
            },
            headers=headers,
        )
        forged = await client.post(
            f"/v1/tasks/{task.task_id}/memories:read-expand",
            json={
                "task_id": new_id(),
                "lease_id": lease_id,
                "query_text": "must fail",
                "memory_ids": [memory.memory_id],
            },
            headers=headers,
        )

    assert activated.status_code == 200, activated.text
    assert activated.json()["activation"]["trigger"] == "tool_error"
    assert "query_text" not in activated.text
    assert expanded.status_code == 200, expanded.text
    assert expanded.json()["memory_ids"] == [memory.memory_id]
    assert "injected clock" in expanded.json()["context"]
    assert forged.status_code == 409
    assert forged.json()["error"]["code"] == "invalid_state"
