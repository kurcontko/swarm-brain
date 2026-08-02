from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.tasks import Task, TaskStatus

ALL_CAPABILITIES = frozenset(item.value for item in Capability)


def new_id() -> str:
    return str(uuid4())


@pytest.fixture
def scope_ids() -> dict[str, str]:
    return {
        "tenant_id": new_id(),
        "project_id": new_id(),
        "repository_id": new_id(),
        "swarm_id": new_id(),
        "run_id": new_id(),
    }


def make_actor(
    scope: dict[str, str],
    *,
    agent_id: str | None = None,
    harness: str = "codex-cli",
    provider: str = "openai",
    model: str = "gpt",
    capabilities: frozenset[str] = ALL_CAPABILITIES,
) -> ActorContext:
    return ActorContext(
        **scope,
        agent_id=agent_id or new_id(),
        harness=harness,
        provider=provider,
        model=model,
        capabilities=capabilities,
    )


def make_task(
    scope: dict[str, str],
    *,
    task_id: str | None = None,
    title: str = "Investigate failing integration",
    priority: int = 0,
    created_at: datetime | None = None,
) -> Task:
    timestamp = created_at or datetime.now(UTC)
    return Task(
        **scope,
        task_id=task_id or new_id(),
        title=title,
        status=TaskStatus.READY,
        priority=priority,
        created_at=timestamp,
        updated_at=timestamp,
    )
