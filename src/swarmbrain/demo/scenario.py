"""Deterministic content, fresh identity: the canonical swarm demo fixture.

Every invocation generates new UUIDs so repeated runs against a durable
backend append cleanly, while titles, evidence, and beat structure stay
byte-stable for scripted demos and tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from uuid import uuid4

from swarmbrain.domain.agents import ActorContext, Capability

WORKER_CAPABILITIES = frozenset(
    {
        Capability.RUN_JOIN.value,
        Capability.TASK_CLAIM.value,
        Capability.TASK_CHECKPOINT.value,
        Capability.TASK_COMPLETE.value,
        Capability.TASK_RELEASE.value,
        Capability.LEASE_RENEW.value,
        Capability.MEMORY_PUBLISH.value,
        Capability.MEMORY_RECALL.value,
        Capability.MEMORY_CONFIRM.value,
        Capability.SOURCE_INGEST.value,
        Capability.CONFLICT_REPORT.value,
    }
)

LEAD_CAPABILITIES = frozenset(item.value for item in Capability)

SCOUT_CAPABILITIES = frozenset(
    {
        Capability.RUN_JOIN.value,
        Capability.MEMORY_PUBLISH.value,
        Capability.MEMORY_RECALL.value,
    }
)


@dataclass(frozen=True, slots=True)
class DemoAgent:
    """One roster member; harness/provider/model label the vendor it stands in for."""

    name: str
    harness: str
    provider: str
    model: str
    capabilities: frozenset[str]
    agent_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class DemoTask:
    key: str
    title: str
    description: str
    tags: tuple[str, ...]
    priority: int = 0
    depends_on: tuple[str, ...] = ()
    task_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class DemoChallenge:
    """Hidden verifier material; opaque values enter agent context only via memory."""

    guard_token: str
    procedure_token: str
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class DemoScenario:
    tenant_id: str
    project_id: str
    repository_id: str
    swarm_id: str
    run_id: str
    lead: DemoAgent
    scout: DemoAgent
    workers: tuple[DemoAgent, ...]
    tasks: tuple[DemoTask, ...]
    challenge: DemoChallenge | None = None

    @property
    def racers(self) -> tuple[DemoAgent, ...]:
        return self.all_agents

    @property
    def all_agents(self) -> tuple[DemoAgent, ...]:
        return (self.lead, self.scout, *self.workers)

    def task(self, key: str) -> DemoTask:
        for task in self.tasks:
            if task.key == key:
                return task
        raise KeyError(key)

    def scope(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "repository_id": self.repository_id,
            "swarm_id": self.swarm_id,
            "run_id": self.run_id,
        }


_VENDOR_MIX: tuple[tuple[str, str, str], ...] = (
    ("claude-code", "anthropic", "claude-sonnet-5"),
    ("codex-cli", "openai", "gpt-5-codex"),
    ("gemini-cli", "google", "gemini-3-pro"),
    ("opencode", "qwen", "qwen3-coder-27b"),
)


def build_scenario() -> DemoScenario:
    """Four agents across four vendors in a two-wave, memory-causal task DAG."""

    guard_token = uuid4().hex
    procedure_token = uuid4().hex
    expected_sha256 = hashlib.sha256(f"{guard_token}\n{procedure_token}".encode()).hexdigest()
    lead = DemoAgent(
        name="investigator-01-claude-code",
        harness=_VENDOR_MIX[0][0],
        provider=_VENDOR_MIX[0][1],
        model=_VENDOR_MIX[0][2],
        capabilities=LEAD_CAPABILITIES,
    )
    scout = DemoAgent(
        # ``scout`` is retained as a constructor/API compatibility slot. Here it
        # is the second task-capable investigator, not a fifth observer.
        name="investigator-02-codex-cli",
        harness=_VENDOR_MIX[1][0],
        provider=_VENDOR_MIX[1][1],
        model=_VENDOR_MIX[1][2],
        capabilities=WORKER_CAPABILITIES,
    )
    workers = tuple(
        DemoAgent(
            name=name,
            harness=harness,
            provider=provider,
            model=model,
            capabilities=WORKER_CAPABILITIES,
        )
        for name, (harness, provider, model) in zip(
            ("builder-03-gemini-cli", "verifier-04-opencode"),
            _VENDOR_MIX[2:],
            strict=True,
        )
    )
    tasks = (
        DemoTask(
            key="replay-invariant",
            title="Identify the webhook replay identity invariant",
            description=(
                "Inspect the payment webhook trace and determine which stable delivery "
                "identity must be reserved to prevent a replayed delivery from charging twice."
            ),
            tags=("payments", "webhook", "replay", "identity"),
            priority=20,
        ),
        DemoTask(
            key="replay-procedure",
            title="Derive the safe webhook replay procedure",
            description=(
                "Reproduce the duplicate charge and establish the verified ordering "
                "for reserving the replay guard before the charge side effect."
            ),
            tags=("payments", "webhook", "replay", "procedure"),
            priority=20,
        ),
        DemoTask(
            key="implement-replay-guard",
            title="Implement the webhook replay identity guard procedure",
            description="Webhook replay identity guard procedure: reserve before charge.",
            tags=("webhook", "replay", "identity", "guard", "procedure"),
            priority=10,
            depends_on=("replay-invariant", "replay-procedure"),
        ),
        DemoTask(
            key="verify-replay-guard",
            title="Verify the webhook replay identity guard procedure",
            description="Webhook replay identity guard procedure: reserve before charge.",
            tags=("webhook", "replay", "identity", "guard", "procedure"),
            priority=10,
            depends_on=("replay-invariant", "replay-procedure"),
        ),
    )
    return DemoScenario(
        tenant_id=str(uuid4()),
        project_id=str(uuid4()),
        repository_id=str(uuid4()),
        swarm_id=str(uuid4()),
        run_id=str(uuid4()),
        lead=lead,
        scout=scout,
        workers=workers,
        tasks=tasks,
        challenge=DemoChallenge(
            guard_token=guard_token,
            procedure_token=procedure_token,
            expected_sha256=expected_sha256,
        ),
    )


def actor_context(scenario: DemoScenario, agent: DemoAgent) -> ActorContext:
    """The scoped identity a demo agent presents on the agent plane."""

    return ActorContext(
        **scenario.scope(),
        agent_id=agent.agent_id,
        harness=agent.harness,
        provider=agent.provider,
        model=agent.model,
        capabilities=agent.capabilities,
    )


__all__ = [
    "LEAD_CAPABILITIES",
    "SCOUT_CAPABILITIES",
    "WORKER_CAPABILITIES",
    "DemoAgent",
    "DemoChallenge",
    "DemoScenario",
    "DemoTask",
    "actor_context",
    "build_scenario",
]
