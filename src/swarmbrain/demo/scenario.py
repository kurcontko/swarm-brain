"""Deterministic content, fresh identity: the canonical swarm demo fixture.

Every invocation generates new UUIDs so repeated runs against a durable
backend append cleanly, while titles, evidence, and beat structure stay
byte-stable for scripted demos and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from swarmbrain.domain.agents import Capability

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

    @property
    def racers(self) -> tuple[DemoAgent, ...]:
        return self.workers

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
    """Twelve racing workers across four harness vendors, one lead, one scout."""

    workers = tuple(
        DemoAgent(
            name=f"worker-{index + 1:02d}-{harness}",
            harness=harness,
            provider=provider,
            model=model,
            capabilities=WORKER_CAPABILITIES,
        )
        for index, (harness, provider, model) in enumerate(
            _VENDOR_MIX[i % len(_VENDOR_MIX)] for i in range(12)
        )
    )
    lead = DemoAgent(
        name="lead-claude-code",
        harness="claude-code",
        provider="anthropic",
        model="claude-fable-5",
        capabilities=LEAD_CAPABILITIES,
    )
    scout = DemoAgent(
        name="scout-untrusted",
        harness="opencode",
        provider="qwen",
        model="qwen3-coder-27b",
        capabilities=SCOUT_CAPABILITIES,
    )
    tasks = (
        DemoTask(
            key="flaky-test",
            title="Reproduce the flaky checkout retry test",
            description=(
                "tests/integration/test_checkout_retry.py fails roughly one run in "
                "five on CI. Reproduce locally, capture the failing seed, and record "
                "what state the retry loop observes when it fails."
            ),
            tags=("ci", "flaky-test", "checkout"),
            priority=10,
        ),
        DemoTask(
            key="duplicate-charge",
            title="Trace the duplicate charge on webhook replay",
            description=(
                "Customers report double charges after payment-provider webhook "
                "retries. Trace the replayed delivery through the payments worker "
                "and identify why the second delivery is not treated as a replay."
            ),
            tags=("payments", "bug", "webhook"),
            priority=20,
        ),
        DemoTask(
            key="pool-exhaustion",
            title="Profile connection pool exhaustion under load",
            description=(
                "The API returns 503 bursts at peak traffic. Profile the database "
                "connection pool under load and report saturation, wait times, and "
                "the queries holding connections longest."
            ),
            tags=("performance", "database"),
            priority=0,
        ),
        DemoTask(
            key="replay-regression-test",
            title="Add regression coverage for idempotent webhook replay",
            description=(
                "Add a deterministic regression test that replays an identical "
                "webhook delivery twice and asserts exactly one charge is created."
            ),
            tags=("tests", "payments", "webhook"),
            priority=5,
        ),
        DemoTask(
            key="apply-fix",
            title="Apply the duplicate-charge fix behind a flag",
            description=(
                "Apply the fix for the duplicate webhook charge identified by the "
                "trace task, gated behind a rollout flag, once the root cause is "
                "confirmed."
            ),
            tags=("payments", "fix"),
            priority=30,
            depends_on=("duplicate-charge",),
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
    )


__all__ = [
    "LEAD_CAPABILITIES",
    "SCOUT_CAPABILITIES",
    "WORKER_CAPABILITIES",
    "DemoAgent",
    "DemoScenario",
    "DemoTask",
    "build_scenario",
]
