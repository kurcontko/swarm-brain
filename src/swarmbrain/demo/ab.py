"""A/B comparison: the same task set with and without the shared swarm brain.

Two arms over one fixture:

- **Arm B (with swarm brain)** is *measured*. It runs the ordinary demo over the
  canonical HTTP API and then reads every figure back out of the run's event
  stream, run metrics, and beat data. Nothing here is hand-entered.
- **Arm A (without swarm brain)** is an *explicitly simulated* uncoordinated
  fleet. No API call is faked and no failure is injected; it is a deterministic
  bookkeeping model of what the same roster does on the same tasks when nothing
  is shared across vendors. It is labelled as a simulation everywhere it is
  reported, and its assumptions are carried in the artifact.

Only the per-task step costs are fixture constants, and they are applied
identically to both arms. What differs between the arms is *how many times*
each unit of work happens — and for Arm B those counts come from the run.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.ports.coordination_store import CoordinationStore

from .runner import DEFAULT_LEASE_SECONDS, TOKEN_TTL, DemoAssertionError, DemoReport, DemoRunner
from .scenario import DemoScenario, actor_context, build_scenario

BASELINE_KIND = "simulated_uncoordinated_baseline"
SWARM_KIND = "measured_swarm_brain_run"

BASELINE_LABEL = "Arm A — without swarm brain"
SWARM_LABEL = "Arm B — with swarm brain"


# -- workload fixture -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkloadTask:
    """One task's cost model, shared byte-for-byte by both arms."""

    key: str
    steps: int
    investigative: bool
    derives_discovery: bool = False
    needs_discovery: bool = False


@dataclass(frozen=True, slots=True)
class Workload:
    tasks: tuple[WorkloadTask, ...]
    discovery_steps: int
    crash_task_key: str
    crash_after_steps: int

    def task(self, key: str) -> WorkloadTask:
        for task in self.tasks:
            if task.key == key:
                return task
        raise KeyError(key)

    @property
    def investigative_keys(self) -> tuple[str, ...]:
        return tuple(task.key for task in self.tasks if task.investigative)

    @property
    def minimum_executions(self) -> int:
        return len(self.tasks)

    @property
    def minimum_investigations(self) -> int:
        return len(self.investigative_keys)

    @property
    def minimum_discovery_derivations(self) -> int:
        return sum(task.derives_discovery for task in self.tasks)

    @property
    def minimum_steps(self) -> int:
        """Every task and required discovery done once, with nothing redone."""

        return (
            sum(task.steps for task in self.tasks)
            + self.minimum_discovery_derivations * self.discovery_steps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [
                {
                    "key": task.key,
                    "steps": task.steps,
                    "investigative": task.investigative,
                    "derives_discovery": task.derives_discovery,
                    "needs_discovery": task.needs_discovery,
                }
                for task in self.tasks
            ],
            "discovery_steps": self.discovery_steps,
            "crash_task_key": self.crash_task_key,
            "crash_after_steps": self.crash_after_steps,
            "minimum_executions": self.minimum_executions,
            "minimum_investigations": self.minimum_investigations,
            "minimum_discovery_derivations": self.minimum_discovery_derivations,
            "minimum_steps": self.minimum_steps,
        }


def build_workload() -> Workload:
    """Cost model over the canonical demo DAG; keys mirror ``build_scenario``."""

    return Workload(
        tasks=(
            WorkloadTask(
                key="replay-invariant",
                steps=5,
                investigative=True,
                derives_discovery=True,
            ),
            WorkloadTask(
                key="replay-procedure",
                steps=5,
                investigative=True,
                derives_discovery=True,
            ),
            WorkloadTask(
                key="implement-replay-guard",
                steps=4,
                investigative=False,
                needs_discovery=True,
            ),
            WorkloadTask(
                key="verify-replay-guard",
                steps=3,
                investigative=False,
                needs_discovery=True,
            ),
        ),
        discovery_steps=4,
        crash_task_key="implement-replay-guard",
        crash_after_steps=3,
    )


WORKLOAD = build_workload()


def _total_steps(
    workload: Workload,
    executions: Counter[str],
    discovery_derivations: int,
    steps_redone: int,
) -> int:
    """One scoring function, two arms: only the counts differ."""

    return (
        sum(workload.task(key).steps * count for key, count in executions.items())
        + discovery_derivations * workload.discovery_steps
        + steps_redone
    )


# -- per-arm result ---------------------------------------------------------


@dataclass(slots=True)
class ArmMetrics:
    label: str
    kind: str
    fleets: int
    workers: int
    task_executions: int
    distinct_tasks: int
    duplicate_task_executions: int
    investigations_performed: int
    investigations_minimum: int
    duplicate_investigations: int
    discovery_derivations: int
    discovery_derivations_minimum: int
    discoveries_rederived: int
    discovery_reuses: int
    crash_steps_lost: int
    crash_steps_preserved: int
    duplicate_executions_prevented: int
    total_steps: int
    minimum_steps: int
    executions_by_task: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def simulated(self) -> bool:
        return self.kind == BASELINE_KIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "simulated": self.simulated,
            "fleets": self.fleets,
            "workers": self.workers,
            "task_executions": self.task_executions,
            "distinct_tasks": self.distinct_tasks,
            "duplicate_task_executions": self.duplicate_task_executions,
            "investigations_performed": self.investigations_performed,
            "investigations_minimum": self.investigations_minimum,
            "duplicate_investigations": self.duplicate_investigations,
            "discovery_derivations": self.discovery_derivations,
            "discovery_derivations_minimum": self.discovery_derivations_minimum,
            "discoveries_rederived": self.discoveries_rederived,
            "discovery_reuses": self.discovery_reuses,
            "crash_steps_lost": self.crash_steps_lost,
            "crash_steps_preserved": self.crash_steps_preserved,
            "duplicate_executions_prevented": self.duplicate_executions_prevented,
            "total_steps": self.total_steps,
            "minimum_steps": self.minimum_steps,
            "executions_by_task": dict(sorted(self.executions_by_task.items())),
            "notes": list(self.notes),
            "sources": dict(sorted(self.sources.items())),
        }


# -- Arm A: simulated uncoordinated baseline --------------------------------


BASELINE_NOTES: tuple[str, ...] = (
    "SIMULATED. No HTTP call is made and no failure is injected; this arm is a "
    "deterministic bookkeeping model, not an API run.",
    "Coordination unit is the vendor harness: agents inside one harness share "
    "its own session context, so they do not duplicate each other's work. "
    "Nothing crosses harness boundaries.",
    "This is generous to the baseline. Modelling every worker independently "
    "would multiply the duplication by the fleet size instead of the vendor count.",
    "No transactional claim: each fleet self-assigns from the same priority-"
    "ordered board and works any task whose result it has not itself produced.",
    "No cross-agent recall: a fleet that needs the root-cause discoveries derives "
    "them itself, because another vendor's derivations are invisible to it.",
    "No durable checkpoint: the crashed worker's partial progress dies with it "
    "and its replacement restarts that task from zero.",
)


def simulate_uncoordinated_baseline(
    scenario: DemoScenario,
    workload: Workload = WORKLOAD,
) -> ArmMetrics:
    """Deterministic simulation of the same roster with nothing shared."""

    fleets = tuple(dict.fromkeys(agent.provider for agent in scenario.all_agents))
    executions: Counter[str] = Counter()
    investigations = 0
    derivations = 0
    steps_redone = 0
    crash_used = False

    for _provider in fleets:
        # Each fleet only ever sees results it produced itself.
        fleet_results: set[str] = set()
        known_discoveries = 0
        for task in workload.tasks:
            if task.key in fleet_results:
                continue
            if task.derives_discovery:
                derivations += 1
                known_discoveries += 1
            elif (
                task.needs_discovery
                and (missing := workload.minimum_discovery_derivations - known_discoveries) > 0
            ):
                derivations += missing
                known_discoveries += missing
            executions[task.key] += 1
            if task.investigative:
                investigations += 1
            if task.key == workload.crash_task_key and not crash_used:
                # One worker dies mid-task; with no checkpoint its progress is
                # unrecoverable, so the replacement redoes those steps.
                crash_used = True
                steps_redone += workload.crash_after_steps
            fleet_results.add(task.key)

    total_executions = sum(executions.values())
    return ArmMetrics(
        label=f"{BASELINE_LABEL} (simulated uncoordinated fleet)",
        kind=BASELINE_KIND,
        fleets=len(fleets),
        workers=len(scenario.all_agents),
        task_executions=total_executions,
        distinct_tasks=len(executions),
        duplicate_task_executions=total_executions - len(executions),
        investigations_performed=investigations,
        investigations_minimum=workload.minimum_investigations,
        duplicate_investigations=investigations - workload.minimum_investigations,
        discovery_derivations=derivations,
        discovery_derivations_minimum=workload.minimum_discovery_derivations,
        discoveries_rederived=derivations - workload.minimum_discovery_derivations,
        discovery_reuses=0,
        crash_steps_lost=steps_redone,
        crash_steps_preserved=0,
        duplicate_executions_prevented=0,
        total_steps=_total_steps(workload, executions, derivations, steps_redone),
        minimum_steps=workload.minimum_steps,
        executions_by_task=dict(executions),
        notes=BASELINE_NOTES,
        sources={"all": "simulation over the shared workload fixture"},
    )


# -- Arm B: measured swarm-brain run ----------------------------------------


async def _fetch_events(
    client: httpx.AsyncClient,
    token: str,
    run_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor: str | None = None
    headers = {"Authorization": f"Bearer {token}"}
    for _ in range(20):
        params: dict[str, Any] = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        response = await client.get(f"/v1/runs/{run_id}/events", headers=headers, params=params)
        response.raise_for_status()
        page = response.json()
        events.extend(page["events"])
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return events


async def _fetch_metrics(
    client: httpx.AsyncClient,
    token: str,
    run_id: str,
) -> dict[str, Any]:
    response = await client.get(
        f"/v1/runs/{run_id}/metrics", headers={"Authorization": f"Bearer {token}"}
    )
    response.raise_for_status()
    return dict(response.json())


def _check_ok(report: DemoReport, beat_key: str, check_name: str) -> bool:
    for beat in report.beats:
        if beat.key != beat_key:
            continue
        for check in beat.checks:
            if check.name == check_name:
                return check.ok
    return False


def measure_swarm_arm(
    scenario: DemoScenario,
    report: DemoReport,
    events: list[dict[str, Any]],
    metrics: dict[str, Any],
    workload: Workload = WORKLOAD,
) -> ArmMetrics:
    """Read Arm B's numbers back out of the run that just happened."""

    key_by_task_id = {task.task_id: task.key for task in scenario.tasks}
    fresh: Counter[str] = Counter()
    resumptions: Counter[str] = Counter()
    checkpointed: set[str] = set()
    # A claim on a task that already has a durable checkpoint resumes preserved
    # work; a claim on a task with no checkpoint starts that task from scratch.
    for event in events:
        task_id = event.get("task_id")
        if task_id is None:
            continue
        key = key_by_task_id.get(task_id)
        if key is None:
            continue
        event_type = event["event_type"]
        if event_type == "task.checkpointed":
            checkpointed.add(task_id)
        elif event_type == "task.claimed":
            if task_id in checkpointed:
                resumptions[key] += 1
            else:
                fresh[key] += 1

    executions = sum(fresh.values())
    investigations = sum(fresh[key] for key in workload.investigative_keys)

    derivations = (
        workload.minimum_discovery_derivations
        if _check_ok(report, "shared_discovery", "two_confirmed_memories_published")
        else 0
    )
    memory_authors = {
        str(event["aggregate_id"]): str(event["agent_id"])
        for event in events
        if event["event_type"] in {"memory.added", "memory.superseded"} and event.get("agent_id")
    }
    citations = {
        (str(event["task_id"]), str(event["agent_id"]), str(memory_id))
        for event in events
        if event["event_type"] in {"task.checkpointed", "task.completed"}
        and event.get("task_id")
        and event.get("agent_id")
        for memory_id in (event.get("payload") or {}).get("memory_ids", [])
    }
    reuses = sum(
        memory_authors.get(memory_id) is not None and memory_authors[memory_id] != consumer_id
        for _task_id, consumer_id, memory_id in citations
    )

    handed_off = (
        _check_ok(report, "crash_handoff", "successor_resumes_with_checkpoint")
        and int(metrics.get("crash_handoffs", 0)) >= 1
        and resumptions[workload.crash_task_key] >= 1
    )
    preserved = workload.crash_after_steps if handed_off else 0
    lost = workload.crash_after_steps - preserved

    return ArmMetrics(
        label=f"{SWARM_LABEL} (measured live run)",
        kind=SWARM_KIND,
        fleets=len({agent.provider for agent in scenario.all_agents}),
        workers=len(scenario.all_agents),
        task_executions=executions,
        distinct_tasks=len(fresh),
        duplicate_task_executions=executions - len(fresh),
        investigations_performed=investigations,
        investigations_minimum=workload.minimum_investigations,
        duplicate_investigations=investigations - workload.minimum_investigations,
        discovery_derivations=derivations,
        discovery_derivations_minimum=workload.minimum_discovery_derivations,
        discoveries_rederived=derivations - workload.minimum_discovery_derivations,
        discovery_reuses=reuses,
        crash_steps_lost=lost,
        crash_steps_preserved=preserved,
        duplicate_executions_prevented=int(metrics.get("duplicate_claims_prevented", 0)),
        total_steps=_total_steps(workload, fresh, derivations, lost),
        minimum_steps=workload.minimum_steps,
        executions_by_task=dict(fresh),
        notes=(
            "MEASURED. Every count below is read from the run that just "
            "executed over the canonical HTTP API.",
            f"{int(metrics.get('tasks_completed', 0))}/"
            f"{int(metrics.get('tasks_total', 0))} tasks completed; "
            f"{int(metrics.get('crash_handoffs', 0))} crash handoff(s) recorded.",
            "Step costs come from the same shared fixture the baseline uses; "
            "only the number of times each unit of work happened is measured.",
        ),
        sources={
            "task_executions": (
                "run events: task.claimed with no prior task.checkpointed on that task"
            ),
            "duplicate_task_executions": "run events: fresh claims minus distinct tasks claimed",
            "investigations_performed": "run events: fresh claims on investigative tasks",
            "discovery_derivations": (
                "beat shared_discovery / check two_confirmed_memories_published"
            ),
            "discovery_reuses": (
                "unique task checkpoint/completion memory_ids whose author differs "
                "from the citing agent"
            ),
            "crash_steps_preserved": (
                "beat crash_handoff / check successor_resumes_with_checkpoint, "
                "run metrics.crash_handoffs, and the resuming task.claimed event"
            ),
            "duplicate_executions_prevented": "run metrics.duplicate_claims_prevented",
            "total_steps": "shared fixture step costs applied to the measured counts",
        },
    )


# -- comparison -------------------------------------------------------------


def compare(baseline: ArmMetrics, swarm: ArmMetrics) -> dict[str, Any]:
    steps_saved = baseline.total_steps - swarm.total_steps
    return {
        "duplicate_task_executions_avoided": (
            baseline.duplicate_task_executions - swarm.duplicate_task_executions
        ),
        "duplicate_investigations_avoided": (
            baseline.duplicate_investigations - swarm.duplicate_investigations
        ),
        "discovery_rederivations_avoided": (
            baseline.discoveries_rederived - swarm.discoveries_rederived
        ),
        "discovery_reuses_gained": swarm.discovery_reuses - baseline.discovery_reuses,
        "crash_steps_preserved": swarm.crash_steps_preserved - baseline.crash_steps_preserved,
        "task_executions_avoided": baseline.task_executions - swarm.task_executions,
        "steps_saved": steps_saved,
        "steps_saved_pct": (
            round(100.0 * steps_saved / baseline.total_steps, 1) if baseline.total_steps else 0.0
        ),
        "work_multiplier": (
            round(baseline.total_steps / swarm.total_steps, 2) if swarm.total_steps else 0.0
        ),
    }


@dataclass(slots=True)
class ABReport:
    workload: Workload
    baseline: ArmMetrics
    swarm: ArmMetrics
    deltas: dict[str, Any]
    evidence: dict[str, Any]
    generated_at: str = ""

    @property
    def ok(self) -> bool:
        """Arm B strictly dominates on duplicates avoided and work preserved."""

        return (
            bool(self.evidence.get("ok"))
            and self.deltas["duplicate_task_executions_avoided"] > 0
            and self.deltas["duplicate_investigations_avoided"] > 0
            and self.deltas["discovery_rederivations_avoided"] > 0
            and self.deltas["crash_steps_preserved"] > 0
            and self.deltas["steps_saved"] > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "generated_at": self.generated_at,
            "comparison": "same task set, same roster, with and without the shared swarm brain",
            "honesty": {
                "arm_a": (
                    "simulated: a deterministic model of an uncoordinated fleet, "
                    "not an API run and not an injected failure"
                ),
                "arm_b": (
                    "measured: read back from the live run's event stream, run "
                    "metrics, and beat data"
                ),
                "shared_fixture": (
                    "per-task step costs and the crash point are constants applied "
                    "identically to both arms"
                ),
            },
            "workload": self.workload.to_dict(),
            "arms": {
                "baseline": self.baseline.to_dict(),
                "swarm_brain": self.swarm.to_dict(),
            },
            "deltas": self.deltas,
            "swarm_run_evidence": self.evidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def table(self) -> str:
        return render_table(self)


_ROWS: tuple[tuple[str, str], ...] = (
    ("task executions", "task_executions"),
    ("duplicate task executions", "duplicate_task_executions"),
    ("investigations performed", "investigations_performed"),
    ("duplicate investigations", "duplicate_investigations"),
    ("discovery derivations", "discovery_derivations"),
    ("discoveries re-derived", "discoveries_rederived"),
    ("discoveries reused", "discovery_reuses"),
    ("work lost to the crash (steps)", "crash_steps_lost"),
    ("work preserved by handoff (steps)", "crash_steps_preserved"),
    ("total steps", "total_steps"),
)


def render_table(report: ABReport) -> str:
    baseline = report.baseline
    swarm = report.swarm
    deltas = report.deltas
    lines = [
        f"Swarm Brain A/B — {len(report.workload.tasks)} tasks, {swarm.workers} workers "
        f"across {swarm.fleets} vendor harnesses",
        "",
        f"  Arm A  without swarm brain  [{baseline.kind}]",
        "         a deterministic model of the same roster with no claim arbitration,",
        "         no cross-agent recall, and no durable checkpoint — not an API run",
        f"  Arm B  with swarm brain     [{swarm.kind}]",
        "         every count read back from the live run's events, metrics, and beats",
        "",
        f"  {'metric':<36}{'arm A':>8}{'arm B':>8}{'delta':>9}",
        f"  {'-' * 61}",
    ]
    for title, attribute in _ROWS:
        left = getattr(baseline, attribute)
        right = getattr(swarm, attribute)
        delta = right - left
        lines.append(f"  {title:<36}{left:>8}{right:>8}{delta:>+9}")
    lines.extend(
        [
            "",
            f"  minimum necessary: {report.workload.minimum_executions} executions, "
            f"{report.workload.minimum_investigations} investigations, "
            f"{report.workload.minimum_discovery_derivations} discovery derivations, "
            f"{report.workload.minimum_steps} steps",
            f"  duplicate claims the swarm brain refused outright: "
            f"{swarm.duplicate_executions_prevented} (run metrics)",
            f"  {deltas['steps_saved']} of {baseline.total_steps} steps saved "
            f"({deltas['steps_saved_pct']}%); the uncoordinated fleet does "
            f"{deltas['work_multiplier']}x the work for the same task set",
        ]
    )
    return "\n".join(lines)


async def run_comparison(
    *,
    client: httpx.AsyncClient,
    tokens: RunTokenCodec,
    store: CoordinationStore,
    scenario: DemoScenario | None = None,
    expire_leases: Callable[[datetime], Awaitable[None]],
    narrate: Callable[[str], None] | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: Callable[[], datetime] | None = None,
    workload: Workload = WORKLOAD,
) -> ABReport:
    """Run Arm B for real, simulate Arm A over the same fixture, and diff them."""

    scenario = scenario or build_scenario()
    say = narrate or (lambda _line: None)
    runner = DemoRunner(
        client=client,
        tokens=tokens,
        store=store,
        scenario=scenario,
        expire_leases=expire_leases,
        narrate=narrate,
        lease_seconds=lease_seconds,
        now=now,
    )
    say(f"{SWARM_LABEL}: running the real swarm over the canonical HTTP API")
    report = await runner.run()
    if not report.ok:
        failed = [
            f"{beat.key}:{check.name}"
            for beat in report.beats
            for check in beat.checks
            if not check.ok
        ]
        raise DemoAssertionError(
            f"arm B cannot be measured from a failed run; failing checks: {failed}"
        )

    token = tokens.issue(actor_context(scenario, scenario.lead), ttl=TOKEN_TTL)
    events = await _fetch_events(client, token, scenario.run_id)
    metrics = await _fetch_metrics(client, token, scenario.run_id)

    swarm = measure_swarm_arm(scenario, report, events, metrics, workload)
    say(f"{BASELINE_LABEL}: simulating the same task set with nothing shared")
    baseline = simulate_uncoordinated_baseline(scenario, workload)
    return ABReport(
        workload=workload,
        baseline=baseline,
        swarm=swarm,
        deltas=compare(baseline, swarm),
        evidence=report.to_dict(),
        generated_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "BASELINE_KIND",
    "SWARM_KIND",
    "WORKLOAD",
    "ABReport",
    "ArmMetrics",
    "Workload",
    "WorkloadTask",
    "build_workload",
    "compare",
    "measure_swarm_arm",
    "render_table",
    "run_comparison",
    "simulate_uncoordinated_baseline",
]
