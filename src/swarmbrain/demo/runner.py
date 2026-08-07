"""Drive the canonical HTTP API through the demo beats and record evidence.

The runner owns two planes and keeps them separate:

- the agent plane speaks only HTTP with per-agent bearer tokens, exactly like
  a production MCP bridge;
- the operator plane seeds the run through the ``CoordinationStore`` port,
  the same seam an operator CLI or test harness uses.

Every beat appends named checks to the report instead of asserting silently,
so a demo run produces an auditable artifact rather than a green boolean.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx

from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.tasks import Task, TaskDependency, TaskStatus
from swarmbrain.ports.coordination_store import CoordinationStore

from .scenario import DemoAgent, DemoScenario, actor_context

TOKEN_TTL = timedelta(minutes=30)
DEFAULT_LEASE_SECONDS = 120


class DemoAssertionError(AssertionError):
    """A beat could not continue; the report retains everything before it."""


@dataclass(slots=True)
class BeatCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(slots=True)
class BeatReport:
    key: str
    title: str
    checks: list[BeatCheck] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "data": self.data,
        }


@dataclass(slots=True)
class DemoReport:
    scenario: dict[str, Any]
    beats: list[BeatReport] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.beats) and all(beat.ok for beat in self.beats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scenario": self.scenario,
            "beats": [beat.to_dict() for beat in self.beats],
            "metrics": self.metrics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DemoRunner:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        tokens: RunTokenCodec,
        store: CoordinationStore,
        scenario: DemoScenario,
        expire_leases: Callable[[datetime], Awaitable[None]],
        narrate: Callable[[str], None] | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._tokens = tokens
        self._store = store
        self._scenario = scenario
        self._expire_leases = expire_leases
        self._narrate = narrate or (lambda _line: None)
        self._lease_seconds = lease_seconds
        # Seeds and observed-at stamps must share the backend's notion of now,
        # which a demo with a controllable clock deliberately freezes.
        self._now = now or (lambda: datetime.now(UTC))
        self._agent_tokens: dict[str, str] = {}
        # Race results: task key -> (winning agent, claim payload).
        self._owners: dict[str, tuple[DemoAgent, dict[str, Any]]] = {}
        self._discovery_memory_id: str | None = None
        self._correction_memory_id: str | None = None

    # -- operator plane -----------------------------------------------------

    def _actor(self, agent: DemoAgent) -> ActorContext:
        return actor_context(self._scenario, agent)

    def _token(self, agent: DemoAgent) -> str:
        token = self._agent_tokens.get(agent.agent_id)
        if token is None:
            token = self._tokens.issue(self._actor(agent), ttl=TOKEN_TTL)
            self._agent_tokens[agent.agent_id] = token
        return token

    async def _seed(self) -> None:
        scenario = self._scenario
        now = self._now()
        for spec in scenario.tasks:
            await self._store.add_task(
                Task(
                    **scenario.scope(),
                    task_id=spec.task_id,
                    title=spec.title,
                    description=spec.description,
                    tags=spec.tags,
                    priority=spec.priority,
                    status=TaskStatus.READY,
                    created_at=now,
                    updated_at=now,
                )
            )
            for dependency_key in spec.depends_on:
                await self._store.add_dependency(
                    TaskDependency(
                        task_id=spec.task_id,
                        depends_on_task_id=scenario.task(dependency_key).task_id,
                    )
                )

    # -- agent plane --------------------------------------------------------

    async def _request(
        self,
        agent: DemoAgent,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        idempotent: bool = False,
        idempotency_key: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._token(agent)}"}
        if idempotent:
            headers["Idempotency-Key"] = idempotency_key or str(uuid4())
        return await self._client.request(
            method,
            path,
            headers=headers,
            json=body if body is not None else ({} if method == "POST" else None),
            params=params,
        )

    async def _expect(
        self,
        agent: DemoAgent,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._request(agent, method, path, **kwargs)
        if response.status_code >= 400:
            raise DemoAssertionError(
                f"{agent.name} {method} {path} failed with {response.status_code}: "
                f"{response.text[:500]}"
            )
        return response.json()

    @staticmethod
    def _error_code(response: httpx.Response) -> str:
        try:
            return str(response.json()["error"]["code"])
        except Exception:
            return f"unparseable:{response.status_code}"

    # -- beats --------------------------------------------------------------

    async def run(self) -> DemoReport:
        scenario = self._scenario
        report = DemoReport(
            scenario={
                "run_id": scenario.run_id,
                "repository_id": scenario.repository_id,
                "agents": [
                    {
                        "name": agent.name,
                        "agent_id": agent.agent_id,
                        "harness": agent.harness,
                        "provider": agent.provider,
                        "model": agent.model,
                    }
                    for agent in scenario.all_agents
                ],
                "tasks": [
                    {
                        "key": task.key,
                        "task_id": task.task_id,
                        "title": task.title,
                        "depends_on": list(task.depends_on),
                    }
                    for task in scenario.tasks
                ],
            },
            started_at=datetime.now(UTC).isoformat(),
        )
        await self._seed()
        for beat in (
            self._beat_join,
            self._beat_claim_race,
            self._beat_shared_discovery,
            self._beat_supersession_and_poisoning,
            self._beat_complete_and_replay,
            self._beat_crash_handoff,
            self._beat_dag_unblock,
            self._beat_telemetry,
        ):
            beat_report = await beat()
            report.beats.append(beat_report)
            status = "ok" if beat_report.ok else "FAILED"
            self._narrate(f"[{status}] {beat_report.title}")
            for check in beat_report.checks:
                marker = "✓" if check.ok else "✗"
                self._narrate(f"    {marker} {check.name}: {check.detail}")
            if not beat_report.ok:
                break
        if report.beats and report.beats[-1].key == "telemetry":
            report.metrics = report.beats[-1].data.get("metrics", {})
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    async def _beat_join(self) -> BeatReport:
        beat = BeatReport(key="join", title="Fourteen agents join the run")
        scenario = self._scenario
        joined = []
        for agent in scenario.all_agents:
            payload = await self._expect(
                agent, "POST", f"/v1/runs/{scenario.run_id}/agents:join", body={}
            )
            joined.append(payload)
        providers = {item["provider"] for item in joined}
        beat.checks.append(
            BeatCheck(
                "all_agents_joined",
                len(joined) == len(scenario.all_agents),
                f"{len(joined)} agents joined across providers {sorted(providers)}",
            )
        )
        beat.data["joined"] = len(joined)
        return beat

    async def _beat_claim_race(self) -> BeatReport:
        beat = BeatReport(
            key="claim_race",
            title="Twelve workers race four ready tasks — exactly four leases",
        )
        scenario = self._scenario
        blocked_task = scenario.task("apply-fix")

        async def claim(agent: DemoAgent) -> tuple[DemoAgent, httpx.Response]:
            response = await self._request(
                agent,
                "POST",
                "/v1/tasks:claim",
                body={"lease_seconds": self._lease_seconds},
                idempotent=True,
                idempotency_key=f"race-{agent.agent_id}",
            )
            return agent, response

        results = await asyncio.gather(*(claim(agent) for agent in scenario.racers))
        wins: list[tuple[DemoAgent, dict[str, Any]]] = []
        losses: list[str] = []
        for agent, response in results:
            if response.status_code == 200:
                wins.append((agent, response.json()))
            else:
                losses.append(self._error_code(response))
        for agent, payload in wins:
            task_id = payload["task"]["task_id"]
            key = next(task.key for task in scenario.tasks if task.task_id == task_id)
            self._owners[key] = (agent, payload)

        won_task_ids = {payload["task"]["task_id"] for _, payload in wins}
        won_lease_ids = {payload["lease"]["lease_id"] for _, payload in wins}
        beat.checks.append(
            BeatCheck(
                "exactly_four_wins",
                len(wins) == 4 and len(won_task_ids) == 4 and len(won_lease_ids) == 4,
                f"{len(wins)} claims won 4 distinct tasks with 4 distinct leases",
            )
        )
        beat.checks.append(
            BeatCheck(
                "eight_losers_told_no_task",
                len(losses) == 8 and all(code == "no_task_available" for code in losses),
                f"{len(losses)} losing claims all returned no_task_available",
            )
        )
        beat.checks.append(
            BeatCheck(
                "blocked_task_not_claimed",
                blocked_task.task_id not in won_task_ids,
                "the dependency-blocked fix task was not claimable",
            )
        )
        beat.data["owners"] = {
            key: {"agent": agent.name, "provider": agent.provider}
            for key, (agent, _) in self._owners.items()
        }
        return beat

    async def _beat_shared_discovery(self) -> BeatReport:
        beat = BeatReport(
            key="shared_discovery",
            title="One agent's evidence-backed discovery is recalled by other vendors",
        )
        owner, claim = self._owners["duplicate-charge"]
        excerpt = (
            "webhook_deliveries has no unique constraint on provider_delivery_id; "
            "replayed delivery evt_9f2 charged twice"
        )
        source = await self._expect(
            owner,
            "POST",
            "/v1/evidence/sources",
            body={
                "kind": "log",
                "content_sha256": _sha256(excerpt),
                "observed_at": self._now().isoformat(),
                "uri": "repo://payments/worker.py",
                "occurrence_key": "payments-worker:webhook-replay-trace",
                "task_id": claim["task"]["task_id"],
            },
            idempotent=True,
        )
        evidence = await self._expect(
            owner,
            "POST",
            "/v1/evidence",
            body={
                "source_id": source["source_id"],
                "kind": "log",
                "locator": "payments/worker.py:214",
                "excerpt": excerpt,
                "content_sha256": source["content_sha256"],
            },
            idempotent=True,
        )
        published = await self._expect(
            owner,
            "POST",
            "/v1/memories",
            body={
                "kind": "invariant",
                "desired_state": "confirmed",
                "visibility": "repository",
                "title": "Webhook replay creates duplicate charges",
                "content": (
                    "Replayed payment webhooks create duplicate charges because "
                    "webhook_deliveries has no unique constraint on "
                    "provider_delivery_id; every delivery must be recorded with an "
                    "idempotency key before charging."
                ),
                "tags": ["payments", "webhook", "idempotency"],
                "confidence": 0.9,
                "evidence": [
                    {
                        "evidence_id": evidence["evidence_id"],
                        "source_id": evidence["source_id"],
                        "locator": evidence["locator"],
                        "excerpt": evidence["excerpt"],
                        "content_sha256": evidence["content_sha256"],
                    }
                ],
            },
            idempotent=True,
        )
        self._discovery_memory_id = published["memory"]["memory_id"]
        beat.checks.append(
            BeatCheck(
                "discovery_published",
                published["operation"] == "add" and published["memory"] is not None,
                f"{owner.name} published the confirmed invariant with evidence",
            )
        )

        readers = [
            agent
            for key, (agent, _) in self._owners.items()
            if key != "duplicate-charge" and agent.provider != owner.provider
        ][:2]
        recalled_by = []
        for reader in readers:
            bundle = await self._expect(
                reader,
                "POST",
                "/v1/memories:recall",
                body={"text": "duplicate charge webhook replay idempotency"},
            )
            hit_ids = [hit["memory"]["memory_id"] for hit in bundle["hits"]]
            if self._discovery_memory_id in hit_ids:
                top = next(
                    hit
                    for hit in bundle["hits"]
                    if hit["memory"]["memory_id"] == self._discovery_memory_id
                )
                recalled_by.append(
                    {
                        "agent": reader.name,
                        "provider": reader.provider,
                        "score": top["score"],
                        "evidence_count": len(top.get("evidence", [])),
                    }
                )
        beat.checks.append(
            BeatCheck(
                "cross_vendor_recall",
                len(recalled_by) == len(readers) and len(readers) >= 1,
                f"recalled with evidence by {[r['agent'] for r in recalled_by]}",
            )
        )
        beat.data["publisher"] = {"agent": owner.name, "provider": owner.provider}
        beat.data["recalled_by"] = recalled_by
        return beat

    async def _beat_supersession_and_poisoning(self) -> BeatReport:
        beat = BeatReport(
            key="supersession_poisoning",
            title="An evidence-backed correction supersedes; a poisoning attempt bounces",
        )
        scenario = self._scenario
        scout = scenario.scout
        lead = scenario.lead

        wrong = await self._expect(
            scout,
            "POST",
            "/v1/memories",
            body={
                "kind": "hypothesis",
                "content": (
                    "Duplicate charges are caused by customers double-clicking the "
                    "pay button; no server change is needed."
                ),
                "tags": ["payments"],
                "confidence": 0.4,
            },
            idempotent=True,
        )
        wrong_id = wrong["memory"]["memory_id"]

        excerpt = "replaying delivery evt_9f2 with the same body created charge ch_777 twice"
        source = await self._expect(
            lead,
            "POST",
            "/v1/evidence/sources",
            body={
                "kind": "test_result",
                "content_sha256": _sha256(excerpt),
                "observed_at": self._now().isoformat(),
                "uri": "ci://payments/replay-proof",
                "occurrence_key": "ci:webhook-replay-proof",
            },
            idempotent=True,
        )
        evidence = await self._expect(
            lead,
            "POST",
            "/v1/evidence",
            body={
                "source_id": source["source_id"],
                "kind": "test_result",
                "locator": "tests/test_webhook_replay.py::test_replay_charges_twice",
                "excerpt": excerpt,
                "content_sha256": source["content_sha256"],
            },
            idempotent=True,
        )
        correction = await self._expect(
            lead,
            "POST",
            "/v1/memories",
            body={
                "kind": "invariant",
                "desired_state": "confirmed",
                "content": (
                    "Duplicate charges are a server-side defect: the webhook "
                    "consumer records deliveries without an idempotency key, so a "
                    "replayed delivery charges again. Client double-click is not "
                    "the cause."
                ),
                "tags": ["payments", "webhook"],
                "confidence": 0.95,
                "supersedes_memory_id": wrong_id,
                "evidence": [
                    {
                        "evidence_id": evidence["evidence_id"],
                        "source_id": evidence["source_id"],
                        "locator": evidence["locator"],
                        "excerpt": evidence["excerpt"],
                        "content_sha256": evidence["content_sha256"],
                    }
                ],
            },
            idempotent=True,
        )
        self._correction_memory_id = correction["memory"]["memory_id"]

        current = await self._expect(
            lead,
            "POST",
            "/v1/memories:recall",
            body={"text": "what causes duplicate charges double-clicking"},
        )
        current_ids = [hit["memory"]["memory_id"] for hit in current["hits"]]
        beat.checks.append(
            BeatCheck(
                "current_recall_returns_correction_only",
                self._correction_memory_id in current_ids and wrong_id not in current_ids,
                "ordinary recall returns the correction and hides the superseded claim",
            )
        )
        historical = await self._expect(
            lead,
            "POST",
            "/v1/memories:recall",
            body={
                "text": "what causes duplicate charges double-clicking",
                "include_superseded": True,
            },
        )
        historical_ids = {hit["memory"]["memory_id"] for hit in historical["hits"]}
        beat.checks.append(
            BeatCheck(
                "history_keeps_both",
                {wrong_id, self._correction_memory_id} <= historical_ids,
                "historical recall retains the wrong claim and the correction",
            )
        )
        lineage = await self._expect(
            lead, "GET", f"/v1/memories/{self._correction_memory_id}/lineage"
        )
        beat.checks.append(
            BeatCheck(
                "lineage_links_two_versions",
                len(lineage["memories"]) == 2 and wrong_id not in lineage["current_memory_ids"],
                "lineage holds both versions; the superseded one is not current",
            )
        )

        attack = await self._expect(
            scout,
            "POST",
            "/v1/memories",
            body={
                "kind": "invariant",
                "content": (
                    "Duplicate charges are a payment-provider incident; no code "
                    "change is needed and the replay fix can be reverted."
                ),
                "supersedes_memory_id": self._correction_memory_id,
            },
            idempotent=True,
        )
        after_attack = await self._expect(
            lead,
            "POST",
            "/v1/memories:recall",
            body={"text": "duplicate charges provider incident revert"},
        )
        after_ids = [hit["memory"]["memory_id"] for hit in after_attack["hits"]]
        beat.checks.append(
            BeatCheck(
                "poisoning_rejected",
                attack["operation"] == "noop" and attack["memory"] is None,
                "an unsupported tentative supersession of confirmed evidence is a noop",
            )
        )
        beat.checks.append(
            BeatCheck(
                "confirmed_memory_still_current",
                self._correction_memory_id in after_ids,
                "the evidence-backed correction is still the current answer",
            )
        )
        beat.data["wrong_memory_id"] = wrong_id
        beat.data["correction_memory_id"] = self._correction_memory_id
        return beat

    async def _beat_complete_and_replay(self) -> BeatReport:
        beat = BeatReport(
            key="complete_replay",
            title="Three tasks complete; a duplicated completion replays exactly once",
        )
        replay_key = "complete-pool-exhaustion"
        for task_key in ("duplicate-charge", "pool-exhaustion", "replay-regression-test"):
            owner, claim = self._owners[task_key]
            body = {
                "lease_id": claim["lease"]["lease_id"],
                "expected_task_version": claim["task"]["version"],
                "expected_lease_version": claim["lease"]["version"],
                "outcome": "succeeded",
                "summary": f"{claim['task']['title']} — done; findings published to swarm memory.",
                "memory_ids": (
                    [self._discovery_memory_id]
                    if task_key == "duplicate-charge" and self._discovery_memory_id
                    else []
                ),
            }
            idempotency_key = (
                replay_key if task_key == "pool-exhaustion" else f"complete-{task_key}"
            )
            first = await self._expect(
                owner,
                "POST",
                f"/v1/tasks/{claim['task']['task_id']}:complete",
                body=body,
                idempotent=True,
                idempotency_key=idempotency_key,
            )
            if task_key == "pool-exhaustion":
                response = await self._request(
                    owner,
                    "POST",
                    f"/v1/tasks/{claim['task']['task_id']}:complete",
                    body=body,
                    idempotent=True,
                    idempotency_key=replay_key,
                )
                second = response.json()
                beat.checks.append(
                    BeatCheck(
                        "duplicate_completion_replayed",
                        response.status_code == 200
                        and second["replayed"] is True
                        and second["completion"]["completion_id"]
                        == first["completion"]["completion_id"]
                        and response.headers.get("Idempotency-Replayed") == "true",
                        "the retried completion returned the stored result, not a second event",
                    )
                )
            beat.checks.append(
                BeatCheck(
                    f"completed_{task_key}",
                    first["task"]["status"] == "completed",
                    f"{owner.name} completed '{claim['task']['title']}'",
                )
            )
        return beat

    async def _beat_crash_handoff(self) -> BeatReport:
        beat = BeatReport(
            key="crash_handoff",
            title="A worker dies mid-task; another vendor resumes from its checkpoint",
        )
        scenario = self._scenario
        owner, claim = self._owners["flaky-test"]
        task_id = claim["task"]["task_id"]
        checkpoint = await self._expect(
            owner,
            "POST",
            f"/v1/tasks/{task_id}/checkpoints",
            body={
                "lease_id": claim["lease"]["lease_id"],
                "expected_task_version": claim["task"]["version"],
                "expected_lease_version": claim["lease"]["version"],
                "summary": (
                    "Reproduced the flaky failure with seed 1337: the retry loop "
                    "reads the order row before the payment worker commits it."
                ),
                "discoveries": (
                    "failing seed is 1337",
                    "retry loop lacks a read-after-write barrier",
                ),
                "completed_work": ("reproduced failure locally 4/20 runs",),
                "remaining_work": (
                    "add the read barrier",
                    "re-run 200 seeds to confirm stability",
                ),
                "memory_ids": ([self._discovery_memory_id] if self._discovery_memory_id else []),
            },
            idempotent=True,
        )
        beat.checks.append(
            BeatCheck(
                "checkpoint_written",
                checkpoint["checkpoint"]["sequence"] >= 1,
                f"{owner.name} checkpointed then stopped renewing (simulated crash)",
            )
        )

        expires_at = datetime.fromisoformat(claim["lease"]["expires_at"])
        await self._expire_leases(expires_at)

        successor = next(
            agent
            for agent in scenario.racers
            if agent.provider != owner.provider
            and agent.agent_id != owner.agent_id
            and not any(a.agent_id == agent.agent_id for a, _ in self._owners.values())
        )
        resumed = await self._expect(
            successor,
            "POST",
            "/v1/tasks:claim",
            body={"task_id": task_id, "lease_seconds": self._lease_seconds},
            idempotent=True,
            idempotency_key=f"resume-{successor.agent_id}",
        )
        got_checkpoint = (
            resumed.get("checkpoint") is not None
            and resumed["checkpoint"]["checkpoint_id"] == checkpoint["checkpoint"]["checkpoint_id"]
        )
        beat.checks.append(
            BeatCheck(
                "successor_resumes_with_checkpoint",
                got_checkpoint,
                f"{successor.name} ({successor.provider}) received {owner.name}'s "
                f"({owner.provider}) checkpoint, discoveries, and remaining work",
            )
        )
        bundle = resumed.get("memory") or {}
        seeded_ids = [hit["memory"]["memory_id"] for hit in bundle.get("hits", [])]
        beat.checks.append(
            BeatCheck(
                "bootstrap_memory_included",
                self._discovery_memory_id in seeded_ids,
                "the claim response carried the checkpointed swarm memory as context",
            )
        )

        late = await self._request(
            owner,
            "POST",
            f"/v1/tasks/{task_id}:complete",
            body={
                "lease_id": claim["lease"]["lease_id"],
                "expected_task_version": checkpoint["task"]["version"],
                "expected_lease_version": checkpoint["lease"]["version"],
                "outcome": "succeeded",
                "summary": "late write from the dead worker",
            },
            idempotent=True,
        )
        beat.checks.append(
            BeatCheck(
                "dead_worker_fenced",
                late.status_code == 409,
                f"the crashed worker's stale lease was fenced ({self._error_code(late)})",
            )
        )

        finished = await self._expect(
            successor,
            "POST",
            f"/v1/tasks/{task_id}:complete",
            body={
                "lease_id": resumed["lease"]["lease_id"],
                "expected_task_version": resumed["task"]["version"],
                "expected_lease_version": resumed["lease"]["version"],
                "outcome": "succeeded",
                "summary": (
                    "Added the read-after-write barrier from the checkpoint's "
                    "remaining work; 200 seeded runs green."
                ),
            },
            idempotent=True,
        )
        beat.checks.append(
            BeatCheck(
                "successor_completed",
                finished["task"]["status"] == "completed",
                f"{successor.name} finished the task the crashed worker started",
            )
        )
        beat.data["crashed"] = {"agent": owner.name, "provider": owner.provider}
        beat.data["successor"] = {"agent": successor.name, "provider": successor.provider}
        return beat

    async def _beat_dag_unblock(self) -> BeatReport:
        beat = BeatReport(
            key="dag_unblock",
            title="Completing the root cause unblocks the dependent fix task",
        )
        scenario = self._scenario
        fix_task = scenario.task("apply-fix")
        agent = next(
            candidate
            for candidate in scenario.racers
            if not any(a.agent_id == candidate.agent_id for a, _ in self._owners.values())
        )
        claim = await self._expect(
            agent,
            "POST",
            "/v1/tasks:claim",
            body={"lease_seconds": self._lease_seconds},
            idempotent=True,
            idempotency_key=f"unblocked-{agent.agent_id}",
        )
        beat.checks.append(
            BeatCheck(
                "fix_task_now_claimable",
                claim["task"]["task_id"] == fix_task.task_id,
                "the dependency-blocked task became the next claim after its dependency completed",
            )
        )
        finished = await self._expect(
            agent,
            "POST",
            f"/v1/tasks/{fix_task.task_id}:complete",
            body={
                "lease_id": claim["lease"]["lease_id"],
                "expected_task_version": claim["task"]["version"],
                "expected_lease_version": claim["lease"]["version"],
                "outcome": "succeeded",
                "summary": "Recorded provider_delivery_id uniquely; fix shipped behind flag.",
                "memory_ids": ([self._correction_memory_id] if self._correction_memory_id else []),
            },
            idempotent=True,
        )
        beat.checks.append(
            BeatCheck(
                "all_tasks_completed",
                finished["task"]["status"] == "completed",
                f"{agent.name} completed the final task in the DAG",
            )
        )
        return beat

    async def _beat_telemetry(self) -> BeatReport:
        beat = BeatReport(
            key="telemetry",
            title="The run's events and metrics tell the whole story from the database",
        )
        scenario = self._scenario
        lead = scenario.lead
        events: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(20):
            params: dict[str, Any] = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            page = await self._expect(
                lead, "GET", f"/v1/runs/{scenario.run_id}/events", params=params
            )
            events.extend(page["events"])
            cursor = page.get("next_cursor")
            if not cursor:
                break
        metrics = await self._expect(lead, "GET", f"/v1/runs/{scenario.run_id}/metrics")

        beat.checks.append(
            BeatCheck(
                "five_tasks_completed",
                metrics["tasks_total"] == 5 and metrics["tasks_completed"] == 5,
                f"{metrics['tasks_completed']}/{metrics['tasks_total']} tasks completed",
            )
        )
        beat.checks.append(
            BeatCheck(
                "no_lease_left_behind",
                metrics["active_leases"] == 0,
                "every lease was completed, fenced, or expired",
            )
        )
        beat.checks.append(
            BeatCheck(
                "crash_handoff_counted",
                metrics["crash_handoffs"] >= 1,
                f"{metrics['crash_handoffs']} crash handoff recorded by the store",
            )
        )
        beat.checks.append(
            BeatCheck(
                "idempotent_replay_counted",
                metrics["duplicate_mutations_replayed"] >= 1,
                f"{metrics['duplicate_mutations_replayed']} duplicate mutations replayed "
                "instead of re-executed",
            )
        )
        beat.checks.append(
            BeatCheck(
                "memories_published_and_reused",
                metrics["memories_published"] >= 2 and metrics["memories_reused"] >= 1,
                f"{metrics['memories_published']} memories published "
                "(supersessions counted separately), "
                f"{metrics['memories_reused']} reuses recorded durably",
            )
        )
        beat.checks.append(
            BeatCheck(
                "event_timeline_present",
                len(events) >= 10,
                f"{len(events)} durable swarm events available to the dashboard",
            )
        )
        beat.data["metrics"] = metrics
        beat.data["event_count"] = len(events)
        beat.data["event_types"] = sorted({event["event_type"] for event in events})
        return beat


__all__ = [
    "BeatCheck",
    "BeatReport",
    "DemoAssertionError",
    "DemoReport",
    "DemoRunner",
]
