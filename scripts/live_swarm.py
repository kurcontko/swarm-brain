#!/usr/bin/env python3
"""Operator launcher for a live swarm of real LLM coding agents.

`seed` provisions one fresh run against a durable Swarm Brain API: fresh
scope UUIDs, a small set of self-contained coding tasks, prior swarm memory
worth recalling, one short-lived token per harness, and a ready-to-launch
bundle (env file, MCP server config, prompt) per harness. Nothing about the
agent side is simulated: each harness is a real LLM CLI that speaks the
production `swarmbrain-mcp` stdio bridge with its own bearer token.

`report` reads the bundle back, queries the run's durable event stream and
metrics with a read-only viewer token, and writes a redacted evidence JSON.

Seeding uses the operator plane (the CoordinationStore port, the same seam
the demo CLI uses) because task creation is deliberately not an agent-plane
route. Everything the agents do afterwards is agent-plane HTTP behind the
bridge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow `python scripts/live_swarm.py`
    sys.path.insert(0, str(REPO_ROOT / "src"))

import httpx  # noqa: E402

from swarmbrain.adapters.auth import RunTokenCodec  # noqa: E402
from swarmbrain.config import ApiSettings, BackendKind, BridgeSettings  # noqa: E402
from swarmbrain.demo.scenario import (  # noqa: E402
    WORKER_CAPABILITIES,
    DemoAgent,
    DemoScenario,
    DemoTask,
    actor_context,
)
from swarmbrain.domain.agents import Capability  # noqa: E402
from swarmbrain.domain.tasks import Task, TaskStatus  # noqa: E402

VIEWER_CAPABILITIES = frozenset(
    {
        Capability.RUN_JOIN.value,
        Capability.EVENTS_READ.value,
        Capability.METRICS_READ.value,
        Capability.MEMORY_RECALL.value,
    }
)

SEED_CAPABILITIES = frozenset(
    {
        Capability.RUN_JOIN.value,
        Capability.MEMORY_PUBLISH.value,
        Capability.MEMORY_RECALL.value,
        # Prior findings are seeded as confirmed so a live agent's recall
        # returns settled memory rather than someone else's guess.
        Capability.MEMORY_CONFIRM.value,
    }
)


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    """One real coding CLI and the labels its token will carry."""

    key: str
    harness: str
    provider: str
    model: str
    launch: str


HARNESSES: dict[str, HarnessSpec] = {
    "claude": HarnessSpec(
        key="claude",
        harness="claude-code",
        provider="anthropic",
        # The CLI alias is what the launch command passes; the exact model id
        # the session resolved is recorded in the evidence artifact from the
        # harness's own result JSON.
        model="sonnet",
        launch="claude",
    ),
    "codex": HarnessSpec(
        key="codex",
        harness="codex-cli",
        provider="openai",
        model="gpt-5-codex",
        launch="codex",
    ),
    "gemini": HarnessSpec(
        key="gemini",
        harness="gemini-cli",
        provider="google",
        model="gemini-2.5-pro",
        launch="gemini",
    ),
    "opencode": HarnessSpec(
        key="opencode",
        harness="opencode",
        provider="local",
        model="qwen3-coder",
        launch="opencode",
    ),
}


# Every task is answerable from its own description by reasoning alone, so a
# real LLM can genuinely finish it with no repository checkout, no shell, and
# no file access — only the seven MCP tools.
LIVE_TASKS: tuple[dict[str, Any], ...] = (
    {
        "key": "duplicate-charge-invariant",
        "title": "State the invariant that prevents double charges on webhook replay",
        "priority": 30,
        "tags": ("payments", "webhook", "invariant"),
        "description": (
            "A payments worker consumes provider webhook deliveries. The provider "
            "guarantees at-least-once delivery: after an acknowledgement timeout it "
            "redelivers the same event with the same provider event id. Today the "
            "worker creates one charge per delivery.\n\n"
            "Deliverable, from first principles only (no repository, no shell, no "
            "files): state in one sentence the invariant the worker must hold so that "
            "N identical deliveries of the same provider event produce exactly one "
            "charge, name the durable state that invariant requires, and say why a "
            "check-then-insert without that durable state is not sufficient. Publish "
            "the invariant as a memory with your reasoning attached as evidence, "
            "checkpoint, and complete the task."
        ),
    },
    {
        "key": "lease-handoff-invariant",
        "title": "State the invariant that makes an expired task lease safe to hand off",
        "priority": 20,
        "tags": ("coordination", "leases", "invariant"),
        "description": (
            "Tasks are owned through time-bounded leases. A worker can stall or be "
            "killed without releasing its lease, so another worker must be able to "
            "take the task over once the lease expires. The stalled worker may wake up "
            "afterwards and try to write.\n\n"
            "Deliverable, from first principles only (no repository, no shell, no "
            "files): state in one sentence the invariant that makes takeover safe even "
            "though the old owner is still alive, name the token that must accompany "
            "every task mutation, and say why an expiry timestamp alone (without that "
            "token) does not prevent a stale write. Publish the invariant as a memory "
            "with your reasoning attached as evidence, checkpoint, and complete the task."
        ),
    },
    {
        "key": "retry-amplification-invariant",
        "title": "State the invariant that keeps a client retry loop from amplifying an outage",
        "priority": 10,
        "tags": ("reliability", "retries", "invariant"),
        "description": (
            "During a partial outage, every client of a service retries failed calls. "
            "Fixed-interval retries from many clients converge into synchronized bursts "
            "that keep the service down after it would otherwise recover.\n\n"
            "Deliverable, from first principles only (no repository, no shell, no "
            "files): state in one sentence the invariant a retry policy must satisfy so "
            "that aggregate retry load stays bounded as failures persist, name the two "
            "properties the delay schedule needs and the budget that bounds total "
            "attempts, and say which class of errors must never be retried at all. "
            "Publish the invariant as a memory with your reasoning attached as evidence, "
            "checkpoint, and complete the task."
        ),
    },
)


# Prior swarm memory published by an earlier scout, so `recall_memory` returns
# something the agent can genuinely use instead of an empty page.
SEED_MEMORIES: tuple[dict[str, Any], ...] = (
    {
        "task_key": "duplicate-charge-invariant",
        "title": "The payment provider redelivers webhooks after an ack timeout",
        "content": (
            "The payment provider retries webhook delivery when it does not receive a "
            "2xx acknowledgement within five seconds, reusing the same provider event "
            "id. Delivery is at-least-once, never exactly-once, so every consumer must "
            "treat a repeated provider event id as the same business event."
        ),
        "kind": "observation",
        "tags": ["payments", "webhook", "delivery"],
        "excerpt": (
            "provider docs: 'we retry delivery until acknowledged; the event id is "
            "stable across retries'"
        ),
    },
    {
        "task_key": "lease-handoff-invariant",
        "title": "Task leases in this system carry a monotonic version",
        "content": (
            "Every task lease in this run carries a lease version that increases on "
            "each renewal, and every task mutation is submitted with the expected task "
            "and lease version. The server rejects a mutation whose expected versions "
            "no longer match the durable row."
        ),
        "kind": "observation",
        "tags": ["coordination", "leases", "fencing"],
        "excerpt": "lease renewal response: {\"lease\": {\"version\": 4}}",
    },
    {
        "task_key": "retry-amplification-invariant",
        "title": "Synchronized retries were the amplifier in the last incident",
        "content": (
            "In the previous incident review, the service recovered only after clients "
            "stopped retrying on a fixed one-second interval. The postmortem attributed "
            "the extended outage to synchronized retry bursts rather than to the "
            "original fault."
        ),
        "kind": "observation",
        "tags": ["reliability", "retries", "incident"],
        "excerpt": "postmortem: 'retry storm sustained the outage for 41 minutes past root-cause fix'",
    },
)

PROMPT_TEMPLATE = """\
You are a real coding agent joining a live Swarm Brain run.

Your only interface to the swarm is the `swarmbrain` MCP server. Do not read or
write files, do not run shell commands, and do not use any other tool. Your
identity, run, and permissions come from the bridge's token; never put ids from
this prompt into tool arguments except the ones you receive from tool results.

Every mutating tool takes an `idempotency_key`. Use exactly the keys given
below, one per step, so a retry replays instead of duplicating.

Do these six steps in order and do not stop early:

1. `claim_task` with `idempotency_key` "{prefix}-claim" and `lease_seconds`
   {lease_seconds}. Do not pass a task_id: take whatever the swarm gives you.
   Read the returned task's `task_id`, `title`, and `description`.

2. `recall_memory` with `text` set to a short phrase drawn from the task title
   and the claimed `task_id` as `task_id`. State plainly in your final answer
   whether prior swarm memory existed and whether it changed your answer.

3. Answer the task from its description and whatever you recalled. It is a
   reasoning task; it needs no repository access. Be concrete and specific.

4. `publish_memory` with:
   - `idempotency_key`: "{prefix}-memory"
   - `task_id`: the claimed task_id
   - `kind`: "invariant"
   - `desired_state`: "confirmed"
   - `visibility`: "run"
   - `title`: a short name for the invariant
   - `content`: the invariant plus the durable state or token it requires, as
     one string
   - `confidence`: 0.8
   - `evidence`: [{{"kind": "message", "locator": "reasoning:1", "excerpt":
     "<two or three sentences of your actual reasoning, quoted>"}}]
   Keep the returned `memory_id`.

5. `checkpoint_task` with `idempotency_key` "{prefix}-checkpoint", the claimed
   `task_id`, a one-line `summary`, `completed_work` listing what you did, and
   `memory_ids` containing the memory_id from step 4.

6. `complete_task` with `idempotency_key` "{prefix}-complete", the claimed
   `task_id`, `outcome` "succeeded", a one-line `summary`, and `memory_ids`
   containing the memory_id from step 4.

Then print a short plain-text report: the task_id you claimed, the memory_id
you published, the invariant in one sentence, and what recall returned. If any
tool call fails, print the exact error text instead of retrying blindly.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision and evidence a live run of real LLM agents on Swarm Brain",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="seed a fresh run and write per-harness launch bundles")
    seed.add_argument(
        "--api-url",
        default=os.getenv("SWARMBRAIN_API_URL", "http://127.0.0.1:8080"),
        help="base URL of the running Swarm Brain API (default: %(default)s)",
    )
    seed.add_argument(
        "--out",
        default=None,
        help="directory for env files, MCP configs, and prompts (default: a fresh temp dir)",
    )
    seed.add_argument(
        "--harness",
        action="append",
        default=[],
        choices=sorted(HARNESSES),
        help="harness to provision; repeatable (default: claude)",
    )
    seed.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="HARNESS=MODEL",
        help="override the model label recorded in a harness token; repeatable",
    )
    seed.add_argument("--ttl-seconds", type=int, default=3600, help="token lifetime")
    seed.add_argument(
        "--unattended",
        action="store_true",
        help=(
            "print the unattended launch variants for codex/gemini, which "
            "DISABLE those harnesses' sandbox/approval gates in the bundle "
            "workspace; the default prints interactive-approval commands"
        ),
    )
    seed.add_argument("--lease-seconds", type=int, default=300, help="claim lease duration")
    seed.add_argument(
        "--no-seed-memory",
        action="store_true",
        help="skip publishing prior swarm memory (recall will return an empty page)",
    )

    report = sub.add_parser("report", help="collect durable evidence for a seeded run")
    report.add_argument("--bundle", required=True, help="the --out directory produced by seed")
    report.add_argument(
        "--evidence-dir",
        default=str(REPO_ROOT / "evidence"),
        help="directory for the redacted evidence JSON (default: %(default)s)",
    )
    report.add_argument(
        "--harness-result",
        action="append",
        default=[],
        metavar="HARNESS=FILE",
        help="claude --output-format json result file to read harness identity from",
    )
    report.add_argument(
        "--transcript",
        action="append",
        default=[],
        metavar="HARNESS=FILE",
        help="harness transcript whose tail is embedded in the artifact",
    )
    report.add_argument(
        "--gap",
        action="append",
        default=[],
        metavar="HARNESS=TEXT",
        help="record an honest note about why a harness did not finish the loop",
    )
    report.add_argument(
        "--label",
        default="live-swarm",
        help="artifact filename suffix (default: %(default)s)",
    )
    return parser


def _pairs(values: list[str], flag: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, rest = value.partition("=")
        if not separator or not key or not rest:
            raise SystemExit(f"{flag} expects HARNESS=VALUE, got: {value}")
        parsed[key.strip()] = rest.strip()
    return parsed


def build_live_scenario(
    harness_keys: list[str],
    *,
    models: dict[str, str],
) -> tuple[DemoScenario, dict[str, HarnessSpec]]:
    """A fresh run: one worker per harness, plus a viewer and a memory scout."""

    workers: list[DemoAgent] = []
    by_agent: dict[str, HarnessSpec] = {}
    for key in harness_keys:
        spec = HARNESSES[key]
        agent = DemoAgent(
            name=f"live-{spec.key}",
            harness=spec.harness,
            provider=spec.provider,
            model=models.get(key, spec.model),
            capabilities=WORKER_CAPABILITIES,
        )
        workers.append(agent)
        by_agent[agent.agent_id] = spec
    viewer = DemoAgent(
        name="viewer",
        harness="operator-console",
        provider="swarmbrain",
        model="console",
        capabilities=VIEWER_CAPABILITIES,
    )
    scout = DemoAgent(
        name="prior-scout",
        harness="operator-seed",
        provider="swarmbrain",
        model="seed",
        capabilities=SEED_CAPABILITIES,
    )
    tasks = tuple(
        DemoTask(
            key=str(item["key"]),
            title=str(item["title"]),
            description=str(item["description"]),
            tags=tuple(item["tags"]),
            priority=int(item["priority"]),
        )
        for item in LIVE_TASKS
    )
    scenario = DemoScenario(
        tenant_id=str(uuid4()),
        project_id=str(uuid4()),
        repository_id=str(uuid4()),
        swarm_id=str(uuid4()),
        run_id=str(uuid4()),
        lead=viewer,
        scout=scout,
        workers=tuple(workers),
        tasks=tasks,
    )
    return scenario, by_agent


async def seed_tasks(scenario: DemoScenario) -> None:
    """Create the run's tasks through the operator plane (CoordinationStore)."""

    from swarmbrain.application.runtime import build_runtime

    settings = ApiSettings.from_env()
    if settings.backend is not BackendKind.COCKROACH:
        raise SystemExit(
            "live_swarm seeds a durable run; set SWARMBRAIN_BACKEND=cockroach and "
            "SWARMBRAIN_DATABASE_URL to the same database the API is serving"
        )
    runtime = build_runtime(settings)
    if runtime.coordination_store is None:
        raise SystemExit("the composed runtime does not expose a coordination store")
    await runtime.start()
    try:
        now = datetime.now(UTC)
        for task in scenario.tasks:
            await runtime.coordination_store.add_task(
                Task(
                    **scenario.scope(),
                    task_id=task.task_id,
                    title=task.title,
                    description=task.description,
                    status=TaskStatus.READY,
                    priority=task.priority,
                    tags=task.tags,
                    created_at=now,
                    updated_at=now,
                )
            )
    finally:
        await runtime.close()


async def publish_prior_memory(
    scenario: DemoScenario,
    *,
    api_url: str,
    token: str,
) -> list[str]:
    """Publish the prior findings the live agents will recall, as the scout.

    This goes through the same bridge client the MCP server uses, so the
    seeded rows are shaped exactly like agent-published memory (inline
    evidence is registered as a source plus an evidence row first).
    """

    from swarmbrain.transports.mcp.client import SwarmBrainHttpClient

    client = SwarmBrainHttpClient(
        BridgeSettings(
            api_url=api_url,
            agent_token=token,
            expected_run_id=scenario.run_id,
            expected_agent_id=scenario.scout.agent_id,
        )
    )
    published: list[str] = []
    try:
        await client.join()
        for index, item in enumerate(SEED_MEMORIES):
            task = scenario.task(str(item["task_key"]))
            result = await client.publish_memory(
                content=str(item["content"]),
                kind=str(item["kind"]),
                desired_state="confirmed",
                visibility="run",
                task_id=task.task_id,
                title=str(item["title"]),
                tags=list(item["tags"]),
                confidence=0.9,
                evidence=[
                    {
                        "kind": "document",
                        "locator": "seed:1",
                        "excerpt": str(item["excerpt"]),
                    }
                ],
                metadata={"seeded_by": "scripts/live_swarm.py"},
                idempotency_key=f"live-seed-memory-{index}-{scenario.run_id}",
            )
            published.append(str(result["memory"]["memory_id"]))
    finally:
        await client.close()
    return published


def _uv() -> str:
    return shutil.which("uv") or "uv"


def _bridge_command() -> tuple[str, list[str]]:
    return _uv(), ["run", "--directory", str(REPO_ROOT), "--extra", "mcp", "swarmbrain-mcp"]


def _bridge_env(*, api_url: str, token: str, run_id: str, agent_id: str) -> dict[str, str]:
    return {
        "SWARMBRAIN_API_URL": api_url,
        "SWARMBRAIN_AGENT_TOKEN": token,
        "SWARMBRAIN_RUN_ID": run_id,
        "SWARMBRAIN_AGENT_ID": agent_id,
        "SWARMBRAIN_REQUEST_TIMEOUT_SECONDS": "30",
    }


def _write(path: Path, text: str) -> Path:
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def write_bundle(
    out: Path,
    *,
    scenario: DemoScenario,
    specs: dict[str, HarnessSpec],
    tokens: dict[str, str],
    api_url: str,
    lease_seconds: int,
    seeded_memory_ids: list[str],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    command, args = _bridge_command()
    manifest_agents: list[dict[str, Any]] = []

    for agent in scenario.workers:
        spec = specs[agent.agent_id]
        environment = _bridge_env(
            api_url=api_url,
            token=tokens[agent.agent_id],
            run_id=scenario.run_id,
            agent_id=agent.agent_id,
        )
        _write(
            out / f"{spec.key}.env",
            "\n".join(f"export {name}={value!r}" for name, value in environment.items()),
        )
        # Claude Code and Gemini CLI both read {"mcpServers": {...}}.
        _write(
            out / f"{spec.key}.mcp.json",
            json.dumps(
                {
                    "mcpServers": {
                        "swarmbrain": {
                            "command": command,
                            "args": args,
                            "env": environment,
                        }
                    }
                },
                indent=2,
            ),
        )
        if spec.key == "codex":
            # Codex CLI reads TOML from $CODEX_HOME. A private home leaves the
            # operator's ~/.codex/config.toml untouched; the credential file is
            # linked, never copied, so no secret is duplicated into the bundle.
            toml_env = ", ".join(f'{name} = "{value}"' for name, value in environment.items())
            toml_args = ", ".join(f'"{value}"' for value in args)
            home = out / f"{spec.key}-home"
            home.mkdir(parents=True, exist_ok=True)
            home.chmod(0o700)
            workspace = out / f"{spec.key}-workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            _write(
                home / "config.toml",
                "\n".join(
                    (
                        # Top-level keys must precede the first table: TOML
                        # would otherwise scope them under the mcp_servers one.
                        'approval_policy = "never"',
                        'sandbox_mode = "read-only"',
                        "",
                        f'[projects."{workspace}"]',
                        'trust_level = "trusted"',
                        "",
                        "[mcp_servers.swarmbrain]",
                        f'command = "{command}"',
                        f"args = [{toml_args}]",
                        f"env = {{ {toml_env} }}",
                        "startup_timeout_sec = 60",
                    )
                ),
            )
            credential = Path.home() / ".codex" / "auth.json"
            link = home / "auth.json"
            if credential.exists() and not link.exists():
                link.symlink_to(credential)
        elif spec.key == "gemini":
            # Gemini CLI reads .gemini/settings.json from the working directory.
            settings = out / f"{spec.key}-workspace" / ".gemini"
            settings.mkdir(parents=True, exist_ok=True)
            _write(
                settings / "settings.json",
                json.dumps(
                    {
                        "mcpServers": {
                            "swarmbrain": {
                                "command": command,
                                "args": args,
                                "env": environment,
                                "timeout": 60000,
                            }
                        }
                    },
                    indent=2,
                ),
            )
        elif spec.key == "opencode":
            # opencode reads opencode.json from the working directory.
            workspace = out / f"{spec.key}-workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            _write(
                workspace / "opencode.json",
                json.dumps(
                    {
                        "$schema": "https://opencode.ai/config.json",
                        "mcp": {
                            "swarmbrain": {
                                "type": "local",
                                "command": [command, *args],
                                "enabled": True,
                                "environment": environment,
                            }
                        },
                    },
                    indent=2,
                ),
            )
        _write(
            out / f"{spec.key}.prompt.md",
            PROMPT_TEMPLATE.format(
                prefix=f"{spec.key}-{agent.agent_id[:8]}",
                lease_seconds=lease_seconds,
            ),
        )
        manifest_agents.append(
            {
                "harness_key": spec.key,
                "agent_id": agent.agent_id,
                "name": agent.name,
                "harness": agent.harness,
                "provider": agent.provider,
                "model": agent.model,
                "capabilities": sorted(agent.capabilities),
            }
        )

    viewer_env = {
        "SWARMBRAIN_API_URL": api_url,
        "SWARMBRAIN_RUN_ID": scenario.run_id,
        "SWARMBRAIN_VIEWER_TOKEN": tokens[scenario.lead.agent_id],
    }
    _write(
        out / "viewer.env",
        "\n".join(f"export {name}={value!r}" for name, value in viewer_env.items()),
    )

    manifest = {
        "api_url": api_url,
        "created_at": datetime.now(UTC).isoformat(),
        "lease_seconds": lease_seconds,
        "scope": scenario.scope(),
        "viewer_agent_id": scenario.lead.agent_id,
        "scout_agent_id": scenario.scout.agent_id,
        "seeded_memory_ids": seeded_memory_ids,
        "agents": manifest_agents,
        "tasks": [
            {
                "key": task.key,
                "task_id": task.task_id,
                "title": task.title,
                "priority": task.priority,
                "tags": list(task.tags),
            }
            for task in scenario.tasks
        ],
    }
    # run.json carries no credentials; tokens live only in the *.env and
    # *.mcp.json files, which are written 0600 outside the repository.
    _write(out / "run.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def launch_lines(out: Path, spec: HarnessSpec, *, unattended: bool = False) -> list[str]:
    mcp = out / f"{spec.key}.mcp.json"
    prompt = out / f"{spec.key}.prompt.md"
    transcript = out / f"{spec.key}.transcript.jsonl"
    if spec.key == "claude":
        return [
            f"claude -p \"$(cat {prompt})\" \\",
            f"  --mcp-config {mcp} --strict-mcp-config \\",
            '  --allowedTools "mcp__swarmbrain" \\',
            f"  --model {spec.model} --max-budget-usd 2 \\",
            f"  --output-format stream-json --verbose | tee {transcript}",
        ]
    text_transcript = out / f"{spec.key}.transcript.txt"
    if spec.key == "codex":
        # This codex build gates every MCP tool call behind an approval that a
        # fully non-interactive `exec` session answers with "cancelled", even
        # with approval_policy = "never". The reliable safe path is an
        # interactive session where the operator approves the first tool call;
        # the sandbox/approval bypass is printed only under --unattended,
        # which the operator must pass deliberately.
        if not unattended:
            return [
                f"CODEX_HOME={out / f'{spec.key}-home'} \\",
                f"  codex -C {out / f'{spec.key}-workspace'} \"$(cat {prompt})\"",
                "# interactive: approve the swarmbrain MCP tool calls when prompted.",
                "# For an unattended take, re-run seed with --unattended and read the",
                "# warning it prints.",
            ]
        return [
            "# WARNING: this DISABLES codex's sandbox and approval gate. Run it",
            "# only in this disposable workspace, and only if you accept that the",
            "# session executes without any approval prompt. Untested here.",
            f"CODEX_HOME={out / f'{spec.key}-home'} \\",
            f"  codex exec --skip-git-repo-check -C {out / f'{spec.key}-workspace'} \\",
            "  --dangerously-bypass-approvals-and-sandbox \\",
            # codex blocks on stdin when it is neither a TTY nor closed.
            f"  \"$(cat {prompt})\" < /dev/null | tee {text_transcript}",
        ]
    if spec.key == "gemini":
        # --yolo auto-approves every tool, including gemini's built-ins; keep
        # it behind the same explicit opt-in and default to interactive.
        if not unattended:
            return [
                f"cd {out / f'{spec.key}-workspace'} && \\",
                "  gemini --allowed-mcp-server-names swarmbrain \\",
                f"  \"$(cat {prompt})\" | tee {text_transcript}",
                "# interactive: approve the swarmbrain MCP tool calls when prompted.",
            ]
        return [
            "# WARNING: --yolo auto-approves every gemini tool call in this",
            "# workspace. Deliberate opt-in only.",
            f"cd {out / f'{spec.key}-workspace'} && \\",
            "  gemini --yolo --allowed-mcp-server-names swarmbrain \\",
            f"  \"$(cat {prompt})\" | tee {text_transcript}",
        ]
    return [
        f"cd {out / f'{spec.key}-workspace'} && \\",
        f"  opencode run \"$(cat {prompt})\" | tee {text_transcript}",
    ]


def print_launch(
    out: Path,
    manifest: dict[str, Any],
    specs: dict[str, HarnessSpec],
    *,
    unattended: bool = False,
) -> None:
    api_url = str(manifest["api_url"])
    run_id = str(manifest["scope"]["run_id"])
    print()
    print(f"run id     {run_id}")
    print(f"api        {api_url}")
    print(f"bundle     {out}   (tokens live here, 0600, outside the repository)")
    print(f"console    {api_url}/console   (paste the run id and $SWARMBRAIN_VIEWER_TOKEN)")
    print()
    print("seeded tasks:")
    for task in manifest["tasks"]:
        print(f"  [{task['priority']:>3}] {task['title']}")
    print()
    for agent in manifest["agents"]:
        spec = specs[str(agent["agent_id"])]
        print(f"--- {agent['harness']} / {agent['provider']} / {agent['model']} ---")
        for line in launch_lines(out, spec, unattended=unattended):
            print(f"  {line}")
        print()
    print("watch the durable event stream:")
    print(f"  source {out / 'viewer.env'}")
    print(
        '  curl -s -H "Authorization: Bearer $SWARMBRAIN_VIEWER_TOKEN" '
        f'"{api_url}/v1/runs/{run_id}/events?limit=200" | jq -r '
        "'.events[] | \"\\(.occurred_at) \\(.event_type)\"'"
    )
    print()
    print("then collect redacted evidence:")
    print(f"  uv run --extra crdb python scripts/live_swarm.py report --bundle {out}")
    print()


async def _seed(args: argparse.Namespace) -> None:
    secret = os.getenv("SWARMBRAIN_TOKEN_SECRET")
    if not secret:
        raise SystemExit("SWARMBRAIN_TOKEN_SECRET is required and must match the running API")
    harness_keys = args.harness or ["claude"]
    scenario, specs = build_live_scenario(harness_keys, models=_pairs(args.model, "--model"))
    api_url = str(args.api_url).rstrip("/")

    async with httpx.AsyncClient(timeout=5.0) as probe:
        try:
            ready = await probe.get(f"{api_url}/readyz")
        except httpx.TransportError as exc:
            raise SystemExit(f"no Swarm Brain API at {api_url}: {exc}") from exc
    if ready.status_code != 200:
        raise SystemExit(f"Swarm Brain API at {api_url} is not ready: {ready.text}")

    await seed_tasks(scenario)

    codec = RunTokenCodec(secret)
    ttl = timedelta(seconds=args.ttl_seconds)
    tokens = {
        agent.agent_id: codec.issue(actor_context(scenario, agent), ttl=ttl)
        for agent in scenario.all_agents
    }

    seeded_memory_ids: list[str] = []
    if not args.no_seed_memory:
        seeded_memory_ids = await publish_prior_memory(
            scenario,
            api_url=api_url,
            token=tokens[scenario.scout.agent_id],
        )

    if args.out:
        out = Path(args.out).expanduser().resolve()
    else:
        import tempfile

        out = Path(tempfile.mkdtemp(prefix="swarmbrain-live-"))
    if out.is_relative_to(REPO_ROOT):
        raise SystemExit("--out must live outside the repository; it holds bearer tokens")

    manifest = write_bundle(
        out,
        scenario=scenario,
        specs=specs,
        tokens=tokens,
        api_url=api_url,
        lease_seconds=args.lease_seconds,
        seeded_memory_ids=seeded_memory_ids,
    )
    print_launch(out, manifest, specs, unattended=args.unattended)


_REDACTIONS: list[tuple[str, str]] = []


def _redact(value: str) -> str:
    cleaned = value
    for needle, replacement in _REDACTIONS:
        cleaned = cleaned.replace(needle, replacement)
    cleaned = cleaned.replace(str(REPO_ROOT), "<repo>")
    return cleaned.replace(str(Path.home()), "~")


def _rendered_transcript(records: list[dict[str, Any]]) -> str:
    """A readable transcript: assistant prose, tool calls, tool outcomes."""

    lines: list[str] = []
    for record in records:
        kind = record.get("type")
        message = record.get("message") or {}
        content = message.get("content")
        if kind == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and str(block.get("text", "")).strip():
                    lines.append(f"assistant: {str(block['text']).strip()}")
                elif block.get("type") == "tool_use":
                    arguments = json.dumps(block.get("input") or {}, sort_keys=True)
                    lines.append(f"tool_use  {block.get('name')} {arguments[:240]}")
        elif kind == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    status = "error" if block.get("is_error") else "ok"
                    body = json.dumps(block.get("content"), sort_keys=True)
                    lines.append(f"tool_result {status} {body[:240]}")
        elif kind == "result":
            lines.append(f"result    {record.get('subtype')}")
    return "\n".join(lines)


def _harness_identity(path: Path) -> dict[str, Any]:
    """Pull the model the harness actually reported from its result JSON."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": f"unreadable harness result: {exc}"}
    records: list[dict[str, Any]] = []
    try:  # a single --output-format json object
        parsed = json.loads(text)
        records = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:  # --output-format stream-json is JSONL
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    if not records:
        return {"error": "no JSON records in harness result"}
    payload = next(
        (item for item in reversed(records) if item.get("type") == "result"),
        records[-1],
    )
    tool_calls = [
        str(block.get("name"))
        for record in records
        if record.get("type") == "assistant"
        for block in (record.get("message") or {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    usage = payload.get("modelUsage") or {}
    identity: dict[str, Any] = {
        "reported_models": sorted(usage) if isinstance(usage, dict) else [],
        "tool_calls_in_order": tool_calls,
        "subtype": payload.get("subtype"),
        "is_error": payload.get("is_error"),
        "num_turns": payload.get("num_turns"),
        "duration_ms": payload.get("duration_ms"),
        "total_cost_usd": payload.get("total_cost_usd"),
    }
    result_text = payload.get("result")
    if isinstance(result_text, str):
        identity["result_tail"] = _redact(result_text)[-2000:]
    rendered = _rendered_transcript(records)
    if rendered:
        identity["transcript_tail"] = _redact(rendered)[-4000:]
    return {key: value for key, value in identity.items() if value not in (None, [])}


async def _report(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle).expanduser().resolve()
    _REDACTIONS.append((str(bundle), "<bundle>"))
    manifest = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
    viewer_token = ""
    for line in (bundle / "viewer.env").read_text(encoding="utf-8").splitlines():
        if "SWARMBRAIN_VIEWER_TOKEN" in line:
            viewer_token = line.split("=", 1)[1].strip().strip("'\"")
    if not viewer_token:
        raise SystemExit(f"no viewer token in {bundle / 'viewer.env'}")

    api_url = str(manifest["api_url"]).rstrip("/")
    run_id = str(manifest["scope"]["run_id"])
    async with httpx.AsyncClient(
        base_url=api_url,
        timeout=20.0,
        headers={"Authorization": f"Bearer {viewer_token}"},
    ) as client:
        events_response = await client.get(
            f"/v1/runs/{run_id}/events", params={"limit": 1000}
        )
        events_response.raise_for_status()
        events = list(events_response.json()["events"])
        metrics_response = await client.get(f"/v1/runs/{run_id}/metrics")
        metrics_response.raise_for_status()
        metrics = metrics_response.json()

    agents_by_id = {str(agent["agent_id"]): agent for agent in manifest["agents"]}
    tasks_by_id = {str(task["task_id"]): task for task in manifest["tasks"]}
    # The scout and the viewer are operator identities, not LLM agents; naming
    # them keeps the timeline honest about which rows a real model produced.
    labels: dict[str, dict[str, str]] = {
        agent_id: {"name": str(agent["name"]), "harness": str(agent["harness"])}
        for agent_id, agent in agents_by_id.items()
    }
    labels[str(manifest["scout_agent_id"])] = {
        "name": "prior-scout",
        "harness": "operator-seed",
    }
    labels[str(manifest["viewer_agent_id"])] = {
        "name": "viewer",
        "harness": "operator-console",
    }

    timeline = []
    for event in events:
        label = labels.get(str(event.get("agent_id")), {})
        entry = {
            "occurred_at": event["occurred_at"],
            "event_type": event["event_type"],
            "aggregate_type": event["aggregate_type"],
            "task": tasks_by_id.get(str(event.get("task_id")), {}).get("key"),
            "agent": label.get("name"),
            "harness": label.get("harness"),
        }
        timeline.append({key: value for key, value in entry.items() if value is not None})

    per_agent: dict[str, Any] = {}
    for agent_id, agent in agents_by_id.items():
        own = [event for event in events if str(event.get("agent_id")) == agent_id]
        per_agent[str(agent["harness_key"])] = {
            "identity": {
                key: agent[key] for key in ("harness", "provider", "model", "name")
            },
            "event_types": [event["event_type"] for event in own],
            "tasks_touched": sorted(
                {
                    tasks_by_id[str(event["task_id"])]["key"]
                    for event in own
                    if str(event.get("task_id")) in tasks_by_id
                }
            ),
            "completed_the_loop": all(
                any(event["event_type"] == expected for event in own)
                for expected in ("task.claimed", "memory.added", "task.completed")
            ),
        }

    for key, path in _pairs(args.harness_result, "--harness-result").items():
        per_agent.setdefault(key, {})["reported_by_harness"] = _harness_identity(Path(path))
    for key, note in _pairs(args.gap, "--gap").items():
        per_agent.setdefault(key, {})["gap"] = note
    for key, path in _pairs(args.transcript, "--transcript").items():
        try:
            tail = Path(path).read_text(encoding="utf-8")[-4000:]
        except OSError as exc:
            tail = f"unreadable transcript: {exc}"
        per_agent.setdefault(key, {})["transcript_tail"] = _redact(tail)

    artifact = {
        "kind": "live-swarm",
        "collected_at": datetime.now(UTC).isoformat(),
        "api_url": api_url,
        "run_id": run_id,
        "note": (
            "Every agent below is a real LLM CLI process that spoke the production "
            "stdio MCP bridge (swarmbrain-mcp) against a CockroachDB-backed API. "
            "`identity.model` is the label carried in that agent's bearer token; "
            "`reported_by_harness.reported_models` is what the harness itself "
            "reported using. Tokens and absolute home paths are redacted."
        ),
        "tasks": manifest["tasks"],
        "seeded_memory_ids": manifest["seeded_memory_ids"],
        "agents": per_agent,
        "event_timeline": timeline,
        "event_counts": {
            event_type: sum(1 for event in events if event["event_type"] == event_type)
            for event_type in sorted({str(event["event_type"]) for event in events})
        },
        "metrics": metrics,
    }

    directory = Path(args.evidence_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}-{args.label}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"evidence written to {path}")
    for key, detail in per_agent.items():
        if "completed_the_loop" in detail:
            verdict = "complete" if detail["completed_the_loop"] else "PARTIAL"
            print(f"  {key}: {verdict} ({', '.join(detail['event_types']) or 'no events'})")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "seed":
        asyncio.run(_seed(args))
        return
    asyncio.run(_report(args))


if __name__ == "__main__":
    main()
