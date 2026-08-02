from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from conftest import make_actor, make_task
from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
)

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
TOKEN_SECRET = "0123456789abcdef-restart-live-secret"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


class _Ack(ContractModel):
    ok: bool = True


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


async def _start_api(port: int) -> asyncio.subprocess.Process:
    assert DATABASE_URL is not None
    environment = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "SWARMBRAIN_BACKEND": "cockroach",
        "SWARMBRAIN_DATABASE_URL": DATABASE_URL,
        "SWARMBRAIN_DATABASE_POOL_MIN_SIZE": "1",
        "SWARMBRAIN_DATABASE_POOL_MAX_SIZE": "12",
        "SWARMBRAIN_TOKEN_SECRET": TOKEN_SECRET,
        "SWARMBRAIN_HOST": "127.0.0.1",
        "SWARMBRAIN_PORT": str(port),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "swarmbrain.cli.api",
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=1.0) as client:
        for _ in range(100):
            if process.returncode is not None:
                raise RuntimeError(f"Swarm Brain API exited with status {process.returncode}")
            try:
                response = await client.get("/readyz")
                if response.status_code == 200:
                    return process
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.05)
    await _stop_process(process)
    raise RuntimeError("Swarm Brain API did not become ready")


async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


@dataclass(slots=True)
class _McpCall:
    name: str
    arguments: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]


@dataclass(slots=True)
class _McpWorker:
    actor: ActorContext
    _requests: asyncio.Queue[_McpCall | None]
    _owner: asyncio.Task[None]
    _closed: bool = False

    @classmethod
    async def start(cls, actor: ActorContext, port: int) -> _McpWorker:
        requests: asyncio.Queue[_McpCall | None] = asyncio.Queue()
        ready = asyncio.get_running_loop().create_future()
        owner = asyncio.create_task(cls._serve(actor, port, requests, ready))
        try:
            await ready
        except BaseException:
            await asyncio.gather(owner, return_exceptions=True)
            raise
        return cls(actor=actor, _requests=requests, _owner=owner)

    @staticmethod
    async def _serve(
        actor: ActorContext,
        port: int,
        requests: asyncio.Queue[_McpCall | None],
        ready: asyncio.Future[None],
    ) -> None:
        token = RunTokenCodec(TOKEN_SECRET).issue(actor)
        environment = {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "SWARMBRAIN_API_URL": f"http://127.0.0.1:{port}",
            "SWARMBRAIN_AGENT_TOKEN": token,
            "SWARMBRAIN_RUN_ID": actor.run_id,
            "SWARMBRAIN_AGENT_ID": actor.agent_id,
            "SWARMBRAIN_REQUEST_TIMEOUT_SECONDS": "2",
        }
        try:
            with open(os.devnull, "w", encoding="utf-8") as error_log:
                async with stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=["-m", "swarmbrain.transports.mcp.server"],
                        env=environment,
                        cwd=str(PROJECT_ROOT),
                    ),
                    errlog=error_log,
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        ready.set_result(None)
                        while request := await requests.get():
                            try:
                                result = await session.call_tool(
                                    request.name,
                                    arguments=request.arguments,
                                )
                                messages = [
                                    str(getattr(item, "text", ""))
                                    for item in result.content
                                    if getattr(item, "text", None)
                                ]
                                assert not result.isError, "; ".join(messages)
                                assert result.structuredContent is not None
                                request.future.set_result(dict(result.structuredContent))
                            except Exception as exc:
                                request.future.set_exception(exc)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    async def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        if self._closed or self._owner.done():
            raise RuntimeError("MCP worker is not running")
        future = asyncio.get_running_loop().create_future()
        await self._requests.put(_McpCall(name=name, arguments=arguments, future=future))
        return await future

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if not self._owner.done():
                await self._requests.put(None)
        await self._owner


async def _expire_lease(database: CockroachDatabase, lease_id: str) -> None:
    async def body(connection: Any) -> _Ack:
        await connection.execute(
            """
            UPDATE task_leases
            SET claimed_at = now() - INTERVAL '30 seconds',
                renewed_at = now() - INTERVAL '30 seconds',
                expires_at = now() - INTERVAL '1 second'
            WHERE id = %s
            """,
            (lease_id,),
        )
        return _Ack()

    await database.run(body)


async def _cleanup(database: CockroachDatabase, actor: ActorContext) -> None:
    async def body(connection: Any) -> _Ack:
        scope = (actor.tenant_id, actor.run_id)
        await connection.execute(
            "DELETE FROM outbox_events WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM action_attempts WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM idempotency_records WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM memory_conflicts WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM memories WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute("DELETE FROM evidence WHERE tenant_id = %s", (actor.tenant_id,))
        await connection.execute(
            "DELETE FROM sources WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM runs WHERE tenant_id = %s AND id = %s",
            scope,
        )
        return _Ack()

    await database.run(body)


async def test_four_stdio_agents_survive_api_and_mcp_restart_with_durable_handoff(
    scope_ids: dict[str, str],
) -> None:
    assert DATABASE_URL is not None
    database = CockroachDatabase(DATABASE_URL, min_size=1, max_size=12)
    await database.start()
    coordination = CockroachCoordinationStore(database)
    memory = CockroachMemoryStore(database)
    actors = [
        make_actor(scope_ids, harness="claude-code", provider="anthropic", model="claude-local"),
        make_actor(scope_ids, harness="codex-cli", provider="openai", model="codex-local"),
        make_actor(scope_ids, harness="gemini-cli", provider="google", model="gemini-local"),
        make_actor(scope_ids, harness="opencode", provider="local", model="qwen-local"),
    ]
    tasks = [
        make_task(scope_ids, title=f"Restart gate task {index}", priority=index)
        for index in range(4)
    ]
    for task in tasks:
        await coordination.add_task(task)

    port = _unused_port()
    api: asyncio.subprocess.Process | None = None
    workers: list[_McpWorker] = []
    try:
        api = await _start_api(port)
        for actor in actors:
            workers.append(await _McpWorker.start(actor, port))

        claims = await asyncio.gather(
            *(
                worker.call(
                    "claim_task",
                    idempotency_key=f"restart-claim-{index}",
                    lease_seconds=60,
                )
                for index, worker in enumerate(workers)
            )
        )
        assert len({claim["task"]["task_id"] for claim in claims}) == 4
        assert len({claim["lease"]["lease_id"] for claim in claims}) == 4

        source_content = b"CockroachDB SQL listens on port 26257"
        source = await memory.register_source(
            actors[0],
            RegisterEvidenceSourceCommand(
                idempotency_key="restart-source",
                kind=EvidenceKind.COMMAND_OUTPUT,
                content_sha256=hashlib.sha256(source_content).hexdigest(),
                occurrence_key="restart-demo:cockroach-port",
                observed_at=datetime.now(UTC),
            ),
        )
        evidence = await memory.add_evidence(
            actors[0],
            AddEvidenceCommand(
                idempotency_key="restart-evidence",
                source_id=source.source_id,
                kind=EvidenceKind.COMMAND_OUTPUT,
                locator="stdout:1",
                excerpt="SQL port 26257",
            ),
        )
        wrong = await workers[0].call(
            "publish_memory",
            idempotency_key="restart-memory-wrong",
            content="The database port is 5432",
            kind="hypothesis",
            visibility="repository",
        )
        corrected_arguments = {
            "idempotency_key": "restart-memory-corrected",
            "content": "The CockroachDB SQL port is 26257",
            "kind": "invariant",
            "desired_state": "confirmed",
            "visibility": "repository",
            "evidence": [evidence.as_ref().model_dump(mode="json")],
            "supersedes_memory_id": wrong["memory"]["memory_id"],
        }
        corrected = await workers[0].call("publish_memory", **corrected_arguments)
        corrected_id = corrected["memory"]["memory_id"]
        recalled = await workers[2].call("recall_memory", text="CockroachDB SQL port")
        assert [hit["memory"]["memory_id"] for hit in recalled["hits"]] == [corrected_id]
        conflict = await workers[0].call(
            "report_conflict",
            idempotency_key="restart-conflict",
            memory_ids=[wrong["memory"]["memory_id"], corrected_id],
            description="The two persisted SQL ports are incompatible",
            evidence=[evidence.as_ref().model_dump(mode="json")],
        )

        checkpointed = await workers[1].call(
            "checkpoint_task",
            idempotency_key="restart-checkpoint",
            task_id=claims[1]["task"]["task_id"],
            summary="worker stopped after locating the parser",
            discoveries=["parser isolated"],
            remaining_work=["patch serializer"],
        )
        stopped_lease_id = checkpointed["lease"]["lease_id"]
        await workers[1].close()
        workers.pop(1)

        # Finish one worker's original task, then restart its entire stdio MCP
        # process. It will become the successor after the stopped lease expires.
        await workers[2].call(
            "complete_task",
            idempotency_key="restart-complete-successor-original",
            task_id=claims[3]["task"]["task_id"],
            summary="original assignment completed",
        )
        await workers[2].close()
        workers.pop(2)

        # Restart the API while two MCP processes still own live leases. Their
        # renewal loops must reconnect; their next fenced mutations prove that
        # the durable ownership survived the process boundary.
        await _stop_process(api)
        api = None
        await asyncio.sleep(0.2)
        api = await _start_api(port)

        replay = await workers[0].call("publish_memory", **corrected_arguments)
        assert replay["replayed"] is True
        assert replay["memory"]["memory_id"] == corrected_id
        await workers[0].call(
            "complete_task",
            idempotency_key="restart-complete-first",
            task_id=claims[0]["task"]["task_id"],
            summary="completed after API restart",
        )
        await workers[1].call(
            "complete_task",
            idempotency_key="restart-complete-third",
            task_id=claims[2]["task"]["task_id"],
            summary="completed after API restart",
        )

        successor = await _McpWorker.start(actors[3], port)
        workers.append(successor)
        await _expire_lease(database, stopped_lease_id)
        handoff = await successor.call(
            "claim_task",
            idempotency_key="restart-handoff-claim",
            task_id=claims[1]["task"]["task_id"],
            lease_seconds=60,
        )
        assert handoff["checkpoint"]["checkpoint_id"] == checkpointed["checkpoint"]["checkpoint_id"]
        assert handoff["checkpoint"]["remaining_work"] == ["patch serializer"]
        await successor.call(
            "complete_task",
            idempotency_key="restart-complete-handoff",
            task_id=handoff["task"]["task_id"],
            summary="successor resumed durable checkpoint",
        )

        token = RunTokenCodec(TOKEN_SECRET).issue(actors[0])
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            event_response = await client.get(
                f"/v1/runs/{actors[0].run_id}/events", params={"limit": 1000}
            )
        assert event_response.status_code == 200
        events = event_response.json()["events"]
        assert sum(event["event_type"] == "task.completed" for event in events) == 4
        assert any(event["event_type"] == "memory.added" for event in events)
        assert any(event["event_type"] == "conflict.reported" for event in events)

        async def durable_counts(connection: Any) -> _Ack:
            cursor = await connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM tasks
                     WHERE tenant_id = %s AND run_id = %s AND state = 'completed') AS tasks,
                    (SELECT count(*) FROM task_checkpoints
                     WHERE tenant_id = %s AND run_id = %s) AS checkpoints,
                    (SELECT count(*) FROM memories
                     WHERE tenant_id = %s AND run_id = %s) AS memories,
                    (SELECT count(*) FROM memory_conflicts
                     WHERE tenant_id = %s AND run_id = %s) AS conflicts,
                    (SELECT count(*) FROM idempotency_records
                     WHERE tenant_id = %s AND run_id = %s AND status = 'completed') AS idempotency,
                    (SELECT count(*) FROM swarm_events
                     WHERE tenant_id = %s AND run_id = %s) AS events
                """,
                (
                    actors[0].tenant_id,
                    actors[0].run_id,
                    actors[0].tenant_id,
                    actors[0].run_id,
                    actors[0].tenant_id,
                    actors[0].run_id,
                    actors[0].tenant_id,
                    actors[0].run_id,
                    actors[0].tenant_id,
                    actors[0].run_id,
                    actors[0].tenant_id,
                    actors[0].run_id,
                ),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert int(row["tasks"]) == 4
            assert int(row["checkpoints"]) == 1
            assert int(row["memories"]) == 2
            assert int(row["conflicts"]) == 1
            assert int(row["idempotency"]) >= 12
            assert int(row["events"]) == len(events)
            return _Ack()

        await database.run(durable_counts)
        assert conflict["conflict"]["status"] == "open"
    finally:
        for worker in reversed(workers):
            await worker.close()
        await _stop_process(api)
        try:
            await _cleanup(database, actors[0])
        finally:
            await database.close()
