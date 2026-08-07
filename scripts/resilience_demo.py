#!/usr/bin/env python3
"""Rehearse the node-kill resilience beat against a real three-node CockroachDB.

The script owns the whole rehearsal: it starts a fresh three-node insecure
cluster on non-conflicting ports, installs the Swarm Brain schema, runs the
scripted swarm demo against node 1, SIGKILLs a *non-gateway* node at a
deterministic point mid-run, lets the remaining beats finish against the
surviving quorum, then re-reads the run's tasks, leases, memories and events
from a surviving node and writes a redacted evidence artifact.

What it proves: a CockroachDB node can die mid-run without the swarm losing a
task, a lease, a memory or an event.

What it deliberately does not prove: gateway failover. The API pool is
connected to one node's SQL port; killing that node would break the client's
own connections. A production deployment puts a load balancer (or a multi-host
connection string) in front of the SQL layer. The rehearsal kills node 3 and
says exactly that.

Known flake with the default trigger: if the kill lands on a range whose
leaseholder was the dying node, the API answers 503 ``ambiguous_commit`` with
``retryable: true`` — correct behaviour, since CockroachDB cannot prove whether
COMMIT took effect and the kernel refuses to guess. ``swarmbrain-demo`` does not
retry on that, so the run aborts with nothing lost. The artifact records
``failure_mode`` and still reads the data state off the survivors.
``--kill-trigger lease-wait`` avoids the window entirely. See
docs/resilience-demo.md.

Usage:

    uv run scripts/resilience_demo.py
    uv run scripts/resilience_demo.py --kill-trigger lease-wait --pause-before-demo

Environment overrides:

    COCKROACH_BINARY     path to the cockroach binary (default: discovered)
    RESILIENCE_STORE_DIR directory for the three node stores and logs
                         (default: a fresh temporary directory, removed on exit)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parent.parent
DATABASE = "swarmbrain_resilience"
TOKEN_SECRET = "resilience-demo-secret-0123"  # noqa: S105 - local insecure rehearsal only
BEAT_LINE = re.compile(r"^\[(ok|FAILED)\]\s+(.+)$")
LEASE_WAIT_LINE = re.compile(r"^waiting \d+s for the crashed worker's lease to expire")
AMBIGUOUS = re.compile(r"ambiguous_commit")
CANDIDATE_BINARIES = (
    Path.home() / ".local" / "bin" / "cockroach",
    Path("/usr/local/bin/cockroach"),
    Path("/opt/homebrew/bin/cockroach"),
)


class RehearsalError(RuntimeError):
    """A rehearsal precondition or bounded wait failed."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _log(message: str) -> None:
    print(f"[resilience {_now()}] {message}", flush=True)


def _redact(text: str) -> str:
    """Strip the operator's home directory out of any path-shaped string."""

    return text.replace(str(Path.home()), "~")


# ---------------------------------------------------------------------------
# cluster


@dataclass
class Node:
    index: int
    sql_port: int
    http_port: int
    store: Path
    log_dir: Path
    stdout_path: Path
    process: subprocess.Popen[bytes] | None = None
    killed_at: str | None = None

    @property
    def sql_addr(self) -> str:
        return f"localhost:{self.sql_port}"

    @property
    def http_addr(self) -> str:
        return f"localhost:{self.http_port}"

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def describe(self, *, cockroach_node_id: str | None = None) -> dict[str, Any]:
        return {
            "index": self.index,
            "cockroach_node_id": cockroach_node_id,
            "sql_addr": self.sql_addr,
            "http_addr": self.http_addr,
            "store": _redact(str(self.store)),
            "pid": self.process.pid if self.process is not None else None,
            "killed_at": self.killed_at,
        }


def discover_binary() -> Path:
    override = os.getenv("COCKROACH_BINARY", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RehearsalError(f"COCKROACH_BINARY is not an executable file: {candidate}")
        return candidate
    found = shutil.which("cockroach")
    if found:
        return Path(found)
    for candidate in CANDIDATE_BINARIES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    running = _binary_of_running_cockroach()
    if running is not None:
        return running
    raise RehearsalError(
        "cockroach binary not found. Set COCKROACH_BINARY to its absolute path, "
        "or put it on PATH."
    )


def _binary_of_running_cockroach() -> Path | None:
    """Last resort: read the image path of an already-running cockroach process."""

    listing = subprocess.run(
        ["ps", "axo", "pid=,comm="],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    for line in listing.stdout.splitlines():
        pid, _, comm = line.strip().partition(" ")
        if Path(comm.strip()).name != "cockroach" or not pid.isdigit():
            continue
        opened = subprocess.run(
            ["lsof", "-p", pid, "-a", "-d", "txt", "-Fn"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        for entry in opened.stdout.splitlines():
            if entry.startswith("n") and entry.rstrip().endswith("/cockroach"):
                candidate = Path(entry[1:].strip())
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
    return None


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_for_port(port: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.25)
    raise RehearsalError(f"port {port} did not start accepting connections within {timeout:.0f}s")


class Cluster:
    """A throwaway three-node insecure CockroachDB cluster."""

    def __init__(
        self,
        *,
        binary: Path,
        base_dir: Path,
        sql_base_port: int,
        http_base_port: int,
        node_count: int = 3,
    ) -> None:
        self.binary = binary
        self.base_dir = base_dir
        self.nodes = [
            Node(
                index=index,
                sql_port=sql_base_port + index - 1,
                http_port=http_base_port + index - 1,
                store=base_dir / f"n{index}",
                log_dir=base_dir / "logs" / f"n{index}",
                stdout_path=base_dir / "logs" / f"n{index}.out",
            )
            for index in range(1, node_count + 1)
        ]
        self._stdout_handles: list[Any] = []

    @property
    def gateway(self) -> Node:
        return self.nodes[0]

    def node(self, index: int) -> Node:
        for node in self.nodes:
            if node.index == index:
                return node
        raise RehearsalError(f"no such node index: {index}")

    @property
    def join(self) -> str:
        return ",".join(node.sql_addr for node in self.nodes)

    def version(self) -> str:
        result = subprocess.run(
            [str(self.binary), "version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Build Tag:"):
                return line.split(":", 1)[1].strip()
        return "unknown"

    def preflight(self) -> None:
        busy = [
            port
            for node in self.nodes
            for port in (node.sql_port, node.http_port)
            if not _port_is_free(port)
        ]
        if busy:
            raise RehearsalError(
                "these ports are already in use: "
                + ", ".join(str(port) for port in busy)
                + ". A previous rehearsal may still be running; stop it (or pass "
                "--sql-base-port/--http-base-port) and retry."
            )

    def start(self, *, timeout: float = 90.0) -> None:
        for node in self.nodes:
            node.store.mkdir(parents=True, exist_ok=True)
            node.log_dir.mkdir(parents=True, exist_ok=True)
            node.stdout_path.parent.mkdir(parents=True, exist_ok=True)
            handle = node.stdout_path.open("wb")
            self._stdout_handles.append(handle)
            node.process = subprocess.Popen(
                [
                    str(self.binary),
                    "start",
                    "--insecure",
                    f"--store={node.store}",
                    f"--listen-addr={node.sql_addr}",
                    f"--http-addr={node.http_addr}",
                    f"--join={self.join}",
                    f"--log-dir={node.log_dir}",
                    "--cache=128MiB",
                    "--max-sql-memory=256MiB",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            _log(f"node {node.index} starting on {node.sql_addr} (pid {node.process.pid})")
        for node in self.nodes:
            _wait_for_port(node.sql_port, timeout=timeout)

    def init(self, *, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            result = subprocess.run(
                [
                    str(self.binary),
                    "init",
                    "--insecure",
                    f"--host={self.gateway.sql_addr}",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            combined = f"{result.stdout}{result.stderr}"
            if result.returncode == 0 or "already been initialized" in combined:
                _log("cluster initialized")
                return
            last = combined.strip()
            time.sleep(1.0)
        raise RehearsalError(f"cluster init did not succeed within {timeout:.0f}s: {last}")

    def sql(
        self,
        node: Node,
        statement: str,
        *,
        database: str | None = None,
        timeout: float = 60.0,
        check: bool = True,
    ) -> list[dict[str, str]]:
        command = [
            str(self.binary),
            "sql",
            "--insecure",
            f"--host={node.sql_addr}",
            "--format=csv",
            "-e",
            statement,
        ]
        if database is not None:
            command.insert(4, f"--database={database}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            if check:
                raise RehearsalError(
                    f"SQL against {node.sql_addr} failed ({result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip()}\nstatement: {statement}"
                )
            return []
        return list(csv.DictReader(StringIO(result.stdout)))

    def node_status(
        self, *, ranges: bool = False, timeout: float = 60.0, check: bool = True
    ) -> list[dict[str, str]]:
        """Read `cockroach node status`.

        This is the supported replacement for ``crdb_internal.gossip_nodes``:
        v26 gates the ``crdb_internal`` and ``system`` interfaces behind
        ``allow_unsafe_internals``, and this rehearsal stays on supported
        surfaces so the evidence is reproducible by an operator.
        """

        command = [
            str(self.binary),
            "node",
            "status",
            "--insecure",
            f"--host={self.gateway.sql_addr}",
            "--format=csv",
        ]
        if ranges:
            command.append("--ranges")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            if check:
                raise RehearsalError(f"`cockroach node status` failed: {result.stderr.strip()}")
            return []
        return list(csv.DictReader(StringIO(result.stdout)))

    def wait_for_live_nodes(self, expected: int, *, timeout: float = 120.0) -> list[dict[str, str]]:
        deadline = time.monotonic() + timeout
        rows: list[dict[str, str]] = []
        while time.monotonic() < deadline:
            rows = self.node_status(check=False)
            live = [row for row in rows if not _falsy(row.get("is_live"))]
            if len(live) == expected:
                _log(f"{expected} nodes live: {[row.get('address') for row in live]}")
                return rows
            time.sleep(1.0)
        raise RehearsalError(
            f"cluster did not report {expected} live nodes within {timeout:.0f}s (last: {rows})"
        )

    def wait_for_replication(self, *, timeout: float = 180.0) -> dict[str, Any]:
        """Wait until no range is under-replicated (fewer than three replicas)."""

        started = time.monotonic()
        deadline = started + timeout
        under = -1
        total = -1
        while time.monotonic() < deadline:
            rows = self.sql(
                self.gateway,
                "SELECT count(*) AS total, "
                "count(*) FILTER (WHERE array_length(replicas, 1) < 3) AS under "
                "FROM [SHOW CLUSTER RANGES]",
                check=False,
                timeout=120.0,
            )
            if rows:
                total = int(rows[0]["total"])
                under = int(rows[0]["under"])
                if under == 0 and total > 0:
                    waited = time.monotonic() - started
                    _log(f"all {total} ranges have >= 3 replicas after {waited:.1f}s")
                    return {
                        "total_ranges": total,
                        "under_replicated_ranges": under,
                        "waited_seconds": round(waited, 1),
                    }
            time.sleep(2.0)
        raise RehearsalError(
            f"ranges were still under-replicated after {timeout:.0f}s "
            f"(total={total}, under_replicated={under})"
        )

    def kill(self, index: int) -> dict[str, Any]:
        node = self.node(index)
        if node.process is None or node.process.poll() is not None:
            raise RehearsalError(f"node {index} is not running; cannot kill it")
        pid = node.process.pid
        os.kill(pid, signal.SIGKILL)
        node.killed_at = _now()
        try:
            node.process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - SIGKILL is immediate
            raise RehearsalError(f"node {index} (pid {pid}) survived SIGKILL") from exc
        _log(f"SIGKILL delivered to node {index} (pid {pid}) on {node.sql_addr}")
        return {"node_index": index, "pid": pid, "sql_addr": node.sql_addr, "at": node.killed_at}

    def stop(self) -> None:
        for node in self.nodes:
            process = node.process
            if process is None or process.poll() is not None:
                continue
            process.terminate()
        for node in self.nodes:
            process = node.process
            if process is None:
                continue
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        for handle in self._stdout_handles:
            handle.close()
        self._stdout_handles.clear()


# ---------------------------------------------------------------------------
# demo subprocess


@dataclass
class BeatObservation:
    ordinal: int
    status: str
    title: str
    seconds_after_start: float
    after_kill: bool


@dataclass
class DemoOutcome:
    command: list[str]
    exit_code: int
    duration_seconds: float
    timed_out: bool
    beats: list[BeatObservation] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    report: dict[str, Any] | None = None
    kill: dict[str, Any] | None = None
    failure_mode: str | None = None


def run_demo_with_kill(
    *,
    cluster: Cluster,
    kill_node_index: int,
    kill_trigger: str,
    kill_after_beats: int,
    lease_seconds: int,
    demo_evidence_dir: Path,
    timeout: float,
) -> DemoOutcome:
    """Run swarmbrain-demo against node 1 and SIGKILL a non-gateway node mid-run.

    Both kill points are deterministic, keyed off the demo's own narration:

    ``lease-wait`` (default) fires when the crash-handoff beat announces its
                   real-time wait for a lease to expire — a window in which the
                   demo client has no transaction in flight. Beats 6 to 8 then
                   run entirely against the degraded quorum. Deterministic.
    ``beat``       fires the moment the Nth beat status line arrives, i.e.
                   between two beats but with the next beat's write traffic
                   starting immediately. The harsher probe, and the one that
                   finds the ambiguous-commit window described in
                   docs/resilience-demo.md.
    """

    uv = shutil.which("uv")
    if uv is None:
        raise RehearsalError("uv was not found on PATH; the demo runs via `uv run`")
    command = [
        uv,
        "run",
        "--extra",
        "serve",
        "--extra",
        "crdb",
        "swarmbrain-demo",
        "--lease-seconds",
        str(lease_seconds),
        "--evidence-dir",
        str(demo_evidence_dir),
    ]
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "SWARMBRAIN_BACKEND": "cockroach",
        "SWARMBRAIN_DATABASE_URL": database_url(cluster.gateway),
        "SWARMBRAIN_TOKEN_SECRET": TOKEN_SECRET,
    }
    outcome = DemoOutcome(
        command=list(command),
        exit_code=-1,
        duration_seconds=0.0,
        timed_out=False,
    )
    _log(f"starting the swarm demo against node 1 ({cluster.gateway.sql_addr})")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    timed_out = threading.Event()

    def _abort() -> None:
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(timeout, _abort)
    watchdog.start()
    beat_ordinal = 0
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\n")
            outcome.output.append(line)
            print(f"    demo | {line}", flush=True)
            match = BEAT_LINE.match(line)
            if match is not None:
                beat_ordinal += 1
                outcome.beats.append(
                    BeatObservation(
                        ordinal=beat_ordinal,
                        status=match.group(1),
                        title=match.group(2).strip(),
                        seconds_after_start=round(time.monotonic() - started, 2),
                        after_kill=outcome.kill is not None,
                    )
                )
            if outcome.kill is not None:
                continue
            if kill_trigger == "beat":
                if match is None or beat_ordinal != kill_after_beats:
                    continue
                after = {
                    "trigger_line": line,
                    "ordinal": beat_ordinal,
                    "status": match.group(1),
                    "title": match.group(2).strip(),
                }
            else:
                if LEASE_WAIT_LINE.match(line) is None:
                    continue
                after = {
                    "trigger_line": line,
                    "ordinal": beat_ordinal,
                    "status": "idle",
                    "title": "the crash-handoff beat's real-time lease wait",
                }
            killed = cluster.kill(kill_node_index)
            killed["seconds_after_demo_start"] = round(time.monotonic() - started, 2)
            killed["trigger"] = kill_trigger
            killed["after_beat"] = after
            outcome.kill = killed
        outcome.exit_code = process.wait()
    finally:
        watchdog.cancel()
        if process.poll() is None:  # pragma: no cover - only on an unexpected break
            process.kill()
            process.wait(timeout=30)
    outcome.duration_seconds = round(time.monotonic() - started, 2)
    outcome.timed_out = timed_out.is_set()
    outcome.report = _newest_report(demo_evidence_dir)
    outcome.failure_mode = _classify_failure(outcome)
    return outcome


def _classify_failure(outcome: DemoOutcome) -> str | None:
    """Name the failure so an aborted take is diagnosed from the artifact alone."""

    if outcome.exit_code == 0 and not outcome.timed_out:
        return None
    if outcome.timed_out:
        return "demo_timed_out"
    if any(AMBIGUOUS.search(line) for line in outcome.output):
        return "demo_client_did_not_retry_ambiguous_commit"
    if any(beat.status == "FAILED" for beat in outcome.beats):
        return "demo_beat_check_failed"
    return "demo_exited_nonzero"


def _newest_report(directory: Path) -> dict[str, Any] | None:
    artifacts = sorted(directory.glob("*-swarm-demo.json"))
    if not artifacts:
        return None
    with artifacts[-1].open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else None


def database_url(node: Node) -> str:
    return f"postgresql://root@{node.sql_addr}/{DATABASE}?sslmode=disable"


# ---------------------------------------------------------------------------
# schema


def install_schema(cluster: Cluster) -> dict[str, Any]:
    uv = shutil.which("uv")
    if uv is None:
        raise RehearsalError("uv was not found on PATH; schema install runs via `uv run`")
    cluster.sql(cluster.gateway, f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    env = {
        **os.environ,
        "SWARMBRAIN_BACKEND": "cockroach",
        "SWARMBRAIN_DATABASE_URL": database_url(cluster.gateway),
        "SWARMBRAIN_TOKEN_SECRET": TOKEN_SECRET,
    }
    outputs: dict[str, Any] = {}
    for action in ("install", "verify"):
        result = subprocess.run(
            [uv, "run", "--extra", "crdb", "swarmbrain-schema", action],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if result.returncode != 0:
            raise RehearsalError(
                f"swarmbrain-schema {action} failed ({result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        outputs[action] = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        _log(f"schema {action}: {outputs[action]}")
    return outputs


# ---------------------------------------------------------------------------
# post-kill data state


COUNT_QUERY = """
SELECT
    (SELECT count(*) FROM tasks WHERE run_id = '{run}') AS tasks,
    (SELECT count(*) FROM tasks WHERE run_id = '{run}' AND state = 'completed')
        AS tasks_completed,
    (SELECT count(*) FROM task_leases WHERE run_id = '{run}') AS leases,
    (SELECT count(*) FROM task_leases WHERE run_id = '{run}' AND status = 'active')
        AS leases_active,
    (SELECT count(*) FROM task_leases WHERE run_id = '{run}' AND status = 'completed')
        AS leases_completed,
    (SELECT count(*) FROM task_checkpoints WHERE run_id = '{run}') AS checkpoints,
    (SELECT count(*) FROM task_completions WHERE run_id = '{run}') AS task_completions,
    (SELECT count(*) FROM memories WHERE run_id = '{run}') AS memories,
    (SELECT count(*) FROM memories WHERE run_id = '{run}' AND recorded_to IS NULL)
        AS memories_current,
    (SELECT count(*) FROM swarm_events WHERE run_id = '{run}') AS swarm_events,
    (SELECT count(*) FROM outbox_events WHERE run_id = '{run}') AS outbox_events,
    (SELECT count(*) FROM agents WHERE run_id = '{run}') AS agents
"""


def _checked_run_id(run_id: str) -> str:
    """Interpolating into SQL is only safe because this must be a UUID."""

    try:
        return str(UUID(run_id))
    except ValueError as exc:
        raise RehearsalError(f"run id is not a UUID: {run_id!r}") from exc


def read_counts(cluster: Cluster, node: Node, run_id: str) -> dict[str, int]:
    rows = cluster.sql(node, COUNT_QUERY.format(run=_checked_run_id(run_id)), database=DATABASE)
    if not rows:
        raise RehearsalError(f"count query returned no rows from {node.sql_addr}")
    return {key: int(value) for key, value in rows[0].items()}


def read_event_types(cluster: Cluster, node: Node, run_id: str) -> dict[str, int]:
    rows = cluster.sql(
        node,
        "SELECT event_type, count(*) AS n FROM swarm_events "
        f"WHERE run_id = '{_checked_run_id(run_id)}' GROUP BY event_type ORDER BY event_type",
        database=DATABASE,
    )
    return {row["event_type"]: int(row["n"]) for row in rows}


def wait_for_dead_node(cluster: Cluster, dead_index: int, *, timeout: float = 120.0) -> list[dict]:
    """Poll `cockroach node status` until the killed node is reported not live.

    Nodes are matched by SQL address: CockroachDB assigns node IDs in join
    order, which is a race between the three processes and does not track this
    script's node indices.
    """

    node = cluster.node(dead_index)
    deadline = time.monotonic() + timeout
    rows: list[dict[str, str]] = []
    while time.monotonic() < deadline:
        rows = cluster.node_status(ranges=True, check=False)
        dead = [row for row in rows if _falsy(row.get("is_live"))]
        if len(dead) == 1 and dead[0].get("address") == node.sql_addr:
            return rows
        time.sleep(2.0)
    raise RehearsalError(
        f"node {dead_index} ({node.sql_addr}) was not reported dead within {timeout:.0f}s; "
        f"last status: {rows}"
    )


def _falsy(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"false", "f", "0", "no"}


# ---------------------------------------------------------------------------
# assertions and evidence


@dataclass
class Assertion:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def build_assertions(
    *,
    outcome: DemoOutcome,
    node_status: Sequence[dict[str, str]],
    counts_by_node: dict[str, dict[str, int]],
    killed_index: int,
) -> list[Assertion]:
    report = outcome.report or {}
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    beats_after_kill = [beat for beat in outcome.beats if beat.after_kill]
    failed_beats = [beat.title for beat in outcome.beats if beat.status != "ok"]
    live = [row for row in node_status if not _falsy(row.get("is_live"))]
    dead = [row for row in node_status if _falsy(row.get("is_live"))]
    survivors = sorted(counts_by_node)
    agreed = len({json.dumps(value, sort_keys=True) for value in counts_by_node.values()}) == 1
    reference = counts_by_node[survivors[0]] if survivors else {}

    assertions = [
        Assertion(
            "node_killed_mid_run",
            outcome.kill is not None and bool(beats_after_kill),
            (
                f"node {killed_index} SIGKILLed after beat "
                f"{outcome.kill['after_beat']['ordinal'] if outcome.kill else '?'}; "
                f"{len(beats_after_kill)} beats completed afterwards"
            ),
        ),
        Assertion(
            "demo_exited_zero",
            outcome.exit_code == 0 and not outcome.timed_out,
            f"swarmbrain-demo exit code {outcome.exit_code} after {outcome.duration_seconds}s",
        ),
        Assertion(
            "every_beat_green",
            bool(outcome.beats) and not failed_beats,
            (
                f"{len(outcome.beats)} beats reported, all ok"
                if not failed_beats
                else f"failed beats: {failed_beats}"
            ),
        ),
        Assertion(
            "report_ok",
            bool(report.get("ok")),
            f"demo evidence report ok={report.get('ok')}",
        ),
        Assertion(
            "one_node_dead_cluster_serving",
            len(dead) == 1 and len(live) == 2,
            f"{len(live)} live nodes, {len(dead)} dead node(s) per `cockroach node status`",
        ),
        Assertion(
            "no_unavailable_ranges",
            all(int(row.get("ranges_unavailable") or 0) == 0 for row in live),
            "every surviving node reports ranges_unavailable=0",
        ),
        Assertion(
            "surviving_nodes_agree",
            agreed and len(survivors) >= 2,
            f"identical counts read from {survivors}",
        ),
        Assertion(
            "all_tasks_still_completed",
            reference.get("tasks_completed") == reference.get("tasks") == metrics.get("tasks_total"),
            (
                f"{reference.get('tasks_completed')}/{reference.get('tasks')} tasks completed "
                f"in the database; demo metrics report tasks_total={metrics.get('tasks_total')}"
            ),
        ),
        Assertion(
            "no_lease_left_active",
            reference.get("leases_active") == 0 and (reference.get("leases") or 0) > 0,
            (
                f"{reference.get('leases')} leases persisted, "
                f"{reference.get('leases_active')} still active"
            ),
        ),
        Assertion(
            "memories_and_events_readable",
            (reference.get("memories") or 0) > 0 and (reference.get("swarm_events") or 0) > 0,
            (
                f"{reference.get('memories')} memories and {reference.get('swarm_events')} "
                "swarm events readable from the surviving quorum"
            ),
        ),
    ]
    return assertions


def write_evidence(evidence_dir: Path, payload: dict[str, Any]) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact = evidence_dir / f"{stamp}-node-kill-resilience.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def summarize_beats(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    beats = report.get("beats")
    if not isinstance(beats, list):
        return []
    summary = []
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        checks = beat.get("checks") if isinstance(beat.get("checks"), list) else []
        summary.append(
            {
                "key": beat.get("key"),
                "title": beat.get("title"),
                "ok": beat.get("ok"),
                "checks_passed": sum(1 for check in checks if isinstance(check, dict) and check.get("ok")),
                "checks_total": len(checks),
            }
        )
    return summary


# ---------------------------------------------------------------------------
# entry point


def _resolve_base_dir(argument: str | None) -> tuple[Path, bool]:
    """Return the store root and whether this script created the directory itself.

    A directory the operator named is emptied rather than removed, so a rerun
    against the same path is still a fresh cluster but the path itself survives.
    """

    configured = argument or os.getenv("RESILIENCE_STORE_DIR", "").strip()
    if not configured:
        return Path(tempfile.mkdtemp(prefix="swarmbrain-resilience-")), True
    base = Path(configured).expanduser().resolve()
    if base == REPO_ROOT or base.is_relative_to(REPO_ROOT):
        raise RehearsalError(f"refusing to place CockroachDB stores inside the repository: {base}")
    if base.exists():
        _empty_directory(base)
    base.mkdir(parents=True, exist_ok=True)
    return base, False


def _empty_directory(base: Path) -> None:
    for child in base.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rehearse the node-kill resilience beat: three-node CockroachDB, "
            "swarm demo mid-run, one non-gateway node SIGKILLed."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "directory for node stores and logs (default: $RESILIENCE_STORE_DIR, "
            "else a fresh temporary directory). Never inside the repository."
        ),
    )
    parser.add_argument("--sql-base-port", type=int, default=26260)
    parser.add_argument("--http-base-port", type=int, default=8290)
    parser.add_argument(
        "--kill-node",
        type=int,
        default=3,
        help="node index to SIGKILL; must not be 1, the demo's SQL gateway (default: 3)",
    )
    parser.add_argument(
        "--kill-after-beats",
        type=int,
        default=2,
        help="deliver SIGKILL once this many demo beats have reported (default: 2, the claim race)",
    )
    parser.add_argument(
        "--kill-trigger",
        choices=("beat", "lease-wait"),
        default="lease-wait",
        help=(
            "when to SIGKILL: 'lease-wait' (default) fires during the crash-handoff beat's "
            "real-time lease wait, when the demo client has no transaction in flight — "
            "deterministic, and the take to shoot; 'beat' fires between beats while write "
            "traffic continues, which is the harsher probe and passes about half the time "
            "because swarmbrain-demo does not retry a retryable 503 (see "
            "docs/resilience-demo.md)"
        ),
    )
    parser.add_argument("--lease-seconds", type=int, default=20)
    parser.add_argument("--evidence-dir", default=str(REPO_ROOT / "evidence"))
    parser.add_argument(
        "--demo-timeout-seconds",
        type=float,
        default=300.0,
        help="hard bound on the demo subprocess (default: 300)",
    )
    parser.add_argument(
        "--pause-before-demo",
        action="store_true",
        help=(
            "wait for Enter once the cluster is replicated and the schema is installed, "
            "so a screen recording can start on the interesting 25 seconds"
        ),
    )
    parser.add_argument(
        "--keep-cluster",
        action="store_true",
        help="leave the cluster running and the stores on disk (debugging only)",
    )
    return parser


def _flatten(rows: Iterable[dict[str, str]], keys: Sequence[str]) -> list[dict[str, str]]:
    return [{key: row.get(key, "") for key in keys if key in row} for row in rows]


def main() -> int:
    args = _parser().parse_args()
    if args.kill_node == 1:
        raise SystemExit(
            "refusing to kill node 1: it is the demo's SQL gateway, and this rehearsal "
            "deliberately does not claim gateway failover"
        )
    binary = discover_binary()
    base_dir, created_dir = _resolve_base_dir(args.data_dir)
    cluster = Cluster(
        binary=binary,
        base_dir=base_dir,
        sql_base_port=args.sql_base_port,
        http_base_port=args.http_base_port,
    )
    started_at = _now()
    started_monotonic = time.monotonic()
    payload: dict[str, Any] = {
        "artifact": "node-kill-resilience",
        "started_at": started_at,
        "ok": False,
    }
    exit_code = 1
    try:
        _log(f"cockroach binary: {_redact(str(binary))} ({cluster.version()})")
        _log(f"store root: {_redact(str(base_dir))}")
        cluster.preflight()
        cluster.start()
        cluster.init()
        status_before = cluster.wait_for_live_nodes(len(cluster.nodes))
        node_ids = {row.get("address", ""): row.get("id", "") for row in status_before}
        replication = cluster.wait_for_replication()
        schema = install_schema(cluster)
        demo_evidence_dir = base_dir / "demo-evidence"
        demo_evidence_dir.mkdir(parents=True, exist_ok=True)
        if args.pause_before_demo:
            if not sys.stdin.isatty():
                raise RehearsalError("--pause-before-demo needs an interactive terminal")
            _log("cluster replicated and schema installed. Press Enter to start the run.")
            input()
        outcome = run_demo_with_kill(
            cluster=cluster,
            kill_node_index=args.kill_node,
            kill_trigger=args.kill_trigger,
            kill_after_beats=args.kill_after_beats,
            lease_seconds=args.lease_seconds,
            demo_evidence_dir=demo_evidence_dir,
            timeout=args.demo_timeout_seconds,
        )
        if outcome.kill is None:
            raise RehearsalError(
                f"the kill never fired: the demo never reached the '{args.kill_trigger}' "
                f"trigger. Demo tail: {outcome.output[-5:]}"
            )
        report = outcome.report or {}
        run_id = ""
        scenario = report.get("scenario") if isinstance(report, dict) else None
        if isinstance(scenario, dict):
            run_id = str(scenario.get("run_id") or "")
        if not run_id:
            # The demo aborted before writing its artifact. The run row is still
            # there, and reading its data state is exactly the evidence that
            # matters: an aborted client must not mean lost committed work.
            _log("demo wrote no artifact; recovering the run id from the database")
            rows = cluster.sql(
                cluster.gateway,
                "SELECT id FROM runs ORDER BY created_at DESC LIMIT 1",
                database=DATABASE,
                check=False,
            )
            run_id = rows[0]["id"] if rows else ""
        if not run_id:
            raise RehearsalError(
                "no run id: the demo wrote no evidence artifact and the runs table is empty"
            )
        if outcome.failure_mode is not None:
            _log(f"demo did not pass; classified failure mode: {outcome.failure_mode}")

        _log("waiting for the cluster to report the killed node as dead")
        node_status = wait_for_dead_node(cluster, args.kill_node)
        survivors = [node for node in cluster.nodes if node.alive]
        counts_by_node = {
            f"node{node.index}@{node.sql_addr}": read_counts(cluster, node, run_id)
            for node in survivors
        }
        event_types = read_event_types(cluster, survivors[-1], run_id)
        ranges_after = cluster.sql(
            cluster.gateway,
            "SELECT count(*) AS total_ranges, "
            "count(*) FILTER (WHERE array_length(voting_replicas, 1) >= 2) AS ranges_with_quorum "
            "FROM [SHOW CLUSTER RANGES]",
            check=False,
        )
        assertions = build_assertions(
            outcome=outcome,
            node_status=node_status,
            counts_by_node=counts_by_node,
            killed_index=args.kill_node,
        )
        ok = all(assertion.ok for assertion in assertions)
        payload.update(
            {
                "ok": ok,
                "finished_at": _now(),
                "duration_seconds": round(time.monotonic() - started_monotonic, 1),
                "cluster": {
                    "cockroach_version": cluster.version(),
                    "cockroach_binary": _redact(str(binary)),
                    "mode": "insecure, single host, three processes",
                    "store_root": _redact(str(base_dir)),
                    "database": DATABASE,
                    "gateway_node_index": cluster.gateway.index,
                    "database_url": database_url(cluster.gateway),
                    "nodes": [
                        node.describe(cockroach_node_id=node_ids.get(node.sql_addr))
                        for node in cluster.nodes
                    ],
                    "replication_before_kill": replication,
                    "schema": schema,
                },
                "kill": {
                    **outcome.kill,
                    "cockroach_node_id": node_ids.get(cluster.node(args.kill_node).sql_addr),
                    "signal": "SIGKILL",
                    "is_gateway": False,
                    "beats_completed_after_kill": [
                        {
                            "ordinal": beat.ordinal,
                            "status": beat.status,
                            "title": beat.title,
                            "seconds_after_demo_start": beat.seconds_after_start,
                        }
                        for beat in outcome.beats
                        if beat.after_kill
                    ],
                },
                "demo": {
                    "command": ["uv", *outcome.command[1:]],
                    "lease_seconds": args.lease_seconds,
                    "exit_code": outcome.exit_code,
                    "timed_out": outcome.timed_out,
                    "failure_mode": outcome.failure_mode,
                    "duration_seconds": outcome.duration_seconds,
                    "run_id": run_id,
                    "verdict": "PASSED" if outcome.exit_code == 0 else "FAILED",
                    "beats": summarize_beats(report),
                    "metrics": report.get("metrics", {}),
                    "narration_tail": outcome.output[-12:],
                },
                "cluster_after_kill": {
                    "node_status": _flatten(
                        node_status,
                        (
                            "id",
                            "address",
                            "is_available",
                            "is_live",
                            "ranges",
                            "ranges_unavailable",
                            "ranges_underreplicated",
                        ),
                    ),
                    "ranges": ranges_after[0] if ranges_after else {},
                    "live_nodes": sum(1 for row in node_status if not _falsy(row.get("is_live"))),
                    "dead_nodes": sum(1 for row in node_status if _falsy(row.get("is_live"))),
                    "queries_served_from": sorted(node.sql_addr for node in survivors),
                },
                "data_state": {
                    "read_from": sorted(counts_by_node),
                    "counts": counts_by_node,
                    "swarm_event_types": event_types,
                },
                "assertions": [assertion.to_dict() for assertion in assertions],
                "does_not_prove": [
                    "gateway failover: the API pool holds SQL connections to node 1 only, so "
                    "killing node 1 would break the client's own sockets. Production puts a load "
                    "balancer or a multi-host connection string in front of the SQL layer.",
                    "multi-region or cross-AZ behaviour: all three nodes run on one host.",
                    "disk loss or corruption: the node is SIGKILLed, its store is left intact.",
                    "client-side retry of an ambiguous commit: when the kill lands on a range "
                    "whose leaseholder was the dying node, the kernel correctly answers 503 "
                    "ambiguous_commit with retryable=true (it refuses to guess whether COMMIT "
                    "took effect, and its 0.5s idempotency-record poll is far shorter than "
                    "lease recovery). swarmbrain-demo does not implement that retry, so such a "
                    "run aborts even though nothing was lost. failure_mode names it.",
                ],
            }
        )
        artifact = write_evidence(Path(args.evidence_dir), payload)
        print()
        for assertion in assertions:
            marker = "PASS" if assertion.ok else "FAIL"
            print(f"  [{marker}] {assertion.name}: {assertion.detail}", flush=True)
        verdict = "PASSED" if ok else "FAILED"
        if outcome.failure_mode == "demo_client_did_not_retry_ambiguous_commit":
            print(
                "\n  note: the API answered 503 ambiguous_commit (retryable) on the first write\n"
                "  after the kill, and swarmbrain-demo does not retry it. Nothing was lost —\n"
                "  the counts above were read off the surviving quorum. Retake, or shoot with\n"
                "  --kill-trigger lease-wait. See docs/resilience-demo.md."
            )
        print(f"\nnode-kill resilience rehearsal {verdict}; evidence written to {artifact}")
        exit_code = 0 if ok else 1
    except RehearsalError as exc:
        payload.update({"finished_at": _now(), "error": str(exc)})
        artifact = write_evidence(Path(args.evidence_dir), payload)
        print(f"\nnode-kill resilience rehearsal FAILED: {exc}", file=sys.stderr, flush=True)
        print(f"partial evidence written to {artifact}", file=sys.stderr, flush=True)
        exit_code = 2
    finally:
        if args.keep_cluster:
            _log(f"--keep-cluster: leaving the cluster up and stores in {_redact(str(base_dir))}")
        else:
            _log("tearing the cluster down")
            cluster.stop()
            if base_dir.exists():
                if created_dir:
                    shutil.rmtree(base_dir, ignore_errors=True)
                else:
                    _empty_directory(base_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
