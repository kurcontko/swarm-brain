"""Fail-closed validation for a future official MemoryArena execution."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .contracts import (
    OFFICIAL_DATASET,
    OFFICIAL_DATASET_CONFIGS,
    OFFICIAL_DATASET_SPLIT,
    OFFICIAL_MEMORY_SYSTEM_NAME,
    OFFICIAL_METRICS,
    OFFICIAL_REPOSITORY,
    PAPER_DECLARED_TASK_GROUPS,
    PAPER_TABLE_COMPONENT_TOTAL,
    PINNED_REPOSITORY_COMMIT,
    canonical_json,
)

PREFLIGHT_SCHEMA_VERSION = 1
PAPER_TASK_AGENT_MODEL = "gpt-5.1-mini"
OFFICIAL_CHECKOUT_SHA256 = {
    "README.md": "717264347ce69182d4b4a7809e1e8deabdbe570c6950f42f347c5522e85eb243",
    "memory/README.md": "7d854e688896e44a23621a42a7959a2150ba3a271bf823ad407f102933816504",
    "memory/client.py": "9efc1c037c9b94d4b17e5d69c4ed80bd5ad7c7f982688b56903b7e0116d62bff",
    "memory/server.py": "9afd21b346ba4b70f1f6a504260f2e4cd4707d918a9040f95e2c31fd00e97211",
    "run_math.py": "6d652ac1218e2d296a80e3fdb3281106ed1a16187ed3d648636b45f6ce895b6a",
    "run_search.py": "f58ee0eb1e7ad209f251f11ee57949a9e6077e4296b04f1177479e89041ac431",
    "run_shopping.py": "9efa3e92ff5bfc98cb9955a494c51c02099ddfa16b4842bbf1c3633424af95e4",
    "run_travel.py": "3b35937f0caba4c2b614f42d1b0cd0d4f32242973e4942bca5eda89df82e5d5f",
}
_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TASK_NAMES = frozenset({"shopping", "travel_planner", "search", "math", "phys"})


class MemoryArenaPreflightError(ValueError):
    """A preflight input is malformed rather than merely not ready."""


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    expected: Any
    actual: Any
    detail: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema_version: int
    benchmark: dict[str, Any]
    bridge_compatible: bool
    official_protocol_ready: bool
    official_execution_supported: bool
    checks: tuple[PreflightCheck, ...]
    blockers: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def run_preflight(
    checkout: Path,
    *,
    dataset_manifest: Path | None = None,
    configs: tuple[Path, ...] = (),
) -> PreflightReport:
    """Validate immutable inputs without importing or executing upstream code."""

    checks: list[PreflightCheck] = []
    raw_checkout = checkout
    try:
        checkout = _resolve_without_symlinks(raw_checkout, label="official checkout")
        checkout_path_error: str | None = None
    except MemoryArenaPreflightError as exc:
        checkout_path_error = str(exc)
        checkout = raw_checkout.absolute()
    checks.append(
        PreflightCheck(
            name="official_checkout_path_safe",
            passed=checkout_path_error is None,
            expected="existing path with no symbolic-link component",
            actual=checkout_path_error,
            detail="official checkout path and every parent component must not be symbolic links",
        )
    )
    commit = _git_value(checkout, "rev-parse", "HEAD") if checkout_path_error is None else None
    checks.append(
        PreflightCheck(
            name="official_repository_commit",
            passed=commit == PINNED_REPOSITORY_COMMIT,
            expected=PINNED_REPOSITORY_COMMIT,
            actual=commit,
            detail="exact upstream commit required",
        )
    )
    dirty = (
        _git_value(checkout, "status", "--porcelain", "--untracked-files=all")
        if checkout_path_error is None
        else None
    )
    checks.append(
        PreflightCheck(
            name="official_tracked_tree_clean",
            passed=dirty == "",
            expected="",
            actual=dirty,
            detail="tracked upstream files must be unmodified and untracked files must be absent",
        )
    )

    mismatches: dict[str, str | None] = {}
    for relative, expected in OFFICIAL_CHECKOUT_SHA256.items():
        actual: str | None = None
        if checkout_path_error is None:
            try:
                path = _safe_child(checkout, relative)
                actual = _sha256_file(path) if path.is_file() else None
            except (OSError, MemoryArenaPreflightError):
                actual = None
        if actual != expected:
            mismatches[relative] = actual
    checks.append(
        PreflightCheck(
            name="official_code_identity",
            passed=not mismatches,
            expected=OFFICIAL_CHECKOUT_SHA256,
            actual=mismatches,
            detail="client, server, runner, and README bytes are pinned",
        )
    )

    config_check = _validate_configs(configs)
    checks.append(config_check)
    dataset_check, dataset_total, reconciliation = _validate_dataset_manifest(dataset_manifest)
    checks.append(dataset_check)
    checks.append(
        PreflightCheck(
            name="paper_task_count_reconciled_by_pinned_dataset",
            passed=(
                dataset_check.passed
                and dataset_total == PAPER_DECLARED_TASK_GROUPS
                and reconciliation
            ),
            expected={
                "paper_declared_task_groups": PAPER_DECLARED_TASK_GROUPS,
                "paper_table_component_total": PAPER_TABLE_COMPONENT_TOTAL,
                "pinned_dataset_task_groups": PAPER_DECLARED_TASK_GROUPS,
                "resolution_attested": True,
            },
            actual={
                "paper_declared_task_groups": PAPER_DECLARED_TASK_GROUPS,
                "paper_table_component_total": PAPER_TABLE_COMPONENT_TOTAL,
                "pinned_dataset_task_groups": dataset_total,
                "resolution_attested": reconciliation,
            },
            detail=(
                "the paper declares 766 but its five public component counts sum to 736; "
                "an immutable dataset manifest must resolve the difference"
            ),
        )
    )

    # This repository intentionally implements only the official memory API.
    # The upstream commit calls itself a preview and does not publish a frozen
    # full-run result schema/compiler that binds all five datasets to SR/PS.
    checks.append(
        PreflightCheck(
            name="official_result_compiler_available",
            passed=False,
            expected=True,
            actual=False,
            detail=(
                "pinned upstream is preview code; no verified 766-task artifact schema and "
                "SR/PS compiler are exposed, so official score execution is unsupported"
            ),
        )
    )
    bridge_names = {
        "official_checkout_path_safe",
        "official_repository_commit",
        "official_tracked_tree_clean",
        "official_code_identity",
    }
    bridge_compatible = all(check.passed for check in checks if check.name in bridge_names)
    official_ready = all(check.passed for check in checks)
    blockers = tuple(check.detail for check in checks if not check.passed)
    return PreflightReport(
        schema_version=PREFLIGHT_SCHEMA_VERSION,
        benchmark={
            "name": "MemoryArena",
            "repository": OFFICIAL_REPOSITORY,
            "repository_commit": PINNED_REPOSITORY_COMMIT,
            "dataset": OFFICIAL_DATASET,
            "dataset_split": OFFICIAL_DATASET_SPLIT,
            "paper_declared_task_groups": PAPER_DECLARED_TASK_GROUPS,
            "paper_table_component_total": PAPER_TABLE_COMPONENT_TOTAL,
            "metrics": list(OFFICIAL_METRICS),
            "task_agent_model": PAPER_TASK_AGENT_MODEL,
            "protocol_kind": "official-paper-sr-ps",
            "custom_causal_1_2_4": False,
        },
        bridge_compatible=bridge_compatible,
        official_protocol_ready=official_ready,
        official_execution_supported=False,
        checks=tuple(checks),
        blockers=blockers,
    )


def _validate_configs(configs: tuple[Path, ...]) -> PreflightCheck:
    if not configs:
        return PreflightCheck(
            name="official_run_configs",
            passed=False,
            expected={
                "task_names": sorted(_TASK_NAMES),
                "memory_system_name": OFFICIAL_MEMORY_SYSTEM_NAME,
                "agent_model": PAPER_TASK_AGENT_MODEL,
            },
            actual=None,
            detail="five paper-protocol config overlays were not supplied",
        )
    task_names: set[str] = set()
    fingerprints: dict[str, str] = {}
    failures: list[str] = []
    for raw_path in configs:
        path = raw_path
        try:
            path = _resolve_without_symlinks(raw_path, label="official config")
            payload = _strict_json_file(path)
            if not isinstance(payload, dict):
                raise MemoryArenaPreflightError("root must be an object")
            task_name = payload.get("task_name")
            if task_name not in _TASK_NAMES:
                raise MemoryArenaPreflightError("unsupported task_name")
            if task_name in task_names:
                raise MemoryArenaPreflightError("duplicate task_name")
            task_names.add(task_name)
            agent = payload.get("agent")
            memory = payload.get("memory")
            if not isinstance(agent, dict) or not isinstance(memory, dict):
                raise MemoryArenaPreflightError("agent and memory must be objects")
            if agent.get("model_name") != PAPER_TASK_AGENT_MODEL:
                raise MemoryArenaPreflightError(
                    f"agent.model_name must be {PAPER_TASK_AGENT_MODEL!r}"
                )
            if memory.get("memory_system_name") != OFFICIAL_MEMORY_SYSTEM_NAME:
                raise MemoryArenaPreflightError(
                    f"memory_system_name must be {OFFICIAL_MEMORY_SYSTEM_NAME!r}"
                )
            url_key = {
                "search": "memory_url",
                "math": "base_url",
                "phys": "base_url",
            }.get(task_name, "server_url")
            _validate_loopback_url(memory.get(url_key))
            fingerprints[task_name] = _sha256_file(path)
        except (OSError, UnicodeError, json.JSONDecodeError, MemoryArenaPreflightError) as exc:
            failures.append(f"{path.name}:{type(exc).__name__}")
    passed = task_names == _TASK_NAMES and not failures
    return PreflightCheck(
        name="official_run_configs",
        passed=passed,
        expected={
            "task_names": sorted(_TASK_NAMES),
            "memory_system_name": OFFICIAL_MEMORY_SYSTEM_NAME,
            "agent_model": PAPER_TASK_AGENT_MODEL,
        },
        actual={
            "task_names": sorted(task_names),
            "config_sha256": fingerprints,
            "failures": failures,
        },
        detail=(
            "config overlays must cover all five tasks, use the paper task model, and route "
            "only the memory API to the local Swarm Brain bridge"
        ),
    )


def _validate_dataset_manifest(
    path: Path | None,
) -> tuple[PreflightCheck, int | None, bool]:
    expected = {
        "dataset": OFFICIAL_DATASET,
        "split": OFFICIAL_DATASET_SPLIT,
        "configs": sorted(OFFICIAL_DATASET_CONFIGS),
        "task_groups": PAPER_DECLARED_TASK_GROUPS,
        "immutable_revision": True,
        "artifacts_hashed": True,
    }
    if path is None:
        return (
            PreflightCheck(
                name="official_dataset_identity",
                passed=False,
                expected=expected,
                actual=None,
                detail="immutable local dataset manifest was not supplied",
            ),
            None,
            False,
        )
    raw_path = path
    actual: dict[str, Any] = {"manifest_path": raw_path.name}
    total: int | None = None
    reconciled = False
    try:
        path = _resolve_without_symlinks(raw_path, label="dataset manifest")
        payload = _strict_json_file(path)
        if not isinstance(payload, dict):
            raise MemoryArenaPreflightError("dataset manifest root must be an object")
        if payload.get("schema_version") != 1:
            raise MemoryArenaPreflightError("unsupported dataset manifest schema_version")
        if payload.get("dataset") != OFFICIAL_DATASET:
            raise MemoryArenaPreflightError("dataset identifier mismatch")
        if payload.get("split") != OFFICIAL_DATASET_SPLIT:
            raise MemoryArenaPreflightError("dataset split mismatch")
        revision = payload.get("revision")
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise MemoryArenaPreflightError("dataset revision must be an immutable hex revision")
        raw_configs = payload.get("configs")
        if not isinstance(raw_configs, dict) or set(raw_configs) != OFFICIAL_DATASET_CONFIGS:
            raise MemoryArenaPreflightError("dataset configs must match the five official configs")
        counts: dict[str, int] = {}
        artifact_hashes: dict[str, str] = {}
        root = path.parent
        for name in sorted(OFFICIAL_DATASET_CONFIGS):
            row = raw_configs[name]
            if not isinstance(row, dict):
                raise MemoryArenaPreflightError(f"configs.{name} must be an object")
            count = row.get("task_group_count")
            if type(count) is not int or count < 1:
                raise MemoryArenaPreflightError(f"configs.{name}.task_group_count must be positive")
            expected_hash = row.get("sha256")
            if not isinstance(expected_hash, str) or _SHA256_RE.fullmatch(expected_hash) is None:
                raise MemoryArenaPreflightError(f"configs.{name}.sha256 is invalid")
            relative = row.get("path")
            if not isinstance(relative, str) or not relative:
                raise MemoryArenaPreflightError(f"configs.{name}.path is invalid")
            artifact = _safe_child(root, relative)
            if not artifact.is_file():
                raise MemoryArenaPreflightError(f"configs.{name} artifact is missing or unsafe")
            actual_hash = _sha256_file(artifact)
            if actual_hash != expected_hash:
                raise MemoryArenaPreflightError(f"configs.{name} artifact hash mismatch")
            counts[name] = count
            artifact_hashes[name] = actual_hash
        total = sum(counts.values())
        resolution = payload.get("protocol_reconciliation")
        if not isinstance(resolution, dict):
            raise MemoryArenaPreflightError("protocol_reconciliation must be an object")
        resolution_relative = resolution.get("resolution_path")
        if not isinstance(resolution_relative, str) or not resolution_relative:
            raise MemoryArenaPreflightError(
                "protocol_reconciliation.resolution_path must be a non-empty relative path"
            )
        resolution_artifact = _safe_child(root, resolution_relative)
        if not resolution_artifact.is_file():
            raise MemoryArenaPreflightError("protocol reconciliation artifact is missing or unsafe")
        resolution_artifact_sha256 = _sha256_file(resolution_artifact)
        reconciled = (
            resolution.get("paper_declared_task_groups") == PAPER_DECLARED_TASK_GROUPS
            and resolution.get("paper_table_component_total") == PAPER_TABLE_COMPONENT_TOTAL
            and resolution.get("resolved_task_groups") == total
            and isinstance(resolution.get("resolution_sha256"), str)
            and _SHA256_RE.fullmatch(resolution["resolution_sha256"]) is not None
            and resolution["resolution_sha256"] == resolution_artifact_sha256
        )
        if not reconciled:
            raise MemoryArenaPreflightError("protocol count reconciliation is incomplete")
        if total != PAPER_DECLARED_TASK_GROUPS:
            raise MemoryArenaPreflightError("dataset does not contain the declared 766 task groups")
        actual = {
            "revision": revision,
            "manifest_sha256": _sha256_file(path),
            "task_group_counts": counts,
            "task_groups": total,
            "artifact_sha256": artifact_hashes,
            "reconciliation_artifact_sha256": resolution_artifact_sha256,
            "reconciled": reconciled,
        }
    except (OSError, UnicodeError, json.JSONDecodeError, MemoryArenaPreflightError) as exc:
        actual["failure"] = f"{type(exc).__name__}:{exc}"
        return (
            PreflightCheck(
                name="official_dataset_identity",
                passed=False,
                expected=expected,
                actual=actual,
                detail="dataset revision, five config artifacts, counts, and hashes must be pinned",
            ),
            total,
            reconciled,
        )
    return (
        PreflightCheck(
            name="official_dataset_identity",
            passed=True,
            expected=expected,
            actual=actual,
            detail="immutable dataset identity and artifact bytes verified",
        ),
        total,
        reconciled,
    )


def _git_value(checkout: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(checkout), *args),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _strict_json_file(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise MemoryArenaPreflightError("JSON input is missing or a symbolic link")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MemoryArenaPreflightError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise MemoryArenaPreflightError(f"non-finite JSON number {value!r}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )


def _resolve_without_symlinks(path: Path, *, label: str) -> Path:
    """Resolve a path only after rejecting symlinks in its lexical walk."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part in {"", "."}:
            continue
        if part == "..":
            cursor = cursor.parent
            continue
        cursor /= part
        if cursor.is_symlink():
            raise MemoryArenaPreflightError(
                f"{label} contains symbolic-link component {cursor.name!r}"
            )
    return absolute.resolve()


def _validate_loopback_url(value: Any) -> None:
    if not isinstance(value, str):
        raise MemoryArenaPreflightError("memory URL must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise MemoryArenaPreflightError("memory URL must be an uncredentialed loopback HTTP URL")


def _safe_child(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute():
        raise MemoryArenaPreflightError("artifact path must be relative")
    safe_root = _resolve_without_symlinks(root, label="artifact root")
    candidate = _resolve_without_symlinks(
        safe_root / candidate_relative,
        label="artifact path",
    )
    if not candidate.is_relative_to(safe_root):
        raise MemoryArenaPreflightError("artifact path escapes its manifest root")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_preflight_json(report: PreflightReport) -> str:
    return canonical_json(report.as_json())


__all__ = [
    "OFFICIAL_CHECKOUT_SHA256",
    "PAPER_TASK_AGENT_MODEL",
    "PREFLIGHT_SCHEMA_VERSION",
    "MemoryArenaPreflightError",
    "PreflightCheck",
    "PreflightReport",
    "canonical_preflight_json",
    "run_preflight",
]
