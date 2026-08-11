#!/usr/bin/env python3
"""Evaluate the repository's frozen SOTA evidence gates.

This command deliberately treats a missing artifact as a failed gate.  It is a
claim-readiness check, not a best-effort dashboard: architecture, unit tests,
and retrieval-only metrics cannot stand in for the end-to-end evidence named
by ``benchmarks/sota/manifest.json``. Every semantic dimension of the claim must
also be covered by a required gate; informational evidence cannot broaden it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "sota" / "manifest.json"
_COMPARISON_OPERATORS = frozenset({"eq", "ne", "gt", "gte", "lt", "lte"})
_ARTIFACT_OPERATORS = frozenset({"artifact_sha256", "artifact_bytes"})
_REPLAY_TIMEOUT_SECONDS = 60
_MANIFEST_SCHEMA_VERSION = 2
_CLAIM_SCOPE_POLICY = "every-dimension-covered-by-required-gate"
_CLAIM_DIMENSION_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
# Only deterministic report compilers which were inspected to operate entirely on
# local evidence belong here. Runtime benchmark drivers are deliberately excluded:
# readiness verification must never make provider or benchmark endpoint calls.
_SAFE_REPLAY_COMPILERS: dict[str, frozenset[str]] = {
    "scripts/build_gatemem_report.py": frozenset({"--gatemem-dir", "--evidence"}),
    "scripts/build_longmemeval_official_report.py": frozenset({"--evidence"}),
    "scripts/build_longmemeval_retrieval_report.py": frozenset({"--run"}),
    "scripts/build_longmemeval_v2_report.py": frozenset(
        {
            "--small-package",
            "--small-sidecar",
            "--medium-package",
            "--medium-sidecar",
        }
    ),
    "scripts/build_mem2act_report.py": frozenset({"--dataset-dir", "--run"}),
    "scripts/build_multi_agent_causal_report.py": frozenset({"--evidence", "--workload-manifest"}),
}
_FORBIDDEN_REPLAY_OPTIONS = frozenset({"--output", "--output-prefix", "--out-dir"})


@dataclass(frozen=True, slots=True)
class CheckResult:
    pointer: str
    operator: str
    expected: Any
    actual: Any
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class CompilerReplayResult:
    compiler: str | None
    status: str
    compiler_sha256: str | None
    artifact_sha256: str | None
    replay_sha256: str | None
    message: str


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    title: str
    required: bool
    claim_dimensions: tuple[str, ...]
    artifact: str
    status: str
    checks: tuple[CheckResult, ...]
    compiler_replay: CompilerReplayResult | None
    message: str


@dataclass(frozen=True, slots=True)
class ClaimDimensionCoverage:
    dimension: str
    required_gate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    schema_version: int
    claim: str
    claim_sha256: str
    claim_dimensions: tuple[str, ...]
    claim_coverage: tuple[ClaimDimensionCoverage, ...]
    frozen_at: str
    compiler_replay_required: bool
    ready: bool
    required_passed: int
    required_total: int
    gates: tuple[GateResult, ...]


class ManifestError(ValueError):
    """The readiness manifest is malformed or unsafe."""


class PointerMissing(KeyError):
    """A JSON pointer does not resolve in an evidence artifact."""


class StrictJsonError(ValueError):
    """A readiness artifact violates canonical JSON safety rules."""


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise StrictJsonError(f"duplicate JSON field: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise StrictJsonError(f"non-finite JSON number: {value}")


def _strict_json_loads(raw: str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_fields,
        parse_constant=_reject_nonfinite,
    )


def _claim_dimensions(raw: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{label} must be a non-empty list")
    dimensions: list[str] = []
    for value in raw:
        if not isinstance(value, str) or _CLAIM_DIMENSION_RE.fullmatch(value) is None:
            raise ManifestError(
                f"{label} entries must use lowercase kebab-case claim-dimension names"
            )
        dimensions.append(value)
    if len(dimensions) != len(set(dimensions)):
        raise ManifestError(f"{label} entries must be unique")
    return tuple(dimensions)


def _claim_scope(raw: Any, *, claim: str) -> tuple[str, ...]:
    expected_fields = {"claim_sha256", "coverage_policy", "dimensions"}
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ManifestError(
            "manifest claim_scope requires exactly claim_sha256, coverage_policy, and dimensions"
        )
    claim_sha256 = raw.get("claim_sha256")
    if not isinstance(claim_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", claim_sha256) is None:
        raise ManifestError("manifest claim_scope claim_sha256 must be a lowercase SHA-256 digest")
    try:
        observed_claim_sha256 = hashlib.sha256(claim.encode("utf-8")).hexdigest()
    except UnicodeError as exc:
        raise ManifestError("readiness manifest claim must be valid UTF-8") from exc
    if claim_sha256 != observed_claim_sha256:
        raise ManifestError("manifest claim_scope claim_sha256 does not bind the exact claim text")
    if raw.get("coverage_policy") != _CLAIM_SCOPE_POLICY:
        raise ManifestError(f"manifest claim_scope coverage_policy must be {_CLAIM_SCOPE_POLICY!r}")
    return _claim_dimensions(raw.get("dimensions"), label="manifest claim_scope dimensions")


def _validate_claim_coverage(
    gates_raw: list[Any],
    *,
    claim_dimensions: tuple[str, ...],
) -> tuple[ClaimDimensionCoverage, ...]:
    declared_dimensions = set(claim_dimensions)
    metadata: list[tuple[str, bool, tuple[str, ...]]] = []
    for raw in gates_raw:
        if not isinstance(raw, dict):
            raise ManifestError("each gate must be an object")
        gate_id = raw.get("id")
        required = raw.get("required", True)
        if not isinstance(gate_id, str) or not gate_id.strip():
            raise ManifestError("gate id must be a non-empty string")
        if not isinstance(required, bool):
            raise ManifestError(f"gate {gate_id!r} required must be boolean")
        dimensions = _claim_dimensions(
            raw.get("claim_dimensions"),
            label=f"gate {gate_id!r} claim_dimensions",
        )
        metadata.append((gate_id, required, dimensions))

    ids = [gate_id for gate_id, _, _ in metadata]
    if len(ids) != len(set(ids)):
        raise ManifestError("readiness gate ids must be unique")
    gate_dimensions = {dimension for _, _, dimensions in metadata for dimension in dimensions}
    undeclared_dimensions = sorted(gate_dimensions.difference(declared_dimensions))
    if undeclared_dimensions:
        raise ManifestError(
            "gate claim_dimensions are absent from manifest claim_scope: "
            + ", ".join(undeclared_dimensions)
        )
    claim_coverage = tuple(
        ClaimDimensionCoverage(
            dimension=dimension,
            required_gate_ids=tuple(
                gate_id
                for gate_id, required, dimensions in metadata
                if required and dimension in dimensions
            ),
        )
        for dimension in claim_dimensions
    )
    uncovered_dimensions = [
        coverage.dimension for coverage in claim_coverage if not coverage.required_gate_ids
    ]
    if uncovered_dimensions:
        raise ManifestError(
            "claim_scope dimensions must each be covered by a required gate: "
            + ", ".join(uncovered_dimensions)
        )
    return claim_coverage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete machine-readable report instead of the concise table",
    )
    return parser


def _json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ManifestError(f"JSON pointer must be empty or start with '/': {pointer!r}")
    current = payload
    for raw_segment in pointer[1:].split("/"):
        segment = raw_segment.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                index = int(segment)
            except ValueError as exc:
                raise PointerMissing(pointer) from exc
            if 0 <= index < len(current):
                current = current[index]
                continue
        raise PointerMissing(pointer)
    return current


def _ordered_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ManifestError(f"{label} must be a finite number")
    return numeric


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "exists":
        return True
    if operator == "nonempty":
        return actual is not None and hasattr(actual, "__len__") and len(actual) > 0
    if operator not in _COMPARISON_OPERATORS:
        raise ManifestError(f"unsupported check operator {operator!r}")
    if operator == "eq":
        return _strict_json_equal(actual, expected)
    if operator == "ne":
        return not _strict_json_equal(actual, expected)
    left = _ordered_number(actual, label="actual value")
    right = _ordered_number(expected, label="expected value")
    if operator == "gt":
        return left > right
    if operator == "gte":
        return left >= right
    if operator == "lt":
        return left < right
    return left <= right


def _strict_json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and left == right
    return type(left) is type(right) and left == right


def _artifact_path(repo_root: Path, value: Any) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("gate artifact must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ManifestError("gate artifact must be repository-relative")
    resolved_root = repo_root.resolve()
    candidate = resolved_root / relative
    if candidate.is_symlink():
        raise ManifestError("gate artifact cannot be a symbolic link")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ManifestError("gate artifact cannot escape the repository root")
    return relative.as_posix(), resolved


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _path_has_symlink_component(repo_root: Path, relative: Path) -> bool:
    candidate = repo_root.resolve()
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            return True
    return False


def _replay_argument_value(token: str) -> str | None:
    if token.startswith("--"):
        _, separator, value = token.partition("=")
        return value if separator else None
    if token.startswith("-"):
        return None
    return token


def _validate_replay_argument(
    token: Any,
    *,
    repo_root: Path,
    artifact_path: Path,
) -> str:
    if not isinstance(token, str) or not token or "\x00" in token:
        raise ManifestError("compiler replay arguments must be non-empty strings")
    if "://" in token:
        raise ManifestError("compiler replay arguments cannot contain endpoint URLs")
    option = token.partition("=")[0]
    if option in _FORBIDDEN_REPLAY_OPTIONS:
        raise ManifestError("compiler replay controls its output path internally")
    value = _replay_argument_value(token)
    if value is None:
        return token
    relative = Path(value)
    if relative.is_absolute():
        raise ManifestError("compiler replay arguments must not contain absolute paths")
    if ".." in relative.parts:
        raise ManifestError("compiler replay arguments cannot escape the repository root")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ManifestError("compiler replay arguments cannot escape the repository root")
    if _path_has_symlink_component(resolved_root, relative):
        raise ManifestError("compiler replay arguments cannot traverse symbolic links")
    if resolved == artifact_path.resolve():
        raise ManifestError("compiler replay cannot use the final report as input evidence")
    return token


def _compiler_replay(
    repo_root: Path,
    artifact_path: Path,
    raw: Any,
) -> CompilerReplayResult:
    if not isinstance(raw, dict):
        raise ManifestError("compiler_replay must be an object")
    if set(raw) != {"compiler", "arguments"}:
        raise ManifestError("compiler_replay requires exactly compiler and arguments")
    compiler = raw.get("compiler")
    arguments = raw.get("arguments")
    if not isinstance(compiler, str) or compiler not in _SAFE_REPLAY_COMPILERS:
        raise ManifestError("compiler_replay compiler is not in the offline allowlist")
    if not isinstance(arguments, list):
        raise ManifestError("compiler_replay arguments must be a list")
    compiler_relative, compiler_path = _artifact_path(repo_root, compiler)
    if (
        not compiler_path.is_file()
        or compiler_path.is_symlink()
        or _path_has_symlink_component(repo_root.resolve(), Path(compiler_relative))
    ):
        raise ManifestError("compiler_replay compiler is missing or unsafe")
    validated_arguments = tuple(
        _validate_replay_argument(
            argument,
            repo_root=repo_root,
            artifact_path=artifact_path,
        )
        for argument in arguments
    )
    configured_options = {argument.partition("=")[0] for argument in validated_arguments}
    missing_options = _SAFE_REPLAY_COMPILERS[compiler].difference(configured_options)
    if missing_options:
        missing = ", ".join(sorted(missing_options))
        raise ManifestError(f"compiler_replay is missing required input options: {missing}")

    compiler_sha256 = _sha256_path(compiler_path)
    artifact_sha256 = _sha256_path(artifact_path)
    with tempfile.TemporaryDirectory(prefix="swarmbrain-sota-replay-") as temporary:
        replay_path = Path(temporary) / "replayed-report.json"
        command = (
            sys.executable,
            "-B",
            str(compiler_path),
            *validated_arguments,
            "--output",
            str(replay_path),
        )
        # Provider credentials and user configuration are intentionally omitted.
        # The allowlisted compiler receives only enough environment to execute the
        # current Python checkout deterministically.
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root.resolve(),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_REPLAY_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CompilerReplayResult(
                compiler=compiler_relative,
                status="error",
                compiler_sha256=compiler_sha256,
                artifact_sha256=artifact_sha256,
                replay_sha256=None,
                message=f"offline compiler replay could not complete: {type(exc).__name__}",
            )
        if completed.returncode != 0:
            return CompilerReplayResult(
                compiler=compiler_relative,
                status="error",
                compiler_sha256=compiler_sha256,
                artifact_sha256=artifact_sha256,
                replay_sha256=None,
                message=f"offline compiler replay exited with status {completed.returncode}",
            )
        if not replay_path.is_file() or replay_path.is_symlink():
            return CompilerReplayResult(
                compiler=compiler_relative,
                status="error",
                compiler_sha256=compiler_sha256,
                artifact_sha256=artifact_sha256,
                replay_sha256=None,
                message="offline compiler replay did not emit a safe report artifact",
            )
        try:
            _strict_json_loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, StrictJsonError) as exc:
            return CompilerReplayResult(
                compiler=compiler_relative,
                status="error",
                compiler_sha256=compiler_sha256,
                artifact_sha256=artifact_sha256,
                replay_sha256=None,
                message=f"offline compiler replay emitted invalid JSON: {type(exc).__name__}",
            )
        replay_sha256 = _sha256_path(replay_path)
        passed = artifact_path.read_bytes() == replay_path.read_bytes()
        return CompilerReplayResult(
            compiler=compiler_relative,
            status="passed" if passed else "failed",
            compiler_sha256=compiler_sha256,
            artifact_sha256=artifact_sha256,
            replay_sha256=replay_sha256,
            message=(
                "current-tree compiler replay exactly reproduced the report"
                if passed
                else "current-tree compiler replay did not reproduce the report bytes"
            ),
        )


def _evaluate_check(payload: Any, raw: Any, *, repo_root: Path) -> CheckResult:
    if not isinstance(raw, dict):
        raise ManifestError("each gate check must be an object")
    pointer = raw.get("pointer")
    operator = raw.get("operator")
    if not isinstance(pointer, str) or not isinstance(operator, str):
        raise ManifestError("gate checks require string pointer and operator fields")
    expected = raw.get("expected")
    try:
        actual = _json_pointer(payload, pointer)
    except PointerMissing:
        return CheckResult(
            pointer=pointer,
            operator=operator,
            expected=expected,
            actual=None,
            passed=False,
            message="JSON pointer is missing",
        )
    if operator in _ARTIFACT_OPERATORS:
        expected_pointer = raw.get("expected_pointer")
        if not isinstance(expected_pointer, str):
            raise ManifestError(f"{operator} checks require a string expected_pointer field")
        try:
            expected = _json_pointer(payload, expected_pointer)
        except PointerMissing:
            return CheckResult(
                pointer=pointer,
                operator=operator,
                expected=None,
                actual=None,
                passed=False,
                message="artifact expected pointer is missing",
            )
        _, artifact_path = _artifact_path(repo_root, actual)
        if not artifact_path.is_file() or artifact_path.is_symlink():
            return CheckResult(
                pointer=pointer,
                operator=operator,
                expected=expected,
                actual=None,
                passed=False,
                message="bound artifact is missing or unsafe",
            )
        if operator == "artifact_sha256":
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
            actual = digest.hexdigest()
        else:
            actual = artifact_path.stat().st_size
        passed = actual == expected
        return CheckResult(
            pointer=pointer,
            operator=operator,
            expected=expected,
            actual=actual,
            passed=passed,
            message="passed" if passed else "bound artifact comparison failed",
        )
    passed = _compare(actual, operator, expected)
    return CheckResult(
        pointer=pointer,
        operator=operator,
        expected=expected,
        actual=actual,
        passed=passed,
        message="passed" if passed else "comparison failed",
    )


def _evaluate_gate(
    repo_root: Path,
    raw: Any,
    *,
    require_compiler_replay: bool,
) -> GateResult:
    if not isinstance(raw, dict):
        raise ManifestError("each gate must be an object")
    gate_id = raw.get("id")
    title = raw.get("title")
    required = raw.get("required", True)
    if not isinstance(gate_id, str) or not gate_id.strip():
        raise ManifestError("gate id must be a non-empty string")
    if not isinstance(title, str) or not title.strip():
        raise ManifestError(f"gate {gate_id!r} title must be a non-empty string")
    if not isinstance(required, bool):
        raise ManifestError(f"gate {gate_id!r} required must be boolean")
    claim_dimensions = _claim_dimensions(
        raw.get("claim_dimensions"),
        label=f"gate {gate_id!r} claim_dimensions",
    )
    artifact, path = _artifact_path(repo_root, raw.get("artifact"))
    checks_raw = raw.get("checks")
    if not isinstance(checks_raw, list) or not checks_raw:
        raise ManifestError(f"gate {gate_id!r} must define at least one check")
    if not path.is_file():
        return GateResult(
            gate_id=gate_id,
            title=title,
            required=required,
            claim_dimensions=claim_dimensions,
            artifact=artifact,
            status="missing",
            checks=(),
            compiler_replay=None,
            message=(
                "required evidence artifact is missing"
                if required
                else "informational evidence artifact is missing"
            ),
        )
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, StrictJsonError) as exc:
        return GateResult(
            gate_id=gate_id,
            title=title,
            required=required,
            claim_dimensions=claim_dimensions,
            artifact=artifact,
            status="error",
            checks=(),
            compiler_replay=None,
            message=f"evidence artifact is unreadable: {type(exc).__name__}",
        )
    checks = tuple(_evaluate_check(payload, item, repo_root=repo_root) for item in checks_raw)
    replay_raw = raw.get("compiler_replay")
    replay = _compiler_replay(repo_root, path, replay_raw) if replay_raw is not None else None
    replay_required_but_missing = require_compiler_replay and replay is None
    replay_passed = replay is None or replay.status == "passed"
    checks_passed = all(check.passed for check in checks)
    passed = checks_passed and replay_passed
    if replay_required_but_missing:
        passed = False
    if replay_required_but_missing:
        message = "compiler replay is required but not configured"
    elif replay is not None and replay.status == "error":
        message = replay.message
    elif replay is not None and replay.status == "failed":
        message = "current-tree compiler replay failed"
    elif not checks_passed:
        message = "one or more checks failed"
    elif replay is None:
        message = "all evidence checks passed"
    else:
        message = "all evidence checks and required replay passed"
    return GateResult(
        gate_id=gate_id,
        title=title,
        required=required,
        claim_dimensions=claim_dimensions,
        artifact=artifact,
        status=(
            "passed"
            if passed
            else "error"
            if replay is not None and replay.status == "error"
            else "failed"
        ),
        checks=checks,
        compiler_replay=replay,
        message=message,
    )


def evaluate_manifest(manifest_path: Path, *, repo_root: Path = REPO_ROOT) -> ReadinessReport:
    try:
        payload = _strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, StrictJsonError) as exc:
        raise ManifestError(f"cannot read readiness manifest: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("readiness manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    claim = payload.get("claim")
    claim_scope_raw = payload.get("claim_scope")
    frozen_at = payload.get("frozen_at")
    gates_raw = payload.get("gates")
    verification = payload.get("verification")
    if schema_version != _MANIFEST_SCHEMA_VERSION:
        raise ManifestError("unsupported readiness manifest schema_version")
    if not isinstance(claim, str) or not claim.strip():
        raise ManifestError("readiness manifest requires a non-empty claim")
    if not isinstance(frozen_at, str) or not frozen_at.strip():
        raise ManifestError("readiness manifest requires frozen_at")
    claim_dimensions = _claim_scope(claim_scope_raw, claim=claim)
    if not isinstance(gates_raw, list) or not gates_raw:
        raise ManifestError("readiness manifest requires at least one gate")
    if verification is None:
        require_compiler_replay = False
    else:
        if not isinstance(verification, dict) or set(verification) != {"require_compiler_replay"}:
            raise ManifestError("manifest verification requires exactly require_compiler_replay")
        require_compiler_replay = verification.get("require_compiler_replay")
        if require_compiler_replay is not True:
            raise ManifestError("manifest require_compiler_replay must be true when configured")
    claim_coverage = _validate_claim_coverage(
        gates_raw,
        claim_dimensions=claim_dimensions,
    )
    gates = tuple(
        _evaluate_gate(
            repo_root,
            gate,
            require_compiler_replay=require_compiler_replay,
        )
        for gate in gates_raw
    )
    required = tuple(gate for gate in gates if gate.required)
    passed = sum(gate.status == "passed" for gate in required)
    return ReadinessReport(
        schema_version=schema_version,
        claim=claim,
        claim_sha256=hashlib.sha256(claim.encode("utf-8")).hexdigest(),
        claim_dimensions=claim_dimensions,
        claim_coverage=claim_coverage,
        frozen_at=frozen_at,
        compiler_replay_required=require_compiler_replay,
        ready=passed == len(required),
        required_passed=passed,
        required_total=len(required),
        gates=gates,
    )


def _render_text(report: ReadinessReport) -> str:
    lines = [
        f"SOTA readiness: {report.required_passed}/{report.required_total} required gates passed",
        f"Claim: {report.claim}",
        f"Claim SHA-256: {report.claim_sha256}",
        "Claim dimensions: " + ", ".join(report.claim_dimensions),
        f"Targets frozen: {report.frozen_at}",
        "Current-tree compiler replay: "
        f"{'required' if report.compiler_replay_required else 'not required'}",
        "",
    ]
    for gate in report.gates:
        marker = {"passed": "PASS", "failed": "FAIL", "missing": "MISS", "error": "ERR"}[
            gate.status
        ]
        qualifier = "required" if gate.required else "informational"
        lines.append(f"[{marker}] {gate.gate_id}: {gate.title} ({qualifier})")
        lines.append(f"       claim dimensions: {', '.join(gate.claim_dimensions)}")
        lines.append(f"       {gate.artifact} — {gate.message}")
        if gate.compiler_replay is not None and gate.compiler_replay.status != "passed":
            lines.append(
                "       compiler replay "
                f"{gate.compiler_replay.status}: {gate.compiler_replay.message}"
            )
        for check in gate.checks:
            if check.passed:
                continue
            lines.append(
                f"       {check.pointer} {check.operator} {check.expected!r}; "
                f"actual={check.actual!r}"
            )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    try:
        report = evaluate_manifest(args.manifest)
    except ManifestError as exc:
        print(f"invalid SOTA readiness manifest: {exc}")
        return 2
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(_render_text(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
