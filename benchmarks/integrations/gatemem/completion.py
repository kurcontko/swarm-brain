"""Atomic completion marker binding one GateMem prediction/audit pair."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from .contracts import GATEMEM_COMMIT, GATEMEM_DOMAINS, GateMemContractError

COMPLETION_ARTIFACT_TYPE = "swarmbrain-gatemem-completion"
COMPLETION_SCHEMA_VERSION = 1
EXECUTION_LINEAGE_SCHEMA_VERSION = 1

_EXECUTION_MODES = frozenset({"uninterrupted", "checkpointed", "resumed", "complete_replay"})
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_IMPLEMENTATION_PATHS = frozenset(
    {
        "benchmarks/integrations/gatemem/completion.py",
        "benchmarks/integrations/gatemem/resume.py",
        "scripts/run_gatemem_external.py",
        "src/swarmbrain/application/runtime.py",
        "pyproject.toml",
        "uv.lock",
    }
)


def default_completion_path(predictions_path: str | Path) -> Path:
    predictions = Path(predictions_path)
    return predictions.with_suffix(predictions.suffix + ".completion.json")


def build_execution_lineage(
    *,
    mode: str,
    completed_prefix_episodes: int,
    completed_episodes: int,
    authenticated_state_payload_sha256: str | None,
    implementation_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Build content-free completion lineage under one strict schema."""

    resume_enabled = mode != "uninterrupted"
    resume_used = mode in {"resumed", "complete_replay"}
    lineage = {
        "schema_version": EXECUTION_LINEAGE_SCHEMA_VERSION,
        "mode": mode,
        "resume_enabled": resume_enabled,
        "resume_used": resume_used,
        "completed_prefix_episodes": completed_prefix_episodes,
        "completed_episodes": completed_episodes,
        "authenticated_state_payload_sha256": authenticated_state_payload_sha256,
        "implementation": json.loads(
            json.dumps(
                implementation_fingerprint,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        ),
    }
    _validate_execution_lineage(lineage)
    return lineage


def write_completion_manifest(
    path: str | Path,
    *,
    domain: str,
    predictions_path: str | Path,
    audit_path: str | Path,
    execution_lineage: dict[str, Any],
) -> dict[str, Any]:
    """Replace the marker only after both canonical artifacts are durable."""

    manifest_path = Path(path).resolve()
    predictions = Path(predictions_path).resolve()
    audit = Path(audit_path).resolve()
    if domain not in GATEMEM_DOMAINS:
        raise GateMemContractError(f"unsupported GateMem completion domain: {domain!r}")
    if len({manifest_path, predictions, audit}) != 3:
        raise GateMemContractError(
            "GateMem completion manifest, predictions, and audit must use distinct paths"
        )
    artifacts = {
        "predictions": _artifact_identity(predictions),
        "run_audit": _artifact_identity(audit),
    }
    lineage = json.loads(
        json.dumps(execution_lineage, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    _validate_execution_lineage(lineage)
    payload = {
        "artifact_type": COMPLETION_ARTIFACT_TYPE,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "benchmark": "GateMem",
        "gatemem_commit": GATEMEM_COMMIT,
        "domain": domain,
        "artifacts": artifacts,
        "execution_lineage": lineage,
    }
    _atomic_replace_json(manifest_path, payload)
    return payload


def validate_completion_manifest(
    path: str | Path,
    *,
    domain: str,
    predictions_path: str | Path,
    audit_path: str | Path,
    expected_completed_episodes: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    predictions = Path(predictions_path).resolve()
    audit = Path(audit_path).resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise GateMemContractError(f"GateMem completion manifest is missing: {manifest_path}")
    try:
        raw = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_fields)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateMemContractError(
            f"cannot read GateMem completion manifest: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_type",
        "schema_version",
        "benchmark",
        "gatemem_commit",
        "domain",
        "artifacts",
        "execution_lineage",
    }:
        raise GateMemContractError("GateMem completion manifest schema is malformed")
    if (
        payload.get("artifact_type") != COMPLETION_ARTIFACT_TYPE
        or payload.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or payload.get("benchmark") != "GateMem"
        or payload.get("gatemem_commit") != GATEMEM_COMMIT
        or payload.get("domain") != domain
    ):
        raise GateMemContractError("GateMem completion manifest provenance is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"predictions", "run_audit"}:
        raise GateMemContractError("GateMem completion artifact set is malformed")
    expected = {
        "predictions": _artifact_identity(predictions),
        "run_audit": _artifact_identity(audit),
    }
    if artifacts != expected:
        raise GateMemContractError(
            "GateMem completion manifest does not bind the supplied prediction/audit pair"
        )
    _validate_execution_lineage(
        payload.get("execution_lineage"),
        expected_completed_episodes=expected_completed_episodes,
    )
    return payload


def _validate_execution_lineage(
    value: Any,
    *,
    expected_completed_episodes: int | None = None,
) -> None:
    expected_keys = {
        "schema_version",
        "mode",
        "resume_enabled",
        "resume_used",
        "completed_prefix_episodes",
        "completed_episodes",
        "authenticated_state_payload_sha256",
        "implementation",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise GateMemContractError("GateMem completion execution lineage is malformed")
    mode = value.get("mode")
    if (
        value.get("schema_version") != EXECUTION_LINEAGE_SCHEMA_VERSION
        or mode not in _EXECUTION_MODES
    ):
        raise GateMemContractError("GateMem completion execution lineage is unsupported")
    prefix = _nonnegative_int(
        value.get("completed_prefix_episodes"),
        "completed_prefix_episodes",
    )
    completed = _nonnegative_int(value.get("completed_episodes"), "completed_episodes")
    if completed < 1 or prefix > completed:
        raise GateMemContractError("GateMem completion episode lineage is inconsistent")
    if expected_completed_episodes is not None and completed != expected_completed_episodes:
        raise GateMemContractError("GateMem completion episode lineage does not match coverage")

    digest = value.get("authenticated_state_payload_sha256")
    expected: dict[str, tuple[bool, bool]] = {
        "uninterrupted": (False, False),
        "checkpointed": (True, False),
        "resumed": (True, True),
        "complete_replay": (True, True),
    }
    expected_enabled, expected_used = expected[mode]
    if (
        value.get("resume_enabled") is not expected_enabled
        or value.get("resume_used") is not expected_used
    ):
        raise GateMemContractError("GateMem completion resume lineage is inconsistent")
    if mode == "uninterrupted":
        if prefix != 0 or digest is not None:
            raise GateMemContractError("GateMem uninterrupted lineage cannot name resume state")
    else:
        if not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest):
            raise GateMemContractError(
                "GateMem checkpointed lineage lacks an authenticated state digest"
            )
        if mode == "checkpointed" and prefix != 0:
            raise GateMemContractError("GateMem checkpointed lineage has a resumed prefix")
        if mode == "resumed" and not 0 < prefix < completed:
            raise GateMemContractError("GateMem resumed lineage has an invalid prefix")
        if mode == "complete_replay" and prefix != completed:
            raise GateMemContractError("GateMem complete replay lineage is incomplete")

    implementation = value.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != {"tree_sha256", "files"}:
        raise GateMemContractError("GateMem completion implementation fingerprint is malformed")
    tree_digest = implementation.get("tree_sha256")
    files = implementation.get("files")
    if (
        not isinstance(tree_digest, str)
        or not _HEX_SHA256.fullmatch(tree_digest)
        or not isinstance(files, dict)
        or not _REQUIRED_IMPLEMENTATION_PATHS.issubset(files)
        or any(
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or ".." in Path(path).parts
            or not isinstance(file_digest, str)
            or not _HEX_SHA256.fullmatch(file_digest)
            for path, file_digest in files.items()
        )
    ):
        raise GateMemContractError("GateMem completion implementation fingerprint is invalid")
    encoded_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded_files).hexdigest() != tree_digest:
        raise GateMemContractError("GateMem completion implementation tree digest is invalid")


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GateMemContractError(f"GateMem completion {name} must be a non-negative integer")
    return value


def _artifact_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GateMemContractError(f"GateMem completion artifact is missing: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise GateMemContractError(f"cannot read GateMem completion artifact: {path}") from exc
    return {
        "name": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GateMemContractError(f"GateMem completion JSON repeats field {key!r}")
        value[key] = item
    return value


def _atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise GateMemContractError("GateMem completion manifest cannot be a symbolic link")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    file_descriptor = -1
    temporary_name: str | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise GateMemContractError(
            f"cannot atomically write GateMem completion manifest: {path}"
        ) from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                Path(temporary_name).unlink()


__all__ = [
    "COMPLETION_ARTIFACT_TYPE",
    "COMPLETION_SCHEMA_VERSION",
    "EXECUTION_LINEAGE_SCHEMA_VERSION",
    "build_execution_lineage",
    "default_completion_path",
    "validate_completion_manifest",
    "write_completion_manifest",
]
