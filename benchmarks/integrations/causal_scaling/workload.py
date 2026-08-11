"""Immutable, separately reviewed workload manifest for causal-scaling runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import CausalScalingError, CausalTask, canonical_json, sha256_json

WORKLOAD_SCHEMA = "swarmbrain-causal-workload-v1"
REVIEWED_STATUS = "reviewed"


@dataclass(frozen=True, slots=True)
class WorkloadTaskEntry:
    task_id: str
    cluster_id: str
    public_task_sha256: str
    hidden_verifier_sha256: str
    task_fingerprint: str


@dataclass(frozen=True, slots=True)
class WorkloadManifest:
    schema: str
    workload_id: str
    workload_revision: str
    source: str
    source_revision: str
    verifier_schema: str
    review_status: str
    review_revision: str
    task_count: int
    cluster_count: int
    tasks: tuple[WorkloadTaskEntry, ...]

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def validate(self, tasks: tuple[CausalTask, ...]) -> None:
        if self.schema != WORKLOAD_SCHEMA:
            raise CausalScalingError("unsupported causal workload manifest schema")
        for name, value in (
            ("workload_id", self.workload_id),
            ("workload_revision", self.workload_revision),
            ("source", self.source),
            ("source_revision", self.source_revision),
            ("verifier_schema", self.verifier_schema),
            ("review_status", self.review_status),
            ("review_revision", self.review_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CausalScalingError(f"workload manifest {name} must be non-empty")
        if len(self.tasks) < 2:
            raise CausalScalingError("workload manifest requires at least two tasks")
        if len({entry.task_id for entry in self.tasks}) != len(self.tasks):
            raise CausalScalingError("workload manifest task IDs must be unique")
        if len({entry.cluster_id for entry in self.tasks}) < 2:
            raise CausalScalingError("workload manifest requires at least two clusters")
        if self.task_count != len(self.tasks):
            raise CausalScalingError("workload manifest task_count mismatch")
        if self.cluster_count != len({entry.cluster_id for entry in self.tasks}):
            raise CausalScalingError("workload manifest cluster_count mismatch")
        for entry in self.tasks:
            for name, digest in (
                ("public_task_sha256", entry.public_task_sha256),
                ("hidden_verifier_sha256", entry.hidden_verifier_sha256),
                ("task_fingerprint", entry.task_fingerprint),
            ):
                _validate_sha256(digest, name)
        expected = tuple(
            WorkloadTaskEntry(
                task_id=task.task_id,
                cluster_id=task.cluster_id,
                public_task_sha256=task.public_fingerprint,
                hidden_verifier_sha256=task.hidden_verifier_fingerprint,
                task_fingerprint=task.fingerprint,
            )
            for task in sorted(tasks, key=lambda item: item.task_id)
        )
        if self.tasks != expected:
            raise CausalScalingError(
                "runtime tasks do not exactly match the separately pinned workload manifest"
            )


def load_workload_manifest(value: dict[str, Any] | str | Path) -> WorkloadManifest:
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CausalScalingError(
                f"cannot read causal workload manifest: {type(exc).__name__}"
            ) from exc
    if not isinstance(payload, dict):
        raise CausalScalingError("causal workload manifest must be an object")
    try:
        task_rows = payload.get("tasks")
        if not isinstance(task_rows, list):
            raise CausalScalingError("workload manifest tasks must be a list")
        entries = tuple(WorkloadTaskEntry(**row) for row in task_rows if isinstance(row, dict))
        if len(entries) != len(task_rows):
            raise CausalScalingError("workload manifest task entries must be objects")
        manifest = WorkloadManifest(
            schema=payload.get("schema"),
            workload_id=payload.get("workload_id"),
            workload_revision=payload.get("workload_revision"),
            source=payload.get("source"),
            source_revision=payload.get("source_revision"),
            verifier_schema=payload.get("verifier_schema"),
            review_status=payload.get("review_status"),
            review_revision=payload.get("review_revision"),
            task_count=payload.get("task_count"),
            cluster_count=payload.get("cluster_count"),
            tasks=entries,
        )
    except TypeError as exc:
        raise CausalScalingError("causal workload manifest has invalid fields") from exc
    canonical_json(asdict(manifest))
    if canonical_json(payload) != canonical_json(asdict(manifest)):
        raise CausalScalingError("causal workload manifest has unknown or coerced fields")
    # Validate identity fields even before runtime tasks are available.
    if manifest.schema != WORKLOAD_SCHEMA:
        raise CausalScalingError("unsupported causal workload manifest schema")
    for name, field_value in (
        ("workload_id", manifest.workload_id),
        ("workload_revision", manifest.workload_revision),
        ("source", manifest.source),
        ("source_revision", manifest.source_revision),
        ("verifier_schema", manifest.verifier_schema),
        ("review_status", manifest.review_status),
        ("review_revision", manifest.review_revision),
    ):
        if not isinstance(field_value, str) or not field_value.strip():
            raise CausalScalingError(f"workload manifest {name} must be non-empty")
    if manifest.task_count != len(manifest.tasks):
        raise CausalScalingError("workload manifest task_count mismatch")
    clusters = {entry.cluster_id for entry in manifest.tasks}
    if manifest.cluster_count != len(clusters):
        raise CausalScalingError("workload manifest cluster_count mismatch")
    if manifest.task_count < 2 or manifest.cluster_count < 2:
        raise CausalScalingError("workload manifest requires at least two tasks and clusters")
    for entry in manifest.tasks:
        for name, digest in (
            ("public_task_sha256", entry.public_task_sha256),
            ("hidden_verifier_sha256", entry.hidden_verifier_sha256),
            ("task_fingerprint", entry.task_fingerprint),
        ):
            _validate_sha256(digest, name)
    return manifest


def _validate_sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CausalScalingError(f"workload manifest {field} must be a SHA-256 digest")


__all__ = [
    "REVIEWED_STATUS",
    "WORKLOAD_SCHEMA",
    "WorkloadManifest",
    "WorkloadTaskEntry",
    "load_workload_manifest",
]
