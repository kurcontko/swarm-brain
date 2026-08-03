"""Pure row/domain mapping helpers for the Cockroach memory repository."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from swarmbrain.application.memory_policy import memory_content_text
from swarmbrain.domain.conflicts import (
    Conflict,
    ConflictResolution,
    ConflictSeverity,
    ConflictStatus,
)
from swarmbrain.domain.evidence import (
    ArtifactRef,
    Evidence,
    EvidenceRef,
    EvidenceSource,
    SourceReviewState,
    SourceTrust,
)
from swarmbrain.domain.memory import (
    Memory,
    MemoryLink,
    MemoryState,
    Visibility,
)


def _uuid_text(value: object | None) -> str | None:
    return None if value is None else str(value)


def _json_object(value: object | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Cockroach JSONB object column did not decode to a mapping")
    return dict(value)


def _hex_digest(value: object | None) -> str | None:
    if value is None:
        return None
    return bytes(value).hex()


def source_from_row(row: Mapping[str, Any]) -> EvidenceSource:
    return EvidenceSource(
        source_id=str(row["id"]),
        run_id=str(row["run_id"]),
        task_id=_uuid_text(row.get("task_id")),
        kind=str(row["source_type"]),
        uri=row.get("uri"),
        content_sha256=_hex_digest(row["content_sha256"]),
        occurrence_key=str(row["occurrence_key"]),
        trust=SourceTrust(str(row["trust_label"])),
        review_state=SourceReviewState(str(row["review_state"])),
        observed_at=row.get("valid_at") or row["recorded_at"],
        recorded_at=row["recorded_at"],
        reviewed_by_agent_id=_uuid_text(row.get("reviewed_by_agent_id")),
        reviewed_at=row.get("reviewed_at"),
        rejection_reason=row.get("rejection_reason"),
        version=int(row["version"]),
        metadata=_json_object(row.get("metadata")),
    )


def evidence_from_row(row: Mapping[str, Any]) -> Evidence:
    raw_artifact = row.get("artifact")
    artifact = None if raw_artifact is None else ArtifactRef.model_validate(raw_artifact)
    return Evidence(
        evidence_id=str(row["id"]),
        source_id=str(row["source_id"]),
        kind=str(row["kind"]),
        locator=row.get("locator"),
        excerpt=row.get("excerpt"),
        content_sha256=_hex_digest(row.get("content_sha256")),
        artifact=artifact,
        recorded_at=row["recorded_at"],
        metadata=_json_object(row.get("metadata")),
    )


def memory_from_row(
    row: Mapping[str, Any],
    evidence: Sequence[EvidenceRef] = (),
) -> Memory:
    confidence = row["confidence"]
    if isinstance(confidence, Decimal):
        confidence = float(confidence)
    return Memory(
        memory_id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        project_id=str(row["project_id"]),
        repository_id=str(row["repository_id"]),
        swarm_id=str(row["swarm_id"]),
        run_id=str(row["run_id"]),
        task_id=_uuid_text(row.get("task_id")),
        author_agent_id=str(row["agent_id"]),
        kind=str(row["kind"]),
        state=MemoryState(str(row["state"])),
        visibility=Visibility(str(row["visibility"])),
        content=(
            row.get("content_json")
            if row.get("content_json") is not None
            else str(row["content"])
        ),
        title=row.get("title"),
        tags=tuple(str(item) for item in (row.get("tags") or ())),
        confidence=float(confidence),
        evidence=tuple(evidence),
        valid_from=row["valid_from"],
        valid_to=row.get("valid_to"),
        recorded_from=row["recorded_from"],
        recorded_to=row.get("recorded_to"),
        supersedes_memory_id=_uuid_text(row.get("supersedes_id")),
        superseded_by_memory_id=_uuid_text(row.get("superseded_by_id")),
        version=int(row["version"]),
        metadata=_json_object(row.get("metadata")),
    )


def link_from_row(row: Mapping[str, Any]) -> MemoryLink:
    return MemoryLink(
        link_id=str(row["id"]),
        source_memory_id=str(row["source_memory_id"]),
        target_memory_id=str(row["target_memory_id"]),
        kind=str(row["link_type"]),
        reason=row.get("reason"),
        created_at=row["created_at"],
    )


def conflict_from_parts(
    row: Mapping[str, Any],
    memory_ids: Sequence[str],
    reported_evidence: Sequence[EvidenceRef],
) -> Conflict:
    raw_resolution = row.get("resolution")
    resolution = (
        None if raw_resolution is None else ConflictResolution.model_validate(raw_resolution)
    )
    return Conflict(
        conflict_id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        project_id=str(row["project_id"]),
        repository_id=str(row["repository_id"]),
        swarm_id=str(row["swarm_id"]),
        run_id=str(row["run_id"]),
        task_id=_uuid_text(row.get("task_id")),
        memory_ids=tuple(memory_ids),
        description=str(row["description"]),
        severity=ConflictSeverity(str(row["severity"])),
        status=ConflictStatus(str(row["state"])),
        reported_by_agent_id=str(row["reported_by_agent_id"]),
        reported_evidence=tuple(reported_evidence),
        resolution=resolution,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        version=int(row["version"]),
        metadata=_json_object(row.get("metadata")),
    )


def dedup_scope(visibility: Visibility, run_id: str, task_id: str | None) -> str:
    if visibility is Visibility.REPOSITORY:
        return "repository"
    if visibility is Visibility.RUN:
        return f"run:{run_id}"
    if task_id is None:
        raise ValueError("task-visible memory requires task_id")
    return f"task:{task_id}"


def lexical_score(query: str, memory: Memory) -> tuple[float, tuple[str, ...]]:
    query_tokens = set(re.findall(r"[\w-]+", query.casefold()))
    haystack = " ".join(
        (memory.title or "", memory_content_text(memory.content), *memory.tags)
    )
    memory_tokens = set(re.findall(r"[\w-]+", haystack.casefold()))
    overlap = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
    substring = query.casefold() in haystack.casefold()
    score = min(1.0, overlap + (0.2 if substring else 0.0))
    reasons = ("lexical_overlap",) if overlap else ("scope_match",)
    return score, reasons


__all__ = [
    "conflict_from_parts",
    "dedup_scope",
    "evidence_from_row",
    "lexical_score",
    "link_from_row",
    "memory_from_row",
    "source_from_row",
]
