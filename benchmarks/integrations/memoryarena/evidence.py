"""Content-free operation evidence for the MemoryArena bridge.

The ledger deliberately cannot serialize user IDs, chunks, questions, or
rendered prompts.  A future official-result compiler can bind raw harness logs
to these rows through hashes without copying benchmark content into a public
SOTA artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Any

from .contracts import (
    OFFICIAL_METRICS,
    PAPER_DECLARED_TASK_GROUPS,
    PINNED_REPOSITORY_COMMIT,
    MemoryArenaContractError,
    canonical_json,
    require_sha256,
    sha256_json,
)

EVIDENCE_SCHEMA_VERSION = 2
EVIDENCE_PROTOCOL = "swarmbrain-memoryarena-bridge-evidence-v2"
_OPERATIONS = frozenset({"initialize", "add", "wrap_user_prompt", "cleanup"})
_FORBIDDEN_KEYS = frozenset(
    {
        "chunk",
        "content",
        "memory_context",
        "prompt",
        "question",
        "raw",
        "text",
        "user_id",
    }
)


@dataclass(frozen=True, slots=True)
class OperationEvidence:
    sequence: int
    invocation_id: str
    operation: str
    scope_sha256: str
    request_sha256: str
    request_bytes: int
    response_sha256: str | None
    success: bool
    error_code: str | None
    memory_count_before: int
    memory_count_after: int
    selected_memory_ids_sha256: str
    selected_memory_count: int
    dropped_memory_count: int
    embedding_work_completed: int
    context_estimated_tokens: int
    dense_required: bool
    dense_completed: bool
    dense_fallback: bool
    latency_ms: float

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise MemoryArenaContractError("evidence sequence must be non-negative")
        if self.operation not in _OPERATIONS:
            raise MemoryArenaContractError("unsupported evidence operation")
        for label, value in (
            ("scope_sha256", self.scope_sha256),
            ("request_sha256", self.request_sha256),
            ("selected_memory_ids_sha256", self.selected_memory_ids_sha256),
        ):
            require_sha256(value, label)
        if self.response_sha256 is not None:
            require_sha256(self.response_sha256, "response_sha256")
        if not self.invocation_id.startswith("ma_inv_"):
            raise MemoryArenaContractError("invalid MemoryArena invocation ID")
        for label, value in (
            ("request_bytes", self.request_bytes),
            ("memory_count_before", self.memory_count_before),
            ("memory_count_after", self.memory_count_after),
            ("selected_memory_count", self.selected_memory_count),
            ("dropped_memory_count", self.dropped_memory_count),
            ("embedding_work_completed", self.embedding_work_completed),
            ("context_estimated_tokens", self.context_estimated_tokens),
        ):
            if type(value) is not int or value < 0:
                raise MemoryArenaContractError(f"{label} must be a non-negative integer")
        if not isinstance(self.latency_ms, (int, float)) or self.latency_ms < 0:
            raise MemoryArenaContractError("latency_ms must be non-negative")
        if self.success != (self.error_code is None):
            raise MemoryArenaContractError("success and error_code disagree")
        for label, value in (
            ("dense_required", self.dense_required),
            ("dense_completed", self.dense_completed),
            ("dense_fallback", self.dense_fallback),
        ):
            if type(value) is not bool:
                raise MemoryArenaContractError(f"{label} must be a boolean")
        if self.dense_completed and not self.dense_required:
            raise MemoryArenaContractError("dense completion requires a dense request")
        if self.dense_fallback and (not self.dense_required or self.dense_completed):
            raise MemoryArenaContractError("dense fallback requires an incomplete dense request")


class ContentFreeEvidenceLedger:
    """Append-only bridge journal with a stable export contract."""

    def __init__(self) -> None:
        self._events: list[OperationEvidence] = []
        self._next_sequence = 0

    def next_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def append(self, event: OperationEvidence) -> None:
        if event.sequence != len(self._events):
            raise MemoryArenaContractError("evidence events must be appended in sequence")
        expected_id = invocation_id(event.sequence, event.operation, event.scope_sha256)
        if event.invocation_id != expected_id:
            raise MemoryArenaContractError("evidence invocation ID is not canonical")
        self._events.append(event)

    @property
    def events(self) -> tuple[OperationEvidence, ...]:
        return tuple(self._events)

    def export(self) -> dict[str, Any]:
        rows = [asdict(event) for event in self._events]
        payload = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "protocol": EVIDENCE_PROTOCOL,
            "benchmark": {
                "name": "MemoryArena",
                "repository_commit": PINNED_REPOSITORY_COMMIT,
                "paper_declared_task_groups": PAPER_DECLARED_TASK_GROUPS,
                "metrics": list(OFFICIAL_METRICS),
            },
            "content_policy": "hashes-counts-latencies-only",
            "event_count": len(rows),
            "events": rows,
            "events_sha256": sha256_json(rows),
        }
        assert_content_free(payload)
        return payload


def invocation_id(sequence: int, operation: str, scope_sha256: str) -> str:
    return "ma_inv_" + sha256_json(
        {"operation": operation, "scope_sha256": scope_sha256, "sequence": sequence}
    )


def empty_ids_sha256() -> str:
    return sha256_json([])


def ordered_ids_sha256(values: tuple[str, ...] | list[str]) -> str:
    return sha256_json(list(values))


def elapsed_ms(started_ns: int) -> float:
    return max(0.0, (perf_counter_ns() - started_ns) / 1_000_000.0)


def assert_content_free(value: Any, *, path: str = "$") -> None:
    """Reject common content-bearing keys anywhere in exported evidence."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_KEYS or normalized.startswith("raw_"):
                raise MemoryArenaContractError(
                    f"content-bearing evidence key is forbidden at {path}.{key}"
                )
            assert_content_free(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_content_free(item, path=f"{path}[{index}]")
    else:
        # Canonical serialization also rejects non-finite floats and exotic
        # objects.  Calling it at the root is cheap; leaf calls are harmless.
        canonical_json(value)


__all__ = [
    "ContentFreeEvidenceLedger",
    "EVIDENCE_PROTOCOL",
    "EVIDENCE_SCHEMA_VERSION",
    "OperationEvidence",
    "assert_content_free",
    "elapsed_ms",
    "empty_ids_sha256",
    "invocation_id",
    "ordered_ids_sha256",
]
