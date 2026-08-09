"""Pure hidden gate for the demo's measured memory-context ablation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

VERIFICATION_METADATA_KEY = "demo_verification"
REQUIRED_SLOTS = ("guard", "procedure")


@dataclass(frozen=True, slots=True)
class ContextVerification:
    passed: bool
    delivered_memory_ids: tuple[str, ...]
    accepted_memory_ids: tuple[str, ...]
    missing_slots: tuple[str, ...]
    answer_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "delivered_memory_ids": list(self.delivered_memory_ids),
            "accepted_memory_ids": list(self.accepted_memory_ids),
            "missing_slots": list(self.missing_slots),
            "answer_sha256": self.answer_sha256,
        }


def verify_memory_context(
    delivered_context: str | Sequence[Mapping[str, Any]],
    *,
    expected_sha256: str,
) -> ContextVerification:
    """Verify current, confirmed slot tokens found only in the supplied hits.

    There is deliberately no store or scenario argument: an empty hit sequence
    cannot recover the opaque values by another route.
    """

    hits = (
        _hits_from_activation_context(delivered_context)
        if isinstance(delivered_context, str)
        else delivered_context
    )
    delivered: list[str] = []
    values: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for hit in hits:
        raw_memory = hit.get("memory")
        if not isinstance(raw_memory, Mapping):
            continue
        memory_id = str(raw_memory.get("memory_id") or "")
        if memory_id:
            delivered.append(memory_id)
        if raw_memory.get("state") != "confirmed" or raw_memory.get("recorded_to") is not None:
            continue
        raw_metadata = raw_memory.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            continue
        marker = raw_metadata.get(VERIFICATION_METADATA_KEY)
        if not isinstance(marker, Mapping):
            continue
        slot = str(marker.get("slot") or "")
        token = str(marker.get("token") or "")
        if slot not in REQUIRED_SLOTS or not token or not memory_id:
            continue
        previous = values.get(slot)
        if previous is not None and previous != (token, memory_id):
            ambiguous.add(slot)
            continue
        values[slot] = (token, memory_id)

    missing = tuple(slot for slot in REQUIRED_SLOTS if slot not in values or slot in ambiguous)
    if missing:
        return ContextVerification(
            passed=False,
            delivered_memory_ids=tuple(dict.fromkeys(delivered)),
            accepted_memory_ids=(),
            missing_slots=missing,
            answer_sha256=None,
        )

    answer_sha256 = hashlib.sha256(
        "\n".join(values[slot][0] for slot in REQUIRED_SLOTS).encode("utf-8")
    ).hexdigest()
    return ContextVerification(
        passed=answer_sha256 == expected_sha256,
        delivered_memory_ids=tuple(dict.fromkeys(delivered)),
        accepted_memory_ids=tuple(values[slot][1] for slot in REQUIRED_SLOTS),
        missing_slots=(),
        answer_sha256=answer_sha256,
    )


def _hits_from_activation_context(context: str) -> tuple[dict[str, Any], ...]:
    """Parse only the canonical, budgeted activation representation.

    This is intentionally a tiny demo verifier, not a general memory parser.
    It proves that the hidden gate consumes exactly the string delivered over
    HTTP/MCP rather than reaching into the excluded in-process RecallBundle.
    """

    hits: list[dict[str, Any]] = []
    for block in context.split("\n\n"):
        lines = block.splitlines()
        if not lines or not lines[0].startswith("[memory:") or not lines[0].endswith("]"):
            continue
        memory_id = lines[0][len("[memory:") : -1]
        memory: dict[str, Any] = {
            "memory_id": memory_id,
            "state": None,
            "recorded_to": None,
            "metadata": {},
        }
        for line in lines[1:]:
            if line.startswith("kind="):
                fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
                memory["state"] = fields.get("state")
            elif line.startswith("recorded_to="):
                memory["recorded_to"] = line.split("=", 1)[1]
            elif line.startswith("metadata="):
                try:
                    decoded = json.loads(line.split("=", 1)[1])
                except (TypeError, ValueError):
                    continue
                if isinstance(decoded, dict):
                    memory["metadata"] = decoded
        hits.append({"memory": memory})
    return tuple(hits)


__all__ = [
    "REQUIRED_SLOTS",
    "VERIFICATION_METADATA_KEY",
    "ContextVerification",
    "verify_memory_context",
]
