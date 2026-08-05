from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from swarmbrain.domain.common import MemoryContent
from swarmbrain.domain.memory import (
    Memory,
    MemoryOperation,
    MemoryPolicyDecision,
    MemoryState,
    RememberCommand,
)


def memory_content_text(value: MemoryContent) -> str:
    """Return the stable textual projection used by lexical and vector adapters."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_memory_text(value: MemoryContent) -> str:
    """Stable fingerprint input; structured content stays type-distinct from text."""

    if isinstance(value, str):
        # Preserve the original v0 fingerprint for every legacy text payload.
        return re.sub(r"\s+", " ", value.strip()).casefold()
    return f"json:{memory_content_text(value)}"


def memory_text_sha256(value: MemoryContent) -> str:
    return hashlib.sha256(canonical_memory_text(value).encode("utf-8")).hexdigest()


class ConservativeMemoryPolicy:
    """Deterministic add/update/merge/noop policy adapted from Sen.

    Independent observations append by default.  Exact identity is useful for
    fingerprinting and retrieval, but is not treated as proof that two memories
    are the same event.  Only an explicit supersession target can update or
    merge an existing memory.
    """

    model_name = "swarmbrain-conservative-v0"

    async def decide(
        self,
        command: RememberCommand,
        current_memories: Sequence[Memory],
    ) -> MemoryPolicyDecision:
        target_id = command.supersedes_memory_id
        if target_id is not None:
            target = next(
                (memory for memory in current_memories if memory.memory_id == target_id),
                None,
            )
            if target is None:
                return MemoryPolicyDecision(
                    operation=MemoryOperation.NOOP,
                    reason="explicit supersession target is not current in the authenticated scope",
                    confidence=1.0,
                )

            exact = canonical_memory_text(target.content) == canonical_memory_text(command.content)
            if exact:
                return MemoryPolicyDecision(
                    operation=MemoryOperation.MERGE,
                    reason="exact canonical memory already exists; attach corroborating evidence",
                    confidence=1.0,
                    target_memory_ids=(target.memory_id,),
                )

            if self._would_poison(target, command):
                return MemoryPolicyDecision(
                    operation=MemoryOperation.NOOP,
                    reason=(
                        "confirmed memory may only be superseded by a confirmed, "
                        "evidence-backed replacement"
                    ),
                    confidence=1.0,
                )

            return MemoryPolicyDecision(
                operation=MemoryOperation.UPDATE,
                reason="caller proposed an explicit append-only supersession target",
                confidence=0.95,
                target_memory_ids=(target.memory_id,),
            )

        return MemoryPolicyDecision(
            operation=MemoryOperation.ADD,
            reason="append independent memory; no explicit supersession target",
            confidence=0.9,
        )

    @staticmethod
    def _would_poison(target: Memory, command: RememberCommand) -> bool:
        return target.state is MemoryState.CONFIRMED and (
            command.desired_state is not MemoryState.CONFIRMED or not command.evidence
        )
