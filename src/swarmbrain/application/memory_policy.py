from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from swarmbrain.domain.memory import (
    Memory,
    MemoryOperation,
    MemoryPolicyDecision,
    MemoryState,
    RememberCommand,
)


def canonical_memory_text(value: str) -> str:
    """Conservative text identity used for deduplication, never semantic mutation."""

    return re.sub(r"\s+", " ", value.strip()).casefold()


def memory_text_sha256(value: str) -> str:
    return hashlib.sha256(canonical_memory_text(value).encode("utf-8")).hexdigest()


class ConservativeMemoryPolicy:
    """Deterministic add/update/merge/noop policy adapted from Sen.

    Semantic similarity may be used later to propose relationships, but only
    exact canonical identity or an explicit supersession target may authorize a
    mutating operation in v0.
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
                        "unsupported tentative memory cannot supersede confirmed "
                        "evidence-backed memory"
                    ),
                    confidence=1.0,
                )

            return MemoryPolicyDecision(
                operation=MemoryOperation.UPDATE,
                reason="caller proposed an explicit append-only supersession target",
                confidence=0.95,
                target_memory_ids=(target.memory_id,),
            )

        incoming_hash = memory_text_sha256(command.content)
        duplicate = next(
            (
                memory
                for memory in current_memories
                if memory_text_sha256(memory.content) == incoming_hash
                and memory.kind is command.kind
                and memory.visibility is command.visibility
                and memory.task_id == command.task_id
            ),
            None,
        )
        if duplicate is not None:
            return MemoryPolicyDecision(
                operation=MemoryOperation.MERGE,
                reason="same scope, kind, and exact canonical content already exist",
                confidence=0.99,
                target_memory_ids=(duplicate.memory_id,),
            )

        return MemoryPolicyDecision(
            operation=MemoryOperation.ADD,
            reason="new scoped memory with no exact duplicate or supersession target",
            confidence=0.9,
        )

    @staticmethod
    def _would_poison(target: Memory, command: RememberCommand) -> bool:
        return (
            target.state is MemoryState.CONFIRMED
            and bool(target.evidence)
            and (command.desired_state is not MemoryState.CONFIRMED or not command.evidence)
        )
