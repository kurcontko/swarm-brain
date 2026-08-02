"""Narrow asynchronous ports for memory, evidence, and conflicts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import EvidenceId, MemoryId, SourceId
from swarmbrain.domain.conflicts import (
    ReportConflictCommand,
    ReportConflictResult,
    ResolveConflictCommand,
    ResolveConflictResult,
)
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    Evidence,
    EvidenceSource,
    RegisterEvidenceSourceCommand,
    RejectSourceCommand,
    ReviewSourceCommand,
    SourceRejectionResult,
)
from swarmbrain.domain.memory import (
    Memory,
    MemoryLineage,
    MemoryPolicyDecision,
    RecallBundle,
    RecallQuery,
    RememberCommand,
    RememberResult,
    ReviewMemoryCommand,
    ReviewMemoryResult,
)


@runtime_checkable
class MemoryOperationPolicy(Protocol):
    """Select add/update/merge/delete/noop without performing storage I/O."""

    async def decide(
        self,
        command: RememberCommand,
        current_memories: Sequence[Memory],
    ) -> MemoryPolicyDecision: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Append-only temporal memory persistence and recall."""

    async def remember(
        self,
        actor: ActorContext,
        command: RememberCommand,
        policy: MemoryOperationPolicy,
    ) -> RememberResult: ...

    async def recall(
        self,
        actor: ActorContext,
        query: RecallQuery,
    ) -> RecallBundle: ...

    async def get_lineage(
        self,
        actor: ActorContext,
        memory_id: MemoryId,
    ) -> MemoryLineage: ...

    async def reject_source(
        self,
        actor: ActorContext,
        command: RejectSourceCommand,
    ) -> SourceRejectionResult: ...


@runtime_checkable
class MemoryReviewStore(Protocol):
    """Capability-gated confirmation/refutation boundary."""

    async def review_memory(
        self,
        actor: ActorContext,
        command: ReviewMemoryCommand,
    ) -> ReviewMemoryResult: ...


@runtime_checkable
class EvidenceStore(Protocol):
    """Preserve original sources and precise evidence references."""

    async def register_source(
        self,
        actor: ActorContext,
        command: RegisterEvidenceSourceCommand,
    ) -> EvidenceSource: ...

    async def add_evidence(
        self,
        actor: ActorContext,
        command: AddEvidenceCommand,
    ) -> Evidence: ...

    async def review_source(
        self,
        actor: ActorContext,
        command: ReviewSourceCommand,
    ) -> EvidenceSource: ...

    async def get_source(
        self,
        actor: ActorContext,
        source_id: SourceId,
    ) -> EvidenceSource | None: ...

    async def get_evidence(
        self,
        actor: ActorContext,
        evidence_id: EvidenceId,
    ) -> Evidence | None: ...


@runtime_checkable
class ConflictStore(Protocol):
    """Contradiction lifecycle kept separate from ordinary recall storage."""

    async def report_conflict(
        self,
        actor: ActorContext,
        command: ReportConflictCommand,
    ) -> ReportConflictResult: ...

    async def resolve_conflict(
        self,
        actor: ActorContext,
        command: ResolveConflictCommand,
    ) -> ResolveConflictResult: ...


__all__ = [
    "ConflictStore",
    "EvidenceStore",
    "MemoryOperationPolicy",
    "MemoryReviewStore",
    "MemoryStore",
]
