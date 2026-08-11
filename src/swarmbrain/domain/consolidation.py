"""Evidence-bound contracts for asynchronous memory consolidation."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from .common import (
    AgentId,
    Confidence,
    ContractModel,
    MemoryContent,
    MemoryId,
    MutationCommand,
    RunId,
    TaskId,
)
from .evidence import EvidenceRef, Sha256
from .extraction import ProviderDescriptor
from .memory import Memory, MemoryKindValue, MemoryOperation, Visibility

ObservationKey = Annotated[
    str,
    StringConstraints(pattern=r"^m(?:0|[1-9][0-9]?)$"),
]
ConsolidationTag = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class ConsolidationActionKind(StrEnum):
    """The complete set of model-proposable consolidation actions."""

    APPEND = "append"
    SUPERSEDE = "supersede"
    LINK = "link"
    NOOP = "noop"


class ConsolidationRoute(StrEnum):
    DETERMINISTIC = "deterministic"
    PROVIDER = "provider"
    FALLBACK = "fallback"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def memory_snapshot_sha256(memory: Memory) -> str:
    """Identify every persisted field, including exact evidence references."""

    payload = _canonical_json(memory.model_dump(mode="json"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConsolidationObservation(ContractModel):
    """Bounded immutable projection frozen before provider work is queued."""

    key: ObservationKey
    memory_id: MemoryId
    memory_version: int = Field(ge=1)
    memory_sha256: Sha256
    run_id: RunId
    task_id: TaskId | None = None
    author_agent_id: AgentId
    kind: MemoryKindValue
    visibility: Visibility
    content: MemoryContent
    title: str | None = Field(default=None, max_length=500)
    tags: tuple[str, ...] = Field(default=(), max_length=64)
    confidence: Confidence
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=128)

    @classmethod
    def from_memory(cls, key: str, memory: Memory) -> ConsolidationObservation:
        return cls(
            key=key,
            memory_id=memory.memory_id,
            memory_version=memory.version,
            memory_sha256=memory_snapshot_sha256(memory),
            run_id=memory.run_id,
            task_id=memory.task_id,
            author_agent_id=memory.author_agent_id,
            kind=memory.kind,
            visibility=memory.visibility,
            content=memory.content,
            title=memory.title,
            tags=memory.tags,
            confidence=memory.confidence,
            evidence=memory.evidence,
        )


def consolidation_input_sha256(
    observations: tuple[ConsolidationObservation, ...],
    *,
    use_provider: bool,
    max_actions: int,
) -> str:
    payload = {
        "observations": [item.model_dump(mode="json") for item in observations],
        "use_provider": use_provider,
        "max_actions": max_actions,
        "policy_revision": "evidence-gated-v1",
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class ConsolidationWorkPayload(ContractModel):
    """Immutable Observer output consumed asynchronously by a Reflector."""

    observations: tuple[ConsolidationObservation, ...] = Field(min_length=2, max_length=32)
    use_provider: bool = False
    max_actions: int = Field(default=4, ge=1, le=8)
    input_sha256: Sha256
    task_id: TaskId | None = None
    policy_revision: str = Field(default="evidence-gated-v1", pattern=r"^evidence-gated-v1$")

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        keys = [item.key for item in self.observations]
        expected = [f"m{index}" for index in range(len(self.observations))]
        if keys != expected:
            raise ValueError("observation keys must be contiguous and ordered")
        memory_ids = [item.memory_id for item in self.observations]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("observed memory IDs must be unique")
        expected_sha256 = consolidation_input_sha256(
            self.observations,
            use_provider=self.use_provider,
            max_actions=self.max_actions,
        )
        if self.input_sha256 != expected_sha256:
            raise ValueError("input_sha256 must identify the exact observed snapshot")
        return self


class ScheduleConsolidationCommand(MutationCommand):
    memory_ids: tuple[MemoryId, ...] = Field(min_length=2, max_length=32)
    task_id: TaskId | None = None

    @field_validator("memory_ids")
    @classmethod
    def unique_memories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("memory_ids must be unique")
        return value


class ConsolidationProposal(ContractModel):
    """Provider-neutral proposal referencing opaque observation keys only."""

    action: ConsolidationActionKind
    kind: MemoryKindValue | None = None
    content: MemoryContent | None = None
    title: str | None = Field(default=None, max_length=500)
    tags: tuple[ConsolidationTag, ...] = Field(default=(), max_length=32)
    confidence: Confidence = 0.5
    support_keys: tuple[ObservationKey, ...] = Field(default=(), max_length=32)
    target_key: ObservationKey | None = None
    reason: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_action_shape(self) -> Self:
        if len(self.support_keys) != len(set(self.support_keys)):
            raise ValueError("support_keys must be unique")
        if self.action is ConsolidationActionKind.NOOP:
            if (
                self.kind is not None
                or self.content is not None
                or self.title is not None
                or bool(self.tags)
                or bool(self.support_keys)
                or self.target_key is not None
            ):
                raise ValueError("noop proposals cannot carry a memory mutation")
            return self
        if self.kind is None or self.content is None or not self.support_keys:
            raise ValueError("mutating proposals require kind, content, and support_keys")
        if self.action is ConsolidationActionKind.SUPERSEDE:
            if self.target_key is None or self.target_key not in self.support_keys:
                raise ValueError("supersede target must be one of the supporting observations")
        elif self.target_key is not None:
            raise ValueError("only supersede proposals may carry target_key")
        if self.action is ConsolidationActionKind.LINK and len(self.support_keys) < 2:
            raise ValueError("link proposals require at least two supporting observations")
        return self


def consolidation_plan_sha256(proposals: tuple[ConsolidationProposal, ...]) -> str:
    payload = [proposal.model_dump(mode="json") for proposal in proposals]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class ConsolidationReflection(ContractModel):
    route: ConsolidationRoute
    proposals: tuple[ConsolidationProposal, ...] = Field(max_length=8)
    input_sha256: Sha256
    plan_sha256: Sha256
    provider: ProviderDescriptor | None = None
    fallback_reason: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.plan_sha256 != consolidation_plan_sha256(self.proposals):
            raise ValueError("plan_sha256 must identify the exact proposal sequence")
        if self.route is ConsolidationRoute.PROVIDER and self.provider is None:
            raise ValueError("provider route requires provider provenance")
        if self.route is ConsolidationRoute.FALLBACK and not self.fallback_reason:
            raise ValueError("fallback route requires a bounded reason")
        return self


class ConsolidationActionResult(ContractModel):
    action: ConsolidationActionKind
    operation: MemoryOperation
    memory_id: MemoryId | None = None
    replayed: bool = False


class ConsolidationApplyResult(ContractModel):
    status: str = Field(pattern=r"^(applied|noop|stale_noop)$")
    input_sha256: Sha256
    plan_sha256: Sha256
    actions: tuple[ConsolidationActionResult, ...] = Field(max_length=8)


__all__ = [
    "ConsolidationActionKind",
    "ConsolidationActionResult",
    "ConsolidationApplyResult",
    "ConsolidationObservation",
    "ConsolidationProposal",
    "ConsolidationReflection",
    "ConsolidationRoute",
    "ConsolidationWorkPayload",
    "ObservationKey",
    "ScheduleConsolidationCommand",
    "consolidation_input_sha256",
    "consolidation_plan_sha256",
    "memory_snapshot_sha256",
]
