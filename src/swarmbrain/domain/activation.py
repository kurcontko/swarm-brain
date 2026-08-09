"""Selective memory-activation contracts.

Activation is the decision to place recalled memory in an agent's working
context.  It is deliberately distinct from retrieval (a ranked candidate may
still be dropped) and from citation (the agent may not use an activated
memory).  The request contains only identifiers and policy knobs; query text is
an ephemeral application-service argument so dumping this contract can never
persist it accidentally.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, field_validator, model_validator

from .common import (
    AgentId,
    ContractModel,
    JsonObject,
    LeaseId,
    MemoryId,
    RunId,
    TaskId,
    UUIDString,
)
from .memory import RecallBundle
from .retrieval import RetrievalPurpose


class ActivationTrigger(StrEnum):
    """Server-observed conditions that may warrant memory intervention."""

    TASK_CLAIM = "task_claim"
    DEPENDENCY_UNBLOCKED = "dependency_unblocked"
    TOOL_ERROR = "tool_error"
    REPEATED_FAILURE = "repeated_failure"
    CHECKPOINT_RESUME = "checkpoint_resume"
    EXPLICIT = "explicit"


class ActivationDecision(StrEnum):
    """The bounded intervention selected for one activation request."""

    SKIP = "skip"
    RECALL = "recall"
    DEEP_RECALL = "deep_recall"
    DEFER = "defer"


class ActivationReason(StrEnum):
    """Closed, content-free reason codes allowed across telemetry boundaries."""

    EMPTY_QUERY = "empty_activation_query"
    CONTEXT_ACTIVATED = "memory_context_activated"
    NO_RELEVANT_MEMORY = "no_relevant_memory"
    NO_RECALLABLE_CANDIDATES = "no_relevant_recallable_candidates"
    BELOW_RELEVANCE_FLOOR = "below_relevance_floor"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    RECALL_UNAVAILABLE = "memory_recall_unavailable"


def memory_activation_id(
    task_id: TaskId,
    lease_id: LeaseId,
    trigger: ActivationTrigger,
) -> str:
    """Return the stable identity of one trigger under one task lease.

    A retry of the same trigger receives the same identity, while a new lease
    or trigger cannot collide with the previous intervention.  The versioned
    prefix leaves room to change the identity recipe deliberately later.
    """

    return str(
        uuid5(
            NAMESPACE_URL,
            ":".join(
                (
                    "swarmbrain-memory-activation-v1",
                    str(task_id),
                    str(lease_id),
                    trigger.value,
                )
            ),
        )
    )


class MemoryActivationRequest(ContractModel):
    """Safe-to-record policy input for a memory intervention.

    The retrieval query is intentionally absent.  Callers pass it ephemerally
    to :meth:`MemoryActivationService.activate`; telemetry may serialize this
    model without storing task text, errors, prompts, or memory content.
    """

    task_id: TaskId
    lease_id: LeaseId
    trigger: ActivationTrigger
    purpose: RetrievalPurpose | None = None
    seed_memory_ids: tuple[MemoryId, ...] = ()
    token_budget: int = Field(default=2048, ge=1, le=131_072)
    min_score: float = Field(default=0.4, ge=0.0, le=1.0, allow_inf_nan=False)
    limit: int = Field(default=12, ge=1, le=100)

    @field_validator("seed_memory_ids")
    @classmethod
    def unique_seeds(cls, value: tuple[MemoryId, ...]) -> tuple[MemoryId, ...]:
        return tuple(dict.fromkeys(value))

    @property
    def activation_id(self) -> str:
        return memory_activation_id(self.task_id, self.lease_id, self.trigger)


class MemoryActivationTelemetry(ContractModel):
    """Content-free outcome suitable for events, metrics, and audit storage."""

    activation_id: UUIDString
    run_id: RunId
    agent_id: AgentId
    task_id: TaskId
    lease_id: LeaseId
    trigger: ActivationTrigger
    decision: ActivationDecision
    purpose: RetrievalPurpose
    reason: ActivationReason
    trace_id: UUIDString | None = None
    memory_ids: tuple[MemoryId, ...] = Field(default=(), max_length=100)
    memory_versions: dict[MemoryId, int] = Field(default_factory=dict)
    dropped_memory_ids: tuple[MemoryId, ...] = ()
    token_budget: int = Field(ge=1)
    estimated_tokens: int = Field(default=0, ge=0)
    min_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    candidate_count: int = Field(default=0, ge=0)
    truncated: bool = False

    @field_validator("memory_ids", "dropped_memory_ids")
    @classmethod
    def unique_memory_ids(cls, value: tuple[MemoryId, ...]) -> tuple[MemoryId, ...]:
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def decision_matches_selected_memory(self) -> Self:
        expected_id = memory_activation_id(self.task_id, self.lease_id, self.trigger)
        if self.activation_id != expected_id:
            raise ValueError("activation_id must match task, lease, and trigger")
        selected = bool(self.memory_ids)
        if self.decision in {ActivationDecision.RECALL, ActivationDecision.DEEP_RECALL}:
            if not selected:
                raise ValueError("recall decisions require at least one activated memory")
        elif selected:
            raise ValueError("skip and defer decisions cannot activate memory")
        if self.estimated_tokens > self.token_budget:
            raise ValueError("estimated activation tokens cannot exceed the token budget")
        if set(self.memory_ids) & set(self.dropped_memory_ids):
            raise ValueError("selected and dropped activation memory IDs must not overlap")
        if set(self.memory_versions) != set(self.memory_ids):
            raise ValueError("memory_versions must identify every selected memory exactly once")
        if any(version < 1 for version in self.memory_versions.values()):
            raise ValueError("activated memory versions must be positive")
        return self


class MemoryActivationResult(ContractModel):
    """Activation telemetry plus ephemeral context for the current agent.

    ``bundle`` and ``rendered_context`` are excluded from model serialization
    and hidden from repr.  Persisting ``model_dump()`` therefore stores only the
    explicitly content-free telemetry envelope.
    """

    telemetry: MemoryActivationTelemetry
    bundle: RecallBundle | None = Field(default=None, exclude=True, repr=False)
    rendered_context: str = Field(default="", exclude=True, repr=False)

    @model_validator(mode="after")
    def context_matches_telemetry(self) -> Self:
        bundle_ids = (
            tuple(hit.memory.memory_id for hit in self.bundle.hits)
            if self.bundle is not None
            else ()
        )
        bundle_versions = (
            {hit.memory.memory_id: hit.memory.version for hit in self.bundle.hits}
            if self.bundle is not None
            else {}
        )
        if bundle_ids != self.telemetry.memory_ids:
            raise ValueError("activation telemetry must identify the returned bundle")
        if bundle_versions != self.telemetry.memory_versions:
            raise ValueError("activation telemetry must version the returned bundle")
        selected = self.telemetry.decision in {
            ActivationDecision.RECALL,
            ActivationDecision.DEEP_RECALL,
        }
        if selected and not self.rendered_context:
            raise ValueError("activated memory requires rendered context")
        if not selected and self.rendered_context:
            raise ValueError("skip and defer results cannot carry rendered context")
        return self

    def telemetry_payload(self) -> JsonObject:
        """Return the only representation permitted at a persistence boundary."""

        return self.telemetry.model_dump(mode="json")

    @property
    def activation_id(self) -> str:
        return self.telemetry.activation_id

    @property
    def decision(self) -> ActivationDecision:
        return self.telemetry.decision


__all__ = [
    "ActivationDecision",
    "ActivationReason",
    "ActivationTrigger",
    "MemoryActivationRequest",
    "MemoryActivationResult",
    "MemoryActivationTelemetry",
    "memory_activation_id",
]
