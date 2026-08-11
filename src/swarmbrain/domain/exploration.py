"""Bounded agentic memory read-and-expand contracts."""

from __future__ import annotations

from typing import Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from .common import ContractModel, LeaseId, MemoryId, TaskId


class ReadExpandMemoryRequest(ContractModel):
    """Read exact recallable memories and optionally follow canonical links.

    This is the second step after iterative ``recall_memory`` search. The
    request is tied to the caller's current task lease so a task-visible seed
    cannot be used as an existence or scope oracle for another task.
    """

    task_id: TaskId
    lease_id: LeaseId
    query_text: str = Field(min_length=1, max_length=8_192)
    memory_ids: tuple[MemoryId, ...] = Field(min_length=1, max_length=8)
    max_depth: int = Field(default=1, ge=0, le=2)
    max_fanout: int = Field(default=4, ge=1, le=8)
    token_budget: int = Field(default=4_096, ge=1, le=16_384)
    include_evidence: bool = True
    referenced_valid_from: AwareDatetime | None = None
    referenced_valid_to: AwareDatetime | None = None

    @field_validator("query_text")
    @classmethod
    def bounded_nonblank_query(cls, value: str) -> str:
        bounded = value.strip()
        if not bounded:
            raise ValueError("read-expand query must not be blank")
        return bounded

    @field_validator("memory_ids")
    @classmethod
    def unique_seed_ids(cls, value: tuple[MemoryId, ...]) -> tuple[MemoryId, ...]:
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def referenced_validity_is_a_bounded_interval(self) -> Self:
        has_from = self.referenced_valid_from is not None
        has_to = self.referenced_valid_to is not None
        if has_from != has_to:
            raise ValueError(
                "referenced_valid_from and referenced_valid_to must be provided together"
            )
        if (
            self.referenced_valid_from is not None
            and self.referenced_valid_to is not None
            and self.referenced_valid_to <= self.referenced_valid_from
        ):
            raise ValueError("referenced_valid_to must be later than referenced_valid_from")
        return self


class ReadExpandMemoryResult(ContractModel):
    """One bounded canonical context assembled from exact and linked memory."""

    task_id: TaskId
    lease_id: LeaseId
    context: str = Field(default="", max_length=524_288, repr=False)
    memory_ids: tuple[MemoryId, ...] = Field(default=(), max_length=100)
    memory_versions: dict[MemoryId, int] = Field(default_factory=dict)
    provenance: dict[MemoryId, tuple[str, ...]] = Field(default_factory=dict)
    dropped_memory_ids: tuple[MemoryId, ...] = Field(default=(), max_length=100)
    token_budget: int = Field(ge=1, le=16_384)
    estimated_tokens: int = Field(default=0, ge=0)
    max_depth: int = Field(ge=0, le=2)
    truncated: bool = False

    @field_validator("memory_ids", "dropped_memory_ids")
    @classmethod
    def unique_result_ids(cls, value: tuple[MemoryId, ...]) -> tuple[MemoryId, ...]:
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def content_matches_selected_ids(self) -> Self:
        selected = set(self.memory_ids)
        if set(self.memory_versions) != selected:
            raise ValueError("memory_versions must identify each returned memory exactly once")
        if set(self.provenance) != selected:
            raise ValueError("provenance must identify each returned memory exactly once")
        if any(version < 1 for version in self.memory_versions.values()):
            raise ValueError("returned memory versions must be positive")
        if selected & set(self.dropped_memory_ids):
            raise ValueError("returned and dropped memory IDs must not overlap")
        if self.estimated_tokens > self.token_budget:
            raise ValueError("read-expand output cannot exceed its token budget")
        if bool(self.context) != bool(self.memory_ids):
            raise ValueError("read-expand context and returned IDs must be present together")
        return self


__all__ = ["ReadExpandMemoryRequest", "ReadExpandMemoryResult"]
