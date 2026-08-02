"""Authenticated swarm identity and agent-presence contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, StringConstraints, field_validator, model_validator

from .common import (
    AgentId,
    ContractModel,
    JsonObject,
    ProjectId,
    RepositoryId,
    RunId,
    SwarmId,
    TenantId,
    utc_now,
)

CapabilityName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]


class Capability(StrEnum):
    """Built-in authorization capabilities understood by the v0 services."""

    RUN_JOIN = "run:join"
    TASK_CLAIM = "task:claim"
    TASK_CHECKPOINT = "task:checkpoint"
    TASK_COMPLETE = "task:complete"
    TASK_RELEASE = "task:release"
    LEASE_RENEW = "lease:renew"
    MEMORY_PUBLISH = "memory:publish"
    MEMORY_RECALL = "memory:recall"
    MEMORY_CONFIRM = "memory:confirm"
    MEMORY_REFUTE = "memory:refute"
    SOURCE_INGEST = "source:ingest"
    SOURCE_REVIEW = "source:review"
    CONFLICT_REPORT = "conflict:report"
    CONFLICT_RESOLVE = "conflict:resolve"
    EVENTS_READ = "events:read"
    METRICS_READ = "metrics:read"


class AgentStatus(StrEnum):
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    REVOKED = "revoked"


class ActorContext(ContractModel):
    """Identity established by authentication, never by model tool input.

    HTTP auth or the local stdio bridge constructs this object from verified
    token claims. Application methods receive it separately from commands so
    tenant, run, repository, agent, and capability claims cannot be spoofed in
    a request body.
    """

    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    agent_id: AgentId
    harness: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    provider: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    capabilities: frozenset[CapabilityName]
    harness_version: str | None = Field(default=None, max_length=100)
    token_id: str | None = Field(default=None, max_length=255)
    authenticated_at: AwareDatetime = Field(default_factory=utc_now)
    expires_at: AwareDatetime | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def normalize_capabilities(cls, value: frozenset[str]) -> frozenset[str]:
        normalized = frozenset(item.strip().lower() for item in value)
        if "" in normalized:
            raise ValueError("capabilities cannot contain an empty value")
        return normalized

    @model_validator(mode="after")
    def validate_expiry(self) -> Self:
        if self.expires_at is not None and self.expires_at <= self.authenticated_at:
            raise ValueError("expires_at must be later than authenticated_at")
        return self

    @property
    def repo_id(self) -> str:
        """Concise compatibility spelling for logs and token claims."""

        return self.repository_id

    def has_capability(self, capability: Capability | str) -> bool:
        return str(capability).lower() in self.capabilities


class Agent(ContractModel):
    """Durable run membership derived from an authenticated actor."""

    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    agent_id: AgentId
    harness: str
    provider: str
    model: str
    capabilities: frozenset[CapabilityName]
    status: AgentStatus = AgentStatus.ACTIVE
    joined_at: AwareDatetime = Field(default_factory=utc_now)
    last_seen_at: AwareDatetime = Field(default_factory=utc_now)
    version: int = Field(default=1, ge=1)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.last_seen_at < self.joined_at:
            raise ValueError("last_seen_at cannot precede joined_at")
        return self


__all__ = ["ActorContext", "Agent", "AgentStatus", "Capability", "CapabilityName"]
