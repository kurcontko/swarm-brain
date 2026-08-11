"""Content-free observational associations between memory use and task outcomes.

These records are deliberately *silver*, not causal labels. An association
means only that an agent cited a memory it had actually received through a
fenced activation before reporting a task outcome. Retrieval and ranking do
not consume this contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, model_validator

from .common import (
    AgentId,
    ContractModel,
    LeaseId,
    MemoryId,
    ProjectId,
    RepositoryId,
    RunId,
    SwarmId,
    TaskId,
    TenantId,
    UUIDString,
)
from .tasks import TaskOutcome

MAX_OUTCOME_ASSOCIATIONS_PER_COMPLETION = 100


class OutcomeAssociationKind(StrEnum):
    """Epistemic status of a memory/outcome association."""

    OBSERVATIONAL_SILVER = "observational_silver"


def memory_outcome_association_id(
    *,
    tenant_id: TenantId,
    project_id: ProjectId,
    repository_id: RepositoryId,
    swarm_id: SwarmId,
    run_id: RunId,
    task_id: TaskId,
    lease_id: LeaseId,
    consumer_agent_id: AgentId,
    memory_id: MemoryId,
    memory_version: int,
) -> str:
    """Return a deterministic identity for one proven-use observation."""

    return str(
        uuid5(
            NAMESPACE_URL,
            ":".join(
                (
                    "swarmbrain-memory-outcome-association-v1",
                    str(tenant_id),
                    str(project_id),
                    str(repository_id),
                    str(swarm_id),
                    str(run_id),
                    str(task_id),
                    str(lease_id),
                    str(consumer_agent_id),
                    str(memory_id),
                    str(memory_version),
                )
            ),
        )
    )


class MemoryOutcomeAssociation(ContractModel):
    """A content-free, non-causal observation of proven memory use."""

    association_id: UUIDString
    kind: OutcomeAssociationKind = OutcomeAssociationKind.OBSERVATIONAL_SILVER
    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    task_id: TaskId
    lease_id: LeaseId
    consumer_agent_id: AgentId
    memory_id: MemoryId
    memory_version: int = Field(ge=1)
    outcome: TaskOutcome
    observed_at: AwareDatetime

    @model_validator(mode="after")
    def identity_matches_scope(self) -> Self:
        expected = memory_outcome_association_id(
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            repository_id=self.repository_id,
            swarm_id=self.swarm_id,
            run_id=self.run_id,
            task_id=self.task_id,
            lease_id=self.lease_id,
            consumer_agent_id=self.consumer_agent_id,
            memory_id=self.memory_id,
            memory_version=self.memory_version,
        )
        if self.association_id != expected:
            raise ValueError("association_id must match the complete observation scope")
        return self


__all__ = [
    "MAX_OUTCOME_ASSOCIATIONS_PER_COMPLETION",
    "MemoryOutcomeAssociation",
    "OutcomeAssociationKind",
    "memory_outcome_association_id",
]
