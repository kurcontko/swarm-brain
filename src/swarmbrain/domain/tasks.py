"""Task DAG, claim, checkpoint, completion, and release contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from .activation import MemoryActivationTelemetry
from .common import (
    AgentId,
    ArtifactId,
    CheckpointId,
    ContentText,
    ContractModel,
    JsonObject,
    LeaseId,
    MemoryId,
    MutationCommand,
    ProjectId,
    RepositoryId,
    RunId,
    ShortText,
    SwarmId,
    TaskId,
    TenantId,
    UUIDString,
    utc_now,
)
from .leases import TaskLease
from .memory import RecallBundle


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    CLAIMED = "claimed"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DependencyKind(StrEnum):
    BLOCKS = "blocks"
    RELATED = "related"


class TaskOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskDependency(ContractModel):
    task_id: TaskId
    depends_on_task_id: TaskId
    kind: DependencyKind = DependencyKind.BLOCKS
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def no_self_dependency(self) -> Self:
        if self.task_id == self.depends_on_task_id:
            raise ValueError("a task cannot depend on itself")
        return self


class Task(ContractModel):
    task_id: TaskId
    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    title: ShortText
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=0, ge=-1000, le=1000)
    tags: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = Field(default_factory=frozenset)
    created_by_agent_id: AgentId | None = None
    claimed_by_agent_id: AgentId | None = None
    active_lease_id: LeaseId | None = None
    available_at: AwareDatetime | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    completed_at: AwareDatetime | None = None
    version: int = Field(default=1, ge=1)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(
                dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip())
            )
        return value

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.status is TaskStatus.CLAIMED and (
            self.claimed_by_agent_id is None or self.active_lease_id is None
        ):
            raise ValueError("claimed tasks require an owner and active lease")
        if self.status is TaskStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed tasks require completed_at")
        return self


class ClaimTaskCommand(MutationCommand):
    """Claim a specific task or the next eligible task in the actor's run."""

    task_id: TaskId | None = None
    required_tags: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = Field(default_factory=frozenset)
    lease_seconds: int = Field(default=120, ge=15, le=3600)
    expected_task_version: int | None = Field(default=None, ge=1)

    @field_validator("required_tags", mode="before")
    @classmethod
    def normalize_required_tags(cls, value: object) -> object:
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(
                dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip())
            )
        return value


class TaskCheckpoint(ContractModel):
    checkpoint_id: CheckpointId
    task_id: TaskId
    lease_id: LeaseId
    run_id: RunId
    agent_id: AgentId
    sequence: int = Field(ge=1)
    summary: ContentText
    discoveries: tuple[str, ...] = ()
    completed_work: tuple[str, ...] = ()
    remaining_work: tuple[str, ...] = ()
    memory_ids: tuple[MemoryId, ...] = ()
    artifact_ids: tuple[ArtifactId, ...] = ()
    state: JsonObject = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)


class CheckpointCommand(MutationCommand):
    task_id: TaskId
    lease_id: LeaseId
    expected_task_version: int = Field(ge=1)
    expected_lease_version: int = Field(ge=1)
    summary: ContentText
    discoveries: tuple[str, ...] = ()
    completed_work: tuple[str, ...] = ()
    remaining_work: tuple[str, ...] = ()
    memory_ids: tuple[MemoryId, ...] = ()
    artifact_ids: tuple[ArtifactId, ...] = ()
    state: JsonObject = Field(default_factory=dict)


class TaskCompletion(ContractModel):
    completion_id: UUIDString
    task_id: TaskId
    lease_id: LeaseId
    run_id: RunId
    agent_id: AgentId
    outcome: TaskOutcome
    summary: ContentText
    memory_ids: tuple[MemoryId, ...] = ()
    artifact_ids: tuple[ArtifactId, ...] = ()
    completed_at: AwareDatetime = Field(default_factory=utc_now)


class CompleteTaskCommand(MutationCommand):
    task_id: TaskId
    lease_id: LeaseId
    expected_task_version: int = Field(ge=1)
    expected_lease_version: int = Field(ge=1)
    outcome: TaskOutcome = TaskOutcome.SUCCEEDED
    summary: ContentText
    memory_ids: tuple[MemoryId, ...] = ()
    artifact_ids: tuple[ArtifactId, ...] = ()


class ReleaseTaskCommand(MutationCommand):
    task_id: TaskId
    lease_id: LeaseId
    expected_task_version: int = Field(ge=1)
    expected_lease_version: int = Field(ge=1)
    reason: ContentText


class ClaimTaskResult(ContractModel):
    task: Task
    lease: TaskLease
    checkpoint: TaskCheckpoint | None = None
    unblocked_by_task_ids: tuple[TaskId, ...] = ()
    # Kept for in-process compatibility only. The structured bundle repeats
    # the query and carries metadata outside the activation token budget, so it
    # must never cross HTTP/MCP serialization boundaries.
    memory: RecallBundle | None = Field(default=None, exclude=True, repr=False)
    activation: MemoryActivationTelemetry | None = None
    activation_context: str | None = Field(default=None, max_length=524_288, repr=False)
    replayed: bool = False

    @field_validator("unblocked_by_task_ids")
    @classmethod
    def unique_unblocking_tasks(cls, value: tuple[TaskId, ...]) -> tuple[TaskId, ...]:
        return tuple(dict.fromkeys(value))

    @property
    def initial_memory(self) -> RecallBundle | None:
        return self.memory


class CheckpointResult(ContractModel):
    task: Task
    lease: TaskLease
    checkpoint: TaskCheckpoint
    replayed: bool = False


class CompletionResult(ContractModel):
    task: Task
    lease: TaskLease
    completion: TaskCompletion
    replayed: bool = False


class ReleaseResult(ContractModel):
    task: Task
    lease: TaskLease
    replayed: bool = False


__all__ = [
    "CheckpointCommand",
    "CheckpointResult",
    "ClaimTaskCommand",
    "ClaimTaskResult",
    "CompleteTaskCommand",
    "CompletionResult",
    "DependencyKind",
    "ReleaseResult",
    "ReleaseTaskCommand",
    "Task",
    "TaskCheckpoint",
    "TaskCompletion",
    "TaskDependency",
    "TaskOutcome",
    "TaskStatus",
]
