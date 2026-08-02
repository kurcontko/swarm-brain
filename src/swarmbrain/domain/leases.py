"""Exclusive task lease contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from .common import (
    AgentId,
    ContractModel,
    LeaseId,
    MutationCommand,
    RunId,
    TaskId,
    utc_now,
)


class LeaseStatus(StrEnum):
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    COMPLETED = "completed"


class TaskLease(ContractModel):
    """A versioned, exclusive right to mutate one task."""

    lease_id: LeaseId
    task_id: TaskId
    run_id: RunId
    owner_agent_id: AgentId
    status: LeaseStatus = LeaseStatus.ACTIVE
    acquired_at: AwareDatetime = Field(default_factory=utc_now)
    renewed_at: AwareDatetime | None = None
    expires_at: AwareDatetime
    released_at: AwareDatetime | None = None
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.expires_at <= self.acquired_at:
            raise ValueError("expires_at must be later than acquired_at")
        if self.renewed_at is not None and self.renewed_at < self.acquired_at:
            raise ValueError("renewed_at cannot precede acquired_at")
        if self.released_at is not None and self.released_at < self.acquired_at:
            raise ValueError("released_at cannot precede acquired_at")
        if (
            self.status in {LeaseStatus.RELEASED, LeaseStatus.COMPLETED}
            and self.released_at is None
        ):
            raise ValueError("settled leases require released_at")
        return self


class RenewLeaseCommand(MutationCommand):
    lease_id: LeaseId
    expected_version: int = Field(ge=1)
    extension_seconds: int = Field(default=120, ge=15, le=3600)


class RenewLeaseResult(ContractModel):
    lease: TaskLease
    replayed: bool = False


__all__ = ["LeaseStatus", "RenewLeaseCommand", "RenewLeaseResult", "TaskLease"]
