from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    AUTHENTICATION_REQUIRED = "authentication_required"
    INVALID_TOKEN = "invalid_token"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    NO_TASK_AVAILABLE = "no_task_available"
    INVALID_STATE = "invalid_state"
    LEASE_LOST = "lease_lost"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    MEMORY_POLICY_REJECTED = "memory_policy_rejected"
    CONFLICT_REQUIRES_EVIDENCE = "conflict_requires_evidence"
    AMBIGUOUS_COMMIT = "ambiguous_commit"


class SwarmBrainError(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.details = dict(details or {})


class AuthenticationRequired(SwarmBrainError):
    def __init__(self, message: str = "a bearer token is required") -> None:
        super().__init__(ErrorCode.AUTHENTICATION_REQUIRED, message, status_code=401)


class InvalidAuthentication(SwarmBrainError):
    def __init__(self, message: str = "the bearer token is invalid") -> None:
        super().__init__(ErrorCode.INVALID_TOKEN, message, status_code=401)


class Forbidden(SwarmBrainError):
    def __init__(self, capability: str) -> None:
        super().__init__(
            ErrorCode.FORBIDDEN,
            f"actor lacks required capability: {capability}",
            status_code=403,
            details={"required_capability": capability},
        )


class ResourceNotFound(SwarmBrainError):
    def __init__(self, resource: str, resource_id: str) -> None:
        super().__init__(
            ErrorCode.NOT_FOUND,
            f"{resource} not found: {resource_id}",
            status_code=404,
            details={"resource": resource, "resource_id": resource_id},
        )


class NoTaskAvailable(SwarmBrainError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.NO_TASK_AVAILABLE,
            "no claimable task is currently available",
            status_code=409,
            retryable=True,
        )


class InvalidState(SwarmBrainError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(
            ErrorCode.INVALID_STATE,
            message,
            status_code=409,
            details=details,
        )


class LeaseLost(SwarmBrainError):
    def __init__(self, lease_id: str, message: str = "lease is not current and owned") -> None:
        super().__init__(
            ErrorCode.LEASE_LOST,
            message,
            status_code=409,
            retryable=False,
            details={"lease_id": lease_id},
        )


class IdempotencyConflict(SwarmBrainError):
    def __init__(self, key: str) -> None:
        super().__init__(
            ErrorCode.IDEMPOTENCY_CONFLICT,
            "idempotency key was already used with a different request",
            status_code=409,
            details={"idempotency_key": key},
        )


class AmbiguousCommit(SwarmBrainError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.AMBIGUOUS_COMMIT,
            "the transaction outcome could not yet be resolved from its idempotency key",
            status_code=503,
            retryable=True,
        )


class PolicyRejected(SwarmBrainError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            ErrorCode.MEMORY_POLICY_REJECTED,
            reason,
            status_code=409,
        )


class ConflictRequiresEvidence(SwarmBrainError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.CONFLICT_REQUIRES_EVIDENCE,
            "conflict resolution requires at least one evidence reference",
            status_code=422,
        )
