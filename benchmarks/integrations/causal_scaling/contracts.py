"""Provider-neutral contracts for budget-matched causal swarm trials.

Only :class:`PublicTask` crosses the execution boundary.  Hidden verifier
payloads stay with the injected scorer, and runtime memory-use evidence comes
from a separate reader of Swarm Brain's public run-event surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol

JsonObject = dict[str, Any]
AGENT_COUNTS = (1, 2, 4)


class CausalScalingError(ValueError):
    """A causal-scaling protocol or evidence invariant was violated."""


class ExecutionKind(StrEnum):
    """Whether a run can contribute to the SOTA claim."""

    MEASURED_EXTERNAL = "measured_external"
    SMOKE_FAKE = "smoke_fake"


class FailureStage(StrEnum):
    EXECUTION = "execution"
    USAGE_VALIDATION = "usage_validation"
    OUTCOME_SCORING = "outcome_scoring"
    EVIDENCE_READ = "evidence_read"
    EVIDENCE_VALIDATION = "evidence_validation"


def canonical_json(value: Any) -> str:
    """Return the single canonical encoding used for every evidence digest."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CausalScalingError("benchmark payload is not canonical JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CausalScalingError(f"{field_name} must be a non-empty string")


def _sha256(value: str, field_name: str) -> None:
    _nonempty(value, field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CausalScalingError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PublicTask:
    """The allowlisted task view available to an agent provider."""

    task_id: str
    prompt: str
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.task_id, "task_id")
        _nonempty(self.prompt, "prompt")
        canonical_json(self.metadata)


@dataclass(frozen=True, slots=True)
class CausalTask:
    """Internal task with hidden material available only to the scorer."""

    task_id: str
    cluster_id: str
    prompt: str
    evaluation_payload: JsonObject
    public_metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.task_id, "task_id")
        _nonempty(self.cluster_id, "cluster_id")
        _nonempty(self.prompt, "prompt")
        canonical_json(self.evaluation_payload)
        canonical_json(self.public_metadata)

    def public_view(self) -> PublicTask:
        return PublicTask(
            task_id=self.task_id,
            prompt=self.prompt,
            metadata=self.public_metadata,
        )

    @property
    def public_fingerprint(self) -> str:
        return sha256_json(asdict(self.public_view()))

    @property
    def hidden_verifier_fingerprint(self) -> str:
        return sha256_json(self.evaluation_payload)

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class TotalBudget:
    """A rollout-level cap; it is divided among agents, never copied."""

    model_tokens: int
    tool_calls: int

    def __post_init__(self) -> None:
        for name, value in (
            ("model_tokens", self.model_tokens),
            ("tool_calls", self.tool_calls),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 4:
                raise CausalScalingError(f"{name} must be an integer >= 4")


@dataclass(frozen=True, slots=True)
class AgentBudget:
    agent_slot: int
    model_tokens: int
    tool_calls: int

    def __post_init__(self) -> None:
        if self.agent_slot < 1:
            raise CausalScalingError("agent_slot must be positive")
        if self.model_tokens < 0 or self.tool_calls < 0:
            raise CausalScalingError("per-agent budgets must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Pinned identity and decoding digest for one stable agent slot."""

    agent_slot: int
    provider: str
    model: str
    revision: str
    decoding_config_sha256: str

    def __post_init__(self) -> None:
        if not 1 <= self.agent_slot <= 4:
            raise CausalScalingError("model profile agent_slot must be in [1, 4]")
        for name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("revision", self.revision),
        ):
            _nonempty(value, f"model profile {name}")
        _sha256(self.decoding_config_sha256, "decoding_config_sha256")


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    """Immutable execution, model, tool, and environment identity."""

    execution_kind: ExecutionKind
    comparable: bool
    code_revision: str
    adapter_name: str
    adapter_revision: str
    environment_digest: str
    tool_catalog_sha256: str
    tool_runtime_revision: str
    model_profiles: tuple[ModelProfile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.execution_kind, ExecutionKind):
            raise CausalScalingError("execution_kind must be an ExecutionKind")
        if not isinstance(self.comparable, bool):
            raise CausalScalingError("comparable must be boolean")
        for name, value in (
            ("code_revision", self.code_revision),
            ("adapter_name", self.adapter_name),
            ("adapter_revision", self.adapter_revision),
            ("environment_digest", self.environment_digest),
            ("tool_runtime_revision", self.tool_runtime_revision),
        ):
            _nonempty(value, name)
        _sha256(self.tool_catalog_sha256, "tool_catalog_sha256")
        slots = tuple(profile.agent_slot for profile in self.model_profiles)
        if slots != (1, 2, 3, 4):
            raise CausalScalingError("model_profiles must contain ordered slots 1, 2, 3, 4")
        if self.execution_kind is ExecutionKind.SMOKE_FAKE and self.comparable:
            raise CausalScalingError("smoke/fake executions cannot be marked comparable")
        if self.execution_kind is ExecutionKind.MEASURED_EXTERNAL and not self.comparable:
            raise CausalScalingError("measured_external executions must be comparable")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    """One isolated randomized cell presented to an executor."""

    rollout_key: str
    schedule_index: int
    seed: int
    agent_count: int
    memory_enabled: bool
    task: PublicTask
    total_budget: TotalBudget
    agent_budgets: tuple[AgentBudget, ...]
    model_profiles: tuple[ModelProfile, ...]


@dataclass(frozen=True, slots=True)
class AgentUsage:
    agent_slot: int
    provider: str
    model: str
    revision: str
    input_tokens: int
    output_tokens: int
    tool_calls: int

    def __post_init__(self) -> None:
        if self.agent_slot < 1:
            raise CausalScalingError("usage agent_slot must be positive")
        for name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("revision", self.revision),
        ):
            _nonempty(value, f"usage {name}")
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("tool_calls", self.tool_calls),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CausalScalingError(f"usage {name} must be a non-negative integer")

    @property
    def model_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Provider result before hidden scoring or runtime evidence projection."""

    run_id: str
    raw_output: str
    agent_usage: tuple[AgentUsage, ...]
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        if not isinstance(self.raw_output, str):
            raise CausalScalingError("raw_output must be a string")
        canonical_json(self.metadata)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Task-verifier outcome.  Success is deliberately binary."""

    success: bool
    score: float
    details: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise CausalScalingError("score success must be boolean")
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise CausalScalingError("score must be finite and in [0, 1]")
        canonical_json(self.details)


@dataclass(frozen=True, slots=True)
class RuntimeEventEnvelope:
    """Complete projection of ``GET /v1/runs/{run_id}/events``."""

    source: str
    run_id: str
    complete: bool
    page_count: int
    events: tuple[JsonObject, ...]

    def __post_init__(self) -> None:
        _nonempty(self.source, "runtime event source")
        _nonempty(self.run_id, "runtime event run_id")
        if not isinstance(self.complete, bool):
            raise CausalScalingError("runtime event complete must be boolean")
        if not isinstance(self.page_count, int) or isinstance(self.page_count, bool):
            raise CausalScalingError("runtime event page_count must be an integer")
        if self.page_count < 1:
            raise CausalScalingError("runtime event page_count must be positive")
        canonical_json(self.events)


@dataclass(frozen=True, slots=True)
class MemoryUseProof:
    source: str
    event_stream_sha256: str
    activation_event_ids: tuple[str, ...]
    citation_event_ids: tuple[str, ...]
    matched_memory_ids: tuple[str, ...]
    matched_activation_citations: int
    memory_absence_proven: bool


@dataclass(frozen=True, slots=True)
class FailureRecord:
    stage: FailureStage
    error_type: str
    message: str


class RolloutExecutor(Protocol):
    """Provider adapter.  It never receives hidden task evaluation payloads."""

    async def execute(self, request: RolloutRequest) -> ExecutionResult: ...


class OutcomeScorer(Protocol):
    """Hidden task verifier kept outside the model execution boundary."""

    async def score(self, task: CausalTask, result: ExecutionResult) -> ScoreResult: ...


class RuntimeEvidenceReader(Protocol):
    """Load complete public Swarm Brain events for one isolated rollout."""

    async def read_events(
        self,
        request: RolloutRequest,
        result: ExecutionResult,
    ) -> RuntimeEventEnvelope: ...


__all__ = [
    "AGENT_COUNTS",
    "AgentBudget",
    "AgentUsage",
    "CausalScalingError",
    "CausalTask",
    "EvaluationProvenance",
    "ExecutionKind",
    "ExecutionResult",
    "FailureRecord",
    "FailureStage",
    "JsonObject",
    "MemoryUseProof",
    "ModelProfile",
    "OutcomeScorer",
    "PublicTask",
    "RolloutExecutor",
    "RolloutRequest",
    "RuntimeEventEnvelope",
    "RuntimeEvidenceReader",
    "ScoreResult",
    "TotalBudget",
    "canonical_json",
    "sha256_json",
]
