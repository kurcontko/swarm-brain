"""Public, leakage-fenced contracts for the Mem2ActBench harness.

The benchmark label contains the target tool call, the evidence source IDs,
and a synthesized evolution chain.  Those fields terminate at
``Mem2ActTask``: a reader receives only ``ReaderRequest`` and a memory bridge
receives only the fixed public conversation corpus plus the query text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

JsonObject = dict[str, Any]


class Mem2ActContractError(ValueError):
    """A dataset, provider, prediction, or evaluation invariant was violated."""


@dataclass(frozen=True, slots=True)
class CorpusSession:
    session_id: str
    original_conversation_ids: tuple[str, ...]
    turns: tuple[JsonObject, ...]
    turn_count: int
    token_count: int

    def public_view(self) -> PublicConversationSession:
        """Strip construction provenance before crossing the memory boundary."""

        return PublicConversationSession(
            session_id=self.session_id,
            turns=tuple(
                {key: value for key, value in turn.items() if key != "source_id"}
                for turn in self.turns
            ),
            turn_count=self.turn_count,
            token_count=self.token_count,
        )


@dataclass(frozen=True, slots=True)
class PublicConversationSession:
    """Published conversation only; no QA or construction provenance IDs."""

    session_id: str
    turns: tuple[JsonObject, ...]
    turn_count: int
    token_count: int


@dataclass(frozen=True, slots=True)
class OracleMemory:
    """Gold evidence exposed only in the explicitly named oracle arm."""

    attribute: str
    fact: str
    source_text: str

    def render(self) -> str:
        parts = [f"Memory attribute: {self.attribute}", f"Memory fact: {self.fact}"]
        if self.source_text:
            parts.append(f"Supporting utterance: {self.source_text}")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class Mem2ActTask:
    """Internal label-bearing task; never pass this object to a reader."""

    qa_id: str
    query: str
    source_conversation_ids: tuple[str, ...]
    oracle_memories: tuple[OracleMemory, ...]
    gold_tool_name: str
    gold_arguments: JsonObject
    target_tool_schema: JsonObject
    complexity_level: str


@dataclass(frozen=True, slots=True)
class DatasetFingerprint:
    repo_commit: str
    files_sha256: dict[str, str]
    task_count: int
    session_count: int
    unresolved_source_ids: tuple[str, ...]
    known_data_repairs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Mem2ActDataset:
    tasks: tuple[Mem2ActTask, ...]
    sessions: tuple[CorpusSession, ...]
    tool_catalog: tuple[JsonObject, ...]
    tool_catalog_sha256: str
    fingerprint: DatasetFingerprint


@dataclass(frozen=True, slots=True)
class IngestionResult:
    memory_count: int
    latency_ms: float
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    memory_id: str
    content: str
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    memories: tuple[RetrievedMemory, ...]
    latency_ms: float
    total_candidates: int
    truncated: bool
    metadata: JsonObject = field(default_factory=dict)


class MemoryBridge(Protocol):
    """Injectable boundary around a public Swarm Brain runtime or HTTP client."""

    async def ingest(self, sessions: tuple[PublicConversationSession, ...]) -> IngestionResult: ...

    async def retrieve(
        self,
        query: str,
        *,
        limit: int,
        token_budget: int | None,
    ) -> RetrievalResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReaderRequest:
    """The complete allowlisted view visible to a tool-selection model.

    Deliberately absent: QA ID, arm name, source IDs, evolution metadata,
    target arguments, and gold grounding data.  ``target_tool_given`` receives
    exactly the published target schema/name, matching paper section 4.1;
    ``full_catalog`` receives the same complete catalog for every task.
    """

    condition: str
    query: str
    memory_contexts: tuple[str, ...]
    tool_catalog: tuple[JsonObject, ...]


@dataclass(frozen=True, slots=True)
class ReaderResult:
    """Raw provider response and accounting; parsing belongs to the harness."""

    raw_prediction: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.raw_prediction, str):
            raise Mem2ActContractError("reader raw_prediction must be a string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise Mem2ActContractError("reader model must be a non-empty string")
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise Mem2ActContractError(f"reader {name} must be a non-negative integer")
        if self.latency_ms is not None and (
            not isinstance(self.latency_ms, (int, float))
            or isinstance(self.latency_ms, bool)
            or self.latency_ms < 0
        ):
            raise Mem2ActContractError("reader latency_ms must be non-negative")


class ToolSelectionReader(Protocol):
    async def select_tool(self, request: ReaderRequest) -> ReaderResult: ...


@dataclass(frozen=True, slots=True)
class ToolPrediction:
    name: str
    arguments: JsonObject


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    tool_correct: int
    exact_tool_and_arguments: int
    correct_slots: int
    gold_slots: int
    predicted_slots: int
    parameter_true_positives: int
    parameter_false_positives: int
    parameter_false_negatives: int
    slot_accuracy: float
    parameter_precision: float
    parameter_recall: float
    parameter_f1: float


@dataclass(frozen=True, slots=True)
class FailureRecord:
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    qa_id: str
    condition: str
    arm: str
    query: str
    complexity_level: str
    memory_contexts: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    retrieved_scores: tuple[float, ...]
    retrieval_reasons: tuple[tuple[str, ...], ...]
    retrieval_total_candidates: int
    retrieval_truncated: bool
    retrieval_latency_ms: float
    reader_wall_latency_ms: float
    reader_reported_latency_ms: float | None
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reader_model: str | None
    reader_metadata: JsonObject
    raw_prediction: str | None
    parsed_prediction: ToolPrediction | None
    gold_tool_name: str
    gold_arguments: JsonObject
    metrics: TaskMetrics
    failure: FailureRecord | None
    tool_catalog_sha256: str


__all__ = [
    "CorpusSession",
    "DatasetFingerprint",
    "FailureRecord",
    "IngestionResult",
    "JsonObject",
    "Mem2ActContractError",
    "Mem2ActDataset",
    "Mem2ActTask",
    "MemoryBridge",
    "OracleMemory",
    "PredictionRecord",
    "PublicConversationSession",
    "ReaderRequest",
    "ReaderResult",
    "RetrievalResult",
    "RetrievedMemory",
    "TaskMetrics",
    "ToolPrediction",
    "ToolSelectionReader",
]
