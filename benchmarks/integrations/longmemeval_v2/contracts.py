"""Strict, query-private contracts for the LongMemEval-V2 Swarm adapter.

The official harness gives a memory backend only the public question text, an
optional image path, and a random run-local invocation handle.  Dataset labels
and the stable question ID deliberately do not cross this boundary.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

PINNED_REPOSITORY_COMMIT = "ef67f10aacd9080c75aeb2dd527a0af25dc26f1b"
EXPECTED_READER_MODEL = "Qwen/Qwen3.5-9B"
EXPECTED_JUDGE_MODEL = "gpt-5.2"
EXPECTED_QUESTIONS = 451
SIDECAR_SCHEMA_VERSION = 3
ADAPTER_REVISION = "swarmbrain-longmemeval-v2-v4"
MEMORY_TYPE = "swarmbrain"
METHOD = "swarmbrain"

SOTA_EMBEDDING_PROVIDER = "OpenAICompatibleEmbeddingProvider"
SOTA_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
SOTA_EMBEDDING_DIMENSIONS = 4_096
SOTA_EMBEDDING_QUERY_INSTRUCTION = (
    "Given a question about past agent trajectories, retrieve relevant memory entries "
    "that help answer it."
)
SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256 = hashlib.sha256(
    SOTA_EMBEDDING_QUERY_INSTRUCTION.encode("utf-8")
).hexdigest()

TRACE_DIGEST_METADATA_KEY = "swarmbrain_operation_trace_sha256"
RAW_TRACE_DIGEST_METADATA_KEY = "swarmbrain_raw_operation_trace_sha256"

JsonObject = dict[str, Any]

_ENVIRONMENT_VARIABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_URL_USERINFO_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/@\s]*:[^/@\s]*@")
_URL_SECRET_QUERY_RE = re.compile(
    r"(?i)[?&](?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)=[^&#\s]+"
)
_INLINE_AUTH_RE = re.compile(r"(?i)^\s*(?:bearer|basic)\s+\S+")
_COMMON_SECRET_PREFIX_RE = re.compile(
    r"(?i)^(?:sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|xox[baprs]-\S+)"
)


class LongMemEvalV2AdapterError(ValueError):
    """An adapter, official-harness, or evidence invariant was violated."""


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongMemEvalV2AdapterError(f"{label} must be a non-empty string")
    return value.strip()


def _unique_ids(value: Any, *, label: str, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise LongMemEvalV2AdapterError(f"{label} must be a sequence")
    identifiers = tuple(_required_text(item, f"{label}[]") for item in value)
    if len(identifiers) > maximum:
        raise LongMemEvalV2AdapterError(f"{label} cannot contain more than {maximum} IDs")
    if len(set(identifiers)) != len(identifiers):
        raise LongMemEvalV2AdapterError(f"{label} must contain unique IDs")
    return identifiers


def _bridge_param_key_parts(key: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", separated)
    return tuple(part for part in re.split(r"[^a-z0-9]+", separated.lower()) if part)


def _secret_bearing_key(parts: tuple[str, ...]) -> bool:
    tokens = set(parts)
    compact = "".join(parts)
    if any(
        marker in compact
        for marker in (
            "apikey",
            "accesstoken",
            "authtoken",
            "bearertoken",
            "clientsecret",
            "privatekey",
        )
    ):
        return True
    if tokens & {
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
    }:
        return True
    if {"api", "key"}.issubset(tokens):
        return True
    if {"private", "key"}.issubset(tokens):
        return True
    return "token" in tokens and (
        len(tokens) == 1 or bool(tokens & {"access", "auth", "bearer", "refresh", "session"})
    )


def _secret_env_reference(parts: tuple[str, ...]) -> bool:
    if not parts:
        return False
    if parts[-1] == "env":
        return _secret_bearing_key(parts[:-1])
    if len(parts) >= 2 and parts[-2:] in {("env", "var"), ("environment", "variable")}:
        return _secret_bearing_key(parts[:-2])
    return False


def _looks_like_inline_secret(value: str) -> bool:
    return bool(
        _INLINE_AUTH_RE.match(value)
        or _COMMON_SECRET_PREFIX_RE.match(value)
        or _URL_USERINFO_RE.match(value)
        or _URL_SECRET_QUERY_RE.search(value)
        or "-----BEGIN PRIVATE KEY-----" in value
        or "-----BEGIN RSA PRIVATE KEY-----" in value
        or "-----BEGIN OPENSSH PRIVATE KEY-----" in value
    )


def _validate_bridge_param(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise LongMemEvalV2AdapterError(f"{path} keys must be non-empty strings")
            parts = _bridge_param_key_parts(key)
            item_path = f"{path}.{key}"
            if _secret_env_reference(parts):
                if not isinstance(item, str) or not _ENVIRONMENT_VARIABLE_RE.fullmatch(item):
                    raise LongMemEvalV2AdapterError(
                        f"{item_path} must name an environment variable"
                    )
                continue
            if _secret_bearing_key(parts):
                raise LongMemEvalV2AdapterError(
                    f"{item_path} cannot persist secret material; use a *_env key"
                )
            _validate_bridge_param(item, path=item_path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_bridge_param(item, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LongMemEvalV2AdapterError(f"{path} must be finite JSON data")
        return
    if isinstance(value, str):
        if _looks_like_inline_secret(value):
            raise LongMemEvalV2AdapterError(
                f"{path} appears to contain inline secret material; use an environment variable"
            )
        return
    raise LongMemEvalV2AdapterError(f"{path} must contain only JSON-compatible values")


@dataclass(frozen=True, slots=True)
class RecallMemoryResult:
    """Content-free search result from Swarm Brain's ``recall_memory``."""

    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_ids",
            _unique_ids(self.memory_ids, label="recall_memory.memory_ids", maximum=100),
        )


@dataclass(frozen=True, slots=True)
class ReadExpandMemoryResult:
    """Exact bounded context from Swarm Brain's ``read_expand_memory``."""

    memory_ids: tuple[str, ...]
    context: str

    def __post_init__(self) -> None:
        identifiers = _unique_ids(
            self.memory_ids,
            label="read_expand_memory.memory_ids",
            maximum=100,
        )
        context = _required_text(self.context, "read_expand_memory.context")
        if not identifiers:
            raise LongMemEvalV2AdapterError(
                "read_expand_memory must return at least one canonical memory ID"
            )
        object.__setattr__(self, "memory_ids", identifiers)
        object.__setattr__(self, "context", context)


class SwarmOperationBridge(Protocol):
    """Synchronous seam around canonical Swarm memory operations.

    Implementations may wrap a local runtime or an authenticated HTTP client.
    They must not receive question IDs, gold answers, question types, haystack
    labels, or evaluator configuration.
    """

    def insert_trajectory(self, trajectory: Mapping[str, Any]) -> None: ...

    def recall_memory(self, query: str, *, limit: int) -> RecallMemoryResult: ...

    def read_expand_memory(
        self,
        query: str,
        *,
        memory_ids: tuple[str, ...],
        max_depth: int,
        max_fanout: int,
        token_budget: int,
    ) -> ReadExpandMemoryResult: ...

    def embedding_evidence(self) -> EmbeddingRuntimeEvidence:
        """Return content-free embedding proof after successful read-expand."""

        ...

    def close(self) -> None:
        """Release task leases, runtime resources, transports, and worker threads.

        The official harness has no lifecycle hook, so the adapter invokes this
        method explicitly. Implementations must be synchronous and idempotent.
        """

        ...


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Protocol-affecting backend settings persisted in official run inputs."""

    tier: str
    operating_point: str
    dataset_revision: str
    dataset_manifest_sha256: str
    bridge_factory: str
    bridge_params: JsonObject
    recall_limit: int = 8
    max_depth: int = 1
    max_fanout: int = 4
    token_budget: int = 16_384

    def __post_init__(self) -> None:
        if self.tier not in {"small", "medium"}:
            raise LongMemEvalV2AdapterError("tier must be 'small' or 'medium'")
        for name, value in (
            ("operating_point", self.operating_point),
            ("dataset_revision", self.dataset_revision),
            ("bridge_factory", self.bridge_factory),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        manifest = _required_text(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        if len(manifest) != 64 or any(
            character not in "0123456789abcdef" for character in manifest
        ):
            raise LongMemEvalV2AdapterError(
                "dataset_manifest_sha256 must be a lowercase SHA-256 digest"
            )
        object.__setattr__(self, "dataset_manifest_sha256", manifest)
        if not isinstance(self.bridge_params, dict) or not all(
            isinstance(key, str) and key for key in self.bridge_params
        ):
            raise LongMemEvalV2AdapterError("bridge_params must be a JSON object")
        _validate_bridge_param(self.bridge_params, path="bridge_params")
        if (
            not isinstance(self.recall_limit, int)
            or isinstance(self.recall_limit, bool)
            or not 1 <= self.recall_limit <= 100
        ):
            raise LongMemEvalV2AdapterError("recall_limit must be an integer in [1, 100]")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or not 1 <= self.max_depth <= 2
        ):
            raise LongMemEvalV2AdapterError("max_depth must be an integer in [1, 2]")
        if (
            not isinstance(self.max_fanout, int)
            or isinstance(self.max_fanout, bool)
            or not 1 <= self.max_fanout <= 8
        ):
            raise LongMemEvalV2AdapterError("max_fanout must be an integer in [1, 8]")
        if (
            not isinstance(self.token_budget, int)
            or isinstance(self.token_budget, bool)
            or not 1 <= self.token_budget <= 16_384
        ):
            raise LongMemEvalV2AdapterError("token_budget must be an integer in [1, 16384]")

    def memory_params(self) -> JsonObject:
        return {
            "adapter_revision": ADAPTER_REVISION,
            "benchmark_repository_commit": PINNED_REPOSITORY_COMMIT,
            "reader_model": EXPECTED_READER_MODEL,
            "judge_model": EXPECTED_JUDGE_MODEL,
            "tier": self.tier,
            "operating_point": self.operating_point,
            "dataset_revision": self.dataset_revision,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "bridge_factory": self.bridge_factory,
            "bridge_params": dict(self.bridge_params),
            "recall_limit": self.recall_limit,
            "max_depth": self.max_depth,
            "max_fanout": self.max_fanout,
            "token_budget": self.token_budget,
        }

    @classmethod
    def from_memory_params(cls, value: Any) -> AdapterConfig:
        if not isinstance(value, dict):
            raise LongMemEvalV2AdapterError("memory_params must be an object")
        expected_fields = {
            "adapter_revision",
            "benchmark_repository_commit",
            "reader_model",
            "judge_model",
            "tier",
            "operating_point",
            "dataset_revision",
            "dataset_manifest_sha256",
            "bridge_factory",
            "bridge_params",
            "recall_limit",
            "max_depth",
            "max_fanout",
            "token_budget",
        }
        if set(value) != expected_fields:
            raise LongMemEvalV2AdapterError(
                "memory_params fields differ from the pinned adapter protocol"
            )
        exact = {
            "adapter_revision": ADAPTER_REVISION,
            "benchmark_repository_commit": PINNED_REPOSITORY_COMMIT,
            "reader_model": EXPECTED_READER_MODEL,
            "judge_model": EXPECTED_JUDGE_MODEL,
        }
        for field, expected in exact.items():
            if value.get(field) != expected:
                raise LongMemEvalV2AdapterError(
                    f"memory_params.{field} must equal pinned value {expected!r}"
                )
        return cls(
            tier=value["tier"],
            operating_point=value["operating_point"],
            dataset_revision=value["dataset_revision"],
            dataset_manifest_sha256=value["dataset_manifest_sha256"],
            bridge_factory=value["bridge_factory"],
            bridge_params=value["bridge_params"],
            recall_limit=value["recall_limit"],
            max_depth=value["max_depth"],
            max_fanout=value["max_fanout"],
            token_budget=value["token_budget"],
        )


@dataclass(frozen=True, slots=True)
class RawOperation:
    sequence: int
    operation: str
    depth: int
    seed_memory_ids: tuple[str, ...]
    result_memory_ids: tuple[str, ...]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class EmbeddingRuntimeEvidence:
    """Content-free provider and call-accounting proof for one question scope."""

    retrieval_mode: str
    sota_capable: bool
    provider: str | None
    model: str | None
    model_revision: str | None
    dimensions: int | None
    response_model_requirement: str | None
    query_instruction_sha256: str | None
    inserted_memories: int
    embedding_work_completed: int
    call_accounting_source: str
    document_inputs: int
    document_batch_calls: int
    document_successful_http_calls: int
    document_http_attempts: int
    query_calls: int
    query_successful_http_calls: int
    query_http_attempts: int
    exact_response_model_verified: bool
    deterministic_fallback_used: bool

    def __post_init__(self) -> None:
        for name in (
            "sota_capable",
            "exact_response_model_verified",
            "deterministic_fallback_used",
        ):
            if type(getattr(self, name)) is not bool:
                raise LongMemEvalV2AdapterError(f"embedding evidence {name} must be boolean")
        for name in (
            "provider",
            "model",
            "model_revision",
            "response_model_requirement",
            "query_instruction_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise LongMemEvalV2AdapterError(
                    f"embedding evidence {name} must be null or a non-empty string"
                )
        if self.dimensions is not None and (
            type(self.dimensions) is not int or self.dimensions < 2
        ):
            raise LongMemEvalV2AdapterError(
                "embedding evidence dimensions must be null or an integer of at least 2"
            )
        if self.retrieval_mode not in {"lexical", "deterministic_hybrid", "openai_hybrid"}:
            raise LongMemEvalV2AdapterError("embedding evidence has an unsupported retrieval mode")
        for name in (
            "inserted_memories",
            "embedding_work_completed",
            "document_inputs",
            "document_batch_calls",
            "document_successful_http_calls",
            "document_http_attempts",
            "query_calls",
            "query_successful_http_calls",
            "query_http_attempts",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise LongMemEvalV2AdapterError(
                    f"embedding evidence {name} must be a non-negative integer"
                )
        if self.inserted_memories < 1:
            raise LongMemEvalV2AdapterError(
                "embedding evidence must cover at least one inserted memory"
            )
        if not isinstance(self.call_accounting_source, str) or not (
            self.call_accounting_source.strip()
        ):
            raise LongMemEvalV2AdapterError(
                "embedding evidence call_accounting_source must be non-empty"
            )
        if self.document_http_attempts < self.document_successful_http_calls:
            raise LongMemEvalV2AdapterError(
                "document HTTP attempts cannot be below successful calls"
            )
        if self.query_http_attempts < self.query_successful_http_calls:
            raise LongMemEvalV2AdapterError("query HTTP attempts cannot be below successful calls")

        if self.retrieval_mode == "openai_hybrid":
            revision = _required_text(self.model_revision, "embedding model_revision")
            if len(revision) > 255 or _looks_like_inline_secret(revision):
                raise LongMemEvalV2AdapterError(
                    "embedding model_revision must be a public identifier of at most 255 characters"
                )
            expected = {
                "sota_capable": True,
                "provider": SOTA_EMBEDDING_PROVIDER,
                "model": SOTA_EMBEDDING_MODEL,
                "response_model_requirement": SOTA_EMBEDDING_MODEL,
                "query_instruction_sha256": SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
                "call_accounting_source": "provider-observed",
                "embedding_work_completed": self.inserted_memories,
                "document_inputs": self.inserted_memories,
                "document_batch_calls": self.inserted_memories,
                "document_successful_http_calls": self.inserted_memories,
                "query_calls": 1,
                "query_successful_http_calls": 1,
                "exact_response_model_verified": True,
                "deterministic_fallback_used": False,
            }
            for name, expected_value in expected.items():
                if getattr(self, name) != expected_value:
                    raise LongMemEvalV2AdapterError(
                        f"SOTA embedding evidence {name} must equal {expected_value!r}"
                    )
            if (
                type(self.dimensions) is not int
                or not 32 <= self.dimensions <= SOTA_EMBEDDING_DIMENSIONS
            ):
                raise LongMemEvalV2AdapterError(
                    "SOTA embedding evidence dimensions must be an integer in [32, 4096]"
                )
            if self.document_http_attempts < self.inserted_memories:
                raise LongMemEvalV2AdapterError(
                    "SOTA embedding evidence lacks document HTTP attempt coverage"
                )
            if self.query_http_attempts < 1:
                raise LongMemEvalV2AdapterError(
                    "SOTA embedding evidence lacks query HTTP attempt coverage"
                )
            return

        if self.sota_capable:
            raise LongMemEvalV2AdapterError(
                "lexical and deterministic embedding modes are development-only"
            )
        if self.model_revision is not None or self.response_model_requirement is not None:
            raise LongMemEvalV2AdapterError(
                "development embedding evidence cannot claim a remote model revision"
            )
        if self.query_instruction_sha256 is not None:
            raise LongMemEvalV2AdapterError(
                "development embedding evidence cannot claim the SOTA query instruction"
            )
        if self.exact_response_model_verified:
            raise LongMemEvalV2AdapterError(
                "development embedding evidence cannot verify a remote response model"
            )
        if self.deterministic_fallback_used:
            raise LongMemEvalV2AdapterError(
                "development modes must be selected explicitly, not reported as fallback"
            )
        if self.document_successful_http_calls or self.document_http_attempts:
            raise LongMemEvalV2AdapterError(
                "development embedding evidence cannot claim document HTTP calls"
            )
        if self.query_successful_http_calls or self.query_http_attempts:
            raise LongMemEvalV2AdapterError(
                "development embedding evidence cannot claim query HTTP calls"
            )
        if self.retrieval_mode == "lexical":
            if self.provider is not None or self.model is not None or self.dimensions is not None:
                raise LongMemEvalV2AdapterError(
                    "lexical evidence cannot claim an embedding provider"
                )
            if any(
                (
                    self.embedding_work_completed,
                    self.document_inputs,
                    self.document_batch_calls,
                    self.query_calls,
                )
            ):
                raise LongMemEvalV2AdapterError(
                    "lexical evidence cannot claim embedding work or calls"
                )
        else:
            if (
                self.provider != "DeterministicEmbeddingProvider"
                or not isinstance(self.model, str)
                or type(self.dimensions) is not int
                or self.embedding_work_completed != self.inserted_memories
                or self.document_inputs != self.inserted_memories
                or self.document_batch_calls != self.inserted_memories
                or self.query_calls != 1
            ):
                raise LongMemEvalV2AdapterError(
                    "deterministic development evidence does not reconcile with its explicit mode"
                )

    def as_json(self) -> JsonObject:
        return {
            "retrieval_mode": self.retrieval_mode,
            "sota_capable": self.sota_capable,
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "response_model_requirement": self.response_model_requirement,
            "query_instruction_sha256": self.query_instruction_sha256,
            "inserted_memories": self.inserted_memories,
            "embedding_work_completed": self.embedding_work_completed,
            "call_accounting": {
                "source": self.call_accounting_source,
                "document_inputs": self.document_inputs,
                "document_batch_calls": self.document_batch_calls,
                "document_successful_http_calls": self.document_successful_http_calls,
                "document_http_attempts": self.document_http_attempts,
                "query_calls": self.query_calls,
                "query_successful_http_calls": self.query_successful_http_calls,
                "query_http_attempts": self.query_http_attempts,
            },
            "exact_response_model_verified": self.exact_response_model_verified,
            "deterministic_fallback_used": self.deterministic_fallback_used,
        }

    @classmethod
    def from_json(cls, value: Any) -> EmbeddingRuntimeEvidence:
        if not isinstance(value, dict):
            raise LongMemEvalV2AdapterError("embedding evidence must be an object")
        expected = {
            "retrieval_mode",
            "sota_capable",
            "provider",
            "model",
            "model_revision",
            "dimensions",
            "response_model_requirement",
            "query_instruction_sha256",
            "inserted_memories",
            "embedding_work_completed",
            "call_accounting",
            "exact_response_model_verified",
            "deterministic_fallback_used",
        }
        if set(value) != expected:
            raise LongMemEvalV2AdapterError("embedding evidence fields differ from schema")
        accounting = value.get("call_accounting")
        accounting_fields = {
            "source",
            "document_inputs",
            "document_batch_calls",
            "document_successful_http_calls",
            "document_http_attempts",
            "query_calls",
            "query_successful_http_calls",
            "query_http_attempts",
        }
        if not isinstance(accounting, dict) or set(accounting) != accounting_fields:
            raise LongMemEvalV2AdapterError(
                "embedding evidence call_accounting fields differ from schema"
            )
        return cls(
            retrieval_mode=value["retrieval_mode"],
            sota_capable=value["sota_capable"],
            provider=value["provider"],
            model=value["model"],
            model_revision=value["model_revision"],
            dimensions=value["dimensions"],
            response_model_requirement=value["response_model_requirement"],
            query_instruction_sha256=value["query_instruction_sha256"],
            inserted_memories=value["inserted_memories"],
            embedding_work_completed=value["embedding_work_completed"],
            call_accounting_source=accounting["source"],
            document_inputs=accounting["document_inputs"],
            document_batch_calls=accounting["document_batch_calls"],
            document_successful_http_calls=accounting["document_successful_http_calls"],
            document_http_attempts=accounting["document_http_attempts"],
            query_calls=accounting["query_calls"],
            query_successful_http_calls=accounting["query_successful_http_calls"],
            query_http_attempts=accounting["query_http_attempts"],
            exact_response_model_verified=value["exact_response_model_verified"],
            deterministic_fallback_used=value["deterministic_fallback_used"],
        )


@dataclass(frozen=True, slots=True)
class RawQueryTrace:
    """Run-local, content-free trace keyed by the harness's opaque handle."""

    opaque_invocation_id: str
    operations: tuple[RawOperation, ...]
    embedding: EmbeddingRuntimeEvidence


@dataclass(frozen=True, slots=True)
class BoundQueryEvidence:
    """Compiler-shaped evidence after the harness restores the question ID."""

    question_id: str
    domain: str
    query_tokens: int
    query_latency_ms: float
    operations: tuple[JsonObject, ...]
    embedding: EmbeddingRuntimeEvidence
    trace_sha256: str

    def sidecar_row(self, *, unanswered: bool) -> JsonObject:
        return {
            "question_id": self.question_id,
            "domain": self.domain,
            "query_tokens": self.query_tokens,
            "query_latency_ms": self.query_latency_ms,
            "query_failed": False,
            "unanswered": unanswered,
            "operations": [dict(operation) for operation in self.operations],
            "embedding": self.embedding.as_json(),
        }


__all__ = [
    "ADAPTER_REVISION",
    "AdapterConfig",
    "BoundQueryEvidence",
    "EmbeddingRuntimeEvidence",
    "EXPECTED_JUDGE_MODEL",
    "EXPECTED_QUESTIONS",
    "EXPECTED_READER_MODEL",
    "JsonObject",
    "LongMemEvalV2AdapterError",
    "MEMORY_TYPE",
    "METHOD",
    "PINNED_REPOSITORY_COMMIT",
    "RAW_TRACE_DIGEST_METADATA_KEY",
    "RawOperation",
    "RawQueryTrace",
    "ReadExpandMemoryResult",
    "RecallMemoryResult",
    "SIDECAR_SCHEMA_VERSION",
    "SOTA_EMBEDDING_DIMENSIONS",
    "SOTA_EMBEDDING_MODEL",
    "SOTA_EMBEDDING_PROVIDER",
    "SOTA_EMBEDDING_QUERY_INSTRUCTION",
    "SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256",
    "SwarmOperationBridge",
    "TRACE_DIGEST_METADATA_KEY",
]
