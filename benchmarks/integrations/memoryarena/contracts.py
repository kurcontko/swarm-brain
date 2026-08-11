"""Pinned public contracts for the MemoryArena compatibility boundary.

The upstream repository is a preview.  Constants in this module describe only
facts visible in the pinned paper/repository; they are not score claims.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

OFFICIAL_REPOSITORY = "https://github.com/ZexueHe/MemoryArena"
PINNED_REPOSITORY_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
OFFICIAL_DATASET = "ZexueHe/memoryarena"
OFFICIAL_DATASET_SPLIT = "test"
OFFICIAL_MEMORY_SYSTEM_NAME = "swarmbrain"
OFFICIAL_METRICS = ("progress_score", "success_rate")

DETERMINISTIC_EMBEDDING_MODE = "deterministic"
SEMANTIC_EMBEDDING_MODE = "openai_semantic"
SEMANTIC_EMBEDDING_PROVIDER = "OpenAICompatibleEmbeddingProvider"
SEMANTIC_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
SEMANTIC_EMBEDDING_DIMENSIONS = 4_096
SEMANTIC_EMBEDDING_QUERY_INSTRUCTION = (
    "Given the current observation or question in an interdependent multi-session agent task, "
    "retrieve prior interaction memories that help determine the next action."
)
SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256 = (
    "5f12a399815ecbb080d4c0b5fd8f1f82b8a7a6dff9e8375ed5a6a955503a10ec"
)
if (
    hashlib.sha256(SEMANTIC_EMBEDDING_QUERY_INSTRUCTION.encode("utf-8")).hexdigest()
    != SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256
):  # pragma: no cover - import-time invariant
    raise RuntimeError("MemoryArena semantic query-instruction digest drifted")

# The paper overview reports 766 task groups.  Its public per-domain table
# reports the five counts below, which sum to 736.  Do not silently pick one.
PAPER_DECLARED_TASK_GROUPS = 766
PAPER_TABLE_DOMAIN_TASK_GROUPS = {
    "bundled_shopping": 150,
    "formal_reasoning_math": 40,
    "formal_reasoning_phys": 20,
    "group_travel_planner": 270,
    "progressive_search": 256,
}
PAPER_TABLE_COMPONENT_TOTAL = sum(PAPER_TABLE_DOMAIN_TASK_GROUPS.values())

OFFICIAL_DATASET_CONFIGS = frozenset(PAPER_TABLE_DOMAIN_TASK_GROUPS)
OFFICIAL_RUNNER_FILES = (
    "run_shopping.py",
    "run_travel.py",
    "run_search.py",
    "run_math.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_ENVIRONMENT_VARIABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


class MemoryArenaContractError(ValueError):
    """A request, pinned input, or evidence object violated the bridge contract."""


class MemoryArenaNotInitialized(MemoryArenaContractError):
    """The official user scope has not been initialized."""


class MemoryArenaSystemMismatch(MemoryArenaContractError):
    """The request names a different memory implementation for this scope."""


class MemoryArenaUnsupportedSystem(MemoryArenaContractError):
    """The bridge was asked to instantiate an implementation it does not own."""


def canonical_json(value: Any) -> str:
    """Encode strict, deterministic JSON for fingerprints and evidence binding."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise MemoryArenaContractError("value is not canonical-JSON serializable") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MemoryArenaContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_public_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryArenaContractError(f"{label} must be a non-empty string")
    text = value.strip()
    if len(text) > maximum:
        raise MemoryArenaContractError(f"{label} exceeds {maximum} characters")
    return text


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    """Closed bridge configuration with an explicit evidence capability mode."""

    memory_system_name: str = OFFICIAL_MEMORY_SYSTEM_NAME
    recall_limit: int = 3
    memory_context_token_budget: int = 2_048
    embedding_mode: str = DETERMINISTIC_EMBEDDING_MODE
    embedding_dimensions: int = 256
    embedding_base_url: str | None = None
    embedding_api_key_env: str | None = None
    embedding_model_id: str | None = None
    embedding_model_revision: str | None = None
    embedding_response_model: str | None = None
    max_user_id_chars: int = 512
    max_chunk_chars: int = 262_144
    max_question_chars: int = 131_072

    def __post_init__(self) -> None:
        if self.memory_system_name != OFFICIAL_MEMORY_SYSTEM_NAME:
            raise MemoryArenaContractError(
                f"memory_system_name must be exactly {OFFICIAL_MEMORY_SYSTEM_NAME!r}"
            )
        for name, value, minimum, maximum in (
            ("recall_limit", self.recall_limit, 1, 100),
            (
                "memory_context_token_budget",
                self.memory_context_token_budget,
                1,
                131_072,
            ),
            ("embedding_dimensions", self.embedding_dimensions, 2, 4_096),
            ("max_user_id_chars", self.max_user_id_chars, 1, 4_096),
            ("max_chunk_chars", self.max_chunk_chars, 1, 1_048_576),
            ("max_question_chars", self.max_question_chars, 1, 1_048_576),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise MemoryArenaContractError(
                    f"{name} must be an integer in [{minimum}, {maximum}]"
                )
        if self.embedding_mode not in {
            DETERMINISTIC_EMBEDDING_MODE,
            SEMANTIC_EMBEDDING_MODE,
        }:
            raise MemoryArenaContractError(
                "embedding_mode must be exactly 'deterministic' or 'openai_semantic'"
            )
        semantic_fields = {
            "embedding_base_url": self.embedding_base_url,
            "embedding_api_key_env": self.embedding_api_key_env,
            "embedding_model_id": self.embedding_model_id,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_response_model": self.embedding_response_model,
        }
        if self.embedding_mode == DETERMINISTIC_EMBEDDING_MODE:
            supplied = sorted(name for name, value in semantic_fields.items() if value is not None)
            if supplied:
                raise MemoryArenaContractError(
                    "semantic embedding settings require embedding_mode='openai_semantic': "
                    + ", ".join(supplied)
                )
            return

        base_url = require_public_text(
            self.embedding_base_url,
            "embedding_base_url",
            maximum=2_048,
        )
        if base_url != self.embedding_base_url or base_url.endswith("/"):
            raise MemoryArenaContractError(
                "embedding_base_url must be canonical without whitespace or a trailing slash"
            )
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise MemoryArenaContractError(
                "embedding_base_url must be an http(s) URL without credentials, query, or fragment"
            )
        api_key_env = require_public_text(
            self.embedding_api_key_env,
            "embedding_api_key_env",
            maximum=128,
        )
        if _ENVIRONMENT_VARIABLE_RE.fullmatch(api_key_env) is None:
            raise MemoryArenaContractError(
                "embedding_api_key_env must name an environment variable"
            )
        if self.embedding_model_id != SEMANTIC_EMBEDDING_MODEL_ID:
            raise MemoryArenaContractError(
                f"embedding_model_id must be exactly {SEMANTIC_EMBEDDING_MODEL_ID!r}"
            )
        if self.embedding_dimensions != SEMANTIC_EMBEDDING_DIMENSIONS:
            raise MemoryArenaContractError(
                f"semantic embedding_dimensions must equal {SEMANTIC_EMBEDDING_DIMENSIONS}"
            )
        revision = require_public_text(
            self.embedding_model_revision,
            "embedding_model_revision",
            maximum=64,
        )
        if _IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
            raise MemoryArenaContractError(
                "embedding_model_revision must be a 40- or 64-character lowercase hex revision"
            )
        if self.embedding_response_model != SEMANTIC_EMBEDDING_MODEL_ID:
            raise MemoryArenaContractError(
                f"embedding_response_model must be exactly {SEMANTIC_EMBEDDING_MODEL_ID!r}"
            )


__all__ = [
    "BridgeConfig",
    "DETERMINISTIC_EMBEDDING_MODE",
    "MemoryArenaContractError",
    "MemoryArenaNotInitialized",
    "MemoryArenaSystemMismatch",
    "MemoryArenaUnsupportedSystem",
    "OFFICIAL_DATASET",
    "OFFICIAL_DATASET_CONFIGS",
    "OFFICIAL_DATASET_SPLIT",
    "OFFICIAL_MEMORY_SYSTEM_NAME",
    "OFFICIAL_METRICS",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_RUNNER_FILES",
    "PAPER_DECLARED_TASK_GROUPS",
    "PAPER_TABLE_COMPONENT_TOTAL",
    "PAPER_TABLE_DOMAIN_TASK_GROUPS",
    "PINNED_REPOSITORY_COMMIT",
    "SEMANTIC_EMBEDDING_DIMENSIONS",
    "SEMANTIC_EMBEDDING_MODE",
    "SEMANTIC_EMBEDDING_MODEL_ID",
    "SEMANTIC_EMBEDDING_PROVIDER",
    "SEMANTIC_EMBEDDING_QUERY_INSTRUCTION",
    "SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256",
    "canonical_json",
    "require_public_text",
    "require_sha256",
    "sha256_json",
    "sha256_text",
]
