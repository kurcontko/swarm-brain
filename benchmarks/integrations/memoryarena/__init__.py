"""Official MemoryArena memory-API integration scaffold."""

from .contracts import (
    DETERMINISTIC_EMBEDDING_MODE,
    OFFICIAL_DATASET,
    OFFICIAL_MEMORY_SYSTEM_NAME,
    OFFICIAL_METRICS,
    OFFICIAL_REPOSITORY,
    PAPER_DECLARED_TASK_GROUPS,
    PAPER_TABLE_COMPONENT_TOTAL,
    PINNED_REPOSITORY_COMMIT,
    SEMANTIC_EMBEDDING_DIMENSIONS,
    SEMANTIC_EMBEDDING_MODE,
    SEMANTIC_EMBEDDING_MODEL_ID,
    SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256,
    BridgeConfig,
    MemoryArenaContractError,
)
from .preflight import PreflightReport, run_preflight
from .runtime_bridge import MemoryArenaRuntimeBridge
from .server import create_memoryarena_app

__all__ = [
    "BridgeConfig",
    "DETERMINISTIC_EMBEDDING_MODE",
    "MemoryArenaContractError",
    "MemoryArenaRuntimeBridge",
    "OFFICIAL_DATASET",
    "OFFICIAL_MEMORY_SYSTEM_NAME",
    "OFFICIAL_METRICS",
    "OFFICIAL_REPOSITORY",
    "PAPER_DECLARED_TASK_GROUPS",
    "PAPER_TABLE_COMPONENT_TOTAL",
    "PINNED_REPOSITORY_COMMIT",
    "PreflightReport",
    "SEMANTIC_EMBEDDING_DIMENSIONS",
    "SEMANTIC_EMBEDDING_MODE",
    "SEMANTIC_EMBEDDING_MODEL_ID",
    "SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256",
    "create_memoryarena_app",
    "run_preflight",
]
