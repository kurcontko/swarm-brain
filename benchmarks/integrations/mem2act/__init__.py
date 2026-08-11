"""Reproducible Mem2ActBench integration."""

from importlib import import_module
from typing import Any

from .contracts import (
    CorpusSession,
    Mem2ActContractError,
    Mem2ActDataset,
    MemoryBridge,
    PublicConversationSession,
    ReaderRequest,
    ReaderResult,
    ToolSelectionReader,
)
from .dataset import (
    MEM2ACT_REPO_COMMIT,
    OFFICIAL_MEM2ACT_SPEC,
    DatasetSpec,
    load_mem2act_dataset,
)
from .report import Mem2ActReportError, compile_mem2act_report
from .runner import (
    BenchmarkConfig,
    BenchmarkResult,
    Mem2ActEvaluator,
    write_benchmark_outputs,
)

_LAZY_EXPORTS = {
    "OpenAICompatibleReaderConfig": (".openai_reader", "OpenAICompatibleReaderConfig"),
    "OpenAICompatibleReaderUnavailable": (
        ".openai_reader",
        "OpenAICompatibleReaderUnavailable",
    ),
    "OpenAICompatibleToolSelectionReader": (
        ".openai_reader",
        "OpenAICompatibleToolSelectionReader",
    ),
    "RuntimeMemoryBridge": (".runtime_bridge", "RuntimeMemoryBridge"),
    "build_openai_compatible_reader": (".openai_reader", "build_reader"),
    "build_openai_semantic_in_memory_bridge": (
        ".runtime_bridge",
        "build_openai_semantic_in_memory_bridge",
    ),
    "build_public_in_memory_bridge": (".runtime_bridge", "build_public_in_memory_bridge"),
}


def __getattr__(name: str) -> Any:
    """Load provider/runtime adapters only when their public export is requested."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "CorpusSession",
    "DatasetSpec",
    "MEM2ACT_REPO_COMMIT",
    "Mem2ActContractError",
    "Mem2ActDataset",
    "Mem2ActEvaluator",
    "Mem2ActReportError",
    "MemoryBridge",
    "OFFICIAL_MEM2ACT_SPEC",
    "OpenAICompatibleReaderConfig",
    "OpenAICompatibleReaderUnavailable",
    "OpenAICompatibleToolSelectionReader",
    "PublicConversationSession",
    "ReaderRequest",
    "ReaderResult",
    "RuntimeMemoryBridge",
    "ToolSelectionReader",
    "build_public_in_memory_bridge",
    "build_openai_semantic_in_memory_bridge",
    "build_openai_compatible_reader",
    "compile_mem2act_report",
    "load_mem2act_dataset",
    "write_benchmark_outputs",
]
