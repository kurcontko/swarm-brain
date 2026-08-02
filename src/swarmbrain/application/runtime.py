"""Composition root shared by HTTP and local tests."""

from __future__ import annotations

from dataclasses import dataclass

from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.adapters.memory import InMemoryKernel

from .conflict_service import ConflictService
from .coordination import CoordinationService
from .memory_policy import ConservativeMemoryPolicy
from .memory_service import MemoryService


@dataclass(frozen=True, slots=True)
class SwarmBrainRuntime:
    kernel: InMemoryKernel
    coordination: CoordinationService
    memory: MemoryService
    conflicts: ConflictService
    tokens: RunTokenCodec


def build_in_memory_runtime(token_secret: str) -> SwarmBrainRuntime:
    kernel = InMemoryKernel()
    policy = ConservativeMemoryPolicy()
    memory = MemoryService(kernel, policy, review_store=kernel)
    coordination = CoordinationService(kernel, memory_service=memory)
    return SwarmBrainRuntime(
        kernel=kernel,
        coordination=coordination,
        memory=memory,
        conflicts=ConflictService(kernel),
        tokens=RunTokenCodec(token_secret),
    )


__all__ = ["SwarmBrainRuntime", "build_in_memory_runtime"]
