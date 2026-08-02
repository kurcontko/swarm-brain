"""Composition root shared by HTTP and local tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.config import ApiSettings, BackendKind

from .conflict_service import ConflictService
from .coordination import CoordinationService
from .memory_policy import ConservativeMemoryPolicy
from .memory_service import MemoryService


class RuntimeLifecycle(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def health(self) -> None: ...


class _CockroachResource(Protocol):
    async def start(self, *, verify_schema: bool = True) -> None: ...

    async def close(self) -> None: ...

    async def health(self) -> None: ...


class _InMemoryLifecycle:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health(self) -> None:
        return None


class _CockroachLifecycle:
    def __init__(self, database: _CockroachResource) -> None:
        self._database = database

    async def start(self) -> None:
        await self._database.start(verify_schema=True)

    async def close(self) -> None:
        await self._database.close()

    async def health(self) -> None:
        await self._database.health()


@dataclass(frozen=True, slots=True)
class SwarmBrainRuntime:
    backend: BackendKind
    coordination: CoordinationService
    memory: MemoryService
    conflicts: ConflictService
    tokens: RunTokenCodec
    kernel: InMemoryKernel | None = None
    _lifecycle: RuntimeLifecycle = field(default_factory=_InMemoryLifecycle, repr=False)

    async def start(self) -> None:
        """Open runtime resources and verify prerequisites without applying DDL."""

        await self._lifecycle.start()

    async def close(self) -> None:
        await self._lifecycle.close()

    async def ready(self) -> bool:
        try:
            await self._lifecycle.health()
        except Exception:
            return False
        return True


def build_in_memory_runtime(token_secret: str) -> SwarmBrainRuntime:
    kernel = InMemoryKernel()
    policy = ConservativeMemoryPolicy()
    memory = MemoryService(kernel, policy, review_store=kernel)
    coordination = CoordinationService(kernel, memory_service=memory)
    return SwarmBrainRuntime(
        backend=BackendKind.MEMORY,
        coordination=coordination,
        memory=memory,
        conflicts=ConflictService(kernel),
        tokens=RunTokenCodec(token_secret),
        kernel=kernel,
    )


def build_runtime(settings: ApiSettings) -> SwarmBrainRuntime:
    """Compose the selected backend without opening network resources."""

    if settings.backend is BackendKind.MEMORY:
        return build_in_memory_runtime(settings.token_secret)
    return _build_cockroach_runtime(settings)


def _build_cockroach_runtime(settings: ApiSettings) -> SwarmBrainRuntime:
    # Keep the optional driver and durable repositories out of memory-only imports.
    from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
    from swarmbrain.adapters.cockroach.database import CockroachDatabase
    from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore

    assert settings.database_url is not None
    database = CockroachDatabase(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    coordination_store = CockroachCoordinationStore(database)
    memory_store = CockroachMemoryStore(database)
    policy = ConservativeMemoryPolicy()
    memory = MemoryService(memory_store, policy, review_store=memory_store)
    coordination = CoordinationService(coordination_store, memory_service=memory)
    return SwarmBrainRuntime(
        backend=BackendKind.COCKROACH,
        coordination=coordination,
        memory=memory,
        conflicts=ConflictService(memory_store),
        tokens=RunTokenCodec(settings.token_secret),
        _lifecycle=_CockroachLifecycle(database),
    )


__all__ = ["SwarmBrainRuntime", "build_in_memory_runtime", "build_runtime"]
