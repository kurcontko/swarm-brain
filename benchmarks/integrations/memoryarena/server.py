"""FastAPI wiring compatible with MemoryArena's pinned ``MemoryClient``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict

from .contracts import (
    MemoryArenaContractError,
    MemoryArenaNotInitialized,
    MemoryArenaSystemMismatch,
    MemoryArenaUnsupportedSystem,
)
from .runtime_bridge import MemoryArenaRuntimeBridge


class InitializeRequest(BaseModel):
    # Upstream uses Pydantic's default extra-field behavior.  Keeping it here
    # avoids breaking an otherwise valid pinned MemoryClient request.
    model_config = ConfigDict(extra="ignore")

    user_id: str
    memory_system_name: str


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    chunk: str
    memory_system_name: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    question: str
    memory_system_name: str


def create_memoryarena_app(
    bridge: MemoryArenaRuntimeBridge | None = None,
    *,
    close_on_shutdown: bool = True,
) -> Any:
    """Create the local compatibility application without opening a socket."""

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised without the serve extra
        raise RuntimeError("MemoryArena HTTP compatibility requires swarmbrain[serve]") from exc

    runtime_bridge = bridge or MemoryArenaRuntimeBridge()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if close_on_shutdown:
            await runtime_bridge.close()

    app = FastAPI(title="Memory Agent Server", lifespan=lifespan)
    app.state.memoryarena_bridge = runtime_bridge

    @app.exception_handler(MemoryArenaNotInitialized)
    async def not_initialized(_request: Request, exc: MemoryArenaNotInitialized) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(MemoryArenaSystemMismatch)
    async def mismatched(_request: Request, exc: MemoryArenaSystemMismatch) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(MemoryArenaUnsupportedSystem)
    async def unsupported(_request: Request, exc: MemoryArenaUnsupportedSystem) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(MemoryArenaContractError)
    async def invalid_contract(_request: Request, exc: MemoryArenaContractError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/memory/initialize")
    async def initialize(request: InitializeRequest) -> dict[str, Any]:
        return await runtime_bridge.initialize(request.user_id, request.memory_system_name)

    @app.post("/memory/add")
    async def add(request: AddRequest) -> dict[str, Any]:
        return await runtime_bridge.add(
            request.user_id,
            request.memory_system_name,
            request.chunk,
        )

    @app.post("/memory/wrap_user_prompt")
    async def wrap_user_prompt(request: QueryRequest) -> dict[str, Any]:
        return await runtime_bridge.wrap_user_prompt(
            request.user_id,
            request.memory_system_name,
            request.question,
        )

    return app


__all__ = [
    "AddRequest",
    "InitializeRequest",
    "QueryRequest",
    "create_memoryarena_app",
]
