from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from typing import Any

import httpx
import pytest

from swarmbrain.application import runtime as runtime_module
from swarmbrain.application.runtime import build_in_memory_runtime, build_runtime
from swarmbrain.config import ApiSettings, BackendKind, BridgeSettings
from swarmbrain.transports.http import create_app

SECRET = "0123456789abcdef-local-test-secret"
DATABASE_URL = "postgresql://root@127.0.0.1:26257/swarmbrain?sslmode=disable"
API_ENVIRONMENT = (
    "SWARMBRAIN_BACKEND",
    "SWARMBRAIN_TOKEN_SECRET",
    "SWARMBRAIN_DATABASE_URL",
    "SWARMBRAIN_DATABASE_POOL_MIN_SIZE",
    "SWARMBRAIN_DATABASE_POOL_MAX_SIZE",
    "SWARMBRAIN_RETRIEVAL_DENSE_MIN_SIMILARITY",
    "SWARMBRAIN_RETRIEVAL_DENSE_ANN_BEAM_SIZE",
    "SWARMBRAIN_HOST",
    "SWARMBRAIN_PORT",
)


def _clear_api_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in API_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def test_api_settings_require_an_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_api_environment(monkeypatch)
    monkeypatch.setenv("SWARMBRAIN_TOKEN_SECRET", SECRET)

    with pytest.raises(RuntimeError, match="SWARMBRAIN_BACKEND"):
        ApiSettings.from_env()


def test_api_settings_accept_an_explicit_memory_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_api_environment(monkeypatch)
    monkeypatch.setenv("SWARMBRAIN_BACKEND", "memory")
    monkeypatch.setenv("SWARMBRAIN_TOKEN_SECRET", SECRET)

    settings = ApiSettings.from_env()

    assert settings.backend is BackendKind.MEMORY
    assert settings.database_url is None
    assert settings.database_pool_min_size == 1
    assert settings.database_pool_max_size == 12
    assert settings.retrieval_dense_min_similarity == pytest.approx(0.0)
    assert settings.retrieval_dense_ann_beam_size == 32


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {
                "SWARMBRAIN_BACKEND": "memory",
                "SWARMBRAIN_DATABASE_URL": DATABASE_URL,
            },
            "database URL is not allowed",
        ),
        ({"SWARMBRAIN_BACKEND": "cockroach"}, "database URL is required"),
        ({"SWARMBRAIN_BACKEND": "sqlite"}, "memory, cockroach"),
        (
            {
                "SWARMBRAIN_BACKEND": "cockroach",
                "SWARMBRAIN_DATABASE_URL": "not-a-database-url",
            },
            "must use postgres",
        ),
        (
            {
                "SWARMBRAIN_BACKEND": "cockroach",
                "SWARMBRAIN_DATABASE_URL": DATABASE_URL,
                "SWARMBRAIN_DATABASE_POOL_MIN_SIZE": "5",
                "SWARMBRAIN_DATABASE_POOL_MAX_SIZE": "4",
            },
            "pool minimum",
        ),
        (
            {
                "SWARMBRAIN_BACKEND": "memory",
                "SWARMBRAIN_RETRIEVAL_DENSE_MIN_SIMILARITY": "1.1",
            },
            "minimum similarity",
        ),
        (
            {
                "SWARMBRAIN_BACKEND": "memory",
                "SWARMBRAIN_RETRIEVAL_DENSE_ANN_BEAM_SIZE": "0",
            },
            "beam size",
        ),
    ],
)
def test_api_settings_fail_closed_for_inconsistent_backend_configuration(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    message: str,
) -> None:
    _clear_api_environment(monkeypatch)
    monkeypatch.setenv("SWARMBRAIN_TOKEN_SECRET", SECRET)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        ApiSettings.from_env()


def test_settings_representations_do_not_expose_credentials() -> None:
    settings = ApiSettings(
        backend=BackendKind.COCKROACH,
        token_secret=SECRET,
        database_url=DATABASE_URL,
    )
    bridge = BridgeSettings("http://127.0.0.1:8080", "agent-token-secret")

    assert SECRET not in repr(settings)
    assert DATABASE_URL not in repr(settings)
    assert "agent-token-secret" not in repr(bridge)


class _FakeDatabase:
    instances: list[_FakeDatabase] = []

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int,
        max_size: int,
    ) -> None:
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.start_calls: list[bool] = []
        self.health_calls = 0
        self.close_calls = 0
        self.__class__.instances.append(self)

    async def start(self, *, verify_schema: bool = True) -> None:
        self.start_calls.append(verify_schema)

    async def health(self) -> None:
        self.health_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


class _FakeCoordinationStore:
    def __init__(self, database: _FakeDatabase) -> None:
        self.database = database


class _FakeMemoryStore:
    def __init__(self, database: _FakeDatabase) -> None:
        self.database = database


def _install_fake_cockroach_repositories(monkeypatch: pytest.MonkeyPatch) -> None:
    database_module = ModuleType("swarmbrain.adapters.cockroach.database")
    database_module.CockroachDatabase = _FakeDatabase  # type: ignore[attr-defined]
    coordination_module = ModuleType("swarmbrain.adapters.cockroach.coordination")
    coordination_module.CockroachCoordinationStore = _FakeCoordinationStore  # type: ignore[attr-defined]
    memory_module = ModuleType("swarmbrain.adapters.cockroach.memory")
    memory_module.CockroachMemoryStore = _FakeMemoryStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, database_module.__name__, database_module)
    monkeypatch.setitem(sys.modules, coordination_module.__name__, coordination_module)
    monkeypatch.setitem(sys.modules, memory_module.__name__, memory_module)


@pytest.mark.asyncio
async def test_cockroach_composition_is_lazy_and_startup_only_verifies_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeDatabase.instances.clear()
    _install_fake_cockroach_repositories(monkeypatch)
    settings = ApiSettings(
        backend=BackendKind.COCKROACH,
        token_secret=SECRET,
        database_url=DATABASE_URL,
        database_pool_min_size=2,
        database_pool_max_size=7,
    )

    runtime = build_runtime(settings)
    database = _FakeDatabase.instances[-1]

    assert runtime.backend is BackendKind.COCKROACH
    assert database.start_calls == []
    assert runtime.coordination.store.database is database
    assert runtime.memory.store.database is database

    await runtime.start()
    assert database.start_calls == [True]
    assert await runtime.ready() is True
    await runtime.close()
    assert database.health_calls == 1
    assert database.close_calls == 1


class _Lifecycle:
    def __init__(self, *, health_error: Exception | None = None) -> None:
        self.health_error = health_error
        self.started = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def health(self) -> None:
        if self.health_error is not None:
            raise self.health_error


@pytest.mark.asyncio
async def test_http_liveness_is_independent_and_readiness_does_not_leak_errors() -> None:
    leaked_value = "postgresql://root:secret@database.internal/swarmbrain"
    lifecycle = _Lifecycle(health_error=RuntimeError(leaked_value))
    runtime = replace(build_in_memory_runtime(SECRET), _lifecycle=lifecycle)
    transport = httpx.ASGITransport(app=create_app(runtime))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        liveness = await client.get("/healthz")
        readiness = await client.get("/readyz")

    assert liveness.status_code == 200
    assert liveness.json() == {"status": "ok"}
    assert readiness.status_code == 503
    assert readiness.json() == {"status": "not_ready"}
    assert leaked_value not in readiness.text


@pytest.mark.asyncio
async def test_http_lifespan_starts_and_closes_the_runtime() -> None:
    lifecycle = _Lifecycle()
    runtime = replace(build_in_memory_runtime(SECRET), _lifecycle=lifecycle)
    app = create_app(runtime)

    async with app.router.lifespan_context(app):
        assert lifecycle.started == 1
        assert lifecycle.closed == 0

    assert lifecycle.closed == 1


@pytest.mark.asyncio
async def test_http_readiness_reports_ready_for_a_healthy_backend() -> None:
    transport = httpx.ASGITransport(app=create_app(build_in_memory_runtime(SECRET)))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_memory_composition_does_not_cross_the_cockroach_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(_settings: ApiSettings) -> Any:
        raise AssertionError("memory composition imported the CockroachDB backend")

    monkeypatch.setattr(runtime_module, "_build_cockroach_runtime", unexpected)
    settings = ApiSettings(backend=BackendKind.MEMORY, token_secret=SECRET)

    runtime = build_runtime(settings)

    assert runtime.backend is BackendKind.MEMORY
    assert await runtime.ready() is True
