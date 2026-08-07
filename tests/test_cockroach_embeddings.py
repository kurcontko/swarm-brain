"""Non-live checks for the fixed-width, fully scoped Cockroach ANN adapter."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.cockroach.memory import (
    COCKROACH_VECTOR_DIMENSIONS,
    CockroachMemoryStore,
)
from swarmbrain.application.errors import ResourceNotFound
from swarmbrain.domain.memory import EmbeddingVector
from swarmbrain.ports.embeddings import EmbeddingIndex


class _Cursor:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        rowcount: int = 0,
    ) -> None:
        self.rows = rows or []
        self.rowcount = rowcount

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows

    async def fetchone(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


class _RecordingConnection:
    def __init__(self, *, upsert_rowcount: int = 1) -> None:
        self.upsert_rowcount = upsert_rowcount
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> _Cursor:
        self.calls.append((sql, parameters))
        if "INSERT INTO memory_vector_embeddings" in sql:
            return _Cursor(rowcount=self.upsert_rowcount)
        if "FROM memories" in sql:
            return _Cursor(
                [
                    {
                        "id": parameters[0],
                        "repository_id": parameters[3],
                        "run_id": UUID(new_id()),
                        "task_id": None,
                        "visibility": "repository",
                        "kind": "observation",
                        "metadata": {},
                        "version": 3,
                        "normalized_sha256": b"content-digest",
                    }
                ]
            )
        if "UPSERT INTO retrieval_vectors_1024" in sql:
            return _Cursor(rowcount=1)
        return _Cursor(
            [
                {"memory_id": UUID(new_id()), "distance": 0.125},
                {"memory_id": UUID(new_id()), "distance": 0.75},
            ]
        )


class _Pool:
    def __init__(self, connection: _RecordingConnection) -> None:
        self.value = connection

    @asynccontextmanager
    async def connection(self) -> Any:
        yield self.value


def _store(connection: _RecordingConnection) -> CockroachMemoryStore:
    return CockroachMemoryStore(SimpleNamespace(pool=_Pool(connection)))  # type: ignore[arg-type]


def _vector(*, memory_id: str, dimensions: int = COCKROACH_VECTOR_DIMENSIONS) -> EmbeddingVector:
    return EmbeddingVector(
        memory_id=memory_id,
        model="amazon.titan-embed-text-v2:0",
        dimensions=dimensions,
        values=tuple([1.0] + [0.0] * (dimensions - 1)),
    )


def test_cockroach_store_satisfies_embedding_index_protocol() -> None:
    assert isinstance(_store(_RecordingConnection()), EmbeddingIndex)


@pytest.mark.asyncio
async def test_vector_upsert_derives_and_filters_complete_scope(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    connection = _RecordingConnection()
    memory_id = new_id()

    await _store(connection).upsert_embeddings(
        actor,
        [_vector(memory_id=memory_id)],
        idempotency_key="embed-once",
    )

    sql, parameters = connection.calls[0]
    assert "FROM memories AS m" in sql
    assert "m.tenant_id = %s" in sql
    assert "m.project_id = %s" in sql
    assert "m.repository_id = %s" in sql
    assert "ON CONFLICT (memory_id, model) DO UPDATE" in sql
    assert parameters[3:] == (
        UUID(memory_id),
        actor.tenant_id,
        actor.project_id,
        actor.repository_id,
    )
    projection_sql, projection_parameters = connection.calls[2]
    assert "UPSERT INTO retrieval_vectors_1024" in projection_sql
    assert "projection_signature" in projection_sql
    assert "resource_version" in projection_sql
    assert "content_sha256" in projection_sql
    assert projection_parameters[0:3] == (
        actor.tenant_id,
        actor.project_id,
        actor.repository_id,
    )
    assert projection_parameters[5] == f"repository:{actor.repository_id}"


@pytest.mark.asyncio
async def test_vector_upsert_reports_out_of_scope_memory_as_not_found(
    scope_ids: dict[str, str],
) -> None:
    with pytest.raises(ResourceNotFound):
        await _store(_RecordingConnection(upsert_rowcount=0)).upsert_embeddings(
            make_actor(scope_ids),
            [_vector(memory_id=new_id())],
            idempotency_key="embed-missing",
        )


@pytest.mark.asyncio
async def test_vector_search_filters_tenant_project_repository_and_model(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    connection = _RecordingConnection()
    query = tuple([1.0] + [0.0] * (COCKROACH_VECTOR_DIMENSIONS - 1))

    matches = await _store(connection).search_embeddings(
        actor,
        query,
        model="amazon.titan-embed-text-v2:0",
        limit=7,
        min_score=0.5,
    )

    sql, parameters = connection.calls[0]
    assert "tenant_id = %s" in sql
    assert "project_id = %s" in sql
    assert "repository_id = %s" in sql
    assert "model = %s" in sql
    assert "ORDER BY embedding <=> %s::VECTOR" in sql
    assert parameters[1:5] == (
        actor.tenant_id,
        actor.project_id,
        actor.repository_id,
        "amazon.titan-embed-text-v2:0",
    )
    assert len(matches) == 1
    assert matches[0].score == pytest.approx(0.875)


@pytest.mark.asyncio
async def test_cockroach_adapter_rejects_non_1024_vectors_before_database_access(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    connection = _RecordingConnection()
    store = _store(connection)

    with pytest.raises(ValueError, match="1024 dimensions"):
        await store.upsert_embeddings(
            actor,
            [_vector(memory_id=new_id(), dimensions=4)],
            idempotency_key="wrong-width",
        )
    with pytest.raises(ValueError, match="1024 dimensions"):
        await store.search_embeddings(
            actor,
            (1.0, 0.0, 0.0, 0.0),
            model="amazon.titan-embed-text-v2:0",
        )

    assert connection.calls == []
