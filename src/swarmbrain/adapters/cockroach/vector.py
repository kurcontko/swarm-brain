"""Shared fixed-width CockroachDB vector encoding."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any
from uuid import UUID

from swarmbrain.domain.memory import EmbeddingVector, Visibility
from swarmbrain.retrieval.dense import DENSE_PROJECTION_ID, dense_projection_signature
from swarmbrain.retrieval.projection import domain_lane, projection_scope_key

COCKROACH_VECTOR_DIMENSIONS = 1024


def vector_literal(values: Sequence[float]) -> str:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != COCKROACH_VECTOR_DIMENSIONS:
        raise ValueError(
            f"CockroachDB vector index requires {COCKROACH_VECTOR_DIMENSIONS} dimensions"
        )
    if any(not isfinite(value) for value in normalized):
        raise ValueError("embedding values must be finite")
    return "[" + ",".join(repr(value) for value in normalized) + "]"


async def upsert_memory_dense_projection(
    connection: Any,
    vector: EmbeddingVector,
    *,
    tenant_id: str,
    project_id: str,
    repository_id: str,
    swarm_id: str | None = None,
    run_id: str | None = None,
) -> bool:
    """Derive and atomically upsert all authoritative dense projection fields."""

    literal = vector_literal(vector.values)
    clauses = [
        "id = %s",
        "tenant_id = %s",
        "project_id = %s",
        "repository_id = %s",
    ]
    parameters: list[Any] = [
        UUID(vector.memory_id),
        tenant_id,
        project_id,
        repository_id,
    ]
    if swarm_id is not None:
        clauses.append("swarm_id = %s")
        parameters.append(swarm_id)
    if run_id is not None:
        clauses.append("run_id = %s")
        parameters.append(run_id)
    cursor = await connection.execute(
        f"""
        SELECT id, repository_id, run_id, task_id, visibility, kind, metadata,
               version, normalized_sha256
        FROM memories
        WHERE {" AND ".join(clauses)}
        """,
        tuple(parameters),
    )
    row = await cursor.fetchone()
    if row is None:
        return False

    visibility = Visibility(str(row["visibility"]))
    scope_key = projection_scope_key(
        visibility,
        repository_id=str(row["repository_id"]),
        run_id=str(row["run_id"]),
        task_id=None if row["task_id"] is None else str(row["task_id"]),
    )
    signature = dense_projection_signature(vector.model, vector.dimensions)
    await connection.execute(
        """
        UPSERT INTO retrieval_vectors_1024 (
            tenant_id, project_id, repository_id, projection_id,
            projection_signature, scope_key, resource_type, resource_id,
            resource_version, canonical_id, domain_lane, content_sha256,
            model, dimensions, embedding, indexed_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, 'memory', %s, %s, %s, %s,
            %s, %s, %s, %s::VECTOR, now()
        )
        """,
        (
            tenant_id,
            project_id,
            repository_id,
            DENSE_PROJECTION_ID,
            signature,
            scope_key,
            row["id"],
            int(row["version"]),
            row["id"],
            domain_lane(str(row["kind"]), row.get("metadata") or {}),
            row["normalized_sha256"],
            vector.model.strip(),
            vector.dimensions,
            literal,
        ),
    )
    return True


__all__ = [
    "COCKROACH_VECTOR_DIMENSIONS",
    "upsert_memory_dense_projection",
    "vector_literal",
]
