from __future__ import annotations

import asyncio
import hashlib
from importlib.resources import files
from typing import Any


def read_schema() -> str:
    """Return the versioned CockroachDB schema without applying it."""

    return files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")


def split_sql_statements(script: str) -> tuple[str, ...]:
    """Split the checked-in DDL without treating quoted semicolons as boundaries."""

    statements: list[str] = []
    buffer: list[str] = []
    single_quoted = False
    double_quoted = False
    line_comment = False
    block_comment = False
    index = 0
    while index < len(script):
        char = script[index]
        next_char = script[index + 1] if index + 1 < len(script) else ""

        if line_comment:
            buffer.append(char)
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            buffer.append(char)
            if char == "*" and next_char == "/":
                buffer.append(next_char)
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if not single_quoted and not double_quoted and char == "-" and next_char == "-":
            buffer.extend((char, next_char))
            line_comment = True
            index += 2
            continue
        if not single_quoted and not double_quoted and char == "/" and next_char == "*":
            buffer.extend((char, next_char))
            block_comment = True
            index += 2
            continue
        if char == "'" and not double_quoted:
            buffer.append(char)
            if single_quoted and next_char == "'":
                buffer.append(next_char)
                index += 2
                continue
            single_quoted = not single_quoted
            index += 1
            continue
        if char == '"' and not single_quoted:
            buffer.append(char)
            if double_quoted and next_char == '"':
                buffer.append(next_char)
                index += 2
                continue
            double_quoted = not double_quoted
            index += 1
            continue
        if char == ";" and not single_quoted and not double_quoted:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer.clear()
            index += 1
            continue
        buffer.append(char)
        index += 1

    remainder = "".join(buffer).strip()
    if remainder:
        statements.append(remainder)
    if single_quoted or double_quoted or block_comment:
        raise ValueError("schema contains an unterminated quote or block comment")
    return tuple(statements)


async def install_schema(database_url: str) -> int:
    """Apply schema with pre-v7 writers quiesced as an explicit operator action."""

    from psycopg import AsyncConnection
    from psycopg.rows import dict_row

    from swarmbrain.retrieval.projection import RETRIEVAL_PROJECTION_ID

    from .database import (
        REQUIRED_COLUMNS,
        REQUIRED_INDEXES,
        REQUIRED_RETRIEVAL_INDEX_SHAPES,
        SCHEMA_VERSION,
        incompatible_retrieval_schema_objects,
    )
    from .memory_mappers import memory_from_row
    from .retrieval import upsert_memory_retrieval_projection

    statements = split_sql_statements(read_schema())
    connection = await AsyncConnection.connect(
        database_url,
        autocommit=True,
        row_factory=dict_row,
    )
    try:
        for statement in statements:
            await connection.execute(statement)

        # SQL supplies a minimal crash-safe overlay, then this authoritative
        # application projector makes pre-v7 rows byte-for-byte equivalent to
        # new writes (NFKC/casefold, metadata identifiers, and bounded lookup).
        # One SERIALIZABLE transaction gives the scan, per-memory replacement,
        # and stale-row sweep a common snapshot. Concurrent v7 writes either
        # keep their synchronous projection or force a bounded full retry.
        # Pre-v7 writers do not maintain this projection and must be quiesced
        # for the deployment barrier documented in README and schema docs.
        async def rebuild_projection() -> None:
            last_id = None
            while True:
                where = "" if last_id is None else "WHERE m.id > %s"
                parameters: tuple[Any, ...] = () if last_id is None else (last_id,)
                cursor = await connection.execute(
                    f"""
                    SELECT
                        m.id, m.tenant_id, m.project_id, m.repository_id,
                        m.swarm_id, m.run_id, m.task_id, m.agent_id, m.kind,
                        m.state, m.visibility, m.content, m.content_json, m.title,
                        m.tags, m.confidence, m.valid_from, m.valid_to,
                        m.recorded_from, m.recorded_to, m.supersedes_id,
                        m.superseded_by_id, m.version, m.metadata
                    FROM memories AS m
                    {where}
                    ORDER BY m.id
                    LIMIT 500
                    """,
                    parameters,
                )
                rows = await cursor.fetchall()
                if not rows:
                    break
                for row in rows:
                    await upsert_memory_retrieval_projection(
                        connection,
                        memory_from_row(row),
                    )
                last_id = rows[-1]["id"]

            await connection.execute(
                """
                DELETE FROM retrieval_documents AS d
                WHERE d.projection_id = %s
                  AND d.resource_type = 'memory'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM memories AS m
                      WHERE m.id = d.resource_id
                        AND m.tenant_id = d.tenant_id
                        AND m.project_id = d.project_id
                        AND m.repository_id = d.repository_id
                        AND m.version = d.resource_version
                        AND d.canonical_id = m.id
                        AND d.scope_key = CASE m.visibility
                            WHEN 'repository' THEN concat('repository:', m.repository_id)
                            WHEN 'run' THEN concat('run:', m.run_id)
                            ELSE concat('task:', m.task_id::STRING)
                        END
                  )
                """,
                (RETRIEVAL_PROJECTION_ID,),
            )
            await connection.execute(
                """
                DELETE FROM retrieval_exact_terms AS t
                WHERE t.projection_id = %s
                  AND t.resource_type = 'memory'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM retrieval_documents AS d
                      WHERE d.tenant_id = t.tenant_id
                        AND d.project_id = t.project_id
                        AND d.repository_id = t.repository_id
                        AND d.projection_id = t.projection_id
                        AND d.scope_key = t.scope_key
                        AND d.resource_type = t.resource_type
                        AND d.resource_id = t.resource_id
                        AND d.resource_version = t.resource_version
                  )
                """,
                (RETRIEVAL_PROJECTION_ID,),
            )

        for attempt in range(1, 6):
            try:
                async with connection.transaction():
                    await rebuild_projection()
                break
            except Exception as exc:
                if getattr(exc, "sqlstate", None) != "40001" or attempt == 5:
                    raise
                await asyncio.sleep(0.025 * (2 ** (attempt - 1)))
        columns_cursor = await connection.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s::STRING[])
            """,
            (sorted(REQUIRED_COLUMNS),),
        )
        column_rows = await columns_cursor.fetchall()
        present: dict[str, set[str]] = {}
        for row in column_rows:
            present.setdefault(str(row["table_name"]), set()).add(str(row["column_name"]))
        missing = {
            table: sorted(columns - present.get(table, set()))
            for table, columns in REQUIRED_COLUMNS.items()
            if not columns.issubset(present.get(table, set()))
        }
        if missing:
            details = ", ".join(
                f"{table}({', '.join(columns)})" for table, columns in sorted(missing.items())
            )
            raise RuntimeError(f"schema installation left incompatible objects: {details}")
        indexes_cursor = await connection.execute(
            """
            SELECT indexname, indexdef
            FROM pg_catalog.pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = ANY(%s::STRING[])
            """,
            (sorted(REQUIRED_INDEXES | REQUIRED_RETRIEVAL_INDEX_SHAPES.keys()),),
        )
        index_rows = await indexes_cursor.fetchall()
        present_indexes = {str(row["indexname"]) for row in index_rows}
        missing_indexes = sorted(REQUIRED_INDEXES - present_indexes)
        if missing_indexes:
            raise RuntimeError(
                "schema installation left missing indexes: " + ", ".join(missing_indexes)
            )
        create_cursor = await connection.execute("SHOW CREATE TABLE retrieval_documents")
        create_row = await create_cursor.fetchone()
        retrieval_ddl = "" if create_row is None else str(create_row["create_statement"])
        incompatible_objects = incompatible_retrieval_schema_objects(
            index_rows,
            retrieval_ddl,
        )
        if incompatible_objects:
            raise RuntimeError(
                "schema installation left incompatible retrieval objects: "
                + ", ".join(incompatible_objects)
            )
        digest = hashlib.sha256(read_schema().encode("utf-8")).digest()
        await connection.execute(
            """
            UPSERT INTO swarmbrain_schema_versions (
                version, description, schema_sha256, installed_at
            ) VALUES (%s, %s, %s, now())
            """,
            (
                SCHEMA_VERSION,
                "Swarm Brain v7 exact, FTS simple, trigram, and scoped vector retrieval",
                digest,
            ),
        )
    finally:
        await connection.close()
    return len(statements)


async def verify_schema(database_url: str) -> None:
    """Verify schema presence without running DDL."""

    from .database import CockroachDatabase

    database = CockroachDatabase(database_url, min_size=1, max_size=1)
    try:
        await database.start(verify_schema=True)
    finally:
        await database.close()


__all__ = ["install_schema", "read_schema", "split_sql_statements", "verify_schema"]
