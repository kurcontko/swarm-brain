from __future__ import annotations

import hashlib
from importlib.resources import files


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
    """Apply the idempotent schema as an explicit operator action."""

    from psycopg import AsyncConnection
    from psycopg.rows import dict_row

    from .database import REQUIRED_COLUMNS, REQUIRED_INDEXES, SCHEMA_VERSION

    statements = split_sql_statements(read_schema())
    connection = await AsyncConnection.connect(
        database_url,
        autocommit=True,
        row_factory=dict_row,
    )
    try:
        for statement in statements:
            await connection.execute(statement)
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
            SELECT indexname
            FROM pg_catalog.pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = ANY(%s::STRING[])
            """,
            (sorted(REQUIRED_INDEXES),),
        )
        present_indexes = {str(row["indexname"]) for row in await indexes_cursor.fetchall()}
        missing_indexes = sorted(REQUIRED_INDEXES - present_indexes)
        if missing_indexes:
            raise RuntimeError(
                "schema installation left missing indexes: " + ", ".join(missing_indexes)
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
                "Swarm Brain flexible memory with scoped VECTOR(1024) semantic recall",
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
