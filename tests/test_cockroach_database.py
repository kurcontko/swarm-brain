from __future__ import annotations

from typing import Any

import pytest

from swarmbrain.adapters.cockroach.database import CockroachDatabase, SchemaNotInstalled


class _UndefinedTable(RuntimeError):
    sqlstate = "42P01"


class _Connection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, statement: str, _params: Any = None) -> Any:
        self.statements.append(statement)
        raise _UndefinedTable("schema version table is absent")


class _Lease:
    def __init__(self, connection: _Connection) -> None:
        self.connection_value = connection

    async def __aenter__(self) -> _Connection:
        return self.connection_value

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Pool:
    def __init__(self) -> None:
        self.connection_value = _Connection()

    def connection(self) -> _Lease:
        return _Lease(self.connection_value)


@pytest.mark.asyncio
async def test_startup_detects_missing_schema_without_running_ddl() -> None:
    pool = _Pool()
    database = CockroachDatabase("postgresql://unused", pool=pool)

    with pytest.raises(SchemaNotInstalled, match="swarmbrain-schema install"):
        await database.start()

    normalized = " ".join(pool.connection_value.statements).upper()
    assert "SELECT" in normalized
    assert all(keyword not in normalized for keyword in ("CREATE ", "ALTER ", "INSERT ", "UPSERT "))
