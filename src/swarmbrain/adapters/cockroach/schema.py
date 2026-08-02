from __future__ import annotations

from importlib.resources import files


def read_schema() -> str:
    """Return the versioned CockroachDB v0 schema without applying it."""

    return files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")
