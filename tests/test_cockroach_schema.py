from __future__ import annotations

import re

from swarmbrain.adapters.cockroach import read_schema

REQUIRED_TABLES = {
    "agents",
    "agent_tokens",
    "runs",
    "tasks",
    "task_dependencies",
    "task_leases",
    "task_checkpoints",
    "task_completions",
    "memories",
    "memory_embeddings",
    "evidence",
    "memory_evidence",
    "memory_links",
    "memory_conflicts",
    "idempotency_records",
    "action_attempts",
    "outbox_events",
    "swarm_events",
}


def _create_table_blocks(schema: str) -> dict[str, str]:
    pattern = re.compile(
        r"CREATE TABLE IF NOT EXISTS\s+(?P<name>[a-z_]+)\s*\((?P<body>.*?)\n\);",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return {match.group("name").lower(): match.group("body") for match in pattern.finditer(schema)}


def test_schema_contains_required_tables_and_explicit_primary_keys() -> None:
    blocks = _create_table_blocks(read_schema())

    assert blocks.keys() >= REQUIRED_TABLES
    missing_primary_keys = [
        table for table, block in blocks.items() if "PRIMARY KEY" not in block.upper()
    ]
    assert missing_primary_keys == []


def test_schema_encodes_lease_and_idempotency_invariants() -> None:
    schema = read_schema()

    assert "CREATE UNIQUE INDEX IF NOT EXISTS task_leases_one_active_per_task" in schema
    assert "WHERE status = 'active'" in schema
    assert "UNIQUE (tenant_id, actor_id, operation, idempotency_key)" in schema
    assert "visibility != 'task' OR task_id IS NOT NULL" in schema
    assert "recorded_to IS NULL" in schema
    assert "supersedes_id" in schema
    assert "superseded_by_id" in schema


def test_schema_persists_domain_fencing_provenance_and_structured_resolution() -> None:
    blocks = _create_table_blocks(read_schema())

    assert "status STRING NOT NULL" in blocks["agents"]
    assert "version INT8 NOT NULL" in blocks["agents"]
    assert "kind STRING NOT NULL" in blocks["task_dependencies"]
    assert "reviewed_by_agent_id STRING" in blocks["sources"]
    assert "rejection_reason STRING" in blocks["sources"]
    assert "version INT8 NOT NULL" in blocks["sources"]
    assert "resolution JSONB" in blocks["memory_conflicts"]
    assert "aggregate_version INT8 NOT NULL" in blocks["outbox_events"]
    assert "locked_until TIMESTAMPTZ" in blocks["outbox_events"]
    assert "project_id STRING NOT NULL" in blocks["swarm_events"]
    assert "recorded_at TIMESTAMPTZ NOT NULL" in blocks["swarm_events"]


def test_claim_and_event_indexes_cover_contract_filters() -> None:
    schema = read_schema()

    assert "STORING (required_capabilities, tags, title, description, version)" in schema
    assert "WHERE status IN ('pending', 'failed')" in schema
    assert "aggregate_type, aggregate_id, aggregate_version" in schema


def test_schema_avoids_sequential_primary_key_antipatterns() -> None:
    schema = read_schema().upper()

    assert "AUTO_INCREMENT" not in schema
    assert re.search(r"\bSERIAL\b", schema) is None
    assert "SELECT *" not in schema
