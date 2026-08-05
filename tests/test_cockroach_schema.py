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
    "memory_vector_embeddings",
    "evidence",
    "memory_evidence",
    "memory_links",
    "memory_conflicts",
    "idempotency_records",
    "action_attempts",
    "outbox_events",
    "outbox_work_items",
    "outbox_work_attempts",
    "outbox_work_effects",
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
    assert "UNIQUE (tenant_id, run_id, actor_id, operation, idempotency_key)" in schema
    assert "visibility != 'task' OR task_id IS NOT NULL" in schema
    assert "recorded_to IS NULL" in schema
    assert "supersedes_id" in schema
    assert "superseded_by_id" in schema


def test_memory_schema_keeps_lifecycle_strict_but_semantics_open() -> None:
    schema = read_schema()
    blocks = _create_table_blocks(schema)

    assert "content_json JSONB NULL" in blocks["memories"]
    assert "memories_state_check" in blocks["memories"]
    assert "memories_visibility_check" in blocks["memories"]
    assert "memories_kind_check CHECK" not in blocks["memories"]
    assert "sources_type_check CHECK" not in blocks["sources"]
    assert "evidence_kind_check CHECK" not in blocks["evidence"]
    assert "memory_links_type_check CHECK" not in blocks["memory_links"]
    assert "DROP INDEX IF EXISTS memories_current_identity" in schema
    assert "CREATE INDEX IF NOT EXISTS memories_current_fingerprint" in schema
    assert "CREATE UNIQUE INDEX IF NOT EXISTS memories_current_fingerprint" not in schema


def test_vector_schema_is_additive_fixed_width_and_fully_scope_prefixed() -> None:
    schema = read_schema()
    blocks = _create_table_blocks(schema)

    assert "embedding FLOAT8[] NOT NULL" in blocks["memory_embeddings"]
    vectors = blocks["memory_vector_embeddings"]
    assert "tenant_id STRING NOT NULL" in vectors
    assert "project_id STRING NOT NULL" in vectors
    assert "repository_id STRING NOT NULL" in vectors
    assert "model STRING(255) NOT NULL" in vectors
    assert "embedding VECTOR(1024) NOT NULL" in vectors
    assert "CHECK (dimensions = 1024)" in vectors
    assert "CREATE VECTOR INDEX IF NOT EXISTS memory_vector_embeddings_ann" in schema
    assert "tenant_id, project_id, repository_id, model, embedding vector_cosine_ops" in " ".join(
        schema.split()
    )


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
    assert "WHERE status IN ('pending', 'publishing', 'failed')" in schema
    assert "aggregate_type, aggregate_id, aggregate_version" in schema


def test_external_work_queue_is_separate_leased_fenced_and_idempotent() -> None:
    schema = read_schema()
    blocks = _create_table_blocks(schema)

    work = blocks["outbox_work_items"]
    assert "lease_token UUID" in work
    assert "lease_version INT8 NOT NULL" in work
    assert "locked_until TIMESTAMPTZ" in work
    assert "UNIQUE (tenant_id, run_id, kind, dedupe_key)" in work
    assert "status != 'leased'" in work
    assert "outbox_work_items_claim" in schema
    assert "outbox_work_items_expired_leases" in schema
    assert "WHERE status = 'leased'" in schema
    assert "PRIMARY KEY (work_id, attempt, stage)" in blocks["outbox_work_attempts"]
    assert "PRIMARY KEY (work_id, effect_key)" in blocks["outbox_work_effects"]


def test_schema_avoids_sequential_primary_key_antipatterns() -> None:
    schema = read_schema().upper()

    assert "AUTO_INCREMENT" not in schema
    assert re.search(r"\bSERIAL\b", schema) is None
    assert "SELECT *" not in schema
