from __future__ import annotations

import re

from swarmbrain.adapters.cockroach import read_schema
from swarmbrain.adapters.cockroach.database import incompatible_retrieval_schema_objects

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
    "retrieval_vectors_1024",
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
        r"CREATE TABLE IF NOT EXISTS\s+(?P<name>[a-z0-9_]+)\s*\((?P<body>.*?)\n\);",
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


def test_graph_retrieval_has_relation_prefixed_directional_covering_indexes() -> None:
    normalized = " ".join(read_schema().split())

    assert (
        "CREATE INDEX IF NOT EXISTS memory_links_from_type ON memory_links "
        "(source_memory_id, link_type, created_at DESC, id) "
        "STORING (target_memory_id, reason, metadata)"
    ) in normalized
    assert (
        "CREATE INDEX IF NOT EXISTS memory_links_to_type ON memory_links "
        "(target_memory_id, link_type, created_at DESC, id) "
        "STORING (source_memory_id, reason, metadata)"
    ) in normalized


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


def test_retrieval_v8_dense_projection_is_versioned_signed_and_scope_keyed() -> None:
    schema = read_schema()
    vectors = _create_table_blocks(schema)["retrieval_vectors_1024"]
    normalized = " ".join(schema.split())

    for field in (
        "projection_id STRING NOT NULL",
        "projection_signature STRING NOT NULL",
        "scope_key STRING NOT NULL",
        "resource_version INT8 NOT NULL",
        "canonical_id UUID NOT NULL",
        "domain_lane STRING NOT NULL",
        "content_sha256 BYTES NOT NULL",
        "model STRING(255) NOT NULL",
        "embedding VECTOR(1024) NOT NULL",
    ):
        assert field in vectors
    assert "CHECK (dimensions = 1024)" in vectors
    assert "CREATE VECTOR INDEX IF NOT EXISTS retrieval_vectors_1024_ann_v2" in schema
    assert (
        "tenant_id, project_id, repository_id, resource_type, projection_id, "
        "projection_signature, scope_key, embedding vector_cosine_ops"
    ) in normalized
    assert "memory-content-v1:current:cosine" in schema
    assert "dense-v2|memory-content-v1:current:cosine|normalization=provider|" in schema
    assert "truncation=provider|dimensions=" in schema


def test_retrieval_v7_has_scoped_simple_fts_trigram_and_exact_projections() -> None:
    schema = read_schema()
    blocks = _create_table_blocks(schema)

    documents = blocks["retrieval_documents"]
    assert "search_text STRING NOT NULL" in documents
    assert "lookup_text STRING NOT NULL" in documents
    assert "to_tsvector('simple', search_text)" in documents
    assert "resource_version INT8 NOT NULL" in documents
    assert "retrieval_documents_fts" in schema
    assert "tenant_id, project_id, repository_id, projection_id, scope_key, search_tsv" in " ".join(
        schema.split()
    )
    assert "retrieval_documents_lookup_trgm" in schema
    assert "lookup_text gin_trgm_ops" in schema
    assert "retrieval_exact_terms_by_resource" in schema
    assert "websearch_to_tsquery" not in schema


def test_retrieval_verifier_rejects_same_named_wrong_index_definition() -> None:
    index_rows = [
        {
            "indexname": "retrieval_documents_fts",
            "indexdef": "CREATE INDEX retrieval_documents_fts ON retrieval_documents "
            "USING gin (tenant_id, project_id, repository_id, projection_id, scope_key, "
            "search_tsv)",
        },
        {
            "indexname": "retrieval_documents_lookup_trgm",
            "indexdef": "CREATE INDEX retrieval_documents_lookup_trgm ON retrieval_documents "
            "USING btree (tenant_id)",
        },
        {
            "indexname": "retrieval_exact_terms_pkey",
            "indexdef": "CREATE UNIQUE INDEX retrieval_exact_terms_pkey ON "
            "retrieval_exact_terms USING btree (tenant_id, project_id, repository_id, "
            "projection_id, scope_key, normalized_term, term_kind, resource_type, resource_id)",
        },
        {
            "indexname": "retrieval_exact_terms_by_resource",
            "indexdef": "CREATE INDEX retrieval_exact_terms_by_resource ON "
            "retrieval_exact_terms USING btree (tenant_id, project_id, repository_id, "
            "projection_id, scope_key, resource_type, resource_id)",
        },
        {
            "indexname": "memory_links_from_type",
            "indexdef": "CREATE INDEX memory_links_from_type ON memory_links USING btree "
            "(source_memory_id, link_type, created_at DESC, id) "
            "STORING (target_memory_id, reason, metadata)",
        },
        {
            "indexname": "memory_links_to_type",
            "indexdef": "CREATE INDEX memory_links_to_type ON memory_links USING btree "
            "(target_memory_id, link_type, created_at DESC, id) "
            "STORING (source_memory_id, reason, metadata)",
        },
    ]
    ddl = """
        CREATE TABLE retrieval_documents (
            tenant_id STRING NOT NULL, project_id STRING NOT NULL,
            repository_id STRING NOT NULL, projection_id STRING NOT NULL,
            scope_key STRING NOT NULL, resource_type STRING NOT NULL,
            resource_id UUID NOT NULL, search_text STRING NOT NULL,
            lookup_text STRING NOT NULL,
            search_tsv TSVECTOR AS (to_tsvector('simple', search_text)) STORED,
            PRIMARY KEY (tenant_id, project_id, repository_id, projection_id,
                         scope_key, resource_type, resource_id)
        )
    """

    assert incompatible_retrieval_schema_objects(index_rows, ddl) == (
        "retrieval_documents_lookup_trgm",
    )

    wrong_graph = {
        "indexname": "memory_links_from_type",
        "indexdef": "CREATE INDEX memory_links_from_type ON memory_links USING btree "
        "(source_memory_id, created_at, link_type, id)",
    }
    graph_rows = [
        wrong_graph if row["indexname"] == "memory_links_from_type" else row for row in index_rows
    ]
    assert incompatible_retrieval_schema_objects(graph_rows, ddl) == (
        "retrieval_documents_lookup_trgm",
        "memory_links_from_type",
    )

    dense_ddl = """
        CREATE TABLE retrieval_vectors_1024 (
            tenant_id STRING NOT NULL, project_id STRING NOT NULL,
            repository_id STRING NOT NULL, projection_id STRING NOT NULL,
            projection_signature STRING NOT NULL, scope_key STRING NOT NULL,
            resource_type STRING NOT NULL, resource_id UUID NOT NULL,
            resource_version INT8 NOT NULL, content_sha256 BYTES NOT NULL,
            embedding VECTOR(1024) NOT NULL,
            PRIMARY KEY (tenant_id, project_id, repository_id, projection_signature,
                         scope_key, resource_type, resource_id)
        )
    """
    wrong_dense = {
        "indexname": "retrieval_vectors_1024_ann_v2",
        "indexdef": "CREATE INDEX retrieval_vectors_1024_ann_v2 ON retrieval_vectors_1024 "
        "USING btree (tenant_id, embedding)",
    }
    assert incompatible_retrieval_schema_objects(
        [*index_rows, wrong_dense],
        ddl,
        dense_ddl,
    ) == (
        "retrieval_documents_lookup_trgm",
        "retrieval_vectors_1024_ann_v2",
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
