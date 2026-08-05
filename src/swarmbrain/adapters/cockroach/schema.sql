-- Swarm Brain v0 CockroachDB schema.
-- Applying this file is an explicit operator action; application startup never
-- runs DDL. All mutation paths use SERIALIZABLE transactions and an outbox row.

CREATE TABLE IF NOT EXISTS swarmbrain_schema_versions (
    version INT8 PRIMARY KEY,
    description STRING NOT NULL,
    schema_sha256 BYTES NOT NULL,
    installed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    tenant_id STRING NOT NULL,
    id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    state STRING NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL,
    CONSTRAINT runs_state_check CHECK (state IN ('active', 'completed', 'cancelled')),
    PRIMARY KEY (tenant_id, id)
);

CREATE TABLE IF NOT EXISTS agents (
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    harness STRING NOT NULL,
    provider STRING NOT NULL,
    model STRING NOT NULL,
    status STRING NOT NULL DEFAULT 'active',
    capabilities STRING[] NOT NULL DEFAULT ARRAY[]::STRING[],
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    version INT8 NOT NULL DEFAULT 1,
    CONSTRAINT agents_status_check CHECK (status IN ('active', 'disconnected', 'revoked')),
    PRIMARY KEY (tenant_id, run_id, id),
    CONSTRAINT agents_run_fk FOREIGN KEY (tenant_id, run_id)
        REFERENCES runs (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    agent_id STRING NOT NULL,
    token_hash BYTES NOT NULL UNIQUE,
    capabilities STRING[] NOT NULL DEFAULT ARRAY[]::STRING[],
    issued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    CONSTRAINT agent_tokens_agent_fk FOREIGN KEY (tenant_id, run_id, agent_id)
        REFERENCES agents (tenant_id, run_id, id) ON DELETE CASCADE,
    CONSTRAINT agent_tokens_expiry_check CHECK (expires_at > issued_at)
);

CREATE INDEX IF NOT EXISTS agent_tokens_active_lookup
    ON agent_tokens (tenant_id, token_hash)
    STORING (run_id, agent_id, capabilities, expires_at, revoked_at);

CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    run_id STRING NOT NULL,
    title STRING NOT NULL,
    description STRING NOT NULL DEFAULT '',
    state STRING NOT NULL DEFAULT 'ready',
    priority INT8 NOT NULL DEFAULT 0,
    version INT8 NOT NULL DEFAULT 1,
    required_capabilities STRING[] NOT NULL DEFAULT ARRAY[]::STRING[],
    tags STRING[] NOT NULL DEFAULT ARRAY[]::STRING[],
    created_by_agent_id STRING NULL,
    claimed_by_agent_id STRING NULL,
    active_lease_id UUID NULL,
    available_at TIMESTAMPTZ NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL,
    CONSTRAINT tasks_state_check CHECK (
        state IN ('pending', 'blocked', 'ready', 'claimed', 'completed', 'failed', 'cancelled')
    ),
    CONSTRAINT tasks_run_fk FOREIGN KEY (tenant_id, run_id)
        REFERENCES runs (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS tasks_claim_queue
    ON tasks (tenant_id, run_id, state, priority DESC, created_at, id)
    STORING (required_capabilities, tags, title, description, version);

CREATE TABLE IF NOT EXISTS task_dependencies (
    task_id UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    depends_on_task_id UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    kind STRING NOT NULL DEFAULT 'blocks',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT task_dependencies_no_self CHECK (task_id != depends_on_task_id),
    CONSTRAINT task_dependencies_kind_check CHECK (kind IN ('blocks', 'related')),
    PRIMARY KEY (task_id, depends_on_task_id)
);

CREATE INDEX IF NOT EXISTS task_dependencies_reverse
    ON task_dependencies (depends_on_task_id)
    STORING (task_id);

CREATE TABLE IF NOT EXISTS task_leases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    agent_id STRING NOT NULL,
    status STRING NOT NULL DEFAULT 'active',
    version INT8 NOT NULL DEFAULT 1,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    renewed_at TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    released_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    CONSTRAINT task_leases_status_check CHECK (
        status IN ('active', 'expired', 'released', 'completed')
    ),
    CONSTRAINT task_leases_expiry_check CHECK (expires_at > claimed_at),
    CONSTRAINT task_leases_agent_fk FOREIGN KEY (tenant_id, run_id, agent_id)
        REFERENCES agents (tenant_id, run_id, id) ON DELETE RESTRICT
);

-- Expired rows are first transitioned to status='expired' inside the claim
-- transaction. This partial unique index then prevents two current owners even
-- when concurrent claim transactions race.
CREATE UNIQUE INDEX IF NOT EXISTS task_leases_one_active_per_task
    ON task_leases (task_id)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS task_leases_agent_active
    ON task_leases (tenant_id, run_id, agent_id, status, expires_at)
    STORING (task_id, version);

CREATE TABLE IF NOT EXISTS task_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    lease_id UUID NOT NULL REFERENCES task_leases (id) ON DELETE RESTRICT,
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    agent_id STRING NOT NULL,
    sequence INT8 NOT NULL,
    summary STRING NOT NULL,
    discoveries JSONB NOT NULL DEFAULT '[]'::JSONB,
    completed_work JSONB NOT NULL DEFAULT '[]'::JSONB,
    remaining_work JSONB NOT NULL DEFAULT '[]'::JSONB,
    memory_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    artifact_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    state JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT task_checkpoints_sequence_check CHECK (sequence > 0),
    UNIQUE (task_id, sequence)
);

CREATE INDEX IF NOT EXISTS task_checkpoints_latest
    ON task_checkpoints (task_id, sequence DESC)
    STORING (
        lease_id, agent_id, summary, discoveries, completed_work,
        remaining_work, memory_ids, artifact_ids, state, created_at
    );

CREATE TABLE IF NOT EXISTS task_completions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    lease_id UUID NOT NULL REFERENCES task_leases (id) ON DELETE RESTRICT,
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    agent_id STRING NOT NULL,
    outcome STRING NOT NULL,
    summary STRING NOT NULL,
    memory_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    artifact_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT task_completions_outcome_check CHECK (outcome IN ('succeeded', 'failed')),
    UNIQUE (task_id)
);

CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    run_id STRING NOT NULL,
    task_id UUID NULL REFERENCES tasks (id) ON DELETE SET NULL,
    agent_id STRING NOT NULL,
    source_type STRING NOT NULL,
    occurrence_key STRING NOT NULL,
    uri STRING NULL,
    content_sha256 BYTES NOT NULL,
    content STRING NULL,
    trust_label STRING NOT NULL DEFAULT 'unknown',
    review_state STRING NOT NULL DEFAULT 'pending',
    valid_at TIMESTAMPTZ NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_by_agent_id STRING NULL,
    reviewed_at TIMESTAMPTZ NULL,
    rejection_reason STRING NULL,
    version INT8 NOT NULL DEFAULT 1,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    CONSTRAINT sources_trust_check CHECK (trust_label IN ('unknown', 'trusted', 'untrusted')),
    CONSTRAINT sources_review_check CHECK (review_state IN ('pending', 'approved', 'rejected')),
    UNIQUE (tenant_id, repository_id, run_id, occurrence_key)
);

-- Semantic labels are an open application vocabulary.  Existing v3 clusters
-- carried a closed allowlist, so installation explicitly removes it.
ALTER TABLE sources DROP CONSTRAINT IF EXISTS sources_type_check;

CREATE INDEX IF NOT EXISTS sources_scope_lookup
    ON sources (tenant_id, project_id, repository_id, swarm_id, run_id, id)
    STORING (
        task_id, agent_id, source_type, occurrence_key, uri, content_sha256,
        trust_label, review_state, valid_at, recorded_at, version
    );

CREATE TABLE IF NOT EXISTS source_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    chunk_index INT8 NOT NULL,
    content STRING NOT NULL,
    token_count INT8 NOT NULL DEFAULT 0,
    char_start INT8 NOT NULL,
    char_end INT8 NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT source_chunks_offsets_check CHECK (
        chunk_index >= 0 AND token_count >= 0 AND char_start >= 0 AND char_end >= char_start
    ),
    UNIQUE (source_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    run_id STRING NOT NULL,
    task_id UUID NULL REFERENCES tasks (id) ON DELETE SET NULL,
    agent_id STRING NOT NULL,
    kind STRING NOT NULL,
    state STRING NOT NULL DEFAULT 'tentative',
    visibility STRING NOT NULL DEFAULT 'run',
    content STRING NOT NULL,
    content_json JSONB NULL,
    title STRING NULL,
    tags STRING[] NOT NULL DEFAULT ARRAY[]::STRING[],
    normalized_sha256 BYTES NOT NULL,
    dedup_scope STRING NOT NULL,
    confidence DECIMAL(5,4) NOT NULL DEFAULT 0.5000,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to TIMESTAMPTZ NULL,
    recorded_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_to TIMESTAMPTZ NULL,
    supersedes_id UUID NULL REFERENCES memories (id) ON DELETE RESTRICT,
    superseded_by_id UUID NULL REFERENCES memories (id) ON DELETE RESTRICT,
    source_id UUID NULL REFERENCES sources (id) ON DELETE SET NULL,
    previous_state STRING NULL,
    policy_reason STRING NOT NULL DEFAULT '',
    policy_confidence DECIMAL(5,4) NOT NULL DEFAULT 1.0000,
    version INT8 NOT NULL DEFAULT 1,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    CONSTRAINT memories_state_check CHECK (
        state IN ('tentative', 'confirmed', 'refuted', 'superseded')
    ),
    CONSTRAINT memories_visibility_check CHECK (visibility IN ('task', 'run', 'repository')),
    CONSTRAINT memories_task_visibility_check CHECK (visibility != 'task' OR task_id IS NOT NULL),
    CONSTRAINT memories_confidence_check CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT memories_policy_confidence_check CHECK (
        policy_confidence >= 0 AND policy_confidence <= 1
    ),
    CONSTRAINT memories_previous_state_check CHECK (
        previous_state IS NULL OR previous_state IN ('tentative', 'confirmed', 'refuted')
    ),
    CONSTRAINT memories_valid_interval_check CHECK (valid_to IS NULL OR valid_to > valid_from),
    CONSTRAINT memories_recorded_interval_check CHECK (
        recorded_to IS NULL OR recorded_to > recorded_from
    ),
    CONSTRAINT memories_supersession_check CHECK (supersedes_id IS NULL OR supersedes_id != id)
);

ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_json JSONB NULL;
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_kind_check;

CREATE INDEX IF NOT EXISTS memories_current_scope
    ON memories (tenant_id, repository_id, run_id, visibility, state, recorded_from DESC, id)
    STORING (task_id, agent_id, kind, confidence, valid_from, valid_to, content)
    WHERE recorded_to IS NULL;

CREATE INDEX IF NOT EXISTS memories_task_current
    ON memories (task_id, state, recorded_from DESC, id)
    STORING (kind, confidence, content, agent_id)
    WHERE task_id IS NOT NULL AND recorded_to IS NULL;

CREATE INDEX IF NOT EXISTS memories_source
    ON memories (source_id)
    STORING (state, recorded_to, supersedes_id, superseded_by_id)
    WHERE source_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS memories_one_successor
    ON memories (supersedes_id)
    WHERE supersedes_id IS NOT NULL;

-- A fingerprint is a retrieval hint, not a uniqueness claim: two agents may
-- independently observe the same content and both observations are preserved.
DROP INDEX IF EXISTS memories_current_identity;

CREATE INDEX IF NOT EXISTS memories_current_fingerprint
    ON memories (tenant_id, repository_id, dedup_scope, kind, normalized_sha256)
    WHERE recorded_to IS NULL AND state != 'refuted';

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    model STRING NOT NULL,
    dimensions INT8 NOT NULL,
    embedding FLOAT8[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_embeddings_dimensions_check CHECK (dimensions > 0),
    PRIMARY KEY (memory_id, model)
);

-- v6 adds the ANN plane additively.  The legacy FLOAT8[] table above remains
-- intact so installing v6 never converts or discards pre-existing vectors.
CREATE TABLE IF NOT EXISTS memory_vector_embeddings (
    memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    tenant_id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    model STRING(255) NOT NULL,
    dimensions INT8 NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_vector_embeddings_dimensions_check CHECK (dimensions = 1024),
    PRIMARY KEY (memory_id, model)
);

-- Every ANN lookup must constrain the complete authenticated repository scope
-- and the embedding model before cosine similarity work is considered.
CREATE VECTOR INDEX IF NOT EXISTS memory_vector_embeddings_ann
    ON memory_vector_embeddings (
        tenant_id, project_id, repository_id, model, embedding vector_cosine_ops
    );

CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    kind STRING NOT NULL,
    locator STRING NULL,
    content_sha256 BYTES NULL,
    excerpt STRING NULL,
    source_id UUID NOT NULL REFERENCES sources (id) ON DELETE RESTRICT,
    artifact JSONB NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE evidence DROP CONSTRAINT IF EXISTS evidence_kind_check;

CREATE INDEX IF NOT EXISTS evidence_source_lookup
    ON evidence (source_id, id)
    STORING (tenant_id, kind, locator, content_sha256, excerpt, artifact, metadata, recorded_at);

CREATE TABLE IF NOT EXISTS memory_evidence (
    memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    evidence_id UUID NOT NULL REFERENCES evidence (id) ON DELETE RESTRICT,
    relation STRING NOT NULL DEFAULT 'supports',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, evidence_id, relation)
);

ALTER TABLE memory_evidence DROP CONSTRAINT IF EXISTS memory_evidence_relation_check;

CREATE INDEX IF NOT EXISTS memory_evidence_by_evidence
    ON memory_evidence (evidence_id, memory_id)
    STORING (relation, created_at);

CREATE TABLE IF NOT EXISTS memory_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    link_type STRING NOT NULL,
    reason STRING NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_links_no_self CHECK (source_memory_id != target_memory_id),
    UNIQUE (source_memory_id, target_memory_id, link_type)
);

ALTER TABLE memory_links DROP CONSTRAINT IF EXISTS memory_links_type_check;

CREATE INDEX IF NOT EXISTS memory_links_from
    ON memory_links (source_memory_id, created_at, id)
    STORING (target_memory_id, link_type, reason, metadata);

CREATE INDEX IF NOT EXISTS memory_links_to
    ON memory_links (target_memory_id, created_at, id)
    STORING (source_memory_id, link_type, reason, metadata);

CREATE TABLE IF NOT EXISTS memory_conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    run_id STRING NOT NULL,
    task_id UUID NULL REFERENCES tasks (id) ON DELETE SET NULL,
    reported_by_agent_id STRING NOT NULL,
    description STRING NOT NULL,
    severity STRING NOT NULL DEFAULT 'medium',
    state STRING NOT NULL DEFAULT 'open',
    resolution JSONB NULL,
    resolved_by_agent_id STRING NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ NULL,
    version INT8 NOT NULL DEFAULT 1,
    CONSTRAINT memory_conflicts_state_check CHECK (state IN ('open', 'resolved', 'dismissed')),
    CONSTRAINT memory_conflicts_severity_check CHECK (
        severity IN ('low', 'medium', 'high', 'critical')
    ),
    CONSTRAINT memory_conflicts_resolution_check CHECK (
        state = 'open' OR resolved_at IS NOT NULL
    )
);

CREATE TABLE IF NOT EXISTS memory_conflict_members (
    conflict_id UUID NOT NULL REFERENCES memory_conflicts (id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    position INT8 NOT NULL,
    PRIMARY KEY (conflict_id, memory_id),
    UNIQUE (conflict_id, position)
);

CREATE TABLE IF NOT EXISTS conflict_evidence (
    conflict_id UUID NOT NULL REFERENCES memory_conflicts (id) ON DELETE CASCADE,
    evidence_id UUID NOT NULL REFERENCES evidence (id) ON DELETE RESTRICT,
    relation STRING NOT NULL DEFAULT 'supports_resolution',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (conflict_id, evidence_id, relation)
);

CREATE TABLE IF NOT EXISTS idempotency_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    actor_id STRING NOT NULL,
    operation STRING NOT NULL,
    idempotency_key STRING NOT NULL,
    request_sha256 BYTES NOT NULL,
    status STRING NOT NULL DEFAULT 'started',
    response_status INT8 NULL,
    response_body JSONB NULL,
    resource_id STRING NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT idempotency_records_status_check CHECK (
        status IN ('started', 'completed', 'failed')
    ),
    UNIQUE (tenant_id, run_id, actor_id, operation, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idempotency_records_expiry
    ON idempotency_records (expires_at, id)
    STORING (status);

CREATE INDEX IF NOT EXISTS idempotency_records_lookup
    ON idempotency_records (tenant_id, run_id, actor_id, operation, idempotency_key)
    STORING (request_sha256, status, response_status, response_body, resource_id, expires_at);

CREATE TABLE IF NOT EXISTS action_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_record_id UUID NULL REFERENCES idempotency_records (id) ON DELETE SET NULL,
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    agent_id STRING NOT NULL,
    operation STRING NOT NULL,
    attempt INT8 NOT NULL,
    status STRING NOT NULL,
    sqlstate STRING NULL,
    error_code STRING NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT action_attempts_attempt_check CHECK (attempt > 0),
    CONSTRAINT action_attempts_status_check CHECK (status IN ('started', 'succeeded', 'failed'))
);

CREATE INDEX IF NOT EXISTS action_attempts_operation
    ON action_attempts (tenant_id, run_id, operation, created_at DESC, id)
    STORING (agent_id, attempt, status, sqlstate, error_code);

CREATE TABLE IF NOT EXISTS outbox_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL UNIQUE,
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    aggregate_type STRING NOT NULL,
    aggregate_id STRING NOT NULL,
    aggregate_version INT8 NOT NULL,
    event_type STRING NOT NULL,
    dedupe_key STRING NOT NULL UNIQUE,
    payload JSONB NOT NULL,
    status STRING NOT NULL DEFAULT 'pending',
    attempts INT8 NOT NULL DEFAULT 0,
    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    locked_until TIMESTAMPTZ NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ NULL,
    last_error STRING NULL,
    version INT8 NOT NULL DEFAULT 1,
    CONSTRAINT outbox_events_status_check CHECK (
        status IN ('pending', 'publishing', 'published', 'failed')
    ),
    CONSTRAINT outbox_events_attempts_check CHECK (attempts >= 0)
);

CREATE INDEX IF NOT EXISTS outbox_events_unpublished
    ON outbox_events (available_at, locked_until, id)
    STORING (
        event_id, tenant_id, run_id, aggregate_type, aggregate_id, aggregate_version,
        event_type, payload, status, version
    )
    WHERE status IN ('pending', 'publishing', 'failed');

CREATE TABLE IF NOT EXISTS source_extractions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    source_id UUID NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    extractor_name STRING NOT NULL,
    extractor_version STRING NOT NULL,
    route STRING NOT NULL,
    status STRING NOT NULL,
    deterministic_candidates JSONB NOT NULL DEFAULT '[]'::JSONB,
    model_candidates JSONB NOT NULL DEFAULT '[]'::JSONB,
    model_name STRING NULL,
    model_version STRING NULL,
    prompt_sha256 BYTES NULL,
    fallback_reason STRING NULL,
    last_error STRING NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    version INT8 NOT NULL DEFAULT 1,
    CONSTRAINT source_extractions_route_check CHECK (
        route IN ('coding', 'general', 'skip')
    ),
    CONSTRAINT source_extractions_status_check CHECK (
        status IN ('completed', 'fallback', 'failed')
    ),
    UNIQUE (source_id, extractor_name, extractor_version)
);

CREATE INDEX IF NOT EXISTS source_extractions_scope
    ON source_extractions (tenant_id, run_id, source_id, updated_at DESC, id)
    STORING (
        extractor_name, extractor_version, route, status,
        deterministic_candidates, model_candidates, model_name, model_version,
        prompt_sha256, fallback_reason, last_error, version
    );

-- External/model work is a separate leased queue from the immutable event
-- relay above. Workers claim and fence rows in a short transaction, perform
-- provider work after COMMIT, then apply validated effects in another short
-- transaction.
CREATE TABLE IF NOT EXISTS outbox_work_items (
    id UUID PRIMARY KEY,
    tenant_id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    run_id STRING NOT NULL,
    task_id UUID NULL REFERENCES tasks (id) ON DELETE SET NULL,
    requested_by_agent_id STRING NOT NULL,
    kind STRING NOT NULL,
    subject_id UUID NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::JSONB,
    dedupe_key STRING NOT NULL,
    priority INT8 NOT NULL DEFAULT 0,
    status STRING NOT NULL DEFAULT 'pending',
    attempts INT8 NOT NULL DEFAULT 0,
    max_attempts INT8 NOT NULL DEFAULT 5,
    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    locked_by STRING NULL,
    lease_token UUID NULL,
    lease_version INT8 NOT NULL DEFAULT 0,
    locked_until TIMESTAMPTZ NULL,
    outcome STRING NULL,
    result JSONB NULL,
    last_error STRING NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL,
    version INT8 NOT NULL DEFAULT 1,
    CONSTRAINT outbox_work_items_run_fk FOREIGN KEY (tenant_id, run_id)
        REFERENCES runs (tenant_id, id) ON DELETE CASCADE,
    CONSTRAINT outbox_work_items_kind_check CHECK (
        kind IN ('extract_source', 'embed_memory', 'persist_artifact')
    ),
    CONSTRAINT outbox_work_items_status_check CHECK (
        status IN ('pending', 'leased', 'retry', 'completed', 'failed', 'cancelled')
    ),
    CONSTRAINT outbox_work_items_attempts_check CHECK (
        attempts >= 0 AND attempts <= max_attempts AND max_attempts > 0
    ),
    CONSTRAINT outbox_work_items_versions_check CHECK (lease_version >= 0 AND version > 0),
    CONSTRAINT outbox_work_items_lease_check CHECK (
        status != 'leased'
        OR (locked_by IS NOT NULL AND lease_token IS NOT NULL AND locked_until IS NOT NULL)
    ),
    UNIQUE (tenant_id, run_id, kind, dedupe_key)
);

CREATE INDEX IF NOT EXISTS outbox_work_items_claim
    ON outbox_work_items (priority DESC, available_at, created_at, id)
    STORING (
        kind, status, attempts, max_attempts, locked_until, subject_id,
        lease_version, version
    )
    WHERE status IN ('pending', 'retry', 'leased');

CREATE INDEX IF NOT EXISTS outbox_work_items_expired_leases
    ON outbox_work_items (locked_until, id)
    STORING (attempts, max_attempts)
    WHERE status = 'leased';

CREATE INDEX IF NOT EXISTS outbox_work_items_subject
    ON outbox_work_items (subject_id, kind, created_at DESC, id)
    STORING (tenant_id, run_id, status, outcome);

CREATE TABLE IF NOT EXISTS outbox_work_attempts (
    work_id UUID NOT NULL REFERENCES outbox_work_items (id) ON DELETE CASCADE,
    attempt INT8 NOT NULL,
    stage STRING NOT NULL,
    outcome STRING NOT NULL,
    provider STRING NULL,
    model STRING NULL,
    revision STRING NULL,
    prompt_id STRING NULL,
    prompt_sha256 BYTES NULL,
    input_sha256 BYTES NOT NULL,
    output_sha256 BYTES NULL,
    candidate_count INT8 NOT NULL DEFAULT 0,
    error STRING NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT outbox_work_attempts_stage_check CHECK (
        stage IN ('deterministic', 'provider', 'validation', 'apply')
    ),
    CONSTRAINT outbox_work_attempts_outcome_check CHECK (
        outcome IN ('succeeded', 'fallback', 'rejected', 'failed', 'skipped')
    ),
    CONSTRAINT outbox_work_attempts_values_check CHECK (
        attempt > 0 AND candidate_count >= 0 AND finished_at >= started_at
    ),
    PRIMARY KEY (work_id, attempt, stage)
);

CREATE INDEX IF NOT EXISTS outbox_work_attempts_timeline
    ON outbox_work_attempts (work_id, attempt, started_at, stage)
    STORING (
        outcome, provider, model, revision, input_sha256, output_sha256,
        candidate_count, error
    );

CREATE TABLE IF NOT EXISTS outbox_work_effects (
    work_id UUID NOT NULL REFERENCES outbox_work_items (id) ON DELETE CASCADE,
    effect_key STRING NOT NULL,
    kind STRING NOT NULL,
    payload_sha256 BYTES NOT NULL,
    resource_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT outbox_work_effects_kind_check CHECK (kind IN ('memory')),
    PRIMARY KEY (work_id, effect_key)
);

CREATE INDEX IF NOT EXISTS outbox_work_effects_resource
    ON outbox_work_effects (resource_id, kind)
    STORING (work_id, effect_key, payload_sha256, created_at);

CREATE TABLE IF NOT EXISTS swarm_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id STRING NOT NULL,
    project_id STRING NOT NULL,
    repository_id STRING NOT NULL,
    swarm_id STRING NOT NULL,
    run_id STRING NOT NULL,
    agent_id STRING NULL,
    task_id UUID NULL REFERENCES tasks (id) ON DELETE SET NULL,
    event_type STRING NOT NULL,
    aggregate_type STRING NOT NULL,
    aggregate_id STRING NOT NULL,
    aggregate_version INT8 NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::JSONB,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    correlation_id UUID NULL,
    causation_id UUID NULL,
    idempotency_key STRING NULL,
    CONSTRAINT swarm_events_run_fk FOREIGN KEY (tenant_id, run_id)
        REFERENCES runs (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS swarm_events_run_timeline
    ON swarm_events (tenant_id, run_id, occurred_at, id)
    STORING (
        project_id, repository_id, swarm_id, agent_id, task_id, event_type,
        aggregate_type, aggregate_id, aggregate_version, payload, recorded_at,
        correlation_id, causation_id, idempotency_key
    );
