-- Swarm Brain v0 CockroachDB schema.
-- Applying this file is an explicit operator action; application startup never
-- runs DDL. All mutation paths use SERIALIZABLE transactions and an outbox row.

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
    renewed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
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
    CONSTRAINT sources_type_check CHECK (
        source_type IN ('source_code', 'command_output', 'test_result', 'log',
                        'message', 'document', 'url', 'artifact', 'memory')
    ),
    UNIQUE (tenant_id, repository_id, occurrence_key)
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
    title STRING NULL,
    tags STRING[] NOT NULL DEFAULT ARRAY[]::STRING[],
    normalized_sha256 BYTES NOT NULL,
    confidence DECIMAL(5,4) NOT NULL DEFAULT 0.5000,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to TIMESTAMPTZ NULL,
    recorded_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_to TIMESTAMPTZ NULL,
    supersedes_id UUID NULL REFERENCES memories (id) ON DELETE RESTRICT,
    superseded_by_id UUID NULL REFERENCES memories (id) ON DELETE RESTRICT,
    source_id UUID NULL REFERENCES sources (id) ON DELETE SET NULL,
    policy_reason STRING NOT NULL DEFAULT '',
    policy_confidence DECIMAL(5,4) NOT NULL DEFAULT 1.0000,
    version INT8 NOT NULL DEFAULT 1,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    CONSTRAINT memories_kind_check CHECK (
        kind IN ('observation', 'invariant', 'hypothesis', 'decision', 'attempt',
                 'outcome', 'procedure', 'warning', 'handoff')
    ),
    CONSTRAINT memories_state_check CHECK (
        state IN ('tentative', 'confirmed', 'refuted', 'superseded')
    ),
    CONSTRAINT memories_visibility_check CHECK (visibility IN ('task', 'run', 'repository')),
    CONSTRAINT memories_task_visibility_check CHECK (visibility != 'task' OR task_id IS NOT NULL),
    CONSTRAINT memories_confidence_check CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT memories_policy_confidence_check CHECK (
        policy_confidence >= 0 AND policy_confidence <= 1
    ),
    CONSTRAINT memories_valid_interval_check CHECK (valid_to IS NULL OR valid_to > valid_from),
    CONSTRAINT memories_recorded_interval_check CHECK (
        recorded_to IS NULL OR recorded_to > recorded_from
    ),
    CONSTRAINT memories_supersession_check CHECK (supersedes_id IS NULL OR supersedes_id != id)
);

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

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    model STRING NOT NULL,
    dimensions INT8 NOT NULL,
    embedding FLOAT8[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_embeddings_dimensions_check CHECK (dimensions > 0),
    PRIMARY KEY (memory_id, model)
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
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT evidence_kind_check CHECK (
        kind IN ('source_code', 'command_output', 'test_result', 'log',
                 'message', 'document', 'url', 'artifact', 'memory')
    )
);

CREATE TABLE IF NOT EXISTS memory_evidence (
    memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    evidence_id UUID NOT NULL REFERENCES evidence (id) ON DELETE RESTRICT,
    relation STRING NOT NULL DEFAULT 'supports',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_evidence_relation_check CHECK (
        relation IN ('supports', 'refutes', 'derived_from')
    ),
    PRIMARY KEY (memory_id, evidence_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    link_type STRING NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_links_no_self CHECK (source_memory_id != target_memory_id),
    CONSTRAINT memory_links_type_check CHECK (
        link_type IN ('supersedes', 'derived_from', 'supports', 'contradicts',
                      'duplicate_of', 'merged_from', 'related_to')
    ),
    UNIQUE (source_memory_id, target_memory_id, link_type)
);

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
    UNIQUE (tenant_id, actor_id, operation, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idempotency_records_expiry
    ON idempotency_records (expires_at, id)
    STORING (status);

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
    tenant_id STRING NOT NULL,
    run_id STRING NOT NULL,
    aggregate_type STRING NOT NULL,
    aggregate_id STRING NOT NULL,
    aggregate_version INT8 NOT NULL,
    event_type STRING NOT NULL,
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
    ON outbox_events (available_at, id)
    STORING (
        tenant_id, run_id, aggregate_type, aggregate_id, aggregate_version,
        event_type, payload, status, locked_until, version
    )
    WHERE status IN ('pending', 'failed');

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
