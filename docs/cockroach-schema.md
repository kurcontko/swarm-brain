# Swarm Brain CockroachDB schema

## Current v9 retrieval addendum

Schema v9 keeps every v8 projection and adds one durable retrieval-reuse
counter. `memories` remains the canonical source of truth:

- `retrieval_documents`: deterministic `search_text`, bounded `lookup_text`, a
  stored `to_tsvector('simple', search_text)`, a fully scope-prefixed inverted
  FTS index, and a fully scope-prefixed trigram index;
- `retrieval_exact_terms`: normalized IDs, digests, titles, tags, paths,
  symbols, tests, commands, commits, and error identifiers under a B-tree key;
- `retrieval_vectors_1024`: current-memory vectors with resource version,
  content digest, domain lane, stable scope key, and a signature covering the
  complete embedding representation;
- `memory_links`: open semantic links plus covering
  `(source_memory_id, link_type, created_at DESC, id)` and
  `(target_memory_id, link_type, created_at DESC, id)` indexes for bounded
  bidirectional expansion;
- `retrieval_reuse_counters`: one row per authenticated run under primary key
  `(tenant_id, run_id)`, holding `reuse_count`, `recall_count`, the owning
  project/repository/swarm, and first/last write times.

Projected exact-term lookups start at the scope-prefixed exact B-tree; FTS and
trigram lookups start at their scope-prefixed inverted indexes. Each uses a
candidate-driven lookup join to canonical `memories`, then applies
visibility/state/trust/world/system-time predicates before its lane `LIMIT`.
Direct UUID/private-seed lookups use the canonical primary key. Final hydration
repeats the canonical predicates without a recency limit. Projection writes for
publish/merge are in the same transaction as canonical memory; the installer
rebuilds pre-v7 rows through the same application projector used by new writes.
That rebuild is one retryable SERIALIZABLE transaction with a stale scope/version
sweep. Existing compatible legacy vectors are copied into the explicit v2
projection during the v8 operator migration; the legacy table remains intact.

The v8 vector index prefix is `(tenant_id, project_id, repository_id,
resource_type, projection_id, projection_signature, scope_key)`. Runtime
executes one equality-bound ANN query per allowed repository/run/task scope.
CockroachDB only accelerates filters represented by the complete vector prefix,
so lifecycle, bitemporal, kind, version/digest, and trust checks run against
canonical `memories` in the same snapshot after ANN. The candidate window
doubles when those checks under-fill the lane, up to a hard bound. The exact
oracle instead forces `@primary` and applies canonical eligibility before the
exact vector sort. `EXPLAIN` verifies vector-index selection; ANN Recall@k
against that oracle verifies search accuracy.

The graph lane is a second stage, not an unseeded table scan. Direct
exact/FTS/trigram/dense results are fused first; a purpose-owned plan selects at
most 16 seeds, one or two hops, eight eligible neighbors per node, an explicit
relation allowlist, and a total edge-examination budget. CockroachDB executes
batched `LATERAL` scans over both directional link indexes. It over-fetches a
small, fixed multiple of fan-out, canonically validates target IDs under the
same retrieval snapshot, and only then assigns fan-out slots. Thus an
out-of-scope, stale, refuted, or untrusted endpoint is neither traversed nor
allowed to displace an eligible neighbor. Each target is canonically validated
before it can take a fan-out slot; final hydration repeats the same predicates.
Edge `created_at` is bounded by the requested recorded time (or the snapshot
clock); node validity uses the normal bitemporal predicates.

This deliberately avoids a production recursive CTE. The application loop has
an explicit hop terminator, path-local cycle prevention, deterministic
relation/query/hop decay, a hard total budget, and complete edge/node path
provenance. `EXPLAIN` gates verify both directional link indexes and reject
full scans for the emitted hop SQL.

`retrieval_reuse_counters` is telemetry, not a retrieval input. Recall itself
stays a read-only snapshot; after that snapshot commits, the memory adapter runs
one short transaction whose single statement is a primary-key
`INSERT ... ON CONFLICT (tenant_id, run_id) DO UPDATE` that adds the number of
distinct public hits to `reuse_count` and one to `recall_count`. The conflict
branch updates only when the stored project/repository/swarm match the
authenticated scope, so a run identity cannot merge foreign telemetry, and the
`(tenant_id, run_id)` foreign key to `runs` cascades on run deletion. Only counts
are persisted: recall text and memory content never reach this table. The write
is fire-and-forget — a failure is logged and dropped, never surfaced to recall.
`get_run_metrics` reads `reuse_count` with one primary-key lookup inside its
already run-scoped query, so `memories_reused` now has parity between the local
and CockroachDB backends.

Installing v9 is additive: the counter table starts empty and no projection is
rebuilt. Because schema verification includes the checksum of `schema.sql`, an
existing v8 database must still run `schema install` before v9 processes start.

An upgrade from pre-v8 requires an explicit writer barrier: stop all old API
and worker processes that can publish memory or embeddings, run `schema
install` and `schema verify`, and only then start v8 writers. A pre-v8 writer
does not maintain every v8 projection and can otherwise commit memory or an
embedding after the migration snapshot. Concurrent v8 writers are protected by
the shared schema/projection contract, but mixed-version online writes are
unsupported. The lexical rebuild is `O(N)` and a single transaction, so
production operators must rehearse duration and contention on a representative
copy before scheduling the maintenance window.

Startup verification checks the critical index methods, ordered prefixes,
graph direction/relation/time ordering (including descending edge time),
`gin_trgm_ops`, `vector_cosine_ops`, exact-term key, signed vector projection
shape, the exact `(tenant_id, run_id)` reuse-counter key, and stored
`to_tsvector('simple', ...)` definition in addition to names, columns, schema
version, and checksum.

The authoritative implementation is
[schema.sql](../src/swarmbrain/adapters/cockroach/schema.sql); the operational
status and remaining benchmark limitations are in
[retrieval-status.md](retrieval-status.md).

> **Historical pre-v6 inventory.** The detailed table notes below preserve the
> original P0 design and are not a current schema reference. The authoritative
> DDL is
> [schema.sql](../src/swarmbrain/adapters/cockroach/schema.sql),
> while current ingestion, open semantic labels, structured memory content,
> durable work, and fixed-width `VECTOR(1024)` behavior are documented in
> [API contracts](api.md).

The paragraphs and inventory below describe the historical v0/P0 starting
point, not the current runtime. At that point the executable API used only
`InMemoryKernel`, and Cockroach repositories were future P1 work. Today the
authoritative v9 DDL remains an explicit operator action (API startup never
runs migrations), while durable Cockroach repositories, runtime composition,
projection maintenance, schema verification, and live retrieval gates are
implemented. Use the addendum above and `schema.sql` for current facts.

## Conventions

- Every table has an explicit primary key. Distributed entity and event keys
  are UUIDs generated with `gen_random_uuid()`; stable external scope IDs are
  `STRING` and appear in composite keys where appropriate.
- Time is `TIMESTAMPTZ`; structured extension data is `JSONB`; capability and
  tag sets are `STRING[]`; digests are `BYTES` where compared by the database.
- CockroachDB's default `SERIALIZABLE` isolation is retained. Application
  mutations use one short explicit transaction and append their event/outbox
  row before commit.
- Covering indexes use `STORING`; queue/timeline scans use deterministic
  `(time, id)` ordering rather than `OFFSET` pagination.
- The schema is single-region-neutral in v0. A later multi-region deployment
  may assign localities without changing public IDs or contracts.

## Complete table and index inventory

### Identity and run scope

`runs`

- Primary key: `(tenant_id, id)`.
- Required scope: `project_id`, `repository_id`, `swarm_id`.
- State is `active`, `completed`, or `cancelled`; metadata and lifecycle times
  are retained.
- No secondary index is currently defined.

`agents`

- Primary key: `(tenant_id, run_id, id)`.
- Carries authenticated harness/provider/model metadata, capabilities,
  `joined_at`, and `last_seen_at`; status is `active`, `disconnected`, or
  `revoked`, with an optimistic version.
- Foreign key `(tenant_id, run_id) → runs(tenant_id, id) ON DELETE CASCADE`.
- No secondary index is currently defined.

`agent_tokens`

- Primary key: UUID `id`; `token_hash` is globally unique. Raw tokens are never
  stored.
- Foreign key `(tenant_id, run_id, agent_id) → agents(...) ON DELETE CASCADE`.
- Check: `expires_at > issued_at`; `revoked_at IS NULL` and database time are
  required for authentication.
- Covering index `agent_tokens_active_lookup(tenant_id, token_hash) STORING
  (run_id, agent_id, capabilities, expires_at, revoked_at)` supports token
  resolution. Despite the name, expiry/revocation remain query predicates.

### Tasks, dependencies, leases, and checkpoints

`tasks`

- Primary key: UUID `id`; foreign key `(tenant_id, run_id) → runs(...) ON
  DELETE CASCADE`.
- States: `pending`, `blocked`, `ready`, `claimed`, `completed`, `failed`,
  `cancelled`.
- `version` is the optimistic/fencing version; required capabilities and tags
  are arrays. Priority is an `INT8`, higher first. Creator/current claimant,
  active lease, and availability time are denormalized for the typed task
  resource and must agree with lease state.
- Covering index `tasks_claim_queue(tenant_id, run_id, state, priority DESC,
  created_at, id) STORING (required_capabilities, tags, title, description,
  version)` drives deterministic eligible-task selection.

`task_dependencies`

- Primary key `(task_id, depends_on_task_id)`; both columns reference `tasks(id)
  ON DELETE CASCADE`.
- Kind is `blocks` or `related`; a check prevents a self edge. DAG cycle
  detection and same-run/scope validation are application invariants.
- Covering index `task_dependencies_reverse(depends_on_task_id) STORING
  (task_id)` supports downstream readiness updates.

`task_leases`

- Primary key: UUID `id`; task foreign key cascades, authenticated agent foreign
  key restricts deletion.
- States: `active`, `expired`, `released`, `completed`; `expires_at > claimed_at`.
- `version` is a fencing token checked by renew/checkpoint/complete/release.
- Partial unique index `task_leases_one_active_per_task(task_id) WHERE status =
  'active'` is the database guarantee of at most one current owner.
- Covering index `task_leases_agent_active(tenant_id, run_id, agent_id, status,
  expires_at) STORING (task_id, version)` supports runtime renewal and agent
  inventory.

An elapsed lease remains `active` until a claim/expiry transaction changes its
state to `expired`. Claim must do that transition before inserting the new
active row; the partial unique index then closes the race between workers.

`task_checkpoints`

- Primary key: UUID `id`; task foreign key cascades and lease foreign key
  restricts deletion.
- Check `sequence > 0`; unique `(task_id, sequence)` serializes the task journal.
- Payload is a summary, JSON arrays for discoveries and completed/remaining
  work, UUID arrays for memory/artifact IDs, and opaque JSON resumable state.
  Authenticated tenant/run/agent are recorded.
- Covering index `task_checkpoints_latest(task_id, sequence DESC) STORING
  (lease_id, agent_id, summary, discoveries, completed_work, remaining_work,
  memory_ids, artifact_ids, state, created_at)` supports handoff without an
  extra lookup.

`task_completions`

- Primary key: UUID `id`; task foreign key cascades and lease foreign key
  restricts deletion.
- Outcome is `succeeded` or `failed`; the row preserves authenticated
  tenant/run/agent, summary, memory/artifact UUID arrays, and completion time.
- Unique `(task_id)` gives one durable completion record per task. Mutation
  retries return that record through the idempotency result rather than
  appending another completion.
- No secondary index is currently defined.

### Sources and temporal memory

`sources`

- Primary key: UUID `id`.
- Carries tenant/project/repository/run/agent provenance, optional task, source
  type, required occurrence key, optional URI/content, content SHA-256,
  optional world time, reviewer/review/rejection data, optimistic version, and
  metadata.
- Trust is `unknown`, `trusted`, or `untrusted`; review state is `pending`,
  `approved`, or `rejected`.
- Source type is `source_code`, `command_output`, `test_result`, `log`,
  `message`, `document`, `url`, `artifact`, or `memory`.
- Unique `(tenant_id, repository_id, occurrence_key)` preserves distinct
  occurrences of identical content while making a retried occurrence
  idempotent. The public command allows omission, in which case the service
  derives a stable occurrence key from authenticated scope, source digest, and
  mutation idempotency key before persistence.
- No secondary index is currently defined.

`source_chunks`

- Primary key: UUID `id`; `source_id → sources(id) ON DELETE CASCADE`.
- Unique `(source_id, chunk_index)`.
- Checks non-negative chunk/token/character offsets and `char_end >= char_start`.
- No secondary index is currently defined.

`memories`

- Primary key: UUID `id`; optional task/source foreign keys use `SET NULL` on
  deletion, while self-referencing supersession pointers use `RESTRICT`.
- Kinds: `observation`, `invariant`, `hypothesis`, `decision`, `attempt`,
  `outcome`, `procedure`, `warning`, `handoff`.
- States: `tentative`, `confirmed`, `refuted`, `superseded`; visibility: `task`,
  `run`, `repository`. Task visibility requires a `task_id`.
- `confidence` and `policy_confidence` are constrained to `[0, 1]`.
- Content also carries optional title, normalized tags, normalized digest,
  explicit policy reason/confidence, and optimistic `version >= 1` (the latter
  is enforced by repositories; the current DDL has no positive-version check).
- World interval is `[valid_from, valid_to)` and system interval is
  `[recorded_from, recorded_to)`; both require a strictly increasing end when
  present. A current row has `recorded_to IS NULL`.
- `supersedes_id` cannot self-reference; two-way chain consistency, one
  successor, and poisoning policy are transaction invariants.
- Partial covering index `memories_current_scope(tenant_id, repository_id,
  run_id, visibility, state, recorded_from DESC, id) STORING (task_id, agent_id,
  kind, confidence, valid_from, valid_to, content) WHERE recorded_to IS NULL`.
- Partial covering index `memories_task_current(task_id, state, recorded_from
  DESC, id) STORING (kind, confidence, content, agent_id) WHERE task_id IS NOT
  NULL AND recorded_to IS NULL`.
- Partial covering index `memories_source(source_id) STORING (state,
  recorded_to, supersedes_id, superseded_by_id) WHERE source_id IS NOT NULL`
  supports rejection rollback and lineage.

`memory_embeddings`

- Primary key `(memory_id, model)`; memory foreign key cascades.
- `dimensions > 0`; the current DDL stores `FLOAT8[]`. Repository validation
  must ensure array length equals `dimensions` and is compatible with the named
  model.
- v0 has no vector index. Lexical/scoped correctness cannot depend on an
  embedding. A future fixed-dimension `VECTOR(n)` index is a migration, not an
  implicit change to this table.

`evidence`

- Primary key: UUID `id`; required source foreign key uses `ON DELETE RESTRICT`
  so a cited source cannot disappear.
- Kinds: `source_code`, `command_output`, `test_result`, `log`, `message`,
  `document`, `url`, `artifact`, `memory`.
- Carries locator, optional content digest, bounded excerpt, optional artifact
  JSON, metadata, and recorded time.
- No secondary index is currently defined.

`memory_evidence`

- Primary key `(memory_id, evidence_id, relation)`; memory cascades, evidence is
  restricted.
- Relation is `supports`, `refutes`, or `derived_from`.
- No secondary index is currently defined.

`memory_links`

- Primary key: UUID `id`; both memory foreign keys cascade.
- Unique `(source_memory_id, target_memory_id, link_type)` and a no-self check.
- Types: `supersedes`, `derived_from`, `supports`, `contradicts`,
  `duplicate_of`, `merged_from`, `related_to`.
- No secondary index is currently defined.

### Conflicts

`memory_conflicts`

- Primary key: UUID `id`; optional task reference uses `SET NULL`.
- States are `open`, `resolved`, `dismissed`. Non-open state requires a
  `resolved_at`; repository policy additionally requires resolver identity and
  evidence for a resolution.
- Severity is `low`, `medium`, `high`, or `critical`. The row records reporter,
  scope, description, optional structured JSON resolution/resolver, metadata,
  created/updated/resolved times, and optimistic version.
- No secondary index is currently defined.

`memory_conflict_members`

- Primary key `(conflict_id, memory_id)`; both foreign keys cascade.
- Unique `(conflict_id, position)` preserves deterministic presentation order.
- No secondary index is currently defined.

`conflict_evidence`

- Primary key `(conflict_id, evidence_id, relation)`; conflict cascades and
  evidence is restricted.
- `relation` defaults to `supports_resolution`; allowed values are currently
  controlled by the application, not a DDL check.
- No secondary index is currently defined.

### Reliability and audit

`idempotency_records`

- Primary key: UUID `id`; unique `(tenant_id, actor_id, operation,
  idempotency_key)`.
- Stores request SHA-256, state `started`/`completed`/`failed`, canonical HTTP
  status/body, resource ID, and expiry.
- Index `idempotency_records_expiry(expires_at, id) STORING (status)` supports
  bounded cleanup.

`action_attempts`

- Primary key: UUID `id`; optional idempotency-record foreign key uses `SET NULL`.
- `attempt > 0`; states `started`, `succeeded`, `failed`; records SQLSTATE and
  stable error code without secrets.
- Covering index `action_attempts_operation(tenant_id, run_id, operation,
  created_at DESC, id) STORING (agent_id, attempt, status, sqlstate, error_code)`.

`outbox_events`

- Primary key: UUID `id`; non-negative `attempts`; status is `pending`,
  `publishing`, `published`, or `failed`.
- Stores scope, aggregate type/ID/version, event type/payload, availability and
  lock times, occurrence/publication times, last error, and optimistic version.
- Partial covering index `outbox_events_unpublished(available_at, id) STORING
  (tenant_id, run_id, aggregate_type, aggregate_id, aggregate_version,
  event_type, payload, status, locked_until, version) WHERE status IN
  ('pending','failed')` drives fenced keyset polling.

`swarm_events`

- Primary key: UUID `id`; run scope foreign key cascades, optional task uses `SET
  NULL`.
- Append-only audit/read-model event with complete
  tenant/project/repository/swarm/run scope, optional agent/task, required
  aggregate type/ID/version, payload, occurred/recorded times, optional
  correlation/causation IDs, and idempotency key.
- Covering index `swarm_events_run_timeline(tenant_id, run_id, occurred_at, id)
  STORING (project_id, repository_id, swarm_id, agent_id, task_id, event_type,
  aggregate_type, aggregate_id, aggregate_version, payload, recorded_at,
  correlation_id, causation_id, idempotency_key)` supports event streaming and
  keyset pagination.

For migration review, the complete named-constraint inventory is:

```text
runs_state_check
agents_status_check, agents_run_fk
agent_tokens_agent_fk, agent_tokens_expiry_check
tasks_state_check, tasks_run_fk
task_dependencies_no_self, task_dependencies_kind_check
task_leases_status_check, task_leases_expiry_check, task_leases_agent_fk
task_checkpoints_sequence_check
task_completions_outcome_check
sources_trust_check, sources_review_check, sources_type_check
source_chunks_offsets_check
memories_kind_check, memories_state_check, memories_visibility_check
memories_task_visibility_check, memories_confidence_check
memories_policy_confidence_check, memories_valid_interval_check
memories_recorded_interval_check, memories_supersession_check
memory_embeddings_dimensions_check
evidence_kind_check
memory_evidence_relation_check
memory_links_no_self, memory_links_type_check
memory_conflicts_state_check, memory_conflicts_severity_check
memory_conflicts_resolution_check
idempotency_records_status_check
action_attempts_attempt_check, action_attempts_status_check
outbox_events_status_check, outbox_events_attempts_check
swarm_events_run_fk
```

Unnamed primary-key, unique, and inline foreign-key constraints are listed in
their table entries above; their generated database names are not a stable API.

## Invariant ownership

The DDL directly enforces explicit primary keys, enum-like state checks,
interval/confidence checks, foreign keys, task-visibility shape, uniqueness of
active leases, idempotency keys, checkpoint sequence, links, and membership.
The following are mandatory repository/application invariants and must not be
claimed from DDL alone:

1. Every statement is scoped by authenticated `tenant_id`, `repository_id`, and
   `run_id`; denormalized scope fields must match referenced task/lease/memory
   rows.
2. Task dependency edges stay in the same run and form a DAG; a task is
   claimable only when every dependency is completed.
3. Claim expires elapsed active leases, locks/rechecks an eligible task, inserts
   one active lease, advances the task version/state, and writes audit/outbox in
   one transaction.
4. Renew/checkpoint/complete/release require the current active, unexpired lease
   owned by `ActorContext.agent_id`, plus expected lease/task version. Database
   time, not agent time, decides expiry.
5. Every mutation reserves or replays an idempotency row. Reusing a key with a
   different request digest is `409 idempotency_conflict`; the canonical stored
   result is returned for the same digest.
6. Supersession locks the prior current memory, validates evidence/policy,
   inserts the successor and link, closes the old system-time interval, updates
   both pointers/state, and writes audit/outbox atomically. A conditional update
   prevents a second successor even though the DDL has no unique constraint on
   `supersedes_id`.
7. Ordinary recall enforces scope and
   `recorded_to IS NULL AND state NOT IN ('refuted','superseded')` in SQL, and
   excludes rejected/untrusted sources according to policy. Historical recall
   applies explicit system/world as-of interval predicates.
8. Confirm/refute/conflict-resolve and source rejection are capability-gated.
   Unsupported tentative content cannot supersede confirmed evidence-backed
   content.
9. Each durable mutation appends exactly one canonical `swarm_events` row and
   one `outbox_events` row in its business transaction. External calls,
   embeddings, notifications, and artifact writes never occur inside it.
10. Source approve/reject checks the expected source version; non-pending rows
    require reviewer and review time, and rejected rows require a reason.
11. Outbox claim uses availability/status plus expected version and
    `locked_until` as a fencing condition. Publish/fail transitions are
    idempotent and never mark a row published before the external sink accepts
    the stable event ID.

The current DDL deliberately leaves some cross-scope consistency to scoped
repositories because UUID entity keys are globally unique. Cockroach repository
contract tests must prove these predicates; trusting a request's denormalized
scope would be a security defect. The P0 in-memory adapter already fail-closes
task associations, dependency scope, checkpoint/completion memory IDs, and
evidence ID/source pairs; P1 must preserve those checks transactionally.

## Transaction boundaries

Each bullet is one whole `run_serializable` closure. Remote work is excluded.

| Mutation | Locked/read state | Atomic writes and result |
| --- | --- | --- |
| Join agent | authenticated run; token subject | insert/update agent presence, event/outbox, idempotent response. Token issuance is a separate privileged operation. |
| Claim task | idempotency row; elapsed leases; dependency-ready task in deterministic queue order | expire old lease, insert active lease, set task `claimed` and advance version, event/outbox, response. Initial recall occurs after commit and is attached to the response; it does not lengthen the claim lock. |
| Renew lease | lease and task by authenticated owner + expected version | extend from database time, advance lease version/heartbeat, event if configured, stable response. Runtime-only endpoint. |
| Checkpoint | idempotency row, task, active unexpired owned lease, latest sequence | insert next checkpoint, optional handoff memory/evidence, event/outbox, response. |
| Complete task | idempotency row, task, active unexpired owned lease | insert the unique task-completion record, set task/lease completed and versions/times, unblock newly ready dependents, event/outbox, response. Duplicate request replays one completion. |
| Release task | idempotency row, task, active unexpired owned lease | set lease released, return task to `ready` or `blocked`, advance version, event/outbox, response. |
| Remember | idempotency row; referenced task/source/evidence; possible exact duplicate or predecessor | source/evidence joins, accepted new/merged/superseding memory, links, event/outbox, policy result, response. Embedding follows asynchronously. |
| Reject source | source plus all current derived memories/links | mark source rejected/untrusted, close/refute derived current state without erasing it, event/outbox, response. |
| Report conflict | idempotency row and all member memories/evidence in scope | conflict, ordered members, evidence, event/outbox, response. |
| Resolve conflict | idempotency row, open conflict, members, selected/replacement memory | evidence-backed append/supersession/refutation, conflict close, event/outbox, response. |

Read-only recall and lineage use a consistent read but do not acquire mutation
locks. If a historical system-time value is exposed, all participating reads
use the same timestamp. Event and metric pagination is keyset-based.

## `40001`, `40003`, and idempotency

`swarmbrain.adapters.cockroach.retry.run_serializable` reruns the entire
database-only closure for SQLSTATE `40001` with bounded exponential backoff and
jitter. The closure must contain no network call, model invocation, file write,
message, or other irreversible side effect. Generated entity IDs should be
stable within the command or recoverable through the idempotency row.

SQLSTATE `40003` is `statement_completion_unknown`: commit may have succeeded.
`run_serializable` raises `AmbiguousTransactionResult` immediately and never
blindly replays. The application then queries
`idempotency_records(tenant_id, actor_id, operation, key)`:

- `completed` with matching request digest: return the stored status/body;
- `started`: poll/re-read within a bounded interval, then return a retryable
  `503 ambiguous_commit` carrying the same request ID;
- absent after a consistent read: the caller may retry the same command and
  key;
- different request digest: return `409 idempotency_conflict`.

Non-`40001` SQL errors are not automatically retried. In particular, unique or
foreign-key violations map to domain conflicts/not-found errors. Connection
class `08xxx` and shutdown class `57xxx` can also obscure whether a COMMIT
reached the cluster; unless the driver proves failure occurred before commit,
they follow the same idempotency-resolution path rather than a blind mutation
replay.

Attempt logging must not be placed only in a transaction that is about to roll
back. The retry hook or a post-resolution audit path records sanitized attempt
state independently; business truth remains the idempotency record and event
ledger.

## Query and migration verification

The schema tests assert required tables, explicit primary keys, active-lease and
idempotency invariants, current-memory predicates, persisted fencing/provenance
and structured resolution fields, covering claim/event indexes, and avoidance
of sequential-key anti-patterns. Retry tests prove whole-closure `40001`
replay, single-attempt `40003`, no replay of unrelated SQLSTATEs, and retry
policy validation.

This was originally a static-only design pass. The current v7 retrieval gate
now records the SQL emitted by exact, FTS, and trigram gateways, runs `EXPLAIN`
on that exact JOIN/filter/ranking SQL against CockroachDB 26.2, and asserts the
exact B-tree/FTS/trigram indexes plus candidate-driven canonical lookup joins.
Broader production-scale load and contention measurements remain operational
work, not something the focused live gate claims to prove.
