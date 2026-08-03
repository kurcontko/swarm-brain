# Swarm Brain v0 API contracts

This document defines one canonical contract surface for application services,
HTTP, stdio MCP, AgentCore, and the dashboard. The current Pydantic source is
under `swarm-brain/src/swarmbrain/domain/`; current storage/capability protocols
are under `swarm-brain/src/swarmbrain/ports/`. HTTP and MCP are implemented only
when their transport modules and tests exist; the route/tool sections below are
the required contract, not a claim that a declaration is already runnable.

## Contract rules

All domain models inherit `ContractModel`, which is Pydantic v2 with
`extra="forbid"`, `frozen=True`, default validation, and field-name population.
All public identifiers are canonical UUID strings. All datetimes are timezone-
aware; transports emit UTC RFC 3339 values. Confidence and scores are finite
numbers in `[0, 1]`.

Mutation command models inherit `MutationCommand` and contain an
`idempotency_key` of 1–255 non-whitespace characters. `ActorContext` is always
a separate service argument. It is created from verified token/IAM state and
never parsed from a command body or model tool arguments.

Notation below uses `?` for nullable/optional, `T[]` for an immutable tuple or
set serialized as a JSON array, and `={value}` for a default. Unless shown as
open metadata, unlisted fields are rejected.

## Pydantic v2 model catalog

### Identity and authorization

`Capability` values are `run:join`, `task:claim`, `task:checkpoint`,
`task:complete`, `task:release`, `lease:renew`, `memory:publish`,
`memory:recall`, `memory:confirm`, `memory:refute`, `source:review`,
`conflict:report`, `conflict:resolve`, `events:read`, and `metrics:read`.

`ActorContext`

```text
tenant_id, project_id, repository_id, swarm_id, run_id, agent_id: UUID
harness: string(1..100)
provider: string(1..100)
model: string(1..200)
capabilities: unique string[]
harness_version?: string(max 100)
token_id?: string(max 255)
authenticated_at: aware datetime = now
expires_at?: aware datetime (> authenticated_at)
metadata: object = {}
```

`Agent` adds durable `status` (`active|disconnected|revoked`), `joined_at`,
`last_seen_at >= joined_at`, `version >= 1`, and metadata to the same identity
and harness fields. `ActorContext` is auth input to services; `Agent` is a
stored/returned resource.

### Tasks, leases, and handoff

Enums:

- `TaskStatus`: `pending|ready|claimed|blocked|completed|failed|cancelled`;
- `DependencyKind`: `blocks|related`;
- `TaskOutcome`: `succeeded|failed`;
- `LeaseStatus`: `active|released|expired|completed`.

`TaskDependency`

```text
task_id, depends_on_task_id: UUID (different)
kind: DependencyKind = blocks
created_at: aware datetime = now
```

`Task`

```text
task_id, tenant_id, project_id, repository_id, swarm_id, run_id: UUID
title: non-empty string(max 4096)
description: string = ""
status: TaskStatus = pending
priority: integer[-1000,1000] = 0
tags: normalized unique lowercase string[] = []
required_capabilities: unique string[] = []
created_by_agent_id?, claimed_by_agent_id?, active_lease_id?: UUID
available_at?: aware datetime
created_at, updated_at: aware datetime (updated_at >= created_at)
completed_at?: aware datetime
version: integer >= 1
metadata: object = {}
```

A claimed task requires owner and active lease; a completed task requires
`completed_at`.

`ClaimTaskCommand`

```text
idempotency_key: string
task_id?: UUID                 # absent means next eligible task
required_tags: normalized unique lowercase string[] = []
required_capabilities: unique string[] = []
lease_seconds: integer[15,3600] = 120
expected_task_version?: integer >= 1
```

`TaskLease`

```text
lease_id, task_id, run_id, owner_agent_id: UUID
status: LeaseStatus = active
acquired_at: aware datetime
renewed_at?: aware datetime
expires_at: aware datetime (> acquired_at)
released_at?: aware datetime
version: integer >= 1
```

Released/completed leases require `released_at`.

`RenewLeaseCommand`: `idempotency_key`, `lease_id`, `expected_version >= 1`,
`extension_seconds[15,3600]=120`. `RenewLeaseResult`: `lease`, `replayed=false`.
This command is HTTP/runtime-visible, not model-visible.

`TaskCheckpoint`

```text
checkpoint_id, task_id, lease_id, run_id, agent_id: UUID
sequence: integer >= 1
summary: non-empty string
discoveries, completed_work, remaining_work: string[] = []
memory_ids, artifact_ids: UUID[] = []
state: object = {}
created_at: aware datetime = now
```

`CheckpointCommand` contains `idempotency_key`, `task_id`, `lease_id`,
`expected_task_version >= 1`, `expected_lease_version >= 1`, and the checkpoint
payload except server-created IDs/timestamps/sequence.

`TaskCompletion` contains `completion_id`, task/lease/run/agent IDs, `outcome`,
non-empty `summary`, memory/artifact ID arrays, and `completed_at`.
`CompleteTaskCommand` contains the idempotency key, task/lease IDs and expected
versions, `outcome=succeeded`, summary, and memory/artifact IDs.
`ReleaseTaskCommand` contains the key, task/lease IDs and expected versions, and
a non-empty reason.

Results are:

- `ClaimTaskResult(task, lease, checkpoint?, memory?: RecallBundle,
  replayed=false)`;
- `CheckpointResult(task, lease, checkpoint, replayed=false)`;
- `CompletionResult(task, lease, completion, replayed=false)`;
- `ReleaseResult(task, lease, replayed=false)`.

`ClaimTaskResult.memory` is the initial memory/handoff bundle; it is also
available as the `initial_memory` property in Python.

### Source and evidence

Built-in semantic labels and closed operational enums:

- `EvidenceKind`: `source_code|command_output|test_result|log|message|document|
  url|artifact|memory`; callers may also use an application-defined non-empty
  string label (max 255 characters), for example `application/pdf`;
- `SourceTrust`: `unknown|trusted|untrusted`;
- `SourceReviewState`: `pending|approved|rejected`.

SHA-256 values are 64 lowercase hexadecimal characters.

`ArtifactRef`: `artifact_id`, non-empty URI/media type, SHA-256, non-negative
size, created time, metadata. `StoreArtifactCommand` contains idempotency key,
name, media type, digest, size, optional task, metadata; binary bytes travel to
the artifact port separately and never in this model.

`EvidenceSource`

```text
source_id, run_id: UUID
task_id?: UUID
kind: EvidenceKind or application-defined semantic label
uri?: string(max 2048)
content_sha256: sha256
occurrence_key?: string(1..512)
trust: SourceTrust = unknown
review_state: SourceReviewState = pending
observed_at, recorded_at: aware datetime
reviewed_by_agent_id?, reviewed_at?: UUID/datetime
rejection_reason?: string(max 4096)
version: integer >= 1
metadata: object = {}
```

Reviewed sources require reviewer/time; rejected sources require a reason.

`EvidenceRef` is a precise citation: `evidence_id`, `source_id`, optional
locator/excerpt/content SHA-256/artifact, and metadata. `Evidence` adds kind and
recorded time.

Source/evidence mutation commands are `RegisterEvidenceSourceCommand`,
`AddEvidenceCommand`, `ReviewSourceCommand`, and `RejectSourceCommand`; each
adds an idempotency key. Rejection has its own command because it must roll back
derived current state transactionally. `SourceRejectionResult` returns the
reviewed source, rolled-back memory IDs, and replay status.

### Temporal memory

Built-in semantic labels and closed operational enums:

- `MemoryKind`: `observation|invariant|hypothesis|decision|attempt|outcome|
  procedure|warning|handoff`; callers may also use an application-defined
  non-empty string label (max 255 characters);
- `MemoryState`: `tentative|confirmed|refuted|superseded`;
- `Visibility`: `task|run|repository`;
- `MemoryOperation`: `add|update|merge|delete|noop`;
- `MemoryReviewDecision`: `confirm|refute`;
- `MemoryLinkKind`: `supersedes|derived_from|supports|contradicts|duplicate_of|
  merged_from|related_to`; application-defined relation labels are also valid.

`Memory`

```text
memory_id, tenant_id, project_id, repository_id, swarm_id, run_id: UUID
task_id?: UUID
author_agent_id: UUID
kind, state=tentative, visibility=run
content: required non-null JSON value (string, object, array, number, boolean)
title?: string(max 500)
tags: normalized unique lowercase string[] = []
confidence: number[0,1] = 0.5
evidence: EvidenceRef[] = []
valid_from: aware datetime
valid_to?: aware datetime (> valid_from)
recorded_from: aware datetime = now
recorded_to?: aware datetime (> recorded_from)
supersedes_memory_id?, superseded_by_memory_id?: UUID (not self)
version: integer >= 1
metadata: object = {}
```

Task visibility requires a task ID; superseded state requires
`superseded_by_memory_id`.

`RememberCommand` intentionally omits actor-owned scope. It contains
`idempotency_key`, kind/content, `desired_state=tentative`, visibility, optional
task/title/world interval, tags, confidence, evidence, optional
`supersedes_memory_id`, related memory IDs, and metadata. A newly appended row
cannot request `desired_state=superseded`; confirmation/refutation and an
explicit predecessor remain capability- and policy-gated. Tenant, repository,
run, author, system time, version, and policy outcome are server-owned.

The policy is append-by-default: identical content published under a different
idempotency key remains a distinct observation. Exact-content hashing is a
retrieval hint, not a uniqueness constraint. Only an explicit
`supersedes_memory_id` can update an existing current assertion; an explicit
target with identical canonical content may merge corroborating evidence.
Structured content has a deterministic text projection for recall, but the
original JSON value is preserved losslessly.

`MemoryPolicyDecision` returns operation, non-empty reason, confidence, and
target memory IDs. Update/merge/delete require targets; add forbids targets.
`RememberResult` returns the same operation, decision, optional resulting
memory, affected memories, and replay status. Add/update/merge require a
resulting memory. `delete` is a refutation/tombstone operation, not a hard row
delete.

`RecallQuery`

```text
text: non-empty string
task_id?: UUID
kinds: semantic-label[] = []   # built-in or custom; empty means all
visibilities: Visibility[] = [task,run,repository]
states?: MemoryState[]         # default effective states: tentative,confirmed
include_refuted: boolean = false
include_superseded: boolean = false
world_at?, recorded_at?: aware datetime
min_score: number[0,1] = 0
limit: integer[1,100] = 10
include_evidence: boolean = true
include_lineage: boolean = false
```

`RecallHit(memory, score, reasons=[], evidence=[])` and
`RecallBundle(query, hits=[], generated_at, total_candidates>=0,
truncated=false)` are the canonical recall result.

`MemoryLink` has a server ID, source/target IDs, kind, evidence, optional
reason, and creation time; self-links are invalid. `MemoryLineage` contains the
requested ID, all version rows, links, and IDs current at the selected system
time. The requested/current IDs must refer to included rows.

`ReviewMemoryCommand` contains key, memory ID, expected version,
confirm/refute decision, reason, and at least one evidence reference;
`ReviewMemoryResult` returns memory and replay status. It is capability-gated
and not a model-visible v0 tool.

Extraction proposals follow the same flexible content/kind contract. Their
`spans` collection is optional: every supplied span is validated as an exact
quotation of the immutable raw source, while no spans denotes a derived
synthesis and does not invent character offsets.

`EmbeddingVector` validates finite values and exact dimension length;
`EmbeddingMatch` is a memory ID and `[0,1]` score. These are adapter contracts,
not HTTP/MCP memory payloads.

### Conflicts

Enums:

- `ConflictStatus`: `open|resolved|dismissed`;
- `ConflictSeverity`: `low|medium|high|critical`;
- `ConflictResolutionKind`: `prefer_existing|prefer_newer|supersede|merge|
  refute_all|dismiss`.

`ConflictResolutionProposal` contains kind, non-empty rationale, at least one
evidence reference, and supported/refuted memory ID arrays. Non-dismiss
resolutions must identify an outcome; one memory cannot appear in both arrays.
`ConflictResolution` adds server resolution ID, resolver agent, and time.

`Conflict`

```text
conflict_id, tenant_id, project_id, repository_id, swarm_id, run_id: UUID
task_id?: UUID
memory_ids: >=2 unique UUIDs
description: non-empty string
severity: medium
status: open
reported_by_agent_id: UUID
reported_evidence: EvidenceRef[] = []
resolution?: ConflictResolution
created_at, updated_at: aware datetime
version: integer >= 1
metadata: object = {}
```

Resolved conflicts require resolution; open conflicts forbid it.
`ReportConflictCommand` contains key, unique memory IDs, description, optional
task, severity, evidence, metadata. `ResolveConflictCommand` contains key,
conflict ID, expected version, and proposal. Results wrap the conflict and
replay status.

### Events, outbox, and metrics

`SwarmEvent` is an immutable, scoped event with `event_id`, event and aggregate
types, aggregate ID/version, all scope IDs, optional agent/task, JSON payload,
occurred/recorded times, correlation/causation IDs, and optional idempotency key.
Built-in event types cover join, claim, renew, checkpoint, complete, release,
memory add/supersede/confirm/refute, source rejection, and conflict
report/resolve.

`AuditEvent` records actor/action/resource, outcome
(`succeeded|rejected|failed|indeterminate`), reason/time/details. `OutboxEvent`
wraps a `SwarmEvent` with status (`pending|publishing|published|failed`), attempt,
availability/lock/publication times, error, and version. Relay commands claim a
bounded batch or mark one expected version published/failed.

`EventPage(events, next_cursor?)` uses an opaque keyset cursor. `RunMetrics`
contains non-negative task/lease/checkpoint/handoff/memory/conflict/duplicate
counters plus custom numeric metrics.

## Application service protocols

Transports depend on these use-case boundaries, not directly on a SQL adapter.
The current application facades have the following exact public methods. They
enforce capabilities and delegate to typed ports; their presence does not prove
that a durable or in-memory store implementation exists.

```python
class CoordinationService:
    async def join(self, actor: ActorContext) -> Agent: ...
    async def add_task(self, task: Task) -> Task: ...  # administrative seam
    async def claim(self, actor: ActorContext, command: ClaimTaskCommand) -> ClaimTaskResult: ...
    async def renew(self, actor: ActorContext, command: RenewLeaseCommand) -> RenewLeaseResult: ...
    async def checkpoint(self, actor: ActorContext, command: CheckpointCommand) -> CheckpointResult: ...
    async def complete(self, actor: ActorContext, command: CompleteTaskCommand) -> CompletionResult: ...
    async def release(self, actor: ActorContext, command: ReleaseTaskCommand) -> ReleaseResult: ...
    async def events(self, actor: ActorContext, *, cursor: str | None = None, limit: int = 100) -> EventPage: ...
    async def metrics(self, actor: ActorContext) -> RunMetrics: ...

class HandoffService:
    async def checkpoint(self, actor: ActorContext, command: CheckpointCommand) -> CheckpointResult: ...

class MemoryService:
    async def publish(self, actor: ActorContext, command: RememberCommand) -> RememberResult: ...
    async def recall(self, actor: ActorContext, query: RecallQuery) -> RecallBundle: ...
    async def lineage(self, actor: ActorContext, memory_id: MemoryId) -> MemoryLineage: ...
    async def review(self, actor: ActorContext, command: ReviewMemoryCommand) -> ReviewMemoryResult: ...
    async def reject_source(self, actor: ActorContext, command: RejectSourceCommand) -> SourceRejectionResult: ...

class ConflictService:
    async def report(self, actor: ActorContext, command: ReportConflictCommand) -> ReportConflictResult: ...
    async def resolve(self, actor: ActorContext, command: ResolveConflictCommand) -> ResolveConflictResult: ...

class AuditService:
    async def list_events(self, actor: ActorContext, *, cursor: str | None = None, limit: int = 100) -> EventPage: ...
    async def metrics(self, actor: ActorContext) -> RunMetrics: ...
```

Each actor-facing method checks the listed capability before accessing a store
and maps all resource lookups through authenticated scope. `claim` composes a
coordination commit with a post-commit memory/handoff read. A memory recall
failure must not roll back or duplicate an already committed lease; the result
can carry an empty bundle plus a stable retryable warning if necessary.

Current source protocols are `@runtime_checkable`, async, and storage-oriented:

- `CoordinationStore`: `join_agent`, `add_task`, claim/renew/checkpoint/complete/
  release, event listing, and metrics;
- `MemoryOperationPolicy.decide(command, current_memories)`;
  `MemoryStore.remember(actor, command, policy)`, `recall(actor, query)`,
  `get_lineage(actor, memory_id)`, `reject_source(actor, command)`;
  `MemoryReviewStore.review_memory(actor, command)`;
- `EvidenceStore.register_source`, `add_evidence`, `review_source`,
  `get_source`, and `get_evidence`, all with explicit actor and typed
  command/ID arguments;
- `ConflictStore.report_conflict(actor, command)` and
  `resolve_conflict(actor, command)`;
- `EmbeddingProvider`, scope-aware `EmbeddingIndex`;
- `ArtifactReader`, `ArtifactWriter`, `ArtifactStore`;
- `EventSink`, `AuditLog`, `OutboxStore`, `EventReader`, `MetricsReader`.

Adapters may compose these small protocols, but no application method returns
driver rows, `dict[str, Any]`, FastAPI responses, or MCP objects.

## Canonical HTTP API

All endpoints are under `/v1`, consume/produce JSON except artifact bytes, and
require `Authorization: Bearer <run-token>`. Command mutation routes require
`Idempotency-Key`; agent join is a natural upsert keyed by authenticated
tenant/run/agent identity and accepts an empty body. The HTTP adapter injects
the header and any path resource ID into the domain command; those fields need
not be repeated in the JSON body. If a compatibility client sends both, values
must match exactly or the request is rejected.

| Method and path | Body → response | Capability | Success |
| --- | --- | --- | --- |
| `POST /v1/runs/{run_id}/agents:join` | empty object → `Agent` | `run:join` | `200` |
| `POST /v1/tasks:claim` | `ClaimTaskCommand` minus header key → `ClaimTaskResult` | `task:claim` | `200` |
| `POST /v1/leases/{lease_id}:renew` | expected version, extension → `RenewLeaseResult` | `lease:renew` | `200` |
| `POST /v1/memories` | `RememberCommand` minus header key → `RememberResult` | `memory:publish` | `200` |
| `POST /v1/memories:recall` | `RecallQuery` → `RecallBundle` | `memory:recall` | `200` |
| `GET /v1/memories/{memory_id}/lineage` | no body → `MemoryLineage` | `memory:recall` | `200` |
| `POST /v1/tasks/{task_id}/checkpoints` | `CheckpointCommand` minus path/key → `CheckpointResult` | `task:checkpoint` | `200` |
| `POST /v1/tasks/{task_id}:complete` | `CompleteTaskCommand` minus path/key → `CompletionResult` | `task:complete` | `200` |
| `POST /v1/tasks/{task_id}:release` | `ReleaseTaskCommand` minus path/key → `ReleaseResult` | `task:release` | `200` |
| `POST /v1/conflicts` | `ReportConflictCommand` minus key → `ReportConflictResult` | `conflict:report` | `200` |
| `POST /v1/conflicts/{conflict_id}:resolve` | expected version + proposal → `ResolveConflictResult` | `conflict:resolve` | `200` |
| `GET /v1/runs/{run_id}/events?cursor=&limit=` | no body → `EventPage` | `events:read` | `200` |
| `GET /v1/runs/{run_id}/metrics` | no body → `RunMetrics` | `metrics:read` | `200` |

Additional capability-gated review/source/artifact endpoints may be added
without becoming MCP tools. They must reuse the domain commands above.

Path `run_id` must equal the authenticated run. Task/lease/memory/conflict IDs
are looked up with tenant/repository/run predicates; an out-of-scope ID is
reported as not found or scope mismatch without revealing its existence.

### Idempotency behavior

The canonical request digest includes operation name, authenticated actor,
normalized path parameters, and validated command payload; it excludes tracing
headers. The store reserves `(tenant_id, actor_id, operation,
idempotency_key)` inside the business transaction.

- same key and digest after completion: return stored logical status/body with
  `Idempotency-Replayed: true` and result `replayed=true`;
- same key, different digest: `409 idempotency_conflict`;
- concurrent in-progress duplicate: wait/re-read briefly or return retryable
  `409/503` with the same request ID, never execute a second mutation;
- ambiguous `40003`: resolve the record; never blind replay;
- clients may retry timeouts only with the same key.

## Model-visible MCP tools

The stdio server registers exactly six tools. It reads the API URL/token from
its environment and calls canonical HTTP. No input contains tenant, project,
repository, swarm, run, agent, harness, provider, model, token, capability, or
author fields.

| Tool | Input | Output |
| --- | --- | --- |
| `claim_task` | `idempotency_key`, optional task ID/required tags/required capabilities/lease seconds/expected task version | `ClaimTaskResult`, including checkpoint and initial memory |
| `recall_memory` | `RecallQuery` fields | `RecallBundle` |
| `publish_memory` | `RememberCommand` fields | `RememberResult` |
| `checkpoint_task` | `CheckpointCommand` fields | `CheckpointResult` |
| `complete_task` | `CompleteTaskCommand` fields | `CompletionResult` |
| `report_conflict` | `ReportConflictCommand` fields | `ReportConflictResult` |

`claim_task` begins automatic lease renewal in the bridge. Completion stops it.
Renewal, release, memory review/source rejection, conflict resolution, events,
metrics, token issuance, and join are runtime/operator/dashboard HTTP actions,
not model tools. The bridge contains no memory policy, ranking, claim logic, or
direct database access.

## Error model

All non-2xx responses use:

```json
{
  "error": {
    "code": "lease_lost",
    "message": "lease is not current and owned",
    "request_id": "00000000-0000-0000-0000-000000000000",
    "retryable": false,
    "details": {"lease_id": "00000000-0000-0000-0000-000000000000"}
  }
}
```

Messages/details never contain tokens, database URLs, SQL, source content, or
cross-scope resource data.

| HTTP | Stable code | Meaning/retry |
| --- | --- | --- |
| `400` | `invalid_request` | Malformed JSON/header/path; fix request. |
| `401` | `authentication_required`, `invalid_token` | Missing, invalid, expired, or revoked token; refresh out of band. |
| `403` | `forbidden`, `scope_mismatch` | Missing capability or authenticated scope mismatch; do not retry unchanged. |
| `404` | `not_found` | Resource absent or deliberately hidden by scope. |
| `409` | `no_task_available` | Retryable after work/dependency/lease state changes. |
| `409` | `invalid_state`, `lease_lost` | Refresh task/lease; stale owner/version is fenced. |
| `409` | `idempotency_conflict` | Key was reused for a different normalized command. |
| `409` | `memory_policy_rejected` | Proposal violates conservative memory policy. |
| `422` | `validation_error` | Strict Pydantic validation failed; return bounded field locations. |
| `422` | `conflict_requires_evidence` | Resolver must attach evidence. |
| `429` | `rate_limited` | Retry after server-provided delay with same idempotency key. |
| `503` | `ambiguous_commit` | Outcome unresolved; retry/poll only with the same key. |
| `503` | `temporarily_unavailable` | Read or pre-commit dependency unavailable; honor retry metadata. |
| `500` | `internal_error` | Opaque request ID; no internals exposed. |

The current application error enum implements the central subset
(`authentication_required`, `invalid_token`, `forbidden`, `not_found`,
`no_task_available`, `invalid_state`, `lease_lost`, `idempotency_conflict`,
`memory_policy_rejected`, `conflict_requires_evidence`, `ambiguous_commit`).
The HTTP adapter additionally emits a bounded `validation_error` envelope.
Explicit `scope_mismatch`, rate-limit, and opaque catch-all internal mappings
remain planned; current out-of-scope lookups use `not_found` where hiding
existence is required.

## Current conformance boundary

The current tree contains these strict models/ports, capability-gated
application facades, in-memory runtime, signed-token codec, canonical FastAPI
route functions, and exactly six registered MCP functions over the HTTP client.
Thirty-two checked-in tests cover strict domain schemas, mutation idempotency
shape, auth-owned-field rejection, canonical UUID/time validation, runtime port
conformance, CockroachDB DDL/retry, and ten deterministic in-memory scenarios:
claim contention, dependency blocking, idempotent completion with one
event/outbox row, crash handoff with stale-owner fencing, cross-agent recall,
supersession/history, poisoning resistance, source-rejection rollback, and
conflict resolution, plus fail-closed cross-repository recall/lineage and
rejection of unregistered evidence references. The in-memory adapter also
validates task associations, dependency scope, checkpoint/completion memory
IDs, and evidence source/ID pairs before mutation. Transport tests cover signed
tokens, authentication/scope,
replay headers, typed OpenAPI request/response references, a full in-memory
vertical slice across every canonical route, the exact MCP tool schema/HTTP
translation, and automatic bridge renewal against fake HTTP. They do not yet
prove a real stdio process, live CockroachDB, or restart durability. P0 is
deliberately in-memory and has no Cockroach repository. Likewise, DDL must stay
aligned with domain enums/fields through schema/contract tests; a field in one
layer is not evidence another layer persists it. The PR gates in
[implementation plan](implementation-plan.md) define the evidence needed to
advance that boundary.
