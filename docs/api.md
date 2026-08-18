# Swarm Brain v0 API contracts

This document defines one canonical contract surface for application services,
HTTP, stdio MCP, AgentCore, and the dashboard. The current Pydantic source is
under `src/swarmbrain/domain/`; current storage/capability protocols are under
`src/swarmbrain/ports/`. HTTP and MCP are implemented only
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
`source:ingest`, `conflict:report`, `conflict:resolve`, `events:read`, and
`metrics:read`.

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

- `ClaimTaskResult(task, lease, checkpoint?, unblocked_by_task_ids=[],
  activation?: MemoryActivationTelemetry, activation_context?: string,
  replayed=false)`;
- `CheckpointResult(task, lease, checkpoint, replayed=false)`;
- `CompletionResult(task, lease, completion, replayed=false)`;
- `ReleaseResult(task, lease, replayed=false)`.

`ClaimTaskResult.memory` remains an in-process compatibility field, also
available as the `initial_memory` Python property, but it is explicitly
excluded from model serialization. HTTP and MCP therefore never return the raw
`RecallBundle`; wire consumers receive only the bounded `activation_context`
and content-free `activation` telemetry.

#### Selective task and in-task activation

`MemoryActivationRequest` identifies the task, lease, trigger, optional
retrieval purpose and checkpoint seed IDs, plus a token budget, relevance floor,
and result limit. It intentionally contains no query text. The corresponding
`MemoryActivationTelemetry` records stable activation/run/agent/task/lease IDs,
trigger, `skip|recall|deep_recall|defer` decision, closed reason and purpose,
optional retrieval trace ID, selected and dropped memory IDs, budget and
estimated tokens, exact selected-memory versions, score floor, candidate count,
and truncation. It contains no task text, prompt, memory content, provider error,
or rendered context.

On a task claim, the coordination service constructs an ephemeral query from
the task and optional checkpoint, then retrieves only `confirmed` memories.
The default final limit is 12, the relevance floor is `0.4`, and the activation
budget is 2,048 estimated tokens. Canonically rendered memory blocks are packed
greedily to that budget; selected IDs and dropped IDs are recorded separately.
Empty or irrelevant retrieval skips injection, while an unavailable optional
recall lane defers it. Checkpoint resume uses the deeper handoff-recovery purpose
and privately seeds cited checkpoint IDs, but the final context remains subject
to the same limit and budget.

Activation runs after the lease commit and only for actors with
`memory:recall`. A failure cannot hide or duplicate the successful claim: the
claim returns with no activation context. Activation telemetry must persist
before context is exposed, so an event-store failure also returns only the
committed claim rather than delivering untracked memory. The activation
transaction also reapplies the canonical lifecycle, temporal, scope, and trust
predicates and requires every selected memory version to remain unchanged. This
withholds stale rendered context, including a selection whose evidence changed
after recall. The raw bundle exists only long enough for in-process
compatibility; the canonical context string is the sole memory-content
representation that crosses the claim HTTP/MCP boundary.
Because that context is deliberately ephemeral rather than part of the durable
claim response, an idempotent claim replay returns the stable task, lease, and
checkpoint with `replayed=true` but does not recompute activation against newer
memory state when its deterministic activation event already exists. If the
claim committed but the process stopped before recording any activation event,
the first replay repairs that missing post-commit effect while the lease is
still valid; concurrent repairs converge on the same deterministic event.

### Source and evidence

Built-in semantic labels and closed operational enums:

- `EvidenceKind`: `source_code|command_output|test_result|log|message|document|
  url|artifact|memory`; callers may also use an application-defined non-empty
  string label (max 255 characters), for example `application/pdf`;
- `SourceTrust`: `unknown|trusted|untrusted`;
- `SourceReviewState`: `pending|approved|rejected`.

Source rejection is terminal. `RejectSourceCommand` rolls back memories that
lose their final acceptable source, and a rejected source cannot later be
reviewed back to pending or approved; new evidence must be registered as a new
source occurrence.

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

#### Durable raw-source ingestion

`IngestSourceBody` contains only caller-owned source material: `kind`, `content`,
optional `observed_at`, `task_id`, `uri`, `occurrence_key`, `metadata`, and an
optional compatibility echo of the `Idempotency-Key` header. Trust, provider
selection, extraction revision, chunk size, and queue priority are operator
configuration and are rejected if supplied in the public body.

The durable receipt is intentionally small:

```text
SourceIngestResult
  source: EvidenceSource
  work_id: UUID
  chunk_count: integer >= 1
  replayed: boolean = false
```

The status reader exposes `queued|leased|completed|failed|cancelled`, attempt
counts, bounded outcome/route/candidate count, resulting memory IDs, and public
timestamps. It never returns raw content, chunks, a lease token, worker owner,
provider details, prompts, or internal error text.

The dependency-free structured route accepts the explicit media type
`application/vnd.swarmbrain.memory+json`. Its content is a JSON envelope whose
`memories` array contains ordinary `ExtractionCandidate` values:

```json
{
  "memories": [
    {
      "kind": "org.example/runbook-rule",
      "content": {"retry_sqlstate": "40001", "strategy": ["backoff", "retry"]},
      "title": "Serializable retry rule",
      "tags": ["cockroachdb", "retry"],
      "confidence": 0.9
    }
  ]
}
```

For ordinary raw-source work, deterministic extraction runs first. Operators
may additionally enable the OpenAI-compatible typed-memory compiler with
`SWARMBRAIN_INGEST_USE_PROVIDER=true`,
`SWARMBRAIN_EXTRACTION_PROVIDER=openai`, and a complete model/base-URL profile.
The public ingest request cannot select or configure this provider.

The provider receives immutable source chunks as untrusted data and must answer
through a strict JSON schema. It may propose bounded memory kind/content,
candidate-local keys and relations, title, tags, confidence, event/valid time,
aliases, namespaced metadata entries, and verbatim source quotations. It cannot
set a memory ID, scope, author, visibility, lifecycle state, trust decision, or
storage policy outcome. Quotations carry a chunk index and occurrence rather
than provider-invented offsets; the adapter resolves exact offsets locally and
rejects a quotation that cannot be matched to the preserved chunk.

Provider count/byte/time limits, Pydantic validation, exact-span checks,
candidate-graph checks, and local deduplication all run before fenced apply. A
provider outage, malformed response, or rejected candidate graph records only a
bounded failure class and returns the deterministic candidates with
`status=fallback`; raw provider messages are not persisted or exposed through
the status reader. Successful provenance identifies provider, model, optional
revision, and the versioned prompt digest without turning provider output into
trusted state.

Storage still assigns scope, state, visibility, IDs, author, system time, and
lineage. A candidate without exact spans receives deterministic whole-source
evidence (digest and source relation, no invented excerpt). Rejecting that
source refutes its derived current memories and cancels pending, retryable, or
leased extraction work in the same fenced transaction.

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
occurred_at?: aware datetime     # provenance-backed event observation
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
`occurred_at`, `supersedes_memory_id`, related memory IDs, and metadata.
`occurred_at` is accepted only with at least one immutable evidence reference;
it has no required ordering relationship with the world-validity or
system-recording intervals. A newly appended row
may also carry `derived_from_memory_ids`; those typed lineage targets must be
current confirmed memories, every target must contribute at least one exact
immutable evidence ref, and the output evidence must be a subset of their
combined refs. A newly appended row cannot request `desired_state=superseded`;
confirmation/refutation and an
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
memory_ids: UUID[] = []        # deprecated HTTP v1 compatibility selector
kinds: semantic-label[] = []   # built-in or custom; empty means all
visibilities: Visibility[] = [task,run,repository]
states?: MemoryState[]         # default effective states: tentative,confirmed
include_refuted: boolean = false
include_superseded: boolean = false
world_at?: aware datetime        # valid at one world-time instant
referenced_valid_from?, referenced_valid_to?: aware datetime
                                # supplied together; half-open overlap routing
occurrence_time_prior_from?, occurrence_time_prior_to?: aware datetime
                                # supplied together; soft event-time prior only
recorded_at?: aware datetime     # system-time/as-of selector
min_score: number[0,1] = 0     # floors calibrated per-hit relevance
                               # (lane-max-v1), not the ranked score; 0 keeps
                               # today's behavior, higher values allow abstention
limit: integer[1,100] = 10
include_evidence: boolean = true
include_lineage: boolean = false
```

`RecallHit(memory, score, reasons=[], evidence=[])` and
`RecallBundle(query, hits=[], generated_at, total_candidates>=0,
truncated=false)` are the canonical recall result.

Retrieval purpose, intent, enabled lanes, lane budgets, fusion weights, and the
full trace are server-owned and are not fields of `RecallQuery` or
`RecallBundle`. Ordinary HTTP/MCP recall uses `interactive_recall`; task claim
uses `task_bootstrap` (or `handoff_recovery` on checkpoint resume) and may
privately seed checkpoint memory IDs. Automatic claim activation is narrower
than interactive recall: it requests confirmed state only, applies its score
floor, renders canonical blocks, and packs the selected blocks to the request's
token budget before any content reaches the claimant. Exact, FTS `simple`, and
trigram candidates are fused with weighted RRF, then every ID is revalidated
through canonical scope/state/trust/world/system-time predicates. The
deprecated `memory_ids` field remains accepted by HTTP v1 for compatibility but
is no longer used as an internal hydration transport. No zero-score
`scope_match` is emitted.

`world_at` and the referenced-validity pair are mutually exclusive valid-time
selectors. The interval pair matches `[memory.valid_from, memory.valid_to)` by
strict half-open overlap; unlike default recall it does not also require the
memory to be valid now. `recorded_at` is orthogonal and may be combined with
either selector. The occurrence-prior pair is also orthogonal: when explicitly
provided, it adds a reciprocal-distance temporal candidate lane over non-null
`memory.occurred_at`. It is never copied into canonical validity predicates,
and a memory with no occurrence time remains eligible and receives no temporal
contribution. Without the pair, occurrence time is not read by ranking. The
server does not infer either interval from query text.

`MemoryActivationCommand(task_id, lease_id, trigger, query_text,
seed_memory_ids=[], token_budget=2048, min_score=0.4, limit=12,
referenced_valid_from?, referenced_valid_to?)` exposes only
`tool_error|repeated_failure|explicit`; automatic claim, resume, and dependency
triggers remain server-owned. Query text is bounded to 8,192 characters and is
discarded before telemetry persistence. The content-free referenced interval is
retained in telemetry only so the commit-time proof can reapply the same
selector. The task/lease must be the caller's active owned lease before recall
begins, and storage revalidates that lease plus each selected memory's exact
version and canonical eligibility before returning `MemoryActivationDelivery`.
A deterministic replay returns stored telemetry without historical context.

`recall_memory` is the iterative search step and retains bounded lane reasons.
`ReadExpandMemoryRequest` is its exact read step: 1–8 IDs, graph depth `0..2`,
fanout `1..8`, a total 1–16,384-token budget, and the same optional referenced
validity interval. Seeds and neighbors are
hydrated under one canonical snapshot and confirmed/current/scope/time/trust
filters. `ReadExpandMemoryResult` contains exact selected versions, bounded
provenance, dropped IDs, estimated tokens, truncation, and one canonical context.
Canonical activation/read-expand blocks render the memory's `valid_from` and,
when finite, `valid_to` alongside recorded time; `occurred_at` is rendered only
when provenance supplied it.
HTTP routes are `/v1/tasks/{task_id}/memories:activate` and
`/v1/tasks/{task_id}/memories:read-expand`.

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
memory add/supersede/confirm/refute/activate, source rejection, and conflict
report/resolve.

`AuditEvent` records actor/action/resource, outcome
(`succeeded|rejected|failed|indeterminate`), reason/time/details. `OutboxEvent`
wraps a `SwarmEvent` with status (`pending|publishing|published|failed`), attempt,
availability/lock/publication times, error, and version. Relay commands claim a
bounded batch or mark one expected version published/failed.

`EventPage(events, next_cursor?)` uses an opaque keyset cursor. `RunMetrics`
contains non-negative task/lease/checkpoint/handoff/memory/conflict/duplicate
counters plus custom numeric metrics. Its memory lifecycle counters have
deliberately different meanings:

- `memory_activation_attempts` counts content-free `memory.activated` decision
  events, including skip/defer outcomes;
- `memories_activated` counts the memory IDs actually selected by those events;
- `memories_cited` counts distinct durable references by task, lease,
  consuming agent, and memory ID; checkpoint and completion citations under
  one lease are deduplicated, while a later lease is a distinct use;
- `cross_agent_memory_uses` counts citations of another agent's memory that can
  also be tied to activation for the same task, lease, and consuming agent.

`memories_reused` is deprecated compatibility telemetry: it counts memories
returned by recall and is not evidence that an agent saw, cited, or used them.

`MemoryOutcomeAssociation` is a separate content-free, read-only offline
contract. It is emitted only when a completion citation intersects an exact
same-task/same-lease/same-consumer activation and the activated memory version
is still canonically recallable. Its `observational_silver` kind is not a
causal quality label. It stores scope IDs, memory/version, outcome, and time;
query, memory content, task summary, and prompt text are absent. Neither
successful nor failed associations affect retrieval ranking.

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
    async def checkpoint(
        self, actor: ActorContext, command: CheckpointCommand
    ) -> CheckpointResult: ...
    async def complete(
        self, actor: ActorContext, command: CompleteTaskCommand
    ) -> CompletionResult: ...
    async def release(self, actor: ActorContext, command: ReleaseTaskCommand) -> ReleaseResult: ...
    async def events(
        self, actor: ActorContext, *, cursor: str | None = None, limit: int = 100
    ) -> EventPage: ...
    async def metrics(self, actor: ActorContext) -> RunMetrics: ...


class HandoffService:
    async def checkpoint(
        self, actor: ActorContext, command: CheckpointCommand
    ) -> CheckpointResult: ...


class MemoryService:
    async def publish(self, actor: ActorContext, command: RememberCommand) -> RememberResult: ...
    async def recall(self, actor: ActorContext, query: RecallQuery) -> RecallBundle: ...
    async def lineage(self, actor: ActorContext, memory_id: MemoryId) -> MemoryLineage: ...
    async def review(
        self, actor: ActorContext, command: ReviewMemoryCommand
    ) -> ReviewMemoryResult: ...
    async def reject_source(
        self, actor: ActorContext, command: RejectSourceCommand
    ) -> SourceRejectionResult: ...


class EvidenceService:
    async def register_source(
        self, actor: ActorContext, command: RegisterEvidenceSourceCommand
    ) -> EvidenceSource: ...
    async def add_evidence(self, actor: ActorContext, command: AddEvidenceCommand) -> Evidence: ...
    async def review_source(
        self, actor: ActorContext, command: ReviewSourceCommand
    ) -> EvidenceSource: ...


class ExtractionService:
    async def ingest(
        self, actor: ActorContext, command: IngestRawSourceCommand
    ) -> SourceIngestResult: ...
    async def status(self, actor: ActorContext, source_id: str) -> SourceExtractionStatus: ...


class ConflictService:
    async def report(
        self, actor: ActorContext, command: ReportConflictCommand
    ) -> ReportConflictResult: ...
    async def resolve(
        self, actor: ActorContext, command: ResolveConflictCommand
    ) -> ResolveConflictResult: ...


class AuditService:
    async def list_events(
        self, actor: ActorContext, *, cursor: str | None = None, limit: int = 100
    ) -> EventPage: ...
    async def metrics(self, actor: ActorContext) -> RunMetrics: ...
```

Each actor-facing method checks the listed capability before accessing a store
and maps all resource lookups through authenticated scope. `claim` composes a
coordination commit with post-commit selective memory activation. A memory
recall failure must not roll back or duplicate an already committed lease; the
result simply omits activation content while retaining the committed task and
lease.

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
- `SourceIngestStore` for atomic source/chunk/work persistence and safe status;
  `WorkQueueStore` for compatible claims, fenced apply/failure, and effect-once
  records;
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
| `POST /v1/evidence/sources` | caller-owned source descriptor → `EvidenceSource` | `memory:publish` | `200` |
| `POST /v1/evidence` | `AddEvidenceCommand` minus header key → `Evidence` | `memory:publish` | `200` |
| `POST /v1/sources:ingest` | `IngestSourceBody` → `SourceIngestResult` | `source:ingest` | `202` |
| `GET /v1/sources/{source_id}/extraction` | no body → `SourceExtractionStatus` | `source:ingest` | `200` |
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

Activation in `ClaimTaskResult` is optional enrichment. It runs only when the
actor has both `task:claim` and `memory:recall`; a claim-only actor receives no
activation context. The raw `memory` compatibility field is excluded rather
than serialized as `null`. Once a lease is committed, a recall failure is
logged and the successful claim is still returned, so the owner never loses
the lease identity because the optional retrieval lane is unavailable.

### Durable worker process

CockroachDB ingestion is deliberately unavailable on the in-memory backend.
After the schema is installed explicitly, run the API and worker as separate
processes against the same database:

```bash
export SWARMBRAIN_BACKEND=cockroach
export SWARMBRAIN_DATABASE_URL='postgresql://root@127.0.0.1:26257/swarmbrain?sslmode=disable'
export SWARMBRAIN_TOKEN_SECRET='replace-with-a-local-secret'

uv run --extra serve --extra crdb swarmbrain-api
uv run --extra crdb swarmbrain-worker
```

`swarmbrain-worker --once` performs one bounded queue cycle. The long-running
form polls until `SIGINT` or `SIGTERM`, then closes its pool. Extraction claims
only work carrying the same deterministic extractor name and revision. The
worker limit is fixed at one until lease heartbeats are implemented; effects
remain exactly-once, while a slow external inference may be attempted again
after lease expiry. If the last allowed attempt expires with its worker, the
next claim cycle terminalizes it as `failed` with `lease_expired` instead of
leaving an unclaimable leased row.

When consolidation is enabled, the same supervisor also claims
`consolidate_memory` work. It stages the exact bounded reflection plan under
the lease fence before publishing any effect, reuses that plan after a crash,
and completes only after every deterministic action idempotency key has passed
through normal memory governance. See [the consolidation runtime](consolidation-runtime.md).

`SWARMBRAIN_EMBEDDINGS=none|deterministic|bedrock|openai` controls the optional
dense lane. CockroachDB schema v8 introduced an additive `retrieval_vectors_1024`
projection with canonical resource version/content digest, repository/run/task
scope key, domain lane, and a signature covering renderer, current mode,
cosine metric, provider normalization/truncation, model, and dimensions. Its
vector index prefix is `(tenant, project, repository, resource_type,
projection_id, signature, scope_key)`; every ANN query binds the complete
prefix. CockroachDB cannot accelerate arbitrary non-prefix filters, so the
gateway validates lifecycle, bitemporal, kind, version/digest, and source trust
against canonical memory in the same snapshot, geometrically widening an
under-filled ANN window up to a bounded cap. Structured content is embedded
through its canonical deterministic text projection. `deterministic` is
credential-free and intended for tests/local flow verification. Bedrock is a
lazy optional integration and runs outside database transactions.

The source-ingest transaction persists the resolved embedding model on
extraction work. Fenced extraction apply atomically creates one deduplicated
`embed_memory` child per materialized memory, so a rolling deploy cannot
silently switch models. The provider computes a vector before the retrieval or
writer transaction; legacy vector UPSERT, signed v2 projection UPSERT, and work
completion then commit together behind the current lease fence. A stale worker
therefore cannot create or overwrite either vector representation. Query-time
dense candidates participate in the same weighted RRF and trace as exact, FTS,
and trigram candidates. Dense-provider or ANN-index failure is recorded as a
degraded lane and the other lanes still complete. Current-only dense is skipped
for explicit historical/refuted/superseded recall.

`SWARMBRAIN_RETRIEVAL_DENSE_MIN_SIMILARITY` sets an optional raw cosine floor
before fusion. Its default is `0.0` because a universal cutoff is not calibrated
across embedding models; zero/negative contributions are still excluded by
fusion. `SWARMBRAIN_RETRIEVAL_DENSE_ANN_BEAM_SIZE` is applied with `SET LOCAL`
for the CockroachDB ANN branch. Both are operator policy, not public recall
fields, and should be tuned against the exact vector oracle.

The runtime also enables an internal second-stage graph lane over
`memory_links`; this does not add fields to `RecallQuery` or `RecallBundle`.
Exact/FTS/trigram/dense candidates fuse first and seed a purpose-owned bounded
expansion. Interactive, historical, and repository-orientation recall use one
hop; task bootstrap, handoff recovery, planning, and conflict review use two.
The plan records seed, fan-out, edge, relation, and candidate budgets. Every
endpoint is rechecked for tenant/project/repository, visibility, run/task,
state, kind, bitemporal eligibility, and source trust before it can consume an
eligible fan-out slot, and final hydration repeats those predicates. Candidate
traces retain the exact edge/node path and relation sequence; public hits only
receive bounded graph reasons. A graph failure is a degraded lane and does not
discard valid direct results.

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

The stdio server registers exactly nine tools. It reads the API URL/token from
its environment and calls canonical HTTP. No input contains tenant, project,
repository, swarm, run, agent, harness, provider, model, token, capability, or
author fields.

| Tool | Input | Output |
| --- | --- | --- |
| `claim_task` | `idempotency_key`, optional task ID/required tags/required capabilities/lease seconds/expected task version | `ClaimTaskResult`, including checkpoint, activation telemetry, and bounded activation context (never a raw recall bundle) |
| `recall_memory` | `RecallQuery` fields | `RecallBundle` |
| `activate_memory` | task ID, caller-selectable trigger, ephemeral query, seeds and bounded policy; lease is bridge-owned | `MemoryActivationDelivery` |
| `read_expand_memory` | task ID, ephemeral query, 1–8 IDs, depth/fanout/budget/evidence controls; lease is bridge-owned | `ReadExpandMemoryResult` |
| `publish_memory` | `RememberCommand` fields | `RememberResult` |
| `checkpoint_task` | `CheckpointCommand` fields | `CheckpointResult` |
| `complete_task` | `CompleteTaskCommand` fields | `CompletionResult` |
| `report_conflict` | `ReportConflictCommand` fields | `ReportConflictResult` |
| `ingest_memory_source` | caller-owned `IngestSourceBody` fields plus idempotency key | `SourceIngestResult` |

`claim_task` begins automatic lease renewal in the bridge. Completion stops it.
Renewal, release, memory review/source rejection, conflict resolution, events,
metrics, token issuance, and join are runtime/operator/dashboard HTTP actions,
not model tools. The bridge contains no memory policy, ranking, claim logic, or
direct database access.

Inline MCP evidence is registered before its parent mutation. Its deterministic
child namespace includes the authenticated agent, parent operation, parent
idempotency key, and evidence position. This preserves restart replay without
conflating the same caller key across tools or agents; changing evidence under
the same actor/operation/key is rejected before another HTTP request.

## Scripted runtime demonstration

`swarmbrain-demo` instantiates exactly four agents with four provider labels and
four tasks arranged in two waves. Wave A contains two independent
investigations, so two claimants win while two agents receive
`no_task_available`; both Wave-B tasks depend on both investigations and become
claimable only after Wave A completes. Those two waiting agents then own Wave B.

Wave A publishes and confirms two fresh opaque facts. Wave-B claims receive
only their packed `activation_context`, and the demo's pure verifier runs once
with an empty context and once with that exact wire string. The latter must
recover both facts while superseded and unsupported guidance remains absent.
Checkpoint and completion commands explicitly cite accepted memory IDs, a
crashed claimant resumes under a different provider from its checkpoint, and a
late write is fenced. The final checks read durable events and require the
activation, citation, and proven cross-agent-use counters to agree with that
causal path.

With `--ab`, the no-framework arm is explicitly labeled as a deterministic
simulation over the shared fixture. Only the Swarm Brain arm is measured from
the live HTTP run, event ledger, and runtime metrics.

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

The current tree implements the strict models/ports, capability-gated services,
signed-token codec, canonical FastAPI routes, nine MCP functions, CockroachDB
repositories, and separate durable worker described above. The checked-in live
gates cover API→worker→API restart, scope-safe status/recall, spanless evidence,
source rejection, effect-once replay, source→child-embedding→semantic recall,
stale-vector fencing, final-attempt lease expiry, fixed-width vector
persistence, and cross-project/repository ANN isolation. Schema tests keep
Pydantic contracts, required columns/indexes, and the additive legacy-vector
migration aligned.

Provider quality and long-running lease-heartbeat behavior remain outside this
conformance claim. The deterministic provider proves orchestration and storage,
not semantic model quality. The staged PR gates in the project's
implementation plan remain the authority for broader release evidence.
