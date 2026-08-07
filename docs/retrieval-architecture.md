# Swarm Brain standalone retrieval architecture

Status: standalone architecture; phases 0–3 code delivered, with representative
lexical/dense/hybrid benchmark exits still open
Research snapshot: 2026-08-06
Implementation snapshot: 2026-08-06

This document defines how retrieval should work in Swarm Brain as a standalone
project. It combines the current Swarm Brain implementation audit with the
PostgreSQL/CockroachDB retrieval research and turns it into a product-specific
architecture.

> This is a design document, not a claim that every component described below
> is implemented. Current implementation facts and proposed changes are labeled
> separately.

The authoritative implementation ledger is
[retrieval-status.md](retrieval-status.md). The pre-v7 audit below is retained
as the baseline that motivated the change; statements under that historical
heading must not be read as the current runtime state.

## Decision

Swarm Brain will be a separate repository, package, service, schema owner, and
release lifecycle.

Swarm Brain will own its retrieval subsystem. Sen will not be a runtime
dependency, a shared database, or a mutation authority. Sen remains useful as:

- a research source;
- a benchmark and regression harness;
- a fixture source for temporal, contradiction, multi-hop, and no-answer
  behavior;
- an optional external candidate provider in the future, behind a neutral
  protocol.

The default product architecture is therefore:

```text
Swarm Brain
├── coordination kernel
├── canonical temporal memory
├── evidence and trust
├── retrieval engine
├── rebuildable CockroachDB projections
├── HTTP API
└── MCP bridge
```

The architecture explicitly rejects:

- runtime imports from Sen;
- shared Sen/Swarm Brain tables;
- cross-project foreign keys;
- synchronous dual-write;
- direct writes to canonical memory by a retrieval service;
- copying `Session.recall` or a complete Sen repository implementation;
- post-retrieval authorization as the only security boundary.

## Why this boundary

Swarm Brain and Sen solve overlapping but different problems.

Swarm Brain is a coordination and memory product for heterogeneous coding-agent
swarms. It must own:

- authenticated tenant/project/repository/swarm/run/agent scope;
- task DAGs, claims, leases, checkpoints, and crash handoff;
- source/evidence provenance;
- temporal and bitemporal memory;
- memory confirmation, refutation, and supersession;
- poisoning guards and conflict resolution;
- public HTTP and MCP contracts;
- database schema and migrations.

Sen is a broader temporal-memory kernel and retrieval research implementation.
Its strongest value to Swarm Brain is behavioral knowledge, not class reuse.

The standalone decision is already reflected in the current project:

- [`README.md`](../README.md) declares that Swarm Brain does not import Sen or
  Mnemotree at runtime;
- [`pyproject.toml`](../pyproject.toml) defines an independent package and
  dependency set;
- [`src/swarmbrain/`](../src/swarmbrain/) contains independent domain,
  application, ports, adapters, HTTP, MCP, CLI, and worker layers.

There is no meaningful runtime import coupling to remove. The current v1
retrieval subsystem establishes that product boundary; the remaining work is
the measured dense/source/graph expansion described in the later phases.

## Scope

This architecture covers:

- current and historical memory recall;
- task-claim bootstrap context;
- exact, lexical, fuzzy, dense, temporal, source, and graph retrieval;
- CockroachDB-native projections;
- fusion, reranking, diversity, and packing;
- asynchronous projection maintenance;
- projection failures and rebuilds;
- optional external candidate providers;
- repository extraction and migration from the current monorepo incubation.

It does not define:

- model training;
- a public OAuth deployment;
- a general-purpose web search product;
- automatic mutation authority for retrieval models;
- arbitrary SQL execution by agents;
- full GraphRAG or ColBERT implementation in the first production release.

## SOTA retrieval pattern

State-of-the-art retrieval is not a single vector index. The robust production
pattern is a multi-stage system:

```text
authenticated scope / ACL / tenant / time
                    ↓
purpose + intent + entity + temporal parsing
                    ↓
┌ exact ┬ lexical ┬ fuzzy ┬ dense ┬ temporal ┬ source ┬ graph ┬ hierarchy ┐
                    ↓
canonical resource collapse
                    ↓
rank fusion
                    ↓
authoritative hydration
                    ↓
optional reranking
                    ↓
deduplication + diversity + source expansion
                    ↓
context packing + sufficiency
                    ↓
result, corrective retry, or abstention
```

The fresh direct proof point is
[Hindsight, ACL 2026](https://aclanthology.org/2026.acl-demo.27/), which uses
PostgreSQL/pgvector and combines vector, keyword, graph, and temporal retrieval.
The architectural lesson is more important than its headline benchmark score:
several complementary candidate generators are fused and reranked.

[BEIR](https://arxiv.org/abs/2104.08663) provides the broader lesson that no
single retriever wins every domain. Swarm Brain should therefore preserve
per-lane observability and evaluate combinations on coding-agent workloads.

## Two independent axes

Swarm Brain must distinguish domain lanes from retrieval signals.

### Domain lanes

A domain lane describes what a resource means:

| Domain lane | Swarm Brain content |
|---|---|
| Knowledge | observation, invariant, hypothesis, decision |
| Playbook | procedure, warning, gotcha, strategy |
| Execution history | attempt, outcome, failure, recovery |
| Handoff | checkpoint, unfinished work, crash recovery |
| Evidence/source | source code, command output, test result, log, document, artifact |
| Coordination | task state, dependency, completion, open conflict |
| Graph | supersedes, supports, contradicts, related-to, task dependency |
| Summary | task, run, repository, or topic summary |

Current and planned open semantic labels are a good foundation. Domain lane
should be a derived projection property rather than a closed storage enum.

Recommended mapping:

| Built-in kind | Default domain lane |
|---|---|
| `observation` | knowledge or execution history, based on metadata |
| `invariant` | knowledge |
| `hypothesis` | knowledge |
| `decision` | knowledge |
| `attempt` | execution history |
| `outcome` | execution history |
| `procedure` | playbook |
| `warning` | playbook |
| `handoff` | handoff |
| custom namespaced kind | configured mapping or knowledge fallback |

Custom memory kinds may provide a non-authoritative lane hint in namespaced
metadata, for example:

```json
{
  "retrieval": {
    "domain_lane": "playbook",
    "representations": ["canonical", "summary"]
  }
}
```

The server validates and normalizes the hint. It is not trusted for
authorization or lifecycle decisions.

### Retrieval signals

A retrieval signal describes how a resource was found:

- `exact`;
- `structured`;
- `lexical`;
- `fuzzy`;
- `dense`;
- `temporal`;
- `entity`;
- `graph`;
- `neighborhood`;
- `summary`;
- `reranker`.

One procedure may be found by exact path lookup, lexical matching, dense
similarity, and a link from an earlier failed attempt. Those are four signals
for one canonical resource.

## Pre-v7 implementation assessment (historical baseline)

### What is already strong

At the pre-v7 baseline, the implementation already provided:

- server-derived `ActorContext` and capability checks;
- tenant/project/repository/swarm/run/task visibility;
- append-only memory versions;
- valid-time and recorded-time intervals;
- source/evidence relationships;
- source rejection and rollback;
- supersession and lineage;
- conservative write policy;
- conflict reporting and resolution;
- idempotent mutations;
- transactional events and outbox rows;
- durable work queue infrastructure;
- in-memory and CockroachDB adapters;
- canonical HTTP and seven-tool MCP surfaces.

The canonical memory contract lives in
[`domain/memory.py`](../src/swarmbrain/domain/memory.py). The current
CockroachDB schema contains sources, chunks, memories, evidence, links,
conflicts, events, outbox, and work state in
[`adapters/cockroach/schema.sql`](../src/swarmbrain/adapters/cockroach/schema.sql).

### Pre-v7 recall was not an indexed lexical retriever

The pre-v7 CockroachDB recall implementation in
[`adapters/cockroach/memory.py`](../src/swarmbrain/adapters/cockroach/memory.py):

1. applies authoritative scope/state/trust/time predicates;
2. orders eligible rows by recency;
3. reads at most 100–2000 rows;
4. calculates token overlap in Python;
5. sorts that bounded pool by the resulting score.

Consequences:

- a relevant older procedure can be absent from the candidate pool;
- relevance is evaluated after a recency cutoff;
- broad repositories eventually lose recall;
- the database does not use an FTS index;
- source chunks, graph links, conflicts, and checkpoints are not candidate
  generators.

The scorer in
[`adapters/cockroach/memory_mappers.py`](../src/swarmbrain/adapters/cockroach/memory_mappers.py)
uses token overlap plus a substring bonus. This is a deterministic fallback,
not a production lexical search engine.

### Pre-v7 score-zero results prevented clean abstention

The public query defaults to `min_score=0`. Before retrieval v1, a memory with
no lexical overlap could still be returned with the reason `scope_match`.

This is unsafe for task bootstrap because an unrelated recent memory can enter
the context simply because it is visible. Retrieval should return an empty
result when there is no relevant evidence, except for deterministic
must-include records selected by a purpose-specific plan.

### MemoryStore owns too many responsibilities

The current [`MemoryStore`](../src/swarmbrain/ports/memory_store.py) owns:

- canonical writes;
- canonical reads;
- lifecycle validation;
- source rejection;
- final recall;
- final ranking.

[`MemoryService.recall`](../src/swarmbrain/application/memory_service.py)
therefore delegates final retrieval to one storage adapter.

This boundary makes it difficult to:

- execute lanes in parallel;
- add a second storage backend;
- preserve score breakdowns;
- rerank independently of persistence;
- test fusion separately;
- substitute a remote candidate provider;
- compare exact, lexical, dense, and graph ablations.

### Existing data is underused

The schema already contains retrieval-relevant data:

- `source_chunks`;
- `memory_evidence`;
- `memory_links`;
- `memory_conflicts`;
- task dependencies;
- checkpoints;
- completions;
- durable work effects.

Current ordinary recall searches canonical memories and now uses
`memory_links` as a bounded second-stage expansion lane. Source chunks,
conflicts, task dependencies, checkpoints, completions, and durable work
effects remain future candidate or expansion lanes.

### Lineage is not bounded graph retrieval

The public query exposes `include_lineage`, but ordinary recall does not use it.
The dedicated lineage lookup traverses a connected component recursively.

Graph retrieval needs a different contract:

- seed from exact/lexical/dense/entity hits;
- maximum 1–2 hops by default;
- cycle prevention;
- relation allowlist;
- per-edge decay;
- scope/trust/time checks on every node and edge;
- maximum fan-out and total node budget;
- path provenance.

That contract is now implemented for `memory_links`. Direct lanes fuse first;
their strongest candidates (plus explicit private seeds) drive a fixed one- or
two-hop expansion. The plan records seed count, per-node fan-out, total edge
budget, relation allowlist, and graph candidate budget. Path-local visited-node
sets prevent cycles. Relation, direction, hop, and bounded query-text overlap
produce the graph rank; graph rank then contributes through weighted RRF like
every other signal.

The CockroachDB adapter does not use the existing unbounded lineage recursive
CTE on the query critical path. It batches covering scans over separate
source/type and target/type indexes, over-fetches at most four times fan-out,
canonically validates those endpoints, and assigns the eight fan-out slots
only to eligible nodes. Every next frontier therefore already satisfies scope,
lifecycle, bitemporal, and trust policy. The complete edge/node sequence stays
in `Candidate.path`, while final hydration remains authoritative.

### Dense compatibility was short of the final lane contract (historical)

An isolated P2 implementation adds:

- CockroachDB `VECTOR`;
- a vector index;
- embedding provider ports;
- a durable embedding worker;
- evidence registration;
- tiered supersession.

Useful ideas to retain:

- embedding provider as a narrow port;
- vector projection separate from canonical content;
- ANN returns IDs, not authoritative content;
- canonical hydration after ANN;
- durable provider work;
- deterministic fake provider for tests.

Remaining problems before dense becomes a first-class v1 lane:

- the fixed-width vector plane has passed its live gate, but lacks canonical
  resource version/content digest/projection signature in each row;
- ANN natively filters tenant/repository/model but not visibility scope;
- ANN selects `query.limit` before authoritative filtering, reducing filtered
  recall;
- lexical and cosine are combined through `max(score)` despite incomparable
  scales;
- the schema fixes `VECTOR(1024)` while configuration accepts other
  dimensions;
- internal `memory_ids` selection leaks into the public query contract;
- embedding work is enqueued after the canonical commit and relies on client
  retry to repair a crash window;
- it needs reconciliation with flexible JSON memory and open semantic labels.

The evidence/lifecycle and fixed-width compatibility work was preserved. Schema
v8 now supplies the separate signed, versioned, scope-keyed dense projection
described later in this document; dense participates in the same
candidate/version/trace/RRF contract. This subsection remains the baseline that
motivated v2, not the current runtime state.

## Target package architecture

Recommended standalone package structure:

```text
src/swarmbrain/
├── domain/
│   ├── memory.py
│   ├── evidence.py
│   ├── tasks.py
│   └── retrieval.py
├── application/
│   ├── memory_service.py
│   ├── retrieval_service.py
│   ├── coordination.py
│   └── projection_service.py
├── retrieval/
│   ├── planner.py
│   ├── orchestrator.py
│   ├── fusion.py
│   ├── reranking.py
│   ├── diversity.py
│   └── packing.py
├── ports/
│   ├── memory_store.py
│   ├── retrieval.py
│   ├── embeddings.py
│   ├── reranker.py
│   └── projection_queue.py
├── adapters/
│   ├── cockroach/
│   │   ├── memory.py
│   │   ├── retrieval.py
│   │   ├── projections.py
│   │   └── schema.sql
│   ├── memory/
│   ├── embeddings/
│   └── external_retrieval/
├── workers/
│   ├── extraction.py
│   └── projections.py
└── transports/
    ├── http/
    └── mcp/
```

Dependencies continue to point inward:

```text
transport/adapters → application/retrieval → ports/domain
```

No CockroachDB row, HTTP request, MCP object, or provider SDK type crosses into
the domain/application contracts.

## Core retrieval contracts

### Retrieval purpose

Retrieval purpose is server-selected context, not arbitrary model authority:

```text
interactive_recall
task_bootstrap
handoff_recovery
planning
conflict_review
historical_audit
repository_orientation
```

The same query text can produce different lane plans for different purposes.

### RetrievalPlan

Conceptual contract:

```python
class RetrievalPlan:
    purpose: RetrievalPurpose
    intent: RetrievalIntent
    domain_lanes: frozenset[str]
    signal_lanes: frozenset[str]
    world_at: datetime | None
    recorded_at: datetime | None
    hard_scope: RetrievalScope
    lane_budgets: Mapping[str, int]
    max_graph_hops: int
    graph_seed_limit: int
    graph_max_fanout: int
    graph_edge_budget: int
    graph_link_types: frozenset[str]
    rerank: bool
    diversify: bool
    token_budget: int | None
```

`hard_scope` is derived from authenticated actor state and validated request
selectors. It is never supplied by a model as identity.

### Candidate

```python
class Candidate:
    resource_type: str
    resource_id: str
    resource_version: int
    canonical_id: str
    domain_lane: str
    signal: str
    rank: int
    raw_score: float | None
    projection_id: str | None
    projection_version: str | None
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    path: tuple[str, ...]
```

Candidates contain references and retrieval diagnostics. They are not
authoritative content.

### CandidateBatch

```python
class CandidateBatch:
    lane: str
    candidates: tuple[Candidate, ...]
    examined_count: int
    latency_ms: float
    truncated: bool
    degraded: bool
    projection_watermark: str | None
```

### RetrievalGateway

```python
class RetrievalGateway(Protocol):
    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
    ) -> CandidateBatch: ...
```

Graph expansion has a separate staged port because it consumes the first RRF
result rather than running independently beside the direct lanes:

```python
class GraphExpansionGateway(Protocol):
    async def expand(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        seeds: Sequence[FusedCandidate],
    ) -> CandidateBatch: ...
```

The name intentionally does not mention Sen, CockroachDB, pgvector, or a
specific model.

### CanonicalMemoryReader

```python
class CanonicalMemoryReader(Protocol):
    async def hydrate_recallable(
        self,
        actor: ActorContext,
        query: RecallQuery,
        candidate_ids: Sequence[str],
    ) -> tuple[Memory, ...]: ...
```

`candidate_ids` is an internal selector. It should not become a model-visible
field of `RecallQuery`.

### RetrievalTrace

The internal trace should retain:

- purpose and selected plan;
- parsed identifiers/entities/time;
- per-lane budget;
- per-lane candidate count;
- raw score and rank;
- latency;
- projection version/watermark;
- fusion contribution;
- hydration rejection reason;
- reranker result;
- dedup/diversity decision;
- packing decision;
- degraded lanes.

Public responses may expose a bounded safe summary. Full traces belong in
audited diagnostics.

## Read path

Recommended orchestration:

```text
RecallQuery + authenticated ActorContext
                  ↓
RetrievalPlanner
                  ↓
parallel candidate generation
  exact | FTS | trigram | dense | temporal | source | graph
                  ↓
collapse representations to canonical IDs
                  ↓
weighted RRF
                  ↓
canonical hydration + revalidation
                  ↓
bounded expansion
                  ↓
optional temporal-aware reranker
                  ↓
dedup/diversity
                  ↓
context packing
                  ↓
RecallBundle + safe trace summary
```

### Defense in depth

Every candidate-generating lane must apply as many authoritative hard
constraints as its index supports before its own `LIMIT`:

- tenant;
- project/repository;
- visibility;
- run/task;
- state/currentness;
- world time;
- recorded time;
- rejected/untrusted source rules.

After fusion, all candidates are hydrated through the canonical reader and
revalidated. Revalidation is defense in depth and stale-projection protection;
it must not be the only filter protecting tenant scope.

### Fresh overlay

Asynchronous projections can lag. The read path should add a small canonical
overlay:

- recently written current memories;
- latest task checkpoint/handoff;
- explicit IDs referenced by the active task;
- records whose projection work is still pending.

This provides read-your-writes without making vector generation synchronous.

### Abstention

The engine should return no relevance hits when:

- all scores are below the lane threshold;
- only scope matches exist;
- all candidates fail canonical hydration;
- evidence is insufficient for a high-stakes purpose.

Purpose-specific must-include records, such as the latest checkpoint for the
claimed task, are labeled deterministic context rather than relevance hits.

## Query profiles

### Interactive recall

Default flow:

1. exact identifiers;
2. lexical and dense in parallel;
3. temporal lane when time language is present;
4. first RRF over direct lanes;
5. graph/source expansion when justified;
6. final RRF including staged expansion;
7. canonical hydration;
8. optional reranker;
9. evidence-aware packing.

### Task bootstrap

The current claim path now selects a server-owned `task_bootstrap` purpose. Its
query includes title, description, tags, required capabilities, latest
checkpoint summary, discoveries, remaining work, and private checkpoint memory
ID seeds. It retrieves flat v1 hits and labels the implemented handoff,
playbook, prior-attempt, and knowledge sections in `reasons`.

The fuller versioned bootstrap response should additionally compose:

1. latest checkpoint and handoff for the task;
2. memory IDs explicitly referenced by checkpoints;
3. current warnings, procedures, decisions, and invariants;
4. similar attempts and outcomes from the repository;
5. named files, symbols, test names, commands, commits, and errors;
6. open conflicts affecting selected memories;
7. source/evidence expansion for final results.

The result should preserve sections:

```text
must_include
handoff
playbook
prior_attempts
evidence
open_conflicts
```

This is more useful than one flat score-sorted list.

### Handoff recovery

Priorities:

- exact latest checkpoint;
- incomplete work;
- previous lease owner;
- task-local handoff memories;
- latest test/command evidence;
- repository-level warnings and procedures;
- active conflicts.

Dense similarity is supplemental, not the first lane.

### Planning

Priorities:

- confirmed decisions/invariants;
- active warnings;
- procedures;
- successful and failed outcomes;
- current task graph;
- open conflicts;
- relevant evidence.

### Conflict review

Conflict review intentionally differs from ordinary recall:

- includes refuted and superseded versions;
- expands contradiction/support links;
- includes source trust and review state;
- preserves temporal ordering;
- returns evidence paths;
- does not let topical similarity override lifecycle facts.

### Historical audit

Historical audit uses explicit `world_at` and/or `recorded_at`:

- bitemporal SQL selects the eligible corpus;
- exact/lexical operates inside that corpus;
- semantic ranking is exact over the bounded temporal candidate set or uses a
  separate historical projection;
- no global recency decay.

## CockroachDB projections

Canonical tables remain the source of truth. Retrieval tables can be dropped
and rebuilt.

### Exact and structured lane

Use canonical typed columns and B-tree/covering indexes for:

- memory/task/source IDs;
- task IDs;
- run/repository scope;
- memory kind;
- state;
- visibility;
- valid and recorded intervals;
- content digest;
- tags where suitable;
- occurrence key;
- commit hashes, paths, symbols, test IDs, and error codes extracted into
  typed or projection columns.

Highly structured facts should not be forced through embeddings.

### Lexical projection

Recommended logical shape:

```text
retrieval_documents
- resource_type
- resource_id
- resource_version
- canonical_id
- tenant_id
- project_id
- repository_id
- visibility
- scope_key
- projection_id
- domain_lane
- search_text
- lookup_text
- search_tsv
- content_sha256
- indexed_at
```

`search_text` is a deterministic textual representation of:

- title;
- string content;
- canonical JSON serialization for structured content;
- tags;
- selected typed metadata;
- stable source locator information.

`search_tsv` should be a stored computed `TSVECTOR` using `simple` as the
initial coding/multilingual baseline.

Why `simple`:

- Swarm Brain content mixes code, English, Polish, identifiers, and logs;
- CockroachDB has no Polish dictionary;
- stemming code identifiers is often harmful;
- dense retrieval handles semantic paraphrases;
- trigram handles spelling and substrings.

The initial benchmark should compare `simple` with English configuration for
English-only repositories, rather than assume one global analyzer.

### Fuzzy/identifier projection

Do not build a huge trigram index over all arbitrary memory text by default.
Maintain a bounded `lookup_text` containing:

- file paths;
- symbols;
- test names;
- command names;
- commit hashes;
- package/module names;
- error codes;
- titles;
- aliases.

Use a trigram index for:

- misspellings;
- substrings;
- partial identifiers;
- path fragments;
- symbol lookup.

Exact equality/prefix remains preferred for full IDs and known structured
fields.

### Vector projection

Recommended fixed-dimension table:

```text
retrieval_vectors_1024
- tenant_id
- repository_id
- projection_id
- scope_key
- resource_type
- resource_id
- resource_version
- canonical_id
- domain_lane
- content_sha256
- embedding VECTOR(1024)
- indexed_at
```

Recommended vector index prefix:

```text
tenant_id,
project_id,
repository_id,
resource_type,
projection_id,
projection_signature,
scope_key,
embedding vector_cosine_ops
```

CockroachDB uses vector prefix columns only when all prefixes have equality
constraints or complete tuple `IN` constraints. This directly determines the
projection design.

### Scope key

Each projected record has one stable retrieval scope:

```text
repository:<repository_id>
run:<run_id>
task:<task_id>
```

For an authenticated task query, the allowed vector trees can be selected with
complete tuples corresponding to:

- repository visibility;
- current run visibility;
- selected current task visibility.

This pushes visibility before ANN rather than relying entirely on post-filter
hydration.

CockroachDB does not accelerate arbitrary non-prefix predicates on a vector
index. Current runtime therefore validates lifecycle, bitemporal, kind,
version/digest, and trust against canonical memory in the same snapshot after
the prefix-scoped ANN result. It geometrically widens an under-filled candidate
window to a bounded cap. The exact oracle applies that canonical eligibility
before exact vector sorting; comparing ANN with this oracle measures the recall
cost of both approximation and post-filter underfill.

If the deployment uses multiple regions, scope/locality design must also keep
retrieval close to the owning tenant or repository. Cross-region global recall
should merge regional top-k results instead of accidentally performing a
wide distributed graph traversal.

### Projection ID and dimension

`projection_id` must identify:

- embedding provider;
- model;
- dimension;
- normalization;
- truncation;
- input renderer version;
- domain representation;
- current/history mode.

Example:

```text
qwen3-embedding-1024:cosine:l2norm:memory-text-v2:current
```

CockroachDB `VECTOR(n)` has a fixed dimension. Dimension is a deployment/schema
invariant, not a per-row runtime preference.

Model migration options:

1. create a new projection table/index;
2. dual-build asynchronously;
3. compare old/new retrieval;
4. switch the active projection pointer;
5. retain rollback capability;
6. remove the old projection after the safe window.

Do not reinterpret an existing vector table under a new model signature.

### Current versus historical semantic retrieval

Dynamic time ranges are not useful vector prefix columns.

Current recall:

- uses a current-only vector projection;
- superseded/refuted/source-rejected resources are removed or tombstoned;
- canonical hydration rejects stale projection rows;
- overfetch accounts for temporary lag.

Historical recall:

- first applies bitemporal SQL;
- exact-scores vectors for the bounded eligible set;
- or uses a separate historical projection keyed by a coarse, equality
  selectable bucket;
- never applies current-only global recency logic.

### ANN tuning

Measure CockroachDB ANN against exact vector ground truth:

- Recall@k;
- filtered Recall@k;
- p50/p95/p99;
- CPU;
- rows and bytes read;
- tenant/repository size;
- scope selectivity;
- fresh versus old records;
- search beam;
- partition sizes;
- candidate overfetch.

Do not claim that ANN is correct because a `vector search` plan appears.
`EXPLAIN ANALYZE` validates the plan; exact oracle comparison validates recall.

### Source/evidence lane

`source_chunks` should become a searchable projection with inherited,
denormalized scope fields sufficient for pre-limit filtering.

Candidate types:

- source code span;
- command output span;
- test output span;
- log span;
- document chunk;
- artifact metadata.

Search the smallest evidence unit, then expand to:

- source;
- neighboring chunks;
- canonical memory derived from it;
- precise evidence locator.

Evidence search does not bypass source trust/review state.

### Graph lane

The implemented slice uses existing memory links. Task dependencies remain a
future structural lane.

Recommended graph candidate contract includes:

- target resource;
- path;
- relation sequence;
- edge evidence;
- cumulative decay;
- hop count.

Default limits:

- 1–2 hops;
- relation allowlist by purpose;
- bounded fan-out;
- cycle prevention;
- tenant/repository/run/task and time checks at each step.

Large PPR/community processing belongs in asynchronous projections, not the
critical query path.

Current policy is deliberately small and reproducible:

- explicit seeds first, then top direct fused hits; at most 16 seeds;
- one hop for interactive/historical/orientation, two for
  bootstrap/handoff/planning/conflict review;
- eight canonically eligible neighbors per node;
- at most four-times-fan-out raw edge over-fetch per node and a separate total
  edge budget;
- only built-in `supports`, `supersedes`, `derived_from`, `merged_from`,
  `contradicts`, `duplicate_of`, and `related_to` relations;
- relation-specific weight, `0.85` step decay, reverse-direction penalty for
  asymmetric relations, and a bounded query-overlap gate with a `0.60` floor;
- best deterministic path per canonical target and full path provenance;
- graph RRF weight below direct exact/lexical/dense evidence, except for a
  modest conflict-review increase.

The in-memory and CockroachDB adapters share the scoring contract. CockroachDB
uses fixed-depth application iteration because it exposes fan-out and total
budgets directly and keeps both directional index scans auditable. PostgreSQL
recursive `CYCLE`/path arrays remain useful for offline lineage; CockroachDB's
own documentation warns that some recursive CTEs are not yet optimized, so an
unbounded connected-component query is not used for synchronous recall.

### Summary/hierarchy lane

Later stages may add:

- task summaries;
- run summaries;
- repository playbook summaries;
- topic summaries;
- parent/child representations.

Every generated summary must preserve:

- source IDs;
- child memory IDs and versions;
- generator/model signature;
- creation time;
- validity/freshness status;
- rebuildability.

Summary drift and stale lineage make this a later-phase feature.

## Fusion

### Start with weighted RRF

Candidate score scales are not comparable:

- exact match;
- `ts_rank`;
- trigram similarity;
- cosine similarity;
- recency;
- graph activation.

Use [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
as the initial fusion:

```text
score(candidate) = Σ lane_weight / (k + rank_in_lane)
```

`k=60` is a conventional starting point, not a guaranteed optimum.

Initial lane weights should be purpose-specific and treated as configuration
with explicit versioning. Examples:

- exact identifiers dominate interactive code lookup;
- handoff/checkpoint dominates task bootstrap;
- temporal dominates historical audit;
- procedure/warning receives a boost for planning;
- graph is disabled unless the plan justifies it.

Do not use `max(lexical_score, cosine_score)`. It compares uncalibrated scales
and discards agreement between lanes.

### Learned fusion

Once labeled retrieval data exists, compare RRF against:

- calibrated weighted score combination;
- feature-based learning-to-rank;
- purpose-specific logistic models.

Features can include:

- per-lane rank and score;
- number of agreeing lanes;
- memory state/confidence;
- evidence trust;
- temporal relation;
- domain lane;
- task/repository proximity;
- projection freshness.

Hard authorization and lifecycle rules remain outside learned ranking.

## Reranking

Cross-encoder reranking is appropriate for a small fused pool, for example
top 30–100.

The reranker must not receive mutation authority. It ranks eligible evidence.

A generic topical reranker can damage:

- chronological relevance;
- state-as-of correctness;
- update/supersession ordering;
- multi-hop evidence coverage;
- source diversity.

Swarm Brain's future reranker should include:

- purpose/intent;
- valid and recorded time;
- current/superseded/refuted state;
- source trust;
- evidence coverage;
- graph path;
- task/run proximity.

If no temporal-aware reranker is available, deterministic ordering is safer for
historical audit and conflict review.

## Diversity and packing

After reranking:

- collapse multiple representations to one canonical resource;
- group hits from the same source/session/task;
- preserve independent evidence for multi-hop;
- apply MMR or a simpler diversity rule;
- expand selected evidence to source/neighbors;
- pack under an explicit token/record budget.

Packing should prefer:

- concise canonical memory;
- exact citations;
- enough raw evidence to verify the claim;
- multiple required hops;
- latest checkpoint for bootstrap;
- warnings before redundant observations.

Do not let a single verbose source crowd out all other evidence.

## Write and projection path

### Atomic intent

The canonical mutation transaction should persist:

```text
canonical memory/source/evidence change
+ swarm event
+ outbox/projection work item
```

The transaction must not call:

- embedding providers;
- rerankers;
- remote Sen;
- artifact networks;
- external search engines.

### Durable projection work

Projection work should be inserted inside the canonical mutation transaction,
not enqueued in an unrelated after-commit call that relies on the client to
retry.

Dedupe key:

```text
retrieval:{resource_type}:{resource_id}:{resource_version}:{projection_id}
```

Worker flow:

1. lease work with fencing;
2. load the canonical snapshot;
3. verify resource version;
4. render deterministic search text;
5. generate embedding outside a database transaction;
6. idempotently upsert the projection;
7. reject an older result if a newer resource version exists;
8. record effect/attempt;
9. complete the work lease.

### Projection-complete change tracking

Every lifecycle operation must provide the complete set of affected resources:

- add;
- confirm;
- refute;
- supersede;
- merge;
- source rejection;
- restoration of a predecessor;
- trust/review change;
- content or metadata revision;
- projection configuration change.

An event containing only `operation=update` is insufficient for deterministic
external projection maintenance.

Two safe event patterns:

1. include a complete versioned projection snapshot;
2. include all changed IDs/versions and let the worker call an internal
   canonical snapshot reader.

The second pattern sends less sensitive content through the event stream and
keeps canonical ownership explicit.

### Failure semantics

- canonical writes succeed without embeddings;
- exact/lexical/temporal retrieval remains available during provider outage;
- worker retries are idempotent and fenced;
- duplicate events are safe;
- out-of-order older revisions are ignored;
- projection lag is observable;
- query traces report degraded lanes;
- local fresh overlay provides read-your-writes;
- projection tables can be rebuilt from canonical state/event history;
- an unavailable projection lowers quality, not authorization correctness.

## Public API and MCP

Keep the public route and MCP tool shape stable where possible.

The model-visible query should continue to express:

- text;
- task selector;
- kinds;
- visibilities;
- current/historical selectors;
- result limit;
- evidence/lineage preferences.

Server-owned behavior:

- authenticated scope;
- purpose profile;
- plan;
- lane selection;
- candidate budgets;
- provider selection;
- fusion version;
- safe degradation.

Do not add tenant, repository, run, agent, capability, or arbitrary scope keys
to MCP arguments.

Optional trusted HTTP/admin fields may select a retrieval profile, but they
cannot broaden authenticated scope.

## Optional Sen integration

### Default position

Sen is not part of production runtime.

Swarm Brain should first implement and benchmark:

- exact;
- CockroachDB FTS;
- trigram;
- C-SPANN;
- temporal SQL;
- source/evidence;
- bounded graph;
- RRF.

### When to consider a remote provider

Consider a separate retrieval service only if measured requirements cannot be
met locally, for example:

- BM25 quality materially exceeds CockroachDB FTS;
- late interaction/ColBERT is required;
- very large graph or hierarchical retrieval needs independent compute;
- search must scale independently from coordination OLTP;
- PostgreSQL extensions provide a demonstrated workload-specific advantage.

### Neutral external boundary

```text
Swarm Brain canonical transaction
          ↓
transactional outbox
          ↓ at-least-once
external projection service
          ↓
candidate IDs + ranks + diagnostics
          ↓
Swarm Brain canonical hydration
          ↓
public RecallBundle
```

The adapter name may be `SenHttpCandidateRetriever`, but the core protocol must
remain provider-neutral.

Possible internal endpoints:

```text
POST /internal/v1/projection-events
POST /internal/v1/retrieval:candidates
GET  /internal/v1/projection-status
```

External candidate response:

```json
{
  "generation": "retrieval-v3",
  "watermark": "opaque",
  "partial": false,
  "candidates": [
    {
      "resource_type": "memory",
      "resource_id": "uuid",
      "resource_version": 4,
      "canonical_id": "uuid",
      "domain_lane": "playbook",
      "rank": 1,
      "signals": [
        {"signal": "lexical", "rank": 2, "score": 7.81},
        {"signal": "dense", "rank": 1, "score": 0.83}
      ]
    }
  ]
}
```

Rules:

- deduplicate ingestion by event ID;
- order by resource version/projection revision;
- ignore stale events;
- detect version gaps and reconcile;
- acknowledge only after durable projection write;
- return IDs and diagnostics, not authoritative lifecycle state;
- rehydrate and revalidate every result in Swarm Brain;
- short timeout and circuit breaker;
- local Cockroach retrieval remains the fallback;
- no remote call inside a canonical transaction;
- no mutation or policy authority for the external service.

## Repository extraction — completed

The standalone repository was extracted with rewritten history and verified
blob-for-blob against the Sen commits. The current repository root owns
`src/`, `tests/`, `docs/`, packaging, schema, and release history; the SHA map
is in [history/sen-extraction-map.md](history/sen-extraction-map.md). There is
no runtime import or filesystem dependency on Sen/Mnemotree.

### Historical pre-extraction shape

The code is already a self-contained package inside the Sen repository. The
main non-runtime coupling is organizational:

- documentation lives partly under the parent `docs/swarm-brain/`;
- active task files live under the parent `current_tasks/`;
- README contains incubation references to parent/sibling projects;
- CI and release lifecycle are inherited from the parent workspace;
- multiple in-flight branches/worktrees must be reconciled.

### Historical safe extraction sequence

1. Finish and commit flexible-memory independently.
2. Split P2 evidence/lifecycle changes from vector/retrieval changes.
3. Rework the vector plane before merge.
4. Move active Swarm Brain documentation into this project's `docs/`.
5. Add project-local `AGENTS.md`, CI, lockfile, container, and release config.
6. Create a clean standalone release candidate.
7. Run all fast, live CockroachDB, restart, HTTP, and MCP gates without access
   to parent Sen files.
8. Extract the `swarm-brain/` history into a new repository.
9. Publish/build from the new repository.
10. Keep the old subtree temporarily as a read-only migration pointer.
11. Remove the old subtree only after history, artifacts, docs, and release
    verification.

History can be preserved through a subtree split or a path-filtered repository
rewrite. Documentation should be moved under `swarm-brain/` before the final
split if it must retain the same history boundary.

### Standalone repository checklist

```text
swarm-brain/
├── AGENTS.md
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── docs/
├── src/
├── tests/
├── benchmarks/
├── scripts/
└── .github/workflows/
```

The standalone project must not depend on:

- relative paths into Sen;
- Sen test fixtures loaded directly from the filesystem;
- Sen environment variables;
- Sen schema installation;
- parent repository caches or task state.

Benchmark fixtures can be copied under an explicit license/provenance record or
consumed through a published artifact.

## Migration phases

### Phase 0 — establish the product boundary

Status: complete in the standalone repository.

Deliverables:

- flexible JSON memory and open semantic vocabulary committed;
- P2 split into reviewable changes;
- docs inside `swarm-brain/docs`;
- standalone build/test configuration;
- no parent filesystem dependency.

Exit criteria:

- package builds and all existing tests pass from the project directory;
- docs links do not escape to required parent files;
- runtime import scan finds no Sen/Mnemotree dependency.

### Phase 1 — retrieval contracts and correctness baseline

Status: implemented on `feat/retrieval-v1`; unit and live CockroachDB suite
passed for the branch handoff.

Deliverables:

- `RetrievalPlan`;
- `Candidate` and `CandidateBatch`;
- `RetrievalTrace`;
- `RetrievalGateway`;
- canonical hydration port;
- exact candidate lane;
- no zero-score relevance hits;
- purpose-specific task bootstrap;
- one canonical adapter snapshot across candidate generation and hydration,
  including CockroachDB dense v2 ANN lookup;
- per-lane CockroachDB savepoints so one SQL failure remains a degraded signal
  instead of aborting the shared snapshot;
- gold retrieval fixture suite.

Exit criteria:

- current public API/MCP remains compatible;
- no-result queries abstain;
- old exact matches outside the newest 2000 records are found;
- hard scope/state/trust/time applies before lane limits.

### Phase 2 — indexed lexical and identifiers

Status: implementation delivered as CockroachDB schema v7 plus the in-memory
reference adapter. Focused identifier, live exact/FTS/trigram/RRF/abstention,
actual runtime SQL `EXPLAIN`, and pre-v7 rebuild gates pass. The measured
lexical-only baseline below remains an evaluation exit, so phase 2 is not
declared benchmark-complete.

Deliverables:

- deterministic `search_text`;
- FTS `simple` projection;
- inverted index;
- bounded `lookup_text`;
- trigram signal;
- exact/lexical/fuzzy diagnostics;
- NFKC/casefold query parity and bounded SQL parameters;
- transactional, retryable pre-v7 rebuild with stale projection cleanup;
- RRF fusion.

Exit criteria:

- paths, symbols, tests, commands, hashes, and errors have focused tests;
- lexical retrieval is database-indexed;
- no Python recency pre-cutoff determines lexical recall;
- lexical-only baseline is measured.

### Phase 3 — vector projection

Status: implementation delivered as CockroachDB schema v8 plus an exact
in-memory reference lane. The signed current projection, equality-bound
repository/run/task ANN branches, same-snapshot canonical validation with
bounded adaptive widening, RRF/trace integration, durable fenced dual-write,
configurable cosine floor/beam, and exact primary-index oracle have focused
tests. A small live correctness gate is not a representative
quality/performance benchmark, so the measured exit criteria below remain
open.

Deliverables:

- fixed-dimension vector projection;
- scope-key vector prefix;
- projection/model signature;
- atomic durable projection work;
- idempotent worker;
- canonical hydration;
- overfetch;
- exact vector oracle;
- live CockroachDB vector tests.

Exit criteria:

- vector dimension mismatch fails at startup/schema verification;
- every vector query constrains the full prefix;
- filtered Recall@k is measured across visibility/selectivity buckets;
- projection lag and stale candidates are observable;
- dense-only and hybrid results are benchmarked.

### Phase 4 — source, evidence, temporal, and graph

Status: the bounded `memory_links` graph slice is implemented with in-memory
and CockroachDB parity, staged RRF, path provenance, same-snapshot canonical
checks, directional covering indexes, and live `EXPLAIN` gates. Source/evidence,
historical dense, checkpoint/handoff, task-dependency, and conflict projections
remain open.

Deliverables:

- searchable source chunks;
- source/evidence expansion;
- current/history routing;
- historical exact semantic ranking;
- bounded graph retrieval;
- checkpoint/handoff lane;
- conflict-review profile.

Exit criteria:

- source rejection removes affected current recall;
- restored predecessors reappear correctly;
- historical point/range/as-of fixtures pass;
- graph never exceeds configured hop/fan-out budgets;
- every graph result has path provenance.

### Phase 5 — reranking and packing

Deliverables:

- optional reranker port;
- purpose-aware rerank policy;
- diversity;
- canonical representation collapse;
- token-budget packing;
- sufficiency/abstention;
- end-to-end answer/citation evaluation.

Exit criteria:

- reranker improves relevant profiles without regressing temporal/conflict
  profiles;
- degraded mode works without the reranker;
- multi-hop evidence coverage survives diversity;
- packing is deterministic for a fixed candidate set/config.

### Phase 6 — optional external retrieval

Only after a measured need:

- projection event contract;
- relay;
- external candidate API;
- snapshot/backfill/reconciliation;
- circuit breaker;
- local fallback;
- provider contract tests.

Exit criteria:

- external service can be removed without losing canonical data;
- outage degrades quality but not coordination or authorization;
- duplicate/reordered events converge;
- candidate versions/watermarks are verified;
- Swarm Brain remains the only public/security boundary.

## Verification and benchmarks

### Retrieval quality

Measure:

- Recall@k per lane;
- fused Recall@k;
- MRR;
- nDCG;
- all-required-evidence recall;
- temporal precision;
- state-as-of correctness;
- contradiction/update correctness;
- abstention/no-answer accuracy;
- source/citation correctness.

### Security and policy

Mandatory invariants:

- zero cross-tenant leaks;
- zero cross-repository leaks;
- zero run/task visibility leaks;
- rejected/untrusted sources excluded by default;
- refuted/superseded records excluded by default;
- external candidate IDs cannot bypass hydration;
- model-supplied metadata cannot broaden scope;
- retrieval output cannot confirm/refute/supersede memory.

### ANN

Measure against exact vector search:

- Recall@k;
- filtered Recall@k;
- candidate underfill;
- tenant/repository size;
- scope selectivity;
- filter/vector correlation;
- beam and partition parameters;
- overfetch;
- stale projection rate;
- p50/p95/p99.

### Performance and operations

Measure:

- end-to-end recall latency;
- per-lane latency;
- QPS;
- CPU;
- rows/bytes read;
- vector/FTS storage;
- write amplification;
- projection throughput;
- projection freshness lag;
- rebuild time;
- worker retry/dead-letter behavior;
- model/token cost.

### Required fixtures

At minimum:

- exact task/memory/source ID;
- file path;
- symbol;
- test name;
- command;
- commit hash;
- error code;
- semantic paraphrase;
- old relevant record beyond 2000 recent rows;
- zero-match/no-answer;
- latest checkpoint;
- crash handoff;
- successful previous procedure;
- failed previous attempt;
- temporal point/range/as-of;
- supersession;
- source rejection and predecessor restoration;
- contradiction;
- two-hop evidence;
- cross-tenant/repository/run/task isolation;
- projection lag;
- embedding outage;
- duplicate and reordered projection work.

### Ablations

Every material retrieval release should compare:

```text
exact only
lexical only
dense only
exact + lexical
lexical + dense
hybrid + temporal
hybrid + source
hybrid + graph
hybrid + reranker
full pipeline
```

Evaluate candidate recall and final task/answer behavior. A raw recall
improvement can disappear after reranking, deduplication, or context
truncation.

## Acceptance criteria for the standalone architecture

### Repository

- New repository builds without Sen or Mnemotree.
- No runtime imports or required filesystem links to parent projects.
- Documentation, migrations, tests, and release config are project-owned.
- Schema installation and verification are owned only by `swarmbrain-schema`.

### Canonical correctness

- CockroachDB remains the source of truth.
- Mutation, event, and projection work intent are atomic.
- Projection tables can be deleted and rebuilt.
- Duplicate, reordered, and restarted workers cannot regress a projection.
- Source rejection and supersession converge after replay/rebuild.

### Retrieval

- Exact/lexical works without an embedding provider.
- No zero-score relevance results.
- Hard filters execute before every feasible lane limit.
- All results undergo canonical hydration.
- RRF preserves per-lane contributions.
- Historical queries preserve bitemporal semantics.
- Task bootstrap has a dedicated plan and structured context sections.
- Projection outage is visible and safely degraded.

### CockroachDB

- Vector dimension and model signature are verified.
- Vector queries constrain complete prefix tuples.
- Vector index use is checked with `EXPLAIN`.
- Recall is checked against an exact oracle.
- Multi-region locality is explicitly tested when enabled.
- Live VECTOR, restart, and schema-upgrade gates pass before release.

### Public boundary

- HTTP and seven MCP tools remain compatible unless intentionally versioned.
- Actor identity remains server-derived.
- No public query accepts tenant/repository/agent identity.
- Retrieval providers never receive mutation authority.
- Optional external retrieval is replaceable and fail-safe.

## Recommended next sequence after retrieval v1

1. Add resource version/content digest/projection signature to the vector row.
2. Bring dense behind `CandidateBatch`, canonical eligibility validation, RRF,
   and trace.
3. Measure ANN Recall@k against an exact vector oracle.
4. Measure the delivered memory-link graph slice, then add
   source/handoff/temporal/task-dependency lanes in measured slices.
5. Persist bounded audited traces and lane latency/underfill/freshness metrics.
6. Add reranking only after lane ablations.
7. Consider an external Sen provider only after a measured local gap.

## Final recommendation

Build Swarm Brain as a standalone coordination and memory product with a
provider-neutral retrieval engine and CockroachDB-native projections.

Use Sen to transfer:

- bitemporal and source-preserving semantics;
- retrieval evaluation methodology;
- intent/lane ideas;
- RRF and ranking diagnostics;
- temporal, contradiction, multi-hop, and abstention fixtures.

Do not transfer:

- the full Sen session orchestrator;
- broad repository classes;
- benchmark-specific heuristics without local ablation;
- personal-memory ontology unrelated to coding swarms;
- mutation authority based on similarity;
- hidden coupling to the Sen repository.

This preserves independent ownership and releases while keeping a clean future
seam for an optional external candidate provider.

## Primary references

- [Hindsight, ACL 2026](https://aclanthology.org/2026.acl-demo.27/)
- [BEIR](https://arxiv.org/abs/2104.08663)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [pgvector](https://github.com/pgvector/pgvector)
- [PostgreSQL full-text search](https://www.postgresql.org/docs/current/textsearch-controls.html)
- [CockroachDB vector indexes](https://www.cockroachlabs.com/docs/v26.2/vector-indexes)
- [CockroachDB full-text search](https://www.cockroachlabs.com/docs/v26.2/full-text-search)
- [CockroachDB trigram indexes](https://www.cockroachlabs.com/docs/v26.2/trigram-indexes)
- [CockroachDB recursive CTEs](https://www.cockroachlabs.com/docs/v26.2/common-table-expressions)
- [PostgreSQL recursive search and cycle detection](https://www.postgresql.org/docs/current/queries-with.html)
- [CockroachDB table localities](https://www.cockroachlabs.com/docs/v26.2/table-localities)
- [SPLADE v2](https://arxiv.org/abs/2109.10086)
- [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/)
- [RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)
- [GraphRAG](https://arxiv.org/abs/2404.16130)
- [HippoRAG 2](https://arxiv.org/abs/2502.14802)
- [Query-Aware Spreading Activation](https://arxiv.org/abs/2606.30133)
- [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)
