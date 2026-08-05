# Swarm Brain donor source map

This map records what was actually inspected, what is adapted, and what is
explicitly rejected. It is not a copy list: all accepted entries are semantic
or small-pattern donors for new focused Swarm Brain code.

> This is a donor-provenance snapshot. Six-tool and early-P0 inventory below is
> historical; the current seven-tool runtime and retrieval v1 state are in
> [API contracts](api.md) and [retrieval status](retrieval-status.md).

## Inspected refs

### `sen`

| Ref | Exact commit | Finding |
| --- | --- | --- |
| `split/memory-policy` and `origin/split/memory-policy` | `67e653d33fbc66680b9806ae3f3410fb061beea2` | Requested donor; an ancestor of the current checkout. |
| current `hackathon/demo-20260718` HEAD | `b32817292738c4bf5be26c707f87dbee150810e0` | 219 commits beyond the requested donor; preferred where it has stricter storage, trust, reconciliation, or CockroachDB behavior. |
| `main` | `875cd58dba95cf1b2d7fcb75dbca4d05984c8b2c` | Inspected for recency/orientation; not selected over current tested implementations. |

The current worktree was not switched or cleaned. Exact committed objects were
read with `git show`, `git grep`, and `git ls-tree`.

### `mnemotree`

The requested `agent-layer-clean` ref is missing both locally and under
`origin`; it cannot be treated as an inspected or reproducible donor.

| Ref | Exact commit | Finding |
| --- | --- | --- |
| `origin/codex/agent-layer-mcp` | `df8895a28df9f6cebc352c46d1777d98539d55b7` | Best broad agent-layer donor. |
| `origin/feature/agent-scoping` | `5376546e2dd9c13b5f59c178343608e76c67f294` | Best narrow scope-propagation donor. |
| merge base of those refs | `929a520961f63f2fd933ff04a4306d4f0461c1c8` | Used to isolate agent-layer changes. |
| current `feature/test-coverage-improvements` checkout | `80e268fa0dc20e5cca12c150bdd0eae7a9287a98` | Dirty checkout predates agent-layer changes; no requested scope/tool contracts are present. |

Relevant broad-branch commits are
`7c78b6c32e6c2de80f6ffaafe03fe0eacff88b1d` (scope and coordination
tables/tools), `a143e32d4180f9e778d0eee009afaa7ca38d36b4` (observation and
summary-first recall), `8f8b9d5a72df50336fbaf723e34ec5c3ae44d013`
(registration/configuration), and
`df8895a28df9f6cebc352c46d1777d98539d55b7` (confidence, compaction,
agentic retrieval). Narrow-branch commits
`73b3cd5b88223ccedb61b4ea8808891adbb76403` and
`5376546e2dd9c13b5f59c178343608e76c67f294` carry scope fields end to end
and forward recall filters respectively.

## Base-versus-donor decision

| Swarm Brain destination | Base implementation | Donor use |
| --- | --- | --- |
| `domain/` | New focused Pydantic v2 contracts | Sen supplies temporal/source/policy invariants; Mnemotree supplies only observation vocabulary and strict boundary inspiration. |
| `application/` | New capability-gated coordination, memory, handoff, conflict, and audit services | Mnemotree's async orchestration shape and Sen's conservative decision taxonomy are adapted in small functions. |
| `ports/` | New use-case/capability-specific async protocols | Mnemotree's capability-oriented `Protocol` organization is the selected structural donor; its raw dictionaries are rejected. |
| `adapters/cockroach/` | New schema resource and retry kernel in P0; repositories, durable idempotency, and outbox relay in P1 | Sen's current CRDB transactions are behavioral evidence; neither donor schema/repository is a code base. DDL presence is not a current durability claim. |
| `transports/http/` | New canonical FastAPI mapping | Neither donor HTTP/demo server is a base. |
| `transports/mcp/` | New six-tool stdio bridge over HTTP | Only lazy optional import, centralized registration, and JSON-safe facade patterns are adapted. |
| extraction workers | New coding-memory router and validated candidates | Mnemotree's lazy router/extractor pipeline is the structural donor; Sen's source provenance and policy remain authoritative after extraction. |

Thus the repository/package base is this self-contained repository,
not either donor tree. Every row below names the narrow donor and
the replacement boundary.

## Accepted `sen` donors

Use the listed ref for reproducibility. “Adapt” means reproduce the invariant
behind a new Swarm Brain contract or repository method; it does not mean copy
the containing module.

| Ref and path | Exact symbols/regions | Adaptation |
| --- | --- | --- |
| `67e653d...:src/sen/models.py` | `SourceRecord` (57–66), `SourceChunk` (69–79), `MemoryEvent` (95–100), `FactVersion` (305–330), `FactWriteResult` (333–341) | Source/chunk provenance, bitemporal bounds, supersession pointers, and explicit write outcome. Prefer current `SourceRecord.occurrence_key` at `b328172...:src/sen/models.py` (57–67), introduced by `893b0108c36dfeb79e72c698f736386d59a11b256`, so identical content from distinct occurrences is not collapsed. |
| `67e653d...:src/sen/storage/sqlite/repository.py` | source ingestion region 616–795 | Behavioral reference for source-preserving ingestion only. Prefer current `SourceInsertResult` and the source contract in `b328172...:src/sen/storage/base.py` (49–60), plus shared repository contract tests. |
| `b328172...:src/sen/storage/shared.py` | `_normalize_text`, `_text_sha256`, `critical_tokens`, `can_merge_non_slotted_duplicate` (258–285) | Conservative exact/near duplicate policy; critical-token mismatch blocks an unsafe merge. The current Cockroach implementation at `src/sen/storage/crdb/repository.py` (1424–1631) is a behavior reference, not a class donor. |
| `67e653d...:src/sen/storage/sqlite/repository.py` | fact-write/timeline regions 3381–3713, 3816–3916, 4042 onward | Original bitemporal write, current/history recall, and supersession behavior. Prefer current `SQLiteRepository.write_fact` region 3227–3592, including changes `68eb88f7e14bafcc4ee7a13e7f76a394b64851d` and `f711a4d2532a87047bb80f8e7a445a323ac1d387`. |
| `67e653d...:src/sen/storage/sqlite/repository.py` | source rejection region 1327–1465 and helper region 5802–5893 | Reject-with-derived-state rollback semantics. Prefer the transactional Cockroach implementation in `b328172...:src/sen/storage/crdb/repository.py` (2745–2969), which locks affected state and writes audit data atomically. |
| `b328172...:src/sen/trust.py` | `source_is_trusted`, `sanitize_recall_for_context` (83–135) | Fail-closed read boundary and citation-preserving sanitization. The `looks_like_injection` regex is not an authorization or mutation signal and is rejected below. |
| `67e653d...:src/sen/policy.py` | `MemoryPolicyFact`, `MemoryPolicyInput`, `MemoryPolicyDecision`, `MemoryPolicy`, rules, `RuleBasedMemoryPolicy`, evaluation contracts (12–52, 116–126, 309–452) | Explicit `add/update/merge/delete/noop` result with reason and confidence; ordered deterministic policy rules and fixture-driven evaluation. Current flags in `b328172...:src/sen/policy.py` (92–99, 359–385), commit `ea3eaaa7bee61f4663e74f29852f8b5aac8badf8`, remain useful policy gates. |
| `b328172...:src/sen/api.py` | reconciliation taxonomy around 2071–2231 | Distinguish duplicate, coexist, and supersede. Port the taxonomy and conservative preconditions, not the large orchestration or model authority. |
| `b328172...:src/sen/storage/crdb/connection.py` | `is_retryable`, `is_ambiguous`, `run_transaction` (113–174) | Central whole-transaction retry for `40001` and a separate ambiguous-result path. Swarm Brain tightens `40003` handling around idempotency instead of replay. |
| `b328172...:src/sen/storage/base.py` and `src/sen/models.py` | `FleetCoordinationRepository` (62–91), `ActionClaim` (116–132), `ActionClaimResult` (133–152) | Behavioral seed for idempotent action ownership; replace with task-specific ports, authenticated agent ownership, lease fencing, and CockroachDB transactions. |
| `b328172...:src/sen/storage/crdb/repository.py` | action-claim regions 5051–5367 and 5433–5530 | Cockroach serializable claim/idempotency behavior reference. New task claims require dependency readiness, partial unique active lease, expiry/handoff, owner/version checks, and initial memory. |
| `b328172...:src/sen/mcp.py` | `_jsonable`, `_load_fastmcp`, `build_mcp_server`, `main` | Lazy optional MCP dependency, centralized registration, stdio entry point, and JSON-safe boundary. The actual Sen tools and session API are not donors. |

At `67e653d33fbc66680b9806ae3f3410fb061beea2`, the most relevant tests to
translate are `tests/test_phase12_kernel.py` (especially 11–55 and 182–285) and
`tests/test_recall_dedup.py` (94–117). At current
`b32817292738c4bf5be26c707f87dbee150810e0`, additional donors are
`tests/test_fact_timeline_guard.py` (25–176),
`tests/test_reconcile_writes.py` (50–201),
`tests/test_memory_policy_flags.py` (19–110), and
`tests/test_repository_contract.py` regions 136–182, 210–280, and 400–566.
Cockroach action/rejection behavior is covered by
`tests/test_crdb_action_claims.py` and `tests/test_crdb_repository_slices.py`.
Swarm Brain translates the invariant, not fixture-specific subject or
benchmark vocabulary.

## Accepted `mnemotree` donors

All paths in this section refer to
`df8895a28df9f6cebc352c46d1777d98539d55b7` unless another ref is named.

| Path | Exact symbols/regions | Adaptation |
| --- | --- | --- |
| `src/mnemotree/core/models.py` | `ObservationStatus`, `ObservationKind`, `compute_observation_confidence` (70–109); scope/observation fields inside `MemoryItem` (168–303) | Translate `attempt`, `result` → `outcome`, `decision`, `handoff`, `warning`, and `observation`. Make `hypothesis` a kind with tentative state, and add explicit `superseded` state. Do not copy `MemoryItem`. |
| `src/mnemotree/core/memory.py` | `RememberOptions` (233), `RecallFilters` (262); ingest orchestration 429–846; default scope and refuted exclusion 1418–1579 | Async application boundary and explicit scope/filter propagation. Replace dataclasses and raw dictionaries with focused Pydantic models, and push all scope/state filters into storage. |
| `src/mnemotree/store/protocols.py` | `MemoryCRUDStore`, `SupportsVectorSearch`, `SupportsMemoryListing`, `SupportsSummaries`, `SupportsLeases` | Capability-oriented async `Protocol` pattern. Swarm Brain ports return typed domain results and are split by transaction/use-case ownership. |
| `src/mnemotree/core/_internal/persistence.py` | `Persistence`, `DefaultPersistence` | Small persistence facade pattern only; no concrete donor store crosses the boundary. |
| `src/mnemotree/store/_schema.py` | scope/index migration 171–255; lease and summary shapes 274–333 | Conceptual field inventory. CockroachDB uses UUID/TIMESTAMPTZ/JSONB, active-lease uniqueness, fencing, idempotency, and outbox invariants instead. |
| `src/mnemotree/store/sqlite_vec_store.py` | summary and lease behavior 795–1028 | Behavioral test seed for claim/renew/release/list and summary upsert; no SQLite code is ported. |
| `src/mnemotree/mcp/server.py` | `agent_remember_observation`/`agent_recall_context` (508–619); summaries/claims/status (685–842); bundle/compaction (844–1003); `_register_tools`, lazy FastMCP, `main` (1781–1858) | Only the pattern “plain async function → application service/HTTP client → JSON-safe result” and centralized registration. Swarm Brain exposes exactly six model-visible tools. |
| `src/mnemotree/core/_internal/enrichment.py` | `EnrichmentPipeline`, `StandardEnrichmentPipeline` | Concurrent optional stages and explicit fallback for hybrid extraction. |
| `src/mnemotree/inference/router.py`, `extractor.py`, `local_analyzer.py` | `RouterInference`, `ExtractorInference`, `LocalModelAnalyzer` | Lazy model loading, two-stage routing/extraction, and off-event-loop local inference. Replace personal labels with Swarm Brain kinds and validate provenance-bearing Pydantic candidates. |

Tests worth translating are `tests/core/test_agent_scoping.py`,
`tests/core/test_agent_layer.py`, `tests/mcp/test_agent_tools.py`,
`tests/mcp/test_server.py`, and `tests/store/test_filters.py`. They are not
sufficient acceptance evidence because they do not cover authenticated
identity, distributed concurrent leases, idempotency, ownership/fencing, or
CockroachDB retries.

## Explicit rejections

### From `sen`

- Reject `src/sen/api.py` as a module. It is a large session/retrieval
  orchestrator; only named bitemporal/reconciliation semantics above are
  adapted.
- Reject both `src/sen/storage/sqlite/repository.py` and
  `src/sen/storage/crdb/repository.py` as classes. Their broad subject-memory
  interfaces and schema are not Swarm Brain task/lease repositories.
- Reject `src/sen/server.py`, `src/sen/app.py`, and their demo/session identity
  model. FastAPI factory/OpenAPI wiring may inform tests, not domain contracts.
- Reject benchmark-specific `packing/`, `source_events.py`, most `query.py`,
  LongMemEval adapters, personal ontology/profile/prospective-memory concepts,
  and provider-specific reconciliation heuristics from the v0 core.
- Reject embedding similarity or LLM output as automatic mutation authority.
  Reject hard delete of source-derived history.
- Reject `trust.looks_like_injection` as a security decision. It may contribute
  a warning signal, never scope access, confirmation, refutation, or
  supersession.

### From `mnemotree`

- Reject `MemoryItem` wholesale: it mixes personal memory, vectors, graph,
  temporal state, emotions, decay, access tracking, provenance, and agent
  coordination.
- Reject the 3,362-line `MemoryCore` and 1,862-line `mcp/server.py` monoliths.
- Exclude `user_id`, `conversation_id`, autobiographical/episodic/conditioning
  taxonomy, emotion/valence/arousal, user-profile reinforcement/decay, and
  conversation-observer/consolidation paths.
- Reject post-retrieval scope filtering in `core/memory.py` (1546–1555). It is
  neither an authorization boundary nor recall-correct.
- Reject MCP identity/scope arguments, mutation-by-ID without ownership checks,
  and lease renew/release by `lease_id` without owner, capability, version, and
  expiry checks.
- Reject SQLite process-local locking/check-then-upsert for distributed claims;
  process-local observation counters; fire-and-forget compaction that loses
  errors; and metadata-only confidence wiring.
- Reject duplicate embedding/LLM protocols. Swarm Brain has one canonical port
  per capability.

## Built from scratch

No donor has the required security and transactional semantics, so Swarm Brain
owns new implementations of:

- `ActorContext`, agent tokens, IAM mapping, and auth-derived identity;
- runs/swarms, task DAG eligibility, task leases, renewals, fencing, checkpoints,
  completion/release, and crash handoff;
- mutation idempotency records and `40001` versus `40003` handling;
- evidence-backed conflict reporting/resolution and poisoning guards;
- CockroachDB schema and retry kernel now, followed by P1 repositories,
  transactional outbox, action attempts, and swarm event ledger;
- canonical HTTP routes and the exact six-tool stdio MCP surface.

The current target schema and transaction design are documented in
[CockroachDB schema](cockroach-schema.md); public and application contracts are
in [API contracts](api.md).
