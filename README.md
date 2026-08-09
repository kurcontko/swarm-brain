# Swarm Brain

Swarm Brain is a vendor-neutral coordination and temporal-memory kernel for a
swarm of heterogeneous coding agents. This repository is a self-contained Python
project; it does not import either donor project, `sen` or `mnemotree`, at runtime.
Instead, it adapts their narrow, audited contracts and semantics behind new
ports and CockroachDB-oriented transaction boundaries.

> Swarm Brain adapts memory semantics and API patterns from our pre-existing
> open-source projects **Sen** and **mnemotree** (disclosed; neither is a
> runtime dependency, neither is imported, and neither co-owns the schema).
> All Swarm Brain code — the domain and application layers, the CockroachDB
> backend and schema, the retrieval engine, the HTTP API, the MCP bridge, the
> demo application, and the AWS integration — was written during the submission
> period, June 30 – August 18, 2026.

The project was extracted from the Sen monorepo with its Swarm Brain commit
history intact. See the [extraction map](docs/history/sen-extraction-map.md) for
the original and rewritten commit IDs.

See the [current retrieval status](docs/retrieval-status.md), the
[standalone architecture](docs/retrieval-architecture.md), the full
[PostgreSQL/CockroachDB SOTA research dump](docs/research/sota-retrieval-postgresql-cockroachdb-2026-08-02.md),
and the [agent memory & retrieval SOTA research dump](docs/research/sota-agent-memory-retrieval-2026-08-07.md).

## What it includes

- authenticated tenant/project/repository/swarm/run/agent context;
- transactional task claim, lease renewal, checkpoint, completion, and release;
- idempotent mutation handling;
- task/run/repository-scoped temporal memories with evidence and lineage;
- append-by-default memory observations, explicit supersession, and poisoning guards;
- lossless JSON memory documents with open application-defined semantic labels;
- purpose-aware exact, FTS `simple`, trigram, versioned dense, bounded graph
  expansion, and weighted-RRF retrieval;
- private canonical hydration with scope/state/trust/bitemporal revalidation;
- evidence-backed conflict reporting and resolution;
- an in-memory adapter for deterministic local development and tests;
- an explicit CockroachDB schema command and a pooled async composition seam;
- durable, fenced extraction/embedding/artifact work with deterministic fallback;
- the canonical FastAPI surface and a seven-tool stdio MCP bridge.

Backend selection is fail-closed. `SWARMBRAIN_BACKEND` must be either `memory`
or `cockroach`; there is no implicit fallback. The memory backend rejects a
database URL, while the CockroachDB backend requires one. CockroachDB imports
remain lazy, so a memory-only install does not require the optional driver.

API startup opens the selected backend and verifies its prerequisites. It never
installs or changes database schema. Operators must run the schema command
explicitly before starting a CockroachDB-backed API.

## Flexible memory contract

Memory content is a required, non-null JSON value: existing strings remain
fully compatible, while objects, arrays, numbers, and booleans round-trip
without being flattened. `MemoryKind`, `EvidenceKind`, and `MemoryLinkKind`
list useful built-ins but do not form a closed ontology; applications may use
their own labels such as `org.acme/preference` or `application/pdf`.

Independent publications append by default, even when kind and content are
identical. Idempotency still makes retries of the same command exactly-once,
while an explicit `supersedes_memory_id` remains the only path that replaces a
current assertion. The database stores a deterministic text projection for
search and the original structured value in `content_json`.

Extraction candidates may cite zero or more exact source spans. A supplied
span is still checked character-for-character against the preserved source;
zero spans means a source-derived synthesis, not a fabricated quotation. Scope,
visibility, lifecycle states, trust/review state, auth, lease fencing, and
idempotency remain strict.

## Scripted swarm demo

One command drives the full swarm story over the canonical HTTP API — twelve
simulated heterogeneous workers (Claude Code / Codex / Gemini / Qwen roster
labels) racing four ready tasks, cross-vendor evidence-backed recall, an
evidence-backed supersession with a rejected poisoning attempt, an idempotent
duplicate completion, a crash handoff across vendors with stale-lease fencing,
a dependency-blocked task unblocking, and the run's durable events and metrics:

```bash
uv run --extra serve swarmbrain-demo
```

The default run composes an in-process in-memory backend with a controllable
clock, so lease expiry is demonstrated without waiting. With the CockroachDB
environment set (see below), the same beats run against the durable backend
and lease expiry elapses in real time:

```bash
uv run --extra serve --extra crdb swarmbrain-demo
```

Every beat appends named checks to a JSON evidence artifact under
`evidence/`. The simulated workers speak exactly the protocol a real harness
speaks through the MCP bridge; the roster labels record the vendor mix the
scenario stands in for.

## Development

Run the test and lint suites from this directory:

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev ruff check src tests
```

Start the development API with an in-memory kernel:

```bash
export SWARMBRAIN_BACKEND=memory
export SWARMBRAIN_TOKEN_SECRET=local-development-secret
uv run --extra serve swarmbrain-api
```

The probes have deliberately different meanings:

```bash
curl --fail http://127.0.0.1:8080/healthz
curl --fail http://127.0.0.1:8080/readyz
```

`/healthz` reports process liveness. `/readyz` checks the selected backend and
returns HTTP 503 with a bounded `{"status":"not_ready"}` body when it is not
usable; backend exception text and connection strings are not returned.

Copy [`.env.example`](.env.example) for the complete environment-variable map.

## CockroachDB backend

Install or verify schema as an explicit operator action:

> Upgrading from a pre-v8 deployment requires a writer barrier: stop every old
> API and worker that can publish memory or embeddings, run `schema install` and `schema
> verify`, then start only current-version processes. Do not run the rebuild
> concurrently with pre-v8 writers, because those writers do not maintain every
> v8 retrieval projection. A large existing memory set also makes `install` an
> `O(N)` maintenance operation; rehearse and budget its transaction time first.
> The current schema version is v9; the v8→v9 step is additive (the durable
> `retrieval_reuse_counters` table) and needs only `schema install` + `verify`,
> with no projection rebuild.

```bash
export SWARMBRAIN_BACKEND=cockroach
export SWARMBRAIN_DATABASE_URL='postgresql://root@127.0.0.1:26257/swarmbrain?sslmode=disable'
export SWARMBRAIN_TOKEN_SECRET=local-development-secret

uv run --extra crdb swarmbrain-schema install
uv run --extra crdb swarmbrain-schema verify
uv run --extra serve --extra crdb swarmbrain-api
# In a separate process, using the same database and embedding configuration:
uv run --extra crdb swarmbrain-worker
```

The durable composition creates one `CockroachDatabase` pool shared by the
coordination and memory repositories. Pool bounds are controlled with
`SWARMBRAIN_DATABASE_POOL_MIN_SIZE` and
`SWARMBRAIN_DATABASE_POOL_MAX_SIZE`.

When embeddings are enabled, dense v2 writes a separate current-memory
`retrieval_vectors_1024` projection carrying canonical resource version,
content digest, scope key, renderer/model signature, and domain lane. Query
embeddings are generated before the read snapshot; CockroachDB then runs one
fully prefix-bound ANN branch per allowed repository/run/task scope. In the
same snapshot it validates canonical lifecycle/trust/time, version, and digest,
adaptively widens an under-filled ANN window, and then fuses dense ranks with
exact/FTS/trigram ranks. `SWARMBRAIN_RETRIEVAL_DENSE_MIN_SIMILARITY`
controls the optional raw cosine floor (disabled by default until calibrated),
while
`SWARMBRAIN_RETRIEVAL_DENSE_ANN_BEAM_SIZE` controls the per-query CockroachDB
beam. Tune both only against the exact-vector oracle and a representative saved
retrieval run; see [retrieval evaluation](docs/retrieval-evaluation.md).

Use the [local restart demo](docs/restart-demo.md) to test API and
database process restarts against a persistent local store. The guide defines a
manual acceptance gate; its presence is not a claim that the gate has passed.

## Local run tokens and MCP

Issue a short-lived local agent token:

```bash
uv run swarmbrain-token \
  --tenant 11111111-1111-1111-1111-111111111111 \
  --project 22222222-2222-2222-2222-222222222222 \
  --repository 33333333-3333-3333-3333-333333333333 \
  --swarm 44444444-4444-4444-4444-444444444444 \
  --run 55555555-5555-5555-5555-555555555555 \
  --agent 66666666-6666-6666-6666-666666666666 \
  --capability run:join --capability task:claim --capability task:checkpoint \
  --capability task:complete --capability lease:renew \
  --capability memory:publish --capability memory:recall \
  --capability source:ingest \
  --capability conflict:report
```

Run the local MCP bridge after setting `SWARMBRAIN_API_URL`,
`SWARMBRAIN_AGENT_TOKEN`, `SWARMBRAIN_RUN_ID`, and `SWARMBRAIN_AGENT_ID`:

```bash
uv run --extra mcp swarmbrain-mcp
```

This closed stdio MVP uses locally issued signed bearer tokens, not OAuth. Do
not commit token secrets, database credentials, or issued agent tokens.

Token rotation is deliberately simple and fail-closed: stop the API, replace
`SWARMBRAIN_TOKEN_SECRET`, restart the API, issue fresh short-lived tokens, and
restart each MCP bridge with its new token. Tokens signed with the previous
secret fail immediately after the API restart. The v0 stdio path has no
dual-secret grace window or per-token online revocation check, so checkpoint
leased work before a planned rotation; an interrupted worker is recovered by
normal lease expiry and checkpoint handoff.

## Console

The API serves a read-only swarm console at `GET /console`: a single
self-contained page (no bundler, no CDN, no data baked in) that polls the
canonical read routes every two seconds and draws task custody, crash handoffs,
run counters, the durable event ledger, and memory lineage.

The page is public; the data is not. Paste the run id and a viewer token into
its connection dialog — the token stays in that tab's session storage. Issue one
with the same `swarmbrain-token` command as above, keeping the grant to
`--capability events:read --capability metrics:read --capability memory:recall`
and matching `--run` to the run you want to watch.

## License

MIT — see [LICENSE](LICENSE). Security policy: [SECURITY.md](SECURITY.md).
