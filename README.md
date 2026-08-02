# Swarm Brain

Swarm Brain is a vendor-neutral coordination and temporal-memory kernel for a
swarm of heterogeneous coding agents. This directory is a self-contained Python
project; it does not import either parent `sen` or sibling `mnemotree` at runtime.
Instead, it adapts their narrow, audited contracts and semantics behind new
ports and CockroachDB-oriented transaction boundaries.

## What it includes

- authenticated tenant/project/repository/swarm/run/agent context;
- transactional task claim, lease renewal, checkpoint, completion, and release;
- idempotent mutation handling;
- task/run/repository-scoped temporal memories with evidence and lineage;
- append-only supersession, conservative deduplication, and poisoning guards;
- evidence-backed conflict reporting and resolution;
- an in-memory adapter for deterministic local development and tests;
- an explicit CockroachDB schema command and a pooled async composition seam;
- the canonical FastAPI surface and a six-tool stdio MCP bridge;
- cheap evidence registration (`POST /v1/evidence/sources`, `POST /v1/evidence`),
  with the bridge registering inline evidence material before publishing;
- an optional semantic memory plane: publishes enqueue durable `embed_memory`
  work, the `swarmbrain-worker` process embeds outside any database
  transaction (deterministic local provider or Amazon Bedrock via
  `swarmbrain[aws]`), vectors land in a CockroachDB `VECTOR(1024)` column
  behind a tenant/repository/model prefix-scoped `VECTOR INDEX`, and recall
  blends ANN matches into lexical hits without bypassing visibility filters.

Backend selection is fail-closed. `SWARMBRAIN_BACKEND` must be either `memory`
or `cockroach`; there is no implicit fallback. The memory backend rejects a
database URL, while the CockroachDB backend requires one. CockroachDB imports
remain lazy, so a memory-only install does not require the optional driver.

API startup opens the selected backend and verifies its prerequisites. It never
installs or changes database schema. Operators must run the schema command
explicitly before starting a CockroachDB-backed API.

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

```bash
export SWARMBRAIN_BACKEND=cockroach
export SWARMBRAIN_DATABASE_URL='postgresql://root@127.0.0.1:26257/swarmbrain?sslmode=disable'
export SWARMBRAIN_TOKEN_SECRET=local-development-secret

uv run --extra crdb swarmbrain-schema install
uv run --extra crdb swarmbrain-schema verify
uv run --extra serve --extra crdb swarmbrain-api
```

The durable composition creates one `CockroachDatabase` pool shared by the
coordination and memory repositories. Pool bounds are controlled with
`SWARMBRAIN_DATABASE_POOL_MIN_SIZE` and
`SWARMBRAIN_DATABASE_POOL_MAX_SIZE`.

Use the [local restart demo](../docs/swarm-brain/restart-demo.md) to test API and
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
