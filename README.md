# Swarm Brain v0

Swarm Brain is a vendor-neutral coordination and temporal-memory kernel for a
swarm of heterogeneous coding agents. This directory is a self-contained Python
project; it does not import either parent `sen` or sibling `mnemotree` at runtime.
Instead, it adapts their narrow, audited contracts and semantics behind new
ports and CockroachDB-oriented transaction boundaries.

## What v0 includes

- authenticated tenant/project/repository/swarm/run/agent context;
- transactional task claim, lease renewal, checkpoint, completion, and release;
- idempotent mutation handling;
- task/run/repository-scoped temporal memories with evidence and lineage;
- append-only supersession, conservative deduplication, and poisoning guards;
- evidence-backed conflict reporting and resolution;
- an in-memory adapter for deterministic local development and tests;
- CockroachDB DDL and a centrally tested serialization-retry kernel;
- the canonical FastAPI surface and a six-tool stdio MCP bridge.

The executable P0 backend is intentionally in-memory. The durable CockroachDB
repositories are the next milestone; DDL presence is not a durability claim.

## Development

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev python -m compileall src tests
```

Start the development API with an in-memory kernel:

```bash
export SWARMBRAIN_TOKEN_SECRET=local-development-secret
uv run --extra serve swarmbrain-api
```

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

Run the local MCP bridge after setting `SWARMBRAIN_API_URL` and
`SWARMBRAIN_AGENT_TOKEN`:

```bash
uv run --extra mcp swarmbrain-mcp
```

The CockroachDB schema is an explicit operator-applied resource; the API never
mutates schema at startup. `SWARMBRAIN_DATABASE_URL` is rejected by the P0 CLI
until the P1 durable repository is composed, preventing accidental non-durable
operation under a misleading database configuration.
