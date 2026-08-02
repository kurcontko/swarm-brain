# Swarm Brain local restart gate

Status: procedure defined; no successful run is attested by this document.

This manual gate checks that a CockroachDB-backed Swarm Brain deployment keeps
memory and idempotent responses across both API and database process restarts.
It uses one local CockroachDB node for a lightweight persistence check. A
single-node run does not test replication, leaseholder movement, node failure,
or multi-region behavior.

The commands use insecure CockroachDB mode and fixed local credentials. They
are suitable only for local development.

## 1. Start a persistent local database

Run these commands from `swarm-brain/`. Keep the data directory for the whole
gate; choosing a different directory on restart invalidates the test.

```bash
export SWARMBRAIN_DEMO_ROOT="${TMPDIR:-/tmp}/swarmbrain-restart-demo"
mkdir -p "$SWARMBRAIN_DEMO_ROOT/data" "$SWARMBRAIN_DEMO_ROOT/logs"

cockroach start-single-node \
  --insecure \
  --listen-addr=127.0.0.1:26257 \
  --http-addr=127.0.0.1:8081 \
  --store="$SWARMBRAIN_DEMO_ROOT/data" \
  --log-dir="$SWARMBRAIN_DEMO_ROOT/logs" \
  --pid-file="$SWARMBRAIN_DEMO_ROOT/cockroach.pid" \
  --background

cockroach sql --insecure --host=127.0.0.1:26257 \
  -e 'CREATE DATABASE IF NOT EXISTS swarmbrain;'
```

Port 8081 is intentional: the Swarm Brain API uses 8080 by default.

Configure the API and install schema explicitly:

```bash
export SWARMBRAIN_BACKEND=cockroach
export SWARMBRAIN_DATABASE_URL='postgresql://root@127.0.0.1:26257/swarmbrain?sslmode=disable'
export SWARMBRAIN_DATABASE_POOL_MIN_SIZE=1
export SWARMBRAIN_DATABASE_POOL_MAX_SIZE=4
export SWARMBRAIN_TOKEN_SECRET=local-restart-demo-secret

uv run --extra crdb swarmbrain-schema install
uv run --extra crdb swarmbrain-schema verify
```

Schema installation is deliberately separate from API startup. A missing or
incompatible schema must make startup fail instead of triggering DDL.

## 2. Start the API and issue a token

In terminal A:

```bash
uv run --extra serve --extra crdb swarmbrain-api
```

In terminal B, wait for both probes. Liveness alone is not sufficient:

```bash
curl --fail http://127.0.0.1:8080/healthz
curl --fail http://127.0.0.1:8080/readyz
```

Issue a one-hour token with the capabilities used by this gate. Capture the
single output line in `SWARMBRAIN_AGENT_TOKEN` without placing it in a file:

```bash
export SWARMBRAIN_AGENT_TOKEN="$(uv run swarmbrain-token \
  --tenant 11111111-1111-1111-1111-111111111111 \
  --project 22222222-2222-2222-2222-222222222222 \
  --repository 33333333-3333-3333-3333-333333333333 \
  --swarm 44444444-4444-4444-4444-444444444444 \
  --run 55555555-5555-5555-5555-555555555555 \
  --agent 66666666-6666-6666-6666-666666666666 \
  --ttl-seconds 3600 \
  --capability memory:publish \
  --capability memory:recall \
  --capability events:read)"
```

For normal local use, prefer the default 30-minute lifetime. To rotate the
signing key, stop the API, replace `SWARMBRAIN_TOKEN_SECRET`, restart the API,
issue fresh tokens, and restart the MCP bridges with those tokens. Old tokens
then fail signature verification immediately. This closed stdio MVP has no
OAuth flow, dual-secret overlap, or database-backed per-token revocation check;
short lifetime plus signing-key rotation are the current invalidation tools.
Checkpoint active work before a planned rotation so normal lease expiry can
hand it to a successor if a bridge does not return.

## 3. Write and replay one idempotent mutation

Publish a sentinel memory and save the response outside the repository:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer $SWARMBRAIN_AGENT_TOKEN" \
  -H 'Idempotency-Key: restart-demo-memory-v1' \
  -H 'Content-Type: application/json' \
  -d '{"kind":"observation","content":"restart demo sentinel 2026-08-02","visibility":"run"}' \
  http://127.0.0.1:8080/v1/memories \
  -o "${TMPDIR:-/tmp}/swarmbrain-before-restart.json"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $SWARMBRAIN_AGENT_TOKEN" \
  -H 'Idempotency-Key: restart-demo-memory-v1' \
  -H 'Content-Type: application/json' \
  -d '{"kind":"observation","content":"restart demo sentinel 2026-08-02","visibility":"run"}' \
  http://127.0.0.1:8080/v1/memories \
  -o "${TMPDIR:-/tmp}/swarmbrain-replay-before-restart.json"
```

Before continuing, inspect both files. They must contain the same `memory_id`;
the second response must have `replayed: true`.

## 4. Restart both processes without replacing storage

Stop the foreground API in terminal A with Ctrl-C. Then stop and restart the
database from terminal B using the same data, log, and PID paths:

```bash
kill "$(cat "$SWARMBRAIN_DEMO_ROOT/cockroach.pid")"

cockroach start-single-node \
  --insecure \
  --listen-addr=127.0.0.1:26257 \
  --http-addr=127.0.0.1:8081 \
  --store="$SWARMBRAIN_DEMO_ROOT/data" \
  --log-dir="$SWARMBRAIN_DEMO_ROOT/logs" \
  --pid-file="$SWARMBRAIN_DEMO_ROOT/cockroach.pid" \
  --background

cockroach sql --insecure --host=127.0.0.1:26257 -e 'SELECT 1;'
uv run --extra crdb swarmbrain-schema verify
```

Start the API again in terminal A with the same environment and command. Do not
run schema installation during restart:

```bash
uv run --extra serve --extra crdb swarmbrain-api
```

Wait for `curl --fail http://127.0.0.1:8080/readyz` to succeed.

## 5. Verify durable state and durable idempotency

Recall the sentinel after restart:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer $SWARMBRAIN_AGENT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text":"restart demo sentinel","include_evidence":false}' \
  http://127.0.0.1:8080/v1/memories:recall \
  -o "${TMPDIR:-/tmp}/swarmbrain-recall-after-restart.json"
```

Repeat the exact publish request and key from step 3, writing its response to
`${TMPDIR:-/tmp}/swarmbrain-replay-after-restart.json`. The gate passes only if:

- `/readyz` returned HTTP 200 after both processes restarted;
- recall returned the original sentinel and the original `memory_id`;
- the repeated mutation returned HTTP 200, the original `memory_id`, and
  `replayed: true`;
- `swarmbrain-schema verify` passed after restart;
- API startup did not execute schema installation or any DDL.

Record command output and the three response artifacts when using this as
release evidence. Until those artifacts exist and are independently checked,
the restart durability gate remains unproven.
