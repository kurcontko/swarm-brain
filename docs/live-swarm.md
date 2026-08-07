# Live swarm: real coding agents on the production MCP bridge

This runbook drives real LLM coding agents — not scripted clients — through a
Swarm Brain run backed by CockroachDB. Each harness is an ordinary coding CLI
(`claude`, `codex`, `gemini`, `opencode`) launched headless with the production
stdio bridge `swarmbrain-mcp` configured as an MCP server. The model chooses
when to call `claim_task`, `recall_memory`, `publish_memory`, `checkpoint_task`,
and `complete_task`; the bridge holds the bearer token, renews the lease in the
background, and every mutation lands in the durable event stream.

Nothing here is simulated. The scripted demo (`swarmbrain-demo`) asserts the
coordination beats; this runbook proves a real model can drive them.

Related: [restart demo](restart-demo.md) · [architecture](architecture.md) ·
[API](api.md)

## What is real, and what is not

| Piece | Real? |
| --- | --- |
| The agent | A real LLM CLI process, its own model, its own decisions |
| The transport | The production `swarmbrain-mcp` stdio bridge, one process per agent |
| Auth | A per-agent signed bearer token; scope comes only from the token |
| Storage | CockroachDB; tasks, memory, evidence, leases, and events are durable |
| The tasks | Seeded by the operator; small reasoning tasks answerable without a checkout |
| Prior memory | Seeded by an operator "scout" identity before the agents start |

The tasks and the prior memory are fixture content, exactly as in any demo.
Everything the agent does with them is not.

## Prerequisites

- CockroachDB reachable, with the Swarm Brain schema installed (see the
  [README](../README.md) CockroachDB section).
- `uv` on `PATH`, and at least one coding CLI installed and already logged in.
- A token secret shared by the API and the launcher.

## Terminal 1 — the API

```bash
export SWARMBRAIN_BACKEND=cockroach
export SWARMBRAIN_DATABASE_URL='postgresql://root@127.0.0.1:26257/swarmbrain_demo?sslmode=disable'
export SWARMBRAIN_TOKEN_SECRET='local-demo-secret-0123456789'
export SWARMBRAIN_PORT=8099

uv run --extra serve --extra crdb swarmbrain-api
```

Wait for `{"status":"ready"}` from `curl --fail http://127.0.0.1:8099/readyz`.

## Terminal 2 — seed the run

Same environment as terminal 1 (the launcher seeds tasks through the operator
plane, so it needs the database URL as well as the token secret):

```bash
uv run --extra crdb --extra mcp python scripts/live_swarm.py seed \
  --api-url http://127.0.0.1:8099 \
  --harness claude --harness codex
```

This creates a fresh run (new tenant/project/repository/swarm/run UUIDs), seeds
three self-contained invariant tasks, publishes one prior finding per task as a
scout so `recall_memory` returns something real, issues one short-lived token
per harness plus a read-only viewer token, and writes a launch bundle to a fresh
temp directory (override with `--out`, which must be outside the repository
because it holds bearer tokens; files are written `0600`).

It then prints the exact launch command for every requested harness. Useful
flags: `--lease-seconds`, `--ttl-seconds`, `--model claude=opus`,
`--no-seed-memory`.

## Terminal 3 — launch a harness

Copy the printed command. For Claude Code it looks like this (`$BUNDLE` is the
bundle directory the seed step printed):

```bash
claude -p "$(cat $BUNDLE/claude.prompt.md)" \
  --mcp-config $BUNDLE/claude.mcp.json --strict-mcp-config \
  --allowedTools "mcp__swarmbrain" \
  --model sonnet --max-budget-usd 2 \
  --output-format stream-json --verbose | tee $BUNDLE/claude.transcript.jsonl
```

`--strict-mcp-config` guarantees the session sees no MCP server other than
Swarm Brain, and `--allowedTools "mcp__swarmbrain"` allows that server's tools
without a permission prompt. Swap `--output-format stream-json --verbose` for
`--output-format text` if you want a clean human-readable terminal for the
camera; the JSONL form is what the evidence collector reads.

The session claims a task, recalls prior memory, reasons out the invariant,
publishes it with its own reasoning attached as evidence, checkpoints, and
completes — typically seven turns and about thirty seconds.

## What to point the camera at

Open `http://127.0.0.1:8099/console`, paste the run id and the viewer token:

```bash
source $BUNDLE/viewer.env
echo $SWARMBRAIN_RUN_ID
echo $SWARMBRAIN_VIEWER_TOKEN
```

The console polls the same authenticated read routes an operator would use and
draws task custody, run counters, the durable event ledger, and memory lineage.
Frame it beside the harness terminal: as the model calls each tool, a row
appears in the ledger. That side-by-side is the shot — a real coding agent on
the left, durable swarm state on the right.

A terminal-only alternative:

```bash
source $BUNDLE/viewer.env
curl -s -H "Authorization: Bearer $SWARMBRAIN_VIEWER_TOKEN" \
  "$SWARMBRAIN_API_URL/v1/runs/$SWARMBRAIN_RUN_ID/events?limit=200" \
  | jq -r '.events[] | "\(.occurred_at) \(.event_type)"'
```

## The cross-vendor beat

Seed with several `--harness` flags and launch each printed command in its own
terminal against the same run. Three tasks are seeded, so up to three agents can
each claim their own; the claim path is fenced, so two agents never get the same
task, and a memory one vendor publishes is recallable by the others.

The launcher writes a config in each harness's own format, always isolated from
the operator's global configuration:

| Harness | Config the launcher writes | How it is used |
| --- | --- | --- |
| `claude` | `claude.mcp.json` (`mcpServers`) | `--mcp-config` |
| `codex` | `codex-home/config.toml` (`[mcp_servers.swarmbrain]`) | `CODEX_HOME=...` plus `-C codex-workspace` |
| `gemini` | `gemini-workspace/.gemini/settings.json` | run `gemini` from that directory |
| `opencode` | `opencode-workspace/opencode.json` | run `opencode` from that directory |

Every harness also gets a generic `<harness>.mcp.json` and a sourceable
`<harness>.env`, so an unlisted CLI can be wired up by hand.

For codex the launcher also symlinks `~/.codex/auth.json` into the private home
so the isolated configuration still authenticates; no credential is copied.

## Known harness gaps

Observed on 2026-08-07 against API `0.1.0` and schema v9. These are harness
issues, not Swarm Brain issues; the bridge behaved identically under all of
them.

- **codex-cli.** The bridge starts, authenticates, and `agent.joined` lands in
  the event stream, but every MCP tool call in a non-interactive `codex exec`
  session is answered with `user cancelled MCP tool call`. Setting
  `approval_policy = "never"`, marking the project trusted, and closing stdin do
  not change it. The printed default is therefore an **interactive** session in
  which the operator approves the swarmbrain tool calls by hand. codex's
  documented automation escape hatch,
  `--dangerously-bypass-approvals-and-sandbox`, disables the harness sandbox
  and every approval gate; the launcher prints it only when seeding with
  `--unattended`, behind an explicit warning, and it was **not** exercised on
  the reference machine. Also note codex blocks forever on stdin when stdin is
  neither a TTY nor closed, hence the `< /dev/null` in the unattended variant.
- **Authentication is per-harness.** `gemini` and `opencode` were installed but
  not logged in on the reference machine, so they were configured but not run.
  Log each CLI in once before filming.
- **TOML scoping.** If you hand-edit the generated codex config, keep
  `approval_policy` and `sandbox_mode` above the first `[table]` header;
  otherwise TOML scopes them under `[mcp_servers.swarmbrain]` and codex rejects
  the file.

No product gap was found in the MCP tool schemas: the Claude Code session used
`claim_task`, `recall_memory`, `publish_memory`, `checkpoint_task`, and
`complete_task` correctly on the first attempt, with no retries and no
malformed arguments. `ingest_memory_source` and `report_conflict` are the two
bridge tools this prompt does not exercise.

## Collect evidence

After the harnesses finish:

```bash
uv run --extra crdb python scripts/live_swarm.py report --bundle $BUNDLE \
  --harness-result claude=$BUNDLE/claude.transcript.jsonl \
  --transcript codex=$BUNDLE/codex.transcript.txt
```

This reads the run's events and metrics with the viewer token and writes a
redacted JSON artifact to `evidence/`: the run id, each agent's harness,
provider, and token model label, the models the harness itself reported, the
tool-call order, the durable event sequence per task, and run metrics. Bearer
tokens, absolute home paths, and the bundle path are stripped. Add
`--gap harness="..."` to record honestly why a harness did not finish.

A completed reference artifact is
[`evidence/20260807T132559Z-live-swarm-real-agents.json`](../evidence/20260807T132559Z-live-swarm-real-agents.json):
one Claude Code session completing the full loop, one codex session that joined
but was blocked by its own approval gate.

## Cleanup

- Stop each harness (they exit on their own) and the API (`Ctrl-C`).
- Delete the bundle directory: `rm -rf $BUNDLE`. It contains live bearer tokens.
- Leave the database alone. Every seed creates a fresh run; nothing is
  overwritten, and the run stays queryable for as long as you want the footage
  to be reproducible.
