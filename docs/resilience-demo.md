# Node-kill resilience rehearsal

Status: rehearsed against CockroachDB v26.2.1 on 2026-08-07 (previous scenario)
and again on 2026-08-17 UTC against the **current** eight-beat scenario — three
local nodes, one node SIGKILLed mid-run. Evidence goes to
[`evidence/`](../evidence/) with the suffix `-node-kill-resilience.json`.

The 2026-08-17 current-scenario rehearsal passed the harsher `beat` trigger on
its first and only run: node 3 killed after beat 2 during live write traffic,
all six remaining beats green, zero unavailable ranges, both survivors
identical
([`20260817T231337Z`](../evidence/20260817T231337Z-node-kill-resilience.json)).

2026-08-07 tallies (previous scenario):

| Trigger | Runs | Passed |
| --- | --- | --- |
| `lease-wait` — **default** — kill while the demo client is idle | 3 | 3 |
| `beat` — the harsher probe — kill while write traffic continues | 7 | 3 |

Every aborted `beat` run failed the same single way, on the same single request,
documented below. It is a gap in the demo client, not in the kernel, and **nothing
was lost when it fired** — the evidence from a failed run shows every row committed
before the abort intact and identical on both surviving nodes.

`lease-wait` is the default because it is the take an operator can shoot. `beat` is
one flag away, is the more revealing test, and is why the flake below is documented
instead of hidden.

This is the operator script behind shot **C3** of the demo video script. One
command drives the whole beat:

```bash
uv run scripts/resilience_demo.py
```

## What the script does

[`scripts/resilience_demo.py`](../scripts/resilience_demo.py) is self-contained
and idempotent — it owns every process it starts and tears all of them down in a
`finally` block, including on failure.

1. **Starts a fresh three-node insecure cluster.** SQL/RPC on `localhost:26260-26262`,
   HTTP on `localhost:8290-8292` — deliberately clear of the conventional
   `26257`/`8080`, so an already-running local CockroachDB is never touched. Stores
   and logs go to a fresh `tempfile` directory (override with `RESILIENCE_STORE_DIR`
   or `--data-dir`); the script refuses any store path inside the repository.
2. **Waits for real replication.** `cockroach init`, then a bounded wait until all
   three nodes are live and `SHOW CLUSTER RANGES` reports zero ranges with fewer
   than three replicas. Without this wait the kill would prove nothing: a range
   whose only replica sat on the doomed node has no quorum to survive with.
3. **Installs the schema.** `CREATE DATABASE swarmbrain_resilience`, then
   `swarmbrain-schema install` followed by `swarmbrain-schema verify`, both as
   subprocesses with the cluster's connection string. No DDL is run implicitly.
4. **Runs the scripted swarm demo against node 1** —
   `uv run --extra serve --extra crdb swarmbrain-demo --lease-seconds 20` — with its
   stdout streamed line by line to the console and mirrored into the evidence.
5. **Kills a node at a deterministic point.** The script reads the demo's own
   narration and fires `SIGKILL` at node 3's process on a chosen trigger
   (see [Kill triggers](#kill-triggers)).
6. **Asserts the outcome.** The demo must exit 0 with every beat green; the cluster
   must report exactly one dead node and two live ones with zero unavailable ranges;
   and the run's rows must still be readable.
7. **Re-reads the data state from both survivors.** Tasks, leases, checkpoints,
   completions, memories, swarm events, outbox events and agents for that run id,
   counted independently on node 1 and node 2, and required to agree. If the demo
   aborted without writing an artifact, the run id is recovered from the `runs`
   table so this evidence is collected anyway — an aborted client must not be
   allowed to hide the state of the database.
8. **Writes `evidence/<stamp>-node-kill-resilience.json`** — topology, which node was
   killed and how many seconds into the run, which beat it followed, which beats
   completed afterwards, the demo verdict and metrics, `cockroach node status`
   showing the dead node, the final counts, and an explicit `does_not_prove` list.
   Home-directory paths are redacted; no token or secret is written.

## What it proves

Three artifacts are kept in `evidence/`, one per outcome worth understanding:

| Artifact | Trigger | Outcome |
| --- | --- | --- |
| [`20260807T125659Z`](../evidence/20260807T125659Z-node-kill-resilience.json) | `lease-wait` | passed; the reference take |
| [`20260807T123433Z`](../evidence/20260807T123433Z-node-kill-resilience.json) | `beat` | passed; kill during live write traffic |
| [`20260807T123712Z`](../evidence/20260807T123712Z-node-kill-resilience.json) | `beat` | aborted on `ambiguous_commit`; data intact |

Taking the passing `beat`-trigger run, the harshest of the three:

- Node 3 was SIGKILLed **2.2 s into the demo**, immediately after the claim-race
  beat. **Six of eight beats ran entirely after the kill** and all six passed,
  including the crash-handoff beat that waits out a real 20-second lease and then
  hands the task to another vendor's agent.
- `swarmbrain-demo` exited 0. All eight beats green, all 25 checks green.
- `cockroach node status` afterwards: **two live nodes, one dead**, with
  `ranges_unavailable = 0` on both survivors. Every range kept a voting quorum.
- The run's committed state was identical read from node 1 and read from node 2:
  5/5 tasks completed, 6 leases (0 still active), 1 checkpoint, 5 completions,
  3 memory versions (2 current after the supersession), 33 swarm events, 33 outbox
  events, 14 agents. No task, no lease, no memory, no event was lost.

The survivors legitimately report a non-zero `ranges_underreplicated` after the
kill — the cluster knows it is one replica short and would start re-replicating
once `server.time_until_store_dead` (5 minutes by default) elapsed. Under-replicated
is not unavailable: every range still had two of three voters, which is quorum,
which is why the swarm never noticed.

## Kill triggers

`--kill-trigger lease-wait` (default)
: Fires when the crash-handoff beat announces its real-time wait for a lease to
  expire — a window in which the demo client has no transaction in flight. Beats 6,
  7 and 8 (the crash handoff, the DAG unblock and the full telemetry read) then run
  entirely against the degraded two-node quorum, including a successor agent
  claiming and completing the crashed worker's task. Deterministic, at the cost of
  not exercising the in-flight-transaction case. Recorded in
  [`evidence/20260807T125659Z-node-kill-resilience.json`](../evidence/20260807T125659Z-node-kill-resilience.json):
  killed after beat 5, three beats green afterwards, 5/5 tasks and 33 events.

`--kill-trigger beat`
: Fires between two beats, with the next beat's write traffic starting immediately.
  The strongest claim — the cluster loses a node while the swarm is actively writing
  — and the one that hits the ambiguous-commit window below in roughly half of runs.
  A passing example is
  [`evidence/20260807T123433Z-node-kill-resilience.json`](../evidence/20260807T123433Z-node-kill-resilience.json):
  killed 2.2 s in, right after the claim race, six beats green afterwards.

## The known flake: `ambiguous_commit`

Four of the seven `beat`-trigger rehearsals aborted about two seconds after the
kill. All four failed on the identical request — the first mutation the demo issues
after the kill, at the head of the shared-discovery beat:

```
POST /v1/evidence/sources failed with 503:
{"error":{"code":"ambiguous_commit","retryable":true,
  "message":"the transaction outcome could not yet be resolved from its idempotency key"}}
```

The consistency is the tell: this is not random corruption, it is one specific
window. The kill lands at T+2.2 s; that first write goes out a moment later, while
the ranges the dying node held leases for are still electing new leaseholders.

This is the kernel behaving correctly and the demo client not holding up its end:

- When the killed node was the leaseholder for a range the transaction wrote,
  CockroachDB can return an ambiguous result — it cannot prove whether `COMMIT`
  took effect. `run_serializable` in the CockroachDB adapter refuses to replay the
  body (replaying could duplicate effects) and raises `AmbiguousTransactionResult`.
- The kernel then tries to settle the question by reading the operation's
  idempotency record — but only for about half a second (five polls with a rising
  backoff). Range-lease recovery after a node death takes longer than that, so the
  record is still `started` and the answer is honestly "not yet known".
- The API therefore returns **503 `ambiguous_commit` with `retryable: true`**. The
  contract is explicit: retry the same request with the same idempotency key, and
  the stored record will either replay the committed response or the operation will
  execute fresh.
- `swarmbrain-demo` does not implement that retry. `DemoRunner._expect` treats any
  non-2xx as a fatal assertion, so the beat fails and the demo exits 1.

Nothing is lost when this fires. The rehearsal script keeps going on purpose: it
recovers the run id from the database, records `demo.failure_mode =
"demo_client_did_not_retry_ambiguous_commit"`, and still reads the data state off
both survivors. The artifact from one such abort,
[`evidence/20260807T123712Z-node-kill-resilience.json`](../evidence/20260807T123712Z-node-kill-resilience.json),
shows exactly that: the demo stopped after two beats, and the two survivors both
report 14 agents, 5 tasks, 4 leases and 18 swarm events — every row that had been
committed before the abort, byte-identical from either node, on a cluster reporting
one dead node, `ranges_unavailable = 0` and all 153 ranges holding quorum. The
client gave up; the database did not lose anything.

Two ways to deal with it:

- **For the video**, shoot with the default `lease-wait` trigger, which never enters
  the window. Use `--kill-trigger beat` only if you are willing to retake.
- **For the product**, the demo client should retry once or twice on a `retryable`
  503 with the same idempotency key — which is exactly what the API contract asks
  of an agent harness. That is a change in `src/swarmbrain/demo/runner.py`
  (`_expect`), outside this script's scope, and it would make the `beat` trigger
  deterministic too. Until it lands, the `beat` trigger is a useful probe rather
  than a green gate.

## What it deliberately does not prove

- **Gateway failover.** The demo's connection pool is pointed at exactly one node's
  SQL port (node 1). Killing node 1 would break the client's own sockets, and no
  amount of Raft quorum fixes that. This rehearsal kills a **non-gateway** node and
  claims nothing more; the script refuses `--kill-node 1` outright. A production
  deployment puts a load balancer — or a multi-host connection string with
  client-side failover — in front of the SQL layer; that is the piece that turns
  quorum survival into client survival. The narration for shot C3 must not overstate
  this.
- **Multi-region or cross-AZ behaviour.** All three nodes are processes on one host.
- **Disk loss or corruption.** The node is SIGKILLed; its store is left intact.

## Running it for the video

```bash
# Terminal 1 (the one on camera), from the repository root:
uv run scripts/resilience_demo.py --pause-before-demo
```

`--pause-before-demo` sets up the cluster and schema, then waits for Enter — so the
slow, uninteresting minutes are done before the recording starts and the take is
just the twenty-five seconds that matter: beats scrolling, one `SIGKILL` line in the
middle, beats continuing, and the PASS block at the end.

For a second pane showing the cluster's own view, run this after the kill:

```bash
"${COCKROACH_BINARY:-cockroach}" node status --insecure --host=localhost:26260 --ranges --format=csv
```

Add `--keep-cluster` to leave the cluster up for exploration afterwards; without it
the nodes are stopped and the stores deleted.

The on-screen text in the video script says "Node 2 killed mid-run." Say **node 3**,
or change the caption: the script kills its own node 3, which CockroachDB happens to
number as node id 2 because ids are assigned in join order. The evidence records
both numbers, and the caption should match whichever the console shows.

## Expected runtime

About **2.5 minutes** end to end, dominated by two waits:

| Phase | Typical |
| --- | --- |
| Three nodes start, cluster init | ~5 s |
| Wait for full three-way replication | ~31 s |
| Schema install + verify (79 statements, replicated DDL) | ~90 s |
| Demo run including the kill and a real 20 s lease expiry | ~25 s |
| Post-kill liveness and count queries, teardown | ~5 s |

Only the fourth row belongs on camera.

## Troubleshooting

- **`cockroach binary not found`** — set `COCKROACH_BINARY` to its absolute path.
  The script otherwise checks `PATH`, then `~/.local/bin`, `/usr/local/bin`,
  `/opt/homebrew/bin`, and finally reads the image path of any cockroach process
  already running.
- **`these ports are already in use`** — a previous rehearsal is still up. Stop it,
  or pass `--sql-base-port` / `--http-base-port`.
- **`the kill never fired`** — the demo never reached the configured trigger; the
  error prints the demo's last lines.
- **The script depends on the demo's narration format** (`[ok] <title>` and the
  lease-wait line) and on its `--lease-seconds` / `--evidence-dir` flags. If
  `swarmbrain-demo` changes those, update the two regexes at the top of the script.
- Every wait is bounded and fails with a specific message; a failure still writes a
  partial evidence artifact and still tears the cluster down.
