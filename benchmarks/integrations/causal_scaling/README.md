# 1/2/4-agent causal-scaling adapter

This integration is the evidence path for the `multi_agent_causal_scaling`
SOTA gate. It does not reuse the demo's simulated A/B baseline. Every task and
seed instead runs six isolated measured cells: 1, 2, and 4 agents, each with
memory enabled and disabled.

The rollout-level model-token and tool-call caps are identical in all six
cells. `split_budget` divides each cap across agent slots; adding agents never
multiplies it. At least five fixed seeds randomize execution order. Executors,
hidden task scorers, and public-runtime event readers are injected through the
protocols in `contracts.py`, so no model vendor is built into the evaluator.

Memory use is not inferred from recall. The compiler independently replays the
complete public `GET /v1/runs/{run_id}/events` projection and requires an exact
task/lease/consumer/memory intersection between `memory.activated` and a
durable checkpoint or completion citation. A no-memory cell must prove that
both are absent. Every cell has a unique run ID, raw output, per-agent usage,
typed failures, immutable provenance, and an unaggregated outcome row.

`memory_dependent_gain` is the paired difference-in-differences

```text
(memory_N - no_memory_N) - (memory_M - no_memory_M)
```

for the same task and seed. Its interval is a deterministic task-cluster
bootstrap. The raw memory-arm team-success difference is reported only as a
diagnostic and cannot establish the causal memory claim.

## Workload pin required

No reviewed causal workload is bundled yet. This is an intentional claim
blocker: arbitrary toy tasks must not satisfy the SOTA gate. Before any
canonical run, review and commit a separate `swarmbrain-causal-workload-v1`
manifest containing a stable workload ID/revision, source/revision,
verifier-schema revision, review revision, and per-task digests for the public
prompt, hidden verifier, and complete task. Then replace
`PIN_REVIEWED_WORKLOAD_SHA256_BEFORE_RUNNING` in the SOTA manifest with that
exact reviewed manifest digest.

Once real provider evidence and the pinned workload exist, compile it with:

```bash
uv run --extra dev python scripts/build_multi_agent_causal_report.py \
  --evidence /path/to/raw-causal-run.json \
  --workload-manifest /path/to/reviewed-workload.json
```

The compiler rejects smoke/fake adapters, draft workloads, partial arms,
missing counterfactuals, changed tasks, duplicate run IDs, unequal or exceeded
budgets, model/provenance drift, typed rollout failures, incomplete event
streams, and activation without matching citation. It does not claim official
MemoryArena comparability; that would require using and pinning that benchmark's
tasks and protocol explicitly.
