# Retrieval benchmark artifacts

Saved runs and metric reports produced by `scripts/run_retrieval_eval.py`. The
written analysis lives in `docs/retrieval-benchmark.md`; these files are the
raw evidence behind it.

| File | Contents |
| --- | --- |
| `swarm-native-memory-run.json` | Per-lane rankings for the 40 judged queries on the in-memory kernel |
| `swarm-native-memory-report.json` | Metrics, per-intent slices, latency and the `min_score` sweep for that run |
| `swarm-native-cockroach-run.json` | The same 40 queries on live CockroachDB |
| `swarm-native-cockroach-report.json` | The same metrics plus ANN Recall@k against the exact-vector oracle |
| `swarm-native-memory-nodense-run.json` | Lane ablation: identical queries with the dense lane disabled |
| `swarm-native-memory-nodense-report.json` | Metrics for that ablation |
| `longmemeval-s-memory-run.json` | Per-lane rankings for all 500 LongMemEval-S questions |
| `longmemeval-s-memory-report.json` | Metrics, per-question-type and abstention slices |
| `longmemeval-s-memory-nodense-*.json` | The same LongMemEval-S run with the dense lane disabled |

Runs use the saved-run format defined in `docs/retrieval-evaluation.md` and can
be rescored directly:

```bash
uv run --extra dev python scripts/evaluate_retrieval_runs.py \
  benchmarks/retrieval/swarm-native-cockroach-run.json --k 10
```

Reruns overwrite these files in place, so a re-measurement shows up as a
reviewable diff. Identifiers in the rankings are corpus keys (track 1) or
`<position>:<session_id>` (track 2), never production UUIDs.

The 278 MB LongMemEval-S release is not stored here. The runner caches it in
`~/.cache/swarmbrain-eval/` (override with `SWARMBRAIN_EVAL_DATA_DIR`) and
verifies its pinned SHA-256 on every run.
