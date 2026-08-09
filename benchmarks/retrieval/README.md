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
| `*-openai-*.json` | Both tracks re-measured with a served semantic embedder instead of the hash stand-in |

The `-openai` family is written separately so the deterministic files above stay
byte-reproducible; they are the frozen 2026-08-07 baseline the written analysis
quotes, and they are not regenerated when the retriever changes.

Three fields in every run and report carry the quality-per-token story:

- `final_relevance` (run) — the calibrated, rank-independent relevance behind
  each returned hit, in hit order. It is what `RecallQuery.min_score` gates on,
  so a saved run can be replayed at any floor without re-executing it.
- `bundle_by_floor` (report) — mean bundle size, precision, recall and
  no-answer behaviour at each floor. Recall@k asks whether the answer was
  found; this asks how much of what the caller was handed was worth reading.
- `final_tokens` (run) and `answer_in_context` (report) — estimated reader cost
  per hit, and whether the gold evidence survives a token budget rather than
  merely placing inside k. Token counts come from a documented chars/4 proxy
  (`swarmbrain.retrieval.packing`), not a tokenizer; they are comparative, not
  a cost figure.

Each report's `fused` lane is plain weighted RRF and `final` is the published
bundle. Relevance reranking ships disabled, so in every canonical run above the
two are the same ranking. The `longmemeval-s-rerank-experiment-*` files are the
one exception: they were produced with the stage enabled, and are kept as the
evidence behind the decision not to ship it (see the second 2026-08-09 addendum
in `docs/retrieval-benchmark.md`). Their `fused` lane is bit-identical to the
shipped configuration's `final` lane, which is what makes them a clean A/B.
**Do not quote the experiment files for headline numbers** — the canonical
LongMemEval Recall@10 is 0.976, in `longmemeval-s-memory-openai-report.json`.

## End-to-end QA artifacts (`longmemeval-s-qa-*`)

`scripts/run_longmemeval_qa.py` adds the reader and judge stages the files
above deliberately stop short of, and writes three files per run, tagged with
the reader model and, for sampled runs, the sample size:

| File | Contents |
| --- | --- |
| `longmemeval-s-qa-<reader>-hypotheses.jsonl` | Official hypothesis format — one `{"question_id", "hypothesis"}` object per line, exactly what LongMemEval's `src/evaluation/evaluate_qa.py` consumes |
| `longmemeval-s-qa-<reader>-run.json` | Per question: retrieved session keys, calibrated relevance, hypothesis, reader/judge token counts, dev-judge label and raw verdict |
| `longmemeval-s-qa-<reader>-report.json` | `dev_judge_accuracy` overall, per question type and per abstention slice, plus retrieval support, token totals and latency |

**Every accuracy in those files is a development-judge accuracy.** The judge
prompts are copied verbatim from `evaluate_qa.py`, but they are answered by
whatever `--judge-model` names rather than by the official
`gpt-4o-2024-08-06`, so the numbers are for iteration only. A publishable
LongMemEval score comes from running the official script over the saved
hypothesis file. That is why the hypothesis file is a first-class artifact:
it is the input the official judge needs, and it is generated once.

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
