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
- `final_tokens` (run) is the development-only chars/4 item estimate. A
  publishable LongMemEval run additionally carries `exact_context_packing`:
  provider-observed token counts and decision traces over the complete official
  reader prompt. It also carries `exact_context_material`: the public benchmark
  question/date and only the ranked top-10 session role/content records needed
  to reconstruct every observed prompt. `answer_in_context` uses those
  observations only when `context_token_accounting.exact_model_tokenizer` is
  true; otherwise it remains an explicitly non-publishable estimate.

Exact mode adds no tokenizer dependency. It uses a repository-local JSONL
tokenizer executable and tokenizer artifact whose paths, byte lengths and
operator-pinned SHA-256 digests are recorded in the run and rechecked by the
offline report compiler. Every response must echo the requested text digest,
model, revision and artifact digest and provide a unique provider request ID.
The compiler rebuilds every traced prompt from `exact_context_material`, checks
its digest, rejects inconsistent observation reuse, and exactly reconciles the
unique-request character and UTF-8 byte totals. The SOTA gate rejects chars/4
counts, partial-session counts, missing response identities, or tokenizer files
that no longer match their evidence bindings. Exact run files are therefore
materially larger and include public LongMemEval text; do not use this artifact
mode for a private corpus without an explicit data-handling decision.

Each report's `fused` lane is plain weighted RRF and `final` is the published
bundle. Relevance reranking ships disabled, so in every canonical run above the
two are the same ranking. The `longmemeval-s-rerank-experiment-*` files are the
one exception: they were produced with the stage enabled, and are kept as the
evidence behind the decision not to ship it (see the second 2026-08-09 addendum
in `docs/retrieval-benchmark.md`). Their `fused` lane is bit-identical to the
shipped configuration's `final` lane, which is what makes them a clean A/B.
**Do not quote the experiment files for headline numbers.** The saved 0.976
Recall@10 semantic run used the superseded pre-September-2025 histories. The
current official cleaned release has a different pinned digest, so
`benchmarks/sota/manifest.json` rejects that artifact until a semantic rerun
is preserved. A cleaned full-500 lexical-only diagnostic reproduced
Recall@10 0.8848, but it does not replace the missing semantic run.

## End-to-end QA artifacts (`longmemeval-s-qa-*`)

`scripts/run_longmemeval_qa.py` adds the reader and judge stages the files
above deliberately stop short of, and writes four files per run, tagged with
the reader model and, for sampled runs, the sample size:

| File | Contents |
| --- | --- |
| `longmemeval-s-qa-<reader>-hypotheses.jsonl` | Official hypothesis format — one `{"question_id", "hypothesis"}` object per line, exactly what LongMemEval's `src/evaluation/evaluate_qa.py` consumes |
| `longmemeval-s-qa-<reader>-chat-receipts.jsonl` | Content-bearing schema-v2 sidecar with exact credential-free HTTP request-body bytes, UTF-8 prompt bytes, endpoint, and decoded HTTP response-body bytes for every successful reader/development-judge call, base64 framed and digest bound |
| `longmemeval-s-qa-<reader>-run.json` | Schema-v4 per-question evidence plus exact hypothesis, retrieval-source, and chat-receipt path/bytes/SHA bindings, QA/retrieval implementation trees, and receipt indexes/digests |
| `longmemeval-s-qa-<reader>-report.json` | Schema-v4 report bound to the exact run and hypothesis bytes, with `dev_judge_accuracy` overall, slices, retrieval support, replayed token totals and latency |

**Every accuracy in those files is a development-judge accuracy.** The judge
prompts are copied verbatim from `evaluate_qa.py`, but they are answered by
whatever `--judge-model` names rather than by the official
`gpt-4o-2024-08-06`, so the numbers are for iteration only. A publishable
LongMemEval score comes from running the official script over the saved
hypothesis file. That is why the hypothesis file is a first-class artifact:
it is the input the official judge needs, and it is generated once.

For promotion evidence, use
`scripts/run_longmemeval_official_judge.py` rather than the upstream command
directly. The compatibility runner preserves the upstream prompt/model/
decoding/label semantics but also writes one `official_judge` raw receipt per
question. `scripts/build_longmemeval_official_report.py` requires that sidecar
and re-derives each label and token count from it. It also reloads the exact
dataset artifact bound by the generation run, reconstructs every reader prompt
from that dataset plus the saved retrieval bundle, and reconstructs every
official judge prompt from the dataset plus saved hypothesis. The upstream
evaluator alone discards response bytes and usage and therefore cannot clear
schema v4.

Pass `--retrieval-run <longmemeval-s-memory-*-run.json>` to reuse a completed
retrieval artifact for reader generation. Publishable replay is fail-closed: it
requires the schema-v2 artifact/protocol and canonical implementation tree, the
pinned cleaned full-500 corpus (23,867 sessions), the Qwen3-Embedding-0.6B
provider/model/dimension/instruction contract, exact response-model checking,
provider-observed call reconciliation, and zero degraded lanes. It then checks
selected-question coverage, recall depth, haystack counts, session-position
keys and relevance values before reconstructing the exact final contexts with
zero new embedding calls. `--allow-nonpublishable-retrieval-run` is an explicit
development escape; the resulting provenance says why it is nonpublishable and
the official-report compiler rejects it. Experimental referenced-time routing
is separately opt-in via `--temporal-query-routing`; its artifacts remain a
noncanonical A/B.

A full publishable replay also requires `--reader-revision` and, by default,
each chat response must report the exact requested model and a non-empty
provider request ID. The content-bearing sidecar retains the exact
credential-free request body, endpoint, prompt, and response body (but never
the API credential); the public run stores only its
artifact identity plus per-call receipt indexes, digests, normalized fields,
and accounting. The official compiler strictly reparses model, request ID,
content, finish reason, and reconciled provider usage from those bytes and
rejects any request-control, normalized-field, or source-prompt drift. For
DeepSeek V4, set `--reader-thinking-mode enabled|disabled` explicitly; its
provider default is thinking enabled and ignores temperature. `--allow-unverified-reader-response`
exists for development endpoints only; official report compilation rejects
those runs. Treat the sidecar as benchmark data containing conversation text.

Runs use the saved-run format defined in `docs/retrieval-evaluation.md` and can
be rescored directly:

```bash
uv run --extra dev python scripts/evaluate_retrieval_runs.py \
  benchmarks/retrieval/swarm-native-cockroach-run.json --k 10
```

Fresh schema-v2 reports cryptographically bind the exact saved-run bytes and
record the retrieval protocol, implementation tree, dense provider/model,
dimensions, query/document call accounting, lane degradation, ranking depth,
and temporal-routing state. The SOTA readiness evaluator recomputes the linked
run digest and size inside the repository; copied metrics or an unbound report
cannot satisfy the retrieval gate. Endpoint-backed runs additionally require
the response-reported model to match the pinned request and reconcile observed
successful HTTP calls with all 500 document batches and 500 queries.

Reruns overwrite these files in place, so a re-measurement shows up as a
reviewable diff. Identifiers in the rankings are corpus keys (track 1) or
`<position>:<session_id>` (track 2), never production UUIDs.

The 265 MiB LongMemEval-S cleaned release is not stored here. The runner caches it in
`~/.cache/swarmbrain-eval/` (override with `SWARMBRAIN_EVAL_DATA_DIR`) and
verifies its pinned SHA-256 on every run.
