# LongMemEval-S learned-reranker paired A/B

This integration measures one isolated retrieval change: the fixed fused
candidate order versus an opt-in learned scorer over exactly the same query,
candidate payloads, temporal payloads, tokenizer input, and evaluation depths.
It does not call a reader or judge and therefore does not claim LongMemEval QA
accuracy.

The canonical design is fixed at a 50-candidate window, learned blend
`alpha=1`, Recall/MRR/nDCG at 5 and 10, and a 10,000-resample paired question
bootstrap with seed `20260809`. There is deliberately no pass threshold in this
artifact yet.

## Evidence boundary

A comparison consists of three immutable inputs:

1. The byte-pinned cleaned LongMemEval-S dataset.
2. One schema-v2 source retrieval run whose `rankings.fused` lane supplies both
   arms. Its `final` ranking must still be the unchanged fused prefix.
3. One JSONL row per question containing the authoritative
   `LearnedRerankTrace` emitted by `RetrievalService`.

The provider sees runtime memory UUIDs, while LongMemEval judgments use session
keys such as `003:session-id`. The bridge never rewrites one as the other. The
compiler reconstructs every memory document and temporal projection from the
official dataset, verifies their core digests positionally against the trace,
then derives a UUID-to-session-key map used only for scoring.

Candidate projection and request/response verification reuse the production
contracts directly:

- `canonical_memory_rerank_input`
- `build_learned_rerank_request`
- `request_usage_dimensions`
- `LearnedRerankTrace`, `LearnedRerankReceipt`, and `LearnedRerankResult`
- `validate_learned_rerank_result`

Consequently, offline replay recomputes the core query, candidate-pool, request,
and response digests. It also reconciles character, UTF-8 byte, request-byte,
and provider token accounting. `usage.tokenized_input_sha256` is response-bound
and referenced by both counterfactual arms.

## Raw trace bridge

The resumable canonical runner is `scripts/run_longmemeval_reranker_ab.py`.
It requires repository-local source/trace/run artifact paths plus explicit
SHA-256 pins for the local executable and deployment manifest, and an immutable
model revision. The semantic embedding API key is read only from
`SWARMBRAIN_EMBEDDINGS_API_KEY`; reranker environment values can be inherited
only by allowlisted variable name and are never placed in argv or logs.

Each completed row is serialized as canonical JSONL and the complete verified
prefix is durably replaced as one atomic checkpoint. On resume, every existing
line is strictly reparsed, rebuilt with `build_trace_row`, checked against the
pinned dataset/source/identity/policy, and screened for reused client or
provider request IDs before any provider process is launched. For each new
question, the live pre-learned fused UUID order and the learned scorer's input
UUID order must both map exactly to the saved source session-key window before
the row is accepted. The final run manifest is created atomically only after
the authoritative compiler replay passes, local adapter call accounting
reconciles with all newly accepted rows, and the semantic replay's observed
embedding calls reconcile with those same newly executed questions. Dataset
and source artifact digests are rechecked after execution.

The implementation fingerprint is captured before scoring and covers the
complete `src/swarmbrain` Python tree plus the benchmark scripts and lock
inputs. The final manifest must reproduce that starting fingerprint, so a
mid-run code edit prevents publication. Local reranker and embedding resources
are closed independently on failure or cancellation.

One operational limitation is explicit: runtime UUID-to-session-key mapping is
available only after a question's retrieval execution returns. A live fused
order drift can therefore consume one local scorer call, but it is rejected
before the trace row is checkpointed and cannot enter the compiled evidence.

```bash
python scripts/run_longmemeval_reranker_ab.py \
  --source-retrieval benchmarks/retrieval/longmemeval-s-semantic-run.json \
  --traces benchmarks/retrieval/longmemeval-s-reranker-ab-traces.jsonl \
  --run benchmarks/retrieval/longmemeval-s-reranker-ab-run.json \
  --reranker-executable /deployment/qwen3-reranker-jsonl \
  --reranker-executable-sha256 <sha256> \
  --reranker-manifest /deployment/manifest.json \
  --reranker-manifest-sha256 <sha256> \
  --reranker-revision <immutable-revision> \
  --embeddings-base-url http://127.0.0.1:8000/v1
```

The command intentionally has no API-key flag. `--lme-download` is the only
mode that performs a dataset download; otherwise the pinned local file must
already exist.

After a successful service call, project the trace without changing its IDs or
digests:

```python
row = build_trace_row(
    case_index=index,
    record=official_dataset_row,
    source_case=source_retrieval_case,
    policy=policy,
    learned_trace=execution.trace.learned_rerank,
)
```

Write rows in official dataset order as strict newline-terminated JSONL. Once
that file is durable, `build_run_manifest(...)` binds its bytes, the source
retrieval bytes, the composite scorer/model/tokenizer/deployment identity, the
canonical policy, implementation tree, and provider accounting.

Use a separate run manifest for each immutable scorer configuration. A
single-component Qwen3 control and SmartSearch-style cross-encoder-only or
cross-encoder-plus-ColBERT (`0.7/0.3`) deployments must therefore remain
separate comparisons; their component artifacts, tokenizer artifacts, weights,
deployment manifest, adapter, and sanitized runtime environment are all part
of the core identity.

## Offline compilation

```bash
python scripts/build_longmemeval_reranker_report.py \
  --run benchmarks/retrieval/longmemeval-s-reranker-ab-run.json \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output benchmarks/retrieval/longmemeval-s-reranker-ab-report.json
```

The canonical CLI requires the official dataset digest, all 500 questions, a
publishable semantic source retrieval run, and source/current-tree identity.
It makes no model, tokenizer, embedding, database, reader, or judge calls.

The report contains paired deltas and confidence intervals overall and for the
temporal-reasoning, knowledge-update (conflict), and multi-session slices. Each
metric also records improved, regressed, and tied question counts. Any omitted,
duplicated, or added candidate; non-finite/out-of-range score; unstable ordering;
request-ID reuse; response identity drift; input/k change; accounting mismatch;
or byte mutation fails compilation.

The local process receipt is digest-bound but not cryptographically signed.
The compiler reconciles composite identity bundle digests; it does not reopen
model/tokenizer weights or prove that the process actually served those bytes.
A publishable model-weight identity therefore still needs local artifact replay
or a signed external attestation.
