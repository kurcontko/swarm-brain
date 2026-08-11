# LongMemEval-V2 Swarm adapter

This integration keeps the official query-privacy boundary intact while
producing the content-free operation evidence required by
`scripts/build_longmemeval_v2_report.py`.

The official backend receives only question text, an optional public image
path, and a random run-local handle. It performs exactly:

1. `recall_memory(query)` to obtain canonical seed IDs.
2. `read_expand_memory(query, seed IDs)` to obtain the bounded context returned
   to the fixed reader.

This is a fixed two-stage recall-then-expand path, not an adaptive multi-round
agent loop. The report records that distinction explicitly; a later adaptive
controller must earn its own evidence rather than inheriting the two-stage
trace's `iterative_search_read_expand` compatibility flag.

The backend cannot truthfully create compiler invocation IDs itself: those IDs
depend on the stable question ID, and the official harness deliberately keeps
that ID private during `Memory.query`. The runner therefore journals an opaque,
content-free trace in memory. After the official harness has truncated and
token-counted the context, a narrow prompt-row binder restores the question ID,
pseudonymizes memory IDs, computes deterministic invocation IDs, reconciles
tokens and latency, and replaces provisional metadata with the final trace
digest. The stable question ID never crosses back into the memory bridge.

## Offline checks

No endpoint or model is contacted by either command:

```bash
python scripts/run_longmemeval_v2_swarm.py dry-run

python scripts/run_longmemeval_v2_swarm.py preflight \
  --repository /path/to/pinned/LongMemEval-V2 \
  --data-root /path/to/longmemeval-v2-data \
  --tier small \
  --dataset-revision <immutable-dataset-revision> \
  --expected-dataset-manifest-sha256 <recorded-sha256>
```

Preflight requires benchmark commit
`ef67f10aacd9080c75aeb2dd527a0af25dc26f1b`, a clean checkout, all 451
questions, complete tier haystacks, every referenced trajectory and image, and
an exact caller-recorded dataset manifest. A digest is never learned and
silently trusted during the same official run.

## Official run boundary

A bridge factory uses `module:callable` syntax. It receives `AdapterConfig` and
must return a synchronous `SwarmOperationBridge`. The bridge owns ingestion,
scope/task lease setup, and any local-runtime or authenticated transport. Its
search methods receive query text only. `close()` is required, synchronous, and
idempotent. The runner closes each per-question bridge immediately after its
prompt row is built (or in a run-level `finally` after any failure); a genuinely
shared haystack remains open until all of its queries finish.

`bridge_params` are copied into the official runtime package and therefore must
contain public configuration only. Inline API keys, bearer tokens, passwords,
cookies, private keys, and other credential-bearing fields are rejected
recursively. Pass only an environment-variable name through a credential key
ending in `_env`, for example `{"api_key_env": "SWARMBRAIN_API_KEY"}`; the
bridge factory resolves that variable at runtime.

Running the harness is deliberately opt-in because it calls the fixed reader
`Qwen/Qwen3.5-9B` and judge `gpt-5.2`:

```bash
python scripts/run_longmemeval_v2_swarm.py run \
  --repository /path/to/pinned/LongMemEval-V2 \
  --data-root /path/to/longmemeval-v2-data \
  --domain web --tier small --operating-point balanced \
  --dataset-revision <immutable-dataset-revision> \
  --expected-dataset-manifest-sha256 <recorded-sha256> \
  --bridge-factory benchmarks.integrations.longmemeval_v2.runtime_bridge:build_local_runtime_bridge \
  --bridge-params '{"backend":"in_memory","retrieval_mode":"openai_hybrid","chunk_chars":6000,"embedding_base_url":"http://embeddings/v1","embedding_api_key_env":"LME_V2_EMBEDDING_API_KEY","embedding_model_revision":"<public-immutable-checkpoint-revision>","embedding_dimensions":4096}' \
  --reader-base-url http://reader/v1 \
  --evaluator-base-url https://judge/v1 \
  --output-dir /runs/swarmbrain_web_small \
  --operation-ledger /evidence/swarmbrain_web_small.json \
  --allow-model-api-calls
```

The bundled factory is the executable default for Swarm Brain itself. It uses
the canonical in-memory composition root and application services—not a fixture
or a benchmark-only retrieval implementation. For every ordered question
haystack it derives deterministic tenant/project/repository/swarm/run/agent/task
UUIDs from the first public trajectory fingerprint, pinned protocol, and a
run-owned instance ordinal (never the private question ID), creates a fresh
runtime, joins the run, adds and claims a task, publishes confirmed task-scoped
state chunks, drains the durable embedding worker, then calls canonical `recall` and lease-bound
`read_expand_memory`. Adjacent chunks are linked through the normal memory-link
path. Shutdown completes (or releases) the task, closes the runtime, stops its
event-loop thread, and drops the instance before the reader phase.

Bundled public bridge parameters are strict:

- `backend`: only `"in_memory"` (default) is accepted.
- `retrieval_mode`: `"openai_hybrid"` for publishable evidence;
  `"deterministic_hybrid"` (default) and `"lexical"` are development-only.
- `chunk_chars`: 1024–32768 (default 6000).
- `embedding_dimensions`: 32–4096. `openai_hybrid` defaults to the official
  `Qwen/Qwen3-Embedding-8B` checkpoint's native 4096 dimensions and records any
  explicit projection dimension in evidence.
- `embedding_base_url`: required for `openai_hybrid`; it must be HTTP(S) and
  cannot contain credentials, a query string, or a fragment.
- `embedding_api_key_env`: required environment-variable name. The resolved
  value is never persisted.
- `embedding_model_revision`: required nonempty public checkpoint revision.
- `embedding_model` and `embedding_response_model`: optional but, if present,
  must both equal the pinned `Qwen/Qwen3-Embedding-8B` identifier.
- `dense_min_similarity` and `recall_min_score`: finite values in `[0, 1]`.

Unknown fields, remote backends, misspelled modes, and embedding options on the
wrong mode fail closed. The runtime's token-signing secret is generated
ephemerally in process and is never placed in `memory_params`, runtime
artifacts, logs, or evidence ledgers.

The publishable path uses `OpenAICompatibleEmbeddingProvider` with exact
response-model verification and the official RAG instruction, `Given a
question about past agent trajectories, retrieve relevant memory entries that
help answer it.` Only its SHA-256 digest is stored in evidence. Every inserted
canonical memory must produce one completed durable embedding and one
provider-observed document HTTP success; every question must produce exactly
one provider-observed query HTTP success. The bridge reconciles attempts and
successes separately and rejects the canonical runtime's normal lexical
degradation if the remote query embedding fails. Per-query content-free proof
records provider, model, caller-pinned public revision, dimensions, instruction
digest, document/query accounting, exact response-model verification, and that
no deterministic fallback occurred. The ledger, external sidecar, packaged
`per_question.jsonl` digest, strict report compiler, and SOTA manifest all bind
and revalidate that proof.

Run both domains for every operating point, then use the pinned repository's
leaderboard scripts to build the official package (pass `--method swarmbrain`
when reshaping each operating point). Bind the external ledgers afterward:

```bash
python scripts/run_longmemeval_v2_swarm.py sidecar \
  --package /submissions/swarmbrain-small \
  --operation-ledger /evidence/swarmbrain_web_small.json \
  --operation-ledger /evidence/swarmbrain_enterprise_small.json \
  --output /evidence/swarmbrain-small-sidecar.json
```

The final write is atomic and gated by the strict compiler. Incomplete package
coverage, model/revision/protocol drift, token or latency mismatch, broken
search-to-expand linkage, a changed package file, or a trace digest mismatch
leaves no sidecar artifact.

Each operation ledger records both the canonical protocol SHA-256 and the exact
runtime memory-config SHA-256 at execution time. Sidecar construction recomputes
both values from the packaged official artifacts, validates the complete pinned
`AdapterConfig`, and rejects post-run configuration drift before compiling any
claim.

The bundled dry run is a protocol fixture only. It is explicitly marked as
non-leaderboard, non-SOTA evidence.
