# Mem2ActBench integration

This integration measures whether Swarm Brain retrieval improves memory-grounded tool use on
the 400-task Mem2ActBench evaluation subset. It produces measurement artifacts only. It does
not ship a score or make a comparative/SOTA claim.

The paper is [Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in
Task-Oriented Autonomous Agents](https://aclanthology.org/2026.acl-long.370/). The upstream
repository does not release an evaluator. The parameter metrics here are therefore a strict,
auditable reimplementation of the definitions in paper section 4.1, not output from an
"official scorer."

## Pinned data contract

The loader requires the official checkout at commit
`b00726940b5abbe9bd324bdd7a2cb272f5c62a29`. It uses `toolmembench_small/`, whose name is
historical: it is the complete paper evaluation subset with 400 tasks and 429 sessions. The
2,029-session `Mem2ActBench/` directory is the construction corpus, not the fixed evaluation
subset.

| File | SHA-256 |
| --- | --- |
| `toolmembench_small/qa_dataset.jsonl` | `c5e3f47799d850b607d0ff56829335f843e208ad4d3b1eae89a814eae1974b09` |
| `toolmembench_small/toolmem_conversation.jsonl` | `c935adfbe0e1743b8eb373eba31611b97cf396ee09516f68f6621e49365cbcaf` |
| `toolmembench_small/benchmark_statistics.json` | `a66ae640bb34e1bc1eb2362f8a0ede511017782002ce3f288b73ae2c9fc3c600` |

Loading fails closed on commit or digest drift, absent/reordered/duplicate task IDs, duplicate
sessions or source ownership, invalid JSON (including duplicate keys and non-finite numbers),
malformed calls/schemas, and unexpected source-reference changes.

The pinned upstream release has two disclosed integrity defects:

- Eleven task provenance IDs are absent from the 429-session JSONL. The loader accepts exactly
  the pinned set and rejects any additional or silently repaired gap. Normal retrieval does not
  use those labels: it ingests all 429 published sessions. Oracle retrieval uses the published
  evolution-chain evidence and never fabricates a missing raw conversation.
- `qa_283` serializes `tool_call.name` as the number `4`, while its public target schema names
  `4D Dream Dictionary`. The loader accepts only that exact value/schema pair and records the
  deterministic schema-name repair in the dataset fingerprint.

These defects remain visible in every report. A future upstream repair requires new reviewed
hashes rather than being accepted implicitly.

## Two evaluation conditions

The harness separates parameter-grounding comparability from stricter tool selection.

| Condition | Reader's tool view | Intended metric use |
| --- | --- | --- |
| `target_tool_given` | Exactly the generic published target schema and name; never target arguments, grounding labels, or evidence | Paper-comparable parameter precision/recall/F1 and slot accuracy |
| `full_catalog` | The same complete, deduplicated catalog for every task and arm | Stricter tool accuracy and exact tool-plus-arguments |

Paper section 4.1 says its main parameter-grounding results provide the ground-truth tool.
Consequently, report fields `/memory/parameter_f1`, `/memory/parameter_precision`,
`/memory/parameter_recall`, and `/memory/slot_accuracy` come only from `target_tool_given`.
`/memory/tool_accuracy` and `/memory/exact_tool_and_arguments` come only from `full_catalog`.
The condition is named beside every gate-facing value, and full details remain under
`/conditions`.

The paper's Table 4 hybrid-at-5 result, parameter F1 `0.307`, is a passive-retrieval ablation
baseline, not the SOTA frontier. The highest main-table value is A-mem's `0.3593` with
Qwen2.5-72B-Instruct. Both references and their roles are encoded in the report so a later gate
can freeze the comparable frontier without reinterpreting old artifacts.

## Three paired arms

Every question runs all three arms inside both conditions:

- `no_memory`: query plus the condition's tool view, with no retrieved context.
- `swarm`: the fixed 429-session corpus is ingested once; Swarm Brain receives only the natural
  language query. One frozen `RetrievalResult` is reused across both conditions.
- `oracle`: the published evolution-chain facts and supporting utterances, excluding source IDs,
  target arguments, grounding annotations, and the target call label.

The reader receives an allowlisted `ReaderRequest` containing only `condition`, `query`,
`memory_contexts`, and `tool_catalog`. It never receives a QA ID or arm name. The memory bridge
receives only an allowlisted public corpus at ingestion and the query text at recall. Session and
turn construction/source IDs are stripped before the injectable bridge boundary. These narrow
types make the leakage boundary inspectable in tests.

## Metrics

A reader must return strict JSON with exactly this shape:

```json
{"name": "tool name", "arguments": {"parameter": "value"}}
```

Values use type-sensitive, recursive JSON equality: `true`, `1`, and `1.0` are distinct. A slot
is one top-level argument. An extra predicted argument is a false positive; a missing or wrong
gold argument is a false negative. A wrong tool gets zero parameter credit. Reports include:

- tool accuracy and exact tool-plus-arguments;
- micro and macro parameter precision, recall, and F1;
- micro slot accuracy;
- no-memory/swarm/oracle task-level metrics;
- deterministic paired question-cluster bootstrap intervals for `swarm - no_memory` and
  `oracle - swarm` within each condition;
- raw responses, parsed predictions, exact reader contexts, retrieval IDs/scores/reasons,
  latency, token counts, provider metadata, and typed failures for every attempt.

The paired bootstrap resamples questions, not individual slots, so every arm outcome and all
slots belonging to a question remain correlated.

## Running

The default memory bridge uses `build_in_memory_runtime` and only the public
`runtime.memory.publish` / `runtime.memory.recall` application services. A custom public HTTP or
durable bridge can be supplied as a zero-argument `module:callable` returning the `MemoryBridge`
protocol. The reader is another zero-argument factory returning `ToolSelectionReader`; factories
may be async and may obtain provider configuration from their own environment.

The default bridge has no embedding provider and is a reproducible lexical smoke/default, not an
implicit reproduction of the paper's BGE-m3 hybrid arm. The report records the bridge class,
runtime backend, embedding provider, and embedding model. A comparable semantic run should inject
the intended, separately pinned bridge configuration rather than leaving this provenance implicit.

The canonical semantic bridge uses the same public runtime while draining and reconciling every
durable corpus-embedding job before the first query. It requires the endpoint to report the exact
requested model on every response and records provider-observed document, query, successful HTTP,
and retry-attempt counts. The SOTA gate requires 429 completed corpus embeddings and exactly 400
query embeddings; a configured-but-undrained dense lane cannot masquerade as a semantic run.

The repository includes a canonical OpenAI-compatible reader factory. Its prompt and canonical
payload layout are pinned as `mem2act-tool-selection-reader-v1`; decoding is fixed to temperature
zero, top-p one, seed zero, and one response. It requests a strict JSON schema whose tool-name enum
is derived from the allowlisted `ReaderRequest`, then independently rejects malformed JSON,
duplicate or extra top-level fields, tools outside that request's catalog, model aliases, missing
or inconsistent token usage, and truncated responses. Transient transport and 408/409/425/429/5xx
failures use bounded exponential retry; protocol failures never retry. The configured API-key
environment variable is read only when a request is sent and its value is never put in metadata.

Configure the fixed Qwen reader without placing a credential on the command line:

```bash
export MEM2ACT_READER_BASE_URL=https://reader.example/v1
export MEM2ACT_READER_MODEL=Qwen/Qwen2.5-72B-Instruct
export MEM2ACT_READER_REVISION=immutable-checkpoint-id
export MEM2ACT_READER_API_KEY_ENV=MEM2ACT_READER_API_KEY
export MEM2ACT_READER_API_KEY=...

export MEM2ACT_EMBEDDINGS_BASE_URL=https://embeddings.example/v1
export MEM2ACT_EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-0.6B
export MEM2ACT_EMBEDDINGS_REVISION=immutable-checkpoint-id
export MEM2ACT_EMBEDDINGS_API_KEY_ENV=MEM2ACT_EMBEDDINGS_API_KEY
export MEM2ACT_EMBEDDINGS_API_KEY=...
```

For an explicitly unauthenticated self-hosted endpoint, set
`MEM2ACT_READER_API_KEY_ENV=`. Optional bounded controls are
`MEM2ACT_READER_TIMEOUT_SECONDS`, `MEM2ACT_READER_MAX_RETRIES`,
`MEM2ACT_READER_BACKOFF_INITIAL_SECONDS`, `MEM2ACT_READER_BACKOFF_MAX_SECONDS`,
`MEM2ACT_READER_MAX_OUTPUT_TOKENS`, `MEM2ACT_READER_MAX_INPUT_BYTES`, and
`MEM2ACT_READER_MAX_RESPONSE_BYTES`.

```bash
.venv/bin/python scripts/run_mem2act_bench.py \
  --repo /private/tmp/swarmbrain-mem2act \
  --reader-factory benchmarks.integrations.mem2act.openai_reader:build_reader \
  --memory-bridge-factory benchmarks.integrations.mem2act.runtime_bridge:build_openai_semantic_in_memory_bridge \
  --expected-reader-model "$MEM2ACT_READER_MODEL" \
  --reader-revision "$MEM2ACT_READER_REVISION" \
  --output-prefix benchmarks/sota/mem2act
```

This writes three artifacts:

- `mem2act-predictions.jsonl` contains the raw per-attempt evidence used for scoring;
- `mem2act-run.json` binds that JSONL by path, byte count, row count, and SHA-256, and records the
  dataset, configuration, implementations, ingestion evidence, and hashes of the compiler/runtime
  source tree;
- `mem2act-report.json` is compiled offline from those two inputs. The compiler reparses every raw
  reader response, recomputes every task metric and aggregate, and does not call a model.

Existing files are protected; `--overwrite` must be explicit. `--task-limit` is for smoke tests
and marks the protocol incomplete. A full comparable artifact requires all 400 tasks, both
conditions, all three arms, one fixed reader model/revision, and zero unaccounted failures.

The report can be reproduced later without provider access:

```bash
.venv/bin/python scripts/build_mem2act_report.py \
  --run benchmarks/sota/mem2act-run.json \
  --dataset-dir benchmarks/sota/evidence/mem2act/dataset \
  --output benchmarks/sota/mem2act-report-rebuilt.json
```

The replay dataset directory must contain the pinned files at their original relative paths:
`toolmembench_small/qa_dataset.jsonl`,
`toolmembench_small/toolmem_conversation.jsonl`, and
`toolmembench_small/benchmark_statistics.json`. The compiler verifies all three SHA-256 digests,
reconstructs the 400 tasks and canonical tool catalog, and checks every raw prediction row's task,
gold label, target/full-catalog identity, and oracle context against those bytes. The live runner
uses its already verified `--repo` checkout for this compilation; the path itself is not embedded
in report bytes. A later readiness replay uses the repository-local evidence copy above.

Offline compilation fails closed on duplicate-key or non-finite JSON, unsafe or symlinked paths,
artifact byte/row/hash drift, implementation-tree drift, missing paired cells, changed frozen
contexts, pinned-dataset drift, tool-catalog drift, reader-model drift, canonical prompt/request
drift, parsed-response drift, or stored metrics that do not recompute. Nonofficial fixture runs are
explicitly marked with `official_dataset_verified=false` and can never become complete protocol
artifacts. The report binds both raw inputs and records that the current source tree was verified.
The run and prediction files may be relocated together inside the repository, but the prediction
filename must continue to match the run artifact's safe sibling reference and bound hash.

The compiler deliberately keeps `scoring.official_evaluator_released` set to `false`. Rebuilding
the report proves reproducibility of this disclosed paper-definition reimplementation; it cannot
turn it into output from an unreleased official evaluator or establish scorer comparability.

The stable gate-facing report projection is:

- `/evaluation/oracle_arm` and `/evaluation/no_memory_arm`;
- `/memory/parameter_f1` and `/memory/slot_accuracy` from `target_tool_given`;
- `/memory/exact_tool_and_arguments` and `/memory/tool_accuracy` from `full_catalog`;
- `/comparison/memory_vs_no_memory_ci95/{lower,upper}` for target-tool-given parameter F1.
- `/conditions/full_catalog/paired_bootstrap/pairs/swarm-minus-no_memory/`
  `exact_tool_and_arguments/{ci_low,ci_high}` for the stricter exact-call gain.

Passing a numeric threshold proves only that the pinned run cleared that declared comparator.
It does not by itself establish SOTA.
