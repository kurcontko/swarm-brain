# MemoryArena benchmark integration

Status: the memory API bridge is runnable and tested; official MemoryArena score
claims are blocked.

## Pinned upstream boundary

- Repository: `https://github.com/ZexueHe/MemoryArena`
- Commit: `6cd9de14b71915e39ac742a20dc33785e14b6aab`
- Dataset identity published by the project: `ZexueHe/memoryarena`, split
  `test`, with configs `bundled_shopping`, `group_travel_planner`,
  `progressive_search`, `formal_reasoning_math`, and
  `formal_reasoning_phys`
- Paper metrics: success rate (SR) and progress score (PS)
- Paper task-agent setting used by the strict config contract:
  `gpt-5.1-mini`

The official protocol and the repository's custom causal diagnostic are
different experiments:

| Experiment | Unit and arms | Claim it can support |
| --- | --- | --- |
| MemoryArena paper | Paper-declared 766 task groups across five configs; SR and PS | Continual-memory task performance under the official domains |
| Swarm Brain causal diagnostic | Paired, budget-matched 1/2/4-agent memory/no-memory cells | Memory-dependent multi-agent scaling |

Do not merge their rows, rename one as the other, or interpret MemoryArena as a
causal 1/2/4-agent result.

## Implemented compatibility seam

`benchmarks.integrations.memoryarena.server.create_memoryarena_app` implements
the three routes used by the pinned `memory/client.py`:

- `POST /memory/initialize` creates or replaces one scope.
- `POST /memory/add` publishes the chunk through `MemoryService`, drains the
  corresponding embedding work item before returning, and fails if the vector
  projection is incomplete.
- `POST /memory/wrap_user_prompt` recalls through canonical hybrid retrieval,
  applies the runtime token-budget packer, and renders the official outer
  `<memory_context>` and `User:` prompt shape.

Only `memory_system_name="swarmbrain"` is accepted. An official runner must use
a config overlay that changes the memory-system name and local memory URL only;
the task model, dataset, environment, limits, and scoring settings remain
frozen. Uninitialized users return the upstream `404` detail, and a mismatched
system returns the upstream `400` detail.

Each upstream `user_id` owns a separately composed runtime, vector index, work
queue, authenticated actor, and lock. The official runners use that value as a
task or task-group scope. Reinitialization replaces the old runtime and cleanup
is idempotent. No raw user ID enters Swarm Brain actor metadata.

The default scaffold uses `DeterministicEmbeddingProvider` solely to exercise
the real enqueue/lease/upsert/recall path without credentials, model calls, or
network calls. Its evidence always reports `publishable=false`; it must not be
described as a paper result.

The explicit `openai_semantic` mode is the semantic execution profile. It fixes
`Qwen/Qwen3-Embedding-8B` at 4096 dimensions, accepts only a 40- or 64-character
lowercase operator-declared model revision, requires the endpoint to return the
exact fixed model ID, and applies a frozen MemoryArena query instruction with
SHA-256
`5f12a399815ecbb080d4c0b5fd8f1f82b8a7a6dff9e8375ed5a6a955503a10ec`.
The API key is read only through a validated environment-variable name; neither
the credential, variable name, endpoint URL, nor instruction text is exported.

The OpenAI-compatible response model field attests only the served alias; it
does not prove which immutable weights revision the endpoint loaded. Therefore
the bridge records `model_revision_source=operator-declared-unverified`,
`model_revision_binding_verified=false`, and `publishable=false`, even when all
semantic calls and dense traces reconcile. Publication requires a future
trusted deployment attestation or local model/deployment manifest whose exact
path, bytes, and SHA-256 are bound to the endpoint and verified by the result
compiler. A revision-shaped CLI string alone is never treated as that proof.

One bridge-owned provider is shared across isolated task runtimes and closed
only when the bridge closes. Per-scope cleanup cannot close it. Semantic
evidence reconciles provider-observed document inputs, batch calls, query calls,
successful HTTP calls, attempts, stored memories, and completed embedding work.
Every required retrieval must contain one non-degraded dense lane. An
unavailable provider or dense gateway fails the request and records a fallback
event; it cannot silently continue as a successful semantic run.

## Content-free evidence

The bridge ledger records only operation names, sequence/invocation IDs, scope
and request/response hashes, byte/token/count fields, pseudonymous selected-ID
digests, embedding completion counts, latency, and closed error-class names. It
cannot export chunks, questions, prompts, memory context, or raw user IDs.

That is enough for a future strict compiler to bind an official harness ledger
to bridge activity while keeping benchmark content out of the public report.
It is deliberately not a scorer and contains no SR, PS, or frontier claim.

## Preflight

Validate only the pinned official memory API:

```bash
python scripts/run_memoryarena_external.py preflight \
  --checkout /path/to/MemoryArena \
  --bridge-only
```

Serve the local no-model bridge after that validation:

```bash
python scripts/run_memoryarena_external.py serve \
  --checkout /path/to/MemoryArena \
  --port 8000 \
  --evidence-output /tmp/memoryarena-bridge-evidence.json
```

Serve the strict semantic profile only after placing the credential in the
named environment variable (the value is never accepted as a CLI argument):

```bash
uv run --extra serve python scripts/run_memoryarena_external.py serve \
  --checkout /path/to/MemoryArena \
  --embedding-mode openai_semantic \
  --embedding-base-url https://embedding-endpoint.example \
  --embedding-api-key-env MEMORYARENA_EMBEDDING_API_KEY \
  --embedding-model-id Qwen/Qwen3-Embedding-8B \
  --embedding-model-revision <immutable-40-or-64-character-lowercase-hex> \
  --embedding-response-model Qwen/Qwen3-Embedding-8B \
  --embedding-dimensions 4096 \
  --evidence-output /tmp/memoryarena-semantic-evidence.json
```

This starts only the compatible memory service. Semantic bridge evidence is a
diagnostic input to a future official result compiler, not publishable revision
provenance, an SR/PS score, or a SOTA claim by itself.

Full-protocol preflight additionally requires five config overlays and an
immutable local dataset manifest:

```json
{
  "schema_version": 1,
  "dataset": "ZexueHe/memoryarena",
  "revision": "<immutable 40-to-64-character lowercase hex revision>",
  "split": "test",
  "configs": {
    "bundled_shopping": {
      "path": "bundled_shopping.parquet",
      "sha256": "<sha256>",
      "task_group_count": 0
    },
    "formal_reasoning_math": {
      "path": "formal_reasoning_math.parquet",
      "sha256": "<sha256>",
      "task_group_count": 0
    },
    "formal_reasoning_phys": {
      "path": "formal_reasoning_phys.parquet",
      "sha256": "<sha256>",
      "task_group_count": 0
    },
    "group_travel_planner": {
      "path": "group_travel_planner.parquet",
      "sha256": "<sha256>",
      "task_group_count": 0
    },
    "progressive_search": {
      "path": "progressive_search.parquet",
      "sha256": "<sha256>",
      "task_group_count": 0
    }
  },
  "protocol_reconciliation": {
    "paper_declared_task_groups": 766,
    "paper_table_component_total": 736,
    "resolved_task_groups": 766,
    "resolution_path": "protocol-reconciliation.json",
    "resolution_sha256": "<sha256 of the external reconciliation record>"
  }
}
```

The zero counts above are placeholders and intentionally fail validation.
Artifact paths are relative to the manifest; symlinks, path escapes, hash
mismatches, mutable revisions, missing configs, or totals other than 766 fail
closed. The reconciliation record is also a required, hash-bound regular file;
the manifest, reconciliation record, checkout, configs, and every component of
their artifact paths must be free of symbolic links. Checkout validation treats
untracked files as a dirty worktree.

## Exact upstream blocker

The paper overview declares 766 task groups, but the five published component
counts are 150 + 270 + 256 + 40 + 20 = 736. The pinned repository calls itself
a preview, keeps the dataset external, and does not publish a frozen full-run
artifact schema/compiler binding all five configs to SR and PS. Its checked-in
configs also use `gpt-5-mini`, not the paper setting required above.

For those reasons, `preflight` always reports
`official_execution_supported=false` at this commit. This is intentional: the
script can serve the compatibility API but cannot launch an official run or
compile scores. Resolving the boundary requires an immutable dataset snapshot,
the five verified paper config overlays, a documented 766/736 reconciliation,
and a pinned official SR/PS compiler.

## Superseded manifest replacement candidate

`benchmarks/integrations/memoryarena/manifest-replacement-fragment.json`
preserves the former proposal to replace `multi_agent_causal_scaling`; it never
edited the live manifest. Manifest schema v2 supersedes that proposal:
MemoryArena now covers the separate `interdependent-environment-action`
dimension, while the causal 1/2/4-agent gate is required for
`multi-agent-causal-memory-gain`. Replacing either with the other would leave a
claim dimension uncovered and the readiness evaluator rejects the manifest.
The historical fragment's `scientific_scope_equivalent=false` records exactly
why it must not be applied. Current preview evidence also cannot pass its
dataset, compiler, frontier, or preview-boundary checks.
