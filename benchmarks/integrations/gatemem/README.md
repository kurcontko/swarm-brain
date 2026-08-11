# GateMem external-prediction integration

This adapter runs GateMem episodes incrementally through Swarm Brain and writes
the JSONL format accepted by GateMem's official external scorer. It is a
reproducible evaluation harness, not a benchmark result or a SOTA claim.

The integration is pinned to the official
[`rzhub/GateMem`](https://github.com/rzhub/GateMem) commit
`603f9f4b4ba4b77f043c20f85687fa016fd720b0`. Before reading a domain it checks
that commit plus SHA-256 digests for the episode file, checkpoint file,
external scorer, and prediction-format documentation. Local dataset or scorer
edits therefore fail closed.

## Security boundary

The raw checkpoint row is converted immediately to `PublicCheckpoint`. Only
these fields survive:

- checkpoint, episode, and as-of turn identifiers;
- requester principal ID and role;
- query text.

`query_type`, `attack_type`, `expected_action`, `judge_spec`, `leak_targets`,
and legacy gold/policy fields have no representation in the runner or answer
model request. Recursive guards also reject those names in public relationship
facts, audience manifests, answer-model output, predictions, and audit data.

The data path is:

```text
official episode turns -> public turn allowlist -> audience policy
  -> isolated (episode, principal) Swarm run -> current-memory recall
  -> fixed-token context pack -> answer-model protocol -> predictions.jsonl

official hidden checkpoint annotations -----------------------> scorer only
```

Each `(episode_id, principal_id)` maps to a deterministic, distinct Swarm Brain
`run_id` and `agent_id`. Actor-owned tenant/project/repository/swarm/run IDs
come from the runtime actor or bearer token; they are never accepted from a
memory request body. Swarm Brain then revalidates every published or recalled
memory against that actor scope. A turn reaches a principal view only if the
configured `AudiencePolicy` names it.

The default `SpeakerOnlyAudiencePolicy` is intentionally conservative and is
useful for contract tests and smoke runs. It is not a complete GateMem access
policy and should not be used to claim a full benchmark result. A full run must
use a separately reviewed audience manifest derived only from public episode
principals and relationships:

```json
{
  "schema_version": 1,
  "gatemem_commit": "603f9f4b4ba4b77f043c20f85687fa016fd720b0",
  "episodes": {
    "episode-id": {
      "t001": ["principal-a", "principal-b"],
      "t002": ["principal-a"]
    }
  }
}
```

The mapping must cover every ingested turn, cannot broadcast implicitly, and
cannot name an unknown principal. Its SHA-256 is recorded in the run audit.

## Active forgetting

An explicit public deletion directive is never written verbatim as a new
memory. The deterministic interpreter finds targets only among current memories
inside the same principal scope and publishes a content-free tombstone with
`supersedes_memory_id`. Swarm Brain's append-only history retains provenance,
while ordinary current recall excludes the superseded version. The harness also
keeps hashes of distinctive deleted phrases and suppresses later turns that try
to reintroduce the deleted value. Neither the directive nor deleted plaintext
is copied into the audit artifact.

This is behavioral active forgetting over the public/current recall surface,
not a claim of physical database erasure. The audit records the predecessor and
tombstone memory IDs, versions, content digests, principal scope digest, and
latency.

## Verify and run

Prepare the pinned official checkout:

```bash
git clone https://github.com/rzhub/GateMem.git /private/tmp/swarmbrain-gatemem
git -C /private/tmp/swarmbrain-gatemem checkout \
  603f9f4b4ba4b77f043c20f85687fa016fd720b0

.venv/bin/python scripts/run_gatemem_external.py \
  --gatemem-dir /private/tmp/swarmbrain-gatemem \
  --domain office \
  --verify-only
```

The in-memory runtime backend exercises the canonical `MemoryService` and is
the simplest reproducible execution path. Point the answer reader at an
OpenAI-compatible endpoint and use an audited audience manifest for a full
evaluation:

```bash
export GATEMEM_ANSWER_BASE_URL=http://127.0.0.1:8000/v1
export GATEMEM_ANSWER_MODEL=reader-model
export GATEMEM_ANSWER_REVISION=immutable-provider-reported-revision

.venv/bin/python scripts/run_gatemem_external.py \
  --gatemem-dir /private/tmp/swarmbrain-gatemem \
  --domain office \
  --backend memory \
  --audience-manifest /path/to/office-audiences.json \
  --predictions /tmp/gatemem-office-predictions.jsonl \
  --audit-output /tmp/gatemem-office-audit.json \
  --context-token-budget 4096 \
  --score-out-dir /tmp/gatemem-office-scores
```

### Opt-in authenticated resume state

Long runs can checkpoint after each **complete episode**. Set a local HMAC key
of at least 32 bytes in an environment variable, then add `--resume-state`:

```bash
export GATEMEM_RESUME_HMAC_KEY='replace-with-at-least-32-random-bytes-kept-local'

.venv/bin/python scripts/run_gatemem_external.py \
  --gatemem-dir /private/tmp/swarmbrain-gatemem \
  --domain office \
  --backend memory \
  --audience-manifest /path/to/office-audiences.json \
  --answer-base-url http://127.0.0.1:8000/v1 \
  --answer-model reader-model \
  --answer-revision immutable-provider-reported-revision \
  --predictions /tmp/gatemem-office-predictions.jsonl \
  --audit-output /tmp/gatemem-office-audit.json \
  --resume-state /tmp/gatemem-office.resume.json
```

Resume is opt-in; runs without `--resume-state` retain the original behavior.
It is restricted to `--backend memory`. A partial episode can leave durable
HTTP state ahead of its earlier checkpoints, so HTTP resume fails closed until
the backend supports isolated attempt scopes or snapshot restoration.
Use `--resume-key-env NAME` to name a different environment variable. The key
value is never serialized. The state file contains model-visible prediction
evidence, so protect it like the final artifacts and retain the key until the
run has been finalized.

The checkpoint boundary is deliberately an episode, not an individual query.
Checkpoints within one episode share incremental ingestion and deletion state;
restarting in the middle would splice incompatible memory histories. Completed
episodes have independent deterministic principal scopes, so they can be
skipped safely. Episode and checkpoint chunks must form an exact prefix of the
official task order, and predictions and their ingest audit are authenticated
as one paired unit.

Before any remaining reader call, the runner authenticates and strictly
validates the state and requires an exact fingerprint match for:

- pinned GateMem repository revision and episode/checkpoint SHA-256 values;
- domain, selected official-order episode/checkpoint IDs, and audience-policy
  type plus manifest SHA-256;
- answer endpoint, exact model name and provider-reported immutable revision,
  key-variable name, timeout, fixed prompt version/hash, protocol, and decoding
  configuration;
- harness schema, turn/deletion schemas, interpreter, official scorer and
  prediction-format hashes;
- the exact local GateMem integration, Swarm Brain Python source tree,
  `pyproject.toml`, and `uv.lock` digests;
- checkout/input paths, backend, HTTP token-manifest hash when applicable,
  retrieval parameters, context budget, resume key-variable/state paths, and
  final artifact/scorer paths.

Any mismatch, duplicate JSON field, non-finite value, reordered/duplicate
checkpoint, modified evidence, wrong HMAC key, or concurrent invocation fails
closed. Choose a new `.resume.json` path for a deliberately changed run; do not
edit or merge state files.

Resume state is written with mode `0600`, fsynced, and atomically replaced under
an exclusive process lock. Its top-level artifact marker is
`swarmbrain-gatemem-resume-state`, not the official JSONL or audit schema.
Partial predictions and audits are never written to the canonical output
paths. Once every selected episode is authenticated, the canonical files are
each atomically replaced and a final completion manifest binds their exact
names, sizes, and SHA-256 digests. The report compiler requires that marker, so
a crash between artifact replacements cannot promote a stale pair. Use
`--completion-manifest` to override its default
`<predictions>.completion.json` path. A fully complete state can regenerate all
three artifacts without another answer-model call.

The completion marker also carries a content-free `execution_lineage` block.
It distinguishes uninterrupted, checkpointed, resumed, and complete-state replay
runs; records the authenticated resume-payload SHA-256 and completed prefix/final
episode counts; and embeds the exact implementation fingerprint already bound by
the resume state. It never contains the HMAC key, authentication tag/key ID, or
resumable prediction chunks. Runs without `--resume-state` explicitly record
`uninterrupted` lineage with no state digest. The report compiler rejects missing,
inconsistent, or malformed lineage and requires its final episode count to match
official domain coverage.

`--episode-id` still denotes a smoke/partial-domain run even when that selected
resume plan says `complete`, and the report compiler continues to reject it as
full-domain evidence.

`--score-out-dir` prints, but does not execute, the exact pinned official
rule-scorer command. Run that command in the GateMem environment. Add the
official judge flags when producing judge-backed scores:

```bash
python /private/tmp/swarmbrain-gatemem/bench/scripts/score_predictions.py \
  --data_dir /private/tmp/swarmbrain-gatemem/bench/data/office \
  --predictions /tmp/gatemem-office-predictions.jsonl \
  --out_dir /tmp/gatemem-office-scores \
  --gate_by_action \
  --use_llm_judge \
  --judge_provider openai \
  --judge_model gpt-4o
```

Use `--episode-id` for a smoke run. A partial prediction file is not a full
domain result.

After generating and officially scoring all four complete domains, compile the
machine-checkable SOTA-gate artifact:

```bash
.venv/bin/python scripts/build_gatemem_report.py \
  --gatemem-dir /private/tmp/swarmbrain-gatemem \
  --evidence education /tmp/education.jsonl /tmp/education.audit.json /tmp/education-scores \
  --evidence household /tmp/household.jsonl /tmp/household.audit.json /tmp/household-scores \
  --evidence medical /tmp/medical.jsonl /tmp/medical.audit.json /tmp/medical-scores \
  --evidence office /tmp/office.jsonl /tmp/office.audit.json /tmp/office-scores \
  --output benchmarks/sota/gatemem-report.json
```

The compiler will not emit a partial result. It requires the pinned 91 episodes
and 2,218 unique checkpoints; exact prediction, normalized-prediction, rule
score, and judge-score coverage; action-gated judging with zero parse failures;
one fixed reader and judge; provider token usage for every checkpoint; and a
complete public-data-derived audience manifest for every domain. It recomputes
utility, over-refusal, and the per-checkpoint union of answer/judge leakage with
prompt-context leakage, then writes the exact `dataset` and `metrics` fields
consumed by `benchmarks/sota/manifest.json`. Every source artifact and pinned
dataset/scorer file is SHA-256 recorded. The hash-binding completion manifest
must match each prediction/audit pair. Merely having the compiler does not
constitute a passing GateMem result.

The HTTP backend uses only `POST /v1/memories` and
`POST /v1/memories:recall`. It requires `--api-url` and a local token manifest:

```json
{
  "schema_version": 1,
  "gatemem_commit": "603f9f4b4ba4b77f043c20f85687fa016fd720b0",
  "tokens": {
    "episode-id::principal-id": "opaque-bearer-token"
  }
}
```

Tokens must carry the deterministic scope IDs and only `memory:publish` and
`memory:recall`. Token values are never written to predictions or audit data.

## Output contract and audit

Each prediction line contains exactly the official join key and output object,
plus a namespaced audit object:

```json
{
  "checkpoint_id": "...",
  "output": {
    "action": "answer",
    "answer": "...",
    "answer_structured": {},
    "used_record_ids": [],
    "memory_audit": {},
    "llm_usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    }
  },
  "swarmbrain_audit": {}
}
```

GateMem's external normalizer consumes `checkpoint_id` and `output` and ignores
the namespaced audit field. `output.memory_audit` follows GateMem's native
prompt-exposure schema and contains the exact recalled-memory list shown to the
reader; this is necessary for the official rule scorer's end-to-end context
leakage checks. The separate run audit remains content-free. It records:

- pinned dataset commit and adapter/config versions;
- audience-policy type and manifest digest;
- recalled memory IDs, versions, source turns/speakers, scores, and content
  digests, without copying prompt context;
- fixed context budget and estimated packed tokens;
- provider-reported answer-call input/output tokens for every checkpoint;
- incremental ingest, recall, answer, and total query latency;
- append, supersession, and safe-noop ingestion operations.

Real answer calls are behind `AnswerModel`; deterministic tests use a fake and
assert that hidden labels never cross the boundary. Invalid JSON, invented
record citations, extra output keys, cross-scope memories, stale deleted
memories, unknown memory schemas, or unpinned provenance abort the run rather
than silently producing a scoreable row.

Scoreable SOTA runs require provider-reported answer-call token counts. The
adapter emits them in GateMem's native `output.llm_usage` shape and the report
compiler reconciles every row and the official scorer summary against the
namespaced Swarm audit. This matches Table 4's `Tok./ckpt` accounting: answer
generation input plus output tokens only. Judge, embedding, and memory-ingest
calls are excluded, while ingestion time remains part of GateMem's separate
wall-clock metric.
