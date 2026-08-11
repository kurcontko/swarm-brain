# E6c v2 merged-lane confirmation: execution-control amendment

Status: frozen before any E6c extraction, merged ranking, M20 selection,
context-gate outcome, or reader/judge call. Outcome-blind R0 dense scoring was
already in progress, but no dense score or aggregate outcome was inspected.

Protocol version:
`swarmbrain-longmemeval-e6c-merged-lane-confirmation-v2-control-corrected`

## Inherited scientific design

This amendment inherits the cohort, arms, models, prompts, statistical tests,
thresholds, sample size, stopping rule, cost boundary, and claim boundary from
the frozen E6c v1 protocol byte-for-byte. It changes execution controls only.

The inherited evidence is bound by these identities:

- E6c v1 protocol file SHA-256:
  `ae12821691167c1d8f3a7082396178efa72bf3e256b97f4cd4cecdd66b4ebed0`.
- E6c v1 runner file SHA-256:
  `b7dd361ffaf8a063a45342f81f59137f08cfe4b08d851e4fe99b4176b3c69339`.
- E6c v1 manifest artifact SHA-256:
  `cf0e7326336d064ff84aad442c57ae0dddb2ca93e144a19238a5ba4313449d13`.
- Auxiliary dense manifest artifact SHA-256:
  `ce61a9343f3023e25d8ab72a19c2efc6598e452e93d60b37d04ee43e99564209`.

The v1 output namespace is control-only and receives no external calls. The v2
run has a new manifest and output namespace. It may reuse dense artifacts only
after exact replay against the inherited auxiliary dense manifest.

## Reason for the amendment

A pre-API read-only audit found five execution-control defects in the v1
runner: G0 was compiled only after QA; journal integrity was checked only in
aggregate; partial QA could dead-end before WAL reconciliation; an interrupted
two-file pack commit could not recover; and a premature report could seal an
incomplete state while omitting QA-completion and exact WAL bindings.

These are implementation-control defects, not observed experimental outcomes.
The v1 namespace is retired without a quality verdict. No arm, endpoint,
metric, threshold, seed, sample, or claim changes in v2.

## Corrected pre-QA gate

After pack replay and before the first reader or judge call, v2 must seal a
`pre-qa-gates.json` artifact. It binds the v2 manifest, every dense,
extraction, ranking, selection, pack, prompt, and context-case artifact, the
context diagnostic, and both G0 and G1.

G0 additionally reconstructs the expected extraction route set from domain
evidence. Every expected route must map one-to-one to exactly one reservation,
raw response WAL, and settlement. Request bytes, response bytes, attempts,
latency, exact local prompt tokens, maximum output tokens, conservative
reservation, settled cost, and provider request-ID uniqueness are replayed.
The exact reservation, response, and settlement artifact-hash lists are bound
into the pre-QA gate. Ordered aggregate hashes also bind every per-value
evidence JSON file and every application-attempt JSONL ledger, with exact file
counts, byte counts, and no extra sidecars. There must be zero unresolved
reservations.

Reader/judge calls are permitted only when the sealed pre-QA artifact says both
G0 and G1 passed. A G1 outcome failure rejects M20 without QA. An integrity or
provider failure leaves the run incomplete and has no quality interpretation.

## Recovery semantics

The prompt and pack files are a deterministic pair. If interruption leaves one
side only, v2 recomputes both from sealed selection evidence, verifies the
existing side byte-for-byte at the object level, and writes only the missing
side. Divergent retained evidence is never repaired or overwritten.

Before any QA call, v2 writes the passing pre-QA gate durably. If QA state later
exists, resume loads that gate first and enters the carrier WAL reconciler
directly. It does not rebuild aggregate context cases from partial QA state.
A response WAL may be settled and converted to its receipt exactly once. A
reservation with no response remains unresolved and is not reissued.

The all-phase driver skips context-case recompilation when partial QA state is
present and resumes QA directly. Complete per-question QA then produces the
single completion artifact and the final QA-aware diagnostic. A retained
completion artifact is validated before resume and is forbidden unless all 160
per-question QA artifacts already exist; a missing completion artifact after
all QA cases is reconstructed only after their exact replay. This completed-QA
finalization path is offline and does not require an API key or provider
availability because it cannot issue a call. Partial QA still requires the
frozen endpoint and credentials.

## Corrected finalization

`report.json` is terminal and may be emitted only in one of two states:

1. G0 passed and G1 failed with no durable QA state; or
2. G0 and G1 passed and all 160 three-arm QA cases, receipts, completion
   metadata, and QA journal routes replay exactly, allowing G2 to be decided.

No report is emitted for partial, provider-error, unresolved-reservation, or
integrity-failure states. Such states are resumable/incomplete, never mapped to
`reject-M20-without-posthoc-switch`.

Final G0 repeats the one-to-one extraction validation and, when QA ran, builds
the expected QA route set from every receipt. It verifies receipt identity and
role, request/response bytes, accounting, unique provider request IDs, and
exact route coverage. The report binds the pre-QA gate, QA completion when
required, and every reservation, response-WAL, and settlement artifact hash.

All scientific decisions and the allowed claim remain exactly those in E6c v1.
