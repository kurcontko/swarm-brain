# Paired LongMemEval selection QA evidence

This package compiles the frozen full-500 LongMemEval-S E1, E2, E6, and E7
selection experiments after an external runner has produced both answers and
binary judge labels. It is deliberately offline: importing or running it does
not call a reader, judge, tokenizer, model, database, or network service.

The comparison boundary is fixed at an 8,192-token **full reader prompt**.
Whole turns are indivisible and packing must skip an oversized turn and
continue. Baseline and candidate use exactly one prompt serializer/tokenizer,
reader identity/template, and judge identity/template. Those identities and
every source artifact are SHA-256 bound by the run manifest.

Protocol `swarmbrain-longmemeval-selection-qa-paired-v2` registers exact E2,
E6, and E7 input profiles. E7 evidence adds fields to case schema v2 and
constructor accounting to report schema v2. Because the registry is embedded
in the run's frozen protocol, older manifests and case rows are intentionally
not silently reinterpreted.

## Evidence boundary

`build_run_manifest(...)` accepts only artifact-root-relative inputs and binds
the exact dataset and LF-only, newline-terminated JSONL case artifact. The case
schema omits explicit question, context, answer, reader-prompt, and
judge-prompt text fields. It still carries externally supplied identifiers, so
the compiler does not claim that arbitrary identifier metadata is semantically
content-free. The compiler:

- requires exact dataset-order coverage with one baseline and candidate cell
  for every question;
- recomputes dataset record, question, reference-answer, source-corpus,
  candidate-pool, reader-material, and judge-material digests;
- accepts only canonical question-local session IDs or immutable F0 turn IDs;
- derives any/all-gold and answer-session MRR after selection from the pinned
  dataset, so gold IDs never enter the selection envelope;
- requires each cell identity to bind one exact ordered E1, E2, E6, or E7 input
  profile. E1-A permits only common routing/query/canonical-turn fields plus
  attested raw lexical and dense scores; E1-B replaces those scores with the
  derived E1-A RRF score and attested CrossEncoder logit; E1-C adds attested
  raw ColBERT score; E1-D adds deterministic threshold metadata. E2-A..E2-E
  bind the fixed E1-A top-20 source, query-turn cosines, and only the
  context-cosine and deterministic organization policies each cell consumes.
  E6/R0..R5 and R-neg bind exactly their registered source-only key families,
  query-time raw scores, construction receipts, and graph/policy inputs. E7-A,
  E7-B, and E7-C bind the E1-B top-50 selection trace, official-preflight
  manifest, query-window batch trace, normalized constructor-receipt set,
  rendered reader-context digest, and the cell's grounded-versus-abstractive
  claim;
- rejects duplicate/replayed request IDs and provider request IDs, identity
  drift, non-finite accounting, partial pairs, and artifact-byte changes.

Every E7 row carries an exact `protocol_evidence` object and a digest over that
object. The compiler reconciles its source E1-B trace with the selection input
artifact, its reader-context digest with the context delivered to the reader,
and its normalized receipt count with constructor calls. All E7 rows and arms
must bind one run-wide preflight manifest. Source-selection, window-batch, and
non-empty receipt-set digests are question-bound: cross-question replay is
rejected, while two E7 arms for the same question must agree on their shared
source and window construction. E7-A must bind the empty receipt list and the
raw-source grounding claim. E7-B/C must bind at least one unsigned normalized
constructor receipt. E7-C accepts only the byte-grounded claim; it cannot be
relabeled as abstractive construction.

The selection input, preflight, window-batch, and normalized-receipt artifacts
are represented by digests rather than reopened. Consequently, the compiler
proves only that the declared selection fields exactly equal a registered
profile, their content-free bindings are internally consistent, and
`gold_fields_used` is false. It records that absence of hidden gold use,
constructor provenance, and semantic faithfulness are not cryptographically
proven.

The profiles make E2, E6, and E7 rows structurally admissible instead of
forcing them to masquerade as an E1 cell. They do not by themselves prove a
runnable prompt. Before any reader call, the separate official preflight must
reopen the F0 corpus and validate the exact prompt-packing trace, tokenizer
identity, 8,192-token budget, candidate order, kept IDs, and final prompt
bytes. In particular, the v1 turn packer rejects duplicate turn IDs:
parity-style E2-B, E2-C, and E2-D outputs with cross-chain reuse cannot proceed
under that packer; E2-E is the registered deduplicated rendering cell. An
accepted input-profile name or preflight digest is never a substitute for
running and preserving that preflight.

Each reader/judge row includes a deterministic receipt-envelope digest over
case, arm, stage, identities, both request IDs, request/response hashes,
answer/label outcome, and exact stage accounting. These envelope digests and
both request-ID namespaces are globally unique. The underlying receipt bytes
remain unopened and `externally-attested-unsigned`; this compiler cannot
authenticate which weights an external provider served.

## Statistics and promotion

QA confidence uses 10,000 paired bootstrap draws with seed `20260809`. Each
draw resamples independently *within every question-type stratum* while
preserving all stratum sizes. The report contains overall and per-type raw
correct counts, QA deltas, any/all-gold, answer-session MRR, exact prompt-token
summaries, operational/end-to-end latency, and stage-level call/token/cost
totals.

Primary-track eligibility requires all frozen rules: QA CI lower bound `> 0`,
no type's net accuracy regression greater than 2 percentage points, any-gold
noninferiority with margin exactly zero, complete metric reporting, and no
baseline Pareto dominance on QA, p95 prompt tokens, p95 operational latency,
and total construction-plus-query cost. Construction-plus-query cost is
strictly embedding plus reranker plus constructor cost. Constructor call,
input-token, output-token, cost, and stage-latency accounting is
reported as its own stage; constructor latency is included in operational and
end-to-end latency. Reader cost is reported separately and is included in
total cost, but cannot move the construction-plus-query Pareto axis.

E6/R-neg remains a negative control: its profile can be compiled for analysis,
but the report marks it intrinsically ineligible for structural policy
eligibility regardless of its measured metrics.

E7-B is also intrinsically ineligible under protocol v2. Its allowed
abstractive output carries source spans but the E7 constructor does not prove
semantic faithfulness, and this compiler has no authenticated faithfulness
verifier. A caller-supplied faithfulness digest is rejected rather than treated
as authority; admitting authenticated faithfulness requires a new protocol
with a verifying trust boundary. E7-C is intrinsically admissible because its
output is byte-grounded, but receives no shortcut: it must pass the same
canonical-primary, QA, regression, context, efficiency/Pareto, and held-out
gates as every other eligible candidate. E7-A is the raw control and likewise
passes only through the existing gates.

That is still insufficient for even structural policy eligibility. The
compiler requires `--heldout-confirmation` to point to a separate confirmation
whose preregistration predates its results, pins a dataset digest different
from LongMemEval-S, binds the same frozen cell/model/prompt identities, and
binds complete per-question paired binary-label evidence. The compiler derives
the held-out QA direction from those rows; it does not accept an
`independent=true` or `benefit=true` assertion.

That check establishes structural integrity only. The separate
dataset digest, preregistration timestamp, receipt digests, and labels are
still externally attested and unsigned: their provenance and chronology are
not cryptographically authenticated, and byte distinction does not prove
semantic independence. Noncanonical/custom primary datasets can never set
`structural_offline_policy_eligible`. `serving_promotion_eligible` is always
false: authenticated execution evidence and explicit serving approval remain
mandatory. No compiler result is an empirical SOTA claim.

```bash
uv run --extra dev python scripts/build_longmemeval_selection_report.py \
  --run benchmarks/selection/selection-qa-run.json \
  --output benchmarks/selection/selection-qa-report.json
```

Omitting held-out confirmation is valid for analysis. It leaves structural
eligibility false; serving eligibility is false in every case.
