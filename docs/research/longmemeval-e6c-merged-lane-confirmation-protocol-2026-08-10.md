# LongMemEval E6c merged-lane confirmation protocol

Freeze date: 2026-08-10  
Protocol: `swarmbrain-longmemeval-e6c-merged-lane-confirmation-v1`  
Status: preregistered; implementation and cohort audit complete; no E6c
extraction, retrieval, packing, gold-context, reader, or judge outcome has
been generated

This protocol freezes one answer-evidence-disjoint, fresh-question-composition
confirmation of a policy selected post hoc on E6b. It authorizes DeepSeek V4
Flash source-only extraction and conditional paired development QA. It permits
zero official GPT-4o calls and authorizes no serving, official-score, paper
reproduction, cross-corpus, or SOTA claim.

## Development disclosure and decision question

E6b was a 160-question development experiment. Equal-family raw plus
merged-SFK RRF improved answer-session coverage and reduced prompt tokens, but
its MRR fell from `0.9548128655` to `0.9526666667`; its preregistered G1
therefore rejected it and no QA ran.

After that rejection, a sealed, no-new-model-call sweep compared 56
specifications: two controls and 54 alternatives spanning raw weights, RRF
constants, raw anchors, raw-prefix quotas, merged depths, and fallback gates.
Six candidates were repacked exactly. The strongest development policy was:

> Rank only the complete merged-SFK navigation-key family, take its first 20
> keys in lane order, and hydrate the 20 bound immutable raw values. Do not
> fuse or query the raw lane.

On E6b this post-hoc policy had MRR `0.972017544`, any/all/answer-session recall
`1.0 / 0.98 / 0.991444444`, 686,326 total prompt tokens, p95 `6316.5`, and no
whole-turn drops. It beat equal RRF in eight MRR cases and lost in two; that
win/loss evidence alone was weak and multiplicity was substantial. These are
development-selection facts, not confirmation evidence.

E6c asks whether this one frozen policy, `M20`, improves delivered-context
first-gold ranking over both raw retrieval and equal RRF on a fresh
answer-evidence-disjoint LongMemEval cohort, without losing observed coverage
or delivered-token efficiency, and—only if that context gate passes—whether it
improves paired answer correctness under a frozen DeepSeek reader/judge
protocol.

M20 changes both membership and order. E6c is not a same-set reranker test and
cannot support a learned-reranker claim.

## Frozen source, models, and runtime

- Dataset: `/private/tmp/longmemeval_s_cleaned.json`.
- Dataset SHA-256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Records: 500 in source order; positions are zero based.
- E6c output:
  `/private/tmp/swarmbrain-longmemeval-e6c-merged-n160-v1`.
- Auxiliary raw-dense output:
  `/private/tmp/swarmbrain-longmemeval-e1-e6c-n160-v1`.
- Canonical value: immutable F0 raw-turn projection.
- Dense scorer: `Qwen/Qwen3-Embedding-0.6B`, revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, exhaustive question-local
  cosine ranking using the existing query instruction, last-valid-token
  pooling, float32 L2 normalization, right truncation at 8,192 model tokens,
  and deterministic score/key-ID ordering.
- Local runtime: CPython `3.12.13`, Torch `2.7.1`, Transformers `4.55.4`, MPS,
  float16 model weights, maximum Qwen batch size 8.
- Extractor, reader, and development judge: API alias
  `deepseek-v4-flash`, thinking disabled, temperature `0.0`, official DeepSeek
  `/v1/chat/completions` endpoint. This mutable alias is not an authenticated
  weight snapshot.
- Exact DeepSeek tokenizer artifact SHA-256:
  `b61454900c793d7565e0467616cadd25ff69f3c1c17493d91627c7727ea70807`.
- Complete prompt budget: 8,192 exact chat tokens.
- Official GPT-4o calls: **zero**.

The run manifest binds the protocol, implementation files, model snapshots,
tokenizer executable, prompts, dataset, cohort, output namespaces, retry
rules, prices, and decision gates before the first outcome-producing phase.
Any change creates a new protocol and namespace.

## Fresh cohort and leakage exclusion

The original E6 pilot used ten positions. E6b used 160 other positions. Their
union contains 170 unique questions and has canonical position-list SHA-256:

`4e0eeb7d01bb481c20ae89cf2fb481fac3fce26b4a38160df4fffc7f33cd02f5`.

LongMemEval reuses source content across question records. Before selecting
E6c, an outcome-blind source audit therefore excludes any otherwise unused
question whose answer evidence overlaps those 170 questions by at least one
of four exact identities:

1. answer-session ID;
2. canonical hash of `{parent_session_id,parent_session_date,turns}`;
3. canonical hash of the complete answer-session turn array; or
4. canonical hash of any raw answer-session turn.

The audit excludes 30 additional positions:

```text
26,42,65,69,126,130,131,175,203,230,236,238,242,256,286,
290,292,294,297,300,302,304,328,357,358,362,373,376,416,439
```

Their position-list SHA-256 is
`7ef254d9d2ab49456196e1c6687d7452d204e2ff5308acbc70710da7e882a903`.
The full excluded-200 list SHA-256 is
`1412f422f86135fa10a471d80b6ff14ff869e3fb31790c8f5a55414ac7c1374a`;
the eligible-300 list SHA-256 is
`a28999daecd53849d8882c06115b20b997349aad8637304330915324cc6b74eb`.

Canonical JSON uses UTF-8, `ensure_ascii=False`, sorted object keys,
separators `(",", ":")`, `allow_nan=False`, no Unicode normalization, and
lowercase SHA-256.

Under CPython `3.12.13`, seeds are inspected from `20260811` upward. For each
seed:

```python
tuple(sorted(random.Random(seed).sample(eligible_positions, 160)))
```

qualifies only if it has exactly 25 knowledge-update, 42 multi-session, 18
single-session-assistant, 9 single-session-preference, 22 single-session-user,
44 temporal-reasoning, and 9 `_abs` questions. The first qualifying seed is
`20564941`, after 304,131 candidates. No model output, query, answer text,
retrieval score, or judge label was used.

The canonical positions are:

```text
1,6,14,16,21,22,23,24,28,33,34,35,37,41,43,45,49,50,51,63,
67,68,70,74,76,81,82,84,87,88,94,95,101,102,108,109,111,112,
115,119,129,133,134,141,144,148,152,157,158,161,162,163,165,
171,172,174,176,178,182,184,187,188,190,191,192,196,197,202,
204,212,215,220,232,233,234,244,245,246,247,248,252,255,257,
262,268,272,273,279,281,282,287,293,306,307,308,309,310,311,
314,316,318,320,321,322,324,327,335,336,338,342,344,346,348,
350,352,360,361,366,372,375,378,381,382,385,386,390,392,398,
402,410,414,415,418,421,426,427,428,430,435,437,438,441,447,
448,453,455,457,458,465,467,470,472,477,482,489,491,494,497,
498,499
```

The cohort has 7,599 sessions, 78,648 raw turns, 307 answer sessions, and
3,540 answer turns. Core seals are:

| Binding | SHA-256 |
| --- | --- |
| Full selector object | `3c75d59f98e42b27b18baf7ecf3a627b96a5d80cb0ac48a21b35706f7aeeb8e8` |
| Selected positions | `f8873da961ccbc46eec787290d2e04b57d8a40e2ba092f07fd1fd4d2d8f90907` |
| Ordered question IDs | `79c03e2559eb8b9fb373ad9f05e353038cc403447d234f83c53fd46329488507` |
| Full run rows | `47a6d96762c2f347c70b316fc56f6388376f4a886d4777320dd69cc332a15eb5` |
| Source binding rows | `b1c31180922c871af172e63a76e5aff467cb4ce1e026915c483e35a7e6b3db5b` |
| Selected history fingerprints | `d0cd9404671142176cf24c9393011cdaf393dfcc0967df10ab250c4de0b1baa2` |
| Canonical selected-record array | `0f2654a2a90f67ce6fb87099e7a81ca720fe8f598312e21081d998d1cf9d8822` |

### Independence limitation

All 500 complete question-local history fingerprints are distinct, and the
selected answer evidence has zero overlap with development under all four
rules above. Nevertheless, LongMemEval inherits a shared distractor pool.
Every eligible and selected question shares some non-answer distractor content
with development. The selected cohort shares 992 distractor session IDs or
ID-plus-turn identities, 1,089 distractor turn arrays, and 11,309 distractor
turn hashes. Date-bearing serialized documents remain distinct because dates
shift.

E6c is therefore an **answer-evidence-disjoint fresh same-benchmark cohort
with inherited distractor-pool reuse**, not a corpus-independent or external
held-out benchmark. The fixed type/abstention mixture is also not a
proportional sample of the leakage-filtered eligible 300.

## Frozen arms

### R0: raw top 20

1. Score every canonical raw turn against the question using pinned Qwen.
2. Sort by cosine descending and the existing deterministic canonical tie
   rule.
3. Take exactly 20 unique canonical raw values.

### R1: equal-family RRF top 20

1. Create exactly one source-only merged summary/fact/keyword navigation key
   per raw value with DeepSeek.
2. Score complete raw and merged-SFK indexes separately with pinned Qwen.
3. Retain family depth 20, deduplicate within family by best key rank, and fuse
   with equal weights and RRF `k=60`.
4. Take the first 20 fused canonical values.

### M20: merged lane top 20

1. Use the same complete merged-SFK index and scores as R1.
2. Do not query or fuse the raw lane for this policy.
3. Take merged-SFK key ranks 1–20 in exact lane order. There is exactly one
   merged key per source value, so this yields 20 unique values without
   backfill.
4. Hydrate each key to its bound byte-identical raw value.

All reader context contains raw turns only. Derived text is navigation
metadata and never reader evidence. Gold answer, answer-session IDs, question
type, abstention status, and QA outcomes are unavailable to extraction,
ranking, selection, and packing.

## Extraction, ranking, packing, and carrier disclosure

Every one of the 78,648 source values receives a source-only DeepSeek
extraction application. The exact request, first 2xx response, provider usage,
request ID hash, latency, invalid application attempts, retries, and pricing
settlement are durably retained. There are at most three application-schema
attempts and four HTTP attempts per application; a valid response stops that
route. Extraction concurrency is 24.

Raw and merged ranking are fresh exhaustive local scans. R0/R1 carrier
functions and the durable external-call journal are imported from the frozen,
tested E6b runner without editing its file. They run under a new E6c protocol,
manifest, and output namespace. Some low-level carrier artifact-type strings
retain an `e6b` schema prefix; the protocol version and manifest binding are
authoritative, and E6c emits distinct selection, pack, case, diagnostic, QA
completion, and final-report artifacts. This reuse is explicitly fingerprinted
and is not a claim that E6b itself continued or changed.

Each arm is independently packed by the unchanged whole-turn packer. The
complete reader prompt is counted with the exact pinned tokenizer. Skipping a
turn because it would exceed 8,192 tokens is a whole-turn drop; truncating or
editing a raw turn is forbidden.

## G0 integrity gate

G0 passes only if all 160 questions have byte-replayable dense, extraction,
ranking, selection, pack, and case artifacts; every source value is covered;
all request/response WAL and usage settlements reconcile; every local model,
tokenizer, prompt, sample, implementation, and manifest binding matches; and
there are zero unresolved external-call reservations. A partial or failed
route remains in the denominator and makes the run incomplete, not a quality
failure or exclusion.

## G1 confirmatory context gate

There are exactly 151 gold-eligible non-abstention questions. The primary
ranking outcome is delivered-prompt answer-session MRR: relevance units are
official answer-session IDs, ranked units are raw turns, and the first raw turn
whose parent session is gold determines reciprocal rank. Candidate and prompt
MRR must coincide because every arm must deliver all 20 candidates.

For each paired M20-minus-comparator MRR delta, use a question-type-stratified
paired percentile bootstrap with 100,000 resamples, seed `20260810`, and the
one-sided 95% lower bound (fifth percentile). The bootstrap unit is the
question-local history. G1 is a conjunctive intersection-union gate and passes
only if all conditions hold:

1. M20's paired MRR lower bound is strictly above zero versus R0.
2. M20's paired MRR lower bound is strictly above zero versus R1.
3. Observed M20 any-gold, all-gold, and answer-session recall are each at least
   R1. These are cohort-preservation safeguards, not population-level
   noninferiority intervals.
4. M20 total and fixed linear-interpolation p95 complete-prompt tokens are each
   no greater than both R0 and R1.
5. Every arm/question selects exactly 20 unique canonical values, delivers all
   20 complete raw values in the same order, stays within 8,192 exact tokens,
   and has zero whole-turn drops.

Average precision and nDCG over gold sessions, deduplicating sessions at first
occurrence, are frozen secondary diagnostics and cannot rescue a failed gate.
Because success requires every component, no comparator is chosen post hoc;
disjunctive endpoint claims require multiplicity correction or remain
descriptive.

If G1 fails, reject M20. Do not run QA, change margins, extend the sample, or
switch post hoc to a raw anchor, quota, reranker, or other screened policy.

## Conditional G2 paired DeepSeek QA gate

Only after G0 and G1 pass, run all three arms on all 160 questions. Each arm
gets one DeepSeek V4 Flash reader answer and one development-judge call. Judge
prompts contain the public question, reference answer, and one hypothesis but
not the arm identity. Within each question-type stratum, the six permutations
of R0/R1/M20 execution order cycle deterministically with counts differing by
at most one.

For M20 versus each of R0 and R1, compute the same frozen one-sided,
question-type-stratified paired bootstrap over per-question Boolean
correctness deltas. G2 passes only if, against **both** comparators:

1. the one-sided 95% lower bound is strictly above zero;
2. the observed accuracy gain is at least 0.02;
3. no question-type observed delta is below -0.05; and
4. the nine-question abstention subgroup has nonnegative observed delta.

All 160 cases remain intention-to-treat. A missing, malformed, or
non-replayable response makes the run incomplete; there are no post-outcome
exclusions or optional reruns. Reader and judge use the same mutable DeepSeek
alias, so a passing result means improvement only under this frozen
DeepSeek-based evaluation protocol.

At n=160, the two-comparator lower-bound gate has limited power for a two-point
effect; for example, a small number of discordant wins may still yield a zero
lower bound. An inconclusive result remains a rejection under this protocol.
No margin or sample change is allowed after outcomes.

## Cost and execution boundary

The operator authorized DeepSeek use until the provider account returns a
balance or transport error and explicitly requested that expected cost not be
an optimization constraint. A high `$100` engineering ledger ceiling exists
only to catch runaway route generation; the frozen finite route count is the
real scope. Every successful or conservatively unseen attempt is still
accounted in integer micro-USD. A provider balance error is an incomplete run,
not evidence; an unresolved durable reservation cannot be silently reissued.

Execution order is:

1. freeze manifests and verify cohort/runtime;
2. stage fresh exhaustive R0 dense scores;
3. complete source-only extraction;
4. rank raw and merged indexes;
5. materialize the three frozen selections;
6. exact-pack all arms;
7. compile G0/G1 without QA;
8. conditionally run three-arm DeepSeek reader/judge QA;
9. replay all evidence and emit the final report.

No statistical interim, optional stopping, seed replacement, route removal,
or outcome-dependent repair is permitted.

## Claim boundary and next experiment

If G0–G2 pass, the strongest allowed claim is:

> On an answer-evidence-disjoint fresh LongMemEval same-benchmark cohort with
> inherited distractor-pool reuse, merged-SFK-only retrieval improved
> first-gold ranking over raw-only and equal-RRF retrieval, preserved observed
> coverage, used no more delivered context, and improved paired answer
> correctness under the frozen DeepSeek V4 Flash evaluation protocol.

It does not establish official LongMemEval performance, independent-corpus or
model-independent generalization, paper reproduction, or SOTA. A pass freezes
M20 unchanged and authorizes the next experiment: exact-protocol external
confirmation on a genuinely different corpus such as BEAM/LIGHT. A failure
rejects M20 without a post-hoc replacement.
