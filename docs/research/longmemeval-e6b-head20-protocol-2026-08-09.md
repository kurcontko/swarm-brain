# LongMemEval E6b head-20 development protocol

Freeze date: 2026-08-10  
Protocol: `swarmbrain-longmemeval-e6b-head20-development-v1`  
Status: preregistered; deterministic local dense staging in progress; no external calls executed

This document freezes one larger development comparison of raw retrieval
against source-preserving merged representation keys. It authorizes no serving
change and no SOTA, official LongMemEval, paper-reproduction, causal-production,
or held-out claim.

## Decision question

E6 v2 moved answer-session MRR from `0.9083` to `0.9167`, but its two arms tied
`9/10` on paired DeepSeek development QA. Full R1 also delivered 12,786 more
prompt tokens because its two family heads produced 24--32 hydrated values,
versus exactly 20 for R0.

A read-only sensitivity check truncated each existing R1 fused order to its
first 20 hydrated values. It preserved every reported context-quality metric,
including R1 MRR `0.9166666666666666`, while reducing exact complete prompt
tokens from R0's `47,595` to `46,710` (`-885`, or `-1.86%`). Those ten cases
motivated this protocol and are excluded from its sample and statistics.

E6b asks one question: **does the R1 ranking signal produce a reproducible
paired QA improvement when both arms receive exactly 20 candidate values?**
Retrieval-only improvement, an accuracy tie, or an efficiency regression is
not enough.

The paper-derived rationale remains narrow:

- UnifiedMem supports raw plus independently indexed derived navigation keys,
  but also shows that retrieval recall and answer usefulness can diverge.
- Memora supports separating navigation keys from high-fidelity values.
- LazyMem makes delivered context size part of the quality frontier and warns
  that editing or adding context after successful retrieval can introduce the
  remaining errors.

E6b therefore scales the one retained E6 mechanism while removing its measured
candidate-count confound. It does not introduce R2 keys, graphs, query-time
construction, learned routing, consolidation, or a reranker.

## Frozen source and identities

- Dataset: `/private/tmp/longmemeval_s_cleaned.json`.
- E6b output namespace:
  `/private/tmp/swarmbrain-longmemeval-e6b-head20-n160-v1`.
- Dataset SHA-256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Dataset records: 500, with zero-based dataset positions.
- Canonical value: the existing immutable F0 turn projection.
- Representation contract: `E6/SB-HMR-v1` with a new run-level head-20
  protocol. Derived keys remain navigation metadata and never become reader
  evidence.
- Dense scorer: `Qwen/Qwen3-Embedding-0.6B` at revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, using the exact E6-v2 query,
  pooling, truncation, cosine, and runtime contracts. Both raw and derived-key
  scores are computed fresh for these 160 questions. Raw scores are staged by
  rerunning the frozen E1-v3 dense phase into the new
  `/private/tmp/swarmbrain-longmemeval-e1-e6b-n160-v1` namespace and are then
  replayed as E6b source artifacts; no prior pilot's dense artifacts are reused.
  Derived-key scores are computed in the E6b ranking phase.
- Extractor, reader, and development judge: `deepseek-v4-flash`, thinking
  disabled and temperature `0.0`, through the official DeepSeek
  `/v1/chat/completions` endpoint. This is a mutable provider alias, not an
  authenticated weight snapshot.
- Exact tokenizer artifact SHA-256:
  `b61454900c793d7565e0467616cadd25ff69f3c1c17493d91627c7727ea70807`.
- Source E6-v2 report SHA-256:
  `c56dd9ff90c881f52416c53be52e654018adb29cbcc8f847fe92eaf34933bb49`.
- Source E6-v2 run-manifest SHA-256:
  `ab6a7a5d943ac46c0b423c7bb71c35fedcdc45f44a54ea9bb453b20f6a99ba1b`.
- GPT-4o calls permitted: **zero**. No official judge is run under this
  protocol.

The new run manifest must bind the implementation tree, this protocol file,
all local model snapshots, prompts, tokenizer, prices, selected-question
binding, run order, and output namespace before the first external call. A
changed file, model, prompt, cap, sample, retry rule, or output namespace is a
new protocol, not a continuation.

## Arms

### R0: raw head 20

1. Score every canonical raw turn against the question with the pinned Qwen
   scorer.
2. Order by the existing deterministic raw-family ranking contract.
3. Take exactly the first 20 canonical values before packing.
4. Hydrate their byte-identical raw values only.

### R1H20: raw plus merged-SFK head 20

1. For every canonical value, make exactly one source-only DeepSeek extraction
   call under the E6 merged summary/fact/keyword prompt and evidence contract.
   The extractor receives the source value only; question text, question type,
   answer, answer sessions, and judge material are forbidden.
2. Score the complete raw and merged-SFK indexes separately with the same
   pinned Qwen scorer.
3. Retain depth 20 within each family, deduplicate values within family, and
   fuse the two family heads using equal-family RRF with `k=60` and the existing
   deterministic tie break.
4. **Truncate the fused hydrated-value order to its first 20 values before
   packing.** No 21st or later fused value may influence packing or QA.
5. Hydrate the same byte-identical raw value type as R0. Merged-SFK text is
   never delivered to the reader.

Both arms use the unchanged whole-turn packer and exact complete-prompt ceiling
of 8,192 DeepSeek chat tokens. Candidate order, skip-over-budget behavior,
reader prompt, reader maximum output, development-judge prompt, judge maximum
output, and response validation remain identical to E6 v2. Reader-arm order is
counterbalanced by zero-based run position: even positions execute R0 first;
odd positions execute R1H20 first.

Gold answers, answer-session IDs, question type, abstention status, and judge
labels are unavailable to extraction, scoring, fusion, hydration, and packing.
They may be opened only by the post-hoc context compiler and development judge.

## Fresh sample and exclusion

The original E6 development questions are excluded by zero-based dataset
position before selection:

| Position | Question ID | Type | Abstention |
| ---: | --- | --- | :---: |
| 17 | `ad7109d1` | single-session-user | no |
| 29 | `37d43f65` | single-session-user | no |
| 160 | `1c0ddc50` | single-session-preference | no |
| 169 | `1f2b8d4f` | multi-session | no |
| 185 | `a1cc6108` | multi-session | no |
| 221 | `37f165cf` | multi-session | no |
| 228 | `6456829e_abs` | multi-session | yes |
| 394 | `01493427` | knowledge-update | no |
| 422 | `cf22b7bf` | knowledge-update | no |
| 478 | `e982271f` | single-session-assistant | no |

The canonical JSON exclusion-row SHA-256 is
`468b2ce21fa3885b4a1e17cb15e19299b7ecb8f184431f25c649b182edb87ff2`.
Each row has exactly `abstention`, `position`, `question_id`, and
`question_type`, and rows are ordered by position.

Among the remaining 490 records, proportional largest-remainder targets for
160 questions are 22 single-session-user, 42 multi-session, 10
single-session-preference, 43 temporal-reasoning, 25 knowledge-update, and 18
single-session-assistant. The abstention target is fixed separately at 10 of
the 29 remaining official `_abs` records.

The realized sample differs from the type target by at most one case:

| Question type | Remaining | Target | Selected |
| --- | ---: | ---: | ---: |
| single-session-user | 68 | 22 | 22 |
| multi-session | 129 | 42 | 42 |
| single-session-preference | 29 | 10 | 10 |
| temporal-reasoning | 133 | 43 | 42 |
| knowledge-update | 76 | 25 | 25 |
| single-session-assistant | 55 | 18 | 19 |
| **Total** | **490** | **160** | **160** |

The selected sample contains exactly 10 abstention questions and 150
non-abstention questions. It contains 79,130 canonical source turns. "Fresh"
means only that none of these records appeared in the motivating ten-question
pilots. LongMemEval has already influenced the design, so this is a larger
development sample, not a separately held-out benchmark or corpus.

## Deterministic selector

Selection is frozen to CPython `3.12.13` and the existing semantics of:

```python
tuple(sorted(random.Random(seed).sample(range(500), 160)))
```

The seed was chosen without reading questions, answers, answer-session IDs,
turn text, retrieval scores, or model outcomes. Starting at integer seed
`20260810`, inspect seeds in increasing order and accept the first candidate
that satisfies all three metadata-only conditions:

1. it is disjoint from the ten excluded positions;
2. it contains exactly ten question IDs for which the official rule
   `"_abs" in question_id` is true; and
3. every realized main question-type count differs from its target above by
   at most one.

The first qualifying seed is **`20282059`**. The sorted positions are the
canonical run order. There is no resampling, replacement, type-specific
routing, or outcome-dependent substitution.

All digests below use UTF-8 JSON with `ensure_ascii=False`, sorted object keys,
separators `(",", ":")`, `allow_nan=False`, and lowercase SHA-256, matching
`canonical_json_bytes` in the E6 representation contracts.

| Binding | SHA-256 |
| --- | --- |
| Canonical list of 160 positions | `13188de9750669fc53e8704dbefa60d4f30b4dff9992edd0d1c9053b03ae7b95` |
| `{protocol,total,sample,seed,positions}` selector binding | `dfdd07fe82288be055b0e6359cad5cc0192a3663b17202c9b9997e628f6353d4` |
| Ordered `{position,question_id}` rows | `41a7d73a546b768606fd50b655c53e2d2b9ed570a16e15f3719619183342dd03` |
| Ordered question-ID list | `0047fd256b16dcdc424957508f227149c095932ec38fafbb3f8037f674b16b34` |
| Runner ordered-question binding | `e8f39c354b4a1773fbf194d2547227f3730a86c8e05e351c099ece36def1c84b` |
| Full run rows shown below | `0da325831620039dd2cd96a20aa56f0b80bc3c4a5fae7b2d6f6e44eca872786a` |
| First-40 run rows | `a7fa2bfa0c61c25773f106cf3abc550c07b1efeec1797013adb3b4d95c73521d` |

For the selector-binding digest, `protocol` is
`python-3.12-random-sample-sorted-v1`, `total` is `500`, `sample` is `160`,
and `seed` is `20282059`. A full run row has exactly `abstention`, `position`,
`question_id`, `question_type`, and zero-based `run_position`.
The runner ordered-question binding has exactly `abs`, `position`,
`question_id`, and `question_type`, in sorted-position order; `abs` applies the
same official `_abs` classification.

## First-40 operational tranche

Run positions 0--39 are an operational scheduling tranche only. Because the
dataset is type-grouped and run order is sorted dataset position, this tranche
contains 22 single-session-user and 18 multi-session questions, including two
abstention questions. It is deliberately **not** a statistical interim sample.

At the tranche boundary, operators may inspect only mechanical evidence:
manifest and sample hashes, route coverage, response-WAL durability, schema
validity, tokenizer reconciliation, replay equality, unresolved reservations,
and spend-ledger consistency. They must not compile or inspect aggregate QA,
paired correctness, MRR, gold-context metrics, question-type outcomes, or a
promotion verdict.

The tranche cannot change the arms, prompts, cap, sample, ordering, retry
policy, or final gates. It cannot trigger a quality/futility stop, extend a
promising result, replace a failed case, or select a different seed. A transport
or process interruption resumes the same sealed routes. A non-replayable
mechanical failure or hard-cap exhaustion makes the run incomplete and
ineligible; it does not create a 40-question result.

## Cost and call boundary

The run has a hard external-cost ceiling of **5,600,000 integer micro-USD
(`$5.60`)**, below the approximately `$5.696313` remaining after E6 v2. The
durable global ledger must reserve pessimistically before every call, price all
input tokens as cache misses, charge all observed or conservatively unseen
retries, settle from retained provider usage, and finish with zero unresolved
reservations.

At complete coverage there are 79,130 source-only extraction applications. If
the context gate advances, there are also 320 reader and 320 development-judge
calls. Retries and invalid application attempts are additional and charged.

- Direct ten-question cost scaling gives a conservative planning estimate of
  `$4.858992`.
- Scaling extraction by this sample's exact source-turn count and QA by case
  count gives an indicative estimate of about `$4.594`; this is not a cap or
  billing claim.
- Extrapolating the separate worst observed E6-v2 per-case components remains
  below about `$5.19` for 160 cases.

The `$5.60` ledger ceiling is authoritative. If all required artifacts cannot
be completed below it, the protocol returns `incomplete-cost-cap`; no partial
quality result is eligible. Local Qwen compute has zero monetary cost in this
ledger, but its calls, input tokens, and unauthenticated local latency remain
reported separately. No GPT-4o reservation or call is permitted.

## Execution order

1. Freeze the manifest, output namespace, implementation inventory, selection
   bindings, model/tokenizer identities, pricing, and empty durable ledger.
2. Execute extraction, fresh raw/merged Qwen scoring, fusion, head-20
   truncation, and exact packing for run positions 0--39. Perform only the
   mechanical operational audit above.
3. Resume the identical phases for positions 40--159. Reopen and replay every
   case artifact from disk before compiling context metrics.
4. Apply the complete-160 integrity and context gate. If it fails, reject
   R1H20 and make no reader or judge calls.
5. If the context gate passes, execute the two DeepSeek reader arms and two
   DeepSeek development-judge calls for every case in the frozen counterbalanced
   order. Retain credential-free raw request/response bytes before parsing.
6. Reopen all 160 paired cases and all journal artifacts, compile the frozen
   statistics once, and apply the QA gate. The motivating ten E6 questions are
   never pooled into any E6b metric or interval.

## Exact gates

All gates are conjunctive and fail closed.

### G0: integrity and coverage

G0 passes only when:

- the source, exclusion, selector, question, run-order, implementation,
  tokenizer, model, prompt, and pricing bindings match the frozen manifest;
- all 160 selected cases and both head-20 representation traces replay exactly;
- every R1H20 value has exactly one valid source-only merged-SFK construction
  result and complete accounting;
- every external route has one reconciled reservation, raw-response WAL, and
  settlement chain, including all failed attempts and retries;
- no route crosses a question, source version, arm, or provider-request-ID
  namespace;
- there are zero unresolved reservations, no missing or replacement cases,
  and total external cost is at most 5,600,000 micro-USD; and
- GPT-4o call and reservation counts are both zero.

An incomplete run is not a negative quality result; it is simply ineligible.

### G1: complete-160 context and efficiency

Post-hoc gold metrics use only the 160 fresh selected cases. G1 passes only
when all of the following hold:

1. R1H20-minus-R0 mean candidate and delivered-prompt `any_gold_in_context`,
   `all_gold_in_context`, and answer-session recall are each greater than or
   equal to zero, with margin exactly `0.0`.
2. R1H20 mean answer-session MRR is strictly greater than R0 mean
   answer-session MRR for both the candidate order and delivered prompt.
3. R1H20 total exact complete-prompt tokens are less than or equal to R0 total
   exact complete-prompt tokens.
4. R1H20 p95 exact complete-prompt tokens are less than or equal to R0 p95,
   using the existing linear-interpolation percentile function.
5. Both arms bind exactly 20 pre-packing canonical candidates per case, use the
   same 8,192-token ceiling, and report any whole-turn drops caused by that
   ceiling.

G1 is evaluated only after all 160 context artifacts exist. MRR improvement
alone cannot promote R1H20; it merely authorizes the paired QA stage.

### G2: paired DeepSeek development QA

G2 passes only when all of the following hold:

1. All 160 questions have one replayable reader result and one replayable
   development-judge label for each arm.
2. The overall paired accuracy delta `R1H20 - R0` is positive and its
   stratified percentile paired-question bootstrap 95% confidence interval has
   lower bound strictly greater than `0.0`.
3. For each of the six main question types,
   `accuracy(R1H20) - accuracy(R0) >= -0.02`.
4. Across the ten selected abstention questions,
   `accuracy(R1H20) - accuracy(R0) >= 0.0`.
5. R0 does not Pareto-dominate R1H20 on development accuracy, p95 prompt
   tokens, p95 operational latency, and total construction-plus-query cost.
6. G0 and G1 remain true when recomputed from the sealed final case set.

The QA interval reuses the existing
`stratified-percentile-paired-question-bootstrap-v1` contract exactly: paired
question resampling independently within each main question-type stratum,
10,000 draws, seed `20260809`, confidence `0.95`, and the existing
linear-interpolation percentile function. Arm-order counterbalancing does not
form a bootstrap stratum. Per-type raw correct, improved, regressed, and tied
counts must accompany every percentage.

There is no post-hoc alternate judge parse, question exclusion, sample pooling,
one-sided interval, parameter sweep, or relaxed margin. Any such change is a
new experiment ID.

## Decision semantics

- Failure of G0: `incomplete-or-invalid-e6b-run`; no quality inference.
- Passage of G0 but failure of G1: `reject-R1H20-at-context-gate`; no QA spend.
- Passage of G0/G1 but failure of G2: `reject-R1H20-at-development-qa-gate`.
- Passage of all gates: `retain-R1H20-for-separate-heldout-confirmation-only`.

Even the final outcome is not eligible for composition, serving promotion, an
official LongMemEval score, or a SOTA claim. A passing result authorizes only a
separately preregistered comparison on a task or corpus that did not motivate
E1, E2, E6, or E6b. GPT-4o remains outside this protocol.

## Sealed run order

Rows 0--39 are the operational tranche. `abs` is the official
`"_abs" in question_id` classification.

| Run | Dataset position | Question ID | Type | abs |
| ---: | ---: | --- | --- | :---: |
| 0 | 0 | `e47becba` | single-session-user | no |
| 1 | 2 | `51a45a95` | single-session-user | no |
| 2 | 4 | `1e043500` | single-session-user | no |
| 3 | 7 | `6f9b354f` | single-session-user | no |
| 4 | 8 | `58ef2f1c` | single-session-user | no |
| 5 | 11 | `7527f7e2` | single-session-user | no |
| 6 | 12 | `c960da58` | single-session-user | no |
| 7 | 19 | `dccbc061` | single-session-user | no |
| 8 | 25 | `95bcc1c8` | single-session-user | no |
| 9 | 40 | `15745da0` | single-session-user | no |
| 10 | 44 | `001be529` | single-session-user | no |
| 11 | 48 | `545bd2b5` | single-session-user | no |
| 12 | 52 | `8e9d538c` | single-session-user | no |
| 13 | 54 | `c19f7a0b` | single-session-user | no |
| 14 | 55 | `4100d0a0` | single-session-user | no |
| 15 | 57 | `1faac195` | single-session-user | no |
| 16 | 58 | `faba32e5` | single-session-user | no |
| 17 | 59 | `f4f1d8a4` | single-session-user | no |
| 18 | 61 | `36580ce8` | single-session-user | no |
| 19 | 62 | `3d86fd0a` | single-session-user | no |
| 20 | 64 | `0862e8bf_abs` | single-session-user | yes |
| 21 | 66 | `bc8a6e93_abs` | single-session-user | yes |
| 22 | 71 | `6d550036` | multi-session | no |
| 23 | 79 | `dd2973ad` | multi-session | no |
| 24 | 80 | `c4a1ceb8` | multi-session | no |
| 25 | 83 | `46a3abf7` | multi-session | no |
| 26 | 86 | `gpt4_2f8be40d` | multi-session | no |
| 27 | 89 | `88432d0a` | multi-session | no |
| 28 | 90 | `80ec1f4f` | multi-session | no |
| 29 | 97 | `2318644b` | multi-session | no |
| 30 | 99 | `gpt4_d12ceb0e` | multi-session | no |
| 31 | 100 | `00ca467f` | multi-session | no |
| 32 | 103 | `eeda8a6d` | multi-session | no |
| 33 | 104 | `2788b940` | multi-session | no |
| 34 | 107 | `129d1232` | multi-session | no |
| 35 | 110 | `a9f6b44c` | multi-session | no |
| 36 | 113 | `gpt4_ab202e7f` | multi-session | no |
| 37 | 116 | `edced276` | multi-session | no |
| 38 | 121 | `c2ac3c61` | multi-session | no |
| 39 | 123 | `gpt4_372c3eed` | multi-session | no |
| 40 | 127 | `80ec1f4f_abs` | multi-session | yes |
| 41 | 128 | `eeda8a6d_abs` | multi-session | yes |
| 42 | 135 | `0edc2aef` | single-session-preference | no |
| 43 | 137 | `32260d93` | single-session-preference | no |
| 44 | 139 | `afdc33df` | single-session-preference | no |
| 45 | 147 | `d24813b1` | single-session-preference | no |
| 46 | 149 | `95228167` | single-session-preference | no |
| 47 | 151 | `75f70248` | single-session-preference | no |
| 48 | 153 | `1da05512` | single-session-preference | no |
| 49 | 154 | `fca70973` | single-session-preference | no |
| 50 | 155 | `b6025781` | single-session-preference | no |
| 51 | 156 | `a89d7624` | single-session-preference | no |
| 52 | 164 | `cc06de0d` | multi-session | no |
| 53 | 166 | `4f54b7c9` | multi-session | no |
| 54 | 167 | `85fa3a3f` | multi-session | no |
| 55 | 168 | `9aaed6a3` | multi-session | no |
| 56 | 179 | `681a1674` | multi-session | no |
| 57 | 186 | `9ee3ecd6` | multi-session | no |
| 58 | 189 | `27016adc` | multi-session | no |
| 59 | 194 | `a96c20ee` | multi-session | no |
| 60 | 198 | `6c49646a` | multi-session | no |
| 61 | 200 | `0ea62687` | multi-session | no |
| 62 | 201 | `67e0d0f2` | multi-session | no |
| 63 | 205 | `60159905` | multi-session | no |
| 64 | 207 | `73d42213` | multi-session | no |
| 65 | 208 | `bc149d6b` | multi-session | no |
| 66 | 209 | `099778bb` | multi-session | no |
| 67 | 214 | `a3332713` | multi-session | no |
| 68 | 216 | `a08a253f` | multi-session | no |
| 69 | 222 | `8e91e7d9` | multi-session | no |
| 70 | 223 | `87f22b4a` | multi-session | no |
| 71 | 226 | `21d02d0d` | multi-session | no |
| 72 | 229 | `e5ba910e_abs` | multi-session | yes |
| 73 | 231 | `ba358f49_abs` | multi-session | yes |
| 74 | 237 | `gpt4_fa19884c` | temporal-reasoning | no |
| 75 | 241 | `gpt4_b5700ca9` | temporal-reasoning | no |
| 76 | 243 | `gpt4_1d4ab0c9` | temporal-reasoning | no |
| 77 | 249 | `gpt4_8279ba02` | temporal-reasoning | no |
| 78 | 251 | `gpt4_a1b77f9c` | temporal-reasoning | no |
| 79 | 253 | `gpt4_7a0daae1` | temporal-reasoning | no |
| 80 | 258 | `4dfccbf7` | temporal-reasoning | no |
| 81 | 259 | `gpt4_61e13b3c` | temporal-reasoning | no |
| 82 | 260 | `gpt4_45189cb4` | temporal-reasoning | no |
| 83 | 261 | `2ebe6c90` | temporal-reasoning | no |
| 84 | 264 | `gpt4_d6585ce8` | temporal-reasoning | no |
| 85 | 265 | `gpt4_4ef30696` | temporal-reasoning | no |
| 86 | 267 | `6e984301` | temporal-reasoning | no |
| 87 | 269 | `gpt4_f420262c` | temporal-reasoning | no |
| 88 | 275 | `gpt4_98f46fc6` | temporal-reasoning | no |
| 89 | 276 | `gpt4_af6db32f` | temporal-reasoning | no |
| 90 | 283 | `gpt4_e414231e` | temporal-reasoning | no |
| 91 | 285 | `gpt4_7bc6cf22` | temporal-reasoning | no |
| 92 | 289 | `b46e15ee` | temporal-reasoning | no |
| 93 | 291 | `gpt4_1e4a8aec` | temporal-reasoning | no |
| 94 | 298 | `9a707b82` | temporal-reasoning | no |
| 95 | 301 | `0bc8ad93` | temporal-reasoning | no |
| 96 | 303 | `gpt4_8279ba03` | temporal-reasoning | no |
| 97 | 312 | `2c63a862` | temporal-reasoning | no |
| 98 | 313 | `gpt4_385a5000` | temporal-reasoning | no |
| 99 | 319 | `gpt4_6ed717ea` | temporal-reasoning | no |
| 100 | 325 | `982b5123` | temporal-reasoning | no |
| 101 | 326 | `b9cfe692` | temporal-reasoning | no |
| 102 | 329 | `gpt4_483dd43c` | temporal-reasoning | no |
| 103 | 330 | `e4e14d04` | temporal-reasoning | no |
| 104 | 331 | `c9f37c46` | temporal-reasoning | no |
| 105 | 332 | `gpt4_2c50253f` | temporal-reasoning | no |
| 106 | 341 | `993da5e2` | temporal-reasoning | no |
| 107 | 345 | `gpt4_88806d6e` | temporal-reasoning | no |
| 108 | 347 | `gpt4_93f6379c` | temporal-reasoning | no |
| 109 | 349 | `gpt4_2f56ae70` | temporal-reasoning | no |
| 110 | 351 | `gpt4_78cf46a3` | temporal-reasoning | no |
| 111 | 353 | `gpt4_1a1dc16d` | temporal-reasoning | no |
| 112 | 359 | `8c18457d` | temporal-reasoning | no |
| 113 | 363 | `c8090214_abs` | temporal-reasoning | yes |
| 114 | 364 | `gpt4_c27434e8_abs` | temporal-reasoning | yes |
| 115 | 365 | `gpt4_fe651585_abs` | temporal-reasoning | yes |
| 116 | 368 | `830ce83f` | knowledge-update | no |
| 117 | 370 | `945e3d21` | knowledge-update | no |
| 118 | 371 | `d7c942c3` | knowledge-update | no |
| 119 | 383 | `f9e8c073` | knowledge-update | no |
| 120 | 387 | `45dc21b6` | knowledge-update | no |
| 121 | 388 | `5a4f22c0` | knowledge-update | no |
| 122 | 389 | `6071bd76` | knowledge-update | no |
| 123 | 396 | `2133c1b5` | knowledge-update | no |
| 124 | 399 | `7a87bd0c` | knowledge-update | no |
| 125 | 400 | `e61a7584` | knowledge-update | no |
| 126 | 403 | `8fb83627` | knowledge-update | no |
| 127 | 405 | `22d2cb42` | knowledge-update | no |
| 128 | 408 | `7e974930` | knowledge-update | no |
| 129 | 411 | `5831f84d` | knowledge-update | no |
| 130 | 412 | `eace081b` | knowledge-update | no |
| 131 | 413 | `affe2881` | knowledge-update | no |
| 132 | 419 | `dfde3500` | knowledge-update | no |
| 133 | 425 | `06db6396` | knowledge-update | no |
| 134 | 429 | `dad224aa` | knowledge-update | no |
| 135 | 432 | `5c40ec5b` | knowledge-update | no |
| 136 | 433 | `c6853660` | knowledge-update | no |
| 137 | 434 | `26bdc477` | knowledge-update | no |
| 138 | 440 | `0ddfec37_abs` | knowledge-update | yes |
| 139 | 442 | `89941a94` | knowledge-update | no |
| 140 | 443 | `07741c45` | knowledge-update | no |
| 141 | 444 | `7161e7e2` | single-session-assistant | no |
| 142 | 446 | `89527b6b` | single-session-assistant | no |
| 143 | 451 | `1903aded` | single-session-assistant | no |
| 144 | 452 | `ceb54acb` | single-session-assistant | no |
| 145 | 459 | `488d3006` | single-session-assistant | no |
| 146 | 462 | `1d4da289` | single-session-assistant | no |
| 147 | 463 | `8464fc84` | single-session-assistant | no |
| 148 | 464 | `8aef76bc` | single-session-assistant | no |
| 149 | 469 | `3249768e` | single-session-assistant | no |
| 150 | 473 | `e8a79c70` | single-session-assistant | no |
| 151 | 475 | `e3fc4d6e` | single-session-assistant | no |
| 152 | 480 | `fca762bc` | single-session-assistant | no |
| 153 | 481 | `7a8d0b71` | single-session-assistant | no |
| 154 | 485 | `41275add` | single-session-assistant | no |
| 155 | 488 | `561fabcd` | single-session-assistant | no |
| 156 | 490 | `ac031881` | single-session-assistant | no |
| 157 | 493 | `c8f1aeed` | single-session-assistant | no |
| 158 | 495 | `c7cf7dfd` | single-session-assistant | no |
| 159 | 496 | `e48988bc` | single-session-assistant | no |
