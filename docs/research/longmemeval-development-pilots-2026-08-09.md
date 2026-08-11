# LongMemEval real-model development pilots (2026-08-09)

These pilots close three previously unexecuted model boundaries in the frozen
selection/organization protocol. They are exploratory results on the same
seeded 10-question development sample, not official LongMemEval scores, not a
sealed held-out confirmation, and not evidence for a production change.

## Shared boundary

- Dataset: cleaned LongMemEval-S, SHA-256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Sample: 10 of 500 questions, seed `20260807`.
- Dense model: `Qwen/Qwen3-Embedding-0.6B` at
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Reader and development judge: `deepseek-v4-flash`, thinking disabled.
- Prompt ceiling: exact 8,192 complete DeepSeek-chat tokens.
- Every provider prompt-token count matched the independently pinned local
  tokenizer count. Raw credential-free request and response bytes are retained
  in sealed receipt sidecars.
- The official `gpt-4o-2024-08-06` judge was deliberately not run. It remains a
  final confirmation step only after a stable candidate wins development and
  held-out comparisons.

## E1: independent learned reranking

Protocol: `swarmbrain-longmemeval-e1-real-model-development-v3`.

E1-A is production-shaped lexical/Qwen weighted RRF. E1-B reranks the exact
E1-A 128-turn pool with `mixedbread-ai/mxbai-rerank-large-v1` at
`98f655841d5caf0b16eaff79c2b4ca109d920d17` using raw logits.

| Metric | E1-A | E1-B |
| --- | ---: | ---: |
| Development accuracy | 8/10 | 8/10 |
| Mean complete prompt tokens | 8,181.7 | 8,179.0 |
| Mean kept turns | 31.9 | 29.9 |
| Any/all gold session in prompt | 1.0 / 1.0 | 1.0 / 1.0 |
| Mean answer-session recall | 1.0 | 1.0 |
| Candidate MRR | 0.9091 | 0.9333 |

All ten paired QA labels tied. The mean accuracy delta is `0.0`, with paired
bootstrap 95% interval `[0.0, 0.0]`. The reranker improved candidate MRR but
not end-to-end answers, so E1-B is rejected and E1-C/E1-D are not justified by
this sample.

- Report artifact SHA-256:
  `907bbf571458d556f285617362206e2b648dc62ddb6e392ea81f143fe90f216e`.
- Run manifest SHA-256:
  `57204a2fea2d49a7b46669b5810470da6c3df3b2a93ab7c311390c38a47a3f2d`.
- Pessimistic DeepSeek cost upper bound: `$0.0238875`.

## E2: context-conditioned Chain-of-Memory organization

Protocol: `swarmbrain-longmemeval-e2-real-model-development-v1`.

The experiment fixes the initial candidate set to E1-A top 20. E2-D executes
the preregistered product gate with three chains and beta `0.5`, preserving
cross-chain reuse as an organization-only parity diagnostic. The v1 exact
packer rejects repeated turn IDs, so reader cell E2-E removes cross-chain
duplicates while preserving E2-D's evolution and order. E2-A is the unchanged
retrieval-order control.

The reader comparison is unusually well isolated: both arms kept the identical
20 turns for every question. E2-A used 3,503--8,135 prompt tokens; E2-E added
only 7--14 header tokens. Thus the paired result tests order, not recall,
selection, or truncation at the reader budget.

| Metric | E2-A | E2-E |
| --- | ---: | ---: |
| Development accuracy | 9/10 | 8/10 |
| Mean complete prompt tokens | 5,770.5 | 5,778.9 |
| Mean kept turns | 20.0 | 20.0 |
| Any/all gold session in prompt | 1.0 / 1.0 | 1.0 / 1.0 |
| Mean answer-session recall | 1.0 | 1.0 |
| Candidate MRR | 0.9091 | 0.9111 |
| Mean reader latency (ms) | 2,189.5 | 1,677.2 |

Paired outcomes were zero improvements, one regression, and nine ties. The
mean E2-E-minus-E2-A accuracy delta is `-0.10`; paired bootstrap 95% interval
is `[-0.30, 0.0]`.

The mechanism also failed to transfer as an adaptive truncator. E2-D averaged
55.8 rendered occurrences and 538.8 context-similarity calls out of hard
maxima 60 and 570. Eight questions built three full 20-turn chains; only two
questions stopped any chain early. Cross-chain deduplication reduced the mean
render to 20 unique turns, but the induced order did not improve QA.

- Report artifact SHA-256:
  `b55eac32ae48ff6f0b4608f0a83a5381798298545cd1e858fffa87986d462599`.
- Run manifest SHA-256:
  `9043e59e40390dd62973adda1dc6f71e4544a08e709e4b573f0ec633059b6b1b`.
- Ordered organization artifact-set SHA-256:
  `c096f39d8db2f47f8b6abe13ca54ee15b963557dfd758ccb71cdc5d016dae23b`.
- Ordered QA receipt-set SHA-256:
  `32b6b58f7aa706026016ca07020fdb4b4303ff948268a6afcc0b2dc574779967`.
- Pessimistic DeepSeek cost upper bound: `$0.01715966`.

## E7: source-grounded query-time construction

Protocol: `swarmbrain-longmemeval-e7-real-model-development-v1`.

E7-A is the raw chronological E1-B top 50. E7-C applies the LazyMem-shaped
radius-2/window-8/stride-7 constructor transfer, but permits only
byte-verifiable verbatim or extractive output. The first selected question
produced 29 frozen windows with 147 message appearances. Raw constructor
request/response bytes were retained before normalization.

The predeclared one-question end-to-end smoke stopped during construction. The
first five windows returned schema-valid receipts for 21 messages, but marked
all 21 `DROP`. Window index 5 contained seven messages while the provider
returned only six decisions, all `DROP`. Because an omitted decision cannot be
attributed to a source message without inventing evidence, strict replay
rejected the response and no context pack, reader, or judge ran.

- Run manifest artifact SHA-256:
  `477f13fd9c1244bac8109d33ec3288296a6a017702ae761fc8ee53a8acfaada0`.
- First window artifact SHA-256:
  `6402135c608f5598f84cbd52e9c9fb7fe444a935e0924831305d2ae513b21384`.
- Raw six-call receipt file SHA-256:
  `5e031457bc1bec94e74e0ced8b2cca17cf124eeee5d0c72db8291a585ba38d31`.
- API usage before the stop: 9,696 prompt and 1,309 completion tokens.
- Pessimistic DeepSeek cost upper bound: `$0.00172396`.

This is a protocol-reliability rejection, not an E7 QA score. The exact raw
failure remains preserved; it must not be repaired, normalized, or silently
retried inside this frozen v1 run. A future schema-retry protocol would be a
new preregistered experiment and cannot retroactively qualify v1.

## Decision before E6

Neither independent learned reranking nor this Chain-of-Memory transfer is a
positive mechanism, and E7-C did not clear its construction-reliability smoke
gate. Do not compose them, scale them, or change serving. The next development
experiment should test the different remaining causal lever: a narrow
source-preserving R0-versus-R1 multi-key representation diagnostic. Its
construction evidence and cost must be reported outside selection-report v2,
which cannot yet validate E6 receipts. Future beta/model or structured-output
retry diagnostics may explain the E2/E7 failures, but they are not promotion
candidates and should not preempt the frozen E6 comparison.

## E6: source-preserving merged representation keys

Protocol: `swarmbrain-longmemeval-e6-r0-r1-development-v2`.

E6-R0 replays the frozen E1 exhaustive Qwen scores over canonical raw turns.
E6-R1 adds exactly one source-only merged summary/fact/keyword navigation key
per raw turn, embeds that family with the same pinned Qwen model, combines raw
and merged families with equal-family RRF, and hydrates only the authoritative
raw turns. The derived key is never shown to the reader as memory content.

The initial v1 manifest was superseded before execution after its local
preflight exposed a quadratic implementation path: every question-local value
recomputed the digest of the complete 246,750-turn projection. The v1 output
contains only a zero-byte process lock and its sealed manifest--no reservation,
response, settlement, phase artifact, provider call, or E6 spend. Its manifest
artifact SHA-256 is
`094e2046fd4864dcd73a7f36df0178343417fc89aed07b7c55c43d3b58626d17`.

v2 caches that digest once after complete corpus validation and eagerly caches
the 5,248 immutable values selected by the frozen E1 sample. Exact differential
checks across all ten questions preserved value tuples and projection hashes.
The cache changes no request bytes, routes, ordering, scores, prompts, cost
accounting, or gate input; local unauthenticated caller-clock timing remains
excluded from the gate. The corrected implementation loaded the full corpus
and cached the selected values in 19.57 seconds, versus more than 15 minutes
without reaching a first call in the aborted v1 preflight.

| Metric | E6-R0 | E6-R1 |
| --- | ---: | ---: |
| Development accuracy | 9/10 | 9/10 |
| Mean complete prompt tokens | 4,759.5 | 6,038.1 |
| Total complete prompt tokens | 47,595 | 60,381 |
| Any/all gold session in prompt | 1.0 / 1.0 | 1.0 / 1.0 |
| Mean answer-session recall | 1.0 | 1.0 |
| Answer-session MRR | 0.9083 | 0.9167 |
| Construction calls | 0 | 5,248 |
| Construction cost upper bound | `$0` | `$0.287570` |

The context-first gate advanced R1 because its MRR improved by `0.00833` while
every other gold-context metric tied at `1.0`; therefore R0 did not strictly
Pareto-dominate it. The optional paired DeepSeek reader/development-judge stage
then tied on every case: both arms scored 9/10, for an accuracy delta of `0.0`.
R1 used 12,786 additional prompt tokens and added construction cost and
latency, so the ten-question evidence does not establish an end-to-end gain.

All 5,248 extraction outputs were schema-valid on their first application
attempt. The final journal reconciles 5,288 reservations, raw response WALs,
and settlements one-for-one: 5,248 extraction calls, 20 reader calls, and 20
development-judge calls, with zero unresolved reservations. The total
pessimistic DeepSeek cost upper bound is `$0.303687`. No official GPT-4o call
was made.

- Report artifact SHA-256:
  `c56dd9ff90c881f52416c53be52e654018adb29cbcc8f847fe92eaf34933bb49`.
- Diagnostic artifact SHA-256:
  `4ea0b70c27b7f9d8db5ece13be05150ebb645b8efb706b147eb0d8b8bd7d5aae`.
- Run manifest SHA-256:
  `ab6a7a5d943ac46c0b423c7bb71c35fedcdc45f44a54ea9bb453b20f6a99ba1b`.
- Implementation tree SHA-256:
  `fc669483472f87d5acdfb7797107d34ba82ebb5f77c7f0e78035f34cce4940cd`.

The frozen verdict is `retain-R1-for-further-development-only`. It is not
eligible for composition, serving promotion, held-out confirmation, or the
one final official GPT-4o run. If continued, it requires a newly frozen,
larger paired development protocol first.

## Updated decision

E1 learned reranking and E2 Chain-of-Memory ordering are rejected on this
development sample. E7 is rejected at construction reliability. E6 is the
only retained mechanism, but its small MRR movement did not transfer to QA and
comes with a substantial prompt/construction tax. Do not compose or serve it.
The next experiment must first test whether the E6 MRR signal survives a
larger paired development sample and whether a budget-matched R1 variant can
convert that signal into accuracy. Official GPT-4o evaluation remains deferred
until one stable candidate clears development and held-out gates.
