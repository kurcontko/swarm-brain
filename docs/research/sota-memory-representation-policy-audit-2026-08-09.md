# Memory representation and learned-policy frontier audit

Date: 2026-08-09

Status: research and experiment-design note only. It changes no serving
behaviour and reports no Swarm Brain quality result. The sources are author
papers and author repositories inspected at the immutable revisions below.

## Executive finding

Swarm Brain is well aligned with the safety and lifecycle substrate of the
2026 memory-agent literature. The implemented benchmark controls now also
cover the strongest new evidence on **retrieval keys** and
**query-time construction**, while learned memory control remains deferred.

The newest controlled studies change the near-term ordering of work:

1. Finish the frozen turn-level selection and exact-prompt QA comparison.
2. Test broad retrieval followed by query-time selective construction, while
   retaining byte-level links to the immutable raw turns.
3. Test multi-key representations while retaining the immutable raw value.
4. Test descriptive entity activation without graph expansion before adding
   graph traversal.
5. Add one-hop expansion only with an explicit post-expansion reranker.
6. Consider learned memory operations only after the fixed pipeline has a
   held-out outcome signal and immutable action receipts.

This ordering follows the ablations. LazyMem v2 finds that retrieving raw
messages broadly and constructing evidence only after the query is known can
move the accuracy/context frontier. UnifiedMem finds that raw dialogue plus
derived summary/fact/keyword keys is a very strong flat baseline, that a
similarity graph can hurt, and that direct entity activation is already close
to one-hop expansion. Memora independently supports separating an indexed
abstraction/cue layer from a high-fidelity value. Memory-R1 and AgeMem support
learning memory actions, but neither supplies a directly comparable,
production-safe Swarm Brain policy recipe.

## Primary-source snapshot

| System | Author paper | Author artifact frozen for this audit | Reproducibility class |
| --- | --- | --- | --- |
| UnifiedMem | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1232/) | [`AvatarMemory/UnifiedMem` at `3df9428e6a788d2c2ab6b859c85b937a0128ba2f`](https://github.com/AvatarMemory/UnifiedMem/tree/3df9428e6a788d2c2ab6b859c85b937a0128ba2f) | **Strong controlled evidence, partial exact reproduction.** The repository covers index/retrieve/QA/evaluation, but model deployments and several generated artifacts are not immutable provider identities. |
| Memora | [arXiv:2602.03315v2](https://arxiv.org/abs/2602.03315v2) | [`microsoft/Memora` at `dec3f8f2444eace7004fc084abe1be9f3d88270e`](https://github.com/microsoft/Memora/tree/dec3f8f2444eace7004fc084abe1be9f3d88270e) | **Runnable, not environment-pinned.** The one-commit release includes LongMemEval/LoCoMo runners, but dependencies are unpinned and external deployment revisions are not attestable from the repository. |
| LazyMem | [arXiv:2607.22690v2](https://arxiv.org/abs/2607.22690v2) | [`allacnobug/LazyMem` at `af4109960aacb90d6dba994e9103a36a165cc380`](https://github.com/allacnobug/LazyMem/tree/af4109960aacb90d6dba994e9103a36a165cc380) | **Runnable paper pipeline, model-heavy reproduction.** The two-commit release pins window mechanics and dependency versions, but requires retriever/reranker, constructor, reward-judge, and answer-model deployments and publishes no generated run artifacts or weights in the repository. |
| Memory-R1 | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.583/) | [`yansikuan/memory-r1` at `9c413a2413c4fee160ec05445856c1529d63ac7a`](https://github.com/yansikuan/memory-r1/tree/9c413a2413c4fee160ec05445856c1529d63ac7a) | **Paper only for implementation purposes.** The author repository contains a README and figures and still says code is coming soon. |
| AgeMem | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.981/) | [`y1y5/AgeMem` at `98f563f907d67b2f2436e3ae7b7ceff32e482814`](https://github.com/y1y5/AgeMem/tree/98f563f907d67b2f2436e3ae7b7ceff32e482814) | **Code released, different task family.** It trains on HotpotQA and evaluates on five QA/embodied reasoning tasks, not persistent-dialogue or LongMemEval. |

Paper scores below are evidence about each paper's own protocol. They are not
Swarm Brain scores and must not be placed on one leaderboard unless dataset,
reader, judge, prompt, context budget, and model snapshots all match.

## Exact evidence and implications

### UnifiedMem: keys usually matter before graph complexity

UnifiedMem decomposes systems into key, value, index operation, index
structure, retrieval, and answer-generation choices. Its controlled
LongMemEval experiments provide unusually direct guidance for Swarm Brain.

| Finding | Reported evidence | Swarm Brain implication |
| --- | --- | --- |
| Preserve raw content as a retrieval key while adding derived keys. | On LongMemEval-S, `session,[S,F,K]` reports R@5/R@10 `.9379/.9833`, versus session-only `.9021/.9714`; `[S,F,K]` without the independent raw key reports `.9045/.9642`. `S`, `F`, and `K` are summary, factual statements, and keywords. | Do not replace evidence-bearing content with a summary. Add independently addressable derived keys that hydrate the same immutable value. |
| Similarity edges can make retrieval worse. | `[S,F,K]` R@10 `.9642`; SimGraph `.9498`; KnowGraph `.9665`; descriptive-entity graph `.9713`. The authors attribute the SimGraph regression to noisy expansion without effective reranking. | Keep similarity-link traversal disabled by default. Treat it as a negative control, not a presumed improvement. |
| Activate entities, not relation triples. | Direct entity activation reports R@10 `.9905`; triple activation `.9809`. | Entity descriptions are a better first graph key than triples for this task family. |
| Expansion is not automatically valuable. | Without expansion, entity-key ranking reports R@10 `.9904`; one-hop with entity score alone falls to `.9643`; one-hop with entity score plus graph support count reaches `.9928`. | Run no-expansion first. If expansion is tested, rerank every hydrated value after expansion and record the graph-derived secondary signal. |
| Retrieval recall and answer usefulness can diverge. | With GPT-4o-mini extraction, text-embedding-3-small, and GPT-4o answering, graph retrieval reaches R@10 `.9928`. QA is `.892` when raw session values are delivered but `.690` when only graph keys are delivered. | Retrieval keys are navigational metadata, not necessarily reader evidence. Always hydrate source-backed values before prompt packing. |
| Richer graphs cost materially more to construct. | On LongMemEval-S, the paper reports 156 minutes for flat construction and 1,181 minutes for graph construction; retrieval is 45 ms/query flat and 44 ms/query graph in that setup. | Include construction cost and storage growth in the promotion frontier; query latency alone hides the dominant graph cost. |

The paper also finds that update/no-op operations increase memory recall and
end-to-end QA on HaluMem despite decreasing extraction precision. That result
supports testing state maintenance, but not destructive in-place mutation:
Swarm Brain should preserve append-only versions and expose update/no-op as
policy decisions over those versions.

### Memora: separate identity/access keys from high-fidelity values

Memora represents each entry as:

- a concrete memory value, retained in full and not directly indexed;
- one indexed primary abstraction, used as the canonical concept and update
  target; and
- multiple indexed cue anchors, providing many-to-many access paths.

Candidate abstractions are matched to the existing store and consolidated by
an LLM only after cosine prefiltering. The paper's default update threshold is
`.80`; its LoCoMo ablation reports overall judge score `.795` with no update,
`.801` at `.80`, and `.799` at `.60`, while `.60` triggers 3.4 times as many
updates. The threshold is paper-specific evidence, not a universal production
constant.

On the full 500-question LongMemEval-S split, the paper reports binary judge
accuracy of `83.8%` for semantic retrieval and `87.4%` for prompted policy
retrieval, with average contexts of about 2.1k and 2.9k tokens respectively.
The prompted policy uses a bounded state `(query, working set, frontier,
remaining budget)` and actions REFINE, EXPAND, and STOP. The public default
configuration uses top-k 30, cue top-k 20, at most four prompted-policy steps,
update threshold `.80`, GPT-4.1-mini as reader/curator, and GPT-4o-mini as
judge. The release does not pin package versions or provider deployment
snapshots, so these numbers remain an external target rather than a replayed
baseline.

The most transferable result is structural: index a bounded navigation layer,
then hydrate immutable detail. Swarm Brain should not copy Memora's choice to
leave raw values unindexed until a paired ablation beats the strong
UnifiedMem-style raw-plus-derived control.

### LazyMem: preserve broadly, construct only after the query is known

LazyMem retrieves raw messages with dense/BM25 RRF plus a CrossEncoder,
retains the top 50, restores a radius of two local messages, and processes the
result in overlapping parallel windows capped at eight messages with stride
seven. A memory-processing model emits one KEEP/DROP decision per message and
can preserve, extract, or compress kept content. Duplicate kept messages are
resolved by retaining the longer compression, then the evidence is restored
to chronological order for the answer model.

The v2 paper reports LongMemEval LLM-judge accuracy `.85` with an average of
213 answer-context memory tokens for the trained 4B constructor and `.93` with
1,041 tokens for the Qwen3-32B constructor. Its protocol is not comparable to
Swarm Brain's official track: the released pipeline uses Qwen3-Embedding-8B,
BGE-Reranker-v2-M3, Qwen3-32B as the frozen answer model, DeepSeek-V4-Pro as
the evaluation judge, and a type-stratified 360/40/100 train/validation/test
split. The paper's result is therefore evidence for the mechanism and
accuracy-efficiency objective, not a scalar Swarm Brain threshold.

The transferable mechanism is important: write-time compression cannot know
which future detail matters, while query-time construction can first optimize
coverage and then remove noise. The failure mode is equally important. The
paper attributes most remaining LongMemEval errors to memory editing after the
required evidence was already retrieved. Swarm Brain should therefore test a
strict grounded variant alongside the paper-shaped cell, requiring every KEEP
to cite immutable UTF-8 source spans and forbidding unverifiable paraphrase.

### Memory-R1: learned writes and learned reading are separate decisions

Memory-R1 trains two agents separately:

- a Memory Manager chooses ADD, UPDATE, DELETE, or NOOP; and
- an Answer Agent receives 60 similarity-retrieved memories and learns memory
  distillation plus answer generation.

The manager is trained against downstream answer exact match, with the answer
agent frozen; the answer agent is then trained with the manager fixed. This
decoupling is important because it reduces attribution ambiguity. The paper
trains on 152 LoCoMo questions and evaluates LongMemEval zero-shot, but it uses
the **oracle** LongMemEval variant and its own reader/judge setup. Its reported
LongMemEval numbers therefore cannot be compared to Swarm Brain's official
LongMemEval-S track.

The paper's outcome signal is also insufficient for Swarm Brain governance on
its own. A policy may learn whether content appears useful, but it must never
learn authority over tenant/project scope, visibility, evidence trust,
retention enforcement, or destructive deletion. A learned DELETE should map
to a reviewable proposal or lifecycle transition, never physical evidence
loss.

### AgeMem: unified tools are promising, but the benchmark transfer is open

AgeMem exposes six actions: ADD, UPDATE, DELETE for long-term memory and
RETRIEVE, SUMMARY, FILTER for short-term context. It progressively trains
information acquisition, distractor/context control, and integrated reasoning
with step-wise GRPO. On Qwen2.5-7B, the full policy averages `41.96%` across
ALFWorld, SciWorld, PDDL, BabyAI, and HotpotQA versus `33.43%` without RL; on
Qwen3-4B the corresponding values are `54.31%` and `45.59%`.

Those are useful cross-domain results, but the authors explicitly identify
persistent, long-term dialogue and real-user interaction as future work. They
train only on HotpotQA, use task-family-specific success/progress/judge
metrics, and do not evaluate LongMemEval. AgeMem therefore supports a future
tool-policy experiment, not a current SOTA claim for this framework.

## Alignment with Swarm Brain today

| Frontier capability | Current alignment | Remaining gap |
| --- | --- | --- |
| Immutable high-fidelity value | **Strong.** Canonical memory content is append-only, evidence-linked, versioned, scoped, and bitemporal. | Reader experiments still need to prove when raw turns, sessions, or extracted records are the best hydrated value. |
| Multi-key retrieval | **Executed development evidence; partial serving alignment.** E6 v2 generated and journal-replayed 5,248 source-only merged keys, scored raw and merged families with pinned Qwen, hydrated only canonical raw turns, and completed paired DeepSeek QA. R1 preserved perfect gold-context recall and raised answer-session MRR from `.9083` to `.9167`. | QA tied 9/10, while R1 added 12,786 prompt tokens and `$0.287570` construction cost. The result is retained only for a larger budget-matched study; no multi-key cell is wired to serving. R2-R5 remain unexecuted. |
| Query-time construction | **Executed reliability evidence.** E7-A/B/C bind the replayed E1-B top 50, exact LazyMem v2 radius-2/window-8/stride-7 geometry, source-safe constructor inputs, retained raw provider-response bytes, strictly replayed normalized decisions and reconciled usage, chronological deduplication, and byte-level source spans. | The one-question E7-C smoke failed closed on its sixth constructor response: six decisions were returned for seven messages after five all-DROP windows. No context, reader, or judge ran. E7 is therefore a construction-reliability rejection, not a QA result, and is not wired to serving. |
| Update/consolidation | **Stronger safety substrate than the papers.** Updates append versions; evidence, lineage, scope, trust, and stale-snapshot checks are enforced locally. | The selection policy remains heuristic/provider-proposed and has no held-out learned-operation result. |
| Graph memory | **Partial.** Typed, provenance-backed links and bounded exact rehydration exist. | There is no descriptive-entity index, and current links are not an experimentally validated LongMemEval graph. Similarity expansion must remain off absent evidence. |
| Context control | **Strong deterministic substrate with one model-backed run.** Whole-memory packing, hard budgets, exact rehydration, text-free traces, exact 4,096/8,192/16,384 full-prompt turn packing, and E7 query-time construction controls exist. E6 completed exact 8,192-token packing and paired QA. | E6-R1 was not budget-matched to R0 and did not improve QA. No learned REFINE/EXPAND/STOP or RETRIEVE/SUMMARY/FILTER policy has been evaluated. |
| Outcome attribution | **Strong provenance, weak causal evidence.** Exact activation/citation/version associations exist and are offline-only. | No randomized or held-out action-policy experiment separates useful decisions from correlated retrieval. |
| Empirical standing | **Development evidence is now reproducible; SOTA is not established.** E1, E2, E6, and E7 have frozen real-model artifacts with raw-call replay. E6's 5,288 external-call journals rebuild to the same report SHA. | The required full-500, repeated, held-out, protocol-comparable QA evidence is absent. Agentic, swarm-causal, GateMem, MemoryArena, and Mem2Act gates also remain open. |

## Frozen representation experiment contract (R0/R1 executed)

The benchmark tree implements **E6/SB-HMR-v1** (harmonic multi-key
representation). E6-R0/R1 have now executed on the frozen ten-question
development sample with DeepSeek extraction/QA, pinned local Qwen ranking,
exact tokenization, and complete journal replay. R1 produced a small MRR gain
but tied QA and paid a substantial efficiency tax, so it remains a Swarm Brain
transfer hypothesis rather than a Memora or UnifiedMem reproduction. R2-R5
and R-neg retain the pure offline contract below and remain unexecuted.

### Invariants

- Keep the source value byte-identical and evidence-bound. Derived keys never
  replace or silently edit it.
- Bind every derived key to source ID/version/hash, extractor prompt hash,
  model/deployment identity, and output hash.
- The extractor may propose summary, facts, keywords, primary abstraction,
  cue anchors, and entity descriptions; local code owns identity, scope,
  lifecycle, and lineage.
- Run all cells with the same candidate corpus, query set, reader prompt,
  tokenizer, 8,192-token full-prompt ceiling, judge, and paired report.
- Hydrate the same value type before packing. A key-only reader diagnostic is
  separate and cannot be the primary QA cell.
- Do not use gold answers, gold sessions, or question types during extraction,
  scoring, consolidation, routing, or packing.

### Cells

| Cell | Retrieval representation | Purpose |
| --- | --- | --- |
| R0 | Raw value only | Current flat control. |
| R1 | Raw key plus one merged `[summary, facts, keywords]` key | Direct UnifiedMem transfer control. |
| R2 | Raw key plus separate summary, fact, and keyword keys | Tests whether independent fine-grained access beats merging. |
| R3 | Primary abstraction plus cue anchors, hydrating raw value | Memora-shaped navigation control. Raw-key inclusion is a separate `R3+raw` diagnostic. |
| R4 | Descriptive entity activation, no expansion, hydrating raw value | Strong graph-key baseline before traversal. |
| R5 | R4 plus one-hop expansion and deterministic `(entity score, graph support count, prior rank, value ID)` ordering | Tests whether bounded expansion adds value after an explicit reranker. |
| R-neg | Similarity-edge one-hop expansion | Preserved negative control; never eligible for serving promotion by itself. |

Use the same weighted-RRF boundary when a value is reached by multiple keys.
Report both key-level and hydrated-value-level recall so a large key fan-out
cannot masquerade as better evidence. Also report index tokens/bytes, derived
objects per source value, construction tokens/latency/cost, update rate, and
orphan/duplicate key rates.

### Consolidation diagnostic

Only after R1-R5 are frozen, add a shadow consolidation experiment with
create/update/no-op over derived navigation objects. Test no-update, `.80`
cosine prefilter plus local identity validation, and a lower-threshold negative
control. Never merge canonical raw evidence. Never use physical deletion.

## Learned-policy experiment after fixed-pipeline evidence

The first learned-policy experiment should be narrow and reversible:

1. Freeze the winning fixed representation, retrieval, selection, and packing
   pipeline.
2. Train only a bounded query-time action policy over REFINE, EXPAND, and STOP
   or RETRIEVE, SUMMARY, and FILTER. Do not jointly learn writes yet.
3. Use immutable action receipts and a reward vector containing paired answer
   correctness, evidence grounding, redundancy, delivered tokens, latency,
   and cost. Keep each component separately reportable; do not hide them in one
   scalar at evaluation time.
4. Train on a preregistered training split; choose on validation; evaluate once
   on a separately sealed corpus that did not motivate E1/E2/E6.
5. Compare against the exact fixed policy at equal budgets and model identity.
6. Only then consider ADD/UPDATE/NOOP learning. DELETE remains a governed
   proposal, and all hard scope/trust/evidence rules remain outside the policy.

Memory-R1's alternating frozen-agent schedule is the safer starting point than
joint end-to-end training. AgeMem's six-tool policy becomes appropriate only
after each tool has a deterministic baseline and an auditable no-op/failure
semantics.

## Frozen query-time construction contract (reliability smoke executed)

The benchmark tree implements **E7** as a separate family after E1-B:

- E7-A delivers the chronological raw top 50 with no constructor;
- E7-B accepts query-conditioned KEEP/DROP plus verbatim, extractive, or
  caller-attested abstractive content; and
- E7-C accepts only locally byte-verifiable verbatim/extractive content.

All cells use the same source-byte-verified question/current-date preflight and
the exact paper window geometry. Constructor requests contain no question
type, answer, gold session, `has_answer`, or judge material. Each normalized
response is bound to one decision per window message, unique request IDs,
model/prompt/tokenizer identities, accounting, and cited source spans. E7 v2
retains each exact raw provider response privately and strictly replays the
response model, provider ID, stopped choice, normalized decisions, and
reconciled usage. The E7-C smoke made six constructor calls and then failed
closed when the sixth response omitted one of seven required decisions. Raw
bytes remain preserved and externally attested rather than endpoint-
authenticated; no reader or judge ran, and the frozen v1 failure cannot be
silently retried or repaired.

## Bottom line

Architecturally, Swarm Brain is close to the frontier's durable substrate and
ahead of the papers on governance and replayability. Empirically, it is not
SOTA yet: reproducible ten-question development evidence exists, but no
current full, held-out, protocol-comparable QA artifact has passed the gate.

The most defensible improvement is not to add a large graph or RL controller
immediately. It is to preserve immutable values and test the only surviving
signal--raw plus merged navigation keys--at the same 20-value reader head as
the raw control on a larger paired development sample. Direct entity
activation should follow only if that cheaper control fails. Learned policies
come after fixed, budget-matched comparisons, inside the existing hard
governance boundary.
