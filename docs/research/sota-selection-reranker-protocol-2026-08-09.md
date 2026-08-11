# Reranking and global-selection protocol audit

Date: 2026-08-09

Status: research and experiment-design note only. It changes no serving
behaviour. The audit uses primary sources: author papers, author repositories,
and publisher model cards.

## Executive correction

The next experiment should not be described as "add the Mnemis Qwen3
reranker." In Mnemis's own System-1 ablation, replacing RRF with
Qwen3-Reranker-8B changed the overall LoCoMo score from **89.1 to 89.1**. The
full-system improvement came from adding hierarchical global selection:
System-1 RAG + Graph scored 89.1, System-2 alone 87.7, and their combination
93.3. Qwen3-Reranker-8B remains useful as a Mnemis reproduction control, but it
is not the strongest evidence for a quality improvement by itself.

The higher-evidence experiments are:

1. SmartSearch-style CrossEncoder ranking and budget-aware compilation over a
   sufficiently deep, turn-level candidate pool.
2. Chain-of-Memory's context-conditioned chain organization and adaptive path
   truncation.
3. A multi-key representation experiment that preserves raw values while
   adding summary/fact/keyword, abstraction/cue, and descriptive-entity keys,
   as specified in the
   [representation and policy audit](sota-memory-representation-policy-audit-2026-08-09.md).
4. Mnemis global selection as a separate, more expensive hierarchy experiment.
5. MAGMA-inspired graph-view routing only after the above controls, because the
   released MAGMA code does not execute the paper's stated beam-search method.

## Primary-source snapshot

| System | Author paper | Author artifact frozen for this audit | Reproducibility class |
| --- | --- | --- | --- |
| Mnemis | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1096/) | [`microsoft/Mnemis` at `4552fed19bc0cde7b990a6ceb0365cd75b1b3453`](https://github.com/microsoft/Mnemis/tree/4552fed19bc0cde7b990a6ceb0365cd75b1b3453) | **Partial / underspecified.** The repository releases global-selection code, prompts, and result contexts, but not the complete ingestion, System-1, reranking, and evaluation harness. |
| SmartSearch | [arXiv:2603.15599v1](https://arxiv.org/abs/2603.15599v1), [author source](https://arxiv.org/src/2603.15599v1) | No author code release was linked or found. Model-publisher cards are linked below. | **Partial / no code.** Most ranking constants are explicit, but passage serialization, immutable model revisions, tokenizer, and hardware are not. The paper is internally inconsistent about the ColBERT identity/size. |
| Chain-of-Memory (CoM) | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.534/) | [`Xiucheng-Xu/CoM` at `52aa7ffd641059435c4585b6d9dad660518be635`](https://github.com/Xiucheng-Xu/CoM/tree/52aa7ffd641059435c4585b6d9dad660518be635) | **Partial / paper-code divergent.** The chain algorithm is runnable, but released retrieval inserts a top-10 session prefilter not stated in the paper and ignores its `max_chain_length` argument. |
| MAGMA | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1709/) | [`FredJiang0324/MAGMA` at `467cb70b67ac337b22fdb42194d37c04ad701b62`](https://github.com/FredJiang0324/MAGMA/tree/467cb70b67ac337b22fdb42194d37c04ad701b62) | **Code-divergent / not exactly replicable.** The paper's beam-search coefficients are incomplete, and the released query path calls a materially different BFS traversal. |

Downloaded paper/source checksums used during extraction:

| Artifact | SHA-256 |
| --- | --- |
| Mnemis ACL PDF | `1caed16fc8729d4f9e7d76b0746885e4c40233daf18a4931ad1a9769c36ce7d4` |
| SmartSearch arXiv v1 PDF | `1cb94a223220b1622ab0f344f1b35847013cf72fd09709dcb351a2a8651fd8a4` |
| SmartSearch arXiv v1 source tar | `5bab1c12ef920cce0d385dfab66208fbc3df2b95f1f21df07ef0d90d2176ece2` |
| CoM ACL PDF | `7fd73702d4673cb042221c9330fd72ec70d48f050065c9bab97c632113232b5d` |
| MAGMA ACL PDF | `53077bf08bb681f0ce29e5bfed1920c4e7068000beb9beead41bd98c91298501` |

## Exact protocol audit

### Mnemis

| Aspect | Primary-source protocol | Missing or divergent detail |
| --- | --- | --- |
| Memory representation | Base graph contains raw Episodes, Entities, factual Edges, and Episodic edges. A many-to-many hierarchy abstracts entities into Categories. | The hierarchy's minimum child count `n`, maximum layer count, batching, and exact construction deployment are not reported. |
| System-1 candidates | For each of Episodes, Entities, and Edges: cosine embedding search plus BM25 full-text search, then per-type RRF. Embeddings are `Qwen3-Embedding-0.6B` at MRL dimension 128. | Initial cosine/BM25 depths and RRF constant are not stated. Appendix A accidentally calls the embedder `Qwen3-Reranker-0.6B`; the method and setup call it `Qwen3-Embedding-0.6B`. |
| System-2 selection | Start with **all** top-layer categories. At each layer an LLM sees category name and tag and selects every potentially useful category; there is no strict top-k. `get_all_children=true` selects all descendants without deeper LLM calls. Selected lowest-level entities hydrate all directly connected episodes and edges plus entities across those edges. | The paper does not pin the selector model revision. Released [`GlobalSelectorConfig`](https://github.com/microsoft/Mnemis/blob/4552fed19bc0cde7b990a6ceb0365cd75b1b3453/global_selection/global_selector.py#L85-L94) confirms tag=true and summary=false; the call uses only generic `ModelSize.large`. |
| Learned reranking and ordering | Union System-1 and System-2 results, then rerank Episodes, Entities/Categories, and Edges separately with `Qwen3-Reranker-8B`. System-2 is unordered and therefore cannot simply be RRF-fused. | Exact input depth to the learned reranker, input serializer, inference settings, and immutable model revision are absent. |
| Final context | Top `k=10` Episodes, top `2k=20` Entities including Categories, and top `2k=20` Edges. The top-k sweep uses `k={5,10,30,50}`. | No final tokenizer or answer-prompt token ceiling is reported. |
| Answer and judge | GPT-4.1-mini backend; GPT-4.1-mini grader; official dataset judge prompts; binary 0/1. Released result config names deployment `gpt-41-mini-shortco-2025-04-14-Bing` and caps RAG at 10 and graph at 20. | The released deployment name is not a portable immutable model identity. This judge is not interchangeable with Swarm Brain's official GPT-4o judge track. |
| Cost | Whole LoCoMo dataset: base ingestion `3.87e7` prompt + `1.06e6` completion tokens and 1111.40 s; hierarchy ingestion `1.39e7` + `9.27e5` and 3873.26 s; global selection `1.37e6` + `1.21e5` and 3637.65 s. | Table 3 does not isolate answer-generation cost and warns that runtime depends on database latency and parallelism. |
| Decisive ablation | LoCoMo: S1 RAG 73.8; S1 Graph 81.6; S1 hybrid 89.1; S1 + Qwen3 reranker 89.1; S2 only 87.7; S1+S2 93.3. Full-system rerankers: Qwen3-0.6B 92.6, BGE-v2-M3 92.7, Qwen3-8B 93.3. Combined `k=5/10/30/50`: 92.2/93.3/93.9/93.4. | The high-confidence signal is **global selection + local retrieval**, not Qwen3 reranking alone. |

The released selector implements the top-down loop and one-hop hydration
directly ([selector](https://github.com/microsoft/Mnemis/blob/4552fed19bc0cde7b990a6ceb0365cd75b1b3453/global_selection/global_selector.py#L267-L319)) and releases the permissive "select all useful nodes" prompt
([prompt](https://github.com/microsoft/Mnemis/blob/4552fed19bc0cde7b990a6ceb0365cd75b1b3453/global_selection/prompts.py#L1-L23)). The repository itself says that only global selection, prompts, contexts,
and results are provided
([README](https://github.com/microsoft/Mnemis/blob/4552fed19bc0cde7b990a6ceb0365cd75b1b3453/README.md#L63-L66)). Exact end-to-end reproduction is therefore not possible from the public artifact.

### SmartSearch

| Aspect | Primary-source protocol | Missing or inconsistent detail |
| --- | --- | --- |
| Candidate generation | spaCy `en_core_web_sm`: proper noun weight 3, noun 2, verb 1, named-entity bonus +1. Exact substring retrieval; discovered person/org/location/event entities weight 2.5. PRF takes nouns/proper nouns present in at least two of the top 10 passages, weight 0.5, then performs one more grep pass. | Passage boundaries and exact text serialization are not specified. There is no released implementation. |
| Candidate depth | Score the retrieved candidate set. Mean grep candidates are 431 of 601 passages on LoCoMo and 120 of 494 on LongMemEval-S. The adaptive threshold is only applied after a fused top-`K=60` preselection. | `120` and `431` are means, not hard caps. No maximum candidate count is stated. Top-62 in the appendix is the average number fitting 2,000 words, not retrieval depth. |
| CrossEncoder | `mxbai-rerank-large-v1`, 435M DeBERTaV3, MS MARCO, SentenceTransformers pointwise scoring. The publisher identity is [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1). | Paper does not pin a model revision, library revision, dtype, batch size, CPU, or score activation. |
| ColBERT | Late-interaction ranker, run in parallel with the CrossEncoder. Weighted RRF uses `k=60`, `w_CE=0.7`, `w_CB=0.3`. | Figure 1 says `answerai-small-v1`, 110M; Appendix calls it `ColBERT v2`; the publisher's [`answerdotai/answerai-colbert-small-v1`](https://huggingface.co/answerdotai/answerai-colbert-small-v1) is 33.4M, not 110M. The exact model actually evaluated cannot be established from the paper. |
| Truncation/order | Fixed default: append passages in rank order to a 2,000-**word** limit. Adaptive: take RRF top 60, compute the maximum CE score within that preselected head, retain passage `d` when `CE(q,d) >= 0.03 * max CE`, then apply a 4,000-word ceiling. | The text sometimes calls these token budgets while tables distinguish words from model tokens. Tokenizer and context serializer are not pinned. |
| Reader/judge | LoCoMo main ablations: GPT-4o-mini answer and judge. LongMemEval ablations: Claude Sonnet 4.6 answer, GPT-4o-mini judge. Main LongMemEval: GPT-4.1-mini answer, GPT-4o-mini judge, same answer prompt as compared systems. Full 500, six types excluding abstention. | Model snapshots and exact external deployment identities are absent. Numbers from these tracks must not be mixed. |
| Cost | CrossEncoder and ColBERT run in parallel on CPU, stated wall time about 650 ms/query. Main LME context averages 3,392 model tokens. | CPU model and threading are absent, so 650 ms is not a portable budget. Ingestion/search CPU cost is not reported as a full accounting ledger. |
| Decisive ablations | LoCoMo: no reranker 76.8; MiniLM 84.7; bge-base 88.6; bge-large 90.7; bge-large+ColBERT 91.2; mxbai-large+ColBERT 91.9. The two-model fusion adds only 0.5 points over bge-large in the live table; the model-quality ladder supplies most of the gain. Query expansion adds 9.2 points on LME. | SmartSearch supports a strong CrossEncoder experiment, but does not prove that adding a second ranker will help an already strong session-level hybrid. |

Additional negative results worth preserving: equal CE/ColBERT weights, three-way
RRF, z-score/product fusion, cumulative-score pruning, and MMR did not beat the
chosen two-model configuration. The final indexed LoCoMo error analysis assigns
59% of errors to answer inference, 24% to reranking/budget, 12% to search misses,
and 5% to missing corpus evidence.

### Chain-of-Memory

| Aspect | Primary-source protocol | Missing or divergent detail |
| --- | --- | --- |
| Memory unit and retrieval | Paper: one dialogue turn per memory node with text, timestamp, role, and embedding; cosine-retrieve top `K=20` using `Qwen3-Embedding-8B`. | Released code first embeds whole sessions, keeps the top 10, embeds their turns, and then keeps 20 turns ([pipeline](https://github.com/Xiucheng-Xu/CoM/blob/52aa7ffd641059435c4585b6d9dad660518be635/src/com/pipeline.py#L124-L156)). This top-10 session gate is absent from the paper method. |
| Anchors | Initialize the top `L` retrieved nodes as separate chains. Released run default is `L=3`, top-k anchor sampling. | The paper method does not state the numerical `L`; it is only recoverable from released defaults. |
| Evolution score | For each chain, choose the remaining candidate maximizing `cos(candidate, query) * cos(candidate, chain_context)`. Released async path re-embeds the concatenated chain after every append. | Exact embedding input instruction, provider revision, batching, and numerical precision are not pinned. |
| Truncation | Stop if current best score `< beta * previous_appended_score`; `beta=0.5`. Candidates are removed only from the current chain and may be reused in another chain. | Released `max_chain_length=20` is assigned to `_` and never enforced ([filter](https://github.com/Xiucheng-Xu/CoM/blob/52aa7ffd641059435c4585b6d9dad660518be635/src/com/filter.py#L204-L264)); the top-20 pool is the practical bound. |
| Context order | Sort chains by anchor-query score, concatenate chain nodes in evolution order, emit `=== Evidence Chain N ===` blocks. Released code does not deduplicate nodes across chains. | The paper does not define a separate hard prompt-token ceiling. Duplicate evidence can affect both tokens and attention. |
| Reader/judge | GPT-4o-mini or Qwen3-32B Non-Thinking as both answer model and judge; standardized prompts; temperature 0 in released config. | Released example leaves embedding and judge model IDs blank. Immutable model/provider revisions are absent. This is not the official LongMemEval judge protocol. |
| Cost | LME GPT-4o-mini: 74.20, 8.24k tokens, 2730 s; Qwen3-32B: 76.40, 8.81k, 2002 s. Paper labels token values total end-to-end consumption in thousands, but their scale and discussion read like per-query averages; the unit interpretation is ambiguous. | Hardware, concurrency, provider latency, and whether the table includes every embedding call are not fully specified. |
| Decisive ablations | Qwen LME: turn RAG 66.0 vs CoM 76.4; query-only gate 66.15, context-only 69.35, weighted average 72.55, product 76.40; `beta=.3/.5/.7` gives 73.95/76.40/70.14. Cross-encoder baseline using `bge-reranker-base` scores 63.20 vs CoM 76.40. | The reranker baseline's expanded candidate-pool depth is not stated and its code is not released. It is supportive, not an exact head-to-head recipe. |

The released chain loop and merge order are directly auditable in
[`filter.py`](https://github.com/Xiucheng-Xu/CoM/blob/52aa7ffd641059435c4585b6d9dad660518be635/src/com/filter.py#L79-L264). The code/paper retrieval divergence must be an explicit experimental axis rather than silently choosing one.

### MAGMA

| Aspect | Paper protocol | Released-code divergence |
| --- | --- | --- |
| Representation | Four orthogonal semantic, temporal, causal, and entity graphs over event nodes. | The release includes many additional node/link subtypes and dataset-oriented heuristics; it is not a minimal implementation of the four equations. |
| Anchors | Dense + lexical + temporal signals fused with RRF `k=60`; vector top 20; temporal interval is a hard filter. | Actual query adds a full scan and uses candidate depths 30/30/40 for multi-hop, 15/15/20 for temporal, and 20/20/25 otherwise ([query path](https://github.com/FredJiang0324/MAGMA/blob/467cb70b67ac337b22fdb42194d37c04ad701b62/memory/query_engine.py#L682-L760)). |
| Traversal | Heuristic beam search; transition `exp(lambda1 * structural_alignment + lambda2 * cosine)`; cumulative decay `gamma`; retain top beam nodes per step; topological order by query intent. Appendix: depth 5, max nodes 200, drop threshold .15, `lambda1=1`, `lambda2=.3-.7`. | `gamma` and beam width are not reported. The released `_probabilistic_beam_search` has defaults beam 10, max visited 50, lambda .6/.4, but has no call site. Actual query calls `_adaptive_graph_traversal`, a BFS ([call](https://github.com/FredJiang0324/MAGMA/blob/467cb70b67ac337b22fdb42194d37c04ad701b62/memory/query_engine.py#L797-L813), [BFS](https://github.com/FredJiang0324/MAGMA/blob/467cb70b67ac337b22fdb42194d37c04ad701b62/memory/query_engine.py#L1099-L1195)). |
| Actual bounds | Paper parameter ranges: similarity `.10-.30`; entity weights `2.5-6`; temporal `.5-4`; causal `3-5`; phrase `2.5-5`. | Released BFS uses intent depths 4-12, max nodes 800, neighbor fanout 8/10, and at most 400 neighbor encodings. The code also contains hard-coded temporal keyword-to-session mappings, which must never enter a benchmark-independent Swarm Brain policy. |
| Context | Temporal queries sorted by time; causal queries topologically sorted; each node includes timestamp, content, and reference ID; low-probability nodes summarized into brevity codes. | No exact token ceiling, summarization algorithm, or brevity-code protocol is specified. Released formatter behaviour differs from the paper's algorithm. |
| Models and judging | Default `all-MiniLM-L6-v2` 384-d, optional `text-embedding-3-small` 1536-d; GPT-4o-mini temperature 0 for inference and judge; custom continuous 0-1 rubric. | Model revisions and a protocol-compatible official LongMemEval judge are absent. The reported 61.2 LME average is not comparable to a binary official-judge score. |
| Cost and ablation | LoCoMo build 0.39 h, 3.37k tokens/query, 1.47 s/query. Full MAGMA .700; no adaptive policy .637; no causal .644; no temporal .647; no entity .666. Single graphs: causal .590, temporal .577, entity .531. | Because the executed retrieval does not match the paper algorithm, these results do not identify which published beam-search parameters caused the gains. |

MAGMA is useful evidence for **testing relation views one at a time**, but its
public artifacts do not support an exact reproduction claim.

## Alignment with the current Swarm Brain architecture

| Capability | Current state | Paper alignment and implication |
| --- | --- | --- |
| Hybrid retrieval | Exact, lexical, fuzzy, dense, temporal, and graph lanes; weighted RRF `k=60`; bounded lane depths. | Strong alignment with the System-1/SmartSearch/MAGMA anchor stage. No need to replace this before testing compilation. |
| Semantic retrieval evidence | A prior full-500 LongMemEval-S session-level diagnostic with Qwen3-Embedding-0.6B measured final Recall@10 `.976`, MRR@10 about `.906-.908`. Its saved artifact predates the current cleaned-dataset/compiler digest and does not pass the present SOTA gate. | This remains a directional diagnostic only. It cannot be compared with paper QA scores or with turn-level reranking results, and it must be rerun before a current claim. |
| Candidate granularity | Production LongMemEval retrieval stores one entire conversation session as one memory. The evaluation tree now also has an immutable, source-byte-bound turn projection with IDs `(question_id, session_position, turn_position)` and exact timestamp/role/content serialization. A benchmark-only E1-A boundary can validate and fuse externally observed lexical/dense turn rankings at fixed depths `128/128`, weights `3/4`, RRF `k=60`, and cap `128`. | The F0 projection, deterministic E1-A fusion, and exact full-prompt turn packer are implemented without changing production memory. They do not execute or authenticate the external scorers/tokenizer; no full ranking receipt or model-backed outcome artifact exists yet. A learned reranker over only ten already-good sessions remains a different experiment and may repeat Mnemis's neutral S1 result. |
| Deterministic reranking | `relevance_reranked` exists, but planner alpha is zero everywhere after LongMemEval Recall@10 regressed `.976 -> .971`. | Correctly not shipped. This negative result does not test a learned CrossEncoder. |
| Learned reranking | The working tree contains a bounded, identity- and receipt-verifying score-only port, an artifact-pinned local JSONL adapter, optional `RetrievalService` integration for at most 128 candidates, and a fail-closed paired-report compiler for a distinct session-level, window-50, `alpha=1` retrieval control. It is constructor-disabled by default. Benchmark-only E1-B/C/D now validate CE logits, CE/ColBERT rank fusion, and top-60 relative-threshold selection over the exact E1-A pool. | The deterministic mathematics and evidence boundaries are implemented, but no Qwen3, CrossEncoder, ColBERT, reader, or judge run has been executed. The paired QA compiler can validate structural evidence; unsigned receipts cannot authorize serving or establish a SOTA result. |
| Packing | Greedy whole-memory packing is default; experimental facility-location lost on the full-500 diagnostic. Exact full-prompt tokenizer accounting is available in the evaluation path. | Keep the measured greedy negative/positive controls. Add score-adaptive filtering as a separate stage; do not revive MMR/facility objectives merely because they appear sophisticated. |
| Graph retrieval | Canonical typed links, exact rehydration, and bounded one/two-hop expansion. LongMemEval session memories have no links, so the graph lane returns zero candidates there. | This is a flat relation substrate, not Mnemis's category hierarchy or MAGMA's four materialized views. |
| Adaptive policy | A fail-closed, rank-independent adaptive policy contract exists but explicitly is not wired to serving. | Good safety alignment with MAGMA's routing thesis, but it needs a held-out graph prior and one-view-at-a-time evidence. |
| Context organization | Serving still renders/packs sessions as independent blocks, normally chronological after selection. The benchmark tree now contains pure E2-A..E2-E organization over one fixed 20-turn pool, including `L=3`, raw query/context products, `beta=.5`, released-code stop semantics, bounded traces, a separate cross-chain-dedup cell, and exact 4,096/8,192/16,384 full-prompt turn packing. | The CoM-shaped organizer and packer are offline transfer controls, not wired to serving. Similarities and tokenizer observations remain externally attested, and no model-backed exact-budget reader QA exists. The artifacts explicitly disclaim exact paper parity and quality improvement. No hierarchical global-selection output exists. |

The important diagnosis is a **compilation/granularity gap**, not a raw-recall
gap. At session granularity, the current top ten already contain a gold session
for nearly every question, but the unbounded bundle averages about 32,750
estimated tokens. The papers suggest improving which turn-level evidence
survives and how it is ordered before asking for more retrieval lanes.

## Frozen experiment design, ordered by risk-adjusted expected value

### F0 — Freeze the comparison boundary first

- Dataset: official cleaned LongMemEval-S, all 500 questions, SHA-256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Add an evaluation-only immutable turn projection. ID is
  `(question_id, session_position, turn_position)`; serialized text is exactly
  timestamp, role, and original content. Preserve parent session ID/date for
  gold-session metrics and provenance. Do not mutate production memories.
- Use one frozen reader, answer prompt, judge, judge prompt, tokenizer revision,
  and serializer across all Swarm Brain cells. Report paper-protocol tracks
  separately; never compare Mnemis's GPT-4.1-mini grader scalar with the
  official GPT-4o judge scalar.
- Primary delivered-context budget: exactly 8,192 reader tokens, including the
  full prompt. Also report evidence-only curves at 4,096 and 16,384 tokens.
  Whole turns are indivisible; skip an oversized turn and continue.
- Primary metrics: paired end-to-end QA accuracy; any/all gold-session in
  delivered context; answer-session MRR; exact mean/p50/p95 prompt tokens;
  reranker and embedding calls; ingestion/query tokens separately; p50/p95
  latency; and per-question-type deltas.
- Freeze all values below before reading end-to-end results. A changed value is
  a new named experiment version, not a continuation of the same run.

### E1 — SmartSearch ranking/compilation control

This runs first because it is the cheapest strong learned-ranking control and
is needed to interpret any later CoM gain.

Frozen model identities observed from publisher registries on 2026-08-09:

| Component | Immutable revision for the Swarm Brain experiment |
| --- | --- |
| CrossEncoder | [`mixedbread-ai/mxbai-rerank-large-v1@98f655841d5caf0b16eaff79c2b4ca109d920d17`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/tree/98f655841d5caf0b16eaff79c2b4ca109d920d17) |
| ColBERT hypothesis | [`answerdotai/answerai-colbert-small-v1@c72aa89bc61afdd85373643f3a1a75b2aad6e0fe`](https://huggingface.co/answerdotai/answerai-colbert-small-v1/tree/c72aa89bc61afdd85373643f3a1a75b2aad6e0fe) |

The ColBERT choice is an explicit Swarm Brain hypothesis, not a claim that this
is the exact inconsistent SmartSearch artifact.

Candidate protocol:

- Run current lexical+dense turn retrieval and weighted RRF `k=60`; freeze the
  first 128 unique, canonically hydrated turns. The 128 cap matches the bounded
  learned-reranker contract. Record pre-cap and post-cap gold-session recall.
- Score every frozen candidate with the CrossEncoder. Convert its single logit
  with sigmoid to `[0,1]`; tie-break by prior RRF rank then candidate ID.
- When ColBERT is enabled, fuse CE and ColBERT ranks using RRF `k=60`, weights
  `.7/.3`; tie-break identically.
- Adaptive cell: take fused top 60, compute `max(CE)` within that head, keep
  candidates satisfying `CE >= .03 * max(CE)`, then greedily pack whole turns
  to the exact 8,192-token prompt ceiling in fused-rank order.
- Separate paper-shape diagnostic: the same top-60/alpha rule with a 4,000-word
  ceiling. Label it word-budget evidence only; do not substitute it for exact
  model-token accounting.

Frozen cells:

| Cell | Ranking and selection |
| --- | --- |
| E1-A | Existing turn-level weighted RRF; exact-token greedy pack. |
| E1-B | CrossEncoder only; exact-token greedy pack. |
| E1-C | CE + ColBERT weighted RRF; exact-token greedy pack. |
| E1-D | E1-C plus top-60, alpha `.03` score-adaptive filtering. |

Do not add PRF/entity expansion in these four cells. If E1-D wins, run expansion
as E1-v2 so ranking and candidate-generation effects remain attributable.

### E2 — Chain-of-Memory organization

This has the largest paper-reported QA upside, but runs second because it needs
the F0 turn compiler and an independent reranker control.

Primary transfer experiment holds initial retrieval fixed: use the first 20
turns from E1-A's frozen RRF ranking. Frozen CoM parameters are `K=20`, `L=3`,
the first three candidates in that immutable RRF order as anchors,
multiplicative gate, concatenated-chain re-embedding, and `beta=.5`. Do not
reselect anchor membership by query cosine; the separate paper-text diagnostic
starts from a cosine-sorted pool, so its first three are already cosine top-3.
A candidate is removed only within its chain; cross-chain reuse and duplicate
rendering are preserved for the parity cell. Completed chain blocks are ordered
by anchor-query score. For the fair-budget cell, append whole chain turns in
that order until the exact 8,192-token prompt ceiling is reached.

Frozen cells:

| Cell | Organization |
| --- | --- |
| E2-A | Top 20 turns in retrieval order, no chain evolution. |
| E2-B | Query-only successor score, beta `.5`. |
| E2-C | Product gate, no adaptive path truncation; chain may consume all 20 candidates. |
| E2-D | Full product gate + beta `.5`. |
| E2-E | E2-D with cross-chain deduplication; product adaptation, not paper parity. |

Two diagnostics keep reproduction claims honest:

- Paper-text diagnostic: global turn cosine retrieval, top 20, using
  [`Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af).
- Released-code diagnostic: top 10 sessions by cosine, then top 20 turns within
  those sessions, same chain parameters. Report it as code parity, not paper
  parity.

The primary transfer cell should retain
[`Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3)
to isolate organization from an 8B model swap. Do not combine the model and
algorithm changes in the headline delta.

### E3 — Compose only demonstrated winners

If and only if E1 and E2 each improve the same frozen QA track, run one
composition without retuning: E1-D selects the head, its first 20 turns seed
E2-D, and the exact 8,192-token pack remains unchanged. This tests
**independent relevance -> context-conditioned organization**. Do not introduce
query expansion, hierarchy, or graph-view routing in this cell.

### E4 — Mnemis global-selection experiment and reranker control

First run Qwen3 reranking over System-1 as a reproduction control, not as the
expected winner. Freeze
[`Qwen/Qwen3-Reranker-8B@77d193c791ed757ca307ee72715aa132723da912`](https://huggingface.co/Qwen/Qwen3-Reranker-8B/tree/77d193c791ed757ca307ee72715aa132723da912), score the same candidate IDs as E1, and change no candidate generation or packing.

An exact Mnemis hierarchy replication is blocked by missing construction
parameters. If author clarification is unavailable, the only honest runnable
alternative is a separately named **SB-HGS-v1** hypothesis, frozen before any
QA score:

- minimum 4 children/category, maximum 12;
- at most 2 category parents/child;
- maximum 4 hierarchy layers; stop when a layer does not reduce node count;
- selector sees UUID, name, and tag, never summary;
- begin with all top-layer categories, no top-k; preserve the
  `get_all_children` shortcut;
- hydrate one-hop episodes, edges, and neighbor entities;
- compare S1 only, S1 + Qwen rerank, S2 only, and S1+S2;
- final type caps 10 Episodes, 20 Entities/Categories, 20 Edges.

These invented bounds make SB-HGS-v1 executable and bounded; they must never be
reported as Mnemis's parameters. Hierarchy construction and query-time costs
must be reported separately.

### E5 — MAGMA-inspired relation-view ablation

Do not claim MAGMA reproduction. Run a named **SB-MG-v1** offline hypothesis
only after E1-E4:

- semantic, temporal, causal, and entity projections are materialized from
  provenance-backed canonical links;
- compare each single view, all views with router off, and all views with router
  on;
- freeze RRF `k=60`, vector top 20, beam width 10, depth 5, max nodes 200,
  relative drop `.15`, `lambda1=1`, `lambda2=.5`, and cumulative decay
  `gamma=1`;
- never include dataset-specific names, month-to-session tables, or answer IDs;
- keep hard scope/lifecycle/trust/time hydration before and after expansion.

Beam width and gamma are Swarm Brain choices because the paper omits them. This
experiment tests the multi-view thesis, not the published executable.

## Promotion rule

Promote no serving change from an offline recall increase alone. LongMemEval
has already influenced this repository's design, so the frozen full-500 run is
a confirmation/paper-comparison track, not a fresh held-out set. A candidate
must satisfy all of the following on that track:

1. Paired end-to-end QA delta has a stratified paired-bootstrap 95% confidence
   interval whose lower bound is above zero.
2. No question type regresses by more than two percentage points; report raw
   counts because the preference slice is small.
3. Any-gold-in-context is non-inferior and all-gold-in-context, exact prompt
   tokens, and latency are reported at the same budget.
4. The candidate is not dominated on QA, p95 delivered tokens, p95 latency, and
   construction-plus-query cost.
5. Model IDs, immutable revisions, tokenizer artifacts, candidate-pool hashes,
   prompts, and per-case outputs are replayable. Any rerun or parameter change
   receives a new experiment ID and is disclosed.
6. Before a serving promotion, a separately preregistered task or corpus that
   did not motivate these choices must show the same directional QA benefit.

## Bottom line

Swarm Brain is already aligned with the papers' strong System-1 substrate:
hybrid lanes, weighted RRF 60, semantic embeddings, typed temporal/graph
signals, bounded hydration, and exact-budget evaluation. It is not yet aligned
with their strongest **context compilation** mechanisms. The current high
session Recall@10 masks a coarse-granularity, high-token bundle.

The immutable turn compiler, deterministic E1-A/B/C/D boundaries, exact
full-prompt packer, learned-score port, structural paired evidence compiler,
and pure CoM organization cells are now built.
The highest-value next step is therefore to generate the frozen external turn
rankings, run the SmartSearch CE/CE+ColBERT controls, run CoM over the identical
candidate head, and measure same-budget reader/judge QA. Compose only winners,
then test source-preserving multi-key representations before expensive
hierarchy or graph traversal. Treat Qwen3/Mnemis hierarchy and MAGMA views as
distinct later experiments. That ordering follows the ablation evidence rather
than paper headline model names.
