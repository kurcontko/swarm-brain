# SOTA agent memory & retrieval — research dump 2026-08-07

Live web research (BrightData, four parallel research agents, 2026-08-07) on
state-of-the-art agent memory, graph retrieval, hybrid fusion/abstention,
embeddings, context engineering, and evaluation — mapped to Swarm Brain's
measured gaps. Complements the DB-focused
[PostgreSQL/CockroachDB SOTA dump](sota-retrieval-postgresql-cockroachdb-2026-08-02.md);
this document covers the layers above the database. The four agents' full,
unabridged reports are preserved in
[raw reports](sota-agent-memory-retrieval-2026-08-07-raw-reports.md).

Anchoring measured problems (from [retrieval benchmark](../retrieval-benchmark.md)):

- **P1 — no abstention.** No-answer recall 0.00: all six out-of-corpus queries
  return ten hits with top-1 public score 1.00, because the public score is
  anchored to RRF rank, not relevance. An unmeasured raw-cosine floor of 0.2
  regressed a live gate; floors are now 0.0.
- **P2 — placeholder dense lane.** LongMemEval-S fused Recall@10 0.768 /
  MRR@10 0.574 sits below lexical-only 0.885 / 0.792 because the deterministic
  hash embedder drags fusion down. The 1024-dim provider-neutral slot needs a
  real model.
- **P3 — graph lane precision.** Graph expansion lifts overall Recall@10
  0.863 → 0.882 and `multi_evidence` MRR 0.32 → 0.61, but costs −0.10 MRR@10
  overall on decoy-heavy queries. It needs per-query gating.
- **Unbuilt lanes:** evidence/source-chunk expansion, deterministic
  handoff/checkpoint context, temporal/entity routing, diversity/context
  packing, reranker, persistent trace sink; 90-memory/40-query gold corpus is
  too small.

## Executive summary — ranked recommendations

1. **Fix abstention with a two-channel design, not a score floor.** Keep RRF
   (or TM2C2 convex fusion) purely for ordering; carry per-lane raw scores
   (raw cosine, ts_rank, trigram similarity, exact-hit flag) as a parallel
   evidence channel and gate no-answer on that channel with an empirically
   calibrated threshold — conformal calibration gives a distribution-free
   guarantee and works today, even with the hash embedder. Generate the
   no-answer eval set with the UAEval4RAG recipe instead of hand-writing it.
2. **Fill the 1024-dim dense slot with Qwen3-Embedding-0.6B** — the only
   leading open model that is simultaneously native-1024-dim, Apache 2.0,
   instruction-aware, code-capable, 32K-context, and CPU-serveable, with a
   same-family reranker. Re-calibrate every threshold on the swap.
3. **Gate the graph lane per query with a learned complexity weight.** The
   decoy regression is a published, named phenomenon with a proven fix:
   EA-GraphRAG makes the graph lane's fusion weight a per-query complexity
   score from a tiny classifier trained on ~200 "graph helped / hurt"
   disagreement samples — which the benchmark harness already produces — and
   beats both always-graph and never-graph on every dataset. Combine with
   HippoRAG 2's skip-graph fallback (no seed passes the relevance filter →
   return direct RRF untouched), small default graph weight, and CS-RAG's
   per-hop sufficiency check. The same classifier doubles as the lane router
   (identifier → exact/trigram, conceptual → dense/FTS, temporal →
   valid-time predicates).
4. **Add a write-time distillation pass (Observer/Reflector).** The
   LongMemEval leaderboard's top system (Mastra OM, 94.87%) attributes its win
   to dated, prioritized, compressed observations — distilled memories beat
   oracle raw context. Map ADD/UPDATE/DELETE/NOOP write gates onto
   append + supersession.
5. **Extract referenced dates as a third indexed timestamp** (beyond
   valid/recorded time) and emit valid-time ranges inline in recalled context —
   the top drivers of temporal-reasoning gains across Zep and Mastra OM.
6. **Build the packing lane as budgeted submodular selection, not plain MMR**,
   and add answer-in-context (did the gold span survive into the packed
   bundle?) to the benchmark harness — packed-bundle survival separates
   exact-match by 4.6× even at equal recall.
7. **Scale the gold corpus from traces, not authorship.** Trajectory-derived
   gold (which memories a successful run actually consulted) plus LLM-generated
   qrels with a human-audited slice and `_abs`-style unanswerable questions;
   report quality-per-token. Adopt LongMemEval-V2 early — its coding-agent
   environment-experience framing is Swarm Brain's true benchmark and almost
   nobody has published on it.
8. **Expose iterative retrieval affordances** (search / read-expand /
   follow-links) beside the deterministic bootstrap bundle — agentic
   evidence-gathering measured 72.5% vs 48.5% for one-shot RAG memory on
   LongMemEval-V2.

## Graph-based retrieval (2025–2026) — and the fix for P3

### The decoy regression is a published, named phenomenon

- **GraphRAG-Bench / "When to use Graphs in RAG"**
  ([arXiv:2506.05690](https://arxiv.org/abs/2506.05690), ICLR 2026;
  [benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)) — the
  definitive study, 4,076 questions, 11 systems. Verbatim: "Basic RAG is
  comparable to or outperforms GraphRAG in simple fact retrieval tasks...
  GraphRAG's extra graph-based processing may introduce redundant or noisy
  information for simpler queries." Prior measurements it cites: GraphRAG
  −13.4% accuracy on Natural Questions vs vanilla RAG, −16.6% on
  time-sensitive queries, 2.3× latency. The crossover: at knowledge-breadth
  ≈1.3 / reasoning-depth ≈1.8 plain RAG wins; from breadth ≥2.6 / depth ≥5
  graph wins decisively — the exact shape of the measured Swarm Brain result
  (multi_evidence MRR 0.32→0.61 up, overall MRR −0.10 down). Also: graph
  *quality* (density/clustering) predicts value — HippoRAG 2's degree-8.75,
  clustering-0.657 graph outperformed sparse degree-1.48 graphs. Computing
  avg degree and clustering of the memory-link graph is a trivial SQL health
  metric worth adding.
- **CS-RAG** ([arXiv:2603.14828](https://arxiv.org/abs/2603.14828), Mar 2026)
  formally names the mechanism: "spurious noise induces retrieval drift
  toward plausible but unsupported triples," and "incomplete information
  leads to retrieval hallucination by forcing continuation through
  under-supported graph structure" — which is what a fixed always-2-hop
  traversal does on decoy queries. Its mitigation is a **per-hop sufficiency
  check**: expand hop 2 only if hop 1 actually produced above-threshold
  evidence.
- **GraphRAG-FI** ([arXiv:2503.13804](https://arxiv.org/abs/2503.13804)):
  two-stage filtering — coarse traversal-time gate, then re-score expanded
  candidates against the query with the strongest scorer *before* they enter
  final fusion, rather than admitting them on the traversal gate alone.

### HippoRAG 2 — the flagship whose whole design is the mitigation catalog

[arXiv:2502.14802](https://arxiv.org/abs/2502.14802) (ICML 2025,
[code](https://github.com/OSU-NLP-Group/HippoRAG)). Built explicitly because
prior structure-augmented RAG "drops considerably below standard RAG" on
simple factual queries. First graph system that also wins on simple NQ (78.0
vs 75.4 dense) while gaining +5–14 Recall@5 on multi-hop. Its mitigations, in
order of transplantability:

1. **Skip-graph fallback**: an LLM "recognition memory" filter prunes seed
   triples; if the filter empties the seed set, the system falls back to pure
   dense retrieval — no graph. Analog: filter seeds against the query before
   traversal; if none pass, return direct RRF untouched.
2. **Heavy seed downweighting**: passage-node seeds enter PPR with weight
   factor **0.05**; the sweep {0.01…0.5} shows monotone degradation as graph
   influence rises past a small optimum. The graph lane's fusion weight
   should be small by default — downweight, don't equal-weight.
3. **Query-to-triple linking** (+12.5 Recall@5 over NER-to-node): gate seeds
   with query-contextualized relevance, not entity presence.
4. Error analysis warning: 26% of its failures were *over-filtering* seeds on
   multi-hop queries — keep the gate threshold moderate.

Cost note: its connectivity is mostly cheap edges — 1.13M synonym
(embedding-similarity) edges vs 140k LLM-extracted relation edges (8:1).

### EA-GraphRAG — the direct blueprint for per-query graph gating

[arXiv:2602.03578](https://arxiv.org/abs/2602.03578) (Feb 2026). No LLM in
the router: ~85 syntactic/lexical features (parse complexity, entity counts,
dependency distances, question-type markers) → tiny MLP → complexity score
s(q) ∈ (0,1), trained as "will graph beat dense on this query?" on **only
~200 disagreement samples** (queries where the two pipelines disagreed).
Routing: s(q) ≥ τ_H → graph only; ≤ τ_L → dense only (40× faster path);
between → **complexity-weighted RRF where the graph lane's weight IS s(q)**.
Measured: beats both HippoRAG 2 and ColBERTv2 on every dataset including
single-hop NQ (69.1 vs 68.5/65.3 mixed-benchmark accuracy).

Transplant to Swarm Brain: the final weighted RRF already exists — make
`w_graph = s(q)` from a logistic regression over ~10 domain features (query
length, distinct symbol/path/commit count, conjunction/causal markers, top-1
exact/FTS lane score as a "single-fact" signal, dense–lexical agreement).
The benchmark harness already produces the training labels (per-query
graph-on vs graph-off RR deltas) — Adaptive-RAG
([NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)) validates
exactly this silver-label recipe. Below τ_L, skip traversal entirely and
reclaim the lane's latency.

### PPR vs bounded BFS, and offline graph projections

- No paper directly A/Bs PPR against gated bounded BFS; the evidence pattern:
  PPR wins on dense high-quality graphs with filtered seeds but leaks mass to
  hub nodes (50% of HippoRAG 2's failures had correct seeds and bad top-5,
  with no per-node query gate possible mid-walk); gated BFS wins for dynamic
  graphs, low latency, and precision control — the current fixed-depth
  SQL traversal is the right regime for a live memory graph. A middle path if
  wanted later: Andersen-Chung-Lang forward-push approximate PPR is
  implementable application-side over SQL adjacency reads with a residual
  threshold ([local-algorithm convergence result](https://openreview.net/forum?id=n28wnc2QTc)).
- **LazyGraphRAG**
  ([Microsoft, Nov 2024](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)):
  concept graph from noun-phrase co-occurrence — **zero LLM index cost** —
  plus a query-time relevance-test budget with iterative deepening that stops
  descending after z consecutive irrelevant communities. Two steals: cheap
  co-occurrence links are competitive with LLM extraction, and
  **early termination after N consecutive gate failures per seed** is a decoy
  defense the traversal can adopt today.
- **Offline projections that fit "async job → SQL table"**: Zep/Graphiti uses
  **label propagation instead of Leiden** specifically because a new node
  joins the plurality community of its neighbors in one step (one SQL query
  per insert, periodic full refresh) — write `community_id` back to a table
  and use community disjointness as another skip-graph signal. Nightly
  node2vec over memory links (provable community recovery,
  [NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/015a8c69bedcb0a7b2ed2e1678f34399-Abstract-Conference.html);
  structural+semantic hybrid beats either alone,
  [arXiv:2605.18410](https://arxiv.org/abs/2605.18410)) stored in a
  `graph_embedding` column turns "graph-nearby" into an O(1) vector check —
  a decoy-resistant alternative to runtime traversal on borderline queries.
- **Precomputed PPR** for stable hub memories can be materialized offline
  (`seed_id, node_id, ppr_score`), as in the
  [AWS HippoRAG reference](https://aws.amazon.com/blogs/machine-learning/hipporag-neurobiologically-inspired-rag-using-amazon-bedrock-amazon-neptune-and-personalized-pagerank/).

### Temporal edges and typed-edge weights

- **Graphiti's edge invalidation is itself a decoy mitigation**: in an
  agent-swarm memory the dominant decoy class is *superseded* memories about
  the same artifact; time-scoping edges (t_invalid set when a contradicting
  edge arrives) removes them from the expansion frontier automatically.
  Swarm Brain already bounds edges by `recorded_at` — extending the graph
  lane to respect supersession state on *edges*, not just endpoint hydration,
  closes the loop.
- **Typed-edge weight learning has weak published support** — the practice
  everywhere (HippoRAG 2, Graphiti, LightRAG) is hand-set. Closest work:
  ReG/Weak-to-Strong GraphRAG
  ([arXiv:2506.22518](https://arxiv.org/abs/2506.22518)) shows outcome
  feedback ("did this path help answer?") suffices to clean path supervision.
  Practical: fit the existing per-type weights and hop-decay constant offline
  by coordinate ascent against benchmark MRR — same disagreement-set
  infrastructure as the router; expect second-order gains vs seed gating.

### Write-time graph construction

Deterministic **shared-artifact links (same path / symbol / commit) are
co-occurrence edges over a controlled vocabulary with near-zero
false-positive rate — higher precision than LLM-extracted relations**, which
carry CS-RAG's spurious-noise risk. Since memories already carry normalized
paths/symbols/commits in `retrieval_exact_terms`, these should be the
highest-weight edge types, generated automatically at publish. Tiering per
KET-RAG ([arXiv:2502.09304](https://arxiv.org/abs/2502.09304)): expensive
LLM/pseudo-query links (HopRAG,
[ACL Findings 2025](https://aclanthology.org/2025.findings-acl.97.pdf)) only
for high-value hub memories; cheap deterministic links everywhere. LightRAG's
independent-eval collapse (F1 6.6 in HippoRAG 2's benchmark — its generated
index text pollutes results) is the cautionary tale: use the graph only to
*reach and rank* stored memories, never to inject graph-derived generated
text — which the current design already respects, and PAGE-RAG
([arXiv:2607.19301](https://arxiv.org/abs/2607.19301)) elevates to a
principle: link provenance is a prior, never a substitute for query-text
evidence.

## Agent memory systems (2025–2026)

### Zep / Graphiti — closest cousin (bitemporal knowledge graph)

[arXiv:2501.13956](https://arxiv.org/abs/2501.13956) (Jan 2025),
[github.com/getzep/graphiti](https://github.com/getzep/graphiti).
Three-layer graph: episode subgraph (raw non-lossy input), semantic entity
subgraph (extracted entities + fact edges), community subgraph (label
propagation, chosen over Leiden so new nodes join communities incrementally).
Write path: extraction over current + last 4 messages, reflexion-style second
pass, entity dedup via embedding + fulltext candidates + LLM resolution; edge
dedup constrained to edges between the same entity pair (bounds cost). Four
timestamps per edge (t′_created/t′_expired transactional, t_valid/t_invalid
validity); on contradiction the old edge gets t_invalid = new edge's t_valid.
Read: cosine + BM25 + BFS seeded by recent episodes, then rerankers (RRF, MMR,
episode-mentions frequency, node-distance, cross-encoder). Context template
emits facts with their valid/invalid date ranges. DMR 94.8%; LongMemEval-S up
to +18.5% accuracy, ~90% latency cut vs full context (though independent runs
place Zep at 71.2% with gpt-4o, well behind 2026 systems).

Steals for Swarm Brain: entity-pair-constrained conflict detection;
episode-mentions and node-distance as cheap extra RRF lanes (reference
frequency is a natural swarm salience/trust prior); valid-time ranges inline
in recalled context; reflexion-style two-pass extraction.

### Mem0 — the canonical write-gate

[arXiv:2504.19413](https://arxiv.org/abs/2504.19413) (Apr 2025). Extraction
phase (LLM over new messages + rolling summary) then update phase: for each
candidate, retrieve top-similar existing memories and choose
**ADD / UPDATE / DELETE / NOOP** — eager write-time consolidation. Claims 26%
relative over OpenAI memory on LOCOMO, 91% lower p95, >90% token savings
(disputed by Zep; LoCoMo itself now distrusted — see evaluation section).
Mapping onto the append-by-default contract: ADD→new observation, UPDATE→new
observation + supersession link, DELETE→supersession-with-tombstone, NOOP→drop
duplicate. The existing hybrid retrieval pipeline is the candidate generator
for the gate, so the marginal cost is one LLM decision per publish.

### Letta / sleep-time compute

[arXiv:2504.13171](https://arxiv.org/abs/2504.13171),
[letta.com/blog/sleep-time-compute](https://www.letta.com/blog/sleep-time-compute/)
(Apr 2025). A sleep-time agent shares memory blocks with the primary agent and
reorganizes them offline; ~5× reduction in test-time compute for equal
accuracy on Stateful GSM-Symbolic; cost amortizes across queries. CRDB-native
version: scheduled jobs for contradiction sweeps (entity-pair partitioned),
durative-fact derivation, topic summaries, pre-materialized as-of-now snapshot
tables. The bitemporal ledger makes all derived state safely re-derivable.

### A-Mem (NeurIPS 2025) — agentic link generation

[arXiv:2502.12110](https://arxiv.org/abs/2502.12110) (~899 citations).
Zettelkasten-style notes; an LLM proposes links between each new note and its
retrieval neighbors, and can trigger "memory evolution" — rewriting older
notes. The link-generation pass would densify the typed-link graph that the
1–2 hop expansion walks; evolution, which is an anti-pattern for the audit
model, becomes clean on a bitemporal substrate: evolution events are new
versions with supersession. That combination is unpublished territory.

### Mastra Observational Memory — current LongMemEval SOTA, no retrieval

[mastra.ai/research/observational-memory](https://mastra.ai/research/observational-memory)
(Feb 2026). Observer converts history into dense, dated, priority-tagged
observations at a token threshold (3–6× text compression, 5–40× on tool-call
traces); Reflector restructures the log at a second threshold (merge related,
drop superseded, keep dates). Context is stable/append-only, hence
prompt-cacheable; ~30k average tokens. **Three-date model**: observation date,
referenced date ("flight is Jan 31"), computed relative date. Results
(longmemeval_s): gpt-4o 84.23% — **beats the 82.4% oracle**, i.e. distilled
observations outperform raw ground-truth sessions; gpt-5-mini 94.87% (highest
recorded); temporal-reasoning 95.5%.

Imports: (1) referenced-date extraction as a third indexed timestamp — the
single highest-leverage temporal feature; (2) token-threshold Observer as the
distillation policy for worker transcripts before publishing to the shared
store; (3) the oracle-beating result justifies aggressive consolidation as a
quality play, not just a cost play.

### Hindsight — fact/experience/belief separation

[arXiv:2512.12818](https://arxiv.org/abs/2512.12818) (Dec 2025). Four logical
networks — world facts, agent experiences, entity summaries, evolving
beliefs — separating evidence from inference; retain/recall/reflect
operations. 91.4% LongMemEval (gemini-3-pro), 89.61% LoCoMo. For Swarm Brain:
add a `belief` memory kind with mandatory evidence links, revised only by
reflection/consolidation jobs — extends the existing evidence/trust model to
traceable inference.

### MemOS, MIRIX, MAGMA, Memory-R1, cognee — shorter notes

- **MemOS** [arXiv:2507.03724](https://arxiv.org/abs/2507.03724): MemCube
  lifecycle state machine (generate→activate→merge→archive→expire) driven by
  usage stats — Swarm Brain has "replaced" but no non-use-driven cold/archive
  state.
- **MIRIX** [arXiv:2507.07957](https://arxiv.org/abs/2507.07957): six memory
  types each with a manager agent plus a meta-manager that routes writes and
  queries — the generalized form of the missing temporal/entity query router.
- **MAGMA** [arXiv:2601.03236](https://arxiv.org/abs/2601.03236) (ACL 2026):
  one store, four graph views (semantic/temporal/causal/entity), retrieval
  planner composes traversals per query type. Cheap version: per-query-class
  lane-weight profiles.
- **Memory-R1** [arXiv:2508.19828](https://arxiv.org/abs/2508.19828): the
  ADD/UPDATE/DELETE/NOOP gate is trainable with RL from only 152 QA pairs —
  the write gate need not stay hand-prompted once the benchmark harness can
  provide reward. Also "memory distillation": a post-retrieval filter pruning
  the fused set against the query — a stage the pipeline currently lacks.
- **cognee** [github.com/topoteretes/cognee](https://github.com/topoteretes/cognee):
  ontology-driven Extract-Cognify-Load — template for write-time extraction
  from non-conversational inputs (commit diffs, CI logs, tool outputs) into a
  small ontology (Repo, Service, Failure, Fix, Decision).
- **MemoryBank** [arXiv:2305.10250](https://arxiv.org/abs/2305.10250):
  Ebbinghaus decay with retrieval rehearsal — as a *ranking feature* (never
  deletion): last-retrieved-at + retrieval-count feed a recency/rehearsal
  score into fusion; cross-agent reuse count doubles as an implicit trust vote.
  The persistent reuse counters (schema v9) are the substrate.

### LongMemEval leaderboard — who leads and why

Benchmark: [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) (ICLR 2025).
Standings (Mastra-compiled, Feb 2026): Mastra OM gpt-5-mini **94.87%**;
Mastra OM gemini-3-pro 93.27%; Hindsight gemini-3-pro 91.40%; Hindsight
GPT-OSS-120B 89.00%; EmergenceMem 86.00%; Supermemory 85.20%; Mastra OM gpt-4o
84.23% (oracle: 82.4%); plain RAG top-20 80.05%; Zep 71.20%; full context
60.20%.

Attributed drivers, in order: (1) temporal anchoring of stored items
(three dates; valid ranges in context); (2) **write-time distillation quality
over retrieval sophistication** — zero-retrieval OM beats four-lane Hindsight;
(3) structured memory extracts more from better models; (4) multi-session
synthesis is the open frontier (~87% ceiling). Benchmark hygiene: LoCoMo is
now distrusted (judge-prompt swings ~10%, per-vendor judge configs yield
58–92% for the same system); LongMemEval's fixed per-question judges make it
the credible standard.

**LongMemEval-V2** [arXiv:2605.12493](https://arxiv.org/abs/2605.12493)
(May 2026): 451 questions over up to 500 trajectories / 115M tokens of *agent*
history — static state recall, dynamic state tracking, workflow knowledge,
environment gotchas, premise awareness — scored on the returned evidence
bundle ("context gathering"), which matches the task-bootstrap product
exactly. Best method: AgentRunbook-C (trajectories as files + a coding agent
gathering evidence) 72.5% vs 48.5% best RAG, at heavy latency. This is
arguably Swarm Brain's true benchmark, nearly unpublished-on, and the
files-plus-coding-agent winner is the system a CRDB-backed hybrid retriever
should try to beat on the accuracy–latency Pareto frontier.

### Multi-agent / swarm shared memory

- **Governed Shared Memory / MemClaw**
  [arXiv:2606.24535](https://arxiv.org/abs/2606.24535) (Jun 2026) — the
  closest published work to Swarm Brain's whole thesis. Four failure modes:
  unauthorized leakage, stale propagation, contradiction persistence,
  provenance collapse; four primitives: scoped retrieval, temporal
  supersession, provenance tracking, policy-governed propagation. Two
  production findings that translate into direct audits here: (a)
  **GET-by-ID must enforce the same scope/trust filters as search** (their
  sub-tenant scope was bypassed on direct point reads); (b) **a synchronous
  near-duplicate gate must not run before contradiction detection** (their
  dedup rejected contradictory writes before supersession could fire).
  Depth-N provenance-chain reconstruction with writer identity is a
  publishable governance eval.
- **Collaborative Memory**
  [arXiv:2505.18279](https://arxiv.org/abs/2505.18279): asymmetric,
  time-evolving read/write policies over provenance-carrying fragments —
  model for per-agent scoping across heterogeneous workers.
- **Agent Workflow Memory** [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)
  (ICML 2025): agents induce reusable workflows from their own successful
  trajectories; +24.6% relative on Mind2Web, +51.1% on WebArena. Recipe for a
  consolidation job that mines completed task traces into procedural
  memories, with reuse count and downstream success rate as ranking features.
- **Voyager** [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) /
  **ExpeL** [arXiv:2308.10144](https://arxiv.org/abs/2308.10144): the two
  reusable-experience currencies — executable skills (admitted only with
  execution evidence: CI pass, exit 0 — machine-checkable, plugs into the
  existing evidence system) and distilled insights (provenance + review state
  only).
- Surveys: [ACL Findings 2026 memory-evolution survey](https://aclanthology.org/2026.findings-acl.2069.pdf);
  ["Are We Ready For An Agent-Native Memory System?"](https://arxiv.org/html/2606.24775v1)
  (Jun 2026 — current systems are conversation-native, not agent-native);
  Red Hat ([From context to dreams](https://next.redhat.com/2026/06/01/from-context-to-dreams-architecting-memory-for-ai-agents/))
  states no standard cross-agent memory-sharing mechanism exists — Swarm
  Brain's positioning.

### Temporal knowledge and time-aware retrieval

- **OpenAI temporal-agents cookbook**
  ([developers.openai.com](https://developers.openai.com/cookbook/examples/partners/temporal_agents_with_knowledge_graphs/temporal_agents),
  Jul 2025): classify facts **static vs dynamic** at ingest — skip
  invalidation checks for static facts, spend LLM contradiction budget on
  dynamic ones; classify queries current-state / as-of-T / change-over-time
  and route to different retrieval templates. The taxonomy maps one-to-one
  onto SQL templates over the existing bitemporal columns — this is the
  missing temporal routing layer.
- **Temporal Semantic Memory**
  ([ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1496.pdf)):
  offline consolidation episodic → temporal KG → durative facts with duration
  intervals; fits the bitemporal schema with zero schema change.
- Ecosystem consensus (multiple 2026 analyses): flat vector stores have no
  time model; invalidate-but-never-delete with date-range-annotated retrieval
  is the accepted answer. Swarm Brain is architecturally ahead on storage; the
  gap is purely routing/classification and date-annotated context assembly.

## Hybrid fusion, calibration, abstention, rerankers, embeddings

### Why RRF cannot abstain — and the two-channel fix (P1)

- ["The Retrieval Emptiness Problem"](https://tianpan.co/blog/2026-04-16-rag-retrieval-abstention-empty-corpus)
  (Apr 2026) and Google's
  [Sufficient Context](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)
  (ICLR 2025): top-k is a ranking primitive that always returns the k
  least-bad neighbors; with insufficient-but-retrieved context, hallucination
  jumped ~10% → **66%**. Retrieval should be treated as classification with a
  null class. Fixed cosine floors are folk constants — distributions are
  model- and corpus-specific (exactly why the 0.2 floor regressed), and
  per-query-class thresholds capture 2–3× the precision of a global cutoff.
- ["Beyond the Reranker"](https://arxiv.org/html/2606.28367v1) (Jun 2026):
  use a **separate acceptance threshold per score source**, never one
  threshold on the fused output — "gate on lanes, fuse for order."
- **Conformal calibration** — TRAQ
  ([NAACL 2024](https://aclanthology.org/2024.naacl-long.210.pdf)),
  [Conformal Abstention for RAG](https://www.tmls.nyc/research/conformal-abstention-rag)
  (Jul 2026): threshold *any* monotone uncertainty score (max raw cosine,
  reranker logit, lane agreement) on a held-out calibration set for a
  distribution-free, finite-sample error guarantee. Cheapest fix: log
  per-lane raw scores beside the RRF output, calibrate on in-corpus +
  out-of-corpus queries; works today with the hash embedder and transfers by
  re-running calibration after the embedder swap.
- **UAEval4RAG** ([ACL 2025](https://aclanthology.org/2025.acl-long.415.pdf),
  [code](https://github.com/SalesforceAIResearch/Unanswerability_RAGE)):
  synthesizes unanswerable query sets for any corpus across six categories
  including out-of-database — use it to generate the no-answer eval set so
  the abstention gate cannot silently regress.
- **zerank-2** ([calibrated-classifier writeup](https://zeroentropy.dev/articles/smarter-context-compression-for-llm-pipelines-zerank-2-as-a-calibrated-classifier/),
  [model](https://huggingface.co/zeroentropy/zerank-2)): a cross-encoder whose
  score is a calibrated relevance probability (0.8 ≈ 80%), with published
  threshold bands; nothing clears the threshold → abstain. **ZeroEntropy was
  acquired by Notion (Jul 24, 2026) and relicensed its models Apache 2.0**, so
  zerank-2/-small/-nano are now self-hostable. A nano-scale gate over the
  top-10 fused candidates yields the abstention decision RRF cannot.

### Fusion beyond vanilla RRF

- Bruch et al., [ACM TOIS 2023](https://dl.acm.org/doi/10.1145/3596512):
  RRF is sensitive to k and lane count; **TM2C2** (convex combination of
  theoretically-min-max-normalized scores) "significantly outperforms RRF on
  all datasets," and its convex weights are tunable from a handful of labeled
  queries — the benchmark harness already has them.
- [Hybrid-search trade-offs study](https://arxiv.org/abs/2508.01405)
  (Aug 2025, 11 datasets): documents **weakest-link degradation** — one bad
  lane drags fused quality down — motivating per-query lane skipping; the
  hash-dense drag on LongMemEval-S fusion is a textbook instance.
- k=60 is a historical default, not an optimum
  ([overview](https://www.emergentmind.com/topics/reciprocal-rank-fusion-rrf)).
- Learned fusion verdict: the field's energy went to tuned convex fusion +
  rerankers; a tuned convex fusion plus a cross-encoder dominates
  learned-fusion-without-reranker. Don't over-invest in fusion machinery.

### Rerankers (all Apache 2.0 unless noted)

| Model | Size | Key numbers | Note |
|---|---|---|---|
| [zerank-2 / -small / -nano](https://huggingface.co/zeroentropy/zerank-2) | — | 85%+ compression at 90%+ recall in production case study | calibrated probabilities → abstention for free; Apache 2.0 since Jul 2026 |
| [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | 0.6B | MTEB-R 65.80, **MTEB-Code 73.42** (4B: 81.20; bge-v2-m3: 41.38) | instruction-aware, 32K ctx, GGUF/CPU; logit scores need calibration |
| [mxbai-rerank-base-v2](https://www.mixedbread.com/blog/mxbai-rerank-v2) | 0.5B | BEIR nDCG@10 55.57 (large-v2 57.49 vs cohere-3.5 55.39); 0.67s/query vs bge-m3 3.05s | trained on code/SQL/JSON/tool retrieval |
| [jina-reranker-v3](https://arxiv.org/abs/2509.25085) | 0.6B | ~62 BEIR nDCG@10, listwise | likely CC-BY-NC — disqualifying |
| [Cohere Rerank 4](https://cohere.com/blog/rerank-4) | API | — | closed; reference point only |

The deferred-reranker plan stays sound for *ranking*, but the abstention fix
may justify pulling forward a minimal reranker used only as a gate over the
top-10, not as a ranker. CPU budget: 0.5–0.6B over top-10 is feasible
(ONNX/GGUF int8); top-100 is not.

### Embeddings for the 1024-dim slot (P2)

**Recommendation: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)**
([paper](https://arxiv.org/pdf/2506.05176), Jun 2025) — native 1024-dim (no
truncation, no projection), Apache 2.0, 32K context, instruction-aware
(per-lane/per-class instructions), 100+ languages including programming
languages, CPU-serveable (~600MB int8 GGUF; vLLM/TEI/Ollama/llama.cpp), 8B
sibling is MTEB-multilingual #1 (70.58), same-family reranker keeps
query/document understanding consistent across stages. Upgrade path:
Qwen3-Embedding-4B (2560-d, MRL→1024) — but only after benchmarking, because
MRL truncation is measurably lossy even for MRL-trained models
(["Matryoshka Is Dead"](https://zeroentropy.dev/articles/matryoshka-is-dead/),
Apr 2026; Milvus CCKM agrees: <1–2.5% loss but nonzero).

Alternatives: [BGE-M3](https://huggingface.co/BAAI/bge-m3) (MIT, 1024-d dense
+ learned sparse + ColBERT multi-vectors from one pass — attractive if a
learned-sparse lane is wanted; 2024 model, weaker on code);
[snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
(Apache 2.0, 1024-d, throughput-optimized; no code story). Rejected:
jina-embeddings-v4 (3.8B, CC-BY-NC), EmbeddingGemma (max 768-d, 2K ctx),
granite-r2 (768-d).

On swap: re-run both benchmark tracks and **re-calibrate every threshold** —
all thresholds learned under the hash embedder are void.

### Query understanding and routing

- **Adaptive-RAG** [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)
  (NAACL 2024, ~766 citations): a small classifier routing retrieval strategy
  matches multi-step quality at a fraction of the cost; headroom is in
  routing accuracy. Swarm Brain analog is **lane routing** and needs no
  model: identifier-shaped queries (paths, `::`, camelCase, hex, error codes)
  → exact + trigram, downweight dense; conceptual → dense + FTS; temporal →
  valid-time-predicated lanes; multi-hop/bootstrap → enable graph, else skip.
- **HyDE is demoted** (["The Coverage Illusion"](https://arxiv.org/html/2605.27220v1),
  May 2026; [EMNLP 2024 Findings](https://aclanthology.org/2024.findings-emnlp.103.pdf)):
  always-on rewriting costs more than it returns and hurts lexical lanes;
  route first, rewrite only low-confidence conceptual queries — which
  requires the abstention confidence signal anyway. Cheap multi-query
  paraphrase fusion through existing RRF is the better-evidenced option.

## Context engineering, evaluation, DB-native advances

### Context packing for the bootstrap bundle

- [Anthropic effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  (Sep 2025): attention is a finite budget; hybrid strategy = small up-front
  bundle + just-in-time retrieval tools; subagent reports converge on a
  1–2k-token size target — a good target for the handoff/checkpoint lane.
- **Lost-in-the-middle is attenuated, not solved**
  ([Chroma context rot](https://www.trychroma.com/research/context-rot),
  Jul 2025): degradation tracks total context load and distractor density.
  Order the bundle deliberately (handoff + constraints first, playbook last,
  the rest in the middle) and cap its size — a smaller high-precision bundle
  beats a bigger high-recall one.
- **Budgeted submodular packing beats MMR-as-default**
  ([arXiv:2607.00725](https://arxiv.org/abs/2607.00725), Jul 2026): introduces
  **answer-in-context (AiC)** — did the gold span survive into the *packed*
  bundle — which separates exact-match by 4.6× even when retrieval recall is
  equal; greedy facility-location packing beats top-k truncation and
  LLMLingua-2 across readers and budgets. Blueprint for the packing lane
  (same greedy loop as MMR, better objective) and a cheap new harness metric.
- **Handoff schema convergence**
  ([cross-harness survey](https://gist.github.com/badlogic/cd2ef65b0697c4dbe2d13fbecb0a0a5f),
  Dec 2025): Claude Code / Codex / OpenCode / Amp converge on
  accomplished / in-progress / files / next-steps / constraints; Amp's
  **goal-conditioned handoff extraction** (condition what is pulled from the
  prior handoff on the *claimed task's* goal) is the strongest idea for task
  bootstrap; Codex pairs the summary with recent raw spans — argues the
  bundle should carry a few raw evidence spans beside the summary.
- [Cursor semantic search](https://cursor.com/blog/semsearch) (Nov 2025):
  +12.5% avg QA accuracy over grep-only agents; trained on trace-derived
  labels ("what would have helped at step t") — both a justification for the
  dense lane and the strongest argument for building the **trace sink first**
  (it mints training/eval labels for free).

### Iterative retrieval affordances

[SWE-Explore](https://arxiv.org/abs/2606.07297) (Jun 2026): agentic explorers
form "a clear tier above classical retrieval"; file-level localization is
saturated, line-level coverage under budget differentiates. LongMemEval-V2:
agentic evidence-gathering 72.5% vs 48.5% best RAG. Deep-research systems
(Anthropic multi-agent, OpenAI deep research) all converged on iterative
search→read→refine loops. Implication: keep the deterministic bootstrap
bundle for latency, and expose iterative affordances — batched multi-query
recall in one round-trip, a read/expand tool (full memory + evidence spans +
linked memories), and lane provenance in responses so agents can steer their
next query.

### Evidence/source-chunk lane

- [Anthropic contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval):
  prepending a 50–100-token situating header to each chunk before embedding
  and BM25 cut top-20 retrieval failure 49% (67% with reranking). For the
  evidence lane: header = {source, memory title, section path, timestamp} —
  cheap, and fits the exact-span citation model.
- [voyage-context-3](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
  (Jul 2025): contextualized chunk embeddings beat manual contextual
  retrieval by 6.76% and late chunking by 23.66% (NDCG@10, 93 datasets);
  binary 512-d beats float-3072 OpenAI at 0.5% of storage. Ceiling option if
  the embedder is swappable; late chunking loses head-to-head
  ([arXiv:2504.19754](https://arxiv.org/abs/2504.19754)).
- RAPTOR is no longer frontier; the one idea worth keeping is periodic
  synthesized summary memories over clusters (per-component playbook
  regeneration) retrieved in the same lanes as leaves.
- **PropMem/MemEval lessons**
  ([Prosus writeup](https://medium.com/prosus-ai-tech-blog/memeval-benchmarking-memory-for-ai-agents-932d3fd9f3b4),
  Mar 2026): atomic ~25-word propositions with dates resolved to absolute at
  extraction; **entity-scoped retrieval was their single largest accuracy
  win** (a WHERE clause, and in CRDB a vector-index prefix column); soft
  recency — near-duplicate (cosine >0.85) older fact gets a 30% score
  penalty, kept not deleted — a ready-made bitemporal supersession scoring
  policy.

### Evaluation: scaling the gold corpus and judging credibly

- **Trajectory-derived gold** (SWE-Explore + Cursor pattern): for each
  completed swarm task, identify which memories the successful run actually
  consulted/needed → gold labels for "task bootstrap for task T." The planned
  trace sink makes this nearly free; the corpus then scales with usage, not
  authoring.
- **TREC RAG 2026** ([trec-rag.github.io](https://trec-rag.github.io/)):
  UMBRELA LLM-generated qrels + nugget scoring + a human-verified priority
  slice ([RAGDoll](https://github.com/castorini/RAGDoll)) is the
  best-practice template for a scaled, credible harness.
- **LLM-judge reliability** ([2026 systematic eval](https://arxiv.org/html/2606.19544v1);
  [survey](https://www.sciencedirect.com/science/article/pii/S2666675825004564)):
  raw agreement ~80% masks much lower chance-corrected κ; >50% failure on
  bias probes. Use binary rubric-anchored checks ("does the bundle contain a
  span stating X?"), randomized ordering, a ~10% human-audited calibration
  slice, report κ, never compare across judge configs.
- **Quality-per-token as headline metric**
  ([MemEval](https://github.com/ProsusAI/MemEval)): identical systems score
  58–92% under different judge configs and token costs vary 12× — bundle
  tokens per point of Recall@k/nDCG is the credible comparison and favors the
  bounded-budget design.
- **MemoryAgentBench** [arXiv:2507.05257](https://arxiv.org/abs/2507.05257):
  its "conflict resolution" competency maps to trust states + supersession —
  add conflicting-memory questions to the gold corpus.
- Include `_abs`-style unanswerable questions (LongMemEval pattern) and
  UAEval4RAG-generated out-of-database queries for the no-answer metric.

### DB-native advances since mid-2025

- **CockroachDB C-SPANN maturation**
  ([design](https://www.cockroachlabs.com/blog/cspann-real-time-indexing-billions-vectors/);
  [v25.3](https://www.cockroachlabs.com/docs/releases/v25.3),
  [v25.4](https://www.cockroachlabs.com/docs/releases/v25.4)): v25.3 added
  **cosine and inner-product vector indexes** (25.2 was Euclidean-only);
  v25.4 added **online index backfill** on populated tables. RaBitQ 1-bit
  quantization with exact rerank is built in (~200 bytes/vector), making
  per-evidence-chunk vectors cheap. Prefix columns give a separate K-means
  tree per prefix value — entity/scope-prefixed ANN is PropMem's
  entity-scoping win expressed in DDL. Action: verify the deployed version
  ≥25.3 for the cosine dense lane; 25.4 matters for adding the evidence lane
  to populated tables.
- **BM25-in-SQL wave**: three native-BM25 Postgres engines shipped
  ([VectorChord-BM25](https://github.com/tensorchord/VectorChord-bm25),
  [ParadeDB pg_search](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual),
  [Timescale pg_textsearch](https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres),
  Oct 2025). None run on CockroachDB, but BM25 scoring is implementable in
  plain SQL over a term-frequency side table (per-memory term counts +
  corpus document frequencies) — at Swarm Brain corpus sizes correctness
  matters more than WAND-style pruning speed.
- **Hybrid RRF + MMR as pure-SQL patterns** are now textbook
  ([MariaDB shipped RRF docs](https://mariadb.com/docs/server/reference/sql-structure/vectors/optimizing-hybrid-search-query-with-reciprocal-rank-fusion-rrf);
  [RRF/MMR in PL/pgSQL](https://medium.com/open-source-journal/rrf-and-mmr-in-postgres-what-they-mean-and-how-to-implement-them-in-pl-pgsql-63d9bf2dc313)):
  validation that the lane architecture is SOTA-aligned, and the diversity
  pass of the packing lane can run in-database in the same round-trip using a
  stored token_count column.

## Prioritized roadmap mapped to architecture seams

| # | Recommendation | Seam ([architecture](../retrieval-architecture.md)) | Fixes | Cost |
|---|---|---|---|---|
| 1 | Per-lane raw-score evidence channel + conformal-calibrated abstention gate; UAEval4RAG no-answer set | §Fusion, §Abstention | P1 | Low — logging + calibration script |
| 2 | Qwen3-Embedding-0.6B in the 1024-dim slot; re-benchmark, re-calibrate | §Vector projection | P2 | Low-medium — provider integration |
| 3 | Per-query graph gating: complexity-weighted graph RRF weight (EA-GraphRAG recipe, silver labels from the harness), skip-graph fallback on weak seeds, per-hop sufficiency check, supersession-aware edges; same classifier routes all lanes | §Query profiles, graph lane | P3, weakest-link | Low-medium — logistic regression over ~10 features |
| 3a | Deterministic shared-artifact links (same path/symbol/commit) as highest-weight edge types, auto-generated at publish; graph health metrics (avg degree, clustering) | write path, graph lane | graph density/quality | Low — data already in `retrieval_exact_terms` |
| 4 | Referenced-date extraction (third timestamp) + valid-time ranges inline in recalled context | §Read path, write path | temporal reasoning | Low-medium |
| 5 | Submodular token-budgeted packing + answer-in-context harness metric | §Diversity and packing | bootstrap quality | Medium |
| 6 | Write-gate consolidation (ADD/UPDATE-supersede/NOOP against retrieval neighbors; static-vs-dynamic classification; dedup must not preempt contradiction detection) | write path | consolidation debt | Medium |
| 7 | Trace-sink-first, then trajectory-derived gold labels; LongMemEval-V2 adoption; quality-per-token reporting | §Verification | eval debt (P2/P4 phases) | Medium |
| 8 | Iterative affordances: batched multi-query recall, read/expand, lane provenance | §Public API and MCP | agentic depth | Medium |
| 9 | Minimal reranker as abstention gate over top-10 (zerank-2-nano or Qwen3-Reranker-0.6B) | §Reranking | P1 hardening | Medium |
| 10 | Sleep-time consolidation jobs: contradiction sweeps, durative facts, summary memories, as-of-now snapshots; belief memory kind with evidence links | new async projection | inference layer | High |
| 11 | Governance audits + evals from MemClaw: GET-by-ID scope parity, provenance-chain reconstruction | security tests | trust story | Low |
| 12 | Reuse/rehearsal ranking feature from schema-v9 counters (Ebbinghaus decay, cross-agent reuse as trust vote) | §Fusion | swarm-native ranking | Low |

Items 1–3 directly close the three measured problems and are all low-cost;
items 4–7 are the highest-leverage quality investments per the LongMemEval
leaderboard evidence; the rest build the swarm-native moat.
