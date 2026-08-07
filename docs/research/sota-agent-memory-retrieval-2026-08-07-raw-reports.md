# SOTA research — raw agent reports, 2026-08-07

Full, unabridged reports from the four parallel research agents (live
BrightData web research, 2026-08-07) behind the synthesized
[agent memory & retrieval SOTA dump](sota-agent-memory-retrieval-2026-08-07.md).
Kept verbatim so no source, number, or applicability note is lost to
summarization. Report 3's bottom line was folded into the synthesis; the
per-report "applicability" notes here carry additional detail.

Contents:

1. Report 1 — agent memory systems (architectures, consolidation,
   LongMemEval leaderboard, multi-agent shared memory, temporal knowledge).
2. Report 2 — graph-based retrieval / GraphRAG (flagship systems, when graph
   hurts, adaptive routing, PPR vs BFS, temporal edges, write-time
   construction).
3. Report 3 — hybrid fusion, calibration, abstention, rerankers, embeddings.
4. Report 4 — context engineering, agentic retrieval loops, chunk
   enrichment, evaluation, DB-native advances.

---
# Report 1 — State-of-the-Art Agent Memory Systems (2025–2026)

Working context: swarm-brain already has bitemporality, append-by-default + explicit supersession, evidence/trust states, poisoning guards, typed memory links, and hybrid RRF retrieval with bounded graph expansion. Applicability notes below therefore focus on what it lacks: write-time extraction/consolidation policy, temporal query routing, offline consolidation, and cross-agent reuse ranking/governance.

---

## 1. Memory-system architectures: what they do at write time vs read time

### Zep / Graphiti — closest cousin (bitemporal knowledge graph)
- Source: https://arxiv.org/abs/2501.13956 (Jan 2025); https://github.com/getzep/graphiti; Neo4j writeup https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/
- Mechanism (from full paper text): Three-layer graph — (1) **episode subgraph** storing raw messages/text/JSON non-lossily; (2) **semantic entity subgraph** of extracted entities and fact edges; (3) **community subgraph** of entity clusters with map-reduce-style summaries, maintained incrementally via **label propagation** (chosen over Leiden specifically because a single recursive propagation step lets new nodes join communities dynamically, deferring full refreshes). Write path: entity extraction over current message + last n=4 messages, a reflexion-style second pass to catch missed entities, entity embedding (1024-d) + fulltext candidate search + LLM entity-resolution prompt for dedup; fact (edge) extraction with hyper-edge support (same fact replicated across entity pairs); edge dedup constrained to edges *between the same entity pair* (both a correctness and complexity win). **Bitemporal model**: timeline T (event chronology) and T′ (ingestion/transaction), four timestamps per edge — t′_created, t′_expired (transactional) and t_valid, t_invalid (validity). On ingest, an LLM compares each new edge against semantically related existing edges; on temporally overlapping contradiction it sets the old edge's t_invalid = new edge's t_valid (new information always wins, ordered by T′). Read path: three lanes — cosine similarity, BM25, and **breadth-first search seeded by recent episodes** (so recently mentioned entities pull in their neighborhoods) — then reranking (RRF, MMR, an **episode-mentions reranker** boosting frequently referenced facts, a **node-distance reranker** relative to a centroid node, and cross-encoder as the expensive option). The context constructor emits facts *with their valid/invalid date ranges* in the prompt.
- Results: DMR 94.8% (gpt-4-turbo) vs MemGPT 93.4%; LongMemEval-S up to +18.5% accuracy and ~90% latency reduction vs full-context baseline. Note: independent runs place Zep at 71.2% on LongMemEval with gpt-4o (Mastra leaderboard, below), well behind 2026 systems.
- Applicability: swarm-brain already matches the bitemporal edge model. What it can steal: (a) the **entity-pair-constrained dedup** trick to bound conflict-detection cost at write time; (b) **episode-mentions and node-distance rerankers** as cheap additional RRF lanes (frequency-of-reference is a natural trust/salience prior in a swarm); (c) emitting **valid-time ranges inline in retrieved context** — Zep's context template shows the LLM the date range of every fact, which is most of what makes temporal questions answerable; (d) reflexion-style two-pass extraction, which swarm-brain currently has no equivalent of (it stores what agents write rather than extracting).

### Mem0
- Source: https://arxiv.org/abs/2504.19413 (Apr 2025); https://mem0.ai
- Mechanism: Two-phase write pipeline: an **extraction phase** (LLM reads new message pairs plus a rolling conversation summary and recent messages, emits candidate salient memories) and an **update phase** where, for each candidate, the top-similar existing memories are retrieved and an LLM tool-call chooses **ADD / UPDATE / DELETE / NOOP** — i.e., consolidation and conflict resolution are done eagerly at write time against a compact fact store. Mem0-g variant stores memories as a knowledge graph (entity nodes, relation edges) to capture relational structure. Read time is plain top-k vector retrieval over the consolidated facts.
- Results (LOCOMO, per abstract): consistently above six baseline categories across single-hop, temporal, multi-hop, open-domain; **26% relative improvement (LLM-as-judge) over OpenAI's memory**; graph variant ~2% higher than base; **91% lower p95 latency and >90% token savings vs full-context**. Caveat: Zep publicly disputed Mem0's comparative LoCoMo claims ("Lies, Damn Lies, & Statistics," https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/), and LoCoMo itself is now widely considered unreliable (see §3).
- Applicability: Mem0's ADD/UPDATE/DELETE/NOOP gate is the canonical **write-time consolidation policy** swarm-brain lacks. For an append-by-default store, the mapping is: ADD→new observation, UPDATE→new observation + supersession link, DELETE→supersession-with-tombstone, NOOP→drop as duplicate. The key design point is that the decision is made *against retrieved near-neighbors at write time*, which is cheap in swarm-brain because the hybrid retrieval pipeline already exists — reuse it as the candidate generator for the write gate.

### Letta (MemGPT lineage)
- Sources: https://www.letta.com/blog/sleep-time-compute/ (Apr 2025); https://arxiv.org/abs/2504.13171; vendor comparisons https://vectorize.io/articles/mem0-vs-letta (2026)
- Mechanism: MemGPT's OS metaphor — self-editing memory via tools: **core memory blocks** (always-in-context, agent-editable via memory_replace/memory_insert), **recall memory** (conversation search), **archival memory** (vector store), with the agent itself deciding when to page information in/out. 2025-26 evolution: Letta introduced **sleep-time agents** — a second agent that shares memory blocks with the primary agent and reorganizes/rewrites them offline (see §2), and per 2026 surveys (https://entropi.ai/blog/context-engineering-ai-memory-landscape-2026) ships 128K-default contexts with overhauled compaction.
- Results: MemGPT DMR 93.4% (superseded by Zep and by full-context baselines — DMR is saturated).
- Applicability: Letta's contribution to swarm-brain is less the paging model (swarm workers each have their own harness) and more the **shared-memory-block-with-background-editor** pattern: a consolidation daemon that holds write access to the same store the workers read, structurally identical to a swarm-brain "librarian" agent doing supersession/merge passes off the hot path.

### A-Mem (NeurIPS 2025)
- Source: https://arxiv.org/abs/2502.12110; https://github.com/WujiangXu/A-mem-sys; https://openreview.net/forum?id=FiM0M8gcct
- Mechanism: Zettelkasten-style **agentic note construction**: each new memory becomes a structured note (contextual description, keywords, tags) generated by an LLM; the system then retrieves related historical notes and an LLM decides which **links** to create; crucially, integration of a new note can trigger **memory evolution** — rewriting the contextual descriptions/attributes of *existing* notes so the network's understanding refines over time. No fixed schema or operation set; organization is agent-driven.
- Results: improvements over SOTA baselines on six foundation models (LoCoMo-based eval); ~899 citations, so it is the reference point for "agentic" (LLM-decides) memory organization.
- Applicability: A-Mem's **link generation step is an automated version of swarm-brain's typed memory links** — today swarm-brain links are presumably written explicitly by agents; an A-Mem-style pass that proposes links between a new observation and its retrieval neighbors would densify the graph that the 1-2 hop expansion walks. Its "memory evolution" (retroactively rewriting old notes) is the anti-pattern for swarm-brain's audit model — but the bitemporal answer is clean: evolution events become new versions with supersession, preserving recorded-time history. That combination (A-Mem behavior, bitemporal substrate) is genuinely novel territory.

### MemoryBank (AAAI 2024, lineage)
- Source: https://arxiv.org/abs/2305.10250
- Mechanism: the earliest widely-cited forgetting policy — memory strength decays per the **Ebbinghaus forgetting curve**, with retrieval "rehearsal" resetting/boosting strength; plus daily-event summarization and evolving user-personality summaries.
- Applicability: the Ebbinghaus decay-with-rehearsal signal is a cheap, principled **ranking feature** (not a deletion policy) for swarm-brain: last-retrieved-at and retrieval-count per memory feed a recency/rehearsal score into the RRF fusion. In a swarm, "how often any agent has re-used this memory" doubles as an implicit trust vote.

### cognee
- Sources: https://github.com/topoteretes/cognee; https://www.cognee.ai/blog/guides/open-source-memory-frameworks-llm-agents (2026)
- Mechanism: ECL pipeline — **Extract, Cognify, Load**: documents/conversations are chunked, "cognified" into a knowledge graph (entities, relations, summaries) plus embeddings, loaded into pluggable graph+vector backends; retrieval combines graph traversal and vector search. Positioned as data-pipeline-first (developer defines ontologies and tasks) rather than conversation-first.
- Results: In the June 2026 evaluation "Are We Ready For An Agent-Native Memory System?" (https://arxiv.org/html/2606.24775v1), **cognee attains the highest Answer F1 on LoCoMo** while MemOS attains the highest Exact Match (8.9); the paper groups Cognee, MemOS, MemoryOS as "closest to agent-native."
- Applicability: cognee's ontology-driven cognify step is the template for swarm-brain's missing **write-time extraction for non-conversational inputs** (commit diffs, CI logs, tool outputs) — define a small ontology (Repo, Service, Failure, Fix, Decision) and cognify worker transcripts into typed nodes rather than free-text observations.

### MemOS
- Sources: https://arxiv.org/abs/2507.03724 (Jul 2025, "MemOS: A Memory OS for AI System"); precursor https://arxiv.org/abs/2505.22101; https://github.com/MemTensor/MemOS
- Mechanism: treats memory as a schedulable system resource. The **MemCube** is the unit: payload + metadata (IDs, provenance, access control, lifecycle state), unifying three memory forms — **plaintext** (retrievable text/graph), **activation** (KV-cache), and **parametric** (adapters/weights) — with promotion/demotion paths between them (hot plaintext can be compiled into KV or parameters; stale parametric knowledge can be demoted back to plaintext). Provides scheduling, lifecycle (generate→activate→merge→archive→expire), versioning, and audit across MemCubes; 2026 releases emphasize millisecond-level async memory add.
- Results: strongest LoCoMo EM (8.9) in the 2606.24775 evaluation; the project claims large temporal-reasoning gains over OpenAI's memory on LoCoMo (vendor-reported).
- Applicability: mostly a vocabulary/architecture validation — swarm-brain's observation rows with trust/review/lifecycle states are MemCubes-for-plaintext. The transferable idea is the explicit **lifecycle state machine with scheduled transitions** (active→merged→archived→expired) driven by usage statistics, which swarm-brain's supersession covers only partially (it has "replaced" but not "cold/archive" driven by non-use).

### MIRIX
- Source: https://arxiv.org/abs/2507.07957 (2025)
- Mechanism: a **multi-agent memory system**: six specialized memory types (core, episodic, semantic, procedural, resource, knowledge-vault) each managed by a dedicated memory-manager agent, coordinated by a **meta-memory-manager** that routes incoming information to the right store and routes queries to the right manager(s). Evaluated on ScreenshotVQA (multimodal) and LoCoMo (strong scores, ~85% range vendor-reported).
- Applicability: the router pattern — **classify-then-route at both write and read time** — is exactly the "temporal query routing" swarm-brain lacks generalized to all memory types: a cheap classifier deciding "is this procedural (how-to), semantic (fact), episodic (what happened), or temporal (as-of)" and dispatching to different retrieval configurations (e.g., temporal queries get valid-time-filtered lanes, procedural queries get skill-library ranking).

### Hindsight
- Source: https://arxiv.org/abs/2512.12818 (Dec 2025)
- Mechanism: memory as a structured substrate organized into **four logical networks: world facts, agent experiences, synthesized entity summaries, and evolving beliefs** — an explicit separation of **evidence from inference**. Three operations: **retain** (temporal, entity-aware incremental conversion of streams into a queryable bank), **recall** (four parallel retrieval strategies with neural reranking, per Mastra's description), **reflect** (a reasoning layer that answers and *updates beliefs traceably* over the bank).
- Results: with an open 20B backbone, lifts LongMemEval accuracy 39%→83.6% vs same-backbone full-context and beats full-context GPT-4o; scaled backbone reaches **91.4% LongMemEval** (gemini-3-pro) and **89.61% LoCoMo** (vs 75.78% prior best open system).
- Applicability: the fact/experience/belief split maps directly onto swarm-brain's evidence/trust model and extends it: swarm-brain has evidence-backed memories but (apparently) no first-class **derived-belief type whose provenance links point at supporting observations** and which reflection passes are allowed to revise. Adding a `belief` memory kind with mandatory evidence links, revised only by consolidation jobs, would replicate Hindsight's traceable-inference property on top of existing supersession.

### Mastra Observational Memory (OM) — current LongMemEval SOTA
- Source: https://mastra.ai/research/observational-memory (Feb 9, 2026); code: https://github.com/mastra-ai/mastra/tree/main/packages/memory/src/processors/observational-memory
- Mechanism: **no retrieval at all**. Two background agents watch the conversation: an **Observer** converts raw message history into dense, dated, priority-tagged observations once unobserved history passes a token threshold (3–6× compression on text; anecdotally 5–40× on tool-call-heavy traces), and a **Reflector** restructures/condenses the observation log when it passes a second threshold (merging related items, dropping superseded ones). The context window is stable and append-only (memory prefix + live tail), hence fully prompt-cacheable; average context across the whole benchmark run was ~30k tokens. **Temporal anchoring**: every observation carries up to three dates — observation date, referenced date ("flight is Jan 31"), and computed relative date ("2 days from today"). Format is deliberately plain text: two-level bullets, emoji priorities (signal from Observer to Reflector), dated titled sections. Triggers are token-count-based, not time-based.
- Results (longmemeval_s, gemini-2.5-flash as Observer/Reflector): **gpt-4o 84.23%** (beats the oracle's 82.4%, i.e., its compressed observations outperform raw ground-truth sessions), gemini-3-flash 89.20%, gemini-3-pro 93.27%, **gpt-5-mini 94.87% — highest recorded**. Per-category with gpt-5-mini: knowledge-update 96.2%, temporal-reasoning 95.5%, multi-session 87.2% (ties Hindsight; apparent cross-system ceiling), single-session-preference 100%.
- Applicability: three concrete imports for swarm-brain: (1) the **three-date model** — swarm-brain has valid-time and recorded-time but apparently not "referenced date extracted from content" as a separate indexed field; OM's results suggest this is the single highest-leverage feature for temporal-reasoning; (2) **token-threshold-triggered Observer/Reflector** as the write-time distillation policy for worker transcripts before anything hits the shared store (workers currently write raw-ish observations; an Observer pass standardizes density and dating); (3) the finding that **compressed observations beat oracle raw context** justifies aggressive consolidation — it is not merely a cost play.

### MAGMA
- Source: https://arxiv.org/abs/2601.03236 (Jan 2026, ACL 2026: https://aclanthology.org/2026.acl-long.1709/); code https://github.com/FredJiang0324/MAGMA
- Mechanism: represents each memory item simultaneously across **four orthogonal graph views — semantic, temporal, causal, and entity** — within one store, and **decouples memory representation from retrieval logic**: queries are answered by composing traversals over whichever views the query type needs, giving transparent reasoning paths and fine-grained retrieval control. Targets long-term conversational memory and multi-hop reasoning.
- Applicability: validation plus extension of swarm-brain's typed-edge design: swarm-brain's typed links already approximate multiple views in one graph; MAGMA's lesson is to make the **retrieval planner view-aware** — route temporal questions along temporal edges, "why" questions along causal edges — rather than fusing everything through one weighted RRF. A per-query-type lane-weight profile is the cheap version.

### Memory-R1
- Source: https://arxiv.org/abs/2508.19828 (Aug 2025, v5 2026); https://github.com/yansikuan/memory-r1/
- Mechanism: replaces prompt-engineered write policy with **RL**: a Memory Manager agent is trained (PPO/GRPO, outcome-reward) to choose ADD/UPDATE/DELETE/NOOP, and an Answer Agent is trained to **distill** noisy retrieved memories (memory distillation: filter retrieved set before answering). Trained with only **152 QA pairs**, generalizes across question types and three benchmarks including LoCoMo.
- Applicability: evidence that the write-gate policy (§Mem0) need not be hand-prompted — with swarm-brain's benchmark harness already in place, the ADD/UPDATE(supersede)/NOOP gate is trainable from end-task reward. Also "memory distillation" = a post-retrieval filter stage swarm-brain's pipeline lacks (it fuses and expands, but nothing prunes the fused set against the query before prompting).

### Field surveys worth citing
- "From Storage to Experience: A Survey on the Evolution of LLM Agent Memory" — ACL Findings 2026, https://aclanthology.org/2026.findings-acl.2069.pdf — frames the field's arc from passive stores toward experience-centric memory.
- "Are We Ready For An Agent-Native Memory System?" — https://arxiv.org/html/2606.24775v1 (Jun 2026) — cross-system LoCoMo comparison (MemOS best EM, cognee best F1) and an argument that current systems are conversation-native, not agent-native (tool traces, environment state).
- TsinghuaC3I curated list: https://github.com/TsinghuaC3I/Awesome-Memory-for-Agents (tracks self-evolving-agent and cross-task-experience-sharing papers).

---

## 2. Consolidation, summarization, forgetting, sleep-time compute

### Sleep-time Compute (Letta + UC Berkeley)
- Sources: https://arxiv.org/abs/2504.13171 (Apr 2025); https://www.letta.com/blog/sleep-time-compute/
- Mechanism: models "think" offline about existing context *before* queries arrive — anticipating likely questions, pre-computing inferences, and rewriting the memory/context representation into a more useful form; at query time the agent runs against the pre-digested state. In Letta's product this is a **sleep-time agent sharing memory blocks** with the primary agent and reorganizing them between interactions.
- Results: reduces test-time compute needed for equal accuracy by **~5×** on Stateful GSM-Symbolic (confirmed via paper/HN discussion https://news.ycombinator.com/item?id=48281226); scaling sleep-time compute further raises accuracy double-digit percent, and cost amortizes when multiple queries hit the same context.
- Applicability: swarm-brain's idle time between hackathon-style swarm bursts is free compute. A CRDB-native version: scheduled jobs that (a) pre-compute entity/topic summaries over recent observations, (b) run contradiction sweeps (pairwise LLM checks within entity-pair partitions, per Zep), (c) pre-materialize "current state as of now" snapshot tables so as-of-now queries don't pay bitemporal resolution cost. The bitemporal ledger makes all of this safely re-derivable.

### Observational Memory's Reflector (see §1)
- The current best-performing consolidation policy on record is embarrassingly simple: token-budget-triggered rewrite of an observation log — merge related, drop superseded, keep dates. No graph, no embeddings. Strong evidence that swarm-brain's consolidation MVP should be a Reflector pass emitting superseding summary-observations, not a complex clustering pipeline.

### Temporal Semantic Memory for Personalized LLM Agents (Su et al.)
- Source: https://aclanthology.org/2026.findings-acl.1496.pdf (ACL Findings 2026)
- Mechanism: two-stage consolidation — construct a **temporal knowledge graph from episodic memory**, then consolidate it into **time-aware "durative" memory** (facts with duration intervals), so retrieval can respect when facts held. This is the academic mirror of Zep's valid/invalid ranges, arrived at via offline consolidation rather than write-time invalidation.
- Applicability: pattern for swarm-brain's offline path: episodic observations → derived durative facts with valid-time intervals, stored as first-class derived memories with evidence links. Fits the existing bitemporal schema with zero schema change.

### LayerMem (under review, 2026)
- Source: https://openreview.net/pdf/8307c681e8e6c67511f2fd2c94c7e2499e5aede2.pdf — "Query-Adaptive Hierarchical Memory via Sleep-time computation": builds layered memory hierarchies offline and adapts which layer answers which query. Early-stage but confirms the trend: **hierarchy built offline, selected per-query online**.

### Consolidation-lineage context
- Red Hat "From context to dreams" (Jun 2026, https://next.redhat.com/2026/06/01/from-context-to-dreams-architecting-memory-for-ai-agents/) and https://ogham-mcp.dev/blog/memory-consolidation-lineage/ document the 2025-26 industry shift to sleep-stage consolidation; Red Hat explicitly notes there is still **no standard mechanism for cross-agent memory sharing** — the gap swarm-brain targets.

---

## 3. LongMemEval leaderboard: who leads and why

Benchmark: LongMemEval, ICLR 2025, https://arxiv.org/abs/2410.10813 — 500 questions, ~50 sessions each (~115k tokens avg for -S), six categories (single-session-user/assistant/preference, knowledge-update, temporal-reasoning, multi-session).

Current standings (Mastra-compiled leaderboard, Feb 2026, cross-checked against vendor pages):

| System | Model | Overall | Attribution |
|---|---|---|---|
| Mastra OM | gpt-5-mini | **94.87%** | observer/reflector distillation, three-date temporal anchoring, stable cacheable context, zero retrieval |
| Mastra OM | gemini-3-pro | 93.27% | same |
| Hindsight | gemini-3-pro | 91.40% | four-network fact/experience/belief separation, 4 parallel retrieval strategies + neural reranking |
| Hindsight | GPT-OSS 120B | 89.00% | same, open backbone |
| EmergenceMem "Internal" | gpt-4o | 86.00% | closed config, multi-step reranking (https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) |
| Supermemory | gemini-3-pro | 85.20% | (https://supermemory.ai/research) |
| Mastra OM | gpt-4o | 84.23% | beats oracle (82.4%) |
| EmergenceMem Simple | gpt-4o | 82.40% | RAG with session decomposition; their Jun 2025 post argued "RAG-like methods have largely solved" LME |
| Mastra RAG topK-20 | gpt-4o | 80.05% | plain RAG baseline |
| Zep | gpt-4o | 71.20% | temporal KG |
| Full context | gpt-4o | 60.20% | — |

Other claimed results (vendor/community, not independently verified): OMEGA 95.4% (466/500) at 50ms retrieval (https://omegamax.co/benchmarks); Sibyl 95.6% (https://x.com/sibylcap/status/2044967335145197912); a Reddit-reported 96.4% top-50 with Gemini 3 Flash (https://www.reddit.com/r/AI_Agents/comments/1tgm63n/); RetainDB 79% overall (https://www.retaindb.com/benchmark); Mem0's own 2026 benchmark roundup cites 94.4% (https://mem0.ai/blog/ai-memory-benchmarks-in-2026).

What top systems attribute gains to:
1. **Temporal anchoring of stored items** (OM's three dates; Zep's valid ranges in context) — biggest driver on temporal-reasoning and knowledge-update categories.
2. **Write-time distillation quality over retrieval sophistication** — OM beats four-lane-retrieval-plus-reranker Hindsight with zero retrieval; OM beats the oracle, meaning distilled observations > raw correct sessions.
3. **Model scaling asymmetry**: structured observation logs extract more from better models (OM +9pts gpt-4o→gemini-3-pro vs Supermemory +3.6) — dense structured memory is increasingly favored as actor models improve.
4. **Multi-session synthesis is the open frontier**: 87.2% ceiling shared by OM and Hindsight.
5. Benchmark hygiene: LoCoMo is now distrusted (no standard judge, ~10% score swings by judge prompt, F1 penalizing correct verbose answers — Mastra; Zep's critique post); LongMemEval's per-question judge prompts make it the credible standard. **LongMemEval-V2** (https://arxiv.org/abs/2605.12493, May 2026, Wu et al./UCLA) redefines the target for *agent* (not chat) memory: 451 questions over up to 500 trajectories / 115M tokens of web-agent history, testing static state recall, dynamic state tracking, workflow knowledge, environment gotchas, premise awareness, under a "context gathering" formulation (memory system returns compact evidence). Best method: **AgentRunbook-C** — trajectories stored as files + a coding agent gathering evidence in a sandbox — 72.5% vs 48.5% for best RAG (AgentRunbook-R: knowledge pools for raw states, events, strategy notes) and 69.3% for an off-the-shelf coding agent; coding-agent methods pay heavy latency.
- Applicability: swarm-brain benchmarks LongMemEval-S retrieval-only — the leaderboard says its next points come from (a) referenced-date extraction + valid-range-annotated context assembly, (b) a distillation pass before storage, not more retrieval lanes. And **LME-V2 is arguably swarm-brain's true benchmark** (coding-agent environment experience, gotchas, workflows); worth adopting early since almost nobody has published on it yet — and notably its best system (files + coding agent) is a challenge to the database approach that a CRDB-backed hybrid retriever should try to beat on the accuracy-latency Pareto frontier.

---

## 4. Multi-agent / swarm shared memory, cross-agent reuse, trust and provenance

### Governed Shared Memory / MemClaw (closest published work to swarm-brain's whole thesis)
- Source: https://arxiv.org/abs/2606.24535 (Jun 2026)
- Mechanism: formalizes the **fleet-memory problem** with four failure modes — **unauthorized leakage, stale propagation, contradiction persistence, provenance collapse** — and four primitives: **scoped retrieval, temporal supersession, provenance tracking, policy-governed propagation**, implemented in MemClaw (production multi-tenant memory service) and evaluated live via the ArgusFleet harness.
- Results: 100% reconstruction of depth-4 derivation chains with correct writer identity at sub-second per-hop; high intra-fleet visibility with zero cross-fleet leakage; write-to-visible latency one search round-trip in strong-write mode. Two production negatives: (1) **asymmetric scope enforcement** — tenant isolation held but sub-tenant scope was bypassed on direct GET-by-id of agent-scoped credentials (found and fixed during the study); (2) **pipeline ordering conflict** — a synchronous near-duplicate gate **rejected contradictory writes before the async contradiction detector could see them**, so supersession never fired for those items.
- Applicability: highest-value paper for swarm-brain. Its vocabulary matches swarm-brain feature-for-feature (temporal supersession, provenance, poisoning-adjacent governance), so it is both citation-cover and a checklist. Two direct action items: audit that (a) **point-reads by ID enforce the same scope/trust filters as search paths**, and (b) swarm-brain's **dedup guard runs after (or jointly with) contradiction detection** — if a near-duplicate gate sits in front of the supersession logic, contradicting updates can be silently swallowed. Also: depth-N derivation-chain reconstruction with writer identity is a measurable eval swarm-brain could publish.

### Collaborative Memory
- Source: https://arxiv.org/abs/2505.18279 (May 2025, Rezazadeh et al.)
- Mechanism: multi-user, multi-agent memory with **asymmetric, time-evolving access controls**: two-tier private/shared memory, memory fragments carrying provenance (which user/agent/resource produced them), with read/write policies as dynamic bipartite graphs so permissions can change over time and past writes remain governed.
- Applicability: the model for swarm-brain's per-agent scoping when heterogeneous workers (Claude Code vs Codex vs Gemini) shouldn't all see everything (e.g., repo-scoped or customer-scoped fleets): attach read/write policy predicates to memory rows and evaluate at retrieval; time-evolving policies fit naturally over recorded-time.

### Agent Workflow Memory (AWM)
- Source: https://arxiv.org/abs/2409.07429 (ICML 2025; https://github.com/zorazrw/agent-workflow-memory)
- Mechanism: agents **induce reusable workflows** (common sub-routine action sequences) from their own successful trajectories, store them in memory, and condition future task solving on them; works offline (from training examples) or online (self-induced from test-stream successes, no labels).
- Results: +24.6% relative step-wise success on Mind2Web (cross-task) and +51.1% relative on WebArena (35.6% overall SOTA at the time), while reducing steps; generalizes across tasks/sites/domains.
- Applicability: the concrete recipe for **agent-generated knowledge reuse** in swarm-brain: a consolidation job that mines completed swarm task traces for recurring successful action sequences ("how we fix flaky CRDB test X", "how to run the LongMemEval bench") and writes them as procedural memories. Reuse count and downstream success rate become ranking features — this is the "cross-agent reuse ranking" swarm-brain lacks.

### Voyager (lineage) and ExpeL
- Voyager: https://arxiv.org/abs/2305.16291 (2023) — ever-growing **skill library** of verified, executable code skills, indexed by embedding of the skill's docstring; skills are only added after environment-verified success, and compose into more complex skills. The verification-before-admission rule is the ancestor of swarm-brain's evidence-gated trust states applied to procedural memory.
- ExpeL: https://arxiv.org/abs/2308.10144 (2023/AAAI 2024) — extracts natural-language **insights and rules from pooled cross-task experiences** (successes and failures) without weight updates. Together these define the two reusable-experience currencies: executable skills and distilled insights.
- Applicability: swarm-brain should treat these as two distinct memory kinds with different admission policies: skills require execution evidence (CI pass, command exit 0 — machine-checkable, plugs into the existing evidence system); insights require only provenance plus review state.

### Surveys
- "Memory in LLM-based Multi-agent Systems: Mechanisms, Challenges, and Collective Intelligence" — Wu & Shu, Emory, Dec 2025, https://www.techrxiv.org/doi/10.36227/techrxiv.176539617.79044553 — first comprehensive LLM-MAS memory survey; taxonomizes local vs shared memory, and flags trust/consistency in shared stores as open.
- "Multi-Agent LLM Systems: From Emergent Collaboration to Learned Coordination" — https://www.preprints.org/manuscript/202511.1370 (2025) — covers shared-memory motifs and communication topologies.
- Red Hat (Jun 2026, URL above): industry statement that no standard cross-agent memory-sharing mechanism exists — supporting swarm-brain's positioning.

---

## 5. Temporal knowledge and time-aware retrieval ("what was true at T", "what changed")

### Zep/Graphiti — see §1. The reference implementation: valid/invalid ranges set by LLM contradiction detection at write time; point-in-time answering = filter edges where t_valid ≤ T < t_invalid; "what changed" = edges whose t_valid or t_invalid falls in a window. Retrieved context includes date ranges so the actor LLM does the final temporal reasoning.

### OpenAI Cookbook: Temporal Agents with Knowledge Graphs
- Source: https://developers.openai.com/cookbook/examples/partners/temporal_agents_with_knowledge_graphs/temporal_agents (Jul 2025); practitioner walk-through https://medium.com/@aiwithakashgoyal/temporal-agents-in-graphos-building-time-aware-knowledge-graphs-with-multi-level-ingestion-ee448441929c
- Mechanism: triplet statements classified at ingestion as **static vs dynamic (and atemporal)**; dynamic facts get valid_at/expired_at; an **invalidation agent** checks new statements against temporally overlapping existing ones; point-in-time queries filter on valid_at/expired_at. Notably includes a **temporal-query classification/routing step**: questions are classified (current-state vs point-in-time vs change-over-time) and routed to different retrieval templates.
- Applicability: the static/dynamic fact classification at write time is cheap and directly usable — swarm-brain can skip invalidation checks for facts classified static ("service X is written in Go") and focus LLM contradiction budget on dynamic ones ("deploy target is Y"). The query-routing taxonomy (current / as-of-T / diff-over-window) maps one-to-one onto SQL templates over swarm-brain's bitemporal columns — this is precisely the missing "temporal query routing" layer: a classifier in front of the retrieval pipeline selecting valid-time predicates and lane weights.

### Temporal Semantic Memory (ACL Findings 2026) — see §2: episodic→temporal-KG→durative memory consolidation; supports duration-aware retrieval.

### MAGMA's temporal graph view (§1) — temporal relations as a dedicated traversal dimension, selected per query type.

### Bitemporal RDF implementations
- Source: https://www.mdpi.com/2227-9709/13/4/61 (2026) — database-theory treatment of bitemporal RDF with time-slice ("what was true at t") and rollback ("what did we believe at t") query patterns and benchmark datasets. Useful citation to formally name swarm-brain's two query classes (valid-time slice vs transaction-time rollback).

### Ecosystem consensus
Multiple 2026 practitioner analyses (https://atlan.com/know/vector-database-vs-knowledge-graph-agent-memory/, https://thedatapraxis.com/blog/knowledge-graphs-for-ai-agents/, https://cognitivx.io/blog/mem0-vs-zep-vs-letta-vs-cognee) converge on: flat vector stores have no time model (stale and fresh vectors coexist silently, scoring ~0.58-0.67 on temporal-recall probes per https://github.com/Keyan-sm/temporal-recall), and invalidate-but-never-delete with date-range-annotated retrieval is the accepted answer. swarm-brain is architecturally ahead here; the gap is purely the routing/classification layer and date-annotated context assembly, not the storage model.

---

## Synthesis: ranked gap list for swarm-brain

1. **Write-time distillation (Observer pass)** — the single biggest lever per the LongMemEval leaderboard: dated, prioritized, compressed observations before storage; OM proves distilled beats raw-oracle. (Mastra OM, Mem0 extraction phase.)
2. **Referenced-date extraction as a third indexed timestamp** (beyond valid/recorded): observation-date + content-referenced-date + relative-date resolution. (OM three-date model; Zep t_ref.)
3. **Temporal query routing**: classify queries current-state / as-of-T / what-changed / procedural / causal, and select valid-time predicates + RRF lane weights per class. (OpenAI temporal agents cookbook, MIRIX meta-manager, MAGMA view-aware retrieval.)
4. **Write-gate consolidation policy**: retrieve-neighbors → ADD/UPDATE(supersede)/NOOP decision, with static-vs-dynamic fact classification to budget LLM contradiction checks; entity-pair-scoped dedup. Beware the MemClaw ordering bug: dedup must not preempt contradiction detection. (Mem0, Zep, Memory-R1, MemClaw.)
5. **Cross-agent reuse ranking**: retrieval-count/rehearsal decay (MemoryBank), downstream-success-weighted procedural memories (AWM), execution-evidence-gated skill admission (Voyager) — reuse-by-other-agents as a trust signal is unpublished territory swarm-brain could own.
6. **Offline consolidation jobs** (sleep-time): scheduled contradiction sweeps, durative-fact derivation, community/topic summaries, pre-materialized as-of-now snapshots. (Letta sleep-time compute ~5× test-time savings; Temporal Semantic Memory.)
7. **Belief layer**: derived beliefs with mandatory evidence links, revisable only by reflection jobs — extends the existing evidence/trust model to inferences. (Hindsight's four networks.)
8. **Governance evals to publish**: depth-N provenance-chain reconstruction with writer identity, scope-enforcement symmetry between search and GET-by-id, zero cross-fleet leakage. (MemClaw/ArgusFleet.)
9. **Benchmark positioning**: keep LongMemEval-S, avoid LoCoMo (judge-prompt irreproducibility), and adopt **LongMemEval-V2** (agent environment-experience, 115M-token histories) early — it matches swarm-brain's coding-swarm domain and has almost no published entrants; the target to beat is AgentRunbook-C's 72.5% at lower latency.

---

# Report 2 — State-of-the-Art Graph-Based Retrieval / GraphRAG (2025–2026)

Scope note: all findings below were verified against live sources on 2026-08-07. Applicability notes reference the swarm-brain design: 4-lane RRF (k=60) → bounded memory-link graph expansion (≤16 seeds, 1–2 hops, 8 neighbors/node, typed-relation weights, 0.85/hop decay, query-gate floor 0.60, application-side fixed-depth traversal) → final weighted RRF, all inside CockroachDB + app code.

---

## 1. Flagship systems: mechanisms, costs, measured gains, failure modes

### HippoRAG 2 (OSU, ICML 2025) — the single most relevant system
- **Source**: https://arxiv.org/abs/2502.14802 ("From RAG to Memory", Gutiérrez et al., v2 June 2025); code https://github.com/OSU-NLP-Group/HippoRAG. Predecessor HippoRAG 1: NeurIPS 2024, https://arxiv.org/abs/2405.14831.
- **Mechanism (verified from full paper)**: Offline: LLM OpenIE extracts schema-free triples → phrase nodes + relation edges; synonym edges added between phrase nodes with embedding similarity > 0.8; **each passage becomes a passage node** connected to its phrases via "contains" context edges ("dense–sparse integration"). Online: query embedding matched **directly against triple embeddings** (not entities — "query-to-triple linking"), top-5 triples; an LLM **"recognition memory" filter** prunes irrelevant triples; PPR (damping 0.5, python-igraph) seeded with ≤5 phrase nodes (score = avg rank score of filtered triples containing them) **plus ALL passage nodes** with reset probability ∝ embedding similarity × **weight factor 0.05**; passages ranked by final PageRank mass. If the filter empties the triple set, it **falls back to pure dense retrieval — skipping graph search entirely**.
- **Why it exists — the key headline for your problem**: the abstract states verbatim that prior structure-augmented RAG's "performance on more basic factual memory tasks drops considerably below standard RAG. We address this unintended deterioration." HippoRAG 1's NER-seeded PPR was entity-centric and noisy; HippoRAG 2 is the published fix for exactly the "graph lane hurts simple queries" regression.
- **Measured results** (Llama-3.3-70B, NV-Embed-v2): Recall@5 avg 78.2 vs 73.4 for the best dense retriever (MuSiQue 74.7 vs 69.7 = +5.0; 2Wiki 90.4 vs 76.5 = +13.9; **and it also wins on simple NQ: 78.0 vs 75.4** — the first graph system to not lose on simple QA). QA F1 avg 59.8 vs 57.0 dense; GraphRAG 49.6, RAPTOR 48.8, HippoRAG 1 53.1, LightRAG a catastrophic 6.6.
- **Ablations (mitigation evidence)**: query-to-triple vs NER-to-node = **+12.5 avg Recall@5**; removing passage nodes: 87.1 → 81.0 avg (multi-hop); removing the recognition-memory filter: 87.1 → 86.4 (modest +0.7 but consistent); passage-node weight sweep {0.01, 0.05, 0.1, 0.3, 0.5} → 0.05 optimal; higher weights *hurt* (77.9 at 0.5 on MuSiQue vs 80.5 at 0.05).
- **Failure modes (Appendix E, 100 failed retrievals)**: triple filtering and graph search are the two main error sources — in **26% of failures no supporting-doc phrase survived filtering** (over-aggressive pruning), in 18% filtering left zero triples (dense fallback triggered), and in 50% seeds were correct but PPR still failed to surface the right passages in top-5. Graph construction itself failed in only 2%.
- **Cost**: indexing 99.5 min on 11.6k passages (vs 12.1 min dense, 277 min GraphRAG, 235 min LightRAG); 1.2 s/query (vs 0.3 dense, 10.7 GraphRAG, 13.3 LightRAG); highest GPU memory (9.9 GB, fact embeddings). Graph on 11.6k passages: 97k nodes, 1.4M edges, of which **1.13M are synonym edges** (8:1 over extracted relation edges).
- **Applicability to swarm-brain**: (a) your "seeds = top direct-RRF results" is structurally the analog of HippoRAG 2's passage-node seeding — but their lesson is these seeds must be **heavily downweighted (0.05)** relative to the direct lanes, i.e., the graph lane's fusion weight should be small by default; (b) the empty-filter → skip-graph fallback is a proven, cheap gate: if no seed passes a relevance filter above threshold, return direct RRF unchanged; (c) query-to-triple linking ≈ your query-text gate — but they apply the gate to *seeds before expansion*, not just to neighbors, and use an explicit filtering step (cross-encoder or LLM) on the seed set. Their error analysis warns the filter threshold matters: too aggressive costs multi-hop recall (their 26% failure bucket).

### Microsoft GraphRAG (2024, still the reference "global" system)
- **Sources**: https://arxiv.org/abs/2404.16130 (Edge et al., "From Local to Global"); dataflow docs https://microsoft.github.io/graphrag/index/default_dataflow/; dynamic community selection: https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/ (Nov 2024).
- **Mechanism**: LLM entity/relation extraction → entity graph → **hierarchical Leiden community detection** (precomputed offline) → LLM community summaries → global search answers via map-reduce over community summaries; local search anchors on entities and pulls neighborhoods.
- **Measured failure mode**: on GraphRAG-Bench (see §2), MS-GraphRAG global consumed **331,375 prompt tokens per query** (vs 879 for vanilla RAG) and had the worst fact-retrieval accuracy of all systems; Context Relevance on medical fact queries collapsed to 2.7–7.5% despite 65–89% recall — extreme high-recall/low-precision. Dynamic community selection (rate community relevance with a cheap LLM before descending; prune irrelevant subtrees) cut global-search cost ~77% at comparable quality.
- **Applicability**: the Leiden-communities-as-offline-job pattern maps directly to an async CockroachDB job writing `community_id` per memory row; Zep's label-propagation variant is even more SQL-friendly because it updates incrementally. Dynamic community selection is a template for "confidence-gated descent": rate before expanding.

### LazyGraphRAG (Microsoft, Nov 2024)
- **Source**: https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/ (Edge, Trinh, Larson); integrated into Microsoft Discovery/Azure Local as of June 2025.
- **Mechanism (verified from blog)**: index uses **NLP noun-phrase extraction + co-occurrence** to build a concept graph — **zero LLM calls at index time**; graph statistics + community structure computed offline. At query time: LLM expands query into 3–5 subqueries; chunks ranked best-first by embedding similarity; communities ranked by their top-k chunks; a **cheap LLM sentence-level relevance assessor** tests chunks in community rank order (breadth-first), **recursing into sub-communities after z consecutive zero-relevant communities** (iterative deepening); stops at a "relevance test budget." One scalar knob (budget: 100/500/1500) controls cost↔quality.
- **Measured**: index cost = vector-RAG cost = 0.1% of full GraphRAG. At budget 500 (4% of GraphRAG global query cost) it "significantly outperforms all conditions on both local and global queries" (win-rate eval on 5,590 AP news articles, 100 queries vs 8 baselines incl. GraphRAG local/global/DRIFT, RAPTOR, 64k-context vector RAG); >700× lower query cost than GraphRAG global at comparable global quality.
- **Applicability**: two transplantable ideas. (1) **Co-occurrence concept graphs are competitive without LLM extraction** — validates building memory-links from cheap signals (shared noun phrases, shared file paths/symbols/commits) rather than LLM relation extraction. (2) The **relevance-test budget with early termination** ("z successive misses → stop descending") is a decoy defense: your traversal could stop expanding a seed after N consecutive neighbors fail the query-text gate, instead of always exhausting 8 neighbors × 2 hops.

### LightRAG (HKU, EMNLP 2025 Findings)
- **Sources**: https://aclanthology.org/2025.findings-emnlp.568.pdf; https://github.com/hkuds/lightrag; https://lightrag.github.io/.
- **Mechanism**: LLM builds an entity–relation graph with per-element key-value text; **dual-level retrieval**: low-level (specific entities and their neighbors) + high-level (abstract "themes" via relation/global keys extracted from the query); union of both fused with vector retrieval; incremental index updates supported.
- **Measured reality check**: strong in its own paper's win-rate evals, but in independent evals it underperforms badly: HippoRAG 2's benchmark gives it **F1 6.6 avg** (its LLM-generated index text pollutes the corpus); GraphRAG-Bench shows decent recall (73–87%) but weak context relevance (33–38%) — systematic over-retrieval of loosely related graph elements; ~4,200 prompt tokens/query.
- **Applicability**: mainly a cautionary tale — injecting graph-derived *generated text* into results is the noise mechanism; retrieval should use the graph only for *ranking/reaching* stored memories (which swarm-brain already does). Its query keyword split (specific vs. thematic) is a cheap query-type signal for gating.

### 2026 successors (routing/robustness generation)
Covered in §2–§3: EA-GraphRAG (Feb 2026), PAGE-RAG (Jul 2026), CS-RAG (Mar 2026), plus HopRAG (ACL Findings 2025, https://aclanthology.org/2025.findings-acl.97.pdf — passage graph with **LLM pseudo-query edges**; retrieve-reason-prune traversal; reports 76.78% higher answer accuracy than plain dense on multi-hop; edges are directed "logical" links between chunks — an analog of your typed memory links) and FlowRAG (https://arxiv.org/html/2606.17856v1, quad-level heterogeneous graph, flow-diffusion retrieval whose complexity scales with retrieved-subgraph size, not full graph — same argument that motivates your bounded traversal).

---

## 2. When graph expansion HURTS: published analyses of the exact regression you measured

### GraphRAG-Bench / "When to use Graphs in RAG" (Xiamen U/HKPU, ICLR 2026) — the definitive study
- **Source**: https://arxiv.org/abs/2506.05690 (v2 Oct 2025; ICLR 2026 poster https://iclr.cc/virtual/2026/poster/10007992); benchmark https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.
- **Design**: 4,076 questions over two corpora (NCCN medical guidelines = dense explicit structure; Gutenberg novels = implicit structure), 4 difficulty levels; difficulty measured on two axes — **Knowledge Breadth (# triples needed)** and **Reasoning Depth (# inference hops)** — not raw hop count. 11 systems tested (MS-GraphRAG local/global, LightRAG, HippoRAG 1/2, Fast-GraphRAG, RAPTOR, LazyGraphRAG, KGP, StructRAG, KET-RAG) vs vanilla RAG ± reranker.
- **Core findings (quoted)**:
  - "Basic RAG is comparable to or outperforms GraphRAG in simple fact retrieval tasks... GraphRAG's extra graph-based processing may introduce redundant or noisy information for simpler queries, degrading answer quality." Numbers: Novel fact-retrieval ACC — RAG+rerank 60.9 vs MS-GraphRAG 49.3, HippoRAG 52.9, RAPTOR 49.3.
  - Crossover point: at breadth ≈1.3/depth ≈1.8 (Level 1) RAG wins; from breadth ≥2.6/depth ≥5 (Level 2+) graph wins decisively (HippoRAG Evidence Recall 87.9–90.9% on Levels 2–3; HippoRAG2 Complex-Reasoning ACC 53.4 vs RAG 42.9).
  - Cites prior measurements: "GraphRAG achieves **13.4% lower accuracy on Natural Questions** compared to vanilla RAG, with particularly poor performance on time-sensitive queries (**16.6% accuracy drop**)... 2.3× higher latency."
  - Noise mechanism named explicitly: "the graph used in GraphRAG introduces several logically relevant but redundant information" — i.e., *decoys that are graph-connected but query-irrelevant*, precisely the decoy-heavy MRR regression.
  - Recommendations: (1) "Prioritize precise retrieval" — minimize redundancy; (2) "Build quality graphs, not just large ones" — HippoRAG2's dense, high-clustering graph (avg degree 8.75, clustering 0.657) outperformed sparse graphs (MS-GraphRAG degree 1.48); (3) "Actively manage context growth" — search boundaries.
- **Applicability**: gives you the routing criterion empirically: **gate the graph lane on estimated knowledge-breadth/reasoning-depth of the query**, and the graph-quality criterion: your memory-link graph's value depends on density/clustering — worth computing avg degree and clustering coefficient of the swarm-brain link graph as a health metric (trivial SQL aggregates).

### CS-RAG / "Toward Robust GraphRAG" (Mar 2026)
- **Source**: https://arxiv.org/abs/2603.14828 (v2 May 2026); code https://github.com/myz12138/CS-RAG/.
- **Finding**: two KG issue modes cause failures — **"spurious noise induces retrieval drift toward plausible but unsupported triples"** (the decoy mechanism, formally named) and "incomplete information leads to retrieval hallucination by forcing continuation through under-supported graph structure." Mitigation without KG repair: plan the query as **ordered atomic constraints**, do anchor- and relation-aware retrieval per constraint, and run a **sufficiency check before each propagation step** — expand another hop only if current evidence "can safely induce variable bindings"; otherwise fall back to textual (dense) recovery. Stable under controlled KG-issue injection; less sensitive to which LLM built the graph.
- **Applicability**: the sufficiency check is a per-hop gate: in your traversal, only take hop 2 from a node if hop-1 evidence actually matched the query above a threshold (not just the 0.60 floor per neighbor, but an aggregate "did hop 1 produce anything good" check). "Forcing continuation through under-supported structure" is exactly what a fixed 2-hop-always traversal does on decoy queries.

### GraphRAG-FI (Michigan State, Mar 2025)
- **Source**: https://arxiv.org/abs/2503.13804 (Guo, Shomer, Zeng, Han, Wang, Tang).
- **Finding/mechanism**: identifies "(1) retrieving noisy and irrelevant information can degrade performance and (2) excessive reliance on external knowledge suppresses the model's intrinsic reasoning." Fix: **two-stage filtering** of retrieved graph information + **logits-based selection** deciding when to trust retrieval vs the model. Significant reasoning gains across backbones on KGQA.
- **Applicability**: supports a two-stage design: coarse filter (cheap: trigram/embedding gate — you have this at 0.60) then fine filter (rerank expanded candidates against the query with your strongest scorer before letting them into final RRF, rather than admitting them with only the traversal-time gate).

### EA-GraphRAG (Feb 2026) — see §3; its Table 1 independently reconfirms the regression (HippoRAG NQ 57.2 vs ColBERTv2 68.7).

### HippoRAG 2 — its entire design is the mitigation catalog: seed filtering (recognition memory), seed downweighting (0.05), query-contextualized linking, dense fallback on empty seeds. See §1.

**Synthesis for the decoy-MRR fix — mitigation strategies with published evidence:**
1. **Query classification to skip the graph** — EA-GraphRAG (best evidence, §3), Adaptive-RAG lineage.
2. **Confidence-gated expansion** — CS-RAG sufficiency check per hop; LazyGraphRAG early termination after z consecutive misses; HippoRAG 2 empty-filter fallback.
3. **Seed quality filtering** — HippoRAG 2 recognition memory (+0.7 recall; but 26% of failures were over-filtering — keep threshold moderate); GraphRAG-FI two-stage filtering.
4. **Downweight, don't drop** — HippoRAG 2's 0.05 passage-seed factor and EA-GraphRAG's continuous complexity-weighted RRF (§3) both show the winning move is a *soft* weight on the graph contribution, tuned low.
5. **Edge-weight learning** — weakest published support; see §5.

---

## 3. Adaptive / routed graph retrieval — per-query decisions

### EA-GraphRAG — "Use Graph When It Needs" (HK PolyU, Feb 2026) — the closest published blueprint for the gating problem
- **Source**: https://arxiv.org/abs/2602.03578 (Dong, Zhang, Xiao, Chen, Zhou, Huang).
- **Mechanism (verified from full paper)**:
  1. **Syntactic feature constructor** (no LLM, no embedding model): ~85 features from constituency parse (Stanza) — words/sentences/clauses/T-units/coordinate phrases/complex nominals/verb phrases and their ratios (C/S, DC/T, CN/T, VP/T...); dependency-parse features (max/avg dependency distance, long-range dependency counts, relation-type counts, tree depth/width/branching); semantic-lexical features (named-entity counts and types, entity density, question-type indicators, negation/passive/coordination markers); interaction terms (entities per token, depth per token, connectors per clause). Mutual-information feature selection, z-scored.
  2. **Complexity scorer**: small MLP (256/128/64, feature attention, residuals) → sigmoid score s(q) ∈ (0,1). Trained as binary classification "will GraphRAG beat RAG on this query?" using **only ~200 disagreement samples** — queries where dense and graph pipelines disagreed (one right, one wrong) on a held-out split. BCE + label smoothing, AdamW.
  3. **Routing**: s(q) ≥ τ_H → graph only; s(q) ≤ τ_L → dense only; in between → **complexity-aware weighted RRF**: `RRF(c) = (1−s(q))·1/(k+r_dense(c)) + s(q)·1/(k+r_graph(c))` — the graph lane's RRF weight IS the complexity score.
- **Measured**: Mix benchmark (NQ+PopQA+HotpotQA+2Wiki, 4k queries): Acc 71.6 / GPT-Acc 76.9 vs HippoRAG2 68.5/73.6, ColBERTv2 65.3/70.2; single-hop NQ 69.1 (beats both the graph and dense systems it routes between); 2Wiki 76.3 vs HippoRAG2's 67.7. Latency: dense 0.08 s, graph 3.23 s, router-only system 1.14 s, full with fusion 2.19 s ("96.4% reduction" on dense-routed queries). Ablation: dense-only 65.1 → +graph 70.5 → +fusion 71.6.
- **Applicability — this is the highest-leverage transplant**: the final weighted RRF already exists; make the graph lane's weight `w_graph = s(q)` computed by a tiny classifier. In swarm-brain's setting the features can be even cheaper and domain-specific: token count, count of distinct symbols/paths/commit-refs in the query, presence of exact-match signals (if the exact/FTS lanes score very high, the query is single-fact → suppress graph), conjunction/comparison words ("and", "between", "both", "why", "history of"), and count of distinct entities. Critically, their training recipe fits the measured benchmark exactly: there are already per-query labels of "graph lane helped / hurt MRR" — **train on the disagreement set** (queries where graph-on and graph-off rankings differ), needing only a few hundred examples. A logistic regression over ~10 hand features would likely capture most of the gain and runs in microseconds in app code.

### Adaptive-RAG (NAACL 2024) — the lineage root
- **Sources**: https://aclanthology.org/2024.naacl-long.389/; https://arxiv.org/abs/2403.14403; code https://github.com/starsuzi/Adaptive-RAG (Jeong et al.).
- **Mechanism**: a small classifier (T5-large) predicts query complexity into three classes — no retrieval / single-step retrieval / iterative multi-step retrieval — trained on **silver labels**: (a) which strategy actually answered each training query correctly (smallest sufficient strategy wins ties), (b) dataset-inherent bias (single-hop datasets → simple, multi-hop datasets → complex). No human labels.
- **Measured**: on six QA datasets (SQuAD/NQ/TriviaQA single-hop; MuSiQue/HotpotQA/2Wiki multi-hop) it matches always-multi-step accuracy at a fraction of the cost and beats always-single-step; the classifier itself is far cheaper than one retrieval round.
- **Applicability**: validates outcome-derived silver labels — swarm-brain's benchmark harness can label queries automatically (did graph expansion improve or degrade RR for this query?), no annotation needed.

### PAGE-RAG (Jul 2026)
- **Source**: https://arxiv.org/abs/2607.19301 (Chen, An, Guo, Wang); code https://github.com/CXY0112/PAGE-RAG.
- **Mechanism**: treats auto-built graphs as "inherently incomplete projections" — a **semantic skeleton for navigation, never a replacement knowledge source**; **query-adaptive retrieval routing** selects among retrieval operators (textual evidence, graph traversal, hybrid) per query needs; strict knowledge-boundary control (abstain rather than extend past evidence). Reports competitive answer quality with improved retrieval efficiency and reliability on long-document QA.
- **Applicability**: philosophical match for swarm-brain: memory links should *navigate to* stored memories (which then get scored on their own text), never contribute score merely for being connected. If a neighbor can't justify itself against the query text, its link provenance shouldn't rescue it — supports scoring expanded candidates with the same text-relevance machinery as direct hits, using link weight only as a tiebreaker/prior.

### GraphRAG dynamic community selection (Microsoft, Nov 2024)
- **Source**: https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/ — cheap-LLM relevance rating of community summaries before descending the hierarchy; prunes irrelevant branches; ~77% cost cut. The "rate-before-expand" pattern at community granularity.

---

## 4. PPR vs bounded BFS; async precomputed projections

### PPR (HippoRAG family) vs bounded BFS (Zep/Graphiti, AriGraph, the swarm-brain design)
- **PPR characteristics** (HippoRAG 2, §1): global stationary distribution; damping 0.5 (i.e., heavy weight on staying near seeds — a low damping factor is itself a locality bound); naturally blends multiple seeds and *soft* multi-hop influence with geometric decay — the 0.85/hop decay approximates the same geometry. Cost: full-graph sparse matrix iteration; HippoRAG runs it in-process with python-igraph in ~1.2 s/query including LLM filtering. Failure mode (their error analysis): even with correct seeds, PPR mass leaks to high-degree hub nodes — 50% of their failures had good seeds but bad top-5. Hub leakage is worse for PPR than for gated BFS, because PPR has no per-node query gate.
- **Bounded BFS characteristics** (Zep paper §3.1, https://arxiv.org/html/2501.13956v1): "breadth-first search reveals contextual similarities — nodes and edges closer in the graph appear in more similar conversational contexts"; Graphiti's BFS accepts arbitrary seed nodes (they seed with *recent episodes* — a temporal prior) and searches n-hops. BFS is exact, cheap at small fan-out, trivially expressible as batched SQL `WHERE src IN (...)` per hop, and allows per-neighbor gating — which PPR cannot do mid-walk.
- **Published tradeoff verdict**: no paper directly A/Bs PPR against gated bounded BFS on the same graph. But the evidence pattern is: PPR wins when the graph is dense and high-quality (HippoRAG 2's degree-8.75 graph) and seeds are filtered; gated BFS wins for dynamic graphs, low latency, and precision control. GraphRAG-Bench's "build quality graphs" finding implies the graph's density determines which regime you're in. A middle path exists: **push-based approximate PPR (Andersen-Chung-Lang style forward push) is application-side implementable over SQL adjacency reads** with a residual threshold, giving PPR-like soft decay while keeping the bounded-read property. OpenReview "Query-Aware Flow Diffusion" (https://openreview.net/forum?id=n28wnc2QTc) proves such local algorithms converge with complexity scaling in retrieved-subgraph size only.

### Async precomputed projections that fit "offline job → SQL table"
- **Community detection**: MS-GraphRAG hierarchical Leiden (offline batch, https://microsoft.github.io/graphrag/index/default_dataflow/); **Zep/Graphiti uses label propagation instead of Leiden explicitly because it supports incremental single-step updates** — "when the system adds a new entity node... it surveys the communities of neighboring nodes [and] assigns the new node to the community held by the plurality of its neighbors," with periodic full refreshes to correct drift (Zep paper §2.3). That plurality-vote update is a single SQL query per insert; a periodic full label-propagation job is a few dozen lines of app code writing `community_id` back to a table. Use: community-level gating (skip expansion when seed communities are disjoint from query-matched communities), diversity control, and community summaries as retrievable memories.
- **Node embeddings**: node2vec community-recovery guarantees: NeurIPS 2024 https://papers.nips.cc/paper_files/paper/2024/hash/015a8c69bedcb0a7b2ed2e1678f34399-Abstract-Conference.html and https://arxiv.org/abs/2310.17712 (node2vec embeddings consistently recover (degree-corrected) SBM communities); Nature Communications 2024 https://www.nature.com/articles/s41467-024-52355-w (DeepWalk/LINE/node2vec provably resolve communities down to a detectability threshold). Practitioner pattern of computing graph embeddings next to the data store: https://memgraph.com/blog/scaling-graphrag-embeddings; hybrid Node2Vec+text-embedding retrieval studied in https://arxiv.org/abs/2605.18410 (Feb 2026 — structural + semantic embeddings combined beat either alone on citation-graph retrieval). Use in swarm-brain: nightly node2vec over the memory-link graph (random walks are just repeated adjacency reads), vectors stored in a `graph_embedding` column; then "graph-nearby" becomes an O(1) vector similarity check at query time — a decoy-resistant *replacement* for runtime traversal on borderline queries, since structural proximity is consulted without admitting any un-gated neighbor.
- **Precomputed PPR**: for stable "hub" memories, top-N PPR vectors can be materialized offline into a table (`seed_id, node_id, ppr_score`), refreshed async — the AWS reference implementation (https://aws.amazon.com/blogs/machine-learning/hipporag-neurobiologically-inspired-rag-using-amazon-bedrock-amazon-neptune-and-personalized-pagerank/, Jul 2026) runs PPR outside the serving path for the same reason.

---

## 5. Temporal knowledge graphs and typed-edge ranking

### Zep / Graphiti (Jan 2025) — the temporal reference design
- **Sources**: https://arxiv.org/abs/2501.13956 (Rasmussen, Paliychuk, Beauvais, Ryan, Chalef); code https://github.com/getzep/graphiti.
- **Temporal mechanism (verified from paper §2.2.3)**: **bi-temporal model with four timestamps per edge** — transactional timeline (t′_created, t′_expired: when the system learned/retired the fact) and validity timeline (t_valid, t_invalid: when the fact held true in the world); handles absolute and relative time expressions against the episode's reference time. **Edge invalidation**: new edges are compared by an LLM against semantically related existing edges; on temporally-overlapping contradiction, the old edge's t_invalid is set to the new edge's t_valid — "Graphiti consistently prioritizes new information." Nothing is deleted; point-in-time queries remain possible.
- **Retrieval**: three lanes — cosine, BM25, and n-hop BFS (seeds can be recent episodes) — then rerankers: RRF, MMR, **episode-mentions frequency** (frequently-referenced facts rank higher), **node-distance from a designated centroid node** (locality reranking), and cross-encoder (most accurate, most expensive). Facts are rendered into context *with their valid-from/valid-to ranges*.
- **Measured**: DMR 94.8% vs MemGPT 93.4%; LongMemEval up to +18.5% accuracy with **90% lower latency** than full-context baselines (115k-token conversations); biggest wins on cross-session synthesis and temporal reasoning.
- **Applicability**: swarm-brain is a temporal memory kernel — the bi-temporal columns (valid/invalid + created/expired) and contradiction-invalidation are directly implementable as SQL columns plus a write-time check comparing a new memory against link-adjacent memories about the same artifact. Two Zep rerankers are cheap SQL wins: mention-frequency (count of link references to a memory as a popularity prior) and node-distance-to-centroid (rerank by hop distance from the current task's focus memory). And: **stale-edge invalidation is a decoy mitigation** — decoys in an agent-swarm memory are often *superseded* memories about the same file/symbol; time-scoping edges removes them from expansion.

### Typed-edge ranking: learned vs hand-set
- **State of the literature**: no strong published result directly learns *relation-type* scalar weights for RAG graph traversal; the practice in HippoRAG 2 (relation vs synonym vs context edges), Graphiti (typed relations, hand-treated), and LightRAG is hand-set or untyped. The closest published work:
  - **Weak-to-Strong GraphRAG / ReG** (https://arxiv.org/abs/2506.22518; ICLR 2026 submission https://openreview.net/forum?id=GtjELGHkPB): graph retrievers trained with weak supervision retrieve spurious paths; ReG uses **LLM feedback to refine the weak supervision signal** (removing noisy positive/negative path labels) and structure-organizes evidence, improving KGQA and transferring across LLMs. Mechanism-level lesson: what's worth learning is *which links are good evidence paths*, and outcome feedback (did this path help answer?) is a sufficient training signal.
  - **GNN-based edge scoring** (G-Retriever, GNN-RAG lineage) learns edge importance implicitly but needs a GNN stack — out of scope for SQL+app constraints.
- **Applicability**: the typed-relation weights can be *fit, not learned online*: with the measured benchmark, run coordinate ascent / logistic regression on per-type weights (and the hop-decay constant) against MRR, using EA-GraphRAG-style disagreement examples. That's an offline job writing a small weights table — the same infrastructure as the router in §3, no model serving required. Prior expectation from the literature: gains from tuned type weights are second-order compared to seed gating and fusion-weight gating.

---

## 6. Graph construction at write time from agent output

- **LLM entity/relation extraction — quality and cost**: HippoRAG 2's OpenIE is the quality ceiling for open extraction, but its own graph stats show the extracted relation edges (140k) are dwarfed by cheap **synonym edges (1.13M)** — i.e., even the flagship system gets most of its connectivity from embedding-similarity links, not LLM relations. GraphRAG-Bench shows extraction-heavy indexes cost 12–13× more tokens than the corpus itself (GraphRAG 115.5M input tokens on an 11.6k-passage corpus) and produce highly variable graph quality (avg degree 1.48–13.31 across systems). CS-RAG (§2) documents that LLM-built KGs systematically contain spurious triples (→ retrieval drift) and gaps (→ hallucinated continuation) — so a write-time extraction pipeline must expect an imperfect graph and put robustness in the *retriever* (their thesis: mitigate at retrieval, don't try to repair the KG).
- **Lightweight alternatives with published validation**:
  - **Noun-phrase co-occurrence** — LazyGraphRAG (§1): a concept graph from NLP noun phrases + co-occurrence, no LLM, supports SOTA-quality retrieval when paired with query-time relevance testing. Direct precedent for cheap link construction.
  - **LLM pseudo-query edges** — HopRAG (https://aclanthology.org/2025.findings-acl.97.pdf): edges between chunks created by generating questions a chunk raises and linking to chunks that answer them; "logic-aware" traversal follows edges whose pseudo-queries match the user query. Expensive at write time but a good fit for *selective* linking of high-value memories.
  - **KET-RAG** (https://arxiv.org/abs/2502.09304): hybrid indexing — full LLM KG ("skeleton") only for a PageRank-selected core of important chunks, plus a cheap **keyword–chunk bipartite graph** for everything else; retains most of full-KG quality at ~an order of magnitude lower indexing cost. Template for tiering: expensive links for hub memories, cheap links elsewhere.
  - **Shared-artifact links**: no paper evaluates file-path/commit links specifically, but three lines of evidence support them strongly: Zep episodic edges (memory ↔ episode-of-origin) are exactly "shared provenance" links and carry much of Graphiti's retrieval value; software-engineering RAG on issue databases (LinkSO/Astute-style, arXiv 2404.17723) exploits explicit artifact links; and the co-occurrence result above generalizes — **a link "same file path / same symbol / same commit" is a co-occurrence edge over a controlled vocabulary with near-zero false-positive rate**, i.e., *higher precision than LLM-extracted relations*. Since swarm-brain memories already carry paths/symbols/commits, these deterministic links should be the highest-weight edge types; LLM-extracted semantic relations, if added, should get lower weight (they carry CS-RAG's spurious-noise risk).

---

## Direct answers for the two asked-for outcomes

**Fixes for the decoy-heavy MRR regression (ranked by strength of published evidence):**
1. **Complexity-weighted graph-lane fusion** (EA-GraphRAG, https://arxiv.org/abs/2602.03578): make the graph lane's weight in the final RRF equal to a per-query complexity score s(q) from a tiny classifier trained on ~200 disagreement examples from the benchmark (labels: graph helped / hurt RR). Their measured result: beats both always-graph and never-graph on every dataset, including single-hop.
2. **Skip-graph fallback on weak seeds** (HippoRAG 2): filter seeds against the query before traversal (cross-encoder or the dense scorer); if none pass, return direct RRF untouched. Costs ~0 on multi-evidence queries, eliminates the graph lane exactly when it decoys.
3. **Downweight, don't equal-weight, graph-reached candidates** (HippoRAG 2's 0.05 factor; sweep shows monotone degradation as graph influence rises past a small optimum).
4. **Per-hop sufficiency gating + early termination** (CS-RAG; LazyGraphRAG's z-consecutive-misses rule): only take hop 2 if hop 1 produced above-threshold matches; stop a seed's expansion after N consecutive gate failures.
5. **Temporal edge invalidation** (Graphiti): time-scope links so superseded memories — a major decoy class in agent memory — leave the expansion frontier automatically.
6. **Score expanded candidates with the same text-relevance machinery as direct hits** (PAGE-RAG's "graph as skeleton, not source"): link provenance should be a prior, never a substitute for query-text evidence.

**Per-query gating of the graph lane — concrete recipe synthesized from the literature:** features = query length, distinct entity/symbol/path/commit count, conjunction-comparison-causal markers, top-1 exact/FTS lane score (high → single-fact → suppress graph), dense–lexical lane agreement; model = logistic regression or small MLP (EA-GraphRAG's is 256/128/64 but their evidence suggests most signal is in the features); labels = silver, from benchmark outcomes (Adaptive-RAG's recipe); output = continuous s(q) used both as the RRF weight of the graph lane and (thresholded τ_L/τ_H) to skip traversal entirely below τ_L — reclaiming the graph lane's latency (their dense-routed path is 40× faster) as a bonus.

Highest-value full-text reads: arXiv 2602.03578 (EA-GraphRAG), 2502.14802 (HippoRAG 2), 2506.05690 (GraphRAG-Bench), 2501.13956 (Zep/Graphiti), 2603.14828 (CS-RAG), and the LazyGraphRAG blog post.

---

# Report 3 — Hybrid Fusion, Reranking, Calibration, Abstention, Embeddings (2025–2026)

## 1. Score Calibration & Abstention / No-Answer

### The core diagnosis: top-k ranking is the wrong abstraction for abstention
- Source: "The Retrieval Emptiness Problem" — https://tianpan.co/blog/2026-04-16-rag-retrieval-abstention-empty-corpus (Apr 2026); Google Research "Sufficient Context" (ICLR 2025, https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/).
- Vector search always returns k "least-bad" neighbors; distance ≠ relevance. Sufficient-context study: with sufficient context Gemma hallucinates ~10%; with insufficient (off-topic but retrieved) context → **66%**. Reframe: retrieval is a **classification problem whose output space includes null**.
- On thresholds: raw cosine floors "necessary but not sufficient": (a) fixed numbers are folk defaults — distributions are model/corpus-specific; (b) global threshold wrong even within one system — specific factual queries match at 0.75+, exploratory at 0.5–0.6; per-query-class thresholds capture **2–3x** precision improvement of global cutoff; (c) production pattern: **"retrieve wide (50–100 candidates, no strict floor), classify narrow (cross-encoder with calibrated threshold)"**.
- Applicability P1: fix is not a better floor on fused RRF output — it's a separate abstention channel.

### Calibrated reranker scores as the abstention signal (zerank-2)
- https://zeroentropy.dev/articles/smarter-context-compression-for-llm-pipelines-zerank-2-as-a-calibrated-classifier/ (Apr 2026); https://huggingface.co/zeroentropy/zerank-2
- Multilingual instruction-following cross-encoder trained via zELO so **score is a calibrated relevance probability: 0.8 ≈ 80% chance of relevance**. Binary classifier: `score >= threshold → relevant; else discard`; nothing clears threshold → abstain. Threshold bands (0.2 high-recall, 0.4 default, 0.6 high-precision, 0.8 near-certain) + calibration recipe.
- Numbers: clinical pipeline 85%+ context compression at 90%+ recall; ~50–100x cheaper than LLM relevance classification; p50 ~130ms (API).
- **ZeroEntropy acquired by Notion (Jul 24, 2026), models relicensed Apache 2.0** (https://zeroentropy.dev/articles/zeroentropy-is-joining-notion/) — zerank-2, -small, -nano now open-weight/self-hostable.
- Most direct fix for no-answer=0.00; nano variant over top ~10 fused candidates gives abstention decision RRF cannot.

### Conformal prediction for retrieval/RAG abstention
- **TRAQ** (NAACL 2024, cited 45): https://aclanthology.org/2024.naacl-long.210.pdf — conformal prediction over retrieval+generation; calibration set of (query, relevant-passage-score) pairs → score quantile guaranteeing true passage inclusion with prob ≥ 1−α; below quantile → abstain.
- **Conformal Abstention for RAG** (TMLS, Jul 2026): https://www.tmls.nyc/research/conformal-abstention-rag — abstention as threshold on ANY uncertainty score with distribution-free finite-sample guarantee; score needn't be calibrated itself.
- UQ for RAG survey (2025): https://arxiv.org/html/2510.11483v2 — abstention function as first-class pipeline component.
- Cheapest guaranteed fix, no new model: log per-lane **raw** scores alongside RRF; calibration set of in-corpus + out-of-corpus queries; conformal-threshold on max-raw-cosine or monotone combination. Works today with hash embedder, transfers via re-calibration.

### No-answer / unanswerability benchmarks
- **UAEval4RAG** (Salesforce, ACL 2025): https://aclanthology.org/2025.acl-long.415.pdf, code https://github.com/SalesforceAIResearch/Unanswerability_RAGE — **synthesizes unanswerable request sets for any corpus**, 6-way taxonomy (underspecified, false-presupposition, nonsensical, modality-limited, safety-concerned, **out-of-database**). Metrics: unanswerable-acceptance ratio + answerable accuracy jointly.
- Evidence-Calibrated RAG for Unanswerable QA (Jul 2026): https://www.researchgate.net/publication/407482949 — separates retrieval-coverage prediction from abstention calibration.
- Applicability: auto-generate out-of-corpus no-answer eval set for swarm-brain memories instead of hand-writing.

### Per-lane acceptance thresholds
- "Beyond the Reranker" (Jun 2026): https://arxiv.org/html/2606.28367v1 — cross-encoder reranker accounts for most pipeline quality; **separate acceptance threshold per score source** rather than one on fused output. "Gate on lanes, fuse for order."

## 2. Fusion Beyond Vanilla RRF

### Rank fusion vs normalized score fusion
- **Bruch et al., ACM TOIS 2023**: https://dl.acm.org/doi/10.1145/3596512 — RRF sensitive to k and lane count; **TM2C2 (convex combination of theoretically-min-max-normalized scores) "significantly outperforms RRF on all datasets"**; convex weights sample-efficient (handful of labeled queries).
- **"Experimental Analysis of Trade-offs in Hybrid Search"** (arXiv Aug 2025, https://arxiv.org/abs/2508.01405) — 4 hybrid paradigms, 11 datasets: tensor/late-interaction fusion consistently outperforms RRF; documents **"weakest-link degradation"** — bad lane drags fusion down → motivates per-query lane skipping; publishes decision map.
- OpenSearch RRF (Feb 2025): https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/ ; MongoDB RRF vs Relative Score Fusion: https://medium.com/mongodb/reciprocal-rank-fusion-and-relative-score-fusion-classic-hybrid-search-techniques-3bf91008b81d — RSF preserves score magnitude RRF throws away.
- RRF k-sensitivity: https://www.emergentmind.com/topics/reciprocal-rank-fusion-rrf (Nov 2025); k=60 is historical default not optimum; MariaDB docs treat k as primary knob (https://mariadb.com/docs/server/reference/sql-structure/vectors/optimizing-hybrid-search-query-with-reciprocal-rank-fusion-rrf).
- Applicability: keep RRF for ordering + parallel TM2C2-style normalized score channel for gating/public score; or replace final fusion with TM2C2 convex fusion (ranks better AND preserves absolute signal).

### Learned fusion / LTR
- No strong dedicated 2025–26 work for LTR-over-lane-features at small scale; energy went to convex score fusion + rerankers. Multi-stage hybrid framework (https://www.researchgate.net/publication/404870322): RRF improves recall; cross-encoder reranking is primary driver of final quality. Don't over-invest in fancy fusion.

## 3. Rerankers 2025–2026

### mxbai-rerank-v2 (Mixedbread) — Apache 2.0
- https://www.mixedbread.com/blog/mxbai-rerank-v2 (Mar 2025); base-v2 0.5B, large-v2 1.5B; RL-trained (GRPO) on Qwen-2.5; 8K ctx; 100+ langs; trained for **code, SQL, JSON, tool retrieval**.
- BEIR nDCG@10 (BM25 first stage): large-v2 **57.49**; base-v2 **55.57**; vs cohere-rerank-3.5 55.39, bge-reranker-v2-gemma 55.38, jina-reranker-v2-base 54.35, bge-reranker-v2-m3 53.94. Latency (A100, per query): base-v2 0.67s vs bge-v2-m3 3.05s. CPU: 0.5B over top-10 feasible (ONNX int8), top-100 not.

### Qwen3-Reranker (0.6B/4B/8B) — Apache 2.0
- https://qwenlm.github.io/blog/qwen3-embedding/ (Jun 2025); paper https://arxiv.org/pdf/2506.05176 (cited ~479).
- nDCG@10 over top-100 dense: 0.6B: MTEB-R 65.80, **MTEB-Code 73.42**; 4B: 69.76 / **81.20**; 8B: 69.02 / 81.22. bge-reranker-v2-m3 MTEB-Code: 41.38. 32K ctx, instruction-aware. GGUF for llama.cpp CPU serving.
- Caveat: yes/no-logit scorer — not calibrated probabilities; calibrate before abstention use.

### jina-reranker-v3 / v3.5 — listwise 0.6B, license caveat
- https://huggingface.co/jinaai/jina-reranker-v3, paper https://arxiv.org/abs/2509.25085 (Sep 2025). "Last but not late" interaction, ~62 nDCG@10 BEIR at 0.6B. v3.5 mid-2026. **CC-BY-NC likely — probably disqualified.**

### Cohere Rerank 4 — closed API
- https://cohere.com/blog/rerank-4 (Dec 11, 2025). API-only; fails provider-neutral constraint; open models above beat Rerank 3.5.

### ColBERT late interaction
- Still viable; BGE-M3 gives ColBERT vectors free; but more infra complexity than cross-encoder for top-k≤50 window (https://zeroentropy.dev/articles/open-source-alternatives-to-cohere-rerank/, Jan 2026).

### Abstention value of rerankers
- Rerankers give absolute thresholdable score RRF destroys. Redis roundup (https://redis.io/blog/top-reranking-models-rag-accuracy/): most reranker scores relative, must recalibrate; zerank-2 exception. Ranking for swarm-brain: **zerank-2-small/nano > Qwen3-Reranker-0.6B > mxbai-rerank-base-v2**. Abstention fix may justify pulling forward minimal reranker as gate over top-10 only.

## 4. Embedding Models for 1024-dim Slot

### RECOMMENDED: Qwen3-Embedding-0.6B (native 1024-dim)
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B, paper https://arxiv.org/pdf/2506.05176 (Jun 2025).
- Native **1024-dim**, Apache 2.0, 32K ctx, MRL, **instruction-aware**, 100+ langs incl. programming languages, 0.6B CPU-serveable (GGUF ~600MB int8), vLLM/TEI/Ollama/llama.cpp. 8B sibling MTEB-multilingual #1 (70.58). ~7.9% relative improvement over BGE-M3 multilingual (https://medium.com/@mrAryanKumar/comparative-analysis-of-qwen-3-and-bge-m3-embedding-models-for-multilingual-information-retrieval-72c0e6895413). Upgrade path: 4B (2560-d, MRL→1024) same family. Cosine-trained; last-token EOS pooling; keep instructions consistent index/query.

### BGE-M3 — 1024-dim, MIT
- https://huggingface.co/BAAI/bge-m3. 568M, **dense + learned sparse + ColBERT from one pass**, 8192 ctx, MIT. Milvus CCKM (Mar 2026, https://milvus.io/blog/choose-embedding-model-rag-2026.md): 0.940 cross-lingual R@1, 0.973 needle (0.920 @8K). 2024 model; code retrieval not its strength. Attractive if learned-sparse lane wanted.

### snowflake-arctic-embed-l-v2.0 — 1024-dim, Apache 2.0
- https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0, paper https://arxiv.org/html/2412.04506v1. 568M, MRL→256, 8192 ctx, 74 langs, throughput-optimized. No code instruction-tuning; code numbers undocumented.

### MRL caveat — "Matryoshka Is Dead" (ZeroEntropy, Apr 2026)
- https://zeroentropy.dev/articles/matryoshka-is-dead/ — MRL truncation not lossless even for MRL-trained models (<1–2.5% at 256d but nonzero). Prefer native-1024 over truncating larger model unless benchmarked on own corpus.

### jina-embeddings-v4 — 3.8B, CC-BY-NC — skip. EmbeddingGemma — 768-dim max — skip. granite-embedding-r2 — 768-dim — skip.
- jina-v4: https://huggingface.co/jinaai/jina-embeddings-v4 (cited 102); jina-code-embeddings-0.5b/1.5b (Sep 2025, https://arxiv.org/html/2508.21290v1) — verify license.
- EmbeddingGemma: https://developers.googleblog.com/en/introducing-embeddinggemma/ (Sep 2025), 308M, max 768-d, 2K ctx.
- granite-embedding-r2: https://huggingface.co/ibm-granite/granite-embedding-english-r2 (Aug 2025, 149M, 768-d).

### Verdict P2
Qwen3-Embedding-0.6B. Fallbacks: BGE-M3 (sparse/ColBERT signals) or arctic-l-v2.0 (throughput). Re-run two-track benchmark and **re-calibrate all thresholds on swap** — thresholds are model-specific.

## 5. Query Understanding

### Adaptive-RAG — https://arxiv.org/abs/2403.14403 (NAACL 2024, cited 766)
- Small classifier routes {no retrieval, single-step, multi-step}; matches multi-step quality at fraction of cost; headroom is in routing accuracy.
- swarm-brain analog: **lane routing** via regex/heuristic classifier: identifier-like (`/`, `::`, `_`, camelCase, hex, error codes) → exact+trigram, downweight dense; conceptual → dense+FTS; temporal → temporal index; multi-hop → enable graph, else skip. Mitigates weakest-link degradation.

### HyDE in 2026: demoted to routed fallback
- "The Coverage Illusion" (May 2026): https://arxiv.org/html/2605.27220v1 — always-on HyDE adds cost for marginal gain; **route first, rewrite only queries that need it**. EMNLP 2024 Findings (https://aclanthology.org/2024.findings-emnlp.103.pdf, cited 48): HyDE helps dense, poor with lexical retrievers.
- Verdict: don't build HyDE now; trigger only on conceptual+low-confidence branch (needs abstention signal anyway). Cheap multi-query rewriting (2–3 lexical paraphrases via existing RRF) better-evidenced low-cost option (https://haystack.deepset.ai/blog/query-expansion).

## Bottom line (agent 3)
1. Two-channel architecture: RRF/TM2C2 for ordering; parallel evidence channel of per-lane raw scores for abstention gating.
2. Calibrate gate empirically: UAEval4RAG no-answer set + conformal threshold (now) or zerank-2-small/nano calibrated gate (later); τ ≈ 0.3–0.4 start, per-query-class τ for 2–3x precision.
3. Qwen3-Embedding-0.6B for the 1024-dim slot; re-calibrate everything on swap.
4. Reranker shortlist when un-deferred: zerank-2 family, Qwen3-Reranker-0.6B (MTEB-Code 73.42), mxbai-rerank-base-v2; top-10–50 only.
5. Zero-cost heuristic query classifier for lane routing.

---

# Report 4 — Context Engineering, RAG Evaluation, and DB-Native Retrieval (2025–2026)

## 1. Context engineering / context packing for coding agents

### Anthropic "Effective Context Engineering for AI Agents" (Sep 29, 2025)
- **Source:** https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- **Mechanism:** Frames context as a finite "attention budget" degraded by context rot (n² attention stretching). Core prescriptions: (a) smallest set of high-signal tokens; (b) **just-in-time retrieval** — agents hold lightweight identifiers (file paths, stored queries) and load data via tools at runtime, instead of pre-inference retrieval; (c) **hybrid strategy** — Claude Code drops CLAUDE.md in up-front, everything else via glob/grep just-in-time; (d) three long-horizon techniques: compaction, structured note-taking (agentic memory files), sub-agent architectures where a subagent burns tens of thousands of tokens exploring but returns a **1,000–2,000-token distilled summary**.
- **Results:** Qualitative + references Chroma's context-rot measurements; multi-agent system post reports substantial improvement over single-agent on research tasks (https://www.anthropic.com/engineering/multi-agent-research-system, Jun 2025 — lead agent + parallel subagents, each returning condensed summaries; ~15x token cost of chat).
- **swarm-brain applicability:** The "task bootstrap" bundle is exactly the up-front half of the hybrid; the missing half is the just-in-time affordance (see §2). The 1–2k-token subagent-report convention is a good size target for the handoff/checkpoint deterministic-context lane. Follow-up cookbook comparing memory/compaction/tool-clearing strategies with costs: https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools (Mar 2026).

### Compaction/handoff implementations across harnesses (comparative survey, Dec 2025)
- **Source:** https://gist.github.com/badlogic/cd2ef65b0697c4dbe2d13fbecb0a0a5f (verified against openai/codex and sst/opencode source links inside)
- **Mechanisms (per harness):**
  - **Claude Code:** auto-compact at ~95% capacity; LLM summary preserving "accomplished / in-progress / files / next steps / constraints"; continues with summary **plus the 5 most recently accessed files**; `/compact <custom instructions>` for targeted summaries. Known failure: cumulative loss over multiple compactions; "goes off the rails" if fired mid-task.
  - **Codex CLI:** token-threshold trigger (`model_auto_compact_token_limit`, e.g. 180k/244k per model); new history = initial context + **last ~20k tokens of raw user messages** + summary; summary is prefixed with an explicit handoff framing ("Another language model started to solve this problem and produced a summary…").
  - **OpenCode:** **prunes before compacting** — protects last 40k tokens of tool output, prunes older tool outputs when >20k tokens are prunable; overflow check is `tokens > context_limit − output_limit`.
  - **Amp (Sourcegraph):** no auto-compaction; **goal-conditioned handoff** — user states the next goal, a secondary model extracts only goal-relevant information into a fresh thread; thread-references for on-demand extraction from other threads.
- **swarm-brain applicability:** The handoff memory type should adopt the converged schema (accomplished / in-progress / files / next steps / constraints / key decisions), and Amp's **goal-conditioned extraction** is the strongest idea for task bootstrap: condition what is pulled from the prior handoff on the *claimed task's* goal, not just recency. Codex's "keep recent raw messages alongside summary" argues the bootstrap bundle should pair the summary handoff with a few raw recent evidence spans.

### Cursor: semantic search measurably improves agents (Nov 6, 2025)
- **Source:** https://cursor.com/blog/semsearch
- **Mechanism:** Custom embedding model + semantic-search tool *in addition to* grep. Training signal: mine **agent session traces**, have an LLM rank in retrospect what content would have been most helpful at each step, train embeddings to match those rankings.
- **Results:** +12.5% avg QA accuracy (range 6.5%–23.5% across frontier models) on Cursor Context Bench; A/B: +0.3% code retention overall, **+2.6% on codebases ≥1,000 files**; 2.2% more dissatisfied follow-ups without semantic search.
- **swarm-brain applicability:** Direct justification for the dense lane coexisting with exact/FTS lanes (grep-analog). More importantly, the trace-derived training/eval signal is the strongest argument for building the **persistent trace sink lane first**: once traces exist, you can (a) derive gold retrieval labels ("what memory should have been recalled at step t"), and (b) tune RRF lane weights against them — no embedding fine-tune required to get value.

### Lost-in-the-middle status in 2026 + context rot
- **Sources:** original: https://arxiv.org/abs/2307.03172 (Liu et al., 2023/TACL 2024); empirical 2025 update: Chroma "Context Rot" https://www.trychroma.com/research/context-rot (Jul 2025); practice guides https://atlan.com/know/llm/lost-in-the-middle-problem/ (Jun 2026), https://www.getmaxim.ai/articles/solving-the-lost-in-the-middle-problem-advanced-rag-techniques-for-long-context-llms/ (Oct 2025).
- **Status:** The U-shaped position bias is attenuated but **not solved** in 2025–2026 frontier models; Chroma showed performance degrades with input length even on trivially simple tasks, worsened by semantically-similar distractors — i.e., the problem is now framed less as "middle position" and more as "total context load + distractor density." Standard mitigations remain: strategic ordering (most-critical first and last), aggressive filtering over stuffing.
- **swarm-brain applicability:** Order the bootstrap bundle deliberately: handoff + task constraints at the top, playbook at the bottom (both privileged positions), prior attempts/knowledge sections in the middle, and cap bundle size — Chroma's result says a smaller high-precision bundle beats a bigger high-recall one.

### Diversity selection and token-budgeted packing: submodular > MMR-as-default
- **Sources:** "Recall Is Not Enough: A Reader-Context Diagnostic for Budget-Constrained RAG," arXiv:2607.00725 (Jul 2026, v2 Aug 2026) https://arxiv.org/abs/2607.00725; budget-aware MMR routing for clinical text, ACL Findings 2026 https://aclanthology.org/2026.findings-acl.2114.pdf
- **Mechanism (2607.00725):** Introduces **answer-in-context (AiC)** — did the gold answer survive into the *packed* context — as the metric budgeted RAG should optimize, since recall@k is scored on the retrieved set but the reader consumes the packed set. Casts packing as **budgeted submodular maximization** (facility-location/coverage-style objective; MMR is the canonical redundancy-aware baseline it generalizes).
- **Results:** AiC adds ΔR² = 0.17–0.27 over recall across three multi-hop datasets; even when all gold was retrieved, whether packing kept the answer separates exact-match by **4.6×**. The submodular packer beats deployed top-k truncation and LLMLingua-2 compression across 3 reader families, 4 scales, 4 budgets at equal-or-lower token cost; reaches parity with a hand-tuned query-focused heuristic.
- **swarm-brain applicability:** This is the blueprint for the unbuilt **context-packing (MMR) lane**: (1) implement greedy budgeted submodular selection (facility location over embedding similarities + relevance term) rather than plain MMR — same greedy loop, better objective; (2) add **answer-in-context** to the benchmark harness alongside Recall@k/MRR/nDCG — for the gold corpus, check whether the gold memory's key span survives into the token-budgeted bundle, not just the top-k list. Cheap to add and it measures what the agent actually sees.

## 2. Agentic retrieval loops vs single-shot RAG

### SWE-Explore: agentic exploration is a clear tier above classical retrieval (Jun 2026)
- **Source:** https://arxiv.org/abs/2606.07297 (848 issues, 10 languages, 203 repos)
- **Mechanism:** Isolates repository exploration: given repo + issue, return a ranked list of relevant code regions under a fixed **line budget**. Ground truth is line-level, distilled from independent agent trajectories that successfully solved the same issue (the code regions their solution paths actually consulted). Evaluates coverage, ranking, and context-efficiency; shows these track downstream repair behavior.
- **Results:** "Agentic explorers form a clear tier above classical retrieval" (BM25, one-shot lexical or embedding retrieval). File-level localization is largely saturated for modern methods; **line-level coverage and efficient ranking under budget** are the differentiators.
- **swarm-brain applicability:** (a) A single recall call is a "classical retrieval" system by this taxonomy — the memory API should expose **iterative affordances**: a search tool (query + filters), a read/expand tool (fetch full memory + evidence spans + linked memories), and link-following, so agents can loop search→read→refine. (b) The trajectory-derived line-level gold methodology is directly reusable for scaling the eval corpus (§4). (c) Their "budget + coverage + ranking" metric triple maps cleanly onto scoring the task-bootstrap bundle.

### LongMemEval-V2: coding-agent-style evidence gathering crushes RAG memory (May 2026)
- **Source:** https://arxiv.org/abs/2605.12493
- **Results:** On 451 questions over histories up to 500 trajectories/115M tokens, **AgentRunbook-C** (stores trajectories as files, invokes a coding agent in a sandbox to gather evidence) hits **72.5%** avg accuracy vs **48.5%** for the strongest RAG-based memory (AgentRunbook-R, with knowledge pools for raw state observations, events, and strategy notes) and 69.3% for an off-the-shelf coding agent. Caveat: agentic gathering has high latency; it advances but does not close the accuracy–latency Pareto frontier.
- **swarm-brain applicability:** The strongest measured evidence that a memory system serving agents should support **agentic interrogation of the store**, not only one-shot recall. Practical design: keep the deterministic bootstrap bundle for latency-critical task claim, and expose SQL/tool-level iterative search for when the agent needs depth — a two-track accuracy/latency tradeoff mirroring their Pareto framing. Their tri-pool structure (raw observations / events / strategy notes) resembles the evidence / episodic / playbook split — validation of the schema.

### Deep-research pattern consolidation (2025)
- **Sources:** https://www.anthropic.com/engineering/multi-agent-research-system (Jun 2025); https://openai.com/index/introducing-deep-research/ (Feb 2025); survey https://arxiv.org/html/2506.18096v2 (Sep 2025); https://www.langchain.com/blog/open-deep-research (Jul 2025)
- **Mechanism/status:** The field converged on: planner decomposes → parallel subagents each run iterative search→read→refine loops → each returns a compact synthesis → lead agent integrates. OpenAI's deep research is end-to-end trained to decide when to search/read; Anthropic's is orchestrated. Iterative, multi-query retrieval consistently beats single-shot on complex information tasks in every published comparison; single-shot RAG survives only for simple, low-latency lookups.
- **swarm-brain applicability:** For hackathon scope, one concrete steal: allow the recall API to accept **multiple queries in one call** (batched lanes) so an agent's refine step costs one round-trip, and return "why retrieved" lane provenance so agents can steer their next query.

## 3. Contextual retrieval & chunk enrichment

### Anthropic Contextual Retrieval (Sep 2024 — baseline, still the reference numbers)
- **Source:** https://www.anthropic.com/engineering/contextual-retrieval
- **Mechanism/results:** Prepend an LLM-generated chunk-situating context (50–100 tokens, cached-prompt-cheap) to each chunk before embedding + before BM25 indexing. Contextual embeddings + contextual BM25 cut top-20 retrieval failure rate 49% (5.7%→2.9%); with reranking, 67%.
- **swarm-brain status:** By 2026 this is the standard "cheap upgrade"; the frontier moved to model-native contextualization (below).

### voyage-context-3: contextualized chunk embeddings beat both contextual retrieval and late chunking (Jul 2025)
- **Source:** https://blog.voyageai.com/2025/07/23/voyage-context-3/
- **Mechanism:** Model embeds the whole document in one pass and emits one vector per chunk that fuses local detail with document-global context — no manual augmentation, drop-in (same vector count/dims as standard embeddings, unlike ColBERT).
- **Results (NDCG@10, 93 datasets):** beats OpenAI-v3-large by 14.24% (chunk-level) / 7.89% (doc-level); Cohere-v4 by 12.56%/5.64%; **Jina-v3 late chunking by 23.66%/20.54%; Anthropic-style contextual retrieval by 6.76%/2.40%**. Chunking-strategy sensitivity halved (2.06% vs 4.34% variance); binary 512-dim beats OpenAI float-3072 by 0.73% at **0.5% of the vector storage cost**.
- **swarm-brain applicability:** For the unbuilt **evidence/source-chunk lane**: if the embedding provider is a free choice, contextualized chunk embeddings make DIY contextual retrieval obsolete. If locked to a generic embedder, do Anthropic-style prepending at ingest: each evidence chunk gets a header of {source doc/file, memory title, section path, timestamp} before embedding and before FTS indexing — cheap and fits the citation-of-exact-spans model perfectly. The binary-quantization result also pairs with CRDB's RaBitQ story (§5): small vectors are no longer an accuracy sacrifice.

### Late chunking vs contextual retrieval head-to-head (ECIR 2025 workshop)
- **Source:** "Reconstructing Context," https://arxiv.org/abs/2504.19754 (Apr 2025); late chunking origin: https://openreview.net/pdf?id=74QmBTV0Zf (Jina)
- **Result:** Contextual retrieval preserves semantic coherence better but costs more compute (LLM call per chunk); late chunking is cheaper but sacrifices relevance and completeness. Confirms the ranking voyage measured independently.
- **Applicability:** If implementing chunk enrichment on CRDB, choose contextual prepending over late chunking — the corpora (handoffs, playbooks, code notes) are small enough that the per-chunk LLM cost is trivial, and quality wins.

### RAPTOR / hierarchical summaries: 2025–2026 status
- **Sources:** original https://arxiv.org/abs/2401.18059 (ICLR 2024); enhancement: semantic chunking + adaptive clustering, Frontiers in CS 2025 https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1710121/full; productized in RAGFlow https://ragflow.io/docs/enable_raptor
- **Status:** No longer the research frontier (contextualized embeddings + agentic exploration absorbed the attention), but alive as a product feature for corpus-level questions. Incremental tree maintenance remains its weak point.
- **swarm-brain applicability:** Low priority. The typed-link graph + knowledge sections already provide the hierarchy RAPTOR approximates. The one RAPTOR idea worth keeping: periodically synthesized "summary memories" over clusters of episodic memories (e.g., per-component playbook regeneration) retrieved in the same lanes as leaves.

### PropMem: enrichment lessons from a measured memory system (Mar 2026)
- **Source:** https://medium.com/prosus-ai-tech-blog/memeval-benchmarking-memory-for-ai-agents-932d3fd9f3b4 (code: https://github.com/ProsusAI/MemEval)
- **Mechanisms with measured impact:** (1) **atomic propositions** (~25 words, one fact per entry, relative dates resolved to absolute at extraction — "last week" → "week of May 8, 2023") instead of 400-token chunks: top-30 slots hold 30 distinct facts, not 3 mixed chunks; (2) **entity-filtered retrieval** (identify the question's entity, scope all search to it) — *their single largest accuracy improvement*; (3) question-type routing (strict abstain-if-unsupported prompt for factual, reason-freely for inferential) via lightweight classifier, not an LLM call; (4) dedup by normalized text; (5) **soft recency**: when two propositions about the same entity have cosine >0.85 but different dates, older one gets a 30% score penalty (kept, not deleted). They explicitly rejected knowledge graphs and multi-stage pipelines as not paying off at this scope; total 65% fewer tokens than their chunk-RAG baseline.
- **swarm-brain applicability:** This is the closest published analog to the **temporal/entity routing lane**: entity extraction at query time + scoped search is cheap and was their biggest win — implementable as a WHERE clause over the JSON entity fields before RRF. The 30%-penalty-not-deletion recency rule is a ready-made policy for bitemporal supersession scoring, and date-resolution-at-ingest is essential for the bitemporal validity fields.

## 4. RAG / memory evaluation SOTA

### TREC RAG 2026: the "agent-first" track, current best-practice harness
- **Source:** https://trec-rag.github.io/ (guidelines/tools: https://github.com/TREC-RAG/trec-rag-skills, https://github.com/castorini/RAGDoll)
- **Mechanism:** 2026 track is agent-first; corpus switched from MS MARCO v2.1 to NVIDIA ClimbMix-400b; two tasks (Retrieval; RAG with grounded summarized answers). Evaluation stack: **UMBRELA** LLM-generated qrels, **nugget-based** answer scoring (RAG25 nuggets), ResearchRubrics, and **RAGDoll**, an end-to-end open toolkit from gold-standard construction to long-form answer scoring, with priority-ranked manual assessment layered on top of automatic judging.
- **swarm-brain applicability:** RAGDoll's pattern — automatic LLM qrels + nugget scoring, with a small human-verified priority slice — is the current best-practice template for the eval harness, and UMBRELA-style LLM qrels are the accepted way to make a scaled gold set credible.

### LongMemEval (ICLR 2025) and LongMemEval-V2 (2026) — including abstention
- **Sources:** https://xiaowu0162.github.io/long-mem-eval/, https://github.com/xiaowu0162/longmemeval, https://arxiv.org/abs/2605.12493
- **Mechanism:** V1 (already used as LongMemEval-S) tests five abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, **abstention** — a dedicated `_abs` subset (~30 questions) about events that never happened, graded on correctly declining. V2 shifts to *agent environment experience*: 451 questions, five abilities (static state recall, dynamic state tracking, workflow knowledge, **environment gotchas**, premise awareness), histories to 115M tokens, "context gathering" formulation (memory system returns compact evidence, downstream QA scores it); abstention questions handled via manually verified anchor mapping.
- **swarm-brain applicability:** V2's ability taxonomy is a better template for gold-corpus question categories than generic QA — especially "environment gotchas" and "workflow knowledge," which map to playbooks and prior-attempt memories. Its context-gathering formulation (score the evidence bundle, not the final answer) matches the task-bootstrap product exactly and is cheaper to judge.

### MemoryAgentBench, MemEval, BEAM — the 2025–2026 agent-memory benchmark landscape
- **Sources:** MemoryAgentBench https://arxiv.org/abs/2507.05257 (2025; four competencies: accurate retrieval, test-time learning, long-range understanding, conflict resolution; https://github.com/HUST-AI-HYZ/MemoryAgentBench); MemEval https://github.com/ProsusAI/MemEval (Mar 2026); BEAM (2026, 1M-token long-term memory; see https://www.linkedin.com/pulse/what-beam-memory-benchmark-paper-shows-1m-context-window-isnt-enough-rj9qf)
- **Key finding (MemEval):** on LoCoMo, published "judge accuracy" ranges **58%–92% for the same benchmark** purely from differing LLMs/embedders/scoring — vendor numbers are incomparable; token cost across 9 memory systems varies **12×**. Their fix: freeze LLM, embedder, and scoring across all systems, and report **quality-per-token**.
- **swarm-brain applicability:** Report quality-per-token (bundle tokens per point of Recall@k/nDCG) as a headline metric — it is now the credible way to present memory-system numbers, and it differentiates the bounded-graph/packing design. MemoryAgentBench's "conflict resolution" competency maps to trust states + bitemporal supersession — add conflicting-memory questions to the gold corpus.

### LLM-judge reliability (2026 findings)
- **Sources:** survey: Gu et al., "A survey on LLM-as-a-judge," https://www.sciencedirect.com/science/article/pii/S2666675825004564 (2026); large-scale systematic eval: https://arxiv.org/html/2606.19544v1 (Jun 2026); NeurIPS 2025 rating-indeterminacy: https://neurips.cc/virtual/2025/poster/117308; practitioner synthesis: https://www.adaline.ai/blog/llm-as-a-judge-reliability-bias (Apr 2026)
- **Findings:** Strong judges reach ~80%+ raw agreement with humans on well-structured tasks, but 2026 work shows (a) **kappa deflation** — chance-corrected agreement (Cohen's κ) is much lower than raw exact-match agreement, consistently across the frontier cohort; (b) frontier models fail >50% of systematic bias probes (position, verbosity, self-preference); (c) forced-choice rating indeterminacy alone can heavily bias judge validation. Best practice: binary rubric-anchored questions rather than Likert scales, randomized ordering, calibration against a human-labeled subset, report κ.
- **swarm-brain applicability:** For no-answer/abstention grading and any LLM-judged answer scoring in the harness: use binary rubric checks ("does the bundle contain a span stating X? yes/no") which are the judge-reliable regime, keep a human-audited calibration slice (~10%), and never compare judge scores produced under different judge configs.

### Scaling the 90-memory gold corpus: SOTA synthetic generation with verification
- **Sources:** multi-agent diverse synthetic QA for RAG eval: https://arxiv.org/html/2508.18929v1 (Aug 2025); Red Hat SDG Hub pipeline: https://developers.redhat.com/articles/2026/02/23/synthetic-data-rag-evaluation-why-your-rag-system-needs-better-testing (Feb 2026); deepeval Golden Synthesizer: https://deepeval.com/docs/golden-synthesizer; RAGAS testset generation: https://thedataguy.pro/writing/2025/04/generating-test-data-with-ragas/; RAGAS remains the default framework (https://superlinked.com/blog/evaluating-retrieval-augmented-generation-ragas, Apr 2026)
- **Consensus recipe (2026):** (1) generate questions *from* the corpus documents with evolution/persona diversification (multi-hop, paraphrase, adversarial-distractor variants); (2) attach provenance — each synthetic question stores the source span that answers it (automatic gold labels); (3) **verify**: LLM-judge filters for answerability/faithfulness + a stratified human audit; (4) deliberately include unanswerable questions (LongMemEval `_abs` pattern — questions about plausible events that never occurred) for the no-answer metric; and (5) for abstention scoring specifically see "Do Retrieval Augmented LMs Know When They Don't Know?" https://arxiv.org/html/2509.01476v3 (2025).
- **The standout method — trajectory-derived gold (SWE-Explore + Cursor pattern):** derive ground truth from *real successful agent runs*: for each completed swarm task, an LLM (or the trajectory itself) identifies which memories were actually consulted/decisive → those become gold labels for the query "task bootstrap for task T." This produced line-level gold for 848 instances in SWE-Explore and trains Cursor's retriever. The planned persistent trace sink makes this nearly free and scales the corpus with usage rather than authoring effort.

## 5. DB-native retrieval advances (new since mid-2025 only)

### CockroachDB C-SPANN: 25.2 preview → 25.3/25.4 maturation
- **Sources:** design deep-dive https://www.cockroachlabs.com/blog/cspann-real-time-indexing-billions-vectors/ (Jun 23, 2025); v25.3 notes https://www.cockroachlabs.com/docs/releases/v25.3; v25.4 notes https://www.cockroachlabs.com/docs/releases/v25.4; distance-metric issue https://github.com/cockroachdb/cockroach/issues/144016
- **Mechanism:** C-SPANN = SPANN/SPFresh + ScaNN ideas on a hierarchical K-means tree; partitions stored as contiguous KV rows (shard/split/merge like normal data); **RaBitQ 1-bit-per-dimension quantization** cuts ~3KB OpenAI vectors to ~200 bytes (~94%), with exact-vector reranking + over-fetch guided by RaBitQ error bounds; fanout ~100 keeps trees ≤5 levels at 10B vectors; **prefix columns** give a separate K-means tree per prefix value (per-tenant/per-user indexes), composable with REGIONAL BY ROW.
- **What's new since mid-2025:** 25.2 (Jun 2025) shipped only Euclidean, no merges/reassignments, offline backfill. **v25.3 added cosine distance and inner product for vector indexes; v25.4 added online table backfills** (create a vector index on populated tables without taking them offline), plus continued work on background merge/reassignment and broader WHERE-filter support.
- **swarm-brain applicability:** (a) If the dense lane predates 25.3, verify cosine-capable vector indexes rather than brute-force `<=>` scans; (b) use **prefix columns** (project_id or agent/task scope, embedding) so recall bundles search per-project trees — this is exactly PropMem's entity-scoping win expressed in DDL; (c) 25.4 online backfill matters if the evidence-chunk lane is added to already-populated tables.

### BM25-in-SQL: VectorChord-BM25, ParadeDB pg_search, and Timescale pg_textsearch
- **Sources:** VectorChord-BM25 (Block-WeakAnd BM25 in Postgres, claims 3× Elasticsearch on ranking workloads): https://blog.vectorchord.ai/vectorchord-bm25-revolutionize-postgresql-search-with-bm25-ranking-3x-faster-than-elasticsearch (Feb 2025, maintained through 2026: https://github.com/tensorchord/VectorChord-bm25); ParadeDB pg_search 0.18.x (Sep 2025, https://pgxn.org/dist/pg_search/0.18.2/) + "Hybrid Search in PostgreSQL: The Missing Manual" https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual (Oct 22, 2025); **new entrant**: Timescale's pg_textsearch bringing true BM25 to Postgres https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres (Oct 23, 2025)
- **Trend:** By late 2025 there are three competing native-BM25-in-Postgres implementations, all built for hybrid (BM25 + vector + RRF) inside SQL; Block-WeakAnd (block-max WAND) top-k pruning is the shared engine idea.
- **swarm-brain applicability:** None run on CockroachDB, but the pattern to steal: CRDB FTS/trigram gives boolean matching + `ts_rank`-style scores, not BM25. BM25 scoring is implementable in plain SQL over a term-frequency side table (per-memory term counts + corpus DF) for small corpus sizes — the hybrid-search "missing manual" post documents exactly the RRF wiring; correctness matters more than WAND-style speed at this scale.

### In-DB quantization: RaBitQ/binary is now the mainstream default
- **Sources:** CRDB C-SPANN RaBitQ (above); VectorChord IVF+RaBitQ benchmark: https://seanpedersen.github.io/posts/vector-databases (Oct 2025); pgvector expression-index binary quantization (`binary_quantize(...)::bit` + `bit_hamming_ops`): https://github.com/pgvector/pgvector; pgvectorscale: 471 QPS @ 99% recall on 50M 1536-dim vectors (https://dev.to/polliog/postgresql-as-a-vector-database-when-to-use-pgvector-vs-pinecone-vs-weaviate-4kfi, Mar 2026)
- **Trend:** Binary/1-bit quantize-then-rerank became the default architecture across CRDB, VectorChord, and pgvector ecosystems in 2025–2026; combined with voyage-context-3's binary-at-512-dims result (§3), storing full-precision vectors for search is now an anti-pattern.
- **swarm-brain applicability:** In CRDB this comes free via C-SPANN — but it strengthens the case for *more* embedded objects (per-evidence-chunk vectors for the evidence lane) since marginal index cost is ~200B/vector.

### Hybrid search + RRF + MMR as pure-SQL patterns (2025–2026 consolidation)
- **Sources:** RRF in SQL: SingleStore https://www.singlestore.com/blog/hybrid-search-using-reciprocal-rank-fusion-in-sql/ (Aug 2025); MariaDB shipped RRF hybrid-search docs https://mariadb.com/docs/server/reference/sql-structure/vectors/optimizing-hybrid-search-query-with-reciprocal-rank-fusion-rrf (Jul 2026); **RRF and MMR implemented in PL/pgSQL**: https://medium.com/open-source-journal/rrf-and-mmr-in-postgres-what-they-mean-and-how-to-implement-them-in-pl-pgsql-63d9bf2dc313 (Apr 2026); pgvector+FTS+RRF walkthrough https://dev.to/lpossamai/building-hybrid-search-for-rag-combining-pgvector-and-full-text-search-with-reciprocal-rank-fusion-6nk (Feb 2026)
- **Trend:** Weighted RRF over parallel lanes (the current design) is now the textbook pattern — databases are shipping it natively. The 2026 novelty is pushing **MMR/diversity re-ranking into SQL** as a post-RRF stage (iterative greedy selection via recursive CTE or procedural loop).
- **swarm-brain applicability:** Validates the architecture as SOTA-aligned; the MMR-in-SQL articles show the packing lane's diversity pass can live in the database (single round-trip: lanes → RRF → greedy diverse top-k under token budget using a stored token_count column), which fits the "deterministic bootstrap bundle" latency goal.

---

## Highest-leverage takeaways for swarm-brain's unbuilt lanes

1. **Context packing:** implement greedy budgeted submodular selection (not plain MMR) and add **answer-in-context** to the harness (arXiv:2607.00725) — the packed bundle, not the retrieved list, is what predicts agent success (4.6× EM separation).
2. **Iterative affordances:** expose search/read/expand tools beyond the one-shot recall call — LongMemEval-V2 measured 72.5% vs 48.5% for agentic evidence-gathering over RAG, and SWE-Explore puts agentic explorers a full tier above one-shot retrieval. Keep the deterministic bootstrap for latency; make iteration available for depth.
3. **Evidence lane:** contextual prepending (memory title + source + section + timestamp headers on each chunk before embedding/FTS) is the proven cheap upgrade (−49% failure rate); contextualized chunk embeddings (voyage-context-3) are the ceiling if the embedder is swappable.
4. **Temporal/entity routing:** PropMem's entity-scoped retrieval was their largest single accuracy win and maps to a WHERE clause + CRDB prefix columns; adopt absolute-date resolution at ingest and 30%-penalty (not deletion) for superseded near-duplicate facts.
5. **Scaling the eval corpus:** trajectory-derived gold (SWE-Explore/Cursor pattern) via the planned trace sink, plus LLM-generated qrels with a human-audited slice (TREC RAG 2026 UMBRELA/RAGDoll pattern), `_abs`-style unanswerable questions for the no-answer metric, binary rubric LLM-judge with κ reporting, and **quality-per-token** as a headline metric (MemEval showed cross-setup judge numbers vary 58–92% and token costs 12×).
6. **DB check:** confirm CRDB ≥25.3 for cosine vector indexes (25.4 for online backfill); C-SPANN's RaBitQ makes per-chunk vectors cheap (~200B each).
