# Paper-driven memory runtime

The research review now has a concrete runtime slice. This note separates what
is shipped from what the cited work still suggests building; the detailed
evidence and links remain in the
[agent-memory research dump](research/sota-agent-memory-retrieval-2026-08-07.md).

## What this branch ships

| Research signal | Framework change | Runtime invariant |
| --- | --- | --- |
| Mastra Observational Memory and Mem0: distilled, typed observations are more useful than replaying raw history | Optional OpenAI-compatible typed extraction proposes bounded candidates, dates, aliases, metadata, and safe local relations; deterministic extraction remains the fallback | Provider output cannot choose identity, scope, trust, visibility, lifecycle, or invented source offsets |
| MemClaw: shared memory needs scoped, provenance-preserving, policy-governed propagation | Automatic activation retrieves confirmed memories only and reuses the existing scope, trust, temporal, and supersession filters | A task claim cannot inject tentative, stale, refuted, untrusted, or out-of-scope memory |
| MemOS and finite-context work: retrieval and activation are different lifecycle decisions | Task claim, checkpoint resume, dependency unblock, tool error, repeated failure, and explicit triggers are wired through one activation service | Query text and memory content never enter activation telemetry; deterministic IDs make one trigger per lease idempotent |
| LongMemEval-V2 and deep-research systems: agents outperform one-shot RAG by iteratively searching, reading, and refining | `recall_memory` is the repeatable search step; `read_expand_memory` reads exact selected IDs and follows at most two bounded link hops | The owned lease is checked first; seeds and neighbors are rehydrated under one scope/lifecycle/time/trust snapshot and total rendered content/evidence is token-capped |
| Observational Memory and temporal graph memory: a query's referenced period is different from both observation time and system as-of time | Public recall, explicit activation, and read-expand accept a complete `referenced_valid_from` / `referenced_valid_to` pair and route by half-open overlap; an opt-in pure parser plus temporal candidate lane is available for held-out evaluation | The pair is validated and mutually exclusive with point selector `world_at`; `recorded_at` remains orthogonal, exact-ID hydration reapplies the interval, and serving never infers a date from free text unless a caller explicitly opts into the parser result |
| LongMemEval temporal-routing failure analysis: retrospective event time must not be confused with observation or validity time | Memories can preserve evidence-backed `occurred_at`; recall can opt into a complete occurrence-prior interval that adds a temporal distance lane | Extraction keeps event time out of `valid_from`; the prior is never a hard filter, unknown occurrence is neutral, and default ranking is unchanged |
| Adaptive retrieval and graph-memory work: lane cost and graph depth should depend on query need and calibrated evidence | A versioned pure policy emits explicit lane weights, graph allow/skip, maximum depth, monotone per-hop thresholds and conservative abstention from rank-independent direct evidence | Missing evidence fails closed; graph evidence cannot rescue an unsupported answer; coefficients are human-reviewed and the policy is not wired into default serving before held-out calibration supplies an honest graph prior |
| Mastra Observational Memory, Mem0, and Memory-R1: useful memory requires a bounded consolidation decision rather than unconditional append | A post-write observer queues immutable snapshots for an asynchronous reflector that can propose append, supersede, link, or no-op | The provider sees opaque keys, while local code derives scope, lifecycle, confidence, exact evidence union, and `DERIVED_FROM` lineage; stale snapshots become no-ops and retries reuse one staged plan |
| Anthropic context engineering and context-rot findings: small relevant context beats indiscriminate history | Canonical memory blocks are greedily packed to a 2,048-token default, with a final result cap and `PackingTrace` | The exact `activation_context` sent over HTTP/MCP is the representation measured against the budget; the raw recall bundle is excluded |
| Answer-in-context and trace-derived evaluation: retrieval is not use | Metrics separately count activation decisions, selected memories, lease-scoped citations, and citations of another agent's memory that match activation for the same task/lease/consumer | Ordinary recall and self-reported citations cannot inflate proven cross-agent use |
| ACL 2026 experience-following error propagation: observed outcomes can inform learning, but naive reuse can amplify bad experience | Completion atomically records a content-free `observational_silver` association only when a citation matches exact same-task/lease/consumer activation and current memory version proof | Failed outcomes never refute, supersede, consolidate, or down-rank memory; the signal is offline-only until a causal evaluation justifies use |
| Multi-agent workflow-memory research: the test must be causal | The demo uses exactly four agents and two task waves; fresh opaque facts exist only in Wave-A memories, and the same hidden verifier fails without context and passes with the delivered Wave-B context | Success requires activation, citation, dependency release, and a fenced cross-provider checkpoint handoff |
| MemoryArena: memory quality must be tested inside interdependent multi-session agent-environment loops | A pinned compatibility server maps the official initialize/add/wrap-prompt seam onto isolated canonical runtimes, durable embedding work, recall, and packing; a strict semantic diagnostic verifies provider accounting and dense-lane coverage | The bridge is not a score, and the semantic diagnostic remains nonpublishable because its model revision is operator-declared rather than provider-attested or deployment-manifest-bound; immutable dataset/config identity, recomputed task counts, the paper's 766-vs-736 inconsistency, and an official SR/PS compiler must also resolve |
| GateMem: shared memory needs utility, access control, and active forgetting in one evaluation | The official action-gated external scorer is bridged to scoped recall, deletion, lineage, and resume-aware completion evidence | The report labels its quality/over-refusal/token target as a cross-system composite envelope rather than a same-system reproduction |
| Mem2ActBench: retrieved memory must change correct tool choice and argument grounding, not merely answer questions | The full-catalog and target-tool-given arms use a canonical runtime bridge with pinned semantic embedding evidence and paired no-memory/oracle controls | The released repository has no official evaluator; the local metric reimplementation cannot satisfy the SOTA gate until calibrated or replaced by an upstream scorer |
| Mnemis (ACL 2026): local semantic retrieval and deliberate global traversal recover complementary evidence | Swarm Brain already has hybrid RRF, a canonical graph lane, bounded expansion, exact rehydration, and an opt-in score-only learned-reranker boundary, but only over a flat relation graph | No Qwen3 run or hierarchy is evidence-backed yet; Mnemis's own System-1 Qwen3 ablation was neutral, while its gain came from hierarchical global selection. Its 91.6 LongMemEval-S score uses a GPT-4.1-mini grader and cannot be compared to our official-GPT-4o track |
| [MAGMA](https://aclanthology.org/2026.acl-long.1709/) (ACL 2026): semantic, temporal, causal, and entity relations are clearer as query-selectable graph views | The canonical relation graph, provenance-backed event time, typed link labels, bounded traversal, and an explicit adaptive-policy contract cover most safety primitives | We do not materialize four orthogonal graph projections or calibrate policy-guided selection among them; adding all lanes at once would make attribution impossible |
| [Fine-Mem](https://aclanthology.org/2026.acl-long.900/) (ACL 2026): downstream evidence use can assign credit to the memory operations that produced it | Exact-version activation/citation matching and content-free outcome associations already provide a stricter provenance substrate than answer-only rewards | The signal remains observational and offline; it must not train or promote a memory policy until a causal intervention separates useful evidence from correlated retrieval |
| LeanMem (arXiv 2608.03463): storage form and retrieval budget should follow compressibility, temporal dynamics, and fidelity need | Typed extraction, provenance-backed `occurred_at`, immutable source evidence, asynchronous consolidation, and an unshipped adaptive retrieval policy provide the safety substrate | We do not yet route writes into profile/event/source-record representations, restrict evolution to event memory, or allocate per-type query budgets; the new 91.8 score is a preprint result under a GPT-4.1-mini judge |
| SmartSearch (arXiv 2603.15599): ranking and context compilation can dominate increasingly elaborate storage | Weighted RRF, bounded candidate windows, exact full-prompt turn packing, answer-in-context measurement, a composite-model-capable reranker port, and benchmark-only E1-A/B/C/D controls make the compilation boundary explicit | CE sigmoid, CE/ColBERT fusion, and top-60 relative-threshold mechanics are implemented but have not consumed real model scores. NER/PRF expansion remains outside E1; the paper's 88.4 result uses GPT-4.1-mini as reader and GPT-4o-mini as judge, so it is an architectural signal rather than an official-protocol scalar |
| [Chain-of-Memory](https://aclanthology.org/2026.acl-long.534/) (ACL 2026): retrieved fragments should become coherent inference paths before adaptive truncation | Chronological serialization, graph-path provenance, exact-ID expansion, token-bounded packing, an evaluation-only immutable turn projection, and pure bounded E2-A..E2-E organization controls preserve enough structure to test this cleanly | Production still packs independent memory blocks; the organizer consumes externally attested similarities, and no model-backed chain run, exact-budget reader run, or path-level QA evidence has been executed |
| [LazyMem v2](https://arxiv.org/abs/2607.22690v2): retrieve raw evidence broadly, then construct compact query-conditioned memory in bounded parallel windows | Benchmark-only E7-A/B/C now replays the real E1-B top 50, reproduces the released radius-2/window-8/stride-7 geometry, binds an authoritative source/question/date/tokenizer preflight, limits constructor inputs to query/date/source messages, retains immutable UTF-8 source spans for every KEEP, and strictly reparses retained raw provider responses plus reconciled usage | No constructor or current paired QA run exists. Raw replay removes normalized-response/usage inconsistency but does not authenticate the claimed endpoint or weights; latency and priced cost remain externally attested. E7-B paraphrases remain unproven, while E7-C is byte-grounded. The paper's `.85`/213-token and `.93`/1,041-token results use its own Qwen3/DeepSeek protocol and cannot be imported into our official track. Constructed context is evaluation-only and not yet exact-full-prompt packed |
| [UnifiedMem](https://aclanthology.org/2026.acl-long.1232/) (ACL 2026): raw values plus derived summary/fact/keyword keys are a strong flat baseline, while similarity expansion can hurt | Canonical content is immutable and separately governed; benchmark-only E6 R0-R5/R-neg controls now enforce complete F0 source coverage, independently scored key families, within-family fan-out suppression, canonical raw hydration, provenance-backed one-hop expansion, and a permanently ineligible similarity negative control | Production search still collapses fields into one projection. No extractor/scorer artifact or representation QA ablation has run, and similarity-link expansion remains disabled by default |
| [Memora](https://arxiv.org/abs/2602.03315v2): index primary abstractions and many-to-many cue anchors while preserving high-fidelity values | Append-only values, typed metadata, bounded retrieval, exact hydration, staged consolidation, and the E6 R3 abstraction/cue control provide a safer test substrate | R3 is an offline contract only: no primary-abstraction/cue artifacts or REFINE/EXPAND/STOP policy have been evaluated. Memora's 87.4 LongMemEval-S result is an external GPT-4.1-mini/GPT-4o-mini protocol result, not a Swarm Brain score |
| [Memory-R1](https://aclanthology.org/2026.acl-long.583/) and [AgeMem](https://aclanthology.org/2026.acl-long.981/): memory writes, retrieval, summarization, filtering, and stopping can be learned from outcomes | The framework already exposes append/update/no-op semantics, content-free action/outcome receipts, exact activation/citation lineage, and hard local governance that can remain outside a learned policy | Memory-R1's public repository contains no implementation and uses oracle LongMemEval; AgeMem trains on HotpotQA and does not test persistent dialogue. A learned-policy result must wait for a sealed held-out task and cannot control scope, trust, evidence, retention, or physical deletion |
| BEAM / LIGHT (ICLR 2026): a memory claim should survive coherent histories from 128K through 10M tokens, where full-context shortcuts stop working | Durable event storage, asynchronous materialization, bounded activation, and exact expansion are scale-compatible primitives | The SOTA suite has no BEAM bridge or 1M/10M evidence, so LongMemEval-S alone cannot establish long-horizon scaling; dataset, reader, judge, context budget, and construction/inference cost must be pinned together |

## End-to-end memory loop

```text
raw source
  -> deterministic + optional typed-provider candidates
  -> exact local evidence validation and fenced materialization
  -> tentative/confirmed governance and bitemporal supersession
  -> trigger-scoped confirmed retrieval
  -> optional source-bound query-time construction
  -> relevance floor and token-bounded canonical packing
  -> agent checkpoint/completion citations
  -> activation/citation/cross-agent outcome metrics
  -> content-free observational/silver outcome associations
```

The critical boundary is between retrieval and activation. Retrieval may
produce a ranked candidate set; activation decides whether any of it is safe
and useful enough to occupy working context. Before that context is released,
the activation transaction revalidates every selected ID against the current
lifecycle, temporal, visibility, and evidence-trust predicates, and requires the
exact selected memory version to remain current. This version proof also catches
partial evidence revocation that leaves the memory itself recallable; a stale
rendered selection is withheld. Citation then records what the agent says it
used, while proven cross-agent use requires that citation to match the activation
event for the same task lease.

The facility-location packer remains an explicit experimental policy, not the
default. On the cleaned full-500 LongMemEval-S diagnostic it lost to greedy
packing at every measured budget (2,048, 4,096, 8,192, and 16,384 tokens).
The result is kept as a reproducible negative finding: a paper-motivated
objective is not promoted unless it improves delivered evidence under the same
budget.

Referenced-time parsing follows the same promotion rule. The parser accepts a
small audited grammar (explicit calendar periods, bounded relative counts,
weekdays and windows), requires a caller-supplied timezone and relative-time
anchor, and returns structured no-match/rejected outcomes. Only a real closed
half-open interval can be copied into recall. The hard-validity temporal lane
is selected for that interval or an explicit `world_at`, ranks canonical
memories by `valid_from` distance, and is parity-tested across in-memory and
Cockroach adapters. A separate explicit occurrence-prior pair ranks only known
`occurred_at` values without changing eligibility. LongMemEval exposes the
hard-validity behavior only behind
`--temporal-query-routing` and records a content-free parse trace per case.
The cleaned full-500 A/B rejected promotion: Recall@10 fell 0.8848 → 0.8727,
MRR@10 fell 0.7923 → 0.7788, and 16k any-gold-in-context fell 0.882 →
0.866. These figures include a deterministic one-case rerun after correcting
the bounded `since ... ago` parser semantics. The cause is representational: a conversation/session timestamp is an
observation time, while the event described by the conversation may have a
different occurrence time. Hard validity filtering therefore removed later
retrospective evidence. The result is retained as a negative finding and the
serving default remains unchanged. Schema v12 now preserves extracted event
time as evidence-backed `occurred_at` and exposes a separate opt-in soft prior.
That contract fixes the representational conflation; it is not promoted into
automatic routing, and no benchmark improvement is claimed until a held-out
A/B supports its weight and query gate.

Dependency-aware claims select `dependency_unblocked` after a blocking
prerequisite completes. During execution, an agent may request one
deterministic `tool_error`, `repeated_failure`, and `explicit` intervention per
lease. The active owned lease is checked before evaluating ephemeral query
text, and the existing commit-time version/trust proof runs before context is
released. Replays return only durable content-free telemetry.

Iterative evidence gathering remains a canonical read path, not an activation
bypass. Agents refine `recall_memory` searches, select up to eight exact IDs,
then call `read_expand_memory` with depth `0..2`, fanout `1..8`, and at most
16,384 estimated tokens. Refuted, stale, untrusted, out-of-scope, and
wrong-task seeds or neighbors disappear during canonical hydration.

## Highest-value next increments

1. Add a paper-protocol LongMemEval track that pins the Mnemis
   GPT-4.1-mini reader/grader family separately from the canonical official
   GPT-4o judge track. Require identical prompts, dataset, retrieval budgets,
   tokenizer accounting, and a paired confidence interval before comparing
   systems. Keep the 96.4 vendor number as a non-comparative stretch target.
2. The evaluation-only atomic-turn projection, deterministic E1-A/B/C/D
   boundaries, exact full-prompt packer, learned-score boundary, and bounded
   E2 organization controls are now frozen. Generate pinned external scores
   and execute the separate ablations: mxbai CrossEncoder, CrossEncoder +
   ColBERT, score-adaptive truncation, and Chain-of-Memory organization.
3. Run E7-A/B/C at the same reader/judge/tokenizer identities: chronological
   raw top 50, query-time construction, and byte-grounded query-time
   construction. Preserve the exact 8-message/stride-7 window receipts and
   compare QA, any/all-gold, delivered tokens, construction latency, and total
   cost. Do not select on the paper's different Qwen3/DeepSeek scalar.
4. Generate source-only extraction and scoring artifacts, then run the frozen
   source-preserving multi-key representation cells in the
   [representation and policy audit](research/sota-memory-representation-policy-audit-2026-08-09.md):
   raw plus derived summary/fact/keyword keys, abstraction/cue keys, direct
   descriptive-entity activation, and only then one-hop expansion with an
   explicit reranker. Retain similarity expansion as a negative control.
   Retain pinned Qwen3 as the Mnemis reproduction control, not an assumed
   winner. Test a provenance-preserving hierarchy/global selector only after
   those controls, and MAGMA-style semantic, temporal, causal, and entity views
   one at a time after that. Do not conflate their gains.
5. Introduce an explicit, source-grounded representation policy for stable
   profiles, evolving events, and immutable detail records. Only event memory
   should be eligible for automatic state consolidation; record memories must
   retain exact source expansion. Keep the policy behind a held-out A/B until
   accuracy and total construction-plus-inference cost improve together.
6. Calibrate the new provenance-backed occurrence-time prior on held-out data,
   then rerun soft-prior versus hard-validity routing as a predeclared A/B. The
   session-time experiment above is a measured rejection, not a knob to retune
   on the same benchmark.
7. Calibrate the versioned adaptive retrieval policy on held-out traces with a
   reviewed graph prior, then integrate per-query graph gating and per-hop
   sufficiency checks. Do not derive the prior from RRF rank or reuse the graph
   lane's post-expansion score as if it existed before traversal.
8. Evaluate a learned memory-operation policy behind the existing hard
   governance boundary, following the direction of
   [AgeMem](https://aclanthology.org/2026.acl-long.981/) and
   [Memory-R1](https://aclanthology.org/2026.acl-long.583/). Use the
   exact-version activation/citation lineage as the attribution primitive for
   a Fine-Mem-style experiment. Learning may choose when and what to remember,
   but never scope, trust, evidence, or deletion enforcement.
9. Measure iterative search/read-expand trajectories on LongMemEval-V2 and
   report accuracy, LAFS, quality per delivered token, and per-hop sufficiency.
10. Run GateMem and Mem2ActBench as first-class gates so governance and
   memory-to-action claims are based on official tasks, not unit-test proxies.
11. Add BEAM at 1M and 10M tokens as the scale gate. Reuse one frozen memory
   artifact across query runs, report construction and inference cost
   separately, and require exact delivered-context tokenizer accounting so a
   larger context dump cannot masquerade as a better memory system.

Those increments should be accepted only when they improve causal task outcomes
or answer-in-context under a fixed token budget. Raw recall count is retained as
deprecated compatibility telemetry, not as evidence of memory value.
