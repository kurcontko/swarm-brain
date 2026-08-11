# SOTA acceptance gates

“State of the art” is an empirical claim about a fixed protocol, not a synonym
for an architecture that resembles recent papers. Swarm Brain therefore keeps
its claim criteria in the machine-readable
[`benchmarks/sota/manifest.json`](../benchmarks/sota/manifest.json). Targets are
frozen with a date and primary source; changing one requires a cited frontier
update, not a disappointing local run.

Run the gate from the repository root:

```bash
uv run --extra dev python scripts/evaluate_sota_readiness.py
```

The command exits zero only when every required evidence artifact exists and
passes every check. Missing artifacts fail closed. `--json` emits the complete
report for CI or a release bundle. The manifest also requires a current-tree,
offline compiler replay: allowlisted report compilers rebuild into a temporary
path with credentials stripped, and the bytes must match the submitted report
exactly. Gates without a safe offline compiler remain failed even if someone
drops a plausible-looking JSON file into the expected path.

Manifest schema v2 also SHA-256-binds the exact wording of the broad claim to
explicit evidence dimensions. Every gate declares the dimensions it supports,
every declared dimension must be covered by at least one **required** gate, and
the evaluator rejects changed claim text, undeclared dimensions, or
informational-only coverage before considering any score. This prevents a
collection of individually valid benchmark artifacts from being promoted into
a broader claim than they test.
In particular, “in agent swarms” now requires the budget-matched 1/2/4-agent
causal gate; MemoryArena's interdependent environment protocol cannot silently
stand in for measured multi-agent scaling.

After each complete hypothesis generation, run the receipt-preserving
compatibility runner. It uses the upstream evaluator's exact prompt function,
`gpt-4o-2024-08-06`, temperature 0, max-tokens 10, and loose `"yes"` label
rule, while retaining the exact credential-free request body, endpoint,
prompt/response bytes, and provider usage that the upstream script discards.
Use the dedicated judge credential variable so a reader-provider key cannot be
forwarded to OpenAI accidentally:

```bash
LONGMEMEVAL_OFFICIAL_JUDGE_API_KEY='...' \
uv run --extra dev python scripts/run_longmemeval_official_judge.py \
  --hypotheses run-1-hypotheses.jsonl \
  --lme-path longmemeval_s_cleaned.json \
  --out-labels run-1-official-labels.jsonl \
  --out-receipts run-1-official-judge-receipts.jsonl
```

Once three label/receipt pairs exist, build the canonical evidence artifact:

```bash
uv run --extra dev python scripts/build_longmemeval_official_report.py \
  --evidence run-1.json run-1-official-labels.jsonl run-1-official-judge-receipts.jsonl \
  --evidence run-2.json run-2-official-labels.jsonl run-2-official-judge-receipts.jsonl \
  --evidence run-3.json run-3-official-labels.jsonl run-3-official-judge-receipts.jsonl
```

The builder rejects partial question sets, mixed readers/prompts/implementations,
nonofficial judge model IDs, duplicate IDs, reused reader request receipts, and
labels whose hypothesis differs from the preserved generation run. It
recomputes path/bytes/SHA bindings for every schema-v4 generation run,
hypothesis file, content-bearing chat-receipt sidecar, raw official-label file,
shared retrieval source, and the exact cleaned dataset artifact. For each
reader call it reconstructs the prompt from the bound dataset plus saved final
retrieval ranking, then strictly replays the retained prompt and response
bytes, response model, provider request ID, hypothesis, finish reason, and
reconciled usage, exact model/temperature/max-token/thinking controls, and
endpoint before accepting the declared immutable revision. It also
checks every delivered context against the bound retrieval ranking and reports
a deterministic question-cluster bootstrap interval across repeated runs.

The compiler likewise reparses every official-judge receipt, requires the
dated GPT-4o response model and a unique provider request ID, reconstructs the
official prompt from the bound dataset and hypothesis, reconciles raw usage,
and derives the Boolean label again from the retained response. A saved
`autoeval_label.label` cannot pass if it disagrees with those bytes, and a
valid response attached to a different prompt cannot pass either. Endpoint
origin and served weights remain externally attested rather than
cryptographically authenticated; the artifact states that trust boundary.

The compiled report explicitly marks the frozen Exabase 0.964 number as
**protocol-incomparable**: that result used a Mem0 fork, Gemini 3 Flash as both
reader and judge, and top-50 retrieval, while this gate uses the official GPT-4o
judge, a separately disclosed reader, and the bound retrieval limit. The
manifest now requires that comparison to be protocol-comparable, so the gate
fails by construction until we either reproduce the Exabase protocol exactly
or replace the threshold with a frozen fixed-reader, fixed-judge frontier.

The 2026 paper frontier does not repair that mismatch by changing only the
number. Mnemis (ACL 2026) reports 0.916 with GPT-4.1-mini as backend and grader;
LeanMem v1 reports 0.918 with a GPT-4.1-mini judge; SmartSearch reports 0.884
with a GPT-4.1-mini reader and GPT-4o-mini judge. None is directly comparable
to this official-judge artifact. The next QA manifest revision must therefore
define a separate exact paper-protocol track or require a baseline rerun under
the candidate's identical reader, prompt, dataset, token budget, and judge. A
paired lower confidence bound above zero is the preferred claim test. Until
that revision and its evidence exist, the impossible Exabase check remains a
visible migration sentinel rather than being weakened into an apples-to-oranges
pass.

## What must be proven

1. **Retrieval and delivered context.** Full-500 LongMemEval-S retrieval and
   answer-in-context under a fixed budget are component gates. The harness has
   produced diagnostics, but the saved semantic artifact predates the current
   cleaned dataset digest and does not pass this gate. Publishable context
   metrics must count the complete chronologically serialized official reader
   prompt through a model/revision-pinned local tokenizer. The tokenizer
   executable and artifact are repository-local byte/SHA-bound evidence, and
   every greedy packing decision carries a provider-observed request identity,
   prompt digest, and exact count. The chars/4 item estimator remains available
   for development but fails the gate. Even a fresh passing component run
   cannot substitute for QA.
2. **Comparable end-to-end QA.** All 500 questions from the current official
   cleaned LongMemEval-S release, its pinned dataset digest, an exact reader
   and judge revision, prompt digests, no missing labels, and repeated runs are
   required. A paper-frontier claim must reproduce the complete paper protocol
   (currently Mnemis 0.916 as the reviewed bar, LeanMem 0.918 as the newer
   preprint bar) or beat a paired baseline executed in the identical frozen
   harness. The canonical official-GPT-4o track remains valuable evidence but
   cannot borrow a GPT-4.1-mini or Gemini scalar. The 0.964 Exabase number is a
   non-comparative product stretch until its Gemini protocol is reproduced.
3. **Environment experience.** LongMemEval-V2 must exercise the disclosed
   search/read/expand control flow with the official fixed reader and judge on both public
   tiers. Each tier must beat the released accuracy point and produce positive
   LAFS gain over the official accuracy-latency frontier, while also reporting
   delivered tokens and tail latency. A sidecar Boolean is only secondary
   metadata: every evaluated query must carry a content-free ordered operation
   trace proving a successful `recall_memory` followed by a successful
   `read_expand_memory` seeded from that search. The evidence compiler checks
   trace order, opaque IDs, bounds, exact delivered-token reconciliation, and
   operation latency against the official per-query record. It also requires
   the trace's canonical SHA-256 digest in the harness-preserved
   `memory_post_query_metadata`, binding every external sidecar trace to its
   package-hashed `per_question.jsonl` row.
   The current bridge is exactly one recall followed by one bounded expansion;
   it records `fixed_two_stage_recall_then_expand` and does not claim an
   adaptive multi-round controller.
4. **Interdependent multi-session action and swarm-causal gain.** These are two
   separate required claim dimensions. The official MemoryArena SR/PS protocol
   covers interdependent environment action. It pins the preview repository and
   HTTP memory seam, but remains
   fail-closed while upstream lacks an immutable 766-task snapshot, reconciled
   task counts, all five paper configuration overlays, and an official full-run
   SR/PS compiler. The paper says 766 task groups, its per-domain table sums to
   736, and the current public dataset is a moving snapshot; those identities
   may not be silently substituted. The semantic bridge additionally records
   its configured embedding revision as `operator-declared-unverified`; exact
   response-model identity, dense-lane coverage, and provider call accounting
   are useful diagnostics, but cannot become publishable evidence until a
   provider attestation or immutable deployment manifest binds the served
   weights to that revision. Full configuration fields and task-group counts
   must then be replayed from the pinned artifacts rather than trusted from a
   sidecar manifest.

   The paired 1/2/4-agent memory-vs-no-memory difference-in-differences harness
   separately covers the claim's “agent swarms” scope. It must show a strictly
   positive lower confidence bound for four agents versus one under equal total
   model and tool budgets, with four versus two non-negative. It is explicitly
   not MemoryArena and cannot satisfy that paper gate; conversely, MemoryArena
   cannot satisfy the causal swarm-scaling dimension.
5. **Governance and forgetting.** The complete pinned GateMem v1 protocol must
   cover all 91 episodes and 2,218 checkpoints with the official external
   scorer, the paper's `gpt-4o-mini` answer model and `gpt-4o` judge, a complete
   action-gated judge, provider token usage, and no missing, duplicate,
   unjudged, or unparsable rows. Search, direct lookup, lineage, historical
   reads, and derived memories all belong to the attack surface. There is no
   defensible single published scalar frontier, so this is a conjunctive
   Pareto gate:

   - access-control violations and active-forgetting failures must both be
     exactly zero, while aggregate utility and MGS must each be at least 70%;
   - each domain's zero-failure utility must reach the best published Table 3
     MGS across the paper's backbone sweep for that domain: Medical 80.1%,
     Office 67.9%, Education 71.0%, and Household 68.5%. Under zero A/F these
     utility floors are also the domain MGS values. The 70% aggregate floor
     clears the 69.6736% all-domain MGS obtained by weighting the rounded
     Deepseek-V4-Pro Long-Context Table 3 U/A/F rows by their category
     checkpoint counts and recomputing MGS. This quality target is deliberately
     a cross-backbone frontier envelope that the fixed GPT-4o-mini run must
     clear, not a same-backbone reproduction;
   - aggregate over-refusal, and Medical over-refusal specifically, must be at
     most 24.8%. Figure 3(b) publishes 24.8% only for Long-Context on Medical
     with GPT-4o-mini, making the Medical check an exact-model comparison.
     Applying that value to the all-domain aggregate is an intentionally
     conservative anti-paralysis cap, not a claim that GateMem published an
     all-domain over-refusal frontier;
   - provider-reported answer-call input plus output tokens must be at most
     1,050/1,240/1,380/1,180 per checkpoint for Medical/Office/Education/
     Household and at most 1,210 overall. These are the GPT-4o-mini Table 4
     ReMem-S values and their checkpoint-weighted aggregate (1,209.55 from
     rounded domain rows), so all token checks are exact-model comparisons.
     This matches the pinned official scorer's `output.llm_usage` accounting.
     It excludes judge, embedding, and memory-ingestion calls; ingestion is
     included only in GateMem's separate wall-clock metric.

   The compiler pins the GateMem repository, dataset, scorer, leaderboard,
   result image, and paper-matrix digests and reconciles every answer-token row
   with the official summary. Its leakage rates are stricter than the paper's
   primary Table 3 labels: they union prompt-context/final-output rule leakage
   with judge leakage and apply action gating. Report that deviation instead of
   presenting the numbers as an identical rerun of the paper protocol.
6. **Memory to action.** The complete 400-task Mem2ActBench release must use a
   pinned Qwen2.5-72B reader and paired no-memory, Swarm, and oracle arms. Its
   paper-comparable target-tool-given condition must exceed the published
   35.93% parameter-F1 frontier, while a separate full-catalog condition
   must show a positive paired 95% lower confidence bound for exact tool plus
   arguments versus no memory. Merely emitting the full-catalog metric cannot
   pass. The upstream release contains no evaluator implementation, so our
   strict metric reimplementation and that protocol limitation stay explicit
   in the report. The gate now requires an upstream official evaluator (or a
   separately reviewed calibration against reproduced published baselines), so
   the local reimplementation alone cannot pass.

The manifest intentionally excludes unit-test counts and architectural
checklists. Those are release requirements, but they do not establish SOTA.
Likewise, retrieval-only Recall@10, simulated agent labels, and development
judges cannot satisfy an end-to-end gate.

## Claim discipline

- Preserve raw runs, hypotheses, judge labels, prompts, model identifiers,
  dataset checksums, seeds, token use, latency, and failures.
- Use paired items and bootstrap confidence intervals for comparisons.
- Report protocol deviations next to every number.
- Keep hard scope, trust, evidence, and version constraints outside any learned
  policy. Learning may choose when, what, and how much to activate; it may not
  relax governance.
- If a newer primary source moves the frontier, update the manifest source,
  threshold, and `frozen_at` together in one reviewable change.
