# Retrieval benchmark — first measured baseline

Measured on 2026-08-07. This is the report that
[retrieval evaluation](retrieval-evaluation.md) requires before phase 2 may be
called benchmark-complete. Everything below is a measurement produced by
`scripts/run_retrieval_eval.py` and scored by the checked-in evaluator
`scripts/evaluate_retrieval_runs.py`. Nothing here is a state-of-the-art claim,
and none of these numbers is comparable with a published leaderboard.

Two tracks are reported:

- **Track 1 — swarm-native corpus.** A versioned coding-swarm corpus with
  hand-written relevance judgments, including explicit no-answer cases. Run on
  the in-memory kernel and on live CockroachDB, where ANN Recall@k against the
  exact-vector oracle is measured as well.
- **Track 2 — LongMemEval-S retrieval subtask.** The official public dataset,
  all 500 questions, evaluated as retrieval only using its labelled evidence
  sessions.

## Read this first

The dense lane in both tracks is driven by
`DeterministicEmbeddingProvider`, a hash bag-of-words embedder that exists so
the enqueue → lease → upsert → ANN → fuse path can run without credentials. It
is **not a semantic model**. Its cosine similarity is a length-normalised
lexical overlap, so:

- every dense number below is a floor and a plumbing proof, not a quality
  claim;
- paraphrase queries are the case dense is supposed to win and it does not;
- a rerun with a real embedding model (Bedrock/Titan `amazon.titan-embed-text-v2:0`
  through `BedrockEmbeddingProvider`) is the listed follow-up, and the fusion
  weights should not be retuned before that rerun exists.

The LongMemEval numbers measure **retrieval only**. LongMemEval's published
numbers are end-to-end QA accuracy with a reader model and an LLM judge. The
two are not comparable in either direction, and no comparison is made here.

## Environment

| Item | Value |
| --- | --- |
| Repository | `swarm-brain`, commit `3f5df4c` |
| Python | 3.13.9, arm64 |
| Host | macOS 15.7.7, Apple silicon, single machine, no isolation from other load |
| CockroachDB | `CockroachDB CCL v26.2.1 (aarch64-apple-darwin21.2)` |
| Evaluation database | `swarmbrain_eval` on `postgresql://root@127.0.0.1:26257`, schema v9 installed and verified by the runner |
| Embedding provider | `DeterministicEmbeddingProvider`, model `deterministic-eval-1024-v0`, 1024 dims, L2-normalised, cosine, no truncation |
| Input renderer | `memory_content_text` → `EmbedMemoryWorkPayload`, projection `memory-content-v1:current:cosine` |
| Recall limit | `RecallQuery.limit = 10`, `min_score = 0.0`, purpose `interactive_recall` |
| Fusion | weighted RRF, `k = 60`, planner-owned lane weights |
| Saved-run depth | 50 ranks per lane; all reported metrics use `k <= 10` |

Latency figures come from one warm machine that was also running an IDE and a
local CockroachDB. Treat them as order-of-magnitude, not as a performance gate.

## Track 1 — swarm-native corpus

### Corpus and judgments

| Item | Value |
| --- | --- |
| Corpus version | `swarm-coding-2026-08-07` |
| Judgments revision | `r1` |
| Memories | 90 (`tests/fixtures/retrieval_eval_corpus/corpus.json`) |
| Queries | 40 (`tests/fixtures/retrieval_eval_corpus/queries.json`) |
| Answerable / no-answer | 34 / 6 |
| Memory links | 53 `related_to` edges over 44 memories, so the graph lane has something to expand |

The corpus is fictional but shaped like real swarm output: invariants,
decisions, procedures, warnings, attempts, outcomes and handoffs across eight
topic clusters (payments/webhooks, CI flakes, database, build, auth,
observability, rollout flags, catalogue search). Identifiers are real-looking
and load-bearing: file paths, dotted symbols, `test_*` names, commit hashes,
`SQLSTATE 40001`, `PAYMENTS-4210`, shell commands, flag aliases. The catalogue
search cluster exists specifically as a decoy field that shares vocabulary with
the payments and database clusters.

Queries are grouped by intent: `identifier_exact` (8), `lexical` (7),
`code_lookup` (3), `fuzzy_typo` (4), `paraphrase` (5), `decoy_heavy` (4),
`multi_evidence` (3), `no_answer` (6).

Judgments were written against the corpus text before the first run and were
not revised afterwards. Two things were added *after* the first run and are
disclosed as iteration: the `min_score` sweep below, and the `--no-dense` lane
ablation. Neither changed a query, a judgment, or the corpus.

### Headline — live CockroachDB, k = 10

Recall@10 / MRR@10 / nDCG@10 over 40 cases (34 answerable):

| Lane | Recall@10 | MRR@10 | nDCG@10 | no-answer precision | no-answer recall |
| --- | --- | --- | --- | --- | --- |
| exact | 0.265 | 0.265 | 0.265 | 0.19 | 1.00 |
| lexical (FTS simple) | 0.877 | 0.810 | 0.810 | 0.00 | 0.00 |
| fuzzy (trigram) | 0.485 | 0.525 | 0.482 | 0.30 | 1.00 |
| dense (hash embedder) | 0.679 | 0.547 | 0.557 | 1.00 | 0.00 |
| graph | 0.064 | 0.093 | 0.059 | 1.00 | 0.00 |
| direct fused | 0.863 | 0.819 | 0.808 | 1.00 | 0.00 |
| final fused | **0.882** | 0.717 | 0.741 | 1.00 | 0.00 |
| final (public hits) | 0.882 | 0.717 | 0.741 | 1.00 | 0.00 |

At k = 5 the same run gives lexical 0.814 / 0.806, direct fused 0.819 / 0.819,
final fused 0.833 / 0.717.

The in-memory kernel reproduces the shape with a slightly weaker lexical lane
(its reference token-overlap scorer is not CockroachDB's `TSVECTOR`):

| Lane | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- |
| exact | 0.265 | 0.265 | 0.265 |
| lexical | 0.868 | 0.780 | 0.777 |
| fuzzy | 0.485 | 0.525 | 0.487 |
| dense | 0.679 | 0.547 | 0.557 |
| graph | 0.059 | 0.098 | 0.059 |
| direct fused | 0.853 | 0.807 | 0.793 |
| final fused | 0.858 | 0.774 | 0.762 |

### Lane ablation by query intent (CockroachDB, Recall@10 / MRR@10)

| Intent | n | exact | lexical | fuzzy | dense | graph | direct fused | final fused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identifier_exact | 8 | 1.00 / 1.00 | 1.00 / 1.00 | 0.44 / 0.50 | 0.70 / 0.52 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| code_lookup | 3 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.83 / 0.48 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| lexical | 7 | 0.14 / 0.14 | 1.00 / 1.00 | 0.79 / 0.79 | 0.86 / 0.69 | 0.00 / 0.00 | 1.00 / 0.93 | 1.00 / 0.79 |
| fuzzy_typo | 4 | 0.00 / 0.00 | 0.75 / 0.62 | 0.62 / 0.75 | 0.75 / 0.62 | 0.00 / 0.00 | 1.00 / 1.00 | 0.88 / 0.88 |
| decoy_heavy | 4 | 0.00 / 0.00 | 1.00 / 1.00 | 0.25 / 0.25 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 0.52 |
| multi_evidence | 3 | 0.00 / 0.00 | 0.78 / 0.61 | 0.33 / 0.44 | 0.33 / 0.11 | 0.22 / 0.44 | 0.78 / 0.36 | 0.67 / 0.32 |
| paraphrase | 5 | 0.00 / 0.00 | 0.50 / 0.24 | 0.00 / 0.00 | 0.20 / 0.27 | 0.30 / 0.37 | 0.20 / 0.25 | 0.50 / 0.27 |
| no_answer | 6 | — | — | — | — | — | — | — |

No-answer rows carry no relevant memory, so recall and MRR are zero by
construction; they are scored separately below.

### Where fusion helps and where it hurts

**Fusion helps on identifier drift.** On `fuzzy_typo` no single lane is good —
lexical 0.75 / 0.62, trigram 0.62 / 0.75 — while direct fused reaches
1.00 / 1.00. This is the clearest case in the corpus where the fused ranking
beats every lane that produced it.

**Fusion is neutral where one lane is already perfect.** On
`identifier_exact`, `code_lookup` and `decoy_heavy`, lexical alone already
scores 1.00 / 1.00 and direct fusion preserves it.

**The graph lane costs precision at the top.** Comparing direct fused with
final fused isolates the bounded graph expansion:

| Intent | direct fused MRR@10 | final fused MRR@10 | delta |
| --- | --- | --- | --- |
| decoy_heavy | 1.00 | 0.52 | −0.48 |
| lexical | 0.93 | 0.79 | −0.14 |
| fuzzy_typo | 1.00 | 0.88 | −0.12 |
| multi_evidence | 0.36 | 0.32 | −0.04 |
| paraphrase | 0.25 | 0.27 | +0.02 |
| overall | 0.819 | 0.717 | −0.102 |

Graph expansion raises overall Recall@10 (0.863 → 0.882) by pulling in linked
evidence, and lowers MRR@10 by 0.10 by inserting neighbours above the direct
hit. On the in-memory run the same comparison shows graph raising
`multi_evidence` MRR from 0.32 to 0.61, which is the intent it was built for —
but the decoy-heavy regression is larger than that gain. The graph RRF weight
(1.75 for interactive recall) and the query-gate floor (0.60) are the two knobs
that should be swept before this lane is enabled by default for interactive
recall; that sweep is P4 debt and is not done here.

**Removing the dense lane makes the direct ranking better on this corpus.**
Running the same 40 queries with no dense lane at all (`--no-dense`, in-memory):

| Configuration | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- |
| direct fused, dense on | 0.853 | 0.807 | 0.793 |
| direct fused, dense off | **0.882** | **0.824** | **0.811** |
| lexical only | 0.868 | 0.780 | 0.777 |
| final fused (shipped config) | 0.858 | 0.774 | 0.762 |

A hash embedder given RRF weight 4.0 against lexical's 3.0 injects more noise
than signal. This is a statement about the stand-in embedder, not about hybrid
retrieval, and it is the sharpest argument for the Titan rerun.

### No-answer behaviour

This is the weakest measured result and it is a real finding.

All six no-answer queries return ten hits with a top-1 public score of exactly
`1.00`. Final no-answer recall is **0.00**: the retriever never abstains on a
topically absent query as long as one token overlaps anything in the corpus.
Only the precision lanes abstain, and they abstain for the wrong reason —
exact abstains on 31 of 40 cases (precision 0.19), trigram on 20 of 40
(precision 0.30).

Raising the caller-supplied floor does not fix it:

| `min_score` | Recall@10 | MRR@10 | no-answer precision | no-answer recall | mean hits returned |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.858 | 0.774 | 1.00 | 0.00 | 10.00 |
| 0.2 | 0.858 | 0.774 | 1.00 | 0.00 | 9.88 |
| 0.4 | 0.858 | 0.774 | 1.00 | 0.00 | 9.18 |
| 0.5 | 0.843 | 0.744 | 0.00 | 0.00 | 9.13 |
| 0.8 | 0.843 | 0.744 | 0.00 | 0.00 | 8.50 |

The cause is structural, not a tuning miss. The public score is normalised
against the best possible rank in the strongest configured lane, so any
candidate that lands at rank one in the lexical lane scores `1.00` regardless
of how weak its raw overlap is. The score therefore carries rank information
and no relevance information, and no threshold on it can separate "the best of
a bad field" from "a good answer".

What this measurement does **not** say: the `hits=[]` guarantee for a query
with zero lexical overlap still holds — that is what the checked-in gold
regression covers. What is missing is a calibrated abstention signal (a raw
lexical/cosine floor exposed separately from the normalised score, or a
score-gap test between rank 1 and the tail). That is now a named gap rather
than an assumption.

### Latency

Per-query wall time for the whole fused path, 40 queries:

| Backend | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| CockroachDB | 65.0 ms | 94.9 ms | 119.9 ms | 119.9 ms |
| in-memory | 21.1 ms | 28.3 ms | 33.4 ms | 33.4 ms |

Per-lane latency on CockroachDB (p50 / p95):

| Lane | p50 | p95 |
| --- | --- | --- |
| exact | 2.4 ms | 5.7 ms |
| graph | 13.9 ms | 21.9 ms |
| lexical | 21.9 ms | 32.8 ms |
| fuzzy | 28.4 ms | 40.2 ms |
| dense (ANN) | 39.3 ms | 50.2 ms |

Lanes run concurrently, so wall time is close to the slowest lane plus
hydration. No lane was degraded in any run.

### ANN Recall@k versus the exact-vector oracle

For all 40 queries, `CockroachDenseRetrievalGateway.retrieve()` (ANN through
`retrieval_vectors_1024_ann_v2` with prefix filtering, same-snapshot canonical
validation and adaptive widening) was compared with `retrieve_exact()`
(`@primary`, eligible set filtered before the exact vector sort) inside the
same snapshot:

| Metric | Value |
| --- | --- |
| mean ANN Recall@5 | 1.000 |
| mean ANN Recall@10 | 1.000 |
| min ANN Recall@5 | 1.000 |
| min ANN Recall@10 | 1.000 |
| ANN latency p50 / p95 | 9.1 ms / 11.0 ms |
| exact-oracle latency p50 / p95 | 27.4 ms / 32.3 ms |
| vectors in scope | 90 |

Caveat that matters more than the number: 90 vectors in one scope is far below
the size at which a vector index makes approximation trade-offs. Perfect ANN
recall here proves the prefix binding, the canonical validation and the
adaptive widening are wired correctly and do not drop eligible rows. It does
not characterise ANN recall at scale, across selectivity buckets, or under
filter–vector correlation. That characterisation is P3 debt and stays open.

## Track 2 — LongMemEval-S retrieval subtask (official dataset)

### Dataset and ingestion mapping

| Item | Value |
| --- | --- |
| Dataset | LongMemEval-S |
| Source | `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json` |
| SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Questions evaluated | 500 of 500 (no sampling) |
| Haystack sessions per question | 38 – 62, median 48 |
| Backend | in-memory kernel only |

The 265 MiB release is never committed. `scripts/run_retrieval_eval.py`
downloads it to `~/.cache/swarmbrain-eval/` (override with
`SWARMBRAIN_EVAL_DATA_DIR`) and verifies the pinned digest on every run. A tiny
synthetic sample in the dataset's own shape lives at
`tests/fixtures/retrieval_eval_corpus/longmemeval_sample.json` so the mapper
stays under test without the release.

Mapping, stated precisely so it can be argued with:

1. **Granularity is one memory per haystack session.** LongMemEval labels
   evidence at session level (`answer_session_ids`), so a session-level memory
   is a 1:1 match for the judgment unit. Turn-level memories would require
   re-deriving relevance from per-turn `has_answer` flags and would change the
   task.
2. **Scope is one fresh runtime per question.** Each question owns its
   haystack, so each question gets its own kernel, tenant/project/repository/run
   and embedding index, and the runtime is discarded afterwards. There is no
   leakage between questions and no shared corpus. When dense retrieval is
   enabled, all sessions for that question are projected through one bounded
   `embed_documents` call rather than one provider request per memory; the
   vectors enter the same scope-aware index before recall.
3. **Memory shape.** `kind=observation`, `state=confirmed`,
   `visibility=repository`, content = the session rendered as
   `role: content` lines, title = `Conversation session recorded <haystack date>`,
   tags `("longmemeval", "session")`, and `valid_from` = the session timestamp
   normalized into the benchmark's explicit UTC calendar. `valid_to` stays
   open because a conversation observation is not known to expire. Session
   identifiers are not put in the searchable text.
4. **Query.** The `question` field verbatim through the real recall path,
   `limit=10`.
5. **Relevance.** A retrieved memory is relevant iff its session id is in
   `answer_session_ids`. Session ids repeat inside 15 haystacks, so memories
   are keyed `<position>:<session_id>` and every position carrying a labelled
   id counts.
6. **Abstention questions.** The 30 `*_abs` questions still carry
   `answer_session_ids` — in LongMemEval those are the sessions a reader must
   consult in order to conclude that the answer was never given. They are
   therefore **not** no-answer retrieval cases and are not folded into
   no-answer precision. They are reported as a separate slice.

Referenced-time query parsing is an explicit experiment, not part of the
baseline. `--temporal-query-routing` anchors the conservative parser to the
record's `question_date`, copies only a closed half-open proposal into
`RecallQuery.referenced_valid_*`, and records status, reason, confidence and
bounds per case. Open, vague, conflicting, comparison-only and unsupported
phrases do not route. This makes a baseline/temporal A/B reviewable without
silently changing serving defaults or treating `recorded_at` as event time.

### Results, all 500 questions

> **Historical result, pending cleaned-release rerun.** The table below was
> measured against the superseded pre-September-2025 history release (SHA-256
> `08d8dad4…`). The current SOTA gate pins the cleaned release above and rejects
> this artifact until the full-500 run is repeated.

| Lane | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| exact | 0.000 | 0.000 | 0.000 | 0.000 |
| lexical | **0.799** | **0.885** | **0.792** | **0.779** |
| fuzzy | 0.000 | 0.000 | 0.000 | 0.000 |
| dense (hash embedder) | 0.351 | 0.500 | 0.344 | 0.351 |
| graph | 0.000 | 0.000 | 0.000 | 0.000 |
| direct fused | 0.594 | 0.768 | 0.574 | 0.581 |
| final fused | 0.594 | 0.768 | 0.574 | 0.581 |
| final fused, dense lane off | **0.799** | **0.885** | **0.792** | **0.779** |

With the dense lane disabled the fused ranking is numerically identical to the
lexical lane, because exact, trigram and graph return nothing on this corpus
(see below). The best measured LongMemEval-S retrieval configuration of this
system is therefore Recall@5 0.799, Recall@10 0.885, MRR@10 0.792,
nDCG@10 0.779 over all 500 questions.

Per-question-type Recall@10:

| Question type | n | lexical | dense | fused |
| --- | --- | --- | --- | --- |
| knowledge-update | 78 | 0.962 | 0.635 | 0.891 |
| single-session-user | 70 | 0.957 | 0.529 | 0.843 |
| temporal-reasoning | 133 | 0.890 | 0.389 | 0.712 |
| multi-session | 133 | 0.858 | 0.427 | 0.706 |
| single-session-assistant | 56 | 0.839 | 0.857 | 0.911 |
| single-session-preference | 30 | 0.700 | 0.233 | 0.533 |

Abstention slice (30 `*_abs` questions), Recall@10 / MRR@10: lexical
0.881 / 0.791, dense 0.461 / 0.291, fused 0.811 / 0.519. Retrieval on
abstention questions behaves like retrieval on the rest; the abstention
difficulty in LongMemEval lives in the reader, which this evaluation does not
run.

Wall time per question (ingest 50 sessions, embed them, run one recall):
p50 130 ms, p95 147 ms, p99 180 ms for the recall itself.

### Referenced-time routing A/B: rejected for default serving

The opt-in parser found 40 closed routable proposals in the cleaned 500
questions, one correctly open/non-routable proposal, 442 no-matches and 17
fail-closed rejections. A full lexical-only A/B then compared the unchanged
baseline with `--temporal-query-routing`:

| Metric | baseline | routed | delta |
| --- | ---: | ---: | ---: |
| Recall@10 | 0.8848 | 0.8727 | -0.0120 |
| MRR@10 | 0.7923 | 0.7788 | -0.0136 |
| nDCG@10 | 0.7795 | 0.7658 | -0.0136 |
| any gold in 16k-token context | 0.882 | 0.866 | -0.016 |
| all gold in 16k-token context | 0.658 | 0.646 | -0.012 |

The failure is concentrated exactly where the feature fired. Across the 40
routed questions, baseline Recall@10/MRR@10 was 0.7175/0.6242; routing produced
0.5671/0.4548, with 3 recall wins, 15 losses and 22 ties. These figures splice
a deterministic rerun of the only case changed by correcting bounded
`since ... ago` parsing into the original full-500 A/B. An offline
temporal-weight sweep on the original run could not recover the baseline: even
zero temporal fusion weight scored 0.8759 overall because the
referenced-validity filter had already removed candidates.

This falsifies the ingestion assumption, not the explicit temporal contract.
A LongMemEval haystack date is the time a conversation was observed; it is not
necessarily the occurrence time of every event mentioned in that session.
Later sessions can retrospectively contain the gold evidence, so treating the
session timestamp as hard event validity deletes useful memories. Concretely,
16 of the 93 gold sessions attached to the 40 routed questions occur at or
after the parsed interval end; none can survive the hard filter. Eight cases
lose at least one gold this way, five lose every gold session and return no
eligible candidate. Those eight cases average -0.6458 Recall@10 and -0.6875
MRR@10. Even the 23 routed cases with at least one gold session timestamp
inside the parsed interval lost 0.0804 Recall@10 on average after temporal
fusion, so converting the lane to a soft prior would also require held-out
calibration. The parser and
parity-tested lane stay available for explicit-validity data, but the A/B flag
remains experimental and is not promoted. Schema v12 now provides a
provenance-backed `occurred_at` timestamp, separate from both session
observation time and the bitemporal intervals, plus an explicit soft-prior
query interval. The prior never becomes a validity filter and unknown event
time is neutral. This corrects the representation boundary; its weight and
query gate still require a held-out A/B before any default promotion.

### What this track shows

- **Exact, trigram and graph contribute nothing here, by construction.** The
  exact lane fires only when the whole normalised query equals a projected
  term (a title, tag, path, symbol, test name, commit or error code); a
  natural-language question never does, so it returned nothing on all 500
  questions. The trigram lane scores the query against the same
  identifier-oriented lookup text, which for an imported chat session is a
  date-shaped title plus two tags, and never reached the `0.25` similarity
  floor. There are no `memory_links` between imported sessions, so the graph
  lane had no seeds to expand. This is the designed behaviour of lanes built
  for code memory applied to a corpus that is not code memory; it is not a
  defect being hidden.
- **The lexical lane carries the whole score**, and it carries it well:
  Recall@10 0.885, MRR@10 0.792 over 500 questions with roughly 50 candidate
  sessions each.
- **The hash dense lane actively hurts fusion.** Fused Recall@10 (0.768) is
  0.117 below lexical alone (0.885) and fused MRR@10 (0.574) is 0.218 below.
  With dense weighted 4.0 against lexical's 3.0, a non-semantic lane
  systematically displaces correct sessions. Turning it off recovers the
  lexical result exactly.
- **Recall@10 0.885 is a retrieval number and only a retrieval number.** It
  says that for 500 LongMemEval-S questions the evidence sessions are inside
  the top ten of roughly fifty candidates 88.5% of the time. It says nothing
  about whether a reader would answer correctly from them, which is what
  LongMemEval's published scores measure.

## Reproducing

```bash
# Track 1, in-memory
uv run --extra dev python scripts/run_retrieval_eval.py --track swarm --backend memory

# Track 1, live CockroachDB (creates and installs swarmbrain_eval itself)
SWARMBRAIN_EVAL_DATABASE_URL="postgresql://root@127.0.0.1:26257/swarmbrain_eval?sslmode=disable" \
  uv run --extra dev python scripts/run_retrieval_eval.py --track swarm --backend cockroach

# Lane ablation without the dense lane
uv run --extra dev python scripts/run_retrieval_eval.py --track swarm --backend memory --no-dense

# Track 2, all 500 LongMemEval-S questions (first run downloads 265 MiB)
uv run --extra dev python scripts/run_retrieval_eval.py \
  --track longmemeval --lme-sample 0 --lme-download

# Experimental temporal-lane A/B; writes a separate *-temporal artifact
uv run --extra dev python scripts/run_retrieval_eval.py \
  --track longmemeval --lme-sample 0 --no-dense --temporal-query-routing

# Authenticated OpenAI-compatible embeddings; the secret value is never a CLI argument
uv run --extra dev python scripts/run_retrieval_eval.py \
  --track longmemeval --lme-sample 0 --embeddings openai \
  --embeddings-base-url "$SWARMBRAIN_EMBEDDINGS_BASE_URL" \
  --embeddings-model Qwen/Qwen3-Embedding-0.6B \
  --embeddings-api-key-env SWARMBRAIN_EMBEDDINGS_API_KEY

# Publishable exact reader-context accounting. The executable implements
# swarmbrain-exact-tokenizer-jsonl-v1; all paths must be repository-local.
uv run --extra dev python scripts/run_retrieval_eval.py \
  --track longmemeval --lme-sample 0 --embeddings openai \
  --embeddings-base-url "$SWARMBRAIN_EMBEDDINGS_BASE_URL" \
  --embeddings-model Qwen/Qwen3-Embedding-0.6B \
  --context-tokenizer-executable benchmarks/sota/evidence/tokenizer/provider \
  --context-tokenizer-executable-sha256 "$TOKENIZER_PROVIDER_SHA256" \
  --context-tokenizer-artifact benchmarks/sota/evidence/tokenizer/tokenizer.json \
  --context-tokenizer-artifact-sha256 "$TOKENIZER_ARTIFACT_SHA256" \
  --context-tokenizer-model "$LME_READER_MODEL" \
  --context-tokenizer-revision "$LME_READER_REVISION"

# Score any saved run directly
uv run --extra dev python scripts/evaluate_retrieval_runs.py \
  benchmarks/retrieval/swarm-native-cockroach-run.json --k 10
```

Saved runs and reports live in `benchmarks/retrieval/`, not in `evidence/`.
`evidence/` holds timestamped, append-only demo artifacts; a benchmark is a
versioned artifact that should be overwritten in place so a rerun produces a
reviewable diff against the previous measurement.

The QA harness can replay a dataset-bound retrieval artifact instead of
regenerating 23,867 document embeddings for every reader run:

```bash
uv run --extra dev python scripts/run_longmemeval_qa.py \
  --lme-path /private/tmp/longmemeval_s_cleaned.json \
  --lme-sample 0 --limit 10 --prompt-style official --no-judge \
  --retrieval-run benchmarks/retrieval/longmemeval-s-memory-openai-run.json \
  --reader-base-url "$LME_READER_BASE_URL" --reader-model "$LME_READER_MODEL" \
  --reader-revision "$LME_READER_REVISION" \
  --reader-api-key-env LME_READER_API_KEY
```

Replay validates the schema-v2 artifact type/protocol and implementation tree,
the cleaned dataset SHA-256 and exact 500-question/23,867-session shape, case
coverage, recall depth, session-position keys and per-hit relevance before the
first reader call. Publishable replay additionally requires the pinned
Qwen3-Embedding-0.6B semantic metadata, exact endpoint response-model contract,
provider-observed document/query/HTTP call reconciliation, and no degraded
lanes. The source path, byte length and SHA-256 are copied into the QA run. The
saved final ranking determines the exact reader bundle; no embedding endpoint
is contacted. Replay therefore requires `--min-score 0.0`: a positive floor can
continue past filtered head candidates during live retrieval, but the saved
final bundle does not carry calibrated relevance for that deeper tail.
This is the intended path for the three official-generation repeats because it
holds retrieval constant and removes 71,601 redundant document-vector
computations plus 3,000 bounded embedding HTTP calls. The one retrieval run
it reuses needs 500 batched document calls and 500 query calls, down from the
old one-request-per-session path's 24,367 calls.

The QA schema also binds the exact hypothesis bytes and records the response-
reported reader model, an operator-pinned deployment/checkpoint revision,
provider request ID, and optional system fingerprint per successful question.
The official-report compiler revalidates all of those receipts, the exact
retrieval bundle copied into every question, and raw official-label bytes. Use
`--allow-nonpublishable-retrieval-run` or
`--allow-unverified-reader-response` only for development; either condition is
rejected by the official compiler.

### Determinism

The corpus content, memory order, injected clock and query set are fixed, and
the metrics are reproducible: two consecutive in-memory runs produced
byte-identical evaluator output at k = 10. The saved runs are not
byte-identical, because canonical memory ids are fresh UUIDs per run and the
in-memory dense index breaks exact cosine ties by id. In two runs this moved
one pair of tied candidates at ranks 32–33 of one no-answer case and changed
nothing at any reported depth. The LongMemEval sample is drawn with an explicit
seed (`--lme-seed`, default `20260807`); the reported run used all 500
questions, so no sampling was applied.

## Caveats

- The dense numbers are a floor produced by a hash embedder, not a semantic
  model. Do not read them as evidence about dense retrieval, and do not retune
  RRF weights from them.
- The swarm-native corpus has 90 memories and 40 queries. It is large enough to
  separate lanes and small enough that a single judgment error moves a category
  average by several points. Per-category cells with n = 3 or 4 are directional
  only.
- The judgments are single-annotator. There is no second annotator and no
  inter-annotator agreement figure.
- ANN Recall@k was measured on 90 vectors in one scope. It proves plumbing, not
  index quality at scale.
- Latency was measured on a developer laptop shared with other work. CPU, rows
  read, bytes read, projection freshness lag and storage per vector were not
  captured; those remain part of the P3 operational debt.
- The corpus is authored, not sampled from a real deployment. It reflects what
  the author believes swarm memory looks like.
- No result here supports a claim of state-of-the-art retrieval quality, and
  none is made.

## Follow-ups

1. Rerun both tracks with `BedrockEmbeddingProvider`
   (`amazon.titan-embed-text-v2:0`, 1024 dims) and report the same tables. Only
   then are dense and fusion-weight conclusions meaningful.
2. Fix abstention: expose a calibrated raw-score floor or a rank-gap test that
   is separate from the rank-anchored public score, then re-measure no-answer
   precision and recall.
3. Sweep the graph lane (RRF weight, query-gate floor, hops, fan-out) against
   the `decoy_heavy` regression and the `multi_evidence` gain before enabling
   two-hop expansion for interactive recall.
4. Grow the corpus and add a second annotator before any learned fusion or
   reranker is tuned, per the rule in
   [retrieval evaluation](retrieval-evaluation.md).
5. Measure ANN recall at scale across scope-selectivity and
   filter–vector-correlation buckets, with beam-size and overfetch sweeps.

Follow-up 2 is answered below and no longer open; follow-ups 1, 3, 4 and 5
still are.

---

# Addendum, 2026-08-08 — calibrated abstention signal

Everything above this line is the original 2026-08-07 measurement and is left
exactly as it was written. This addendum reports a change to the retriever and
a re-measurement of the same corpus with the same evaluator. It closes
follow-up 2 and it does not touch any other conclusion in the report.

## What changed

`RecallQuery.min_score` no longer filters on the public `RecallHit.score`. It
filters on a new server-side quantity, **calibrated candidate relevance**
(`swarmbrain.retrieval.relevance`, version `lane-max-v1`), and the public score
is unchanged in both meaning and value.

The original diagnosis was that the public score is normalised weighted RRF
anchored to the best possible rank in the strongest configured lane, so it
carries rank information and no relevance information. That diagnosis stands.
The fix is to compute the missing quantity separately rather than to redefine
the score, so no caller reading `score` sees a different number.

Relevance is the **maximum** of the lanes' own evidence, each already a
similarity in [0, 1] and none of them a function of rank:

| Component | Source | Why it is relevance and not rank |
| --- | --- | --- |
| exact term | `1.0` when the exact lane produced the candidate | the lane fires only on whole-query equality with a projected term, or on a caller-supplied memory id / planner seed |
| lexical coverage | fraction of the query's distinct content tokens present in the memory's `search_text` projection | recomputed from canonical text, not read from the lane |
| trigram similarity | `trigram_similarity(lookup_text(memory), normalised query)` | recomputed from canonical text, not read from the lane |
| dense cosine | the cosine carried on the dense fusion contribution, clamped to [0, 1] | cannot be recomputed without the query vector; both adapters produce cosine in the same projection space |
| graph | `0.0` | graph activation is a function of a *seed's* fused rank and of edge weights, so it is rank-derived by construction |

Two design points are load-bearing and worth stating plainly.

**Why `max` and not a sum.** A candidate is kept when *some* lane can defend it
on its own evidence. Summing or averaging would let several individually weak
lanes manufacture a passing score, which is the failure being fixed.

**Why the lexical and trigram components are recomputed instead of calibrated
from lane raw scores.** The two adapters' lexical raw scores are on different
scales — the in-memory kernel reports token overlap, CockroachDB reports
`ts_rank`, which is roughly an order of magnitude smaller for the same match.
No affine calibration of those raw scores can mean the same thing on both
backends. Recomputing both components from the canonical memory text, with the
same `swarmbrain.retrieval.projection` helpers both adapters index with, makes
the floor backend-independent by construction. This is verified, not assumed:
see the parity result below.

Cost: no extra queries. Everything comes from the canonical memory that
hydration already loaded plus the per-lane raw scores the trace already
carries. Candidates are evaluated lazily in fused order and the walk stops one
past the public `limit`, so the default path pays for at most `limit + 1`
computations.

## Abstention versus answerable recall — the trade-off curve

Measured by **re-executing all 40 queries at each floor**, not by filtering a
saved top-10 offline. That distinction matters: the floor is applied before
truncation, so raising it lets the server walk further down the fused list and
backfill the remaining slots. An offline filter of the saved ranking would
report a pessimistic recall, and at low floors it would miss that recall
*rises*.

Shipped configuration (all five lanes, `DeterministicEmbeddingProvider`), k = 10:

No-answer recall, no-answer precision and the empty-bundle count are identical
on both backends at every floor; the mean-hits column is the in-memory run and
CockroachDB differs from it by at most 0.03.

| `min_score` | in-memory R@10 | in-memory MRR@10 | CockroachDB R@10 | CockroachDB MRR@10 | no-answer recall | no-answer precision | empty bundles (of 40) | mean hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.858 | 0.774 | 0.882 | 0.717 | 0.00 | 1.00 | 0 | 10.00 |
| 0.10 | **0.873** | 0.779 | **0.897** | 0.717 | 0.00 | 1.00 | 0 | 9.65 |
| 0.20 | 0.848 | 0.779 | 0.853 | 0.745 | 0.17 | 0.50 | 2 | 7.80 |
| 0.25 | 0.843 | 0.787 | 0.848 | 0.776 | 0.50 | 0.75 | 4 | 6.60 |
| 0.30 | 0.828 | 0.760 | 0.828 | 0.766 | 0.50 | 0.50 | 6 | 5.40 |
| 0.35 | 0.804 | 0.770 | 0.804 | 0.797 | 0.83 | 0.62 | 8 | 3.60 |
| **0.40** | **0.784** | 0.777 | **0.784** | **0.814** | **1.00** | 0.55 | 11 | 3.30 |
| 0.50 | 0.735 | 0.765 | 0.735 | 0.789 | 1.00 | 0.50 | 12 | 2.52 |
| 0.60 | 0.667 | 0.667 | 0.667 | 0.686 | 1.00 | 0.40 | 15 | 1.50 |
| 0.80 | 0.441 | 0.471 | 0.441 | 0.471 | 1.00 | 0.25 | 24 | 0.65 |

Compare with the control, the same sweep on the rank-anchored public score
(`no_answer_public_score_sweep`, still produced on every run): in both
shipped-configuration runs it never reaches a no-answer recall above `0.00` at
any floor up to 0.8, reproducing the original finding exactly. The signal, not
the threshold, was the problem.

Lane ablation without the dense lane (`--no-dense`, in-memory), which is what
the abstention signal looks like when the non-semantic stand-in embedder is
removed:

| `min_score` | R@10 | MRR@10 | no-answer recall | empty bundles |
| --- | --- | --- | --- | --- |
| 0.00 | 0.838 | 0.635 | 0.00 | 0 |
| 0.10 | **0.927** | **0.720** | 0.00 | 0 |
| 0.20 | 0.809 | 0.713 | 0.67 | 6 |
| **0.25** | **0.799** | 0.772 | **1.00** | 9 |
| 0.40 | 0.770 | 0.768 | 1.00 | 12 |

**The honest headline.** Full abstention on all six no-answer queries costs
Recall@10 in the shipped configuration: `0.882 → 0.784` on CockroachDB and
`0.858 → 0.784` in memory, a loss of 0.098 and 0.074. MRR@10 *improves* over
the same interval (CockroachDB `0.717 → 0.814`, in-memory `0.774 → 0.777`),
because what the floor removes is mostly ranked above the evidence it keeps.
With the dense lane off, full abstention arrives at 0.25 for a smaller recall
loss (`0.838 → 0.799`). No floor buys abstention for free, and no number below
is presented as if it did.

## Where the recall loss goes

Recall@10 / MRR@10 by query intent, in-memory, shipped configuration:

| Intent | n | floor 0.00 | floor 0.25 | floor 0.40 |
| --- | --- | --- | --- | --- |
| identifier_exact | 8 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| code_lookup | 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| lexical | 7 | 1.000 / 0.929 | 1.000 / 0.929 | 1.000 / 1.000 |
| decoy_heavy | 4 | 1.000 / 0.432 | 0.875 / 0.792 | 0.875 / 0.875 |
| fuzzy_typo | 4 | 0.875 / 0.812 | 0.875 / 0.812 | 0.750 / 0.583 |
| multi_evidence | 3 | 0.556 / 0.611 | **0.889** / 0.611 | 0.556 / 0.528 |
| paraphrase | 5 | 0.400 / 0.400 | 0.200 / 0.200 | 0.100 / 0.200 |

The loss is concentrated in exactly one place. At floor 0.40, four of the five
`paraphrase` queries return an empty bundle — and three of those four were
already returning **no** judged evidence at floor 0.00, so the retriever now
says "I have nothing" instead of returning ten confident wrong answers. The two
genuinely damaged cases are `paraphrase-transaction-conflict` (both judged
memories dropped; their best evidence is 0.18 coverage, 0.07 trigram, 0.15
cosine) and `fuzzy-typo-test-name` (dropped at 0.35; its only evidence is a
0.294 trigram similarity on a mistyped test name). `decoy-search-freshness`
loses one of two judged memories and `multi-duplicate-charge-thread` loses one
of two at 0.40.

Every one of those is a case where lexical, trigram and hash-embedder evidence
are all genuinely near zero — which is precisely the case a real semantic
embedder is supposed to rescue and this one cannot. This is the sharpest
argument yet for follow-up 1, and it is a reason to re-measure this table after
the Titan rerun rather than to pick a lower floor now.

Two intents *gain* from a modest floor: `decoy_heavy` MRR@10 rises 0.432 →
0.792 at 0.25 and `multi_evidence` Recall@10 rises 0.556 → 0.889, because
backfill replaces low-relevance graph neighbours with judged evidence that was
sitting below rank 10.

## What is recommended, and what is not

- **The default stays `min_score = 0.0`.** Nothing about an unfiltered recall
  changed: at floor 0.0 every lane metric in all three saved reports is
  byte-identical to the 2026-08-07 run.
- **A deployment that must abstain should set `min_score = 0.40`** with the
  current lane set, or `0.25` with the dense lane disabled, and accept the
  recall cost in the table above.
- **`min_score = 0.10` is free or better** on this corpus — Recall@10 rises
  (0.882 → 0.897 on CockroachDB, 0.858 → 0.873 in memory), MRR does not fall,
  and mean hits drop from 10.00 to 9.65 — but it produces **no** abstention. It
  is a precision floor, not an abstention floor. It is not made the default
  here, because changing the default would change what every existing caller
  gets from an unfiltered recall, and that is a decision for a release, not for
  a benchmark.
- **No server-side absolute floor was added.** The one conservative candidate
  was dropping candidates whose only defence is a graph contribution; measured,
  it changes no metric at floor 0.0 (those candidates already score 0.0
  relevance and are removed by any positive floor) and it would silently reduce
  what a default caller receives. It was not worth changing the default for.

## Adapter parity

The abstention behaviour is identical on both backends. Across the ten floors
above, the number of empty bundles is the same sequence on the in-memory kernel
and on live CockroachDB (0, 0, 2, 4, 6, 8, 11, 12, 15, 24) and no-answer recall
matches at every floor. Recall differs at low floors only where it already
differed in the original report, because CockroachDB's `TSVECTOR` lexical lane
is stronger than the in-memory reference scorer.

A live test (`tests/retrieval/test_relevance_parity_live.py`) publishes the
same four memories to both backends and asserts that the per-candidate
relevance values and the resulting abstention at `min_score = 0.25` are equal
for every query in a fixed set, including two topically absent ones.

## Latency

The relevance computation adds bounded Python work over already-hydrated text
and issues no queries. Measured on the same machine within minutes of each
other, before and after the change:

| Backend | wall p50 before | wall p50 after |
| --- | --- | --- |
| in-memory | 30.4 ms | 31.3 ms |
| CockroachDB | 91.1 ms | 94.9 ms |

Isolated microbenchmark of the computation itself: 1.4 ms for eleven candidates
(0.26 ms lexical coverage, 1.19 ms lookup/trigram), which accounts for the
in-memory delta. Both absolute numbers are higher than the 2026-08-07 figures
(21.1 ms and 65.0 ms) because this machine was running other work; the
before/after pair, not the absolute value, is the measurement.

CockroachDB ANN Recall@5 and Recall@10 against the exact-vector oracle remain
1.000 mean and 1.000 min over all 40 queries.

## What remains open

- **The floor is corpus-calibrated, not learned.** `0.40` and `0.25` are read
  off the curves above on a 90-memory, 40-query, single-annotator corpus. They
  are honest for this corpus and should be re-derived after the Titan rerun and
  after the corpus grows. No claim is made that they transfer.
- **Paraphrase abstention is a semantic-model gap, not a threshold gap.** With
  a non-semantic embedder, any floor that separates the no-answer queries also
  removes the weakest paraphrase evidence. The correct fix is follow-up 1.
- **The function-word list used by the lexical component is English-only.** On
  a non-English query nothing is filtered and the component degrades to plain
  token coverage, which is more permissive, never less — so the failure
  direction is "returns hits a floor might have dropped", not "drops relevant
  hits". It has not been measured on a non-English corpus, because there is
  none.
- **The pre-v1 store fallback keeps the old semantics.** `MemoryService`
  constructed without a `RetrievalService` falls back to `store.recall()`,
  where `min_score` still filters on that store's own score. Production
  runtimes install the v1 retrieval path, but the two are not the same floor
  and this is not unified.
- **Nothing here measures LongMemEval-S.** That track's saved runs are
  unchanged and were not re-run: it is evaluated at the default
  `min_score = 0.0`, where behaviour is identical, and its `*_abs` questions
  are not no-answer retrieval cases (see Track 2 above).

## Reproducing the addendum

```bash
# in-memory, with the abstention sweep (default thresholds)
uv run --extra dev python scripts/run_retrieval_eval.py --track swarm --backend memory

# live CockroachDB
SWARMBRAIN_EVAL_DATABASE_URL="postgresql://root@127.0.0.1:26257/swarmbrain_eval?sslmode=disable" \
  uv run --extra dev python scripts/run_retrieval_eval.py --track swarm --backend cockroach

# without the dense lane
uv run --extra dev python scripts/run_retrieval_eval.py --track swarm --backend memory --no-dense

# custom floors, or none at all
uv run --extra dev python scripts/run_retrieval_eval.py --track swarm --min-score-sweep 0.0 0.4
```

The sweep lands in each saved report under `no_answer_min_score_sweep`; the
original public-score sweep is retained beside it as
`no_answer_public_score_sweep`. Per-hit relevance for the default run is saved
per case as `final_relevance` next to `final_scores`.

Environment for this addendum: repository `swarm-brain` at `90975f6` plus the
uncommitted change described here, Python 3.13.9 arm64, CockroachDB CCL
v26.2.1, evaluation database `swarmbrain_eval`, same corpus
`swarm-coding-2026-08-07` and judgments `r1` — no query, judgment or corpus
memory was added, removed or edited.

---

# Addendum, 2026-08-09 — semantic dense lane and interactive graph weight

Everything above this line is left exactly as written. This addendum reports
two retriever changes and a re-measurement of both tracks with the same
corpora, judgments and evaluator. It answers follow-up 1 in substance (a real
semantic embedder, though self-hosted rather than Titan) and closes the
fusion-weight part of follow-up 3.

## What changed

1. **A real semantic embedder fills the dense slot.** A new `openai`
   embeddings kind (`swarmbrain.adapters.embeddings.openai_compatible`) points
   the existing dense v2 pipeline at any `/v1/embeddings`-compatible server.
   These runs used `Qwen/Qwen3-Embedding-0.6B` (Apache 2.0, native 1024
   dimensions, cosine) served by vLLM 0.26 on a self-hosted RTX 5070 Ti.
   Queries carry the model's instruction prefix; documents embed raw; vectors
   are L2-normalized client-side so the stored projection does not depend on
   server pooling configuration. Transient network faults retry with backoff;
   protocol errors (wrong dimensions, malformed payloads) fail fast.
2. **The interactive-recall graph weight dropped from 1.75 to 0.5.** Measured
   with the semantic embedder at 1.75, graph expansion improved zero of the
   forty swarm queries and lowered reciprocal rank on seven: with a strong
   dense lane the direct lanes already surface linked evidence, and expansion
   mostly promoted connected decoys. An offline re-fusion sweep over the saved
   lane rankings located the recovery at weights ≤ 0.5 and the live rerun
   confirmed it. Purposes built around traversal keep their previous weights
   and hops (conflict review 2.5; planning and handoff 1.5; task bootstrap
   unchanged), so this changes interactive and historical recall only.

Runs with a non-default embedder write a separate `-openai` file family, so
every deterministic baseline in this report remains byte-reproducible. The
semantic runs are *not* byte-reproducible: they depend on a served model and a
network. Provider, model and dimensions are recorded in each run's metadata.

## Track 1 — swarm corpus, in-memory, final lane, k = 10

| Configuration | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- |
| hash embedder, graph 1.75 (2026-08-07 baseline) | 0.858 | 0.774 | 0.762 |
| Qwen3-0.6B, graph 1.75 | 0.897 | 0.799 | 0.801 |
| Qwen3-0.6B, graph 0.5 (shipped) | 0.926 | 0.912 | 0.882 |
| Qwen3-0.6B, graph 0.5, `min_score = 0.40` | 0.941 | 0.922 | — |

Dense-only rose from 0.679 / 0.547 to 0.852 / 0.767, and direct fusion beats
every single lane (0.931 / 0.912 against lexical's 0.868 / 0.780) — which the
hash embedder never achieved. At graph weight 0.5 the per-query graph impact
is hurt 0 / helped 0 / neutral 34 (was hurt 7 at 1.75), and final fusion now
matches direct fusion instead of trailing it by 0.11 MRR.

The 2026-08-08 addendum predicted that the floor's recall losses were
"precisely the case a real semantic embedder is supposed to rescue". Measured
per intent (final lane, Recall@10 / MRR@10, hash → Qwen at floor 0.0):
`paraphrase` 0.400 / 0.400 → 0.500 / 0.600, `fuzzy_typo` 0.875 / 0.812 →
1.000 / 1.000, `multi_evidence` 0.556 / 0.611 → 1.000 / 0.833, `decoy_heavy`
MRR 0.431 → 0.875.

With the semantic embedder the abstention floor stops being a trade-off on
this corpus: at `min_score = 0.40`, answerable Recall@10 *rises* to 0.941
(weak-but-ranked hits give way to judged evidence from backfill), MRR@10 is
0.922, and no-answer precision / recall are 1.00 / 0.50. Floor 0.50 reaches
no-answer 0.83 / 0.83 at Recall@10 0.843. Bundle precision — the fraction of
returned hits that are judged relevant — rises from 0.153 at floor 0.0 (ten
hits always) to 0.468 at floor 0.50 (about 4.4 hits), which is the
quality-per-token argument for a positive floor in token-budgeted callers.

## Track 2 — LongMemEval-S, all 500 questions

| Lane | R@5 (hash → Qwen) | R@10 (hash → Qwen) | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| lexical | 0.799 → 0.799 | 0.885 → 0.885 | 0.792 → 0.792 | 0.779 → 0.779 |
| dense | 0.351 → 0.939 | 0.500 → 0.962 | 0.344 → 0.914 | 0.351 → 0.913 |
| final fused | 0.594 → 0.924 | 0.768 → 0.976 | 0.574 → 0.906 | 0.581 → 0.906 |

The 2026-08-07 headline inverted. With the hash embedder, fusing the dense
lane *cost* 0.117 Recall@10 against lexical alone; with Qwen3-0.6B the dense
lane is the strongest single lane and final fusion reaches **Recall@10 0.976,
MRR@10 0.906** — 0.091 Recall@10 above the lexical-only configuration that was
previously the best available. The graph lane is inert on this track (zero
candidates in all 500 questions; session memories carry no links), so these
numbers measure the embedder change alone, and the run predating the graph
weight change is unaffected by it.

These remain retrieval-only numbers over ~50-candidate haystacks per question
and are still not comparable with published end-to-end LongMemEval QA
accuracy.

## What this closes and what it opens

- Follow-up 1 is answered in substance: dense and fusion-weight conclusions
  are now drawn from a semantic model. The Titan (`bedrock`) rerun remains
  worthwhile before an AWS deployment, and every calibrated threshold must be
  re-measured if the model changes — the floors above are Qwen-specific.
- The fusion-weight part of follow-up 3 is closed by measurement; the
  remaining graph sweeps (query-gate floor, hops, fan-out, seed limits) are
  still open, as is per-query gating of the lane for the purposes that keep
  higher weights.
- The `paraphrase` intent is still the weakest (0.500 / 0.600); its two
  hardest queries need either query rewriting or a stronger embedder, and are
  the natural first targets for a reranker gate.

Environment for this addendum: repository `swarm-brain` at `b9409d7` plus the
uncommitted changes described here, Python 3.13 arm64 client, vLLM 0.26.0
serving `Qwen/Qwen3-Embedding-0.6B` on RTX 5070 Ti (CUDA), in-memory kernel
for both tracks, same corpus `swarm-coding-2026-08-07` and judgments `r1`, and
the official LongMemEval-S file with the pinned SHA-256 — no query, judgment
or corpus memory was added, removed or edited.


# Addendum, 2026-08-09 (second) — relevance reranking: measured, not shipped

The addendum above closed with the `paraphrase` intent as the weakest on the
swarm corpus and named a reranker gate as its natural first target. This
addendum reports that stage and the decision it produced: **it is implemented,
tested and disabled by default**, because the independent 500-question track
did not reproduce the swarm corpus's gain. Same corpora, same judgments `r1`,
same evaluator; no query or corpus memory added, removed or edited.

Nothing about the shipped configuration changed. This section exists so the
experiment is not silently repeated.

## The hypothesis and the mechanism

Weighted RRF scores *agreement*: a candidate that several lanes each rank in
the middle outranks one that a single strong lane ranks well. That is the right
prior when lane scores are not comparable — which is why RRF is the
calibration-free baseline — and it should become the wrong one once a lane can
state calibrated relevance.

The failure looked real per query. On all three `paraphrase` queries that missed
evidence, one gold memory was found at rank 1 and the second was found *only* by
the dense lane, at ranks 7, 12 and 26. RRF placed those second memories at ranks
11, 15 and 18 — just outside `k = 10` — while the top ten filled with memories
that several lanes ranked mid-field and no lane could defend. The quantity that
separates the two was already being computed: the rank-independent relevance
built for the abstention floor, used only as a gate and never for ordering.

`swarmbrain.retrieval.fusion.relevance_reranked` reorders the head of the fused
ranking by `(1 - alpha) * rrf + alpha * relevance`, with the RRF term as
`raw_rrf` over the largest `raw_rrf` in the window. That normalisation is
strictly monotone in fused order, so `alpha = 0` reproduces weighted RRF exactly
and the stage is a true no-op when disabled — which is how it ships. The window
is bounded (`4 x limit`, floored at 32); candidates past it keep their fused
position. Graph-only candidates score zero relevance by construction and so can
never be promoted by it. Latency is unchanged, because the whole fused list is
already hydrated before relevance is computed: swarm-track wall p50
32.97 → 29.82 ms, p95 35.87 → 36.57 ms.

Every saved run carries its own before/after for the stage: the `fused` lane is
plain weighted RRF and `final` is the bundle after reranking. All deltas below
are that within-run comparison, so they are isolated from the embedder and
graph-weight changes of the previous addendum.

## Track 1 — swarm corpus: a clear win

At `alpha = 0.5`, k = 10, in-memory:

| Embedder configuration | R@10 RRF → reranked | MRR@10 RRF → reranked | nDCG@10 RRF → reranked |
| --- | --- | --- | --- |
| no dense lane | 0.902 → 0.927 (+0.025) | 0.804 → 0.824 (+0.020) | 0.811 → 0.839 (+0.028) |
| hash embedder | 0.873 → 0.887 (+0.015) | 0.807 → 0.826 (+0.019) | 0.802 → 0.821 (+0.019) |
| Qwen3-0.6B | 0.927 → 0.941 (+0.015) | 0.912 → 0.927 (+0.015) | 0.882 → 0.903 (+0.021) |

Positive on all three aggregate metrics in all three lane configurations, and
largest where fusion has least evidence to work with. Per intent with
Qwen3-0.6B, every intent that moved: `paraphrase` 0.500 / 0.600 → 0.600 / 0.700,
`decoy_heavy` MRR 0.875 → 1.000, `multi_evidence` MRR 0.833 → 0.667. No intent
lost recall. The one MRR regression is honest and understood: on
`multi-duplicate-charge-thread` a payments handoff memory scores 0.625 lexical
coverage against the gold memory's 0.600 and takes rank 1, because unweighted
token coverage rewards summary-style memories that mention many query terms.

## Track 2 — LongMemEval-S, all 500 questions: a loss

The same stage, the same `alpha`, on the independent track:

| Slice | n | R@10 RRF → reranked | MRR@10 RRF → reranked |
| --- | --- | --- | --- |
| **overall** | 500 | **0.976 → 0.971 (−0.006)** | **0.908 → 0.905 (−0.002)** |
| single-session-assistant | 56 | 1.000 → 1.000 | 0.916 → 0.991 (+0.075) |
| single-session-user | 70 | 1.000 → 0.986 (−0.014) | 0.866 → 0.925 (+0.060) |
| temporal-reasoning | 133 | 0.967 → 0.967 | 0.906 → 0.887 (−0.019) |
| multi-session | 133 | 0.955 → 0.952 (−0.002) | 0.941 → 0.917 (−0.023) |
| knowledge-update | 78 | 0.994 → 0.987 (−0.006) | 0.981 → 0.975 (−0.005) |
| single-session-preference | 30 | 0.967 → 0.933 (−0.033) | 0.662 → 0.546 (−0.116) |
| abstention slice | 30 | 0.950 → 0.931 (−0.019) | 0.919 → 0.856 (−0.064) |

Five of six question types regress on recall or reciprocal rank, and only
`single-session-assistant` gains outright. No-answer precision and recall stay
1.00 / 1.00 on both sides, so the abstention channel is unaffected either way.

## Why the two tracks disagree

The first hypothesis — that the relevance signal is weak on session-sized
memories — is measurably **wrong**. Relevance separates gold from non-gold about
equally well on both: mean relevance 0.648 for gold against 0.376 for non-gold
on LongMemEval (+0.271), versus 0.771 against 0.477 on the swarm corpus
(+0.294).

The difference is the number of lanes actually returning candidates. Reranking
repairs a *consensus* pathology, and consensus noise scales with lane count. The
swarm corpus fires all five lanes, so mid-ranked junk accumulates and single-lane
finds get buried — the mechanism the stage was built for. On LongMemEval only
lexical and dense return anything at all: its session memories carry no
identifiers, no `related_to` links, and nothing for exact, fuzzy or graph to
match. With two lanes there is little consensus noise to repair, fused Recall@10
is already 0.976, and reordering can mostly only push gold that RRF held at
ranks 8–10 out of the bundle.

Gating the stage on lane count is the obvious next hypothesis. It is
deliberately not implemented: there is no third judged corpus to validate such a
gate on, and deriving it from the two datasets that motivated it is how a
retriever gets fitted to its own benchmark.

## The decision, and the size of the evidence

`RetrievalPlanner._rerank_alpha` returns 0.0 for every purpose. The stage stays
in the tree with its unit tests because the finding is worth keeping and the
`alpha = 0` path is provably a no-op, not because it is half-enabled.

Two facts set the burden of proof against shipping it:

- **The swarm gain is roughly one query.** With 34 answerable queries, one query
  is worth about 0.03 Recall@10. Repeating the identical Qwen configuration
  across runs moves fused Recall@10 by 0.015 (0.9265 vs 0.9118) purely from
  served-embedding nondeterminism — the same magnitude as the measured gain. The
  within-run A/B is exact, but the corpus cannot resolve an effect this small
  against its own noise floor.
- **The LongMemEval loss is consistent**, spread across five of six question
  types and 500 questions with a deterministic lexical lane.

For the same reason `alpha` was never moved to its argmax. On the swarm corpus
`alpha` in 0.6–0.8 scores a further +0.015 Recall@10; adopting it on that
evidence would be choosing the argmax of 34 queries, which
[the evaluation protocol](retrieval-evaluation.md) forbids.

The rejected run is kept as evidence in
`benchmarks/retrieval/longmemeval-s-rerank-experiment-{run,report}.json`. Its
`fused` lane is bit-identical to the shipped configuration's `final` lane, which
is what makes it a clean A/B rather than two separate runs.

## What did ship from this round

Two measurement capabilities, both in every report from now on:

- **`precision_at_k` per lane.** On the swarm corpus it is ceiling-bounded and
  says more about the judgments than the retriever: 34 answerable queries carry
  1.76 relevant memories on average, so no ranking can exceed P@10 ≈ 0.176, and
  the final lane reaches 0.162 — 92% of a ceiling set by the judgments rather
  than by the retriever.
- **`bundle_by_floor`** — what the caller is actually handed once the relevance
  floor applies, which is what a token budget is spent on. Shipped configuration,
  swarm corpus, k = 10:

| Floor | mean bundle | bundle precision | answerable recall | abstentions | no-answer P / R |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 10.00 | 0.162 | 0.926 | 0 | 1.00 / 0.00 |
| 0.40 | 8.06 | 0.239 | 0.912 | 3 | 1.00 / 0.50 |
| 0.50 | 4.65 | 0.418 | 0.843 | 6 | 0.83 / 0.83 |
| 0.60 | 2.56 | 0.535 | 0.770 | 11 | 0.55 / 1.00 |

Floor 0.40 is the setting that pays for itself: bundle size drops 19% and
precision rises 48% for 0.014 of answerable recall, while half the no-answer
queries begin abstaining correctly. These floors are Qwen-specific and must be
re-measured if the embedding model changes.

The absolute values in this table move by roughly ±0.015 recall between runs of
the identical configuration, for the served-embedding reason given above. The
shape — the size/precision trade at each floor — is stable across runs; the
third decimal is not, and should not be quoted as if it were.

`final_relevance` is now recorded on **both** tracks, so any saved run can be
replayed at any floor without re-executing it. The LongMemEval canonical report
predates that change and therefore carries no `bundle_by_floor` yet; producing
one costs a four-hour re-run and is worth folding into the next LongMemEval
measurement rather than doing on its own.

# Addendum, 2026-08-09 (third) — answer-in-context under a token budget

Recall@k asks whether the evidence was ranked. A reader does not consume a
ranking, it consumes a context window, and everything past the budget is not
lower-ranked — it is absent. This addendum adds the metric that survives that
truncation and reports it on both tracks. It changes no retrieval behaviour:
`RetrievalPlan.token_budget` remains unset by default, so packing is a no-op
unless a caller opts in.

## Definitions

`swarmbrain.retrieval.packing` packs a bundle in published rank order until the
budget is exhausted, and scores two rates over answerable cases:

- **any-gold-in-context** — at least one memory carrying the answer survived.
  This is the ceiling on what any reader can get right.
- **all-gold-in-context** — *every* gold memory survived. A question needing
  three sessions is not answerable from one of them, so this is the honest
  ceiling for multi-evidence questions.

An item that does not fit is **skipped, and packing continues** (`greedy`),
rather than terminating the pack at the first overflow (`prefix`). Both policies
are implemented and both are reported below, because the choice turns out to be
worth real accuracy.

**The historical token counts below are development-only estimates.** They use
`estimate_tokens`, a documented characters-over-four proxy floored at one.
They are stable for comparisons but are not cost figures and cannot pass the
SOTA gate. Identifier- and JSON-dense text tokenises worse than 4 chars/token,
so these budgets are optimistic. Every number in this section remains tied to
that estimator and must not be relabelled as an exact measurement.

The current harness also has a publishable exact path without adding a runtime
tokenizer dependency. A repository-local JSONL tokenizer executable loads a
repository-local tokenizer artifact; the operator pins both SHA-256 digests,
the tokenizer model, and its immutable revision. For every packing proposal the
harness serializes the complete official reader prompt (including instruction,
chronologically ordered and renumbered sessions, current date, and question),
asks the pinned local tokenizer for its exact count, and records only the prompt
SHA-256, count, local request ID, and unique provider request ID. Each exact case
also preserves the public benchmark question/date and its ranked top-10 session
role/content records once; it does not duplicate every expanded prompt. The
report compiler rebuilds every initial, proposed, and final prompt from that
material, verifies each SHA-256, rejects inconsistent request/state reuse, and
exactly reconciles character and UTF-8 byte totals. It also verifies the
executable/artifact paths, bytes and hashes, complete 5/10-by-budget greedy
lineage, contiguous provider-observed accounting, and shared serializer
fingerprint. Partial session fragments cannot set `exact_model_tokenizer=true`.
Non-empty prompts are byte-for-byte official. The reference generator has no
empty-session form (it asserts a non-empty history), so hypothetical empty
packing states use the separately disclosed Swarm Brain empty-context note.

This makes an exact full-500 run materially larger than a development run and
embeds public LongMemEval text in the artifact. That replayability is deliberate
for this public benchmark; do not enable exact prompt material on a private
corpus without an explicit data-handling decision.

## Track 1 — swarm corpus: the budget never binds

The full ten-hit bundle averages **608 estimated tokens**. Every budget from
32k down to 2k keeps all ten hits, and any-gold-in-context is 0.971 with
all-gold 0.882 at every one of them. The metric is inert here, which is itself
the finding: for short swarm memories, ranking quality is the whole story and a
context budget is not a constraint worth modelling.

## Track 2 — LongMemEval-S: the budget is the story

Each hit is an entire conversation session. All 500 questions, k = 10, greedy:

| Budget | mean hits kept | mean tokens | any-gold | all-gold | truncated |
| --- | --- | --- | --- | --- | --- |
| none | 10.00 | 32 750 | 0.994 | 0.952 | 0 |
| 32 000 | 9.23 | 29 996 | 0.994 | 0.940 | 302 |
| 16 000 | 4.83 | 15 139 | 0.960 | 0.822 | 500 |
| 8 000 | 2.46 | 7 224 | 0.918 | 0.584 | 500 |
| 4 000 | 1.32 | 3 394 | 0.722 | 0.222 | 500 |
| 2 000 | 0.79 | 1 009 | 0.118 | 0.094 | 500 |

Four things this says that Recall@10 = 0.976 does not:

1. **The unbounded bundle costs about 32 750 tokens**, not the 20–25k assumed
   when the QA workstream's cost envelope was drafted — roughly 30% higher, and
   worth re-checking before committing to a per-run budget.
2. **32k is free.** It truncates 302 of 500 questions and costs *nothing* in
   any-gold and 0.012 in all-gold. Any budget above that is paying for context
   no question needs.
3. **16k costs 3.4 points of ceiling** (0.994 → 0.960) for half the tokens. That
   is the interesting operating point, and all-gold is what pays for it
   (0.952 → 0.822), because multi-evidence questions lose their second and third
   sessions first.
4. **Below 8k it collapses.** At 4k a single session often does not fit; at 2k
   the mean bundle holds 0.79 hits and any-gold is 0.118. There is no clever
   packing at that budget, only a smaller `k` and shorter memories.

### The packing policy is worth measuring, not assuming

Greedy skip-and-continue against prefix stop-at-first-overflow, any-gold:

| Budget | greedy | prefix | delta |
| --- | --- | --- | --- |
| 16 000 | 0.960 | 0.956 | +0.004 |
| 8 000 | 0.918 | 0.912 | +0.006 |
| 4 000 | 0.722 | 0.638 | **+0.084** |
| 2 000 | 0.118 | 0.094 | +0.024 |

At generous budgets the policies agree. At 4k, skipping one oversized session
and admitting the smaller ones behind it recovers 8.4 points of ceiling — the
naive renderer throws that away.

### Per question type, any-gold-in-context

| Question type | n | none | 32k | 16k | 8k | 4k |
| --- | --- | --- | --- | --- | --- | --- |
| knowledge-update | 78 | 1.000 | 1.000 | 1.000 | 1.000 | 0.859 |
| single-session-assistant | 56 | 1.000 | 1.000 | 1.000 | 1.000 | 0.964 |
| multi-session | 133 | 0.992 | 0.992 | 0.970 | 0.932 | 0.782 |
| temporal-reasoning | 133 | 0.992 | 0.992 | 0.962 | 0.910 | 0.692 |
| single-session-user | 70 | 1.000 | 1.000 | 0.957 | 0.886 | 0.586 |
| single-session-preference | 30 | 0.967 | 0.967 | 0.733 | 0.600 | 0.100 |

`single-session-preference` degrades first and hardest — 0.967 → 0.733 at 16k,
where every other type is still above 0.95. It is also the type with the weakest
MRR (0.662): its evidence is ranked late, so it is the first thing a budget
evicts. If a token budget is ever applied, that slice is where it will show up.

## Reproducing

The canonical LongMemEval run predates per-hit token recording, so its
`final_tokens` were backfilled by joining the saved rankings to the pinned
dataset and rendering each session exactly as the harness does. The join is
deterministic and the lane metrics were asserted unchanged across the
regeneration; `final_tokens` is now recorded natively by
`scripts/run_retrieval_eval.py` on both tracks, so the next run produces
`answer_in_context` without the backfill.
