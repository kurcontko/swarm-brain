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
| Source | `https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s` |
| SHA-256 | `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894` |
| Questions evaluated | 500 of 500 (no sampling) |
| Haystack sessions per question | 39 – 66, median 50 |
| Backend | in-memory kernel only |

The 278 MB release is never committed. `scripts/run_retrieval_eval.py`
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
   leakage between questions and no shared corpus.
3. **Memory shape.** `kind=observation`, `state=confirmed`,
   `visibility=repository`, content = the session rendered as
   `role: content` lines, title = `Conversation session recorded <haystack date>`,
   tags `("longmemeval", "session")`. Session identifiers are not put in the
   searchable text.
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

### Results, all 500 questions

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

# Track 2, all 500 LongMemEval-S questions (first run downloads 278 MB)
uv run --extra dev python scripts/run_retrieval_eval.py \
  --track longmemeval --lme-sample 0 --lme-download

# Score any saved run directly
uv run --extra dev python scripts/evaluate_retrieval_runs.py \
  benchmarks/retrieval/swarm-native-cockroach-run.json --k 10
```

Saved runs and reports live in `benchmarks/retrieval/`, not in `evidence/`.
`evidence/` holds timestamped, append-only demo artifacts; a benchmark is a
versioned artifact that should be overwritten in place so a rerun produces a
reviewable diff against the previous measurement.

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
