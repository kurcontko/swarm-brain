# E6/SB-HMR-v1 representation controls

This package implements the paper-derived, benchmark-only representation
ablation frozen in the 2026-08-09 representation-policy audit. It is pure
offline validation, fusion, hydration, and tracing. It never extracts a key,
embeds or scores text, calls a model/network, answers a question, judges an
answer, or alters production serving.

The cells are Swarm Brain transfer hypotheses, not UnifiedMem or Memora
reproductions:

- R0: raw value key only.
- R1: raw plus exactly one merged summary/fact/keyword key per source.
- R2: raw plus separately ranked summary, fact, and keyword families.
- R3: one primary abstraction plus cue anchors, with no raw retrieval key.
  `R3+raw` is intentionally not smuggled into this protocol; it requires a
  separately named diagnostic.
- R4: descriptive entity activation with no adjacency access.
- R5: the byte-identical R4 index/ranking observation plus one-hop entity
  adjacency and a deterministic post-expansion ordering.
- R-neg: one-hop similarity-value expansion, permanently marked ineligible for
  serving promotion by itself.

Every cell returns canonical raw source values. Derived keys are navigation
metadata and never reader evidence.

## Source and extraction boundary

`CanonicalValue` uses the immutable F0 `TurnProjection` serialization. It binds
the dataset artifact, projection, canonical turn/source ID, source-turn version
hash, raw document hash, and exact UTF-8 bytes. Choosing F0 turns—not sessions
or extracted records—is an **SB-HMR-v1 hypothesis**, because the papers do not
freeze Swarm Brain's hydrated value granularity.

`RepresentationCorpus` must also receive the authoritative F0
`TurnProjectionCorpus` and match its complete, ordered question-local turn
slice exactly. A self-consistent digest over a missing, reordered, or forged
subset is insufficient.

Every derived key repeats the exact source/value bindings and binds its own
text hash/bytes and construction receipt. Each complete receipt binds extractor
protocol, model, immutable revision, deployment, model/prompt/identity
artifacts, request/response/construction artifacts, usage, latency, cost,
retry/cache state, and exact output key IDs/hashes. These identities are
caller-attested and explicitly unverified by this offline module.

Navigation/provenance identifiers are generated locally as typed opaque
SHA-256 IDs (`key:`, `entity:`, `receipt:`, and `edge:`). Human-readable entity
names, key text, provider request strings, or arbitrary caller IDs cannot use
those trace fields as a covert content channel.

Extraction has a hard source-only input contract. The request material digest
is recomputed from the raw value hash/bytes, requested family, extractor
identity, and prompt hash. The only allowed field is `source_value`.
`question`, `question_type`, answers, gold sessions, and judge labels cannot be
declared extractor inputs. `question_id` exists only as local routing metadata
and is excluded from request material.

Graph construction has the parallel hard contract: its only declared input is
a source-safe navigation-index digest plus the graph identity/prompt. That
digest covers immutable raw value hashes/bytes and derived navigation-key
bindings, but deliberately excludes audit-only source-artifact, projection,
question-record, and construction-receipt metadata that may change with gold
annotations. The graph remains separately bound to the full representation
index for audit integrity. Question text/type, answers, gold locations, and
judge labels are forbidden construction inputs. Ranking is different by
design—the query hash belongs to query-time scoring, never to offline key or
graph construction.

## Ranking and fan-out control

The audit specifies weighted RRF but not its lane unit, family depth, or
same-family fan-out semantics. SB-HMR-v1 freezes these hypotheses:

- each key family is one equal-weight lane;
- `k=60`, exhaustive scoring of the indexed family, and returned top depth 20;
- after mapping keys to canonical values, only the best rank in a given
  `(family, value)` contributes, so extra facts/cues cannot inflate score;
- different families may each contribute once;
- ties use best prior key rank and canonical value ID;
- the full question-local source corpus is indexed before retrieval, with a
  high 16,384-value safety ceiling; only the hydrated result is capped at 128.

All ceilings (`16,384` source values, `524,288` derived keys, `32` keys per
value/family, depth `20`, returned value cap `128`) are defensive SB-HMR-v1
protocol choices, not paper constants.

Entity descriptions form an entity-to-value binding. R4 ranks entity keys and
hydrates all values bound to directly activated entities; it rejects any graph.
R5 reuses that exact observation, treats adjacency as undirected canonical
pairs, expands only one hop, and never recurses. Hydrated values are ordered by
best seed entity raw score, distinct seed support count, recomputed R4 direct
rank (expanded-only last), then canonical value ID. Direct membership counts as
one support witness. This propagation/support definition, degree 20, and all
tie rules are explicit SB hypotheses.

An entity edge's value provenance must be exact and collectively cover both
endpoints through the corpus's entity-description-to-value bindings; every
declared evidence value must support at least one endpoint. This endpoint
coverage rule is an SB-HMR-v1 provenance hypothesis and prevents an unrelated
source value from legitimizing a digest-consistent forged edge.

R-neg seeds from R0 top-20 raw values. Canonical undirected similarity edges
must score at least `.80`; one-hop activation is `seed RRF × edge similarity`.
It uses distinct seed support, direct rank, and value ID as later tie-breaks.
Its seed choice, threshold, direction, propagation, and ordering are all SB
hypotheses retained only as a bounded negative control.

## Evidence and non-claims

Family observations must be complete, exact-depth, score-ordered, immutable
receipts bound to one query, index, source artifact, projection, scorer, and
observation artifact. Graphs bind the same index/source/projection, construction
identity/accounting, canonical edge order, degree/hop bounds, and exact value
version/hash provenance. Unknown, orphaned, duplicate, cross-question,
cross-source, stale, tampered, or partial material fails closed.

The content-free trace reports key hits and unique hydrated values, per-family
fan-out suppression, derived objects per source, index/value/key UTF-8 bytes,
duplicate/orphan counts, construction tokens/latency/cost and identities,
graph witnesses, exact RRF contributions, and hydrated raw hashes. Static
R0-R5 controls have zero updates; consolidation is a later experiment. Index
token counts are deliberately `null` until supplied by a separate exact
tokenizer receipt—bytes are never converted into estimated tokens.

No trace or result claims paper parity, quality improvement, or serving
eligibility. End-to-end promotion still requires the separately frozen reader,
8,192-token exact prompt, judge, paired report, and held-out evidence.
