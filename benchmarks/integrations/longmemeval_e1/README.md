# LongMemEval E1-B/E1-C/E1-D selection

This package implements three **pure, evaluation-only, SmartSearch-shaped**
selection cells on the exact unique candidate pool emitted by the frozen
LongMemEval E1-A turn fusion. It performs no model, embedding, database, or
network calls.

## Frozen v1 cells

- **E1-B:** apply a numerically stable sigmoid to every externally supplied raw
  CrossEncoder logit, then rank every E1-A candidate by that probability.
- **E1-C:** independently rank CrossEncoder probabilities and ColBERT scores,
  then fuse the two ranks as `0.7/(60 + ce_rank) + 0.3/(60 + colbert_rank)`.
- **E1-D:** take the E1-C top 60, then keep candidates whose CrossEncoder
  probability is `>= 0.03 * max_probability`, where the maximum is computed
  only over that E1-C top-60 preselection head. This denominator scope follows
  the paper's stated operation order.

All ties use the prior E1-A fused rank and then the canonical turn ID. The
reserved `swarmbrain-longmemeval-smartsearch-shaped-e1-v1` identifier rejects
any changed pool cap, RRF constant, weights, head size, threshold, or upstream
E1-A version. A changed study must register a new version explicitly.

## Evidence boundary

Call `bind_e1a_pool()` first and copy its question, exact-query digest, turn
corpus digest, E1-A trace digest, pool digest, and count into each
`PoolScoreObservation`. Each channel must then contain an exact one-to-one
permutation of the fixed pool. Missing, extra, duplicate, non-finite, stale,
cross-question, or differently bound observations fail closed.

The scorer identity records the caller-attested producer, scorer, model,
revision, model-artifact digest, and observation-artifact digest. This package
does **not** reopen those artifacts, verify that the claimed model produced the
scores, or recompute scores. Its trace says so explicitly.

The returned candidates contain hydrated immutable turns. The exported trace
contains IDs, numeric score/rank facts, byte/digest bindings, and decisions,
but no query, answer, or turn text. Each export is a fresh object, so caller
mutation cannot alter the frozen result or its digest.

An `E1SelectionResult` remains a transport value rather than an authentication
token. Any downstream compiler that relies on its order must call
`validate_e1b_result(...)` with the original E1-A result and CrossEncoder
observation. That boundary deterministically rebuilds E1-B and requires exact
object and trace equality, so a caller cannot reorder candidates and merely
patch the self-consistent output digest.

These cells do not implement prompt packing and do not execute a reader or
judge. They prove neither a SmartSearch model reproduction nor a QA gain; those
claims require pinned scorer execution plus downstream held-out evaluation.
