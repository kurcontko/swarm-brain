# Chain-of-Memory E2 organizer

This package is a pure, benchmark-local implementation of the frozen E2
organization cells in
`docs/research/sota-selection-reranker-protocol-2026-08-09.md`. It does not
change production retrieval.

The input is exactly 20 ordered `ChainCandidate` values backed by the immutable
LongMemEval turn compiler. Query cosines and candidate-to-concatenated-chain
cosines are external evidence. Product cells accept either an immutable table
whose keys bind the exact chain prefix and candidate document, or a
caller-attested deterministic callback. The organizer does not load or execute
an embedding model.

Frozen cells:

- E2-A: preserve retrieval order; no chain evolution.
- E2-B: query-only successor score with adaptive path truncation at beta 0.5.
- E2-C: query/context product score without adaptive path truncation.
- E2-D: product score with adaptive path truncation at beta 0.5.
- E2-E: E2-D evolution with cross-chain duplicates removed only while rendering.

For the primary transfer cells, the input is the fixed E1-A RRF top 20 and its
first three candidates are anchors. Anchor membership is not reselected by
query cosine. Completed chain blocks are ordered by anchor query cosine, with
retrieval rank and canonical turn ID as stable ties. The separate paper-text
diagnostic begins with a global query-cosine top 20, so its first three inputs
are also its top three query-cosine candidates; that does not change the
primary RRF-head rule.

A candidate is removed only from the chain currently being built, so it can
occur in other chains. Each chain is bounded by the 20-turn pool; the whole run
is bounded by 57 decisions and 570 context-score evaluations. The parity
rendering ceiling is 60 turn instances.

`content_free_artifact()` includes exact turn IDs, source/document digests,
every evaluated scorecard, chosen successors, chain order, rendered order, and
an overall trace digest without including question, answer, or turn text.

This implementation is paper-inspired transfer evidence. It does not claim an
exact Chain-of-Memory reproduction, validate the external embedding execution,
or establish any QA improvement. Those claims require separately compiled,
same-protocol end-to-end evidence.
