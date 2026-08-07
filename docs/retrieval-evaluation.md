# Retrieval v2 evaluation

Retrieval changes are evaluated at two separate boundaries:

1. Candidate quality: exact, lexical, fuzzy, dense, graph, direct-fused, and
   final-fused rankings are compared with human relevance sets using Recall@k,
   MRR@k, and nDCG@k.
2. ANN quality: the CockroachDB vector-index ranking is compared with the same
   canonical eligibility policy forced through `retrieval_vectors_1024@primary`,
   using ANN Recall@k. The ANN path uses prefix filtering plus same-snapshot
   canonical validation and adaptive widening; the exact oracle filters the
   eligible relation before its exact vector sort.

`EXPLAIN` is necessary to prove index selection, but it cannot prove ANN recall.
Likewise, ANN Recall@k cannot prove end-to-end relevance or no-answer behavior.
Both gates are required.

## Saved-run format

The evaluator accepts one JSON object with a `cases` array:

```json
{
  "corpus_version": "coding-memory-2026-08-06",
  "cases": [
    {
      "case_id": "semantic-lease-fencing",
      "relevant_ids": ["memory-a"],
      "rankings": {
        "exact": [],
        "lexical": [],
        "dense": ["memory-a", "memory-b"],
        "graph": ["memory-c"],
        "direct_fused": ["memory-a", "memory-b"],
        "fused": ["memory-a", "memory-c"]
      }
    }
  ]
}
```

An empty `relevant_ids` set is an explicit no-answer case. An empty ranking is
an abstention. IDs may be opaque fixture IDs; production UUID syntax is not
required in an offline run.

Run:

```bash
uv run python scripts/evaluate_retrieval_runs.py \
  tests/fixtures/retrieval_v2/sample_runs.json --k 10
```

The checked-in sample only proves evaluator determinism. It is not a quality
benchmark and must never be cited as evidence that the retriever is SOTA.

## Required report for a model, weight, or index change

Record:

- corpus version and relevance-judgment revision;
- embedding provider, model, dimensions, normalization, truncation, and input
  renderer signature;
- exact/lexical/fuzzy/dense/graph/direct-fused/final-fused metrics at the
  product context depths;
- no-answer precision and recall;
- CockroachDB version, corpus/scope sizes, selectivity buckets, beam and
  partition settings, candidate overfetch, and ANN Recall@k versus exact;
- p50/p95/p99 latency, CPU, rows/bytes read, projection freshness lag, and
  storage per vector;
- regressions split by query intent: identifiers, code lookup, conceptual,
  task bootstrap, temporal, contradiction, associative/multi-hop, and
  multi-evidence;
- graph hop/seed/fan-out/edge budgets, relation weights, query-gate floor,
  truncation/underfill rate, and path-length buckets.

RRF remains the calibration-free baseline. Do not change its constant or lane
weights from a single aggregate score: compare per-intent ablations and retain
no-answer/security gates. A learned convex fusion or reranker is eligible only
after enough judged cases exist to tune it without evaluating on its training
set.

## Live CockroachDB gate

For each representative scope/selectivity bucket:

1. build the same `RetrievalPlan` and `DenseQuery`;
2. run `CockroachDenseRetrievalGateway.retrieve()` for ANN;
3. run `retrieve_exact()` inside the same canonical snapshot;
4. compute `ann_recall_at_k()` at the candidate depths used by fusion;
5. capture `EXPLAIN ANALYZE` for the ANN query and latency/resource metrics.

Historical semantic recall is not evaluated through this current-only ANN
projection. It requires a bounded bitemporal prefilter followed by exact vector
ranking, or a separately signed historical projection.
