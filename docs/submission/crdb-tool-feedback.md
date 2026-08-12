# Feedback on CockroachDB's AI and retrieval tooling

Written 2026-08-07 from building **Swarm Brain**, a coordination and
temporal-memory kernel for a swarm of heterogeneous coding agents, on
CockroachDB 26.2.1. The database holds the entire state machine: task leases,
fencing tokens, idempotency records, bitemporal memory with supersession
lineage, evidence, a transactional outbox, and a five-lane hybrid retriever
(exact identifiers, full text, trigram, dense ANN, bounded graph) fused with
weighted RRF.

Every point below is something we actually hit while writing that code, with
the file it forced. We have tried to separate *CockroachDB behaviour* from
*our design choice made because of that behaviour*, because conflating the two
makes feedback useless.

Short version of the verdict: **`SERIALIZABLE` by default is the reason this
project exists and it delivered.** Most of the friction is in the vector and
retrieval surface, where the escape hatches exist but the ergonomics push work
into the application that the server is better placed to own.

---

## 1. Vector indexes accelerate only the complete equality prefix

**What we did.** Our authorization model is a scope hierarchy — tenant →
project → repository → repository/run/task visibility. We wanted one ANN query
with a visibility predicate. We could not have one: the vector index only
accelerates predicates that bind *every* prefix column with equality (or a full
tuple `IN`). A range predicate on any prefix column drops the acceleration
entirely.

So the dense gateway issues **one fully prefix-bound ANN branch per authorized
scope**, then validates lifecycle, trust, bitemporal, version, and digest
predicates against the canonical `memories` rows in the same snapshot
(`adapters/cockroach/retrieval.py`, `CockroachDenseRetrievalGateway`). Our
prefix is seven columns wide — well past the one-to-three the docs recommend —
purely to keep the query auth-safe.

**Rough or great.** Rough, and it is the single biggest architectural
constraint we hit. The rule is documented and consistent, which we appreciate;
but "correct authorization" and "index acceleration" pull in opposite
directions, and a wide prefix is the only way to satisfy both.

**What we'd ask for.** Post-filter-aware ANN: let the index accelerate a
prefix-bound scan while a non-prefix predicate is evaluated during traversal
rather than after it, even if only for predicates over columns stored in the
index. Failing that, a documented pattern for "ANN under row-level
authorization" would save every serious adopter the same week we spent.

## 2. Filtered ANN under-fill is the application's problem

**What we did.** When canonical filtering rejects most ANN candidates, the lane
under-fills. We hand-rolled an iterative filtered scan: re-run the ANN branch
with a geometrically doubled window (capped at 2000) until the validated set
reaches the lane budget or the window stops saturating. That loop is thirty
lines of retrieval-critical control flow in `_retrieve_scope_ann`.

**Rough or great.** Rough. pgvector 0.8 ships iterative scans as a server
feature; here it lives in every application that filters ANN results — and
everyone will implement it slightly differently, with slightly different recall.

**What we'd ask for.** A server-side equivalent: an ANN scan that keeps
descending until it yields *k* rows surviving the query's own filters, bounded
by something like `vector_search_max_iterations`. This is the single highest-value
addition we can name.

## 3. `SET vector_search_beam_size` will not take a bound parameter

**What we did.** We render the integer into the SQL string:

```python
await connection.execute(f"SET LOCAL vector_search_beam_size = {self.beam_size}")
```

with a `1 <= beam_size <= 1024` range check at construction, because psycopg
sends a bound value as a string and CockroachDB rejects it.

**Rough or great.** Rough, and needlessly so — it forces string interpolation
into SQL on a *tuning knob*, which is exactly the shape every static analyzer
and security reviewer flags. What makes it feel like an oversight rather than a
design is the asymmetry: two hundred lines earlier in the same file,

```python
await connection.execute("SET LOCAL pg_trgm.similarity_threshold = %s", (threshold,))
```

works fine.

**What we'd ask for.** Accept `SET LOCAL vector_search_beam_size = $1` with an
integer-typed parameter, or coerce a numeric string. Consistency with the other
session settings would be enough.

## 4. `EXPLAIN` proves index selection but says nothing about ANN recall

**What we did.** We gate CI on live `EXPLAIN` over the exact SQL the runtime
emits — FTS, trigram, ANN, and both graph directions — because index selection
is a real regression risk. But beam size is a *quality* parameter, not a
correctness constant, and no plan output tells you what the beam missed. So we
built an exact-vector oracle: the same canonical eligibility policy forced
through `retrieval_vectors_1024@primary`, filtering the eligible relation before
an exact vector sort, and we compute ANN Recall@k against it.

**Rough or great.** `EXPLAIN` itself is great — trustworthy and stable enough
that we assert on it in tests. The gap is that recall is unobservable without
running a second, deliberately slow query.

**What we'd ask for.** ANN diagnostics in `EXPLAIN ANALYZE`: partitions
visited, candidates examined, candidates rejected by the prefix, and whether
the search bottomed out on the beam. That would let operators tune beam size
from production traffic instead of from an offline oracle.

## 5. Adding a vector index to an existing table is a planned migration

**What we did.** Our compatibility backfill copies existing vectors into the
new signed projection *before* `CREATE VECTOR INDEX` runs, because large batch
inserts into a table that already carries a vector index are discouraged and
`IMPORT INTO` does not work on such a table at all. The DDL file carries that
ordering as a comment so nobody reorders it.

**Rough or great.** The constraint is understandable, and ordering the DDL
correctly is cheap. What is not cheap is that this makes "add a vector index to
an existing large table" an operator-planned, offline-shaped migration rather
than an online DDL — for a feature whose main adoption path is exactly that.

Related, and more annoying: the vector-index documentation still warns about
writes being blocked during backfill while the release notes read more
optimistically. We designed for the pessimistic case because we could not tell
which was current.

**What we'd ask for.** (a) A supported bulk-load path for vector-indexed
tables. (b) One authoritative, version-stamped statement of backfill write
behaviour, so adopters do not have to design for the worst reading.

## 6. `VECTOR(n)` fixes a constant that propagates through the whole application

**What we did.** `COCKROACH_VECTOR_DIMENSIONS = 1024` appears as a Python
constant, as a `CHECK (dimensions = 1024)` in two tables, as a validation in
the memory store, and as the `dimensions` argument to the Bedrock Titan V2 call.
Everything agrees by construction, which is good — but it also means the number
is welded into six places.

**Rough or great.** Fine as a design, awkward in practice. Matryoshka-style
dimension truncation (Titan V2 supports 256/512/1024) means a new table and a
new index, not a cast. And the opclass menu is narrow — cosine, L2, inner
product, with no L1, Hamming, or Jaccard — which rules out binary/quantized
embedding strategies entirely.

**What we'd ask for.** Dimension-narrowing casts between `VECTOR(n)` widths,
and Hamming/Jaccard opclasses for binary embeddings. Index recommendations for
vector columns (the way they exist for other indexes) would also help teams who
do not know a prefix is mandatory until their first slow query.

## 7. Full-text search is workable, but you will write the query parser yourself

**What we did.** We keep a `STORED` computed column,
`search_tsv TSVECTOR AS (to_tsvector('simple', search_text)) STORED`, with a
scope-prefixed inverted index — and that part is a genuine pleasure, identical
to PostgreSQL and fully declarative. What we had to write ourselves is the
query side: `_safe_or_tsquery` normalizes NFKC/casefold, regex-tokenizes,
de-duplicates, caps token count, and quotes each token into `'a' | 'b'`,
because there is no `websearch_to_tsquery` and passing raw user text to
`to_tsquery` is a syntax error waiting to happen.

**Rough or great.** Mixed. Storage and indexing: great. Query construction and
ranking: thin. No BM25, no `ts_rank_cd`, no `setweight`, GIN and GiST are the
same implementation, and non-English dictionaries are absent (we chose `simple`
over `english` partly for that reason, and normalize in the application
instead).

**What we'd ask for.** `websearch_to_tsquery` first — it is the single function
that turns "accept free text from a user" from a security exercise into a call.
BM25 ranking second.

## 8. Trigram search is a useful subset with one sharp edge

**What we did.** Fuzzy identifier matching uses `%` against a bounded
`lookup_text` column with `gin_trgm_ops`, and `similarity()` for ranking only.
Because `%` reads `pg_trgm.similarity_threshold` from the session, we pin it
per transaction so a cluster or session default cannot silently change
retrieval semantics or break parity with our in-memory adapter.

**Rough or great.** The pinning works exactly as documented and we have live
tests asserting parity with CockroachDB's own `similarity()` — that is a good
outcome. The sharp edge is that a *session GUC* is load-bearing for query
semantics; forget the `SET LOCAL` and your recall quietly changes. Also,
`word_similarity` and its operators are missing, which matters for matching a
short identifier inside a long string.

**What we'd ask for.** An operator form that takes the threshold explicitly, so
correctness does not depend on session state. And `word_similarity`.

## 9. Index hints are a good escape hatch; "no legal plan" is a bad error

**What we did.** Production SQL carries explicit index hints —
`retrieval_documents@<index>`, `INNER LOOKUP JOIN memories@primary`,
`retrieval_vectors_1024@retrieval_vectors_1024_ann_v2`, and `@primary` for the
oracle — and live `EXPLAIN` tests assert them.

**Rough or great.** Both. The hints exist, they work, and being able to pin a
plan shape in a correctness-critical path is a real strength. But we got there
by hitting two bad plans on 26.2.1: a complex trust predicate inside a forced
lookup join produced *no legal plan at all*, and an unhinted join hashed over a
wide `memories` scan. The first failure gave no indication of which predicate
made the plan illegal.

**What we'd ask for.** When a hint yields no legal plan, name the predicate or
the join condition that blocked it. Right now the fix is bisection by hand.

## 10. Recursive CTEs pushed us into application-side traversal

**What we did.** The graph lane walks `memory_links` one to two hops. The
obvious implementation is a recursive CTE. We wrote a fixed-depth application
loop instead, because CockroachDB 26.2 requires an explicit termination
condition, discourages relying on an outer `LIMIT` in production, and documents
that some recursive CTEs are not yet optimized. Each hop is a bounded `LATERAL`
scan over two directional covering indexes, followed by canonical validation
before fan-out, with explicit seed, hop, fan-out, and edge-examination budgets.

**Rough or great.** Rough as a starting point, but honestly a better outcome:
the budgets are explicit, both directions are trivially `EXPLAIN`-able, and
hidden or untrusted endpoints cannot displace a valid neighbour. Still, this is
the clearest case of "we wanted to write SQL and wrote Python instead".

**What we'd ask for.** Optimized recursive CTEs with documented depth and
row-budget controls. Graph traversal over an edge table is a mainstream
retrieval pattern now, not an exotic one.

## 11. One failed statement poisons the whole transaction, and hybrid retrieval is many statements

**What we did.** Five candidate lanes must see one MVCC snapshot, so they share
one connection and one read transaction. But any single failing statement puts
that transaction into `25P02` and takes the other four lanes and the final
hydration with it. We serialize lane work on the connection and wrap each lane
in its own savepoint, with a live regression test that injects a `42703` into
one lane and asserts the surviving lanes and hydration are unaffected.

**Rough or great.** Rough. This is where "one consistent snapshot" and
"independent failure domains per lane" collide, and savepoints are the only
tool. Every hybrid retriever built on CockroachDB will need this and most will
discover it in production.

**What we'd ask for.** Either an opt-in statement-level rollback mode (Postgres
clients get this via implicit savepoints in some drivers), or explicit
documentation and a code sample for "run N independent read lanes in one
snapshot" — because the naive version works right up until a lane fails.

## 12. Retry ergonomics are correct, and entirely the application's burden

**What we did.** Every mutation is a pure database closure passed to
`run_serializable`, which retries `40001` with bounded exponential backoff and
jitter (5 attempts, 25 ms → 500 ms). `40003` — and, equally, a connection loss
*after* the body returned, or `08xxx`, or `57P01/2/3` — raises
`AmbiguousTransactionResult`, which is never retried blindly: the caller
resolves it through the operation's idempotency key. A concurrent insert can
also surface as `23505` rather than `40001`, which is a third path into the same
resolution logic. That machinery — `idempotency_records`, normalized request
hashing, stored responses, `Idempotency-Replayed` headers — is a meaningful
fraction of the codebase.

**Rough or great.** The semantics are *right*, and we would not trade them for a
weaker isolation level; "the commit may or may not have happened" is a true
statement about distributed systems, not a wart. What is rough is how much of
the burden lands on the application, and how scattered the guidance is. The
constraint that a retryable closure may contain no side effects also shapes the
entire architecture (it is why the outbox exists).

**What we'd ask for.** (a) A single authoritative "ambiguous commit checklist"
in the docs enumerating every SQLSTATE that means *unknown* — `40003`, `08xxx`,
`57Pxx`, and the `23505`-as-a-race case — with the idempotency-ledger pattern
next to it. (b) A retry helper in the async Python ecosystem that people
actually use; `sqlalchemy`-flavoured examples do not help an `asyncio` +
`psycopg3` codebase.

## 13. There is no way to ask "is my schema still correct?"

**What we did.** Our retrieval indexes are correctness-critical: a same-named
B-tree where an inverted index should be, or a vector index with its prefix
columns in the wrong order, passes a name-only check and silently turns every
recall into a broad scan. So `swarmbrain-schema verify` reads `SHOW CREATE
TABLE` and `pg_indexes` and *parses the text* to check index method, opclass
(`gin_trgm_ops`, `vector_cosine_ops`), prefix column order, the computed
`TSVECTOR` expression, and primary-key shape. Column-order verification is
literally successive `find()` calls over DDL text.

Two smaller edges in the same area: `CREATE INVERTED INDEX` comes back as
`USING gin`, so anyone diffing DDL round-trips gets a mismatch; and there is no
server-side "apply this script" path, so we ship a 70-line quote- and
comment-aware SQL statement splitter.

**Rough or great.** Rough. Verification is the difference between a fast system
and a mysteriously slow one, and right now it is string archaeology.

**What we'd ask for.** A structured catalog view exposing, per index: method,
opclass per column, column order, storing columns, and — for vector indexes —
the ops class and prefix arity. `crdb_internal` would be a fine home. This is a
small feature that would delete a hundred lines of fragile parsing from every
serious adopter.

## 14. An `O(N)`, atomic, `40001`-exposed backfill has no good landing zone

**What we did.** Schema install rebuilds the lexical, exact-term, and vector
projections for every pre-existing memory: a keyset-paged loop (500 rows a
page) inside one `SERIALIZABLE` transaction, with a hand-rolled five-attempt
`40001` retry and a final stale-row sweep, so a successful install never
publishes a half-rebuilt projection. The cost is an operator quiesce barrier:
every old writer must be stopped, install and verify must run, and only then may
new processes start. Our own README documents this, and it is the least pleasant
thing we ask of an operator.

**Rough or great.** Rough — and it is not really CockroachDB's fault so much as
a gap. Index backfills are already resumable, observable, online jobs. A
user-defined projection backfill is none of those things, so the three
properties we need (atomic, `O(N)`, retryable) have no intersection.

**What we'd ask for.** A resumable, observable job primitive for user-defined
backfills — the same lifecycle as an index backfill, with progress and rate
limiting — so "rebuild a derived projection" can be online instead of a
maintenance window.

---

## What worked without drama, and mattered most

- **`SERIALIZABLE` by default.** Twelve agents racing four tasks produce exactly
  four leases with one short transaction and a partial unique index. No
  advisory locks, no lock service, no compare-and-swap loop. This is the
  feature that made an agent swarm tractable, and it needed no tuning.
- **Partial and covering indexes**, `STORING (...)`, computed `STORED` columns,
  and `INSERT … ON CONFLICT` all behave exactly like PostgreSQL, so the entire
  bitemporal supersession design ported over unchanged.
- **`EXPLAIN` is stable enough to assert on in CI.** We gate on plan shape for
  five query families and it has not been flaky.
- **The error surface is precise.** `40001` vs `40003` vs `23505` is a
  distinction the application genuinely needs, and CockroachDB reports it
  accurately.

## On the Managed MCP Server — now from real usage

We ran it on 2026-08-12: an OAuth session (PKCE + dynamic client registration —
which worked on the first attempt, against a stock Python MCP client, and
deserves credit for that) pinned to one cluster, driving a ten-call read-only
inspection of the swarm's live memory store, including a vector search through
`select_query` and an `explain_query` receipt showing the ANN index in the
plan (`evidence/20260812T131535Z-managed-mcp-inspection.json`).

Friction found, in order of impact:

1. **`explain_query` failures are opaque.** A query that `EXPLAIN` accepts
   over a SQL connection failed through the tool with only
   `explain query: SQL execution failed` — no SQLSTATE, no server message. By
   contrast `select_query` relays the real error (`column "m.superseded_by"
   does not exist`), which let an agent self-correct in one step. The
   `explain_query` path should relay the underlying error the same way; an
   agent cannot repair a query it cannot see the failure of.
2. **No parameter binding.** Understandable for an agent-facing SQL tool, but
   for vector workloads it means inlining a 1024-dimension literal (~8 KB of
   text) into the statement. A documented size ceiling — or a first-class
   "embed this text with the cluster's model, then search" tool — would remove
   the sharpest edge for exactly the workload this hackathon is about.
3. **`select_query`'s guard is precise and its message is good** ("only SELECT
   statements are allowed, got EXPLAIN") — but it means plan inspection is
   exclusively `explain_query`'s job, which loops back to point 1.

The standing design ask: make **read-only mode plus per-session audit export a
first-class, verifiable artifact**. For a memory system, "here is the audit log
of every query the agent ran against the memory store" is not a nice-to-have —
it is the thing that makes an agent's access to a shared brain reviewable. If a
Managed MCP session can emit a signed, exportable transcript, that becomes a
compliance story rather than a convenience feature.

## Why this workload should matter to CockroachDB

Agentic memory is not one more OLTP app; it is a workload whose buying criteria
are CockroachDB's exact differentiators. Fleet coordination is `SERIALIZABLE`
contention by construction. Governance wants lineage and audit — append-only,
bitemporal SQL. Platform teams want the agent layer to inherit the survivability
and horizontal scale the database was already approved for, and they want
memory that no model vendor owns. Every enterprise standing up an agent
platform in the next two years will need somewhere to put exactly this state —
and the alternatives on the table today are a vendor's opaque session store or
a Redis cache with none of the above. The sharper the vector prefix story, the
managed MCP audit surface, and the filtered-ANN ergonomics get, the easier that
argument becomes to make inside those organizations. We wrote this entry partly
to be able to make it.
