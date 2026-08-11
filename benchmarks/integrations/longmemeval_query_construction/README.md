# E7 query-time memory construction

This evaluation-only package implements a source-bound, LazyMem-shaped
retrieve-then-construct experiment. It makes no constructor, tokenizer,
reader, judge, embedding, database, or network call and changes no serving
path.

The transfer source is [LazyMem v2](https://arxiv.org/abs/2607.22690v2), with
the released repository frozen at commit
`af4109960aacb90d6dba994e9103a36a165cc380`. The paper/repository configuration
retrieves 50 raw messages, restores two neighboring messages on each side,
groups overlapping/touching local spans, splits them into at most eight
messages, and advances long windows with stride seven. That window geometry is
implemented exactly. The released pipeline collapses whitespace before window
construction; E7 deliberately preserves the original source bytes so cited
spans can be checked locally. Swarm Brain's E1-B/F0 identities, source-span
contract, prompt, evidence schema, and promotion rules are separate
hypotheses, so E7 does not claim a LazyMem reproduction or reuse its reported
scores.

## Frozen cells

- **E7-A:** chronological raw E1-B top-50 control, with zero constructor calls.
- **E7-B:** query-conditioned KEEP/DROP construction. Verbatim, extractive, and
  caller-attested abstractive outputs are accepted for analysis.
- **E7-C:** grounded construction. Every KEEP must be byte-verbatim or an exact
  deterministic join of cited UTF-8 source spans; abstractive output fails
  closed.

E7-B tests the paper's central query-time construction idea. E7-C tests whether
most of the benefit survives without letting generated compression become an
unverified fact source.

## Admission boundary

`build_retrieved_turn_pool(...)` requires the original E1-A fusion and
CrossEncoder score evidence, deterministically replays E1-B, and takes exactly
the replayed result's first 50 immutable F0 turns. A caller-assembled or
`dataclasses.replace`-mutated E1-B result is not authority.

`build_query_windows(...)` requires the manifest, authoritative pinned
`source_bytes`, E1-B result, original E1-A fusion, and CrossEncoder evidence.
Before it checks a case or creates a window, it runs the official freezer for
an official manifest or the pinned synthetic freezer otherwise, requires exact
manifest equality, and deterministically replays E1-B. Consequently, a
valid-looking manifest or retrieved pool produced with `dataclasses.replace`
cannot rewrite the query, current date, source record, projection, or selected
turns.

After building the complete window set, the builder issues a private,
process-local sealed authority receipt. `QueryWindowBatch` binds its exact
preflight and pool instances, query/date hashes and byte lengths, authoritative
turn count, and complete window digest to that receipt on every construction,
including `dataclasses.replace`. Its public trace contains only the receipt's
content-free projection. The receipt is an in-process construction boundary,
not a cryptographic signature or portable authentication token.

The constructor sees only:

1. query text;
2. authoritative current date;
3. each local message's timestamp, role, and raw content.

Question type, answer, answer-session IDs, `has_answer`, judge labels, and
benchmark IDs are absent from the rendered request. The content-bearing request
is canonical JSON and SHA-256 bound.

Each external window receipt must cover exactly one decision per message and
bind the constructor model/revision/deployment/artifacts, frozen prompt hashes,
exact normalized response, unique local/provider request IDs, token usage,
latency, and cost. E7 v2 privately retains the exact raw OpenAI-compatible
response bytes and strictly reparses the response model, provider request ID,
single stopped choice, JSON decision schema, and reconciled provider usage on
every receipt validation. The public trace contains only the raw byte count and
digest, hashed provider identity fields, normalized decision bindings, and
replayed usage; it does not copy response content or request IDs.

Receipt/model identity remains externally attested and unsigned: raw replay
proves that the normalized decisions and token counts match the retained bytes,
not that those bytes came from the claimed endpoint or weights. Observed
latency and externally priced cost also remain caller supplied. Those remaining
trust boundaries are stated explicitly in every receipt and result claim.

Every KEEP cites ordered, non-overlapping byte spans in its immutable source
turn. Verbatim and extractive content is recomputed locally. E7-B abstractive
content retains citations but semantic faithfulness remains unproven. Duplicate
messages from overlapping windows use the released LazyMem policy: keep the
longer compression, then the earlier window on a tie. Output is restored to
chronological order.

## Remaining empirical boundary

The result contains source-bound reader context and a text-free, raw-response-
replayable trace, but no real E7 constructor run exists yet. It also does not
establish exact full-reader-prompt token packing, QA accuracy, paper parity, or
serving eligibility. Those require the frozen 8,192-token packer, paired
reader/judge evidence, multiplicity-safe selection, and a sealed confirmation
run. E7 evidence must remain non-promotional until those stages pass.
