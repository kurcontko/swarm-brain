# Evidence-gated memory consolidation

Swarm Brain can run an asynchronous Observer/Reflector loop without giving a
model authority over memory governance. It is disabled by default and enabled
with `SWARMBRAIN_CONSOLIDATION_ENABLED=true`.

The Observer only accepts current, confirmed, evidence-backed memories from the
authenticated run. It canonically rehydrates them under the normal scope,
lifecycle, bitemporal, and source-trust predicates, assigns opaque local keys
(`m0`, `m1`, ...), freezes exact memory versions and `EvidenceRef` values, and
enqueues a bounded `consolidate_memory` item. The complete snapshot is bound by
an SHA-256 digest and a semantic dedupe key.

The Reflector may propose only `append`, `supersede`, `link`, or `noop`. Provider
requests contain opaque observation keys, content, and evidence excerpts, but
no memory IDs, source IDs, or versions. Provider responses can cite only those
keys and cannot contain scope, trust, lifecycle state, evidence identifiers, or
persistence identity. The optional OpenAI-compatible path uses a strict JSON
schema, temperature zero, streamed byte limits, a total timeout, and sanitized
errors. If it is absent or fails, the deterministic fallback safely abstains.

Before any effect, the worker durably stages the exact proposal plan under its
lease fence. A crash after a publication therefore reuses the staged plan and
the same action-index idempotency keys instead of asking the model again.
Before applying it, the service canonically rehydrates every input and compares
the exact version and full snapshot digest. Any change, rejection, loss of
trust, supersession, or scope change turns the whole plan into `stale_noop`.

Every non-noop proposal is translated locally into a `RememberCommand` and sent
through `MemoryService.publish` and the existing conservative policy:

- output lifecycle is always `tentative`;
- visibility is derived as the narrowest safe scope of the supporting inputs;
- confidence cannot exceed the least-confident supporting input;
- evidence is the exact deduplicated union of the immutable refs belonging to
  the provider-selected opaque support keys;
- lineage is persisted as typed `derived_from` links in the same governed
  remember transaction;
- a supersession proposal still obeys the poison guard, so it cannot replace a
  confirmed memory with an unconfirmed synthesis.

Bounds are operator-owned:

- `SWARMBRAIN_CONSOLIDATION_MAX_MEMORIES` (2–32, default 12)
- `SWARMBRAIN_CONSOLIDATION_MAX_ACTIONS` (1–8, default 4)
- `SWARMBRAIN_CONSOLIDATION_MAX_INPUT_BYTES` (1 KiB–1 MB, default 64 KiB)

Provider mode is enabled separately with
`SWARMBRAIN_CONSOLIDATION_USE_PROVIDER=true` and reuses the authenticated
`SWARMBRAIN_EXTRACTION_*` endpoint/model profile. Enabling provider mode without
both consolidation and a complete provider profile fails configuration.
