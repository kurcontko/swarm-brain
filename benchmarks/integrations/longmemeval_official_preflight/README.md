# Official LongMemEval-S packed-run preflight

This evaluation-only package closes the last boundary before an authorized
reader run. It does not call a tokenizer, reader, model, database, or network.
It compiles local corpus bytes, freezes a content-free plan, and later validates
already-produced exact-token prompt artifacts and tokenizer evidence before a
reader call is considered admissible.

## Two-phase admission

`freeze_official_preflight(...)` must run first. It accepts only the cleaned
LongMemEval-S bytes with SHA-256
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
and exactly 500 unique questions in source order. It independently compiles the
F0 turn projection and freezes:

- raw artifact bytes/hash, canonical parsed-record bytes/hash, projection hash,
  ordered question IDs, and per-record/question/date hashes;
- the current exact official answer-template hash and prompt-packer protocol;
- the primary 8,192-token complete-reader-prompt ceiling, indivisible turns,
  and skip-and-continue policy;
- an operator-precommitted exact tokenizer protocol, model, revision, artifact
  hash, and local executable hash.

The returned manifest says `ready_for_external_calls: false`. A caller cannot
mark an arbitrary dataset as official: official mode hard-codes both the
release digest and 500-case shape. `freeze_pinned_preflight(...)` exists for
nonofficial fixture or held-out rehearsals, but its manifest is explicitly
classified nonofficial and the official admission function rejects it.
`load_preflight_manifest_bytes(...)` reloads persisted strict JSON, rejects
duplicate fields and non-finite values, reconstructs every typed binding, and
requires the artifact to equal its recomputed constants and manifest digest.

`PinnedPromptTokenizerAdapter` closes the interface gap between the existing
repository-local `JsonlExactTokenizer` (`count`) and the prompt packer
(`count_prompt`). It requires a fresh boundary, verifies the precommitted
model/revision/artifact/executable identity and zero accounting before use,
then binds each source question digest to the observed complete-prompt hash,
byte count, provider request ID, and exact token count. Identity or accounting
drift fails on either side of every local count.

After every prompt has been packed through the pinned local exact-token
boundary—but before any reader call—`validate_official_prepared_run(...)`
reopens the logical boundary from the original corpus bytes. It requires one
`TurnPromptPackingResult` per official question in exact source order and
reconstructs each prompt from the official template, official question/date,
pinned turn projection, candidate blocks, and kept turn IDs. It rejects:

- missing, duplicate, extra, reordered, or cross-question cases and turns;
- a different corpus, projection, question, date, prompt protocol/template,
  layout literal, budget, packing policy, or tokenizer identity;
- stale candidate-order, decision, observation, history, or final-prompt
  digests;
- acceptance/oversize flags that disagree with exact receipts;
- missing independent initial/final counts, estimator claims, non-monotone or
  reused request IDs, and globally reused provider request IDs;
- tokenizer runtime evidence whose artifact/executable bytes, identity, request
  totals, response totals, provider-ID totals, or counted UTF-8 bytes do not
  reconcile with every prompt observation.

Only then does it emit a content-free `ready_for_reader_calls: true` receipt
binding all 500 prompt-trace digests and the tokenizer runtime evidence.

## Evidence limits

The validator proves internal byte/digest consistency and exact coverage. The
external tokenizer response identities remain provider-observed evidence; the
offline compiler cannot cryptographically prove what proprietary weights a
provider executed. The reader runner still has to require the ready-receipt
digest before making calls, and downstream QA/judge evidence remains a
separate boundary. No output here is a quality or SOTA claim.
