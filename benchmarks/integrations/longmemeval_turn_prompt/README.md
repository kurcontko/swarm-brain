# Exact LongMemEval turn-prompt packing

This evaluation-only package closes the boundary between frozen turn ranking
and the reader. It consumes immutable `TurnProjection` objects in caller-fixed
linear E1 order or caller-fixed ordered CoM blocks. It does not retrieve,
rerank, embed, tokenize, answer, judge, call a network service, or change the
production serving path.

## Frozen protocol

- Protocol: `swarmbrain-longmemeval-turn-prompt-pack-v1`.
- Budgets: 4,096, 8,192, and 16,384 complete reader-prompt tokens; 8,192 is
  primary.
- Prompt: the exact shared
  `scripts._longmemeval_common.OFFICIAL_ANSWER_TEMPLATE`, whose UTF-8 bytes are
  pinned by SHA-256.
- Linear turns are separated by two newlines.
- CoM turns are separated by two newlines and non-empty blocks are rendered as
  `=== Evidence Chain N ===`, in their original block positions. Empty blocks
  are omitted without renumbering later blocks.
- Turns are indivisible. For each candidate, the packer renders and counts the
  complete proposed prompt. A proposal over budget is skipped and scanning
  continues.
- A separate candidate-alone observation identifies genuinely oversized
  turns. The final accepted prompt is counted again independently.

Only an externally supplied exact-token boundary may provide counts. Every
receipt repeats the frozen tokenizer protocol plus the model/revision/artifact
identity digest, query digest, complete-prompt digest and byte count, local
request ID, unique provider request ID, and token count. The packer rejects
identity drift, digest/byte mismatch, non-increasing request IDs, reused
provider IDs, inconsistent repeated counts, or a final prompt outside the
selected budget. Token counts themselves are not required to increase when
text is appended: BPE/SentencePiece tokenization can retokenize a concatenation
boundary. Each exact proposal receipt is authoritative and acceptance depends
only on whether that count is within budget. There is no estimator fallback.

The content-free trace binds candidate and block order, all fixed separators
and chain headers, every receipt, each greedy decision, kept/dropped/oversized
turn IDs, and final prompt bytes/hash/tokens. Raw question and turn text appear
only in `TurnPromptPackingResult.prompt`; the trace retains hashes and immutable
turn provenance. The API accepts no answer, answer-session, rubric, or other
gold field, so those values cannot affect packing.

Parity-style CoM output can repeat turns across chains, but this packing
protocol deliberately rejects duplicate IDs. Use the preregistered deduplicated
rendering cell (or establish a separately named duplicate-aware protocol)
before packing such output.

Tests use a deterministic in-process fixture receipt issuer. It represents an
exact synthetic tokenizer contract for testing only and makes no tokenizer,
model, subprocess, or network call.
