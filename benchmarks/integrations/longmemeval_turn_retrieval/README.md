# LongMemEval E1-A turn-candidate fusion

This package implements the frozen Swarm Brain turn-transfer E1-A retrieval
head. It is pure and offline: callers supply lexical and dense rankings over an
immutable `longmemeval_turns` projection, and the package validates and fuses
them with weighted RRF.

The registered v1 protocol fixes lexical/dense depths at 128/128, weights at
3.0/4.0, RRF `k=60`, and the unique fused head at 128 turns. The weights mirror
the interactive-general production policy, but the lane depths are evaluation
choices: dense depth 128 exceeds the current production serving cap of 100.
This is therefore a named transfer experiment, not production parity or a
paper-parity claim. Changing a value while retaining the v1 protocol version is
rejected; a later held-out design must use a new protocol version.

No scorer, embedding model, database, reader, judge, or network endpoint is
called here. Scorer/projection revisions and artifact digests are immutable
caller attestations. The trace explicitly says that lexical scores, dense
vectors, and dense scores were not recomputed or independently verified.

Fusion receives no gold fields. `evaluate_gold_session_recall` is a separate,
post-hoc function that records pre-cap and post-cap recall without changing the
candidate result. Both the trace and evaluator state that end-to-end QA was not
run and QA improvement is unproven.
