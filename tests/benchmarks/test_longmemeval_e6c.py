from __future__ import annotations

from copy import deepcopy

import pytest
from benchmarks.integrations.longmemeval_e6c.cohort import (
    E6C_ABSTENTION_COUNT,
    E6C_POSITIONS,
    E6C_SAMPLE,
    E6C_TYPE_COUNTS,
    answer_evidence_signature,
    corpus_fingerprint,
)
from benchmarks.integrations.longmemeval_e6c.metrics import (
    E6CMetricsError,
    paired_lower_bound,
)


def _record() -> dict[str, object]:
    return {
        "question_id": "q1",
        "question_type": "single-session-user",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2025/01/01 (Wed) 00:00"],
        "haystack_sessions": [[{"role": "user", "content": "source"}]],
        "answer_session_ids": ["s1"],
    }


def test_frozen_cohort_constants_have_exact_cardinality() -> None:
    assert len(E6C_POSITIONS) == E6C_SAMPLE == 160
    assert tuple(sorted(set(E6C_POSITIONS))) == E6C_POSITIONS
    assert sum(E6C_TYPE_COUNTS.values()) == E6C_SAMPLE
    assert E6C_ABSTENTION_COUNT == 9


def test_history_and_answer_evidence_bind_source_not_query_or_answer_text() -> None:
    first = _record()
    second = deepcopy(first)
    second["question"] = "changed query"
    second["answer"] = "changed answer"
    assert corpus_fingerprint(first) == corpus_fingerprint(second)
    assert answer_evidence_signature(first) == answer_evidence_signature(second)

    changed = deepcopy(first)
    changed["haystack_sessions"][0][0]["content"] = "different source"  # type: ignore[index]
    assert corpus_fingerprint(first) != corpus_fingerprint(changed)
    assert answer_evidence_signature(first) != answer_evidence_signature(changed)


def test_paired_lower_bound_is_deterministic_and_stratified() -> None:
    deltas = [1.0] * 8 + [-1.0] * 2
    strata = ["a"] * 5 + ["b"] * 5
    first = paired_lower_bound(deltas, strata, resamples=2_000, seed=17)
    second = paired_lower_bound(deltas, strata, resamples=2_000, seed=17)
    assert first == second
    assert first["delta"] == pytest.approx(0.6)
    assert first["bootstrap"]["tail"] == "one-sided-lower"

    with pytest.raises(E6CMetricsError, match="aligned"):
        paired_lower_bound([1.0], [], resamples=10)
