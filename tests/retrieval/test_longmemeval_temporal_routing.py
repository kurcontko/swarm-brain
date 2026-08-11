from __future__ import annotations

from datetime import UTC, datetime

import _longmemeval_common as common
import pytest
from _longmemeval_common import (
    parse_longmemeval_datetime,
    parse_longmemeval_temporal_query,
    retrieve_question,
    temporal_parse_metadata,
)

from swarmbrain.domain.retrieval import RetrievalSignal


class _CountingEmbeddingProvider:
    model_name = "counting-test-v1"
    dimensions = 4

    def __init__(self) -> None:
        self.document_batches: list[int] = []
        self.query_calls = 0

    async def embed_documents(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        self.document_batches.append(len(texts))
        return tuple((1.0, 0.0, 0.0, 0.0) for _ in texts)

    async def embed_query(self, text: str) -> tuple[float, ...]:
        assert text
        self.query_calls += 1
        return (1.0, 0.0, 0.0, 0.0)


def _record(question: str = "What kitchen appliance did I buy 10 days ago?") -> dict:
    return {
        "question_id": "temporal-fixture",
        "question_type": "temporal-reasoning",
        "question": question,
        "question_date": "2023/03/25 (Sat) 18:26",
        "answer_session_ids": ["target"],
        "haystack_session_ids": ["old", "target", "future"],
        "haystack_dates": [
            "2023/03/01 (Wed) 09:00",
            "2023/03/15 (Wed) 09:00",
            "2023/03/16 (Thu) 09:00",
        ],
        "haystack_sessions": [
            [{"role": "user", "content": "I bought some garden soil."}],
            [{"role": "user", "content": "I bought a countertop smoker."}],
            [{"role": "user", "content": "I bought a replacement kettle."}],
        ],
    }


def test_longmemeval_datetime_parser_is_strict_and_timezone_explicit() -> None:
    assert parse_longmemeval_datetime("2023/03/15 (Wed) 09:00") == datetime(
        2023, 3, 15, 9, tzinfo=UTC
    )

    with pytest.raises(ValueError, match="weekday mismatch"):
        parse_longmemeval_datetime("2023/03/15 (Tue) 09:00")
    with pytest.raises(ValueError, match="unsupported LongMemEval datetime"):
        parse_longmemeval_datetime("2023-03-15T09:00:00Z")


def test_longmemeval_temporal_parser_routes_only_a_closed_proposal() -> None:
    routed = parse_longmemeval_temporal_query(_record())
    comparison = parse_longmemeval_temporal_query(
        _record("How many days passed between buying the smoker and replacing the kettle?")
    )

    assert routed.closed_referenced_time is not None
    assert routed.closed_referenced_time.referenced_valid_from == datetime(2023, 3, 15, tzinfo=UTC)
    assert temporal_parse_metadata(routed) == {
        "status": "matched",
        "confidence": "medium",
        "reason": "anchored_relative_period",
        "relative": True,
        "routed": True,
        "valid_from": "2023-03-15T00:00:00+00:00",
        "valid_to": "2023-03-16T00:00:00+00:00",
    }
    assert comparison.closed_referenced_time is None
    comparison_metadata = temporal_parse_metadata(comparison)
    assert comparison_metadata is not None
    assert comparison_metadata["routed"] is False


@pytest.mark.asyncio
async def test_temporal_routing_is_opt_in_and_uses_session_valid_time() -> None:
    baseline = await retrieve_question(_record(), limit=3, use_dense=False)
    routed = await retrieve_question(
        _record(),
        limit=3,
        use_dense=False,
        temporal_query_routing=True,
    )

    assert baseline.temporal_parse is None
    assert all(
        batch.lane is not RetrievalSignal.TEMPORAL for batch in baseline.execution.trace.batches
    )

    temporal = next(
        batch for batch in routed.execution.trace.batches if batch.lane is RetrievalSignal.TEMPORAL
    )
    keys = routed.key_by_memory_id
    assert [keys[candidate.canonical_id] for candidate in temporal.candidates] == [
        "001:target",
        "000:old",
    ]
    assert "002:future" not in {keys[hit.memory.memory_id] for hit in routed.execution.bundle.hits}


@pytest.mark.asyncio
async def test_question_corpus_embeddings_are_projected_in_one_provider_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _CountingEmbeddingProvider()
    monkeypatch.setattr(common, "_provider_singleton", provider)

    retrieved = await retrieve_question(_record(), limit=3, use_dense=True)

    assert retrieved.execution.bundle.hits
    assert provider.document_batches == [3]
    assert provider.query_calls == 1
