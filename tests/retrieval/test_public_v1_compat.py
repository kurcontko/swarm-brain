from __future__ import annotations

from swarmbrain.domain.memory import RecallBundle, RecallHit, RecallQuery


def test_retrieval_v1_keeps_public_contract_and_marks_internal_selector_deprecated() -> None:
    assert set(RecallQuery.model_fields) == {
        "text",
        "task_id",
        "memory_ids",
        "kinds",
        "visibilities",
        "states",
        "include_refuted",
        "include_superseded",
        "world_at",
        "recorded_at",
        "min_score",
        "limit",
        "include_evidence",
        "include_lineage",
    }
    assert set(RecallHit.model_fields) == {"memory", "score", "reasons", "evidence"}
    assert set(RecallBundle.model_fields) == {
        "query",
        "hits",
        "generated_at",
        "total_candidates",
        "truncated",
    }
    schema = RecallQuery.model_json_schema()
    assert schema["properties"]["memory_ids"]["deprecated"] is True
    assert "purpose" not in schema["properties"]
    assert "trace" not in RecallBundle.model_fields
