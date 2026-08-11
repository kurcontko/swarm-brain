from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_representation.contracts import (
    KeyFamily,
    RepresentationCorpus,
    canonical_json_bytes,
    compile_question_canonical_values,
    sha256_bytes,
)
from benchmarks.integrations.longmemeval_representation.evidence import (
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS,
    DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS,
    DEEPSEEK_MODEL_ID,
    DeepSeekR1PricingIdentity,
    DeepSeekR1ProviderAttempt,
    RepresentationEvidenceError,
    build_deepseek_r1_attempt_record,
    build_deepseek_r1_evidence_record,
    deepseek_r1_attempt_jsonl_bytes,
    deepseek_r1_evidence_jsonl_bytes,
    deepseek_r1_extractor_identity,
    deepseek_r1_request_bytes,
    load_deepseek_r1_attempt_artifact,
    load_deepseek_r1_evidence_artifact,
    replay_deepseek_r1_attempt_jsonl,
    replay_deepseek_r1_evidence_jsonl,
    replay_deepseek_r1_evidence_record,
    replay_deepseek_r1_response,
)
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source(*, question_id: str = "q-e6-r1", content: str = "SOURCE-ONLY-PRIVATE"):
    record = {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": "PRIVATE-QUESTION-MUST-NOT-ENTER",
        "answer": "PRIVATE-GOLD-MUST-NOT-ENTER",
        "question_date": "2025/01/04 (Sat) 11:00",
        "haystack_session_ids": ["session-1"],
        "haystack_dates": ["2025/01/03 (Fri) 09:07"],
        "haystack_sessions": [[{"role": "user", "content": content}]],
        "answer_session_ids": ["session-1"],
    }
    raw = (json.dumps([record], ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=sha256_bytes(raw),
        source_label="synthetic-longmemeval.json",
    )
    return corpus, compile_question_canonical_values(corpus, question_id=question_id)[0]


def _extractor():
    return deepseek_r1_extractor_identity(
        model_revision="official-api-alias-observed-fixture",
        model_artifact_sha256=_digest("model-artifact"),
        identity_artifact_sha256=_digest("identity-artifact"),
    )


def _pricing() -> DeepSeekR1PricingIdentity:
    return DeepSeekR1PricingIdentity(
        version="deepseek-pricing-fixture-v1",
        artifact_sha256=_digest("pricing-artifact"),
        cache_miss_input_microusd_per_million_tokens=140_000,
        output_microusd_per_million_tokens=280_000,
    )


def _response(
    content: str,
    *,
    request_id: str,
    prompt_tokens: int = 100,
    completion_tokens: int = 8,
    model: str = DEEPSEEK_MODEL_ID,
    finish_reason: str = "stop",
) -> bytes:
    return canonical_json_bytes(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": 1_786_308_624,
            "model": model,
            "system_fingerprint": "fp_fixture",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": prompt_tokens,
            },
        }
    )


def _attempt(
    source,
    content: str,
    *,
    request_id: str,
    prompt_tokens: int = 100,
    completion_tokens: int = 8,
    http_attempts: int = 1,
    latency_microseconds: int = 1_000,
) -> DeepSeekR1ProviderAttempt:
    extractor = _extractor()
    return DeepSeekR1ProviderAttempt(
        raw_request=deepseek_r1_request_bytes(source, extractor),
        raw_response=_response(
            content,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        http_attempts=http_attempts,
        latency_microseconds=latency_microseconds,
    )


def _valid_content(text: str = "concise summary; durable fact; keyword") -> str:
    return json.dumps({"merged_sfk": text}, ensure_ascii=False, separators=(",", ":"))


def _record(source, attempts: tuple[DeepSeekR1ProviderAttempt, ...]) -> dict[str, Any]:
    return build_deepseek_r1_evidence_record(
        source=source,
        extractor=_extractor(),
        pricing=_pricing(),
        application_attempts=attempts,
    )


def _replay(record: dict[str, Any], source):
    return replay_deepseek_r1_evidence_record(
        record,
        source=source,
        extractor=_extractor(),
        pricing=_pricing(),
    )


def test_request_is_canonical_source_only_and_frozen_to_512_tokens() -> None:
    _, source = _source()
    raw = deepseek_r1_request_bytes(source, _extractor())
    body = json.loads(raw)

    assert raw == canonical_json_bytes(body)
    assert set(body) == {"model", "messages", "temperature", "max_tokens", "thinking"}
    assert body["model"] == DEEPSEEK_MODEL_ID
    assert body["max_tokens"] == DEEPSEEK_MAX_TOKENS == 512
    assert body["thinking"] == {"type": "disabled"}
    assert body["messages"][0]["role"] == "user"
    prompt = body["messages"][0]["content"]
    assert json.loads(prompt.split("\nINPUT_JSON:\n", 1)[1]) == {"source_value": source.raw_value}
    assert "PRIVATE-QUESTION-MUST-NOT-ENTER" not in prompt
    assert "PRIVATE-GOLD-MUST-NOT-ENTER" not in prompt
    assert source.question_id not in prompt


def test_valid_attempt_replays_to_existing_e6_key_and_receipt_contracts() -> None:
    corpus, source = _source()
    attempt = _attempt(source, _valid_content(), request_id="provider-valid")
    record = _record(source, (attempt,))
    replayed = _replay(record, source)

    assert replayed.raw_request == attempt.raw_request
    assert replayed.raw_response == attempt.raw_response
    assert replayed.selected_application_attempt == 1
    assert replayed.derived_key.family is KeyFamily.MERGED_SFK
    assert replayed.derived_key.key_text == "concise summary; durable fact; keyword"
    assert replayed.construction_receipt.output_key_ids == (replayed.derived_key.key_id,)
    assert replayed.construction_receipt.input_tokens == 100
    assert replayed.construction_receipt.output_tokens == 8
    assert replayed.construction_receipt.retry_count == 0

    accepted = RepresentationCorpus(
        projection_corpus=corpus,
        values=(source,),
        derived_keys=(replayed.derived_key,),
        construction_receipts=(replayed.construction_receipt,),
    )
    assert accepted.derived_keys == (replayed.derived_key,)


def test_invalid_schema_attempt_is_preserved_before_first_valid_and_all_costs_sum() -> None:
    _, source = _source()
    invalid = _attempt(
        source,
        '{"wrong":"still retained"}',
        request_id="provider-invalid",
        prompt_tokens=90,
        completion_tokens=5,
        http_attempts=2,
        latency_microseconds=2_000,
    )
    valid = _attempt(
        source,
        _valid_content("valid on application retry"),
        request_id="provider-valid",
        prompt_tokens=100,
        completion_tokens=7,
        http_attempts=3,
        latency_microseconds=4_000,
    )
    record = _record(source, (invalid, valid))
    replayed = _replay(record, source)

    assert record["application_attempts"][0]["output_validation"] == {
        "protocol": "strict-merged-sfk-json-schema-v1",
        "accepted": False,
        "error_code": "content-schema-mismatch",
    }
    assert record["application_attempts"][1]["output_validation"]["accepted"] is True
    assert (
        base64.b64decode(record["application_attempts"][0]["provider_response"]["raw_base64"])
        == invalid.raw_response
    )
    assert replayed.selected_application_attempt == 2
    assert replayed.derived_key.key_text == "valid on application retry"
    receipt = replayed.construction_receipt
    assert receipt.input_tokens == 190
    assert receipt.output_tokens == 12
    assert receipt.latency_microseconds == 6_000
    assert receipt.retry_count == 4  # five total HTTP sends minus the initial send
    expected_cost = _pricing().upper_bound_microusd(
        input_tokens=90,
        output_tokens=5,
        retry_count=1,
        request_max_tokens=512,
    ) + _pricing().upper_bound_microusd(
        input_tokens=100,
        output_tokens=7,
        retry_count=2,
        request_max_tokens=512,
    )
    assert receipt.cost_microusd == expected_cost
    assert record["normalized"]["aggregate_accounting"] == {
        "protocol": "provider-reported-openai-compatible-usage-v1",
        "application_attempts": 2,
        "http_attempts": 5,
        "retry_count": 4,
        "input_tokens": 190,
        "output_tokens": 12,
        "total_tokens": 202,
        "latency_microseconds": 6_000,
        "estimated_cost_upper_bound_microusd": expected_cost,
        "pricing_identity_sha256": _pricing().identity_sha256,
    }


def test_standalone_attempt_record_retains_invalid_bytes_before_completion() -> None:
    _, source = _source()
    attempt = _attempt(
        source,
        "not-json-but-provider-envelope-is-valid",
        request_id="provider-invalid",
    )

    record = build_deepseek_r1_attempt_record(
        source=source,
        extractor=_extractor(),
        attempt=attempt,
        application_attempt=1,
    )

    assert record["output_validation"] == {
        "protocol": "strict-merged-sfk-json-schema-v1",
        "accepted": False,
        "error_code": "content-malformed-json",
    }
    assert base64.b64decode(record["provider_request"]["raw_base64"]) == attempt.raw_request
    assert base64.b64decode(record["provider_response"]["raw_base64"]) == attempt.raw_response


def test_standalone_attempt_jsonl_checkpoints_and_recovers_invalid_bytes(
    tmp_path: Path,
) -> None:
    _, source = _source()
    invalid = _attempt(
        source,
        '{"wrong":"retained before retry"}',
        request_id="provider-invalid",
        http_attempts=2,
        latency_microseconds=2_000,
    )
    valid = _attempt(
        source,
        _valid_content("accepted after checkpoint"),
        request_id="provider-valid",
        latency_microseconds=3_000,
    )
    records = tuple(
        build_deepseek_r1_attempt_record(
            source=source,
            extractor=_extractor(),
            attempt=attempt,
            application_attempt=index,
        )
        for index, attempt in enumerate((invalid, valid), start=1)
    )
    raw = deepseek_r1_attempt_jsonl_bytes(records)

    assert replay_deepseek_r1_attempt_jsonl(
        raw,
        source=source,
        extractor=_extractor(),
    ) == (invalid, valid)
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(raw)
    assert load_deepseek_r1_attempt_artifact(
        path,
        source=source,
        extractor=_extractor(),
    ) == (invalid, valid)

    with pytest.raises(RepresentationEvidenceError, match="not canonical"):
        replay_deepseek_r1_attempt_jsonl(
            b" " + raw,
            source=source,
            extractor=_extractor(),
        )

    after_valid = build_deepseek_r1_attempt_record(
        source=source,
        extractor=_extractor(),
        attempt=invalid,
        application_attempt=3,
    )
    with pytest.raises(RepresentationEvidenceError, match="first schema-valid"):
        replay_deepseek_r1_attempt_jsonl(
            deepseek_r1_attempt_jsonl_bytes((*records, after_valid)),
            source=source,
            extractor=_extractor(),
        )


@pytest.mark.parametrize(
    "attempts",
    (
        ("valid", "valid"),
        ("valid", "invalid"),
        ("invalid",),
        ("invalid", "invalid", "invalid"),
    ),
)
def test_application_ledger_must_stop_at_its_first_valid_response(
    attempts: tuple[str, ...],
) -> None:
    _, source = _source()
    values = tuple(
        _attempt(
            source,
            _valid_content(f"valid-{index}") if kind == "valid" else '{"wrong":true}',
            request_id=f"provider-{index}",
        )
        for index, kind in enumerate(attempts)
    )

    with pytest.raises(RepresentationEvidenceError, match="first schema-valid"):
        _record(source, values)


def test_application_and_http_attempt_bounds_fail_closed() -> None:
    _, source = _source()
    valid = _attempt(source, _valid_content(), request_id="provider-valid")
    with pytest.raises(RepresentationEvidenceError, match="outside the frozen bound"):
        _record(source, (valid,) * (DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS + 1))
    with pytest.raises(RepresentationEvidenceError, match="HTTP attempts"):
        DeepSeekR1ProviderAttempt(
            raw_request=valid.raw_request,
            raw_response=valid.raw_response,
            http_attempts=DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS + 1,
            latency_microseconds=1,
        )


def test_strict_response_rejects_model_usage_and_duplicate_field_tampering() -> None:
    with pytest.raises(RepresentationEvidenceError, match="response model"):
        replay_deepseek_r1_response(
            _response(_valid_content(), request_id="provider", model="forged-model")
        )

    usage_drift = json.loads(_response(_valid_content(), request_id="provider"))
    usage_drift["usage"]["total_tokens"] += 1
    with pytest.raises(RepresentationEvidenceError, match="does not reconcile"):
        replay_deepseek_r1_response(canonical_json_bytes(usage_drift))

    duplicate = (
        b'{"id":"provider","id":"shadow","model":"deepseek-v4-flash","choices":[],"usage":{}}'
    )
    with pytest.raises(RepresentationEvidenceError, match="malformed"):
        replay_deepseek_r1_response(duplicate)


@pytest.mark.parametrize(
    "mutation",
    (
        "route",
        "raw_request",
        "attempt_validation",
        "http_retry_count",
        "normalized_cost",
        "extractor",
    ),
)
def test_record_replay_rejects_route_raw_byte_and_normalized_tampering(
    mutation: str,
) -> None:
    _, source = _source()
    invalid = _attempt(source, '{"wrong":true}', request_id="provider-invalid")
    valid = _attempt(source, _valid_content(), request_id="provider-valid")
    record = deepcopy(_record(source, (invalid, valid)))
    if mutation == "route":
        record["route"]["family"] = "summary"
    elif mutation == "raw_request":
        raw = json.loads(valid.raw_request)
        raw["messages"][0]["content"] += "\nPRIVATE-QUESTION-MUST-NOT-ENTER"
        forged = canonical_json_bytes(raw)
        block = record["application_attempts"][1]["provider_request"]
        block["raw_base64"] = base64.b64encode(forged).decode("ascii")
        block["raw_bytes"] = len(forged)
        block["raw_sha256"] = sha256_bytes(forged)
    elif mutation == "attempt_validation":
        record["application_attempts"][0]["output_validation"]["accepted"] = True
    elif mutation == "http_retry_count":
        record["application_attempts"][0]["http_retry_count"] = 9
    elif mutation == "normalized_cost":
        record["normalized"]["aggregate_accounting"]["estimated_cost_upper_bound_microusd"] += 1
    else:
        record["extractor"]["model_revision"] = "forged-revision"

    with pytest.raises(RepresentationEvidenceError):
        _replay(record, source)


def test_jsonl_is_canonical_replayable_and_rejects_noncanonical_or_duplicate_routes(
    tmp_path: Path,
) -> None:
    _, source = _source()
    record = _record(
        source,
        (_attempt(source, _valid_content(), request_id="provider-valid"),),
    )
    raw = deepseek_r1_evidence_jsonl_bytes((record,))

    replayed = replay_deepseek_r1_evidence_jsonl(
        raw,
        sources=(source,),
        extractor=_extractor(),
        pricing=_pricing(),
    )
    assert len(replayed) == 1
    path = tmp_path / "evidence.jsonl"
    path.write_bytes(raw)
    assert (
        load_deepseek_r1_evidence_artifact(
            path,
            sources=(source,),
            extractor=_extractor(),
            pricing=_pricing(),
        )
        == replayed
    )

    with pytest.raises(RepresentationEvidenceError, match="not canonical"):
        replay_deepseek_r1_evidence_jsonl(
            b" " + raw,
            sources=(source,),
            extractor=_extractor(),
            pricing=_pricing(),
        )
    with pytest.raises(RepresentationEvidenceError, match="repeats a source route"):
        replay_deepseek_r1_evidence_jsonl(
            raw + raw,
            sources=(source,),
            extractor=_extractor(),
            pricing=_pricing(),
        )


def test_authoritative_source_and_pricing_identity_are_required_for_replay() -> None:
    _, source = _source()
    _, other_source = _source(question_id="other-question", content="OTHER-SOURCE")
    record = _record(
        source,
        (_attempt(source, _valid_content(), request_id="provider-valid"),),
    )
    with pytest.raises(RepresentationEvidenceError, match="authoritative source"):
        replay_deepseek_r1_evidence_record(
            record,
            source=other_source,
            extractor=_extractor(),
            pricing=_pricing(),
        )
    changed_pricing = DeepSeekR1PricingIdentity(
        version="different-pricing",
        artifact_sha256=_digest("different-pricing-artifact"),
        cache_miss_input_microusd_per_million_tokens=140_000,
        output_microusd_per_million_tokens=280_000,
    )
    with pytest.raises(RepresentationEvidenceError, match="pricing identity"):
        replay_deepseek_r1_evidence_record(
            record,
            source=source,
            extractor=_extractor(),
            pricing=changed_pricing,
        )
