from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import build_longmemeval_official_report as official_report
import pytest
import run_longmemeval_qa as qa
from build_longmemeval_official_report import (
    OFFICIAL_JUDGE_MODEL,
    OFFICIAL_REPORT_ARTIFACT_TYPE,
    OFFICIAL_REPORT_PROTOCOL_VERSION,
    OFFICIAL_REPORT_SCHEMA_VERSION,
    OfficialReportError,
    build_report,
    load_official_run,
)


def _build_fixture_dataset() -> list[dict[str, object]]:
    case_counts = [qa.LONGMEMEVAL_S_SESSION_COUNT // 500] * 500
    for index in range(qa.LONGMEMEVAL_S_SESSION_COUNT % 500):
        case_counts[index] += 1
    records: list[dict[str, object]] = []
    for index, count in enumerate(case_counts):
        question_id = f"q{index:03d}"
        session_ids = [f"session-{index:03d}-{offset:03d}" for offset in range(count)]
        records.append(
            {
                "question_id": question_id,
                "question_type": ("temporal-reasoning" if index % 2 == 0 else "multi-session"),
                "question": f"question {question_id}",
                "answer": f"gold {question_id}",
                "question_date": "2026/08/09 (Sun) 00:00",
                "haystack_session_ids": session_ids,
                "haystack_dates": ["2026/08/08 (Sat) 00:00"] * count,
                "haystack_sessions": [[] for _ in range(count)],
                "answer_session_ids": [],
            }
        )
    return records


_FIXTURE_DATASET = _build_fixture_dataset()
_FIXTURE_DATASET_BY_ID = {str(record["question_id"]): record for record in _FIXTURE_DATASET}
_FIXTURE_DATASET_BYTES = json.dumps(
    _FIXTURE_DATASET,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
_FIXTURE_DATASET_SHA256 = hashlib.sha256(_FIXTURE_DATASET_BYTES).hexdigest()


@pytest.fixture(autouse=True)
def _pin_fixture_dataset_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa, "LONGMEMEVAL_S_SHA256", _FIXTURE_DATASET_SHA256)
    monkeypatch.setattr(
        official_report,
        "LONGMEMEVAL_S_SHA256",
        _FIXTURE_DATASET_SHA256,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def _replace_receipt_prompt(record: dict[str, object], prompt: str) -> None:
    prompt_bytes = prompt.encode("utf-8")
    binding = record["prompt"]
    assert isinstance(binding, dict)
    binding.update(
        {
            "raw_base64": base64.b64encode(prompt_bytes).decode("ascii"),
            "utf8_bytes": len(prompt_bytes),
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        }
    )
    request = record["provider_request"]
    assert isinstance(request, dict)
    request_bytes = base64.b64decode(str(request["raw_base64"]))
    request_body = json.loads(request_bytes)
    request_body["messages"][0]["content"] = prompt
    forged_request = json.dumps(
        request_body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    request.update(
        {
            "raw_base64": base64.b64encode(forged_request).decode("ascii"),
            "raw_bytes": len(forged_request),
            "raw_sha256": hashlib.sha256(forged_request).hexdigest(),
        }
    )


def _replace_request_control(record: dict[str, object], field: str, value: object) -> None:
    request = record["provider_request"]
    assert isinstance(request, dict)
    body = json.loads(base64.b64decode(str(request["raw_base64"])))
    body[field] = value
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    request.update(
        {
            "raw_base64": base64.b64encode(raw).decode("ascii"),
            "raw_bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _reader_result(
    *, question_id: str, hypothesis: str, model: str, suffix: str, prompt: str
) -> qa.ChatResult:
    request_id = f"chatcmpl-{suffix}-{question_id[1:]}"
    raw_response = json.dumps(
        {
            "id": request_id,
            "model": model,
            "system_fingerprint": "fp_fixture",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": hypothesis},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return qa.chat_result_from_raw_response(
        raw_response,
        prompt=prompt,
        attempts=1,
        latency_ms=12.5,
        request_model=model,
        request_temperature=0.0,
        request_max_tokens=4096,
        endpoint_url="https://reader.example/v1/chat/completions",
    )


def _official_judge_result(
    *, question_id: str, label: bool, suffix: str, prompt: str
) -> qa.ChatResult:
    content = "yes" if label else "no"
    raw_response = json.dumps(
        {
            "id": f"judge-{suffix}-{question_id[1:]}",
            "model": OFFICIAL_JUDGE_MODEL,
            "system_fingerprint": "official-judge-fp-fixture",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 101,
                "completion_tokens": 1,
                "total_tokens": 102,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return qa.chat_result_from_raw_response(
        raw_response,
        prompt=prompt,
        attempts=1,
        latency_ms=20.0,
        request_model=OFFICIAL_JUDGE_MODEL,
        request_temperature=0.0,
        request_max_tokens=10,
        endpoint_url="https://api.openai.com/v1/chat/completions",
    )


def _fingerprint(*, qa_harness: bool) -> dict[str, object]:
    if qa_harness:
        files = {
            "scripts/_longmemeval_common.py": "1" * 64,
            "scripts/run_longmemeval_qa.py": "2" * 64,
            "pyproject.toml": "3" * 64,
            "uv.lock": "4" * 64,
            "src/swarmbrain/retrieval/service.py": "5" * 64,
        }
    else:
        files = {
            "scripts/_longmemeval_common.py": "1" * 64,
            "scripts/evaluate_retrieval_runs.py": "2" * 64,
            "scripts/run_retrieval_eval.py": "3" * 64,
            "pyproject.toml": "4" * 64,
            "uv.lock": "5" * 64,
            "src/swarmbrain/retrieval/service.py": "6" * 64,
        }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"tree_sha256": hashlib.sha256(canonical).hexdigest(), "files": files}


def _retrieval_source(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    case_counts = [qa.LONGMEMEVAL_S_SESSION_COUNT // 500] * 500
    for index in range(qa.LONGMEMEVAL_S_SESSION_COUNT % 500):
        case_counts[index] += 1
    source: dict[str, object] = {
        "artifact_type": qa.RETRIEVAL_RUN_ARTIFACT_TYPE,
        "schema_version": qa.RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": qa.RETRIEVAL_PROTOCOL_VERSION,
        "implementation": _fingerprint(qa_harness=False),
        "track": "longmemeval-s",
        "dataset": {
            "name": "LongMemEval-S",
            "evaluated_questions": 500,
            "total_questions": 500,
            "sha256": qa.LONGMEMEVAL_S_SHA256,
        },
        "granularity": "one memory per haystack session",
        "recall_limit": 10,
        "saved_ranking_depth": 50,
        "dense_lane_enabled": True,
        "temporal_query_routing": {"enabled": False, "parser": None},
        "embedding": {
            "provider": qa.QWEN_EMBEDDING_PROVIDER,
            "model": qa.QWEN_EMBEDDING_MODEL,
            "dimensions": qa.QWEN_EMBEDDING_DIMENSIONS,
            "response_model_requirement": qa.QWEN_EMBEDDING_MODEL,
            "query_instruction_sha256": qa.QWEN_QUERY_INSTRUCTION_SHA256,
        },
        "embedding_call_accounting": {
            "source": "provider-observed",
            "document_inputs": qa.LONGMEMEVAL_S_SESSION_COUNT,
            "document_batch_calls": 500,
            "query_calls": 500,
            "successful_http_calls": 1000,
            "http_attempts": 1000,
        },
        "cases": [
            {
                "case_id": f"q{index:03d}",
                "haystack_sessions": count,
                "degraded_lanes": [],
                "rankings": {"final": []},
                "final_relevance": [],
                "temporal_routing": None,
            }
            for index, count in enumerate(case_counts)
        ],
    }
    source_path = tmp_path / "retrieval-source.json"
    _write_json(source_path, source)
    return source_path, source


def _fixture_pair(
    tmp_path: Path,
    *,
    suffix: str,
    labels: tuple[bool, ...] = (True, False),
    reader_model: str = "reader-v1",
    request_suffix: str | None = None,
) -> tuple[Path, Path, Path]:
    questions = [
        {
            "question_id": record["question_id"],
            "question_type": record["question_type"],
            "hypothesis": f"answer {index:03d}",
            "reader_error": None,
            "retrieved_session_keys": [],
            "retrieved_relevance": [],
            "temporal_routing": None,
        }
        for index, record in enumerate(_FIXTURE_DATASET)
    ]
    chat_receipts: list[dict[str, object]] = []
    for question in questions:
        result = _reader_result(
            question_id=question["question_id"],
            hypothesis=question["hypothesis"],
            model=reader_model,
            suffix=request_suffix or suffix,
            prompt=qa.build_reader_prompt(
                _FIXTURE_DATASET_BY_ID[str(question["question_id"])],
                [],
                style="swarm",
                requested=10,
                floored=False,
            ),
        )
        record = qa.chat_receipt_record(question["question_id"], "reader", result)
        receipt_index = len(chat_receipts)
        chat_receipts.append(record)
        question.update(
            {
                "reader_prompt_tokens": result.prompt_tokens,
                "reader_completion_tokens": result.completion_tokens,
                "reader_total_tokens": result.total_tokens,
                "reader_finish_reason": result.finish_reason,
                "reader_attempts": result.attempts,
                "reader_response_model": result.response_model,
                "reader_request_id": result.request_id,
                "reader_system_fingerprint": result.system_fingerprint,
                "reader_prompt_sha256": result.prompt_sha256,
                "reader_prompt_utf8_bytes": result.prompt_utf8_bytes,
                "reader_raw_response_sha256": result.raw_response_sha256,
                "reader_raw_request_sha256": result.raw_request_sha256,
                "reader_receipt": {
                    "index": receipt_index,
                    "sha256": hashlib.sha256(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                },
                "dev_judge_receipt": None,
            }
        )
    chat_receipt_path = tmp_path / f"chat-receipts-{suffix}.jsonl"
    _write_jsonl(chat_receipt_path, chat_receipts)
    hypothesis_path = tmp_path / f"hypotheses-{suffix}.jsonl"
    _write_jsonl(
        hypothesis_path,
        [
            {
                "question_id": question["question_id"],
                "hypothesis": question["hypothesis"],
            }
            for question in questions
        ],
    )
    source_path, source = _retrieval_source(tmp_path)
    source_artifact = qa.artifact_identity(source_path)
    dataset_path = tmp_path / "longmemeval-s-fixture.json"
    dataset_path.write_bytes(_FIXTURE_DATASET_BYTES)
    run = {
        "artifact_type": qa.QA_RUN_ARTIFACT_TYPE,
        "schema_version": qa.QA_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": qa.QA_PROTOCOL_VERSION,
        "implementation": _fingerprint(qa_harness=True),
        "run_id": f"run-{suffix}",
        "harness": "scripts/run_longmemeval_qa.py",
        "task": "longmemeval-s end-to-end QA",
        "started_at": f"2026-08-09T00:00:0{ord(suffix[0]) % 10}+00:00",
        "dataset": {
            "name": "LongMemEval-S",
            "evaluated_questions": 500,
            "total_questions": 500,
            "sha256": qa.LONGMEMEVAL_S_SHA256,
            "artifact": qa.artifact_identity(dataset_path),
        },
        "judge": {"official_judge_model": OFFICIAL_JUDGE_MODEL},
        "hypotheses": qa.artifact_identity(hypothesis_path)["path"],
        "hypothesis_artifact": qa.artifact_identity(hypothesis_path),
        "chat_receipt_artifact": qa.artifact_identity(chat_receipt_path),
        "chat_receipt_count": len(chat_receipts),
        "chat_receipt_protocol": qa.CHAT_RECEIPT_PROTOCOL_VERSION,
        "reader": {
            "model": reader_model,
            "revision": f"{reader_model}-revision-1",
            "revision_source": "operator-pinned deployment/checkpoint",
            "response_model_requirement": reader_model,
            "request_id_required": True,
            "response_parser": qa.CHAT_RESPONSE_PARSER,
            "request_parser": qa.CHAT_REQUEST_PARSER,
            "raw_request_receipts_required": True,
            "raw_prompt_receipts_required": True,
            "raw_response_receipts_required": True,
            "provider_usage_replay_required": True,
            "response_evidence_publishable": True,
            "temperature": 0.0,
            "max_tokens": 4096,
            "thinking_mode": None,
            "thinking_mode_source": "provider-default-omitted",
            "base_url": "https://reader.example",
            "prompt_style": "swarm",
            "prompt_template_source": "official-flat-session-plus-evidence-only",
        },
        "retrieval": {
            "mode": "replayed_saved_run",
            "source_run": source_artifact["path"],
            "source_artifact": source_artifact,
            "source_artifact_type": source["artifact_type"],
            "source_schema_version": source["schema_version"],
            "source_protocol_version": source["protocol_version"],
            "source_implementation": source["implementation"],
            "source_publishable": True,
            "source_publishability_errors": [],
            "granularity": "one memory per haystack session",
            "source_limit": 10,
            "source_saved_ranking_depth": 50,
            "limit": 10,
            "min_score": 0.0,
            "dense_lane_enabled": True,
            "embedding": source["embedding"],
            "source_embedding_call_accounting": source["embedding_call_accounting"],
            "replay_embedding_call_accounting": {
                "document_inputs": 0,
                "document_batch_calls": 0,
                "query_calls": 0,
                "successful_http_calls": 0,
                "source": "artifact-replay-no-provider-calls",
            },
            "temporal_query_routing": source["temporal_query_routing"],
        },
        "questions": questions,
    }
    official = [
        {
            "question_id": question["question_id"],
            "hypothesis": question["hypothesis"],
            "autoeval_label": {
                "model": OFFICIAL_JUDGE_MODEL,
                "label": labels[index % len(labels)],
            },
        }
        for index, question in enumerate(questions)
    ]
    official_judge_receipts = [
        qa.chat_receipt_record(
            question["question_id"],
            "official_judge",
            _official_judge_result(
                question_id=question["question_id"],
                label=labels[index % len(labels)],
                suffix=suffix,
                prompt=qa.judge_prompt(
                    str(_FIXTURE_DATASET_BY_ID[str(question["question_id"])]["question_type"]),
                    str(_FIXTURE_DATASET_BY_ID[str(question["question_id"])]["question"]),
                    str(_FIXTURE_DATASET_BY_ID[str(question["question_id"])]["answer"]),
                    str(question["hypothesis"]),
                    abstention=qa.is_abstention_question(str(question["question_id"])),
                ),
            ),
        )
        for index, question in enumerate(questions)
    ]
    run_path = tmp_path / f"run-{suffix}.json"
    labels_path = tmp_path / f"labels-{suffix}.jsonl"
    judge_receipts_path = tmp_path / f"official-judge-receipts-{suffix}.jsonl"
    _write_json(run_path, run)
    _write_jsonl(labels_path, official)
    _write_jsonl(judge_receipts_path, official_judge_receipts)
    return run_path, labels_path, judge_receipts_path


def test_build_report_requires_and_aggregates_strict_official_runs(tmp_path: Path) -> None:
    pairs = [
        _fixture_pair(tmp_path, suffix="a", labels=(True, False)),
        _fixture_pair(tmp_path, suffix="b", labels=(True, True)),
        _fixture_pair(tmp_path, suffix="c", labels=(False, True)),
    ]
    runs = [load_official_run(run, labels, receipts) for run, labels, receipts in pairs]

    report = build_report(runs, bootstrap_samples=100, bootstrap_seed=7)

    assert report["artifact_type"] == OFFICIAL_REPORT_ARTIFACT_TYPE
    assert report["schema_version"] == OFFICIAL_REPORT_SCHEMA_VERSION
    assert report["protocol_version"] == OFFICIAL_REPORT_PROTOCOL_VERSION
    assert report["judge"] == {
        "official": True,
        "model": OFFICIAL_JUDGE_MODEL,
        "implementation": (
            "https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py"
        ),
        "raw_prompt_and_response_replay": True,
        "raw_request_and_response_replay": True,
        "prompt_reconstructed_from_bound_dataset": True,
        "receipt_protocol": qa.CHAT_RECEIPT_PROTOCOL_VERSION,
        "temperature": 0.0,
        "max_tokens": 10,
    }
    assert report["comparability"]["exabase_0_964_protocol_comparable"] is False
    assert report["runs"]["count"] == 3
    assert report["generation"]["retrieval"]["limit"] == 10
    assert report["generation"]["retrieval_artifact_type"] == qa.RETRIEVAL_RUN_ARTIFACT_TYPE
    assert report["generation"]["retrieval_source_sha256"] == runs[0].retrieval_source_sha256
    assert report["dataset"]["prompt_source_bound"] is True
    assert (
        report["generation"]["reader_prompt_reconstructed_from_bound_dataset_and_retrieval"] is True
    )
    assert len(report["generation"]["protocol_sha256"]) == 64
    assert len(report["runs"]["items"][0]["generation_run_sha256"]) == 64
    assert len(report["runs"]["items"][0]["generated_hypotheses_sha256"]) == 64
    assert len(report["runs"]["items"][0]["official_judge_receipts_sha256"]) == 64
    assert report["runs"]["items"][0]["official_judge_request_ids"] == 500
    assert report["overall"]["accuracy_mean"] == pytest.approx(2 / 3)
    assert report["overall"]["accuracy_ci95"]["samples"] == 100
    assert report["failures"] == {"unjudged_questions": 0, "reader_failures": 0}


def test_official_report_rejects_partial_or_tampered_labels(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="bad")
    records = [json.loads(line) for line in labels_path.read_text().splitlines()]
    records[0]["hypothesis"] = "tampered"
    _write_jsonl(labels_path, records)

    with pytest.raises(OfficialReportError, match="hypothesis mismatch"):
        load_official_run(run_path, labels_path, judge_receipts_path)

    _write_jsonl(labels_path, records[:-1])
    with pytest.raises(OfficialReportError, match="exactly 500"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_rejects_mixed_readers(tmp_path: Path) -> None:
    first = _fixture_pair(tmp_path, suffix="a", reader_model="reader-a")
    second = _fixture_pair(tmp_path, suffix="b", reader_model="reader-b")
    runs = [load_official_run(*first), load_official_run(*second)]

    with pytest.raises(OfficialReportError, match="fixed reader"):
        build_report(runs, bootstrap_samples=100)


def test_official_report_rejects_mixed_protocols_and_duplicate_runs(tmp_path: Path) -> None:
    first = _fixture_pair(tmp_path, suffix="a")
    second = _fixture_pair(tmp_path, suffix="b")
    second_payload = json.loads(second[0].read_text())
    second_payload["retrieval"]["limit"] = 5
    _write_json(second[0], second_payload)

    with pytest.raises(OfficialReportError, match="fixed retrieval protocol"):
        build_report(
            [load_official_run(*first), load_official_run(*second)],
            bootstrap_samples=100,
        )

    repeated = load_official_run(*first)
    with pytest.raises(OfficialReportError, match="unique run IDs"):
        build_report([repeated, repeated], bootstrap_samples=100)


def test_official_report_rejects_nonofficial_judge(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="judge")
    records = [json.loads(line) for line in labels_path.read_text().splitlines()]
    records[0]["autoeval_label"]["model"] = "gpt-4o-mini"
    _write_jsonl(labels_path, records)

    with pytest.raises(OfficialReportError, match="nonofficial judge"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_derives_label_from_raw_gpt4o_response(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(
        tmp_path,
        suffix="judge-label",
    )
    records = [json.loads(line) for line in labels_path.read_text().splitlines()]
    records[0]["autoeval_label"]["label"] = not records[0]["autoeval_label"]["label"]
    _write_jsonl(labels_path, records)

    with pytest.raises(OfficialReportError, match="differs from replayed GPT-4o response"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_reconstructs_reader_prompt_from_bound_sources(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(
        tmp_path,
        suffix="reader-prompt",
    )
    run = json.loads(run_path.read_text())
    receipt_path = Path(run["chat_receipt_artifact"]["path"])
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    _replace_receipt_prompt(receipts[0], "valid receipt attached to the wrong reader prompt")
    _write_jsonl(receipt_path, receipts)
    run["chat_receipt_artifact"] = qa.artifact_identity(receipt_path)
    run["questions"][0]["reader_receipt"]["sha256"] = hashlib.sha256(
        json.dumps(
            receipts[0],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(run_path, run)

    with pytest.raises(OfficialReportError, match="reader prompt differs from bound dataset"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_reconstructs_official_judge_prompt_from_dataset(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(
        tmp_path,
        suffix="judge-prompt",
    )
    receipts = [json.loads(line) for line in judge_receipts_path.read_text().splitlines()]
    _replace_receipt_prompt(receipts[0], "valid GPT-4o response attached to the wrong prompt")
    _write_jsonl(judge_receipts_path, receipts)

    with pytest.raises(
        OfficialReportError, match="official judge prompt differs from bound dataset"
    ):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_replays_exact_reader_and_judge_request_controls(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(
        tmp_path,
        suffix="request-controls",
    )
    run = json.loads(run_path.read_text())
    receipt_path = Path(run["chat_receipt_artifact"]["path"])
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    _replace_request_control(receipts[0], "temperature", 0.7)
    _write_jsonl(receipt_path, receipts)
    run["chat_receipt_artifact"] = qa.artifact_identity(receipt_path)
    run["questions"][0]["reader_receipt"]["sha256"] = hashlib.sha256(
        json.dumps(
            receipts[0],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(run_path, run)
    with pytest.raises(OfficialReportError, match="raw request controls differ"):
        load_official_run(run_path, labels_path, judge_receipts_path)

    run_path, labels_path, judge_receipts_path = _fixture_pair(
        tmp_path,
        suffix="official-request-controls",
    )
    judge_receipts = [json.loads(line) for line in judge_receipts_path.read_text().splitlines()]
    _replace_request_control(judge_receipts[0], "max_tokens", 11)
    _write_jsonl(judge_receipts_path, judge_receipts)
    with pytest.raises(OfficialReportError, match="official judge raw request controls differ"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_rejects_unbound_or_tampered_retrieval_source(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="source")
    run = json.loads(run_path.read_text())
    del run["retrieval"]["source_artifact"]
    _write_json(run_path, run)
    with pytest.raises(OfficialReportError, match="retrieval source artifact"):
        load_official_run(run_path, labels_path, judge_receipts_path)

    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="tamper")
    run = json.loads(run_path.read_text())
    source_path = Path(run["retrieval"]["source_artifact"]["path"])
    source_path.write_text(source_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(OfficialReportError, match="byte length mismatch|SHA-256 mismatch"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_rejects_symlinked_bound_artifacts(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="symlink")
    run = json.loads(run_path.read_text())
    source_path = Path(run["retrieval"]["source_artifact"]["path"])
    link_path = tmp_path / "retrieval-source-link.json"
    link_path.symlink_to(source_path)
    run["retrieval"]["source_artifact"]["path"] = str(link_path)
    run["retrieval"]["source_run"] = str(link_path)
    _write_json(run_path, run)

    with pytest.raises(OfficialReportError, match="symbolic link"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_rejects_tampered_hypothesis_or_qa_envelope(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="hypothesis")
    run = json.loads(run_path.read_text())
    hypothesis_path = Path(run["hypothesis_artifact"]["path"])
    hypothesis_path.write_text(hypothesis_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(OfficialReportError, match="byte length mismatch|SHA-256 mismatch"):
        load_official_run(run_path, labels_path, judge_receipts_path)

    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="schema")
    run = json.loads(run_path.read_text())
    run["schema_version"] = 1
    _write_json(run_path, run)
    with pytest.raises(OfficialReportError, match="schema_version"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_rejects_generation_bundle_not_in_bound_source(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="bundle")
    run = json.loads(run_path.read_text())
    run["questions"][0]["retrieved_session_keys"] = ["forged-session"]
    _write_json(run_path, run)

    with pytest.raises(OfficialReportError, match="retrieval bundle mismatch"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_requires_reader_response_receipts(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="receipt")
    run = json.loads(run_path.read_text())
    run["questions"][0]["reader_response_model"] = "endpoint-alias"
    _write_json(run_path, run)
    with pytest.raises(OfficialReportError, match="reader receipt field reader_response_model"):
        load_official_run(run_path, labels_path, judge_receipts_path)

    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="revision")
    run = json.loads(run_path.read_text())
    run["reader"]["revision"] = None
    _write_json(run_path, run)
    with pytest.raises(OfficialReportError, match="reader revision"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_report_replays_raw_reader_usage_instead_of_trusting_run_fields(
    tmp_path: Path,
) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="raw-usage")
    run = json.loads(run_path.read_text())
    receipt_path = Path(run["chat_receipt_artifact"]["path"])
    records = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    first = records[0]
    response_bytes = base64.b64decode(first["provider_response"]["raw_base64"])
    response = json.loads(response_bytes)
    response["usage"]["total_tokens"] += 1
    forged = json.dumps(
        response,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    first["provider_response"].update(
        {
            "raw_base64": base64.b64encode(forged).decode(),
            "raw_bytes": len(forged),
            "raw_sha256": hashlib.sha256(forged).hexdigest(),
        }
    )
    _write_jsonl(receipt_path, records)
    run["chat_receipt_artifact"] = qa.artifact_identity(receipt_path)
    run["questions"][0]["reader_receipt"]["sha256"] = hashlib.sha256(
        json.dumps(
            first,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write_json(run_path, run)

    with pytest.raises(OfficialReportError, match="token usage does not reconcile"):
        load_official_run(run_path, labels_path, judge_receipts_path)


def test_official_runs_require_disjoint_reader_request_receipts(tmp_path: Path) -> None:
    first = _fixture_pair(tmp_path, suffix="a")
    second = _fixture_pair(tmp_path, suffix="b", request_suffix="a")

    runs = [load_official_run(*first), load_official_run(*second)]
    with pytest.raises(OfficialReportError, match="not independent"):
        build_report(runs, bootstrap_samples=100)


def test_official_report_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    run_path, labels_path, judge_receipts_path = _fixture_pair(tmp_path, suffix="duplicate")
    original = run_path.read_text()
    run_path.write_text('{"run_id":"shadow",' + original[1:], encoding="utf-8")
    with pytest.raises(OfficialReportError, match="duplicate JSON field"):
        load_official_run(run_path, labels_path, judge_receipts_path)
