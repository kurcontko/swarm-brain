from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import run_longmemeval_e6c_v2_merged_lane_external as v2


def _context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=tmp_path,
        carrier=object(),
        manifest={"artifact_sha256": "manifest"},
    )


def test_qa_state_includes_completion_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(v2.e6b, "_qa_durable_state_exists", lambda _carrier: False)
    assert v2._qa_state_exists(context) is False
    (tmp_path / "qa-completion.json").write_text("{}", encoding="utf-8")
    assert v2._qa_state_exists(context) is True


def test_orphan_qa_completion_is_rejected_before_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    (tmp_path / "qa-completion.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(v2.v1, "_qa_artifact_count", lambda _context: 0)
    with pytest.raises(v2.E6CV2Error, match="completion exists before"):
        v2._validate_existing_qa_completion(context)


def test_terminal_report_refuses_passing_g1_without_qa(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        v2,
        "load_pre_qa_gates",
        lambda _context: {"gates": {"G1": {"passed": True}}},
    )
    monkeypatch.setattr(v2.v1, "_qa_artifact_count", lambda _context: 0)
    with pytest.raises(v2.E6CV2Error, match="completed conditional QA"):
        v2.build_report(context)
    assert not (tmp_path / "report.json").exists()


def test_complete_qa_finalizes_offline_before_api_key_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    context.e1 = SimpleNamespace(deepseek_root=tmp_path, selected=())
    monkeypatch.setattr(v2, "_validate_existing_qa_completion", lambda _context: None)
    monkeypatch.setattr(v2, "_qa_state_exists", lambda _context: True)
    monkeypatch.setattr(
        v2,
        "load_pre_qa_gates",
        lambda _context: {"gates": {"G1": {"passed": True}}},
    )
    monkeypatch.setattr(v2.v1, "_qa_artifact_count", lambda _context: v2.v1.E6C_SAMPLE)
    monkeypatch.setattr(v2.v1.e1, "_snapshot_artifact", lambda *_args: "snapshot")
    tokenizer = object()
    monkeypatch.setattr(v2.v1.e1, "DeepSeekExactTokenizer", lambda *_args, **_kwargs: tokenizer)
    expected = {"gates": {"G2": {"passed": True}}}
    monkeypatch.setattr(
        v2,
        "_finalize_complete_qa_offline",
        lambda _context, *, tokenizer: expected,
    )
    monkeypatch.setattr(
        v2.os,
        "getenv",
        lambda *_args: (_ for _ in ()).throw(AssertionError("API key lookup is forbidden")),
    )
    assert (
        v2.run_qa_phase(context, base_url="https://invalid.example", api_key_env="MISSING")
        == expected
    )


def test_provider_request_ids_are_unique() -> None:
    seen: set[str] = set()
    v2._register_provider_request_id(seen, "request-1", route="route-1")
    with pytest.raises(v2.E6CV2Error, match="crosses routes"):
        v2._register_provider_request_id(seen, "request-1", route="route-2")


def test_extraction_sidecar_binding_rejects_extra_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    question = SimpleNamespace(turns=(object(), object()))
    context = SimpleNamespace(
        output_dir=tmp_path,
        carrier=object(),
        e1=SimpleNamespace(selected=(question,)),
    )
    root = tmp_path / "extraction-values" / "question"
    root.mkdir(parents=True)

    def value_path(_carrier: object, _question: object, position: int) -> Path:
        return root / f"{position:05d}.json"

    def attempt_path(_carrier: object, _question: object, position: int) -> Path:
        return root / f"{position:05d}.attempts.jsonl"

    monkeypatch.setattr(v2.e6b, "_value_record_path", value_path)
    monkeypatch.setattr(v2.e6b, "_value_attempt_path", attempt_path)
    for position in range(2):
        value_path(None, None, position).write_text(f"value-{position}", encoding="utf-8")
        attempt_path(None, None, position).write_text(f"attempt-{position}\n", encoding="utf-8")

    binding = v2._extraction_sidecar_binding(context)
    assert binding["value_files"] == 2
    assert binding["attempt_ledger_files"] == 2

    (root / "extra.json").write_text("extra", encoding="utf-8")
    with pytest.raises(v2.E6CV2Error, match="exact domain"):
        v2._extraction_sidecar_binding(context)
