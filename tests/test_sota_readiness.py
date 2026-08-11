from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluate_sota_readiness as readiness_module
import pytest
from evaluate_sota_readiness import ManifestError, evaluate_manifest

_FIXTURE_COMPILER = "scripts/build_fixture_report.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _install_fixture_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exit_status: int = 0,
) -> None:
    compiler = tmp_path / _FIXTURE_COMPILER
    compiler.parent.mkdir(parents=True, exist_ok=True)
    compiler.write_text(
        "\n".join(
            (
                "import argparse",
                "import json",
                "import os",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--evidence', type=Path, required=True)",
                "parser.add_argument('--output', type=Path, required=True)",
                "args = parser.parse_args()",
                f"raise SystemExit({exit_status})"
                if exit_status
                else "payload = json.loads(args.evidence.read_text())",
                "if 'payload' in globals():",
                "    report = payload['report']",
                "    report['environment_has_openai_key'] = 'OPENAI_API_KEY' in os.environ",
                "    args.output.write_text(json.dumps(report))",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        readiness_module._SAFE_REPLAY_COMPILERS,
        _FIXTURE_COMPILER,
        frozenset({"--evidence"}),
    )


def _compiler_replay_manifest(
    artifact: str,
    checks: list[dict[str, object]],
) -> dict[str, object]:
    manifest = _manifest(artifact, checks)
    manifest["verification"] = {"require_compiler_replay": True}
    gate = manifest["gates"][0]
    assert isinstance(gate, dict)
    gate["compiler_replay"] = {
        "compiler": _FIXTURE_COMPILER,
        "arguments": ["--evidence", "raw-evidence.json"],
    }
    return manifest


def _manifest(artifact: str, checks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "claim": "fixture claim",
        "frozen_at": "2026-08-09",
        "claim_scope": {
            "claim_sha256": hashlib.sha256(b"fixture claim").hexdigest(),
            "coverage_policy": "every-dimension-covered-by-required-gate",
            "dimensions": ["fixture-evidence"],
        },
        "gates": [
            {
                "id": "fixture",
                "title": "Fixture evidence",
                "required": True,
                "claim_dimensions": ["fixture-evidence"],
                "artifact": artifact,
                "checks": checks,
            }
        ],
    }


def test_repository_manifest_uses_only_allowlisted_offline_compilers() -> None:
    repository_manifest = Path(__file__).resolve().parents[1] / "benchmarks/sota/manifest.json"
    payload = json.loads(repository_manifest.read_text(encoding="utf-8"))

    configured = 0
    for gate in payload["gates"]:
        replay = gate.get("compiler_replay")
        if replay is None:
            continue
        configured += 1
        compiler = replay["compiler"]
        assert compiler in readiness_module._SAFE_REPLAY_COMPILERS
        options = {argument.partition("=")[0] for argument in replay["arguments"]}
        assert readiness_module._SAFE_REPLAY_COMPILERS[compiler] <= options

    assert configured > 0
    assert payload["schema_version"] == 2
    multi_agent = next(
        gate for gate in payload["gates"] if gate["id"] == "multi_agent_causal_scaling"
    )
    assert multi_agent["required"] is True
    assert multi_agent["claim_dimensions"] == ["multi-agent-causal-memory-gain"]
    mem2act = next(gate for gate in payload["gates"] if gate["id"] == "memory_to_action")
    assert mem2act["compiler_replay"] == {
        "compiler": "scripts/build_mem2act_report.py",
        "arguments": [
            "--run",
            "benchmarks/sota/mem2act-run.json",
            "--dataset-dir",
            "benchmarks/sota/evidence/mem2act/dataset",
        ],
    }


def test_readiness_passes_only_when_every_required_check_passes(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "evidence.json",
        {"metrics": {"accuracy": 0.96, "runs": 3, "reader": "fixed"}},
    )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            "evidence.json",
            [
                {"pointer": "/metrics/accuracy", "operator": "gt", "expected": 0.95},
                {"pointer": "/metrics/runs", "operator": "gte", "expected": 3},
                {"pointer": "/metrics/reader", "operator": "nonempty"},
            ],
        ),
    )

    report = evaluate_manifest(manifest, repo_root=tmp_path)

    assert report.ready
    assert report.required_passed == report.required_total == 1
    assert report.claim_sha256 == hashlib.sha256(b"fixture claim").hexdigest()
    assert report.claim_dimensions == ("fixture-evidence",)
    assert report.claim_coverage[0].dimension == "fixture-evidence"
    assert report.claim_coverage[0].required_gate_ids == ("fixture",)
    assert report.gates[0].status == "passed"


def test_claim_scope_rejects_dimension_covered_only_by_informational_gate(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "evidence.json", {"metric": 1})
    payload = _manifest(
        "evidence.json",
        [{"pointer": "/metric", "operator": "eq", "expected": 1}],
    )
    gate = payload["gates"][0]
    assert isinstance(gate, dict)
    gate["required"] = False
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, payload)

    with pytest.raises(
        ManifestError,
        match="claim_scope dimensions must each be covered by a required gate: fixture-evidence",
    ):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_claim_scope_rejects_gate_dimension_outside_declared_scope(tmp_path: Path) -> None:
    _write_json(tmp_path / "evidence.json", {"metric": 1})
    payload = _manifest(
        "evidence.json",
        [{"pointer": "/metric", "operator": "eq", "expected": 1}],
    )
    gate = payload["gates"][0]
    assert isinstance(gate, dict)
    gate["claim_dimensions"] = ["undeclared-dimension"]
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, payload)

    with pytest.raises(
        ManifestError,
        match="gate claim_dimensions are absent from manifest claim_scope: undeclared-dimension",
    ):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_claim_scope_digest_rejects_broadened_claim_text(tmp_path: Path) -> None:
    _write_json(tmp_path / "evidence.json", {"metric": 1})
    payload = _manifest(
        "evidence.json",
        [{"pointer": "/metric", "operator": "eq", "expected": 1}],
    )
    payload["claim"] = "fixture claim plus an untested robotics frontier"
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, payload)

    with pytest.raises(
        ManifestError,
        match="claim_scope claim_sha256 does not bind the exact claim text",
    ):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_legacy_manifest_cannot_bypass_claim_scope_coverage(tmp_path: Path) -> None:
    _write_json(tmp_path / "evidence.json", {"metric": 1})
    payload = _manifest(
        "evidence.json",
        [{"pointer": "/metric", "operator": "eq", "expected": 1}],
    )
    payload["schema_version"] = 1
    payload.pop("claim_scope")
    gate = payload["gates"][0]
    assert isinstance(gate, dict)
    gate.pop("claim_dimensions")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, payload)

    with pytest.raises(ManifestError, match="unsupported readiness manifest schema_version"):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_required_current_tree_compiler_replay_reproduces_exact_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fixture_compiler(tmp_path, monkeypatch)
    report_payload = {
        "metrics": {"accuracy": 0.96},
        "environment_has_openai_key": False,
    }
    _write_json(tmp_path / "raw-evidence.json", {"report": report_payload})
    _write_json(tmp_path / "evidence.json", report_payload)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _compiler_replay_manifest(
            "evidence.json",
            [{"pointer": "/metrics/accuracy", "operator": "gte", "expected": 0.95}],
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-offline-compiler")

    result = evaluate_manifest(manifest, repo_root=tmp_path)

    assert result.ready
    assert result.compiler_replay_required is True
    replay = result.gates[0].compiler_replay
    assert replay is not None
    assert replay.status == "passed"
    assert replay.compiler == _FIXTURE_COMPILER
    assert (
        replay.compiler_sha256
        == hashlib.sha256((tmp_path / _FIXTURE_COMPILER).read_bytes()).hexdigest()
    )
    assert replay.artifact_sha256 == replay.replay_sha256


def test_compiler_replay_rejects_a_static_report_that_current_compiler_did_not_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fixture_compiler(tmp_path, monkeypatch)
    source_report = {"metrics": {"accuracy": 0.96}, "environment_has_openai_key": False}
    supplied_report = {**source_report, "unsupported_claim": "injected"}
    _write_json(tmp_path / "raw-evidence.json", {"report": source_report})
    _write_json(tmp_path / "evidence.json", supplied_report)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _compiler_replay_manifest(
            "evidence.json",
            [{"pointer": "/metrics/accuracy", "operator": "gte", "expected": 0.95}],
        ),
    )

    result = evaluate_manifest(manifest, repo_root=tmp_path)

    assert not result.ready
    assert result.gates[0].status == "failed"
    replay = result.gates[0].compiler_replay
    assert replay is not None
    assert replay.status == "failed"
    assert replay.artifact_sha256 != replay.replay_sha256


def test_required_compiler_replay_fails_closed_when_gate_has_no_replay(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "evidence.json", {"metric": 1})
    payload = _manifest(
        "evidence.json",
        [{"pointer": "/metric", "operator": "eq", "expected": 1}],
    )
    payload["verification"] = {"require_compiler_replay": True}
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, payload)

    result = evaluate_manifest(manifest, repo_root=tmp_path)

    assert not result.ready
    assert result.gates[0].status == "failed"
    assert result.gates[0].compiler_replay is None
    assert result.gates[0].message == "compiler replay is required but not configured"


def test_compiler_replay_errors_fail_closed_without_trusting_static_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fixture_compiler(tmp_path, monkeypatch, exit_status=7)
    report_payload = {"metric": 1, "environment_has_openai_key": False}
    _write_json(tmp_path / "raw-evidence.json", {"report": report_payload})
    _write_json(tmp_path / "evidence.json", report_payload)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _compiler_replay_manifest(
            "evidence.json",
            [{"pointer": "/metric", "operator": "eq", "expected": 1}],
        ),
    )

    result = evaluate_manifest(manifest, repo_root=tmp_path)

    assert not result.ready
    assert result.gates[0].status == "error"
    replay = result.gates[0].compiler_replay
    assert replay is not None
    assert replay.status == "error"
    assert replay.message == "offline compiler replay exited with status 7"


@pytest.mark.parametrize(
    ("compiler", "arguments", "error"),
    (
        (
            "scripts/run_retrieval_eval.py",
            ["--evidence", "raw-evidence.json"],
            "offline allowlist",
        ),
        (
            _FIXTURE_COMPILER,
            ["--evidence", "/private/tmp/raw-evidence.json"],
            "absolute paths",
        ),
        (
            _FIXTURE_COMPILER,
            ["--evidence", "https://example.invalid/evidence.json"],
            "endpoint URLs",
        ),
        (
            _FIXTURE_COMPILER,
            ["--evidence", "../raw-evidence.json"],
            "escape",
        ),
        (
            _FIXTURE_COMPILER,
            ["--evidence", "evidence.json"],
            "final report",
        ),
        (
            _FIXTURE_COMPILER,
            ["--evidence", "raw-evidence.json", "--output", "other.json"],
            "output path internally",
        ),
    ),
)
def test_compiler_replay_rejects_unsafe_or_endpoint_capable_invocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiler: str,
    arguments: list[str],
    error: str,
) -> None:
    _install_fixture_compiler(tmp_path, monkeypatch)
    report_payload = {"metric": 1, "environment_has_openai_key": False}
    _write_json(tmp_path / "raw-evidence.json", {"report": report_payload})
    _write_json(tmp_path / "evidence.json", report_payload)
    payload = _manifest(
        "evidence.json",
        [{"pointer": "/metric", "operator": "eq", "expected": 1}],
    )
    payload["verification"] = {"require_compiler_replay": True}
    gate = payload["gates"][0]
    assert isinstance(gate, dict)
    gate["compiler_replay"] = {"compiler": compiler, "arguments": arguments}
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, payload)

    with pytest.raises(ManifestError, match=error):
        evaluate_manifest(manifest, repo_root=tmp_path)


@pytest.mark.parametrize(("actual", "expected"), ((1, True), (True, 1), (0, False)))
def test_equality_checks_do_not_conflate_json_booleans_and_numbers(
    tmp_path: Path, actual: object, expected: object
) -> None:
    _write_json(tmp_path / "evidence.json", {"value": actual})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            "evidence.json",
            [{"pointer": "/value", "operator": "eq", "expected": expected}],
        ),
    )

    assert not evaluate_manifest(manifest, repo_root=tmp_path).ready


def test_missing_evidence_fails_closed_without_reading_outside_repo(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest("missing.json", [{"pointer": "", "operator": "exists"}]),
    )

    report = evaluate_manifest(manifest, repo_root=tmp_path)

    assert not report.ready
    assert report.required_passed == 0
    assert report.gates[0].status == "missing"

    _write_json(
        manifest,
        _manifest("../outside.json", [{"pointer": "", "operator": "exists"}]),
    )
    with pytest.raises(ManifestError, match="escape"):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_failed_and_missing_pointer_checks_are_auditable(tmp_path: Path) -> None:
    _write_json(tmp_path / "evidence.json", {"metrics": {"accuracy": 0.90}})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            "evidence.json",
            [
                {"pointer": "/metrics/accuracy", "operator": "gte", "expected": 0.95},
                {"pointer": "/metrics/latency", "operator": "exists"},
            ],
        ),
    )

    report = evaluate_manifest(manifest, repo_root=tmp_path)

    assert not report.ready
    assert report.gates[0].status == "failed"
    assert [check.message for check in report.gates[0].checks] == [
        "comparison failed",
        "JSON pointer is missing",
    ]


def test_artifact_checks_bind_a_report_to_exact_repository_bytes(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_bytes(b'{"run":1}\n')
    digest = hashlib.sha256(run_path.read_bytes()).hexdigest()
    _write_json(
        tmp_path / "evidence.json",
        {
            "run_artifact": {
                "path": "run.json",
                "sha256": digest,
                "bytes": run_path.stat().st_size,
            }
        },
    )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            "evidence.json",
            [
                {
                    "pointer": "/run_artifact/path",
                    "operator": "artifact_sha256",
                    "expected_pointer": "/run_artifact/sha256",
                },
                {
                    "pointer": "/run_artifact/path",
                    "operator": "artifact_bytes",
                    "expected_pointer": "/run_artifact/bytes",
                },
            ],
        ),
    )

    assert evaluate_manifest(manifest, repo_root=tmp_path).ready
    run_path.write_bytes(b'{"run":2}\n')
    report = evaluate_manifest(manifest, repo_root=tmp_path)
    assert not report.ready
    assert report.gates[0].checks[0].message == "bound artifact comparison failed"


def test_artifact_checks_reject_paths_outside_the_repository(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "evidence.json",
        {"run_artifact": {"path": "../outside.json", "sha256": "0" * 64}},
    )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            "evidence.json",
            [
                {
                    "pointer": "/run_artifact/path",
                    "operator": "artifact_sha256",
                    "expected_pointer": "/run_artifact/sha256",
                }
            ],
        ),
    )

    with pytest.raises(ManifestError, match="escape"):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_gate_artifact_rejects_symbolic_links(tmp_path: Path) -> None:
    _write_json(tmp_path / "real.json", {"metric": 1})
    (tmp_path / "linked.json").symlink_to(tmp_path / "real.json")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest("linked.json", [{"pointer": "/metric", "operator": "eq", "expected": 1}]),
    )

    with pytest.raises(ManifestError, match="symbolic link"):
        evaluate_manifest(manifest, repo_root=tmp_path)


@pytest.mark.parametrize("value", [True, "0.96"])
def test_ordered_checks_reject_non_finite_or_non_numeric_values(
    tmp_path: Path, value: object
) -> None:
    _write_json(tmp_path / "evidence.json", {"metric": value})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            "evidence.json",
            [{"pointer": "/metric", "operator": "gte", "expected": 0.95}],
        ),
    )

    with pytest.raises(ManifestError, match="finite number"):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_duplicate_gate_ids_are_rejected(tmp_path: Path) -> None:
    _write_json(tmp_path / "evidence.json", {})
    gate = _manifest("evidence.json", [{"pointer": "", "operator": "exists"}])["gates"][0]
    manifest_payload = {
        "schema_version": 2,
        "claim": "fixture claim",
        "frozen_at": "2026-08-09",
        "claim_scope": {
            "claim_sha256": hashlib.sha256(b"fixture claim").hexdigest(),
            "coverage_policy": "every-dimension-covered-by-required-gate",
            "dimensions": ["fixture-evidence"],
        },
        "gates": [gate, gate],
    }
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, manifest_payload)

    with pytest.raises(ManifestError, match="unique"):
        evaluate_manifest(manifest, repo_root=tmp_path)


@pytest.mark.parametrize(
    "raw_artifact",
    ('{"metric": 1, "metric": 2}', '{"metric": NaN}'),
)
def test_evidence_artifacts_reject_duplicate_fields_and_nonfinite_numbers(
    tmp_path: Path, raw_artifact: str
) -> None:
    (tmp_path / "evidence.json").write_text(raw_artifact, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest("evidence.json", [{"pointer": "/metric", "operator": "exists"}]),
    )

    report = evaluate_manifest(manifest, repo_root=tmp_path)

    assert not report.ready
    assert report.gates[0].status == "error"
    assert "StrictJsonError" in report.gates[0].message


def test_manifest_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,"claim":"x","frozen_at":"now","gates":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="StrictJsonError"):
        evaluate_manifest(manifest, repo_root=tmp_path)


def test_repository_gatemem_gate_rejects_degenerate_metric_evidence(
    tmp_path: Path,
) -> None:
    repository_manifest = Path(__file__).resolve().parents[1] / "benchmarks/sota/manifest.json"
    payload = json.loads(repository_manifest.read_text(encoding="utf-8"))
    gate = next(item for item in payload["gates"] if item["id"] == "governance_and_forgetting")
    configured = {
        (check["pointer"], check["operator"]): check.get("expected") for check in gate["checks"]
    }
    assert configured[("/metrics/utility", "gte")] == 0.70
    assert configured[("/metrics/memory_governance_score", "gte")] == 0.70
    assert configured[("/metrics/over_refusal_rate", "lte")] == 0.248
    assert configured[("/metrics/tokens_per_checkpoint", "lte")] == 1210
    assert configured[("/evaluation/answer_model", "eq")] == "gpt-4o-mini"
    assert configured[("/evaluation/judge_model", "eq")] == "gpt-4o"
    assert not any(
        check["operator"] == "exists" and check["pointer"].startswith("/metrics/")
        for check in gate["checks"]
    )

    evidence_path = tmp_path / "gatemem.json"
    _write_json(
        evidence_path,
        {
            "metrics": {
                "access_control_violation_rate": 0,
                "active_forgetting_failure_rate": 0,
                "utility": 0.01,
                "memory_governance_score": 0.01,
                "over_refusal_rate": 0.99,
                "tokens_per_checkpoint": 1,
            }
        },
    )
    isolated_gate = dict(gate)
    isolated_gate.pop("compiler_replay", None)
    isolated_gate["artifact"] = evidence_path.name
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 2,
            "claim": "GateMem fixture",
            "frozen_at": payload["frozen_at"],
            "claim_scope": {
                "claim_sha256": hashlib.sha256(b"GateMem fixture").hexdigest(),
                "coverage_policy": "every-dimension-covered-by-required-gate",
                "dimensions": gate["claim_dimensions"],
            },
            "gates": [isolated_gate],
        },
    )

    report = evaluate_manifest(manifest, repo_root=tmp_path)

    assert not report.ready
    failed = {
        (check.pointer, check.operator) for check in report.gates[0].checks if not check.passed
    }
    assert ("/metrics/utility", "gte") in failed
    assert ("/metrics/memory_governance_score", "gte") in failed
    assert ("/metrics/over_refusal_rate", "lte") in failed
