from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.integrations.memoryarena import preflight
from benchmarks.integrations.memoryarena.contracts import PAPER_TABLE_DOMAIN_TASK_GROUPS


def _write_valid_dataset_manifest(
    root: Path,
) -> tuple[Path, dict[str, object], Path]:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    counts = dict(PAPER_TABLE_DOMAIN_TASK_GROUPS)
    counts["bundled_shopping"] += (
        preflight.PAPER_DECLARED_TASK_GROUPS - preflight.PAPER_TABLE_COMPONENT_TOTAL
    )
    configs: dict[str, dict[str, object]] = {}
    for name in sorted(preflight.OFFICIAL_DATASET_CONFIGS):
        artifact = artifacts / f"{name}.jsonl"
        artifact.write_text(f'{{"config":"{name}"}}\n', encoding="utf-8")
        configs[name] = {
            "path": artifact.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "task_group_count": counts[name],
        }

    reconciliation = root / "protocol-reconciliation.json"
    reconciliation.write_text('{"resolution":"paper-and-dataset-audit"}\n', encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": preflight.OFFICIAL_DATASET,
        "revision": "a" * 40,
        "split": preflight.OFFICIAL_DATASET_SPLIT,
        "configs": configs,
        "protocol_reconciliation": {
            "paper_declared_task_groups": preflight.PAPER_DECLARED_TASK_GROUPS,
            "paper_table_component_total": preflight.PAPER_TABLE_COMPONENT_TOTAL,
            "resolved_task_groups": preflight.PAPER_DECLARED_TASK_GROUPS,
            "resolution_path": reconciliation.name,
            "resolution_sha256": hashlib.sha256(reconciliation.read_bytes()).hexdigest(),
        },
    }
    manifest = root / "dataset-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload, reconciliation


def test_pinned_bridge_preflight_passes_code_identity_but_fails_official_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkout = tmp_path / "MemoryArena"
    checkout.mkdir()
    pinned: dict[str, str] = {}
    for relative in (
        "README.md",
        "memory/client.py",
        "memory/server.py",
    ):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"fixture:{relative}\n".encode()
        path.write_bytes(payload)
        pinned[relative] = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(preflight, "OFFICIAL_CHECKOUT_SHA256", pinned)

    def fake_git(_checkout: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return preflight.PINNED_REPOSITORY_COMMIT
        assert args == ("status", "--porcelain", "--untracked-files=all")
        return ""

    monkeypatch.setattr(preflight, "_git_value", fake_git)
    report = preflight.run_preflight(checkout)

    assert report.bridge_compatible is True
    assert report.official_protocol_ready is False
    assert report.official_execution_supported is False
    assert any(check.name == "official_result_compiler_available" for check in report.checks)
    assert report.benchmark["paper_declared_task_groups"] == 766
    assert report.benchmark["paper_table_component_total"] == 736
    assert report.benchmark["custom_causal_1_2_4"] is False


def test_untracked_checkout_file_fails_clean_tree_gate(tmp_path: Path, monkeypatch) -> None:
    checkout = tmp_path / "MemoryArena"
    checkout.mkdir()
    monkeypatch.setattr(preflight, "OFFICIAL_CHECKOUT_SHA256", {})
    calls: list[tuple[str, ...]] = []

    def fake_git(_checkout: Path, *args: str) -> str:
        calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return preflight.PINNED_REPOSITORY_COMMIT
        assert args == ("status", "--porcelain", "--untracked-files=all")
        return "?? injected.py"

    monkeypatch.setattr(preflight, "_git_value", fake_git)
    report = preflight.run_preflight(checkout)
    checks = {check.name: check for check in report.checks}

    assert calls[-1] == ("status", "--porcelain", "--untracked-files=all")
    assert checks["official_tracked_tree_clean"].passed is False
    assert report.bridge_compatible is False


def test_checkout_symlink_component_fails_before_git_or_hashing(tmp_path: Path) -> None:
    target = tmp_path / "real-checkout"
    target.mkdir()
    checkout = tmp_path / "checkout-link"
    checkout.symlink_to(target, target_is_directory=True)

    report = preflight.run_preflight(checkout)
    checks = {check.name: check for check in report.checks}

    assert checks["official_checkout_path_safe"].passed is False
    assert checks["official_repository_commit"].actual is None
    assert report.bridge_compatible is False


def test_config_parent_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    real = tmp_path / "real-configs"
    real.mkdir()
    config = real / "shopping.json"
    config.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked-configs"
    linked.symlink_to(real, target_is_directory=True)

    check = preflight._validate_configs((linked / config.name,))

    assert check.passed is False
    assert check.actual["failures"] == ["shopping.json:MemoryArenaPreflightError"]


def test_dataset_manifest_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    manifest, _payload, _reconciliation = _write_valid_dataset_manifest(tmp_path)
    linked_manifest = tmp_path / "linked-manifest.json"
    linked_manifest.symlink_to(manifest)

    check, total, reconciled = preflight._validate_dataset_manifest(linked_manifest)

    assert check.passed is False
    assert "symbolic-link component" in check.actual["failure"]
    assert total is None
    assert reconciled is False


def test_dataset_artifact_symlink_component_is_rejected(tmp_path: Path) -> None:
    manifest, payload, _reconciliation = _write_valid_dataset_manifest(tmp_path)
    linked_artifacts = tmp_path / "linked-artifacts"
    linked_artifacts.symlink_to(tmp_path / "artifacts", target_is_directory=True)
    configs = payload["configs"]
    assert isinstance(configs, dict)
    row = configs["bundled_shopping"]
    assert isinstance(row, dict)
    row["path"] = "linked-artifacts/bundled_shopping.jsonl"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    check, _total, reconciled = preflight._validate_dataset_manifest(manifest)

    assert check.passed is False
    assert "symbolic-link component" in check.actual["failure"]
    assert reconciled is False


def test_reconciliation_artifact_symlink_is_rejected(tmp_path: Path) -> None:
    manifest, payload, reconciliation = _write_valid_dataset_manifest(tmp_path)
    linked_reconciliation = tmp_path / "linked-reconciliation.json"
    linked_reconciliation.symlink_to(reconciliation)
    resolution = payload["protocol_reconciliation"]
    assert isinstance(resolution, dict)
    resolution["resolution_path"] = linked_reconciliation.name
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    check, _total, reconciled = preflight._validate_dataset_manifest(manifest)

    assert check.passed is False
    assert "symbolic-link component" in check.actual["failure"]
    assert reconciled is False


def test_dataset_and_reconciliation_regular_paths_validate(tmp_path: Path) -> None:
    manifest, _payload, reconciliation = _write_valid_dataset_manifest(tmp_path)

    check, total, reconciled = preflight._validate_dataset_manifest(manifest)

    assert check.passed is True
    assert total == preflight.PAPER_DECLARED_TASK_GROUPS
    assert reconciled is True
    assert (
        check.actual["reconciliation_artifact_sha256"]
        == hashlib.sha256(reconciliation.read_bytes()).hexdigest()
    )


def test_manifest_fragment_is_a_real_replacement_and_remains_fail_closed() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/integrations/memoryarena/manifest-replacement-fragment.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["operation"] == "replace_required_gate"
    assert payload["replace_gate_id"] == "multi_agent_causal_scaling"
    assert payload["scientific_scope_equivalent"] is False
    gate = payload["gate"]
    assert gate["id"] == "memoryarena_official"
    checks = {row["pointer"]: row for row in gate["checks"]}
    assert checks["/benchmark/repository_commit"]["expected"] == (
        "6cd9de14b71915e39ac742a20dc33785e14b6aab"
    )
    assert checks["/protocol/paper_declared_task_groups"]["expected"] == 766
    assert checks["/protocol/custom_causal_1_2_4"]["expected"] is False
    assert checks["/dataset/immutable_revision_verified"]["expected"] is True
    assert checks["/evaluation/official_sr_ps_compiler_verified"]["expected"] is True
    assert checks["/readiness/upstream_preview_boundary_resolved"]["expected"] is True
