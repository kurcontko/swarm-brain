from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from benchmarks.integrations.gatemem.completion import (
    build_execution_lineage,
    default_completion_path,
    write_completion_manifest,
)
from benchmarks.integrations.gatemem.contracts import (
    GATEMEM_COMMIT,
    GateMemCheckout,
    GateMemContractError,
)
from benchmarks.integrations.gatemem.report import (
    DOMAIN_COUNTS,
    DomainEvidence,
    build_gatemem_report,
)

CHECKOUT = Path("/private/tmp/swarmbrain-gatemem")


def _readable_git_checkout(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture_implementation_fingerprint() -> dict[str, Any]:
    files = {
        "benchmarks/integrations/gatemem/completion.py": "1" * 64,
        "benchmarks/integrations/gatemem/resume.py": "2" * 64,
        "scripts/run_gatemem_external.py": "3" * 64,
        "src/swarmbrain/application/runtime.py": "4" * 64,
        "pyproject.toml": "5" * 64,
        "uv.lock": "6" * 64,
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _domain_fixture(tmp_path: Path, checkout: GateMemCheckout, domain: str) -> DomainEvidence:
    dataset = checkout.load(domain)
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    judges: list[dict[str, Any]] = []
    counts = {"utility": 0, "privacy": 0, "safety": 0}
    for checkpoint in dataset.checkpoints:
        checkpoint_id = str(checkpoint["checkpoint_id"])
        query_type = str(checkpoint["query_type"])
        counts[query_type] += 1
        action = str(checkpoint["expected_action"])
        predictions.append(
            {
                "checkpoint_id": checkpoint_id,
                "output": {
                    "action": action,
                    "answer": "fixture answer",
                    "answer_structured": {},
                    "used_record_ids": [],
                    "memory_audit": {
                        "schema_version": 1,
                        "stage": "prompt_context",
                        "context_format": "swarmbrain-json-v1",
                        "prompt_context": {
                            "text": "[]",
                            "n_chars": 2,
                            "n_items": 0,
                            "items": [],
                        },
                    },
                    "llm_usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    },
                },
                "swarmbrain_audit": {
                    "schema_version": 1,
                    "gatemem_commit": GATEMEM_COMMIT,
                    "episode_id": checkpoint["episode_id"],
                    "as_of_turn_id": checkpoint["as_of_turn_id"],
                    "tokens": {
                        "provider_usage_reported": True,
                        "provider_input": 10,
                        "provider_output": 2,
                        "context_estimated": 0,
                        "usage_source": "provider",
                    },
                    "latency_ms": {"query_total": 1.0},
                    "answer_model": {
                        "provider": "fixture",
                        "model": "reader-v1",
                        "revision": "reader-v1-revision",
                    },
                },
            }
        )
        scores.append(
            {
                "checkpoint_id": checkpoint_id,
                "query_type": query_type,
                "action_correct": True,
                "utility_correct": True if query_type == "utility" else None,
                "privacy_e2e_leak": False if query_type == "privacy" else None,
                "deletion_e2e_leak": False if query_type == "safety" else None,
                "over_refusal": False,
            }
        )
        judges.append(
            {
                "checkpoint_id": checkpoint_id,
                "query_type": query_type,
                "judge": {
                    "action_ok": True,
                    "utility_ok": True if query_type == "utility" else None,
                    "privacy_leak": False if query_type == "privacy" else None,
                    "deletion_leak": False if query_type == "safety" else None,
                    "parse_ok": True,
                    "gated_by_action": True,
                },
                "llm": {"provider": "fixture", "model": "judge-v1"},
            }
        )

    root = tmp_path / domain
    predictions_path = root / "predictions.jsonl"
    audit_path = root / "audit.json"
    scorer_dir = root / "scores"
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(scorer_dir / "predictions.normalized.jsonl", predictions)
    _write_jsonl(scorer_dir / "scores.jsonl", scores)
    _write_jsonl(scorer_dir / "judge_scores.jsonl", judges)
    _write_json(
        audit_path,
        {
            "schema_version": 1,
            "gatemem_commit": GATEMEM_COMMIT,
            "episodes": DOMAIN_COUNTS[domain]["episodes"],
            "checkpoints": DOMAIN_COUNTS[domain]["checkpoints"],
            "config": {"context_token_budget": 4096},
            "audience_policy": {
                "type": "ManifestAudiencePolicy",
                "manifest_sha256": "a" * 64,
            },
            "ingest_operations": [],
        },
    )
    write_completion_manifest(
        default_completion_path(predictions_path),
        domain=domain,
        predictions_path=predictions_path,
        audit_path=audit_path,
        execution_lineage=build_execution_lineage(
            mode="uninterrupted",
            completed_prefix_episodes=0,
            completed_episodes=DOMAIN_COUNTS[domain]["episodes"],
            authenticated_state_payload_sha256=None,
            implementation_fingerprint=_fixture_implementation_fingerprint(),
        ),
    )
    _write_json(
        scorer_dir / "summary.json",
        {
            "gated_by_action": True,
            "n_checkpoints": len(predictions),
            "utility_accuracy": 1.0,
            "privacy_leakage_rate": 0.0,
            "deletion_leakage_rate": 0.0,
            "privacy_e2e_leakage_rate": 0.0,
            "deletion_e2e_leakage_rate": 0.0,
            "over_refusal_rate": 0.0,
            "llm": {
                "tokens": {
                    "input_tokens": 10 * len(predictions),
                    "output_tokens": 2 * len(predictions),
                    "total_tokens": 12 * len(predictions),
                    "avg_total_tokens_per_ckpt": 12.0,
                }
            },
            "llm_judge": {
                "gated_by_action": True,
                "n_judged": len(predictions),
                "n_action_scored": len(predictions),
                "n_utility": counts["utility"],
                "n_utility_scored": counts["utility"],
                "n_privacy": counts["privacy"],
                "n_privacy_scored": counts["privacy"],
                "n_safety": counts["safety"],
                "n_safety_scored": counts["safety"],
                "judge_parse_failure_rate": 0.0,
                "judge_effective_utility_accuracy": 1.0,
                "judge_privacy_leakage_rate": 0.0,
                "judge_deletion_leakage_rate": 0.0,
                "llm": {"provider": "fixture", "model": "judge-v1"},
            },
        },
    )
    return DomainEvidence.create(
        domain=domain,
        predictions_path=predictions_path,
        audit_path=audit_path,
        scorer_dir=scorer_dir,
    )


@pytest.mark.skipif(
    not _readable_git_checkout(CHECKOUT),
    reason="pinned GateMem checkout absent or unreadable",
)
def test_report_compiler_requires_full_pinned_official_coverage(tmp_path: Path) -> None:
    checkout = GateMemCheckout(CHECKOUT)
    evidence = tuple(
        _domain_fixture(tmp_path, checkout, domain) for domain in sorted(DOMAIN_COUNTS)
    )
    report = build_gatemem_report(checkout=checkout, evidence=evidence)
    assert report["dataset"]["episodes"] == 91
    assert report["dataset"]["checkpoints"] == 2218
    assert report["metrics"] == {
        "access_control_violation_rate": 0.0,
        "active_forgetting_failure_rate": 0.0,
        "utility": 1.0,
        "over_refusal_rate": 0.0,
        "tokens_per_checkpoint": 12.0,
        "memory_governance_score": 1.0,
    }
    assert report["published_comparison"]["quality"]["all_domain_mgs_reference"][
        "value"
    ] == pytest.approx(0.6967356878307394)
    assert report["published_comparison"]["comparability"] == {
        "reference_kind": "cross-system composite envelope",
        "same_system_reproduction": False,
        "interpretation": (
            "Swarm Brain is evaluated with the pinned official GateMem scorer against "
            "independently published quality, over-refusal, and token frontiers; the "
            "envelope is not one published system configuration"
        ),
    }
    assert report["published_comparison"]["tokens"]["domain_tokens_per_checkpoint"] == {
        "education": 1380,
        "household": 1180,
        "medical": 1050,
        "office": 1240,
    }
    assert report["failures"]["unjudged_checkpoints"] == 0
    assert all(len(value) == 64 for value in report["evaluation"]["audience_manifests"].values())
    completion = json.loads(evidence[0].completion_manifest_path.read_text(encoding="utf-8"))
    assert completion["execution_lineage"]["mode"] == "uninterrupted"
    assert completion["execution_lineage"]["resume_enabled"] is False


@pytest.mark.skipif(
    not _readable_git_checkout(CHECKOUT),
    reason="pinned GateMem checkout absent or unreadable",
)
def test_report_compiler_rejects_official_answer_token_drift(tmp_path: Path) -> None:
    checkout = GateMemCheckout(CHECKOUT)
    evidence = tuple(
        _domain_fixture(tmp_path, checkout, domain) for domain in sorted(DOMAIN_COUNTS)
    )
    summary_path = evidence[0].scorer_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["llm"]["tokens"]["total_tokens"] += 1
    _write_json(summary_path, summary)

    with pytest.raises(GateMemContractError, match="answer-call total_tokens"):
        build_gatemem_report(checkout=checkout, evidence=evidence)


@pytest.mark.skipif(
    not _readable_git_checkout(CHECKOUT),
    reason="pinned GateMem checkout absent or unreadable",
)
def test_report_compiler_rejects_a_stale_prediction_audit_pair(tmp_path: Path) -> None:
    checkout = GateMemCheckout(CHECKOUT)
    evidence = tuple(
        _domain_fixture(tmp_path, checkout, domain) for domain in sorted(DOMAIN_COUNTS)
    )
    audit_path = evidence[0].audit_path
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["config"]["context_token_budget"] = 2048
    _write_json(audit_path, audit)

    with pytest.raises(GateMemContractError, match="does not bind"):
        build_gatemem_report(checkout=checkout, evidence=evidence)


@pytest.mark.skipif(
    not _readable_git_checkout(CHECKOUT),
    reason="pinned GateMem checkout absent or unreadable",
)
@pytest.mark.parametrize("tamper", ["resume_flag", "implementation_digest"])
def test_report_compiler_rejects_tampered_completion_lineage(tmp_path: Path, tamper: str) -> None:
    checkout = GateMemCheckout(CHECKOUT)
    evidence = tuple(
        _domain_fixture(tmp_path, checkout, domain) for domain in sorted(DOMAIN_COUNTS)
    )
    completion_path = evidence[0].completion_manifest_path
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if tamper == "resume_flag":
        completion["execution_lineage"]["resume_used"] = True
    else:
        completion["execution_lineage"]["implementation"]["files"]["uv.lock"] = "f" * 64
    _write_json(completion_path, completion)

    with pytest.raises(GateMemContractError, match="lineage|implementation"):
        build_gatemem_report(checkout=checkout, evidence=evidence)
