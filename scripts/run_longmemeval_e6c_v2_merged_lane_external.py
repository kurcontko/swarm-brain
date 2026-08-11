#!/usr/bin/env python3
"""Control-corrected E6c confirmation runner.

The scientific design is inherited from the frozen E6c v1 manifest.  This
overlay adds a pre-QA G0 seal, exact domain-to-WAL reconciliation, deterministic
pack-pair repair, resumable QA control flow, and terminal-only reporting.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts import run_longmemeval_e6b_head20_external as e6b
from scripts import run_longmemeval_e6c_merged_lane_external as v1

PROTOCOL_VERSION: Final = "swarmbrain-longmemeval-e6c-merged-lane-confirmation-v2-control-corrected"
PROTOCOL_PATH: Final = REPO_ROOT / (
    "docs/research/longmemeval-e6c-merged-lane-confirmation-v2-control-amendment-2026-08-10.md"
)
RUNNER_PATH: Final = REPO_ROOT / "scripts/run_longmemeval_e6c_v2_merged_lane_external.py"
V1_PROTOCOL_VERSION: Final = "swarmbrain-longmemeval-e6c-merged-lane-confirmation-v1"
V1_PROTOCOL_PATH: Final = REPO_ROOT / (
    "docs/research/longmemeval-e6c-merged-lane-confirmation-protocol-2026-08-10.md"
)
V1_MANIFEST_SHA256: Final = "cf0e7326336d064ff84aad442c57ae0dddb2ca93e144a19238a5ba4313449d13"
DENSE_MANIFEST_SHA256: Final = "ce61a9343f3023e25d8ab72a19c2efc6598e452e93d60b37d04ee43e99564209"
DEFAULT_V1_OUTPUT: Final = Path("/private/tmp/swarmbrain-longmemeval-e6c-merged-n160-v1")
DEFAULT_OUTPUT: Final = Path("/private/tmp/swarmbrain-longmemeval-e6c-merged-n160-v2")


class E6CV2Error(ValueError):
    """The corrected E6c control or retained evidence is invalid/incomplete."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _control_fingerprint(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        str(path.relative_to(REPO_ROOT)): _sha256_file(path)
        for path in (RUNNER_PATH, PROTOCOL_PATH)
    }
    binding = {
        "source_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "source_v1_implementation_tree_sha256": source_manifest["implementation"]["tree_sha256"],
        "control_files": files,
    }
    return {**binding, "tree_sha256": v1.e1.sha256_json(binding)}


def _retarget_carrier() -> None:
    v1.PROTOCOL_VERSION = PROTOCOL_VERSION
    v1.PROTOCOL_PATH = PROTOCOL_PATH
    v1._configure_carrier()


def _verify_retired_v1_namespace(path: Path) -> None:
    manifest_path = path / "manifest.json"
    manifest = v1.e1.load_json(manifest_path, sealed=True)
    if manifest.get("artifact_sha256") != V1_MANIFEST_SHA256:
        raise E6CV2Error("retired E6c v1 manifest identity drifted")
    forbidden = (
        "external-call-journal",
        "extraction",
        "extraction-summary",
        "ranking",
        "selection",
        "pack",
        "prompts",
        "qa",
        "qa-receipts",
    )
    for name in forbidden:
        candidate = path / name
        if candidate.exists() and (candidate.is_file() or any(candidate.iterdir())):
            raise E6CV2Error("retired E6c v1 namespace contains outcome/API evidence")


def build_context(args: argparse.Namespace) -> v1.E6CContext:
    v1.PROTOCOL_VERSION = V1_PROTOCOL_VERSION
    v1.PROTOCOL_PATH = V1_PROTOCOL_PATH
    carrier_args = argparse.Namespace(**vars(args))
    carrier_args.output_dir = Path(args.v1_output_dir)
    source = v1.build_context(carrier_args)
    if source.manifest.get("artifact_sha256") != V1_MANIFEST_SHA256:
        raise E6CV2Error("source E6c v1 control manifest differs from the amendment")
    if source.e1.manifest.get("artifact_sha256") != DENSE_MANIFEST_SHA256:
        raise E6CV2Error("auxiliary dense manifest differs from the amendment")
    _verify_retired_v1_namespace(source.output_dir)

    output_dir = Path(args.output_dir).resolve()
    for other in (source.output_dir.resolve(), source.e1.output_dir.resolve()):
        if output_dir == other or output_dir in other.parents or other in output_dir.parents:
            raise E6CV2Error("v2, v1-control, and dense namespaces must be disjoint")
    if output_dir.is_symlink():
        raise E6CV2Error("E6c v2 output namespace cannot be a symbolic link")

    payload = deepcopy(source.manifest)
    payload.pop("artifact_sha256")
    payload.update(
        {
            "artifact_type": "swarmbrain-longmemeval-e6c-v2-merged-lane-run-manifest",
            "protocol_version": PROTOCOL_VERSION,
            "output_namespace": str(output_dir),
            "source_v1_control_manifest_sha256": V1_MANIFEST_SHA256,
            "implementation": _control_fingerprint(source.manifest),
            "control_correction": {
                "scientific_design_changed": False,
                "frozen_before_external_calls": True,
                "frozen_before_merged_ranking_or_M20_outcomes": True,
                "outcome_blind_dense_reuse_manifest_sha256": DENSE_MANIFEST_SHA256,
                "pre_QA_G0_required": True,
                "exact_domain_to_WAL_reconciliation_required": True,
                "partial_QA_resumes_before_case_recompilation": True,
                "deterministic_pack_pair_repair": True,
                "terminal_only_final_report": True,
            },
        }
    )
    payload["decision"]["G0"] = {
        "compiled_and_sealed_before_QA": True,
        "all_local_artifacts_exactly_replayed": True,
        "domain_routes_match_journals_one_to_one": True,
        "provider_request_ids_unique_across_routes": True,
        "zero_unresolved_reservations": True,
    }
    payload["reader_and_development_judge"]["conditional_on_G0_and_G1"] = True
    payload["reader_and_development_judge"].pop("conditional_on_G1", None)
    payload["claims"]["v1_control_namespace_retired_without_quality_verdict"] = True
    manifest = v1.e1.seal_artifact(payload)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if v1.e1.load_json(manifest_path, sealed=True) != manifest:
            raise E6CV2Error("E6c v2 namespace belongs to a different frozen manifest")
    else:
        v1.e1.write_json(manifest_path, manifest)

    carrier = e6b.E6Context(
        e1=source.e1,
        source_bytes=source.carrier.source_bytes,
        preflight=source.carrier.preflight,
        output_dir=output_dir,
        manifest=manifest,
        extractor=source.carrier.extractor,
        pricing=source.carrier.pricing,
        values_by_question_id=source.carrier.values_by_question_id,
    )
    context = v1.E6CContext(carrier=carrier, cohort_binding=source.cohort_binding)
    _retarget_carrier()
    return context


def _journal_hashes(
    context: v1.E6CContext,
    namespace: str,
    suffix: str,
) -> list[str]:
    root = context.output_dir / "external-call-journal" / namespace
    if not root.exists():
        return []
    return [
        str(v1.e1.load_json(path, sealed=True)["artifact_sha256"])
        for path in sorted(root.glob(f"*.{suffix}.json"))
    ]


def _qa_state_exists(context: v1.E6CContext) -> bool:
    return (
        e6b._qa_durable_state_exists(context.carrier)
        or (context.output_dir / "qa-completion.json").exists()
    )


def _extraction_sidecar_binding(context: v1.E6CContext) -> dict[str, Any]:
    root = context.output_dir / "extraction-values"
    if root.is_symlink() or not root.is_dir():
        raise E6CV2Error("extraction sidecar root is missing or unsafe")
    value_hashes: list[str] = []
    attempt_hashes: list[str] = []
    expected_paths: set[Path] = set()
    value_bytes = 0
    attempt_bytes = 0
    for question in context.e1.selected:
        for source_position in range(len(question.turns)):
            value_path = e6b._value_record_path(context.carrier, question, source_position)
            attempt_path = e6b._value_attempt_path(context.carrier, question, source_position)
            for path in (value_path, attempt_path):
                if path.is_symlink() or not path.is_file():
                    raise E6CV2Error("extraction sidecar is missing or unsafe")
                expected_paths.add(path)
            value_raw = value_path.read_bytes()
            attempt_raw = attempt_path.read_bytes()
            value_hashes.append(v1.e1.sha256_bytes(value_raw))
            attempt_hashes.append(v1.e1.sha256_bytes(attempt_raw))
            value_bytes += len(value_raw)
            attempt_bytes += len(attempt_raw)
    actual_paths: set[Path] = set()
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise E6CV2Error("extraction sidecar tree contains a symbolic link")
            if path.is_file():
                actual_paths.add(path)
    if actual_paths != expected_paths:
        raise E6CV2Error("extraction sidecar files differ from the exact domain")
    return {
        "value_files": len(value_hashes),
        "value_bytes": value_bytes,
        "ordered_value_file_sha256s_sha256": v1.e1.sha256_json(value_hashes),
        "attempt_ledger_files": len(attempt_hashes),
        "attempt_ledger_bytes": attempt_bytes,
        "ordered_attempt_file_sha256s_sha256": v1.e1.sha256_json(attempt_hashes),
        "ordering": "frozen-question-order-then-zero-based-source-position",
    }


def _register_provider_request_id(
    seen: set[str],
    request_id: str,
    *,
    route: str,
) -> None:
    if not request_id or request_id in seen:
        raise E6CV2Error(f"provider request ID is empty or crosses routes: {route}")
    seen.add(request_id)


def _audit_extraction_journals(
    context: v1.E6CContext,
    *,
    tokenizer: v1.e1.DeepSeekExactTokenizer,
) -> tuple[dict[str, Any], set[str]]:
    expected: dict[str, dict[str, Any]] = {}
    provider_ids: set[str] = set()
    source_values = 0
    application_attempts = 0
    extraction_cost = 0
    for question in context.e1.selected:
        summary, evidences = e6b.replay_extraction_question(context.carrier, question)
        source_values += len(evidences)
        application_attempts += sum(len(item.application_attempts) for item in evidences)
        extraction_cost += int(summary["accounting"]["cost_microusd"])
        for source_position, evidence in enumerate(evidences):
            record = v1.e1.load_json(
                e6b._value_record_path(context.carrier, question, source_position)
            )
            for application_attempt, (attempt, attempt_record) in enumerate(
                zip(evidence.application_attempts, record["application_attempts"], strict=True),
                start=1,
            ):
                route = e6b._extraction_journal_route(
                    question,
                    source_position,
                    application_attempt,
                )
                if route in expected:
                    raise E6CV2Error("extraction domain route is duplicated")
                request_id = str(attempt_record["response"]["provider_request_id"])
                _register_provider_request_id(provider_ids, request_id, route=route)
                request = e6b.replay_chat_request(attempt.raw_request)
                exact_tokens = tokenizer.exact_count(request.prompt)
                reserved = context.carrier.pricing.upper_bound_microusd(
                    input_tokens=exact_tokens,
                    output_tokens=request.max_tokens,
                    retry_count=e6b.DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
                    request_max_tokens=request.max_tokens,
                )
                expected[route] = {
                    "raw_request_sha256": v1.e1.sha256_bytes(attempt.raw_request),
                    "raw_response_sha256": v1.e1.sha256_bytes(attempt.raw_response),
                    "attempts": attempt.http_attempts,
                    "latency_microseconds": attempt.latency_microseconds,
                    "exact_local_prompt_tokens": exact_tokens,
                    "request_max_tokens": request.max_tokens,
                    "reserved_microusd": reserved,
                    "actual_microusd": e6b._attempt_cost_microusd(
                        attempt_record,
                        pricing=context.carrier.pricing,
                    ),
                }
    e6b._validate_expected_journal_bindings(
        context.carrier,
        namespace="extraction",
        expected=expected,
    )
    namespace_cost = e6b._journal_namespace_cost(context.carrier, "extraction")
    journal = {
        suffix: _journal_hashes(context, "extraction", suffix)
        for suffix in ("reservation", "response", "settlement")
    }
    if (
        source_values != sum(len(question.turns) for question in context.e1.selected)
        or namespace_cost != extraction_cost
        or any(len(rows) != application_attempts for rows in journal.values())
    ):
        raise E6CV2Error("extraction G0 accounting or journal coverage is incomplete")
    return (
        {
            "source_values": source_values,
            "application_attempts": application_attempts,
            "cost_microusd": extraction_cost,
            "provider_request_ids": len(provider_ids),
            "expected_routes": len(expected),
            "journal": journal,
            "per_value_sidecars": _extraction_sidecar_binding(context),
        },
        provider_ids,
    )


def _qa_completion_artifact(context: v1.E6CContext) -> dict[str, Any]:
    return v1.e1.seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6c-v2-qa-completion",
            "schema_version": v1.e1.ARTIFACT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "question_count": v1.E6C_SAMPLE,
            "arms": list(v1.ARMS),
            "reader_calls": v1.E6C_SAMPLE * len(v1.ARMS),
            "development_judge_calls": v1.E6C_SAMPLE * len(v1.ARMS),
            "aggregate_QA_statistics_compiled": False,
            "individual_labels_disclosed_by_progress_log": False,
            "official_gpt4o_calls": 0,
        }
    )


def _audit_qa_journals(
    context: v1.E6CContext,
    *,
    tokenizer: v1.e1.DeepSeekExactTokenizer,
    provider_ids: set[str],
) -> dict[str, Any]:
    expected: dict[str, dict[str, Any]] = {}
    qa_cost = 0
    receipt_rows = 0
    for question in context.e1.selected:
        qa_path = e6b.e6_phase_path(context.carrier, "qa", question)
        receipt_path = e6b.e6_jsonl_path(context.carrier, "qa-receipts", question)
        qa = v1.e1.load_json(qa_path, sealed=True)
        receipts = v1.e1._load_receipts(receipt_path)
        e6b.replay_qa_question(
            context.carrier,
            question,
            qa_row=qa,
            receipts=receipts,
            tokenizer=tokenizer,
        )
        qa_cost += int(qa["cost_microusd"])
        receipt_rows += len(receipts)
        for route_index, ((cell, receipt_id, role), receipt) in enumerate(
            zip(e6b._qa_expected_routes(context.carrier, question), receipts, strict=True)
        ):
            result = e6b.validate_chat_receipt_record(receipt)
            route = e6b._qa_journal_route(
                context.carrier,
                question,
                cell=cell,
                role=role,
                route_index=route_index,
            )
            if route in expected:
                raise E6CV2Error("QA domain route is duplicated")
            _register_provider_request_id(provider_ids, str(result.request_id), route=route)
            request = result.request
            exact_tokens = tokenizer.exact_count(request.prompt)
            reserved = context.carrier.pricing.upper_bound_microusd(
                input_tokens=exact_tokens,
                output_tokens=request.max_tokens,
                retry_count=e6b.DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
                request_max_tokens=request.max_tokens,
            )
            expected[route] = {
                "raw_request_sha256": result.raw_request_sha256,
                "raw_response_sha256": result.raw_response_sha256,
                "attempts": result.attempts,
                "latency_microseconds": int(math.ceil(result.latency_ms * 1000.0)),
                "exact_local_prompt_tokens": exact_tokens,
                "request_max_tokens": request.max_tokens,
                "reserved_microusd": reserved,
                "actual_microusd": e6b._chat_cost_microusd(
                    result,
                    pricing=context.carrier.pricing,
                ),
                "receipt_id": receipt_id,
                "role": role,
                "receipt_sha256": v1.e1.sha256_json(receipt),
            }
    e6b._validate_expected_journal_bindings(
        context.carrier,
        namespace="qa",
        expected=expected,
    )
    namespace_cost = e6b._journal_namespace_cost(context.carrier, "qa")
    journal = {
        suffix: _journal_hashes(context, "qa", suffix)
        for suffix in ("reservation", "response", "settlement")
    }
    if namespace_cost != qa_cost or any(len(rows) != len(expected) for rows in journal.values()):
        raise E6CV2Error("QA G0 accounting or journal coverage is incomplete")
    completion_path = context.output_dir / "qa-completion.json"
    completion = v1.e1.load_json(completion_path, sealed=True)
    if completion != _qa_completion_artifact(context):
        raise E6CV2Error("QA completion artifact differs from frozen v2 replay")
    return {
        "questions": v1.E6C_SAMPLE,
        "receipt_rows": receipt_rows,
        "cost_microusd": qa_cost,
        "expected_routes": len(expected),
        "journal": journal,
        "completion": v1._artifact_binding(completion_path, sealed=True),
    }


def _artifact_sets(
    context: v1.E6CContext,
    *,
    qa_complete: bool,
) -> dict[str, list[dict[str, Any]]]:
    artifacts = {
        "dense": v1._phase_artifacts(context, phase="dense", e1_namespace=True),
        "extraction_jsonl": v1._phase_artifacts(context, phase="extraction", sealed=False),
        "extraction_summary": v1._phase_artifacts(context, phase="extraction-summary"),
        "ranking": v1._phase_artifacts(context, phase="ranking"),
        "selection": v1._phase_artifacts(context, phase="selection"),
        "pack": v1._phase_artifacts(context, phase="pack"),
        "prompts": v1._phase_artifacts(context, phase="prompts"),
        "cases-final" if qa_complete else "cases-context": v1._phase_artifacts(
            context,
            phase="cases-final" if qa_complete else "cases-context",
        ),
    }
    if qa_complete:
        artifacts["qa"] = v1._phase_artifacts(context, phase="qa")
        artifacts["qa_receipts"] = v1._phase_artifacts(
            context,
            phase="qa-receipts",
            sealed=False,
        )
    if any(len(rows) != v1.E6C_SAMPLE for rows in artifacts.values()):
        raise E6CV2Error("G0 requires complete per-question artifact coverage")
    return artifacts


def run_pack_phase(context: v1.E6CContext) -> None:
    tokenizer = v1.e1.DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=v1.e1._snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    for question in context.e1.selected:
        pack_path = e6b.e6_phase_path(context.carrier, "pack", question)
        prompt_path = e6b.e6_phase_path(context.carrier, "prompts", question)
        if pack_path.exists() == prompt_path.exists():
            continue
        selection, results, _ = v1.replay_selection_question(context, question)
        pack, prompt, _ = v1._compute_pack_artifacts(
            context,
            question,
            selection=selection,
            results=results,
            tokenizer=tokenizer,
        )
        if prompt_path.exists():
            if v1.e1.load_json(prompt_path, sealed=True) != prompt:
                raise E6CV2Error("retained prompt half differs from deterministic replay")
            v1.e1.write_json(pack_path, pack)
        else:
            if v1.e1.load_json(pack_path, sealed=True) != pack:
                raise E6CV2Error("retained pack half differs from deterministic replay")
            v1.e1.write_json(prompt_path, prompt)
        print(f"  pack: repaired verified pair {question.question_id}", file=sys.stderr, flush=True)
    v1.run_pack_phase(context)


def _pre_qa_path(context: v1.E6CContext) -> Path:
    return context.output_dir / "pre-qa-gates.json"


def build_pre_qa_gates(context: v1.E6CContext) -> dict[str, Any]:
    if _qa_state_exists(context):
        raise E6CV2Error("pre-QA gates cannot be compiled after durable QA state exists")
    diagnostic = v1.build_diagnostic_report(context)
    tokenizer = v1.e1.DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=v1.e1._snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    v1._configure_carrier(cells=(v1.RAW_ARM, v1.RRF_ARM))
    extraction, _ = _audit_extraction_journals(context, tokenizer=tokenizer)
    journal_cost, unresolved = e6b._external_journal_cost(context.carrier)
    if unresolved or journal_cost != extraction["cost_microusd"]:
        raise E6CV2Error("pre-QA external journal is unresolved or does not reconcile")
    artifacts = _artifact_sets(context, qa_complete=False)
    artifact_sets = {
        phase: {
            "count": len(rows),
            "rows_sha256": v1.e1.sha256_json(rows),
            "rows": rows,
        }
        for phase, rows in artifacts.items()
    }
    g0 = {
        "gate": "G0-integrity-replay-before-QA",
        "passed": True,
        "questions": v1.E6C_SAMPLE,
        "source_values": extraction["source_values"],
        "domain_routes": extraction["expected_routes"],
        "provider_request_ids_unique": True,
        "unresolved_external_call_reservations": 0,
        "domain_to_journal_one_to_one_replay": True,
        "all_local_artifacts_replayed": True,
    }
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6c-v2-pre-qa-gates",
        "schema_version": v1.e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "source_v1_control_manifest_sha256": V1_MANIFEST_SHA256,
        "diagnostic_artifact_sha256": diagnostic["artifact_sha256"],
        "artifacts": artifact_sets,
        "extraction_journal_audit": extraction,
        "gates": {"G0": g0, "G1": diagnostic["gates"]["G1"]},
        "QA_calls_observed": 0,
        "official_gpt4o_calls": 0,
    }
    artifact = v1.e1.seal_artifact(payload)
    v1._write_or_verify(_pre_qa_path(context), artifact)
    return artifact


def load_pre_qa_gates(context: v1.E6CContext) -> dict[str, Any]:
    artifact = v1.e1.load_json(_pre_qa_path(context), sealed=True)
    if (
        artifact.get("artifact_type") != "swarmbrain-longmemeval-e6c-v2-pre-qa-gates"
        or artifact.get("protocol_version") != PROTOCOL_VERSION
        or artifact.get("run_manifest_sha256") != context.manifest["artifact_sha256"]
        or artifact.get("gates", {}).get("G0", {}).get("passed") is not True
    ):
        raise E6CV2Error("pre-QA gate artifact is missing, divergent, or non-passing")
    diagnostic = v1.e1.load_json(context.output_dir / "diagnostic-context.json", sealed=True)
    if diagnostic.get("artifact_sha256") != artifact.get("diagnostic_artifact_sha256"):
        raise E6CV2Error("pre-QA gate no longer binds the context diagnostic")
    current_artifacts = _artifact_sets(context, qa_complete=False)
    expected_artifacts = {
        phase: {
            "count": len(rows),
            "rows_sha256": v1.e1.sha256_json(rows),
            "rows": rows,
        }
        for phase, rows in current_artifacts.items()
    }
    if artifact.get("artifacts") != expected_artifacts:
        raise E6CV2Error("local evidence differs from the sealed pre-QA artifact set")
    expected_journal = artifact.get("extraction_journal_audit", {}).get("journal")
    current_journal = {
        suffix: _journal_hashes(context, "extraction", suffix)
        for suffix in ("reservation", "response", "settlement")
    }
    if expected_journal != current_journal:
        raise E6CV2Error("extraction journals differ from the sealed pre-QA route set")
    expected_sidecars = artifact.get("extraction_journal_audit", {}).get("per_value_sidecars")
    if expected_sidecars != _extraction_sidecar_binding(context):
        raise E6CV2Error("per-value extraction evidence differs from the pre-QA seal")
    return artifact


def _validate_existing_qa_completion(context: v1.E6CContext) -> None:
    completion_path = context.output_dir / "qa-completion.json"
    if not completion_path.exists():
        return
    if v1._qa_artifact_count(context) != v1.E6C_SAMPLE:
        raise E6CV2Error("QA completion exists before all per-question QA artifacts")
    completion = v1.e1.load_json(completion_path, sealed=True)
    if completion != _qa_completion_artifact(context):
        raise E6CV2Error("retained QA completion artifact is divergent")


def _finalize_complete_qa_offline(
    context: v1.E6CContext,
    *,
    tokenizer: v1.e1.DeepSeekExactTokenizer,
) -> dict[str, Any]:
    v1._configure_qa_carrier()
    e6b._reconcile_qa_journals(
        context.carrier,
        context.e1.selected,
        tokenizer=tokenizer,
    )
    _, unresolved = e6b._external_journal_cost(context.carrier)
    if unresolved:
        raise E6CV2Error("complete QA artifacts retain unresolved external call state")
    for question in context.e1.selected:
        qa = v1.e1.load_json(
            e6b.e6_phase_path(context.carrier, "qa", question),
            sealed=True,
        )
        receipts = v1.e1._load_receipts(e6b.e6_jsonl_path(context.carrier, "qa-receipts", question))
        e6b.replay_qa_question(
            context.carrier,
            question,
            qa_row=qa,
            receipts=receipts,
            tokenizer=tokenizer,
        )
    v1._write_or_verify(context.output_dir / "qa-completion.json", _qa_completion_artifact(context))
    return v1.build_diagnostic_report(context)


def run_qa_phase(
    context: v1.E6CContext,
    *,
    base_url: str,
    api_key_env: str,
) -> dict[str, Any]:
    _validate_existing_qa_completion(context)
    durable_qa = _qa_state_exists(context)
    pre_qa = load_pre_qa_gates(context) if durable_qa else build_pre_qa_gates(context)
    if pre_qa["gates"]["G1"].get("passed") is not True:
        if durable_qa:
            raise E6CV2Error("failed G1 is incompatible with retained QA state")
        print("  qa: G1 rejected M20; no reader or judge calls executed", file=sys.stderr)
        return v1.e1.load_json(context.output_dir / "diagnostic-context.json", sealed=True)
    tokenizer = v1.e1.DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=v1.e1._snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    if v1._qa_artifact_count(context) == v1.E6C_SAMPLE:
        return _finalize_complete_qa_offline(context, tokenizer=tokenizer)
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise E6CV2Error("E6c v2 QA is frozen to the official DeepSeek endpoint")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise E6CV2Error(f"environment variable {api_key_env!r} is missing")
    v1._configure_qa_carrier()
    asyncio.run(
        e6b._run_qa_async(
            context.carrier,
            selected=context.e1.selected,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key=api_key,
        )
    )
    if v1._qa_artifact_count(context) != v1.E6C_SAMPLE:
        raise E6CV2Error("QA returned without complete three-arm coverage")
    v1._write_or_verify(context.output_dir / "qa-completion.json", _qa_completion_artifact(context))
    return v1.build_diagnostic_report(context)


def build_report(context: v1.E6CContext) -> dict[str, Any]:
    pre_qa = load_pre_qa_gates(context)
    qa_count = v1._qa_artifact_count(context)
    if qa_count not in {0, v1.E6C_SAMPLE}:
        raise E6CV2Error("final report forbids partial QA coverage")
    g1 = pre_qa["gates"]["G1"]
    if g1.get("passed") is True and qa_count != v1.E6C_SAMPLE:
        raise E6CV2Error("passing G1 requires completed conditional QA before reporting")
    if g1.get("passed") is not True and _qa_state_exists(context):
        raise E6CV2Error("failed G1 is incompatible with durable QA state")

    diagnostic = v1.build_diagnostic_report(context)
    if diagnostic["gates"]["G1"] != g1:
        raise E6CV2Error("final diagnostic G1 differs from the sealed pre-QA decision")
    tokenizer = v1.e1.DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=v1.e1._snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    v1._configure_carrier(cells=v1.ARMS if qa_count else (v1.RAW_ARM, v1.RRF_ARM))
    extraction, provider_ids = _audit_extraction_journals(context, tokenizer=tokenizer)
    qa_audit = (
        _audit_qa_journals(context, tokenizer=tokenizer, provider_ids=provider_ids)
        if qa_count
        else None
    )
    journal_total, unresolved = e6b._external_journal_cost(context.carrier)
    qa_cost = 0 if qa_audit is None else int(qa_audit["cost_microusd"])
    if unresolved or journal_total != extraction["cost_microusd"] + qa_cost:
        raise E6CV2Error("final external journals are unresolved or do not reconcile")

    artifacts = _artifact_sets(context, qa_complete=bool(qa_count))
    artifacts["pre-qa-gates"] = [v1._artifact_binding(_pre_qa_path(context), sealed=True)]
    if qa_audit is not None:
        artifacts["qa-completion"] = [qa_audit["completion"]]
    for namespace, audit in (
        ("extraction", extraction),
        ("qa", qa_audit),
    ):
        if audit is None:
            continue
        for suffix, rows in audit["journal"].items():
            artifacts[f"external-{namespace}-{suffix}"] = rows
    artifact_sets = {
        phase: {
            "count": len(rows),
            "rows_sha256": v1.e1.sha256_json(rows),
            "rows": rows,
        }
        for phase, rows in artifacts.items()
    }

    g2 = diagnostic["gates"]["G2"]
    if g1.get("passed") is not True:
        verdict = "reject-M20-at-confirmatory-context-gate"
    elif g2.get("available") is not True:
        raise E6CV2Error("complete conditional QA did not produce a G2 decision")
    elif g2.get("passed") is not True:
        verdict = "reject-M20-at-paired-DeepSeek-QA-gate"
    else:
        verdict = "pass-fresh-LongMemEval-same-benchmark-DeepSeek-confirmation"
    g0 = {
        "gate": "G0-integrity-replay-final",
        "passed": True,
        "questions": v1.E6C_SAMPLE,
        "source_values": extraction["source_values"],
        "extraction_routes": extraction["expected_routes"],
        "QA_routes": 0 if qa_audit is None else qa_audit["expected_routes"],
        "provider_request_ids_unique_across_all_routes": True,
        "unresolved_external_call_reservations": 0,
        "domain_to_journal_one_to_one_replay": True,
        "pre_QA_gate_artifact_sha256": pre_qa["artifact_sha256"],
        "QA_completion_required_and_bound": qa_audit is not None,
    }
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6c-v2-final-report",
        "schema_version": v1.e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "source_v1_control_manifest_sha256": V1_MANIFEST_SHA256,
        "verdict": verdict,
        "classification": "same-benchmark-confirmation-not-official-or-SOTA-score",
        "question_count": v1.E6C_SAMPLE,
        "source_turn_count": extraction["source_values"],
        "artifacts": artifact_sets,
        "artifacts_sha256": v1.e1.sha256_json(
            {phase: rows["rows_sha256"] for phase, rows in artifact_sets.items()}
        ),
        "diagnostic_artifact_sha256": diagnostic["artifact_sha256"],
        "gates": {"G0": g0, "G1": g1, "G2": g2},
        "model_calls": {
            "DeepSeek_source_only_extraction_applications": extraction["application_attempts"],
            "DeepSeek_reader_calls": qa_count * len(v1.ARMS),
            "DeepSeek_development_judge_calls": qa_count * len(v1.ARMS),
            "official_GPT4o_calls": 0,
        },
        "cost": {
            "source_only_extraction_microusd": extraction["cost_microusd"],
            "reader_and_development_judge_microusd": qa_cost,
            "total_external_microusd": journal_total,
            "engineering_ledger_ceiling_microusd": v1.ENGINEERING_LEDGER_CEILING_MICROUSD,
            "within_engineering_ceiling": (journal_total <= v1.ENGINEERING_LEDGER_CEILING_MICROUSD),
            "billed_cost_claimed": False,
        },
        "claim_boundary": {
            "fresh_question_and_whole-history_composition_confirmation": True,
            "underlying_distractor_content_independence": False,
            "answer_evidence_overlap_with_development": False,
            "external_cross-corpus_generalization": False,
            "paper_reproduction": False,
            "official_LongMemEval_score": False,
            "model_independent_QA_improvement": False,
            "SOTA": False,
            "production_promotion": False,
        },
        "next_action": (
            "freeze-and-run-external-BEAM-or-LIGHT-confirmation"
            if verdict == "pass-fresh-LongMemEval-same-benchmark-DeepSeek-confirmation"
            else "reject-M20-without-posthoc-switch"
        ),
    }
    report = v1.e1.seal_artifact(payload)
    v1._write_or_verify(context.output_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "dense",
            "extract",
            "rank",
            "select",
            "pack",
            "preqa",
            "diagnose",
            "qa",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--dataset", type=Path, default=v1.DEFAULT_DATASET)
    parser.add_argument("--e1-output-dir", type=Path, default=v1.DEFAULT_E1_OUTPUT)
    parser.add_argument("--v1-output-dir", type=Path, default=DEFAULT_V1_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-root", type=Path, default=v1.e1.DEFAULT_QWEN_ROOT)
    parser.add_argument("--cross-encoder-root", type=Path, default=v1.e1.DEFAULT_CE_ROOT)
    parser.add_argument("--deepseek-root", type=Path, default=v1.e1.DEFAULT_DEEPSEEK_ROOT)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    return parser


def _run_phase(context: v1.E6CContext, args: argparse.Namespace, phase: str) -> None:
    if phase == "preflight":
        print(
            json.dumps(
                {
                    "manifest_sha256": context.manifest["artifact_sha256"],
                    "source_v1_manifest_sha256": V1_MANIFEST_SHA256,
                    "dense_manifest_sha256": context.e1.manifest["artifact_sha256"],
                    "cohort": context.cohort_binding["digests"],
                    "questions": v1.E6C_SAMPLE,
                    "source_turns": sum(len(q.turns) for q in context.e1.selected),
                },
                sort_keys=True,
            )
        )
    elif phase == "dense":
        v1.e1.run_dense_phase(
            context.e1,
            device=v1.QWEN_DEVICE,
            batch_size=v1.QWEN_BATCH_SIZE,
        )
    elif phase in {"extract", "rank", "select"}:
        v1._run_phase(context, args, phase)
    elif phase == "pack":
        run_pack_phase(context)
    elif phase in {"preqa", "diagnose"}:
        artifact = (
            load_pre_qa_gates(context) if _qa_state_exists(context) else build_pre_qa_gates(context)
        )
        print(json.dumps(artifact["gates"], sort_keys=True))
    elif phase == "qa":
        diagnostic = run_qa_phase(
            context,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
        print(json.dumps(diagnostic["gates"], sort_keys=True))
    elif phase == "report":
        report = build_report(context)
        print(json.dumps({"verdict": report["verdict"], "gates": report["gates"]}))
    else:  # pragma: no cover
        raise E6CV2Error(f"unknown E6c v2 phase: {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = build_context(args)
    if args.phase == "all":
        phases = (
            ("qa", "report")
            if _qa_state_exists(context)
            else ("dense", "extract", "rank", "select", "pack", "preqa", "qa", "report")
        )
    else:
        phases = (args.phase,)
    with e6b._output_process_lock(context.output_dir):
        for phase in phases:
            _run_phase(context, args, phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
