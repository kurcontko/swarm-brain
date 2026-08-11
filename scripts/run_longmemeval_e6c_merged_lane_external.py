#!/usr/bin/env python3
"""Preregistered fresh-cohort E6c merged-SFK confirmation.

E6c compares three fixed retrieval policies on one outcome-blind LongMemEval-S
cohort that excludes every E6/E6b question:

* ``R0``: exhaustive raw-turn Qwen ranking, top 20;
* ``R1``: frozen equal-family raw/merged-SFK RRF-k60, top 20;
* ``M20``: merged-SFK lane top 20, hydrated back to immutable raw turns.

The E6b runner remains byte-untouched.  This runner reuses its already-tested
external-call journal and exhaustive-ranking carrier functions in a distinct
namespace, with an E6c protocol/run-manifest binding.  E6c owns selection,
three-arm packing, context inference, QA scheduling, and final claims.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from benchmarks.integrations.longmemeval_e6c.cohort import (
    E6C_ABSTENTION_COUNT,
    E6C_POSITIONS,
    E6C_PYTHON_VERSION,
    E6C_SAMPLE,
    E6C_SEED,
    build_cohort_binding,
)
from benchmarks.integrations.longmemeval_e6c.cohort import (
    sha256_json as cohort_sha256_json,
)
from benchmarks.integrations.longmemeval_e6c.metrics import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    paired_lower_bound,
    percentile,
)
from benchmarks.integrations.longmemeval_official_preflight import (
    ExactTokenizerPin,
    freeze_official_preflight,
)
from benchmarks.integrations.longmemeval_representation import (
    KEY_FAMILY_DEPTH,
    KeyFamily,
    RepresentationCell,
    RepresentationResult,
    compile_question_canonical_values,
)
from benchmarks.integrations.longmemeval_representation.evidence import (
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS,
    DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS,
    PROMPT_IDENTITY_SHA256,
    deepseek_r1_extractor_identity,
)
from benchmarks.integrations.longmemeval_representation.head_matched import (
    HEAD_MATCHED_VALUE_COUNT,
)
from benchmarks.integrations.longmemeval_representation.merged_lane import (
    MERGED_LANE_PROTOCOL,
    select_merged_lane_top20,
)
from benchmarks.integrations.longmemeval_representation.packing_bridge import (
    RepresentationPromptPackingResult,
    pack_representation_result,
)
from benchmarks.integrations.longmemeval_turns import (
    TurnProjection,
    compile_official_longmemeval_s,
)
from scripts import run_longmemeval_e1_external as e1
from scripts import run_longmemeval_e6b_head20_external as e6b
from scripts._longmemeval_common import QWEN_QUERY_INSTRUCTION

PROTOCOL_VERSION: Final = "swarmbrain-longmemeval-e6c-merged-lane-confirmation-v1"
PROTOCOL_PATH: Final = (
    REPO_ROOT / "docs/research/longmemeval-e6c-merged-lane-confirmation-protocol-2026-08-10.md"
)
DATASET_SHA256: Final = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
DEFAULT_DATASET: Final = Path("/private/tmp/longmemeval_s_cleaned.json")
DEFAULT_E1_OUTPUT: Final = Path("/private/tmp/swarmbrain-longmemeval-e1-e6c-n160-v1")
DEFAULT_OUTPUT: Final = Path("/private/tmp/swarmbrain-longmemeval-e6c-merged-n160-v1")

RAW_ARM: Final = "R0"
RRF_ARM: Final = "R1"
MERGED_ARM: Final = "M20"
ARMS: Final = (RAW_ARM, RRF_ARM, MERGED_ARM)
QA_PERMUTATIONS: Final = tuple(itertools.permutations(ARMS))

QWEN_DEVICE: Final = "mps"
QWEN_BATCH_SIZE: Final = 8
QWEN_TORCH_VERSION: Final = "2.7.1"
QWEN_TRANSFORMERS_VERSION: Final = "4.55.4"
QWEN_DTYPE: Final = "float16"
EXTRACTION_CONCURRENCY: Final = 24
ENGINEERING_LEDGER_CEILING_MICROUSD: Final = 100_000_000
QA_PRACTICAL_EFFECT_FLOOR: Final = 0.02
QA_MAX_TYPE_REGRESSION: Final = 0.05

IMPLEMENTATION_FILES: Final = tuple(
    dict.fromkeys(
        (
            *e6b.IMPLEMENTATION_FILES,
            "benchmarks/integrations/longmemeval_e6c/__init__.py",
            "benchmarks/integrations/longmemeval_e6c/cohort.py",
            "benchmarks/integrations/longmemeval_e6c/metrics.py",
            "benchmarks/integrations/longmemeval_representation/merged_lane.py",
            "scripts/run_longmemeval_e6c_merged_lane_external.py",
            str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        )
    )
)


class E6CError(ValueError):
    """E6c input, runtime, or retained evidence violates the frozen protocol."""


@dataclass(frozen=True, slots=True)
class E6CContext:
    carrier: e6b.E6Context
    cohort_binding: dict[str, Any]

    @property
    def e1(self) -> e1.ExperimentContext:
        return self.carrier.e1

    @property
    def output_dir(self) -> Path:
        return self.carrier.output_dir

    @property
    def manifest(self) -> dict[str, Any]:
        return self.carrier.manifest


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _implementation_fingerprint() -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise E6CError(f"E6c implementation file is missing or unsafe: {relative}")
        files[relative] = _sha256_file(path)
    return {"files": files, "tree_sha256": e1.sha256_json(files)}


def _records(path: Path) -> tuple[dict[str, Any], ...]:
    value = e1.load_json(path)
    if (
        not isinstance(value, list)
        or len(value) != 500
        or any(not isinstance(row, dict) for row in value)
    ):
        raise E6CError("E6c requires the exact 500-object LongMemEval-S array")
    return tuple(value)


def _selected_questions(
    corpus: Any,
    records: tuple[dict[str, Any], ...],
) -> tuple[e1.SelectedQuestion, ...]:
    turns_by_question: dict[str, list[TurnProjection]] = defaultdict(list)
    for turn in corpus.turns:
        turns_by_question[turn.turn_id.question_id].append(turn)
    selected = tuple(
        e1.SelectedQuestion(
            position=position,
            record=records[position],
            turns=tuple(turns_by_question[str(records[position]["question_id"])]),
        )
        for position in E6C_POSITIONS
    )
    if len(selected) != E6C_SAMPLE or sum(len(item.turns) for item in selected) != 78_648:
        raise E6CError("E6c selected question or source-turn count drifted")
    return selected


def _sample_binding(
    corpus: Any,
    selected: Sequence[e1.SelectedQuestion],
) -> list[dict[str, Any]]:
    return [
        {
            "position": question.position,
            "question_id": question.question_id,
            "question_type": str(question.record["question_type"]),
            "source_record": corpus.questions[question.position].source_record.as_dict(),
            "turns": len(question.turns),
        }
        for question in selected
    ]


def _verify_runtime() -> dict[str, str]:
    runtime = {
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "dtype": QWEN_DTYPE,
    }
    expected = {
        "python": E6C_PYTHON_VERSION,
        "torch": QWEN_TORCH_VERSION,
        "transformers": QWEN_TRANSFORMERS_VERSION,
        "dtype": QWEN_DTYPE,
    }
    if runtime != expected:
        raise E6CError(f"E6c runtime differs from freeze: {runtime!r}")
    return runtime


def _build_e1_context(args: argparse.Namespace) -> tuple[e1.ExperimentContext, dict[str, Any]]:
    dataset = Path(args.dataset).resolve()
    if _sha256_file(dataset) != DATASET_SHA256:
        raise E6CError("LongMemEval-S dataset bytes differ from the frozen source")
    corpus = compile_official_longmemeval_s(dataset)
    records = _records(dataset)
    cohort = build_cohort_binding(records)
    selected = _selected_questions(corpus, records)
    sample = _sample_binding(corpus, selected)
    _sample_sha256 = cohort_sha256_json(sample)  # explicit finite-JSON validation

    qwen_root = Path(args.qwen_root).resolve()
    cross_encoder_root = Path(args.cross_encoder_root).resolve()
    deepseek_root = Path(args.deepseek_root).resolve()
    snapshots = {
        "qwen_embedding": e1.verify_snapshot(
            qwen_root,
            name=e1.QWEN_MODEL,
            revision=e1.QWEN_REVISION,
            expected=e1.QWEN_FILES,
        ),
        "mixedbread_cross_encoder": e1.verify_snapshot(
            cross_encoder_root,
            name=e1.CROSS_ENCODER_MODEL,
            revision=e1.CROSS_ENCODER_REVISION,
            expected=e1.CROSS_ENCODER_FILES,
        ),
        "deepseek_tokenizer": e1.verify_snapshot(
            deepseek_root,
            name="deepseek-ai/DeepSeek-V4-Flash",
            revision=e1.DEEPSEEK_REVISION,
            expected=e1.DEEPSEEK_FILES,
        ),
    }
    implementation = e1.implementation_fingerprint()
    output_dir = Path(args.e1_output_dir).resolve()
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6c-raw-dense-run-manifest",
        "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": e1.PROTOCOL_VERSION,
        "parent_protocol_version": PROTOCOL_VERSION,
        "classification": "fresh-same-benchmark-confirmation-auxiliary-dense-evidence",
        "production_configuration": False,
        "output_namespace": str(output_dir),
        "dataset": corpus.source_artifact.as_dict(),
        "turn_projection": corpus.fingerprint(),
        "sample": {
            "seed": E6C_SEED,
            "requested": E6C_SAMPLE,
            "count": len(selected),
            "questions": sample,
            "questions_sha256": _sample_sha256,
            "cohort_selector_sha256": cohort["digests"]["selector_sha256"],
        },
        "cells": ["E1-A-dense-carrier-only"],
        "retrieval": {
            "qwen_query_instruction_sha256": e1.sha256_bytes(
                QWEN_QUERY_INSTRUCTION.encode("utf-8")
            ),
            "qwen_max_length": e1.QWEN_MAX_LENGTH,
            "qwen_padding_side": "right",
            "qwen_batching": {
                "method": "length-sorted-padding-aware-dynamic-batching",
                "maximum_batch_size": QWEN_BATCH_SIZE,
                "maximum_padding_ratio": e1.QWEN_MAX_PADDING_RATIO,
                "attention_cell_budget": e1.QWEN_ATTENTION_CELL_BUDGET,
            },
        },
        "local_execution": {
            "device": QWEN_DEVICE,
            "qwen_maximum_batch_size": QWEN_BATCH_SIZE,
        },
        "model_snapshots": snapshots,
        "implementation": implementation,
        "claims": {
            "gold_fields_used_for_retrieval_or_selection": False,
            "sample_is_content_independent": False,
            "official_longmemeval_score": False,
            "production_policy_changed": False,
        },
    }
    manifest = e1.seal_artifact(payload)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if e1.load_json(manifest_path, sealed=True) != manifest:
            raise E6CError("E6c dense namespace belongs to a different frozen manifest")
    else:
        e1.write_json(manifest_path, manifest)
    return (
        e1.ExperimentContext(
            corpus=corpus,
            records=records,
            selected=selected,
            manifest=manifest,
            output_dir=output_dir,
            qwen_root=qwen_root,
            cross_encoder_root=cross_encoder_root,
            deepseek_root=deepseek_root,
        ),
        cohort,
    )


def _configure_carrier(*, cells: Sequence[str] = (RAW_ARM, RRF_ARM)) -> None:
    """Bind reused E6b carrier functions to E6c without touching source files."""

    e6b.E6_RUN_PROTOCOL_VERSION = PROTOCOL_VERSION
    e6b.MAX_EXTERNAL_COST_MICROUSD = ENGINEERING_LEDGER_CEILING_MICROUSD
    e6b.E6B_SAMPLE = E6C_SAMPLE
    e6b.E6B_ABS_COUNT = E6C_ABSTENTION_COUNT
    e6b.E6_CELLS = tuple(cells)


def build_context(args: argparse.Namespace) -> E6CContext:
    runtime = _verify_runtime()
    e1_context, cohort = _build_e1_context(args)
    output_dir = Path(args.output_dir).resolve()
    e1_output = e1_context.output_dir.resolve()
    if (
        output_dir == e1_output
        or output_dir in e1_output.parents
        or e1_output in output_dir.parents
    ):
        raise E6CError("E6c output and auxiliary dense namespaces must be disjoint")
    if output_dir.is_symlink() or e1_context.output_dir.is_symlink():
        raise E6CError("E6c output namespaces cannot be symbolic links")

    source_bytes = Path(args.dataset).resolve().read_bytes()
    values_by_question_id = MappingProxyType(
        {
            question.question_id: compile_question_canonical_values(
                e1_context.corpus,
                question_id=question.question_id,
            )
            for question in e1_context.selected
        }
    )
    snapshots = {
        "qwen_embedding": e1_context.manifest["model_snapshots"]["qwen_embedding"],
        "deepseek_tokenizer": e1_context.manifest["model_snapshots"]["deepseek_tokenizer"],
    }
    tokenizer_pin = ExactTokenizerPin(
        model=e1.DEEPSEEK_MODEL,
        revision=e1.DEEPSEEK_REVISION,
        artifact_sha256=snapshots["deepseek_tokenizer"]["artifact_sha256"],
        executable_sha256=e6b._tokenizer_executable_sha256(e1_context),
    )
    preflight = freeze_official_preflight(source_bytes, tokenizer=tokenizer_pin)
    extractor = deepseek_r1_extractor_identity(
        model_revision=e6b.EXTRACTOR_MODEL_REVISION,
        model_artifact_sha256=e6b._extractor_model_artifact_sha256(),
        identity_artifact_sha256=e6b._extractor_identity_artifact_sha256(),
    )
    pricing = e6b._pricing_identity()
    implementation = _implementation_fingerprint()
    sample = _sample_binding(e1_context.corpus, e1_context.selected)
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6c-merged-lane-run-manifest",
        "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "classification": "fresh-question-composition-same-benchmark-confirmation",
        "production_configuration": False,
        "output_namespace": str(output_dir),
        "source_e1_manifest_sha256": e1_context.manifest["artifact_sha256"],
        "official_preflight_manifest_sha256": preflight.manifest_sha256,
        "dataset": e1_context.manifest["dataset"],
        "turn_projection": e1_context.manifest["turn_projection"],
        "sample": {
            "seed": E6C_SEED,
            "count": E6C_SAMPLE,
            "source_turns": sum(len(question.turns) for question in e1_context.selected),
            "questions": sample,
            "questions_sha256": e1.sha256_json(sample),
        },
        "sample_selection_preregistration": cohort,
        "development_selection_disclosure": {
            "source": "sealed-E6b-n160-context-outcomes",
            "specifications_screened": 56,
            "selected_policy": MERGED_LANE_PROTOCOL,
            "selected_after_development_outcomes": True,
            "E6b_is_development_only": True,
            "E6c_no_post_outcome_policy_switch": True,
        },
        "arms": {
            RAW_ARM: {
                "policy": "raw-Qwen-top20",
                "inner_registered_representation_cell": RepresentationCell.RAW.value,
            },
            RRF_ARM: {
                "policy": "equal-family-raw-plus-merged-RRF-k60-top20",
                "inner_registered_representation_cell": (RepresentationCell.RAW_MERGED_SFK.value),
            },
            MERGED_ARM: {
                "policy": MERGED_LANE_PROTOCOL,
                "inner_registered_representation_cell": (RepresentationCell.RAW_MERGED_SFK.value),
                "selection_policy_overrides_registered_cell_ranking": True,
            },
        },
        "representation": {
            "canonical_value": "immutable-F0-turn-projection",
            "merged_key_count_per_value": 1,
            "family_depth": KEY_FAMILY_DEPTH,
            "head_values": HEAD_MATCHED_VALUE_COUNT,
            "RRF": {"raw_weight": 1.0, "merged_weight": 1.0, "k": 60},
            "M20_deduplication": "one-key-per-source-value-before-fixed-top20",
            "tie_break": "raw-cosine-descending-then-key-id-ascending",
            "hydration": "byte-identical-canonical-raw-values-only",
        },
        "extraction": {
            "source_only": True,
            "input_fields": ["source_value"],
            "question_query_answer_type_and_judge_fields_forbidden": True,
            "extractor": extractor.content_free_binding(),
            "prompt_identity_sha256": PROMPT_IDENTITY_SHA256,
            "deployment": e6b.DEEPSEEK_DEPLOYMENT_ID,
            "temperature": 0.0,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
            "thinking": "disabled",
            "maximum_http_attempts_per_application_attempt": (DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS),
            "maximum_application_schema_attempts": DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS,
            "concurrency": EXTRACTION_CONCURRENCY,
            "raw_invalid_attempts_retained": True,
            "carrier_implementation": (
                "frozen-E6b-external-journal-functions-under-E6c-manifest-and-namespace"
            ),
            "carrier_artifact_type_names_may_retain_e6b_schema_prefix": True,
        },
        "ranking": {
            "model": e1.QWEN_MODEL,
            "revision": e1.QWEN_REVISION,
            "query_instruction_sha256": e1.sha256_bytes(QWEN_QUERY_INSTRUCTION.encode("utf-8")),
            "raw_scores": "fresh-exhaustive-cosine",
            "merged_scores": "fresh-exhaustive-cosine",
            "device": QWEN_DEVICE,
            "maximum_batch_size": QWEN_BATCH_SIZE,
            "maximum_input_tokens": e1.QWEN_MAX_LENGTH,
            "maximum_padding_ratio": e1.QWEN_MAX_PADDING_RATIO,
            "attention_cell_budget": e1.QWEN_ATTENTION_CELL_BUDGET,
            "cosine_validation_tolerance": e6b.QWEN_COSINE_TOLERANCE,
            "runtime": runtime,
        },
        "packing": {
            "bridge_protocol": "swarmbrain-longmemeval-e6-representation-packing-bridge-v1",
            "complete_reader_prompt_token_budget": e1.TOKEN_BUDGET,
            "exact_tokenizer": tokenizer_pin.content_free_binding(),
            "preflight_admission": {
                "manifest_freeze_executed": True,
                "prepared_run_validation_executed": False,
                "prepared_run_receipt_sha256": None,
                "official_prepared_run_claimed": False,
                "manifest_case_count": len(preflight.cases),
                "selected_confirmation_case_count": E6C_SAMPLE,
                "arms_per_case": len(ARMS),
                "reason": "three-arm same-benchmark confirmation is not official admission",
            },
        },
        "decision": {
            "G1": {
                "gold_eligible_cases": E6C_SAMPLE - E6C_ABSTENTION_COUNT,
                "paired_M20_MRR_lower_bound_above_zero_vs": [RAW_ARM, RRF_ARM],
                "coverage_observed_noninferiority_vs": RRF_ARM,
                "coverage_axes": ["any", "all", "answer_session_recall"],
                "M20_total_and_p95_tokens_no_greater_than": [RAW_ARM, RRF_ARM],
                "exact_complete_raw_values_delivered_per_arm_case": 20,
                "zero_whole_turn_drops": True,
            },
            "G2": {
                "paired_M20_accuracy_lower_bound_above_zero_vs": [RAW_ARM, RRF_ARM],
                "minimum_observed_accuracy_delta": QA_PRACTICAL_EFFECT_FLOOR,
                "maximum_observed_type_regression": QA_MAX_TYPE_REGRESSION,
                "abstention_observed_nonregression": True,
            },
            "paired_bootstrap": {
                "method": BOOTSTRAP_METHOD,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "tail": "one-sided-lower",
                "strata": "question_type",
                "unit": "question-local-history",
            },
            "intersection_union_gate": True,
            "optional_stopping_or_sample_extension_permitted": False,
        },
        "reader_and_development_judge": {
            "conditional_on_G1": True,
            "provider": "DeepSeek API",
            "model": e1.DEEPSEEK_MODEL,
            "deployment": e6b.DEEPSEEK_DEPLOYMENT_ID,
            "thinking": "disabled",
            "temperature": 0.0,
            "reader_max_tokens": e6b.READER_MAX_TOKENS,
            "judge_max_tokens": e6b.JUDGE_MAX_TOKENS,
            "same_mutable_alias_for_reader_and_judge": True,
            "arm_order": "six-permutation-balanced-within-question-type-v1",
            "blind_arm_identity_in_judge_prompt": True,
            "official_gpt4o_calls": 0,
        },
        "cost_accounting": {
            "pricing": pricing.content_free_binding(),
            "engineering_ledger_ceiling_microusd": ENGINEERING_LEDGER_CEILING_MICROUSD,
            "engineering_ceiling_is_not_a_planning_budget": True,
            "authorization": "operator-authorized-DeepSeek-until-provider-balance-error-2026-08-10",
            "provider_balance_or_transport_error_is_resumable_not_a_quality_outcome": True,
        },
        "model_snapshots": snapshots,
        "implementation": implementation,
        "claims": {
            "question_and_whole_history_compositions_excluded_from_development": True,
            "underlying_sessions_or_turns_are_content_independent": False,
            "external_cross_corpus_confirmation": False,
            "official_longmemeval_score": False,
            "official_gpt4o_judge_executed": False,
            "SOTA_claim_authorized": False,
            "production_policy_changed": False,
        },
    }
    manifest = e1.seal_artifact(payload)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if e1.load_json(manifest_path, sealed=True) != manifest:
            raise E6CError("E6c output namespace belongs to a different frozen manifest")
    else:
        e1.write_json(manifest_path, manifest)
    carrier = e6b.E6Context(
        e1=e1_context,
        source_bytes=source_bytes,
        preflight=preflight,
        output_dir=output_dir,
        manifest=manifest,
        extractor=extractor,
        pricing=pricing,
        values_by_question_id=values_by_question_id,
    )
    _configure_carrier()
    return E6CContext(carrier=carrier, cohort_binding=cohort)


def _write_or_verify(path: Path, artifact: Mapping[str, Any]) -> None:
    expected = dict(artifact)
    if path.exists():
        if e1.load_json(path, sealed=True) != expected:
            raise E6CError(f"retained E6c artifact differs from replay: {path}")
        return
    e1.write_json(path, expected)


def _merged_query_accounting(rank: Mapping[str, Any]) -> dict[str, Any]:
    stages = rank["query_accounting"][RRF_ARM]["stages"]
    merged = [stage for stage in stages if stage["name"] == "merged-qwen-index-query-ranking"]
    if len(merged) != 1:
        raise E6CError("E6c M20 requires exactly one merged-Qwen query stage")
    return {
        "complete": True,
        "source": "externally-attested-unverified",
        "stages": merged,
        "stages_sha256": e1.sha256_json(merged),
    }


def replay_selection_question(
    context: E6CContext,
    question: e1.SelectedQuestion,
) -> tuple[
    dict[str, Any],
    dict[str, RepresentationResult],
    dict[str, dict[str, Any]],
]:
    _configure_carrier()
    rank, source_results = e6b.replay_rank_question(context.carrier, question)
    _r0_corpus, r1_corpus, _evidences, _extraction = e6b._representation_corpora(
        context.carrier,
        question,
    )
    merged_rows = e6b._validate_merged_score_rows(
        rank.get("merged_sfk", {}).get("observations"),
        corpus=r1_corpus,
    )
    observation = e6b._family_observation(
        r1_corpus,
        family=KeyFamily.MERGED_SFK,
        question=question,
        scorer=e6b._qwen_scorer_identity(context.carrier),
        rows=merged_rows,
    )
    merged = select_merged_lane_top20(
        source_results[RRF_ARM],
        corpus=r1_corpus,
        observation=observation,
    )
    results = {
        RAW_ARM: source_results[RAW_ARM],
        RRF_ARM: source_results[RRF_ARM],
        MERGED_ARM: merged,
    }
    if any(len(result.hydrated_values) != HEAD_MATCHED_VALUE_COUNT for result in results.values()):
        raise E6CError("every E6c arm must expose exactly 20 candidate raw values")
    accounting = {
        RAW_ARM: dict(rank["query_accounting"][RAW_ARM]),
        RRF_ARM: dict(rank["query_accounting"][RRF_ARM]),
        MERGED_ARM: _merged_query_accounting(rank),
    }
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6c-selection-question",
        "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_carrier_ranking_artifact_sha256": rank["artifact_sha256"],
        "source_extraction_artifact_sha256": rank["source_extraction_artifact_sha256"],
        "merged_observation_sha256": observation.observation_sha256,
        "arms": {
            arm: {
                "outer_arm_id": arm,
                "inner_registered_representation_cell": results[arm].cell.value,
                "representation": results[arm].content_free_artifact(),
                "query_accounting": accounting[arm],
            }
            for arm in ARMS
        },
        "claims": {
            "M20_policy": MERGED_LANE_PROTOCOL,
            "M20_active_query_family": KeyFamily.MERGED_SFK.value,
            "M20_source_materialized_index": "raw-plus-merged-sfk",
            "M20_derived_text_delivered_to_reader": False,
            "gold_or_outcome_fields_used": False,
            "external_call_executed_by_selection": False,
            "production_policy_changed": False,
        },
    }
    artifact = e1.seal_artifact(payload)
    path = e6b.e6_phase_path(context.carrier, "selection", question)
    if path.exists() and e1.load_json(path, sealed=True) != artifact:
        raise E6CError("saved E6c selection differs from exact deterministic replay")
    return artifact, results, accounting


def run_selection_phase(context: E6CContext) -> None:
    for completed, question in enumerate(context.e1.selected, start=1):
        artifact, results, _ = replay_selection_question(context, question)
        path = e6b.e6_phase_path(context.carrier, "selection", question)
        action = "verified" if path.exists() else "built"
        _write_or_verify(path, artifact)
        print(
            f"  select: {completed}/{E6C_SAMPLE} {action} {question.question_id} "
            + " ".join(f"{arm}={len(results[arm].hydrated_values)}" for arm in ARMS),
            file=sys.stderr,
            flush=True,
        )


def _final_tokenizer_receipt_sha256(packed: RepresentationPromptPackingResult) -> str:
    trace = packed.packed.trace
    final = trace.get("final_prompt")
    observations = trace.get("exact_count_observations")
    if not isinstance(final, dict) or not isinstance(observations, list):
        raise E6CError("E6c pack lacks exact tokenizer observations")
    sequence = final.get("final_observation_sequence")
    observation = next(
        (
            item
            for item in observations
            if isinstance(item, dict) and item.get("sequence") == sequence
        ),
        None,
    )
    if not isinstance(observation, dict) or not isinstance(observation.get("receipt"), dict):
        raise E6CError("E6c pack lacks its final tokenizer receipt")
    return e1.sha256_json(observation["receipt"])


def _receipt_namespace(
    context: E6CContext,
    question: e1.SelectedQuestion,
    arm: str,
) -> dict[str, Any]:
    material = {
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_position": question.position,
        "question_id": question.question_id,
        "outer_arm_id": arm,
    }
    return {**material, "namespace_sha256": e1.sha256_json(material)}


def _compute_pack_artifacts(
    context: E6CContext,
    question: e1.SelectedQuestion,
    *,
    selection: Mapping[str, Any],
    results: Mapping[str, RepresentationResult],
    tokenizer: e1.DeepSeekExactTokenizer,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, RepresentationPromptPackingResult],
]:
    packed: dict[str, RepresentationPromptPackingResult] = {}
    prompt_arms: dict[str, Any] = {}
    for arm in ARMS:
        tokenizer.reset_receipts()
        bridge = pack_representation_result(
            results[arm],
            manifest=context.carrier.preflight,
            question_id=question.question_id,
            question=question.question,
            current_date=question.current_date,
            tokenizer=tokenizer,
        )
        packed[arm] = bridge
        receipt_sha256 = _final_tokenizer_receipt_sha256(bridge)
        namespace = _receipt_namespace(context, question, arm)
        prompt_arms[arm] = {
            "prompt": bridge.prompt,
            "prompt_sha256": e1.sha256_bytes(bridge.prompt.encode("utf-8")),
            "prompt_utf8_bytes": len(bridge.prompt.encode("utf-8")),
            "prompt_tokens": bridge.packed.trace["final_prompt"]["tokens"],
            "bridge_trace_sha256": bridge.trace_sha256,
            "packing_trace_sha256": bridge.packed.trace_sha256,
            "final_tokenizer_receipt_sha256": receipt_sha256,
            "tokenizer_receipt_namespace": namespace,
            "namespaced_final_tokenizer_receipt_sha256": e1.sha256_json(
                {
                    "namespace_sha256": namespace["namespace_sha256"],
                    "receipt_sha256": receipt_sha256,
                }
            ),
        }
    prompt_artifact = e1.seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6c-prompts-sensitive",
            "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "question_id": question.question_id,
            "question_position": question.position,
            "classification": "contains-public-benchmark-question-and-turn-text",
            "arms": prompt_arms,
        }
    )
    pack_artifact = e1.seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6c-pack-question",
            "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "question_id": question.question_id,
            "question_position": question.position,
            "source_selection_artifact_sha256": selection["artifact_sha256"],
            "source_representation_trace_sha256s": {arm: results[arm].trace_sha256 for arm in ARMS},
            "prompt_artifact_sha256": prompt_artifact["artifact_sha256"],
            "preflight_manifest_sha256": context.carrier.preflight.manifest_sha256,
            "tokenizer": tokenizer.identity.as_dict(),
            "runtime": {
                "python": sys.version.split()[0],
                "transformers": tokenizer.transformers_version,
            },
            "arms": {arm: packed[arm].content_free_artifact() for arm in ARMS},
            "claims": {
                "complete_reader_prompt_counted": True,
                "whole_canonical_turns_only": True,
                "derived_keys_delivered_to_reader": False,
                "gold_fields_consumed": False,
                "reader_or_judge_executed": False,
                "official_prepared_run_admission": False,
                "production_policy_changed": False,
            },
        }
    )
    return pack_artifact, prompt_artifact, packed


def replay_pack_question(
    context: E6CContext,
    question: e1.SelectedQuestion,
    *,
    tokenizer: e1.DeepSeekExactTokenizer,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, RepresentationPromptPackingResult],
    dict[str, RepresentationResult],
    dict[str, dict[str, Any]],
]:
    selection, results, accounting = replay_selection_question(context, question)
    saved_selection = e1.load_json(
        e6b.e6_phase_path(context.carrier, "selection", question),
        sealed=True,
    )
    if saved_selection != selection:
        raise E6CError("E6c pack source selection is missing or divergent")
    expected_pack, expected_prompt, packed = _compute_pack_artifacts(
        context,
        question,
        selection=selection,
        results=results,
        tokenizer=tokenizer,
    )
    pack_path = e6b.e6_phase_path(context.carrier, "pack", question)
    prompt_path = e6b.e6_phase_path(context.carrier, "prompts", question)
    if pack_path.exists() != prompt_path.exists():
        raise E6CError("E6c pack retains only one side of the prompt/trace pair")
    if not pack_path.exists():
        raise E6CError("E6c pack artifacts are missing")
    pack = e1.load_json(pack_path, sealed=True)
    prompt = e1.load_json(prompt_path, sealed=True)
    if pack != expected_pack or prompt != expected_prompt:
        raise E6CError("saved E6c pack differs from exact deterministic replay")
    return pack, prompt, packed, results, accounting


def run_pack_phase(context: E6CContext) -> None:
    tokenizer = e1.DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=e1._snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    for completed, question in enumerate(context.e1.selected, start=1):
        pack_path = e6b.e6_phase_path(context.carrier, "pack", question)
        prompt_path = e6b.e6_phase_path(context.carrier, "prompts", question)
        if pack_path.exists() or prompt_path.exists():
            _, _, packed, _, _ = replay_pack_question(
                context,
                question,
                tokenizer=tokenizer,
            )
            action = "verified"
        else:
            selection, results, _ = replay_selection_question(context, question)
            saved_selection = e6b.e6_phase_path(context.carrier, "selection", question)
            _write_or_verify(saved_selection, selection)
            pack, prompts, packed = _compute_pack_artifacts(
                context,
                question,
                selection=selection,
                results=results,
                tokenizer=tokenizer,
            )
            e1.write_json(prompt_path, prompts)
            e1.write_json(pack_path, pack)
            replay_pack_question(context, question, tokenizer=tokenizer)
            action = "built"
        print(
            f"  pack: {completed}/{E6C_SAMPLE} {action} {question.question_id} "
            + " ".join(
                f"{arm}={packed[arm].packed.trace['final_prompt']['tokens']}t/"
                f"{len(packed[arm].packed.trace['kept_ids'])}v"
                for arm in ARMS
            ),
            file=sys.stderr,
            flush=True,
        )


def _session_digest(value: str) -> str:
    return e1.sha256_bytes(value.encode("utf-8"))


def _candidate_session_sha256s(
    question: e1.SelectedQuestion,
    result: RepresentationResult,
) -> list[str]:
    session_ids = [str(value) for value in question.record["haystack_session_ids"]]
    return [
        _session_digest(session_ids[value.turn.turn_id.session_position])
        for value in result.hydrated_values
    ]


def _prompt_value_ids(
    result: RepresentationResult,
    packed: RepresentationPromptPackingResult,
) -> list[str]:
    values_by_turn = {value.turn.turn_id: value for value in result.hydrated_values}
    output: list[str] = []
    for payload in packed.packed.trace["kept_ids"]:
        turn_id = e1._turn_id_from_payload(payload)
        value = values_by_turn.get(turn_id)
        if value is None:
            raise E6CError("E6c packed prompt contains a non-hydrated turn")
        output.append(value.value_id)
    return output


def _context_metrics(
    ordered_sessions: Sequence[str], gold_sessions: Sequence[str]
) -> dict[str, Any]:
    gold = set(gold_sessions)
    hit = set(ordered_sessions) & gold
    first = next(
        (rank for rank, session in enumerate(ordered_sessions, start=1) if session in gold),
        None,
    )
    return {
        "any": bool(hit) if gold else None,
        "all": (hit == gold) if gold else None,
        "answer_session_recall": len(hit) / len(gold) if gold else None,
        "answer_session_mrr": 1.0 / first if first is not None else (0.0 if gold else None),
    }


def _deduplicated_sessions(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _average_precision(
    ordered_sessions: Sequence[str], gold_sessions: Sequence[str]
) -> float | None:
    if not gold_sessions:
        return None
    ordered = _deduplicated_sessions(ordered_sessions)
    gold = set(gold_sessions)
    hits = 0
    total = 0.0
    for rank, session in enumerate(ordered, start=1):
        if session in gold:
            hits += 1
            total += hits / rank
    return total / len(gold)


def _ndcg(ordered_sessions: Sequence[str], gold_sessions: Sequence[str]) -> float | None:
    if not gold_sessions:
        return None
    ordered = _deduplicated_sessions(ordered_sessions)
    gold = set(gold_sessions)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, session in enumerate(ordered, start=1)
        if session in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, len(gold) + 1))
    return dcg / ideal


def _numeric_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise E6CError("cannot summarize an empty E6c numeric series")
    normalized = [float(value) for value in values]
    total: int | float = (
        sum(int(value) for value in values)
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values)
        else sum(normalized)
    )
    return {
        "count": len(values),
        "total": total,
        "mean": float(total) / len(values),
        "min": min(values),
        "p50": percentile(normalized, 0.50),
        "p95": percentile(normalized, 0.95),
        "max": max(values),
    }


def _qa_arm_order(
    context: e6b.E6Context,
    question: e1.SelectedQuestion,
) -> tuple[str, ...]:
    question_type = str(question.record["question_type"])
    stratum = [
        candidate
        for candidate in context.e1.selected
        if str(candidate.record["question_type"]) == question_type
    ]
    try:
        within_stratum = next(
            index
            for index, candidate in enumerate(stratum)
            if candidate.position == question.position
            and candidate.question_id == question.question_id
        )
    except StopIteration as exc:
        raise E6CError("QA question is outside its frozen capability stratum") from exc
    offset = int(e1.sha256_bytes(question_type.encode("utf-8"))[:8], 16) % len(QA_PERMUTATIONS)
    return QA_PERMUTATIONS[(offset + within_stratum) % len(QA_PERMUTATIONS)]


def _configure_qa_carrier() -> None:
    _configure_carrier(cells=ARMS)
    e6b._qa_arm_order = _qa_arm_order


def _qa_binding(
    context: E6CContext,
    question: e1.SelectedQuestion,
    *,
    tokenizer: e1.DeepSeekExactTokenizer,
) -> dict[str, Any] | None:
    qa_path = e6b.e6_phase_path(context.carrier, "qa", question)
    receipt_path = e6b.e6_jsonl_path(context.carrier, "qa-receipts", question)
    if qa_path.exists() != receipt_path.exists():
        raise E6CError("E6c QA retains only one side of its artifact/receipt pair")
    if not qa_path.exists():
        return None
    _configure_qa_carrier()
    row = e1.load_json(qa_path, sealed=True)
    receipts = e1._load_receipts(receipt_path)
    e6b.replay_qa_question(
        context.carrier,
        question,
        qa_row=row,
        receipts=receipts,
        tokenizer=tokenizer,
    )
    return {
        "artifact_sha256": row["artifact_sha256"],
        "receipt_jsonl_sha256": e1.sha256_bytes(e1._receipt_bytes(receipts)),
        "arm_order": row["arm_order"],
        "cost_microusd": row["cost_microusd"],
        "arms": {
            arm: {
                "correct": bool(row["arms"][arm]["development_label"]),
                "reader_receipt_sha256": row["arms"][arm]["reader"]["receipt_sha256"],
                "judge_receipt_sha256": row["arms"][arm]["development_judge"]["receipt_sha256"],
            }
            for arm in ARMS
        },
    }


def _case_payload(
    context: E6CContext,
    question: e1.SelectedQuestion,
    *,
    case_index: int,
    prompt: Mapping[str, Any],
    packed: Mapping[str, RepresentationPromptPackingResult],
    results: Mapping[str, RepresentationResult],
    accounting: Mapping[str, Mapping[str, Any]],
    tokenizer: e1.DeepSeekExactTokenizer,
) -> dict[str, Any]:
    gold = (
        []
        if "_abs" in question.question_id
        else sorted(
            {_session_digest(str(value)) for value in question.record["answer_session_ids"]}
        )
    )
    qa = _qa_binding(context, question, tokenizer=tokenizer)
    arms: dict[str, Any] = {}
    for arm in ARMS:
        candidate_sessions = _candidate_session_sha256s(question, results[arm])
        prompt_ids = _prompt_value_ids(results[arm], packed[arm])
        session_by_value = dict(
            zip(
                [value.value_id for value in results[arm].hydrated_values],
                candidate_sessions,
                strict=True,
            )
        )
        prompt_sessions = [session_by_value[value_id] for value_id in prompt_ids]
        arms[arm] = {
            "outer_arm_id": arm,
            "inner_registered_representation_cell": results[arm].cell.value,
            "representation_trace_sha256": results[arm].trace_sha256,
            "candidate_value_ids": [value.value_id for value in results[arm].hydrated_values],
            "candidate_session_sha256s": candidate_sessions,
            "prompt_value_ids": prompt_ids,
            "prompt_tokens": int(prompt["arms"][arm]["prompt_tokens"]),
            "prompt_sha256": str(prompt["arms"][arm]["prompt_sha256"]),
            "candidate": {
                **_context_metrics(candidate_sessions, gold),
                "average_precision": _average_precision(candidate_sessions, gold),
                "ndcg": _ndcg(candidate_sessions, gold),
            },
            "prompt": {
                **_context_metrics(prompt_sessions, gold),
                "average_precision": _average_precision(prompt_sessions, gold),
                "ndcg": _ndcg(prompt_sessions, gold),
            },
            "query_accounting": dict(accounting[arm]),
            "qa": None if qa is None else qa["arms"][arm],
        }
    return {
        "artifact_type": "swarmbrain-longmemeval-e6c-case",
        "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "case_index": case_index,
        "question_id": question.question_id,
        "question_position": question.position,
        "question_type": str(question.record["question_type"]),
        "abstention": "_abs" in question.question_id,
        "gold_session_sha256s": gold,
        "arms": arms,
        "qa_artifact": qa,
    }


def build_case_phase(context: E6CContext) -> tuple[dict[str, Any], ...]:
    tokenizer = e1.DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=e1._snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    qa_count = sum(
        e6b.e6_phase_path(context.carrier, "qa", question).is_file()
        for question in context.e1.selected
    )
    if qa_count not in {0, E6C_SAMPLE}:
        raise E6CError("E6c aggregate cases forbid partial QA coverage")
    case_phase = "cases-final" if qa_count == E6C_SAMPLE else "cases-context"
    cases: list[dict[str, Any]] = []
    for case_index, question in enumerate(context.e1.selected):
        _pack, prompt, packed, results, accounting = replay_pack_question(
            context,
            question,
            tokenizer=tokenizer,
        )
        artifact = e1.seal_artifact(
            _case_payload(
                context,
                question,
                case_index=case_index,
                prompt=prompt,
                packed=packed,
                results=results,
                accounting=accounting,
                tokenizer=tokenizer,
            )
        )
        _write_or_verify(e6b.e6_phase_path(context.carrier, case_phase, question), artifact)
        cases.append(artifact)
        print(
            f"  case: {case_index + 1}/{E6C_SAMPLE} sealed {question.question_id}",
            file=sys.stderr,
            flush=True,
        )
    return tuple(cases)


def _validate_case_sequence(
    context: E6CContext,
    cases: Sequence[Mapping[str, Any]],
) -> None:
    if len(cases) != E6C_SAMPLE:
        raise E6CError("E6c diagnostic requires all 160 frozen cases")
    for index, (case, question) in enumerate(zip(cases, context.e1.selected, strict=True)):
        if (
            case.get("artifact_sha256")
            != e1.sha256_json(
                {key: value for key, value in case.items() if key != "artifact_sha256"}
            )
            or case.get("case_index") != index
            or case.get("question_id") != question.question_id
            or case.get("question_position") != question.position
            or set(case.get("arms", {})) != set(ARMS)
        ):
            raise E6CError("E6c case sequence or seal drifted")


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        raise E6CError("cannot average an empty E6c series")
    return sum(float(value) for value in values) / len(values)


def _arm_context_summary(
    cases: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    eligible = [case for case in cases if not case["abstention"]]
    candidate = [case["arms"][arm]["candidate"] for case in eligible]
    prompt = [case["arms"][arm]["prompt"] for case in eligible]
    return {
        "gold_eligible_cases": len(eligible),
        "candidate": {
            axis: _mean([row[axis] for row in candidate])
            for axis in (
                "any",
                "all",
                "answer_session_recall",
                "answer_session_mrr",
                "average_precision",
                "ndcg",
            )
        },
        "prompt": {
            axis: _mean([row[axis] for row in prompt])
            for axis in (
                "any",
                "all",
                "answer_session_recall",
                "answer_session_mrr",
                "average_precision",
                "ndcg",
            )
        },
        "prompt_tokens": _numeric_summary(
            [int(case["arms"][arm]["prompt_tokens"]) for case in cases]
        ),
        "candidate_value_counts": _numeric_summary(
            [len(case["arms"][arm]["candidate_value_ids"]) for case in cases]
        ),
        "prompt_value_counts": _numeric_summary(
            [len(case["arms"][arm]["prompt_value_ids"]) for case in cases]
        ),
    }


def _context_contrast(
    cases: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
) -> dict[str, Any]:
    eligible = [case for case in cases if not case["abstention"]]
    strata = [str(case["question_type"]) for case in eligible]
    mrr_deltas = [
        float(case["arms"][MERGED_ARM]["prompt"]["answer_session_mrr"])
        - float(case["arms"][baseline]["prompt"]["answer_session_mrr"])
        for case in eligible
    ]
    return {
        "baseline": baseline,
        "candidate": MERGED_ARM,
        "prompt_answer_session_mrr": paired_lower_bound(mrr_deltas, strata),
        "observed_prompt_deltas": {
            axis: _mean(
                [
                    float(case["arms"][MERGED_ARM]["prompt"][axis])
                    - float(case["arms"][baseline]["prompt"][axis])
                    for case in eligible
                ]
            )
            for axis in (
                "any",
                "all",
                "answer_session_recall",
                "answer_session_mrr",
                "average_precision",
                "ndcg",
            )
        },
    }


def _context_gate(
    cases: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [case for case in cases if not case["abstention"]]
    exact_heads = all(
        len(case["arms"][arm]["candidate_value_ids"]) == HEAD_MATCHED_VALUE_COUNT
        for case in cases
        for arm in ARMS
    )
    exact_prompts = all(
        len(case["arms"][arm]["prompt_value_ids"]) == HEAD_MATCHED_VALUE_COUNT
        for case in cases
        for arm in ARMS
    )
    no_drops = all(
        case["arms"][arm]["candidate_value_ids"] == case["arms"][arm]["prompt_value_ids"]
        for case in cases
        for arm in ARMS
    )
    within_budget = all(
        0 < int(case["arms"][arm]["prompt_tokens"]) <= e1.TOKEN_BUDGET
        for case in cases
        for arm in ARMS
    )
    mrr_superiority = {
        baseline: {
            "delta": float(contrasts[baseline]["prompt_answer_session_mrr"]["delta"]),
            "one_sided_95_lower_bound": float(
                contrasts[baseline]["prompt_answer_session_mrr"]["lower_bound"]
            ),
            "passed": float(contrasts[baseline]["prompt_answer_session_mrr"]["lower_bound"]) > 0.0,
        }
        for baseline in (RAW_ARM, RRF_ARM)
    }
    coverage = {
        axis: {
            RRF_ARM: float(summaries[RRF_ARM]["prompt"][axis]),
            MERGED_ARM: float(summaries[MERGED_ARM]["prompt"][axis]),
            "delta": float(summaries[MERGED_ARM]["prompt"][axis])
            - float(summaries[RRF_ARM]["prompt"][axis]),
            "passed": float(summaries[MERGED_ARM]["prompt"][axis])
            >= float(summaries[RRF_ARM]["prompt"][axis]),
        }
        for axis in ("any", "all", "answer_session_recall")
    }
    tokens = {
        statistic: {
            RAW_ARM: summaries[RAW_ARM]["prompt_tokens"][statistic],
            RRF_ARM: summaries[RRF_ARM]["prompt_tokens"][statistic],
            MERGED_ARM: summaries[MERGED_ARM]["prompt_tokens"][statistic],
            "maximum_allowed": min(
                float(summaries[RAW_ARM]["prompt_tokens"][statistic]),
                float(summaries[RRF_ARM]["prompt_tokens"][statistic]),
            ),
            "passed": float(summaries[MERGED_ARM]["prompt_tokens"][statistic])
            <= min(
                float(summaries[RAW_ARM]["prompt_tokens"][statistic]),
                float(summaries[RRF_ARM]["prompt_tokens"][statistic]),
            ),
        }
        for statistic in ("total", "p95")
    }
    passed = bool(
        len(eligible) == E6C_SAMPLE - E6C_ABSTENTION_COUNT
        and exact_heads
        and exact_prompts
        and no_drops
        and within_budget
        and all(row["passed"] for row in mrr_superiority.values())
        and all(row["passed"] for row in coverage.values())
        and all(row["passed"] for row in tokens.values())
    )
    return {
        "gate": "G1-confirmatory-context",
        "passed": passed,
        "gold_eligible_cases": len(eligible),
        "required_gold_eligible_cases": E6C_SAMPLE - E6C_ABSTENTION_COUNT,
        "candidate_heads_exact20_every_arm_case": exact_heads,
        "complete_prompt_values_exact20_every_arm_case": exact_prompts,
        "zero_whole_turn_drops_every_arm_case": no_drops,
        "complete_prompts_within_8192_tokens": within_budget,
        "paired_MRR_superiority": mrr_superiority,
        "observed_coverage_preservation_vs_R1": coverage,
        "token_nonregression_vs_both_comparators": tokens,
        "multiplicity": (
            "conjunctive-intersection-union-gate; all contrasts and safeguards must pass; "
            "no comparator selected after outcomes"
        ),
    }


def _qa_contrast(
    cases: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
) -> dict[str, Any]:
    deltas = [
        float(bool(case["arms"][MERGED_ARM]["qa"]["correct"]))
        - float(bool(case["arms"][baseline]["qa"]["correct"]))
        for case in cases
    ]
    strata = [str(case["question_type"]) for case in cases]
    inference = paired_lower_bound(deltas, strata)
    by_type: dict[str, Any] = {}
    for question_type in sorted({str(case["question_type"]) for case in cases}):
        rows = [case for case in cases if str(case["question_type"]) == question_type]
        baseline_accuracy = _mean(
            [int(bool(case["arms"][baseline]["qa"]["correct"])) for case in rows]
        )
        candidate_accuracy = _mean(
            [int(bool(case["arms"][MERGED_ARM]["qa"]["correct"])) for case in rows]
        )
        delta = candidate_accuracy - baseline_accuracy
        by_type[question_type] = {
            "questions": len(rows),
            "baseline_accuracy": baseline_accuracy,
            "M20_accuracy": candidate_accuracy,
            "delta": delta,
            "margin": -QA_MAX_TYPE_REGRESSION,
            "passed": delta >= -QA_MAX_TYPE_REGRESSION,
        }
    abstention = [case for case in cases if case["abstention"]]
    abstention_delta = _mean(
        [
            int(bool(case["arms"][MERGED_ARM]["qa"]["correct"]))
            - int(bool(case["arms"][baseline]["qa"]["correct"]))
            for case in abstention
        ]
    )
    return {
        "baseline": baseline,
        "candidate": MERGED_ARM,
        "inference": inference,
        "practical_effect_floor": QA_PRACTICAL_EFFECT_FLOOR,
        "point_delta_passed": float(inference["delta"]) >= QA_PRACTICAL_EFFECT_FLOOR,
        "lower_bound_passed": float(inference["lower_bound"]) > 0.0,
        "by_question_type": by_type,
        "all_types_within_no_harm_margin": all(row["passed"] for row in by_type.values()),
        "abstention": {
            "questions": len(abstention),
            "delta": abstention_delta,
            "passed": abstention_delta >= 0.0,
        },
    }


def _qa_gate(
    cases: Sequence[Mapping[str, Any]],
    *,
    context_gate_passed: bool,
) -> dict[str, Any]:
    qa_rows = [case["qa_artifact"] for case in cases]
    available = [row is not None for row in qa_rows]
    if not any(available):
        return {
            "gate": "G2-paired-DeepSeek-QA",
            "available": False,
            "passed": False,
            "reason": "conditional-QA-not-executed",
        }
    if not all(available):
        raise E6CError("E6c G2 forbids partial QA inference")
    contrasts = {
        baseline: _qa_contrast(cases, baseline=baseline) for baseline in (RAW_ARM, RRF_ARM)
    }
    order_counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for case in cases:
        order_counts[str(case["question_type"])][tuple(case["qa_artifact"]["arm_order"])] += 1
    balanced = all(
        max(counter.values()) - min(counter.get(order, 0) for order in QA_PERMUTATIONS) <= 1
        for counter in order_counts.values()
    )
    passed = bool(
        context_gate_passed
        and balanced
        and all(
            contrast["point_delta_passed"]
            and contrast["lower_bound_passed"]
            and contrast["all_types_within_no_harm_margin"]
            and contrast["abstention"]["passed"]
            for contrast in contrasts.values()
        )
    )
    return {
        "gate": "G2-paired-DeepSeek-QA",
        "available": True,
        "passed": passed,
        "G1_passed": context_gate_passed,
        "questions": len(cases),
        "reader_calls": len(cases) * len(ARMS),
        "development_judge_calls": len(cases) * len(ARMS),
        "within_type_six_permutation_balance_passed": balanced,
        "arm_order_counts_by_type": {
            question_type: {">".join(order): counter.get(order, 0) for order in QA_PERMUTATIONS}
            for question_type, counter in sorted(order_counts.items())
        },
        "contrasts": contrasts,
        "reader_judge_protocol_scope": "same-mutable-DeepSeek-v4-flash-alias",
    }


def build_diagnostic_report(context: E6CContext) -> dict[str, Any]:
    cases = build_case_phase(context)
    _validate_case_sequence(context, cases)
    summaries = {arm: _arm_context_summary(cases, arm) for arm in ARMS}
    contrasts = {
        baseline: _context_contrast(cases, baseline=baseline) for baseline in (RAW_ARM, RRF_ARM)
    }
    g1 = _context_gate(cases, summaries, contrasts)
    g2 = _qa_gate(cases, context_gate_passed=bool(g1["passed"]))
    qa_complete = g2.get("available") is True
    source_cases = [
        {
            "case_index": int(case["case_index"]),
            "question_id": str(case["question_id"]),
            "artifact_sha256": str(case["artifact_sha256"]),
        }
        for case in cases
    ]
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6c-diagnostic",
        "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "classification": "same-benchmark-confirmation-not-official-score",
        "case_count": len(cases),
        "question_type_counts": dict(
            sorted(Counter(str(case["question_type"]) for case in cases).items())
        ),
        "abstention_count": sum(bool(case["abstention"]) for case in cases),
        "source_cases": source_cases,
        "source_cases_sha256": e1.sha256_json(source_cases),
        "context_quality": {"arms": summaries, "contrasts": contrasts},
        "gates": {"G1": g1, "G2": g2},
        "qa_complete": qa_complete,
        "claims": {
            "MRR_gold_unit": "answer-session-ID",
            "MRR_rank_unit": "ordered-raw-turn-with-first-gold-session-hit",
            "average_precision_and_ndcg_deduplicate_sessions_at_first_occurrence": True,
            "coverage_deltas_are_observed_not_population-noninferiority-intervals": True,
            "official_longmemeval_score": False,
            "SOTA_claim": False,
        },
    }
    report = e1.seal_artifact(payload)
    name = "diagnostic-final.json" if qa_complete else "diagnostic-context.json"
    _write_or_verify(context.output_dir / name, report)
    return report


def _qa_artifact_count(context: E6CContext) -> int:
    return sum(
        e6b.e6_phase_path(context.carrier, "qa", question).is_file()
        for question in context.e1.selected
    )


def _load_context_diagnostic(context: E6CContext) -> dict[str, Any]:
    path = context.output_dir / "diagnostic-context.json"
    if not path.is_file():
        raise E6CError("partial QA resume requires the frozen context-only diagnostic")
    report = e1.load_json(path, sealed=True)
    if (
        report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("run_manifest_sha256") != context.manifest["artifact_sha256"]
        or report.get("qa_complete") is not False
    ):
        raise E6CError("context-only diagnostic binding drifted")
    return report


def run_qa_phase(
    context: E6CContext,
    *,
    base_url: str,
    api_key_env: str,
) -> dict[str, Any]:
    qa_count = _qa_artifact_count(context)
    diagnostic = (
        build_diagnostic_report(context) if qa_count == 0 else _load_context_diagnostic(context)
    )
    g1 = diagnostic["gates"]["G1"]
    if g1.get("passed") is not True:
        if qa_count or e6b._qa_durable_state_exists(context.carrier):
            raise E6CError("rejected context gate is incompatible with retained QA state")
        print(
            "  qa: E6c G1 rejected M20; no reader or judge calls executed",
            file=sys.stderr,
            flush=True,
        )
        return diagnostic
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise E6CError("E6c QA is frozen to the official DeepSeek endpoint")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise E6CError(f"environment variable {api_key_env!r} is missing")
    tokenizer = e1.DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=e1._snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    _configure_qa_carrier()
    asyncio.run(
        e6b._run_qa_async(
            context.carrier,
            selected=context.e1.selected,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key=api_key,
        )
    )
    if _qa_artifact_count(context) != E6C_SAMPLE:
        raise E6CError("E6c QA returned without complete three-arm coverage")
    completion = e1.seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6c-qa-completion",
            "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "question_count": E6C_SAMPLE,
            "arms": list(ARMS),
            "reader_calls": E6C_SAMPLE * len(ARMS),
            "development_judge_calls": E6C_SAMPLE * len(ARMS),
            "aggregate_QA_statistics_compiled": False,
            "individual_labels_disclosed_by_progress_log": False,
            "official_gpt4o_calls": 0,
        }
    )
    _write_or_verify(context.output_dir / "qa-completion.json", completion)
    return build_diagnostic_report(context)


def _artifact_binding(path: Path, *, sealed: bool) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise E6CError(f"required E6c artifact is missing or unsafe: {path}")
    raw = path.read_bytes()
    binding = {
        "path": str(path),
        "bytes": len(raw),
        "file_sha256": e1.sha256_bytes(raw),
    }
    if sealed:
        value = e1.load_json(path, sealed=True)
        binding["artifact_sha256"] = value["artifact_sha256"]
    return binding


def _phase_artifacts(
    context: E6CContext,
    *,
    phase: str,
    sealed: bool = True,
    e1_namespace: bool = False,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for question in context.e1.selected:
        path = (
            e1.phase_path(context.e1, phase, question)
            if e1_namespace
            else (
                e6b.e6_jsonl_path(context.carrier, phase, question)
                if not sealed
                else e6b.e6_phase_path(context.carrier, phase, question)
            )
        )
        output.append(_artifact_binding(path, sealed=sealed))
    return output


def build_report(context: E6CContext) -> dict[str, Any]:
    qa_count = _qa_artifact_count(context)
    if qa_count not in {0, E6C_SAMPLE}:
        raise E6CError("final E6c report forbids partial QA coverage")
    diagnostic = build_diagnostic_report(context)
    _configure_carrier(cells=ARMS if qa_count else (RAW_ARM, RRF_ARM))
    journal_cost, unresolved = e6b._external_journal_cost(context.carrier)
    if unresolved:
        raise E6CError("final E6c report forbids unresolved external call reservations")

    artifacts: dict[str, list[dict[str, Any]]] = {
        "dense": _phase_artifacts(context, phase="dense", e1_namespace=True),
        "extraction_jsonl": _phase_artifacts(context, phase="extraction", sealed=False),
        "extraction_summary": _phase_artifacts(context, phase="extraction-summary"),
        "ranking": _phase_artifacts(context, phase="ranking"),
        "selection": _phase_artifacts(context, phase="selection"),
        "pack": _phase_artifacts(context, phase="pack"),
        "prompts": _phase_artifacts(context, phase="prompts"),
    }
    case_phase = "cases-final" if qa_count else "cases-context"
    artifacts[case_phase] = _phase_artifacts(context, phase=case_phase)
    if qa_count:
        artifacts["qa"] = _phase_artifacts(context, phase="qa")
        artifacts["qa_receipts"] = _phase_artifacts(
            context,
            phase="qa-receipts",
            sealed=False,
        )

    extraction_rows = [
        e1.load_json(
            e6b.e6_phase_path(context.carrier, "extraction-summary", question),
            sealed=True,
        )
        for question in context.e1.selected
    ]
    extraction_cost = sum(int(row["accounting"]["cost_microusd"]) for row in extraction_rows)
    extraction_values = sum(int(row["source_value_count"]) for row in extraction_rows)
    extraction_applications = sum(
        int(row["accounting"]["application_attempts"]) for row in extraction_rows
    )
    qa_cost = 0
    if qa_count:
        qa_cost = sum(
            int(
                e1.load_json(
                    e6b.e6_phase_path(context.carrier, "qa", question),
                    sealed=True,
                )["cost_microusd"]
            )
            for question in context.e1.selected
        )
    if journal_cost != extraction_cost + qa_cost:
        raise E6CError("external journal cost differs from extraction and QA evidence")

    g1 = diagnostic["gates"]["G1"]
    g2 = diagnostic["gates"]["G2"]
    g0 = {
        "gate": "G0-integrity-replay",
        "passed": bool(
            len(context.e1.selected) == E6C_SAMPLE
            and extraction_values == sum(len(question.turns) for question in context.e1.selected)
            and all(len(rows) == E6C_SAMPLE for rows in artifacts.values())
            and unresolved == 0
        ),
        "questions": len(context.e1.selected),
        "source_values": extraction_values,
        "unresolved_external_call_reservations": unresolved,
        "all_dense_extraction_ranking_selection_pack_case_artifacts_replayed": True,
        "all_raw_provider_request_response_usage_receipts_retained": True,
    }
    if not g0["passed"]:
        verdict = "ineligible-integrity-failure"
    elif not g1["passed"]:
        verdict = "reject-M20-at-confirmatory-context-gate"
    elif not qa_count:
        verdict = "incomplete-conditional-QA-required"
    elif not g2["passed"]:
        verdict = "reject-M20-at-paired-DeepSeek-QA-gate"
    else:
        verdict = "pass-fresh-LongMemEval-same-benchmark-DeepSeek-confirmation"

    artifact_sets = {
        phase: {
            "count": len(rows),
            "rows_sha256": e1.sha256_json(rows),
            "rows": rows,
        }
        for phase, rows in artifacts.items()
    }
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6c-final-report",
        "schema_version": e1.ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "verdict": verdict,
        "classification": "same-benchmark-confirmation-not-official-or-SOTA-score",
        "question_count": E6C_SAMPLE,
        "source_turn_count": extraction_values,
        "artifacts": artifact_sets,
        "artifacts_sha256": e1.sha256_json(
            {phase: rows["rows_sha256"] for phase, rows in artifact_sets.items()}
        ),
        "diagnostic_artifact_sha256": diagnostic["artifact_sha256"],
        "gates": {"G0": g0, "G1": g1, "G2": g2},
        "model_calls": {
            "DeepSeek_source_only_extraction_applications": extraction_applications,
            "DeepSeek_reader_calls": qa_count * len(ARMS),
            "DeepSeek_development_judge_calls": qa_count * len(ARMS),
            "official_GPT4o_calls": 0,
        },
        "cost": {
            "source_only_extraction_microusd": extraction_cost,
            "reader_and_development_judge_microusd": qa_cost,
            "total_external_microusd": journal_cost,
            "engineering_ledger_ceiling_microusd": ENGINEERING_LEDGER_CEILING_MICROUSD,
            "within_engineering_ceiling": (journal_cost <= ENGINEERING_LEDGER_CEILING_MICROUSD),
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
    report = e1.seal_artifact(payload)
    _write_or_verify(context.output_dir / "report.json", report)
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
            "diagnose",
            "qa",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1-output-dir", type=Path, default=DEFAULT_E1_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-root", type=Path, default=e1.DEFAULT_QWEN_ROOT)
    parser.add_argument("--cross-encoder-root", type=Path, default=e1.DEFAULT_CE_ROOT)
    parser.add_argument("--deepseek-root", type=Path, default=e1.DEFAULT_DEEPSEEK_ROOT)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    return parser


def _run_phase(context: E6CContext, args: argparse.Namespace, phase: str) -> None:
    if phase == "preflight":
        print(
            json.dumps(
                {
                    "manifest_sha256": context.manifest["artifact_sha256"],
                    "cohort": context.cohort_binding["digests"],
                    "questions": E6C_SAMPLE,
                    "source_turns": sum(len(question.turns) for question in context.e1.selected),
                },
                sort_keys=True,
            )
        )
    elif phase == "dense":
        e1.run_dense_phase(
            context.e1,
            device=QWEN_DEVICE,
            batch_size=QWEN_BATCH_SIZE,
        )
    elif phase == "extract":
        _configure_carrier()
        e6b.run_extraction_phase(
            context.carrier,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    elif phase == "rank":
        _configure_carrier()
        e6b.run_rank_phase(
            context.carrier,
            device=QWEN_DEVICE,
            batch_size=QWEN_BATCH_SIZE,
        )
    elif phase == "select":
        run_selection_phase(context)
    elif phase == "pack":
        run_pack_phase(context)
    elif phase == "diagnose":
        report = build_diagnostic_report(context)
        print(json.dumps(report["gates"], sort_keys=True))
    elif phase == "qa":
        report = run_qa_phase(
            context,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
        print(json.dumps(report["gates"], sort_keys=True))
    elif phase == "report":
        report = build_report(context)
        print(json.dumps({"verdict": report["verdict"], "gates": report["gates"]}))
    else:  # pragma: no cover - parser constrains phase
        raise E6CError(f"unknown E6c phase: {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = build_context(args)
    phases = (
        ("dense", "extract", "rank", "select", "pack", "diagnose", "qa", "report")
        if args.phase == "all"
        else (args.phase,)
    )
    with e6b._output_process_lock(context.output_dir):
        for phase in phases:
            _run_phase(context, args, phase)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
