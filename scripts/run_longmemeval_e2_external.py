#!/usr/bin/env python3
"""Pinned real-model E2-A/E2-D/E2-E organization development experiment.

The source is the immutable E1-A top-20 from a completed E1 protocol-v3 run.
Qwen3-Embedding supplies candidate-to-concatenated-chain cosines to the frozen
Chain-of-Memory organizer.  The parity organization (E2-D) is preserved as a
diagnostic, while its explicit cross-chain-deduplicated rendering (E2-E) is
packed and compared with retrieval-order control E2-A under one exact
8,192-token DeepSeek prompt budget.  Nothing here changes production serving.
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
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from benchmarks.integrations.chain_of_memory import (
    ChainCandidate,
    ContextCosine,
    E2Cell,
    ExternalSimilarityEvidence,
    K,
    PrecomputedContextSimilarities,
    chain_context_sha256,
    organize_e2,
)
from benchmarks.integrations.longmemeval_turn_prompt import OrderedTurnBlocks, pack_turn_prompt
from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId, TurnProjection
from run_longmemeval_e1_external import (
    ARTIFACT_SCHEMA_VERSION,
    CROSS_ENCODER_BATCH_SIZE,
    DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION,
    DEEPSEEK_MODEL,
    DEEPSEEK_OUTPUT_USD_PER_MILLION,
    DEFAULT_CE_ROOT,
    DEFAULT_DATASET,
    DEFAULT_DEEPSEEK_ROOT,
    DEFAULT_QWEN_ROOT,
    DEFAULT_SAMPLE,
    DEFAULT_SEED,
    QWEN_BATCH_SIZE,
    QWEN_MODEL,
    QWEN_REVISION,
    TOKEN_BUDGET,
    ChatClient,
    ChatResult,
    DeepSeekExactTokenizer,
    ExperimentContext,
    ExternalE1Error,
    QwenEmbedder,
    SelectedQuestion,
    _judge_text_for_arm,
    _load_receipts,
    _mean,
    _paired_bootstrap_interval,
    _receipt_bytes,
    _retrieval_metrics,
    _safe_question_id,
    _snapshot_artifact,
    _turn_id_from_payload,
    _turn_id_payload,
    _upper_bound_cost,
    _wilson_interval,
    atomic_write,
    build_context,
    chat_receipt_record,
    is_abstention_question,
    judge_label,
    load_json,
    phase_path,
    qwen_query_text,
    replay_dense_row,
    seal_artifact,
    sha256_bytes,
    sha256_json,
    validate_chat_receipt_record,
    validate_sealed_artifact,
    verify_provider_prompt_tokens,
    write_json,
)

E2_RUN_PROTOCOL_VERSION = "swarmbrain-longmemeval-e2-real-model-development-v1"
E2_CELLS = ("E2-A", "E2-D", "E2-E")
DEFAULT_E1_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e1-pilot-v3")
DEFAULT_E2_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e2-pilot-v1")

IMPLEMENTATION_FILES = (
    "benchmarks/integrations/chain_of_memory/contracts.py",
    "benchmarks/integrations/chain_of_memory/organizer.py",
    "benchmarks/integrations/longmemeval_turn_prompt/contracts.py",
    "benchmarks/integrations/longmemeval_turn_prompt/packer.py",
    "scripts/run_longmemeval_e1_external.py",
    "scripts/run_longmemeval_e2_external.py",
)


class ExternalE2Error(ExternalE1Error):
    """Saved E2 evidence cannot support deterministic replay."""


@dataclass(frozen=True, slots=True)
class E2Context:
    e1: ExperimentContext
    output_dir: Path
    manifest: dict[str, Any]


def implementation_fingerprint() -> dict[str, Any]:
    files = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_FILES
    }
    return {"files": files, "tree_sha256": sha256_json(files)}


def _e1_namespace(args: argparse.Namespace) -> Namespace:
    return Namespace(
        dataset=args.dataset,
        output_dir=args.e1_output_dir,
        sample=args.sample,
        seed=args.seed,
        qwen_root=args.qwen_root,
        cross_encoder_root=args.cross_encoder_root,
        deepseek_root=args.deepseek_root,
        device=args.device,
        qwen_batch_size=args.qwen_batch_size,
        cross_encoder_batch_size=CROSS_ENCODER_BATCH_SIZE,
    )


def build_e2_context(args: argparse.Namespace) -> E2Context:
    e1 = build_context(_e1_namespace(args))
    dense_sources: list[dict[str, Any]] = []
    for question in e1.selected:
        path = phase_path(e1, "dense", question)
        if not path.exists():
            raise ExternalE2Error("E2 requires every source E1 dense artifact")
        row = load_json(path, sealed=True)
        result = replay_dense_row(e1, question, row)
        dense_sources.append(
            {
                "question_id": question.question_id,
                "artifact_sha256": row["artifact_sha256"],
                "e1a_trace_sha256": result.trace_sha256,
            }
        )
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e2-real-model-run-manifest",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "classification": "development-experiment-not-official-longmemeval-score",
        "production_configuration": False,
        "source_e1_manifest_sha256": e1.manifest["artifact_sha256"],
        "source_e1_dense_artifacts": dense_sources,
        "source_e1_dense_artifacts_sha256": sha256_json(dense_sources),
        "dataset": e1.manifest["dataset"],
        "turn_projection": e1.manifest["turn_projection"],
        "sample": e1.manifest["sample"],
        "cells": list(E2_CELLS),
        "organization": {
            "candidate_source": "frozen-E1-A-top-20",
            "primary_cell": "E2-D",
            "deduplicated_rendering_control": "E2-E",
            "deferred_cell": "E2-C-until-adaptive-cell-shows-signal",
            "context_embedding_max_length": 8192,
            "qwen_padding_side": "right",
        },
        "prompt": {
            "complete_reader_prompt_token_budget": TOKEN_BUDGET,
            "E2-A_layout": "linear-e1",
            "E2-E_layout": "ordered-com-blocks-cross-chain-deduplicated",
            "reader_cells": ["E2-A", "E2-E"],
            "E2-D": "organization-only-v1-packer-rejects-cross-chain-duplicate-IDs",
        },
        "reader_and_development_judge": e1.manifest["reader_and_development_judge"],
        "model_snapshots": {
            "qwen_embedding": e1.manifest["model_snapshots"]["qwen_embedding"],
            "deepseek_tokenizer": e1.manifest["model_snapshots"]["deepseek_tokenizer"],
        },
        "local_execution": {
            "device": args.device,
            "qwen_maximum_batch_size": args.qwen_batch_size,
        },
        "implementation": implementation_fingerprint(),
        "claims": {
            "gold_fields_used_for_organization": False,
            "E2-C_executed": False,
            "paper_parity": False,
            "official_longmemeval_score": False,
            "production_policy_changed": False,
        },
    }
    manifest = seal_artifact(payload)
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if load_json(manifest_path, sealed=True) != manifest:
            raise ExternalE2Error("E2 output directory belongs to a different manifest")
    else:
        write_json(manifest_path, manifest)
    return E2Context(e1=e1, output_dir=output_dir, manifest=manifest)


def e2_phase_path(context: E2Context, phase: str, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.json"
    return context.output_dir / phase / name


def e2_receipt_path(context: E2Context, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.jsonl"
    return context.output_dir / "receipts" / name


def _dense_source(context: E2Context, question: SelectedQuestion):
    path = phase_path(context.e1, "dense", question)
    row = load_json(path, sealed=True)
    return row, replay_dense_row(context.e1, question, row)


def _candidate_pool(
    question: SelectedQuestion,
    e1a: Any,
    dense_row: Mapping[str, Any],
) -> tuple[ChainCandidate, ...]:
    raw_by_id = {
        _turn_id_from_payload(row["turn_id"]): float(row["raw_cosine"])
        for row in dense_row["dense"]["observations"]
    }
    pool = tuple(
        ChainCandidate(
            turn=candidate.turn,
            query_cosine=raw_by_id[candidate.turn_id],
        )
        for candidate in e1a.candidates[:K]
    )
    if len(pool) != K or any(item.turn_id.question_id != question.question_id for item in pool):
        raise ExternalE2Error("E2 candidate pool is not the exact E1-A top-20")
    return pool


def _similarity_identity_payload(identity: ExternalSimilarityEvidence) -> dict[str, str]:
    return {
        "producer": identity.producer,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "artifact_sha256": identity.artifact_sha256,
    }


def _similarity_identity_from_payload(value: Any) -> ExternalSimilarityEvidence:
    if not isinstance(value, dict):
        raise ExternalE2Error("similarity identity must be an object")
    return ExternalSimilarityEvidence(
        producer=str(value["producer"]),
        model_id=str(value["model_id"]),
        model_revision=str(value["model_revision"]),
        artifact_sha256=str(value["artifact_sha256"]),
    )


def _context_entries_from_provisional(
    provisional: Any,
    *,
    candidates: tuple[ChainCandidate, ...],
) -> tuple[ContextCosine, ...]:
    turns_by_id = {candidate.turn_id: candidate.turn for candidate in candidates}
    chains: dict[LongMemEvalTurnId, list[TurnProjection]] = {
        anchor.turn_id: [anchor.turn] for anchor in candidates[:3]
    }
    entries: list[ContextCosine] = []
    for decision in provisional.decisions:
        chain = chains[decision.anchor_turn_id]
        for score in decision.scorecard:
            if score.context_cosine is None:
                raise ExternalE2Error("product decision omitted its context cosine")
            entries.append(
                ContextCosine.from_turns(
                    tuple(chain),
                    turns_by_id[score.candidate_turn_id],
                    score.context_cosine,
                )
            )
        if decision.appended:
            chain.append(turns_by_id[decision.best_candidate_turn_id])
    return tuple(entries)


QUERY_REPLAY_ABSOLUTE_TOLERANCE = 1e-3


def _embedded_batch_binding(batch: Any, texts: Sequence[str]) -> dict[str, Any]:
    if len(texts) != len(batch.token_counts):
        raise ExternalE2Error("embedding accounting does not align with its inputs")
    retry_positions = set(batch.singleton_retry_positions)
    return {
        "inputs": [
            {
                "position": position,
                "input_sha256": sha256_bytes(text.encode("utf-8")),
                "input_utf8_bytes": len(text.encode("utf-8")),
                "input_tokens_after_truncation": batch.token_counts[position],
                "truncated_right_at_8192": batch.truncated[position],
                "singleton_retry_after_nonfinite_batch": position in retry_positions,
            }
            for position, text in enumerate(texts)
        ],
        "batch_plan": [list(group) for group in batch.batch_plan],
        "accounting": {
            "input_count": len(texts),
            "input_tokens_after_truncation": sum(batch.token_counts),
            "truncated_inputs": sum(batch.truncated),
            "singleton_retries_after_nonfinite_batch": len(retry_positions),
            "model_batches": batch.model_batches,
            "padded_model_input_tokens": batch.padded_tokens,
            "padded_attention_cells": batch.padded_attention_cells,
            "model_elapsed_ms": batch.elapsed_ms,
        },
    }


def _context_entry_from_payload(value: Any) -> ContextCosine:
    if not isinstance(value, dict):
        raise ExternalE2Error("context cosine entry must be an object")
    chain_ids = value.get("chain_turn_ids")
    if not isinstance(chain_ids, list):
        raise ExternalE2Error("context cosine chain IDs must be an array")
    return ContextCosine(
        chain_turn_ids=tuple(_turn_id_from_payload(item) for item in chain_ids),
        candidate_turn_id=_turn_id_from_payload(value.get("candidate_turn_id")),
        chain_context_sha256=str(value.get("chain_context_sha256")),
        candidate_document_sha256=str(value.get("candidate_document_sha256")),
        cosine=float(value.get("cosine")),
    )


def _evolution_binding(result: Any) -> dict[str, Any]:
    return {
        "chains": [chain.content_free_binding() for chain in result.chains],
        "decisions": [decision.content_free_binding() for decision in result.decisions],
        "context_similarity_calls": result.context_similarity_calls,
        "context_similarity_table_sha256": result.context_similarity_table_sha256,
    }


def _assert_e2_d_evolution_parity(product: Any, deduplicated: Any) -> None:
    if _evolution_binding(product) != _evolution_binding(deduplicated):
        raise ExternalE2Error("E2-D and E2-E changed more than cross-chain rendering")


def _organization_observation_payload(
    context: E2Context,
    question: SelectedQuestion,
    *,
    dense_row: Mapping[str, Any],
    pool: tuple[ChainCandidate, ...],
    query_batch: Any,
    candidate_batch: Any,
    query_replay: Sequence[dict[str, Any]],
    chain_observations: Sequence[dict[str, Any]],
    entries: tuple[ContextCosine, ...],
) -> dict[str, Any]:
    return {
        "artifact_type": "swarmbrain-longmemeval-e2-qwen-similarity-observation",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_e1_dense_artifact_sha256": dense_row["artifact_sha256"],
        "source_e1a_trace_sha256": dense_row["e1a_trace"]["trace_sha256"],
        "candidate_pool_sha256": sha256_json(
            [
                {
                    "rank": rank,
                    "turn_id": _turn_id_payload(candidate.turn_id),
                    "query_cosine": candidate.query_cosine,
                    "document_sha256": candidate.turn.serialized_document_utf8.sha256,
                }
                for rank, candidate in enumerate(pool, start=1)
            ]
        ),
        "model": {
            "name": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "snapshot_artifact_sha256": _snapshot_artifact(
                context.e1,
                "qwen_embedding",
            ),
            "pooling": "official-last-valid-token-then-l2-normalize-float32",
            "padding_side": "right-MPS-stability-transfer-choice",
            "maximum_input_tokens": 8192,
        },
        "query_replay": {
            "query_input_sha256": sha256_bytes(qwen_query_text(question.question).encode("utf-8")),
            "absolute_tolerance": QUERY_REPLAY_ABSOLUTE_TOLERANCE,
            "maximum_absolute_delta": max(
                (float(row["absolute_delta"]) for row in query_replay),
                default=0.0,
            ),
            "observations": list(query_replay),
            "embedding": _embedded_batch_binding(
                query_batch,
                [qwen_query_text(question.question)],
            ),
        },
        "candidate_embeddings": {
            "turns": [
                {
                    "turn_id": _turn_id_payload(candidate.turn_id),
                    "document_sha256": candidate.turn.serialized_document_utf8.sha256,
                    **_embedded_batch_binding(
                        candidate_batch,
                        [item.turn.serialized_text for item in pool],
                    )["inputs"][position],
                }
                for position, candidate in enumerate(pool)
            ],
            "batch_plan": [list(group) for group in candidate_batch.batch_plan],
            "accounting": _embedded_batch_binding(
                candidate_batch,
                [item.turn.serialized_text for item in pool],
            )["accounting"],
        },
        "chain_prefix_embeddings": list(chain_observations),
        "context_cosines": [entry.content_free_binding() for entry in entries],
        "context_cosine_table_sha256": PrecomputedContextSimilarities(entries).table_sha256,
        "claims": {
            "gold_fields_consumed": False,
            "candidate_documents_embedded_once_per_question": True,
            "chain_prefix_embedded_once_per_unique_prefix": True,
            "embedding_execution_externally_attested_unsigned": True,
            "organizer_replays_without_model_execution": True,
        },
    }


def _compute_organization(
    context: E2Context,
    question: SelectedQuestion,
    *,
    embedder: QwenEmbedder,
) -> dict[str, Any]:
    started = perf_counter()
    dense_row, e1a = _dense_source(context, question)
    pool = _candidate_pool(question, e1a, dense_row)
    candidate_texts = [candidate.turn.serialized_text for candidate in pool]
    query_text = qwen_query_text(question.question)
    query_batch = embedder.embed([query_text])
    candidate_batch = embedder.embed(candidate_texts)
    replayed_cosines = candidate_batch.vectors @ query_batch.vectors[0]
    query_replay: list[dict[str, Any]] = []
    for candidate, replayed in zip(pool, replayed_cosines.tolist(), strict=True):
        value = max(-1.0, min(1.0, float(replayed)))
        delta = abs(value - candidate.query_cosine)
        query_replay.append(
            {
                "turn_id": _turn_id_payload(candidate.turn_id),
                "stored_e1_raw_cosine": candidate.query_cosine,
                "reexecuted_raw_cosine": value,
                "absolute_delta": delta,
            }
        )
    maximum_delta = max(float(row["absolute_delta"]) for row in query_replay)
    if maximum_delta > QUERY_REPLAY_ABSOLUTE_TOLERANCE:
        raise ExternalE2Error(
            "E2 Qwen execution does not reproduce E1 raw query cosines within "
            f"{QUERY_REPLAY_ABSOLUTE_TOLERANCE}: maximum delta={maximum_delta}"
        )

    vectors_by_document_sha256: dict[str, Any] = {}
    for candidate, vector in zip(pool, candidate_batch.vectors, strict=True):
        digest = sha256_bytes(candidate.turn.serialized_text.encode("utf-8"))
        prior = vectors_by_document_sha256.get(digest)
        if prior is not None:
            difference = float(embedder.torch.max(embedder.torch.abs(prior - vector)).item())
            if difference != 0.0:
                raise ExternalE2Error("identical candidate documents embedded differently")
        vectors_by_document_sha256[digest] = vector

    chain_vectors: dict[str, Any] = {}
    chain_observations_by_sha256: dict[str, dict[str, Any]] = {}

    def context_similarity(chain_text: str, candidate_text: str) -> float:
        chain_sha256 = sha256_bytes(chain_text.encode("utf-8"))
        vector = chain_vectors.get(chain_sha256)
        if vector is None:
            batch = embedder.embed([chain_text])
            vector = batch.vectors[0]
            chain_vectors[chain_sha256] = vector
            binding = _embedded_batch_binding(batch, [chain_text])
            chain_observations_by_sha256[chain_sha256] = {
                "chain_context_sha256": chain_sha256,
                "chain_context_utf8_bytes": len(chain_text.encode("utf-8")),
                "embedding": binding["inputs"][0],
                "batch_plan": binding["batch_plan"],
                "accounting": binding["accounting"],
            }
        candidate_sha256 = sha256_bytes(candidate_text.encode("utf-8"))
        candidate_vector = vectors_by_document_sha256.get(candidate_sha256)
        if candidate_vector is None:
            raise ExternalE2Error("organizer requested a candidate outside the fixed top-20")
        cosine = float((vector @ candidate_vector).item())
        return max(-1.0, min(1.0, cosine))

    provisional_identity = ExternalSimilarityEvidence(
        producer="scripts.run_longmemeval_e2_external.QwenEmbedder.provisional",
        model_id=QWEN_MODEL,
        model_revision=QWEN_REVISION,
        artifact_sha256=sha256_json(
            {
                "question_id": question.question_id,
                "purpose": "collect-exact-prefix-cosines-before-evidence-seal",
            }
        ),
    )
    provisional = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=provisional_identity,
        context_similarities=context_similarity,
    )
    entries = _context_entries_from_provisional(provisional, candidates=pool)
    prefix_ids_by_sha256: dict[str, tuple[LongMemEvalTurnId, ...]] = {}
    for entry in entries:
        prior = prefix_ids_by_sha256.setdefault(
            entry.chain_context_sha256,
            entry.chain_turn_ids,
        )
        if prior != entry.chain_turn_ids:
            raise ExternalE2Error("one chain-context digest bound two distinct turn prefixes")
    if set(prefix_ids_by_sha256) != set(chain_observations_by_sha256):
        raise ExternalE2Error("Qwen chain-prefix execution ledger is incomplete")
    chain_observations = []
    for digest, binding in chain_observations_by_sha256.items():
        chain_observations.append(
            {
                **binding,
                "chain_turn_ids": [
                    _turn_id_payload(turn_id) for turn_id in prefix_ids_by_sha256[digest]
                ],
            }
        )
    observation_payload = _organization_observation_payload(
        context,
        question,
        dense_row=dense_row,
        pool=pool,
        query_batch=query_batch,
        candidate_batch=candidate_batch,
        query_replay=query_replay,
        chain_observations=chain_observations,
        entries=entries,
    )
    observation = {
        **observation_payload,
        "artifact_sha256": sha256_json(observation_payload),
    }
    similarity_identity = ExternalSimilarityEvidence(
        producer="scripts.run_longmemeval_e2_external.QwenEmbedder",
        model_id=QWEN_MODEL,
        model_revision=QWEN_REVISION,
        artifact_sha256=observation["artifact_sha256"],
    )
    table = PrecomputedContextSimilarities(entries)
    retrieval = organize_e2(
        pool,
        cell=E2Cell.RETRIEVAL_ORDER,
        similarity_evidence=similarity_identity,
    )
    product = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=similarity_identity,
        context_similarities=table,
    )
    deduplicated = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT_DEDUP,
        similarity_evidence=similarity_identity,
        context_similarities=table,
    )
    _assert_e2_d_evolution_parity(product, deduplicated)
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e2-organization-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_e1_dense_artifact_sha256": dense_row["artifact_sha256"],
        "source_e1a_trace_sha256": e1a.trace_sha256,
        "similarity_identity": _similarity_identity_payload(similarity_identity),
        "similarity_observation": observation,
        "cells": {
            "E2-A": retrieval.content_free_artifact(),
            "E2-D": product.content_free_artifact(),
            "E2-E": deduplicated.content_free_artifact(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": embedder.torch.__version__,
            "transformers": embedder.transformers_version,
            "device": embedder.device,
            "dtype": embedder.dtype_name,
            "maximum_batch_size": embedder.batch_size,
        },
        "wall_ms": (perf_counter() - started) * 1000.0,
        "claims": {
            "gold_fields_consumed": False,
            "E2-D_and_E2-E_evolution_identical": True,
            "E2-D_duplicate_rendering_not_sent_to_reader": True,
            "E2-E_is_cross_chain_deduplicated_transfer_cell": True,
            "reader_or_judge_executed": False,
            "paper_parity": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(payload)


def _replay_similarity_observation(
    context: E2Context,
    question: SelectedQuestion,
    *,
    dense_row: Mapping[str, Any],
    pool: tuple[ChainCandidate, ...],
    value: Any,
) -> tuple[ExternalSimilarityEvidence, PrecomputedContextSimilarities]:
    if not isinstance(value, dict):
        raise ExternalE2Error("similarity observation must be an object")
    digest = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or sha256_json(payload) != digest:
        raise ExternalE2Error("similarity observation digest does not match")
    expected = {
        "artifact_type": "swarmbrain-longmemeval-e2-qwen-similarity-observation",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_e1_dense_artifact_sha256": dense_row["artifact_sha256"],
        "source_e1a_trace_sha256": dense_row["e1a_trace"]["trace_sha256"],
        "candidate_pool_sha256": sha256_json(
            [
                {
                    "rank": rank,
                    "turn_id": _turn_id_payload(candidate.turn_id),
                    "query_cosine": candidate.query_cosine,
                    "document_sha256": candidate.turn.serialized_document_utf8.sha256,
                }
                for rank, candidate in enumerate(pool, start=1)
            ]
        ),
    }
    for key, expected_value in expected.items():
        if type(value.get(key)) is not type(expected_value) or value.get(key) != expected_value:
            raise ExternalE2Error(f"similarity observation {key} drifted")
    expected_model = {
        "name": QWEN_MODEL,
        "revision": QWEN_REVISION,
        "snapshot_artifact_sha256": _snapshot_artifact(context.e1, "qwen_embedding"),
        "pooling": "official-last-valid-token-then-l2-normalize-float32",
        "padding_side": "right-MPS-stability-transfer-choice",
        "maximum_input_tokens": 8192,
    }
    if value.get("model") != expected_model:
        raise ExternalE2Error("similarity observation model binding drifted")

    raw_by_id = {
        _turn_id_from_payload(row["turn_id"]): float(row["raw_cosine"])
        for row in dense_row["dense"]["observations"]
    }
    replay = value.get("query_replay")
    if not isinstance(replay, dict):
        raise ExternalE2Error("similarity observation is missing query replay")
    if replay.get("query_input_sha256") != sha256_bytes(
        qwen_query_text(question.question).encode("utf-8")
    ):
        raise ExternalE2Error("query replay input digest drifted")
    if replay.get("absolute_tolerance") != QUERY_REPLAY_ABSOLUTE_TOLERANCE:
        raise ExternalE2Error("query replay tolerance drifted")
    rows = replay.get("observations")
    if not isinstance(rows, list) or len(rows) != K:
        raise ExternalE2Error("query replay must cover the fixed top-20")
    observed_maximum = 0.0
    for candidate, row in zip(pool, rows, strict=True):
        if not isinstance(row, dict) or row.get("turn_id") != _turn_id_payload(candidate.turn_id):
            raise ExternalE2Error("query replay candidate order drifted")
        stored = row.get("stored_e1_raw_cosine")
        reexecuted = row.get("reexecuted_raw_cosine")
        delta = row.get("absolute_delta")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in (stored, reexecuted, delta)
        ):
            raise ExternalE2Error("query replay contains a non-finite score")
        if float(stored) != raw_by_id[candidate.turn_id]:
            raise ExternalE2Error("query replay stored cosine differs from E1 evidence")
        expected_delta = abs(float(reexecuted) - float(stored))
        if float(delta) != expected_delta:
            raise ExternalE2Error("query replay absolute delta is inconsistent")
        if not -1.0 <= float(reexecuted) <= 1.0:
            raise ExternalE2Error("query replay cosine is outside [-1, 1]")
        observed_maximum = max(observed_maximum, expected_delta)
    if replay.get("maximum_absolute_delta") != observed_maximum:
        raise ExternalE2Error("query replay maximum delta is inconsistent")
    if observed_maximum > QUERY_REPLAY_ABSOLUTE_TOLERANCE:
        raise ExternalE2Error("query replay exceeds its frozen absolute tolerance")

    candidates = value.get("candidate_embeddings")
    if not isinstance(candidates, dict) or not isinstance(candidates.get("turns"), list):
        raise ExternalE2Error("candidate embedding ledger is missing")
    if len(candidates["turns"]) != K:
        raise ExternalE2Error("candidate embedding ledger must cover the top-20")
    for position, (candidate, row) in enumerate(zip(pool, candidates["turns"], strict=True)):
        if not isinstance(row, dict):
            raise ExternalE2Error("candidate embedding row must be an object")
        if row.get("position") != position:
            raise ExternalE2Error("candidate embedding positions drifted")
        if row.get("turn_id") != _turn_id_payload(candidate.turn_id):
            raise ExternalE2Error("candidate embedding turn binding drifted")
        document_sha256 = candidate.turn.serialized_document_utf8.sha256
        if row.get("document_sha256") != document_sha256:
            raise ExternalE2Error("candidate embedding document binding drifted")
        if row.get("input_sha256") != document_sha256:
            raise ExternalE2Error("candidate embedding input digest drifted")

    raw_entries = value.get("context_cosines")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ExternalE2Error("similarity observation has no context cosines")
    entries = tuple(_context_entry_from_payload(item) for item in raw_entries)
    if [entry.content_free_binding() for entry in entries] != raw_entries:
        raise ExternalE2Error("context cosine rows are not canonical")
    table = PrecomputedContextSimilarities(entries)
    if value.get("context_cosine_table_sha256") != table.table_sha256:
        raise ExternalE2Error("context cosine table digest drifted")
    prefix_rows = value.get("chain_prefix_embeddings")
    if not isinstance(prefix_rows, list) or not prefix_rows:
        raise ExternalE2Error("chain-prefix embedding ledger is missing")
    prefixes: dict[str, tuple[LongMemEvalTurnId, ...]] = {}
    for row in prefix_rows:
        if not isinstance(row, dict) or not isinstance(row.get("chain_turn_ids"), list):
            raise ExternalE2Error("chain-prefix embedding row is invalid")
        turn_ids = tuple(_turn_id_from_payload(item) for item in row["chain_turn_ids"])
        turns_by_id = {candidate.turn_id: candidate.turn for candidate in pool}
        try:
            turns = tuple(turns_by_id[turn_id] for turn_id in turn_ids)
        except KeyError as exc:
            raise ExternalE2Error("chain-prefix ledger references a turn outside top-20") from exc
        chain_sha256 = chain_context_sha256(turns)
        if row.get("chain_context_sha256") != chain_sha256:
            raise ExternalE2Error("chain-prefix ledger digest differs from exact text")
        if row.get("chain_context_utf8_bytes") != len(
            "\n\n".join(turn.serialized_text for turn in turns).encode("utf-8")
        ):
            raise ExternalE2Error("chain-prefix ledger byte count drifted")
        if chain_sha256 in prefixes:
            raise ExternalE2Error("chain-prefix ledger repeats an exact prefix")
        prefixes[chain_sha256] = turn_ids
    entry_prefixes = {entry.chain_context_sha256: entry.chain_turn_ids for entry in entries}
    if prefixes != entry_prefixes:
        raise ExternalE2Error("chain-prefix ledger does not exactly cover context cosine prefixes")
    claims = value.get("claims")
    expected_claims = {
        "gold_fields_consumed": False,
        "candidate_documents_embedded_once_per_question": True,
        "chain_prefix_embedded_once_per_unique_prefix": True,
        "embedding_execution_externally_attested_unsigned": True,
        "organizer_replays_without_model_execution": True,
    }
    if claims != expected_claims:
        raise ExternalE2Error("similarity observation claims drifted")
    return (
        ExternalSimilarityEvidence(
            producer="scripts.run_longmemeval_e2_external.QwenEmbedder",
            model_id=QWEN_MODEL,
            model_revision=QWEN_REVISION,
            artifact_sha256=str(digest),
        ),
        table,
    )


def replay_organization_row(
    context: E2Context,
    question: SelectedQuestion,
    row: Any,
) -> tuple[tuple[ChainCandidate, ...], dict[str, Any]]:
    row = validate_sealed_artifact(row)
    dense_row, e1a = _dense_source(context, question)
    pool = _candidate_pool(question, e1a, dense_row)
    expected = {
        "artifact_type": "swarmbrain-longmemeval-e2-organization-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_e1_dense_artifact_sha256": dense_row["artifact_sha256"],
        "source_e1a_trace_sha256": e1a.trace_sha256,
    }
    for key, expected_value in expected.items():
        if type(row.get(key)) is not type(expected_value) or row.get(key) != expected_value:
            raise ExternalE2Error(f"organization row {key} drifted")
    identity, table = _replay_similarity_observation(
        context,
        question,
        dense_row=dense_row,
        pool=pool,
        value=row.get("similarity_observation"),
    )
    if row.get("similarity_identity") != _similarity_identity_payload(identity):
        raise ExternalE2Error("organization similarity identity drifted")
    results = {
        "E2-A": organize_e2(
            pool,
            cell=E2Cell.RETRIEVAL_ORDER,
            similarity_evidence=identity,
        ),
        "E2-D": organize_e2(
            pool,
            cell=E2Cell.PRODUCT_APT,
            similarity_evidence=identity,
            context_similarities=table,
        ),
        "E2-E": organize_e2(
            pool,
            cell=E2Cell.PRODUCT_APT_DEDUP,
            similarity_evidence=identity,
            context_similarities=table,
        ),
    }
    _assert_e2_d_evolution_parity(results["E2-D"], results["E2-E"])
    expected_cells = {cell: result.content_free_artifact() for cell, result in results.items()}
    if row.get("cells") != expected_cells:
        raise ExternalE2Error("saved organization cells differ from deterministic replay")
    expected_claims = {
        "gold_fields_consumed": False,
        "E2-D_and_E2-E_evolution_identical": True,
        "E2-D_duplicate_rendering_not_sent_to_reader": True,
        "E2-E_is_cross_chain_deduplicated_transfer_cell": True,
        "reader_or_judge_executed": False,
        "paper_parity": False,
        "production_policy_changed": False,
    }
    if row.get("claims") != expected_claims:
        raise ExternalE2Error("organization claims drifted")
    return pool, results


def _selected_prefix(context: E2Context, limit: int | None) -> tuple[SelectedQuestion, ...]:
    if limit is None:
        return context.e1.selected
    if isinstance(limit, bool) or limit < 1:
        raise ExternalE2Error("execution limit must be a positive integer")
    return context.e1.selected[:limit]


def run_organize_phase(
    context: E2Context,
    *,
    device: str,
    batch_size: int,
    limit: int | None = None,
) -> None:
    selected = _selected_prefix(context, limit)
    pending: list[SelectedQuestion] = []
    for question in selected:
        path = e2_phase_path(context, "organization", question)
        if path.exists():
            replay_organization_row(context, question, load_json(path, sealed=True))
            print(f"  organize: verified {question.question_id}", file=sys.stderr, flush=True)
        else:
            pending.append(question)
    if not pending:
        print("  organize: all requested artifacts verified", file=sys.stderr, flush=True)
        return
    embedder = QwenEmbedder(context.e1.qwen_root, device=device, batch_size=batch_size)
    try:
        for completed, question in enumerate(pending, start=1):
            row = _compute_organization(context, question, embedder=embedder)
            replay_organization_row(context, question, row)
            write_json(e2_phase_path(context, "organization", question), row)
            d = row["cells"]["E2-D"]
            print(
                f"  organize: {completed}/{len(pending)} {question.question_id} "
                f"chains={d['rendered_turn_count']} calls={d['context_similarity']['calls']} "
                f"wall={row['wall_ms'] / 1000:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    finally:
        embedder.close()


def _compute_packs(
    context: E2Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    organization_path = e2_phase_path(context, "organization", question)
    if not organization_path.exists():
        raise ExternalE2Error("packing requires an E2 organization artifact first")
    organization_row = load_json(organization_path, sealed=True)
    pool, results = replay_organization_row(context, question, organization_row)
    ordered_by_cell = {
        "E2-A": OrderedTurnBlocks.linear(tuple(candidate.turn for candidate in pool)),
        "E2-E": OrderedTurnBlocks.chain_blocks(results["E2-E"].rendered_chains()),
    }
    packed: dict[str, Any] = {}
    prompt_arms: dict[str, Any] = {}
    for cell, ordered in ordered_by_cell.items():
        tokenizer.reset_receipts()
        result = pack_turn_prompt(
            ordered,
            question_id=question.question_id,
            question=question.question,
            current_date=question.current_date,
            token_budget=TOKEN_BUDGET,
            tokenizer=tokenizer,
        )
        artifact = result.content_free_artifact()
        packed[cell] = artifact
        encoded = result.prompt.encode("utf-8")
        prompt_arms[cell] = {
            "prompt": result.prompt,
            "prompt_sha256": sha256_bytes(encoded),
            "prompt_utf8_bytes": len(encoded),
            "prompt_tokens": artifact["final_prompt"]["tokens"],
            "packing_trace_sha256": artifact["trace_sha256"],
        }
    prompt_payload = {
        "artifact_type": "swarmbrain-longmemeval-e2-packed-prompts-sensitive",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "classification": "contains-public-benchmark-question-and-turn-text",
        "arms": prompt_arms,
    }
    prompt_artifact = seal_artifact(prompt_payload)
    pack_payload = {
        "artifact_type": "swarmbrain-longmemeval-e2-pack-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_organization_artifact_sha256": organization_row["artifact_sha256"],
        "source_e2a_trace_sha256": results["E2-A"].trace_sha256,
        "source_e2d_trace_sha256": results["E2-D"].trace_sha256,
        "source_e2e_trace_sha256": results["E2-E"].trace_sha256,
        "prompt_artifact_sha256": prompt_artifact["artifact_sha256"],
        "tokenizer": tokenizer.identity.as_dict(),
        "runtime": {
            "python": sys.version.split()[0],
            "transformers": tokenizer.transformers_version,
        },
        "arms": packed,
        "claims": {
            "complete_reader_prompt_counted": True,
            "whole_turns_only": True,
            "E2_D_not_packed_due_to_duplicate_ID_contract": True,
            "E2_E_cross_chain_deduplicated_before_packing": True,
            "gold_fields_consumed": False,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(pack_payload), prompt_artifact


def replay_pack_row(
    context: E2Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
    pack_row: Any,
    prompt_row: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pack_row = validate_sealed_artifact(pack_row)
    prompt_row = validate_sealed_artifact(prompt_row)
    expected_pack, expected_prompt = _compute_packs(
        context,
        question,
        tokenizer=tokenizer,
    )
    if pack_row != expected_pack:
        raise ExternalE2Error("saved E2 packing artifact differs from exact replay")
    if prompt_row != expected_prompt:
        raise ExternalE2Error("saved E2 plaintext prompt artifact differs from exact replay")
    return pack_row, prompt_row


def run_pack_phase(context: E2Context, *, limit: int | None = None) -> None:
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    selected = _selected_prefix(context, limit)
    for completed, question in enumerate(selected, start=1):
        pack_path = e2_phase_path(context, "pack", question)
        prompt_path = e2_phase_path(context, "prompts", question)
        if pack_path.exists() != prompt_path.exists():
            raise ExternalE2Error("E2 packing has only one side of its bound artifact pair")
        started = perf_counter()
        if pack_path.exists():
            replay_pack_row(
                context,
                question,
                tokenizer=tokenizer,
                pack_row=load_json(pack_path, sealed=True),
                prompt_row=load_json(prompt_path, sealed=True),
            )
            action = "verified"
        else:
            pack_row, prompt_row = _compute_packs(
                context,
                question,
                tokenizer=tokenizer,
            )
            write_json(prompt_path, prompt_row)
            write_json(pack_path, pack_row)
            action = "packed"
        print(
            f"  pack: {completed}/{len(selected)} {action} {question.question_id} "
            f"wall={(perf_counter() - started):.1f}s",
            file=sys.stderr,
            flush=True,
        )


def _qa_arm_order(question: SelectedQuestion) -> tuple[str, str]:
    return ("E2-A", "E2-E") if question.position % 2 == 0 else ("E2-E", "E2-A")


def _qa_receipt_id(question_id: str, cell: str) -> str:
    return f"{question_id}:{cell.casefold()}"


def _normalized_qa_arm(
    question: SelectedQuestion,
    cell: str,
    *,
    prompt_row: Mapping[str, Any],
    tokenizer: DeepSeekExactTokenizer,
    reader: ChatResult,
    judge: ChatResult,
    reader_receipt_sha256: str,
    judge_receipt_sha256: str,
) -> dict[str, Any]:
    prompt = str(prompt_row["arms"][cell]["prompt"])
    exact_reader_tokens = tokenizer.exact_count(prompt)
    if exact_reader_tokens != prompt_row["arms"][cell]["prompt_tokens"]:
        raise ExternalE2Error("E2 packed prompt token count differs from independent replay")
    verify_provider_prompt_tokens(
        reader,
        expected=exact_reader_tokens,
        label=f"{question.question_id}/{cell}/reader",
    )
    judge_input = _judge_text_for_arm(question, reader.content)
    exact_judge_tokens = tokenizer.exact_count(judge_input)
    verify_provider_prompt_tokens(
        judge,
        expected=exact_judge_tokens,
        label=f"{question.question_id}/{cell}/judge",
    )
    return {
        "hypothesis": reader.content,
        "development_judge_text": judge.content,
        "development_label": judge_label(judge.content),
        "reader": {
            "exact_local_prompt_tokens": exact_reader_tokens,
            "api_prompt_tokens": reader.prompt_tokens,
            "completion_tokens": reader.completion_tokens,
            "total_tokens": reader.total_tokens,
            "finish_reason": reader.finish_reason,
            "latency_ms": reader.latency_ms,
            "attempts": reader.attempts,
            "provider_request_id": reader.request_id,
            "raw_request_sha256": reader.raw_request_sha256,
            "raw_response_sha256": reader.raw_response_sha256,
            "receipt_sha256": reader_receipt_sha256,
        },
        "development_judge": {
            "exact_local_prompt_tokens": exact_judge_tokens,
            "api_prompt_tokens": judge.prompt_tokens,
            "completion_tokens": judge.completion_tokens,
            "total_tokens": judge.total_tokens,
            "finish_reason": judge.finish_reason,
            "latency_ms": judge.latency_ms,
            "attempts": judge.attempts,
            "provider_request_id": judge.request_id,
            "raw_request_sha256": judge.raw_request_sha256,
            "raw_response_sha256": judge.raw_response_sha256,
            "receipt_sha256": judge_receipt_sha256,
        },
        "estimated_cost_upper_bound_usd": _upper_bound_cost(reader) + _upper_bound_cost(judge),
    }


def replay_qa_row(
    context: E2Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
    qa_row: Any,
    receipt_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    qa_row = validate_sealed_artifact(qa_row)
    pack_path = e2_phase_path(context, "pack", question)
    prompt_path = e2_phase_path(context, "prompts", question)
    pack_row, prompt_row = replay_pack_row(
        context,
        question,
        tokenizer=tokenizer,
        pack_row=load_json(pack_path, sealed=True),
        prompt_row=load_json(prompt_path, sealed=True),
    )
    raw_receipts = _receipt_bytes(receipt_records)
    expected = {
        "artifact_type": "swarmbrain-longmemeval-e2-deepseek-qa-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_pack_artifact_sha256": pack_row["artifact_sha256"],
        "source_prompt_artifact_sha256": prompt_row["artifact_sha256"],
        "receipt_artifact_sha256": sha256_bytes(raw_receipts),
        "receipt_artifact_bytes": len(raw_receipts),
        "arm_order": list(_qa_arm_order(question)),
    }
    for key, expected_value in expected.items():
        if type(qa_row.get(key)) is not type(expected_value) or qa_row.get(key) != expected_value:
            raise ExternalE2Error(f"E2 QA row {key} differs from frozen evidence")
    arms = qa_row.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"E2-A", "E2-E"}:
        raise ExternalE2Error("E2 QA row must contain exactly A and E")
    if len(receipt_records) != 4:
        raise ExternalE2Error("paired E2 QA requires two reader and two judge receipts")
    replayed: dict[tuple[str, str], ChatResult] = {}
    receipt_sha256: dict[tuple[str, str], str] = {}
    for receipt in receipt_records:
        result = validate_chat_receipt_record(receipt)
        key = (str(receipt["question_id"]), str(receipt["call_role"]))
        if key in replayed:
            raise ExternalE2Error("E2 receipt artifact repeats an arm/role pair")
        replayed[key] = result
        receipt_sha256[key] = sha256_json(receipt)
    total_cost = 0.0
    for cell in ("E2-A", "E2-E"):
        receipt_id = _qa_receipt_id(question.question_id, cell)
        reader_key = (receipt_id, "reader")
        judge_key = (receipt_id, "development_judge")
        if reader_key not in replayed or judge_key not in replayed:
            raise ExternalE2Error("E2 receipts do not cover every arm and role")
        expected_arm = _normalized_qa_arm(
            question,
            cell,
            prompt_row=prompt_row,
            tokenizer=tokenizer,
            reader=replayed[reader_key],
            judge=replayed[judge_key],
            reader_receipt_sha256=receipt_sha256[reader_key],
            judge_receipt_sha256=receipt_sha256[judge_key],
        )
        if arms[cell] != expected_arm:
            raise ExternalE2Error("E2 normalized QA arm differs from raw receipt replay")
        total_cost += float(expected_arm["estimated_cost_upper_bound_usd"])
    if qa_row.get("estimated_cost_upper_bound_usd") != total_cost:
        raise ExternalE2Error("E2 QA cost ledger differs from receipt-derived usage")
    expected_claims = {
        "reader_model": DEEPSEEK_MODEL,
        "development_judge_model": DEEPSEEK_MODEL,
        "thinking_disabled": True,
        "api_and_local_prompt_tokens_reconciled": True,
        "E2_D_reader_executed": False,
        "official_gpt4o_judge_executed": False,
        "official_longmemeval_score": False,
    }
    if qa_row.get("claims") != expected_claims:
        raise ExternalE2Error("E2 QA claims drifted")
    return qa_row


async def _run_qa_calls(
    context: E2Context,
    *,
    tokenizer: DeepSeekExactTokenizer,
    base_url: str,
    api_key: str,
    limit: int | None,
) -> None:
    reader_client = ChatClient(
        base_url=base_url,
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=4096,
        required_response_model=DEEPSEEK_MODEL,
        require_request_id=True,
        thinking_mode="disabled",
    )
    judge_client = ChatClient(
        base_url=base_url,
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=64,
        required_response_model=DEEPSEEK_MODEL,
        require_request_id=True,
        thinking_mode="disabled",
    )
    selected = _selected_prefix(context, limit)
    try:
        for completed, question in enumerate(selected, start=1):
            qa_path = e2_phase_path(context, "qa", question)
            receipts_path = e2_receipt_path(context, question)
            if qa_path.exists() != receipts_path.exists():
                raise ExternalE2Error("E2 QA has only one side of its receipt/result pair")
            if qa_path.exists():
                replay_qa_row(
                    context,
                    question,
                    tokenizer=tokenizer,
                    qa_row=load_json(qa_path, sealed=True),
                    receipt_records=_load_receipts(receipts_path),
                )
                print(
                    f"  qa: {completed}/{len(selected)} verified {question.question_id}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            pack_path = e2_phase_path(context, "pack", question)
            prompt_path = e2_phase_path(context, "prompts", question)
            if not pack_path.exists() or not prompt_path.exists():
                raise ExternalE2Error("E2 QA requires packed prompts first")
            pack_row, prompt_row = replay_pack_row(
                context,
                question,
                tokenizer=tokenizer,
                pack_row=load_json(pack_path, sealed=True),
                prompt_row=load_json(prompt_path, sealed=True),
            )
            started = perf_counter()
            receipts: list[dict[str, Any]] = []
            arms: dict[str, Any] = {}
            for cell in _qa_arm_order(question):
                prompt = str(prompt_row["arms"][cell]["prompt"])
                exact_reader_tokens = tokenizer.exact_count(prompt)
                reader = await reader_client.complete(prompt)
                verify_provider_prompt_tokens(
                    reader,
                    expected=exact_reader_tokens,
                    label=f"{question.question_id}/{cell}/reader",
                )
                judge_input = _judge_text_for_arm(question, reader.content)
                exact_judge_tokens = tokenizer.exact_count(judge_input)
                judge = await judge_client.complete(judge_input)
                verify_provider_prompt_tokens(
                    judge,
                    expected=exact_judge_tokens,
                    label=f"{question.question_id}/{cell}/judge",
                )
                receipt_id = _qa_receipt_id(question.question_id, cell)
                reader_receipt = chat_receipt_record(receipt_id, "reader", reader)
                judge_receipt = chat_receipt_record(
                    receipt_id,
                    "development_judge",
                    judge,
                )
                validate_chat_receipt_record(reader_receipt)
                validate_chat_receipt_record(judge_receipt)
                receipts.extend((reader_receipt, judge_receipt))
                arms[cell] = _normalized_qa_arm(
                    question,
                    cell,
                    prompt_row=prompt_row,
                    tokenizer=tokenizer,
                    reader=reader,
                    judge=judge,
                    reader_receipt_sha256=sha256_json(reader_receipt),
                    judge_receipt_sha256=sha256_json(judge_receipt),
                )
            raw_receipts = _receipt_bytes(receipts)
            payload = {
                "artifact_type": "swarmbrain-longmemeval-e2-deepseek-qa-question",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "protocol_version": E2_RUN_PROTOCOL_VERSION,
                "run_manifest_sha256": context.manifest["artifact_sha256"],
                "question_id": question.question_id,
                "question_position": question.position,
                "source_pack_artifact_sha256": pack_row["artifact_sha256"],
                "source_prompt_artifact_sha256": prompt_row["artifact_sha256"],
                "receipt_artifact_sha256": sha256_bytes(raw_receipts),
                "receipt_artifact_bytes": len(raw_receipts),
                "arm_order": list(_qa_arm_order(question)),
                "arms": arms,
                "estimated_cost_upper_bound_usd": sum(
                    float(arm["estimated_cost_upper_bound_usd"]) for arm in arms.values()
                ),
                "wall_ms": (perf_counter() - started) * 1000.0,
                "cost_basis": {
                    "input": "all prompt tokens pessimistically priced as cache misses",
                    "cache_miss_input_usd_per_million": (DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION),
                    "output_usd_per_million": DEEPSEEK_OUTPUT_USD_PER_MILLION,
                    "attempt_multiplier": True,
                },
                "claims": {
                    "reader_model": DEEPSEEK_MODEL,
                    "development_judge_model": DEEPSEEK_MODEL,
                    "thinking_disabled": True,
                    "api_and_local_prompt_tokens_reconciled": True,
                    "E2_D_reader_executed": False,
                    "official_gpt4o_judge_executed": False,
                    "official_longmemeval_score": False,
                },
            }
            sealed = seal_artifact(payload)
            atomic_write(receipts_path, raw_receipts)
            replay_qa_row(
                context,
                question,
                tokenizer=tokenizer,
                qa_row=sealed,
                receipt_records=tuple(receipts),
            )
            write_json(qa_path, sealed)
            print(
                f"  qa: {completed}/{len(selected)} ran {question.question_id} "
                f"A={arms['E2-A']['development_label']} "
                f"E={arms['E2-E']['development_label']} "
                f"cost<=${payload['estimated_cost_upper_bound_usd']:.5f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        await reader_client.aclose()
        await judge_client.aclose()


def run_qa_phase(
    context: E2Context,
    *,
    base_url: str,
    api_key_env: str,
    limit: int | None = None,
) -> None:
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise ExternalE2Error("development QA is frozen to the official DeepSeek API endpoint")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ExternalE2Error(f"environment variable {api_key_env!r} is missing")
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    asyncio.run(
        _run_qa_calls(
            context,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key=api_key,
            limit=limit,
        )
    )


def _organization_metrics(result: Any) -> dict[str, Any]:
    rendered = result.rendered_candidates()
    unique_ids = {turn.turn_id for turn in rendered}
    return {
        "chain_count": len(result.chains),
        "chain_lengths": [len(chain.turns) for chain in result.chains],
        "decisions": len(result.decisions),
        "appended_decisions": sum(decision.appended for decision in result.decisions),
        "stopped_decisions": sum(not decision.appended for decision in result.decisions),
        "context_similarity_calls": result.context_similarity_calls,
        "rendered_turns": len(rendered),
        "unique_rendered_turns": len(unique_ids),
        "repeated_rendered_turns": len(rendered) - len(unique_ids),
    }


def build_report(context: E2Context) -> dict[str, Any]:
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    cases: list[dict[str, Any]] = []
    source_artifacts: dict[str, list[str]] = {
        "organization": [],
        "pack": [],
        "prompts": [],
        "qa": [],
        "receipts": [],
    }
    for question in context.e1.selected:
        organization = load_json(
            e2_phase_path(context, "organization", question),
            sealed=True,
        )
        pool, results = replay_organization_row(context, question, organization)
        pack = load_json(e2_phase_path(context, "pack", question), sealed=True)
        prompts = load_json(e2_phase_path(context, "prompts", question), sealed=True)
        qa = load_json(e2_phase_path(context, "qa", question), sealed=True)
        receipts = _load_receipts(e2_receipt_path(context, question))
        replay_qa_row(
            context,
            question,
            tokenizer=tokenizer,
            qa_row=qa,
            receipt_records=receipts,
        )
        source_artifacts["organization"].append(organization["artifact_sha256"])
        source_artifacts["pack"].append(pack["artifact_sha256"])
        source_artifacts["prompts"].append(prompts["artifact_sha256"])
        source_artifacts["qa"].append(qa["artifact_sha256"])
        source_artifacts["receipts"].append(sha256_bytes(_receipt_bytes(receipts)))
        arm_metrics: dict[str, Any] = {}
        ranked_turns = {
            "E2-A": tuple(candidate.turn for candidate in pool),
            "E2-E": results["E2-E"].rendered_candidates(),
        }
        for cell in ("E2-A", "E2-E"):
            pack_trace = pack["arms"][cell]
            ranked_ids = [_turn_id_payload(turn.turn_id) for turn in ranked_turns[cell]]
            arm_metrics[cell] = {
                "development_label": qa["arms"][cell]["development_label"],
                "prompt_tokens": pack_trace["final_prompt"]["tokens"],
                "candidate_turns": len(pack_trace["candidate_order"]),
                "kept_turns": len(pack_trace["kept_ids"]),
                "reader_latency_ms": qa["arms"][cell]["reader"]["latency_ms"],
                "judge_latency_ms": qa["arms"][cell]["development_judge"]["latency_ms"],
                "estimated_cost_upper_bound_usd": qa["arms"][cell][
                    "estimated_cost_upper_bound_usd"
                ],
                "retrieval": _retrieval_metrics(
                    question,
                    ranked_ids=ranked_ids,
                    kept_ids=pack_trace["kept_ids"],
                ),
            }
        cases.append(
            {
                "question_id": question.question_id,
                "question_position": question.position,
                "question_type": str(question.record["question_type"]),
                "abstention": is_abstention_question(question.question_id),
                "arm_order": qa["arm_order"],
                "organization": {cell: _organization_metrics(results[cell]) for cell in E2_CELLS},
                "query_replay_maximum_absolute_delta": organization["similarity_observation"][
                    "query_replay"
                ]["maximum_absolute_delta"],
                "arms": arm_metrics,
                "paired_delta": int(arm_metrics["E2-E"]["development_label"])
                - int(arm_metrics["E2-A"]["development_label"]),
            }
        )
    count = len(cases)
    summaries: dict[str, Any] = {}
    for cell in ("E2-A", "E2-E"):
        labels = [bool(case["arms"][cell]["development_label"]) for case in cases]
        successes = sum(labels)
        summaries[cell] = {
            "development_accuracy": successes / count,
            "development_accuracy_wilson_95": list(_wilson_interval(successes, count)),
            "mean_prompt_tokens": _mean(case["arms"][cell]["prompt_tokens"] for case in cases),
            "mean_candidate_turns": _mean(case["arms"][cell]["candidate_turns"] for case in cases),
            "mean_kept_turns": _mean(case["arms"][cell]["kept_turns"] for case in cases),
            "mean_reader_latency_ms": _mean(
                case["arms"][cell]["reader_latency_ms"] for case in cases
            ),
            "mean_estimated_cost_upper_bound_usd": _mean(
                case["arms"][cell]["estimated_cost_upper_bound_usd"] for case in cases
            ),
            "any_gold_session_in_prompt": _mean(
                float(case["arms"][cell]["retrieval"]["any_gold_session_in_prompt"])
                for case in cases
            ),
            "all_gold_sessions_in_prompt": _mean(
                float(case["arms"][cell]["retrieval"]["all_gold_sessions_in_prompt"])
                for case in cases
            ),
            "mean_answer_session_recall": _mean(
                case["arms"][cell]["retrieval"]["answer_session_recall"] for case in cases
            ),
            "mean_candidate_mrr": _mean(
                case["arms"][cell]["retrieval"]["candidate_mrr"] for case in cases
            ),
        }
    organization_summary: dict[str, Any] = {}
    for cell in E2_CELLS:
        organization_summary[cell] = {
            "mean_chain_count": _mean(case["organization"][cell]["chain_count"] for case in cases),
            "mean_context_similarity_calls": _mean(
                case["organization"][cell]["context_similarity_calls"] for case in cases
            ),
            "mean_rendered_turns": _mean(
                case["organization"][cell]["rendered_turns"] for case in cases
            ),
            "mean_unique_rendered_turns": _mean(
                case["organization"][cell]["unique_rendered_turns"] for case in cases
            ),
            "mean_repeated_rendered_turns": _mean(
                case["organization"][cell]["repeated_rendered_turns"] for case in cases
            ),
        }
    differences = [int(case["paired_delta"]) for case in cases]
    bootstrap = _paired_bootstrap_interval(differences)
    improved = sum(value > 0 for value in differences)
    regressed = sum(value < 0 for value in differences)
    tied = count - improved - regressed
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e2-development-pilot-report",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E2_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "source_e1_manifest_sha256": context.e1.manifest["artifact_sha256"],
        "classification": "exploratory-development-sample-not-official-longmemeval-score",
        "question_count": count,
        "reader": DEEPSEEK_MODEL,
        "judge": {
            "model": DEEPSEEK_MODEL,
            "role": "development-judge",
            "official_gpt4o_executed": False,
        },
        "organization": organization_summary,
        "reader_arms": summaries,
        "paired": {
            "contrast": "E2-E-minus-E2-A",
            "mean_accuracy_delta": _mean(differences),
            "bootstrap_95": list(bootstrap),
            "bootstrap_samples": 50_000,
            "bootstrap_seed": 20260809,
            "improved": improved,
            "regressed": regressed,
            "tied": tied,
        },
        "maximum_query_replay_absolute_delta": max(
            case["query_replay_maximum_absolute_delta"] for case in cases
        ),
        "total_estimated_cost_upper_bound_usd": sum(
            case["arms"][cell]["estimated_cost_upper_bound_usd"]
            for case in cases
            for cell in ("E2-A", "E2-E")
        ),
        "source_artifact_sets": {
            key: {"count": len(values), "ordered_sha256": sha256_json(values)}
            for key, values in source_artifacts.items()
        },
        "cases": cases,
        "decision": {
            "e2e_promising_on_development_sample": (
                summaries["E2-E"]["development_accuracy"]
                > summaries["E2-A"]["development_accuracy"]
                and summaries["E2-E"]["mean_answer_session_recall"]
                >= summaries["E2-A"]["mean_answer_session_recall"]
            ),
            "positive_paired_lower_confidence_bound": bootstrap[0] > 0.0,
            "eligible_for_composition_or_production_promotion": False,
            "required_next_if_promising": (
                "scale the frozen E2-E paired cell before composition or production changes"
            ),
        },
        "claims": {
            "gold_used_only_for_posthoc_report_metrics_and_development_judging": True,
            "E2_D_organization_executed_but_not_sent_to_reader": True,
            "E2_E_is_not_paper_parity": True,
            "official_longmemeval_score": False,
            "sealed_heldout_confirmation": False,
            "causal_production_improvement_proven": False,
            "production_policy_changed": False,
        },
    }
    report = seal_artifact(payload)
    write_json(context.output_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("organize", "pack", "qa", "report", "all"),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1-output-dir", type=Path, default=DEFAULT_E1_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_E2_OUTPUT)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--cross-encoder-root", type=Path, default=DEFAULT_CE_ROOT)
    parser.add_argument("--deepseek-root", type=Path, default=DEFAULT_DEEPSEEK_ROOT)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--qwen-batch-size", type=int, default=QWEN_BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="execute only the selected prefix; procedural smoke control, not in the manifest",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.qwen_batch_size < 1:
        raise SystemExit("batch size must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("execution limit must be positive")
    if args.limit is not None and args.phase in {"report", "all"}:
        raise SystemExit("report/all requires the complete frozen sample; omit --limit")
    context = build_e2_context(args)
    phases = ("organize", "pack", "qa", "report") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "organize":
            run_organize_phase(
                context,
                device=args.device,
                batch_size=args.qwen_batch_size,
                limit=args.limit,
            )
        elif phase == "pack":
            run_pack_phase(context, limit=args.limit)
        elif phase == "qa":
            run_qa_phase(
                context,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                limit=args.limit,
            )
        else:
            report = build_report(context)
            print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
