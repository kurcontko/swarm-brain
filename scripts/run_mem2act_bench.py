#!/usr/bin/env python3
"""Run the pinned three-arm Mem2ActBench external evaluation.

The harness owns data loading, leakage fences, parsing, scoring, paired
bootstrap confidence intervals, and artifact persistence.  A reader factory
owns model access and must return an object implementing
``ToolSelectionReader``.  A custom memory bridge can be injected; otherwise
the canonical public in-memory Swarm Brain runtime is used.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.integrations.mem2act import (
    BenchmarkConfig,
    Mem2ActContractError,
    Mem2ActEvaluator,
    build_public_in_memory_bridge,
    load_mem2act_dataset,
    write_benchmark_outputs,
)
from benchmarks.integrations.mem2act.runner import resolve_factory

DEFAULT_REPO = Path(os.getenv("MEM2ACTBENCH_REPO", "/private/tmp/swarmbrain-mem2act"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
        help="official Mem2ActBench checkout at the pinned commit",
    )
    parser.add_argument(
        "--reader-factory",
        required=True,
        help=(
            "zero-argument module:callable returning ToolSelectionReader; "
            "the callable may be async and may read provider settings from its own environment; "
            "canonical: benchmarks.integrations.mem2act.openai_reader:build_reader"
        ),
    )
    parser.add_argument(
        "--memory-bridge-factory",
        help=(
            "optional zero-argument module:callable returning MemoryBridge; "
            "defaults to Swarm Brain's public in-memory runtime"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="writes PREFIX-predictions.jsonl, PREFIX-run.json, and PREFIX-report.json",
    )
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--retrieval-token-budget", type=int, default=8_192)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_080_9)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument(
        "--expected-reader-model",
        help="fail provider calls whose reported model differs from this exact identifier",
    )
    parser.add_argument(
        "--reader-revision",
        help="immutable reader revision recorded in the report (commit/checkpoint/date)",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        help="development smoke subset; the report is marked incomplete unless all 400 run",
    )
    parser.add_argument(
        "--bridge-seed",
        default="official-v1",
        help="scope seed for the default isolated public runtime bridge",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output artifacts",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    dataset_dir = args.repo.expanduser().resolve()
    dataset = load_mem2act_dataset(dataset_dir)
    reader = await resolve_factory(_factory(args.reader_factory)())
    _require_declared_reader_identity(reader, args)
    if args.memory_bridge_factory:
        memory = await resolve_factory(_factory(args.memory_bridge_factory)())
    else:
        memory = await build_public_in_memory_bridge(seed=args.bridge_seed)

    _require_methods(reader, "reader", ("select_tool",))
    _require_methods(memory, "memory bridge", ("ingest", "retrieve", "close"))
    config = BenchmarkConfig(
        retrieval_limit=args.retrieval_limit,
        retrieval_token_budget=args.retrieval_token_budget,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
        task_limit=args.task_limit,
        expected_reader_model=args.expected_reader_model,
        reader_revision=args.reader_revision,
    )
    try:
        result = await Mem2ActEvaluator(dataset, memory, reader, config=config).run()
        paths = write_benchmark_outputs(
            result,
            args.output_prefix,
            dataset_dir=dataset_dir,
            overwrite=args.overwrite,
        )
    finally:
        await memory.close()
        reader_close = getattr(reader, "close", None)
        if callable(reader_close):
            await resolve_factory(reader_close())

    print(f"predictions: {paths.predictions}")
    print(f"run provenance: {paths.run}")
    print(f"report: {paths.report}")
    print(
        "protocol: "
        f"{result.report['evaluated_task_count']} tasks, "
        f"{result.report['prediction_count']} arm predictions, "
        f"complete={result.report['evaluation']['complete_400_task_protocol']}"
    )
    return 0


def _factory(spec: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise Mem2ActContractError("factory must use module:callable syntax")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise Mem2ActContractError(f"cannot load factory {spec!r}") from exc
    if not callable(factory):
        raise Mem2ActContractError(f"factory target is not callable: {spec!r}")
    return factory


def _require_methods(value: Any, name: str, methods: tuple[str, ...]) -> None:
    missing = [method for method in methods if not callable(getattr(value, method, None))]
    if missing:
        raise Mem2ActContractError(f"{name} is missing methods: {missing}")


def _require_declared_reader_identity(reader: Any, args: argparse.Namespace) -> None:
    """Keep adapter identity and gate-facing CLI provenance exactly aligned."""

    missing = object()
    configured_model = getattr(reader, "model", missing)
    if configured_model is not missing and configured_model != args.expected_reader_model:
        raise Mem2ActContractError(
            "reader factory model must exactly match --expected-reader-model"
        )
    configured_revision = getattr(reader, "revision", missing)
    if configured_revision is not missing and configured_revision != args.reader_revision:
        raise Mem2ActContractError("reader factory revision must exactly match --reader-revision")


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except (Mem2ActContractError, FileExistsError) as exc:
        raise SystemExit(f"Mem2ActBench failed closed: {exc}") from exc


if __name__ == "__main__":
    main()
