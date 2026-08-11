#!/usr/bin/env python3
"""Preflight, dry-run, execute, and package LongMemEval-V2 Swarm evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.integrations.longmemeval_v2.contracts import (  # noqa: E402
    LongMemEvalV2AdapterError,
)
from benchmarks.integrations.longmemeval_v2.evidence import (  # noqa: E402
    write_operation_sidecar,
)
from benchmarks.integrations.longmemeval_v2.runner import (  # noqa: E402
    dry_run,
    execute_official_run,
    preflight_official_environment,
)


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("bridge params must be valid JSON") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("bridge params must be one JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LongMemEval-V2 Swarm backend and operation-sidecar runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "dry-run",
        help="exercise the query-private adapter and trace binder without endpoints",
    )

    preflight = subparsers.add_parser(
        "preflight", help="verify checkout, dataset, revision, and fixed model protocol"
    )
    preflight.add_argument("--repository", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--tier", choices=("small", "medium"), required=True)
    preflight.add_argument("--dataset-revision", required=True)
    preflight.add_argument("--expected-dataset-manifest-sha256", default=None)

    run = subparsers.add_parser(
        "run", help="run one complete official domain after strict preflight"
    )
    run.add_argument("--repository", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--operation-ledger", type=Path, required=True)
    run.add_argument("--domain", choices=("web", "enterprise"), required=True)
    run.add_argument("--tier", choices=("small", "medium"), required=True)
    run.add_argument("--operating-point", required=True)
    run.add_argument("--dataset-revision", required=True)
    run.add_argument("--expected-dataset-manifest-sha256", required=True)
    run.add_argument("--bridge-factory", required=True, help="module:callable")
    run.add_argument("--bridge-params", type=_json_object, default={})
    run.add_argument("--reader-base-url", required=True)
    run.add_argument("--reader-api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--evaluator-base-url", required=True)
    run.add_argument("--evaluator-api-key-env", default="OPENAI_API_KEY")
    run.add_argument(
        "--allow-model-api-calls",
        action="store_true",
        help="required acknowledgement: this subcommand calls the fixed reader and judge",
    )

    sidecar = subparsers.add_parser(
        "sidecar", help="bind complete domain ledgers to an official package"
    )
    sidecar.add_argument("--package", type=Path, required=True)
    sidecar.add_argument("--operation-ledger", type=Path, action="append", required=True)
    sidecar.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            print(json.dumps(dry_run(), ensure_ascii=True, sort_keys=True, indent=2))
            return 0
        if args.command == "preflight":
            result = preflight_official_environment(
                args.repository,
                args.data_root,
                tier=args.tier,
                dataset_revision=args.dataset_revision,
                expected_dataset_manifest_sha256=args.expected_dataset_manifest_sha256,
            )
            print(json.dumps(result.as_json(), ensure_ascii=True, sort_keys=True, indent=2))
            return 0 if result.ready else 2
        if args.command == "sidecar":
            payload = write_operation_sidecar(
                args.package,
                args.operation_ledger,
                args.output,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output.resolve()),
                        "tier": payload["tier"],
                        "operating_points": len(payload["operating_points"]),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        if not args.allow_model_api_calls:
            parser.error("run requires --allow-model-api-calls")
        execute_official_run(
            repository=args.repository,
            data_root=args.data_root,
            output_dir=args.output_dir,
            ledger_path=args.operation_ledger,
            domain=args.domain,
            tier=args.tier,
            operating_point=args.operating_point,
            dataset_revision=args.dataset_revision,
            expected_dataset_manifest_sha256=args.expected_dataset_manifest_sha256,
            bridge_factory_spec=args.bridge_factory,
            bridge_params=args.bridge_params,
            reader_base_url=args.reader_base_url,
            reader_api_key_env=args.reader_api_key_env,
            evaluator_base_url=args.evaluator_base_url,
            evaluator_api_key_env=args.evaluator_api_key_env,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir.resolve()),
                    "operation_ledger": str(args.operation_ledger.resolve()),
                    "status": "official domain run complete",
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except LongMemEvalV2AdapterError as exc:
        print(json.dumps({"error": str(exc), "status": "blocked"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
