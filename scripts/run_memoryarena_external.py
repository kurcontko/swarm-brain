#!/usr/bin/env python3
"""Preflight or serve the pinned MemoryArena memory-API bridge.

This scaffold never launches the preview benchmark runners. Preflight and the
default deterministic server make no model/provider call; the explicit strict
semantic server may call only its configured embedding provider. Official
scoring remains fail-closed until the upstream preview exposes a verifiable
full-protocol result boundary.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.integrations.memoryarena import (
    DETERMINISTIC_EMBEDDING_MODE,
    SEMANTIC_EMBEDDING_DIMENSIONS,
    SEMANTIC_EMBEDDING_MODE,
    SEMANTIC_EMBEDDING_MODEL_ID,
    BridgeConfig,
    MemoryArenaContractError,
    MemoryArenaRuntimeBridge,
    create_memoryarena_app,
    run_preflight,
)

DEFAULT_CHECKOUT = Path(
    os.getenv(
        "MEMORYARENA_REPO",
        "/private/tmp/swarmbrain-memoryarena-official-audit",
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="validate immutable external inputs")
    _add_checkout_argument(preflight)
    preflight.add_argument("--dataset-manifest", type=Path)
    preflight.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="repeat once for each of the five paper task config overlays",
    )
    preflight.add_argument(
        "--bridge-only",
        action="store_true",
        help="exit zero when the pinned memory API is compatible, despite official-run blockers",
    )

    serve = commands.add_parser("serve", help="serve the local compatibility API")
    _add_checkout_argument(serve)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--embedding-mode",
        choices=(DETERMINISTIC_EMBEDDING_MODE, SEMANTIC_EMBEDDING_MODE),
        default=DETERMINISTIC_EMBEDDING_MODE,
    )
    serve.add_argument("--embedding-base-url")
    serve.add_argument("--embedding-api-key-env")
    serve.add_argument("--embedding-model-id")
    serve.add_argument("--embedding-model-revision")
    serve.add_argument("--embedding-response-model")
    serve.add_argument("--embedding-dimensions", type=int)
    serve.add_argument(
        "--evidence-output",
        type=Path,
        help="write content-free bridge evidence after graceful shutdown; refuses overwrite",
    )
    return parser


def _add_checkout_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkout",
        type=Path,
        default=DEFAULT_CHECKOUT,
        help="official MemoryArena checkout at the pinned commit",
    )


def run_preflight_command(args: argparse.Namespace) -> int:
    report = run_preflight(
        args.checkout,
        dataset_manifest=args.dataset_manifest,
        configs=tuple(args.config),
    )
    print(json.dumps(report.as_json(), ensure_ascii=False, sort_keys=True, indent=2))
    ready = report.bridge_compatible if args.bridge_only else report.official_protocol_ready
    return 0 if ready else 1


def run_server_command(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65_535:
        raise SystemExit("MemoryArena failed closed: --port must be in [1, 65535]")
    preflight = run_preflight(args.checkout)
    if not preflight.bridge_compatible:
        raise SystemExit("MemoryArena failed closed: pinned official memory API did not validate")
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("MemoryArena failed closed: install swarmbrain[serve]") from exc

    dimensions = args.embedding_dimensions
    if dimensions is None:
        dimensions = (
            SEMANTIC_EMBEDDING_DIMENSIONS if args.embedding_mode == SEMANTIC_EMBEDDING_MODE else 256
        )
    try:
        config = BridgeConfig(
            embedding_mode=args.embedding_mode,
            embedding_dimensions=dimensions,
            embedding_base_url=args.embedding_base_url,
            embedding_api_key_env=args.embedding_api_key_env,
            embedding_model_id=args.embedding_model_id,
            embedding_model_revision=args.embedding_model_revision,
            embedding_response_model=args.embedding_response_model,
        )
        bridge = MemoryArenaRuntimeBridge(config=config)
    except MemoryArenaContractError as exc:
        raise SystemExit(f"MemoryArena failed closed: {exc}") from exc
    app = create_memoryarena_app(bridge)
    if args.embedding_mode == SEMANTIC_EMBEDDING_MODE:
        print(
            "Serving the strict semantic MemoryArena API on 127.0.0.1 with "
            f"{SEMANTIC_EMBEDDING_MODEL_ID}; evidence remains nonpublishable until the "
            "served weights revision is independently bound."
        )
    else:
        print(
            "Serving the local deterministic MemoryArena API on 127.0.0.1; "
            "evidence from this mode is explicitly nonpublishable."
        )
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    if args.evidence_output is not None:
        _write_new_json(args.evidence_output, bridge.evidence())
        print(f"content-free evidence: {args.evidence_output}")
    return 0


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        return run_preflight_command(args)
    return run_server_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
