#!/usr/bin/env python3
"""Generate GateMem external predictions through Swarm Brain memory APIs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.integrations.gatemem.answering import (
    ANSWER_PROMPT_SHA256,
    ANSWER_PROMPT_VERSION,
    ANSWER_PROTOCOL_VERSION,
    OpenAICompatibleAnswerModel,
    answer_decoding_config,
)
from benchmarks.integrations.gatemem.completion import (
    build_execution_lineage,
    default_completion_path,
    write_completion_manifest,
)
from benchmarks.integrations.gatemem.contracts import (
    GATEMEM_COMMIT,
    GATEMEM_DOMAINS,
    GATEMEM_SHA256,
    GateMemCheckout,
    GateMemContractError,
    assert_hidden_fields_absent,
)
from benchmarks.integrations.gatemem.gateway import (
    HttpMemoryGateway,
    RuntimeMemoryGateway,
    StaticTokenProvider,
)
from benchmarks.integrations.gatemem.policy import (
    DeterministicTurnInterpreter,
    ManifestAudiencePolicy,
    SpeakerOnlyAudiencePolicy,
)
from benchmarks.integrations.gatemem.resume import (
    DEFAULT_RESUME_KEY_ENV,
    AuthenticatedResumeStore,
    EpisodeResumeSpec,
    file_sha256,
    load_resume_key,
)
from benchmarks.integrations.gatemem.runner import (
    DELETION_SCHEMA,
    HARNESS_SCHEMA_VERSION,
    TURN_SCHEMA,
    GateMemHarness,
    HarnessConfig,
    HarnessRun,
)

from swarmbrain.application.runtime import build_in_memory_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pinned GateMem domain incrementally through principal-scoped "
            "Swarm Brain memory and write official external predictions"
        )
    )
    parser.add_argument("--gatemem-dir", default="/private/tmp/swarmbrain-gatemem")
    parser.add_argument("--domain", required=True, choices=sorted(GATEMEM_DOMAINS))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--completion-manifest",
        type=Path,
        help=(
            "hash-binding completion marker written after predictions and audit; "
            "defaults to <predictions>.completion.json"
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--backend", choices=("memory", "http"), default="memory")
    parser.add_argument("--api-url", default=os.getenv("SWARMBRAIN_API_URL"))
    parser.add_argument("--token-manifest", type=Path)
    parser.add_argument("--audience-manifest", type=Path)
    parser.add_argument("--answer-base-url", default=os.getenv("GATEMEM_ANSWER_BASE_URL"))
    parser.add_argument("--answer-model", default=os.getenv("GATEMEM_ANSWER_MODEL"))
    parser.add_argument(
        "--answer-revision",
        default=os.getenv("GATEMEM_ANSWER_REVISION"),
        help="immutable provider-reported model/deployment revision",
    )
    parser.add_argument("--answer-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--answer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument(
        "--score-out-dir",
        type=Path,
        help="print the pinned official rule-scorer command after generation",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help=(
            "opt in to authenticated episode-boundary checkpoints for --backend memory; "
            "path must end with .resume.json"
        ),
    )
    parser.add_argument(
        "--resume-key-env",
        default=DEFAULT_RESUME_KEY_ENV,
        help="environment variable containing at least 32 bytes of local HMAC key material",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    checkout = GateMemCheckout(args.gatemem_dir)
    checkout.verify(domain=args.domain)
    if args.verify_only:
        print(
            json.dumps(
                {
                    "status": "verified",
                    "gatemem_commit": GATEMEM_COMMIT,
                    "domain": args.domain,
                },
                sort_keys=True,
            )
        )
        return 0
    _validate_resume_backend(backend=args.backend, resume_state=args.resume_state)
    if args.predictions is None:
        raise GateMemContractError("--predictions is required unless --verify-only is used")
    if not args.answer_base_url or not args.answer_model or not args.answer_revision:
        raise GateMemContractError(
            "--answer-base-url, --answer-model, and --answer-revision "
            "(or their GATEMEM_* env vars) are required"
        )

    dataset = checkout.load(args.domain)
    selected_ids = frozenset(args.episode_id)
    episodes = tuple(
        item
        for item in dataset.episodes
        if not selected_ids or str(item.get("episode_id")) in selected_ids
    )
    if selected_ids:
        found = {str(item.get("episode_id")) for item in episodes}
        missing = selected_ids.difference(found)
        if missing:
            raise GateMemContractError(f"unknown --episode-id values: {sorted(missing)}")
    episode_ids = {str(item.get("episode_id")) for item in episodes}
    checkpoints = tuple(
        item for item in dataset.checkpoints if str(item.get("episode_id")) in episode_ids
    )

    audience_policy = (
        ManifestAudiencePolicy.from_path(args.audience_manifest)
        if args.audience_manifest
        else SpeakerOnlyAudiencePolicy()
    )
    harness_config = HarnessConfig(
        recall_limit=args.recall_limit,
        min_score=args.min_score,
        context_token_budget=args.context_token_budget,
    )
    audit_path = args.audit_output or args.predictions.with_suffix(
        args.predictions.suffix + ".audit.json"
    )
    completion_path = args.completion_manifest or default_completion_path(args.predictions)
    _validate_output_paths(
        predictions=args.predictions,
        audit=audit_path,
        completion=completion_path,
        resume_state=args.resume_state,
        audience_manifest=args.audience_manifest,
        token_manifest=args.token_manifest,
    )
    token_values: dict[str, str] | None = None
    token_manifest_sha256: str | None = None
    if args.backend == "http":
        if not args.api_url or args.token_manifest is None:
            raise GateMemContractError("--backend http requires --api-url and --token-manifest")
        token_values = _load_token_manifest(args.token_manifest)
        token_manifest_sha256 = file_sha256(args.token_manifest, label="token manifest")
    implementation_fingerprint = _implementation_fingerprint()

    episode_checkpoints: dict[str, tuple[dict[str, Any], ...]] = {}
    resume_store: AuthenticatedResumeStore | None = None
    if args.resume_state is not None:
        episode_checkpoints = {
            str(episode.get("episode_id")): tuple(
                checkpoint
                for checkpoint in checkpoints
                if str(checkpoint.get("episode_id")) == str(episode.get("episode_id"))
            )
            for episode in episodes
        }
        resume_specs = tuple(
            EpisodeResumeSpec(
                episode_id=str(episode.get("episode_id")),
                checkpoint_ids=_ordered_checkpoint_ids(
                    episode,
                    episode_checkpoints[str(episode.get("episode_id"))],
                ),
            )
            for episode in episodes
        )
        resume_store = AuthenticatedResumeStore(
            path=args.resume_state,
            key=load_resume_key(args.resume_key_env),
            fingerprint=_resume_fingerprint(
                args=args,
                audience_policy=audience_policy,
                specs=resume_specs,
                audit_path=audit_path,
                completion_path=completion_path,
                token_manifest_sha256=token_manifest_sha256,
                implementation_fingerprint=implementation_fingerprint,
            ),
            episodes=resume_specs,
        )
        with resume_store.locked():
            completed = resume_store.load_or_initialize()
            if len(completed) == len(resume_specs):
                return _write_completed_result(
                    args=args,
                    checkout=checkout,
                    episodes=episodes,
                    audit_path=audit_path,
                    completion_path=completion_path,
                    result=resume_store.combine_complete(completed),
                    resumed_episodes=len(completed),
                    authenticated_state_payload_sha256=(
                        resume_store.authenticated_payload_sha256(completed)
                    ),
                    implementation_fingerprint=implementation_fingerprint,
                )
            return await _execute_run(
                args=args,
                checkout=checkout,
                episodes=episodes,
                checkpoints=checkpoints,
                episode_checkpoints=episode_checkpoints,
                audience_policy=audience_policy,
                harness_config=harness_config,
                token_values=token_values,
                audit_path=audit_path,
                completion_path=completion_path,
                resume_store=resume_store,
                completed=completed,
                implementation_fingerprint=implementation_fingerprint,
            )

    return await _execute_run(
        args=args,
        checkout=checkout,
        episodes=episodes,
        checkpoints=checkpoints,
        episode_checkpoints=episode_checkpoints,
        audience_policy=audience_policy,
        harness_config=harness_config,
        token_values=token_values,
        audit_path=audit_path,
        completion_path=completion_path,
        implementation_fingerprint=implementation_fingerprint,
    )


def _validate_resume_backend(*, backend: str, resume_state: Path | None) -> None:
    if resume_state is not None and backend != "memory":
        raise GateMemContractError(
            "--resume-state currently requires --backend memory; restarting a partial "
            "episode against a durable HTTP backend cannot reproduce earlier checkpoints"
        )


async def _execute_run(
    *,
    args: argparse.Namespace,
    checkout: GateMemCheckout,
    episodes: tuple[dict[str, Any], ...],
    checkpoints: tuple[dict[str, Any], ...],
    episode_checkpoints: dict[str, tuple[dict[str, Any], ...]],
    audience_policy: ManifestAudiencePolicy | SpeakerOnlyAudiencePolicy,
    harness_config: HarnessConfig,
    token_values: dict[str, str] | None,
    audit_path: Path,
    completion_path: Path,
    resume_store: AuthenticatedResumeStore | None = None,
    completed: tuple[HarnessRun, ...] = (),
    implementation_fingerprint: dict[str, Any],
) -> int:
    answer_model = OpenAICompatibleAnswerModel(
        base_url=args.answer_base_url,
        model=args.answer_model,
        model_revision=args.answer_revision,
        api_key=os.getenv(args.answer_api_key_env),
        timeout_seconds=args.answer_timeout_seconds,
    )
    runtime = None
    http_gateway = None
    try:
        if args.backend == "memory":
            runtime = build_in_memory_runtime("gatemem-local-runtime-secret")
            await runtime.start()
            gateway = RuntimeMemoryGateway(runtime)
        else:
            if not args.api_url or token_values is None:
                raise GateMemContractError("validated HTTP backend configuration is missing")
            tokens = StaticTokenProvider(token_values)
            http_gateway = HttpMemoryGateway(base_url=args.api_url, tokens=tokens)
            gateway = http_gateway

        harness = GateMemHarness(
            gateway=gateway,
            answer_model=answer_model,
            audience_policy=audience_policy,
            config=harness_config,
        )
        resumed_episodes = len(completed) if resume_store is not None else None
        if resume_store is None:
            result = await harness.run(episodes=episodes, checkpoints=checkpoints)
            state_payload_sha256 = None
        else:
            for episode in episodes[len(completed) :]:
                episode_id = str(episode.get("episode_id"))
                episode_run = await harness.run(
                    episodes=(episode,),
                    checkpoints=episode_checkpoints[episode_id],
                )
                completed = resume_store.append_episode(completed, episode_run)
            result = resume_store.combine_complete(completed)
            state_payload_sha256 = resume_store.authenticated_payload_sha256(completed)
        return _write_completed_result(
            args=args,
            checkout=checkout,
            episodes=episodes,
            audit_path=audit_path,
            completion_path=completion_path,
            result=result,
            resumed_episodes=resumed_episodes,
            authenticated_state_payload_sha256=state_payload_sha256,
            implementation_fingerprint=implementation_fingerprint,
        )
    finally:
        await answer_model.close()
        if http_gateway is not None:
            await http_gateway.close()
        if runtime is not None:
            await runtime.close()


def _write_completed_result(
    *,
    args: argparse.Namespace,
    checkout: GateMemCheckout,
    episodes: tuple[dict[str, Any], ...],
    audit_path: Path,
    completion_path: Path,
    result: HarnessRun,
    resumed_episodes: int | None,
    authenticated_state_payload_sha256: str | None,
    implementation_fingerprint: dict[str, Any],
) -> int:
    # Canonical artifacts are emitted only after the selected plan is complete.
    result.write_predictions(args.predictions)
    result.write_audit(audit_path)
    if resumed_episodes is None:
        execution_mode = "uninterrupted"
        completed_prefix_episodes = 0
    elif resumed_episodes == 0:
        execution_mode = "checkpointed"
        completed_prefix_episodes = 0
    elif resumed_episodes == len(episodes):
        execution_mode = "complete_replay"
        completed_prefix_episodes = resumed_episodes
    else:
        execution_mode = "resumed"
        completed_prefix_episodes = resumed_episodes
    execution_lineage = build_execution_lineage(
        mode=execution_mode,
        completed_prefix_episodes=completed_prefix_episodes,
        completed_episodes=len(episodes),
        authenticated_state_payload_sha256=authenticated_state_payload_sha256,
        implementation_fingerprint=implementation_fingerprint,
    )
    write_completion_manifest(
        completion_path,
        domain=args.domain,
        predictions_path=args.predictions,
        audit_path=audit_path,
        execution_lineage=execution_lineage,
    )
    summary = {
        "status": "predictions_written",
        "gatemem_commit": GATEMEM_COMMIT,
        "domain": args.domain,
        "episodes": len(episodes),
        "checkpoints": len(result.predictions),
        "predictions": str(args.predictions.resolve()),
        "audit": str(audit_path.resolve()),
        "completion_manifest": str(completion_path.resolve()),
    }
    if resumed_episodes is not None:
        summary["resumed_episodes"] = resumed_episodes
    print(json.dumps(summary, sort_keys=True))
    if args.score_out_dir is not None:
        command = checkout.official_score_command(
            domain=args.domain,
            predictions=args.predictions,
            out_dir=args.score_out_dir,
            python_executable=sys.executable,
        )
        print(f"Official rule scorer: {shlex.join(command)}")
    return 0


def _ordered_checkpoint_ids(
    episode: dict[str, Any], checkpoints: tuple[dict[str, Any], ...]
) -> tuple[str, ...]:
    turns = episode.get("turns")
    if not isinstance(turns, list):
        raise GateMemContractError("GateMem episode turns are malformed")
    turn_positions = {
        str(turn.get("turn_id")): index
        for index, turn in enumerate(turns)
        if isinstance(turn, dict)
    }
    ordered: list[tuple[int, int, str]] = []
    for source_index, checkpoint in enumerate(checkpoints):
        checkpoint_id = checkpoint.get("checkpoint_id")
        as_of_turn_id = checkpoint.get("as_of_turn_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise GateMemContractError("GateMem checkpoint ID is malformed")
        if not isinstance(as_of_turn_id, str) or as_of_turn_id not in turn_positions:
            raise GateMemContractError(f"checkpoint {checkpoint_id!r} has an unknown as-of turn")
        ordered.append((turn_positions[as_of_turn_id], source_index, checkpoint_id))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ordered)


def _resume_fingerprint(
    *,
    args: argparse.Namespace,
    audience_policy: ManifestAudiencePolicy | SpeakerOnlyAudiencePolicy,
    specs: tuple[EpisodeResumeSpec, ...],
    audit_path: Path,
    completion_path: Path,
    token_manifest_sha256: str | None,
    implementation_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    episodes_sha256 = GATEMEM_SHA256[f"bench/data/{args.domain}/episodes.jsonl"]
    checkpoints_sha256 = GATEMEM_SHA256[f"bench/data/{args.domain}/checkpoints.jsonl"]
    dataset_sha256 = hashlib.sha256(
        f"{episodes_sha256}:{checkpoints_sha256}".encode("ascii")
    ).hexdigest()
    audience = {
        "type": type(audience_policy).__name__,
        "manifest_sha256": getattr(audience_policy, "manifest_sha256", None),
    }
    return {
        "schema_version": 1,
        "benchmark": "GateMem",
        "dataset": {
            "repository_commit": GATEMEM_COMMIT,
            "domain": args.domain,
            "episodes_sha256": episodes_sha256,
            "checkpoints_sha256": checkpoints_sha256,
            "combined_sha256": dataset_sha256,
        },
        "implementation": implementation_fingerprint,
        "selection": {
            "episodes": [
                {
                    "episode_id": spec.episode_id,
                    "checkpoint_ids": list(spec.checkpoint_ids),
                }
                for spec in specs
            ]
        },
        "audience_policy": audience,
        "answer_model": {
            "provider": "openai-compatible",
            "base_url": args.answer_base_url.rstrip("/"),
            "model": args.answer_model,
            "revision": args.answer_revision,
            "api_key_env": args.answer_api_key_env,
            "timeout_seconds": args.answer_timeout_seconds,
        },
        "protocol": {
            "answer_protocol_version": ANSWER_PROTOCOL_VERSION,
            "answer_prompt_version": ANSWER_PROMPT_VERSION,
            "answer_prompt_sha256": ANSWER_PROMPT_SHA256,
            "answer_decoding": answer_decoding_config(),
            "harness_schema_version": HARNESS_SCHEMA_VERSION,
            "turn_schema": TURN_SCHEMA,
            "deletion_schema": DELETION_SCHEMA,
            "turn_interpreter": DeterministicTurnInterpreter.__name__,
            "official_scorer_sha256": GATEMEM_SHA256["bench/scripts/score_predictions.py"],
            "prediction_format_sha256": GATEMEM_SHA256["docs/prediction_format.md"],
        },
        "run_parameters": {
            "gatemem_checkout_path": str(Path(args.gatemem_dir).resolve()),
            "backend": args.backend,
            "api_url": args.api_url.rstrip("/") if args.api_url else None,
            "audience_manifest_path": (
                str(args.audience_manifest.resolve())
                if args.audience_manifest is not None
                else None
            ),
            "token_manifest_path": (
                str(args.token_manifest.resolve()) if args.token_manifest is not None else None
            ),
            "token_manifest_sha256": token_manifest_sha256,
            "recall_limit": args.recall_limit,
            "min_score": args.min_score,
            "context_token_budget": args.context_token_budget,
            "predictions_path": str(args.predictions.resolve()),
            "audit_path": str(audit_path.resolve()),
            "completion_manifest_path": str(completion_path.resolve()),
            "resume_state_path": str(args.resume_state.resolve()),
            "resume_key_env": args.resume_key_env,
            "score_out_dir": (
                str(args.score_out_dir.resolve()) if args.score_out_dir is not None else None
            ),
        },
    }


def _implementation_fingerprint() -> dict[str, Any]:
    """Bind resumptions to the exact local harness/runtime implementation."""

    roots = (
        REPO_ROOT / "benchmarks" / "integrations" / "gatemem",
        REPO_ROOT / "src" / "swarmbrain",
    )
    paths = [
        REPO_ROOT / "scripts" / "run_gatemem_external.py",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
    ]
    for root in roots:
        paths.extend(root.rglob("*.py"))
    hashes = {
        path.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": hashes,
    }


def _validate_output_paths(
    *,
    predictions: Path,
    audit: Path,
    completion: Path,
    resume_state: Path | None,
    audience_manifest: Path | None,
    token_manifest: Path | None,
) -> None:
    named_paths = {
        "predictions": predictions.resolve(),
        "audit": audit.resolve(),
        "completion manifest": completion.resolve(),
    }
    if resume_state is not None:
        resume_lock = resume_state.with_name(resume_state.name + ".lock")
        named_paths["resume state"] = resume_state.resolve()
        named_paths["resume lock"] = resume_lock.resolve()
    if len(set(named_paths.values())) != len(named_paths):
        raise GateMemContractError("GateMem output paths must use distinct paths")
    inputs = {
        "audience manifest": audience_manifest,
        "token manifest": token_manifest,
    }
    for input_name, input_path in inputs.items():
        if input_path is not None and input_path.resolve() in named_paths.values():
            raise GateMemContractError(f"{input_name} cannot overlap a GateMem output path")


def _load_token_manifest(path: Path) -> dict[str, str]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateMemContractError(f"invalid token manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise GateMemContractError("token manifest must be an object")
    assert_hidden_fields_absent(raw)
    if raw.get("schema_version") != 1 or raw.get("gatemem_commit") != GATEMEM_COMMIT:
        raise GateMemContractError("token manifest must name schema 1 and the pinned commit")
    tokens = raw.get("tokens")
    if not isinstance(tokens, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in tokens.items()
    ):
        raise GateMemContractError("token manifest tokens must map scope keys to bearer tokens")
    return dict(tokens)


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_run(_parser().parse_args())))
    except GateMemContractError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
