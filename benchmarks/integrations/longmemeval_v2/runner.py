"""Offline preflight and opt-in official LongMemEval-V2 execution boundary."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter import (
    BridgeFactory,
    BridgeLifecycle,
    SwarmQueryAdapter,
    TraceJournal,
    bind_official_prompt_row,
    build_official_memory_class,
    official_prompt_row_binding,
)
from .contracts import (
    EXPECTED_JUDGE_MODEL,
    EXPECTED_QUESTIONS,
    EXPECTED_READER_MODEL,
    MEMORY_TYPE,
    PINNED_REPOSITORY_COMMIT,
    AdapterConfig,
    EmbeddingRuntimeEvidence,
    LongMemEvalV2AdapterError,
    ReadExpandMemoryResult,
    RecallMemoryResult,
)
from .evidence import EvidenceLedger, canonical_sha256, ledger_payload, write_ledger


@dataclass(frozen=True, slots=True)
class PreflightResult:
    ready: bool
    repository_commit: str | None
    repository_clean: bool
    dataset_manifest_sha256: str | None
    expected_dataset_manifest_sha256: str | None
    missing_paths: tuple[str, ...]
    blockers: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "repository_commit": self.repository_commit,
            "repository_clean": self.repository_clean,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "expected_dataset_manifest_sha256": self.expected_dataset_manifest_sha256,
            "missing_paths": list(self.missing_paths),
            "blockers": list(self.blockers),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LongMemEvalV2AdapterError(f"cannot read dataset file {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LongMemEvalV2AdapterError(
                f"invalid dataset JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise LongMemEvalV2AdapterError(
                f"dataset JSONL row at {path}:{line_number} must be an object"
            )
        rows.append(value)
    return rows


def _logical_data_file(data_root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise LongMemEvalV2AdapterError(f"{label} must be a non-empty relative path")
    logical = Path(raw)
    if logical.is_absolute() or ".." in logical.parts:
        raise LongMemEvalV2AdapterError(f"{label} must remain under the dataset root")
    path = data_root / logical
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LongMemEvalV2AdapterError(f"missing referenced dataset file {path}") from exc
    if not resolved.is_relative_to(data_root.resolve()) or not resolved.is_file():
        raise LongMemEvalV2AdapterError(f"{label} escapes the dataset root")
    return path


def dataset_manifest(data_root: Path, *, tier: str, dataset_revision: str) -> dict[str, Any]:
    """Hash core JSON and every image referenced by the public benchmark rows."""

    data_root = data_root.resolve()
    if tier not in {"small", "medium"}:
        raise LongMemEvalV2AdapterError("tier must be small or medium")
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise LongMemEvalV2AdapterError("dataset_revision must be a non-empty string")
    core_paths = (
        data_root / "questions.jsonl",
        data_root / "trajectories.jsonl",
        data_root / "haystacks" / f"lme_v2_{tier}.json",
    )
    for path in core_paths:
        if not path.is_file():
            raise LongMemEvalV2AdapterError(f"missing official dataset file {path}")

    questions = _read_jsonl(core_paths[0])
    trajectories = _read_jsonl(core_paths[1])
    if len(questions) != EXPECTED_QUESTIONS:
        raise LongMemEvalV2AdapterError(
            f"official dataset must contain exactly {EXPECTED_QUESTIONS} questions"
        )
    question_ids: set[str] = set()
    image_paths: dict[str, Path] = {}
    for row in questions:
        question_id = row.get("id")
        if not isinstance(question_id, str) or not question_id or question_id in question_ids:
            raise LongMemEvalV2AdapterError("dataset question IDs must be unique strings")
        question_ids.add(question_id)
        if row.get("domain") not in {"web", "enterprise"}:
            raise LongMemEvalV2AdapterError(f"question {question_id!r} has an invalid domain")
        image = row.get("image")
        if image is not None:
            path = _logical_data_file(data_root, image, label=f"question {question_id!r} image")
            image_paths[Path(str(image)).as_posix()] = path

    trajectory_ids: set[str] = set()
    trajectory_domains: dict[str, str] = {}
    for row in trajectories:
        trajectory_id = row.get("id")
        if (
            not isinstance(trajectory_id, str)
            or not trajectory_id
            or trajectory_id in trajectory_ids
        ):
            raise LongMemEvalV2AdapterError("dataset trajectory IDs must be unique strings")
        trajectory_ids.add(trajectory_id)
        domain = row.get("domain")
        if domain not in {"web", "enterprise"}:
            raise LongMemEvalV2AdapterError(f"trajectory {trajectory_id!r} has an invalid domain")
        trajectory_domains[trajectory_id] = domain
        states = row.get("states", [])
        if not isinstance(states, list):
            raise LongMemEvalV2AdapterError(f"trajectory {trajectory_id!r} states must be a list")
        for index, state in enumerate(states):
            if not isinstance(state, dict):
                raise LongMemEvalV2AdapterError(
                    f"trajectory {trajectory_id!r} state {index} must be an object"
                )
            screenshot = state.get("screenshot")
            if screenshot is not None:
                path = _logical_data_file(
                    data_root,
                    screenshot,
                    label=f"trajectory {trajectory_id!r} screenshot",
                )
                image_paths[Path(str(screenshot)).as_posix()] = path

    try:
        haystack = json.loads(core_paths[2].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongMemEvalV2AdapterError(f"cannot read official haystack {core_paths[2]}") from exc
    if not isinstance(haystack, dict) or set(haystack) != question_ids:
        raise LongMemEvalV2AdapterError("haystack coverage differs from official questions")
    question_domain = {str(row["id"]): str(row["domain"]) for row in questions}
    for question_id, raw_ids in haystack.items():
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) and item for item in raw_ids
        ):
            raise LongMemEvalV2AdapterError(f"haystack {question_id!r} must be a list of IDs")
        if len(set(raw_ids)) != len(raw_ids):
            raise LongMemEvalV2AdapterError(f"haystack {question_id!r} repeats a trajectory")
        for trajectory_id in raw_ids:
            if trajectory_id not in trajectory_ids:
                raise LongMemEvalV2AdapterError(
                    f"haystack {question_id!r} references an unknown trajectory"
                )
            if trajectory_domains[trajectory_id] != question_domain[question_id]:
                raise LongMemEvalV2AdapterError(
                    f"haystack {question_id!r} contains a cross-domain trajectory"
                )

    files = {path.relative_to(data_root).as_posix(): _sha256_file(path) for path in core_paths}
    for logical, path in sorted(image_paths.items()):
        files[logical] = _sha256_file(path)
    files = dict(sorted(files.items()))
    return {
        "dataset_repository": "xiaowu0162/longmemeval-v2",
        "dataset_revision": dataset_revision.strip(),
        "tier": tier,
        "questions": len(questions),
        "trajectories": len(trajectories),
        "files_sha256": files,
        "manifest_sha256": canonical_sha256(
            {
                "dataset_repository": "xiaowu0162/longmemeval-v2",
                "dataset_revision": dataset_revision.strip(),
                "files_sha256": files,
                "tier": tier,
            }
        ),
    }


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def preflight_official_environment(
    repository: Path,
    data_root: Path,
    *,
    tier: str,
    dataset_revision: str,
    expected_dataset_manifest_sha256: str | None,
    reader_model: str = EXPECTED_READER_MODEL,
    judge_model: str = EXPECTED_JUDGE_MODEL,
) -> PreflightResult:
    repository = repository.resolve()
    data_root = data_root.resolve()
    blockers: list[str] = []
    missing_paths: list[str] = []
    commit = _git_value(repository, "rev-parse", "HEAD") if repository.is_dir() else None
    status = _git_value(repository, "status", "--porcelain") if commit is not None else None
    repository_clean = status == ""
    if commit != PINNED_REPOSITORY_COMMIT:
        blockers.append(
            f"benchmark checkout must be commit {PINNED_REPOSITORY_COMMIT}; got {commit!r}"
        )
    if not repository_clean:
        blockers.append("benchmark checkout must have no local modifications or untracked files")
    if reader_model != EXPECTED_READER_MODEL:
        blockers.append(f"reader model must be exactly {EXPECTED_READER_MODEL}")
    if judge_model != EXPECTED_JUDGE_MODEL:
        blockers.append(f"judge model must be exactly {EXPECTED_JUDGE_MODEL}")
    required = (
        data_root / "questions.jsonl",
        data_root / "trajectories.jsonl",
        data_root / "haystacks" / f"lme_v2_{tier}.json",
    )
    missing_paths.extend(str(path) for path in required if not path.is_file())
    actual_manifest: str | None = None
    if missing_paths:
        blockers.append("official LongMemEval-V2 dataset files are missing")
    else:
        try:
            actual_manifest = str(
                dataset_manifest(
                    data_root,
                    tier=tier,
                    dataset_revision=dataset_revision,
                )["manifest_sha256"]
            )
        except LongMemEvalV2AdapterError as exc:
            blockers.append(str(exc))
    if expected_dataset_manifest_sha256 is None:
        blockers.append("expected dataset manifest SHA-256 is required before an official run")
    elif actual_manifest is not None and actual_manifest != expected_dataset_manifest_sha256:
        blockers.append("dataset manifest differs from the explicitly pinned digest")
    return PreflightResult(
        ready=not blockers,
        repository_commit=commit,
        repository_clean=repository_clean,
        dataset_manifest_sha256=actual_manifest,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        missing_paths=tuple(missing_paths),
        blockers=tuple(blockers),
    )


def load_bridge_factory(spec: str) -> BridgeFactory:
    if not isinstance(spec, str) or ":" not in spec:
        raise LongMemEvalV2AdapterError("bridge_factory must use module:callable syntax")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute or "." in attribute:
        raise LongMemEvalV2AdapterError("bridge_factory must use module:callable syntax")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise LongMemEvalV2AdapterError(f"cannot load bridge factory {spec!r}") from exc
    if not callable(factory):
        raise LongMemEvalV2AdapterError(f"bridge factory {spec!r} is not callable")
    return factory


def _import_official_modules(repository: Path) -> tuple[Any, Any, Any]:
    repository_text = str(repository.resolve())
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    memory_module = importlib.import_module("memory_modules.memory")
    harness_module = importlib.import_module("evaluation.harness")
    public_data_module = importlib.import_module("data.public_data")
    for module in (memory_module, harness_module, public_data_module):
        source = Path(str(module.__file__)).resolve()
        if not source.is_relative_to(repository.resolve()):
            raise LongMemEvalV2AdapterError(
                "an already-imported module shadows the pinned LongMemEval-V2 checkout"
            )
    return memory_module, harness_module, public_data_module


def execute_official_run(
    *,
    repository: Path,
    data_root: Path,
    output_dir: Path,
    ledger_path: Path,
    domain: str,
    tier: str,
    operating_point: str,
    dataset_revision: str,
    expected_dataset_manifest_sha256: str,
    bridge_factory_spec: str,
    bridge_params: dict[str, Any],
    reader_base_url: str,
    reader_api_key_env: str,
    evaluator_base_url: str,
    evaluator_api_key_env: str,
) -> None:
    """Run the unmodified official harness after every evidence gate passes.

    Calling this function performs reader and judge API requests.  Merely
    importing the module, running preflight, or running ``dry_run`` does not.
    """

    if domain not in {"web", "enterprise"}:
        raise LongMemEvalV2AdapterError("domain must be web or enterprise")
    preflight = preflight_official_environment(
        repository,
        data_root,
        tier=tier,
        dataset_revision=dataset_revision,
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
    )
    if not preflight.ready:
        raise LongMemEvalV2AdapterError(
            "official preflight failed: " + "; ".join(preflight.blockers)
        )
    output_dir = output_dir.resolve()
    ledger_path = ledger_path.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise LongMemEvalV2AdapterError(f"refusing to overwrite output directory: {output_dir}")
    if ledger_path.exists() or ledger_path.is_symlink():
        raise LongMemEvalV2AdapterError(f"refusing to overwrite operation ledger: {ledger_path}")
    if ledger_path == output_dir or output_dir in ledger_path.parents:
        raise LongMemEvalV2AdapterError("operation ledger must remain external to official output")

    factory = load_bridge_factory(bridge_factory_spec)
    memory_module, harness_module, public_data = _import_official_modules(repository.resolve())
    journal = TraceJournal()
    ledger = EvidenceLedger()
    lifecycle = BridgeLifecycle()
    config = AdapterConfig(
        tier=tier,
        operating_point=operating_point,
        dataset_revision=dataset_revision,
        dataset_manifest_sha256=expected_dataset_manifest_sha256,
        bridge_factory=bridge_factory_spec,
        bridge_params=bridge_params,
    )
    runtime_dir = output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True)
    selected_questions = public_data.materialize_runtime_questions(
        data_root=data_root.resolve(),
        domain=domain,
        question_ids=None,
        limit=None,
        output_path=runtime_dir / "questions.json",
    )
    runtime_haystack = public_data.materialize_runtime_haystack(
        data_root=data_root.resolve(),
        tier=tier,
        selected_questions=selected_questions,
        output_path=runtime_dir / "haystack.json",
    )
    if not isinstance(runtime_haystack, dict) or not runtime_haystack:
        raise LongMemEvalV2AdapterError("official runtime haystack must be a non-empty object")
    shared_haystack = (
        len(
            {
                tuple(trajectory_ids)
                for trajectory_ids in runtime_haystack.values()
                if isinstance(trajectory_ids, list)
            }
        )
        == 1
    )
    build_official_memory_class(
        memory_base=memory_module.Memory,
        register_memory=memory_module.register_memory,
        bridge_factory=factory,
        journal=journal,
        lifecycle=lifecycle,
        # Per-question memories become unreachable as soon as their prompt row
        # is built. Close immediately so hundreds of local runtimes do not
        # remain resident through reader generation and judging. A genuinely
        # shared haystack keeps its one bridge until the run-level finally.
        close_after_query=not shared_haystack,
    )
    memory_config_path = runtime_dir / "memory_config.json"
    public_data.write_json(
        memory_config_path,
        {"memory_type": MEMORY_TYPE, "memory_params": config.memory_params()},
    )

    harness_argv = [
        "evaluation.harness",
        "--domain",
        domain,
        "--questions-path",
        str(runtime_dir / "questions.json"),
        "--haystack-path",
        str(runtime_dir / "haystack.json"),
        "--trajectories-path",
        str(data_root.resolve() / "trajectories.jsonl"),
        "--memory-config-path",
        str(memory_config_path),
        "--output-dir",
        str(output_dir),
        "--model",
        EXPECTED_READER_MODEL,
        "--base-url",
        reader_base_url,
        "--api-key-env",
        reader_api_key_env,
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--max-completion-tokens",
        "20000",
        "--memory-context-max-tokens",
        str(config.token_budget),
        "--reader-max-concurrent-requests",
        "16",
        "--prompt-build-max-workers",
        "1",
        "--shuffle-questions-seed",
        "17",
        "--reader-enable-thinking",
        "--evaluator-model",
        EXPECTED_JUDGE_MODEL,
        "--evaluator-base-url",
        evaluator_base_url,
        "--evaluator-api-key-env",
        evaluator_api_key_env,
        "--evaluator-reasoning-effort",
        "medium",
        "--evaluator-max-completion-tokens",
        "4096",
    ]
    previous_argv = sys.argv
    try:
        sys.argv = harness_argv
        with official_prompt_row_binding(
            harness_module,
            journal=journal,
            ledger=ledger,
            tier=tier,
            operating_point=operating_point,
            domain=domain,
        ):
            harness_module.main()
    finally:
        sys.argv = previous_argv
        active_error = sys.exception()
        try:
            lifecycle.close_all()
        except BaseException as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(
                f"LongMemEval-V2 Swarm bridge cleanup also failed: {type(cleanup_error).__name__}"
            )
    if journal.pending() != 0:
        raise LongMemEvalV2AdapterError("official run left unbound operation traces")
    if len(ledger.snapshot()) != len(selected_questions):
        raise LongMemEvalV2AdapterError("operation ledger coverage differs from selected questions")
    try:
        run_args_payload = json.loads((output_dir / "run_args.json").read_text(encoding="utf-8"))
        memory_config_payload = json.loads(memory_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongMemEvalV2AdapterError(
            "cannot bind the operation ledger to exact official runtime inputs"
        ) from exc
    if not isinstance(run_args_payload, dict) or not isinstance(memory_config_payload, dict):
        raise LongMemEvalV2AdapterError("official runtime inputs must be JSON objects")
    write_ledger(
        ledger_path,
        ledger_payload(
            ledger,
            tier=tier,
            operating_point=operating_point,
            domain=domain,
            dataset_revision=dataset_revision,
            dataset_manifest_sha256=expected_dataset_manifest_sha256,
            run_args=run_args_payload,
            memory_config=memory_config_payload,
        ),
    )


class _DryRunBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.closed = False

    def insert_trajectory(self, trajectory: Any) -> None:
        self.calls.append(("insert_trajectory", tuple(sorted(trajectory))))

    def recall_memory(self, query: str, *, limit: int) -> RecallMemoryResult:
        self.calls.append(("recall_memory", (query, limit)))
        return RecallMemoryResult(("canonical-seed",))

    def read_expand_memory(
        self,
        query: str,
        *,
        memory_ids: tuple[str, ...],
        max_depth: int,
        max_fanout: int,
        token_budget: int,
    ) -> ReadExpandMemoryResult:
        self.calls.append(
            (
                "read_expand_memory",
                (query, memory_ids, max_depth, max_fanout, token_budget),
            )
        )
        return ReadExpandMemoryResult(
            ("canonical-seed", "canonical-neighbor"),
            "Bounded context returned only by read_expand_memory.",
        )

    def close(self) -> None:
        self.closed = True

    def embedding_evidence(self) -> EmbeddingRuntimeEvidence:
        return EmbeddingRuntimeEvidence(
            retrieval_mode="lexical",
            sota_capable=False,
            provider=None,
            model=None,
            model_revision=None,
            dimensions=None,
            response_model_requirement=None,
            query_instruction_sha256=None,
            inserted_memories=1,
            embedding_work_completed=0,
            call_accounting_source="bridge-observed-development-mode",
            document_inputs=0,
            document_batch_calls=0,
            document_successful_http_calls=0,
            document_http_attempts=0,
            query_calls=0,
            query_successful_http_calls=0,
            query_http_attempts=0,
            exact_response_model_verified=False,
            deterministic_fallback_used=False,
        )


def dry_run() -> dict[str, Any]:
    """Exercise the privacy split and compiler-shaped trace with no API calls."""

    bridge = _DryRunBridge()
    journal = TraceJournal()
    ledger = EvidenceLedger()
    config = AdapterConfig(
        tier="small",
        operating_point="dry-run",
        dataset_revision="synthetic-dry-run",
        dataset_manifest_sha256="0" * 64,
        bridge_factory="synthetic:dry_run",
        bridge_params={},
    )
    adapter = SwarmQueryAdapter(bridge, journal, config)
    adapter.insert({"id": "trajectory-fixture", "public": True})
    context = adapter.query(
        "How should the local workflow proceed?",
        opaque_invocation_id="opaque-run-local-handle",
    )
    prompt_row = {
        "question_id": "question-fixture",
        "query_invocation_id": "opaque-run-local-handle",
        "memory_context": context,
        "memory_query_duration_seconds": 1.0,
        "memory_context_original_token_count": 11,
        "memory_context_token_count": 11,
        "memory_context_was_truncated": False,
        "memory_post_query_metadata": adapter.post_query_metadata("opaque-run-local-handle"),
    }
    bound = bind_official_prompt_row(
        prompt_row,
        journal=journal,
        ledger=ledger,
        tier="small",
        operating_point="dry-run",
        domain="web",
    )
    bridge.close()
    evidence = ledger.snapshot()[0]
    serialized = json.dumps(evidence.operations, sort_keys=True)
    if "canonical-seed" in serialized or "opaque-run-local-handle" in serialized:
        raise LongMemEvalV2AdapterError("dry-run trace leaked a raw run-local identifier")
    return {
        "claim_status": "dry-run fixture only; not leaderboard or SOTA evidence",
        "model_api_calls": 0,
        "bridge_calls": [name for name, _ in bridge.calls],
        "question_id_visible_to_bridge": False,
        "trace_bound_in_official_metadata": (
            bound["memory_post_query_metadata"].get("swarmbrain_operation_trace_sha256")
            == evidence.trace_sha256
        ),
        "query": evidence.sidecar_row(unanswered=False),
    }


__all__ = [
    "PreflightResult",
    "dataset_manifest",
    "dry_run",
    "execute_official_run",
    "load_bridge_factory",
    "preflight_official_environment",
]
