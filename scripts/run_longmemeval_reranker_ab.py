#!/usr/bin/env python3
"""Run the pinned LongMemEval-S fixed-fusion versus learned-reranker A/B.

This executable does not download or serve a model. It invokes one explicitly
pinned local JSONL deployment and, for the canonical semantic run, the existing
OpenAI-compatible embedding boundary. Secrets are accepted only from the
process environment; provider stderr is discarded by the local adapter.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from _longmemeval_common import (
    LONGMEMEVAL_S_SHA256,
    configure_embeddings,
    default_longmemeval_path,
    ensure_longmemeval,
    make_provider,
    observed_embedding_call_accounting,
    reset_embedding_call_accounting,
    retrieve_question,
)
from benchmarks.integrations.longmemeval_reranker import (
    LongMemEvalRerankerEvidenceError,
    build_run_manifest,
    build_trace_row,
)
from benchmarks.integrations.longmemeval_reranker.evidence import (
    canonical_json,
    canonical_policy,
    implementation_fingerprint,
    sha256_bytes,
)
from benchmarks.integrations.longmemeval_reranker.report import (
    EXPECTED_QUESTION_COUNT,
    _load_dataset,
    _validate_source_retrieval,
    _validate_trace_rows,
)

from swarmbrain.adapters.reranking.local_jsonl import (
    LOCAL_RERANKER_ENV_ALLOWLIST,
    QWEN3_RERANKER_8B,
    LocalJsonlLearnedReranker,
    LocalJsonlRerankerUnavailable,
)
from swarmbrain.domain.reranking import (
    LearnedRerankerIdentity,
    LearnedRerankPolicy,
    LearnedRerankTrace,
)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_SHA256_CHARS = frozenset("0123456789abcdef")


class LongMemEvalRerankerRunError(LongMemEvalRerankerEvidenceError):
    """The online A/B runner could not produce admissible raw evidence."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise LongMemEvalRerankerRunError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise LongMemEvalRerankerRunError(f"non-finite JSON number {value!r} is forbidden")


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, LongMemEvalRerankerRunError) as exc:
        raise LongMemEvalRerankerRunError(f"{label} is not strict JSON: {exc}") from exc


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json(raw, label=label)
    if not isinstance(value, dict):
        raise LongMemEvalRerankerRunError(f"{label} must contain one JSON object")
    return value


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in _SHA256_CHARS for character in value):
        raise LongMemEvalRerankerRunError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str | Path, *, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise LongMemEvalRerankerRunError(f"{label} cannot be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise LongMemEvalRerankerRunError(f"{label} is missing") from exc
    if not resolved.is_file():
        raise LongMemEvalRerankerRunError(f"{label} must be a regular file")
    return resolved


def _repository_input(path: str | Path, *, root: Path, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise LongMemEvalRerankerRunError(
            f"{label} must be a repository-local relative path without '..'"
        )
    current = root.resolve()
    for part in supplied.parts:
        current /= part
        if current.is_symlink():
            raise LongMemEvalRerankerRunError(f"{label} cannot traverse symbolic links")
    resolved = _regular_file(root / supplied, label=label)
    if not resolved.is_relative_to(root.resolve()):
        raise LongMemEvalRerankerRunError(f"{label} must remain inside the repository")
    return resolved


def _repository_output(path: str | Path, *, root: Path, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_absolute() or not supplied.name or ".." in supplied.parts:
        raise LongMemEvalRerankerRunError(
            f"{label} must be a repository-local relative path without '..'"
        )
    root = root.resolve()
    parent = root
    for part in supplied.parts[:-1]:
        parent /= part
        if parent.is_symlink():
            raise LongMemEvalRerankerRunError(f"{label} cannot traverse symbolic links")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.resolve().is_relative_to(root):
        raise LongMemEvalRerankerRunError(f"{label} must remain inside the repository")
    output = parent / supplied.name
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise LongMemEvalRerankerRunError(f"{label} must be a regular file or absent")
    return output


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, raw: bytes) -> None:
    """Durably replace one checkpoint, never exposing a partial JSONL row."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise LongMemEvalRerankerRunError("refusing to replace a trace symlink")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_create(path: Path, raw: bytes) -> None:
    """Durably publish a completed manifest without replacing an existing one."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite completed reranker run: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite completed reranker run: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _trace_ids(row: dict[str, Any], *, label: str) -> tuple[str, str]:
    try:
        trace = LearnedRerankTrace.model_validate(row["learned"]["trace"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise LongMemEvalRerankerRunError(f"{label} has no valid learned trace") from exc
    if trace.request_id is None or trace.provider_request_id is None:
        raise LongMemEvalRerankerRunError(f"{label} has no complete request receipt")
    if trace.request_id == trace.provider_request_id:
        raise LongMemEvalRerankerRunError(
            f"{label} provider request ID equals the client request ID"
        )
    return trace.request_id, trace.provider_request_id


def _load_resume_prefix(
    path: Path,
    *,
    records: list[dict[str, Any]],
    source_cases: list[tuple[dict[str, Any], list[str]]],
    identity: LearnedRerankerIdentity,
    policy: LearnedRerankPolicy,
) -> tuple[bytes, list[dict[str, Any]], set[str], set[str]]:
    if not path.exists():
        return b"", [], set(), set()
    if path.is_symlink() or not path.is_file():
        raise LongMemEvalRerankerRunError("trace checkpoint must be a regular file")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise LongMemEvalRerankerRunError(
            "trace checkpoint must be non-empty and newline terminated"
        )
    if b"\r" in raw:
        raise LongMemEvalRerankerRunError(
            "trace checkpoint must use canonical LF-only JSONL separators"
        )
    rows: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    provider_request_ids: set[str] = set()
    # The final byte is the required record terminator. Splitting only on LF
    # (instead of ``bytes.splitlines``) prevents VT/FF/other Unicode-style line
    # boundaries from being accepted as JSONL separators.
    for index, line in enumerate(raw[:-1].split(b"\n")):
        if index >= len(records):
            raise LongMemEvalRerankerRunError("trace checkpoint exceeds dataset coverage")
        row = _strict_object(line, label=f"trace checkpoint line {index + 1}")
        canonical_line = canonical_json(row).encode("utf-8")
        if line != canonical_line:
            raise LongMemEvalRerankerRunError(
                f"trace checkpoint line {index + 1} is not canonical JSON"
            )
        source_case, _candidate_ids = source_cases[index]
        try:
            trace = LearnedRerankTrace.model_validate(row["learned"]["trace"])
        except (KeyError, TypeError, ValidationError) as exc:
            raise LongMemEvalRerankerRunError(
                f"trace checkpoint line {index + 1} has no valid learned trace"
            ) from exc
        if trace.identity != identity or trace.policy != policy:
            raise LongMemEvalRerankerRunError(
                f"trace checkpoint line {index + 1} uses a different identity or policy"
            )
        rebuilt = build_trace_row(
            case_index=index,
            record=records[index],
            source_case=source_case,
            policy=policy,
            learned_trace=trace,
        )
        if row != rebuilt:
            raise LongMemEvalRerankerRunError(
                f"trace checkpoint line {index + 1} is not the canonical dataset/source row"
            )
        request_id, provider_request_id = _trace_ids(
            row, label=f"trace checkpoint line {index + 1}"
        )
        if request_id in request_ids:
            raise LongMemEvalRerankerRunError("trace checkpoint reuses a client request ID")
        if provider_request_id in provider_request_ids:
            raise LongMemEvalRerankerRunError("trace checkpoint reuses a provider request ID")
        request_ids.add(request_id)
        provider_request_ids.add(provider_request_id)
        rows.append(row)
    try:
        _validate_trace_rows(
            rows,
            records=records[: len(rows)],
            source_cases=source_cases[: len(rows)],
            identity=identity,
            policy=policy,
        )
    except LongMemEvalRerankerEvidenceError as exc:
        raise LongMemEvalRerankerRunError(
            f"trace checkpoint fails authoritative compiler replay: {exc}"
        ) from exc
    return raw, rows, request_ids, provider_request_ids


def _project_fused_session_keys(retrieved: Any) -> list[str]:
    output: list[str] = []
    for candidate in retrieved.execution.trace.fused_candidates:
        key = retrieved.key_by_memory_id.get(candidate.canonical_id)
        if key is None:
            raise LongMemEvalRerankerRunError(
                "live fused ranking contains a candidate outside the question haystack"
            )
        if key in output:
            raise LongMemEvalRerankerRunError("live fused ranking repeats a session key")
        output.append(key)
    return output


def _validate_live_pairing(
    retrieved: Any,
    *,
    source_candidate_ids: list[str],
) -> LearnedRerankTrace:
    if retrieved.execution.trace.degraded_lanes:
        raise LongMemEvalRerankerRunError("live baseline contains a degraded retrieval lane")
    live_fused = _project_fused_session_keys(retrieved)
    if live_fused[: len(source_candidate_ids)] != source_candidate_ids:
        raise LongMemEvalRerankerRunError(
            "live pre-learned fused session-key order differs from the source case"
        )
    raw_trace = retrieved.execution.trace.learned_rerank
    if raw_trace is None:
        raise LongMemEvalRerankerRunError("live retrieval emitted no learned-rerank trace")
    if not raw_trace.applied or raw_trace.degraded:
        reason = raw_trace.degradation_reason or "incomplete learned-rerank result"
        raise LongMemEvalRerankerRunError(f"live learned reranker failed closed: {reason}")
    mapped_inputs: list[str] = []
    for memory_id in raw_trace.input_ids:
        key = retrieved.key_by_memory_id.get(memory_id)
        if key is None:
            raise LongMemEvalRerankerRunError(
                "learned reranker input contains a candidate outside the question haystack"
            )
        mapped_inputs.append(key)
    if mapped_inputs != source_candidate_ids:
        raise LongMemEvalRerankerRunError(
            "live learned-reranker input order differs from the source fused session-key order"
        )
    return raw_trace


def _reconcile_current_process_accounting(
    provider: LocalJsonlLearnedReranker,
    new_rows: list[dict[str, Any]],
) -> None:
    candidates = 0
    input_tokens = 0
    output_tokens = 0
    for row in new_rows:
        trace = LearnedRerankTrace.model_validate(row["learned"]["trace"])
        assert trace.usage is not None
        candidates += len(trace.input_ids)
        input_tokens += trace.usage.input_tokens
        output_tokens += trace.usage.output_tokens
    expected = {
        "attempts": len(new_rows),
        "successful_requests": len(new_rows),
        "failed_requests": 0,
        "scored_candidates": candidates,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retained_provider_request_ids": len(new_rows),
    }
    if provider.call_accounting != expected:
        raise LongMemEvalRerankerRunError(
            "local reranker call accounting does not reconcile with newly accepted traces"
        )


def _reconcile_embedding_accounting(records: list[dict[str, Any]]) -> None:
    observed = observed_embedding_call_accounting()
    expected = {
        "document_inputs": sum(len(record["haystack_sessions"]) for record in records),
        "document_batch_calls": len(records),
        "query_calls": len(records),
        "successful_http_calls": 2 * len(records),
    }
    if observed is None:
        raise LongMemEvalRerankerRunError(
            "semantic replay did not expose provider-observed embedding accounting"
        )
    for field, wanted in expected.items():
        if observed.get(field) != wanted:
            raise LongMemEvalRerankerRunError(
                f"semantic embedding accounting {field} does not reconcile"
            )
    attempts = observed.get("http_attempts")
    if not isinstance(attempts, int) or attempts < expected["successful_http_calls"]:
        raise LongMemEvalRerankerRunError(
            "semantic embedding HTTP attempts do not cover every successful call"
        )


async def run_longmemeval_reranker_ab(
    *,
    dataset_path: str | Path,
    source_retrieval_path: str | Path,
    traces_path: str | Path,
    run_path: str | Path,
    executable_path: str | Path,
    executable_sha256: str,
    deployment_manifest_path: str | Path,
    deployment_manifest_sha256: str,
    required_model: str,
    required_revision: str,
    reranker_environment: dict[str, str] | None = None,
    artifact_root: Path = REPO_ROOT,
    code_root: Path = REPO_ROOT,
    expected_dataset_sha256: str = LONGMEMEVAL_S_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    require_publishable_source: bool = True,
    require_current_source_implementation: bool = True,
    use_dense: bool = True,
    embeddings_base_url: str | None = None,
    embeddings_model: str | None = None,
    embeddings_api_key: str | None = None,
) -> dict[str, Any]:
    """Execute or resume one exact paired run and atomically publish its manifest."""

    artifact_root = artifact_root.resolve()
    code_root = code_root.resolve()
    implementation_at_start = implementation_fingerprint(code_root)
    source_path = _repository_input(
        source_retrieval_path,
        root=artifact_root,
        label="source retrieval artifact",
    )
    trace_output = _repository_output(traces_path, root=artifact_root, label="trace output")
    run_output = _repository_output(run_path, root=artifact_root, label="run output")
    dataset = _regular_file(dataset_path, label="dataset")
    executable = _regular_file(executable_path, label="reranker executable")
    deployment_manifest = _regular_file(
        deployment_manifest_path,
        label="reranker deployment manifest",
    )
    protected = {dataset, source_path, executable, deployment_manifest}
    if trace_output.resolve() in protected or run_output.resolve() in protected:
        raise LongMemEvalRerankerRunError("an output path aliases an immutable input artifact")
    if trace_output.resolve() == run_output.resolve():
        raise LongMemEvalRerankerRunError("trace and run outputs must be different files")
    if run_output.exists() or run_output.is_symlink():
        raise FileExistsError(f"refusing to overwrite completed reranker run: {run_output}")

    _require_sha256(executable_sha256, label="executable_sha256")
    _require_sha256(deployment_manifest_sha256, label="deployment_manifest_sha256")
    dataset_raw = dataset.read_bytes()
    dataset_digest = sha256_bytes(dataset_raw)
    records = _load_dataset(
        dataset_raw,
        expected_sha256=expected_dataset_sha256,
        expected_questions=expected_question_count,
    )
    del dataset_raw
    source_raw = source_path.read_bytes()
    source_digest = sha256_bytes(source_raw)
    source = _strict_object(source_raw, label="source retrieval artifact")
    del source_raw
    source_cases = _validate_source_retrieval(
        source,
        records,
        dataset_sha256=expected_dataset_sha256,
        require_publishable=require_publishable_source,
        require_current_implementation=require_current_source_implementation,
    )

    provider: LocalJsonlLearnedReranker | None = None
    embedding_provider: Any | None = None
    try:
        provider = LocalJsonlLearnedReranker(
            executable_path=executable,
            executable_sha256=executable_sha256,
            deployment_manifest_path=deployment_manifest,
            deployment_manifest_sha256=deployment_manifest_sha256,
            required_model=required_model,
            required_revision=required_revision,
            environment=reranker_environment,
        )
        policy = canonical_policy(provider.identity)
        if use_dense:
            if not embeddings_base_url or not embeddings_model:
                raise LongMemEvalRerankerRunError(
                    "semantic replay requires an embedding base URL and pinned model ID"
                )
            parsed_url = urlsplit(embeddings_base_url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise LongMemEvalRerankerRunError("embedding base URL must be absolute HTTP(S)")
            if parsed_url.username is not None or parsed_url.password is not None:
                raise LongMemEvalRerankerRunError(
                    "embedding base URL cannot contain credentials; use the API-key environment"
                )
            source_embedding = source.get("embedding")
            if (
                not isinstance(source_embedding, dict)
                or source_embedding.get("model") != embeddings_model
            ):
                raise LongMemEvalRerankerRunError(
                    "embedding model does not match the source retrieval artifact"
                )
            configure_embeddings(
                "openai",
                base_url=embeddings_base_url,
                model_id=embeddings_model,
                api_key=embeddings_api_key,
            )
            embedding_provider = make_provider()
            reset_embedding_call_accounting()

        trace_raw, trace_rows, request_ids, provider_request_ids = _load_resume_prefix(
            trace_output,
            records=records,
            source_cases=source_cases,
            identity=provider.identity,
            policy=policy,
        )
        prefix_count = len(trace_rows)
        for index in range(prefix_count, len(records)):
            record = records[index]
            source_case, source_candidate_ids = source_cases[index]
            retrieved = await retrieve_question(
                record,
                limit=int(source["recall_limit"]),
                min_score=0.0,
                use_dense=use_dense,
                temporal_query_routing=False,
                learned_reranker=provider,
                learned_rerank_policy=policy,
            )
            learned_trace = _validate_live_pairing(
                retrieved,
                source_candidate_ids=source_candidate_ids,
            )
            row = build_trace_row(
                case_index=index,
                record=record,
                source_case=source_case,
                policy=policy,
                learned_trace=learned_trace,
            )
            request_id, provider_request_id = _trace_ids(row, label=f"case {index}")
            if request_id in request_ids:
                raise LongMemEvalRerankerRunError("learned arm reused a client request ID")
            if provider_request_id in provider_request_ids:
                raise LongMemEvalRerankerRunError("learned arm reused a provider request ID")
            request_ids.add(request_id)
            provider_request_ids.add(provider_request_id)
            line = canonical_json(row).encode("utf-8") + b"\n"
            trace_raw += line
            _atomic_replace(trace_output, trace_raw)
            trace_rows.append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(
                    f"  LongMemEval reranker A/B: {index + 1}/{len(records)} questions",
                    file=sys.stderr,
                )

        _reconcile_current_process_accounting(provider, trace_rows[prefix_count:])
        if use_dense:
            _reconcile_embedding_accounting(records[prefix_count:])
        if _sha256_file(dataset) != dataset_digest:
            raise LongMemEvalRerankerRunError("pinned dataset changed during execution")
        if _sha256_file(source_path) != source_digest:
            raise LongMemEvalRerankerRunError("source retrieval artifact changed during execution")
        try:
            _paired_cases, compiler_accounting, _latencies = _validate_trace_rows(
                trace_rows,
                records=records,
                source_cases=source_cases,
                identity=provider.identity,
                policy=policy,
            )
        except LongMemEvalRerankerEvidenceError as exc:
            raise LongMemEvalRerankerRunError(
                f"completed trace JSONL fails authoritative compiler replay: {exc}"
            ) from exc
        manifest = build_run_manifest(
            created_at_utc=datetime.now(UTC).isoformat(),
            dataset_sha256=dataset_digest,
            question_count=len(records),
            source_retrieval_path=source_path,
            traces_path=trace_output,
            identity=provider.identity,
            policy=policy,
            artifact_root=artifact_root,
            code_root=code_root,
            trace_rows=trace_rows,
        )
        if manifest["call_accounting"] != compiler_accounting:
            raise LongMemEvalRerankerRunError(
                "run manifest accounting differs from authoritative compiler replay"
            )
        if manifest["implementation"] != implementation_at_start:
            raise LongMemEvalRerankerRunError("reranker implementation changed during execution")
        run_raw = (
            json.dumps(
                manifest,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        _atomic_create(run_output, run_raw)
        return manifest
    finally:
        try:
            if provider is not None:
                await provider.close()
        finally:
            close = getattr(embedding_provider, "close", None)
            if callable(close):
                await close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_longmemeval_path())
    parser.add_argument("--lme-download", action="store_true")
    parser.add_argument("--source-retrieval", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reranker-executable", type=Path, required=True)
    parser.add_argument("--reranker-executable-sha256", required=True)
    parser.add_argument("--reranker-manifest", type=Path, required=True)
    parser.add_argument("--reranker-manifest-sha256", required=True)
    parser.add_argument("--reranker-model", default=QWEN3_RERANKER_8B)
    parser.add_argument("--reranker-revision", required=True)
    parser.add_argument("--embeddings-base-url", required=True)
    parser.add_argument("--embeddings-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--reranker-env",
        action="append",
        choices=sorted(LOCAL_RERANKER_ENV_ALLOWLIST),
        default=[],
        metavar="NAME",
        help="inherit one allowlisted environment variable by name; values are never logged",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        dataset = ensure_longmemeval(args.dataset, download=args.lme_download)
        missing_environment = [name for name in args.reranker_env if name not in os.environ]
        if missing_environment:
            raise LongMemEvalRerankerRunError(
                "requested reranker environment variables are not defined: "
                + ", ".join(missing_environment)
            )
        environment = {name: os.environ[name] for name in args.reranker_env}
        asyncio.run(
            run_longmemeval_reranker_ab(
                dataset_path=dataset,
                source_retrieval_path=args.source_retrieval,
                traces_path=args.traces,
                run_path=args.run,
                executable_path=args.reranker_executable,
                executable_sha256=args.reranker_executable_sha256,
                deployment_manifest_path=args.reranker_manifest,
                deployment_manifest_sha256=args.reranker_manifest_sha256,
                required_model=args.reranker_model,
                required_revision=args.reranker_revision,
                reranker_environment=environment,
                embeddings_base_url=args.embeddings_base_url,
                embeddings_model=args.embeddings_model,
                embeddings_api_key=os.getenv("SWARMBRAIN_EMBEDDINGS_API_KEY"),
            )
        )
    except (
        FileExistsError,
        LongMemEvalRerankerRunError,
        LocalJsonlRerankerUnavailable,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"LongMemEval reranker A/B failed closed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
