"""Pinned persistent local-JSONL adapter for learned rerankers.

The executable is a deployment-owned model runner.  It reads one canonical
``LearnedRerankRequest`` JSON object per line from stdin and writes one strict
``LearnedRerankResult`` JSON object per line to stdout.  There is no shell and
no network behavior in this adapter.  A single persistent process avoids
reloading an 8B checkpoint per query while a lock keeps the request/response
stream unambiguous.

The deployment manifest enumerates the ordered scorer components, their
immutable model/tokenizer revisions and artifact digests, and their fusion
weights.  Its raw-byte SHA-256 and the executable SHA-256 are operator-pinned;
both are rechecked before each exchange.  This supports either a single
Qwen3-Reranker-8B cross-encoder or a composite CE+ColBERT scorer while exposing
one normalized score through the provider-neutral port.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from swarmbrain.domain.reranking import (
    LearnedRerankerComponent,
    LearnedRerankerIdentity,
    LearnedRerankRequest,
    LearnedRerankResult,
    canonical_rerank_json,
    learned_reranker_model_bundle_payload,
    learned_reranker_tokenizer_bundle_payload,
    rerank_sha256_json,
)
from swarmbrain.retrieval.learned_reranking import (
    LearnedRerankValidationError,
    validate_learned_rerank_result,
)

LOCAL_RERANKER_MANIFEST_SCHEMA = "swarmbrain.local-reranker-manifest.v1"
LOCAL_JSONL_PROTOCOL_REVISION = "swarmbrain.learned-reranker.local-jsonl.v1"
LOCAL_JSONL_PROVIDER = "local-jsonl"
QWEN3_RERANKER_8B = "Qwen/Qwen3-Reranker-8B"

_MAX_MANIFEST_BYTES = 65_536
_MAX_ENVIRONMENT_ENTRIES = 128
_MAX_ENVIRONMENT_BYTES = 65_536
_PROVIDER_REQUEST_ID_WINDOW = 65_536
LOCAL_RERANKER_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "OMP_NUM_THREADS",
        "PYTHONPATH",
        "TOKENIZERS_PARALLELISM",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "TMPDIR",
        "XDG_CACHE_HOME",
    }
)


@dataclass(frozen=True, slots=True)
class _ArtifactPin:
    path: Path
    manifest_path: str
    sha256: str
    size_bytes: int


class LocalJsonlRerankerUnavailable(RuntimeError):
    """The pinned process was unavailable or violated its strict protocol."""


class LocalJsonlLearnedReranker:
    """Persistent, bounded local process adapter for an immutable scorer."""

    def __init__(
        self,
        *,
        executable_path: str | Path,
        executable_sha256: str,
        deployment_manifest_path: str | Path,
        deployment_manifest_sha256: str,
        required_model: str,
        required_revision: str,
        timeout_seconds: float = 20.0,
        max_request_bytes: int = 8_388_608,
        max_response_bytes: int = 1_048_576,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if type(timeout_seconds) not in {int, float} or not 0.05 <= timeout_seconds <= 60.0:
            raise ValueError("timeout_seconds must be in [0.05, 60]")
        if type(max_request_bytes) is not int or not 1_024 <= max_request_bytes <= 67_108_864:
            raise ValueError("max_request_bytes must be an integer in [1024, 67108864]")
        if type(max_response_bytes) is not int or not 1_024 <= max_response_bytes <= 8_388_608:
            raise ValueError("max_response_bytes must be an integer in [1024, 8388608]")
        if not isinstance(required_model, str) or not required_model.strip():
            raise ValueError("required_model must be a non-empty string")
        if len(required_model) > 255:
            raise ValueError("required_model exceeds 255 characters")
        _require_digest(executable_sha256, "executable_sha256")
        _require_digest(deployment_manifest_sha256, "deployment_manifest_sha256")
        self._executable_path = _regular_file(executable_path, "executable_path")
        if not os.access(self._executable_path, os.X_OK):
            raise ValueError("executable_path must be executable")
        self._manifest_path = _regular_file(
            deployment_manifest_path,
            "deployment_manifest_path",
        )
        self._deployment_root = self._manifest_path.parent.resolve(strict=True)
        self._executable_sha256 = executable_sha256
        self._manifest_sha256 = deployment_manifest_sha256
        self._timeout_seconds = float(timeout_seconds)
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._environment = _validated_environment(environment)
        self._identity, self._artifact_pins = self._load_identity(
            required_model=required_model,
            required_revision=required_revision,
        )
        self._artifact_fingerprints: dict[Path, tuple[int, int, int, int]] = {}
        self._verify_pins(full_artifacts=True)
        self._has_launched_process = False
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._attempts = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._scored_candidates = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._provider_request_ids: set[str] = set()
        self._provider_request_order: deque[str] = deque()

    @property
    def identity(self) -> LearnedRerankerIdentity:
        return self._identity

    @property
    def call_accounting(self) -> dict[str, int]:
        return {
            "attempts": self._attempts,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "scored_candidates": self._scored_candidates,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "retained_provider_request_ids": len(self._provider_request_ids),
        }

    async def rerank(self, request: LearnedRerankRequest) -> LearnedRerankResult:
        if request.identity != self._identity:
            raise LocalJsonlRerankerUnavailable(
                "rerank request identity does not match the pinned local deployment"
            )
        request_line = (
            canonical_rerank_json(request.model_dump(mode="json")).encode("utf-8") + b"\n"
        )
        if len(request_line) > self._max_request_bytes:
            raise LocalJsonlRerankerUnavailable("rerank request exceeds adapter byte bound")
        async with self._lock:
            self._require_open()
            self._attempts += 1
            try:
                self._verify_pins(full_artifacts=False)
                process = await self._ensure_process()
                response_line = await asyncio.wait_for(
                    self._exchange(process, request_line),
                    timeout=self._timeout_seconds,
                )
                result = self._parse_response(response_line)
                validate_learned_rerank_result(
                    request,
                    result,
                    expected_identity=self._identity,
                )
                provider_request_id = result.receipt.provider_request_id
                if provider_request_id == request.request_id:
                    raise LocalJsonlRerankerUnavailable(
                        "provider_request_id must be distinct from the client request_id"
                    )
                if provider_request_id in self._provider_request_ids:
                    raise LocalJsonlRerankerUnavailable("provider_request_id was reused")
                self._remember_provider_request_id(provider_request_id)
            except asyncio.CancelledError:
                self._failed_requests += 1
                await self._stop_process()
                raise
            except TimeoutError as exc:
                self._failed_requests += 1
                await self._stop_process()
                raise LocalJsonlRerankerUnavailable("local reranker timed out") from exc
            except LearnedRerankValidationError as exc:
                self._failed_requests += 1
                await self._stop_process()
                raise LocalJsonlRerankerUnavailable(
                    "local reranker receipt does not match the exact input IDs or accounting"
                ) from exc
            except LocalJsonlRerankerUnavailable:
                self._failed_requests += 1
                await self._stop_process()
                raise
            except Exception as exc:
                self._failed_requests += 1
                await self._stop_process()
                raise LocalJsonlRerankerUnavailable(
                    f"local reranker failed: {type(exc).__name__}"
                ) from exc
            self._successful_requests += 1
            self._scored_candidates += len(result.scores)
            self._input_tokens += result.receipt.usage.input_tokens
            self._output_tokens += result.receipt.usage.output_tokens
            return result

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            await self._stop_process()

    async def __aenter__(self) -> LocalJsonlLearnedReranker:
        self._require_open()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        # Constructor startup performs the first full hash.  A later process
        # restart re-hashes all artifacts; during a process lifetime cheap
        # inode/size/mtime checks before each request detect replacement without
        # reading multi-gigabyte shards twice before the first query.
        if self._has_launched_process:
            self._verify_pins(full_artifacts=True)
        self._process = await asyncio.create_subprocess_exec(
            str(self._executable_path),
            "--manifest",
            str(self._manifest_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Provider stderr may contain memory text or secrets and is not
            # evidence.  Discarding it also gives the channel a hard zero-byte
            # storage bound and prevents pipe backpressure.
            stderr=asyncio.subprocess.DEVNULL,
            env=self._environment,
            limit=self._max_response_bytes + 1,
        )
        self._has_launched_process = True
        return self._process

    async def _exchange(
        self,
        process: asyncio.subprocess.Process,
        request_line: bytes,
    ) -> bytes:
        if process.stdin is None or process.stdout is None:
            raise LocalJsonlRerankerUnavailable("local reranker pipes are unavailable")
        process.stdin.write(request_line)
        await process.stdin.drain()
        try:
            response = await process.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise LocalJsonlRerankerUnavailable(
                "local reranker response exceeds adapter byte bound"
            ) from exc
        if not response:
            raise LocalJsonlRerankerUnavailable("local reranker exited without a response")
        if len(response) > self._max_response_bytes:
            raise LocalJsonlRerankerUnavailable(
                "local reranker response exceeds adapter byte bound"
            )
        if not response.endswith(b"\n"):
            raise LocalJsonlRerankerUnavailable("local reranker response is not one JSONL record")
        return response[:-1]

    @staticmethod
    def _parse_response(response: bytes) -> LearnedRerankResult:
        try:
            payload = json.loads(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocalJsonlRerankerUnavailable(
                "local reranker returned invalid UTF-8 JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise LocalJsonlRerankerUnavailable("local reranker response must be an object")
        try:
            return LearnedRerankResult.model_validate(payload)
        except ValidationError as exc:
            raise LocalJsonlRerankerUnavailable(
                "local reranker returned an invalid receipt"
            ) from exc

    async def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except TimeoutError:
            process.kill()
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)

    def _remember_provider_request_id(self, value: str) -> None:
        self._provider_request_ids.add(value)
        self._provider_request_order.append(value)
        if len(self._provider_request_order) <= _PROVIDER_REQUEST_ID_WINDOW:
            return
        expired = self._provider_request_order.popleft()
        self._provider_request_ids.discard(expired)

    def _require_open(self) -> None:
        if self._closed:
            raise LocalJsonlRerankerUnavailable("local reranker adapter is closed")

    def _verify_pins(self, *, full_artifacts: bool) -> None:
        try:
            if _sha256_file(self._executable_path) != self._executable_sha256:
                raise LocalJsonlRerankerUnavailable("local reranker executable digest changed")
            raw = _bounded_read(self._manifest_path, _MAX_MANIFEST_BYTES)
            if hashlib.sha256(raw).hexdigest() != self._manifest_sha256:
                raise LocalJsonlRerankerUnavailable("local reranker deployment manifest changed")
            for pin in self._artifact_pins:
                self._verify_artifact_route(pin)
                stat = pin.path.stat()
                fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if stat.st_size != pin.size_bytes:
                    raise LocalJsonlRerankerUnavailable(
                        f"pinned artifact size changed: {pin.manifest_path}"
                    )
                previous = self._artifact_fingerprints.get(pin.path)
                if previous is not None and fingerprint != previous:
                    raise LocalJsonlRerankerUnavailable(
                        f"pinned artifact file identity changed: {pin.manifest_path}"
                    )
                if full_artifacts and _sha256_file(pin.path) != pin.sha256:
                    raise LocalJsonlRerankerUnavailable(
                        f"pinned artifact digest mismatch: {pin.manifest_path}"
                    )
                self._artifact_fingerprints[pin.path] = fingerprint
        except LocalJsonlRerankerUnavailable:
            raise
        except (OSError, ValueError) as exc:
            raise LocalJsonlRerankerUnavailable(
                f"local reranker pin verification failed: {type(exc).__name__}"
            ) from exc

    def _load_identity(
        self,
        *,
        required_model: str,
        required_revision: str,
    ) -> tuple[LearnedRerankerIdentity, tuple[_ArtifactPin, ...]]:
        raw = _bounded_read(self._manifest_path, _MAX_MANIFEST_BYTES)
        if hashlib.sha256(raw).hexdigest() != self._manifest_sha256:
            raise ValueError("deployment_manifest_sha256 does not match manifest bytes")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("deployment manifest must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "model",
            "revision",
            "components",
        }:
            raise ValueError("deployment manifest has an invalid top-level shape")
        if payload["schema"] != LOCAL_RERANKER_MANIFEST_SCHEMA:
            raise ValueError("deployment manifest schema is unsupported")
        if payload["model"] != required_model:
            raise ValueError("deployment manifest model does not match required_model")
        if payload["revision"] != required_revision:
            raise ValueError("deployment manifest revision does not match required_revision")
        raw_components = payload["components"]
        if not isinstance(raw_components, list):
            raise ValueError("deployment manifest components must be a list")
        expected_component_fields = {
            "role",
            "model",
            "revision",
            "model_artifacts",
            "tokenizer_revision",
            "tokenizer_artifacts",
            "weight",
        }
        if any(
            not isinstance(item, dict) or set(item) != expected_component_fields
            for item in raw_components
        ):
            raise ValueError("deployment manifest component shape is invalid")
        components: list[LearnedRerankerComponent] = []
        artifact_pins: dict[Path, _ArtifactPin] = {}
        for item in raw_components:
            assert isinstance(item, dict)
            model_artifacts = self._manifest_artifacts(
                item["model_artifacts"],
                label=f"{item.get('role', 'component')} model_artifacts",
            )
            tokenizer_artifacts = self._manifest_artifacts(
                item["tokenizer_artifacts"],
                label=f"{item.get('role', 'component')} tokenizer_artifacts",
            )
            for pin in (*model_artifacts, *tokenizer_artifacts):
                existing = artifact_pins.get(pin.path)
                if existing is not None and existing != pin:
                    raise ValueError("one artifact path has conflicting manifest pins")
                artifact_pins[pin.path] = pin
            component_payload = {
                key: value
                for key, value in item.items()
                if key not in {"model_artifacts", "tokenizer_artifacts"}
            }
            component_payload["model_artifact_sha256"] = rerank_sha256_json(
                [_artifact_pin_payload(pin) for pin in model_artifacts]
            )
            component_payload["tokenizer_artifact_sha256"] = rerank_sha256_json(
                [_artifact_pin_payload(pin) for pin in tokenizer_artifacts]
            )
            try:
                components.append(LearnedRerankerComponent.model_validate(component_payload))
            except ValidationError as exc:
                raise ValueError("deployment manifest component identity is invalid") from exc
        component_tuple = tuple(components)
        identity = LearnedRerankerIdentity(
            provider=LOCAL_JSONL_PROVIDER,
            model=required_model,
            revision=required_revision,
            components=component_tuple,
            model_artifact_sha256=rerank_sha256_json(
                learned_reranker_model_bundle_payload(component_tuple)
            ),
            tokenizer_artifact_sha256=rerank_sha256_json(
                learned_reranker_tokenizer_bundle_payload(component_tuple)
            ),
            deployment_manifest_sha256=self._manifest_sha256,
            adapter_artifact_sha256=self._executable_sha256,
            runtime_environment_sha256=rerank_sha256_json(self._environment),
            protocol_revision=LOCAL_JSONL_PROTOCOL_REVISION,
        )
        return identity, tuple(artifact_pins.values())

    def _manifest_artifacts(self, value: object, *, label: str) -> tuple[_ArtifactPin, ...]:
        if not isinstance(value, list) or not 1 <= len(value) <= 256:
            raise ValueError(f"{label} must contain between 1 and 256 files")
        result: list[_ArtifactPin] = []
        seen: set[str] = set()
        for raw in value:
            if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
                raise ValueError(f"{label} contains an invalid artifact entry")
            manifest_path = raw["path"]
            sha256 = raw["sha256"]
            size_bytes = raw["size_bytes"]
            if (
                not isinstance(manifest_path, str)
                or not manifest_path
                or len(manifest_path) > 1_024
                or Path(manifest_path).is_absolute()
                or ".." in Path(manifest_path).parts
            ):
                raise ValueError(f"{label} contains an unsafe relative path")
            if manifest_path in seen:
                raise ValueError(f"{label} contains duplicate artifact paths")
            seen.add(manifest_path)
            _require_digest(sha256, f"{label} sha256")
            if type(size_bytes) is not int or not 0 <= size_bytes <= 1_099_511_627_776:
                raise ValueError(f"{label} size_bytes is invalid")
            unresolved = self._deployment_root / manifest_path
            _reject_symlink_components(
                self._deployment_root,
                Path(manifest_path),
                label=label,
            )
            path = _regular_file(unresolved, f"{label} path")
            try:
                path.relative_to(self._deployment_root)
            except ValueError as exc:
                raise ValueError(f"{label} artifact escapes the deployment root") from exc
            result.append(
                _ArtifactPin(
                    path=path,
                    manifest_path=manifest_path,
                    sha256=sha256,
                    size_bytes=size_bytes,
                )
            )
        return tuple(result)

    def _verify_artifact_route(self, pin: _ArtifactPin) -> None:
        relative = Path(pin.manifest_path)
        try:
            _reject_symlink_components(self._deployment_root, relative, label="pinned artifact")
        except ValueError as exc:
            raise LocalJsonlRerankerUnavailable(
                f"pinned artifact route contains a symlink: {pin.manifest_path}"
            ) from exc
        resolved = (self._deployment_root / relative).resolve(strict=True)
        if resolved != pin.path:
            raise LocalJsonlRerankerUnavailable(
                f"pinned artifact route changed: {pin.manifest_path}"
            )


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must identify a regular file")
    return path


def _require_digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _bounded_read(path: Path, maximum: int) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(maximum + 1)
    if len(value) > maximum:
        raise ValueError(f"{path.name} exceeds the {maximum}-byte bound")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_environment(environment: Mapping[str, str] | None) -> dict[str, str] | None:
    normalized = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if environment is None:
        return normalized
    if len(environment) > _MAX_ENVIRONMENT_ENTRIES:
        raise ValueError("local reranker environment has too many entries")
    total_bytes = 0
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError("local reranker environment contains an invalid entry")
        if key not in LOCAL_RERANKER_ENV_ALLOWLIST:
            raise ValueError(f"local reranker environment key is not allowlisted: {key}")
        total_bytes += len(key.encode("utf-8")) + len(value.encode("utf-8"))
        normalized[key] = value
    if total_bytes > _MAX_ENVIRONMENT_BYTES:
        raise ValueError("local reranker environment exceeds its byte bound")
    return normalized


def _artifact_pin_payload(pin: _ArtifactPin) -> dict[str, object]:
    return {
        "path": pin.manifest_path,
        "sha256": pin.sha256,
        "size_bytes": pin.size_bytes,
    }


def _reject_symlink_components(root: Path, relative: Path, *, label: str) -> None:
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} paths must not contain symlink components")


__all__ = [
    "LOCAL_JSONL_PROTOCOL_REVISION",
    "LOCAL_JSONL_PROVIDER",
    "LOCAL_RERANKER_ENV_ALLOWLIST",
    "LOCAL_RERANKER_MANIFEST_SCHEMA",
    "QWEN3_RERANKER_8B",
    "LocalJsonlLearnedReranker",
    "LocalJsonlRerankerUnavailable",
]
