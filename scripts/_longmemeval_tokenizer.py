"""Strict local JSONL boundary for exact LongMemEval reader-context token counts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

TOKENIZER_PROTOCOL = "swarmbrain-exact-tokenizer-jsonl-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROVIDER_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}")
_RESPONSE_FIELDS = frozenset(
    {
        "protocol",
        "request_id",
        "provider_request_id",
        "text_sha256",
        "token_count",
        "tokenizer_model",
        "tokenizer_revision",
        "tokenizer_artifact_sha256",
    }
)


class ExactTokenizerError(RuntimeError):
    """The local tokenizer boundary or its response violated the exact-count contract."""


class ExactTokenizer(Protocol):
    """Injectable boundary used by retrieval; production uses ``JsonlExactTokenizer``."""

    def count(self, text: str) -> TokenizerObservation: ...

    @property
    def evidence(self) -> dict[str, Any]: ...


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExactTokenizerError(f"duplicate tokenizer response field {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ExactTokenizerError(f"non-finite tokenizer response number {value!r}")


def _safe_repo_file(path: Path, *, repo_root: Path, label: str) -> tuple[Path, str, bytes]:
    if path.is_absolute() or ".." in path.parts:
        raise ExactTokenizerError(f"{label} must be a repository-local relative path")
    root = repo_root.resolve()
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ExactTokenizerError(f"{label} cannot traverse symbolic links")
    try:
        resolved = (root / path).resolve(strict=True)
    except OSError as exc:
        raise ExactTokenizerError(f"{label} is missing: {path}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ExactTokenizerError(f"{label} must resolve to a regular repository file")
    return resolved, resolved.relative_to(root).as_posix(), resolved.read_bytes()


def _file_identity(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    if not _is_sha256(expected_sha256):
        raise ExactTokenizerError(f"{label} expected SHA-256 must be 64 lowercase hex digits")
    resolved, relative, content = _safe_repo_file(path, repo_root=repo_root, label=label)
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ExactTokenizerError(f"{label} does not match its operator-pinned SHA-256")
    return resolved, {"path": relative, "bytes": len(content), "sha256": digest}


@dataclass(frozen=True, slots=True)
class TokenizerObservation:
    request_id: int
    provider_request_id: str
    response_identity_sha256: str
    text_sha256: str
    token_count: int

    def evidence(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "provider_request_id": self.provider_request_id,
            "response_identity_sha256": self.response_identity_sha256,
            "text_sha256": self.text_sha256,
            "token_count": self.token_count,
        }


class JsonlExactTokenizer:
    """Persistent exact tokenizer process with pinned executable and artifact bytes.

    The executable receives only local identity arguments and JSONL text on
    stdin. Its environment excludes provider credentials. Every response must
    repeat the pinned tokenizer identity and carry a unique provider request ID.
    """

    def __init__(
        self,
        *,
        executable: Path,
        executable_sha256: str,
        artifact: Path,
        artifact_sha256: str,
        model: str,
        revision: str,
        repo_root: Path,
        response_timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ExactTokenizerError("tokenizer model must be non-empty")
        if not isinstance(revision, str) or not revision.strip():
            raise ExactTokenizerError("tokenizer revision must be non-empty")
        if not math.isfinite(response_timeout_seconds) or response_timeout_seconds <= 0:
            raise ExactTokenizerError("tokenizer response timeout must be positive")
        executable_path, self.executable_identity = _file_identity(
            executable,
            repo_root=repo_root,
            label="tokenizer executable",
            expected_sha256=executable_sha256,
        )
        if not os.access(executable_path, os.X_OK):
            raise ExactTokenizerError("tokenizer executable is not executable")
        artifact_path, self.artifact_identity = _file_identity(
            artifact,
            repo_root=repo_root,
            label="tokenizer artifact",
            expected_sha256=artifact_sha256,
        )
        self.model = model.strip()
        self.revision = revision.strip()
        self.response_identity_sha256 = tokenizer_response_identity_sha256(
            model=self.model,
            revision=self.revision,
            artifact_sha256=self.artifact_identity["sha256"],
        )
        self.response_timeout_seconds = response_timeout_seconds
        self._request_count = 0
        self._response_count = 0
        self._text_characters = 0
        self._text_utf8_bytes = 0
        self._provider_request_ids: set[str] = set()
        self._closed = False
        command = (
            str(executable_path),
            "--artifact",
            str(artifact_path),
            "--model",
            self.model,
            "--revision",
            self.revision,
            "--protocol",
            TOKENIZER_PROTOCOL,
        )
        try:
            self._process = subprocess.Popen(
                command,
                cwd=repo_root.resolve(),
                env={
                    "PATH": os.environ.get("PATH", os.defpath),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                },
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise ExactTokenizerError("could not start the pinned tokenizer executable") from exc
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise ExactTokenizerError("tokenizer process did not expose JSONL pipes")
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)

    def __enter__(self) -> JsonlExactTokenizer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def count(self, text: str) -> TokenizerObservation:
        if self._closed:
            raise ExactTokenizerError("tokenizer process is closed")
        if not isinstance(text, str):
            raise ExactTokenizerError("tokenizer input must be text")
        self._request_count += 1
        request_id = self._request_count
        encoded = text.encode("utf-8")
        text_sha256 = hashlib.sha256(encoded).hexdigest()
        request = {
            "protocol": TOKENIZER_PROTOCOL,
            "request_id": request_id,
            "text_sha256": text_sha256,
            "text": text,
        }
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(
                json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ExactTokenizerError(
                "tokenizer process closed before accepting a request"
            ) from exc
        if not self._selector.select(self.response_timeout_seconds):
            raise ExactTokenizerError("tokenizer response timed out")
        assert self._process.stdout is not None
        raw = self._process.stdout.readline()
        if not raw:
            raise ExactTokenizerError("tokenizer process closed without a response")
        try:
            response = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_fields,
                parse_constant=_reject_nonfinite,
            )
        except (json.JSONDecodeError, ExactTokenizerError) as exc:
            raise ExactTokenizerError(f"tokenizer emitted invalid strict JSON: {exc}") from exc
        if not isinstance(response, dict) or set(response) != _RESPONSE_FIELDS:
            raise ExactTokenizerError("tokenizer response has unexpected fields")
        expected = {
            "protocol": TOKENIZER_PROTOCOL,
            "request_id": request_id,
            "text_sha256": text_sha256,
            "tokenizer_model": self.model,
            "tokenizer_revision": self.revision,
            "tokenizer_artifact_sha256": self.artifact_identity["sha256"],
        }
        for field, value in expected.items():
            if type(response.get(field)) is not type(value) or response.get(field) != value:
                raise ExactTokenizerError(f"tokenizer response {field} did not match the request")
        provider_request_id = response.get("provider_request_id")
        if (
            not isinstance(provider_request_id, str)
            or _PROVIDER_REQUEST_ID_RE.fullmatch(provider_request_id) is None
            or provider_request_id in self._provider_request_ids
        ):
            raise ExactTokenizerError("tokenizer provider request ID is invalid or reused")
        token_count = response.get("token_count")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
            raise ExactTokenizerError("tokenizer token_count must be a positive integer")
        self._provider_request_ids.add(provider_request_id)
        self._response_count += 1
        self._text_characters += len(text)
        self._text_utf8_bytes += len(encoded)
        return TokenizerObservation(
            request_id=request_id,
            provider_request_id=provider_request_id,
            response_identity_sha256=self.response_identity_sha256,
            text_sha256=text_sha256,
            token_count=token_count,
        )

    @property
    def evidence(self) -> dict[str, Any]:
        return {
            "method": "exact_serialized_reader_prompt",
            "provider": "JsonlExactTokenizer",
            "exact_model_tokenizer": True,
            "tokenizer_model": self.model,
            "tokenizer_revision": self.revision,
            "tokenizer_artifact": self.artifact_identity,
            "tokenizer_executable": self.executable_identity,
            "protocol": TOKENIZER_PROTOCOL,
            "response_identity_sha256": self.response_identity_sha256,
            "observation_accounting": {
                "source": "provider-observed",
                "requests": self._request_count,
                "responses": self._response_count,
                "unique_provider_request_ids": len(self._provider_request_ids),
                "text_characters": self._text_characters,
                "text_utf8_bytes": self._text_utf8_bytes,
                "exact_response_identity_verified": (
                    self._request_count == self._response_count == len(self._provider_request_ids)
                ),
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._selector.close()
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            return_code = self._process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            self._process.wait(timeout=5)
            raise ExactTokenizerError("tokenizer process did not exit after EOF") from exc
        if return_code != 0:
            raise ExactTokenizerError(f"tokenizer process exited with status {return_code}")


def tokenizer_response_identity_sha256(
    *,
    model: str,
    revision: str,
    artifact_sha256: str,
) -> str:
    identity = {
        "protocol": TOKENIZER_PROTOCOL,
        "tokenizer_artifact_sha256": artifact_sha256,
        "tokenizer_model": model,
        "tokenizer_revision": revision,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "ExactTokenizer",
    "ExactTokenizerError",
    "JsonlExactTokenizer",
    "TOKENIZER_PROTOCOL",
    "TokenizerObservation",
    "tokenizer_response_identity_sha256",
]
