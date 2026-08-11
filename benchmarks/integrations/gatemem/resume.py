"""Authenticated, episode-boundary resume state for GateMem external runs."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answering import GATEMEM_ACTIONS
from .contracts import GateMemContractError, assert_hidden_fields_absent
from .runner import HarnessRun

RESUME_ARTIFACT_TYPE = "swarmbrain-gatemem-resume-state"
RESUME_SCHEMA_VERSION = 1
RESUME_HMAC_ALGORITHM = "HMAC-SHA256"
DEFAULT_RESUME_KEY_ENV = "GATEMEM_RESUME_HMAC_KEY"

_MAX_RESUME_BYTES = 512 * 1024 * 1024
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class EpisodeResumeSpec:
    """The immutable official-order checkpoint slice for one episode."""

    episode_id: str
    checkpoint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise GateMemContractError("resume episode ID must be non-empty")
        if not self.checkpoint_ids:
            raise GateMemContractError(
                f"resume episode {self.episode_id!r} must contain checkpoints"
            )
        if any(not item for item in self.checkpoint_ids):
            raise GateMemContractError("resume checkpoint IDs must be non-empty")
        if len(self.checkpoint_ids) != len(set(self.checkpoint_ids)):
            raise GateMemContractError(
                f"resume episode {self.episode_id!r} has duplicate checkpoint IDs"
            )


class AuthenticatedResumeStore:
    """Persist a strictly validated prefix of complete episode results.

    A process lock covers the whole run. Each state replacement is written and
    fsynced in the destination directory before the old state is replaced.
    State is useful only with the caller-provided HMAC key and is deliberately
    not shaped like either an official prediction JSONL or a run audit.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        key: bytes,
        fingerprint: dict[str, Any],
        episodes: Sequence[EpisodeResumeSpec],
    ) -> None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self.path = Path(os.path.abspath(candidate))
        if not self.path.name.endswith(".resume.json"):
            raise GateMemContractError("--resume-state must end with .resume.json")
        if len(key) < 32:
            raise GateMemContractError("GateMem resume HMAC key must contain at least 32 bytes")
        if len(key) > 4096:
            raise GateMemContractError("GateMem resume HMAC key is unreasonably large")
        self._key = bytes(key)
        self._key_id = hashlib.sha256(
            b"swarmbrain-gatemem-resume-key-id-v1\0" + self._key
        ).hexdigest()
        self._fingerprint = _json_clone(fingerprint)
        if not isinstance(self._fingerprint, dict):
            raise GateMemContractError("GateMem resume fingerprint must be an object")
        self._fingerprint_sha256 = _json_sha256(self._fingerprint)
        self.episodes = tuple(episodes)
        if not self.episodes:
            raise GateMemContractError("GateMem resume plan must contain at least one episode")
        episode_ids = [item.episode_id for item in self.episodes]
        checkpoint_ids = [
            checkpoint_id for item in self.episodes for checkpoint_id in item.checkpoint_ids
        ]
        if len(episode_ids) != len(set(episode_ids)):
            raise GateMemContractError("GateMem resume plan has duplicate episode IDs")
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise GateMemContractError("GateMem resume plan has duplicate checkpoint IDs")
        expected_selection = {
            "episodes": [
                {
                    "episode_id": item.episode_id,
                    "checkpoint_ids": list(item.checkpoint_ids),
                }
                for item in self.episodes
            ]
        }
        if self._fingerprint.get("selection") != expected_selection:
            raise GateMemContractError(
                "GateMem resume fingerprint selection does not match the official task plan"
            )
        self._lock_fd: int | None = None

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold an exclusive process lock so two resumptions cannot issue duplicate calls."""

        if self._lock_fd is not None:
            raise GateMemContractError("GateMem resume state is already locked by this process")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise GateMemContractError("GateMem resume state cannot be a symbolic link")
        lock_path = self.path.with_name(self.path.name + ".lock")
        if lock_path.is_symlink():
            raise GateMemContractError("GateMem resume lock cannot be a symbolic link")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags, 0o600)
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if "fd" in locals():
                os.close(fd)
            raise GateMemContractError(
                f"another process holds the GateMem resume lock: {lock_path}"
            ) from exc
        except OSError as exc:
            if "fd" in locals():
                os.close(fd)
            raise GateMemContractError(f"cannot lock GateMem resume state: {lock_path}") from exc
        self._lock_fd = fd
        try:
            yield
        finally:
            self._lock_fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def load_or_initialize(self) -> tuple[HarnessRun, ...]:
        self._require_lock()
        if self.path.exists():
            return self._read()
        self._write(())
        return ()

    def append_episode(
        self,
        completed: Sequence[HarnessRun],
        episode_run: HarnessRun,
    ) -> tuple[HarnessRun, ...]:
        """Append exactly the next complete episode and atomically authenticate it."""

        self._require_lock()
        persisted = self._read()
        if _runs_sha256(persisted) != _runs_sha256(completed):
            raise GateMemContractError("GateMem resume state changed since it was loaded")
        index = len(persisted)
        if index >= len(self.episodes):
            raise GateMemContractError("GateMem resume state is already complete")
        self._validate_episode_run(episode_run, self.episodes[index])
        updated = (*persisted, episode_run)
        self._write(updated)
        return updated

    def combine_complete(self, completed: Sequence[HarnessRun]) -> HarnessRun:
        """Reconstruct canonical paired prediction/audit evidence in official order."""

        self._require_lock()
        runs = tuple(completed)
        if len(runs) != len(self.episodes):
            raise GateMemContractError("GateMem resume state is not complete")
        for run, spec in zip(runs, self.episodes, strict=True):
            self._validate_episode_run(run, spec)
        first = runs[0].audit
        predictions = tuple(row for run in runs for row in run.predictions)
        ingest_operations = [event for run in runs for event in run.audit["ingest_operations"]]
        total_latency = sum(float(run.audit["latency_ms"]["total"]) for run in runs)
        audit = {
            "schema_version": first["schema_version"],
            "benchmark": first["benchmark"],
            "gatemem_commit": first["gatemem_commit"],
            "adapter": first["adapter"],
            "config": _json_clone(first["config"]),
            "audience_policy": _json_clone(first["audience_policy"]),
            "turn_interpreter": first["turn_interpreter"],
            "episodes": len(runs),
            "checkpoints": len(predictions),
            "ingest_operations": _json_clone(ingest_operations),
            "latency_ms": {"total": total_latency},
        }
        assert_hidden_fields_absent(audit)
        return HarnessRun(predictions=predictions, audit=audit)

    def authenticated_payload_sha256(self, completed: Sequence[HarnessRun]) -> str:
        """Digest the complete payload only after re-authenticating persisted state."""

        self._require_lock()
        persisted = self._read()
        if _runs_sha256(persisted) != _runs_sha256(completed):
            raise GateMemContractError("GateMem resume state changed since it was loaded")
        if len(persisted) != len(self.episodes):
            raise GateMemContractError("GateMem resume state is not complete")
        return _json_sha256(self._payload(persisted))

    def _require_lock(self) -> None:
        if self._lock_fd is None:
            raise GateMemContractError("GateMem resume state must be locked before access")

    def _read(self) -> tuple[HarnessRun, ...]:
        if self.path.is_symlink():
            raise GateMemContractError("GateMem resume state cannot be a symbolic link")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = -1
        try:
            file_descriptor = os.open(self.path, flags)
            file_status = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise GateMemContractError("GateMem resume state must be a regular file")
            if file_status.st_size > _MAX_RESUME_BYTES:
                raise GateMemContractError("GateMem resume state exceeds the size limit")
            with os.fdopen(file_descriptor, "rb") as handle:
                file_descriptor = -1
                raw_bytes = handle.read(_MAX_RESUME_BYTES + 1)
            if len(raw_bytes) > _MAX_RESUME_BYTES:
                raise GateMemContractError("GateMem resume state exceeds the size limit")
        except OSError as exc:
            raise GateMemContractError(f"cannot read GateMem resume state: {self.path}") from exc
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GateMemContractError("GateMem resume state must be UTF-8 JSON") from exc
        envelope = _strict_json_object(raw_text)
        expected_envelope_keys = {
            "artifact_type",
            "schema_version",
            "payload",
            "authentication",
        }
        if set(envelope) != expected_envelope_keys:
            raise GateMemContractError("GateMem resume envelope has unexpected fields")
        if (
            envelope.get("artifact_type") != RESUME_ARTIFACT_TYPE
            or envelope.get("schema_version") != RESUME_SCHEMA_VERSION
        ):
            raise GateMemContractError("GateMem resume artifact type or schema is unsupported")
        authentication = envelope.get("authentication")
        if not isinstance(authentication, dict) or set(authentication) != {
            "algorithm",
            "key_id",
            "tag",
        }:
            raise GateMemContractError("GateMem resume authentication block is malformed")
        if (
            authentication.get("algorithm") != RESUME_HMAC_ALGORITHM
            or authentication.get("key_id") != self._key_id
        ):
            raise GateMemContractError("GateMem resume state was authenticated with another key")
        tag = authentication.get("tag")
        if not isinstance(tag, str) or not re.fullmatch(r"[0-9a-f]{64}", tag):
            raise GateMemContractError("GateMem resume authentication tag is malformed")
        authenticated = {
            "artifact_type": RESUME_ARTIFACT_TYPE,
            "schema_version": RESUME_SCHEMA_VERSION,
            "payload": envelope.get("payload"),
            "authentication": {
                "algorithm": RESUME_HMAC_ALGORITHM,
                "key_id": self._key_id,
            },
        }
        expected_tag = _authentication_tag(self._key, authenticated)
        if not hmac.compare_digest(tag, expected_tag):
            raise GateMemContractError("GateMem resume state authentication failed")

        payload = envelope.get("payload")
        if not isinstance(payload, dict) or set(payload) != {
            "status",
            "fingerprint",
            "fingerprint_sha256",
            "completed_episodes",
        }:
            raise GateMemContractError("GateMem resume payload is malformed")
        fingerprint = payload.get("fingerprint")
        fingerprint_digest = payload.get("fingerprint_sha256")
        if not isinstance(fingerprint, dict) or fingerprint_digest != _json_sha256(fingerprint):
            raise GateMemContractError("GateMem resume fingerprint digest is invalid")
        if fingerprint_digest != self._fingerprint_sha256 or _canonical_json(
            fingerprint
        ) != _canonical_json(self._fingerprint):
            raise GateMemContractError(
                "GateMem resume fingerprint does not exactly match this invocation"
            )
        chunks = payload.get("completed_episodes")
        if not isinstance(chunks, list) or len(chunks) > len(self.episodes):
            raise GateMemContractError("GateMem resume completed episode list is malformed")
        status = payload.get("status")
        expected_status = "complete" if len(chunks) == len(self.episodes) else "partial"
        if status != expected_status:
            raise GateMemContractError("GateMem resume status is inconsistent with coverage")
        runs = tuple(
            self._decode_episode_chunk(chunk, self.episodes[index])
            for index, chunk in enumerate(chunks)
        )
        return runs

    def _decode_episode_chunk(self, chunk: Any, spec: EpisodeResumeSpec) -> HarnessRun:
        if not isinstance(chunk, dict) or set(chunk) != {
            "episode_id",
            "checkpoint_ids",
            "predictions",
            "audit",
            "paired_evidence_sha256",
        }:
            raise GateMemContractError("GateMem resume episode chunk is malformed")
        if chunk.get("episode_id") != spec.episode_id or chunk.get("checkpoint_ids") != list(
            spec.checkpoint_ids
        ):
            raise GateMemContractError(
                "GateMem resume episode/checkpoint order is not the official prefix"
            )
        predictions = chunk.get("predictions")
        audit = chunk.get("audit")
        if not isinstance(predictions, list) or not isinstance(audit, dict):
            raise GateMemContractError("GateMem resume paired evidence is malformed")
        paired = {"predictions": predictions, "audit": audit}
        if chunk.get("paired_evidence_sha256") != _json_sha256(paired):
            raise GateMemContractError("GateMem resume paired evidence digest is invalid")
        run = HarnessRun(predictions=tuple(predictions), audit=audit)
        self._validate_episode_run(run, spec)
        return run

    def _validate_episode_run(self, run: HarnessRun, spec: EpisodeResumeSpec) -> None:
        if not isinstance(run, HarnessRun):
            raise GateMemContractError("GateMem resume entry must be a HarnessRun")
        if len(run.predictions) != len(spec.checkpoint_ids):
            raise GateMemContractError(
                f"GateMem resume episode {spec.episode_id!r} has incomplete predictions"
            )
        actual_ids: list[str] = []
        expected_commit = self._fingerprint["dataset"]["repository_commit"]
        expected_provider = self._fingerprint["answer_model"]["provider"]
        expected_model = self._fingerprint["answer_model"]["model"]
        expected_revision = self._fingerprint["answer_model"]["revision"]
        run_parameters = self._fingerprint["run_parameters"]
        for row in run.predictions:
            if not isinstance(row, dict) or set(row) != {
                "checkpoint_id",
                "output",
                "swarmbrain_audit",
            }:
                raise GateMemContractError("GateMem resume prediction row is malformed")
            checkpoint_id = row.get("checkpoint_id")
            if not isinstance(checkpoint_id, str):
                raise GateMemContractError("GateMem resume prediction lacks a checkpoint ID")
            actual_ids.append(checkpoint_id)
            output = row.get("output")
            if not isinstance(output, dict) or set(output) != {
                "action",
                "answer",
                "answer_structured",
                "used_record_ids",
                "memory_audit",
                "llm_usage",
            }:
                raise GateMemContractError("GateMem resume prediction output is malformed")
            if output.get("action") not in GATEMEM_ACTIONS:
                raise GateMemContractError("GateMem resume prediction action is invalid")
            if not isinstance(output.get("answer"), str) or not isinstance(
                output.get("answer_structured"), dict
            ):
                raise GateMemContractError("GateMem resume answer fields are malformed")
            used_record_ids = output.get("used_record_ids")
            if (
                not isinstance(used_record_ids, list)
                or any(not isinstance(item, str) for item in used_record_ids)
                or len(used_record_ids) != len(set(used_record_ids))
            ):
                raise GateMemContractError("GateMem resume record citations are malformed")
            memory_audit = output.get("memory_audit")
            prompt_context = (
                memory_audit.get("prompt_context") if isinstance(memory_audit, dict) else None
            )
            if (
                not isinstance(memory_audit, dict)
                or memory_audit.get("schema_version") != 1
                or memory_audit.get("stage") != "prompt_context"
                or memory_audit.get("context_format") != "swarmbrain-json-v1"
                or not isinstance(prompt_context, dict)
            ):
                raise GateMemContractError("GateMem resume memory audit is malformed")
            prompt_items = prompt_context.get("items")
            if not isinstance(prompt_items, list):
                raise GateMemContractError("GateMem resume prompt-context items are malformed")
            prompt_text = _canonical_json(prompt_items)
            if (
                prompt_context.get("text") != prompt_text
                or prompt_context.get("n_chars") != len(prompt_text)
                or prompt_context.get("n_items") != len(prompt_items)
            ):
                raise GateMemContractError("GateMem resume prompt-context trace is not exact")
            usage = output.get("llm_usage")
            if not isinstance(usage, dict) or set(usage) != {
                "input_tokens",
                "output_tokens",
                "total_tokens",
            }:
                raise GateMemContractError("GateMem resume token usage is malformed")
            input_tokens = _nonnegative_int(usage.get("input_tokens"), "input_tokens")
            output_tokens = _nonnegative_int(usage.get("output_tokens"), "output_tokens")
            if usage.get("total_tokens") != input_tokens + output_tokens:
                raise GateMemContractError("GateMem resume token usage does not reconcile")
            row_audit = row.get("swarmbrain_audit")
            if not isinstance(row_audit, dict):
                raise GateMemContractError("GateMem resume prediction audit is malformed")
            if (
                row_audit.get("schema_version")
                != self._fingerprint["protocol"]["harness_schema_version"]
                or row_audit.get("gatemem_commit") != expected_commit
                or row_audit.get("episode_id") != spec.episode_id
            ):
                raise GateMemContractError("GateMem resume prediction provenance is invalid")
            answer_model = row_audit.get("answer_model")
            if (
                not isinstance(answer_model, dict)
                or answer_model.get("provider") != expected_provider
                or answer_model.get("model") != expected_model
                or answer_model.get("revision") != expected_revision
            ):
                raise GateMemContractError("GateMem resume answer model does not match")
            retrieval = row_audit.get("retrieval")
            if not isinstance(retrieval, dict) or (
                retrieval.get("limit") != run_parameters["recall_limit"]
                or retrieval.get("min_score") != run_parameters["min_score"]
            ):
                raise GateMemContractError("GateMem resume retrieval configuration drifted")
            tokens = row_audit.get("tokens")
            if (
                not isinstance(tokens, dict)
                or tokens.get("context_budget") != run_parameters["context_token_budget"]
                or tokens.get("provider_input") != input_tokens
                or tokens.get("provider_output") != output_tokens
                or tokens.get("provider_usage_reported") is not True
                or tokens.get("usage_source") != "provider"
            ):
                raise GateMemContractError("GateMem resume token/config evidence drifted")
            latency_fields = row_audit.get("latency_ms")
            if not isinstance(latency_fields, dict) or set(latency_fields) != {
                "incremental_ingest",
                "recall",
                "answer",
                "query_total",
            }:
                raise GateMemContractError("GateMem resume prediction latency is malformed")
            for value in latency_fields.values():
                _nonnegative_float(value, "prediction latency")
            assert_hidden_fields_absent(row)
        if tuple(actual_ids) != spec.checkpoint_ids:
            raise GateMemContractError(
                "GateMem resume predictions do not preserve official checkpoint order"
            )

        audit = run.audit
        expected_audit_keys = {
            "schema_version",
            "benchmark",
            "gatemem_commit",
            "adapter",
            "config",
            "audience_policy",
            "turn_interpreter",
            "episodes",
            "checkpoints",
            "ingest_operations",
            "latency_ms",
        }
        if not isinstance(audit, dict) or set(audit) != expected_audit_keys:
            raise GateMemContractError("GateMem resume run audit is malformed")
        expected_config = {
            "recall_limit": run_parameters["recall_limit"],
            "min_score": run_parameters["min_score"],
            "context_token_budget": run_parameters["context_token_budget"],
        }
        if (
            audit.get("schema_version") != self._fingerprint["protocol"]["harness_schema_version"]
            or audit.get("benchmark") != "GateMem"
            or audit.get("gatemem_commit") != expected_commit
            or audit.get("adapter") != "swarmbrain-gatemem-external"
            or audit.get("config") != expected_config
            or audit.get("audience_policy") != self._fingerprint["audience_policy"]
            or audit.get("turn_interpreter") != self._fingerprint["protocol"]["turn_interpreter"]
            or audit.get("episodes") != 1
            or audit.get("checkpoints") != len(spec.checkpoint_ids)
        ):
            raise GateMemContractError("GateMem resume run audit provenance drifted")
        operations = audit.get("ingest_operations")
        if not isinstance(operations, list) or any(
            not isinstance(item, dict) or item.get("episode_id") != spec.episode_id
            for item in operations
        ):
            raise GateMemContractError("GateMem resume ingest provenance is malformed")
        latency = audit.get("latency_ms")
        if not isinstance(latency, dict) or set(latency) != {"total"}:
            raise GateMemContractError("GateMem resume total latency is malformed")
        _nonnegative_float(latency.get("total"), "run latency")
        assert_hidden_fields_absent(audit)

    def _write(self, runs: Sequence[HarnessRun]) -> None:
        payload = self._payload(runs)
        authenticated = {
            "artifact_type": RESUME_ARTIFACT_TYPE,
            "schema_version": RESUME_SCHEMA_VERSION,
            "payload": payload,
            "authentication": {
                "algorithm": RESUME_HMAC_ALGORITHM,
                "key_id": self._key_id,
            },
        }
        envelope = {
            **authenticated,
            "authentication": {
                **authenticated["authentication"],
                "tag": _authentication_tag(self._key, authenticated),
            },
        }
        encoded = (
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(encoded) > _MAX_RESUME_BYTES:
            raise GateMemContractError("GateMem resume state exceeds the size limit")
        _atomic_write_private(self.path, encoded)

    def _payload(self, runs: Sequence[HarnessRun]) -> dict[str, Any]:
        completed = tuple(runs)
        if len(completed) > len(self.episodes):
            raise GateMemContractError("GateMem resume state exceeds planned coverage")
        chunks: list[dict[str, Any]] = []
        for run, spec in zip(completed, self.episodes, strict=False):
            self._validate_episode_run(run, spec)
            predictions = _json_clone(list(run.predictions))
            audit = _json_clone(run.audit)
            paired = {"predictions": predictions, "audit": audit}
            chunks.append(
                {
                    "episode_id": spec.episode_id,
                    "checkpoint_ids": list(spec.checkpoint_ids),
                    **paired,
                    "paired_evidence_sha256": _json_sha256(paired),
                }
            )
        return {
            "status": "complete" if len(completed) == len(self.episodes) else "partial",
            "fingerprint": self._fingerprint,
            "fingerprint_sha256": self._fingerprint_sha256,
            "completed_episodes": chunks,
        }


def load_resume_key(env_name: str) -> bytes:
    """Resolve a local HMAC key by name without ever serializing its value."""

    if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name):
        raise GateMemContractError("--resume-key-env must be a valid environment variable name")
    value = os.getenv(env_name)
    if value is None:
        raise GateMemContractError(
            f"GateMem resume HMAC key environment variable is unset: {env_name}"
        )
    key = value.encode("utf-8")
    if len(key) < 32:
        raise GateMemContractError("GateMem resume HMAC key must contain at least 32 bytes")
    if len(key) > 4096:
        raise GateMemContractError("GateMem resume HMAC key is unreasonably large")
    return key


def file_sha256(path: str | Path, *, label: str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise GateMemContractError(f"cannot hash {label}: {path}") from exc


def _authentication_tag(key: bytes, authenticated: dict[str, Any]) -> str:
    return hmac.new(key, _canonical_json(authenticated).encode("utf-8"), hashlib.sha256).hexdigest()


def _runs_sha256(runs: Sequence[HarnessRun]) -> str:
    return _json_sha256(
        [{"predictions": list(run.predictions), "audit": run.audit} for run in runs]
    )


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GateMemContractError("GateMem resume state must be finite canonical JSON") from exc


def _strict_json_object(raw: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise GateMemContractError(f"GateMem resume JSON repeats field {key!r}")
            decoded[key] = value
        return decoded

    def reject_constant(value: str) -> Any:
        raise GateMemContractError(f"GateMem resume JSON contains non-finite value {value}")

    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise GateMemContractError("GateMem resume state is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise GateMemContractError("GateMem resume state must be a JSON object")
    return decoded


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GateMemContractError(f"GateMem resume {name} must be a non-negative integer")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GateMemContractError(f"GateMem resume {name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise GateMemContractError(f"GateMem resume {name} must be finite and non-negative")
    return converted


def _atomic_write_private(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise GateMemContractError("GateMem resume state cannot be a symbolic link")
    file_descriptor = -1
    temporary_name: str | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise GateMemContractError(f"cannot atomically write GateMem resume state: {path}") from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                Path(temporary_name).unlink()


__all__ = [
    "AuthenticatedResumeStore",
    "DEFAULT_RESUME_KEY_ENV",
    "EpisodeResumeSpec",
    "RESUME_ARTIFACT_TYPE",
    "RESUME_HMAC_ALGORITHM",
    "RESUME_SCHEMA_VERSION",
    "file_sha256",
    "load_resume_key",
]
