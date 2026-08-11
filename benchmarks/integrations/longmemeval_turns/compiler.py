"""Immutable, evaluation-only turn projection for cleaned LongMemEval-S.

The production memory model intentionally remains session-granular.  This
module compiles the benchmark source artifact into a separate turn corpus for
controlled CrossEncoder and Chain-of-Memory experiments.  Every transformation
is byte/digest bound and the public manifest contains no question, answer, or
turn text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OFFICIAL_LONGMEMEVAL_S_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
PROJECTION_ARTIFACT_TYPE = "swarmbrain-longmemeval-s-turn-projection"
PROJECTION_SCHEMA_VERSION = 1
TURN_SERIALIZER_VERSION = "longmemeval-turn-timestamp-role-content-json-v1"
TIMESTAMP_CONVENTION = "longmemeval-wall-clock-as-utc-v1"

IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "uv.lock",
    "benchmarks/__init__.py",
    "benchmarks/integrations/longmemeval_turns/__init__.py",
    "benchmarks/integrations/longmemeval_turns/compiler.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TIMESTAMP_RE = re.compile(
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2}) "
    r"\((?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class LongMemEvalTurnProjectionError(ValueError):
    """The source cannot support a deterministic LongMemEval turn projection."""


def _canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON for digests; Unicode code points are never normalized."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LongMemEvalTurnProjectionError(
            "LongMemEval source contains data that is not finite UTF-8 JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _checked_digest(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LongMemEvalTurnProjectionError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest"
        )
    return value


def _stable_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LongMemEvalTurnProjectionError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise LongMemEvalTurnProjectionError(f"{label} cannot have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise LongMemEvalTurnProjectionError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise LongMemEvalTurnProjectionError(f"{label} must be valid UTF-8 text") from exc
    return value


def _index(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LongMemEvalTurnProjectionError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    result = _index(value, label=label)
    if result == 0:
        raise LongMemEvalTurnProjectionError(f"{label} must be positive")
    return result


def _strict_timestamp(value: Any, *, label: str) -> tuple[str, str]:
    """Return the exact source timestamp and its explicit UTC interpretation.

    LongMemEval timestamps have no timezone.  The benchmark already treats
    them as one UTC calendar, so this projection uses the same convention.  An
    input is accepted only if it is already in the single canonical spelling;
    stripping or reparsing alternate spellings would make IDs reproducible but
    documents unstable.
    """

    if not isinstance(value, str):
        raise LongMemEvalTurnProjectionError(f"{label} must be a string")
    match = _TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise LongMemEvalTurnProjectionError(f"{label} must use YYYY/MM/DD (Ddd) HH:MM exactly")
    try:
        parsed = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=UTC,
        )
    except ValueError as exc:
        raise LongMemEvalTurnProjectionError(f"{label} is not a valid calendar time") from exc
    weekday = _WEEKDAYS[parsed.weekday()]
    if match.group("weekday") != weekday:
        raise LongMemEvalTurnProjectionError(f"{label} weekday does not match its calendar date")
    canonical = (
        f"{parsed.year:04d}/{parsed.month:02d}/{parsed.day:02d} "
        f"({weekday}) {parsed.hour:02d}:{parsed.minute:02d}"
    )
    if canonical != value:
        raise LongMemEvalTurnProjectionError(f"{label} is not canonically serialized")
    return value, parsed.isoformat().replace("+00:00", "Z")


def _document_bytes(*, timestamp: str, role: str, content: str) -> bytes:
    """Serialize precisely timestamp, role, and the unmodified source content."""

    # Do not route this through sorted-key canonical JSON: field order is part
    # of the candidate document protocol and follows the F0 order explicitly.
    payload = {"timestamp": timestamp, "role": role, "content": content}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if json.loads(encoded) != payload:
            raise LongMemEvalTurnProjectionError("turn document did not round-trip exactly")
        return encoded
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        if isinstance(exc, LongMemEvalTurnProjectionError):
            raise
        raise LongMemEvalTurnProjectionError(
            "turn timestamp, role, and content must be valid UTF-8 JSON strings"
        ) from exc


@dataclass(frozen=True, slots=True)
class ByteDigest:
    """Size and SHA-256 binding for bytes intentionally omitted from a manifest."""

    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _index(self.bytes, label="byte binding size")
        _checked_digest(self.sha256, label="byte binding digest")

    @classmethod
    def from_bytes(cls, value: bytes) -> ByteDigest:
        return cls(bytes=len(value), sha256=_sha256_bytes(value))

    def as_dict(self) -> dict[str, int | str]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SourceArtifactBinding:
    """Exact binding to the input JSON artifact, before parsing."""

    label: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _stable_identifier(self.label, label="source artifact label")
        _index(self.bytes, label="source artifact bytes")
        _checked_digest(self.sha256, label="source artifact digest")

    def as_dict(self) -> dict[str, int | str]:
        return {"label": self.label, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True, order=True)
class LongMemEvalTurnId:
    """The only turn identity: question ID plus source array positions."""

    question_id: str
    session_position: int
    turn_position: int

    def __post_init__(self) -> None:
        _stable_identifier(self.question_id, label="turn question_id")
        _index(self.session_position, label="turn session_position")
        _index(self.turn_position, label="turn turn_position")

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.question_id, self.session_position, self.turn_position)

    @property
    def canonical_id(self) -> str:
        """Unambiguous string form for scorer APIs that cannot accept tuples."""

        return _canonical_json_bytes(list(self.as_tuple())).decode("utf-8")


@dataclass(frozen=True, slots=True)
class QuestionBinding:
    question_id: str
    question_position: int
    sessions: int
    turns: int
    source_record: ByteDigest

    def __post_init__(self) -> None:
        _stable_identifier(self.question_id, label="question binding question_id")
        _index(self.question_position, label="question binding position")
        _positive_integer(self.sessions, label="question binding session count")
        _positive_integer(self.turns, label="question binding turn count")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_position": self.question_position,
            "sessions": self.sessions,
            "turns": self.turns,
            "source_record": self.source_record.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class SessionBinding:
    question_id: str
    session_position: int
    parent_session_id: str
    parent_session_date: str
    parent_session_date_utc: str
    turns: int
    source_session: ByteDigest

    def __post_init__(self) -> None:
        _stable_identifier(self.question_id, label="session binding question_id")
        _index(self.session_position, label="session binding position")
        _stable_identifier(self.parent_session_id, label="session binding parent ID")
        _, expected_utc = _strict_timestamp(
            self.parent_session_date,
            label="session binding parent date",
        )
        if self.parent_session_date_utc != expected_utc:
            raise LongMemEvalTurnProjectionError(
                "session binding UTC date differs from its exact parent date"
            )
        _positive_integer(self.turns, label="session binding turn count")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "session_position": self.session_position,
            "parent_session_id": self.parent_session_id,
            "parent_session_date": self.parent_session_date,
            "parent_session_date_utc": self.parent_session_date_utc,
            "turns": self.turns,
            "source_session": self.source_session.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TurnProjection:
    """One immutable candidate document plus its complete source provenance."""

    turn_id: LongMemEvalTurnId
    parent_session_id: str
    parent_session_date: str
    parent_session_date_utc: str
    role: str
    original_content: str
    serialized_text: str
    source_record: ByteDigest
    source_session: ByteDigest
    source_turn: ByteDigest
    original_content_utf8: ByteDigest
    serialized_document_utf8: ByteDigest

    def __post_init__(self) -> None:
        _stable_identifier(self.parent_session_id, label="turn parent session ID")
        _, expected_utc = _strict_timestamp(
            self.parent_session_date,
            label="turn parent session date",
        )
        if self.parent_session_date_utc != expected_utc:
            raise LongMemEvalTurnProjectionError(
                "turn UTC date differs from its exact parent session date"
            )
        _stable_identifier(self.role, label="turn role")
        if not isinstance(self.original_content, str):
            raise LongMemEvalTurnProjectionError("turn original content must be a string")
        try:
            content_bytes = self.original_content.encode("utf-8")
            serialized_bytes = self.serialized_text.encode("utf-8")
        except UnicodeError as exc:
            raise LongMemEvalTurnProjectionError("turn text must be valid UTF-8") from exc
        expected_document = _document_bytes(
            timestamp=self.parent_session_date,
            role=self.role,
            content=self.original_content,
        )
        if serialized_bytes != expected_document:
            raise LongMemEvalTurnProjectionError(
                "serialized turn document differs from timestamp, role, or original content"
            )
        if self.original_content_utf8 != ByteDigest.from_bytes(content_bytes):
            raise LongMemEvalTurnProjectionError("turn content byte binding is inconsistent")
        if self.serialized_document_utf8 != ByteDigest.from_bytes(serialized_bytes):
            raise LongMemEvalTurnProjectionError("turn document byte binding is inconsistent")

    @property
    def serialized_bytes(self) -> bytes:
        return self.serialized_text.encode("utf-8")

    def content_free_binding(self) -> dict[str, Any]:
        """Return provenance and hashes without question, answer, or turn text."""

        return {
            "turn_id": list(self.turn_id.as_tuple()),
            "canonical_turn_id": self.turn_id.canonical_id,
            "parent_session_id": self.parent_session_id,
            "parent_session_date": self.parent_session_date,
            "parent_session_date_utc": self.parent_session_date_utc,
            "source_record": self.source_record.as_dict(),
            "source_session": self.source_session.as_dict(),
            "source_turn": self.source_turn.as_dict(),
            "original_content_utf8": self.original_content_utf8.as_dict(),
            "serialized_document_utf8": self.serialized_document_utf8.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TurnProjectionCorpus:
    """Complete immutable question-local turn projection of one source artifact."""

    source_artifact: SourceArtifactBinding
    parsed_records: ByteDigest
    questions: tuple[QuestionBinding, ...]
    sessions: tuple[SessionBinding, ...]
    turns: tuple[TurnProjection, ...]
    serializer_version: str = TURN_SERIALIZER_VERSION
    timestamp_convention: str = TIMESTAMP_CONVENTION
    _cached_projection_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.serializer_version != TURN_SERIALIZER_VERSION:
            raise LongMemEvalTurnProjectionError("turn corpus serializer version is unsupported")
        if self.timestamp_convention != TIMESTAMP_CONVENTION:
            raise LongMemEvalTurnProjectionError("turn corpus timestamp convention is unsupported")
        if not self.questions or not self.sessions or not self.turns:
            raise LongMemEvalTurnProjectionError(
                "turn corpus must contain questions, sessions, and turns"
            )
        question_by_id: dict[str, QuestionBinding] = {}
        for expected_position, question in enumerate(self.questions):
            if question.question_position != expected_position:
                raise LongMemEvalTurnProjectionError(
                    "question bindings are not in contiguous source order"
                )
            if question.question_id in question_by_id:
                raise LongMemEvalTurnProjectionError("turn corpus repeats a question ID")
            question_by_id[question.question_id] = question

        session_by_key: dict[tuple[str, int], SessionBinding] = {}
        session_counts = {question_id: 0 for question_id in question_by_id}
        for session in self.sessions:
            question = question_by_id.get(session.question_id)
            if question is None:
                raise LongMemEvalTurnProjectionError(
                    "session binding refers to an unknown question"
                )
            key = (session.question_id, session.session_position)
            if key in session_by_key:
                raise LongMemEvalTurnProjectionError("turn corpus repeats a session position")
            if session.session_position != session_counts[session.question_id]:
                raise LongMemEvalTurnProjectionError(
                    "session bindings are not contiguous within a question"
                )
            session_counts[session.question_id] += 1
            session_by_key[key] = session
        if any(
            session_counts[question_id] != question.sessions
            for question_id, question in question_by_id.items()
        ):
            raise LongMemEvalTurnProjectionError(
                "question and session binding counts are inconsistent"
            )
        expected_session_keys = [
            (question.question_id, session_position)
            for question in self.questions
            for session_position in range(question.sessions)
        ]
        if [(item.question_id, item.session_position) for item in self.sessions] != (
            expected_session_keys
        ):
            raise LongMemEvalTurnProjectionError(
                "session bindings do not preserve exact source order"
            )

        seen_turn_ids: set[LongMemEvalTurnId] = set()
        turn_counts = {key: 0 for key in session_by_key}
        question_turn_counts = {question_id: 0 for question_id in question_by_id}
        for turn in self.turns:
            if turn.turn_id in seen_turn_ids:
                raise LongMemEvalTurnProjectionError("turn corpus repeats a turn ID")
            seen_turn_ids.add(turn.turn_id)
            session_key = (turn.turn_id.question_id, turn.turn_id.session_position)
            session = session_by_key.get(session_key)
            if session is None:
                raise LongMemEvalTurnProjectionError(
                    "turn binding refers to an unknown parent session"
                )
            if turn.turn_id.turn_position != turn_counts[session_key]:
                raise LongMemEvalTurnProjectionError(
                    "turn bindings are not contiguous within a parent session"
                )
            question = question_by_id[turn.turn_id.question_id]
            if turn.parent_session_id != session.parent_session_id:
                raise LongMemEvalTurnProjectionError("turn parent session ID is inconsistent")
            if turn.parent_session_date != session.parent_session_date:
                raise LongMemEvalTurnProjectionError("turn parent session date is inconsistent")
            if turn.parent_session_date_utc != session.parent_session_date_utc:
                raise LongMemEvalTurnProjectionError("turn parent UTC date is inconsistent")
            if turn.source_record != question.source_record:
                raise LongMemEvalTurnProjectionError("turn source record binding is inconsistent")
            if turn.source_session != session.source_session:
                raise LongMemEvalTurnProjectionError("turn source session binding is inconsistent")
            turn_counts[session_key] += 1
            question_turn_counts[turn.turn_id.question_id] += 1
        if any(turn_counts[key] != session.turns for key, session in session_by_key.items()):
            raise LongMemEvalTurnProjectionError("session and turn binding counts are inconsistent")
        if any(
            question_turn_counts[question_id] != question.turns
            for question_id, question in question_by_id.items()
        ):
            raise LongMemEvalTurnProjectionError(
                "question and turn binding counts are inconsistent"
            )
        expected_turn_ids = [
            LongMemEvalTurnId(
                question_id=session.question_id,
                session_position=session.session_position,
                turn_position=turn_position,
            )
            for session in self.sessions
            for turn_position in range(session.turns)
        ]
        if [item.turn_id for item in self.turns] != expected_turn_ids:
            raise LongMemEvalTurnProjectionError("turn bindings do not preserve exact source order")

        # The projection payload covers the complete corpus and is immutable.
        # Compute its digest once: question-local compilers may bind hundreds of
        # values to this same digest, and recomputing it per value is quadratic.
        object.__setattr__(
            self,
            "_cached_projection_sha256",
            _sha256_json(self._projection_payload()),
        )

    def by_id(self) -> dict[LongMemEvalTurnId, TurnProjection]:
        return {turn.turn_id: turn for turn in self.turns}

    def _projection_payload(self) -> dict[str, Any]:
        return {
            "artifact_type": PROJECTION_ARTIFACT_TYPE,
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "serializer_version": self.serializer_version,
            "timestamp_convention": self.timestamp_convention,
            "timestamp_source": "haystack_dates[session_position]",
            "official_cleaned_release": {
                "required_sha256": OFFICIAL_LONGMEMEVAL_S_SHA256,
                "verified": self.source_artifact.sha256 == OFFICIAL_LONGMEMEVAL_S_SHA256,
            },
            "source_artifact": self.source_artifact.as_dict(),
            "parsed_records": self.parsed_records.as_dict(),
            "questions": [item.content_free_binding() for item in self.questions],
            "sessions": [item.content_free_binding() for item in self.sessions],
            "turns": [item.content_free_binding() for item in self.turns],
        }

    @property
    def projection_sha256(self) -> str:
        return self._cached_projection_sha256

    def fingerprint(self) -> dict[str, Any]:
        """Compact content-free fingerprint for binding a later experiment run."""

        question_ids = [item.question_id for item in self.questions]
        session_keys = [
            [item.question_id, item.session_position, item.parent_session_id]
            for item in self.sessions
        ]
        turn_ids = [list(item.turn_id.as_tuple()) for item in self.turns]
        documents = [item.serialized_document_utf8.as_dict() for item in self.turns]
        return {
            "artifact_type": PROJECTION_ARTIFACT_TYPE,
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "serializer_version": self.serializer_version,
            "timestamp_convention": self.timestamp_convention,
            "source_artifact": self.source_artifact.as_dict(),
            "parsed_records": self.parsed_records.as_dict(),
            "question_count": len(self.questions),
            "session_count": len(self.sessions),
            "turn_count": len(self.turns),
            "question_ids_sha256": _sha256_json(question_ids),
            "session_keys_sha256": _sha256_json(session_keys),
            "turn_ids_sha256": _sha256_json(turn_ids),
            "turn_documents_sha256": _sha256_json(documents),
            "projection_sha256": self.projection_sha256,
        }

    def content_free_manifest(self, *, repo_root: Path | None = None) -> dict[str, Any]:
        """Full replay manifest; raw questions, answers, and documents are omitted."""

        payload = {
            **self._projection_payload(),
            "fingerprint": self.fingerprint(),
            "implementation": implementation_fingerprint(repo_root=repo_root),
        }
        return {**payload, "manifest_sha256": _sha256_json(payload)}


def _duplicate_key_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LongMemEvalTurnProjectionError(f"source JSON repeats object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise LongMemEvalTurnProjectionError(f"source JSON contains forbidden constant {value}")


def _parse_source_json(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise LongMemEvalTurnProjectionError("source artifact must be UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_key_object,
            parse_constant=_reject_json_constant,
        )
    except LongMemEvalTurnProjectionError:
        raise
    except json.JSONDecodeError as exc:
        raise LongMemEvalTurnProjectionError(
            f"source artifact is not strict JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, list) or not value:
        raise LongMemEvalTurnProjectionError(
            "source artifact must contain a non-empty JSON array of records"
        )
    if any(not isinstance(record, dict) for record in value):
        raise LongMemEvalTurnProjectionError("every source record must be a JSON object")
    return value


def _compile_records(
    records: list[dict[str, Any]],
    *,
    source_artifact: SourceArtifactBinding,
) -> TurnProjectionCorpus:
    questions: list[QuestionBinding] = []
    sessions_out: list[SessionBinding] = []
    turns_out: list[TurnProjection] = []
    seen_question_ids: set[str] = set()
    seen_turn_ids: set[LongMemEvalTurnId] = set()

    for question_position, record in enumerate(records):
        question_id = _stable_identifier(
            record.get("question_id"), label=f"record[{question_position}].question_id"
        )
        if question_id in seen_question_ids:
            raise LongMemEvalTurnProjectionError(
                f"source contains duplicate question_id {question_id!r}"
            )
        seen_question_ids.add(question_id)

        session_ids = record.get("haystack_session_ids")
        session_dates = record.get("haystack_dates")
        sessions = record.get("haystack_sessions")
        if not all(isinstance(value, list) for value in (session_ids, session_dates, sessions)):
            raise LongMemEvalTurnProjectionError(
                f"record[{question_position}] haystack IDs, dates, and sessions must be lists"
            )
        assert isinstance(session_ids, list)
        assert isinstance(session_dates, list)
        assert isinstance(sessions, list)
        if not session_ids:
            raise LongMemEvalTurnProjectionError(
                f"record[{question_position}] must contain at least one haystack session"
            )
        if not len(session_ids) == len(session_dates) == len(sessions):
            raise LongMemEvalTurnProjectionError(
                f"record[{question_position}] haystack arrays are misaligned"
            )

        source_record_bytes = _canonical_json_bytes(record)
        source_record = ByteDigest.from_bytes(source_record_bytes)
        question_turn_count = 0
        for session_position, (session_id_raw, date_raw, turns_raw) in enumerate(
            zip(session_ids, session_dates, sessions, strict=True)
        ):
            session_id = _stable_identifier(
                session_id_raw,
                label=(f"record[{question_position}].haystack_session_ids[{session_position}]"),
            )
            session_date, session_date_utc = _strict_timestamp(
                date_raw,
                label=f"record[{question_position}].haystack_dates[{session_position}]",
            )
            if not isinstance(turns_raw, list) or not turns_raw:
                raise LongMemEvalTurnProjectionError(
                    f"record[{question_position}].haystack_sessions[{session_position}] "
                    "must be a non-empty list"
                )

            session_payload = {
                "parent_session_id": session_id,
                "parent_session_date": session_date,
                "turns": turns_raw,
            }
            source_session = ByteDigest.from_bytes(_canonical_json_bytes(session_payload))
            sessions_out.append(
                SessionBinding(
                    question_id=question_id,
                    session_position=session_position,
                    parent_session_id=session_id,
                    parent_session_date=session_date,
                    parent_session_date_utc=session_date_utc,
                    turns=len(turns_raw),
                    source_session=source_session,
                )
            )

            for turn_position, turn_raw in enumerate(turns_raw):
                turn_label = (
                    f"record[{question_position}].haystack_sessions[{session_position}]"
                    f"[{turn_position}]"
                )
                if not isinstance(turn_raw, dict):
                    raise LongMemEvalTurnProjectionError(f"{turn_label} must be an object")
                if "role" not in turn_raw:
                    raise LongMemEvalTurnProjectionError(f"{turn_label}.role is missing")
                if "content" not in turn_raw:
                    raise LongMemEvalTurnProjectionError(f"{turn_label}.content is missing")
                role = _stable_identifier(turn_raw["role"], label=f"{turn_label}.role")
                content = turn_raw["content"]
                if not isinstance(content, str):
                    raise LongMemEvalTurnProjectionError(f"{turn_label}.content must be a string")
                try:
                    content_bytes = content.encode("utf-8")
                except UnicodeError as exc:
                    raise LongMemEvalTurnProjectionError(
                        f"{turn_label}.content must be valid UTF-8 text"
                    ) from exc

                turn_id = LongMemEvalTurnId(
                    question_id=question_id,
                    session_position=session_position,
                    turn_position=turn_position,
                )
                if turn_id in seen_turn_ids:
                    raise LongMemEvalTurnProjectionError(
                        f"source produces duplicate turn ID {turn_id.as_tuple()!r}"
                    )
                seen_turn_ids.add(turn_id)
                serialized = _document_bytes(
                    timestamp=session_date,
                    role=role,
                    content=content,
                )
                turns_out.append(
                    TurnProjection(
                        turn_id=turn_id,
                        parent_session_id=session_id,
                        parent_session_date=session_date,
                        parent_session_date_utc=session_date_utc,
                        role=role,
                        original_content=content,
                        serialized_text=serialized.decode("utf-8"),
                        source_record=source_record,
                        source_session=source_session,
                        source_turn=ByteDigest.from_bytes(_canonical_json_bytes(turn_raw)),
                        original_content_utf8=ByteDigest.from_bytes(content_bytes),
                        serialized_document_utf8=ByteDigest.from_bytes(serialized),
                    )
                )
                question_turn_count += 1

        questions.append(
            QuestionBinding(
                question_id=question_id,
                question_position=question_position,
                sessions=len(sessions),
                turns=question_turn_count,
                source_record=source_record,
            )
        )

    if len(seen_turn_ids) != len(turns_out):
        raise LongMemEvalTurnProjectionError("turn projection contains duplicate IDs")
    parsed_bytes = _canonical_json_bytes(records)
    return TurnProjectionCorpus(
        source_artifact=source_artifact,
        parsed_records=ByteDigest.from_bytes(parsed_bytes),
        questions=tuple(questions),
        sessions=tuple(sessions_out),
        turns=tuple(turns_out),
    )


def compile_dataset_bytes(
    raw: bytes,
    *,
    expected_sha256: str,
    source_label: str = "longmemeval_s_cleaned.json",
) -> TurnProjectionCorpus:
    """Compile verified source bytes; a caller-supplied digest is mandatory."""

    if not isinstance(raw, bytes) or not raw:
        raise LongMemEvalTurnProjectionError("source artifact must be non-empty bytes")
    expected = _checked_digest(expected_sha256, label="expected source artifact digest")
    observed = _sha256_bytes(raw)
    if observed != expected:
        raise LongMemEvalTurnProjectionError(
            "source artifact digest differs from the required SHA-256"
        )
    label = _stable_identifier(source_label, label="source artifact label")
    records = _parse_source_json(raw)
    return _compile_records(
        records,
        source_artifact=SourceArtifactBinding(label=label, bytes=len(raw), sha256=observed),
    )


def compile_dataset_file(
    path: Path,
    *,
    expected_sha256: str,
    source_label: str | None = None,
) -> TurnProjectionCorpus:
    """Read and compile one local artifact; network acquisition is out of scope."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise LongMemEvalTurnProjectionError(
            "source path must be an existing regular file and cannot be a symbolic link"
        )
    try:
        raw = candidate.read_bytes()
    except OSError as exc:
        raise LongMemEvalTurnProjectionError("cannot read source artifact") from exc
    return compile_dataset_bytes(
        raw,
        expected_sha256=expected_sha256,
        source_label=source_label or candidate.name,
    )


def compile_official_longmemeval_s(path: Path) -> TurnProjectionCorpus:
    """Compile only the pinned cleaned LongMemEval-S artifact from a local path."""

    return compile_dataset_file(
        path,
        expected_sha256=OFFICIAL_LONGMEMEVAL_S_SHA256,
        source_label="longmemeval_s_cleaned.json",
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_fingerprint(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Hash every local file that defines the projection protocol."""

    root = (repo_root or repository_root()).resolve()
    files: dict[str, str] = {}
    for relative in sorted(IMPLEMENTATION_FILES):
        path = root
        unsafe = False
        for part in Path(relative).parts:
            path /= part
            unsafe = unsafe or path.is_symlink()
        if unsafe or not path.is_file():
            raise LongMemEvalTurnProjectionError(
                f"turn projection implementation file is missing or unsafe: {relative}"
            )
        try:
            files[relative] = _sha256_bytes(path.read_bytes())
        except OSError as exc:
            raise LongMemEvalTurnProjectionError(
                f"cannot read turn projection implementation file: {relative}"
            ) from exc
    return {"tree_sha256": _sha256_json(files), "files_sha256": files}


__all__ = [
    "IMPLEMENTATION_FILES",
    "OFFICIAL_LONGMEMEVAL_S_SHA256",
    "PROJECTION_ARTIFACT_TYPE",
    "PROJECTION_SCHEMA_VERSION",
    "TIMESTAMP_CONVENTION",
    "TURN_SERIALIZER_VERSION",
    "ByteDigest",
    "LongMemEvalTurnId",
    "LongMemEvalTurnProjectionError",
    "QuestionBinding",
    "SessionBinding",
    "SourceArtifactBinding",
    "TurnProjection",
    "TurnProjectionCorpus",
    "compile_dataset_bytes",
    "compile_dataset_file",
    "compile_official_longmemeval_s",
    "implementation_fingerprint",
    "repository_root",
]
