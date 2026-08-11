"""Deterministic E7 top-50 expansion and LazyMem-shaped local windowing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.integrations.longmemeval_e1 import (
    ARTIFACT_TYPE as E1_ARTIFACT_TYPE,
)
from benchmarks.integrations.longmemeval_e1 import (
    E1_POOL_CAP,
    E1Cell,
    E1SelectionError,
    E1SelectionResult,
    PoolScoreObservation,
    validate_e1b_result,
)
from benchmarks.integrations.longmemeval_official_preflight import (
    OFFICIAL_DATASET_REQUIREMENT,
    LongMemEvalOfficialPreflightError,
    RunPreflightManifest,
    freeze_official_preflight,
    freeze_pinned_preflight,
)
from benchmarks.integrations.longmemeval_turn_retrieval import TurnFusionResult
from benchmarks.integrations.longmemeval_turns import (
    TurnProjection,
    TurnProjectionCorpus,
)

from .contracts import (
    ARTIFACT_TYPE,
    CONTEXT_RADIUS,
    MAX_WINDOW_MESSAGES,
    PAPER_REPOSITORY,
    PAPER_REPOSITORY_COMMIT,
    PAPER_URL,
    PAPER_WINDOW_PROTOCOL,
    PROTOCOL_VERSION,
    RETRIEVED_TURNS,
    SCHEMA_VERSION,
    WINDOW_STRIDE,
    QueryConstructionError,
    QueryWindow,
    RetrievedTurnPool,
    WindowMessage,
    WindowPosition,
    canonical_json_bytes,
    positive_int,
    required_text,
    sha256_bytes,
    sha256_json,
    sha256_utf8,
)

CONSTRUCTOR_SYSTEM_PROMPT_VERSION = "swarmbrain-e7-constructor-system-v2"
CONSTRUCTOR_USER_PAYLOAD_VERSION = "swarmbrain-e7-constructor-user-json-v2"
CONSTRUCTOR_SYSTEM_PROMPT = (
    "Select query-relevant messages from one chronological memory window. "
    "Return only one JSON object containing one KEEP or DROP decision per "
    "message in order. Cite half-open UTF-8 byte spans from that same message. "
    "KEEP output must remain faithful to cited spans; do not add unsupported facts."
)
CONSTRUCTOR_SYSTEM_PROMPT_SHA256 = sha256_utf8(CONSTRUCTOR_SYSTEM_PROMPT)
CONSTRUCTOR_USER_PROMPT_SHA256 = sha256_utf8(CONSTRUCTOR_USER_PAYLOAD_VERSION)

_WINDOW_BATCH_TRACE_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "paper_transfer",
        "parameters",
        "source",
        "preflight_manifest_sha256",
        "query",
        "constructor_prompts",
        "implementation",
        "source_refreeze",
        "accounting",
        "windows",
        "windows_sha256",
        "claims",
    }
)


_AUTHORITY_SEAL = object()


def _canonical_equal(actual: Any, expected: Any) -> bool:
    return canonical_json_bytes(actual) == canonical_json_bytes(expected)


class _QueryWindowAuthority:
    """Process-local, non-replaceable receipt created by the validated builder."""

    __slots__ = (
        "_current_date_sha256",
        "_current_date_utf8_bytes",
        "_authoritative_question_turn_count",
        "_pool",
        "_preflight",
        "_query_sha256",
        "_query_utf8_bytes",
        "_source_sha256",
        "_source_utf8_bytes",
        "_windows_sha256",
    )

    def __init__(
        self,
        *,
        source_bytes: bytes,
        preflight: RunPreflightManifest,
        pool: RetrievedTurnPool,
        query: str,
        current_date: str,
        authoritative_question_turn_count: int,
        windows: tuple[QueryWindow, ...],
        seal: object,
    ) -> None:
        if seal is not _AUTHORITY_SEAL:
            raise QueryConstructionError(
                "query-window authority can only be issued by the validated builder"
            )
        source_sha256 = sha256_bytes(source_bytes)
        if (
            preflight.dataset.source_sha256 != source_sha256
            or preflight.source_artifact_utf8_bytes != len(source_bytes)
            or pool.source_artifact_sha256 != source_sha256
            or pool.projection_sha256 != preflight.projection_sha256
        ):
            raise QueryConstructionError(
                "query-window authority source, projection, and preflight disagree"
            )
        object.__setattr__(self, "_source_sha256", source_sha256)
        object.__setattr__(self, "_source_utf8_bytes", len(source_bytes))
        object.__setattr__(self, "_preflight", preflight)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_query_sha256", sha256_utf8(query))
        object.__setattr__(self, "_query_utf8_bytes", len(query.encode("utf-8")))
        object.__setattr__(self, "_current_date_sha256", sha256_utf8(current_date))
        object.__setattr__(
            self,
            "_current_date_utf8_bytes",
            len(current_date.encode("utf-8")),
        )
        object.__setattr__(
            self,
            "_authoritative_question_turn_count",
            positive_int(
                authoritative_question_turn_count,
                label="query-window authority question turn count",
            ),
        )
        object.__setattr__(
            self,
            "_windows_sha256",
            sha256_json([window.content_free_binding() for window in windows]),
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("query-window authority is immutable")

    def validate(
        self,
        *,
        pool: RetrievedTurnPool,
        preflight: RunPreflightManifest,
        query: str,
        current_date: str,
        authoritative_question_turn_count: int,
        windows: tuple[QueryWindow, ...],
    ) -> None:
        if pool is not self._pool or preflight is not self._preflight:
            raise QueryConstructionError(
                "window batch pool or preflight differs from its sealed builder authority"
            )
        query_bytes = query.encode("utf-8")
        current_date_bytes = current_date.encode("utf-8")
        if (
            sha256_bytes(query_bytes) != self._query_sha256
            or len(query_bytes) != self._query_utf8_bytes
            or sha256_bytes(current_date_bytes) != self._current_date_sha256
            or len(current_date_bytes) != self._current_date_utf8_bytes
        ):
            raise QueryConstructionError(
                "window batch query or current_date differs from its sealed builder authority"
            )
        if (
            authoritative_question_turn_count != self._authoritative_question_turn_count
            or sha256_json([window.content_free_binding() for window in windows])
            != self._windows_sha256
        ):
            raise QueryConstructionError(
                "window batch turn count or window set differs from its sealed builder authority"
            )

    def content_free_receipt(self) -> dict[str, Any]:
        return {
            "authority": "process-local-sealed-builder-receipt",
            "source_sha256": self._source_sha256,
            "source_utf8_bytes": self._source_utf8_bytes,
            "preflight_manifest_sha256": self._preflight.manifest_sha256,
            "freezer": "official" if self._preflight.dataset.official else "pinned-synthetic",
            "e1b_selection_trace_sha256": self._pool.selection_trace_sha256,
            "e1b_replayed_from_source_evidence": True,
            "authoritative_question_turn_count": self._authoritative_question_turn_count,
            "windows_sha256": self._windows_sha256,
        }


IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "uv.lock",
    "benchmarks/integrations/longmemeval_turns/compiler.py",
    "benchmarks/integrations/longmemeval_e1/contracts.py",
    "benchmarks/integrations/longmemeval_e1/selection.py",
    "benchmarks/integrations/longmemeval_official_preflight/contracts.py",
    "benchmarks/integrations/longmemeval_official_preflight/preflight.py",
    "benchmarks/integrations/longmemeval_query_construction/__init__.py",
    "benchmarks/integrations/longmemeval_query_construction/contracts.py",
    "benchmarks/integrations/longmemeval_query_construction/windowing.py",
    "benchmarks/integrations/longmemeval_query_construction/construction.py",
    "benchmarks/integrations/longmemeval_query_construction/receipts.py",
)


def implementation_fingerprint(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Bind E7 evidence to every local file that defines its semantics."""

    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise QueryConstructionError(f"E7 implementation file is missing or unsafe: {relative}")
        files[relative] = sha256_bytes(path.read_bytes())
    return {
        "files_sha256": files,
        "tree_sha256": sha256_json(files),
    }


@dataclass(frozen=True, slots=True)
class QueryWindowBatch:
    """Content-bearing windows plus a content-free immutable trace."""

    pool: RetrievedTurnPool
    preflight: RunPreflightManifest
    query: str
    current_date: str
    authoritative_question_turn_count: int
    windows: tuple[QueryWindow, ...]
    _authority: _QueryWindowAuthority = field(repr=False, compare=False)
    _trace_canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.pool, RetrievedTurnPool):
            raise QueryConstructionError("window batch pool is invalid")
        if not isinstance(self.preflight, RunPreflightManifest):
            raise QueryConstructionError("window batch preflight is invalid")
        required_text(self.query, label="window batch query")
        required_text(self.current_date, label="window batch current_date")
        if not isinstance(self._authority, _QueryWindowAuthority):
            raise QueryConstructionError("window batch requires a sealed builder authority")
        positive_int(
            self.authoritative_question_turn_count,
            label="window batch authoritative question turn count",
        )
        if sha256_utf8(self.query) != self.pool.query_sha256:
            raise QueryConstructionError("window batch query differs from the E1-B query")
        if self.preflight.projection_sha256 != self.pool.projection_sha256:
            raise QueryConstructionError(
                "window batch preflight projection differs from its retrieved pool"
            )
        if self.preflight.dataset.source_sha256 != self.pool.source_artifact_sha256:
            raise QueryConstructionError(
                "window batch preflight source differs from its retrieved pool"
            )
        matching_cases = tuple(
            case for case in self.preflight.cases if case.question_id == self.pool.question_id
        )
        if len(matching_cases) != 1:
            raise QueryConstructionError(
                "window batch question is missing or repeated in its preflight"
            )
        case = matching_cases[0]
        query_bytes = self.query.encode("utf-8")
        current_date_bytes = self.current_date.encode("utf-8")
        if case.question_sha256 != sha256_bytes(query_bytes) or case.question_utf8_bytes != len(
            query_bytes
        ):
            raise QueryConstructionError(
                "window batch query differs from its authoritative preflight case"
            )
        if case.current_date_sha256 != sha256_bytes(
            current_date_bytes
        ) or case.current_date_utf8_bytes != len(current_date_bytes):
            raise QueryConstructionError(
                "window batch current_date differs from its authoritative preflight case"
            )
        source_refreeze = self._authority.content_free_receipt()
        if not isinstance(self.windows, tuple) or not self.windows:
            raise QueryConstructionError("window batch must contain frozen windows")
        if any(not isinstance(window, QueryWindow) for window in self.windows):
            raise QueryConstructionError("window batch contains an invalid window")
        expected_date_sha256 = sha256_utf8(self.current_date)
        if any(
            (
                window.question_id != self.pool.question_id
                or window.query_sha256 != self.pool.query_sha256
                or window.current_date_sha256 != expected_date_sha256
            )
            for window in self.windows
        ):
            raise QueryConstructionError("window query/current-date binding differs from its batch")
        self._authority.validate(
            pool=self.pool,
            preflight=self.preflight,
            query=self.query,
            current_date=self.current_date,
            authoritative_question_turn_count=self.authoritative_question_turn_count,
            windows=self.windows,
        )
        indexes = [window.window_index for window in self.windows]
        if indexes != list(range(len(self.windows))):
            raise QueryConstructionError("window indexes must be contiguous and zero-based")
        try:
            parsed = json.loads(self._trace_canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise QueryConstructionError("window batch trace is invalid JSON") from exc
        if canonical_json_bytes(parsed).decode("utf-8") != self._trace_canonical_json:
            raise QueryConstructionError("window batch trace is not canonical JSON")
        if set(parsed) != _WINDOW_BATCH_TRACE_FIELDS:
            raise QueryConstructionError("window batch trace fields differ from exact schema")
        if (
            parsed.get("artifact_type") != ARTIFACT_TYPE
            or parsed.get("schema_version") != SCHEMA_VERSION
            or parsed.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise QueryConstructionError("window batch protocol identity drifted")
        if not _canonical_equal(parsed.get("source"), self.pool.content_free_binding()):
            raise QueryConstructionError("window batch source binding drifted")
        if not _canonical_equal(
            parsed.get("query"),
            {
                "question_id": self.pool.question_id,
                "query_sha256": self.pool.query_sha256,
                "current_date_sha256": sha256_utf8(self.current_date),
                "current_date_source": "source-byte-verified-preflight",
            },
        ):
            raise QueryConstructionError("window batch query binding drifted")
        if not _canonical_equal(
            parsed.get("parameters"),
            {
                "retrieved_turns": RETRIEVED_TURNS,
                "context_radius": CONTEXT_RADIUS,
                "maximum_window_messages": MAX_WINDOW_MESSAGES,
                "stride": WINDOW_STRIDE,
                "overlap_messages_for_sliced_windows": MAX_WINDOW_MESSAGES - WINDOW_STRIDE,
                "grouping_gap": 2 * CONTEXT_RADIUS + 1,
                "global_message_index_base": 0,
            },
        ):
            raise QueryConstructionError("window batch parameters drifted")
        if not _canonical_equal(
            parsed.get("paper_transfer"),
            {
                "source": PAPER_URL,
                "repository": PAPER_REPOSITORY,
                "repository_commit": PAPER_REPOSITORY_COMMIT,
                "window_protocol": PAPER_WINDOW_PROTOCOL,
                "paper_reproduction_claimed": False,
            },
        ):
            raise QueryConstructionError("window batch paper-transfer claim drifted")
        window_bindings = [window.content_free_binding() for window in self.windows]
        if not _canonical_equal(parsed.get("windows"), window_bindings):
            raise QueryConstructionError("window batch trace rows drifted")
        if parsed.get("windows_sha256") != sha256_json(window_bindings):
            raise QueryConstructionError("window batch differs from its trace")
        if not _canonical_equal(
            parsed.get("constructor_prompts"),
            {
                "system_version": CONSTRUCTOR_SYSTEM_PROMPT_VERSION,
                "system_sha256": CONSTRUCTOR_SYSTEM_PROMPT_SHA256,
                "user_payload_version": CONSTRUCTOR_USER_PAYLOAD_VERSION,
                "user_payload_sha256": CONSTRUCTOR_USER_PROMPT_SHA256,
            },
        ):
            raise QueryConstructionError("window batch constructor prompt binding drifted")
        if not _canonical_equal(parsed.get("implementation"), implementation_fingerprint()):
            raise QueryConstructionError("window batch implementation fingerprint drifted")
        if parsed.get("preflight_manifest_sha256") != self.preflight.manifest_sha256:
            raise QueryConstructionError("window batch preflight binding drifted")
        if not _canonical_equal(parsed.get("source_refreeze"), source_refreeze):
            raise QueryConstructionError("window batch source-refreeze trace binding drifted")
        accounting = parsed.get("accounting")
        if not isinstance(accounting, dict):
            raise QueryConstructionError("window batch accounting is missing")
        core_ids = {
            message.turn.turn_id
            for window in self.windows
            for message in window.messages
            if message.position is WindowPosition.CORE
        }
        expected_accounting = {
            "authoritative_question_turns": self.authoritative_question_turn_count,
            "retrieved_turns": len(self.pool.turns),
            "segments": len({window.segment_index for window in self.windows}),
            "windows": len(self.windows),
            "window_message_appearances": sum(len(window.messages) for window in self.windows),
            "unique_window_messages": len(
                {message.turn.turn_id for window in self.windows for message in window.messages}
            ),
            "retrieved_turns_missing_from_core": len(
                {turn.turn_id for turn in self.pool.turns} - core_ids
            ),
            "local_model_calls": 0,
            "local_tokenizer_calls": 0,
            "local_reader_calls": 0,
            "local_judge_calls": 0,
            "local_database_calls": 0,
            "local_network_calls": 0,
        }
        if not _canonical_equal(accounting, expected_accounting):
            raise QueryConstructionError("window batch accounting drifted")
        if not _canonical_equal(
            parsed.get("claims"),
            {
                "gold_fields_present_in_constructor_input": False,
                "question_type_present_in_constructor_input": False,
                "current_date_matches_dataset_proven": True,
                "constructor_executed": False,
                "qa_improvement_proven": False,
                "serving_eligibility_proven": False,
            },
        ):
            raise QueryConstructionError("window batch claims drifted")

    @property
    def trace_sha256(self) -> str:
        return sha256_utf8(self._trace_canonical_json)

    def content_free_trace(self) -> dict[str, Any]:
        payload = json.loads(self._trace_canonical_json)
        return {**payload, "trace_sha256": self.trace_sha256}


def _validate_selection_trace(
    selection: E1SelectionResult,
    *,
    corpus: TurnProjectionCorpus,
    query: str,
) -> dict[str, Any]:
    if not isinstance(selection, E1SelectionResult):
        raise QueryConstructionError("E7 source must be an E1SelectionResult")
    if selection.cell is not E1Cell.CROSS_ENCODER:
        raise QueryConstructionError("E7 requires the frozen E1-B CrossEncoder ranking")
    if selection.source_pool_count != E1_POOL_CAP or len(selection.candidates) != E1_POOL_CAP:
        raise QueryConstructionError("E7 requires the complete frozen 128-candidate E1-B output")
    trace = selection.content_free_trace()
    if trace.get("trace_sha256") != selection.trace_sha256:
        raise QueryConstructionError("E1-B trace digest is internally inconsistent")
    if trace.get("artifact_type") != E1_ARTIFACT_TYPE:
        raise QueryConstructionError("E7 source is not an E1 selection trace")
    if trace.get("cell") != E1Cell.CROSS_ENCODER.value:
        raise QueryConstructionError("E1-B result and trace cell disagree")
    source = trace.get("source_e1a")
    output = trace.get("selection")
    if not isinstance(source, dict) or not isinstance(output, dict):
        raise QueryConstructionError("E1-B trace is missing source or selection bindings")
    if source.get("question_id") != selection.question_id:
        raise QueryConstructionError("E1-B question differs from its trace")
    if source.get("query_sha256") != sha256_utf8(query):
        raise QueryConstructionError("query text differs from the E1-B query binding")
    if source.get("turn_corpus_projection_sha256") != corpus.projection_sha256:
        raise QueryConstructionError("E1-B trace is bound to a different turn corpus")
    if source.get("fixed_pool_count") != E1_POOL_CAP:
        raise QueryConstructionError("E1-B trace does not bind the full source pool")
    expected_bindings = [candidate.content_free_binding() for candidate in selection.candidates]
    if output.get("output_candidates") != expected_bindings:
        raise QueryConstructionError("E1-B candidates differ from their trace")
    if output.get("output_candidates_sha256") != sha256_json(expected_bindings):
        raise QueryConstructionError("E1-B candidate digest is inconsistent")
    ranks = [candidate.selected_rank for candidate in selection.candidates]
    if ranks != list(range(1, len(selection.candidates) + 1)):
        raise QueryConstructionError("E1-B selected ranks are not contiguous")
    return trace


def build_retrieved_turn_pool(
    corpus: TurnProjectionCorpus,
    selection: E1SelectionResult,
    *,
    source: TurnFusionResult,
    cross_encoder: PoolScoreObservation,
    query: str,
) -> RetrievedTurnPool:
    """Replay E1-B source evidence and bind its first 50 immutable turns."""

    if not isinstance(corpus, TurnProjectionCorpus):
        raise QueryConstructionError("E7 requires a TurnProjectionCorpus")
    required_text(query, label="E7 query")
    try:
        validate_e1b_result(
            selection,
            source=source,
            cross_encoder=cross_encoder,
        )
    except E1SelectionError as exc:
        raise QueryConstructionError(
            "E7 E1-B result failed deterministic replay from source evidence"
        ) from exc
    _validate_selection_trace(selection, corpus=corpus, query=query)

    corpus_by_id = corpus.by_id()
    turns: list[TurnProjection] = []
    for candidate in selection.candidates[:RETRIEVED_TURNS]:
        authoritative = corpus_by_id.get(candidate.turn_id)
        if authoritative is None or authoritative != candidate.turn:
            raise QueryConstructionError(
                "E1-B candidate is missing from or differs from the authoritative corpus"
            )
        turns.append(authoritative)
    return RetrievedTurnPool(
        question_id=selection.question_id,
        query_sha256=sha256_utf8(query),
        source_artifact_sha256=corpus.source_artifact.sha256,
        projection_sha256=corpus.projection_sha256,
        selection_trace_sha256=selection.trace_sha256,
        turns=tuple(turns),
    )


def _question_turns(
    corpus: TurnProjectionCorpus,
    *,
    question_id: str,
) -> tuple[TurnProjection, ...]:
    turns = tuple(turn for turn in corpus.turns if turn.turn_id.question_id == question_id)
    if not turns:
        raise QueryConstructionError("E7 question has no authoritative turns")
    return turns


def _validate_preflight(
    corpus: TurnProjectionCorpus,
    pool: RetrievedTurnPool,
    preflight: RunPreflightManifest,
    *,
    query: str,
    current_date: str,
) -> None:
    if not isinstance(preflight, RunPreflightManifest):
        raise QueryConstructionError("E7 requires a frozen official-run preflight")
    if preflight.projection_sha256 != corpus.projection_sha256:
        raise QueryConstructionError("E7 preflight is bound to a different turn projection")
    if preflight.dataset.source_sha256 != corpus.source_artifact.sha256:
        raise QueryConstructionError("E7 preflight is bound to a different source artifact")
    if preflight.source_artifact_utf8_bytes != corpus.source_artifact.bytes:
        raise QueryConstructionError("E7 preflight source byte count is inconsistent")
    matches = [case for case in preflight.cases if case.question_id == pool.question_id]
    if len(matches) != 1:
        raise QueryConstructionError("E7 question is missing or repeated in the preflight")
    case = matches[0]
    questions = [item for item in corpus.questions if item.question_id == pool.question_id]
    if len(questions) != 1:
        raise QueryConstructionError("E7 question is missing or repeated in the turn corpus")
    question_binding = questions[0]
    if (
        case.source_record_sha256 != question_binding.source_record.sha256
        or case.source_record_utf8_bytes != question_binding.source_record.bytes
    ):
        raise QueryConstructionError("E7 preflight source-record binding is inconsistent")
    query_bytes = query.encode("utf-8")
    if case.question_sha256 != sha256_bytes(query_bytes) or case.question_utf8_bytes != len(
        query_bytes
    ):
        raise QueryConstructionError("E7 query differs from the authoritative preflight case")
    current_date_bytes = current_date.encode("utf-8")
    if case.current_date_sha256 != sha256_bytes(
        current_date_bytes
    ) or case.current_date_utf8_bytes != len(current_date_bytes):
        raise QueryConstructionError(
            "E7 current_date differs from the authoritative preflight case"
        )


def _refreeze_authoritative_preflight(
    source_bytes: bytes,
    preflight: RunPreflightManifest,
) -> RunPreflightManifest:
    """Rebuild the supplied manifest from source bytes before trusting a case.

    ``RunPreflightManifest`` is immutable, but ``dataclasses.replace`` can still
    create a different valid instance.  Source-derived question/date bindings
    therefore become authoritative only after the whole manifest is reproduced
    from the caller-supplied pinned corpus bytes.
    """

    if not isinstance(preflight, RunPreflightManifest):
        raise QueryConstructionError("E7 requires a frozen official-run preflight")
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise QueryConstructionError("E7 authoritative source must be non-empty bytes")

    dataset = preflight.dataset
    official_identity = (
        OFFICIAL_DATASET_REQUIREMENT.name,
        OFFICIAL_DATASET_REQUIREMENT.source_label,
        OFFICIAL_DATASET_REQUIREMENT.source_sha256,
        OFFICIAL_DATASET_REQUIREMENT.question_count,
    )
    supplied_identity = (
        dataset.name,
        dataset.source_label,
        dataset.source_sha256,
        dataset.question_count,
    )
    if not dataset.official and supplied_identity == official_identity:
        raise QueryConstructionError(
            "E7 refuses a nonofficial downgrade of the pinned LongMemEval-S corpus"
        )

    try:
        if dataset.official:
            rebuilt = freeze_official_preflight(
                source_bytes,
                tokenizer=preflight.tokenizer,
            )
        else:
            rebuilt = freeze_pinned_preflight(
                source_bytes,
                dataset=dataset,
                tokenizer=preflight.tokenizer,
            )
    except LongMemEvalOfficialPreflightError as exc:
        raise QueryConstructionError(
            "E7 authoritative source bytes failed exact preflight refreeze"
        ) from exc
    if rebuilt != preflight:
        raise QueryConstructionError(
            "E7 supplied preflight differs from the manifest rebuilt from authoritative source bytes"
        )
    return rebuilt


def _group_retrieved(
    retrieved_by_time: list[TurnProjection],
) -> list[list[TurnProjection]]:
    groups: list[list[TurnProjection]] = []
    current: list[TurnProjection] = []
    for turn in retrieved_by_time:
        if not current:
            current = [turn]
            continue
        previous = current[-1]
        same_session = previous.turn_id.session_position == turn.turn_id.session_position
        gap = turn.turn_id.turn_position - previous.turn_id.turn_position
        # LazyMem commit af41099 groups centers whose radius-2 local spans
        # overlap or touch: gap <= 2 * radius + 1.
        if same_session and 0 < gap <= (2 * CONTEXT_RADIUS + 1):
            current.append(turn)
        else:
            groups.append(current)
            current = [turn]
    if current:
        groups.append(current)
    return groups


def _slice_window(values: list[TurnProjection]) -> list[list[TurnProjection]]:
    if len(values) <= MAX_WINDOW_MESSAGES:
        return [values]
    chunks: list[list[TurnProjection]] = []
    start = 0
    while start < len(values):
        chunk = values[start : start + MAX_WINDOW_MESSAGES]
        if chunk:
            chunks.append(chunk)
        if start + MAX_WINDOW_MESSAGES >= len(values):
            break
        start += WINDOW_STRIDE
    return chunks


def _window_trace(
    *,
    pool: RetrievedTurnPool,
    preflight: RunPreflightManifest,
    current_date: str,
    source_refreeze: dict[str, Any],
    windows: tuple[QueryWindow, ...],
    full_question_turn_count: int,
) -> str:
    window_bindings = [window.content_free_binding() for window in windows]
    retrieved_ids = {turn.turn_id for turn in pool.turns}
    core_ids = {
        message.turn.turn_id
        for window in windows
        for message in window.messages
        if message.position is WindowPosition.CORE
    }
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "paper_transfer": {
            "source": PAPER_URL,
            "repository": PAPER_REPOSITORY,
            "repository_commit": PAPER_REPOSITORY_COMMIT,
            "window_protocol": PAPER_WINDOW_PROTOCOL,
            "paper_reproduction_claimed": False,
        },
        "parameters": {
            "retrieved_turns": RETRIEVED_TURNS,
            "context_radius": CONTEXT_RADIUS,
            "maximum_window_messages": MAX_WINDOW_MESSAGES,
            "stride": WINDOW_STRIDE,
            "overlap_messages_for_sliced_windows": MAX_WINDOW_MESSAGES - WINDOW_STRIDE,
            "grouping_gap": 2 * CONTEXT_RADIUS + 1,
            "global_message_index_base": 0,
        },
        "source": pool.content_free_binding(),
        "preflight_manifest_sha256": preflight.manifest_sha256,
        "source_refreeze": source_refreeze,
        "query": {
            "question_id": pool.question_id,
            "query_sha256": pool.query_sha256,
            "current_date_sha256": sha256_utf8(current_date),
            "current_date_source": "source-byte-verified-preflight",
        },
        "constructor_prompts": {
            "system_version": CONSTRUCTOR_SYSTEM_PROMPT_VERSION,
            "system_sha256": CONSTRUCTOR_SYSTEM_PROMPT_SHA256,
            "user_payload_version": CONSTRUCTOR_USER_PAYLOAD_VERSION,
            "user_payload_sha256": CONSTRUCTOR_USER_PROMPT_SHA256,
        },
        "implementation": implementation_fingerprint(),
        "accounting": {
            "authoritative_question_turns": full_question_turn_count,
            "retrieved_turns": len(pool.turns),
            "segments": len({window.segment_index for window in windows}),
            "windows": len(windows),
            "window_message_appearances": sum(len(window.messages) for window in windows),
            "unique_window_messages": len(
                {message.turn.turn_id for window in windows for message in window.messages}
            ),
            "retrieved_turns_missing_from_core": len(retrieved_ids - core_ids),
            "local_model_calls": 0,
            "local_tokenizer_calls": 0,
            "local_reader_calls": 0,
            "local_judge_calls": 0,
            "local_database_calls": 0,
            "local_network_calls": 0,
        },
        "windows": window_bindings,
        "windows_sha256": sha256_json(window_bindings),
        "claims": {
            "gold_fields_present_in_constructor_input": False,
            "question_type_present_in_constructor_input": False,
            "current_date_matches_dataset_proven": True,
            "constructor_executed": False,
            "qa_improvement_proven": False,
            "serving_eligibility_proven": False,
        },
    }
    return canonical_json_bytes(payload).decode("utf-8")


def build_query_windows(
    corpus: TurnProjectionCorpus,
    pool: RetrievedTurnPool,
    preflight: RunPreflightManifest,
    *,
    source_bytes: bytes,
    selection: E1SelectionResult,
    e1_source: TurnFusionResult,
    cross_encoder: PoolScoreObservation,
    query: str,
    current_date: str,
) -> QueryWindowBatch:
    """Replay source authorities, then create exact radius-2/8/7 windows."""

    if not isinstance(corpus, TurnProjectionCorpus):
        raise QueryConstructionError("windowing requires a TurnProjectionCorpus")
    if not isinstance(pool, RetrievedTurnPool):
        raise QueryConstructionError("windowing requires a RetrievedTurnPool")
    required_text(query, label="window query")
    required_text(current_date, label="window current_date")
    authoritative_preflight = _refreeze_authoritative_preflight(source_bytes, preflight)
    authoritative_pool = build_retrieved_turn_pool(
        corpus,
        selection,
        source=e1_source,
        cross_encoder=cross_encoder,
        query=query,
    )
    if pool != authoritative_pool:
        raise QueryConstructionError(
            "retrieved pool differs from deterministic replay of E1-B source evidence"
        )
    pool = authoritative_pool
    if sha256_utf8(query) != pool.query_sha256:
        raise QueryConstructionError("window query differs from its retrieved pool")
    if corpus.projection_sha256 != pool.projection_sha256:
        raise QueryConstructionError("retrieved pool is bound to a different projection")
    if corpus.source_artifact.sha256 != pool.source_artifact_sha256:
        raise QueryConstructionError("retrieved pool is bound to a different source artifact")
    _validate_preflight(
        corpus,
        pool,
        authoritative_preflight,
        query=query,
        current_date=current_date,
    )

    all_turns = _question_turns(corpus, question_id=pool.question_id)
    authoritative = {turn.turn_id: turn for turn in all_turns}
    index_by_id = {turn.turn_id: index for index, turn in enumerate(all_turns)}
    rank_by_id = {turn.turn_id: rank for rank, turn in enumerate(pool.turns, start=1)}
    for turn in pool.turns:
        if authoritative.get(turn.turn_id) != turn:
            raise QueryConstructionError("retrieved turn differs from the authoritative corpus")

    retrieved_by_time = sorted(pool.turns, key=lambda turn: index_by_id[turn.turn_id])
    groups = _group_retrieved(retrieved_by_time)
    windows: list[QueryWindow] = []
    for segment_index, group in enumerate(groups):
        session_position = group[0].turn_id.session_position
        session_turns = [
            turn for turn in all_turns if turn.turn_id.session_position == session_position
        ]
        by_position = {turn.turn_id.turn_position: turn for turn in session_turns}
        first = group[0].turn_id.turn_position
        last = group[-1].turn_id.turn_position
        dense = [by_position[position] for position in range(first, last + 1)]
        previous = [
            by_position[position] for position in range(max(0, first - CONTEXT_RADIUS), first)
        ]
        next_values = [
            by_position[position]
            for position in range(
                last + 1,
                min(len(session_turns), last + CONTEXT_RADIUS + 1),
            )
        ]
        expanded = previous + dense + next_values
        dense_ids = {turn.turn_id for turn in dense}
        previous_ids = {turn.turn_id for turn in previous}
        next_ids = {turn.turn_id for turn in next_values}
        base_type = "continuous" if len(group) > 1 else "singleton"
        for chunk in _slice_window(expanded):
            if not any(turn.turn_id in dense_ids for turn in chunk):
                continue
            segment_type = "continuous_sliced" if len(expanded) > MAX_WINDOW_MESSAGES else base_type
            messages: list[WindowMessage] = []
            for turn in chunk:
                if turn.turn_id in dense_ids:
                    position = WindowPosition.CORE
                elif turn.turn_id in previous_ids:
                    position = WindowPosition.PREVIOUS_BRIDGE
                elif turn.turn_id in next_ids:
                    position = WindowPosition.NEXT_BRIDGE
                else:  # pragma: no cover - the partition is constructed above
                    raise QueryConstructionError("window message is outside its segment")
                messages.append(
                    WindowMessage(
                        turn=turn,
                        global_message_index=index_by_id[turn.turn_id],
                        retrieval_rank=rank_by_id.get(turn.turn_id),
                        is_retrieved=turn.turn_id in rank_by_id,
                        position=position,
                    )
                )
            windows.append(
                QueryWindow(
                    question_id=pool.question_id,
                    query_sha256=pool.query_sha256,
                    current_date_sha256=sha256_utf8(current_date),
                    segment_index=segment_index,
                    window_index=len(windows),
                    segment_type=segment_type,
                    messages=tuple(messages),
                )
            )

    retrieved_ids = {turn.turn_id for turn in pool.turns}
    core_ids = {
        message.turn.turn_id
        for window in windows
        for message in window.messages
        if message.position is WindowPosition.CORE
    }
    if retrieved_ids - core_ids:
        raise QueryConstructionError("windowing lost one or more retrieved turns")
    frozen = tuple(windows)
    authority = _QueryWindowAuthority(
        source_bytes=source_bytes,
        preflight=authoritative_preflight,
        pool=pool,
        query=query,
        current_date=current_date,
        authoritative_question_turn_count=len(all_turns),
        windows=frozen,
        seal=_AUTHORITY_SEAL,
    )
    source_refreeze = authority.content_free_receipt()
    return QueryWindowBatch(
        pool=pool,
        preflight=authoritative_preflight,
        query=query,
        current_date=current_date,
        authoritative_question_turn_count=len(all_turns),
        windows=frozen,
        _authority=authority,
        _trace_canonical_json=_window_trace(
            pool=pool,
            preflight=authoritative_preflight,
            current_date=current_date,
            source_refreeze=source_refreeze,
            windows=frozen,
            full_question_turn_count=len(all_turns),
        ),
    )


def constructor_request_bytes(
    batch: QueryWindowBatch,
    window: QueryWindow,
) -> bytes:
    """Render the exact source-safe constructor request for one local window."""

    if not isinstance(batch, QueryWindowBatch):
        raise QueryConstructionError("constructor request requires a QueryWindowBatch")
    if not isinstance(window, QueryWindow) or window not in batch.windows:
        raise QueryConstructionError("constructor request window is outside its batch")
    payload = {
        "system": CONSTRUCTOR_SYSTEM_PROMPT,
        "user_payload_version": CONSTRUCTOR_USER_PAYLOAD_VERSION,
        "query": batch.query,
        "current_date": batch.current_date,
        "window": [
            {
                "timestamp": message.turn.parent_session_date,
                "role": message.turn.role,
                "content": message.turn.original_content,
            }
            for message in window.messages
        ],
        "required_response": {
            "type": "object",
            "fields": ["decisions"],
            "additional_fields": False,
            "decisions": {
                "type": "array",
                "length": len(window.messages),
                "ordered_like_window": True,
                "item_fields": [
                    "op",
                    "style",
                    "compressed_content",
                    "reason",
                    "support_spans",
                ],
                "op_values": ["KEEP", "DROP"],
                "style_values": ["drop", "verbatim", "extractive", "abstractive"],
                "support_span_fields": ["start_byte", "end_byte"],
                "support_span_coordinates": "half-open UTF-8 byte offsets in same message",
                "drop_contract": {
                    "style": "drop",
                    "compressed_content": "",
                    "support_spans": [],
                },
                "extractive_separator": " … ",
            },
        },
    }
    return canonical_json_bytes(payload)


__all__ = [
    "CONSTRUCTOR_SYSTEM_PROMPT",
    "CONSTRUCTOR_SYSTEM_PROMPT_SHA256",
    "CONSTRUCTOR_SYSTEM_PROMPT_VERSION",
    "CONSTRUCTOR_USER_PAYLOAD_VERSION",
    "CONSTRUCTOR_USER_PROMPT_SHA256",
    "QueryWindowBatch",
    "build_query_windows",
    "build_retrieved_turn_pool",
    "constructor_request_bytes",
    "implementation_fingerprint",
]
