"""Pure compiler and validator for a packed LongMemEval-S reader run.

This module consumes local source bytes and already-produced prompt/tokenizer
evidence.  It never executes a tokenizer, reader, model, database, or network
operation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from scripts._longmemeval_common import EMPTY_CONTEXT_NOTE, OFFICIAL_ANSWER_TEMPLATE

from benchmarks.integrations.longmemeval_turn_prompt import (
    ARTIFACT_TYPE as PROMPT_ARTIFACT_TYPE,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    CHAIN_BLOCK_SEPARATOR,
    CHAIN_HEADER_BODY_SEPARATOR,
    CHAIN_HEADER_TEMPLATE,
    CHAIN_TURN_SEPARATOR,
    EMPTY_CONTEXT_NOTE_SHA256,
    HISTORY_SERIALIZER_VERSION,
    LINEAR_TURN_SEPARATOR,
    OFFICIAL_ANSWER_TEMPLATE_SHA256,
    PRIMARY_TOKEN_BUDGET,
    TOKENIZER_PROTOCOL,
    PromptLayout,
    TurnPromptPackingResult,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    PROTOCOL_VERSION as PROMPT_PROTOCOL_VERSION,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    SCHEMA_VERSION as PROMPT_SCHEMA_VERSION,
)
from benchmarks.integrations.longmemeval_turns import (
    LongMemEvalTurnId,
    TurnProjection,
    TurnProjectionCorpus,
    compile_dataset_bytes,
)

from .contracts import (
    OFFICIAL_DATASET_REQUIREMENT,
    DatasetCaseBinding,
    DatasetRequirement,
    ExactTokenizerPin,
    LongMemEvalOfficialPreflightError,
    PreparedRunReceipt,
    RunPreflightManifest,
    canonical_json_bytes,
    checked_integer,
    checked_sha256,
    checked_text,
    sha256_bytes,
    sha256_json,
    sha256_text,
)

_PROVIDER_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}")
_DATE_RE = re.compile(
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2}) "
    r"\((?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

_TRACE_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "classification",
        "production_configuration",
        "packing_policy",
        "budget",
        "reader_prompt",
        "question_input",
        "tokenizer",
        "layout",
        "candidate_order",
        "candidate_order_sha256",
        "candidate_blocks",
        "kept_ids",
        "kept_ids_by_block",
        "dropped_ids",
        "oversized_ids",
        "decisions",
        "decisions_sha256",
        "exact_count_observations",
        "exact_count_observations_sha256",
        "observation_accounting",
        "final_prompt",
        "final_history",
        "claims",
    }
)
_OBSERVATION_FIELDS = frozenset({"sequence", "purpose", "candidate_turn_id", "receipt"})
_RECEIPT_FIELDS = frozenset(
    {
        "request_id",
        "provider_request_id",
        "tokenizer_identity_sha256",
        "query_sha256",
        "prompt_sha256",
        "prompt_utf8_bytes",
        "token_count",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "candidate_turn_id",
        "block_position",
        "position_in_block",
        "selected_before_ids",
        "proposed_ids",
        "singleton_observation_sequence",
        "proposal_observation_sequence",
        "accepted",
        "oversized_alone",
    }
)
_TOKENIZER_EVIDENCE_FIELDS = frozenset(
    {
        "method",
        "provider",
        "exact_model_tokenizer",
        "tokenizer_model",
        "tokenizer_revision",
        "tokenizer_artifact",
        "tokenizer_executable",
        "protocol",
        "response_identity_sha256",
        "observation_accounting",
    }
)
_TOKENIZER_ACCOUNTING_FIELDS = frozenset(
    {
        "source",
        "requests",
        "responses",
        "unique_provider_request_ids",
        "text_characters",
        "text_utf8_bytes",
        "exact_response_identity_verified",
    }
)


@dataclass(frozen=True, slots=True)
class _SourceCaseMaterial:
    question_id: str
    question: str
    current_date: str
    source_record_sha256: str
    source_record_utf8_bytes: int
    question_sha256: str
    question_utf8_bytes: int
    current_date_sha256: str
    current_date_utf8_bytes: int


def _exact_json_equal(actual: Any, expected: Any) -> bool:
    """Compare canonical JSON values without Python's ``True == 1`` aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _strict_records(raw: bytes) -> list[dict[str, Any]]:
    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise LongMemEvalOfficialPreflightError(f"source JSON repeats object key {key!r}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise LongMemEvalOfficialPreflightError(f"source JSON contains non-finite number {value!r}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=reject_constant,
        )
    except LongMemEvalOfficialPreflightError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LongMemEvalOfficialPreflightError(
            "source artifact must be strict UTF-8 JSON"
        ) from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(record, dict) for record in parsed)
    ):
        raise LongMemEvalOfficialPreflightError(
            "source artifact must contain a non-empty array of objects"
        )
    return parsed


def _validate_prompt_sources() -> None:
    if sha256_text(OFFICIAL_ANSWER_TEMPLATE) != OFFICIAL_ANSWER_TEMPLATE_SHA256:
        raise LongMemEvalOfficialPreflightError(
            "official reader template bytes differ from the frozen prompt protocol"
        )
    if sha256_text(EMPTY_CONTEXT_NOTE) != EMPTY_CONTEXT_NOTE_SHA256:
        raise LongMemEvalOfficialPreflightError(
            "empty-context note bytes differ from the frozen prompt protocol"
        )


def _current_date(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise LongMemEvalOfficialPreflightError(f"{label} must be text")
    match = _DATE_RE.fullmatch(value)
    if match is None:
        raise LongMemEvalOfficialPreflightError(f"{label} must use YYYY/MM/DD (Ddd) HH:MM exactly")
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
        raise LongMemEvalOfficialPreflightError(f"{label} is not a valid date") from exc
    expected = (
        f"{parsed.year:04d}/{parsed.month:02d}/{parsed.day:02d} "
        f"({_WEEKDAYS[parsed.weekday()]}) {parsed.hour:02d}:{parsed.minute:02d}"
    )
    if value != expected:
        raise LongMemEvalOfficialPreflightError(f"{label} is not canonically serialized")
    return value


def _source_material(
    raw: bytes,
    *,
    dataset: DatasetRequirement,
) -> tuple[
    TurnProjectionCorpus,
    tuple[_SourceCaseMaterial, ...],
    tuple[DatasetCaseBinding, ...],
]:
    if not isinstance(raw, bytes) or not raw:
        raise LongMemEvalOfficialPreflightError("source artifact must be non-empty bytes")
    if not isinstance(dataset, DatasetRequirement):
        raise LongMemEvalOfficialPreflightError("dataset pin must be DatasetRequirement")
    if sha256_bytes(raw) != dataset.source_sha256:
        raise LongMemEvalOfficialPreflightError(
            "source corpus failed projection: artifact digest differs from its pin"
        )
    records = _strict_records(raw)
    if len(records) != dataset.question_count:
        raise LongMemEvalOfficialPreflightError(
            "source corpus question count differs from the pinned experiment"
        )
    source_cases: list[_SourceCaseMaterial] = []
    for case_index, record in enumerate(records):
        question_id = checked_text(
            record.get("question_id"),
            label=f"record[{case_index}].question_id",
        )
        question = record.get("question")
        if not isinstance(question, str) or not question:
            raise LongMemEvalOfficialPreflightError(
                f"record[{case_index}].question must be non-empty text"
            )
        try:
            question_bytes = question.encode("utf-8")
        except UnicodeError as exc:
            raise LongMemEvalOfficialPreflightError(
                f"record[{case_index}].question must be valid UTF-8"
            ) from exc
        current_date = _current_date(
            record.get("question_date"),
            label=f"record[{case_index}].question_date",
        )
        current_date_bytes = current_date.encode("utf-8")
        source_record_bytes = canonical_json_bytes(record)
        source_cases.append(
            _SourceCaseMaterial(
                question_id=question_id,
                question=question,
                current_date=current_date,
                source_record_sha256=sha256_bytes(source_record_bytes),
                source_record_utf8_bytes=len(source_record_bytes),
                question_sha256=sha256_bytes(question_bytes),
                question_utf8_bytes=len(question_bytes),
                current_date_sha256=sha256_bytes(current_date_bytes),
                current_date_utf8_bytes=len(current_date_bytes),
            )
        )
    del records
    try:
        corpus = compile_dataset_bytes(
            raw,
            expected_sha256=dataset.source_sha256,
            source_label=dataset.source_label,
        )
    except ValueError as exc:
        raise LongMemEvalOfficialPreflightError(f"source corpus failed projection: {exc}") from exc
    if len(corpus.questions) != dataset.question_count:
        raise LongMemEvalOfficialPreflightError(
            "source corpus question count differs from the pinned experiment"
        )
    cases: list[DatasetCaseBinding] = []
    for case_index, (source_case, question_binding) in enumerate(
        zip(source_cases, corpus.questions, strict=True)
    ):
        if source_case.question_id != question_binding.question_id:
            raise LongMemEvalOfficialPreflightError(
                "source record order differs from the compiled turn projection"
            )
        if (
            question_binding.source_record.sha256 != source_case.source_record_sha256
            or question_binding.source_record.bytes != source_case.source_record_utf8_bytes
        ):
            raise LongMemEvalOfficialPreflightError(
                "source record digest differs from the compiled turn projection"
            )
        cases.append(
            DatasetCaseBinding(
                case_index=case_index,
                question_id=source_case.question_id,
                source_record_sha256=question_binding.source_record.sha256,
                source_record_utf8_bytes=question_binding.source_record.bytes,
                question_sha256=source_case.question_sha256,
                question_utf8_bytes=source_case.question_utf8_bytes,
                current_date_sha256=source_case.current_date_sha256,
                current_date_utf8_bytes=source_case.current_date_utf8_bytes,
            )
        )
    return corpus, tuple(source_cases), tuple(cases)


def freeze_pinned_preflight(
    source_bytes: bytes,
    *,
    dataset: DatasetRequirement,
    tokenizer: ExactTokenizerPin,
) -> RunPreflightManifest:
    """Freeze corpus, prompt, budget, coverage, and tokenizer before a run."""

    if not isinstance(tokenizer, ExactTokenizerPin):
        raise LongMemEvalOfficialPreflightError("tokenizer pin must be ExactTokenizerPin")
    _validate_prompt_sources()
    corpus, _, cases = _source_material(source_bytes, dataset=dataset)
    return _manifest_from_material(
        corpus,
        cases=cases,
        dataset=dataset,
        tokenizer=tokenizer,
    )


def _manifest_from_material(
    corpus: TurnProjectionCorpus,
    *,
    cases: tuple[DatasetCaseBinding, ...],
    dataset: DatasetRequirement,
    tokenizer: ExactTokenizerPin,
) -> RunPreflightManifest:
    question_ids = [case.question_id for case in cases]
    return RunPreflightManifest(
        dataset=dataset,
        source_artifact_utf8_bytes=corpus.source_artifact.bytes,
        parsed_records_sha256=corpus.parsed_records.sha256,
        parsed_records_utf8_bytes=corpus.parsed_records.bytes,
        projection_sha256=corpus.projection_sha256,
        question_ids_sha256=sha256_json(question_ids),
        cases=cases,
        tokenizer=tokenizer,
    )


def freeze_official_preflight(
    source_bytes: bytes,
    *,
    tokenizer: ExactTokenizerPin,
) -> RunPreflightManifest:
    """Freeze only the byte-pinned official cleaned 500-question experiment."""

    return freeze_pinned_preflight(
        source_bytes,
        dataset=OFFICIAL_DATASET_REQUIREMENT,
        tokenizer=tokenizer,
    )


def load_preflight_manifest_artifact(value: Mapping[str, Any]) -> RunPreflightManifest:
    """Rehydrate and fully validate one persisted content-free manifest."""

    if not isinstance(value, Mapping):
        raise LongMemEvalOfficialPreflightError("preflight artifact must be a JSON object")
    artifact = dict(value)
    dataset_value = artifact.get("dataset")
    tokenizer_value = artifact.get("tokenizer")
    cases_value = artifact.get("cases")
    if not isinstance(dataset_value, Mapping):
        raise LongMemEvalOfficialPreflightError("preflight artifact dataset must be an object")
    if not isinstance(tokenizer_value, Mapping):
        raise LongMemEvalOfficialPreflightError("preflight artifact tokenizer must be an object")
    if not isinstance(cases_value, list):
        raise LongMemEvalOfficialPreflightError("preflight artifact cases must be a list")
    dataset_payload = dict(dataset_value)
    tokenizer_payload = dict(tokenizer_value)
    dataset = DatasetRequirement(
        name=dataset_payload.get("name"),
        source_label=dataset_payload.get("source_label"),
        source_sha256=dataset_payload.get("source_sha256"),
        question_count=dataset_payload.get("question_count"),
        official=dataset_payload.get("official"),
    )
    tokenizer = ExactTokenizerPin(
        model=tokenizer_payload.get("model"),
        revision=tokenizer_payload.get("revision"),
        artifact_sha256=tokenizer_payload.get("artifact_sha256"),
        executable_sha256=tokenizer_payload.get("executable_sha256"),
        protocol=tokenizer_payload.get("protocol"),
    )
    cases: list[DatasetCaseBinding] = []
    for index, case_value in enumerate(cases_value):
        if not isinstance(case_value, Mapping):
            raise LongMemEvalOfficialPreflightError(
                f"preflight artifact cases[{index}] must be an object"
            )
        case_payload = dict(case_value)
        source_record = case_payload.get("source_record")
        question = case_payload.get("question")
        current_date = case_payload.get("current_date")
        if not all(isinstance(item, Mapping) for item in (source_record, question, current_date)):
            raise LongMemEvalOfficialPreflightError(
                f"preflight artifact cases[{index}] bindings must be objects"
            )
        assert isinstance(source_record, Mapping)
        assert isinstance(question, Mapping)
        assert isinstance(current_date, Mapping)
        cases.append(
            DatasetCaseBinding(
                case_index=case_payload.get("case_index"),
                question_id=case_payload.get("question_id"),
                source_record_sha256=source_record.get("sha256"),
                source_record_utf8_bytes=source_record.get("utf8_bytes"),
                question_sha256=question.get("sha256"),
                question_utf8_bytes=question.get("utf8_bytes"),
                current_date_sha256=current_date.get("sha256"),
                current_date_utf8_bytes=current_date.get("utf8_bytes"),
            )
        )
    manifest = RunPreflightManifest(
        dataset=dataset,
        source_artifact_utf8_bytes=dataset_payload.get("source_artifact_utf8_bytes"),
        parsed_records_sha256=dataset_payload.get("parsed_records_sha256"),
        parsed_records_utf8_bytes=dataset_payload.get("parsed_records_utf8_bytes"),
        projection_sha256=dataset_payload.get("turn_projection_sha256"),
        question_ids_sha256=dataset_payload.get("question_ids_sha256"),
        cases=tuple(cases),
        tokenizer=tokenizer,
    )
    if not _exact_json_equal(artifact, manifest.content_free_artifact()):
        raise LongMemEvalOfficialPreflightError(
            "preflight artifact differs from its exact schema, constants, or manifest digest"
        )
    return manifest


def load_preflight_manifest_bytes(raw: bytes) -> RunPreflightManifest:
    """Strict-JSON loader for a persisted preflight manifest artifact."""

    if not isinstance(raw, bytes) or not raw:
        raise LongMemEvalOfficialPreflightError("preflight manifest bytes must be non-empty")

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise LongMemEvalOfficialPreflightError(
                    f"preflight manifest repeats JSON field {key!r}"
                )
            output[key] = item
        return output

    def reject_constant(value: str) -> None:
        raise LongMemEvalOfficialPreflightError(
            f"preflight manifest contains non-finite number {value!r}"
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=reject_constant,
        )
    except LongMemEvalOfficialPreflightError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LongMemEvalOfficialPreflightError(
            "preflight manifest must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise LongMemEvalOfficialPreflightError("preflight manifest must contain one object")
    return load_preflight_manifest_artifact(value)


def _exact_keys(value: Any, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise LongMemEvalOfficialPreflightError(
            f"{label} fields differ from the frozen schema: {actual}"
        )
    return value


def _turn_id(value: Any, *, label: str) -> LongMemEvalTurnId:
    if not isinstance(value, list) or len(value) != 3:
        raise LongMemEvalOfficialPreflightError(f"{label} must be a three-field turn ID")
    question_id = checked_text(value[0], label=f"{label} question_id")
    session_position = checked_integer(value[1], label=f"{label} session position")
    turn_position = checked_integer(value[2], label=f"{label} turn position")
    return LongMemEvalTurnId(question_id, session_position, turn_position)


def _turn_ids(value: Any, *, label: str) -> list[LongMemEvalTurnId]:
    if not isinstance(value, list):
        raise LongMemEvalOfficialPreflightError(f"{label} must be a list")
    return [_turn_id(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _turn_id_payload(turn_id: LongMemEvalTurnId) -> list[str | int]:
    return list(turn_id.as_tuple())


def _separator_binding(value: str) -> dict[str, Any]:
    encoded = value.encode("utf-8")
    return {"literal": value, "utf8_bytes": len(encoded), "sha256": sha256_bytes(encoded)}


def _expected_layout(
    layout: PromptLayout,
    *,
    block_count: int,
) -> dict[str, Any]:
    headers: list[dict[str, Any]] = []
    if layout is PromptLayout.CHAIN_BLOCKS:
        headers = [
            {
                "block_position": position,
                **_separator_binding(CHAIN_HEADER_TEMPLATE.format(chain_number=position)),
            }
            for position in range(1, block_count + 1)
        ]
    return {
        "layout": layout.value,
        "history_serializer_version": HISTORY_SERIALIZER_VERSION,
        "linear_turn_separator": _separator_binding(LINEAR_TURN_SEPARATOR),
        "chain_header_template": CHAIN_HEADER_TEMPLATE,
        "chain_headers": headers,
        "chain_header_body_separator": _separator_binding(CHAIN_HEADER_BODY_SEPARATOR),
        "chain_turn_separator": _separator_binding(CHAIN_TURN_SEPARATOR),
        "chain_block_separator": _separator_binding(CHAIN_BLOCK_SEPARATOR),
        "empty_context_note": {
            "utf8_bytes": len(EMPTY_CONTEXT_NOTE.encode("utf-8")),
            "sha256": EMPTY_CONTEXT_NOTE_SHA256,
        },
    }


def _candidate_material(
    trace: dict[str, Any],
    *,
    turns_by_id: Mapping[LongMemEvalTurnId, TurnProjection],
    question_id: str,
) -> tuple[PromptLayout, list[list[TurnProjection]], list[TurnProjection]]:
    raw_blocks = trace.get("candidate_blocks")
    if not isinstance(raw_blocks, list):
        raise LongMemEvalOfficialPreflightError("candidate blocks must be a list")
    try:
        layout = PromptLayout(trace.get("layout", {}).get("layout"))
    except (AttributeError, ValueError) as exc:
        raise LongMemEvalOfficialPreflightError("prompt layout is not registered") from exc
    if layout is PromptLayout.LINEAR and len(raw_blocks) != 1:
        raise LongMemEvalOfficialPreflightError("linear prompt must have exactly one block")
    if layout is PromptLayout.CHAIN_BLOCKS and not raw_blocks:
        raise LongMemEvalOfficialPreflightError("chain prompt must have at least one block")
    if not _exact_json_equal(
        trace.get("layout"),
        _expected_layout(layout, block_count=len(raw_blocks)),
    ):
        raise LongMemEvalOfficialPreflightError("prompt layout literals or digests drifted")

    blocks: list[list[TurnProjection]] = []
    flattened: list[TurnProjection] = []
    seen: set[LongMemEvalTurnId] = set()
    for block_index, raw_block in enumerate(raw_blocks):
        ids = _turn_ids(raw_block, label=f"candidate_blocks[{block_index}]")
        block: list[TurnProjection] = []
        for turn_id in ids:
            if turn_id.question_id != question_id:
                raise LongMemEvalOfficialPreflightError(
                    "prompt candidate crosses the question boundary"
                )
            if turn_id in seen:
                raise LongMemEvalOfficialPreflightError("prompt candidate repeats a turn ID")
            turn = turns_by_id.get(turn_id)
            if turn is None:
                raise LongMemEvalOfficialPreflightError(
                    "prompt candidate is absent from the pinned turn projection"
                )
            seen.add(turn_id)
            block.append(turn)
            flattened.append(turn)
        blocks.append(block)

    expected_order = [
        {
            "candidate_position": candidate_position,
            "block_position": block_position,
            "position_in_block": position_in_block,
            "turn": turn.content_free_binding(),
        }
        for candidate_position, (block_position, position_in_block, turn) in enumerate(
            (
                (block_position, position_in_block, turn)
                for block_position, block in enumerate(blocks, start=1)
                for position_in_block, turn in enumerate(block, start=1)
            ),
            start=1,
        )
    ]
    if not _exact_json_equal(trace.get("candidate_order"), expected_order):
        raise LongMemEvalOfficialPreflightError(
            "candidate order differs from exact corpus turn provenance"
        )
    if trace.get("candidate_order_sha256") != sha256_json(expected_order):
        raise LongMemEvalOfficialPreflightError("candidate-order digest is stale or tampered")
    return layout, blocks, flattened


def _render_history(
    layout: PromptLayout,
    blocks: list[list[TurnProjection]],
    kept: set[LongMemEvalTurnId],
) -> str:
    if not kept:
        return EMPTY_CONTEXT_NOTE
    if layout is PromptLayout.LINEAR:
        return LINEAR_TURN_SEPARATOR.join(
            turn.serialized_text for turn in blocks[0] if turn.turn_id in kept
        )
    rendered: list[str] = []
    for block_position, block in enumerate(blocks, start=1):
        selected = [turn.serialized_text for turn in block if turn.turn_id in kept]
        if not selected:
            continue
        header = CHAIN_HEADER_TEMPLATE.format(chain_number=block_position)
        rendered.append(
            f"{header}{CHAIN_HEADER_BODY_SEPARATOR}{CHAIN_TURN_SEPARATOR.join(selected)}"
        )
    return CHAIN_BLOCK_SEPARATOR.join(rendered) if rendered else EMPTY_CONTEXT_NOTE


def _assert_observed_complete_prompt(
    observation: dict[str, Any],
    *,
    layout: PromptLayout,
    blocks: list[list[TurnProjection]],
    selected_ids: set[LongMemEvalTurnId],
    question: str,
    current_date: str,
    label: str,
) -> None:
    history = _render_history(layout, blocks, selected_ids)
    prompt = OFFICIAL_ANSWER_TEMPLATE.format(history, current_date, question)
    encoded = prompt.encode("utf-8")
    receipt = observation["receipt"]
    if receipt["prompt_sha256"] != sha256_bytes(encoded) or receipt["prompt_utf8_bytes"] != len(
        encoded
    ):
        raise LongMemEvalOfficialPreflightError(
            f"{label} exact receipt does not bind its reconstructed complete official prompt"
        )


def _validate_observations(
    trace: dict[str, Any],
    *,
    expected_count: int,
    query_sha256: str,
    tokenizer_identity_sha256: str,
    provider_request_ids: set[str],
    request_ids: set[int],
    previous_request_id: int | None,
) -> tuple[list[dict[str, Any]], int, int]:
    raw = trace.get("exact_count_observations")
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise LongMemEvalOfficialPreflightError(
            "exact-count observation coverage differs from packing decisions"
        )
    if trace.get("exact_count_observations_sha256") != sha256_json(raw):
        raise LongMemEvalOfficialPreflightError("exact-count observation digest is tampered")
    observations: list[dict[str, Any]] = []
    prompt_bytes = 0
    local_provider_ids: set[str] = set()
    local_prompt_hashes: set[str] = set()
    counts_by_prompt_sha256: dict[str, int] = {}
    last_request_id = previous_request_id
    for index, item in enumerate(raw, start=1):
        observation = _exact_keys(item, _OBSERVATION_FIELDS, label=f"observation[{index}]")
        if (
            checked_integer(
                observation.get("sequence"),
                label=f"observation[{index}] sequence",
                minimum=1,
            )
            != index
        ):
            raise LongMemEvalOfficialPreflightError(
                "exact-count observation sequences must be contiguous"
            )
        if observation.get("purpose") not in {
            "initial-empty-context",
            "candidate-alone",
            "greedy-proposal",
            "final-independent-recount",
        }:
            raise LongMemEvalOfficialPreflightError("exact-count observation purpose is invalid")
        candidate = observation.get("candidate_turn_id")
        if candidate is not None:
            _turn_id(candidate, label=f"observation[{index}] candidate")
        receipt = _exact_keys(
            observation.get("receipt"),
            _RECEIPT_FIELDS,
            label=f"observation[{index}] receipt",
        )
        request_id = checked_integer(
            receipt.get("request_id"),
            label=f"observation[{index}] request_id",
            minimum=1,
        )
        if (
            request_id != len(request_ids) + 1
            or request_id in request_ids
            or (last_request_id is not None and request_id <= last_request_id)
        ):
            raise LongMemEvalOfficialPreflightError(
                "tokenizer request IDs must be globally fresh, contiguous, and increasing"
            )
        provider_request_id = receipt.get("provider_request_id")
        if (
            not isinstance(provider_request_id, str)
            or _PROVIDER_REQUEST_ID_RE.fullmatch(provider_request_id) is None
            or provider_request_id in provider_request_ids
        ):
            raise LongMemEvalOfficialPreflightError(
                "tokenizer provider request IDs must be valid and globally unique"
            )
        if receipt.get("tokenizer_identity_sha256") != tokenizer_identity_sha256:
            raise LongMemEvalOfficialPreflightError("tokenizer receipt identity drifted")
        if receipt.get("query_sha256") != query_sha256:
            raise LongMemEvalOfficialPreflightError(
                "tokenizer receipt query does not match the official source question"
            )
        prompt_sha256 = checked_sha256(
            receipt.get("prompt_sha256"),
            label=f"observation[{index}] prompt SHA-256",
        )
        receipt_bytes = checked_integer(
            receipt.get("prompt_utf8_bytes"),
            label=f"observation[{index}] prompt bytes",
            minimum=1,
        )
        checked_integer(
            receipt.get("token_count"),
            label=f"observation[{index}] token count",
            minimum=1,
        )
        prior_count = counts_by_prompt_sha256.get(prompt_sha256)
        if prior_count is not None and prior_count != receipt["token_count"]:
            raise LongMemEvalOfficialPreflightError(
                "repeated exact prompt receipts disagree on token count"
            )
        counts_by_prompt_sha256[prompt_sha256] = receipt["token_count"]
        request_ids.add(request_id)
        provider_request_ids.add(provider_request_id)
        local_provider_ids.add(provider_request_id)
        local_prompt_hashes.add(prompt_sha256)
        last_request_id = request_id
        prompt_bytes += receipt_bytes
        observations.append(observation)

    expected_accounting = {
        "requests": len(observations),
        "responses": len(observations),
        "unique_provider_request_ids": len(local_provider_ids),
        "unique_prompt_digests": len(local_prompt_hashes),
        "request_ids_strictly_monotone": True,
        "repeated_prompt_counts_reconciled": True,
    }
    if not _exact_json_equal(trace.get("observation_accounting"), expected_accounting):
        raise LongMemEvalOfficialPreflightError(
            "prompt observation accounting is incomplete or inconsistent"
        )
    assert last_request_id is not None
    return observations, prompt_bytes, last_request_id


def _validate_decisions(
    trace: dict[str, Any],
    *,
    flattened: list[TurnProjection],
    observations: list[dict[str, Any]],
) -> tuple[list[LongMemEvalTurnId], list[LongMemEvalTurnId], list[LongMemEvalTurnId], int]:
    decisions = trace.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(flattened):
        raise LongMemEvalOfficialPreflightError(
            "packing decision coverage differs from candidate coverage"
        )
    if trace.get("decisions_sha256") != sha256_json(decisions):
        raise LongMemEvalOfficialPreflightError("packing-decision digest is tampered")
    by_sequence = {item["sequence"]: item for item in observations}
    selected: list[LongMemEvalTurnId] = []
    dropped: list[LongMemEvalTurnId] = []
    oversized: list[LongMemEvalTurnId] = []
    accepted_token_count = int(observations[0]["receipt"]["token_count"])
    for index, (decision_raw, turn) in enumerate(zip(decisions, flattened, strict=True)):
        decision = _exact_keys(
            decision_raw,
            _DECISION_FIELDS,
            label=f"decision[{index}]",
        )
        candidate_id = _turn_id(
            decision.get("candidate_turn_id"),
            label=f"decision[{index}] candidate",
        )
        if candidate_id != turn.turn_id:
            raise LongMemEvalOfficialPreflightError(
                "packing decisions do not preserve exact candidate order"
            )
        order_item = trace["candidate_order"][index]
        if (
            checked_integer(
                decision.get("block_position"),
                label=f"decision[{index}] block position",
                minimum=1,
            )
            != order_item["block_position"]
            or checked_integer(
                decision.get("position_in_block"),
                label=f"decision[{index}] position in block",
                minimum=1,
            )
            != order_item["position_in_block"]
        ):
            raise LongMemEvalOfficialPreflightError(
                "packing decision block positions differ from candidate order"
            )
        expected_selected = [_turn_id_payload(item) for item in selected]
        if not _exact_json_equal(decision.get("selected_before_ids"), expected_selected):
            raise LongMemEvalOfficialPreflightError(
                "packing decision selected-before state is inconsistent"
            )
        expected_proposed = [*expected_selected, _turn_id_payload(candidate_id)]
        if not _exact_json_equal(decision.get("proposed_ids"), expected_proposed):
            raise LongMemEvalOfficialPreflightError(
                "packing decision proposal state is inconsistent"
            )
        singleton_sequence = checked_integer(
            decision.get("singleton_observation_sequence"),
            label=f"decision[{index}] singleton sequence",
            minimum=1,
        )
        proposal_sequence = checked_integer(
            decision.get("proposal_observation_sequence"),
            label=f"decision[{index}] proposal sequence",
            minimum=1,
        )
        singleton = by_sequence.get(singleton_sequence)
        proposal = by_sequence.get(proposal_sequence)
        candidate_payload = _turn_id_payload(candidate_id)
        if (
            singleton is None
            or singleton.get("purpose") != "candidate-alone"
            or not _exact_json_equal(singleton.get("candidate_turn_id"), candidate_payload)
        ):
            raise LongMemEvalOfficialPreflightError(
                "packing decision does not bind its candidate-alone observation"
            )
        if (
            proposal is None
            or proposal.get("purpose") != "greedy-proposal"
            or not _exact_json_equal(proposal.get("candidate_turn_id"), candidate_payload)
        ):
            raise LongMemEvalOfficialPreflightError(
                "packing decision does not bind its greedy-proposal observation"
            )
        expected_accepted = proposal["receipt"]["token_count"] <= PRIMARY_TOKEN_BUDGET
        expected_oversized = singleton["receipt"]["token_count"] > PRIMARY_TOKEN_BUDGET
        if type(decision.get("accepted")) is not bool or (
            decision["accepted"] is not expected_accepted
        ):
            raise LongMemEvalOfficialPreflightError(
                "packing acceptance differs from the exact 8,192-token receipt"
            )
        if type(decision.get("oversized_alone")) is not bool or (
            decision["oversized_alone"] is not expected_oversized
        ):
            raise LongMemEvalOfficialPreflightError(
                "oversized-turn classification differs from its exact singleton receipt"
            )
        if decision["accepted"]:
            selected.append(candidate_id)
            accepted_token_count = int(proposal["receipt"]["token_count"])
        else:
            dropped.append(candidate_id)
            if decision["oversized_alone"]:
                oversized.append(candidate_id)
    return selected, dropped, oversized, accepted_token_count


def _validate_packed_case(
    result: TurnPromptPackingResult,
    *,
    case: DatasetCaseBinding,
    source_case: _SourceCaseMaterial,
    turns_by_id: Mapping[LongMemEvalTurnId, TurnProjection],
    tokenizer: ExactTokenizerPin,
    provider_request_ids: set[str],
    request_ids: set[int],
    previous_request_id: int | None,
) -> tuple[str, int, int]:
    if not isinstance(result, TurnPromptPackingResult):
        raise LongMemEvalOfficialPreflightError(
            "packed cases must be TurnPromptPackingResult instances"
        )
    trace = _exact_keys(result.trace, _TRACE_FIELDS, label=f"case {case.question_id} trace")
    expected_header = {
        "artifact_type": PROMPT_ARTIFACT_TYPE,
        "schema_version": PROMPT_SCHEMA_VERSION,
        "protocol_version": PROMPT_PROTOCOL_VERSION,
        "classification": "evaluation-only-exact-turn-prompt-packing",
        "production_configuration": False,
    }
    for field, expected in expected_header.items():
        if type(trace.get(field)) is not type(expected) or trace.get(field) != expected:
            raise LongMemEvalOfficialPreflightError(
                f"case {case.question_id} prompt {field} drifted"
            )
    if not _exact_json_equal(
        trace.get("packing_policy"),
        {
            "method": "exact-complete-prompt-greedy-skip-and-continue",
            "whole_turns_indivisible": True,
            "candidate_order_mutated": False,
            "oversized_definition": "candidate-alone-complete-prompt-tokens>budget",
            "final_independent_recount": True,
        },
    ):
        raise LongMemEvalOfficialPreflightError("prompt packing policy drifted")
    if not _exact_json_equal(
        trace.get("budget"),
        {
            "token_budget": PRIMARY_TOKEN_BUDGET,
            "supported_token_budgets": [4096, 8192, 16384],
            "primary_token_budget": PRIMARY_TOKEN_BUDGET,
            "is_primary": True,
            "counted_surface": "complete-official-reader-prompt",
        },
    ):
        raise LongMemEvalOfficialPreflightError(
            "packed prompt does not use the primary complete-prompt 8,192-token budget"
        )
    if not _exact_json_equal(
        trace.get("reader_prompt"),
        {
            "template_source": "scripts._longmemeval_common.OFFICIAL_ANSWER_TEMPLATE",
            "template_sha256": OFFICIAL_ANSWER_TEMPLATE_SHA256,
            "template_utf8_bytes": len(OFFICIAL_ANSWER_TEMPLATE.encode("utf-8")),
            "history_placeholder": "frozen-turn-history",
        },
    ):
        raise LongMemEvalOfficialPreflightError("official reader prompt template drifted")
    if not _exact_json_equal(trace.get("tokenizer"), tokenizer.identity.as_dict()):
        raise LongMemEvalOfficialPreflightError(
            "packed prompt tokenizer differs from the precommitted exact tokenizer"
        )

    question = source_case.question
    current_date = source_case.current_date
    question_input = trace.get("question_input")
    expected_question_input = {
        "question_id": case.question_id,
        "query_sha256": case.question_sha256,
        "query_utf8_bytes": case.question_utf8_bytes,
        "current_date_sha256": case.current_date_sha256,
        "current_date_utf8_bytes": case.current_date_utf8_bytes,
        "combined_input_sha256": sha256_json({"current_date": current_date, "question": question}),
    }
    if not _exact_json_equal(question_input, expected_question_input):
        raise LongMemEvalOfficialPreflightError(
            "packed prompt question/date binding differs from the official source record"
        )

    layout, blocks, flattened = _candidate_material(
        trace,
        turns_by_id=turns_by_id,
        question_id=case.question_id,
    )
    observations, observation_bytes, last_request_id = _validate_observations(
        trace,
        expected_count=2 + 2 * len(flattened),
        query_sha256=case.question_sha256,
        tokenizer_identity_sha256=tokenizer.identity.identity_sha256,
        provider_request_ids=provider_request_ids,
        request_ids=request_ids,
        previous_request_id=previous_request_id,
    )
    if (
        observations[0].get("purpose") != "initial-empty-context"
        or observations[0].get("candidate_turn_id") is not None
        or observations[-1].get("purpose") != "final-independent-recount"
        or observations[-1].get("candidate_turn_id") is not None
    ):
        raise LongMemEvalOfficialPreflightError(
            "prompt observations lack independent initial/final counts"
        )
    for candidate_index, turn in enumerate(flattened):
        singleton = observations[1 + 2 * candidate_index]
        proposal = observations[2 + 2 * candidate_index]
        candidate_payload = _turn_id_payload(turn.turn_id)
        if (
            singleton.get("purpose") != "candidate-alone"
            or not _exact_json_equal(singleton.get("candidate_turn_id"), candidate_payload)
            or proposal.get("purpose") != "greedy-proposal"
            or not _exact_json_equal(proposal.get("candidate_turn_id"), candidate_payload)
        ):
            raise LongMemEvalOfficialPreflightError(
                "exact-count observations do not preserve candidate scan order"
            )
    if observations[0]["receipt"]["token_count"] > PRIMARY_TOKEN_BUDGET:
        raise LongMemEvalOfficialPreflightError(
            "fixed official prompt exceeds the primary budget before adding evidence"
        )
    _assert_observed_complete_prompt(
        observations[0],
        layout=layout,
        blocks=blocks,
        selected_ids=set(),
        question=question,
        current_date=current_date,
        label="initial empty-context",
    )
    selected, dropped, oversized, accepted_token_count = _validate_decisions(
        trace,
        flattened=flattened,
        observations=observations,
    )
    for decision in trace["decisions"]:
        candidate_id = _turn_id(
            decision["candidate_turn_id"],
            label="decision complete-prompt candidate",
        )
        singleton = observations[decision["singleton_observation_sequence"] - 1]
        proposal = observations[decision["proposal_observation_sequence"] - 1]
        proposed_ids = set(
            _turn_ids(
                decision["proposed_ids"],
                label="decision complete-prompt proposal",
            )
        )
        _assert_observed_complete_prompt(
            singleton,
            layout=layout,
            blocks=blocks,
            selected_ids={candidate_id},
            question=question,
            current_date=current_date,
            label="candidate-alone",
        )
        _assert_observed_complete_prompt(
            proposal,
            layout=layout,
            blocks=blocks,
            selected_ids=proposed_ids,
            question=question,
            current_date=current_date,
            label="greedy proposal",
        )
    if not _exact_json_equal(
        trace.get("kept_ids"),
        [_turn_id_payload(item) for item in selected],
    ):
        raise LongMemEvalOfficialPreflightError("kept IDs do not reconcile with decisions")
    if not _exact_json_equal(
        trace.get("dropped_ids"),
        [_turn_id_payload(item) for item in dropped],
    ):
        raise LongMemEvalOfficialPreflightError("dropped IDs do not reconcile with decisions")
    if not _exact_json_equal(
        trace.get("oversized_ids"),
        [_turn_id_payload(item) for item in oversized],
    ):
        raise LongMemEvalOfficialPreflightError("oversized IDs do not reconcile with decisions")
    selected_set = set(selected)
    expected_kept_by_block = [
        [_turn_id_payload(turn.turn_id) for turn in block if turn.turn_id in selected_set]
        for block in blocks
    ]
    if not _exact_json_equal(trace.get("kept_ids_by_block"), expected_kept_by_block):
        raise LongMemEvalOfficialPreflightError(
            "kept block membership does not reconcile with candidate blocks"
        )

    history = _render_history(layout, blocks, selected_set)
    expected_prompt = OFFICIAL_ANSWER_TEMPLATE.format(history, current_date, question)
    if result.prompt != expected_prompt:
        raise LongMemEvalOfficialPreflightError(
            "packed prompt bytes differ from the exact official template and kept turns"
        )
    prompt_bytes = expected_prompt.encode("utf-8")
    history_bytes = history.encode("utf-8")
    final_receipt = observations[-1]["receipt"]
    _assert_observed_complete_prompt(
        observations[-1],
        layout=layout,
        blocks=blocks,
        selected_ids=selected_set,
        question=question,
        current_date=current_date,
        label="final independent recount",
    )
    expected_final = {
        "sha256": sha256_bytes(prompt_bytes),
        "utf8_bytes": len(prompt_bytes),
        "tokens": final_receipt["token_count"],
        "final_observation_sequence": observations[-1]["sequence"],
        "within_budget": final_receipt["token_count"] <= PRIMARY_TOKEN_BUDGET,
    }
    if not _exact_json_equal(trace.get("final_prompt"), expected_final):
        raise LongMemEvalOfficialPreflightError("final prompt receipt is stale or tampered")
    if (
        final_receipt["prompt_sha256"] != expected_final["sha256"]
        or final_receipt["prompt_utf8_bytes"] != expected_final["utf8_bytes"]
        or final_receipt["token_count"] != accepted_token_count
        or not expected_final["within_budget"]
    ):
        raise LongMemEvalOfficialPreflightError(
            "final exact-token receipt does not reconcile with the accepted prompt state"
        )
    if not _exact_json_equal(
        trace.get("final_history"),
        {
            "sha256": sha256_bytes(history_bytes),
            "utf8_bytes": len(history_bytes),
        },
    ):
        raise LongMemEvalOfficialPreflightError("final history binding is stale or tampered")
    if not _exact_json_equal(
        trace.get("claims"),
        {
            "token_counts_are_external_receipts": True,
            "token_counts_are_estimates": False,
            "external_tokenizer_boundary_invoked": True,
            "packer_loads_tokenizer_or_calls_reader_model": False,
            "tokenizer_artifact_reopened_by_packer": False,
            "gold_fields_consumed": False,
            "qa_improvement": False,
        },
    ):
        raise LongMemEvalOfficialPreflightError("prompt trace claims drifted")
    return result.trace_sha256, len(observations), observation_bytes


def _file_evidence(value: Any, *, expected_sha256: str, label: str) -> dict[str, Any]:
    artifact = _exact_keys(value, frozenset({"path", "bytes", "sha256"}), label=label)
    path = artifact.get("path")
    if not isinstance(path, str) or not path or path != PurePosixPath(path).as_posix():
        raise LongMemEvalOfficialPreflightError(f"{label} path must be canonical relative POSIX")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise LongMemEvalOfficialPreflightError(f"{label} path must remain repository-local")
    checked_integer(artifact.get("bytes"), label=f"{label} bytes", minimum=1)
    if artifact.get("sha256") != expected_sha256:
        raise LongMemEvalOfficialPreflightError(f"{label} SHA-256 differs from its pin")
    return artifact


def _validate_tokenizer_evidence(
    value: Mapping[str, Any],
    *,
    tokenizer: ExactTokenizerPin,
    observation_count: int,
    provider_request_count: int,
    prompt_utf8_bytes: int,
) -> str:
    if not isinstance(value, Mapping):
        raise LongMemEvalOfficialPreflightError("tokenizer evidence must be an object")
    evidence = _exact_keys(dict(value), _TOKENIZER_EVIDENCE_FIELDS, label="tokenizer evidence")
    expected_scalars = {
        "method": "exact_serialized_reader_prompt",
        "provider": "JsonlExactTokenizer",
        "exact_model_tokenizer": True,
        "tokenizer_model": tokenizer.model,
        "tokenizer_revision": tokenizer.revision,
        "protocol": TOKENIZER_PROTOCOL,
        "response_identity_sha256": tokenizer.identity.identity_sha256,
    }
    for field, expected in expected_scalars.items():
        if type(evidence.get(field)) is not type(expected) or evidence.get(field) != expected:
            raise LongMemEvalOfficialPreflightError(
                f"tokenizer evidence {field} differs from the frozen pin"
            )
    _file_evidence(
        evidence.get("tokenizer_artifact"),
        expected_sha256=tokenizer.artifact_sha256,
        label="tokenizer artifact",
    )
    _file_evidence(
        evidence.get("tokenizer_executable"),
        expected_sha256=tokenizer.executable_sha256,
        label="tokenizer executable",
    )
    accounting = _exact_keys(
        evidence.get("observation_accounting"),
        _TOKENIZER_ACCOUNTING_FIELDS,
        label="tokenizer observation accounting",
    )
    if accounting.get("source") != "provider-observed":
        raise LongMemEvalOfficialPreflightError("tokenizer counts are not provider-observed")
    expected_counts = {
        "requests": observation_count,
        "responses": observation_count,
        "unique_provider_request_ids": provider_request_count,
        "text_utf8_bytes": prompt_utf8_bytes,
        "exact_response_identity_verified": True,
    }
    for field, expected in expected_counts.items():
        if type(accounting.get(field)) is not type(expected) or accounting.get(field) != expected:
            raise LongMemEvalOfficialPreflightError(
                f"tokenizer accounting {field} does not reconcile with all packed cases"
            )
    text_characters = checked_integer(
        accounting.get("text_characters"),
        label="tokenizer text characters",
        minimum=1,
    )
    if text_characters > prompt_utf8_bytes:
        raise LongMemEvalOfficialPreflightError(
            "tokenizer character accounting exceeds exact UTF-8 bytes"
        )
    return sha256_json(evidence)


def validate_prepared_run(
    source_bytes: bytes,
    *,
    manifest: RunPreflightManifest,
    packed_cases: tuple[TurnPromptPackingResult, ...],
    tokenizer_evidence: Mapping[str, Any],
) -> PreparedRunReceipt:
    """Admit a fully counted run before any reader call is allowed."""

    if not isinstance(manifest, RunPreflightManifest):
        raise LongMemEvalOfficialPreflightError("manifest must be RunPreflightManifest")
    if not isinstance(packed_cases, tuple):
        raise LongMemEvalOfficialPreflightError("packed cases must be an immutable tuple")
    if len(packed_cases) != len(manifest.cases):
        raise LongMemEvalOfficialPreflightError(
            "packed case coverage does not equal the complete preflight manifest"
        )
    _validate_prompt_sources()
    corpus, source_cases, cases = _source_material(source_bytes, dataset=manifest.dataset)
    rebuilt = _manifest_from_material(
        corpus,
        cases=cases,
        dataset=manifest.dataset,
        tokenizer=manifest.tokenizer,
    )
    if rebuilt != manifest:
        raise LongMemEvalOfficialPreflightError(
            "preflight manifest differs from the exact source bytes or frozen pins"
        )
    turns_by_id = corpus.by_id()
    provider_request_ids: set[str] = set()
    request_ids: set[int] = set()
    previous_request_id: int | None = None
    trace_sha256s: list[str] = []
    observation_count = 0
    prompt_utf8_bytes = 0
    for result, case, source_case in zip(
        packed_cases,
        manifest.cases,
        source_cases,
        strict=True,
    ):
        trace_sha256, case_observations, case_bytes = _validate_packed_case(
            result,
            case=case,
            source_case=source_case,
            turns_by_id=turns_by_id,
            tokenizer=manifest.tokenizer,
            provider_request_ids=provider_request_ids,
            request_ids=request_ids,
            previous_request_id=previous_request_id,
        )
        trace_sha256s.append(trace_sha256)
        observation_count += case_observations
        prompt_utf8_bytes += case_bytes
        previous_request_id = max(request_ids)
    tokenizer_evidence_sha256 = _validate_tokenizer_evidence(
        tokenizer_evidence,
        tokenizer=manifest.tokenizer,
        observation_count=observation_count,
        provider_request_count=len(provider_request_ids),
        prompt_utf8_bytes=prompt_utf8_bytes,
    )
    return PreparedRunReceipt(
        preflight_manifest_sha256=manifest.manifest_sha256,
        packed_case_count=len(packed_cases),
        question_ids_sha256=manifest.question_ids_sha256,
        prompt_trace_sha256s=tuple(trace_sha256s),
        prompt_trace_set_sha256=sha256_json(trace_sha256s),
        tokenizer_evidence_sha256=tokenizer_evidence_sha256,
        exact_count_observation_count=observation_count,
        exact_count_prompt_utf8_bytes=prompt_utf8_bytes,
    )


def validate_official_prepared_run(
    source_bytes: bytes,
    *,
    manifest: RunPreflightManifest,
    packed_cases: tuple[TurnPromptPackingResult, ...],
    tokenizer_evidence: Mapping[str, Any],
) -> PreparedRunReceipt:
    """Require official-full-500 pins, then validate all prepared prompts."""

    if not isinstance(manifest, RunPreflightManifest):
        raise LongMemEvalOfficialPreflightError(
            "official run admission requires a RunPreflightManifest"
        )
    if manifest.dataset != OFFICIAL_DATASET_REQUIREMENT:
        raise LongMemEvalOfficialPreflightError(
            "official run admission requires the exact official full-500 manifest"
        )
    return validate_prepared_run(
        source_bytes,
        manifest=manifest,
        packed_cases=packed_cases,
        tokenizer_evidence=tokenizer_evidence,
    )


__all__ = [
    "freeze_official_preflight",
    "freeze_pinned_preflight",
    "load_preflight_manifest_artifact",
    "load_preflight_manifest_bytes",
    "validate_official_prepared_run",
    "validate_prepared_run",
]
