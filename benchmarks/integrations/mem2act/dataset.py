"""Strict loader for the pinned official Mem2ActBench evaluation release."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    CorpusSession,
    DatasetFingerprint,
    Mem2ActContractError,
    Mem2ActDataset,
    Mem2ActTask,
    OracleMemory,
)

MEM2ACT_REPO_COMMIT = "b00726940b5abbe9bd324bdd7a2cb272f5c62a29"
KNOWN_TOOL_NAME_REPAIRS: dict[str, tuple[Any, str]] = {
    # The pinned JSONL serializes this target name as the number ``4`` while
    # its target schema carries the complete public name.  Accept only this
    # exact defect and use the schema name; any other mismatch fails closed.
    "qa_283": (4, "4D Dream Dictionary"),
}


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    repo_commit: str | None
    qa_path: str
    conversation_path: str
    statistics_path: str
    files_sha256: dict[str, str]
    task_count: int
    session_count: int
    allowed_unresolved_source_ids: frozenset[str] = frozenset()


# ``toolmembench_small`` is the paper's complete 400-task / 429-session
# evaluation subset despite its historical directory name.  ``Mem2ActBench``
# contains all 2,029 constructed sessions and is not the fixed evaluation
# corpus used by the experiments.
OFFICIAL_MEM2ACT_SPEC = DatasetSpec(
    repo_commit=MEM2ACT_REPO_COMMIT,
    qa_path="toolmembench_small/qa_dataset.jsonl",
    conversation_path="toolmembench_small/toolmem_conversation.jsonl",
    statistics_path="toolmembench_small/benchmark_statistics.json",
    files_sha256={
        "toolmembench_small/qa_dataset.jsonl": (
            "c5e3f47799d850b607d0ff56829335f843e208ad4d3b1eae89a814eae1974b09"
        ),
        "toolmembench_small/toolmem_conversation.jsonl": (
            "c935adfbe0e1743b8eb373eba31611b97cf396ee09516f68f6621e49365cbcaf"
        ),
        "toolmembench_small/benchmark_statistics.json": (
            "a66ae640bb34e1bc1eb2362f8a0ede511017782002ce3f288b73ae2c9fc3c600"
        ),
    },
    task_count=400,
    session_count=429,
    # The published evaluation subset has eleven source provenance IDs whose
    # raw turns are absent from its 429-session JSONL.  This exact, pinned set
    # is an upstream data defect, not a wildcard waiver.  Normal retrieval
    # always ingests all 429 published sessions; the oracle arm uses the
    # published evolution-chain evidence and therefore does not fabricate raw
    # conversations for these IDs.  Any additional or repaired gap fails until
    # a new dataset fingerprint/spec is reviewed.
    allowed_unresolved_source_ids=frozenset(
        {
            "live_multiple_842-178-17",
            "live_simple_51-23-0",
            "multi_turn_base_23",
            "toolace_218",
            "toolace_2480",
            "toolace_298",
            "toolace_3835",
            "toolace_5507",
            "toolace_5885",
            "toolace_7633",
            "toolace_7839",
        }
    ),
)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Mem2ActContractError("value is not strict JSON") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as exc:
        raise Mem2ActContractError(f"cannot read required dataset file: {path}") from exc
    return digest.hexdigest()


def load_mem2act_dataset(
    repo_root: str | Path,
    *,
    spec: DatasetSpec = OFFICIAL_MEM2ACT_SPEC,
    verify_git: bool = True,
) -> Mem2ActDataset:
    """Load and validate every task/session before exposing any benchmark data."""

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise Mem2ActContractError(f"Mem2ActBench checkout is missing: {root}")
    commit = _verify_commit(root, spec, verify_git=verify_git)

    actual_hashes: dict[str, str] = {}
    for relative, expected in spec.files_sha256.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise Mem2ActContractError(
                f"dataset fingerprint mismatch for {relative}: expected {expected}, got {actual}"
            )
        actual_hashes[relative] = actual

    raw_tasks = _read_jsonl(root / spec.qa_path)
    raw_sessions = _read_jsonl(root / spec.conversation_path)
    statistics = _read_json(root / spec.statistics_path)

    tasks = _parse_tasks(raw_tasks, expected_count=spec.task_count)
    sessions = _parse_sessions(raw_sessions, expected_count=spec.session_count)
    _validate_statistics(statistics, task_count=len(tasks), session_count=len(sessions))
    unresolved = _validate_source_references(tasks, sessions, spec)
    catalog, catalog_hash = _tool_catalog(tasks)

    return Mem2ActDataset(
        tasks=tasks,
        sessions=sessions,
        tool_catalog=catalog,
        tool_catalog_sha256=catalog_hash,
        fingerprint=DatasetFingerprint(
            repo_commit=commit,
            files_sha256=actual_hashes,
            task_count=len(tasks),
            session_count=len(sessions),
            unresolved_source_ids=tuple(sorted(unresolved)),
            known_data_repairs=tuple(
                f"{qa_id}:tool_call.name={raw!r}->{replacement}"
                for qa_id, (raw, replacement) in sorted(KNOWN_TOOL_NAME_REPAIRS.items())
                if any(task.qa_id == qa_id for task in tasks)
            ),
        ),
    )


def _verify_commit(root: Path, spec: DatasetSpec, *, verify_git: bool) -> str:
    if spec.repo_commit is None:
        return "unversioned-fixture"
    if not verify_git:
        return spec.repo_commit
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise Mem2ActContractError(f"cannot verify Mem2ActBench git checkout: {root}") from exc
    commit = completed.stdout.strip()
    if commit != spec.repo_commit:
        raise Mem2ActContractError(
            f"Mem2ActBench commit mismatch: expected {spec.repo_commit}, got {commit}"
        )
    return commit


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Mem2ActContractError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Mem2ActContractError(f"non-finite JSON number is forbidden: {value}")


def _decode_json(text: str, *, location: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, Mem2ActContractError) as exc:
        raise Mem2ActContractError(f"invalid JSON at {location}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Mem2ActContractError(f"blank JSONL row at {path}:{line_number}")
                raw = _decode_json(line, location=f"{path}:{line_number}")
                if not isinstance(raw, dict):
                    raise Mem2ActContractError(
                        f"JSONL row must be an object at {path}:{line_number}"
                    )
                rows.append(raw)
    except OSError as exc:
        raise Mem2ActContractError(f"cannot read required dataset file: {path}") from exc
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = _decode_json(path.read_text(encoding="utf-8"), location=str(path))
    except OSError as exc:
        raise Mem2ActContractError(f"cannot read required dataset file: {path}") from exc
    if not isinstance(raw, dict):
        raise Mem2ActContractError(f"JSON root must be an object: {path}")
    return raw


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Mem2ActContractError(f"{field} must be a non-empty string")
    return value


def _required_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Mem2ActContractError(f"{field} must be an object")
    canonical_json(value)
    return value


def _required_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise Mem2ActContractError(f"{field} must be a list")
    return value


def _required_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise Mem2ActContractError(f"{field} must be an integer >= {minimum}")
    return value


def _unique_text_list(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    rows = _required_list(value, field)
    if not allow_empty and not rows:
        raise Mem2ActContractError(f"{field} must not be empty")
    values = tuple(_required_text(item, f"{field}[]") for item in rows)
    if len(values) != len(set(values)):
        raise Mem2ActContractError(f"{field} contains duplicate values")
    return values


def _parse_tasks(rows: list[dict[str, Any]], *, expected_count: int) -> tuple[Mem2ActTask, ...]:
    if len(rows) != expected_count:
        raise Mem2ActContractError(
            f"expected exactly {expected_count} Mem2ActBench tasks, found {len(rows)}"
        )
    tasks: list[Mem2ActTask] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows, 1):
        qa_id = _required_text(raw.get("qa_id"), f"task[{index}].qa_id")
        if qa_id in seen:
            raise Mem2ActContractError(f"duplicate Mem2ActBench task ID: {qa_id}")
        seen.add(qa_id)
        expected_id = f"qa_{index:03d}"
        if qa_id != expected_id:
            raise Mem2ActContractError(
                f"task sequence is incomplete or reordered: expected {expected_id}, got {qa_id}"
            )

        sources = _unique_text_list(
            raw.get("source_conversation_ids"), f"{qa_id}.source_conversation_ids"
        )
        chain_raw = _required_list(raw.get("evolution_chain"), f"{qa_id}.evolution_chain")
        oracle: list[OracleMemory] = []
        chain_sources: list[str] = []
        for chain_index, item in enumerate(chain_raw):
            chain = _required_object(item, f"{qa_id}.evolution_chain[{chain_index}]")
            source_id = _required_text(
                chain.get("source_id"), f"{qa_id}.evolution_chain[{chain_index}].source_id"
            )
            chain_sources.append(source_id)
            oracle.append(
                OracleMemory(
                    attribute=_required_text(
                        chain.get("attribute"),
                        f"{qa_id}.evolution_chain[{chain_index}].attribute",
                    ),
                    fact=_required_text(
                        chain.get("fact"), f"{qa_id}.evolution_chain[{chain_index}].fact"
                    ),
                    source_text=_required_text(
                        chain.get("source_text"),
                        f"{qa_id}.evolution_chain[{chain_index}].source_text",
                    ),
                )
            )
        if set(chain_sources) != set(sources):
            raise Mem2ActContractError(
                f"{qa_id} evolution-chain source IDs do not match source_conversation_ids"
            )

        tool_call = _required_object(raw.get("tool_call"), f"{qa_id}.tool_call")
        arguments = _required_object(tool_call.get("arguments"), f"{qa_id}.tool_call.arguments")
        if not arguments:
            raise Mem2ActContractError(f"{qa_id}.tool_call.arguments must not be empty")
        _required_object(tool_call.get("grounding_info"), f"{qa_id}.tool_call.grounding_info")

        schema = _required_object(raw.get("target_tool_schema"), f"{qa_id}.target_tool_schema")
        schema_name = _required_text(schema.get("name"), f"{qa_id}.target_tool_schema.name")
        raw_tool_name = tool_call.get("name")
        repair = KNOWN_TOOL_NAME_REPAIRS.get(qa_id)
        if repair is None:
            tool_name = _required_text(raw_tool_name, f"{qa_id}.tool_call.name")
        elif raw_tool_name == repair[0] and schema_name == repair[1]:
            tool_name = schema_name
        else:
            raise Mem2ActContractError(f"{qa_id} no longer matches its pinned tool-name repair")
        if schema_name != tool_name:
            raise Mem2ActContractError(f"{qa_id} target schema name does not match gold tool")
        parameters = _required_object(
            schema.get("parameters"), f"{qa_id}.target_tool_schema.parameters"
        )
        _required_object(parameters.get("properties"), f"{qa_id}.schema.parameters.properties")

        complexity = _required_object(
            raw.get("complexity_metadata"), f"{qa_id}.complexity_metadata"
        )
        level = _required_text(complexity.get("level"), f"{qa_id}.complexity_metadata.level")
        if level not in {"L1", "L2", "L3", "L4"}:
            raise Mem2ActContractError(f"{qa_id} has unsupported complexity level {level!r}")

        tasks.append(
            Mem2ActTask(
                qa_id=qa_id,
                query=_required_text(raw.get("query"), f"{qa_id}.query"),
                source_conversation_ids=sources,
                oracle_memories=tuple(oracle),
                gold_tool_name=tool_name,
                gold_arguments=arguments,
                target_tool_schema=schema,
                complexity_level=level,
            )
        )
    return tuple(tasks)


def _parse_sessions(
    rows: list[dict[str, Any]], *, expected_count: int
) -> tuple[CorpusSession, ...]:
    if len(rows) != expected_count:
        raise Mem2ActContractError(
            f"expected exactly {expected_count} Mem2ActBench sessions, found {len(rows)}"
        )
    sessions: list[CorpusSession] = []
    seen_sessions: set[str] = set()
    seen_sources: set[str] = set()
    for index, raw in enumerate(rows, 1):
        session_id = _required_text(raw.get("session_id"), f"session[{index}].session_id")
        if session_id in seen_sessions:
            raise Mem2ActContractError(f"duplicate Mem2ActBench session ID: {session_id}")
        seen_sessions.add(session_id)
        sources = _unique_text_list(
            raw.get("original_conversation_ids"),
            f"{session_id}.original_conversation_ids",
            allow_empty=False,
        )
        duplicate_sources = seen_sources.intersection(sources)
        if duplicate_sources:
            raise Mem2ActContractError(
                f"conversation source appears in multiple sessions: {sorted(duplicate_sources)}"
            )
        seen_sources.update(sources)

        turns_raw = _required_list(raw.get("turns"), f"{session_id}.turns")
        if not turns_raw:
            raise Mem2ActContractError(f"{session_id}.turns must not be empty")
        turns: list[dict[str, Any]] = []
        for turn_index, item in enumerate(turns_raw):
            turn = _required_object(item, f"{session_id}.turns[{turn_index}]")
            _required_text(turn.get("role"), f"{session_id}.turns[{turn_index}].role")
            content = turn.get("content")
            if content is not None and not isinstance(content, str):
                raise Mem2ActContractError(
                    f"{session_id}.turns[{turn_index}].content must be a string or null"
                )
            if content is None and not (
                isinstance(turn.get("tool_calls"), list) and turn["tool_calls"]
            ):
                raise Mem2ActContractError(
                    f"{session_id}.turns[{turn_index}] without content requires tool calls"
                )
            turn_source = _required_text(
                turn.get("source_id"), f"{session_id}.turns[{turn_index}].source_id"
            )
            if turn_source not in sources:
                raise Mem2ActContractError(
                    f"{session_id}.turns[{turn_index}] names an undeclared source ID"
                )
            turns.append(turn)

        turn_count = _required_int(raw.get("turn_count"), f"{session_id}.turn_count")
        user_turns = sum(turn["role"] == "user" for turn in turns)
        if turn_count != user_turns:
            raise Mem2ActContractError(
                f"{session_id}.turn_count={turn_count} does not match {user_turns} user turns"
            )
        if not isinstance(raw.get("has_tool_calls"), bool):
            raise Mem2ActContractError(f"{session_id}.has_tool_calls must be boolean")
        sessions.append(
            CorpusSession(
                session_id=session_id,
                original_conversation_ids=sources,
                turns=tuple(turns),
                turn_count=turn_count,
                token_count=_required_int(raw.get("token_count"), f"{session_id}.token_count"),
            )
        )
    return tuple(sessions)


def _validate_statistics(raw: dict[str, Any], *, task_count: int, session_count: int) -> None:
    if _required_int(raw.get("total_sessions"), "statistics.total_sessions") != session_count:
        raise Mem2ActContractError("statistics total_sessions does not match the corpus")
    if _required_int(raw.get("total_qa"), "statistics.total_qa") != task_count:
        raise Mem2ActContractError("statistics total_qa does not match the task file")


def _validate_source_references(
    tasks: tuple[Mem2ActTask, ...],
    sessions: tuple[CorpusSession, ...],
    spec: DatasetSpec,
) -> frozenset[str]:
    corpus_sources = {
        source for session in sessions for source in session.original_conversation_ids
    }
    task_sources = {source for task in tasks for source in task.source_conversation_ids}
    unresolved = frozenset(task_sources.difference(corpus_sources))
    if unresolved != spec.allowed_unresolved_source_ids:
        unexpected = sorted(unresolved.difference(spec.allowed_unresolved_source_ids))
        repaired = sorted(spec.allowed_unresolved_source_ids.difference(unresolved))
        raise Mem2ActContractError(
            "source-reference integrity changed; "
            f"unexpected gaps={unexpected}, repaired pinned gaps={repaired}"
        )
    return unresolved


def _tool_catalog(tasks: tuple[Mem2ActTask, ...]) -> tuple[tuple[dict[str, Any], ...], str]:
    by_canonical: dict[str, dict[str, Any]] = {}
    for task in tasks:
        encoded = canonical_json(task.target_tool_schema)
        by_canonical.setdefault(encoded, task.target_tool_schema)
    entries: list[dict[str, Any]] = []
    for encoded, schema in sorted(
        by_canonical.items(), key=lambda item: (str(item[1].get("name", "")), item[0])
    ):
        entries.append(
            {
                "schema_id": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "schema": schema,
            }
        )
    catalog = tuple(entries)
    catalog_hash = hashlib.sha256(canonical_json(catalog).encode("utf-8")).hexdigest()
    return catalog, catalog_hash


__all__ = [
    "DatasetSpec",
    "KNOWN_TOOL_NAME_REPAIRS",
    "MEM2ACT_REPO_COMMIT",
    "OFFICIAL_MEM2ACT_SPEC",
    "canonical_json",
    "load_mem2act_dataset",
    "sha256_file",
]
