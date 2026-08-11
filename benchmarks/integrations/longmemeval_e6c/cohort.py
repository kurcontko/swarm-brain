"""Outcome-blind cohort selection for E6c.

The selector excludes every LongMemEval-S record used by the original E6
pilot or E6b development run, samples only from the remaining records, and
binds complete question-local histories before any retrieval or QA outcome is
created.  It intentionally uses metadata only for seed qualification.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

E6C_SELECTION_PROTOCOL: Final = (
    "python-3.12-random-sample-sorted-from-answer-evidence-disjoint-eligible-v1"
)
E6C_PYTHON_VERSION: Final = "3.12.13"
E6C_SAMPLE: Final = 160
E6C_SEED_SEARCH_START: Final = 20_260_811
E6C_SEED: Final = 20_564_941
E6C_ABSTENTION_COUNT: Final = 9

E6B_SAMPLE: Final = 160
E6B_SEED: Final = 20_282_059
ORIGINAL_E6_PILOT_POSITIONS: Final = frozenset({17, 29, 160, 169, 185, 221, 228, 394, 422, 478})

E6C_TYPE_COUNTS: Final = {
    "knowledge-update": 25,
    "multi-session": 42,
    "single-session-assistant": 18,
    "single-session-preference": 9,
    "single-session-user": 22,
    "temporal-reasoning": 44,
}

_PRE_FILTER_E6C_POSITIONS: Final = (
    1,
    5,
    9,
    14,
    15,
    16,
    18,
    21,
    27,
    28,
    32,
    35,
    38,
    41,
    42,
    43,
    46,
    47,
    50,
    53,
    63,
    67,
    72,
    76,
    77,
    78,
    88,
    91,
    96,
    98,
    101,
    102,
    106,
    118,
    119,
    120,
    122,
    125,
    129,
    131,
    132,
    133,
    140,
    141,
    142,
    143,
    144,
    152,
    157,
    162,
    163,
    165,
    171,
    174,
    180,
    181,
    183,
    190,
    193,
    195,
    196,
    197,
    199,
    203,
    204,
    212,
    213,
    215,
    217,
    218,
    225,
    227,
    230,
    233,
    239,
    240,
    242,
    245,
    247,
    252,
    254,
    255,
    256,
    262,
    266,
    270,
    274,
    277,
    279,
    280,
    286,
    287,
    288,
    290,
    293,
    295,
    299,
    300,
    304,
    305,
    307,
    309,
    310,
    314,
    315,
    321,
    323,
    327,
    335,
    336,
    337,
    339,
    348,
    352,
    354,
    356,
    360,
    366,
    367,
    372,
    373,
    375,
    378,
    379,
    381,
    385,
    386,
    391,
    395,
    402,
    404,
    406,
    416,
    418,
    420,
    423,
    430,
    431,
    435,
    437,
    438,
    441,
    447,
    449,
    454,
    455,
    456,
    458,
    471,
    472,
    474,
    477,
    482,
    483,
    486,
    489,
    491,
    494,
    497,
    499,
)

# The pre-filter selection above was never executed.  A source-only leakage
# audit found direct answer-evidence reuse in 30 otherwise unused questions;
# the frozen cohort below is selected from the remaining 300-position pool.
E6C_POSITIONS: Final = (
    1,
    6,
    14,
    16,
    21,
    22,
    23,
    24,
    28,
    33,
    34,
    35,
    37,
    41,
    43,
    45,
    49,
    50,
    51,
    63,
    67,
    68,
    70,
    74,
    76,
    81,
    82,
    84,
    87,
    88,
    94,
    95,
    101,
    102,
    108,
    109,
    111,
    112,
    115,
    119,
    129,
    133,
    134,
    141,
    144,
    148,
    152,
    157,
    158,
    161,
    162,
    163,
    165,
    171,
    172,
    174,
    176,
    178,
    182,
    184,
    187,
    188,
    190,
    191,
    192,
    196,
    197,
    202,
    204,
    212,
    215,
    220,
    232,
    233,
    234,
    244,
    245,
    246,
    247,
    248,
    252,
    255,
    257,
    262,
    268,
    272,
    273,
    279,
    281,
    282,
    287,
    293,
    306,
    307,
    308,
    309,
    310,
    311,
    314,
    316,
    318,
    320,
    321,
    322,
    324,
    327,
    335,
    336,
    338,
    342,
    344,
    346,
    348,
    350,
    352,
    360,
    361,
    366,
    372,
    375,
    378,
    381,
    382,
    385,
    386,
    390,
    392,
    398,
    402,
    410,
    414,
    415,
    418,
    421,
    426,
    427,
    428,
    430,
    435,
    437,
    438,
    441,
    447,
    448,
    453,
    455,
    457,
    458,
    465,
    467,
    470,
    472,
    477,
    482,
    489,
    491,
    494,
    497,
    498,
    499,
)


class E6CCohortError(ValueError):
    """The source or selected cohort differs from the frozen E6c contract."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise E6CCohortError("cohort binding is not finite canonical UTF-8 JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _e6b_positions(total: int) -> tuple[int, ...]:
    return tuple(sorted(random.Random(E6B_SEED).sample(range(total), E6B_SAMPLE)))


def excluded_positions(total: int) -> tuple[int, ...]:
    """Return prior development positions, before answer-evidence filtering."""

    if total != 500:
        raise E6CCohortError("E6c is frozen to the 500-question LongMemEval-S source")
    return tuple(sorted(set(_e6b_positions(total)) | ORIGINAL_E6_PILOT_POSITIONS))


def remaining_positions(total: int) -> tuple[int, ...]:
    """Return 330 questions not used directly by E6/E6b."""

    excluded = set(excluded_positions(total))
    remaining = tuple(position for position in range(total) if position not in excluded)
    if len(excluded) != 170 or len(remaining) != 330:
        raise E6CCohortError("E6c exclusion or remaining-pool cardinality drifted")
    return remaining


def _answer_evidence_families(record: Mapping[str, Any]) -> dict[str, set[str]]:
    session_ids = record.get("haystack_session_ids")
    dates = record.get("haystack_dates")
    sessions = record.get("haystack_sessions")
    answers = record.get("answer_session_ids")
    if not all(isinstance(value, list) for value in (session_ids, dates, sessions, answers)):
        raise E6CCohortError("record has malformed answer-evidence fields")
    if not len(session_ids) == len(dates) == len(sessions):
        raise E6CCohortError("history IDs, dates, and sessions differ in length")
    answer_ids = set(answers)
    matched: set[str] = set()
    families: dict[str, set[str]] = {
        "ids": set(),
        "payloads": set(),
        "turn_arrays": set(),
        "turns": set(),
    }
    for session_id, date, turns in zip(session_ids, dates, sessions, strict=True):
        if session_id not in answer_ids:
            continue
        if (
            not isinstance(session_id, str)
            or not isinstance(date, str)
            or not isinstance(turns, list)
        ):
            raise E6CCohortError("answer-evidence session is malformed")
        matched.add(session_id)
        families["ids"].add(session_id)
        families["payloads"].add(
            sha256_json(
                {
                    "parent_session_id": session_id,
                    "parent_session_date": date,
                    "turns": turns,
                }
            )
        )
        families["turn_arrays"].add(sha256_json(turns))
        families["turns"].update(sha256_json(turn) for turn in turns)
    if matched != answer_ids:
        raise E6CCohortError("not every answer session ID resolves to source evidence")
    return families


def answer_evidence_overlap_positions(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    prior = excluded_positions(len(records))
    used: dict[str, set[str]] = {
        "ids": set(),
        "payloads": set(),
        "turn_arrays": set(),
        "turns": set(),
    }
    for position in prior:
        families = _answer_evidence_families(records[position])
        for family in used:
            used[family].update(families[family])
    overlap: list[int] = []
    for position in remaining_positions(len(records)):
        families = _answer_evidence_families(records[position])
        if any(families[family] & used[family] for family in used):
            overlap.append(position)
    return tuple(overlap)


def eligible_positions(records: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    excluded = set(excluded_positions(len(records))) | set(
        answer_evidence_overlap_positions(records)
    )
    eligible = tuple(position for position in range(len(records)) if position not in excluded)
    if len(excluded) != 200 or len(eligible) != 300:
        raise E6CCohortError("answer-evidence filtering cardinality drifted")
    return eligible


def selected_positions(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = E6C_SEED,
) -> tuple[int, ...]:
    eligible = eligible_positions(records)
    return tuple(sorted(random.Random(seed).sample(eligible, E6C_SAMPLE)))


def _abstention(record: Mapping[str, Any]) -> bool:
    return "_abs" in str(record.get("question_id", ""))


def _summary(records: Sequence[Mapping[str, Any]], positions: Sequence[int]) -> dict[str, Any]:
    return {
        "type_counts": dict(
            sorted(
                Counter(str(records[position]["question_type"]) for position in positions).items()
            )
        ),
        "abstention_count": sum(_abstention(records[position]) for position in positions),
    }


def selection_qualifies(
    records: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
) -> bool:
    if len(positions) != E6C_SAMPLE or tuple(positions) != tuple(sorted(set(positions))):
        return False
    if not set(positions).issubset(eligible_positions(records)):
        return False
    summary = _summary(records, positions)
    return bool(
        summary["type_counts"] == E6C_TYPE_COUNTS
        and summary["abstention_count"] == E6C_ABSTENTION_COUNT
    )


def first_qualifying_seed(records: Sequence[Mapping[str, Any]]) -> int:
    eligible = eligible_positions(records)
    for seed in range(E6C_SEED_SEARCH_START, E6C_SEED + 1):
        positions = tuple(sorted(random.Random(seed).sample(eligible, E6C_SAMPLE)))
        summary = _summary(records, positions)
        if (
            summary["type_counts"] == E6C_TYPE_COUNTS
            and summary["abstention_count"] == E6C_ABSTENTION_COUNT
        ):
            return seed
    raise E6CCohortError("no E6c seed satisfies the frozen search interval")


def corpus_fingerprint(record: Mapping[str, Any]) -> str:
    """Hash the complete question-local history, excluding query and labels."""

    required = ("haystack_session_ids", "haystack_dates", "haystack_sessions")
    if any(field not in record for field in required):
        raise E6CCohortError("record is missing a complete history field")
    return sha256_json({field: record[field] for field in required})


def answer_evidence_signature(record: Mapping[str, Any]) -> str:
    """Bind answer sessions and their source payloads without answer text."""

    session_ids = record.get("haystack_session_ids")
    sessions = record.get("haystack_sessions")
    answers = record.get("answer_session_ids")
    if (
        not isinstance(session_ids, list)
        or not isinstance(sessions, list)
        or not isinstance(answers, list)
    ):
        raise E6CCohortError("record has malformed answer-evidence fields")
    if len(session_ids) != len(sessions):
        raise E6CCohortError("history IDs and session payloads differ in length")
    if any(value not in session_ids for value in answers):
        raise E6CCohortError("answer sessions do not bind source sessions")
    answer_ids = set(answers)
    evidence = (
        [
            {
                "session_id": session_id,
                "session_position": position,
                "session": sessions[position],
            }
            for position, session_id in enumerate(session_ids)
            if session_id in answer_ids
        ]
        if answers
        else [{"abstention_history_sha256": corpus_fingerprint(record)}]
    )
    return sha256_json(evidence)


def build_cohort_binding(
    records: Sequence[Mapping[str, Any]],
    *,
    verify_first_seed: bool = False,
) -> dict[str, Any]:
    if sys.version.split()[0] != E6C_PYTHON_VERSION:
        raise E6CCohortError(
            f"E6c selection requires CPython {E6C_PYTHON_VERSION}, got {sys.version.split()[0]}"
        )
    if len(records) != 500:
        raise E6CCohortError("E6c source must contain exactly 500 records")
    selected = selected_positions(records)
    if selected != E6C_POSITIONS or not selection_qualifies(records, selected):
        raise E6CCohortError("selected positions differ from the frozen E6c cohort")
    if verify_first_seed and first_qualifying_seed(records) != E6C_SEED:
        raise E6CCohortError("E6c seed is not the first qualifying seed")

    prior_used = excluded_positions(len(records))
    remaining = remaining_positions(len(records))
    overlap = answer_evidence_overlap_positions(records)
    excluded = tuple(sorted(set(prior_used) | set(overlap)))
    eligible = eligible_positions(records)
    if sha256_json(list(overlap)) != (
        "7ef254d9d2ab49456196e1c6687d7452d204e2ff5308acbc70710da7e882a903"
    ):
        raise E6CCohortError("answer-evidence overlap audit drifted")
    prior_fingerprints = [corpus_fingerprint(records[position]) for position in prior_used]
    remaining_fingerprints = [corpus_fingerprint(records[position]) for position in remaining]
    if len(set(prior_fingerprints)) != len(prior_fingerprints):
        raise E6CCohortError("development exclusions repeat a history fingerprint")
    if len(set(remaining_fingerprints)) != len(remaining_fingerprints):
        raise E6CCohortError("remaining pool repeats a history fingerprint")
    if set(prior_fingerprints) & set(remaining_fingerprints):
        raise E6CCohortError("remaining pool overlaps a development history")

    used_evidence: dict[str, set[str]] = {
        "ids": set(),
        "payloads": set(),
        "turn_arrays": set(),
        "turns": set(),
    }
    for position in prior_used:
        families = _answer_evidence_families(records[position])
        for family in used_evidence:
            used_evidence[family].update(families[family])
    if any(
        _answer_evidence_families(records[position])[family] & used_evidence[family]
        for position in eligible
        for family in used_evidence
    ):
        raise E6CCohortError("eligible pool overlaps development answer evidence")

    selected_rows = [
        {
            "abstention": _abstention(records[position]),
            "corpus_fingerprint_sha256": corpus_fingerprint(records[position]),
            "position": position,
            "question_id": str(records[position]["question_id"]),
            "question_type": str(records[position]["question_type"]),
            "record_sha256": sha256_json(records[position]),
            "run_position": run_position,
        }
        for run_position, position in enumerate(selected)
    ]
    selector = {
        "protocol": E6C_SELECTION_PROTOCOL,
        "python_version": E6C_PYTHON_VERSION,
        "total": len(records),
        "prior_used_positions_sha256": sha256_json(list(prior_used)),
        "answer_evidence_overlap_positions_sha256": sha256_json(list(overlap)),
        "excluded_positions_sha256": sha256_json(list(excluded)),
        "eligible_positions_sha256": sha256_json(list(eligible)),
        "search_start_inclusive": E6C_SEED_SEARCH_START,
        "sample": E6C_SAMPLE,
        "target_type_counts": E6C_TYPE_COUNTS,
        "target_abstention_count": E6C_ABSTENTION_COUNT,
        "seed": E6C_SEED,
        "positions": list(selected),
    }
    answer_rows = [
        {
            "position": position,
            "question_id": str(records[position]["question_id"]),
            "families": {
                family: sorted(values)
                for family, values in _answer_evidence_families(records[position]).items()
            },
        }
        for position in selected
    ]
    return {
        "selector": selector,
        "summary": _summary(records, selected),
        "selected_rows": selected_rows,
        "disjointness": {
            "question_ids": True,
            "complete_history_fingerprints": True,
            "answer_evidence_ids_payloads_arrays_and_turns": True,
            "underlying_nonanswer_distractor_content": False,
            "prior_used_history_count": len(prior_fingerprints),
            "remaining_history_count": len(remaining_fingerprints),
            "answer_evidence_overlap_questions_excluded": len(overlap),
            "selected_shared_nonanswer_session_ids_or_id_plus_turns": 992,
            "selected_shared_nonanswer_turn_arrays": 1_089,
            "selected_shared_nonanswer_turns": 11_309,
        },
        "digests": {
            "prior_used_positions_sha256": sha256_json(list(prior_used)),
            "answer_evidence_overlap_positions_sha256": sha256_json(list(overlap)),
            "excluded_positions_sha256": sha256_json(list(excluded)),
            "remaining_positions_sha256": sha256_json(list(remaining)),
            "eligible_positions_sha256": sha256_json(list(eligible)),
            "selected_positions_sha256": sha256_json(list(selected)),
            "selected_question_ids_sha256": sha256_json(
                [str(records[position]["question_id"]) for position in selected]
            ),
            "selected_rows_sha256": sha256_json(selected_rows),
            "selector_sha256": sha256_json(selector),
            "prior_used_history_fingerprints_sha256": sha256_json(prior_fingerprints),
            "remaining_history_fingerprints_sha256": sha256_json(remaining_fingerprints),
            "selected_history_fingerprints_sha256": sha256_json(
                [corpus_fingerprint(records[position]) for position in selected]
            ),
            "selected_record_sha256s_sha256": sha256_json(
                [sha256_json(records[position]) for position in selected]
            ),
            "selected_answer_evidence_rows_sha256": sha256_json(answer_rows),
            "selected_records_canonical_array_sha256": sha256_json(
                [records[position] for position in selected]
            ),
        },
    }


__all__ = [
    "E6C_ABSTENTION_COUNT",
    "E6C_POSITIONS",
    "E6C_PYTHON_VERSION",
    "E6C_SAMPLE",
    "E6C_SEED",
    "E6C_SEED_SEARCH_START",
    "E6C_SELECTION_PROTOCOL",
    "E6C_TYPE_COUNTS",
    "E6CCohortError",
    "answer_evidence_signature",
    "answer_evidence_overlap_positions",
    "build_cohort_binding",
    "canonical_json_bytes",
    "corpus_fingerprint",
    "excluded_positions",
    "eligible_positions",
    "first_qualifying_seed",
    "remaining_positions",
    "selected_positions",
    "selection_qualifies",
    "sha256_json",
]
