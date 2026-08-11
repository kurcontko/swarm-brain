from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from swarmbrain.retrieval.temporal_query import (
    TEMPORAL_QUERY_PARSER_VERSION,
    ClosedReferencedTime,
    TemporalConfidence,
    TemporalParseReason,
    TemporalParseStatus,
    parse_referenced_time,
)


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_parser_version_is_stable_for_benchmark_metadata() -> None:
    assert TEMPORAL_QUERY_PARSER_VERSION == "conservative-referenced-time-v1"


@pytest.mark.parametrize(
    ("query", "expected_from", "expected_to"),
    [
        ("what happened on 2024-02-29?", _utc(2024, 2, 29), _utc(2024, 3, 1)),
        ("events in 2024-02", _utc(2024, 2, 1), _utc(2024, 3, 1)),
        ("changes during 2024", _utc(2024, 1, 1), _utc(2025, 1, 1)),
        ("2025-12-31", _utc(2025, 12, 31), _utc(2026, 1, 1)),
        ("2025-12", _utc(2025, 12, 1), _utc(2026, 1, 1)),
        ("2025", _utc(2025, 1, 1), _utc(2026, 1, 1)),
    ],
)
def test_explicit_iso_periods_normalize_to_half_open_intervals(
    query: str,
    expected_from: datetime,
    expected_to: datetime,
) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.confidence is TemporalConfidence.HIGH
    assert result.reason is TemporalParseReason.EXPLICIT_PERIOD
    assert result.relative is False
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_to
    assert result.closed_referenced_time == ClosedReferencedTime(
        referenced_valid_from=expected_from,
        referenced_valid_to=expected_to,
    )


def test_temporal_operators_are_case_insensitive() -> None:
    result = parse_referenced_time("BEFORE 2024-02-01", timezone=UTC)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.reason is TemporalParseReason.EXPLICIT_BEFORE
    assert result.interval is not None
    assert result.interval.valid_from is None
    assert result.interval.valid_to == _utc(2024, 2, 1)


@pytest.mark.parametrize(
    ("query", "expected_from", "expected_to"),
    [
        (
            "between 2024-01-30 and 2024-02-02",
            _utc(2024, 1, 30),
            _utc(2024, 2, 3),
        ),
        ("from 2023-11 to 2024-02", _utc(2023, 11, 1), _utc(2024, 3, 1)),
        ("from 2023 to 2024", _utc(2023, 1, 1), _utc(2025, 1, 1)),
    ],
)
def test_explicit_ranges_include_the_final_calendar_period(
    query: str,
    expected_from: datetime,
    expected_to: datetime,
) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.reason is TemporalParseReason.EXPLICIT_RANGE
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_to
    assert result.closed_referenced_time is not None


def test_before_and_after_preserve_open_bounds_and_cannot_route_today() -> None:
    before = parse_referenced_time("before 2024-03", timezone=UTC)
    after = parse_referenced_time("after 2024-03", timezone=UTC)

    assert before.status is TemporalParseStatus.MATCHED
    assert before.reason is TemporalParseReason.EXPLICIT_BEFORE
    assert before.interval is not None
    assert before.interval.valid_from is None
    assert before.interval.valid_to == _utc(2024, 3, 1)
    assert before.closed_referenced_time is None
    assert "RecallQuery requires both" in before.routing_reason

    assert after.status is TemporalParseStatus.MATCHED
    assert after.reason is TemporalParseReason.EXPLICIT_AFTER
    assert after.interval is not None
    assert after.interval.valid_from == _utc(2024, 4, 1)
    assert after.interval.valid_to is None
    assert after.closed_referenced_time is None
    assert "open temporal intervals are non-routable" in after.routing_reason


@pytest.mark.parametrize(
    ("expression", "expected_from", "expected_to"),
    [
        ("today", _utc(2026, 8, 9), _utc(2026, 8, 10)),
        ("yesterday", _utc(2026, 8, 8), _utc(2026, 8, 9)),
        ("tomorrow", _utc(2026, 8, 10), _utc(2026, 8, 11)),
        ("this week", _utc(2026, 8, 3), _utc(2026, 8, 10)),
        ("last week", _utc(2026, 7, 27), _utc(2026, 8, 3)),
        ("next week", _utc(2026, 8, 10), _utc(2026, 8, 17)),
        ("this month", _utc(2026, 8, 1), _utc(2026, 9, 1)),
        ("last month", _utc(2026, 7, 1), _utc(2026, 8, 1)),
        ("next month", _utc(2026, 9, 1), _utc(2026, 10, 1)),
        ("this year", _utc(2026, 1, 1), _utc(2027, 1, 1)),
        ("last year", _utc(2025, 1, 1), _utc(2026, 1, 1)),
        ("next year", _utc(2027, 1, 1), _utc(2028, 1, 1)),
    ],
)
def test_relative_periods_use_only_the_explicit_now_anchor(
    expression: str,
    expected_from: datetime,
    expected_to: datetime,
) -> None:
    anchor = datetime(2026, 8, 9, 19, 22, tzinfo=UTC)

    result = parse_referenced_time(f"during {expression}", timezone=UTC, now=anchor)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.confidence is TemporalConfidence.MEDIUM
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_PERIOD
    assert result.relative is True
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_to


def test_relative_range_is_anchored_and_discloses_lower_confidence() -> None:
    anchor = datetime(2026, 8, 9, 12, tzinfo=UTC)

    result = parse_referenced_time(
        "from last month to this month",
        timezone=UTC,
        now=anchor,
    )

    assert result.status is TemporalParseStatus.MATCHED
    assert result.confidence is TemporalConfidence.MEDIUM
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_RANGE
    assert result.interval is not None
    assert result.interval.valid_from == _utc(2026, 7, 1)
    assert result.interval.valid_to == _utc(2026, 9, 1)


def test_relative_open_interval_is_parsed_but_remains_non_routable() -> None:
    anchor = datetime(2026, 8, 9, 12, tzinfo=UTC)

    result = parse_referenced_time("after yesterday", timezone=UTC, now=anchor)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.confidence is TemporalConfidence.MEDIUM
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_AFTER
    assert result.interval is not None
    assert result.interval.valid_from == _utc(2026, 8, 9)
    assert result.interval.valid_to is None
    assert result.closed_referenced_time is None


@pytest.mark.parametrize(
    "query",
    [
        "today",
        "during last month",
        "before next year",
        "10 days ago",
        "last Tuesday",
        "past two weeks",
        "since three weeks ago",
        "this weekend",
    ],
)
def test_relative_language_is_rejected_without_an_explicit_now(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.REJECTED
    assert result.confidence is TemporalConfidence.NONE
    assert result.reason is TemporalParseReason.RELATIVE_NOW_REQUIRED
    assert result.interval is None


@pytest.mark.parametrize(
    ("query", "anchor", "expected_from", "expected_to"),
    [
        (
            "What kitchen appliance did I buy 10 days ago?",
            _utc(2023, 3, 25),
            _utc(2023, 3, 15),
            _utc(2023, 3, 16),
        ),
        (
            "What was the social media activity I participated five days ago?",
            _utc(2023, 3, 20),
            _utc(2023, 3, 15),
            _utc(2023, 3, 16),
        ),
        (
            "Which book did I finish a week ago?",
            _utc(2023, 2, 7),
            _utc(2023, 1, 31),
            _utc(2023, 2, 1),
        ),
        (
            "I mentioned participating in a sports event two weeks ago. What was the event?",
            _utc(2023, 7, 1),
            _utc(2023, 6, 17),
            _utc(2023, 6, 18),
        ),
        (
            "I mentioned visiting a museum two months ago. Did I visit with a friend?",
            _utc(2023, 3, 11),
            _utc(2023, 1, 11),
            _utc(2023, 1, 12),
        ),
        (
            "What happened one year ago?",
            _utc(2024, 2, 29),
            _utc(2023, 2, 28),
            _utc(2023, 3, 1),
        ),
    ],
)
def test_longmemeval_style_counted_ago_selects_one_target_calendar_day(
    query: str,
    anchor: datetime,
    expected_from: datetime,
    expected_to: datetime,
) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=anchor)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.confidence is TemporalConfidence.MEDIUM
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_PERIOD
    assert result.relative is True
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_to
    assert result.closed_referenced_time is not None


@pytest.mark.parametrize(
    ("query", "expected_match"),
    [
        ("What changed since three weeks ago?", "since three weeks ago"),
        (
            "How many total pieces of writing have I completed since I started writing "
            "again three weeks ago, including short stories, poems, and pieces for the "
            "writing challenge?",
            "since I started writing again three weeks ago",
        ),
    ],
)
def test_since_counted_ago_is_a_window_through_the_anchor_day(
    query: str,
    expected_match: str,
) -> None:
    result = parse_referenced_time(
        query,
        timezone=UTC,
        now=datetime(2023, 5, 30, 17, 14, tzinfo=UTC),
    )

    assert result.status is TemporalParseStatus.MATCHED
    assert result.confidence is TemporalConfidence.MEDIUM
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_WINDOW
    assert result.relative is True
    assert result.matched_text == expected_match
    assert result.interval is not None
    assert result.interval.valid_from == _utc(2023, 5, 9)
    assert result.interval.valid_to == _utc(2023, 5, 31)
    assert result.closed_referenced_time is not None


def test_since_month_ago_clamps_calendar_day_and_preserves_dst_boundaries() -> None:
    new_york = ZoneInfo("America/New_York")
    anchor = datetime(2024, 3, 31, 18, tzinfo=new_york)

    result = parse_referenced_time(
        "What changed since one month ago?",
        timezone=new_york,
        now=anchor,
    )

    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_WINDOW
    assert result.interval is not None
    assert result.interval.valid_from == datetime(2024, 2, 29, tzinfo=new_york)
    assert result.interval.valid_to == datetime(2024, 4, 1, tzinfo=new_york)
    assert result.interval.valid_to.astimezone(UTC) - result.interval.valid_from.astimezone(
        UTC
    ) == timedelta(days=32, hours=-1)


@pytest.mark.parametrize(
    ("query", "anchor", "expected_from", "expected_to"),
    [
        (
            "Who did I meet with during lunch last Tuesday?",
            _utc(2023, 4, 18),
            _utc(2023, 4, 11),
            _utc(2023, 4, 12),
        ),
        (
            "What artist did I start listening to last Friday?",
            _utc(2023, 4, 5),
            _utc(2023, 3, 31),
            _utc(2023, 4, 1),
        ),
        (
            "I received a piece of jewelry last Saturday from whom?",
            _utc(2023, 3, 9),
            _utc(2023, 3, 4),
            _utc(2023, 3, 5),
        ),
    ],
)
def test_last_weekday_is_always_the_strictly_previous_named_day(
    query: str,
    anchor: datetime,
    expected_from: datetime,
    expected_to: datetime,
) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=anchor)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_PERIOD
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_to


@pytest.mark.parametrize(
    ("phrase", "anchor", "expected_from", "expected_to"),
    [
        ("past weekend", _utc(2023, 3, 21), _utc(2023, 3, 18), _utc(2023, 3, 20)),
        ("last weekend", _utc(2023, 3, 20), _utc(2023, 3, 18), _utc(2023, 3, 20)),
        ("last weekend", _utc(2023, 3, 19), _utc(2023, 3, 11), _utc(2023, 3, 13)),
        ("this weekend", _utc(2023, 3, 15), _utc(2023, 3, 18), _utc(2023, 3, 20)),
        ("this weekend", _utc(2023, 3, 19), _utc(2023, 3, 18), _utc(2023, 3, 20)),
    ],
)
def test_weekend_phrases_use_saturday_to_monday_exclusive_iso_week_semantics(
    phrase: str,
    anchor: datetime,
    expected_from: datetime,
    expected_to: datetime,
) -> None:
    result = parse_referenced_time(f"What did I do {phrase}?", timezone=UTC, now=anchor)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_PERIOD
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_to


@pytest.mark.parametrize(
    ("query", "anchor", "expected_from", "expected_to"),
    [
        (
            "How many times did I bake something in the past two weeks?",
            _utc(2023, 5, 30),
            _utc(2023, 5, 17),
            _utc(2023, 5, 31),
        ),
        (
            "Show activity from the past 10 days",
            _utc(2023, 5, 30),
            _utc(2023, 5, 21),
            _utc(2023, 5, 31),
        ),
        (
            "How many ceremonies in the past three months?",
            _utc(2023, 7, 21),
            _utc(2023, 4, 21),
            _utc(2023, 7, 22),
        ),
    ],
)
def test_past_counted_windows_include_the_anchor_calendar_day(
    query: str,
    anchor: datetime,
    expected_from: datetime,
    expected_to: datetime,
) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=anchor)

    assert result.status is TemporalParseStatus.MATCHED
    assert result.confidence is TemporalConfidence.MEDIUM
    assert result.reason is TemporalParseReason.ANCHORED_RELATIVE_WINDOW
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_to


@pytest.mark.parametrize(
    ("phrase", "expected_from"),
    [
        ("twenty-one days ago", _utc(2026, 7, 19)),
        ("one hundred and one days ago", _utc(2026, 4, 30)),
        ("one-hundred days ago", _utc(2026, 5, 1)),
        ("1000 days ago", _utc(2023, 11, 13)),
    ],
)
def test_bounded_cardinal_and_word_numbers_are_deterministic(
    phrase: str,
    expected_from: datetime,
) -> None:
    result = parse_referenced_time(phrase, timezone=UTC, now=_utc(2026, 8, 9))

    assert result.status is TemporalParseStatus.MATCHED
    assert result.interval is not None
    assert result.interval.valid_from == expected_from
    assert result.interval.valid_to == expected_from + timedelta(days=1)


@pytest.mark.parametrize(
    "query",
    [
        "0 days ago",
        "1001 days ago",
        "521 weeks ago",
        "121 months ago",
        "101 years ago",
        "past 1001 days",
        "past 121 months",
        "since 1001 days ago",
    ],
)
def test_relative_counts_are_bounded(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=_utc(2026, 8, 9))

    assert result.status is TemporalParseStatus.REJECTED
    assert result.reason is TemporalParseReason.RELATIVE_COUNT_OUT_OF_RANGE
    assert result.interval is None


@pytest.mark.parametrize(
    "query",
    ["1 days ago", "two day ago", "a weeks ago", "past 1 weeks", "since 1 days ago"],
)
def test_relative_count_and_unit_must_agree(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=_utc(2026, 8, 9))

    assert result.status is TemporalParseStatus.REJECTED
    assert result.reason is TemporalParseReason.INVALID_RELATIVE_COUNT
    assert result.interval is None


@pytest.mark.parametrize(
    "query",
    [
        "a couple of days ago",
        "a few weeks ago",
        "several months ago",
        "during the past few months",
    ],
)
def test_vague_relative_quantities_are_rejected(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=_utc(2026, 8, 9))

    assert result.status is TemporalParseStatus.REJECTED
    assert result.reason is TemporalParseReason.UNSUPPORTED_TEMPORAL_EXPRESSION
    assert result.interval is None


@pytest.mark.parametrize(
    "query",
    [
        "two days ago and last Friday",
        "past two weeks or past three months",
        "last Tuesday and Wednesday",
        "5 days ago in 2024",
        "since three weeks ago and two days ago",
        "since I started in 2024 three weeks ago",
    ],
)
def test_mixed_or_multiple_relative_expressions_are_rejected(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=_utc(2026, 8, 9))

    assert result.status is TemporalParseStatus.REJECTED
    assert result.reason is TemporalParseReason.MULTIPLE_TEMPORAL_EXPRESSIONS
    assert result.interval is None


@pytest.mark.parametrize(
    "query",
    [
        "How many days passed between event A and event B?",
        "How many days passed between the day I started watering my herb garden and the day I harvested herbs?",
        "How many weeks passed between buying my racket and receiving it?",
        "How many days passed between the Sunday mass and the Ash Wednesday service?",
        "How many days have passed since three weeks ago?",
    ],
)
def test_elapsed_time_comparison_questions_do_not_hallucinate_a_filter(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=_utc(2026, 8, 9))

    assert result.status is TemporalParseStatus.NO_MATCH
    assert result.reason is TemporalParseReason.NO_TEMPORAL_EXPRESSION
    assert result.interval is None


def test_anchor_is_converted_to_the_requested_calendar_timezone() -> None:
    los_angeles = ZoneInfo("America/Los_Angeles")
    # Still August 8 in Los Angeles.
    anchor = datetime(2026, 8, 9, 2, 30, tzinfo=UTC)

    result = parse_referenced_time("today", timezone=los_angeles, now=anchor)

    assert result.interval is not None
    assert result.interval.valid_from == datetime(2026, 8, 8, tzinfo=los_angeles)
    assert result.interval.valid_to == datetime(2026, 8, 9, tzinfo=los_angeles)


def test_calendar_day_boundaries_preserve_dst_instead_of_assuming_24_hours() -> None:
    new_york = ZoneInfo("America/New_York")

    result = parse_referenced_time("on 2024-03-10", timezone=new_york)

    assert result.interval is not None
    assert result.interval.valid_from == datetime(2024, 3, 10, tzinfo=new_york)
    assert result.interval.valid_to == datetime(2024, 3, 11, tzinfo=new_york)
    assert result.interval.valid_to.astimezone(UTC) - result.interval.valid_from.astimezone(
        UTC
    ) == timedelta(hours=23)


def test_relative_windows_also_preserve_local_dst_calendar_boundaries() -> None:
    new_york = ZoneInfo("America/New_York")
    anchor = datetime(2024, 3, 11, 18, tzinfo=new_york)

    result = parse_referenced_time("past two days", timezone=new_york, now=anchor)

    assert result.interval is not None
    assert result.interval.valid_from == datetime(2024, 3, 10, tzinfo=new_york)
    assert result.interval.valid_to == datetime(2024, 3, 12, tzinfo=new_york)
    assert result.interval.valid_to.astimezone(UTC) - result.interval.valid_from.astimezone(
        UTC
    ) == timedelta(hours=47)


@pytest.mark.parametrize(
    "query",
    [
        "on 2023-02-29",
        "in 2024-13",
        "on 2024-04-31",
        "during 0000",
        "during 9999",
    ],
)
def test_invalid_calendar_values_fail_closed(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.REJECTED
    assert result.confidence is TemporalConfidence.NONE
    assert result.reason is TemporalParseReason.INVALID_DATE
    assert result.interval is None


@pytest.mark.parametrize("query", ["on 2024-2-03", "during 24-01"])
def test_malformed_or_unsupported_temporal_language_fails_closed(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=datetime(2026, 8, 9, tzinfo=UTC))

    assert result.status is TemporalParseStatus.REJECTED
    assert result.confidence is TemporalConfidence.NONE
    assert result.reason is TemporalParseReason.UNSUPPORTED_TEMPORAL_EXPRESSION


def test_valid_date_mixed_with_unsupported_relative_language_is_conflicting() -> None:
    result = parse_referenced_time(
        "from 2024 to now",
        timezone=UTC,
        now=datetime(2026, 8, 9, tzinfo=UTC),
    )

    assert result.status is TemporalParseStatus.REJECTED
    assert result.confidence is TemporalConfidence.NONE
    assert result.reason is TemporalParseReason.MULTIPLE_TEMPORAL_EXPRESSIONS


@pytest.mark.parametrize(
    "query",
    [
        "between 2025 and 2024",
        "from 2024-03-01 to 2024-02-29",
    ],
)
def test_reversed_ranges_fail_closed(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.REJECTED
    assert result.reason is TemporalParseReason.REVERSED_RANGE


@pytest.mark.parametrize(
    "query",
    [
        "in 2024 before 2025",
        "compare 2024-01-01 and 2024-01-02",
        "from 2024 to 2025 and on 2026-01-01",
        "in 2024 and 2025-2-01",
    ],
)
def test_multiple_or_conflicting_dates_fail_closed(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.REJECTED
    assert result.reason is TemporalParseReason.MULTIPLE_TEMPORAL_EXPRESSIONS
    assert result.interval is None


@pytest.mark.parametrize(
    "query",
    [
        "what is the current deployment status?",
        "inspect release-v2024",
        "read issue_2024",
    ],
)
def test_non_temporal_text_does_not_create_a_filter(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.NO_MATCH
    assert result.confidence is TemporalConfidence.NONE
    assert result.reason is TemporalParseReason.NO_TEMPORAL_EXPRESSION
    assert result.interval is None
    assert result.closed_referenced_time is None


@pytest.mark.parametrize("query", ["ticket 2024", "release 2024", "Q1 2024"])
def test_bare_year_in_prose_is_rejected_as_ambiguous(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC)

    assert result.status is TemporalParseStatus.REJECTED
    assert result.confidence is TemporalConfidence.NONE
    assert result.reason is TemporalParseReason.AMBIGUOUS_BARE_YEAR


@pytest.mark.parametrize("query", ["now", "next Tuesday", "2 days later"])
def test_unsupported_relative_phrases_are_rejected_instead_of_guessed(query: str) -> None:
    result = parse_referenced_time(query, timezone=UTC, now=datetime(2026, 8, 9, tzinfo=UTC))

    assert result.status is TemporalParseStatus.REJECTED
    assert result.reason is TemporalParseReason.UNSUPPORTED_TEMPORAL_EXPRESSION


def test_naive_now_and_invalid_timezone_fail_closed() -> None:
    naive = parse_referenced_time(
        "today",
        timezone=UTC,
        now=datetime(2026, 8, 9),
    )
    invalid_zone = parse_referenced_time("on 2024-01-01", timezone=None)  # type: ignore[arg-type]

    assert naive.status is TemporalParseStatus.REJECTED
    assert naive.reason is TemporalParseReason.INVALID_NOW_ANCHOR
    assert invalid_zone.status is TemporalParseStatus.REJECTED
    assert invalid_zone.reason is TemporalParseReason.INVALID_TIMEZONE


def test_result_is_deterministic_and_exposes_only_referenced_validity() -> None:
    anchor = datetime(2026, 8, 9, 12, tzinfo=UTC)

    first = parse_referenced_time("during this month", timezone=UTC, now=anchor)
    second = parse_referenced_time("during this month", timezone=UTC, now=anchor)

    assert first == second
    assert first.closed_referenced_time is not None
    assert {field.name for field in fields(first.closed_referenced_time)} == {
        "referenced_valid_from",
        "referenced_valid_to",
    }
    assert not hasattr(first.closed_referenced_time, "world_at")
    assert not hasattr(first.closed_referenced_time, "recorded_at")
    assert (
        first.routing_reason == "closed half-open interval is available for explicit opt-in routing"
    )
