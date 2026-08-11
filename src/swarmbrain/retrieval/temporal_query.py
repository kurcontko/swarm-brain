"""Conservative, deterministic parsing of referenced valid-time expressions.

This module is deliberately pure and opt-in.  It does not inspect, construct,
or mutate ``RecallQuery`` objects, and it never produces ``world_at`` or
``recorded_at`` values.  A caller must explicitly choose to copy a *closed*
proposal into ``referenced_valid_from`` and ``referenced_valid_to``.

Only a small, auditable grammar is accepted:

* ISO calendar periods: ``YYYY-MM-DD``, ``YYYY-MM``, and ``YYYY``;
* ``between X and Y`` and ``from X to Y`` (the final calendar period is
  included by normalizing its end to an exclusive boundary);
* ``in X``, ``on X``, ``during X``, ``before X``, and ``after X``;
* ``since N days/weeks/months/years ago`` (including a short intervening
  event clause) through the caller-supplied anchor day; and
* a bounded set of relative periods resolved only against a caller-supplied,
  timezone-aware ``now`` anchor.

Relative calendar semantics are intentionally explicit:

* ``N days/weeks ago`` selects the single local calendar day exactly ``N``
  days or ``7*N`` days before the anchor;
* ``N months/years ago`` selects the corresponding local day after a calendar
  shift, clamping to the target month's final day when necessary;
* ``since N ... ago`` starts on that same shifted target day and includes the
  anchor day;
* ``last <weekday>`` is the most recent *strictly earlier* named weekday;
* a weekend is Saturday through Monday-exclusive, ``last`` and ``past`` mean
  the previous ISO-week weekend, and ``this`` means the anchor ISO week's
  weekend; and
* ``past N days/weeks`` is a trailing window of exactly ``N`` or ``7*N``
  calendar dates including the anchor date. ``past N months`` begins at the
  same-or-clamped local date ``N`` calendar months earlier and includes the
  anchor date.

Counts are bounded to keep accidental filters finite: 1..1000 days, 1..520
weeks, 1..120 months, and 1..100 years.  Digit counts and unambiguous English
number words through one thousand are accepted; vague quantities are not.

Ambiguous, conflicting, malformed, or unsupported temporal language is
rejected rather than guessed.  In particular, open ``before``/``after``
expressions remain open: fabricating year-1/year-9999 sentinels would turn a
correct semantic parse into an unsafe retrieval filter.
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from enum import StrEnum

TEMPORAL_QUERY_PARSER_VERSION = "conservative-referenced-time-v1"


class TemporalParseStatus(StrEnum):
    """Whether a query contains one safe temporal interpretation."""

    MATCHED = "matched"
    NO_MATCH = "no_match"
    REJECTED = "rejected"


class TemporalConfidence(StrEnum):
    """Calibrated confidence class based on grammar, not model probability."""

    HIGH = "high"
    MEDIUM = "medium"
    NONE = "none"


class TemporalParseReason(StrEnum):
    """Stable reason codes suitable for traces and offline evaluation."""

    EXPLICIT_PERIOD = "explicit_period"
    EXPLICIT_RANGE = "explicit_range"
    EXPLICIT_BEFORE = "explicit_before"
    EXPLICIT_AFTER = "explicit_after"
    ANCHORED_RELATIVE_PERIOD = "anchored_relative_period"
    ANCHORED_RELATIVE_WINDOW = "anchored_relative_window"
    ANCHORED_RELATIVE_RANGE = "anchored_relative_range"
    ANCHORED_RELATIVE_BEFORE = "anchored_relative_before"
    ANCHORED_RELATIVE_AFTER = "anchored_relative_after"
    NO_TEMPORAL_EXPRESSION = "no_temporal_expression"
    MULTIPLE_TEMPORAL_EXPRESSIONS = "multiple_temporal_expressions"
    INVALID_DATE = "invalid_date"
    INVALID_TIMEZONE = "invalid_timezone"
    INVALID_NOW_ANCHOR = "invalid_now_anchor"
    RELATIVE_NOW_REQUIRED = "relative_now_required"
    REVERSED_RANGE = "reversed_range"
    RELATIVE_COUNT_OUT_OF_RANGE = "relative_count_out_of_range"
    INVALID_RELATIVE_COUNT = "invalid_relative_count"
    AMBIGUOUS_BARE_YEAR = "ambiguous_bare_year"
    UNSUPPORTED_TEMPORAL_EXPRESSION = "unsupported_temporal_expression"


@dataclass(frozen=True, slots=True)
class ClosedReferencedTime:
    """The only shape that can be copied into today's ``RecallQuery``."""

    referenced_valid_from: datetime
    referenced_valid_to: datetime


@dataclass(frozen=True, slots=True)
class ReferencedTimeInterval:
    """A half-open calendar interval; ``None`` preserves an open boundary."""

    valid_from: datetime | None
    valid_to: datetime | None

    @property
    def is_closed(self) -> bool:
        return self.valid_from is not None and self.valid_to is not None

    def as_closed_referenced_time(self) -> ClosedReferencedTime | None:
        """Return an opt-in routing value only when both bounds are real."""

        if self.valid_from is None or self.valid_to is None:
            return None
        return ClosedReferencedTime(
            referenced_valid_from=self.valid_from,
            referenced_valid_to=self.valid_to,
        )


@dataclass(frozen=True, slots=True)
class TemporalQueryParse:
    """Structured parse result with explicit confidence and failure reason."""

    status: TemporalParseStatus
    confidence: TemporalConfidence
    reason: TemporalParseReason
    interval: ReferencedTimeInterval | None = None
    matched_text: str | None = None
    relative: bool = False

    @property
    def closed_referenced_time(self) -> ClosedReferencedTime | None:
        """Return the current-contract proposal, never an open interval."""

        if self.status is not TemporalParseStatus.MATCHED or self.interval is None:
            return None
        return self.interval.as_closed_referenced_time()

    @property
    def routing_reason(self) -> str:
        """Explain whether this result is safe for current recall routing."""

        if self.status is not TemporalParseStatus.MATCHED:
            return "no referenced-validity proposal is available"
        if self.interval is None or not self.interval.is_closed:
            return (
                "open temporal intervals are non-routable because RecallQuery requires "
                "both referenced_valid_from and referenced_valid_to"
            )
        return "closed half-open interval is available for explicit opt-in routing"


@dataclass(frozen=True, slots=True)
class _CalendarPeriod:
    start: datetime
    end: datetime
    relative: bool
    window: bool = False


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    match: re.Match[str]


class _RelativeParseError(ValueError):
    def __init__(self, reason: TemporalParseReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


_ONES_CORE = r"one|two|three|four|five|six|seven|eight|nine"
_SMALL_NUMBER_CORE = (
    rf"(?:{_ONES_CORE}|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    rf"seventeen|eighteen|nineteen|(?:twenty|thirty|forty|fifty|sixty|seventy|"
    rf"eighty|ninety)(?:[ -](?:{_ONES_CORE}))?)"
)
_HUNDRED_NUMBER_CORE = (
    rf"(?:{_ONES_CORE})[ -]hundred"
    rf"(?:[ -](?:and[ -])?{_SMALL_NUMBER_CORE})?"
)
_WORD_NUMBER_CORE = rf"(?:one[ -]thousand|{_HUNDRED_NUMBER_CORE}|{_SMALL_NUMBER_CORE}|a|an)"
_COUNT_CORE = rf"(?:\d{{1,6}}|{_WORD_NUMBER_CORE})"
_WEEKDAY_CORE = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_COUNTED_AGO_CORE = rf"{_COUNT_CORE}\s+(?:days?|weeks?|months?|years?)\s+ago"
_PAST_WINDOW_CORE = rf"past\s+{_COUNT_CORE}\s+(?:days?|weeks?|months?)"
_MAX_SINCE_CONTEXT_WORDS = 12
_SINCE_CONTEXT_WORD_CORE = r"[\w]+(?:[-'][\w]+)*"
_SINCE_COUNTED_AGO_CORE = (
    rf"since(?P<context>(?:\s+{_SINCE_CONTEXT_WORD_CORE})"
    rf"{{0,{_MAX_SINCE_CONTEXT_WORDS}}}?)\s+"
    rf"(?P<value>{_COUNTED_AGO_CORE})"
)

_ABSOLUTE_CORE = r"\d{4}(?:-\d{2}(?:-\d{2})?)?"
_RELATIVE_CORE = (
    rf"(?:today|yesterday|tomorrow|(?:this|last|next)\s+(?:week|month|year)|"
    rf"last\s+(?:{_WEEKDAY_CORE})|(?:this|last|past)\s+weekend|"
    rf"{_COUNTED_AGO_CORE}|{_PAST_WINDOW_CORE})"
)
# Relative forms go first so a four-digit count in ``1000 days ago`` is not
# prematurely consumed as the valid ISO year ``1000``.
_VALUE_CORE = rf"(?:{_RELATIVE_CORE}|{_ABSOLUTE_CORE})"

_BETWEEN = re.compile(
    rf"(?<!\w)between\s+(?P<left>{_VALUE_CORE})\s+and\s+"
    rf"(?P<right>{_VALUE_CORE})(?![\w-])",
    re.IGNORECASE,
)
_FROM_TO = re.compile(
    rf"(?<!\w)from\s+(?P<left>{_VALUE_CORE})\s+to\s+"
    rf"(?P<right>{_VALUE_CORE})(?![\w-])",
    re.IGNORECASE,
)
_SINCE_COUNTED_AGO = re.compile(
    rf"(?<!\w){_SINCE_COUNTED_AGO_CORE}(?![\w-])",
    re.IGNORECASE,
)
_UNARY = re.compile(
    rf"(?<!\w)(?P<operator>in|on|during|before|after)\s+"
    rf"(?P<value>{_VALUE_CORE})(?![\w-])",
    re.IGNORECASE,
)
_BARE_VALUE = re.compile(rf"(?<![\w-])(?P<value>{_VALUE_CORE})(?![\w-])", re.IGNORECASE)
_ABSOLUTE = re.compile(r"^(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?$")
_COUNTED_AGO = re.compile(
    rf"^(?P<count>{_COUNT_CORE})\s+(?P<unit>days?|weeks?|months?|years?)\s+ago$"
)
_PAST_WINDOW = re.compile(rf"^past\s+(?P<count>{_COUNT_CORE})\s+(?P<unit>days?|weeks?|months?)$")
_LAST_WEEKDAY = re.compile(rf"^last\s+(?P<weekday>{_WEEKDAY_CORE})$")
_WEEKEND = re.compile(r"^(?P<direction>this|last|past)\s+weekend$")
_ELAPSED_SINCE_COMPARISON_PREFIX = re.compile(
    r"(?<!\w)how\s+(?:(?:many\s+(?:days?|weeks?|months?|years?))|"
    r"(?:much\s+time))\s+(?:(?:has|have|had)\s+)?(?:passed|elapsed)\s*$",
    re.IGNORECASE,
)

# These hints are intentionally broader than the accepted grammar.  Their
# presence makes unsupported relative/date-like language a rejection rather
# than a silent "no match" that a caller might misinterpret as current time.
_DATE_SHAPED = re.compile(r"(?<![\w-])\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?(?![\w-])")
_MALFORMED_DATE_HINT = re.compile(
    r"(?<![\w-])(?:\d{1,3}-\d{1,2}(?:-\d{1,2})?|\d{4}-\d{1,2}(?:-\d{1,2})?)(?![\w-])"
)
_RELATIVE_HINT = re.compile(
    rf"(?<!\w)(?:today|yesterday|tomorrow|now|"
    rf"(?:this|last|next|past)\s+(?:day|week|weekend|month|year|{_WEEKDAY_CORE})|"
    rf"{_COUNTED_AGO_CORE}|{_PAST_WINDOW_CORE}|"
    rf"{_COUNT_CORE}\s+(?:days?|weeks?|months?|years?)\s+later)(?!\w)",
    re.IGNORECASE,
)
_WEEKDAY_HINT = re.compile(rf"(?<!\w)(?:{_WEEKDAY_CORE})(?!\w)", re.IGNORECASE)
_VAGUE_RELATIVE_HINT = re.compile(
    r"(?<!\w)(?:(?:past\s+)?(?:a\s+)?(?:couple|few|several)\s+"
    r"(?:of\s+)?(?:days?|weeks?|months?|years?)(?:\s+ago)?)(?!\w)",
    re.IGNORECASE,
)

_NUMBER_VALUES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_MAX_COUNT = {"day": 1000, "week": 520, "month": 120, "year": 100}
_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def parse_referenced_time(
    query_text: str,
    *,
    timezone: tzinfo,
    now: datetime | None = None,
) -> TemporalQueryParse:
    """Parse one referenced valid-time expression without guessing.

    ``timezone`` is mandatory because calendar periods need concrete aware
    boundaries.  ``now`` is optional for absolute ISO expressions but must be
    supplied explicitly for relative language; the function never reads the
    system clock.  An aware anchor is converted into ``timezone`` before its
    local calendar date is used.
    """

    if not _valid_timezone(timezone):
        return _rejected(TemporalParseReason.INVALID_TIMEZONE)
    if now is not None and not _aware(now):
        return _rejected(TemporalParseReason.INVALID_NOW_ANCHOR)

    candidates = _find_candidates(query_text)
    if len(candidates) > 1:
        return _rejected(TemporalParseReason.MULTIPLE_TEMPORAL_EXPRESSIONS)
    if not candidates:
        if (
            _DATE_SHAPED.search(query_text)
            or _MALFORMED_DATE_HINT.search(query_text)
            or _RELATIVE_HINT.search(query_text)
            or _VAGUE_RELATIVE_HINT.search(query_text)
        ):
            return _rejected(TemporalParseReason.UNSUPPORTED_TEMPORAL_EXPRESSION)
        return _no_match()

    candidate = candidates[0]
    # A malformed or unsupported expression outside the selected clause makes
    # the overall query conflicting.  We do not silently select one date.
    if _has_unconsumed_temporal_hint(query_text, candidate.match.span()):
        return _rejected(TemporalParseReason.MULTIPLE_TEMPORAL_EXPRESSIONS)
    if candidate.kind == "bare" and _ambiguous_bare_year(query_text, candidate.match):
        return _rejected(TemporalParseReason.AMBIGUOUS_BARE_YEAR)

    if candidate.kind in {"between", "from_to"}:
        return _parse_range(candidate, timezone=timezone, now=now)
    if candidate.kind == "since_counted_ago":
        return _parse_since_counted_ago(
            query_text,
            candidate,
            timezone=timezone,
            now=now,
        )
    return _parse_single(candidate, timezone=timezone, now=now)


def _find_candidates(query_text: str) -> list[_Candidate]:
    since_counted_ago = [
        _Candidate("since_counted_ago", match) for match in _SINCE_COUNTED_AGO.finditer(query_text)
    ]
    occupied = [candidate.match.span() for candidate in since_counted_ago]
    ranges = [
        *(
            _Candidate("between", match)
            for match in _BETWEEN.finditer(query_text)
            if not _overlaps_any(match.span(), occupied)
        ),
        *(
            _Candidate("from_to", match)
            for match in _FROM_TO.finditer(query_text)
            if not _overlaps_any(match.span(), occupied)
        ),
    ]
    occupied.extend(candidate.match.span() for candidate in ranges)
    unary = [
        _Candidate("unary", match)
        for match in _UNARY.finditer(query_text)
        if not _overlaps_any(match.span(), occupied)
    ]
    occupied.extend(candidate.match.span() for candidate in unary)
    bare = [
        _Candidate("bare", match)
        for match in _BARE_VALUE.finditer(query_text)
        if not _overlaps_any(match.span(), occupied)
    ]
    return sorted(
        (*since_counted_ago, *ranges, *unary, *bare),
        key=lambda candidate: candidate.match.start(),
    )


def _parse_since_counted_ago(
    query_text: str,
    candidate: _Candidate,
    *,
    timezone: tzinfo,
    now: datetime | None,
) -> TemporalQueryParse:
    """Parse a counted-ago event anchor as a trailing closed window.

    The outer ``since`` candidate owns the nested counted-ago expression so
    it cannot also be selected as a one-day bare period.  Temporal language in
    the optional intervening event clause remains conflicting and fails
    closed rather than being silently swallowed by the outer match.
    """

    if _has_unconsumed_temporal_hint(query_text, candidate.match.span("value")):
        return _rejected(TemporalParseReason.MULTIPLE_TEMPORAL_EXPRESSIONS)
    if _ELAPSED_SINCE_COMPARISON_PREFIX.search(query_text[: candidate.match.start()]):
        return _no_match()

    value = candidate.match.group("value")
    period, failure = _period(value, timezone=timezone, now=now)
    if failure is not None:
        return _rejected(failure)
    assert period is not None and period.relative and not period.window
    assert now is not None

    try:
        anchor_end = now.astimezone(timezone).date() + timedelta(days=1)
        window = _calendar_period(
            period.start.date(),
            anchor_end,
            timezone=timezone,
            relative=True,
            window=True,
        )
    except (OverflowError, ValueError):
        return _rejected(TemporalParseReason.INVALID_DATE)

    return TemporalQueryParse(
        status=TemporalParseStatus.MATCHED,
        confidence=TemporalConfidence.MEDIUM,
        reason=TemporalParseReason.ANCHORED_RELATIVE_WINDOW,
        interval=ReferencedTimeInterval(valid_from=window.start, valid_to=window.end),
        matched_text=candidate.match.group(0),
        relative=True,
    )


def _parse_range(
    candidate: _Candidate,
    *,
    timezone: tzinfo,
    now: datetime | None,
) -> TemporalQueryParse:
    left_text = candidate.match.group("left")
    right_text = candidate.match.group("right")
    left, failure = _period(left_text, timezone=timezone, now=now)
    if failure is not None:
        return _rejected(failure)
    right, failure = _period(right_text, timezone=timezone, now=now)
    if failure is not None:
        return _rejected(failure)
    assert left is not None and right is not None
    if left.window or right.window:
        # A trailing window is already an interval, not a range endpoint.
        # Flattening it into another range would discard its stated boundary.
        return _rejected(TemporalParseReason.UNSUPPORTED_TEMPORAL_EXPRESSION)
    if left.start > right.start:
        return _rejected(TemporalParseReason.REVERSED_RANGE)
    relative = left.relative or right.relative
    return TemporalQueryParse(
        status=TemporalParseStatus.MATCHED,
        confidence=TemporalConfidence.MEDIUM if relative else TemporalConfidence.HIGH,
        reason=(
            TemporalParseReason.ANCHORED_RELATIVE_RANGE
            if relative
            else TemporalParseReason.EXPLICIT_RANGE
        ),
        interval=ReferencedTimeInterval(valid_from=left.start, valid_to=right.end),
        matched_text=candidate.match.group(0),
        relative=relative,
    )


def _parse_single(
    candidate: _Candidate,
    *,
    timezone: tzinfo,
    now: datetime | None,
) -> TemporalQueryParse:
    operator = candidate.match.group("operator").lower() if candidate.kind == "unary" else "period"
    value = candidate.match.group("value")
    period, failure = _period(value, timezone=timezone, now=now)
    if failure is not None:
        return _rejected(failure)
    assert period is not None

    if operator == "before":
        interval = ReferencedTimeInterval(valid_from=None, valid_to=period.start)
        reason = (
            TemporalParseReason.ANCHORED_RELATIVE_BEFORE
            if period.relative
            else TemporalParseReason.EXPLICIT_BEFORE
        )
    elif operator == "after":
        interval = ReferencedTimeInterval(valid_from=period.end, valid_to=None)
        reason = (
            TemporalParseReason.ANCHORED_RELATIVE_AFTER
            if period.relative
            else TemporalParseReason.EXPLICIT_AFTER
        )
    else:
        interval = ReferencedTimeInterval(valid_from=period.start, valid_to=period.end)
        reason = (
            TemporalParseReason.ANCHORED_RELATIVE_WINDOW
            if period.window
            else (
                TemporalParseReason.ANCHORED_RELATIVE_PERIOD
                if period.relative
                else TemporalParseReason.EXPLICIT_PERIOD
            )
        )
    return TemporalQueryParse(
        status=TemporalParseStatus.MATCHED,
        confidence=(TemporalConfidence.MEDIUM if period.relative else TemporalConfidence.HIGH),
        reason=reason,
        interval=interval,
        matched_text=candidate.match.group(0),
        relative=period.relative,
    )


def _period(
    value: str,
    *,
    timezone: tzinfo,
    now: datetime | None,
) -> tuple[_CalendarPeriod | None, TemporalParseReason | None]:
    normalized = " ".join(value.lower().split())
    absolute = _ABSOLUTE.fullmatch(normalized)
    if absolute is not None:
        try:
            return _absolute_period(absolute, timezone), None
        except (OverflowError, ValueError):
            return None, TemporalParseReason.INVALID_DATE

    if now is None:
        return None, TemporalParseReason.RELATIVE_NOW_REQUIRED
    try:
        local_now = now.astimezone(timezone)
        return _relative_period(normalized, local_now.date(), timezone), None
    except _RelativeParseError as exc:
        return None, exc.reason
    except (OverflowError, ValueError):
        return None, TemporalParseReason.INVALID_DATE


def _absolute_period(match: re.Match[str], timezone: tzinfo) -> _CalendarPeriod:
    year = int(match.group("year"))
    month_text = match.group("month")
    day_text = match.group("day")
    if day_text is not None:
        start_date = date(year, int(month_text), int(day_text))  # type: ignore[arg-type]
        end_date = start_date + timedelta(days=1)
    elif month_text is not None:
        start_date = date(year, int(month_text), 1)
        end_date = _shift_month(start_date, 1)
    else:
        start_date = date(year, 1, 1)
        end_date = date(year + 1, 1, 1)
    return _calendar_period(start_date, end_date, timezone=timezone, relative=False)


def _relative_period(value: str, anchor_date: date, timezone: tzinfo) -> _CalendarPeriod:
    counted_ago = _COUNTED_AGO.fullmatch(value)
    past_window = _PAST_WINDOW.fullmatch(value)
    last_weekday = _LAST_WEEKDAY.fullmatch(value)
    weekend = _WEEKEND.fullmatch(value)

    if counted_ago is not None:
        count, unit = _validated_count(counted_ago)
        start_date = _ago_date(anchor_date, count=count, unit=unit)
        end_date = start_date + timedelta(days=1)
    elif past_window is not None:
        count, unit = _validated_count(past_window)
        if unit == "day":
            start_date = anchor_date - timedelta(days=count - 1)
        elif unit == "week":
            start_date = anchor_date - timedelta(days=count * 7 - 1)
        else:
            start_date = _shift_calendar_months(anchor_date, -count)
        end_date = anchor_date + timedelta(days=1)
        return _calendar_period(
            start_date,
            end_date,
            timezone=timezone,
            relative=True,
            window=True,
        )
    elif last_weekday is not None:
        target = _WEEKDAY_INDEX[last_weekday.group("weekday")]
        elapsed = (anchor_date.weekday() - target) % 7
        start_date = anchor_date - timedelta(days=elapsed or 7)
        end_date = start_date + timedelta(days=1)
    elif weekend is not None:
        week_start = anchor_date - timedelta(days=anchor_date.weekday())
        if weekend.group("direction") == "this":
            start_date = week_start + timedelta(days=5)
        else:
            start_date = week_start - timedelta(days=2)
        end_date = start_date + timedelta(days=2)
    elif value == "today":
        start_date = anchor_date
        end_date = anchor_date + timedelta(days=1)
    elif value == "yesterday":
        start_date = anchor_date - timedelta(days=1)
        end_date = anchor_date
    elif value == "tomorrow":
        start_date = anchor_date + timedelta(days=1)
        end_date = start_date + timedelta(days=1)
    else:
        direction, unit = value.split()
        offset = {"last": -1, "this": 0, "next": 1}[direction]
        if unit == "week":
            this_week = anchor_date - timedelta(days=anchor_date.weekday())
            start_date = this_week + timedelta(weeks=offset)
            end_date = start_date + timedelta(weeks=1)
        elif unit == "month":
            this_month = anchor_date.replace(day=1)
            start_date = _shift_month(this_month, offset)
            end_date = _shift_month(start_date, 1)
        else:
            start_date = date(anchor_date.year + offset, 1, 1)
            end_date = date(start_date.year + 1, 1, 1)
    return _calendar_period(start_date, end_date, timezone=timezone, relative=True)


def _validated_count(match: re.Match[str]) -> tuple[int, str]:
    count_text = match.group("count")
    raw_unit = match.group("unit")
    unit = raw_unit.removesuffix("s")
    count = _parse_count(count_text)
    if count < 1 or count > _MAX_COUNT[unit]:
        raise _RelativeParseError(TemporalParseReason.RELATIVE_COUNT_OUT_OF_RANGE)
    singular = raw_unit == unit
    if singular != (count == 1):
        raise _RelativeParseError(TemporalParseReason.INVALID_RELATIVE_COUNT)
    return count, unit


def _parse_count(value: str) -> int:
    normalized = value.lower().replace("-", " ")
    if normalized.isdigit():
        if len(normalized) > 1 and normalized.startswith("0"):
            raise _RelativeParseError(TemporalParseReason.INVALID_RELATIVE_COUNT)
        return int(normalized)
    if normalized in {"a", "an"}:
        return 1

    current = 0
    total = 0
    for token in normalized.split():
        if token == "and":
            continue
        if token in _NUMBER_VALUES:
            current += _NUMBER_VALUES[token]
        elif token == "hundred":
            current *= 100
        elif token == "thousand":
            total += current * 1000
            current = 0
        else:  # pragma: no cover - the regex grammar excludes this branch
            raise _RelativeParseError(TemporalParseReason.INVALID_RELATIVE_COUNT)
    return total + current


def _ago_date(anchor_date: date, *, count: int, unit: str) -> date:
    if unit == "day":
        return anchor_date - timedelta(days=count)
    if unit == "week":
        return anchor_date - timedelta(weeks=count)
    if unit == "month":
        return _shift_calendar_months(anchor_date, -count)
    return _shift_calendar_months(anchor_date, -12 * count)


def _calendar_period(
    start_date: date,
    end_date: date,
    *,
    timezone: tzinfo,
    relative: bool,
    window: bool = False,
) -> _CalendarPeriod:
    # Construct each local midnight separately.  Adding 24 hours to an aware
    # datetime would get calendar-day semantics wrong across DST transitions.
    return _CalendarPeriod(
        start=datetime.combine(start_date, datetime.min.time(), tzinfo=timezone),
        end=datetime.combine(end_date, datetime.min.time(), tzinfo=timezone),
        relative=relative,
        window=window,
    )


def _shift_month(value: date, offset: int) -> date:
    month_index = value.year * 12 + value.month - 1 + offset
    year, zero_based_month = divmod(month_index, 12)
    return date(year, zero_based_month + 1, 1)


def _shift_calendar_months(value: date, offset: int) -> date:
    month_index = value.year * 12 + value.month - 1 + offset
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    return date(year, month, min(value.day, monthrange(year, month)[1]))


def _has_unconsumed_temporal_hint(query_text: str, selected: tuple[int, int]) -> bool:
    for pattern in (
        _DATE_SHAPED,
        _MALFORMED_DATE_HINT,
        _RELATIVE_HINT,
        _VAGUE_RELATIVE_HINT,
        _WEEKDAY_HINT,
    ):
        for match in pattern.finditer(query_text):
            if not _contains(selected, match.span()):
                return True
    return False


def _ambiguous_bare_year(query_text: str, match: re.Match[str]) -> bool:
    value = match.group("value")
    if re.fullmatch(r"\d{4}", value) is None:
        return False
    outside = f"{query_text[: match.start()]}{query_text[match.end() :]}"
    # A lone year is safe; a year embedded in prose could just as plausibly be
    # a ticket, count, model, or release.  Callers can use an explicit cue such
    # as "in 2024" when a temporal reading is intended.
    return re.fullmatch(r"[\s?.!,;:'\"()\[\]{}]*", outside) is None


def _valid_timezone(value: tzinfo) -> bool:
    try:
        return datetime(2000, 1, 1, tzinfo=value).utcoffset() is not None
    except (OverflowError, TypeError, ValueError):
        return False


def _aware(value: datetime) -> bool:
    try:
        return value.tzinfo is not None and value.utcoffset() is not None
    except (OverflowError, ValueError):
        return False


def _overlaps_any(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in occupied)


def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _rejected(reason: TemporalParseReason) -> TemporalQueryParse:
    return TemporalQueryParse(
        status=TemporalParseStatus.REJECTED,
        confidence=TemporalConfidence.NONE,
        reason=reason,
    )


def _no_match() -> TemporalQueryParse:
    return TemporalQueryParse(
        status=TemporalParseStatus.NO_MATCH,
        confidence=TemporalConfidence.NONE,
        reason=TemporalParseReason.NO_TEMPORAL_EXPRESSION,
    )
