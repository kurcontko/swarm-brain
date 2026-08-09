"""Token-budgeted packing of a recall bundle, and answer-in-context scoring.

Why this exists
---------------
Recall@k asks whether the evidence was found.  A reader does not consume a
ranking, it consumes a *context window*, and everything past the budget is not
merely lower-ranked — it is absent.  On the swarm corpus the distinction barely
registers, because memories are short.  On LongMemEval-S a single hit is a whole
conversation session, so a ten-hit bundle is tens of thousands of tokens and the
budget binds hard.  ``answer_in_context`` is the quantity that survives that
truncation: did at least one memory carrying the answer make it into the packed
bundle at all.

The packing policy
------------------
Hits are taken in bundle order, which is the ranking the retriever published.
An item that does not fit is **skipped rather than terminating the pack**, and
smaller lower-ranked items may still be admitted.  That is a deliberate choice
between two defensible policies:

``prefix``
    Stop at the first item that does not fit.  Preserves rank contiguity and is
    what a naive renderer does.
``greedy`` (used here)
    Skip the oversized item and keep filling.  Puts strictly more evidence in
    front of the reader for the same budget, which is the point of having a
    budget at all.

Both are reported by :func:`pack_to_budget` through ``policy`` so the choice is
measured rather than assumed.

Token estimation
----------------
The repository deliberately carries no tokenizer dependency, so
:func:`estimate_tokens` is a documented proxy: characters divided by four,
rounded up, floored at one for any non-empty text.  That ratio is the usual
approximation for English prose under byte-pair encodings, and it is *stable*,
which is what a comparative metric needs.  It is not exact, and it should not be
quoted as a cost figure.  Two consequences are worth stating plainly rather than
discovering later:

- code, JSON and identifier-dense text tokenise worse than four characters per
  token, so budgets computed here are optimistic for such content;
- swapping in a real tokenizer changes the absolute numbers and must not be done
  silently, because every published answer-in-context figure is tied to this
  estimator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text``; see the module docstring's caveats."""

    if not text:
        return 0
    return max(1, ceil(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True, slots=True)
class PackedBundle:
    """The prefix of a bundle that fits a token budget."""

    kept_indices: tuple[int, ...]
    used_tokens: int
    dropped_indices: tuple[int, ...]

    @property
    def kept(self) -> int:
        return len(self.kept_indices)


def pack_to_budget(
    sizes: Sequence[int],
    budget: int | None,
    *,
    policy: str = "greedy",
) -> PackedBundle:
    """Select the hits that fit ``budget``, in bundle order.

    ``budget`` of ``None`` means unbounded and keeps everything, which is the
    shipped default and makes this a no-op unless a caller opts in.
    """

    if policy not in {"greedy", "prefix"}:
        raise ValueError("policy must be 'greedy' or 'prefix'")
    if budget is not None and budget < 0:
        raise ValueError("budget must not be negative")

    kept: list[int] = []
    dropped: list[int] = []
    used = 0
    for index, size in enumerate(sizes):
        if budget is None or used + size <= budget:
            kept.append(index)
            used += size
            continue
        dropped.append(index)
        if policy == "prefix":
            dropped.extend(range(index + 1, len(sizes)))
            break
    return PackedBundle(
        kept_indices=tuple(kept),
        used_tokens=used,
        dropped_indices=tuple(dropped),
    )


@dataclass(frozen=True, slots=True)
class AnswerInContextMetrics:
    """Whether the answer survives packing, not merely whether it was ranked.

    ``any_gold`` is the headline: the fraction of answerable cases where at
    least one memory carrying the answer reached the packed bundle.  ``all_gold``
    is the stricter multi-evidence form — a question needing three sessions is
    not answerable from one of them.
    """

    budget: int | None
    policy: str
    cases: int
    answerable_cases: int
    any_gold: float
    all_gold: float
    mean_kept: float
    mean_tokens: float
    truncated_cases: int


def answer_in_context(
    cases: Iterable[tuple[frozenset[str], Sequence[str], Sequence[int]]],
    *,
    budget: int | None,
    policy: str = "greedy",
) -> AnswerInContextMetrics:
    """Score packed bundles.

    Each case supplies its gold ids, the returned ids in bundle order, and the
    per-hit token sizes in the same order.  Cases whose sizes are missing are
    treated as unbounded for that case, so a run recorded before token sizes
    existed degrades to "nothing was truncated" rather than to a silent zero.
    """

    any_hits: list[float] = []
    all_hits: list[float] = []
    kept_counts: list[int] = []
    token_counts: list[int] = []
    total = 0
    answerable = 0
    truncated = 0

    for gold, returned, sizes in cases:
        total += 1
        packed = pack_to_budget(
            list(sizes)[: len(returned)] if sizes else [0] * len(returned),
            budget,
            policy=policy,
        )
        kept_ids = [returned[index] for index in packed.kept_indices if index < len(returned)]
        kept_counts.append(len(kept_ids))
        token_counts.append(packed.used_tokens)
        if packed.dropped_indices:
            truncated += 1
        if not gold:
            continue
        answerable += 1
        present = sum(1 for candidate_id in kept_ids if candidate_id in gold)
        any_hits.append(1.0 if present else 0.0)
        all_hits.append(1.0 if present >= len(gold) else 0.0)

    return AnswerInContextMetrics(
        budget=budget,
        policy=policy,
        cases=total,
        answerable_cases=answerable,
        any_gold=_mean(any_hits),
        all_gold=_mean(all_hits),
        mean_kept=_mean([float(value) for value in kept_counts]),
        mean_tokens=_mean([float(value) for value in token_counts]),
        truncated_cases=truncated,
    )


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


__all__ = [
    "CHARS_PER_TOKEN",
    "AnswerInContextMetrics",
    "PackedBundle",
    "answer_in_context",
    "estimate_tokens",
    "pack_to_budget",
]
