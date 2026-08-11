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

The packing policies
--------------------
Hits are returned in the bundle order published by the retriever.  Two legacy
policies preserve their original behaviour:

``prefix``
    Stop at the first item that does not fit.  Preserves rank contiguity and is
    what a naive renderer does.
``greedy`` (the shipped default)
    Skip the oversized item and keep filling.  Puts strictly more evidence in
    front of the reader for the same budget, which is the point of having a
    budget at all.
``facility_location`` (experimental, explicit opt-in)
    Select a set with a monotone submodular objective combining retrieval
    relevance, distinct query-term coverage, facility-location
    representativeness, and diversity-facet coverage.  A cost-benefit greedy
    pass is compared with a value-greedy pass and the best set wins.  This is a
    deterministic, bounded variant of the standard greedy approximation for a
    submodular knapsack; selected hits are then restored to retrieval order.

The objective follows the facility-location and feature-coverage family used
for document summarization by Lin and Bilmes (ACL 2011, P11-1052).  The two
bounded greedy traversals here are an engineering heuristic, not a claim to the
stronger approximation guarantee of exhaustive partial enumeration.

All are reported by :func:`pack_to_budget` through ``policy`` so the choice is
measured rather than assumed.  The facility-location policy is deliberately
not the default: an offline, paired evaluation has to justify that change.

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

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from heapq import heappop, heappush
from math import ceil, isfinite

from swarmbrain.domain.memory import RecallHit
from swarmbrain.domain.retrieval import PackingPolicy

CHARS_PER_TOKEN = 4
MAX_FACILITY_CANDIDATES = 128
MAX_PACKING_TEXT_CHARS = 32_768
MAX_QUERY_TERMS = 64
MAX_REPRESENTATION_TERMS = 96
MAX_DIVERSITY_LABELS = 16

_PACKING_TOKEN = re.compile(r"\w+")
_PACKING_STOP_WORD_TEXT = (
    "a about after all also an and any are as at be been before being but by can "
    "could did do does for from had has have how if in into is it its may might "
    "must no not of on one or our out over should so some such than that the "
    "their them then there these they this those to up was we were what when "
    "where which while who why will with would you your"
)
_PACKING_STOP_WORDS = frozenset(_PACKING_STOP_WORD_TEXT.split())


@dataclass(frozen=True, slots=True)
class PackingFeatures:
    """Bounded semantic evidence consumed by ``facility_location`` packing.

    ``query_terms`` contains only query terms present in this candidate.
    ``representation_terms`` is a stable bottom-hash sample of the candidate's
    content vocabulary, and ``diversity_labels`` contains coarse facets such as
    memory kind, tag, or author.  The selector never receives raw memory text.
    """

    relevance: float
    query_terms: frozenset[str] = frozenset()
    representation_terms: frozenset[str] = frozenset()
    diversity_labels: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isfinite(self.relevance) or not 0.0 <= self.relevance <= 1.0:
            raise ValueError("packing relevance must be finite and between 0 and 1")
        if len(self.query_terms) > MAX_QUERY_TERMS:
            raise ValueError(f"query_terms must contain at most {MAX_QUERY_TERMS} items")
        if len(self.representation_terms) > MAX_REPRESENTATION_TERMS:
            raise ValueError(
                f"representation_terms must contain at most {MAX_REPRESENTATION_TERMS} items"
            )
        if len(self.diversity_labels) > MAX_DIVERSITY_LABELS:
            raise ValueError(f"diversity_labels must contain at most {MAX_DIVERSITY_LABELS} items")


def build_packing_features(
    text: str,
    *,
    query_terms: Iterable[str],
    relevance: float,
    diversity_labels: Iterable[str] = (),
) -> PackingFeatures:
    """Project text into bounded, deterministic facility-location features.

    A bottom-hash sample avoids a first-paragraph bias while bounding the
    pairwise facility matrix.  Query-term membership is collected during the
    same scan and is bounded by the retrieval query's own 64-term limit.
    """

    normalized_query = frozenset(
        sorted(str(term).casefold() for term in query_terms if str(term))[:MAX_QUERY_TERMS]
    )
    present_query: set[str] = set()
    # Max-heap encoded as negative stable hashes.  Only sampled terms are held,
    # so memory remains O(MAX_REPRESENTATION_TERMS) even for a long session.
    sample_heap: list[tuple[int, str]] = []
    sampled: set[str] = set()
    for match in _PACKING_TOKEN.finditer(text[:MAX_PACKING_TEXT_CHARS].casefold()):
        term = match.group(0)
        if term in normalized_query:
            present_query.add(term)
        if len(term) < 3 or term in _PACKING_STOP_WORDS or term in sampled:
            continue
        digest = int.from_bytes(
            blake2b(term.encode("utf-8"), digest_size=8, person=b"sb-pack-v1").digest(),
            "big",
        )
        entry = (-digest, term)
        if len(sample_heap) < MAX_REPRESENTATION_TERMS:
            heappush(sample_heap, entry)
            sampled.add(term)
            continue
        largest_digest = -sample_heap[0][0]
        largest_term = sample_heap[0][1]
        if (digest, term) >= (largest_digest, largest_term):
            continue
        _removed_digest, removed_term = heappop(sample_heap)
        sampled.remove(removed_term)
        heappush(sample_heap, entry)
        sampled.add(term)

    labels = tuple(
        sorted(
            dict.fromkeys(
                str(label).strip().casefold() for label in diversity_labels if str(label).strip()
            )
        )
    )[:MAX_DIVERSITY_LABELS]
    return PackingFeatures(
        relevance=float(relevance),
        query_terms=frozenset(present_query),
        representation_terms=frozenset(sampled),
        diversity_labels=frozenset(labels),
    )


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text``; see the module docstring's caveats."""

    if not text:
        return 0
    return max(1, ceil(len(text) / CHARS_PER_TOKEN))


def render_recall_hit(hit: RecallHit) -> str:
    """Render one self-describing, source-preserving memory context block.

    The public recall response stays structured.  This compact representation
    is for activation bundles handed to an agent and intentionally includes the
    memory ID so the agent can cite what it used in a checkpoint or completion.
    """

    memory = hit.memory
    content = json.dumps(
        memory.content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    lines = [
        f"[memory:{memory.memory_id}]",
        (
            f"kind={memory.kind} state={memory.state.value} "
            f"visibility={memory.visibility.value} confidence={memory.confidence:.4f}"
        ),
        f"valid_from={memory.valid_from.isoformat()}",
        f"recorded_from={memory.recorded_from.isoformat()}",
    ]
    if memory.occurred_at is not None:
        lines.append(f"occurred_at={memory.occurred_at.isoformat()}")
    if memory.valid_to is not None:
        lines.append(f"valid_to={memory.valid_to.isoformat()}")
    if memory.recorded_to is not None:
        lines.append(f"recorded_to={memory.recorded_to.isoformat()}")
    if memory.author_agent_id:
        lines.append(f"author_agent_id={memory.author_agent_id}")
    if memory.task_id:
        lines.append(f"task_id={memory.task_id}")
    if memory.title:
        lines.append(f"title={json.dumps(memory.title, ensure_ascii=False)}")
    if memory.tags:
        lines.append(
            "tags="
            + json.dumps(
                memory.tags,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if memory.metadata:
        lines.append(
            "metadata="
            + json.dumps(
                memory.metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    lines.append(f"content={content}")
    for evidence in hit.evidence:
        lines.append(
            "evidence="
            + json.dumps(
                evidence.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines)


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
    policy: PackingPolicy | str = PackingPolicy.GREEDY,
    features: Sequence[PackingFeatures] | None = None,
    max_items: int | None = None,
) -> PackedBundle:
    """Select the hits that fit ``budget``, in bundle order.

    ``budget`` of ``None`` means unbounded and keeps everything, which is the
    shipped default and makes this a no-op unless a caller opts in.
    """

    try:
        selected_policy = PackingPolicy(policy)
    except ValueError as exc:
        raise ValueError("policy must be 'greedy', 'prefix', or 'facility_location'") from exc
    if budget is not None and budget < 0:
        raise ValueError("budget must not be negative")
    if any(size < 0 for size in sizes):
        raise ValueError("item sizes must not be negative")
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must not be negative")
    if selected_policy is PackingPolicy.FACILITY_LOCATION:
        if features is None or len(features) != len(sizes):
            raise ValueError("facility_location policy requires one PackingFeatures value per item")
        if budget is not None:
            return _pack_facility_location(
                sizes,
                budget,
                features,
                max_items=max_items,
            )

    kept: list[int] = []
    dropped: list[int] = []
    used = 0
    for index, size in enumerate(sizes):
        if max_items is not None and len(kept) >= max_items:
            dropped.extend(range(index, len(sizes)))
            break
        if budget is None or used + size <= budget:
            kept.append(index)
            used += size
            continue
        dropped.append(index)
        if selected_policy is PackingPolicy.PREFIX:
            dropped.extend(range(index + 1, len(sizes)))
            break
    return PackedBundle(
        kept_indices=tuple(kept),
        used_tokens=used,
        dropped_indices=tuple(dropped),
    )


_RELEVANCE_WEIGHT = 0.35
_QUERY_COVERAGE_WEIGHT = 0.30
_REPRESENTATIVENESS_WEIGHT = 0.25
_DIVERSITY_WEIGHT = 0.10


def _jaccard(left: frozenset[str], right: frozenset[str], *, same: bool) -> float:
    if not left or not right:
        return 1.0 if same else 0.0
    return len(left & right) / len(left | right)


def _pack_facility_location(
    sizes: Sequence[int],
    budget: int,
    features: Sequence[PackingFeatures],
    *,
    max_items: int | None,
) -> PackedBundle:
    """Bounded two-pass greedy maximization of a submodular knapsack."""

    original_count = len(sizes)
    count = min(original_count, MAX_FACILITY_CANDIDATES)
    if count == 0 or max_items == 0:
        return PackedBundle((), 0, tuple(range(len(sizes))))
    sizes = sizes[:count]
    features = features[:count]
    item_limit = count if max_items is None else min(count, max_items)
    query_universe = frozenset().union(*(item.query_terms for item in features))
    diversity_universe = frozenset().union(*(item.diversity_labels for item in features))
    relevance_denominator = max(1, item_limit)
    similarities = tuple(
        tuple(
            _jaccard(
                candidate.representation_terms,
                represented.representation_terms,
                same=candidate_index == represented_index,
            )
            for represented_index, represented in enumerate(features)
        )
        for candidate_index, candidate in enumerate(features)
    )

    def choose(*, density: bool) -> tuple[int, ...]:
        selected: list[int] = []
        selected_set: set[int] = set()
        covered_query: set[str] = set()
        covered_diversity: set[str] = set()
        represented = [0.0] * count
        used = 0
        while len(selected) < item_limit:
            best: tuple[float, float, float, int, int] | None = None
            best_index: int | None = None
            for index, item in enumerate(features):
                if index in selected_set or used + sizes[index] > budget:
                    continue
                query_gain = (
                    len(item.query_terms - covered_query) / len(query_universe)
                    if query_universe
                    else 0.0
                )
                diversity_gain = (
                    len(item.diversity_labels - covered_diversity) / len(diversity_universe)
                    if diversity_universe
                    else 0.0
                )
                facility_gain = (
                    sum(
                        max(0.0, similarities[index][target] - represented[target])
                        for target in range(count)
                    )
                    / count
                )
                gain = (
                    _RELEVANCE_WEIGHT * (item.relevance / relevance_denominator)
                    + _QUERY_COVERAGE_WEIGHT * query_gain
                    + _REPRESENTATIVENESS_WEIGHT * facility_gain
                    + _DIVERSITY_WEIGHT * diversity_gain
                )
                if gain <= 0.0:
                    continue
                cost = max(1, sizes[index])
                priority = gain / cost if density else gain
                # Relevance, smaller cost, then earlier retrieval rank break
                # exact objective ties without relying on set iteration order.
                key = (priority, gain, item.relevance, -cost, -index)
                if best is None or key > best:
                    best = key
                    best_index = index
            if best_index is None:
                break
            selected.append(best_index)
            selected_set.add(best_index)
            used += sizes[best_index]
            covered_query.update(features[best_index].query_terms)
            covered_diversity.update(features[best_index].diversity_labels)
            represented = [
                max(value, similarities[best_index][target])
                for target, value in enumerate(represented)
            ]
        return tuple(selected)

    def objective(selected: tuple[int, ...]) -> float:
        if not selected:
            return 0.0
        query = frozenset().union(*(features[index].query_terms for index in selected))
        diversity = frozenset().union(*(features[index].diversity_labels for index in selected))
        relevance = sum(features[index].relevance for index in selected) / relevance_denominator
        facility = (
            sum(max(similarities[index][target] for index in selected) for target in range(count))
            / count
        )
        return (
            _RELEVANCE_WEIGHT * relevance
            + _QUERY_COVERAGE_WEIGHT * (len(query) / len(query_universe) if query_universe else 0.0)
            + _REPRESENTATIVENESS_WEIGHT * facility
            + _DIVERSITY_WEIGHT
            * (len(diversity) / len(diversity_universe) if diversity_universe else 0.0)
        )

    density_choice = choose(density=True)
    value_choice = choose(density=False)
    choices = (density_choice, value_choice)
    selected = max(
        choices,
        key=lambda choice: (
            objective(choice),
            -sum(sizes[index] for index in choice),
            tuple(-index for index in sorted(choice)),
        ),
    )
    kept = tuple(sorted(selected))
    kept_set = frozenset(kept)
    dropped = tuple(index for index in range(count) if index not in kept_set) + tuple(
        range(count, original_count)
    )
    # ``sizes`` is sliced above, so append the original, unevaluated tail as
    # dropped.  This is the hard bound on pairwise facility computation.
    return PackedBundle(
        kept_indices=kept,
        used_tokens=sum(sizes[index] for index in kept),
        dropped_indices=dropped,
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
    "MAX_DIVERSITY_LABELS",
    "MAX_FACILITY_CANDIDATES",
    "MAX_PACKING_TEXT_CHARS",
    "MAX_QUERY_TERMS",
    "MAX_REPRESENTATION_TERMS",
    "AnswerInContextMetrics",
    "PackedBundle",
    "PackingFeatures",
    "answer_in_context",
    "build_packing_features",
    "estimate_tokens",
    "pack_to_budget",
    "render_recall_hit",
]
