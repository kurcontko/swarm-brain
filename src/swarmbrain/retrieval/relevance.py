"""Rank-independent, adapter-identical candidate relevance.

Why this exists
---------------
The public ``RecallHit.score`` is a *normalised weighted-RRF* score: it is
anchored to the best possible rank in the strongest configured lane, so a
candidate that lands at rank one of the lexical lane scores ``1.00`` no matter
how weak its actual overlap with the query is.  That score therefore carries
rank information and no relevance information, and no threshold on it can tell
"the best of a bad field" apart from "a good answer".  The measured consequence
is recorded in ``docs/retrieval-benchmark.md``: every no-answer query in the
swarm-native corpus returned ten hits with a top-1 score of exactly ``1.00``,
and sweeping the caller floor from 0.0 to 0.8 never produced an abstention.

This module computes the missing quantity: a **per-candidate relevance in
[0, 1] that does not depend on rank position**.  ``RecallQuery.min_score`` is
gated on it by :class:`~swarmbrain.application.retrieval_service.RetrievalService`.

Design
------
1. *Per-lane evidence, each already a similarity in [0, 1].*

   ``exact``
       ``1.0``.  The exact lane fires only when the whole normalised query
       equals a projected term (title, tag, path, symbol, test name, commit,
       error code, content hash) or when the candidate is a caller-supplied
       memory id or planner seed.  Both are unambiguous relevance.
   ``lexical``
       Query-token coverage: the fraction of the query's distinct content
       tokens that appear in the memory's ``search_text`` projection.
   ``fuzzy``
       ``trigram_similarity`` between the memory's ``lookup_text`` projection
       and the normalised query.
   ``dense``
       The cosine carried on the dense fusion contribution, clamped to [0, 1].
   ``graph``
       ``0.0``.  Graph activation is a function of a *seed's* fused rank and of
       edge weights, not of the query, so it is rank-derived by construction and
       carries no independent relevance.  A candidate reached only by graph
       expansion therefore has to be defended by another lane.

2. *Combination is ``max``, deliberately.*  A candidate is kept when **some**
   lane can defend it on its own evidence.  Summing or averaging would let
   several individually-weak lanes manufacture a passing score, which is the
   exact failure mode being fixed here.

3. *Adapter parity is structural, not tested-into-existence.*  The lexical and
   trigram components are **recomputed from the canonical memory text** using
   the same ``swarmbrain.retrieval.projection`` helpers both adapters index
   with, rather than read off ``FusionContribution.raw_score``.  That is
   deliberate: the two adapters' lexical raw scores are on different scales
   (the in-memory kernel reports token overlap, CockroachDB reports
   ``ts_rank``), and pg_trgm's ``similarity`` agrees with the portable
   reference implementation closely but not exactly.  Recomputation makes the
   floor mean the same thing on both backends, which a raw-score calibration
   could not.  The dense cosine is the one component that cannot be recomputed
   without the query vector; both adapters produce cosine in the same
   projection space, so it is read from the contribution.

4. *Cheap, and no extra queries.*  Everything comes from the canonical memory
   the hydration step already loaded plus the per-lane raw scores the trace
   already carries.  The service evaluates candidates lazily in fused order and
   stops once the public ``limit`` is satisfied, so a default
   ``min_score = 0.0`` caller pays for at most ``limit + 1`` candidates.

Known limits, stated rather than hidden
---------------------------------------
- ``_FUNCTION_WORDS`` is a small closed list of English function words.  On a
  non-English query nothing is filtered and the lexical component degrades to
  plain token coverage, which is *more* permissive, never less: the failure
  direction is "returns hits that a floor might have dropped", not "drops a
  relevant hit".
- The lexical component is unweighted coverage.  It has no corpus statistics
  (no IDF), because a per-hit quantity must not depend on which other
  candidates happen to be in the batch — that would make the same memory score
  differently on two backends whose lanes returned different candidate sets.
- With a non-semantic embedding provider the dense component is a
  length-normalised lexical overlap and raises the floor a deployment needs.
  See the benchmark addendum for the measured curve with and without it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from swarmbrain.application.memory_policy import memory_text_sha256
from swarmbrain.domain.memory import Memory
from swarmbrain.domain.retrieval import (
    CandidateRelevance,
    FusionContribution,
    RetrievalSignal,
)

from .projection import (
    MAX_QUERY_CHARS,
    MAX_QUERY_TOKENS,
    lookup_text,
    normalize_term,
    search_text,
    trigram_similarity,
)

RELEVANCE_VERSION = "lane-max-v1"

_TOKENS = re.compile(r"\w+")

# Closed list of English function words.  Removing them keeps a query like
# "how are translations sourced for the czech and polish locales" from scoring
# coverage against every memory that merely contains "for" and "the".  It is a
# heuristic, and it only ever makes the signal more selective; see the module
# docstring for the failure direction on other languages.
_FUNCTION_WORD_TEXT = (
    "a about after all also an and any are as at be been before being but by can "
    "could did do does for from had has have how i if in into is it its may me "
    "might must my no not of on one or our out over should so some such than that "
    "the their them then there these they this those to up us was we were what "
    "when where which while who why will with would you your"
)
_FUNCTION_WORDS = frozenset(_FUNCTION_WORD_TEXT.split())


@dataclass(frozen=True, slots=True)
class RelevanceQuery:
    """Query-side terms, computed once per recall and reused per candidate."""

    normalized: str
    tokens: frozenset[str]


def relevance_query(text: str) -> RelevanceQuery:
    """Project a public query into the terms the relevance signal scores with.

    Tokenisation mirrors the lexical lane exactly — normalise, take the first
    ``MAX_QUERY_TOKENS`` word tokens, deduplicate — and then drops function
    words.  If a query is *only* function words the unfiltered set is kept, so
    such a query still scores coverage rather than collapsing to zero.
    """

    normalized = normalize_term(text[:MAX_QUERY_CHARS])
    raw = _TOKENS.findall(normalized)[:MAX_QUERY_TOKENS]
    if not raw:
        return RelevanceQuery(normalized=normalized, tokens=frozenset())
    content = frozenset(token for token in raw if token not in _FUNCTION_WORDS)
    return RelevanceQuery(normalized=normalized, tokens=content or frozenset(raw))


def lexical_coverage(query: RelevanceQuery, memory: Memory) -> float:
    """Fraction of the query's distinct content tokens present in the memory."""

    if not query.tokens:
        return 0.0
    haystack = normalize_term(
        search_text(
            title=memory.title,
            content=memory.content,
            tags=memory.tags,
            metadata=memory.metadata,
        )
    )
    present = set(_TOKENS.findall(haystack))
    return len(query.tokens & present) / len(query.tokens)


def lookup_similarity(query: RelevanceQuery, memory: Memory) -> float:
    """Trigram similarity against the identifier-oriented lookup projection."""

    if len(query.normalized) < 3:
        return 0.0
    return trigram_similarity(
        lookup_text(
            memory_id=memory.memory_id,
            content_sha256=memory_text_sha256(memory.content),
            title=memory.title,
            content=memory.content,
            tags=memory.tags,
            metadata=memory.metadata,
        ),
        query.normalized,
    )


def candidate_relevance(
    query: RelevanceQuery,
    memory: Memory,
    contributions: Iterable[FusionContribution],
) -> CandidateRelevance:
    """Score one fused candidate's rank-independent relevance in [0, 1].

    ``contributions`` supplies the lanes that produced the candidate and their
    raw per-lane scores; nothing else about the fused ranking is consulted, so
    the result is invariant to rank, to lane weights, and to which other
    candidates are in the batch.
    """

    exact = 0.0
    dense = 0.0
    for contribution in contributions:
        if contribution.lane is RetrievalSignal.EXACT:
            exact = 1.0
        elif contribution.lane is RetrievalSignal.DENSE and contribution.raw_score is not None:
            dense = max(dense, min(1.0, max(0.0, float(contribution.raw_score))))

    # Every component is computed even when the exact lane already pins the
    # result at 1.0: the trace is a diagnostic artifact and a zero there would
    # read as "no lexical evidence" rather than "not consulted".
    coverage = lexical_coverage(query, memory)
    trigram = lookup_similarity(query, memory)
    return CandidateRelevance(
        canonical_id=memory.memory_id,
        relevance=max(exact, coverage, trigram, dense),
        exact_term=exact,
        lexical_coverage=coverage,
        trigram_similarity=trigram,
        dense_cosine=dense,
    )


__all__ = [
    "RELEVANCE_VERSION",
    "RelevanceQuery",
    "candidate_relevance",
    "lexical_coverage",
    "lookup_similarity",
    "relevance_query",
]
