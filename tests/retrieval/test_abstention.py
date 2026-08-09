"""Abstention regression over the checked-in retrieval evaluation corpus.

``docs/retrieval-benchmark.md`` measured the defect this pins: every one of the
six no-answer queries returned ten hits with a top-1 public score of exactly
``1.00``, and no floor on that score produced an abstention.  These tests hold
the fix in place from both ends — the no-answer queries must abstain at the
documented floor, and the answerable queries must still find their judged
evidence at the same floor.

The floor asserted here is ``0.25`` because this configuration runs without the
dense lane.  The shipped configuration includes it, and with the stand-in hash
embedder that raises the floor needed for full abstention to ``0.40``; both
curves are in the benchmark addendum.  Numbers here are structural regressions,
not the benchmark.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType

import pytest

from conftest import make_actor
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import RecallQuery, RememberCommand, Visibility

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "retrieval_eval_corpus"
SCRIPT = REPO_ROOT / "scripts" / "run_retrieval_eval.py"

# The floor documented in the benchmark addendum for the lane set used here.
NO_DENSE_ABSTENTION_FLOOR = 0.25


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swarmbrain_eval_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _Clock:
    def __init__(self, start: object) -> None:
        self.value = start

    def __call__(self) -> object:
        return self.value

    def step(self, seconds: int = 1) -> None:
        self.value = self.value + timedelta(seconds=seconds)  # type: ignore[operator]


@pytest.fixture(scope="module")
def corpus() -> object:
    return runner.load_corpus(CORPUS_DIR)


@pytest.fixture(scope="module")
async def published(
    corpus: object,
) -> tuple[RetrievalService, object, dict[str, str]]:
    """Publish the evaluation corpus once, with exact/lexical/fuzzy/graph lanes."""

    clock = _Clock(corpus.clock)  # type: ignore[attr-defined]
    kernel = InMemoryKernel(clock=clock)
    actor = make_actor(
        {
            name: value
            for name, value in zip(
                ("tenant_id", "project_id", "repository_id", "swarm_id", "run_id"),
                _scope_values(),
                strict=True,
            )
        }
    )
    service = MemoryService(kernel, ConservativeMemoryPolicy(), review_store=kernel)
    by_key: dict[str, str] = {}
    for memory in corpus.memories:  # type: ignore[attr-defined]
        clock.step()
        result = await service.publish(
            actor,
            RememberCommand(
                idempotency_key=f"abstention:{memory.key}",
                kind=memory.kind,
                desired_state=memory.state,
                visibility=Visibility.REPOSITORY,
                title=memory.title,
                content=memory.content,
                tags=memory.tags,
                metadata=memory.metadata,
                related_memory_ids=tuple(by_key[key] for key in memory.related),
            ),
        )
        assert result.memory is not None
        by_key[memory.key] = result.memory.memory_id
    retrieval = RetrievalService(in_memory_retrieval_gateways(kernel), kernel)
    return retrieval, actor, by_key


def _scope_values() -> tuple[str, ...]:
    from conftest import new_id

    return tuple(new_id() for _ in range(5))


def _cases(corpus: object, category: str) -> tuple[object, ...]:
    return tuple(case for case in corpus.queries if case.category == category)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_no_answer_queries_still_return_hits_at_the_default_floor(
    corpus: object,
    published: tuple[RetrievalService, object, dict[str, str]],
) -> None:
    """min_score = 0.0 must behave exactly as it did: rank, do not judge."""

    retrieval, actor, _ = published
    no_answer = _cases(corpus, "no_answer")
    assert len(no_answer) == 6

    for case in no_answer:
        execution = await retrieval.execute(
            actor,
            RecallQuery(text=case.text, limit=10),  # type: ignore[attr-defined]
        )
        assert execution.bundle.hits, case.case_id  # type: ignore[attr-defined]
        assert execution.trace.abstained is False
        # The defect this feature answers: the top hit's public score overstates
        # the evidence behind it, because it is dominated by rank position while
        # nothing in the field is actually relevant.  Asserting the relationship
        # rather than a fixed number keeps this honest across fusion changes:
        # relevance reranking moved the absolute value without touching the
        # defect, and a magic threshold would have hidden that.
        top = execution.trace.candidate_relevance[0]
        assert execution.bundle.hits[0].score > top.relevance, case.case_id  # type: ignore[attr-defined]
        assert top.relevance < NO_DENSE_ABSTENTION_FLOOR, case.case_id  # type: ignore[attr-defined]
        assert (
            max(item.relevance for item in execution.trace.candidate_relevance)
            < NO_DENSE_ABSTENTION_FLOOR
        ), case.case_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_public_score_cannot_separate_no_answer_from_answerable(
    corpus: object,
    published: tuple[RetrievalService, object, dict[str, str]],
) -> None:
    """The load-bearing reason ``min_score`` gates on relevance and not on score.

    Relevance reranking blends calibrated relevance into the published score, so
    it would be reasonable to ask whether a floor on that score could now
    abstain.  It cannot: the two groups' top-1 scores still overlap, which means
    no threshold separates them.  If this ever starts passing in the other
    direction the abstention design is due for a rewrite, and this test is the
    place that should say so.
    """

    retrieval, actor, _ = published

    async def top_scores(category: str) -> list[float]:
        scores: list[float] = []
        for case in _cases(corpus, category):
            execution = await retrieval.execute(
                actor,
                RecallQuery(text=case.text, limit=10),  # type: ignore[attr-defined]
            )
            assert execution.bundle.hits, case.case_id  # type: ignore[attr-defined]
            scores.append(execution.bundle.hits[0].score)
        return scores

    no_answer = await top_scores("no_answer")
    answerable: list[float] = []
    for category in ("paraphrase", "multi_evidence", "decoy_heavy"):
        answerable.extend(await top_scores(category))

    assert no_answer and answerable
    assert max(no_answer) >= min(answerable), (
        "the public score now separates no-answer from answerable queries; "
        "revisit the relevance-gated abstention design"
    )


@pytest.mark.asyncio
async def test_no_answer_queries_abstain_at_the_documented_relevance_floor(
    corpus: object,
    published: tuple[RetrievalService, object, dict[str, str]],
) -> None:
    retrieval, actor, _ = published

    for case in _cases(corpus, "no_answer"):
        execution = await retrieval.execute(
            actor,
            RecallQuery(
                text=case.text,  # type: ignore[attr-defined]
                limit=10,
                min_score=NO_DENSE_ABSTENTION_FLOOR,
            ),
        )
        assert execution.bundle.hits == (), case.case_id  # type: ignore[attr-defined]
        assert execution.trace.abstained is True
        assert execution.trace.abstention_reason == "below_relevance_floor"
        # Candidates existed and were hydrated; they were judged, not missing.
        assert execution.trace.candidate_relevance, case.case_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_answerable_queries_keep_their_judged_evidence_at_that_floor(
    corpus: object,
    published: tuple[RetrievalService, object, dict[str, str]],
) -> None:
    """The floor must not be paid for with the categories it is meant to keep."""

    retrieval, actor, by_key = published
    categories = ("identifier_exact", "code_lookup", "lexical", "decoy_heavy")
    checked = 0

    for category in categories:
        for case in _cases(corpus, category):
            expected = {by_key[key] for key in case.relevant}  # type: ignore[attr-defined]
            assert expected
            execution = await retrieval.execute(
                actor,
                RecallQuery(
                    text=case.text,  # type: ignore[attr-defined]
                    limit=10,
                    min_score=NO_DENSE_ABSTENTION_FLOOR,
                ),
            )
            returned = {hit.memory.memory_id for hit in execution.bundle.hits}
            assert returned & expected, case.case_id  # type: ignore[attr-defined]
            checked += 1

    assert checked == 22
