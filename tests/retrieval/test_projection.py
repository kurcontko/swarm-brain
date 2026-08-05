from __future__ import annotations

from swarmbrain.retrieval.projection import (
    MAX_LOOKUP_TEXT,
    MAX_METADATA_NODES,
    MAX_SEARCH_TEXT,
    exact_terms,
    lookup_text,
    search_text,
    trigram_similarity,
)


def test_projection_covers_identifier_matrix_and_plural_metadata_keys() -> None:
    content = (
        "Run src/db/retry.py via RetryCoordinator.handle and "
        "test_serialization_retry after commit deadbeef; SQLSTATE 40001."
    )
    metadata = {
        "aliases": ["retryctl", "transaction-retry"],
        "commands": ["pytest tests/test_retry.py -q"],
        "paths": ["src/metadata/fallback.py"],
    }

    projected = exact_terms(
        memory_id="11111111-1111-4111-8111-111111111111",
        content_sha256="a" * 64,
        title="Retry Coordinator",
        content=content,
        tags=("critical",),
        metadata=metadata,
    )
    values = {(term.kind, term.value) for term in projected}

    assert ("memory_id", "11111111-1111-4111-8111-111111111111") in values
    assert ("content_sha256", "a" * 64) in values
    assert ("title", "retry coordinator") in values
    assert ("tag", "critical") in values
    assert ("path", "src/db/retry.py") in values
    assert ("symbol", "retrycoordinator.handle") in values
    assert ("test", "test_serialization_retry") in values
    assert ("commit", "deadbeef") in values
    assert ("error", "sqlstate 40001") in values
    assert ("alias", "retryctl") in values
    assert ("command", "pytest tests/test_retry.py -q") in values

    fuzzy = lookup_text(
        memory_id="11111111-1111-4111-8111-111111111111",
        content_sha256="a" * 64,
        title="Retry Coordinator",
        content=content,
        tags=("critical",),
        metadata=metadata,
    )
    assert "retryctl" in fuzzy
    assert "aliase" not in fuzzy


def test_projection_bounds_output_and_metadata_traversal() -> None:
    metadata = {f"ignored-{index}": {"value": index} for index in range(MAX_METADATA_NODES + 500)}
    metadata["aliases"] = ["must-not-require-unbounded-scan"]

    projected_search = search_text(
        title="bounded",
        content="x" * (MAX_SEARCH_TEXT * 2),
        tags=(),
        metadata=metadata,
    )
    projected_lookup = lookup_text(
        memory_id="11111111-1111-4111-8111-111111111111",
        content_sha256="b" * 64,
        title="bounded",
        content="x" * (MAX_SEARCH_TEXT * 2),
        tags=(),
        metadata=metadata,
    )

    assert len(projected_search) == MAX_SEARCH_TEXT
    assert len(projected_lookup) <= MAX_LOOKUP_TEXT


def test_reference_trigram_similarity_matches_cockroach_word_semantics() -> None:
    assert trigram_similarity("word", "worm") == 3 / 7
    assert trigram_similarity("foo bar", "foo baz") == 0.6
    assert trigram_similarity("RetryCoordinator\nretry", "RetryCoordintor") == 0.7
