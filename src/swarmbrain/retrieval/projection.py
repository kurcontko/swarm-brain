"""Deterministic text projections shared by memory and Cockroach adapters."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from swarmbrain.application.memory_policy import memory_content_text
from swarmbrain.domain.common import JsonObject, MemoryContent
from swarmbrain.domain.memory import MemoryKind, MemoryKindValue, Visibility

RETRIEVAL_PROJECTION_ID = "memory-simple-v1"
MAX_SEARCH_TEXT = 262_144
MAX_LOOKUP_TEXT = 8_192
MAX_EXACT_TERM = 4_096
MAX_EXACT_TERMS = 512
MAX_QUERY_CHARS = 8_192
MAX_QUERY_TOKENS = 64
MAX_METADATA_NODES = 2_048
MAX_METADATA_TERMS = 128
FUZZY_SIMILARITY_THRESHOLD = 0.25

_SPACE = re.compile(r"\s+")
_TRIGRAM_WORD = re.compile(r"\w+", flags=re.UNICODE)
_LOOKUP_PATTERNS = (
    ("path", re.compile(r"(?<![\w./-])(?:[\w.@+-]+/)+[\w.@+-]+")),
    ("symbol", re.compile(r"\b[A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)+\b")),
    ("test", re.compile(r"\btest_[A-Za-z0-9_]+\b")),
    ("commit", re.compile(r"\b[0-9a-fA-F]{7,64}\b")),
    ("error", re.compile(r"\b(?:[A-Z][A-Z0-9_]+-\d+|SQLSTATE\s+[0-9A-Z]{5})\b")),
)
_METADATA_LOOKUP_KEYS = frozenset(
    {
        "alias",
        "aliases",
        "command",
        "commands",
        "commit",
        "commits",
        "error",
        "errors",
        "module",
        "modules",
        "package",
        "packages",
        "path",
        "paths",
        "symbol",
        "symbols",
        "test",
        "tests",
    }
)
_METADATA_TERM_KINDS = {
    "alias": "alias",
    "aliases": "alias",
    "command": "command",
    "commands": "command",
    "commit": "commit",
    "commits": "commit",
    "error": "error",
    "errors": "error",
    "module": "module",
    "modules": "module",
    "package": "package",
    "packages": "package",
    "path": "path",
    "paths": "path",
    "symbol": "symbol",
    "symbols": "symbol",
    "test": "test",
    "tests": "test",
}


@dataclass(frozen=True, slots=True)
class ExactTerm:
    value: str
    kind: str


def normalize_term(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def projection_scope_key(
    visibility: Visibility,
    *,
    repository_id: str,
    run_id: str,
    task_id: str | None,
) -> str:
    if visibility is Visibility.REPOSITORY:
        return f"repository:{repository_id}"
    if visibility is Visibility.RUN:
        return f"run:{run_id}"
    if task_id is None:
        raise ValueError("task-visible memory requires task_id")
    return f"task:{task_id}"


def domain_lane(kind: MemoryKindValue, metadata: JsonObject | None = None) -> str:
    retrieval = (metadata or {}).get("retrieval")
    if isinstance(retrieval, dict):
        hint = retrieval.get("domain_lane")
        if isinstance(hint, str) and normalize_term(hint) in {
            "knowledge",
            "playbook",
            "execution_history",
            "handoff",
            "evidence",
            "coordination",
            "graph",
            "summary",
        }:
            return normalize_term(hint)
    value = str(kind)
    if value in {MemoryKind.PROCEDURE.value, MemoryKind.WARNING.value}:
        return "playbook"
    if value in {MemoryKind.ATTEMPT.value, MemoryKind.OUTCOME.value}:
        return "execution_history"
    if value == MemoryKind.HANDOFF.value:
        return "handoff"
    return "knowledge"


def search_text(
    *,
    title: str | None,
    content: MemoryContent,
    tags: tuple[str, ...],
    metadata: JsonObject | None = None,
) -> str:
    """Build a stable multilingual/code-friendly full-text representation."""

    metadata_terms = _metadata_terms(metadata or {})
    return _search_text_from_terms(
        title=title,
        content=content,
        tags=tags,
        metadata_terms=metadata_terms,
    )


def exact_terms(
    *,
    memory_id: str,
    content_sha256: str,
    title: str | None,
    content: MemoryContent,
    tags: tuple[str, ...],
    metadata: JsonObject | None = None,
) -> tuple[ExactTerm, ...]:
    """Return bounded equality terms used by the B-tree exact lane."""

    candidates: list[tuple[str, str]] = [
        ("memory_id", memory_id),
        ("content_sha256", content_sha256),
    ]
    if title:
        candidates.append(("title", title))
    candidates.extend(("tag", tag) for tag in tags)
    content_value = memory_content_text(content)
    if len(content_value) <= MAX_EXACT_TERM:
        candidates.append(("content", content_value))
    metadata_terms = _metadata_terms(metadata or {})
    projected = _search_text_from_terms(
        title=title,
        content=content,
        tags=tags,
        metadata_terms=metadata_terms,
    )
    for kind, pattern in _LOOKUP_PATTERNS:
        candidates.extend((kind, match.group(0)) for match in pattern.finditer(projected))
    candidates.extend(metadata_terms)

    terms: list[ExactTerm] = []
    seen: set[tuple[str, str]] = set()
    for kind, raw in candidates:
        value = normalize_term(raw)
        key = (kind, value)
        if not value or len(value) > MAX_EXACT_TERM or key in seen:
            continue
        seen.add(key)
        terms.append(ExactTerm(value=value, kind=kind))
        if len(terms) >= MAX_EXACT_TERMS:
            break
    return tuple(terms)


def lookup_text(
    *,
    memory_id: str,
    content_sha256: str,
    title: str | None,
    content: MemoryContent,
    tags: tuple[str, ...],
    metadata: JsonObject | None = None,
) -> str:
    terms = exact_terms(
        memory_id=memory_id,
        content_sha256=content_sha256,
        title=title,
        content=content,
        tags=tags,
        metadata=metadata,
    )
    fuzzy_kinds = {
        "alias",
        "command",
        "commit",
        "error",
        "module",
        "package",
        "path",
        "symbol",
        "tag",
        "test",
        "title",
    }
    return "\n".join(term.value for term in terms if term.kind in fuzzy_kinds)[:MAX_LOOKUP_TEXT]


def trigram_similarity(left: str, right: str) -> float:
    """Portable reference similarity for adapter parity and deterministic tests."""

    left_normalized = normalize_term(left)
    right_normalized = normalize_term(right)
    if len(left_normalized) < 3 or len(right_normalized) < 3:
        return 0.0
    left_trigrams = _trigrams(left_normalized)
    right_trigrams = _trigrams(right_normalized)
    if not left_trigrams or not right_trigrams:
        return 0.0
    return len(left_trigrams & right_trigrams) / len(left_trigrams | right_trigrams)


def _trigrams(value: str) -> set[str]:
    trigrams: set[str] = set()
    for word in _TRIGRAM_WORD.findall(value):
        padded = f"  {word} "
        trigrams.update(padded[index : index + 3] for index in range(len(padded) - 2))
    return trigrams


def _metadata_terms(metadata: JsonObject) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    visited: set[int] = set()
    visited_nodes = 0

    def visit(key: str, value: Any, depth: int) -> None:
        nonlocal visited_nodes
        if depth > 4 or len(found) >= MAX_METADATA_TERMS or visited_nodes >= MAX_METADATA_NODES:
            return
        visited_nodes += 1
        normalized_key = normalize_term(key[:128])
        if isinstance(value, dict):
            if id(value) in visited:
                return
            visited.add(id(value))
            for nested_key, nested_value in value.items():
                if visited_nodes >= MAX_METADATA_NODES:
                    break
                visit(str(nested_key)[:128], nested_value, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            if id(value) in visited:
                return
            visited.add(id(value))
            for item in value:
                if visited_nodes >= MAX_METADATA_NODES:
                    break
                visit(key, item, depth + 1)
            return
        if normalized_key in _METADATA_LOOKUP_KEYS and isinstance(value, (str, int)):
            raw_value = str(value)
            if len(raw_value) <= MAX_EXACT_TERM:
                found.append((_METADATA_TERM_KINDS[normalized_key], raw_value))

    for item_key, item_value in metadata.items():
        if visited_nodes >= MAX_METADATA_NODES:
            break
        visit(str(item_key)[:128], item_value, 0)
    return tuple(found)


def _search_text_from_terms(
    *,
    title: str | None,
    content: MemoryContent,
    tags: tuple[str, ...],
    metadata_terms: tuple[tuple[str, str], ...],
) -> str:
    """Join projected fields without first allocating an unbounded intermediate string."""

    parts = (title or "", memory_content_text(content), *tags)
    projected: list[str] = []
    remaining = MAX_SEARCH_TEXT
    for raw in (*parts, *(value for _, value in metadata_terms)):
        if remaining <= 0:
            break
        value = raw.strip()
        if not value:
            continue
        separator = 1 if projected else 0
        if remaining <= separator:
            break
        projected.append(value[: remaining - separator])
        remaining -= len(projected[-1]) + separator
    return "\n".join(projected)


__all__ = [
    "ExactTerm",
    "FUZZY_SIMILARITY_THRESHOLD",
    "MAX_LOOKUP_TEXT",
    "MAX_QUERY_CHARS",
    "MAX_QUERY_TOKENS",
    "MAX_SEARCH_TEXT",
    "RETRIEVAL_PROJECTION_ID",
    "domain_lane",
    "exact_terms",
    "lookup_text",
    "normalize_term",
    "projection_scope_key",
    "search_text",
    "trigram_similarity",
]
