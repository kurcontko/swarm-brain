"""Auditable principal routing and fail-closed active-forgetting policy."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contracts import (
    GATEMEM_COMMIT,
    GateMemContractError,
    PublicEpisode,
    PublicTurn,
    assert_hidden_fields_absent,
)


class AudiencePolicy(Protocol):
    """Choose isolated principal views from public episode/turn data only."""

    def audiences(self, episode: PublicEpisode, turn: PublicTurn) -> frozenset[str]: ...


class SpeakerOnlyAudiencePolicy:
    """Conservative default: a turn enters only its speaker's principal scope."""

    def audiences(self, episode: PublicEpisode, turn: PublicTurn) -> frozenset[str]:
        del episode
        return frozenset({turn.speaker.principal_id})


class ManifestAudiencePolicy:
    """A complete, reviewable turn-to-principal allowlist.

    The manifest may be produced by a policy engine, but the benchmark runner
    consumes only this static artifact.  Every turn must have an entry; there
    is no broadcast or same-episode fallback.
    """

    def __init__(
        self,
        mapping: dict[str, dict[str, frozenset[str]]],
        *,
        manifest_sha256: str,
    ) -> None:
        self._mapping = mapping
        self.manifest_sha256 = manifest_sha256

    @classmethod
    def from_path(cls, path: str | Path) -> ManifestAudiencePolicy:
        source = Path(path)
        try:
            raw_bytes = source.read_bytes()
            raw = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise GateMemContractError(f"invalid audience manifest: {path}") from exc
        if not isinstance(raw, dict):
            raise GateMemContractError("audience manifest must be an object")
        assert_hidden_fields_absent(raw)
        if raw.get("schema_version") != 1:
            raise GateMemContractError("audience manifest schema_version must be 1")
        if raw.get("gatemem_commit") != GATEMEM_COMMIT:
            raise GateMemContractError("audience manifest must name the pinned GateMem commit")
        episodes = raw.get("episodes")
        if not isinstance(episodes, dict):
            raise GateMemContractError("audience manifest episodes must be an object")
        mapping: dict[str, dict[str, frozenset[str]]] = {}
        for episode_id, turns in episodes.items():
            if not isinstance(episode_id, str) or not episode_id or not isinstance(turns, dict):
                raise GateMemContractError("audience manifest episode entries are malformed")
            mapped_turns: dict[str, frozenset[str]] = {}
            for turn_id, principals in turns.items():
                if (
                    not isinstance(turn_id, str)
                    or not turn_id
                    or not isinstance(principals, list)
                    or not principals
                    or any(not isinstance(item, str) or not item for item in principals)
                ):
                    raise GateMemContractError("audience manifest turn entry is malformed")
                if len(principals) != len(set(principals)):
                    raise GateMemContractError("audience manifest principal IDs must be unique")
                mapped_turns[turn_id] = frozenset(principals)
            mapping[episode_id] = mapped_turns
        return cls(mapping, manifest_sha256=hashlib.sha256(raw_bytes).hexdigest())

    def audiences(self, episode: PublicEpisode, turn: PublicTurn) -> frozenset[str]:
        try:
            principals = self._mapping[episode.episode_id][turn.turn_id]
        except KeyError as exc:
            raise GateMemContractError(
                f"audience manifest has no entry for {episode.episode_id}/{turn.turn_id}"
            ) from exc
        unknown = principals.difference(episode.principal_ids)
        if unknown:
            raise GateMemContractError(
                f"audience manifest names unknown principals: {sorted(unknown)}"
            )
        return principals


@dataclass(frozen=True, slots=True)
class KnownMemory:
    memory_id: str
    version: int
    source_turn_id: str
    text: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class TurnPlan:
    remember: bool
    forget_memory_ids: tuple[str, ...] = ()
    new_fence_hashes: tuple[str, ...] = ()
    reason: str = "observation"


class TurnInterpreter(Protocol):
    """Compile one public turn without access to checkpoint annotations."""

    def plan(
        self,
        turn: PublicTurn,
        known_memories: tuple[KnownMemory, ...],
        deletion_fence_hashes: frozenset[str],
    ) -> TurnPlan: ...


_DELETE_DIRECTIVE = re.compile(
    r"(?:\b(?:please\s+)?(?:delete|forget|erase)\b|"
    r"\bdo\s+not\s+retain\b|"
    r"\bshould\s+be\s+(?:treated\s+as\s+)?deleted\b|"
    r"\bshould\s+(?:be\s+)?(?:unavailable|not\s+be\s+retained)\b)",
    flags=re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_@./:+-]*")
_QUOTED = re.compile(r"[\"']([^\"']{3,})[\"']")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{6,}\d)(?!\d)")

_STOPWORDS = frozenset(
    {
        "a",
        "after",
        "again",
        "all",
        "an",
        "and",
        "as",
        "at",
        "be",
        "before",
        "but",
        "by",
        "delete",
        "deleted",
        "do",
        "earlier",
        "exact",
        "for",
        "from",
        "going",
        "i",
        "in",
        "is",
        "it",
        "later",
        "memory",
        "not",
        "now",
        "of",
        "on",
        "only",
        "or",
        "please",
        "prior",
        "retain",
        "should",
        "the",
        "this",
        "to",
        "too",
        "treated",
        "unavailable",
        "value",
        "was",
        "with",
    }
)


class DeterministicTurnInterpreter:
    """Store observations and turn explicit deletion language into tombstones.

    Raw deletion instructions are never persisted.  Target matching uses only
    memories already visible in the same principal scope.  The interpreter
    stores hashes of deleted distinctive phrases so later turns cannot silently
    reintroduce the deleted value.
    """

    def plan(
        self,
        turn: PublicTurn,
        known_memories: tuple[KnownMemory, ...],
        deletion_fence_hashes: frozenset[str],
    ) -> TurnPlan:
        structured_targets, structured_delete = self._structured_deletion(turn, known_memories)
        is_delete = structured_delete or bool(_DELETE_DIRECTIVE.search(turn.text))
        if not is_delete:
            if deletion_fence_hashes.intersection(_candidate_hashes(turn.text)):
                return TurnPlan(remember=False, reason="blocked_by_active_forgetting_fence")
            return TurnPlan(remember=True)

        matched = structured_targets or self._match_targets(turn.text, known_memories)
        fence_atoms = set(_distinctive_atoms(turn.text))
        for memory in known_memories:
            if memory.memory_id in matched:
                fence_atoms.update(_shared_distinctive_atoms(turn.text, memory.text))
        fence_hashes = tuple(sorted(_fence_hash(atom) for atom in fence_atoms))
        return TurnPlan(
            remember=False,
            forget_memory_ids=tuple(sorted(matched)),
            new_fence_hashes=fence_hashes,
            reason="active_forgetting" if matched else "active_forgetting_no_visible_target",
        )

    @staticmethod
    def _structured_deletion(
        turn: PublicTurn, known_memories: tuple[KnownMemory, ...]
    ) -> tuple[set[str], bool]:
        target_turn_ids: set[str] = set()
        saw_delete = False
        for encoded in turn.memory_ops_json:
            operation = json.loads(encoded)
            action = str(
                operation.get("op") or operation.get("type") or operation.get("action") or ""
            ).casefold()
            if action not in {"delete", "forget", "erase", "remove"}:
                continue
            saw_delete = True
            single = operation.get("target_turn_id")
            if isinstance(single, str) and single:
                target_turn_ids.add(single)
            multiple = operation.get("target_turn_ids") or []
            if isinstance(multiple, list):
                target_turn_ids.update(item for item in multiple if isinstance(item, str) and item)
        return (
            {
                memory.memory_id
                for memory in known_memories
                if memory.source_turn_id in target_turn_ids
            },
            saw_delete,
        )

    @staticmethod
    def _match_targets(text: str, known_memories: tuple[KnownMemory, ...]) -> set[str]:
        directive_tokens = set(_meaningful_tokens(text))
        directive_atoms = set(_distinctive_atoms(text))
        scored: list[tuple[int, str]] = []
        for memory in known_memories:
            memory_text = memory.text.casefold()
            atom_hits = sum(1 for atom in directive_atoms if atom in memory_text)
            token_hits = len(directive_tokens.intersection(_meaningful_tokens(memory.text)))
            shared_phrases = len(_shared_distinctive_atoms(text, memory.text))
            if atom_hits == 0 and token_hits < 3 and shared_phrases == 0:
                continue
            score = atom_hits * 100 + shared_phrases * 20 + token_hits
            scored.append((score, memory.memory_id))
        if not scored:
            return set()
        best = max(score for score, _ in scored)
        floor = max(3, int(best * 0.6))
        return {memory_id for score, memory_id in scored if score >= floor}


def _meaningful_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for raw in _TOKEN.findall(text)
        if len(token := raw.casefold().strip(".,:;!?()[]{}\"'")) >= 3 and token not in _STOPWORDS
    )


def _distinctive_atoms(text: str) -> tuple[str, ...]:
    atoms = {match.group(1).strip().casefold() for match in _QUOTED.finditer(text)}
    atoms.update(match.group(0).strip().casefold() for match in _PHONE.finditer(text))
    tokens = _meaningful_tokens(text)
    atoms.update(
        token for token in tokens if len(token) >= 6 and any(char.isdigit() for char in token)
    )
    for size in range(2, min(5, len(tokens)) + 1):
        atoms.update(
            " ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
        )
    return tuple(sorted(atom for atom in atoms if atom))


def _shared_distinctive_atoms(left: str, right: str) -> frozenset[str]:
    left_atoms = set(_distinctive_atoms(left))
    right_casefolded = right.casefold()
    return frozenset(atom for atom in left_atoms if atom in right_casefolded)


def _fence_hash(atom: str) -> str:
    return hashlib.sha256(f"gatemem-active-forgetting-v1:{atom}".encode()).hexdigest()


def _candidate_hashes(text: str) -> frozenset[str]:
    return frozenset(_fence_hash(atom) for atom in _distinctive_atoms(text))


__all__ = [
    "AudiencePolicy",
    "DeterministicTurnInterpreter",
    "KnownMemory",
    "ManifestAudiencePolicy",
    "SpeakerOnlyAudiencePolicy",
    "TurnInterpreter",
    "TurnPlan",
]
