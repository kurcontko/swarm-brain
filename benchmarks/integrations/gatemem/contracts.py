"""Pinned GateMem inputs and the deliberately narrow public agent view.

The official checkpoint files contain both public query fields and hidden
scoring annotations.  Raw checkpoint dictionaries terminate in this module:
the runner and answer model receive only :class:`PublicCheckpoint`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

GATEMEM_COMMIT = "603f9f4b4ba4b77f043c20f85687fa016fd720b0"
GATEMEM_DOMAINS = frozenset({"education", "household", "medical", "office"})

# Hashes from the official checkout above.  The git commit check catches code
# drift; these checks make dataset and external-scorer drift explicit even in a
# checkout with local modifications.
GATEMEM_SHA256 = {
    "bench/data/education/episodes.jsonl": (
        "5971478a96553c2eb18a9f1e079987275da0cddf6175e40abda8a58525e65862"
    ),
    "bench/data/education/checkpoints.jsonl": (
        "2a372a5017a99108c83324d2fed25dbfb7797a2fb25a20fc066ed5b2d05739b5"
    ),
    "bench/data/household/episodes.jsonl": (
        "e2bb506cc1bdc8dc7b16d4a57610147365798d03eb1c326f9197b6c6221efb6f"
    ),
    "bench/data/household/checkpoints.jsonl": (
        "4692c3e1ea91be63debd7b39ccc5fc425bc1a83155c1bfb5c197ba68cefd93ae"
    ),
    "bench/data/medical/episodes.jsonl": (
        "d4b01f62cbd3457715b5958cf2d07d496c6e59280d28041927e2b3ab8a444b9f"
    ),
    "bench/data/medical/checkpoints.jsonl": (
        "2447f5905081e6002f16ad14f7f7dc19e14b7181e5ac8ebd08144d68da6437a0"
    ),
    "bench/data/office/episodes.jsonl": (
        "e15797cc0ab778067e0235377a09811c73182b5e1aa28e2221adc5ef9638a409"
    ),
    "bench/data/office/checkpoints.jsonl": (
        "7186bc1dd66b493f994b802bad8817d7aa5e7a91f498d67f87410a98f8116d0c"
    ),
    "bench/scripts/score_predictions.py": (
        "3d546a21778202959a9df12bac44c196a7f20a248cf5a2cb34f0d9b9c2623d8a"
    ),
    "docs/prediction_format.md": (
        "f1d0b8f947d81e03acb01743319ca4bfebf88349575fabfccbb8de8fcb55d1ed"
    ),
    "docs/assets/leaderboard.json": (
        "40676fb5b5b8682fd57f75deb7c62ad90da74e9d9ce2f23072c91a59d2daa2c8"
    ),
    "docs/assets/main_results.png": (
        "0162d722c6ac477a570ad4ed111b2cfbe8760ecad2e5d9c51530fe48d598f3b3"
    ),
    "configs/sweeps/paper_matrix.yaml": (
        "cdb92621d97042f5bd2260ff4f1273b2a6fedc9e410ca57dedbcaf15faa17f8b"
    ),
}

HIDDEN_CHECKPOINT_FIELDS = frozenset(
    {
        "query_type",
        "attack_type",
        "expected_action",
        "judge_spec",
        "leak_targets",
        "gold_answer_structured",
        "gold_refusal_category",
        "policy_snapshot",
    }
)


class GateMemContractError(ValueError):
    """The pinned dataset or adapter contract was violated."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateMemContractError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise GateMemContractError("GateMem public metadata must be JSON serializable") from exc


def content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicPrincipal:
    principal_id: str
    role: str

    @classmethod
    def from_raw(cls, raw: Any, *, field: str) -> PublicPrincipal:
        if not isinstance(raw, dict):
            raise GateMemContractError(f"{field} must be an object")
        return cls(
            principal_id=_required_text(raw.get("principal_id"), f"{field}.principal_id"),
            role=_required_text(raw.get("role"), f"{field}.role"),
        )


@dataclass(frozen=True, slots=True)
class PublicTurn:
    episode_id: str
    turn_id: str
    timestamp: str | None
    speaker: PublicPrincipal
    text: str
    turn_kind: str | None = None
    record_refs: tuple[str, ...] = ()
    memory_ops_json: tuple[str, ...] = ()

    @classmethod
    def from_raw(cls, episode_id: str, raw: Any) -> PublicTurn:
        if not isinstance(raw, dict):
            raise GateMemContractError("episode.turns[] must be an object")
        refs_raw = raw.get("record_refs") or []
        if not isinstance(refs_raw, list):
            raise GateMemContractError("turn.record_refs must be a list")
        refs = tuple(_required_text(item, "turn.record_refs[]") for item in refs_raw)
        ops_raw = raw.get("memory_ops") or []
        if not isinstance(ops_raw, list):
            raise GateMemContractError("turn.memory_ops must be a list")
        if any(not isinstance(item, dict) for item in ops_raw):
            raise GateMemContractError("turn.memory_ops[] must be an object")
        return cls(
            episode_id=episode_id,
            turn_id=_required_text(raw.get("turn_id"), "turn.turn_id"),
            timestamp=_optional_text(raw.get("timestamp"), "turn.timestamp"),
            speaker=PublicPrincipal.from_raw(raw.get("speaker"), field="turn.speaker"),
            text=_required_text(raw.get("text"), "turn.text"),
            turn_kind=_optional_text(raw.get("turn_kind"), "turn.turn_kind"),
            record_refs=refs,
            memory_ops_json=tuple(_canonical_json(item) for item in ops_raw),
        )


@dataclass(frozen=True, slots=True)
class PublicEpisode:
    episode_id: str
    domain: str
    principals: tuple[PublicPrincipal, ...]
    relationship_facts_json: tuple[str, ...]
    turns: tuple[PublicTurn, ...]

    @classmethod
    def from_raw(cls, raw: Any) -> PublicEpisode:
        if not isinstance(raw, dict):
            raise GateMemContractError("episode must be an object")
        episode_id = _required_text(raw.get("episode_id"), "episode.episode_id")
        domain = _required_text(raw.get("domain"), "episode.domain")
        entities = raw.get("entities") or {}
        if not isinstance(entities, dict):
            raise GateMemContractError("episode.entities must be an object")
        principals_raw = entities.get("principals") or []
        relationships_raw = entities.get("relationships") or []
        if not isinstance(principals_raw, list):
            raise GateMemContractError("episode.entities.principals must be a list")
        if not isinstance(relationships_raw, list) or any(
            not isinstance(item, dict) for item in relationships_raw
        ):
            raise GateMemContractError("episode.entities.relationships must be a list of objects")
        assert_hidden_fields_absent(relationships_raw)
        turns_raw = raw.get("turns") or []
        if not isinstance(turns_raw, list) or not turns_raw:
            raise GateMemContractError("episode.turns must be a non-empty list")

        principals_by_id: dict[str, PublicPrincipal] = {}
        for index, item in enumerate(principals_raw):
            principal = PublicPrincipal.from_raw(
                item, field=f"episode.entities.principals[{index}]"
            )
            previous = principals_by_id.get(principal.principal_id)
            if previous is not None and previous.role != principal.role:
                raise GateMemContractError(
                    f"principal {principal.principal_id!r} has conflicting roles"
                )
            principals_by_id[principal.principal_id] = principal

        turns = tuple(PublicTurn.from_raw(episode_id, item) for item in turns_raw)
        if len({turn.turn_id for turn in turns}) != len(turns):
            raise GateMemContractError(f"episode {episode_id!r} has duplicate turn IDs")
        for turn in turns:
            previous = principals_by_id.get(turn.speaker.principal_id)
            if previous is not None and previous.role != turn.speaker.role:
                raise GateMemContractError(
                    f"speaker {turn.speaker.principal_id!r} has conflicting roles"
                )
            principals_by_id.setdefault(turn.speaker.principal_id, turn.speaker)

        return cls(
            episode_id=episode_id,
            domain=domain,
            principals=tuple(sorted(principals_by_id.values(), key=lambda item: item.principal_id)),
            relationship_facts_json=tuple(_canonical_json(item) for item in relationships_raw),
            turns=turns,
        )

    @property
    def principal_ids(self) -> frozenset[str]:
        return frozenset(item.principal_id for item in self.principals)

    def role_for(self, principal_id: str) -> str | None:
        return next(
            (item.role for item in self.principals if item.principal_id == principal_id), None
        )

    def relationship_facts_for(self, principal_id: str) -> tuple[str, ...]:
        """Return only public relationship rows that explicitly name the requester."""

        selected: list[str] = []
        for encoded in self.relationship_facts_json:
            relationship = json.loads(encoded)
            if any(
                isinstance(value, str)
                and str(key).casefold().endswith("_id")
                and value == principal_id
                for key, value in relationship.items()
            ):
                selected.append(encoded)
        return tuple(selected)


@dataclass(frozen=True, slots=True)
class PublicCheckpoint:
    """Only fields the official GateMem protocol permits the agent to see."""

    checkpoint_id: str
    episode_id: str
    as_of_turn_id: str
    asker: PublicPrincipal
    query_text: str

    @classmethod
    def from_raw(cls, raw: Any) -> PublicCheckpoint:
        if not isinstance(raw, dict):
            raise GateMemContractError("checkpoint must be an object")
        # Deliberately use an allowlist.  Hidden keys may be present in ``raw``
        # because the official scorer needs them; no reference to their values
        # survives this constructor.
        return cls(
            checkpoint_id=_required_text(raw.get("checkpoint_id"), "checkpoint.checkpoint_id"),
            episode_id=_required_text(raw.get("episode_id"), "checkpoint.episode_id"),
            as_of_turn_id=_required_text(raw.get("as_of_turn_id"), "checkpoint.as_of_turn_id"),
            asker=PublicPrincipal.from_raw(raw.get("asker"), field="checkpoint.asker"),
            query_text=_required_text(raw.get("query_text"), "checkpoint.query_text"),
        )


def assert_hidden_fields_absent(value: Any) -> None:
    """Recursively reject hidden annotation names at an agent-facing boundary."""

    if isinstance(value, dict):
        overlap = HIDDEN_CHECKPOINT_FIELDS.intersection(str(key) for key in value)
        if overlap:
            raise GateMemContractError(
                f"hidden GateMem fields crossed the agent boundary: {sorted(overlap)}"
            )
        for item in value.values():
            assert_hidden_fields_absent(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_hidden_fields_absent(item)


@dataclass(frozen=True, slots=True)
class PrincipalScope:
    """One isolated Swarm Brain run for one GateMem episode/principal view."""

    domain: str
    episode_id: str
    principal_id: str
    principal_role: str
    tenant_id: str
    project_id: str
    repository_id: str
    swarm_id: str
    run_id: str
    agent_id: str

    @property
    def key(self) -> str:
        return f"{self.episode_id}::{self.principal_id}"


class ScopeFactory:
    """Derive stable UUID scopes without putting identity in request bodies."""

    def __init__(self, *, seed: str = GATEMEM_COMMIT) -> None:
        self.seed = _required_text(seed, "scope seed")

    def for_principal(
        self,
        *,
        domain: str,
        episode_id: str,
        principal_id: str,
        principal_role: str,
    ) -> PrincipalScope:
        prefix = f"https://gatemem.ai/swarmbrain/{self.seed}"

        def identifier(suffix: str) -> str:
            return str(uuid5(NAMESPACE_URL, f"{prefix}/{suffix}"))

        return PrincipalScope(
            domain=domain,
            episode_id=episode_id,
            principal_id=principal_id,
            principal_role=principal_role,
            tenant_id=identifier("tenant"),
            project_id=identifier(f"domain/{domain}"),
            repository_id=identifier(f"episode/{episode_id}"),
            swarm_id=identifier(f"episode/{episode_id}/swarm"),
            run_id=identifier(f"episode/{episode_id}/principal/{principal_id}/run"),
            agent_id=identifier(f"episode/{episode_id}/principal/{principal_id}/agent"),
        )


@dataclass(frozen=True, slots=True)
class GateMemDataset:
    domain: str
    episodes: tuple[dict[str, Any], ...]
    checkpoints: tuple[dict[str, Any], ...]


class GateMemCheckout:
    """Verify and load the exact official GateMem checkout used by this adapter."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def verify(self, *, domain: str | None = None) -> None:
        if domain is not None and domain not in GATEMEM_DOMAINS:
            raise GateMemContractError(f"unsupported GateMem domain: {domain!r}")
        try:
            result = subprocess.run(
                ["git", "-C", str(self.path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise GateMemContractError(f"not a readable GateMem git checkout: {self.path}") from exc
        actual_commit = result.stdout.strip()
        if actual_commit != GATEMEM_COMMIT:
            raise GateMemContractError(
                f"GateMem checkout is {actual_commit}; expected pinned commit {GATEMEM_COMMIT}"
            )

        paths = [
            "bench/scripts/score_predictions.py",
            "docs/prediction_format.md",
            "docs/assets/leaderboard.json",
            "docs/assets/main_results.png",
            "configs/sweeps/paper_matrix.yaml",
        ]
        selected_domains = (domain,) if domain is not None else tuple(sorted(GATEMEM_DOMAINS))
        for selected in selected_domains:
            paths.extend(
                [
                    f"bench/data/{selected}/episodes.jsonl",
                    f"bench/data/{selected}/checkpoints.jsonl",
                ]
            )
        for relative in paths:
            candidate = self.path / relative
            if not candidate.is_file():
                raise GateMemContractError(f"missing pinned GateMem file: {relative}")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            expected = GATEMEM_SHA256[relative]
            if digest != expected:
                raise GateMemContractError(
                    f"GateMem file digest mismatch for {relative}: {digest} != {expected}"
                )

    def load(self, domain: str) -> GateMemDataset:
        self.verify(domain=domain)
        root = self.path / "bench" / "data" / domain
        return GateMemDataset(
            domain=domain,
            episodes=tuple(_load_jsonl(root / "episodes.jsonl")),
            checkpoints=tuple(_load_jsonl(root / "checkpoints.jsonl")),
        )

    def official_score_command(
        self,
        *,
        domain: str,
        predictions: str | Path,
        out_dir: str | Path,
        python_executable: str = "python",
    ) -> tuple[str, ...]:
        self.verify(domain=domain)
        return (
            python_executable,
            str(self.path / "bench" / "scripts" / "score_predictions.py"),
            "--data_dir",
            str(self.path / "bench" / "data" / domain),
            "--predictions",
            str(Path(predictions).resolve()),
            "--out_dir",
            str(Path(out_dir).resolve()),
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateMemContractError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise GateMemContractError(f"JSONL row must be an object in {path}:{line_number}")
        rows.append(row)
    return rows


__all__ = [
    "GATEMEM_COMMIT",
    "GATEMEM_DOMAINS",
    "GATEMEM_SHA256",
    "GateMemCheckout",
    "GateMemContractError",
    "GateMemDataset",
    "HIDDEN_CHECKPOINT_FIELDS",
    "PrincipalScope",
    "PublicCheckpoint",
    "PublicEpisode",
    "PublicPrincipal",
    "PublicTurn",
    "ScopeFactory",
    "assert_hidden_fields_absent",
    "content_sha256",
]
