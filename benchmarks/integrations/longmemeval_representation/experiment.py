"""Pure/offline E6/SB-HMR-v1 representation ranking and one-hop controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .contracts import (
    ARTIFACT_TYPE,
    CELL_KEY_FAMILIES,
    KEY_FAMILY_DEPTH,
    MAX_EXPANSION_HOPS,
    MAX_HYDRATED_VALUES,
    MAX_NEIGHBORS_PER_NODE,
    PROTOCOL_VERSION,
    RRF_K,
    SCHEMA_VERSION,
    SIMILARITY_EDGE_THRESHOLD,
    CanonicalValue,
    ConstructionReceipt,
    EntityAdjacencyGraph,
    Graph,
    IndexedKeyBinding,
    KeyFamily,
    RankedFamilyObservation,
    RepresentationCell,
    RepresentationCorpus,
    RepresentationError,
    SimilarityAdjacencyGraph,
    ValueProvenance,
    sha256_json,
)


@dataclass(frozen=True, slots=True)
class FamilyValueContribution:
    family: KeyFamily
    best_rank: int
    witness_key_ids: tuple[str, ...]
    suppressed_same_family_hits: int
    rrf_contribution: float

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "best_rank": self.best_rank,
            "witness_key_ids": list(self.witness_key_ids),
            "suppressed_same_family_hits": self.suppressed_same_family_hits,
            "rrf_contribution": self.rrf_contribution,
        }


@dataclass(frozen=True, slots=True)
class HydratedValueScore:
    value: CanonicalValue
    score: float
    best_prior_rank: int | None
    family_contributions: tuple[FamilyValueContribution, ...]
    entity_score: float | None = None
    graph_support_count: int = 0
    expanded: bool = False

    def content_free_binding(self, *, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "value": self.value.content_free_binding(),
            "score": self.score,
            "best_prior_rank": self.best_prior_rank,
            "family_contributions": [
                contribution.content_free_binding() for contribution in self.family_contributions
            ],
            "entity_score": self.entity_score,
            "graph_support_count": self.graph_support_count,
            "expanded": self.expanded,
        }


@dataclass(frozen=True, slots=True)
class RepresentationResult:
    """Ranked canonical raw values plus a content-free evidence trace."""

    cell: RepresentationCell
    hydrated_values: tuple[CanonicalValue, ...]
    trace: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.cell, RepresentationCell):
            raise RepresentationError("representation result cell must be registered")
        if not isinstance(self.hydrated_values, tuple):
            raise RepresentationError("hydrated values must be an immutable tuple")
        if any(not isinstance(value, CanonicalValue) for value in self.hydrated_values):
            raise RepresentationError("result hydration must contain canonical raw values")
        if len(self.hydrated_values) > MAX_HYDRATED_VALUES:
            raise RepresentationError("result exceeds the hydrated-value cap")
        ids = [value.value_id for value in self.hydrated_values]
        if len(set(ids)) != len(ids):
            raise RepresentationError("result hydrates a canonical value more than once")
        if not isinstance(self.trace, dict):
            raise RepresentationError("representation result trace must be a dictionary")
        if self.trace.get("hydrated_value_ids") != ids:
            raise RepresentationError("representation trace and hydrated values disagree")
        hashes = [value.raw_value_sha256 for value in self.hydrated_values]
        if self.trace.get("hydrated_raw_value_hashes") != hashes:
            raise RepresentationError("representation trace and raw value hashes disagree")

    @property
    def hydrated_raw_values(self) -> tuple[str, ...]:
        return tuple(value.raw_value for value in self.hydrated_values)

    @property
    def trace_sha256(self) -> str:
        return sha256_json(self.trace)

    def content_free_artifact(self) -> dict[str, Any]:
        return {**self.trace, "trace_sha256": self.trace_sha256}


def _validate_construction_for_cell(
    corpus: RepresentationCorpus,
    families: tuple[KeyFamily, ...],
) -> tuple[ConstructionReceipt, ...]:
    required_derived = tuple(family for family in families if family is not KeyFamily.RAW)
    receipts: list[ConstructionReceipt] = []
    value_ids = {value.value_id for value in corpus.values}
    key_counts: dict[tuple[str, KeyFamily], int] = {}
    for key in corpus.derived_keys:
        key_counts[(key.source_value_id, key.family)] = (
            key_counts.get((key.source_value_id, key.family), 0) + 1
        )
    for family in required_derived:
        family_receipts = corpus.receipts_for_family(family)
        if {receipt.source_value_id for receipt in family_receipts} != value_ids:
            raise RepresentationError(
                f"cell requires one complete {family.value} construction receipt per value"
            )
        for receipt in family_receipts:
            count = key_counts.get((receipt.source_value_id, family), 0)
            if family in {KeyFamily.MERGED_SFK, KeyFamily.SUMMARY, KeyFamily.PRIMARY_ABSTRACTION}:
                if count != 1:
                    raise RepresentationError(
                        f"{family.value} requires exactly one derived key per source value"
                    )
            elif family is KeyFamily.CUE_ANCHOR and count < 1:
                raise RepresentationError("cue-anchor requires at least one key per source value")
        receipts.extend(family_receipts)
    return tuple(receipts)


def _validate_observations(
    corpus: RepresentationCorpus,
    *,
    cell: RepresentationCell,
    observations: tuple[RankedFamilyObservation, ...],
) -> tuple[KeyFamily, ...]:
    if not isinstance(observations, tuple):
        raise RepresentationError("family observations must be an immutable tuple")
    families = CELL_KEY_FAMILIES[cell]
    if tuple(observation.family for observation in observations) != families:
        raise RepresentationError("cell observations do not match its exact key-family order")
    query_hashes: set[str] = set()
    keys_by_id = corpus.keys_by_id()
    for observation in observations:
        if not isinstance(observation, RankedFamilyObservation):
            raise RepresentationError("observations must be RankedFamilyObservation")
        if observation.question_id != corpus.question_id:
            raise RepresentationError("ranking observation crosses the question boundary")
        if observation.source_artifact_sha256 != corpus.source_artifact_sha256:
            raise RepresentationError("ranking observation crosses the source boundary")
        if observation.projection_sha256 != corpus.projection_sha256:
            raise RepresentationError("ranking observation crosses the projection boundary")
        if observation.index_sha256 != corpus.index_sha256:
            raise RepresentationError("ranking observation index binding is stale or tampered")
        family_keys = corpus.keys_for_family(observation.family)
        if observation.indexed_key_count != len(family_keys):
            raise RepresentationError("ranking indexed-key count does not match the corpus")
        for ranked in observation.ranked_keys:
            key = keys_by_id.get(ranked.key_id)
            if key is None:
                raise RepresentationError("ranking observation contains an unknown key")
            if key.family is not observation.family:
                raise RepresentationError("ranking observation contains a cross-family key")
        query_hashes.add(observation.query_sha256)
    if len(query_hashes) != 1:
        raise RepresentationError("all family rankings must bind one exact query")
    return families


def _general_rrf(
    corpus: RepresentationCorpus,
    observations: tuple[RankedFamilyObservation, ...],
) -> tuple[tuple[HydratedValueScore, ...], dict[str, Any]]:
    keys = corpus.keys_by_id()
    values = corpus.values_by_id()
    hits: dict[tuple[str, KeyFamily], list[tuple[int, str]]] = {}
    family_accounting: dict[str, dict[str, int]] = {}
    for observation in observations:
        reached_values: set[str] = set()
        for rank, ranked in enumerate(observation.ranked_keys, start=1):
            value_id = keys[ranked.key_id].source_value_id
            hits.setdefault((value_id, observation.family), []).append((rank, ranked.key_id))
            reached_values.add(value_id)
        family_accounting[observation.family.value] = {
            "indexed_keys": observation.indexed_key_count,
            "returned_key_hits": len(observation.ranked_keys),
            "unique_values_reached": len(reached_values),
            "same_family_fanout_hits_suppressed": (
                len(observation.ranked_keys) - len(reached_values)
            ),
        }

    by_value: dict[str, list[FamilyValueContribution]] = {}
    for (value_id, family), family_hits in hits.items():
        ordered_hits = sorted(family_hits, key=lambda item: (item[0], item[1]))
        best_rank = ordered_hits[0][0]
        contribution = FamilyValueContribution(
            family=family,
            best_rank=best_rank,
            witness_key_ids=tuple(key_id for _, key_id in ordered_hits),
            suppressed_same_family_hits=len(ordered_hits) - 1,
            rrf_contribution=1.0 / (RRF_K + best_rank),
        )
        by_value.setdefault(value_id, []).append(contribution)

    scored: list[HydratedValueScore] = []
    for value_id, contributions in by_value.items():
        ordered = tuple(sorted(contributions, key=lambda item: item.family.value))
        scored.append(
            HydratedValueScore(
                value=values[value_id],
                score=sum(item.rrf_contribution for item in ordered),
                best_prior_rank=min(item.best_rank for item in ordered),
                family_contributions=ordered,
            )
        )
    scored.sort(
        key=lambda item: (
            -item.score,
            item.best_prior_rank if item.best_prior_rank is not None else 2**63,
            item.value.value_id,
        )
    )
    return tuple(scored), {
        "method": "equal-family-weighted-RRF-after-within-family-value-dedup",
        "rrf_k": RRF_K,
        "family_weight": 1.0,
        "family_accounting": family_accounting,
        "tie_break": "best-prior-key-rank-then-canonical-value-id",
    }


@dataclass(frozen=True, slots=True)
class _EntitySeed:
    entity_id: str
    best_raw_score: float
    best_key_rank: int
    witness_key_ids: tuple[str, ...]


def _entity_seed_state(
    corpus: RepresentationCorpus,
    observation: RankedFamilyObservation,
) -> tuple[dict[str, _EntitySeed], dict[str, set[str]], dict[str, IndexedKeyBinding]]:
    keys = corpus.keys_by_id()
    entity_values: dict[str, set[str]] = {}
    for key in corpus.keys_for_family(KeyFamily.ENTITY_DESCRIPTION):
        assert key.entity_id is not None
        entity_values.setdefault(key.entity_id, set()).add(key.source_value_id)

    hits: dict[str, list[tuple[int, float, str]]] = {}
    for rank, ranked in enumerate(observation.ranked_keys, start=1):
        key = keys[ranked.key_id]
        assert key.entity_id is not None
        hits.setdefault(key.entity_id, []).append((rank, ranked.raw_score, ranked.key_id))
    seeds: dict[str, _EntitySeed] = {}
    for entity_id, entity_hits in hits.items():
        ordered = sorted(entity_hits, key=lambda item: (item[0], item[2]))
        seeds[entity_id] = _EntitySeed(
            entity_id=entity_id,
            best_raw_score=max(item[1] for item in entity_hits),
            best_key_rank=ordered[0][0],
            witness_key_ids=tuple(item[2] for item in ordered),
        )
    return seeds, entity_values, keys


def _entity_direct_scores(
    corpus: RepresentationCorpus,
    observation: RankedFamilyObservation,
) -> tuple[
    tuple[HydratedValueScore, ...],
    dict[str, _EntitySeed],
    dict[str, set[str]],
    dict[str, Any],
]:
    seeds, entity_values, _ = _entity_seed_state(corpus, observation)
    values = corpus.values_by_id()
    witnesses: dict[str, list[_EntitySeed]] = {}
    for seed in seeds.values():
        for value_id in entity_values.get(seed.entity_id, set()):
            witnesses.setdefault(value_id, []).append(seed)
    scored: list[HydratedValueScore] = []
    for value_id, supports in witnesses.items():
        ordered = sorted(supports, key=lambda item: (item.best_key_rank, item.entity_id))
        best_rank = ordered[0].best_key_rank
        all_keys = tuple(key_id for support in ordered for key_id in support.witness_key_ids)
        contribution = FamilyValueContribution(
            family=KeyFamily.ENTITY_DESCRIPTION,
            best_rank=best_rank,
            witness_key_ids=all_keys,
            suppressed_same_family_hits=max(0, len(all_keys) - 1),
            rrf_contribution=1.0 / (RRF_K + best_rank),
        )
        scored.append(
            HydratedValueScore(
                value=values[value_id],
                score=contribution.rrf_contribution,
                best_prior_rank=best_rank,
                family_contributions=(contribution,),
                entity_score=max(item.best_raw_score for item in supports),
                graph_support_count=len({item.entity_id for item in supports}),
            )
        )
    scored.sort(
        key=lambda item: (
            -item.score,
            item.best_prior_rank if item.best_prior_rank is not None else 2**63,
            item.value.value_id,
        )
    )
    # R5 consumes this exact direct rank; never trust a caller-provided one.
    scored = [replace(item, best_prior_rank=rank) for rank, item in enumerate(scored, start=1)]
    accounting = {
        "ranked_entity_keys": len(observation.ranked_keys),
        "unique_seed_entities": len(seeds),
        "same_entity_seed_keys_suppressed": len(observation.ranked_keys) - len(seeds),
        "direct_hydrated_values": len(scored),
    }
    return tuple(scored), seeds, entity_values, accounting


def _validate_value_provenance(
    provenance: ValueProvenance,
    values: dict[str, CanonicalValue],
    *,
    label: str,
) -> None:
    value = values.get(provenance.value_id)
    if value is None:
        raise RepresentationError(f"{label} refers to an unknown canonical value")
    if provenance.source_version_sha256 != value.source_version_sha256:
        raise RepresentationError(f"{label} source version is stale or tampered")
    if provenance.raw_value_sha256 != value.raw_value_sha256:
        raise RepresentationError(f"{label} raw value hash is stale or tampered")


def _validate_graph_binding(corpus: RepresentationCorpus, graph: Graph) -> None:
    if graph.question_id != corpus.question_id:
        raise RepresentationError("adjacency graph crosses the question boundary")
    if graph.source_artifact_sha256 != corpus.source_artifact_sha256:
        raise RepresentationError("adjacency graph crosses the source boundary")
    if graph.projection_sha256 != corpus.projection_sha256:
        raise RepresentationError("adjacency graph crosses the projection boundary")
    if graph.index_sha256 != corpus.index_sha256:
        raise RepresentationError("adjacency graph index binding is stale or tampered")
    if graph.source_input_sha256 != corpus.navigation_index_sha256:
        raise RepresentationError(
            "adjacency graph source-safe navigation binding is stale or tampered"
        )


def _entity_one_hop(
    corpus: RepresentationCorpus,
    observation: RankedFamilyObservation,
    graph: EntityAdjacencyGraph,
) -> tuple[tuple[HydratedValueScore, ...], dict[str, Any]]:
    _validate_graph_binding(corpus, graph)
    direct, seeds, entity_values, direct_accounting = _entity_direct_scores(corpus, observation)
    known_entities = set(entity_values)
    values = corpus.values_by_id()
    adjacency: dict[str, list[tuple[str, str]]] = {entity: [] for entity in known_entities}
    for edge in graph.edges:
        if edge.left_entity_id not in known_entities or edge.right_entity_id not in known_entities:
            raise RepresentationError("entity adjacency edge contains an orphan endpoint")
        endpoint_values = {
            edge.left_entity_id: entity_values[edge.left_entity_id],
            edge.right_entity_id: entity_values[edge.right_entity_id],
        }
        supported_endpoints: set[str] = set()
        for evidence in edge.evidence_values:
            _validate_value_provenance(evidence, values, label="entity edge provenance")
            matching = {
                entity_id
                for entity_id, bound_values in endpoint_values.items()
                if evidence.value_id in bound_values
            }
            if not matching:
                raise RepresentationError(
                    "entity edge provenance is unrelated to both entity endpoints"
                )
            supported_endpoints.update(matching)
        if supported_endpoints != {edge.left_entity_id, edge.right_entity_id}:
            raise RepresentationError(
                "entity edge provenance does not collectively support both endpoints"
            )
        adjacency[edge.left_entity_id].append((edge.right_entity_id, edge.edge_id))
        adjacency[edge.right_entity_id].append((edge.left_entity_id, edge.edge_id))

    direct_rank = {item.value.value_id: rank for rank, item in enumerate(direct, start=1)}
    support: dict[str, list[dict[str, Any]]] = {}
    traversed_edges: set[str] = set()
    for seed in seeds.values():
        targets = [(seed.entity_id, None)]
        targets.extend(adjacency.get(seed.entity_id, ()))
        for target_entity, edge_id in targets:
            if edge_id is not None:
                traversed_edges.add(edge_id)
            for value_id in sorted(entity_values.get(target_entity, set())):
                support.setdefault(value_id, []).append(
                    {
                        "seed_entity_id": seed.entity_id,
                        "seed_key_rank": seed.best_key_rank,
                        "seed_key_ids": list(seed.witness_key_ids),
                        "entity_score": seed.best_raw_score,
                        "target_entity_id": target_entity,
                        "edge_id": edge_id,
                        "hop": 0 if edge_id is None else 1,
                    }
                )

    scored: list[HydratedValueScore] = []
    witness_trace: dict[str, list[dict[str, Any]]] = {}
    for value_id, raw_witnesses in support.items():
        unique: dict[tuple[str, str, str | None], dict[str, Any]] = {}
        for witness in raw_witnesses:
            key = (
                witness["seed_entity_id"],
                witness["target_entity_id"],
                witness["edge_id"],
            )
            unique[key] = witness
        witnesses = sorted(
            unique.values(),
            key=lambda item: (
                item["hop"],
                item["seed_key_rank"],
                item["seed_entity_id"],
                item["target_entity_id"],
                item["edge_id"] or "",
            ),
        )
        best_entity_score = max(float(item["entity_score"]) for item in witnesses)
        graph_support_count = len({item["seed_entity_id"] for item in witnesses})
        prior = direct_rank.get(value_id)
        best_seed_rank = min(int(item["seed_key_rank"]) for item in witnesses)
        witness_keys = tuple(
            dict.fromkeys(key_id for item in witnesses for key_id in item["seed_key_ids"])
        )
        contribution = FamilyValueContribution(
            family=KeyFamily.ENTITY_DESCRIPTION,
            best_rank=best_seed_rank,
            witness_key_ids=witness_keys,
            suppressed_same_family_hits=max(0, len(witness_keys) - 1),
            rrf_contribution=1.0 / (RRF_K + best_seed_rank),
        )
        scored.append(
            HydratedValueScore(
                value=values[value_id],
                score=contribution.rrf_contribution,
                best_prior_rank=prior,
                family_contributions=(contribution,),
                entity_score=best_entity_score,
                graph_support_count=graph_support_count,
                expanded=any(item["hop"] == 1 for item in witnesses) and prior is None,
            )
        )
        witness_trace[value_id] = witnesses
    scored.sort(
        key=lambda item: (
            -(item.entity_score if item.entity_score is not None else float("-inf")),
            -item.graph_support_count,
            item.best_prior_rank if item.best_prior_rank is not None else 2**63,
            item.value.value_id,
        )
    )
    return tuple(scored), {
        **direct_accounting,
        "expansion_hops": MAX_EXPANSION_HOPS,
        "recursive_expansion": False,
        "traversed_edge_ids": sorted(traversed_edges),
        "traversed_edge_count": len(traversed_edges),
        "hydration_witnesses": witness_trace,
        "ordering": (
            "best-entity-raw-score-desc,distinct-seed-support-desc,"
            "recomputed-R4-prior-rank-asc,canonical-value-id-asc"
        ),
    }


def _similarity_one_hop(
    corpus: RepresentationCorpus,
    observation: RankedFamilyObservation,
    graph: SimilarityAdjacencyGraph,
) -> tuple[tuple[HydratedValueScore, ...], dict[str, Any]]:
    _validate_graph_binding(corpus, graph)
    values = corpus.values_by_id()
    keys = corpus.keys_by_id()
    adjacency: dict[str, list[tuple[str, float, str]]] = {value_id: [] for value_id in values}
    for edge in graph.edges:
        _validate_value_provenance(edge.left_value, values, label="similarity edge")
        _validate_value_provenance(edge.right_value, values, label="similarity edge")
        left = edge.left_value.value_id
        right = edge.right_value.value_id
        adjacency[left].append((right, edge.similarity_score, edge.edge_id))
        adjacency[right].append((left, edge.similarity_score, edge.edge_id))

    direct_rank: dict[str, int] = {}
    direct_score: dict[str, float] = {}
    direct_key: dict[str, str] = {}
    for rank, ranked in enumerate(observation.ranked_keys, start=1):
        value_id = keys[ranked.key_id].source_value_id
        direct_rank.setdefault(value_id, rank)
        direct_score.setdefault(value_id, 1.0 / (RRF_K + rank))
        direct_key.setdefault(value_id, ranked.key_id)

    supports: dict[str, list[dict[str, Any]]] = {}
    traversed: set[str] = set()
    for seed_value_id, seed_score in direct_score.items():
        supports.setdefault(seed_value_id, []).append(
            {
                "seed_value_id": seed_value_id,
                "seed_key_id": direct_key[seed_value_id],
                "seed_rank": direct_rank[seed_value_id],
                "edge_id": None,
                "edge_similarity": 1.0,
                "activation_score": seed_score,
                "hop": 0,
            }
        )
        for neighbor, edge_score, edge_id in adjacency[seed_value_id]:
            traversed.add(edge_id)
            supports.setdefault(neighbor, []).append(
                {
                    "seed_value_id": seed_value_id,
                    "seed_key_id": direct_key[seed_value_id],
                    "seed_rank": direct_rank[seed_value_id],
                    "edge_id": edge_id,
                    "edge_similarity": edge_score,
                    "activation_score": seed_score * edge_score,
                    "hop": 1,
                }
            )

    scored: list[HydratedValueScore] = []
    witness_trace: dict[str, list[dict[str, Any]]] = {}
    for value_id, raw_witnesses in supports.items():
        unique = {(item["seed_value_id"], item["edge_id"]): item for item in raw_witnesses}
        witnesses = sorted(
            unique.values(),
            key=lambda item: (
                item["hop"],
                item["seed_rank"],
                item["seed_value_id"],
                item["edge_id"] or "",
            ),
        )
        best = max(float(item["activation_score"]) for item in witnesses)
        prior = direct_rank.get(value_id)
        best_seed_rank = min(int(item["seed_rank"]) for item in witnesses)
        keys_used = tuple(dict.fromkeys(str(item["seed_key_id"]) for item in witnesses))
        contribution = FamilyValueContribution(
            family=KeyFamily.RAW,
            best_rank=best_seed_rank,
            witness_key_ids=keys_used,
            suppressed_same_family_hits=max(0, len(keys_used) - 1),
            rrf_contribution=1.0 / (RRF_K + best_seed_rank),
        )
        scored.append(
            HydratedValueScore(
                value=values[value_id],
                score=best,
                best_prior_rank=prior,
                family_contributions=(contribution,),
                graph_support_count=len({item["seed_value_id"] for item in witnesses}),
                expanded=any(item["hop"] == 1 for item in witnesses) and prior is None,
            )
        )
        witness_trace[value_id] = witnesses
    scored.sort(
        key=lambda item: (
            -item.score,
            -item.graph_support_count,
            item.best_prior_rank if item.best_prior_rank is not None else 2**63,
            item.value.value_id,
        )
    )
    return tuple(scored), {
        "seed_representation": "R0-raw-key-top-depth",
        "expansion_hops": MAX_EXPANSION_HOPS,
        "recursive_expansion": False,
        "similarity_edge_threshold": SIMILARITY_EDGE_THRESHOLD,
        "traversed_edge_ids": sorted(traversed),
        "traversed_edge_count": len(traversed),
        "hydration_witnesses": witness_trace,
        "ordering": (
            "best-(seed-RRF*edge-similarity)-desc,distinct-seed-support-desc,"
            "direct-rank-asc,canonical-value-id-asc"
        ),
    }


def _construction_accounting(
    corpus: RepresentationCorpus,
    receipts: tuple[ConstructionReceipt, ...],
    families: tuple[KeyFamily, ...],
) -> dict[str, Any]:
    active_keys = tuple(key for key in corpus.all_keys if key.family in families)
    derived = tuple(key for key in corpus.derived_keys if key.family in families)
    duplicate_text_count = len(derived) - len(
        {(key.family, key.key_text_sha256) for key in derived}
    )
    count_by_value_family: dict[tuple[str, KeyFamily], int] = {}
    for key in active_keys:
        index = (key.source_value_id, key.family)
        count_by_value_family[index] = count_by_value_family.get(index, 0) + 1
    per_value: list[dict[str, Any]] = []
    for value in corpus.values:
        counts = {
            family.value: count_by_value_family.get((value.value_id, family), 0)
            for family in families
        }
        per_value.append({"value_id": value.value_id, "key_counts": counts})
    extractors = {
        receipt.extractor.identity_sha256: receipt.extractor.content_free_binding()
        for receipt in receipts
    }
    return {
        "canonical_value_count": len(corpus.values),
        "canonical_value_utf8_bytes": sum(value.raw_value_utf8_bytes for value in corpus.values),
        "active_indexed_key_count": len(active_keys),
        "active_indexed_key_utf8_bytes": sum(key.key_text_utf8_bytes for key in active_keys),
        "derived_key_count": len(derived),
        "derived_key_utf8_bytes": sum(key.key_text_utf8_bytes for key in derived),
        "derived_objects_per_source": per_value,
        "derived_objects_per_source_sha256": sha256_json(per_value),
        "construction_receipt_count": len(receipts),
        "construction_receipts_sha256": sha256_json(
            [receipt.content_free_binding() for receipt in receipts]
        ),
        "extractor_identities": [extractors[key] for key in sorted(extractors)],
        "construction_artifact_sha256s": sorted(
            {receipt.construction_artifact_sha256 for receipt in receipts}
        ),
        "construction_accounting": {
            "input_tokens": sum(receipt.input_tokens for receipt in receipts),
            "output_tokens": sum(receipt.output_tokens for receipt in receipts),
            "latency_microseconds": sum(receipt.latency_microseconds for receipt in receipts),
            "cost_microusd": sum(receipt.cost_microusd for receipt in receipts),
            "retry_count": sum(receipt.retry_count for receipt in receipts),
            "cache_hits": sum(int(receipt.cache_hit) for receipt in receipts),
        },
        "duplicate_key_text": {
            "count": duplicate_text_count,
            "denominator": len(derived),
            "definition": "same-family-identical-key-text-sha256-beyond-first",
        },
        "orphan_keys": {"count": 0, "denominator": len(active_keys)},
        "update_rate": {
            "updates": 0,
            "construction_receipts": len(receipts),
            "classification": "static-representation-control-not-consolidation",
        },
        "index_token_count": None,
        "index_token_count_status": (
            "not-inferred; requires a separate exact tokenizer receipt artifact"
        ),
    }


def evaluate_representation_cell(
    corpus: RepresentationCorpus,
    *,
    cell: RepresentationCell,
    observations: tuple[RankedFamilyObservation, ...],
    adjacency: Graph | None = None,
) -> RepresentationResult:
    """Evaluate one frozen representation cell without executing any scorer/model."""

    if not isinstance(corpus, RepresentationCorpus):
        raise RepresentationError("representation input must be RepresentationCorpus")
    if not isinstance(cell, RepresentationCell):
        raise RepresentationError("cell must be a registered RepresentationCell")
    families = _validate_observations(corpus, cell=cell, observations=observations)
    receipts = _validate_construction_for_cell(corpus, families)

    graph_trace: dict[str, Any] | None = None
    if cell is RepresentationCell.ENTITY_ONE_HOP:
        if not isinstance(adjacency, EntityAdjacencyGraph):
            raise RepresentationError("R5 requires exactly one entity-adjacency graph")
        scored, expansion = _entity_one_hop(corpus, observations[0], adjacency)
        graph_trace = adjacency.content_free_binding()
        ranking_method = expansion
    elif cell is RepresentationCell.SIMILARITY_NEGATIVE:
        if not isinstance(adjacency, SimilarityAdjacencyGraph):
            raise RepresentationError("R-neg requires exactly one similarity-adjacency graph")
        scored, expansion = _similarity_one_hop(corpus, observations[0], adjacency)
        graph_trace = adjacency.content_free_binding()
        ranking_method = expansion
    elif cell is RepresentationCell.ENTITY_DIRECT:
        if adjacency is not None:
            raise RepresentationError("R4 direct entity activation cannot consume adjacency")
        scored, _, _, expansion = _entity_direct_scores(corpus, observations[0])
        ranking_method = {
            **expansion,
            "expansion_hops": 0,
            "adjacency_consulted": False,
            "ordering": "within-family-deduplicated-RRF-then-canonical-value-id",
        }
    else:
        if adjacency is not None:
            raise RepresentationError(f"{cell.value} cannot consume an adjacency graph")
        scored, ranking_method = _general_rrf(corpus, observations)

    pre_cap = len(scored)
    selected_scores = scored[:MAX_HYDRATED_VALUES]
    hydrated = tuple(item.value for item in selected_scores)
    value_score_bindings = [
        item.content_free_binding(rank=rank) for rank, item in enumerate(selected_scores, start=1)
    ]
    observation_bindings = [item.content_free_binding() for item in observations]
    accounting = _construction_accounting(corpus, receipts, families)
    trace = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cell": cell.value,
        "classification": "benchmark-only-source-preserving-representation-control",
        "production_configuration": False,
        "paper_reproduction": False,
        "sb_hypothesis": "SB-HMR-v1",
        "frozen_protocol": {
            "key_families": [family.value for family in families],
            "equal_family_weight": 1.0,
            "rrf_k": RRF_K,
            "family_key_depth": KEY_FAMILY_DEPTH,
            "within_family_value_dedup": "best-key-rank-only",
            "hydrated_value_cap": MAX_HYDRATED_VALUES,
            "value_granularity": "immutable-F0-turn-projection",
            "expansion_hops": (
                MAX_EXPANSION_HOPS
                if cell
                in {
                    RepresentationCell.ENTITY_ONE_HOP,
                    RepresentationCell.SIMILARITY_NEGATIVE,
                }
                else 0
            ),
            "max_neighbors_per_node": MAX_NEIGHBORS_PER_NODE,
            "all_uncited_mechanics_are_sb_hmr_v1_hypotheses": True,
        },
        "promotion": {
            "cell_intrinsically_ineligible": cell is RepresentationCell.SIMILARITY_NEGATIVE,
            "reason": (
                "registered negative control"
                if cell is RepresentationCell.SIMILARITY_NEGATIVE
                else "quality eligibility requires downstream held-out paired evidence"
            ),
        },
        "corpus": {
            "question_id": corpus.question_id,
            "source_artifact_sha256": corpus.source_artifact_sha256,
            "projection_sha256": corpus.projection_sha256,
            "index_sha256": corpus.index_sha256,
            "navigation_index_sha256": corpus.navigation_index_sha256,
            "navigation_index_classification": "source-only-navigation-index",
            "canonical_value_count": len(corpus.values),
            "canonical_value_order_sha256": sha256_json(
                [value.content_free_binding() for value in corpus.values]
            ),
            "complete_question_local_corpus_precedes_retrieval": True,
        },
        "observations": observation_bindings,
        "observations_sha256": sha256_json(observation_bindings),
        "ranking": ranking_method,
        "graph": graph_trace,
        "value_scores": value_score_bindings,
        "value_scores_sha256": sha256_json(value_score_bindings),
        "key_level_returned_count": sum(len(item.ranked_keys) for item in observations),
        "hydrated_value_pre_cap_count": pre_cap,
        "hydrated_value_cap": MAX_HYDRATED_VALUES,
        "hydrated_value_ids": [value.value_id for value in hydrated],
        "hydrated_raw_value_hashes": [value.raw_value_sha256 for value in hydrated],
        "hydrated_value_count": len(hydrated),
        "hydration": {
            "reader_evidence": "canonical-raw-value",
            "derived_keys_delivered_to_reader": False,
            "source_values_byte_identical": True,
        },
        "construction_and_index_accounting": accounting,
        "construction_input_contract": {
            "gold_question_type_answer_or_judge_fields_allowed": False,
            "question_text_allowed_for_key_or_graph_construction": False,
            "request_material_digests_recomputed": True,
            "external_execution_identity_attested_not_verified": True,
        },
        "claims": {
            "question_query_consumed_by_ranking": True,
            "question_id_is_local_routing_metadata_only": True,
            "executes_extractor_scorer_model_or_network": False,
            "external_identities_verified_by_this_module": False,
            "quality_improvement": False,
            "serving_change": False,
        },
    }
    return RepresentationResult(cell=cell, hydrated_values=hydrated, trace=trace)


__all__ = [
    "FamilyValueContribution",
    "HydratedValueScore",
    "RepresentationResult",
    "evaluate_representation_cell",
]
