from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_official_preflight import (
    OFFICIAL_DATASET_REQUIREMENT,
    OFFICIAL_QUESTION_COUNT,
    DatasetRequirement,
    ExactTokenizerPin,
    LongMemEvalOfficialPreflightError,
    PinnedPromptTokenizerAdapter,
    freeze_official_preflight,
    freeze_pinned_preflight,
    load_preflight_manifest_artifact,
    load_preflight_manifest_bytes,
    validate_official_prepared_run,
    validate_prepared_run,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    PRIMARY_TOKEN_BUDGET,
    TOKENIZER_PROTOCOL,
    OrderedTurnBlocks,
    TurnPromptPackingResult,
    pack_turn_prompt,
)
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes
from scripts._longmemeval_tokenizer import TokenizerObservation

CURRENT_DATE = "2025/01/04 (Sat) 11:00"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _record(question_id: str, question: str, contents: list[str]) -> dict[str, Any]:
    session_id = f"session-{question_id}"
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": question,
        "answer": f"PRIVATE-GOLD-{question_id}",
        "question_date": CURRENT_DATE,
        "haystack_session_ids": [session_id],
        "haystack_dates": ["2025/01/03 (Fri) 09:07"],
        "haystack_sessions": [
            [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": content,
                }
                for index, content in enumerate(contents)
            ]
        ],
        "answer_session_ids": [session_id],
    }


class FixtureJsonlBoundary:
    """Synthetic JSONL-boundary shape; no tokenizer process or external call."""

    def __init__(self, pin: ExactTokenizerPin, *, stale_text_digest: bool = False) -> None:
        self.pin = pin
        self.requests: list[str] = []
        self._stale_text_digest = stale_text_digest

    def count(self, prompt: str) -> TokenizerObservation:
        self.requests.append(prompt)
        request_id = len(self.requests)
        encoded = prompt.encode("utf-8")
        return TokenizerObservation(
            request_id=request_id,
            provider_request_id=f"fixture-provider-{request_id}",
            response_identity_sha256=self.pin.identity.identity_sha256,
            text_sha256=(
                "f" * 64 if self._stale_text_digest else hashlib.sha256(encoded).hexdigest()
            ),
            token_count=len(encoded),
        )

    @property
    def evidence(self) -> dict[str, Any]:
        return {
            "method": "exact_serialized_reader_prompt",
            "provider": "JsonlExactTokenizer",
            "exact_model_tokenizer": True,
            "tokenizer_model": self.pin.model,
            "tokenizer_revision": self.pin.revision,
            "tokenizer_artifact": {
                "path": "benchmarks/evidence/tokenizer/tokenizer.json",
                "bytes": 1234,
                "sha256": self.pin.artifact_sha256,
            },
            "tokenizer_executable": {
                "path": "benchmarks/evidence/tokenizer/provider",
                "bytes": 4321,
                "sha256": self.pin.executable_sha256,
            },
            "protocol": TOKENIZER_PROTOCOL,
            "response_identity_sha256": self.pin.identity.identity_sha256,
            "observation_accounting": {
                "source": "provider-observed",
                "requests": len(self.requests),
                "responses": len(self.requests),
                "unique_provider_request_ids": len(self.requests),
                "text_characters": sum(len(prompt) for prompt in self.requests),
                "text_utf8_bytes": sum(len(prompt.encode("utf-8")) for prompt in self.requests),
                "exact_response_identity_verified": True,
            },
        }


def _fixture() -> tuple[
    bytes,
    Any,
    tuple[TurnPromptPackingResult, ...],
    dict[str, Any],
]:
    records = [
        _record("q-one", "FIRST PRIVATE QUESTION", ["one-a", "one-b"]),
        _record("q-two", "SECOND PRIVATE QUESTION", ["two-a", "two-b"]),
    ]
    raw = (
        json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    dataset = DatasetRequirement(
        name="Synthetic LongMemEval rehearsal",
        source_label="synthetic-longmemeval.json",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        question_count=len(records),
        official=False,
    )
    pin = ExactTokenizerPin(
        model="fixture/exact-byte-tokenizer",
        revision="fixture-immutable-revision",
        artifact_sha256="a" * 64,
        executable_sha256="b" * 64,
    )
    manifest = freeze_pinned_preflight(raw, dataset=dataset, tokenizer=pin)
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=dataset.source_sha256,
        source_label=dataset.source_label,
    )
    tokenizer = PinnedPromptTokenizerAdapter(FixtureJsonlBoundary(pin), pin=pin)
    by_question = {
        question.question_id: tuple(
            turn for turn in corpus.turns if turn.turn_id.question_id == question.question_id
        )
        for question in corpus.questions
    }
    first = pack_turn_prompt(
        OrderedTurnBlocks.linear(by_question["q-one"]),
        question_id="q-one",
        question=records[0]["question"],
        current_date=records[0]["question_date"],
        token_budget=PRIMARY_TOKEN_BUDGET,
        tokenizer=tokenizer,
    )
    q_two = by_question["q-two"]
    second = pack_turn_prompt(
        OrderedTurnBlocks.chain_blocks(((q_two[0],), (q_two[1],))),
        question_id="q-two",
        question=records[1]["question"],
        current_date=records[1]["question_date"],
        token_budget=PRIMARY_TOKEN_BUDGET,
        tokenizer=tokenizer,
    )
    return raw, manifest, (first, second), tokenizer.evidence


def _mutated_result(
    result: TurnPromptPackingResult,
    mutation: Any,
    *,
    prompt: str | None = None,
) -> TurnPromptPackingResult:
    trace = copy.deepcopy(result.trace)
    mutation(trace)
    return TurnPromptPackingResult(prompt=result.prompt if prompt is None else prompt, trace=trace)


def test_preflight_freezes_source_prompt_budget_tokenizer_and_complete_cases() -> None:
    raw, manifest, packed, evidence = _fixture()

    receipt = validate_prepared_run(
        raw,
        manifest=manifest,
        packed_cases=packed,
        tokenizer_evidence=evidence,
    )

    artifact = manifest.content_free_artifact()
    assert artifact["classification"] == "nonofficial-pinned-reader-preflight"
    assert (
        artifact["prompt"]["template_sha256"] == packed[0].trace["reader_prompt"]["template_sha256"]
    )
    assert artifact["prompt"]["token_budget"] == 8192
    assert artifact["dataset"]["question_count"] == 2
    assert [case["question_id"] for case in artifact["cases"]] == ["q-one", "q-two"]
    assert artifact["tokenizer"]["executable_sha256"] == "b" * 64
    assert artifact["ready_for_external_calls"] is False
    assert receipt.packed_case_count == 2
    assert receipt.exact_count_observation_count == 12
    assert receipt.content_free_artifact()["ready_for_reader_calls"] is True

    serialized = json.dumps(
        {
            "manifest": manifest.content_free_artifact(),
            "receipt": receipt.content_free_artifact(),
        },
        ensure_ascii=False,
    )
    for secret in (
        "FIRST PRIVATE QUESTION",
        "SECOND PRIVATE QUESTION",
        "PRIVATE-GOLD-q-one",
        "one-a",
        "two-b",
    ):
        assert secret not in serialized


def test_official_mode_is_unweakenably_pinned_to_digest_and_500_cases() -> None:
    raw, manifest, packed, evidence = _fixture()

    assert OFFICIAL_DATASET_REQUIREMENT.question_count == OFFICIAL_QUESTION_COUNT == 500
    assert OFFICIAL_DATASET_REQUIREMENT.official is True
    with pytest.raises(LongMemEvalOfficialPreflightError, match="official mode requires"):
        DatasetRequirement(
            name="LongMemEval-S",
            source_label="longmemeval_s_cleaned.json",
            source_sha256="0" * 64,
            question_count=499,
            official=True,
        )
    with pytest.raises(LongMemEvalOfficialPreflightError, match="source corpus failed"):
        freeze_official_preflight(raw, tokenizer=manifest.tokenizer)
    with pytest.raises(LongMemEvalOfficialPreflightError, match="official full-500"):
        validate_official_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=packed,
            tokenizer_evidence=evidence,
        )


@pytest.mark.parametrize("coverage", ["missing", "duplicate", "reordered"])
def test_missing_duplicate_and_reordered_case_coverage_fails_closed(coverage: str) -> None:
    raw, manifest, packed, evidence = _fixture()
    changed = {
        "missing": packed[:1],
        "duplicate": (packed[0], packed[0]),
        "reordered": tuple(reversed(packed)),
    }[coverage]

    with pytest.raises(
        LongMemEvalOfficialPreflightError,
        match="coverage|question/date|request IDs",
    ):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )


def test_manifest_cannot_be_replayed_against_changed_corpus_bytes() -> None:
    raw, manifest, packed, evidence = _fixture()
    changed = raw[:-1] + b" \n"

    with pytest.raises(LongMemEvalOfficialPreflightError, match="source corpus failed"):
        validate_prepared_run(
            changed,
            manifest=manifest,
            packed_cases=packed,
            tokenizer_evidence=evidence,
        )


@pytest.mark.parametrize(
    "field",
    ["template", "budget", "tokenizer", "extra", "boolean-integer-alias"],
)
def test_prompt_template_budget_tokenizer_or_schema_drift_fails_closed(field: str) -> None:
    raw, manifest, packed, evidence = _fixture()

    def mutate(trace: dict[str, Any]) -> None:
        if field == "template":
            trace["reader_prompt"]["template_sha256"] = "f" * 64
        elif field == "budget":
            trace["budget"]["token_budget"] = 4096
            trace["budget"]["is_primary"] = False
        elif field == "tokenizer":
            trace["tokenizer"]["model"] = "wrong/tokenizer"
        elif field == "boolean-integer-alias":
            trace["claims"]["token_counts_are_estimates"] = 0
        else:
            trace["unexpected_field"] = "covert-drift"

    changed = (_mutated_result(packed[0], mutate), packed[1])
    with pytest.raises(
        LongMemEvalOfficialPreflightError,
        match="template|8,192|tokenizer|schema|claims",
    ):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )


def test_exact_official_prompt_is_reconstructed_not_just_self_hashed() -> None:
    raw, manifest, packed, evidence = _fixture()
    forged_prompt = packed[0].prompt + "\nFORGED READER MATERIAL"

    def mutate(trace: dict[str, Any]) -> None:
        encoded = forged_prompt.encode("utf-8")
        final = trace["final_prompt"]
        final["sha256"] = hashlib.sha256(encoded).hexdigest()
        final["utf8_bytes"] = len(encoded)
        observations = trace["exact_count_observations"]
        receipt = observations[-1]["receipt"]
        receipt["prompt_sha256"] = final["sha256"]
        receipt["prompt_utf8_bytes"] = final["utf8_bytes"]
        trace["exact_count_observations_sha256"] = _sha256_json(observations)
        trace["observation_accounting"]["unique_prompt_digests"] = len(
            {item["receipt"]["prompt_sha256"] for item in observations}
        )

    changed = (_mutated_result(packed[0], mutate, prompt=forged_prompt), packed[1])
    with pytest.raises(LongMemEvalOfficialPreflightError, match="exact official template"):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )


def test_receipt_acceptance_is_recomputed() -> None:
    raw, manifest, packed, evidence = _fixture()

    def flip_acceptance(trace: dict[str, Any]) -> None:
        trace["decisions"][0]["accepted"] = False
        trace["decisions_sha256"] = _sha256_json(trace["decisions"])

    changed = (_mutated_result(packed[0], flip_acceptance), packed[1])
    with pytest.raises(LongMemEvalOfficialPreflightError, match="acceptance"):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )


def test_every_intermediate_receipt_and_observation_must_cover_the_complete_prompt() -> None:
    raw, manifest, packed, evidence = _fixture()

    def forge_candidate_alone(trace: dict[str, Any]) -> None:
        observations = trace["exact_count_observations"]
        observations[1]["receipt"]["prompt_sha256"] = "f" * 64
        trace["exact_count_observations_sha256"] = _sha256_json(observations)
        trace["observation_accounting"]["unique_prompt_digests"] = len(
            {item["receipt"]["prompt_sha256"] for item in observations}
        )

    changed = (_mutated_result(packed[0], forge_candidate_alone), packed[1])
    with pytest.raises(LongMemEvalOfficialPreflightError, match="candidate-alone exact receipt"):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )

    def remove_observation(trace: dict[str, Any]) -> None:
        trace["exact_count_observations"].pop(-2)
        trace["exact_count_observations_sha256"] = _sha256_json(trace["exact_count_observations"])

    changed = (_mutated_result(packed[0], remove_observation), packed[1])
    with pytest.raises(LongMemEvalOfficialPreflightError, match="observation coverage"):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )


def test_request_and_provider_request_ids_are_globally_unique() -> None:
    raw, manifest, packed, evidence = _fixture()
    first_provider_id = packed[0].trace["exact_count_observations"][0]["receipt"][
        "provider_request_id"
    ]

    def reuse_provider(trace: dict[str, Any]) -> None:
        observations = trace["exact_count_observations"]
        observations[0]["receipt"]["provider_request_id"] = first_provider_id
        trace["exact_count_observations_sha256"] = _sha256_json(observations)

    changed = (packed[0], _mutated_result(packed[1], reuse_provider))
    with pytest.raises(LongMemEvalOfficialPreflightError, match="globally unique"):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )

    first_last_request = packed[0].trace["exact_count_observations"][-1]["receipt"]["request_id"]

    def reuse_request(trace: dict[str, Any]) -> None:
        observations = trace["exact_count_observations"]
        observations[0]["receipt"]["request_id"] = first_last_request
        trace["exact_count_observations_sha256"] = _sha256_json(observations)

    changed = (packed[0], _mutated_result(packed[1], reuse_request))
    with pytest.raises(LongMemEvalOfficialPreflightError, match="globally fresh"):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=changed,
            tokenizer_evidence=evidence,
        )


@pytest.mark.parametrize("tamper", ["artifact", "executable", "requests", "bytes", "method"])
def test_tokenizer_runtime_evidence_must_reconcile_every_observation(tamper: str) -> None:
    raw, manifest, packed, evidence = _fixture()
    changed = copy.deepcopy(evidence)
    if tamper == "artifact":
        changed["tokenizer_artifact"]["sha256"] = "c" * 64
    elif tamper == "executable":
        changed["tokenizer_executable"]["sha256"] = "d" * 64
    elif tamper == "requests":
        changed["observation_accounting"]["requests"] -= 1
    elif tamper == "bytes":
        changed["observation_accounting"]["text_utf8_bytes"] -= 1
    else:
        changed["method"] = "chars-divided-by-four-estimator"

    with pytest.raises(
        LongMemEvalOfficialPreflightError,
        match="SHA-256|requests|bytes|method",
    ):
        validate_prepared_run(
            raw,
            manifest=manifest,
            packed_cases=packed,
            tokenizer_evidence=changed,
        )


def test_manifest_and_tokenizer_pin_are_immutable_and_digest_tamper_evident() -> None:
    _, manifest, _, _ = _fixture()
    with pytest.raises(LongMemEvalOfficialPreflightError, match="question-ID digest"):
        replace(manifest, question_ids_sha256="f" * 64)
    with pytest.raises(LongMemEvalOfficialPreflightError, match="tokenizer protocol"):
        replace(manifest.tokenizer, protocol="approximate-tokenizer-v0")


def test_persisted_manifest_round_trip_rejects_constant_digest_and_json_tamper() -> None:
    _, manifest, _, _ = _fixture()
    artifact = manifest.content_free_artifact()
    raw = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert load_preflight_manifest_artifact(artifact) == manifest
    assert load_preflight_manifest_bytes(raw) == manifest

    wrong_budget = copy.deepcopy(artifact)
    wrong_budget["prompt"]["token_budget"] = 4096
    with pytest.raises(LongMemEvalOfficialPreflightError, match="exact schema"):
        load_preflight_manifest_artifact(wrong_budget)

    wrong_digest = copy.deepcopy(artifact)
    wrong_digest["manifest_sha256"] = "f" * 64
    with pytest.raises(LongMemEvalOfficialPreflightError, match="manifest digest"):
        load_preflight_manifest_artifact(wrong_digest)

    bool_alias = copy.deepcopy(artifact)
    bool_alias["ready_for_external_calls"] = 0
    with pytest.raises(LongMemEvalOfficialPreflightError, match="exact schema"):
        load_preflight_manifest_artifact(bool_alias)

    duplicate = b'{"artifact_type":"duplicate",' + raw[1:]
    with pytest.raises(LongMemEvalOfficialPreflightError, match="repeats JSON field"):
        load_preflight_manifest_bytes(duplicate)


def test_prompt_tokenizer_adapter_rejects_preused_and_stale_boundaries() -> None:
    _, manifest, _, _ = _fixture()
    preused = FixtureJsonlBoundary(manifest.tokenizer)
    preused.count("unbound earlier prompt")
    with pytest.raises(LongMemEvalOfficialPreflightError, match="fresh accounting"):
        PinnedPromptTokenizerAdapter(preused, pin=manifest.tokenizer)

    stale = PinnedPromptTokenizerAdapter(
        FixtureJsonlBoundary(manifest.tokenizer, stale_text_digest=True),
        pin=manifest.tokenizer,
    )
    with pytest.raises(LongMemEvalOfficialPreflightError, match="complete prompt"):
        stale.count_prompt("official prompt material", query_sha256="1" * 64)
