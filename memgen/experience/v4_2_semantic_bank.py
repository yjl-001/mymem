"""Contracts for the paid, candidate-level V4.2 semantic bank stage.

The API-free V4.2 shortlist is only a discovery and compression artifact.  A
candidate enters the tensor-free bank only after one teacher response judges
every supplied construction example and finds at least five factually valid,
process-coherent examples.  The same response may synthesize target/reference
cards; an independent batched review remains mandatory.

This module contains schemas and deterministic validation only.  It never owns
credentials or network clients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_bank import (
    V4_BANK_RECORD_SCHEMA,
    V4_LAYER_NUMBER,
    V4_MIN_CONSTRUCTION_EXAMPLES,
    V4_RELATIVE_PHASE_DELTA,
    V4_TEACHER_MODEL,
    V4_TEACHER_TEMPERATURE,
    V4_TEACHER_THINKING,
    V4CardReview,
    V4ProcessCard,
    V4RepairSignature,
    parse_v4_card_review,
    parse_v4_process_card,
    validate_v4_card_text,
)
from memgen.experience.v4_2_bank import V42LocalClusterCandidate


V4_2_SEMANTIC_POLICY_SCHEMA = "memgen-v4.2-semantic-policy-v1"
V4_2_SEMANTIC_PROFILE_SCHEMA = "memgen-v4.2-semantic-construction-profile-v1"
V4_2_EVIDENCE_PACKET_SCHEMA = "memgen-v4.2-semantic-evidence-packet-v1"
V4_2_COMBINED_BATCH_SCHEMA = "memgen-v4.2-combined-audit-synthesis-batch-v1"
V4_2_COMBINED_RECORD_SCHEMA = "memgen-v4.2-combined-audit-synthesis-record-v1"
V4_2_REVIEW_BATCH_SCHEMA = "memgen-v4.2-card-review-batch-v1"
V4_2_REVIEW_RECORD_SCHEMA = "memgen-v4.2-card-review-record-v1"
V4_2_PAID_PLAN_SCHEMA = "memgen-v4.2-paid-stage-plan-v1"
V4_2_PAID_PREFLIGHT_SCHEMA = "memgen-v4.2-paid-stage-preflight-v1"
V4_2_BANK_MANIFEST_SCHEMA = "memgen-v4.2-bank-manifest-v1"

V4_2_COMBINED_PROMPT_VERSION = (
    "memgen-v4.2-combined-semantic-audit-synthesis-deepseek-v1"
)
V4_2_REVIEW_PROMPT_VERSION = "memgen-v4.2-batched-card-review-deepseek-v1"

V4_2_DEFAULT_MAX_EVIDENCE = 8
V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE = 4
V4_2_DEFAULT_REVIEW_BATCH_SIZE = 8
V4_2_DEFAULT_MAX_REQUEST_CHARACTERS = 200_000
V4_2_DEFAULT_RUNTIME_BANK_CAP = 32


def _nonempty(owner: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string")
    return value.strip()


def _boolean(owner: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{owner} must be boolean")
    return value


@dataclass(frozen=True)
class V42SemanticConstructionProfile:
    """Frozen choices for candidate-level semantic admission and synthesis."""

    teacher_model: str = V4_TEACHER_MODEL
    temperature: float = V4_TEACHER_TEMPERATURE
    thinking: str = V4_TEACHER_THINKING
    minimum_valid_distinct_support: int = V4_MIN_CONSTRUCTION_EXAMPLES
    maximum_evidence_per_candidate: int = V4_2_DEFAULT_MAX_EVIDENCE
    evidence_selection: str = "all_up_to_cap_else_five_diverse_plus_medoid_near"
    semantic_admission: str = "per_evidence_all_checks_and_minimum_support"
    synthesis_batch_size: int = V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE
    review_batch_size: int = V4_2_DEFAULT_REVIEW_BATCH_SIZE
    max_request_characters: int = V4_2_DEFAULT_MAX_REQUEST_CHARACTERS
    target_source: str = "official_solution_plus_verified_success"
    reference_source: str = "paired_verified_failure"
    target_runtime_bank_cap: int = V4_2_DEFAULT_RUNTIME_BANK_CAP
    injection_layer: int = V4_LAYER_NUMBER
    relative_phase_delta: int = V4_RELATIVE_PHASE_DELTA
    schema_version: str = V4_2_SEMANTIC_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_2_SEMANTIC_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4.2 semantic profile schema")
        if self.teacher_model != V4_TEACHER_MODEL:
            raise ValueError(f"V4.2 semantic construction requires {V4_TEACHER_MODEL}")
        if self.temperature != V4_TEACHER_TEMPERATURE:
            raise ValueError("V4.2 semantic construction temperature is frozen at zero")
        if self.thinking != V4_TEACHER_THINKING:
            raise ValueError("V4.2 semantic construction thinking must be disabled")
        for owner in (
            "minimum_valid_distinct_support",
            "maximum_evidence_per_candidate",
            "synthesis_batch_size",
            "review_batch_size",
            "max_request_characters",
            "target_runtime_bank_cap",
        ):
            value = getattr(self, owner)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"V4.2 semantic {owner} must be a positive integer")
        if self.minimum_valid_distinct_support != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 semantic admission requires five valid examples")
        if self.maximum_evidence_per_candidate != V4_2_DEFAULT_MAX_EVIDENCE:
            raise ValueError("V4.2 semantic evidence cap is frozen at eight")
        if self.evidence_selection != "all_up_to_cap_else_five_diverse_plus_medoid_near":
            raise ValueError("Unexpected V4.2 semantic evidence-selection rule")
        if self.semantic_admission != "per_evidence_all_checks_and_minimum_support":
            raise ValueError("Unexpected V4.2 semantic admission rule")
        if self.target_source != "official_solution_plus_verified_success":
            raise ValueError("Unexpected V4.2 target evidence source")
        if self.reference_source != "paired_verified_failure":
            raise ValueError("Unexpected V4.2 reference evidence source")
        if self.injection_layer != V4_LAYER_NUMBER:
            raise ValueError("V4.2 semantic bank remains frozen at layer twenty four")
        if self.relative_phase_delta != V4_RELATIVE_PHASE_DELTA:
            raise ValueError("V4.2 semantic bank keeps canonical pre-RoPE delta zero")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class V42EvidenceJudgment:
    evidence_id: str
    factually_valid: bool
    supports_shared_failure_mechanism: bool
    supports_shared_repair_operator: bool
    supports_shared_verification_operator: bool
    rationale: str
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        _nonempty("evidence_id", self.evidence_id)
        for owner in (
            "factually_valid",
            "supports_shared_failure_mechanism",
            "supports_shared_repair_operator",
            "supports_shared_verification_operator",
        ):
            _boolean(owner, getattr(self, owner))
        _nonempty("evidence rationale", self.rationale)
        if self.usable:
            if self.exclusion_reason is not None:
                raise ValueError("Usable V4.2 evidence cannot carry an exclusion reason")
        else:
            _nonempty("evidence exclusion_reason", self.exclusion_reason)

    @property
    def usable(self) -> bool:
        return all(
            (
                self.factually_valid,
                self.supports_shared_failure_mechanism,
                self.supports_shared_repair_operator,
                self.supports_shared_verification_operator,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "usable": self.usable}


@dataclass(frozen=True)
class V42CombinedSynthesis:
    candidate_id: str
    evidence_judgments: tuple[V42EvidenceJudgment, ...]
    shared_process_invariant: str | None
    shared_failure_mechanism: str | None
    shared_repair_operator: str | None
    shared_verification_operator: str | None
    valid_distinct_support: int
    coherent: bool
    rejection_reason: str | None
    card: V4ProcessCard | None

    def __post_init__(self) -> None:
        _nonempty("candidate_id", self.candidate_id)
        evidence_ids = [item.evidence_id for item in self.evidence_judgments]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("V4.2 evidence judgments contain duplicate IDs")
        if not V4_MIN_CONSTRUCTION_EXAMPLES <= len(evidence_ids) <= V4_2_DEFAULT_MAX_EVIDENCE:
            raise ValueError("V4.2 synthesis must judge five to eight evidence items")
        usable_count = sum(item.usable for item in self.evidence_judgments)
        if (
            isinstance(self.valid_distinct_support, bool)
            or not isinstance(self.valid_distinct_support, int)
            or self.valid_distinct_support < 0
        ):
            raise ValueError("V4.2 valid distinct support must be a non-negative integer")
        if self.valid_distinct_support != usable_count:
            raise ValueError("V4.2 reported valid support differs from evidence judgments")
        _boolean("coherent", self.coherent)
        shared = (
            self.shared_process_invariant,
            self.shared_failure_mechanism,
            self.shared_repair_operator,
            self.shared_verification_operator,
        )
        if self.coherent:
            if usable_count < V4_MIN_CONSTRUCTION_EXAMPLES:
                raise ValueError("Coherent V4.2 synthesis has fewer than five valid examples")
            for owner, value in zip(
                (
                    "shared_process_invariant",
                    "shared_failure_mechanism",
                    "shared_repair_operator",
                    "shared_verification_operator",
                ),
                shared,
            ):
                validate_v4_card_text(owner, value)
            if self.rejection_reason is not None or self.card is None:
                raise ValueError("Coherent V4.2 synthesis must carry a card only")
            if self.card.cluster_key != self.candidate_id:
                raise ValueError("V4.2 synthesis card identity mismatch")
        else:
            _nonempty("V4.2 semantic rejection_reason", self.rejection_reason)
            if self.card is not None:
                raise ValueError("Rejected V4.2 synthesis cannot carry a card")
            for owner, value in zip(
                (
                    "shared_process_invariant",
                    "shared_failure_mechanism",
                    "shared_repair_operator",
                    "shared_verification_operator",
                ),
                shared,
            ):
                if value is not None:
                    validate_v4_card_text(owner, value)

    @property
    def valid_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence_judgments if item.usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_judgments": [item.to_dict() for item in self.evidence_judgments],
            "shared_process_invariant": self.shared_process_invariant,
            "shared_failure_mechanism": self.shared_failure_mechanism,
            "shared_repair_operator": self.shared_repair_operator,
            "shared_verification_operator": self.shared_verification_operator,
            "valid_distinct_support": self.valid_distinct_support,
            "valid_evidence_ids": list(self.valid_evidence_ids),
            "coherent": self.coherent,
            "rejection_reason": self.rejection_reason,
            "card": None if self.card is None else self.card.to_dict(),
        }


def parse_v4_2_combined_synthesis(
    payload: Mapping[str, Any],
    *,
    candidate_id: str,
    expected_evidence_ids: Sequence[str],
) -> V42CombinedSynthesis:
    if str(payload.get("candidate_id", "")) != candidate_id:
        raise ValueError("V4.2 combined synthesis candidate identity mismatch")
    raw_judgments = payload.get("evidence_judgments")
    if not isinstance(raw_judgments, list):
        raise ValueError("V4.2 combined synthesis is missing evidence judgments")
    judgments = tuple(
        V42EvidenceJudgment(
            evidence_id=item.get("evidence_id"),
            factually_valid=item.get("factually_valid"),
            supports_shared_failure_mechanism=item.get(
                "supports_shared_failure_mechanism"
            ),
            supports_shared_repair_operator=item.get(
                "supports_shared_repair_operator"
            ),
            supports_shared_verification_operator=item.get(
                "supports_shared_verification_operator"
            ),
            rationale=item.get("rationale"),
            exclusion_reason=item.get("exclusion_reason"),
        )
        for item in raw_judgments
        if isinstance(item, Mapping)
    )
    if len(judgments) != len(raw_judgments):
        raise ValueError("V4.2 evidence judgment must be an object")
    if tuple(item.evidence_id for item in judgments) != tuple(expected_evidence_ids):
        raise ValueError("V4.2 combined synthesis evidence order or coverage mismatch")
    coherent = payload.get("coherent")
    card_payload = payload.get("card")
    card: V4ProcessCard | None = None
    if coherent is True:
        if not isinstance(card_payload, Mapping):
            raise ValueError("Coherent V4.2 synthesis is missing a process card")
        if card_payload.get("cluster_key") not in {None, candidate_id}:
            raise ValueError("V4.2 synthesis card candidate identity mismatch")
        card = parse_v4_process_card(card_payload, cluster_key=candidate_id)
    elif card_payload is not None:
        raise ValueError("Rejected V4.2 synthesis returned a process card")
    return V42CombinedSynthesis(
        candidate_id=candidate_id,
        evidence_judgments=judgments,
        shared_process_invariant=payload.get("shared_process_invariant"),
        shared_failure_mechanism=payload.get("shared_failure_mechanism"),
        shared_repair_operator=payload.get("shared_repair_operator"),
        shared_verification_operator=payload.get("shared_verification_operator"),
        valid_distinct_support=payload.get("valid_distinct_support"),
        coherent=coherent,
        rejection_reason=payload.get("rejection_reason"),
        card=card,
    )


def parse_v4_2_combined_batch(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Sequence[str]],
) -> tuple[V42CombinedSynthesis, ...]:
    if payload.get("schema_version") != V4_2_COMBINED_BATCH_SCHEMA:
        raise ValueError("Unexpected V4.2 combined synthesis batch schema")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("V4.2 combined synthesis batch is missing results")
    by_id: dict[str, Mapping[str, Any]] = {}
    observed_order: list[str] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            raise ValueError("V4.2 combined synthesis result must be an object")
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in expected or candidate_id in by_id:
            raise ValueError("V4.2 combined synthesis has unknown or duplicate candidate")
        by_id[candidate_id] = item
        observed_order.append(candidate_id)
    if tuple(observed_order) != tuple(expected):
        raise ValueError("V4.2 combined synthesis batch candidate order or coverage mismatch")
    return tuple(
        parse_v4_2_combined_synthesis(
            by_id[candidate_id],
            candidate_id=candidate_id,
            expected_evidence_ids=expected[candidate_id],
        )
        for candidate_id in expected
    )


def parse_v4_2_review_batch(
    payload: Mapping[str, Any],
    *,
    expected_candidate_ids: Sequence[str],
) -> tuple[V4CardReview, ...]:
    if payload.get("schema_version") != V4_2_REVIEW_BATCH_SCHEMA:
        raise ValueError("Unexpected V4.2 card-review batch schema")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("V4.2 card-review batch is missing results")
    by_id: dict[str, Mapping[str, Any]] = {}
    expected = set(expected_candidate_ids)
    observed_order: list[str] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            raise ValueError("V4.2 card-review result must be an object")
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in expected or candidate_id in by_id:
            raise ValueError("V4.2 card-review has unknown or duplicate candidate")
        by_id[candidate_id] = item
        observed_order.append(candidate_id)
    if tuple(observed_order) != tuple(expected_candidate_ids):
        raise ValueError("V4.2 card-review batch candidate order or coverage mismatch")
    return tuple(
        parse_v4_card_review(by_id[candidate_id], cluster_key=candidate_id)
        for candidate_id in expected_candidate_ids
    )


def build_v4_2_semantic_bank_record(
    *,
    candidate: V42LocalClusterCandidate,
    synthesis: V42CombinedSynthesis,
    review: V4CardReview,
    evidence_packet: Mapping[str, Any],
    signatures: Mapping[str, V4RepairSignature],
    profile: V42SemanticConstructionProfile,
    source_shortlist: Mapping[str, Any],
    semantic_policy_sha256: str,
) -> dict[str, Any]:
    if not synthesis.coherent or synthesis.card is None or not review.approve:
        raise ValueError("Only coherent, independently approved V4.2 cards enter the bank")
    if synthesis.candidate_id != candidate.candidate_id:
        raise ValueError("V4.2 candidate/synthesis identity mismatch")
    if review.cluster_key != candidate.candidate_id:
        raise ValueError("V4.2 candidate/review identity mismatch")
    evidence_by_id = {
        str(item["evidence_id"]): item for item in evidence_packet.get("evidence", ())
    }
    valid_ids = synthesis.valid_evidence_ids
    if set(valid_ids) - set(evidence_by_id) or len(valid_ids) < V4_MIN_CONSTRUCTION_EXAMPLES:
        raise ValueError("V4.2 semantic bank lost valid evidence support")
    if set(valid_ids) - set(signatures):
        raise ValueError("V4.2 semantic bank lost source signatures")
    for experience_id in valid_ids:
        if str(evidence_by_id[experience_id]["sample_id"]) != signatures[
            experience_id
        ].sample_id:
            raise ValueError("V4.2 semantic bank evidence/signature sample mismatch")
    sample_ids = sorted(str(evidence_by_id[item]["sample_id"]) for item in valid_ids)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("V4.2 semantic bank valid evidence repeats a sample")
    cluster = {
        "cluster_key": candidate.candidate_id,
        "title": synthesis.shared_process_invariant,
        "experience_type": "mixed_source_provenance",
        "failure_mechanism": synthesis.shared_failure_mechanism,
        "repair_operator": synthesis.shared_repair_operator,
        "scope_summary": synthesis.card.target.scope,
        "member_experience_ids": list(valid_ids),
        "representative_experience_ids": list(valid_ids),
        "source_candidate_id": candidate.candidate_id,
        "source_membership_sha256": candidate.membership_sha256,
        "source_experience_type_distribution": dict(
            candidate.source_experience_type_distribution
        ),
    }
    semantic_identity = {
        "cluster": cluster,
        "card_sha256": synthesis.card.card_sha256,
        "semantic_profile_sha256": profile.profile_sha256,
        "semantic_policy_sha256": semantic_policy_sha256,
    }
    bank_id = f"v42-bank-{canonical_json_sha256(semantic_identity)[:20]}"
    record = {
        "schema_version": V4_BANK_RECORD_SCHEMA,
        "construction_version": "v4.2",
        "bank_id": bank_id,
        "benchmark": "openai/gsm8k",
        "cluster": cluster,
        "process_card": synthesis.card.to_dict(),
        "semantic_audit": synthesis.to_dict(),
        "review": review.to_dict(),
        "construction": {
            "experience_ids": list(valid_ids),
            "representative_experience_ids": list(valid_ids),
            "sample_ids": sample_ids,
            "distinct_sample_count": len(sample_ids),
            "candidate_member_experience_ids": list(
                candidate.member_experience_ids
            ),
            "evidence_packet_sha256": evidence_packet["packet_sha256"],
            "signature_sha256": {
                item: signatures[item].signature_sha256 for item in valid_ids
            },
        },
        "roles": {
            "target_online_injectable": True,
            "reference_online_injectable": False,
            "auxiliary": None,
        },
        "compiler_contract": {
            "layer_number": V4_LAYER_NUMBER,
            "all_kv_groups": True,
            "canonical_pre_rope": True,
            "relative_phase_delta": V4_RELATIVE_PHASE_DELTA,
            "attention_backend": "sdpa",
        },
        "source_shortlist": dict(source_shortlist),
        "semantic_policy_sha256": semantic_policy_sha256,
        "construction_profile": profile.to_dict(),
        "construction_profile_sha256": profile.profile_sha256,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def build_v4_2_semantic_bank_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    profile: V42SemanticConstructionProfile,
    inputs: Mapping[str, Any],
    teacher: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError("V4.2 semantic bank requires at least one approved record")
    if len(records) > profile.target_runtime_bank_cap:
        raise ValueError("V4.2 semantic bank exceeds the runtime cap")
    bank_ids = [str(item.get("bank_id", "")) for item in records]
    if any(not item for item in bank_ids) or len(set(bank_ids)) != len(bank_ids):
        raise ValueError("V4.2 semantic bank IDs are missing or duplicated")
    for record in records:
        logical = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("schema_version") != V4_BANK_RECORD_SCHEMA
            or record.get("construction_profile_sha256") != profile.profile_sha256
            or record.get("record_sha256") != canonical_json_sha256(logical)
        ):
            raise ValueError("V4.2 semantic bank received an invalid record")
    manifest = {
        "schema_version": V4_2_BANK_MANIFEST_SCHEMA,
        "construction_version": "v4.2",
        "status": "constructed_not_tensor_compiled",
        "qualified_for_online_use": False,
        "benchmark": "openai/gsm8k",
        "record_count": len(records),
        "bank_ids": bank_ids,
        "record_order_sha256": canonical_json_sha256(bank_ids),
        "record_sha256": {
            str(item["bank_id"]): str(item["record_sha256"]) for item in records
        },
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "inputs": dict(inputs),
        "teacher": dict(teacher),
        "archive": dict(archive),
        "auxiliary_banks_materialized": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


__all__ = [
    "V4_2_BANK_MANIFEST_SCHEMA",
    "V4_2_COMBINED_BATCH_SCHEMA",
    "V4_2_COMBINED_PROMPT_VERSION",
    "V4_2_COMBINED_RECORD_SCHEMA",
    "V4_2_DEFAULT_MAX_EVIDENCE",
    "V4_2_DEFAULT_MAX_REQUEST_CHARACTERS",
    "V4_2_DEFAULT_REVIEW_BATCH_SIZE",
    "V4_2_DEFAULT_RUNTIME_BANK_CAP",
    "V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE",
    "V4_2_EVIDENCE_PACKET_SCHEMA",
    "V4_2_PAID_PLAN_SCHEMA",
    "V4_2_PAID_PREFLIGHT_SCHEMA",
    "V4_2_REVIEW_BATCH_SCHEMA",
    "V4_2_REVIEW_PROMPT_VERSION",
    "V4_2_REVIEW_RECORD_SCHEMA",
    "V4_2_SEMANTIC_POLICY_SCHEMA",
    "V4_2_SEMANTIC_PROFILE_SCHEMA",
    "V42CombinedSynthesis",
    "V42EvidenceJudgment",
    "V42SemanticConstructionProfile",
    "build_v4_2_semantic_bank_manifest",
    "build_v4_2_semantic_bank_record",
    "parse_v4_2_combined_batch",
    "parse_v4_2_combined_synthesis",
    "parse_v4_2_review_batch",
]
