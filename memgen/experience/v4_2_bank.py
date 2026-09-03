"""Contracts for API-free MemGen V4.2 local repair clustering.

V4.2 keeps the V4 target/reference side-KV runtime contract, but removes the
per-experience canonicalizer and the all-candidate teacher pair judge from the
clustering path.  Authenticated V4 repair signatures remain the semantic source
of truth.  Three independently embedded views (failure mechanism, repair, and
applicability) form a mutual-kNN graph, and deterministic complete-link groups
become candidates only when at least five distinct construction samples agree.

No class in this module owns an API or teacher configuration.  This is
intentional: completing a V4.2 local cluster plan must not require, read, or
implicitly spend an external API credential.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_bank import (
    V4_LAYER_NUMBER,
    V4_MIN_CONSTRUCTION_EXAMPLES,
    V4_RELATIVE_PHASE_DELTA,
    V4RepairSignature,
    validate_v4_card_text,
)
from memgen.experience.v4_1_bank import (
    V4_1_EMBEDDING_MODEL,
    V4_1_EMBEDDING_REVISION,
)


V4_2_CONSTRUCTION_PROFILE_SCHEMA = "memgen-v4.2-local-construction-profile-v1"
V4_2_LOCAL_ATOM_SCHEMA = "memgen-v4.2-local-repair-atom-v1"
V4_2_LOCAL_CLUSTER_SCHEMA = "memgen-v4.2-local-cluster-candidate-v1"
V4_2_LOCAL_CLUSTER_PLAN_SCHEMA = "memgen-v4.2-local-cluster-plan-v1"
V4_2_EMBEDDING_MANIFEST_SCHEMA = "memgen-v4.2-multiview-embeddings-v1"
V4_2_POSITIVE_EDGE_SCHEMA = "memgen-v4.2-local-positive-edge-v1"
V4_2_REVIEW_PACKET_SCHEMA = "memgen-v4.2-cluster-review-packet-v1"
V4_2_PREFLIGHT_REPORT_SCHEMA = "memgen-v4.2-api-preflight-report-v1"
V4_2_SHORTLIST_PROFILE_SCHEMA = "memgen-v4.2-shortlist-profile-v1"
V4_2_SHORTLIST_MANIFEST_SCHEMA = "memgen-v4.2-synthesis-shortlist-manifest-v1"
V4_2_SHORTLIST_PREFLIGHT_SCHEMA = "memgen-v4.2-shortlist-api-preflight-v1"

V4_2_DEFAULT_NEIGHBOR_COUNT = 32
V4_2_DEFAULT_MECHANISM_THRESHOLD = 0.82
V4_2_DEFAULT_REPAIR_THRESHOLD = 0.82
V4_2_DEFAULT_APPLICABILITY_THRESHOLD = 0.70
V4_2_DEFAULT_MECHANISM_WEIGHT = 0.45
V4_2_DEFAULT_REPAIR_WEIGHT = 0.45
V4_2_DEFAULT_APPLICABILITY_WEIGHT = 0.10
V4_2_DEFAULT_REPRESENTATIVE_COUNT = V4_MIN_CONSTRUCTION_EXAMPLES
V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE = 4
V4_2_DEFAULT_MAX_API_CANDIDATES = 80
V4_2_DEFAULT_PREFERRED_SUPPORT = 6
V4_2_DEFAULT_MAX_SYNTHESIS_CANDIDATES = 48
V4_2_DEFAULT_TARGET_RUNTIME_BANK_CAP = 32
V4_2_DEFAULT_REDUNDANCY_MECHANISM_THRESHOLD = 0.92
V4_2_DEFAULT_REDUNDANCY_REPAIR_THRESHOLD = 0.92
V4_2_DEFAULT_REDUNDANCY_APPLICABILITY_THRESHOLD = 0.85
V4_2_DEFAULT_MIN_SUPPORT_COHESION_QUANTILE = 0.50
V4_2_DEFAULT_REVIEW_BATCH_SIZE = 8
V4_2_SYNTHESIS_SEMANTIC_FIELDS = (
    "problem_structure",
    "decision_point",
    "failure_mechanism",
    "repair_operator",
    "verification_operator",
)


def _identifier(owner: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty identifier")
    normalized = value.strip()
    if any(character.isspace() for character in normalized):
        raise ValueError(f"{owner} must not contain whitespace")
    return normalized


def _unit_interval(owner: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{owner} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        raise ValueError(f"{owner} must be finite and within [-1, 1]")
    return normalized


def _similarity_threshold(owner: str, value: Any) -> float:
    normalized = _unit_interval(owner, value)
    if normalized < 0.0:
        raise ValueError(f"{owner} must be within [0, 1]")
    return normalized


@dataclass(frozen=True)
class V42ConstructionProfile:
    """Authenticated parameters that define one local clustering run."""

    source_signature_schema: str = "memgen-v4-repair-signature-v1"
    grouping_rule: str = "multiview_mutual_knn_complete_link"
    source_experience_type_policy: str = "provenance_not_cluster_boundary"
    answer_serialization_policy: str = "authenticated_format_compliance_only"
    unsupported_policy: str = "archive_without_forced_assignment"
    embedding_model: str = V4_1_EMBEDDING_MODEL
    embedding_revision: str = V4_1_EMBEDDING_REVISION
    neighbor_count: int = V4_2_DEFAULT_NEIGHBOR_COUNT
    mechanism_threshold: float = V4_2_DEFAULT_MECHANISM_THRESHOLD
    repair_threshold: float = V4_2_DEFAULT_REPAIR_THRESHOLD
    applicability_threshold: float = V4_2_DEFAULT_APPLICABILITY_THRESHOLD
    mechanism_weight: float = V4_2_DEFAULT_MECHANISM_WEIGHT
    repair_weight: float = V4_2_DEFAULT_REPAIR_WEIGHT
    applicability_weight: float = V4_2_DEFAULT_APPLICABILITY_WEIGHT
    min_distinct_support: int = V4_MIN_CONSTRUCTION_EXAMPLES
    representative_count: int = V4_2_DEFAULT_REPRESENTATIVE_COUNT
    synthesis_batch_size: int = V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE
    max_api_candidates: int = V4_2_DEFAULT_MAX_API_CANDIDATES
    injection_layer: int = V4_LAYER_NUMBER
    relative_phase_delta: int = V4_RELATIVE_PHASE_DELTA
    schema_version: str = V4_2_CONSTRUCTION_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_2_CONSTRUCTION_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4.2 construction profile schema")
        if self.source_signature_schema != "memgen-v4-repair-signature-v1":
            raise ValueError("V4.2 requires authenticated V4 repair signatures")
        if self.grouping_rule != "multiview_mutual_knn_complete_link":
            raise ValueError("Unexpected V4.2 grouping rule")
        if self.source_experience_type_policy != "provenance_not_cluster_boundary":
            raise ValueError("Unexpected V4.2 experience-type policy")
        if self.answer_serialization_policy != "authenticated_format_compliance_only":
            raise ValueError("Unexpected V4.2 answer-serialization policy")
        if self.unsupported_policy != "archive_without_forced_assignment":
            raise ValueError("Unexpected V4.2 unsupported policy")
        if self.embedding_model != V4_1_EMBEDDING_MODEL:
            raise ValueError("Unexpected V4.2 embedding model")
        if self.embedding_revision != V4_1_EMBEDDING_REVISION:
            raise ValueError("V4.2 embedding revision must remain pinned")
        for owner in (
            "neighbor_count",
            "min_distinct_support",
            "representative_count",
            "synthesis_batch_size",
            "max_api_candidates",
        ):
            value = getattr(self, owner)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"V4.2 {owner} must be a positive integer")
        if self.min_distinct_support != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 minimum distinct support is frozen at five")
        if self.representative_count != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 local synthesis evidence is frozen at five samples")
        for owner in (
            "mechanism_threshold",
            "repair_threshold",
            "applicability_threshold",
        ):
            _similarity_threshold(owner, getattr(self, owner))
        weights = (
            self.mechanism_weight,
            self.repair_weight,
            self.applicability_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in weights
        ):
            raise ValueError("V4.2 view weights must be finite and positive")
        if not math.isclose(sum(float(value) for value in weights), 1.0, abs_tol=1e-12):
            raise ValueError("V4.2 view weights must sum to one")
        if self.injection_layer != V4_LAYER_NUMBER:
            raise ValueError("V4.2 remains frozen at layer 24")
        if self.relative_phase_delta != V4_RELATIVE_PHASE_DELTA:
            raise ValueError("V4.2 keeps canonical pre-RoPE delta zero")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class V42ShortlistProfile:
    """Frozen high-quality basis selection applied after local discovery."""

    source_cluster_schema: str = V4_2_LOCAL_CLUSTER_SCHEMA
    selection_rule: str = "support_cohesion_centroid_nms"
    unsupported_policy: str = "archive_without_recovery"
    minimum_distinct_support: int = V4_MIN_CONSTRUCTION_EXAMPLES
    preferred_distinct_support: int = V4_2_DEFAULT_PREFERRED_SUPPORT
    minimum_support_cohesion_quantile: float = (
        V4_2_DEFAULT_MIN_SUPPORT_COHESION_QUANTILE
    )
    redundancy_mechanism_threshold: float = (
        V4_2_DEFAULT_REDUNDANCY_MECHANISM_THRESHOLD
    )
    redundancy_repair_threshold: float = V4_2_DEFAULT_REDUNDANCY_REPAIR_THRESHOLD
    redundancy_applicability_threshold: float = (
        V4_2_DEFAULT_REDUNDANCY_APPLICABILITY_THRESHOLD
    )
    max_synthesis_candidates: int = V4_2_DEFAULT_MAX_SYNTHESIS_CANDIDATES
    target_runtime_bank_cap: int = V4_2_DEFAULT_TARGET_RUNTIME_BANK_CAP
    synthesis_batch_size: int = V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE
    review_batch_size: int = V4_2_DEFAULT_REVIEW_BATCH_SIZE
    semantic_prompt_fields: tuple[str, ...] = V4_2_SYNTHESIS_SEMANTIC_FIELDS
    embedding_model: str = V4_1_EMBEDDING_MODEL
    embedding_revision: str = V4_1_EMBEDDING_REVISION
    injection_layer: int = V4_LAYER_NUMBER
    relative_phase_delta: int = V4_RELATIVE_PHASE_DELTA
    schema_version: str = V4_2_SHORTLIST_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_2_SHORTLIST_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4.2 shortlist profile schema")
        if self.source_cluster_schema != V4_2_LOCAL_CLUSTER_SCHEMA:
            raise ValueError("V4.2 shortlist requires local cluster candidates")
        if self.selection_rule != "support_cohesion_centroid_nms":
            raise ValueError("Unexpected V4.2 shortlist selection rule")
        if self.unsupported_policy != "archive_without_recovery":
            raise ValueError("V4.2 shortlist must not recover unsupported atoms")
        for owner in (
            "minimum_distinct_support",
            "preferred_distinct_support",
            "max_synthesis_candidates",
            "target_runtime_bank_cap",
            "synthesis_batch_size",
            "review_batch_size",
        ):
            value = getattr(self, owner)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"V4.2 shortlist {owner} must be a positive integer")
        if self.minimum_distinct_support != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 shortlist minimum support remains frozen at five")
        if self.preferred_distinct_support <= self.minimum_distinct_support:
            raise ValueError("V4.2 preferred support must exceed minimum support")
        if self.target_runtime_bank_cap > self.max_synthesis_candidates:
            raise ValueError("Runtime bank cap cannot exceed the synthesis shortlist cap")
        for owner in (
            "redundancy_mechanism_threshold",
            "redundancy_repair_threshold",
            "redundancy_applicability_threshold",
        ):
            _similarity_threshold(owner, getattr(self, owner))
        quantile = self.minimum_support_cohesion_quantile
        if (
            isinstance(quantile, bool)
            or not isinstance(quantile, (int, float))
            or not math.isfinite(float(quantile))
            or not 0.0 <= float(quantile) <= 1.0
        ):
            raise ValueError("V4.2 minimum-support cohesion quantile is invalid")
        if tuple(self.semantic_prompt_fields) != V4_2_SYNTHESIS_SEMANTIC_FIELDS:
            raise ValueError("V4.2 synthesis semantic prompt fields drifted")
        if self.embedding_model != V4_1_EMBEDDING_MODEL:
            raise ValueError("Unexpected V4.2 shortlist embedding model")
        if self.embedding_revision != V4_1_EMBEDDING_REVISION:
            raise ValueError("V4.2 shortlist embedding revision must remain pinned")
        if self.injection_layer != V4_LAYER_NUMBER:
            raise ValueError("V4.2 shortlist remains frozen at layer 24")
        if self.relative_phase_delta != V4_RELATIVE_PHASE_DELTA:
            raise ValueError("V4.2 shortlist keeps canonical pre-RoPE delta zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "semantic_prompt_fields": list(self.semantic_prompt_fields),
        }

    @property
    def profile_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class V42LocalRepairAtom:
    """One immutable, teacher-free clustering atom derived from a V4 signature."""

    experience_id: str
    sample_id: str
    source_experience_type: str
    problem_structure: str
    decision_point: str
    failure_mechanism: str
    repair_operator: str
    verification_operator: str
    source_signature_sha256: str
    schema_version: str = V4_2_LOCAL_ATOM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_2_LOCAL_ATOM_SCHEMA:
            raise ValueError("Unexpected V4.2 local atom schema")
        for owner in (
            "experience_id",
            "sample_id",
            "source_experience_type",
            "source_signature_sha256",
        ):
            _identifier(owner, getattr(self, owner))
        for owner in (
            "problem_structure",
            "decision_point",
            "failure_mechanism",
            "repair_operator",
            "verification_operator",
        ):
            validate_v4_card_text(owner, getattr(self, owner))

    @classmethod
    def from_signature(cls, signature: V4RepairSignature) -> "V42LocalRepairAtom":
        if not signature.applicable:
            raise ValueError("Cannot create a V4.2 local atom from an inapplicable signature")
        if signature.experience_type == "format_compliance":
            raise ValueError("Format-compliance signatures are quarantined before clustering")
        return cls(
            experience_id=signature.experience_id,
            sample_id=signature.sample_id,
            source_experience_type=signature.experience_type,
            problem_structure=signature.problem_structure,
            decision_point=signature.decision_point,
            failure_mechanism=signature.failure_mechanism,
            repair_operator=signature.repair_operator,
            verification_operator=signature.verification_operator,
            source_signature_sha256=signature.signature_sha256,
        )

    @property
    def atom_id(self) -> str:
        return f"v42-atom-{canonical_json_sha256(self.to_dict())[:20]}"

    @property
    def mechanism_text(self) -> str:
        return "\n".join(
            (
                f"Decision point: {self.decision_point}",
                f"Failure mechanism: {self.failure_mechanism}",
            )
        )

    @property
    def repair_text(self) -> str:
        return "\n".join(
            (
                f"Repair operator: {self.repair_operator}",
                f"Verification operator: {self.verification_operator}",
            )
        )

    @property
    def applicability_text(self) -> str:
        return "\n".join(
            (
                f"Problem structure: {self.problem_structure}",
                f"Decision point: {self.decision_point}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_v4_2_local_atom_payload(
    payload: Mapping[str, Any],
) -> V42LocalRepairAtom:
    return V42LocalRepairAtom(
        experience_id=payload.get("experience_id"),
        sample_id=payload.get("sample_id"),
        source_experience_type=payload.get("source_experience_type"),
        problem_structure=payload.get("problem_structure"),
        decision_point=payload.get("decision_point"),
        failure_mechanism=payload.get("failure_mechanism"),
        repair_operator=payload.get("repair_operator"),
        verification_operator=payload.get("verification_operator"),
        source_signature_sha256=payload.get("source_signature_sha256"),
        schema_version=payload.get("schema_version", V4_2_LOCAL_ATOM_SCHEMA),
    )


@dataclass(frozen=True)
class V42LocalClusterCandidate:
    """One API-eligible complete-link cluster with authenticated local evidence."""

    candidate_id: str
    member_experience_ids: tuple[str, ...]
    representative_experience_ids: tuple[str, ...]
    distinct_sample_count: int
    source_experience_type_distribution: tuple[tuple[str, int], ...]
    mechanism_similarity_min: float
    mechanism_similarity_mean: float
    repair_similarity_min: float
    repair_similarity_mean: float
    applicability_similarity_min: float
    applicability_similarity_mean: float
    joint_similarity_min: float
    joint_similarity_mean: float
    membership_sha256: str
    schema_version: str = V4_2_LOCAL_CLUSTER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_2_LOCAL_CLUSTER_SCHEMA:
            raise ValueError("Unexpected V4.2 local cluster schema")
        _identifier("candidate_id", self.candidate_id)
        _identifier("membership_sha256", self.membership_sha256)
        if tuple(sorted(self.member_experience_ids)) != self.member_experience_ids:
            raise ValueError("V4.2 local cluster members must be sorted")
        if not self.member_experience_ids:
            raise ValueError("V4.2 local cluster members must not be empty")
        for value in self.member_experience_ids:
            _identifier("member_experience_ids[]", value)
        if len(set(self.member_experience_ids)) != len(self.member_experience_ids):
            raise ValueError("V4.2 local cluster members contain duplicates")
        if (
            isinstance(self.distinct_sample_count, bool)
            or not isinstance(self.distinct_sample_count, int)
        ):
            raise ValueError("V4.2 distinct sample count must be an integer")
        if self.distinct_sample_count < V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 local cluster has insufficient distinct support")
        if len(self.member_experience_ids) < self.distinct_sample_count:
            raise ValueError("V4.2 distinct support exceeds member count")
        if len(self.representative_experience_ids) != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 local cluster requires exactly five representatives")
        if len(set(self.representative_experience_ids)) != len(
            self.representative_experience_ids
        ):
            raise ValueError("V4.2 local representatives contain duplicates")
        if not set(self.representative_experience_ids).issubset(
            self.member_experience_ids
        ):
            raise ValueError("V4.2 local representatives must be members")
        if tuple(sorted(self.source_experience_type_distribution)) != (
            self.source_experience_type_distribution
        ):
            raise ValueError("V4.2 source-type distribution must be sorted")
        if not self.source_experience_type_distribution:
            raise ValueError("V4.2 source-type distribution must not be empty")
        for name, count in self.source_experience_type_distribution:
            _identifier("source_experience_type_distribution key", name)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    "V4.2 source-type distribution counts must be positive integers"
                )
        if sum(count for _name, count in self.source_experience_type_distribution) != len(
            self.member_experience_ids
        ):
            raise ValueError("V4.2 source-type distribution does not cover members")
        for owner in (
            "mechanism_similarity_min",
            "mechanism_similarity_mean",
            "repair_similarity_min",
            "repair_similarity_mean",
            "applicability_similarity_min",
            "applicability_similarity_mean",
            "joint_similarity_min",
            "joint_similarity_mean",
        ):
            _unit_interval(owner, getattr(self, owner))
        for minimum, mean, owner in (
            (
                self.mechanism_similarity_min,
                self.mechanism_similarity_mean,
                "mechanism",
            ),
            (self.repair_similarity_min, self.repair_similarity_mean, "repair"),
            (
                self.applicability_similarity_min,
                self.applicability_similarity_mean,
                "applicability",
            ),
            (self.joint_similarity_min, self.joint_similarity_mean, "joint"),
        ):
            if minimum > mean:
                raise ValueError(f"V4.2 {owner} minimum exceeds its mean")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "member_experience_ids": list(self.member_experience_ids),
            "representative_experience_ids": list(
                self.representative_experience_ids
            ),
            "source_experience_type_distribution": {
                key: value for key, value in self.source_experience_type_distribution
            },
        }


def validate_v4_2_cluster_payload(
    payload: Mapping[str, Any],
) -> V42LocalClusterCandidate:
    distribution = payload.get("source_experience_type_distribution")
    if not isinstance(distribution, Mapping):
        raise ValueError("V4.2 local cluster is missing source-type distribution")
    return V42LocalClusterCandidate(
        candidate_id=payload.get("candidate_id"),
        member_experience_ids=tuple(payload.get("member_experience_ids", ())),
        representative_experience_ids=tuple(
            payload.get("representative_experience_ids", ())
        ),
        distinct_sample_count=payload.get("distinct_sample_count"),
        source_experience_type_distribution=tuple(
            sorted((str(key), int(value)) for key, value in distribution.items())
        ),
        mechanism_similarity_min=payload.get("mechanism_similarity_min"),
        mechanism_similarity_mean=payload.get("mechanism_similarity_mean"),
        repair_similarity_min=payload.get("repair_similarity_min"),
        repair_similarity_mean=payload.get("repair_similarity_mean"),
        applicability_similarity_min=payload.get("applicability_similarity_min"),
        applicability_similarity_mean=payload.get("applicability_similarity_mean"),
        joint_similarity_min=payload.get("joint_similarity_min"),
        joint_similarity_mean=payload.get("joint_similarity_mean"),
        membership_sha256=payload.get("membership_sha256"),
        schema_version=payload.get("schema_version", V4_2_LOCAL_CLUSTER_SCHEMA),
    )


__all__ = [
    "V4_2_CONSTRUCTION_PROFILE_SCHEMA",
    "V4_2_DEFAULT_APPLICABILITY_THRESHOLD",
    "V4_2_DEFAULT_APPLICABILITY_WEIGHT",
    "V4_2_DEFAULT_MAX_API_CANDIDATES",
    "V4_2_DEFAULT_MAX_SYNTHESIS_CANDIDATES",
    "V4_2_DEFAULT_MIN_SUPPORT_COHESION_QUANTILE",
    "V4_2_DEFAULT_MECHANISM_THRESHOLD",
    "V4_2_DEFAULT_MECHANISM_WEIGHT",
    "V4_2_DEFAULT_NEIGHBOR_COUNT",
    "V4_2_DEFAULT_PREFERRED_SUPPORT",
    "V4_2_DEFAULT_REDUNDANCY_APPLICABILITY_THRESHOLD",
    "V4_2_DEFAULT_REDUNDANCY_MECHANISM_THRESHOLD",
    "V4_2_DEFAULT_REDUNDANCY_REPAIR_THRESHOLD",
    "V4_2_DEFAULT_REPAIR_THRESHOLD",
    "V4_2_DEFAULT_REPAIR_WEIGHT",
    "V4_2_DEFAULT_REPRESENTATIVE_COUNT",
    "V4_2_DEFAULT_REVIEW_BATCH_SIZE",
    "V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE",
    "V4_2_DEFAULT_TARGET_RUNTIME_BANK_CAP",
    "V4_2_EMBEDDING_MANIFEST_SCHEMA",
    "V4_2_LOCAL_ATOM_SCHEMA",
    "V4_2_LOCAL_CLUSTER_PLAN_SCHEMA",
    "V4_2_LOCAL_CLUSTER_SCHEMA",
    "V4_2_POSITIVE_EDGE_SCHEMA",
    "V4_2_PREFLIGHT_REPORT_SCHEMA",
    "V4_2_REVIEW_PACKET_SCHEMA",
    "V4_2_SHORTLIST_MANIFEST_SCHEMA",
    "V4_2_SHORTLIST_PREFLIGHT_SCHEMA",
    "V4_2_SHORTLIST_PROFILE_SCHEMA",
    "V4_2_SYNTHESIS_SEMANTIC_FIELDS",
    "V42ConstructionProfile",
    "V42LocalClusterCandidate",
    "V42LocalRepairAtom",
    "V42ShortlistProfile",
    "validate_v4_2_cluster_payload",
    "validate_v4_2_local_atom_payload",
]
