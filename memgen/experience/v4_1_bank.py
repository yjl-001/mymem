"""Contracts for evidence-preserving MemGen V4.1 bank construction.

V4.1 keeps the online V4 target/reference side-bank interface, but replaces
the path-dependent free-text map/reduce stage.  Existing per-experience V4
repair signatures remain immutable evidence.  A second teacher pass maps each
signature to a bounded process atom, semantic retrieval proposes cross-type
candidate edges, and only explicitly judged, clique-consistent groups may be
audited for runtime admission.

The distinction between ``source_experience_type`` and ``memory_role`` is
intentional.  The former records how the verifier classified the final
failure; the latter records whether the reusable memory concerns reasoning or
answer serialization.  Verifier outcome types are never hard cluster keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_bank import (
    V4_BANK_RECORD_SCHEMA,
    V4_LAYER_NUMBER,
    V4_MAX_CONSTRUCTION_EXAMPLES,
    V4_MIN_CONSTRUCTION_EXAMPLES,
    V4_RELATIVE_PHASE_DELTA,
    V4_TEACHER_MODEL,
    V4_TEACHER_THINKING,
    V4_TEACHER_TEMPERATURE,
    V4CardReview,
    V4ProcessCard,
    V4RepairSignature,
    validate_v4_card_text,
)


V4_1_CONSTRUCTION_PROFILE_SCHEMA = "memgen-v4.1-construction-profile-v1"
V4_1_CANONICAL_ATOM_SCHEMA = "memgen-v4.1-canonical-repair-atom-v1"
V4_1_CANONICAL_PAYLOAD_SCHEMA = "memgen-v4.1-canonical-repair-batch-v1"
V4_1_PAIR_PAYLOAD_SCHEMA = "memgen-v4.1-candidate-pair-judgments-v1"
V4_1_PAIR_JUDGMENT_SCHEMA = "memgen-v4.1-candidate-pair-judgment-v1"
V4_1_CLUSTER_AUDIT_SCHEMA = "memgen-v4.1-cluster-coherence-audit-v1"
V4_1_CLUSTER_PLAN_SCHEMA = "memgen-v4.1-repair-cluster-plan-v1"
V4_1_BANK_MANIFEST_SCHEMA = "memgen-v4-bank-manifest-v2"

V4_1_CANONICAL_PROMPT_VERSION = "memgen-v4.1-canonical-atom-deepseek-v1"
V4_1_PAIR_PROMPT_VERSION = "memgen-v4.1-pair-judge-deepseek-v1"
V4_1_AUDIT_PROMPT_VERSION = "memgen-v4.1-cluster-audit-deepseek-v1"
V4_1_CARD_PROMPT_VERSION = "memgen-v4.1-process-card-deepseek-v1"
V4_1_REVIEW_PROMPT_VERSION = "memgen-v4.1-card-review-deepseek-v1"

V4_1_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
V4_1_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
V4_1_DEFAULT_CANONICAL_BATCH_SIZE = 24
V4_1_DEFAULT_PAIR_BATCH_SIZE = 24
V4_1_DEFAULT_NEIGHBOR_COUNT = 12


V4_1_MEMORY_ROLES = (
    "reasoning_process",
    "answer_serialization",
    "unusable",
)
V4_1_STATE_SCOPES = (
    "problem_parsing",
    "relation_translation",
    "plan_selection",
    "equation_setup",
    "intermediate_computation",
    "running_state",
    "unit_conversion",
    "constraint_check",
    "answer_selection",
    "answer_serialization",
    "other",
)
V4_1_MECHANISM_FAMILIES = (
    "omit_required_component",
    "duplicate_component",
    "wrong_reference_or_antecedent",
    "wrong_operation_or_direction",
    "wrong_base_or_denominator",
    "wrong_order_or_state",
    "unit_or_rate_mismatch",
    "approximation_or_rounding",
    "arithmetic_execution",
    "unsupported_assumption",
    "premature_finalization",
    "output_representation",
    "other",
)
V4_1_REPAIR_FAMILIES = (
    "enumerate_and_bind_quantities",
    "translate_relations_explicitly",
    "track_running_state",
    "align_units_and_rates",
    "decompose_then_aggregate",
    "solve_equation_systematically",
    "verify_operation_direction",
    "preserve_exactness_then_round",
    "recompute_and_cross_check",
    "verify_requested_quantity",
    "canonicalize_final_answer",
    "other",
)
V4_1_APPLICABILITY_FAMILIES = (
    "additive_composition",
    "sequential_updates",
    "multiplicative_scaling",
    "proportions_and_ratios",
    "fractions_and_percentages",
    "rates_and_unit_conversions",
    "comparisons_and_differences",
    "algebraic_relations",
    "counting_and_grouping",
    "temporal_reasoning",
    "geometric_measurement",
    "general_multistep",
    "answer_serialization",
    "other",
)


_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _identifier(owner: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty identifier")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"{owner} is not a canonical identifier")
    return normalized


def _choice(owner: str, value: Any, choices: Sequence[str]) -> str:
    normalized = _identifier(owner, value)
    if normalized not in choices:
        raise ValueError(f"{owner} has unsupported value: {normalized}")
    return normalized


def _boolean(owner: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{owner} must be boolean")
    return value


def _issues(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("issues must be an array")
    result = tuple(validate_v4_card_text("issues[]", item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError("issues contains duplicates")
    return result


@dataclass(frozen=True)
class V41ConstructionProfile:
    """Authenticated choices that affect V4.1 logical construction output."""

    teacher_model: str = V4_TEACHER_MODEL
    temperature: float = V4_TEACHER_TEMPERATURE
    thinking: str = V4_TEACHER_THINKING
    min_construction_examples: int = V4_MIN_CONSTRUCTION_EXAMPLES
    max_construction_examples: int = V4_MAX_CONSTRUCTION_EXAMPLES
    source_signature_schema: str = "memgen-v4-repair-signature-v1"
    grouping_rule: str = "canonical_atom_candidate_graph_clique_audit"
    source_experience_type_policy: str = "audit_metadata_not_cluster_boundary"
    answer_serialization_policy: str = "exclude_from_target_bank"
    unsupported_policy: str = "archive_without_forced_assignment"
    target_source: str = "official_solution_plus_verified_success"
    reference_source: str = "paired_verified_failure"
    embedding_model: str = V4_1_EMBEDDING_MODEL
    embedding_revision: str = V4_1_EMBEDDING_REVISION
    neighbor_count: int = V4_1_DEFAULT_NEIGHBOR_COUNT
    canonical_batch_size: int = V4_1_DEFAULT_CANONICAL_BATCH_SIZE
    pair_batch_size: int = V4_1_DEFAULT_PAIR_BATCH_SIZE
    injection_layer: int = V4_LAYER_NUMBER
    relative_phase_delta: int = V4_RELATIVE_PHASE_DELTA
    schema_version: str = V4_1_CONSTRUCTION_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_1_CONSTRUCTION_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4.1 construction profile schema")
        if self.teacher_model != V4_TEACHER_MODEL:
            raise ValueError(f"V4.1 construction requires {V4_TEACHER_MODEL}")
        if self.temperature != V4_TEACHER_TEMPERATURE:
            raise ValueError("V4.1 teacher temperature is frozen at zero")
        if self.thinking != V4_TEACHER_THINKING:
            raise ValueError("V4.1 teacher thinking mode is frozen at disabled")
        if self.min_construction_examples != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.1 minimum support is frozen at five")
        if self.max_construction_examples != V4_MAX_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.1 representative cap is frozen at ten")
        if self.source_signature_schema != "memgen-v4-repair-signature-v1":
            raise ValueError("V4.1 requires authenticated V4 repair signatures")
        if self.grouping_rule != "canonical_atom_candidate_graph_clique_audit":
            raise ValueError("Unexpected V4.1 grouping rule")
        if self.source_experience_type_policy != "audit_metadata_not_cluster_boundary":
            raise ValueError("Unexpected V4.1 experience-type policy")
        if self.answer_serialization_policy != "exclude_from_target_bank":
            raise ValueError("Unexpected V4.1 answer-serialization policy")
        if self.unsupported_policy != "archive_without_forced_assignment":
            raise ValueError("Unexpected V4.1 unsupported policy")
        if self.target_source != "official_solution_plus_verified_success":
            raise ValueError("Unexpected V4.1 target evidence source")
        if self.reference_source != "paired_verified_failure":
            raise ValueError("Unexpected V4.1 reference evidence source")
        if self.embedding_model != V4_1_EMBEDDING_MODEL:
            raise ValueError("Unexpected V4.1 embedding model")
        if self.embedding_revision != V4_1_EMBEDDING_REVISION:
            raise ValueError("V4.1 embedding revision must be immutable")
        for owner, value in (
            ("neighbor_count", self.neighbor_count),
            ("canonical_batch_size", self.canonical_batch_size),
            ("pair_batch_size", self.pair_batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"V4.1 {owner} must be a positive integer")
        if self.injection_layer != V4_LAYER_NUMBER:
            raise ValueError("V4.1 remains frozen at layer 24")
        if self.relative_phase_delta != V4_RELATIVE_PHASE_DELTA:
            raise ValueError("V4.1 uses canonical pre-RoPE keys with delta zero")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class V41CanonicalRepairAtom:
    """One bounded process abstraction derived from an immutable V4 signature."""

    experience_id: str
    sample_id: str
    source_experience_type: str
    memory_role: str
    state_scope: str
    mechanism_family: str
    repair_family: str
    applicability_family: str
    failure_transition: str
    repair_action: str
    applicability_condition: str
    verification_action: str
    source_signature_sha256: str
    exclusion_reason: str | None
    schema_version: str = V4_1_CANONICAL_ATOM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_1_CANONICAL_ATOM_SCHEMA:
            raise ValueError("Unexpected V4.1 canonical atom schema")
        _identifier("experience_id", self.experience_id)
        _identifier("sample_id", self.sample_id)
        _identifier("source_experience_type", self.source_experience_type)
        _choice("memory_role", self.memory_role, V4_1_MEMORY_ROLES)
        _choice("state_scope", self.state_scope, V4_1_STATE_SCOPES)
        _choice(
            "mechanism_family",
            self.mechanism_family,
            V4_1_MECHANISM_FAMILIES,
        )
        _choice("repair_family", self.repair_family, V4_1_REPAIR_FAMILIES)
        _choice(
            "applicability_family",
            self.applicability_family,
            V4_1_APPLICABILITY_FAMILIES,
        )
        for owner, value in (
            ("failure_transition", self.failure_transition),
            ("repair_action", self.repair_action),
            ("applicability_condition", self.applicability_condition),
            ("verification_action", self.verification_action),
        ):
            validate_v4_card_text(owner, value)
        _identifier("source_signature_sha256", self.source_signature_sha256)
        if self.memory_role == "reasoning_process":
            if self.exclusion_reason is not None:
                raise ValueError("Reasoning atoms cannot carry an exclusion reason")
            if self.state_scope == "answer_serialization":
                raise ValueError("Reasoning atoms cannot use answer-serialization scope")
            if self.mechanism_family == "output_representation":
                raise ValueError("Output-representation atoms must be quarantined")
            if self.repair_family == "canonicalize_final_answer":
                raise ValueError("Final-answer canonicalization must be quarantined")
            if self.applicability_family == "answer_serialization":
                raise ValueError("Answer-serialization applicability must be quarantined")
        elif self.memory_role == "answer_serialization":
            if (
                self.state_scope != "answer_serialization"
                or self.mechanism_family != "output_representation"
                or self.repair_family != "canonicalize_final_answer"
                or self.applicability_family != "answer_serialization"
            ):
                raise ValueError(
                    "Answer-serialization atoms must use the dedicated quarantine categories"
                )
            validate_v4_card_text("exclusion_reason", self.exclusion_reason)
        else:
            if (
                self.state_scope != "other"
                or self.mechanism_family != "other"
                or self.repair_family != "other"
                or self.applicability_family != "other"
            ):
                raise ValueError("Unusable atoms must use the dedicated other categories")
            validate_v4_card_text("exclusion_reason", self.exclusion_reason)

    @property
    def atom_id(self) -> str:
        return f"atom-{canonical_json_sha256(self.to_dict())[:20]}"

    @property
    def canonical_key(self) -> tuple[str, str, str, str]:
        return (
            self.state_scope,
            self.mechanism_family,
            self.repair_family,
            self.applicability_family,
        )

    @property
    def embedding_text(self) -> str:
        return "\n".join(
            (
                f"Failure transition: {self.failure_transition}",
                f"Repair action: {self.repair_action}",
                f"Applicability: {self.applicability_condition}",
                f"Verification: {self.verification_action}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V41PairJudgment:
    pair_id: str
    left_seed_id: str
    right_seed_id: str
    same_failure_mechanism: bool
    same_repair_action: bool
    compatible_applicability: bool
    process_only: bool
    merge: bool
    evidence: str
    issues: tuple[str, ...]
    schema_version: str = V4_1_PAIR_JUDGMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_1_PAIR_JUDGMENT_SCHEMA:
            raise ValueError("Unexpected V4.1 pair-judgment schema")
        _identifier("pair_id", self.pair_id)
        _identifier("left_seed_id", self.left_seed_id)
        _identifier("right_seed_id", self.right_seed_id)
        if self.left_seed_id >= self.right_seed_id:
            raise ValueError("V4.1 pair endpoints must be canonically ordered")
        for owner, value in (
            ("same_failure_mechanism", self.same_failure_mechanism),
            ("same_repair_action", self.same_repair_action),
            ("compatible_applicability", self.compatible_applicability),
            ("process_only", self.process_only),
            ("merge", self.merge),
        ):
            _boolean(owner, value)
        expected_merge = all(
            (
                self.same_failure_mechanism,
                self.same_repair_action,
                self.compatible_applicability,
                self.process_only,
            )
        )
        if self.merge != expected_merge:
            raise ValueError("V4.1 pair merge decision is inconsistent")
        validate_v4_card_text("pair evidence", self.evidence)
        if self.merge == bool(self.issues):
            raise ValueError("V4.1 pair issues are inconsistent with merge")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "issues": list(self.issues)}


@dataclass(frozen=True)
class V41ClusterAudit:
    candidate_id: str
    coherent: bool
    process_only: bool
    transferable: bool
    serialization_free: bool
    leakage_free: bool
    approve: bool
    title: str
    failure_mechanism: str
    repair_operator: str
    scope_summary: str
    evidence: str
    issues: tuple[str, ...]
    schema_version: str = V4_1_CLUSTER_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_1_CLUSTER_AUDIT_SCHEMA:
            raise ValueError("Unexpected V4.1 cluster-audit schema")
        _identifier("candidate_id", self.candidate_id)
        for owner, value in (
            ("coherent", self.coherent),
            ("process_only", self.process_only),
            ("transferable", self.transferable),
            ("serialization_free", self.serialization_free),
            ("leakage_free", self.leakage_free),
            ("approve", self.approve),
        ):
            _boolean(owner, value)
        expected_approval = all(
            (
                self.coherent,
                self.process_only,
                self.transferable,
                self.serialization_free,
                self.leakage_free,
            )
        )
        if self.approve != expected_approval:
            raise ValueError("V4.1 cluster-audit approval is inconsistent")
        for owner, value in (
            ("audit title", self.title),
            ("audit failure_mechanism", self.failure_mechanism),
            ("audit repair_operator", self.repair_operator),
            ("audit scope_summary", self.scope_summary),
            ("audit evidence", self.evidence),
        ):
            validate_v4_card_text(owner, value)
        if self.approve == bool(self.issues):
            raise ValueError("V4.1 cluster-audit issues are inconsistent with approval")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "issues": list(self.issues)}


@dataclass(frozen=True)
class V41RepairCluster:
    """One audited process cluster; verifier outcome types are only metadata."""

    cluster_key: str
    candidate_id: str
    title: str
    failure_mechanism: str
    repair_operator: str
    scope_summary: str
    member_experience_ids: tuple[str, ...]
    representative_experience_ids: tuple[str, ...]
    source_experience_type_distribution: tuple[tuple[str, int], ...]
    canonical_seed_ids: tuple[str, ...]
    audit_sha256: str

    def __post_init__(self) -> None:
        _identifier("cluster_key", self.cluster_key)
        _identifier("candidate_id", self.candidate_id)
        for owner, value in (
            ("cluster title", self.title),
            ("cluster failure_mechanism", self.failure_mechanism),
            ("cluster repair_operator", self.repair_operator),
            ("cluster scope_summary", self.scope_summary),
        ):
            validate_v4_card_text(owner, value)
        if len(set(self.member_experience_ids)) != len(self.member_experience_ids):
            raise ValueError("V4.1 cluster members contain duplicates")
        if len(self.member_experience_ids) < V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.1 cluster has fewer than five members")
        if not (
            V4_MIN_CONSTRUCTION_EXAMPLES
            <= len(self.representative_experience_ids)
            <= V4_MAX_CONSTRUCTION_EXAMPLES
        ):
            raise ValueError("V4.1 cluster requires five to ten representatives")
        if len(set(self.representative_experience_ids)) != len(
            self.representative_experience_ids
        ):
            raise ValueError("V4.1 representatives contain duplicates")
        if not set(self.representative_experience_ids).issubset(
            self.member_experience_ids
        ):
            raise ValueError("V4.1 representatives must be cluster members")
        if not self.source_experience_type_distribution:
            raise ValueError("V4.1 cluster is missing experience-type provenance")
        if any(count <= 0 for _name, count in self.source_experience_type_distribution):
            raise ValueError("V4.1 experience-type counts must be positive")
        if sum(count for _name, count in self.source_experience_type_distribution) != len(
            self.member_experience_ids
        ):
            raise ValueError("V4.1 experience-type counts do not cover members")
        if tuple(sorted(self.source_experience_type_distribution)) != (
            self.source_experience_type_distribution
        ):
            raise ValueError("V4.1 experience-type distribution must be sorted")
        if not self.canonical_seed_ids or len(set(self.canonical_seed_ids)) != len(
            self.canonical_seed_ids
        ):
            raise ValueError("V4.1 canonical seed IDs are missing or duplicated")
        _identifier("audit_sha256", self.audit_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_key": self.cluster_key,
            "candidate_id": self.candidate_id,
            "memory_role": "reasoning_process",
            "title": self.title,
            "failure_mechanism": self.failure_mechanism,
            "repair_operator": self.repair_operator,
            "scope_summary": self.scope_summary,
            "member_experience_ids": list(self.member_experience_ids),
            "representative_experience_ids": list(
                self.representative_experience_ids
            ),
            "source_experience_type_distribution": {
                key: value for key, value in self.source_experience_type_distribution
            },
            "canonical_seed_ids": list(self.canonical_seed_ids),
            "audit_sha256": self.audit_sha256,
        }


def parse_v4_1_canonical_atom(
    payload: Mapping[str, Any],
    *,
    signature: V4RepairSignature,
) -> V41CanonicalRepairAtom:
    if payload.get("schema_version") not in {None, V4_1_CANONICAL_ATOM_SCHEMA}:
        raise ValueError("Unexpected V4.1 canonical atom payload schema")
    memory_role = payload.get("memory_role")
    if signature.experience_type == "format_compliance":
        # The verifier has already established that the numeric answer is
        # correct and only its final representation failed.  This deterministic
        # safety boundary is stronger than the teacher's redundant role fields.
        memory_role = "answer_serialization"
    state_scope = payload.get("state_scope")
    mechanism_family = payload.get("mechanism_family")
    repair_family = payload.get("repair_family")
    applicability_family = payload.get("applicability_family")
    exclusion_reason = payload.get("exclusion_reason")
    if memory_role == "reasoning_process":
        # ``exclusion_reason`` is metadata, not semantic evidence.  Providers
        # commonly emit an empty string or explanatory text instead of JSON
        # null.  Canonicalize it locally rather than rejecting and splitting a
        # sound semantic batch.
        exclusion_reason = None
    elif memory_role == "answer_serialization":
        state_scope = "answer_serialization"
        mechanism_family = "output_representation"
        repair_family = "canonicalize_final_answer"
        applicability_family = "answer_serialization"
        exclusion_reason = "answer serialization is outside the reasoning process bank"
    elif memory_role == "unusable":
        state_scope = "other"
        mechanism_family = "other"
        repair_family = "other"
        applicability_family = "other"
        exclusion_reason = "the signature does not ground one reusable reasoning transition"
    atom = V41CanonicalRepairAtom(
        experience_id=signature.experience_id,
        sample_id=signature.sample_id,
        source_experience_type=signature.experience_type,
        memory_role=memory_role,
        state_scope=state_scope,
        mechanism_family=mechanism_family,
        repair_family=repair_family,
        applicability_family=applicability_family,
        failure_transition=payload.get("failure_transition"),
        repair_action=payload.get("repair_action"),
        applicability_condition=payload.get("applicability_condition"),
        verification_action=payload.get("verification_action"),
        source_signature_sha256=signature.signature_sha256,
        exclusion_reason=exclusion_reason,
    )
    return atom


def parse_v4_1_pair_judgment(
    payload: Mapping[str, Any],
    *,
    pair_id: str,
    left_seed_id: str,
    right_seed_id: str,
) -> V41PairJudgment:
    if payload.get("schema_version") not in {None, V4_1_PAIR_JUDGMENT_SCHEMA}:
        raise ValueError("Unexpected V4.1 pair-judgment payload schema")
    return V41PairJudgment(
        pair_id=pair_id,
        left_seed_id=left_seed_id,
        right_seed_id=right_seed_id,
        same_failure_mechanism=payload.get("same_failure_mechanism"),
        same_repair_action=payload.get("same_repair_action"),
        compatible_applicability=payload.get("compatible_applicability"),
        process_only=payload.get("process_only"),
        merge=payload.get("merge"),
        evidence=payload.get("evidence"),
        issues=_issues(payload.get("issues")),
    )


def parse_v4_1_cluster_audit(
    payload: Mapping[str, Any], *, candidate_id: str
) -> V41ClusterAudit:
    if payload.get("schema_version") not in {None, V4_1_CLUSTER_AUDIT_SCHEMA}:
        raise ValueError("Unexpected V4.1 cluster-audit payload schema")
    return V41ClusterAudit(
        candidate_id=candidate_id,
        coherent=payload.get("coherent"),
        process_only=payload.get("process_only"),
        transferable=payload.get("transferable"),
        serialization_free=payload.get("serialization_free"),
        leakage_free=payload.get("leakage_free"),
        approve=payload.get("approve"),
        title=payload.get("title"),
        failure_mechanism=payload.get("failure_mechanism"),
        repair_operator=payload.get("repair_operator"),
        scope_summary=payload.get("scope_summary"),
        evidence=payload.get("evidence"),
        issues=_issues(payload.get("issues")),
    )


def build_v4_1_bank_record(
    *,
    cluster: V41RepairCluster,
    card: V4ProcessCard,
    review: V4CardReview,
    signatures: Sequence[V4RepairSignature],
    atoms: Sequence[V41CanonicalRepairAtom],
    construction_input_sha256: str,
    profile: V41ConstructionProfile,
) -> dict[str, Any]:
    """Build a V4-compatible tensor-free record with V4.1 provenance."""

    if card.cluster_key != cluster.cluster_key or review.cluster_key != cluster.cluster_key:
        raise ValueError("V4.1 cluster/card/review identity mismatch")
    if not review.approve:
        raise ValueError("Only approved V4.1 cards may enter the runtime bank")
    signature_by_id = {item.experience_id: item for item in signatures}
    atom_by_id = {item.experience_id: item for item in atoms}
    members = tuple(cluster.member_experience_ids)
    if set(members) - set(signature_by_id) or set(members) - set(atom_by_id):
        raise ValueError("V4.1 bank record lost signature or atom evidence")
    sample_ids = sorted({signature_by_id[item].sample_id for item in members})
    if len(sample_ids) < V4_MIN_CONSTRUCTION_EXAMPLES:
        raise ValueError("V4.1 bank record has insufficient distinct support")
    semantic = {
        "cluster": cluster.to_dict(),
        "card_sha256": card.card_sha256,
        "profile_sha256": profile.profile_sha256,
    }
    bank_id = f"v4-{cluster.cluster_key}-{canonical_json_sha256(semantic)[:12]}"
    record = {
        "schema_version": V4_BANK_RECORD_SCHEMA,
        "construction_version": "v4.1",
        "bank_id": bank_id,
        "benchmark": "openai/gsm8k",
        "cluster": cluster.to_dict(),
        "process_card": card.to_dict(),
        "review": review.to_dict(),
        "construction": {
            "experience_ids": list(members),
            "representative_experience_ids": list(
                cluster.representative_experience_ids
            ),
            "sample_ids": sample_ids,
            "distinct_sample_count": len(sample_ids),
            "input_sha256": construction_input_sha256,
            "signature_sha256": {
                item: signature_by_id[item].signature_sha256
                for item in sorted(members)
            },
            "canonical_atom_sha256": {
                item: canonical_json_sha256(atom_by_id[item].to_dict())
                for item in sorted(members)
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
        "construction_profile": profile.to_dict(),
        "construction_profile_sha256": profile.profile_sha256,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def build_v4_1_bank_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    profile: V41ConstructionProfile,
    inputs: Mapping[str, Any],
    teacher: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError("V4.1 bank manifest requires approved records")
    bank_ids = [str(record.get("bank_id", "")) for record in records]
    if any(not item for item in bank_ids) or len(set(bank_ids)) != len(bank_ids):
        raise ValueError("V4.1 bank IDs are missing or duplicated")
    for record in records:
        if record.get("schema_version") != V4_BANK_RECORD_SCHEMA:
            raise ValueError("V4.1 manifest received an unexpected record schema")
        logical = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if record.get("record_sha256") != canonical_json_sha256(logical):
            raise ValueError("V4.1 bank record hash mismatch")
        if record.get("construction_profile_sha256") != profile.profile_sha256:
            raise ValueError("V4.1 bank record profile drifted")
    manifest = {
        "schema_version": V4_1_BANK_MANIFEST_SCHEMA,
        "construction_version": "v4.1",
        "status": "constructed_not_tensor_compiled",
        "qualified_for_online_use": False,
        "benchmark": "openai/gsm8k",
        "record_count": len(records),
        "bank_ids": bank_ids,
        "record_order_sha256": canonical_json_sha256(bank_ids),
        "record_sha256": {
            str(record["bank_id"]): str(record["record_sha256"])
            for record in records
        },
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "inputs": dict(inputs),
        "teacher": dict(teacher),
        "auxiliary_banks_materialized": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


__all__ = [
    "V4_1_APPLICABILITY_FAMILIES",
    "V4_1_AUDIT_PROMPT_VERSION",
    "V4_1_BANK_MANIFEST_SCHEMA",
    "V4_1_CANONICAL_ATOM_SCHEMA",
    "V4_1_CANONICAL_PAYLOAD_SCHEMA",
    "V4_1_CANONICAL_PROMPT_VERSION",
    "V4_1_CARD_PROMPT_VERSION",
    "V4_1_CLUSTER_AUDIT_SCHEMA",
    "V4_1_CLUSTER_PLAN_SCHEMA",
    "V4_1_DEFAULT_CANONICAL_BATCH_SIZE",
    "V4_1_DEFAULT_NEIGHBOR_COUNT",
    "V4_1_DEFAULT_PAIR_BATCH_SIZE",
    "V4_1_EMBEDDING_MODEL",
    "V4_1_EMBEDDING_REVISION",
    "V4_1_MECHANISM_FAMILIES",
    "V4_1_MEMORY_ROLES",
    "V4_1_PAIR_JUDGMENT_SCHEMA",
    "V4_1_PAIR_PAYLOAD_SCHEMA",
    "V4_1_PAIR_PROMPT_VERSION",
    "V4_1_REPAIR_FAMILIES",
    "V4_1_REVIEW_PROMPT_VERSION",
    "V4_1_STATE_SCOPES",
    "V41CanonicalRepairAtom",
    "V41ClusterAudit",
    "V41ConstructionProfile",
    "V41PairJudgment",
    "V41RepairCluster",
    "build_v4_1_bank_manifest",
    "build_v4_1_bank_record",
    "parse_v4_1_canonical_atom",
    "parse_v4_1_cluster_audit",
    "parse_v4_1_pair_judgment",
]
