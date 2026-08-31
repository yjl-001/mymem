"""Pure contracts for the two-stage MemGen V4 repair-bank system.

V4 deliberately starts from verifier-backed construction episodes rather than
upgrading any V3 memory record.  This module contains only deterministic,
CPU-light schemas and validation helpers.  Network/model entry points live in
``scripts/`` and layer-24 tensor compilation lives in ``memgen.model``.

The offline construction contract is MI-inspired but adapted to MemGen's
cross-problem repair setting:

* DeepSeek V4 Flash abstracts one repair signature per verified success/failure
  pair.
* Signatures are grouped by failure mechanism plus repair operator, never by
  evaluation reward.
* A runtime bank requires at least five distinct construction problem IDs.
* Target cards describe the desired process; reference cards describe the
  recurring undesired process.  Reference cards are never injected online.
* All generated records are instance-free, provenance-bound, and frozen before
  online selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256


V4_REPAIR_SIGNATURE_SCHEMA = "memgen-v4-repair-signature-v1"
V4_CLUSTER_PLAN_SCHEMA = "memgen-v4-repair-cluster-plan-v1"
V4_PROCESS_CARD_SCHEMA = "memgen-v4-process-card-v1"
V4_CARD_REVIEW_SCHEMA = "memgen-v4-process-card-review-v1"
V4_BANK_RECORD_SCHEMA = "memgen-v4-bank-record-v1"
V4_BANK_MANIFEST_SCHEMA = "memgen-v4-bank-manifest-v1"
V4_CONSTRUCTION_PROFILE_SCHEMA = "memgen-v4-construction-profile-v1"

V4_TEACHER_MODEL = "deepseek-v4-flash"
V4_TEACHER_TEMPERATURE = 0.0
V4_TEACHER_THINKING = "disabled"
V4_MIN_CONSTRUCTION_EXAMPLES = 5
V4_MAX_CONSTRUCTION_EXAMPLES = 10
V4_LAYER_NUMBER = 24
V4_RELATIVE_PHASE_DELTA = 0
V4_MAX_SELECTOR_ATTEMPTS = 3
V4_RECOVERY_LOW_TOKEN_COUNT = 2
V4_MAX_ACTIVE_STEPS = 32

V4_SIGNATURE_PROMPT_VERSION = "memgen-v4-repair-signature-deepseek-v1"
V4_CLUSTER_PROMPT_VERSION = "memgen-v4-repair-cluster-deepseek-v1"
V4_CARD_PROMPT_VERSION = "memgen-v4-process-card-deepseek-v1"
V4_CARD_REVIEW_PROMPT_VERSION = "memgen-v4-process-card-review-deepseek-v1"

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_INSTANCE_LITERAL_PATTERNS = {
    "digit": re.compile(r"\d"),
    "boxed_answer": re.compile(r"\\boxed|boxed\s*\{", re.IGNORECASE),
    "latex_fraction": re.compile(r"\\frac", re.IGNORECASE),
    "gsm8k_answer_marker": re.compile(r"####"),
    "explicit_equation": re.compile(
        r"(?:^|\s)[A-Za-z][A-Za-z0-9_]*\s*=\s*[-+*/()A-Za-z0-9_.]+"
    ),
}


def _require_nonempty_string(owner: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string")
    return value.strip()


def _require_identifier(owner: str, value: Any) -> str:
    identifier = _require_nonempty_string(owner, value)
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"{owner} is not a canonical identifier")
    return identifier


def _require_bool(owner: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{owner} must be boolean")
    return value


def _string_list(owner: str, value: Any, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{owner} must be an array")
    result = tuple(_require_nonempty_string(f"{owner}[]", item) for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{owner} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{owner} contains duplicates")
    return result


def v4_card_leakage_reasons(text: str) -> tuple[str, ...]:
    """Return deterministic instance-leakage reasons for one generated field."""

    normalized = _require_nonempty_string("card text", text)
    return tuple(
        name for name, pattern in _INSTANCE_LITERAL_PATTERNS.items() if pattern.search(normalized)
    )


def validate_v4_card_text(owner: str, text: Any) -> str:
    value = _require_nonempty_string(owner, text)
    reasons = v4_card_leakage_reasons(value)
    if reasons:
        raise ValueError(f"{owner} contains instance-specific content: {list(reasons)}")
    return value


@dataclass(frozen=True)
class V4ConstructionProfile:
    """Frozen choices shared by every V4 construction artifact."""

    teacher_model: str = V4_TEACHER_MODEL
    temperature: float = V4_TEACHER_TEMPERATURE
    thinking: str = V4_TEACHER_THINKING
    min_construction_examples: int = V4_MIN_CONSTRUCTION_EXAMPLES
    max_construction_examples: int = V4_MAX_CONSTRUCTION_EXAMPLES
    grouping_rule: str = "failure_mechanism_plus_repair_operator"
    target_source: str = "official_solution_plus_verified_success"
    reference_source: str = "paired_verified_failure"
    injection_layer: int = V4_LAYER_NUMBER
    relative_phase_delta: int = V4_RELATIVE_PHASE_DELTA
    schema_version: str = V4_CONSTRUCTION_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_CONSTRUCTION_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4 construction profile schema")
        if self.teacher_model != V4_TEACHER_MODEL:
            raise ValueError(f"V4 construction requires {V4_TEACHER_MODEL}")
        if self.temperature != 0.0:
            raise ValueError("V4 construction temperature is frozen at zero")
        if self.thinking != V4_TEACHER_THINKING:
            raise ValueError("V4 construction thinking mode is frozen at disabled")
        if self.min_construction_examples != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4 requires exactly five examples as the minimum support")
        if self.max_construction_examples != V4_MAX_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4 construction representative cap is frozen at ten")
        if self.grouping_rule != "failure_mechanism_plus_repair_operator":
            raise ValueError("Unexpected V4 grouping rule")
        if self.target_source != "official_solution_plus_verified_success":
            raise ValueError("Unexpected V4 target evidence source")
        if self.reference_source != "paired_verified_failure":
            raise ValueError("Unexpected V4 reference evidence source")
        if self.injection_layer != V4_LAYER_NUMBER:
            raise ValueError("V4 initial implementation is frozen at layer 24")
        if self.relative_phase_delta != 0:
            raise ValueError("V4 uses canonical pre-RoPE keys with delta zero")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class V4RepairSignature:
    """Instance-free repair abstraction for one verified construction pair."""

    experience_id: str
    sample_id: str
    experience_type: str
    problem_structure: str
    decision_point: str
    failure_mechanism: str
    repair_operator: str
    verification_operator: str
    applicable: bool
    rejection_reason: str | None
    source_provenance_sha256: str
    schema_version: str = V4_REPAIR_SIGNATURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_REPAIR_SIGNATURE_SCHEMA:
            raise ValueError("Unexpected V4 repair-signature schema")
        _require_identifier("experience_id", self.experience_id)
        _require_identifier("sample_id", self.sample_id)
        _require_identifier("experience_type", self.experience_type)
        _require_nonempty_string("source_provenance_sha256", self.source_provenance_sha256)
        for field in (
            "problem_structure",
            "decision_point",
            "failure_mechanism",
            "repair_operator",
            "verification_operator",
        ):
            validate_v4_card_text(field, getattr(self, field))
        if self.applicable and self.rejection_reason is not None:
            raise ValueError("Applicable V4 signature cannot have a rejection reason")
        if not self.applicable:
            _require_nonempty_string("rejection_reason", self.rejection_reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def signature_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class V4RepairCluster:
    """One runtime-bank proposal produced from repair signatures."""

    cluster_key: str
    title: str
    experience_type: str
    failure_mechanism: str
    repair_operator: str
    scope_summary: str
    member_experience_ids: tuple[str, ...]
    representative_experience_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier("cluster_key", self.cluster_key)
        _require_identifier("experience_type", self.experience_type)
        for field in (
            "title",
            "failure_mechanism",
            "repair_operator",
            "scope_summary",
        ):
            validate_v4_card_text(field, getattr(self, field))
        if len(set(self.member_experience_ids)) != len(self.member_experience_ids):
            raise ValueError("Cluster member IDs contain duplicates")
        if len(self.member_experience_ids) < V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4 runtime cluster has fewer than five members")
        if not (
            V4_MIN_CONSTRUCTION_EXAMPLES
            <= len(self.representative_experience_ids)
            <= V4_MAX_CONSTRUCTION_EXAMPLES
        ):
            raise ValueError("V4 cluster must choose five to ten representatives")
        if len(set(self.representative_experience_ids)) != len(
            self.representative_experience_ids
        ):
            raise ValueError("Cluster representative IDs contain duplicates")
        if not set(self.representative_experience_ids).issubset(
            self.member_experience_ids
        ):
            raise ValueError("Cluster representatives must be members")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "member_experience_ids": list(self.member_experience_ids),
            "representative_experience_ids": list(
                self.representative_experience_ids
            ),
        }


@dataclass(frozen=True)
class V4TargetProcessCard:
    scope: str
    diagnosis: str
    action: str
    verification: str
    do_not_use_when: str

    def __post_init__(self) -> None:
        for field in ("scope", "diagnosis", "action", "verification", "do_not_use_when"):
            validate_v4_card_text(f"target.{field}", getattr(self, field))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def descriptor(self) -> str:
        return "\n".join(
            (
                f"Scope: {self.scope}",
                f"Diagnosis: {self.diagnosis}",
                f"Action: {self.action}",
                f"Verification: {self.verification}",
                f"Do not use when: {self.do_not_use_when}",
            )
        )


@dataclass(frozen=True)
class V4ReferenceProcessCard:
    undesired_pattern: str
    failure_signal: str
    failure_mechanism: str
    contrast_boundary: str

    def __post_init__(self) -> None:
        for field in (
            "undesired_pattern",
            "failure_signal",
            "failure_mechanism",
            "contrast_boundary",
        ):
            validate_v4_card_text(f"reference.{field}", getattr(self, field))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def descriptor(self) -> str:
        return "\n".join(
            (
                f"Undesired pattern: {self.undesired_pattern}",
                f"Failure signal: {self.failure_signal}",
                f"Failure mechanism: {self.failure_mechanism}",
                f"Contrast boundary: {self.contrast_boundary}",
            )
        )


@dataclass(frozen=True)
class V4ProcessCard:
    cluster_key: str
    target: V4TargetProcessCard
    reference: V4ReferenceProcessCard
    support_summary: str
    target_reference_distinction: str
    schema_version: str = V4_PROCESS_CARD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_PROCESS_CARD_SCHEMA:
            raise ValueError("Unexpected V4 process-card schema")
        _require_identifier("cluster_key", self.cluster_key)
        validate_v4_card_text("support_summary", self.support_summary)
        validate_v4_card_text(
            "target_reference_distinction", self.target_reference_distinction
        )
        if self.target.descriptor == self.reference.descriptor:
            raise ValueError("V4 target and reference descriptors are identical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cluster_key": self.cluster_key,
            "target": self.target.to_dict(),
            "reference": self.reference.to_dict(),
            "support_summary": self.support_summary,
            "target_reference_distinction": self.target_reference_distinction,
        }

    @property
    def card_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class V4CardReview:
    cluster_key: str
    target_grounded: bool
    reference_grounded: bool
    process_only: bool
    target_reference_distinct: bool
    transferable: bool
    leakage_free: bool
    approve: bool
    evidence: str
    issues: tuple[str, ...]
    schema_version: str = V4_CARD_REVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_CARD_REVIEW_SCHEMA:
            raise ValueError("Unexpected V4 card-review schema")
        _require_identifier("cluster_key", self.cluster_key)
        for field in (
            "target_grounded",
            "reference_grounded",
            "process_only",
            "target_reference_distinct",
            "transferable",
            "leakage_free",
            "approve",
        ):
            _require_bool(field, getattr(self, field))
        _require_nonempty_string("review evidence", self.evidence)
        if any(not isinstance(issue, str) or not issue.strip() for issue in self.issues):
            raise ValueError("V4 card-review issues must be non-empty strings")
        component_approval = all(
            (
                self.target_grounded,
                self.reference_grounded,
                self.process_only,
                self.target_reference_distinct,
                self.transferable,
                self.leakage_free,
            )
        )
        if self.approve != component_approval or self.approve == bool(self.issues):
            raise ValueError("V4 card-review approval is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "issues": list(self.issues)}


def parse_v4_repair_signature(
    payload: Mapping[str, Any],
    *,
    experience_id: str,
    sample_id: str,
    experience_type: str,
    source_provenance_sha256: str,
) -> V4RepairSignature:
    """Parse a teacher payload while binding all identity fields deterministically."""

    applicable = _require_bool("applicable", payload.get("applicable"))
    rejection = payload.get("rejection_reason")
    if rejection is not None:
        rejection = _require_nonempty_string("rejection_reason", rejection)
    return V4RepairSignature(
        experience_id=experience_id,
        sample_id=sample_id,
        experience_type=experience_type,
        problem_structure=payload.get("problem_structure"),
        decision_point=payload.get("decision_point"),
        failure_mechanism=payload.get("failure_mechanism"),
        repair_operator=payload.get("repair_operator"),
        verification_operator=payload.get("verification_operator"),
        applicable=applicable,
        rejection_reason=rejection,
        source_provenance_sha256=source_provenance_sha256,
    )


def parse_v4_cluster_plan(
    payload: Mapping[str, Any],
    *,
    signatures: Sequence[V4RepairSignature],
) -> tuple[tuple[V4RepairCluster, ...], tuple[str, ...]]:
    """Validate full, disjoint coverage of all applicable repair signatures."""

    if payload.get("schema_version") != V4_CLUSTER_PLAN_SCHEMA:
        raise ValueError("Unexpected V4 cluster-plan payload schema")
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list):
        raise ValueError("V4 cluster payload is missing clusters")
    rejected_ids = _string_list(
        "rejected_experience_ids",
        payload.get("rejected_experience_ids"),
        allow_empty=True,
    )
    signature_by_id = {item.experience_id: item for item in signatures}
    if len(signature_by_id) != len(signatures):
        raise ValueError("V4 repair signatures have duplicate experience IDs")
    applicable_ids = {
        item.experience_id for item in signatures if item.applicable
    }
    inapplicable_ids = set(signature_by_id) - applicable_ids
    if set(rejected_ids) & inapplicable_ids:
        raise ValueError("Inapplicable signatures must not enter the clustering payload")

    clusters: list[V4RepairCluster] = []
    assigned: list[str] = []
    cluster_keys: set[str] = set()
    for index, raw in enumerate(raw_clusters):
        if not isinstance(raw, Mapping):
            raise ValueError(f"clusters[{index}] must be an object")
        members = _string_list(
            f"clusters[{index}].member_experience_ids",
            raw.get("member_experience_ids"),
        )
        representatives = _string_list(
            f"clusters[{index}].representative_experience_ids",
            raw.get("representative_experience_ids"),
        )
        unknown = set(members) - applicable_ids
        if unknown:
            raise ValueError(f"V4 cluster contains unknown/inapplicable IDs: {sorted(unknown)}")
        experiences = [signature_by_id[item] for item in members]
        experience_types = {item.experience_type for item in experiences}
        sample_ids = {item.sample_id for item in experiences}
        if len(experience_types) != 1:
            raise ValueError("V4 cluster mixes experience types")
        if len(sample_ids) < V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4 cluster has fewer than five distinct sample IDs")
        representative_samples = {
            signature_by_id[item].sample_id for item in representatives
        }
        if len(representative_samples) != len(representatives):
            raise ValueError("V4 representatives must use distinct problem IDs")
        cluster = V4RepairCluster(
            cluster_key=raw.get("cluster_key"),
            title=raw.get("title"),
            experience_type=next(iter(experience_types)),
            failure_mechanism=raw.get("failure_mechanism"),
            repair_operator=raw.get("repair_operator"),
            scope_summary=raw.get("scope_summary"),
            member_experience_ids=members,
            representative_experience_ids=representatives,
        )
        if cluster.cluster_key in cluster_keys:
            raise ValueError("V4 cluster keys are duplicated")
        cluster_keys.add(cluster.cluster_key)
        clusters.append(cluster)
        assigned.extend(members)

    if len(set(assigned)) != len(assigned):
        raise ValueError("An experience was assigned to multiple V4 clusters")
    covered = set(assigned) | set(rejected_ids)
    if covered != applicable_ids:
        missing = sorted(applicable_ids - covered)
        extra = sorted(covered - applicable_ids)
        raise ValueError(f"V4 clustering coverage mismatch: missing={missing}, extra={extra}")
    return tuple(clusters), rejected_ids


def parse_v4_process_card(
    payload: Mapping[str, Any], *, cluster_key: str
) -> V4ProcessCard:
    if payload.get("schema_version") not in {None, V4_PROCESS_CARD_SCHEMA}:
        raise ValueError("Unexpected V4 process-card payload schema")
    target = payload.get("target")
    reference = payload.get("reference")
    if not isinstance(target, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("V4 process-card payload requires target and reference objects")
    return V4ProcessCard(
        cluster_key=cluster_key,
        target=V4TargetProcessCard(
            scope=target.get("scope"),
            diagnosis=target.get("diagnosis"),
            action=target.get("action"),
            verification=target.get("verification"),
            do_not_use_when=target.get("do_not_use_when"),
        ),
        reference=V4ReferenceProcessCard(
            undesired_pattern=reference.get("undesired_pattern"),
            failure_signal=reference.get("failure_signal"),
            failure_mechanism=reference.get("failure_mechanism"),
            contrast_boundary=reference.get("contrast_boundary"),
        ),
        support_summary=payload.get("support_summary"),
        target_reference_distinction=payload.get("target_reference_distinction"),
    )


def parse_v4_card_review(
    payload: Mapping[str, Any], *, cluster_key: str
) -> V4CardReview:
    if payload.get("schema_version") not in {None, V4_CARD_REVIEW_SCHEMA}:
        raise ValueError("Unexpected V4 card-review payload schema")
    issues = _string_list("review.issues", payload.get("issues"), allow_empty=True)
    return V4CardReview(
        cluster_key=cluster_key,
        target_grounded=payload.get("target_grounded"),
        reference_grounded=payload.get("reference_grounded"),
        process_only=payload.get("process_only"),
        target_reference_distinct=payload.get("target_reference_distinct"),
        transferable=payload.get("transferable"),
        leakage_free=payload.get("leakage_free"),
        approve=payload.get("approve"),
        evidence=payload.get("evidence"),
        issues=issues,
    )


def v4_bank_id(cluster: V4RepairCluster, card: V4ProcessCard) -> str:
    """Derive a stable bank ID from semantic content and construction membership."""

    digest = canonical_json_sha256(
        {
            "cluster": cluster.to_dict(),
            "card_sha256": card.card_sha256,
        }
    )
    return f"v4-{cluster.cluster_key}-{digest[:12]}"


def build_v4_bank_record(
    *,
    cluster: V4RepairCluster,
    card: V4ProcessCard,
    review: V4CardReview,
    signatures: Sequence[V4RepairSignature],
    construction_input_sha256: str,
    profile: V4ConstructionProfile,
) -> dict[str, Any]:
    """Build one frozen, tensor-free V4 bank record."""

    if card.cluster_key != cluster.cluster_key or review.cluster_key != cluster.cluster_key:
        raise ValueError("V4 cluster/card/review identity mismatch")
    if not review.approve:
        raise ValueError("Only approved V4 cards may enter the runtime bank")
    signature_by_id = {item.experience_id: item for item in signatures}
    missing = set(cluster.member_experience_ids) - set(signature_by_id)
    if missing:
        raise ValueError(f"V4 bank record is missing signatures: {sorted(missing)}")
    construction_sample_ids = sorted(
        {signature_by_id[item].sample_id for item in cluster.member_experience_ids}
    )
    if len(construction_sample_ids) < V4_MIN_CONSTRUCTION_EXAMPLES:
        raise ValueError("V4 bank record has insufficient distinct construction support")
    bank_id = v4_bank_id(cluster, card)
    record = {
        "schema_version": V4_BANK_RECORD_SCHEMA,
        "bank_id": bank_id,
        "benchmark": "openai/gsm8k",
        "cluster": cluster.to_dict(),
        "process_card": card.to_dict(),
        "review": review.to_dict(),
        "construction": {
            "experience_ids": list(cluster.member_experience_ids),
            "representative_experience_ids": list(
                cluster.representative_experience_ids
            ),
            "sample_ids": construction_sample_ids,
            "distinct_sample_count": len(construction_sample_ids),
            "input_sha256": construction_input_sha256,
            "signature_sha256": {
                experience_id: signature_by_id[experience_id].signature_sha256
                for experience_id in sorted(cluster.member_experience_ids)
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
            "relative_phase_delta": 0,
            "attention_backend": "sdpa",
        },
        "construction_profile": profile.to_dict(),
        "construction_profile_sha256": profile.profile_sha256,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def build_v4_bank_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    profile: V4ConstructionProfile,
    inputs: Mapping[str, Any],
    teacher: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an authenticated manifest for approved tensor-free V4 records."""

    if not records:
        raise ValueError("V4 bank manifest requires at least one approved record")
    bank_ids = [str(record.get("bank_id", "")) for record in records]
    if any(not value for value in bank_ids) or len(set(bank_ids)) != len(bank_ids):
        raise ValueError("V4 bank IDs are missing or duplicated")
    for record in records:
        if record.get("schema_version") != V4_BANK_RECORD_SCHEMA:
            raise ValueError("V4 manifest received an unexpected record schema")
        stored = record.get("record_sha256")
        logical = {key: value for key, value in record.items() if key != "record_sha256"}
        if stored != canonical_json_sha256(logical):
            raise ValueError("V4 bank record hash mismatch")
        if record.get("construction_profile_sha256") != profile.profile_sha256:
            raise ValueError("V4 bank record construction profile drifted")
    manifest = {
        "schema_version": V4_BANK_MANIFEST_SCHEMA,
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
    "V4_BANK_MANIFEST_SCHEMA",
    "V4_BANK_RECORD_SCHEMA",
    "V4_CARD_PROMPT_VERSION",
    "V4_CARD_REVIEW_PROMPT_VERSION",
    "V4_CLUSTER_PROMPT_VERSION",
    "V4_LAYER_NUMBER",
    "V4_MAX_ACTIVE_STEPS",
    "V4_MAX_CONSTRUCTION_EXAMPLES",
    "V4_MAX_SELECTOR_ATTEMPTS",
    "V4_MIN_CONSTRUCTION_EXAMPLES",
    "V4_RECOVERY_LOW_TOKEN_COUNT",
    "V4_RELATIVE_PHASE_DELTA",
    "V4_SIGNATURE_PROMPT_VERSION",
    "V4_TEACHER_MODEL",
    "V4_TEACHER_TEMPERATURE",
    "V4_TEACHER_THINKING",
    "V4CardReview",
    "V4ConstructionProfile",
    "V4ProcessCard",
    "V4ReferenceProcessCard",
    "V4RepairCluster",
    "V4RepairSignature",
    "V4TargetProcessCard",
    "build_v4_bank_manifest",
    "build_v4_bank_record",
    "parse_v4_card_review",
    "parse_v4_cluster_plan",
    "parse_v4_process_card",
    "parse_v4_repair_signature",
    "v4_bank_id",
    "v4_card_leakage_reasons",
]
