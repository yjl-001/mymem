"""Deterministic V4.2 bank construction from the authenticated shortlist.

The local-direct variant is an explicit low-cost ablation.  It accepts the
already authenticated V4.2 shortlist without a second model-based semantic
audit, selects one joint medoid from each candidate's bounded evidence, and
maps that medoid's existing repair signature into a target/reference process
card.  The resulting bank is provisional and remains unqualified for online
use until side-KV and selector artifacts are compiled and calibrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_bank import (
    V4_BANK_RECORD_SCHEMA,
    V4_LAYER_NUMBER,
    V4_MIN_CONSTRUCTION_EXAMPLES,
    V4_RELATIVE_PHASE_DELTA,
    V4ProcessCard,
    V4ReferenceProcessCard,
    V4TargetProcessCard,
)
from memgen.experience.v4_2_bank import (
    V42ConstructionProfile,
    V42LocalClusterCandidate,
    V42LocalRepairAtom,
)


V4_2_LOCAL_DIRECT_PROFILE_SCHEMA = "memgen-v4.2-local-direct-profile-v1"
V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA = (
    "memgen-v4.2-local-direct-bank-manifest-v1"
)
V4_2_LOCAL_DIRECT_REPORT_SCHEMA = "memgen-v4.2-local-direct-report-v1"
V4_2_LOCAL_DIRECT_MEDOID_SCHEMA = "memgen-v4.2-local-direct-medoid-v1"

V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION = "v4.2-local-direct"
V4_2_LOCAL_DIRECT_MAX_EVIDENCE = 8
V4_2_LOCAL_DIRECT_BANK_CAP = 32
EMBEDDING_VIEW_NAMES = ("mechanism", "repair", "applicability")

V4_2_LOCAL_DIRECT_IMPLEMENTATION_PATHS = (
    "memgen/experience/phase1.py",
    "memgen/experience/v4_bank.py",
    "memgen/experience/v4_2_bank.py",
    "memgen/experience/v4_2_semantic_bank.py",
    "memgen/experience/v4_2_local_direct.py",
    "scripts/build_v4_2_local_direct_bank.py",
)

_DO_NOT_USE_WHEN = (
    "Do not apply this process when the active problem structure or decision "
    "point does not match the stated scope."
)
_CONTRAST_BOUNDARY = (
    "The target applies the repair before committing to the next inference, "
    "while the reference continues from the unresolved failure state."
)
_SUPPORT_SUMMARY = (
    "This provisional memory comes from an authenticated locally coherent "
    "cluster and uses its joint medoid as the descriptor source."
)
_TARGET_REFERENCE_DISTINCTION = (
    "The target performs the stored repair and verifies the resulting state, "
    "while the reference preserves the recurring failure process."
)


def local_direct_implementation_hashes(project_root: Path) -> dict[str, str]:
    """Return the exact implementation identity used by local-direct banks."""

    return {
        relative: file_sha256(project_root / relative)
        for relative in V4_2_LOCAL_DIRECT_IMPLEMENTATION_PATHS
    }


@dataclass(frozen=True)
class V42LocalDirectProfile:
    """Frozen, zero-API construction choices for the provisional bank."""

    admission_basis: str = "authenticated_local_shortlist"
    semantic_audit_performed: bool = False
    independent_review_performed: bool = False
    medoid_selection: str = (
        "maximum_mean_weighted_multiview_cosine_then_experience_id"
    )
    descriptor_source: str = "joint_medoid_existing_repair_signature"
    minimum_distinct_support: int = V4_MIN_CONSTRUCTION_EXAMPLES
    maximum_evidence_per_candidate: int = V4_2_LOCAL_DIRECT_MAX_EVIDENCE
    target_runtime_bank_cap: int = V4_2_LOCAL_DIRECT_BANK_CAP
    injection_layer: int = V4_LAYER_NUMBER
    relative_phase_delta: int = V4_RELATIVE_PHASE_DELTA
    target_online_only: bool = True
    auxiliary_banks_materialized: bool = False
    schema_version: str = V4_2_LOCAL_DIRECT_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_2_LOCAL_DIRECT_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4.2 local-direct profile schema")
        if self.admission_basis != "authenticated_local_shortlist":
            raise ValueError("Unexpected V4.2 local-direct admission basis")
        if self.semantic_audit_performed or self.independent_review_performed:
            raise ValueError("V4.2 local-direct must not claim semantic review")
        if self.medoid_selection != (
            "maximum_mean_weighted_multiview_cosine_then_experience_id"
        ):
            raise ValueError("Unexpected V4.2 local-direct medoid rule")
        if self.descriptor_source != "joint_medoid_existing_repair_signature":
            raise ValueError("Unexpected V4.2 local-direct descriptor source")
        if self.minimum_distinct_support != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 local-direct requires five distinct examples")
        if self.maximum_evidence_per_candidate != V4_2_LOCAL_DIRECT_MAX_EVIDENCE:
            raise ValueError("V4.2 local-direct evidence cap is frozen at eight")
        if self.target_runtime_bank_cap != V4_2_LOCAL_DIRECT_BANK_CAP:
            raise ValueError("V4.2 local-direct bank cap is frozen at thirty two")
        if self.injection_layer != V4_LAYER_NUMBER:
            raise ValueError("V4.2 local-direct remains frozen at layer twenty four")
        if self.relative_phase_delta != V4_RELATIVE_PHASE_DELTA:
            raise ValueError("V4.2 local-direct keeps relative phase delta zero")
        if not self.target_online_only or self.auxiliary_banks_materialized:
            raise ValueError("V4.2 local-direct role policy drifted")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


def _normalized_weights(profile: V42ConstructionProfile) -> dict[str, float]:
    weights = {
        "mechanism": float(profile.mechanism_weight),
        "repair": float(profile.repair_weight),
        "applicability": float(profile.applicability_weight),
    }
    if set(weights) != set(EMBEDDING_VIEW_NAMES):
        raise ValueError("V4.2 local-direct embedding views drifted")
    if any(value <= 0.0 for value in weights.values()) or not np.isclose(
        sum(weights.values()), 1.0
    ):
        raise ValueError("V4.2 local-direct embedding weights are invalid")
    return weights


def select_joint_medoid(
    evidence_ids: Sequence[str],
    *,
    atom_index: Mapping[str, int],
    embeddings: Mapping[str, np.ndarray],
    construction_profile: V42ConstructionProfile,
) -> tuple[str, dict[str, Any]]:
    """Select the most central retained evidence under the frozen three views."""

    identifiers = tuple(str(item) for item in evidence_ids)
    if not (
        V4_MIN_CONSTRUCTION_EXAMPLES
        <= len(identifiers)
        <= V4_2_LOCAL_DIRECT_MAX_EVIDENCE
    ):
        raise ValueError("V4.2 local-direct medoid requires five to eight evidence items")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("V4.2 local-direct evidence IDs contain duplicates")
    if any(identifier not in atom_index for identifier in identifiers):
        raise ValueError("V4.2 local-direct evidence is missing from local atoms")
    weights = _normalized_weights(construction_profile)
    for name in EMBEDDING_VIEW_NAMES:
        value = embeddings.get(name)
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise ValueError(f"V4.2 local-direct {name} embeddings are invalid")
        if value.shape[0] <= max(atom_index[item] for item in identifiers):
            raise ValueError(f"V4.2 local-direct {name} coverage is incomplete")

    rows: list[dict[str, Any]] = []
    for identifier in identifiers:
        index = atom_index[identifier]
        by_view: dict[str, float] = {}
        others = tuple(item for item in identifiers if item != identifier)
        for name in EMBEDDING_VIEW_NAMES:
            vector = embeddings[name][index]
            similarities = [
                float(np.dot(vector, embeddings[name][atom_index[other]]))
                for other in others
            ]
            by_view[name] = float(np.mean(similarities))
        joint = sum(weights[name] * by_view[name] for name in EMBEDDING_VIEW_NAMES)
        if not np.isfinite(joint) or any(
            not np.isfinite(value) for value in by_view.values()
        ):
            raise ValueError("V4.2 local-direct medoid score is non-finite")
        rows.append(
            {
                "experience_id": identifier,
                "mean_similarity_by_view": by_view,
                "weighted_joint_mean_similarity": joint,
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["weighted_joint_mean_similarity"]),
            str(item["experience_id"]),
        )
    )
    selected = str(rows[0]["experience_id"])
    diagnostics = {
        "schema_version": V4_2_LOCAL_DIRECT_MEDOID_SCHEMA,
        "selection_rule": V42LocalDirectProfile().medoid_selection,
        "view_weights": weights,
        "selected_experience_id": selected,
        "selected_ranked_score": rows[0],
        "ranked_scores": rows,
    }
    diagnostics["diagnostics_sha256"] = canonical_json_sha256(diagnostics)
    return selected, diagnostics


def build_local_direct_process_card(
    *, candidate_id: str, atom: V42LocalRepairAtom
) -> V4ProcessCard:
    """Map one authenticated medoid signature into a process card verbatim."""

    return V4ProcessCard(
        cluster_key=candidate_id,
        target=V4TargetProcessCard(
            scope=atom.problem_structure,
            diagnosis=atom.decision_point,
            action=atom.repair_operator,
            verification=atom.verification_operator,
            do_not_use_when=_DO_NOT_USE_WHEN,
        ),
        reference=V4ReferenceProcessCard(
            undesired_pattern=atom.failure_mechanism,
            failure_signal=atom.decision_point,
            failure_mechanism=atom.failure_mechanism,
            contrast_boundary=_CONTRAST_BOUNDARY,
        ),
        support_summary=_SUPPORT_SUMMARY,
        target_reference_distinction=_TARGET_REFERENCE_DISTINCTION,
    )


def build_local_direct_bank_record(
    *,
    candidate: V42LocalClusterCandidate,
    packet: Mapping[str, Any],
    medoid: V42LocalRepairAtom,
    medoid_diagnostics: Mapping[str, Any],
    profile: V42LocalDirectProfile,
    source_shortlist: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one provisional bank record without claiming semantic approval."""

    candidate_id = str(packet.get("candidate_id", ""))
    if candidate_id != candidate.candidate_id:
        raise ValueError("V4.2 local-direct candidate identity mismatch")
    evidence = packet.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("V4.2 local-direct packet is missing evidence")
    evidence_ids = [str(item.get("evidence_id", "")) for item in evidence]
    sample_ids = [str(item.get("sample_id", "")) for item in evidence]
    if not (
        V4_MIN_CONSTRUCTION_EXAMPLES
        <= len(evidence_ids)
        <= profile.maximum_evidence_per_candidate
    ):
        raise ValueError("V4.2 local-direct packet has invalid support")
    if (
        len(set(evidence_ids)) != len(evidence_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or any(not item for item in evidence_ids + sample_ids)
    ):
        raise ValueError("V4.2 local-direct construction support is duplicated")
    if not set(evidence_ids).issubset(candidate.member_experience_ids):
        raise ValueError("V4.2 local-direct evidence is outside its candidate")
    if medoid.experience_id not in evidence_ids:
        raise ValueError("V4.2 local-direct medoid is outside retained evidence")
    medoid_record = next(
        item for item in evidence if item["evidence_id"] == medoid.experience_id
    )
    if (
        medoid_record.get("sample_id") != medoid.sample_id
        or medoid_record.get("source_signature_sha256")
        != medoid.source_signature_sha256
    ):
        raise ValueError("V4.2 local-direct medoid provenance drifted")
    card = build_local_direct_process_card(candidate_id=candidate_id, atom=medoid)
    semantic_identity = {
        "candidate_id": candidate_id,
        "packet_sha256": packet.get("packet_sha256"),
        "medoid_experience_id": medoid.experience_id,
        "card_sha256": card.card_sha256,
        "profile_sha256": profile.profile_sha256,
    }
    bank_id = f"v42-local-direct-{canonical_json_sha256(semantic_identity)[:20]}"
    cluster = {
        "cluster_key": candidate_id,
        "title": medoid.problem_structure,
        "experience_type": "mixed_source_provenance",
        "failure_mechanism": medoid.failure_mechanism,
        "repair_operator": medoid.repair_operator,
        "scope_summary": medoid.problem_structure,
        "member_experience_ids": evidence_ids,
        "representative_experience_ids": evidence_ids,
        "source_candidate_id": candidate_id,
        "source_membership_sha256": candidate.membership_sha256,
        "source_experience_type_distribution": dict(
            candidate.source_experience_type_distribution
        ),
    }
    record = {
        "schema_version": V4_BANK_RECORD_SCHEMA,
        "construction_version": V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
        "bank_id": bank_id,
        "benchmark": "openai/gsm8k",
        "quality_tier": "provisional_local_direct",
        "cluster": cluster,
        "process_card": card.to_dict(),
        "local_direct_admission": {
            "admission_basis": profile.admission_basis,
            "semantic_audit_performed": False,
            "independent_review_performed": False,
            "candidate_accepted_without_semantic_review": True,
            "descriptor_source": profile.descriptor_source,
            "medoid": dict(medoid_diagnostics),
        },
        "construction": {
            "experience_ids": evidence_ids,
            "representative_experience_ids": evidence_ids,
            "sample_ids": sample_ids,
            "distinct_sample_count": len(sample_ids),
            "candidate_member_experience_ids": list(
                candidate.member_experience_ids
            ),
            "evidence_packet_sha256": packet.get("packet_sha256"),
            "source_signature_sha256": {
                str(item["evidence_id"]): str(item["source_signature_sha256"])
                for item in evidence
            },
            "joint_medoid_experience_id": medoid.experience_id,
            "joint_medoid_sample_id": medoid.sample_id,
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
        "construction_profile": profile.to_dict(),
        "construction_profile_sha256": profile.profile_sha256,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def build_local_direct_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    profile: V42LocalDirectProfile,
    inputs: Mapping[str, Any],
    source_signature_teacher: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the authenticated manifest for a provisional local-direct bank."""

    if not records or len(records) > profile.target_runtime_bank_cap:
        raise ValueError("V4.2 local-direct manifest has an invalid bank count")
    bank_ids = [str(item.get("bank_id", "")) for item in records]
    if any(not item for item in bank_ids) or len(set(bank_ids)) != len(bank_ids):
        raise ValueError("V4.2 local-direct bank IDs are missing or duplicated")
    for record in records:
        logical = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("schema_version") != V4_BANK_RECORD_SCHEMA
            or record.get("construction_version")
            != V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION
            or record.get("construction_profile_sha256") != profile.profile_sha256
            or record.get("record_sha256") != canonical_json_sha256(logical)
        ):
            raise ValueError("V4.2 local-direct manifest received an invalid record")
    manifest = {
        "schema_version": V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
        "construction_version": V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
        "status": "constructed_not_tensor_compiled",
        "qualified_for_online_use": False,
        "quality_tier": "provisional_local_direct",
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
        "source_signature_teacher": dict(source_signature_teacher),
        "semantic_review": {
            "performed": False,
            "reviewer": None,
            "external_api_calls_made": 0,
        },
        "api_key_read": False,
        "external_api_calls_made": 0,
        "auxiliary_banks_materialized": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


__all__ = [
    "V4_2_LOCAL_DIRECT_BANK_CAP",
    "V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA",
    "V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION",
    "V4_2_LOCAL_DIRECT_IMPLEMENTATION_PATHS",
    "V4_2_LOCAL_DIRECT_MAX_EVIDENCE",
    "V4_2_LOCAL_DIRECT_MEDOID_SCHEMA",
    "V4_2_LOCAL_DIRECT_PROFILE_SCHEMA",
    "V4_2_LOCAL_DIRECT_REPORT_SCHEMA",
    "V42LocalDirectProfile",
    "build_local_direct_bank_record",
    "build_local_direct_manifest",
    "build_local_direct_process_card",
    "local_direct_implementation_hashes",
    "select_joint_medoid",
]
