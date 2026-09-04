"""Authenticated static curation for the provisional V4.2 local-direct bank.

The local-direct bank is deliberately broad enough to test the complete V4
pipeline without a paid semantic pass.  This module derives a smaller bank
from an explicit, hash-bound review policy.  It does not edit the source bank,
does not invoke a model, and does not claim that the underlying construction
evidence received an independent semantic audit.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_bank import (
    V4_BANK_RECORD_SCHEMA,
    V4_LAYER_NUMBER,
    V4_MIN_CONSTRUCTION_EXAMPLES,
    V4_RELATIVE_PHASE_DELTA,
)
from memgen.experience.v4_2_local_direct import (
    V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
    V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
)


V4_2_CURATED_POLICY_SCHEMA = "memgen-v4.2-local-curation-policy-v1"
V4_2_CURATED_PROFILE_SCHEMA = "memgen-v4.2-local-curated-profile-v1"
V4_2_CURATED_PROFILE_RECORD_SCHEMA = (
    "memgen-v4.2-local-curated-profile-record-v1"
)
V4_2_CURATED_BANK_MANIFEST_SCHEMA = (
    "memgen-v4.2-local-curated-bank-manifest-v1"
)
V4_2_CURATED_DECISION_RECORD_SCHEMA = (
    "memgen-v4.2-local-curation-decision-v1"
)
V4_2_CURATED_REPORT_SCHEMA = "memgen-v4.2-local-curation-report-v1"
V4_2_CURATED_CONSTRUCTION_VERSION = "v4.2-local-curated"
V4_2_CURATED_QUALITY_TIER = "provisional_local_curated"

V4_2_CURATED_EXPECTED_SOURCE_RECORDS = 24
V4_2_CURATED_EXPECTED_SOURCE_EVIDENCE = 167
V4_2_CURATED_EXPECTED_RECORDS = 17
V4_2_CURATED_EXPECTED_EVIDENCE = 116
V4_2_CURATED_RETAINED_DECISIONS = frozenset({"primary", "conditional"})
V4_2_CURATED_ALLOWED_DECISIONS = frozenset(
    {*V4_2_CURATED_RETAINED_DECISIONS, "quarantine", "hard_reject"}
)
V4_2_CURATED_EXPECTED_DECISION_COUNTS = {
    "primary": 11,
    "conditional": 6,
    "quarantine": 3,
    "hard_reject": 4,
}

V4_2_CURATED_IMPLEMENTATION_PATHS = (
    "memgen/experience/phase1.py",
    "memgen/experience/v4_bank.py",
    "memgen/experience/v4_2_local_direct.py",
    "memgen/experience/v4_2_curated.py",
    "scripts/curate_v4_2_local_direct_bank.py",
    "configs/experiments/gsm8k/v4_2_local_curation_policy.json",
)


def curated_implementation_hashes(project_root: Path) -> dict[str, str]:
    """Return the exact code and policy identity used by curated banks."""

    return {
        relative: file_sha256(project_root / relative)
        for relative in V4_2_CURATED_IMPLEMENTATION_PATHS
    }


@dataclass(frozen=True)
class V42CuratedProfile:
    """Frozen claims and runtime contract for the static curated route."""

    admission_basis: str = "hash_bound_static_process_card_review"
    retention_rule: str = "retain_primary_and_conditional"
    static_process_card_review_performed: bool = True
    full_construction_evidence_review_performed: bool = False
    independent_review_performed: bool = False
    semantic_api_audit_performed: bool = False
    expected_source_record_count: int = V4_2_CURATED_EXPECTED_SOURCE_RECORDS
    expected_source_evidence_count: int = V4_2_CURATED_EXPECTED_SOURCE_EVIDENCE
    expected_retained_record_count: int = V4_2_CURATED_EXPECTED_RECORDS
    expected_retained_evidence_count: int = V4_2_CURATED_EXPECTED_EVIDENCE
    minimum_distinct_support: int = V4_MIN_CONSTRUCTION_EXAMPLES
    injection_layer: int = V4_LAYER_NUMBER
    relative_phase_delta: int = V4_RELATIVE_PHASE_DELTA
    target_online_only: bool = True
    auxiliary_banks_materialized: bool = False
    schema_version: str = V4_2_CURATED_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_2_CURATED_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4.2 curated profile schema")
        if self.admission_basis != "hash_bound_static_process_card_review":
            raise ValueError("Unexpected V4.2 curated admission basis")
        if self.retention_rule != "retain_primary_and_conditional":
            raise ValueError("Unexpected V4.2 curated retention rule")
        if not self.static_process_card_review_performed:
            raise ValueError("V4.2 curated requires a static process-card review")
        if (
            self.full_construction_evidence_review_performed
            or self.independent_review_performed
            or self.semantic_api_audit_performed
        ):
            raise ValueError("V4.2 curated must not overclaim semantic review")
        if (
            self.expected_source_record_count
            != V4_2_CURATED_EXPECTED_SOURCE_RECORDS
            or self.expected_source_evidence_count
            != V4_2_CURATED_EXPECTED_SOURCE_EVIDENCE
            or self.expected_retained_record_count != V4_2_CURATED_EXPECTED_RECORDS
            or self.expected_retained_evidence_count
            != V4_2_CURATED_EXPECTED_EVIDENCE
        ):
            raise ValueError("V4.2 curated frozen counts drifted")
        if self.minimum_distinct_support != V4_MIN_CONSTRUCTION_EXAMPLES:
            raise ValueError("V4.2 curated memories require five distinct examples")
        if self.injection_layer != V4_LAYER_NUMBER:
            raise ValueError("V4.2 curated remains frozen at layer twenty four")
        if self.relative_phase_delta != V4_RELATIVE_PHASE_DELTA:
            raise ValueError("V4.2 curated keeps relative phase delta zero")
        if not self.target_online_only or self.auxiliary_banks_materialized:
            raise ValueError("V4.2 curated role policy drifted")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


def load_and_validate_curation_policy(
    path: Path, *, source_manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """Load a complete policy bound to the authenticated 24-record source."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V4.2 curation policy must be a JSON object")
    if value.get("schema_version") != V4_2_CURATED_POLICY_SCHEMA:
        raise ValueError("Unexpected V4.2 curation-policy schema")
    if source_manifest.get("schema_version") != V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA:
        raise ValueError("V4.2 curation requires a local-direct source manifest")
    bindings = (
        ("source_manifest_sha256", source_manifest.get("manifest_sha256")),
        ("source_profile_sha256", source_manifest.get("profile_sha256")),
        ("source_record_order_sha256", source_manifest.get("record_order_sha256")),
        ("expected_source_record_count", source_manifest.get("record_count")),
    )
    for policy_field, source_value in bindings:
        if value.get(policy_field) != source_value:
            raise ValueError(f"V4.2 curation policy {policy_field} binding drifted")
    if (
        value.get("expected_source_record_count")
        != V4_2_CURATED_EXPECTED_SOURCE_RECORDS
        or value.get("expected_source_evidence_count")
        != V4_2_CURATED_EXPECTED_SOURCE_EVIDENCE
        or value.get("expected_retained_record_count")
        != V4_2_CURATED_EXPECTED_RECORDS
        or value.get("expected_retained_evidence_count")
        != V4_2_CURATED_EXPECTED_EVIDENCE
    ):
        raise ValueError("V4.2 curation policy frozen counts drifted")
    if set(value.get("retained_decisions", ())) != V4_2_CURATED_RETAINED_DECISIONS:
        raise ValueError("V4.2 curation policy retention rule drifted")
    if value.get("expected_decision_counts") != V4_2_CURATED_EXPECTED_DECISION_COUNTS:
        raise ValueError("V4.2 curation policy decision counts drifted")

    raw_decisions = value.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("V4.2 curation policy decisions must be a list")
    decisions: list[dict[str, str]] = []
    for raw in raw_decisions:
        if not isinstance(raw, Mapping):
            raise ValueError("V4.2 curation policy decision is not an object")
        decision = {
            "bank_id": str(raw.get("bank_id", "")).strip(),
            "decision": str(raw.get("decision", "")).strip(),
            "reason": str(raw.get("reason", "")).strip(),
            "semantic_category": str(raw.get("semantic_category", "")).strip(),
        }
        if not all(decision.values()):
            raise ValueError("V4.2 curation policy has an incomplete decision")
        if decision["decision"] not in V4_2_CURATED_ALLOWED_DECISIONS:
            raise ValueError("V4.2 curation policy has an unknown decision")
        decisions.append(decision)
    policy_ids = [item["bank_id"] for item in decisions]
    source_ids = [str(item) for item in source_manifest.get("bank_ids", ())]
    if policy_ids != source_ids or len(set(policy_ids)) != len(policy_ids):
        raise ValueError("V4.2 curation policy must cover source order exactly once")
    if dict(Counter(item["decision"] for item in decisions)) != {
        key: value for key, value in V4_2_CURATED_EXPECTED_DECISION_COUNTS.items()
    }:
        raise ValueError("V4.2 curation policy observed decision counts drifted")
    return value, tuple(decisions)


def build_curated_record(
    *,
    source_record: Mapping[str, Any],
    decision: Mapping[str, str],
    policy_sha256: str,
    profile: V42CuratedProfile,
) -> dict[str, Any]:
    """Copy one retained card while making the curation decision explicit."""

    if decision.get("decision") not in V4_2_CURATED_RETAINED_DECISIONS:
        raise ValueError("Cannot build a curated record from an excluded decision")
    if source_record.get("bank_id") != decision.get("bank_id"):
        raise ValueError("V4.2 curation record/decision identity mismatch")
    if source_record.get("schema_version") != V4_BANK_RECORD_SCHEMA:
        raise ValueError("Unexpected V4.2 source bank-record schema")
    source_hash = source_record.get("record_sha256")
    source_logical = {
        key: value for key, value in source_record.items() if key != "record_sha256"
    }
    if source_hash != canonical_json_sha256(source_logical):
        raise ValueError("V4.2 source bank-record hash mismatch")
    if source_record.get("construction_version") != V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION:
        raise ValueError("V4.2 curated source construction version drifted")

    record = deepcopy(dict(source_record))
    record.pop("record_sha256", None)
    record["construction_version"] = V4_2_CURATED_CONSTRUCTION_VERSION
    record["quality_tier"] = V4_2_CURATED_QUALITY_TIER
    record["curation"] = {
        "decision": str(decision["decision"]),
        "reason": str(decision["reason"]),
        "semantic_category": str(decision["semantic_category"]),
        "policy_sha256": policy_sha256,
        "source_record_sha256": source_hash,
        "bank_identity_preserved": True,
        "static_process_card_review_performed": True,
        "full_construction_evidence_review_performed": False,
        "independent_review_performed": False,
        "semantic_api_audit_performed": False,
    }
    record["curation_profile"] = profile.to_dict()
    record["curation_profile_sha256"] = profile.profile_sha256
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def build_curated_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    source_records_path: Path,
    source_profile_path: Path,
    source_report_path: Path,
    policy_path: Path,
    policy_sha256: str,
    decisions: Sequence[Mapping[str, str]],
    profile: V42CuratedProfile,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the tensor-free manifest consumed by V4 side-KV compilation."""

    bank_ids = [str(item.get("bank_id", "")) for item in records]
    decision_ids = [str(item.get("bank_id", "")) for item in decisions]
    source_ids = [str(item) for item in source_manifest.get("bank_ids", ())]
    decision_counts = dict(Counter(item.get("decision") for item in decisions))
    retained_ids = [
        str(item["bank_id"])
        for item in decisions
        if item["decision"] in V4_2_CURATED_RETAINED_DECISIONS
    ]
    if (
        decision_ids != source_ids
        or len(set(decision_ids)) != len(decision_ids)
        or decision_counts != V4_2_CURATED_EXPECTED_DECISION_COUNTS
    ):
        raise ValueError("V4.2 curated manifest decision coverage drifted")
    if (
        bank_ids != retained_ids
        or len(set(bank_ids)) != len(bank_ids)
        or len(records) != profile.expected_retained_record_count
    ):
        raise ValueError("V4.2 curated manifest retained order/count drifted")
    evidence_count = 0
    for record in records:
        logical = {key: value for key, value in record.items() if key != "record_sha256"}
        sample_ids = record.get("construction", {}).get("sample_ids", ())
        if (
            record.get("schema_version") != V4_BANK_RECORD_SCHEMA
            or record.get("construction_version")
            != V4_2_CURATED_CONSTRUCTION_VERSION
            or record.get("curation_profile_sha256") != profile.profile_sha256
            or record.get("record_sha256") != canonical_json_sha256(logical)
            or not isinstance(sample_ids, list)
            or len(sample_ids) < V4_MIN_CONSTRUCTION_EXAMPLES
            or len(sample_ids) != len(set(sample_ids))
            or record.get("construction", {}).get("distinct_sample_count")
            != len(sample_ids)
        ):
            raise ValueError("V4.2 curated manifest received an invalid record")
        evidence_count += len(sample_ids)
    if evidence_count != profile.expected_retained_evidence_count:
        raise ValueError("V4.2 curated evidence count drifted")

    excluded_ids = [
        str(item["bank_id"])
        for item in decisions
        if item["decision"] not in V4_2_CURATED_RETAINED_DECISIONS
    ]
    source_inputs = deepcopy(dict(source_manifest.get("inputs", {})))
    if not all(
        isinstance(source_inputs.get(field), str) and source_inputs[field]
        for field in ("experiences_sha256", "split_manifest_sha256")
    ):
        raise ValueError("V4.2 curated source data bindings are incomplete")
    source_inputs["repository"] = {
        "implementation_sha256": dict(implementation_sha256)
    }
    manifest = {
        "schema_version": V4_2_CURATED_BANK_MANIFEST_SCHEMA,
        "construction_version": V4_2_CURATED_CONSTRUCTION_VERSION,
        "status": "constructed_not_tensor_compiled",
        "qualified_for_online_use": False,
        "quality_tier": V4_2_CURATED_QUALITY_TIER,
        "benchmark": "openai/gsm8k",
        "record_count": len(records),
        "evidence_count": evidence_count,
        "bank_ids": bank_ids,
        "record_order_sha256": canonical_json_sha256(bank_ids),
        "record_sha256": {
            str(item["bank_id"]): str(item["record_sha256"]) for item in records
        },
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "inputs": source_inputs,
        "source_local_direct": {
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_file_sha256": file_sha256(source_manifest_path),
            "manifest_logical_sha256": source_manifest["manifest_sha256"],
            "profile_sha256": source_manifest["profile_sha256"],
            "record_order_sha256": source_manifest["record_order_sha256"],
            "record_count": source_manifest["record_count"],
            "bank_records_file_sha256": file_sha256(source_records_path),
            "construction_profile_file_sha256": file_sha256(source_profile_path),
            "local_direct_report_file_sha256": file_sha256(source_report_path),
        },
        "curation": {
            "policy_path": str(policy_path.resolve()),
            "policy_file_sha256": file_sha256(policy_path),
            "policy_sha256": policy_sha256,
            "retained_decisions": sorted(V4_2_CURATED_RETAINED_DECISIONS),
            "decision_counts": decision_counts,
            "retained_bank_ids": retained_ids,
            "excluded_bank_ids": excluded_ids,
        },
        "semantic_review": {
            "static_process_card_review_performed": True,
            "full_construction_evidence_review_performed": False,
            "independent_review_performed": False,
            "semantic_api_audit_performed": False,
        },
        "source_signature_teacher": deepcopy(
            dict(source_manifest.get("source_signature_teacher", {}))
        ),
        "api_key_read": False,
        "external_api_calls_made": 0,
        "auxiliary_banks_materialized": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


__all__ = [
    "V4_2_CURATED_ALLOWED_DECISIONS",
    "V4_2_CURATED_BANK_MANIFEST_SCHEMA",
    "V4_2_CURATED_CONSTRUCTION_VERSION",
    "V4_2_CURATED_DECISION_RECORD_SCHEMA",
    "V4_2_CURATED_EXPECTED_DECISION_COUNTS",
    "V4_2_CURATED_EXPECTED_EVIDENCE",
    "V4_2_CURATED_EXPECTED_RECORDS",
    "V4_2_CURATED_EXPECTED_SOURCE_EVIDENCE",
    "V4_2_CURATED_EXPECTED_SOURCE_RECORDS",
    "V4_2_CURATED_IMPLEMENTATION_PATHS",
    "V4_2_CURATED_POLICY_SCHEMA",
    "V4_2_CURATED_PROFILE_RECORD_SCHEMA",
    "V4_2_CURATED_PROFILE_SCHEMA",
    "V4_2_CURATED_QUALITY_TIER",
    "V4_2_CURATED_REPORT_SCHEMA",
    "V4_2_CURATED_RETAINED_DECISIONS",
    "V42CuratedProfile",
    "build_curated_manifest",
    "build_curated_record",
    "curated_implementation_hashes",
    "load_and_validate_curation_policy",
]
