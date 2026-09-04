#!/usr/bin/env python3
"""Build a provisional V4.2 bank directly from the authenticated shortlist.

This command is deliberately model-free and network-free.  It authenticates
the completed semantic preflight, selects one deterministic joint medoid per
retained candidate, maps existing signature fields into target/reference
cards, and emits a tensor-free bank.  It never performs or claims a semantic
audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl
from memgen.experience.v4_2_bank import (
    V42ShortlistProfile,
)
from memgen.experience.v4_2_local_direct import (
    V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
    V4_2_LOCAL_DIRECT_REPORT_SCHEMA,
    V42LocalDirectProfile,
    build_local_direct_bank_record,
    build_local_direct_manifest,
    local_direct_implementation_hashes,
    select_joint_medoid,
)
from memgen.experience.v4_2_semantic_bank import (
    V4_2_EVIDENCE_PACKET_SCHEMA,
    V4_2_PAID_PLAN_SCHEMA,
    V4_2_PAID_PREFLIGHT_SCHEMA,
    V42SemanticConstructionProfile,
)
from scripts.build_v4_1_repair_bank import load_authenticated_signatures
from scripts.build_v4_repair_bank import _validate_split_manifest, load_v4_experiences
from scripts.select_v4_2_bank_candidates import (
    SELECTED_CANDIDATE_SCHEMA,
    _profile_record as shortlist_profile_record,
    load_authenticated_local_construction,
    validate_completed_output,
)


SEMANTIC_PROFILE_RECORD_SCHEMA = "memgen-v4.2-semantic-profile-record-v1"
POLICY_EXCLUSION_RECORD_SCHEMA = "memgen-v4.2-policy-exclusion-record-v1"
LOCAL_DIRECT_PROFILE_RECORD_SCHEMA = "memgen-v4.2-local-direct-profile-record-v1"
LOCAL_DIRECT_SELECTION_RECORD_SCHEMA = "memgen-v4.2-local-direct-selection-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-signatures", type=Path, required=True)
    parser.add_argument("--source-construction-profile", type=Path, required=True)
    parser.add_argument("--local-construction-dir", type=Path, required=True)
    parser.add_argument("--shortlist-dir", type=Path, required=True)
    parser.add_argument("--semantic-preflight-dir", type=Path, required=True)
    parser.add_argument("--semantic-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _logical_hash(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: item for key, item in value.items() if key != field})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selected_records(path: Path) -> tuple[dict[str, Any], ...]:
    values = tuple(dict(item) for item in iter_jsonl(path))
    for rank, record in enumerate(values, start=1):
        if record.get("schema_version") != SELECTED_CANDIDATE_SCHEMA:
            raise ValueError("Unexpected V4.2 selected-candidate schema")
        if record.get("record_sha256") != _logical_hash(record, "record_sha256"):
            raise ValueError("V4.2 selected-candidate hash mismatch")
        if record.get("selection_rank") != rank:
            raise ValueError("V4.2 selected-candidate rank drifted")
    if not values:
        raise ValueError("V4.2 shortlist contains no selected candidates")
    return values


def _authenticate_shortlist(
    directory: Path, *, local_source_info: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    profile_record = _load_json(directory / "construction_profile.json")
    profile = V42ShortlistProfile(**profile_record.get("profile", {}))
    expected = shortlist_profile_record(profile, source_info=local_source_info)
    preflight = validate_completed_output(directory, expected_profile_record=expected)
    if preflight is None:
        raise ValueError("V4.2 local-direct requires a completed shortlist")
    manifest = _load_json(directory / "synthesis_shortlist_manifest.json")
    return (
        _selected_records(directory / "selected_synthesis_candidates.jsonl"),
        manifest,
        preflight,
    )


def _authenticate_semantic_preflight(
    directory: Path,
    *,
    semantic_policy_path: Path,
    selected_records: Sequence[Mapping[str, Any]],
    shortlist_manifest: Mapping[str, Any],
    shortlist_preflight: Mapping[str, Any],
    candidates: Mapping[str, Any],
    atoms: Mapping[str, Any],
    signatures: Mapping[str, Any],
    experiences: Mapping[str, Mapping[str, Any]],
    experiences_path: Path,
    split_manifest_path: Path,
    source_signature_info: Mapping[str, Any],
    local_source_info: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    required = (
        "construction_profile.json",
        "semantic_evidence_packets.jsonl",
        "policy_exclusions.jsonl",
        "paid_stage_plan.json",
        "api_preflight_report.json",
    )
    for name in required:
        if not (directory / name).is_file():
            raise ValueError(f"Missing V4.2 semantic preflight artifact: {name}")

    profile_record = _load_json(directory / "construction_profile.json")
    if profile_record.get("schema_version") != SEMANTIC_PROFILE_RECORD_SCHEMA:
        raise ValueError("Unexpected V4.2 semantic profile-record schema")
    semantic_profile = V42SemanticConstructionProfile(
        **profile_record.get("profile", {})
    )
    if profile_record.get("profile_sha256") != semantic_profile.profile_sha256:
        raise ValueError("V4.2 semantic profile hash mismatch")

    plan = _load_json(directory / "paid_stage_plan.json")
    if plan.get("schema_version") != V4_2_PAID_PLAN_SCHEMA:
        raise ValueError("Unexpected V4.2 paid-stage plan schema")
    if plan.get("plan_sha256") != _logical_hash(plan, "plan_sha256"):
        raise ValueError("V4.2 paid-stage plan hash mismatch")
    preflight = _load_json(directory / "api_preflight_report.json")
    if preflight.get("schema_version") != V4_2_PAID_PREFLIGHT_SCHEMA:
        raise ValueError("Unexpected V4.2 semantic preflight-report schema")
    if preflight.get("report_sha256") != _logical_hash(preflight, "report_sha256"):
        raise ValueError("V4.2 semantic preflight report hash mismatch")
    if (
        preflight.get("status") != "semantic_evidence_ready_api_not_started"
        or preflight.get("api_key_read") is not False
        or preflight.get("external_api_calls_made") != 0
        or preflight.get("automatic_paid_stage_transition") is not False
        or preflight.get("qualified_for_online_use") is not False
    ):
        raise ValueError("V4.2 semantic preflight is not a zero-API input basis")
    if (
        preflight.get("paid_stage_plan_sha256") != plan["plan_sha256"]
        or preflight.get("profile_sha256") != semantic_profile.profile_sha256
    ):
        raise ValueError("V4.2 semantic preflight plan/profile binding drifted")

    policy = _load_json(semantic_policy_path)
    policy_sha256 = canonical_json_sha256(policy)
    authenticated_candidate_exclusions = {
        str(item.get("candidate_id", ""))
        for item in policy.get("candidate_exclusions", ())
        if isinstance(item, Mapping)
    }
    if (
        preflight.get("semantic_policy_sha256") != policy_sha256
        or profile_record.get("inputs", {})
        .get("semantic_policy", {})
        .get("file_sha256")
        != file_sha256(semantic_policy_path)
    ):
        raise ValueError("V4.2 semantic policy binding drifted")

    source_shortlist = profile_record.get("inputs", {}).get("source_shortlist", {})
    expected_shortlist = {
        "profile_sha256": shortlist_manifest.get("profile_sha256"),
        "manifest_sha256": shortlist_manifest.get("manifest_sha256"),
        "report_sha256": shortlist_preflight.get("report_sha256"),
    }
    if source_shortlist != expected_shortlist or preflight.get(
        "source_shortlist"
    ) != expected_shortlist:
        raise ValueError("V4.2 semantic source-shortlist binding drifted")

    inputs = profile_record.get("inputs", {})
    if (
        inputs.get("experiences", {}).get("sha256")
        != file_sha256(experiences_path)
        or inputs.get("experiences", {}).get("count") != len(experiences)
        or inputs.get("split_manifest", {}).get("file_sha256")
        != file_sha256(split_manifest_path)
        or inputs.get("source_signatures") != dict(source_signature_info)
        or inputs.get("local_construction") != dict(local_source_info)
    ):
        raise ValueError("V4.2 semantic upstream input binding drifted")

    exclusions = tuple(dict(item) for item in iter_jsonl(directory / "policy_exclusions.jsonl"))
    candidate_exclusions: set[str] = set()
    evidence_exclusions: dict[str, set[str]] = {}
    for record in exclusions:
        if record.get("schema_version") != POLICY_EXCLUSION_RECORD_SCHEMA:
            raise ValueError("Unexpected V4.2 policy-exclusion schema")
        if record.get("record_sha256") != _logical_hash(record, "record_sha256"):
            raise ValueError("V4.2 policy-exclusion hash mismatch")
        candidate_id = str(record.get("candidate_id", ""))
        if record.get("level") == "candidate":
            if candidate_id in candidate_exclusions:
                raise ValueError("V4.2 policy repeats a candidate exclusion")
            candidate_exclusions.add(candidate_id)
        elif record.get("level") == "evidence":
            experience_id = str(record.get("experience_id", ""))
            if not experience_id:
                raise ValueError("V4.2 policy evidence exclusion is missing an ID")
            bucket = evidence_exclusions.setdefault(candidate_id, set())
            if experience_id in bucket:
                raise ValueError("V4.2 policy repeats an evidence exclusion")
            bucket.add(experience_id)
        else:
            raise ValueError("V4.2 policy exclusion has an unknown level")

    selected_ids = [str(item["candidate"]["candidate_id"]) for item in selected_records]
    if not candidate_exclusions.issubset(selected_ids):
        raise ValueError("V4.2 candidate exclusion is outside the shortlist")
    if not authenticated_candidate_exclusions.issubset(candidate_exclusions):
        raise ValueError("V4.2 authenticated candidate exclusion was not applied")
    expected_packet_ids = [
        candidate_id for candidate_id in selected_ids if candidate_id not in candidate_exclusions
    ]
    packets = tuple(
        dict(item) for item in iter_jsonl(directory / "semantic_evidence_packets.jsonl")
    )
    if [str(item.get("candidate_id", "")) for item in packets] != expected_packet_ids:
        raise ValueError("V4.2 semantic packet candidate order or coverage drifted")

    plan_packets: dict[str, str] = {}
    for batch in plan.get("batches", ()):
        packet_hashes = batch.get("packet_sha256", {})
        if not isinstance(packet_hashes, Mapping):
            raise ValueError("V4.2 paid plan has invalid packet bindings")
        for candidate_id, packet_sha256 in packet_hashes.items():
            if candidate_id in plan_packets:
                raise ValueError("V4.2 paid plan repeats a candidate")
            plan_packets[str(candidate_id)] = str(packet_sha256)

    selected_by_id = {
        str(item["candidate"]["candidate_id"]): item for item in selected_records
    }
    evidence_count = 0
    distribution: Counter[int] = Counter()
    for packet in packets:
        if packet.get("schema_version") != V4_2_EVIDENCE_PACKET_SCHEMA:
            raise ValueError("Unexpected V4.2 semantic evidence-packet schema")
        if packet.get("packet_sha256") != _logical_hash(packet, "packet_sha256"):
            raise ValueError("V4.2 semantic evidence-packet hash mismatch")
        candidate_id = str(packet["candidate_id"])
        selected = selected_by_id[candidate_id]
        candidate = candidates.get(candidate_id)
        if candidate is None or candidate.to_dict() != selected["candidate"]:
            raise ValueError("V4.2 semantic packet candidate drifted")
        if (
            packet.get("selection_rank") != selected["selection_rank"]
            or packet.get("membership_sha256") != candidate.membership_sha256
            or packet.get("semantic_policy_sha256") != policy_sha256
            or packet.get("source_shortlist_record_sha256")
            != selected.get("record_sha256")
            or plan_packets.get(candidate_id) != packet["packet_sha256"]
        ):
            raise ValueError("V4.2 semantic packet provenance drifted")
        evidence = packet.get("evidence")
        if not isinstance(evidence, list) or packet.get("evidence_count") != len(evidence):
            raise ValueError("V4.2 semantic packet evidence count drifted")
        if not 5 <= len(evidence) <= 8:
            raise ValueError("V4.2 local-direct requires five to eight evidence items")
        evidence_ids = [str(item.get("evidence_id", "")) for item in evidence]
        sample_ids = [str(item.get("sample_id", "")) for item in evidence]
        if len(set(evidence_ids)) != len(evidence_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("V4.2 semantic packet repeats evidence or samples")
        if set(evidence_ids) & evidence_exclusions.get(candidate_id, set()):
            raise ValueError("V4.2 semantic packet recovered policy-excluded evidence")
        for item in evidence:
            experience_id = str(item["evidence_id"])
            atom = atoms.get(experience_id)
            signature = signatures.get(experience_id)
            experience = experiences.get(experience_id)
            if atom is None or signature is None or experience is None:
                raise ValueError("V4.2 semantic evidence is outside authenticated inputs")
            expected_signature = {
                "problem_structure": atom.problem_structure,
                "decision_point": atom.decision_point,
                "failure_mechanism": atom.failure_mechanism,
                "repair_operator": atom.repair_operator,
                "verification_operator": atom.verification_operator,
            }
            if (
                item.get("sample_id") != atom.sample_id
                or item.get("sample_id") != signature.sample_id
                or item.get("sample_id") != experience["sample_id"]
                or item.get("source_experience_type")
                != atom.source_experience_type
                or item.get("semantic_signature") != expected_signature
                or item.get("source_signature_sha256") != signature.signature_sha256
                or item.get("source_provenance_sha256")
                != experience["provenance_sha256"]
            ):
                raise ValueError("V4.2 semantic evidence source binding drifted")
        evidence_count += len(evidence)
        distribution[len(evidence)] += 1

    if set(plan_packets) != set(expected_packet_ids):
        raise ValueError("V4.2 paid plan packet coverage drifted")
    expected_distribution = {
        str(key): value for key, value in sorted(distribution.items())
    }
    if (
        preflight.get("source_selected_candidate_count") != len(selected_ids)
        or preflight.get("authenticated_policy_excluded_candidate_count")
        != len(authenticated_candidate_exclusions)
        or preflight.get("preflight_excluded_candidate_count")
        != len(candidate_exclusions)
        or preflight.get("planned_candidate_count") != len(packets)
        or preflight.get("evidence_count") != evidence_count
        or preflight.get("evidence_count_distribution") != expected_distribution
        or preflight.get("policy_excluded_evidence_count")
        != sum(len(values) for values in evidence_exclusions.values())
    ):
        raise ValueError("V4.2 semantic preflight counts drifted")
    return packets, exclusions, profile_record, plan, preflight


def build_outputs(
    *,
    packets: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Any],
    atoms: Sequence[Any],
    embeddings: Mapping[str, Any],
    construction_profile: Any,
    source_shortlist: Mapping[str, Any],
    profile: V42LocalDirectProfile,
    manifest_inputs: Mapping[str, Any],
    source_signature_teacher: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
]:
    atom_index = {item.experience_id: index for index, item in enumerate(atoms)}
    atom_by_id = {item.experience_id: item for item in atoms}
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for packet in packets:
        candidate_id = str(packet["candidate_id"])
        evidence_ids = [str(item["evidence_id"]) for item in packet["evidence"]]
        medoid_id, diagnostics = select_joint_medoid(
            evidence_ids,
            atom_index=atom_index,
            embeddings=embeddings,
            construction_profile=construction_profile,
        )
        record = build_local_direct_bank_record(
            candidate=candidates[candidate_id],
            packet=packet,
            medoid=atom_by_id[medoid_id],
            medoid_diagnostics=diagnostics,
            profile=profile,
            source_shortlist=source_shortlist,
        )
        records.append(record)
        selection = {
            "schema_version": LOCAL_DIRECT_SELECTION_RECORD_SCHEMA,
            "candidate_id": candidate_id,
            "selection_rank": packet["selection_rank"],
            "bank_id": record["bank_id"],
            "evidence_packet_sha256": packet["packet_sha256"],
            "medoid": diagnostics,
        }
        selection["record_sha256"] = canonical_json_sha256(selection)
        selections.append(selection)
    manifest = build_local_direct_manifest(
        records=records,
        profile=profile,
        inputs=manifest_inputs,
        source_signature_teacher=source_signature_teacher,
    )
    report = {
        "schema_version": V4_2_LOCAL_DIRECT_REPORT_SCHEMA,
        "construction_version": V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
        "status": "local_direct_bank_constructed_not_tensor_compiled",
        "quality_tier": "provisional_local_direct",
        "qualified_for_online_use": False,
        "admission_basis": profile.admission_basis,
        "semantic_audit_performed": False,
        "independent_review_performed": False,
        "api_key_read": False,
        "external_api_calls_made": 0,
        "source_candidate_count": len(packets),
        "bank_record_count": len(records),
        "evidence_count": sum(len(item["evidence"]) for item in packets),
        "joint_medoid_count": len(selections),
        "bank_manifest_sha256": manifest["manifest_sha256"],
        "record_order_sha256": manifest["record_order_sha256"],
        "profile_sha256": profile.profile_sha256,
        "next_stage": (
            "compile provisional layer twenty four target and reference side KV, "
            "then require selector-anchor qualification"
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return tuple(records), tuple(selections), manifest, report


def _write_or_validate_outputs(
    output_dir: Path,
    *,
    profile_record: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    resume: bool,
) -> None:
    values: dict[str, Any] = {
        "construction_profile.json": dict(profile_record),
        "medoid_selections.jsonl": tuple(selections),
        "bank_records.jsonl": tuple(records),
        "bank_manifest.json": dict(manifest),
        "local_direct_report.json": dict(report),
    }
    existing = [name for name in values if (output_dir / name).exists()]
    if existing:
        if not resume:
            raise ValueError("V4.2 local-direct output exists; pass --resume")
        if len(existing) != len(values):
            raise ValueError("V4.2 local-direct output is incomplete")
        for name, expected in values.items():
            path = output_dir / name
            actual: Any = (
                tuple(dict(item) for item in iter_jsonl(path))
                if name.endswith(".jsonl")
                else _load_json(path)
            )
            if actual != expected:
                raise ValueError(f"V4.2 local-direct output drifted: {name}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "construction_profile.json", profile_record)
    _write_jsonl(output_dir / "medoid_selections.jsonl", selections)
    _write_jsonl(output_dir / "bank_records.jsonl", records)
    _write_json(output_dir / "bank_manifest.json", manifest)
    _write_json(output_dir / "local_direct_report.json", report)


def main() -> None:
    args = parse_args()
    for owner in (
        "experiences",
        "split_manifest",
        "source_signatures",
        "source_construction_profile",
        "semantic_policy",
    ):
        path = getattr(args, owner).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Missing V4.2 local-direct input: {path}")
        setattr(args, owner, path)
    for owner in ("local_construction_dir", "shortlist_dir", "semantic_preflight_dir"):
        path = getattr(args, owner).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Missing V4.2 local-direct directory: {path}")
        setattr(args, owner, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    protected = {
        PROJECT_ROOT.resolve(),
        args.experiences.parent,
        args.local_construction_dir,
        args.shortlist_dir,
        args.semantic_preflight_dir,
    }
    if args.output_dir in protected:
        raise ValueError("V4.2 local-direct output must differ from all input directories")

    split_manifest = _validate_split_manifest(
        args.split_manifest, dataset_revision=args.dataset_revision
    )
    experiences = load_v4_experiences(
        args.experiences, split_manifest=split_manifest
    )
    signatures, source_signature_info = load_authenticated_signatures(
        args.source_signatures,
        source_profile_path=args.source_construction_profile,
        experiences=experiences,
    )
    (
        construction_profile,
        _local_plan,
        atoms,
        candidate_values,
        _review_packets,
        embeddings,
        local_source_info,
    ) = load_authenticated_local_construction(args.local_construction_dir)
    local_profile_record = _load_json(
        args.local_construction_dir / "construction_profile.json"
    )
    if local_profile_record.get("source_signatures") != source_signature_info:
        raise ValueError("V4.2 local construction uses different source signatures")
    selected_records, shortlist_manifest, shortlist_preflight = _authenticate_shortlist(
        args.shortlist_dir, local_source_info=local_source_info
    )
    candidates = {item.candidate_id: item for item in candidate_values}
    atoms_by_id = {item.experience_id: item for item in atoms}
    signatures_by_id = {item.experience_id: item for item in signatures}
    experiences_by_id = {str(item["experience_id"]): item for item in experiences}
    packets, exclusions, semantic_profile_record, plan, preflight = (
        _authenticate_semantic_preflight(
            args.semantic_preflight_dir,
            semantic_policy_path=args.semantic_policy,
            selected_records=selected_records,
            shortlist_manifest=shortlist_manifest,
            shortlist_preflight=shortlist_preflight,
            candidates=candidates,
            atoms=atoms_by_id,
            signatures=signatures_by_id,
            experiences=experiences_by_id,
            experiences_path=args.experiences,
            split_manifest_path=args.split_manifest,
            source_signature_info=source_signature_info,
            local_source_info=local_source_info,
        )
    )
    del exclusions

    profile = V42LocalDirectProfile()
    implementation = local_direct_implementation_hashes(PROJECT_ROOT)
    source_shortlist = semantic_profile_record["inputs"]["source_shortlist"]
    manifest_inputs = {
        "experiences_path": str(args.experiences),
        "experiences_sha256": file_sha256(args.experiences),
        "split_manifest_path": str(args.split_manifest),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "source_signatures": dict(source_signature_info),
        "local_construction": dict(local_source_info),
        "source_shortlist": dict(source_shortlist),
        "semantic_preflight": {
            "directory": str(args.semantic_preflight_dir),
            "report_sha256": preflight["report_sha256"],
            "paid_stage_plan_sha256": plan["plan_sha256"],
            "semantic_policy_sha256": preflight["semantic_policy_sha256"],
            "evidence_packet_file_sha256": file_sha256(
                args.semantic_preflight_dir / "semantic_evidence_packets.jsonl"
            ),
        },
        "repository": {"implementation_sha256": implementation},
    }
    records, selections, manifest, report = build_outputs(
        packets=packets,
        candidates=candidates,
        atoms=atoms,
        embeddings=embeddings,
        construction_profile=construction_profile,
        source_shortlist=source_shortlist,
        profile=profile,
        manifest_inputs=manifest_inputs,
        source_signature_teacher=source_signature_info["teacher"],
    )
    profile_record = {
        "schema_version": LOCAL_DIRECT_PROFILE_RECORD_SCHEMA,
        "construction_version": V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
        "stage": "authenticated_shortlist_to_joint_medoid_bank",
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "semantic_audit_performed": False,
        "independent_review_performed": False,
        "api_key_read": False,
        "external_api_calls_made": 0,
        "inputs": manifest_inputs,
    }
    profile_record["record_sha256"] = canonical_json_sha256(profile_record)
    _write_or_validate_outputs(
        args.output_dir,
        profile_record=profile_record,
        records=records,
        selections=selections,
        manifest=manifest,
        report=report,
        resume=args.resume,
    )
    print(
        f"[v4.2-local-direct] PASS banks={len(records)} "
        f"evidence={report['evidence_count']} medoids={len(selections)} "
        "semantic_audit=false api_key_read=false api_calls=0",
        flush=True,
    )
    print(
        f"[v4.2-local-direct] report={args.output_dir / 'local_direct_report.json'}",
        flush=True,
    )
    print(
        f"[v4.2-local-direct] manifest={args.output_dir / 'bank_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4.2-local-direct] error: {exc}", file=sys.stderr, flush=True)
        raise
