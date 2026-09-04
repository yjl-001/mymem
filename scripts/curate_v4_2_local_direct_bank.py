#!/usr/bin/env python3
"""Extract the authenticated 17-record V4.2 curated bank without an API.

The source 24-record local-direct bank remains immutable.  This command
authenticates it, requires the repository's hash-bound decision policy to
cover every source record in exact order, and writes a new tensor-free bank
plus a complete retained/excluded audit trail.
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
from memgen.experience.v4_2_curated import (
    V4_2_CURATED_CONSTRUCTION_VERSION,
    V4_2_CURATED_DECISION_RECORD_SCHEMA,
    V4_2_CURATED_EXPECTED_SOURCE_EVIDENCE,
    V4_2_CURATED_PROFILE_RECORD_SCHEMA,
    V4_2_CURATED_REPORT_SCHEMA,
    V4_2_CURATED_RETAINED_DECISIONS,
    V42CuratedProfile,
    build_curated_manifest,
    build_curated_record,
    curated_implementation_hashes,
    load_and_validate_curation_policy,
)
from memgen.experience.v4_2_local_direct import (
    V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
    V4_2_LOCAL_DIRECT_REPORT_SCHEMA,
    V42LocalDirectProfile,
    local_direct_implementation_hashes,
)


V4_2_LOCAL_DIRECT_PROFILE_RECORD_SCHEMA = (
    "memgen-v4.2-local-direct-profile-record-v1"
)


def _validate_source_manifest(value: Mapping[str, Any]) -> V42LocalDirectProfile:
    if value.get("schema_version") != V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA:
        raise ValueError("Unexpected V4.2 local-direct source manifest schema")
    if value.get("manifest_sha256") != _logical_hash(value, "manifest_sha256"):
        raise ValueError("V4.2 local-direct source manifest hash mismatch")
    if (
        value.get("status") != "constructed_not_tensor_compiled"
        or value.get("qualified_for_online_use") is not False
        or value.get("benchmark") != "openai/gsm8k"
        or value.get("auxiliary_banks_materialized") is not False
        or value.get("api_key_read") is not False
        or value.get("external_api_calls_made") != 0
    ):
        raise ValueError("V4.2 local-direct source manifest contract drifted")
    profile = V42LocalDirectProfile(**value.get("profile", {}))
    if value.get("profile_sha256") != profile.profile_sha256:
        raise ValueError("V4.2 local-direct source profile hash mismatch")
    bank_ids = value.get("bank_ids")
    if (
        not isinstance(bank_ids, list)
        or len(bank_ids) != value.get("record_count")
        or len(set(bank_ids)) != len(bank_ids)
        or value.get("record_order_sha256") != canonical_json_sha256(bank_ids)
        or set(value.get("record_sha256", {})) != set(bank_ids)
    ):
        raise ValueError("V4.2 local-direct source namespace drifted")
    if value.get("inputs", {}).get("repository", {}).get(
        "implementation_sha256"
    ) != local_direct_implementation_hashes(PROJECT_ROOT):
        raise ValueError("V4.2 local-direct implementation identity drifted")
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def authenticate_source_bank(
    source_dir: Path,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    """Authenticate all standard local-direct source artifacts."""

    paths = {
        "records": source_dir / "bank_records.jsonl",
        "manifest": source_dir / "bank_manifest.json",
        "profile": source_dir / "construction_profile.json",
        "report": source_dir / "local_direct_report.json",
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing V4.2 local-direct source {name}: {path}")

    manifest = _load_json(paths["manifest"])
    _validate_source_manifest(manifest)
    records = tuple(dict(item) for item in iter_jsonl(paths["records"]))
    if [item.get("bank_id") for item in records] != manifest.get("bank_ids"):
        raise ValueError("V4.2 local-direct record order differs from its manifest")
    evidence_count = 0
    for record in records:
        bank_id = str(record.get("bank_id", ""))
        if (
            record.get("record_sha256") != _logical_hash(record, "record_sha256")
            or manifest.get("record_sha256", {}).get(bank_id)
            != record.get("record_sha256")
        ):
            raise ValueError("V4.2 local-direct source record binding drifted")
        sample_ids = record.get("construction", {}).get("sample_ids")
        if (
            not isinstance(sample_ids, list)
            or len(sample_ids) < 5
            or len(sample_ids) != len(set(sample_ids))
            or record.get("construction", {}).get("distinct_sample_count")
            != len(sample_ids)
        ):
            raise ValueError("V4.2 local-direct source support drifted")
        evidence_count += len(sample_ids)
    if evidence_count != V4_2_CURATED_EXPECTED_SOURCE_EVIDENCE:
        raise ValueError("V4.2 local-direct source evidence count drifted")

    profile_record = _load_json(paths["profile"])
    if (
        profile_record.get("schema_version")
        != V4_2_LOCAL_DIRECT_PROFILE_RECORD_SCHEMA
        or profile_record.get("record_sha256")
        != _logical_hash(profile_record, "record_sha256")
    ):
        raise ValueError("V4.2 local-direct source profile record is invalid")
    source_profile = V42LocalDirectProfile(**profile_record.get("profile", {}))
    if (
        profile_record.get("profile_sha256") != source_profile.profile_sha256
        or manifest.get("profile_sha256") != source_profile.profile_sha256
        or profile_record.get("inputs") != manifest.get("inputs")
    ):
        raise ValueError("V4.2 local-direct source profile binding drifted")

    report = _load_json(paths["report"])
    if (
        report.get("schema_version") != V4_2_LOCAL_DIRECT_REPORT_SCHEMA
        or report.get("report_sha256") != _logical_hash(report, "report_sha256")
        or report.get("bank_manifest_sha256") != manifest.get("manifest_sha256")
        or report.get("record_order_sha256") != manifest.get("record_order_sha256")
        or report.get("bank_record_count") != len(records)
        or report.get("evidence_count") != evidence_count
        or report.get("external_api_calls_made") != 0
        or report.get("api_key_read") is not False
    ):
        raise ValueError("V4.2 local-direct source report binding drifted")
    return records, manifest, profile_record, report, paths


def build_outputs(
    *,
    source_records: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_profile_record: Mapping[str, Any],
    source_report: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    policy_path: Path,
    policy: Mapping[str, Any],
    decisions: Sequence[Mapping[str, str]],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    profile = V42CuratedProfile()
    policy_sha256 = canonical_json_sha256(policy)
    implementation = curated_implementation_hashes(PROJECT_ROOT)
    decision_by_id = {str(item["bank_id"]): item for item in decisions}
    curated_records: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    for source_rank, source_record in enumerate(source_records, start=1):
        bank_id = str(source_record["bank_id"])
        decision = decision_by_id[bank_id]
        retained = decision["decision"] in V4_2_CURATED_RETAINED_DECISIONS
        if retained:
            curated_records.append(
                build_curated_record(
                    source_record=source_record,
                    decision=decision,
                    policy_sha256=policy_sha256,
                    profile=profile,
                )
            )
        audit = {
            "schema_version": V4_2_CURATED_DECISION_RECORD_SCHEMA,
            "source_rank": source_rank,
            "bank_id": bank_id,
            "decision": decision["decision"],
            "retained": retained,
            "reason": decision["reason"],
            "semantic_category": decision["semantic_category"],
            "distinct_sample_count": source_record["construction"][
                "distinct_sample_count"
            ],
            "source_record_sha256": source_record["record_sha256"],
            "policy_sha256": policy_sha256,
        }
        audit["record_sha256"] = canonical_json_sha256(audit)
        decision_records.append(audit)

    manifest = build_curated_manifest(
        records=curated_records,
        source_manifest=source_manifest,
        source_manifest_path=source_paths["manifest"],
        source_records_path=source_paths["records"],
        source_profile_path=source_paths["profile"],
        source_report_path=source_paths["report"],
        policy_path=policy_path,
        policy_sha256=policy_sha256,
        decisions=decisions,
        profile=profile,
        implementation_sha256=implementation,
    )
    input_binding = {
        "source_directory": str(source_paths["manifest"].parent.resolve()),
        "source_manifest_logical_sha256": source_manifest["manifest_sha256"],
        "source_manifest_file_sha256": file_sha256(source_paths["manifest"]),
        "source_records_file_sha256": file_sha256(source_paths["records"]),
        "source_profile_record_sha256": source_profile_record["record_sha256"],
        "source_report_sha256": source_report["report_sha256"],
        "policy_path": str(policy_path.resolve()),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": policy_sha256,
        "repository": {"implementation_sha256": implementation},
    }
    profile_record = {
        "schema_version": V4_2_CURATED_PROFILE_RECORD_SCHEMA,
        "construction_version": V4_2_CURATED_CONSTRUCTION_VERSION,
        "stage": "authenticated_static_bank_curation",
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "inputs": input_binding,
        "api_key_read": False,
        "external_api_calls_made": 0,
    }
    profile_record["record_sha256"] = canonical_json_sha256(profile_record)
    counts = Counter(item["decision"] for item in decisions)
    report = {
        "schema_version": V4_2_CURATED_REPORT_SCHEMA,
        "construction_version": V4_2_CURATED_CONSTRUCTION_VERSION,
        "status": "curated_bank_constructed_not_tensor_compiled",
        "quality_tier": "provisional_local_curated",
        "qualified_for_online_use": False,
        "source_record_count": len(source_records),
        "source_evidence_count": sum(
            int(item["construction"]["distinct_sample_count"])
            for item in source_records
        ),
        "retained_record_count": len(curated_records),
        "retained_evidence_count": sum(
            int(item["construction"]["distinct_sample_count"])
            for item in curated_records
        ),
        "excluded_record_count": len(source_records) - len(curated_records),
        "decision_counts": dict(sorted(counts.items())),
        "retained_bank_ids": list(manifest["bank_ids"]),
        "excluded_bank_ids": list(manifest["curation"]["excluded_bank_ids"]),
        "profile_sha256": profile.profile_sha256,
        "policy_sha256": policy_sha256,
        "bank_manifest_sha256": manifest["manifest_sha256"],
        "record_order_sha256": manifest["record_order_sha256"],
        "static_process_card_review_performed": True,
        "full_construction_evidence_review_performed": False,
        "independent_review_performed": False,
        "semantic_api_audit_performed": False,
        "api_key_read": False,
        "external_api_calls_made": 0,
        "next_stage": (
            "compile layer twenty four target and reference side KV, then "
            "construct and calibrate selector anchors"
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return (
        tuple(curated_records),
        tuple(decision_records),
        manifest,
        profile_record,
        report,
    )


def _write_or_validate_outputs(
    output_dir: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    profile_record: Mapping[str, Any],
    report: Mapping[str, Any],
    resume: bool,
) -> None:
    values: dict[str, Any] = {
        "construction_profile.json": dict(profile_record),
        "curation_decisions.jsonl": tuple(decisions),
        "bank_records.jsonl": tuple(records),
        "bank_manifest.json": dict(manifest),
        "curation_report.json": dict(report),
    }
    existing = [name for name in values if (output_dir / name).exists()]
    if existing:
        if not resume:
            raise ValueError("V4.2 curated output exists; pass --resume")
        if len(existing) != len(values):
            raise ValueError("V4.2 curated output is incomplete")
        for name, expected in values.items():
            path = output_dir / name
            actual: Any = (
                tuple(dict(item) for item in iter_jsonl(path))
                if name.endswith(".jsonl")
                else _load_json(path)
            )
            if actual != expected:
                raise ValueError(f"V4.2 curated output drifted: {name}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "construction_profile.json", profile_record)
    _write_jsonl(output_dir / "curation_decisions.jsonl", decisions)
    _write_jsonl(output_dir / "bank_records.jsonl", records)
    _write_json(output_dir / "bank_manifest.json", manifest)
    _write_json(output_dir / "curation_report.json", report)


def main() -> None:
    args = parse_args()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.policy = args.policy.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.source_dir.is_dir():
        raise ValueError(f"Missing V4.2 local-direct source directory: {args.source_dir}")
    if not args.policy.is_file():
        raise ValueError(f"Missing V4.2 curation policy: {args.policy}")
    if args.output_dir in {PROJECT_ROOT.resolve(), args.source_dir}:
        raise ValueError("V4.2 curated output must differ from repository and source")

    records, manifest, source_profile, source_report, source_paths = (
        authenticate_source_bank(args.source_dir)
    )
    policy, decisions = load_and_validate_curation_policy(
        args.policy, source_manifest=manifest
    )
    outputs = build_outputs(
        source_records=records,
        source_manifest=manifest,
        source_profile_record=source_profile,
        source_report=source_report,
        source_paths=source_paths,
        policy_path=args.policy,
        policy=policy,
        decisions=decisions,
    )
    curated_records, decision_records, curated_manifest, profile_record, report = outputs
    _write_or_validate_outputs(
        args.output_dir,
        records=curated_records,
        decisions=decision_records,
        manifest=curated_manifest,
        profile_record=profile_record,
        report=report,
        resume=args.resume,
    )
    print(
        "[v4.2-curated] PASS "
        f"source={report['source_record_count']} "
        f"retained={report['retained_record_count']} "
        f"excluded={report['excluded_record_count']} "
        f"evidence={report['retained_evidence_count']} "
        "api_key_read=false api_calls=0",
        flush=True,
    )
    print(
        f"[v4.2-curated] report={args.output_dir / 'curation_report.json'}",
        flush=True,
    )
    print(
        f"[v4.2-curated] manifest={args.output_dir / 'bank_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4.2-curated] error: {exc}", file=sys.stderr, flush=True)
        raise
