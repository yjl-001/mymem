"""Authenticated recovery of V4 source evidence from semantic packets.

The V4.2 semantic preflight packets preserve the raw question, official GSM8K
solution, verified success trajectory, verified failure trajectory, and both
verifier records for every evidence item.  This module defines a deliberately
separate lineage for replaying those source trajectories after the original
Phase-1 directory has been lost.

Recovery never claims to recreate the byte-identical Phase-1 files or the old
token-risk artifact.  It authenticates the surviving packet file through the
curated bank, rejoins the public GSM8K rows, and emits records that can only be
consumed when the recovery manifest is supplied explicitly.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.v4_2_semantic_bank import V4_2_EVIDENCE_PACKET_SCHEMA


V4_QUESTION_RECOVERY_SCHEMA = "memgen-v4-question-recovery-lineage-v1"
V4_RECOVERED_EXPERIENCE_SCHEMA = "memgen-v4-recovered-source-experience-v1"
V4_QUESTION_RECOVERY_STATUS = "sealed_semantic_packet_source_replay"
V4_QUESTION_RECOVERY_EXPECTED_BANKS = 17
V4_QUESTION_RECOVERY_EXPECTED_EVIDENCE = 116

_SAMPLE_ID_RE = re.compile(
    r"^gsm8k-(?P<dataset_split>train|test)-(?P<source_index>[0-9]+)-"
    r"(?P<question_prefix>[0-9a-f]{12})$"
)


def _logical_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return canonical_json_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"created_at", hash_field}
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve_artifact_path(manifest_path: Path, value: Mapping[str, Any]) -> Path:
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("V4 question-recovery artifact has no path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_hashed_json(
    value: Mapping[str, Any],
    *,
    field: str,
    owner: str,
    exclude_created_at: bool = False,
) -> None:
    logical = {
        key: item
        for key, item in value.items()
        if key != field and (key != "created_at" or not exclude_created_at)
    }
    if value.get(field) != canonical_json_sha256(logical):
        raise ValueError(f"{owner} logical hash mismatch")


def _validate_file_entry(
    manifest_path: Path, entry: Mapping[str, Any], *, owner: str
) -> Path:
    path = _resolve_artifact_path(manifest_path, entry)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing V4 question-recovery {owner}: {path}")
    if entry.get("sha256") != file_sha256(path):
        raise ValueError(f"V4 question-recovery {owner} file hash mismatch")
    return path


def _parse_sample_identity(sample_id: str, question: str) -> dict[str, Any]:
    match = _SAMPLE_ID_RE.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"Unexpected GSM8K sample ID: {sample_id}")
    question_sha256 = text_sha256(question.strip())
    if not question_sha256.startswith(match.group("question_prefix")):
        raise ValueError(f"GSM8K sample/question hash mismatch: {sample_id}")
    return {
        "dataset_split": match.group("dataset_split"),
        "source_index": int(match.group("source_index")),
        "question_sha256": question_sha256,
    }


def _processed_gsm8k_solution(answer: str) -> str:
    parts = str(answer).strip().split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def _packet_evidence(path: Path) -> dict[str, dict[str, Any]]:
    evidence_by_id: dict[str, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    for packet in iter_jsonl(path):
        if packet.get("schema_version") != V4_2_EVIDENCE_PACKET_SCHEMA:
            raise ValueError("Unexpected V4.2 semantic evidence-packet schema")
        _validate_hashed_json(
            packet, field="packet_sha256", owner="V4.2 semantic evidence packet"
        )
        evidence = packet.get("evidence")
        if not isinstance(evidence, list) or packet.get("evidence_count") != len(evidence):
            raise ValueError("V4.2 semantic evidence-packet count mismatch")
        for raw in evidence:
            if not isinstance(raw, Mapping):
                raise ValueError("V4.2 semantic evidence item is not an object")
            item = dict(raw)
            experience_id = str(item.get("evidence_id", ""))
            sample_id = str(item.get("sample_id", ""))
            if not experience_id or experience_id in evidence_by_id:
                raise ValueError("V4.2 semantic evidence IDs are missing or duplicated")
            if not sample_id or sample_id in sample_ids:
                raise ValueError("V4.2 semantic evidence samples are missing or duplicated")
            required_strings = (
                "source_experience_type",
                "question",
                "official_solution",
                "verified_success_trajectory",
                "verified_failure_trajectory",
                "source_provenance_sha256",
                "source_signature_sha256",
                "construction_input_sha256",
            )
            if any(not isinstance(item.get(field), str) or not item[field].strip() for field in required_strings):
                raise ValueError("V4.2 semantic evidence item is incomplete")
            _parse_sample_identity(sample_id, str(item["question"]))
            evidence_by_id[experience_id] = item
            sample_ids.add(sample_id)
    if not evidence_by_id:
        raise ValueError("V4.2 semantic packet file has no evidence")
    return evidence_by_id


def _bank_membership(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    bank_by_experience: dict[str, str] = {}
    sample_by_experience: dict[str, str] = {}
    for record in records:
        bank_id = str(record.get("bank_id", ""))
        construction = record.get("construction", {})
        experience_ids = construction.get("experience_ids")
        sample_ids = construction.get("sample_ids")
        if (
            not bank_id
            or not isinstance(experience_ids, list)
            or not isinstance(sample_ids, list)
            or len(experience_ids) != len(sample_ids)
            or len(experience_ids) < 5
            or len(set(str(value) for value in experience_ids)) != len(experience_ids)
            or len(set(str(value) for value in sample_ids)) != len(sample_ids)
        ):
            raise ValueError("V4 curated bank construction membership is invalid")
        for experience_id, sample_id in zip(experience_ids, sample_ids):
            normalized_experience_id = str(experience_id)
            if normalized_experience_id in bank_by_experience:
                raise ValueError("V4 curated bank repeats a construction experience")
            bank_by_experience[normalized_experience_id] = bank_id
            sample_by_experience[normalized_experience_id] = str(sample_id)
    return bank_by_experience, sample_by_experience


def _validate_bank_inputs(
    *,
    bank_records_path: Path,
    bank_manifest_path: Path,
    semantic_packets_path: Path,
    side_kv_manifest_path: Path,
    expected_bank_count: int,
    expected_evidence_count: int,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    records = tuple(dict(item) for item in iter_jsonl(bank_records_path))
    bank_manifest = _load_json(bank_manifest_path)
    _validate_hashed_json(
        bank_manifest,
        field="manifest_sha256",
        owner="V4 curated bank manifest",
        exclude_created_at=True,
    )
    bank_ids = [str(record.get("bank_id", "")) for record in records]
    if (
        len(records) != expected_bank_count
        or bank_ids != bank_manifest.get("bank_ids")
        or bank_manifest.get("record_count") != len(records)
        or bank_manifest.get("evidence_count") != expected_evidence_count
        or len(set(bank_ids)) != len(bank_ids)
    ):
        raise ValueError("V4 question recovery requires the expected curated bank")
    for record in records:
        bank_id = str(record["bank_id"])
        _validate_hashed_json(record, field="record_sha256", owner="V4 bank record")
        if bank_manifest.get("record_sha256", {}).get(bank_id) != record["record_sha256"]:
            raise ValueError("V4 bank record is not bound by its manifest")

    expected_packet_sha256 = (
        bank_manifest.get("inputs", {})
        .get("semantic_preflight", {})
        .get("evidence_packet_file_sha256")
    )
    if expected_packet_sha256 != file_sha256(semantic_packets_path):
        raise ValueError(
            "Semantic packet file is not the exact evidence source of the curated bank"
        )

    side_kv_manifest = _load_json(side_kv_manifest_path)
    _validate_hashed_json(
        side_kv_manifest, field="manifest_sha256", owner="V4 side-KV manifest"
    )
    side_records = side_kv_manifest.get("records")
    if not isinstance(side_records, list):
        raise ValueError("V4 side-KV manifest has no role records")
    target_bank_ids = [
        str(item.get("bank_id", ""))
        for item in side_records
        if isinstance(item, Mapping) and item.get("role") == "target"
    ]
    reference_bank_ids = [
        str(item.get("bank_id", ""))
        for item in side_records
        if isinstance(item, Mapping) and item.get("role") == "reference"
    ]
    if (
        side_kv_manifest.get("source", {}).get("bank_manifest_logical_sha256")
        != bank_manifest.get("manifest_sha256")
        or target_bank_ids != bank_ids
        or reference_bank_ids != bank_ids
        or side_kv_manifest.get("bank_count") != len(bank_ids)
    ):
        raise ValueError("V4 side-KV is not compiled from the supplied curated bank")
    reasoner = side_kv_manifest.get("reasoner")
    if not isinstance(reasoner, Mapping) or any(
        not isinstance(reasoner.get(field), str) or not reasoner[field]
        for field in ("model_name", "model_revision", "tokenizer_revision")
    ):
        raise ValueError("V4 side-KV reasoner provenance is incomplete")

    bank_by_experience, sample_by_experience = _bank_membership(records)
    if len(bank_by_experience) != expected_evidence_count:
        raise ValueError("V4 curated bank evidence membership count drifted")
    return (
        records,
        bank_manifest,
        side_kv_manifest,
        bank_by_experience,
        sample_by_experience,
    )


def build_recovered_source_records(
    *,
    semantic_packets_path: Path,
    bank_records_path: Path,
    bank_manifest_path: Path,
    side_kv_manifest_path: Path,
    split_manifest: Mapping[str, Any],
    train_records: Sequence[Mapping[str, Any]],
    expected_bank_count: int = V4_QUESTION_RECOVERY_EXPECTED_BANKS,
    expected_evidence_count: int = V4_QUESTION_RECOVERY_EXPECTED_EVIDENCE,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Authenticate and recover exactly the evidence retained by the bank."""

    (
        bank_records,
        bank_manifest,
        side_kv_manifest,
        bank_by_experience,
        sample_by_experience,
    ) = _validate_bank_inputs(
        bank_records_path=bank_records_path,
        bank_manifest_path=bank_manifest_path,
        semantic_packets_path=semantic_packets_path,
        side_kv_manifest_path=side_kv_manifest_path,
        expected_bank_count=expected_bank_count,
        expected_evidence_count=expected_evidence_count,
    )
    evidence_by_id = _packet_evidence(semantic_packets_path)
    missing = sorted(set(bank_by_experience) - set(evidence_by_id))
    if missing:
        raise ValueError(f"Semantic packets lost curated evidence: {missing[:3]}")

    split_entries = {
        str(item.get("sample_id", "")): item
        for item in split_manifest.get("samples", ())
        if item.get("logical_split") == "bank-source"
    }
    split_sha256 = str(split_manifest.get("manifest_sha256", ""))
    if not split_entries or not split_sha256:
        raise ValueError("Rebuilt GSM8K split manifest has no bank-source identity")
    reasoner = dict(side_kv_manifest["reasoner"])
    records: list[dict[str, Any]] = []
    verifier_versions: Counter[str] = Counter()
    for experience_id in sorted(bank_by_experience):
        evidence = evidence_by_id[experience_id]
        sample_id = sample_by_experience[experience_id]
        if evidence.get("sample_id") != sample_id:
            raise ValueError("Semantic packet and curated bank sample IDs differ")
        source = split_entries.get(sample_id)
        if source is None:
            raise ValueError(f"Recovered evidence is not bank-source: {sample_id}")
        identity = _parse_sample_identity(sample_id, str(evidence["question"]))
        if any(source.get(field) != identity[field] for field in identity):
            raise ValueError("Rebuilt GSM8K split identity differs from semantic evidence")
        source_index = int(source["source_index"])
        if source_index < 0 or source_index >= len(train_records):
            raise ValueError("Recovered GSM8K source index is out of range")
        dataset_item = train_records[source_index]
        question = str(dataset_item["question"]).strip()
        official_solution = str(dataset_item["answer"]).strip()
        if (
            question != str(evidence["question"]).strip()
            or official_solution != str(evidence["official_solution"]).strip()
            or text_sha256(question) != source.get("question_sha256")
            or text_sha256(official_solution) != source.get("answer_sha256")
        ):
            raise ValueError("Public GSM8K row differs from preserved semantic evidence")

        success = str(evidence["verified_success_trajectory"]).strip()
        failure = str(evidence["verified_failure_trajectory"]).strip()
        processed_solution = _processed_gsm8k_solution(official_solution)
        target_diagnostic = diagnose_gsm8k_completion(success, processed_solution)
        reference_diagnostic = diagnose_gsm8k_completion(failure, processed_solution)
        if (
            target_diagnostic.get("reward") != 1.0
            or reference_diagnostic.get("reward") != 0.0
            or evidence.get("target_verifier", {}).get("reward") != 1.0
            or evidence.get("reference_verifier", {}).get("reward") != 0.0
        ):
            raise ValueError("Recovered semantic evidence fails strict verifier replay")
        verifier_versions[str(target_diagnostic.get("version", "unknown"))] += 1

        source_record = {
            "logical_split": "bank-source",
            "dataset_split": str(source["dataset_split"]),
            "source_index": source_index,
            "question_sha256": str(source["question_sha256"]),
            "answer_sha256": str(source["answer_sha256"]),
            "split_manifest_sha256": split_sha256,
        }
        recovered = {
            "schema_version": V4_RECOVERED_EXPERIENCE_SCHEMA,
            "experience_id": experience_id,
            "sample_id": sample_id,
            "bank_id": bank_by_experience[experience_id],
            "source": source_record,
            "context": question,
            "trajectory": success,
            "reference_trajectory": failure,
            "outcome": "verified_success",
            "reward": 1.0,
            "reference_evidence": "verified_failure",
            "reference_failure_types": list(
                evidence.get("reference_verifier", {}).get("failure_types") or []
            ),
            "experience_type": str(evidence["source_experience_type"]),
            "target_episode_id": f"semantic-packet:{experience_id}:verified-success",
            "reference_episode_id": f"semantic-packet:{experience_id}:verified-failure",
            "target_verifier": dict(evidence["target_verifier"]),
            "reference_verifier": dict(evidence["reference_verifier"]),
            "replayed_target_verifier": target_diagnostic,
            "replayed_reference_verifier": reference_diagnostic,
            "student": {
                "model_name": reasoner["model_name"],
                "model_revision": reasoner["model_revision"],
                "tokenizer_revision": reasoner["tokenizer_revision"],
            },
            "recovery_provenance": {
                "source": "authenticated_v4_2_semantic_evidence_packet",
                "source_provenance_sha256": evidence["source_provenance_sha256"],
                "source_signature_sha256": evidence["source_signature_sha256"],
                "construction_input_sha256": evidence["construction_input_sha256"],
                "semantic_signature": evidence["semantic_signature"],
                "exact_source_question": True,
                "exact_source_success_trajectory": True,
                "exact_source_failure_trajectory": True,
                "original_phase1_file_recreated": False,
            },
        }
        recovered["record_sha256"] = canonical_json_sha256(recovered)
        records.append(recovered)

    summary = {
        "bank_count": len(bank_records),
        "evidence_count": len(records),
        "experience_type_counts": dict(
            sorted(Counter(row["experience_type"] for row in records).items())
        ),
        "strict_verifier_version_counts": dict(sorted(verifier_versions.items())),
        "bank_support": dict(sorted(Counter(bank_by_experience.values()).items())),
        "bank_manifest_logical_sha256": bank_manifest["manifest_sha256"],
        "side_kv_manifest_logical_sha256": side_kv_manifest["manifest_sha256"],
    }
    return tuple(records), summary


def build_question_recovery_manifest(
    *,
    recovery_id: str,
    semantic_packets_path: Path,
    bank_records_path: Path,
    bank_manifest_path: Path,
    side_kv_manifest_path: Path,
    split_manifest_path: Path,
    recovered_experiences_path: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the replay lineage without claiming byte-identical Phase-1 recovery."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", recovery_id):
        raise ValueError("Recovery ID must use 3-128 lowercase safe characters")
    side_kv_manifest = _load_json(side_kv_manifest_path)
    manifest = {
        "schema_version": V4_QUESTION_RECOVERY_SCHEMA,
        "recovery_id": recovery_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": V4_QUESTION_RECOVERY_STATUS,
        "benchmark": "openai/gsm8k",
        "offline_only": True,
        "qualified_for_online_use": False,
        "claims": {
            "original_phase1_file_recovery_claim": False,
            "original_risk_artifact_recovery_claim": False,
            "same_source_question": True,
            "same_source_success_trajectory": True,
            "same_source_failure_trajectory": True,
            "same_curated_bank_membership": True,
            "held_out_generalization_claim": False,
        },
        "reasoner": dict(side_kv_manifest["reasoner"]),
        "counts": dict(summary),
        "sources": {
            "semantic_evidence_packets": {
                "path": str(semantic_packets_path.resolve()),
                "sha256": file_sha256(semantic_packets_path),
            },
            "curated_bank_records": {
                "path": str(bank_records_path.resolve()),
                "sha256": file_sha256(bank_records_path),
            },
            "curated_bank_manifest": {
                "path": str(bank_manifest_path.resolve()),
                "sha256": file_sha256(bank_manifest_path),
                "logical_sha256": summary["bank_manifest_logical_sha256"],
            },
            "side_kv_manifest": {
                "path": str(side_kv_manifest_path.resolve()),
                "sha256": file_sha256(side_kv_manifest_path),
                "logical_sha256": summary["side_kv_manifest_logical_sha256"],
            },
        },
        "artifacts": {
            "rebuilt_split_manifest": {
                "path": split_manifest_path.name,
                "sha256": file_sha256(split_manifest_path),
                "logical_sha256": _load_json(split_manifest_path)["manifest_sha256"],
            },
            "recovered_experiences": {
                "path": recovered_experiences_path.name,
                "sha256": file_sha256(recovered_experiences_path),
                "row_count": int(summary["evidence_count"]),
            },
        },
        "risk_policy": {
            "old_artifact_is_not_recreated": True,
            "required_source_label": "semantic_packet_replay_strict_verifier_no_ai",
            "fit_holdout_unit": "experience_id",
            "external_teacher_required": False,
        },
        "external_api_calls_made": 0,
    }
    manifest["manifest_sha256"] = _logical_hash(manifest, "manifest_sha256")
    return manifest


def validate_question_recovery_manifest(
    manifest_path: Path, *, verify_source_files: bool = True
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != V4_QUESTION_RECOVERY_SCHEMA
        or manifest.get("status") != V4_QUESTION_RECOVERY_STATUS
        or manifest.get("benchmark") != "openai/gsm8k"
        or manifest.get("offline_only") is not True
        or manifest.get("qualified_for_online_use") is not False
        or manifest.get("external_api_calls_made") != 0
    ):
        raise ValueError("Unexpected or unsafe V4 question-recovery manifest")
    if manifest.get("manifest_sha256") != _logical_hash(manifest, "manifest_sha256"):
        raise ValueError("V4 question-recovery manifest hash mismatch")
    claims = manifest.get("claims", {})
    expected_claims = {
        "original_phase1_file_recovery_claim": False,
        "original_risk_artifact_recovery_claim": False,
        "same_source_question": True,
        "same_source_success_trajectory": True,
        "same_source_failure_trajectory": True,
        "same_curated_bank_membership": True,
        "held_out_generalization_claim": False,
    }
    if claims != expected_claims:
        raise ValueError("V4 question-recovery claims drifted")
    reasoner = manifest.get("reasoner", {})
    if any(not reasoner.get(field) for field in (
        "model_name", "model_revision", "tokenizer_revision"
    )):
        raise ValueError("V4 question-recovery reasoner provenance is incomplete")
    if verify_source_files:
        for owner, entry in manifest.get("sources", {}).items():
            _validate_file_entry(manifest_path, entry, owner=owner)
        for owner, entry in manifest.get("artifacts", {}).items():
            _validate_file_entry(manifest_path, entry, owner=owner)
    return manifest


def load_recovered_source_experiences(
    manifest_path: Path,
    *,
    experiences_path: Path | None = None,
    verify_source_files: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Load hash-bound replay records; ordinary Phase-1 loaders reject them."""

    manifest = validate_question_recovery_manifest(
        manifest_path, verify_source_files=verify_source_files
    )
    entry = manifest["artifacts"]["recovered_experiences"]
    expected_path = _resolve_artifact_path(manifest_path.resolve(), entry)
    path = experiences_path.expanduser().resolve() if experiences_path else expected_path
    if path != expected_path or entry.get("sha256") != file_sha256(path):
        raise ValueError("Recovered experiences differ from the recovery manifest")
    rows = tuple(dict(item) for item in iter_jsonl(path))
    if len(rows) != entry.get("row_count") or len(rows) != manifest.get("counts", {}).get(
        "evidence_count"
    ):
        raise ValueError("Recovered experience count differs from recovery manifest")
    source_paths = {
        name: _resolve_artifact_path(manifest_path.resolve(), entry)
        for name, entry in manifest["sources"].items()
    }
    expected_bank_count = int(manifest.get("counts", {}).get("bank_count", 0))
    expected_evidence_count = int(manifest.get("counts", {}).get("evidence_count", 0))
    (
        _bank_records,
        _bank_manifest,
        _side_kv_manifest,
        bank_by_experience,
        sample_by_experience,
    ) = _validate_bank_inputs(
        bank_records_path=source_paths["curated_bank_records"],
        bank_manifest_path=source_paths["curated_bank_manifest"],
        semantic_packets_path=source_paths["semantic_evidence_packets"],
        side_kv_manifest_path=source_paths["side_kv_manifest"],
        expected_bank_count=expected_bank_count,
        expected_evidence_count=expected_evidence_count,
    )
    evidence_by_id = _packet_evidence(source_paths["semantic_evidence_packets"])
    split_path = _resolve_artifact_path(
        manifest_path.resolve(), manifest["artifacts"]["rebuilt_split_manifest"]
    )
    split_manifest = _load_json(split_path)
    split_entries = {
        str(item.get("sample_id", "")): item
        for item in split_manifest.get("samples", ())
        if item.get("logical_split") == "bank-source"
    }
    expected_student = {
        field: manifest["reasoner"][field]
        for field in ("model_name", "model_revision", "tokenizer_revision")
    }
    seen_experience_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != V4_RECOVERED_EXPERIENCE_SCHEMA:
            raise ValueError("Unexpected recovered source-experience schema")
        _validate_hashed_json(
            row, field="record_sha256", owner="recovered source experience"
        )
        experience_id = str(row.get("experience_id", ""))
        sample_id = str(row.get("sample_id", ""))
        if (
            not experience_id
            or experience_id in seen_experience_ids
            or not sample_id
            or sample_id in seen_sample_ids
            or row.get("outcome") != "verified_success"
            or row.get("reward") != 1.0
            or row.get("reference_evidence") != "verified_failure"
            or row.get("target_verifier", {}).get("reward") != 1.0
            or row.get("reference_verifier", {}).get("reward") != 0.0
            or not str(row.get("trajectory", "")).strip()
            or not str(row.get("reference_trajectory", "")).strip()
        ):
            raise ValueError("Recovered source experience is incomplete or duplicated")
        provenance = row.get("recovery_provenance", {})
        if (
            provenance.get("source")
            != "authenticated_v4_2_semantic_evidence_packet"
            or provenance.get("exact_source_question") is not True
            or provenance.get("exact_source_success_trajectory") is not True
            or provenance.get("exact_source_failure_trajectory") is not True
            or provenance.get("original_phase1_file_recreated") is not False
        ):
            raise ValueError("Recovered source-experience claims drifted")
        evidence = evidence_by_id.get(experience_id)
        source = row.get("source", {})
        split_entry = split_entries.get(sample_id)
        expected_provenance = {
            "source": "authenticated_v4_2_semantic_evidence_packet",
            "source_provenance_sha256": evidence.get("source_provenance_sha256")
            if evidence
            else None,
            "source_signature_sha256": evidence.get("source_signature_sha256")
            if evidence
            else None,
            "construction_input_sha256": evidence.get("construction_input_sha256")
            if evidence
            else None,
            "semantic_signature": evidence.get("semantic_signature") if evidence else None,
            "exact_source_question": True,
            "exact_source_success_trajectory": True,
            "exact_source_failure_trajectory": True,
            "original_phase1_file_recreated": False,
        }
        if (
            evidence is None
            or bank_by_experience.get(experience_id) != row.get("bank_id")
            or sample_by_experience.get(experience_id) != sample_id
            or evidence.get("sample_id") != sample_id
            or str(evidence.get("question", "")).strip() != row.get("context")
            or str(evidence.get("verified_success_trajectory", "")).strip()
            != row.get("trajectory")
            or str(evidence.get("verified_failure_trajectory", "")).strip()
            != row.get("reference_trajectory")
            or evidence.get("target_verifier") != row.get("target_verifier")
            or evidence.get("reference_verifier") != row.get("reference_verifier")
            or provenance != expected_provenance
            or row.get("student") != expected_student
            or split_entry is None
            or source.get("logical_split") != "bank-source"
            or source.get("dataset_split") != split_entry.get("dataset_split")
            or source.get("source_index") != split_entry.get("source_index")
            or source.get("question_sha256") != split_entry.get("question_sha256")
            or source.get("answer_sha256") != split_entry.get("answer_sha256")
            or source.get("split_manifest_sha256")
            != split_manifest.get("manifest_sha256")
        ):
            raise ValueError("Recovered source experience lost its packet/bank binding")
        seen_experience_ids.add(experience_id)
        seen_sample_ids.add(sample_id)
    if set(seen_experience_ids) != set(bank_by_experience):
        raise ValueError("Recovered source experiences do not cover curated membership")
    return tuple(sorted(rows, key=lambda item: str(item["experience_id"])))


def recovered_experience_selection(
    rows: Iterable[Mapping[str, Any]], *, allowed_experience_types: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select risk evidence without manufacturing an AI-review approval route."""

    allowed = set(allowed_experience_types)
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        experience_type = str(row.get("experience_type", ""))
        if experience_type not in allowed:
            skipped[experience_type] += 1
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No recovered experiences remain after risk type selection")
    return selected, {
        "source": "semantic_packet_replay_strict_verifier_no_ai",
        "ai_review_approval_claim": False,
        "selected_count": len(selected),
        "selected_by_experience_type": dict(
            sorted(Counter(row["experience_type"] for row in selected).items())
        ),
        "skipped_by_experience_type": dict(sorted(skipped.items())),
        "selection_provenance_sha256": canonical_json_sha256(
            {
                "experience_ids": [row["experience_id"] for row in selected],
                "allowed_experience_types": sorted(allowed),
                "source": "semantic_packet_replay_strict_verifier_no_ai",
            }
        ),
    }


__all__ = [
    "V4_QUESTION_RECOVERY_EXPECTED_BANKS",
    "V4_QUESTION_RECOVERY_EXPECTED_EVIDENCE",
    "V4_QUESTION_RECOVERY_SCHEMA",
    "V4_RECOVERED_EXPERIENCE_SCHEMA",
    "build_question_recovery_manifest",
    "build_recovered_source_records",
    "load_recovered_source_experiences",
    "recovered_experience_selection",
    "validate_question_recovery_manifest",
]
