"""Authenticated lineage contract for one GSM8K Phase-1 + V3.4 risk pair.

The research runners historically accepted two unrelated paths.  This module
turns them into one immutable, content-addressed input set and makes any
downstream V4 compatibility (or incompatibility) explicit.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import (
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
)
from memgen.experience.risk import (
    TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA,
    approved_experiences,
)


PHASE1_RISK_LINEAGE_SCHEMA = "memgen-gsm8k-phase1-risk-lineage-v1"
_LINEAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")

PHASE1_ARTIFACTS: tuple[tuple[str, str, bool], ...] = (
    ("split_manifest", "split_manifest.json", False),
    ("student_rollouts", "student_rollouts.jsonl", False),
    ("rollout_summary", "rollout_summary.json", False),
    ("verified_experiences", "verified_experiences.jsonl", False),
    ("experience_build_report", "experience_build_report.json", False),
    ("teacher_reflections", "teacher_reflections.jsonl", False),
    ("ai_review_records", "ai_review_records.jsonl", False),
    ("ai_approved_bank_records", "ai_approved_bank_records.jsonl", False),
    ("ai_rejected_bank_records", "ai_rejected_bank_records.jsonl", True),
    ("deferred_bank_records", "deferred_bank_records.jsonl", True),
    ("quarantined_bank_records", "quarantined_bank_records.jsonl", True),
    ("ai_review_report", "ai_review_report.json", False),
)

RISK_ARTIFACTS: tuple[tuple[str, str, bool], ...] = (
    ("artifact", "token-entropy-risk-gate-v3.4.pt", False),
    ("report", "token_entropy_risk_report.json", False),
    ("evidence", "token_entropy_risk_evidence.jsonl", False),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_artifacts(
    directory: Path,
    definitions: Sequence[tuple[str, str, bool]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, filename, allow_empty in definitions:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if not allow_empty and path.stat().st_size == 0:
            raise ValueError(f"Required lineage artifact is empty: {path}")
        paths[name] = path
    return paths


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _artifact_entry(path: Path, *, lineage_root: Path, jsonl: bool) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(lineage_root.resolve()))
    except ValueError:
        relative = None
    result: dict[str, Any] = {
        "path": str(resolved),
        "relative_path": relative,
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if jsonl:
        result["row_count"] = _line_count(resolved)
    return result


def _validate_split_manifest(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError("Unexpected Phase-1 split manifest schema")
    logical = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    }
    if value.get("manifest_sha256") != canonical_json_sha256(logical):
        raise ValueError("Phase-1 split manifest logical hash mismatch")
    if value.get("overlap_check", {}).get("passed") is not True:
        raise ValueError("Phase-1 split manifest did not pass overlap checking")


def _validate_manifest_logical_hash(value: Mapping[str, Any], *, owner: str) -> None:
    stored = value.get("manifest_sha256")
    if not isinstance(stored, str) or not stored:
        raise ValueError(f"{owner} has no logical manifest hash")
    logical = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    }
    if stored != canonical_json_sha256(logical):
        raise ValueError(f"{owner} logical manifest hash mismatch")


def _review_hash_bindings(
    report: Mapping[str, Any], phase1_paths: Mapping[str, Path]
) -> None:
    expected = {
        "experiences_sha256": "verified_experiences",
        "teacher_records_sha256": "teacher_reflections",
        "review_records_sha256": "ai_review_records",
        "approved_sha256": "ai_approved_bank_records",
        "rejected_sha256": "ai_rejected_bank_records",
        "deferred_sha256": "deferred_bank_records",
        "quarantined_sha256": "quarantined_bank_records",
    }
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Phase-1 AI review report has no artifact bindings")
    for field, name in expected.items():
        if artifacts.get(field) != file_sha256(phase1_paths[name]):
            raise ValueError(f"Phase-1 AI review binding drifted: {field}")


def _downstream_compatibility(
    *,
    phase1_paths: Mapping[str, Path],
    bank_manifest_path: Path | None,
    side_kv_manifest_path: Path | None,
) -> dict[str, Any]:
    if bank_manifest_path is None and side_kv_manifest_path is None:
        return {
            "checked": False,
            "compatible": None,
            "status": "not_checked",
            "reason": "No downstream V4 manifests were supplied while sealing the lineage.",
        }
    if bank_manifest_path is None or side_kv_manifest_path is None:
        raise ValueError(
            "Downstream compatibility requires both bank and side-KV manifests"
        )
    bank_manifest_path = bank_manifest_path.resolve()
    side_kv_manifest_path = side_kv_manifest_path.resolve()
    if not bank_manifest_path.is_file() or not side_kv_manifest_path.is_file():
        raise FileNotFoundError("Downstream V4 bank or side-KV manifest is missing")
    bank = _load_json(bank_manifest_path)
    side = _load_json(side_kv_manifest_path)
    _validate_manifest_logical_hash(bank, owner="V4 bank manifest")
    _validate_manifest_logical_hash(side, owner="V4 side-KV manifest")
    inputs = bank.get("inputs", {})
    source = side.get("source", {})
    checks = {
        "bank_experiences_sha256": inputs.get("experiences_sha256")
        == file_sha256(phase1_paths["verified_experiences"]),
        "bank_split_manifest_sha256": inputs.get("split_manifest_sha256")
        == file_sha256(phase1_paths["split_manifest"]),
        "side_kv_bank_manifest_file_sha256": source.get(
            "bank_manifest_file_sha256"
        )
        == file_sha256(bank_manifest_path),
        "side_kv_bank_manifest_logical_sha256": source.get(
            "bank_manifest_logical_sha256"
        )
        == bank.get("manifest_sha256"),
    }
    compatible = all(checks.values())
    return {
        "checked": True,
        "compatible": compatible,
        "status": (
            "compatible_with_supplied_v4_archive"
            if compatible
            else "incompatible_rebuild_or_original_data_recovery_required"
        ),
        "checks": checks,
        "bank_manifest": {
            "path": str(bank_manifest_path),
            "file_sha256": file_sha256(bank_manifest_path),
            "logical_sha256": bank["manifest_sha256"],
        },
        "side_kv_manifest": {
            "path": str(side_kv_manifest_path),
            "file_sha256": file_sha256(side_kv_manifest_path),
            "logical_sha256": side["manifest_sha256"],
        },
    }


def build_phase1_risk_lineage(
    *,
    lineage_id: str,
    lineage_root: Path,
    phase1_dir: Path,
    risk_dir: Path,
    risk_artifact: Mapping[str, Any],
    repository_revision: str | None,
    bank_manifest_path: Path | None = None,
    side_kv_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and seal one complete Phase-1/risk input lineage."""

    if not _LINEAGE_ID.fullmatch(lineage_id):
        raise ValueError(
            "lineage-id must be 3-128 lowercase letters, digits, '.', '_' or '-'"
        )
    lineage_root = lineage_root.resolve()
    phase1_dir = phase1_dir.resolve()
    risk_dir = risk_dir.resolve()
    for owner, directory in (("Phase-1", phase1_dir), ("risk", risk_dir)):
        try:
            directory.relative_to(lineage_root)
        except ValueError as exc:
            raise ValueError(f"{owner} directory must be inside the lineage root") from exc

    phase1_paths = _require_artifacts(phase1_dir, PHASE1_ARTIFACTS)
    risk_paths = _require_artifacts(risk_dir, RISK_ARTIFACTS)
    split = _load_json(phase1_paths["split_manifest"])
    _validate_split_manifest(split)

    rollout_summary = _load_json(phase1_paths["rollout_summary"])
    if rollout_summary.get("output_sha256") != file_sha256(
        phase1_paths["student_rollouts"]
    ):
        raise ValueError("Phase-1 rollout summary is not bound to student rollouts")
    if rollout_summary.get("split_manifest_sha256") != split.get("manifest_sha256"):
        raise ValueError("Phase-1 rollout summary uses a different logical split")

    experience_report = _load_json(phase1_paths["experience_build_report"])
    if experience_report.get("rollouts_sha256") != file_sha256(
        phase1_paths["student_rollouts"]
    ):
        raise ValueError("Phase-1 experience report is not bound to student rollouts")
    if experience_report.get("experiences_sha256") != file_sha256(
        phase1_paths["verified_experiences"]
    ):
        raise ValueError("Phase-1 experience report is not bound to experiences")

    review_report = _load_json(phase1_paths["ai_review_report"])
    _review_hash_bindings(review_report, phase1_paths)
    experiences = list(iter_jsonl(phase1_paths["verified_experiences"]))
    approved = list(iter_jsonl(phase1_paths["ai_approved_bank_records"]))
    selected, selection = approved_experiences(
        approved,
        experiences,
        allowed_experience_types=("answer_correctness",),
    )

    risk_report = _load_json(risk_paths["report"])
    risk_sha256 = file_sha256(risk_paths["artifact"])
    approved_sha256 = file_sha256(phase1_paths["ai_approved_bank_records"])
    experiences_sha256 = file_sha256(phase1_paths["verified_experiences"])
    evidence_sha256 = file_sha256(risk_paths["evidence"])
    if (
        risk_report.get("status") != "passed"
        or risk_report.get("qualification", {}).get("passed") is not True
        or risk_report.get("artifact", {}).get("sha256") != risk_sha256
        or risk_report.get("inputs", {}).get("approved_bank_sha256")
        != approved_sha256
        or risk_report.get("inputs", {}).get("verified_experiences_sha256")
        != experiences_sha256
        or risk_report.get("evidence_trace", {}).get("sha256") != evidence_sha256
    ):
        raise ValueError("V3.4 risk report is incomplete or bound to other inputs")
    if (
        risk_artifact.get("schema_version") != TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA
        or risk_artifact.get("status") != "passed"
        or risk_artifact.get("qualification", {}).get("passed") is not True
        or risk_artifact.get("inputs", {}).get("approved_bank_sha256")
        != approved_sha256
        or risk_artifact.get("inputs", {}).get("verified_experiences_sha256")
        != experiences_sha256
    ):
        raise ValueError("V3.4 risk artifact is unqualified or bound to other inputs")

    student = selected[0].get("student", {})
    reasoner = risk_artifact.get("reasoner", {})
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if student.get(field) != reasoner.get(field):
            raise ValueError(f"Risk reasoner differs from Phase-1 student: {field}")

    downstream = _downstream_compatibility(
        phase1_paths=phase1_paths,
        bank_manifest_path=bank_manifest_path,
        side_kv_manifest_path=side_kv_manifest_path,
    )
    phase1_entries = {
        name: _artifact_entry(
            phase1_paths[name],
            lineage_root=lineage_root,
            jsonl=filename.endswith(".jsonl"),
        )
        for name, filename, _allow_empty in PHASE1_ARTIFACTS
    }
    risk_entries = {
        name: _artifact_entry(
            risk_paths[name],
            lineage_root=lineage_root,
            jsonl=filename.endswith(".jsonl"),
        )
        for name, filename, _allow_empty in RISK_ARTIFACTS
    }
    manifest: dict[str, Any] = {
        "schema_version": PHASE1_RISK_LINEAGE_SCHEMA,
        "lineage_id": lineage_id,
        "created_at": _utc_now(),
        "status": "sealed_phase1_risk_lineage",
        "immutable_after_seal": True,
        "repository_revision": repository_revision,
        "benchmark": "openai/gsm8k",
        "phase1": {
            "directory": str(phase1_dir),
            "artifacts": phase1_entries,
            "split_manifest_logical_sha256": split["manifest_sha256"],
            "dataset": split.get("dataset"),
            "split_policy": split.get("policy"),
            "split_counts": split.get("counts"),
            "rollout_configuration": rollout_summary.get("rollout_configuration"),
            "student": rollout_summary.get("student"),
            "verified_experience_count": len(experiences),
            "ai_approved_record_count": len(approved),
            "risk_eligible_answer_correctness_count": len(selected),
            "risk_selection": selection,
        },
        "risk": {
            "directory": str(risk_dir),
            "artifacts": risk_entries,
            "schema_version": risk_artifact.get("schema_version"),
            "artifact_id": risk_artifact.get("artifact_id"),
            "reasoner": reasoner,
            "construction": risk_artifact.get("construction"),
            "qualification": risk_artifact.get("qualification"),
            "event_counts": risk_artifact.get("event_counts"),
            "inputs": risk_artifact.get("inputs"),
        },
        "authentication": {
            "phase1_internal_bindings_passed": True,
            "risk_bound_to_exact_phase1_files": True,
            "risk_reasoner_matches_phase1_student": True,
        },
        "downstream_v4": downstream,
        "canonical_usage": {
            "phase1_dir": str(phase1_dir),
            "token_risk_artifact": str(risk_paths["artifact"].resolve()),
            "environment_file": str((lineage_root / "USE_THIS_LINEAGE.env").resolve()),
        },
        "warning": (
            "A regenerated Phase-1 is not interchangeable with an older V4 bank. "
            "Use downstream_v4.compatible before any V4 cache/oracle run."
        ),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def validate_sealed_lineage(manifest: Mapping[str, Any], *, path: Path) -> None:
    """Fail closed if a sealed manifest or any referenced artifact drifted."""

    if (
        manifest.get("schema_version") != PHASE1_RISK_LINEAGE_SCHEMA
        or manifest.get("status") != "sealed_phase1_risk_lineage"
        or manifest.get("immutable_after_seal") is not True
    ):
        raise ValueError("Unexpected or unsealed Phase-1/risk lineage manifest")
    logical = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_json_sha256(logical):
        raise ValueError(f"Lineage manifest hash mismatch: {path}")
    for section_name in ("phase1", "risk"):
        artifacts = manifest.get(section_name, {}).get("artifacts", {})
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ValueError(f"Lineage {section_name} artifact index is missing")
        for name, entry in artifacts.items():
            artifact_path = Path(str(entry.get("path", "")))
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if file_sha256(artifact_path) != entry.get("sha256"):
                raise ValueError(
                    f"Sealed lineage artifact drifted: {section_name}.{name}"
                )


__all__ = [
    "PHASE1_ARTIFACTS",
    "PHASE1_RISK_LINEAGE_SCHEMA",
    "RISK_ARTIFACTS",
    "build_phase1_risk_lineage",
    "validate_sealed_lineage",
]
