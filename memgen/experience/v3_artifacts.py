"""Fail-closed provenance checks shared by V3 offline and online scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3 import V3_OFFLINE_REPORT_SCHEMA


RETRIEVAL_KEY_BANK_SCHEMA = "experience-memory-retrieval-key-bank-v1"
SIDE_KV_BANK_SCHEMA = "canonical-side-kv-bank-v2"


def load_formal_e0_report(path: Path) -> dict[str, Any]:
    """Authenticate the existing layer-24 side-KV mechanism qualification."""

    value = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = value.get("final_report_sha256")
    actual_hash = canonical_json_sha256({
        key: item for key, item in value.items() if key != "final_report_sha256"
    })
    if expected_hash != actual_hash:
        raise ValueError("E0 final report hash mismatch")
    if (
        value.get("schema_version") != "experience-memory-e0-final-report-v2"
        or value.get("status") != "passed"
        or value.get("formal_e0_passed") is not True
        or value.get("task_accuracy_used") is not False
        or value.get("attention_implementation") != "sdpa"
        or not value.get("requirements")
        or not all(item is True for item in value["requirements"].values())
    ):
        raise ValueError("V3 requires a formally passed answer-blind E0 report")
    return value


def authenticate_e0_inputs(
    *,
    e0_report: Mapping[str, Any],
    memory_records_path: Path,
    side_kv_manifest_path: Path,
) -> None:
    """Bind V3 inputs to the exact records and KV bank qualified by E0."""

    verified = e0_report.get("compile_report", {}).get(
        "verified_artifact_sha256", {}
    )
    if (
        verified.get("memory_records") != file_sha256(memory_records_path)
        or verified.get("side_kv_manifest") != file_sha256(side_kv_manifest_path)
    ):
        raise ValueError("V3 records or side-KV manifest differ from qualified E0")


def validate_cross_bank_metadata(
    *,
    records: Sequence[MemoryRecord],
    side_manifest: Mapping[str, Any],
    key_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove record/key/value alignment without loading large tensor artifacts."""

    if not records:
        raise ValueError("V3 memory records are empty")
    if side_manifest.get("schema_version") != SIDE_KV_BANK_SCHEMA:
        raise ValueError("Unexpected V3 side-KV manifest schema")
    expected_side_hash = side_manifest.get("manifest_sha256")
    actual_side_hash = canonical_json_sha256({
        key: value
        for key, value in side_manifest.items()
        if key != "manifest_sha256"
    })
    if expected_side_hash != actual_side_hash:
        raise ValueError("V3 side-KV manifest hash mismatch")
    if (
        int(side_manifest.get("layer_number", -1)) != 24
        or side_manifest.get("canonical_pre_rope") is not True
        or side_manifest.get("compiler", {}).get("attention_backend") != "sdpa"
    ):
        raise ValueError("V3 side-KV value contract must be canonical layer-24 SDPA")

    record_ids = [record.memory_id for record in records]
    side_entries = list(side_manifest.get("records", []))
    side_ids = [str(entry.get("memory_id", "")) for entry in side_entries]
    if side_ids != record_ids:
        raise ValueError("V3 text and side-KV record order differs")
    for record, entry in zip(records, side_entries):
        if (
            record.kv_layer != 24
            or entry.get("payload_hash") != record.payload_hash
            or int(entry.get("kv_valid_slot_count", -1)) != record.token_count
        ):
            raise ValueError("V3 text and side-KV record metadata differs")

    if key_manifest is not None:
        if key_manifest.get("schema_version") != RETRIEVAL_KEY_BANK_SCHEMA:
            raise ValueError("Unexpected V3 retrieval-key manifest schema")
        expected_key_hash = key_manifest.get("manifest_sha256")
        actual_key_hash = canonical_json_sha256({
            key: value
            for key, value in key_manifest.items()
            if key != "manifest_sha256"
        })
        if expected_key_hash != actual_key_hash:
            raise ValueError("V3 retrieval-key manifest hash mismatch")
        key_entries = list(key_manifest.get("records", []))
        key_ids = [str(entry.get("memory_id", "")) for entry in key_entries]
        if key_ids != record_ids:
            raise ValueError("V3 text, key, and side-KV record order differs")
        key_reasoner = key_manifest.get("reasoner", {})
        side_reasoner = side_manifest.get("reasoner", {})
        for field_name in ("model_name", "model_revision", "tokenizer_revision"):
            if key_reasoner.get(field_name) != side_reasoner.get(field_name):
                raise ValueError(
                    "V3 retrieval-key and side-KV reasoner provenance differs"
                )
        if key_reasoner.get("attention_implementation") != "sdpa":
            raise ValueError("V3 retrieval keys were not encoded under SDPA")
        for record, entry in zip(records, key_entries):
            if entry.get("payload_hash") != record.payload_hash:
                raise ValueError("V3 retrieval key points to a different KV payload")

    return {
        "record_count": len(records),
        "record_order_sha256": canonical_json_sha256(record_ids),
        "payload_alignment_sha256": canonical_json_sha256([
            {"memory_id": record.memory_id, "payload_hash": record.payload_hash}
            for record in records
        ]),
        "layer_number": 24,
        "canonical_pre_rope": True,
        "attention_implementation": "sdpa",
    }


def load_v3_offline_report(
    path: Path,
    *,
    memory_records_path: Path,
    side_kv_manifest_path: Path,
    retrieval_key_manifest_path: Path,
    e0_final_report_path: Path,
) -> dict[str, Any]:
    """Authenticate the one-pass V3 offline qualification report."""

    value = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = value.get("report_sha256")
    actual_hash = canonical_json_sha256({
        key: item for key, item in value.items() if key != "report_sha256"
    })
    if expected_hash != actual_hash:
        raise ValueError("V3 offline report hash mismatch")
    if (
        value.get("schema_version") != V3_OFFLINE_REPORT_SCHEMA
        or value.get("status") != "passed"
        or value.get("formal_v3_offline_passed") is not True
        or value.get("task_accuracy_used") is not False
        or not value.get("requirements")
        or not all(item is True for item in value["requirements"].values())
    ):
        raise ValueError("V3 offline report did not pass qualification")
    inputs = value.get("inputs", {})
    expected_inputs = {
        "memory_records_sha256": file_sha256(memory_records_path),
        "side_kv_manifest_sha256": file_sha256(side_kv_manifest_path),
        "retrieval_key_manifest_sha256": file_sha256(retrieval_key_manifest_path),
        "e0_final_report_sha256": file_sha256(e0_final_report_path),
    }
    if any(inputs.get(key) != expected for key, expected in expected_inputs.items()):
        raise ValueError("V3 runtime inputs differ from the offline qualification")
    return value
