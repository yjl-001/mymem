"""Shared local I/O and authentication for the complete V4.3 pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from memgen.experience.v4_3_bank import (
    CONSTRUCTION_POLICY, authenticate, canonical_hash, file_hash, validate_manifest,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSONL objects: {path}")
    return rows


def atomic_json(path: Path, value: Mapping[str, Any], *, immutable: bool = False) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink output: {path}")
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing artifact belongs to another input/profile: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".v43-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def implementation_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    return {name: file_hash(ROOT / name) for name in paths}


def local_artifact(manifest_path: Path, entry: Mapping[str, Any]) -> Path:
    name = entry.get("path")
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Artifact must be a local filename")
    path = manifest_path.parent / name
    if path.is_symlink() or not path.is_file() or file_hash(path) != entry.get("sha256"):
        raise ValueError(f"Artifact missing or corrupted: {path}")
    return path


def load_construction(directory: Path) -> dict[str, Any]:
    """Authenticate the bundle, tier partitions, lineage, and current builder."""
    bundle = read_json(directory / "construction_bundle_manifest.json")
    authenticate(bundle, "manifest_sha256", "V4.3 construction bundle")
    if bundle.get("schema_version") != "memgen-v4.3-construction-bundle-v1":
        raise ValueError("Unexpected V4.3 bundle schema")
    outputs = {}
    for name, entry in bundle["artifacts"].items():
        path = local_artifact(directory / "construction_bundle_manifest.json",
                              {"path": name, "sha256": entry["file_sha256"]})
        value = read_jsonl(path) if name.endswith(".jsonl") else read_json(path)
        if canonical_hash(value) != entry["logical_sha256"]:
            raise ValueError(f"Construction logical hash mismatch: {name}")
        outputs[name] = value
    if outputs["construction_policy.json"] != CONSTRUCTION_POLICY:
        raise ValueError("Construction policy drifted")
    expected = implementation_hashes(("memgen/experience/v4_3_bank.py", "scripts/build_v4_3_unified_bank.py"))
    if bundle["inputs"]["implementation_sha256"] != expected:
        raise ValueError("Construction implementation identity drifted")
    candidates = outputs["candidate_bank_records.jsonl"]
    if len(candidates) != 17 or len({r["source_v42_bank_id"] for r in candidates}) != 17:
        raise ValueError("V4.3 candidate coverage mismatch")
    for record in candidates:
        validate_record(record)
    for tier in ("primary", "conditional"):
        records, manifest = outputs[f"{tier}_bank_records.jsonl"], outputs[f"{tier}_bank_manifest.json"]
        validate_manifest(manifest, records)
        if records != [r for r in candidates if r["quality_tier"] == tier and r["qualification"]["construction_qualified"]]:
            raise ValueError("V4.3 tier partition mismatch")
        if manifest["inputs"] != bundle["inputs"]:
            raise ValueError("V4.3 input binding mismatch")
    lineage = outputs["source_v42_to_v43_lineage.json"]
    authenticate(lineage, "lineage_sha256", "V4.3 lineage")
    if (lineage["source_v42_to_v43"] != {r["source_v42_bank_id"]: r["bank_id"] for r in candidates}
            or lineage["record_sha256"] != {r["bank_id"]: r["record_sha256"] for r in candidates}
            or lineage["inputs"] != bundle["inputs"]):
        raise ValueError("V4.3 lineage identity mismatch")
    outputs["bundle"] = bundle
    return outputs
