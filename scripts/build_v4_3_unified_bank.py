#!/usr/bin/env python3
"""Build the offline V4.3 unified bank using only surviving local artifacts.

The production counts and consensus thresholds are frozen in v4_3_bank.py.
No model, network client, provider credential, or evaluation path is needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_bank import (
    build_outputs, canonical_hash, file_hash, seal, text_hash, validate_manifest,
)

IMPLEMENTATION_PATHS = ("memgen/experience/v4_3_bank.py", "scripts/build_v4_3_unified_bank.py")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSONL object: {path}")
        result.append(value)
    return result


def encode_outputs(outputs: Mapping[str, Any]) -> dict[str, bytes]:
    payloads = {}
    for name, value in outputs.items():
        if name.endswith(".jsonl"):
            text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in value)
        else:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        payloads[name] = text.encode("utf-8")
    bundle = seal({
        "schema_version": "memgen-v4.3-construction-bundle-v1",
        "inputs": outputs["construction_report.json"]["inputs"],
        "artifacts": {name: {"file_sha256": text_hash(payload.decode("utf-8")),
                              "logical_sha256": canonical_hash(outputs[name])}
                      for name, payload in sorted(payloads.items())},
        "offline_only": True, "qualified_for_online_use": False,
    }, "manifest_sha256")
    payloads["construction_bundle_manifest.json"] = (json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return payloads


def write_or_validate(output_dir: Path, payloads: Mapping[str, bytes], *, resume: bool, validate_only: bool = False) -> None:
    """Validate every existing file before writing any missing artifact.

    A failed interrupted build can resume without deleting valid files. The
    bundle seal is installed last, so incomplete directories are not sealed.
    """
    if output_dir.is_symlink():
        raise ValueError("V4.3 output directory cannot be a symlink")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("V4.3 output must be a directory")
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    if existing and not (resume or validate_only):
        raise ValueError("V4.3 output exists; pass --resume to authenticate it")
    for path in existing:
        if path.name not in payloads or path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected output entry; refusing to overwrite: {path}")
        if path.read_bytes() != payloads[path.name]:
            raise ValueError(f"V4.3 output hash/content drift: {path}")
    missing = [name for name in payloads if not (output_dir / name).exists()]
    if validate_only:
        if missing:
            raise ValueError(f"V4.3 output incomplete: {missing}")
        return
    if "construction_bundle_manifest.json" not in missing and missing:
        raise ValueError("Sealed V4.3 output lost artifacts; refusing to silently repair")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        # Exclusive link installation avoids clobbering a concurrent writer.
        fd, temporary = tempfile.mkstemp(prefix=".v43-write-", dir=output_dir.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, output_dir / name)
        finally:
            Path(temporary).unlink(missing_ok=True)


def construct(*, source_dir: Path, packets_path: Path, policy_path: Path) -> dict[str, Any]:
    paths = {"records": source_dir / "bank_records.jsonl", "manifest": source_dir / "bank_manifest.json",
             "packets": packets_path, "policy": policy_path}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing required local artifact: {path}")
    hashes = {name: file_hash(path) for name, path in paths.items()}
    records, packets = read_jsonl(paths["records"]), read_jsonl(paths["packets"])
    manifest, policy = read_json(paths["manifest"]), read_json(paths["policy"])
    if hashes != {name: file_hash(path) for name, path in paths.items()}:
        raise ValueError("Input changed while being read")
    outputs = build_outputs(records=records, manifest=manifest, packets=packets, policy=policy,
                            input_hashes=hashes,
                            implementation_hashes={p: file_hash(ROOT / p) for p in IMPLEMENTATION_PATHS})
    for tier in ("primary", "conditional"):
        validate_manifest(outputs[f"{tier}_bank_manifest.json"], outputs[f"{tier}_bank_records.jsonl"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="V4.2 curated directory")
    parser.add_argument("--semantic-packets", type=Path, required=True)
    parser.add_argument("--curation-policy", type=Path, default=ROOT / "configs/experiments/gsm8k/v4_2_local_curation_policy.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Recompute and authenticate all existing outputs without writing")
    args = parser.parse_args()
    # Check the provided path before resolve removes the final symlink.
    if args.output_dir.expanduser().is_symlink():
        raise ValueError("V4.3 output directory cannot be a symlink")
    source = args.source_dir.expanduser().resolve()
    packets = args.semantic_packets.expanduser().resolve()
    policy = args.curation_policy.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if (output == ROOT or output == source or source in output.parents
            or output in source.parents or output == ROOT / "docs/figures"
            or ROOT / "docs/figures" in output.parents
            or any(output == p or output in p.parents for p in (packets, policy))):
        raise ValueError("Output must be separate from the repository root, protected figures, and input artifacts")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    print(f"[v4.3] stage=construction repo_revision={revision}", flush=True)
    print(f"[v4.3] source={source}\n[v4.3] packets={packets}\n[v4.3] policy={policy}\n[v4.3] output={output}", flush=True)
    outputs = construct(source_dir=source, packets_path=packets, policy_path=policy)
    write_or_validate(output, encode_outputs(outputs), resume=args.resume, validate_only=args.validate_only)
    report = outputs["construction_report.json"]
    summary = {k: report[k] for k in ("status", "candidate_count", "consumed_evidence_count", "qualified_tier_counts",
                                      "quarantined_count", "external_api_calls_made", "model_loaded", "qualified_for_online_use")}
    print("[v4.3] " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"[v4.3] authenticated_bundle={output / 'construction_bundle_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
