#!/usr/bin/env python3
"""Seal or verify one canonical GSM8K Phase-1 + V3.4 risk lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.input_lineage import (
    build_phase1_risk_lineage,
    validate_sealed_lineage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--lineage-root", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--risk-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path)
    parser.add_argument("--side-kv-manifest", type=Path)
    return parser.parse_args()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_environment(
    path: Path, manifest_path: Path, manifest: dict[str, Any]
) -> None:
    usage = manifest["canonical_usage"]
    downstream = manifest["downstream_v4"]
    compatible = downstream.get("compatible")
    if compatible is True:
        compatibility_value = "true"
    elif compatible is False:
        compatibility_value = "false"
    else:
        compatibility_value = "not_checked"
    lines = [
        "# Generated from an authenticated, immutable MemGen input lineage.",
        f"# lineage_manifest_sha256={manifest['manifest_sha256']}",
        f"export MEMGEN_INPUT_LINEAGE_ID={shlex.quote(manifest['lineage_id'])}",
        f"export MEMGEN_PHASE1_DIR={shlex.quote(usage['phase1_dir'])}",
        "export MEMGEN_TOKEN_RISK_ARTIFACT="
        + shlex.quote(usage["token_risk_artifact"]),
        "export MEMGEN_INPUT_LINEAGE_MANIFEST="
        + shlex.quote(str(manifest_path)),
        "export MEMGEN_DOWNSTREAM_V4_COMPATIBLE="
        + shlex.quote(compatibility_value),
    ]
    if compatible is True:
        lines.extend(
            (
                "export MEMGEN_V4_BANK_MANIFEST="
                + shlex.quote(downstream["bank_manifest"]["path"]),
                "export MEMGEN_V4_SIDE_KV_MANIFEST="
                + shlex.quote(downstream["side_kv_manifest"]["path"]),
            )
        )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.environment_output = args.environment_output.expanduser().resolve()
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        validate_sealed_lineage(existing, path=args.output)
        if existing.get("lineage_id") != args.lineage_id:
            raise ValueError("Existing sealed lineage uses another lineage ID")
        _write_environment(args.environment_output, args.output, existing)
        print(
            f"[phase1-risk-lineage] verified sealed lineage: {args.output} "
            f"sha256={existing['manifest_sha256']}"
        )
        return

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The server Torch environment is required to authenticate the risk artifact"
        ) from exc
    artifact_path = args.risk_dir.expanduser().resolve() / (
        "token-entropy-risk-gate-v3.4.pt"
    )
    risk_artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=False
    )
    manifest = build_phase1_risk_lineage(
        lineage_id=args.lineage_id,
        lineage_root=args.lineage_root.expanduser().resolve(),
        phase1_dir=args.phase1_dir.expanduser().resolve(),
        risk_dir=args.risk_dir.expanduser().resolve(),
        risk_artifact=risk_artifact,
        repository_revision=_git_revision(),
        bank_manifest_path=(
            args.bank_manifest.expanduser().resolve() if args.bank_manifest else None
        ),
        side_kv_manifest_path=(
            args.side_kv_manifest.expanduser().resolve()
            if args.side_kv_manifest
            else None
        ),
    )
    _write_json_atomic(args.output, manifest)
    _write_environment(args.environment_output, args.output, manifest)
    print(
        f"[phase1-risk-lineage] sealed: {args.output} "
        f"sha256={manifest['manifest_sha256']}"
    )
    print(
        "[phase1-risk-lineage] downstream_v4="
        f"{manifest['downstream_v4']['status']}"
    )
    print(f"[phase1-risk-lineage] use: source {args.environment_output}")


if __name__ == "__main__":
    main()
