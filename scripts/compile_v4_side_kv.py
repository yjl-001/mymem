#!/usr/bin/env python3
"""Compile approved V4 target/reference process cards into layer-24 side KV.

This is the second offline V4 step.  It accepts only an authenticated,
tensor-free V4 bank built by ``build_v4_repair_bank.py``.  Both target and
reference roles are compiled for provenance and contrast audits, but the
resulting loader exposes target memories only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-records", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reasoner-manifest",
        type=Path,
        help="Authenticated existing side-KV manifest supplying exact reasoner revisions.",
    )
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


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


def _resolve_model_sequence_limit(config: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    for value in (
        getattr(config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**9:
            candidates.append(value)
    if not candidates:
        raise ValueError("Unable to resolve a finite V4 model sequence limit")
    return min(candidates)


def main() -> None:
    args = parse_args()
    if args.layer != 24:
        raise ValueError("The initial V4 side-KV compiler is frozen at layer 24")
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "V4 side-KV compilation requires the repository Torch/Transformers environment"
        ) from exc

    from memgen.model.v4_side_kv import (
        V4SideKVBankLoader,
        V4SideKVCompiler,
        validate_v4_tensor_free_manifest,
    )

    records_path = args.bank_records.expanduser().resolve()
    manifest_path = args.bank_manifest.expanduser().resolve()
    if not records_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("V4 bank records and manifest must both exist")
    records = list(iter_jsonl(records_path))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_v4_tensor_free_manifest(source_manifest)

    reasoner_manifest_sha256: str | None = None
    if args.reasoner_manifest is not None:
        reasoner_manifest_path = args.reasoner_manifest.expanduser().resolve()
        source_reasoner_manifest = json.loads(
            reasoner_manifest_path.read_text(encoding="utf-8")
        )
        logical = {
            key: value
            for key, value in source_reasoner_manifest.items()
            if key != "manifest_sha256"
        }
        if source_reasoner_manifest.get("manifest_sha256") != canonical_json_sha256(
            logical
        ):
            raise ValueError("V4 reasoner manifest hash mismatch")
        source_reasoner = source_reasoner_manifest.get("reasoner", {})
        model_name = str(source_reasoner.get("model_name", ""))
        model_revision_request = str(source_reasoner.get("model_revision", ""))
        tokenizer_revision_request = str(source_reasoner.get("tokenizer_revision", ""))
        if not all((model_name, model_revision_request, tokenizer_revision_request)):
            raise ValueError("V4 reasoner manifest has incomplete provenance")
        if args.model is not None and args.model != model_name:
            raise ValueError("Explicit V4 model differs from reasoner manifest")
        if (
            args.model_revision is not None
            and args.model_revision != model_revision_request
        ):
            raise ValueError("Explicit V4 model revision differs from reasoner manifest")
        if (
            args.tokenizer_revision is not None
            and args.tokenizer_revision != tokenizer_revision_request
        ):
            raise ValueError("Explicit V4 tokenizer revision differs from reasoner manifest")
        reasoner_manifest_sha256 = file_sha256(reasoner_manifest_path)
    else:
        if not args.model or not args.model_revision:
            raise ValueError(
                "Provide --reasoner-manifest or both --model and --model-revision"
            )
        model_name = args.model
        model_revision_request = args.model_revision
        tokenizer_revision_request = args.tokenizer_revision or args.model_revision

    config = AutoConfig.from_pretrained(
        model_name, revision=model_revision_request
    )
    resolved_model_revision = str(
        getattr(config, "_commit_hash", None) or model_revision_request
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=tokenizer_revision_request,
    )
    resolved_tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or tokenizer_revision_request
    )
    sequence_limit = _resolve_model_sequence_limit(config, tokenizer)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision_request,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    resolved_loaded_revision = str(
        getattr(model.config, "_commit_hash", None) or model_revision_request
    )
    if resolved_loaded_revision != resolved_model_revision:
        raise ValueError("V4 config/model revisions resolved to different commits")
    model.eval()

    compiler = V4SideKVCompiler(
        model=model,
        tokenizer=tokenizer,
        reasoner_name=model_name,
        reasoner_revision=resolved_model_revision,
        tokenizer_revision=resolved_tokenizer_revision,
        model_sequence_limit=sequence_limit,
        layer_number=args.layer,
    )
    compiled = compiler.compile(
        records=records,
        source_manifest=source_manifest,
        source_manifest_path=manifest_path,
    )
    output_dir = args.output_dir.expanduser().resolve()
    tensor_path, compiled_manifest_path = compiled.save(output_dir)

    # A fresh authenticated load makes the compiler output self-verifying.
    loader = V4SideKVBankLoader(
        manifest_path=compiled_manifest_path,
        expected_reasoner_name=model_name,
        expected_reasoner_revision=resolved_model_revision,
        expected_tokenizer_revision=resolved_tokenizer_revision,
    )
    if len(loader.bank_ids) != len(records):
        raise ValueError("V4 compiled loader bank coverage differs from source records")
    compiled_manifest = loader.manifest
    report = {
        "schema_version": "memgen-v4-side-kv-compile-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "side_kv_compilation_passed",
        "qualified_for_online_target_loading": True,
        "compiler_git_revision": _git_revision(),
        "configuration": {
            "layer_number": args.layer,
            "all_kv_groups": True,
            "canonical_pre_rope": True,
            "relative_phase_delta": 0,
            "attention_implementation": "sdpa",
            "dtype": args.dtype,
            "device": args.device,
            "target_online_only": True,
            "auxiliary_banks_materialized": False,
        },
        "reasoner": {
            "model_name": model_name,
            "model_revision": resolved_model_revision,
            "tokenizer_revision": resolved_tokenizer_revision,
            "model_sequence_limit": sequence_limit,
        },
        "inputs": {
            "bank_records_path": str(records_path),
            "bank_records_sha256": file_sha256(records_path),
            "bank_manifest_path": str(manifest_path),
            "bank_manifest_sha256": file_sha256(manifest_path),
            "bank_manifest_logical_sha256": source_manifest["manifest_sha256"],
            "reasoner_manifest_sha256": reasoner_manifest_sha256,
        },
        "artifacts": {
            "side_kv_tensor": {
                "path": tensor_path.name,
                "sha256": file_sha256(tensor_path),
            },
            "side_kv_manifest": {
                "path": compiled_manifest_path.name,
                "sha256": file_sha256(compiled_manifest_path),
                "logical_sha256": compiled_manifest["manifest_sha256"],
            },
        },
        "counts": {
            "bank_count": compiled_manifest["bank_count"],
            "role_record_count": compiled_manifest["record_count"],
            "target_count": len(loader.bank_ids),
            "reference_count": sum(
                item.get("role") == "reference"
                for item in compiled_manifest["records"]
            ),
        },
    }
    report_path = output_dir / "v4_side_kv_compile_report.json"
    _write_json(report_path, report)
    print(
        "[v4-side-kv] complete "
        f"banks={len(loader.bank_ids)} tensor={tensor_path} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-side-kv] error: {exc}", file=sys.stderr)
        raise
