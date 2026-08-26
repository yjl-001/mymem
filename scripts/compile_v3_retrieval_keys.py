#!/usr/bin/env python3
"""Compile/qualify V3 embedding keys against the passed layer-24 KV bank."""

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

from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
)
from memgen.experience.v3 import V3_OFFLINE_REPORT_SCHEMA
from memgen.experience.v3_artifacts import (
    authenticate_e0_inputs,
    load_formal_e0_report,
    validate_cross_bank_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.retrieval_keys import (
        RetrievalKeyBankLoader,
        RetrievalKeyCompiler,
        RetrievalKeyCompilerConfig,
    )

    e0_report = load_formal_e0_report(args.e0_final_report)
    authenticate_e0_inputs(
        e0_report=e0_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
    )
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    initial_alignment = validate_cross_bank_metadata(
        records=records, side_manifest=side_manifest
    )
    reasoner = side_manifest["reasoner"]
    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    resolved_model_revision = str(
        getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]
    )
    resolved_tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or reasoner["tokenizer_revision"]
    )
    if (
        resolved_model_revision != reasoner["model_revision"]
        or resolved_tokenizer_revision != reasoner["tokenizer_revision"]
    ):
        raise ValueError("Resolved V3 key compiler reasoner revision drifted")

    compiler = RetrievalKeyCompiler(
        model=model,
        tokenizer=tokenizer,
        reasoner_name=reasoner["model_name"],
        reasoner_revision=reasoner["model_revision"],
        tokenizer_revision=reasoner["tokenizer_revision"],
        config=RetrievalKeyCompilerConfig(layer_number=24),
    )
    bank = compiler.compile(records)
    tensor_path, manifest_path = bank.save(args.output_dir)
    loader = RetrievalKeyBankLoader(
        manifest_path=manifest_path,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    cosine_matrix = torch.matmul(loader.embeddings, loader.embeddings.T)
    diagonal = cosine_matrix.diag()
    row_maximum = cosine_matrix.max(dim=1).values
    self_similarity_dominates = bool(
        torch.all(diagonal >= row_maximum - 1e-6).item()
    )
    off_diagonal = cosine_matrix.masked_fill(
        torch.eye(len(records), dtype=torch.bool),
        torch.finfo(cosine_matrix.dtype).min,
    )
    nearest_other = off_diagonal.max(dim=1).values
    self_other_margins = diagonal - nearest_other
    key_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_alignment = validate_cross_bank_metadata(
        records=records,
        side_manifest=side_manifest,
        key_manifest=key_manifest,
    )
    if initial_alignment != final_alignment:
        raise RuntimeError("V3 key compilation changed cross-bank alignment")

    report = {
        "schema_version": V3_OFFLINE_REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_v3_offline_passed": True,
        "task_accuracy_used": False,
        "compiler_git_revision": git_revision(),
        "configuration": {
            "retrieval_key": {
                "layer_number": 24,
                "hidden_state_tuple_index": 24,
                "representation": "decoder_layer_output",
                "key_source": "sanitized_fields.when_facing",
                "pooling": "last_valid_token",
                "normalization": "l2",
                "retrieval_method": "exact_cosine",
            },
            "kv_value": {
                "layer_number": 24,
                "canonical_pre_rope": True,
                "value_source": "full_when_facing_prefer_avoid_payload",
            },
            "dtype": args.dtype,
            "attention_implementation": "sdpa",
        },
        "inputs": {
            "memory_records_sha256": file_sha256(args.memory_records),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "e0_final_report_sha256": file_sha256(args.e0_final_report),
            "retrieval_key_manifest_sha256": file_sha256(manifest_path),
        },
        "artifacts": {
            "retrieval_key_tensor": {
                "path": tensor_path.name,
                "sha256": file_sha256(tensor_path),
            },
            "retrieval_key_manifest": {
                "path": manifest_path.name,
                "sha256": file_sha256(manifest_path),
                "logical_sha256": key_manifest["manifest_sha256"],
            },
        },
        "alignment": final_alignment,
        "retrieval_key_audit": {
            "self_similarity_min": float(diagonal.min().item()),
            "self_similarity_max": float(diagonal.max().item()),
            "self_similarity_dominates_each_row": self_similarity_dominates,
            "nearest_other_similarity_mean": float(nearest_other.mean().item()),
            "self_nearest_other_margin_mean": float(
                self_other_margins.mean().item()
            ),
            "self_nearest_other_margin_min": float(
                self_other_margins.min().item()
            ),
        },
        "requirements": {
            "formal_e0_side_kv_qualification_reused": True,
            "layer_24_fixed": True,
            "record_key_value_ids_aligned": True,
            "payload_hashes_aligned": True,
            "key_embeddings_finite": True,
            "key_embeddings_l2_normalized": True,
            "self_similarity_dominates_each_row": self_similarity_dominates,
            "exact_cosine_top2_supported": len(records) >= 2,
            "answer_and_task_accuracy_not_used": True,
        },
    }
    if not all(report["requirements"].values()):
        raise RuntimeError("V3 offline qualification requirements were not met")
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = args.output_dir / "v3_offline_report.json"
    write_json(report_path, report)
    print(
        f"[v3-offline] passed records={len(records)} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
