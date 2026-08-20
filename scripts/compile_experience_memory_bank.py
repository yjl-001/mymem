#!/usr/bin/env python3
"""Compile the frozen Phase-1 bank into E0 MemoryRecords, BM25, and side KV."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.memory import (
    ApprovedMemorySourceSelector,
    MemoryArtifactAuditor,
    MemoryBankBuilder,
    MemoryRecordCompiler,
    MemorySanitizerConfig,
    PayloadSanitizer,
)
from memgen.experience.retrieval import (
    BM25Config,
    BM25MemoryIndex,
    TextAnalyzer,
    TextAnalyzerConfig,
)
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--verified-experiences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-payload-tokens", type=int, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--source-overlap-ngram-tokens", type=int, default=8)
    parser.add_argument("--evidence-overlap-ngram-tokens", type=int, default=6)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Build payload/BM25 audit artifacts without loading the reasoner. "
            "Not a formal E0 pass."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
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
    if args.layer != 24:
        raise ValueError("E0-v1 is frozen to layer 24")
    if args.max_payload_tokens <= 0:
        raise ValueError("--max-payload-tokens must be positive")

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "E0 compilation requires the repository Torch/Transformers environment"
        ) from exc

    tokenizer_revision_request = args.tokenizer_revision or args.model_revision
    config = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    resolved_model_revision = str(getattr(config, "_commit_hash", None) or args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=tokenizer_revision_request,
    )
    resolved_tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or tokenizer_revision_request
    )

    approved_records = list(iter_jsonl(args.approved_bank))
    verified_experiences = list(iter_jsonl(args.verified_experiences))
    sanitizer_config = MemorySanitizerConfig(
        max_payload_tokens=args.max_payload_tokens,
        source_overlap_ngram_tokens=args.source_overlap_ngram_tokens,
        evidence_overlap_ngram_tokens=args.evidence_overlap_ngram_tokens,
        forbid_numeric_literals=True,
    )
    record_compiler = MemoryRecordCompiler(
        tokenizer=tokenizer,
        sanitizer=PayloadSanitizer(sanitizer_config),
        reasoner_name=args.model,
        reasoner_revision=resolved_model_revision,
        tokenizer_revision=resolved_tokenizer_revision,
        kv_layer=args.layer,
    )
    builder = MemoryBankBuilder(
        selector=ApprovedMemorySourceSelector(
            allowed_experience_types=("answer_correctness",)
        ),
        compiler=record_compiler,
    )
    result = builder.build(approved_records, verified_experiences)
    if not result.records:
        raise RuntimeError("No runtime-safe MemoryRecords survived E0 payload audit")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "memory_compilation_trace.jsonl"
    audit_path = output_dir / "payload_audit_report.json"
    records_path = output_dir / "memory_records.v1.jsonl"
    index_path = output_dir / "bm25_index.v1.json"
    e0_report_path = output_dir / "e0_report.json"
    write_jsonl(trace_path, (item.to_dict() for item in result.trace))

    records = list(result.records)
    side_kv_artifacts: dict[str, Any] | None = None
    if not args.text_only:
        from memgen.model.side_kv import (
            CanonicalSideKVCompiler,
            SideKVCompilerConfig,
        )

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.model_revision,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(args.device)
        model.eval()
        compiler = CanonicalSideKVCompiler(
            model=model,
            tokenizer=tokenizer,
            reasoner_name=args.model,
            reasoner_revision=resolved_model_revision,
            tokenizer_revision=resolved_tokenizer_revision,
            config=SideKVCompilerConfig(layer_number=args.layer),
        )
        bank = compiler.compile(records)
        tensor_path, manifest_path = bank.save(output_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry_by_id = {
            str(entry["memory_id"]): entry for entry in manifest["records"]
        }
        records = [
            replace(
                record,
                canonical_pre_rope_kv={
                    "compiled": True,
                    "relative_phase_delta": 0,
                    "artifact": tensor_path.name,
                    "artifact_sha256": manifest["tensor_artifact"]["sha256"],
                    "manifest": manifest_path.name,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "record_index": int(entry_by_id[record.memory_id]["index"]),
                    "kv_valid_slot_count": int(
                        entry_by_id[record.memory_id]["kv_valid_slot_count"]
                    ),
                    "tensor_layout": "kv_head,slot,head_dim",
                },
            )
            for record in records
        ]
        side_kv_artifacts = {
            "tensor_path": tensor_path.name,
            "tensor_sha256": file_sha256(tensor_path),
            "manifest_path": manifest_path.name,
            "manifest_sha256": file_sha256(manifest_path),
            "logical_manifest_sha256": manifest["manifest_sha256"],
        }

    artifact_audit = MemoryArtifactAuditor(
        tokenizer=tokenizer,
        expected_token_budget=args.max_payload_tokens,
        expected_kv_layer=args.layer,
        expected_kv_compiled=not args.text_only,
    ).assert_valid(records)
    write_jsonl(records_path, (record.to_dict() for record in records))
    analyzer = TextAnalyzer(TextAnalyzerConfig())
    bm25 = BM25MemoryIndex(
        records=records,
        analyzer=analyzer,
        config=BM25Config(k1=args.bm25_k1, b=args.bm25_b),
    )
    write_json(index_path, bm25.to_dict())

    audit_report = dict(result.report)
    audit_report["pre_kv_text_record_set_sha256"] = audit_report.pop(
        "record_set_sha256"
    )
    audit_report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "payload_audit_passed",
            "post_sanitization_forbidden_pattern_count": 0,
            "artifact_validation": artifact_audit,
            "serialized_record_set_sha256": artifact_audit["record_set_sha256"],
            "inputs": {
                "approved_bank_path": str(args.approved_bank.resolve()),
                "approved_bank_sha256": file_sha256(args.approved_bank),
                "verified_experiences_path": str(args.verified_experiences.resolve()),
                "verified_experiences_sha256": file_sha256(args.verified_experiences),
            },
            "reasoner": {
                "model_name": args.model,
                "model_revision": resolved_model_revision,
                "tokenizer_revision": resolved_tokenizer_revision,
            },
            "artifacts": {
                "trace_sha256": file_sha256(trace_path),
                "records_sha256": file_sha256(records_path),
                "bm25_index_sha256": file_sha256(index_path),
            },
        }
    )
    write_json(audit_path, audit_report)

    e0_report = {
        "schema_version": "experience-memory-e0-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "text_only_passed"
            if args.text_only
            else "kv_compilation_passed_pending_runtime_audit"
        ),
        "formal_e0_passed": False,
        "compiler_git_revision": git_revision(),
        "configuration": {
            "sanitizer": asdict(sanitizer_config),
            "bm25": asdict(bm25.config),
            "analyzer": asdict(analyzer.config),
            "layer": args.layer,
            "dtype": args.dtype,
            "text_only": args.text_only,
        },
        "artifacts": {
            "memory_records": {
                "path": records_path.name,
                "sha256": file_sha256(records_path),
            },
            "compilation_trace": {
                "path": trace_path.name,
                "sha256": file_sha256(trace_path),
            },
            "payload_audit": {
                "path": audit_path.name,
                "sha256": file_sha256(audit_path),
            },
            "bm25_index": {
                "path": index_path.name,
                "sha256": file_sha256(index_path),
                "logical_sha256": bm25.to_dict()["index_sha256"],
            },
            "side_kv": side_kv_artifacts,
        },
        "runtime_audit": {
            "required": True,
            "completed": False,
            "requirements": [
                "memory_attention_mass_recorded_and_positive",
                "canonical_rope_shared_phase_identity",
                "disabled_path_logit_parity",
                "native_cache_prefix_preserved",
                "native_cache_length_excludes_memory_slots",
                "active_memory_changes_logits_above_noise_floor",
                "calibration_cases_answer_blind_and_preanswer",
            ],
        },
        "artifact_set_sha256": canonical_json_sha256(
            {
                "records": file_sha256(records_path),
                "trace": file_sha256(trace_path),
                "audit": file_sha256(audit_path),
                "index": file_sha256(index_path),
                "side_kv": side_kv_artifacts,
            }
        ),
    }
    write_json(e0_report_path, e0_report)
    print(
        f"[experience-memory] records={len(records)} status={e0_report['status']} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
