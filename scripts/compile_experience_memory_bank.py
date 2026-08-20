#!/usr/bin/env python3
"""Compile the frozen Phase-1 bank into E0 MemoryRecords, BM25, and side KV."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
    MemoryBuildResult,
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
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
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


def resolve_model_sequence_limit(config: Any, tokenizer: Any) -> int:
    """Resolve the real model/tokenizer context ceiling, ignoring HF sentinels."""

    candidates = []
    for value in (
        getattr(config, "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**9:
            candidates.append(value)
    if not candidates:
        raise ValueError("Unable to resolve a finite model sequence limit")
    return min(candidates)


@dataclass(frozen=True)
class E0ArtifactPaths:
    """Stable artifact locations shared by successful and failed compilation."""

    output_dir: Path
    trace: Path
    payload_audit: Path
    records: Path
    bm25_index: Path
    e0_report: Path

    @classmethod
    def create(cls, output_dir: Path) -> "E0ArtifactPaths":
        resolved_output = output_dir.expanduser()
        resolved_output.mkdir(parents=True, exist_ok=True)
        return cls(
            output_dir=resolved_output,
            trace=resolved_output / "memory_compilation_trace.jsonl",
            payload_audit=resolved_output / "payload_audit_report.json",
            records=resolved_output / "memory_records.v2.jsonl",
            bm25_index=resolved_output / "bm25_index.v1.json",
            e0_report=resolved_output / "e0_report.json",
        )


def persist_no_survivor_diagnostics(
    *,
    paths: E0ArtifactPaths,
    result: MemoryBuildResult,
    approved_bank: Path,
    verified_experiences: Path,
    reasoner_name: str,
    reasoner_revision: str,
    tokenizer_revision: str,
    model_sequence_limit: int,
    sanitizer_config: MemorySanitizerConfig,
    bm25_config: BM25Config,
    analyzer_config: TextAnalyzerConfig,
    layer: int,
    dtype: str,
    text_only: bool,
) -> None:
    """Persist actionable audit evidence before failing a zero-record build."""

    write_jsonl(paths.trace, (item.to_dict() for item in result.trace))
    created_at = datetime.now(timezone.utc).isoformat()
    rejection_counts = dict(result.report.get("rejection_reason_counts", {}))
    failure = {
        "code": "no_runtime_safe_records",
        "message": "No runtime-safe MemoryRecords survived E0 payload audit",
        "rejection_reason_counts": rejection_counts,
    }

    audit_report = dict(result.report)
    audit_report["pre_kv_text_record_set_sha256"] = audit_report.pop(
        "record_set_sha256"
    )
    audit_report.update(
        {
            "created_at": created_at,
            "status": "payload_audit_failed_no_records",
            "failure": failure,
            "inputs": {
                "approved_bank_path": str(approved_bank.resolve()),
                "approved_bank_sha256": file_sha256(approved_bank),
                "verified_experiences_path": str(verified_experiences.resolve()),
                "verified_experiences_sha256": file_sha256(verified_experiences),
            },
            "reasoner": {
                "model_name": reasoner_name,
                "model_revision": reasoner_revision,
                "tokenizer_revision": tokenizer_revision,
            },
            "artifacts": {
                "trace_sha256": file_sha256(paths.trace),
                "records_sha256": None,
                "bm25_index_sha256": None,
            },
        }
    )
    write_json(paths.payload_audit, audit_report)

    e0_report = {
        "schema_version": "experience-memory-e0-report-v2",
        "created_at": created_at,
        "status": "failed_no_runtime_safe_records",
        "formal_e0_passed": False,
        "compiler_git_revision": git_revision(),
        "failure": failure,
        "configuration": {
            "sanitizer": asdict(sanitizer_config),
            "model_sequence_limit": model_sequence_limit,
            "bm25": asdict(bm25_config),
            "analyzer": asdict(analyzer_config),
            "layer": layer,
            "dtype": dtype,
            "text_only": text_only,
        },
        "artifacts": {
            "memory_records": None,
            "compilation_trace": {
                "path": paths.trace.name,
                "sha256": file_sha256(paths.trace),
            },
            "payload_audit": {
                "path": paths.payload_audit.name,
                "sha256": file_sha256(paths.payload_audit),
            },
            "bm25_index": None,
            "side_kv": None,
        },
        "runtime_audit": {
            "required": True,
            "completed": False,
            "blocked_by": "no_runtime_safe_records",
        },
        "artifact_set_sha256": canonical_json_sha256(
            {
                "trace": file_sha256(paths.trace),
                "audit": file_sha256(paths.payload_audit),
            }
        ),
    }
    write_json(paths.e0_report, e0_report)


def main() -> None:
    args = parse_args()
    if args.layer != 24:
        raise ValueError("E0-v1 is frozen to layer 24")

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
    model_sequence_limit = resolve_model_sequence_limit(config, tokenizer)

    approved_records = list(iter_jsonl(args.approved_bank))
    verified_experiences = list(iter_jsonl(args.verified_experiences))
    sanitizer_config = MemorySanitizerConfig(forbid_numeric_literals=True)
    record_compiler = MemoryRecordCompiler(
        tokenizer=tokenizer,
        sanitizer=PayloadSanitizer(sanitizer_config),
        reasoner_name=args.model,
        reasoner_revision=resolved_model_revision,
        tokenizer_revision=resolved_tokenizer_revision,
        model_sequence_limit=model_sequence_limit,
        kv_layer=args.layer,
    )
    builder = MemoryBankBuilder(
        selector=ApprovedMemorySourceSelector(
            allowed_experience_types=("answer_correctness",)
        ),
        compiler=record_compiler,
    )
    result = builder.build(approved_records, verified_experiences)
    paths = E0ArtifactPaths.create(args.output_dir)
    if not result.records:
        persist_no_survivor_diagnostics(
            paths=paths,
            result=result,
            approved_bank=args.approved_bank,
            verified_experiences=args.verified_experiences,
            reasoner_name=args.model,
            reasoner_revision=resolved_model_revision,
            tokenizer_revision=resolved_tokenizer_revision,
            model_sequence_limit=model_sequence_limit,
            sanitizer_config=sanitizer_config,
            bm25_config=BM25Config(k1=args.bm25_k1, b=args.bm25_b),
            analyzer_config=TextAnalyzerConfig(),
            layer=args.layer,
            dtype=args.dtype,
            text_only=args.text_only,
        )
        raise RuntimeError(
            "No runtime-safe MemoryRecords survived E0 payload audit; "
            f"inspect {paths.payload_audit} and {paths.trace}"
        )

    output_dir = paths.output_dir
    trace_path = paths.trace
    audit_path = paths.payload_audit
    records_path = paths.records
    index_path = paths.bm25_index
    e0_report_path = paths.e0_report
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
        expected_model_sequence_limit=model_sequence_limit,
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
        "schema_version": "experience-memory-e0-report-v2",
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
            "model_sequence_limit": model_sequence_limit,
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
