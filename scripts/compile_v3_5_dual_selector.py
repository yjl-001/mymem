#!/usr/bin/env python3
"""Compile and qualify the answer-blind MemGen V3.5 dual-key bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.v3_5_selector import (
    V35_APPLICABILITY_CALIBRATION_SCHEMA,
    calibrate_applicability_selector,
    v35_artifact_sha256,
)
from memgen.experience.v3_artifacts import (
    authenticate_e0_inputs,
    load_formal_e0_report,
    load_v3_offline_report,
    validate_cross_bank_metadata,
)


V35_OFFLINE_REPORT_SCHEMA = "experience-memory-v3.5-offline-report-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--verified-experiences", type=Path, required=True)
    parser.add_argument(
        "--v3-retrieval-key-manifest", type=Path, required=True
    )
    parser.add_argument("--v3-offline-report", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--applicability-calibration-output", type=Path)
    parser.add_argument(
        "--applicability-calibration-markdown-output", type=Path
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16",),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git_revision() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("V3.5 compiler requires an auditable git revision") from exc
    if not value:
        raise RuntimeError("V3.5 compiler resolved an empty git revision")
    return value


def compiler_repository_state() -> dict[str, Any]:
    """Bind offline artifacts to the scoped V3.5 implementation worktree."""

    from memgen.model.v3_5_retrieval import (
        v35_implementation_files_sha256,
    )

    implementation_files = v35_implementation_files_sha256()
    try:
        diff = subprocess.check_output(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                *implementation_files,
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "V3.5 compiler requires an auditable scoped git diff"
        ) from exc
    return {
        "git_revision": git_revision(),
        "tracked_diff_sha256": text_sha256(diff),
        "implementation_files_sha256": implementation_files,
        "implementation_set_sha256": canonical_json_sha256(
            implementation_files
        ),
    }


def _resolved_revision(value: Any, fallback: str) -> str:
    resolved = str(value or fallback)
    if not resolved:
        raise ValueError("V3.5 reasoner/tokenizer revision is empty")
    return resolved


def _source_questions(
    *,
    records: Sequence[MemoryRecord],
    verified_experiences: Sequence[Mapping[str, Any]],
    question_encoder: Any,
    applicability_embeddings: Any,
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import torch
    from memgen.model.v3_5_retrieval import (
        canonicalize_v35_query_embedding,
    )

    verified_by_id: dict[str, Mapping[str, Any]] = {}
    for value in verified_experiences:
        experience_id = str(value.get("experience_id", ""))
        if not experience_id or experience_id in verified_by_id:
            raise ValueError(
                "V3.5 source self-retrieval requires unique verified IDs"
            )
        verified_by_id[experience_id] = value
    memory_ids = [str(entry["memory_id"]) for entry in entries]
    if memory_ids != [record.memory_id for record in records]:
        raise ValueError("V3.5 source self-retrieval record order drifted")
    pairs: list[dict[str, Any]] = []
    for record in records:
        verified = verified_by_id.get(record.source_experience_id)
        if verified is None:
            raise ValueError(
                f"{record.memory_id} is missing its verified source question"
            )
        context = verified.get("context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(
                f"{record.memory_id} verified source context is empty"
            )
        encoded = question_encoder.encode(context)
        query = canonicalize_v35_query_embedding(
            encoded.embedding,
            expected_width=int(applicability_embeddings.shape[1]),
            owner="source question",
        )
        scores = torch.mv(applicability_embeddings, query)
        ranked_indices = sorted(
            range(len(memory_ids)),
            key=lambda index: (-float(scores[index].item()), memory_ids[index]),
        )
        own_index = memory_ids.index(record.memory_id)
        own_rank = ranked_indices.index(own_index) + 1
        raw_own_score = float(scores[own_index].item())
        if raw_own_score < -1.00001 or raw_own_score > 1.00001:
            raise RuntimeError("V3.5 exact cosine escaped its numerical range")
        own_score = max(-1.0, min(1.0, raw_own_score))
        pairs.append({
            "memory_id": record.memory_id,
            "source_experience_id": record.source_experience_id,
            "own_memory_rank": own_rank,
            "own_positive_score": own_score,
            "question_query": encoded.to_dict(),
        })
    return pairs


def _applicability_markdown(artifact: Mapping[str, Any]) -> str:
    calibration = artifact.get("calibration", {})
    metrics = artifact.get("metrics", {})
    train = metrics.get("train") or {}
    heldout = metrics.get("holdout") or {}
    requirements = artifact.get("requirements", {})
    lines = [
        "# MemGen V3.5 Applicability Calibration",
        "",
        f"- Status: `{artifact.get('status')}`",
        "- Task accuracy used: `false`",
        "- Answer or reward used: `false`",
        f"- Frozen shortlist k: `{calibration.get('shortlist_k')}`",
        (
            "- Inclusive applicability score floor: "
            f"`{calibration.get('minimum_applicability_score')}`"
        ),
        f"- Train Recall@k: `{train.get('recall_at_k')}`",
        f"- Heldout Recall@k: `{heldout.get('recall_at_k')}`",
        (
            "- Heldout own-positive retention: "
            f"`{calibration.get('heldout_own_positive_retained_fraction')}`"
        ),
        "",
        "This is a positive-retention floor, not a complete relevance classifier.",
        "Other memories were not treated as strict negative examples.",
        "",
        "## Requirements",
        "",
    ]
    lines.extend(
        f"- {name}: `{'passed' if value is True else 'failed'}`"
        for name, value in sorted(requirements.items())
    )
    return "\n".join(lines)


def _offline_markdown(report: Mapping[str, Any]) -> str:
    calibration = report.get("applicability_calibration", {})
    reproduction = report.get("applicability_reproduction_audit", {})
    lines = [
        "# MemGen V3.5 Dual-Key Offline Qualification",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Formal V3.5 offline passed: `{str(report.get('formal_v3_5_offline_passed')).lower()}`",
        f"- Records: `{report.get('record_count')}`",
        "- Layer: `24`",
        "- Pooling: `last_valid_token`",
        "- Normalization: `l2`",
        (
            "- Reproduced legacy applicability embeddings: "
            f"`{reproduction.get('exact_reproduction_count')}/"
            f"{reproduction.get('record_count')}`"
        ),
        f"- Frozen shortlist k: `{calibration.get('shortlist_k')}`",
        (
            "- Applicability floor: "
            f"`{calibration.get('minimum_applicability_score')}`"
        ),
        "- Task accuracy/answers/rewards used: `false`",
        "",
        "The dynamic key contains only verbatim sanitized `when_facing` plus "
        "the V3.5 answer-format-canonicalized, E0-sanitized Phase-1 "
        "`transferable_decision`. The existing side-KV bank is referenced and "
        "aligned; it is not recompiled.",
        "",
        "## Requirements",
        "",
    ]
    lines.extend(
        f"- {name}: `{'passed' if value is True else 'failed'}`"
        for name, value in sorted(report.get("requirements", {}).items())
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.retrieval_keys import RetrievalKeyBankLoader
    from memgen.model.v3_5_retrieval import (
        DualRetrievalKeyBankLoader,
        DualRetrievalKeyCompiler,
        DualRetrievalKeyCompilerConfig,
        QuestionOnlyEncoder,
        validate_v35_split_manifest,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    applicability_path = (
        args.applicability_calibration_output
        or args.output_dir / "applicability_calibration.json"
    )
    applicability_markdown_path = (
        args.applicability_calibration_markdown_output
        or applicability_path.with_suffix(".md")
    )

    e0_report = load_formal_e0_report(args.e0_final_report)
    authenticate_e0_inputs(
        e0_report=e0_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
    )
    v3_offline_report = load_v3_offline_report(
        args.v3_offline_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
        retrieval_key_manifest_path=args.v3_retrieval_key_manifest,
        e0_final_report_path=args.e0_final_report,
    )
    if v3_offline_report.get("configuration", {}).get("dtype") != "bfloat16":
        raise ValueError("V3.5 requires a bfloat16 legacy applicability bank")
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    approved_records = tuple(iter_jsonl(args.approved_bank))
    verified_experiences = tuple(iter_jsonl(args.verified_experiences))
    split_manifest = validate_v35_split_manifest(
        json.loads(args.split_manifest.read_text(encoding="utf-8"))
    )
    side_manifest = json.loads(
        args.side_kv_manifest.read_text(encoding="utf-8")
    )
    old_key_bank = RetrievalKeyBankLoader(
        manifest_path=args.v3_retrieval_key_manifest
    )
    alignment = validate_cross_bank_metadata(
        records=records,
        side_manifest=side_manifest,
        key_manifest=old_key_bank.manifest,
    )

    reasoner = side_manifest["reasoner"]
    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    resolved_model_revision = _resolved_revision(
        getattr(model.config, "_commit_hash", None),
        str(reasoner["model_revision"]),
    )
    resolved_tokenizer_revision = _resolved_revision(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        str(reasoner["tokenizer_revision"]),
    )
    if (
        resolved_model_revision != reasoner["model_revision"]
        or resolved_tokenizer_revision != reasoner["tokenizer_revision"]
    ):
        raise ValueError("Resolved V3.5 reasoner/tokenizer revision drifted")

    old_tensor_info = old_key_bank.manifest.get("tensor_artifact", {})
    old_tensor_relative = Path(str(old_tensor_info.get("path", "")))
    if (
        not old_tensor_relative.parts
        or old_tensor_relative.is_absolute()
        or ".." in old_tensor_relative.parts
    ):
        raise ValueError("V3.5 reused V3 key tensor path is unsafe")
    old_artifact_root = args.v3_retrieval_key_manifest.parent.resolve()
    old_tensor_path = (old_artifact_root / old_tensor_relative).resolve()
    if (
        old_artifact_root not in old_tensor_path.parents
        or not old_tensor_path.is_file()
        or file_sha256(old_tensor_path) != old_tensor_info.get("sha256")
    ):
        raise ValueError("V3.5 cannot authenticate the reused V3 key tensor")
    repository = compiler_repository_state()
    revision = str(repository["git_revision"])
    provenance = {
        "memory_records_sha256": file_sha256(args.memory_records),
        "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
        "e0_final_report_sha256": file_sha256(args.e0_final_report),
        "v3_retrieval_key_manifest_sha256": file_sha256(
            args.v3_retrieval_key_manifest
        ),
        "v3_retrieval_key_tensor_sha256": file_sha256(old_tensor_path),
        "v3_offline_report_sha256": file_sha256(args.v3_offline_report),
        "phase1_approved_bank_sha256": file_sha256(args.approved_bank),
        "verified_experiences_sha256": file_sha256(args.verified_experiences),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "split_manifest_logical_sha256": split_manifest["manifest_sha256"],
        "dataset_revision": split_manifest["dataset"]["revision"],
        "compiler_git_revision": revision,
        "compiler_tracked_diff_sha256": repository[
            "tracked_diff_sha256"
        ],
        "compiler_implementation_files_sha256": repository[
            "implementation_files_sha256"
        ],
        "compiler_implementation_set_sha256": repository[
            "implementation_set_sha256"
        ],
    }

    compiler = DualRetrievalKeyCompiler(
        model=model,
        tokenizer=tokenizer,
        reasoner_name=str(reasoner["model_name"]),
        reasoner_revision=str(reasoner["model_revision"]),
        tokenizer_revision=str(reasoner["tokenizer_revision"]),
        config=DualRetrievalKeyCompilerConfig(layer_number=24),
    )
    compiled = compiler.compile(
        records=records,
        approved_records=approved_records,
        verified_experiences=verified_experiences,
        applicability_key_bank=old_key_bank,
        side_kv_manifest=side_manifest,
        split_manifest=split_manifest,
        artifact_provenance=provenance,
    )
    tensor_path, manifest_path = compiled.save(args.output_dir)
    dual_loader = DualRetrievalKeyBankLoader(
        manifest_path=manifest_path,
        expected_reasoner_name=str(reasoner["model_name"]),
        expected_reasoner_revision=str(reasoner["model_revision"]),
        expected_tokenizer_revision=str(reasoner["tokenizer_revision"]),
        expected_input_hashes=provenance,
    )

    question_encoder = QuestionOnlyEncoder(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        layer_number=24,
    )
    source_pairs = _source_questions(
        records=records,
        verified_experiences=verified_experiences,
        question_encoder=question_encoder,
        applicability_embeddings=dual_loader.applicability_embeddings,
        entries=dual_loader.entries,
    )
    calibration_result = calibrate_applicability_selector(
        source_pairs, memory_count=len(records)
    )
    calibration_source = {
        **provenance,
        "dual_key_manifest_sha256": file_sha256(manifest_path),
        "dual_key_manifest_logical_sha256": dual_loader.manifest_sha256,
        "dual_key_tensor_sha256": file_sha256(tensor_path),
        "source_question_encoder": (
            "verified_experience.context.strip_question_only"
        ),
    }
    applicability_artifact = {
        "schema_version": V35_APPLICABILITY_CALIBRATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": calibration_source,
        **calibration_result,
        "source_pair_audit": source_pairs,
    }
    applicability_artifact["artifact_sha256"] = v35_artifact_sha256(
        applicability_artifact
    )
    applicability_artifact["logical_sha256"] = applicability_artifact[
        "artifact_sha256"
    ]
    write_json(applicability_path, applicability_artifact)
    write_text(
        applicability_markdown_path,
        _applicability_markdown(applicability_artifact),
    )

    calibration_passed = applicability_artifact["status"] == "passed" and all(
        value is True
        for value in applicability_artifact.get("requirements", {}).values()
    )
    reproduction = dual_loader.manifest["applicability_reproduction_audit"]
    requirements = {
        "formal_e0_side_kv_qualification_reused": True,
        "formal_v3_offline_key_qualification_reused": True,
        "approved_verified_memory_join_unique": True,
        "source_record_and_provenance_aligned": True,
        "phase1_split_manifest_logical_hash_authenticated": True,
        "phase1_split_overlap_and_dataset_revision_authenticated": True,
        "phase1_sources_match_split_members": True,
        "compiler_implementation_identity_bound": True,
        "layer_24_fixed": True,
        "last_valid_token_pooling_fixed": True,
        "l2_normalization_fixed": True,
        "sdpa_reasoner_fixed": True,
        "old_applicability_embeddings_reproduced_per_record": (
            reproduction.get("all_exact") is True
            and reproduction.get("exact_reproduction_count") == len(records)
        ),
        "dynamic_text_excludes_verification_reference_and_avoid": True,
        "record_key_value_ids_and_order_aligned": (
            dual_loader.manifest["record_order_sha256"]
            == alignment["record_order_sha256"]
        ),
        "payload_hash_kv_slots_and_layer_aligned": True,
        "dual_embeddings_finite_l2_and_hash_authenticated": True,
        "source_question_encoder_is_question_only": True,
        "applicability_calibration_qualified": calibration_passed,
        "task_accuracy_not_used": True,
        "answer_or_reward_not_used": True,
        "side_kv_values_not_recompiled": True,
    }
    passed = all(value is True for value in requirements.values())
    report = {
        "schema_version": V35_OFFLINE_REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "not_qualified",
        "formal_v3_5_offline_passed": passed,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "compiler_git_revision": revision,
        "record_count": len(records),
        "configuration": {
            "layer_number": 24,
            "representation": "decoder_layer_output",
            "pooling": "last_valid_token",
            "normalization": "l2",
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "applicability_key_source": "sanitized_fields.when_facing",
            "dynamic_key_source": (
                "when_facing_plus_sanitized_transferable_decision_only"
            ),
            "source_question_query": "context.strip_question_only",
            "retrieval_method": "exact_cosine",
        },
        "inputs": calibration_source,
        "artifacts": {
            "dual_key_tensor": {
                "path": tensor_path.name,
                "sha256": file_sha256(tensor_path),
            },
            "dual_key_manifest": {
                "path": manifest_path.name,
                "sha256": file_sha256(manifest_path),
                "logical_sha256": dual_loader.manifest_sha256,
            },
            "applicability_calibration": {
                "path": str(applicability_path.resolve()),
                "sha256": file_sha256(applicability_path),
                "logical_sha256": applicability_artifact["artifact_sha256"],
            },
        },
        "alignment": alignment,
        "applicability_reproduction_audit": reproduction,
        "applicability_calibration": applicability_artifact["calibration"],
        "applicability_metrics": applicability_artifact["metrics"],
        "requirements": requirements,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = args.output_dir / "offline_report.json"
    report_markdown_path = args.output_dir / "offline_report.md"
    write_json(report_path, report)
    write_text(report_markdown_path, _offline_markdown(report))
    print(
        f"[v3.5-offline] status={report['status']} records={len(records)} "
        f"report={report_path}",
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
