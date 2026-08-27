#!/usr/bin/env python3
"""Resumable vanilla-vs-V3 GSM8K evaluation with full online diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import GSM8K_VERIFIER_VERSION, diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.risk import ENTROPY_RISK_ARTIFACT_SCHEMA
from memgen.experience.v3 import (
    ExperienceMemoryV3Profile,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_selector import load_margin_selector_calibration
from memgen.experience.v3_artifacts import (
    authenticate_e0_inputs,
    load_formal_e0_report,
    load_v3_offline_report,
    validate_cross_bank_metadata,
)
from memgen.experience.v3_eval import summarize_v3_rows


V3_EVAL_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
V3_EVAL_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"
V3_EVAL_REPORT_SCHEMA = "experience-memory-v3-evaluation-report-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--logical-split",
        choices=("calibration-val", "dev-test", "final-test"),
        required=True,
    )
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--retrieval-key-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--v3-offline-report", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--risk-artifact", type=Path, required=True)
    parser.add_argument("--selector-calibration", type=Path)
    parser.add_argument(
        "--retrieval-embedding-transform",
        choices=(
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
        ),
        default=V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the slice remainder")
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=GSM8K_PROMPT_CONTRACT.max_new_tokens,
    )
    parser.add_argument("--save-query-embeddings", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def repository_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    revision = run("rev-parse", "HEAD") or None
    status = run("status", "--short")
    diff = run("diff", "--binary", "HEAD")
    implementation_paths = (
        "memgen/experience/v3.py",
        "memgen/experience/v3_selector.py",
        "memgen/experience/v3_artifacts.py",
        "memgen/experience/v3_eval.py",
        "memgen/model/retrieval_keys.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_runtime.py",
        "scripts/evaluate_v3_experience_memory.py",
    )
    implementation_hashes = {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in implementation_paths
    }
    return {
        "git_revision": revision,
        "worktree_dirty": bool(status),
        "git_status_sha256": text_sha256(status),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "implementation_files_sha256": implementation_hashes,
        "implementation_set_sha256": canonical_json_sha256(
            implementation_hashes
        ),
    }


def load_split_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError("Unexpected GSM8K split manifest schema")
    expected = value.get("manifest_sha256")
    actual = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    if expected != actual or value.get("overlap_check", {}).get("passed") is not True:
        raise ValueError("GSM8K split manifest hash or overlap audit failed")
    return value


def processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def score_condition(
    *,
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    ground_truth: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    completion_ids = [int(value) for value in completion_token_ids]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    verifier = diagnose_gsm8k_completion(completion, ground_truth)
    return {
        "completion": completion,
        "completion_token_ids": completion_ids,
        "completion_token_ids_sha256": canonical_json_sha256(completion_ids),
        "generated_token_count": len(completion_ids),
        "strict_correct": verifier["reward"] == 1.0,
        "format_correct": bool(verifier["format_valid"]),
        "strict_reward": float(verifier["reward"]),
        "scorer_version": GSM8K_VERIFIER_VERSION,
        "runtime_seconds": runtime_seconds,
    }


def online_diagnostics(result: Any) -> dict[str, Any]:
    outcomes = Counter(attempt.outcome for attempt in result.retrieval_attempts)
    attention = tuple(result.attention_traces)
    return {
        "retrieval_attempt_count": result.retrieval_attempt_count,
        "rearm_count": result.rearm_count,
        "activation_count": outcomes["activated"],
        "replacement_count": result.replacement_count,
        "duplicate_count": result.duplicate_count,
        "abstain_count": outcomes["abstained"],
        "memory_attention_step_count": len(attention),
        "attempt_budget_respected": (
            result.retrieval_attempt_count <= 3
        ),
        "query_context_is_full_prefix": all(
            attempt.retrieval_decision.query["query_token_count"]
            == attempt.retrieval_decision.query["prompt_token_count"]
            + attempt.retrieval_decision.query["partial_cot_token_count"]
            for attempt in result.retrieval_attempts
        ),
        "native_cache_excludes_memory_slots": all(
            trace.trace.native_key_length == trace.processed_prefix_token_count
            for trace in attention
        ),
        "memory_attention_mass_finite_and_positive": all(
            math.isfinite(float(trace.trace.memory_attention_mass))
            and float(trace.trace.memory_attention_mass) > 0.0
            for trace in attention
        ),
    }


def load_existing_rows(
    *,
    path: Path,
    profile_sha256: str,
    selected_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    for value in iter_jsonl(path):
        if (
            value.get("schema_version") != V3_EVAL_ROW_SCHEMA
            or value.get("profile_sha256") != profile_sha256
        ):
            raise ValueError("Existing V3 result row uses a different run profile")
        expected_row_hash = value.get("row_sha256")
        actual_row_hash = canonical_json_sha256({
            key: item
            for key, item in value.items()
            if key not in {"created_at", "row_sha256"}
        })
        if expected_row_hash != actual_row_hash:
            raise ValueError("Existing V3 result row hash mismatch")
        sample_id = str(value.get("sample_id", ""))
        if sample_id not in selected_ids or sample_id in completed:
            raise ValueError("Existing V3 results contain unknown or duplicate samples")
        completed.add(sample_id)
        rows.append(value)
    return rows, completed


def progress_report(
    *,
    status: str,
    profile_sha256: str,
    selected_count: int,
    rows: Sequence[Mapping[str, Any]],
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": V3_EVAL_REPORT_SCHEMA,
        "updated_at": utc_now(),
        "status": status,
        "profile_sha256": profile_sha256,
        "selected_sample_count": selected_count,
        "completed_sample_count": len(rows),
        "remaining_sample_count": selected_count - len(rows),
        "error": dict(error) if error is not None else None,
        "summary": summarize_v3_rows(rows) if rows else None,
    }
    value["report_sha256"] = canonical_json_sha256(value)
    return value


def evaluation_profile_sha256(value: Mapping[str, Any]) -> str:
    material = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "repository", "profile_sha256"}
    }
    repository = value.get("repository", {})
    material["code_identity"] = {
        "git_revision": repository.get("git_revision"),
        "tracked_diff_sha256": repository.get("tracked_diff_sha256"),
        "implementation_set_sha256": repository.get(
            "implementation_set_sha256"
        ),
    }
    return canonical_json_sha256(material)


def main() -> None:
    args = parse_args()
    if (
        args.offset < 0
        or args.limit < 0
        or args.parity_samples < 0
        or args.max_new_tokens <= 0
    ):
        raise ValueError("V3 evaluation received invalid limits")
    if args.max_new_tokens != GSM8K_PROMPT_CONTRACT.max_new_tokens:
        raise ValueError("V3 evaluation must use the canonical GSM8K token budget")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime, compare_token_sequences
    from memgen.model.retrieval_keys import (
        EmbeddingMemoryRetriever,
        FullPrefixQueryEncoder,
        RetrievalKeyBankLoader,
    )
    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader
    from memgen.model.v3_runtime import (
        EntropyHysteresisGate,
        OnlineExperienceMemorySystemV3,
    )

    split_manifest = load_split_manifest(args.split_manifest)
    selected = [
        item
        for item in split_manifest["samples"]
        if item.get("logical_split") == args.logical_split
    ][args.offset:]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("Selected V3 evaluation slice is empty")
    dataset_splits = {str(item.get("dataset_split", "")) for item in selected}
    if len(dataset_splits) != 1:
        raise ValueError("V3 logical split maps to multiple dataset splits")
    dataset_split = next(iter(dataset_splits))
    if args.logical_split == "final-test" and dataset_split != "test":
        raise ValueError("V3 final-test must be the official GSM8K test split")

    selector_calibration = None
    if args.selector_calibration is not None:
        selector_calibration = load_margin_selector_calibration(
            args.selector_calibration
        )
        expected_key_sha256 = selector_calibration.get("source", {}).get(
            "retrieval_key_manifest_sha256"
        )
        if expected_key_sha256 != file_sha256(args.retrieval_key_manifest):
            raise ValueError(
                "V3 selector calibration uses a different retrieval key bank"
            )
        calibration_transform = selector_calibration.get("source", {}).get(
            "retrieval_embedding_transform",
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
        )
        if calibration_transform != args.retrieval_embedding_transform:
            raise ValueError(
                "Selector calibration uses a different retrieval embedding transform"
            )
        profile = ExperienceMemoryV3Profile(
            retrieval_embedding_transform=args.retrieval_embedding_transform,
            retrieval_abstention_policy="top1_top2_margin",
            retrieval_min_top1_top2_margin=float(
                selector_calibration["calibration"][
                    "minimum_top1_top2_margin"
                ]
            ),
        )
    else:
        profile = ExperienceMemoryV3Profile(
            retrieval_embedding_transform=args.retrieval_embedding_transform
        )
    e0_report = load_formal_e0_report(args.e0_final_report)
    authenticate_e0_inputs(
        e0_report=e0_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
    )
    load_v3_offline_report(
        args.v3_offline_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
        retrieval_key_manifest_path=args.retrieval_key_manifest,
        e0_final_report_path=args.e0_final_report,
    )
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    key_manifest = json.loads(
        args.retrieval_key_manifest.read_text(encoding="utf-8")
    )
    alignment = validate_cross_bank_metadata(
        records=records,
        side_manifest=side_manifest,
        key_manifest=key_manifest,
    )
    reasoner = side_manifest["reasoner"]

    risk_artifact = torch.load(
        args.risk_artifact, map_location="cpu", weights_only=False
    )
    if risk_artifact.get("schema_version") != ENTROPY_RISK_ARTIFACT_SCHEMA:
        raise ValueError("V3 evaluation requires the canonical risk artifact")
    expected_prompt_contract = GSM8K_PROMPT_CONTRACT.metadata(
        chat_template=CONVERSATION_TEMPLATE
    )
    if risk_artifact.get("prompt_contract") != expected_prompt_contract:
        raise ValueError("V3 risk artifact uses a different prompt contract")
    heldout = risk_artifact.get("risk_gate", {}).get("heldout_diagnostic", {})
    if float(heldout.get("heldout_roc_auc", 0.0)) < float(
        heldout.get("minimum_heldout_roc_auc", 1.0)
    ):
        raise ValueError("V3 risk artifact failed its held-out diagnostic")
    for field_name in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field_name) != reasoner.get(
            field_name
        ):
            raise ValueError("V3 risk and memory reasoner provenance differs")
    gate = EntropyHysteresisGate.from_artifact(risk_artifact)

    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    resolved_model = str(
        getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]
    )
    resolved_tokenizer = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or reasoner["tokenizer_revision"]
    )
    if (
        resolved_model != reasoner["model_revision"]
        or resolved_tokenizer != reasoner["tokenizer_revision"]
    ):
        raise ValueError("Resolved V3 evaluation reasoner revision drifted")

    key_bank = RetrievalKeyBankLoader(
        manifest_path=args.retrieval_key_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    side_loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    side_entries = {
        str(entry["memory_id"]): entry for entry in side_manifest["records"]
    }
    retriever = EmbeddingMemoryRetriever(
        key_bank=key_bank,
        records=records,
        kv_valid_slot_counts={
            memory_id: int(entry["kv_valid_slot_count"])
            for memory_id, entry in side_entries.items()
        },
        profile=profile,
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=profile.layer_number,
        audit_canonical_rope=False,
        memory_score_normalization=profile.memory_score_normalization,
        memory_score_bias=profile.memory_score_bias,
    )
    baseline_runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    v3_runtime = OnlineExperienceMemorySystemV3(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        gate=gate,
        query_encoder=FullPrefixQueryEncoder(
            model=model, device=args.device, layer_number=profile.layer_number
        ),
        retriever=retriever,
        loader=side_loader,
        controller=controller,
        profile=profile,
    )

    selected_ids = [str(item["sample_id"]) for item in selected]
    interpretation = (
        "reused_official_test_descriptive_evaluation"
        if args.logical_split == "final-test"
        else (
            "answer_blind_centered_retrieval_validation"
            if profile.retrieval_embedding_transform
            == V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED
            else "answer_blind_margin_selector_validation"
            if selector_calibration is not None
            else "system_validation"
        )
    )
    run_profile = {
        "schema_version": V3_EVAL_PROFILE_SCHEMA,
        "created_at": utc_now(),
        "repository": repository_state(),
        "evaluation_interpretation": interpretation,
        "independent_final_confirmation": False,
        "logical_split": args.logical_split,
        "dataset_split": dataset_split,
        "dataset_revision": split_manifest["dataset"]["revision"],
        "selected_sample_count": len(selected),
        "selected_sample_ids_sha256": canonical_json_sha256(selected_ids),
        "slice": {"offset": args.offset, "limit": args.limit},
        "reasoner": reasoner | {"runtime_dtype": args.dtype},
        "prompt_contract": expected_prompt_contract,
        "system_profile": profile.to_dict(),
        "retrieval_embedding_space": retriever.embedding_space_audit,
        "selector_calibration": (
            {
                "schema_version": selector_calibration.get("schema_version"),
                "artifact_sha256": selector_calibration.get(
                    "artifact_sha256"
                ),
                "policy": selector_calibration.get("policy"),
                "source": selector_calibration.get("source"),
                "calibration": selector_calibration.get("calibration"),
                "task_accuracy_used": selector_calibration.get(
                    "task_accuracy_used"
                ),
                "answer_or_reward_used": selector_calibration.get(
                    "answer_or_reward_used"
                ),
                "first_attempt_selection_concentration": (
                    selector_calibration.get(
                        "first_attempt_selection_concentration"
                    )
                ),
            }
            if selector_calibration is not None
            else None
        ),
        "hysteresis_gate": gate.config.to_dict(),
        "hysteresis_threshold_provenance": {
            key: risk_artifact.get("construction", {}).get(key)
            for key in (
                "high_entropy_quantile",
                "high_entropy_threshold",
                "low_entropy_quantile",
                "low_entropy_threshold",
                "sink_token_count",
            )
        },
        "risk_diagnostic_qualification": {
            "heldout_roc_auc": heldout.get("heldout_roc_auc"),
            "minimum_heldout_roc_auc": heldout.get("minimum_heldout_roc_auc"),
            "online_control_role": "diagnostic_only",
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "vanilla": baseline_runtime.native_generation_config_dict,
            "v3": v3_runtime.decoding.config.to_dict() | {
                "use_cache": True,
                "batch_size": 1,
                "implementation": "explicit_live_native_kv_cache",
            },
            "parity_sample_count": min(args.parity_samples, len(selected)),
        },
        "logging": {
            "append_flush_fsync_per_sample": True,
            "resume_requires_profile_hash": True,
            "query_embeddings_sidecar": args.save_query_embeddings,
            "query_embedding_sidecar_representation": (
                "raw_unit_before_retrieval_embedding_transform"
            ),
            "full_logits_saved": False,
            "full_hidden_states_saved": False,
        },
        "metric_contract": {
            "strict_accuracy": "official_gsm8k_first_boxed_reward",
            "format_accuracy": "first_boxed_parseable",
            "generated_token_count": "through_first_eos_inclusive_else_full_budget",
            "diagnostic_answer_accuracy_aggregated": False,
        },
        "alignment": alignment,
        "inputs": {
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "retrieval_key_manifest_sha256": file_sha256(
                args.retrieval_key_manifest
            ),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "v3_offline_report_sha256": file_sha256(args.v3_offline_report),
            "e0_final_report_sha256": file_sha256(args.e0_final_report),
            "risk_artifact_sha256": file_sha256(args.risk_artifact),
            "selector_calibration_sha256": (
                file_sha256(args.selector_calibration)
                if args.selector_calibration is not None
                else None
            ),
        },
    }
    run_profile["profile_sha256"] = evaluation_profile_sha256(run_profile)
    profile_sha256 = run_profile["profile_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "run_profile.json"
    results_path = args.output_dir / "results.jsonl"
    report_path = args.output_dir / "run_report.json"
    if profile_path.exists():
        existing_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if (
            existing_profile.get("profile_sha256")
            != evaluation_profile_sha256(existing_profile)
            or existing_profile.get("profile_sha256") != profile_sha256
        ):
            raise ValueError("Cannot resume V3 evaluation with a different profile")
    else:
        if results_path.exists() and results_path.stat().st_size:
            raise ValueError("V3 results exist without an authenticated run profile")
        write_json_atomic(profile_path, run_profile)

    rows, completed_ids = load_existing_rows(
        path=results_path,
        profile_sha256=profile_sha256,
        selected_ids=set(selected_ids),
    )
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=dataset_split,
        revision=split_manifest["dataset"]["revision"],
    )
    write_json_atomic(report_path, progress_report(
        status="running",
        profile_sha256=profile_sha256,
        selected_count=len(selected),
        rows=rows,
    ))

    try:
        with results_path.open("a", encoding="utf-8") as output_handle:
            for position, entry in enumerate(selected):
                sample_id = str(entry["sample_id"])
                if sample_id in completed_ids:
                    continue
                source = dataset[int(entry["source_index"])]
                question = str(source["question"]).strip()
                answer = str(source["answer"]).strip()
                if (
                    text_sha256(question) != entry.get("question_sha256")
                    or text_sha256(answer) != entry.get("answer_sha256")
                ):
                    raise ValueError(f"Dataset hash drift for {sample_id}")
                prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, question)
                ground_truth = processed_solution(answer)

                sample_started = time.perf_counter()
                vanilla_started = time.perf_counter()
                vanilla_ids = baseline_runtime.generate_vanilla(prompt_ids)
                vanilla_seconds = time.perf_counter() - vanilla_started
                parity = None
                if position < args.parity_samples:
                    cache_ids = baseline_runtime.generate_cache_greedy(prompt_ids)
                    parity = compare_token_sequences(
                        vanilla_ids, cache_ids
                    ).to_dict()
                    if not parity["exact_match"]:
                        raise RuntimeError(
                            f"Vanilla/cache parity failed for {sample_id}"
                        )

                v3_started = time.perf_counter()
                v3_result = v3_runtime.generate(prompt_token_ids=prompt_ids)
                v3_seconds = time.perf_counter() - v3_started
                vanilla_condition = score_condition(
                    tokenizer=tokenizer,
                    completion_token_ids=vanilla_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=vanilla_seconds,
                )
                v3_condition = score_condition(
                    tokenizer=tokenizer,
                    completion_token_ids=v3_result.completion_token_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=v3_seconds,
                )
                diagnostics = online_diagnostics(v3_result)
                if not all(
                    diagnostics[key]
                    for key in (
                        "attempt_budget_respected",
                        "query_context_is_full_prefix",
                        "native_cache_excludes_memory_slots",
                        "memory_attention_mass_finite_and_positive",
                    )
                ):
                    raise RuntimeError(f"V3 integrity check failed for {sample_id}")

                query_sidecar = None
                if args.save_query_embeddings and v3_result.query_embeddings:
                    from safetensors.torch import save_file

                    safe_sample_id = "".join(
                        character if character.isalnum() or character in "-_" else "_"
                        for character in sample_id
                    )
                    sidecar_path = (
                        args.output_dir
                        / "query_embeddings"
                        / f"{safe_sample_id}.safetensors"
                    )
                    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(
                        {
                            f"attempt_{index:02d}": embedding.contiguous()
                            for index, embedding in enumerate(
                                v3_result.query_embeddings, start=1
                            )
                        },
                        str(sidecar_path),
                        metadata={
                            "schema_version": (
                                "experience-memory-v3-query-embeddings-v1"
                            ),
                            "sample_id": sample_id,
                            "representation": (
                                "raw_unit_before_retrieval_embedding_transform"
                            ),
                        },
                    )
                    query_sidecar = {
                        "path": str(sidecar_path.relative_to(args.output_dir)),
                        "sha256": file_sha256(sidecar_path),
                        "attempt_count": len(v3_result.query_embeddings),
                    }

                row = {
                    "schema_version": V3_EVAL_ROW_SCHEMA,
                    "created_at": utc_now(),
                    "profile_sha256": profile_sha256,
                    "sample_id": sample_id,
                    "logical_split": args.logical_split,
                    "dataset_split": dataset_split,
                    "source_index": int(entry["source_index"]),
                    "question_sha256": text_sha256(question),
                    "answer_sha256": text_sha256(answer),
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
                    "cache_parity": parity,
                    "conditions": {
                        "vanilla": vanilla_condition,
                        "v3": v3_condition | {
                            "online_diagnostics": diagnostics,
                            "runtime_trace": v3_result.to_dict(),
                            "query_embedding_sidecar": query_sidecar,
                        },
                    },
                    "paired_generated_token_delta_v3_minus_vanilla": (
                        v3_condition["generated_token_count"]
                        - vanilla_condition["generated_token_count"]
                    ),
                    "sample_runtime_seconds": time.perf_counter() - sample_started,
                }
                row["row_sha256"] = canonical_json_sha256({
                    key: value for key, value in row.items() if key != "created_at"
                })
                output_handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                output_handle.flush()
                os.fsync(output_handle.fileno())
                rows.append(row)
                completed_ids.add(sample_id)
                write_json_atomic(report_path, progress_report(
                    status="running",
                    profile_sha256=profile_sha256,
                    selected_count=len(selected),
                    rows=rows,
                ))
                print(
                    f"[v3-eval] {len(rows)}/{len(selected)} sample={sample_id} "
                    f"strict={int(v3_condition['strict_correct'])} "
                    f"attempts={diagnostics['retrieval_attempt_count']}",
                    flush=True,
                )
    except Exception as error:
        write_json_atomic(report_path, progress_report(
            status="failed_resumable",
            profile_sha256=profile_sha256,
            selected_count=len(selected),
            rows=rows,
            error={"type": type(error).__name__, "message": str(error)},
        ))
        raise
    finally:
        controller.close()

    if len(rows) != len(selected):
        raise RuntimeError("V3 evaluation ended before every selected sample completed")
    final_report = progress_report(
        status="completed",
        profile_sha256=profile_sha256,
        selected_count=len(selected),
        rows=rows,
    )
    final_report["evaluation_interpretation"] = interpretation
    final_report["independent_final_confirmation"] = False
    final_report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in final_report.items() if key != "report_sha256"
    })
    write_json_atomic(report_path, final_report)
    print(
        f"[v3-eval] completed samples={len(rows)} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
