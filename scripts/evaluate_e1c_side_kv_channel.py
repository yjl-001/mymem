#!/usr/bin/env python3
"""Evaluate prompt-end side-KV against same-path split-prefill controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1_staged import (
    E1B_MANIFEST_SCHEMA,
    E1B_RESULTS_SCHEMA,
    E1B_SUMMARY_SCHEMA,
    E1C_MEMORY_SCORE_NORMALIZATION,
    E1C_RESULTS_SCHEMA,
    E1C_SUMMARY_SCHEMA,
    E1BRetrievalAssignment,
    render_single_experience,
)
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from scripts.e1_staged_common import (
    PairedConditionComparison,
    PairedConditionDiagnostics,
    effect_is_positive,
    format_transfer_diagnostic,
    load_hashed_manifest,
    paired_condition_effect,
    processed_solution,
    prompt_token_ids,
    score_completion,
    summarize_conditions,
    token_sequence_diagnostic,
    utc_now,
    validate_resolved_revisions,
    write_json,
)


SPLIT_PREFILL_PATH = "split-before-final-prompt-token-v1"
E1B_FULL_PREFILL_PATH = "e1b-full-prompt-prefill-reference-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--e1b-results", type=Path, required=True)
    parser.add_argument("--e1b-run-report", type=Path, required=True)
    parser.add_argument("--e1b-summary", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(value) for value in iter_jsonl(path)]


def compact_trace_artifact(
    traces: Sequence[Any], *, completion_length: int, prompt_length: int
) -> dict[str, Any]:
    native_lengths = [int(trace.native_key_length) for trace in traces]
    masses = [float(trace.memory_attention_mass) for trace in traces]
    expected_lengths = list(range(prompt_length, prompt_length + completion_length))
    memory_ids = {str(trace.memory_id) for trace in traces}
    slot_counts = {int(trace.memory_slot_count) for trace in traces}
    normalizations = {str(trace.memory_score_normalization) for trace in traces}
    return {
        "trace_count": len(traces),
        "expected_trace_count": completion_length,
        "memory_ids": sorted(memory_ids),
        "memory_slot_counts": sorted(slot_counts),
        "memory_score_normalizations": sorted(normalizations),
        "native_key_lengths": native_lengths,
        "native_key_lengths_sha256": canonical_json_sha256(native_lengths),
        "memory_attention_masses": masses,
        "memory_attention_masses_sha256": canonical_json_sha256(masses),
        "mean_memory_attention_mass": sum(masses) / len(masses),
        "min_memory_attention_mass": min(masses),
        "max_memory_attention_mass": max(masses),
        "one_trace_per_generated_token": len(traces) == completion_length,
        "native_cache_length_matches_real_tokens": native_lengths == expected_lengths,
        "all_memory_attention_mass_finite_and_positive": all(
            math.isfinite(value) and value > 0.0 for value in masses
        ),
        "memory_id_constant": len(memory_ids) == 1,
        "memory_slot_count_constant": len(slot_counts) == 1,
        "normalization_constant": normalizations == {E1C_MEMORY_SCORE_NORMALIZATION},
    }


def _first_step_diagnostic(value: Any) -> dict[str, Any]:
    if value is None:
        raise RuntimeError("Audited split prefill did not return first-step metrics")
    return asdict(value)


def _reference_condition(
    previous: Mapping[str, Any], condition: str
) -> dict[str, Any]:
    row = dict(previous["conditions"][condition])
    row["prefill_path"] = E1B_FULL_PREFILL_PATH
    return row


def _summarize_prefill_path_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for name in ("no_memory", "matched_text", "shuffled_text"):
        rows = [record["prefill_path_diagnostics"][name] for record in records]
        token_rows = [row["full_e1b_vs_split_tokens"] for row in rows]
        logits_rows = [row["full_vs_split_first_step_logits"] for row in rows]
        by_condition[name] = {
            "sample_count": len(rows),
            "complete_trajectory_match_count": sum(
                row["exact_match"] for row in token_rows
            ),
            "first_token_match_count": sum(
                row["first_token_match"] for row in token_rows
            ),
            "mean_common_prefix_token_count": sum(
                int(row["common_prefix_token_count"]) for row in token_rows
            ) / len(rows),
            "mean_first_step_kl_full_to_split": sum(
                float(row["kl_full_to_split"]) for row in logits_rows
            ) / len(rows),
            "max_first_step_kl_full_to_split": max(
                float(row["kl_full_to_split"]) for row in logits_rows
            ),
            "mean_first_step_max_absolute_error": sum(
                float(row["max_absolute_error"]) for row in logits_rows
            ) / len(rows),
            "max_first_step_absolute_error": max(
                float(row["max_absolute_error"]) for row in logits_rows
            ),
            "first_step_top1_changed_count": sum(
                row["top1_changed"] for row in logits_rows
            ),
        }
    repeat_rows = [
        record["prefill_path_diagnostics"]["split_no_memory_repeat"]
        for record in records
    ]
    return {
        "interpretation": (
            "full-vs-split is a numerical diagnostic, not a mechanism gate; "
            "all primary E1-C contrasts use the split-prefill path"
        ),
        "full_e1b_vs_split": by_condition,
        "split_no_memory_repeat_exact_match_count": sum(
            row["exact_match"] for row in repeat_rows
        ),
        "matched_side_kv_baseline_first_token_match_count": sum(
            record["prefill_path_diagnostics"][
                "matched_side_kv_baseline_first_token_matches_split_no_memory"
            ]
            for record in records
        ),
        "shuffled_side_kv_baseline_first_token_match_count": sum(
            record["prefill_path_diagnostics"][
                "shuffled_side_kv_baseline_first_token_matches_split_no_memory"
            ]
            for record in records
        ),
    }


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime
    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader

    manifest = load_hashed_manifest(
        args.assignment_manifest, schema=E1B_MANIFEST_SCHEMA
    )
    assignments = tuple(
        E1BRetrievalAssignment.from_dict(value) for value in manifest["assignments"]
    )
    if any(not assignment.assigned for assignment in assignments):
        raise ValueError("E1-C requires complete matched/shuffled E1-B assignments")
    e1b_run = json.loads(args.e1b_run_report.read_text(encoding="utf-8"))
    if (
        e1b_run.get("status") != "completed"
        or e1b_run.get("results", {}).get("sha256")
        != file_sha256(args.e1b_results)
        or e1b_run.get("inputs", {}).get("assignment_manifest_sha256")
        != file_sha256(args.assignment_manifest)
    ):
        raise ValueError("E1-B artifacts do not match their run report")
    e1b_summary = json.loads(args.e1b_summary.read_text(encoding="utf-8"))
    if e1b_summary.get("schema_version") != E1B_SUMMARY_SCHEMA:
        raise ValueError("E1-C requires the current E1-B summary schema")
    component_handoff = e1b_summary.get("component_diagnostic", {})
    if component_handoff.get("e1c_component_diagnostic_allowed") is not True:
        raise ValueError("E1-B artifacts are not valid for E1-C component diagnosis")
    source_e1b_formal_passed = e1b_summary.get("formal_e1b_passed") is True
    if (
        e1b_summary.get("provenance", {}).get(
            "assignment_manifest_logical_sha256"
        )
        != manifest["manifest_sha256"]
        or e1b_summary.get("provenance", {}).get("results_sha256")
        != file_sha256(args.e1b_results)
    ):
        raise ValueError("E1-B summary does not authenticate these artifacts")
    e1b_records = read_jsonl(args.e1b_results)
    if any(record.get("schema_version") != E1B_RESULTS_SCHEMA for record in e1b_records):
        raise ValueError("Unexpected E1-B results schema")
    e1b_by_sample = {str(record["sample_id"]): record for record in e1b_records}
    if set(e1b_by_sample) != {item.sample_id for item in assignments}:
        raise ValueError("E1-B results and assignment samples differ")
    for assignment in assignments:
        previous = e1b_by_sample[assignment.sample_id]
        if (
            previous.get("assignment_manifest_sha256")
            != manifest["manifest_sha256"]
            or previous.get("preanswer_used_in_second_prompt") is not False
            or previous.get("matched_memory", {}).get("memory_id")
            != assignment.matched_memory.memory_id
            or previous.get("shuffled_memory", {}).get("memory_id")
            != assignment.shuffled_memory.memory_id
            or previous.get("conditions", {}).get("no_memory", {}).get(
                "completion_token_ids_sha256"
            )
            != assignment.preanswer_completion_token_ids_sha256
        ):
            raise ValueError("E1-B result pairing/provenance is inconsistent")

    inputs = manifest["inputs"]
    if file_sha256(args.memory_records) != inputs["memory_records_sha256"]:
        raise ValueError("MemoryRecords differ from frozen E1-B assignment")
    if file_sha256(args.side_kv_manifest) != inputs["side_kv_manifest_sha256"]:
        raise ValueError("Side-KV bank differs from frozen E1-B assignment")
    if file_sha256(args.split_manifest) != inputs["split_manifest_sha256"]:
        raise ValueError("Split manifest differs from frozen E1-B assignment")
    memory_records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    memory_by_id = {record.memory_id: record for record in memory_records}
    used_memory_ids = {
        choice.memory_id
        for assignment in assignments
        for choice in (assignment.matched_memory, assignment.shuffled_memory)
        if choice is not None
    }
    if not used_memory_ids <= set(memory_by_id):
        raise ValueError("E1-C assignments reference unknown MemoryRecords")

    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    reasoner = manifest["reasoner"]
    layer = int(reasoner["side_kv_layer"])
    if layer != int(side_manifest["layer_number"]):
        raise ValueError("E1-B and side-KV layer provenance differ")
    if args.dtype != reasoner["dtype"]:
        raise ValueError("E1-C dtype differs from E1-B")

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
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    validate_resolved_revisions(
        model=model, tokenizer=tokenizer, reasoner=reasoner, label="E1-C"
    )
    loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=layer,
        audit_canonical_rope=False,
        memory_score_normalization=E1C_MEMORY_SCORE_NORMALIZATION,
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=int(manifest["configuration"]["max_new_tokens"]),
    )
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=inputs["dataset_revision"],
    )

    reference_conditions = (
        "e1b_full_no_memory",
        "e1b_full_matched_text",
        "e1b_full_shuffled_text",
    )
    primary_conditions = (
        "split_no_memory",
        "split_matched_text",
        "split_shuffled_text",
        "matched_persistent_side_kv",
        "shuffled_persistent_side_kv",
    )
    conditions = reference_conditions + primary_conditions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    output_records: list[dict[str, Any]] = []
    try:
        with results_path.open("w", encoding="utf-8") as handle:
            for position, assignment in enumerate(assignments, start=1):
                source = dataset[assignment.source_index]
                question = str(source["question"]).strip()
                if text_sha256(question) != assignment.question_sha256:
                    raise ValueError(
                        f"Question hash mismatch for {assignment.sample_id}"
                    )
                ground_truth = processed_solution(str(source["answer"]).strip())
                base_prompt_ids = prompt_token_ids(
                    tokenizer, question=question, memory_text=None
                )
                if (
                    len(base_prompt_ids) != assignment.base_prompt_token_count
                    or canonical_json_sha256(base_prompt_ids)
                    != assignment.base_prompt_token_ids_sha256
                ):
                    raise ValueError(f"Base prompt drift for {assignment.sample_id}")
                previous = e1b_by_sample[assignment.sample_id]
                condition_rows = {
                    "e1b_full_no_memory": _reference_condition(
                        previous, "no_memory"
                    ),
                    "e1b_full_matched_text": _reference_condition(
                        previous, "matched_text"
                    ),
                    "e1b_full_shuffled_text": _reference_condition(
                        previous, "shuffled_text"
                    ),
                }

                started = time.perf_counter()
                split_no_memory = runtime.generate_prompt_split(
                    base_prompt_ids, audit_full_prefill=True
                )
                split_no_memory_seconds = time.perf_counter() - started
                split_no_memory_repeat = runtime.generate_prompt_split(
                    base_prompt_ids, audit_full_prefill=False
                )
                condition_rows["split_no_memory"] = score_completion(
                    tokenizer=tokenizer,
                    completion_token_ids=split_no_memory.completion_token_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=split_no_memory_seconds,
                    prompt_token_count=len(base_prompt_ids),
                )
                condition_rows["split_no_memory"]["prefill_path"] = (
                    SPLIT_PREFILL_PATH
                )
                prefill_diagnostics: dict[str, Any] = {
                    "no_memory": {
                        "full_e1b_vs_split_tokens": token_sequence_diagnostic(
                            assignment.preanswer_completion_token_ids,
                            split_no_memory.completion_token_ids,
                        ),
                        "full_vs_split_first_step_logits": _first_step_diagnostic(
                            split_no_memory.full_prefill_diagnostic
                        ),
                    },
                    "split_no_memory_repeat": token_sequence_diagnostic(
                        split_no_memory.completion_token_ids,
                        split_no_memory_repeat.completion_token_ids,
                    ),
                }

                for label, choice in (
                    ("matched", assignment.matched_memory),
                    ("shuffled", assignment.shuffled_memory),
                ):
                    assert choice is not None
                    memory_record = memory_by_id[choice.memory_id]
                    if (
                        memory_record.payload_hash != choice.payload_hash
                        or memory_record.token_count != choice.token_count
                    ):
                        raise ValueError(
                            f"Text memory metadata drift for {choice.memory_id}"
                        )
                    memory_text = render_single_experience(memory_record)
                    treatment_prompt_ids = prompt_token_ids(
                        tokenizer, question=question, memory_text=memory_text
                    )
                    previous_text = previous["conditions"][f"{label}_text"]
                    if (
                        previous_text.get("prompt_token_ids_sha256")
                        != canonical_json_sha256(treatment_prompt_ids)
                    ):
                        raise ValueError(
                            f"E1-B {label} text prompt drift for {assignment.sample_id}"
                        )
                    started = time.perf_counter()
                    split_text = runtime.generate_prompt_split(
                        treatment_prompt_ids, audit_full_prefill=True
                    )
                    elapsed = time.perf_counter() - started
                    condition_name = f"split_{label}_text"
                    condition_rows[condition_name] = score_completion(
                        tokenizer=tokenizer,
                        completion_token_ids=split_text.completion_token_ids,
                        ground_truth=ground_truth,
                        runtime_seconds=elapsed,
                        prompt_token_count=len(treatment_prompt_ids),
                        memory_ids=(choice.memory_id,),
                    )
                    condition_rows[condition_name]["prompt_token_ids_sha256"] = (
                        canonical_json_sha256(treatment_prompt_ids)
                    )
                    condition_rows[condition_name]["prefill_path"] = (
                        SPLIT_PREFILL_PATH
                    )
                    prefill_diagnostics[f"{label}_text"] = {
                        "full_e1b_vs_split_tokens": token_sequence_diagnostic(
                            previous_text["completion_token_ids"],
                            split_text.completion_token_ids,
                        ),
                        "full_vs_split_first_step_logits": _first_step_diagnostic(
                            split_text.full_prefill_diagnostic
                        ),
                    }

                for condition, choice in (
                    ("matched_persistent_side_kv", assignment.matched_memory),
                    ("shuffled_persistent_side_kv", assignment.shuffled_memory),
                ):
                    assert choice is not None
                    memory = loader.get(
                        choice.memory_id,
                        device=args.device,
                        dtype=next(model.parameters()).dtype,
                    )
                    if (
                        memory.payload_hash != choice.payload_hash
                        or memory.valid_slot_count != choice.kv_valid_slot_count
                        or memory.layer_number != layer
                    ):
                        raise ValueError(
                            f"Side-KV metadata drift for {choice.memory_id}"
                        )
                    started = time.perf_counter()
                    generated = runtime.generate_prompt_with_persistent_memory(
                        prompt_token_ids=base_prompt_ids,
                        memory=memory,
                        controller=controller,
                    )
                    elapsed = time.perf_counter() - started
                    baseline_first_token_matches = bool(
                        split_no_memory.completion_token_ids
                        and generated.baseline_first_token_id
                        == split_no_memory.completion_token_ids[0]
                    )
                    trace = compact_trace_artifact(
                        generated.attention_traces,
                        completion_length=len(generated.completion_token_ids),
                        prompt_length=len(base_prompt_ids),
                    )
                    trace.update({
                        "first_step_logits_kl_baseline_to_memory": (
                            generated.first_step_logits_kl
                        ),
                        "first_step_top1_changed": generated.first_step_top1_changed,
                        "baseline_first_token_id": generated.baseline_first_token_id,
                        "baseline_first_token_matches_split_no_memory": (
                            baseline_first_token_matches
                        ),
                    })
                    condition_rows[condition] = score_completion(
                        tokenizer=tokenizer,
                        completion_token_ids=generated.completion_token_ids,
                        ground_truth=ground_truth,
                        runtime_seconds=elapsed,
                        prompt_token_count=len(base_prompt_ids),
                        memory_ids=(choice.memory_id,),
                        side_kv=trace,
                    )
                    condition_rows[condition]["prefill_path"] = SPLIT_PREFILL_PATH
                    label = condition.split("_", 1)[0]
                    prefill_diagnostics[
                        f"{label}_side_kv_baseline_first_token_matches_split_no_memory"
                    ] = baseline_first_token_matches

                record = {
                    "schema_version": E1C_RESULTS_SCHEMA,
                    "sample_id": assignment.sample_id,
                    "logical_split": assignment.logical_split,
                    "question_sha256": assignment.question_sha256,
                    "assignment_manifest_sha256": manifest["manifest_sha256"],
                    "e1b_result_sha256": canonical_json_sha256(previous),
                    "matched_memory": assignment.matched_memory.to_dict(),
                    "shuffled_memory": assignment.shuffled_memory.to_dict(),
                    "primary_prefill_path": SPLIT_PREFILL_PATH,
                    "prefill_path_diagnostics": prefill_diagnostics,
                    "conditions": condition_rows,
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
                output_records.append(record)
                if position % 10 == 0 or position == len(assignments):
                    print(f"[e1c-eval] {position}/{len(assignments)}", flush=True)
    finally:
        controller.close()

    condition_summary = summarize_conditions(output_records, conditions)
    diagnostic_builder = PairedConditionDiagnostics(
        output_records, bootstrap_resamples=args.bootstrap_resamples
    )
    paired_diagnostics = diagnostic_builder.summarize(
        (
            PairedConditionComparison(
                "matched_side_kv_vs_split_no_memory",
                "matched_persistent_side_kv",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "shuffled_side_kv_vs_split_no_memory",
                "shuffled_persistent_side_kv",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "matched_side_kv_vs_shuffled_side_kv",
                "matched_persistent_side_kv",
                "shuffled_persistent_side_kv",
            ),
            PairedConditionComparison(
                "matched_side_kv_vs_split_matched_text",
                "matched_persistent_side_kv",
                "split_matched_text",
            ),
            PairedConditionComparison(
                "split_matched_text_vs_split_no_memory",
                "split_matched_text",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "split_shuffled_text_vs_split_no_memory",
                "split_shuffled_text",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "split_matched_text_vs_split_shuffled_text",
                "split_matched_text",
                "split_shuffled_text",
            ),
        )
    )
    accuracy_effects = paired_diagnostics["accuracy_effects"]
    format_effects = paired_diagnostics["format_effects"]
    format_transfer = format_transfer_diagnostic(
        text_effect=format_effects["split_matched_text_vs_split_no_memory"],
        side_kv_effect=format_effects["matched_side_kv_vs_split_no_memory"],
    )
    exact_slot_sample_ids = {
        assignment.sample_id
        for assignment in assignments
        if assignment.shuffled_memory is not None
        and assignment.matched_memory.kv_valid_slot_count
        == assignment.shuffled_memory.kv_valid_slot_count
    }
    exact_slot_records = [
        record
        for record in output_records
        if record["sample_id"] in exact_slot_sample_ids
    ]
    exact_slot_sensitivity = {
        "paired_sample_count": len(exact_slot_records),
        "matched_vs_shuffled_accuracy": (
            paired_condition_effect(
                exact_slot_records,
                treatment="matched_persistent_side_kv",
                control="shuffled_persistent_side_kv",
                field="final_reward",
                resamples=args.bootstrap_resamples,
            )
            if len(exact_slot_records) >= 2
            else None
        ),
        "matched_vs_shuffled_format": (
            paired_condition_effect(
                exact_slot_records,
                treatment="matched_persistent_side_kv",
                control="shuffled_persistent_side_kv",
                field="format_valid",
                resamples=args.bootstrap_resamples,
            )
            if len(exact_slot_records) >= 2
            else None
        ),
    }
    side_rows = [
        record["conditions"][condition]["side_kv"]
        for record in output_records
        for condition in (
            "matched_persistent_side_kv",
            "shuffled_persistent_side_kv",
        )
    ]
    side_trace_invariants_passed = all(
        row[requirement]
        for row in side_rows
        for requirement in (
            "one_trace_per_generated_token",
            "native_cache_length_matches_real_tokens",
            "all_memory_attention_mass_finite_and_positive",
            "memory_id_constant",
            "memory_slot_count_constant",
            "normalization_constant",
            "baseline_first_token_matches_split_no_memory",
        )
    )
    split_repeat_deterministic = all(
        record["prefill_path_diagnostics"]["split_no_memory_repeat"]["exact_match"]
        for record in output_records
    )
    primary_path_integrity = all(
        record["primary_prefill_path"] == SPLIT_PREFILL_PATH
        and all(
            record["conditions"][condition]["prefill_path"] == SPLIT_PREFILL_PATH
            for condition in primary_conditions
        )
        for record in output_records
    )
    mechanism_passed = (
        side_trace_invariants_passed
        and split_repeat_deterministic
        and primary_path_integrity
    )
    acceptance = {
        "persistent_side_kv_mechanism_integrity": mechanism_passed,
        "matched_side_kv_accuracy_above_split_no_memory": effect_is_positive(
            accuracy_effects["matched_side_kv_vs_split_no_memory"]
        ),
        "matched_side_kv_accuracy_above_shuffled": effect_is_positive(
            accuracy_effects["matched_side_kv_vs_shuffled_side_kv"]
        ),
        "matched_side_kv_format_not_below_split_no_memory": (
            float(
                format_effects["matched_side_kv_vs_split_no_memory"][
                    "mean_treatment_minus_control"
                ]
            )
            >= 0.0
        ),
    }
    prefill_summary = _summarize_prefill_path_diagnostics(output_records)
    summary = {
        "schema_version": E1C_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "status": "passed" if all(acceptance.values()) else "did_not_pass",
        "formal_e1c_passed": all(acceptance.values()),
        "source_e1b_formal_passed": source_e1b_formal_passed,
        "sample_count": len(output_records),
        "condition_roles": {
            "cross_stage_references": list(reference_conditions),
            "same_path_primary_conditions": list(primary_conditions),
        },
        "conditions": condition_summary,
        "accuracy_effects": accuracy_effects,
        "diagnostic_answer_effects": paired_diagnostics[
            "diagnostic_answer_effects"
        ],
        "format_effects": format_effects,
        "format_positive_control_transfer": format_transfer,
        "strict_accuracy_transition_diagnostics": (
            paired_diagnostics["strict_accuracy_transition_diagnostics"]
        ),
        "completion_difference_diagnostics": paired_diagnostics[
            "completion_difference_diagnostics"
        ],
        "exact_slot_count_sensitivity": exact_slot_sensitivity,
        "component_diagnostic": {
            "status": "passed" if mechanism_passed else "failed",
            "persistent_side_kv_mechanism_passed": mechanism_passed,
            "formal_task_pass_required_for_diagnostic": False,
        },
        "prefill_path_diagnostics": prefill_summary,
        "mechanism_diagnostics": {
            "normalization": E1C_MEMORY_SCORE_NORMALIZATION,
            "primary_prefill_path": SPLIT_PREFILL_PATH,
            "all_runtime_invariants_passed": mechanism_passed,
            "side_trace_invariants_passed": side_trace_invariants_passed,
            "split_no_memory_repeat_deterministic": split_repeat_deterministic,
            "primary_path_integrity": primary_path_integrity,
            "mean_matched_memory_attention_mass": sum(
                record["conditions"]["matched_persistent_side_kv"]["side_kv"][
                    "mean_memory_attention_mass"
                ]
                for record in output_records
            ) / len(output_records),
            "mean_shuffled_memory_attention_mass": sum(
                record["conditions"]["shuffled_persistent_side_kv"]["side_kv"][
                    "mean_memory_attention_mass"
                ]
                for record in output_records
            ) / len(output_records),
            "mean_matched_first_step_logits_kl": sum(
                record["conditions"]["matched_persistent_side_kv"]["side_kv"][
                    "first_step_logits_kl_baseline_to_memory"
                ]
                for record in output_records
            ) / len(output_records),
            "mean_shuffled_first_step_logits_kl": sum(
                record["conditions"]["shuffled_persistent_side_kv"]["side_kv"][
                    "first_step_logits_kl_baseline_to_memory"
                ]
                for record in output_records
            ) / len(output_records),
            "matched_first_step_top1_changed_count": sum(
                record["conditions"]["matched_persistent_side_kv"]["side_kv"][
                    "first_step_top1_changed"
                ]
                for record in output_records
            ),
            "shuffled_first_step_top1_changed_count": sum(
                record["conditions"]["shuffled_persistent_side_kv"]["side_kv"][
                    "first_step_top1_changed"
                ]
                for record in output_records
            ),
        },
        "acceptance": acceptance,
    }
    write_json(args.output_dir / "e1c_summary.json", summary)
    write_json(args.output_dir / "run_report.json", {
        "schema_version": "experience-memory-e1c-run-report-v2",
        "created_at": utc_now(),
        "status": "completed",
        "sample_count": len(output_records),
        "inputs": {
            "assignment_manifest_sha256": file_sha256(args.assignment_manifest),
            "e1b_results_sha256": file_sha256(args.e1b_results),
            "e1b_run_report_sha256": file_sha256(args.e1b_run_report),
            "e1b_summary_sha256": file_sha256(args.e1b_summary),
            "memory_records_sha256": file_sha256(args.memory_records),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "split_manifest_sha256": file_sha256(args.split_manifest),
        },
        "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
    })
    print(
        f"[e1c-eval] status={summary['status']} output={args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()
