#!/usr/bin/env python3
"""Evaluate prompt-end persistent side-KV using the frozen E1-B assignments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence


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
)
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from scripts.e1_staged_common import (
    effect_is_positive,
    load_hashed_manifest,
    paired_condition_effect,
    processed_solution,
    prompt_token_ids,
    score_completion,
    summarize_conditions,
    utc_now,
    validate_resolved_revisions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--e1b-results", type=Path, required=True)
    parser.add_argument("--e1b-run-report", type=Path, required=True)
    parser.add_argument("--e1b-summary", type=Path, required=True)
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
    if e1b_run.get("results", {}).get("sha256") != file_sha256(args.e1b_results):
        raise ValueError("E1-B results do not match their run report")
    e1b_summary = json.loads(args.e1b_summary.read_text(encoding="utf-8"))
    if (
        e1b_summary.get("schema_version") != E1B_SUMMARY_SCHEMA
        or e1b_summary.get("formal_e1b_passed") is not True
    ):
        raise ValueError("E1-C is only permitted after formal E1-B passes")
    if (
        e1b_summary.get("provenance", {}).get(
            "assignment_manifest_logical_sha256"
        )
        != manifest["manifest_sha256"]
        or e1b_summary.get("provenance", {}).get("results_sha256")
        != file_sha256(args.e1b_results)
    ):
        raise ValueError("E1-B passing summary does not authenticate these artifacts")
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
        ):
            raise ValueError("E1-B result pairing/provenance is inconsistent")

    inputs = manifest["inputs"]
    if file_sha256(args.side_kv_manifest) != inputs["side_kv_manifest_sha256"]:
        raise ValueError("Side-KV bank differs from frozen E1-B assignment")
    if file_sha256(args.split_manifest) != inputs["split_manifest_sha256"]:
        raise ValueError("Split manifest differs from frozen E1-B assignment")
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

    conditions = (
        "no_memory",
        "matched_text",
        "shuffled_text",
        "matched_persistent_side_kv",
        "shuffled_persistent_side_kv",
    )
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
                if canonical_json_sha256(base_prompt_ids) != (
                    assignment.base_prompt_token_ids_sha256
                ):
                    raise ValueError(f"Base prompt drift for {assignment.sample_id}")
                previous = e1b_by_sample[assignment.sample_id]
                split_vanilla_ids = runtime.generate_prompt_split_vanilla(
                    base_prompt_ids
                )
                prompt_split_no_memory_parity = split_vanilla_ids == (
                    assignment.preanswer_completion_token_ids
                )
                condition_rows = {
                    name: previous["conditions"][name]
                    for name in ("no_memory", "matched_text", "shuffled_text")
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
                    started = time.perf_counter()
                    generated = runtime.generate_prompt_with_persistent_memory(
                        prompt_token_ids=base_prompt_ids,
                        memory=memory,
                        controller=controller,
                    )
                    elapsed = time.perf_counter() - started
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
                record = {
                    "schema_version": E1C_RESULTS_SCHEMA,
                    "sample_id": assignment.sample_id,
                    "logical_split": assignment.logical_split,
                    "question_sha256": assignment.question_sha256,
                    "assignment_manifest_sha256": manifest["manifest_sha256"],
                    "e1b_result_sha256": canonical_json_sha256(previous),
                    "prompt_split_no_memory_parity": (
                        prompt_split_no_memory_parity
                    ),
                    "matched_memory": assignment.matched_memory.to_dict(),
                    "shuffled_memory": assignment.shuffled_memory.to_dict(),
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
    accuracy_effects = {
        "matched_side_kv_vs_no_memory": paired_condition_effect(
            output_records,
            treatment="matched_persistent_side_kv",
            control="no_memory",
            field="final_reward",
            resamples=args.bootstrap_resamples,
        ),
        "matched_side_kv_vs_shuffled_side_kv": paired_condition_effect(
            output_records,
            treatment="matched_persistent_side_kv",
            control="shuffled_persistent_side_kv",
            field="final_reward",
            resamples=args.bootstrap_resamples,
        ),
        "matched_side_kv_vs_matched_text": paired_condition_effect(
            output_records,
            treatment="matched_persistent_side_kv",
            control="matched_text",
            field="final_reward",
            resamples=args.bootstrap_resamples,
        ),
    }
    format_effect = paired_condition_effect(
        output_records,
        treatment="matched_persistent_side_kv",
        control="no_memory",
        field="format_valid",
        resamples=args.bootstrap_resamples,
    )
    side_rows = [
        record["conditions"][condition]["side_kv"]
        for record in output_records
        for condition in (
            "matched_persistent_side_kv",
            "shuffled_persistent_side_kv",
        )
    ]
    mechanism_passed = all(
        row[requirement]
        for row in side_rows
        for requirement in (
            "one_trace_per_generated_token",
            "native_cache_length_matches_real_tokens",
            "all_memory_attention_mass_finite_and_positive",
            "memory_id_constant",
            "memory_slot_count_constant",
            "normalization_constant",
        )
    ) and all(
        record["prompt_split_no_memory_parity"] for record in output_records
    )
    acceptance = {
        "persistent_side_kv_mechanism_integrity": mechanism_passed,
        "matched_side_kv_accuracy_above_no_memory": effect_is_positive(
            accuracy_effects["matched_side_kv_vs_no_memory"]
        ),
        "matched_side_kv_accuracy_above_shuffled": effect_is_positive(
            accuracy_effects["matched_side_kv_vs_shuffled_side_kv"]
        ),
        "matched_side_kv_format_not_below_no_memory": (
            float(format_effect["mean_treatment_minus_control"]) >= 0.0
        ),
    }
    summary = {
        "schema_version": E1C_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "status": "passed" if all(acceptance.values()) else "did_not_pass",
        "formal_e1c_passed": all(acceptance.values()),
        "sample_count": len(output_records),
        "conditions": condition_summary,
        "accuracy_effects": accuracy_effects,
        "matched_side_kv_vs_no_memory_format": format_effect,
        "mechanism_diagnostics": {
            "normalization": E1C_MEMORY_SCORE_NORMALIZATION,
            "all_runtime_invariants_passed": mechanism_passed,
            "prompt_split_no_memory_parity_count": sum(
                record["prompt_split_no_memory_parity"]
                for record in output_records
            ),
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
        },
        "acceptance": acceptance,
    }
    write_json(args.output_dir / "e1c_summary.json", summary)
    write_json(args.output_dir / "run_report.json", {
        "schema_version": "experience-memory-e1c-run-report-v1",
        "created_at": utc_now(),
        "status": "completed",
        "sample_count": len(output_records),
        "inputs": {
            "assignment_manifest_sha256": file_sha256(args.assignment_manifest),
            "e1b_results_sha256": file_sha256(args.e1b_results),
            "e1b_summary_sha256": file_sha256(args.e1b_summary),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
        },
        "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
    })
    print(
        f"[e1c-eval] status={summary['status']} output={args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()
