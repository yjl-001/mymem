#!/usr/bin/env python3
"""Evaluate frozen E1-B BM25 assignments through the explicit text channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1_staged import (
    E1B_MANIFEST_SCHEMA,
    E1B_RESULTS_SCHEMA,
    E1B_SUMMARY_SCHEMA,
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
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime

    manifest = load_hashed_manifest(
        args.assignment_manifest, schema=E1B_MANIFEST_SCHEMA
    )
    if manifest.get("status") != "frozen" or manifest.get("answer_or_reward_used") is not False:
        raise ValueError("E1-B assignment manifest is not frozen and answer-blind")
    inputs = manifest["inputs"]
    if file_sha256(args.memory_records) != inputs["memory_records_sha256"]:
        raise ValueError("MemoryRecords differ from E1-B assignment input")
    if file_sha256(args.split_manifest) != inputs["split_manifest_sha256"]:
        raise ValueError("Split manifest differs from E1-B assignment input")
    assignments = tuple(
        E1BRetrievalAssignment.from_dict(value) for value in manifest["assignments"]
    )
    if any(not assignment.assigned for assignment in assignments):
        raise ValueError("E1-B manifest has an assignment without shuffled control")
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    record_by_id = {record.memory_id: record for record in records}
    used_ids = {
        choice.memory_id
        for assignment in assignments
        for choice in (assignment.matched_memory, assignment.shuffled_memory)
        if choice is not None
    }
    if not used_ids <= set(record_by_id):
        raise ValueError("E1-B assignments reference unknown MemoryRecords")

    reasoner = manifest["reasoner"]
    if args.dtype != reasoner["dtype"]:
        raise ValueError("E1-B runtime dtype differs from frozen assignment")
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
        model=model,
        tokenizer=tokenizer,
        reasoner=reasoner,
        label="E1-B evaluation",
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

    conditions = ("no_memory", "matched_text", "shuffled_text")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    output_records: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for position, assignment in enumerate(assignments, start=1):
            source = dataset[assignment.source_index]
            question = str(source["question"]).strip()
            if text_sha256(question) != assignment.question_sha256:
                raise ValueError(f"Question hash mismatch for {assignment.sample_id}")
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
            decoded_preanswer = tokenizer.decode(
                list(assignment.preanswer_completion_token_ids),
                skip_special_tokens=True,
            ).strip()
            if text_sha256(decoded_preanswer) != assignment.preanswer_completion_text_sha256:
                raise ValueError(f"Preanswer decode drift for {assignment.sample_id}")
            condition_rows = {
                "no_memory": score_completion(
                    tokenizer=tokenizer,
                    completion_token_ids=assignment.preanswer_completion_token_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=None,
                    prompt_token_count=len(base_prompt_ids),
                )
            }
            for condition, choice in (
                ("matched_text", assignment.matched_memory),
                ("shuffled_text", assignment.shuffled_memory),
            ):
                assert choice is not None
                memory_record = record_by_id[choice.memory_id]
                memory_text = render_single_experience(memory_record)
                treatment_prompt_ids = prompt_token_ids(
                    tokenizer, question=question, memory_text=memory_text
                )
                started = time.perf_counter()
                completion_ids = runtime.generate_vanilla(treatment_prompt_ids)
                elapsed = time.perf_counter() - started
                condition_rows[condition] = score_completion(
                    tokenizer=tokenizer,
                    completion_token_ids=completion_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=elapsed,
                    prompt_token_count=len(treatment_prompt_ids),
                    memory_ids=(choice.memory_id,),
                )
                condition_rows[condition]["prompt_token_ids_sha256"] = (
                    canonical_json_sha256(treatment_prompt_ids)
                )
            record = {
                "schema_version": E1B_RESULTS_SCHEMA,
                "sample_id": assignment.sample_id,
                "logical_split": assignment.logical_split,
                "question_sha256": assignment.question_sha256,
                "assignment_manifest_sha256": manifest["manifest_sha256"],
                "preanswer_used_in_second_prompt": False,
                "retrieval_query": assignment.retrieval_query,
                "matched_memory": assignment.matched_memory.to_dict(),
                "shuffled_memory": assignment.shuffled_memory.to_dict(),
                "conditions": condition_rows,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            output_records.append(record)
            if position % 10 == 0 or position == len(assignments):
                print(f"[e1b-eval] {position}/{len(assignments)}", flush=True)

    conditions_summary = summarize_conditions(output_records, conditions)
    accuracy_effects = {
        "matched_vs_no_memory": paired_condition_effect(
            output_records,
            treatment="matched_text",
            control="no_memory",
            field="final_reward",
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_shuffled": paired_condition_effect(
            output_records,
            treatment="matched_text",
            control="shuffled_text",
            field="final_reward",
            resamples=args.bootstrap_resamples,
        ),
    }
    format_effect = paired_condition_effect(
        output_records,
        treatment="matched_text",
        control="no_memory",
        field="format_valid",
        resamples=args.bootstrap_resamples,
    )
    acceptance = {
        "matched_accuracy_above_no_memory": effect_is_positive(
            accuracy_effects["matched_vs_no_memory"]
        ),
        "matched_accuracy_above_shuffled": effect_is_positive(
            accuracy_effects["matched_vs_shuffled"]
        ),
        "matched_format_not_below_no_memory": (
            float(format_effect["mean_treatment_minus_control"]) >= 0.0
        ),
        "first_response_never_in_second_prompt": all(
            record["preanswer_used_in_second_prompt"] is False
            for record in output_records
        ),
    }
    margins = [
        float(item.retrieval_query["top1_top2_margin"])
        for item in assignments
        if item.retrieval_query.get("top1_top2_margin") is not None
    ]
    prompt_token_differences = [
        abs(
            int(record["conditions"]["matched_text"]["prompt_token_count"])
            - int(record["conditions"]["shuffled_text"]["prompt_token_count"])
        )
        for record in output_records
    ]
    summary = {
        "schema_version": E1B_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "status": "passed" if all(acceptance.values()) else "did_not_pass",
        "formal_e1b_passed": all(acceptance.values()),
        "sample_count": len(output_records),
        "provenance": {
            "assignment_manifest_logical_sha256": manifest["manifest_sha256"],
            "assignment_manifest_file_sha256": file_sha256(
                args.assignment_manifest
            ),
            "results_sha256": file_sha256(results_path),
        },
        "conditions": conditions_summary,
        "accuracy_effects": accuracy_effects,
        "matched_vs_no_memory_format": format_effect,
        "retrieval_diagnostics": {
            "mean_top1_score": sum(
                float(item.retrieval_query["top1_score"]) for item in assignments
            ) / len(assignments),
            "mean_top1_top2_margin": (
                sum(margins) / len(margins) if margins else None
            ),
        },
        "pairing_diagnostics": {
            "shuffle": manifest["configuration"]["shuffle"],
            "exact_prompt_token_count_match_count": sum(
                value == 0 for value in prompt_token_differences
            ),
            "mean_absolute_prompt_token_count_difference": sum(
                prompt_token_differences
            ) / len(prompt_token_differences),
            "max_absolute_prompt_token_count_difference": max(
                prompt_token_differences
            ),
        },
        "acceptance": acceptance,
    }
    write_json(args.output_dir / "e1b_summary.json", summary)
    write_json(args.output_dir / "run_report.json", {
        "schema_version": "experience-memory-e1b-run-report-v1",
        "created_at": utc_now(),
        "status": "completed",
        "sample_count": len(output_records),
        "inputs": {
            "assignment_manifest_sha256": file_sha256(args.assignment_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "split_manifest_sha256": file_sha256(args.split_manifest),
        },
        "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
    })
    print(
        f"[e1b-eval] status={summary['status']} output={args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()
