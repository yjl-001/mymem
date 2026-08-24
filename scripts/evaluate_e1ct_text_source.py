#!/usr/bin/env python3
"""Decompose E1-C text effects into wrapper and MemoryRecord payload sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1_staged import (
    E1B_MANIFEST_SCHEMA,
    E1C_RESULTS_SCHEMA,
    E1C_SUMMARY_SCHEMA,
    E1CT_RESULTS_SCHEMA,
    E1CT_SUMMARY_SCHEMA,
    E1BRetrievalAssignment,
    E1CTTextSourceDecision,
    render_single_experience,
    render_single_experience_guidance,
    render_single_experience_payload,
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
    load_hashed_manifest,
    processed_solution,
    prompt_token_ids,
    score_completion,
    summarize_conditions,
    utc_now,
    validate_resolved_revisions,
    write_json,
)


SPLIT_PREFILL_PATH = "split-before-final-prompt-token-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--e1c-results", type=Path, required=True)
    parser.add_argument("--e1c-run-report", type=Path, required=True)
    parser.add_argument("--e1c-summary", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(value) for value in iter_jsonl(path)]


def _copy_source_condition(
    source_record: Mapping[str, Any], condition: str
) -> dict[str, Any]:
    row = dict(source_record["conditions"][condition])
    if row.get("prefill_path") != SPLIT_PREFILL_PATH:
        raise ValueError(f"Source E1-C condition is not split-prefill: {condition}")
    row["source_artifact_role"] = "frozen_e1c_v3_reference"
    return row


def _source_e1c_mechanism_valid(record: Mapping[str, Any]) -> bool:
    primary_conditions = (
        "split_no_memory",
        "split_matched_text",
        "split_shuffled_text",
        "matched_persistent_side_kv",
        "shuffled_persistent_side_kv",
    )
    if record.get("primary_prefill_path") != SPLIT_PREFILL_PATH or not all(
        record.get("conditions", {}).get(condition, {}).get("prefill_path")
        == SPLIT_PREFILL_PATH
        for condition in primary_conditions
    ):
        return False
    if record.get("prefill_path_diagnostics", {}).get(
        "split_no_memory_repeat", {}
    ).get("exact_match") is not True:
        return False
    requirements = (
        "one_trace_per_generated_token",
        "native_cache_length_matches_real_tokens",
        "all_memory_attention_mass_finite_and_positive",
        "memory_id_constant",
        "memory_slot_count_constant",
        "normalization_constant",
        "baseline_first_token_matches_split_no_memory",
    )
    return all(
        record.get("conditions", {}).get(condition, {}).get("side_kv", {}).get(
            requirement
        )
        is True
        for condition in (
            "matched_persistent_side_kv",
            "shuffled_persistent_side_kv",
        )
        for requirement in requirements
    )


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime

    manifest = load_hashed_manifest(
        args.assignment_manifest, schema=E1B_MANIFEST_SCHEMA
    )
    artifact_hashes = {
        "assignment_manifest": file_sha256(args.assignment_manifest),
        "e1c_results": file_sha256(args.e1c_results),
        "e1c_run_report": file_sha256(args.e1c_run_report),
        "e1c_summary": file_sha256(args.e1c_summary),
        "memory_records": file_sha256(args.memory_records),
        "split_manifest": file_sha256(args.split_manifest),
    }
    assignments = tuple(
        E1BRetrievalAssignment.from_dict(value) for value in manifest["assignments"]
    )
    if any(not assignment.assigned for assignment in assignments):
        raise ValueError("E1C-T requires complete matched/shuffled assignments")
    inputs = manifest["inputs"]
    if artifact_hashes["memory_records"] != inputs["memory_records_sha256"]:
        raise ValueError("MemoryRecords differ from the frozen E1-B assignment")
    if artifact_hashes["split_manifest"] != inputs["split_manifest_sha256"]:
        raise ValueError("Split manifest differs from the frozen E1-B assignment")

    e1c_run = json.loads(args.e1c_run_report.read_text(encoding="utf-8"))
    if (
        e1c_run.get("schema_version") != "experience-memory-e1c-run-report-v2"
        or e1c_run.get("status") != "completed"
        or e1c_run.get("results", {}).get("sha256")
        != artifact_hashes["e1c_results"]
        or e1c_run.get("inputs", {}).get("assignment_manifest_sha256")
        != artifact_hashes["assignment_manifest"]
        or e1c_run.get("inputs", {}).get("memory_records_sha256")
        != artifact_hashes["memory_records"]
        or e1c_run.get("inputs", {}).get("split_manifest_sha256")
        != artifact_hashes["split_manifest"]
    ):
        raise ValueError("E1-C v3 artifacts do not match their run report")
    e1c_summary = json.loads(args.e1c_summary.read_text(encoding="utf-8"))
    if (
        e1c_summary.get("schema_version") != E1C_SUMMARY_SCHEMA
        or e1c_summary.get("sample_count") != len(assignments)
    ):
        raise ValueError("E1C-T requires the matching E1-C v3 summary schema")
    e1c_records = _read_jsonl(args.e1c_results)
    if any(record.get("schema_version") != E1C_RESULTS_SCHEMA for record in e1c_records):
        raise ValueError("Unexpected E1-C results schema")
    e1c_by_sample = {str(record["sample_id"]): record for record in e1c_records}
    if set(e1c_by_sample) != {assignment.sample_id for assignment in assignments}:
        raise ValueError("E1-C source results and assignments have different samples")
    if not all(_source_e1c_mechanism_valid(record) for record in e1c_records):
        raise ValueError(
            "E1C-T requires mechanism-valid evidence in every E1-C result"
        )

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
        raise ValueError("Assignments reference unknown MemoryRecords")

    reasoner = manifest["reasoner"]
    if args.dtype != reasoner["dtype"]:
        raise ValueError("E1C-T dtype differs from the frozen assignment")
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
        model=model, tokenizer=tokenizer, reasoner=reasoner, label="E1C-T"
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

    source_conditions = (
        "split_no_memory",
        "split_matched_text",
        "split_shuffled_text",
    )
    new_conditions = (
        "split_wrapper_only",
        "split_payload_only_matched",
        "split_payload_only_shuffled",
    )
    conditions = source_conditions + new_conditions
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
            source_e1c = e1c_by_sample[assignment.sample_id]
            if (
                source_e1c.get("assignment_manifest_sha256")
                != manifest["manifest_sha256"]
                or source_e1c.get("primary_prefill_path") != SPLIT_PREFILL_PATH
                or source_e1c.get("matched_memory", {}).get("memory_id")
                != assignment.matched_memory.memory_id
                or source_e1c.get("shuffled_memory", {}).get("memory_id")
                != assignment.shuffled_memory.memory_id
            ):
                raise ValueError(f"E1-C provenance drift for {assignment.sample_id}")
            condition_rows = {
                condition: _copy_source_condition(source_e1c, condition)
                for condition in source_conditions
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
                    raise ValueError(f"Memory metadata drift for {choice.memory_id}")
                wrapped_prompt_ids = prompt_token_ids(
                    tokenizer,
                    question=question,
                    memory_text=render_single_experience(memory_record),
                )
                source_wrapped = condition_rows[f"split_{label}_text"]
                if (
                    source_wrapped.get("prompt_token_ids_sha256")
                    != canonical_json_sha256(wrapped_prompt_ids)
                ):
                    raise ValueError(
                        f"Wrapped text prompt drift for {assignment.sample_id}"
                    )
                payload_prompt_ids = prompt_token_ids(
                    tokenizer,
                    question=question,
                    memory_text=render_single_experience_payload(memory_record),
                )
                started = time.perf_counter()
                generated = runtime.generate_prompt_split(
                    payload_prompt_ids, audit_full_prefill=False
                )
                elapsed = time.perf_counter() - started
                condition_name = f"split_payload_only_{label}"
                condition_rows[condition_name] = score_completion(
                    tokenizer=tokenizer,
                    completion_token_ids=generated.completion_token_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=elapsed,
                    prompt_token_count=len(payload_prompt_ids),
                    memory_ids=(choice.memory_id,),
                )
                condition_rows[condition_name].update({
                    "prefill_path": SPLIT_PREFILL_PATH,
                    "prompt_token_ids_sha256": canonical_json_sha256(
                        payload_prompt_ids
                    ),
                    "text_source": "memory_record_payload_only",
                })

            wrapper_prompt_ids = prompt_token_ids(
                tokenizer,
                question=question,
                memory_text=render_single_experience_guidance(),
            )
            started = time.perf_counter()
            wrapper_generated = runtime.generate_prompt_split(
                wrapper_prompt_ids, audit_full_prefill=False
            )
            wrapper_elapsed = time.perf_counter() - started
            condition_rows["split_wrapper_only"] = score_completion(
                tokenizer=tokenizer,
                completion_token_ids=wrapper_generated.completion_token_ids,
                ground_truth=ground_truth,
                runtime_seconds=wrapper_elapsed,
                prompt_token_count=len(wrapper_prompt_ids),
            )
            condition_rows["split_wrapper_only"].update({
                "prefill_path": SPLIT_PREFILL_PATH,
                "prompt_token_ids_sha256": canonical_json_sha256(
                    wrapper_prompt_ids
                ),
                "text_source": "single_experience_guidance_wrapper_only",
            })
            record = {
                "schema_version": E1CT_RESULTS_SCHEMA,
                "sample_id": assignment.sample_id,
                "logical_split": assignment.logical_split,
                "question_sha256": assignment.question_sha256,
                "assignment_manifest_sha256": manifest["manifest_sha256"],
                "source_e1c_result_sha256": canonical_json_sha256(source_e1c),
                "source_e1c_results_file_sha256": artifact_hashes["e1c_results"],
                "matched_memory": assignment.matched_memory.to_dict(),
                "shuffled_memory": assignment.shuffled_memory.to_dict(),
                "primary_prefill_path": SPLIT_PREFILL_PATH,
                "conditions": condition_rows,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            output_records.append(record)
            if position % 10 == 0 or position == len(assignments):
                print(f"[e1ct-eval] {position}/{len(assignments)}", flush=True)

    condition_summary = summarize_conditions(output_records, conditions)
    diagnostic_builder = PairedConditionDiagnostics(
        output_records, bootstrap_resamples=args.bootstrap_resamples
    )
    paired_diagnostics = diagnostic_builder.summarize(
        (
            PairedConditionComparison(
                "wrapper_only_vs_no_memory",
                "split_wrapper_only",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "payload_only_matched_vs_no_memory",
                "split_payload_only_matched",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "payload_only_shuffled_vs_no_memory",
                "split_payload_only_shuffled",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "wrapped_matched_vs_no_memory",
                "split_matched_text",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "wrapped_shuffled_vs_no_memory",
                "split_shuffled_text",
                "split_no_memory",
            ),
            PairedConditionComparison(
                "wrapped_matched_vs_payload_only_matched",
                "split_matched_text",
                "split_payload_only_matched",
            ),
            PairedConditionComparison(
                "wrapped_shuffled_vs_payload_only_shuffled",
                "split_shuffled_text",
                "split_payload_only_shuffled",
            ),
            PairedConditionComparison(
                "payload_only_matched_vs_payload_only_shuffled",
                "split_payload_only_matched",
                "split_payload_only_shuffled",
            ),
            PairedConditionComparison(
                "wrapped_matched_vs_wrapped_shuffled",
                "split_matched_text",
                "split_shuffled_text",
            ),
        )
    )
    decision = E1CTTextSourceDecision.from_effects(
        format_effects=paired_diagnostics["format_effects"],
        diagnostic_answer_effects=paired_diagnostics[
            "diagnostic_answer_effects"
        ],
    )
    path_integrity = all(
        record["primary_prefill_path"] == SPLIT_PREFILL_PATH
        and all(
            record["conditions"][condition]["prefill_path"]
            == SPLIT_PREFILL_PATH
            for condition in conditions
        )
        for record in output_records
    )
    if not path_integrity:
        raise RuntimeError("E1C-T condition path integrity failed")
    summary = {
        "schema_version": E1CT_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "status": "completed",
        "formal_task_claim": False,
        "sample_count": len(output_records),
        "source_e1c_formal_passed_reported": (
            e1c_summary.get("formal_e1c_passed") is True
        ),
        "condition_roles": {
            "frozen_e1c_v3_references": list(source_conditions),
            "new_text_source_conditions": list(new_conditions),
        },
        "conditions": condition_summary,
        "accuracy_effects": paired_diagnostics["accuracy_effects"],
        "diagnostic_answer_effects": paired_diagnostics[
            "diagnostic_answer_effects"
        ],
        "format_effects": paired_diagnostics["format_effects"],
        "strict_accuracy_transition_diagnostics": paired_diagnostics[
            "strict_accuracy_transition_diagnostics"
        ],
        "completion_difference_diagnostics": paired_diagnostics[
            "completion_difference_diagnostics"
        ],
        "component_diagnostic": {
            "text_source_decomposition_completed": True,
            "source_e1c_mechanism_revalidated_from_results": True,
            "same_split_prefill_path_integrity": path_integrity,
            "fixed_strength_test_allowed": (
                decision.next_step == "e1cs_fixed_log10_memory_odds_test"
            ),
        },
        "decision": decision.to_dict(),
    }
    write_json(args.output_dir / "e1ct_summary.json", summary)
    write_json(args.output_dir / "run_report.json", {
        "schema_version": "experience-memory-e1ct-run-report-v1",
        "created_at": utc_now(),
        "status": "completed",
        "sample_count": len(output_records),
        "inputs": {
            "assignment_manifest_sha256": artifact_hashes[
                "assignment_manifest"
            ],
            "e1c_results_sha256": artifact_hashes["e1c_results"],
            "e1c_run_report_sha256": artifact_hashes["e1c_run_report"],
            "e1c_summary_sha256": artifact_hashes["e1c_summary"],
            "memory_records_sha256": artifact_hashes["memory_records"],
            "split_manifest_sha256": artifact_hashes["split_manifest"],
        },
        "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
        "summary": {
            "path": "e1ct_summary.json",
            "sha256": file_sha256(args.output_dir / "e1ct_summary.json"),
        },
    })
    print(
        f"[e1ct-eval] next_step={decision.next_step} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
