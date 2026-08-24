#!/usr/bin/env python3
"""Audit E1 pairing and summarize matched-memory causal effects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.e1 import (
    E1_CONDITIONS,
    E1_MANIFEST_SCHEMA,
    E1_RESULTS_SCHEMA,
    E1_SUMMARY_SCHEMA,
    E1Assignment,
    paired_binary_effect,
)
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
)
from memgen.experience.system import ExperienceMemorySystemProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--min-primary-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def condition_metric(
    records: list[dict[str, Any]],
    condition: str,
    field: str,
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, bool | float]:
    selected = [
        record for record in records if predicate is None or predicate(record)
    ]

    def value(record: dict[str, Any]) -> bool | float:
        row = record["conditions"][condition]
        if field == "diagnostic_answer_correct":
            return row.get("verifier", {}).get(
                "diagnostic_answer_correct"
            ) is True
        return row[field]

    return {
        str(record["sample_id"]): value(record)
        for record in selected
    }


def paired(
    records: list[dict[str, Any]],
    treatment: str,
    control: str,
    field: str,
    *,
    predicate: Callable[[dict[str, Any]], bool] | None,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    return paired_binary_effect(
        condition_metric(records, treatment, field, predicate=predicate),
        condition_metric(records, control, field, predicate=predicate),
        seed=seed,
        resamples=resamples,
    )


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0 or args.min_primary_pairs <= 0:
        raise ValueError("Bootstrap count and minimum primary pairs must be positive")

    manifest = json.loads(args.assignment_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != E1_MANIFEST_SCHEMA:
        raise ValueError("Unexpected E1 assignment manifest schema")
    logical_manifest_hash = manifest.get("manifest_sha256")
    actual_manifest_hash = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        }
    )
    if logical_manifest_hash != actual_manifest_hash:
        raise ValueError("E1 assignment manifest hash mismatch")
    assignments = {
        item.sample_id: item
        for item in (
            E1Assignment.from_dict(value) for value in manifest["assignments"]
        )
    }
    profile = ExperienceMemorySystemProfile.from_dict(
        manifest.get("configuration", {}).get("system_profile", {})
    )
    records = list(iter_jsonl(args.results))
    if any(record.get("schema_version") != E1_RESULTS_SCHEMA for record in records):
        raise ValueError("Unexpected E1 result schema")
    records_by_id = {str(record["sample_id"]): record for record in records}
    if len(records_by_id) != len(records) or set(records_by_id) != set(assignments):
        raise ValueError("E1 results and assignments do not cover identical sample IDs")
    run_report = json.loads(args.run_report.read_text(encoding="utf-8"))
    if (
        run_report.get("schema_version") != "experience-memory-e1-run-report-v2"
        or run_report.get("system_profile") != profile.to_dict()
        or run_report.get("status") != "completed"
        or run_report.get("results", {}).get("sha256") != file_sha256(args.results)
        or run_report.get("inputs", {}).get("assignment_manifest_sha256")
        != file_sha256(args.assignment_manifest)
    ):
        raise ValueError("E1 run report does not authenticate the supplied artifacts")

    pairing_violations: list[dict[str, str]] = []
    for sample_id, record in records_by_id.items():
        assignment = assignments[sample_id]
        conditions = record.get("conditions", {})
        if (
            record.get("assignment_manifest_sha256") != logical_manifest_hash
            or record.get("question_sha256") != assignment.question_sha256
            or record.get("logical_split") != assignment.logical_split
            or record.get("system_profile") != profile.to_dict()
        ):
            pairing_violations.append(
                {"sample_id": sample_id, "reason": "assignment_identity_mismatch"}
            )
        if set(conditions) != set(E1_CONDITIONS):
            pairing_violations.append(
                {"sample_id": sample_id, "reason": "condition_set_mismatch"}
            )
            continue
        if record.get("prefix_token_ids_sha256") != assignment.prefix_token_ids_sha256:
            pairing_violations.append(
                {"sample_id": sample_id, "reason": "prefix_hash_mismatch"}
            )
        if assignment.assigned:
            matched = conditions["matched_persistent_memory"]
            shuffled = conditions["shuffled_persistent_memory"]
            assert assignment.matched_memory is not None
            assert assignment.shuffled_memory is not None
            frozen_completion_prefix = list(
                assignment.prefix_token_ids[assignment.prompt_token_count :]
            )
            for condition, output, choice in (
                ("matched", matched, assignment.matched_memory),
                ("shuffled", shuffled, assignment.shuffled_memory),
            ):
                if (
                    output.get("memory_id") != choice.memory_id
                    or output.get("payload_hash") != choice.payload_hash
                    or output.get("side_kv_applied") is not True
                ):
                    pairing_violations.append(
                        {
                            "sample_id": sample_id,
                            "reason": f"{condition}_memory_assignment_mismatch",
                        }
                    )
                completion_ids = output.get("completion_token_ids")
                if not isinstance(completion_ids, list):
                    pairing_violations.append(
                        {
                            "sample_id": sample_id,
                            "reason": f"{condition}_completion_prefix_mismatch",
                        }
                    )
                    continue
                if (
                    completion_ids[: len(frozen_completion_prefix)]
                    != frozen_completion_prefix
                ):
                    pairing_violations.append(
                        {
                            "sample_id": sample_id,
                            "reason": f"{condition}_completion_prefix_mismatch",
                        }
                    )
                trace = output.get("memory_attention") or {}
                mass = trace.get("mean_memory_attention_mass")
                if not isinstance(mass, (int, float)) or float(mass) <= 0:
                    pairing_violations.append(
                        {
                            "sample_id": sample_id,
                            "reason": f"{condition}_memory_attention_not_positive",
                        }
                    )
                expected_trace_count = (
                    len(completion_ids) - len(frozen_completion_prefix)
                )
                if expected_trace_count <= 0:
                    pairing_violations.append({
                        "sample_id": sample_id,
                        "reason": f"{condition}_no_post_trigger_completion",
                    })
                    continue
                expected_native_lengths = list(range(
                    len(assignment.prefix_token_ids),
                    len(assignment.prefix_token_ids) + expected_trace_count,
                ))
                masses = trace.get("memory_attention_masses")
                trace_count = int(trace.get("trace_count", -1))
                if (
                    trace.get("memory_ids") != [choice.memory_id]
                    or trace.get("memory_slot_counts")
                    != [choice.kv_valid_slot_count]
                    or trace_count != expected_trace_count
                    or trace.get("native_key_lengths") != expected_native_lengths
                    or trace.get("native_key_lengths_sha256")
                    != canonical_json_sha256(expected_native_lengths)
                    or not isinstance(masses, list)
                    or len(masses) != trace_count
                    or not all(
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        and float(value) > 0.0
                        for value in (masses or [])
                    )
                    or trace.get("memory_attention_masses_sha256")
                    != canonical_json_sha256(masses)
                    or trace.get("memory_score_normalizations")
                    != [profile.memory_score_normalization]
                    or trace.get("memory_score_biases")
                    != [profile.memory_score_bias]
                    or any(
                        trace.get(requirement) is not True
                        for requirement in (
                            "one_trace_per_post_trigger_token",
                            "native_cache_length_matches_real_tokens",
                            "all_memory_attention_mass_finite_and_positive",
                            "memory_id_constant_and_matched",
                            "memory_slot_count_constant_and_matched",
                            "normalization_constant_and_matched",
                            "memory_score_bias_constant_and_matched",
                            "baseline_first_token_matches_gate_observation",
                        )
                    )
                ):
                    pairing_violations.append(
                        {
                            "sample_id": sample_id,
                            "reason": f"{condition}_side_kv_trace_mismatch",
                        }
                    )
        elif any(
            conditions[name].get("side_kv_applied")
            for name in (
                "matched_persistent_memory",
                "shuffled_persistent_memory",
            )
        ):
            pairing_violations.append(
                {"sample_id": sample_id, "reason": "memory_applied_to_unassigned_sample"}
            )
        elif any(
            conditions[name].get("completion_token_ids_sha256")
            != conditions["gate_observation_only"].get(
                "completion_token_ids_sha256"
            )
            for name in (
                "matched_persistent_memory",
                "shuffled_persistent_memory",
            )
        ):
            pairing_violations.append(
                {
                    "sample_id": sample_id,
                    "reason": "unassigned_condition_changed_completion",
                }
            )

    ordered_records = [records_by_id[sample_id] for sample_id in sorted(records_by_id)]
    assigned_predicate = lambda record: bool(record.get("assigned"))
    exact_slot_predicate = lambda record: bool(record.get("assigned")) and (
        int(record["matched_memory"]["kv_valid_slot_count"])
        == int(record["shuffled_memory"]["kv_valid_slot_count"])
    )
    primary = {
        "matched_vs_shuffled_accuracy": paired(
            ordered_records,
            "matched_persistent_memory",
            "shuffled_persistent_memory",
            "final_reward",
            predicate=assigned_predicate,
            seed=args.seed,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_gate_accuracy": paired(
            ordered_records,
            "matched_persistent_memory",
            "gate_observation_only",
            "final_reward",
            predicate=assigned_predicate,
            seed=args.seed + 1,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_vanilla_format": paired(
            ordered_records,
            "matched_persistent_memory",
            "vanilla",
            "format_valid",
            predicate=None,
            seed=args.seed + 2,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_shuffled_diagnostic_answer": paired(
            ordered_records,
            "matched_persistent_memory",
            "shuffled_persistent_memory",
            "diagnostic_answer_correct",
            predicate=assigned_predicate,
            seed=args.seed + 6,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_gate_diagnostic_answer": paired(
            ordered_records,
            "matched_persistent_memory",
            "gate_observation_only",
            "diagnostic_answer_correct",
            predicate=assigned_predicate,
            seed=args.seed + 7,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_shuffled_format": paired(
            ordered_records,
            "matched_persistent_memory",
            "shuffled_persistent_memory",
            "format_valid",
            predicate=assigned_predicate,
            seed=args.seed + 8,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_gate_format": paired(
            ordered_records,
            "matched_persistent_memory",
            "gate_observation_only",
            "format_valid",
            predicate=assigned_predicate,
            seed=args.seed + 9,
            resamples=args.bootstrap_resamples,
        ),
    }
    intention_to_treat = {
        "matched_vs_shuffled_accuracy": paired(
            ordered_records,
            "matched_persistent_memory",
            "shuffled_persistent_memory",
            "final_reward",
            predicate=None,
            seed=args.seed + 3,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_gate_accuracy": paired(
            ordered_records,
            "matched_persistent_memory",
            "gate_observation_only",
            "final_reward",
            predicate=None,
            seed=args.seed + 4,
            resamples=args.bootstrap_resamples,
        ),
    }
    exact_slot_sensitivity = paired(
        ordered_records,
        "matched_persistent_memory",
        "shuffled_persistent_memory",
        "final_reward",
        predicate=exact_slot_predicate,
        seed=args.seed + 5,
        resamples=args.bootstrap_resamples,
    )

    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in E1_CONDITIONS:
        rows = [record["conditions"][condition] for record in ordered_records]
        condition_summaries[condition] = {
            "sample_count": len(rows),
            "accuracy": mean([float(row["final_reward"]) for row in rows]),
            "format_accuracy": mean([float(bool(row["format_valid"])) for row in rows]),
            "diagnostic_answer_accuracy": mean([
                float(
                    row.get("verifier", {}).get("diagnostic_answer_correct")
                    is True
                )
                for row in rows
            ]),
            "diagnostic_answer_coverage": mean([
                float(
                    row.get("verifier", {}).get("diagnostic_answer_correct")
                    is not None
                )
                for row in rows
            ]),
            "mean_generation_length": mean(
                [float(row["generation_length"]) for row in rows]
            ),
            "side_kv_applied_count": sum(bool(row["side_kv_applied"]) for row in rows),
        }
    matched_outputs = [
        record["conditions"]["matched_persistent_memory"]
        for record in ordered_records
        if record.get("assigned")
    ]
    shuffled_outputs = [
        record["conditions"]["shuffled_persistent_memory"]
        for record in ordered_records
        if record.get("assigned")
    ]
    retrieval_scores = [
        float(record["matched_memory"]["retrieval_score"])
        for record in ordered_records
        if record.get("assigned")
    ]
    mechanism_diagnostics = {
        "persistent_from_trigger_through_eos": True,
        "runtime_profile": profile.to_dict(),
        "matched_mean_memory_attention_mass": mean(
            [
                float(row["memory_attention"]["mean_memory_attention_mass"])
                for row in matched_outputs
            ]
        ),
        "shuffled_mean_memory_attention_mass": mean(
            [
                float(row["memory_attention"]["mean_memory_attention_mass"])
                for row in shuffled_outputs
            ]
        ),
        "matched_mean_first_step_logits_kl": mean(
            [
                float(row["first_step_logits_kl_baseline_to_memory"])
                for row in matched_outputs
            ]
        ),
        "shuffled_mean_first_step_logits_kl": mean(
            [
                float(row["first_step_logits_kl_baseline_to_memory"])
                for row in shuffled_outputs
            ]
        ),
        "mean_bm25_top1_score": mean(retrieval_scores),
        "matched_mean_trace_count": mean([
            float(row["memory_attention"]["trace_count"])
            for row in matched_outputs
        ]),
        "shuffled_mean_trace_count": mean([
            float(row["memory_attention"]["trace_count"])
            for row in shuffled_outputs
        ]),
        "matched_baseline_first_token_match_count": sum(
            row["memory_attention"][
                "baseline_first_token_matches_gate_observation"
            ]
            for row in matched_outputs
        ),
        "shuffled_baseline_first_token_match_count": sum(
            row["memory_attention"][
                "baseline_first_token_matches_gate_observation"
            ]
            for row in shuffled_outputs
        ),
    }

    matched_shuffled = primary["matched_vs_shuffled_accuracy"]
    matched_gate = primary["matched_vs_gate_accuracy"]
    format_effect = primary["matched_vs_vanilla_format"]

    def lower_bound_positive(effect: dict[str, Any]) -> bool:
        interval = effect.get("bootstrap_95_ci")
        return bool(
            effect.get("paired_sample_count", 0) >= args.min_primary_pairs
            and isinstance(interval, list)
            and interval[0] > 0
        )

    acceptance = {
        "assignment_and_pairing_integrity": not pairing_violations,
        "vanilla_matches_gate_observation_only": all(
            record.get("vanilla_matches_gate_observation_only")
            for record in ordered_records
        ),
        "minimum_primary_pair_count": (
            matched_shuffled["paired_sample_count"] >= args.min_primary_pairs
        ),
        "matched_accuracy_above_shuffled": lower_bound_positive(
            matched_shuffled
        ),
        "matched_accuracy_above_gate_only": lower_bound_positive(matched_gate),
        "matched_format_not_below_vanilla": (
            format_effect["mean_treatment_minus_control"] is not None
            and format_effect["mean_treatment_minus_control"] >= 0
        ),
    }
    frozen_effect_criteria_passed = all(acceptance.values())
    runtime_integrity_passed = (
        not pairing_violations
        and acceptance["vanilla_matches_gate_observation_only"]
    )
    output = {
        "schema_version": E1_SUMMARY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if runtime_integrity_passed else "invalid",
        # The current layer-24 channel failed E1C-S. E1-D can measure the
        # frozen effect criteria, but cannot override that upstream stop rule.
        "formal_e1_passed": False,
        "frozen_effect_criteria_passed": frozen_effect_criteria_passed,
        # E1-D is a calibration-side engineering evaluation. Passing its
        # frozen checks is not, by itself, a held-out task-effect claim.
        "formal_task_claim": False,
        "component_diagnostic": {
            "full_system_implemented": True,
            "gate_retrieval_persistent_side_kv_pipeline_executed": True,
            "runtime_and_pairing_integrity_passed": runtime_integrity_passed,
            "implementation_completeness_is_not_task_effectiveness": True,
        },
        "system_profile": profile.to_dict(),
        "logical_split": manifest["logical_split"],
        "sample_count": len(ordered_records),
        "triggered_count": sum(bool(record.get("triggered")) for record in ordered_records),
        "assigned_count": sum(bool(record.get("assigned")) for record in ordered_records),
        "conditions": condition_summaries,
        "primary_assigned_subset": primary,
        "intention_to_treat": intention_to_treat,
        "exact_slot_count_sensitivity": exact_slot_sensitivity,
        "mechanism_diagnostics": mechanism_diagnostics,
        "pairing_violations": pairing_violations,
        "acceptance": acceptance,
        "criterion": {
            "bootstrap_resamples": args.bootstrap_resamples,
            "minimum_primary_pairs": args.min_primary_pairs,
            "accuracy": "lower endpoint of paired 95% bootstrap CI must be above zero",
            "format": "observed matched format accuracy must not be below vanilla",
        },
        "inputs": {
            "assignment_manifest_path": str(args.assignment_manifest.resolve()),
            "assignment_manifest_sha256": file_sha256(args.assignment_manifest),
            "assignment_manifest_logical_sha256": logical_manifest_hash,
            "results_path": str(args.results.resolve()),
            "results_sha256": file_sha256(args.results),
            "run_report_path": str(args.run_report.resolve()),
            "run_report_sha256": file_sha256(args.run_report),
        },
    }
    output["summary_sha256"] = canonical_json_sha256(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[e1-summary] status={output['status']} "
        f"effect_criteria={output['frozen_effect_criteria_passed']} "
        f"assigned={output['assigned_count']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
