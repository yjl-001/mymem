#!/usr/bin/env python3
"""Audit and summarize the frozen vanilla/matched E1 evaluation."""

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
    E1EvaluationScope,
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

    return {str(record["sample_id"]): value(record) for record in selected}


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


def persistent_trace_valid(
    *,
    trace: dict[str, Any],
    assignment: E1Assignment,
    completion_ids: list[int],
    profile: ExperienceMemorySystemProfile,
) -> bool:
    choice = assignment.matched_memory
    assert choice is not None
    frozen_prefix = list(
        assignment.prefix_token_ids[assignment.prompt_token_count :]
    )
    expected_trace_count = len(completion_ids) - len(frozen_prefix)
    if expected_trace_count <= 0:
        return False
    expected_native_lengths = list(
        range(
            len(assignment.prefix_token_ids),
            len(assignment.prefix_token_ids) + expected_trace_count,
        )
    )
    masses = trace.get("memory_attention_masses")
    if not isinstance(masses, list):
        return False
    required_invariants = (
        "one_trace_per_post_trigger_token",
        "native_cache_length_matches_real_tokens",
        "all_memory_attention_mass_finite_and_positive",
        "memory_id_constant_and_matched",
        "memory_slot_count_constant_and_matched",
        "normalization_constant_and_matched",
        "memory_score_bias_constant_and_matched",
        "baseline_first_token_matches_observation",
    )
    return bool(
        trace.get("memory_ids") == [choice.memory_id]
        and trace.get("memory_slot_counts") == [choice.kv_valid_slot_count]
        and trace.get("trace_count") == expected_trace_count
        and len(masses) == expected_trace_count
        and all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in masses
        )
        and trace.get("memory_attention_masses_sha256")
        == canonical_json_sha256(masses)
        and trace.get("native_key_lengths") == expected_native_lengths
        and trace.get("native_key_lengths_sha256")
        == canonical_json_sha256(expected_native_lengths)
        and trace.get("memory_score_normalizations")
        == [profile.memory_score_normalization]
        and trace.get("memory_score_biases") == [profile.memory_score_bias]
        and all(trace.get(name) is True for name in required_invariants)
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0 or args.min_primary_pairs <= 0:
        raise ValueError("Bootstrap count and minimum primary pairs must be positive")

    manifest = json.loads(args.assignment_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != E1_MANIFEST_SCHEMA:
        raise ValueError("Unexpected E1 assignment manifest schema")
    logical_manifest_hash = manifest.get("manifest_sha256")
    actual_manifest_hash = canonical_json_sha256({
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    if logical_manifest_hash != actual_manifest_hash:
        raise ValueError("E1 assignment manifest hash mismatch")
    profile = ExperienceMemorySystemProfile.from_dict(
        manifest.get("configuration", {}).get("system_profile", {})
    )
    scope = E1EvaluationScope.from_logical_split(
        str(manifest.get("logical_split"))
    )
    if (
        manifest.get("dataset_split") != scope.dataset_split
        or manifest.get("evaluation_role") != scope.evaluation_role
    ):
        raise ValueError("E1 manifest evaluation scope is inconsistent")
    assignments = {
        item.sample_id: item
        for item in (
            E1Assignment.from_dict(value) for value in manifest["assignments"]
        )
    }

    records = list(iter_jsonl(args.results))
    if any(record.get("schema_version") != E1_RESULTS_SCHEMA for record in records):
        raise ValueError("Unexpected E1 result schema")
    records_by_id = {str(record["sample_id"]): record for record in records}
    if len(records_by_id) != len(records) or set(records_by_id) != set(assignments):
        raise ValueError("E1 results and assignments cover different sample IDs")

    run_report = json.loads(args.run_report.read_text(encoding="utf-8"))
    if (
        run_report.get("schema_version") != "experience-memory-e1-run-report-v8"
        or run_report.get("status") != "completed"
        or run_report.get("logical_split") != manifest.get("logical_split")
        or run_report.get("dataset_split") != manifest.get("dataset_split")
        or run_report.get("evaluation_role") != manifest.get("evaluation_role")
        or run_report.get("prompt_contract") != manifest.get("prompt_contract")
        or run_report.get("generation_contract", {}).get("max_new_tokens")
        != manifest.get("configuration", {}).get("max_new_tokens")
        or run_report.get("generation_contract", {}).get("vanilla")
        != manifest.get("configuration", {}).get("vanilla_generation")
        or run_report.get("generation_contract", {}).get("matched")
        != {
            **manifest.get("configuration", {}).get(
                "observation_generation", {}
            ),
            "memory_injection_policy": profile.injection_policy,
        }
        or run_report.get("system_profile") != profile.to_dict()
        or run_report.get("results", {}).get("sha256") != file_sha256(args.results)
        or run_report.get("inputs", {}).get("assignment_manifest_sha256")
        != file_sha256(args.assignment_manifest)
    ):
        raise ValueError("E1 run report does not authenticate the supplied artifacts")

    violations: list[dict[str, str]] = []
    for sample_id, record in records_by_id.items():
        assignment = assignments[sample_id]
        conditions = record.get("conditions", {})
        if (
            record.get("assignment_manifest_sha256") != logical_manifest_hash
            or record.get("question_sha256") != assignment.question_sha256
            or record.get("logical_split") != assignment.logical_split
            or record.get("dataset_split") != assignment.dataset_split
            or record.get("evaluation_role") != manifest.get("evaluation_role")
            or record.get("system_profile") != profile.to_dict()
            or record.get("assigned") is not assignment.assigned
            or record.get("triggered") is not assignment.triggered
            or record.get("prefix_token_ids_sha256")
            != assignment.prefix_token_ids_sha256
            or record.get("retrieval_query") != assignment.retrieval_query
            or record.get("matched_memory")
            != (
                assignment.matched_memory.to_dict()
                if assignment.matched_memory is not None
                else None
            )
        ):
            violations.append({
                "sample_id": sample_id,
                "reason": "assignment_identity_mismatch",
            })
        if set(conditions) != set(E1_CONDITIONS):
            violations.append({
                "sample_id": sample_id,
                "reason": "condition_set_mismatch",
            })
            continue

        matched = conditions["matched"]
        vanilla = conditions["vanilla"]
        if assignment.assigned:
            choice = assignment.matched_memory
            assert choice is not None
            completion_ids = matched.get("completion_token_ids")
            frozen_prefix = list(
                assignment.prefix_token_ids[assignment.prompt_token_count :]
            )
            if (
                matched.get("memory_id") != choice.memory_id
                or matched.get("payload_hash") != choice.payload_hash
                or matched.get("side_kv_applied") is not True
                or not isinstance(completion_ids, list)
                or completion_ids[: len(frozen_prefix)] != frozen_prefix
                or not persistent_trace_valid(
                    trace=matched.get("memory_attention") or {},
                    assignment=assignment,
                    completion_ids=(
                        completion_ids if isinstance(completion_ids, list) else []
                    ),
                    profile=profile,
                )
            ):
                violations.append({
                    "sample_id": sample_id,
                    "reason": "matched_memory_mismatch",
                })
        elif (
            matched.get("side_kv_applied")
            or matched.get("completion_token_ids_sha256")
            != vanilla.get("completion_token_ids_sha256")
        ):
            violations.append({
                "sample_id": sample_id,
                "reason": "memory_changed_unassigned_sample",
            })

    ordered = [records_by_id[sample_id] for sample_id in sorted(records_by_id)]
    assigned = lambda record: bool(record.get("assigned"))
    primary = {
        "matched_vs_vanilla_accuracy": paired(
            ordered,
            "matched",
            "vanilla",
            "final_reward",
            predicate=assigned,
            seed=args.seed,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_vanilla_diagnostic_answer": paired(
            ordered,
            "matched",
            "vanilla",
            "diagnostic_answer_correct",
            predicate=assigned,
            seed=args.seed + 1,
            resamples=args.bootstrap_resamples,
        ),
        "matched_vs_vanilla_format": paired(
            ordered,
            "matched",
            "vanilla",
            "format_valid",
            predicate=assigned,
            seed=args.seed + 2,
            resamples=args.bootstrap_resamples,
        ),
    }
    intention_to_treat = {
        metric: paired(
            ordered,
            "matched",
            "vanilla",
            field,
            predicate=None,
            seed=args.seed + offset,
            resamples=args.bootstrap_resamples,
        )
        for metric, field, offset in (
            ("accuracy", "final_reward", 3),
            ("diagnostic_answer", "diagnostic_answer_correct", 4),
            ("format", "format_valid", 5),
        )
    }
    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in E1_CONDITIONS:
        rows = [record["conditions"][condition] for record in ordered]
        condition_summaries[condition] = {
            "sample_count": len(rows),
            "accuracy": mean([float(row["final_reward"]) for row in rows]),
            "format_accuracy": mean([
                float(bool(row["format_valid"])) for row in rows
            ]),
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
            "mean_generation_length": mean([
                float(row["generation_length"]) for row in rows
            ]),
            "side_kv_applied_count": sum(
                bool(row["side_kv_applied"]) for row in rows
            ),
        }

    matched_outputs = [
        record["conditions"]["matched"]
        for record in ordered
        if record.get("assigned")
    ]
    mechanism = {
        "persistent_from_trigger_through_eos": True,
        "runtime_profile": profile.to_dict(),
        "mean_memory_attention_mass": mean([
            float(row["memory_attention"]["mean_memory_attention_mass"])
            for row in matched_outputs
        ]),
        "mean_first_step_logits_kl": mean([
            float(row["first_step_logits_kl_baseline_to_memory"])
            for row in matched_outputs
        ]),
        "mean_trace_count": mean([
            float(row["memory_attention"]["trace_count"])
            for row in matched_outputs
        ]),
        "baseline_first_token_match_count": sum(
            row["memory_attention"][
                "baseline_first_token_matches_observation"
            ]
            for row in matched_outputs
        ),
        "mean_bm25_top1_score": mean([
            float(record["matched_memory"]["retrieval_score"])
            for record in ordered
            if record.get("assigned")
        ]),
    }

    primary_accuracy = primary["matched_vs_vanilla_accuracy"]
    interval = primary_accuracy.get("bootstrap_95_ci")
    accuracy_positive = bool(
        primary_accuracy.get("paired_sample_count", 0) >= args.min_primary_pairs
        and isinstance(interval, list)
        and interval[0] > 0
    )
    format_effect = intention_to_treat["format"].get(
        "mean_treatment_minus_control"
    )
    acceptance = {
        "assignment_and_runtime_integrity": not violations,
        "vanilla_matches_internal_observation": all(
            record.get("vanilla_matches_internal_observation")
            for record in ordered
        ),
        "minimum_primary_pair_count": (
            primary_accuracy["paired_sample_count"] >= args.min_primary_pairs
        ),
        "matched_accuracy_above_vanilla": accuracy_positive,
        "matched_format_not_below_vanilla": (
            format_effect is not None and format_effect >= 0
        ),
    }
    runtime_integrity = (
        acceptance["assignment_and_runtime_integrity"]
        and acceptance["vanilla_matches_internal_observation"]
    )
    output = {
        "schema_version": E1_SUMMARY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if runtime_integrity else "invalid",
        "formal_e1_passed": False,
        "formal_task_claim": False,
        "frozen_effect_criteria_passed": all(acceptance.values()),
        "component_diagnostic": {
            "full_system_implemented": True,
            "gate_retrieval_persistent_side_kv_pipeline_executed": True,
            "runtime_integrity_passed": runtime_integrity,
            "implementation_completeness_is_not_task_effectiveness": True,
        },
        "system_profile": profile.to_dict(),
        "logical_split": manifest["logical_split"],
        "dataset_split": manifest["dataset_split"],
        "evaluation_role": manifest["evaluation_role"],
        "sample_count": len(ordered),
        "triggered_count": sum(bool(record.get("triggered")) for record in ordered),
        "assigned_count": sum(bool(record.get("assigned")) for record in ordered),
        "conditions": condition_summaries,
        "primary_assigned_subset": primary,
        "intention_to_treat": intention_to_treat,
        "mechanism_diagnostics": mechanism,
        "integrity_violations": violations,
        "acceptance": acceptance,
        "interpretation_limit": (
            "Matched-vs-vanilla estimates the end-to-end effect of gate, retrieval, "
            "memory content, and persistent side-KV activation together; it does not "
            "identify any component's isolated contribution."
        ),
        "criterion": {
            "bootstrap_resamples": args.bootstrap_resamples,
            "minimum_primary_pairs": args.min_primary_pairs,
            "accuracy": (
                "assigned-subset matched-vs-vanilla paired 95% bootstrap CI "
                "lower bound above zero"
            ),
            "format": "matched format accuracy must not be below vanilla",
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
