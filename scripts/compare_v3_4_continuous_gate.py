#!/usr/bin/env python3
"""Compare matched V3.4 continuous joint-gate and V3.1 dev runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_v3_selector_evaluations import (
    load_profile,
    load_rows,
    mechanism_summary,
    metric_map,
    write_json_atomic,
)
from memgen.experience.e1 import paired_binary_effect
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3 import (
    V34_QUERY_POOLING_CURRENT_TOKEN,
    V34_SYSTEM_PROFILE_SCHEMA,
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_SYSTEM_PROFILE_SCHEMA,
)
from memgen.experience.v3_selector import (
    load_margin_selector_calibration,
    numeric_summary,
)


COMPARISON_SCHEMA = "experience-memory-v3.4-continuous-gate-comparison-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31-results", type=Path, required=True)
    parser.add_argument("--v31-profile", type=Path, required=True)
    parser.add_argument("--v31-calibration", type=Path, required=True)
    parser.add_argument("--v34-results", type=Path, required=True)
    parser.add_argument("--v34-profile", type=Path, required=True)
    parser.add_argument("--v34-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_condition(
    *,
    profile: Mapping[str, Any],
    calibration_path: Path,
    expected_version: str,
) -> dict[str, Any]:
    system = profile.get("system_profile", {})
    expected = (
        {
            "schema_version": V3_SYSTEM_PROFILE_SCHEMA,
            "query_pooling": V3_QUERY_POOLING_BOUNDARY_LAST,
            "risk_role": "diagnostic_only",
            "boundary_policy": "pre_answer_comma_period_newline",
        }
        if expected_version == "v3"
        else {
            "schema_version": V34_SYSTEM_PROFILE_SCHEMA,
            "query_pooling": V34_QUERY_POOLING_CURRENT_TOKEN,
            "risk_role": "online_joint_control",
            "boundary_policy": "none_pre_answer_every_generated_token",
        }
    )
    if (
        profile.get("system_version", "v3") != expected_version
        or profile.get("logical_split") != "dev-test"
        or system.get("retrieval_abstention_policy")
        != "top1_top2_margin"
        or any(system.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(f"Unexpected {expected_version} comparison condition")
    calibration = load_margin_selector_calibration(calibration_path)
    source = calibration.get("source", {})
    threshold = calibration.get("calibration", {}).get(
        "minimum_top1_top2_margin"
    )
    embedded = profile.get("selector_calibration") or {}
    if (
        source.get("query_pooling", V3_QUERY_POOLING_BOUNDARY_LAST)
        != expected["query_pooling"]
        or source.get("retrieval_key_manifest_sha256")
        != profile.get("inputs", {}).get("retrieval_key_manifest_sha256")
        or threshold != system.get("retrieval_min_top1_top2_margin")
        or embedded.get("artifact_sha256")
        != calibration.get("artifact_sha256")
        or profile.get("inputs", {}).get("selector_calibration_sha256")
        != file_sha256(calibration_path)
    ):
        raise ValueError("Comparison run is not bound to its selector calibration")
    if expected_version == "v3.4" and (
        source.get("system_version") != "v3.4"
        or source.get("risk_artifact_sha256")
        != profile.get("inputs", {}).get("risk_artifact_sha256")
    ):
        raise ValueError("V3.4 calibration is not bound to its token-risk gate")
    return calibration


def trace_stratum_summary(
    rows: Mapping[str, Mapping[str, Any]], *, conditioned: bool
) -> dict[str, Any]:
    traces = [
        trace
        for row in rows.values()
        for trace in row["conditions"]["v3"]["runtime_trace"]["boundary_traces"]
        if bool(trace.get("active_memory_conditioned")) is conditioned
    ]
    entropies = [float(trace["entropy"]) for trace in traces]
    risks = [float(trace["persistence_risk_score"]) for trace in traces]
    vocabulary = [float(trace["vocabulary_entropy"]) for trace in traces]
    margins = [float(trace["top1_top2_logit_margin"]) for trace in traces]
    return {
        "observation_count": len(traces),
        "attention_entropy": numeric_summary(entropies) if entropies else None,
        "persistence_risk": numeric_summary(risks) if risks else None,
        "vocabulary_entropy": numeric_summary(vocabulary) if vocabulary else None,
        "top1_top2_logit_margin": numeric_summary(margins) if margins else None,
        "joint_trigger_qualified_count": sum(
            trace.get("joint_trigger_qualified") is True for trace in traces
        ),
    }


def completion_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in row["conditions"]["v3"]["completion_token_ids"]
    )


def markdown(value: Mapping[str, Any]) -> str:
    strict = value["paired_v34_minus_v31"]["strict"]
    formatting = value["paired_v34_minus_v31"]["format"]
    v31 = value["mechanism"]["v31_boundary_gate"]
    v34 = value["mechanism"]["v34_continuous_joint_gate"]
    return "\n".join([
        "# MemGen V3.4 continuous joint-gate matched dev comparison",
        "",
        f"- Integrity passed: `{str(value['integrity']['passed']).lower()}`",
        f"- Samples: {strict['paired_sample_count']}",
        "",
        "## Paired V3.4 minus V3.1 task results",
        "",
        "| Metric | V3.1 | V3.4 | Delta | Improved | Harmed | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Strict | {strict['control_accuracy']} | {strict['treatment_accuracy']} | {strict['mean_treatment_minus_control']} | {strict['treatment_correct_control_wrong']} | {strict['treatment_wrong_control_correct']} | {strict['mcnemar_exact_two_sided_p']} |",
        f"| Format | {formatting['control_accuracy']} | {formatting['treatment_accuracy']} | {formatting['mean_treatment_minus_control']} | {formatting['treatment_correct_control_wrong']} | {formatting['treatment_wrong_control_correct']} | {formatting['mcnemar_exact_two_sided_p']} |",
        "",
        f"- Strict bootstrap 95% CI: `{strict['bootstrap_95_ci']}`",
        f"- Mean generated-token delta: {value['generated_token_delta']['mean']}",
        f"- Exact V3 completion matches: {value['completion_parity']['exact_match_count']} / {strict['paired_sample_count']}",
        "",
        "## Online mechanism",
        "",
        "| Condition | Attempts | Activations | Replacements | Duplicates | Abstains | Re-arms | Attention steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| V3.1 | {v31['retrieval_attempt_count']} | {v31['activation_count']} | {v31['replacement_count']} | {v31['duplicate_count']} | {v31['abstain_count']} | {v31['rearm_count']} | {v31['memory_attention_step_count']} |",
        f"| V3.4 | {v34['retrieval_attempt_count']} | {v34['activation_count']} | {v34['replacement_count']} | {v34['duplicate_count']} | {v34['abstain_count']} | {v34['rearm_count']} | {v34['memory_attention_step_count']} |",
        "",
        "## V3.4 treated-state risk drift",
        "",
        f"- Native observations: {value['v34_gate_strata']['native']['observation_count']}",
        f"- Memory-conditioned observations: {value['v34_gate_strata']['memory_conditioned']['observation_count']}",
        "",
        "The native/conditioned risk strata are descriptive because memory changes the later hidden-state distribution.",
        "",
    ])


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    v31_profile = load_profile(args.v31_profile)
    v34_profile = load_profile(args.v34_profile)
    v31_calibration = validate_condition(
        profile=v31_profile,
        calibration_path=args.v31_calibration,
        expected_version="v3",
    )
    v34_calibration = validate_condition(
        profile=v34_profile,
        calibration_path=args.v34_calibration,
        expected_version="v3.4",
    )
    comparable = (
        "logical_split",
        "dataset_split",
        "dataset_revision",
        "selected_sample_count",
        "selected_sample_ids_sha256",
        "reasoner",
        "prompt_contract",
        "alignment",
    )
    if any(v31_profile.get(key) != v34_profile.get(key) for key in comparable):
        raise ValueError("V3.1 and V3.4 evaluations are not matched")
    input_fields = (
        "split_manifest_sha256",
        "memory_records_sha256",
        "retrieval_key_manifest_sha256",
        "side_kv_manifest_sha256",
        "v3_offline_report_sha256",
        "e0_final_report_sha256",
    )
    if any(
        v31_profile.get("inputs", {}).get(key)
        != v34_profile.get("inputs", {}).get(key)
        for key in input_fields
    ):
        raise ValueError("V3.1 and V3.4 use different memory/data inputs")
    v31_rows = load_rows(
        args.v31_results, profile_sha256=str(v31_profile["profile_sha256"])
    )
    v34_rows = load_rows(
        args.v34_results, profile_sha256=str(v34_profile["profile_sha256"])
    )
    if set(v31_rows) != set(v34_rows):
        raise ValueError("V3.1 and V3.4 sample IDs differ")
    strict = paired_binary_effect(
        metric_map(v34_rows, "strict_correct"),
        metric_map(v31_rows, "strict_correct"),
        seed=args.seed,
        resamples=args.bootstrap_resamples,
    )
    formatting = paired_binary_effect(
        metric_map(v34_rows, "format_correct"),
        metric_map(v31_rows, "format_correct"),
        seed=args.seed + 1,
        resamples=args.bootstrap_resamples,
    )
    sample_ids = sorted(v31_rows)
    token_deltas = [
        int(v34_rows[sample_id]["conditions"]["v3"]["generated_token_count"])
        - int(v31_rows[sample_id]["conditions"]["v3"]["generated_token_count"])
        for sample_id in sample_ids
    ]
    exact_matches = sum(
        completion_ids(v31_rows[sample_id])
        == completion_ids(v34_rows[sample_id])
        for sample_id in sample_ids
    )
    value: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "logical_split": "dev-test",
        "integrity": {
            "passed": True,
            "profiles_authenticated": True,
            "rows_authenticated": True,
            "sample_ids_matched": True,
            "shared_data_and_memory_inputs": True,
        },
        "paired_v34_minus_v31": {
            "strict": strict,
            "format": formatting,
        },
        "generated_token_delta": numeric_summary(token_deltas),
        "completion_parity": {
            "exact_match_count": exact_matches,
            "mismatch_count": len(sample_ids) - exact_matches,
        },
        "mechanism": {
            "v31_boundary_gate": mechanism_summary(v31_rows),
            "v34_continuous_joint_gate": mechanism_summary(v34_rows),
        },
        "v34_gate_strata": {
            "native": trace_stratum_summary(v34_rows, conditioned=False),
            "memory_conditioned": trace_stratum_summary(
                v34_rows, conditioned=True
            ),
        },
        "calibrations": {
            "v31_artifact_sha256": v31_calibration.get("artifact_sha256"),
            "v34_artifact_sha256": v34_calibration.get("artifact_sha256"),
        },
        "inputs": {
            "v31_results_sha256": file_sha256(args.v31_results),
            "v31_profile_sha256": file_sha256(args.v31_profile),
            "v34_results_sha256": file_sha256(args.v34_results),
            "v34_profile_sha256": file_sha256(args.v34_profile),
        },
    }
    value["report_sha256"] = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "report_sha256"}
    })
    write_json_atomic(args.output, value)
    args.output.with_suffix(".md").write_text(markdown(value), encoding="utf-8")
    print(
        f"[v3.4-compare] samples={len(sample_ids)} "
        f"strict_delta={strict['mean_treatment_minus_control']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
