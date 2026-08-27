#!/usr/bin/env python3
"""Compare matched V3.1 boundary-last and V3.3 pre-boundary dev runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_v3_selector_evaluations import (
    load_profile,
    load_rows,
    mechanism_summary,
    metric_map,
    write_json_atomic,
)
from memgen.experience.e1 import paired_binary_effect
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import file_sha256, iter_jsonl
from memgen.experience.v3 import (
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_PRE_BOUNDARY,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_selector import (
    load_margin_selector_calibration,
    numeric_summary,
    selection_concentration,
    selector_calibration_query_pooling,
)


COMPARISON_SCHEMA = "experience-memory-v3-query-pooling-comparison-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31-results", type=Path, required=True)
    parser.add_argument("--v31-profile", type=Path, required=True)
    parser.add_argument("--v31-calibration", type=Path, required=True)
    parser.add_argument("--v33-results", type=Path, required=True)
    parser.add_argument("--v33-profile", type=Path, required=True)
    parser.add_argument("--v33-calibration", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_condition(
    *,
    profile: Mapping[str, Any],
    calibration_path: Path,
    expected_pooling: str,
) -> dict[str, Any]:
    system = profile.get("system_profile", {})
    actual_pooling = str(
        system.get("query_pooling", V3_QUERY_POOLING_BOUNDARY_LAST)
    )
    if (
        system.get("retrieval_abstention_policy") != "top1_top2_margin"
        or system.get(
            "retrieval_embedding_transform",
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
        )
        != V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
        or actual_pooling != expected_pooling
    ):
        raise ValueError("Query-pooling comparison received a wrong condition")
    calibration = load_margin_selector_calibration(calibration_path)
    threshold = calibration.get("calibration", {}).get(
        "minimum_top1_top2_margin"
    )
    embedded = profile.get("selector_calibration") or {}
    if (
        selector_calibration_query_pooling(calibration) != expected_pooling
        or calibration.get("source", {}).get(
            "retrieval_embedding_transform",
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
        )
        != V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
        or calibration.get("source", {}).get(
            "retrieval_key_manifest_sha256"
        )
        != profile.get("inputs", {}).get("retrieval_key_manifest_sha256")
        or threshold != system.get("retrieval_min_top1_top2_margin")
        or embedded.get("artifact_sha256")
        != calibration.get("artifact_sha256")
        or profile.get("inputs", {}).get("selector_calibration_sha256")
        != file_sha256(calibration_path)
    ):
        raise ValueError(
            "Evaluation is not bound to its pooling-specific calibration"
        )
    return calibration


def normalized_system(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(profile.get("system_profile", {}))
    value.setdefault("query_pooling", V3_QUERY_POOLING_BOUNDARY_LAST)
    value.setdefault(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    return value


def calibration_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_sha256": value.get("artifact_sha256"),
        "source": value.get("source"),
        "calibration": value.get("calibration"),
        "first_attempt_selection_concentration": value.get(
            "first_attempt_selection_concentration"
        ),
    }


def retrieval_summary(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    complete_memory_ids: Sequence[str],
) -> dict[str, Any]:
    all_top1: list[str] = []
    all_selected: list[str] = []
    all_margins: list[float] = []
    first_top1: list[str] = []
    first_selected: list[str] = []
    first_margins: list[float] = []
    boundary_groups: defaultdict[
        tuple[int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for sample_id, row in rows.items():
        condition = row["conditions"]["v3"]
        attempts = condition["runtime_trace"]["retrieval_attempts"]
        for position, attempt in enumerate(attempts):
            decision = attempt["retrieval_decision"]
            query = decision["query"]
            hits = list(decision.get("hits", []))
            if len(hits) < 2:
                raise ValueError("Query-pooling comparison requires top-2 traces")
            top1_id = str(hits[0]["memory_id"])
            margin = float(query["top1_top2_margin"])
            all_top1.append(top1_id)
            all_margins.append(margin)
            if attempt.get("selected_memory_id") is not None:
                all_selected.append(str(attempt["selected_memory_id"]))
            if position == 0:
                first_top1.append(top1_id)
                first_margins.append(margin)
                if attempt.get("selected_memory_id") is not None:
                    first_selected.append(str(attempt["selected_memory_id"]))
                boundary_groups[(
                    int(attempt["boundary_token_id"]),
                    str(attempt.get("boundary_token_text", "")),
                )].append({
                    "sample_id": sample_id,
                    "top1_memory_id": top1_id,
                    "selected_memory_id": attempt.get("selected_memory_id"),
                    "margin": margin,
                    "outcome": str(attempt["outcome"]),
                    "strict_correct": bool(condition["strict_correct"]),
                    "format_correct": bool(condition["format_correct"]),
                })

    def concentration(memory_ids: Sequence[str]) -> dict[str, Any]:
        return selection_concentration(
            memory_ids, complete_memory_ids=complete_memory_ids
        )

    strata = []
    for (token_id, token_text), values in boundary_groups.items():
        selected = [
            str(value["selected_memory_id"])
            for value in values
            if value["selected_memory_id"] is not None
        ]
        outcomes = Counter(str(value["outcome"]) for value in values)
        strata.append({
            "boundary_token_id": token_id,
            "boundary_token_text": token_text,
            "sample_count": len(values),
            "top1_selection_concentration": concentration([
                str(value["top1_memory_id"]) for value in values
            ]),
            "post_margin_selection_concentration": concentration(selected),
            "margin": numeric_summary([
                float(value["margin"]) for value in values
            ]),
            "activation_or_replacement_count": (
                outcomes["activated"] + outcomes["replaced"]
            ),
            "duplicate_count": outcomes["duplicate"],
            "abstain_count": outcomes["abstained"],
            "strict_accuracy": sum(
                bool(value["strict_correct"]) for value in values
            ) / len(values),
            "format_accuracy": sum(
                bool(value["format_correct"]) for value in values
            ) / len(values),
            "sample_ids": [str(value["sample_id"]) for value in values],
        })
    strata.sort(key=lambda item: (
        -int(item["sample_count"]), int(item["boundary_token_id"])
    ))
    return {
        "all_attempts": {
            "attempt_count": len(all_top1),
            "top1_selection_concentration": concentration(all_top1),
            "post_margin_selection_concentration": concentration(all_selected),
            "margin": numeric_summary(all_margins) if all_margins else None,
        },
        "first_attempts": {
            "attempt_count": len(first_top1),
            "top1_selection_concentration": concentration(first_top1),
            "post_margin_selection_concentration": concentration(first_selected),
            "margin": numeric_summary(first_margins) if first_margins else None,
            "boundary_strata": strata,
        },
    }


def markdown_report(value: Mapping[str, Any]) -> str:
    strict = value["paired_v33_minus_v31"]["strict"]
    formatting = value["paired_v33_minus_v31"]["format"]
    v31_mechanism = value["mechanism"]["v31_boundary_last"]
    v33_mechanism = value["mechanism"]["v33_pre_boundary"]
    v31_retrieval = value["retrieval"]["v31_boundary_last"]["first_attempts"]
    v33_retrieval = value["retrieval"]["v33_pre_boundary"]["first_attempts"]
    v31_concentration = v31_retrieval["top1_selection_concentration"]
    v33_concentration = v33_retrieval["top1_selection_concentration"]
    lines = [
        "# MemGen V3.3 pre-boundary matched dev comparison",
        "",
        f"- Integrity passed: `{str(value['integrity']['passed']).lower()}`",
        f"- Logical split: `{value['logical_split']}`",
        f"- Samples: {strict['paired_sample_count']}",
        "",
        "## Paired V3.3 minus V3.1 task results",
        "",
        "| Metric | V3.1 boundary-last | V3.3 pre-boundary | Delta | Improved | Harmed | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Strict | {strict['control_accuracy']} | {strict['treatment_accuracy']} | "
        f"{strict['mean_treatment_minus_control']} | "
        f"{strict['treatment_correct_control_wrong']} | "
        f"{strict['treatment_wrong_control_correct']} | "
        f"{strict['mcnemar_exact_two_sided_p']} |",
        f"| Format | {formatting['control_accuracy']} | {formatting['treatment_accuracy']} | "
        f"{formatting['mean_treatment_minus_control']} | "
        f"{formatting['treatment_correct_control_wrong']} | "
        f"{formatting['treatment_wrong_control_correct']} | "
        f"{formatting['mcnemar_exact_two_sided_p']} |",
        "",
        f"- Strict bootstrap 95% CI: `{strict['bootstrap_95_ci']}`",
        f"- Mean generated-token delta: {value['paired_token_delta_v33_minus_v31']['mean']}",
        f"- Exact V3 completion matches: {value['completion_parity']['exact_match_count']} / {strict['paired_sample_count']}",
        "",
        "## First-attempt retrieval geometry on dev",
        "",
        "| Condition | Threshold | Attempts | Top-1 share | Gini | Selected memories | Median margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V3.1 boundary-last | {value['calibration']['v31_boundary_last']['calibration']['minimum_top1_top2_margin']} | "
        f"{v31_retrieval['attempt_count']} | {v31_concentration['top1_share']} | "
        f"{v31_concentration['gini']} | {v31_concentration['selected_memory_count']} | "
        f"{v31_retrieval['margin']['median'] if v31_retrieval['margin'] else None} |",
        f"| V3.3 pre-boundary | {value['calibration']['v33_pre_boundary']['calibration']['minimum_top1_top2_margin']} | "
        f"{v33_retrieval['attempt_count']} | {v33_concentration['top1_share']} | "
        f"{v33_concentration['gini']} | {v33_concentration['selected_memory_count']} | "
        f"{v33_retrieval['margin']['median'] if v33_retrieval['margin'] else None} |",
        "",
        "## Online mechanism",
        "",
        "| Condition | Attempts | Activations | Replacements | Duplicates | Abstains | Re-arms | Attention steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| V3.1 | {v31_mechanism['retrieval_attempt_count']} | {v31_mechanism['activation_count']} | "
        f"{v31_mechanism['replacement_count']} | {v31_mechanism['duplicate_count']} | "
        f"{v31_mechanism['abstain_count']} | {v31_mechanism['rearm_count']} | "
        f"{v31_mechanism['memory_attention_step_count']} |",
        f"| V3.3 | {v33_mechanism['retrieval_attempt_count']} | {v33_mechanism['activation_count']} | "
        f"{v33_mechanism['replacement_count']} | {v33_mechanism['duplicate_count']} | "
        f"{v33_mechanism['abstain_count']} | {v33_mechanism['rearm_count']} | "
        f"{v33_mechanism['memory_attention_step_count']} |",
        "",
        "## V3.3 first-attempt boundary strata",
        "",
        "| Boundary | Samples | Top-1 share | Selected memories | Median margin | Strict accuracy | Abstains |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stratum in v33_retrieval["boundary_strata"]:
        concentration = stratum["top1_selection_concentration"]
        lines.append(
            f"| `{json.dumps(stratum['boundary_token_text'], ensure_ascii=False)}` "
            f"({stratum['boundary_token_id']}) | {stratum['sample_count']} | "
            f"{concentration['top1_share']} | {concentration['selected_memory_count']} | "
            f"{stratum['margin']['median']} | {stratum['strict_accuracy']} | "
            f"{stratum['abstain_count']} |"
        )
    lines.extend([
        "",
        "This is a matched dev-test comparison. Boundary-stratum accuracy is descriptive and not causal.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    v31_profile = load_profile(args.v31_profile)
    v33_profile = load_profile(args.v33_profile)
    v31_calibration = validate_condition(
        profile=v31_profile,
        calibration_path=args.v31_calibration,
        expected_pooling=V3_QUERY_POOLING_BOUNDARY_LAST,
    )
    v33_calibration = validate_condition(
        profile=v33_profile,
        calibration_path=args.v33_calibration,
        expected_pooling=V3_QUERY_POOLING_PRE_BOUNDARY,
    )
    v31_system = normalized_system(v31_profile)
    v33_system = normalized_system(v33_profile)
    for value in (v31_system, v33_system):
        value.pop("query_pooling", None)
        value.pop("retrieval_min_top1_top2_margin", None)
    if v31_system != v33_system:
        raise ValueError("V3.1 and V3.3 change fields beyond query pooling")

    comparable_profile_fields = (
        "logical_split",
        "dataset_split",
        "dataset_revision",
        "selected_sample_count",
        "selected_sample_ids_sha256",
        "reasoner",
        "prompt_contract",
        "generation",
        "hysteresis_gate",
        "alignment",
    )
    if any(
        v31_profile.get(field) != v33_profile.get(field)
        for field in comparable_profile_fields
    ):
        raise ValueError("V3.1 and V3.3 do not share one evaluation contract")
    comparable_inputs = (
        "split_manifest_sha256",
        "memory_records_sha256",
        "retrieval_key_manifest_sha256",
        "side_kv_manifest_sha256",
        "v3_offline_report_sha256",
        "e0_final_report_sha256",
        "risk_artifact_sha256",
    )
    if any(
        v31_profile.get("inputs", {}).get(field)
        != v33_profile.get("inputs", {}).get(field)
        for field in comparable_inputs
    ):
        raise ValueError("V3.1 and V3.3 use different frozen inputs")

    expected_memory_sha256 = v31_profile.get("inputs", {}).get(
        "memory_records_sha256"
    )
    if file_sha256(args.memory_records) != expected_memory_sha256:
        raise ValueError("Memory records differ from the matched evaluations")
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    complete_memory_ids = [record.memory_id for record in records]

    v31_rows = load_rows(
        args.v31_results,
        profile_sha256=str(v31_profile["profile_sha256"]),
    )
    v33_rows = load_rows(
        args.v33_results,
        profile_sha256=str(v33_profile["profile_sha256"]),
    )
    expected_count = int(v31_profile["selected_sample_count"])
    same_ids = set(v31_rows) == set(v33_rows)
    complete = len(v31_rows) == len(v33_rows) == expected_count
    if not same_ids or not complete:
        raise ValueError("V3.1 and V3.3 rows are incomplete or unmatched")

    vanilla_mismatches = []
    exact_matches = 0
    token_deltas = []
    for sample_id in sorted(v31_rows):
        v31 = v31_rows[sample_id]["conditions"]
        v33 = v33_rows[sample_id]["conditions"]
        if (
            v31["vanilla"]["completion_token_ids_sha256"]
            != v33["vanilla"]["completion_token_ids_sha256"]
            or v31["vanilla"]["strict_correct"]
            != v33["vanilla"]["strict_correct"]
            or v31["vanilla"]["format_correct"]
            != v33["vanilla"]["format_correct"]
        ):
            vanilla_mismatches.append(sample_id)
        exact_matches += int(
            v31["v3"]["completion_token_ids_sha256"]
            == v33["v3"]["completion_token_ids_sha256"]
        )
        token_deltas.append(
            int(v33["v3"]["generated_token_count"])
            - int(v31["v3"]["generated_token_count"])
        )
    if vanilla_mismatches:
        raise ValueError("Vanilla generations differ across V3.1 and V3.3")

    report = {
        "schema_version": COMPARISON_SCHEMA,
        "created_at": utc_now(),
        "status": "completed",
        "logical_split": v31_profile["logical_split"],
        "experimental_variable": (
            "query_pooling_with_pooling_specific_answer_blind_margin_calibration"
        ),
        "integrity": {
            "passed": True,
            "same_sample_ids": same_ids,
            "both_runs_complete": complete,
            "vanilla_exact_match": not vanilla_mismatches,
            "non_pooling_system_profile_equal": True,
            "frozen_inputs_equal": True,
        },
        "calibration": {
            "v31_boundary_last": calibration_summary(v31_calibration),
            "v33_pre_boundary": calibration_summary(v33_calibration),
        },
        "paired_v33_minus_v31": {
            "strict": paired_binary_effect(
                metric_map(v33_rows, "strict_correct"),
                metric_map(v31_rows, "strict_correct"),
                seed=args.seed,
                resamples=args.bootstrap_resamples,
            ),
            "format": paired_binary_effect(
                metric_map(v33_rows, "format_correct"),
                metric_map(v31_rows, "format_correct"),
                seed=args.seed + 1,
                resamples=args.bootstrap_resamples,
            ),
        },
        "paired_token_delta_v33_minus_v31": numeric_summary([
            float(value) for value in token_deltas
        ]),
        "completion_parity": {
            "exact_match_count": exact_matches,
            "exact_mismatch_count": expected_count - exact_matches,
        },
        "mechanism": {
            "v31_boundary_last": mechanism_summary(v31_rows),
            "v33_pre_boundary": mechanism_summary(v33_rows),
        },
        "retrieval": {
            "v31_boundary_last": retrieval_summary(
                v31_rows, complete_memory_ids=complete_memory_ids
            ),
            "v33_pre_boundary": retrieval_summary(
                v33_rows, complete_memory_ids=complete_memory_ids
            ),
        },
        "implementation": {
            "files_sha256": {
                "scripts/compare_v3_query_pooling.py": file_sha256(
                    PROJECT_ROOT / "scripts/compare_v3_query_pooling.py"
                ),
            },
        },
        "inputs": {
            "v31_results_sha256": file_sha256(args.v31_results),
            "v31_profile_sha256": file_sha256(args.v31_profile),
            "v31_calibration_sha256": file_sha256(args.v31_calibration),
            "v33_results_sha256": file_sha256(args.v33_results),
            "v33_profile_sha256": file_sha256(args.v33_profile),
            "v33_calibration_sha256": file_sha256(args.v33_calibration),
            "memory_records_sha256": file_sha256(args.memory_records),
        },
        "interpretation": (
            "matched_dev_query_pooling_comparison_not_independent_final_confirmation"
        ),
    }
    write_json_atomic(args.output, report)
    args.output.with_suffix(".md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(
        f"[v3.3-comparison] samples={expected_count} "
        f"strict_delta={report['paired_v33_minus_v31']['strict']['mean_treatment_minus_control']:.6f} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
