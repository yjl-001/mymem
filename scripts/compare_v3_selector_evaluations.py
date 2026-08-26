#!/usr/bin/env python3
"""Compare matched disabled-selector and margin-selector V3 evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.e1 import paired_binary_effect
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3_selector import numeric_summary


COMPARISON_SCHEMA = "experience-memory-v3-selector-comparison-v1"
EVALUATION_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
EVALUATION_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--margin-results", type=Path, required=True)
    parser.add_argument("--margin-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
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


def load_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != EVALUATION_PROFILE_SCHEMA
        or value.get("profile_sha256") != evaluation_profile_sha256(value)
    ):
        raise ValueError(f"Invalid V3 evaluation profile: {path}")
    return value


def load_rows(path: Path, *, profile_sha256: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            actual_hash = canonical_json_sha256({
                key: item
                for key, item in row.items()
                if key not in {"created_at", "row_sha256"}
            })
            if (
                row.get("schema_version") != EVALUATION_ROW_SCHEMA
                or row.get("profile_sha256") != profile_sha256
                or row.get("row_sha256") != actual_hash
                or not sample_id
                or sample_id in rows
            ):
                raise ValueError(f"Invalid V3 result row at {path}:{line_number}")
            rows[sample_id] = row
    return rows


def metric_map(
    rows: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, bool]:
    return {
        sample_id: bool(row["conditions"]["v3"][field])
        for sample_id, row in rows.items()
    }


def mechanism_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        row["conditions"]["v3"]["online_diagnostics"]
        for row in rows.values()
    ]
    attempts = [int(item["retrieval_attempt_count"]) for item in diagnostics]
    return {
        "sample_count": len(rows),
        "questions_with_attempt": sum(value > 0 for value in attempts),
        "retrieval_attempt_count": sum(attempts),
        "activation_count": sum(int(item["activation_count"]) for item in diagnostics),
        "replacement_count": sum(int(item["replacement_count"]) for item in diagnostics),
        "duplicate_count": sum(int(item["duplicate_count"]) for item in diagnostics),
        "abstain_count": sum(int(item["abstain_count"]) for item in diagnostics),
        "rearm_count": sum(int(item["rearm_count"]) for item in diagnostics),
        "memory_attention_step_count": sum(
            int(item["memory_attention_step_count"]) for item in diagnostics
        ),
    }


def markdown_report(value: Mapping[str, Any]) -> str:
    strict = value["paired_margin_minus_baseline"]["strict"]
    formatting = value["paired_margin_minus_baseline"]["format"]
    baseline = value["mechanism"]["baseline"]
    margin = value["mechanism"]["margin"]
    lines = [
        "# MemGen V3.1 selector comparison",
        "",
        f"- Integrity passed: `{str(value['integrity']['passed']).lower()}`",
        f"- Logical split: `{value['logical_split']}`",
        f"- Samples: {strict['paired_sample_count']}",
        f"- Frozen minimum margin: `{value['margin_selector']['minimum_top1_top2_margin']}`",
        "",
        "## Paired margin-selector minus disabled-selector",
        "",
        "| Metric | Baseline | Margin | Delta | Improved | Harmed | McNemar p |",
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
        f"- Mean generated-token delta: "
        f"{value['paired_token_delta_margin_minus_baseline']['mean']}",
        f"- Exact V3 completion matches: "
        f"{value['completion_parity']['v3_exact_match_count']} / "
        f"{strict['paired_sample_count']}",
        "",
        "## Mechanism",
        "",
        "| Condition | Attempts | Activations | Replacements | Duplicates | Abstains | Attention steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Disabled | {baseline['retrieval_attempt_count']} | "
        f"{baseline['activation_count']} | {baseline['replacement_count']} | "
        f"{baseline['duplicate_count']} | {baseline['abstain_count']} | "
        f"{baseline['memory_attention_step_count']} |",
        f"| Margin | {margin['retrieval_attempt_count']} | "
        f"{margin['activation_count']} | {margin['replacement_count']} | "
        f"{margin['duplicate_count']} | {margin['abstain_count']} | "
        f"{margin['memory_attention_step_count']} |",
        "",
        "This comparison is valid only for the matched logical split in these artifacts; it is not an independent final-test confirmation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    baseline_profile = load_profile(args.baseline_profile)
    margin_profile = load_profile(args.margin_profile)
    baseline_system = baseline_profile.get("system_profile", {})
    margin_system = margin_profile.get("system_profile", {})
    if baseline_system.get("retrieval_abstention_policy") != "disabled":
        raise ValueError("Baseline evaluation does not disable abstention")
    if margin_system.get("retrieval_abstention_policy") != "top1_top2_margin":
        raise ValueError("Treatment evaluation is not margin-qualified")
    baseline_system_core = dict(baseline_system)
    margin_system_core = dict(margin_system)
    for value in (baseline_system_core, margin_system_core):
        value.pop("retrieval_abstention_policy", None)
        value.pop("retrieval_min_top1_top2_margin", None)
    if baseline_system_core != margin_system_core:
        raise ValueError("Selector comparison changes non-selector system fields")
    selector_calibration = margin_profile.get("selector_calibration") or {}
    calibrated_threshold = selector_calibration.get("calibration", {}).get(
        "minimum_top1_top2_margin"
    )
    if (
        not selector_calibration.get("artifact_sha256")
        or selector_calibration.get("task_accuracy_used") is not False
        or selector_calibration.get("answer_or_reward_used") is not False
        or calibrated_threshold
        != margin_system.get("retrieval_min_top1_top2_margin")
    ):
        raise ValueError("Margin run is not bound to its answer-blind calibration")
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
        baseline_profile.get(field) != margin_profile.get(field)
        for field in comparable_profile_fields
    ):
        raise ValueError("Selector evaluations do not cover the same data contract")
    comparable_input_fields = (
        "split_manifest_sha256",
        "memory_records_sha256",
        "retrieval_key_manifest_sha256",
        "side_kv_manifest_sha256",
        "v3_offline_report_sha256",
        "e0_final_report_sha256",
        "risk_artifact_sha256",
    )
    if any(
        baseline_profile.get("inputs", {}).get(field)
        != margin_profile.get("inputs", {}).get(field)
        for field in comparable_input_fields
    ):
        raise ValueError("Selector evaluations use different frozen inputs")

    baseline_rows = load_rows(
        args.baseline_results,
        profile_sha256=str(baseline_profile["profile_sha256"]),
    )
    margin_rows = load_rows(
        args.margin_results,
        profile_sha256=str(margin_profile["profile_sha256"]),
    )
    expected_count = int(baseline_profile["selected_sample_count"])
    same_ids = set(baseline_rows) == set(margin_rows)
    complete = len(baseline_rows) == len(margin_rows) == expected_count
    if not same_ids or not complete:
        raise ValueError("Selector evaluations are incomplete or use different samples")
    vanilla_mismatches = []
    v3_exact_matches = 0
    token_deltas = []
    for sample_id in sorted(baseline_rows):
        baseline = baseline_rows[sample_id]["conditions"]
        margin = margin_rows[sample_id]["conditions"]
        if (
            baseline["vanilla"]["completion_token_ids_sha256"]
            != margin["vanilla"]["completion_token_ids_sha256"]
            or baseline["vanilla"]["strict_correct"]
            != margin["vanilla"]["strict_correct"]
            or baseline["vanilla"]["format_correct"]
            != margin["vanilla"]["format_correct"]
        ):
            vanilla_mismatches.append(sample_id)
        if (
            baseline["v3"]["completion_token_ids_sha256"]
            == margin["v3"]["completion_token_ids_sha256"]
        ):
            v3_exact_matches += 1
        token_deltas.append(
            int(margin["v3"]["generated_token_count"])
            - int(baseline["v3"]["generated_token_count"])
        )
    if vanilla_mismatches:
        raise ValueError("Vanilla generations differ across selector evaluations")
    integrity_passed = True
    report = {
        "schema_version": COMPARISON_SCHEMA,
        "created_at": utc_now(),
        "status": "completed",
        "implementation": {
            "files_sha256": {
                "memgen/experience/e1.py": file_sha256(
                    PROJECT_ROOT / "memgen/experience/e1.py"
                ),
                "scripts/compare_v3_selector_evaluations.py": file_sha256(
                    PROJECT_ROOT
                    / "scripts/compare_v3_selector_evaluations.py"
                ),
            },
        },
        "logical_split": baseline_profile["logical_split"],
        "integrity": {
            "passed": integrity_passed,
            "same_sample_ids": same_ids,
            "both_runs_complete": complete,
            "vanilla_exact_match": not vanilla_mismatches,
            "vanilla_mismatch_examples": vanilla_mismatches[:20],
        },
        "margin_selector": {
            "minimum_top1_top2_margin": margin_system[
                "retrieval_min_top1_top2_margin"
            ],
            "selector_calibration": margin_profile.get(
                "selector_calibration"
            ),
        },
        "paired_margin_minus_baseline": {
            "strict": paired_binary_effect(
                metric_map(margin_rows, "strict_correct"),
                metric_map(baseline_rows, "strict_correct"),
                seed=args.seed,
                resamples=args.bootstrap_resamples,
            ),
            "format": paired_binary_effect(
                metric_map(margin_rows, "format_correct"),
                metric_map(baseline_rows, "format_correct"),
                seed=args.seed + 1,
                resamples=args.bootstrap_resamples,
            ),
        },
        "paired_token_delta_margin_minus_baseline": numeric_summary(
            [float(value) for value in token_deltas]
        ),
        "completion_parity": {
            "v3_exact_match_count": v3_exact_matches,
            "v3_exact_mismatch_count": expected_count - v3_exact_matches,
        },
        "mechanism": {
            "baseline": mechanism_summary(baseline_rows),
            "margin": mechanism_summary(margin_rows),
        },
        "inputs": {
            "baseline_results_sha256": file_sha256(args.baseline_results),
            "baseline_profile_sha256": file_sha256(args.baseline_profile),
            "margin_results_sha256": file_sha256(args.margin_results),
            "margin_profile_sha256": file_sha256(args.margin_profile),
        },
        "interpretation": (
            "matched_selector_comparison_not_independent_final_confirmation"
        ),
    }
    report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in report.items() if key != "created_at"
    })
    write_json_atomic(args.output, report)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        f"[v3.1-compare] samples={expected_count} "
        f"integrity={integrity_passed} "
        f"strict_delta={report['paired_margin_minus_baseline']['strict']['mean_treatment_minus_control']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
