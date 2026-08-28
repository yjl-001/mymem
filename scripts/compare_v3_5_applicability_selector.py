#!/usr/bin/env python3
"""Compare a matched V3.5 applicability-aware run with a frozen baseline.

The primary intended comparison is V3.5 minus V3.4 on exploratory dev-test.
The same authenticated path can be used for the optional V3.1 comparison by
passing ``--baseline-version v3.1``.  Sample identity, evaluation contracts,
frozen common inputs, and every vanilla generation must match exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.e1 import paired_binary_effect
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3_5_selector import (
    V35_SELECTOR_POLICY,
    load_v35_selector_calibration,
)


COMPARISON_SCHEMA = (
    "experience-memory-v3.5-applicability-selector-comparison-v1"
)
EVALUATION_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
EVALUATION_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"
V35_EVALUATION_PROFILE_SCHEMA = (
    "experience-memory-v3.5-evaluation-profile-v1"
)
V35_EVALUATION_ROW_SCHEMA = "experience-memory-v3.5-evaluation-row-v1"
V3_SYSTEM_PROFILE_SCHEMA = "experience-memory-system-profile-v3"
V34_SYSTEM_PROFILE_SCHEMA = "experience-memory-system-profile-v3.4"
V35_SYSTEM_PROFILE_SCHEMA = "experience-memory-system-profile-v3.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-results", "--v34-results",
        dest="baseline_results", type=Path, required=True,
    )
    parser.add_argument(
        "--baseline-profile", "--v34-profile",
        dest="baseline_profile", type=Path, required=True,
    )
    parser.add_argument(
        "--baseline-version",
        choices=("v3.4", "v3.1"),
        default="v3.4",
    )
    parser.add_argument("--v35-results", type=Path, required=True)
    parser.add_argument("--v35-profile", type=Path, required=True)
    parser.add_argument(
        "--v35-selector-calibration", "--selector-calibration",
        dest="v35_selector_calibration", type=Path, required=True,
    )
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


def load_profile(path: Path, *, expected_schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != expected_schema
        or value.get("profile_sha256") != evaluation_profile_sha256(value)
    ):
        raise ValueError(f"Invalid evaluation profile: {path}")
    return value


def _row_sha256(row: Mapping[str, Any]) -> str:
    return canonical_json_sha256({
        key: item
        for key, item in row.items()
        if key not in {"created_at", "row_sha256"}
    })


def load_rows(
    path: Path, *, profile_sha256: str, expected_schema: str
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if (
                row.get("schema_version") != expected_schema
                or row.get("profile_sha256") != profile_sha256
                or row.get("row_sha256") != _row_sha256(row)
                or not sample_id
                or sample_id in rows
            ):
                raise ValueError(f"Invalid result row at {path}:{line_number}")
            for condition_name in ("vanilla", "v3"):
                condition = row.get("conditions", {}).get(condition_name, {})
                ids = [int(value) for value in condition.get(
                    "completion_token_ids", []
                )]
                if (
                    int(condition.get("generated_token_count", -1)) != len(ids)
                    or condition.get("completion_token_ids_sha256")
                    != canonical_json_sha256(ids)
                ):
                    raise ValueError(
                        f"Invalid completion hash/count at {path}:{line_number}"
                    )
            rows[sample_id] = row
    return rows


def _system_version(profile: Mapping[str, Any]) -> str:
    return str(profile.get("system_version", "v3"))


def validate_baseline_profile(
    profile: Mapping[str, Any], *, requested_version: str
) -> None:
    system = profile.get("system_profile", {})
    actual_version = _system_version(profile)
    if requested_version == "v3.4":
        expected_schema = V34_SYSTEM_PROFILE_SCHEMA
        versions = {"v3.4"}
        required = {
            "risk_role": "online_joint_control",
            "boundary_policy": "none_pre_answer_every_generated_token",
            "query_pooling": "current_generated_token",
        }
    else:
        expected_schema = V3_SYSTEM_PROFILE_SCHEMA
        versions = {"v3", "v3.1"}
        required = {
            "risk_role": "diagnostic_only",
            "boundary_policy": "pre_answer_comma_period_newline",
            "query_pooling": "last_valid_token",
        }
    if (
        actual_version not in versions
        or profile.get("logical_split") != "dev-test"
        or system.get("schema_version") != expected_schema
        or any(system.get(key) != value for key, value in required.items())
    ):
        raise ValueError(f"Unexpected {requested_version} baseline profile")


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def validate_v35_profile(
    profile: Mapping[str, Any], *, selector_path: Path
) -> dict[str, Any]:
    system = profile.get("system_profile", {})
    inputs = profile.get("inputs", {})
    selector_input_hashes = {
        name: inputs.get(name)
        for name in (
            "dual_key_manifest_sha256",
            "applicability_calibration_sha256",
            "risk_artifact_sha256",
        )
    }
    if (
        _system_version(profile) != "v3.5"
        or profile.get("logical_split") != "dev-test"
        or system.get("schema_version") != V35_SYSTEM_PROFILE_SCHEMA
        or system.get("selector_policy") != V35_SELECTOR_POLICY
        or system.get("risk_role") != "online_joint_control"
        or system.get("boundary_policy")
        != "none_pre_answer_every_generated_token"
        or system.get("query_pooling") != "current_generated_token"
        or system.get("abstain_policy")
        != "terminal_consume_attempt_clear_current_memory"
        or profile.get("calibration_trace_only") is not False
        or system.get("calibration_trace_only") is not False
        or profile.get("task_results_used_for_selector_decision") is not False
        or any(
            not isinstance(value, str) or not value
            for value in selector_input_hashes.values()
        )
    ):
        raise ValueError("Unexpected V3.5 applicability-aware dev profile")
    selector_file_hash = file_sha256(selector_path)
    if (
        profile.get("inputs", {}).get("selector_calibration_sha256")
        != selector_file_hash
    ):
        raise ValueError("V3.5 profile is not bound to its selector artifact")
    selector = load_v35_selector_calibration(
        selector_path,
        expected_input_hashes=selector_input_hashes,
    )
    calibration = selector.get("calibration", {})
    profile_k = _first_present(system, (
        "applicability_shortlist_k", "shortlist_k"
    ))
    profile_floor = _first_present(system, (
        "minimum_applicability_score", "applicability_score_floor"
    ))
    profile_margin = _first_present(system, (
        "minimum_dynamic_top1_top2_margin",
        "retrieval_min_top1_top2_margin",
    ))
    embedded = profile.get("selector_calibration") or {}
    if (
        selector.get("status") != "passed"
        or selector.get("task_accuracy_used") is not False
        or selector.get("answer_or_reward_used") is not False
        or not all(selector.get("requirements", {}).values())
        or int(profile_k) != int(calibration.get("shortlist_k", -1))
        or float(profile_floor)
        != float(calibration.get("minimum_applicability_score", math.nan))
        or float(profile_margin)
        != float(calibration.get(
            "minimum_dynamic_top1_top2_margin", math.nan
        ))
        or (
            embedded
            and embedded.get("artifact_sha256")
            != selector.get("artifact_sha256")
        )
    ):
        raise ValueError("V3.5 selector binding/answer-blind contract failed")
    return selector


def _required_equal(
    left: Mapping[str, Any], right: Mapping[str, Any], field: str
) -> bool:
    return (
        field in left
        and field in right
        and left.get(field) is not None
        and left.get(field) == right.get(field)
    )


def validate_profile_compatibility(
    baseline: Mapping[str, Any],
    v35: Mapping[str, Any],
    *,
    baseline_version: str,
) -> None:
    common_fields = (
        "logical_split",
        "dataset_split",
        "dataset_revision",
        "selected_sample_count",
        "selected_sample_ids_sha256",
        "reasoner",
        "prompt_contract",
    )
    if any(not _required_equal(baseline, v35, field) for field in common_fields):
        raise ValueError("Baseline and V3.5 profiles are not sample/model matched")
    baseline_generation = baseline.get("generation", {})
    v35_generation = v35.get("generation", {})
    generation_fields = ("max_new_tokens", "vanilla")
    if any(
        not _required_equal(
            baseline_generation, v35_generation, field
        )
        for field in generation_fields
    ):
        raise ValueError("Baseline and V3.5 decoding contracts differ")
    baseline_inputs = baseline.get("inputs", {})
    v35_inputs = v35.get("inputs", {})
    common_inputs = (
        "split_manifest_sha256",
        "memory_records_sha256",
        "side_kv_manifest_sha256",
        "e0_final_report_sha256",
        "retrieval_key_manifest_sha256",
        "v3_offline_report_sha256",
    )
    if any(
        not _required_equal(baseline_inputs, v35_inputs, field)
        for field in common_inputs
    ):
        raise ValueError("Baseline and V3.5 use different frozen inputs")
    for inputs in (baseline_inputs, v35_inputs):
        if not isinstance(inputs.get("risk_artifact_sha256"), str) or not inputs.get(
            "risk_artifact_sha256"
        ):
            raise ValueError("Comparison profile has no authenticated risk artifact")
    if (
        baseline_version == "v3.4"
        and baseline_inputs["risk_artifact_sha256"]
        != v35_inputs["risk_artifact_sha256"]
    ):
        raise ValueError("V3.4 and V3.5 use different token-risk artifacts")
    baseline_gate = baseline.get("hysteresis_gate", {})
    v35_gate = v35.get("hysteresis_gate", {})
    gate_fields = (
        "high_entropy_threshold",
        "low_entropy_threshold",
        "risk_threshold",
        "rearm_low_entropy_token_count",
    )
    if baseline_version == "v3.4":
        for field in gate_fields:
            if not _required_equal(baseline_gate, v35_gate, field):
                raise ValueError("Baseline and V3.5 token-risk gates differ")


def metric_map(
    rows: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, bool]:
    return {
        sample_id: bool(row["conditions"]["v3"][field])
        for sample_id, row in rows.items()
    }


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("Comparison numeric values must be finite")
    return {
        "count": len(normalized),
        "total": sum(values),
        "mean": sum(normalized) / len(normalized),
        "median": _percentile(normalized, 0.5),
        "p95": _percentile(normalized, 0.95),
        "p99": _percentile(normalized, 0.99),
        "min": min(values),
        "max": max(values),
    }


def mechanism_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        row["conditions"]["v3"].get("online_diagnostics", {})
        for row in rows.values()
    ]
    attempts = [int(item.get("retrieval_attempt_count", 0)) for item in diagnostics]

    def total(name: str) -> int:
        return sum(int(item.get(name, 0)) for item in diagnostics)

    return {
        "sample_count": len(rows),
        "questions_with_attempt": sum(value > 0 for value in attempts),
        "retrieval_attempt_count": sum(attempts),
        "attempt_count_distribution": dict(sorted(Counter(attempts).items())),
        "activation_count": total("activation_count"),
        "replacement_count": total("replacement_count"),
        "duplicate_count": total("duplicate_count"),
        "abstain_count": total("abstain_count"),
        "terminal_abstain_count": total("terminal_abstain_count"),
        "clear_on_terminal_abstain_count": total(
            "clear_on_terminal_abstain_count"
        ),
        "rearm_count": total("rearm_count"),
        "memory_attention_step_count": total("memory_attention_step_count"),
        "static_selector_unavailable_count": sum(
            item.get("static_selector_unavailable") is True
            for item in diagnostics
        ),
    }


def _identity_mismatches(
    baseline_rows: Mapping[str, Mapping[str, Any]],
    v35_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    sample_mismatches: list[str] = []
    vanilla_mismatches: list[str] = []
    identity_fields = (
        "logical_split",
        "dataset_split",
        "source_index",
        "question_sha256",
        "answer_sha256",
        "prompt_token_count",
        "prompt_token_ids_sha256",
    )
    vanilla_fields = (
        "completion",
        "completion_token_ids_sha256",
        "generated_token_count",
        "strict_correct",
        "format_correct",
        "strict_reward",
        "scorer_version",
    )
    for sample_id in sorted(baseline_rows):
        baseline = baseline_rows[sample_id]
        treatment = v35_rows[sample_id]
        if any(baseline.get(field) != treatment.get(field) for field in identity_fields):
            sample_mismatches.append(sample_id)
        baseline_vanilla = baseline["conditions"]["vanilla"]
        treatment_vanilla = treatment["conditions"]["vanilla"]
        if (
            baseline_vanilla.get("completion_token_ids")
            != treatment_vanilla.get("completion_token_ids")
            or any(
                baseline_vanilla.get(field) != treatment_vanilla.get(field)
                for field in vanilla_fields
            )
        ):
            vanilla_mismatches.append(sample_id)
    return sample_mismatches, vanilla_mismatches


def markdown_report(value: Mapping[str, Any]) -> str:
    key = value["primary_paired_comparison_key"]
    strict = value[key]["strict"]
    formatting = value[key]["format"]
    baseline_label = value["baseline_version"]
    baseline = value["mechanism"]["baseline"]
    v35 = value["mechanism"]["v35"]
    return "\n".join([
        "# MemGen V3.5 applicability-aware matched dev comparison",
        "",
        f"- Integrity passed: `{str(value['integrity']['passed']).lower()}`",
        f"- Interpretation: `{value['interpretation']}`",
        f"- Baseline: `{baseline_label}`",
        f"- Samples: {strict['paired_sample_count']}",
        "",
        f"## Paired V3.5 minus {baseline_label}",
        "",
        "| Metric | Baseline | V3.5 | Delta | Improved | Harmed | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Strict | {strict['control_accuracy']} | {strict['treatment_accuracy']} | "
        f"{strict['mean_treatment_minus_control']} | "
        f"{strict['treatment_correct_control_wrong']} | "
        f"{strict['treatment_wrong_control_correct']} | "
        f"{strict['mcnemar_exact_two_sided_p']} |",
        f"| Format | {formatting['control_accuracy']} | "
        f"{formatting['treatment_accuracy']} | "
        f"{formatting['mean_treatment_minus_control']} | "
        f"{formatting['treatment_correct_control_wrong']} | "
        f"{formatting['treatment_wrong_control_correct']} | "
        f"{formatting['mcnemar_exact_two_sided_p']} |",
        "",
        f"- Strict paired-bootstrap 95% CI: `{strict['bootstrap_95_ci']}`",
        f"- Mean generated-token delta: "
        f"{value['generated_token_delta_v35_minus_baseline']['mean']}",
        f"- Exact V3 completion matches: "
        f"{value['completion_parity']['exact_match_count']} / "
        f"{strict['paired_sample_count']}",
        "",
        "## Mechanism",
        "",
        "| Condition | Attempts | Activations | Replacements | Duplicates | Abstains | Terminal abstains | Clears | Re-arms | Attention steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {baseline_label} | {baseline['retrieval_attempt_count']} | "
        f"{baseline['activation_count']} | {baseline['replacement_count']} | "
        f"{baseline['duplicate_count']} | {baseline['abstain_count']} | "
        f"{baseline['terminal_abstain_count']} | "
        f"{baseline['clear_on_terminal_abstain_count']} | "
        f"{baseline['rearm_count']} | {baseline['memory_attention_step_count']} |",
        f"| V3.5 | {v35['retrieval_attempt_count']} | "
        f"{v35['activation_count']} | {v35['replacement_count']} | "
        f"{v35['duplicate_count']} | {v35['abstain_count']} | "
        f"{v35['terminal_abstain_count']} | "
        f"{v35['clear_on_terminal_abstain_count']} | "
        f"{v35['rearm_count']} | {v35['memory_attention_step_count']} |",
        "",
        "This is an exploratory matched-dev comparison of a compound selector "
        "plus lifecycle revision. It is not an independent confirmation and "
        "does not authorize final-test.",
        "",
    ])


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    baseline_profile = load_profile(
        args.baseline_profile, expected_schema=EVALUATION_PROFILE_SCHEMA
    )
    v35_profile = load_profile(
        args.v35_profile, expected_schema=V35_EVALUATION_PROFILE_SCHEMA
    )
    validate_baseline_profile(
        baseline_profile, requested_version=args.baseline_version
    )
    selector = validate_v35_profile(
        v35_profile, selector_path=args.v35_selector_calibration
    )
    validate_profile_compatibility(
        baseline_profile,
        v35_profile,
        baseline_version=args.baseline_version,
    )
    baseline_rows = load_rows(
        args.baseline_results,
        profile_sha256=str(baseline_profile["profile_sha256"]),
        expected_schema=EVALUATION_ROW_SCHEMA,
    )
    v35_rows = load_rows(
        args.v35_results,
        profile_sha256=str(v35_profile["profile_sha256"]),
        expected_schema=V35_EVALUATION_ROW_SCHEMA,
    )
    expected_count = int(v35_profile["selected_sample_count"])
    if (
        set(baseline_rows) != set(v35_rows)
        or len(baseline_rows) != expected_count
        or len(v35_rows) != expected_count
    ):
        raise ValueError("Comparison runs are incomplete or use different samples")
    identity_mismatches, vanilla_mismatches = _identity_mismatches(
        baseline_rows, v35_rows
    )
    if identity_mismatches:
        raise ValueError("Matched rows have different sample/prompt identity")
    if vanilla_mismatches:
        raise ValueError("Vanilla generations differ across matched runs")
    sample_ids = sorted(v35_rows)
    strict = paired_binary_effect(
        metric_map(v35_rows, "strict_correct"),
        metric_map(baseline_rows, "strict_correct"),
        seed=args.seed,
        resamples=args.bootstrap_resamples,
    )
    formatting = paired_binary_effect(
        metric_map(v35_rows, "format_correct"),
        metric_map(baseline_rows, "format_correct"),
        seed=args.seed + 1,
        resamples=args.bootstrap_resamples,
    )
    token_deltas = [
        int(v35_rows[sample_id]["conditions"]["v3"]["generated_token_count"])
        - int(
            baseline_rows[sample_id]["conditions"]["v3"][
                "generated_token_count"
            ]
        )
        for sample_id in sample_ids
    ]
    exact_matches = sum(
        v35_rows[sample_id]["conditions"]["v3"][
            "completion_token_ids_sha256"
        ]
        == baseline_rows[sample_id]["conditions"]["v3"][
            "completion_token_ids_sha256"
        ]
        for sample_id in sample_ids
    )
    comparison_key = (
        "paired_v35_minus_v34"
        if args.baseline_version == "v3.4"
        else "paired_v35_minus_v31"
    )
    report: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "created_at": utc_now(),
        "status": "completed",
        "logical_split": "dev-test",
        "baseline_version": args.baseline_version,
        "primary_paired_comparison_key": comparison_key,
        "integrity": {
            "passed": True,
            "profiles_authenticated": True,
            "rows_authenticated": True,
            "sample_ids_exactly_matched": True,
            "sample_and_prompt_identity_matched": True,
            "shared_profile_contract_matched": True,
            "shared_frozen_inputs_matched": True,
            "vanilla_exact_match": True,
            "vanilla_mismatch_examples": [],
            "v35_selector_authenticated_and_answer_blind": True,
        },
        comparison_key: {"strict": strict, "format": formatting},
        "paired_v35_minus_baseline": {
            "strict": strict,
            "format": formatting,
        },
        "generated_token_delta_v35_minus_baseline": numeric_summary(
            token_deltas
        ),
        "condition_generated_tokens": {
            "baseline": numeric_summary([
                int(row["conditions"]["v3"]["generated_token_count"])
                for row in baseline_rows.values()
            ]),
            "v35": numeric_summary([
                int(row["conditions"]["v3"]["generated_token_count"])
                for row in v35_rows.values()
            ]),
        },
        "completion_parity": {
            "exact_match_count": exact_matches,
            "mismatch_count": expected_count - exact_matches,
        },
        "mechanism": {
            "baseline": mechanism_summary(baseline_rows),
            "v35": mechanism_summary(v35_rows),
        },
        "selector": {
            "artifact_sha256": selector.get("artifact_sha256"),
            "task_accuracy_used": False,
            "answer_or_reward_used": False,
            "calibration": selector.get("calibration"),
        },
        "inputs": {
            "baseline_results_sha256": file_sha256(args.baseline_results),
            "baseline_profile_file_sha256": file_sha256(args.baseline_profile),
            "baseline_profile_sha256": baseline_profile["profile_sha256"],
            "v35_results_sha256": file_sha256(args.v35_results),
            "v35_profile_file_sha256": file_sha256(args.v35_profile),
            "v35_profile_sha256": v35_profile["profile_sha256"],
            "v35_selector_calibration_sha256": file_sha256(
                args.v35_selector_calibration
            ),
        },
        "interpretation": "exploratory_matched_dev_not_independent_confirmation",
        "compound_revision": {
            "selector_change": "question_only_applicability_shortlist_then_dynamic_rerank",
            "lifecycle_change": "terminal_abstain_clear_current_memory",
            "single_change_attribution_supported": False,
        },
        "versioned_gate_compatibility": {
            "risk_artifact_equal": (
                baseline_profile.get("inputs", {}).get("risk_artifact_sha256")
                == v35_profile.get("inputs", {}).get("risk_artifact_sha256")
            ),
            "risk_artifact_equality_required": (
                args.baseline_version == "v3.4"
            ),
            "v31_boundary_risk_difference_allowed": (
                args.baseline_version == "v3.1"
            ),
            "interpretation": (
                "V3.1 uses the legacy boundary diagnostic artifact and gate; "
                "the optional V3.5-minus-V3.1 result is not a gate-isolated "
                "comparison."
                if args.baseline_version == "v3.1"
                else "V3.4 and V3.5 share the frozen continuous token-risk artifact."
            ),
        },
        "qualified_for_final_test": False,
    }
    report["report_sha256"] = canonical_json_sha256({
        key: item
        for key, item in report.items()
        if key not in {"created_at", "report_sha256"}
    })
    write_json_atomic(args.output, report)
    args.output.with_suffix(".md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(
        f"[v3.5-compare] baseline={args.baseline_version} "
        f"samples={expected_count} "
        f"strict_delta={strict['mean_treatment_minus_control']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
