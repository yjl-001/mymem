#!/usr/bin/env python3
"""Apply frozen V3.5 exploratory matched-dev qualification criteria.

Passing this gate means only ``qualified_for_user_review``.  The repeatedly
inspected dev split is not an independent confirmation, and this script always
keeps ``qualified_for_final_test`` false pending a separate user authorization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3_5_selector import load_v35_selector_calibration


SCHEMA = "experience-memory-v3.5-dev-qualification-v1"
COMPARISON_SCHEMA = (
    "experience-memory-v3.5-applicability-selector-comparison-v1"
)
ANALYSIS_SCHEMA = "experience-memory-v3-analysis-report-v1"
MINIMUM_STRICT_DELTA = 0.0
MINIMUM_STRICT_CI_LOWER_BOUND = -0.015
MINIMUM_FORMAT_DELTA = -0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument(
        "--selector-calibration", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-strict-delta", type=float, default=MINIMUM_STRICT_DELTA
    )
    parser.add_argument(
        "--minimum-strict-ci-lower-bound",
        type=float,
        default=MINIMUM_STRICT_CI_LOWER_BOUND,
    )
    parser.add_argument(
        "--minimum-format-delta", type=float, default=MINIMUM_FORMAT_DELTA
    )
    return parser.parse_args()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_authenticated_comparison(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "report_sha256"}
    })
    if (
        value.get("schema_version") != COMPARISON_SCHEMA
        or value.get("status") != "completed"
        or value.get("report_sha256") != expected
        or value.get("logical_split") != "dev-test"
        or value.get("baseline_version") != "v3.4"
    ):
        raise ValueError("Invalid V3.5-minus-V3.4 matched-dev comparison")
    return value


def _load_authenticated_analysis(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = canonical_json_sha256({
        key: item for key, item in value.items() if key != "report_sha256"
    })
    if (
        value.get("schema_version") != ANALYSIS_SCHEMA
        or value.get("status") != "completed"
        or value.get("report_sha256") != expected
        or value.get("run", {}).get("logical_split") != "dev-test"
        or value.get("run", {}).get("system_profile", {}).get(
            "schema_version"
        )
        != "experience-memory-system-profile-v3.5"
    ):
        raise ValueError("Invalid V3.5 dev analysis report")
    return value


def _violation_count(analysis: Mapping[str, Any], name: str) -> int:
    safety = analysis.get("safety", {})
    counts = safety.get("violation_counts", {})
    if name in counts:
        return int(counts[name])
    detail = safety.get("violations", {}).get(name)
    if isinstance(detail, Mapping):
        return int(detail.get("count", len(detail.get("sample_ids", []))))
    if isinstance(detail, list):
        return len(detail)
    return -1


def markdown(value: Mapping[str, Any]) -> str:
    observed = value["observed"]
    thresholds = value["thresholds"]
    failed = [
        name for name, passed in value["requirements"].items() if not passed
    ]
    return "\n".join([
        "# MemGen V3.5 exploratory matched-dev qualification",
        "",
        f"- Status: `{value['status']}`",
        f"- Qualified for user review: "
        f"`{str(value['qualified_for_user_review']).lower()}`",
        "- Qualified for final-test: `false`",
        f"- Strict delta: {observed['strict_delta']} "
        f"(minimum {thresholds['minimum_strict_delta']})",
        f"- Strict CI lower bound: {observed['strict_ci_lower_bound']} "
        f"(minimum {thresholds['minimum_strict_ci_lower_bound']})",
        f"- Format delta: {observed['format_delta']} "
        f"(minimum {thresholds['minimum_format_delta']})",
        f"- Failed requirements: `{json.dumps(failed)}`",
        "",
        "This gate combines hard integrity/safety requirements with the frozen "
        "V3.5-minus-V3.4 engineering thresholds. Passing does not authorize or "
        "launch final-test; explicit user authorization remains required.",
        "",
    ])


def main() -> None:
    args = parse_args()
    if (
        args.minimum_strict_delta != MINIMUM_STRICT_DELTA
        or args.minimum_strict_ci_lower_bound
        != MINIMUM_STRICT_CI_LOWER_BOUND
        or args.minimum_format_delta != MINIMUM_FORMAT_DELTA
    ):
        raise ValueError("V3.5 exploratory qualification thresholds are frozen")
    comparison = _load_authenticated_comparison(args.comparison)
    analysis = _load_authenticated_analysis(args.analysis)
    selector = load_v35_selector_calibration(args.selector_calibration)
    comparison_selector = comparison.get("selector", {})
    if (
        selector.get("status") != "passed"
        or selector.get("task_accuracy_used") is not False
        or selector.get("answer_or_reward_used") is not False
        or not all(selector.get("requirements", {}).values())
        or comparison_selector.get("artifact_sha256")
        != selector.get("artifact_sha256")
        or comparison.get("inputs", {}).get(
            "v35_selector_calibration_sha256"
        )
        != file_sha256(args.selector_calibration)
        or analysis.get("run", {}).get("profile_sha256")
        != comparison.get("inputs", {}).get("v35_profile_sha256")
        or analysis.get("run", {}).get("run_profile_file_sha256")
        != comparison.get("inputs", {}).get("v35_profile_file_sha256")
        or analysis.get("run", {}).get("results_file_sha256")
        != comparison.get("inputs", {}).get("v35_results_sha256")
    ):
        raise ValueError("Qualification inputs are not mutually authenticated")
    paired = comparison.get("paired_v35_minus_v34", {})
    strict = paired.get("strict", {})
    formatting = paired.get("format", {})
    strict_ci = strict.get("bootstrap_95_ci")
    if not isinstance(strict_ci, list) or len(strict_ci) != 2:
        raise ValueError("V3.5 comparison has no strict bootstrap interval")
    observed = {
        "strict_delta": float(strict["mean_treatment_minus_control"]),
        "strict_ci_lower_bound": float(strict_ci[0]),
        "format_delta": float(formatting["mean_treatment_minus_control"]),
    }
    if (
        not all(math.isfinite(value) for value in observed.values())
        or not all(-1.0 <= value <= 1.0 for value in observed.values())
        or float(strict_ci[0]) > float(strict_ci[1])
    ):
        raise ValueError("V3.5 comparison task metrics are invalid")
    comparison_count = int(strict.get("paired_sample_count", -1))
    analysis_count = int(analysis.get("run", {}).get("selected_sample_count", -1))
    analysis_scope_count = int(
        analysis.get("paired_analysis", {}).get("overall", {}).get(
            "sample_count", -1
        )
    )
    if (
        comparison_count <= 0
        or int(formatting.get("paired_sample_count", -1)) != comparison_count
        or analysis_count != comparison_count
        or analysis_scope_count != comparison_count
    ):
        raise ValueError("V3.5 qualification inputs are incomplete or count-mismatched")
    thresholds = {
        "minimum_strict_delta": args.minimum_strict_delta,
        "minimum_strict_ci_lower_bound": args.minimum_strict_ci_lower_bound,
        "minimum_format_delta": args.minimum_format_delta,
    }
    zero_attempt = analysis.get("zero_attempt_parity", {})
    static_unavailable = analysis.get("static_selector_unavailable_parity", {})
    violation_names = (
        "selected_outside_shortlist",
        "stale_attention_after_terminal_clear",
        "terminal_state_drift",
        "full_prefix_query",
        "kv_alignment",
        "attempt_budget",
        "rearm",
    )
    violations = {
        name: _violation_count(analysis, name) for name in violation_names
    }
    requirements = {
        "comparison_integrity_passed": (
            comparison.get("integrity", {}).get("passed") is True
        ),
        "analysis_integrity_passed": (
            analysis.get("integrity", {}).get("passed") is True
        ),
        "analysis_v35_safety_audit_passed": (
            analysis.get("safety", {}).get("applicable") is True
            and analysis.get("safety", {}).get("passed") is True
        ),
        "zero_attempt_exact_parity": (
            int(zero_attempt.get("mismatch_count", -1)) == 0
        ),
        "static_unavailable_exact_parity": (
            int(static_unavailable.get("mismatch_count", -1)) == 0
        ),
        "selected_outside_shortlist_zero": (
            violations["selected_outside_shortlist"] == 0
        ),
        "kv_alignment_violations_zero": (
            violations["kv_alignment"] == 0
        ),
        "stale_attention_after_terminal_clear_zero": (
            violations["stale_attention_after_terminal_clear"] == 0
        ),
        "terminal_state_drift_zero": (
            violations["terminal_state_drift"] == 0
        ),
        "full_prefix_query_violations_zero": (
            violations["full_prefix_query"] == 0
        ),
        "attempt_budget_violations_zero": (
            violations["attempt_budget"] == 0
        ),
        "rearm_violations_zero": violations["rearm"] == 0,
        "selector_task_accuracy_not_used": (
            selector.get("task_accuracy_used") is False
        ),
        "selector_answer_or_reward_not_used": (
            selector.get("answer_or_reward_used") is False
        ),
        "strict_point_delta_passed": (
            observed["strict_delta"] >= args.minimum_strict_delta
        ),
        "strict_ci_lower_bound_passed": (
            observed["strict_ci_lower_bound"]
            >= args.minimum_strict_ci_lower_bound
        ),
        "format_point_delta_passed": (
            observed["format_delta"] >= args.minimum_format_delta
        ),
    }
    qualified_for_user_review = all(requirements.values())
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if qualified_for_user_review else "not_qualified",
        "evaluation_interpretation": (
            "exploratory_matched_dev_not_independent_confirmation"
        ),
        "qualified_for_user_review": qualified_for_user_review,
        "qualified_for_final_test": False,
        "final_test_blocked_pending_explicit_user_authorization": True,
        "observed": observed,
        "thresholds": thresholds,
        "safety_violation_counts": violations,
        "requirements": requirements,
        "inputs": {
            "comparison_sha256": file_sha256(args.comparison),
            "comparison_report_sha256": comparison["report_sha256"],
            "analysis_sha256": file_sha256(args.analysis),
            "analysis_report_sha256": analysis["report_sha256"],
            "selector_calibration_sha256": file_sha256(
                args.selector_calibration
            ),
            "selector_artifact_sha256": selector["artifact_sha256"],
        },
        "guardrails": {
            "dev_has_been_repeatedly_inspected": True,
            "independent_confirmation": False,
            "compound_revision_single_change_attribution_supported": False,
            "automatic_final_test_launch_allowed": False,
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
        f"[v3.5-dev-qualification] status={value['status']} "
        f"qualified_for_user_review={qualified_for_user_review} "
        "qualified_for_final_test=False "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
