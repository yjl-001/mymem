#!/usr/bin/env python3
"""Apply frozen go/no-go criteria to the matched V3.4 dev comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256


SCHEMA = "experience-memory-v3.4-dev-qualification-v1"
COMPARISON_SCHEMA = "experience-memory-v3.4-continuous-gate-comparison-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-strict-delta", type=float, default=0.0)
    parser.add_argument(
        "--minimum-strict-ci-lower-bound", type=float, default=-0.015
    )
    parser.add_argument("--minimum-format-delta", type=float, default=-0.005)
    return parser.parse_args()


def markdown(value: Mapping[str, Any]) -> str:
    observed = value["observed"]
    thresholds = value["thresholds"]
    return "\n".join([
        "# MemGen V3.4 matched-dev qualification",
        "",
        f"- Status: `{value['status']}`",
        f"- Qualified for final-test: `{str(value['qualified_for_final_test']).lower()}`",
        f"- Strict delta: {observed['strict_delta']} (minimum {thresholds['minimum_strict_delta']})",
        f"- Strict CI lower bound: {observed['strict_ci_lower_bound']} (minimum {thresholds['minimum_strict_ci_lower_bound']})",
        f"- Format delta: {observed['format_delta']} (minimum {thresholds['minimum_format_delta']})",
        "",
        "The selector was calibrated answer-blindly; this go/no-go decision uses dev-test outcomes only.",
        "",
    ])


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    expected_hash = canonical_json_sha256({
        key: item
        for key, item in comparison.items()
        if key not in {"created_at", "report_sha256"}
    })
    if (
        comparison.get("schema_version") != COMPARISON_SCHEMA
        or comparison.get("status") != "completed"
        or comparison.get("report_sha256") != expected_hash
        or comparison.get("integrity", {}).get("passed") is not True
    ):
        raise ValueError("Invalid V3.4 matched-dev comparison")
    paired = comparison["paired_v34_minus_v31"]
    strict = paired["strict"]
    formatting = paired["format"]
    strict_ci = strict.get("bootstrap_95_ci")
    if not isinstance(strict_ci, list) or len(strict_ci) != 2:
        raise ValueError("V3.4 comparison has no strict bootstrap interval")
    observed = {
        "strict_delta": float(strict["mean_treatment_minus_control"]),
        "strict_ci_lower_bound": float(strict_ci[0]),
        "format_delta": float(formatting["mean_treatment_minus_control"]),
    }
    thresholds = {
        "minimum_strict_delta": args.minimum_strict_delta,
        "minimum_strict_ci_lower_bound": args.minimum_strict_ci_lower_bound,
        "minimum_format_delta": args.minimum_format_delta,
    }
    requirements = {
        "comparison_integrity_passed": True,
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
    qualified = all(requirements.values())
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if qualified else "not_qualified",
        "qualified_for_final_test": qualified,
        "observed": observed,
        "thresholds": thresholds,
        "requirements": requirements,
        "inputs": {
            "comparison_sha256": file_sha256(args.comparison),
            "comparison_report_sha256": comparison["report_sha256"],
        },
    }
    value["report_sha256"] = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "report_sha256"}
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(markdown(value), encoding="utf-8")
    print(
        f"[v3.4-dev-qualification] status={value['status']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
