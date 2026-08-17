#!/usr/bin/env python3
"""Summarize Phase 2 vector-construction comparisons without selecting a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-report", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compiler = json.loads(args.compiler_report.read_text(encoding="utf-8"))
    rows = []
    for artifact in compiler.get("artifacts", []):
        method = str(artifact["method"])
        report_path = args.calibration_root / method / "phase2_calibration_report.json"
        if not report_path.exists():
            rows.append({"method": method, "status": "not_calibrated", "artifact": artifact})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        controls = report.get("confirmation_controls", {})
        real = controls.get("real_vector", {})
        vanilla = controls.get("vanilla", {})
        rows.append(
            {
                "method": method,
                "status": "calibrated",
                "artifact": artifact,
                "calibration_report": str(report_path),
                "passed": report.get("passed"),
                "selected": report.get("selected"),
                "acceptance": report.get("acceptance"),
                "confirmation": {
                    "real_accuracy": real.get("accuracy"),
                    "vanilla_accuracy": vanilla.get("accuracy"),
                    "real_format_accuracy": real.get("format_accuracy"),
                    "vanilla_format_accuracy": vanilla.get("format_accuracy"),
                },
            }
        )
    result = {
        "schema_version": "phase2-vector-construction-comparison-v1",
        "compiler_report": str(args.compiler_report),
        "compiler_unavailable_methods": compiler.get("unavailable_methods", []),
        "methods": rows,
        "selection_policy": (
            "This report is descriptive only. It must not select a construction method; "
            "any method advanced beyond calibration-val requires a separately frozen evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[phase2-ablation-summary] methods={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
