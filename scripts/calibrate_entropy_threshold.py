#!/usr/bin/env python3
"""Choose an entropy-gate threshold from a calibration evaluation trace."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a FlashMem-style entropy threshold from entropy_gate_trace.csv."
    )
    parser.add_argument("trace_csv", type=Path)
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.85,
        help="Candidate-entropy quantile used as threshold (default: 0.85).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to entropy_threshold.json beside the trace file.",
    )
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("No entropy values found in trace")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    args = parse_args()
    trace_path = args.trace_csv.expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file does not exist: {trace_path}")

    values = []
    with trace_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Budget-exhausted rows intentionally omit entropy because no
            # attention forward was run; they are not calibration candidates.
            if row.get("entropy"):
                values.append(float(row["entropy"]))

    threshold = percentile(values, args.quantile)
    result = {
        "trace_csv": str(trace_path),
        "candidate_count": len(values),
        "quantile": args.quantile,
        "entropy_threshold": threshold,
        "command_override": (
            "--set model.weaver.insertion_strategy.entropy_threshold="
            f"{threshold:.8f}"
        ),
    }
    output_path = args.output_json or trace_path.with_name("entropy_threshold.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
