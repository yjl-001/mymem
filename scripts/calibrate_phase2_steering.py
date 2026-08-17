#!/usr/bin/env python3
"""Calibrate a global steering vector on GSM8K calibration-val only.

The script first measures candidate-boundary entropies, then searches real
vector settings on a deterministic tuning prefix.  It freezes the winner and
runs the required six controls on a disjoint confirmation suffix of the same
calibration split.  ``final-test`` is never read here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.phase2 import (
    STEERING_CALIBRATION_SCHEMA,
    parse_csv_numbers,
    select_calibration_winner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--layers", default="8,16,24")
    parser.add_argument("--alphas", default="0.02,0.05")
    parser.add_argument("--gate-slopes", default="0.10,0.25")
    parser.add_argument("--max-injections-grid", default="2")
    parser.add_argument("--entropy-quantile", type=float, default=0.85)
    parser.add_argument("--tune-size", type=int, default=100)
    parser.add_argument("--confirm-size", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--r-max", type=float, default=0.10)
    parser.add_argument("--sink-token-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("No candidate-boundary entropy values were recorded")
    if not 0 <= quantile <= 1:
        raise ValueError("entropy-quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entropy_values(results_path: Path) -> list[float]:
    values: list[float] = []
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for event in record.get("intervention_trace", []):
                if "entropy" in event:
                    values.append(float(event["entropy"]))
    return values


def run_evaluation(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    condition: str,
    layer: int,
    alpha: float,
    entropy_threshold: float,
    gate_slope: float,
    max_injections: int,
    offset: int,
    limit: int,
    random_boundary_rate: float = 0.15,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_steering_vector.py"),
        "--artifact", str(args.artifact),
        "--split-manifest", str(args.split_manifest),
        "--output-dir", str(output_dir),
        "--condition", condition,
        "--model", args.model,
        "--model-revision", args.model_revision,
        "--dataset-revision", args.dataset_revision,
        "--logical-split", "calibration-val",
        "--layer", str(layer),
        "--alpha", str(alpha),
        "--entropy-threshold", str(entropy_threshold),
        "--gate-slope", str(gate_slope),
        "--max-injections", str(max_injections),
        "--r-max", str(args.r_max),
        "--sink-token-count", str(args.sink_token_count),
        "--max-new-tokens", str(args.max_new_tokens),
        "--offset", str(offset),
        "--limit", str(limit),
        "--seed", str(args.seed),
        "--random-boundary-rate", str(random_boundary_rate),
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation,
    ]
    print("[phase2-calibration]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return load_json(output_dir / "summary.json")


def main() -> None:
    args = parse_args()
    if args.tune_size <= 0 or args.confirm_size <= 0:
        raise ValueError("tune-size and confirm-size must be positive")
    if args.max_new_tokens <= 0 or args.r_max <= 0 or args.sink_token_count < 0:
        raise ValueError("Invalid generation/safety argument")
    layers = list(parse_csv_numbers(args.layers, integer=True))
    alphas = list(parse_csv_numbers(args.alphas))
    slopes = list(parse_csv_numbers(args.gate_slopes))
    budgets = list(parse_csv_numbers(args.max_injections_grid, integer=True))
    if any(value <= 0 for value in layers) or any(value < 0 for value in alphas):
        raise ValueError("layers must be positive and alphas non-negative")
    if any(value <= 0 for value in slopes) or any(value < 0 for value in budgets):
        raise ValueError("gate slopes must be positive and budgets non-negative")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir = output_dir / "entropy_bootstrap"
    # Threshold is immaterial to entropy_only; a finite sentinel is supplied
    # only to satisfy the common evaluator interface.
    bootstrap = run_evaluation(
        args,
        output_dir=bootstrap_dir,
        condition="entropy_only",
        layer=layers[0],
        alpha=0.0,
        entropy_threshold=0.0,
        gate_slope=slopes[0],
        max_injections=0,
        offset=0,
        limit=args.tune_size,
    )
    values = entropy_values(bootstrap_dir / "results.jsonl")
    threshold = percentile(values, args.entropy_quantile)
    threshold_artifact = {
        "schema_version": "phase2-entropy-threshold-v1",
        "source_results_sha256": bootstrap["results"]["sha256"],
        "candidate_count": len(values),
        "quantile": args.entropy_quantile,
        "entropy_threshold": threshold,
        "logical_split": "calibration-val",
        "offset": 0,
        "limit": args.tune_size,
    }
    (output_dir / "entropy_threshold.json").write_text(
        json.dumps(threshold_artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    vanilla = run_evaluation(
        args,
        output_dir=output_dir / "tune_vanilla",
        condition="vanilla",
        layer=layers[0],
        alpha=0.0,
        entropy_threshold=threshold,
        gate_slope=slopes[0],
        max_injections=0,
        offset=0,
        limit=args.tune_size,
    )
    calibration_rows: list[dict[str, Any]] = []
    for layer in layers:
        for alpha in alphas:
            for slope in slopes:
                for budget in budgets:
                    config = {
                        "layer": layer,
                        "alpha": alpha,
                        "gate_slope": slope,
                        "max_injections": budget,
                        "entropy_threshold": threshold,
                    }
                    config_id = f"l{layer}_a{alpha:g}_s{slope:g}_b{budget}"
                    summary = run_evaluation(
                        args,
                        output_dir=output_dir / "tune_real_vector" / config_id,
                        condition="real_vector",
                        layer=layer,
                        alpha=alpha,
                        entropy_threshold=threshold,
                        gate_slope=slope,
                        max_injections=budget,
                        offset=0,
                        limit=args.tune_size,
                    )
                    calibration_rows.append(
                        {
                            "condition": "real_vector",
                            "config": config,
                            "accuracy": summary["accuracy"],
                            "format_accuracy": summary["format_accuracy"],
                            "vanilla_format_accuracy": vanilla["format_accuracy"],
                            "mean_injections": summary["mean_injections"],
                            "safety_failed": (
                                summary["disabled_injection_count"] > 0
                                or summary["max_observed_relative_delta_norm"] > args.r_max
                            ),
                            "summary_path": str((output_dir / "tune_real_vector" / config_id / "summary.json")),
                        }
                    )
    winner = select_calibration_winner(calibration_rows)
    selected_config = dict(winner["config"])
    real_tune_summary = load_json(Path(winner["summary_path"]))
    candidate_count = int(real_tune_summary["candidate_boundary_count"])
    random_boundary_rate = (
        float(real_tune_summary["injection_applied_count"]) / candidate_count
        if candidate_count
        else 0.0
    )

    controls: dict[str, dict[str, Any]] = {}
    for condition in (
        "vanilla",
        "entropy_only",
        "real_vector",
        "random_boundary",
        "random_vector",
        "reversed_vector",
    ):
        controls[condition] = run_evaluation(
            args,
            output_dir=output_dir / "confirmation" / condition,
            condition=condition,
            layer=int(selected_config["layer"]),
            alpha=float(selected_config["alpha"]),
            entropy_threshold=float(selected_config["entropy_threshold"]),
            gate_slope=float(selected_config["gate_slope"]),
            max_injections=int(selected_config["max_injections"]),
            offset=args.tune_size,
            limit=args.confirm_size,
            random_boundary_rate=random_boundary_rate,
        )
    real = controls["real_vector"]
    acceptance = {
        "real_beats_random_vector": real["accuracy"] > controls["random_vector"]["accuracy"],
        "real_beats_reversed_vector": real["accuracy"] > controls["reversed_vector"]["accuracy"],
        "real_beats_random_boundary": real["accuracy"] > controls["random_boundary"]["accuracy"],
        "format_not_below_vanilla": real["format_accuracy"] >= controls["vanilla"]["format_accuracy"],
        "no_disabled_or_overlimit_injection": (
            real["disabled_injection_count"] == 0
            and real["max_observed_relative_delta_norm"] <= args.r_max
        ),
    }
    report = {
        "schema_version": STEERING_CALIBRATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.artifact),
        "split": {
            "logical_split": "calibration-val",
            "tune": {"offset": 0, "count": args.tune_size},
            "confirmation": {"offset": args.tune_size, "count": args.confirm_size},
        },
        "entropy_threshold": threshold_artifact,
        "vanilla_tune_summary": vanilla,
        "calibration_candidates": calibration_rows,
        "selected": winner,
        "random_boundary_rate": random_boundary_rate,
        "confirmation_controls": controls,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
    }
    report["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    report_path = output_dir / "phase2_calibration_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[phase2-calibration] selected={selected_config} passed={report['passed']} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
