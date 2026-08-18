#!/usr/bin/env python3
"""Summarize the fixed one-shot entropy-risk steering probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = ("vanilla", "entropy_only", "real_vector", "random_vector", "reversed_vector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_decision(record: dict[str, Any]) -> dict[str, Any] | None:
    for event in record.get("intervention_trace", []):
        if event.get("high_entropy") and event.get("first_high_entropy_decision_available"):
            return event
    return None


def entropy_summary(records: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    if mode not in {"triggered", "injected"}:
        raise ValueError(f"Unsupported entropy-summary mode: {mode}")
    events = [
        event
        for record in records
        for event in record.get("intervention_trace", [])
        if (
            bool(event.get("entropy_triggered", False))
            if mode == "triggered"
            else bool(event.get("injection", {}).get("applied"))
        )
        and "entropy_delta_to_next_candidate" in event
    ]
    deltas = [float(event["entropy_delta_to_next_candidate"]) for event in events]
    return {
        "measured_count": len(deltas),
        "mean_delta_to_next_candidate": sum(deltas) / len(deltas) if deltas else None,
        "entropy_decreased_rate": (
            sum(bool(event.get("entropy_decreased_to_next_candidate")) for event in events) / len(deltas)
            if deltas else None
        ),
    }


def main() -> None:
    args = parse_args()
    diagnostic = json.loads(args.diagnostic_report.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "passed":
        raise ValueError("The held-out bank risk diagnostic did not pass; online summary is invalid")
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        root = args.evaluation_root / condition
        records_by_condition[condition] = load_jsonl(root / "results.jsonl")
        summaries[condition] = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    vanilla = {str(row["sample_id"]): row for row in records_by_condition["vanilla"]}
    entropy_only = {str(row["sample_id"]): row for row in records_by_condition["entropy_only"]}
    if set(vanilla) != set(entropy_only):
        raise ValueError("vanilla and entropy-only do not cover identical sample IDs")
    vanilla_matches_entropy_only = all(
        vanilla[sample_id]["completion"] == entropy_only[sample_id]["completion"] for sample_id in vanilla
    )
    baseline_decisions = {
        sample_id: first_decision(record) for sample_id, record in entropy_only.items()
    }
    prefix_matched: dict[str, bool] = {}
    for condition in ("real_vector", "random_vector", "reversed_vector"):
        rows = {str(row["sample_id"]): row for row in records_by_condition[condition]}
        if set(rows) != set(entropy_only):
            raise ValueError(f"{condition} does not cover identical sample IDs")
        prefix_matched[condition] = all(
            math.isclose(
                float((first_decision(rows[sample_id]) or {})["entropy"]),
                float((baseline_decisions[sample_id] or {})["entropy"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                float((first_decision(rows[sample_id]) or {})["persistence_risk_score"]),
                float((baseline_decisions[sample_id] or {})["persistence_risk_score"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for sample_id in rows
            if baseline_decisions[sample_id] is not None
        )
    entropy = {
        "entropy_only": entropy_summary(records_by_condition["entropy_only"], mode="triggered"),
        "real_vector": entropy_summary(records_by_condition["real_vector"], mode="injected"),
        "random_vector": entropy_summary(records_by_condition["random_vector"], mode="injected"),
        "reversed_vector": entropy_summary(records_by_condition["reversed_vector"], mode="injected"),
    }
    real_delta = entropy["real_vector"]["mean_delta_to_next_candidate"]
    acceptance = {
        "vanilla_matches_entropy_only": vanilla_matches_entropy_only,
        "first_decision_prefix_matched": all(prefix_matched.values()),
        "format_not_below_vanilla": summaries["real_vector"]["format_accuracy"] >= summaries["vanilla"]["format_accuracy"],
        "no_disabled_or_overlimit_injection": (
            summaries["real_vector"]["disabled_injection_count"] == 0
            and summaries["real_vector"]["max_observed_relative_delta_norm"]
            <= summaries["real_vector"]["config"]["r_max"]
        ),
        "real_recovers_faster_than_entropy_only": (
            real_delta is not None and entropy["entropy_only"]["mean_delta_to_next_candidate"] is not None
            and real_delta < entropy["entropy_only"]["mean_delta_to_next_candidate"]
        ),
        "real_recovers_faster_than_random_vector": (
            real_delta is not None and entropy["random_vector"]["mean_delta_to_next_candidate"] is not None
            and real_delta < entropy["random_vector"]["mean_delta_to_next_candidate"]
        ),
        "real_recovers_faster_than_reversed_vector": (
            real_delta is not None and entropy["reversed_vector"]["mean_delta_to_next_candidate"] is not None
            and real_delta < entropy["reversed_vector"]["mean_delta_to_next_candidate"]
        ),
    }
    output = {
        "schema_version": "phase2-entropy-risk-probe-summary-v2",
        "diagnostic": {
            "path": str(args.diagnostic_report),
            "risk_diagnostic": diagnostic.get("risk_diagnostic"),
            "four_cell_counts": diagnostic.get("four_cell_counts"),
        },
        "conditions": {
            condition: {
                "accuracy": summaries[condition].get("accuracy"),
                "format_accuracy": summaries[condition].get("format_accuracy"),
                "mean_injections": summaries[condition].get("mean_injections"),
                "injection_logit_diagnostics": summaries[condition].get("injection_logit_diagnostics"),
            }
            for condition in CONDITIONS
        },
        "entropy_recovery": entropy,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[phase2-risk-probe] passed={output['passed']} output={args.output}")


if __name__ == "__main__":
    main()
