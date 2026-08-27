#!/usr/bin/env python3
"""Compare matched V3.1 raw-margin and V3.2 centered-margin evaluations."""

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

from compare_v3_selector_evaluations import (
    load_profile,
    load_rows,
    mechanism_summary,
    metric_map,
    write_json_atomic,
)
from memgen.experience.e1 import paired_binary_effect
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
)
from memgen.experience.v3 import (
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_selector import (
    load_margin_selector_calibration,
    numeric_summary,
)


COMPARISON_SCHEMA = "experience-memory-v3-retrieval-transform-comparison-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31-results", type=Path, required=True)
    parser.add_argument("--v31-profile", type=Path, required=True)
    parser.add_argument("--v31-calibration", type=Path, required=True)
    parser.add_argument("--v32-results", type=Path, required=True)
    parser.add_argument("--v32-profile", type=Path, required=True)
    parser.add_argument("--v32-calibration", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def system_transform(profile: Mapping[str, Any]) -> str:
    return str(profile.get("system_profile", {}).get(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    ))


def validate_margin_run(
    *,
    profile: Mapping[str, Any],
    calibration_path: Path,
    expected_transform: str,
) -> dict[str, Any]:
    system = profile.get("system_profile", {})
    if (
        system.get("retrieval_abstention_policy") != "top1_top2_margin"
        or system_transform(profile) != expected_transform
    ):
        raise ValueError("Retrieval-transform comparison received a wrong condition")
    calibration = load_margin_selector_calibration(calibration_path)
    embedded = profile.get("selector_calibration") or {}
    source_transform = calibration.get("source", {}).get(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    threshold = calibration.get("calibration", {}).get(
        "minimum_top1_top2_margin"
    )
    if (
        source_transform != expected_transform
        or calibration.get("source", {}).get(
            "retrieval_key_manifest_sha256"
        )
        != profile.get("inputs", {}).get("retrieval_key_manifest_sha256")
        or threshold != system.get("retrieval_min_top1_top2_margin")
        or embedded.get("artifact_sha256") != calibration.get("artifact_sha256")
        or profile.get("inputs", {}).get("selector_calibration_sha256")
        != file_sha256(calibration_path)
    ):
        raise ValueError("Evaluation is not bound to its transform-specific calibration")
    return calibration


def normalized_system(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(profile.get("system_profile", {}))
    value.setdefault(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    return value


def safe_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "experience_type": record.experience_type,
        "when_facing": record.sanitized_fields.get("when_facing"),
        "prefer": record.sanitized_fields.get("prefer"),
        "avoid": record.sanitized_fields.get("avoid"),
        "payload_hash": record.payload_hash,
    }


def calibration_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_sha256": value.get("artifact_sha256"),
        "source": value.get("source"),
        "calibration": value.get("calibration"),
        "first_attempt_selection_concentration": value.get(
            "first_attempt_selection_concentration"
        ),
    }


def hub_payloads(
    *,
    v31_calibration: Mapping[str, Any],
    v32_calibration: Mapping[str, Any],
    record_by_id: Mapping[str, MemoryRecord],
    limit: int = 10,
) -> list[dict[str, Any]]:
    def counts(calibration: Mapping[str, Any]) -> dict[str, int]:
        concentration = calibration.get(
            "first_attempt_selection_concentration", {}
        )
        return {
            str(item["memory_id"]): int(item["count"])
            for item in concentration.get("top_by_frequency", [])
        }

    raw_counts = counts(v31_calibration)
    centered_counts = counts(v32_calibration)
    candidates = sorted(
        set(raw_counts) | set(centered_counts),
        key=lambda memory_id: (
            -max(raw_counts.get(memory_id, 0), centered_counts.get(memory_id, 0)),
            memory_id,
        ),
    )[:limit]
    values = []
    for memory_id in candidates:
        if memory_id not in record_by_id:
            raise ValueError("Calibration selected an unknown memory ID")
        values.append(
            safe_payload(record_by_id[memory_id])
            | {
                "v31_first_attempt_count": raw_counts.get(memory_id, 0),
                "v32_first_attempt_count": centered_counts.get(memory_id, 0),
            }
        )
    return values


def markdown_report(value: Mapping[str, Any]) -> str:
    strict = value["paired_v32_minus_v31"]["strict"]
    formatting = value["paired_v32_minus_v31"]["format"]
    v31_cal = value["calibration"]["v31_raw"][
        "first_attempt_selection_concentration"
    ]
    v32_cal = value["calibration"]["v32_centered"][
        "first_attempt_selection_concentration"
    ]
    v31_mechanism = value["mechanism"]["v31_raw"]
    v32_mechanism = value["mechanism"]["v32_centered"]
    lines = [
        "# MemGen V3.2 centered-retrieval comparison",
        "",
        f"- Integrity passed: `{str(value['integrity']['passed']).lower()}`",
        f"- Logical split: `{value['logical_split']}`",
        f"- Samples: {strict['paired_sample_count']}",
        "",
        "## Calibration retrieval geometry",
        "",
        "| Condition | Margin threshold | Top-1 share | Gini | Selected memories |",
        "|---|---:|---:|---:|---:|",
        f"| V3.1 raw | {value['calibration']['v31_raw']['calibration']['minimum_top1_top2_margin']} | "
        f"{v31_cal['top1_share']} | {v31_cal['gini']} | {v31_cal['selected_memory_count']} |",
        f"| V3.2 centered | {value['calibration']['v32_centered']['calibration']['minimum_top1_top2_margin']} | "
        f"{v32_cal['top1_share']} | {v32_cal['gini']} | {v32_cal['selected_memory_count']} |",
        "",
        "## Paired V3.2 minus V3.1",
        "",
        "| Metric | V3.1 | V3.2 | Delta | Improved | Harmed | McNemar p |",
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
        f"- Mean generated-token delta: {value['paired_token_delta_v32_minus_v31']['mean']}",
        f"- Exact V3 completion matches: {value['completion_parity']['exact_match_count']} / {strict['paired_sample_count']}",
        "",
        "## Mechanism",
        "",
        "| Condition | Attempts | Activations | Replacements | Duplicates | Abstains | Attention steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V3.1 raw | {v31_mechanism['retrieval_attempt_count']} | {v31_mechanism['activation_count']} | "
        f"{v31_mechanism['replacement_count']} | {v31_mechanism['duplicate_count']} | "
        f"{v31_mechanism['abstain_count']} | {v31_mechanism['memory_attention_step_count']} |",
        f"| V3.2 centered | {v32_mechanism['retrieval_attempt_count']} | {v32_mechanism['activation_count']} | "
        f"{v32_mechanism['replacement_count']} | {v32_mechanism['duplicate_count']} | "
        f"{v32_mechanism['abstain_count']} | {v32_mechanism['memory_attention_step_count']} |",
        "",
        "This is a matched dev-test retrieval-transform experiment, not an independent final-test confirmation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    v31_profile = load_profile(args.v31_profile)
    v32_profile = load_profile(args.v32_profile)
    v31_calibration = validate_margin_run(
        profile=v31_profile,
        calibration_path=args.v31_calibration,
        expected_transform=V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    v32_calibration = validate_margin_run(
        profile=v32_profile,
        calibration_path=args.v32_calibration,
        expected_transform=V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
    )

    v31_core = normalized_system(v31_profile)
    v32_core = normalized_system(v32_profile)
    for value in (v31_core, v32_core):
        value.pop("retrieval_embedding_transform", None)
        value.pop("retrieval_min_top1_top2_margin", None)
    if v31_core != v32_core:
        raise ValueError("V3.1 and V3.2 change fields beyond retrieval geometry")
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
        v31_profile.get(field) != v32_profile.get(field)
        for field in comparable_profile_fields
    ):
        raise ValueError("V3.1 and V3.2 do not share one evaluation contract")
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
        != v32_profile.get("inputs", {}).get(field)
        for field in comparable_inputs
    ):
        raise ValueError("V3.1 and V3.2 use different frozen inputs")
    expected_memory_sha256 = v31_profile.get("inputs", {}).get(
        "memory_records_sha256"
    )
    if file_sha256(args.memory_records) != expected_memory_sha256:
        raise ValueError("Memory records differ from the matched evaluations")

    v31_rows = load_rows(
        args.v31_results,
        profile_sha256=str(v31_profile["profile_sha256"]),
    )
    v32_rows = load_rows(
        args.v32_results,
        profile_sha256=str(v32_profile["profile_sha256"]),
    )
    expected_count = int(v31_profile["selected_sample_count"])
    same_ids = set(v31_rows) == set(v32_rows)
    complete = len(v31_rows) == len(v32_rows) == expected_count
    if not same_ids or not complete:
        raise ValueError("V3.1 and V3.2 rows are incomplete or unmatched")
    vanilla_mismatches = []
    exact_matches = 0
    token_deltas = []
    for sample_id in sorted(v31_rows):
        v31 = v31_rows[sample_id]["conditions"]
        v32 = v32_rows[sample_id]["conditions"]
        if (
            v31["vanilla"]["completion_token_ids_sha256"]
            != v32["vanilla"]["completion_token_ids_sha256"]
            or v31["vanilla"]["strict_correct"]
            != v32["vanilla"]["strict_correct"]
            or v31["vanilla"]["format_correct"]
            != v32["vanilla"]["format_correct"]
        ):
            vanilla_mismatches.append(sample_id)
        exact_matches += int(
            v31["v3"]["completion_token_ids_sha256"]
            == v32["v3"]["completion_token_ids_sha256"]
        )
        token_deltas.append(
            int(v32["v3"]["generated_token_count"])
            - int(v31["v3"]["generated_token_count"])
        )
    if vanilla_mismatches:
        raise ValueError("Vanilla generations differ across V3.1 and V3.2")

    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    record_by_id = {record.memory_id: record for record in records}
    report = {
        "schema_version": COMPARISON_SCHEMA,
        "created_at": utc_now(),
        "status": "completed",
        "logical_split": v31_profile["logical_split"],
        "experimental_variable": (
            "retrieval_embedding_transform_with_transform_specific_"
            "answer_blind_margin_recalibration"
        ),
        "integrity": {
            "passed": True,
            "same_sample_ids": same_ids,
            "both_runs_complete": complete,
            "vanilla_exact_match": not vanilla_mismatches,
            "non_retrieval_system_profile_equal": True,
            "frozen_inputs_equal": True,
        },
        "calibration": {
            "v31_raw": calibration_summary(v31_calibration),
            "v32_centered": calibration_summary(v32_calibration),
        },
        "paired_v32_minus_v31": {
            "strict": paired_binary_effect(
                metric_map(v32_rows, "strict_correct"),
                metric_map(v31_rows, "strict_correct"),
                seed=args.seed,
                resamples=args.bootstrap_resamples,
            ),
            "format": paired_binary_effect(
                metric_map(v32_rows, "format_correct"),
                metric_map(v31_rows, "format_correct"),
                seed=args.seed + 1,
                resamples=args.bootstrap_resamples,
            ),
        },
        "paired_token_delta_v32_minus_v31": numeric_summary(
            [float(value) for value in token_deltas]
        ),
        "completion_parity": {
            "exact_match_count": exact_matches,
            "exact_mismatch_count": expected_count - exact_matches,
        },
        "mechanism": {
            "v31_raw": mechanism_summary(v31_rows),
            "v32_centered": mechanism_summary(v32_rows),
        },
        "dominant_calibration_memory_payloads": hub_payloads(
            v31_calibration=v31_calibration,
            v32_calibration=v32_calibration,
            record_by_id=record_by_id,
        ),
        "inputs": {
            "v31_results_sha256": file_sha256(args.v31_results),
            "v31_profile_sha256": file_sha256(args.v31_profile),
            "v31_calibration_sha256": file_sha256(args.v31_calibration),
            "v32_results_sha256": file_sha256(args.v32_results),
            "v32_profile_sha256": file_sha256(args.v32_profile),
            "v32_calibration_sha256": file_sha256(args.v32_calibration),
            "memory_records_sha256": file_sha256(args.memory_records),
        },
        "interpretation": (
            "matched_dev_retrieval_transform_comparison_not_independent_"
            "final_confirmation"
        ),
    }
    report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in report.items() if key != "created_at"
    })
    write_json_atomic(args.output, report)
    args.output.with_suffix(".md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(
        f"[v3.2-compare] samples={expected_count} integrity=true "
        f"strict_delta={report['paired_v32_minus_v31']['strict']['mean_treatment_minus_control']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
