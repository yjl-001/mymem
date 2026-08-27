#!/usr/bin/env python3
"""Qualify centered retrieval from answer-blind calibration hubness only."""

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

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3 import (
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_selector import load_margin_selector_calibration


QUALIFICATION_SCHEMA = "experience-memory-v3-centered-calibration-qualification-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31-calibration", type=Path, required=True)
    parser.add_argument("--v32-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def transform(value: Mapping[str, Any]) -> str:
    return str(value.get("source", {}).get(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    ))


def concentration(value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = value.get("first_attempt_selection_concentration")
    if not isinstance(result, Mapping):
        raise ValueError("Selector calibration has no concentration report")
    return result


def markdown_report(value: Mapping[str, Any]) -> str:
    raw = value["conditions"]["v31_raw"]
    centered = value["conditions"]["v32_centered"]
    lines = [
        "# MemGen V3.2 centered calibration qualification",
        "",
        f"- Status: `{value['status']}`",
        f"- Qualified for matched dev-test: "
        f"`{str(value['qualified_for_dev_test']).lower()}`",
        "",
        "| Metric | V3.1 raw | V3.2 centered | Delta | Improved |",
        "|---|---:|---:|---:|---:|",
        f"| First-memory top-1 share | {raw['top1_share']} | "
        f"{centered['top1_share']} | {value['delta_v32_minus_v31']['top1_share']} | "
        f"{value['requirements']['top1_share_decreased']} |",
        f"| Selection Gini | {raw['gini']} | {centered['gini']} | "
        f"{value['delta_v32_minus_v31']['gini']} | "
        f"{value['requirements']['selection_gini_decreased']} |",
        f"| Selected memories | {raw['selected_memory_count']} | "
        f"{centered['selected_memory_count']} | "
        f"{value['delta_v32_minus_v31']['selected_memory_count']} | diagnostic |",
        "",
        "Qualification is answer-blind and uses calibration-val retrieval traces only.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    v31 = load_margin_selector_calibration(args.v31_calibration)
    v32 = load_margin_selector_calibration(args.v32_calibration)
    if transform(v31) != V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE:
        raise ValueError("V3.1 calibration is not from the raw retrieval space")
    if transform(v32) != V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED:
        raise ValueError("V3.2 calibration is not from the centered retrieval space")
    v31_concentration = concentration(v31)
    v32_concentration = concentration(v32)
    if (
        v31_concentration.get("bank_memory_count")
        != v32_concentration.get("bank_memory_count")
    ):
        raise ValueError("Raw and centered calibrations use different memory banks")
    raw_top1 = float(v31_concentration["top1_share"])
    centered_top1 = float(v32_concentration["top1_share"])
    raw_gini = float(v31_concentration["gini"])
    centered_gini = float(v32_concentration["gini"])
    v31_key_bank_sha256 = v31.get("source", {}).get(
        "retrieval_key_manifest_sha256"
    )
    v32_key_bank_sha256 = v32.get("source", {}).get(
        "retrieval_key_manifest_sha256"
    )
    v31_completed_count = int(
        v31.get("source", {}).get("completed_sample_count", -1)
    )
    v32_completed_count = int(
        v32.get("source", {}).get("completed_sample_count", -1)
    )
    v31_first_attempt_count = int(
        v31.get("calibration", {}).get("sample_count", -1)
    )
    v32_first_attempt_count = int(
        v32.get("calibration", {}).get("sample_count", -1)
    )
    requirements = {
        "both_artifacts_answer_blind": (
            v31.get("task_accuracy_used") is False
            and v31.get("answer_or_reward_used") is False
            and v32.get("task_accuracy_used") is False
            and v32.get("answer_or_reward_used") is False
        ),
        "same_memory_bank": (
            bool(v31_key_bank_sha256)
            and v31_key_bank_sha256 == v32_key_bank_sha256
        ),
        "same_completed_calibration_sample_count": (
            v31_completed_count > 0
            and v31_completed_count == v32_completed_count
        ),
        "same_first_attempt_sample_count": (
            v31_first_attempt_count > 0
            and v31_first_attempt_count == v32_first_attempt_count
            and v31_first_attempt_count
            == int(v31_concentration["selection_count"])
            == int(v32_concentration["selection_count"])
        ),
        "same_target_retained_fraction": (
            v31.get("calibration", {}).get("target_retained_fraction")
            == v32.get("calibration", {}).get("target_retained_fraction")
        ),
        "top1_share_decreased": centered_top1 < raw_top1,
        "selection_gini_decreased": centered_gini < raw_gini,
    }
    qualified = all(requirements.values())
    report = {
        "schema_version": QUALIFICATION_SCHEMA,
        "created_at": utc_now(),
        "status": "qualified" if qualified else "not_qualified",
        "qualified_for_dev_test": qualified,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "conditions": {
            "v31_raw": dict(v31_concentration),
            "v32_centered": dict(v32_concentration),
        },
        "delta_v32_minus_v31": {
            "top1_share": centered_top1 - raw_top1,
            "gini": centered_gini - raw_gini,
            "selected_memory_count": (
                int(v32_concentration["selected_memory_count"])
                - int(v31_concentration["selected_memory_count"])
            ),
            "normalized_entropy": (
                float(v32_concentration["normalized_entropy"])
                - float(v31_concentration["normalized_entropy"])
            ),
        },
        "requirements": requirements,
        "inputs": {
            "v31_calibration_sha256": file_sha256(args.v31_calibration),
            "v32_calibration_sha256": file_sha256(args.v32_calibration),
        },
        "interpretation": "answer_blind_calibration_geometry_stop_gate",
    }
    report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in report.items() if key != "created_at"
    })
    write_json_atomic(args.output, report)
    args.output.with_suffix(".md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(
        f"[v3.2-calibration-qualification] status={report['status']} "
        f"top1_delta={report['delta_v32_minus_v31']['top1_share']} "
        f"gini_delta={report['delta_v32_minus_v31']['gini']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
