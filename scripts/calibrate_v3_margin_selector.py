#!/usr/bin/env python3
"""Build an answer-blind V3.1 margin threshold from calibration-val logs."""

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
from memgen.experience.v3 import (
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_METHODS,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    V3_RETRIEVAL_EMBEDDING_TRANSFORMS,
)
from memgen.experience.v3_selector import (
    V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
    V3_MARGIN_SELECTOR_POLICY,
    calibration_artifact_sha256,
    numeric_summary,
    retained_margin_threshold,
    selection_concentration,
)


EVALUATION_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
EVALUATION_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"
RETRIEVAL_KEY_BANK_SCHEMA = "experience-memory-retrieval-key-bank-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-profile", type=Path, required=True)
    parser.add_argument("--retrieval-key-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-retained-fraction", type=float, default=0.5
    )
    parser.add_argument("--minimum-triggered-samples", type=int, default=32)
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
        raise ValueError("Invalid V3 calibration run profile")
    system = value.get("system_profile", {})
    retrieval_transform = system.get(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    query_pooling = system.get(
        "query_pooling", V3_QUERY_POOLING_BOUNDARY_LAST
    )
    if (
        value.get("logical_split") != "calibration-val"
        or system.get("retrieval_abstention_policy") != "disabled"
        or system.get("retrieval_min_top1_top2_margin") not in {None, ""}
        or retrieval_transform not in V3_RETRIEVAL_EMBEDDING_TRANSFORMS
        or query_pooling not in V3_QUERY_POOLING_METHODS
    ):
        raise ValueError(
            "Selector calibration requires an abstention-disabled calibration-val run"
        )
    return value


def load_key_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("Retrieval key manifest differs from the calibration run")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("manifest_sha256")
    actual = canonical_json_sha256({
        key: item for key, item in value.items() if key != "manifest_sha256"
    })
    records = value.get("records", [])
    if (
        value.get("schema_version") != RETRIEVAL_KEY_BANK_SCHEMA
        or expected != actual
        or not records
    ):
        raise ValueError("Invalid retrieval key manifest")
    return value


def collect_first_attempts(
    path: Path, *, profile_sha256: str
) -> tuple[list[float], list[str], int]:
    margins: list[float] = []
    memory_ids: list[str] = []
    row_count = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            expected_row_hash = row.get("row_sha256")
            actual_row_hash = canonical_json_sha256({
                key: item
                for key, item in row.items()
                if key not in {"created_at", "row_sha256"}
            })
            if (
                row.get("schema_version") != EVALUATION_ROW_SCHEMA
                or row.get("profile_sha256") != profile_sha256
                or expected_row_hash != actual_row_hash
                or not sample_id
                or sample_id in seen
            ):
                raise ValueError(f"Invalid V3 row at line {line_number}")
            seen.add(sample_id)
            row_count += 1
            attempts = row["conditions"]["v3"]["runtime_trace"][
                "retrieval_attempts"
            ]
            if not attempts:
                continue
            first = attempts[0]
            decision = first["retrieval_decision"]
            query = decision["query"]
            margin = query.get("top1_top2_margin")
            selected_id = first.get("selected_memory_id")
            if (
                decision.get("status") != "selected"
                or selected_id is None
                or margin is None
                or not math.isfinite(float(margin))
                or float(margin) < 0.0
            ):
                raise ValueError(
                    f"Calibration first attempt is not a selected finite-margin result: {sample_id}"
                )
            margins.append(float(margin))
            memory_ids.append(str(selected_id))
    return margins, memory_ids, row_count


def markdown_report(value: Mapping[str, Any]) -> str:
    calibration = value["calibration"]
    concentration = value["first_attempt_selection_concentration"]
    lines = [
        "# MemGen V3 margin selector calibration",
        "",
        f"- Status: `{value['status']}`",
        f"- Source split: `{value['source']['logical_split']}`",
        f"- Retrieval embedding transform: "
        f"`{value['source']['retrieval_embedding_transform']}`",
        f"- Query pooling: `{value['source']['query_pooling']}`",
        f"- First-attempt sample count: {calibration['sample_count']}",
        f"- Minimum top1-top2 margin: `{calibration['minimum_top1_top2_margin']}`",
        f"- Target retained fraction: {calibration['target_retained_fraction']}",
        f"- Actual retained fraction: {calibration['actual_retained_fraction']}",
        f"- First-memory top-1 share: {concentration['top1_share']}",
        f"- Selection Gini: {concentration['gini']}",
        "",
        "This artifact is answer-blind: it reads first-attempt retrieval traces but not task accuracy, answers, or rewards.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if (
        not 0.0 < args.target_retained_fraction <= 1.0
        or args.minimum_triggered_samples <= 0
    ):
        raise ValueError("Invalid selector calibration limits")
    profile = load_profile(args.run_profile)
    key_manifest_sha256 = str(
        profile.get("inputs", {}).get("retrieval_key_manifest_sha256", "")
    )
    key_manifest = load_key_manifest(
        args.retrieval_key_manifest,
        expected_sha256=key_manifest_sha256,
    )
    retrieval_transform = profile.get("system_profile", {}).get(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    query_pooling = profile.get("system_profile", {}).get(
        "query_pooling", V3_QUERY_POOLING_BOUNDARY_LAST
    )
    margins, memory_ids, row_count = collect_first_attempts(
        args.results,
        profile_sha256=str(profile["profile_sha256"]),
    )
    if row_count != int(profile.get("selected_sample_count", -1)):
        raise ValueError("Calibration results are incomplete")
    if len(margins) < args.minimum_triggered_samples:
        raise ValueError("Too few triggered samples for selector calibration")
    threshold = retained_margin_threshold(
        margins,
        target_retained_fraction=args.target_retained_fraction,
    )
    complete_memory_ids = [
        str(item["memory_id"]) for item in key_manifest["records"]
    ]
    artifact = {
        "schema_version": V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
        "created_at": utc_now(),
        "status": "passed",
        "policy": V3_MARGIN_SELECTOR_POLICY,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "implementation": {
            "files_sha256": {
                "memgen/experience/v3_selector.py": file_sha256(
                    PROJECT_ROOT / "memgen/experience/v3_selector.py"
                ),
                "scripts/calibrate_v3_margin_selector.py": file_sha256(
                    PROJECT_ROOT / "scripts/calibrate_v3_margin_selector.py"
                ),
            },
        },
        "source": {
            "logical_split": "calibration-val",
            "scope": "first_retrieval_attempt_per_triggered_question",
            "run_profile_sha256": profile["profile_sha256"],
            "run_profile_file_sha256": file_sha256(args.run_profile),
            "results_file_sha256": file_sha256(args.results),
            "retrieval_key_manifest_sha256": key_manifest_sha256,
            "risk_artifact_sha256": profile.get("inputs", {}).get(
                "risk_artifact_sha256"
            ),
            "system_version": profile.get("system_version", "v3"),
            "system_profile_schema": profile.get("system_profile", {}).get(
                "schema_version"
            ),
            "retrieval_embedding_transform": retrieval_transform,
            "query_pooling": query_pooling,
            "completed_sample_count": row_count,
        },
        "calibration": {
            "sample_count": len(margins),
            "minimum_top1_top2_margin": threshold["threshold"],
            "target_retained_fraction": threshold[
                "target_retained_fraction"
            ],
            "target_retained_count": threshold["target_retained_count"],
            "actual_retained_count": threshold["actual_retained_count"],
            "actual_retained_fraction": threshold[
                "actual_retained_fraction"
            ],
            "tie_policy": threshold["tie_policy"],
            "margin_summary": numeric_summary(margins),
        },
        "first_attempt_selection_concentration": selection_concentration(
            memory_ids,
            complete_memory_ids=complete_memory_ids,
        ),
        "requirements": {
            "source_is_calibration_val": True,
            "source_profile_is_authenticated": True,
            "source_rows_are_complete_and_authenticated": True,
            "source_selector_has_abstention_disabled": True,
            "first_attempt_only": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "retrieval_key_manifest_is_bound": True,
            "risk_artifact_is_bound": True,
            "system_profile_is_bound": True,
            "retrieval_embedding_transform_is_bound": True,
            "query_pooling_is_bound": True,
            "threshold_is_finite_and_nonnegative": True,
        },
    }
    artifact["artifact_sha256"] = calibration_artifact_sha256(artifact)
    write_json_atomic(args.output, artifact)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(artifact), encoding="utf-8")
    print(
        f"[v3-calibration] first_attempts={len(margins)} "
        f"threshold={threshold['threshold']:.9g} "
        f"retained={threshold['actual_retained_fraction']:.4f} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
