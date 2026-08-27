#!/usr/bin/env python3
"""Calibrate the V3.3 margin selector from a qualified pooling audit."""

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
    V3_QUERY_POOLING_PRE_BOUNDARY,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_pooling import (
    V3_POOLING_AUDIT_SCHEMA,
    V3_POOLING_PRE_BOUNDARY,
    V3_POOLING_SAMPLE_SCHEMA,
)
from memgen.experience.v3_selector import (
    V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
    V3_MARGIN_SELECTOR_POLICY,
    calibration_artifact_sha256,
    numeric_summary,
    retained_margin_threshold,
    selection_concentration,
)


RETRIEVAL_KEY_BANK_SCHEMA = "experience-memory-retrieval-key-bank-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooling-audit", type=Path, required=True)
    parser.add_argument("--pooling-samples", type=Path, required=True)
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


def load_pooling_audit(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    actual_hash = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "report_sha256"}
    })
    qualification = value.get("qualification", {})
    candidate_qualification = qualification.get("candidates", {}).get(
        V3_POOLING_PRE_BOUNDARY, {}
    )
    requirements = value.get("requirements", {})
    if (
        value.get("schema_version") != V3_POOLING_AUDIT_SCHEMA
        or value.get("status") != "passed"
        or value.get("report_sha256") != actual_hash
        or value.get("task_accuracy_used") is not False
        or value.get("answer_or_reward_used") is not False
        or not requirements
        or not all(item is True for item in requirements.values())
        or qualification.get("recommended_candidate")
        != V3_POOLING_PRE_BOUNDARY
        or candidate_qualification.get("qualified") is not True
    ):
        raise ValueError("Pooling audit did not qualify the V3.3 candidate")
    specification = value.get("candidates", {}).get(
        V3_POOLING_PRE_BOUNDARY, {}
    ).get("specification", {})
    if (
        specification.get("key_pooling") != "last_valid_token"
        or specification.get("query_pooling")
        != V3_QUERY_POOLING_PRE_BOUNDARY
        or value.get("pooling_contract", {}).get("embedding_transform")
        != V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
    ):
        raise ValueError("Pooling audit candidate contract drifted")
    return value


def load_key_manifest(
    path: Path, *, expected_file_sha256: str
) -> tuple[dict[str, Any], list[str]]:
    if file_sha256(path) != expected_file_sha256:
        raise ValueError("Retrieval key bank differs from the pooling audit")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = value.get("manifest_sha256")
    actual_hash = canonical_json_sha256({
        key: item for key, item in value.items() if key != "manifest_sha256"
    })
    records = value.get("records", [])
    memory_ids = [str(item.get("memory_id", "")) for item in records]
    if (
        value.get("schema_version") != RETRIEVAL_KEY_BANK_SCHEMA
        or expected_hash != actual_hash
        or not memory_ids
        or any(not memory_id for memory_id in memory_ids)
        or len(set(memory_ids)) != len(memory_ids)
    ):
        raise ValueError("Invalid retrieval key manifest")
    return value, memory_ids


def load_candidate_samples(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    if file_sha256(path) != expected_file_sha256:
        raise ValueError("Pooling sample traces differ from the audit report")
    samples: list[dict[str, Any]] = []
    margins: list[float] = []
    memory_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = str(sample.get("sample_id", ""))
            actual_hash = canonical_json_sha256({
                key: item
                for key, item in sample.items()
                if key != "sample_sha256"
            })
            candidate = sample.get("candidates", {}).get(
                V3_POOLING_PRE_BOUNDARY, {}
            )
            hits = list(candidate.get("hits", []))
            margin = candidate.get("top1_top2_margin")
            scores = [float(hit.get("score", float("nan"))) for hit in hits]
            if (
                sample.get("schema_version") != V3_POOLING_SAMPLE_SCHEMA
                or sample.get("sample_sha256") != actual_hash
                or not sample_id
                or sample_id in seen
                or not candidate.get("query_embedding_sha256")
                or len(hits) < 2
                or [int(hit.get("rank", -1)) for hit in hits[:2]] != [1, 2]
                or not all(math.isfinite(score) for score in scores)
                or scores != sorted(scores, reverse=True)
                or margin is None
                or not math.isfinite(float(margin))
                or float(margin) < 0.0
                or not math.isclose(
                    float(margin), scores[0] - scores[1], abs_tol=1e-12
                )
            ):
                raise ValueError(
                    f"Invalid pre-boundary pooling sample at line {line_number}"
                )
            seen.add(sample_id)
            samples.append(sample)
            margins.append(float(margin))
            memory_ids.append(str(hits[0]["memory_id"]))
    return samples, margins, memory_ids


def markdown_report(value: Mapping[str, Any]) -> str:
    calibration = value["calibration"]
    concentration = value["first_attempt_selection_concentration"]
    return "\n".join([
        "# MemGen V3.3 pre-boundary margin selector calibration",
        "",
        f"- Status: `{value['status']}`",
        f"- Source split: `{value['source']['logical_split']}`",
        f"- Source candidate: `{value['source']['pooling_candidate']}`",
        f"- Query pooling: `{value['source']['query_pooling']}`",
        f"- First-attempt sample count: {calibration['sample_count']}",
        f"- Minimum top1-top2 margin: `{calibration['minimum_top1_top2_margin']}`",
        f"- Target retained fraction: {calibration['target_retained_fraction']}",
        f"- Actual retained fraction: {calibration['actual_retained_fraction']}",
        f"- First-memory top-1 share: {concentration['top1_share']}",
        f"- Selection Gini: {concentration['gini']}",
        f"- Selected memories: {concentration['selected_memory_count']}",
        "",
        "This threshold is answer-blind and is derived only from the qualified calibration-val pooling geometry audit.",
        "",
    ])


def main() -> None:
    args = parse_args()
    if (
        not 0.0 < args.target_retained_fraction <= 1.0
        or args.minimum_triggered_samples <= 0
    ):
        raise ValueError("Invalid V3.3 selector calibration limits")
    audit = load_pooling_audit(args.pooling_audit)
    sample_artifact = audit.get("artifacts", {}).get("sample_traces", {})
    if sample_artifact.get("path") != args.pooling_samples.name:
        raise ValueError("Pooling sample path differs from the audit report")
    _, complete_memory_ids = load_key_manifest(
        args.retrieval_key_manifest,
        expected_file_sha256=str(
            audit.get("inputs", {}).get("retrieval_key_manifest_sha256", "")
        ),
    )
    samples, margins, selected_memory_ids = load_candidate_samples(
        args.pooling_samples,
        expected_file_sha256=str(sample_artifact.get("sha256", "")),
    )
    if (
        len(samples) != int(audit.get("sample_count", -1))
        or len(margins) < args.minimum_triggered_samples
    ):
        raise ValueError("Pooling audit sample set is incomplete")
    concentration = selection_concentration(
        selected_memory_ids, complete_memory_ids=complete_memory_ids
    )
    expected_candidate = audit["candidates"][V3_POOLING_PRE_BOUNDARY]
    if (
        concentration != expected_candidate["selection_concentration"]
        or numeric_summary(margins)
        != expected_candidate["top1_top2_margin"]
    ):
        raise ValueError("Pooling sample traces do not reproduce audit geometry")
    threshold = retained_margin_threshold(
        margins,
        target_retained_fraction=args.target_retained_fraction,
    )
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
                "scripts/calibrate_v3_pre_boundary_selector.py": file_sha256(
                    PROJECT_ROOT
                    / "scripts/calibrate_v3_pre_boundary_selector.py"
                ),
            },
        },
        "source": {
            "logical_split": "calibration-val",
            "scope": "first_retrieval_attempt_per_triggered_question",
            "pooling_candidate": V3_POOLING_PRE_BOUNDARY,
            "query_pooling": V3_QUERY_POOLING_PRE_BOUNDARY,
            "retrieval_embedding_transform": (
                V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
            ),
            "retrieval_key_manifest_sha256": file_sha256(
                args.retrieval_key_manifest
            ),
            "pooling_audit_report_sha256": file_sha256(args.pooling_audit),
            "pooling_audit_artifact_sha256": audit["report_sha256"],
            "pooling_sample_traces_sha256": file_sha256(
                args.pooling_samples
            ),
            "completed_sample_count": int(
                audit.get("source", {}).get("source_row_count", -1)
            ),
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
        "first_attempt_selection_concentration": concentration,
        "requirements": {
            "source_is_calibration_val": True,
            "source_pooling_audit_is_authenticated": True,
            "source_pooling_candidate_is_qualified_and_recommended": True,
            "source_sample_traces_are_authenticated_and_complete": True,
            "first_attempt_only": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "retrieval_key_manifest_is_bound": True,
            "retrieval_embedding_transform_is_bound": True,
            "query_pooling_is_bound": True,
            "threshold_is_finite_and_nonnegative": True,
        },
    }
    artifact["artifact_sha256"] = calibration_artifact_sha256(artifact)
    write_json_atomic(args.output, artifact)
    args.output.with_suffix(".md").write_text(
        markdown_report(artifact), encoding="utf-8"
    )
    print(
        f"[v3.3-calibration] first_attempts={len(margins)} "
        f"threshold={threshold['threshold']:.9g} "
        f"retained={threshold['actual_retained_fraction']:.4f} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
