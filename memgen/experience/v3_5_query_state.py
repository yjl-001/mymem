"""Pure contracts for the V3.5 query-state decomposition audit.

The audit holds source anchors and key banks fixed while comparing four
pre-registered layer-24 query representations.  These helpers do not select a
runtime winner or change the formal V3.5 qualification.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from memgen.experience.v3_5_hubness import compare_variant_rows


V35_QUERY_STATE_REPORT_SCHEMA = (
    "experience-memory-v3.5-dynamic-query-state-decomposition-report-v1"
)
V35_QUERY_STATE_EVIDENCE_SCHEMA = (
    "experience-memory-v3.5-dynamic-query-state-decomposition-evidence-v1"
)
V35_QUERY_STATE_TENSOR_SCHEMA = (
    "experience-memory-v3.5-dynamic-query-state-decomposition-tensors-v1"
)

V35_QUERY_STATE_VARIANTS = (
    "prompt_boundary",
    "current_token",
    "prompt_subtracted_delta",
    "local_reasoning_window_16",
)
V35_QUERY_STATE_KEY_VARIANTS = (
    "applicability_key",
    "dynamic_key",
)
V35_QUERY_STATE_BASELINE = "current_token"
V35_QUERY_STATE_PRIMARY_KEY = "applicability_key"
V35_QUERY_STATE_PRIMARY_SIDE = "reference"
V35_QUERY_STATE_LOCAL_WINDOW = 16


def rank_correlation(
    left: Sequence[int], right: Sequence[int]
) -> float | None:
    """Return Pearson correlation over two already-ranked value sequences."""

    if len(left) != len(right):
        raise ValueError("query-state rank sequences have different lengths")
    if not left:
        return None
    x = tuple(float(value) for value in left)
    y = tuple(float(value) for value in right)
    if not all(math.isfinite(value) for value in x + y):
        raise ValueError("query-state ranks must be finite")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )
    x_scale = sum((value - x_mean) ** 2 for value in x)
    y_scale = sum((value - y_mean) ** 2 for value in y)
    denominator = math.sqrt(x_scale * y_scale)
    return numerator / denominator if denominator > 0.0 else None


def compare_query_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare paired query variants while keeping the key variant fixed."""

    def normalize(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **row,
            "tensor_name": str(row["anchor_tensor_name"]),
        }

    baseline = [normalize(row) for row in baseline_rows]
    candidate = [normalize(row) for row in candidate_rows]
    result = compare_variant_rows(baseline, candidate)

    def identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row["trajectory_side"]),
            str(row["anchor_tensor_name"]),
            str(row["memory_id"]),
        )

    baseline_by_id = {identity(row): row for row in baseline_rows}
    candidate_by_id = {identity(row): row for row in candidate_rows}
    ordered = sorted(baseline_by_id)
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("query-state variants do not cover identical anchors")
    baseline_ranks = [
        int(baseline_by_id[key]["own_memory_rank"]) for key in ordered
    ]
    candidate_ranks = [
        int(candidate_by_id[key]["own_memory_rank"]) for key in ordered
    ]
    top1_same_count = sum(
        str(baseline_by_id[key]["top1_memory_id"])
        == str(candidate_by_id[key]["top1_memory_id"])
        for key in ordered
    )
    result.update({
        "own_rank_correlation": rank_correlation(
            baseline_ranks, candidate_ranks
        ),
        "top1_same_count": top1_same_count,
        "top1_same_fraction": top1_same_count / len(ordered),
    })
    return result


__all__ = [
    "V35_QUERY_STATE_BASELINE",
    "V35_QUERY_STATE_EVIDENCE_SCHEMA",
    "V35_QUERY_STATE_KEY_VARIANTS",
    "V35_QUERY_STATE_LOCAL_WINDOW",
    "V35_QUERY_STATE_PRIMARY_KEY",
    "V35_QUERY_STATE_PRIMARY_SIDE",
    "V35_QUERY_STATE_REPORT_SCHEMA",
    "V35_QUERY_STATE_TENSOR_SCHEMA",
    "V35_QUERY_STATE_VARIANTS",
    "compare_query_rows",
    "rank_correlation",
]
