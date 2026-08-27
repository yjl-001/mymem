"""Answer-blind calibration contracts for the V3.1 retrieval selector."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v3 import (
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_METHODS,
)


V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA = (
    "experience-memory-v3-margin-selector-calibration-v1"
)
V3_MARGIN_SELECTOR_POLICY = "top1_top2_margin"


def percentile_linear(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("Percentile requires values and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("Selector calibration values must be finite")
    return {
        "count": len(normalized),
        "min": min(normalized),
        "mean": sum(normalized) / len(normalized),
        "median": percentile_linear(normalized, 0.5),
        "p05": percentile_linear(normalized, 0.05),
        "p25": percentile_linear(normalized, 0.25),
        "p75": percentile_linear(normalized, 0.75),
        "p95": percentile_linear(normalized, 0.95),
        "max": max(normalized),
    }


def retained_margin_threshold(
    margins: Sequence[float], *, target_retained_fraction: float
) -> dict[str, Any]:
    """Choose a deterministic answer-blind threshold from first-attempt margins."""

    if not margins:
        raise ValueError("Margin calibration requires triggered samples")
    if not 0.0 < target_retained_fraction <= 1.0:
        raise ValueError("Target retained fraction must be in (0, 1]")
    values = [float(value) for value in margins]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Retrieval margins must be finite and non-negative")
    desired_count = max(1, int(math.ceil(
        len(values) * target_retained_fraction
    )))
    threshold = sorted(values, reverse=True)[desired_count - 1]
    retained_count = sum(value >= threshold for value in values)
    return {
        "threshold": threshold,
        "target_retained_fraction": target_retained_fraction,
        "target_retained_count": desired_count,
        "actual_retained_count": retained_count,
        "actual_retained_fraction": retained_count / len(values),
        "tie_policy": "retain_margin_greater_than_or_equal_to_threshold",
    }


def gini_coefficient(values: Sequence[int | float]) -> float:
    normalized = sorted(float(value) for value in values)
    if not normalized or any(value < 0.0 for value in normalized):
        raise ValueError("Gini values must be a non-empty non-negative sequence")
    total = sum(normalized)
    if total == 0.0:
        return 0.0
    count = len(normalized)
    weighted = sum(
        (index + 1) * value for index, value in enumerate(normalized)
    )
    return (2.0 * weighted) / (count * total) - (count + 1.0) / count


def selection_concentration(
    memory_ids: Sequence[str], *, complete_memory_ids: Sequence[str]
) -> dict[str, Any]:
    complete = [str(value) for value in complete_memory_ids]
    if not complete or len(set(complete)) != len(complete):
        raise ValueError("Complete selector memory IDs must be unique and non-empty")
    counts = Counter(str(value) for value in memory_ids)
    if any(memory_id not in set(complete) for memory_id in counts):
        raise ValueError("Selection contains an unknown memory ID")
    all_counts = [counts[memory_id] for memory_id in complete]
    total = sum(all_counts)
    probabilities = [value / total for value in all_counts if value > 0] if total else []
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized_entropy = (
        entropy / math.log(len(complete)) if len(complete) > 1 and total else 0.0
    )
    ranked = sorted(
        (
            {"memory_id": memory_id, "count": counts[memory_id]}
            for memory_id in complete
            if counts[memory_id]
        ),
        key=lambda item: (-int(item["count"]), str(item["memory_id"])),
    )
    return {
        "selection_count": total,
        "selected_memory_count": len(ranked),
        "bank_memory_count": len(complete),
        "gini": gini_coefficient(all_counts),
        "normalized_entropy": normalized_entropy,
        "top1_share": ranked[0]["count"] / total if total else None,
        "top5_share": (
            sum(int(item["count"]) for item in ranked[:5]) / total
            if total
            else None
        ),
        "top_by_frequency": ranked,
    }


def calibration_artifact_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "artifact_sha256"}
    })


def selector_calibration_query_pooling(value: Mapping[str, Any]) -> str:
    """Resolve pooling, treating pre-V3.3 artifacts as boundary-last only."""

    pooling = str(
        value.get("source", {}).get(
            "query_pooling", V3_QUERY_POOLING_BOUNDARY_LAST
        )
    )
    if pooling not in V3_QUERY_POOLING_METHODS:
        raise ValueError("Unexpected selector-calibration query pooling")
    return pooling


def load_margin_selector_calibration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("artifact_sha256")
    if expected != calibration_artifact_sha256(value):
        raise ValueError("V3 selector calibration hash mismatch")
    requirements = value.get("requirements", {})
    calibration = value.get("calibration", {})
    source = value.get("source", {})
    threshold = calibration.get("minimum_top1_top2_margin")
    query_pooling = selector_calibration_query_pooling(value)
    if (
        value.get("schema_version") != V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA
        or value.get("status") != "passed"
        or value.get("policy") != V3_MARGIN_SELECTOR_POLICY
        or value.get("task_accuracy_used") is not False
        or value.get("answer_or_reward_used") is not False
        or source.get("logical_split") != "calibration-val"
        or not requirements
        or not all(item is True for item in requirements.values())
        or threshold is None
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
        or query_pooling not in V3_QUERY_POOLING_METHODS
    ):
        raise ValueError("V3 selector calibration did not pass qualification")
    return value
