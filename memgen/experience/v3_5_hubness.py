"""Pure reporting contracts for the V3.5 dynamic hubness audit.

The corresponding experiment reuses authenticated first-gate query tensors
and dynamic keys.  It compares three pre-registered geometries without model
generation, answer/reward access, threshold fitting, or formal qualification.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from memgen.experience.v3_5_source_alignment import (
    V35_SOURCE_ALIGNMENT_PERMUTATION_COUNT,
    V35_SOURCE_ALIGNMENT_PERMUTATION_SEED,
    V35_SOURCE_ALIGNMENT_RECALL_KS,
    percentile_linear,
    permutation_null,
    rank_metrics,
)


V35_HUBNESS_REPORT_SCHEMA = (
    "experience-memory-v3.5-dynamic-hubness-decomposition-report-v1"
)
V35_HUBNESS_EVIDENCE_SCHEMA = (
    "experience-memory-v3.5-dynamic-hubness-decomposition-evidence-v1"
)
V35_HUBNESS_TRANSFORM_SCHEMA = (
    "experience-memory-v3.5-dynamic-hubness-transforms-v1"
)

V35_HUBNESS_VARIANTS = (
    "raw",
    "key_centroid_centered",
    "key_centroid_centered_remove_pc1",
)
V35_HUBNESS_PRIMARY_SIDE = "reference"


def numeric_summary(values: Sequence[float]) -> dict[str, Any] | None:
    """Summarize finite values using deterministic linear percentiles."""

    normalized = tuple(float(value) for value in values)
    if not normalized:
        return None
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("hubness summary values must be finite")
    return {
        "count": len(normalized),
        "minimum": min(normalized),
        "p05": percentile_linear(normalized, 0.05),
        "median": percentile_linear(normalized, 0.50),
        "mean": sum(normalized) / len(normalized),
        "p95": percentile_linear(normalized, 0.95),
        "maximum": max(normalized),
    }


def selection_gini(counts: Sequence[int]) -> float:
    """Return the Gini coefficient over the complete key bank."""

    normalized = tuple(int(value) for value in counts)
    if any(value < 0 for value in normalized):
        raise ValueError("hubness counts must be non-negative")
    if not normalized or sum(normalized) == 0:
        return 0.0
    ordered = sorted(normalized)
    total = sum(ordered)
    count = len(ordered)
    return (
        sum(
            (2 * index - count - 1) * value
            for index, value in enumerate(ordered, start=1)
        )
        / (count * total)
    )


def selection_hubness(
    rows: Sequence[Mapping[str, Any]], memory_ids: Sequence[str]
) -> dict[str, Any]:
    """Measure top-1 concentration without treating other keys as negatives."""

    ids = tuple(str(value) for value in memory_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("hubness requires unique non-empty bank IDs")
    counts = Counter(str(row["top1_memory_id"]) for row in rows)
    if any(memory_id not in ids for memory_id in counts):
        raise ValueError("hubness row selected a key outside the bank")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = len(rows)
    top_two_count = sum(count for _, count in ordered[:2])
    return {
        "query_count": total,
        "selected_memory_count": len(counts),
        "selected_memory_fraction": len(counts) / len(ids),
        "top1_share": ordered[0][1] / total if total else 0.0,
        "top2_combined_share": top_two_count / total if total else 0.0,
        "selection_gini_over_full_bank": selection_gini(
            [counts.get(memory_id, 0) for memory_id in ids]
        ),
        "top_memories": [
            {
                "memory_id": memory_id,
                "top1_count": count,
                "top1_share": count / total,
            }
            for memory_id, count in ordered[:10]
        ] if total else [],
    }


def anchor_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    memory_ids: Sequence[str],
    permutation_count: int = V35_SOURCE_ALIGNMENT_PERMUTATION_COUNT,
) -> dict[str, Any]:
    """Summarize one geometry on one first-gate trajectory side."""

    ids = tuple(str(value) for value in memory_ids)
    partitions = {
        name: [
            int(row["own_memory_rank"])
            for row in rows
            if str(row[field]) == value
        ]
        for name, field, value in (
            ("selector_train", "selector_partition", "train"),
            ("selector_holdout", "selector_partition", "holdout"),
            ("risk_fit_train", "risk_partition", "train"),
            ("risk_fit_holdout", "risk_partition", "holdout"),
        )
    }
    return {
        "query_count": len(rows),
        "all": rank_metrics(
            [int(row["own_memory_rank"]) for row in rows],
            memory_count=len(ids),
        ),
        **{
            name: rank_metrics(ranks, memory_count=len(ids))
            for name, ranks in partitions.items()
        },
        "score_geometry": {
            field: numeric_summary([float(row[field]) for row in rows])
            for field in (
                "own_memory_score",
                "own_minus_best_other_score",
                "top1_top2_margin",
            )
        },
        "hubness": selection_hubness(rows, ids),
        "permutation_null": permutation_null(
            rows,
            memory_count=len(ids),
            iterations=permutation_count,
            seed=V35_SOURCE_ALIGNMENT_PERMUTATION_SEED,
        ),
    }


def compare_variant_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute paired deltas; positive rank delta means candidate improved."""

    def identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row["trajectory_side"]),
            str(row["tensor_name"]),
            str(row["memory_id"]),
        )

    raw = {identity(row): row for row in raw_rows}
    candidate = {identity(row): row for row in candidate_rows}
    if not raw_rows or not candidate_rows:
        raise ValueError("hubness comparison requires non-empty paired rows")
    if len(raw) != len(raw_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("hubness comparison rows must have unique identities")
    if raw.keys() != candidate.keys():
        raise ValueError("hubness variants do not cover identical queries")

    ordered = sorted(raw)
    rank_deltas = [
        int(raw[key]["own_memory_rank"])
        - int(candidate[key]["own_memory_rank"])
        for key in ordered
    ]
    gap_deltas = [
        float(candidate[key]["own_minus_best_other_score"])
        - float(raw[key]["own_minus_best_other_score"])
        for key in ordered
    ]
    raw_ranks = [int(raw[key]["own_memory_rank"]) for key in ordered]
    candidate_ranks = [
        int(candidate[key]["own_memory_rank"]) for key in ordered
    ]
    recall_deltas = {}
    for k in V35_SOURCE_ALIGNMENT_RECALL_KS:
        raw_recall = sum(rank <= k for rank in raw_ranks) / len(raw_ranks)
        candidate_recall = (
            sum(rank <= k for rank in candidate_ranks) / len(candidate_ranks)
        )
        recall_deltas[f"recall_at_{k}"] = candidate_recall - raw_recall
    return {
        "paired_query_count": len(ordered),
        "candidate_rank_improved_count": sum(value > 0 for value in rank_deltas),
        "candidate_rank_worsened_count": sum(value < 0 for value in rank_deltas),
        "rank_tie_count": sum(value == 0 for value in rank_deltas),
        "raw_minus_candidate_rank": numeric_summary(rank_deltas),
        "candidate_minus_raw_own_best_other_gap": numeric_summary(gap_deltas),
        "recall_delta_candidate_minus_raw": recall_deltas,
        "top1_recovered_count": sum(
            raw_rank != 1 and candidate_rank == 1
            for raw_rank, candidate_rank in zip(raw_ranks, candidate_ranks)
        ),
        "top1_lost_count": sum(
            raw_rank == 1 and candidate_rank != 1
            for raw_rank, candidate_rank in zip(raw_ranks, candidate_ranks)
        ),
    }


__all__ = [
    "V35_HUBNESS_EVIDENCE_SCHEMA",
    "V35_HUBNESS_PRIMARY_SIDE",
    "V35_HUBNESS_REPORT_SCHEMA",
    "V35_HUBNESS_TRANSFORM_SCHEMA",
    "V35_HUBNESS_VARIANTS",
    "anchor_summary",
    "compare_variant_rows",
    "numeric_summary",
    "selection_gini",
    "selection_hubness",
]
