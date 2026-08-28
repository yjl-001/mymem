"""Pure contracts for the V3.5 dynamic-key component audit.

The audit keeps the authenticated runtime queries fixed and changes only the
aligned key matrix used for exact-cosine ranking.  Its three variants are
pre-registered: the applicability key, the current full dynamic key, and the
normalized paired difference from applicability to dynamic.  The difference
is a diagnostic direction, not a separately encoded decision key and not an
online selector proposal.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from memgen.experience.v3_5_hubness import compare_variant_rows


V35_KEY_COMPONENT_REPORT_SCHEMA = (
    "experience-memory-v3.5-dynamic-key-component-report-v1"
)
V35_KEY_COMPONENT_EVIDENCE_SCHEMA = (
    "experience-memory-v3.5-dynamic-key-component-evidence-v1"
)
V35_KEY_COMPONENT_TENSOR_SCHEMA = (
    "experience-memory-v3.5-dynamic-key-component-tensors-v1"
)

V35_KEY_COMPONENT_VARIANTS = (
    "applicability_key",
    "dynamic_key",
    "paired_decision_residual",
)
V35_KEY_COMPONENT_PRIMARY_SIDE = "reference"
V35_KEY_COMPONENT_CURRENT_VARIANT = "dynamic_key"


def pairwise_variant_comparisons(
    rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    sides: Sequence[str] = ("reference", "target"),
) -> dict[str, Any]:
    """Compare every fixed key component with explicit baseline/candidate roles."""

    if tuple(rows_by_variant) != V35_KEY_COMPONENT_VARIANTS:
        raise ValueError("key-component variant order drifted")
    result: dict[str, Any] = {}
    for baseline_index, baseline in enumerate(V35_KEY_COMPONENT_VARIANTS):
        for candidate in V35_KEY_COMPONENT_VARIANTS[baseline_index + 1 :]:
            name = f"{candidate}_versus_{baseline}"
            result[name] = {
                "baseline_variant": baseline,
                "candidate_variant": candidate,
                "by_side": {},
            }
            for side in sides:
                baseline_rows = [
                    row
                    for row in rows_by_variant[baseline]
                    if str(row["trajectory_side"]) == side
                ]
                candidate_rows = [
                    row
                    for row in rows_by_variant[candidate]
                    if str(row["trajectory_side"]) == side
                ]
                result[name]["by_side"][side] = compare_variant_rows(
                    baseline_rows, candidate_rows
                )
    return result


__all__ = [
    "V35_KEY_COMPONENT_CURRENT_VARIANT",
    "V35_KEY_COMPONENT_EVIDENCE_SCHEMA",
    "V35_KEY_COMPONENT_PRIMARY_SIDE",
    "V35_KEY_COMPONENT_REPORT_SCHEMA",
    "V35_KEY_COMPONENT_TENSOR_SCHEMA",
    "V35_KEY_COMPONENT_VARIANTS",
    "pairwise_variant_comparisons",
]
