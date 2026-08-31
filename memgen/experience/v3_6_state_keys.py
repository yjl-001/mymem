"""Pure contracts for the V3.6 source-state retrieval-key audit.

The audit replaces isolated text keys with authenticated causal-prefix states
without changing the memory payload.  Reference/failure first-gate states are
the keys and paired target/success first-gate states are the queries.  The
same-question prompt representation is retained only as an explicit identity
control, so a high cross-trajectory score cannot silently be interpreted as
cross-problem transfer.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from memgen.experience.v3_5_hubness import compare_variant_rows
from memgen.experience.v3_5_query_state import rank_correlation


V36_STATE_KEY_REPORT_SCHEMA = (
    "experience-memory-v3.6-source-state-retrieval-key-report-v1"
)
V36_STATE_KEY_EVIDENCE_SCHEMA = (
    "experience-memory-v3.6-source-state-retrieval-key-evidence-v1"
)
V36_STATE_KEY_BANK_SCHEMA = (
    "experience-memory-v3.6-source-state-retrieval-key-bank-v1"
)

V36_STATE_KEY_VARIANTS = (
    "text_applicability__target_current_control",
    "state_prompt__target_prompt_identity_control",
    "state_current__target_current",
    "state_delta__target_delta",
    "state_local16__target_local16",
)
V36_STATE_KEY_PRIMARY_VARIANT = "state_current__target_current"
V36_STATE_KEY_TEXT_CONTROL = "text_applicability__target_current_control"
V36_STATE_KEY_IDENTITY_CONTROL = (
    "state_prompt__target_prompt_identity_control"
)
V36_STATE_KEY_TRAJECTORY_KEY_SIDE = "reference"
V36_STATE_KEY_TRAJECTORY_QUERY_SIDE = "target"


def compare_state_key_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two variants over the exact same paired target queries."""

    comparison = compare_variant_rows(baseline_rows, candidate_rows)

    def identity(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row["tensor_name"]), str(row["memory_id"])

    baseline = {identity(row): row for row in baseline_rows}
    candidate = {identity(row): row for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError("state-key variants do not cover identical queries")
    ordered = sorted(baseline)
    baseline_ranks = [
        int(baseline[key]["own_memory_rank"]) for key in ordered
    ]
    candidate_ranks = [
        int(candidate[key]["own_memory_rank"]) for key in ordered
    ]
    same_top1_count = sum(
        str(baseline[key]["top1_memory_id"])
        == str(candidate[key]["top1_memory_id"])
        for key in ordered
    )
    comparison.update({
        "own_rank_correlation": rank_correlation(
            baseline_ranks, candidate_ranks
        ),
        "top1_same_count": same_top1_count,
        "top1_same_fraction": same_top1_count / len(ordered),
    })
    return comparison


__all__ = [
    "V36_STATE_KEY_BANK_SCHEMA",
    "V36_STATE_KEY_EVIDENCE_SCHEMA",
    "V36_STATE_KEY_IDENTITY_CONTROL",
    "V36_STATE_KEY_PRIMARY_VARIANT",
    "V36_STATE_KEY_REPORT_SCHEMA",
    "V36_STATE_KEY_TEXT_CONTROL",
    "V36_STATE_KEY_TRAJECTORY_KEY_SIDE",
    "V36_STATE_KEY_TRAJECTORY_QUERY_SIDE",
    "V36_STATE_KEY_VARIANTS",
    "compare_state_key_rows",
]
