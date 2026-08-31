"""Pure contracts for the V3.8 failure-only full-bank causal audit.

V3.8 exhaustively treats every authenticated V3.6 state-key memory on each
V3.7 gate-eligible baseline failure.  The resulting binary utility matrix
separates three otherwise confounded losses: no helpful value in the audited
bank, failure to retrieve an existing helpful value into top-k, and failure to
place an already-retrieved helpful value at rank one.

These helpers are deliberately model-free.  They validate and summarize saved
treatment evidence without selecting an online retrieval variant or fitting a
threshold on the diagnostic outcomes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from memgen.experience.v3_7_cross_problem import V37_RETRIEVAL_VARIANTS


V38_PROFILE_SCHEMA = "experience-memory-v3.8-full-bank-causal-profile-v1"
V38_TREATMENT_SCHEMA = "experience-memory-v3.8-full-bank-causal-treatment-v1"
V38_UTILITY_MATRIX_SCHEMA = (
    "experience-memory-v3.8-full-bank-causal-utility-matrix-v1"
)
V38_REPORT_SCHEMA = "experience-memory-v3.8-full-bank-causal-report-v1"


def _validate_universe(
    values: Sequence[str], *, owner: str
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if (
        not normalized
        or any(not value for value in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError(f"{owner} must be non-empty, unique, and ordered")
    return normalized


def build_utility_matrix(
    *,
    query_ids: Sequence[str],
    memory_ids: Sequence[str],
    treatment_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a complete baseline-failure treatment Cartesian product."""

    queries = _validate_universe(query_ids, owner="query IDs")
    memories = _validate_universe(memory_ids, owner="memory IDs")
    query_set = set(queries)
    memory_set = set(memories)
    by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in treatment_rows:
        sample_id = str(row.get("sample_id", ""))
        memory_id = str(row.get("memory_id", ""))
        pair = (sample_id, memory_id)
        if (
            sample_id not in query_set
            or memory_id not in memory_set
            or pair in by_pair
        ):
            raise ValueError("treatment rows contain an invalid or duplicate pair")
        baseline = float(row.get("baseline_reward", float("nan")))
        treatment = float(row.get("treatment_reward", float("nan")))
        utility = int(row.get("causal_utility", 99))
        if baseline != 0.0 or treatment not in {0.0, 1.0}:
            raise ValueError("V3.8 requires strict baseline failures and binary reward")
        if utility != int(treatment) or utility not in {0, 1}:
            raise ValueError("V3.8 utility must equal treatment reward on failures")
        ranks = row.get("rank_by_variant")
        scores = row.get("score_by_variant")
        if (
            not isinstance(ranks, Mapping)
            or set(ranks) != set(V37_RETRIEVAL_VARIANTS)
            or not isinstance(scores, Mapping)
            or set(scores) != set(V37_RETRIEVAL_VARIANTS)
        ):
            raise ValueError("treatment rows have non-canonical retrieval evidence")
        if any(
            int(ranks[variant]) < 1
            or int(ranks[variant]) > len(memories)
            or not math.isfinite(float(scores[variant]))
            for variant in V37_RETRIEVAL_VARIANTS
        ):
            raise ValueError("treatment retrieval ranks or scores are invalid")
        by_pair[pair] = row

    expected_count = len(queries) * len(memories)
    if len(by_pair) != expected_count:
        missing = expected_count - len(by_pair)
        raise ValueError(f"full-bank treatment matrix is incomplete by {missing} pairs")

    utility_rows = [
        [int(by_pair[(sample_id, memory_id)]["causal_utility"]) for memory_id in memories]
        for sample_id in queries
    ]
    helpful_by_query = {
        sample_id: [
            memory_id
            for memory_id, utility in zip(memories, utility_rows[index])
            if utility == 1
        ]
        for index, sample_id in enumerate(queries)
    }
    helpful_by_memory = {
        memory_id: [
            sample_id
            for sample_id, utilities in zip(queries, utility_rows)
            if utilities[index] == 1
        ]
        for index, memory_id in enumerate(memories)
    }
    return {
        "query_ids": list(queries),
        "memory_ids": list(memories),
        "shape": [len(queries), len(memories)],
        "utilities": utility_rows,
        "helpful_memory_ids_by_query": helpful_by_query,
        "helped_query_ids_by_memory": helpful_by_memory,
    }


def _random_hit_probability(*, memory_count: int, helpful_count: int, k: int) -> float:
    if not (0 <= helpful_count <= memory_count and 0 <= k <= memory_count):
        raise ValueError("random hit probability received an invalid count")
    if helpful_count == 0 or k == 0:
        return 0.0
    if k > memory_count - helpful_count:
        return 1.0
    return 1.0 - (
        math.comb(memory_count - helpful_count, k) / math.comb(memory_count, k)
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[int]) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _dominant_gap(gaps: Mapping[str, int]) -> str:
    maximum = max(gaps.values(), default=0)
    winners = [name for name, count in gaps.items() if count == maximum]
    return winners[0] if len(winners) == 1 else "ambiguous_tie"


def summarize_full_bank_matrix(
    *,
    query_ids: Sequence[str],
    memory_ids: Sequence[str],
    treatment_rows: Sequence[Mapping[str, Any]],
    diagnosis_k: int,
) -> dict[str, Any]:
    """Summarize causal coverage and retrieval losses without model fitting."""

    queries = _validate_universe(query_ids, owner="query IDs")
    memories = _validate_universe(memory_ids, owner="memory IDs")
    if diagnosis_k <= 0 or diagnosis_k > len(memories):
        raise ValueError("diagnosis_k must be inside the memory universe")
    matrix = build_utility_matrix(
        query_ids=queries,
        memory_ids=memories,
        treatment_rows=treatment_rows,
    )
    rows_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in treatment_rows:
        rows_by_query[str(row["sample_id"])].append(row)

    helpful_counts = [
        len(matrix["helpful_memory_ids_by_query"][sample_id])
        for sample_id in queries
    ]
    recoverable_count = sum(count > 0 for count in helpful_counts)
    unrecoverable_count = len(queries) - recoverable_count
    helpful_pair_count = sum(helpful_counts)
    memory_help_counts = [
        len(matrix["helped_query_ids_by_memory"][memory_id])
        for memory_id in memories
    ]
    helpful_memory_count = sum(count > 0 for count in memory_help_counts)
    reusable_memory_count = sum(count > 1 for count in memory_help_counts)
    top_helpful_memories = sorted(
        (
            {
                "memory_id": memory_id,
                "helped_query_count": len(
                    matrix["helped_query_ids_by_memory"][memory_id]
                ),
            }
            for memory_id in memories
            if matrix["helped_query_ids_by_memory"][memory_id]
        ),
        key=lambda value: (-int(value["helped_query_count"]), str(value["memory_id"])),
    )

    requested_ks = tuple(sorted(
        {
            k
            for k in (1, 2, 3, 4, 5, 10, 32, diagnosis_k)
            if k <= len(memories)
        }
    ))
    variants: dict[str, Any] = {}
    dominant_by_variant: dict[str, str] = {}
    for variant in V37_RETRIEVAL_VARIANTS:
        hit_counts = {k: 0 for k in requested_ks}
        random_expected_hits = {k: 0.0 for k in requested_ks}
        reciprocal_ranks: list[float] = []
        first_helpful_ranks: list[int] = []
        per_query: dict[str, Any] = {}
        for sample_id, helpful_count in zip(queries, helpful_counts):
            helpful = set(matrix["helpful_memory_ids_by_query"][sample_id])
            ranks = {
                str(row["memory_id"]): int(row["rank_by_variant"][variant])
                for row in rows_by_query[sample_id]
            }
            if set(ranks) != set(memories) or set(ranks.values()) != set(
                range(1, len(memories) + 1)
            ):
                raise ValueError(f"{variant} is not a complete ranking for {sample_id}")
            best = min((ranks[memory_id] for memory_id in helpful), default=None)
            if best is not None:
                first_helpful_ranks.append(best)
                reciprocal_ranks.append(1.0 / best)
            for k in requested_ks:
                hit_counts[k] += int(best is not None and best <= k)
                random_expected_hits[k] += _random_hit_probability(
                    memory_count=len(memories), helpful_count=helpful_count, k=k
                )
            per_query[sample_id] = {
                "helpful_memory_count": helpful_count,
                "first_helpful_rank": best,
                "top1_helpful": best == 1,
                f"top{diagnosis_k}_contains_helpful": (
                    best is not None and best <= diagnosis_k
                ),
            }

        top1_hit = hit_counts[1]
        topk_hit = sum(
            value[f"top{diagnosis_k}_contains_helpful"]
            for value in per_query.values()
        )
        gap_counts = {
            "authenticated_bank_or_value_coverage": unrecoverable_count,
            "candidate_retrieval": recoverable_count - topk_hit,
            "top1_reranking": topk_hit - top1_hit,
        }
        dominant = _dominant_gap(gap_counts)
        dominant_by_variant[variant] = dominant
        variants[variant] = {
            "recoverable_query_count": recoverable_count,
            "mrr_first_helpful_on_recoverable": _mean(reciprocal_ranks),
            "mean_first_helpful_rank_on_recoverable": _mean(first_helpful_ranks),
            "median_first_helpful_rank_on_recoverable": _median(
                first_helpful_ranks
            ),
            "helpful_hit_at_k": {
                str(k): {
                    "count": hit_counts[k],
                    "fraction_of_all_failures": hit_counts[k] / len(queries),
                    "fraction_of_recoverable_failures": (
                        hit_counts[k] / recoverable_count
                        if recoverable_count
                        else 0.0
                    ),
                    "uniform_random_expected_count": random_expected_hits[k],
                    "uniform_random_expected_fraction_of_recoverable": (
                        random_expected_hits[k] / recoverable_count
                        if recoverable_count
                        else 0.0
                    ),
                    "observed_minus_uniform_random_expected_count": (
                        hit_counts[k] - random_expected_hits[k]
                    ),
                    "observed_over_uniform_random_expected": (
                        hit_counts[k] / random_expected_hits[k]
                        if random_expected_hits[k] > 0.0
                        else None
                    ),
                }
                for k in requested_ks
            },
            "pipeline_decomposition_at_diagnosis_k": {
                "diagnosis_k": diagnosis_k,
                "no_helpful_in_authenticated_bank": unrecoverable_count,
                "helpful_exists_but_missed_top_k": recoverable_count - topk_hit,
                "helpful_in_top_k_but_missed_top1": topk_hit - top1_hit,
                "helpful_at_top1": top1_hit,
                "partition_sum": len(queries),
            },
            "unresolved_gap_counts": gap_counts,
            "largest_observed_unresolved_gap": dominant,
            "per_query": per_query,
        }

    consensus_values = set(dominant_by_variant.values())
    cross_variant_consensus = (
        next(iter(consensus_values))
        if len(consensus_values) == 1
        else "variant_dependent"
    )
    origins = Counter(str(row.get("evidence_origin", "")) for row in treatment_rows)
    return {
        "failure_query_count": len(queries),
        "authenticated_state_key_memory_count": len(memories),
        "expected_treatment_pair_count": len(queries) * len(memories),
        "observed_treatment_pair_count": len(treatment_rows),
        "treatment_evidence_origins": dict(sorted(origins.items())),
        "causal_coverage": {
            "recoverable_failure_count": recoverable_count,
            "recoverable_failure_fraction": recoverable_count / len(queries),
            "no_helpful_memory_failure_count": unrecoverable_count,
            "no_helpful_memory_failure_fraction": unrecoverable_count / len(queries),
            "helpful_pair_count": helpful_pair_count,
            "helpful_pair_fraction": helpful_pair_count
            / (len(queries) * len(memories)),
            "helpful_memories_per_query": {
                "minimum": min(helpful_counts),
                "maximum": max(helpful_counts),
                "mean": _mean(helpful_counts),
                "median": _median(helpful_counts),
            },
            "memory_count_helpful_for_any_query": helpful_memory_count,
            "memory_count_helpful_for_multiple_queries": reusable_memory_count,
            "top_helpful_memories": top_helpful_memories,
        },
        "retrieval_variants": variants,
        "bottleneck_attribution": {
            "method": (
                "descriptive_mutually_exclusive_failure_partition_without_fitted_"
                "thresholds"
            ),
            "diagnosis_k": diagnosis_k,
            "largest_gap_by_variant": dominant_by_variant,
            "cross_variant_consensus": cross_variant_consensus,
            "does_not_select_variant": True,
            "does_not_fit_threshold": True,
        },
        "interpretation_limits": {
            "full_bank_means_authenticated_v36_state_key_universe": True,
            "memories_without_authenticated_state_keys_excluded": True,
            "failure_only_sweep_cannot_measure_harm": True,
            "non_helpful_may_reflect_value_or_fixed_injection_policy": True,
            "diagnostic_does_not_qualify_online_use": True,
        },
    }


__all__ = [
    "V38_PROFILE_SCHEMA",
    "V38_REPORT_SCHEMA",
    "V38_TREATMENT_SCHEMA",
    "V38_UTILITY_MATRIX_SCHEMA",
    "build_utility_matrix",
    "summarize_full_bank_matrix",
]
