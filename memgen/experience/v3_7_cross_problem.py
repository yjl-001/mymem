"""Pure contracts for the cross-problem causal applicability audit.

The audit deliberately separates retrieval from treatment.  Retrieval scores
are computed without task labels, while a frozen reasoner/side-KV branch later
assigns each evaluated query-memory pair a causal utility in ``{-1, 0, 1}``.
These helpers contain only deterministic ranking and aggregation logic so they
can be tested without a model, GPU, dataset download, or answer access.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence


V37_CAUSAL_PROFILE_SCHEMA = (
    "experience-memory-v3.7-cross-problem-causal-applicability-profile-v1"
)
V37_CAUSAL_QUERY_SCHEMA = (
    "experience-memory-v3.7-cross-problem-causal-applicability-query-v1"
)
V37_CAUSAL_TREATMENT_SCHEMA = (
    "experience-memory-v3.7-cross-problem-causal-applicability-treatment-v1"
)
V37_CAUSAL_REPORT_SCHEMA = (
    "experience-memory-v3.7-cross-problem-causal-applicability-report-v1"
)

V37_RETRIEVAL_VARIANTS = (
    "state_current",
    "state_delta",
    "state_local16",
    "text_applicability",
    "rrf_local16_delta",
)
V37_STATE_COMPONENT_BY_VARIANT = {
    "state_current": "current_token",
    "state_delta": "prompt_subtracted_delta",
    "state_local16": "local_reasoning_window_16",
}
V37_PRIMARY_VARIANT = "state_local16"
V37_DYNAMIC_CONTROL = "state_delta"
V37_TEXT_CONTROL = "text_applicability"
V37_FUSION_VARIANT = "rrf_local16_delta"


def stable_rank(
    memory_ids: Sequence[str], scores: Sequence[float]
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Rank finite scores descending with a stable memory-ID tie break."""

    if len(memory_ids) != len(scores) or not memory_ids:
        raise ValueError("ranking requires equal non-empty memory IDs and scores")
    normalized_ids = tuple(str(memory_id) for memory_id in memory_ids)
    if any(not memory_id for memory_id in normalized_ids) or len(
        set(normalized_ids)
    ) != len(normalized_ids):
        raise ValueError("ranking memory IDs must be unique and non-empty")
    normalized_scores = tuple(float(score) for score in scores)
    if not all(math.isfinite(score) for score in normalized_scores):
        raise ValueError("ranking scores must be finite")
    ordered = tuple(
        memory_id
        for memory_id, _ in sorted(
            zip(normalized_ids, normalized_scores),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return ordered, {
        memory_id: index + 1 for index, memory_id in enumerate(ordered)
    }


def reciprocal_rank_fusion_scores(
    memory_ids: Sequence[str],
    left_ranks: Mapping[str, int],
    right_ranks: Mapping[str, int],
    *,
    rank_constant: int = 60,
) -> tuple[float, ...]:
    """Return label-free two-channel reciprocal-rank-fusion scores."""

    if rank_constant <= 0:
        raise ValueError("RRF rank constant must be positive")
    ids = tuple(str(memory_id) for memory_id in memory_ids)
    if set(left_ranks) != set(ids) or set(right_ranks) != set(ids):
        raise ValueError("RRF rank maps must cover the exact memory universe")
    expected = set(range(1, len(ids) + 1))
    if set(int(value) for value in left_ranks.values()) != expected or set(
        int(value) for value in right_ranks.values()
    ) != expected:
        raise ValueError("RRF inputs must be complete one-based rankings")
    return tuple(
        1.0 / (rank_constant + int(left_ranks[memory_id]))
        + 1.0 / (rank_constant + int(right_ranks[memory_id]))
        for memory_id in ids
    )


def candidate_union(
    rankings: Mapping[str, Sequence[str]],
    *,
    top_k: int,
    random_memory_ids: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Build the shared treatment pool and retain every selection source."""

    if top_k <= 0:
        raise ValueError("candidate top_k must be positive")
    if tuple(rankings) != V37_RETRIEVAL_VARIANTS:
        raise ValueError("candidate rankings use a non-canonical variant order")
    sources: dict[str, list[str]] = defaultdict(list)
    for variant, ordered in rankings.items():
        normalized = tuple(str(memory_id) for memory_id in ordered)
        if len(normalized) < top_k or len(set(normalized)) != len(normalized):
            raise ValueError(f"candidate ranking is invalid for {variant}")
        for memory_id in normalized[:top_k]:
            sources[memory_id].append(variant)
    normalized_random = tuple(str(memory_id) for memory_id in random_memory_ids)
    if len(set(normalized_random)) != len(normalized_random):
        raise ValueError("random candidate memory IDs must be unique")
    deterministic_ids = set(sources)
    for normalized in normalized_random:
        if not normalized:
            raise ValueError("random candidate memory ID is empty")
        if normalized in deterministic_ids:
            raise ValueError("random controls must be disjoint from retriever top-k")
        sources[normalized].append("random_control")
    ordered_pool = tuple(sorted(sources))
    return ordered_pool, {
        memory_id: tuple(sources[memory_id]) for memory_id in ordered_pool
    }


def causal_utility(*, baseline_reward: float, treatment_reward: float) -> int:
    """Return the strict binary-reward treatment effect."""

    baseline = float(baseline_reward)
    treatment = float(treatment_reward)
    if baseline not in {0.0, 1.0} or treatment not in {0.0, 1.0}:
        raise ValueError("causal applicability requires binary strict rewards")
    return int(treatment - baseline)


def summarize_causal_rows(
    *,
    query_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    candidate_top_k: int,
) -> dict[str, Any]:
    """Summarize complete shared-pool interventions without fitting a selector."""

    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be positive")
    query_by_id: dict[str, Mapping[str, Any]] = {}
    for row in query_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in query_by_id:
            raise ValueError("causal query rows have missing or duplicate sample IDs")
        query_by_id[sample_id] = row
    treatments_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for row in treatment_rows:
        sample_id = str(row.get("sample_id", ""))
        memory_id = str(row.get("memory_id", ""))
        pair = (sample_id, memory_id)
        if sample_id not in query_by_id or not memory_id or pair in seen_pairs:
            raise ValueError("causal treatment rows have an invalid query-memory pair")
        seen_pairs.add(pair)
        utility = causal_utility(
            baseline_reward=float(row["baseline_reward"]),
            treatment_reward=float(row["treatment_reward"]),
        )
        if int(row.get("causal_utility", 99)) != utility:
            raise ValueError("stored causal utility differs from strict rewards")
        treatments_by_query[sample_id].append(row)

    selected_count = len(query_rows)
    eligible_rows = [row for row in query_rows if row.get("gate_eligible") is True]
    eligible_ids = {str(row["sample_id"]) for row in eligible_rows}
    no_gate_count = selected_count - len(eligible_rows)
    for row in eligible_rows:
        sample_id = str(row["sample_id"])
        expected = tuple(str(value) for value in row["candidate_memory_ids"])
        observed = tuple(
            sorted(str(value["memory_id"]) for value in treatments_by_query[sample_id])
        )
        if tuple(sorted(expected)) != observed:
            raise ValueError(f"incomplete causal treatment pool for {sample_id}")
    unexpected = set(treatments_by_query) - eligible_ids
    if unexpected:
        raise ValueError("treatments exist for a query without an eligible gate")

    baseline_rewards = [float(row["baseline"]["strict_reward"]) for row in query_rows]
    eligible_baselines = [
        float(row["baseline"]["strict_reward"]) for row in eligible_rows
    ]
    pool_helpful_queries = 0
    pool_harmful_queries = 0
    oracle_rewards: list[float] = []
    random_utilities: list[int] = []
    helpful_pair_count = 0
    harmful_pair_count = 0
    neutral_pair_count = 0
    for query in eligible_rows:
        sample_id = str(query["sample_id"])
        rows = treatments_by_query[sample_id]
        utilities = [int(row["causal_utility"]) for row in rows]
        helpful_pair_count += sum(value == 1 for value in utilities)
        harmful_pair_count += sum(value == -1 for value in utilities)
        neutral_pair_count += sum(value == 0 for value in utilities)
        pool_helpful_queries += int(any(value == 1 for value in utilities))
        pool_harmful_queries += int(any(value == -1 for value in utilities))
        baseline_reward = float(query["baseline"]["strict_reward"])
        # The causal oracle may always abstain.  It is a ceiling over the
        # evaluated memory pool plus the no-memory baseline, never a policy
        # forced to inject a harmful memory.
        oracle_rewards.append(max(
            baseline_reward,
            *(float(row["treatment_reward"]) for row in rows),
        ))
        for row in rows:
            if "random_control" in tuple(row.get("candidate_sources", ())):
                random_utilities.append(int(row["causal_utility"]))

    variants: dict[str, Any] = {}
    for variant in V37_RETRIEVAL_VARIANTS:
        top1_rewards: list[float] = []
        top1_utilities: list[int] = []
        helpful_hit_by_k = {k: 0 for k in range(1, candidate_top_k + 1)}
        harmful_hit_by_k = {k: 0 for k in range(1, candidate_top_k + 1)}
        helpful_query_count = 0
        harmful_query_count = 0
        reciprocal_first_helpful: list[float] = []
        for query in eligible_rows:
            rows = treatments_by_query[str(query["sample_id"])]
            by_rank = sorted(
                rows, key=lambda row: int(row["rank_by_variant"][variant])
            )
            top1 = by_rank[0]
            if int(top1["rank_by_variant"][variant]) != 1:
                raise ValueError(f"top-1 treatment is missing for {variant}")
            top1_rewards.append(float(top1["treatment_reward"]))
            top1_utilities.append(int(top1["causal_utility"]))
            helpful_ranks = [
                int(row["rank_by_variant"][variant])
                for row in rows
                if int(row["causal_utility"]) == 1
            ]
            harmful_ranks = [
                int(row["rank_by_variant"][variant])
                for row in rows
                if int(row["causal_utility"]) == -1
            ]
            if helpful_ranks:
                helpful_query_count += 1
                best = min(helpful_ranks)
                reciprocal_first_helpful.append(1.0 / best)
                for k in helpful_hit_by_k:
                    helpful_hit_by_k[k] += int(best <= k)
            if harmful_ranks:
                harmful_query_count += 1
                best_harm = min(harmful_ranks)
                for k in harmful_hit_by_k:
                    harmful_hit_by_k[k] += int(best_harm <= k)

        count = len(eligible_rows)
        baseline_mean = sum(eligible_baselines) / count if count else 0.0
        baseline_failure_count = sum(value == 0.0 for value in eligible_baselines)
        baseline_success_count = sum(value == 1.0 for value in eligible_baselines)
        top1_helpful_count = sum(value == 1 for value in top1_utilities)
        top1_harmful_count = sum(value == -1 for value in top1_utilities)
        variants[variant] = {
            "eligible_query_count": count,
            "top1_accuracy": sum(top1_rewards) / count if count else 0.0,
            "top1_accuracy_uplift": (
                sum(top1_rewards) / count - baseline_mean if count else 0.0
            ),
            "top1_helpful_count": top1_helpful_count,
            "top1_helpful_fraction": (
                top1_helpful_count / count
                if count
                else 0.0
            ),
            "top1_helpful_fraction_of_baseline_failures": (
                top1_helpful_count / baseline_failure_count
                if baseline_failure_count
                else 0.0
            ),
            "top1_harmful_count": top1_harmful_count,
            "top1_harmful_fraction": (
                top1_harmful_count / count
                if count
                else 0.0
            ),
            "top1_harmful_fraction_of_baseline_successes": (
                top1_harmful_count / baseline_success_count
                if baseline_success_count
                else 0.0
            ),
            "top1_net_utility_mean": (
                sum(top1_utilities) / count if count else 0.0
            ),
            "pool_conditional_helpful_query_count": helpful_query_count,
            "pool_conditional_mrr_first_helpful": (
                sum(reciprocal_first_helpful) / helpful_query_count
                if helpful_query_count
                else 0.0
            ),
            "helpful_hit_at_k": {
                str(k): {
                    "count": helpful_hit_by_k[k],
                    "fraction_of_pool_helpful_queries": (
                        helpful_hit_by_k[k] / helpful_query_count
                        if helpful_query_count
                        else 0.0
                    ),
                }
                for k in helpful_hit_by_k
            },
            "pool_conditional_harmful_query_count": harmful_query_count,
            "harmful_hit_at_k": {
                str(k): {
                    "count": harmful_hit_by_k[k],
                    "fraction_of_pool_harmful_queries": (
                        harmful_hit_by_k[k] / harmful_query_count
                        if harmful_query_count
                        else 0.0
                    ),
                }
                for k in harmful_hit_by_k
            },
        }

    eligible_count = len(eligible_rows)
    baseline_accuracy = (
        sum(baseline_rewards) / selected_count if selected_count else 0.0
    )
    eligible_baseline_accuracy = (
        sum(eligible_baselines) / eligible_count if eligible_count else 0.0
    )
    eligible_failure_count = sum(value == 0.0 for value in eligible_baselines)
    eligible_success_count = sum(value == 1.0 for value in eligible_baselines)
    oracle_accuracy = (
        sum(oracle_rewards) / eligible_count if eligible_count else 0.0
    )
    return {
        "selected_query_count": selected_count,
        "gate_eligible_query_count": eligible_count,
        "no_gate_query_count": no_gate_count,
        "gate_coverage": eligible_count / selected_count if selected_count else 0.0,
        "baseline_accuracy_all_selected": baseline_accuracy,
        "baseline_accuracy_gate_eligible": eligible_baseline_accuracy,
        "baseline_failure_count_gate_eligible": eligible_failure_count,
        "baseline_success_count_gate_eligible": eligible_success_count,
        "evaluated_pool_oracle_accuracy_gate_eligible": oracle_accuracy,
        "evaluated_pool_oracle_uplift_gate_eligible": (
            oracle_accuracy - eligible_baseline_accuracy
        ),
        "evaluated_pool_any_helpful_query_count": pool_helpful_queries,
        "evaluated_pool_any_helpful_query_fraction": (
            pool_helpful_queries / eligible_count if eligible_count else 0.0
        ),
        "evaluated_pool_any_helpful_fraction_of_baseline_failures": (
            pool_helpful_queries / eligible_failure_count
            if eligible_failure_count
            else 0.0
        ),
        "evaluated_pool_any_harmful_query_count": pool_harmful_queries,
        "evaluated_pool_any_harmful_query_fraction": (
            pool_harmful_queries / eligible_count if eligible_count else 0.0
        ),
        "evaluated_pool_any_harmful_fraction_of_baseline_successes": (
            pool_harmful_queries / eligible_success_count
            if eligible_success_count
            else 0.0
        ),
        "treatment_pair_counts": {
            "helpful": helpful_pair_count,
            "harmful": harmful_pair_count,
            "neutral": neutral_pair_count,
            "total": helpful_pair_count + harmful_pair_count + neutral_pair_count,
        },
        "random_control": {
            "pair_count": len(random_utilities),
            "helpful_fraction": (
                sum(value == 1 for value in random_utilities) / len(random_utilities)
                if random_utilities
                else 0.0
            ),
            "harmful_fraction": (
                sum(value == -1 for value in random_utilities) / len(random_utilities)
                if random_utilities
                else 0.0
            ),
            "mean_utility": (
                sum(random_utilities) / len(random_utilities)
                if random_utilities
                else 0.0
            ),
        },
        "variants": variants,
        "interpretation_limits": {
            "oracle_is_evaluated_candidate_pool_only": True,
            "helpful_recall_is_pool_conditional": True,
            "full_bank_helpful_memories_not_exhaustively_treated": True,
        },
    }


__all__ = [
    "V37_CAUSAL_PROFILE_SCHEMA",
    "V37_CAUSAL_QUERY_SCHEMA",
    "V37_CAUSAL_REPORT_SCHEMA",
    "V37_CAUSAL_TREATMENT_SCHEMA",
    "V37_DYNAMIC_CONTROL",
    "V37_FUSION_VARIANT",
    "V37_PRIMARY_VARIANT",
    "V37_RETRIEVAL_VARIANTS",
    "V37_STATE_COMPONENT_BY_VARIANT",
    "V37_TEXT_CONTROL",
    "candidate_union",
    "causal_utility",
    "reciprocal_rank_fusion_scores",
    "stable_rank",
    "summarize_causal_rows",
]
