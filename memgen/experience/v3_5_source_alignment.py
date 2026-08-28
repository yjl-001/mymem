"""Pure contracts for the V3.5 dynamic source-state alignment audit.

The audit is deliberately separate from the formal V3.5 selector
qualification.  It asks whether an authenticated memory's dynamic key can be
recovered from the target/reference reasoning trajectories that produced that
memory.  No task answer, reward, accuracy, side-KV treatment, or online
selector threshold participates in these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence


V35_SOURCE_ALIGNMENT_REPORT_SCHEMA = (
    "experience-memory-v3.5-dynamic-source-alignment-report-v1"
)
V35_SOURCE_ALIGNMENT_EVIDENCE_SCHEMA = (
    "experience-memory-v3.5-dynamic-source-alignment-evidence-v1"
)
V35_SOURCE_ALIGNMENT_QUERY_SIDECAR_SCHEMA = (
    "experience-memory-v3.5-dynamic-source-alignment-queries-v1"
)

V35_SOURCE_ALIGNMENT_PRIMARY_ANCHOR = (
    "reference_first_counterfactual_joint_gate_event"
)
V35_SOURCE_ALIGNMENT_PERMUTATION_SEED = 3517
V35_SOURCE_ALIGNMENT_PERMUTATION_COUNT = 10_000
V35_SOURCE_ALIGNMENT_RECALL_KS = (1, 5, 10, 32)


def percentile_linear(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile requires values and q in [0, 1]")
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("percentile values must be finite")
    ordered = sorted(normalized)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stable_score_ranking(
    *, memory_ids: Sequence[str], scores: Sequence[float]
) -> tuple[int, ...]:
    """Rank indices by descending score and ascending memory ID."""

    ids = tuple(str(value) for value in memory_ids)
    values = tuple(float(value) for value in scores)
    if not ids or len(ids) != len(values):
        raise ValueError("ranking requires equal non-empty IDs and scores")
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("ranking memory IDs must be unique and non-empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ranking scores must be finite")
    return tuple(
        sorted(range(len(ids)), key=lambda index: (-values[index], ids[index]))
    )


def score_query(
    *,
    memory_ids: Sequence[str],
    scores: Sequence[float],
    own_memory_id: str,
    top_n: int = 5,
    include_rank_lookup: bool = True,
) -> dict[str, Any]:
    """Build the auditable rank summary for one source-state query."""

    ids = tuple(str(value) for value in memory_ids)
    values = tuple(float(value) for value in scores)
    if own_memory_id not in ids:
        raise ValueError("own memory is absent from the dynamic bank")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 2:
        raise ValueError("source alignment top_n must be an integer >= 2")
    ranking = stable_score_ranking(memory_ids=ids, scores=values)
    own_index = ids.index(own_memory_id)
    own_rank = ranking.index(own_index) + 1
    own_score = values[own_index]
    best_other_score = max(
        value for index, value in enumerate(values) if index != own_index
    )
    top_indices = ranking[: min(top_n, len(ranking))]
    hits = [
        {
            "memory_id": ids[index],
            "score": values[index],
            "rank": rank,
        }
        for rank, index in enumerate(top_indices, start=1)
    ]
    result = {
        "own_memory_id": own_memory_id,
        "own_memory_rank": own_rank,
        "own_memory_score": own_score,
        "own_minus_best_other_score": own_score - best_other_score,
        "top1_memory_id": hits[0]["memory_id"],
        "top1_score": hits[0]["score"],
        "top2_score": hits[1]["score"],
        "top1_top2_margin": hits[0]["score"] - hits[1]["score"],
        "top_hits": hits,
    }
    if include_rank_lookup:
        # The full rank lookup is used only while building the deterministic
        # permutation null and is intentionally omitted from evidence rows.
        result["rank_by_memory_id"] = {
            ids[index]: rank
            for rank, index in enumerate(ranking, start=1)
        }
    return result


@dataclass(frozen=True)
class CounterfactualGateObservation:
    """One native, teacher-forced pre-answer token observation."""

    reasoning_rank: int
    attention_entropy: float
    risk_score: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.reasoning_rank, bool)
            or not isinstance(self.reasoning_rank, int)
            or self.reasoning_rank < 0
        ):
            raise ValueError("reasoning rank must be a non-negative integer")
        if not all(
            math.isfinite(float(value))
            for value in (self.attention_entropy, self.risk_score)
        ):
            raise ValueError("counterfactual gate observations must be finite")


def counterfactual_attempts(
    observations: Sequence[CounterfactualGateObservation],
    *,
    high_entropy_threshold: float,
    low_entropy_threshold: float,
    risk_threshold: float,
    rearm_low_token_count: int = 2,
    maximum_attempts: int = 3,
) -> tuple[dict[str, Any], ...]:
    """Replay the frozen V3.4 control policy without any memory treatment.

    Every selected event consumes an attempt and disarms the gate.  Consecutive
    low-entropy tokens re-arm it, but the token completing re-arm cannot also
    trigger.  This produces deterministic source anchors while leaving the
    recorded target/reference trajectory unchanged.
    """

    thresholds = (
        float(high_entropy_threshold),
        float(low_entropy_threshold),
        float(risk_threshold),
    )
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("counterfactual gate thresholds must be finite")
    if thresholds[1] > thresholds[0]:
        raise ValueError("low entropy threshold exceeds high threshold")
    if (
        isinstance(rearm_low_token_count, bool)
        or rearm_low_token_count <= 0
        or isinstance(maximum_attempts, bool)
        or maximum_attempts <= 0
    ):
        raise ValueError("counterfactual gate counts must be positive")
    ordered = tuple(observations)
    if any(
        current.reasoning_rank >= following.reasoning_rank
        for current, following in zip(ordered, ordered[1:])
    ):
        raise ValueError("counterfactual observations must be strictly ordered")

    state = "ARMED"
    low_streak = 0
    attempts: list[dict[str, Any]] = []
    for observation in ordered:
        if state == "EXHAUSTED":
            break
        entropy = float(observation.attention_entropy)
        risk = float(observation.risk_score)
        if state == "DISARMED":
            if entropy <= thresholds[1]:
                low_streak += 1
                if low_streak >= rearm_low_token_count:
                    state = "ARMED"
                    low_streak = 0
            else:
                low_streak = 0
            # The token completing re-arm never triggers immediately.
            continue
        joint = entropy >= thresholds[0] and risk > thresholds[2]
        if not joint:
            continue
        attempt_number = len(attempts) + 1
        next_state = (
            "EXHAUSTED"
            if attempt_number >= maximum_attempts
            else "DISARMED"
        )
        attempts.append({
            "attempt_number": attempt_number,
            "reasoning_rank": observation.reasoning_rank,
            "attention_entropy": entropy,
            "risk_score": risk,
            "state_before": "ARMED",
            "state_after": next_state,
        })
        state = next_state
        low_streak = 0
    return tuple(attempts)


def rank_metrics(
    ranks: Sequence[int], *, memory_count: int
) -> dict[str, Any] | None:
    """Aggregate own-memory ranks without treating other memories as negatives."""

    if not ranks:
        return None
    if isinstance(memory_count, bool) or memory_count <= 1:
        raise ValueError("source alignment requires at least two memories")
    values = tuple(int(value) for value in ranks)
    if any(value < 1 or value > memory_count for value in values):
        raise ValueError("source alignment rank is outside the bank")
    result: dict[str, Any] = {
        "sample_count": len(values),
        "mrr": sum(1.0 / value for value in values) / len(values),
        "mean_own_memory_rank": sum(values) / len(values),
        "median_own_memory_rank": percentile_linear(values, 0.5),
        "p95_own_memory_rank": percentile_linear(values, 0.95),
        "maximum_own_memory_rank": max(values),
    }
    for k in V35_SOURCE_ALIGNMENT_RECALL_KS:
        effective = min(k, memory_count)
        result[f"recall_at_{k}"] = (
            sum(value <= effective for value in values) / len(values)
        )
        result[f"uniform_rank_reference_at_{k}"] = effective / memory_count
    return result


def permutation_null(
    rows: Sequence[Mapping[str, Any]],
    *,
    memory_count: int,
    iterations: int = V35_SOURCE_ALIGNMENT_PERMUTATION_COUNT,
    seed: int = V35_SOURCE_ALIGNMENT_PERMUTATION_SEED,
) -> dict[str, Any] | None:
    """Shuffle query-to-own-ID bindings while preserving the score geometry."""

    if not rows:
        return None
    if isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("permutation count must be positive")
    own_ids = [str(row.get("own_memory_id", "")) for row in rows]
    if any(not value for value in own_ids) or len(set(own_ids)) != len(own_ids):
        raise ValueError("permutation rows require unique own memory IDs")
    lookups: list[Mapping[str, Any]] = []
    observed_ranks: list[int] = []
    for row, own_id in zip(rows, own_ids):
        lookup = row.get("rank_by_memory_id")
        if not isinstance(lookup, Mapping) or own_id not in lookup:
            raise ValueError("permutation row has no authenticated rank lookup")
        lookups.append(lookup)
        observed_ranks.append(int(lookup[own_id]))

    observed = rank_metrics(observed_ranks, memory_count=memory_count)
    assert observed is not None
    metric_names = [
        *(f"recall_at_{k}" for k in V35_SOURCE_ALIGNMENT_RECALL_KS),
        "mrr",
    ]
    null_values = {name: [] for name in metric_names}
    rng = random.Random(int(seed))
    permuted = list(own_ids)
    for _ in range(iterations):
        rng.shuffle(permuted)
        ranks = [
            int(lookup[assigned_id])
            for lookup, assigned_id in zip(lookups, permuted)
        ]
        metrics = rank_metrics(ranks, memory_count=memory_count)
        assert metrics is not None
        for name in metric_names:
            null_values[name].append(float(metrics[name]))

    summaries: dict[str, Any] = {}
    for name in metric_names:
        values = null_values[name]
        actual = float(observed[name])
        summaries[name] = {
            "observed": actual,
            "null_mean": sum(values) / len(values),
            "null_p05": percentile_linear(values, 0.05),
            "null_p50": percentile_linear(values, 0.50),
            "null_p95": percentile_linear(values, 0.95),
            "one_sided_enrichment_p_value": (
                1 + sum(value >= actual for value in values)
            ) / (iterations + 1),
        }
    return {
        "policy": "shuffle_query_to_own_memory_binding_preserve_score_matrix",
        "seed": int(seed),
        "iterations": int(iterations),
        "sample_count": len(rows),
        "metrics": summaries,
    }


__all__ = [
    "CounterfactualGateObservation",
    "V35_SOURCE_ALIGNMENT_EVIDENCE_SCHEMA",
    "V35_SOURCE_ALIGNMENT_PERMUTATION_COUNT",
    "V35_SOURCE_ALIGNMENT_PERMUTATION_SEED",
    "V35_SOURCE_ALIGNMENT_PRIMARY_ANCHOR",
    "V35_SOURCE_ALIGNMENT_QUERY_SIDECAR_SCHEMA",
    "V35_SOURCE_ALIGNMENT_RECALL_KS",
    "V35_SOURCE_ALIGNMENT_REPORT_SCHEMA",
    "counterfactual_attempts",
    "percentile_linear",
    "permutation_null",
    "rank_metrics",
    "score_query",
    "stable_score_ranking",
]
