"""Pure contracts for the answer-blind V3.3 retrieval-pooling audit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256


V3_POOLING_AUDIT_SCHEMA = "experience-memory-v3-pooling-audit-v1"
V3_POOLING_SAMPLE_SCHEMA = "experience-memory-v3-pooling-sample-v1"
V3_POOLING_EMBEDDING_SCHEMA = "experience-memory-v3-pooling-embeddings-v1"

V3_POOLING_BASELINE = "key_last__query_boundary_last"
V3_POOLING_PRE_BOUNDARY = "key_last__query_pre_boundary"
V3_POOLING_PARTIAL_MEAN = "key_mean__query_partial_mean"
V3_POOLING_FULL_MEAN = "key_mean__query_full_mean"
V3_POOLING_CANDIDATES = (
    V3_POOLING_BASELINE,
    V3_POOLING_PRE_BOUNDARY,
    V3_POOLING_PARTIAL_MEAN,
    V3_POOLING_FULL_MEAN,
)


def reconstruct_first_attempt_prefix(
    *,
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
    generated_boundary_index: int,
    query_audit: Mapping[str, Any],
) -> tuple[int, ...]:
    """Rebuild and authenticate the exact full-prefix retrieval query tokens."""

    prompt = tuple(int(value) for value in prompt_token_ids)
    completion = tuple(int(value) for value in completion_token_ids)
    partial_count = int(generated_boundary_index) + 1
    if partial_count < 2 or partial_count > len(completion):
        raise ValueError("Pooling source boundary is invalid")
    prefix = prompt + completion[:partial_count]
    if (
        len(prefix) != int(query_audit.get("query_token_count", -1))
        or len(prompt) != int(query_audit.get("prompt_token_count", -1))
        or partial_count
        != int(query_audit.get("partial_cot_token_count", -1))
        or canonical_json_sha256(list(prefix))
        != query_audit.get("query_token_ids_sha256")
    ):
        raise ValueError("Pooling first-attempt prefix reconstruction failed")
    return prefix


def stable_top_indices(
    scores: Sequence[float], *, top_k: int
) -> tuple[int, ...]:
    """Rank descending scores with bank order as the deterministic tie break."""

    if top_k <= 0:
        raise ValueError("Pooling audit top-k must be positive")
    normalized = [float(value) for value in scores]
    if not normalized:
        return ()
    return tuple(sorted(
        range(len(normalized)),
        key=lambda index: (-normalized[index], index),
    )[:top_k])


def qualify_pooling_candidate(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen answer-blind geometry gate to one candidate."""

    requirements = {
        "top1_share_decreased": (
            float(candidate["top1_share"]) < float(baseline["top1_share"])
        ),
        "selection_gini_decreased": (
            float(candidate["gini"]) < float(baseline["gini"])
        ),
        "selected_memory_count_not_lower": (
            int(candidate["selected_memory_count"])
            >= int(baseline["selected_memory_count"])
        ),
        "normalized_entropy_increased": (
            float(candidate["normalized_entropy"])
            > float(baseline["normalized_entropy"])
        ),
    }
    return {
        "qualified": all(requirements.values()),
        "requirements": requirements,
        "delta_vs_baseline": {
            "top1_share": (
                float(candidate["top1_share"])
                - float(baseline["top1_share"])
            ),
            "gini": float(candidate["gini"]) - float(baseline["gini"]),
            "selected_memory_count": (
                int(candidate["selected_memory_count"])
                - int(baseline["selected_memory_count"])
            ),
            "normalized_entropy": (
                float(candidate["normalized_entropy"])
                - float(baseline["normalized_entropy"])
            ),
        },
    }


def rank_qualified_pooling_candidates(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return qualified non-baseline candidates in the frozen geometry order."""

    if V3_POOLING_BASELINE not in summaries:
        raise ValueError("Pooling audit summaries have no reproduced baseline")
    baseline = summaries[V3_POOLING_BASELINE]
    qualified = []
    for name in V3_POOLING_CANDIDATES:
        if name == V3_POOLING_BASELINE:
            continue
        if name not in summaries:
            raise ValueError(f"Pooling audit summary is missing {name}")
        result = qualify_pooling_candidate(
            baseline=baseline,
            candidate=summaries[name],
        )
        if result["qualified"]:
            qualified.append(name)
    return tuple(sorted(
        qualified,
        key=lambda name: (
            float(summaries[name]["gini"]),
            float(summaries[name]["top1_share"]),
            -int(summaries[name]["selected_memory_count"]),
            name,
        ),
    ))
