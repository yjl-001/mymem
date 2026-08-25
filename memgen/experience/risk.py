"""Deterministic contracts used to compile the entropy-risk gate artifact."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256


ENTROPY_RISK_ARTIFACT_SCHEMA = "entropy-risk-gate-artifact-v2"
RISK_ELIGIBLE_EXPERIENCE_TYPES = frozenset({
    "answer_correctness",
    "format_compliance",
    "mixed_or_unclassified_task_failure",
})
def approved_experiences(
    approved_records: Iterable[Mapping[str, Any]],
    experiences: Iterable[Mapping[str, Any]],
    *,
    allowed_experience_types: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join Pro-approved records to their verifier-backed trajectories."""

    allowed = set(allowed_experience_types or RISK_ELIGIBLE_EXPERIENCE_TYPES)
    if not allowed:
        raise ValueError("allowed_experience_types must not be empty")
    experience_by_id: dict[str, Mapping[str, Any]] = {}
    for experience in experiences:
        experience_id = str(experience.get("experience_id", ""))
        if not experience_id or experience_id in experience_by_id:
            raise ValueError(
                f"Missing or duplicate verified experience_id: {experience_id!r}"
            )
        experience_by_id[experience_id] = experience

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    skipped_types: Counter[str] = Counter()
    for record in approved_records:
        experience_id = str(record.get("experience_id", ""))
        if not experience_id or experience_id in seen_ids:
            raise ValueError(
                f"Missing or duplicate approved experience_id: {experience_id!r}"
            )
        seen_ids.add(experience_id)
        gate = record.get("ai_review_gate")
        if not isinstance(gate, Mapping) or gate.get("route") != "ai_approved":
            raise ValueError(f"{experience_id} is not an ai_approved bank record")
        if record.get("reference_evidence") != "verified_failure":
            raise ValueError(f"{experience_id} lacks verified_failure evidence")
        experience = experience_by_id.get(experience_id)
        if experience is None:
            raise ValueError(
                f"Approved record {experience_id} has no verified experience"
            )
        for key in ("provenance_sha256", "source", "student", "experience_type"):
            if record.get(key) != experience.get(key):
                raise ValueError(f"{experience_id} has mismatched {key}")
        if record.get("source_episode_ids") != {
            "target": experience.get("target_episode_id"),
            "reference": experience.get("reference_episode_id"),
        }:
            raise ValueError(f"{experience_id} has mismatched source_episode_ids")
        if experience.get("reference_evidence") != "verified_failure":
            raise ValueError(
                f"{experience_id} verified record lost failure provenance"
            )
        if experience.get("reference_verifier", {}).get("reward") != 0.0:
            raise ValueError(f"{experience_id} reference verifier is not a failure")
        experience_type = str(experience.get("experience_type", ""))
        if experience_type not in allowed:
            skipped_types[experience_type] += 1
            continue
        if not str(experience.get("trajectory", "")).strip() or not str(
            experience.get("reference_trajectory", "")
        ).strip():
            raise ValueError(f"{experience_id} has an empty trajectory")
        selected.append(dict(experience))

    if not selected:
        raise ValueError("No approved experiences remain after type selection")
    report = {
        "selected_count": len(selected),
        "selected_by_experience_type": dict(sorted(Counter(
            str(item["experience_type"]) for item in selected
        ).items())),
        "skipped_by_experience_type": dict(sorted(skipped_types.items())),
        "selection_provenance_sha256": canonical_json_sha256({
            "experience_ids": [item["experience_id"] for item in selected],
            "allowed_experience_types": sorted(allowed),
        }),
    }
    return selected, report


def stable_uniform(seed: int, *parts: str) -> float:
    """Return a platform-independent deterministic value in ``[0, 1)``."""

    material = "|".join([str(seed), *parts]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value / float(2**64)


def entropy_quantile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linear-interpolated entropy quantile."""

    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("entropy quantile requires values and q in [0, 1]")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("entropy values must be finite")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def entropy_transition_label(
    *,
    current_entropy: float,
    next_entropy: float | None,
    high_threshold: float,
    low_threshold: float,
) -> str | None:
    """Classify a high-entropy transition as recovery or persistence."""

    if not all(
        math.isfinite(value)
        for value in (current_entropy, high_threshold, low_threshold)
    ):
        raise ValueError("entropy thresholds must be finite")
    if high_threshold < low_threshold:
        raise ValueError("high_threshold must be at least low_threshold")
    if current_entropy < high_threshold or next_entropy is None:
        return None
    if not math.isfinite(next_entropy):
        raise ValueError("next_entropy must be finite when present")
    return "recovery" if next_entropy <= low_threshold else "persistence"


def deterministic_train_partition(
    identifier: str, *, seed: int, train_fraction: float
) -> bool:
    """Assign an entire experience to a stable fit/holdout partition."""

    if not identifier or not 0.0 < train_fraction < 1.0:
        raise ValueError("invalid risk partition configuration")
    return stable_uniform(seed, "phase2-risk-split", identifier) < train_fraction


def binary_roc_auc(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> float:
    """Compute tie-aware ROC AUC without an external metrics dependency."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and aligned")
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores)]
    if not all(math.isfinite(score) for _, score in pairs):
        raise ValueError("scores must be finite")
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both labels")
    ordered = sorted(enumerate(pairs), key=lambda item: item[1][1])
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1][1] == ordered[index][1][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    positive_rank_sum = sum(
        rank for rank, (label, _) in zip(ranks, pairs) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def binary_average_precision(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> float:
    """Compute tie-aware positive-class average precision."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and aligned")
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores)]
    if not all(math.isfinite(score) for _, score in pairs):
        raise ValueError("scores must be finite")
    positives = sum(label for label, _ in pairs)
    if positives == 0:
        raise ValueError("Average precision requires a positive label")
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    true_positives = 0
    false_positives = 0
    result = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        group = ordered[index:end]
        group_positives = sum(label for label, _ in group)
        true_positives += group_positives
        false_positives += len(group) - group_positives
        if group_positives:
            precision = true_positives / (true_positives + false_positives)
            result += (group_positives / positives) * precision
        index = end
    return result
