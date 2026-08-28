"""Pure, answer-blind contracts for the MemGen V3.5 selector.

This module deliberately contains no model or task-evaluation code.  It owns
the deterministic source-pair split, positive-only applicability calibration,
first-attempt dynamic-margin calibration, logical artifact hashes, and the
fail-closed loaders consumed by V3.5 runtime entry points.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.risk import deterministic_train_partition
from memgen.experience.v3 import (
    ApplicabilityAwareRetrievalDecision,
    V35_RETRIEVAL_DECISION_SCHEMA,
    V35_SELECTOR_POLICY,
    V35_SYSTEM_PROFILE_SCHEMA,
)
from memgen.experience.v3_selector import (
    numeric_summary,
    percentile_linear,
)


V35_DUAL_KEY_BANK_SCHEMA = "experience-memory-v3.5-dual-key-bank-v1"
V35_APPLICABILITY_CALIBRATION_SCHEMA = (
    "experience-memory-v3.5-applicability-calibration-v1"
)
V35_SELECTOR_CALIBRATION_SCHEMA = (
    "experience-memory-v3.5-selector-calibration-v1"
)

V35_SOURCE_PAIR_PARTITION_SEED = 3501
V35_SOURCE_PAIR_TRAIN_FRACTION = 0.8
V35_MAX_SHORTLIST_K = 32
V35_MIN_TRAIN_OWN_MEMORY_RECALL = 0.95
V35_MIN_HELDOUT_OWN_MEMORY_RECALL = 0.95
V35_APPLICABILITY_SCORE_FLOOR_QUANTILE = 0.05
V35_MIN_HELDOUT_POSITIVE_RETENTION = 0.90
V35_DYNAMIC_TARGET_RETAINED_FRACTION = 0.50
V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY = (
    "retain_score_greater_than_or_equal_to_floor"
)
V35_DYNAMIC_MARGIN_TIE_POLICY = (
    "retain_margin_greater_than_or_equal_to_threshold"
)
V35_APPLICABILITY_FLOOR_ROLE = (
    "positive_retention_not_full_relevance_classifier"
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_V35_STATIC_QUESTION_QUERY_SCHEMA = (
    "experience-memory-v3.5-static-question-query-v1"
)
_V35_SOURCE_QUESTION_ENCODER = (
    "verified_experience.context.strip_question_only"
)

_V35_APPLICABILITY_REQUIREMENT_KEYS = frozenset({
    "train_partition_nonempty",
    "heldout_partition_nonempty",
    "minimal_shortlist_k_at_most_32_exists",
    "train_own_memory_recall_at_k_at_least_0_95",
    "heldout_own_memory_recall_at_k_at_least_0_95",
    "heldout_positive_retained_fraction_at_least_0_90",
    "positive_only_no_negative_labels",
    "task_accuracy_not_used",
    "answer_or_reward_not_used",
})
_V35_SELECTOR_REQUIREMENT_KEYS = frozenset({
    "source_is_calibration_val",
    "source_profile_is_authenticated",
    "source_rows_are_complete_and_authenticated",
    "source_is_trace_only",
    "first_attempt_only",
    "task_accuracy_not_used",
    "answer_or_reward_not_used",
    "dual_key_manifest_is_authenticated_and_bound",
    "applicability_calibration_is_authenticated_and_bound",
    "static_shortlist_is_fixed_and_bound",
    "dynamic_queries_are_full_prefix_and_authenticated",
    "dynamic_queries_disable_side_kv",
    "dynamic_rerank_is_inside_static_shortlist",
    "first_attempt_sample_count_sufficient",
    "insufficient_shortlist_fraction_acceptable",
    "threshold_is_finite_and_nonnegative",
    "inclusive_tie_policy",
})

_V35_APPLICABILITY_SOURCE_SHA256_FIELDS = frozenset({
    "memory_records_sha256",
    "side_kv_manifest_sha256",
    "e0_final_report_sha256",
    "v3_retrieval_key_manifest_sha256",
    "v3_retrieval_key_tensor_sha256",
    "v3_offline_report_sha256",
    "phase1_approved_bank_sha256",
    "verified_experiences_sha256",
    "split_manifest_sha256",
    "split_manifest_logical_sha256",
    "compiler_tracked_diff_sha256",
    "compiler_implementation_set_sha256",
    "dual_key_manifest_sha256",
    "dual_key_manifest_logical_sha256",
    "dual_key_tensor_sha256",
})
_V35_SELECTOR_SOURCE_SHA256_FIELDS = frozenset({
    "run_profile_sha256",
    "run_profile_file_sha256",
    "results_file_sha256",
    "dual_key_manifest_sha256",
    "dual_key_manifest_logical_sha256",
    "applicability_calibration_sha256",
    "applicability_calibration_artifact_sha256",
    "risk_artifact_sha256",
})


def v35_artifact_sha256(value: Mapping[str, Any]) -> str:
    """Return the stable logical hash of a V3.5 JSON artifact.

    Wall-clock creation time and the self-authenticating hash fields are not
    part of the logical identity.  Every calibration loader recomputes this
    value before inspecting qualification fields.
    """

    if not isinstance(value, Mapping):
        raise ValueError("V3.5 artifact must be a mapping")
    return canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "artifact_sha256", "logical_sha256"}
    })


# Explicit alias used by reports that call the authenticated identity a
# logical hash rather than an artifact hash.
v35_artifact_logical_sha256 = v35_artifact_sha256


def deterministic_source_pair_partition(
    memory_id: str,
    source_experience_id: str,
    *,
    seed: int = V35_SOURCE_PAIR_PARTITION_SEED,
    train_fraction: float = V35_SOURCE_PAIR_TRAIN_FRACTION,
) -> str:
    """Assign ``memory_id:source_experience_id`` to the frozen V3.5 split."""

    if seed != V35_SOURCE_PAIR_PARTITION_SEED or not math.isclose(
        train_fraction,
        V35_SOURCE_PAIR_TRAIN_FRACTION,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("V3.5 source-pair partition is frozen to 3501/0.8")
    memory_id = str(memory_id).strip()
    source_experience_id = str(source_experience_id).strip()
    if not memory_id or not source_experience_id or ":" in memory_id:
        raise ValueError("V3.5 source pair requires unambiguous non-empty IDs")
    identifier = f"{memory_id}:{source_experience_id}"
    return (
        "train"
        if deterministic_train_partition(
            identifier,
            seed=V35_SOURCE_PAIR_PARTITION_SEED,
            train_fraction=V35_SOURCE_PAIR_TRAIN_FRACTION,
        )
        else "holdout"
    )


def partition_source_pairs(
    source_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and deterministically partition one source pair per memory."""

    if not source_pairs:
        raise ValueError("V3.5 applicability calibration needs source pairs")
    seen_memory_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "holdout": [],
    }
    for item in source_pairs:
        if not isinstance(item, Mapping):
            raise ValueError("V3.5 source pairs must be mappings")
        memory_id = str(item.get("memory_id", "")).strip()
        source_id = str(item.get("source_experience_id", "")).strip()
        if not memory_id or not source_id:
            raise ValueError("V3.5 source pair IDs must be non-empty")
        if memory_id in seen_memory_ids or source_id in seen_source_ids:
            raise ValueError("V3.5 source-pair join must be one-to-one")
        seen_memory_ids.add(memory_id)
        seen_source_ids.add(source_id)
        split = deterministic_source_pair_partition(memory_id, source_id)
        normalized = dict(item)
        normalized["memory_id"] = memory_id
        normalized["source_experience_id"] = source_id
        partitions[split].append(normalized)
    for split in partitions:
        partitions[split].sort(key=lambda item: str(item["memory_id"]))
    return {
        "seed": V35_SOURCE_PAIR_PARTITION_SEED,
        "train_fraction": V35_SOURCE_PAIR_TRAIN_FRACTION,
        "train_memory_ids": [
            str(item["memory_id"]) for item in partitions["train"]
        ],
        "heldout_memory_ids": [
            str(item["memory_id"]) for item in partitions["holdout"]
        ],
        "train": tuple(partitions["train"]),
        "holdout": tuple(partitions["holdout"]),
    }


def _validated_ranks(
    ranks: Sequence[int], *, memory_count: int
) -> tuple[int, ...]:
    if isinstance(memory_count, bool) or memory_count <= 0:
        raise ValueError("V3.5 memory_count must be positive")
    if not ranks:
        raise ValueError("V3.5 own-memory ranks must be non-empty")
    values: list[int] = []
    for rank in ranks:
        if isinstance(rank, bool) or int(rank) != rank:
            raise ValueError("V3.5 own-memory ranks must be integers")
        normalized = int(rank)
        if not 1 <= normalized <= memory_count:
            raise ValueError("V3.5 own-memory rank is outside the bank")
        values.append(normalized)
    return tuple(values)


def own_memory_recall(ranks: Sequence[int], *, k: int) -> float:
    if isinstance(k, bool) or k <= 0:
        raise ValueError("Recall@k requires positive integer k")
    if not ranks:
        raise ValueError("Recall@k requires own-memory ranks")
    values = tuple(int(rank) for rank in ranks)
    if any(rank <= 0 for rank in values):
        raise ValueError("Own-memory ranks must be positive")
    return sum(rank <= k for rank in values) / len(values)


def select_minimal_shortlist_k(
    ranks: Sequence[int],
    *,
    memory_count: int,
    minimum_recall: float = V35_MIN_TRAIN_OWN_MEMORY_RECALL,
    maximum_k: int = V35_MAX_SHORTLIST_K,
) -> int | None:
    """Return the smallest k <= 32 reaching train own-memory Recall@k."""

    values = _validated_ranks(ranks, memory_count=memory_count)
    if not math.isclose(
        minimum_recall,
        V35_MIN_TRAIN_OWN_MEMORY_RECALL,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or maximum_k != V35_MAX_SHORTLIST_K:
        raise ValueError("V3.5 shortlist selection is frozen to Recall .95/k<=32")
    for k in range(1, min(maximum_k, memory_count) + 1):
        if own_memory_recall(values, k=k) >= minimum_recall:
            return k
    return None


def own_memory_rank_metrics(
    ranks: Sequence[int], *, memory_count: int, shortlist_k: int
) -> dict[str, Any]:
    """Report only positive own-memory ranking metrics (no false negatives)."""

    values = _validated_ranks(ranks, memory_count=memory_count)
    if isinstance(shortlist_k, bool) or not 1 <= shortlist_k <= memory_count:
        raise ValueError("V3.5 shortlist k is outside the memory bank")
    return {
        "sample_count": len(values),
        "recall_at_1": own_memory_recall(values, k=1),
        "recall_at_5": own_memory_recall(
            values, k=min(5, memory_count)
        ),
        "recall_at_10": own_memory_recall(
            values, k=min(10, memory_count)
        ),
        "recall_at_k": own_memory_recall(values, k=shortlist_k),
        "shortlist_k": shortlist_k,
        "mrr": sum(1.0 / rank for rank in values) / len(values),
        "median_own_memory_rank": percentile_linear(values, 0.5),
        "p95_own_memory_rank": percentile_linear(values, 0.95),
    }


def applicability_score_floor(
    positive_scores: Sequence[float],
    *,
    quantile: float = V35_APPLICABILITY_SCORE_FLOOR_QUANTILE,
) -> dict[str, Any]:
    """Freeze the inclusive fifth-percentile train-positive cosine floor."""

    if not math.isclose(
        quantile,
        V35_APPLICABILITY_SCORE_FLOOR_QUANTILE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("V3.5 applicability floor is frozen to p05")
    if not positive_scores:
        raise ValueError("V3.5 applicability floor needs train positives")
    scores = tuple(float(score) for score in positive_scores)
    if not all(math.isfinite(score) and -1.0 <= score <= 1.0 for score in scores):
        raise ValueError("V3.5 applicability scores must be finite cosines")
    floor = percentile_linear(scores, quantile)
    retained_count = sum(score >= floor for score in scores)
    return {
        "minimum_applicability_score": floor,
        "applicability_score_floor_quantile": quantile,
        "applicability_score_floor_tie_policy": (
            V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY
        ),
        "applicability_floor_role": V35_APPLICABILITY_FLOOR_ROLE,
        "positive_count": len(scores),
        "positive_retained_count": retained_count,
        "positive_retained_fraction": retained_count / len(scores),
        "positive_score_summary": numeric_summary(scores),
    }


def retained_dynamic_margin_threshold(
    margins: Sequence[float],
    *,
    target_retained_fraction: float = V35_DYNAMIC_TARGET_RETAINED_FRACTION,
) -> dict[str, Any]:
    """Freeze the inclusive answer-blind first-attempt 50% margin threshold."""

    if not math.isclose(
        target_retained_fraction,
        V35_DYNAMIC_TARGET_RETAINED_FRACTION,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("V3.5 dynamic retained fraction is frozen to 0.50")
    if not margins:
        raise ValueError("V3.5 dynamic calibration needs first attempts")
    values = tuple(float(margin) for margin in margins)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("V3.5 dynamic margins must be finite and non-negative")
    desired_count = max(
        1, int(math.ceil(len(values) * target_retained_fraction))
    )
    threshold = sorted(values, reverse=True)[desired_count - 1]
    retained_count = sum(value >= threshold for value in values)
    return {
        "minimum_dynamic_top1_top2_margin": threshold,
        "target_retained_fraction": target_retained_fraction,
        "target_retained_count": desired_count,
        "actual_retained_count": retained_count,
        "actual_retained_fraction": retained_count / len(values),
        "dynamic_margin_tie_policy": V35_DYNAMIC_MARGIN_TIE_POLICY,
        "first_attempt_count": len(values),
        "margin_summary": numeric_summary(values),
    }


def calibrate_applicability_selector(
    source_pairs: Sequence[Mapping[str, Any]], *, memory_count: int
) -> dict[str, Any]:
    """Build the pure static calibration/qualification payload.

    Each pair must contain ``memory_id``, ``source_experience_id``, the unique
    positive's ``own_memory_rank``, and ``own_positive_score``.  No answer,
    reward, task accuracy, or putative negative label is accepted or used.
    """

    if isinstance(memory_count, bool) or memory_count <= 0:
        raise ValueError("V3.5 memory_count must be positive")
    if len(source_pairs) != memory_count:
        raise ValueError(
            "V3.5 applicability calibration requires one source pair per memory"
        )
    forbidden = {
        "answer",
        "reward",
        "strict_correct",
        "format_correct",
        "task_accuracy",
    }
    for item in source_pairs:
        if forbidden.intersection(item):
            raise ValueError(
                "V3.5 applicability calibration must remain answer-blind"
            )
    partitioned = partition_source_pairs(source_pairs)
    train = partitioned["train"]
    holdout = partitioned["holdout"]
    train_ranks = tuple(int(item["own_memory_rank"]) for item in train)
    holdout_ranks = tuple(int(item["own_memory_rank"]) for item in holdout)
    train_scores = tuple(float(item["own_positive_score"]) for item in train)
    holdout_scores = tuple(float(item["own_positive_score"]) for item in holdout)

    # Validate every provided positive even if a very small fixture happens to
    # produce an empty deterministic side of the partition.
    all_ranks = tuple(
        int(item["own_memory_rank"])
        for item in (*train, *holdout)
    )
    _validated_ranks(all_ranks, memory_count=memory_count)
    all_scores = tuple(
        float(item["own_positive_score"])
        for item in (*train, *holdout)
    )
    if not all(
        math.isfinite(score) and -1.0 <= score <= 1.0
        for score in all_scores
    ):
        raise ValueError("V3.5 own-positive scores must be finite cosines")

    shortlist_k = (
        select_minimal_shortlist_k(
            train_ranks, memory_count=memory_count
        )
        if train_ranks
        else None
    )
    floor_result = (
        applicability_score_floor(train_scores) if train_scores else None
    )
    floor = (
        float(floor_result["minimum_applicability_score"])
        if floor_result is not None
        else None
    )
    heldout_retained_fraction = (
        sum(score >= floor for score in holdout_scores) / len(holdout_scores)
        if floor is not None and holdout_scores
        else None
    )
    train_metrics = (
        own_memory_rank_metrics(
            train_ranks,
            memory_count=memory_count,
            shortlist_k=shortlist_k,
        )
        if shortlist_k is not None and train_ranks
        else None
    )
    heldout_metrics = (
        own_memory_rank_metrics(
            holdout_ranks,
            memory_count=memory_count,
            shortlist_k=shortlist_k,
        )
        if shortlist_k is not None and holdout_ranks
        else None
    )
    requirements = {
        "train_partition_nonempty": bool(train),
        "heldout_partition_nonempty": bool(holdout),
        "minimal_shortlist_k_at_most_32_exists": shortlist_k is not None,
        "train_own_memory_recall_at_k_at_least_0_95": (
            train_metrics is not None
            and train_metrics["recall_at_k"]
            >= V35_MIN_TRAIN_OWN_MEMORY_RECALL
        ),
        "heldout_own_memory_recall_at_k_at_least_0_95": (
            heldout_metrics is not None
            and heldout_metrics["recall_at_k"]
            >= V35_MIN_HELDOUT_OWN_MEMORY_RECALL
        ),
        "heldout_positive_retained_fraction_at_least_0_90": (
            heldout_retained_fraction is not None
            and heldout_retained_fraction
            >= V35_MIN_HELDOUT_POSITIVE_RETENTION
        ),
        "positive_only_no_negative_labels": True,
        "task_accuracy_not_used": True,
        "answer_or_reward_not_used": True,
    }
    calibration = {
        "memory_count": memory_count,
        "shortlist_k": shortlist_k,
        "maximum_shortlist_k": min(V35_MAX_SHORTLIST_K, memory_count),
        "shortlist_bank_fraction": (
            shortlist_k / memory_count if shortlist_k is not None else None
        ),
        "minimum_applicability_score": floor,
        "applicability_score_floor_quantile": (
            V35_APPLICABILITY_SCORE_FLOOR_QUANTILE
        ),
        "applicability_score_floor_tie_policy": (
            V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY
        ),
        "applicability_floor_role": V35_APPLICABILITY_FLOOR_ROLE,
        "train_own_memory_recall_at_k": (
            train_metrics["recall_at_k"] if train_metrics else None
        ),
        "heldout_own_memory_recall_at_k": (
            heldout_metrics["recall_at_k"] if heldout_metrics else None
        ),
        "heldout_own_positive_retained_fraction": (
            heldout_retained_fraction
        ),
    }
    public_partition = {
        key: partitioned[key]
        for key in (
            "seed",
            "train_fraction",
            "train_memory_ids",
            "heldout_memory_ids",
        )
    }
    return {
        "status": (
            "passed" if all(value is True for value in requirements.values())
            else "not_qualified"
        ),
        "partition": public_partition,
        "calibration": calibration,
        "metrics": {
            "train": train_metrics,
            "holdout": heldout_metrics,
            "train_positive_scores": (
                floor_result["positive_score_summary"]
                if floor_result is not None
                else None
            ),
            "heldout_positive_scores": (
                numeric_summary(holdout_scores) if holdout_scores else None
            ),
        },
        "requirements": requirements,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
    }


def _load_authenticated_mapping(path: Path | str) -> dict[str, Any]:
    artifact_path = Path(path)
    try:
        value = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read V3.5 calibration artifact") from error
    if not isinstance(value, dict):
        raise ValueError("V3.5 calibration artifact must be a JSON object")
    expected = value.get("artifact_sha256")
    actual = v35_artifact_sha256(value)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("V3.5 calibration artifact hash mismatch")
    logical = value.get("logical_sha256")
    if logical is not None and logical != actual:
        raise ValueError("V3.5 calibration logical hash mismatch")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _require_sha256_fields(
    source: Mapping[str, Any], *, fields: frozenset[str], owner: str
) -> None:
    missing_or_invalid = sorted(
        field for field in fields if not _is_sha256(source.get(field))
    )
    if missing_or_invalid:
        raise ValueError(
            f"V3.5 {owner} source hash fields are missing or invalid: "
            f"{missing_or_invalid}"
        )


def _require_passed_answer_blind_artifact(
    value: Mapping[str, Any], *, requirement_keys: frozenset[str]
) -> None:
    requirements = value.get("requirements")
    if (
        value.get("status") != "passed"
        or value.get("task_accuracy_used") is not False
        or value.get("answer_or_reward_used") is not False
        or not isinstance(requirements, Mapping)
        or set(requirements) != requirement_keys
        or not all(item is True for item in requirements.values())
    ):
        raise ValueError("V3.5 calibration did not pass answer-blind qualification")


def _validate_expected_input_hashes(
    value: Mapping[str, Any],
    expected_input_hashes: Mapping[str, str] | None,
) -> None:
    source = value.get("source")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("V3.5 calibration must bind its source artifacts")
    if expected_input_hashes is None:
        return
    if not isinstance(expected_input_hashes, Mapping):
        raise ValueError("Expected V3.5 input hashes must be a mapping")
    for name, expected in expected_input_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or not _is_sha256(expected)
            or source.get(name) != expected
        ):
            raise ValueError(f"V3.5 calibration input hash mismatch: {name}")


def _validated_calibration_common(
    calibration: Any, *, memory_count: int | None = None
) -> tuple[int, float]:
    if not isinstance(calibration, Mapping):
        raise ValueError("V3.5 calibration payload is missing")
    shortlist_k = calibration.get("shortlist_k")
    floor = calibration.get("minimum_applicability_score")
    if (
        isinstance(shortlist_k, bool)
        or not isinstance(shortlist_k, int)
        or not 1 <= shortlist_k <= V35_MAX_SHORTLIST_K
        or (
            memory_count is not None
            and shortlist_k > min(V35_MAX_SHORTLIST_K, memory_count)
        )
        or isinstance(floor, bool)
        or not isinstance(floor, (int, float))
        or not math.isfinite(float(floor))
        or not -1.0 <= float(floor) <= 1.0
    ):
        raise ValueError("V3.5 shortlist/floor calibration is invalid")
    return shortlist_k, float(floor)


def _validate_static_question_query(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("V3.5 applicability source question query is missing")
    token_ids = value.get("static_question_token_ids")
    embedding_norm = value.get("static_question_embedding_norm")
    if (
        value.get("schema_version") != _V35_STATIC_QUESTION_QUERY_SCHEMA
        or not _is_sha256(value.get("static_question_text_sha256"))
        or not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        )
        or value.get("static_question_token_count") != len(token_ids)
        or value.get("static_question_token_ids_sha256")
        != canonical_json_sha256(token_ids)
        or not _is_sha256(value.get("static_question_embedding_sha256"))
        or isinstance(embedding_norm, bool)
        or not isinstance(embedding_norm, (int, float))
        or not math.isclose(
            float(embedding_norm),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        or value.get("layer_number") != 24
        or value.get("representation") != "decoder_layer_output"
        or value.get("pooling") != "last_valid_token"
        or value.get("normalization") != "l2"
        or value.get("side_kv_disabled") is not True
        or value.get("chat_wrapper_included") is not False
        or value.get("prompt_boilerplate_included") is not False
        or value.get("add_special_tokens") is not False
    ):
        raise ValueError("V3.5 applicability source question contract drifted")


def _validate_applicability_source(value: Mapping[str, Any]) -> None:
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("V3.5 applicability calibration source is missing")
    _require_sha256_fields(
        source,
        fields=_V35_APPLICABILITY_SOURCE_SHA256_FIELDS,
        owner="applicability calibration",
    )
    implementation_files = source.get("compiler_implementation_files_sha256")
    revision = source.get("compiler_git_revision")
    if (
        not isinstance(source.get("dataset_revision"), str)
        or not str(source["dataset_revision"]).strip()
        or not isinstance(revision, str)
        or _GIT_REVISION_PATTERN.fullmatch(revision) is None
        or source.get("source_question_encoder")
        != _V35_SOURCE_QUESTION_ENCODER
        or not isinstance(implementation_files, Mapping)
        or not implementation_files
        or any(
            not isinstance(path, str)
            or not path
            or not _is_sha256(digest)
            for path, digest in implementation_files.items()
        )
        or source.get("compiler_implementation_set_sha256")
        != canonical_json_sha256(dict(implementation_files))
    ):
        raise ValueError("V3.5 applicability calibration source contract drifted")


def _validate_applicability_reproduction(value: Mapping[str, Any]) -> None:
    calibration = value.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("V3.5 applicability calibration payload is missing")
    memory_count = calibration.get("memory_count")
    source_pairs = value.get("source_pair_audit")
    if (
        isinstance(memory_count, bool)
        or not isinstance(memory_count, int)
        or memory_count <= 0
        or not isinstance(source_pairs, list)
        or len(source_pairs) != memory_count
        or any(not isinstance(item, Mapping) for item in source_pairs)
    ):
        raise ValueError("V3.5 applicability memory/source-pair count drifted")
    for item in source_pairs:
        _validate_static_question_query(item.get("question_query"))
    try:
        reproduced = calibrate_applicability_selector(
            source_pairs, memory_count=memory_count
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "V3.5 applicability source-pair reproduction failed"
        ) from error
    for field in (
        "status",
        "partition",
        "calibration",
        "metrics",
        "requirements",
        "task_accuracy_used",
        "answer_or_reward_used",
    ):
        if value.get(field) != reproduced[field]:
            raise ValueError(
                f"V3.5 applicability calibration reproduction drifted: {field}"
            )


def _validated_nonnegative_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V3.5 {owner} must be a non-negative integer")
    return value


def _validate_margin_summary(value: Any, *, sample_count: int) -> None:
    ordered_fields = ("min", "p05", "p25", "median", "p75", "p95", "max")
    expected_fields = {"count", "mean", *ordered_fields}
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("count") != sample_count
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), (int, float))
            or not math.isfinite(float(value[field]))
            or float(value[field]) < 0.0
            for field in (*ordered_fields, "mean")
        )
    ):
        raise ValueError("V3.5 dynamic margin summary is invalid")
    ordered = [float(value[field]) for field in ordered_fields]
    if (
        ordered != sorted(ordered)
        or not ordered[0] <= float(value["mean"]) <= ordered[-1]
    ):
        raise ValueError("V3.5 dynamic margin summary is inconsistent")


def _validate_selector_source(
    value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int]:
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("V3.5 selector calibration source is missing")
    _require_sha256_fields(
        source,
        fields=_V35_SELECTOR_SOURCE_SHA256_FIELDS,
        owner="selector calibration",
    )
    completed_count = source.get("completed_sample_count")
    if (
        source.get("logical_split") != "calibration-val"
        or source.get("scope")
        != "first_retrieval_attempt_per_triggered_question"
        or source.get("system_version") != "v3.5"
        or source.get("system_profile_schema") != V35_SYSTEM_PROFILE_SCHEMA
        or source.get("calibration_trace_only") is not True
        or isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or completed_count <= 0
    ):
        raise ValueError("V3.5 selector calibration source contract drifted")
    return source, completed_count


def _validate_selector_calibration(
    value: Mapping[str, Any], *, completed_count: int
) -> None:
    calibration = value.get("calibration")
    _validated_calibration_common(calibration)
    assert isinstance(calibration, Mapping)
    sample_count = _validated_nonnegative_int(
        calibration.get("sample_count"), owner="selector sample_count"
    )
    first_attempt_count = _validated_nonnegative_int(
        calibration.get("first_attempt_count"),
        owner="selector first-attempt count",
    )
    target_count = _validated_nonnegative_int(
        calibration.get("target_retained_count"),
        owner="selector target retained count",
    )
    actual_count = _validated_nonnegative_int(
        calibration.get("actual_retained_count"),
        owner="selector actual retained count",
    )
    available_count = _validated_nonnegative_int(
        calibration.get("static_selector_available_sample_count"),
        owner="selector available sample count",
    )
    insufficient_count = _validated_nonnegative_int(
        calibration.get("insufficient_shortlist_sample_count"),
        owner="selector insufficient-shortlist sample count",
    )
    selected_memory_count = _validated_nonnegative_int(
        calibration.get("first_attempt_selected_memory_count"),
        owner="selector selected-memory count",
    )
    threshold = calibration.get("minimum_dynamic_top1_top2_margin")
    target = calibration.get("target_retained_fraction")
    actual = calibration.get("actual_retained_fraction")
    insufficient_fraction = calibration.get("insufficient_shortlist_fraction")
    frequency = calibration.get("first_attempt_selected_memory_frequency")
    if (
        sample_count <= 0
        or first_attempt_count != sample_count
        or target != V35_DYNAMIC_TARGET_RETAINED_FRACTION
        or target_count != max(1, math.ceil(sample_count * float(target)))
        or not target_count <= actual_count <= sample_count
        or isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isclose(
            float(actual),
            actual_count / sample_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not V35_DYNAMIC_TARGET_RETAINED_FRACTION <= float(actual) <= 1.0
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
        or calibration.get("dynamic_margin_tie_policy")
        != V35_DYNAMIC_MARGIN_TIE_POLICY
        or calibration.get("applicability_score_floor_tie_policy")
        != V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY
        or available_count + insufficient_count != completed_count
        or sample_count > available_count
        or isinstance(insufficient_fraction, bool)
        or not isinstance(insufficient_fraction, (int, float))
        or not math.isclose(
            float(insufficient_fraction),
            insufficient_count / completed_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(frequency, list)
        or len(frequency) != selected_memory_count
    ):
        raise ValueError("V3.5 dynamic selector calibration did not qualify")
    normalized_frequency: list[tuple[str, int]] = []
    for item in frequency:
        if not isinstance(item, Mapping):
            raise ValueError("V3.5 selector memory frequency is malformed")
        memory_id = item.get("memory_id")
        count = item.get("count")
        if (
            not isinstance(memory_id, str)
            or not memory_id
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError("V3.5 selector memory frequency is invalid")
        normalized_frequency.append((memory_id, count))
    if (
        len({memory_id for memory_id, _count in normalized_frequency})
        != len(normalized_frequency)
        or sum(count for _memory_id, count in normalized_frequency) != sample_count
        or normalized_frequency
        != sorted(normalized_frequency, key=lambda item: (-item[1], item[0]))
    ):
        raise ValueError("V3.5 selector memory frequency is inconsistent")
    _validate_margin_summary(
        calibration.get("margin_summary"), sample_count=sample_count
    )
    summary = calibration["margin_summary"]
    if not float(summary["min"]) <= float(threshold) <= float(summary["max"]):
        raise ValueError("V3.5 selector threshold is outside its margin summary")


def load_v35_applicability_calibration(
    path: Path | str,
    *,
    expected_input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load only a passed, authenticated positive-only static calibration."""

    value = _load_authenticated_mapping(path)
    if value.get("schema_version") != V35_APPLICABILITY_CALIBRATION_SCHEMA:
        raise ValueError("Unexpected V3.5 applicability calibration schema")
    _require_passed_answer_blind_artifact(
        value, requirement_keys=_V35_APPLICABILITY_REQUIREMENT_KEYS
    )
    _validate_applicability_source(value)
    _validate_expected_input_hashes(value, expected_input_hashes)
    _validate_applicability_reproduction(value)
    partition = value.get("partition")
    if not isinstance(partition, Mapping):
        raise ValueError("V3.5 applicability partition is missing")
    train_ids = partition.get("train_memory_ids")
    heldout_ids = partition.get("heldout_memory_ids")
    if (
        partition.get("seed") != V35_SOURCE_PAIR_PARTITION_SEED
        or partition.get("train_fraction")
        != V35_SOURCE_PAIR_TRAIN_FRACTION
        or not isinstance(train_ids, list)
        or not isinstance(heldout_ids, list)
        or not train_ids
        or not heldout_ids
        or any(not isinstance(item, str) or not item for item in (*train_ids, *heldout_ids))
        or len(set(train_ids)) != len(train_ids)
        or len(set(heldout_ids)) != len(heldout_ids)
        or set(train_ids).intersection(heldout_ids)
    ):
        raise ValueError("V3.5 applicability partition contract drifted")
    calibration = value.get("calibration")
    assert isinstance(calibration, Mapping)
    memory_count = calibration.get("memory_count")
    assert isinstance(memory_count, int) and not isinstance(memory_count, bool)
    _validated_calibration_common(calibration, memory_count=memory_count)
    train_recall = calibration.get("train_own_memory_recall_at_k")
    heldout_recall = calibration.get("heldout_own_memory_recall_at_k")
    heldout_retention = calibration.get(
        "heldout_own_positive_retained_fraction"
    )
    if (
        calibration.get("applicability_score_floor_quantile")
        != V35_APPLICABILITY_SCORE_FLOOR_QUANTILE
        or calibration.get("applicability_score_floor_tie_policy")
        != V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY
        or calibration.get("applicability_floor_role")
        != V35_APPLICABILITY_FLOOR_ROLE
        or not isinstance(train_recall, (int, float))
        or not math.isfinite(float(train_recall))
        or float(train_recall) < V35_MIN_TRAIN_OWN_MEMORY_RECALL
        or not isinstance(heldout_recall, (int, float))
        or not math.isfinite(float(heldout_recall))
        or float(heldout_recall) < V35_MIN_HELDOUT_OWN_MEMORY_RECALL
        or not isinstance(heldout_retention, (int, float))
        or not math.isfinite(float(heldout_retention))
        or float(heldout_retention) < V35_MIN_HELDOUT_POSITIVE_RETENTION
    ):
        raise ValueError("V3.5 applicability calibration did not qualify")
    return value


def load_v35_selector_calibration(
    path: Path | str,
    *,
    expected_input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load only the final answer-blind V3.5 dynamic selector artifact."""

    value = _load_authenticated_mapping(path)
    if (
        value.get("schema_version") != V35_SELECTOR_CALIBRATION_SCHEMA
        or value.get("policy") != V35_SELECTOR_POLICY
    ):
        raise ValueError("Unexpected V3.5 selector calibration schema/policy")
    _require_passed_answer_blind_artifact(
        value, requirement_keys=_V35_SELECTOR_REQUIREMENT_KEYS
    )
    _source, completed_count = _validate_selector_source(value)
    _validate_expected_input_hashes(value, expected_input_hashes)
    _validate_selector_calibration(value, completed_count=completed_count)
    return value


__all__ = [
    "ApplicabilityAwareRetrievalDecision",
    "V35_APPLICABILITY_CALIBRATION_SCHEMA",
    "V35_APPLICABILITY_FLOOR_ROLE",
    "V35_APPLICABILITY_SCORE_FLOOR_QUANTILE",
    "V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY",
    "V35_DUAL_KEY_BANK_SCHEMA",
    "V35_DYNAMIC_MARGIN_TIE_POLICY",
    "V35_DYNAMIC_TARGET_RETAINED_FRACTION",
    "V35_MAX_SHORTLIST_K",
    "V35_RETRIEVAL_DECISION_SCHEMA",
    "V35_SELECTOR_CALIBRATION_SCHEMA",
    "V35_SELECTOR_POLICY",
    "V35_SOURCE_PAIR_PARTITION_SEED",
    "V35_SOURCE_PAIR_TRAIN_FRACTION",
    "applicability_score_floor",
    "calibrate_applicability_selector",
    "deterministic_source_pair_partition",
    "load_v35_applicability_calibration",
    "load_v35_selector_calibration",
    "own_memory_rank_metrics",
    "own_memory_recall",
    "partition_source_pairs",
    "retained_dynamic_margin_threshold",
    "select_minimal_shortlist_k",
    "v35_artifact_logical_sha256",
    "v35_artifact_sha256",
]
