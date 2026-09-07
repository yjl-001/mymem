#!/usr/bin/env python3
"""Run CPU-only V4 selector/state diagnostics from a source-state cache.

The auditor derives local1/4/8/16/32 features, performs leave-one-sample-out
ranking, compares raw and bank-specific empirical-tail scores, measures
hubness, and freezes diagnostic thresholds.  It never loads a tokenizer,
reasoner, or side-KV tensor and never emits an online-qualified artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_source_state import (
    V4_SOURCE_STATE_AUDIT_SCHEMA,
    V4_SOURCE_STATE_WINDOWS,
    V4SourceStateCache,
    derive_window_mean,
    independent_support,
    load_source_state_cache,
)


MAX_UNSAFE_RATE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--windows",
        default="1,4,8,16,32",
        help="Comma-separated latest-token windows derived from raw32.",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1",
        help="Prompt-semantic weight in alpha*prompt+(1-alpha)*dynamic.",
    )
    return parser.parse_args()


def _parse_windows(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("V4 CPU audit windows must be integers") from exc
    if not result or len(set(result)) != len(result) or any(
        item not in V4_SOURCE_STATE_WINDOWS for item in result
    ):
        raise ValueError("V4 CPU audit windows must be unique members of 1,4,8,16,32")
    return result


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("V4 CPU audit alphas must be numeric") from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in result)
    ):
        raise ValueError("V4 CPU audit alphas must be unique values in [0,1]")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    temporary.replace(path)
    return count


def _logmeanexp(values: Any) -> Any:
    import torch

    return torch.logsumexp(values, dim=0) - math.log(int(values.numel()))


def _sample_balanced_similarity(
    *,
    vectors_by_sample: Mapping[str, Sequence[Any]],
    query: Any,
) -> Any:
    """Give each independent sample equal mass while retaining its attempts."""

    import torch

    if not vectors_by_sample or any(not values for values in vectors_by_sample.values()):
        raise ValueError("V4 CPU audit has an empty sample-balanced evidence side")
    per_sample = [
        _logmeanexp(torch.stack(list(vectors)) @ query)
        for _, vectors in sorted(vectors_by_sample.items())
    ]
    return _logmeanexp(torch.stack(per_sample))


def _event_vectors(
    cache: V4SourceStateCache, *, window: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch.nn.functional as F

    if cache.tensors is None:
        raise ValueError("V4 CPU audit requires loaded cache tensors")
    tensors = cache.tensors
    prompt_vectors: dict[str, Any] = {}
    dynamic_vectors: dict[str, Any] = {}
    aligned_vectors: dict[str, Any] = {}
    for event in cache.events:
        event_id = str(event["event_id"])
        kind = event["event_kind"]
        rows = event["tensor_rows"]
        if kind == "prompt_semantic":
            index = int(rows["question_mean_states"])
            prompt_vectors[str(event["independent_sample_id"])] = F.normalize(
                tensors["question_mean_states"][index].float(), dim=0
            ).cpu()
        elif kind == "failure_gate_attempt":
            index = int(rows["failure_gate_windows"])
            dynamic_vectors[event_id] = derive_window_mean(
                tensors["failure_gate_windows"][index],
                tensors["failure_gate_masks"][index],
                size=window,
            )
            aligned_vectors[event_id] = derive_window_mean(
                tensors["aligned_success_windows"][index],
                tensors["aligned_success_masks"][index],
                size=window,
            )
        elif kind == "success_gate_attempt":
            index = int(rows["success_gate_windows"])
            dynamic_vectors[event_id] = derive_window_mean(
                tensors["success_gate_windows"][index],
                tensors["success_gate_masks"][index],
                size=window,
            )
    return prompt_vectors, dynamic_vectors, aligned_vectors


def _score_bank(
    *,
    query_event: Mapping[str, Any],
    bank_id: str,
    failure_events: Sequence[Mapping[str, Any]],
    prompt_events: Sequence[Mapping[str, Any]],
    prompt_vectors: Mapping[str, Any],
    dynamic_vectors: Mapping[str, Any],
    aligned_vectors: Mapping[str, Any],
) -> tuple[float, float]:
    import torch

    query_sample = str(query_event["independent_sample_id"])
    query_dynamic = dynamic_vectors[str(query_event["event_id"])]
    query_prompt = prompt_vectors[query_sample]
    positive_by_sample: dict[str, list[Any]] = defaultdict(list)
    negative_by_sample: dict[str, list[Any]] = defaultdict(list)
    for event in failure_events:
        sample_id = str(event["independent_sample_id"])
        if sample_id == query_sample:
            continue
        event_id = str(event["event_id"])
        if event["bank_id"] == bank_id:
            positive_by_sample[sample_id].append(dynamic_vectors[event_id])
        negative_by_sample[sample_id].append(aligned_vectors[event_id])
        if event["bank_id"] != bank_id:
            negative_by_sample[sample_id].append(dynamic_vectors[event_id])
    if not positive_by_sample or not negative_by_sample:
        raise ValueError("V4 CPU audit LOO fold has an empty dynamic evidence side")
    positive = _sample_balanced_similarity(
        vectors_by_sample=positive_by_sample,
        query=query_dynamic,
    )
    negative = _sample_balanced_similarity(
        vectors_by_sample=negative_by_sample,
        query=query_dynamic,
    )
    dynamic_score = float((positive - negative).item())
    positive_prompt_samples = sorted({
        str(event["independent_sample_id"])
        for event in prompt_events
        if event["bank_id"] == bank_id
        and event["independent_sample_id"] != query_sample
    })
    if not positive_prompt_samples:
        raise ValueError("V4 CPU audit LOO fold has no prompt-semantic support")
    prompt_scores = torch.stack(
        [prompt_vectors[sample_id] for sample_id in positive_prompt_samples]
    ) @ query_prompt
    prompt_score = float(_logmeanexp(prompt_scores).item())
    return prompt_score, dynamic_score


def _raw_query_rows(
    cache: V4SourceStateCache, *, window: int, alpha: float
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    prompt_vectors, dynamic_vectors, aligned_vectors = _event_vectors(
        cache, window=window
    )
    failure_events = [
        event for event in cache.events if event["event_kind"] == "failure_gate_attempt"
    ]
    prompt_events = [
        event for event in cache.events if event["event_kind"] == "prompt_semantic"
    ]
    success_events = [
        event for event in cache.events if event["event_kind"] == "success_gate_attempt"
    ]
    support = independent_support(cache.events, event_kind="failure_gate_attempt")
    bank_ids = tuple(sorted(bank_id for bank_id, count in support.items() if count >= 2))
    if len(bank_ids) < 2:
        raise ValueError("V4 CPU audit requires two banks with two gate-reachable samples")
    rows: list[dict[str, Any]] = []
    failure_queries = [
        event for event in failure_events if event["bank_id"] in set(bank_ids)
    ]
    for event in failure_queries + success_events:
        scores: list[dict[str, Any]] = []
        for bank_id in bank_ids:
            prompt_score, dynamic_score = _score_bank(
                query_event=event,
                bank_id=bank_id,
                failure_events=failure_events,
                prompt_events=prompt_events,
                prompt_vectors=prompt_vectors,
                dynamic_vectors=dynamic_vectors,
                aligned_vectors=aligned_vectors,
            )
            combined = alpha * prompt_score + (1.0 - alpha) * dynamic_score
            scores.append(
                {
                    "bank_id": bank_id,
                    "prompt_score": prompt_score,
                    "dynamic_score": dynamic_score,
                    "combined_raw_score": combined,
                }
            )
        rows.append(
            {
                "event_id": str(event["event_id"]),
                "query_kind": (
                    "failure" if event["event_kind"] == "failure_gate_attempt" else "success_safety"
                ),
                "experience_id": str(event["experience_id"]),
                "sample_id": str(event["sample_id"]),
                "independent_sample_id": str(event["independent_sample_id"]),
                "source_bank_id": str(event["bank_id"]),
                "is_medoid": bool(event["is_medoid"]),
                "curation_tier": str(event["curation_tier"]),
                "gate_attempt_number": int(event["attempt_number"]),
                "scores": scores,
            }
        )
    return rows, bank_ids


def _empirical_tail_score(value: float, background: Sequence[float]) -> float:
    if not background:
        raise ValueError("V4 bank-specific normalization has no background")
    tail_probability = (
        1.0 + sum(float(item) >= value for item in background)
    ) / (len(background) + 1.0)
    return -math.log(tail_probability)


def _rank_rows(
    raw_rows: Sequence[Mapping[str, Any]], *, normalization: str
) -> list[dict[str, Any]]:
    if normalization not in {"raw", "bank_empirical_tail"}:
        raise ValueError("Unknown V4 CPU audit normalization")
    background_by_bank: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in raw_rows:
        for score in row["scores"]:
            if (
                row["query_kind"] == "success_safety"
                or score["bank_id"] != row["source_bank_id"]
            ):
                background_by_bank[str(score["bank_id"])].append(
                    (
                        str(row["independent_sample_id"]),
                        float(score["combined_raw_score"]),
                    )
                )
    ranked_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        ranked_scores: list[dict[str, Any]] = []
        for score in row["scores"]:
            raw = float(score["combined_raw_score"])
            if normalization == "raw":
                selection_score = raw
            else:
                values_by_sample: dict[str, list[float]] = defaultdict(list)
                for sample_id, value in background_by_bank[str(score["bank_id"])]:
                    if sample_id != row["independent_sample_id"]:
                        values_by_sample[sample_id].append(value)
                background = [
                    sum(values) / len(values)
                    for _, values in sorted(values_by_sample.items())
                ]
                selection_score = _empirical_tail_score(raw, background)
            ranked_scores.append(
                {
                    **dict(score),
                    "selection_score": selection_score,
                }
            )
        ranked_scores.sort(key=lambda item: (-float(item["selection_score"]), str(item["bank_id"])))
        top1, top2 = ranked_scores[:2]
        ranked_rows.append(
            {
                **{key: value for key, value in row.items() if key != "scores"},
                "normalization": normalization,
                "top1_bank_id": top1["bank_id"],
                "top1_score": float(top1["selection_score"]),
                "top2_bank_id": top2["bank_id"],
                "top2_score": float(top2["selection_score"]),
                "margin": float(top1["selection_score"] - top2["selection_score"]),
                "correct_top1": (
                    top1["bank_id"] == row["source_bank_id"]
                    if row["query_kind"] == "failure"
                    else None
                ),
                "ranked_scores": ranked_scores,
            }
        )
    return ranked_rows


def _threshold_candidates(values: Sequence[float], *, floor_zero: bool) -> list[float]:
    if not values:
        return []
    candidates = {float(value) for value in values}
    if floor_zero:
        candidates = {max(0.0, value) for value in candidates}
        candidates.add(0.0)
    candidates.add(math.nextafter(max(candidates), math.inf))
    return sorted(candidates)


def _calibrate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if row["query_kind"] == "failure"]
    successes = [row for row in rows if row["query_kind"] == "success_safety"]
    if not failures or not successes:
        return {
            "status": "not_run_missing_failure_or_reachable_success_queries",
            "qualified": False,
        }
    absolute_candidates = _threshold_candidates(
        [float(row["top1_score"]) for row in rows], floor_zero=False
    )
    margin_candidates = _threshold_candidates(
        [float(row["margin"]) for row in rows], floor_zero=True
    )
    safe: list[dict[str, Any]] = []
    for absolute in absolute_candidates:
        for margin in margin_candidates:
            selected = lambda row: (
                float(row["top1_score"]) >= absolute
                and float(row["margin"]) >= margin
            )
            success_false_events = sum(selected(row) for row in successes)
            failure_wrong_events = sum(
                selected(row) and row["correct_top1"] is not True for row in failures
            )
            failure_correct_events = sum(
                selected(row) and row["correct_top1"] is True for row in failures
            )
            success_samples = {
                str(row["independent_sample_id"]) for row in successes
            }
            failure_samples = {
                str(row["independent_sample_id"]) for row in failures
            }
            success_false_samples = {
                str(row["independent_sample_id"])
                for row in successes
                if selected(row)
            }
            failure_wrong_samples = {
                str(row["independent_sample_id"])
                for row in failures
                if selected(row) and row["correct_top1"] is not True
            }
            failure_correct_samples = {
                str(row["independent_sample_id"])
                for row in failures
                if selected(row) and row["correct_top1"] is True
            }
            if (
                len(success_false_samples) / len(success_samples)
                <= MAX_UNSAFE_RATE + 1e-12
                and len(failure_wrong_samples) / len(failure_samples)
                <= MAX_UNSAFE_RATE + 1e-12
            ):
                safe.append(
                    {
                        "absolute_threshold": absolute,
                        "margin_threshold": margin,
                        "failure_correct_selected_event_count": failure_correct_events,
                        "failure_correct_selected_independent_sample_count": len(
                            failure_correct_samples
                        ),
                        "failure_correct_independent_sample_coverage": len(
                            failure_correct_samples
                        )
                        / len(failure_samples),
                        "failure_wrong_selected_event_count": failure_wrong_events,
                        "failure_wrong_selected_independent_sample_count": len(
                            failure_wrong_samples
                        ),
                        "failure_wrong_independent_sample_rate": len(
                            failure_wrong_samples
                        )
                        / len(failure_samples),
                        "success_false_selected_event_count": success_false_events,
                        "success_false_selected_independent_sample_count": len(
                            success_false_samples
                        ),
                        "success_false_independent_sample_rate": len(
                            success_false_samples
                        )
                        / len(success_samples),
                    }
                )
    if not safe:
        return {"status": "no_safe_threshold_pair", "qualified": False}
    selected = sorted(
        safe,
        key=lambda item: (
            -item["failure_correct_selected_independent_sample_count"],
            -item["failure_correct_selected_event_count"],
            item["failure_wrong_selected_independent_sample_count"]
            + item["success_false_selected_independent_sample_count"],
            item["absolute_threshold"],
            item["margin_threshold"],
        ),
    )[0]
    return {
        "status": "diagnostic_threshold_selected",
        "qualified": False,
        "qualification_withheld_reason": "oracle_causal_utility_not_yet_applied",
        "maximum_unsafe_rate": MAX_UNSAFE_RATE,
        "safe_candidate_count": len(safe),
        "selected": selected,
    }


def _hubness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["top1_bank_id"]) for row in rows)
    total = sum(counts.values())
    ordered = dict(sorted(counts.items()))
    return {
        "query_count": total,
        "independent_query_sample_count": len(
            {str(row["independent_sample_id"]) for row in rows}
        ),
        "top1_count_by_bank": ordered,
        "top1_independent_query_sample_count_by_bank": {
            bank_id: len(
                {
                    str(row["independent_sample_id"])
                    for row in rows
                    if row["top1_bank_id"] == bank_id
                }
            )
            for bank_id in sorted(counts)
        },
        "maximum_top1_share": max(counts.values(), default=0) / total if total else 0.0,
        "herfindahl_index": (
            sum((count / total) ** 2 for count in counts.values()) if total else 0.0
        ),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        return None
    return sum(
        first * second
        for first, second in zip(centered_left, centered_right)
    ) / denominator


def _bank_size_bias(
    rows: Sequence[Mapping[str, Any]],
    *,
    bank_ids: Sequence[str],
    failure_support: Mapping[str, int],
) -> dict[str, Any]:
    """Expose whether larger gate-support banks dominate top-1 selections."""

    selections = Counter(str(row["top1_bank_id"]) for row in rows)
    support_values = [int(failure_support[bank_id]) for bank_id in bank_ids]
    selection_values = [int(selections[bank_id]) for bank_id in bank_ids]
    propensities = [
        selections[bank_id] / failure_support[bank_id] for bank_id in bank_ids
    ]
    positive_propensities = [value for value in propensities if value > 0.0]
    zero_selection_bank_count = sum(value == 0.0 for value in propensities)
    propensity_ratio = (
        max(positive_propensities) / min(positive_propensities)
        if positive_propensities
        else None
    )
    correlation = _pearson(
        [float(value) for value in support_values],
        [float(value) for value in selection_values],
    )
    return {
        "diagnostic_only": True,
        "heuristic": (
            "flag when support/top1 Pearson >= 0.8 or support-normalized "
            "selection propensity ratio >= 2, including a zero-selection bank"
        ),
        "support_top1_pearson_correlation": correlation,
        "maximum_to_minimum_positive_support_normalized_selection_ratio": (
            propensity_ratio
        ),
        "zero_top1_selection_bank_count": zero_selection_bank_count,
        "potential_bank_size_bias_detected": (
            (correlation is not None and correlation >= 0.8)
            or (propensity_ratio is not None and propensity_ratio >= 2.0)
            or (bool(positive_propensities) and zero_selection_bank_count > 0)
        ),
        "per_bank": {
            bank_id: {
                "failure_gate_independent_sample_support": int(
                    failure_support[bank_id]
                ),
                "top1_selection_event_count": int(selections[bank_id]),
                "top1_selection_events_per_support": (
                    selections[bank_id] / failure_support[bank_id]
                ),
            }
            for bank_id in bank_ids
        },
    }


def _sample_macro_correct_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    by_sample: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["independent_sample_id"])].append(
            row["correct_top1"] is True
        )
    if not by_sample:
        return 0.0
    return sum(
        sum(values) / len(values) for values in by_sample.values()
    ) / len(by_sample)


def _per_bank(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failure_rows = [row for row in rows if row["query_kind"] == "failure"]
    result: dict[str, Any] = {}
    for bank_id in sorted({str(row["source_bank_id"]) for row in failure_rows}):
        bank_rows = [row for row in failure_rows if row["source_bank_id"] == bank_id]
        samples = {str(row["sample_id"]) for row in bank_rows}
        independent_samples = {
            str(row["independent_sample_id"]) for row in bank_rows
        }
        correct = sum(row["correct_top1"] is True for row in bank_rows)
        result[bank_id] = {
            "gate_event_count": len(bank_rows),
            "sample_id_count": len(samples),
            "independent_sample_count": len(independent_samples),
            "correct_top1_event_count": correct,
            "correct_top1_event_rate": correct / len(bank_rows),
            "correct_top1_independent_sample_macro_rate": (
                _sample_macro_correct_rate(bank_rows)
            ),
        }
    return result


def audit_cache(
    cache: V4SourceStateCache,
    *,
    windows: Sequence[int],
    alphas: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the complete CPU ablation grid and return best diagnostic rows."""

    if cache.tensors is None:
        raise ValueError("V4 CPU audit requires cache tensors")
    failure_support = independent_support(
        cache.events, event_kind="failure_gate_attempt"
    )
    eligible_banks = sorted(
        bank_id for bank_id, count in failure_support.items() if count >= 2
    )
    base_report = {
        "schema_version": V4_SOURCE_STATE_AUDIT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_offline_diagnostic",
        "offline_only": True,
        "qualified_for_online_use": False,
        "online_artifacts_generated": False,
        "reasoner_loaded": False,
        "support_unit": "independent_sample",
        "multiple_attempts_are_observations_not_support": True,
        "evidence_weighting": (
            "equal_independent_sample_mass_within_sample_attempt_logmeanexp"
        ),
        "empirical_background_weighting": "equal_independent_sample_mean",
        "matched_success_aligned_role": "repair_direction_control_only",
        "success_safety_negative_role": "actual_success_gate_states_only",
        "cache_manifest_logical_sha256": cache.manifest["manifest_sha256"],
        "cache_gate_reachability_logical_sha256": cache.reachability_report[
            "report_sha256"
        ],
        "windows": list(windows),
        "alphas": list(alphas),
        "normalizations": ["raw", "bank_empirical_tail"],
        "failure_gate_support_by_bank": failure_support,
        "construction_support_by_bank": {
            bank_id: int(values["construction_independent_sample_count"])
            for bank_id, values in cache.reachability_report["per_bank"].items()
        },
        "success_gate_support_by_bank": independent_support(
            cache.events, event_kind="success_gate_attempt"
        ),
    }
    if len(eligible_banks) < 2:
        return (
            {
                **base_report,
                "diagnostic_scope_sufficient": False,
                "insufficient_scope_reason": (
                    "fewer_than_two_banks_with_two_independent_gate_reachable_samples"
                ),
                "eligible_bank_ids": eligible_banks,
                "ablation_count": 0,
                "selected_diagnostic": None,
                "ablations": [],
            },
            [],
        )
    ablations: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[int, float, str], list[dict[str, Any]]] = {}
    for window in windows:
        for alpha in alphas:
            raw_rows, bank_ids = _raw_query_rows(cache, window=window, alpha=alpha)
            for normalization in ("raw", "bank_empirical_tail"):
                rows = _rank_rows(raw_rows, normalization=normalization)
                rows_by_key[(window, alpha, normalization)] = rows
                failures = [row for row in rows if row["query_kind"] == "failure"]
                successes = [row for row in rows if row["query_kind"] == "success_safety"]
                correct = sum(row["correct_top1"] is True for row in failures)
                ablations.append(
                    {
                        "window": window,
                        "alpha": alpha,
                        "normalization": normalization,
                        "bank_count": len(bank_ids),
                        "failure_gate_event_count": len(failures),
                        "failure_independent_sample_count": len(
                            {str(row["independent_sample_id"]) for row in failures}
                        ),
                        "success_safety_gate_event_count": len(successes),
                        "success_safety_independent_sample_count": len(
                            {str(row["independent_sample_id"]) for row in successes}
                        ),
                        "correct_top1_event_count": correct,
                        "correct_top1_event_rate": correct / len(failures) if failures else 0.0,
                        "correct_top1_independent_sample_macro_rate": (
                            _sample_macro_correct_rate(failures)
                        ),
                        "hubness": _hubness(rows),
                        "bank_size_bias": _bank_size_bias(
                            rows,
                            bank_ids=bank_ids,
                            failure_support=failure_support,
                        ),
                        "threshold_calibration": _calibrate(rows),
                        "per_bank": _per_bank(rows),
                    }
                )
    selected = sorted(
        ablations,
        key=lambda item: (
            -item["correct_top1_independent_sample_macro_rate"],
            -item["correct_top1_event_rate"],
            item["hubness"]["maximum_top1_share"],
            0 if item["normalization"] == "bank_empirical_tail" else 1,
            item["window"],
            item["alpha"],
        ),
    )[0]
    selected_key = (
        int(selected["window"]),
        float(selected["alpha"]),
        str(selected["normalization"]),
    )
    report = {
        **base_report,
        "diagnostic_scope_sufficient": True,
        "eligible_bank_ids": eligible_banks,
        "ablation_count": len(ablations),
        "selected_diagnostic": selected,
        "ablations": ablations,
    }
    return report, rows_by_key[selected_key]


def main() -> None:
    args = parse_args()
    windows = _parse_windows(args.windows)
    alphas = _parse_alphas(args.alphas)
    cache = load_source_state_cache(args.cache_manifest, load_tensors=True)
    report, rows = audit_cache(cache, windows=windows, alphas=alphas)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "v4_source_state_cpu_audit_rows.jsonl"
    report_path = output_dir / "v4_source_state_cpu_audit_report.json"
    _write_jsonl(rows_path, rows)
    report["artifacts"] = {
        "selected_diagnostic_rows": {
            "path": rows_path.name,
            "sha256": file_sha256(rows_path),
            "row_count": len(rows),
        },
        "online_selector_tensor": None,
        "online_selector_manifest": None,
    }
    report["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key not in {"created_at", "report_sha256"}}
    )
    _write_json(report_path, report)
    print(
        f"[v4-source-state-cpu-audit] complete ablations={report['ablation_count']} "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-source-state-cpu-audit] error: {exc}", file=sys.stderr)
        raise
