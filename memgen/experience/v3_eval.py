"""Strict/format/token aggregation for V3 GSM8K evaluation logs."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence


def percentile_linear(values: Sequence[float | int], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("Percentile needs values and a quantile in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty numeric sequence")
    normalized = [float(value) for value in values]
    return {
        "total": sum(values),
        "mean": sum(normalized) / len(normalized),
        "median": float(median(normalized)),
        "p95": percentile_linear(normalized, 0.95),
        "max": max(values),
    }


def summarize_v3_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only official strict accuracy, format accuracy, and tokens."""

    if not rows:
        raise ValueError("Cannot summarize empty V3 evaluation results")
    conditions: dict[str, Any] = {}
    for condition in ("vanilla", "v3"):
        values = [row["conditions"][condition] for row in rows]
        strict = [bool(value["strict_correct"]) for value in values]
        formatting = [bool(value["format_correct"]) for value in values]
        tokens = [int(value["generated_token_count"]) for value in values]
        conditions[condition] = {
            "sample_count": len(values),
            "strict_correct_count": sum(strict),
            "strict_accuracy": sum(strict) / len(strict),
            "format_correct_count": sum(formatting),
            "format_accuracy": sum(formatting) / len(formatting),
            "generated_tokens": numeric_summary(tokens),
        }

    token_deltas = [
        int(row["conditions"]["v3"]["generated_token_count"])
        - int(row["conditions"]["vanilla"]["generated_token_count"])
        for row in rows
    ]
    attempt_values = [
        row["conditions"]["v3"]["online_diagnostics"] for row in rows
    ]
    return {
        "sample_count": len(rows),
        "conditions": conditions,
        "paired": {
            "strict_accuracy_delta_v3_minus_vanilla": (
                conditions["v3"]["strict_accuracy"]
                - conditions["vanilla"]["strict_accuracy"]
            ),
            "format_accuracy_delta_v3_minus_vanilla": (
                conditions["v3"]["format_accuracy"]
                - conditions["vanilla"]["format_accuracy"]
            ),
            "generated_token_delta_v3_minus_vanilla": numeric_summary(
                token_deltas
            ),
        },
        "online_diagnostics": {
            "questions_with_retrieval_attempt": sum(
                int(value["retrieval_attempt_count"]) > 0
                for value in attempt_values
            ),
            "retrieval_attempt_count": sum(
                int(value["retrieval_attempt_count"]) for value in attempt_values
            ),
            "rearm_count": sum(int(value["rearm_count"]) for value in attempt_values),
            "activation_count": sum(
                int(value["activation_count"]) for value in attempt_values
            ),
            "replacement_count": sum(
                int(value["replacement_count"]) for value in attempt_values
            ),
            "duplicate_count": sum(
                int(value["duplicate_count"]) for value in attempt_values
            ),
            "abstain_count": sum(
                int(value["abstain_count"]) for value in attempt_values
            ),
            "memory_attention_step_count": sum(
                int(value["memory_attention_step_count"])
                for value in attempt_values
            ),
        },
        "metric_contract": {
            "task_metric": "strict_official_gsm8k_first_boxed",
            "format_metric": "first_boxed_parseable",
            "generated_token_count": "through_first_eos_inclusive_else_full_budget",
            "diagnostic_answer_accuracy_aggregated": False,
        },
    }
