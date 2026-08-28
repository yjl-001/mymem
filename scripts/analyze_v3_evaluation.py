#!/usr/bin/env python3
"""Analyze a completed V3 JSONL evaluation without loading the reasoner.

The analyzer is deliberately CPU-only and streams the potentially large JSONL
file.  It validates the frozen run/row hashes, checks online invariants, builds
paired strict/format comparisons, and diagnoses retrieval, replacement,
attention, and generation-length behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3 import (
    V34_GENERATION_RESULT_SCHEMA,
    V34_SYSTEM_PROFILE_SCHEMA,
    V35_GENERATION_RESULT_SCHEMA,
    V35_RETRIEVAL_DECISION_SCHEMA,
    V35_SYSTEM_PROFILE_SCHEMA,
    V3_GENERATION_RESULT_SCHEMA,
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_METHODS,
    V3_QUERY_POOLING_PRE_BOUNDARY,
    V3_SYSTEM_PROFILE_SCHEMA,
    query_embedding_token_index,
)


EVALUATION_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
EVALUATION_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"
V35_EVALUATION_PROFILE_SCHEMA = (
    "experience-memory-v3.5-evaluation-profile-v1"
)
V35_EVALUATION_ROW_SCHEMA = "experience-memory-v3.5-evaluation-row-v1"
EVALUATION_PROFILE_SCHEMAS = frozenset({
    EVALUATION_PROFILE_SCHEMA,
    V35_EVALUATION_PROFILE_SCHEMA,
})
GENERATION_RESULT_SCHEMA = V3_GENERATION_RESULT_SCHEMA
GENERATION_RESULT_SCHEMAS = frozenset({
    GENERATION_RESULT_SCHEMA,
    V34_GENERATION_RESULT_SCHEMA,
    V35_GENERATION_RESULT_SCHEMA,
})
ANALYSIS_REPORT_SCHEMA = "experience-memory-v3-analysis-report-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Defaults to --output with a .md suffix.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-memory-samples", type=int, default=5)
    parser.add_argument(
        "--skip-row-hash-validation",
        action="store_true",
        help="Faster, but removes end-to-end JSONL tamper detection.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def evaluation_profile_sha256(value: Mapping[str, Any]) -> str:
    material = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "repository", "profile_sha256"}
    }
    repository = value.get("repository", {})
    material["code_identity"] = {
        "git_revision": repository.get("git_revision"),
        "tracked_diff_sha256": repository.get("tracked_diff_sha256"),
        "implementation_set_sha256": repository.get(
            "implementation_set_sha256"
        ),
    }
    return canonical_json_sha256(material)


def load_run_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") not in EVALUATION_PROFILE_SCHEMAS:
        raise ValueError("Unexpected V3 evaluation profile schema")
    expected = value.get("profile_sha256")
    actual = evaluation_profile_sha256(value)
    if expected != actual:
        raise ValueError("V3 evaluation profile hash mismatch")
    return value


def percentile_linear(values: Sequence[float | int], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("Percentile needs values and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(
    values: Sequence[float | int], *, include_min: bool = True
) -> dict[str, Any] | None:
    if not values:
        return None
    normalized = [float(value) for value in values]
    result: dict[str, Any] = {
        "count": len(values),
        "total": sum(values),
        "mean": sum(normalized) / len(normalized),
        "median": percentile_linear(normalized, 0.5),
        "p95": percentile_linear(normalized, 0.95),
        "p99": percentile_linear(normalized, 0.99),
        "max": max(values),
    }
    if include_min:
        result["min"] = min(values)
    return result


def exact_mcnemar_two_sided(
    vanilla_only_correct: int, v3_only_correct: int
) -> float | None:
    discordant = vanilla_only_correct + v3_only_correct
    if discordant == 0:
        return None
    tail = min(vanilla_only_correct, v3_only_correct)
    numerator = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * (numerator / (2**discordant)))


def bootstrap_mean_ci(
    values: Sequence[int | float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not values or resamples <= 0:
        return None
    normalized = [float(value) for value in values]
    rng = random.Random(seed)
    count = len(normalized)
    estimates = []
    for _ in range(resamples):
        estimates.append(
            sum(normalized[rng.randrange(count)] for _ in range(count)) / count
        )
    return {
        "method": "paired_nonparametric_bootstrap",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "lower": percentile_linear(estimates, 0.025),
        "upper": percentile_linear(estimates, 0.975),
    }


@dataclass(frozen=True)
class CompactSample:
    sample_id: str
    vanilla_strict: bool
    v3_strict: bool
    vanilla_format: bool
    v3_format: bool
    vanilla_tokens: int
    v3_tokens: int
    completion_exact_match: bool
    attempt_count: int
    rearm_count: int
    activation_count: int
    replacement_count: int
    duplicate_count: int
    abstain_count: int
    outcome_sequence: str
    first_memory_id: str | None
    final_memory_id: str | None
    first_top1_score: float | None
    first_top1_top2_margin: float | None
    first_trigger_entropy: float | None
    first_trigger_boundary_index: int | None
    mean_activation_kl: float | None
    max_activation_kl: float | None
    activation_top1_change_count: int
    attention_step_count: int
    mean_attention_mass: float | None
    max_attention_mass: float | None
    min_attention_mass: float | None
    query_encoding_seconds: float
    retrieval_seconds: float
    attempt_total_seconds: float
    static_selector_unavailable: bool = False
    static_shortlist_size: int = 0
    static_top1_memory_id: str | None = None
    first_selected_static_score: float | None = None
    terminal_abstain_count: int = 0
    clear_on_terminal_abstain_count: int = 0
    active_memory_lifetime_tokens: int = 0
    vanilla_numeric_correct_but_format_invalid: bool = False
    v3_numeric_correct_but_format_invalid: bool = False
    vanilla_answer_marker_seen: bool | None = None
    v3_answer_marker_seen: bool | None = None
    vanilla_first_answer_marker_token_index: int | None = None
    v3_first_answer_marker_token_index: int | None = None
    attempt_to_first_answer_marker_distances: tuple[int, ...] = ()
    attempts_with_subsequent_answer_marker_count: int = 0
    late_attempt_within_32_tokens_count: int = 0
    marker_missing_attempt_count: int = 0
    marker_not_after_attempt_count: int = 0

    @property
    def strict_delta(self) -> int:
        return int(self.v3_strict) - int(self.vanilla_strict)

    @property
    def format_delta(self) -> int:
        return int(self.v3_format) - int(self.vanilla_format)

    @property
    def token_delta(self) -> int:
        return self.v3_tokens - self.vanilla_tokens

    @property
    def triggered(self) -> bool:
        return self.attempt_count > 0


class IntegrityAudit:
    def __init__(self, *, example_limit: int = 50):
        self.example_limit = example_limit
        self.check_counts: Counter[str] = Counter()
        self.failure_counts: Counter[str] = Counter()
        self.failure_examples: list[dict[str, str]] = []

    def check(self, condition: bool, code: str, sample_id: str) -> None:
        self.check_counts[code] += 1
        if condition:
            return
        self.failure_counts[code] += 1
        if len(self.failure_examples) < self.example_limit:
            self.failure_examples.append({"sample_id": sample_id, "code": code})

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": not self.failure_counts,
            "check_counts": dict(sorted(self.check_counts.items())),
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "failure_examples": self.failure_examples,
        }


class StreamingDiagnostics:
    def __init__(self) -> None:
        self.selected_memory_attempt_counts: Counter[str] = Counter()
        self.attention_by_memory_count: Counter[str] = Counter()
        self.attention_by_memory_sum: defaultdict[str, float] = defaultdict(float)
        self.cache_parity_checked = 0
        self.cache_parity_failed = 0
        self.static_top1_memory_counts: Counter[str] = Counter()
        self.static_shortlist_memory_counts: Counter[str] = Counter()


class V35SafetyAudit:
    """Keep qualification-critical violations explicit and sample-addressable."""

    NAMES = (
        "selected_outside_shortlist",
        "stale_attention_after_terminal_clear",
        "terminal_state_drift",
        "full_prefix_query",
        "kv_alignment",
        "attempt_budget",
        "rearm",
    )

    def __init__(self, *, example_limit: int = 200):
        self.example_limit = example_limit
        self.sample_ids: dict[str, list[str]] = {
            name: [] for name in self.NAMES
        }

    def violation(self, name: str, sample_id: str, condition: bool) -> None:
        if name not in self.sample_ids:
            raise ValueError(f"Unknown V3.5 safety violation: {name}")
        if condition and sample_id not in self.sample_ids[name]:
            if len(self.sample_ids[name]) < self.example_limit:
                self.sample_ids[name].append(sample_id)

    def to_dict(self) -> dict[str, Any]:
        counts = {
            name: len(self.sample_ids[name]) for name in self.NAMES
        }
        return {
            "passed": not any(counts.values()),
            "violation_counts": counts,
            "violations": {
                name: {
                    "count": counts[name],
                    "sample_ids": list(self.sample_ids[name]),
                }
                for name in self.NAMES
            },
            "qualification_critical": list(self.NAMES),
        }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_row_hash(row: Mapping[str, Any]) -> bool:
    expected = row.get("row_sha256")
    actual = canonical_json_sha256({
        key: value
        for key, value in row.items()
        if key not in {"created_at", "row_sha256"}
    })
    return expected == actual


def _static_shortlist_ids(trace: Mapping[str, Any]) -> list[str]:
    values = trace.get("shortlist_memory_ids")
    if isinstance(values, list):
        return [str(value) for value in values]
    values = trace.get("post_floor_shortlist", [])
    if not isinstance(values, list):
        return []
    return [
        str(value.get("memory_id", ""))
        for value in values
        if isinstance(value, Mapping)
    ]


def _decision_shortlist_ids(decision: Mapping[str, Any]) -> list[str]:
    values = decision.get("static_shortlist", [])
    if not isinstance(values, list):
        return []
    return [
        str(value.get("memory_id", ""))
        if isinstance(value, Mapping)
        else str(value)
        for value in values
    ]


def answer_marker_distance_contract(
    attempts: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    *,
    first_marker_token_index: int | None,
) -> dict[str, Any]:
    """Recompute the descriptive attempt-to-first-marker diagnostics."""

    expected_rows: list[dict[str, Any]] = []
    distances: list[int] = []
    missing_count = 0
    not_after_count = 0
    affects_contract_respected = True
    for ordinal, attempt in enumerate(attempts, start=1):
        try:
            observation_raw = attempt.get(
                "generated_observation_index",
                attempt.get("generated_boundary_index"),
            )
            affects_raw = attempt.get("affects_generated_token_index")
            if isinstance(observation_raw, bool) or isinstance(affects_raw, bool):
                raise ValueError
            observation_index = int(observation_raw)
            affects_index = int(affects_raw)
        except (TypeError, ValueError):
            observation_index = None
            affects_index = None
            affects_contract_respected = False
        if (
            observation_index is None
            or observation_index < 0
            or affects_index != observation_index + 1
        ):
            affects_contract_respected = False
        distance = None
        if first_marker_token_index is None:
            missing_count += 1
        elif affects_index is not None and first_marker_token_index >= affects_index:
            distance = first_marker_token_index - affects_index
            distances.append(distance)
        else:
            not_after_count += 1
        expected_rows.append({
            "attempt_number": int(attempt.get("attempt_number", ordinal)),
            "generated_observation_index": observation_index,
            "affects_generated_token_index": affects_index,
            "first_answer_marker_token_index": first_marker_token_index,
            "tokens_until_first_answer_marker": distance,
        })
    logged_rows = diagnostics.get("answer_marker_attempt_distances")
    expected_keys = {
        "attempt_number",
        "generated_observation_index",
        "affects_generated_token_index",
        "first_answer_marker_token_index",
        "tokens_until_first_answer_marker",
    }
    logged_shape_ok = (
        isinstance(logged_rows, list)
        and len(logged_rows) == len(expected_rows)
        and all(
            isinstance(item, Mapping) and set(item) == expected_keys
            for item in logged_rows
        )
    )
    late_count = sum(0 <= distance <= 32 for distance in distances)
    valid = (
        logged_shape_ok
        and list(logged_rows) == expected_rows
        and diagnostics.get("first_answer_marker_token_index")
        == first_marker_token_index
        and diagnostics.get("attempt_affects_index_contract_respected")
        is affects_contract_respected
        and affects_contract_respected
        and int(diagnostics.get(
            "attempts_with_subsequent_answer_marker_count", -1
        ))
        == len(distances)
        and int(diagnostics.get("late_attempt_within_32_tokens_count", -1))
        == late_count
    )
    return {
        "valid": valid,
        "distances": tuple(distances),
        "attempts_with_subsequent_answer_marker_count": len(distances),
        "late_attempt_within_32_tokens_count": late_count,
        "marker_missing_attempt_count": missing_count,
        "marker_not_after_attempt_count": not_after_count,
    }


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def extract_sample(
    row: Mapping[str, Any],
    *,
    expected_profile_sha256: str,
    expected_risk_role: str,
    audit: IntegrityAudit,
    streaming: StreamingDiagnostics,
    validate_hash: bool,
    is_v35: bool = False,
    safety: V35SafetyAudit | None = None,
    expected_generation_schema: str | None = None,
) -> CompactSample:
    sample_id = str(row.get("sample_id", ""))
    audit.check(bool(sample_id), "sample_id_present", sample_id or "<missing>")
    audit.check(
        row.get("schema_version")
        == (V35_EVALUATION_ROW_SCHEMA if is_v35 else EVALUATION_ROW_SCHEMA),
        "row_schema_matches",
        sample_id,
    )
    audit.check(
        row.get("profile_sha256") == expected_profile_sha256,
        "row_profile_hash_matches",
        sample_id,
    )
    if validate_hash:
        audit.check(validate_row_hash(row), "row_hash_matches", sample_id)

    conditions = row.get("conditions", {})
    vanilla = conditions.get("vanilla", {})
    v3 = conditions.get("v3", {})
    vanilla_ids = tuple(int(value) for value in vanilla.get("completion_token_ids", []))
    v3_ids = tuple(int(value) for value in v3.get("completion_token_ids", []))
    vanilla_tokens = int(vanilla.get("generated_token_count", -1))
    v3_tokens = int(v3.get("generated_token_count", -1))
    audit.check(
        vanilla_tokens == len(vanilla_ids),
        "vanilla_token_count_matches_ids",
        sample_id,
    )
    audit.check(
        v3_tokens == len(v3_ids),
        "v3_token_count_matches_ids",
        sample_id,
    )
    audit.check(
        vanilla.get("completion_token_ids_sha256")
        == canonical_json_sha256(list(vanilla_ids)),
        "vanilla_completion_hash_matches",
        sample_id,
    )
    audit.check(
        v3.get("completion_token_ids_sha256")
        == canonical_json_sha256(list(v3_ids)),
        "v3_completion_hash_matches",
        sample_id,
    )
    vanilla_strict = bool(vanilla.get("strict_correct"))
    v3_strict = bool(v3.get("strict_correct"))
    vanilla_format = bool(vanilla.get("format_correct"))
    v3_format = bool(v3.get("format_correct"))
    vanilla_numeric_format_invalid = bool(
        vanilla.get("numeric_correct_but_format_invalid", False)
    )
    v3_numeric_format_invalid = bool(
        v3.get("numeric_correct_but_format_invalid", False)
    )
    vanilla_marker_seen = _bool_or_none(vanilla.get("answer_marker_seen"))
    v3_marker_seen = _bool_or_none(v3.get("answer_marker_seen"))
    vanilla_marker_index_raw = vanilla.get("first_answer_marker_token_index")
    v3_marker_index_raw = v3.get("first_answer_marker_token_index")
    vanilla_marker_index = (
        int(vanilla_marker_index_raw)
        if isinstance(vanilla_marker_index_raw, int)
        and not isinstance(vanilla_marker_index_raw, bool)
        else None
    )
    v3_marker_index = (
        int(v3_marker_index_raw)
        if isinstance(v3_marker_index_raw, int)
        and not isinstance(v3_marker_index_raw, bool)
        else None
    )
    if is_v35:
        audit.check(
            isinstance(vanilla_marker_seen, bool)
            and (vanilla_marker_index is not None) is vanilla_marker_seen
            and (
                vanilla_marker_index is None
                or 0 <= vanilla_marker_index < vanilla_tokens
            ),
            "v35_vanilla_first_answer_marker_index_is_valid",
            sample_id,
        )
        audit.check(
            isinstance(v3_marker_seen, bool)
            and (v3_marker_index is not None) is v3_marker_seen
            and (v3_marker_index is None or 0 <= v3_marker_index < v3_tokens),
            "v35_v3_first_answer_marker_index_is_valid",
            sample_id,
        )
    audit.check(
        not vanilla_strict or vanilla_format,
        "vanilla_strict_implies_format",
        sample_id,
    )
    audit.check(
        not v3_strict or v3_format,
        "v3_strict_implies_format",
        sample_id,
    )
    audit.check(
        int(row.get("paired_generated_token_delta_v3_minus_vanilla", 10**9))
        == v3_tokens - vanilla_tokens,
        "paired_token_delta_matches",
        sample_id,
    )

    runtime = v3.get("runtime_trace", {})
    diagnostics = v3.get("online_diagnostics", {})
    if is_v35:
        audit.check(
            runtime.get("answer_marker_seen") is v3_marker_seen
            and diagnostics.get("answer_marker_seen") is v3_marker_seen,
            "v35_runtime_and_condition_answer_marker_presence_match",
            sample_id,
        )
    if expected_generation_schema is None:
        expected_generation_schema = (
            V35_GENERATION_RESULT_SCHEMA if is_v35 else GENERATION_RESULT_SCHEMA
        )
    audit.check(
        runtime.get("schema_version") == expected_generation_schema,
        "runtime_trace_schema_matches",
        sample_id,
    )
    runtime_ids = tuple(
        int(value) for value in runtime.get("completion_token_ids", [])
    )
    audit.check(runtime_ids == v3_ids, "runtime_completion_matches_v3", sample_id)
    attempts = list(runtime.get("retrieval_attempts", []))
    boundaries = list(runtime.get("boundary_traces", []))
    attention = list(runtime.get("attention_traces", []))
    attempt_count = len(attempts)
    marker_distance = {
        "valid": True,
        "distances": (),
        "attempts_with_subsequent_answer_marker_count": 0,
        "late_attempt_within_32_tokens_count": 0,
        "marker_missing_attempt_count": 0,
        "marker_not_after_attempt_count": 0,
    }
    if is_v35:
        marker_distance = answer_marker_distance_contract(
            attempts,
            diagnostics,
            first_marker_token_index=v3_marker_index,
        )
        audit.check(
            marker_distance["valid"] is True,
            "v35_attempt_to_first_answer_marker_distances_match",
            sample_id,
        )
    attempt_budget_ok = attempt_count <= 3
    audit.check(attempt_budget_ok, "attempt_budget_at_most_three", sample_id)
    if is_v35 and safety is not None:
        safety.violation("attempt_budget", sample_id, not attempt_budget_ok)
    audit.check(
        int(diagnostics.get("retrieval_attempt_count", -1)) == attempt_count,
        "attempt_count_matches_diagnostics",
        sample_id,
    )
    audit.check(
        [int(item.get("attempt_number", -1)) for item in attempts]
        == list(range(1, attempt_count + 1)),
        "attempt_numbers_contiguous",
        sample_id,
    )
    outcome_sequence = ">".join(
        str(item.get("outcome", "unknown")) for item in attempts
    ) or "no_attempt"
    outcome_counts = Counter(str(item.get("outcome", "unknown")) for item in attempts)
    activation_count = outcome_counts["activated"]
    replacement_count = outcome_counts["replaced"]
    duplicate_count = outcome_counts["duplicate"]
    abstain_count = outcome_counts["abstained"]
    terminal_abstain_count = sum(
        item.get("terminal_abstain") is True for item in attempts
    )
    clear_on_terminal_abstain_count = sum(
        item.get("memory_cleared_on_abstain") is True for item in attempts
    )
    audit.check(
        int(diagnostics.get("activation_count", -1)) == activation_count,
        "activation_count_matches",
        sample_id,
    )
    audit.check(
        int(diagnostics.get("replacement_count", -1)) == replacement_count,
        "replacement_count_matches",
        sample_id,
    )
    audit.check(
        int(diagnostics.get("duplicate_count", -1)) == duplicate_count,
        "duplicate_count_matches",
        sample_id,
    )
    audit.check(
        int(diagnostics.get("abstain_count", -1)) == abstain_count,
        "abstain_count_matches",
        sample_id,
    )
    for diagnostic_name in (
        "attempt_budget_respected",
        "query_context_is_full_prefix",
        "native_cache_excludes_memory_slots",
        "memory_attention_mass_finite_and_positive",
    ):
        audit.check(
            diagnostics.get(diagnostic_name) is True,
            f"online_diagnostic_{diagnostic_name}",
            sample_id,
        )
    static_trace = runtime.get("static_selector_trace")
    static_shortlist_ids: list[str] = []
    static_selector_unavailable = False
    static_top1_memory_id: str | None = None
    if is_v35:
        static_mapping = static_trace if isinstance(static_trace, Mapping) else {}
        static_query = static_mapping.get("query", {})
        static_shortlist_ids = _static_shortlist_ids(static_mapping)
        post_floor = static_mapping.get("post_floor_shortlist", [])
        pre_floor = static_mapping.get("pre_floor_top_k", [])
        if not isinstance(post_floor, list):
            post_floor = []
        if not isinstance(pre_floor, list):
            pre_floor = []
        if post_floor and isinstance(post_floor[0], Mapping):
            static_top1_memory_id = str(post_floor[0].get("memory_id", "")) or None
        static_selector_unavailable = (
            static_mapping.get("static_selector_unavailable") is True
        )
        try:
            static_floor = float(static_mapping.get("score_floor", float("nan")))
            static_k = int(static_mapping.get("shortlist_k", -1))
            pre_ids = [str(item["memory_id"]) for item in pre_floor]
            post_ids = [str(item["memory_id"]) for item in post_floor]
            pre_scores = [float(item["static_score"]) for item in pre_floor]
            post_scores = [float(item["static_score"]) for item in post_floor]
            expected_post = [
                item
                for item in pre_floor
                if float(item["static_score"]) >= static_floor
            ]
            static_lists_well_formed = True
        except (KeyError, TypeError, ValueError):
            static_floor = float("nan")
            static_k = -1
            pre_ids = []
            post_ids = []
            pre_scores = []
            post_scores = []
            expected_post = []
            static_lists_well_formed = False
        if len(post_floor) >= 2:
            expected_unavailable_reason = None
        elif not pre_floor:
            expected_unavailable_reason = "empty_bank"
        elif not post_floor:
            expected_unavailable_reason = "below_applicability_floor"
        else:
            expected_unavailable_reason = "insufficient_shortlist"
        static_query_ok = (
            bool(static_mapping)
            and static_mapping.get("schema_version")
            == "experience-memory-v3.5-static-shortlist-v1"
            and isinstance(static_query, Mapping)
            and bool(static_query.get("static_question_text_sha256"))
            and int(static_query.get("static_question_token_count", 0)) > 0
            and bool(static_query.get("static_question_token_ids_sha256"))
            and bool(static_query.get("static_question_embedding_sha256"))
            and int(static_query.get("layer_number", -1)) == 24
            and static_query.get("pooling") == "last_valid_token"
            and static_query.get("normalization") == "l2"
            and static_query.get("side_kv_disabled") is True
            and static_mapping.get("shortlist_fixed_for_generation") is True
            and static_mapping.get("retrieval_method") == "exact_cosine"
            and static_mapping.get("stable_tie_break")
            == "memory_id_ascending"
            and static_mapping.get("score_floor_tie_policy")
            == "retain_score_greater_than_or_equal_to_floor"
            and static_lists_well_formed
            and math.isfinite(static_floor)
            and -1.0 <= static_floor <= 1.0
            and 1 <= static_k <= 32
            and len(pre_floor) <= static_k
            and len(post_floor) <= static_k
            and len(pre_ids) == len(set(pre_ids))
            and len(post_ids) == len(set(post_ids))
            and all(pre_ids)
            and all(post_ids)
            and all(math.isfinite(score) for score in pre_scores + post_scores)
            and all(
                left_score > right_score
                or (left_score == right_score and left_id < right_id)
                for left_score, right_score, left_id, right_id in zip(
                    pre_scores, pre_scores[1:], pre_ids, pre_ids[1:]
                )
            )
            and [
                int(item.get("original_global_rank", -1))
                for item in pre_floor
            ]
            == list(range(1, len(pre_floor) + 1))
            and post_floor == expected_post
            and len(static_shortlist_ids) == len(set(static_shortlist_ids))
            and all(static_shortlist_ids)
            and static_shortlist_ids == post_ids
            and static_mapping.get("shortlist_nonempty") is bool(post_floor)
            and bool(static_mapping.get("applicability_bank_manifest_sha256"))
        )
        audit.check(
            static_query_ok,
            "v35_static_selector_trace_is_authenticated",
            sample_id,
        )
        available_shape_ok = (
            static_mapping.get("unavailable_reason")
            == expected_unavailable_reason
            and static_selector_unavailable
            is (expected_unavailable_reason is not None)
        )
        audit.check(
            available_shape_ok,
            "v35_static_selector_availability_is_consistent",
            sample_id,
        )
        audit.check(
            diagnostics.get("static_selector_unavailable")
            is static_selector_unavailable,
            "v35_static_selector_diagnostic_matches",
            sample_id,
        )
        audit.check(
            diagnostics.get("static_selector_unavailable_reason")
            == expected_unavailable_reason
            and int(diagnostics.get("static_shortlist_size", -1))
            == len(static_shortlist_ids)
            and diagnostics.get("static_shortlist_ids_sha256")
            == canonical_json_sha256(static_shortlist_ids)
            and diagnostics.get("static_shortlist_fixed_for_generation") is True
            and diagnostics.get("static_query_side_kv_disabled") is True
            and diagnostics.get("both_query_encodings_side_kv_disabled") is True,
            "v35_static_selector_diagnostics_are_complete",
            sample_id,
        )
        if static_top1_memory_id is not None:
            streaming.static_top1_memory_counts[static_top1_memory_id] += 1
        for memory_id in static_shortlist_ids:
            streaming.static_shortlist_memory_counts[memory_id] += 1
    if attempt_count:
        audit.check(
            str(attempts[0].get("outcome")) in {"activated", "abstained"},
            "first_attempt_is_activation_or_abstain",
            sample_id,
        )
    else:
        audit.check(
            not attention and runtime.get("final_memory_id") is None,
            "no_attempt_has_no_memory_exposure",
            sample_id,
        )
        audit.check(
            vanilla_ids == v3_ids,
            "zero_attempt_completion_matches_vanilla",
            sample_id,
        )

    selected_memory_ids = []
    query_encoding_seconds = 0.0
    retrieval_seconds = 0.0
    attempt_total_seconds = 0.0
    activation_kls = []
    activation_top1_change_count = 0
    first_top1_score = None
    first_margin = None
    first_selected_static_score = None
    expected_active_memory_id: str | None = None
    terminal_boundary_indices: list[int] = []
    clear_points: list[tuple[int, str]] = []
    for position, attempt in enumerate(attempts):
        decision = attempt.get("retrieval_decision", {})
        query = decision.get("query", {})
        hits = list(decision.get("hits", []))
        boundary_index = int(attempt.get("generated_boundary_index", -1))
        outcome = str(attempt.get("outcome", "unknown"))
        previous_id = (
            str(attempt.get("previous_memory_id"))
            if attempt.get("previous_memory_id") is not None
            else None
        )
        active_after_id = (
            str(attempt.get("active_memory_id_after"))
            if attempt.get("active_memory_id_after") is not None
            else None
        )
        audit.check(
            previous_id == expected_active_memory_id,
            "attempt_previous_memory_is_continuous",
            sample_id,
        )
        prompt_count = int(query.get("prompt_token_count", -1))
        partial_count = int(query.get("partial_cot_token_count", -1))
        query_count = int(query.get("query_token_count", -1))
        full_prefix_ok = (
            query_count == prompt_count + partial_count
            and partial_count == boundary_index + 1
            and query.get("context") == "question_plus_full_partial_cot"
            and (
                query.get("encoder_state")
                in {
                    "pure_prefix_reencode_side_kv_disabled",
                    "pure_prefix_side_kv_suspended",
                }
            )
            and bool(query.get("query_token_ids_sha256"))
            and bool(query.get("query_embedding_sha256"))
            and (
                not is_v35
                or (
                    query.get("side_kv_disabled") is True
                    and int(query.get("encoded_full_prefix_token_count", -1))
                    == query_count
                    and query.get("pooling") == "current_generated_token"
                    and query.get("normalization") == "l2"
                    and int(query.get("layer_number", -1)) == 24
                    and int(query.get("query_embedding_token_index", -1))
                    == query_count - 1
                    and int(query.get(
                        "query_embedding_causal_context_token_count", -1
                    ))
                    == query_count
                )
            )
        )
        if is_v35 and safety is not None:
            safety.violation("full_prefix_query", sample_id, not full_prefix_ok)
        audit.check(
            query_count == prompt_count + partial_count,
            "query_count_is_prompt_plus_full_partial",
            sample_id,
        )
        audit.check(
            partial_count == boundary_index + 1,
            "query_partial_reaches_current_boundary",
            sample_id,
        )
        audit.check(
            query.get("context") == "question_plus_full_partial_cot",
            "query_context_is_full_partial",
            sample_id,
        )
        audit.check(
            query.get("encoder_state")
            in {
                "pure_prefix_reencode_side_kv_disabled",
                "pure_prefix_side_kv_suspended",
            },
            "query_encoder_is_pure_prefix",
            sample_id,
        )
        query_pooling = str(
            query.get("pooling", V3_QUERY_POOLING_BOUNDARY_LAST)
        )
        audit.check(
            query_pooling in V3_QUERY_POOLING_METHODS,
            "query_pooling_is_known",
            sample_id,
        )
        if query_pooling in V3_QUERY_POOLING_METHODS:
            expected_embedding_index = query_embedding_token_index(
                token_count=query_count, pooling=query_pooling
            )
            has_position_audit = query.get(
                "query_embedding_token_index"
            ) is not None
            audit.check(
                has_position_audit
                or query_pooling == V3_QUERY_POOLING_BOUNDARY_LAST,
                "pre_boundary_query_has_position_audit",
                sample_id,
            )
            if has_position_audit:
                boundary_token_id = int(attempt.get("boundary_token_id", -1))
                position_ok = (
                    int(query.get("encoded_full_prefix_token_count", -1))
                    == query_count
                    and int(query.get("query_embedding_token_index", -1))
                    == expected_embedding_index
                    and int(query.get(
                        "query_embedding_causal_context_token_count", -1
                    ))
                    == expected_embedding_index + 1
                )
                if not is_v35:
                    position_ok = (
                        position_ok
                        and int(query.get("trigger_boundary_token_index", -1))
                        == query_count - 1
                        and int(query.get("trigger_boundary_token_id", -1))
                        == boundary_token_id
                        and bool(
                            query.get("trigger_boundary_excluded_from_pooling")
                        )
                        is (query_pooling == V3_QUERY_POOLING_PRE_BOUNDARY)
                    )
                audit.check(
                    position_ok,
                    "query_pooling_position_audit_is_consistent",
                    sample_id,
                )
                if attempt.get("query_embedding_token_id") is not None:
                    audit.check(
                        int(attempt["query_embedding_token_id"])
                        == int(query["query_embedding_token_id"]),
                        "attempt_query_token_matches_decision",
                        sample_id,
                    )
        audit.check(
            query.get("method")
            in {
                "exact_cosine",
                "exact_cosine_within_static_applicability_shortlist",
            },
            "retrieval_method_is_exact_cosine",
            sample_id,
        )
        embedding_transform = query.get("embedding_transform", "none")
        audit.check(
            embedding_transform
            in {"none", "key_bank_centroid_center_l2"},
            "retrieval_embedding_transform_is_known",
            sample_id,
        )
        query_norm = _finite_or_none(query.get("query_embedding_norm"))
        audit.check(
            query_norm is not None and math.isclose(query_norm, 1.0, abs_tol=1e-5),
            "query_embedding_is_unit_norm",
            sample_id,
        )
        search_query_norm = _finite_or_none(
            query.get("search_query_embedding_norm", query_norm)
        )
        audit.check(
            search_query_norm is not None
            and math.isclose(search_query_norm, 1.0, abs_tol=1e-5),
            "search_query_embedding_is_unit_norm",
            sample_id,
        )
        if embedding_transform == "key_bank_centroid_center_l2":
            audit.check(
                bool(query.get("raw_key_centroid_sha256"))
                and bool(query.get("search_key_embeddings_sha256"))
                and bool(query.get("search_query_embedding_sha256")),
                "centered_retrieval_artifacts_are_identified",
                sample_id,
            )
        selected_id = attempt.get("selected_memory_id")
        scores = [float(hit.get("score", float("nan"))) for hit in hits]
        empty_bank_abstain = (
            selected_id is None
            and str(attempt.get("outcome")) == "abstained"
            and str(decision.get("status")) == "empty_bank"
        )
        audit.check(
            (
                bool(scores)
                and all(math.isfinite(score) for score in scores)
                and scores == sorted(scores, reverse=True)
            )
            or (empty_bank_abstain and not scores),
            "retrieval_hits_are_finite_and_ranked",
            sample_id,
        )
        if is_v35:
            decision_shortlist_ids = _decision_shortlist_ids(decision)
            query_shortlist_ids = [
                str(value)
                for value in query.get("static_shortlist_ids", [])
            ]
            selected_inside = (
                selected_id is None
                or str(selected_id) in static_shortlist_ids
            )
            shortlist_consistent = (
                decision.get("schema_version")
                == V35_RETRIEVAL_DECISION_SCHEMA
                and decision_shortlist_ids == static_shortlist_ids
                and query_shortlist_ids == static_shortlist_ids
                and query.get("static_shortlist_fixed_for_generation") is True
                and query.get(
                    "dynamic_search_restricted_to_static_shortlist"
                )
                is True
                and all(
                    str(hit.get("memory_id", "")) in static_shortlist_ids
                    for hit in hits
                )
                and selected_inside
            )
            audit.check(
                shortlist_consistent,
                "v35_dynamic_search_stays_inside_static_shortlist",
                sample_id,
            )
            if safety is not None:
                safety.violation(
                    "selected_outside_shortlist",
                    sample_id,
                    not shortlist_consistent or not selected_inside,
                )
            status = str(decision.get("status", ""))
            threshold = _finite_or_none(
                query.get("minimum_top1_top2_margin")
            )
            margin = _finite_or_none(query.get("top1_top2_margin"))
            static_score = _finite_or_none(
                query.get("selected_memory_static_score")
            )
            score_floor = _finite_or_none(
                query.get("minimum_applicability_score")
            )
            expected_margin_passed = (
                margin is not None
                and (threshold is None or margin >= threshold)
            )
            expected_static_passed = (
                static_score is not None
                and score_floor is not None
                and static_score >= score_floor
            )
            decision_shape_ok = (
                status
                in {
                    "selected",
                    "static_shortlist_unavailable",
                    "below_applicability_floor",
                    "insufficient_shortlist",
                    "below_dynamic_margin",
                    "empty_bank",
                }
                and query.get("decision_reason") == status
                and query.get("static_condition_passed")
                is expected_static_passed
                and query.get("dynamic_margin_condition_passed")
                is expected_margin_passed
                and query.get("joint_admission_passed")
                is (status == "selected")
                and (
                    (
                        status == "selected"
                        and selected_id is not None
                        and expected_static_passed
                        and expected_margin_passed
                    )
                    or (
                        status != "selected"
                        and selected_id is None
                        and outcome == "abstained"
                    )
                )
            )
            audit.check(
                decision_shape_ok,
                "v35_static_dynamic_joint_admission_is_valid",
                sample_id,
            )
            kv_aligned = (
                status != "selected"
                or query.get("selected_memory_kv_metadata_aligned") is True
            )
            audit.check(
                kv_aligned,
                "v35_selected_memory_kv_metadata_is_aligned",
                sample_id,
            )
            if safety is not None:
                safety.violation("kv_alignment", sample_id, not kv_aligned)
        abstention_policy = query.get("abstention_policy")
        if abstention_policy is not None:
            threshold = _finite_or_none(
                query.get("minimum_top1_top2_margin")
            )
            margin = _finite_or_none(query.get("top1_top2_margin"))
            qualified = query.get("margin_qualified")
            if abstention_policy == "disabled":
                audit.check(
                    threshold is None and qualified is None,
                    "disabled_margin_abstention_has_no_threshold",
                    sample_id,
                )
            elif abstention_policy == "top1_top2_margin":
                status = str(decision.get("status"))
                expected_qualified = (
                    threshold is not None
                    and margin is not None
                    and margin >= threshold
                )
                audit.check(
                    threshold is not None
                    and threshold >= 0.0
                    and qualified is expected_qualified
                    and (
                        (
                            status == "selected"
                            and expected_qualified
                            and selected_id is not None
                        )
                        or (
                            status == "below_margin"
                            and not expected_qualified
                            and selected_id is None
                            and outcome == "abstained"
                        )
                    ),
                    "margin_abstention_decision_is_valid",
                    sample_id,
                )
            else:
                audit.check(
                    False,
                    "known_retrieval_abstention_policy",
                    sample_id,
                )
        if selected_id is not None:
            selected_id = str(selected_id)
            selected_memory_ids.append(selected_id)
            streaming.selected_memory_attempt_counts[selected_id] += 1
            audit.check(
                bool(hits) and hits[0].get("memory_id") == selected_id,
                "selected_memory_is_top1",
                sample_id,
            )
        abstain_shape = (
            selected_id is None
            and active_after_id == (None if is_v35 else previous_id)
        )
        expected_outcome_shape = {
            "abstained": abstain_shape,
            "activated": (
                previous_id is None
                and selected_id is not None
                and active_after_id == selected_id
            ),
            "replaced": (
                previous_id is not None
                and selected_id is not None
                and selected_id != previous_id
                and active_after_id == selected_id
            ),
            "duplicate": (
                previous_id is not None
                and selected_id == previous_id
                and active_after_id == previous_id
            ),
        }.get(outcome, False)
        audit.check(
            expected_outcome_shape,
            "attempt_outcome_memory_transition_is_valid",
            sample_id,
        )
        if is_v35:
            terminal = attempt.get("terminal_abstain")
            cleared = attempt.get("memory_cleared_on_abstain")
            cleared_id = (
                str(attempt.get("cleared_memory_id"))
                if attempt.get("cleared_memory_id") is not None
                else None
            )
            actual_path = attempt.get("actual_path_after_abstain")
            actual_memory_after = (
                str(attempt.get("actual_path_memory_id_after"))
                if attempt.get("actual_path_memory_id_after") is not None
                else None
            )
            if outcome == "abstained":
                terminal_boundary_indices.append(boundary_index)
                base_terminal_ok = (
                    terminal is True
                    and active_after_id is None
                    and actual_path == "native"
                    and actual_memory_after is None
                    and position == len(attempts) - 1
                )
                if previous_id is None:
                    clear_shape_ok = (
                        cleared is False
                        and cleared_id is None
                        and attempt.get("clear_affects_generated_token_index")
                        is None
                        and attempt.get("deactivation_forward_seconds") is None
                        and attempt.get("deactivation_first_step_logits_kl")
                        is None
                        and attempt.get("deactivation_first_step_top1_changed")
                        is None
                        and attempt.get("deactivation_baseline_first_token_id")
                        is None
                        and attempt.get("deactivation_native_first_token_id")
                        is None
                    )
                else:
                    clear_index = int(attempt.get(
                        "clear_affects_generated_token_index", -1
                    ))
                    deactivation_seconds = _finite_or_none(
                        attempt.get("deactivation_forward_seconds")
                    )
                    deactivation_kl = _finite_or_none(
                        attempt.get("deactivation_first_step_logits_kl")
                    )
                    clear_shape_ok = (
                        cleared is True
                        and cleared_id == previous_id
                        and clear_index == boundary_index + 1
                        and deactivation_seconds is not None
                        and deactivation_seconds >= 0.0
                        and deactivation_kl is not None
                        and deactivation_kl >= 0.0
                        and isinstance(
                            attempt.get("deactivation_first_step_top1_changed"),
                            bool,
                        )
                        and isinstance(
                            attempt.get("deactivation_baseline_first_token_id"),
                            int,
                        )
                        and isinstance(
                            attempt.get("deactivation_native_first_token_id"),
                            int,
                        )
                        and attempt.get("deactivation_first_step_top1_changed")
                        is (
                            attempt.get("deactivation_baseline_first_token_id")
                            != attempt.get("deactivation_native_first_token_id")
                        )
                        and 0 <= clear_index < len(v3_ids)
                        and int(attempt.get("deactivation_native_first_token_id"))
                        == v3_ids[clear_index]
                    )
                    if cleared_id is not None:
                        clear_points.append((boundary_index, cleared_id))
                terminal_shape_ok = base_terminal_ok and clear_shape_ok
            else:
                terminal_shape_ok = (
                    terminal is False
                    and cleared is False
                    and cleared_id is None
                    and actual_path is None
                    and actual_memory_after == active_after_id
                    and attempt.get("deactivation_forward_seconds") is None
                    and attempt.get("deactivation_first_step_logits_kl") is None
                    and attempt.get("deactivation_first_step_top1_changed") is None
                    and attempt.get("deactivation_baseline_first_token_id") is None
                    and attempt.get("deactivation_native_first_token_id") is None
                    and attempt.get("clear_affects_generated_token_index") is None
                )
            audit.check(
                terminal_shape_ok,
                "v35_terminal_abstain_lifecycle_is_valid",
                sample_id,
            )
            if safety is not None:
                safety.violation(
                    "terminal_state_drift", sample_id, not terminal_shape_ok
                )
        expected_active_memory_id = active_after_id
        if position == 0:
            first_top1_score = _finite_or_none(query.get("top1_score"))
            first_margin = _finite_or_none(query.get("top1_top2_margin"))
            first_selected_static_score = _finite_or_none(
                query.get("selected_memory_static_score")
            )
        query_encoding_seconds += float(attempt.get("query_encoding_seconds", 0.0))
        retrieval_seconds += float(attempt.get("retrieval_seconds", 0.0))
        attempt_total_seconds += float(attempt.get("attempt_total_seconds", 0.0))
        kl = _finite_or_none(attempt.get("activation_first_step_logits_kl"))
        if kl is not None:
            activation_kls.append(kl)
        top1_changed = attempt.get("activation_first_step_top1_changed")
        activation_top1_change_count += int(top1_changed is True)
        audit.check(
            (
                outcome in {"activated", "replaced"}
                and kl is not None
                and isinstance(top1_changed, bool)
            )
            or (
                outcome in {"duplicate", "abstained"}
                and kl is None
                and top1_changed is None
            ),
            "activation_counterfactual_matches_outcome",
            sample_id,
        )
        timing_values = (
            attempt.get("query_encoding_seconds"),
            attempt.get("retrieval_seconds"),
            attempt.get("attempt_total_seconds"),
        )
        audit.check(
            all(
                _finite_or_none(value) is not None and float(value) >= 0.0
                for value in timing_values
            ),
            "attempt_timings_are_finite_and_nonnegative",
            sample_id,
        )

    if is_v35:
        selected_outside_count = sum(
            memory_id not in static_shortlist_ids
            for memory_id in selected_memory_ids
        )
        audit.check(
            diagnostics.get("dynamic_search_restricted_to_static_shortlist")
            is True
            and diagnostics.get(
                "selected_memory_belongs_to_static_shortlist"
            )
            is (selected_outside_count == 0)
            and int(diagnostics.get(
                "selected_outside_static_shortlist_count", -1
            ))
            == selected_outside_count,
            "v35_dynamic_shortlist_diagnostics_match_attempts",
            sample_id,
        )
        if safety is not None:
            safety.violation(
                "selected_outside_shortlist",
                sample_id,
                selected_outside_count != 0,
            )

    boundary_by_index = {
        int(item.get("generated_boundary_index", -1)): item for item in boundaries
    }
    audit.check(
        len(boundary_by_index) == len(boundaries),
        "boundary_indices_are_unique",
        sample_id,
    )
    for boundary in boundaries:
        audit.check(
            boundary.get("risk_role") == expected_risk_role,
            "risk_role_matches_system_profile",
            sample_id,
        )
        audit.check(
            math.isfinite(float(boundary.get("entropy", float("nan"))))
            and math.isfinite(
                float(boundary.get("persistence_risk_score", float("nan")))
            ),
            "boundary_entropy_and_risk_are_finite",
            sample_id,
        )
        if expected_risk_role == "online_joint_control":
            audit.check(
                boundary.get("trace_scope")
                == "every_pre_answer_generated_token"
                and math.isfinite(
                    float(boundary.get("vocabulary_entropy", float("nan")))
                )
                and math.isfinite(
                    float(
                        boundary.get(
                            "top1_top2_logit_margin", float("nan")
                        )
                    )
                ),
                "continuous_token_diagnostics_are_complete",
                sample_id,
            )
    if expected_risk_role == "online_joint_control":
        audit.check(
            [
                int(item.get("generated_observation_index", -1))
                for item in boundaries
            ]
            == list(range(len(boundaries))),
            "continuous_gate_observation_indices_are_contiguous",
            sample_id,
        )
        conditioned_count = sum(
            item.get("active_memory_conditioned") is True
            for item in boundaries
        )
        qualified_count = sum(
            item.get("joint_trigger_qualified") is True
            for item in boundaries
        )
        audit.check(
            int(diagnostics.get("gate_observation_count", -1))
            == len(boundaries)
            and int(
                diagnostics.get(
                    "memory_conditioned_gate_observation_count", -1
                )
            )
            == conditioned_count
            and int(
                diagnostics.get("native_gate_observation_count", -1)
            )
            == len(boundaries) - conditioned_count
            and int(
                diagnostics.get("joint_trigger_qualified_count", -1)
            )
            == qualified_count,
            "continuous_gate_diagnostic_counts_match",
            sample_id,
        )
        for item in boundaries:
            action = str(item.get("action", ""))
            if action == "rearmed":
                rearm_ok = (
                    int(item.get("low_entropy_streak_before", -1)) == 1
                    and int(item.get("low_entropy_streak_after", -1)) == 0
                    and item.get("state_before") == "DISARMED"
                    and item.get("state_after") == "ARMED"
                    and int(item.get("retrieval_attempt_count_before", -1))
                    == int(item.get("retrieval_attempt_count_after", -2))
                    and float(item.get("entropy", float("inf")))
                    <= float(item.get("low_entropy_threshold", float("-inf")))
                )
                audit.check(
                    rearm_ok,
                    "continuous_rearm_requires_second_low_token",
                    sample_id,
                )
                if is_v35 and safety is not None:
                    safety.violation("rearm", sample_id, not rearm_ok)
            if action == "retrieval_attempt":
                audit.check(
                    item.get("joint_trigger_qualified") is True
                    and float(item.get("entropy", float("-inf")))
                    >= float(item.get("high_entropy_threshold", float("inf")))
                    and float(
                        item.get("persistence_risk_score", float("-inf"))
                    )
                    > float(
                        item.get("persistence_risk_threshold", float("inf"))
                    ),
                    "continuous_attempt_requires_joint_entropy_risk",
                    sample_id,
                )
        if is_v35:
            summary = runtime.get("summary", {})
            rearm_summary_ok = (
                summary.get("two_low_rearm_respected") is True
                and summary.get("second_low_rearms_without_trigger") is True
                and summary.get("no_rearm_after_terminal_abstain") is True
                and diagnostics.get("two_low_rearm_respected") is True
                and diagnostics.get("second_low_rearms_without_trigger") is True
                and diagnostics.get("no_rearm_after_terminal_abstain") is True
            )
            terminal_state_ok = True
            if terminal_boundary_indices:
                first_terminal = min(terminal_boundary_indices)
                terminal_trace = boundary_by_index.get(first_terminal, {})
                terminal_state_ok = (
                    terminal_trace.get("action") == "retrieval_attempt"
                    and terminal_trace.get("state_after") == "EXHAUSTED"
                    and not any(
                        int(item.get("generated_boundary_index", -1))
                        > first_terminal
                        and item.get("action")
                        in {"rearmed", "retrieval_attempt"}
                        for item in boundaries
                    )
                )
            audit.check(
                rearm_summary_ok,
                "v35_rearm_summary_is_safe",
                sample_id,
            )
            audit.check(
                terminal_state_ok,
                "v35_terminal_abstain_exhausts_gate",
                sample_id,
            )
            if safety is not None:
                safety.violation("rearm", sample_id, not rearm_summary_ok)
                safety.violation(
                    "terminal_state_drift", sample_id, not terminal_state_ok
                )
    first_trigger_entropy = None
    first_trigger_boundary_index = None
    if attempts:
        first_trigger_boundary_index = int(attempts[0].get(
            "generated_boundary_index", -1
        ))
        first_boundary = boundary_by_index.get(first_trigger_boundary_index, {})
        first_trigger_entropy = _finite_or_none(first_boundary.get("entropy"))
        for attempt in attempts:
            matching_boundary = boundary_by_index.get(
                int(attempt.get("generated_boundary_index", -1)), {}
            )
            audit.check(
                matching_boundary.get("action") == "retrieval_attempt",
                "attempt_has_matching_boundary_trace",
                sample_id,
            )

    audit.check(
        (
            str(runtime.get("final_memory_id"))
            if runtime.get("final_memory_id") is not None
            else None
        )
        == expected_active_memory_id,
        "final_memory_matches_attempt_state",
        sample_id,
    )

    masses = []
    for trace in attention:
        memory_id = str(trace.get("memory_id", ""))
        mass = float(trace.get("memory_attention_mass", float("nan")))
        native_length = int(trace.get("native_key_length", -1))
        processed_length = int(trace.get("processed_prefix_token_count", -2))
        audit.check(
            int(trace.get("layer_number", -1)) == 24,
            "attention_is_layer_24",
            sample_id,
        )
        audit.check(
            native_length == processed_length,
            "native_cache_excludes_memory_slots",
            sample_id,
        )
        audit.check(
            math.isfinite(mass) and mass > 0.0,
            "memory_attention_mass_is_positive",
            sample_id,
        )
        if math.isfinite(mass):
            masses.append(mass)
            streaming.attention_by_memory_count[memory_id] += 1
            streaming.attention_by_memory_sum[memory_id] += mass
    audit.check(
        len(attention) == int(diagnostics.get("memory_attention_step_count", -1)),
        "attention_step_count_matches",
        sample_id,
    )
    stale_attention_count = sum(
        int(trace.get("generated_input_index", -1)) >= clear_index
        and str(trace.get("memory_id", "")) == cleared_memory_id
        for clear_index, cleared_memory_id in clear_points
        for trace in attention
    )
    active_memory_lifetime_tokens = sum(
        int(span.get("attention_step_count", 0))
        for span in runtime.get("memory_activation_spans", [])
        if isinstance(span, Mapping)
    )
    if is_v35:
        transitions = list(runtime.get("memory_transitions", []))
        clear_transitions = [
            transition
            for transition in transitions
            if transition.get("transition")
            == "deactivated_on_terminal_abstain"
        ]
        transition_shape_ok = (
            len(clear_transitions) == clear_on_terminal_abstain_count
            and all(
                transition.get("previous_memory_id") is not None
                and transition.get("next_memory_id") is None
                and int(transition.get("affects_generated_token_index", -1))
                == int(transition.get("generated_boundary_index", -2)) + 1
                for transition in clear_transitions
            )
        )
        stale_summary_ok = (
            stale_attention_count == 0
            and int(diagnostics.get(
                "stale_memory_attention_after_terminal_clear_count", -1
            ))
            == stale_attention_count
            and diagnostics.get("terminal_clear_attention_safe") is True
            and int(runtime.get("summary", {}).get(
                "stale_memory_attention_after_terminal_clear_count", -1
            ))
            == stale_attention_count
            and runtime.get("summary", {}).get(
                "terminal_clear_attention_safe"
            )
            is True
        )
        kv_summary_ok = (
            diagnostics.get("selected_memory_kv_metadata_aligned") is True
            and int(diagnostics.get(
                "selected_memory_kv_alignment_unlogged_count", -1
            ))
            == 0
        )
        attempt_summary_ok = (
            diagnostics.get("attempt_budget_respected") is True
            and runtime.get("summary", {}).get(
                "max_three_attempts_respected"
            )
            is True
        )
        full_prefix_summary_ok = (
            diagnostics.get("query_context_is_full_prefix") is True
            and diagnostics.get("both_query_encodings_side_kv_disabled") is True
        )
        audit.check(
            transition_shape_ok,
            "v35_terminal_clear_transition_is_valid",
            sample_id,
        )
        audit.check(
            stale_summary_ok,
            "v35_no_stale_attention_after_terminal_clear",
            sample_id,
        )
        audit.check(
            kv_summary_ok,
            "v35_kv_alignment_diagnostics_are_safe",
            sample_id,
        )
        audit.check(
            attempt_summary_ok,
            "v35_attempt_budget_diagnostics_are_safe",
            sample_id,
        )
        audit.check(
            full_prefix_summary_ok,
            "v35_full_prefix_query_diagnostics_are_safe",
            sample_id,
        )
        initial_state_ok = (
            runtime.get("summary", {}).get("initial_gate_state")
            == ("EXHAUSTED" if static_selector_unavailable else "ARMED")
        )
        final_state = runtime.get("final_gate_state")
        final_memory = runtime.get("final_memory_id")
        final_state_ok = (
            runtime.get("summary", {}).get("static_selector_unavailable")
            is static_selector_unavailable
            and runtime.get("summary", {}).get("final_gate_state")
            == final_state
            and runtime.get("summary", {}).get("final_memory_id")
            == final_memory
            and diagnostics.get("final_gate_state") == final_state
            and diagnostics.get("final_memory_id") == final_memory
            and (
                not terminal_boundary_indices
                or (final_state == "EXHAUSTED" and final_memory is None)
            )
            and (
                not static_selector_unavailable
                or (final_state == "EXHAUSTED" and final_memory is None)
            )
        )
        audit.check(
            initial_state_ok,
            "v35_static_availability_sets_initial_gate_state",
            sample_id,
        )
        audit.check(
            final_state_ok,
            "v35_final_gate_and_memory_state_is_consistent",
            sample_id,
        )
        if safety is not None:
            safety.violation(
                "stale_attention_after_terminal_clear",
                sample_id,
                not stale_summary_ok,
            )
            safety.violation(
                "terminal_state_drift",
                sample_id,
                not transition_shape_ok
                or not initial_state_ok
                or not final_state_ok,
            )
            safety.violation("kv_alignment", sample_id, not kv_summary_ok)
            safety.violation(
                "attempt_budget", sample_id, not attempt_summary_ok
            )
            safety.violation(
                "full_prefix_query", sample_id, not full_prefix_summary_ok
            )

    cache_parity = row.get("cache_parity")
    if cache_parity is not None:
        streaming.cache_parity_checked += 1
        if cache_parity.get("exact_match") is not True:
            streaming.cache_parity_failed += 1
        audit.check(
            cache_parity.get("exact_match") is True,
            "sampled_native_cache_parity",
            sample_id,
        )

    rearm_count = sum(
        boundary.get("action") == "rearmed" for boundary in boundaries
    )
    audit.check(
        rearm_count == int(diagnostics.get("rearm_count", -1)),
        "rearm_count_matches",
        sample_id,
    )
    runtime_summary = runtime.get("summary", {})
    for field, expected_value in (
        ("retrieval_attempt_count", attempt_count),
        ("rearm_count", rearm_count),
        ("replacement_count", replacement_count),
        ("duplicate_count", duplicate_count),
        ("memory_attention_step_count", len(attention)),
    ):
        audit.check(
            int(runtime_summary.get(field, -1)) == expected_value,
            f"runtime_summary_{field}_matches",
            sample_id,
        )
    if is_v35:
        for field, expected_value in (
            ("abstain_count", abstain_count),
            ("terminal_abstain_count", terminal_abstain_count),
            (
                "clear_on_terminal_abstain_count",
                clear_on_terminal_abstain_count,
            ),
        ):
            audit.check(
                int(runtime_summary.get(field, -1)) == expected_value
                and int(diagnostics.get(field, -1)) == expected_value,
                f"v35_{field}_matches",
                sample_id,
            )
    return CompactSample(
        sample_id=sample_id,
        vanilla_strict=vanilla_strict,
        v3_strict=v3_strict,
        vanilla_format=vanilla_format,
        v3_format=v3_format,
        vanilla_tokens=vanilla_tokens,
        v3_tokens=v3_tokens,
        completion_exact_match=vanilla_ids == v3_ids,
        attempt_count=attempt_count,
        rearm_count=rearm_count,
        activation_count=activation_count,
        replacement_count=replacement_count,
        duplicate_count=duplicate_count,
        abstain_count=abstain_count,
        outcome_sequence=outcome_sequence,
        first_memory_id=selected_memory_ids[0] if selected_memory_ids else None,
        final_memory_id=(
            str(runtime.get("final_memory_id"))
            if runtime.get("final_memory_id") is not None
            else None
        ),
        first_top1_score=first_top1_score,
        first_top1_top2_margin=first_margin,
        first_trigger_entropy=first_trigger_entropy,
        first_trigger_boundary_index=first_trigger_boundary_index,
        mean_activation_kl=(
            sum(activation_kls) / len(activation_kls) if activation_kls else None
        ),
        max_activation_kl=max(activation_kls) if activation_kls else None,
        activation_top1_change_count=activation_top1_change_count,
        attention_step_count=len(attention),
        mean_attention_mass=sum(masses) / len(masses) if masses else None,
        max_attention_mass=max(masses) if masses else None,
        min_attention_mass=min(masses) if masses else None,
        query_encoding_seconds=query_encoding_seconds,
        retrieval_seconds=retrieval_seconds,
        attempt_total_seconds=attempt_total_seconds,
        static_selector_unavailable=static_selector_unavailable,
        static_shortlist_size=len(static_shortlist_ids),
        static_top1_memory_id=static_top1_memory_id,
        first_selected_static_score=first_selected_static_score,
        terminal_abstain_count=terminal_abstain_count,
        clear_on_terminal_abstain_count=clear_on_terminal_abstain_count,
        active_memory_lifetime_tokens=active_memory_lifetime_tokens,
        vanilla_numeric_correct_but_format_invalid=(
            vanilla_numeric_format_invalid
        ),
        v3_numeric_correct_but_format_invalid=v3_numeric_format_invalid,
        vanilla_answer_marker_seen=vanilla_marker_seen,
        v3_answer_marker_seen=v3_marker_seen,
        vanilla_first_answer_marker_token_index=vanilla_marker_index,
        v3_first_answer_marker_token_index=v3_marker_index,
        attempt_to_first_answer_marker_distances=tuple(
            marker_distance["distances"]
        ),
        attempts_with_subsequent_answer_marker_count=int(
            marker_distance["attempts_with_subsequent_answer_marker_count"]
        ),
        late_attempt_within_32_tokens_count=int(
            marker_distance["late_attempt_within_32_tokens_count"]
        ),
        marker_missing_attempt_count=int(
            marker_distance["marker_missing_attempt_count"]
        ),
        marker_not_after_attempt_count=int(
            marker_distance["marker_not_after_attempt_count"]
        ),
    )


def paired_metric_summary(
    samples: Sequence[CompactSample],
    *,
    metric: str,
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    if metric not in {"strict", "format"}:
        raise ValueError("paired metric must be strict or format")
    vanilla_values = [bool(getattr(item, f"vanilla_{metric}")) for item in samples]
    v3_values = [bool(getattr(item, f"v3_{metric}")) for item in samples]
    both_correct = sum(
        vanilla and v3 for vanilla, v3 in zip(vanilla_values, v3_values)
    )
    vanilla_only = sum(
        vanilla and not v3 for vanilla, v3 in zip(vanilla_values, v3_values)
    )
    v3_only = sum(
        not vanilla and v3 for vanilla, v3 in zip(vanilla_values, v3_values)
    )
    both_wrong = len(samples) - both_correct - vanilla_only - v3_only
    deltas = [
        int(v3) - int(vanilla)
        for vanilla, v3 in zip(vanilla_values, v3_values)
    ]
    return {
        "sample_count": len(samples),
        "vanilla_correct_count": sum(vanilla_values),
        "v3_correct_count": sum(v3_values),
        "vanilla_accuracy": (
            sum(vanilla_values) / len(samples) if samples else None
        ),
        "v3_accuracy": sum(v3_values) / len(samples) if samples else None,
        "delta_v3_minus_vanilla": sum(deltas) / len(samples) if samples else None,
        "net_correct_count_delta": sum(deltas),
        "paired_table": {
            "both_correct": both_correct,
            "vanilla_only_correct_harmed": vanilla_only,
            "v3_only_correct_improved": v3_only,
            "both_wrong": both_wrong,
            "discordant_count": vanilla_only + v3_only,
        },
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
            vanilla_only, v3_only
        ),
        "paired_bootstrap_95_ci": bootstrap_mean_ci(
            deltas,
            resamples=bootstrap_resamples,
            seed=seed,
        ),
    }


def scope_summary(
    samples: Sequence[CompactSample],
    *,
    bootstrap_resamples: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    token_deltas = [item.token_delta for item in samples]
    vanilla_format_wrong = sum(
        item.vanilla_format and not item.vanilla_strict for item in samples
    )
    v3_format_wrong = sum(
        item.v3_format and not item.v3_strict for item in samples
    )
    vanilla_numeric_format_invalid = sum(
        item.vanilla_numeric_correct_but_format_invalid for item in samples
    )
    v3_numeric_format_invalid = sum(
        item.v3_numeric_correct_but_format_invalid for item in samples
    )
    marker_pairs = Counter(
        (
            item.vanilla_answer_marker_seen,
            item.v3_answer_marker_seen,
        )
        for item in samples
        if item.vanilla_answer_marker_seen is not None
        and item.v3_answer_marker_seen is not None
    )
    marker_distances = [
        distance
        for item in samples
        for distance in item.attempt_to_first_answer_marker_distances
    ]
    attempt_count = sum(item.attempt_count for item in samples)
    subsequent_marker_count = sum(
        item.attempts_with_subsequent_answer_marker_count for item in samples
    )
    late_attempt_count = sum(
        item.late_attempt_within_32_tokens_count for item in samples
    )
    return {
        "sample_count": len(samples),
        "strict": paired_metric_summary(
            samples,
            metric="strict",
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        "format": paired_metric_summary(
            samples,
            metric="format",
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 1,
        ),
        "formatted_but_strictly_wrong": {
            "vanilla": vanilla_format_wrong,
            "v3": v3_format_wrong,
            "delta_v3_minus_vanilla": v3_format_wrong - vanilla_format_wrong,
        },
        "descriptive_numeric_correct_but_format_invalid": {
            "formal_metric": False,
            "vanilla_count": vanilla_numeric_format_invalid,
            "v3_count": v3_numeric_format_invalid,
            "delta_v3_minus_vanilla": (
                v3_numeric_format_invalid - vanilla_numeric_format_invalid
            ),
        },
        "descriptive_answer_marker_pairs": {
            "both_seen": marker_pairs[(True, True)],
            "vanilla_seen_v3_absent": marker_pairs[(True, False)],
            "vanilla_absent_v3_seen": marker_pairs[(False, True)],
            "both_absent": marker_pairs[(False, False)],
            "paired_observation_count": sum(marker_pairs.values()),
        },
        "descriptive_attempt_to_first_answer_marker": {
            "formal_metric": False,
            "attempt_count": attempt_count,
            "attempts_with_subsequent_answer_marker_count": (
                subsequent_marker_count
            ),
            "distance_tokens": numeric_summary(marker_distances),
            "late_attempt_within_32_tokens_count": late_attempt_count,
            "late_attempt_rate_over_all_attempts": (
                late_attempt_count / attempt_count if attempt_count else None
            ),
            "late_attempt_rate_over_attempts_with_subsequent_marker": (
                late_attempt_count / subsequent_marker_count
                if subsequent_marker_count
                else None
            ),
            "marker_missing_attempt_count": sum(
                item.marker_missing_attempt_count for item in samples
            ),
            "marker_not_after_attempt_count": sum(
                item.marker_not_after_attempt_count for item in samples
            ),
        },
        "tokens": {
            "vanilla": numeric_summary([item.vanilla_tokens for item in samples]),
            "v3": numeric_summary([item.v3_tokens for item in samples]),
            "paired_delta_v3_minus_vanilla": numeric_summary(token_deltas),
            "positive_delta_count": sum(value > 0 for value in token_deltas),
            "zero_delta_count": sum(value == 0 for value in token_deltas),
            "negative_delta_count": sum(value < 0 for value in token_deltas),
        },
        "completion_exact_match_count": sum(
            item.completion_exact_match for item in samples
        ),
        "completion_exact_match_rate": (
            sum(item.completion_exact_match for item in samples) / len(samples)
            if samples
            else None
        ),
    }


def grouped_scope_summary(
    samples: Sequence[CompactSample],
    key: Callable[[CompactSample], str],
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[CompactSample]] = defaultdict(list)
    for sample in samples:
        groups[key(sample)].append(sample)
    return [
        {"group": label, **scope_summary(group)}
        for label, group in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]


def marker_proximity_group(sample: CompactSample) -> str:
    if not sample.attempt_count:
        return "no_attempt"
    if sample.late_attempt_within_32_tokens_count:
        return "late_attempt_within_32_tokens"
    if sample.marker_missing_attempt_count:
        return "answer_marker_missing"
    if sample.marker_not_after_attempt_count:
        return "answer_marker_not_after_attempt"
    return "subsequent_marker_more_than_32_tokens"


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average
        index = end
    return ranks


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(
        left * right for left, right in zip(centered_x, centered_y)
    ) / denominator


def association_summary(
    samples: Sequence[CompactSample], attribute: str
) -> dict[str, Any]:
    selected = [
        sample for sample in samples if getattr(sample, attribute) is not None
    ]
    x = [float(getattr(sample, attribute)) for sample in selected]
    strict = [float(sample.strict_delta) for sample in selected]
    formatting = [float(sample.format_delta) for sample in selected]
    return {
        "sample_count": len(selected),
        "attribute": attribute,
        "attribute_summary": numeric_summary(x),
        "strict_delta": {
            "pearson": pearson_correlation(x, strict),
            "spearman": pearson_correlation(_average_ranks(x), _average_ranks(strict)),
        },
        "format_delta": {
            "pearson": pearson_correlation(x, formatting),
            "spearman": pearson_correlation(
                _average_ranks(x), _average_ranks(formatting)
            ),
        },
    }


def quartile_summary(
    samples: Sequence[CompactSample], attribute: str
) -> list[dict[str, Any]]:
    selected = sorted(
        (sample for sample in samples if getattr(sample, attribute) is not None),
        key=lambda sample: (float(getattr(sample, attribute)), sample.sample_id),
    )
    if not selected:
        return []
    result = []
    for quartile in range(4):
        start = len(selected) * quartile // 4
        end = len(selected) * (quartile + 1) // 4
        group = selected[start:end]
        if not group:
            continue
        values = [float(getattr(sample, attribute)) for sample in group]
        result.append({
            "quartile": quartile + 1,
            "range_min": min(values),
            "range_max": max(values),
            **scope_summary(group),
        })
    return result


def sample_brief(sample: CompactSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "token_delta": sample.token_delta,
        "vanilla_tokens": sample.vanilla_tokens,
        "v3_tokens": sample.v3_tokens,
        "strict_transition": f"{int(sample.vanilla_strict)}->{int(sample.v3_strict)}",
        "format_transition": f"{int(sample.vanilla_format)}->{int(sample.v3_format)}",
        "attempt_count": sample.attempt_count,
        "outcome_sequence": sample.outcome_sequence,
        "first_memory_id": sample.first_memory_id,
        "final_memory_id": sample.final_memory_id,
        "first_top1_top2_margin": sample.first_top1_top2_margin,
        "first_selected_static_score": sample.first_selected_static_score,
        "static_shortlist_size": sample.static_shortlist_size,
        "static_selector_unavailable": sample.static_selector_unavailable,
        "terminal_abstain_count": sample.terminal_abstain_count,
        "clear_on_terminal_abstain_count": (
            sample.clear_on_terminal_abstain_count
        ),
        "first_answer_marker_token_index": (
            sample.v3_first_answer_marker_token_index
        ),
        "attempt_to_first_answer_marker_distances": list(
            sample.attempt_to_first_answer_marker_distances
        ),
        "late_attempt_within_32_tokens_count": (
            sample.late_attempt_within_32_tokens_count
        ),
        "active_memory_lifetime_tokens": sample.active_memory_lifetime_tokens,
        "mean_activation_kl": sample.mean_activation_kl,
        "mean_attention_mass": sample.mean_attention_mass,
    }


def memory_group_analysis(
    samples: Sequence[CompactSample],
    *,
    attribute: str,
    min_samples: int,
    top_k: int,
) -> dict[str, Any]:
    groups: defaultdict[str, list[CompactSample]] = defaultdict(list)
    for sample in samples:
        memory_id = getattr(sample, attribute)
        if memory_id is not None:
            groups[str(memory_id)].append(sample)
    summaries = [
        {"memory_id": memory_id, **scope_summary(group)}
        for memory_id, group in groups.items()
    ]
    by_frequency = sorted(
        summaries,
        key=lambda item: (-int(item["sample_count"]), str(item["memory_id"])),
    )[:top_k]
    eligible = [
        item for item in summaries if int(item["sample_count"]) >= min_samples
    ]
    best = sorted(
        eligible,
        key=lambda item: (
            -int(item["strict"]["net_correct_count_delta"]),
            -int(item["sample_count"]),
            str(item["memory_id"]),
        ),
    )[:top_k]
    worst = sorted(
        eligible,
        key=lambda item: (
            int(item["strict"]["net_correct_count_delta"]),
            -int(item["sample_count"]),
            str(item["memory_id"]),
        ),
    )[:top_k]
    return {
        "grouping_attribute": attribute,
        "memory_count": len(groups),
        "minimum_samples_for_effect_ranking": min_samples,
        "top_by_frequency": by_frequency,
        "best_strict_net_effect": best,
        "worst_strict_net_effect": worst,
        "warning": "Exploratory grouping; no multiple-comparison correction.",
    }


def concentration_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    """Describe selector concentration without treating geometry as a metric."""

    normalized = {
        str(memory_id): int(count)
        for memory_id, count in counts.items()
        if int(count) > 0
    }
    total = sum(normalized.values())
    if not total:
        return {
            "selection_count": 0,
            "selected_memory_count": 0,
            "top1_share": None,
            "gini_over_selected_memories": None,
            "frequency": [],
        }
    values = sorted(normalized.values())
    weighted = sum(
        (index + 1) * value for index, value in enumerate(values)
    )
    count = len(values)
    gini = (2.0 * weighted) / (count * total) - (count + 1.0) / count
    frequency = [
        {"memory_id": memory_id, "count": value}
        for memory_id, value in sorted(
            normalized.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "selection_count": total,
        "selected_memory_count": count,
        "top1_share": frequency[0]["count"] / total,
        "gini_over_selected_memories": gini,
        "frequency": frequency,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    overall = report["paired_analysis"]["overall"]
    triggered = report["paired_analysis"]["triggered"]
    no_trigger = report["paired_analysis"]["no_trigger"]
    strict = overall["strict"]
    formatting = overall["format"]
    mechanism = report["mechanism"]
    integrity = report["integrity"]

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.4f}%"

    def confidence_interval(metric: Mapping[str, Any]) -> str:
        interval = metric.get("paired_bootstrap_95_ci")
        if interval is None:
            return "n/a"
        return (
            f"[{percent(interval['lower'])}, {percent(interval['upper'])}]"
        )

    def scope_table(groups: Sequence[Mapping[str, Any]]) -> str:
        rows = [
            "| Group | N | Strict net | Improved | Harmed | Format net | Mean token delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for group in groups:
            group_strict = group["strict"]
            group_format = group["format"]
            token_delta = group["tokens"]["paired_delta_v3_minus_vanilla"]
            rows.append(
                f"| {group['group']} | {group['sample_count']} | "
                f"{group_strict['net_correct_count_delta']} | "
                f"{group_strict['paired_table']['v3_only_correct_improved']} | "
                f"{group_strict['paired_table']['vanilla_only_correct_harmed']} | "
                f"{group_format['net_correct_count_delta']} | "
                f"{token_delta['mean']} |"
            )
        return "\n".join(rows)

    def memory_table(items: Sequence[Mapping[str, Any]]) -> str:
        rows = [
            "| Memory ID | N | Strict net | Improved | Harmed | Mean token delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in items[:10]:
            item_strict = item["strict"]
            rows.append(
                f"| {item['memory_id']} | {item['sample_count']} | "
                f"{item_strict['net_correct_count_delta']} | "
                f"{item_strict['paired_table']['v3_only_correct_improved']} | "
                f"{item_strict['paired_table']['vanilla_only_correct_harmed']} | "
                f"{item['tokens']['paired_delta_v3_minus_vanilla']['mean']} |"
            )
        return "\n".join(rows)

    def outlier_table(items: Sequence[Mapping[str, Any]]) -> str:
        rows = [
            "| Sample | Token delta | Strict | Attempts | Outcomes | Memory |",
            "|---|---:|---:|---:|---|---|",
        ]
        for item in items[:10]:
            rows.append(
                f"| {item['sample_id']} | {item['token_delta']} | "
                f"{item['strict_transition']} | {item['attempt_count']} | "
                f"{item['outcome_sequence']} | {item['final_memory_id']} |"
            )
        return "\n".join(rows)

    def signal_table(attributes: Sequence[str]) -> str:
        correlations = report["exploratory_associations"]["correlations"]
        quartiles = report["exploratory_associations"][
            "equal_count_rank_quartiles"
        ]
        rows = [
            "| Signal | N | Mean | Strict Pearson | Strict Spearman | Q1→Q4 strict net |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for attribute in attributes:
            association = correlations[attribute]
            summary = association["attribute_summary"]
            quartile_nets = [
                str(item["strict"]["net_correct_count_delta"])
                for item in quartiles[attribute]
            ]
            rows.append(
                f"| {attribute} | {association['sample_count']} | "
                f"{summary['mean'] if summary else 'n/a'} | "
                f"{association['strict_delta']['pearson']} | "
                f"{association['strict_delta']['spearman']} | "
                f"{' → '.join(quartile_nets) or 'n/a'} |"
            )
        return "\n".join(rows)

    strict_net = int(strict["net_correct_count_delta"])
    if strict_net > 0:
        strict_read = f"V3 gained {strict_net} net strict-correct samples."
    elif strict_net < 0:
        strict_read = f"V3 lost {-strict_net} net strict-correct samples."
    else:
        strict_read = "V3 had zero net strict-correct sample change."
    parity_mismatches = int(report["zero_attempt_parity"]["mismatch_count"])
    parity_read = (
        "Zero-attempt generation parity held exactly."
        if parity_mismatches == 0
        else (
            f"WARNING: {parity_mismatches} zero-attempt samples diverged; "
            "inspect parity before attributing changes to memory."
        )
    )

    lines = [
        "# MemGen V3 evaluation analysis",
        "",
        f"- Samples: {overall['sample_count']}",
        f"- Integrity passed: `{str(integrity['passed']).lower()}`",
        f"- Interpretation: `{report['run']['evaluation_interpretation']}`",
        "",
        "## Paired task results",
        "",
        "| Metric | Vanilla | V3 | Delta | Improved | Harmed | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Strict | {percent(strict['vanilla_accuracy'])} | "
            f"{percent(strict['v3_accuracy'])} | "
            f"{percent(strict['delta_v3_minus_vanilla'])} | "
            f"{strict['paired_table']['v3_only_correct_improved']} | "
            f"{strict['paired_table']['vanilla_only_correct_harmed']} | "
            f"{strict['mcnemar_exact_two_sided_p']} |"
        ),
        (
            f"| Format | {percent(formatting['vanilla_accuracy'])} | "
            f"{percent(formatting['v3_accuracy'])} | "
            f"{percent(formatting['delta_v3_minus_vanilla'])} | "
            f"{formatting['paired_table']['v3_only_correct_improved']} | "
            f"{formatting['paired_table']['vanilla_only_correct_harmed']} | "
            f"{formatting['mcnemar_exact_two_sided_p']} |"
        ),
        "",
        f"- Strict paired-bootstrap 95% CI: "
        f"{confidence_interval(strict)}",
        f"- Format paired-bootstrap 95% CI: "
        f"{confidence_interval(formatting)}",
        f"- Triggered strict net effect: "
        f"{triggered['strict']['net_correct_count_delta']} / "
        f"{triggered['sample_count']} samples",
        f"- No-trigger strict net effect: "
        f"{no_trigger['strict']['net_correct_count_delta']} / "
        f"{no_trigger['sample_count']} samples",
        "",
        "## Online mechanism",
        "",
        f"- Questions with attempts: {mechanism['questions_with_attempt']} "
        f"({percent(mechanism['question_trigger_rate'])})",
        f"- Retrieval attempts: {mechanism['retrieval_attempt_count']}",
        f"- Activations / replacements / duplicates / abstains: "
        f"{mechanism['activation_count']} / {mechanism['replacement_count']} / "
        f"{mechanism['duplicate_count']} / {mechanism['abstain_count']}",
        f"- Terminal abstains / clear-on-abstain: "
        f"{mechanism['terminal_abstain_count']} / "
        f"{mechanism['clear_on_terminal_abstain_count']}",
        f"- Static-selector unavailable: "
        f"{mechanism['static_selector_unavailable_count']}",
        f"- Attempts with a subsequent first answer marker / within 32 tokens: "
        f"{mechanism['attempts_with_subsequent_answer_marker_count']} / "
        f"{mechanism['late_attempt_within_32_tokens_count']}",
        f"- Attempt→first-marker distance (descriptive): "
        f"`{json.dumps(mechanism['attempt_to_first_answer_marker_distance_tokens'], sort_keys=True)}`",
        f"- Marker missing / marker not after attempt counts: "
        f"{mechanism['marker_missing_attempt_count']} / "
        f"{mechanism['marker_not_after_attempt_count']}",
        f"- Re-arms: {mechanism['rearm_count']}",
        f"- Memory attention steps: {mechanism['memory_attention_step_count']}",
        f"- Activation first-step top-1 changes: "
        f"{mechanism['activation_first_step_top1_change_count']} across "
        f"{mechanism['questions_with_activation_top1_change']} questions",
        f"- Attempts/question distribution: "
        f"`{json.dumps(mechanism['attempt_count_distribution'], sort_keys=True)}`",
        "",
        "## Zero-attempt parity",
        "",
        f"- Zero-attempt samples: {report['zero_attempt_parity']['sample_count']}",
        f"- Exact completion matches: "
        f"{report['zero_attempt_parity']['completion_exact_match_count']}",
        f"- Mismatches: {report['zero_attempt_parity']['mismatch_count']}",
        f"- Static-unavailable exact-parity mismatches: "
        f"{report['static_selector_unavailable_parity']['mismatch_count']}",
        "",
        "## V3.5 safety violations",
        "",
        f"- Counts: `{json.dumps(report['safety']['violation_counts'], sort_keys=True)}`",
        "",
        "## Token deltas",
        "",
        f"- Mean: {overall['tokens']['paired_delta_v3_minus_vanilla']['mean']}",
        f"- Median: {overall['tokens']['paired_delta_v3_minus_vanilla']['median']}",
        f"- P95 / P99 / max: "
        f"{overall['tokens']['paired_delta_v3_minus_vanilla']['p95']} / "
        f"{overall['tokens']['paired_delta_v3_minus_vanilla']['p99']} / "
        f"{overall['tokens']['paired_delta_v3_minus_vanilla']['max']}",
        "",
        "## Outcome strata",
        "",
        "### Retrieval attempts per question",
        "",
        scope_table(report["stratified_analysis"]["by_attempt_count"]),
        "",
        "### Replacement",
        "",
        scope_table(report["stratified_analysis"]["by_has_replacement"]),
        "",
        "### Duplicate retrieval",
        "",
        scope_table(report["stratified_analysis"]["by_has_duplicate"]),
        "",
        "### Margin abstention",
        "",
        scope_table(report["stratified_analysis"]["by_has_abstain"]),
        "",
        "### Activation first-step top-1 change",
        "",
        scope_table(report["stratified_analysis"]["by_activation_top1_change"]),
        "",
        "### Outcome sequence",
        "",
        scope_table(report["stratified_analysis"]["by_outcome_sequence"]),
        "",
        "### Terminal clear lifecycle",
        "",
        scope_table(report["stratified_analysis"]["by_terminal_clear"]),
        "",
        "### Attempt proximity to first answer marker (descriptive only)",
        "",
        scope_table(
            report["stratified_analysis"][
                "by_attempt_to_first_answer_marker_proximity"
            ]
        ),
        "",
        "## Exploratory online-signal associations",
        "",
        signal_table((
            "first_top1_score",
            "first_top1_top2_margin",
            "first_trigger_entropy",
            "first_trigger_boundary_index",
            "mean_activation_kl",
            "mean_attention_mass",
            "attention_step_count",
            "attempt_count",
        )),
        "",
        "## First-memory diagnostics",
        "",
        "### Most frequent",
        "",
        memory_table(report["memory_analysis"]["first_memory"]["top_by_frequency"]),
        "",
        "### Worst exploratory strict net effects (minimum sample filter applied)",
        "",
        memory_table(report["memory_analysis"]["first_memory"]["worst_strict_net_effect"]),
        "",
        "## Token outliers",
        "",
        "### Largest positive deltas",
        "",
        outlier_table(report["token_tail_analysis"]["largest_positive_deltas"]),
        "",
        "### Largest negative deltas",
        "",
        outlier_table(report["token_tail_analysis"]["largest_negative_deltas"]),
        "",
        "## Quick read",
        "",
        f"- {strict_read}",
        f"- {parity_read}",
        f"- Numeric-correct/format-invalid is descriptive only: "
        f"{report['descriptive_diagnostics']['numeric_correct_but_format_invalid']}",
        f"- Attempt→first-answer-marker timing is descriptive only: "
        f"{report['descriptive_diagnostics']['attempt_to_first_answer_marker']}",
        "- Use the triggered/replacement/memory strata in the JSON to localize the effect; these are descriptive, not causal.",
        "",
        "## Integrity failures",
        "",
        f"- Counts: `{json.dumps(integrity['failure_counts'], sort_keys=True)}`",
        f"- First examples: `{json.dumps(integrity['failure_examples'][:10], ensure_ascii=False)}`",
        "",
        "## Interpretation guardrails",
        "",
        "- The official test was reused; this is descriptive evaluation, not an independent confirmation.",
        "- Memory, quartile, and correlation analyses are exploratory and are not multiple-testing corrected.",
        "- Associations between attention/KL/retrieval scores and outcome do not establish causality.",
        "",
        "See the JSON report for complete strata, memory rankings, correlations, integrity failures, and outlier IDs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if (
        args.bootstrap_resamples < 0
        or args.top_k <= 0
        or args.min_memory_samples <= 0
    ):
        raise ValueError("Invalid analysis limits")
    profile = load_run_profile(args.run_profile)
    is_v35 = (
        profile.get("schema_version") == V35_EVALUATION_PROFILE_SCHEMA
        or profile.get("system_version") == "v3.5"
        or profile.get("system_profile", {}).get("schema_version")
        == V35_SYSTEM_PROFILE_SCHEMA
    )
    expected_profile_schema = (
        V35_EVALUATION_PROFILE_SCHEMA if is_v35 else EVALUATION_PROFILE_SCHEMA
    )
    if profile.get("schema_version") != expected_profile_schema:
        raise ValueError("Evaluation/profile version schema combination drifted")
    expected_profile_sha256 = str(profile["profile_sha256"])
    expected_risk_role = str(
        profile.get("system_profile", {}).get(
            "risk_role", "diagnostic_only"
        )
    )
    system_profile_schema = profile.get("system_profile", {}).get(
        "schema_version"
    )
    expected_generation_schema = {
        V3_SYSTEM_PROFILE_SCHEMA: V3_GENERATION_RESULT_SCHEMA,
        V34_SYSTEM_PROFILE_SCHEMA: V34_GENERATION_RESULT_SCHEMA,
        V35_SYSTEM_PROFILE_SCHEMA: V35_GENERATION_RESULT_SCHEMA,
    }.get(system_profile_schema)
    if (
        expected_generation_schema is None
        and system_profile_schema is None
        and profile.get("system_version") is None
    ):
        # Historical analyzer fixtures predate explicit V3 system schemas.
        expected_generation_schema = V3_GENERATION_RESULT_SCHEMA
    if expected_generation_schema is None:
        raise ValueError("Unknown V3 system profile schema")
    expected_count = int(profile.get("selected_sample_count", -1))
    audit = IntegrityAudit()
    safety = V35SafetyAudit()
    streaming = StreamingDiagnostics()
    samples: list[CompactSample] = []
    seen_ids: set[str] = set()
    with args.results.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {error}"
                ) from error
            sample = extract_sample(
                row,
                expected_profile_sha256=expected_profile_sha256,
                expected_risk_role=expected_risk_role,
                audit=audit,
                streaming=streaming,
                validate_hash=not args.skip_row_hash_validation,
                is_v35=is_v35,
                safety=safety,
                expected_generation_schema=expected_generation_schema,
            )
            audit.check(
                sample.sample_id not in seen_ids,
                "sample_id_unique",
                sample.sample_id,
            )
            seen_ids.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise ValueError("V3 results JSONL is empty")
    audit.check(
        len(samples) == expected_count,
        "completed_count_matches_profile",
        "<run>",
    )

    triggered = [sample for sample in samples if sample.triggered]
    no_trigger = [sample for sample in samples if not sample.triggered]
    mismatched_no_trigger = [
        sample for sample in no_trigger if not sample.completion_exact_match
    ]
    static_unavailable_samples = [
        sample for sample in samples if sample.static_selector_unavailable
    ]
    mismatched_static_unavailable = [
        sample
        for sample in static_unavailable_samples
        if not sample.completion_exact_match
    ]
    mechanism = {
        "questions_with_attempt": len(triggered),
        "question_trigger_rate": len(triggered) / len(samples),
        "retrieval_attempt_count": sum(item.attempt_count for item in samples),
        "attempts_per_all_question": sum(item.attempt_count for item in samples)
        / len(samples),
        "attempts_per_triggered_question": (
            sum(item.attempt_count for item in triggered) / len(triggered)
            if triggered
            else None
        ),
        "activation_count": sum(item.activation_count for item in samples),
        "replacement_count": sum(item.replacement_count for item in samples),
        "duplicate_count": sum(item.duplicate_count for item in samples),
        "abstain_count": sum(item.abstain_count for item in samples),
        "terminal_abstain_count": sum(
            item.terminal_abstain_count for item in samples
        ),
        "clear_on_terminal_abstain_count": sum(
            item.clear_on_terminal_abstain_count for item in samples
        ),
        "rearm_count": sum(item.rearm_count for item in samples),
        "memory_attention_step_count": sum(
            item.attention_step_count for item in samples
        ),
        "activation_first_step_top1_change_count": sum(
            item.activation_top1_change_count for item in samples
        ),
        "questions_with_activation_top1_change": sum(
            item.activation_top1_change_count > 0 for item in samples
        ),
        "attention_steps_per_triggered_question": (
            sum(item.attention_step_count for item in triggered) / len(triggered)
            if triggered
            else None
        ),
        "attempt_count_distribution": dict(sorted(Counter(
            item.attempt_count for item in samples
        ).items())),
        "outcome_sequence_distribution": dict(sorted(Counter(
            item.outcome_sequence for item in samples
        ).items())),
        "static_selector_unavailable_count": len(
            static_unavailable_samples
        ),
        "attempts_with_subsequent_answer_marker_count": sum(
            item.attempts_with_subsequent_answer_marker_count
            for item in samples
        ),
        "late_attempt_within_32_tokens_count": sum(
            item.late_attempt_within_32_tokens_count for item in samples
        ),
        "attempt_to_first_answer_marker_distance_tokens": numeric_summary([
            distance
            for item in samples
            for distance in item.attempt_to_first_answer_marker_distances
        ]),
        "marker_missing_attempt_count": sum(
            item.marker_missing_attempt_count for item in samples
        ),
        "marker_not_after_attempt_count": sum(
            item.marker_not_after_attempt_count for item in samples
        ),
        "active_memory_lifetime_tokens": numeric_summary([
            item.active_memory_lifetime_tokens for item in triggered
        ]),
        "sampled_native_cache_parity": {
            "checked": streaming.cache_parity_checked,
            "failed": streaming.cache_parity_failed,
        },
        "latency_seconds": {
            "query_encoding": numeric_summary([
                item.query_encoding_seconds for item in samples
            ]),
            "retrieval": numeric_summary([
                item.retrieval_seconds for item in samples
            ]),
            "attempt_total": numeric_summary([
                item.attempt_total_seconds for item in samples
            ]),
        },
    }

    associations = {}
    quartiles = {}
    for attribute in (
        "first_top1_score",
        "first_top1_top2_margin",
        "first_selected_static_score",
        "first_trigger_entropy",
        "first_trigger_boundary_index",
        "mean_activation_kl",
        "mean_attention_mass",
        "attention_step_count",
        "attempt_count",
        "active_memory_lifetime_tokens",
    ):
        associations[attribute] = association_summary(triggered, attribute)
        quartiles[attribute] = quartile_summary(triggered, attribute)

    top_positive = sorted(
        samples, key=lambda item: (-item.token_delta, item.sample_id)
    )[:args.top_k]
    top_negative = sorted(
        samples, key=lambda item: (item.token_delta, item.sample_id)
    )[:args.top_k]
    max_budget = int(
        profile.get("generation", {}).get("max_new_tokens", 1024)
    )
    memory_attention = []
    for memory_id, count in streaming.attention_by_memory_count.items():
        memory_attention.append({
            "memory_id": memory_id,
            "attention_step_count": count,
            "mean_attention_mass": (
                streaming.attention_by_memory_sum[memory_id] / count
            ),
            "selected_attempt_count": streaming.selected_memory_attempt_counts[
                memory_id
            ],
        })
    memory_attention.sort(
        key=lambda item: (-int(item["attention_step_count"]), item["memory_id"])
    )

    report = {
        "schema_version": ANALYSIS_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed",
        "run": {
            "profile_sha256": expected_profile_sha256,
            "run_profile_file_sha256": file_sha256(args.run_profile),
            "results_file_sha256": file_sha256(args.results),
            "evaluation_interpretation": profile.get(
                "evaluation_interpretation"
            ),
            "independent_final_confirmation": profile.get(
                "independent_final_confirmation"
            ),
            "logical_split": profile.get("logical_split"),
            "dataset_revision": profile.get("dataset_revision"),
            "selected_sample_count": expected_count,
            "system_version": profile.get("system_version", "v3"),
            "evaluation_profile_schema": profile.get("schema_version"),
            "system_profile": profile.get("system_profile"),
            "hysteresis_gate": profile.get("hysteresis_gate"),
            "inputs": profile.get("inputs"),
        },
        "analysis_configuration": {
            "results_path": str(args.results.resolve()),
            "run_profile_path": str(args.run_profile.resolve()),
            "row_hash_validation": not args.skip_row_hash_validation,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "top_k": args.top_k,
            "min_memory_samples": args.min_memory_samples,
        },
        "integrity": audit.to_dict(),
        "paired_analysis": {
            "overall": scope_summary(
                samples,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
            ),
            "triggered": scope_summary(triggered),
            "no_trigger": scope_summary(no_trigger),
        },
        "paired_transition_samples": {
            "strict_improved": [
                sample_brief(item)
                for item in samples
                if item.strict_delta > 0
            ],
            "strict_harmed": [
                sample_brief(item)
                for item in samples
                if item.strict_delta < 0
            ],
            "format_improved": [
                sample_brief(item)
                for item in samples
                if item.format_delta > 0
            ],
            "format_harmed": [
                sample_brief(item)
                for item in samples
                if item.format_delta < 0
            ],
        },
        "zero_attempt_parity": {
            "sample_count": len(no_trigger),
            "completion_exact_match_count": (
                len(no_trigger) - len(mismatched_no_trigger)
            ),
            "mismatch_count": len(mismatched_no_trigger),
            "mismatch_examples": [
                sample_brief(item) for item in mismatched_no_trigger[:args.top_k]
            ],
        },
        "static_selector_unavailable_parity": {
            "applicable": is_v35,
            "sample_count": len(static_unavailable_samples),
            "completion_exact_match_count": (
                len(static_unavailable_samples)
                - len(mismatched_static_unavailable)
            ),
            "mismatch_count": len(mismatched_static_unavailable),
            "mismatch_examples": [
                sample_brief(item)
                for item in mismatched_static_unavailable[:args.top_k]
            ],
        },
        "safety": {
            "applicable": is_v35,
            **safety.to_dict(),
        },
        "mechanism": mechanism,
        "stratified_analysis": {
            "by_attempt_count": grouped_scope_summary(
                samples, lambda item: str(item.attempt_count)
            ),
            "by_outcome_sequence": grouped_scope_summary(
                samples, lambda item: item.outcome_sequence
            ),
            "by_has_replacement": grouped_scope_summary(
                triggered,
                lambda item: (
                    "has_replacement"
                    if item.replacement_count
                    else "no_replacement"
                ),
            ),
            "by_has_duplicate": grouped_scope_summary(
                triggered,
                lambda item: "has_duplicate" if item.duplicate_count else "no_duplicate",
            ),
            "by_has_abstain": grouped_scope_summary(
                triggered,
                lambda item: "has_abstain" if item.abstain_count else "no_abstain",
            ),
            "by_terminal_clear": grouped_scope_summary(
                triggered,
                lambda item: (
                    "terminal_clear"
                    if item.clear_on_terminal_abstain_count
                    else "no_terminal_clear"
                ),
            ),
            "by_static_selector_availability": grouped_scope_summary(
                samples,
                lambda item: (
                    "static_unavailable"
                    if item.static_selector_unavailable
                    else "static_available"
                ),
            ),
            "by_static_shortlist_size": grouped_scope_summary(
                samples, lambda item: str(item.static_shortlist_size)
            ),
            "by_activation_top1_change": grouped_scope_summary(
                triggered,
                lambda item: (
                    "top1_changed"
                    if item.activation_top1_change_count
                    else "top1_unchanged_or_not_applicable"
                ),
            ),
            "by_attempt_to_first_answer_marker_proximity": (
                grouped_scope_summary(samples, marker_proximity_group)
            ),
        },
        "exploratory_associations": {
            "warning": (
                "Associations are descriptive, non-causal, and not corrected "
                "for multiple testing."
            ),
            "correlations": associations,
            "equal_count_rank_quartiles": quartiles,
        },
        "v35_static_dynamic_lifecycle_strata": {
            "applicable": is_v35,
            "static_applicability_score_quartiles": quartiles[
                "first_selected_static_score"
            ],
            "dynamic_margin_quartiles": quartiles[
                "first_top1_top2_margin"
            ],
            "active_memory_lifetime_quartiles": quartiles[
                "active_memory_lifetime_tokens"
            ],
            "late_activation_position_quartiles": quartiles[
                "first_trigger_boundary_index"
            ],
            "terminal_clear": grouped_scope_summary(
                triggered,
                lambda item: (
                    "terminal_clear"
                    if item.clear_on_terminal_abstain_count
                    else "no_terminal_clear"
                ),
            ),
            "activation_paths": grouped_scope_summary(
                triggered, lambda item: item.outcome_sequence
            ),
            "attempt_to_first_answer_marker_proximity": (
                grouped_scope_summary(samples, marker_proximity_group)
            ),
        },
        "memory_analysis": {
            "first_memory": memory_group_analysis(
                triggered,
                attribute="first_memory_id",
                min_samples=args.min_memory_samples,
                top_k=args.top_k,
            ),
            "final_memory": memory_group_analysis(
                triggered,
                attribute="final_memory_id",
                min_samples=args.min_memory_samples,
                top_k=args.top_k,
            ),
            "selected_attempt_frequency": [
                {"memory_id": memory_id, "attempt_count": count}
                for memory_id, count in streaming.selected_memory_attempt_counts.most_common(
                    args.top_k
                )
            ],
            "attention_top_by_exposure": memory_attention[:args.top_k],
        },
        "selector_geometry": {
            "applicable": is_v35,
            "static_selector_available_count": (
                len(samples) - len(static_unavailable_samples)
            ),
            "static_selector_unavailable_count": len(
                static_unavailable_samples
            ),
            "static_selector_availability_rate": (
                (len(samples) - len(static_unavailable_samples)) / len(samples)
            ),
            "static_shortlist_size": numeric_summary([
                item.static_shortlist_size for item in samples
            ]),
            "static_top1_concentration": concentration_summary(
                streaming.static_top1_memory_counts
            ),
            "dynamic_selected_concentration": concentration_summary(
                streaming.selected_memory_attempt_counts
            ),
            "selected_memory_count": len(
                streaming.selected_memory_attempt_counts
            ),
            "mem_051": {
                "memory_id": "mem-051ae8fcf60f21781c7f145f",
                "static_shortlist_question_count": int(
                    streaming.static_shortlist_memory_counts[
                        "mem-051ae8fcf60f21781c7f145f"
                    ]
                ),
                "actual_selected_attempt_count": int(
                    streaming.selected_memory_attempt_counts[
                        "mem-051ae8fcf60f21781c7f145f"
                    ]
                ),
                "first_selected_strict_improved_sample_ids": [
                    item.sample_id
                    for item in samples
                    if item.first_memory_id
                    == "mem-051ae8fcf60f21781c7f145f"
                    and item.strict_delta > 0
                ],
                "first_selected_strict_harmed_sample_ids": [
                    item.sample_id
                    for item in samples
                    if item.first_memory_id
                    == "mem-051ae8fcf60f21781c7f145f"
                    and item.strict_delta < 0
                ],
            },
            "own_source_audit": profile.get("applicability_calibration"),
        },
        "descriptive_diagnostics": {
            "not_formal_accuracy_metrics": True,
            "numeric_correct_but_format_invalid": scope_summary(samples)[
                "descriptive_numeric_correct_but_format_invalid"
            ],
            "answer_marker_pairs": scope_summary(samples)[
                "descriptive_answer_marker_pairs"
            ],
            "attempt_to_first_answer_marker": scope_summary(samples)[
                "descriptive_attempt_to_first_answer_marker"
            ],
            "vanilla_marker_seen_v3_absent_sample_ids": [
                item.sample_id
                for item in samples
                if item.vanilla_answer_marker_seen is True
                and item.v3_answer_marker_seen is False
            ],
            "late_attempt_within_32_tokens_sample_ids": [
                item.sample_id
                for item in samples
                if item.late_attempt_within_32_tokens_count > 0
            ],
            "answer_marker_missing_after_attempt_sample_ids": [
                item.sample_id
                for item in samples
                if item.marker_missing_attempt_count > 0
            ],
            "answer_marker_not_after_attempt_sample_ids": [
                item.sample_id
                for item in samples
                if item.marker_not_after_attempt_count > 0
            ],
        },
        "token_tail_analysis": {
            "max_new_tokens": max_budget,
            "vanilla_at_budget_count": sum(
                item.vanilla_tokens == max_budget for item in samples
            ),
            "v3_at_budget_count": sum(
                item.v3_tokens == max_budget for item in samples
            ),
            "both_at_budget_count": sum(
                item.vanilla_tokens == max_budget and item.v3_tokens == max_budget
                for item in samples
            ),
            "v3_only_reached_budget_count": sum(
                item.vanilla_tokens != max_budget and item.v3_tokens == max_budget
                for item in samples
            ),
            "vanilla_only_reached_budget_count": sum(
                item.vanilla_tokens == max_budget and item.v3_tokens != max_budget
                for item in samples
            ),
            "largest_positive_deltas": [sample_brief(item) for item in top_positive],
            "largest_negative_deltas": [sample_brief(item) for item in top_negative],
            "runaway_v3_only_sample_ids": [
                item.sample_id
                for item in samples
                if item.v3_tokens == max_budget
                and item.vanilla_tokens != max_budget
            ],
            "token_outlier_sample_ids": [
                item.sample_id for item in top_positive
            ],
        },
        "interpretation_guardrails": {
            "official_test_reused": (
                profile.get("evaluation_interpretation")
                == "reused_official_test_descriptive_evaluation"
            ),
            "independent_final_confirmation": False,
            "subgroup_results_are_exploratory": True,
            "multiple_comparison_correction_applied": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(args.output, report)
    markdown_path = args.markdown_output or args.output.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        f"[v3-analysis] samples={len(samples)} "
        f"integrity={report['integrity']['passed']} "
        f"strict_delta={report['paired_analysis']['overall']['strict']['net_correct_count_delta']} "
        f"json={args.output} markdown={markdown_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
