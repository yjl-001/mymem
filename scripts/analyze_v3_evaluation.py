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

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v3 import (
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_METHODS,
    V3_QUERY_POOLING_PRE_BOUNDARY,
    query_embedding_token_index,
)


EVALUATION_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
EVALUATION_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"
GENERATION_RESULT_SCHEMA = "experience-memory-v3-generation-result-v1"
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
    if value.get("schema_version") != EVALUATION_PROFILE_SCHEMA:
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


def extract_sample(
    row: Mapping[str, Any],
    *,
    expected_profile_sha256: str,
    audit: IntegrityAudit,
    streaming: StreamingDiagnostics,
    validate_hash: bool,
) -> CompactSample:
    sample_id = str(row.get("sample_id", ""))
    audit.check(bool(sample_id), "sample_id_present", sample_id or "<missing>")
    audit.check(
        row.get("schema_version") == EVALUATION_ROW_SCHEMA,
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
    audit.check(
        runtime.get("schema_version") == GENERATION_RESULT_SCHEMA,
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
    audit.check(attempt_count <= 3, "attempt_budget_at_most_three", sample_id)
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
    expected_active_memory_id: str | None = None
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
            == "pure_prefix_reencode_side_kv_disabled",
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
                audit.check(
                    int(query.get("encoded_full_prefix_token_count", -1))
                    == query_count
                    and int(query.get("query_embedding_token_index", -1))
                    == expected_embedding_index
                    and int(query.get("query_embedding_causal_context_token_count", -1))
                    == expected_embedding_index + 1
                    and int(query.get("trigger_boundary_token_index", -1))
                    == query_count - 1
                    and int(query.get("trigger_boundary_token_id", -1))
                    == boundary_token_id
                    and bool(
                        query.get("trigger_boundary_excluded_from_pooling")
                    )
                    is (query_pooling == V3_QUERY_POOLING_PRE_BOUNDARY),
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
            query.get("method") == "exact_cosine",
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
        expected_outcome_shape = {
            "abstained": (
                selected_id is None and active_after_id == previous_id
            ),
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
        expected_active_memory_id = active_after_id
        if position == 0:
            first_top1_score = _finite_or_none(query.get("top1_score"))
            first_margin = _finite_or_none(query.get("top1_top2_margin"))
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
            boundary.get("risk_role") == "diagnostic_only",
            "risk_role_is_diagnostic_only",
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
    expected_profile_sha256 = str(profile["profile_sha256"])
    expected_count = int(profile.get("selected_sample_count", -1))
    audit = IntegrityAudit()
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
                audit=audit,
                streaming=streaming,
                validate_hash=not args.skip_row_hash_validation,
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
        "first_trigger_entropy",
        "first_trigger_boundary_index",
        "mean_activation_kl",
        "mean_attention_mass",
        "attention_step_count",
        "attempt_count",
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
            "evaluation_interpretation": profile.get(
                "evaluation_interpretation"
            ),
            "independent_final_confirmation": profile.get(
                "independent_final_confirmation"
            ),
            "logical_split": profile.get("logical_split"),
            "dataset_revision": profile.get("dataset_revision"),
            "selected_sample_count": expected_count,
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
            "by_activation_top1_change": grouped_scope_summary(
                triggered,
                lambda item: (
                    "top1_changed"
                    if item.activation_top1_change_count
                    else "top1_unchanged_or_not_applicable"
                ),
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
