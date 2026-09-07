"""Pure schemas and aggregation for the V4 oracle target/reference audit."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_source_state import V4SourceStateCache


V4_ORACLE_PROFILE_SCHEMA = "memgen-v4-oracle-causal-profile-v1"
V4_ORACLE_PLAN_SCHEMA = "memgen-v4-oracle-causal-plan-v1"
V4_ORACLE_CASE_SCHEMA = "memgen-v4-oracle-causal-case-v1"
V4_ORACLE_RESULT_SCHEMA = "memgen-v4-oracle-causal-result-v1"
V4_ORACLE_REPORT_SCHEMA = "memgen-v4-oracle-causal-report-v1"
V4_ORACLE_BRANCHES = ("baseline", "target", "reference")


def _logical_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return canonical_json_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"created_at", hash_field}
        }
    )


def finalize_case(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("case_sha256", None)
    value.setdefault("schema_version", V4_ORACLE_CASE_SCHEMA)
    _validate_case(value, require_hash=False)
    value["case_sha256"] = canonical_json_sha256(value)
    return value


def _validate_case(row: Mapping[str, Any], *, require_hash: bool) -> None:
    if row.get("schema_version") != V4_ORACLE_CASE_SCHEMA:
        raise ValueError("Unexpected V4 oracle case schema")
    if row.get("case_kind") not in {"failure_oracle", "success_safety"}:
        raise ValueError("Unexpected V4 oracle case kind")
    required = (
        "case_id",
        "source_event_id",
        "experience_id",
        "sample_id",
        "independent_sample_id",
        "bank_id",
        "prefix_token_ids_sha256",
        "trajectory_side",
        "curation_tier",
    )
    if any(not isinstance(row.get(field), str) or not row[field] for field in required):
        raise ValueError("V4 oracle case identity is incomplete")
    if row.get("curation_tier") not in {"primary", "conditional"}:
        raise ValueError("V4 oracle case curation tier is invalid")
    if not isinstance(row.get("is_medoid"), bool):
        raise ValueError("V4 oracle case medoid flag is invalid")
    attempt = row.get("gate_attempt_number")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 3:
        raise ValueError("V4 oracle case gate attempt is invalid")
    if (
        row.get("case_kind") == "failure_oracle"
        and row.get("trajectory_side") != "verified_failure"
    ):
        raise ValueError("V4 failure oracle must branch from the failure trajectory")
    if (
        row.get("case_kind") == "success_safety"
        and row.get("trajectory_side") != "verified_success_actual_gate"
    ):
        raise ValueError("V4 success safety must use an actual success gate")
    reasoning_rank = row.get("reasoning_rank")
    token_position = row.get("token_position")
    prefix_count = row.get("prefix_token_count")
    prompt_count = row.get("prompt_token_count")
    if (
        isinstance(reasoning_rank, bool)
        or not isinstance(reasoning_rank, int)
        or reasoning_rank < 0
        or isinstance(token_position, bool)
        or not isinstance(token_position, int)
        or token_position < 0
        or isinstance(prefix_count, bool)
        or not isinstance(prefix_count, int)
        or prefix_count != token_position + 1
        or isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or not 0 < prompt_count < prefix_count
    ):
        raise ValueError("V4 oracle case prefix/token alignment is invalid")
    if (
        row.get("branch_roles") != list(V4_ORACLE_BRANCHES)
        or row.get("target_memory_id") != row["bank_id"]
        or row.get("reference_memory_id") != f"{row['bank_id']}::reference"
        or row.get("reference_online_injectable") is not False
    ):
        raise ValueError("V4 oracle case branch-role contract drifted")
    if require_hash:
        logical = {key: value for key, value in row.items() if key != "case_sha256"}
        if row.get("case_sha256") != canonical_json_sha256(logical):
            raise ValueError("V4 oracle case hash mismatch")


def validate_oracle_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != V4_ORACLE_PLAN_SCHEMA:
        raise ValueError("Unexpected V4 oracle plan schema")
    if (
        plan.get("offline_only") is not True
        or plan.get("qualified_for_online_use") is not False
        or plan.get("held_out_generalization_claim") is not False
        or plan.get("audit_interpretation")
        != "optimistic_source_positive_control_and_mechanism_qualification"
    ):
        raise ValueError("V4 oracle plan research-scope contract drifted")
    configuration = plan.get("configuration", {})
    if (
        configuration.get("layer_number") != 24
        or configuration.get("attention_implementation") != "sdpa"
        or configuration.get("dtype") != "bfloat16"
        or configuration.get("all_kv_groups") is not True
        or configuration.get("canonical_pre_rope") is not True
        or configuration.get("relative_phase_delta") != 0
        or configuration.get("decoding") != "greedy"
        or configuration.get("maximum_continuation_tokens") != 32
        or configuration.get("memory_lifecycle") != "v4_nonpersistent_bounded_episode"
    ):
        raise ValueError("V4 oracle plan frozen configuration drifted")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("V4 oracle plan contains no cases")
    case_ids: set[str] = set()
    for row in cases:
        _validate_case(row, require_hash=True)
        case_id = str(row["case_id"])
        if case_id in case_ids:
            raise ValueError("V4 oracle case IDs are duplicated")
        case_ids.add(case_id)
    expected_kind_counts = {
        kind: sum(row["case_kind"] == kind for row in cases)
        for kind in ("failure_oracle", "success_safety")
    }
    expected_support = {
        kind: len(
            {
                str(row["independent_sample_id"])
                for row in cases
                if row["case_kind"] == kind
            }
        )
        for kind in ("failure_oracle", "success_safety")
    }
    if (
        plan.get("case_count") != len(cases)
        or plan.get("case_count_by_kind") != expected_kind_counts
        or plan.get("independent_sample_count_by_kind") != expected_support
    ):
        raise ValueError("V4 oracle plan case counts are inconsistent")
    bank_ids = plan.get("bank_ids")
    if (
        not isinstance(bank_ids, list)
        or not bank_ids
        or bank_ids != sorted(bank_ids)
        or len(set(bank_ids)) != len(bank_ids)
        or any(row["bank_id"] not in bank_ids for row in cases)
    ):
        raise ValueError("V4 oracle plan bank namespace is invalid")
    if plan.get("case_order_sha256") != canonical_json_sha256(
        [str(row["case_id"]) for row in cases]
    ):
        raise ValueError("V4 oracle case order hash mismatch")
    unreachable = plan.get("gate_unreachable_failures")
    if not isinstance(unreachable, list):
        raise ValueError("V4 oracle plan lacks unreachable-failure accounting")
    unreachable_samples = [
        str(row.get("independent_sample_id", "")) for row in unreachable
    ]
    if (
        plan.get("gate_unreachable_failure_count") != len(unreachable)
        or any(not value for value in unreachable_samples)
        or len(set(unreachable_samples)) != len(unreachable_samples)
        or any(
            row.get("counted_as_memory_ineffective") is not False
            or any(
                not isinstance(row.get(field), str) or not row[field]
                for field in ("experience_id", "sample_id", "bank_id", "reason")
            )
            for row in unreachable
        )
        or any(row["bank_id"] not in bank_ids for row in unreachable)
    ):
        raise ValueError("V4 oracle unreachable failures are invalid")
    reachable_failure_samples = {
        str(row["independent_sample_id"])
        for row in cases
        if row["case_kind"] == "failure_oracle"
    }
    if reachable_failure_samples.intersection(unreachable_samples):
        raise ValueError("V4 oracle reachable/unreachable failure sets overlap")
    if plan.get("plan_sha256") != _logical_hash(plan, "plan_sha256"):
        raise ValueError("V4 oracle plan hash mismatch")


def build_oracle_plan(
    cache: V4SourceStateCache,
    *,
    attempt_policy: str = "all",
    limit_per_kind: int = 0,
) -> dict[str, Any]:
    """Build branch cases only from gate events authenticated by the cache."""

    if attempt_policy not in {"all", "first"}:
        raise ValueError("V4 oracle attempt policy must be all or first")
    if isinstance(limit_per_kind, bool) or limit_per_kind < 0:
        raise ValueError("V4 oracle case limit must be non-negative")
    prompts = {
        str(event["sample_id"]): event
        for event in cache.events
        if event["event_kind"] == "prompt_semantic"
    }
    selected_events: list[Mapping[str, Any]] = []
    for event_kind in ("failure_gate_attempt", "success_gate_attempt"):
        candidates = [
            event for event in cache.events if event["event_kind"] == event_kind
        ]
        if attempt_policy == "first":
            candidates = [
                event for event in candidates if int(event["attempt_number"]) == 1
            ]
        if limit_per_kind:
            candidates = candidates[:limit_per_kind]
        selected_events.extend(candidates)
    cases: list[dict[str, Any]] = []
    for event in selected_events:
        failure = event["event_kind"] == "failure_gate_attempt"
        case_kind = "failure_oracle" if failure else "success_safety"
        prefix_hash = str(event["prefix_alignment"]["prefix_token_ids_sha256"])
        case_id = (
            f"{case_kind}::{event['experience_id']}::attempt-{event['attempt_number']}"
        )
        cases.append(
            finalize_case(
                {
                    "case_id": case_id,
                    "case_kind": case_kind,
                    "source_event_id": str(event["event_id"]),
                    "experience_id": str(event["experience_id"]),
                    "sample_id": str(event["sample_id"]),
                    "independent_sample_id": str(event["independent_sample_id"]),
                    "bank_id": str(event["bank_id"]),
                    "is_medoid": bool(event["is_medoid"]),
                    "curation_tier": str(event["curation_tier"]),
                    "gate_attempt_number": int(event["attempt_number"]),
                    "reasoning_rank": int(event["reasoning_rank"]),
                    "token_position": int(event["token_position"]),
                    "prompt_token_count": int(
                        prompts[str(event["sample_id"])]["prompt_token_count"]
                    ),
                    "prefix_token_count": int(
                        event["prefix_alignment"]["prefix_token_count"]
                    ),
                    "prefix_token_ids_sha256": prefix_hash,
                    "trajectory_side": (
                        "verified_failure" if failure else "verified_success_actual_gate"
                    ),
                    "branch_roles": list(V4_ORACLE_BRANCHES),
                    "target_memory_id": str(event["bank_id"]),
                    "reference_memory_id": f"{event['bank_id']}::reference",
                    "reference_online_injectable": False,
                }
            )
        )
    reachable_failure_samples = {
        str(event["independent_sample_id"])
        for event in cache.events
        if event["event_kind"] == "failure_gate_attempt"
    }
    unreachable = [
        {
            "experience_id": str(event["experience_id"]),
            "sample_id": str(event["sample_id"]),
            "independent_sample_id": str(event["independent_sample_id"]),
            "bank_id": str(event["bank_id"]),
            "is_medoid": bool(event["is_medoid"]),
            "curation_tier": str(event["curation_tier"]),
            "reason": str(
                event.get("failure_gate_rejection_reason")
                or "failure_has_no_joint_gate"
            ),
            "counted_as_memory_ineffective": False,
        }
        for _sample_id, event in sorted(prompts.items())
        if str(event["independent_sample_id"]) not in reachable_failure_samples
    ]
    bank_ids = sorted(
        str(value)
        for value in (
            cache.manifest.get("bank_ids")
            or {str(event["bank_id"]) for event in prompts.values()}
        )
    )
    plan: dict[str, Any] = {
        "schema_version": V4_ORACLE_PLAN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "oracle_branch_plan_built",
        "offline_only": True,
        "qualified_for_online_use": False,
        "held_out_generalization_claim": False,
        "audit_interpretation": "optimistic_source_positive_control_and_mechanism_qualification",
        "cache_manifest_logical_sha256": cache.manifest["manifest_sha256"],
        "cache_gate_reachability_logical_sha256": cache.reachability_report[
            "report_sha256"
        ],
        "repository_revision": cache.manifest["repository_revision"],
        "bank_ids": bank_ids,
        "provenance": {
            "construction_profile_sha256": cache.manifest["provenance"][
                "construction_profile_sha256"
            ],
            "bank_manifest_logical_sha256": cache.manifest["provenance"][
                "bank_manifest_logical_sha256"
            ],
            "side_kv_manifest_logical_sha256": cache.manifest["provenance"][
                "side_kv_manifest_logical_sha256"
            ],
            "reasoner": dict(cache.manifest["reasoner"]),
        },
        "configuration": {
            "layer_number": 24,
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "all_kv_groups": True,
            "canonical_pre_rope": True,
            "relative_phase_delta": 0,
            "decoding": "greedy",
            "maximum_continuation_tokens": 32,
            "memory_lifecycle": "v4_nonpersistent_bounded_episode",
            "recovery_low_entropy_token_count": 2,
            "maximum_active_steps": 32,
            "attempt_policy": attempt_policy,
            "limit_per_kind": limit_per_kind,
        },
        "case_count": len(cases),
        "case_count_by_kind": {
            kind: sum(row["case_kind"] == kind for row in cases)
            for kind in ("failure_oracle", "success_safety")
        },
        "independent_sample_count_by_kind": {
            kind: len(
                {
                    str(row["independent_sample_id"])
                    for row in cases
                    if row["case_kind"] == kind
                }
            )
            for kind in ("failure_oracle", "success_safety")
        },
        "gate_unreachable_failure_count": len(unreachable),
        "gate_unreachable_failures": unreachable,
        "case_order_sha256": canonical_json_sha256(
            [str(row["case_id"]) for row in cases]
        ),
        "cases": cases,
    }
    plan["plan_sha256"] = _logical_hash(plan, "plan_sha256")
    validate_oracle_plan(plan)
    return plan


def finalize_result(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("record_sha256", None)
    value.setdefault("schema_version", V4_ORACLE_RESULT_SCHEMA)
    _validate_result(value, require_hash=False)
    value["record_sha256"] = canonical_json_sha256(value)
    return value


def _validate_result(row: Mapping[str, Any], *, require_hash: bool) -> None:
    if row.get("schema_version") != V4_ORACLE_RESULT_SCHEMA:
        raise ValueError("Unexpected V4 oracle result schema")
    if row.get("case_kind") not in {"failure_oracle", "success_safety"}:
        raise ValueError("Unexpected V4 oracle result case kind")
    if any(
        not isinstance(row.get(field), str) or not row[field]
        for field in (
            "profile_sha256",
            "plan_sha256",
            "case_id",
            "case_sha256",
            "source_event_id",
            "experience_id",
            "sample_id",
            "independent_sample_id",
            "bank_id",
            "trajectory_side",
            "prefix_token_ids_sha256",
        )
    ):
        raise ValueError("V4 oracle result identity is incomplete")
    prefix_count = row.get("prefix_token_count")
    prompt_count = row.get("prompt_token_count")
    if (
        isinstance(prefix_count, bool)
        or not isinstance(prefix_count, int)
        or prefix_count <= 1
        or isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or not 0 < prompt_count < prefix_count
    ):
        raise ValueError("V4 oracle result prefix length is invalid")
    branches = row.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != set(V4_ORACLE_BRANCHES):
        raise ValueError("V4 oracle result lacks the three exact-prefix branches")
    for role, branch in branches.items():
        if branch.get("role") != role:
            raise ValueError("V4 oracle result branch role drifted")
        reward = branch.get("strict_reward")
        continuation = branch.get("continuation_token_ids")
        full_completion = branch.get("full_completion_token_ids")
        if isinstance(reward, bool) or reward not in {0.0, 1.0}:
            raise ValueError("V4 oracle result branch reward is invalid")
        if (
            not isinstance(continuation, list)
            or any(isinstance(token, bool) or not isinstance(token, int) for token in continuation)
            or branch.get("continuation_token_ids_sha256")
            != canonical_json_sha256(continuation)
            or not isinstance(full_completion, list)
            or any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in full_completion
            )
            or branch.get("full_completion_token_ids_sha256")
            != canonical_json_sha256(full_completion)
            or not isinstance(branch.get("generated_continuation"), str)
            or not isinstance(branch.get("full_completion"), str)
        ):
            raise ValueError("V4 oracle result branch continuation is invalid")
        if (
            not isinstance(branch.get("task_success"), bool)
            or branch["task_success"] is not (float(reward) == 1.0)
            or not isinstance(branch.get("format_valid"), bool)
            or not isinstance(branch.get("failure_types"), list)
            or not all(isinstance(value, str) for value in branch["failure_types"])
        ):
            raise ValueError("V4 oracle result branch correctness diagnostics are invalid")
        for field in (
            "first_step_logits_kl",
            "baseline_top1_log_probability_delta",
            "branch_top1_log_probability_delta",
        ):
            metric = branch.get(field)
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
            ):
                raise ValueError("V4 oracle result branch logit diagnostics are invalid")
        for field in (
            "baseline_top1_token_id",
            "branch_top1_token_id",
            "baseline_top1_rank_under_branch",
            "branch_top1_rank_under_baseline",
            "initial_cache_length",
            "first_output_cache_length",
        ):
            metric = branch.get(field)
            minimum = 1 if "rank" in field or field == "first_output_cache_length" else 0
            if isinstance(metric, bool) or not isinstance(metric, int) or metric < minimum:
                raise ValueError("V4 oracle result branch rank/cache diagnostics are invalid")
        if any(
            not isinstance(branch.get(field), bool)
            for field in ("first_step_top1_changed", "answer_marker_seen", "eos_seen")
        ):
            raise ValueError("V4 oracle result branch boolean diagnostics are invalid")
    if (
        branches["baseline"].get("memory_id") is not None
        or branches["target"].get("memory_id") != row["bank_id"]
        or branches["reference"].get("memory_id")
        != f"{row['bank_id']}::reference"
        or branches["baseline"].get("attention_traces") != []
        or not isinstance(branches["target"].get("attention_traces"), list)
        or not branches["target"]["attention_traces"]
        or not isinstance(branches["reference"].get("attention_traces"), list)
        or not branches["reference"]["attention_traces"]
        or row.get("reference_online_injectable") is not False
    ):
        raise ValueError("V4 oracle result target/reference role integrity failed")
    parity = row.get("prefix_cache_parity")
    initial_lengths = (
        parity.get("branch_initial_cache_lengths")
        if isinstance(parity, Mapping)
        else None
    )
    first_lengths = (
        parity.get("branch_first_output_cache_lengths")
        if isinstance(parity, Mapping)
        else None
    )
    if (
        not isinstance(parity, Mapping)
        or parity.get("all_branches_share_exact_prefix") is not True
        or parity.get("all_branches_start_from_cloned_cache") is not True
        or parity.get("initial_cache_lengths_equal") is not True
        or parity.get("initial_cache_tensors_exactly_equal") is not True
        or parity.get("branch_cache_storage_is_independent") is not True
        or parity.get("prefix_token_ids_sha256")
        != row["prefix_token_ids_sha256"]
        or parity.get("prefix_token_count") != prefix_count
        or parity.get("replayed_cache_length") != prefix_count - 1
        or not isinstance(parity.get("replayed_cache_geometry_sha256"), str)
        or not parity["replayed_cache_geometry_sha256"]
        or not isinstance(initial_lengths, Mapping)
        or set(initial_lengths) != set(V4_ORACLE_BRANCHES)
        or len(set(initial_lengths.values())) != 1
        or set(initial_lengths.values()) != {prefix_count - 1}
        or not isinstance(first_lengths, Mapping)
        or set(first_lengths) != set(V4_ORACLE_BRANCHES)
        or set(first_lengths.values()) != {prefix_count}
    ):
        raise ValueError("V4 oracle result prefix/cache parity failed")
    contrasts = row.get("contrasts")
    if not isinstance(contrasts, Mapping):
        raise ValueError("V4 oracle result lacks branch contrasts")
    for field in (
        "baseline_wrong_to_target_correct",
        "baseline_wrong_to_reference_correct",
        "target_harmed_baseline",
        "target_better_than_reference",
    ):
        if not isinstance(contrasts.get(field), bool):
            raise ValueError("V4 oracle result contrast flags are invalid")
    baseline_reward = float(branches["baseline"]["strict_reward"])
    target_reward = float(branches["target"]["strict_reward"])
    reference_reward = float(branches["reference"]["strict_reward"])
    expected_contrasts = {
        "baseline_wrong_to_target_correct": baseline_reward == 0.0
        and target_reward == 1.0,
        "baseline_wrong_to_reference_correct": baseline_reward == 0.0
        and reference_reward == 1.0,
        "target_harmed_baseline": baseline_reward == 1.0 and target_reward == 0.0,
        "target_better_than_reference": target_reward > reference_reward,
    }
    expected_deltas = {
        "target_minus_baseline_reward": target_reward - baseline_reward,
        "reference_minus_baseline_reward": reference_reward - baseline_reward,
        "target_minus_reference_reward": target_reward - reference_reward,
    }
    flags_differ = any(
        contrasts[field] is not expected
        for field, expected in expected_contrasts.items()
    )
    deltas_differ = any(
        isinstance(contrasts.get(field), bool)
        or not isinstance(contrasts.get(field), (int, float))
        or not math.isclose(
            float(contrasts[field]), expected, rel_tol=0.0, abs_tol=1e-12
        )
        for field, expected in expected_deltas.items()
    )
    if flags_differ or deltas_differ:
        raise ValueError("V4 oracle result branch contrasts are inconsistent")
    if require_hash:
        logical = {key: value for key, value in row.items() if key != "record_sha256"}
        if row.get("record_sha256") != canonical_json_sha256(logical):
            raise ValueError("V4 oracle result hash mismatch")


def validate_result_against_plan(
    row: Mapping[str, Any], *, plan: Mapping[str, Any], profile_sha256: str
) -> None:
    _validate_result(row, require_hash=True)
    if (
        row.get("plan_sha256") != plan.get("plan_sha256")
        or row.get("profile_sha256") != profile_sha256
    ):
        raise ValueError("V4 oracle result profile/plan binding differs")
    cases = {str(case["case_id"]): case for case in plan["cases"]}
    case = cases.get(str(row["case_id"]))
    if case is None or row.get("case_sha256") != case.get("case_sha256"):
        raise ValueError("V4 oracle result case binding differs")
    for field in (
        "case_kind",
        "source_event_id",
        "experience_id",
        "sample_id",
        "independent_sample_id",
        "bank_id",
        "is_medoid",
        "curation_tier",
        "gate_attempt_number",
        "trajectory_side",
        "prompt_token_count",
        "prefix_token_count",
        "prefix_token_ids_sha256",
    ):
        if row.get(field) != case.get(field):
            raise ValueError("V4 oracle result case identity drifted")


def _group_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "case_count": 0,
            "independent_sample_count": 0,
            "baseline_accuracy": None,
            "target_accuracy": None,
            "reference_accuracy": None,
            "baseline_independent_sample_macro_accuracy": None,
            "target_independent_sample_macro_accuracy": None,
            "reference_independent_sample_macro_accuracy": None,
            "baseline_wrong_to_target_correct_count": 0,
            "baseline_wrong_to_target_correct_independent_sample_count": 0,
            "baseline_wrong_to_reference_correct_count": 0,
            "baseline_wrong_to_reference_correct_independent_sample_count": 0,
            "target_harm_count": 0,
            "target_harm_independent_sample_count": 0,
            "target_better_than_reference_count": 0,
            "target_better_than_reference_independent_sample_count": 0,
            "mean_target_minus_reference_reward": None,
            "independent_sample_macro_target_minus_reference_reward": None,
        }
    rewards = {
        role: [float(row["branches"][role]["strict_reward"]) for row in rows]
        for role in V4_ORACLE_BRANCHES
    }
    target_minus_reference = [
        target - reference
        for target, reference in zip(rewards["target"], rewards["reference"])
    ]
    rows_by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row["independent_sample_id"]), []).append(row)

    def macro_accuracy(role: str) -> float:
        return sum(
            sum(float(row["branches"][role]["strict_reward"]) for row in sample_rows)
            / len(sample_rows)
            for sample_rows in rows_by_sample.values()
        ) / len(rows_by_sample)

    macro_target_reference_delta = sum(
        sum(
            float(row["branches"]["target"]["strict_reward"])
            - float(row["branches"]["reference"]["strict_reward"])
            for row in sample_rows
        )
        / len(sample_rows)
        for sample_rows in rows_by_sample.values()
    ) / len(rows_by_sample)
    return {
        "case_count": len(rows),
        "independent_sample_count": len(
            {str(row["independent_sample_id"]) for row in rows}
        ),
        "baseline_accuracy": sum(rewards["baseline"]) / len(rows),
        "target_accuracy": sum(rewards["target"]) / len(rows),
        "reference_accuracy": sum(rewards["reference"]) / len(rows),
        "baseline_independent_sample_macro_accuracy": macro_accuracy("baseline"),
        "target_independent_sample_macro_accuracy": macro_accuracy("target"),
        "reference_independent_sample_macro_accuracy": macro_accuracy("reference"),
        "baseline_wrong_to_target_correct_count": sum(
            bool(row["contrasts"]["baseline_wrong_to_target_correct"]) for row in rows
        ),
        "baseline_wrong_to_target_correct_independent_sample_count": len({
            str(row["independent_sample_id"])
            for row in rows
            if row["contrasts"]["baseline_wrong_to_target_correct"]
        }),
        "baseline_wrong_to_reference_correct_count": sum(
            bool(row["contrasts"]["baseline_wrong_to_reference_correct"]) for row in rows
        ),
        "baseline_wrong_to_reference_correct_independent_sample_count": len({
            str(row["independent_sample_id"])
            for row in rows
            if row["contrasts"]["baseline_wrong_to_reference_correct"]
        }),
        "target_harm_count": sum(
            bool(row["contrasts"]["target_harmed_baseline"]) for row in rows
        ),
        "target_harm_independent_sample_count": len({
            str(row["independent_sample_id"])
            for row in rows
            if row["contrasts"]["target_harmed_baseline"]
        }),
        "target_better_than_reference_count": sum(
            bool(row["contrasts"]["target_better_than_reference"]) for row in rows
        ),
        "target_better_than_reference_independent_sample_count": len({
            str(row["independent_sample_id"])
            for row in rows
            if row["contrasts"]["target_better_than_reference"]
        }),
        "mean_target_minus_reference_reward": sum(target_minus_reference) / len(rows),
        "independent_sample_macro_target_minus_reference_reward": (
            macro_target_reference_delta
        ),
    }


def aggregate_oracle_results(
    *,
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    profile_sha256: str,
) -> dict[str, Any]:
    """Aggregate all required slices while keeping unreachable failures separate."""

    validate_oracle_plan(plan)
    seen: set[str] = set()
    for row in rows:
        validate_result_against_plan(row, plan=plan, profile_sha256=profile_sha256)
        case_id = str(row["case_id"])
        if case_id in seen:
            raise ValueError("V4 oracle result cases are duplicated")
        seen.add(case_id)
    expected_ids = {str(case["case_id"]) for case in plan["cases"]}
    complete = seen == expected_ids
    groups: dict[str, dict[str, Any]] = {}
    dimensions = {
        "case_kind": ("failure_oracle", "success_safety"),
        "curation_tier": ("primary", "conditional"),
        "medoid_role": ("medoid", "non_medoid"),
        "gate_attempt_number": (1, 2, 3),
    }
    for dimension, values in dimensions.items():
        groups[dimension] = {}
        for value in values:
            if dimension == "medoid_role":
                subset = [row for row in rows if bool(row["is_medoid"]) == (value == "medoid")]
            else:
                subset = [row for row in rows if row[dimension] == value]
            groups[dimension][str(value)] = _group_metrics(subset)
    per_bank: dict[str, Any] = {}
    for bank_id in plan["bank_ids"]:
        bank_rows = [row for row in rows if row["bank_id"] == bank_id]
        unreachable = [
            row
            for row in plan["gate_unreachable_failures"]
            if row["bank_id"] == bank_id
        ]
        per_bank[bank_id] = {
            **_group_metrics(bank_rows),
            "gate_unreachable_failure_count": len(unreachable),
            "gate_unreachable_independent_sample_count": len(
                {str(row["independent_sample_id"]) for row in unreachable}
            ),
            "gate_unreachable_counted_as_memory_ineffective": False,
            "failure_oracle": _group_metrics(
                [row for row in bank_rows if row["case_kind"] == "failure_oracle"]
            ),
            "success_safety": _group_metrics(
                [row for row in bank_rows if row["case_kind"] == "success_safety"]
            ),
        }
    report: dict[str, Any] = {
        "schema_version": V4_ORACLE_REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "completed_mechanism_diagnostic"
            if complete
            else "incomplete_mechanism_diagnostic"
        ),
        "offline_only": True,
        "qualified_for_online_use": False,
        "online_artifacts_generated": False,
        "held_out_generalization_claim": False,
        "audit_interpretation": "optimistic_source_positive_control_and_mechanism_qualification",
        "causal_qualification_decision": "pending_research_interpretation",
        "profile_sha256": profile_sha256,
        "plan_sha256": plan["plan_sha256"],
        "expected_case_count": len(expected_ids),
        "completed_case_count": len(rows),
        "complete": complete,
        "gate_unreachable_failure_count": plan["gate_unreachable_failure_count"],
        "gate_unreachable_counted_as_memory_ineffective": False,
        "overall": _group_metrics(rows),
        "by_dimension": groups,
        "per_bank": per_bank,
    }
    report["report_sha256"] = _logical_hash(report, "report_sha256")
    return report


__all__ = [
    "V4_ORACLE_BRANCHES",
    "V4_ORACLE_CASE_SCHEMA",
    "V4_ORACLE_PLAN_SCHEMA",
    "V4_ORACLE_PROFILE_SCHEMA",
    "V4_ORACLE_REPORT_SCHEMA",
    "V4_ORACLE_RESULT_SCHEMA",
    "aggregate_oracle_results",
    "build_oracle_plan",
    "finalize_case",
    "finalize_result",
    "validate_oracle_plan",
    "validate_result_against_plan",
]
