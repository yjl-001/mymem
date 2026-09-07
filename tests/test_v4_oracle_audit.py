from __future__ import annotations

from pathlib import Path
import unittest

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_oracle_audit import (
    aggregate_oracle_results,
    build_oracle_plan,
    finalize_result,
    validate_oracle_plan,
    validate_result_against_plan,
)
from memgen.experience.v4_source_state import (
    FAILURE_TENSORS,
    PROMPT_TENSORS,
    SUCCESS_GATE_TENSORS,
    V4SourceStateCache,
    build_gate_reachability_report,
    finalize_event,
)


ROOT = Path(__file__).resolve().parents[1]


def common(sample_id: str, *, medoid: bool) -> dict:
    return {
        "experience_id": f"experience-{sample_id}",
        "sample_id": sample_id,
        "independent_sample_id": f"independent-{sample_id}",
        "bank_id": "bank-a",
        "benchmark": "openai/gsm8k",
        "logical_split": "bank-source",
        "dataset_split": "train",
        "source_index": 0,
        "question_sha256": f"question-{sample_id}",
        "is_medoid": medoid,
        "curation_tier": "primary",
        "construction_profile_sha256": "construction-profile",
        "bank_record_sha256": "bank-record",
        "contrast_pair": {
            "target_episode_id": f"target-{sample_id}",
            "reference_episode_id": f"reference-{sample_id}",
            "paired_success_failure": True,
        },
        "completion_hashes": {
            "verified_success_completion_sha256": f"success-{sample_id}",
            "verified_failure_completion_sha256": f"failure-{sample_id}",
        },
    }


def prompt(sample_id: str, row: int, *, failures: int, successes: int) -> dict:
    return finalize_event(
        {
            **common(sample_id, medoid=sample_id == "s1"),
            "event_id": f"experience-{sample_id}::prompt-semantic",
            "event_kind": "prompt_semantic",
            "online_reachable_safety_negative": False,
            "tensor_rows": {name: row for name in PROMPT_TENSORS},
            "prompt_token_count": 10,
            "prompt_token_ids_sha256": f"prompt-{sample_id}",
            "question_token_start": 2,
            "question_token_end_exclusive": 8,
            "question_token_count": 6,
            "failure_gate_eligible": failures > 0,
            "failure_gate_attempt_count": failures,
            "failure_gate_rejection_reason": (
                None if failures else "failure_has_no_joint_gate"
            ),
            "success_gate_eligible": successes > 0,
            "success_gate_attempt_count": successes,
            "success_gate_rejection_reason": (
                None if successes else "success_has_no_joint_gate"
            ),
        }
    )


def gate(sample_id: str, row: int, attempt: int, *, success: bool) -> dict:
    kind = "success_gate_attempt" if success else "failure_gate_attempt"
    tensors = SUCCESS_GATE_TENSORS if success else FAILURE_TENSORS
    value = {
        **common(sample_id, medoid=sample_id == "s1"),
        "event_id": f"experience-{sample_id}::{kind}::attempt-{attempt}",
        "event_kind": kind,
        "online_reachable_safety_negative": success,
        "attempt_number": attempt,
        "reasoning_rank": attempt,
        "candidate_rank": attempt + 1,
        "token_position": 10 + attempt,
        "window_token_count": attempt + 1,
        "tensor_rows": {name: row for name in tensors},
        "gate_diagnostics": {
            "gate_eligible": True,
            "gate_rejection_reason": None,
            "attention_entropy": 2.0,
            "persistence_risk": 0.2,
            "high_entropy_threshold": 1.5,
            "low_entropy_threshold": 1.0,
            "risk_threshold": 0.1,
            "state_before": "ARMED",
            "state_after": "DISARMED",
            "logit_summary": {
                "maximum_logit": 5.0,
                "top1_top2_logit_gap": 0.5,
                "logsumexp": 6.0,
                "predictive_entropy": 1.0,
            },
        },
        "prefix_alignment": {
            "prefix_token_count": 11 + attempt,
            "prefix_token_ids_sha256": f"prefix-{sample_id}-{kind}-{attempt}",
            "prefix_includes_current_token": True,
            "token_position_matches_prefix_end": True,
        },
    }
    if not success:
        value["matched_success_alignment"] = {
            "state_role": "offline_repair_direction_control",
            "online_reachable_safety_negative": False,
            "alignment_method": "normalized_reasoning_progress_endpoint_preserving",
            "reasoning_rank": attempt,
            "token_position": 20 + attempt,
            "window_token_count": attempt + 1,
            "prefix_token_ids_sha256": f"aligned-{sample_id}-{attempt}",
        }
    return finalize_event(value)


def cache_fixture() -> V4SourceStateCache:
    events = (
        prompt("s1", 0, failures=2, successes=1),
        gate("s1", 0, 1, success=False),
        gate("s1", 1, 2, success=False),
        gate("s1", 0, 1, success=True),
        prompt("s2", 1, failures=0, successes=0),
    )
    reachability = build_gate_reachability_report(
        events=events, bank_ids=("bank-a",)
    )
    manifest = {
        "manifest_sha256": "cache-manifest",
        "repository_revision": "repository-revision",
        "reasoner": {
            "model_name": "fixture-model",
            "model_revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
        },
        "provenance": {
            "construction_profile_sha256": "construction-profile",
            "bank_manifest_logical_sha256": "bank-manifest",
            "side_kv_manifest_logical_sha256": "side-kv-manifest",
        },
    }
    return V4SourceStateCache(
        manifest_path=Path("fixture-manifest.json"),
        manifest=manifest,
        events=events,
        reachability_report=reachability,
        tensors=None,
    )


def result_for_case(plan: dict, case: dict, *, rewards=(0.0, 1.0, 0.0)) -> dict:
    branches = {}
    for role, reward in zip(("baseline", "target", "reference"), rewards):
        continuation = [1, 2]
        full_completion = [7, 8, *continuation]
        branches[role] = {
            "role": role,
            "strict_reward": float(reward),
            "task_success": float(reward) == 1.0,
            "format_valid": float(reward) == 1.0,
            "diagnostic_answer_correct": float(reward) == 1.0,
            "failure_types": [] if reward else ["boxed_answer_mismatch"],
            "continuation_token_ids": continuation,
            "continuation_token_ids_sha256": canonical_json_sha256(continuation),
            "generated_continuation": "fixture continuation",
            "full_completion": "fixture completion",
            "full_completion_token_ids": full_completion,
            "full_completion_token_ids_sha256": canonical_json_sha256(
                full_completion
            ),
            "memory_id": (
                None
                if role == "baseline"
                else "bank-a" if role == "target" else "bank-a::reference"
            ),
            "attention_traces": [] if role == "baseline" else [{"mass": 0.1}],
            "lifecycle": None if role == "baseline" else {"state": "CLOSED"},
            "first_step_logits_kl": 0.0 if role == "baseline" else 0.1,
            "first_step_top1_changed": role != "baseline",
            "baseline_top1_token_id": 1,
            "branch_top1_token_id": 1 if role == "baseline" else 2,
            "baseline_top1_rank_under_branch": 1,
            "branch_top1_rank_under_baseline": 1,
            "baseline_top1_log_probability_delta": 0.0,
            "branch_top1_log_probability_delta": 0.0,
            "initial_cache_length": int(case["prefix_token_count"]) - 1,
            "first_output_cache_length": int(case["prefix_token_count"]),
            "answer_marker_seen": False,
            "eos_seen": False,
        }
    baseline, target, reference = rewards
    return finalize_result(
        {
            "profile_sha256": "profile-sha",
            "plan_sha256": plan["plan_sha256"],
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
            "case_kind": case["case_kind"],
            "source_event_id": case["source_event_id"],
            "experience_id": case["experience_id"],
            "sample_id": case["sample_id"],
            "independent_sample_id": case["independent_sample_id"],
            "bank_id": case["bank_id"],
            "is_medoid": case["is_medoid"],
            "curation_tier": case["curation_tier"],
            "gate_attempt_number": case["gate_attempt_number"],
            "trajectory_side": case["trajectory_side"],
            "prompt_token_count": case["prompt_token_count"],
            "prefix_token_count": case["prefix_token_count"],
            "prefix_token_ids_sha256": case["prefix_token_ids_sha256"],
            "branches": branches,
            "prefix_cache_parity": {
                "all_branches_share_exact_prefix": True,
                "all_branches_start_from_cloned_cache": True,
                "initial_cache_lengths_equal": True,
                "initial_cache_tensors_exactly_equal": True,
                "branch_cache_storage_is_independent": True,
                "prefix_token_count": case["prefix_token_count"],
                "prefix_token_ids_sha256": case["prefix_token_ids_sha256"],
                "replayed_cache_length": case["prefix_token_count"] - 1,
                "replayed_cache_geometry_sha256": "cache-geometry",
                "branch_initial_cache_lengths": {
                    "baseline": case["prefix_token_count"] - 1,
                    "target": case["prefix_token_count"] - 1,
                    "reference": case["prefix_token_count"] - 1,
                },
                "branch_first_output_cache_lengths": {
                    "baseline": case["prefix_token_count"],
                    "target": case["prefix_token_count"],
                    "reference": case["prefix_token_count"],
                },
            },
            "contrasts": {
                "baseline_wrong_to_target_correct": baseline == 0.0 and target == 1.0,
                "baseline_wrong_to_reference_correct": baseline == 0.0 and reference == 1.0,
                "target_harmed_baseline": baseline == 1.0 and target == 0.0,
                "target_better_than_reference": target > reference,
                "target_minus_baseline_reward": target - baseline,
                "reference_minus_baseline_reward": reference - baseline,
                "target_minus_reference_reward": target - reference,
            },
            "reference_online_injectable": False,
        }
    )


class V4OracleAuditTests(unittest.TestCase):
    def test_plan_uses_all_failure_attempts_and_actual_success_gate(self) -> None:
        plan = build_oracle_plan(cache_fixture(), attempt_policy="all")
        validate_oracle_plan(plan)
        self.assertEqual(plan["case_count"], 3)
        self.assertEqual(plan["case_count_by_kind"]["failure_oracle"], 2)
        self.assertEqual(plan["case_count_by_kind"]["success_safety"], 1)
        self.assertEqual(plan["gate_unreachable_failure_count"], 1)
        self.assertFalse(
            plan["gate_unreachable_failures"][0]["counted_as_memory_ineffective"]
        )
        success = next(
            case for case in plan["cases"] if case["case_kind"] == "success_safety"
        )
        self.assertEqual(success["trajectory_side"], "verified_success_actual_gate")
        self.assertEqual(success["branch_roles"], ["baseline", "target", "reference"])
        self.assertFalse(success["reference_online_injectable"])
        self.assertEqual(
            plan["provenance"]["construction_profile_sha256"],
            "construction-profile",
        )
        self.assertEqual(
            plan["provenance"]["bank_manifest_logical_sha256"],
            "bank-manifest",
        )
        self.assertEqual(
            plan["provenance"]["side_kv_manifest_logical_sha256"],
            "side-kv-manifest",
        )

    def test_first_attempt_policy_does_not_duplicate_sample_support(self) -> None:
        plan = build_oracle_plan(cache_fixture(), attempt_policy="first")
        self.assertEqual(plan["case_count_by_kind"]["failure_oracle"], 1)
        self.assertEqual(
            plan["independent_sample_count_by_kind"]["failure_oracle"], 1
        )
        self.assertEqual(plan["configuration"]["maximum_continuation_tokens"], 32)
        self.assertEqual(
            plan["configuration"]["memory_lifecycle"],
            "v4_nonpersistent_bounded_episode",
        )

    def test_result_requires_three_branches_and_exact_prefix_parity(self) -> None:
        plan = build_oracle_plan(cache_fixture(), attempt_policy="first")
        row = result_for_case(plan, plan["cases"][0])
        validate_result_against_plan(
            row, plan=plan, profile_sha256="profile-sha"
        )
        broken = dict(row)
        broken["branches"] = dict(broken["branches"])
        broken["branches"].pop("reference")
        broken.pop("record_sha256")
        with self.assertRaisesRegex(ValueError, "three exact-prefix branches"):
            finalize_result(broken)

        broken_parity = dict(row)
        broken_parity.pop("record_sha256")
        broken_parity["prefix_cache_parity"] = dict(
            broken_parity["prefix_cache_parity"]
        )
        broken_parity["prefix_cache_parity"][
            "initial_cache_tensors_exactly_equal"
        ] = False
        with self.assertRaisesRegex(ValueError, "prefix/cache parity"):
            finalize_result(broken_parity)

    def test_incomplete_audit_never_qualifies_online_artifacts(self) -> None:
        plan = build_oracle_plan(cache_fixture(), attempt_policy="all")
        row = result_for_case(plan, plan["cases"][0])
        report = aggregate_oracle_results(
            plan=plan, rows=[row], profile_sha256="profile-sha"
        )
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "incomplete_mechanism_diagnostic")
        self.assertFalse(report["qualified_for_online_use"])
        self.assertFalse(report["online_artifacts_generated"])
        self.assertFalse(report["gate_unreachable_counted_as_memory_ineffective"])

    def test_aggregate_reports_required_slices_and_target_reference_delta(self) -> None:
        plan = build_oracle_plan(cache_fixture(), attempt_policy="all")
        rows = [result_for_case(plan, case) for case in plan["cases"]]
        report = aggregate_oracle_results(
            plan=plan, rows=rows, profile_sha256="profile-sha"
        )
        self.assertTrue(report["complete"])
        self.assertEqual(report["overall"]["case_count"], 3)
        self.assertEqual(
            report["overall"]["mean_target_minus_reference_reward"], 1.0
        )
        self.assertIn("bank-a", report["per_bank"])
        self.assertIn("medoid_role", report["by_dimension"])
        self.assertIn("gate_attempt_number", report["by_dimension"])
        self.assertIn("curation_tier", report["by_dimension"])
        self.assertEqual(
            report["overall"][
                "target_better_than_reference_independent_sample_count"
            ],
            1,
        )
        self.assertFalse(
            report["per_bank"]["bank-a"][
                "gate_unreachable_counted_as_memory_ineffective"
            ]
        )

    def test_reference_access_remains_offline_only(self) -> None:
        source = (ROOT / "memgen/model/v4_oracle.py").read_text(encoding="utf-8")
        online_source = (ROOT / "memgen/model/v4_side_kv.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def get_reference_offline(", source)
        self.assertIn('entry.get("online_injectable") is not False', source)
        self.assertNotIn("def get_reference(", online_source)


if __name__ == "__main__":
    unittest.main()
