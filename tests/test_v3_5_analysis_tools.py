from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3 import ExperienceMemoryV3Profile
from memgen.experience.v3_5_selector import (
    V35_DYNAMIC_MARGIN_TIE_POLICY,
    V35_SELECTOR_CALIBRATION_SCHEMA,
    V35_SELECTOR_POLICY,
    load_v35_selector_calibration,
    v35_artifact_sha256,
)


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_script("analyze_v3_evaluation")
CALIBRATE = load_script("calibrate_v3_5_dynamic_selector")
COMPARE = load_script("compare_v3_5_applicability_selector")
QUALIFY_PATH = PROJECT_ROOT / "scripts" / "qualify_v3_5_dev.py"


def condition(token_ids=(1, 0)):
    values = list(token_ids)
    return {
        "completion": "fixture",
        "completion_token_ids": values,
        "completion_token_ids_sha256": canonical_json_sha256(values),
        "generated_token_count": len(values),
        "strict_correct": False,
        "format_correct": False,
        "strict_reward": 0.0,
        "numeric_correct_but_format_invalid": False,
        "answer_marker_seen": False,
        "first_answer_marker_token_index": None,
        "scorer_version": "fixture",
        "runtime_seconds": 0.1,
    }


def valid_v35_unavailable_row():
    vanilla = condition()
    v35 = condition()
    static_trace = {
        "schema_version": "experience-memory-v3.5-static-shortlist-v1",
        "query": {
            "static_question_text_sha256": "question",
            "static_question_token_count": 2,
            "static_question_token_ids_sha256": "tokens",
            "static_question_embedding_sha256": "embedding",
            "layer_number": 24,
            "pooling": "last_valid_token",
            "normalization": "l2",
            "side_kv_disabled": True,
        },
        "score_floor": 0.2,
        "score_floor_tie_policy": (
            "retain_score_greater_than_or_equal_to_floor"
        ),
        "shortlist_k": 2,
        "pre_floor_top_k": [
            {
                "memory_id": "memory-a",
                "static_score": 0.1,
                "original_global_rank": 1,
            },
            {
                "memory_id": "memory-b",
                "static_score": 0.0,
                "original_global_rank": 2,
            },
        ],
        "post_floor_shortlist": [],
        "shortlist_memory_ids": [],
        "shortlist_nonempty": False,
        "static_selector_unavailable": True,
        "unavailable_reason": "below_applicability_floor",
        "applicability_bank_manifest_sha256": "manifest",
        "shortlist_fixed_for_generation": True,
        "retrieval_method": "exact_cosine",
        "stable_tie_break": "memory_id_ascending",
    }
    summary = {
        "retrieval_attempt_count": 0,
        "rearm_count": 0,
        "replacement_count": 0,
        "duplicate_count": 0,
        "memory_attention_step_count": 0,
        "gate_observation_count": 0,
        "joint_trigger_qualified_count": 0,
        "native_gate_observation_count": 0,
        "memory_conditioned_gate_observation_count": 0,
        "abstain_count": 0,
        "terminal_abstain_count": 0,
        "clear_on_terminal_abstain_count": 0,
        "static_selector_unavailable": True,
        "max_three_attempts_respected": True,
        "two_low_rearm_respected": True,
        "second_low_rearms_without_trigger": True,
        "no_rearm_after_terminal_abstain": True,
        "stale_memory_attention_after_terminal_clear_count": 0,
        "terminal_clear_attention_safe": True,
        "initial_gate_state": "EXHAUSTED",
        "final_gate_state": "EXHAUSTED",
        "final_memory_id": None,
    }
    runtime = {
        "schema_version": "experience-memory-v3.5-generation-result-v1",
        "completion_token_ids": list(v35["completion_token_ids"]),
        "completion_token_ids_sha256": v35["completion_token_ids_sha256"],
        "generated_token_count": v35["generated_token_count"],
        "boundary_traces": [],
        "retrieval_attempts": [],
        "memory_transitions": [],
        "memory_activation_spans": [],
        "attention_traces": [],
        "final_gate_state": "EXHAUSTED",
        "final_memory_id": None,
        "answer_marker_seen": False,
        "static_selector_trace": static_trace,
        "summary": summary,
    }
    diagnostics = {
        "retrieval_attempt_count": 0,
        "rearm_count": 0,
        "activation_count": 0,
        "replacement_count": 0,
        "duplicate_count": 0,
        "abstain_count": 0,
        "memory_attention_step_count": 0,
        "gate_observation_count": 0,
        "joint_trigger_qualified_count": 0,
        "native_gate_observation_count": 0,
        "memory_conditioned_gate_observation_count": 0,
        "attempt_budget_respected": True,
        "query_context_is_full_prefix": True,
        "native_cache_excludes_memory_slots": True,
        "memory_attention_mass_finite_and_positive": True,
        "static_selector_unavailable": True,
        "static_selector_unavailable_reason": "below_applicability_floor",
        "static_shortlist_size": 0,
        "static_shortlist_ids_sha256": canonical_json_sha256([]),
        "static_shortlist_fixed_for_generation": True,
        "static_query_side_kv_disabled": True,
        "dynamic_query_side_kv_disabled": True,
        "both_query_encodings_side_kv_disabled": True,
        "dynamic_search_restricted_to_static_shortlist": True,
        "selected_memory_belongs_to_static_shortlist": True,
        "selected_outside_static_shortlist_count": 0,
        "selected_memory_kv_metadata_aligned": True,
        "selected_memory_kv_alignment_unlogged_count": 0,
        "terminal_abstain_count": 0,
        "clear_on_terminal_abstain_count": 0,
        "no_rearm_after_terminal_abstain": True,
        "two_low_rearm_respected": True,
        "second_low_rearms_without_trigger": True,
        "stale_memory_attention_after_terminal_clear_count": 0,
        "terminal_clear_attention_safe": True,
        "final_gate_state": "EXHAUSTED",
        "final_memory_id": None,
        "answer_marker_seen": False,
        "first_answer_marker_token_index": None,
        "answer_marker_attempt_distances": [],
        "attempt_affects_index_contract_respected": True,
        "attempts_with_subsequent_answer_marker_count": 0,
        "late_attempt_within_32_tokens_count": 0,
    }
    row = {
        "schema_version": "experience-memory-v3.5-evaluation-row-v1",
        "profile_sha256": "profile",
        "sample_id": "sample-unavailable",
        "paired_generated_token_delta_v3_minus_vanilla": 0,
        "conditions": {
            "vanilla": vanilla,
            "v3": v35
            | {
                "runtime_trace": runtime,
                "online_diagnostics": diagnostics,
            },
        },
    }
    return row


def selector_artifact():
    dual_hash = "1" * 64
    dual_logical_hash = "2" * 64
    applicability_hash = "3" * 64
    applicability_artifact_hash = "4" * 64
    risk_hash = "5" * 64
    value = {
        "schema_version": V35_SELECTOR_CALIBRATION_SCHEMA,
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": "passed",
        "policy": V35_SELECTOR_POLICY,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "source": {
            "logical_split": "calibration-val",
            "scope": "first_retrieval_attempt_per_triggered_question",
            "run_profile_sha256": "6" * 64,
            "run_profile_file_sha256": "7" * 64,
            "results_file_sha256": "8" * 64,
            "dual_key_manifest_sha256": dual_hash,
            "dual_key_manifest_logical_sha256": dual_logical_hash,
            "applicability_calibration_sha256": applicability_hash,
            "applicability_calibration_artifact_sha256": (
                applicability_artifact_hash
            ),
            "risk_artifact_sha256": risk_hash,
            "system_version": "v3.5",
            "system_profile_schema": "experience-memory-system-profile-v3.5",
            "calibration_trace_only": True,
            "completed_sample_count": 2,
        },
        "calibration": {
            "sample_count": 2,
            "shortlist_k": 2,
            "minimum_applicability_score": 0.2,
            "applicability_score_floor_tie_policy": (
                "retain_score_greater_than_or_equal_to_floor"
            ),
            "minimum_dynamic_top1_top2_margin": 0.2,
            "target_retained_fraction": 0.5,
            "target_retained_count": 1,
            "actual_retained_count": 1,
            "actual_retained_fraction": 0.5,
            "dynamic_margin_tie_policy": V35_DYNAMIC_MARGIN_TIE_POLICY,
            "first_attempt_count": 2,
            "margin_summary": {
                "count": 2,
                "min": 0.1,
                "p05": 0.105,
                "p25": 0.125,
                "median": 0.15,
                "mean": 0.15,
                "p75": 0.175,
                "p95": 0.195,
                "max": 0.2,
            },
            "static_selector_available_sample_count": 2,
            "insufficient_shortlist_sample_count": 0,
            "insufficient_shortlist_fraction": 0.0,
            "first_attempt_selected_memory_count": 2,
            "first_attempt_selected_memory_frequency": [
                {"memory_id": "memory-a", "count": 1},
                {"memory_id": "memory-b", "count": 1},
            ],
        },
        "requirements": {
            "source_is_calibration_val": True,
            "source_profile_is_authenticated": True,
            "source_rows_are_complete_and_authenticated": True,
            "source_is_trace_only": True,
            "first_attempt_only": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "dual_key_manifest_is_authenticated_and_bound": True,
            "applicability_calibration_is_authenticated_and_bound": True,
            "static_shortlist_is_fixed_and_bound": True,
            "dynamic_queries_are_full_prefix_and_authenticated": True,
            "dynamic_queries_disable_side_kv": True,
            "dynamic_rerank_is_inside_static_shortlist": True,
            "first_attempt_sample_count_sufficient": True,
            "insufficient_shortlist_fraction_acceptable": True,
            "threshold_is_finite_and_nonnegative": True,
            "inclusive_tie_policy": True,
        },
    }
    value["artifact_sha256"] = v35_artifact_sha256(value)
    return value


def calibration_profile():
    system = ExperienceMemoryV3Profile.applicability_aware_continuous(
        applicability_shortlist_k=2,
        applicability_score_floor=0.2,
        calibration_trace_only=True,
    ).to_dict()
    value = {
        "schema_version": "experience-memory-v3.5-evaluation-profile-v1",
        "system_version": "v3.5",
        "logical_split": "calibration-val",
        "selected_sample_count": 64,
        "selected_sample_ids_sha256": "selected-ids",
        "slice": {"offset": 0, "limit": 64},
        "calibration_trace_only": True,
        "task_results_used_for_selector_decision": False,
        "selector_decision_data_contract": {
            "task_accuracy_used": False,
            "answer_or_reward_used": False,
            "first_attempt_dynamic_margins_only": True,
        },
        "system_profile": system,
        "inputs": {"risk_artifact_sha256": "risk-artifact"},
        "logging": {
            "query_embeddings_sidecar": True,
            "query_embeddings_sidecar_required_for_calibration": True,
            "query_embedding_sidecar_representation": (
                "dynamic_query_l2_normalized_exact_audit"
            ),
            "raw_query_token_ids_saved": True,
            "calibration_query_hash_reproduction_required": True,
        },
    }
    value["profile_sha256"] = CALIBRATE.evaluation_profile_sha256(value)
    return value


class V35AnalysisToolTests(unittest.TestCase):
    def test_calibration_profile_enforces_the_complete_frozen_v35_contract(self):
        frozen_fields = {
            "query_encoder_state": "wrong",
            "query_normalization": "none",
            "retrieval_method": "approximate",
            "retrieval_embedding_transform": "center_l2",
            "retrieval_top_k": 3,
            "selected_memory_count": 2,
            "attention_backend": "eager",
            "replacement_policy": "append",
            "duplicate_policy": "ignore",
            "abstain_policy": "keep_memory",
            "injection_policy": "one_token",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            valid = calibration_profile()
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(CALIBRATE.load_profile(path)["system_version"], "v3.5")
            for field, invalid_value in frozen_fields.items():
                with self.subTest(field=field):
                    tampered = json.loads(json.dumps(valid))
                    tampered["system_profile"][field] = invalid_value
                    tampered["profile_sha256"] = (
                        CALIBRATE.evaluation_profile_sha256(tampered)
                    )
                    path.write_text(json.dumps(tampered), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "answer-blind V3.5"):
                        CALIBRATE.load_profile(path)

    def test_dynamic_threshold_is_answer_blind_inclusive_fifty_percent(self):
        calibrated = CALIBRATE.retained_margin_threshold(
            (0.9, 0.8, 0.8, 0.1), target_retained_fraction=0.5
        )
        self.assertEqual(calibrated["minimum_dynamic_top1_top2_margin"], 0.8)
        self.assertEqual(calibrated["target_retained_count"], 2)
        self.assertEqual(calibrated["actual_retained_count"], 3)
        self.assertEqual(calibrated["actual_retained_fraction"], 0.75)
        self.assertEqual(
            calibrated["dynamic_margin_tie_policy"],
            V35_DYNAMIC_MARGIN_TIE_POLICY,
        )

    def test_dynamic_calibration_recomputes_margin_and_exact_replay(self):
        row = {
            "schema_version": "experience-memory-v3.5-evaluation-row-v1",
            "profile_sha256": "profile",
            "sample_id": "sample-margin-tamper",
            "question_sha256": "question",
            "prompt_token_ids_sha256": canonical_json_sha256([10, 11, 12]),
            "conditions": {
                "v3": {
                    "runtime_trace": {
                        "schema_version": (
                            "experience-memory-v3.5-generation-result-v1"
                        ),
                        "static_selector_trace": {
                            "schema_version": "experience-memory-v3.5-static-shortlist-v1",
                            "query": {
                                "schema_version": (
                                    "experience-memory-v3.5-static-question-query-v1"
                                ),
                                "static_question_text_sha256": "question",
                                "static_question_token_ids": [1],
                                "static_question_token_count": 1,
                                "static_question_token_ids_sha256": canonical_json_sha256([1]),
                                "static_question_embedding_sha256": "static-embedding",
                                "static_question_embedding_norm": 1.0,
                                "layer_number": 24,
                                "representation": "decoder_layer_output",
                                "pooling": "last_valid_token",
                                "normalization": "l2",
                                "side_kv_disabled": True,
                                "chat_wrapper_included": False,
                                "prompt_boilerplate_included": False,
                                "add_special_tokens": False,
                            },
                            "score_floor": 0.2,
                            "score_floor_tie_policy": "retain_score_greater_than_or_equal_to_floor",
                            "shortlist_k": 2,
                            "pre_floor_top_k": [
                                {"memory_id": "memory-a", "static_score": 0.8, "original_global_rank": 1},
                                {"memory_id": "memory-b", "static_score": 0.7, "original_global_rank": 2},
                            ],
                            "post_floor_shortlist": [
                                {"memory_id": "memory-a", "static_score": 0.8, "original_global_rank": 1},
                                {"memory_id": "memory-b", "static_score": 0.7, "original_global_rank": 2},
                            ],
                            "shortlist_memory_ids": ["memory-a", "memory-b"],
                            "shortlist_nonempty": True,
                            "static_selector_unavailable": False,
                            "unavailable_reason": None,
                            "shortlist_fixed_for_generation": True,
                            "retrieval_method": "exact_cosine",
                            "stable_tie_break": "memory_id_ascending",
                        },
                        "completion_token_ids": [13],
                        "retrieval_attempts": [{
                            "attempt_number": 1,
                            "generated_observation_index": 0,
                            "generated_boundary_index": 0,
                            "query_embedding_token_id": 13,
                            "selected_memory_id": "memory-a",
                            "retrieval_decision": {
                                "schema_version": (
                                    "experience-memory-v3.5-retrieval-decision-v1"
                                ),
                                "status": "selected",
                                "static_shortlist": [
                                    {"memory_id": "memory-a"},
                                    {"memory_id": "memory-b"},
                                ],
                                "hits": [
                                    {"memory_id": "memory-a", "score": 0.8},
                                    {"memory_id": "memory-b", "score": 0.4},
                                ],
                                "query": {
                                    "method": (
                                        "exact_cosine_within_static_applicability_shortlist"
                                    ),
                                    "context": "question_plus_full_partial_cot",
                                    "encoder_state": (
                                        "pure_prefix_reencode_side_kv_disabled"
                                    ),
                                    "side_kv_disabled": True,
                                    "prompt_token_count": 3,
                                    "partial_cot_token_count": 1,
                                    "query_token_count": 4,
                                    "query_token_ids": [10, 11, 12, 13],
                                    "query_token_ids_sha256": canonical_json_sha256([10, 11, 12, 13]),
                                    "query_embedding_sha256": "embedding",
                                    "query_embedding_norm": 1.0,
                                    "layer_number": 24,
                                    "pooling": "current_generated_token",
                                    "normalization": "l2",
                                    "encoded_full_prefix_token_count": 4,
                                    "query_embedding_token_index": 3,
                                    "query_embedding_token_id": 13,
                                    "query_embedding_causal_context_token_count": 4,
                                    "static_shortlist_fixed_for_generation": True,
                                    "dynamic_search_restricted_to_static_shortlist": True,
                                    "dynamic_search_candidate_count": 2,
                                    "selected_memory_kv_metadata_aligned": True,
                                    "minimum_applicability_score": 0.2,
                                    "top1_score": 0.8,
                                    "top2_score": 0.4,
                                    # Deliberately differs from 0.8 - 0.4.
                                    "top1_top2_margin": 0.7,
                                },
                            },
                        }],
                    }
                }
            },
        }
        row["row_sha256"] = canonical_json_sha256(row)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed reproduction"):
                CALIBRATE.collect_first_attempts(
                    path,
                    profile_sha256="profile",
                    known_memory_ids={"memory-a", "memory-b"},
                )

            # Re-hashing a row after a one-ULP, internally self-consistent score
            # edit must not defeat the independent CPU-float32 replay binding.
            decision = row["conditions"]["v3"]["runtime_trace"][
                "retrieval_attempts"
            ][0]["retrieval_decision"]
            tampered_top1 = math.nextafter(0.8, math.inf)
            decision["hits"][0]["score"] = tampered_top1
            decision["query"]["top1_score"] = tampered_top1
            decision["query"]["top1_top2_margin"] = tampered_top1 - 0.4
            row["row_sha256"] = CALIBRATE._row_sha256(row)
            self.assertEqual(row["row_sha256"], CALIBRATE._row_sha256(row))
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            class FakeQueryTensor:
                def reshape(self, *_shape):
                    return self

                def float(self):
                    return self

                def contiguous(self):
                    return self

            class FakeDynamicBank:
                def __getitem__(self, _indices):
                    return self

            class FakeScalar:
                def __init__(self, value):
                    self.value = value

                def item(self):
                    return self.value

            class FakeScores:
                def __getitem__(self, index):
                    return FakeScalar((0.8, 0.4)[index])

            fake_torch = types.ModuleType("torch")
            fake_torch.mv = lambda _bank, _query: FakeScores()
            with (
                mock.patch.object(
                    CALIBRATE,
                    "_verify_query_sidecar",
                    return_value={"attempt_01": FakeQueryTensor()},
                ),
                mock.patch.dict(sys.modules, {"torch": fake_torch}),
                self.assertRaisesRegex(ValueError, "failed reproduction"),
            ):
                CALIBRATE.collect_first_attempts(
                    path,
                    profile_sha256="profile",
                    known_memory_ids={"memory-a", "memory-b"},
                    sidecar_root=Path(temporary),
                    dynamic_embeddings=FakeDynamicBank(),
                    ordered_memory_ids=("memory-a", "memory-b"),
                )

    def test_static_trace_direct_list_and_availability_are_authenticated(self):
        row = valid_v35_unavailable_row()
        row["question_sha256"] = "question"
        static = row["conditions"]["v3"]["runtime_trace"][
            "static_selector_trace"
        ]
        static.update({
            "query": {
                "schema_version": (
                    "experience-memory-v3.5-static-question-query-v1"
                ),
                "static_question_text_sha256": "question",
                "static_question_token_ids": [1],
                "static_question_token_count": 1,
                "static_question_token_ids_sha256": canonical_json_sha256([1]),
                "static_question_embedding_sha256": "embedding",
                "static_question_embedding_norm": 1.0,
                "layer_number": 24,
                "representation": "decoder_layer_output",
                "pooling": "last_valid_token",
                "normalization": "l2",
                "side_kv_disabled": True,
                "chat_wrapper_included": False,
                "prompt_boilerplate_included": False,
                "add_special_tokens": False,
            },
            "score_floor_tie_policy": (
                "retain_score_greater_than_or_equal_to_floor"
            ),
            "pre_floor_top_k": [
                {
                    "memory_id": "memory-a",
                    "static_score": 0.8,
                    "original_global_rank": 1,
                },
                {
                    "memory_id": "memory-b",
                    "static_score": 0.7,
                    "original_global_rank": 2,
                },
            ],
            "post_floor_shortlist": [
                {
                    "memory_id": "memory-a",
                    "static_score": 0.8,
                    "original_global_rank": 1,
                },
                {
                    "memory_id": "memory-b",
                    "static_score": 0.7,
                    "original_global_rank": 2,
                },
            ],
            "shortlist_memory_ids": ["memory-a", "memory-b"],
            "shortlist_nonempty": True,
            "static_selector_unavailable": False,
            "unavailable_reason": None,
            "retrieval_method": "exact_cosine",
            "stable_tie_break": "memory_id_ascending",
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            for field, value in (
                ("shortlist_memory_ids", ["memory-b", "memory-a"]),
                ("static_selector_unavailable", True),
            ):
                with self.subTest(field=field):
                    tampered = json.loads(json.dumps(row))
                    tampered["conditions"]["v3"]["runtime_trace"][
                        "static_selector_trace"
                    ][field] = value
                    tampered["row_sha256"] = CALIBRATE._row_sha256(tampered)
                    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "static selector"):
                        CALIBRATE.collect_first_attempts(
                            path,
                            profile_sha256="profile",
                            known_memory_ids={"memory-a", "memory-b"},
                        )

    def test_v35_analyzer_accepts_static_unavailable_native_parity(self):
        audit = ANALYSIS.IntegrityAudit()
        safety = ANALYSIS.V35SafetyAudit()
        sample = ANALYSIS.extract_sample(
            valid_v35_unavailable_row(),
            expected_profile_sha256="profile",
            expected_risk_role="online_joint_control",
            audit=audit,
            streaming=ANALYSIS.StreamingDiagnostics(),
            validate_hash=False,
            is_v35=True,
            safety=safety,
        )
        self.assertTrue(sample.static_selector_unavailable)
        self.assertTrue(sample.completion_exact_match)
        self.assertTrue(audit.to_dict()["passed"], audit.to_dict())
        self.assertTrue(safety.to_dict()["passed"])
        mismatched = ANALYSIS.IntegrityAudit()
        ANALYSIS.extract_sample(
            valid_v35_unavailable_row(),
            expected_profile_sha256="profile",
            expected_risk_role="online_joint_control",
            audit=mismatched,
            streaming=ANALYSIS.StreamingDiagnostics(),
            validate_hash=False,
            is_v35=True,
            safety=ANALYSIS.V35SafetyAudit(),
            expected_generation_schema=(
                "experience-memory-v3.4-generation-result-v1"
            ),
        )
        self.assertEqual(
            mismatched.to_dict()["failure_counts"][
                "runtime_trace_schema_matches"
            ],
            1,
        )

    def test_safety_violations_are_kept_separate(self):
        safety = ANALYSIS.V35SafetyAudit()
        for name in safety.NAMES:
            safety.violation(name, f"sample-{name}", True)
        value = safety.to_dict()
        self.assertFalse(value["passed"])
        self.assertEqual(set(value["violation_counts"]), set(safety.NAMES))
        self.assertTrue(all(count == 1 for count in value["violation_counts"].values()))

    def test_answer_marker_distance_contract_and_descriptive_summary(self):
        attempts = [
            {
                "attempt_number": 1,
                "generated_observation_index": 1,
                "affects_generated_token_index": 2,
            },
            {
                "attempt_number": 2,
                "generated_observation_index": 10,
                "affects_generated_token_index": 11,
            },
        ]
        diagnostics = {
            "first_answer_marker_token_index": 20,
            "answer_marker_attempt_distances": [
                {
                    "attempt_number": 1,
                    "generated_observation_index": 1,
                    "affects_generated_token_index": 2,
                    "first_answer_marker_token_index": 20,
                    "tokens_until_first_answer_marker": 18,
                },
                {
                    "attempt_number": 2,
                    "generated_observation_index": 10,
                    "affects_generated_token_index": 11,
                    "first_answer_marker_token_index": 20,
                    "tokens_until_first_answer_marker": 9,
                },
            ],
            "attempt_affects_index_contract_respected": True,
            "attempts_with_subsequent_answer_marker_count": 2,
            "late_attempt_within_32_tokens_count": 2,
        }
        marker = ANALYSIS.answer_marker_distance_contract(
            attempts, diagnostics, first_marker_token_index=20
        )
        self.assertTrue(marker["valid"])
        self.assertEqual(marker["distances"], (18, 9))
        tampered = json.loads(json.dumps(diagnostics))
        tampered["late_attempt_within_32_tokens_count"] = 1
        self.assertFalse(ANALYSIS.answer_marker_distance_contract(
            attempts, tampered, first_marker_token_index=20
        )["valid"])

        audit = ANALYSIS.IntegrityAudit()
        base = ANALYSIS.extract_sample(
            valid_v35_unavailable_row(),
            expected_profile_sha256="profile",
            expected_risk_role="online_joint_control",
            audit=audit,
            streaming=ANALYSIS.StreamingDiagnostics(),
            validate_hash=False,
            is_v35=True,
            safety=ANALYSIS.V35SafetyAudit(),
        )
        sample = replace(
            base,
            attempt_count=2,
            attempt_to_first_answer_marker_distances=(18, 9),
            attempts_with_subsequent_answer_marker_count=2,
            late_attempt_within_32_tokens_count=2,
        )
        summary = ANALYSIS.scope_summary([sample])[
            "descriptive_attempt_to_first_answer_marker"
        ]
        self.assertFalse(summary["formal_metric"])
        self.assertEqual(summary["distance_tokens"]["median"], 13.5)
        self.assertEqual(summary["late_attempt_within_32_tokens_count"], 2)

    def test_v31_risk_difference_is_allowed_but_v34_is_not(self):
        common = {
            "logical_split": "dev-test",
            "dataset_split": "test",
            "dataset_revision": "dataset",
            "selected_sample_count": 2,
            "selected_sample_ids_sha256": "ids",
            "reasoner": {"name": "model"},
            "prompt_contract": {"name": "prompt"},
            "generation": {"max_new_tokens": 8, "vanilla": {"greedy": True}},
            "inputs": {
                "split_manifest_sha256": "split",
                "memory_records_sha256": "memory",
                "side_kv_manifest_sha256": "kv",
                "e0_final_report_sha256": "e0",
                "retrieval_key_manifest_sha256": "keys",
                "v3_offline_report_sha256": "offline",
                "risk_artifact_sha256": "old-risk",
            },
            "hysteresis_gate": {
                "high_entropy_threshold": 4.5,
                "low_entropy_threshold": 4.0,
                "risk_threshold": 0.007,
                "rearm_low_entropy_token_count": 2,
            },
        }
        treatment = json.loads(json.dumps(common))
        treatment["inputs"]["risk_artifact_sha256"] = "token-risk"
        COMPARE.validate_profile_compatibility(
            common, treatment, baseline_version="v3.1"
        )
        with self.assertRaisesRegex(ValueError, "token-risk artifacts"):
            COMPARE.validate_profile_compatibility(
                common, treatment, baseline_version="v3.4"
            )
        treatment["inputs"]["risk_artifact_sha256"] = "old-risk"
        treatment["hysteresis_gate"]["risk_threshold"] = 0.008
        with self.assertRaisesRegex(ValueError, "token-risk gates"):
            COMPARE.validate_profile_compatibility(
                common, treatment, baseline_version="v3.4"
            )
        treatment = json.loads(json.dumps(common))
        treatment["inputs"]["retrieval_key_manifest_sha256"] = "other-keys"
        with self.assertRaisesRegex(ValueError, "different frozen inputs"):
            COMPARE.validate_profile_compatibility(
                common, treatment, baseline_version="v3.1"
            )

    def test_vanilla_hash_mismatch_fails_closed(self):
        left = valid_v35_unavailable_row()
        right = json.loads(json.dumps(left))
        right["conditions"]["vanilla"]["completion_token_ids"] = [9, 0]
        sample, vanilla = COMPARE._identity_mismatches(
            {"sample": left}, {"sample": right}
        )
        self.assertEqual(sample, [])
        self.assertEqual(vanilla, ["sample"])
        right = json.loads(json.dumps(left))
        right["answer_sha256"] = "different-ground-truth"
        sample, vanilla = COMPARE._identity_mismatches(
            {"sample": left}, {"sample": right}
        )
        self.assertEqual(sample, ["sample"])
        self.assertEqual(vanilla, [])

    def test_v35_profile_and_selector_source_hashes_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            selector_path = Path(temporary) / "selector.json"
            selector = selector_artifact()
            selector_path.write_text(json.dumps(selector), encoding="utf-8")
            profile = {
                "system_version": "v3.5",
                "logical_split": "dev-test",
                "calibration_trace_only": False,
                "task_results_used_for_selector_decision": False,
                "system_profile": {
                    "schema_version": "experience-memory-system-profile-v3.5",
                    "selector_policy": V35_SELECTOR_POLICY,
                    "risk_role": "online_joint_control",
                    "boundary_policy": "none_pre_answer_every_generated_token",
                    "query_pooling": "current_generated_token",
                    "abstain_policy": (
                        "terminal_consume_attempt_clear_current_memory"
                    ),
                    "calibration_trace_only": False,
                    "applicability_shortlist_k": 2,
                    "applicability_score_floor": 0.2,
                    "retrieval_min_top1_top2_margin": 0.2,
                },
                "inputs": {
                    "selector_calibration_sha256": file_sha256(selector_path),
                    "dual_key_manifest_sha256": "1" * 64,
                    "applicability_calibration_sha256": "3" * 64,
                    "risk_artifact_sha256": "5" * 64,
                },
            }
            self.assertEqual(
                COMPARE.validate_v35_profile(
                    profile, selector_path=selector_path
                )["artifact_sha256"],
                selector["artifact_sha256"],
            )
            profile["inputs"]["dual_key_manifest_sha256"] = "9" * 64
            with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                COMPARE.validate_v35_profile(
                    profile, selector_path=selector_path
                )

    def test_qualification_can_pass_user_review_but_never_final_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selector_path = root / "selector.json"
            selector = selector_artifact()
            selector_path.write_text(json.dumps(selector), encoding="utf-8")
            load_v35_selector_calibration(selector_path)

            comparison = {
                "schema_version": (
                    "experience-memory-v3.5-applicability-selector-comparison-v1"
                ),
                "created_at": "2026-01-01T00:00:00+00:00",
                "status": "completed",
                "logical_split": "dev-test",
                "baseline_version": "v3.4",
                "integrity": {"passed": True},
                "paired_v35_minus_v34": {
                    "strict": {
                        "paired_sample_count": 2,
                        "mean_treatment_minus_control": 0.0,
                        "bootstrap_95_ci": [0.0, 0.0],
                    },
                    "format": {
                        "paired_sample_count": 2,
                        "mean_treatment_minus_control": 0.0,
                    },
                },
                "selector": {"artifact_sha256": selector["artifact_sha256"]},
                "inputs": {
                    "v35_selector_calibration_sha256": file_sha256(selector_path),
                    "v35_profile_sha256": "profile",
                    "v35_profile_file_sha256": "profile-file",
                    "v35_results_sha256": "results-file",
                },
            }
            comparison["report_sha256"] = canonical_json_sha256({
                key: value
                for key, value in comparison.items()
                if key not in {"created_at", "report_sha256"}
            })
            comparison_path = root / "comparison.json"
            comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

            analysis = {
                "schema_version": "experience-memory-v3-analysis-report-v1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "status": "completed",
                "run": {
                    "profile_sha256": "profile",
                    "run_profile_file_sha256": "profile-file",
                    "results_file_sha256": "results-file",
                    "logical_split": "dev-test",
                    "selected_sample_count": 2,
                    "system_profile": {
                        "schema_version": "experience-memory-system-profile-v3.5"
                    },
                },
                "integrity": {"passed": True},
                "paired_analysis": {"overall": {"sample_count": 2}},
                "zero_attempt_parity": {"mismatch_count": 0},
                "static_selector_unavailable_parity": {"mismatch_count": 0},
                "safety": {
                    "applicable": True,
                    "passed": True,
                    "violation_counts": {
                        name: 0 for name in ANALYSIS.V35SafetyAudit.NAMES
                    }
                },
            }
            analysis["report_sha256"] = canonical_json_sha256(analysis)
            analysis_path = root / "analysis.json"
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            output_path = root / "qualification.json"
            subprocess.run(
                [
                    sys.executable,
                    str(QUALIFY_PATH),
                    "--comparison",
                    str(comparison_path),
                    "--analysis",
                    str(analysis_path),
                    "--selector-calibration",
                    str(selector_path),
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            qualified = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(qualified["qualified_for_user_review"])
            self.assertFalse(qualified["qualified_for_final_test"])
            self.assertTrue(
                qualified["final_test_blocked_pending_explicit_user_authorization"]
            )

            mixed_analysis = json.loads(json.dumps(analysis))
            mixed_analysis["run"]["results_file_sha256"] = "other-results-file"
            mixed_analysis["report_sha256"] = canonical_json_sha256({
                key: value
                for key, value in mixed_analysis.items()
                if key != "report_sha256"
            })
            mixed_path = root / "mixed-analysis.json"
            mixed_path.write_text(json.dumps(mixed_analysis), encoding="utf-8")
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(QUALIFY_PATH),
                    "--comparison",
                    str(comparison_path),
                    "--analysis",
                    str(mixed_path),
                    "--selector-calibration",
                    str(selector_path),
                    "--output",
                    str(root / "mixed-qualification.json"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("mutually authenticated", rejected.stderr)

            relaxed = subprocess.run(
                [
                    sys.executable,
                    str(QUALIFY_PATH),
                    "--comparison",
                    str(comparison_path),
                    "--analysis",
                    str(analysis_path),
                    "--selector-calibration",
                    str(selector_path),
                    "--output",
                    str(root / "relaxed-qualification.json"),
                    "--minimum-strict-delta",
                    "-1.0",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(relaxed.returncode, 0)
            self.assertIn("thresholds are frozen", relaxed.stderr)


if __name__ == "__main__":
    unittest.main()
