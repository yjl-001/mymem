from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from memgen.experience.e1 import MemoryChoice
from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v3 import (
    ApplicabilityAwareRetrievalDecision,
    ExperienceMemoryV3Profile,
    V34_QUERY_POOLING_CURRENT_TOKEN,
    V34_SYSTEM_PROFILE_SCHEMA,
    V35_GENERATION_RESULT_SCHEMA,
    V35_RETRIEVAL_DECISION_SCHEMA,
    V35_SELECTOR_POLICY,
    V35_SYSTEM_PROFILE_SCHEMA,
    query_embedding_token_index,
)
from memgen.experience.v3_5_selector import (
    V35_APPLICABILITY_CALIBRATION_SCHEMA,
    V35_APPLICABILITY_FLOOR_ROLE,
    V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY,
    V35_DUAL_KEY_BANK_SCHEMA,
    V35_DYNAMIC_MARGIN_TIE_POLICY,
    V35_SELECTOR_CALIBRATION_SCHEMA,
    applicability_score_floor,
    calibrate_applicability_selector,
    deterministic_source_pair_partition,
    load_v35_applicability_calibration,
    load_v35_selector_calibration,
    own_memory_rank_metrics,
    retained_dynamic_margin_threshold,
    select_minimal_shortlist_k,
    v35_artifact_sha256,
)


def _write_artifact(path: Path, value: dict) -> None:
    value["artifact_sha256"] = v35_artifact_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_sha256(label: str) -> str:
    return canonical_json_sha256({"fixture": label})


def _question_query(index: int) -> dict:
    token_ids = [index + 1]
    return {
        "schema_version": "experience-memory-v3.5-static-question-query-v1",
        "static_question_text_sha256": _fixture_sha256(f"question-{index}"),
        "static_question_token_count": len(token_ids),
        "static_question_token_ids_sha256": canonical_json_sha256(token_ids),
        "static_question_embedding_sha256": _fixture_sha256(
            f"question-embedding-{index}"
        ),
        "static_question_embedding_norm": 1.0,
        "static_question_token_ids": token_ids,
        "layer_number": 24,
        "representation": "decoder_layer_output",
        "pooling": "last_valid_token",
        "normalization": "l2",
        "side_kv_disabled": True,
        "chat_wrapper_included": False,
        "prompt_boilerplate_included": False,
        "add_special_tokens": False,
    }


def _applicability_source() -> dict:
    implementation_files = {
        "memgen/experience/v3_5_selector.py": _fixture_sha256(
            "selector-implementation"
        )
    }
    source = {
        field: _fixture_sha256(field)
        for field in (
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
            "dual_key_manifest_sha256",
            "dual_key_manifest_logical_sha256",
            "dual_key_tensor_sha256",
        )
    }
    source.update({
        "dataset_revision": "dataset-revision",
        "compiler_git_revision": "a" * 40,
        "compiler_implementation_files_sha256": implementation_files,
        "compiler_implementation_set_sha256": canonical_json_sha256(
            implementation_files
        ),
        "source_question_encoder": (
            "verified_experience.context.strip_question_only"
        ),
    })
    return source


def _selector_requirements() -> dict[str, bool]:
    return {
        name: True
        for name in (
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
        )
    }


class V35ProfileAndDecisionContractTests(unittest.TestCase):
    def test_profile_has_independent_frozen_v35_semantics_and_roundtrip(self) -> None:
        profile = ExperienceMemoryV3Profile.applicability_aware_continuous(
            applicability_shortlist_k=7,
            applicability_score_floor=0.42,
            retrieval_min_top1_top2_margin=0.006,
        )
        self.assertEqual(profile.schema_version, V35_SYSTEM_PROFILE_SCHEMA)
        self.assertEqual(profile.layer_number, 24)
        self.assertEqual(profile.query_pooling, V34_QUERY_POOLING_CURRENT_TOKEN)
        self.assertEqual(
            query_embedding_token_index(
                token_count=9, pooling=profile.query_pooling
            ),
            8,
        )
        self.assertEqual(
            profile.boundary_policy,
            "none_pre_answer_every_generated_token",
        )
        self.assertEqual(
            profile.selector_policy,
            V35_SELECTOR_POLICY,
        )
        self.assertEqual(
            profile.abstain_policy,
            "terminal_consume_attempt_clear_current_memory",
        )
        self.assertEqual(
            profile.injection_policy,
            "persistent_until_replace_terminal_abstain_or_eos",
        )
        self.assertEqual(profile.max_retrieval_attempts, 3)
        self.assertEqual(profile.rearm_low_entropy_token_count, 2)
        self.assertEqual(
            ExperienceMemoryV3Profile.from_dict(profile.to_dict()), profile
        )
        self.assertEqual(
            V35_GENERATION_RESULT_SCHEMA,
            "experience-memory-v3.5-generation-result-v1",
        )

    def test_trace_only_mode_is_explicit_and_final_margin_fails_closed(self) -> None:
        trace = ExperienceMemoryV3Profile.applicability_aware_continuous(
            applicability_shortlist_k=1,
            applicability_score_floor=-0.1,
            calibration_trace_only=True,
        )
        self.assertTrue(trace.calibration_trace_only)
        self.assertEqual(trace.retrieval_abstention_policy, "disabled")
        self.assertIsNone(trace.retrieval_min_top1_top2_margin)
        with self.assertRaisesRegex(ValueError, "requires a frozen"):
            ExperienceMemoryV3Profile.applicability_aware_continuous(
                applicability_shortlist_k=3,
                applicability_score_floor=0.0,
            )
        with self.assertRaisesRegex(ValueError, "cannot freeze"):
            ExperienceMemoryV3Profile.applicability_aware_continuous(
                applicability_shortlist_k=3,
                applicability_score_floor=0.0,
                retrieval_min_top1_top2_margin=0.1,
                calibration_trace_only=True,
            )
        with self.assertRaisesRegex(ValueError, "layer 24"):
            ExperienceMemoryV3Profile(
                layer_number=23,
                schema_version=V35_SYSTEM_PROFILE_SCHEMA,
            )

    def test_v31_and_v34_profile_roundtrip_remains_unchanged(self) -> None:
        v31 = ExperienceMemoryV3Profile()
        v34 = ExperienceMemoryV3Profile.continuous_token_joint(
            retrieval_abstention_policy="top1_top2_margin",
            retrieval_min_top1_top2_margin=0.004,
        )
        self.assertEqual(
            ExperienceMemoryV3Profile.from_dict(v31.to_dict()), v31
        )
        self.assertEqual(v34.schema_version, V34_SYSTEM_PROFILE_SCHEMA)
        self.assertEqual(
            ExperienceMemoryV3Profile.from_dict(v34.to_dict()), v34
        )
        self.assertEqual(v31.abstain_policy, "consume_attempt_keep_current_memory")
        self.assertEqual(v34.abstain_policy, "consume_attempt_keep_current_memory")

        # Historical serialized profiles predate the V3.5-only fields.  Their
        # absence must resolve to the old selector semantics, not be treated as
        # profile drift.
        for historical in (v31.to_dict(), v34.to_dict()):
            for field_name in (
                "selector_policy",
                "applicability_shortlist_k",
                "applicability_score_floor",
                "calibration_trace_only",
            ):
                historical.pop(field_name)
            restored = ExperienceMemoryV3Profile.from_dict(historical)
            self.assertEqual(
                restored.schema_version, historical["schema_version"]
            )
            self.assertEqual(
                restored.selector_policy,
                "global_full_prefix_exact_cosine",
            )
            self.assertIsNone(restored.applicability_shortlist_k)

    def test_decision_requires_selected_memory_to_belong_to_shortlist(self) -> None:
        choice = MemoryChoice(
            memory_id="memory-b",
            payload_hash="payload-b",
            token_count=5,
            kv_valid_slot_count=5,
            retrieval_score=0.7,
            retrieval_rank=1,
        )
        shortlist = (
            {"memory_id": "memory-a", "static_score": 0.8},
            {"memory_id": "memory-b", "static_score": 0.7},
        )
        decision = ApplicabilityAwareRetrievalDecision(
            status="selected",
            query={"top1_top2_margin": 0.1},
            hits=(
                {"memory_id": "memory-b", "score": 0.7},
                {"memory_id": "memory-a", "score": 0.6},
            ),
            matched_memory=choice,
            static_shortlist=shortlist,
        )
        self.assertTrue(decision.selected)
        self.assertEqual(
            decision.to_dict()["schema_version"],
            V35_RETRIEVAL_DECISION_SCHEMA,
        )
        with self.assertRaisesRegex(ValueError, "belong"):
            ApplicabilityAwareRetrievalDecision(
                status="selected",
                query={},
                hits=({"memory_id": "memory-b", "score": 0.7},),
                matched_memory=choice,
                static_shortlist=(
                    {"memory_id": "memory-a", "static_score": 0.8},
                ),
            )
        abstain = ApplicabilityAwareRetrievalDecision(
            status="below_dynamic_margin",
            query={"top1_top2_margin": 0.001},
            hits=(
                {"memory_id": "memory-a", "score": 0.6},
                {"memory_id": "memory-b", "score": 0.59},
            ),
            matched_memory=None,
            static_shortlist=shortlist,
        )
        self.assertFalse(abstain.selected)


class V35PureCalibrationContractTests(unittest.TestCase):
    def test_schema_names_are_independent_of_old_selector_artifacts(self) -> None:
        self.assertEqual(
            V35_DUAL_KEY_BANK_SCHEMA,
            "experience-memory-v3.5-dual-key-bank-v1",
        )
        self.assertEqual(
            V35_APPLICABILITY_CALIBRATION_SCHEMA,
            "experience-memory-v3.5-applicability-calibration-v1",
        )
        self.assertEqual(
            V35_SELECTOR_CALIBRATION_SCHEMA,
            "experience-memory-v3.5-selector-calibration-v1",
        )

    def test_partition_is_frozen_deterministic_and_uses_both_ids(self) -> None:
        assignments = [
            deterministic_source_pair_partition(f"memory-{index}", f"source-{index}")
            for index in range(100)
        ]
        self.assertEqual(
            assignments,
            [
                deterministic_source_pair_partition(
                    f"memory-{index}", f"source-{index}"
                )
                for index in range(100)
            ],
        )
        self.assertIn("train", assignments)
        self.assertIn("holdout", assignments)
        changed_sources = [
            deterministic_source_pair_partition(
                f"memory-{index}", f"source-other-{index}"
            )
            for index in range(100)
        ]
        self.assertNotEqual(assignments, changed_sources)
        with self.assertRaisesRegex(ValueError, "3501/0.8"):
            deterministic_source_pair_partition(
                "memory", "source", seed=3502
            )

    def test_minimal_k_recall_metrics_and_failure_at_32(self) -> None:
        ranks = (1,) * 94 + (2,) * 6
        self.assertEqual(
            select_minimal_shortlist_k(ranks, memory_count=40), 2
        )
        metrics = own_memory_rank_metrics(
            ranks, memory_count=40, shortlist_k=2
        )
        self.assertEqual(metrics["recall_at_1"], 0.94)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertGreater(metrics["mrr"], 0.9)
        self.assertIsNone(
            select_minimal_shortlist_k(
                (33,) * 100, memory_count=40
            )
        )

    def test_fifth_percentile_floor_and_admission_are_inclusive(self) -> None:
        result = applicability_score_floor((0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertAlmostEqual(result["minimum_applicability_score"], 0.05)
        self.assertEqual(result["positive_retained_count"], 4)
        self.assertEqual(
            result["applicability_score_floor_tie_policy"],
            V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY,
        )
        tied = applicability_score_floor((0.2, 0.2, 0.2, 0.4))
        self.assertEqual(tied["minimum_applicability_score"], 0.2)
        self.assertEqual(tied["positive_retained_count"], 4)

    def test_applicability_calibration_qualifies_train_and_heldout_blindly(self) -> None:
        pairs = [
            {
                "memory_id": f"memory-{index:03d}",
                "source_experience_id": f"source-{index:03d}",
                "own_memory_rank": 1,
                "own_positive_score": 0.7,
            }
            for index in range(100)
        ]
        result = calibrate_applicability_selector(pairs, memory_count=100)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["task_accuracy_used"])
        self.assertFalse(result["answer_or_reward_used"])
        self.assertEqual(result["partition"]["seed"], 3501)
        self.assertEqual(result["partition"]["train_fraction"], 0.8)
        self.assertEqual(result["calibration"]["shortlist_k"], 1)
        self.assertEqual(
            result["calibration"]["heldout_own_memory_recall_at_k"], 1.0
        )
        self.assertEqual(
            result["calibration"]["heldout_own_positive_retained_fraction"],
            1.0,
        )
        self.assertEqual(
            result["calibration"]["applicability_floor_role"],
            V35_APPLICABILITY_FLOOR_ROLE,
        )
        with self.assertRaisesRegex(ValueError, "answer-blind"):
            calibrate_applicability_selector(
                [pairs[0] | {"answer": "forbidden"}], memory_count=1
            )

    def test_heldout_recall_or_positive_retention_failure_is_not_qualified(self) -> None:
        pairs = []
        for index in range(100):
            memory_id = f"memory-{index:03d}"
            source_id = f"source-{index:03d}"
            heldout = (
                deterministic_source_pair_partition(memory_id, source_id)
                == "holdout"
            )
            pairs.append({
                "memory_id": memory_id,
                "source_experience_id": source_id,
                "own_memory_rank": 33 if heldout else 1,
                "own_positive_score": -0.5 if heldout else 0.5,
            })
        result = calibrate_applicability_selector(pairs, memory_count=100)
        self.assertEqual(result["status"], "not_qualified")
        self.assertFalse(
            result["requirements"][
                "heldout_own_memory_recall_at_k_at_least_0_95"
            ]
        )
        self.assertFalse(
            result["requirements"][
                "heldout_positive_retained_fraction_at_least_0_90"
            ]
        )

    def test_dynamic_threshold_retains_50_percent_with_inclusive_ties(self) -> None:
        result = retained_dynamic_margin_threshold((0.1, 0.2, 0.3, 0.4))
        self.assertEqual(result["minimum_dynamic_top1_top2_margin"], 0.3)
        self.assertEqual(result["actual_retained_fraction"], 0.5)
        tied = retained_dynamic_margin_threshold((0.1, 0.2, 0.2, 0.4))
        self.assertEqual(tied["minimum_dynamic_top1_top2_margin"], 0.2)
        self.assertEqual(tied["actual_retained_count"], 3)
        self.assertEqual(
            tied["dynamic_margin_tie_policy"],
            V35_DYNAMIC_MARGIN_TIE_POLICY,
        )

    def test_logical_hash_ignores_time_but_not_contract_content(self) -> None:
        first = {
            "schema_version": V35_SELECTOR_CALIBRATION_SCHEMA,
            "created_at": "first",
            "status": "passed",
        }
        second = dict(first, created_at="second")
        self.assertEqual(v35_artifact_sha256(first), v35_artifact_sha256(second))
        second["status"] = "not_qualified"
        self.assertNotEqual(
            v35_artifact_sha256(first), v35_artifact_sha256(second)
        )


class V35FailClosedLoaderTests(unittest.TestCase):
    @staticmethod
    def _applicability_artifact() -> dict:
        pairs = [
            {
                "memory_id": f"memory-{index:03d}",
                "source_experience_id": f"source-{index:03d}",
                "own_memory_rank": 1,
                "own_positive_score": 0.7,
                "question_query": _question_query(index),
            }
            for index in range(100)
        ]
        calibrated = calibrate_applicability_selector(
            pairs, memory_count=100
        )
        return {
            "schema_version": V35_APPLICABILITY_CALIBRATION_SCHEMA,
            "created_at": "2026-01-01T00:00:00+00:00",
            "source": _applicability_source(),
            **calibrated,
            "source_pair_audit": pairs,
        }

    @staticmethod
    def _selector_artifact() -> dict:
        dynamic = retained_dynamic_margin_threshold((0.1, 0.2, 0.3, 0.4))
        return {
            "schema_version": V35_SELECTOR_CALIBRATION_SCHEMA,
            "created_at": "2026-01-01T00:00:00+00:00",
            "status": "passed",
            "policy": V35_SELECTOR_POLICY,
            "task_accuracy_used": False,
            "answer_or_reward_used": False,
            "source": {
                "logical_split": "calibration-val",
                "scope": "first_retrieval_attempt_per_triggered_question",
                "run_profile_sha256": _fixture_sha256("run-profile-logical"),
                "run_profile_file_sha256": _fixture_sha256("run-profile-file"),
                "results_file_sha256": _fixture_sha256("results-file"),
                "dual_key_manifest_sha256": _fixture_sha256(
                    "dual-key-manifest-file"
                ),
                "dual_key_manifest_logical_sha256": _fixture_sha256(
                    "dual-key-manifest-logical"
                ),
                "applicability_calibration_sha256": _fixture_sha256(
                    "applicability-file"
                ),
                "applicability_calibration_artifact_sha256": _fixture_sha256(
                    "applicability-logical"
                ),
                "risk_artifact_sha256": _fixture_sha256("risk-file"),
                "system_version": "v3.5",
                "system_profile_schema": V35_SYSTEM_PROFILE_SCHEMA,
                "calibration_trace_only": True,
                "completed_sample_count": 4,
            },
            "calibration": {
                "sample_count": 4,
                "shortlist_k": 8,
                "minimum_applicability_score": 0.2,
                "applicability_score_floor_tie_policy": (
                    V35_APPLICABILITY_SCORE_FLOOR_TIE_POLICY
                ),
                **dynamic,
                "static_selector_available_sample_count": 4,
                "insufficient_shortlist_sample_count": 0,
                "insufficient_shortlist_fraction": 0.0,
                "first_attempt_selected_memory_count": 2,
                "first_attempt_selected_memory_frequency": [
                    {"memory_id": "memory-a", "count": 2},
                    {"memory_id": "memory-b", "count": 2},
                ],
            },
            "requirements": _selector_requirements(),
        }

    def test_applicability_loader_authenticates_and_checks_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "applicability.json"
            artifact = self._applicability_artifact()
            _write_artifact(path, artifact)
            loaded = load_v35_applicability_calibration(
                path,
                expected_input_hashes={
                    "dual_key_manifest_sha256": _fixture_sha256(
                        "dual_key_manifest_sha256"
                    )
                },
            )
            self.assertEqual(loaded["calibration"]["shortlist_k"], 1)
            with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                load_v35_applicability_calibration(
                    path,
                    expected_input_hashes={
                        "dual_key_manifest_sha256": _fixture_sha256("wrong")
                    },
                )
            artifact["task_accuracy_used"] = True
            _write_artifact(path, artifact)
            with self.assertRaisesRegex(ValueError, "answer-blind"):
                load_v35_applicability_calibration(path)

    def test_selector_loader_rejects_old_schema_tampering_and_nonqualification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            artifact = self._selector_artifact()
            _write_artifact(path, artifact)
            loaded = load_v35_selector_calibration(
                path,
                expected_input_hashes={
                    "dual_key_manifest_sha256": _fixture_sha256(
                        "dual-key-manifest-file"
                    ),
                    "risk_artifact_sha256": _fixture_sha256("risk-file"),
                },
            )
            self.assertEqual(
                loaded["calibration"]["minimum_dynamic_top1_top2_margin"],
                0.3,
            )

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["calibration"]["minimum_dynamic_top1_top2_margin"] = 0.4
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_v35_selector_calibration(path)

            old = self._selector_artifact()
            old["schema_version"] = (
                "experience-memory-v3-margin-selector-calibration-v1"
            )
            _write_artifact(path, old)
            with self.assertRaisesRegex(ValueError, "schema/policy"):
                load_v35_selector_calibration(path)

            failed = self._selector_artifact()
            failed["status"] = "not_qualified"
            _write_artifact(path, failed)
            with self.assertRaisesRegex(ValueError, "answer-blind qualification"):
                load_v35_selector_calibration(path)

    def test_applicability_loader_rejects_self_hashed_structural_forgeries(self) -> None:
        mutations = {
            "placeholder requirements": lambda value: value.update(
                requirements={"placeholder": True}
            ),
            "missing source provenance": lambda value: value["source"].pop(
                "memory_records_sha256"
            ),
            "memory count mismatch": lambda value: value["calibration"].update(
                memory_count=99
            ),
            "nondeterministic partition": lambda value: (
                value["partition"]["train_memory_ids"].append(
                    value["partition"]["heldout_memory_ids"].pop()
                )
            ),
            "question contract drift": lambda value: value["source_pair_audit"][
                0
            ]["question_query"].update(side_kv_disabled=False),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "applicability.json"
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    artifact = copy.deepcopy(self._applicability_artifact())
                    mutate(artifact)
                    _write_artifact(path, artifact)
                    with self.assertRaises(ValueError):
                        load_v35_applicability_calibration(path)

    def test_selector_loader_rejects_self_hashed_source_and_count_forgeries(self) -> None:
        mutations = {
            "placeholder requirements": lambda value: value.update(
                requirements={"placeholder": True}
            ),
            "missing run profile hash": lambda value: value["source"].pop(
                "run_profile_file_sha256"
            ),
            "wrong system version": lambda value: value["source"].update(
                system_version="v3.4"
            ),
            "completed count mismatch": lambda value: value["source"].update(
                completed_sample_count=5
            ),
            "first attempt count mismatch": lambda value: value[
                "calibration"
            ].update(first_attempt_count=3),
            "retained count mismatch": lambda value: value["calibration"].update(
                actual_retained_count=3
            ),
            "frequency mismatch": lambda value: value["calibration"][
                "first_attempt_selected_memory_frequency"
            ][0].update(count=1),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    artifact = copy.deepcopy(self._selector_artifact())
                    mutate(artifact)
                    _write_artifact(path, artifact)
                    with self.assertRaises(ValueError):
                        load_v35_selector_calibration(path)

            reward_used = self._selector_artifact()
            reward_used["answer_or_reward_used"] = True
            _write_artifact(path, reward_used)
            with self.assertRaisesRegex(ValueError, "answer-blind qualification"):
                load_v35_selector_calibration(path)


if __name__ == "__main__":
    unittest.main()
