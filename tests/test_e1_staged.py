from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from types import SimpleNamespace
import unittest

from memgen.experience.e1 import MemoryChoice
from memgen.experience.e1_staged import (
    CompletionAwareRetrievalQueryBuilder,
    ConstrainedKMedoidsCatalogBuilder,
    E1BRetrievalAssignment,
    E1BRetrievalDeranger,
    E1CSFixedStrengthDecision,
    E1CTTextSourceDecision,
    render_experience_catalog,
    render_single_experience,
    render_single_experience_guidance,
    render_single_experience_payload,
)
from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.system import ExperienceMemorySystemProfile
from scripts.e1_staged_common import (
    PairedConditionComparison,
    PairedConditionDiagnostics,
    completion_difference_summary,
    e1c_source_mechanism_valid,
    format_transfer_diagnostic,
    strict_accuracy_transition_diagnostics,
    summarize_conditions,
    token_sequence_diagnostic,
)
from scripts.evaluate_e1c_side_kv_channel import compact_trace_artifact
from scripts.evaluate_e1cs_fixed_strength import FixedStrengthTraceAuditor
from scripts.evaluate_e1_experience_memory import compact_persistent_trace


@dataclass(frozen=True)
class FakeMemoryRecord:
    memory_id: str
    payload_hash: str
    sanitized_retrieval_key: str
    sanitized_contrast_payload: str


def choice(memory_id: str, slots: int = 10) -> MemoryChoice:
    return MemoryChoice(
        memory_id=memory_id,
        payload_hash=f"payload-{memory_id}",
        token_count=slots,
        kv_valid_slot_count=slots,
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def retrieval_assignment(sample_index: int, memory_id: str) -> E1BRetrievalAssignment:
    completion = (100 + sample_index, 200 + sample_index)
    return E1BRetrievalAssignment(
        sample_id=f"sample-{sample_index}",
        logical_split="calibration-val",
        dataset_split="train",
        source_index=sample_index,
        question_sha256=f"question-{sample_index}",
        base_prompt_token_ids_sha256=f"prompt-{sample_index}",
        base_prompt_token_count=8,
        preanswer_completion_token_ids=completion,
        preanswer_completion_token_ids_sha256=canonical_json_sha256(list(completion)),
        preanswer_completion_text_sha256=f"text-{sample_index}",
        retrieval_query={"query_hash": f"query-{sample_index}"},
        matched_memory=choice(memory_id),
    )


class RepresentativeCatalogTests(unittest.TestCase):
    def test_catalogs_use_real_records_and_equal_record_counts(self) -> None:
        records = tuple(
            FakeMemoryRecord(
                memory_id=f"mem-{index:02d}",
                payload_hash=f"hash-{index:02d}",
                sanitized_retrieval_key=(
                    f"rate equation verify arithmetic cluster{index % 3} "
                    f"strategy{index}"
                ),
                sanitized_contrast_payload=(
                    "When facing: a multi step relation\n"
                    f"Prefer: verify the operation pattern {chr(97 + index)}\n"
                    "Avoid: assuming an unchecked intermediate result"
                ),
            )
            for index in range(10)
        )
        builder = ConstrainedKMedoidsCatalogBuilder(
            records=records,  # type: ignore[arg-type]
            token_counter=lambda text: len(text.split()),
            token_budget=100,
            maximum_iterations=5,
        )
        representative = builder.build_representative()
        random_control = builder.build_random_control(
            representative=representative, seed=17
        )
        all_ids = {record.memory_id for record in records}
        self.assertTrue(set(representative.memory_ids) <= all_ids)
        self.assertEqual(len(representative.memory_ids), len(random_control.memory_ids))
        self.assertNotEqual(representative.memory_ids, random_control.memory_ids)
        self.assertLessEqual(representative.token_count, 100)
        self.assertLessEqual(random_control.token_count, 100)
        self.assertEqual(
            sum(int(cluster["size"]) for cluster in representative.clusters),
            len(records),
        )

    def test_shared_capacity_makes_every_equal_count_catalog_feasible(self) -> None:
        records = tuple(
            FakeMemoryRecord(
                memory_id=f"mem-{index:02d}",
                payload_hash=f"hash-{index:02d}",
                sanitized_retrieval_key=f"strategy cluster {index}",
                sanitized_contrast_payload=(
                    "When facing: relation\nPrefer: verify\nAvoid: assume "
                    + " ".join(["detail"] * (index * 4))
                ),
            )
            for index in range(8)
        )
        token_counter = lambda text: len(text.split())
        builder = ConstrainedKMedoidsCatalogBuilder(
            records=records,  # type: ignore[arg-type]
            token_counter=token_counter,
            token_budget=80,
        )
        count = builder.capacity_report["universally_feasible_memory_count"]
        self.assertGreater(count, 0)
        for selected in combinations(records, count):
            self.assertLessEqual(
                token_counter(render_experience_catalog(selected)), 80
            )
        representative = builder.build_representative()
        controls_list = []
        for seed in (17, 42, 73):
            controls_list.append(builder.build_random_control(
                representative=representative,
                seed=seed,
                excluded_catalog_memory_ids=[
                    catalog.memory_ids for catalog in controls_list
                ],
            ))
        controls = tuple(controls_list)
        self.assertTrue(all(len(item.memory_ids) == count for item in controls))
        self.assertEqual(
            len({representative.memory_ids, *(item.memory_ids for item in controls)}),
            4,
        )

    def test_single_experience_rendering_has_auditable_components(self) -> None:
        record = FakeMemoryRecord(
            memory_id="mem-a",
            payload_hash="hash-a",
            sanitized_retrieval_key="verify relation",
            sanitized_contrast_payload=(
                "When facing: a relation\n"
                "Prefer: verify the relation\n"
                "Avoid: assuming the result"
            ),
        )
        guidance = render_single_experience_guidance()
        payload = render_single_experience_payload(record)  # type: ignore[arg-type]
        self.assertEqual(
            render_single_experience(record),  # type: ignore[arg-type]
            f"{guidance}\n\n{payload}",
        )
        self.assertNotIn("When facing", guidance)
        self.assertTrue(payload.startswith("When facing:"))


class CompletionAwareQueryTests(unittest.TestCase):
    def test_preanswer_is_truncated_and_math_literals_are_removed(self) -> None:
        builder = CompletionAwareRetrievalQueryBuilder()
        query = builder.build(
            question="A shop combines two rates. What total is needed?",
            completion=(
                "First identify the rate relation and verify each operation. "
                "The intermediate total is 42. Final answer: \\boxed{42}."
            ),
        )
        self.assertNotIn("42", query.normalized_partial_cot)
        self.assertNotIn("boxed", query.normalized_partial_cot.casefold())
        self.assertIn("verify", query.analyzed_terms)
        self.assertEqual(
            query.schema_version,
            "experience-memory-completion-retrieval-query-v1",
        )


class E1BRetrievalAssignmentTests(unittest.TestCase):
    def test_round_trip_and_derangement_preserve_memory_multiset(self) -> None:
        inputs = (
            retrieval_assignment(0, "a"),
            retrieval_assignment(1, "a"),
            retrieval_assignment(2, "b"),
            retrieval_assignment(3, "b"),
        )
        output, report = E1BRetrievalDeranger(seed=42).assign(inputs)
        restored = tuple(
            E1BRetrievalAssignment.from_dict(item.to_dict()) for item in output
        )
        self.assertEqual(restored, output)
        self.assertTrue(all(item.assigned for item in output))
        self.assertTrue(all(
            item.matched_memory.memory_id != item.shuffled_memory.memory_id
            for item in output
            if item.shuffled_memory is not None
        ))
        self.assertEqual(
            report["matched_memory_multiset_sha256"],
            report["shuffled_memory_multiset_sha256"],
        )


class PersistentTraceArtifactTests(unittest.TestCase):
    def test_triggered_persistent_trace_covers_every_post_gate_token(self) -> None:
        profile = ExperienceMemorySystemProfile()
        traces = tuple(
            SimpleNamespace(
                native_key_length=30 + index,
                memory_attention_mass=0.08,
                memory_id="memory-a",
                memory_slot_count=12,
                memory_score_normalization="log_valid_slots",
                memory_score_bias=profile.memory_score_bias,
            )
            for index in range(4)
        )
        artifact = compact_persistent_trace(
            traces,
            completion_length=9,
            prefix_length=30,
            prompt_token_count=25,
            expected_memory_id="memory-a",
            expected_slot_count=12,
            expected_baseline_first_token_id=7,
            actual_baseline_first_token_id=7,
            profile=profile,
        )
        self.assertEqual(artifact["expected_trace_count"], 4)
        self.assertTrue(artifact["one_trace_per_post_trigger_token"])
        self.assertTrue(artifact["native_cache_length_matches_real_tokens"])
        self.assertTrue(
            artifact["baseline_first_token_matches_gate_observation"]
        )

    def test_compact_trace_proves_cache_and_persistence_invariants(self) -> None:
        traces = tuple(
            SimpleNamespace(
                native_key_length=20 + index,
                memory_attention_mass=0.1 + index * 0.01,
                memory_id="memory-a",
                memory_slot_count=12,
                memory_score_normalization="log_valid_slots",
            )
            for index in range(3)
        )
        artifact = compact_trace_artifact(
            traces, completion_length=3, prompt_length=20
        )
        self.assertTrue(artifact["one_trace_per_generated_token"])
        self.assertTrue(artifact["native_cache_length_matches_real_tokens"])
        self.assertTrue(artifact["memory_id_constant"])
        self.assertTrue(artifact["normalization_constant"])

    def test_fixed_strength_trace_records_the_preregistered_bias(self) -> None:
        traces = tuple(
            SimpleNamespace(
                native_key_length=20 + index,
                memory_attention_mass=0.12,
                memory_id="memory-a",
                memory_slot_count=12,
                memory_score_normalization="log_valid_slots",
                memory_score_bias=math.log(10.0),
            )
            for index in range(3)
        )
        artifact = FixedStrengthTraceAuditor(
            normalization="log_valid_slots",
            memory_score_bias=math.log(10.0),
        ).compact(traces, completion_length=3, prompt_length=20)
        self.assertTrue(artifact["memory_score_bias_constant"])
        self.assertTrue(
            artifact["memory_score_bias_matches_preregistered_value"]
        )
        self.assertAlmostEqual(artifact["memory_odds_multiplier"], 10.0)

    def test_e1ct_validates_mechanism_from_authenticated_result_rows(self) -> None:
        path = "split-before-final-prompt-token-v1"
        side_kv = {
            "one_trace_per_generated_token": True,
            "native_cache_length_matches_real_tokens": True,
            "all_memory_attention_mass_finite_and_positive": True,
            "memory_id_constant": True,
            "memory_slot_count_constant": True,
            "normalization_constant": True,
            "baseline_first_token_matches_split_no_memory": True,
            "memory_score_normalizations": ["log_valid_slots"],
        }
        record = {
            "primary_prefill_path": path,
            "prefill_path_diagnostics": {
                "split_no_memory_repeat": {"exact_match": True}
            },
            "conditions": {
                condition: {
                    "prefill_path": path,
                    "side_kv": (
                        dict(side_kv) if "persistent" in condition else None
                    ),
                }
                for condition in (
                    "split_no_memory",
                    "split_matched_text",
                    "split_shuffled_text",
                    "matched_persistent_side_kv",
                    "shuffled_persistent_side_kv",
                )
            },
        }
        self.assertTrue(e1c_source_mechanism_valid(
            record,
            split_prefill_path=path,
            expected_normalization="log_valid_slots",
        ))
        record["conditions"]["matched_persistent_side_kv"]["side_kv"][
            "normalization_constant"
        ] = False
        self.assertFalse(e1c_source_mechanism_valid(
            record,
            split_prefill_path=path,
            expected_normalization="log_valid_slots",
        ))


class StagedDiagnosticSummaryTests(unittest.TestCase):
    @staticmethod
    def _condition(
        *, reward: bool, format_valid: bool, answer_correct: bool, token: int
    ) -> dict[str, object]:
        return {
            "final_reward": float(reward),
            "format_valid": format_valid,
            "generation_length": 1,
            "prompt_token_count": 8,
            "completion_token_ids": [token],
            "completion_token_ids_sha256": canonical_json_sha256([token]),
            "verifier": {"diagnostic_answer_correct": answer_correct},
        }

    def test_strict_transitions_separate_format_and_answer_content(self) -> None:
        rows = [
            {
                "sample_id": "format-gain",
                "conditions": {
                    "control": self._condition(
                        reward=False, format_valid=False, answer_correct=True, token=1
                    ),
                    "treatment": self._condition(
                        reward=True, format_valid=True, answer_correct=True, token=2
                    ),
                },
            },
            {
                "sample_id": "answer-gain",
                "conditions": {
                    "control": self._condition(
                        reward=False, format_valid=True, answer_correct=False, token=3
                    ),
                    "treatment": self._condition(
                        reward=True, format_valid=True, answer_correct=True, token=4
                    ),
                },
            },
            {
                "sample_id": "format-loss",
                "conditions": {
                    "control": self._condition(
                        reward=True, format_valid=True, answer_correct=True, token=5
                    ),
                    "treatment": self._condition(
                        reward=False, format_valid=False, answer_correct=True, token=6
                    ),
                },
            },
            {
                "sample_id": "answer-loss",
                "conditions": {
                    "control": self._condition(
                        reward=True, format_valid=True, answer_correct=True, token=7
                    ),
                    "treatment": self._condition(
                        reward=False, format_valid=True, answer_correct=False, token=8
                    ),
                },
            },
        ]
        transitions = strict_accuracy_transition_diagnostics(
            rows, treatment="treatment", control="control"
        )
        self.assertEqual(transitions["format_only_gain_count"], 1)
        self.assertEqual(transitions["diagnostic_answer_gain_count"], 1)
        self.assertEqual(transitions["format_only_loss_count"], 1)
        self.assertEqual(transitions["diagnostic_answer_loss_count"], 1)
        differences = completion_difference_summary(
            rows, treatment="treatment", control="control"
        )
        self.assertEqual(differences["different_completion_count"], 4)
        summary = summarize_conditions(rows, ("control", "treatment"))
        self.assertEqual(summary["control"]["diagnostic_answer_accuracy"], 0.75)
        self.assertEqual(summary["treatment"]["diagnostic_answer_accuracy"], 0.75)
        diagnostic_builder = PairedConditionDiagnostics(
            rows, bootstrap_resamples=100
        )
        paired = diagnostic_builder.summarize(
            (
                PairedConditionComparison(
                    "treatment_vs_control", "treatment", "control"
                ),
            )
        )
        self.assertEqual(
            paired["strict_accuracy_transition_diagnostics"][
                "treatment_vs_control"
            ]["format_only_gain_count"],
            1,
        )

    def test_format_transfer_requires_a_positive_text_control(self) -> None:
        observed = format_transfer_diagnostic(
            text_effect={"mean_treatment_minus_control": 0.2},
            side_kv_effect={"mean_treatment_minus_control": 0.1},
        )
        self.assertEqual(observed["status"], "observed")
        self.assertTrue(observed["positive_direction_transferred"])
        unavailable = format_transfer_diagnostic(
            text_effect={"mean_treatment_minus_control": 0.0},
            side_kv_effect={"mean_treatment_minus_control": 0.1},
        )
        self.assertEqual(unavailable["status"], "no_positive_text_control")
        self.assertIsNone(unavailable["positive_direction_transferred"])

    def test_e1cs_decision_separates_channel_capacity_from_task_claim(self) -> None:
        supported = E1CSFixedStrengthDecision(
            mechanism_integrity_passed=True,
            matched_attention_mass_in_target_band=True,
            shuffled_attention_mass_in_target_band=True,
            matched_format_positive_control_transferred=True,
            shuffled_format_positive_control_transferred=True,
            matched_significant_answer_harm=False,
        )
        self.assertTrue(supported.channel_capacity_supported)
        self.assertEqual(
            supported.next_step,
            "record_channel_capacity_then_redesign_experience_and_retrieval",
        )
        no_transfer = E1CSFixedStrengthDecision(
            mechanism_integrity_passed=True,
            matched_attention_mass_in_target_band=True,
            shuffled_attention_mass_in_target_band=True,
            matched_format_positive_control_transferred=False,
            shuffled_format_positive_control_transferred=False,
            matched_significant_answer_harm=False,
        )
        self.assertFalse(no_transfer.channel_capacity_supported)
        self.assertEqual(
            no_transfer.next_step, "stop_current_layer24_side_kv_channel"
        )

    def test_token_sequence_diagnostic_locates_first_divergence(self) -> None:
        diagnostic = token_sequence_diagnostic(
            (10, 20, 30, 40), (10, 20, 99)
        )
        self.assertFalse(diagnostic["exact_match"])
        self.assertTrue(diagnostic["first_token_match"])
        self.assertEqual(diagnostic["common_prefix_token_count"], 2)
        self.assertEqual(diagnostic["first_divergence_index"], 2)
        parity = token_sequence_diagnostic((1, 2), (1, 2))
        self.assertTrue(parity["exact_match"])
        self.assertIsNone(parity["first_divergence_index"])

    def test_text_source_decision_only_allows_strength_after_payload_control(self) -> None:
        positive = {"bootstrap_95_ci": [0.05, 0.2]}
        inconclusive = {"bootstrap_95_ci": [-0.05, 0.1]}
        negative = {"bootstrap_95_ci": [-0.2, -0.01]}
        format_effects = {
            "wrapped_matched_vs_no_memory": positive,
            "wrapper_only_vs_no_memory": inconclusive,
            "payload_only_matched_vs_no_memory": positive,
            "payload_only_shuffled_vs_no_memory": positive,
        }
        diagnostic_effects = {
            "payload_only_matched_vs_no_memory": negative,
        }
        decision = E1CTTextSourceDecision.from_effects(
            format_effects=format_effects,
            diagnostic_answer_effects=diagnostic_effects,
        )
        self.assertEqual(
            decision.next_step, "e1cs_fixed_log10_memory_odds_test"
        )
        self.assertTrue(decision.matched_payload_significant_answer_harm)
        self.assertTrue(
            decision.to_dict()[
                "payload_positive_control_replicated_under_shuffle"
            ]
        )

        wrapper_only = E1CTTextSourceDecision.from_effects(
            format_effects={
                **format_effects,
                "wrapper_only_vs_no_memory": positive,
                "payload_only_matched_vs_no_memory": inconclusive,
                "payload_only_shuffled_vs_no_memory": inconclusive,
            },
            diagnostic_answer_effects={
                "payload_only_matched_vs_no_memory": inconclusive,
            },
        )
        self.assertEqual(
            wrapper_only.next_step, "align_side_kv_compiler_text_contract"
        )


if __name__ == "__main__":
    unittest.main()
