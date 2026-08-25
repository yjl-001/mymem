from __future__ import annotations

import unittest

from memgen.experience.e1 import (
    E1_CONDITIONS,
    E1Assignment,
    E1EvaluationScope,
    GateObservation,
    MemoryChoice,
    paired_binary_effect,
)
from memgen.experience.phase1 import canonical_json_sha256


def memory_choice(memory_id: str, slots: int) -> MemoryChoice:
    return MemoryChoice(
        memory_id=memory_id,
        payload_hash=f"payload-{memory_id}",
        token_count=slots,
        kv_valid_slot_count=slots,
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def assignment(sample_id: str, choice: MemoryChoice | None) -> E1Assignment:
    completion = (31, 32)
    prompt = (11, 12)
    prefix = prompt + (31,)
    return E1Assignment(
        sample_id=sample_id,
        logical_split="dev-test",
        dataset_split="train",
        source_index=int(sample_id.rsplit("-", 1)[-1]),
        question_sha256=f"question-{sample_id}",
        prompt_token_count=len(prompt),
        prompt_token_ids_sha256=canonical_json_sha256(list(prompt)),
        observation_completion_token_ids=completion,
        observation_completion_token_ids_sha256=canonical_json_sha256(
            list(completion)
        ),
        gate_observation=GateObservation(
            generated_boundary_index=0,
            boundary_token_id=31,
            entropy=2.0,
            entropy_threshold=1.5,
            persistence_risk_score=0.2,
            persistence_risk_threshold=0.0,
        ),
        prefix_token_ids=prefix,
        prefix_token_ids_sha256=canonical_json_sha256(list(prefix)),
        retrieval_query={"query_hash": f"query-{sample_id}"},
        matched_memory=choice,
        abstain_reason=None if choice is not None else "no_hit",
    )


def final_test_assignment(sample_id: str) -> E1Assignment:
    value = assignment(sample_id, memory_choice("final", 100)).to_dict()
    value["logical_split"] = "final-test"
    value["dataset_split"] = "test"
    return E1Assignment.from_dict(value)


class E1AssignmentTests(unittest.TestCase):
    def test_final_test_scope_maps_to_official_test(self) -> None:
        scope = E1EvaluationScope.from_logical_split("final-test")
        self.assertEqual(scope.dataset_split, "test")
        self.assertEqual(scope.evaluation_role, "final_evaluation")

    def test_round_trip_preserves_frozen_assignment(self) -> None:
        original = assignment("sample-0", memory_choice("a", 100))
        restored = E1Assignment.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(
            E1_CONDITIONS,
            (
                "vanilla",
                "matched",
            ),
        )

    def test_rejects_a_treatment_prefix_with_the_wrong_boundary(self) -> None:
        value = assignment("sample-0", memory_choice("a", 100)).to_dict()
        value["prefix_token_ids"][-1] = 99
        value["prefix_token_ids_sha256"] = canonical_json_sha256(
            value["prefix_token_ids"]
        )
        with self.assertRaisesRegex(ValueError, "boundary token"):
            E1Assignment.from_dict(value)

    def test_final_test_requires_the_official_test_dataset_split(self) -> None:
        self.assertEqual(final_test_assignment("sample-0").dataset_split, "test")
        value = final_test_assignment("sample-0").to_dict()
        value["dataset_split"] = "train"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            E1Assignment.from_dict(value)


class PairedEffectTests(unittest.TestCase):
    def test_reports_discordant_pairs_and_bootstrap_interval(self) -> None:
        treatment = {"a": 1, "b": 1, "c": 1, "d": 0}
        control = {"a": 0, "b": 0, "c": 1, "d": 0}
        report = paired_binary_effect(
            treatment,
            control,
            seed=42,
            resamples=200,
        )
        self.assertEqual(report["paired_sample_count"], 4)
        self.assertEqual(report["treatment_correct_control_wrong"], 2)
        self.assertEqual(report["treatment_wrong_control_correct"], 0)
        self.assertEqual(report["mean_treatment_minus_control"], 0.5)
        self.assertIsInstance(report["bootstrap_95_ci"], list)


if __name__ == "__main__":
    unittest.main()
