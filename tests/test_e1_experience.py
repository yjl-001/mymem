from __future__ import annotations

from collections import Counter
import unittest

from memgen.experience.e1 import (
    E1Assignment,
    GateObservation,
    MatchedMemoryDeranger,
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
        shuffled_memory=None,
        abstain_reason=None if choice is not None else "no_hit",
    )


class E1AssignmentTests(unittest.TestCase):
    def test_round_trip_preserves_frozen_assignment(self) -> None:
        original = assignment("sample-0", memory_choice("a", 100))
        restored = E1Assignment.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_rejects_a_treatment_prefix_with_the_wrong_boundary(self) -> None:
        value = assignment("sample-0", memory_choice("a", 100)).to_dict()
        value["prefix_token_ids"][-1] = 99
        value["prefix_token_ids_sha256"] = canonical_json_sha256(
            value["prefix_token_ids"]
        )
        with self.assertRaisesRegex(ValueError, "boundary token"):
            E1Assignment.from_dict(value)


class MatchedMemoryDerangerTests(unittest.TestCase):
    def test_preserves_the_exact_memory_multiset_without_self_assignment(self) -> None:
        inputs = (
            assignment("sample-0", memory_choice("a", 100)),
            assignment("sample-1", memory_choice("a", 100)),
            assignment("sample-2", memory_choice("b", 120)),
            assignment("sample-3", memory_choice("b", 120)),
            assignment("sample-4", memory_choice("c", 140)),
            assignment("sample-5", memory_choice("c", 140)),
        )
        output, report = MatchedMemoryDeranger(seed=42).assign(inputs)
        matched = Counter(item.matched_memory.memory_id for item in output)
        shuffled = Counter(item.shuffled_memory.memory_id for item in output)
        self.assertEqual(matched, shuffled)
        self.assertTrue(
            all(
                item.matched_memory.memory_id != item.shuffled_memory.memory_id
                for item in output
            )
        )
        self.assertEqual(report["assigned_count"], len(inputs))
        self.assertEqual(
            report["matched_memory_multiset_sha256"],
            report["shuffled_memory_multiset_sha256"],
        )
        self.assertTrue(
            all(item.shuffled_memory.retrieval_score is None for item in output)
        )

    def test_fails_when_one_retrieval_id_cannot_be_deranged(self) -> None:
        inputs = (
            assignment("sample-0", memory_choice("a", 100)),
            assignment("sample-1", memory_choice("a", 100)),
            assignment("sample-2", memory_choice("a", 100)),
            assignment("sample-3", memory_choice("b", 120)),
        )
        with self.assertRaisesRegex(ValueError, "too concentrated"):
            MatchedMemoryDeranger().assign(inputs)


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
