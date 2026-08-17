from __future__ import annotations

import copy
import unittest

from memgen.experience.phase1 import ROLLOUT_SCHEMA, build_verified_experiences
from memgen.experience.phase2 import (
    approved_experiences,
    last_completion_boundary,
    phase1_mechanism_cluster,
    select_calibration_winner,
    soft_entropy_gate,
    stable_uniform,
    validate_evidence_anchor,
)


def rollout(episode_id: str, reward: float, trajectory: str) -> dict:
    return {
        "schema_version": ROLLOUT_SCHEMA,
        "episode_id": episode_id,
        "sample_id": "gsm8k-train-0-fixture",
        "source": {
            "dataset": "openai/gsm8k",
            "dataset_revision": "fixture",
            "dataset_split": "train",
            "logical_split": "bank-source",
            "source_index": 0,
            "question_sha256": "question",
            "split_manifest_sha256": "manifest",
        },
        "context": "A fixture question?",
        "trajectory": trajectory,
        "outcome": "verified_success" if reward else "verified_failure",
        "reward": reward,
        "verifier": {
            "name": "fixture",
            "reward": reward,
            "expected_answer": "4",
        },
        "student": {"model_name": "fixture", "model_revision": "rev", "frozen": True},
        "rollout_configuration": {"sampling_seed": 42, "temperature": 0.8},
    }


def approved_record(experience: dict) -> dict:
    return {
        "experience_id": experience["experience_id"],
        "ai_review_gate": {"route": "ai_approved"},
        "reference_evidence": "verified_failure",
        "provenance_sha256": experience["provenance_sha256"],
        "source": copy.deepcopy(experience["source"]),
        "student": copy.deepcopy(experience["student"]),
        "experience_type": experience["experience_type"],
        "source_episode_ids": {
            "target": experience["target_episode_id"],
            "reference": experience["reference_episode_id"],
        },
    }


class Phase2SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout("success", 1.0, "First step. Therefore \\boxed{4}"),
                rollout("failure", 0.0, "Wrong step. Therefore \\boxed{5}"),
            ]
        )
        self.experience = experiences[0]

    def test_only_approved_verified_pairs_are_selected(self) -> None:
        selected, report = approved_experiences(
            [approved_record(self.experience)], [self.experience]
        )
        self.assertEqual([item["experience_id"] for item in selected], [self.experience["experience_id"]])
        self.assertEqual(report["selected_count"], 1)

    def test_rejects_mismatched_provenance(self) -> None:
        record = approved_record(self.experience)
        record["provenance_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "mismatched provenance_sha256"):
            approved_experiences([record], [self.experience])

    def test_uses_only_frozen_phase1_bank_text_for_mechanism_bucket(self) -> None:
        record = approved_record(self.experience)
        record["bank"] = {
            "reference": {
                "failure_mechanism": "The arithmetic calculation applies the operation incorrectly.",
                "failure_signal": "The intermediate total is inconsistent.",
            },
            "evidence": {"reference_observation": "An incorrect total is shown."},
        }
        selected, _ = approved_experiences([record], [self.experience])
        self.assertEqual(phase1_mechanism_cluster(selected[0]), "arithmetic_or_numeric")
        self.assertIsNone(phase1_mechanism_cluster(self.experience))


class Phase2BoundaryAndGateTests(unittest.TestCase):
    def test_finds_last_merged_delimiter_token(self) -> None:
        token_text = {1: "prompt", 2: " first.", 3: " second,\n", 4: "final"}
        self.assertEqual(
            last_completion_boundary(
                [1, 2, 3, 4], completion_start=1, decode_token=token_text.__getitem__
            ),
            2,
        )

    def test_soft_gate_and_random_control_are_stable(self) -> None:
        self.assertAlmostEqual(soft_entropy_gate(5.0, 5.0, 0.1), 0.5)
        self.assertGreater(soft_entropy_gate(6.0, 5.0, 0.1), 0.99)
        self.assertEqual(stable_uniform(42, "sample", "1"), stable_uniform(42, "sample", "1"))
        self.assertNotEqual(stable_uniform(42, "sample", "1"), stable_uniform(42, "sample", "2"))

    def test_anchor_accepts_exact_equation_spans_but_rejects_final_box(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout("success", 1.0, "Compute 2 + 2 = 4. Therefore \\boxed{4}"),
                rollout("failure", 0.0, "Compute 2 + 2 = 5. Therefore \\boxed{5}"),
            ]
        )
        experience = experiences[0]
        payload = {
            "decision": "anchor",
            "mechanism_cluster": "arithmetic_or_numeric",
            "target_anchor": {"quote": "2 + 2 = 4"},
            "reference_anchor": {"quote": "2 + 2 = 5"},
        }
        self.assertEqual(validate_evidence_anchor(payload, experience), [])
        payload["target_anchor"]["quote"] = "\\boxed{4}"
        self.assertIn(
            "target_quote_is_final_answer_formatting",
            validate_evidence_anchor(payload, experience),
        )


class Phase2CalibrationTests(unittest.TestCase):
    def test_selects_safe_format_preserving_winner(self) -> None:
        winner = select_calibration_winner(
            [
                {
                    "condition": "real_vector",
                    "config": {"layer": 8},
                    "accuracy": 0.6,
                    "format_accuracy": 0.9,
                    "vanilla_format_accuracy": 0.9,
                    "mean_injections": 1.0,
                    "safety_failed": False,
                },
                {
                    "condition": "real_vector",
                    "config": {"layer": 16},
                    "accuracy": 0.8,
                    "format_accuracy": 0.8,
                    "vanilla_format_accuracy": 0.9,
                    "mean_injections": 0.5,
                    "safety_failed": False,
                },
                {
                    "condition": "real_vector",
                    "config": {"layer": 24},
                    "accuracy": 0.9,
                    "format_accuracy": 0.9,
                    "vanilla_format_accuracy": 0.9,
                    "mean_injections": 1.0,
                    "safety_failed": True,
                },
            ]
        )
        self.assertEqual(winner["config"]["layer"], 8)


if __name__ == "__main__":
    unittest.main()
