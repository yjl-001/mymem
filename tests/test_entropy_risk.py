from __future__ import annotations

import copy
import unittest

from memgen.experience.phase1 import ROLLOUT_SCHEMA, build_verified_experiences
from memgen.experience.risk import (
    approved_experiences,
    binary_average_precision,
    binary_balanced_accuracy,
    binary_roc_auc,
    deterministic_train_partition,
    entropy_quantile,
    entropy_transition_label,
    select_balanced_accuracy_threshold,
    select_recovery_horizon,
    stable_low_recovery_offset,
    token_entropy_transition_label,
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
        "student": {
            "model_name": "fixture",
            "model_revision": "rev",
            "frozen": True,
        },
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


class ApprovedExperienceSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        experiences, _ = build_verified_experiences([
            rollout("success", 1.0, "First step. Therefore \\boxed{4}"),
            rollout("failure", 0.0, "Wrong step. Therefore \\boxed{5}"),
        ])
        self.experience = experiences[0]

    def test_only_approved_verified_pairs_are_selected(self) -> None:
        selected, report = approved_experiences(
            [approved_record(self.experience)], [self.experience]
        )
        self.assertEqual(
            [item["experience_id"] for item in selected],
            [self.experience["experience_id"]],
        )
        self.assertEqual(report["selected_count"], 1)

    def test_rejects_mismatched_provenance(self) -> None:
        record = approved_record(self.experience)
        record["provenance_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "mismatched provenance_sha256"):
            approved_experiences([record], [self.experience])


class EntropyRiskContractTests(unittest.TestCase):
    def test_transition_label_uses_current_and_next_entropy(self) -> None:
        self.assertEqual(entropy_quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertEqual(
            entropy_transition_label(
                current_entropy=5.0,
                next_entropy=3.0,
                high_threshold=5.0,
                low_threshold=3.0,
            ),
            "recovery",
        )
        self.assertEqual(
            entropy_transition_label(
                current_entropy=5.0,
                next_entropy=3.5,
                high_threshold=5.0,
                low_threshold=3.0,
            ),
            "persistence",
        )

    def test_partition_and_metrics_are_deterministic(self) -> None:
        assigned = deterministic_train_partition(
            "experience-1", seed=42, train_fraction=0.5
        )
        self.assertEqual(
            assigned,
            deterministic_train_partition(
                "experience-1", seed=42, train_fraction=0.5
            ),
        )
        self.assertEqual(
            binary_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0
        )
        self.assertEqual(
            binary_average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
            1.0,
        )
        self.assertEqual(
            binary_balanced_accuracy([0, 0, 1, 1], [0, 0, 1, 1]),
            1.0,
        )

    def test_token_label_requires_stable_low_run_and_censors_tail(self) -> None:
        entropies = [5.0, 4.0, 2.0, 1.5, 5.5, 4.5]
        self.assertEqual(
            stable_low_recovery_offset(
                entropies,
                current_index=0,
                low_threshold=2.0,
                stable_token_count=2,
            ),
            3,
        )
        self.assertEqual(
            token_entropy_transition_label(
                entropies,
                current_index=0,
                high_threshold=5.0,
                low_threshold=2.0,
                recovery_horizon=4,
            ),
            "recovery",
        )
        self.assertIsNone(
            token_entropy_transition_label(
                entropies,
                current_index=4,
                high_threshold=5.0,
                low_threshold=2.0,
                recovery_horizon=3,
            )
        )
        self.assertEqual(
            token_entropy_transition_label(
                [5.0, 4.0, 4.0, 4.0],
                current_index=0,
                high_threshold=5.0,
                low_threshold=2.0,
                recovery_horizon=3,
            ),
            "persistence",
        )

    def test_risk_threshold_is_calibrated_from_shifted_train_scores(self) -> None:
        result = select_balanced_accuracy_threshold(
            [False, False, True, True],
            [-0.9, -0.8, -0.3, -0.2],
        )
        self.assertEqual(result["threshold"], -0.8)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["predicted_persistence_fraction"], 0.5)

    def test_recovery_horizon_is_train_sequence_derived_and_capped(self) -> None:
        result = select_recovery_horizon(
            [
                [5.0, 1.0, 1.0],
                [5.0, 4.0, 4.0, 1.0, 1.0],
            ],
            high_threshold=5.0,
            low_threshold=1.0,
            stable_token_count=2,
            quantile=0.75,
            maximum_horizon=4,
        )
        self.assertEqual(result["recovery_horizon"], 4)
        self.assertEqual(result["recovered_event_count"], 2)


if __name__ == "__main__":
    unittest.main()
