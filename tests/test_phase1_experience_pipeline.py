from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.phase1 import (
    EXPERIENCE_SCHEMA,
    ROLLOUT_SCHEMA,
    audit_teacher_record,
    build_verified_experiences,
    create_gsm8k_split_manifest,
    summarize_human_review,
)
from scripts.build_teacher_bank import jsonl_examples


def rollout(*, episode_id: str, reward: float, trajectory: str) -> dict:
    outcome = "verified_success" if reward == 1.0 else "verified_failure"
    return {
        "schema_version": ROLLOUT_SCHEMA,
        "episode_id": episode_id,
        "sample_id": "gsm8k-train-0-abc",
        "source": {
            "dataset": "openai/gsm8k",
            "dataset_revision": "fixture",
            "dataset_split": "train",
            "logical_split": "bank-source",
            "source_index": 0,
            "question_sha256": "abc",
            "split_manifest_sha256": "manifest",
        },
        "context": "A fixture question?",
        "trajectory": trajectory,
        "outcome": outcome,
        "reward": reward,
        "verifier": {
            "name": "fixture",
            "reward": reward,
            "feedback": f"fixture {outcome}",
        },
        "student": {"model_name": "fixture", "model_revision": "rev", "frozen": True},
        "rollout_configuration": {"sampling_seed": int(reward), "temperature": 0.8},
    }


def teacher_record(experience: dict) -> dict:
    return {
        "schema_version": "teacher-bank-record-v2",
        "experience_id": experience["experience_id"],
        "reference_evidence": "verified_failure",
        "source_episode_ids": {
            "target": experience["target_episode_id"],
            "reference": experience["reference_episode_id"],
        },
        "provenance_sha256": experience["provenance_sha256"],
        "source": copy.deepcopy(experience["source"]),
        "student": copy.deepcopy(experience["student"]),
        "rollout_configuration": copy.deepcopy(experience["rollout_configuration"]),
        "target_verifier": copy.deepcopy(experience["target_verifier"]),
        "reference_verifier": copy.deepcopy(experience["reference_verifier"]),
        "bank": {
            "target": {
                "situation_signature": "A direct plan remains consistent with the task constraints.",
                "transferable_decision": "Continue the supported calculation without changing goals.",
                "verification_rule": "Check each derived relation against the stated conditions.",
                "applicability_boundary": "Use only while the current plan remains logically supported.",
                "confidence": 0.9,
            },
            "reference": {
                "competing_pattern": "Abandoning a supported path for an unrelated shortcut.",
                "failure_signal": "The new step no longer follows from the prior reasoning.",
                "failure_mechanism": "A goal shift introduces unsupported operations and a wrong result.",
                "non_reuse_boundary": "Do not reuse when new evidence genuinely invalidates the plan.",
                "confidence": 0.9,
            },
            "quality": {
                "target_supported": True,
                "reference_supported": True,
                "target_reference_distinct": True,
                "contains_instance_specific_details": False,
                "issues": [],
            },
        },
    }


class SplitManifestTests(unittest.TestCase):
    def test_splits_are_stable_and_disjoint(self) -> None:
        train = [
            {"question": f"train question {index}", "answer": f"answer {index}"}
            for index in range(10)
        ]
        test = [
            {"question": f"test question {index}", "answer": f"test answer {index}"}
            for index in range(3)
        ]
        first = create_gsm8k_split_manifest(
            train,
            test,
            bank_source_size=6,
            calibration_val_size=2,
            seed=42,
            dataset_revision="fixture",
        )
        second = create_gsm8k_split_manifest(
            train,
            test,
            bank_source_size=6,
            calibration_val_size=2,
            seed=42,
            dataset_revision="fixture",
        )
        self.assertTrue(first["overlap_check"]["passed"])
        self.assertEqual(first["counts"], {
            "bank-source": 6,
            "calibration-val": 2,
            "dev-test": 2,
            "final-test": 3,
        })
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        first_assignments = [
            (item["sample_id"], item["logical_split"]) for item in first["samples"]
        ]
        second_assignments = [
            (item["sample_id"], item["logical_split"]) for item in second["samples"]
        ]
        self.assertEqual(first_assignments, second_assignments)


class VerifiedExperienceTests(unittest.TestCase):
    def test_pairs_only_success_with_failure(self) -> None:
        records = [
            rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
            rollout(episode_id="failure", reward=0.0, trajectory="invalid \\boxed{5}"),
        ]
        experiences, report = build_verified_experiences(records)
        self.assertEqual(len(experiences), 1)
        experience = experiences[0]
        self.assertEqual(experience["schema_version"], EXPERIENCE_SCHEMA)
        self.assertEqual(experience["target_episode_id"], "success")
        self.assertEqual(experience["reference_episode_id"], "failure")
        self.assertEqual(experience["reference_evidence"], "verified_failure")
        self.assertEqual(report["verified_experience_count"], 1)

    def test_rejects_non_bank_source_rollout(self) -> None:
        record = rollout(episode_id="bad-split", reward=0.0, trajectory="wrong")
        record["source"]["logical_split"] = "calibration-val"
        with self.assertRaisesRegex(ValueError, "not from bank-source"):
            build_verified_experiences([record])

    def test_teacher_builder_requires_verifier_provenance(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
                rollout(episode_id="failure", reward=0.0, trajectory="invalid \\boxed{5}"),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiences.jsonl"
            path.write_text(json.dumps(experiences[0]) + "\n", encoding="utf-8")
            loaded = list(jsonl_examples(path, offset=0, limit=1))
            self.assertEqual(loaded[0]["reference_evidence"], "verified_failure")

            invalid = copy.deepcopy(experiences[0])
            invalid["reference_verifier"]["reward"] = 1.0
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero-reward verifier"):
                list(jsonl_examples(path, offset=0, limit=1))


class TeacherAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
                rollout(episode_id="failure", reward=0.0, trajectory="invalid \\boxed{5}"),
            ]
        )
        self.experience = experiences[0]

    def test_approved_record_has_no_reasons(self) -> None:
        self.assertEqual(audit_teacher_record(teacher_record(self.experience), self.experience), [])

    def test_teacher_inferred_reference_is_rejected(self) -> None:
        record = teacher_record(self.experience)
        record["reference_evidence"] = "teacher_inferred"
        reasons = audit_teacher_record(record, self.experience)
        self.assertIn("reference_not_verified_failure", reasons)

    def test_equivalent_or_instance_specific_records_are_rejected(self) -> None:
        record = copy.deepcopy(teacher_record(self.experience))
        record["bank"]["reference"] = copy.deepcopy(record["bank"]["target"])
        record["bank"]["reference"]["competing_pattern"] = "Repeat 42 from the target."
        record["bank"]["quality"]["target_reference_distinct"] = False
        reasons = audit_teacher_record(record, self.experience)
        self.assertIn("instance_specific_literal_detected", reasons)
        self.assertIn("teacher_marks_target_reference_equivalent", reasons)


class HumanReviewTests(unittest.TestCase):
    def test_requires_complete_ninety_percent_agreement(self) -> None:
        records = []
        for index in range(10):
            records.append({
                "experience_id": f"experience-{index}",
                "human_review": {
                    "target_supported": True,
                    "reference_supported": True,
                    "target_reference_distinct": True,
                    "factually_consistent": index != 0,
                },
            })
        result = summarize_human_review(
            records,
            required_sample_size=10,
            required_agreement=0.9,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["agreement"], 0.9)

        records[1]["human_review"]["factually_consistent"] = None
        incomplete = summarize_human_review(
            records,
            required_sample_size=10,
            required_agreement=0.9,
        )
        self.assertFalse(incomplete["passed"])


if __name__ == "__main__":
    unittest.main()
