from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.input_lineage import (
    PHASE1_RISK_LINEAGE_SCHEMA,
    build_phase1_risk_lineage,
    validate_sealed_lineage,
)
from memgen.experience.phase1 import (
    canonical_json_sha256,
    create_gsm8k_split_manifest,
    file_sha256,
    write_jsonl,
)
from memgen.experience.risk import TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class Phase1RiskLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "lineage"
        self.phase1 = self.root / "phase1"
        self.risk = self.root / "risk_v3_4"
        self.phase1.mkdir(parents=True)
        self.risk.mkdir(parents=True)

        split = create_gsm8k_split_manifest(
            [
                {"question": "first?", "answer": "one"},
                {"question": "second?", "answer": "two"},
                {"question": "third?", "answer": "three"},
            ],
            [{"question": "test?", "answer": "four"}],
            bank_source_size=2,
            calibration_val_size=1,
            seed=42,
            dataset_revision="dataset-commit",
            train_fingerprint="train-fingerprint",
            test_fingerprint="test-fingerprint",
        )
        write_json(self.phase1 / "split_manifest.json", split)
        write_jsonl(self.phase1 / "student_rollouts.jsonl", ({"row": 1},))
        rollouts_sha = file_sha256(self.phase1 / "student_rollouts.jsonl")
        student = {
            "model_name": "reasoner",
            "model_revision": "model-commit",
            "tokenizer_revision": "tokenizer-commit",
        }
        write_json(
            self.phase1 / "rollout_summary.json",
            {
                "output_sha256": rollouts_sha,
                "split_manifest_sha256": split["manifest_sha256"],
                "student": student,
                "rollout_configuration": {"base_seed": 42},
            },
        )
        experience = {
            "experience_id": "experience-one",
            "sample_id": "sample-one",
            "source": {"logical_split": "bank-source"},
            "student": student,
            "experience_type": "answer_correctness",
            "target_episode_id": "target-one",
            "reference_episode_id": "reference-one",
            "reference_evidence": "verified_failure",
            "reference_verifier": {"reward": 0.0},
            "trajectory": "successful reasoning",
            "reference_trajectory": "failed reasoning",
            "provenance_sha256": "provenance",
        }
        write_jsonl(self.phase1 / "verified_experiences.jsonl", (experience,))
        experiences_sha = file_sha256(self.phase1 / "verified_experiences.jsonl")
        write_json(
            self.phase1 / "experience_build_report.json",
            {
                "rollouts_sha256": rollouts_sha,
                "experiences_sha256": experiences_sha,
            },
        )
        write_jsonl(self.phase1 / "teacher_reflections.jsonl", ({"row": 1},))
        write_jsonl(self.phase1 / "ai_review_records.jsonl", ({"row": 1},))
        approved = {
            "experience_id": experience["experience_id"],
            "source": experience["source"],
            "student": student,
            "experience_type": "answer_correctness",
            "provenance_sha256": experience["provenance_sha256"],
            "reference_evidence": "verified_failure",
            "source_episode_ids": {
                "target": experience["target_episode_id"],
                "reference": experience["reference_episode_id"],
            },
            "ai_review_gate": {"route": "ai_approved"},
        }
        write_jsonl(self.phase1 / "ai_approved_bank_records.jsonl", (approved,))
        for name in (
            "ai_rejected_bank_records.jsonl",
            "deferred_bank_records.jsonl",
            "quarantined_bank_records.jsonl",
        ):
            write_jsonl(self.phase1 / name, ())
        review_hashes = {
            "experiences_sha256": experiences_sha,
            "teacher_records_sha256": file_sha256(
                self.phase1 / "teacher_reflections.jsonl"
            ),
            "review_records_sha256": file_sha256(
                self.phase1 / "ai_review_records.jsonl"
            ),
            "approved_sha256": file_sha256(
                self.phase1 / "ai_approved_bank_records.jsonl"
            ),
            "rejected_sha256": file_sha256(
                self.phase1 / "ai_rejected_bank_records.jsonl"
            ),
            "deferred_sha256": file_sha256(
                self.phase1 / "deferred_bank_records.jsonl"
            ),
            "quarantined_sha256": file_sha256(
                self.phase1 / "quarantined_bank_records.jsonl"
            ),
        }
        write_json(self.phase1 / "ai_review_report.json", {"artifacts": review_hashes})

        (self.risk / "token-entropy-risk-gate-v3.4.pt").write_bytes(b"risk")
        write_jsonl(self.risk / "token_entropy_risk_evidence.jsonl", ({"row": 1},))
        approved_sha = review_hashes["approved_sha256"]
        inputs = {
            "approved_bank_sha256": approved_sha,
            "verified_experiences_sha256": experiences_sha,
        }
        self.risk_artifact = {
            "schema_version": TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA,
            "artifact_id": "risk-fixture",
            "status": "passed",
            "qualification": {"passed": True},
            "reasoner": {**student, "attention_implementation": "sdpa"},
            "construction": {"fixed_layer": 24},
            "event_counts": {"train": {}, "holdout": {}},
            "inputs": inputs,
        }
        write_json(
            self.risk / "token_entropy_risk_report.json",
            {
                "status": "passed",
                "qualification": {"passed": True},
                "inputs": inputs,
                "artifact": {
                    "sha256": file_sha256(
                        self.risk / "token-entropy-risk-gate-v3.4.pt"
                    )
                },
                "evidence_trace": {
                    "sha256": file_sha256(
                        self.risk / "token_entropy_risk_evidence.jsonl"
                    )
                },
            },
        )

    def build(self, **kwargs):
        return build_phase1_risk_lineage(
            lineage_id="gsm8k-v4-rebuild-one",
            lineage_root=self.root,
            phase1_dir=self.phase1,
            risk_dir=self.risk,
            risk_artifact=self.risk_artifact,
            repository_revision="revision",
            **kwargs,
        )

    def test_builds_sealed_exact_phase1_risk_lineage(self) -> None:
        manifest = self.build()
        self.assertEqual(manifest["schema_version"], PHASE1_RISK_LINEAGE_SCHEMA)
        self.assertEqual(manifest["status"], "sealed_phase1_risk_lineage")
        self.assertTrue(manifest["authentication"]["risk_bound_to_exact_phase1_files"])
        self.assertEqual(manifest["downstream_v4"]["status"], "not_checked")
        self.assertEqual(
            manifest["manifest_sha256"],
            canonical_json_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            ),
        )
        path = self.root / "phase1_risk_lineage_manifest.json"
        write_json(path, manifest)
        validate_sealed_lineage(manifest, path=path)

    def test_rejects_risk_artifact_bound_to_other_phase1(self) -> None:
        self.risk_artifact["inputs"]["verified_experiences_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "bound to other inputs"):
            self.build()

    def test_reports_old_v4_archive_as_incompatible(self) -> None:
        bank = {
            "inputs": {
                "experiences_sha256": "old-experiences",
                "split_manifest_sha256": "old-split",
            }
        }
        bank["manifest_sha256"] = canonical_json_sha256(bank)
        bank_path = self.root / "old_bank_manifest.json"
        write_json(bank_path, bank)
        side = {
            "source": {
                "bank_manifest_file_sha256": file_sha256(bank_path),
                "bank_manifest_logical_sha256": bank["manifest_sha256"],
            }
        }
        side["manifest_sha256"] = canonical_json_sha256(side)
        side_path = self.root / "old_side_kv_manifest.json"
        write_json(side_path, side)
        manifest = self.build(
            bank_manifest_path=bank_path,
            side_kv_manifest_path=side_path,
        )
        compatibility = manifest["downstream_v4"]
        self.assertFalse(compatibility["compatible"])
        self.assertEqual(
            compatibility["status"],
            "incompatible_rebuild_or_original_data_recovery_required",
        )
        self.assertFalse(compatibility["checks"]["bank_experiences_sha256"])

    def test_sealed_lineage_detects_artifact_drift(self) -> None:
        manifest = self.build()
        path = self.root / "phase1_risk_lineage_manifest.json"
        write_json(path, manifest)
        with (self.phase1 / "verified_experiences.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "artifact drifted"):
            validate_sealed_lineage(manifest, path=path)


class Phase1RiskRunnerContractTests(unittest.TestCase):
    def test_runner_uses_named_lineage_and_requires_paid_opt_in(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root / "scripts/experiments/gsm8k/run_phase1_risk_lineage.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", runner)
        self.assertIn("--allow-paid-phase1", runner)
        self.assertIn("lineages/gsm8k/$LINEAGE_ID", runner)
        self.assertIn("USE_THIS_LINEAGE.env", runner)
        self.assertNotIn("evaluate_v4_experience_memory.py", runner)
        self.assertNotIn("run_v4_source_oracle_audit.sh", runner)


if __name__ == "__main__":
    unittest.main()
