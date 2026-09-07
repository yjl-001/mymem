from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, text_sha256
from memgen.experience.v4_2_semantic_bank import V4_2_EVIDENCE_PACKET_SCHEMA
from memgen.experience.v4_question_recovery import (
    V4_QUESTION_RECOVERY_SCHEMA,
    V4_RECOVERED_EXPERIENCE_SCHEMA,
    build_question_recovery_manifest,
    build_recovered_source_records,
    load_recovered_source_experiences,
    recovered_experience_selection,
    validate_question_recovery_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict] | tuple[dict, ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class V4QuestionRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = [
            {
                "question": f"Fixture question {index}?",
                "answer": f"A short rationale.\n#### {index + 1}",
            }
            for index in range(5)
        ]
        self.sample_ids = [
            f"gsm8k-train-{index}-{text_sha256(row['question'])[:12]}"
            for index, row in enumerate(self.dataset)
        ]
        self.experience_ids = [f"experience-{index}" for index in range(5)]
        samples = [
            {
                "sample_id": sample_id,
                "logical_split": "bank-source",
                "dataset_split": "train",
                "source_index": index,
                "question_sha256": text_sha256(self.dataset[index]["question"]),
                "answer_sha256": text_sha256(self.dataset[index]["answer"]),
            }
            for index, sample_id in enumerate(self.sample_ids)
        ]
        self.split = {
            "schema_version": "gsm8k-split-manifest-v1",
            "samples": samples,
        }
        self.split["manifest_sha256"] = canonical_json_sha256(self.split)

        evidence = []
        for index, (experience_id, sample_id) in enumerate(
            zip(self.experience_ids, self.sample_ids)
        ):
            evidence.append(
                {
                    "evidence_id": experience_id,
                    "sample_id": sample_id,
                    "source_experience_type": "answer_correctness",
                    "semantic_signature": {
                        "problem_structure": "fixture",
                        "decision_point": "compute",
                        "failure_mechanism": "wrong arithmetic",
                        "repair_operator": "recompute",
                        "verification_operator": "substitute",
                    },
                    "question": self.dataset[index]["question"],
                    "official_solution": self.dataset[index]["answer"],
                    "verified_success_trajectory": f"Work. \\boxed{{{index + 1}}}",
                    "verified_failure_trajectory": "Wrong. \\boxed{999}",
                    "target_verifier": {"reward": 1.0},
                    "reference_verifier": {
                        "reward": 0.0,
                        "failure_types": ["boxed_answer_mismatch"],
                    },
                    "source_provenance_sha256": f"source-{index}",
                    "source_signature_sha256": f"signature-{index}",
                    "construction_input_sha256": f"construction-{index}",
                }
            )
        packet = {
            "schema_version": V4_2_EVIDENCE_PACKET_SCHEMA,
            "candidate_id": "candidate-a",
            "evidence_count": len(evidence),
            "evidence": evidence,
        }
        packet["packet_sha256"] = canonical_json_sha256(packet)
        self.packet_path = self.root / "semantic_evidence_packets.jsonl"
        write_jsonl(self.packet_path, [packet])

        bank_record = {
            "bank_id": "bank-a",
            "construction": {
                "experience_ids": self.experience_ids,
                "sample_ids": self.sample_ids,
            },
        }
        bank_record["record_sha256"] = canonical_json_sha256(bank_record)
        self.bank_records_path = self.root / "bank_records.jsonl"
        write_jsonl(self.bank_records_path, [bank_record])
        bank_manifest = {
            "record_count": 1,
            "evidence_count": 5,
            "bank_ids": ["bank-a"],
            "record_sha256": {"bank-a": bank_record["record_sha256"]},
            "inputs": {
                "semantic_preflight": {
                    "evidence_packet_file_sha256": file_sha256(self.packet_path)
                }
            },
        }
        bank_manifest["manifest_sha256"] = canonical_json_sha256(bank_manifest)
        self.bank_manifest_path = self.root / "bank_manifest.json"
        write_json(self.bank_manifest_path, bank_manifest)

        side_kv = {
            "created_at": "fixture-time",
            "bank_count": 1,
            "records": [
                {"bank_id": "bank-a", "role": "target"},
                {"bank_id": "bank-a", "role": "reference"},
            ],
            "source": {
                "bank_manifest_logical_sha256": bank_manifest["manifest_sha256"]
            },
            "reasoner": {
                "model_name": "Qwen/fixture",
                "model_revision": "model-revision",
                "tokenizer_revision": "tokenizer-revision",
            },
        }
        side_kv["manifest_sha256"] = canonical_json_sha256(side_kv)
        self.side_kv_path = self.root / "v4_side_kv_manifest.json"
        write_json(self.side_kv_path, side_kv)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_lineage(self) -> tuple[Path, tuple[dict, ...]]:
        rows, summary = build_recovered_source_records(
            semantic_packets_path=self.packet_path,
            bank_records_path=self.bank_records_path,
            bank_manifest_path=self.bank_manifest_path,
            side_kv_manifest_path=self.side_kv_path,
            split_manifest=self.split,
            train_records=self.dataset,
            expected_bank_count=1,
            expected_evidence_count=5,
        )
        split_path = self.root / "split_manifest.json"
        experiences_path = self.root / "recovered_source_experiences.jsonl"
        write_json(split_path, self.split)
        write_jsonl(experiences_path, rows)
        manifest = build_question_recovery_manifest(
            recovery_id="fixture-recovery",
            semantic_packets_path=self.packet_path,
            bank_records_path=self.bank_records_path,
            bank_manifest_path=self.bank_manifest_path,
            side_kv_manifest_path=self.side_kv_path,
            split_manifest_path=split_path,
            recovered_experiences_path=experiences_path,
            summary=summary,
        )
        manifest_path = self.root / "v4_question_recovery_manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path, rows

    def test_recovers_exact_packet_trajectories_without_phase1_claim(self) -> None:
        manifest_path, rows = self.build_lineage()
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["schema_version"], V4_RECOVERED_EXPERIENCE_SCHEMA)
        self.assertEqual(rows[0]["trajectory"], "Work. \\boxed{1}")
        self.assertEqual(rows[0]["reference_trajectory"], "Wrong. \\boxed{999}")
        self.assertEqual(rows[0]["replayed_target_verifier"]["reward"], 1.0)
        self.assertEqual(rows[0]["replayed_reference_verifier"]["reward"], 0.0)
        self.assertFalse(
            rows[0]["recovery_provenance"]["original_phase1_file_recreated"]
        )

        manifest = validate_question_recovery_manifest(manifest_path)
        self.assertEqual(manifest["schema_version"], V4_QUESTION_RECOVERY_SCHEMA)
        self.assertFalse(manifest["claims"]["original_phase1_file_recovery_claim"])
        self.assertFalse(manifest["claims"]["original_risk_artifact_recovery_claim"])
        self.assertTrue(manifest["claims"]["same_source_question"])
        self.assertTrue(manifest["claims"]["same_source_failure_trajectory"])
        self.assertEqual(len(load_recovered_source_experiences(manifest_path)), 5)

    def test_curated_membership_must_be_present_in_packet(self) -> None:
        packet = json.loads(self.packet_path.read_text(encoding="utf-8").splitlines()[0])
        packet["evidence"] = packet["evidence"][:-1]
        packet["evidence_count"] = len(packet["evidence"])
        packet.pop("packet_sha256")
        packet["packet_sha256"] = canonical_json_sha256(packet)
        write_jsonl(self.packet_path, [packet])
        bank_manifest = json.loads(self.bank_manifest_path.read_text(encoding="utf-8"))
        bank_manifest["inputs"]["semantic_preflight"][
            "evidence_packet_file_sha256"
        ] = file_sha256(self.packet_path)
        bank_manifest.pop("manifest_sha256")
        bank_manifest["manifest_sha256"] = canonical_json_sha256(bank_manifest)
        write_json(self.bank_manifest_path, bank_manifest)
        side_kv = json.loads(self.side_kv_path.read_text(encoding="utf-8"))
        side_kv["source"]["bank_manifest_logical_sha256"] = bank_manifest[
            "manifest_sha256"
        ]
        side_kv.pop("manifest_sha256")
        side_kv["manifest_sha256"] = canonical_json_sha256(side_kv)
        write_json(self.side_kv_path, side_kv)
        with self.assertRaisesRegex(ValueError, "lost curated evidence"):
            build_recovered_source_records(
                semantic_packets_path=self.packet_path,
                bank_records_path=self.bank_records_path,
                bank_manifest_path=self.bank_manifest_path,
                side_kv_manifest_path=self.side_kv_path,
                split_manifest=self.split,
                train_records=self.dataset,
                expected_bank_count=1,
                expected_evidence_count=5,
            )

    def test_sealed_lineage_detects_source_packet_drift(self) -> None:
        manifest_path, _rows = self.build_lineage()
        with self.packet_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "file hash mismatch"):
            validate_question_recovery_manifest(manifest_path)

    def test_risk_selection_is_honest_no_ai_route(self) -> None:
        manifest_path, _rows = self.build_lineage()
        selected, report = recovered_experience_selection(
            load_recovered_source_experiences(manifest_path),
            allowed_experience_types=("answer_correctness",),
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            report["source"], "semantic_packet_replay_strict_verifier_no_ai"
        )
        self.assertFalse(report["ai_review_approval_claim"])

    def test_runner_is_zero_paid_api_and_supports_staged_smoke_full(self) -> None:
        runner = ROOT / "scripts/experiments/gsm8k/run_v4_question_recovery.sh"
        subprocess.run(["bash", "-n", str(runner)], check=True)
        source = runner.read_text(encoding="utf-8")
        self.assertIn("--mode smoke|full", source)
        self.assertIn("recover_v4_source_evidence.py", source)
        self.assertIn("--question-recovery-manifest", source)
        self.assertIn("compile_token_entropy_risk_gate.py", source)
        self.assertIn("unset DEEPSEEK_API_KEY GLM_API_KEY", source)
        self.assertNotIn("build_teacher_bank.py", source)
        self.assertNotIn("review_experience_bank.py", source)
        self.assertNotIn("evaluate_v4_experience_memory.py", source)


if __name__ == "__main__":
    unittest.main()
