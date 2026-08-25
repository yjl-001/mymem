from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from scripts.audit_side_kv_mechanism import validate_case, validate_compile_report


class CompileArtifactValidationTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        artifact_names = {
            "memory_records": "memory_records.v2.jsonl",
            "compilation_trace": "memory_compilation_trace.jsonl",
            "payload_audit": "payload_audit_report.json",
            "bm25_index": "bm25_index.v1.json",
        }
        artifact_entries = {}
        for name, filename in artifact_names.items():
            path = root / filename
            path.write_text(f"{name}\n", encoding="utf-8")
            artifact_entries[name] = {
                "path": filename,
                "sha256": file_sha256(path),
            }
        tensor_path = root / "side_kv_bank.safetensors"
        manifest_path = root / "side_kv_manifest.json"
        tensor_path.write_bytes(b"tensor-fixture")
        manifest_path.write_text("{}\n", encoding="utf-8")
        side_kv = {
            "tensor_path": tensor_path.name,
            "tensor_sha256": file_sha256(tensor_path),
            "manifest_path": manifest_path.name,
            "manifest_sha256": file_sha256(manifest_path),
            "logical_manifest_sha256": "logical-fixture",
        }
        artifact_entries["side_kv"] = side_kv
        report = {
            "schema_version": "experience-memory-e0-report-v3",
            "status": "kv_compilation_passed_pending_runtime_audit",
            "configuration": {"attention_implementation": "sdpa"},
            "artifacts": artifact_entries,
            "artifact_set_sha256": canonical_json_sha256(
                {
                    "records": artifact_entries["memory_records"]["sha256"],
                    "trace": artifact_entries["compilation_trace"]["sha256"],
                    "audit": artifact_entries["payload_audit"]["sha256"],
                    "index": artifact_entries["bm25_index"]["sha256"],
                    "side_kv": side_kv,
                }
            ),
        }
        report_path = root / "e0_report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, manifest_path, root / artifact_names["memory_records"]

    def test_validates_the_complete_compile_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path, manifest_path, _ = self.build_fixture(root)
            report, verified = validate_compile_report(
                report_path=report_path,
                side_kv_manifest_path=manifest_path,
            )
            self.assertEqual(
                report["status"],
                "kv_compilation_passed_pending_runtime_audit",
            )
            self.assertEqual(len(verified), 6)

    def test_rejects_a_post_compile_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path, manifest_path, records_path = self.build_fixture(root)
            records_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed hash validation"):
                validate_compile_report(
                    report_path=report_path,
                    side_kv_manifest_path=manifest_path,
                )


class AuditCaseValidationTests(unittest.TestCase):
    def valid_case(self) -> dict:
        prefix = [10, 11, 12]
        return {
            "schema_version": "side-kv-mechanism-audit-case-input-v2",
            "case_id": "calibration-case",
            "memory_id": "memory-id",
            "logical_split": "calibration-val",
            "answer_or_reward_used": False,
            "selection_policy": "first_preanswer_reasoning_delimiter",
            "question_sha256": "question-hash",
            "prompt_token_count": 2,
            "generated_boundary_index": 0,
            "boundary_token_id": 12,
            "prefix_token_ids": prefix,
            "prefix_token_ids_sha256": canonical_json_sha256(prefix),
        }

    def test_accepts_an_answer_blind_calibration_case(self) -> None:
        case = self.valid_case()
        self.assertEqual(
            validate_case(case),
            ("calibration-case", "memory-id", [10, 11, 12], 2),
        )

    def test_rejects_a_case_that_used_answer_or_reward(self) -> None:
        case = self.valid_case()
        case["answer_or_reward_used"] = True
        with self.assertRaisesRegex(ValueError, "not marked answer-blind"):
            validate_case(case)


if __name__ == "__main__":
    unittest.main()
