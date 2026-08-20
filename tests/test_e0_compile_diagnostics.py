from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.memory import (
    MemoryBuildResult,
    MemoryBuildTrace,
    MemorySanitizerConfig,
)
from memgen.experience.retrieval import BM25Config, TextAnalyzerConfig
from scripts.compile_experience_memory_bank import (
    E0ArtifactPaths,
    persist_no_survivor_diagnostics,
)


class E0CompileDiagnosticsTests(unittest.TestCase):
    def test_zero_survivor_build_persists_actionable_failure_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved_bank = root / "approved.jsonl"
            verified_experiences = root / "verified.jsonl"
            approved_bank.write_text("{}\n", encoding="utf-8")
            verified_experiences.write_text("{}\n", encoding="utf-8")
            result = MemoryBuildResult(
                records=(),
                trace=(
                    MemoryBuildTrace(
                        source_index=0,
                        experience_id="experience-1",
                        status="rejected_payload",
                        reasons=("payload_exceeds_token_budget",),
                    ),
                ),
                report={
                    "schema_version": "experience-memory-build-report-v1",
                    "input_approved_count": 1,
                    "input_verified_experience_count": 1,
                    "selected_source_count": 1,
                    "accepted_record_count": 0,
                    "status_counts": {"rejected_payload": 1},
                    "rejection_reason_counts": {
                        "payload_exceeds_token_budget": 1
                    },
                    "token_count": {
                        "min": None,
                        "median": None,
                        "p95": None,
                        "max": None,
                    },
                    "payload_hash_unique_count": 0,
                    "policy": {"max_payload_tokens": 128},
                    "record_set_sha256": "empty-record-set-hash",
                },
            )
            paths = E0ArtifactPaths.create(root / "output")

            persist_no_survivor_diagnostics(
                paths=paths,
                result=result,
                approved_bank=approved_bank,
                verified_experiences=verified_experiences,
                reasoner_name="reasoner",
                reasoner_revision="model-revision",
                tokenizer_revision="tokenizer-revision",
                sanitizer_config=MemorySanitizerConfig(max_payload_tokens=128),
                bm25_config=BM25Config(),
                analyzer_config=TextAnalyzerConfig(),
                layer=24,
                dtype="bfloat16",
                text_only=True,
            )

            audit = json.loads(paths.payload_audit.read_text(encoding="utf-8"))
            report = json.loads(paths.e0_report.read_text(encoding="utf-8"))
            trace = [
                json.loads(line)
                for line in paths.trace.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(audit["status"], "payload_audit_failed_no_records")
            self.assertEqual(audit["accepted_record_count"], 0)
            self.assertEqual(
                audit["failure"]["rejection_reason_counts"],
                {"payload_exceeds_token_budget": 1},
            )
            self.assertEqual(report["status"], "failed_no_runtime_safe_records")
            self.assertFalse(report["formal_e0_passed"])
            self.assertEqual(report["runtime_audit"]["blocked_by"], "no_runtime_safe_records")
            self.assertIsNone(report["artifacts"]["memory_records"])
            self.assertFalse(paths.records.exists())
            self.assertFalse(paths.bm25_index.exists())
            self.assertEqual(trace[0]["reasons"], ["payload_exceeds_token_budget"])


if __name__ == "__main__":
    unittest.main()
