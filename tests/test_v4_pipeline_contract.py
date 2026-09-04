from __future__ import annotations

from pathlib import Path
import unittest

import scripts.compile_v4_selector_anchors as anchor_compiler


ROOT = Path(__file__).resolve().parents[1]


class V4PipelineContractTests(unittest.TestCase):
    def test_progress_alignment_is_endpoint_preserving(self) -> None:
        self.assertEqual(
            anchor_compiler._normalized_progress_rank(
                0, source_count=5, target_count=9
            ),
            0,
        )
        self.assertEqual(
            anchor_compiler._normalized_progress_rank(
                4, source_count=5, target_count=9
            ),
            8,
        )
        self.assertEqual(
            anchor_compiler._normalized_progress_rank(
                2, source_count=5, target_count=9
            ),
            4,
        )

    def test_threshold_calibration_abstains_on_nonapplicable_states(self) -> None:
        rows = [
            {
                "query_kind": "failure",
                "top1_score": 1.0,
                "margin": 0.5,
                "correct_top1": True,
                "source_bank_id": "bank-a",
            }
            for _ in range(10)
        ]
        rows.extend(
            {
                "query_kind": "success",
                "top1_score": 0.1,
                "margin": 0.01,
                "correct_top1": None,
                "source_bank_id": "bank-a",
            }
            for _ in range(20)
        )
        absolute, margin, report = anchor_compiler._calibrate_thresholds(rows)
        self.assertEqual(absolute, 0.1)
        self.assertEqual(margin, 0.5)
        self.assertEqual(report["selected"]["failure_correct_coverage"], 1.0)
        self.assertEqual(report["selected"]["success_false_rate"], 0.0)
        self.assertTrue(report["qualified"])

    def test_side_kv_compiles_reference_but_exposes_target_only(self) -> None:
        source = (ROOT / "memgen/model/v4_side_kv.py").read_text(encoding="utf-8")
        self.assertIn("V4_1_BANK_MANIFEST_SCHEMA", source)
        self.assertIn("V41ConstructionProfile", source)
        self.assertIn("V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA", source)
        self.assertIn("V42LocalDirectProfile", source)
        self.assertIn("local_direct_implementation_hashes", source)
        self.assertIn('role="target"', source)
        self.assertIn('role="reference"', source)
        self.assertIn("def get_target(", source)
        self.assertNotIn("def get_reference(", source)
        for variant in ("raw_descriptor", "internal_principle", "hidden_note"):
            self.assertIn(variant, source)

    def test_online_runtime_cannot_request_reference_memory(self) -> None:
        source = (ROOT / "memgen/model/v4_online.py").read_text(encoding="utf-8")
        self.assertIn("self.loader.get_target(", source)
        self.assertNotIn("get_reference(", source)
        self.assertIn("episode.apply_selection", source)
        self.assertIn("episode.observe_decoded_token", source)

    def test_runner_keeps_offline_and_online_entry_points_explicit(self) -> None:
        source = (
            ROOT / "scripts/experiments/gsm8k/run_v4_system.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("build_v4_1_repair_bank.py", source)
        self.assertIn("repair_signatures.jsonl", source)
        self.assertIn("compile_v4_side_kv.py", source)
        self.assertIn("compile_v4_selector_anchors.py", source)
        self.assertIn("evaluate_v4_experience_memory.py", source)
        self.assertNotIn('"final-test"', source)


if __name__ == "__main__":
    unittest.main()
