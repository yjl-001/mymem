from __future__ import annotations

import unittest
from pathlib import Path

from memgen.experience.v3_5_hubness import (
    V35_HUBNESS_PRIMARY_SIDE,
    V35_HUBNESS_REPORT_SCHEMA,
    V35_HUBNESS_VARIANTS,
    anchor_summary,
    compare_variant_rows,
    numeric_summary,
    selection_hubness,
)


class V35HubnessContractTests(unittest.TestCase):
    def test_variants_are_fixed_without_pc_count_search(self) -> None:
        self.assertEqual(
            V35_HUBNESS_VARIANTS,
            (
                "raw",
                "key_centroid_centered",
                "key_centroid_centered_remove_pc1",
            ),
        )
        self.assertEqual(V35_HUBNESS_PRIMARY_SIDE, "reference")
        self.assertEqual(
            V35_HUBNESS_REPORT_SCHEMA,
            "experience-memory-v3.5-dynamic-hubness-decomposition-report-v1",
        )

    def test_numeric_summary_is_deterministic(self) -> None:
        summary = numeric_summary((4.0, 1.0, 3.0, 2.0))
        assert summary is not None
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["minimum"], 1.0)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["maximum"], 4.0)

    def test_hubness_reports_top_two_concentration_over_full_bank(self) -> None:
        rows = [
            {"top1_memory_id": memory_id}
            for memory_id in ("mem-a", "mem-a", "mem-a", "mem-b", "mem-c")
        ]
        result = selection_hubness(
            rows, ("mem-a", "mem-b", "mem-c", "mem-d")
        )
        self.assertEqual(result["selected_memory_count"], 3)
        self.assertEqual(result["selected_memory_fraction"], 0.75)
        self.assertEqual(result["top1_share"], 0.6)
        self.assertEqual(result["top2_combined_share"], 0.8)
        self.assertEqual(result["top_memories"][0]["memory_id"], "mem-a")

    def test_variant_comparison_uses_positive_delta_for_improvement(self) -> None:
        common = {
            "trajectory_side": "reference",
            "memory_id": "mem-a",
            "tensor_name": "reference_0001",
        }
        raw = [{
            **common,
            "own_memory_rank": 8,
            "own_minus_best_other_score": -0.05,
        }]
        candidate = [{
            **common,
            "own_memory_rank": 2,
            "own_minus_best_other_score": -0.01,
        }]
        result = compare_variant_rows(raw, candidate)
        self.assertEqual(result["candidate_rank_improved_count"], 1)
        self.assertEqual(result["candidate_rank_worsened_count"], 0)
        self.assertEqual(result["raw_minus_candidate_rank"]["mean"], 6.0)
        self.assertAlmostEqual(
            result["candidate_minus_raw_own_best_other_gap"]["mean"], 0.04
        )
        self.assertEqual(
            result["recall_delta_candidate_minus_raw"]["recall_at_5"], 1.0
        )

    def test_anchor_summary_preserves_partitions_and_permutation(self) -> None:
        memory_ids = ("mem-a", "mem-b", "mem-c")
        rows = []
        for index, own_id in enumerate(memory_ids):
            ranking = {
                own_id: 1,
                memory_ids[(index + 1) % 3]: 2,
                memory_ids[(index + 2) % 3]: 3,
            }
            rows.append({
                "own_memory_rank": 1,
                "own_memory_score": 0.8,
                "own_minus_best_other_score": 0.1,
                "top1_top2_margin": 0.1,
                "top1_memory_id": own_id,
                "own_memory_id": own_id,
                "rank_by_memory_id": ranking,
                "selector_partition": "train" if index < 2 else "holdout",
                "risk_partition": "train" if index != 1 else "holdout",
            })
        result = anchor_summary(
            rows,
            memory_ids=memory_ids,
            permutation_count=100,
        )
        self.assertEqual(result["all"]["recall_at_1"], 1.0)
        self.assertEqual(result["selector_train"]["sample_count"], 2)
        self.assertEqual(result["selector_holdout"]["sample_count"], 1)
        self.assertEqual(result["risk_fit_holdout"]["sample_count"], 1)
        self.assertEqual(result["hubness"]["selected_memory_count"], 3)
        self.assertIsNotNone(result["permutation_null"])

    def test_runner_is_diagnostic_only_and_does_not_run_reasoner(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_5_dynamic_hubness_audit.sh"
        ).read_text(encoding="utf-8")
        source_runner = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_5_dynamic_source_alignment_audit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("v3_5_dynamic_hubness", runner)
        self.assertIn("reasoner_forward_or_generation_run", runner)
        self.assertNotIn("--device", runner)
        self.assertNotIn("final-test", runner)
        self.assertNotIn("dynamic_hubness", source_runner)


if __name__ == "__main__":
    unittest.main()
