from __future__ import annotations

import unittest
from pathlib import Path

from memgen.experience.v3_5_query_state import (
    V35_QUERY_STATE_BASELINE,
    V35_QUERY_STATE_KEY_VARIANTS,
    V35_QUERY_STATE_LOCAL_WINDOW,
    V35_QUERY_STATE_PRIMARY_KEY,
    V35_QUERY_STATE_PRIMARY_SIDE,
    V35_QUERY_STATE_REPORT_SCHEMA,
    V35_QUERY_STATE_VARIANTS,
    compare_query_rows,
    rank_correlation,
)


def _row(
    *,
    rank: int,
    gap: float,
    top1: str,
    side: str = "reference",
) -> dict[str, object]:
    return {
        "trajectory_side": side,
        "memory_id": "mem-a",
        "anchor_tensor_name": f"{side}_0001",
        "tensor_name": f"variant__{side}_0001",
        "own_memory_rank": rank,
        "own_minus_best_other_score": gap,
        "top1_memory_id": top1,
    }


class V35QueryStateContractTests(unittest.TestCase):
    def test_query_and_key_variants_are_fixed(self) -> None:
        self.assertEqual(
            V35_QUERY_STATE_VARIANTS,
            (
                "prompt_boundary",
                "current_token",
                "prompt_subtracted_delta",
                "local_reasoning_window_16",
            ),
        )
        self.assertEqual(
            V35_QUERY_STATE_KEY_VARIANTS,
            ("applicability_key", "dynamic_key"),
        )
        self.assertEqual(V35_QUERY_STATE_LOCAL_WINDOW, 16)
        self.assertEqual(V35_QUERY_STATE_BASELINE, "current_token")
        self.assertEqual(V35_QUERY_STATE_PRIMARY_KEY, "applicability_key")
        self.assertEqual(V35_QUERY_STATE_PRIMARY_SIDE, "reference")
        self.assertEqual(
            V35_QUERY_STATE_REPORT_SCHEMA,
            "experience-memory-v3.5-dynamic-query-state-"
            "decomposition-report-v1",
        )

    def test_rank_correlation_is_deterministic(self) -> None:
        self.assertAlmostEqual(rank_correlation((1, 2, 3), (2, 4, 6)), 1.0)
        self.assertAlmostEqual(rank_correlation((1, 2, 3), (6, 4, 2)), -1.0)
        self.assertIsNone(rank_correlation((1, 1), (2, 3)))
        with self.assertRaisesRegex(ValueError, "different lengths"):
            rank_correlation((1,), (1, 2))

    def test_query_comparison_pairs_by_authenticated_anchor(self) -> None:
        baseline = [
            _row(rank=8, gap=-0.08, top1="mem-hub"),
            _row(
                rank=6,
                gap=-0.06,
                top1="mem-other",
                side="target",
            ),
        ]
        candidate = [
            _row(rank=2, gap=-0.02, top1="mem-a"),
            _row(rank=3, gap=-0.03, top1="mem-a", side="target"),
        ]
        result = compare_query_rows(baseline, candidate)
        self.assertEqual(result["candidate_rank_improved_count"], 2)
        self.assertEqual(result["top1_same_count"], 0)
        self.assertEqual(result["top1_same_fraction"], 0.0)
        self.assertAlmostEqual(
            result["candidate_minus_raw_own_best_other_gap"]["mean"],
            0.045,
        )

    def test_query_comparison_rejects_anchor_coverage_drift(self) -> None:
        baseline = [_row(rank=8, gap=-0.08, top1="mem-hub")]
        candidate = [_row(rank=2, gap=-0.02, top1="mem-a")]
        candidate[0]["anchor_tensor_name"] = "reference_9999"
        with self.assertRaisesRegex(ValueError, "identical queries|identical anchors"):
            compare_query_rows(baseline, candidate)

    def test_runner_is_gpu_diagnostic_and_has_no_generation_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_5_dynamic_query_state_audit.sh"
        ).read_text(encoding="utf-8")
        audit = (
            root / "scripts/audit_v3_5_dynamic_query_state.py"
        ).read_text(encoding="utf-8")
        source_runner = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_5_dynamic_source_alignment_audit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("v3_5_dynamic_query_state", runner)
        self.assertIn("--device", runner)
        self.assertIn('"reasoner_forward_run": True', audit)
        self.assertIn('"generation_run": False', audit)
        self.assertIn('"query_variant_selected": False', audit)
        self.assertIn("local_reasoning_window_size", audit)
        self.assertNotIn("final-test", runner)
        self.assertNotIn("dynamic_query_state", source_runner)


if __name__ == "__main__":
    unittest.main()
