from __future__ import annotations

import unittest
from pathlib import Path

from memgen.experience.v3_5_key_components import (
    V35_KEY_COMPONENT_CURRENT_VARIANT,
    V35_KEY_COMPONENT_PRIMARY_SIDE,
    V35_KEY_COMPONENT_REPORT_SCHEMA,
    V35_KEY_COMPONENT_VARIANTS,
    pairwise_variant_comparisons,
)


def _row(
    *,
    rank: int,
    gap: float,
    side: str = "reference",
) -> dict[str, object]:
    return {
        "trajectory_side": side,
        "memory_id": "mem-a",
        "tensor_name": f"{side}-0001",
        "own_memory_rank": rank,
        "own_minus_best_other_score": gap,
    }


class V35KeyComponentContractTests(unittest.TestCase):
    def test_fixed_variants_have_no_search_or_centering(self) -> None:
        self.assertEqual(
            V35_KEY_COMPONENT_VARIANTS,
            (
                "applicability_key",
                "dynamic_key",
                "paired_decision_residual",
            ),
        )
        self.assertEqual(V35_KEY_COMPONENT_CURRENT_VARIANT, "dynamic_key")
        self.assertEqual(V35_KEY_COMPONENT_PRIMARY_SIDE, "reference")
        self.assertEqual(
            V35_KEY_COMPONENT_REPORT_SCHEMA,
            "experience-memory-v3.5-dynamic-key-component-report-v1",
        )

    def test_pairwise_comparison_covers_all_three_fixed_pairs(self) -> None:
        rows = {
            "applicability_key": [
                _row(rank=10, gap=-0.10),
                _row(rank=9, gap=-0.09, side="target"),
            ],
            "dynamic_key": [
                _row(rank=7, gap=-0.07),
                _row(rank=8, gap=-0.08, side="target"),
            ],
            "paired_decision_residual": [
                _row(rank=2, gap=-0.02),
                _row(rank=3, gap=-0.03, side="target"),
            ],
        }
        result = pairwise_variant_comparisons(rows)
        self.assertEqual(
            tuple(result),
            (
                "dynamic_key_versus_applicability_key",
                "paired_decision_residual_versus_applicability_key",
                "paired_decision_residual_versus_dynamic_key",
            ),
        )
        residual_vs_dynamic = result[
            "paired_decision_residual_versus_dynamic_key"
        ]
        self.assertEqual(
            residual_vs_dynamic["by_side"]["reference"][
                "candidate_rank_improved_count"
            ],
            1,
        )
        self.assertEqual(
            residual_vs_dynamic["by_side"]["target"][
                "raw_minus_candidate_rank"
            ]["mean"],
            5.0,
        )

    def test_pairwise_comparison_rejects_variant_order_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "variant order"):
            pairwise_variant_comparisons({
                "dynamic_key": [],
                "applicability_key": [],
                "paired_decision_residual": [],
            })

    def test_runner_is_diagnostic_only_and_requires_prior_hubness(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_5_dynamic_key_component_audit.sh"
        ).read_text(encoding="utf-8")
        audit = (
            root / "scripts/audit_v3_5_dynamic_key_components.py"
        ).read_text(encoding="utf-8")
        self.assertIn("v3_5_dynamic_key_components", runner)
        self.assertIn("v3_5_dynamic_hubness", runner)
        self.assertIn("reasoner_forward_or_generation_run", runner)
        self.assertNotIn("--device", runner)
        self.assertNotIn("final-test", runner)
        self.assertIn(
            "l2_normalize(dynamic_key_i_minus_applicability_key_i)", audit
        )
        self.assertIn('"variant_selected": False', audit)


if __name__ == "__main__":
    unittest.main()
