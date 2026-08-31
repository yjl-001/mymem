from __future__ import annotations

import unittest
from pathlib import Path

from memgen.experience.v3_6_state_keys import (
    V36_STATE_KEY_BANK_SCHEMA,
    V36_STATE_KEY_IDENTITY_CONTROL,
    V36_STATE_KEY_PRIMARY_VARIANT,
    V36_STATE_KEY_REPORT_SCHEMA,
    V36_STATE_KEY_TEXT_CONTROL,
    V36_STATE_KEY_TRAJECTORY_KEY_SIDE,
    V36_STATE_KEY_TRAJECTORY_QUERY_SIDE,
    V36_STATE_KEY_VARIANTS,
    compare_state_key_rows,
)


def _row(
    *,
    memory_id: str,
    tensor_name: str,
    rank: int,
    gap: float,
    top1: str,
) -> dict[str, object]:
    return {
        "trajectory_side": "target",
        "memory_id": memory_id,
        "tensor_name": tensor_name,
        "own_memory_rank": rank,
        "own_minus_best_other_score": gap,
        "top1_memory_id": top1,
    }


class V36StateKeyContractTests(unittest.TestCase):
    def test_variants_and_roles_are_fixed(self) -> None:
        self.assertEqual(
            V36_STATE_KEY_VARIANTS,
            (
                "text_applicability__target_current_control",
                "state_prompt__target_prompt_identity_control",
                "state_current__target_current",
                "state_delta__target_delta",
                "state_local16__target_local16",
            ),
        )
        self.assertEqual(
            V36_STATE_KEY_PRIMARY_VARIANT,
            "state_current__target_current",
        )
        self.assertEqual(
            V36_STATE_KEY_TEXT_CONTROL,
            "text_applicability__target_current_control",
        )
        self.assertEqual(
            V36_STATE_KEY_IDENTITY_CONTROL,
            "state_prompt__target_prompt_identity_control",
        )
        self.assertEqual(V36_STATE_KEY_TRAJECTORY_KEY_SIDE, "reference")
        self.assertEqual(V36_STATE_KEY_TRAJECTORY_QUERY_SIDE, "target")
        self.assertEqual(
            V36_STATE_KEY_REPORT_SCHEMA,
            "experience-memory-v3.6-source-state-retrieval-key-report-v1",
        )
        self.assertEqual(
            V36_STATE_KEY_BANK_SCHEMA,
            "experience-memory-v3.6-source-state-retrieval-key-bank-v1",
        )

    def test_comparison_is_paired_by_target_anchor(self) -> None:
        baseline = [
            _row(
                memory_id="mem-a",
                tensor_name="target_0001",
                rank=7,
                gap=-0.07,
                top1="mem-hub",
            ),
            _row(
                memory_id="mem-b",
                tensor_name="target_0002",
                rank=4,
                gap=-0.04,
                top1="mem-hub",
            ),
        ]
        candidate = [
            _row(
                memory_id="mem-a",
                tensor_name="target_0001",
                rank=1,
                gap=0.02,
                top1="mem-a",
            ),
            _row(
                memory_id="mem-b",
                tensor_name="target_0002",
                rank=2,
                gap=-0.01,
                top1="mem-a",
            ),
        ]
        result = compare_state_key_rows(baseline, candidate)
        self.assertEqual(result["candidate_rank_improved_count"], 2)
        self.assertEqual(result["top1_recovered_count"], 1)
        self.assertEqual(result["top1_same_count"], 0)

    def test_comparison_rejects_query_coverage_drift(self) -> None:
        baseline = [
            _row(
                memory_id="mem-a",
                tensor_name="target_0001",
                rank=7,
                gap=-0.07,
                top1="mem-hub",
            )
        ]
        candidate = [
            _row(
                memory_id="mem-a",
                tensor_name="target_9999",
                rank=1,
                gap=0.02,
                top1="mem-a",
            )
        ]
        with self.assertRaisesRegex(ValueError, "identical queries"):
            compare_state_key_rows(baseline, candidate)

    def test_runner_is_offline_and_preserves_value_payload(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_6_source_state_key_audit.sh"
        ).read_text(encoding="utf-8")
        audit = (
            root / "scripts/audit_v3_6_source_state_keys.py"
        ).read_text(encoding="utf-8")
        self.assertIn("v3_6_source_state_keys", runner)
        self.assertIn('"reasoner_forward_or_generation_run": False', audit)
        self.assertIn('"side_kv_payload_changed": False', audit)
        self.assertIn("full_when_facing_prefer_avoid", audit)
        self.assertIn("target_and_reference_tensor_origins_distinct", audit)
        self.assertIn("exact_cross_trajectory_embedding_matches_measured", audit)
        self.assertNotIn("final-test", runner)
        self.assertNotIn("--device", runner)


if __name__ == "__main__":
    unittest.main()
