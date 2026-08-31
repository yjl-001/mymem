from __future__ import annotations

import json
from pathlib import Path
import unittest

from memgen.experience.v3_7_cross_problem import V37_RETRIEVAL_VARIANTS
from memgen.experience.v3_8_full_bank import (
    build_utility_matrix,
    summarize_full_bank_matrix,
)


def _rows() -> list[dict[str, object]]:
    queries = ("q1", "q2", "q3")
    memories = ("m1", "m2", "m3", "m4")
    current_order = ("m1", "m2", "m3", "m4")
    text_order = ("m3", "m4", "m2", "m1")
    utilities = {
        "q1": {"m1": 0, "m2": 1, "m3": 0, "m4": 0},
        "q2": {"m1": 0, "m2": 0, "m3": 0, "m4": 0},
        "q3": {"m1": 1, "m2": 0, "m3": 0, "m4": 1},
    }
    rows: list[dict[str, object]] = []
    for sample_id in queries:
        for memory_id in memories:
            rank_by_variant = {
                variant: (
                    text_order.index(memory_id) + 1
                    if variant == "text_applicability"
                    else current_order.index(memory_id) + 1
                )
                for variant in V37_RETRIEVAL_VARIANTS
            }
            score_by_variant = {
                variant: float(5 - rank_by_variant[variant])
                for variant in V37_RETRIEVAL_VARIANTS
            }
            utility = utilities[sample_id][memory_id]
            rows.append({
                "sample_id": sample_id,
                "memory_id": memory_id,
                "baseline_reward": 0.0,
                "treatment_reward": float(utility),
                "causal_utility": utility,
                "rank_by_variant": rank_by_variant,
                "score_by_variant": score_by_variant,
                "evidence_origin": "test",
            })
    return rows


class V38FullBankContractsTest(unittest.TestCase):
    def test_matrix_is_complete_and_preserves_fixed_order(self) -> None:
        # JSONL artifacts use sort_keys=True, so nested retrieval mappings do
        # not retain the tuple declaration order after a real round trip.
        rows = json.loads(json.dumps(_rows(), sort_keys=True))
        matrix = build_utility_matrix(
            query_ids=("q1", "q2", "q3"),
            memory_ids=("m1", "m2", "m3", "m4"),
            treatment_rows=rows,
        )
        self.assertEqual(matrix["shape"], [3, 4])
        self.assertEqual(
            matrix["utilities"],
            [[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
        )
        self.assertEqual(matrix["helpful_memory_ids_by_query"]["q2"], [])
        self.assertEqual(
            matrix["helped_query_ids_by_memory"]["m4"], ["q3"]
        )

    def test_summary_separates_coverage_retrieval_and_reranking(self) -> None:
        summary = summarize_full_bank_matrix(
            query_ids=("q1", "q2", "q3"),
            memory_ids=("m1", "m2", "m3", "m4"),
            treatment_rows=_rows(),
            diagnosis_k=2,
        )
        coverage = summary["causal_coverage"]
        self.assertEqual(coverage["recoverable_failure_count"], 2)
        self.assertEqual(coverage["no_helpful_memory_failure_count"], 1)
        self.assertEqual(coverage["helpful_pair_count"], 3)
        self.assertEqual(coverage["memory_count_helpful_for_any_query"], 3)

        current = summary["retrieval_variants"]["state_current"]
        self.assertEqual(current["helpful_hit_at_k"]["1"]["count"], 1)
        self.assertEqual(current["helpful_hit_at_k"]["2"]["count"], 2)
        decomposition = current["pipeline_decomposition_at_diagnosis_k"]
        self.assertEqual(decomposition["no_helpful_in_authenticated_bank"], 1)
        self.assertEqual(decomposition["helpful_exists_but_missed_top_k"], 0)
        self.assertEqual(decomposition["helpful_in_top_k_but_missed_top1"], 1)
        self.assertEqual(decomposition["helpful_at_top1"], 1)
        self.assertEqual(decomposition["partition_sum"], 3)

        text = summary["retrieval_variants"]["text_applicability"]
        self.assertEqual(text["helpful_hit_at_k"]["1"]["count"], 0)
        self.assertEqual(text["helpful_hit_at_k"]["2"]["count"], 1)
        self.assertGreater(
            text["helpful_hit_at_k"]["2"]["uniform_random_expected_count"],
            0.0,
        )
        self.assertTrue(
            summary["interpretation_limits"][
                "failure_only_sweep_cannot_measure_harm"
            ]
        )

    def test_matrix_rejects_missing_pairs_and_non_failure_baselines(self) -> None:
        rows = _rows()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_utility_matrix(
                query_ids=("q1", "q2", "q3"),
                memory_ids=("m1", "m2", "m3", "m4"),
                treatment_rows=rows[:-1],
            )
        rows[0]["baseline_reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "baseline failures"):
            build_utility_matrix(
                query_ids=("q1", "q2", "q3"),
                memory_ids=("m1", "m2", "m3", "m4"),
                treatment_rows=rows,
            )

    def test_nonstandard_diagnosis_k_is_reported(self) -> None:
        summary = summarize_full_bank_matrix(
            query_ids=("q1", "q2", "q3"),
            memory_ids=("m1", "m2", "m3", "m4"),
            treatment_rows=_rows(),
            diagnosis_k=3,
        )
        self.assertIn(
            "3",
            summary["retrieval_variants"]["state_current"][
                "helpful_hit_at_k"
            ],
        )

    def test_runner_documents_authenticated_bank_scope(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / "scripts/experiments/gsm8k/run_v3_8_failure_full_bank_causal_audit.sh"
        ).read_text(encoding="utf-8")
        audit = (root / "scripts/audit_v3_8_failure_full_bank_causal.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--v37-run-dir", runner)
        self.assertIn("complete_authenticated_v36_state_key_bank", audit)
        self.assertIn("full_original_side_kv_bank_exhaustively_treated", audit)


if __name__ == "__main__":
    unittest.main()
