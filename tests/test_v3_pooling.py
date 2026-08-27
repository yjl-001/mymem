from __future__ import annotations

import unittest

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v3_pooling import (
    V3_POOLING_BASELINE,
    V3_POOLING_FULL_MEAN,
    V3_POOLING_PARTIAL_MEAN,
    V3_POOLING_PRE_BOUNDARY,
    qualify_pooling_candidate,
    rank_qualified_pooling_candidates,
    reconstruct_first_attempt_prefix,
    stable_top_indices,
)


class V3PoolingAuditTests(unittest.TestCase):
    def test_reconstructs_authenticated_full_first_attempt_prefix(self) -> None:
        prefix = [10, 11, 20, 21]
        result = reconstruct_first_attempt_prefix(
            prompt_token_ids=(10, 11),
            completion_token_ids=(20, 21, 22),
            generated_boundary_index=1,
            query_audit={
                "query_token_count": 4,
                "prompt_token_count": 2,
                "partial_cot_token_count": 2,
                "query_token_ids_sha256": canonical_json_sha256(prefix),
            },
        )
        self.assertEqual(result, tuple(prefix))
        with self.assertRaisesRegex(ValueError, "reconstruction failed"):
            reconstruct_first_attempt_prefix(
                prompt_token_ids=(10, 11),
                completion_token_ids=(20, 21, 22),
                generated_boundary_index=1,
                query_audit={
                    "query_token_count": 4,
                    "prompt_token_count": 2,
                    "partial_cot_token_count": 2,
                    "query_token_ids_sha256": "wrong",
                },
            )

    def test_stable_ranking_uses_bank_order_for_equal_scores(self) -> None:
        self.assertEqual(
            stable_top_indices((0.2, 0.9, 0.9, 0.1), top_k=3),
            (1, 2, 0),
        )

    def test_candidate_must_pass_all_answer_blind_geometry_gates(self) -> None:
        baseline = {
            "top1_share": 0.45,
            "gini": 0.95,
            "selected_memory_count": 23,
            "normalized_entropy": 0.30,
        }
        candidate = {
            "top1_share": 0.30,
            "gini": 0.80,
            "selected_memory_count": 30,
            "normalized_entropy": 0.50,
        }
        result = qualify_pooling_candidate(
            baseline=baseline, candidate=candidate
        )
        self.assertTrue(result["qualified"])
        harmed_support = dict(candidate, selected_memory_count=22)
        self.assertFalse(qualify_pooling_candidate(
            baseline=baseline,
            candidate=harmed_support,
        )["qualified"])

    def test_qualified_candidates_use_frozen_geometry_ranking(self) -> None:
        summaries = {
            V3_POOLING_BASELINE: {
                "top1_share": 0.45,
                "gini": 0.95,
                "selected_memory_count": 23,
                "normalized_entropy": 0.30,
            },
            V3_POOLING_PRE_BOUNDARY: {
                "top1_share": 0.25,
                "gini": 0.70,
                "selected_memory_count": 28,
                "normalized_entropy": 0.60,
            },
            V3_POOLING_PARTIAL_MEAN: {
                "top1_share": 0.20,
                "gini": 0.70,
                "selected_memory_count": 31,
                "normalized_entropy": 0.65,
            },
            V3_POOLING_FULL_MEAN: {
                "top1_share": 0.35,
                "gini": 0.80,
                "selected_memory_count": 12,
                "normalized_entropy": 0.40,
            },
        }
        self.assertEqual(
            rank_qualified_pooling_candidates(summaries),
            (V3_POOLING_PARTIAL_MEAN, V3_POOLING_PRE_BOUNDARY),
        )


if __name__ == "__main__":
    unittest.main()
