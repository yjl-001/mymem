from __future__ import annotations

import unittest

from memgen.experience.v3_7_cross_problem import (
    V37_RETRIEVAL_VARIANTS,
    candidate_union,
    causal_utility,
    reciprocal_rank_fusion_scores,
    stable_rank,
    summarize_causal_rows,
)


class V37CrossProblemContractsTest(unittest.TestCase):
    def test_stable_rank_and_rrf_are_deterministic(self) -> None:
        memory_ids = ("mem-b", "mem-a", "mem-c")
        ordered, ranks = stable_rank(memory_ids, (0.5, 0.5, 0.1))
        self.assertEqual(ordered, ("mem-a", "mem-b", "mem-c"))
        self.assertEqual(ranks, {"mem-a": 1, "mem-b": 2, "mem-c": 3})

        fused = reciprocal_rank_fusion_scores(
            memory_ids,
            {"mem-a": 1, "mem-b": 2, "mem-c": 3},
            {"mem-a": 3, "mem-b": 2, "mem-c": 1},
            rank_constant=60,
        )
        fused_order, _ = stable_rank(memory_ids, fused)
        # Reciprocal rank is convex: the symmetric 1/3 extremes narrowly beat
        # the 2/2 middle, and the exact a/c tie is broken by memory ID.
        self.assertEqual(fused_order, ("mem-a", "mem-c", "mem-b"))

    def test_candidate_union_retains_overlapping_sources(self) -> None:
        rankings = {
            variant: ("mem-a", "mem-b", "mem-c")
            for variant in V37_RETRIEVAL_VARIANTS
        }
        pool, sources = candidate_union(
            rankings,
            top_k=1,
            random_memory_ids=("mem-c",),
        )
        self.assertEqual(pool, ("mem-a", "mem-c"))
        self.assertEqual(sources["mem-a"], V37_RETRIEVAL_VARIANTS)
        self.assertEqual(sources["mem-c"], ("random_control",))

    def test_causal_utility_rejects_non_binary_rewards(self) -> None:
        self.assertEqual(causal_utility(baseline_reward=0, treatment_reward=1), 1)
        self.assertEqual(causal_utility(baseline_reward=1, treatment_reward=0), -1)
        self.assertEqual(causal_utility(baseline_reward=1, treatment_reward=1), 0)
        with self.assertRaises(ValueError):
            causal_utility(baseline_reward=0.5, treatment_reward=1)

    def test_summary_separates_oracle_and_retriever_utility(self) -> None:
        query_rows = [
            {
                "sample_id": "q1",
                "gate_eligible": True,
                "candidate_memory_ids": ["mem-a", "mem-b", "mem-c"],
                "baseline": {"strict_reward": 0.0},
            },
            {
                "sample_id": "q2",
                "gate_eligible": True,
                "candidate_memory_ids": ["mem-a", "mem-b", "mem-c"],
                "baseline": {"strict_reward": 1.0},
            },
            {
                "sample_id": "q3",
                "gate_eligible": False,
                "candidate_memory_ids": [],
                "baseline": {"strict_reward": 0.0},
            },
        ]
        local_ranks = {"mem-a": 1, "mem-b": 2, "mem-c": 3}
        other_ranks = {"mem-b": 1, "mem-a": 2, "mem-c": 3}
        rows = []
        rewards = {
            "q1": {"mem-a": 1.0, "mem-b": 0.0, "mem-c": 0.0},
            "q2": {"mem-a": 0.0, "mem-b": 1.0, "mem-c": 1.0},
        }
        baselines = {"q1": 0.0, "q2": 1.0}
        for sample_id in ("q1", "q2"):
            for memory_id in ("mem-a", "mem-b", "mem-c"):
                treatment = rewards[sample_id][memory_id]
                utility = int(treatment - baselines[sample_id])
                rows.append({
                    "sample_id": sample_id,
                    "memory_id": memory_id,
                    "baseline_reward": baselines[sample_id],
                    "treatment_reward": treatment,
                    "causal_utility": utility,
                    "candidate_sources": (
                        ["random_control"] if memory_id == "mem-c" else []
                    ),
                    "rank_by_variant": {
                        variant: (
                            local_ranks[memory_id]
                            if variant == "state_local16"
                            else other_ranks[memory_id]
                        )
                        for variant in V37_RETRIEVAL_VARIANTS
                    },
                })

        summary = summarize_causal_rows(
            query_rows=query_rows,
            treatment_rows=rows,
            candidate_top_k=2,
        )
        self.assertEqual(summary["selected_query_count"], 3)
        self.assertEqual(summary["gate_eligible_query_count"], 2)
        self.assertEqual(summary["evaluated_pool_any_helpful_query_count"], 1)
        self.assertAlmostEqual(
            summary["baseline_accuracy_gate_eligible"], 0.5
        )
        self.assertAlmostEqual(
            summary["evaluated_pool_oracle_accuracy_gate_eligible"], 1.0
        )
        local = summary["variants"]["state_local16"]
        self.assertEqual(local["top1_helpful_count"], 1)
        self.assertEqual(local["top1_harmful_count"], 1)
        self.assertAlmostEqual(local["top1_net_utility_mean"], 0.0)
        delta = summary["variants"]["state_delta"]
        self.assertEqual(delta["top1_helpful_count"], 0)
        self.assertEqual(delta["top1_harmful_count"], 0)

    def test_pool_oracle_can_abstain_from_an_all_harmful_pool(self) -> None:
        summary = summarize_causal_rows(
            query_rows=[{
                "sample_id": "q1",
                "gate_eligible": True,
                "candidate_memory_ids": ["mem-a"],
                "baseline": {"strict_reward": 1.0},
            }],
            treatment_rows=[{
                "sample_id": "q1",
                "memory_id": "mem-a",
                "baseline_reward": 1.0,
                "treatment_reward": 0.0,
                "causal_utility": -1,
                "candidate_sources": [],
                "rank_by_variant": {
                    variant: 1 for variant in V37_RETRIEVAL_VARIANTS
                },
            }],
            candidate_top_k=1,
        )
        self.assertEqual(
            summary["evaluated_pool_oracle_accuracy_gate_eligible"], 1.0
        )
        self.assertEqual(summary["evaluated_pool_oracle_uplift_gate_eligible"], 0.0)
        self.assertEqual(
            summary["variants"]["state_local16"]["top1_accuracy_uplift"], -1.0
        )


if __name__ == "__main__":
    unittest.main()
