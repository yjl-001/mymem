from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_v3_5_dynamic_source_alignment import (
    _paired_anchor_comparison,
    tokenize_trajectory,
)

from memgen.experience.v3_5_source_alignment import (
    CounterfactualGateObservation,
    V35_SOURCE_ALIGNMENT_PRIMARY_ANCHOR,
    V35_SOURCE_ALIGNMENT_REPORT_SCHEMA,
    counterfactual_attempts,
    permutation_null,
    rank_metrics,
    score_query,
    stable_score_ranking,
)


class V35SourceAlignmentContractTests(unittest.TestCase):
    def test_schema_and_primary_anchor_are_independent_diagnostics(self) -> None:
        self.assertEqual(
            V35_SOURCE_ALIGNMENT_REPORT_SCHEMA,
            "experience-memory-v3.5-dynamic-source-alignment-report-v1",
        )
        self.assertEqual(
            V35_SOURCE_ALIGNMENT_PRIMARY_ANCHOR,
            "reference_first_counterfactual_joint_gate_event",
        )

    def test_stable_score_ranking_uses_memory_id_for_ties(self) -> None:
        ranking = stable_score_ranking(
            memory_ids=("mem-c", "mem-a", "mem-b"),
            scores=(0.4, 0.7, 0.7),
        )
        self.assertEqual(ranking, (1, 2, 0))

    def test_score_query_reports_own_rank_gap_and_top_hits(self) -> None:
        result = score_query(
            memory_ids=("mem-a", "mem-b", "mem-c"),
            scores=(0.3, 0.8, 0.5),
            own_memory_id="mem-c",
            top_n=3,
        )
        self.assertEqual(result["own_memory_rank"], 2)
        self.assertAlmostEqual(result["own_memory_score"], 0.5)
        self.assertAlmostEqual(result["own_minus_best_other_score"], -0.3)
        self.assertEqual(result["top1_memory_id"], "mem-b")
        self.assertEqual(
            [item["memory_id"] for item in result["top_hits"]],
            ["mem-b", "mem-c", "mem-a"],
        )
        self.assertEqual(result["rank_by_memory_id"]["mem-c"], 2)

    def test_score_query_can_omit_full_rank_lookup_for_token_curves(self) -> None:
        result = score_query(
            memory_ids=("mem-a", "mem-b", "mem-c"),
            scores=(0.3, 0.8, 0.5),
            own_memory_id="mem-c",
            top_n=2,
            include_rank_lookup=False,
        )
        self.assertEqual(result["own_memory_rank"], 2)
        self.assertEqual(len(result["top_hits"]), 2)
        self.assertNotIn("rank_by_memory_id", result)

    def test_counterfactual_gate_rearms_after_two_low_without_same_token_trigger(self) -> None:
        observations = [
            CounterfactualGateObservation(0, 10.0, 1.0),
            CounterfactualGateObservation(1, 1.0, 1.0),
            CounterfactualGateObservation(2, 1.0, 1.0),
            CounterfactualGateObservation(3, 10.0, 1.0),
            CounterfactualGateObservation(4, 10.0, 1.0),
            CounterfactualGateObservation(5, 1.0, 1.0),
            CounterfactualGateObservation(6, 1.0, 1.0),
            CounterfactualGateObservation(7, 10.0, 1.0),
        ]
        attempts = counterfactual_attempts(
            observations,
            high_entropy_threshold=5.0,
            low_entropy_threshold=2.0,
            risk_threshold=0.0,
        )
        self.assertEqual(
            [item["reasoning_rank"] for item in attempts], [0, 3, 7]
        )
        self.assertEqual(
            [item["attempt_number"] for item in attempts], [1, 2, 3]
        )
        self.assertEqual(attempts[-1]["state_after"], "EXHAUSTED")

    def test_counterfactual_gate_uses_strict_risk_threshold(self) -> None:
        attempts = counterfactual_attempts(
            [
                CounterfactualGateObservation(0, 10.0, 0.5),
                CounterfactualGateObservation(1, 10.0, 0.5001),
            ],
            high_entropy_threshold=5.0,
            low_entropy_threshold=2.0,
            risk_threshold=0.5,
        )
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["reasoning_rank"], 1)

    def test_rank_metrics_preserve_uniform_reference(self) -> None:
        metrics = rank_metrics((1, 5, 10, 20), memory_count=40)
        assert metrics is not None
        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["recall_at_1"], 0.25)
        self.assertEqual(metrics["recall_at_5"], 0.5)
        self.assertEqual(metrics["recall_at_10"], 0.75)
        self.assertEqual(metrics["recall_at_32"], 1.0)
        self.assertEqual(metrics["uniform_rank_reference_at_10"], 0.25)

    def test_paired_anchor_comparison_uses_positive_delta_for_target_better(self) -> None:
        comparison = _paired_anchor_comparison(
            target_rows=(
                {"memory_id": "mem-a", "own_memory_rank": 2},
                {"memory_id": "mem-b", "own_memory_rank": 7},
            ),
            reference_rows=(
                {"memory_id": "mem-a", "own_memory_rank": 9},
                {"memory_id": "mem-b", "own_memory_rank": 3},
                {"memory_id": "mem-c", "own_memory_rank": 1},
            ),
        )
        self.assertEqual(comparison["paired_count"], 2)
        self.assertEqual(comparison["target_better_count"], 1)
        self.assertEqual(comparison["reference_better_count"], 1)
        self.assertEqual(comparison["reference_only_count"], 1)
        self.assertEqual(
            comparison["reference_minus_target_rank"]["mean"], 1.5
        )

    def test_permutation_null_is_deterministic_and_preserves_rank_geometry(self) -> None:
        rows = [
            {
                "own_memory_id": own,
                "rank_by_memory_id": lookup,
            }
            for own, lookup in (
                ("mem-a", {"mem-a": 1, "mem-b": 2, "mem-c": 3}),
                ("mem-b", {"mem-a": 3, "mem-b": 1, "mem-c": 2}),
                ("mem-c", {"mem-a": 2, "mem-b": 3, "mem-c": 1}),
            )
        ]
        first = permutation_null(
            rows, memory_count=3, iterations=100, seed=17
        )
        second = permutation_null(
            rows, memory_count=3, iterations=100, seed=17
        )
        self.assertEqual(first, second)
        assert first is not None
        self.assertEqual(first["sample_count"], 3)
        self.assertEqual(
            first["metrics"]["recall_at_1"]["observed"], 1.0
        )
        self.assertLessEqual(
            first["metrics"]["recall_at_1"][
                "one_sided_enrichment_p_value"
            ],
            0.25,
        )

    def test_invalid_observation_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            CounterfactualGateObservation(0.5, 1.0, 1.0)  # type: ignore[arg-type]

    def test_trajectory_contract_stops_before_answer_marker(self) -> None:
        class Tokenizer:
            @staticmethod
            def encode(value: str, add_special_tokens: bool = False) -> list[int]:
                self = value
                assert add_special_tokens is False
                return [ord(character) for character in self]

        prompt = [1, 2, 3]
        completion = "reasoning \\boxed{7} ignored"
        tokenized = tokenize_trajectory(Tokenizer(), prompt, completion)
        self.assertEqual(
            tokenized.pre_answer_token_count,
            len("reasoning "),
        )
        self.assertEqual(
            tokenized.reasoning_indices,
            tuple(range(len(prompt), len(prompt) + len("reasoning "))),
        )

    def test_audit_runner_is_separate_from_formal_v35_runner(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runner = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_5_dynamic_source_alignment_audit.sh"
        ).read_text(encoding="utf-8")
        formal = (
            root
            / "scripts/experiments/gsm8k/"
            "run_v3_5_applicability_selector_experiment.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("v3_5_dynamic_source_alignment", runner)
        self.assertIn("formal_v3_5_qualification_changed", runner)
        self.assertNotIn("final-test", runner)
        self.assertNotIn("dynamic_source_alignment", formal)


if __name__ == "__main__":
    unittest.main()
