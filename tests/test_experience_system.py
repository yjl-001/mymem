from __future__ import annotations

from dataclasses import dataclass
import unittest

from memgen.experience.retrieval import (
    BM25MemoryIndex,
    RetrievalQueryBuilder,
    RetrievalQueryConfig,
)
from memgen.experience.system import (
    ExperienceMemorySystemProfile,
    SemanticMemoryRetriever,
)


@dataclass(frozen=True)
class FakeRecord:
    memory_id: str
    payload_hash: str
    token_count: int
    sanitized_retrieval_key: str


class FakeTokenizer:
    def __init__(self, decoded: dict[int, str]):
        self.decoded = decoded

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(self.decoded[int(token)] for token in token_ids)


class ExperienceMemorySystemContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = (
            FakeRecord(
                memory_id="mem-rate",
                payload_hash="payload-rate",
                token_count=12,
                sanitized_retrieval_key=(
                    "multi step rate relation verify intermediate operation"
                ),
            ),
            FakeRecord(
                memory_id="mem-geometry",
                payload_hash="payload-geometry",
                token_count=9,
                sanitized_retrieval_key="geometry area perimeter shape",
            ),
        )
        self.profile = ExperienceMemorySystemProfile()
        self.tokenizer = FakeTokenizer({
            1: "identify",
            2: "the",
            3: "rate",
            4: "relation",
            5: "Final answer",
        })
        index = BM25MemoryIndex(records=self.records)  # type: ignore[arg-type]
        self.retriever = SemanticMemoryRetriever(
            index=index,
            query_builder=RetrievalQueryBuilder(
                tokenizer=self.tokenizer,
                analyzer=index.analyzer,
                config=RetrievalQueryConfig(
                    partial_cot_window_tokens=(
                        self.profile.partial_cot_window_tokens
                    )
                ),
            ),
            kv_valid_slot_counts={"mem-rate": 12, "mem-geometry": 9},
            profile=self.profile,
        )

    def test_profile_round_trip_preserves_the_fixed_reference_configuration(self) -> None:
        restored = ExperienceMemorySystemProfile.from_dict(
            self.profile.to_dict()
        )
        self.assertEqual(restored, self.profile)
        self.assertAlmostEqual(restored.memory_odds_multiplier, 10.0)
        self.assertEqual(
            restored.injection_policy, "persistent_from_trigger_through_eos"
        )
        with self.assertRaisesRegex(ValueError, "profile schema"):
            ExperienceMemorySystemProfile.from_dict({})

    def test_retrieval_uses_question_and_partial_cot_and_joins_side_kv_metadata(self) -> None:
        decision = self.retriever.retrieve(
            question="A worker combines two rates in a multi step relation.",
            partial_cot_token_ids=(1, 2, 3, 4),
        )
        self.assertTrue(decision.selected)
        self.assertEqual(decision.matched_memory.memory_id, "mem-rate")
        self.assertEqual(decision.matched_memory.kv_valid_slot_count, 12)
        self.assertEqual(decision.query["method"], "bm25")
        self.assertGreaterEqual(len(decision.hits), 1)
        serialized = decision.to_dict()
        self.assertNotIn("query_text", serialized["query"])

    def test_answer_marker_in_partial_cot_fails_closed(self) -> None:
        decision = self.retriever.retrieve(
            question="A rate problem.",
            partial_cot_token_ids=(1, 5),
        )
        self.assertEqual(decision.status, "query_rejected")
        self.assertFalse(decision.selected)
        self.assertIn(
            "retrieval_query_contains_final_answer_marker",
            decision.rejection_reasons,
        )


if __name__ == "__main__":
    unittest.main()
