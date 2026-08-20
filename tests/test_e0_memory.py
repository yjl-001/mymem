from __future__ import annotations

import copy
from dataclasses import replace
import unittest

from memgen.experience.memory import (
    ApprovedMemorySourceSelector,
    MemoryArtifactAuditor,
    MemoryBankBuilder,
    MemoryRecord,
    MemoryRecordCompiler,
    MemoryRecordRejected,
    MemorySanitizerConfig,
    PayloadSanitizer,
)
from memgen.experience.retrieval import BM25MemoryIndex, RetrievalQueryBuilder


class WhitespaceTokenizer:
    """Deterministic tokenizer fixture; production uses the frozen reasoner tokenizer."""

    def __init__(self) -> None:
        self._token_by_id: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for token in text.split():
            token_id = sum((index + 1) * ord(char) for index, char in enumerate(token))
            while token_id in self._token_by_id and self._token_by_id[token_id] != token:
                token_id += 1
            self._token_by_id[token_id] = token
            ids.append(token_id)
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(self._token_by_id[int(token_id)] for token_id in token_ids)


def supported_assessment() -> dict:
    return {"status": "supported", "evidence": "Independent grounded review."}


def approved_record(experience_id: str = "experience-1") -> dict:
    target = {
        "situation_signature": "A multistep calculation must preserve the stated relationships.",
        "transferable_decision": "Carry the supported relation forward before simplifying.",
        "verification_rule": "Check that every operation preserves the intended quantity.",
        "applicability_boundary": "Use this while the current relation remains supported.",
        "confidence": 0.9,
    }
    reference = {
        "competing_pattern": "Replacing a supported relation with an unrelated shortcut.",
        "failure_signal": "A derived quantity no longer follows from the previous statement.",
        "failure_mechanism": (
            "The shortcut changes the relation and propagates an incorrect result."
        ),
        "non_reuse_boundary": "Reconsider only when new evidence invalidates the relation.",
        "confidence": 0.9,
    }
    return {
        "schema_version": "teacher-bank-record-v3",
        "experience_id": experience_id,
        "experience_type": "answer_correctness",
        "reference_evidence": "verified_failure",
        "reference_failure_types": ["boxed_answer_mismatch"],
        "provenance_sha256": "phase1-provenance",
        "source": {
            "logical_split": "bank-source",
            "question_sha256": "question-hash",
        },
        "student": {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "model_revision": "student-revision",
            "tokenizer_revision": "tokenizer-revision",
            "frozen": True,
        },
        "source_episode_ids": {"target": "target-episode", "reference": "reference-episode"},
        "bank": {
            "experience_type": "answer_correctness",
            "failure_types": ["boxed_answer_mismatch"],
            "target": target,
            "reference": reference,
            "evidence": {
                "target_observation": "The accepted response reached the verified output.",
                "reference_observation": "The rejected response reached a different output.",
            },
        },
        "ai_review_gate": {
            "schema_version": "phase1-ai-review-record-v2",
            "prompt_version": "phase1-ai-review-v2-field-evidence-rubric",
            "route": "ai_approved",
            "review_provenance_sha256": "review-provenance",
            "ai_review": {
                "decision": "approve",
                "confidence": 0.9,
                "field_assessments": {
                    "target": {
                        key: supported_assessment()
                        for key in (
                            "situation_signature",
                            "transferable_decision",
                            "verification_rule",
                            "applicability_boundary",
                        )
                    },
                    "reference": {
                        key: supported_assessment()
                        for key in (
                            "competing_pattern",
                            "failure_signal",
                            "failure_mechanism",
                            "non_reuse_boundary",
                        )
                    },
                },
                "pair_assessments": {
                    key: supported_assessment()
                    for key in (
                        "target_reference_distinct",
                        "factually_consistent",
                        "causal_attribution",
                        "failure_type_compatibility",
                        "transferable_without_instance_leakage",
                    )
                },
                "issues": [],
            },
        },
    }


def legacy_approved_record(experience_id: str = "experience-1") -> dict:
    record = approved_record(experience_id)
    supported_criteria = {
        field: True
        for field in (
            "target_supported",
            "reference_supported",
            "target_reference_distinct",
            "factually_consistent",
            "failure_type_aligned",
            "transferable_without_instance_leakage",
        )
    }
    record["ai_review_gate"] = {
        "schema_version": "phase1-ai-review-record-v1",
        "prompt_version": "phase1-ai-review-v1-independent-evidence",
        "route": "ai_approved",
        "review_provenance_sha256": "legacy-review-provenance",
        "routing_confidence_threshold": 0.85,
        "automatic_gate": {"passed": True, "reasons": []},
        "ai_review": {
            "decision": "approve",
            "confidence": 0.91,
            "criteria": supported_criteria,
            "evidence": {
                "target": "The successful trajectory supports the strategy.",
                "reference": "The failed trajectory supports the warning.",
            },
            "issues": [],
            "uncertainty_reason": "",
        },
    }
    return record


def verified_experience(experience_id: str = "experience-1") -> dict:
    approved = approved_record(experience_id)
    return {
        "experience_id": experience_id,
        "experience_type": "answer_correctness",
        "reference_evidence": "verified_failure",
        "reference_failure_types": ["boxed_answer_mismatch"],
        "provenance_sha256": approved["provenance_sha256"],
        "source": copy.deepcopy(approved["source"]),
        "student": copy.deepcopy(approved["student"]),
        "target_episode_id": "target-episode",
        "reference_episode_id": "reference-episode",
        "context": "Jordan arranges several packages according to a changing schedule.",
        "trajectory": "A valid response carefully derives each subtotal before the final result.",
        "reference_trajectory": (
            "A failed response changes direction and reaches a conflicting result."
        ),
        "target_verifier": {"reward": 1.0, "expected_answer": "seventeen"},
        "reference_verifier": {
            "reward": 0.0,
            "expected_answer": "seventeen",
            "predicted_answer": "nineteen",
        },
    }


def builder(tokenizer: WhitespaceTokenizer, *, budget: int = 128) -> MemoryBankBuilder:
    return MemoryBankBuilder(
        selector=ApprovedMemorySourceSelector(),
        compiler=MemoryRecordCompiler(
            tokenizer=tokenizer,
            sanitizer=PayloadSanitizer(MemorySanitizerConfig(max_payload_tokens=budget)),
            reasoner_name="Qwen/Qwen2.5-1.5B-Instruct",
            reasoner_revision="student-revision",
            tokenizer_revision="tokenizer-revision",
        ),
    )


class MemoryRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def test_builds_only_the_three_allowlisted_payload_lines(self) -> None:
        approved = approved_record()
        approved["bank"]["evidence"]["target_observation"] = "FORBIDDEN_EVIDENCE_SENTINEL"
        approved["ai_review_gate"]["ai_review"]["pair_assessments"][
            "factually_consistent"
        ]["evidence"] = "FORBIDDEN_REVIEW_SENTINEL"
        result = builder(self.tokenizer).build([approved], [verified_experience()])
        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record.approved_route, "ai_approved")
        self.assertEqual(record.experience_type, "answer_correctness")
        self.assertEqual(record.sanitized_contrast_payload.count("\n"), 2)
        self.assertTrue(record.sanitized_contrast_payload.startswith("When facing:"))
        self.assertIn("\nPrefer:", record.sanitized_contrast_payload)
        self.assertIn("\nAvoid:", record.sanitized_contrast_payload)
        self.assertNotIn("FORBIDDEN_EVIDENCE_SENTINEL", record.sanitized_contrast_payload)
        self.assertNotIn("FORBIDDEN_REVIEW_SENTINEL", record.sanitized_retrieval_key)
        self.assertNotIn("Reconsider only", record.sanitized_contrast_payload)

    def test_numeric_literal_rejects_the_whole_record_without_redaction(self) -> None:
        for literal in ("4", "four"):
            with self.subTest(literal=literal):
                approved = approved_record()
                approved["bank"]["target"]["verification_rule"] = (
                    f"Repeat the check {literal} times."
                )
                result = builder(self.tokenizer).build(
                    [approved], [verified_experience()]
                )
                self.assertEqual(result.records, ())
                self.assertEqual(result.trace[0].status, "rejected_payload")
                self.assertTrue(
                    any(
                        "numeric_or_math_literal" in reason
                        for reason in result.trace[0].reasons
                    )
                )

    def test_long_source_quote_rejects_the_record(self) -> None:
        experience = verified_experience()
        experience["trajectory"] = (
            "Carefully preserve every relation before applying the next supported operation."
        )
        approved = approved_record()
        approved["bank"]["target"]["transferable_decision"] = experience["trajectory"]
        result = builder(self.tokenizer).build([approved], [experience])
        self.assertEqual(result.records, ())
        self.assertTrue(
            any("overlaps_target_trajectory" in reason for reason in result.trace[0].reasons)
        )

    def test_requires_a_supported_pro_review_and_correct_type(self) -> None:
        approved = approved_record()
        approved["ai_review_gate"]["ai_review"]["field_assessments"]["target"][
            "verification_rule"
        ]["status"] = "partially_supported"
        result = builder(self.tokenizer).build([approved], [verified_experience()])
        self.assertEqual(result.records, ())
        self.assertEqual(result.trace[0].status, "rejected_selection")
        self.assertIn(
            "pro_target_verification_rule_not_supported",
            result.trace[0].reasons,
        )

    def test_accepts_the_frozen_legacy_pro_review_with_all_quality_gates(self) -> None:
        result = builder(self.tokenizer).build(
            [legacy_approved_record()],
            [verified_experience()],
        )
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.trace[0].status, "accepted")
        self.assertEqual(
            result.report["accepted_review_profile_counts"],
            {"legacy_independent_criteria_v1": 1},
        )

    def test_legacy_pro_review_fails_closed_on_a_false_criterion(self) -> None:
        approved = legacy_approved_record()
        approved["ai_review_gate"]["ai_review"]["criteria"][
            "factually_consistent"
        ] = False
        result = builder(self.tokenizer).build([approved], [verified_experience()])
        self.assertEqual(result.records, ())
        self.assertIn(
            "legacy_pro_factually_consistent_not_supported",
            result.trace[0].reasons,
        )

    def test_legacy_pro_review_accepts_the_migrated_integrity_audit_shape(self) -> None:
        approved = legacy_approved_record()
        gate = approved["ai_review_gate"]
        gate.pop("automatic_gate")
        gate["deterministic_audit"] = {
            "integrity_passed": True,
            "reasons": ["semantic_warning_fixture"],
            "integrity_reasons": [],
            "semantic_warnings": ["semantic_warning_fixture"],
        }
        result = builder(self.tokenizer).build([approved], [verified_experience()])
        self.assertEqual(len(result.records), 1)

    def test_legacy_pro_review_fails_closed_below_frozen_confidence(self) -> None:
        approved = legacy_approved_record()
        approved["ai_review_gate"]["ai_review"]["confidence"] = 0.84
        result = builder(self.tokenizer).build([approved], [verified_experience()])
        self.assertEqual(result.records, ())
        self.assertIn(
            "legacy_pro_confidence_below_routing_threshold",
            result.trace[0].reasons,
        )

    def test_unknown_pro_review_schema_is_not_shape_inferred(self) -> None:
        approved = legacy_approved_record()
        approved["ai_review_gate"]["schema_version"] = "unregistered-review-schema"
        result = builder(self.tokenizer).build([approved], [verified_experience()])
        self.assertEqual(result.records, ())
        self.assertIn("unsupported_pro_review_schema", result.trace[0].reasons)

    def test_deduplicates_payloads_but_preserves_a_trace(self) -> None:
        first_approved = approved_record("experience-1")
        first_verified = verified_experience("experience-1")
        second_approved = approved_record("experience-2")
        second_verified = verified_experience("experience-2")
        result = builder(self.tokenizer).build(
            [first_approved, second_approved],
            [first_verified, second_verified],
        )
        self.assertEqual(len(result.records), 1)
        self.assertEqual(
            [item.status for item in result.trace],
            ["accepted", "rejected_duplicate_payload"],
        )

    def test_token_budget_is_fail_closed(self) -> None:
        result = builder(self.tokenizer, budget=3).build(
            [approved_record()], [verified_experience()]
        )
        self.assertEqual(result.records, ())
        self.assertIn("payload_exceeds_token_budget", result.trace[0].reasons)

    def test_reasoner_revision_drift_rejects_the_record(self) -> None:
        compiler = MemoryRecordCompiler(
            tokenizer=self.tokenizer,
            sanitizer=PayloadSanitizer(
                MemorySanitizerConfig(max_payload_tokens=128)
            ),
            reasoner_name="Qwen/Qwen2.5-1.5B-Instruct",
            reasoner_revision="different-revision",
            tokenizer_revision="tokenizer-revision",
        )
        result = MemoryBankBuilder(
            selector=ApprovedMemorySourceSelector(),
            compiler=compiler,
        ).build([approved_record()], [verified_experience()])
        self.assertEqual(result.records, ())
        self.assertIn(
            "reasoner_revision_differs_from_phase1_student",
            result.trace[0].reasons,
        )

    def test_independent_artifact_audit_detects_payload_tampering(self) -> None:
        result = builder(self.tokenizer).build(
            [approved_record()], [verified_experience()]
        )
        record = result.records[0]
        auditor = MemoryArtifactAuditor(
            tokenizer=self.tokenizer,
            expected_token_budget=128,
        )
        self.assertEqual(auditor.assert_valid([record])["status"], "passed")
        self.assertEqual(MemoryRecord.from_dict(record.to_dict()), record)
        tampered = replace(
            record,
            sanitized_contrast_payload=record.sanitized_contrast_payload + " 7",
        )
        audit = auditor.audit([tampered])
        self.assertEqual(audit["status"], "failed")
        reasons = audit["violations"][0]["reasons"]
        self.assertIn("payload_numeric_or_math_literal", reasons)
        self.assertIn("payload_hash_mismatch", reasons)


class BM25AndQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def records(self):
        first = approved_record("relation")
        first_verified = verified_experience("relation")
        second = approved_record("units")
        second_verified = verified_experience("units")
        second["bank"]["target"].update(
            {
                "situation_signature": "A rate problem requires consistent measurement units.",
                "transferable_decision": (
                    "Convert quantities into compatible units before combining them."
                ),
                "verification_rule": "Check that every rate uses the same time basis.",
                "applicability_boundary": "Use this when rates connect unlike units.",
            }
        )
        second["bank"]["reference"].update(
            {
                "competing_pattern": "Combining quantities expressed in incompatible units.",
                "failure_signal": "A rate silently switches its measurement basis.",
                "failure_mechanism": "Incompatible units make the combined rate invalid.",
            }
        )
        result = builder(self.tokenizer).build(
            [first, second], [first_verified, second_verified]
        )
        self.assertEqual(len(result.records), 2)
        return result.records

    def test_bm25_prefers_the_lexically_matching_memory(self) -> None:
        records = self.records()
        index = BM25MemoryIndex(records=records)
        hits = index.search("convert the rate into consistent units", top_k=2)
        self.assertEqual(hits[0].memory_id, records[1].memory_id)
        self.assertGreater(hits[0].score, 0.0)
        self.assertEqual(
            index.to_dict()["schema_version"],
            "experience-memory-bm25-index-v1",
        )
        loaded = BM25MemoryIndex.from_dict(records=records, value=index.to_dict())
        self.assertEqual(
            loaded.search("consistent units")[0].memory_id,
            records[1].memory_id,
        )
        self.assertEqual(index.search("unseen vocabulary"), [])

    def test_query_uses_only_the_fixed_partial_window_and_removes_numbers_from_terms(self) -> None:
        ids = self.tokenizer.encode("old reasoning convert rate into 12 compatible units")
        query = RetrievalQueryBuilder(tokenizer=self.tokenizer).build(
            question="A cyclist travels 30 miles in 2 hours.",
            partial_cot_token_ids=ids,
        )
        self.assertNotIn("30", query.analyzed_terms)
        self.assertNotIn("12", query.analyzed_terms)
        self.assertIn("rate", query.analyzed_terms)
        self.assertEqual(query.query_hash, query.to_dict()["query_hash"])

    def test_query_abstains_after_a_final_answer_marker(self) -> None:
        ids = self.tokenizer.encode("Therefore the final answer is complete")
        with self.assertRaises(MemoryRecordRejected):
            RetrievalQueryBuilder(tokenizer=self.tokenizer).build(
                question="A fixture question.", partial_cot_token_ids=ids
            )


if __name__ == "__main__":
    unittest.main()
