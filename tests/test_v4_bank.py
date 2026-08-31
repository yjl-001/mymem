from __future__ import annotations

from types import SimpleNamespace
import unittest

from memgen.experience.phase1 import canonical_json_sha256, text_sha256
from memgen.experience.v4_bank import (
    V4_CLUSTER_PLAN_SCHEMA,
    V4CardReview,
    V4ConstructionProfile,
    V4ProcessCard,
    V4ReferenceProcessCard,
    V4RepairCluster,
    V4RepairSignature,
    V4TargetProcessCard,
    build_v4_bank_manifest,
    build_v4_bank_record,
    parse_v4_cluster_plan,
    parse_v4_process_card,
    parse_v4_repair_signature,
)
from scripts.build_v4_repair_bank import (
    _rejected_signature_after_invalid_response,
    attach_official_solutions,
    cluster_messages,
    process_card_messages,
    repair_signature_messages,
)


def signature(suffix: str, *, experience_type: str = "answer_correctness") -> V4RepairSignature:
    return V4RepairSignature(
        experience_id=f"experience-{suffix}",
        sample_id=f"sample-{suffix}",
        experience_type=experience_type,
        problem_structure="a sequence of dependent quantity updates",
        decision_point="before combining changes into the tracked quantity",
        failure_mechanism="an intermediate state is discarded too early",
        repair_operator="carry each intermediate state into the next update",
        verification_operator="check every update against the preceding state",
        applicable=True,
        rejection_reason=None,
        source_provenance_sha256=f"provenance-{suffix}",
    )


def cluster() -> V4RepairCluster:
    members = tuple(f"experience-{suffix}" for suffix in "abcde")
    return V4RepairCluster(
        cluster_key="dependent-state-update",
        title="Preserve dependent state updates",
        experience_type="answer_correctness",
        failure_mechanism="an intermediate state is discarded too early",
        repair_operator="carry each intermediate state into the next update",
        scope_summary="sequential changes depend on the immediately preceding state",
        member_experience_ids=members,
        representative_experience_ids=members,
    )


def card() -> V4ProcessCard:
    return V4ProcessCard(
        cluster_key="dependent-state-update",
        target=V4TargetProcessCard(
            scope="Use this when later changes depend on an updated quantity.",
            diagnosis="The reasoning skips a state that later operations depend on.",
            action="Write each intermediate state before applying the next change.",
            verification="Check that every operation consumes the immediately prior state.",
            do_not_use_when="Do not use this when all changes share an unchanged base.",
        ),
        reference=V4ReferenceProcessCard(
            undesired_pattern="The reasoning combines changes while dropping intermediate state.",
            failure_signal="A later operation is applied to an earlier quantity.",
            failure_mechanism="Discarded state breaks the dependency between operations.",
            contrast_boundary="The target preserves each state before continuing.",
        ),
        support_summary="Independent constructions share the same dependency error.",
        target_reference_distinction="The target preserves state while the reference drops it.",
    )


def review(*, approve: bool = True) -> V4CardReview:
    return V4CardReview(
        cluster_key="dependent-state-update",
        target_grounded=approve,
        reference_grounded=approve,
        process_only=approve,
        target_reference_distinct=approve,
        transferable=approve,
        leakage_free=approve,
        approve=approve,
        evidence="The process contrast is supported across the construction set.",
        issues=() if approve else ("The proposed process is not grounded.",),
    )


class V4BankContractTests(unittest.TestCase):
    def test_invalid_teacher_response_becomes_nonapplicable_audit_record(self) -> None:
        value = _rejected_signature_after_invalid_response(
            {
                "experience_id": "experience-a",
                "sample_id": "sample-a",
                "experience_type": "answer_correctness",
                "source_provenance_sha256": "provenance-a",
            }
        )
        self.assertFalse(value.applicable)
        self.assertIn("outside the instance free schema", value.rejection_reason)
        self.assertEqual(
            value.repair_operator,
            "exclude the unvalidated example from repair clustering",
        )

    def test_profile_freezes_deepseek_and_layer(self) -> None:
        profile = V4ConstructionProfile()
        self.assertEqual(profile.teacher_model, "deepseek-v4-flash")
        self.assertEqual(profile.injection_layer, 24)
        self.assertEqual(profile.min_construction_examples, 5)
        self.assertEqual(profile.max_construction_examples, 10)
        self.assertEqual(profile.thinking, "disabled")
        with self.assertRaisesRegex(ValueError, "deepseek-v4-flash"):
            V4ConstructionProfile(teacher_model="gpt-5.5-thinking")
        with self.assertRaisesRegex(ValueError, "layer 24"):
            V4ConstructionProfile(injection_layer=28)

    def test_signature_parser_binds_identity_and_rejects_literals(self) -> None:
        payload = {
            "problem_structure": "a proportional comparison",
            "decision_point": "before choosing the direction of the ratio",
            "failure_mechanism": "the compared roles are reversed",
            "repair_operator": "name both roles before forming the ratio",
            "verification_operator": "check the ratio direction against the question",
            "applicable": True,
            "rejection_reason": None,
        }
        value = parse_v4_repair_signature(
            payload,
            experience_id="experience-a",
            sample_id="sample-a",
            experience_type="answer_correctness",
            source_provenance_sha256="provenance-a",
        )
        self.assertEqual(value.experience_id, "experience-a")
        payload["repair_operator"] = "multiply by 7"
        with self.assertRaisesRegex(ValueError, "instance-specific"):
            parse_v4_repair_signature(
                payload,
                experience_id="experience-a",
                sample_id="sample-a",
                experience_type="answer_correctness",
                source_provenance_sha256="provenance-a",
            )

    def test_cluster_requires_five_distinct_problems_and_exact_coverage(self) -> None:
        signatures = tuple(signature(suffix) for suffix in "abcdef")
        payload = {
            "schema_version": V4_CLUSTER_PLAN_SCHEMA,
            "clusters": [
                {
                    "cluster_key": "dependent-state-update",
                    "title": "Preserve dependent state updates",
                    "failure_mechanism": "an intermediate state is discarded too early",
                    "repair_operator": "carry each state into the next update",
                    "scope_summary": "sequential changes depend on the preceding state",
                    "member_experience_ids": [
                        f"experience-{suffix}" for suffix in "abcde"
                    ],
                    "representative_experience_ids": [
                        f"experience-{suffix}" for suffix in "abcde"
                    ],
                }
            ],
            "rejected_experience_ids": ["experience-f"],
        }
        clusters, rejected = parse_v4_cluster_plan(payload, signatures=signatures)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(rejected, ("experience-f",))
        payload["rejected_experience_ids"] = []
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            parse_v4_cluster_plan(payload, signatures=signatures)
        payload["schema_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "cluster-plan"):
            parse_v4_cluster_plan(payload, signatures=signatures)

    def test_cluster_cannot_mix_experience_types(self) -> None:
        signatures = tuple(signature(suffix) for suffix in "abcd") + (
            signature("e", experience_type="format_compliance"),
        )
        payload = {
            "schema_version": V4_CLUSTER_PLAN_SCHEMA,
            "clusters": [
                {
                    "cluster_key": "mixed-cluster",
                    "title": "Mixed cluster",
                    "failure_mechanism": "a process mistake",
                    "repair_operator": "apply a process repair",
                    "scope_summary": "a broad process scope",
                    "member_experience_ids": [
                        f"experience-{suffix}" for suffix in "abcde"
                    ],
                    "representative_experience_ids": [
                        f"experience-{suffix}" for suffix in "abcde"
                    ],
                }
            ],
            "rejected_experience_ids": [],
        }
        with self.assertRaisesRegex(ValueError, "mixes experience types"):
            parse_v4_cluster_plan(payload, signatures=signatures)

    def test_process_card_is_process_only_and_reference_is_noninjectable(self) -> None:
        value = card()
        self.assertIn("Action:", value.target.descriptor)
        self.assertIn("Undesired pattern:", value.reference.descriptor)
        payload = value.to_dict()
        payload["target"]["action"] = "Return \\boxed{12}."
        with self.assertRaisesRegex(ValueError, "instance-specific"):
            parse_v4_process_card(payload, cluster_key=value.cluster_key)

        signatures = tuple(signature(suffix) for suffix in "abcde")
        profile = V4ConstructionProfile()
        record = build_v4_bank_record(
            cluster=cluster(),
            card=value,
            review=review(),
            signatures=signatures,
            construction_input_sha256="construction-sha",
            profile=profile,
        )
        self.assertTrue(record["roles"]["target_online_injectable"])
        self.assertFalse(record["roles"]["reference_online_injectable"])
        self.assertIsNone(record["roles"]["auxiliary"])
        self.assertEqual(record["compiler_contract"]["layer_number"], 24)

        manifest = build_v4_bank_manifest(
            records=(record,),
            profile=profile,
            inputs={"experiences_sha256": "input-sha"},
            teacher={"model": "deepseek-v4-flash"},
        )
        self.assertFalse(manifest["qualified_for_online_use"])
        self.assertEqual(manifest["record_count"], 1)

    def test_review_must_match_component_flags_and_issues(self) -> None:
        self.assertTrue(review().approve)
        self.assertFalse(review(approve=False).approve)
        with self.assertRaisesRegex(ValueError, "approval is inconsistent"):
            V4CardReview(
                cluster_key="dependent-state-update",
                target_grounded=True,
                reference_grounded=True,
                process_only=True,
                target_reference_distinct=True,
                transferable=True,
                leakage_free=True,
                approve=True,
                evidence="The card appears grounded.",
                issues=("But an issue was reported.",),
            )

    def test_official_solution_join_authenticates_question_and_answer(self) -> None:
        question = "A training question?"
        answer = "A worked reference solution. #### 4"
        sample = {
            "sample_id": "sample-a",
            "logical_split": "bank-source",
            "dataset_split": "train",
            "source_index": 0,
            "question_sha256": text_sha256(question),
            "answer_sha256": text_sha256(answer),
        }
        manifest = {
            "dataset": {"revision": "main", "train_fingerprint": "fingerprint"},
            "samples": [sample],
        }

        class FakeDataset(list):
            _fingerprint = "fingerprint"

        experience = {
            "experience_id": "experience-a",
            "sample_id": "sample-a",
            "experience_type": "answer_correctness",
            "context": question,
            "trajectory": "A verified successful trajectory.",
            "reference_trajectory": "A verified failed trajectory.",
            "target_verifier": {"reward": 1.0},
            "reference_verifier": {"reward": 0.0},
            "provenance_sha256": "provenance-a",
            "source": {"source_index": 0},
        }
        joined = attach_official_solutions(
            (experience,),
            split_manifest=manifest,
            dataset_revision="main",
            dataset=FakeDataset([{"question": question, "answer": answer}]),
        )
        self.assertEqual(joined[0]["official_solution"], answer)
        self.assertEqual(
            joined[0]["construction_input_sha256"],
            canonical_json_sha256(
                {
                    key: item
                    for key, item in joined[0].items()
                    if key != "construction_input_sha256"
                }
            ),
        )

    def test_prompts_encode_mi_adaptation_without_old_memory_input(self) -> None:
        example = {
            "experience_id": "experience-a",
            "sample_id": "sample-a",
            "experience_type": "answer_correctness",
            "question": "question text",
            "official_solution": "official solution",
            "verified_success_trajectory": "successful process",
            "verified_failure_trajectory": "failed process",
            "reference_verifier": {"reward": 0.0},
        }
        signature_prompt = repair_signature_messages(example)
        self.assertIn("official solution", signature_prompt[1]["content"])
        self.assertIn("failed process", signature_prompt[1]["content"])

        signatures = tuple(signature(suffix) for suffix in "abcde")
        clustering_prompt = cluster_messages(signatures)
        self.assertIn(
            "failure mechanism",
            " ".join(clustering_prompt[0]["content"].split()),
        )
        self.assertNotIn("when_facing", clustering_prompt[1]["content"])

        examples_by_id = {
            f"experience-{suffix}": {
                **example,
                "experience_id": f"experience-{suffix}",
                "sample_id": f"sample-{suffix}",
            }
            for suffix in "abcde"
        }
        card_prompt = process_card_messages(
            cluster(),
            signatures_by_id={item.experience_id: item for item in signatures},
            examples_by_id=examples_by_id,
        )
        normalized_card_system = " ".join(card_prompt[0]["content"].split())
        self.assertIn("official solutions", normalized_card_system)
        self.assertIn("undesired process", normalized_card_system)


if __name__ == "__main__":
    unittest.main()
