from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
    CLUSTER_MAP_SCHEMA,
    CLUSTER_REDUCE_SCHEMA,
    _bounded_batches,
    _merge_exact_prototypes,
    _parse_json_object,
    _rejected_signature_after_invalid_response,
    _signature_sort_key,
    attach_official_solutions,
    build_cluster_plan_map_reduce,
    cluster_messages,
    cluster_reduce_messages,
    finalize_cluster_payload,
    parse_cluster_map_payload,
    parse_cluster_reduce_payload,
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
        payload["rejected_experience_ids"] = ["experience-a", "experience-f"]
        with self.assertRaisesRegex(ValueError, "also marked rejected"):
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

    def test_cluster_map_reduce_expands_members_and_enforces_coverage(self) -> None:
        signatures = tuple(signature(suffix) for suffix in "abcdef")
        map_payload = {
            "schema_version": CLUSTER_MAP_SCHEMA,
            "cluster_definitions": [
                {
                    "local_cluster_key": "dependent-update",
                    "title": "Preserve dependent updates",
                    "failure_mechanism": "an intermediate state is discarded too early",
                    "repair_operator": "carry each state into the next update",
                    "scope_summary": "later changes depend on the preceding state",
                },
                {
                    "local_cluster_key": "chained-state",
                    "title": "Preserve chained state",
                    "failure_mechanism": "a dependent state is dropped before reuse",
                    "repair_operator": "retain each state for the following update",
                    "scope_summary": "successive changes consume updated state",
                },
            ],
            "assignments": {
                "experience-a": "dependent-update",
                "experience-b": "dependent-update",
                "experience-c": "dependent-update",
                "experience-d": "chained-state",
                "experience-e": "chained-state",
                "experience-f": "chained-state",
            },
        }
        prototypes, rejected = parse_cluster_map_payload(
            map_payload,
            signatures=signatures,
            unit_id="map-fixture",
        )
        self.assertEqual(len(prototypes), 2)
        self.assertEqual(rejected, ())
        unsupported_payload = finalize_cluster_payload(
            prototypes,
            rejected_experience_ids=(),
            signatures=signatures,
        )
        unsupported_clusters, unsupported_rejected = parse_v4_cluster_plan(
            unsupported_payload,
            signatures=signatures,
        )
        self.assertEqual(unsupported_clusters, ())
        self.assertEqual(
            set(unsupported_rejected),
            {item.experience_id for item in signatures},
        )

        reduce_payload = {
            "schema_version": CLUSTER_REDUCE_SCHEMA,
            "cluster_definitions": [
                {
                    "merged_cluster_key": "dependent-state",
                    "title": "Preserve dependent state updates",
                    "failure_mechanism": "an intermediate state is discarded too early",
                    "repair_operator": "carry each state into the next update",
                    "scope_summary": "later changes depend on the preceding state",
                }
            ],
            "assignments": {
                item["prototype_id"]: "dependent-state"
                for item in prototypes
            },
        }
        merged, rejected_prototypes = parse_cluster_reduce_payload(
            reduce_payload,
            prototypes=prototypes,
            unit_id="reduce-fixture",
        )
        self.assertEqual(rejected_prototypes, ())
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["member_experience_ids"]), 6)

        final_payload = finalize_cluster_payload(
            merged,
            rejected_experience_ids=(),
            signatures=signatures,
        )
        clusters, final_rejected = parse_v4_cluster_plan(
            final_payload,
            signatures=signatures,
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].representative_experience_ids), 6)
        self.assertEqual(final_rejected, ())

        del map_payload["assignments"]["experience-a"]
        with self.assertRaisesRegex(ValueError, "assignment coverage mismatch"):
            parse_cluster_map_payload(
                map_payload,
                signatures=signatures,
                unit_id="map-missing",
            )

    def test_cluster_assignment_map_rejects_ambiguous_or_invalid_keys(self) -> None:
        signatures = tuple(signature(suffix) for suffix in "abc")
        base_payload = {
            "schema_version": CLUSTER_MAP_SCHEMA,
            "cluster_definitions": [
                {
                    "local_cluster_key": "dependent-update",
                    "title": "Preserve dependent updates",
                    "failure_mechanism": "an intermediate state is discarded too early",
                    "repair_operator": "carry each state into the next update",
                    "scope_summary": "later changes depend on the preceding state",
                }
            ],
            "assignments": {
                item.experience_id: "dependent-update" for item in signatures
            },
        }

        undefined = json.loads(json.dumps(base_payload))
        undefined["assignments"]["experience-a"] = "undefined-key"
        with self.assertRaisesRegex(ValueError, "undefined cluster keys"):
            parse_cluster_map_payload(
                undefined,
                signatures=signatures,
                unit_id="map-undefined",
            )

        unused = json.loads(json.dumps(base_payload))
        unused["cluster_definitions"].append(
            {
                **unused["cluster_definitions"][0],
                "local_cluster_key": "unused-cluster",
            }
        )
        with self.assertRaisesRegex(ValueError, "unused cluster definitions"):
            parse_cluster_map_payload(
                unused,
                signatures=signatures,
                unit_id="map-unused",
            )

        duplicate_definition = json.loads(json.dumps(base_payload))
        duplicate_definition["cluster_definitions"].append(
            dict(duplicate_definition["cluster_definitions"][0])
        )
        with self.assertRaisesRegex(ValueError, "duplicate cluster key"):
            parse_cluster_map_payload(
                duplicate_definition,
                signatures=signatures,
                unit_id="map-duplicate-definition",
            )

        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            _parse_json_object(
                '{"assignments":{"experience-a":"first",'
                '"experience-a":"second"}}'
            )

    def test_cluster_requests_are_physically_bounded(self) -> None:
        signatures = tuple(signature(f"item-{index}") for index in range(137))
        batches = _bounded_batches(
            signatures,
            batch_size=48,
            key=_signature_sort_key,
        )
        self.assertEqual(sum(len(batch) for batch in batches), len(signatures))
        self.assertEqual([len(batch) for batch in batches], [48, 48, 41])
        self.assertTrue(all(len(batch) <= 48 for batch in batches))
        self.assertLess(len(cluster_messages(batches[0])[1]["content"]), 200_000)

        map_payload = {
            "schema_version": CLUSTER_MAP_SCHEMA,
            "cluster_definitions": [
                {
                    "local_cluster_key": "dependent-update",
                    "title": "Preserve dependent updates",
                    "failure_mechanism": "an intermediate state is discarded too early",
                    "repair_operator": "carry each state into the next update",
                    "scope_summary": "later changes depend on the preceding state",
                }
            ],
            "assignments": {
                item.experience_id: "dependent-update" for item in batches[0]
            },
        }
        prototypes, _ = parse_cluster_map_payload(
            map_payload,
            signatures=batches[0],
            unit_id="bounded-map",
        )
        self.assertLess(
            len(cluster_reduce_messages(prototypes)[1]["content"]),
            200_000,
        )

    def test_process_card_prompt_uses_only_bounded_representatives(self) -> None:
        signatures = tuple(signature(suffix) for suffix in "abcdef")
        value = V4RepairCluster(
            cluster_key="bounded-evidence",
            title="Preserve dependent state updates",
            experience_type="answer_correctness",
            failure_mechanism="an intermediate state is discarded too early",
            repair_operator="carry each state into the next update",
            scope_summary="later changes depend on the preceding state",
            member_experience_ids=tuple(item.experience_id for item in signatures),
            representative_experience_ids=tuple(
                item.experience_id for item in signatures[:5]
            ),
        )
        examples_by_id = {
            item.experience_id: {
                "experience_id": item.experience_id,
                "sample_id": item.sample_id,
                "question": "question text",
                "official_solution": "official solution",
                "verified_success_trajectory": "successful process",
                "verified_failure_trajectory": "failed process",
                "reference_verifier": {"reward": 0.0},
            }
            for item in signatures
        }
        messages = process_card_messages(
            value,
            signatures_by_id={item.experience_id: item for item in signatures},
            examples_by_id=examples_by_id,
        )
        self.assertIn("experience-e", messages[1]["content"])
        self.assertNotIn("experience-f", messages[1]["content"])
        self.assertIn('\"support_count\": 6', messages[1]["content"])

    def test_map_reduce_checkpoints_resume_without_reissuing_requests(self) -> None:
        signatures = tuple(signature(suffix) for suffix in "abcdef")

        def map_payload(member_ids: list[str]) -> dict:
            return {
                "schema_version": CLUSTER_MAP_SCHEMA,
                "cluster_definitions": [
                    {
                        "local_cluster_key": "dependent-update",
                        "title": "Preserve dependent updates",
                        "failure_mechanism": (
                            "an intermediate state is discarded too early"
                        ),
                        "repair_operator": "carry each state into the next update",
                        "scope_summary": "later changes depend on the preceding state",
                    }
                ],
                "assignments": {
                    experience_id: "dependent-update"
                    for experience_id in member_ids
                },
            }

        class FakeClusterClient:
            def __init__(self, responses: list[dict]):
                self.responses = list(responses)
                self.call_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def call(self, _messages, *, response_parser, **_kwargs):
                self.call_count += 1
                return response_parser(
                    json.dumps(self.responses.pop(0), sort_keys=True)
                )

        args = SimpleNamespace(
            cluster_map_batch_size=3,
            cluster_reduce_batch_size=10,
            cluster_max_tokens=8000,
            resume=False,
            model="deepseek-v4-flash",
            base_url="https://api.example.test",
            temperature=0.0,
            thinking="disabled",
        )
        first_map_payload = map_payload(
            ["experience-a", "experience-b", "experience-c"]
        )
        second_map_payload = map_payload(
            ["experience-d", "experience-e", "experience-f"]
        )
        second_map_payload["cluster_definitions"][0].update(
            {
                "local_cluster_key": "chained-state",
                "title": "Preserve chained state",
                "failure_mechanism": "a dependent state is dropped before reuse",
                "repair_operator": "retain each state for the following update",
                "scope_summary": "successive changes consume updated state",
            }
        )
        second_map_payload["assignments"] = {
            experience_id: "chained-state"
            for experience_id in second_map_payload["assignments"]
        }
        first_prototype = parse_cluster_map_payload(
            first_map_payload,
            signatures=signatures[:3],
            unit_id="map-answer_correctness-00000",
        )[0][0]
        second_prototype = parse_cluster_map_payload(
            second_map_payload,
            signatures=signatures[3:],
            unit_id="map-answer_correctness-00001",
        )[0][0]
        post_map_prototypes = _merge_exact_prototypes(
            [first_prototype, second_prototype],
            unit_id="post-map",
        )
        reduce_payload = {
            "schema_version": CLUSTER_REDUCE_SCHEMA,
            "cluster_definitions": [
                {
                    "merged_cluster_key": "dependent-state",
                    "title": "Preserve dependent state updates",
                    "failure_mechanism": "an intermediate state is discarded too early",
                    "repair_operator": "carry each state into the next update",
                    "scope_summary": "later changes depend on the preceding state",
                }
            ],
            "assignments": {
                item["prototype_id"]: "dependent-state"
                for item in post_map_prototypes
            },
        }
        first_client = FakeClusterClient(
            [first_map_payload, second_map_payload, reduce_payload]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "cluster_map_shards.jsonl"
            reduce_path = Path(temp_dir) / "cluster_reduce_batches.jsonl"
            with patch(
                "scripts.build_v4_repair_bank._client",
                return_value=first_client,
            ):
                first_payload, first_diagnostics = build_cluster_plan_map_reduce(
                    signatures,
                    args=args,
                    api_key="fixture-key",
                    map_checkpoint_path=map_path,
                    reduce_checkpoint_path=reduce_path,
                )
            self.assertEqual(first_client.call_count, 3)
            self.assertEqual(first_diagnostics["new_map_request_count"], 2)
            self.assertEqual(first_diagnostics["new_reduce_request_count"], 1)
            self.assertEqual(len(first_payload["clusters"]), 1)

            args.resume = True
            resumed_client = FakeClusterClient([])
            with patch(
                "scripts.build_v4_repair_bank._client",
                return_value=resumed_client,
            ):
                resumed_payload, resumed_diagnostics = build_cluster_plan_map_reduce(
                    signatures,
                    args=args,
                    api_key="fixture-key",
                    map_checkpoint_path=map_path,
                    reduce_checkpoint_path=reduce_path,
                )
            self.assertEqual(resumed_client.call_count, 0)
            self.assertEqual(resumed_diagnostics["new_map_request_count"], 0)
            self.assertEqual(resumed_diagnostics["new_reduce_request_count"], 0)
            self.assertEqual(resumed_payload, first_payload)


if __name__ == "__main__":
    unittest.main()
