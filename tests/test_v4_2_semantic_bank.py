from __future__ import annotations

import json
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_bank import (
    V4CardReview,
    V4ProcessCard,
    V4ReferenceProcessCard,
    V4RepairSignature,
    V4TargetProcessCard,
)
from memgen.experience.v4_2_bank import V42LocalClusterCandidate, V42LocalRepairAtom
from memgen.experience.v4_2_semantic_bank import (
    V4_2_COMBINED_BATCH_SCHEMA,
    V4_2_REVIEW_BATCH_SCHEMA,
    V42SemanticConstructionProfile,
    build_v4_2_semantic_bank_record,
    parse_v4_2_combined_batch,
    parse_v4_2_review_batch,
)
import scripts.build_v4_2_semantic_bank as builder


def signature(index: int) -> V4RepairSignature:
    return V4RepairSignature(
        experience_id=f"experience-{index}",
        sample_id=f"sample-{index}",
        experience_type="answer_correctness",
        problem_structure="a sequence of dependent quantity updates",
        decision_point="before applying the next relation to the current state",
        failure_mechanism="an earlier state is reused after it has changed",
        repair_operator="carry the updated state into each following relation",
        verification_operator="check that every relation consumes the preceding state",
        applicable=True,
        rejection_reason=None,
        source_provenance_sha256=f"provenance-{index}",
    )


def candidate(count: int = 9) -> V42LocalClusterCandidate:
    members = tuple(f"experience-{index}" for index in range(count))
    return V42LocalClusterCandidate(
        candidate_id="candidate-dependent-state",
        member_experience_ids=members,
        representative_experience_ids=members[:5],
        distinct_sample_count=count,
        source_experience_type_distribution=(("answer_correctness", count),),
        mechanism_similarity_min=0.86,
        mechanism_similarity_mean=0.91,
        repair_similarity_min=0.87,
        repair_similarity_mean=0.92,
        applicability_similarity_min=0.78,
        applicability_similarity_mean=0.88,
        joint_similarity_min=0.86,
        joint_similarity_mean=0.91,
        membership_sha256="membership-dependent-state",
    )


def card() -> V4ProcessCard:
    return V4ProcessCard(
        cluster_key="candidate-dependent-state",
        target=V4TargetProcessCard(
            scope="Use this when each relation depends on an updated state.",
            diagnosis="The reasoning reuses a state that has already changed.",
            action="Carry each updated state into the following relation.",
            verification="Check that every relation consumes the preceding state.",
            do_not_use_when="Do not use this when all relations share an unchanged base.",
        ),
        reference=V4ReferenceProcessCard(
            undesired_pattern="The reasoning repeatedly applies relations to an earlier state.",
            failure_signal="A later relation ignores an intervening state update.",
            failure_mechanism="Reusing an earlier state breaks the dependency chain.",
            contrast_boundary="The target preserves every state transition before continuing.",
        ),
        support_summary="Independent examples share the same state dependency failure.",
        target_reference_distinction="The target advances state while the reference reuses old state.",
    )


def combined_payload(*, coherent: bool = True, usable_count: int = 5) -> dict:
    judgments = []
    for index in range(5):
        usable = index < usable_count
        judgments.append(
            {
                "evidence_id": f"experience-{index}",
                "factually_valid": usable,
                "supports_shared_failure_mechanism": usable,
                "supports_shared_repair_operator": usable,
                "supports_shared_verification_operator": usable,
                "rationale": "The trajectory exposes the same state transition.",
                "exclusion_reason": None if usable else "The example does not support the transition.",
            }
        )
    return {
        "candidate_id": "candidate-dependent-state",
        "evidence_judgments": judgments,
        "shared_process_invariant": "Each dependent relation must consume the preceding state.",
        "shared_failure_mechanism": "An earlier state is reused after an intervening update.",
        "shared_repair_operator": "Carry each updated state into the following relation.",
        "shared_verification_operator": "Check each relation against the immediately preceding state.",
        "valid_distinct_support": usable_count,
        "coherent": coherent,
        "rejection_reason": None if coherent else "Fewer than five examples support one transition.",
        "card": card().to_dict() if coherent else None,
    }


def approved_review() -> V4CardReview:
    return V4CardReview(
        cluster_key="candidate-dependent-state",
        target_grounded=True,
        reference_grounded=True,
        process_only=True,
        target_reference_distinct=True,
        transferable=True,
        leakage_free=True,
        approve=True,
        evidence="Every retained pair supports the process contrast.",
        issues=(),
    )


class V42SemanticBankTests(unittest.TestCase):
    def test_profile_freezes_cost_and_runtime_contracts(self) -> None:
        profile = V42SemanticConstructionProfile()
        self.assertEqual(profile.teacher_model, "deepseek-v4-flash")
        self.assertEqual(profile.minimum_valid_distinct_support, 5)
        self.assertEqual(profile.maximum_evidence_per_candidate, 8)
        self.assertEqual(profile.synthesis_batch_size, 4)
        self.assertEqual(profile.review_batch_size, 8)
        self.assertEqual(profile.injection_layer, 24)
        self.assertEqual(profile.relative_phase_delta, 0)
        with self.assertRaisesRegex(ValueError, "five valid examples"):
            V42SemanticConstructionProfile(minimum_valid_distinct_support=4)

    def test_combined_parser_requires_exact_evidence_order_and_five_valid(self) -> None:
        expected = {"candidate-dependent-state": tuple(f"experience-{i}" for i in range(5))}
        payload = {"schema_version": V4_2_COMBINED_BATCH_SCHEMA, "results": [combined_payload()]}
        result = parse_v4_2_combined_batch(payload, expected=expected)
        self.assertTrue(result[0].coherent)
        self.assertEqual(result[0].valid_distinct_support, 5)

        bad_order = combined_payload()
        bad_order["evidence_judgments"] = list(reversed(bad_order["evidence_judgments"]))
        with self.assertRaisesRegex(ValueError, "order or coverage"):
            parse_v4_2_combined_batch(
                {"schema_version": V4_2_COMBINED_BATCH_SCHEMA, "results": [bad_order]},
                expected=expected,
            )
        with self.assertRaisesRegex(ValueError, "fewer than five"):
            parse_v4_2_combined_batch(
                {"schema_version": V4_2_COMBINED_BATCH_SCHEMA, "results": [combined_payload(usable_count=4)]},
                expected=expected,
            )

    def test_evidence_selection_uses_all_eligible_when_cap_reached(self) -> None:
        value = candidate(9)
        atoms = {item.experience_id: V42LocalRepairAtom.from_signature(item) for item in (signature(i) for i in range(9))}
        atom_index = {experience_id: index for index, experience_id in enumerate(value.member_experience_ids)}
        matrix = np.eye(9, dtype=np.float32)
        embeddings = {name: matrix for name in builder.EMBEDDING_VIEW_NAMES}
        selected = builder.select_evidence_ids(
            value,
            atom_index=atom_index,
            atoms=atoms,
            embeddings=embeddings,
            weights={"mechanism": 0.45, "repair": 0.45, "applicability": 0.10},
            excluded_experience_ids={"experience-3"},
        )
        self.assertEqual(len(selected), 8)
        self.assertNotIn("experience-3", selected)

    def test_large_cluster_preserves_five_representatives_then_adds_medoid_near(self) -> None:
        value = candidate(10)
        atoms = {item.experience_id: V42LocalRepairAtom.from_signature(item) for item in (signature(i) for i in range(10))}
        atom_index = {experience_id: index for index, experience_id in enumerate(value.member_experience_ids)}
        rows = np.asarray([[1.0, (index + 1) / 20.0] for index in range(10)], dtype=np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        selected = builder.select_evidence_ids(
            value,
            atom_index=atom_index,
            atoms=atoms,
            embeddings={name: rows for name in builder.EMBEDDING_VIEW_NAMES},
            weights={"mechanism": 0.45, "repair": 0.45, "applicability": 0.10},
            excluded_experience_ids=set(),
        )
        self.assertEqual(selected[:5], value.representative_experience_ids)
        self.assertEqual(len(selected), 8)

    def test_request_packing_uses_rendered_character_limit(self) -> None:
        items = tuple({"candidate_id": f"candidate-{index}", "evidence": []} for index in range(5))
        batches = builder.pack_requests(
            items,
            batch_size=2,
            max_characters=1000,
            message_builder=lambda batch: [{"role": "user", "content": json.dumps(batch)}],
        )
        self.assertEqual(tuple(len(batch) for batch in batches), (2, 2, 1))
        with self.assertRaisesRegex(ValueError, "character guardrail"):
            builder.pack_requests(
                ({"payload": "x" * 100},),
                batch_size=1,
                max_characters=10,
                message_builder=lambda batch: [{"role": "user", "content": json.dumps(batch)}],
            )

    def test_review_requires_exact_coverage_and_consistent_approval(self) -> None:
        payload = {
            "schema_version": V4_2_REVIEW_BATCH_SCHEMA,
            "results": [{"candidate_id": "candidate-dependent-state", **approved_review().to_dict()}],
        }
        result = parse_v4_2_review_batch(payload, expected_candidate_ids=("candidate-dependent-state",))
        self.assertTrue(result[0].approve)
        payload["results"][0]["approve"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            parse_v4_2_review_batch(payload, expected_candidate_ids=("candidate-dependent-state",))

    def test_review_checkpoint_is_normalized_and_resumable(self) -> None:
        synthesis = parse_v4_2_combined_batch(
            {"schema_version": V4_2_COMBINED_BATCH_SCHEMA, "results": [combined_payload()]},
            expected={"candidate-dependent-state": tuple(f"experience-{i}" for i in range(5))},
        )[0]
        packet = {
            "candidate_id": "candidate-dependent-state",
            "packet_sha256": "packet-hash",
            "evidence": [
                {"evidence_id": f"experience-{i}", "sample_id": f"sample-{i}"}
                for i in range(5)
            ],
        }
        response = {
            "schema_version": V4_2_REVIEW_BATCH_SCHEMA,
            "results": [
                {"candidate_id": "candidate-dependent-state", **approved_review().to_dict()}
            ],
        }

        class FakeClient:
            def __init__(self, *, fail: bool = False) -> None:
                self.fail = fail

            def call(self, _messages, *, response_parser, **_kwargs):
                if self.fail:
                    raise AssertionError("resumed review made an API call")
                return response_parser(json.dumps(response))

        args = SimpleNamespace(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            temperature=0.0,
            thinking="disabled",
            resume=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "review.jsonl"
            calls = [0]
            reviews = builder.run_review_stage(
                (packet,),
                {"candidate-dependent-state": synthesis},
                client=FakeClient(),
                checkpoint_path=checkpoint,
                args=args,
                profile=V42SemanticConstructionProfile(),
                call_counter=calls,
            )
            self.assertTrue(reviews["candidate-dependent-state"].approve)
            self.assertEqual(calls, [1])
            args.resume = True
            resumed_calls = [0]
            resumed = builder.run_review_stage(
                (packet,),
                {"candidate-dependent-state": synthesis},
                client=FakeClient(fail=True),
                checkpoint_path=checkpoint,
                args=args,
                profile=V42SemanticConstructionProfile(),
                call_counter=resumed_calls,
            )
        self.assertTrue(resumed["candidate-dependent-state"].approve)
        self.assertEqual(resumed_calls, [0])

    def test_combined_checkpoint_is_validated_and_resumable(self) -> None:
        packet = {
            "candidate_id": "candidate-dependent-state",
            "packet_sha256": "packet-hash",
            "evidence": [
                {"evidence_id": f"experience-{i}", "sample_id": f"sample-{i}"}
                for i in range(5)
            ],
        }
        response = {
            "schema_version": V4_2_COMBINED_BATCH_SCHEMA,
            "results": [combined_payload()],
        }

        class FakeClient:
            def __init__(self, *, fail: bool = False) -> None:
                self.fail = fail

            def call(self, _messages, *, response_parser, **_kwargs):
                if self.fail:
                    raise AssertionError("resumed synthesis made an API call")
                return response_parser(json.dumps(response))

        args = SimpleNamespace(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            temperature=0.0,
            thinking="disabled",
            resume=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "combined.jsonl"
            calls = [0]
            syntheses = builder.run_combined_stage(
                (packet,),
                client=FakeClient(),
                checkpoint_path=checkpoint,
                args=args,
                profile=V42SemanticConstructionProfile(),
                call_counter=calls,
            )
            self.assertTrue(syntheses["candidate-dependent-state"].coherent)
            self.assertEqual(calls, [1])
            args.resume = True
            resumed_calls = [0]
            resumed = builder.run_combined_stage(
                (packet,),
                client=FakeClient(fail=True),
                checkpoint_path=checkpoint,
                args=args,
                profile=V42SemanticConstructionProfile(),
                call_counter=resumed_calls,
            )
        self.assertTrue(resumed["candidate-dependent-state"].coherent)
        self.assertEqual(resumed_calls, [0])

    def test_bank_record_keeps_only_valid_evidence_and_sorts_samples(self) -> None:
        value = candidate(5)
        signatures = {f"experience-{i}": signature(i) for i in range(5)}
        payload = combined_payload()
        payload["evidence_judgments"] = list(reversed(payload["evidence_judgments"]))
        synthesis = parse_v4_2_combined_batch(
            {"schema_version": V4_2_COMBINED_BATCH_SCHEMA, "results": [payload]},
            expected={"candidate-dependent-state": tuple(f"experience-{i}" for i in reversed(range(5)))},
        )[0]
        packet = {
            "packet_sha256": "packet-hash",
            "evidence": [
                {"evidence_id": f"experience-{i}", "sample_id": f"sample-{i}"}
                for i in reversed(range(5))
            ],
        }
        record = build_v4_2_semantic_bank_record(
            candidate=value,
            synthesis=synthesis,
            review=approved_review(),
            evidence_packet=packet,
            signatures=signatures,
            profile=V42SemanticConstructionProfile(),
            source_shortlist={"manifest_sha256": "shortlist-hash"},
            semantic_policy_sha256="policy-hash",
        )
        self.assertEqual(record["construction"]["sample_ids"], sorted(record["construction"]["sample_ids"]))
        self.assertTrue(record["roles"]["target_online_injectable"])
        self.assertFalse(record["roles"]["reference_online_injectable"])
        self.assertEqual(record["compiler_contract"]["layer_number"], 24)
        self.assertEqual(record["record_sha256"], canonical_json_sha256({key: item for key, item in record.items() if key != "record_sha256"}))

    def test_preflight_argument_parsing_never_reads_environment(self) -> None:
        argv = [
            "build_v4_2_semantic_bank.py",
            "--experiences", "experiences.jsonl",
            "--split-manifest", "split.json",
            "--source-signatures", "signatures.jsonl",
            "--source-construction-profile", "source.json",
            "--local-construction-dir", "local",
            "--shortlist-dir", "shortlist",
            "--semantic-policy", "policy.json",
            "--output-dir", "output",
        ]
        self.assertNotIn("environ", inspect.getsource(builder.parse_args))
        with patch.object(sys, "argv", argv):
            args = builder.parse_args()
        self.assertEqual(args.stage, "preflight")
        self.assertFalse(args.approve_paid_stage)
        main_source = inspect.getsource(builder.main)
        self.assertLess(
            main_source.index('if args.stage == "preflight"'),
            main_source.index("os.environ.get(args.api_key_env)"),
        )
        self.assertLess(
            main_source.index("if not args.approve_paid_stage"),
            main_source.index("os.environ.get(args.api_key_env)"),
        )

    def test_paid_endpoint_and_credential_name_are_allowlisted(self) -> None:
        args = SimpleNamespace(
            model="deepseek-v4-flash",
            temperature=0.0,
            thinking="disabled",
            api_key_env="ANOTHER_SECRET",
            base_url="https://api.deepseek.com",
            synthesis_max_tokens=1,
            review_max_tokens=1,
            retries=1,
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            proxy_retry_initial_seconds=1,
            proxy_retry_max_seconds=1,
            proxy_retries=0,
            stage="paid",
            approve_paid_stage=True,
        )
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            builder.validate_cli(args)
        args.api_key_env = "DEEPSEEK_API_KEY"
        args.base_url = "https://example.com"
        with self.assertRaisesRegex(ValueError, "official DeepSeek"):
            builder.validate_cli(args)

    def test_policy_resolves_human_audited_evidence_to_experience_identity(self) -> None:
        value = candidate(5)
        selected = {
            "candidate": value.to_dict(),
            "representative_provenance": [
                {"evidence_id": f"evidence-{i + 1}", "experience_id": f"experience-{i}"}
                for i in range(5)
            ],
        }
        policy = {
            "schema_version": "memgen-v4.2-semantic-policy-v1",
            "benchmark": "openai/gsm8k",
            "source_shortlist_profile_sha256": "profile",
            "source_shortlist_manifest_sha256": "manifest",
            "source_shortlist_report_sha256": "report",
            "candidate_exclusions": [],
            "representative_evidence_exclusions": [
                {"candidate_id": value.candidate_id, "evidence_id": "evidence-3", "reason": "factual contradiction"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            _raw, candidate_exclusions, evidence_exclusions, policy_hash = builder.load_semantic_policy(
                path,
                shortlist_profile_sha256="profile",
                shortlist_manifest_sha256="manifest",
                shortlist_report_sha256="report",
                selected_records=(selected,),
            )
        self.assertEqual(candidate_exclusions, {})
        self.assertEqual(evidence_exclusions[value.candidate_id], {"experience-2": "factual contradiction"})
        self.assertEqual(policy_hash, canonical_json_sha256(policy))


if __name__ == "__main__":
    unittest.main()
