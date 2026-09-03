from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_bank import (
    V4CardReview,
    V4ProcessCard,
    V4ReferenceProcessCard,
    V4RepairSignature,
    V4TargetProcessCard,
)
from memgen.experience.v4_1_bank import (
    V41CanonicalRepairAtom,
    V41ClusterAudit,
    V41ConstructionProfile,
    V41PairJudgment,
    V41RepairCluster,
    build_v4_1_bank_manifest,
    build_v4_1_bank_record,
    parse_v4_1_canonical_atom,
)
from scripts.build_v4_1_repair_bank import (
    audit_candidates,
    build_candidate_pairs,
    build_cluster_plan,
    build_exact_seeds,
    form_clique_candidates,
)


def signature(suffix: str, *, experience_type: str = "answer_correctness") -> V4RepairSignature:
    return V4RepairSignature(
        experience_id=f"experience-{suffix}",
        sample_id=f"sample-{suffix}",
        experience_type=experience_type,
        problem_structure="a multistep quantity relation",
        decision_point="before applying a relation to the running state",
        failure_mechanism="the relation is applied in the wrong direction",
        repair_operator="bind each relation before updating the state",
        verification_operator="check the direction against the stated relation",
        applicable=True,
        rejection_reason=None,
        source_provenance_sha256=f"source-{suffix}",
    )


def atom(
    suffix: str,
    *,
    experience_type: str = "answer_correctness",
    memory_role: str = "reasoning_process",
    repair_action: str = "bind the relation before updating the state",
) -> V41CanonicalRepairAtom:
    source = signature(suffix, experience_type=experience_type)
    serialization = memory_role == "answer_serialization"
    return V41CanonicalRepairAtom(
        experience_id=source.experience_id,
        sample_id=source.sample_id,
        source_experience_type=source.experience_type,
        memory_role=memory_role,
        state_scope="answer_serialization" if serialization else "relation_translation",
        mechanism_family="output_representation" if serialization else "wrong_operation_or_direction",
        repair_family="canonicalize_final_answer" if serialization else "verify_operation_direction",
        applicability_family="answer_serialization" if serialization else "algebraic_relations",
        failure_transition=(
            "the final representation violates the required form"
            if serialization
            else "a stated relation is applied in the opposite direction"
        ),
        repair_action=(
            "render the already derived answer in the required form"
            if serialization
            else repair_action
        ),
        applicability_condition=(
            "the reasoning is complete and only representation remains"
            if serialization
            else "a directional relation updates a running state"
        ),
        verification_action=(
            "check final representation without changing the reasoning"
            if serialization
            else "compare the update direction with the stated relation"
        ),
        source_signature_sha256=source.signature_sha256,
        exclusion_reason=(
            "final representation is outside the reasoning process bank"
            if serialization
            else None
        ),
    )


def positive_pair(left: str, right: str) -> V41PairJudgment:
    left, right = sorted((left, right))
    return V41PairJudgment(
        pair_id=f"pair-{left}-{right}",
        left_seed_id=left,
        right_seed_id=right,
        same_failure_mechanism=True,
        same_repair_action=True,
        compatible_applicability=True,
        process_only=True,
        merge=True,
        evidence="both seeds encode the same reasoning transition",
        issues=(),
    )


class V41BankTests(unittest.TestCase):
    def test_profile_freezes_online_contract_and_pinned_embedding(self) -> None:
        profile = V41ConstructionProfile()
        self.assertEqual(profile.injection_layer, 24)
        self.assertEqual(profile.relative_phase_delta, 0)
        self.assertEqual(profile.teacher_model, "deepseek-v4-flash")
        self.assertEqual(profile.embedding_model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(len(profile.embedding_revision), 40)

    def test_serialization_is_quarantined_from_reasoning_role(self) -> None:
        value = atom("alpha", memory_role="answer_serialization")
        self.assertEqual(value.memory_role, "answer_serialization")
        with self.assertRaisesRegex(ValueError, "answer-serialization|quarantined"):
            V41CanonicalRepairAtom(
                **{
                    **value.to_dict(),
                    "memory_role": "reasoning_process",
                    "exclusion_reason": None,
                }
            )

    def test_verified_format_compliance_cannot_be_relabelled_as_reasoning(self) -> None:
        source = signature("format", experience_type="format_compliance")
        reasoning_payload = atom("format").to_dict()
        for field in (
            "experience_id",
            "sample_id",
            "source_experience_type",
            "source_signature_sha256",
            "schema_version",
        ):
            reasoning_payload.pop(field)
        with self.assertRaisesRegex(ValueError, "format-compliance"):
            parse_v4_1_canonical_atom(reasoning_payload, signature=source)

    def test_exact_seed_can_cross_source_experience_types(self) -> None:
        atoms = (
            atom("alpha", experience_type="answer_correctness"),
            atom("beta", experience_type="formatting_failure"),
        )
        embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        seeds = build_exact_seeds(atoms, embeddings)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(
            seeds[0]["source_experience_type_distribution"],
            {"answer_correctness": 1, "formatting_failure": 1},
        )

    def test_candidate_retrieval_is_cross_type_and_deterministic(self) -> None:
        seeds = (
            {
                "seed_id": "seed-alpha",
                "repair_family": "verify_operation_direction",
                "mechanism_family": "wrong_operation_or_direction",
                "centroid": np.asarray([1.0, 0.0], dtype=np.float32),
            },
            {
                "seed_id": "seed-beta",
                "repair_family": "track_running_state",
                "mechanism_family": "wrong_order_or_state",
                "centroid": np.asarray([0.9, 0.1], dtype=np.float32),
            },
        )
        first = build_candidate_pairs(seeds, neighbor_count=1)
        second = build_candidate_pairs(seeds, neighbor_count=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertIn("global_semantic", first[0]["retrieval_sources"])

    def test_clique_rule_blocks_transitive_chain_merge(self) -> None:
        seeds = tuple(
            {
                "seed_id": f"seed-{name}",
                "distinct_sample_count": 1,
                "member_experience_ids": [f"experience-{name}"],
            }
            for name in ("alpha", "beta", "gamma")
        )
        judgments = (
            positive_pair("seed-alpha", "seed-beta"),
            positive_pair("seed-beta", "seed-gamma"),
        )
        candidates = form_clique_candidates(seeds, judgments)
        groups = sorted(tuple(item["seed_ids"]) for item in candidates)
        self.assertEqual(groups, [("seed-alpha", "seed-beta"), ("seed-gamma",)])

    def test_pair_and_audit_decisions_must_match_component_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "merge decision"):
            V41PairJudgment(
                pair_id="pair-alpha",
                left_seed_id="seed-alpha",
                right_seed_id="seed-beta",
                same_failure_mechanism=True,
                same_repair_action=True,
                compatible_applicability=False,
                process_only=True,
                merge=True,
                evidence="the applicability differs",
                issues=(),
            )
        with self.assertRaisesRegex(ValueError, "approval"):
            V41ClusterAudit(
                candidate_id="candidate-alpha",
                coherent=True,
                process_only=True,
                transferable=False,
                serialization_free=True,
                leakage_free=True,
                approve=True,
                title="directional update",
                failure_mechanism="a relation is reversed",
                repair_operator="bind the relation direction before updating",
                scope_summary="directional relations over a running state",
                evidence="the examples share one transition",
                issues=(),
            )

    def test_supported_candidate_requires_and_passes_multi_example_audit(self) -> None:
        names = ("alpha", "beta", "gamma", "delta", "epsilon")
        atoms = tuple(atom(name) for name in names)
        signatures = tuple(signature(name) for name in names)
        embeddings = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04]],
            dtype=np.float32,
        )
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        seeds = build_exact_seeds(atoms, embeddings)
        candidate = {
            "candidate_id": "candidate-alpha",
            "seed_ids": [seeds[0]["seed_id"]],
            "member_experience_ids": [item.experience_id for item in atoms],
        }
        payload = {
            "coherent": True,
            "process_only": True,
            "transferable": True,
            "serialization_free": True,
            "leakage_free": True,
            "approve": True,
            "title": "directional relation update",
            "failure_mechanism": "a relation is applied in the opposite direction",
            "repair_operator": "bind relation direction before updating the state",
            "scope_summary": "directional relations that update a running state",
            "evidence": "independent examples share one process transition",
            "issues": [],
        }

        class FakeClient:
            def call(self, _messages, *, response_parser, **_kwargs):
                return response_parser(json.dumps(payload))

        args = SimpleNamespace(
            resume=False,
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            temperature=0.0,
            thinking="disabled",
        )
        with tempfile.TemporaryDirectory() as directory:
            clusters, diagnostics = audit_candidates(
                (candidate,),
                seeds=seeds,
                atoms=atoms,
                signatures=signatures,
                embeddings=embeddings,
                checkpoint_path=Path(directory) / "audit.jsonl",
                client=FakeClient(),
                args=args,
            )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].representative_experience_ids), 5)
        self.assertEqual(diagnostics["unsupported_reasoning_experience_ids"], [])

    def test_v41_manifest_keeps_the_side_kv_record_contract(self) -> None:
        signatures = tuple(signature(name) for name in ("alpha", "beta", "gamma", "delta", "epsilon"))
        atoms = tuple(atom(name) for name in ("alpha", "beta", "gamma", "delta", "epsilon"))
        audit = V41ClusterAudit(
            candidate_id="candidate-alpha",
            coherent=True,
            process_only=True,
            transferable=True,
            serialization_free=True,
            leakage_free=True,
            approve=True,
            title="directional relation update",
            failure_mechanism="a stated relation is applied in the opposite direction",
            repair_operator="bind the relation direction before updating the state",
            scope_summary="directional relations that update a running state",
            evidence="independent examples share the same process transition",
            issues=(),
        )
        cluster = V41RepairCluster(
            cluster_key="repair-alpha",
            candidate_id=audit.candidate_id,
            title=audit.title,
            failure_mechanism=audit.failure_mechanism,
            repair_operator=audit.repair_operator,
            scope_summary=audit.scope_summary,
            member_experience_ids=tuple(item.experience_id for item in signatures),
            representative_experience_ids=tuple(item.experience_id for item in signatures),
            source_experience_type_distribution=(("answer_correctness", 5),),
            canonical_seed_ids=("seed-alpha",),
            audit_sha256=canonical_json_sha256(audit.to_dict()),
        )
        card = V4ProcessCard(
            cluster_key=cluster.cluster_key,
            target=V4TargetProcessCard(
                scope="a directional relation updates a running state",
                diagnosis="the update direction can be reversed",
                action="bind the relation direction before updating the state",
                verification="compare the update against the stated relation",
                do_not_use_when="the relation has no directional update",
            ),
            reference=V4ReferenceProcessCard(
                undesired_pattern="applying the relation in the opposite direction",
                failure_signal="the updated state contradicts the stated relation",
                failure_mechanism="the transition reverses the required direction",
                contrast_boundary="the target binds direction before the update",
            ),
            support_summary="independent examples share the same directional transition",
            target_reference_distinction="the target preserves rather than reverses direction",
        )
        review = V4CardReview(
            cluster_key=cluster.cluster_key,
            target_grounded=True,
            reference_grounded=True,
            process_only=True,
            target_reference_distinct=True,
            transferable=True,
            leakage_free=True,
            approve=True,
            evidence="all claims are supported by the construction examples",
            issues=(),
        )
        profile = V41ConstructionProfile()
        record = build_v4_1_bank_record(
            cluster=cluster,
            card=card,
            review=review,
            signatures=signatures,
            atoms=atoms,
            construction_input_sha256="construction-alpha",
            profile=profile,
        )
        manifest = build_v4_1_bank_manifest(
            records=(record,),
            profile=profile,
            inputs={
                "repository": {
                    "implementation_sha256": {"implementation": "authenticated"}
                }
            },
            teacher={"model": "deepseek-v4-flash"},
        )
        self.assertEqual(manifest["schema_version"], "memgen-v4-bank-manifest-v2")
        self.assertEqual(record["schema_version"], "memgen-v4-bank-record-v1")
        self.assertEqual(record["construction_version"], "v4.1")

    def test_cluster_plan_archives_serialization_without_forced_assignment(self) -> None:
        reasoning = tuple(atom(name) for name in ("alpha", "beta", "gamma", "delta", "epsilon"))
        serialized = atom("zeta", memory_role="answer_serialization")
        signatures = tuple(
            signature(item.experience_id.removeprefix("experience-"), experience_type=item.source_experience_type)
            for item in (*reasoning, serialized)
        )
        audit = V41ClusterAudit(
            candidate_id="candidate-alpha",
            coherent=True,
            process_only=True,
            transferable=True,
            serialization_free=True,
            leakage_free=True,
            approve=True,
            title="directional relation update",
            failure_mechanism="a relation is reversed",
            repair_operator="bind direction before updating",
            scope_summary="directional updates over a running state",
            evidence="the examples share one process",
            issues=(),
        )
        cluster = V41RepairCluster(
            cluster_key="repair-alpha",
            candidate_id=audit.candidate_id,
            title=audit.title,
            failure_mechanism=audit.failure_mechanism,
            repair_operator=audit.repair_operator,
            scope_summary=audit.scope_summary,
            member_experience_ids=tuple(item.experience_id for item in reasoning),
            representative_experience_ids=tuple(item.experience_id for item in reasoning),
            source_experience_type_distribution=(("answer_correctness", 5),),
            canonical_seed_ids=("seed-alpha",),
            audit_sha256=canonical_json_sha256(audit.to_dict()),
        )
        plan = build_cluster_plan(
            signatures=signatures,
            atoms=(*reasoning, serialized),
            seeds=(),
            pairs=(),
            judgments=(),
            clusters=(cluster,),
            audit_diagnostics={
                "audits": (),
                "unsupported_reasoning_experience_ids": (),
                "audit_rejected_experience_ids": (),
            },
        )
        self.assertEqual(
            plan["archive"]["answer_serialization_experience_ids"],
            [serialized.experience_id],
        )


if __name__ == "__main__":
    unittest.main()
