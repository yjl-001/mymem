from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_bank import V4RepairSignature, v4_card_leakage_reasons
from memgen.experience.v4_2_bank import (
    V42ConstructionProfile,
    V42LocalClusterCandidate,
    V42LocalRepairAtom,
)
from memgen.experience.v4_2_local_direct import (
    V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
    V42LocalDirectProfile,
    build_local_direct_bank_record,
    build_local_direct_manifest,
    build_local_direct_process_card,
    local_direct_implementation_hashes,
    select_joint_medoid,
)
from memgen.experience.v4_2_semantic_bank import (
    V4_2_EVIDENCE_PACKET_SCHEMA,
    V4_2_PAID_PLAN_SCHEMA,
    V4_2_PAID_PREFLIGHT_SCHEMA,
    V42SemanticConstructionProfile,
)
import scripts.build_v4_2_local_direct_bank as builder


ROOT = Path(__file__).resolve().parents[1]


def atom(index: int) -> V42LocalRepairAtom:
    return V42LocalRepairAtom(
        experience_id=f"experience-{index}",
        sample_id=f"sample-{index}",
        source_experience_type="answer_correctness",
        problem_structure="a sequence of dependent quantity updates",
        decision_point="before applying a relation to the current state",
        failure_mechanism="an earlier state is reused after it has changed",
        repair_operator="carry the updated state into each following relation",
        verification_operator="check that every relation consumes the preceding state",
        source_signature_sha256=f"signature-{index}",
    )


def candidate(count: int = 5) -> V42LocalClusterCandidate:
    identifiers = tuple(f"experience-{index}" for index in range(count))
    return V42LocalClusterCandidate(
        candidate_id="candidate-dependent-state",
        member_experience_ids=identifiers,
        representative_experience_ids=identifiers[:5],
        distinct_sample_count=count,
        source_experience_type_distribution=(("answer_correctness", count),),
        mechanism_similarity_min=0.85,
        mechanism_similarity_mean=0.90,
        repair_similarity_min=0.86,
        repair_similarity_mean=0.91,
        applicability_similarity_min=0.75,
        applicability_similarity_mean=0.84,
        joint_similarity_min=0.84,
        joint_similarity_mean=0.89,
        membership_sha256="membership-dependent-state",
    )


def packet(count: int = 5) -> dict:
    value = {
        "candidate_id": "candidate-dependent-state",
        "selection_rank": 1,
        "evidence": [
            {
                "evidence_id": f"experience-{index}",
                "sample_id": f"sample-{index}",
                "source_signature_sha256": f"signature-{index}",
            }
            for index in range(count)
        ],
    }
    value["packet_sha256"] = canonical_json_sha256(value)
    return value


def normalized_rows() -> np.ndarray:
    angles = (-0.40, -0.20, 0.0, 0.20, 0.40)
    return np.asarray(
        [[math.cos(angle), math.sin(angle)] for angle in angles],
        dtype=np.float32,
    )


def signature(index: int) -> V4RepairSignature:
    source = atom(index)
    return V4RepairSignature(
        experience_id=source.experience_id,
        sample_id=source.sample_id,
        experience_type=source.source_experience_type,
        problem_structure=source.problem_structure,
        decision_point=source.decision_point,
        failure_mechanism=source.failure_mechanism,
        repair_operator=source.repair_operator,
        verification_operator=source.verification_operator,
        applicable=True,
        rejection_reason=None,
        source_provenance_sha256=f"provenance-{index}",
    )


class V42LocalDirectBankTests(unittest.TestCase):
    def test_profile_truthfully_marks_the_unreviewed_provisional_route(self) -> None:
        profile = V42LocalDirectProfile()
        self.assertEqual(profile.admission_basis, "authenticated_local_shortlist")
        self.assertFalse(profile.semantic_audit_performed)
        self.assertFalse(profile.independent_review_performed)
        self.assertEqual(profile.minimum_distinct_support, 5)
        self.assertEqual(profile.maximum_evidence_per_candidate, 8)
        self.assertEqual(profile.target_runtime_bank_cap, 32)
        self.assertEqual(profile.injection_layer, 24)
        self.assertEqual(profile.relative_phase_delta, 0)
        with self.assertRaisesRegex(ValueError, "must not claim semantic review"):
            V42LocalDirectProfile(semantic_audit_performed=True)

    def test_joint_medoid_uses_all_three_views_and_is_deterministic(self) -> None:
        identifiers = tuple(f"experience-{index}" for index in range(5))
        rows = normalized_rows()
        selected, diagnostics = select_joint_medoid(
            identifiers,
            atom_index={identifier: index for index, identifier in enumerate(identifiers)},
            embeddings={name: rows for name in ("mechanism", "repair", "applicability")},
            construction_profile=V42ConstructionProfile(),
        )
        self.assertEqual(selected, "experience-2")
        self.assertEqual(diagnostics["selected_experience_id"], selected)
        self.assertEqual(
            diagnostics["diagnostics_sha256"],
            canonical_json_sha256(
                {
                    key: value
                    for key, value in diagnostics.items()
                    if key != "diagnostics_sha256"
                }
            ),
        )

        identical = np.asarray([[1.0, 0.0]] * 5, dtype=np.float32)
        tied, _ = select_joint_medoid(
            tuple(reversed(identifiers)),
            atom_index={identifier: index for index, identifier in enumerate(identifiers)},
            embeddings={
                name: identical for name in ("mechanism", "repair", "applicability")
            },
            construction_profile=V42ConstructionProfile(),
        )
        self.assertEqual(tied, "experience-0")

    def test_joint_medoid_rejects_insufficient_or_duplicate_evidence(self) -> None:
        rows = normalized_rows()
        kwargs = {
            "atom_index": {f"experience-{index}": index for index in range(5)},
            "embeddings": {
                name: rows for name in ("mechanism", "repair", "applicability")
            },
            "construction_profile": V42ConstructionProfile(),
        }
        with self.assertRaisesRegex(ValueError, "five to eight"):
            select_joint_medoid(tuple(f"experience-{index}" for index in range(4)), **kwargs)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            select_joint_medoid(
                ("experience-0", "experience-1", "experience-2", "experience-3", "experience-3"),
                **kwargs,
            )

    def test_process_card_is_a_deterministic_signature_mapping(self) -> None:
        source = atom(2)
        card = build_local_direct_process_card(
            candidate_id="candidate-dependent-state", atom=source
        )
        self.assertEqual(card.target.scope, source.problem_structure)
        self.assertEqual(card.target.diagnosis, source.decision_point)
        self.assertEqual(card.target.action, source.repair_operator)
        self.assertEqual(card.target.verification, source.verification_operator)
        self.assertEqual(card.reference.undesired_pattern, source.failure_mechanism)
        self.assertEqual(card.reference.failure_mechanism, source.failure_mechanism)
        for value in (
            *card.target.to_dict().values(),
            *card.reference.to_dict().values(),
            card.support_summary,
            card.target_reference_distinction,
        ):
            self.assertEqual(v4_card_leakage_reasons(value), ())

    def test_record_preserves_support_roles_and_unreviewed_provenance(self) -> None:
        source_candidate = candidate()
        source_packet = packet()
        medoid = atom(2)
        identifiers = tuple(f"experience-{index}" for index in range(5))
        rows = normalized_rows()
        selected, diagnostics = select_joint_medoid(
            identifiers,
            atom_index={identifier: index for index, identifier in enumerate(identifiers)},
            embeddings={name: rows for name in ("mechanism", "repair", "applicability")},
            construction_profile=V42ConstructionProfile(),
        )
        self.assertEqual(selected, medoid.experience_id)
        record = build_local_direct_bank_record(
            candidate=source_candidate,
            packet=source_packet,
            medoid=medoid,
            medoid_diagnostics=diagnostics,
            profile=V42LocalDirectProfile(),
            source_shortlist={"manifest_sha256": "shortlist-manifest"},
        )
        self.assertEqual(record["quality_tier"], "provisional_local_direct")
        self.assertFalse(record["local_direct_admission"]["semantic_audit_performed"])
        self.assertFalse(record["local_direct_admission"]["independent_review_performed"])
        self.assertEqual(
            record["cluster"]["member_experience_ids"],
            record["construction"]["experience_ids"],
        )
        self.assertEqual(record["construction"]["distinct_sample_count"], 5)
        self.assertEqual(record["construction"]["joint_medoid_experience_id"], selected)
        self.assertEqual(
            record["roles"],
            {
                "target_online_injectable": True,
                "reference_online_injectable": False,
                "auxiliary": None,
            },
        )
        self.assertEqual(record["compiler_contract"]["layer_number"], 24)
        self.assertEqual(
            record["record_sha256"],
            canonical_json_sha256(
                {key: value for key, value in record.items() if key != "record_sha256"}
            ),
        )

    def test_manifest_authenticates_local_direct_records_and_provenance(self) -> None:
        source_candidate = candidate()
        source_packet = packet()
        rows = normalized_rows()
        identifiers = tuple(f"experience-{index}" for index in range(5))
        selected, diagnostics = select_joint_medoid(
            identifiers,
            atom_index={identifier: index for index, identifier in enumerate(identifiers)},
            embeddings={name: rows for name in ("mechanism", "repair", "applicability")},
            construction_profile=V42ConstructionProfile(),
        )
        profile = V42LocalDirectProfile()
        record = build_local_direct_bank_record(
            candidate=source_candidate,
            packet=source_packet,
            medoid=atom(int(selected.rsplit("-", 1)[1])),
            medoid_diagnostics=diagnostics,
            profile=profile,
            source_shortlist={"manifest_sha256": "shortlist-manifest"},
        )
        manifest = build_local_direct_manifest(
            records=(record,),
            profile=profile,
            inputs={
                "repository": {
                    "implementation_sha256": local_direct_implementation_hashes(ROOT)
                }
            },
            source_signature_teacher={"model": "historical-source-teacher"},
        )
        self.assertEqual(
            manifest["schema_version"], V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA
        )
        self.assertFalse(manifest["semantic_review"]["performed"])
        self.assertIsNone(manifest["semantic_review"]["reviewer"])
        self.assertEqual(manifest["external_api_calls_made"], 0)
        self.assertEqual(
            manifest["manifest_sha256"],
            canonical_json_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            ),
        )
        tampered = dict(manifest)
        tampered["qualified_for_online_use"] = True
        self.assertNotEqual(
            tampered["manifest_sha256"],
            canonical_json_sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "manifest_sha256"
                }
            ),
        )

    def test_builder_has_no_credential_or_network_interface(self) -> None:
        source = inspect.getsource(builder)
        self.assertNotIn("DEEPSEEK_API_KEY", source)
        self.assertNotIn("TeacherClient", source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("urllib", source)
        argument_source = inspect.getsource(builder.parse_args)
        self.assertNotIn("api-key", argument_source)
        self.assertNotIn("model", argument_source)
        self.assertNotIn("base-url", argument_source)

    def test_semantic_preflight_authentication_binds_every_evidence(self) -> None:
        source_candidate = candidate()
        source_atoms = {f"experience-{index}": atom(index) for index in range(5)}
        source_signatures = {
            f"experience-{index}": signature(index) for index in range(5)
        }
        source_experiences = {
            f"experience-{index}": {
                "experience_id": f"experience-{index}",
                "sample_id": f"sample-{index}",
                "provenance_sha256": f"provenance-{index}",
            }
            for index in range(5)
        }
        selected_record = {
            "selection_rank": 1,
            "candidate": source_candidate.to_dict(),
            "record_sha256": "selected-record",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic_dir = root / "semantic"
            semantic_dir.mkdir()
            experiences_path = root / "experiences.jsonl"
            experiences_path.write_text("fixtures\n", encoding="utf-8")
            split_path = root / "split.json"
            split_path.write_text("{}\n", encoding="utf-8")
            policy_path = root / "policy.json"
            policy = {
                "schema_version": "memgen-v4.2-semantic-policy-v1",
                "benchmark": "openai/gsm8k",
            }
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            source_signature_info = {"sha256": "signatures", "count": 5}
            local_source_info = {"directory": "local", "candidate_count": 1}
            source_shortlist = {
                "profile_sha256": "shortlist-profile",
                "manifest_sha256": "shortlist-manifest",
                "report_sha256": "shortlist-report",
            }
            semantic_profile = V42SemanticConstructionProfile()
            profile_record = {
                "schema_version": "memgen-v4.2-semantic-profile-record-v1",
                "profile": semantic_profile.to_dict(),
                "profile_sha256": semantic_profile.profile_sha256,
                "inputs": {
                    "experiences": {
                        "path": str(experiences_path),
                        "sha256": file_sha256(experiences_path),
                        "count": 5,
                    },
                    "split_manifest": {"file_sha256": file_sha256(split_path)},
                    "source_signatures": source_signature_info,
                    "local_construction": local_source_info,
                    "source_shortlist": source_shortlist,
                    "semantic_policy": {"file_sha256": file_sha256(policy_path)},
                },
            }
            (semantic_dir / "construction_profile.json").write_text(
                json.dumps(profile_record), encoding="utf-8"
            )
            evidence = []
            for index in range(5):
                source_atom = source_atoms[f"experience-{index}"]
                source_signature = source_signatures[f"experience-{index}"]
                evidence.append(
                    {
                        "evidence_id": source_atom.experience_id,
                        "sample_id": source_atom.sample_id,
                        "source_experience_type": source_atom.source_experience_type,
                        "semantic_signature": {
                            "problem_structure": source_atom.problem_structure,
                            "decision_point": source_atom.decision_point,
                            "failure_mechanism": source_atom.failure_mechanism,
                            "repair_operator": source_atom.repair_operator,
                            "verification_operator": source_atom.verification_operator,
                        },
                        "question": "fixture question",
                        "official_solution": "fixture solution",
                        "verified_success_trajectory": "fixture success",
                        "verified_failure_trajectory": "fixture failure",
                        "target_verifier": {},
                        "reference_verifier": {},
                        "source_provenance_sha256": f"provenance-{index}",
                        "source_signature_sha256": source_signature.signature_sha256,
                        "construction_input_sha256": f"construction-{index}",
                    }
                )
            evidence_packet = {
                "schema_version": V4_2_EVIDENCE_PACKET_SCHEMA,
                "candidate_id": source_candidate.candidate_id,
                "selection_rank": 1,
                "membership_sha256": source_candidate.membership_sha256,
                "semantic_policy_sha256": canonical_json_sha256(policy),
                "evidence_selection_rule": "all_up_to_cap_else_five_diverse_plus_medoid_near",
                "evidence_count": 5,
                "evidence": evidence,
                "source_shortlist_record_sha256": selected_record["record_sha256"],
            }
            evidence_packet["packet_sha256"] = canonical_json_sha256(evidence_packet)
            (semantic_dir / "semantic_evidence_packets.jsonl").write_text(
                json.dumps(evidence_packet) + "\n", encoding="utf-8"
            )
            (semantic_dir / "policy_exclusions.jsonl").write_text("", encoding="utf-8")
            plan = {
                "schema_version": V4_2_PAID_PLAN_SCHEMA,
                "batches": [
                    {
                        "packet_sha256": {
                            source_candidate.candidate_id: evidence_packet["packet_sha256"]
                        }
                    }
                ],
            }
            plan["plan_sha256"] = canonical_json_sha256(plan)
            (semantic_dir / "paid_stage_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            preflight = {
                "schema_version": V4_2_PAID_PREFLIGHT_SCHEMA,
                "status": "semantic_evidence_ready_api_not_started",
                "api_key_read": False,
                "external_api_calls_made": 0,
                "automatic_paid_stage_transition": False,
                "qualified_for_online_use": False,
                "paid_stage_plan_sha256": plan["plan_sha256"],
                "profile_sha256": semantic_profile.profile_sha256,
                "semantic_policy_sha256": canonical_json_sha256(policy),
                "source_shortlist": source_shortlist,
                "source_selected_candidate_count": 1,
                "authenticated_policy_excluded_candidate_count": 0,
                "preflight_excluded_candidate_count": 0,
                "planned_candidate_count": 1,
                "evidence_count": 5,
                "evidence_count_distribution": {"5": 1},
                "policy_excluded_evidence_count": 0,
            }
            preflight["report_sha256"] = canonical_json_sha256(preflight)
            (semantic_dir / "api_preflight_report.json").write_text(
                json.dumps(preflight), encoding="utf-8"
            )

            packets, exclusions, *_ = builder._authenticate_semantic_preflight(
                semantic_dir,
                semantic_policy_path=policy_path,
                selected_records=(selected_record,),
                shortlist_manifest={
                    "profile_sha256": "shortlist-profile",
                    "manifest_sha256": "shortlist-manifest",
                },
                shortlist_preflight={"report_sha256": "shortlist-report"},
                candidates={source_candidate.candidate_id: source_candidate},
                atoms=source_atoms,
                signatures=source_signatures,
                experiences=source_experiences,
                experiences_path=experiences_path,
                split_manifest_path=split_path,
                source_signature_info=source_signature_info,
                local_source_info=local_source_info,
            )
            self.assertEqual(len(packets), 1)
            self.assertEqual(len(packets[0]["evidence"]), 5)
            self.assertEqual(exclusions, ())

            tampered = dict(evidence_packet)
            tampered["evidence"] = list(evidence_packet["evidence"])
            tampered["evidence"][0] = dict(tampered["evidence"][0])
            tampered["evidence"][0]["sample_id"] = "sample-tampered"
            (semantic_dir / "semantic_evidence_packets.jsonl").write_text(
                json.dumps(tampered) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "packet hash mismatch"):
                builder._authenticate_semantic_preflight(
                    semantic_dir,
                    semantic_policy_path=policy_path,
                    selected_records=(selected_record,),
                    shortlist_manifest={
                        "profile_sha256": "shortlist-profile",
                        "manifest_sha256": "shortlist-manifest",
                    },
                    shortlist_preflight={"report_sha256": "shortlist-report"},
                    candidates={source_candidate.candidate_id: source_candidate},
                    atoms=source_atoms,
                    signatures=source_signatures,
                    experiences=source_experiences,
                    experiences_path=experiences_path,
                    split_manifest_path=split_path,
                    source_signature_info=source_signature_info,
                    local_source_info=local_source_info,
                )


if __name__ == "__main__":
    unittest.main()
