from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import scripts.build_v4_2_local_clusters as local_builder

from memgen.experience.v4_bank import V4RepairSignature
from memgen.experience.v4_2_bank import (
    V42ConstructionProfile,
    V42LocalClusterCandidate,
    V42LocalRepairAtom,
    validate_v4_2_cluster_payload,
)
from scripts.build_v4_2_local_clusters import (
    build_local_cluster_candidates,
    build_multiview_positive_edges,
    build_preflight_report,
    build_cluster_plan,
    build_review_packet,
    form_complete_link_groups,
    partition_signatures,
)


def signature(
    suffix: str,
    *,
    sample_id: str | None = None,
    experience_type: str = "answer_correctness",
    applicable: bool = True,
) -> V4RepairSignature:
    return V4RepairSignature(
        experience_id=f"experience-{suffix}",
        sample_id=sample_id or f"sample-{suffix}",
        experience_type=experience_type,
        problem_structure="a multistep quantity relation",
        decision_point="before applying a relation to the running state",
        failure_mechanism="the relation is applied in the wrong direction",
        repair_operator="bind each relation before updating the state",
        verification_operator="check the direction against the stated relation",
        applicable=applicable,
        rejection_reason=None if applicable else "not a reusable process repair",
        source_provenance_sha256=f"source-{suffix}",
    )


def atom(
    suffix: str,
    *,
    sample_id: str | None = None,
    experience_type: str = "answer_correctness",
) -> V42LocalRepairAtom:
    return V42LocalRepairAtom.from_signature(
        signature(
            suffix,
            sample_id=sample_id,
            experience_type=experience_type,
        )
    )


def normalized(rows: list[list[float]]) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    return (value / np.linalg.norm(value, axis=1, keepdims=True)).astype(
        np.float32
    )


def complete_edges(
    atoms: tuple[V42LocalRepairAtom, ...],
    *,
    similarity: float = 0.9,
) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for left_index, left in enumerate(atoms):
        for right in atoms[left_index + 1 :]:
            left_id, right_id = sorted((left.experience_id, right.experience_id))
            result.append(
                {
                    "left_experience_id": left_id,
                    "right_experience_id": right_id,
                    "mechanism_similarity": similarity,
                    "repair_similarity": similarity,
                    "applicability_similarity": similarity,
                    "joint_similarity": similarity,
                }
            )
    return tuple(result)


class V42LocalClusterTests(unittest.TestCase):
    def test_profile_has_no_paid_teacher_and_keeps_runtime_contract(self) -> None:
        profile = V42ConstructionProfile()
        payload = profile.to_dict()
        self.assertEqual(profile.injection_layer, 24)
        self.assertEqual(profile.relative_phase_delta, 0)
        self.assertEqual(profile.min_distinct_support, 5)
        self.assertEqual(profile.representative_count, 5)
        self.assertNotIn("teacher_model", payload)
        self.assertNotIn("base_url", payload)
        self.assertNotIn("api_key", payload)

    def test_profile_rejects_negative_similarity_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            V42ConstructionProfile(mechanism_threshold=-0.1)

    def test_partition_is_deterministic_and_quarantines_only_fixed_roles(self) -> None:
        values = (
            signature("reasoning"),
            signature("format", experience_type="format_compliance"),
            signature("rejected", applicable=False),
        )
        atoms, archive = partition_signatures(values)
        self.assertEqual(
            tuple(item.experience_id for item in atoms),
            ("experience-reasoning",),
        )
        self.assertEqual(
            archive["answer_serialization_experience_ids"],
            ("experience-format",),
        )
        self.assertEqual(
            archive["source_nonapplicable_experience_ids"],
            ("experience-rejected",),
        )

    def test_atom_embeds_three_separate_semantic_views(self) -> None:
        value = atom("view")
        self.assertIn(value.failure_mechanism, value.mechanism_text)
        self.assertNotIn(value.repair_operator, value.mechanism_text)
        self.assertIn(value.repair_operator, value.repair_text)
        self.assertIn(value.verification_operator, value.repair_text)
        self.assertIn(value.problem_structure, value.applicability_text)

    def test_positive_edge_requires_mutual_neighbor_and_all_thresholds(self) -> None:
        atoms = (
            atom("alpha", experience_type="answer_correctness"),
            atom("beta", experience_type="reasoning_failure"),
            atom("gamma"),
        )
        values = normalized([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
        embeddings = {name: values.copy() for name in ("mechanism", "repair", "applicability")}
        edges, diagnostics, _joint = build_multiview_positive_edges(
            atoms,
            embeddings,
            profile=V42ConstructionProfile(neighbor_count=1),
        )
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["left_experience_id"], "experience-alpha")
        self.assertEqual(edges[0]["right_experience_id"], "experience-beta")
        self.assertEqual(diagnostics["mutual_knn_pair_count"], 1)

        blocked = {
            **embeddings,
            "applicability": normalized([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        }
        rejected, rejected_diagnostics, _joint = build_multiview_positive_edges(
            atoms,
            blocked,
            profile=V42ConstructionProfile(neighbor_count=1),
        )
        self.assertEqual(rejected, ())
        self.assertGreaterEqual(
            rejected_diagnostics["below_applicability_threshold"], 1
        )

    def test_complete_link_blocks_a_transitive_chain(self) -> None:
        atoms = (atom("alpha"), atom("beta"), atom("gamma"))
        edges = (
            {
                "left_experience_id": "experience-alpha",
                "right_experience_id": "experience-beta",
                "joint_similarity": 0.9,
            },
            {
                "left_experience_id": "experience-beta",
                "right_experience_id": "experience-gamma",
                "joint_similarity": 0.9,
            },
        )
        groups = form_complete_link_groups(atoms, edges)
        self.assertIn(("experience-alpha", "experience-beta"), groups)
        self.assertIn(("experience-gamma",), groups)

    def test_candidate_requires_five_distinct_samples_and_uses_five_representatives(self) -> None:
        atoms = tuple(
            atom(name)
            for name in ("alpha", "beta", "gamma", "delta", "epsilon")
        )
        embeddings = normalized(
            [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04]]
        )
        candidates, unsupported = build_local_cluster_candidates(
            (tuple(item.experience_id for item in atoms),),
            atoms=atoms,
            edges=complete_edges(atoms),
            joint_embeddings=embeddings,
            profile=V42ConstructionProfile(),
        )
        self.assertEqual(unsupported, ())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].distinct_sample_count, 5)
        self.assertEqual(len(candidates[0].representative_experience_ids), 5)
        representative_samples = {
            next(
                item.sample_id
                for item in atoms
                if item.experience_id == experience_id
            )
            for experience_id in candidates[0].representative_experience_ids
        }
        self.assertEqual(len(representative_samples), 5)

    def test_duplicate_problem_support_is_archived_not_forced(self) -> None:
        atoms = (
            atom("alpha", sample_id="sample-shared"),
            atom("beta", sample_id="sample-shared"),
            atom("gamma"),
            atom("delta"),
            atom("epsilon"),
        )
        candidates, unsupported = build_local_cluster_candidates(
            (tuple(item.experience_id for item in atoms),),
            atoms=atoms,
            edges=(),
            joint_embeddings=normalized([[1.0, 0.0]] * len(atoms)),
            profile=V42ConstructionProfile(),
        )
        self.assertEqual(candidates, ())
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0]["distinct_sample_count"], 4)

    def test_cluster_round_trip_keeps_authenticated_membership(self) -> None:
        atoms = tuple(
            atom(name)
            for name in ("alpha", "beta", "gamma", "delta", "epsilon")
        )
        member_ids = tuple(sorted(item.experience_id for item in atoms))
        candidate = V42LocalClusterCandidate(
            candidate_id="candidate-alpha",
            member_experience_ids=member_ids,
            representative_experience_ids=member_ids,
            distinct_sample_count=5,
            source_experience_type_distribution=(("answer_correctness", 5),),
            mechanism_similarity_min=0.85,
            mechanism_similarity_mean=0.9,
            repair_similarity_min=0.85,
            repair_similarity_mean=0.9,
            applicability_similarity_min=0.75,
            applicability_similarity_mean=0.8,
            joint_similarity_min=0.83,
            joint_similarity_mean=0.88,
            membership_sha256="membership-alpha",
        )
        self.assertEqual(
            validate_v4_2_cluster_payload(candidate.to_dict()), candidate
        )

    def test_preflight_reports_zero_calls_and_bounded_future_request_count(self) -> None:
        atoms = tuple(
            atom(name)
            for name in ("alpha", "beta", "gamma", "delta", "epsilon")
        )
        candidates, _unsupported = build_local_cluster_candidates(
            (tuple(item.experience_id for item in atoms),),
            atoms=atoms,
            edges=complete_edges(atoms),
            joint_embeddings=normalized([[1.0, 0.0]] * len(atoms)),
            profile=V42ConstructionProfile(),
        )
        report = build_preflight_report(
            candidates=candidates,
            atoms=atoms,
            profile=V42ConstructionProfile(),
            diagnostics={},
            cluster_plan_sha256="plan-alpha",
        )
        self.assertEqual(report["external_api_calls_made"], 0)
        self.assertFalse(report["api_key_read"])
        self.assertFalse(report["automatic_paid_stage_transition"])
        self.assertEqual(report["planned_initial_synthesis_requests"], 1)
        self.assertTrue(report["within_candidate_guardrail"])
        self.assertEqual(report["cluster_plan_sha256"], "plan-alpha")

    def test_cluster_plan_authenticates_artifacts_and_covers_every_signature(self) -> None:
        atoms = tuple(
            sorted(
                (atom(name) for name in ("alpha", "beta", "gamma", "delta", "epsilon")),
                key=lambda item: item.experience_id,
            )
        )
        edges = complete_edges(atoms)
        candidates, unsupported = build_local_cluster_candidates(
            (tuple(item.experience_id for item in atoms),),
            atoms=atoms,
            edges=edges,
            joint_embeddings=normalized([[1.0, 0.0]] * len(atoms)),
            profile=V42ConstructionProfile(),
        )
        signatures = tuple(
            signature(item.experience_id.removeprefix("experience-"))
            for item in atoms
        ) + (
            signature("format", experience_type="format_compliance"),
            signature("rejected", applicable=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for name in (
                "construction_profile.json",
                "local_atoms.jsonl",
                "multiview_embeddings_manifest.json",
                "mechanism_embeddings.npy",
                "repair_embeddings.npy",
                "applicability_embeddings.npy",
                "positive_edges.jsonl",
                "positive_edge_manifest.json",
                "local_clusters.jsonl",
                "cluster_review_packets.jsonl",
            ):
                (output_dir / name).write_bytes(b"authenticated fixture")
            plan = build_cluster_plan(
                signatures=signatures,
                atoms=atoms,
                signature_archive={
                    "source_nonapplicable_experience_ids": (
                        "experience-rejected",
                    ),
                    "answer_serialization_experience_ids": (
                        "experience-format",
                    ),
                },
                unsupported_groups=unsupported,
                candidates=candidates,
                edge_diagnostics={"positive_edge_count": len(edges)},
                groups=(tuple(item.experience_id for item in atoms),),
                profile=V42ConstructionProfile(),
                source_signature_info={"sha256": "source-signatures"},
                output_dir=output_dir,
            )
        self.assertFalse(plan["qualified_for_online_use"])
        self.assertEqual(plan["external_api_calls"], 0)
        self.assertEqual(plan["diagnostics"]["qualified_cluster_count"], 1)
        self.assertEqual(len(plan["artifacts"]), 10)
        self.assertIn("plan_sha256", plan)

    def test_review_packet_contains_only_five_authenticated_representatives(self) -> None:
        atoms = tuple(
            sorted(
                (atom(name) for name in ("alpha", "beta", "gamma", "delta", "epsilon")),
                key=lambda item: item.experience_id,
            )
        )
        candidates, _unsupported = build_local_cluster_candidates(
            (tuple(item.experience_id for item in atoms),),
            atoms=atoms,
            edges=complete_edges(atoms),
            joint_embeddings=normalized([[1.0, 0.0]] * len(atoms)),
            profile=V42ConstructionProfile(),
        )
        packet = build_review_packet(
            candidates[0],
            atoms_by_id={item.experience_id: item for item in atoms},
        )
        self.assertEqual(len(packet["representatives"]), 5)
        self.assertEqual(
            {item["sample_id"] for item in packet["representatives"]},
            {item.sample_id for item in atoms},
        )
        self.assertIn("representative_evidence_sha256", packet)
        self.assertIn("packet_sha256", packet)

    def test_local_cluster_entrypoint_has_no_teacher_or_credential_access(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_v4_2_local_clusters.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("TeacherClient", source)
        self.assertNotIn("DEEPSEEK_API_KEY", source)
        self.assertNotIn("os.environ", source)
        self.assertNotIn("urllib", source)

    def test_main_writes_complete_artifacts_and_resume_reuses_embeddings(self) -> None:
        signatures = tuple(
            signature(name)
            for name in ("alpha", "beta", "gamma", "delta", "epsilon")
        )
        source_info = {
            "path": "/authenticated/repair_signatures.jsonl",
            "sha256": "source-signatures",
            "profile_path": "/authenticated/construction_profile.json",
            "profile_file_sha256": "source-profile",
            "profile_sha256": "source-profile-logical",
            "prompt_version": "memgen-v4-repair-signature-deepseek-v1",
            "teacher": {"model": "deepseek-v4-flash"},
            "count": len(signatures),
            "applicable_count": len(signatures),
        }
        values = normalized([[1.0, 0.0]] * len(signatures))
        embeddings = {
            name: values.copy()
            for name in ("mechanism", "repair", "applicability")
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "construction_v4_2_local"
            argv = [
                "build_v4_2_local_clusters.py",
                "--experiences",
                "experiences.jsonl",
                "--split-manifest",
                "split_manifest.json",
                "--source-signatures",
                "repair_signatures.jsonl",
                "--source-construction-profile",
                "source_profile.json",
                "--output-dir",
                str(output_dir),
            ]
            common_patches = (
                patch.object(local_builder, "_validate_split_manifest", return_value={}),
                patch.object(local_builder, "load_v4_experiences", return_value=({},)),
                patch.object(
                    local_builder,
                    "load_authenticated_signatures",
                    return_value=(signatures, source_info),
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2], patch.object(
                local_builder, "embed_view_texts", return_value=embeddings
            ), patch("sys.argv", argv):
                local_builder.main()

            expected = {
                "construction_profile.json",
                "local_atoms.jsonl",
                "mechanism_embeddings.npy",
                "repair_embeddings.npy",
                "applicability_embeddings.npy",
                "multiview_embeddings_manifest.json",
                "positive_edges.jsonl",
                "positive_edge_manifest.json",
                "local_clusters.jsonl",
                "cluster_review_packets.jsonl",
                "local_cluster_plan.json",
                "api_preflight_report.json",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected
            )
            preflight = local_builder.json.loads(
                (output_dir / "api_preflight_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(preflight["qualified_candidate_count"], 1)
            self.assertEqual(preflight["external_api_calls_made"], 0)

            resume_argv = [*argv, "--resume"]
            resume_patches = (
                patch.object(local_builder, "_validate_split_manifest", return_value={}),
                patch.object(local_builder, "load_v4_experiences", return_value=({},)),
                patch.object(
                    local_builder,
                    "load_authenticated_signatures",
                    return_value=(signatures, source_info),
                ),
            )
            with resume_patches[0], resume_patches[1], resume_patches[2], patch.object(
                local_builder,
                "embed_view_texts",
                side_effect=AssertionError("resume must not re-encode"),
            ), patch("sys.argv", resume_argv):
                local_builder.main()


if __name__ == "__main__":
    unittest.main()
