from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_bank import V4RepairSignature
from memgen.experience.v4_2_bank import (
    V42ConstructionProfile,
    V42LocalClusterCandidate,
    V42LocalRepairAtom,
    V42ShortlistProfile,
)
import scripts.build_v4_2_local_clusters as local_builder
import scripts.select_v4_2_bank_candidates as shortlist_builder


def signature(suffix: str) -> V4RepairSignature:
    return V4RepairSignature(
        experience_id=f"experience-{suffix}",
        sample_id=f"sample-{suffix}",
        experience_type="answer_correctness",
        problem_structure="a multistep quantity relation",
        decision_point="before applying a relation to the running state",
        failure_mechanism="the relation is applied in the wrong direction",
        repair_operator="bind each relation before updating the state",
        verification_operator="check the direction against the stated relation",
        applicable=True,
        rejection_reason=None,
        source_provenance_sha256=f"source-{suffix}",
    )


def atom(suffix: str) -> V42LocalRepairAtom:
    return V42LocalRepairAtom.from_signature(signature(suffix))


def candidate(
    suffix: str,
    *,
    support: int,
    weakest_margin: float = 0.5,
) -> V42LocalClusterCandidate:
    members = tuple(f"experience-{suffix}-{index:02d}" for index in range(support))
    mechanism_min = 0.82 + 0.18 * weakest_margin
    repair_min = 0.82 + 0.18 * weakest_margin
    applicability_min = 0.70 + 0.30 * weakest_margin
    return V42LocalClusterCandidate(
        candidate_id=f"candidate-{suffix}",
        member_experience_ids=members,
        representative_experience_ids=members[:5],
        distinct_sample_count=support,
        source_experience_type_distribution=(("answer_correctness", support),),
        mechanism_similarity_min=mechanism_min,
        mechanism_similarity_mean=min(1.0, mechanism_min + 0.03),
        repair_similarity_min=repair_min,
        repair_similarity_mean=min(1.0, repair_min + 0.03),
        applicability_similarity_min=applicability_min,
        applicability_similarity_mean=min(1.0, applicability_min + 0.03),
        joint_similarity_min=0.85 + 0.10 * weakest_margin,
        joint_similarity_mean=0.88 + 0.08 * weakest_margin,
        membership_sha256=f"membership-{suffix}",
    )


def centroids(
    values: dict[str, list[float]],
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for candidate_id, row in values.items():
        value = np.asarray(row, dtype=np.float32)
        value = value / np.linalg.norm(value)
        result[candidate_id] = {
            name: value.copy()
            for name in shortlist_builder.EMBEDDING_VIEW_NAMES
        }
    return result


class V42ShortlistTests(unittest.TestCase):
    def test_profile_freezes_small_basis_and_has_no_teacher_configuration(self) -> None:
        profile = V42ShortlistProfile()
        payload = profile.to_dict()
        self.assertEqual(profile.preferred_distinct_support, 6)
        self.assertEqual(profile.max_synthesis_candidates, 48)
        self.assertEqual(profile.target_runtime_bank_cap, 32)
        self.assertEqual(profile.synthesis_batch_size, 4)
        self.assertEqual(profile.review_batch_size, 8)
        self.assertEqual(profile.injection_layer, 24)
        self.assertNotIn("teacher_model", payload)
        self.assertNotIn("base_url", payload)
        self.assertNotIn("api_key", payload)

    def test_profile_rejects_runtime_cap_above_synthesis_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "Runtime bank cap"):
            V42ShortlistProfile(
                max_synthesis_candidates=16,
                target_runtime_bank_cap=17,
            )

    def test_quality_uses_the_weakest_normalized_view_margin(self) -> None:
        value = candidate("quality", support=6, weakest_margin=0.4)
        quality, threshold = shortlist_builder.build_candidate_quality(
            (value,),
            construction_profile=V42ConstructionProfile(),
            shortlist_profile=V42ShortlistProfile(),
        )
        row = quality[value.candidate_id]
        self.assertAlmostEqual(
            row["weakest_normalized_minimum_margin"], 0.4, places=6
        )
        self.assertAlmostEqual(threshold, 0.4, places=6)
        self.assertEqual(row["support_tier"], "preferred")

    def test_centroid_nms_keeps_stronger_candidate_without_transitive_merge(self) -> None:
        values = (
            candidate("alpha", support=7, weakest_margin=0.6),
            candidate("beta", support=6, weakest_margin=0.7),
            candidate("gamma", support=5, weakest_margin=0.9),
        )
        profile = V42ShortlistProfile()
        quality, threshold = shortlist_builder.build_candidate_quality(
            values,
            construction_profile=V42ConstructionProfile(),
            shortlist_profile=profile,
        )
        geometry = centroids(
            {
                "candidate-alpha": [1.0, 0.0],
                "candidate-beta": [0.94, 0.342],
                "candidate-gamma": [0.766, 0.643],
            }
        )
        edges, pairs = shortlist_builder.build_redundancy_geometry(
            values,
            geometry,
            construction_profile=V42ConstructionProfile(),
            shortlist_profile=profile,
        )
        selected, decisions = shortlist_builder.select_synthesis_shortlist(
            values,
            quality,
            edges,
            pairs,
            profile=profile,
            minimum_support_cohesion_threshold=threshold,
        )
        self.assertEqual(selected, ("candidate-alpha", "candidate-gamma"))
        self.assertEqual(
            decisions["candidate-beta"]["reason"],
            "candidate_centroid_redundant",
        )
        self.assertEqual(
            decisions["candidate-beta"]["redundant_with_candidate_id"],
            "candidate-alpha",
        )

    def test_minimum_support_requires_median_cohesion(self) -> None:
        values = (
            candidate("preferred", support=6, weakest_margin=0.5),
            candidate("weak", support=5, weakest_margin=0.1),
            candidate("strong", support=5, weakest_margin=0.9),
        )
        profile = V42ShortlistProfile()
        quality, threshold = shortlist_builder.build_candidate_quality(
            values,
            construction_profile=V42ConstructionProfile(),
            shortlist_profile=profile,
        )
        geometry = centroids(
            {
                "candidate-preferred": [1.0, 0.0, 0.0],
                "candidate-weak": [0.0, 1.0, 0.0],
                "candidate-strong": [0.0, 0.0, 1.0],
            }
        )
        edges, pairs = shortlist_builder.build_redundancy_geometry(
            values,
            geometry,
            construction_profile=V42ConstructionProfile(),
            shortlist_profile=profile,
        )
        selected, decisions = shortlist_builder.select_synthesis_shortlist(
            values,
            quality,
            edges,
            pairs,
            profile=profile,
            minimum_support_cohesion_threshold=threshold,
        )
        self.assertEqual(
            selected, ("candidate-preferred", "candidate-strong")
        )
        self.assertEqual(
            decisions["candidate-weak"]["reason"],
            "minimum_support_below_cohesion_quantile",
        )

    def test_synthesis_budget_is_a_hard_cap(self) -> None:
        values = tuple(
            candidate(name, support=support, weakest_margin=margin)
            for name, support, margin in (
                ("alpha", 8, 0.8),
                ("beta", 7, 0.7),
                ("gamma", 6, 0.6),
            )
        )
        profile = V42ShortlistProfile(
            max_synthesis_candidates=2,
            target_runtime_bank_cap=2,
        )
        quality, threshold = shortlist_builder.build_candidate_quality(
            values,
            construction_profile=V42ConstructionProfile(),
            shortlist_profile=profile,
        )
        geometry = centroids(
            {
                "candidate-alpha": [1.0, 0.0, 0.0],
                "candidate-beta": [0.0, 1.0, 0.0],
                "candidate-gamma": [0.0, 0.0, 1.0],
            }
        )
        edges, pairs = shortlist_builder.build_redundancy_geometry(
            values,
            geometry,
            construction_profile=V42ConstructionProfile(),
            shortlist_profile=profile,
        )
        selected, decisions = shortlist_builder.select_synthesis_shortlist(
            values,
            quality,
            edges,
            pairs,
            profile=profile,
            minimum_support_cohesion_threshold=threshold,
        )
        self.assertEqual(selected, ("candidate-alpha", "candidate-beta"))
        self.assertEqual(
            decisions["candidate-gamma"]["reason"],
            "synthesis_candidate_budget_exceeded",
        )

    def test_semantic_packet_strips_provenance_from_model_evidence(self) -> None:
        atoms = tuple(atom(name) for name in ("alpha", "beta", "gamma", "delta", "epsilon"))
        members = tuple(sorted(item.experience_id for item in atoms))
        value = V42LocalClusterCandidate(
            candidate_id="candidate-compact",
            member_experience_ids=members,
            representative_experience_ids=members,
            distinct_sample_count=5,
            source_experience_type_distribution=(("answer_correctness", 5),),
            mechanism_similarity_min=0.9,
            mechanism_similarity_mean=0.92,
            repair_similarity_min=0.9,
            repair_similarity_mean=0.92,
            applicability_similarity_min=0.85,
            applicability_similarity_mean=0.9,
            joint_similarity_min=0.89,
            joint_similarity_mean=0.91,
            membership_sha256="membership-compact",
        )
        by_id = {item.experience_id: item for item in atoms}
        packet = local_builder.build_review_packet(value, atoms_by_id=by_id)
        record = shortlist_builder._semantic_packet(
            value,
            packet,
            {"candidate_id": value.candidate_id},
            {"selection_rank": 1},
        )
        self.assertEqual(len(record["semantic_evidence"]), 5)
        for evidence in record["semantic_evidence"]:
            self.assertNotIn("experience_id", evidence)
            self.assertNotIn("sample_id", evidence)
            self.assertNotIn("source_signature_sha256", evidence)
        self.assertEqual(len(record["representative_provenance"]), 5)

    def test_preflight_counts_only_selected_candidates_and_zero_api_calls(self) -> None:
        profile = V42ShortlistProfile()
        selected = tuple(
            {
                "semantic_evidence": [{"failure_mechanism": "reusable process"}],
            }
            for _index in range(48)
        )
        quality_report = {
            "source_candidate_count": 95,
            "rejected_candidate_count": 47,
            "decision_counts": {"selected": 48, "rejected": 47},
            "minimum_support_cohesion_threshold": 0.5,
            "redundancy_edge_count": 0,
            "report_sha256": "quality-report",
        }
        manifest = {"manifest_sha256": "shortlist-manifest"}
        report = shortlist_builder.build_preflight(
            profile=profile,
            selected_records=selected,
            quality_report=quality_report,
            manifest=manifest,
        )
        self.assertEqual(report["selected_synthesis_candidate_count"], 48)
        self.assertEqual(report["planned_initial_synthesis_requests"], 12)
        self.assertEqual(report["maximum_followup_review_requests"], 6)
        self.assertEqual(report["maximum_total_paid_requests"], 18)
        self.assertEqual(report["minimum_support_cohesion_threshold"], 0.5)
        self.assertEqual(report["redundancy_edge_count"], 0)
        self.assertEqual(report["external_api_calls_made"], 0)
        self.assertFalse(report["api_key_read"])
        self.assertFalse(report["automatic_paid_stage_transition"])

    def test_shortlist_entrypoint_has_no_teacher_or_credential_access(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "select_v4_2_bank_candidates.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("TeacherClient", source)
        self.assertNotIn("DEEPSEEK_API_KEY", source)
        self.assertNotIn("os.environ", source)
        self.assertNotIn("urllib", source)

    def test_end_to_end_shortlist_and_resume_use_authenticated_local_artifacts(self) -> None:
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
        embedding = np.asarray([[1.0, 0.0]] * len(signatures), dtype=np.float32)
        embeddings = {
            name: embedding.copy()
            for name in local_builder.EMBEDDING_VIEW_NAMES
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_dir = root / "local"
            shortlist_dir = root / "shortlist"
            local_argv = [
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
                str(local_dir),
            ]
            with patch.object(
                local_builder, "_validate_split_manifest", return_value={}
            ), patch.object(
                local_builder, "load_v4_experiences", return_value=({},)
            ), patch.object(
                local_builder,
                "load_authenticated_signatures",
                return_value=(signatures, source_info),
            ), patch.object(
                local_builder, "embed_view_texts", return_value=embeddings
            ), patch(
                "sys.argv", local_argv
            ):
                local_builder.main()

            shortlist_argv = [
                "select_v4_2_bank_candidates.py",
                "--local-construction-dir",
                str(local_dir),
                "--output-dir",
                str(shortlist_dir),
            ]
            with patch("sys.argv", shortlist_argv):
                shortlist_builder.main()
            preflight = json.loads(
                (shortlist_dir / "api_preflight_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(preflight["source_candidate_count"], 1)
            self.assertEqual(preflight["selected_synthesis_candidate_count"], 1)
            self.assertEqual(preflight["external_api_calls_made"], 0)

            with patch.object(
                shortlist_builder,
                "candidate_centroids",
                side_effect=AssertionError("completed resume must be reused"),
            ), patch("sys.argv", [*shortlist_argv, "--resume"]):
                shortlist_builder.main()

            audit_dir = root / "audit"
            environment = os.environ.copy()
            environment.pop("DEEPSEEK_API_KEY", None)
            environment.update(
                {
                    "MEMGEN_PYTHON_BIN": sys.executable,
                    "MEMGEN_V4_2_LOCAL_DIR": str(local_dir),
                    "MEMGEN_V4_2_SHORTLIST_DIR": str(shortlist_dir),
                    "MEMGEN_V4_2_AUDIT_DIR": str(audit_dir),
                    "MEMGEN_V4_2_SKIP_SEMANTIC": "1",
                }
            )
            completed = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "test.sh")],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertIn("[v4.2-test] PASS", completed.stdout)
            test_report = json.loads(
                (audit_dir / "v4_2_shortlist_test_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(test_report["status"], "PASS")
            self.assertEqual(
                test_report["summary"]["selected_synthesis_candidate_count"],
                1,
            )
            self.assertTrue(
                test_report["assertions"]["authenticated_resume_passed"]
            )
            self.assertEqual(
                test_report["assertions"]["external_api_calls_made"], 0
            )


if __name__ == "__main__":
    unittest.main()
