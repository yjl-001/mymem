from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.v3 import ExperienceMemoryV3Profile
from memgen.experience.v3_selector import (
    V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
    V3_MARGIN_SELECTOR_POLICY,
    calibration_artifact_sha256,
    load_margin_selector_calibration,
    retained_margin_threshold,
    selection_concentration,
)
from memgen.experience.phase1 import canonical_json_sha256, file_sha256


CALIBRATION_SCRIPT = PROJECT_ROOT / "scripts" / "calibrate_v3_margin_selector.py"
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts" / "compare_v3_selector_evaluations.py"


def evaluation_profile_sha256(value):
    material = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "repository", "profile_sha256"}
    }
    repository = value.get("repository", {})
    material["code_identity"] = {
        "git_revision": repository.get("git_revision"),
        "tracked_diff_sha256": repository.get("tracked_diff_sha256"),
        "implementation_set_sha256": repository.get(
            "implementation_set_sha256"
        ),
    }
    return canonical_json_sha256(material)


class V3SelectorTests(unittest.TestCase):
    def test_profile_supports_frozen_margin_abstention(self) -> None:
        baseline = ExperienceMemoryV3Profile()
        self.assertEqual(baseline.retrieval_abstention_policy, "disabled")
        self.assertIsNone(baseline.retrieval_min_top1_top2_margin)

        treatment = ExperienceMemoryV3Profile(
            retrieval_abstention_policy="top1_top2_margin",
            retrieval_min_top1_top2_margin=0.004,
        )
        self.assertEqual(
            ExperienceMemoryV3Profile.from_dict(treatment.to_dict()),
            treatment,
        )
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            ExperienceMemoryV3Profile(
                retrieval_abstention_policy="top1_top2_margin",
                retrieval_min_top1_top2_margin=-1e-6,
            )
        with self.assertRaisesRegex(ValueError, "cannot set a margin"):
            ExperienceMemoryV3Profile(
                retrieval_min_top1_top2_margin=0.004,
            )

    def test_threshold_retains_a_deterministic_upper_fraction(self) -> None:
        result = retained_margin_threshold(
            (0.1, 0.2, 0.3, 0.4), target_retained_fraction=0.5
        )
        self.assertEqual(result["threshold"], 0.3)
        self.assertEqual(result["actual_retained_count"], 2)
        tied = retained_margin_threshold(
            (0.1, 0.2, 0.2, 0.4), target_retained_fraction=0.5
        )
        self.assertEqual(tied["threshold"], 0.2)
        self.assertEqual(tied["actual_retained_count"], 3)

    def test_selection_concentration_includes_unselected_bank_entries(self) -> None:
        result = selection_concentration(
            ("memory-a", "memory-a", "memory-b"),
            complete_memory_ids=("memory-a", "memory-b", "memory-c"),
        )
        self.assertEqual(result["selection_count"], 3)
        self.assertEqual(result["selected_memory_count"], 2)
        self.assertAlmostEqual(result["top1_share"], 2 / 3)
        self.assertGreater(result["gini"], 0.0)

    def test_calibration_artifact_is_content_authenticated(self) -> None:
        artifact = {
            "schema_version": V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
            "created_at": "2026-01-01T00:00:00+00:00",
            "status": "passed",
            "policy": V3_MARGIN_SELECTOR_POLICY,
            "task_accuracy_used": False,
            "answer_or_reward_used": False,
            "source": {"logical_split": "calibration-val"},
            "calibration": {"minimum_top1_top2_margin": 0.004},
            "requirements": {"answer_blind": True},
        }
        artifact["artifact_sha256"] = calibration_artifact_sha256(artifact)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            loaded = load_margin_selector_calibration(path)
            self.assertEqual(
                loaded["calibration"]["minimum_top1_top2_margin"], 0.004
            )
            artifact["calibration"]["minimum_top1_top2_margin"] = 0.005
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_margin_selector_calibration(path)

    def test_calibration_script_uses_only_authenticated_first_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key_path = root / "retrieval_key_manifest.json"
            key_manifest = {
                "schema_version": "experience-memory-retrieval-key-bank-v1",
                "records": [
                    {"memory_id": "memory-a"},
                    {"memory_id": "memory-b"},
                ],
            }
            key_manifest["manifest_sha256"] = canonical_json_sha256(
                key_manifest
            )
            key_path.write_text(json.dumps(key_manifest), encoding="utf-8")
            profile = {
                "schema_version": "experience-memory-v3-evaluation-profile-v1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "repository": {
                    "git_revision": "revision",
                    "tracked_diff_sha256": "diff",
                    "implementation_set_sha256": "implementation",
                },
                "logical_split": "calibration-val",
                "selected_sample_count": 2,
                "system_profile": {
                    "retrieval_abstention_policy": "disabled",
                },
                "inputs": {
                    "retrieval_key_manifest_sha256": file_sha256(key_path),
                },
            }
            profile["profile_sha256"] = evaluation_profile_sha256(profile)
            profile_path = root / "run_profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            rows = []
            for index, margin in enumerate((0.1, 0.4)):
                row = {
                    "schema_version": "experience-memory-v3-evaluation-row-v1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "profile_sha256": profile["profile_sha256"],
                    "sample_id": f"sample-{index}",
                    "conditions": {
                        "v3": {
                            "strict_correct": bool(index),
                            "runtime_trace": {
                                "retrieval_attempts": [{
                                    "selected_memory_id": "memory-a",
                                    "retrieval_decision": {
                                        "status": "selected",
                                        "query": {
                                            "top1_top2_margin": margin,
                                        },
                                    },
                                }],
                            },
                        },
                    },
                }
                row["row_sha256"] = canonical_json_sha256({
                    key: value
                    for key, value in row.items()
                    if key != "created_at"
                })
                rows.append(row)
            results_path = root / "results.jsonl"
            results_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output_path = root / "selector.json"
            subprocess.run(
                [
                    sys.executable,
                    str(CALIBRATION_SCRIPT),
                    "--results",
                    str(results_path),
                    "--run-profile",
                    str(profile_path),
                    "--retrieval-key-manifest",
                    str(key_path),
                    "--output",
                    str(output_path),
                    "--target-retained-fraction",
                    "0.5",
                    "--minimum-triggered-samples",
                    "2",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            artifact = load_margin_selector_calibration(output_path)
            self.assertEqual(
                artifact["calibration"]["minimum_top1_top2_margin"], 0.4
            )
            self.assertFalse(artifact["task_accuracy_used"])
            self.assertTrue(output_path.with_suffix(".md").is_file())

    def test_comparison_script_requires_matched_vanilla_and_pairs_v3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common_inputs = {
                name: f"hash-{name}"
                for name in (
                    "split_manifest_sha256",
                    "memory_records_sha256",
                    "retrieval_key_manifest_sha256",
                    "side_kv_manifest_sha256",
                    "v3_offline_report_sha256",
                    "e0_final_report_sha256",
                    "risk_artifact_sha256",
                )
            }

            def profile(policy, threshold=None):
                value = {
                    "schema_version": "experience-memory-v3-evaluation-profile-v1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "repository": {
                        "git_revision": f"revision-{policy}",
                        "tracked_diff_sha256": "diff",
                        "implementation_set_sha256": f"implementation-{policy}",
                    },
                    "logical_split": "dev-test",
                    "dataset_split": "train",
                    "dataset_revision": "dataset",
                    "selected_sample_count": 2,
                    "selected_sample_ids_sha256": "samples",
                    "prompt_contract": {"name": "fixture"},
                    "system_profile": {
                        "retrieval_abstention_policy": policy,
                        "retrieval_min_top1_top2_margin": threshold,
                    },
                    "inputs": common_inputs,
                }
                if policy == "top1_top2_margin":
                    value["selector_calibration"] = {
                        "artifact_sha256": "selector-artifact",
                        "task_accuracy_used": False,
                        "answer_or_reward_used": False,
                        "calibration": {
                            "minimum_top1_top2_margin": threshold,
                        },
                    }
                value["profile_sha256"] = evaluation_profile_sha256(value)
                return value

            baseline_profile = profile("disabled")
            margin_profile = profile("top1_top2_margin", 0.3)

            def rows(profile_value, strict_values, *, margin):
                values = []
                for index, strict in enumerate(strict_values):
                    diagnostics = {
                        "retrieval_attempt_count": 1,
                        "activation_count": int(not margin or index == 1),
                        "replacement_count": 0,
                        "duplicate_count": 0,
                        "abstain_count": int(margin and index == 0),
                        "rearm_count": 0,
                        "memory_attention_step_count": int(
                            not margin or index == 1
                        ),
                    }
                    value = {
                        "schema_version": "experience-memory-v3-evaluation-row-v1",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "profile_sha256": profile_value["profile_sha256"],
                        "sample_id": f"sample-{index}",
                        "conditions": {
                            "vanilla": {
                                "completion_token_ids_sha256": f"vanilla-{index}",
                                "strict_correct": bool(index),
                                "format_correct": bool(index),
                            },
                            "v3": {
                                "completion_token_ids_sha256": (
                                    f"{'margin' if margin else 'baseline'}-{index}"
                                ),
                                "strict_correct": strict,
                                "format_correct": strict,
                                "generated_token_count": 10 + index,
                                "online_diagnostics": diagnostics,
                            },
                        },
                    }
                    value["row_sha256"] = canonical_json_sha256({
                        key: item
                        for key, item in value.items()
                        if key != "created_at"
                    })
                    values.append(value)
                return values

            baseline_rows = rows(
                baseline_profile, (False, True), margin=False
            )
            margin_rows = rows(margin_profile, (True, True), margin=True)
            paths = {
                "baseline_profile": root / "baseline_profile.json",
                "margin_profile": root / "margin_profile.json",
                "baseline_results": root / "baseline_results.jsonl",
                "margin_results": root / "margin_results.jsonl",
            }
            paths["baseline_profile"].write_text(
                json.dumps(baseline_profile), encoding="utf-8"
            )
            paths["margin_profile"].write_text(
                json.dumps(margin_profile), encoding="utf-8"
            )
            paths["baseline_results"].write_text(
                "".join(json.dumps(row) + "\n" for row in baseline_rows),
                encoding="utf-8",
            )
            paths["margin_results"].write_text(
                "".join(json.dumps(row) + "\n" for row in margin_rows),
                encoding="utf-8",
            )
            output_path = root / "comparison.json"
            subprocess.run(
                [
                    sys.executable,
                    str(COMPARISON_SCRIPT),
                    "--baseline-results",
                    str(paths["baseline_results"]),
                    "--baseline-profile",
                    str(paths["baseline_profile"]),
                    "--margin-results",
                    str(paths["margin_results"]),
                    "--margin-profile",
                    str(paths["margin_profile"]),
                    "--output",
                    str(output_path),
                    "--bootstrap-resamples",
                    "100",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            comparison = json.loads(output_path.read_text(encoding="utf-8"))
            strict = comparison["paired_margin_minus_baseline"]["strict"]
            self.assertTrue(comparison["integrity"]["passed"])
            self.assertEqual(strict["treatment_correct_control_wrong"], 1)
            self.assertEqual(strict["treatment_wrong_control_correct"], 0)
            self.assertEqual(comparison["mechanism"]["margin"]["abstain_count"], 1)


if __name__ == "__main__":
    unittest.main()
