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

from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3_selector import (
    V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
    V3_MARGIN_SELECTOR_POLICY,
    calibration_artifact_sha256,
)


COMPARISON_SCRIPT = (
    PROJECT_ROOT / "scripts" / "compare_v3_retrieval_transforms.py"
)
QUALIFICATION_SCRIPT = (
    PROJECT_ROOT / "scripts" / "qualify_v3_centered_calibration.py"
)


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


class V3RetrievalTransformTests(unittest.TestCase):
    def test_matched_comparison_binds_transform_specific_calibrations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = MemoryRecord(
                memory_id="memory-a",
                source_experience_id="experience-a",
                experience_type="recovery",
                approved_route="target",
                source_logical_split="memory-train",
                phase1_provenance_sha256="phase1",
                review_provenance_sha256="review",
                source_record_sha256="source",
                reasoner_name="reasoner",
                reasoner_revision="model-rev",
                tokenizer_revision="tokenizer-rev",
                sanitized_fields={
                    "when_facing": "a hard arithmetic decomposition",
                    "prefer": "verify each intermediate relation",
                    "avoid": "skip structural checks",
                },
                payload_diagnostics={},
                sanitized_retrieval_key="a hard arithmetic decomposition",
                sanitized_contrast_payload="payload",
                payload_hash="payload-a",
                token_ids_sha256="tokens",
                token_count=3,
                model_sequence_limit=1024,
            )
            memory_path = root / "memory_records.jsonl"
            memory_path.write_text(
                json.dumps(record.to_dict()) + "\n", encoding="utf-8"
            )

            def calibration(
                transform,
                threshold,
                selection_count,
                top_count,
                top1_share,
                gini,
            ):
                value = {
                    "schema_version": V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "status": "passed",
                    "policy": V3_MARGIN_SELECTOR_POLICY,
                    "task_accuracy_used": False,
                    "answer_or_reward_used": False,
                    "source": {
                        "logical_split": "calibration-val",
                        "retrieval_embedding_transform": transform,
                        "retrieval_key_manifest_sha256": "key-bank",
                        "completed_sample_count": 2,
                    },
                    "calibration": {
                        "minimum_top1_top2_margin": threshold,
                        "sample_count": 2,
                        "target_retained_fraction": 0.5,
                    },
                    "first_attempt_selection_concentration": {
                        "selection_count": selection_count,
                        "selected_memory_count": 1,
                        "bank_memory_count": 1,
                        "gini": gini,
                        "normalized_entropy": 0.0,
                        "top1_share": top1_share,
                        "top5_share": 1.0,
                        "top_by_frequency": [{
                            "memory_id": "memory-a",
                            "count": top_count,
                        }],
                    },
                    "requirements": {"answer_blind": True},
                }
                value["artifact_sha256"] = calibration_artifact_sha256(value)
                return value

            calibration_values = {
                "v31": calibration("none", 0.1, 2, 2, 1.0, 0.5),
                "v32": calibration(
                    "key_bank_centroid_center_l2", 0.2, 2, 1, 0.5, 0.0
                ),
            }
            calibration_paths = {}
            for name, value in calibration_values.items():
                path = root / f"{name}_calibration.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                calibration_paths[name] = path

            qualification_output = root / "qualification.json"
            subprocess.run(
                [
                    sys.executable,
                    str(QUALIFICATION_SCRIPT),
                    "--v31-calibration",
                    str(calibration_paths["v31"]),
                    "--v32-calibration",
                    str(calibration_paths["v32"]),
                    "--output",
                    str(qualification_output),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            qualification = json.loads(
                qualification_output.read_text(encoding="utf-8")
            )
            self.assertTrue(qualification["qualified_for_dev_test"])

            common_inputs = {
                name: f"hash-{name}"
                for name in (
                    "split_manifest_sha256",
                    "retrieval_key_manifest_sha256",
                    "side_kv_manifest_sha256",
                    "v3_offline_report_sha256",
                    "e0_final_report_sha256",
                    "risk_artifact_sha256",
                )
            } | {"memory_records_sha256": file_sha256(memory_path)}
            common_inputs["retrieval_key_manifest_sha256"] = "key-bank"

            def profile(name, transform, threshold):
                calibration_value = calibration_values[name]
                value = {
                    "schema_version": (
                        "experience-memory-v3-evaluation-profile-v1"
                    ),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "repository": {
                        "git_revision": f"revision-{name}",
                        "tracked_diff_sha256": "diff",
                        "implementation_set_sha256": f"implementation-{name}",
                    },
                    "logical_split": "dev-test",
                    "dataset_split": "train",
                    "dataset_revision": "dataset",
                    "selected_sample_count": 2,
                    "selected_sample_ids_sha256": "samples",
                    "reasoner": {"model_name": "reasoner"},
                    "prompt_contract": {"name": "prompt"},
                    "generation": {"max_new_tokens": 1024},
                    "hysteresis_gate": {"high": 1.0, "low": 0.5},
                    "alignment": {"record_count": 1},
                    "system_profile": {
                        "layer_number": 24,
                        "retrieval_embedding_transform": transform,
                        "retrieval_abstention_policy": "top1_top2_margin",
                        "retrieval_min_top1_top2_margin": threshold,
                    },
                    "selector_calibration": {
                        "artifact_sha256": calibration_value[
                            "artifact_sha256"
                        ],
                    },
                    "inputs": common_inputs | {
                        "selector_calibration_sha256": file_sha256(
                            calibration_paths[name]
                        ),
                    },
                }
                value["profile_sha256"] = evaluation_profile_sha256(value)
                return value

            profiles = {
                "v31": profile("v31", "none", 0.1),
                "v32": profile(
                    "v32", "key_bank_centroid_center_l2", 0.2
                ),
            }
            profile_paths = {}
            result_paths = {}
            for name, profile_value in profiles.items():
                profile_path = root / f"{name}_profile.json"
                profile_path.write_text(
                    json.dumps(profile_value), encoding="utf-8"
                )
                profile_paths[name] = profile_path
                rows = []
                for index in range(2):
                    diagnostics = {
                        "retrieval_attempt_count": 1,
                        "activation_count": int(name == "v31"),
                        "replacement_count": 0,
                        "duplicate_count": 0,
                        "abstain_count": int(name == "v32"),
                        "rearm_count": 0,
                        "memory_attention_step_count": int(name == "v31"),
                    }
                    row = {
                        "schema_version": (
                            "experience-memory-v3-evaluation-row-v1"
                        ),
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
                                "completion_token_ids_sha256": f"{name}-{index}",
                                "strict_correct": bool(index or name == "v32"),
                                "format_correct": bool(index or name == "v32"),
                                "generated_token_count": 10 + index,
                                "online_diagnostics": diagnostics,
                            },
                        },
                    }
                    row["row_sha256"] = canonical_json_sha256({
                        key: item
                        for key, item in row.items()
                        if key != "created_at"
                    })
                    rows.append(row)
                results_path = root / f"{name}_results.jsonl"
                results_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                result_paths[name] = results_path

            output = root / "comparison.json"
            subprocess.run(
                [
                    sys.executable,
                    str(COMPARISON_SCRIPT),
                    "--v31-results",
                    str(result_paths["v31"]),
                    "--v31-profile",
                    str(profile_paths["v31"]),
                    "--v31-calibration",
                    str(calibration_paths["v31"]),
                    "--v32-results",
                    str(result_paths["v32"]),
                    "--v32-profile",
                    str(profile_paths["v32"]),
                    "--v32-calibration",
                    str(calibration_paths["v32"]),
                    "--memory-records",
                    str(memory_path),
                    "--output",
                    str(output),
                    "--bootstrap-resamples",
                    "100",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["integrity"]["passed"])
            self.assertEqual(
                report["paired_v32_minus_v31"]["strict"][
                    "treatment_correct_control_wrong"
                ],
                1,
            )
            self.assertEqual(
                report["dominant_calibration_memory_payloads"][0][
                    "when_facing"
                ],
                "a hard arithmetic decomposition",
            )
            self.assertTrue(output.with_suffix(".md").is_file())


if __name__ == "__main__":
    unittest.main()
