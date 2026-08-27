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

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.memory import MemoryRecord
from memgen.experience.v3 import (
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_PRE_BOUNDARY,
)
from memgen.experience.v3_pooling import (
    V3_POOLING_AUDIT_SCHEMA,
    V3_POOLING_PRE_BOUNDARY,
    V3_POOLING_SAMPLE_SCHEMA,
)
from memgen.experience.v3_selector import (
    V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
    V3_MARGIN_SELECTOR_POLICY,
    calibration_artifact_sha256,
    load_margin_selector_calibration,
    numeric_summary,
    selection_concentration,
    selector_calibration_query_pooling,
)


CALIBRATION_SCRIPT = (
    PROJECT_ROOT / "scripts" / "calibrate_v3_pre_boundary_selector.py"
)
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts" / "compare_v3_query_pooling.py"


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


class V3PreBoundaryCalibrationTests(unittest.TestCase):
    def test_qualified_pooling_audit_builds_bound_margin_selector(self) -> None:
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

            sample_path = root / "pooling_audit_samples.jsonl"
            samples = []
            for index, (top1, top2, first, second) in enumerate((
                (0.75, 0.5, "memory-a", "memory-b"),
                (0.625, 0.5, "memory-b", "memory-a"),
            )):
                sample = {
                    "schema_version": V3_POOLING_SAMPLE_SCHEMA,
                    "sample_id": f"sample-{index}",
                    "candidates": {
                        V3_POOLING_PRE_BOUNDARY: {
                            "query_embedding_sha256": f"query-{index}",
                            "hits": [
                                {
                                    "memory_id": first,
                                    "score": top1,
                                    "rank": 1,
                                },
                                {
                                    "memory_id": second,
                                    "score": top2,
                                    "rank": 2,
                                },
                            ],
                            "top1_top2_margin": top1 - top2,
                        },
                    },
                }
                sample["sample_sha256"] = canonical_json_sha256(sample)
                samples.append(sample)
            sample_path.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )
            margins = [0.25, 0.125]
            concentration = selection_concentration(
                ("memory-a", "memory-b"),
                complete_memory_ids=("memory-a", "memory-b"),
            )
            audit = {
                "schema_version": V3_POOLING_AUDIT_SCHEMA,
                "created_at": "2026-01-01T00:00:00+00:00",
                "status": "passed",
                "task_accuracy_used": False,
                "answer_or_reward_used": False,
                "sample_count": 2,
                "source": {"source_row_count": 10},
                "pooling_contract": {"embedding_transform": "none"},
                "candidates": {
                    V3_POOLING_PRE_BOUNDARY: {
                        "specification": {
                            "key_pooling": "last_valid_token",
                            "query_pooling": (
                                V3_QUERY_POOLING_PRE_BOUNDARY
                            ),
                        },
                        "selection_concentration": concentration,
                        "top1_top2_margin": numeric_summary(margins),
                    },
                },
                "qualification": {
                    "recommended_candidate": V3_POOLING_PRE_BOUNDARY,
                    "candidates": {
                        V3_POOLING_PRE_BOUNDARY: {"qualified": True}
                    },
                },
                "artifacts": {
                    "sample_traces": {
                        "path": sample_path.name,
                        "sha256": file_sha256(sample_path),
                    },
                },
                "requirements": {"answer_blind": True},
                "inputs": {
                    "retrieval_key_manifest_sha256": file_sha256(key_path)
                },
            }
            audit["report_sha256"] = canonical_json_sha256({
                key: value
                for key, value in audit.items()
                if key != "created_at"
            })
            audit_path = root / "pooling_audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            output_path = root / "selector.json"
            subprocess.run(
                [
                    sys.executable,
                    str(CALIBRATION_SCRIPT),
                    "--pooling-audit",
                    str(audit_path),
                    "--pooling-samples",
                    str(sample_path),
                    "--retrieval-key-manifest",
                    str(key_path),
                    "--output",
                    str(output_path),
                    "--minimum-triggered-samples",
                    "2",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            selector = load_margin_selector_calibration(output_path)
            self.assertEqual(
                selector_calibration_query_pooling(selector),
                V3_QUERY_POOLING_PRE_BOUNDARY,
            )
            self.assertAlmostEqual(
                selector["calibration"]["minimum_top1_top2_margin"], 0.25
            )
            self.assertEqual(
                selector["first_attempt_selection_concentration"],
                concentration,
            )
            self.assertEqual(
                selector["source"]["pooling_audit_report_sha256"],
                file_sha256(audit_path),
            )

    def test_matched_comparison_binds_pooling_specific_calibrations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for suffix in ("a", "b"):
                records.append(MemoryRecord(
                    memory_id=f"memory-{suffix}",
                    source_experience_id=f"experience-{suffix}",
                    experience_type="recovery",
                    approved_route="target",
                    source_logical_split="memory-train",
                    phase1_provenance_sha256="phase1",
                    review_provenance_sha256="review",
                    source_record_sha256=f"source-{suffix}",
                    reasoner_name="reasoner",
                    reasoner_revision="model-rev",
                    tokenizer_revision="tokenizer-rev",
                    sanitized_fields={
                        "when_facing": f"case {suffix}",
                        "prefer": "verify",
                        "avoid": "guess",
                    },
                    payload_diagnostics={},
                    sanitized_retrieval_key=f"case {suffix}",
                    sanitized_contrast_payload=f"payload {suffix}",
                    payload_hash=f"payload-{suffix}",
                    token_ids_sha256=f"tokens-{suffix}",
                    token_count=2,
                    model_sequence_limit=1024,
                ))
            memory_path = root / "memory_records.jsonl"
            memory_path.write_text(
                "".join(json.dumps(record.to_dict()) + "\n" for record in records),
                encoding="utf-8",
            )
            key_hash = "retrieval-key-bank"

            def calibration(pooling, threshold):
                source = {
                    "logical_split": "calibration-val",
                    "retrieval_embedding_transform": "none",
                    "retrieval_key_manifest_sha256": key_hash,
                }
                if pooling != V3_QUERY_POOLING_BOUNDARY_LAST:
                    source["query_pooling"] = pooling
                value = {
                    "schema_version": V3_MARGIN_SELECTOR_CALIBRATION_SCHEMA,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "status": "passed",
                    "policy": V3_MARGIN_SELECTOR_POLICY,
                    "task_accuracy_used": False,
                    "answer_or_reward_used": False,
                    "source": source,
                    "calibration": {
                        "minimum_top1_top2_margin": threshold,
                        "sample_count": 2,
                    },
                    "first_attempt_selection_concentration": (
                        selection_concentration(
                            ("memory-a", "memory-b"),
                            complete_memory_ids=("memory-a", "memory-b"),
                        )
                    ),
                    "requirements": {"answer_blind": True},
                }
                value["artifact_sha256"] = calibration_artifact_sha256(value)
                return value

            calibrations = {
                "v31": calibration(V3_QUERY_POOLING_BOUNDARY_LAST, 0.1),
                "v33": calibration(V3_QUERY_POOLING_PRE_BOUNDARY, 0.05),
            }
            calibration_paths = {}
            for name, value in calibrations.items():
                path = root / f"{name}_calibration.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                calibration_paths[name] = path

            common_inputs = {
                "split_manifest_sha256": "split",
                "memory_records_sha256": file_sha256(memory_path),
                "retrieval_key_manifest_sha256": key_hash,
                "side_kv_manifest_sha256": "side-kv",
                "v3_offline_report_sha256": "offline",
                "e0_final_report_sha256": "e0",
                "risk_artifact_sha256": "risk",
            }

            def profile(name, pooling, threshold):
                value = {
                    "schema_version": "experience-memory-v3-evaluation-profile-v1",
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
                    "alignment": {"record_count": 2},
                    "system_profile": {
                        "layer_number": 24,
                        "query_pooling": pooling,
                        "retrieval_embedding_transform": "none",
                        "retrieval_abstention_policy": "top1_top2_margin",
                        "retrieval_min_top1_top2_margin": threshold,
                    },
                    "selector_calibration": {
                        "artifact_sha256": calibrations[name][
                            "artifact_sha256"
                        ]
                    },
                    "inputs": common_inputs | {
                        "selector_calibration_sha256": file_sha256(
                            calibration_paths[name]
                        )
                    },
                }
                value["profile_sha256"] = evaluation_profile_sha256(value)
                return value

            profiles = {
                "v31": profile(
                    "v31", V3_QUERY_POOLING_BOUNDARY_LAST, 0.1
                ),
                "v33": profile(
                    "v33", V3_QUERY_POOLING_PRE_BOUNDARY, 0.05
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
                    v3_strict = bool(index or name == "v33")
                    top1 = "memory-a" if index == 0 else "memory-b"
                    top2 = "memory-b" if index == 0 else "memory-a"
                    margin = 0.2 if name == "v31" else 0.1
                    diagnostics = {
                        "retrieval_attempt_count": 1,
                        "activation_count": 1,
                        "replacement_count": 0,
                        "duplicate_count": 0,
                        "abstain_count": 0,
                        "rearm_count": 0,
                        "memory_attention_step_count": 1,
                    }
                    row = {
                        "schema_version": "experience-memory-v3-evaluation-row-v1",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "profile_sha256": profile_value["profile_sha256"],
                        "sample_id": f"sample-{index}",
                        "conditions": {
                            "vanilla": {
                                "completion_token_ids_sha256": f"vanilla-{index}",
                                "strict_correct": bool(index),
                                "format_correct": True,
                            },
                            "v3": {
                                "completion_token_ids_sha256": f"{name}-{index}",
                                "strict_correct": v3_strict,
                                "format_correct": True,
                                "generated_token_count": 10 + index,
                                "online_diagnostics": diagnostics,
                                "runtime_trace": {
                                    "retrieval_attempts": [{
                                        "boundary_token_id": 11,
                                        "boundary_token_text": ",",
                                        "outcome": "activated",
                                        "selected_memory_id": top1,
                                        "retrieval_decision": {
                                            "hits": [
                                                {
                                                    "memory_id": top1,
                                                    "score": 0.8,
                                                },
                                                {
                                                    "memory_id": top2,
                                                    "score": 0.8 - margin,
                                                },
                                            ],
                                            "query": {
                                                "top1_top2_margin": margin
                                            },
                                        },
                                    }],
                                },
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
                    "--v33-results",
                    str(result_paths["v33"]),
                    "--v33-profile",
                    str(profile_paths["v33"]),
                    "--v33-calibration",
                    str(calibration_paths["v33"]),
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
                report["paired_v33_minus_v31"]["strict"][
                    "treatment_correct_control_wrong"
                ],
                1,
            )
            self.assertEqual(
                report["retrieval"]["v33_pre_boundary"]["first_attempts"][
                    "boundary_strata"
                ][0]["boundary_token_text"],
                ",",
            )
            self.assertTrue(output.with_suffix(".md").is_file())


if __name__ == "__main__":
    unittest.main()
