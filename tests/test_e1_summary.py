from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from memgen.experience.e1 import (
    E1_CONDITIONS,
    E1_MANIFEST_SCHEMA,
    E1_RESULTS_SCHEMA,
)
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.system import ExperienceMemorySystemProfile
from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE

from test_e1_experience import assignment, memory_choice


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = PROJECT_ROOT / "scripts/summarize_e1_experience_memory.py"


class E1SummaryTests(unittest.TestCase):
    def test_summarizes_authenticated_gate_vs_matched_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = [
                assignment(f"sample-{index}", memory_choice(f"m{index}", 100 + index))
                for index in range(2)
            ]
            profile = ExperienceMemorySystemProfile()
            prompt_contract = GSM8K_PROMPT_CONTRACT.metadata(
                chat_template=CONVERSATION_TEMPLATE
            )
            vanilla_generation = {
                "implementation": "transformers_generate",
                "decoding": "greedy",
                "use_cache": True,
                "batch_size": 1,
            }
            gate_generation = {
                "implementation": "explicit_live_kv_cache",
                "decoding": "greedy",
                "use_cache": True,
                "batch_size": 1,
            }
            manifest = {
                "schema_version": E1_MANIFEST_SCHEMA,
                "created_at": "fixture",
                "status": "frozen",
                "answer_or_reward_used": False,
                "logical_split": "dev-test",
                "dataset_split": "train",
                "evaluation_role": "development_diagnostic",
                "prompt_contract": prompt_contract,
                "reasoner": {"layer": 24},
                "configuration": {
                    "max_new_tokens": 1024,
                    "vanilla_generation": vanilla_generation,
                    "gate_generation": gate_generation,
                    "system_profile": profile.to_dict(),
                },
                "summary": {"sample_count": len(assignments)},
                "assignments": [item.to_dict() for item in assignments],
            }
            manifest["manifest_sha256"] = canonical_json_sha256({
                key: value
                for key, value in manifest.items()
                if key not in {"created_at", "manifest_sha256"}
            })
            manifest_path = root / "assignment.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            results_path = root / "results.jsonl"
            with results_path.open("w", encoding="utf-8") as handle:
                for index, item in enumerate(assignments):
                    assert item.matched_memory is not None
                    conditions = {}
                    for condition in E1_CONDITIONS:
                        applied = condition == "matched_persistent_memory"
                        completion = [31, 50 + index] if applied else [31, 40 + index]
                        conditions[condition] = {
                            "final_reward": float(applied and index == 0),
                            "format_valid": True,
                            "generation_length": len(completion),
                            "completion_token_ids": completion,
                            "completion_token_ids_sha256": canonical_json_sha256(
                                completion
                            ),
                            "side_kv_applied": applied,
                            "verifier": {"diagnostic_answer_correct": False},
                            "memory_id": (
                                item.matched_memory.memory_id if applied else None
                            ),
                            "payload_hash": (
                                item.matched_memory.payload_hash if applied else None
                            ),
                            "memory_attention": (
                                {
                                    "trace_count": 1,
                                    "memory_ids": [item.matched_memory.memory_id],
                                    "memory_slot_counts": [
                                        item.matched_memory.kv_valid_slot_count
                                    ],
                                    "memory_score_normalizations": [
                                        profile.memory_score_normalization
                                    ],
                                    "memory_score_biases": [
                                        profile.memory_score_bias
                                    ],
                                    "native_key_lengths": [
                                        len(item.prefix_token_ids)
                                    ],
                                    "native_key_lengths_sha256": canonical_json_sha256(
                                        [len(item.prefix_token_ids)]
                                    ),
                                    "memory_attention_masses": [0.2],
                                    "memory_attention_masses_sha256": canonical_json_sha256(
                                        [0.2]
                                    ),
                                    "mean_memory_attention_mass": 0.2,
                                    "one_trace_per_post_trigger_token": True,
                                    "native_cache_length_matches_real_tokens": True,
                                    "all_memory_attention_mass_finite_and_positive": True,
                                    "memory_id_constant_and_matched": True,
                                    "memory_slot_count_constant_and_matched": True,
                                    "normalization_constant_and_matched": True,
                                    "memory_score_bias_constant_and_matched": True,
                                    "baseline_first_token_matches_gate_observation": True,
                                }
                                if applied
                                else None
                            ),
                            "first_step_logits_kl_baseline_to_memory": (
                                0.01 if applied else None
                            ),
                        }
                    record = {
                        "schema_version": E1_RESULTS_SCHEMA,
                        "sample_id": item.sample_id,
                        "logical_split": item.logical_split,
                        "dataset_split": item.dataset_split,
                        "evaluation_role": "development_diagnostic",
                        "question_sha256": item.question_sha256,
                        "assignment_manifest_sha256": manifest["manifest_sha256"],
                        "assigned": True,
                        "triggered": True,
                        "prefix_token_ids_sha256": item.prefix_token_ids_sha256,
                        "retrieval_query": item.retrieval_query,
                        "matched_memory": item.matched_memory.to_dict(),
                        "system_profile": profile.to_dict(),
                        "vanilla_matches_gate_observation_only": True,
                        "conditions": conditions,
                    }
                    handle.write(json.dumps(record) + "\n")

            run_report = {
                "schema_version": "experience-memory-e1-run-report-v5",
                "status": "completed",
                "logical_split": "dev-test",
                "dataset_split": "train",
                "evaluation_role": "development_diagnostic",
                "prompt_contract": prompt_contract,
                "generation_contract": {
                    "max_new_tokens": 1024,
                    "vanilla": vanilla_generation,
                    "gate_observation_only": gate_generation,
                },
                "system_profile": profile.to_dict(),
                "results": {"sha256": file_sha256(results_path)},
                "inputs": {
                    "assignment_manifest_sha256": file_sha256(manifest_path)
                },
            }
            run_report_path = root / "run_report.json"
            run_report_path.write_text(
                json.dumps(run_report) + "\n", encoding="utf-8"
            )
            output = root / "summary.json"
            result = subprocess.run(
                [
                    "python",
                    str(SUMMARY_SCRIPT),
                    "--assignment-manifest",
                    str(manifest_path),
                    "--results",
                    str(results_path),
                    "--run-report",
                    str(run_report_path),
                    "--output",
                    str(output),
                    "--bootstrap-resamples",
                    "100",
                    "--min-primary-pairs",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["assigned_count"], 2)
            self.assertEqual(summary["integrity_violations"], [])
            self.assertTrue(
                summary["acceptance"]["assignment_and_runtime_integrity"]
            )
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["formal_e1_passed"])
            self.assertFalse(summary["formal_task_claim"])


if __name__ == "__main__":
    unittest.main()
