from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from memgen.experience.phase1 import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/compare_gsm8k_attention_backends.py"
CONDITIONS = (
    "native_transformers_generate",
    "explicit_live_kv_cache",
)


class BaseAttentionComparisonTests(unittest.TestCase):
    def write_backend(
        self,
        *,
        root: Path,
        backend: str,
        completions: list[list[int]],
        rewards: list[float],
    ) -> Path:
        directory = root / backend
        directory.mkdir()
        records = []
        for index, (completion, reward) in enumerate(zip(completions, rewards)):
            condition = {
                "completion_token_ids": completion,
                "final_reward": reward,
            }
            records.append({
                "schema_version": "gsm8k-base-generation-parity-result-v3",
                "sample_id": f"sample-{index}",
                "logical_split": "final-test",
                "dataset_split": "test",
                "source_index": index,
                "question_sha256": f"question-{index}",
                "prompt_token_count": 10 + index,
                "prompt_token_ids_sha256": f"prompt-{index}",
                "attention_implementation": backend,
                "conditions": {
                    name: dict(condition) for name in CONDITIONS
                },
            })
        results_path = directory / "results.jsonl"
        results_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        accuracy = sum(rewards) / len(rewards)
        summary = {
            "schema_version": "gsm8k-base-generation-parity-report-v3",
            "logical_split": "final-test",
            "dataset_split": "test",
            "sample_count": len(records),
            "exact_token_parity": True,
            "prompt_contract": {"contract_sha256": "prompt-contract"},
            "generation_contract": {
                "attention_implementation": backend,
                **{
                    name: {
                        "batch_size": 1,
                        "decoding": "greedy",
                        "max_new_tokens": 1024,
                    }
                    for name in CONDITIONS
                },
            },
            "reasoner": {
                "model_name": "reasoner",
                "model_revision": "revision",
                "tokenizer_revision": "tokenizer",
                "dtype": "bfloat16",
                "attention_implementation": backend,
            },
            "conditions": {
                name: {
                    "sample_count": len(records),
                    "accuracy": accuracy,
                    "diagnostic_answer_accuracy": accuracy,
                    "format_accuracy": accuracy,
                    "mean_generation_length": sum(map(len, completions))
                    / len(completions),
                }
                for name in CONDITIONS
            },
            "inputs": {
                "split_manifest_sha256": "split",
                "side_kv_manifest_sha256": "side-kv",
            },
            "results": {"sha256": file_sha256(results_path)},
        }
        (directory / "base_parity_summary.json").write_text(
            json.dumps(summary) + "\n", encoding="utf-8"
        )
        return directory

    def test_compares_only_the_attention_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eager = self.write_backend(
                root=root,
                backend="eager",
                completions=[[1, 2], [3, 4]],
                rewards=[0.0, 0.0],
            )
            flash = self.write_backend(
                root=root,
                backend="flash_attention_2",
                completions=[[1, 2], [3, 5]],
                rewards=[1.0, 0.0],
            )
            output = root / "comparison.json"
            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--reference-name",
                    "eager",
                    "--reference-dir",
                    str(eager),
                    "--candidate-name",
                    "flash_attention_2",
                    "--candidate-dir",
                    str(flash),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            native = report["cross_backend_token_comparison"][
                "native_transformers_generate"
            ]
            self.assertEqual(native["token_mismatch_count"], 1)
            self.assertEqual(native["candidate_correct_reference_wrong"], 1)
            self.assertEqual(
                report["candidate_minus_reference"][
                    "native_transformers_generate"
                ][
                    "candidate_minus_reference_accuracy"
                ],
                0.5,
            )
            self.assertEqual(report["fixed_contract"]["batch_size"], 1)
            self.assertEqual(report["reference_backend"], "eager")
            self.assertEqual(report["candidate_backend"], "flash_attention_2")


if __name__ == "__main__":
    unittest.main()
