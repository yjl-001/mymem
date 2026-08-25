from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/experiments/gsm8k/run_e1d_full_system.sh"


class E1RunnerConfigurationTests(unittest.TestCase):
    def run_print_config(
        self, *, logical_split: str, limit: str
    ) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1 = root / "phase1"
            e0 = root / "e0"
            phase1.mkdir()
            e0.mkdir()
            (phase1 / "split_manifest.json").write_text(
                json.dumps({"dataset": {"revision": "dataset-revision"}}) + "\n",
                encoding="utf-8",
            )
            side_manifest = {
                "reasoner": {
                    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "model_revision": "model-revision",
                    "tokenizer_revision": "tokenizer-revision",
                }
            }
            required = {
                "memory_records.v2.jsonl": "{}\n",
                "bm25_index.v1.json": "{}\n",
                "side_kv_manifest.json": json.dumps(side_manifest) + "\n",
                "e0_final_report.json": "{}\n",
            }
            for name, content in required.items():
                (e0 / name).write_text(content, encoding="utf-8")
            risk = root / "risk.pt"
            risk.write_bytes(b"fixture")
            env_path = root / "e1.env"
            env_path.write_text(
                "\n".join(
                    (
                        f'export MEMGEN_OUTPUT_ROOT="{root / "output"}"',
                        'export MEMGEN_E1_CUDA_VISIBLE_DEVICES="2"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["MEMGEN_E1_ENV_FILE"] = str(env_path)
            result = subprocess.run(
                [
                    "bash",
                    str(RUNNER),
                    "--print-config",
                    "--logical-split",
                    logical_split,
                    "--limit",
                    limit,
                    str(phase1),
                    str(e0),
                    str(risk),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return dict(
                line.split("=", 1)
                for line in result.stdout.splitlines()
                if "=" in line
            )

    def test_print_config_derives_frozen_artifact_metadata(self) -> None:
        parsed = self.run_print_config(
            logical_split="calibration-val", limit="8"
        )
        self.assertEqual(parsed["model_revision"], "model-revision")
        self.assertEqual(parsed["tokenizer_revision"], "tokenizer-revision")
        self.assertEqual(parsed["dataset_revision"], "dataset-revision")
        self.assertEqual(parsed["logical_split"], "calibration-val")
        self.assertEqual(parsed["limit"], "8")
        self.assertEqual(parsed["max_new_tokens"], "1024")
        self.assertEqual(parsed["cuda_visible_devices"], "2")
        self.assertEqual(
            parsed["system_profile"],
            "layer24-bm25-top1-gate-persistent-logslots-log10-v1",
        )

    def test_print_config_allows_full_final_test(self) -> None:
        parsed = self.run_print_config(logical_split="final-test", limit="0")
        self.assertEqual(parsed["logical_split"], "final-test")
        self.assertEqual(parsed["dataset_split"], "test")
        self.assertEqual(parsed["evaluation_role"], "final_evaluation")
        self.assertEqual(parsed["limit"], "0")


if __name__ == "__main__":
    unittest.main()
