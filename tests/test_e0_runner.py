from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/experiments/gsm8k/run_e0_experience_memory.sh"


class E0RunnerConfigurationTests(unittest.TestCase):
    def build_fixture(self, root: Path, *, token_budget: str = "128") -> tuple[Path, Path]:
        phase1_dir = root / "phase1"
        phase1_dir.mkdir()
        approved = {
            "student": {
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "model_revision": "model-commit",
                "tokenizer_revision": "tokenizer-commit",
            }
        }
        (phase1_dir / "ai_approved_bank_records.jsonl").write_text(
            json.dumps(approved) + "\n",
            encoding="utf-8",
        )
        (phase1_dir / "verified_experiences.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (phase1_dir / "split_manifest.json").write_text(
            json.dumps({"dataset": {"revision": "dataset-commit"}}) + "\n",
            encoding="utf-8",
        )
        env_path = root / "e0.env"
        env_path.write_text(
            "\n".join(
                (
                    f'export MEMGEN_OUTPUT_ROOT="{root / "output"}"',
                    f'export MEMGEN_E0_MAX_PAYLOAD_TOKENS="{token_budget}"',
                    'export MEMGEN_E0_CUDA_VISIBLE_DEVICES="3"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return phase1_dir, env_path

    def run_config(self, phase1_dir: Path, env_path: Path) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        environment["MEMGEN_E0_ENV_FILE"] = str(env_path)
        return subprocess.run(
            ["bash", str(RUNNER), "--print-config", str(phase1_dir)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_derives_frozen_revisions_from_phase1_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phase1_dir, env_path = self.build_fixture(Path(directory))
            result = self.run_config(phase1_dir, env_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            parsed = dict(
                line.split("=", 1)
                for line in result.stdout.splitlines()
                if "=" in line
            )
            self.assertEqual(parsed["model_revision"], "model-commit")
            self.assertEqual(parsed["tokenizer_revision"], "tokenizer-commit")
            self.assertEqual(parsed["dataset_revision"], "dataset-commit")
            self.assertEqual(parsed["max_payload_tokens"], "128")
            self.assertEqual(parsed["cuda_visible_devices"], "3")
            self.assertEqual(parsed["dtype"], "bfloat16")

    def test_rejects_a_non_numeric_payload_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            phase1_dir, env_path = self.build_fixture(
                Path(directory),
                token_budget="auto",
            )
            result = self.run_config(phase1_dir, env_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be a positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
