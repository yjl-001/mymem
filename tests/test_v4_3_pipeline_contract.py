from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from memgen.experience import v4_3_bank as bank
from scripts import build_v4_3_unified_bank as builder
from tests.test_v4_3_bank import fixture


ROOT = Path(__file__).resolve().parents[1]


def write_fixture(root: Path, data=None):
    data = fixture() if data is None else data
    source = root / "curated"
    source.mkdir()
    packets = root / "semantic_evidence_packets.jsonl"
    policy = root / "policy.json"
    packets.write_text("".join(json.dumps(p) + "\n" for p in data["packets"]))
    policy.write_text(json.dumps(data["policy"]))
    data["manifest"]["inputs"]["semantic_preflight"]["evidence_packet_file_sha256"] = bank.file_hash(packets)
    data["manifest"]["curation"]["policy_file_sha256"] = bank.file_hash(policy)
    (source / "bank_manifest.json").write_text(json.dumps(bank.seal(data["manifest"], "manifest_sha256")))
    (source / "bank_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data["records"]))
    return source, packets, policy


class V43PipelineTests(unittest.TestCase):
    def test_cli_build_resume_validate_and_byte_identical_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, packets, policy = write_fixture(root)
            output = root / "unified"
            cmd = [sys.executable, str(ROOT / "scripts/build_v4_3_unified_bank.py"),
                   "--source-dir", str(source), "--semantic-packets", str(packets),
                   "--curation-policy", str(policy), "--output-dir", str(output)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn('"consumed_evidence_count": 116', first.stdout)
            original = {p.name: p.read_bytes() for p in output.iterdir()}
            for flag in ("--resume", "--validate-only"):
                run = subprocess.run(cmd + [flag], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(original, {p.name: p.read_bytes() for p in output.iterdir()})
            second = root / "second"
            run = subprocess.run(cmd[:-1] + [str(second)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(original, {p.name: p.read_bytes() for p in second.iterdir()})
            bundle = builder.read_json(output / "construction_bundle_manifest.json")
            bank.authenticate(bundle, "manifest_sha256", "bundle")
            for name, entry in bundle["artifacts"].items():
                self.assertEqual(bank.file_hash(output / name), entry["file_sha256"])

    def test_resume_validates_every_file_before_any_write(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.mkdir()
            (output / "a.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "drift"):
                builder.write_or_validate(output, {"b.json": b"new", "a.json": b"original"}, resume=True)
            self.assertFalse((output / "b.json").exists())
            self.assertEqual((output / "a.json").read_bytes(), b"changed")

    def test_interrupted_unsealed_output_can_resume_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.mkdir()
            a = output / "a.json"
            a.write_bytes(b"first")
            original_inode = a.stat().st_ino
            expected = {"a.json": b"first", "b.json": b"second", "construction_bundle_manifest.json": b"seal"}
            builder.write_or_validate(output, expected, resume=True)
            self.assertEqual(a.stat().st_ino, original_inode)
            self.assertEqual((output / "b.json").read_bytes(), b"second")

    def test_missing_sealed_artifact_is_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.mkdir()
            (output / "construction_bundle_manifest.json").write_bytes(b"seal")
            with self.assertRaisesRegex(ValueError, "lost artifacts"):
                builder.write_or_validate(output, {"a.json": b"data", "construction_bundle_manifest.json": b"seal"}, resume=True)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"bank_id":"a", "bank_id":"b"}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                builder.read_json(path)

    def test_no_external_network_key_read_or_model_import_in_cpu_build(self):
        with tempfile.TemporaryDirectory() as temp:
            source, packets, policy = write_fixture(Path(temp))
            # Execute the entire construction under denied imports, network and
            # environment reads, not merely inspect source strings.
            code = r'''
import importlib.abc, os, socket, sys
class DenyHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "sentence_transformers", "requests", "httpx", "openai", "anthropic"}:
            raise RuntimeError("Forbidden model/network import: " + fullname)
sys.meta_path.insert(0, DenyHeavy())
from scripts.build_v4_3_unified_bank import construct
from pathlib import Path
def forbidden(*args, **kwargs):
    raise RuntimeError("Forbidden network or environment access")
socket.socket = forbidden
os.getenv = forbidden
type(os.environ).__getitem__ = forbidden
outputs = construct(source_dir=Path(sys.argv[1]), packets_path=Path(sys.argv[2]), policy_path=Path(sys.argv[3]))
assert outputs["construction_report.json"]["consumed_evidence_count"] == 116
assert not any(name in sys.modules for name in ("torch", "transformers", "numpy", "requests", "httpx"))
print("CPU construction without model, network, or environment reads: PASS")
'''
            run = subprocess.run([sys.executable, "-c", code, str(source), str(packets), str(policy)],
                                 cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("PASS", run.stdout)

    def test_legacy_entry_point_and_frozen_files_preserved(self):
        root = (ROOT / "test.sh").read_text()
        self.assertIn("run_v4_3_unified_bank_experiment.sh", root)
        self.assertIn('"${1:-}" == "legacy"', root)
        self.assertIn("run_v4_question_recovery.sh", (ROOT / "scripts/experiments/gsm8k/run_v4_2_recovered_legacy.sh").read_text())
        expected = {
            "memgen/model/e1_runtime.py": "aadad19974338e8f5f67be058a9adc0b8fac3468a3ab3b19a766098c2874b3e4",
            "memgen/model/side_kv.py": "c6b028ea115e7735bf8926cc9eb3b006d68073e802ed03173720ca08ebab3207",
            "memgen/model/v3_5_retrieval.py": "5f718385e86a1d2a3ea3cc924f9b1a62fa3b8223b8d4b517d46a7293a35feccf",
            "memgen/model/v3_runtime.py": "0db40e1fffa7fb3e000810498d0433c92d6297c0eff9b676cff8a0e1963c0fd9",
        }
        for relative, sha in expected.items():
            self.assertEqual(bank.file_hash(ROOT / relative), sha)


if __name__ == "__main__":
    unittest.main()
