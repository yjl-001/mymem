"""Run the real shell orchestration with a local recording Python stand-in."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "test.sh"


class V43ShellTests(unittest.TestCase):
    def run_fixture(self, root, mode="all", fail_smoke=False, missing_cache=False, validate=False, fail_construction=False, bank_scope="all"):
        paths = {"MEMGEN_V4_SEMANTIC_PACKETS": root / "packets.jsonl",
                 "MEMGEN_V4_CURATED_BANK_DIR": root / "curated",
                 "MEMGEN_V4_SIDE_KV_DIR": root / "old-side",
                 "MEMGEN_V43_CACHE_MANIFEST": root / "source" / "cache.json",
                 "MEMGEN_V43_RISK_ARTIFACT": root / "risk.pt",
                 "MEMGEN_V43_AUDIT_ROOT": root / "audit",
                 "MEMGEN_V43_BANK_DIR": root / "bank",
                 "MEMGEN_V43_SIDE_KV_DIR": root / "side"}
        inputs = [paths["MEMGEN_V4_SEMANTIC_PACKETS"], paths["MEMGEN_V4_CURATED_BANK_DIR"] / "bank_records.jsonl",
                  paths["MEMGEN_V4_CURATED_BANK_DIR"] / "bank_manifest.json",
                  paths["MEMGEN_V4_SIDE_KV_DIR"] / "v4_side_kv_manifest.json", paths["MEMGEN_V43_RISK_ARTIFACT"]]
        if not missing_cache:
            inputs.append(paths["MEMGEN_V43_CACHE_MANIFEST"])
        if bank_scope == "primary":
            inputs.extend([paths["MEMGEN_V43_BANK_DIR"] / "construction_bundle_manifest.json",
                           paths["MEMGEN_V43_SIDE_KV_DIR"] / "v4_3_primary_side_kv_manifest.json"])
        for p in inputs:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture only")
        stub = root / "record-python"
        stub.write_text("#!" + sys.executable + "\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == '-c':
    compile(args[1], '<primary-preflight>', 'exec')
assert not any(k in os.environ for k in ('GLM_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'))
assert ('DEEPSEEK_API_KEY' in os.environ) == args[0].endswith('build_v4_3_deepseek_bank.py')
with open(os.environ['CALL_LOG'], 'a') as handle:
    handle.write(json.dumps(args) + '\\n')
if args[0].endswith('diagnose_v4_3_construction.py') and os.environ.get('FAIL_CONSTRUCTION') == '1':
    sys.exit(19)
if args[0].endswith('audit_v4_3_unified_memory.py'):
    mode = args[args.index('--mode') + 1]
    if mode == 'smoke' and os.environ.get('FAIL_SMOKE') == '1':
        sys.exit(17)
    out = pathlib.Path(args[args.index('--output-dir') + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'v4_3_audit_report.json').write_text('{}')
    (out / 'v4_3_core_summary.json').write_text('{}')
''')
        stub.chmod(0o755)
        env = {**os.environ, **{k: str(v) for k, v in paths.items()},
               "MEMGEN_PYTHON_BIN": str(stub), "CALL_LOG": str(root / "calls.jsonl"),
               "FAIL_SMOKE": str(int(fail_smoke)), "MEMGEN_V43_VALIDATE_ONLY": str(int(validate))}
        env["FAIL_CONSTRUCTION"] = str(int(fail_construction))
        env["MEMGEN_V43_BANK_SCOPE"] = bank_scope
        for key in ("DEEPSEEK_API_KEY", "GLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            env[key] = "sentinel-not-a-key"
        result = subprocess.run(["bash", str(RUNNER), mode], cwd=ROOT, env=env, text=True, capture_output=True)
        log = root / "calls.jsonl"
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    def test_complete_order_and_key_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_fixture(Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            scripts = [call for call in calls if call[0] != "-m" and not call[0].endswith("diagnose_v4_3_construction.py")]
            self.assertTrue(calls[1][0].endswith("diagnose_v4_3_construction.py"))
            self.assertIn("--require-auditable", calls[1])
            self.assertEqual([Path(call[0]).name for call in scripts], ["build_v4_3_deepseek_bank.py", "compile_v4_3_side_kv.py",
                               "audit_v4_3_unified_memory.py", "audit_v4_3_unified_memory.py"])
            self.assertEqual(scripts[2][scripts[2].index("--mode") + 1], "smoke")
            self.assertEqual(scripts[3][scripts[3].index("--mode") + 1], "full")
            self.assertIn("--smoke-report", scripts[3])
            self.assertTrue(all("--resume" in call for call in scripts))

    def test_smoke_failure_prevents_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_fixture(Path(tmp), fail_smoke=True)
            self.assertEqual(result.returncode, 17)
            self.assertFalse(any("full" in call for call in calls))

    def test_primary_reuses_artifacts_without_teacher_or_compiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_fixture(Path(tmp), bank_scope="primary")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(any("build_v4_3_deepseek_bank.py" in c[0] or "compile_v4_3_side_kv.py" in c[0] for c in calls))
            audits = [c for c in calls if c[0].endswith("audit_v4_3_unified_memory.py")]
            self.assertEqual(len(audits), 2)
            self.assertTrue(all(c[c.index("--bank-scope") + 1] == "primary" for c in audits))

    def test_construct_mode_does_not_require_gpu_or_source_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_fixture(Path(tmp), mode="construct", missing_cache=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0][0].endswith("build_v4_3_deepseek_bank.py"))

    def test_missing_cache_does_not_regenerate_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_fixture(Path(tmp), missing_cache=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required existing input", result.stderr)
            self.assertEqual(calls, [])

    def test_validate_only_propagates_to_every_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_fixture(Path(tmp), validate=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(all("--validate-only" in call for call in calls if call[0] != "-m" and not call[0].endswith("diagnose_v4_3_construction.py")))

    def test_construction_quarantine_prevents_compilation_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_fixture(Path(tmp), fail_construction=True)
            self.assertEqual(result.returncode, 19)
            self.assertEqual(len(calls), 2)
            self.assertIn("stage=construction", result.stderr)


if __name__ == "__main__":
    unittest.main()
