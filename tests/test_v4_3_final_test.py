from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from memgen.experience.phase1 import create_gsm8k_split_manifest, canonical_json_sha256
from memgen.experience.v4_3_artifacts import read_json
from memgen.experience.v4_3_bank import seal, text_hash
from memgen.experience.v4_3_question_selector import fit_selector
from scripts import run_v4_3_final_test as driver
from tests.test_v4_3_side_kv import torch
from tests import test_v4_3_gated_prefix as gated_tests
from tests.test_v4_3_question_selector import utility_rows


class FinalPlanTests(unittest.TestCase):
    def test_full_official_test_coverage_and_overlap(self):
        train = [{"question": f"Train {i}", "answer": f"Answer {i}"} for i in range(5)]
        test = [{"question": f"Test {i}", "answer": f"Test answer {i}"} for i in range(3)]
        manifest = create_gsm8k_split_manifest(train, test, bank_source_size=2, calibration_val_size=2, seed=43, dataset_revision="pinned")
        entries = driver.final_entries(manifest)
        self.assertEqual([e["source_index"] for e in entries], [0, 1, 2])
        self.assertTrue(all(e["dataset_split"] == "test" for e in entries))
        for mutation in ("missing", "overlap", "duplicate_index"):
            bad = deepcopy(manifest)
            if mutation == "missing":
                bad["samples"].pop()
            elif mutation == "overlap":
                bad["samples"][-1]["question_sha256"] = bad["samples"][0]["question_sha256"]
            else:
                bad["samples"][-1]["source_index"] = 0
            bad["manifest_sha256"] = canonical_json_sha256({k: v for k, v in bad.items() if k not in {"created_at", "manifest_sha256"}})
            with self.assertRaisesRegex(ValueError, "entire official test"):
                driver.final_entries(bad)

    def test_token_statistics_include_all_lengths_and_limit(self):
        self.assertEqual(driver.token_stats([1, 2, 3, 1024]), {
            "total": 1030, "mean": 257.5, "median": 2.5, "min": 1, "max": 1024,
            "p90": 1024, "at_1024_token_limit_count": 1})


@unittest.skipIf(torch is None, "Torch required")
class FinalRuntimeTests(unittest.TestCase):
    def runtime(self, trigger=True, box_after=5):
        return gated_tests.GatedPrefixTests.runtime(self, trigger=trigger, box_after=box_after)

    def test_full_pipeline_freezes_before_outcomes_scores_after_branches_and_resumes(self):
        import memgen.model.v4_3_question_selector as native
        from memgen.model.v4_3_prefix_equivalence import prefix_bank
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = self.runtime()
            records = [{"bank_id": b, "descriptor": "Check total quantities.", "record_sha256": b} for b in ("bank-a", "bank-b")]
            # Real prefix compiler requires the production Bank ID namespace.
            for r in records:
                r["bank_id"] = "v43-"+r["bank_id"]
                prefix_bank(root / "eq" / "prefix_kv", r, rt, "eq")
            ids = [r["bank_id"] for r in records]
            training, tuning = utility_rows("train"), utility_rows("tune")
            for rows in (training, tuning):
                for row in rows:
                    row["rewards"] = {("v43-"+b if b.startswith("bank-") else b): value for b, value in row["rewards"].items()}
            selector = fit_selector(training, tuning, ids, {ids[0]: [1., 0.], ids[1]: [-1., 0.]}, "source")
            selector = seal({**selector, "semantic_threshold": .5}, "selector_sha256")
            dataset = [{"question": f"How many remain in group {i}?", "answer": "Solution.\n#### 37"} for i in range(2)]
            entries = [{"sample_id": f"test-{i}", "source_index": i, "dataset_split": "test", "logical_split": "final-test",
                        "question_sha256": text_hash(d["question"]), "answer_sha256": text_hash(d["answer"])} for i, d in enumerate(dataset)]
            profile = seal({"samples": entries, "reasoner": {}, "dataset": {"revision": "fixture", "test_size": 2},
                "bank_ids": ids, "semantic_threshold": .5, "source_equivalence_profile_sha256": "eq",
                "selector_sha256": selector["selector_sha256"], "evaluation_role": "full_official_final_test_frozen_policy",
                "gate_config": rt.gate.config.to_dict(), "generated_token_policy": "completion_ids_only"}, "profile_sha256")
            args = SimpleNamespace(output_dir=root / "out", selector_dir=root / "selector", equivalence_dir=root / "eq",
                bank_dir=root / "bank", side_kv_dir=root / "side", cache_manifest=root / "source" / "cache.json",
                device="cpu", resume=True, validate_only=False, plan_only=False)
            dataset_module = ModuleType("datasets")
            def load_dataset(*a, **kw):
                self.assertEqual(kw, {"revision": "fixture", "split": "test"})
                return dataset
            dataset_module.load_dataset = load_dataset
            real_generate, real_score = native.generate, driver.score_branch
            calls, scoring = [], []
            def generate(rt, question, *a, **kw):
                sid = "test-0" if question == dataset[0]["question"] else "test-1"
                self.assertTrue((args.output_dir / "samples" / sid / "decision.json").exists())
                calls.append(sid)
                if len(calls) == 2:
                    raise RuntimeError("interrupt before native prefix finishes")
                return real_generate(rt, question, *a, **kw)
            def score(*a, **kw):
                sid = "test-0" if len(scoring) < 3 else "test-1"
                for b in driver.BRANCHES:
                    row = read_json(args.output_dir / "samples" / sid / (b+".json"))
                    self.assertNotIn("strict_reward", row["result"])
                scoring.append(sid)
                return real_score(*a, **kw)
            with patch.object(driver, "parse_args", return_value=args), \
                 patch.object(driver, "prepare", return_value=(profile, selector, records, rt.gate)), \
                 patch.dict("sys.modules", {"datasets": dataset_module}), patch.object(native, "load_runtime", return_value=rt), \
                 patch.object(native, "encode_text", side_effect=lambda rt, q: [1., 0.] if q == dataset[0]["question"] else [0., 1.]), \
                 patch.object(native, "generate", side_effect=generate), patch.object(driver, "score_branch", side_effect=score):
                with self.assertRaisesRegex(RuntimeError, "interrupt before"):
                    driver.main()
                self.assertEqual(scoring, [])
                saved = args.output_dir / "samples" / "test-0" / "baseline.json"
                timestamp = saved.stat().st_mtime_ns
                driver.main()
                self.assertEqual(saved.stat().st_mtime_ns, timestamp)
                self.assertEqual(len(scoring), 6)
                summary = read_json(args.output_dir / "brief_summary.json")
                self.assertTrue(summary["complete"])
                self.assertTrue(summary["official_test_used"])
                self.assertEqual(summary["sample_count"], 2)
                self.assertEqual(summary["gate"]["selected_count"], 1)
                cases = [read_json(args.output_dir / "samples" / e["sample_id"] / "scored.json") for e in entries]
                for b in driver.BRANCHES:
                    lengths = [len(c["branches"][b]["continuation_token_ids"]) for c in cases]
                    self.assertEqual(summary["overall"][b]["generated_tokens"]["total"], sum(lengths))
                    self.assertEqual(summary["overall"][b]["generated_tokens"]["mean"], sum(lengths)/2)
                self.assertGreater(summary["offline_memory_tokens_excluded_from_generation"]["native_prefix_input_tokens_total"], 0)
                files = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                args.validate_only = True
                with patch.object(native, "load_runtime", side_effect=AssertionError("No model on complete validate")), \
                     patch.object(dataset_module, "load_dataset", side_effect=AssertionError("No dataset on complete validate")):
                    driver.main()
                self.assertEqual(files, {p: p.stat().st_mtime_ns for p in files})
                (args.output_dir / "samples" / "test-0" / "decision.json").unlink()
                with self.assertRaisesRegex(ValueError, "lack their prior"):
                    driver.main()


class FinalShellTests(unittest.TestCase):
    def test_dispatch(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "record-python"
            script.write_text('#!/bin/sh\nfor arg in "$@"; do echo "$arg"; done\n')
            script.chmod(0o755)
            run = subprocess.run(["bash", "test.sh", "final-test", "--plan-only"], cwd=root,
                env={**os.environ, "MEMGEN_PYTHON_BIN": str(script), "MEMGEN_V43_FINAL_TEST_ROOT": "/fixture/final"}, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            args = run.stdout.splitlines()
            self.assertEqual(args[0], "scripts/run_v4_3_final_test.py")
            self.assertEqual(args[args.index("--output-dir")+1], "/fixture/final")
            self.assertIn("--resume", args)


if __name__ == "__main__":
    unittest.main()
