from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from memgen.experience.phase1 import create_gsm8k_split_manifest, canonical_json_sha256
from memgen.experience.v4_3_bank import canonical_hash, seal
from memgen.experience.v4_3_artifacts import atomic_json, read_json
from memgen.experience.v4_3_question_selector import (
    NO_MEMORY, POLICY, partition_samples, checked_dataset_rows, fit_selector, predict, evaluate,
)
from tests.test_v4_3_side_kv import torch, tiny_model
from tests.test_v4_3_prefix_equivalence import PrefixTokenizer


def utility_rows(split, n=12, harmful=False):
    rows = []
    for i in range(n):
        sign = 1 if i % 2 == 0 else -1
        rows.append({"sample_id": f"{split}-{i}", "selector_split": split, "feature": [sign, 0.],
                     "rewards": {NO_MEMORY: int(harmful), "bank-a": 0 if harmful else int(sign == 1),
                                 "bank-b": 0 if harmful else int(sign == -1)}})
    return rows


class QuestionSelectorTests(unittest.TestCase):
    def test_learns_utility_not_source_membership_and_eval_cannot_change_prediction(self):
        train, tune, evaluation = [utility_rows(s) for s in ("train", "tune", "eval")]
        selector = fit_selector(train, tune, ["bank-a", "bank-b"], {"bank-a": [1., 0.], "bank-b": [-1., 0.]}, "profile")
        self.assertEqual(predict(selector, [1., 0.])["selected_bank"], "bank-a")
        self.assertEqual(predict(selector, [-1., 0.])["selected_bank"], "bank-b")
        predictions = {r["sample_id"]: {"selector_sha256": selector["selector_sha256"], "decision": predict(selector, r["feature"])} for r in evaluation}
        result = evaluate(selector, evaluation, predictions)
        self.assertEqual(result["methods"]["selector"]["correct"], len(evaluation))
        self.assertEqual(result["selector_regret_count"], 0)
        before = deepcopy(selector)
        changed = deepcopy(evaluation)
        for r in changed:
            r["rewards"]["bank-a"], r["rewards"]["bank-b"] = r["rewards"]["bank-b"], r["rewards"]["bank-a"]
        self.assertEqual(evaluate(selector, changed, predictions)["methods"]["selector"]["correct"], 0)
        self.assertEqual(selector, before)
        leaked = deepcopy(predictions)
        leaked[evaluation[0]["sample_id"]]["decision"]["selected_bank"] = "bank-b"
        with self.assertRaisesRegex(ValueError, "frozen question-only"):
            evaluate(selector, evaluation, leaked)

    def test_harmful_memory_abstains_and_split_overlap_rejected(self):
        train, tune = utility_rows("train", harmful=True), utility_rows("tune", harmful=True)
        s = fit_selector(train, tune, ["bank-a", "bank-b"], {"bank-a": [1., 0.], "bank-b": [-1., 0.]}, "profile")
        self.assertEqual(predict(s, [1., 0.])["selected_bank"], NO_MEMORY)
        self.assertEqual(s["fixed_bank_from_train"], NO_MEMORY)
        with self.assertRaisesRegex(ValueError, "split isolation"):
            fit_selector(utility_rows("eval"), tune, ["bank-a", "bank-b"], s["card_features"], "profile")
        tune[0]["sample_id"] = train[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_selector(train, tune, ["bank-a", "bank-b"], s["card_features"], "profile")

    def test_partition_authenticates_content_and_is_disjoint(self):
        train = [{"question": f"Question {i}?", "answer": f"Answer {i}"} for i in range(40)]
        test = [{"question": "Official test never loaded", "answer": "secret"}]
        manifest = create_gsm8k_split_manifest(train, test, bank_source_size=15, calibration_val_size=20, seed=42, dataset_revision="pinned")
        source = next(e for e in manifest["samples"] if e["logical_split"] == "bank-source")
        packets = [{"evidence": [{"sample_id": source["sample_id"], "question": train[source["source_index"]]["question"]}]}]
        entries = partition_samples(manifest, packets, counts=(8, 6, 6))
        self.assertEqual(entries, partition_samples(manifest, packets, counts=(8, 6, 6)))
        self.assertEqual([sum(e["selector_split"] == s for e in entries) for s in ("train", "tune", "eval")], [8, 6, 6])
        self.assertEqual(len({e["question_sha256"] for e in entries}), 20)
        self.assertTrue(all(e["dataset_split"] == "train" and e["logical_split"] == "calibration-val" for e in entries))
        checked_dataset_rows(train, entries)
        bad = deepcopy(train)
        bad[entries[0]["source_index"]]["answer"] += " changed"
        with self.assertRaisesRegex(ValueError, "revision drift"):
            checked_dataset_rows(bad, entries)
        packets[0]["evidence"][0]["sample_id"] = entries[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "Construction sample"):
            partition_samples(manifest, packets, counts=(8, 6, 6))


@unittest.skipIf(torch is None, "Torch required")
class QuestionSelectorRuntimeTests(unittest.TestCase):
    def runtime(self):
        from memgen.model.side_kv import SideKVAttentionController
        from memgen.model.v4_3_runtime import V43UnifiedRuntime
        model = tiny_model(torch.bfloat16)
        controller = SideKVAttentionController(model=model, layer_number=24)
        self.addCleanup(controller.close)
        gate = SimpleNamespace(config=SimpleNamespace(layer_number=24, risk_role="online_joint_control", rearm_low_entropy_token_count=2))
        return V43UnifiedRuntime(model=model, tokenizer=PrefixTokenizer(box_after=4), device="cpu", gate=gate, controller=controller)

    def test_consumption_matches_existing_native_prefix_branch(self):
        from memgen.model.v4_3_question_selector import generate, encode_text
        from memgen.model.v4_3_prefix_equivalence import prefix_bank, run_equivalence
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.runtime()
            r = {"bank_id": "v43-bank-a", "descriptor": "Check remaining time.", "record_sha256": "a"}
            memory = prefix_bank(Path(tmp), r, runtime, "eq")
            q = "What speed is needed?"
            prefix, result = generate(runtime, q, r["descriptor"], memory)
            _, _, kv, _ = run_equivalence(runtime, q, r["descriptor"], *memory, atol=.05, rtol=.02)
            self.assertEqual(result["continuation_token_ids"], kv["continuation_token_ids"])
            self.assertEqual(result["initial_cache_length"], kv["initial_cache_length"])
            self.assertEqual(result["active_step_count"], 0)
            self.assertAlmostEqual(np.linalg.norm(encode_text(runtime, q)), 1., places=5)

    def test_end_to_end_orders_train_tune_freeze_predict_eval_and_readonly_resume(self):
        import scripts.run_v4_3_question_selector as driver
        from memgen.model import v4_3_question_selector as runtime_module
        from memgen.model.v4_3_prefix_equivalence import prefix_bank
        from memgen.experience.v4_3_bank import text_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.runtime()
            records = [{"bank_id": "v43-bank-"+b, "descriptor": "Check " + b + " quantities.", "record_sha256": b} for b in ("a", "b")]
            eq = {"profile_sha256": "eq"}
            for r in records:
                prefix_bank(root / "eq" / "prefix_kv", r, runtime, "eq")
            dataset = [{"question": f"How many items in group {i}?", "answer": "Calculate.\n#### 37"} for i in range(4)]
            entries = [{"sample_id": f"sample-{i}", "selector_split": s, "source_index": i,
                        "question_sha256": text_hash(dataset[i]["question"]), "answer_sha256": text_hash(dataset[i]["answer"])}
                       for i, s in enumerate(("train", "train", "tune", "eval"))]
            profile = seal({"samples": entries, "bank_ids": [r["bank_id"] for r in records], "dataset": {"revision": "fixture"},
                            "reasoner": {}, "policy": POLICY}, "profile_sha256")
            args = SimpleNamespace(output_dir=root / "out", equivalence_dir=root / "eq", bank_dir=root / "bank", side_kv_dir=root / "side",
                                   cache_manifest=root / "source" / "cache.json", device="cpu", resume=True, plan_only=False, validate_only=False)
            prepared = ({}, {}, None, records)
            dataset_module = ModuleType("datasets")
            dataset_module.load_dataset = lambda *a, **kw: dataset
            real_generate = runtime_module.generate
            calls = []
            def guarded_generate(rt, question, *a, **kw):
                if question == dataset[3]["question"]:
                    self.assertTrue((args.output_dir / "selector.json").exists())
                    self.assertTrue((args.output_dir / "samples" / "sample-3" / "prediction.json").exists())
                calls.append(question)
                return real_generate(rt, question, *a, **kw)
            with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "prepare", return_value=(prepared, eq, profile)), \
                 patch.dict("sys.modules", {"datasets": dataset_module}), patch.object(runtime_module, "load_runtime", return_value=runtime), \
                 patch.object(runtime_module, "generate", side_effect=guarded_generate):
                driver.main()
                self.assertEqual(len(calls), 12)
                result = read_json(args.output_dir / "report.json")
                self.assertTrue(result["complete"])
                self.assertEqual(result["evaluation"]["methods"]["selector"]["count"], 1)
                files = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                args.validate_only = True
                driver.main()
                self.assertEqual(len(calls), 12)
                self.assertEqual(files, {p: p.stat().st_mtime_ns for p in files})
                # Missing pre-outcome decisions cannot be reconstructed post hoc.
                (args.output_dir / "samples" / "sample-3" / "prediction.json").unlink()
                with self.assertRaisesRegex(ValueError, "previously frozen"):
                    driver.main()


if __name__ == "__main__":
    unittest.main()
