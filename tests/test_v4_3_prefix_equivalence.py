from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

from tests.test_v4_3_side_kv import tiny_model, TinyTokenizer, torch
from memgen.experience.v4_3_bank import canonical_hash, seal
from memgen.experience.v4_3_artifacts import read_json, atomic_json
from scripts import audit_v4_3_prefix_equivalence as driver

if torch is not None:
    from memgen.model.v4_3_prefix_equivalence import (
        prefix_bank, memory_prefix_ids, restore_cache, split_prefix, run_equivalence,
    )
    from memgen.model.side_kv import SideKVAttentionController
    from memgen.model.v4_3_runtime import V43UnifiedRuntime


class PrefixTokenizer(TinyTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["role"] + " " + m["content"] + " end " for m in messages) + (" assistant" if add_generation_prompt else "")


@unittest.skipIf(torch is None, "Torch/Transformers required")
class PrefixEquivalenceTests(unittest.TestCase):
    def runtime(self, dtype=None):
        model = tiny_model(dtype or torch.float32)
        controller = SideKVAttentionController(model=model, layer_number=24)
        self.addCleanup(controller.close)
        gate = SimpleNamespace(config=SimpleNamespace(layer_number=24, risk_role="online_joint_control", rearm_low_entropy_token_count=2))
        return V43UnifiedRuntime(model=model, tokenizer=PrefixTokenizer(box_after=4), device="cpu", gate=gate, controller=controller)

    def record(self):
        return {"bank_id": "v43-bank-fixture", "descriptor": "Check the remaining time before dividing distance.", "record_sha256": "source"}

    def test_native_cache_disk_roundtrip_and_multiple_questions_float32_bfloat16(self):
        for dtype, atol, rtol in ((torch.float32, 1e-5, 1e-4), (torch.bfloat16, .05, .02)):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                runtime = self.runtime(dtype)
                record = self.record()
                ids, tensors = prefix_bank(Path(tmp), record, runtime, "profile")
                before = {k: t.clone() for k, t in tensors.items()}
                ids2, reloaded = prefix_bank(Path(tmp), record, runtime, "profile", validate_only=True)
                self.assertEqual(ids, ids2)
                self.assertEqual(len(reloaded), 48)
                self.assertTrue(all(torch.equal(tensors[k], reloaded[k]) for k in tensors))
                for question in ("How many minutes remain?", "What speed is required for the remaining distance?"):
                    prefix, text, kv, d = run_equivalence(runtime, question, record["descriptor"], ids, reloaded, atol=atol, rtol=rtol)
                    self.assertTrue(d["numerical_pass"], d)
                    self.assertTrue(d["behavioral_pass"], d)
                    self.assertEqual(d["step_count"], len(text["continuation_token_ids"]))
                    self.assertEqual(text["initial_cache_length"], len(prefix)-1)
                    self.assertEqual(kv["initial_cache_length"], len(prefix)-1)
                    self.assertEqual(kv["active_step_count"], 0)
                self.assertTrue(all(torch.equal(before[k], reloaded[k]) for k in before))
                c = restore_cache(reloaded, "cpu")
                c.layers[0].keys.add_(1)
                self.assertTrue(torch.equal(before["0.k"], reloaded["0.k"]))

    def test_token_boundary_and_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, record = self.runtime(), self.record()
            ids, tensors = prefix_bank(Path(tmp), record, runtime, "profile")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                split_prefix(runtime, "How many?", record["descriptor"], ids + [100])
            with self.assertRaisesRegex(ValueError, "identity/hash"):
                prefix_bank(Path(tmp), record, runtime, "other-profile", validate_only=True)
            path = Path(tmp) / (record["bank_id"] + ".safetensors")
            path.write_bytes(path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "identity/hash"):
                prefix_bank(Path(tmp), record, runtime, "profile", validate_only=True)

    def test_diagnostic_detects_bad_values_even_if_final_answers_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, record = self.runtime(), self.record()
            ids, tensors = prefix_bank(Path(tmp), record, runtime, "profile")
            tensors["0.v"] = tensors["0.v"] + 10
            _, _, _, d = run_equivalence(runtime, "How many?", record["descriptor"], ids, tensors, atol=1e-5, rtol=1e-4)
            self.assertFalse(d["numerical_pass"])
            self.assertGreater(d["max_logits_abs"], 1e-3)

    def test_persisted_complete_resume_validation_and_mismatch_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime, record = self.runtime(), self.record()
            case = dict(case_id="case", case_sha256="case-sha", sample_id="sample", bank_id=record["bank_id"])
            profile = seal({"configuration": {"atol": 1e-5, "rtol": 1e-4}, "cases": [case]}, "profile_sha256")
            ids, tensors = prefix_bank(root / "prefix_kv", record, runtime, profile["profile_sha256"])
            _, text, kv, d = run_equivalence(runtime, "How many?", record["descriptor"], ids, tensors, atol=1e-5, rtol=1e-4)
            text["strict_reward"] = kv["strict_reward"] = 1.
            branches = dict(baseline=text, visible_text=text, native_prefix_kv=kv, frozen_side_kv=text)
            manifest = read_json(root / "prefix_kv" / (record["bank_id"] + ".json"))
            row = seal({**case, "profile_sha256": profile["profile_sha256"], "branches": branches,
                        "diagnostics": d, "prefix_manifest_sha256": manifest["manifest_sha256"]})
            driver.validate_row(row, case, profile)
            atomic_json(root / "profile.json", profile)
            atomic_json(root / "cases" / "case.json", row)
            summary = driver.summarize([row], profile)
            self.assertTrue(summary["equivalence_passed"])
            atomic_json(root / "core_summary.json", summary)
            args = SimpleNamespace(output_dir=root, bank_dir=root.parent / "absent-bank", side_kv_dir=root.parent / "absent-side",
                                   cache_manifest=root.parent / "absent-source" / "cache.json", resume=True, validate_only=True, plan_only=False)
            prepared = ({}, {}, None, [], {})
            before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
            with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "prepare_experiment", return_value=(prepared, [case], profile)):
                driver.main()
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
            bad = deepcopy(row)
            bad["diagnostics"]["steps"][0]["within_tolerance"] = False
            bad["diagnostics"]["numerical_pass"] = False
            bad = seal(bad)
            self.assertFalse(driver.summarize([bad], profile)["equivalence_passed"])
            atomic_json(root / "cases" / "case.json", bad)
            atomic_json(root / "core_summary.json", driver.summarize([bad], profile))
            with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "prepare_experiment", return_value=(prepared, [case], profile)):
                with self.assertRaises(SystemExit) as error:
                    driver.main()
                self.assertEqual(error.exception.code, 2)

    def test_driver_generates_four_real_branches_and_resumes_without_model_load(self):
        from memgen.model.side_kv import SideKVMemory
        from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = self.record()
            model, tokenizer = tiny_model(torch.bfloat16), PrefixTokenizer(box_after=4)
            tokenizer.init_kwargs = {}
            reasoner = dict(model_name="tiny-local", model_revision="a"*40, tokenizer_revision="b"*40)
            evidence = {"e": {"question": "How many minutes remain?", "official_solution": "Calculate.\n#### 37"}}
            prompt = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, evidence["e"]["question"])
            case = dict(case_id="case", case_sha256="case-sha", sample_id="sample", bank_id=record["bank_id"],
                        experience_id="e", audit_layer="visible_content", prompt_token_count=len(prompt),
                        prefix_token_count=len(prompt), prefix_token_ids_sha256=canonical_hash(prompt))
            profile = seal({"configuration": {"atol": .05, "rtol": .02}, "cases": [case],
                            "source_experiment": {"reasoner": reasoner, "source_reasoner": reasoner}}, "profile_sha256")
            memory = SideKVMemory(memory_id=record["bank_id"], payload_hash="card", keys=torch.randn(1, 3, 8).bfloat16(),
                                  values=torch.randn(1, 3, 8).bfloat16(), slot_mask=torch.ones(3, dtype=torch.bool), layer_number=24)
            loader = SimpleNamespace(get_memory=lambda *a, **kw: memory)
            prepared = ({}, evidence, None, [record], {record["bank_id"]: loader})
            args = SimpleNamespace(output_dir=root / "out", bank_dir=root / "bank", side_kv_dir=root / "side", device="cpu",
                                   cache_manifest=root / "source" / "cache.json", token_risk_artifact=root / "risk.pt",
                                   resume=True, validate_only=False, plan_only=False, atol=.05, rtol=.02)
            class Gate:
                config = SimpleNamespace(layer_number=24, risk_role="online_joint_control", rearm_low_entropy_token_count=2, low_entropy_threshold=0.)
                def probe(self, **kw):
                    output = kw["model"](input_ids=kw["boundary_token"], attention_mask=kw["attention_mask"],
                                         past_key_values=kw["past_key_values"], use_cache=True, return_dict=True)
                    return SimpleNamespace(entropy=1., output=output)
            with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "prepare_experiment", return_value=(prepared, [case], profile)), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load_model, \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch("torch.load", return_value={"reasoner": reasoner}), \
                 patch("memgen.model.v3_runtime.EntropyHysteresisGate.from_token_artifact", return_value=Gate()):
                driver.main()
                self.assertEqual(load_model.call_count, 1)
                summary = read_json(args.output_dir / "core_summary.json")
                self.assertTrue(summary["equivalence_passed"])
                self.assertEqual(set(summary["overall"]), set(driver.BRANCHES))
                args.validate_only = True
                driver.main()
                self.assertEqual(load_model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
