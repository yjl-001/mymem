from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from memgen.experience.v4_3_artifacts import atomic_json, read_json
from memgen.experience.v4_3_bank import canonical_hash, seal, text_hash
from tests.test_v4_3_side_kv import torch, tiny_model
from tests.test_v4_3_prefix_equivalence import PrefixTokenizer


@unittest.skipIf(torch is None, "Torch required")
class MemoryTimingTests(unittest.TestCase):
    def runtime(self, trigger=True, box_after=4):
        from memgen.model.e1_runtime import EntropyRiskGate, EntropyRiskGateConfig
        from memgen.model.v3_runtime import EntropyHysteresisGate, EntropyHysteresisConfig
        from memgen.model.v4_3_runtime import V43UnifiedRuntime
        from memgen.model.side_kv import SideKVAttentionController
        model = tiny_model()
        high = -1. if trigger else 1e6
        diagnostic = EntropyRiskGate(recovery_center=torch.ones(16), persistence_center=-torch.ones(16),
            config=EntropyRiskGateConfig(layer_number=24, sink_token_count=0, entropy_threshold=high, risk_threshold=-3.))
        gate = EntropyHysteresisGate(diagnostic_gate=diagnostic, config=EntropyHysteresisConfig(
            layer_number=24, sink_token_count=0, high_entropy_threshold=high, low_entropy_threshold=-2.,
            risk_threshold=-3., risk_role="online_joint_control", rearm_low_entropy_token_count=2))
        controller = SideKVAttentionController(model=model, layer_number=24)
        self.addCleanup(controller.close)
        return V43UnifiedRuntime(model=model, tokenizer=PrefixTokenizer(box_after=box_after), device="cpu", gate=gate, controller=controller)

    def test_contextual_append_matches_full_native_forward_all_layers(self):
        from memgen.model.v4_3_memory_timing import append_tokens, memory_tokens
        from memgen.model.e1_runtime import clone_cache
        rt = self.runtime()
        for dtype, atol in ((torch.float32, 1e-5), (torch.bfloat16, .04)):
            rt.model.to(dtype=dtype)
            q = rt.visible_prefix("How many remain?", None)
            m = memory_tokens(rt, "Subtract used items from the total.")
            with torch.inference_mode():
                original = rt.model(input_ids=rt._tensor(q), use_cache=True).past_key_values
                before = clone_cache(original)
                appended = append_tokens(rt, original, m)
                full = rt.model(input_ids=rt._tensor(q+m), use_cache=True)
            self.assertTrue(torch.allclose(appended.logits[:, -1], full.logits[:, -1], atol=atol, rtol=.02))
            self.assertEqual(len(appended.past_key_values.to_legacy_cache()), 24)
            for old, actual, expected in zip(before.to_legacy_cache(), appended.past_key_values.to_legacy_cache(), full.past_key_values.to_legacy_cache()):
                for a, b, c in zip(old, actual, expected):
                    self.assertTrue(torch.equal(a, b[:, :, :len(q)]))
                    self.assertTrue(torch.allclose(b, c, atol=atol, rtol=.02))

    def test_real_gate_first_token_and_prompt_end_cache_accounting(self):
        from memgen.model.v4_3_memory_timing import generate_delayed
        from memgen.model.v4_3_question_selector import generate
        rt = self.runtime()
        _, baseline = generate(rt, "How many remain?")
        for mode, expected_position in (("prompt_end", 0), ("entropy_gate", 1)):
            prefix, result = generate_delayed(rt, "How many remain?", "Check the total.", mode)
            self.assertEqual(result["activation_count"], 1)
            self.assertEqual(result["injection_generated_token_count"], expected_position)
            self.assertEqual(len(result["continuation_token_ids"]), 4)
            self.assertEqual(result["final_cache_length"], len(prefix)+4+result["injected_token_count"]-1)
            self.assertEqual(result["continuation_token_ids"][:expected_position], baseline["continuation_token_ids"][:expected_position])
            self.assertEqual(len(result["gate_traces"]), expected_position)
            self.assertEqual(len(rt.controller.traces), 0)

    def test_no_trigger_abstention_and_answer_marker_preserve_vanilla(self):
        from memgen.model.v4_3_memory_timing import generate_delayed
        from memgen.model.v4_3_question_selector import generate
        rt = self.runtime(trigger=False)
        _, base = generate(rt, "How many remain?")
        _, result = generate_delayed(rt, "How many remain?", "Check the total.", "entropy_gate")
        self.assertEqual(base["continuation_token_ids"], result["continuation_token_ids"])
        self.assertEqual(result["activation_count"], 0)
        self.assertEqual(len(result["gate_traces"]), 3)
        with patch.object(rt.gate, "probe", side_effect=AssertionError("Abstention must not probe")):
            _, abstain = generate_delayed(rt, "How many remain?", None, "entropy_gate")
        self.assertEqual(abstain["continuation_token_ids"], base["continuation_token_ids"])
        rt.tokenizer.decode = lambda ids, **kw: "final answer" if ids else ""
        with patch.object(rt.gate, "probe", side_effect=AssertionError("Answer marker must suppress gate")):
            _, stopped = generate_delayed(rt, "How many remain?", "Check total.", "entropy_gate", maximum_completion_tokens=3)
        self.assertEqual(stopped["non_activation_reason"], "answer_marker_before_trigger")

    def test_memory_excluded_from_scoring_and_completion_budget(self):
        from memgen.model.v4_3_memory_timing import generate_delayed, memory_tokens
        from scripts.audit_v4_3_unified_memory import score_branch
        rt = self.runtime(box_after=10000)
        q, card = "How many remain?", "Check units."
        prefix, result = generate_delayed(rt, q, card, "prompt_end", maximum_completion_tokens=3)
        self.assertEqual(len(result["continuation_token_ids"]), 3)
        self.assertEqual(result["stop_reason"], "maximum_completion_tokens")
        self.assertGreater(result["injected_token_count"], 3)
        observed = []
        def decode(ids, **kw):
            observed.append(list(ids))
            return "reasoning"
        with patch.object(rt.tokenizer, "decode", side_effect=decode):
            scored = score_branch(rt.tokenizer, prefix, len(prefix), result, "Solution.\n#### 37")
        self.assertTrue(all(ids == result["continuation_token_ids"] for ids in observed))
        self.assertEqual(scored["strict_reward"], 0.)
        rt.model.config.max_position_embeddings = len(prefix)+len(memory_tokens(rt, card))+2
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            generate_delayed(rt, q, card, "prompt_end", maximum_completion_tokens=3)

    def test_driver_resume_decisions_and_readonly_report(self):
        import scripts.run_v4_3_memory_timing as driver
        import memgen.model.v4_3_memory_timing as consumer
        import memgen.model.v4_3_question_selector as encoder
        from scripts.audit_v4_3_unified_memory import score_branch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = self.runtime()
            dataset = [{"question": f"How many in group {i}?", "answer": "Solution.\n#### 37"} for i in range(2)]
            entries = [{"sample_id": f"sample-{i}", "source_index": i, "selector_split": "eval",
                        "question_sha256": text_hash(d["question"]), "answer_sha256": text_hash(d["answer"])} for i, d in enumerate(dataset)]
            bindings = {e["sample_id"]: {"selected_bank": "bank-a" if i == 0 else "no_memory"} for i, e in enumerate(entries)}
            profile = seal({"samples": entries, "bindings": bindings, "reasoner": {}, "dataset": {"revision": "fixture"},
                            "bank_ids": ["bank-a"], "fixed_bank_reference": "bank-a", "selector": "frozen_semantic_question_only",
                            "semantic_threshold": .5, "evaluation_role": "diagnostic"}, "profile_sha256")
            refs = {}
            for e, d in zip(entries, dataset):
                prefix, result = encoder.generate(rt, d["question"])
                scored = score_branch(rt.tokenizer, prefix, len(prefix), result, d["answer"])
                refs[e["sample_id"]] = {b: deepcopy(scored) for b in ("baseline", "native_prefix_kv", "fixed_from_train")}
            args = SimpleNamespace(output_dir=root / "out", selector_dir=root / "selector", equivalence_dir=root / "eq",
                bank_dir=root / "bank", side_kv_dir=root / "side", cache_manifest=root / "source" / "cache.json",
                device="cpu", resume=True, validate_only=False, plan_only=False)
            dataset_module = ModuleType("datasets")
            dataset_module.load_dataset = lambda *a, **kw: dataset
            calls = []
            real = consumer.generate_delayed
            def guarded(*a, **kw):
                self.assertTrue((args.output_dir / "samples" / "sample-0" / "decision.json").exists())
                calls.append(a[-1])
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                return real(*a, **kw)
            with patch.object(driver, "parse_args", return_value=args), \
                 patch.object(driver, "prepare", return_value=(profile, refs, [{"bank_id": "bank-a", "descriptor": "Check total."}], rt.gate)), \
                 patch.dict("sys.modules", {"datasets": dataset_module}), patch.object(encoder, "load_runtime", return_value=rt):
                with patch.object(consumer, "generate_delayed", side_effect=guarded):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        driver.main()
                saved = args.output_dir / "samples" / "sample-0" / "prompt_end.json"
                timestamp = saved.stat().st_mtime_ns
                with patch.object(consumer, "generate_delayed", wraps=real) as resumed:
                    driver.main()
                self.assertEqual(resumed.call_count, 1)
                self.assertEqual(saved.stat().st_mtime_ns, timestamp)
                summary = read_json(args.output_dir / "brief_summary.json")
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["gate"]["selected_count"], 1)
                self.assertEqual(summary["gate"]["activation_count"], 1)
                self.assertEqual(summary["overall"]["baseline"]["memory_use_count"], 0)
                files = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                args.validate_only = True
                with patch.object(encoder, "load_runtime", side_effect=AssertionError("Must not load model")):
                    driver.main()
                self.assertEqual(files, {p: p.stat().st_mtime_ns for p in files})
                (args.output_dir / "samples" / "sample-0" / "decision.json").unlink()
                with self.assertRaisesRegex(ValueError, "lack a prior frozen decision"):
                    driver.main()

    def test_prepare_binds_existing_semantic_decisions_and_rejects_source_drift(self):
        import scripts.run_v4_3_memory_timing as driver
        from memgen.experience.v4_3_question_selector import POLICY, predict
        from scripts import run_v4_3_question_selector as source
        from memgen.model.v3_runtime import EntropyHysteresisGate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = self.runtime()
            entries = [{"sample_id": f"s-{i}", "question_sha256": f"q-{i}", "selector_split": s}
                       for i, s in enumerate(("train", "train", "tune", "eval"))]
            sp = seal({"samples": entries, "seed": 43, "bank_ids": ["bank-a", "bank-b"], "reasoner": {},
                       "runtime_versions": {}, "dataset": {"revision": "fixture"}, "policy": POLICY}, "profile_sha256")
            atomic_json(root / "profile.json", sp)
            cards = {"bank-a": [1., 0.], "bank-b": [-1., 0.]}
            atomic_json(root / "card_features.json", seal({"profile_sha256": sp["profile_sha256"], "features": cards}))
            def write_sample(e, outcomes=True):
                common = dict(profile_sha256=sp["profile_sha256"], sample_id=e["sample_id"], question_sha256=e["question_sha256"])
                sample = source.sample_path(root, e)
                atomic_json(sample / "feature.json", seal({**common, "feature": [1., 0.]}))
                if outcomes:
                    for action in ("no_memory", "bank-a", "bank-b"):
                        result = {"strict_reward": float(action == "bank-a"), "continuation_token_ids": [1],
                                  "continuation_token_ids_sha256": canonical_hash([1])}
                        atomic_json(sample / (action+".json"), seal({**common, "action": action, "result": result}))
            for e in entries[:3]:
                write_sample(e)
            selector = source.fit_and_save(root, sp, cards)
            e = entries[3]
            common = dict(profile_sha256=sp["profile_sha256"], sample_id=e["sample_id"], question_sha256=e["question_sha256"])
            atomic_json(source.sample_path(root, e) / "prediction.json", seal({**common,
                        "selector_sha256": selector["selector_sha256"], "decision": predict(selector, [1., 0.])}))
            write_sample(e)
            source.write_report(root, source.report(root, sp, selector))
            risk_path = root / "risk.pt"
            torch.save({"reasoner": {}}, risk_path)
            args = SimpleNamespace(selector_dir=root, token_risk_artifact=risk_path, device="cpu")
            with patch.object(source, "prepare", return_value=((None, None, None, []), {"source_experiment": {"source_reasoner": {}}}, sp)), \
                 patch.object(EntropyHysteresisGate, "from_token_artifact", return_value=rt.gate):
                profile, refs, _, _ = driver.prepare(args)
                self.assertEqual(profile["bindings"][e["sample_id"]]["selected_bank"], "bank-a")
                self.assertEqual(refs[e["sample_id"]]["native_prefix_kv"]["strict_reward"], 1.)
                bad = read_json(root / "report.json")
                bad["complete"] = False
                atomic_json(root / "report.json", bad)
                with self.assertRaisesRegex(ValueError, "complete authenticated"):
                    driver.prepare(args)


class TimingShellTests(unittest.TestCase):
    def test_dispatch_paths_and_flags(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "record-python"
            script.write_text('#!/bin/sh\nfor arg in "$@"; do echo "$arg"; done\n')
            script.chmod(0o755)
            result = subprocess.run(["bash", "test.sh", "timing", "--plan-only"], cwd=root,
                env={**os.environ, "MEMGEN_PYTHON_BIN": str(script), "MEMGEN_V43_SELECTOR_ROOT": "/fixture/selector",
                     "MEMGEN_V43_TIMING_ROOT": "/fixture/timing"}, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = result.stdout.splitlines()
            self.assertEqual(args[0], "scripts/run_v4_3_memory_timing.py")
            self.assertEqual(args[args.index("--selector-dir")+1], "/fixture/selector")
            self.assertEqual(args[args.index("--output-dir")+1], "/fixture/timing")
            self.assertIn("--resume", args)
            self.assertIn("--plan-only", args)


if __name__ == "__main__":
    unittest.main()
