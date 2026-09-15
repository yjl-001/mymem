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
from tests.test_v4_3_side_kv import torch
from tests import test_v4_3_memory_timing as timing_tests


@unittest.skipIf(torch is None, "Torch required")
class GatedPrefixTests(unittest.TestCase):
    def runtime(self, trigger=True, box_after=5):
        return timing_tests.MemoryTimingTests.runtime(self, trigger=trigger, box_after=box_after)

    def memory(self, runtime, root):
        from memgen.model.v4_3_prefix_equivalence import prefix_bank
        from memgen.model.v4_3_gated_prefix import virtual_prefix
        record = {"bank_id": "v43-bank-fixture", "descriptor": "Subtract used items from the total.", "record_sha256": "record"}
        ids, tensors = prefix_bank(root, record, runtime, "eq")
        return record, ids, tensors, virtual_prefix(runtime.model, ids, tensors)

    def test_negative_prefix_matches_native_full_prefix_when_visible_from_start(self):
        from memgen.model.v4_3_gated_prefix import VirtualPrefixReader
        from memgen.model.v4_3_prefix_equivalence import split_prefix
        with tempfile.TemporaryDirectory() as tmp:
            for dtype, atol in ((torch.float32, 2e-5), (torch.bfloat16, .03)):
                rt = self.runtime()
                rt.model.to(dtype=dtype)
                record, ids, tensors, memory = self.memory(rt, Path(tmp) / str(dtype))
                originals = {k: v.clone() for k, v in tensors.items()}
                rt.controller.close()
                full = split_prefix(rt, "How many remain?", record["descriptor"], ids)
                query = full[len(ids):]
                reader = VirtualPrefixReader(rt.model, memory)
                try:
                    with torch.inference_mode():
                        reference = rt.model(input_ids=rt._tensor(full), use_cache=True)
                        reader.activate()
                        actual = rt.model(input_ids=rt._tensor(query), use_cache=True)
                    self.assertTrue(torch.allclose(actual.logits, reference.logits[:, len(ids):], atol=atol, rtol=.03))
                    self.assertEqual(actual.past_key_values.get_seq_length(), len(query))
                    self.assertEqual(reader.counts, [1]*24)
                    for layer, pair in enumerate(actual.past_key_values.to_legacy_cache()):
                        # Values have no rotary positions and should match the native prefix reference.
                        self.assertTrue(torch.allclose(pair[1], reference.past_key_values.to_legacy_cache()[layer][1][:, :, len(ids):], atol=atol, rtol=.03))
                        self.assertTrue(torch.equal(memory[layer][1], tensors[f"{layer}.v"]))
                    self.assertTrue(all(torch.equal(v, tensors[k]) for k, v in originals.items()))
                finally:
                    reader.close()

    def test_gate_reads_next_forward_all_layers_preserves_history_and_no_replay(self):
        from memgen.model.v4_3_gated_prefix import generate_gated
        from memgen.model.v4_3_question_selector import generate
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime()
            _, _, _, memory = self.memory(rt, Path(tmp))
            _, baseline = generate(rt, "How many remain?")
            lengths = []
            def observe(module, args, kwargs):
                lengths.append(kwargs["input_ids"].shape[-1])
            hook = rt.model.register_forward_pre_hook(observe, with_kwargs=True)
            try:
                prefix, result = generate_gated(rt, "How many remain?", memory)
            finally:
                hook.remove()
            self.assertEqual(result["gate_trigger_count"], 1)
            self.assertEqual(result["activation_count"], 1)
            self.assertEqual(result["activation"]["unchanged_generated_token_count"], 2)
            self.assertEqual(result["continuation_token_ids"][:2], baseline["continuation_token_ids"][:2])
            self.assertTrue(result["history_kv_preserved"])
            self.assertEqual(result["layer_read_counts"], [3]*24)
            self.assertTrue(all(0 < r["memory_attention_mass"] < 1 for r in result["first_read_by_layer"].values()))
            self.assertEqual(result["final_cache_length"], len(prefix)+5-1)
            self.assertEqual(lengths, [len(prefix)-1]+[1]*5)
            self.assertEqual(result["inserted_token_count"], 0)
            self.assertEqual(result["replayed_token_count"], 0)

    def test_gate_never_triggers_or_stops_before_first_read(self):
        from memgen.model.v4_3_gated_prefix import generate_gated
        from memgen.model.v4_3_question_selector import generate
        with tempfile.TemporaryDirectory() as tmp:
            for trigger, box_after in ((False, 5), (True, 2)):
                rt = self.runtime(trigger, box_after)
                _, _, _, memory = self.memory(rt, Path(tmp) / str(trigger))
                _, baseline = generate(rt, "How many remain?")
                _, result = generate_gated(rt, "How many remain?", memory)
                self.assertEqual(result["continuation_token_ids"], baseline["continuation_token_ids"])
                self.assertEqual(result["activation_count"], 0)
                self.assertEqual(result["gate_trigger_count"], int(trigger))
                self.assertEqual(result["layer_read_counts"], [0]*24)
                if trigger:
                    self.assertEqual(result["non_activation_reason"], "generation_stopped_before_first_memory_read")

    def test_budget_rope_checks_and_attention_teardown_on_failure(self):
        from memgen.model.v4_3_gated_prefix import generate_gated, virtual_prefix
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(box_after=1000)
            _, ids, tensors, memory = self.memory(rt, Path(tmp))
            rt.controller.close()
            originals = [layer.self_attn.forward for layer in rt.model.model.layers]
            with patch("memgen.model.v4_3_gated_prefix.history_equal", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "rewrote historical"):
                    generate_gated(rt, "How many remain?", memory, maximum_completion_tokens=4)
            self.assertEqual(originals, [layer.self_attn.forward for layer in rt.model.model.layers])
            rt.model.model.rotary_emb.rope_type = "dynamic"
            with self.assertRaisesRegex(ValueError, "fixed-frequency"):
                virtual_prefix(rt.model, ids, tensors)
            rt.model.model.rotary_emb.rope_type = "default"
            rt.model.config.max_position_embeddings = 2
            with self.assertRaisesRegex(ValueError, "exceeds context"):
                generate_gated(rt, "How many remain?", memory)

    def test_driver_runs_once_resumes_without_model_and_binds_source(self):
        import scripts.run_v4_3_gated_prefix as driver
        import memgen.model.v4_3_gated_prefix as consumer
        import memgen.model.v4_3_question_selector as encoder
        from scripts.audit_v4_3_unified_memory import score_branch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = self.runtime()
            record, _, _, _ = self.memory(rt, root / "eq" / "prefix_kv")
            dataset = [{"question": f"How many remain in group {i}?", "answer": "Solution.\n#### 37"} for i in range(2)]
            entries = [{"sample_id": f"sample-{i}", "source_index": i, "selector_split": "eval",
                        "question_sha256": text_hash(d["question"]), "answer_sha256": text_hash(d["answer"])} for i, d in enumerate(dataset)]
            bindings = {e["sample_id"]: {"selected_bank": record["bank_id"] if i == 0 else "no_memory"} for i, e in enumerate(entries)}
            profile = seal({"samples": entries, "bindings": bindings, "reasoner": {}, "dataset": {"revision": "fixture"},
                            "bank_ids": [record["bank_id"]], "fixed_bank_reference": record["bank_id"],
                            "selector": "frozen_semantic_question_only", "semantic_threshold": .5,
                            "evaluation_role": "diagnostic", "source_equivalence_profile_sha256": "eq",
                            "gate_config": rt.gate.config.to_dict()}, "profile_sha256")
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
            real = consumer.generate_gated
            def guarded(*a, **kw):
                self.assertTrue((args.output_dir / "samples" / "sample-0" / "decision.json").exists())
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("simulated interruption")
                return real(*a, **kw)
            with patch.object(driver, "parse_args", return_value=args), \
                 patch.object(driver, "prepare", return_value=(profile, refs, [record], rt.gate)), \
                 patch.dict("sys.modules", {"datasets": dataset_module}), patch.object(encoder, "load_runtime", return_value=rt), \
                 patch.object(consumer, "generate_gated", side_effect=guarded):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    driver.main()
                saved = args.output_dir / "samples" / "sample-0" / "baseline.json"
                timestamp = saved.stat().st_mtime_ns
                driver.main()
                self.assertEqual(len(calls), 2)
                self.assertEqual(saved.stat().st_mtime_ns, timestamp)
                summary = read_json(args.output_dir / "brief_summary.json")
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["gate"]["activation_count"], 1)
                self.assertEqual(summary["integrity"]["history_kv_failure_count"], 0)
                self.assertEqual(summary["integrity"]["pre_activation_trajectory_mismatch_count"], 0)
                files = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                args.validate_only = True
                with patch.object(encoder, "load_runtime", side_effect=AssertionError("Must not load model")):
                    driver.main()
                self.assertEqual(files, {p: p.stat().st_mtime_ns for p in files})
                refs["sample-0"]["baseline"]["strict_reward"] = 0.
                with self.assertRaisesRegex(ValueError, "reference result drift"):
                    driver.main()


class GatedPrefixShellTests(unittest.TestCase):
    def test_dispatch(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "record-python"
            script.write_text('#!/bin/sh\nfor arg in "$@"; do echo "$arg"; done\n')
            script.chmod(0o755)
            result = subprocess.run(["bash", "test.sh", "gated-prefix", "--plan-only"], cwd=root,
                env={**os.environ, "MEMGEN_PYTHON_BIN": str(script), "MEMGEN_V43_SELECTOR_ROOT": "/fixture/selector",
                     "MEMGEN_V43_GATED_PREFIX_ROOT": "/fixture/gated"}, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = result.stdout.splitlines()
            self.assertEqual(args[0], "scripts/run_v4_3_gated_prefix.py")
            self.assertEqual(args[args.index("--output-dir")+1], "/fixture/gated")
            self.assertIn("--resume", args)
            self.assertIn("--plan-only", args)


if __name__ == "__main__":
    unittest.main()
