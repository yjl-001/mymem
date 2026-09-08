from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path

from tests.test_v4_3_side_kv import TinyTokenizer, tiny_model, torch

if torch is not None:
    from memgen.model.side_kv import SideKVAttentionController, SideKVMemory
    from memgen.model.v4_3_runtime import V43UnifiedRuntime
    from memgen.model.v4_3_side_kv import MEMORY_SCORE_BIAS
    from memgen.model.v4_3_side_kv import V43SideKVCompiler, V43SideKVBankLoader, save_compiled
    from tests.test_v4_3_side_kv import tier_subset
    from tests.test_v4_3_bank import fixture
    from memgen.experience.v4_3_bank import build_outputs
    from tests.test_v4_oracle_runtime import FakeGate, FakeController, FakeTokenizer, FakeModel, cache_with_length


@unittest.skipIf(torch is None, "Torch/Transformers test environment unavailable")
class V43RuntimeTests(unittest.TestCase):
    def test_real_qwen_exact_prefix_four_branches_and_native_cache(self):
        model, tokenizer = tiny_model(torch.bfloat16), TinyTokenizer(box_after=40)
        outputs = build_outputs(**fixture())
        records, manifest = tier_subset(outputs, count=3)
        reasoner = dict(model_name="tiny-local-qwen", model_revision="a" * 40,
                        tokenizer_revision="b" * 40, model_sequence_limit=4096)
        tensors, compiled = V43SideKVCompiler(model=model, tokenizer=tokenizer, reasoner=reasoner).compile(records, manifest)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_compiled(Path(tmp), tensors, compiled)
            loader = V43SideKVBankLoader(path, source_manifest=manifest, source_records=records, expected_reasoner=reasoner)
        controller = SideKVAttentionController(model=model, layer_number=24, audit_canonical_rope=True,
                                               memory_score_normalization="log_valid_slots", memory_score_bias=MEMORY_SCORE_BIAS)
        class Gate:
            config = SimpleNamespace(layer_number=24, risk_role="online_joint_control", rearm_low_entropy_token_count=2, low_entropy_threshold=0.)
            def probe(self, **kwargs):
                output = kwargs["model"](input_ids=kwargs["boundary_token"], attention_mask=kwargs["attention_mask"],
                    past_key_values=kwargs["past_key_values"], use_cache=True, return_dict=True)
                return SimpleNamespace(entropy=1., output=output)
        runtime = V43UnifiedRuntime(model=model, tokenizer=tokenizer, device="cpu", gate=Gate(), controller=controller)
        memories = {"baseline": None}
        for i, name in enumerate(("matched", "near_wrong", "far_wrong")):
            memories[name] = loader.get_memory(records[i]["bank_id"], device="cpu", dtype=torch.bfloat16)
        try:
            results, parity = runtime.run_latent(prefix=[7, 8, 9, 10], prompt_count=2, memories=memories)
            self.assertTrue(parity["branch_cache_storage_is_independent"])
            self.assertTrue(parity["initial_cache_tensors_exactly_equal"])
            for name, result in results.items():
                self.assertEqual(result["initial_cache_length"], 3)
                self.assertLessEqual(result["active_step_count"], 32)
                self.assertEqual(result["final_cache_length"], 3 + len(result["continuation_token_ids"]))
                if name != "baseline":
                    self.assertGreater(result["active_step_count"], 0)
                    self.assertGreater(result["first_step_logits_kl"], 0)
            self.assertTrue(any(r["post_memory_native_step_count"] > 0 for k, r in results.items() if k != "baseline"))
        finally:
            controller.close()

    def fake_runtime(self, box_after=None, low_threshold=0):
        controller = FakeController()
        gate = FakeGate(controller)
        gate.config = SimpleNamespace(layer_number=24, risk_role="online_joint_control", rearm_low_entropy_token_count=2, low_entropy_threshold=low_threshold)
        return V43UnifiedRuntime(model=FakeModel(), tokenizer=FakeTokenizer(box_after_completion_tokens=box_after),
                                 device="cpu", gate=gate, controller=controller)

    def test_memory_32_steps_then_native_to_complete_1024_budget(self):
        runtime = self.fake_runtime()
        result, _ = runtime.decode(prefix=[7, 8, 9, 10], prompt_count=2, cache=cache_with_length(3), memory=SimpleNamespace(memory_id="v43-bank-test"))
        self.assertEqual(result["active_step_count"], 32)
        self.assertEqual(len(result["continuation_token_ids"]), 1022)
        self.assertEqual(result["post_memory_native_step_count"], 990)
        self.assertEqual(result["activation_count"], 1)

    def test_two_low_tokens_unload_and_continue_to_boxed_answer(self):
        runtime = self.fake_runtime(box_after=40, low_threshold=1)
        result, _ = runtime.decode(prefix=[7, 8, 9, 10], prompt_count=2, cache=cache_with_length(3), memory=SimpleNamespace(memory_id="v43-bank-test"))
        self.assertEqual(result["active_step_count"], 2)
        self.assertEqual(len(result["continuation_token_ids"]), 38)
        self.assertEqual(result["stop_reason"], "completed_boxed_answer")

    def test_visible_prompt_is_content_matched_and_has_no_latent_memory(self):
        runtime = self.fake_runtime()
        runtime.tokenizer = TinyTokenizer(box_after=3)
        class Model(FakeModel):
            def __call__(self, *, past_key_values=None, **kwargs):
                return super().__call__(past_key_values=past_key_values, **kwargs)
        runtime.model = Model()
        descriptors = {"baseline": None, "matched": "Check the remaining time.", "near_wrong": "Check the total value.", "far_wrong": "Combine all variable coefficients."}
        memory_ids = {k: None if k == "baseline" else "v43-bank-" + k for k in descriptors}
        results, parity = runtime.run_visible(question="How many parcels remain?", descriptors=descriptors, memory_ids=memory_ids)
        self.assertFalse(parity["all_branches_share_exact_prefix"])
        self.assertGreater(len(set(parity["branch_prefix_sha256"].values())), 1)
        self.assertTrue(all(not r["attention_traces"] and r["memory_id"] is None for r in results.values()))


if __name__ == "__main__":
    unittest.main()
