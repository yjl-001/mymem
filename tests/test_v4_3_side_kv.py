from copy import deepcopy
from pathlib import Path
import re
import tempfile
import unittest
import subprocess
import sys

from memgen.experience import v4_3_bank as bank
from tests.test_v4_3_bank import fixture

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from memgen.model.v4_3_side_kv import V43SideKVCompiler, V43SideKVBankLoader, save_compiled
except ModuleNotFoundError:
    torch = None


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 127
    model_max_length = 4096
    def __init__(self, box_after=40):
        self.box_after = box_after
    def encode(self, text, add_special_tokens=False):
        return [4 + int(bank.text_hash(word)[:8], 16) % 120 for word in re.findall(r"\S+", text)]
    def decode(self, ids, skip_special_tokens=False):
        return "reasoning \\boxed{37}" if len(ids) >= self.box_after else "reasoning"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["role"] + " " + m["content"] for m in messages) + " assistant"


def tiny_model(dtype=None):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=128, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=24, num_attention_heads=2, num_key_value_heads=1,
                        max_position_embeddings=4096, eos_token_id=127, pad_token_id=0)
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).to(dtype=dtype or torch.float32).eval()


def tier_subset(outputs, tier="primary", count=2):
    records = outputs[f"{tier}_bank_records.jsonl"][:count]
    manifest = deepcopy(outputs[f"{tier}_bank_manifest.json"])
    ids = [r["bank_id"] for r in records]
    manifest.update(bank_ids=ids, bank_count=len(ids), record_count=len(ids),
                    record_order_sha256=bank.canonical_hash(ids),
                    record_sha256={r["bank_id"]: r["record_sha256"] for r in records},
                    evidence_count=sum(r["construction"]["distinct_sample_count"] for r in records))
    return records, bank.seal(manifest, "manifest_sha256")


@unittest.skipIf(torch is None, "Torch/Transformers test environment unavailable")
class V43SideKVTests(unittest.TestCase):
    def test_offline_import_does_not_load_training_model_or_tensorboard(self):
        code = "import sys; from memgen.model.v4_3_side_kv import V43SideKVCompiler; assert 'memgen.model.modeling_memgen' not in sys.modules; assert 'tensorboard' not in sys.modules"
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    @classmethod
    def setUpClass(cls):
        cls.outputs = bank.build_outputs(**fixture())
        cls.model, cls.tokenizer = tiny_model(), TinyTokenizer()
        cls.reasoner = dict(model_name="tiny-local-qwen", model_revision="a" * 40,
                            tokenizer_revision="b" * 40, model_sequence_limit=4096)
        cls.records, cls.manifest = tier_subset(cls.outputs)
        cls.compiler = V43SideKVCompiler(model=cls.model, tokenizer=cls.tokenizer, reasoner=cls.reasoner)
        cls.tensors, cls.compiled = cls.compiler.compile(cls.records, cls.manifest)

    def test_one_bank_one_memory_three_content_variants(self):
        self.assertEqual(self.compiled["bank_count"], 2)
        self.assertEqual(self.compiled["record_count"], 2)
        for record, entry in zip(self.records, self.compiled["records"]):
            self.assertEqual(entry["memory_id"], record["bank_id"])
            self.assertNotIn("role", entry)
            self.assertEqual(entry["kv_valid_slot_count"], 3 * len(self.tokenizer.encode(record["descriptor"])))
            self.assertEqual([v["name"] for v in entry["variants"]], ["raw_descriptor", "internal_principle", "hidden_note"])

    def test_raw_variant_matches_native_input_ln_kv_projection(self):
        ids = torch.tensor([self.tokenizer.encode(self.records[0]["descriptor"])])
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
            layer = self.model.model.layers[23]
            expected = layer.self_attn.k_proj(layer.input_layernorm(out.hidden_states[23])).reshape(ids.shape[1], 1, 8).transpose(0, 1)
        actual = self.tensors["keys"][0, :, :ids.shape[1]]
        self.assertTrue(torch.equal(actual, expected))

    def test_loader_authenticates_and_rejects_legacy_reference_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_compiled(Path(tmp), self.tensors, self.compiled)
            loader = V43SideKVBankLoader(path, source_manifest=self.manifest, source_records=self.records, expected_reasoner=self.reasoner)
            bid = self.records[0]["bank_id"]
            memory = loader.get_memory(bid, device="cpu", dtype=torch.float32)
            copy = loader.get_memory(bid, device="cpu", dtype=torch.float32)
            self.assertNotEqual(memory.keys.data_ptr(), copy.keys.data_ptr())
            for bad in (bid + "::reference", self.records[0]["source_v42_bank_id"]):
                with self.assertRaises(KeyError):
                    loader.get_memory(bad, device="cpu", dtype=torch.float32)
            self.assertFalse(hasattr(loader, "get_reference_offline"))
            self.assertFalse(loader.manifest["qualified_for_online_use"])

    def test_tensor_file_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_compiled(Path(tmp), self.tensors, self.compiled)
            tensor_path = Path(tmp) / "v4_3_primary_side_kv.safetensors"
            with tensor_path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                V43SideKVBankLoader(path, source_manifest=self.manifest, source_records=self.records)

    def test_bfloat16_compile_and_round_trip(self):
        model = tiny_model(torch.bfloat16)
        compiler = V43SideKVCompiler(model=model, tokenizer=self.tokenizer, reasoner=self.reasoner)
        tensors, manifest = compiler.compile(self.records[:1], tier_subset(self.outputs, count=1)[1])
        self.assertEqual(tensors["keys"].dtype, torch.bfloat16)
        self.assertTrue(manifest["production_dtype"])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_compiled(Path(tmp), tensors, manifest)
            loader = V43SideKVBankLoader(path, source_manifest=tier_subset(self.outputs, count=1)[1], source_records=self.records[:1])
            memory = loader.get_memory(self.records[0]["bank_id"], device="cpu", dtype=torch.bfloat16)
            self.assertEqual(memory.keys.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(memory.keys, tensors["keys"][0]))


if __name__ == "__main__":
    unittest.main()
