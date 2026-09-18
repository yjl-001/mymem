"""Real tiny-Qwen integration: native cache, generation, compile/evaluate and resume."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

from tests.test_local_bank_construction import (
    FixtureTeacher, FixtureReasoner, fixture_split,
)
from tests.test_v4_3_prefix_equivalence import PrefixTokenizer
from tests.test_v4_3_side_kv import tiny_model, torch
from memgen.experience.bank_construction.artifacts import Store, read_json
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.rollouts import run_rollouts
from memgen.experience.bank_construction.review import run_review
from memgen.experience.bank_construction.experiences import run_experiences
from memgen.experience.bank_construction.grouping import run_grouping
from memgen.experience.bank_construction.cards import run_cards
from memgen.experience.bank_construction.compilation import run_compile, validate_compiled
from memgen.experience.bank_construction.evaluation import run_evaluate


@unittest.skipIf(torch is None, "Torch/Transformers required")
class RuntimeTests(unittest.TestCase):
    def test_real_datasets_split_is_independent_of_global_rng(self):
        from datasets import Dataset, DatasetDict
        import numpy as np
        from memgen.experience.bank_construction.dataset import prepare_split
        from data.gsm8k.splits import split_gsm8k
        raw = DatasetDict({"train": Dataset.from_list([{"question": f"q{i}", "answer": "reason\n#### 2"} for i in range(20)]),
                           "test": Dataset.from_list([{"question": "held out", "answer": "reason\n#### 2"}])})
        a = prepare_split(raw, ConstructionConfig(), "revision")
        np.random.seed(987)
        b = prepare_split(raw, ConstructionConfig(), "revision")
        self.assertEqual(a, b)
        self.assertEqual(a["counts"], {"train": 18, "valid": 2, "test": 1})
        self.assertIs(split_gsm8k(raw)["test"], raw["test"])
        self.assertEqual(set(r["source_index"] for name in ("train", "valid") for r in a["splits"][name]), set(range(20)))

    def test_real_local_qwen3_directory_load_and_weight_identity(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from memgen.experience.bank_construction.config import ModelConfig
        from memgen.experience.bank_construction.sources import resolve_model
        from memgen.model.local_bank import LocalModel
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "question": 3}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
            fast.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
            fast.save_pretrained(path)
            cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=4096,
                             eos_token_id=2, pad_token_id=0)
            Qwen3ForCausalLM(cfg).save_pretrained(path)
            settings = ModelConfig(str(path), device="cpu", dtype="float32")
            identity = resolve_model(settings)
            model = LocalModel(identity, settings)
            try:
                self.assertFalse(any(p.requires_grad for p in model.model.parameters()))
                prompt = model.tokenizer.apply_chat_template([{"role": "user", "content": "question"}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                result = model.generate(prompt, seed=1, max_new_tokens=4, sampling=True, temperature=.7, top_p=.8, top_k=20)
                self.assertGreater(result["token_count"], 0)
                self.assertIn(result["stop_reason"], {"eos", "length"})
            finally:
                model.close()
            (path / "config.json").write_text((path / "config.json").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "differ"):
                LocalModel(identity, settings)

    def local_model(self):
        from memgen.model.local_bank import LocalModel, native_runtime
        model = LocalModel.__new__(LocalModel)
        model.model = tiny_model()
        model.tokenizer = PrefixTokenizer(box_after=4)
        model.device = torch.device("cpu")
        model.context_limit = 4096
        model.runtime = native_runtime(model.model, model.tokenizer, "cpu")
        return model

    def test_sampling_reproducible_and_completion_not_truncation(self):
        model = self.local_model()
        self.addCleanup(model.close)
        # Zero logits avoid model-dependent early EOS in greedy tests.
        with torch.no_grad():
            for parameter in model.model.parameters():
                parameter.zero_()
        a = model.generate("question", seed=17, max_new_tokens=4, sampling=False, stop_on_box=True)
        self.assertEqual(a["stop_reason"], "completed_boxed_answer")
        self.assertFalse(a["truncated"])
        self.assertEqual(a["token_count"], 4)  # At the budget AND naturally completed.
        b = model.generate("question", seed=17, max_new_tokens=3, sampling=False, stop_on_box=True)
        self.assertTrue(b["truncated"])
        self.assertEqual(b["stop_reason"], "length")
        kwargs = dict(seed=29, max_new_tokens=4, sampling=True, temperature=.8, top_p=.95, top_k=0, stop_on_box=True)
        self.assertEqual(model.generate("question", **kwargs), model.generate("question", **kwargs))
        model.context_limit = 2
        with self.assertRaisesRegex(ValueError, "no truncation"):
            model.generate("question", seed=1, max_new_tokens=4, sampling=False)

    def test_native_cache_matches_existing_consumer_and_is_not_mutated(self):
        from memgen.model.v4_3_prefix_equivalence import prefix_bank, run_equivalence
        model = self.local_model()
        self.addCleanup(model.close)
        with tempfile.TemporaryDirectory() as tmp:
            record = {"bank_id": "v43-bank-native-fixture", "record_sha256": "source", "descriptor": "Track remainder."}
            ids, tensors = prefix_bank(Path(tmp), record, model.runtime, "profile")
            before = {k: v.clone() for k, v in tensors.items()}
            _, text, kv, comparison = run_equivalence(model.runtime, "How many?", record["descriptor"], ids, tensors,
                                                      atol=1e-5, rtol=1e-4)
            self.assertTrue(comparison["numerical_pass"])
            self.assertEqual(text["continuation_token_ids"], kv["continuation_token_ids"])
            self.assertTrue(all(torch.equal(before[k], v) for k, v in tensors.items()))

    def test_train_cards_compile_valid_utilities_and_cached_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            config = replace(ConstructionConfig(), val_ratio=.34, train_limit=2, group_batch_size=1)
            profile = {"fixture": "runtime", "teacher": {"source": "fixture"}}
            store = Store(Path(tmp), profile)
            store.put("split", fixture_split(config), {})
            run_rollouts(store, config, FixtureReasoner)
            teacher = FixtureTeacher()
            run_review(store, teacher)
            run_experiences(store, teacher)
            run_grouping(store, teacher, config)
            run_cards(store, teacher, config)
            compiled = run_compile(store, config, self.local_model)
            self.assertEqual(compiled["bank_count"], 1)
            summary = run_evaluate(store, config, self.local_model)
            self.assertEqual(summary["sample_count"], 2)
            self.assertFalse(summary["official_test_used"])
            self.assertEqual(summary["semantic_top1"]["count"], 2)
            self.assertEqual(len(list((store.root / "valid_results").rglob("*.json"))), 4)
            def unexpected():
                raise AssertionError("Resume should not load weights")
            self.assertEqual(compiled, run_compile(store, config, unexpected))
            self.assertEqual(summary, run_evaluate(store, config, unexpected))
            from memgen.experience.bank_construction.pipeline import run
            from memgen.experience.bank_construction.audit import audit_complete
            run(store, config, profile, stage="evaluate", reasoner_factory=unexpected, teacher_factory=unexpected)
            self.assertTrue((store.root / "brief_summary.json").exists())
            self.assertEqual(audit_complete(store, config)["bank_count"], 1)
            path = store.root / "prefix_kv" / (compiled["entries"][0]["bank_id"] + ".safetensors")
            with path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "tensor drift"):
                validate_compiled(store, compiled)


if __name__ == "__main__":
    unittest.main()
