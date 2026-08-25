from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for generation parity tests")
class E1RuntimeParityTests(unittest.TestCase):
    def test_reports_the_first_token_divergence(self) -> None:
        from memgen.model.e1_runtime import compare_token_sequences

        parity = compare_token_sequences((1, 2, 3), (1, 4, 3))
        self.assertFalse(parity.exact_match)
        self.assertEqual(parity.shared_prefix_length, 1)
        self.assertEqual(parity.first_mismatch_index, 1)
        self.assertEqual(parity.reference_token_id, 2)
        self.assertEqual(parity.candidate_token_id, 4)

    def test_native_and_explicit_cache_paths_share_greedy_semantics(self) -> None:
        from memgen.model.e1_runtime import (
            GreedyE1Runtime,
            compare_token_sequences,
        )

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

        class FakeModel:
            generation_config = SimpleNamespace(repetition_penalty=1.1)

            @staticmethod
            def get_input_embeddings():
                return lambda input_ids: input_ids.unsqueeze(-1).float()

            @staticmethod
            def generate(*, inputs_embeds, generation_config, **kwargs):
                del kwargs
                self.assertFalse(generation_config.use_cache)
                self.assertEqual(generation_config.repetition_penalty, 1.0)
                return torch.tensor([[3, 0]], device=inputs_embeds.device)

            @staticmethod
            def __call__(*, input_ids, attention_mask, past_key_values=None, **kwargs):
                del input_ids, attention_mask, kwargs
                next_token = 3 if past_key_values is None else 0
                logits = torch.full((1, 1, 8), -10.0)
                logits[0, 0, next_token] = 10.0
                return SimpleNamespace(
                    logits=logits,
                    past_key_values={"cached": True},
                )

        runtime = GreedyE1Runtime(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device="cpu",
            max_new_tokens=4,
        )
        native = runtime.generate_vanilla((10, 11))
        cached = runtime.generate_cache_greedy((10, 11))
        self.assertEqual(native, (3, 0))
        self.assertEqual(cached, native)
        self.assertTrue(compare_token_sequences(native, cached).exact_match)
        self.assertEqual(
            runtime.native_generation_config_dict["model_input"],
            "inputs_embeds",
        )
        self.assertFalse(runtime.native_generation_config_dict["use_cache"])
        self.assertTrue(runtime.cache_generation_config_dict["use_cache"])

    def test_trigger_prefix_replay_preserves_observation_chunking(self) -> None:
        from memgen.model.e1_runtime import GreedyE1Runtime

        calls = []

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

        class FakeModel:
            @staticmethod
            def __call__(
                *, input_ids, attention_mask, past_key_values=None, **kwargs
            ):
                del kwargs
                calls.append((
                    tuple(input_ids.shape),
                    tuple(attention_mask.shape),
                    past_key_values,
                ))
                return SimpleNamespace(
                    past_key_values={"call_count": len(calls)}
                )

        runtime = GreedyE1Runtime(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device="cpu",
            max_new_tokens=8,
        )
        cache = runtime._replay_prefix_cache(
            prefix_token_ids=(10, 11, 20, 21, 22),
            prompt_token_count=2,
        )
        self.assertEqual(
            [(input_shape, mask_shape) for input_shape, mask_shape, _ in calls],
            [((1, 2), (1, 2)), ((1, 1), (1, 3)), ((1, 1), (1, 4))],
        )
        self.assertEqual(cache, {"call_count": 3})


if __name__ == "__main__":
    unittest.main()
