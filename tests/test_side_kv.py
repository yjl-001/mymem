from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # Lightweight local environments run the pure E0 tests.
    torch = None
    nn = None


@unittest.skipIf(torch is None, "Torch is required for side-KV tensor tests")
class SideKVMathTests(unittest.TestCase):
    def test_canonical_score_is_invariant_to_a_shared_rope_position(self) -> None:
        from memgen.model.side_kv import (
            apply_rotary_pos_emb,
            canonical_memory_scores,
            shared_rope_score_relative_error,
        )

        query = torch.tensor([[[[0.4, -0.2, 0.7, 0.1]]]], dtype=torch.float32)
        key = torch.tensor([[[[0.3, 0.6, -0.5, 0.2]]]], dtype=torch.float32)
        baseline = canonical_memory_scores(query, key, scaling=0.5)
        for angle in (0.0, 0.3, 1.7):
            cos = torch.full((1, 1, 4), torch.cos(torch.tensor(angle)))
            sin = torch.full((1, 1, 4), torch.sin(torch.tensor(angle)))
            rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)
            rotated_score = torch.matmul(
                rotated_query, rotated_key.transpose(2, 3)
            ) * 0.5
            self.assertTrue(torch.allclose(baseline, rotated_score, atol=1e-6))
            relative_error = shared_rope_score_relative_error(
                query_pre_rope=query,
                canonical_memory_keys=key,
                cos=cos,
                sin=sin,
                scaling=0.5,
            )
            self.assertLessEqual(relative_error, 1e-6)

    def test_repeat_kv_maps_each_group_to_contiguous_query_heads(self) -> None:
        from memgen.model.side_kv import repeat_kv

        value = torch.tensor([[[[1.0]], [[2.0]]]])
        repeated = repeat_kv(value, 2)
        self.assertEqual(tuple(repeated.shape), (1, 4, 1, 1))
        self.assertEqual(repeated.flatten().tolist(), [1.0, 1.0, 2.0, 2.0])


if torch is not None:
    class FakeConfig:
        num_attention_heads = 4
        num_key_value_heads = 2
        hidden_size = 8
        _attn_implementation = "eager"


    class FakeAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = FakeConfig()
            self.layer_idx = 0
            self.head_dim = 2
            self.num_key_value_groups = 2
            self.scaling = self.head_dim**-0.5
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 4, bias=False)
            self.v_proj = nn.Linear(8, 4, bias=False)
            self.o_proj = nn.Linear(8, 8, bias=False)

        def forward(
            self,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            del (
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                kwargs,
            )
            return "native-forward"


    class FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = FakeAttention()


    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([FakeBlock()])


    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeBackbone()


    class FakeCache:
        def __init__(self, keys, values) -> None:
            self.keys = keys
            self.values = values

        def update(self, keys, values, layer_idx, cache_kwargs):
            self.layer_idx = layer_idx
            self.cache_kwargs = cache_kwargs
            self.keys = torch.cat([self.keys, keys], dim=-2)
            self.values = torch.cat([self.values, values], dim=-2)
            return self.keys, self.values


@unittest.skipIf(torch is None, "Torch is required for side-KV tensor tests")
class SideKVControllerTests(unittest.TestCase):
    def test_memory_uses_a_side_path_without_extending_the_native_cache(self) -> None:
        from memgen.model.side_kv import SideKVAttentionController, SideKVMemory

        torch.manual_seed(7)
        model = FakeModel().eval()
        attention = model.model.layers[0].self_attn
        controller = SideKVAttentionController(
            model=model,
            layer_number=1,
            audit_canonical_rope=True,
        )
        native_keys = torch.randn(1, 2, 3, 2)
        native_values = torch.randn(1, 2, 3, 2)
        prefix_keys = native_keys.clone()
        cache = FakeCache(native_keys, native_values)
        memory = SideKVMemory(
            memory_id="memory-fixture",
            payload_hash="payload",
            keys=torch.randn(2, 2, 2),
            values=torch.randn(2, 2, 2),
            slot_mask=torch.tensor([True, True]),
            layer_number=1,
        )
        hidden = torch.randn(1, 1, 8)
        cos = torch.ones(1, 1, 2)
        sin = torch.zeros(1, 1, 2)
        mask = torch.zeros(1, 1, 1, 4)
        try:
            with controller.use_memory(memory):
                output, weights = attention(
                    hidden_states=hidden,
                    position_embeddings=(cos, sin),
                    attention_mask=mask,
                    past_key_value=cache,
                    cache_position=torch.tensor([3]),
                )
            self.assertEqual(tuple(output.shape), (1, 1, 8))
            self.assertEqual(tuple(weights.shape), (1, 4, 1, 6))
            self.assertEqual(cache.keys.shape[-2], 4)
            self.assertTrue(torch.equal(cache.keys[..., :3, :], prefix_keys))
            self.assertEqual(len(controller.traces), 1)
            trace = controller.traces[0]
            self.assertEqual(trace.native_key_length, 4)
            self.assertEqual(trace.memory_slot_count, 2)
            self.assertGreater(trace.memory_attention_mass, 0.0)
            self.assertLess(trace.memory_attention_mass, 1.0)
            self.assertIsNotNone(trace.canonical_rope_score_relative_error)
            self.assertLessEqual(trace.canonical_rope_score_relative_error, 1e-6)
            self.assertEqual(attention(hidden_states=hidden), "native-forward")
        finally:
            controller.close()

    def test_context_manager_clears_memory_after_an_exception(self) -> None:
        from memgen.model.side_kv import SideKVAttentionController, SideKVMemory

        model = FakeModel().eval()
        controller = SideKVAttentionController(model=model, layer_number=1)
        memory = SideKVMemory(
            memory_id="memory-fixture",
            payload_hash="payload",
            keys=torch.ones(2, 1, 2),
            values=torch.ones(2, 1, 2),
            slot_mask=torch.tensor([True]),
            layer_number=1,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with controller.use_memory(memory):
                    raise RuntimeError("fixture")
            self.assertEqual(
                model.model.layers[0].self_attn(hidden_states=torch.ones(1, 1, 8)),
                "native-forward",
            )
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
