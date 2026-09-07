from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch

    from memgen.model.v4_oracle import (
        V4OracleExactPrefixRuntime,
        _complete_boxed_answer_seen,
    )
    from memgen.model.side_kv import SideKVAttentionTrace
except ModuleNotFoundError:  # pragma: no cover - minimal local environments
    torch = None
    V4OracleExactPrefixRuntime = None
    SideKVAttentionTrace = None
    _complete_boxed_answer_seen = None


def cache_with_length(length: int):
    return (
        (
            torch.zeros((1, 1, length, 1), dtype=torch.float32),
            torch.zeros((1, 1, length, 1), dtype=torch.float32),
        ),
    )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def __init__(self, *, box_after_completion_tokens: int | None = None) -> None:
        self.box_after_completion_tokens = box_after_completion_tokens

    def decode(self, token_ids, *, skip_special_tokens=False):
        del skip_special_tokens
        if (
            self.box_after_completion_tokens is not None
            and len(token_ids) >= self.box_after_completion_tokens
        ):
            return "reasoning \\boxed{42}"
        return "reasoning"


class FakeModel:
    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
        return_dict,
    ):
        del input_ids, past_key_values, use_cache, return_dict
        logits = torch.tensor([[[0.0, 2.0, 1.0, -1.0]]], dtype=torch.float32)
        return SimpleNamespace(
            logits=logits,
            past_key_values=cache_with_length(int(attention_mask.shape[1])),
        )


class FakeController:
    layer_number = 24

    def __init__(self) -> None:
        self._traces = []
        self.active_memory_id = None

    @property
    def traces(self):
        return tuple(self._traces)

    def activate(self, memory) -> None:
        self.active_memory_id = memory.memory_id

    def deactivate(self) -> None:
        self.active_memory_id = None

    def clear_traces(self) -> None:
        self._traces = []


class FakeGate:
    config = SimpleNamespace(
        layer_number=24,
        risk_role="online_joint_control",
        rearm_low_entropy_token_count=2,
        low_entropy_threshold=0.0,
    )

    def __init__(self, controller: FakeController) -> None:
        self.controller = controller

    def probe(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        output = kwargs["model"](
            input_ids=kwargs["boundary_token"],
            attention_mask=attention_mask,
            past_key_values=kwargs["past_key_values"],
            use_cache=True,
            return_dict=True,
        )
        native_length = int(attention_mask.shape[1])
        if self.controller.active_memory_id is not None:
            self.controller._traces.append(
                SideKVAttentionTrace(
                    memory_id=str(self.controller.active_memory_id),
                    layer_number=24,
                    query_length=1,
                    native_key_length=native_length,
                    memory_slot_count=1,
                    memory_attention_mass=0.1,
                    native_attention_mass=0.9,
                    canonical_rope_score_relative_error=0.0,
                    memory_mass_by_query_head=(0.1,),
                    memory_mass_by_kv_group=(0.1,),
                )
            )
        return SimpleNamespace(entropy=1.0, output=output)


@unittest.skipIf(torch is None, "torch is unavailable")
class V4OracleRuntimeTests(unittest.TestCase):
    def runtime(self, tokenizer: FakeTokenizer) -> V4OracleExactPrefixRuntime:
        controller = FakeController()
        return V4OracleExactPrefixRuntime(
            model=FakeModel(),
            tokenizer=tokenizer,
            device="cpu",
            gate=FakeGate(controller),
            controller=controller,
            maximum_completion_tokens=1024,
        )

    def test_complete_box_detector_requires_balanced_closing_brace(self) -> None:
        self.assertFalse(_complete_boxed_answer_seen("answer \\boxed"))
        self.assertFalse(_complete_boxed_answer_seen("answer \\boxed{42"))
        self.assertTrue(_complete_boxed_answer_seen("answer \\boxed{42}"))
        self.assertTrue(
            _complete_boxed_answer_seen("answer \\boxed{\\frac{1}{2}}")
        )

    def test_completion_budget_is_separate_from_local_32_token_window(self) -> None:
        runtime = self.runtime(FakeTokenizer())
        prefix = (7, 8, 9, 10)
        result, _scores = runtime._run_branch(
            role="baseline",
            prefix_token_ids=prefix,
            prompt_token_count=2,
            initial_cache=cache_with_length(len(prefix) - 1),
            baseline_first_scores=None,
            memory=None,
        )
        self.assertEqual(result.prefix_completion_token_count, 2)
        self.assertEqual(result.generation_budget_from_prefix, 1022)
        self.assertEqual(len(result.continuation_token_ids), 1022)
        self.assertEqual(len(result.local_continuation_token_ids), 32)
        self.assertEqual(result.stop_reason, "maximum_completion_tokens")
        self.assertFalse(result.complete_boxed_answer_seen)

    def test_complete_box_stops_after_local_or_native_continuation(self) -> None:
        runtime = self.runtime(
            FakeTokenizer(box_after_completion_tokens=40)
        )
        prefix = (7, 8, 9, 10)
        result, _scores = runtime._run_branch(
            role="baseline",
            prefix_token_ids=prefix,
            prompt_token_count=2,
            initial_cache=cache_with_length(len(prefix) - 1),
            baseline_first_scores=None,
            memory=None,
        )
        self.assertEqual(len(result.local_continuation_token_ids), 32)
        self.assertEqual(len(result.continuation_token_ids), 38)
        self.assertEqual(result.stop_reason, "completed_boxed_answer")
        self.assertTrue(result.answer_marker_seen)
        self.assertTrue(result.complete_boxed_answer_seen)
        self.assertFalse(result.complete_boxed_answer_seen_within_local_window)

    def test_memory_unloads_at_32_but_generation_continues_to_outcome_budget(self) -> None:
        runtime = self.runtime(FakeTokenizer())
        prefix = (7, 8, 9, 10)
        result, _scores = runtime._run_branch(
            role="target",
            prefix_token_ids=prefix,
            prompt_token_count=2,
            initial_cache=cache_with_length(len(prefix) - 1),
            baseline_first_scores=torch.tensor([[0.0, 2.0, 1.0, -1.0]]),
            memory=SimpleNamespace(memory_id="bank-a"),
        )
        self.assertEqual(len(result.attention_traces), 32)
        self.assertEqual(len(result.local_continuation_token_ids), 32)
        self.assertEqual(len(result.continuation_token_ids), 1022)
        reasons = [
            transition["reason"] for transition in result.lifecycle["transitions"]
        ]
        self.assertIn("maximum_active_window", reasons)
        self.assertEqual(result.lifecycle["state"], "CLOSED")
        self.assertEqual(result.stop_reason, "maximum_completion_tokens")


if __name__ == "__main__":
    unittest.main()
