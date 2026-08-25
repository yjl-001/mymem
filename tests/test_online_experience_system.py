from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for online generation tests")
class OnlineExperienceMemorySystemTests(unittest.TestCase):
    def test_first_joint_trigger_retrieves_once_and_keeps_memory_active(self) -> None:
        from memgen.experience.e1 import MemoryChoice
        from memgen.experience.system import (
            ExperienceMemorySystemProfile,
            SemanticRetrievalDecision,
        )
        from memgen.model.e1_runtime import GateProbe
        from memgen.model.experience_system import OnlineExperienceMemorySystem

        profile = ExperienceMemorySystemProfile()
        choice = MemoryChoice(
            memory_id="memory-a",
            payload_hash="payload-a",
            token_count=2,
            kv_valid_slot_count=2,
            retrieval_score=1.0,
            retrieval_rank=1,
        )
        decision = SemanticRetrievalDecision(
            status="selected",
            query={"query_hash": "query-a"},
            hits=({"memory_id": "memory-a"},),
            matched_memory=choice,
        )

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            @staticmethod
            def decode(token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join("." if int(token) == 1 else "work" for token in token_ids)

        class FakeRetriever:
            def __init__(self):
                self.profile = profile
                self.calls = []

            def retrieve(self, *, question, partial_cot_token_ids):
                self.calls.append((question, tuple(partial_cot_token_ids)))
                return decision

        class FakeLoader:
            @staticmethod
            def get(memory_id, *, device, dtype):
                del device, dtype
                return SimpleNamespace(
                    memory_id=memory_id,
                    payload_hash="payload-a",
                    valid_slot_count=2,
                    layer_number=24,
                )

        class FakeController:
            layer_number = 24
            memory_score_normalization = "log_valid_slots"
            memory_score_bias = profile.memory_score_bias

            def __init__(self):
                self.active = None
                self._traces = []

            @property
            def traces(self):
                return tuple(self._traces)

            def clear_traces(self):
                self._traces.clear()

            @contextmanager
            def use_memory(self, memory):
                self.active = memory
                try:
                    yield self
                finally:
                    self.active = None

        controller = FakeController()

        class FakeModel:
            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.zeros(1))
                self.active_tokens = iter((3, 4, 0))

            def parameters(self):
                yield self.parameter

            def __call__(self, *, input_ids, attention_mask, **kwargs):
                del input_ids, kwargs
                if controller.active is None:
                    next_token = 1
                else:
                    next_token = next(self.active_tokens)
                    controller._traces.append(SimpleNamespace(
                        memory_id=controller.active.memory_id,
                        native_key_length=int(attention_mask.shape[-1]),
                        memory_slot_count=controller.active.valid_slot_count,
                        memory_score_normalization=(
                            controller.memory_score_normalization
                        ),
                        memory_score_bias=controller.memory_score_bias,
                        memory_attention_mass=0.08,
                    ))
                logits = torch.full((1, 1, 8), -10.0)
                logits[0, 0, next_token] = 10.0
                return SimpleNamespace(
                    logits=logits,
                    past_key_values={"length": int(attention_mask.shape[-1])},
                )

        model = FakeModel()

        class FakeGate:
            config = SimpleNamespace(
                layer_number=24,
                entropy_threshold=1.0,
                risk_threshold=0.0,
            )

            @staticmethod
            def triggered(probe):
                return probe.entropy >= 1.0 and probe.risk_score > 0.0

            @staticmethod
            def probe(**kwargs):
                attention_mask = kwargs["attention_mask"]
                logits = torch.full((1, 1, 8), -10.0)
                logits[0, 0, 2] = 10.0
                return GateProbe(
                    entropy=2.0,
                    risk_score=0.5,
                    output=SimpleNamespace(
                        logits=logits,
                        past_key_values={
                            "length": int(attention_mask.shape[-1])
                        },
                    ),
                )

        retriever = FakeRetriever()
        system = OnlineExperienceMemorySystem(
            model=model,
            tokenizer=FakeTokenizer(),
            device="cpu",
            max_new_tokens=4,
            gate=FakeGate(),
            retriever=retriever,
            loader=FakeLoader(),
            controller=controller,
            profile=profile,
        )
        result = system.generate(
            question="rate question", prompt_token_ids=(10, 11)
        )
        self.assertEqual(result.completion_token_ids, (1, 3, 4, 0))
        self.assertEqual(retriever.calls, [("rate question", (1,))])
        self.assertTrue(result.triggered)
        self.assertTrue(result.side_kv_applied)
        self.assertEqual(
            [trace.native_key_length for trace in result.attention_traces],
            [3, 4, 5],
        )
        self.assertIsNone(controller.active)


if __name__ == "__main__":
    unittest.main()
