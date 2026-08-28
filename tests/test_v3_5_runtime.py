from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in CPU-light CI
    torch = None


class V35ResultSerializationWithoutTorchTests(unittest.TestCase):
    def test_to_dict_accesses_native_gate_observation_property(self):
        """Exercise the exact result class even when heavyweight torch is absent."""

        runtime_path = (
            Path(__file__).resolve().parents[1] / "memgen" / "model" / "v3_runtime.py"
        )
        parsed = ast.parse(runtime_path.read_text(encoding="utf-8"))
        result_class = next(
            node
            for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "V3GenerationResult"
        )
        isolated_module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                result_class,
            ],
            type_ignores=[],
        )
        namespace = {
            "dataclass": dataclass,
            "field": field,
            "V3_GENERATION_RESULT_SCHEMA": "experience-memory-v3-generation-result-v1",
            "V35_GENERATION_RESULT_SCHEMA": (
                "experience-memory-v3.5-generation-result-v1"
            ),
            "canonical_json_sha256": lambda value: f"sha:{len(value)}",
        }
        exec(
            compile(
                ast.fix_missing_locations(isolated_module),
                str(runtime_path),
                "exec",
            ),
            namespace,
        )
        result_type = namespace["V3GenerationResult"]
        result = result_type(
            completion_token_ids=(),
            boundary_traces=(),
            retrieval_attempts=(),
            memory_transitions=(),
            attention_traces=(),
            final_gate_state="ARMED",
            final_memory_id=None,
            answer_marker_seen=False,
            schema_version=namespace["V35_GENERATION_RESULT_SCHEMA"],
            static_selector_trace={"static_selector_unavailable": False},
        )

        payload = result.to_dict()

        self.assertEqual(payload["summary"]["native_gate_observation_count"], 0)
        self.assertEqual(
            payload["summary"]["memory_conditioned_gate_observation_count"], 0
        )
        self.assertIsInstance(
            result_type.native_gate_observation_count, property
        )
        self.assertNotIsInstance(
            result_type.native_gate_observation_count.fget, property
        )


@dataclass(frozen=True)
class FakeStaticSelection:
    memory_ids: tuple[str, ...]
    available: bool
    unavailable_reason: str | None = None

    @property
    def shortlist_nonempty(self) -> bool:
        return bool(self.memory_ids)

    def to_dict(self) -> dict[str, Any]:
        shortlist = [
            {
                "memory_id": memory_id,
                "static_score": 0.9 - index * 0.1,
                "original_global_rank": index + 1,
            }
            for index, memory_id in enumerate(self.memory_ids)
        ]
        return {
            "schema_version": "fake-v3.5-static-shortlist-v1",
            "shortlist_memory_ids": list(self.memory_ids),
            "pre_floor_top_k": shortlist,
            "post_floor_shortlist": shortlist,
            "shortlist_nonempty": self.shortlist_nonempty,
            "static_selector_unavailable": not self.available,
            "unavailable_reason": self.unavailable_reason,
            "query": {
                "static_question_text_sha256": "question-sha",
                "static_question_token_count": 3,
                "side_kv_disabled": True,
            },
        }


class MutableCache:
    def __init__(self, length: int = 0):
        self.length = length
        self.crop_calls: list[int] = []

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        if length < 0 or length > self.length:
            raise ValueError("invalid fake cache crop")
        self.crop_calls.append(length)
        self.length = length


class FakeTokenizer:
    pad_token_id = 7
    eos_token_id = 0

    @staticmethod
    def decode(token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(f"t{int(token_id)}" for token_id in token_ids)


class MarkerTokenizer(FakeTokenizer):
    marker_token_id = 6

    @classmethod
    def decode(
        cls, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        del skip_special_tokens
        return "".join(
            "\\boxed{" if int(token_id) in {cls.marker_token_id, cls.eos_token_id}
            else f"t{int(token_id)}"
            for token_id in token_ids
        )


class FakeController:
    layer_number = 24
    memory_score_normalization = "log_valid_slots"

    def __init__(self, *, memory_score_bias: float):
        self.memory_score_bias = memory_score_bias
        self.active_memory = None
        self._traces: list[Any] = []

    @property
    def traces(self) -> tuple[Any, ...]:
        return tuple(self._traces)

    def clear_traces(self) -> None:
        self._traces.clear()

    def truncate_traces(self, length: int) -> None:
        del self._traces[length:]

    def activate(self, memory: Any) -> None:
        self.active_memory = memory

    def deactivate(self) -> None:
        self.active_memory = None

    @contextmanager
    def suspend_memory(self):
        previous = self.active_memory
        self.active_memory = None
        try:
            yield previous
        finally:
            self.active_memory = previous

    def append_trace(self, native_length: int) -> None:
        if self.active_memory is None:
            return
        from memgen.model.side_kv import SideKVAttentionTrace

        self._traces.append(SideKVAttentionTrace(
            memory_id=self.active_memory.memory_id,
            layer_number=24,
            query_length=1,
            native_key_length=native_length,
            memory_slot_count=self.active_memory.valid_slot_count,
            memory_attention_mass=0.1,
            native_attention_mass=0.9,
            canonical_rope_score_relative_error=None,
            memory_mass_by_query_head=(0.1,),
            memory_mass_by_kv_group=(0.1,),
            memory_score_normalization=self.memory_score_normalization,
            memory_score_bias=self.memory_score_bias,
        ))


class FakeModel:
    def __init__(
        self,
        *,
        controller: FakeController,
        token_for_path: Callable[[int, str | None], int],
    ):
        self.controller = controller
        self.token_for_path = token_for_path
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[int, str | None]] = []
        self.caches: list[MutableCache] = []

    def parameters(self):
        yield self.parameter

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        del kwargs
        input_count = int(input_ids.shape[-1])
        native_length = int(attention_mask.shape[-1])
        if past_key_values is None:
            cache = MutableCache()
            self.caches.append(cache)
        else:
            cache = past_key_values
        if cache.length + input_count != native_length:
            raise AssertionError(
                f"fake cache drift: {cache.length}+{input_count}!={native_length}"
            )
        cache.length += input_count
        memory_id = (
            self.controller.active_memory.memory_id
            if self.controller.active_memory is not None
            else None
        )
        self.calls.append((native_length, memory_id))
        self.controller.append_trace(native_length)
        next_token = int(self.token_for_path(native_length, memory_id))
        logits = torch.full((1, 1, 8), -10.0)
        logits[0, 0, next_token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


class FakeGate:
    def __init__(self, entropies: Sequence[float]):
        self.config = SimpleNamespace(
            layer_number=24,
            high_entropy_threshold=1.0,
            low_entropy_threshold=0.5,
            risk_threshold=0.0,
            risk_role="online_joint_control",
            rearm_low_entropy_token_count=2,
        )
        self._entropies = iter(entropies)
        self.call_count = 0
        self.clone_flags: list[bool] = []

    def probe(self, **kwargs):
        from memgen.model.e1_runtime import GateProbe

        self.call_count += 1
        clone = bool(kwargs["clone_past_key_values"])
        self.clone_flags.append(clone)
        output = kwargs["model"](
            input_ids=kwargs["boundary_token"],
            attention_mask=kwargs["attention_mask"],
            past_key_values=kwargs["past_key_values"],
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        return GateProbe(
            entropy=float(next(self._entropies)),
            risk_score=0.5,
            output=output,
        )

    def trigger_qualified(self, probe) -> bool:
        return bool(
            probe.entropy >= self.config.high_entropy_threshold
            and probe.risk_score > self.config.risk_threshold
        )


class FakeQueryEncoder:
    layer_number = 24

    def __init__(self, *, query_pooling: str, controller: FakeController):
        self.query_pooling = query_pooling
        self.controller = controller
        self.calls: list[tuple[int, ...]] = []

    def encode(self, token_ids: Sequence[int]):
        if self.controller.active_memory is not None:
            raise AssertionError("dynamic query was memory-conditioned")
        self.calls.append(tuple(int(token_id) for token_id in token_ids))
        return torch.tensor([1.0, 0.0])


class FakeLoader:
    @staticmethod
    def get(memory_id: str, *, device: str, dtype):
        del device, dtype
        suffix = memory_id.rsplit("-", 1)[-1]
        return SimpleNamespace(
            memory_id=memory_id,
            payload_hash=f"payload-{suffix}",
            valid_slot_count=2,
            layer_number=24,
        )


class FakeV35Retriever:
    def __init__(
        self,
        *,
        profile: Any,
        static_selection: FakeStaticSelection,
        outcomes: Sequence[str | None],
    ):
        self.profile = profile
        self.shortlist_k = profile.applicability_shortlist_k
        self.applicability_score_floor = profile.applicability_score_floor
        self.dynamic_min_top1_top2_margin = (
            profile.retrieval_min_top1_top2_margin
        )
        self.static_selection = static_selection
        self._outcomes = iter(outcomes)
        self.prepare_calls: list[str] = []
        self.retrieve_calls: list[dict[str, Any]] = []

    def prepare_question(self, question: str) -> FakeStaticSelection:
        self.prepare_calls.append(question)
        return self.static_selection

    def retrieve(
        self,
        *,
        query_embedding,
        query_token_ids,
        prompt_token_count: int,
        static_context: FakeStaticSelection,
    ):
        from memgen.experience.e1 import MemoryChoice
        from memgen.experience.v3 import ApplicabilityAwareRetrievalDecision

        if static_context is not self.static_selection:
            raise AssertionError("dynamic selector received a different shortlist")
        outcome = next(self._outcomes)
        query_ids = tuple(int(token_id) for token_id in query_token_ids)
        self.retrieve_calls.append({
            "query_embedding": query_embedding.detach().clone(),
            "query_token_ids": query_ids,
            "prompt_token_count": prompt_token_count,
            "static_context": static_context,
        })
        shortlist = tuple(
            {
                "memory_id": memory_id,
                "static_score": 0.9 - index * 0.1,
                "original_global_rank": index + 1,
            }
            for index, memory_id in enumerate(static_context.memory_ids)
        )
        hits = tuple(
            {
                "memory_id": memory_id,
                "payload_hash": f"payload-{memory_id.rsplit('-', 1)[-1]}",
                "score": 0.9 - index * 0.2,
                "rank": index + 1,
            }
            for index, memory_id in enumerate(static_context.memory_ids[:2])
        )
        query = {
            "query_token_count": len(query_ids),
            "prompt_token_count": prompt_token_count,
            "partial_cot_token_count": len(query_ids) - prompt_token_count,
            "query_embedding_token_id": query_ids[-1],
            "query_token_ids_sha256": f"query-{len(query_ids)}",
            "side_kv_disabled": True,
            "static_shortlist_memory_ids": list(static_context.memory_ids),
        }
        if outcome is None:
            return ApplicabilityAwareRetrievalDecision(
                status="below_dynamic_margin",
                query=query,
                hits=hits,
                matched_memory=None,
                static_shortlist=shortlist,
            )
        suffix = outcome.rsplit("-", 1)[-1]
        choice = MemoryChoice(
            memory_id=outcome,
            payload_hash=f"payload-{suffix}",
            token_count=2,
            kv_valid_slot_count=2,
            retrieval_score=0.9,
            retrieval_rank=1,
        )
        selected_hits = sorted(
            hits,
            key=lambda hit: hit["memory_id"] != outcome,
        )
        return ApplicabilityAwareRetrievalDecision(
            status="selected",
            query=query,
            hits=tuple(selected_hits),
            matched_memory=choice,
            static_shortlist=shortlist,
        )


class FakeLegacyRetriever:
    def __init__(self, *, profile: Any, outcomes: Sequence[str | None]):
        self.profile = profile
        self._outcomes = iter(outcomes)

    def retrieve(
        self,
        *,
        query_embedding,
        query_token_ids,
        prompt_token_count: int,
    ):
        from memgen.experience.e1 import MemoryChoice
        from memgen.experience.v3 import EmbeddingRetrievalDecision

        del query_embedding
        outcome = next(self._outcomes)
        query_ids = tuple(int(token_id) for token_id in query_token_ids)
        query = {
            "query_token_count": len(query_ids),
            "prompt_token_count": prompt_token_count,
            "partial_cot_token_count": len(query_ids) - prompt_token_count,
            "query_embedding_token_id": query_ids[-1],
        }
        hits = (
            {"memory_id": "memory-a", "score": 0.9},
            {"memory_id": "memory-b", "score": 0.89},
        )
        if outcome is None:
            return EmbeddingRetrievalDecision(
                status="below_margin",
                query=query,
                hits=hits,
                matched_memory=None,
            )
        choice = MemoryChoice(
            memory_id=outcome,
            payload_hash="payload-a",
            token_count=2,
            kv_valid_slot_count=2,
            retrieval_score=0.9,
            retrieval_rank=1,
        )
        return EmbeddingRetrievalDecision(
            status="selected",
            query=query,
            hits=hits,
            matched_memory=choice,
        )


@unittest.skipIf(torch is None, "Torch is required for V3.5 runtime tests")
class V35RuntimeTests(unittest.TestCase):
    @staticmethod
    def _profile():
        from memgen.experience.v3 import ExperienceMemoryV3Profile

        return ExperienceMemoryV3Profile.applicability_aware_continuous(
            applicability_shortlist_k=2,
            applicability_score_floor=-1.0,
            retrieval_min_top1_top2_margin=0.1,
        )

    def _system(
        self,
        *,
        profile,
        static_selection: FakeStaticSelection,
        outcomes: Sequence[str | None],
        entropies: Sequence[float],
        token_for_path: Callable[[int, str | None], int],
        max_new_tokens: int,
        tokenizer: Any | None = None,
    ):
        from memgen.model.v3_runtime import OnlineExperienceMemorySystemV3

        controller = FakeController(
            memory_score_bias=profile.memory_score_bias
        )
        model = FakeModel(
            controller=controller,
            token_for_path=token_for_path,
        )
        gate = FakeGate(entropies)
        query_encoder = FakeQueryEncoder(
            query_pooling=profile.query_pooling,
            controller=controller,
        )
        retriever = FakeV35Retriever(
            profile=profile,
            static_selection=static_selection,
            outcomes=outcomes,
        )
        system = OnlineExperienceMemorySystemV3(
            model=model,
            tokenizer=tokenizer or FakeTokenizer(),
            device="cpu",
            max_new_tokens=max_new_tokens,
            gate=gate,
            query_encoder=query_encoder,
            retriever=retriever,
            loader=FakeLoader(),
            controller=controller,
            profile=profile,
        )
        return system, model, gate, query_encoder, retriever, controller

    def test_marker_completed_by_final_budget_token_is_recorded(self):
        profile = self._profile()
        system, _, _, _, _, _ = self._system(
            profile=profile,
            static_selection=FakeStaticSelection(
                memory_ids=("memory-a", "memory-b"),
                available=True,
            ),
            outcomes=(),
            entropies=(),
            token_for_path=lambda length, memory_id: MarkerTokenizer.marker_token_id,
            max_new_tokens=1,
            tokenizer=MarkerTokenizer(),
        )

        result = system.generate(
            prompt_token_ids=(10, 11), question="question"
        )

        self.assertEqual(
            result.completion_token_ids,
            (MarkerTokenizer.marker_token_id,),
        )
        self.assertTrue(result.answer_marker_seen)
        self.assertEqual(result.final_gate_state, "CLOSED")

    def test_marker_completed_by_eos_token_is_recorded(self):
        profile = self._profile()
        system, _, _, _, _, _ = self._system(
            profile=profile,
            static_selection=FakeStaticSelection(
                memory_ids=("memory-a", "memory-b"),
                available=True,
            ),
            outcomes=(),
            entropies=(),
            token_for_path=lambda length, memory_id: MarkerTokenizer.eos_token_id,
            max_new_tokens=4,
            tokenizer=MarkerTokenizer(),
        )

        result = system.generate(
            prompt_token_ids=(10, 11), question="question"
        )

        self.assertEqual(result.completion_token_ids, (MarkerTokenizer.eos_token_id,))
        self.assertTrue(result.answer_marker_seen)
        self.assertEqual(result.final_gate_state, "CLOSED")

    @staticmethod
    def _native_completion(
        *,
        token_for_path: Callable[[int, str | None], int],
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        controller = FakeController(memory_score_bias=0.0)
        model = FakeModel(
            controller=controller,
            token_for_path=token_for_path,
        )
        ids = [10, 11]
        past = None
        for _ in range(max_new_tokens):
            full = torch.tensor([ids], dtype=torch.long)
            output = model(
                input_ids=full if past is None else full[:, -1:],
                attention_mask=torch.ones_like(full),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
            ids.append(next_token)
            past = output.past_key_values
            if next_token == 0:
                break
        return tuple(ids[2:])

    def test_static_unavailable_starts_exhausted_with_native_zero_attempt_parity(self):
        from memgen.experience.v3 import V35_GENERATION_RESULT_SCHEMA

        profile = self._profile()
        token_for_path = lambda length, memory_id: (
            0 if length >= 5 else 1
        )
        system, model, gate, encoder, retriever, controller = self._system(
            profile=profile,
            static_selection=FakeStaticSelection(
                memory_ids=("memory-a",),
                available=False,
                unavailable_reason="insufficient_shortlist",
            ),
            outcomes=(),
            entropies=(),
            token_for_path=token_for_path,
            max_new_tokens=4,
        )
        result = system.generate(
            prompt_token_ids=(10, 11),
            question="  original question  ",
        )
        self.assertEqual(
            result.completion_token_ids,
            self._native_completion(
                token_for_path=token_for_path,
                max_new_tokens=4,
            ),
        )
        self.assertEqual(result.schema_version, V35_GENERATION_RESULT_SCHEMA)
        self.assertEqual(result.retrieval_attempt_count, 0)
        self.assertEqual(result.boundary_traces, ())
        self.assertEqual(gate.call_count, 0)
        self.assertEqual(encoder.calls, [])
        self.assertEqual(retriever.retrieve_calls, [])
        self.assertEqual(retriever.prepare_calls, ["  original question  "])
        self.assertTrue(all(memory_id is None for _, memory_id in model.calls))
        self.assertIsNone(controller.active_memory)
        serialized = result.to_dict()
        self.assertTrue(
            serialized["summary"]["static_selector_unavailable"]
        )
        self.assertTrue(
            serialized["summary"]["max_three_attempts_respected"]
        )

    def test_no_active_abstain_is_terminal_and_never_rearms(self):
        profile = self._profile()
        token_for_path = lambda length, memory_id: (
            0 if length >= 5 else 1
        )
        system, _, gate, encoder, retriever, _ = self._system(
            profile=profile,
            static_selection=FakeStaticSelection(
                memory_ids=("memory-a", "memory-b"),
                available=True,
            ),
            outcomes=(None,),
            entropies=(2.0, 0.1, 0.1),
            token_for_path=token_for_path,
            max_new_tokens=4,
        )
        result = system.generate(
            prompt_token_ids=(10, 11), question="question"
        )
        self.assertEqual(result.completion_token_ids, (1, 1, 1, 0))
        self.assertEqual(result.retrieval_attempt_count, 1)
        self.assertEqual(result.rearm_count, 0)
        attempt = result.retrieval_attempts[0]
        self.assertEqual(attempt.outcome, "abstained")
        self.assertTrue(attempt.terminal_abstain)
        self.assertFalse(attempt.memory_cleared_on_abstain)
        self.assertIsNone(attempt.cleared_memory_id)
        self.assertEqual(attempt.actual_path_after_abstain, "native")
        self.assertIsNone(attempt.actual_path_memory_id_after)
        self.assertIsNone(attempt.deactivation_forward_seconds)
        self.assertEqual(result.memory_transitions, ())
        self.assertEqual([len(ids) for ids in encoder.calls], [3])
        self.assertEqual(
            retriever.retrieve_calls[0]["query_token_ids"], (10, 11, 1)
        )
        self.assertEqual(
            [trace.action for trace in result.boundary_traces],
            ["retrieval_attempt", "observed_exhausted", "observed_exhausted"],
        )
        self.assertTrue(all(flag is False for flag in gate.clone_flags))
        serialized = result.to_dict()
        self.assertEqual(serialized["summary"]["terminal_abstain_count"], 1)
        self.assertTrue(
            serialized["summary"]["no_rearm_after_terminal_abstain"]
        )

    def test_active_abstain_rolls_back_clears_and_uses_native_t_plus_one(self):
        profile = self._profile()

        def token_for_path(length: int, memory_id: str | None) -> int:
            if length >= 5:
                return 0
            if memory_id == "memory-a":
                return 2
            if length == 4:
                return 3
            return 1

        system, model, _, encoder, _, controller = self._system(
            profile=profile,
            static_selection=FakeStaticSelection(
                memory_ids=("memory-a", "memory-b"),
                available=True,
            ),
            outcomes=("memory-a", None),
            entropies=(2.0, 2.0, 0.1),
            token_for_path=token_for_path,
            max_new_tokens=4,
        )
        result = system.generate(
            prompt_token_ids=(10, 11), question="question"
        )
        # At the second attempt old memory predicts token 2, but the terminal
        # clear re-forward makes token 3 the actual t+1 token.
        self.assertEqual(result.completion_token_ids, (1, 2, 3, 0))
        self.assertEqual([len(ids) for ids in encoder.calls], [3, 4])
        self.assertEqual(
            [attempt.outcome for attempt in result.retrieval_attempts],
            ["activated", "abstained"],
        )
        cleared = result.retrieval_attempts[1]
        self.assertTrue(cleared.terminal_abstain)
        self.assertTrue(cleared.memory_cleared_on_abstain)
        self.assertEqual(cleared.cleared_memory_id, "memory-a")
        self.assertEqual(cleared.clear_affects_generated_token_index, 2)
        self.assertEqual(cleared.deactivation_baseline_first_token_id, 2)
        self.assertEqual(cleared.deactivation_native_first_token_id, 3)
        self.assertTrue(cleared.deactivation_first_step_top1_changed)
        self.assertGreater(cleared.deactivation_first_step_logits_kl, 0.0)
        self.assertIsNotNone(cleared.deactivation_forward_seconds)
        self.assertEqual(cleared.actual_path_after_abstain, "native")
        self.assertIsNone(cleared.actual_path_memory_id_after)
        self.assertEqual(
            [transition.transition for transition in result.memory_transitions],
            ["activated", "deactivated_on_terminal_abstain"],
        )
        transition = result.memory_transitions[-1]
        self.assertEqual(transition.previous_memory_id, "memory-a")
        self.assertIsNone(transition.next_memory_id)
        self.assertEqual(model.caches[0].crop_calls, [2, 3])
        self.assertEqual(
            [(trace.generated_input_index, trace.trace.memory_id)
             for trace in result.attention_traces],
            [(0, "memory-a")],
        )
        self.assertIsNone(result.final_memory_id)
        self.assertIsNone(controller.active_memory)
        serialized = result.to_dict()
        self.assertEqual(
            serialized["summary"][
                "stale_memory_attention_after_terminal_clear_count"
            ],
            0,
        )
        self.assertTrue(
            serialized["summary"]["terminal_clear_attention_safe"]
        )

    def test_two_low_rearm_duplicate_replace_and_max_three_are_preserved(self):
        profile = self._profile()

        def token_for_path(length: int, memory_id: str | None) -> int:
            if length >= 11:
                return 0
            if memory_id == "memory-b":
                return 3
            return 2 if memory_id == "memory-a" else 1

        system, model, _, encoder, _, _ = self._system(
            profile=profile,
            static_selection=FakeStaticSelection(
                memory_ids=("memory-a", "memory-b"),
                available=True,
            ),
            outcomes=("memory-a", "memory-a", "memory-b"),
            entropies=(
                2.0, 0.1, 0.1, 2.0, 0.1, 0.1, 2.0, 2.0, 2.0
            ),
            token_for_path=token_for_path,
            max_new_tokens=10,
        )
        result = system.generate(
            prompt_token_ids=(10, 11), question="question"
        )
        self.assertEqual(result.retrieval_attempt_count, 3)
        self.assertEqual(result.rearm_count, 2)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.replacement_count, 1)
        self.assertEqual(
            [attempt.outcome for attempt in result.retrieval_attempts],
            ["activated", "duplicate", "replaced"],
        )
        self.assertEqual([len(ids) for ids in encoder.calls], [3, 6, 9])
        self.assertEqual(
            [trace.action for trace in result.boundary_traces],
            [
                "retrieval_attempt",
                "observed_disarmed_low_streak",
                "rearmed",
                "retrieval_attempt",
                "observed_disarmed_low_streak",
                "rearmed",
                "retrieval_attempt",
                "observed_exhausted",
                "observed_exhausted",
            ],
        )
        self.assertEqual(
            [
                trace.retrieval_attempt_count_after
                - trace.retrieval_attempt_count_before
                for trace in result.boundary_traces
                if trace.action == "rearmed"
            ],
            [0, 0],
        )
        self.assertEqual(model.caches[0].crop_calls, [2, 8])
        self.assertEqual(result.final_memory_id, "memory-b")
        self.assertEqual(result.completion_token_ids[-1], 0)
        self.assertTrue(all(
            attempt.terminal_abstain is False
            and attempt.memory_cleared_on_abstain is False
            for attempt in result.retrieval_attempts
        ))
        serialized = result.to_dict()
        self.assertTrue(serialized["summary"]["max_three_attempts_respected"])
        self.assertTrue(serialized["summary"]["two_low_rearm_respected"])
        self.assertTrue(
            serialized["summary"]["second_low_rearms_without_trigger"]
        )

    def test_v34_abstain_keeps_active_memory_and_legacy_trace_shape(self):
        from memgen.experience.v3 import (
            ExperienceMemoryV3Profile,
            V34_GENERATION_RESULT_SCHEMA,
        )
        from memgen.model.v3_runtime import OnlineExperienceMemorySystemV3

        profile = ExperienceMemoryV3Profile.continuous_token_joint(
            retrieval_abstention_policy="top1_top2_margin",
            retrieval_min_top1_top2_margin=0.1,
        )
        controller = FakeController(
            memory_score_bias=profile.memory_score_bias
        )

        def token_for_path(length: int, memory_id: str | None) -> int:
            if length >= 7:
                return 0
            return 2 if memory_id == "memory-a" else 1

        model = FakeModel(
            controller=controller,
            token_for_path=token_for_path,
        )
        gate = FakeGate((2.0, 0.1, 0.1, 2.0, 0.1))
        encoder = FakeQueryEncoder(
            query_pooling=profile.query_pooling,
            controller=controller,
        )
        system = OnlineExperienceMemorySystemV3(
            model=model,
            tokenizer=FakeTokenizer(),
            device="cpu",
            max_new_tokens=6,
            gate=gate,
            query_encoder=encoder,
            retriever=FakeLegacyRetriever(
                profile=profile,
                outcomes=("memory-a", None),
            ),
            loader=FakeLoader(),
            controller=controller,
            profile=profile,
        )
        result = system.generate(prompt_token_ids=(10, 11))
        self.assertEqual(result.schema_version, V34_GENERATION_RESULT_SCHEMA)
        self.assertEqual(
            [attempt.outcome for attempt in result.retrieval_attempts],
            ["activated", "abstained"],
        )
        abstain = result.retrieval_attempts[-1]
        self.assertIsNone(abstain.terminal_abstain)
        self.assertEqual(abstain.active_memory_id_after, "memory-a")
        self.assertEqual(result.final_memory_id, "memory-a")
        self.assertNotIn("terminal_abstain", abstain.to_dict())
        self.assertNotIn("static_selector_trace", result.to_dict())
        self.assertEqual(
            [transition.transition for transition in result.memory_transitions],
            ["activated"],
        )


if __name__ == "__main__":
    unittest.main()
