from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v3 import ExperienceMemoryV3Profile
from memgen.experience.v3_artifacts import validate_cross_bank_metadata
from memgen.experience.v3_eval import summarize_v3_rows


class V3PureContractTests(unittest.TestCase):
    def test_profile_freezes_full_prefix_three_attempt_replacement(self) -> None:
        profile = ExperienceMemoryV3Profile()
        self.assertEqual(profile.layer_number, 24)
        self.assertEqual(profile.query_context, "question_plus_full_partial_cot")
        self.assertEqual(profile.max_retrieval_attempts, 3)
        self.assertEqual(profile.replacement_policy, "replace_current_memory")
        self.assertEqual(profile.risk_role, "diagnostic_only")
        with self.assertRaisesRegex(ValueError, "layer 24"):
            ExperienceMemoryV3Profile(layer_number=23)

    def test_cross_bank_validation_binds_embedding_key_to_kv_value(self) -> None:
        record = SimpleNamespace(
            memory_id="memory-a",
            payload_hash="payload-a",
            token_count=7,
            kv_layer=24,
        )
        reasoner = {
            "model_name": "reasoner",
            "model_revision": "model-rev",
            "tokenizer_revision": "tokenizer-rev",
        }
        side = {
            "schema_version": "canonical-side-kv-bank-v2",
            "canonical_pre_rope": True,
            "layer_number": 24,
            "compiler": {"attention_backend": "sdpa"},
            "reasoner": reasoner,
            "records": [{
                "memory_id": "memory-a",
                "payload_hash": "payload-a",
                "kv_valid_slot_count": 7,
            }],
        }
        side["manifest_sha256"] = canonical_json_sha256(side)
        key = {
            "schema_version": "experience-memory-retrieval-key-bank-v1",
            "reasoner": reasoner | {"attention_implementation": "sdpa"},
            "records": [{
                "memory_id": "memory-a",
                "payload_hash": "payload-a",
            }],
        }
        key["manifest_sha256"] = canonical_json_sha256(key)
        report = validate_cross_bank_metadata(
            records=(record,), side_manifest=side, key_manifest=key
        )
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["layer_number"], 24)
        key["records"][0]["payload_hash"] = "other"
        key["manifest_sha256"] = canonical_json_sha256({
            name: value for name, value in key.items() if name != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "different KV payload"):
            validate_cross_bank_metadata(
                records=(record,), side_manifest=side, key_manifest=key
            )

    def test_summary_has_only_strict_format_and_token_task_metrics(self) -> None:
        def row(vanilla, v3, vanilla_tokens, v3_tokens, attempts):
            return {
                "conditions": {
                    "vanilla": {
                        "strict_correct": vanilla,
                        "format_correct": True,
                        "generated_token_count": vanilla_tokens,
                    },
                    "v3": {
                        "strict_correct": v3,
                        "format_correct": v3,
                        "generated_token_count": v3_tokens,
                        "online_diagnostics": {
                            "retrieval_attempt_count": attempts,
                            "rearm_count": max(0, attempts - 1),
                            "activation_count": int(attempts > 0),
                            "replacement_count": int(attempts > 1),
                            "duplicate_count": 0,
                            "abstain_count": 0,
                            "memory_attention_step_count": attempts * 2,
                        },
                    },
                }
            }

        summary = summarize_v3_rows((
            row(False, True, 10, 12, 2),
            row(True, True, 20, 18, 0),
        ))
        self.assertEqual(summary["conditions"]["vanilla"]["strict_accuracy"], 0.5)
        self.assertEqual(summary["conditions"]["v3"]["strict_accuracy"], 1.0)
        self.assertEqual(
            summary["paired"]["generated_token_delta_v3_minus_vanilla"]["total"],
            0.0,
        )
        self.assertFalse(
            summary["metric_contract"]["diagnostic_answer_accuracy_aggregated"]
        )


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for the V3 state-machine test")
class V3RuntimeStateMachineTests(unittest.TestCase):
    def test_exact_cosine_uses_bank_order_as_stable_tie_breaker(self) -> None:
        from memgen.model.retrieval_keys import EmbeddingMemoryRetriever

        profile = ExperienceMemoryV3Profile()
        records = tuple(
            SimpleNamespace(
                memory_id=f"memory-{suffix}",
                payload_hash=f"payload-{suffix}",
                token_count=2,
            )
            for suffix in ("a", "b")
        )
        entries = tuple({
            "index": index,
            "memory_id": record.memory_id,
            "payload_hash": record.payload_hash,
            "payload_token_count": record.token_count,
            "key_embedding_sha256": f"key-{index}",
        } for index, record in enumerate(records))
        key_bank = SimpleNamespace(
            embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            entries=entries,
            entry_by_id={entry["memory_id"]: entry for entry in entries},
        )
        retriever = EmbeddingMemoryRetriever(
            key_bank=key_bank,
            records=records,
            kv_valid_slot_counts={"memory-a": 2, "memory-b": 2},
            profile=profile,
        )
        decision = retriever.retrieve(
            query_embedding=torch.tensor([1.0, 0.0]),
            query_token_ids=(10, 11, 12),
            prompt_token_count=2,
        )
        self.assertEqual(
            [hit["memory_id"] for hit in decision.hits],
            ["memory-a", "memory-b"],
        )
        self.assertEqual(decision.matched_memory.memory_id, "memory-a")

    def test_margin_selector_abstains_but_retains_top2_diagnostics(self) -> None:
        from memgen.model.retrieval_keys import EmbeddingMemoryRetriever

        profile = ExperienceMemoryV3Profile(
            retrieval_abstention_policy="top1_top2_margin",
            retrieval_min_top1_top2_margin=0.1,
        )
        records = tuple(
            SimpleNamespace(
                memory_id=f"memory-{suffix}",
                payload_hash=f"payload-{suffix}",
                token_count=2,
            )
            for suffix in ("a", "b")
        )
        entries = tuple({
            "index": index,
            "memory_id": record.memory_id,
            "payload_hash": record.payload_hash,
            "payload_token_count": record.token_count,
            "key_embedding_sha256": f"key-{index}",
        } for index, record in enumerate(records))
        key_bank = SimpleNamespace(
            embeddings=torch.tensor([[1.0, 0.0], [0.999, 0.0447]]),
            entries=entries,
            entry_by_id={entry["memory_id"]: entry for entry in entries},
        )
        retriever = EmbeddingMemoryRetriever(
            key_bank=key_bank,
            records=records,
            kv_valid_slot_counts={"memory-a": 2, "memory-b": 2},
            profile=profile,
        )
        decision = retriever.retrieve(
            query_embedding=torch.tensor([1.0, 0.0]),
            query_token_ids=(10, 11, 12),
            prompt_token_count=2,
        )
        self.assertEqual(decision.status, "below_margin")
        self.assertIsNone(decision.matched_memory)
        self.assertEqual(len(decision.hits), 2)
        self.assertFalse(decision.query["margin_qualified"])
        self.assertEqual(
            decision.query["minimum_top1_top2_margin"], 0.1
        )

    def test_rearm_allows_three_attempts_and_replaces_current_memory(self) -> None:
        from memgen.experience.e1 import MemoryChoice
        from memgen.experience.v3 import EmbeddingRetrievalDecision
        from memgen.model.e1_runtime import GateProbe
        from memgen.model.side_kv import SideKVAttentionTrace
        from memgen.model.v3_runtime import OnlineExperienceMemorySystemV3

        profile = ExperienceMemoryV3Profile()

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            @staticmethod
            def decode(token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join("." if int(token) == 1 else "x" for token in token_ids)

        class FakeController:
            layer_number = 24
            memory_score_normalization = profile.memory_score_normalization
            memory_score_bias = profile.memory_score_bias

            def __init__(self):
                self.active_memory = None
                self._traces = []

            @property
            def traces(self):
                return tuple(self._traces)

            def clear_traces(self):
                self._traces.clear()

            def truncate_traces(self, length):
                del self._traces[length:]

            def activate(self, memory):
                self.active_memory = memory

            def deactivate(self):
                self.active_memory = None

            @contextmanager
            def suspend_memory(self):
                previous = self.active_memory
                self.active_memory = None
                try:
                    yield previous
                finally:
                    self.active_memory = previous

            def append_trace(self, native_length):
                if self.active_memory is None:
                    return
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

        controller = FakeController()

        class FakeModel:
            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.zeros(1))

            def parameters(self):
                yield self.parameter

            def __call__(self, *, attention_mask, **kwargs):
                del kwargs
                controller.append_trace(int(attention_mask.shape[-1]))
                next_token = (
                    0
                    if controller.active_memory is not None
                    and controller.active_memory.memory_id == "memory-b"
                    else 1
                )
                logits = torch.full((1, 1, 4), -10.0)
                logits[0, 0, next_token] = 10.0
                return SimpleNamespace(
                    logits=logits,
                    past_key_values={"length": int(attention_mask.shape[-1])},
                )

        model = FakeModel()

        class FakeGate:
            config = SimpleNamespace(
                layer_number=24,
                high_entropy_threshold=1.0,
                low_entropy_threshold=0.5,
                risk_threshold=0.0,
                risk_role="diagnostic_only",
            )

            def __init__(self):
                self.entropies = iter((2.0, 0.1, 2.0, 0.1, 2.0))

            def probe(self, **kwargs):
                attention_mask = kwargs["attention_mask"]
                controller.append_trace(int(attention_mask.shape[-1]))
                next_token = (
                    0
                    if controller.active_memory is not None
                    and controller.active_memory.memory_id == "memory-b"
                    else 1
                )
                logits = torch.full((1, 1, 4), -10.0)
                logits[0, 0, next_token] = 10.0
                return GateProbe(
                    entropy=next(self.entropies),
                    risk_score=-100.0,
                    output=SimpleNamespace(
                        logits=logits,
                        past_key_values={
                            "length": int(attention_mask.shape[-1])
                        },
                    ),
                )

        class FakeQueryEncoder:
            layer_number = 24

            def __init__(self):
                self.calls = []

            def encode(self, token_ids):
                self.calls.append(tuple(token_ids))
                return torch.tensor([1.0, 0.0])

        query_encoder = FakeQueryEncoder()

        class FakeRetriever:
            def __init__(self):
                self.profile = profile
                self.memory_ids = iter(("memory-a", "memory-a", "memory-b"))

            def retrieve(
                self, *, query_embedding, query_token_ids, prompt_token_count
            ):
                del query_embedding
                memory_id = next(self.memory_ids)
                choice = MemoryChoice(
                    memory_id=memory_id,
                    payload_hash=f"payload-{memory_id[-1]}",
                    token_count=2,
                    kv_valid_slot_count=2,
                    retrieval_score=1.0,
                    retrieval_rank=1,
                )
                return EmbeddingRetrievalDecision(
                    status="selected",
                    query={
                        "query_token_count": len(query_token_ids),
                        "prompt_token_count": prompt_token_count,
                        "partial_cot_token_count": (
                            len(query_token_ids) - prompt_token_count
                        ),
                    },
                    hits=({"memory_id": memory_id},),
                    matched_memory=choice,
                )

        class FakeLoader:
            @staticmethod
            def get(memory_id, *, device, dtype):
                del device, dtype
                return SimpleNamespace(
                    memory_id=memory_id,
                    payload_hash=f"payload-{memory_id[-1]}",
                    valid_slot_count=2,
                    layer_number=24,
                )

        system = OnlineExperienceMemorySystemV3(
            model=model,
            tokenizer=FakeTokenizer(),
            device="cpu",
            max_new_tokens=6,
            gate=FakeGate(),
            query_encoder=query_encoder,
            retriever=FakeRetriever(),
            loader=FakeLoader(),
            controller=controller,
            profile=profile,
        )
        result = system.generate(prompt_token_ids=(10, 11))
        self.assertEqual(result.completion_token_ids, (1, 1, 1, 1, 1, 0))
        self.assertEqual(result.retrieval_attempt_count, 3)
        self.assertEqual(result.rearm_count, 2)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.replacement_count, 1)
        self.assertEqual(
            [attempt.outcome for attempt in result.retrieval_attempts],
            ["activated", "duplicate", "replaced"],
        )
        self.assertEqual([len(value) for value in query_encoder.calls], [3, 5, 7])
        self.assertEqual(
            [trace.action for trace in result.boundary_traces],
            [
                "retrieval_attempt",
                "rearmed",
                "retrieval_attempt",
                "rearmed",
                "retrieval_attempt",
            ],
        )
        self.assertEqual(len(result.attention_traces), 5)
        self.assertEqual(result.final_memory_id, "memory-b")
        self.assertEqual(result.final_gate_state, "CLOSED")
        self.assertIsNone(controller.active_memory)


if __name__ == "__main__":
    unittest.main()
