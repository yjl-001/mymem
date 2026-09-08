from copy import deepcopy
from types import SimpleNamespace
import unittest

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.experience.v4_3_bank import canonical_hash, seal, text_hash
from memgen.experience.v4_3_reasoner import identity, replay_contract, resolved_reasoner, validate_tokenizer_replay

SERVER_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["role"] + ": " + m["content"] for m in messages) + "\nassistant: "


def replay_fixture():
    source = dict(model_name="Qwen/Qwen2.5-1.5B-Instruct", model_revision=SERVER_REVISION, tokenizer_revision="main")
    events, evidence = [], {}
    tokenizer = Tokenizer()
    for i in range(116):
        eid, sid = f"e{i}", f"s{i}"
        e = {"sample_id": sid, "question": f"Compute the quantity for sample {i}.",
             "verified_success_trajectory": f"A valid operation for sample {i}.",
             "verified_failure_trajectory": f"An invalid operation for sample {i}."}
        evidence[eid] = e
        common = {"sample_id": sid, "experience_id": eid, "question_sha256": text_hash(e["question"]),
                  "completion_hashes": {"verified_success_completion_sha256": text_hash(e["verified_success_trajectory"]),
                                        "verified_failure_completion_sha256": text_hash(e["verified_failure_trajectory"])}}
        prompt = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, e["question"])
        events.append(seal({**common, "event_id": eid + "prompt", "event_kind": "prompt_semantic",
                           "prompt_token_count": len(prompt), "prompt_token_ids_sha256": canonical_hash(prompt)}))
        for kind, field in (("failure_gate_attempt", "verified_failure_trajectory"), ("success_gate_attempt", "verified_success_trajectory")):
            prefix = prompt + tokenizer.encode(e[field])[:10]
            events.append(seal({**common, "event_id": eid + kind, "event_kind": kind, "token_position": len(prefix) - 1,
                               "prefix_alignment": {"prefix_token_ids_sha256": canonical_hash(prefix)}}))
    cache = SimpleNamespace(events=events, manifest={"reasoner": source, "manifest_sha256": "c" * 64})
    return dict(cache=cache, evidence=evidence, source_reasoner=source, packets_sha256="d" * 64)


class V43ReasonerTests(unittest.TestCase):
    def test_actual_server_legacy_main_resolves_to_frozen_model_commit_without_mutation(self):
        source = replay_fixture()["source_reasoner"]
        before = deepcopy(source)
        effective = resolved_reasoner(source)
        self.assertEqual(effective["model_revision"], SERVER_REVISION)
        self.assertEqual(effective["tokenizer_revision"], SERVER_REVISION)
        self.assertEqual(source, before)

    def test_already_pinned_tokenizer_is_not_replaced(self):
        source = dict(model_name="model", model_revision="a" * 40, tokenizer_revision="b" * 40)
        self.assertEqual(resolved_reasoner(source), source)
        for bad in (None, "latest", "release"):
            with self.assertRaisesRegex(ValueError, "Tokenizer revision"):
                resolved_reasoner({**source, "tokenizer_revision": bad})
        with self.assertRaisesRegex(ValueError, "model revision"):
            resolved_reasoner({**source, "model_revision": "main"})

    def test_all_116_prompts_and_actual_gates_are_verified(self):
        inputs = replay_fixture()
        proof = validate_tokenizer_replay(tokenizer=Tokenizer(), **inputs)
        self.assertEqual(proof["prompt_count"], 116)
        self.assertEqual(proof["event_count"], 348)
        self.assertEqual(proof, replay_contract(**inputs))
        self.assertFalse(proof["historical_tokenizer_file_equivalence_claim"])
        self.assertEqual(proof["source_reasoner"]["tokenizer_revision"], "main")

    def test_mismatch_outside_smoke_subset_stops_migration(self):
        inputs = replay_fixture()
        class ChangedTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                return ids + [999] if "sample 115." in text else ids
        with self.assertRaisesRegex(ValueError, "legacy prompt"):
            validate_tokenizer_replay(tokenizer=ChangedTokenizer(), **inputs)

    def test_same_prompts_but_different_gate_tokens_are_rejected(self):
        inputs = replay_fixture()
        class ChangedTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                if text.startswith("A valid operation"):
                    ids[0] += 1
                return ids
        with self.assertRaisesRegex(ValueError, "gate prefix"):
            validate_tokenizer_replay(tokenizer=ChangedTokenizer(), **inputs)

    def test_incomplete_cache_or_mismatched_source_cannot_certify_migration(self):
        inputs = replay_fixture()
        inputs["cache"].events = inputs["cache"].events[:-3]
        with self.assertRaisesRegex(ValueError, "full 116"):
            replay_contract(**inputs)
        inputs = replay_fixture()
        inputs["source_reasoner"] = {**inputs["source_reasoner"], "model_revision": "f" * 40}
        with self.assertRaisesRegex(ValueError, "identities differ"):
            replay_contract(**inputs)


if __name__ == "__main__":
    unittest.main()
