"""Real artifact handoff from 17/116 construction through both tiers to audit planning."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tests.test_v4_3_side_kv import TinyTokenizer, tiny_model, torch
from tests.test_v4_3_pipeline_contract import write_fixture
from tests.test_v4_source_state_cache import prompt_event, gate_event, fake_tensors
from memgen.experience.v4_3_artifacts import read_jsonl
from memgen.experience.v4_3_bank import canonical_hash, file_hash, text_hash
from scripts.build_v4_3_unified_bank import construct, encode_outputs, write_or_validate
from scripts.audit_v4_3_unified_memory import prepare
from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.experience.v4_3_reasoner import resolved_reasoner
from memgen.experience.v4_3_artifacts import atomic_json
from memgen.experience.v4_3_bank import seal


@unittest.skipIf(torch is None, "Torch/Transformers test environment unavailable")
class V43ArtifactIntegrationTests(unittest.TestCase):
    def test_authenticated_source_cache_and_both_compiled_tiers_feed_real_prepare(self):
        self.artifact_handoff(legacy=False)

    def test_legacy_main_compiler_pins_replays_and_feeds_audit_without_rewriting_source(self):
        self.artifact_handoff(legacy=True)

    def test_deepseek_cards_compile_both_tiers_and_feed_full_audit_plan(self):
        self.artifact_handoff(legacy=True, semantic=True)

    def artifact_handoff(self, *, legacy, semantic=False):
        from memgen.experience.v4_source_state import finalize_event, save_source_state_cache
        from memgen.model.v4_3_side_kv import V43SideKVCompiler, save_compiled

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, packets_path, policy = write_fixture(root)
            bank_dir, side_dir = root / "unified", root / "compiled"
            if semantic:
                from scripts.build_v4_3_deepseek_bank import construct as deepseek_construct
                from tests.test_v4_3_deepseek import FakeClient
                import os
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "synthetic-test-only"}):
                    outputs = deepseek_construct(source_dir=source, packets_path=packets_path, policy_path=policy,
                        output_dir=bank_dir, cache_dir=root / "teacher-cache", client_factory=FakeClient)
            else:
                outputs = construct(source_dir=source, packets_path=packets_path, policy_path=policy)
                write_or_validate(bank_dir, encode_outputs(outputs), resume=False)
            source_reasoner = dict(model_name="tiny-local-qwen", model_revision="a" * 40,
                                   tokenizer_revision="main" if legacy else "b" * 40, model_sequence_limit=4096)
            reasoner = resolved_reasoner(source_reasoner)
            tokenizer, model = TinyTokenizer(), tiny_model(torch.bfloat16)
            tokenizer.init_kwargs = {}
            evidence = {e["evidence_id"]: e for p in read_jsonl(packets_path) for e in p["evidence"]}
            events = []
            row = 0
            for r in outputs["candidate_bank_records.jsonl"]:
                for eid, sid in zip(r["construction"]["experience_ids"], r["construction"]["sample_ids"]):
                    e = evidence[eid]
                    common = {"experience_id": eid, "sample_id": sid,
                        "independent_sample_id": canonical_hash({"benchmark": "openai/gsm8k", "logical_split": "bank-source", "sample_id": sid}),
                        "question_sha256": text_hash(e["question"].strip()), "curation_tier": r["quality_tier"],
                        "bank_record_sha256": r["curation_provenance"]["source_curated_record_sha256"],
                        "completion_hashes": {"verified_success_completion_sha256": text_hash(e["verified_success_trajectory"].strip()),
                                              "verified_failure_completion_sha256": text_hash(e["verified_failure_trajectory"].strip())}}
                    samples = [prompt_event("a1", r["source_v42_bank_id"], row, failure_attempts=1, success_attempts=1),
                               gate_event("a1", r["source_v42_bank_id"], row, 1, success=False),
                               gate_event("a1", r["source_v42_bank_id"], row, 1, success=True)]
                    for event in samples:
                        event.update(common)
                        event["event_id"] = eid + "::" + event["event_kind"]
                        prompt = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, e["question"])
                        if event["event_kind"] == "prompt_semantic":
                            event.update(prompt_token_count=len(prompt), prompt_token_ids_sha256=canonical_hash(prompt))
                        else:
                            field = "verified_success_trajectory" if event["event_kind"] == "success_gate_attempt" else "verified_failure_trajectory"
                            prefix = prompt + tokenizer.encode(e[field].strip())[:3]
                            event["token_position"] = len(prefix) - 1
                            event["prefix_alignment"].update(prefix_token_count=len(prefix), prefix_token_ids_sha256=canonical_hash(prefix))
                        events.append(finalize_event(event))
                    row += 1
            # Source-state values are explicit synthetic fixtures; these tests
            # exercise file/schema binding, not recovered trajectory semantics.
            tensor_spec = fake_tensors(events)
            tensors = {name: torch.tensor(t.rows, dtype=torch.bool) if name.endswith("_masks")
                       else torch.zeros(t.shape, dtype=torch.bfloat16) for name, t in tensor_spec.items()}
            risk_path = root / "risk.pt"
            risk_path.write_bytes(b"synthetic risk identity fixture; never deserialized")
            cache_path, _ = save_source_state_cache(output_dir=root / "cache", tensors=tensors, events=events,
                repository_revision="fixture", reasoner=source_reasoner,
                configuration={"layer_number": 24, "attention_implementation": "sdpa", "dtype": "bfloat16",
                               "maximum_gate_attempts": 3, "maximum_hidden_window": 32, "support_unit": "independent_sample"},
                provenance={"construction_profile_sha256": "construction-profile", "bank_manifest_logical_sha256": "bank-manifest",
                    "side_kv_manifest_logical_sha256": "side-kv-manifest", "inputs": {
                        "bank_records_sha256": file_hash(source / "bank_records.jsonl"),
                        "bank_manifest_file_sha256": file_hash(source / "bank_manifest.json"),
                        "token_risk_artifact_sha256": file_hash(risk_path)}}, implementation_sha256={"fixture.py": "fixture-sha"})
            cache_bytes = cache_path.read_bytes()
            if legacy:
                from scripts import compile_v4_3_side_kv as cli
                old_path = root / "legacy_reasoner_manifest.json"
                atomic_json(old_path, seal({"reasoner": source_reasoner}, "manifest_sha256"))
                command = ["compile", "--bank-dir", str(bank_dir), "--reasoner-manifest", str(old_path),
                           "--output-dir", str(side_dir), "--device", "cpu", "--cache-manifest", str(cache_path),
                           "--semantic-packets", str(packets_path), "--resume"]
                with patch("sys.argv", command), patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer, patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load_model:
                    cli.main()
                    self.assertEqual(load_tokenizer.call_args.kwargs["revision"], "a" * 40)
                    self.assertEqual(load_model.call_args.kwargs["revision"], "a" * 40)
                with patch("sys.argv", command), patch("transformers.AutoTokenizer.from_pretrained") as load_tokenizer, patch("transformers.AutoModelForCausalLM.from_pretrained") as load_model:
                    cli.main()
                    load_tokenizer.assert_not_called()
                    load_model.assert_not_called()
                self.assertEqual(cache_bytes, cache_path.read_bytes())
            else:
                compiler = V43SideKVCompiler(model=model, tokenizer=tokenizer, reasoner=reasoner)
                for tier in ("primary", "conditional"):
                    tensors, manifest = compiler.compile(outputs[f"{tier}_bank_records.jsonl"], outputs[f"{tier}_bank_manifest.json"])
                    save_compiled(side_dir, tensors, manifest)
            args = SimpleNamespace(bank_dir=bank_dir, side_kv_dir=side_dir, semantic_packets=packets_path,
                                   cache_manifest=cache_path, token_risk_artifact=risk_path, mode="smoke", all_bank_sweep=False,
                                   bank_scope="all")
            prepared = prepare(args)
            self.assertEqual(len(prepared[3]), 17)
            self.assertEqual(len(prepared[4]), 17)
            self.assertEqual(prepared[-2]["case_count"], 16)
            self.assertFalse(prepared[-2]["qualified_for_online_use"])
            self.assertEqual(prepared[-1]["experiment"]["source_reasoner"]["tokenizer_revision"], source_reasoner["tokenizer_revision"])
            self.assertEqual(prepared[-1]["experiment"]["reasoner"]["tokenizer_revision"], reasoner["tokenizer_revision"])
            args.mode = "full"
            full = prepare(args)
            self.assertEqual(full[-2]["case_count"], 464)
            self.assertEqual(prepared[-1]["experiment_identity_sha256"], full[-1]["experiment_identity_sha256"])
            # Primary scope reuses the very same compiled files, without a
            # conditional manifest/tensor dependency or conditional controls.
            for path in side_dir.glob("*conditional*"):
                path.unlink()
            args.bank_scope = "primary"
            primary = prepare(args)
            self.assertEqual(len(primary[3]), 11)
            self.assertEqual(len(primary[4]), 11)
            self.assertEqual(primary[-2]["source_sample_count"], 76)
            self.assertEqual(primary[-2]["case_count"], 304)
            self.assertNotEqual(primary[-1]["experiment_identity_sha256"], full[-1]["experiment_identity_sha256"])
            self.assertTrue(all(c["quality_tier"] == "primary" for c in primary[-2]["cases"]))
            from scripts.audit_v4_3_prefix_equivalence import prepare_experiment
            args.atol, args.rtol, args.device = .05, .02, "cpu"
            _, eq_cases, eq_profile = prepare_experiment(args)
            self.assertEqual(len(eq_cases), 76)
            self.assertEqual(eq_profile["configuration"]["bank_count"], 11)
            self.assertTrue(all(c["audit_layer"] == "visible_content" for c in eq_cases))
            risk_path.write_bytes(b"different fixture")
            with self.assertRaisesRegex(ValueError, "lineage/file identity"):
                prepare(args)


if __name__ == "__main__":
    unittest.main()
