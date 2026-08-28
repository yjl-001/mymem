from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - repository tests have Torch.
    torch = None

from memgen.experience.memory import (
    ApprovedMemorySourceSelector,
    MemoryBankBuilder,
    MemoryRecordCompiler,
    MemorySanitizerConfig,
    PayloadSanitizer,
)
from memgen.experience.phase1 import (
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    text_sha256,
)
from memgen.experience.v3_5_selector import V35_DUAL_KEY_BANK_SCHEMA


COMPILE_SCRIPT = PROJECT_ROOT / "scripts" / "compile_v3_5_dual_selector.py"


class V35SourceStructureTests(unittest.TestCase):
    def test_retriever_has_exactly_one_retrieve_definition(self) -> None:
        source_path = PROJECT_ROOT / "memgen" / "model" / "v3_5_retrieval.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        retriever = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ApplicabilityAwareMemoryRetriever"
        )
        retrieve_definitions = [
            node
            for node in retriever.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "retrieve"
        ]
        self.assertEqual(len(retrieve_definitions), 1)


class FixtureTokenizer:
    def __init__(self) -> None:
        self.id_by_token: dict[str, int] = {}
        self.token_by_id: dict[int, str] = {}
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls.append((text, add_special_tokens))
        values: list[int] = []
        for token in text.split():
            if token not in self.id_by_token:
                token_id = len(self.id_by_token) + 11
                self.id_by_token[token] = token_id
                self.token_by_id[token_id] = token
            values.append(self.id_by_token[token])
        return values

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(self.token_by_id[int(value)] for value in token_ids)


if torch is not None:
    class FixtureModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(
                torch.zeros((), dtype=torch.bfloat16)
            )
            self.model = SimpleNamespace(layers=tuple(object() for _ in range(24)))
            self.seen: list[tuple[int, ...]] = []

        def forward(self, *, input_ids, **kwargs):
            del kwargs
            token_ids = tuple(int(value) for value in input_ids[0])
            self.seen.append(token_ids)
            rows = []
            for position, token_id in enumerate(token_ids):
                rows.append([
                    float(token_id % 7 + 1),
                    float(token_id % 11 + 2),
                    float(token_id % 13 + position + 1),
                    1.0,
                ])
            hidden = torch.tensor(
                [rows], dtype=torch.float32, device=input_ids.device
            )
            return SimpleNamespace(hidden_states=tuple(hidden for _ in range(25)))


def _supported_assessment() -> dict:
    return {"status": "supported", "evidence": "Independent grounded review."}


def _fixture_context(suffix: str) -> str:
    return f"A {suffix} word problem asks for a derived quantity."


def _alphabetic_suffix(index: int) -> str:
    letters = ""
    value = index
    while True:
        letters = chr(ord("a") + value % 26) + letters
        value = value // 26 - 1
        if value < 0:
            return f"source{letters}"


def _approved_record(suffix: str, source_index: int) -> dict:
    target = {
        "situation_signature": (
            f"A {suffix} calculation preserves each stated relationship."
        ),
        "transferable_decision": (
            f"Carry the {suffix} supported relation forward before simplifying."
        ),
        "verification_rule": (
            f"VERIFIER_{suffix} inspect every operation before finalization."
        ),
        "applicability_boundary": (
            f"Use this while the {suffix} relation remains supported."
        ),
        "confidence": 0.9,
    }
    reference = {
        "competing_pattern": f"AVOID_{suffix} replace the relation with a shortcut.",
        "failure_signal": f"The {suffix} quantity stops following the statement.",
        "failure_mechanism": f"The {suffix} shortcut changes the relation.",
        "non_reuse_boundary": "Reconsider only when new evidence invalidates it.",
        "confidence": 0.9,
    }
    experience_id = f"experience-{suffix}"
    return {
        "schema_version": "teacher-bank-record-v3",
        "experience_id": experience_id,
        "experience_type": "answer_correctness",
        "reference_evidence": "verified_failure",
        "reference_failure_types": ["boxed_answer_mismatch"],
        "provenance_sha256": f"phase1-{suffix}",
        "source": {
            "dataset": "openai/gsm8k",
            "dataset_revision": "fixture-revision",
            "dataset_split": "train",
            "logical_split": "bank-source",
            "source_index": source_index,
            "question_sha256": text_sha256(_fixture_context(suffix)),
            "split_manifest_sha256": "pending",
        },
        "student": {
            "model_name": "reasoner",
            "model_revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "frozen": True,
        },
        "source_episode_ids": {
            "target": f"target-{suffix}",
            "reference": f"reference-{suffix}",
        },
        "bank": {
            "experience_type": "answer_correctness",
            "failure_types": ["boxed_answer_mismatch"],
            "target": target,
            "reference": reference,
            "evidence": {
                "target_observation": "The accepted response was verified.",
                "reference_observation": "The rejected response failed.",
            },
        },
        "ai_review_gate": {
            "schema_version": "phase1-ai-review-record-v2",
            "prompt_version": "phase1-ai-review-v2-field-evidence-rubric",
            "route": "ai_approved",
            "review_provenance_sha256": f"review-{suffix}",
            "ai_review": {
                "decision": "approve",
                "confidence": 0.9,
                "field_assessments": {
                    "target": {
                        key: _supported_assessment()
                        for key in (
                            "situation_signature",
                            "transferable_decision",
                            "verification_rule",
                            "applicability_boundary",
                        )
                    },
                    "reference": {
                        key: _supported_assessment()
                        for key in (
                            "competing_pattern",
                            "failure_signal",
                            "failure_mechanism",
                            "non_reuse_boundary",
                        )
                    },
                },
                "pair_assessments": {
                    key: _supported_assessment()
                    for key in (
                        "target_reference_distinct",
                        "factually_consistent",
                        "causal_attribution",
                        "failure_type_compatibility",
                        "transferable_without_instance_leakage",
                    )
                },
                "issues": [],
            },
        },
    }


def _verified_experience(approved: dict) -> dict:
    suffix = approved["experience_id"].split("-", 1)[1]
    source = approved["source"]
    return {
        "experience_id": approved["experience_id"],
        "sample_id": (
            f"gsm8k-train-{source['source_index']}-"
            f"{str(source['question_sha256'])[:12]}"
        ),
        "experience_type": approved["experience_type"],
        "reference_evidence": approved["reference_evidence"],
        "reference_failure_types": copy.deepcopy(
            approved["reference_failure_types"]
        ),
        "provenance_sha256": approved["provenance_sha256"],
        "source": copy.deepcopy(approved["source"]),
        "student": copy.deepcopy(approved["student"]),
        "target_episode_id": approved["source_episode_ids"]["target"],
        "reference_episode_id": approved["source_episode_ids"]["reference"],
        "context": _fixture_context(suffix),
        "trajectory": f"The accepted {suffix} reasoning preserves the relation.",
        "reference_trajectory": f"The rejected {suffix} reasoning uses a shortcut.",
        "target_verifier": {"reward": 1.0},
        "reference_verifier": {"reward": 0.0},
    }


def _split_manifest(approved_records: tuple[dict, ...]) -> dict:
    samples = []
    for approved in approved_records:
        source = approved["source"]
        samples.append({
            "sample_id": (
                f"gsm8k-train-{source['source_index']}-"
                f"{str(source['question_sha256'])[:12]}"
            ),
            "logical_split": "bank-source",
            "dataset_split": "train",
            "source_index": source["source_index"],
            "question_sha256": source["question_sha256"],
            "answer_sha256": text_sha256(
                f"fixture-answer-{source['source_index']}"
            ),
        })
    reserved_index = len(samples)
    reserved_context = _fixture_context("gamma")
    samples.append({
        "sample_id": (
            f"gsm8k-train-{reserved_index}-"
            f"{text_sha256(reserved_context)[:12]}"
        ),
        "logical_split": "bank-source",
        "dataset_split": "train",
        "source_index": reserved_index,
        "question_sha256": text_sha256(reserved_context),
        "answer_sha256": text_sha256("fixture-answer-gamma"),
    })
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "created_at": "2026-01-01T00:00:00+00:00",
        "dataset": {
            "name": "openai/gsm8k",
            "configuration": "main",
            "revision": "fixture-revision",
            "train_fingerprint": "fixture-train",
            "test_fingerprint": "fixture-test",
            "train_size": len(samples),
            "test_size": 0,
        },
        "policy": {
            "seed": 3501,
            "bank_source_size": len(samples),
            "calibration_val_size": 0,
            "dev_test_size": 0,
            "final_test_source": "official-test",
        },
        "counts": {
            "bank-source": len(samples),
            "calibration-val": 0,
            "dev-test": 0,
            "final-test": 0,
        },
        "overlap_check": {"passed": True, "overlap_count": 0},
        "samples": samples,
    }
    manifest["manifest_sha256"] = canonical_json_sha256({
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    for approved in approved_records:
        approved["source"]["split_manifest_sha256"] = manifest[
            "manifest_sha256"
        ]
    return manifest


@unittest.skipIf(torch is None, "Torch is required for V3.5 retrieval tests")
class V35DualCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        from memgen.model.retrieval_keys import (
            RetrievalKeyBankLoader,
            RetrievalKeyCompiler,
        )

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer = FixtureTokenizer()
        self.model = FixtureModel().eval()
        self.approved = tuple(
            _approved_record(value, index)
            for index, value in enumerate(("alpha", "beta"))
        )
        self.split_manifest = _split_manifest(self.approved)
        self.verified = tuple(_verified_experience(value) for value in self.approved)
        builder = MemoryBankBuilder(
            selector=ApprovedMemorySourceSelector(),
            compiler=MemoryRecordCompiler(
                tokenizer=self.tokenizer,
                sanitizer=PayloadSanitizer(MemorySanitizerConfig()),
                reasoner_name="reasoner",
                reasoner_revision="model-revision",
                tokenizer_revision="tokenizer-revision",
                model_sequence_limit=4096,
                kv_layer=24,
            ),
        )
        self.builder = builder
        result = builder.build(self.approved, self.verified)
        self.assertEqual(len(result.records), 2)
        self.records = result.records
        old_compiler = RetrievalKeyCompiler(
            model=self.model,
            tokenizer=self.tokenizer,
            reasoner_name="reasoner",
            reasoner_revision="model-revision",
            tokenizer_revision="tokenizer-revision",
        )
        old_bank = old_compiler.compile(self.records)
        _, old_manifest_path = old_bank.save(self.root / "old")
        self.old_loader = RetrievalKeyBankLoader(manifest_path=old_manifest_path)
        self.side_manifest = {
            "schema_version": "canonical-side-kv-bank-v2",
            "canonical_pre_rope": True,
            "relative_phase_delta": 0,
            "layer_number": 24,
            "compiler": {"attention_backend": "sdpa"},
            "reasoner": {
                "model_name": "reasoner",
                "model_revision": "model-revision",
                "tokenizer_revision": "tokenizer-revision",
            },
            "record_count": 2,
            "records": [
                {
                    "index": index,
                    "memory_id": record.memory_id,
                    "payload_hash": record.payload_hash,
                    "payload_token_count": record.token_count,
                    "kv_valid_slot_count": record.token_count,
                }
                for index, record in enumerate(self.records)
            ],
            "record_order_sha256": canonical_json_sha256(
                [record.memory_id for record in self.records]
            ),
        }
        self.side_manifest["manifest_sha256"] = canonical_json_sha256(
            self.side_manifest
        )
        from memgen.model.v3_5_retrieval import (
            v35_implementation_files_sha256,
        )

        implementation_files = v35_implementation_files_sha256()
        self.provenance = {
            "memory_records_sha256": "records",
            "side_kv_manifest_sha256": "side",
            "e0_final_report_sha256": "e0",
            "v3_retrieval_key_manifest_sha256": file_sha256(old_manifest_path),
            "v3_retrieval_key_tensor_sha256": self.old_loader.manifest[
                "tensor_artifact"
            ]["sha256"],
            "v3_offline_report_sha256": "old-report",
            "phase1_approved_bank_sha256": "approved",
            "verified_experiences_sha256": "verified",
            "split_manifest_sha256": "split",
            "split_manifest_logical_sha256": self.split_manifest[
                "manifest_sha256"
            ],
            "dataset_revision": "fixture-revision",
            "compiler_git_revision": "revision",
            "compiler_tracked_diff_sha256": "tracked-diff",
            "compiler_implementation_files_sha256": implementation_files,
            "compiler_implementation_set_sha256": canonical_json_sha256(
                implementation_files
            ),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _compiler(self):
        from memgen.model.v3_5_retrieval import DualRetrievalKeyCompiler

        return DualRetrievalKeyCompiler(
            model=self.model,
            tokenizer=self.tokenizer,
            reasoner_name="reasoner",
            reasoner_revision="model-revision",
            tokenizer_revision="tokenizer-revision",
        )

    def test_compiler_reproduces_old_applicability_and_excludes_other_fields(self) -> None:
        from memgen.model.v3_5_retrieval import (
            DualRetrievalKeyBankLoader,
            V35_DYNAMIC_KEY_SOURCE,
        )

        self.model.seen.clear()
        compiled = self._compiler().compile(
            records=self.records,
            approved_records=self.approved,
            verified_experiences=self.verified,
            applicability_key_bank=self.old_loader,
            side_kv_manifest=self.side_manifest,
            split_manifest=self.split_manifest,
            artifact_provenance=self.provenance,
        )
        self.assertEqual(tuple(compiled.applicability_embeddings.shape), (2, 4))
        self.assertEqual(tuple(compiled.dynamic_embeddings.shape), (2, 4))
        self.assertTrue(torch.allclose(
            compiled.applicability_embeddings.norm(dim=-1), torch.ones(2)
        ))
        self.assertTrue(torch.allclose(
            compiled.dynamic_embeddings.norm(dim=-1), torch.ones(2)
        ))
        self.assertEqual(len(self.model.seen), 4)
        for index, record in enumerate(self.records):
            approved = self.approved[index]
            applicability_text = record.sanitized_fields["when_facing"].strip()
            dynamic_text = (
                f"When facing: {applicability_text}\n"
                f"Prefer: {approved['bank']['target']['transferable_decision']}"
            )
            self.assertEqual(
                self.model.seen[index * 2],
                tuple(self.tokenizer.encode(applicability_text, False)),
            )
            self.assertEqual(
                self.model.seen[index * 2 + 1],
                tuple(self.tokenizer.encode(dynamic_text, False)),
            )
            dynamic_tokens = set(self.model.seen[index * 2 + 1])
            verifier_sentinel = self.tokenizer.encode(
                f"VERIFIER_{('alpha', 'beta')[index]}", False
            )[0]
            avoid_sentinel = self.tokenizer.encode(
                f"AVOID_{('alpha', 'beta')[index]}", False
            )[0]
            self.assertNotIn(verifier_sentinel, dynamic_tokens)
            self.assertNotIn(avoid_sentinel, dynamic_tokens)
            entry = compiled.manifest["records"][index]
            self.assertEqual(
                entry["applicability_key_text_sha256"],
                text_sha256(applicability_text),
            )
            self.assertEqual(entry["dynamic_key_source"], V35_DYNAMIC_KEY_SOURCE)
            self.assertTrue(entry["applicability_embedding_exact_reproduction"])
            self.assertEqual(entry["kv_valid_slot_count"], record.token_count)

        _, manifest_path = compiled.save(self.root / "dual")
        loader = DualRetrievalKeyBankLoader(
            manifest_path=manifest_path,
            expected_reasoner_name="reasoner",
            expected_input_hashes=self.provenance,
        )
        self.assertTrue(torch.equal(
            loader.applicability_embeddings,
            self.old_loader.embeddings,
        ))
        self.assertEqual(
            loader.manifest["schema_version"], V35_DUAL_KEY_BANK_SCHEMA
        )
        self.assertEqual(
            loader.manifest["record_order_sha256"],
            canonical_json_sha256([record.memory_id for record in self.records]),
        )

    def test_source_record_provenance_mismatch_fails_closed(self) -> None:
        tampered = [copy.deepcopy(value) for value in self.approved]
        tampered[0]["bank"]["target"]["transferable_decision"] = (
            "Use a wholly different transferable decision."
        )
        with self.assertRaisesRegex(ValueError, "source provenance mismatch"):
            self._compiler().compile(
                records=self.records,
                approved_records=tampered,
                verified_experiences=self.verified,
                applicability_key_bank=self.old_loader,
                side_kv_manifest=self.side_manifest,
                split_manifest=self.split_manifest,
                artifact_provenance=self.provenance,
            )
    def test_duplicate_or_unreproduced_sources_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate experience_id"):
            self._compiler().compile(
                records=self.records,
                approved_records=(*self.approved, self.approved[0]),
                verified_experiences=self.verified,
                applicability_key_bank=self.old_loader,
                side_kv_manifest=self.side_manifest,
                split_manifest=self.split_manifest,
                artifact_provenance=self.provenance,
            )
        self.old_loader.embeddings[0] = torch.nn.functional.normalize(
            torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0
        )
        with self.assertRaisesRegex(ValueError, "reproduction failed"):
            self._compiler().compile(
                records=self.records,
                approved_records=self.approved,
                verified_experiences=self.verified,
                applicability_key_bank=self.old_loader,
                side_kv_manifest=self.side_manifest,
                split_manifest=self.split_manifest,
                artifact_provenance=self.provenance,
            )

    def test_memory_records_may_be_strict_subset_of_phase1_sources(self) -> None:
        extra_valid = _approved_record("gamma", 2)
        extra_valid["source"]["split_manifest_sha256"] = self.split_manifest[
            "manifest_sha256"
        ]
        extra_rejected = _approved_record("delta", 99)
        extra_rejected["ai_review_gate"]["route"] = "ai_rejected"
        approved = (*self.approved, extra_valid, extra_rejected)
        verified = (
            *self.verified,
            _verified_experience(extra_valid),
            _verified_experience(extra_rejected),
        )
        compiled = self._compiler().compile(
            records=self.records,
            approved_records=approved,
            verified_experiences=verified,
            applicability_key_bank=self.old_loader,
            side_kv_manifest=self.side_manifest,
            split_manifest=self.split_manifest,
            artifact_provenance=self.provenance,
        )
        audit = compiled.manifest["source_join"]
        self.assertEqual(audit["approved_input_count"], 4)
        self.assertEqual(audit["verified_input_count"], 4)
        self.assertEqual(audit["validated_source_count"], 3)
        self.assertEqual(audit["selector_rejected_source_count"], 1)
        self.assertEqual(audit["selected_memory_source_count"], 2)
        self.assertEqual(audit["unselected_valid_source_count"], 1)

    def test_real_baseline_shape_allows_192_sources_and_161_memories(self) -> None:
        approved = tuple(
            _approved_record(_alphabetic_suffix(index), index)
            for index in range(192)
        )
        split_manifest = _split_manifest(approved)
        verified = tuple(_verified_experience(value) for value in approved)
        selected_build = self.builder.build(approved[:161], verified)
        self.assertEqual(len(selected_build.records), 161)
        joined, audit = self._compiler()._join_sources(
            records=selected_build.records,
            approved_records=approved,
            verified_experiences=verified,
            split_manifest=split_manifest,
        )
        self.assertEqual(len(joined), 161)
        self.assertEqual(audit["approved_input_count"], 192)
        self.assertEqual(audit["validated_source_count"], 192)
        self.assertEqual(audit["selected_memory_source_count"], 161)
        self.assertEqual(audit["unselected_valid_source_count"], 31)

    def test_split_manifest_and_source_membership_tampering_fail_closed(self) -> None:
        tampered_manifest = copy.deepcopy(self.split_manifest)
        tampered_manifest["overlap_check"]["passed"] = False
        tampered_manifest["manifest_sha256"] = canonical_json_sha256({
            key: value
            for key, value in tampered_manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        })
        tampered_provenance = {
            **self.provenance,
            "split_manifest_logical_sha256": tampered_manifest[
                "manifest_sha256"
            ],
        }
        with self.assertRaisesRegex(ValueError, "overlap audit"):
            self._compiler().compile(
                records=self.records,
                approved_records=self.approved,
                verified_experiences=self.verified,
                applicability_key_bank=self.old_loader,
                side_kv_manifest=self.side_manifest,
                split_manifest=tampered_manifest,
                artifact_provenance=tampered_provenance,
            )

        approved = [copy.deepcopy(value) for value in self.approved]
        verified = [copy.deepcopy(value) for value in self.verified]
        approved[0]["source"]["split_manifest_sha256"] = "wrong-logical-hash"
        verified[0]["source"]["split_manifest_sha256"] = "wrong-logical-hash"
        with self.assertRaisesRegex(ValueError, "split provenance differs"):
            self._compiler().compile(
                records=self.records,
                approved_records=approved,
                verified_experiences=verified,
                applicability_key_bank=self.old_loader,
                side_kv_manifest=self.side_manifest,
                split_manifest=self.split_manifest,
                artifact_provenance=self.provenance,
            )

    def test_dynamic_final_answer_boilerplate_is_rejected(self) -> None:
        from memgen.model.v3_5_retrieval import (
            validate_v35_dynamic_text_component,
        )

        for text in ("check the final answer", "FINAL-ANSWER", "boxed", r"\fbox"):
            with self.subTest(text=text), self.assertRaisesRegex(
                ValueError, "prohibited"
            ):
                validate_v35_dynamic_text_component(owner="fixture", text=text)

        approved = [copy.deepcopy(value) for value in self.approved]
        approved[0]["bank"]["target"]["transferable_decision"] = (
            "Check the final answer before proceeding."
        )
        records = list(self.records)
        records[0] = replace(
            records[0],
            source_record_sha256=canonical_json_sha256(approved[0]),
        )
        with self.assertRaisesRegex(ValueError, "prohibited"):
            self._compiler().compile(
                records=records,
                approved_records=approved,
                verified_experiences=self.verified,
                applicability_key_bank=self.old_loader,
                side_kv_manifest=self.side_manifest,
                split_manifest=self.split_manifest,
                artifact_provenance=self.provenance,
            )


@unittest.skipIf(torch is None, "Torch is required for V3.5 retrieval tests")
class V35QuestionEncoderTests(unittest.TestCase):
    def test_question_encoder_uses_stripped_unwrapped_text_and_suspends_memory(self) -> None:
        from memgen.model.v3_5_retrieval import QuestionOnlyEncoder

        class Controller:
            def __init__(self) -> None:
                self.calls = 0

            @contextmanager
            def suspend_memory(self):
                self.calls += 1
                yield

        tokenizer = FixtureTokenizer()
        model = FixtureModel().eval()
        controller = Controller()
        encoder = QuestionOnlyEncoder(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            controller=controller,
        )
        query = encoder.encode("  original question text  ")
        self.assertEqual(query.text, "original question text")
        self.assertEqual(tokenizer.calls[0], ("original question text", False))
        self.assertEqual(model.seen, [query.token_ids])
        self.assertEqual(controller.calls, 1)
        audit = query.to_dict()
        self.assertTrue(audit["side_kv_disabled"])
        self.assertFalse(audit["chat_wrapper_included"])
        self.assertFalse(audit["prompt_boilerplate_included"])
        self.assertEqual(audit["layer_number"], 24)
        self.assertEqual(audit["pooling"], "last_valid_token")
        self.assertAlmostEqual(float(query.embedding.norm().item()), 1.0)


@unittest.skipIf(torch is None, "Torch is required for V3.5 retrieval tests")
class V35ApplicabilityRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        from memgen.model.retrieval_keys import tensor_sha256
        from memgen.model.v3_5_retrieval import (
            CompiledDualRetrievalKeyBank,
            DualRetrievalKeyBankLoader,
            DualRetrievalKeyCompilerConfig,
            V35_APPLICABILITY_KEY_SOURCE,
            V35_DYNAMIC_KEY_SOURCE,
            v35_implementation_files_sha256,
        )

        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.memory_ids = ("mem-b", "mem-a", "mem-c")
        applicability = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        dynamic = torch.tensor([
            [0.8, 0.6],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        implementation_files = v35_implementation_files_sha256()
        provenance = {
            "memory_records_sha256": "records",
            "side_kv_manifest_sha256": "side",
            "e0_final_report_sha256": "e0",
            "v3_retrieval_key_manifest_sha256": "old-manifest",
            "v3_retrieval_key_tensor_sha256": "old-tensor",
            "v3_offline_report_sha256": "old-report",
            "phase1_approved_bank_sha256": "approved",
            "verified_experiences_sha256": "verified",
            "split_manifest_sha256": "split",
            "split_manifest_logical_sha256": "split-logical",
            "dataset_revision": "dataset-revision",
            "compiler_git_revision": "revision",
            "compiler_tracked_diff_sha256": "tracked-diff",
            "compiler_implementation_files_sha256": implementation_files,
            "compiler_implementation_set_sha256": canonical_json_sha256(
                implementation_files
            ),
        }
        entries = []
        for index, memory_id in enumerate(self.memory_ids):
            entries.append({
                "index": index,
                "memory_id": memory_id,
                "source_experience_id": f"source-{memory_id}",
                "payload_hash": f"payload-{memory_id}",
                "payload_token_count": 2,
                "kv_layer": 24,
                "kv_valid_slot_count": 2,
                "applicability_key_source": V35_APPLICABILITY_KEY_SOURCE,
                "applicability_key_text_sha256": f"text-{memory_id}",
                "applicability_key_token_count": 2,
                "applicability_key_token_ids_sha256": f"tokens-{memory_id}",
                "applicability_key_embedding_sha256": tensor_sha256(
                    applicability[index]
                ),
                "applicability_key_embedding_norm": 1.0,
                "reproduced_applicability_key_embedding_sha256": tensor_sha256(
                    applicability[index]
                ),
                "applicability_embedding_exact_reproduction": True,
                "dynamic_key_source": V35_DYNAMIC_KEY_SOURCE,
                "dynamic_key_text_sha256": f"dynamic-text-{memory_id}",
                "dynamic_key_token_count": 2,
                "dynamic_key_token_ids_sha256": f"dynamic-tokens-{memory_id}",
                "dynamic_key_embedding_sha256": tensor_sha256(dynamic[index]),
                "dynamic_key_embedding_norm": 1.0,
                "source_record_sha256": f"record-{memory_id}",
                "phase1_provenance_sha256": f"phase1-{memory_id}",
                "review_provenance_sha256": f"review-{memory_id}",
                "review_validation_profile": "fixture",
                "source_sample_id": f"sample-{memory_id}",
                "source_dataset_revision": "dataset-revision",
                "source_dataset_split": "train",
                "source_logical_split": "bank-source",
                "source_index": index,
                "source_question_sha256": f"question-{memory_id}",
                "source_split_manifest_sha256": "split-logical",
                "split_member_sha256": f"member-{memory_id}",
            })
        order_hash = canonical_json_sha256(list(self.memory_ids))
        manifest = {
            "schema_version": V35_DUAL_KEY_BANK_SCHEMA,
            "created_at": "2026-01-01T00:00:00+00:00",
            "reasoner": {
                "model_name": "reasoner",
                "model_revision": "model-revision",
                "tokenizer_revision": "tokenizer-revision",
                "attention_implementation": "sdpa",
            },
            "compiler": asdict(DualRetrievalKeyCompilerConfig()),
            "model_compute_dtype": "bfloat16",
            "sanitizer": asdict(
                MemorySanitizerConfig(forbid_numeric_literals=True)
            ),
            "input_artifacts": provenance,
            "record_count": 3,
            "record_order_sha256": order_hash,
            "ordered_memory_ids_sha256": order_hash,
            "embedding_shape": [3, 2],
            "embedding_dtype": "torch.float32",
            "tensor_names": {
                "applicability": "applicability_key_embeddings",
                "dynamic": "dynamic_key_embeddings",
            },
            "source_join": {
                "policy": "approved_verified_memory_one_to_one_fail_closed",
                "joined_record_count": 3,
                "dynamic_decision_path": "bank.target.transferable_decision",
                "approved_input_count": 3,
                "verified_input_count": 3,
                "validated_source_count": 3,
                "selector_rejected_source_count": 0,
                "selected_memory_source_count": 3,
                "unselected_valid_source_count": 0,
                "validated_source_ids_sha256": canonical_json_sha256(
                    [f"source-{memory_id}" for memory_id in sorted(self.memory_ids)]
                ),
                "selected_memory_source_ids_sha256": canonical_json_sha256(
                    [f"source-{memory_id}" for memory_id in self.memory_ids]
                ),
                "unselected_valid_source_ids_sha256": canonical_json_sha256([]),
                "selector_rejected_source_ids_sha256": canonical_json_sha256([]),
            },
            "phase1_split_audit": {
                "schema_version": SPLIT_MANIFEST_SCHEMA,
                "manifest_logical_sha256": "split-logical",
                "dataset_revision": "dataset-revision",
                "overlap_check_verified": True,
                "joined_bank_source_member_count": 3,
                "authenticated_valid_source_member_count": 3,
                "all_sources_match_authenticated_members": True,
            },
            "applicability_reproduction_audit": {
                "source_schema": "experience-memory-retrieval-key-bank-v1",
                "record_count": 3,
                "exact_reproduction_count": 3,
                "all_exact": True,
            },
            "records": entries,
        }
        compiled = CompiledDualRetrievalKeyBank(
            applicability_embeddings=applicability,
            dynamic_embeddings=dynamic,
            manifest=manifest,
        )
        _, manifest_path = compiled.save(root)
        self.manifest_path = manifest_path
        self.loader = DualRetrievalKeyBankLoader(
            manifest_path=manifest_path,
            expected_input_hashes=provenance,
        )
        self.records = tuple(
            SimpleNamespace(
                memory_id=memory_id,
                payload_hash=f"payload-{memory_id}",
                token_count=2,
                kv_layer=24,
            )
            for memory_id in self.memory_ids
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _encoder(embedding):
        from memgen.model.v3_5_retrieval import QuestionOnlyQuery

        class Encoder:
            def encode(self, question):
                return QuestionOnlyQuery(
                    text=question.strip(),
                    token_ids=(11, 12),
                    embedding=torch.tensor(embedding, dtype=torch.float32),
                )

        return Encoder()

    def _retriever(
        self, static_embedding=(1.0, 0.0), floor=0.5, profile=None
    ):
        from memgen.model.v3_5_retrieval import ApplicabilityAwareMemoryRetriever

        return ApplicabilityAwareMemoryRetriever(
            key_bank=self.loader,
            records=self.records,
            kv_valid_slot_counts={memory_id: 2 for memory_id in self.memory_ids},
            question_encoder=self._encoder(static_embedding),
            shortlist_k=2,
            applicability_score_floor=floor,
            dynamic_min_top1_top2_margin=0.1,
            profile=profile,
        )

    def test_static_shortlist_uses_memory_id_tie_break_and_is_stable(self) -> None:
        retriever = self._retriever()
        first = retriever.prepare_question(" question only ")
        second = retriever.prepare_question("question only")
        self.assertTrue(first.available)
        self.assertEqual(first.memory_ids, ("mem-a", "mem-b"))
        self.assertEqual(first.to_dict(), second.to_dict())
        trace = first.to_dict()
        self.assertEqual(trace["shortlist_memory_ids"], ["mem-a", "mem-b"])
        self.assertEqual(trace["query"]["static_question_token_ids"], [11, 12])
        self.assertEqual(
            [item["original_global_rank"] for item in trace["post_floor_shortlist"]],
            [1, 2],
        )
        self.assertEqual(trace["stable_tie_break"], "memory_id_ascending")

    def test_dynamic_exact_rerank_is_restricted_to_shortlist(self) -> None:
        retriever = self._retriever()
        static = retriever.prepare_question("question")
        decision = retriever.retrieve(
            query_embedding=torch.tensor([1.0, 0.0]),
            query_token_ids=(100, 101, 102),
            prompt_token_count=2,
            static_context=static,
        )
        self.assertTrue(decision.selected)
        self.assertEqual(decision.matched_memory.memory_id, "mem-b")
        self.assertNotIn("mem-c", [hit["memory_id"] for hit in decision.hits])
        self.assertEqual(
            decision.query["dynamic_search_candidate_count"], 2
        )
        self.assertEqual(decision.query["partial_cot_token_count"], 1)
        self.assertEqual(decision.query["query_embedding_token_id"], 102)
        self.assertEqual(decision.query["query_token_ids"], [100, 101, 102])
        self.assertTrue(decision.query["side_kv_disabled"])
        self.assertTrue(decision.query["joint_admission_passed"])
        self.assertTrue(decision.query["selected_memory_kv_metadata_aligned"])
        self.assertEqual(decision.query["decision_reason"], "selected")

    def test_unit_query_bits_score_audit_and_saved_sidecar_are_identical(self) -> None:
        from safetensors.torch import load_file, save_file

        from memgen.model.retrieval_keys import tensor_sha256
        from memgen.model.v3_5_retrieval import (
            canonicalize_v35_query_embedding,
        )

        # This is within the frozen unit-norm tolerance, but normalizing it a
        # second time changes float32 bits.  It reproduces the cross-layer bug
        # between FullPrefixQueryEncoder, retriever scoring, and the sidecar.
        query = torch.tensor([0.6000024, 0.8000032], dtype=torch.float32)
        self.assertAlmostEqual(float(query.norm().item()), 1.0, delta=1e-5)
        self.assertFalse(torch.equal(
            torch.nn.functional.normalize(query, dim=0), query
        ))
        canonical = canonicalize_v35_query_embedding(
            query, expected_width=2, owner="test"
        )
        self.assertTrue(torch.equal(canonical, query))

        retriever = self._retriever(static_embedding=(1.000004, 0.0))
        static = retriever.prepare_question("question")
        static_input = torch.tensor([1.000004, 0.0], dtype=torch.float32)
        self.assertEqual(
            static.query["static_question_embedding_sha256"],
            tensor_sha256(static_input),
        )
        decision = retriever.retrieve(
            query_embedding=query,
            query_token_ids=(100, 101, 102),
            prompt_token_count=2,
            static_context=static,
        )
        self.assertEqual(
            decision.query["query_embedding_sha256"], tensor_sha256(query)
        )
        shortlist_indices = [
            self.loader.index_by_id[memory_id]
            for memory_id in static.memory_ids
        ]
        replay_scores = torch.mv(
            self.loader.dynamic_embeddings[shortlist_indices], query
        )
        replay_by_id = {
            memory_id: float(replay_scores[index].item())
            for index, memory_id in enumerate(static.memory_ids)
        }
        for hit in decision.hits:
            self.assertEqual(
                hit["score"], replay_by_id[str(hit["memory_id"])]
            )

        sidecar_path = Path(self.temporary.name) / "query-sidecar.safetensors"
        save_file({"attempt_01": query.contiguous()}, str(sidecar_path))
        saved = load_file(str(sidecar_path), device="cpu")["attempt_01"]
        self.assertTrue(torch.equal(saved, query))
        self.assertEqual(
            tensor_sha256(saved), decision.query["query_embedding_sha256"]
        )

        non_unit = torch.tensor([3.0, 4.0], dtype=torch.float32)
        normalized_once = torch.nn.functional.normalize(non_unit, dim=0)
        non_unit_decision = retriever.retrieve(
            query_embedding=non_unit,
            query_token_ids=(100, 101, 102),
            prompt_token_count=2,
            static_context=static,
        )
        self.assertEqual(
            non_unit_decision.query["query_embedding_sha256"],
            tensor_sha256(normalized_once),
        )

    def test_non_calibration_profile_omits_raw_reproduction_token_ids(self) -> None:
        profile = SimpleNamespace(
            layer_number=24,
            query_context="question_plus_full_partial_cot",
            query_encoder_state="pure_prefix_reencode_side_kv_disabled",
            query_pooling="current_generated_token",
            query_normalization="l2",
            calibration_trace_only=False,
        )
        retriever = self._retriever(profile=profile)
        static = retriever.prepare_question("question")
        self.assertNotIn("static_question_token_ids", static.query)
        decision = retriever.retrieve(
            query_embedding=torch.tensor([1.0, 0.0]),
            query_token_ids=(100, 101, 102),
            prompt_token_count=2,
            static_context=static,
        )
        self.assertNotIn("query_token_ids", decision.query)

    def test_below_margin_and_insufficient_shortlist_have_explicit_reasons(self) -> None:
        retriever = self._retriever()
        static = retriever.prepare_question("question")
        abstained = retriever.retrieve(
            query_embedding=torch.tensor([1.0, 2.0]),
            query_token_ids=(100, 101, 102),
            prompt_token_count=2,
            static_context=static,
        )
        self.assertEqual(abstained.status, "below_dynamic_margin")
        self.assertFalse(abstained.selected)
        self.assertFalse(abstained.query["dynamic_margin_condition_passed"])
        unavailable_retriever = self._retriever(
            static_embedding=(0.0, 1.0), floor=0.5
        )
        unavailable = unavailable_retriever.prepare_question("question")
        self.assertFalse(unavailable.available)
        self.assertEqual(unavailable.unavailable_reason, "insufficient_shortlist")
        decision = unavailable_retriever.retrieve(
            query_embedding=torch.tensor([1.0, 0.0]),
            query_token_ids=(100, 101, 102),
            prompt_token_count=2,
            static_context=unavailable,
        )
        self.assertEqual(decision.status, "insufficient_shortlist")
        self.assertFalse(decision.selected)

    def test_negative_dynamic_top1_is_selected_without_an_absolute_score_gate(self) -> None:
        retriever = self._retriever()
        static = retriever.prepare_question("question")
        decision = retriever.retrieve(
            query_embedding=torch.tensor([0.0, -1.0]),
            query_token_ids=(100, 101, 102),
            prompt_token_count=2,
            static_context=static,
        )
        self.assertTrue(decision.selected)
        self.assertEqual(decision.matched_memory.memory_id, "mem-b")
        self.assertLess(decision.matched_memory.retrieval_score, 0.0)
        self.assertAlmostEqual(
            decision.matched_memory.retrieval_score,
            decision.hits[0]["score"],
        )
        self.assertEqual(decision.query["decision_reason"], "selected")

    def test_loader_and_side_metadata_fail_closed(self) -> None:
        from memgen.model.v3_5_retrieval import (
            ApplicabilityAwareMemoryRetriever,
            DualRetrievalKeyBankLoader,
        )

        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        value["records"][0]["payload_hash"] = "tampered"
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        value = copy.deepcopy(self.loader.manifest)
        value["source_join"]["policy"] = "unchecked_join"
        value["manifest_sha256"] = canonical_json_sha256({
            key: item for key, item in value.items() if key != "manifest_sha256"
        })
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source-join contract drifted"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        value = copy.deepcopy(self.loader.manifest)
        value["source_join"]["unselected_valid_source_count"] = 1
        value["manifest_sha256"] = canonical_json_sha256({
            key: item for key, item in value.items() if key != "manifest_sha256"
        })
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source-join contract drifted"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        value = copy.deepcopy(self.loader.manifest)
        value["records"][0]["review_validation_profile"] = ""
        value["manifest_sha256"] = canonical_json_sha256({
            key: item for key, item in value.items() if key != "manifest_sha256"
        })
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "record metadata is incomplete"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        value = copy.deepcopy(self.loader.manifest)
        value["model_compute_dtype"] = "float32"
        value["manifest_sha256"] = canonical_json_sha256({
            key: item for key, item in value.items() if key != "manifest_sha256"
        })
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "compute dtype drifted"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        value = copy.deepcopy(self.loader.manifest)
        value["input_artifacts"][
            "compiler_implementation_set_sha256"
        ] = "tampered-implementation-set"
        value["manifest_sha256"] = canonical_json_sha256({
            key: item for key, item in value.items() if key != "manifest_sha256"
        })
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "implementation identity drifted"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        value = copy.deepcopy(self.loader.manifest)
        value["tensor_artifact"]["path"] = "../escaped.safetensors"
        value["manifest_sha256"] = canonical_json_sha256({
            key: item for key, item in value.items() if key != "manifest_sha256"
        })
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tensor path is unsafe"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        value = copy.deepcopy(self.loader.manifest)
        value["tensor_artifact"]["path"] = str(
            (self.manifest_path.parent / "absolute.safetensors").resolve()
        )
        value["manifest_sha256"] = canonical_json_sha256({
            key: item for key, item in value.items() if key != "manifest_sha256"
        })
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tensor path is unsafe"):
            DualRetrievalKeyBankLoader(manifest_path=self.manifest_path)

        with self.assertRaisesRegex(ValueError, "side-KV metadata differs"):
            ApplicabilityAwareMemoryRetriever(
                key_bank=self.loader,
                records=self.records,
                kv_valid_slot_counts={
                    "mem-a": 2,
                    "mem-b": 1,
                    "mem-c": 2,
                },
                question_encoder=self._encoder((1.0, 0.0)),
                shortlist_k=2,
                applicability_score_floor=0.5,
                dynamic_min_top1_top2_margin=0.1,
            )


class V35DualCompilerScriptTests(unittest.TestCase):
    def test_compiler_imports_repository_text_hash_helper(self) -> None:
        tree = ast.parse(COMPILE_SCRIPT.read_text(encoding="utf-8"))
        phase1_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "memgen.experience.phase1"
            for alias in node.names
        }
        self.assertIn("text_sha256", phase1_imports)

    def test_cli_exposes_all_authenticated_inputs_and_separate_calibration_output(self) -> None:
        result = subprocess.run(
            [sys.executable, str(COMPILE_SCRIPT), "--help"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        for flag in (
            "--memory-records",
            "--side-kv-manifest",
            "--e0-final-report",
            "--approved-bank",
            "--verified-experiences",
            "--v3-retrieval-key-manifest",
            "--v3-offline-report",
            "--split-manifest",
            "--output-dir",
            "--applicability-calibration-output",
        ):
            self.assertIn(flag, result.stdout)

    def test_cli_requires_split_manifest_before_loading_gpu_dependencies(self) -> None:
        arguments = [
            sys.executable,
            str(COMPILE_SCRIPT),
            "--memory-records", "records.jsonl",
            "--side-kv-manifest", "side.json",
            "--e0-final-report", "e0.json",
            "--approved-bank", "approved.jsonl",
            "--verified-experiences", "verified.jsonl",
            "--v3-retrieval-key-manifest", "keys.json",
            "--v3-offline-report", "v3.json",
            "--output-dir", "output",
        ]
        result = subprocess.run(
            arguments,
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--split-manifest", result.stderr)


if __name__ == "__main__":
    unittest.main()
