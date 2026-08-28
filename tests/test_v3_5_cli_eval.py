from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from memgen.experience.phase1 import canonical_json_sha256
from scripts import evaluate_v3_experience_memory as evaluate
from scripts import run_online_experience_memory_v3 as online


def versioned_args(**overrides):
    values = {
        "system_version": "v3.5",
        "logical_split": "dev-test",
        "query_pooling": None,
        "dual_key_manifest": Path("dual.json"),
        "applicability_calibration": Path("applicability.json"),
        "selector_calibration": Path("selector.json"),
        "calibration_trace_only": False,
        "save_query_embeddings": False,
        "retrieval_embedding_transform": "none",
        "dtype": "bfloat16",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _Attempt:
    def __init__(self, value):
        self.value = value
        self.outcome = value["outcome"]
        self.retrieval_decision = SimpleNamespace(
            query=value["retrieval_decision"]["query"]
        )

    def to_dict(self):
        return dict(self.value)


class _Result:
    def __init__(self):
        selected = {
            "attempt_number": 1,
            "generated_observation_index": 2,
            "affects_generated_token_index": 3,
            "outcome": "activated",
            "selected_memory_id": "memory-a",
            "terminal_abstain": False,
            "retrieval_decision": {
                "query": {
                    "query_token_count": 8,
                    "prompt_token_count": 5,
                    "partial_cot_token_count": 3,
                    "encoded_full_prefix_token_count": 8,
                    "context": "question_plus_full_partial_cot",
                    "encoder_state": "pure_prefix_reencode_side_kv_disabled",
                    "side_kv_disabled": True,
                    "selected_memory_kv_metadata_aligned": True,
                    "dynamic_search_restricted_to_static_shortlist": True,
                    "static_shortlist_ids": ["memory-a", "memory-b"],
                },
            },
        }
        terminal = {
            "attempt_number": 2,
            "generated_observation_index": 3,
            "affects_generated_token_index": 4,
            "outcome": "abstained",
            "selected_memory_id": None,
            "terminal_abstain": True,
            "memory_cleared_on_abstain": True,
            "cleared_memory_id": "memory-a",
            "active_memory_id_after": None,
            "actual_path_after_abstain": "native",
            "actual_path_memory_id_after": None,
            "deactivation_forward_seconds": 0.01,
            "deactivation_first_step_logits_kl": 0.001,
            "deactivation_first_step_top1_changed": False,
            "deactivation_baseline_first_token_id": 7,
            "deactivation_native_first_token_id": 7,
            "clear_affects_generated_token_index": 4,
            "retrieval_decision": {
                "query": {
                    "query_token_count": 9,
                    "prompt_token_count": 5,
                    "partial_cot_token_count": 4,
                    "encoded_full_prefix_token_count": 9,
                    "context": "question_plus_full_partial_cot",
                    "encoder_state": "pure_prefix_reencode_side_kv_disabled",
                    "side_kv_disabled": True,
                    "selected_memory_kv_metadata_aligned": True,
                    "dynamic_search_restricted_to_static_shortlist": True,
                    "static_shortlist_ids": ["memory-a", "memory-b"],
                },
            },
        }
        self.retrieval_attempts = (_Attempt(selected), _Attempt(terminal))
        self.attention_traces = ()
        self.boundary_traces = ()
        self.retrieval_attempt_count = 2
        self.rearm_count = 1
        self.replacement_count = 0
        self.duplicate_count = 0
        self.native_gate_observation_count = 2
        self.memory_conditioned_gate_observation_count = 1
        self.final_gate_state = "EXHAUSTED"
        self.final_memory_id = None
        self.answer_marker_seen = False
        self.static_selector_trace = {
            "query": {"side_kv_disabled": True},
            "post_floor_shortlist": [
                {"memory_id": "memory-a", "static_score": 0.7},
                {"memory_id": "memory-b", "static_score": 0.6},
            ],
            "static_selector_unavailable": False,
            "unavailable_reason": None,
            "shortlist_fixed_for_generation": True,
        }

    def to_dict(self):
        return {
            "static_selector_trace": self.static_selector_trace,
            "retrieval_attempts": [
                attempt.to_dict() for attempt in self.retrieval_attempts
            ],
            "summary": {
                "terminal_abstain_count": 1,
                "clear_on_terminal_abstain_count": 1,
                "no_rearm_after_terminal_abstain": True,
                "two_low_rearm_respected": True,
                "second_low_rearms_without_trigger": True,
                "stale_memory_attention_after_terminal_clear_count": 0,
                "terminal_clear_attention_safe": True,
            },
        }


class V35CliEvaluationTests(unittest.TestCase):
    def test_v35_uses_distinct_resume_schemas(self) -> None:
        self.assertEqual(
            evaluate.evaluation_schemas("v3.5"),
            (
                evaluate.V35_EVAL_PROFILE_SCHEMA,
                evaluate.V35_EVAL_ROW_SCHEMA,
                evaluate.V35_EVAL_REPORT_SCHEMA,
            ),
        )
        self.assertEqual(
            evaluate.evaluation_schemas("v3.4"),
            (
                evaluate.V3_EVAL_PROFILE_SCHEMA,
                evaluate.V3_EVAL_ROW_SCHEMA,
                evaluate.V3_EVAL_REPORT_SCHEMA,
            ),
        )

    def test_v35_final_requires_all_three_selector_inputs(self) -> None:
        for missing in (
            "dual_key_manifest",
            "applicability_calibration",
            "selector_calibration",
        ):
            with self.subTest(missing=missing):
                args = versioned_args(**{missing: None})
                with self.assertRaises(ValueError):
                    evaluate._resolve_and_validate_versioned_args(args)

    def test_v35_requires_bfloat16_in_both_entry_points(self) -> None:
        for dtype in ("float16", "float32"):
            with self.subTest(entry_point="evaluate", dtype=dtype):
                with self.assertRaisesRegex(ValueError, "dtype bfloat16"):
                    evaluate._resolve_and_validate_versioned_args(
                        versioned_args(dtype=dtype)
                    )
            with self.subTest(entry_point="online", dtype=dtype):
                with self.assertRaisesRegex(ValueError, "dtype bfloat16"):
                    online._resolve_and_validate_versioned_args(
                        versioned_args(dtype=dtype)
                    )

        for resolver in (
            evaluate._resolve_and_validate_versioned_args,
            online._resolve_and_validate_versioned_args,
        ):
            missing_dtype = versioned_args()
            del missing_dtype.dtype
            with self.subTest(entry_point=resolver.__module__, dtype="missing"):
                with self.assertRaisesRegex(ValueError, "dtype bfloat16"):
                    resolver(missing_dtype)

        legacy_evaluate = versioned_args(
            system_version="v3.4",
            dtype="float16",
            dual_key_manifest=None,
            applicability_calibration=None,
            selector_calibration=None,
        )
        evaluate._resolve_and_validate_versioned_args(legacy_evaluate)
        legacy_online = versioned_args(
            system_version="v3",
            dtype="float32",
            dual_key_manifest=None,
            applicability_calibration=None,
            selector_calibration=None,
        )
        online._resolve_and_validate_versioned_args(legacy_online)

    def test_v35_trace_only_is_explicit_answer_blind_calibration_profile(self) -> None:
        args = versioned_args(
            logical_split="calibration-val",
            selector_calibration=None,
            calibration_trace_only=True,
            save_query_embeddings=True,
        )
        evaluate._resolve_and_validate_versioned_args(args)
        self.assertEqual(args.query_pooling, "current_generated_token")

        with self.assertRaisesRegex(ValueError, "calibration-val"):
            evaluate._resolve_and_validate_versioned_args(
                versioned_args(
                    selector_calibration=None,
                    calibration_trace_only=True,
                    save_query_embeddings=True,
                    logical_split="dev-test",
                )
            )
        with self.assertRaisesRegex(ValueError, "cannot use a final selector"):
            evaluate._resolve_and_validate_versioned_args(
                versioned_args(
                    calibration_trace_only=True,
                    logical_split="calibration-val",
                    save_query_embeddings=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "save-query-embeddings"):
            evaluate._resolve_and_validate_versioned_args(
                versioned_args(
                    selector_calibration=None,
                    calibration_trace_only=True,
                    logical_split="calibration-val",
                    save_query_embeddings=False,
                )
            )

    def test_v35_final_test_is_blocked(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate user authorization"):
            evaluate._resolve_and_validate_versioned_args(
                versioned_args(logical_split="final-test")
            )

    def test_legacy_cli_defaults_and_behavior_remain_available(self) -> None:
        args = versioned_args(
            system_version="v3.4",
            dual_key_manifest=None,
            applicability_calibration=None,
            selector_calibration=None,
        )
        evaluate._resolve_and_validate_versioned_args(args)
        self.assertEqual(args.query_pooling, "current_generated_token")

        v3 = versioned_args(
            system_version="v3",
            dual_key_manifest=None,
            applicability_calibration=None,
            selector_calibration=None,
        )
        online._resolve_and_validate_versioned_args(v3)
        self.assertEqual(v3.query_pooling, "last_valid_token")

    def test_v35_rejects_legacy_transform_and_legacy_only_artifact_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy retrieval transform"):
            evaluate._resolve_and_validate_versioned_args(
                versioned_args(
                    retrieval_embedding_transform="key_bank_centroid_center_l2"
                )
            )
        with self.assertRaisesRegex(ValueError, "requires --dual-key-manifest"):
            online._resolve_and_validate_versioned_args(
                versioned_args(dual_key_manifest=None)
            )

    def test_v35_dispatches_only_to_v35_selector_loader(self) -> None:
        applicability = {
            "calibration": {
                "shortlist_k": 4,
                "minimum_applicability_score": 0.25,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("dual.json", "applicability.json", "selector.json", "risk.pt"):
                (root / name).write_text(name, encoding="utf-8")
            args = versioned_args(
                dual_key_manifest=root / "dual.json",
                applicability_calibration=root / "applicability.json",
                selector_calibration=root / "selector.json",
            )
            args.risk_artifact = root / "risk.pt"
            with (
                mock.patch(
                    "memgen.experience.v3_5_selector.load_v35_applicability_calibration",
                    return_value=applicability,
                ),
                mock.patch(
                    "memgen.experience.v3_5_selector.load_v35_selector_calibration",
                    side_effect=ValueError(
                        "Unexpected V3.5 selector calibration schema/policy"
                    ),
                ) as v35_loader,
            ):
                with self.assertRaisesRegex(ValueError, "V3.5 selector"):
                    evaluate._load_v35_profile_and_artifacts(args)
            v35_loader.assert_called_once()

    def test_online_diagnostics_exposes_static_and_terminal_safety(self) -> None:
        diagnostics = evaluate.online_diagnostics(
            _Result(),
            system_version="v3.5",
            first_answer_marker_token_index=5,
        )
        self.assertEqual(diagnostics["terminal_abstain_count"], 1)
        self.assertEqual(diagnostics["clear_on_terminal_abstain_count"], 1)
        self.assertTrue(diagnostics["terminal_clear_attention_safe"])
        self.assertTrue(diagnostics["no_rearm_after_terminal_abstain"])
        self.assertTrue(diagnostics["terminal_abstain_actual_path_native"])
        self.assertTrue(diagnostics["terminal_clear_native_reforward_audited"])
        self.assertTrue(diagnostics["both_query_encodings_side_kv_disabled"])
        self.assertTrue(
            diagnostics["dynamic_search_restricted_to_static_shortlist"]
        )
        self.assertTrue(
            diagnostics["selected_memory_belongs_to_static_shortlist"]
        )
        self.assertTrue(diagnostics["selected_memory_kv_metadata_aligned"])
        self.assertEqual(diagnostics["selected_outside_static_shortlist_count"], 0)
        self.assertTrue(
            diagnostics["attempt_affects_index_contract_respected"]
        )
        self.assertEqual(
            [
                value["tokens_until_first_answer_marker"]
                for value in diagnostics["answer_marker_attempt_distances"]
            ],
            [2, 1],
        )
        self.assertEqual(
            diagnostics["late_attempt_within_32_tokens_count"], 2
        )

    def test_online_diagnostics_rejects_self_consistent_truncated_query(self) -> None:
        result = _Result()
        query = result.retrieval_attempts[0].value["retrieval_decision"]["query"]
        query["query_token_count"] = 7
        query["partial_cot_token_count"] = 2
        query["encoded_full_prefix_token_count"] = 7
        diagnostics = evaluate.online_diagnostics(
            result,
            system_version="v3.5",
            first_answer_marker_token_index=None,
        )
        self.assertFalse(diagnostics["query_context_is_full_prefix"])

    def test_trace_only_raw_tokens_reproduce_static_and_dynamic_hashes(self) -> None:
        static_ids = [9, 10]
        prompt_ids = [1, 2]
        completion_ids = [3, 4]
        first_query = prompt_ids + completion_ids[:1]
        second_query = prompt_ids + completion_ids[:2]
        runtime_trace = {
            "static_selector_trace": {
                "query": {
                    "static_question_token_ids": static_ids,
                    "static_question_token_count": len(static_ids),
                    "static_question_token_ids_sha256": canonical_json_sha256(
                        static_ids
                    ),
                    "static_question_text_sha256": evaluate.text_sha256(
                        "question"
                    ),
                    "static_question_embedding_sha256": "embedding",
                    "static_question_embedding_norm": 1.0,
                    "layer_number": 24,
                    "pooling": "last_valid_token",
                    "normalization": "l2",
                    "side_kv_disabled": True,
                    "chat_wrapper_included": False,
                    "prompt_boilerplate_included": False,
                    "add_special_tokens": False,
                }
            },
            "retrieval_attempts": [
                {
                    "generated_observation_index": 0,
                    "retrieval_decision": {
                        "query": {
                            "query_token_ids": first_query,
                            "query_token_count": len(first_query),
                            "prompt_token_count": len(prompt_ids),
                            "partial_cot_token_count": 1,
                            "encoded_full_prefix_token_count": len(first_query),
                            "query_token_ids_sha256": canonical_json_sha256(
                                first_query
                            ),
                            "context": "question_plus_full_partial_cot",
                            "encoder_state": (
                                "pure_prefix_reencode_side_kv_disabled"
                            ),
                            "pooling": "current_generated_token",
                            "normalization": "l2",
                            "side_kv_disabled": True,
                            "query_embedding_token_index": len(first_query) - 1,
                            "query_embedding_token_id": first_query[-1],
                            "query_embedding_causal_context_token_count": len(
                                first_query
                            ),
                        }
                    },
                },
                {
                    "generated_observation_index": 1,
                    "retrieval_decision": {
                        "query": {
                            "query_token_ids": second_query,
                            "query_token_count": len(second_query),
                            "prompt_token_count": len(prompt_ids),
                            "partial_cot_token_count": 2,
                            "encoded_full_prefix_token_count": len(second_query),
                            "query_token_ids_sha256": canonical_json_sha256(
                                second_query
                            ),
                            "context": "question_plus_full_partial_cot",
                            "encoder_state": (
                                "pure_prefix_reencode_side_kv_disabled"
                            ),
                            "pooling": "current_generated_token",
                            "normalization": "l2",
                            "side_kv_disabled": True,
                            "query_embedding_token_index": len(second_query) - 1,
                            "query_embedding_token_id": second_query[-1],
                            "query_embedding_causal_context_token_count": len(
                                second_query
                            ),
                        }
                    },
                },
            ],
        }
        tokenizer = SimpleNamespace(
            encode=lambda _text, add_special_tokens=False: static_ids
        )
        evaluate.validate_calibration_reproduction_trace(
            runtime_trace=runtime_trace,
            tokenizer=tokenizer,
            question=" question ",
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
        )
        runtime_trace["retrieval_attempts"][1]["retrieval_decision"]["query"][
            "query_token_ids"
        ] = second_query[:-1]
        with self.assertRaisesRegex(RuntimeError, "dynamic full-prefix"):
            evaluate.validate_calibration_reproduction_trace(
                runtime_trace=runtime_trace,
                tokenizer=tokenizer,
                question="question",
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
            )

    def test_v35_sidecar_preserves_exact_audit_bits_without_renormalizing(self) -> None:
        try:
            import torch
            from memgen.model.retrieval_keys import tensor_sha256
        except ImportError:
            self.skipTest("torch is unavailable")

        first_normalized = None
        second_normalized = None
        for values in (
            (0.1, 0.2, 0.3),
            (1.0, 2.0, 7.0, 11.0),
            (0.17, -0.29, 0.43, -0.61, 0.73),
        ):
            first = torch.nn.functional.normalize(
                torch.tensor(values, dtype=torch.float32), dim=0
            ).contiguous()
            second = torch.nn.functional.normalize(first, dim=0).contiguous()
            if tensor_sha256(first) != tensor_sha256(second):
                first_normalized = first
                second_normalized = second
                break
        self.assertIsNotNone(
            first_normalized,
            "test vectors unexpectedly made float32 L2 normalization idempotent",
        )
        assert first_normalized is not None
        assert second_normalized is not None
        audit_sha256 = tensor_sha256(first_normalized)
        runtime_trace = {
            "retrieval_attempts": [
                {
                    "retrieval_decision": {
                        "query": {"query_embedding_sha256": audit_sha256}
                    }
                }
            ]
        }
        self.assertNotEqual(audit_sha256, tensor_sha256(second_normalized))
        for helper in (
            evaluate.prepare_v35_query_sidecar_embeddings,
            online.prepare_v35_query_sidecar_embeddings,
        ):
            with self.subTest(helper=helper.__module__):
                prepared = helper(
                    query_embeddings=(first_normalized,),
                    runtime_trace=runtime_trace,
                )
                self.assertEqual(tensor_sha256(prepared[0]), audit_sha256)
                self.assertTrue(torch.equal(prepared[0], first_normalized))

        tampered_trace = {
            "retrieval_attempts": [
                {
                    "retrieval_decision": {
                        "query": {
                            "query_embedding_sha256": tensor_sha256(
                                second_normalized
                            )
                        }
                    }
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "exact audit"):
            evaluate.prepare_v35_query_sidecar_embeddings(
                query_embeddings=(first_normalized,),
                runtime_trace=tampered_trace,
            )

    def test_resume_rejects_legacy_row_schema_even_with_matching_profile_hash(self) -> None:
        profile_sha256 = "profile"
        row = {
            "schema_version": evaluate.V3_EVAL_ROW_SCHEMA,
            "profile_sha256": profile_sha256,
            "sample_id": "sample-a",
            "conditions": {
                "vanilla": {
                    "strict_correct": True,
                    "format_correct": True,
                    "generated_token_count": 2,
                },
                "v3": {
                    "strict_correct": True,
                    "format_correct": True,
                    "generated_token_count": 2,
                    "online_diagnostics": {},
                },
            },
        }
        row["row_sha256"] = canonical_json_sha256(row)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different run profile"):
                evaluate.load_existing_rows(
                    path=path,
                    profile_sha256=profile_sha256,
                    selected_ids={"sample-a"},
                    row_schema=evaluate.V35_EVAL_ROW_SCHEMA,
                )

    def test_trace_only_resume_authenticates_required_query_sidecars(self) -> None:
        profile_sha256 = "profile"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sidecar = root / "query_embeddings" / "sample-a.safetensors"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_bytes(b"authenticated-sidecar")
            row = {
                "schema_version": evaluate.V35_EVAL_ROW_SCHEMA,
                "profile_sha256": profile_sha256,
                "sample_id": "sample-a",
                "conditions": {
                    "vanilla": {
                        "strict_correct": True,
                        "format_correct": True,
                        "generated_token_count": 2,
                    },
                    "v3": {
                        "strict_correct": True,
                        "format_correct": True,
                        "generated_token_count": 2,
                        "online_diagnostics": {},
                        "runtime_trace": {
                            "retrieval_attempts": [{"attempt_number": 1}]
                        },
                        "query_embedding_sidecar": {
                            "path": str(sidecar.relative_to(root)),
                            "sha256": evaluate.file_sha256(sidecar),
                            "attempt_count": 1,
                            "representation": (
                                evaluate.V35_QUERY_SIDECAR_REPRESENTATION
                            ),
                        },
                    },
                },
            }
            row["row_sha256"] = canonical_json_sha256(row)
            results = root / "results.jsonl"
            results.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded, completed = evaluate.load_existing_rows(
                path=results,
                profile_sha256=profile_sha256,
                selected_ids={"sample-a"},
                row_schema=evaluate.V35_EVAL_ROW_SCHEMA,
                sidecar_root=root,
                require_v35_query_sidecars=True,
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(completed, {"sample-a"})

            sidecar.write_bytes(b"tampered-sidecar")
            with self.assertRaisesRegex(ValueError, "query sidecar is invalid"):
                evaluate.load_existing_rows(
                    path=results,
                    profile_sha256=profile_sha256,
                    selected_ids={"sample-a"},
                    row_schema=evaluate.V35_EVAL_ROW_SCHEMA,
                    sidecar_root=root,
                    require_v35_query_sidecars=True,
                )

    def test_numeric_correct_without_box_is_diagnostic_only(self) -> None:
        tokenizer = SimpleNamespace(
            decode=lambda _ids, skip_special_tokens=True: "Therefore the answer is 42"
        )
        scored = evaluate.score_condition(
            tokenizer=tokenizer,
            completion_token_ids=(1, 2),
            ground_truth="work\\boxed{42}",
            runtime_seconds=0.1,
        )
        self.assertFalse(scored["strict_correct"])
        self.assertFalse(scored["format_correct"])
        self.assertTrue(scored["numeric_correct_but_format_invalid"])
        self.assertTrue(scored["answer_marker_seen"])
        self.assertEqual(scored["first_answer_marker_token_index"], 0)

    def test_first_answer_marker_index_uses_decoded_token_prefixes(self) -> None:
        pieces = {
            1: "Reasoning ",
            2: "continues. ",
            3: "Final ",
            4: "answer is 42",
        }
        tokenizer = SimpleNamespace(
            decode=lambda ids, skip_special_tokens=True: "".join(
                pieces[int(token_id)] for token_id in ids
            )
        )
        self.assertEqual(
            evaluate.first_answer_marker_token_index(
                tokenizer=tokenizer,
                completion_token_ids=(1, 2, 3, 4),
            ),
            3,
        )
        self.assertIsNone(
            evaluate.first_answer_marker_token_index(
                tokenizer=tokenizer,
                completion_token_ids=(1, 2),
            )
        )

    def test_repository_state_hashes_full_v35_implementation_set(self) -> None:
        state = evaluate.repository_state()
        hashes = state["implementation_files_sha256"]
        for implementation in (
            "memgen/experience/v3_5_selector.py",
            "memgen/model/v3_5_retrieval.py",
            "scripts/compile_v3_5_dual_selector.py",
            "scripts/calibrate_v3_5_dynamic_selector.py",
            "scripts/analyze_v3_evaluation.py",
            "scripts/compare_v3_5_applicability_selector.py",
            "scripts/qualify_v3_5_dev.py",
            "scripts/run_online_experience_memory_v3.py",
            "scripts/evaluate_v3_experience_memory.py",
            "scripts/experiments/gsm8k/run_v3_5_applicability_selector_experiment.sh",
        ):
            self.assertIn(implementation, hashes)
        self.assertEqual(state["missing_implementation_files"], [])
        self.assertEqual(
            state["implementation_set_sha256"], canonical_json_sha256(hashes)
        )


if __name__ == "__main__":
    unittest.main()
