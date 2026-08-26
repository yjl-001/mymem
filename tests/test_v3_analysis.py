from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_v3_evaluation.py"


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("analyze_v3_evaluation", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_analysis_module()


def condition(*, token_ids, strict, formatting):
    return {
        "completion": "fixture",
        "completion_token_ids": list(token_ids),
        "completion_token_ids_sha256": canonical_json_sha256(list(token_ids)),
        "generated_token_count": len(token_ids),
        "strict_correct": strict,
        "format_correct": formatting,
        "strict_reward": float(strict),
        "scorer_version": "fixture",
        "runtime_seconds": 0.1,
    }


def attempt(number, *, boundary_index, memory_id, outcome):
    return {
        "attempt_number": number,
        "generated_boundary_index": boundary_index,
        "boundary_token_id": 9,
        "outcome": outcome,
        "previous_memory_id": None if number == 1 else "memory-a",
        "selected_memory_id": memory_id,
        "active_memory_id_after": memory_id,
        "retrieval_decision": {
            "schema_version": "embedding-memory-retrieval-decision-v1",
            "status": "selected",
            "query": {
                "method": "exact_cosine",
                "context": "question_plus_full_partial_cot",
                "encoder_state": "pure_prefix_reencode_side_kv_disabled",
                "pooling": "last_valid_token",
                "normalization": "l2",
                "query_token_count": 3 + boundary_index + 1,
                "prompt_token_count": 3,
                "partial_cot_token_count": boundary_index + 1,
                "query_token_ids_sha256": "query",
                "query_embedding_sha256": "embedding",
                "query_embedding_norm": 1.0,
                "top_k_requested": 2,
                "top1_score": 0.8 - number * 0.01,
                "top2_score": 0.4,
                "top1_top2_margin": 0.4 - number * 0.01,
            },
            "hits": [
                {"memory_id": memory_id, "score": 0.8 - number * 0.01},
                {"memory_id": "memory-z", "score": 0.4},
            ],
            "matched_memory": {"memory_id": memory_id},
        },
        "query_encoding_seconds": 0.01,
        "retrieval_seconds": 0.002,
        "memory_load_seconds": 0.001,
        "activation_forward_seconds": 0.003,
        "attempt_total_seconds": 0.02,
        "activation_first_step_logits_kl": 0.05 * number,
        "activation_first_step_top1_changed": True,
        "activation_baseline_first_token_id": 7,
    }


def boundary(index, *, action):
    return {
        "generated_boundary_index": index,
        "boundary_token_id": 9,
        "state_before": "ARMED" if action == "retrieval_attempt" else "DISARMED",
        "state_after": "DISARMED" if action == "retrieval_attempt" else "ARMED",
        "entropy": 2.0 if action == "retrieval_attempt" else 0.1,
        "high_entropy_threshold": 1.0,
        "low_entropy_threshold": 0.5,
        "persistence_risk_score": -10.0,
        "persistence_risk_threshold": 0.0,
        "risk_role": "diagnostic_only",
        "action": action,
        "retrieval_attempt_count_before": 0,
        "retrieval_attempt_count_after": int(action == "retrieval_attempt"),
        "active_memory_id_before": None,
        "active_memory_id_after": "memory-a",
    }


def attention(memory_id, index):
    return {
        "generated_input_index": index,
        "processed_prefix_token_count": 4 + index,
        "memory_id": memory_id,
        "layer_number": 24,
        "query_length": 1,
        "native_key_length": 4 + index,
        "memory_slot_count": 2,
        "memory_attention_mass": 0.1,
        "native_attention_mass": 0.9,
        "canonical_rope_score_relative_error": None,
        "memory_mass_by_query_head": [0.1],
        "memory_mass_by_kv_group": [0.1],
        "memory_score_normalization": "log_valid_slots",
        "memory_score_bias": 2.302585092994046,
        "schema_version": "side-kv-attention-trace-v3",
    }


def make_row(
    *,
    profile_sha256,
    sample_id,
    vanilla_ids,
    v3_ids,
    vanilla_strict,
    v3_strict,
    attempts,
    boundaries,
    attentions,
    final_memory_id,
    cache_parity=None,
):
    vanilla_value = condition(
        token_ids=vanilla_ids,
        strict=vanilla_strict,
        formatting=vanilla_strict,
    )
    v3_value = condition(
        token_ids=v3_ids,
        strict=v3_strict,
        formatting=v3_strict,
    )
    outcomes = [value["outcome"] for value in attempts]
    diagnostics = {
        "retrieval_attempt_count": len(attempts),
        "rearm_count": sum(value["action"] == "rearmed" for value in boundaries),
        "activation_count": outcomes.count("activated"),
        "replacement_count": outcomes.count("replaced"),
        "duplicate_count": outcomes.count("duplicate"),
        "abstain_count": outcomes.count("abstained"),
        "memory_attention_step_count": len(attentions),
        "attempt_budget_respected": True,
        "query_context_is_full_prefix": True,
        "native_cache_excludes_memory_slots": True,
        "memory_attention_mass_finite_and_positive": True,
    }
    runtime = {
        "schema_version": "experience-memory-v3-generation-result-v1",
        "completion_token_ids": list(v3_ids),
        "completion_token_ids_sha256": canonical_json_sha256(list(v3_ids)),
        "generated_token_count": len(v3_ids),
        "boundary_traces": boundaries,
        "retrieval_attempts": attempts,
        "memory_transitions": [],
        "memory_activation_spans": [],
        "attention_traces": attentions,
        "final_gate_state": "CLOSED",
        "final_memory_id": final_memory_id,
        "answer_marker_seen": False,
        "summary": diagnostics,
    }
    row = {
        "schema_version": "experience-memory-v3-evaluation-row-v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "profile_sha256": profile_sha256,
        "sample_id": sample_id,
        "logical_split": "final-test",
        "dataset_split": "test",
        "source_index": int(sample_id.split("-")[-1]),
        "question_sha256": "question",
        "answer_sha256": "answer",
        "prompt_token_count": 3,
        "prompt_token_ids_sha256": "prompt",
        "cache_parity": cache_parity,
        "conditions": {
            "vanilla": vanilla_value,
            "v3": v3_value
            | {
                "online_diagnostics": diagnostics,
                "runtime_trace": runtime,
                "query_embedding_sidecar": None,
            },
        },
        "paired_generated_token_delta_v3_minus_vanilla": (
            len(v3_ids) - len(vanilla_ids)
        ),
        "sample_runtime_seconds": 0.2,
    }
    row["row_sha256"] = canonical_json_sha256({
        key: value for key, value in row.items() if key != "created_at"
    })
    return row


class V3AnalysisTests(unittest.TestCase):
    def test_exact_mcnemar(self) -> None:
        self.assertEqual(ANALYSIS.exact_mcnemar_two_sided(0, 2), 0.5)
        self.assertIsNone(ANALYSIS.exact_mcnemar_two_sided(0, 0))

    def test_end_to_end_analysis(self) -> None:
        profile = {
            "schema_version": "experience-memory-v3-evaluation-profile-v1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "repository": {
                "git_revision": "revision",
                "tracked_diff_sha256": "diff",
                "implementation_set_sha256": "implementation",
            },
            "evaluation_interpretation": (
                "reused_official_test_descriptive_evaluation"
            ),
            "independent_final_confirmation": False,
            "logical_split": "final-test",
            "dataset_revision": "dataset",
            "selected_sample_count": 4,
            "system_profile": {"layer_number": 24},
            "hysteresis_gate": {"risk_role": "diagnostic_only"},
            "generation": {"max_new_tokens": 4},
            "inputs": {"results": "fixture"},
        }
        profile["profile_sha256"] = ANALYSIS.evaluation_profile_sha256(profile)
        rows = [
            make_row(
                profile_sha256=profile["profile_sha256"],
                sample_id="sample-0",
                vanilla_ids=(1, 0),
                v3_ids=(1, 0),
                vanilla_strict=False,
                v3_strict=False,
                attempts=[],
                boundaries=[],
                attentions=[],
                final_memory_id=None,
                cache_parity={"exact_match": True},
            ),
            make_row(
                profile_sha256=profile["profile_sha256"],
                sample_id="sample-1",
                vanilla_ids=(1, 0),
                v3_ids=(2, 0),
                vanilla_strict=False,
                v3_strict=True,
                attempts=[
                    attempt(
                        1,
                        boundary_index=0,
                        memory_id="memory-a",
                        outcome="activated",
                    )
                ],
                boundaries=[boundary(0, action="retrieval_attempt")],
                attentions=[attention("memory-a", 0)],
                final_memory_id="memory-a",
            ),
            make_row(
                profile_sha256=profile["profile_sha256"],
                sample_id="sample-2",
                vanilla_ids=(1, 2, 0),
                v3_ids=(2, 2, 2, 0),
                vanilla_strict=True,
                v3_strict=False,
                attempts=[
                    attempt(
                        1,
                        boundary_index=0,
                        memory_id="memory-a",
                        outcome="activated",
                    ),
                    attempt(
                        2,
                        boundary_index=2,
                        memory_id="memory-b",
                        outcome="replaced",
                    ),
                ],
                boundaries=[
                    boundary(0, action="retrieval_attempt"),
                    boundary(1, action="rearmed"),
                    boundary(2, action="retrieval_attempt"),
                ],
                attentions=[
                    attention("memory-a", 0),
                    attention("memory-b", 1),
                ],
                final_memory_id="memory-b",
            ),
            make_row(
                profile_sha256=profile["profile_sha256"],
                sample_id="sample-3",
                vanilla_ids=(3, 0),
                v3_ids=(3, 0),
                vanilla_strict=True,
                v3_strict=True,
                attempts=[],
                boundaries=[],
                attentions=[],
                final_memory_id=None,
            ),
        ]
        # A direct KL estimate can be microscopically negative because the
        # float32 softmax/log-softmax reduction is not an exact arithmetic
        # proof of non-negativity. It remains a valid finite diagnostic.
        rows[1]["conditions"]["v3"]["runtime_trace"]["retrieval_attempts"][0][
            "activation_first_step_logits_kl"
        ] = -1e-8
        rows[1]["row_sha256"] = canonical_json_sha256({
            key: value
            for key, value in rows[1].items()
            if key not in {"created_at", "row_sha256"}
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "run_profile.json"
            results_path = root / "results.jsonl"
            output_path = root / "analysis.json"
            profile_path.write_text(
                json.dumps(profile, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            results_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--results",
                    str(results_path),
                    "--run-profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--bootstrap-resamples",
                    "100",
                    "--top-k",
                    "2",
                    "--min-memory-samples",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("integrity=True", completed.stdout)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(report["integrity"]["passed"])
            strict = report["paired_analysis"]["overall"]["strict"]
            self.assertEqual(strict["net_correct_count_delta"], 0)
            self.assertEqual(
                strict["paired_table"]["v3_only_correct_improved"], 1
            )
            self.assertEqual(
                strict["paired_table"]["vanilla_only_correct_harmed"], 1
            )
            self.assertEqual(report["mechanism"]["retrieval_attempt_count"], 3)
            self.assertEqual(report["mechanism"]["replacement_count"], 1)
            self.assertEqual(report["zero_attempt_parity"]["mismatch_count"], 0)
            self.assertTrue(output_path.with_suffix(".md").is_file())


if __name__ == "__main__":
    unittest.main()
