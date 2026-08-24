#!/usr/bin/env python3
"""Run the one-shot E1C-S fixed-strength persistent side-KV diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1_staged import (
    E1B_MANIFEST_SCHEMA,
    E1C_MEMORY_SCORE_NORMALIZATION,
    E1C_RESULTS_SCHEMA,
    E1C_SUMMARY_SCHEMA,
    E1CS_MAX_MEAN_MEMORY_ATTENTION_MASS,
    E1CS_MEMORY_SCORE_BIAS,
    E1CS_MIN_MEAN_MEMORY_ATTENTION_MASS,
    E1CS_RESULTS_SCHEMA,
    E1CS_SUMMARY_SCHEMA,
    E1CT_RESULTS_SCHEMA,
    E1CT_SUMMARY_SCHEMA,
    E1BRetrievalAssignment,
    E1CSFixedStrengthDecision,
)
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from scripts.e1_staged_common import (
    PairedConditionComparison,
    PairedConditionDiagnostics,
    e1c_source_mechanism_valid,
    effect_is_negative,
    effect_is_positive,
    load_hashed_manifest,
    processed_solution,
    prompt_token_ids,
    score_completion,
    summarize_conditions,
    utc_now,
    validate_resolved_revisions,
    write_json,
)


SPLIT_PREFILL_PATH = "split-before-final-prompt-token-v1"
E1CS_STRENGTH_LABEL = "fixed-log10-memory-odds-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--e1c-results", type=Path, required=True)
    parser.add_argument("--e1c-run-report", type=Path, required=True)
    parser.add_argument("--e1c-summary", type=Path, required=True)
    parser.add_argument("--e1ct-results", type=Path, required=True)
    parser.add_argument("--e1ct-run-report", type=Path, required=True)
    parser.add_argument("--e1ct-summary", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(value) for value in iter_jsonl(path)]


def _frozen_condition(
    source: Mapping[str, Any],
    condition: str,
    *,
    artifact_role: str,
) -> dict[str, Any]:
    row = dict(source["conditions"][condition])
    if row.get("prefill_path") != SPLIT_PREFILL_PATH:
        raise ValueError(f"Frozen condition is not split-prefill: {condition}")
    row["source_artifact_role"] = artifact_role
    row["source_condition"] = condition
    return row


def _is_exact_e1ct_copy_of_e1c(
    e1ct_row: Mapping[str, Any], e1c_row: Mapping[str, Any]
) -> bool:
    candidate = dict(e1ct_row)
    candidate.pop("source_artifact_role", None)
    return canonical_json_sha256(candidate) == canonical_json_sha256(e1c_row)


@dataclass(frozen=True)
class FixedStrengthTraceAuditor:
    """Compile per-token traces into an auditable fixed-strength artifact."""

    normalization: str
    memory_score_bias: float

    def compact(
        self,
        traces: Sequence[Any],
        *,
        completion_length: int,
        prompt_length: int,
    ) -> dict[str, Any]:
        if not traces:
            raise RuntimeError("Fixed-strength generation returned no side-KV traces")
        native_lengths = [int(trace.native_key_length) for trace in traces]
        masses = [float(trace.memory_attention_mass) for trace in traces]
        expected_lengths = list(
            range(prompt_length, prompt_length + completion_length)
        )
        memory_ids = {str(trace.memory_id) for trace in traces}
        slot_counts = {int(trace.memory_slot_count) for trace in traces}
        normalizations = {
            str(trace.memory_score_normalization) for trace in traces
        }
        biases = {float(trace.memory_score_bias) for trace in traces}
        bias_is_expected = biases == {self.memory_score_bias}
        return {
            "strength_label": E1CS_STRENGTH_LABEL,
            "trace_count": len(traces),
            "expected_trace_count": completion_length,
            "memory_ids": sorted(memory_ids),
            "memory_slot_counts": sorted(slot_counts),
            "memory_score_normalizations": sorted(normalizations),
            "memory_score_biases": sorted(biases),
            "memory_odds_multiplier": math.exp(self.memory_score_bias),
            "native_key_lengths": native_lengths,
            "native_key_lengths_sha256": canonical_json_sha256(native_lengths),
            "memory_attention_masses": masses,
            "memory_attention_masses_sha256": canonical_json_sha256(masses),
            "mean_memory_attention_mass": sum(masses) / len(masses),
            "min_memory_attention_mass": min(masses),
            "max_memory_attention_mass": max(masses),
            "one_trace_per_generated_token": len(traces) == completion_length,
            "native_cache_length_matches_real_tokens": (
                native_lengths == expected_lengths
            ),
            "all_memory_attention_mass_finite_and_positive": all(
                math.isfinite(value) and value > 0.0 for value in masses
            ),
            "memory_id_constant": len(memory_ids) == 1,
            "memory_slot_count_constant": len(slot_counts) == 1,
            "normalization_constant": normalizations == {self.normalization},
            "memory_score_bias_constant": len(biases) == 1,
            "memory_score_bias_matches_preregistered_value": bias_is_expected,
        }


class FixedStrengthConditionRunner:
    """Execute one persistent side-KV arm under the frozen E1C-S strength."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        runtime: Any,
        loader: Any,
        controller: Any,
        layer_number: int,
        device: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.loader = loader
        self.controller = controller
        self.layer_number = layer_number
        self.device = device
        self.trace_auditor = FixedStrengthTraceAuditor(
            normalization=E1C_MEMORY_SCORE_NORMALIZATION,
            memory_score_bias=E1CS_MEMORY_SCORE_BIAS,
        )

    def run(
        self,
        *,
        choice: Any,
        prompt_token_ids_value: Sequence[int],
        baseline_first_token_id: int,
        ground_truth: str,
    ) -> dict[str, Any]:
        memory = self.loader.get(
            choice.memory_id,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        if (
            memory.payload_hash != choice.payload_hash
            or memory.valid_slot_count != choice.kv_valid_slot_count
            or memory.layer_number != self.layer_number
        ):
            raise ValueError(f"Side-KV metadata drift for {choice.memory_id}")
        started = time.perf_counter()
        generated = self.runtime.generate_prompt_with_persistent_memory(
            prompt_token_ids=prompt_token_ids_value,
            memory=memory,
            controller=self.controller,
        )
        elapsed = time.perf_counter() - started
        baseline_matches = (
            generated.baseline_first_token_id == baseline_first_token_id
        )
        trace = self.trace_auditor.compact(
            generated.attention_traces,
            completion_length=len(generated.completion_token_ids),
            prompt_length=len(prompt_token_ids_value),
        )
        trace.update({
            "first_step_logits_kl_baseline_to_memory": (
                generated.first_step_logits_kl
            ),
            "first_step_top1_changed": generated.first_step_top1_changed,
            "baseline_first_token_id": generated.baseline_first_token_id,
            "baseline_first_token_matches_split_no_memory": baseline_matches,
        })
        row = score_completion(
            tokenizer=self.tokenizer,
            completion_token_ids=generated.completion_token_ids,
            ground_truth=ground_truth,
            runtime_seconds=elapsed,
            prompt_token_count=len(prompt_token_ids_value),
            memory_ids=(choice.memory_id,),
            side_kv=trace,
        )
        row.update({
            "prefill_path": SPLIT_PREFILL_PATH,
            "strength_label": E1CS_STRENGTH_LABEL,
        })
        return row


def _mean_side_metric(
    records: Sequence[Mapping[str, Any]], condition: str, metric: str
) -> float:
    return sum(
        float(record["conditions"][condition]["side_kv"][metric])
        for record in records
    ) / len(records)


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime
    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader

    manifest = load_hashed_manifest(
        args.assignment_manifest, schema=E1B_MANIFEST_SCHEMA
    )
    assignments = tuple(
        E1BRetrievalAssignment.from_dict(value) for value in manifest["assignments"]
    )
    if len(assignments) <= 1 or any(not assignment.assigned for assignment in assignments):
        raise ValueError("E1C-S requires complete matched/shuffled assignments")

    artifact_hashes = {
        "assignment_manifest": file_sha256(args.assignment_manifest),
        "e1c_results": file_sha256(args.e1c_results),
        "e1c_run_report": file_sha256(args.e1c_run_report),
        "e1c_summary": file_sha256(args.e1c_summary),
        "e1ct_results": file_sha256(args.e1ct_results),
        "e1ct_run_report": file_sha256(args.e1ct_run_report),
        "e1ct_summary": file_sha256(args.e1ct_summary),
        "side_kv_manifest": file_sha256(args.side_kv_manifest),
        "split_manifest": file_sha256(args.split_manifest),
    }
    frozen_inputs = manifest["inputs"]
    if (
        artifact_hashes["side_kv_manifest"]
        != frozen_inputs["side_kv_manifest_sha256"]
        or artifact_hashes["split_manifest"]
        != frozen_inputs["split_manifest_sha256"]
    ):
        raise ValueError("E0/split artifacts differ from the frozen assignment")

    e1c_run = json.loads(args.e1c_run_report.read_text(encoding="utf-8"))
    if (
        e1c_run.get("schema_version") != "experience-memory-e1c-run-report-v2"
        or e1c_run.get("status") != "completed"
        or e1c_run.get("results", {}).get("sha256")
        != artifact_hashes["e1c_results"]
        or e1c_run.get("inputs", {}).get("assignment_manifest_sha256")
        != artifact_hashes["assignment_manifest"]
        or e1c_run.get("inputs", {}).get("side_kv_manifest_sha256")
        != artifact_hashes["side_kv_manifest"]
        or e1c_run.get("inputs", {}).get("split_manifest_sha256")
        != artifact_hashes["split_manifest"]
        or e1c_run.get("inputs", {}).get("memory_records_sha256")
        != frozen_inputs["memory_records_sha256"]
    ):
        raise ValueError("E1-C v3 artifacts do not match their run report")
    e1c_summary = json.loads(args.e1c_summary.read_text(encoding="utf-8"))
    if (
        e1c_summary.get("schema_version") != E1C_SUMMARY_SCHEMA
        or e1c_summary.get("sample_count") != len(assignments)
        or e1c_summary.get("component_diagnostic", {}).get("status") != "passed"
    ):
        raise ValueError("E1C-S requires a mechanism-valid E1-C v3 summary")
    e1c_records = _read_jsonl(args.e1c_results)
    if any(record.get("schema_version") != E1C_RESULTS_SCHEMA for record in e1c_records):
        raise ValueError("Unexpected E1-C results schema")
    e1c_by_sample = {str(record["sample_id"]): record for record in e1c_records}
    if len(e1c_records) != len(assignments) or set(e1c_by_sample) != {
        assignment.sample_id for assignment in assignments
    }:
        raise ValueError("E1-C source results and assignments have different samples")
    if not all(
        e1c_source_mechanism_valid(
            record,
            split_prefill_path=SPLIT_PREFILL_PATH,
            expected_normalization=E1C_MEMORY_SCORE_NORMALIZATION,
        )
        for record in e1c_records
    ):
        raise ValueError("E1C-S requires valid per-sample E1-C mechanism evidence")

    e1ct_run = json.loads(args.e1ct_run_report.read_text(encoding="utf-8"))
    if (
        e1ct_run.get("schema_version") != "experience-memory-e1ct-run-report-v1"
        or e1ct_run.get("status") != "completed"
        or e1ct_run.get("results", {}).get("sha256")
        != artifact_hashes["e1ct_results"]
        or e1ct_run.get("summary", {}).get("sha256")
        != artifact_hashes["e1ct_summary"]
        or e1ct_run.get("inputs", {}).get("assignment_manifest_sha256")
        != artifact_hashes["assignment_manifest"]
        or e1ct_run.get("inputs", {}).get("e1c_results_sha256")
        != artifact_hashes["e1c_results"]
        or e1ct_run.get("inputs", {}).get("e1c_run_report_sha256")
        != artifact_hashes["e1c_run_report"]
        or e1ct_run.get("inputs", {}).get("e1c_summary_sha256")
        != artifact_hashes["e1c_summary"]
        or e1ct_run.get("inputs", {}).get("split_manifest_sha256")
        != artifact_hashes["split_manifest"]
        or e1ct_run.get("inputs", {}).get("memory_records_sha256")
        != frozen_inputs["memory_records_sha256"]
    ):
        raise ValueError("E1C-T artifacts do not match their run report")
    e1ct_summary = json.loads(args.e1ct_summary.read_text(encoding="utf-8"))
    if (
        e1ct_summary.get("schema_version") != E1CT_SUMMARY_SCHEMA
        or e1ct_summary.get("sample_count") != len(assignments)
        or e1ct_summary.get("component_diagnostic", {}).get(
            "fixed_strength_test_allowed"
        )
        is not True
        or e1ct_summary.get("decision", {}).get("next_step")
        != "e1cs_fixed_log10_memory_odds_test"
    ):
        raise ValueError("E1C-T did not authorize the fixed-strength diagnostic")
    e1ct_records = _read_jsonl(args.e1ct_results)
    if any(
        record.get("schema_version") != E1CT_RESULTS_SCHEMA
        for record in e1ct_records
    ):
        raise ValueError("Unexpected E1C-T results schema")
    e1ct_by_sample = {str(record["sample_id"]): record for record in e1ct_records}
    if len(e1ct_records) != len(assignments) or set(e1ct_by_sample) != set(
        e1c_by_sample
    ):
        raise ValueError("E1C-T and E1-C source samples differ")

    reasoner = manifest["reasoner"]
    layer_number = int(reasoner["side_kv_layer"])
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    if layer_number != int(side_manifest["layer_number"]):
        raise ValueError("Frozen assignment and side-KV layer differ")
    if args.dtype != reasoner["dtype"]:
        raise ValueError("E1C-S dtype differs from the frozen assignment")

    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    validate_resolved_revisions(
        model=model, tokenizer=tokenizer, reasoner=reasoner, label="E1C-S"
    )
    loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=layer_number,
        audit_canonical_rope=False,
        memory_score_normalization=E1C_MEMORY_SCORE_NORMALIZATION,
        memory_score_bias=E1CS_MEMORY_SCORE_BIAS,
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=int(manifest["configuration"]["max_new_tokens"]),
    )
    condition_runner = FixedStrengthConditionRunner(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        loader=loader,
        controller=controller,
        layer_number=layer_number,
        device=args.device,
    )
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=frozen_inputs["dataset_revision"],
    )

    frozen_conditions = (
        "split_no_memory",
        "split_payload_only_matched",
        "split_payload_only_shuffled",
        "normalized_matched_persistent_side_kv",
        "normalized_shuffled_persistent_side_kv",
    )
    new_conditions = (
        "fixed_log10_matched_persistent_side_kv",
        "fixed_log10_shuffled_persistent_side_kv",
    )
    conditions = frozen_conditions + new_conditions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    output_records: list[dict[str, Any]] = []
    try:
        with results_path.open("w", encoding="utf-8") as handle:
            for position, assignment in enumerate(assignments, start=1):
                source_e1c = e1c_by_sample[assignment.sample_id]
                source_e1ct = e1ct_by_sample[assignment.sample_id]
                if (
                    source_e1c.get("assignment_manifest_sha256")
                    != manifest["manifest_sha256"]
                    or source_e1ct.get("assignment_manifest_sha256")
                    != manifest["manifest_sha256"]
                    or source_e1ct.get("source_e1c_result_sha256")
                    != canonical_json_sha256(source_e1c)
                    or source_e1ct.get("source_e1c_results_file_sha256")
                    != artifact_hashes["e1c_results"]
                ):
                    raise ValueError(
                        f"Frozen E1-C/E1C-T provenance drift for {assignment.sample_id}"
                    )
                if not all(
                    _is_exact_e1ct_copy_of_e1c(
                        source_e1ct["conditions"][condition],
                        source_e1c["conditions"][condition],
                    )
                    for condition in (
                        "split_no_memory",
                        "split_matched_text",
                        "split_shuffled_text",
                    )
                ):
                    raise ValueError(
                        f"E1C-T frozen condition drift for {assignment.sample_id}"
                    )
                for label, choice in (
                    ("matched", assignment.matched_memory),
                    ("shuffled", assignment.shuffled_memory),
                ):
                    assert choice is not None
                    if (
                        source_e1c.get(f"{label}_memory") != choice.to_dict()
                        or source_e1ct.get(f"{label}_memory") != choice.to_dict()
                        or source_e1c["conditions"][
                            f"{label}_persistent_side_kv"
                        ].get("memory_ids") != [choice.memory_id]
                        or source_e1ct["conditions"][
                            f"split_payload_only_{label}"
                        ].get("memory_ids") != [choice.memory_id]
                    ):
                        raise ValueError(
                            f"Frozen memory ID drift for {assignment.sample_id}"
                        )

                source = dataset[assignment.source_index]
                question = str(source["question"]).strip()
                if text_sha256(question) != assignment.question_sha256:
                    raise ValueError(
                        f"Question hash mismatch for {assignment.sample_id}"
                    )
                base_prompt_ids = prompt_token_ids(
                    tokenizer, question=question, memory_text=None
                )
                if (
                    len(base_prompt_ids) != assignment.base_prompt_token_count
                    or canonical_json_sha256(base_prompt_ids)
                    != assignment.base_prompt_token_ids_sha256
                ):
                    raise ValueError(f"Base prompt drift for {assignment.sample_id}")
                ground_truth = processed_solution(str(source["answer"]).strip())

                condition_rows = {
                    "split_no_memory": _frozen_condition(
                        source_e1ct,
                        "split_no_memory",
                        artifact_role="frozen_e1ct_reference",
                    ),
                    "split_payload_only_matched": _frozen_condition(
                        source_e1ct,
                        "split_payload_only_matched",
                        artifact_role="frozen_e1ct_reference",
                    ),
                    "split_payload_only_shuffled": _frozen_condition(
                        source_e1ct,
                        "split_payload_only_shuffled",
                        artifact_role="frozen_e1ct_reference",
                    ),
                    "normalized_matched_persistent_side_kv": _frozen_condition(
                        source_e1c,
                        "matched_persistent_side_kv",
                        artifact_role="frozen_e1c_v3_normalized_reference",
                    ),
                    "normalized_shuffled_persistent_side_kv": _frozen_condition(
                        source_e1c,
                        "shuffled_persistent_side_kv",
                        artifact_role="frozen_e1c_v3_normalized_reference",
                    ),
                }
                baseline_ids = condition_rows["split_no_memory"].get(
                    "completion_token_ids", []
                )
                if not baseline_ids:
                    raise ValueError(
                        f"No frozen split baseline tokens for {assignment.sample_id}"
                    )
                for label, choice in (
                    ("matched", assignment.matched_memory),
                    ("shuffled", assignment.shuffled_memory),
                ):
                    assert choice is not None
                    condition_rows[
                        f"fixed_log10_{label}_persistent_side_kv"
                    ] = condition_runner.run(
                        choice=choice,
                        prompt_token_ids_value=base_prompt_ids,
                        baseline_first_token_id=int(baseline_ids[0]),
                        ground_truth=ground_truth,
                    )

                record = {
                    "schema_version": E1CS_RESULTS_SCHEMA,
                    "sample_id": assignment.sample_id,
                    "logical_split": assignment.logical_split,
                    "question_sha256": assignment.question_sha256,
                    "assignment_manifest_sha256": manifest["manifest_sha256"],
                    "source_e1c_result_sha256": canonical_json_sha256(source_e1c),
                    "source_e1ct_result_sha256": canonical_json_sha256(source_e1ct),
                    "matched_memory": assignment.matched_memory.to_dict(),
                    "shuffled_memory": assignment.shuffled_memory.to_dict(),
                    "primary_prefill_path": SPLIT_PREFILL_PATH,
                    "strength": {
                        "label": E1CS_STRENGTH_LABEL,
                        "normalization": E1C_MEMORY_SCORE_NORMALIZATION,
                        "memory_score_bias": E1CS_MEMORY_SCORE_BIAS,
                        "memory_odds_multiplier": math.exp(E1CS_MEMORY_SCORE_BIAS),
                    },
                    "conditions": condition_rows,
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
                output_records.append(record)
                if position % 10 == 0 or position == len(assignments):
                    print(f"[e1cs-eval] {position}/{len(assignments)}", flush=True)
    finally:
        controller.close()

    condition_summary = summarize_conditions(output_records, conditions)
    comparisons = (
        PairedConditionComparison(
            "fixed_matched_vs_no_memory",
            "fixed_log10_matched_persistent_side_kv",
            "split_no_memory",
        ),
        PairedConditionComparison(
            "fixed_shuffled_vs_no_memory",
            "fixed_log10_shuffled_persistent_side_kv",
            "split_no_memory",
        ),
        PairedConditionComparison(
            "fixed_matched_vs_fixed_shuffled",
            "fixed_log10_matched_persistent_side_kv",
            "fixed_log10_shuffled_persistent_side_kv",
        ),
        PairedConditionComparison(
            "fixed_matched_vs_normalized_matched",
            "fixed_log10_matched_persistent_side_kv",
            "normalized_matched_persistent_side_kv",
        ),
        PairedConditionComparison(
            "fixed_shuffled_vs_normalized_shuffled",
            "fixed_log10_shuffled_persistent_side_kv",
            "normalized_shuffled_persistent_side_kv",
        ),
        PairedConditionComparison(
            "fixed_matched_vs_payload_only_matched",
            "fixed_log10_matched_persistent_side_kv",
            "split_payload_only_matched",
        ),
        PairedConditionComparison(
            "fixed_shuffled_vs_payload_only_shuffled",
            "fixed_log10_shuffled_persistent_side_kv",
            "split_payload_only_shuffled",
        ),
    )
    paired = PairedConditionDiagnostics(
        output_records, bootstrap_resamples=args.bootstrap_resamples
    ).summarize(comparisons)

    side_rows = [
        record["conditions"][condition]["side_kv"]
        for record in output_records
        for condition in new_conditions
    ]
    trace_requirements = (
        "one_trace_per_generated_token",
        "native_cache_length_matches_real_tokens",
        "all_memory_attention_mass_finite_and_positive",
        "memory_id_constant",
        "memory_slot_count_constant",
        "normalization_constant",
        "memory_score_bias_constant",
        "memory_score_bias_matches_preregistered_value",
        "baseline_first_token_matches_split_no_memory",
    )
    side_trace_invariants_passed = all(
        row.get(requirement) is True
        for row in side_rows
        for requirement in trace_requirements
    )
    path_integrity = all(
        record["primary_prefill_path"] == SPLIT_PREFILL_PATH
        and all(
            record["conditions"][condition]["prefill_path"]
            == SPLIT_PREFILL_PATH
            for condition in conditions
        )
        for record in output_records
    )
    source_mechanism_revalidated = all(
        e1c_source_mechanism_valid(
            e1c_by_sample[record["sample_id"]],
            split_prefill_path=SPLIT_PREFILL_PATH,
            expected_normalization=E1C_MEMORY_SCORE_NORMALIZATION,
        )
        for record in output_records
    )
    mechanism_integrity = (
        side_trace_invariants_passed
        and path_integrity
        and source_mechanism_revalidated
    )
    matched_mass = _mean_side_metric(
        output_records,
        "fixed_log10_matched_persistent_side_kv",
        "mean_memory_attention_mass",
    )
    shuffled_mass = _mean_side_metric(
        output_records,
        "fixed_log10_shuffled_persistent_side_kv",
        "mean_memory_attention_mass",
    )
    in_target = lambda value: (
        E1CS_MIN_MEAN_MEMORY_ATTENTION_MASS
        <= value
        <= E1CS_MAX_MEAN_MEMORY_ATTENTION_MASS
    )
    decision = E1CSFixedStrengthDecision(
        mechanism_integrity_passed=mechanism_integrity,
        matched_attention_mass_in_target_band=in_target(matched_mass),
        shuffled_attention_mass_in_target_band=in_target(shuffled_mass),
        matched_format_positive_control_transferred=effect_is_positive(
            paired["format_effects"]["fixed_matched_vs_no_memory"]
        ),
        shuffled_format_positive_control_transferred=effect_is_positive(
            paired["format_effects"]["fixed_shuffled_vs_no_memory"]
        ),
        matched_significant_answer_harm=effect_is_negative(
            paired["diagnostic_answer_effects"]["fixed_matched_vs_no_memory"]
        ),
    )
    summary = {
        "schema_version": E1CS_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "status": "completed",
        "formal_task_claim": False,
        "component_diagnostic": {
            "question": (
                "Can the frozen layer-24 side-KV channel transfer the known "
                "payload format effect at one preregistered stronger setting?"
            ),
            "one_shot_strength_test": True,
            "source_e1c_mechanism_revalidated_from_results": (
                source_mechanism_revalidated
            ),
            "all_runtime_invariants_passed": mechanism_integrity,
            "task_accuracy_is_diagnostic_only": True,
        },
        "sample_count": len(output_records),
        "condition_roles": {
            "frozen_e1ct_references": [
                "split_no_memory",
                "split_payload_only_matched",
                "split_payload_only_shuffled",
            ],
            "frozen_e1c_normalized_references": [
                "normalized_matched_persistent_side_kv",
                "normalized_shuffled_persistent_side_kv",
            ],
            "new_fixed_strength_conditions": list(new_conditions),
        },
        "fixed_strength": {
            "label": E1CS_STRENGTH_LABEL,
            "normalization": E1C_MEMORY_SCORE_NORMALIZATION,
            "memory_score_bias": E1CS_MEMORY_SCORE_BIAS,
            "memory_odds_multiplier": math.exp(E1CS_MEMORY_SCORE_BIAS),
            "mean_attention_mass_target_band": [
                E1CS_MIN_MEAN_MEMORY_ATTENTION_MASS,
                E1CS_MAX_MEAN_MEMORY_ATTENTION_MASS,
            ],
        },
        "conditions": condition_summary,
        "accuracy_effects": paired["accuracy_effects"],
        "diagnostic_answer_effects": paired["diagnostic_answer_effects"],
        "format_effects": paired["format_effects"],
        "strict_accuracy_transition_diagnostics": paired[
            "strict_accuracy_transition_diagnostics"
        ],
        "completion_difference_diagnostics": paired[
            "completion_difference_diagnostics"
        ],
        "mechanism_diagnostics": {
            "primary_prefill_path": SPLIT_PREFILL_PATH,
            "same_path_integrity": path_integrity,
            "side_trace_invariants_passed": side_trace_invariants_passed,
            "mean_matched_memory_attention_mass": matched_mass,
            "mean_shuffled_memory_attention_mass": shuffled_mass,
            "mean_matched_first_step_logits_kl": _mean_side_metric(
                output_records,
                "fixed_log10_matched_persistent_side_kv",
                "first_step_logits_kl_baseline_to_memory",
            ),
            "mean_shuffled_first_step_logits_kl": _mean_side_metric(
                output_records,
                "fixed_log10_shuffled_persistent_side_kv",
                "first_step_logits_kl_baseline_to_memory",
            ),
            "matched_first_step_top1_changed_count": sum(
                record["conditions"][
                    "fixed_log10_matched_persistent_side_kv"
                ]["side_kv"]["first_step_top1_changed"]
                for record in output_records
            ),
            "shuffled_first_step_top1_changed_count": sum(
                record["conditions"][
                    "fixed_log10_shuffled_persistent_side_kv"
                ]["side_kv"]["first_step_top1_changed"]
                for record in output_records
            ),
        },
        "decision": decision.to_dict(),
    }
    summary_path = args.output_dir / "e1cs_summary.json"
    write_json(summary_path, summary)
    write_json(args.output_dir / "run_report.json", {
        "schema_version": "experience-memory-e1cs-run-report-v1",
        "created_at": utc_now(),
        "status": "completed",
        "sample_count": len(output_records),
        "inputs": {
            f"{name}_sha256": value for name, value in artifact_hashes.items()
        },
        "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
        "summary": {"path": summary_path.name, "sha256": file_sha256(summary_path)},
    })
    print(
        f"[e1cs-eval] outcome={decision.outcome_profile} "
        f"next_step={decision.next_step} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
