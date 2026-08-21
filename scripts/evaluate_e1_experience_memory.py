#!/usr/bin/env python3
"""Execute the four frozen E1 conditions and score GSM8K after generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1 import (
    E1_CONDITIONS,
    E1_MANIFEST_SCHEMA,
    E1_RESULTS_SCHEMA,
    E1Assignment,
)
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    text_sha256,
)
from memgen.experience.phase2 import build_gsm8k_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def prompt_token_ids(tokenizer: Any, question: str) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        build_gsm8k_messages(question),
        tokenize=False,
        add_generation_prompt=True,
    )
    return [
        int(value)
        for value in tokenizer.encode(prompt, add_special_tokens=False)
    ]


def condition_result(
    *,
    tokenizer: Any,
    completion_token_ids: tuple[int, ...],
    ground_truth: str,
    runtime_seconds: float | None,
    memory_id: str | None = None,
    payload_hash: str | None = None,
    attention_trace: Any | None = None,
    first_step_logits_kl: float | None = None,
    first_step_top1_changed: bool | None = None,
) -> dict[str, Any]:
    completion = tokenizer.decode(
        list(completion_token_ids), skip_special_tokens=True
    ).strip()
    verifier = diagnose_gsm8k_completion(completion, ground_truth)
    return {
        "completion": completion,
        "completion_token_ids": list(completion_token_ids),
        "completion_token_ids_sha256": canonical_json_sha256(
            list(completion_token_ids)
        ),
        "generation_length": len(completion_token_ids),
        "verifier": verifier,
        "final_reward": verifier["reward"],
        "format_valid": verifier["format_valid"],
        "runtime_seconds": runtime_seconds,
        "memory_id": memory_id,
        "payload_hash": payload_hash,
        "side_kv_applied": memory_id is not None,
        "memory_attention": (
            attention_trace.to_dict() if attention_trace is not None else None
        ),
        "first_step_logits_kl_baseline_to_memory": first_step_logits_kl,
        "first_step_top1_changed": first_step_top1_changed,
    }


def main() -> None:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime
    from memgen.model.side_kv import (
        SideKVAttentionController,
        SideKVBankLoader,
    )

    manifest = json.loads(args.assignment_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != E1_MANIFEST_SCHEMA:
        raise ValueError("Unexpected E1 assignment manifest schema")
    expected_manifest_hash = manifest.get("manifest_sha256")
    actual_manifest_hash = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        }
    )
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("E1 assignment manifest hash mismatch")
    if manifest.get("status") != "frozen" or manifest.get("answer_or_reward_used") is not False:
        raise ValueError("E1 assignments are not frozen and answer-blind")
    inputs = manifest.get("inputs", {})
    for path, expected_hash, label in (
        (
            args.side_kv_manifest,
            inputs.get("side_kv_manifest_sha256"),
            "side-KV manifest",
        ),
        (
            args.split_manifest,
            inputs.get("split_manifest_sha256"),
            "split manifest",
        ),
    ):
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"E1 {label} differs from the frozen assignment input")

    assignments = tuple(
        E1Assignment.from_dict(value) for value in manifest["assignments"]
    )
    if len(assignments) != int(manifest.get("summary", {}).get("sample_count", -1)):
        raise ValueError("E1 assignment count differs from manifest summary")
    if any(not item.assigned for item in assignments if item.matched_memory is not None):
        raise ValueError("E1 assignment has matched memory without shuffled control")

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    dataset_revision = str(inputs.get("dataset_revision", ""))
    if split_manifest.get("dataset", {}).get("revision") != dataset_revision:
        raise ValueError("E1 dataset revision differs from the split manifest")
    split_entries = {
        str(item["sample_id"]): item for item in split_manifest["samples"]
    }
    if any(item.sample_id not in split_entries for item in assignments):
        raise ValueError("E1 assignments include unknown split sample IDs")

    reasoner = manifest.get("reasoner", {})
    model_name = str(reasoner.get("model_name", ""))
    model_revision = str(reasoner.get("model_revision", ""))
    tokenizer_revision = str(reasoner.get("tokenizer_revision", ""))
    layer = int(reasoner.get("layer", -1))
    if layer != 24 or reasoner.get("attention_implementation") != "eager":
        raise ValueError("E1-v1 requires eager side-KV at layer 24")
    if args.dtype != reasoner.get("dtype"):
        raise ValueError("Runtime dtype differs from the assignment manifest")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=tokenizer_revision
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
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    resolved_model_revision = str(
        getattr(model.config, "_commit_hash", None) or model_revision
    )
    resolved_tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or tokenizer_revision
    )
    if (
        resolved_model_revision != model_revision
        or resolved_tokenizer_revision != tokenizer_revision
    ):
        raise ValueError("E1 runtime reasoner differs from the assignment manifest")

    loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=model_name,
        expected_reasoner_revision=model_revision,
        expected_tokenizer_revision=tokenizer_revision,
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=layer,
        audit_canonical_rope=False,
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=int(manifest["configuration"]["max_new_tokens"]),
    )
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=dataset_revision,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    records: list[dict[str, Any]] = []
    try:
        with results_path.open("w", encoding="utf-8") as output_handle:
            for position, assignment in enumerate(assignments, start=1):
                source = dataset[assignment.source_index]
                question = str(source["question"]).strip()
                answer = str(source["answer"]).strip()
                if text_sha256(question) != assignment.question_sha256:
                    raise ValueError(
                        f"Question hash mismatch for {assignment.sample_id}"
                    )
                prompt_ids = prompt_token_ids(tokenizer, question)
                if (
                    len(prompt_ids) != assignment.prompt_token_count
                    or canonical_json_sha256(prompt_ids)
                    != assignment.prompt_token_ids_sha256
                ):
                    raise ValueError(
                        f"Prompt tokenization drift for {assignment.sample_id}"
                    )
                ground_truth = processed_solution(answer)

                started = time.perf_counter()
                vanilla_ids = runtime.generate_vanilla(prompt_ids)
                vanilla_seconds = time.perf_counter() - started
                gate_ids = assignment.observation_completion_token_ids
                conditions: dict[str, dict[str, Any]] = {
                    "vanilla": condition_result(
                        tokenizer=tokenizer,
                        completion_token_ids=vanilla_ids,
                        ground_truth=ground_truth,
                        runtime_seconds=vanilla_seconds,
                    ),
                    "gate_observation_only": condition_result(
                        tokenizer=tokenizer,
                        completion_token_ids=gate_ids,
                        ground_truth=ground_truth,
                        runtime_seconds=None,
                    ),
                }

                if assignment.assigned:
                    assert assignment.matched_memory is not None
                    assert assignment.shuffled_memory is not None
                    started = time.perf_counter()
                    matched_runtime = runtime.generate_with_memory(
                        prefix_token_ids=assignment.prefix_token_ids,
                        prompt_token_count=assignment.prompt_token_count,
                        memory=loader.get(
                            assignment.matched_memory.memory_id,
                            device=args.device,
                            dtype=next(model.parameters()).dtype,
                        ),
                        controller=controller,
                    )
                    matched_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    shuffled_runtime = runtime.generate_with_memory(
                        prefix_token_ids=assignment.prefix_token_ids,
                        prompt_token_count=assignment.prompt_token_count,
                        memory=loader.get(
                            assignment.shuffled_memory.memory_id,
                            device=args.device,
                            dtype=next(model.parameters()).dtype,
                        ),
                        controller=controller,
                    )
                    shuffled_seconds = time.perf_counter() - started
                    conditions["matched_memory"] = condition_result(
                        tokenizer=tokenizer,
                        completion_token_ids=matched_runtime.completion_token_ids,
                        ground_truth=ground_truth,
                        runtime_seconds=matched_seconds,
                        memory_id=assignment.matched_memory.memory_id,
                        payload_hash=assignment.matched_memory.payload_hash,
                        attention_trace=matched_runtime.attention_trace,
                        first_step_logits_kl=matched_runtime.first_step_logits_kl,
                        first_step_top1_changed=matched_runtime.first_step_top1_changed,
                    )
                    conditions["shuffled_memory"] = condition_result(
                        tokenizer=tokenizer,
                        completion_token_ids=shuffled_runtime.completion_token_ids,
                        ground_truth=ground_truth,
                        runtime_seconds=shuffled_seconds,
                        memory_id=assignment.shuffled_memory.memory_id,
                        payload_hash=assignment.shuffled_memory.payload_hash,
                        attention_trace=shuffled_runtime.attention_trace,
                        first_step_logits_kl=shuffled_runtime.first_step_logits_kl,
                        first_step_top1_changed=shuffled_runtime.first_step_top1_changed,
                    )
                else:
                    for condition in ("matched_memory", "shuffled_memory"):
                        conditions[condition] = condition_result(
                            tokenizer=tokenizer,
                            completion_token_ids=gate_ids,
                            ground_truth=ground_truth,
                            runtime_seconds=None,
                        )
                if set(conditions) != set(E1_CONDITIONS):
                    raise RuntimeError("E1 runner did not produce all four conditions")
                record = {
                    "schema_version": E1_RESULTS_SCHEMA,
                    "sample_id": assignment.sample_id,
                    "logical_split": assignment.logical_split,
                    "question_sha256": assignment.question_sha256,
                    "assignment_manifest_sha256": expected_manifest_hash,
                    "triggered": assignment.triggered,
                    "assigned": assignment.assigned,
                    "abstain_reason": assignment.abstain_reason,
                    "prefix_token_ids_sha256": assignment.prefix_token_ids_sha256,
                    "gate_observation": (
                        assignment.gate_observation.to_dict()
                        if assignment.gate_observation is not None
                        else None
                    ),
                    "retrieval_query": assignment.retrieval_query,
                    "matched_memory": (
                        assignment.matched_memory.to_dict()
                        if assignment.matched_memory is not None
                        else None
                    ),
                    "shuffled_memory": (
                        assignment.shuffled_memory.to_dict()
                        if assignment.shuffled_memory is not None
                        else None
                    ),
                    "vanilla_matches_gate_observation_only": (
                        vanilla_ids == gate_ids
                    ),
                    "conditions": conditions,
                }
                output_handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                output_handle.flush()
                records.append(record)
                if position % 10 == 0 or position == len(assignments):
                    print(
                        f"[e1-eval] {position}/{len(assignments)} "
                        f"assigned={sum(item['assigned'] for item in records)}",
                        flush=True,
                    )
    finally:
        controller.close()

    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in E1_CONDITIONS:
        condition_rows = [record["conditions"][condition] for record in records]
        condition_summaries[condition] = {
            "accuracy": sum(float(item["final_reward"]) for item in condition_rows)
            / len(condition_rows),
            "format_accuracy": sum(bool(item["format_valid"]) for item in condition_rows)
            / len(condition_rows),
            "mean_generation_length": sum(
                int(item["generation_length"]) for item in condition_rows
            )
            / len(condition_rows),
            "side_kv_applied_count": sum(
                bool(item["side_kv_applied"]) for item in condition_rows
            ),
        }
    run_report = {
        "schema_version": "experience-memory-e1-run-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "sample_count": len(records),
        "triggered_count": sum(bool(item["triggered"]) for item in records),
        "assigned_count": sum(bool(item["assigned"]) for item in records),
        "vanilla_gate_token_parity": all(
            item["vanilla_matches_gate_observation_only"] for item in records
        ),
        "conditions": condition_summaries,
        "inputs": {
            "assignment_manifest_path": str(args.assignment_manifest.resolve()),
            "assignment_manifest_sha256": file_sha256(args.assignment_manifest),
            "assignment_manifest_logical_sha256": expected_manifest_hash,
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "split_manifest_sha256": file_sha256(args.split_manifest),
        },
        "results": {
            "path": results_path.name,
            "sha256": file_sha256(results_path),
        },
    }
    write_json(args.output_dir / "run_report.json", run_report)
    print(
        f"[e1-eval] completed samples={len(records)} "
        f"parity={run_report['vanilla_gate_token_parity']} "
        f"output={results_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
