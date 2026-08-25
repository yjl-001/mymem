#!/usr/bin/env python3
"""Compare canonical native and explicit-cache base-reasoner generation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1 import E1EvaluationScope
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    text_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--logical-split",
        choices=("calibration-val", "dev-test", "final-test"),
        default="calibration-val",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=32, help="Zero selects the full split."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "flash_attention_2"),
        default="eager",
        help="The only model-execution variable in the backend diagnostic.",
    )
    return parser.parse_args()


def load_hashed_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("manifest_sha256")
    actual = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    if expected != actual:
        raise ValueError(f"Manifest hash mismatch: {path}")
    return value


def processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def score_completion(
    *, tokenizer: Any, token_ids: tuple[int, ...], ground_truth: str
) -> dict[str, Any]:
    completion = tokenizer.decode(
        list(token_ids), skip_special_tokens=True
    ).strip()
    verifier = diagnose_gsm8k_completion(completion, ground_truth)
    return {
        "completion": completion,
        "completion_token_ids": list(token_ids),
        "completion_token_ids_sha256": canonical_json_sha256(list(token_ids)),
        "generation_length": len(token_ids),
        "final_reward": verifier["reward"],
        "format_valid": verifier["format_valid"],
        "verifier": verifier,
    }


def aggregate(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    rows = [record["conditions"][condition] for record in records]
    return {
        "sample_count": len(rows),
        "accuracy": sum(float(row["final_reward"]) for row in rows) / len(rows),
        "format_accuracy": sum(bool(row["format_valid"]) for row in rows)
        / len(rows),
        "diagnostic_answer_accuracy": sum(
            row["verifier"].get("diagnostic_answer_correct") is True
            for row in rows
        ) / len(rows),
        "mean_generation_length": sum(
            int(row["generation_length"]) for row in rows
        ) / len(rows),
    }


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit < 0:
        raise ValueError("offset and limit must be non-negative")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime, compare_token_sequences

    split_manifest = load_hashed_manifest(args.split_manifest)
    if not split_manifest.get("overlap_check", {}).get("passed"):
        raise ValueError("GSM8K split manifest did not pass overlap audit")
    scope = E1EvaluationScope.from_logical_split(args.logical_split)
    selected = [
        item
        for item in split_manifest["samples"]
        if item.get("logical_split") == scope.logical_split
    ][args.offset :]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("Selected GSM8K parity split is empty")
    if {item.get("dataset_split") for item in selected} != {scope.dataset_split}:
        raise ValueError("Logical split contains an unexpected dataset split")

    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    expected_side_hash = side_manifest.get("manifest_sha256")
    actual_side_hash = canonical_json_sha256({
        key: item
        for key, item in side_manifest.items()
        if key != "manifest_sha256"
    })
    if expected_side_hash != actual_side_hash:
        raise ValueError("Side-KV manifest hash mismatch")
    reasoner = side_manifest.get("reasoner", {})
    model_name = str(reasoner.get("model_name", ""))
    model_revision = str(reasoner.get("model_revision", ""))
    tokenizer_revision = str(reasoner.get("tokenizer_revision", ""))
    if not all((model_name, model_revision, tokenizer_revision)):
        raise ValueError("Side-KV manifest has incomplete reasoner provenance")

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
        dtype=dtype,
        attn_implementation=args.attention_implementation,
    ).to(args.device)
    model.eval()
    if str(getattr(model.config, "_commit_hash", None) or model_revision) != model_revision:
        raise ValueError("Runtime model revision differs from E0")
    if str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or tokenizer_revision
    ) != tokenizer_revision:
        raise ValueError("Runtime tokenizer revision differs from E0")

    dataset_revision = str(split_manifest.get("dataset", {}).get("revision", ""))
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=scope.dataset_split,
        revision=dataset_revision,
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=GSM8K_PROMPT_CONTRACT.max_new_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    records: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for position, sample in enumerate(selected, start=1):
            source = dataset[int(sample["source_index"])]
            question = str(source["question"]).strip()
            answer = str(source["answer"]).strip()
            if text_sha256(question) != sample["question_sha256"]:
                raise ValueError(f"Question hash mismatch for {sample['sample_id']}")
            if text_sha256(answer) != sample["answer_sha256"]:
                raise ValueError(f"Answer hash mismatch for {sample['sample_id']}")
            prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, question)
            started = time.perf_counter()
            native_ids = runtime.generate_vanilla(prompt_ids)
            native_seconds = time.perf_counter() - started
            started = time.perf_counter()
            cache_ids = runtime.generate_cache_greedy(prompt_ids)
            cache_seconds = time.perf_counter() - started
            parity = compare_token_sequences(native_ids, cache_ids)
            ground_truth = processed_solution(answer)
            record = {
                "schema_version": "gsm8k-base-generation-parity-result-v3",
                "sample_id": sample["sample_id"],
                "logical_split": scope.logical_split,
                "dataset_split": scope.dataset_split,
                "source_index": int(sample["source_index"]),
                "question_sha256": sample["question_sha256"],
                "prompt_token_count": len(prompt_ids),
                "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
                "attention_implementation": args.attention_implementation,
                "parity": parity.to_dict(),
                "runtime_seconds": {
                    "native_transformers_generate": native_seconds,
                    "explicit_live_kv_cache": cache_seconds,
                },
                "conditions": {
                    "native_transformers_generate": score_completion(
                        tokenizer=tokenizer,
                        token_ids=native_ids,
                        ground_truth=ground_truth,
                    ),
                    "explicit_live_kv_cache": score_completion(
                        tokenizer=tokenizer,
                        token_ids=cache_ids,
                        ground_truth=ground_truth,
                    ),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            records.append(record)
            print(
                f"[base-parity] {position}/{len(selected)} "
                f"exact={sum(item['parity']['exact_match'] for item in records)}",
                flush=True,
            )

    mismatch_ids = [
        record["sample_id"]
        for record in records
        if not record["parity"]["exact_match"]
    ]
    mismatch_index_counts = Counter(
        int(record["parity"]["first_mismatch_index"])
        for record in records
        if record["parity"]["first_mismatch_index"] is not None
    )
    conditions = {
        name: aggregate(records, name)
        for name in (
            "native_transformers_generate",
            "explicit_live_kv_cache",
        )
    }
    summary = {
        "schema_version": "gsm8k-base-generation-parity-report-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not mismatch_ids else "failed_token_parity",
        "formal_memory_claim": False,
        "logical_split": scope.logical_split,
        "dataset_split": scope.dataset_split,
        "sample_count": len(records),
        "exact_token_parity": not mismatch_ids,
        "parity_mismatch_count": len(mismatch_ids),
        "parity_mismatch_sample_ids": mismatch_ids,
        "first_mismatch_index_counts": {
            str(index): count
            for index, count in sorted(mismatch_index_counts.items())
        },
        "first_token_mismatch_count": mismatch_index_counts.get(0, 0),
        "mean_shared_prefix_length": sum(
            int(record["parity"]["shared_prefix_length"])
            for record in records
        ) / len(records),
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "generation_contract": {
            "attention_implementation": args.attention_implementation,
            "native_transformers_generate": {
                **runtime.native_generation_config_dict,
                "implementation": "transformers_generate",
            },
            "explicit_live_kv_cache": {
                **runtime.cache_generation_config_dict,
                "implementation": "explicit_live_kv_cache",
            },
        },
        "model_generation_defaults": {
            "do_sample": bool(model.generation_config.do_sample),
            "repetition_penalty": float(
                model.generation_config.repetition_penalty
            ),
            "temperature": float(model.generation_config.temperature),
            "top_p": float(model.generation_config.top_p),
            "top_k": int(model.generation_config.top_k),
            "eos_token_id": model.generation_config.eos_token_id,
            "pad_token_id": model.generation_config.pad_token_id,
        },
        "reasoner": {
            "model_name": model_name,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "dtype": args.dtype,
            "attention_implementation": args.attention_implementation,
        },
        "conditions": conditions,
        "inputs": {
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
        },
        "results": {
            "path": results_path.name,
            "sha256": file_sha256(results_path),
        },
    }
    summary_path = args.output_dir / "base_parity_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[base-parity] status={summary['status']} output={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
