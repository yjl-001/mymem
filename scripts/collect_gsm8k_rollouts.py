#!/usr/bin/env python3
"""Collect frozen-student GSM8K rollouts and label them with the project verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.experience.phase1 import (
    ROLLOUT_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    text_sha256,
)
from memgen.chat_templates import CONVERSATION_TEMPLATE
from data.utils.math_utils import (
    diagnose_gsm8k_completion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--limit", type=int, default=0, help="0 means all bank-source samples")
    parser.add_argument("--rollouts-per-sample", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    rationale = parts[0]
    clean_answer = parts[-1].strip()
    return (rationale + "\\boxed{" + clean_answer + "}").strip()


def _resolve_model_revision(model: Any, requested_revision: str) -> str:
    return str(getattr(model.config, "_commit_hash", None) or requested_revision)


def main() -> None:
    args = parse_args()
    if args.rollouts_per_sample <= 0 or args.batch_size <= 0:
        raise ValueError("rollouts-per-sample and batch-size must be positive")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if args.temperature <= 0:
        raise ValueError("Phase 1 rollout sampling requires temperature > 0")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with args.split_manifest.open(encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    if not split_manifest.get("overlap_check", {}).get("passed"):
        raise ValueError("split manifest did not pass overlap checking")
    expected_manifest_hash = split_manifest.get("manifest_sha256")
    actual_manifest_hash = canonical_json_sha256(
        {
            key: value
            for key, value in split_manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        }
    )
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("split manifest hash mismatch")
    if split_manifest.get("dataset", {}).get("revision") != args.dataset_revision:
        raise ValueError("dataset revision differs from split manifest")

    selected = [
        item for item in split_manifest["samples"] if item["logical_split"] == "bank-source"
    ]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("split manifest contains no selected bank-source samples")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=args.dataset_revision,
    )
    expected_train_fingerprint = split_manifest.get("dataset", {}).get("train_fingerprint")
    actual_train_fingerprint = getattr(dataset, "_fingerprint", None)
    if expected_train_fingerprint and expected_train_fingerprint != actual_train_fingerprint:
        raise ValueError("loaded GSM8K train fingerprint differs from split manifest")
    for item in selected:
        source = dataset[int(item["source_index"])]
        if text_sha256(str(source["question"]).strip()) != item["question_sha256"]:
            raise ValueError(f"question hash mismatch for {item['sample_id']}")
        if text_sha256(str(source["answer"]).strip()) != item["answer_sha256"]:
            raise ValueError(f"answer hash mismatch for {item['sample_id']}")

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()

    model_revision = _resolve_model_revision(model, args.model_revision)
    tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or args.model_revision
    )
    rollout_config = {
        "do_sample": True,
        "rollouts_per_sample": args.rollouts_per_sample,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "base_seed": args.seed,
        "chat_template_sha256": text_sha256(CONVERSATION_TEMPLATE),
        "prompt_version": GSM8K_PROMPT_CONTRACT.version,
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
    }
    student = {
        "model_name": args.model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "frozen": True,
        "memgen_augmentation": False,
    }

    tasks: list[tuple[dict[str, Any], int]] = []
    for sample in selected:
        for rollout_index in range(args.rollouts_per_sample):
            tasks.append((sample, rollout_index))

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    success_count = 0
    failure_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, len(tasks), args.batch_size):
            batch_tasks = tasks[batch_start : batch_start + args.batch_size]
            batch_seed = args.seed + batch_start // args.batch_size
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)

            prompts: list[str] = []
            source_rows: list[dict[str, Any]] = []
            for sample, _ in batch_tasks:
                source = dataset[int(sample["source_index"])]
                question = str(source["question"]).strip()
                prompts.append(GSM8K_PROMPT_CONTRACT.render(tokenizer, question))
                source_rows.append(source)

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for batch_position, ((sample, rollout_index), source, completion) in enumerate(
                zip(batch_tasks, source_rows, completions)
            ):
                completion = completion.strip()
                solution = _processed_solution(str(source["answer"]).strip())
                diagnosis = diagnose_gsm8k_completion(completion, solution)
                reward = diagnosis["reward"]
                outcome = "verified_success" if reward == 1.0 else "verified_failure"
                success_count += int(reward == 1.0)
                failure_count += int(reward == 0.0)
                feedback = (
                    "GSM8K strict verifier accepted the required boxed final answer."
                    if reward == 1.0
                    else "GSM8K strict verifier rejected the task response; see failure_types."
                )
                episode_id = f"{sample['sample_id']}-rollout-{rollout_index}"
                record = {
                    "schema_version": ROLLOUT_SCHEMA,
                    "episode_id": episode_id,
                    "sample_id": sample["sample_id"],
                    "source": {
                        "dataset": "openai/gsm8k",
                        "dataset_revision": args.dataset_revision,
                        "dataset_split": "train",
                        "logical_split": "bank-source",
                        "source_index": sample["source_index"],
                        "question_sha256": sample["question_sha256"],
                        "split_manifest_sha256": expected_manifest_hash,
                    },
                    "context": str(source["question"]).strip(),
                    "trajectory": completion,
                    "outcome": outcome,
                    "reward": reward,
                    "verifier": {
                        "name": "data.utils.math_utils.compute_score",
                        **diagnosis,
                        "feedback": feedback,
                    },
                    "student": student,
                    "rollout_configuration": {
                        **rollout_config,
                        "sampling_seed": batch_seed,
                        "sampling_batch_index": batch_start // args.batch_size,
                        "sampling_batch_position": batch_position,
                        "rollout_index": rollout_index,
                    },
                    "created_at": created_at,
                    "code_revision": _git_revision(),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            print(
                f"[rollouts] {min(batch_start + len(batch_tasks), len(tasks))}/{len(tasks)}",
                flush=True,
            )

    summary_path = args.summary_output or output_path.with_name("rollout_summary.json")
    summary = {
        "schema_version": "verified-student-rollout-summary-v1",
        "created_at": created_at,
        "output": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": expected_manifest_hash,
        "student": student,
        "rollout_configuration": rollout_config,
        "selected_sample_count": len(selected),
        "rollout_count": len(tasks),
        "verified_success_count": success_count,
        "verified_failure_count": failure_count,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[rollouts] complete: {output_path} sha256={summary['output_sha256']}")


if __name__ == "__main__":
    main()
