#!/usr/bin/env python3
"""Build answer-blind calibration prefixes for the E0 side-KV mechanism audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
    write_jsonl,
)
from memgen.experience.phase2 import build_gsm8k_messages


_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--logical-split", default="calibration-val")
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_split_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    actual = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        }
    )
    if expected != actual or not manifest.get("overlap_check", {}).get("passed"):
        raise ValueError("Invalid or overlapping GSM8K split manifest")
    return manifest


def first_reasoning_boundary(tokenizer: Any, completion_ids: list[int]) -> int | None:
    decoded_prefix = ""
    for index, token_id in enumerate(completion_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        decoded_prefix += token_text
        if _ANSWER_MARKER_RE.search(decoded_prefix):
            return None
        if token_text.rstrip(" \t").endswith((",", ".", "\n")):
            return index
    return None


def main() -> None:
    args = parse_args()
    if args.logical_split != "calibration-val":
        raise ValueError("E0 audit cases must come from calibration-val")
    if args.case_count <= 0 or args.max_new_tokens <= 0:
        raise ValueError("case-count and max-new-tokens must be positive")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    manifest = load_split_manifest(args.split_manifest)
    if manifest.get("dataset", {}).get("revision") != args.dataset_revision:
        raise ValueError("Dataset revision differs from the split manifest")
    samples = [
        item
        for item in manifest["samples"]
        if item.get("logical_split") == args.logical_split
    ]
    if not samples:
        raise ValueError("Split manifest has no calibration-val samples")
    memory_records = list(iter_jsonl(args.memory_records))
    memory_ids = [str(record.get("memory_id", "")) for record in memory_records]
    if not memory_ids or any(not memory_id for memory_id in memory_ids):
        raise ValueError("Memory records do not contain valid memory IDs")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=args.dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    cases: list[dict[str, Any]] = []
    skipped_without_boundary = 0
    for sample in samples:
        source = dataset[int(sample["source_index"])]
        question = str(source["question"]).strip()
        if text_sha256(question) != sample["question_sha256"]:
            raise ValueError(f"Question hash mismatch for {sample['sample_id']}")
        prompt = tokenizer.apply_chat_template(
            build_gsm8k_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        attention_mask = torch.ones_like(inputs)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
        completion_ids = [int(value) for value in generated[0, len(prompt_ids) :].tolist()]
        boundary_index = first_reasoning_boundary(tokenizer, completion_ids)
        if boundary_index is None:
            skipped_without_boundary += 1
            continue
        selected_completion = completion_ids[: boundary_index + 1]
        memory_id = memory_ids[len(cases) % len(memory_ids)]
        prefix_ids = prompt_ids + selected_completion
        cases.append(
            {
                "schema_version": "side-kv-mechanism-audit-case-input-v1",
                "case_id": f"{sample['sample_id']}-boundary-{boundary_index}",
                "sample_id": sample["sample_id"],
                "logical_split": args.logical_split,
                "question_sha256": sample["question_sha256"],
                "memory_id": memory_id,
                "answer_or_reward_used": False,
                "prompt_token_count": len(prompt_ids),
                "generated_boundary_index": boundary_index,
                "boundary_token_id": selected_completion[-1],
                "prefix_token_ids": prefix_ids,
                "prefix_token_ids_sha256": canonical_json_sha256(prefix_ids),
                "selection_policy": "first_preanswer_reasoning_delimiter",
            }
        )
        if len(cases) >= args.case_count:
            break
    if len(cases) < args.case_count:
        raise RuntimeError(
            f"Only {len(cases)} of {args.case_count} calibration audit cases had a usable boundary"
        )

    write_jsonl(args.output, cases)
    report_path = args.report_output or args.output.with_name("audit_case_build_report.json")
    report = {
        "schema_version": "side-kv-mechanism-audit-case-build-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_or_reward_used": False,
        "logical_split": args.logical_split,
        "case_count": len(cases),
        "skipped_without_boundary": skipped_without_boundary,
        "selection_policy": "first_preanswer_reasoning_delimiter",
        "inputs": {
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "model": args.model,
            "model_revision": str(
                getattr(model.config, "_commit_hash", None) or args.model_revision
            ),
            "dataset_revision": args.dataset_revision,
            "max_new_tokens": args.max_new_tokens,
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": file_sha256(args.output),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[side-kv-audit-cases] cases={len(cases)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
