#!/usr/bin/env python3
"""Run E1-A fixed representative/random text catalogs on a frozen GSM8K split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1_staged import (
    E1A_CATALOG_MANIFEST_SCHEMA,
    E1A_RESULTS_SCHEMA,
    E1A_SUMMARY_SCHEMA,
    ExperienceCatalog,
)
from memgen.experience.phase1 import file_sha256, text_sha256
from scripts.e1_staged_common import (
    effect_is_positive,
    load_hashed_manifest,
    paired_condition_effect,
    processed_solution,
    prompt_token_ids,
    score_completion,
    select_split_samples,
    summarize_conditions,
    utc_now,
    validate_resolved_revisions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--logical-split",
        choices=("calibration-val", "dev-test"),
        default="calibration-val",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime

    catalog_manifest = load_hashed_manifest(
        args.catalog_manifest, schema=E1A_CATALOG_MANIFEST_SCHEMA
    )
    if catalog_manifest.get("status") != "frozen":
        raise ValueError("E1-A catalog manifest is not frozen")
    catalogs = tuple(
        ExperienceCatalog.from_dict(value) for value in catalog_manifest["catalogs"]
    )
    catalog_by_name = {catalog.name: catalog for catalog in catalogs}
    expected_catalog_names = {
        "representative_bank_text",
        "random_bank_text_seed17",
        "random_bank_text_seed42",
        "random_bank_text_seed73",
    }
    if set(catalog_by_name) != expected_catalog_names:
        raise ValueError("E1-A catalog conditions drifted")

    split_manifest = load_hashed_manifest(args.split_manifest)
    selected = select_split_samples(
        split_manifest,
        logical_split=args.logical_split,
        offset=args.offset,
        limit=args.limit,
    )
    dataset_revision = str(split_manifest["dataset"]["revision"])
    reasoner = catalog_manifest["reasoner"]
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
        model=model, tokenizer=tokenizer, reasoner=reasoner, label="E1-A"
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    dataset = load_dataset(
        "openai/gsm8k", "main", split="train", revision=dataset_revision
    )

    conditions = ("no_memory",) + tuple(sorted(catalog_by_name))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    records: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for position, sample in enumerate(selected, start=1):
            source = dataset[int(sample["source_index"])]
            question = str(source["question"]).strip()
            if text_sha256(question) != sample["question_sha256"]:
                raise ValueError(f"Question hash mismatch for {sample['sample_id']}")
            ground_truth = processed_solution(str(source["answer"]).strip())
            condition_rows: dict[str, dict[str, Any]] = {}
            for condition in conditions:
                catalog = catalog_by_name.get(condition)
                memory_text = catalog.rendered_text if catalog is not None else None
                prompt_ids = prompt_token_ids(
                    tokenizer, question=question, memory_text=memory_text
                )
                started = time.perf_counter()
                completion_ids = runtime.generate_vanilla(prompt_ids)
                elapsed = time.perf_counter() - started
                condition_rows[condition] = score_completion(
                    tokenizer=tokenizer,
                    completion_token_ids=completion_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=elapsed,
                    prompt_token_count=len(prompt_ids),
                    memory_ids=catalog.memory_ids if catalog is not None else (),
                )
            record = {
                "schema_version": E1A_RESULTS_SCHEMA,
                "sample_id": sample["sample_id"],
                "logical_split": args.logical_split,
                "source_index": sample["source_index"],
                "question_sha256": sample["question_sha256"],
                "catalog_manifest_sha256": catalog_manifest["manifest_sha256"],
                "conditions": condition_rows,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            records.append(record)
            if position % 10 == 0 or position == len(selected):
                print(f"[e1a-eval] {position}/{len(selected)}", flush=True)

    condition_summary = summarize_conditions(records, conditions)
    representative_vs_no = paired_condition_effect(
        records,
        treatment="representative_bank_text",
        control="no_memory",
        field="final_reward",
        resamples=args.bootstrap_resamples,
    )
    representative_format_vs_no = paired_condition_effect(
        records,
        treatment="representative_bank_text",
        control="no_memory",
        field="format_valid",
        resamples=args.bootstrap_resamples,
    )
    random_effects = {
        name: paired_condition_effect(
            records,
            treatment=name,
            control="no_memory",
            field="final_reward",
            resamples=args.bootstrap_resamples,
        )
        for name in sorted(expected_catalog_names - {"representative_bank_text"})
    }
    acceptance = {
        "representative_accuracy_above_no_memory": effect_is_positive(
            representative_vs_no
        ),
        "representative_format_not_below_no_memory": (
            float(representative_format_vs_no["mean_treatment_minus_control"]) >= 0.0
        ),
    }
    summary = {
        "schema_version": E1A_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "status": "passed" if all(acceptance.values()) else "did_not_pass",
        "formal_e1a_passed": all(acceptance.values()),
        "logical_split": args.logical_split,
        "sample_count": len(records),
        "provenance": {
            "catalog_manifest_logical_sha256": catalog_manifest[
                "manifest_sha256"
            ],
            "catalog_manifest_file_sha256": file_sha256(args.catalog_manifest),
            "results_sha256": file_sha256(results_path),
        },
        "conditions": condition_summary,
        "primary": {
            "representative_vs_no_memory_accuracy": representative_vs_no,
            "representative_vs_no_memory_format": representative_format_vs_no,
        },
        "random_bank_sensitivity": random_effects,
        "acceptance": acceptance,
    }
    write_json(args.output_dir / "e1a_summary.json", summary)
    write_json(args.output_dir / "run_report.json", {
        "schema_version": "experience-memory-e1a-run-report-v1",
        "created_at": utc_now(),
        "status": "completed",
        "sample_count": len(records),
        "configuration": {
            "logical_split": args.logical_split,
            "offset": args.offset,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "attention_implementation": "eager",
        },
        "inputs": {
            "catalog_manifest_sha256": file_sha256(args.catalog_manifest),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "dataset_revision": dataset_revision,
        },
        "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
    })
    print(
        f"[e1a-eval] status={summary['status']} output={args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()
