#!/usr/bin/env python3
"""Build the answer-blind completion-aware BM25 assignment manifest for E1-B."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1 import MemoryChoice
from memgen.experience.e1_staged import (
    E1B_MANIFEST_SCHEMA,
    E1B_SHUFFLE_SEED,
    CompletionAwareRetrievalQueryBuilder,
    E1BRetrievalAssignment,
    E1BRetrievalDeranger,
)
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.retrieval import BM25MemoryIndex
from scripts.e1_staged_common import (
    load_hashed_manifest,
    prompt_token_ids,
    select_split_samples,
    validate_resolved_revisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--logical-split",
        choices=("calibration-val", "dev-test"),
        default="calibration-val",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=768)
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

    e0 = json.loads(args.e0_final_report.read_text(encoding="utf-8"))
    if e0.get("formal_e0_passed") is not True or e0.get("task_accuracy_used") is not False:
        raise ValueError("E1-B requires a formally passed answer-blind E0")
    split_manifest = load_hashed_manifest(args.split_manifest)
    selected = select_split_samples(
        split_manifest,
        logical_split=args.logical_split,
        offset=args.offset,
        limit=args.limit,
    )
    dataset_revision = str(split_manifest["dataset"]["revision"])
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    record_by_id = {record.memory_id: record for record in records}
    bm25 = BM25MemoryIndex.from_dict(
        records=records,
        value=json.loads(args.bm25_index.read_text(encoding="utf-8")),
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    if side_manifest.get("manifest_sha256") != canonical_json_sha256({
        key: value for key, value in side_manifest.items() if key != "manifest_sha256"
    }):
        raise ValueError("Side-KV manifest hash mismatch")
    side_entries = {
        str(item["memory_id"]): item for item in side_manifest.get("records", [])
    }
    if set(side_entries) != set(record_by_id):
        raise ValueError("BM25 text index and side-KV bank cover different memories")
    for memory_id, entry in side_entries.items():
        record = record_by_id[memory_id]
        if (
            entry.get("payload_hash") != record.payload_hash
            or int(entry.get("kv_valid_slot_count", -1)) != record.token_count
        ):
            raise ValueError("Text/side-KV memory metadata mismatch")

    reasoner = side_manifest["reasoner"]
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
        model=model,
        tokenizer=tokenizer,
        reasoner=reasoner,
        label="E1-B assignment",
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    query_builder = CompletionAwareRetrievalQueryBuilder(analyzer=bm25.analyzer)
    dataset = load_dataset(
        "openai/gsm8k", "main", split="train", revision=dataset_revision
    )

    assignments: list[E1BRetrievalAssignment] = []
    for position, sample in enumerate(selected, start=1):
        source = dataset[int(sample["source_index"])]
        question = str(source["question"]).strip()
        del source
        if text_sha256(question) != sample["question_sha256"]:
            raise ValueError(f"Question hash mismatch for {sample['sample_id']}")
        base_prompt_ids = prompt_token_ids(
            tokenizer, question=question, memory_text=None
        )
        preanswer_ids = runtime.generate_vanilla(base_prompt_ids)
        preanswer_text = tokenizer.decode(
            list(preanswer_ids), skip_special_tokens=True
        ).strip()
        query = query_builder.build(question=question, completion=preanswer_text)
        hits = bm25.search(query.query_text, top_k=2)
        if not hits:
            raise RuntimeError(
                f"E1-B requires a positive BM25 hit for {sample['sample_id']}"
            )
        top = hits[0]
        entry = side_entries[top.memory_id]
        matched = MemoryChoice(
            memory_id=top.memory_id,
            payload_hash=top.payload_hash,
            token_count=top.token_count,
            kv_valid_slot_count=int(entry["kv_valid_slot_count"]),
            retrieval_score=float(top.score),
            retrieval_rank=top.rank,
        )
        query_artifact = query.to_dict(include_text=False)
        query_artifact.update({
            "method": "bm25",
            "query_policy": "question-plus-sanitized-complete-preanswer-v1",
            "sanitized_preanswer_sha256": canonical_json_sha256(
                query.normalized_partial_cot
            ),
            "top1_score": float(top.score),
            "top2_score": float(hits[1].score) if len(hits) > 1 else None,
            "top1_top2_margin": (
                float(top.score - hits[1].score) if len(hits) > 1 else None
            ),
        })
        assignments.append(E1BRetrievalAssignment(
            sample_id=str(sample["sample_id"]),
            logical_split=args.logical_split,
            dataset_split=str(sample["dataset_split"]),
            source_index=int(sample["source_index"]),
            question_sha256=str(sample["question_sha256"]),
            base_prompt_token_ids_sha256=canonical_json_sha256(base_prompt_ids),
            base_prompt_token_count=len(base_prompt_ids),
            preanswer_completion_token_ids=preanswer_ids,
            preanswer_completion_token_ids_sha256=canonical_json_sha256(
                list(preanswer_ids)
            ),
            preanswer_completion_text_sha256=text_sha256(preanswer_text),
            retrieval_query=query_artifact,
            matched_memory=matched,
        ))
        if position % 10 == 0 or position == len(selected):
            print(f"[e1b-assignment] {position}/{len(selected)}", flush=True)

    frozen, shuffle_report = E1BRetrievalDeranger(
        seed=E1B_SHUFFLE_SEED
    ).assign(assignments)
    manifest = {
        "schema_version": E1B_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "answer_or_reward_used": False,
        "logical_split": args.logical_split,
        "reasoner": {
            "model_name": reasoner["model_name"],
            "model_revision": reasoner["model_revision"],
            "tokenizer_revision": reasoner["tokenizer_revision"],
            "dtype": args.dtype,
            "attention_implementation": "eager",
            "side_kv_layer": int(side_manifest["layer_number"]),
        },
        "configuration": {
            "offset": args.offset,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "first_response_use": "query_only_never_second_prompt",
            "retrieval": {
                "method": "bm25",
                "top_k_assignment": 1,
                "top_k_diagnostic": 2,
                "query_policy": "question-plus-sanitized-complete-preanswer-v1",
                "analyzer": asdict(bm25.analyzer.config),
                "bm25": asdict(bm25.config),
            },
            "shuffle": shuffle_report,
        },
        "inputs": {
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_path": str(args.memory_records.resolve()),
            "memory_records_sha256": file_sha256(args.memory_records),
            "bm25_index_path": str(args.bm25_index.resolve()),
            "bm25_index_sha256": file_sha256(args.bm25_index),
            "side_kv_manifest_path": str(args.side_kv_manifest.resolve()),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "e0_final_report_path": str(args.e0_final_report.resolve()),
            "e0_final_report_sha256": file_sha256(args.e0_final_report),
            "dataset_revision": dataset_revision,
        },
        "summary": {
            "sample_count": len(frozen),
            "assigned_count": sum(item.assigned for item in frozen),
            "answer_or_reward_used": False,
        },
        "assignments": [item.to_dict() for item in frozen],
    }
    manifest["manifest_sha256"] = canonical_json_sha256({
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[e1b-assignment] frozen={len(frozen)} output={args.output}", flush=True
    )


if __name__ == "__main__":
    main()
