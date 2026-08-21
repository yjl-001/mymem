#!/usr/bin/env python3
"""Freeze the representative and random Phase 1 experience catalogs for E1-A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1_staged import (
    E1A_CATALOG_MANIFEST_SCHEMA,
    E1A_CATALOG_TOKEN_BUDGET,
    E1A_RANDOM_SEEDS,
    ConstrainedKMedoidsCatalogBuilder,
)
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl
from memgen.experience.retrieval import BM25MemoryIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    e0 = json.loads(args.e0_final_report.read_text(encoding="utf-8"))
    if e0.get("formal_e0_passed") is not True or e0.get("task_accuracy_used") is not False:
        raise ValueError("E1-A requires a formally passed answer-blind E0")
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    bm25 = BM25MemoryIndex.from_dict(
        records=records,
        value=json.loads(args.bm25_index.read_text(encoding="utf-8")),
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    expected_side_hash = side_manifest.get("manifest_sha256")
    if expected_side_hash != canonical_json_sha256({
        key: value for key, value in side_manifest.items() if key != "manifest_sha256"
    }):
        raise ValueError("Side-KV manifest hash mismatch")
    side_ids = {str(item["memory_id"]) for item in side_manifest.get("records", [])}
    if side_ids != {record.memory_id for record in records}:
        raise ValueError("Text records and side-KV bank cover different memories")

    reasoner = side_manifest["reasoner"]
    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE

    def token_counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    builder = ConstrainedKMedoidsCatalogBuilder(
        records=records,
        token_counter=token_counter,
        analyzer=bm25.analyzer,
        token_budget=E1A_CATALOG_TOKEN_BUDGET,
    )
    representative = builder.build_representative()
    random_catalogs = tuple(
        builder.build_random_control(representative=representative, seed=seed)
        for seed in E1A_RANDOM_SEEDS
    )
    catalogs = (representative,) + random_catalogs
    if any(len(item.memory_ids) != len(representative.memory_ids) for item in catalogs):
        raise RuntimeError("E1-A catalogs do not have equal record counts")
    if len({item.memory_ids for item in catalogs}) != len(catalogs):
        raise RuntimeError("E1-A representative/random catalogs must be distinct")

    manifest = {
        "schema_version": E1A_CATALOG_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "task_answer_or_reward_used": False,
        "configuration": {
            "catalog_token_budget": E1A_CATALOG_TOKEN_BUDGET,
            "random_seeds": list(E1A_RANDOM_SEEDS),
            "representative_method": representative.method,
            "memory_record_count": len(records),
        },
        "reasoner": {
            "model_name": reasoner["model_name"],
            "model_revision": reasoner["model_revision"],
            "tokenizer_revision": reasoner["tokenizer_revision"],
        },
        "inputs": {
            "memory_records_path": str(args.memory_records.resolve()),
            "memory_records_sha256": file_sha256(args.memory_records),
            "bm25_index_path": str(args.bm25_index.resolve()),
            "bm25_index_sha256": file_sha256(args.bm25_index),
            "side_kv_manifest_path": str(args.side_kv_manifest.resolve()),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "e0_final_report_path": str(args.e0_final_report.resolve()),
            "e0_final_report_sha256": file_sha256(args.e0_final_report),
        },
        "catalogs": [catalog.to_dict() for catalog in catalogs],
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
        f"[e1a-catalog] representative_count={len(representative.memory_ids)} "
        f"tokens={representative.token_count} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
