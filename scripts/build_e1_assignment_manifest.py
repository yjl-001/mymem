#!/usr/bin/env python3
"""Build the answer-blind, immutable E1 gate/retrieval assignment manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1 import (
    E1_MANIFEST_SCHEMA,
    E1Assignment,
    MemoryChoice,
)
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
    write_jsonl,
)
from memgen.experience.risk import (
    ENTROPY_RISK_ARTIFACT_SCHEMA,
    build_gsm8k_messages,
)
from memgen.experience.retrieval import (
    BM25MemoryIndex,
    RetrievalQueryBuilder,
    RetrievalQueryConfig,
)
from memgen.experience.system import (
    ExperienceMemorySystemProfile,
    SemanticMemoryRetriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--risk-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
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


def load_hashed_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("manifest_sha256")
    actual = canonical_json_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"created_at", "manifest_sha256"}
        }
    )
    if expected != actual:
        raise ValueError(f"Manifest hash mismatch: {path}")
    return value


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


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit <= 0 or args.max_new_tokens <= 0:
        raise ValueError("E1 requires non-negative offset and positive budgets")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import EntropyRiskGate, GreedyE1Runtime

    e0_final = json.loads(args.e0_final_report.read_text(encoding="utf-8"))
    expected_e0_hash = e0_final.get("final_report_sha256")
    actual_e0_hash = canonical_json_sha256({
        key: value
        for key, value in e0_final.items()
        if key != "final_report_sha256"
    })
    if expected_e0_hash != actual_e0_hash:
        raise ValueError("E0 final report hash mismatch")
    if e0_final.get("status") != "passed" or e0_final.get("formal_e0_passed") is not True:
        raise ValueError("E1 requires a formally passed E0 artifact set")
    if e0_final.get("task_accuracy_used") is not False:
        raise ValueError("E0 artifact does not prove answer-blind mechanism qualification")
    if not e0_final.get("requirements") or not all(
        value is True for value in e0_final["requirements"].values()
    ):
        raise ValueError("E0 final report contains an unmet mechanism requirement")

    split_manifest = load_hashed_manifest(args.split_manifest)
    if not split_manifest.get("overlap_check", {}).get("passed"):
        raise ValueError("GSM8K split manifest did not pass overlap audit")
    dataset_revision = str(split_manifest.get("dataset", {}).get("revision", ""))
    if not dataset_revision:
        raise ValueError("Split manifest is missing dataset revision")
    selected = [
        item
        for item in split_manifest["samples"]
        if item.get("logical_split") == args.logical_split
    ][args.offset :]
    selected = selected[: args.limit]
    if not selected:
        raise ValueError("Selected E1 split is empty")

    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    bm25_value = json.loads(args.bm25_index.read_text(encoding="utf-8"))
    bm25 = BM25MemoryIndex.from_dict(records=records, value=bm25_value)
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    expected_side_hash = side_manifest.get("manifest_sha256")
    actual_side_hash = canonical_json_sha256(
        {
            key: item
            for key, item in side_manifest.items()
            if key != "manifest_sha256"
        }
    )
    if expected_side_hash != actual_side_hash:
        raise ValueError("Side-KV manifest hash mismatch")
    profile = ExperienceMemorySystemProfile()
    if int(side_manifest.get("layer_number", -1)) != profile.layer_number:
        raise ValueError("System profile and side-KV layer differ")
    side_entries = {
        str(item["memory_id"]): item for item in side_manifest.get("records", [])
    }
    if set(side_entries) != {record.memory_id for record in records}:
        raise ValueError("Text index and side-KV bank cover different memory IDs")
    for record in records:
        entry = side_entries[record.memory_id]
        if (
            entry.get("payload_hash") != record.payload_hash
            or int(entry.get("kv_valid_slot_count", -1)) != record.token_count
        ):
            raise ValueError("Text/side-KV memory metadata mismatch")

    risk_artifact = torch.load(
        args.risk_artifact, map_location="cpu", weights_only=False
    )
    if risk_artifact.get("schema_version") != ENTROPY_RISK_ARTIFACT_SCHEMA:
        raise ValueError("Unexpected entropy-risk artifact schema")
    heldout = risk_artifact.get("risk_gate", {}).get("heldout_diagnostic", {})
    if float(heldout.get("heldout_roc_auc", 0.0)) < float(
        heldout.get("minimum_heldout_roc_auc", 1.0)
    ):
        raise ValueError("Entropy-risk artifact did not pass held-out qualification")
    gate = EntropyRiskGate.from_artifact(risk_artifact)
    if gate.config.layer_number != profile.layer_number:
        raise ValueError("System gate and side-KV must use the same layer")

    reasoner = side_manifest.get("reasoner", {})
    model_name = str(reasoner.get("model_name", ""))
    model_revision = str(reasoner.get("model_revision", ""))
    tokenizer_revision = str(reasoner.get("tokenizer_revision", ""))
    risk_reasoner = risk_artifact.get("reasoner", {})
    if (
        risk_reasoner.get("model_name") != model_name
        or risk_reasoner.get("model_revision") != model_revision
        or risk_reasoner.get("tokenizer_revision") != tokenizer_revision
    ):
        raise ValueError("Risk artifact and E0 reasoner provenance differ")

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
        raise ValueError("Runtime model/tokenizer revision differs from E0")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=dataset_revision,
    )
    query_builder = RetrievalQueryBuilder(
        tokenizer=tokenizer,
        analyzer=bm25.analyzer,
        config=RetrievalQueryConfig(
            partial_cot_window_tokens=profile.partial_cot_window_tokens
        ),
    )
    retriever = SemanticMemoryRetriever(
        index=bm25,
        query_builder=query_builder,
        kv_valid_slot_counts={
            memory_id: int(entry["kv_valid_slot_count"])
            for memory_id, entry in side_entries.items()
        },
        profile=profile,
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    assignments: list[E1Assignment] = []
    for position, sample in enumerate(selected, start=1):
        source = dataset[int(sample["source_index"])]
        question = str(source["question"]).strip()
        del source
        if text_sha256(question) != sample["question_sha256"]:
            raise ValueError(f"Question hash mismatch for {sample['sample_id']}")
        prompt_ids = prompt_token_ids(tokenizer, question)
        observation = runtime.generate_observation_only(
            prompt_token_ids=prompt_ids,
            gate=gate,
        )
        retrieval_query: dict[str, Any] | None = None
        matched: MemoryChoice | None = None
        if observation.gate_observation is None:
            abstain_reason = "no_joint_entropy_risk_trigger"
        else:
            completion_prefix = observation.prefix_token_ids[len(prompt_ids) :]
            decision = retriever.retrieve(
                question=question,
                partial_cot_token_ids=completion_prefix,
            )
            retrieval_query = decision.to_dict()
            matched = decision.matched_memory
            abstain_reason = decision.abstain_reason
        assignment = E1Assignment(
            sample_id=str(sample["sample_id"]),
            logical_split=args.logical_split,
            dataset_split=str(sample["dataset_split"]),
            source_index=int(sample["source_index"]),
            question_sha256=str(sample["question_sha256"]),
            prompt_token_count=len(prompt_ids),
            prompt_token_ids_sha256=canonical_json_sha256(prompt_ids),
            observation_completion_token_ids=observation.completion_token_ids,
            observation_completion_token_ids_sha256=canonical_json_sha256(
                list(observation.completion_token_ids)
            ),
            gate_observation=observation.gate_observation,
            prefix_token_ids=observation.prefix_token_ids,
            prefix_token_ids_sha256=(
                canonical_json_sha256(list(observation.prefix_token_ids))
                if observation.prefix_token_ids
                else None
            ),
            retrieval_query=retrieval_query,
            matched_memory=matched,
            abstain_reason=abstain_reason,
        )
        assignments.append(assignment)
        if position % 10 == 0 or position == len(selected):
            print(
                f"[e1-assignment] {position}/{len(selected)} "
                f"triggered={sum(item.triggered for item in assignments)} "
                f"retrieved={sum(item.matched_memory is not None for item in assignments)}",
                flush=True,
            )

    trace_path = args.output.with_name("observation_assignments.jsonl")
    write_jsonl(trace_path, (item.to_dict() for item in assignments))
    frozen_assignments = tuple(assignments)
    abstain_counts: dict[str, int] = {}
    for item in frozen_assignments:
        if item.abstain_reason:
            abstain_counts[item.abstain_reason] = (
                abstain_counts.get(item.abstain_reason, 0) + 1
            )
    manifest: dict[str, Any] = {
        "schema_version": E1_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "answer_or_reward_used": False,
        "logical_split": args.logical_split,
        "reasoner": {
            "model_name": model_name,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "layer": profile.layer_number,
            "dtype": args.dtype,
            "attention_implementation": "eager",
        },
        "configuration": {
            "max_new_tokens": args.max_new_tokens,
            "offset": args.offset,
            "limit": args.limit,
            "memory_count_per_trigger": 1,
            "memory_slot_budget": int(
                side_manifest["tensor_shape"]["keys"][2]
            ),
            "gate": gate.config_dict,
            "system_profile": profile.to_dict(),
            "retrieval": {
                "method": "bm25",
                "top_k_assignment": profile.selected_memory_count,
                "top_k_diagnostic": profile.retrieval_top_k,
                "query": asdict(query_builder.config),
                "analyzer": asdict(bm25.analyzer.config),
                "bm25": asdict(bm25.config),
            },
            "injection_policy": profile.injection_policy,
            "assignment_policy": (
                "observation_only_gate_and_retrieval_then_frozen_replay"
            ),
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
            "risk_artifact_path": str(args.risk_artifact.resolve()),
            "risk_artifact_sha256": file_sha256(args.risk_artifact),
            "dataset_revision": dataset_revision,
        },
        "summary": {
            "sample_count": len(frozen_assignments),
            "triggered_count": sum(item.triggered for item in frozen_assignments),
            "assigned_count": sum(item.assigned for item in frozen_assignments),
            "abstain_reason_counts": dict(sorted(abstain_counts.items())),
        },
        "assignments": [item.to_dict() for item in frozen_assignments],
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        }
    )
    write_json(args.output, manifest)
    report_path = args.report_output or args.output.with_name(
        "assignment_build_report.json"
    )
    report = {
        "schema_version": "experience-memory-e1-assignment-build-report-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "answer_or_reward_used": False,
        "summary": manifest["summary"],
        "observation_trace": {
            "path": trace_path.name,
            "sha256": file_sha256(trace_path),
        },
        "manifest": {
            "path": str(args.output.resolve()),
            "sha256": file_sha256(args.output),
            "logical_sha256": manifest["manifest_sha256"],
        },
    }
    write_json(report_path, report)
    print(
        f"[e1-assignment] frozen samples={len(frozen_assignments)} "
        f"assigned={manifest['summary']['assigned_count']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
