#!/usr/bin/env python3
"""Run one answer-blind online gate→retrieval→persistent-side-KV generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.phase2 import (
    STEERING_VECTOR_ARTIFACT_SCHEMA,
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
    parser.add_argument("--question", required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--risk-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    question = args.question.strip()
    if not question:
        raise ValueError("Online experience system requires a non-empty question")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import EntropyRiskGate
    from memgen.model.experience_system import OnlineExperienceMemorySystem
    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader

    profile = ExperienceMemorySystemProfile()
    e0_final = json.loads(args.e0_final_report.read_text(encoding="utf-8"))
    expected_e0_hash = e0_final.get("final_report_sha256")
    actual_e0_hash = canonical_json_sha256({
        key: value
        for key, value in e0_final.items()
        if key != "final_report_sha256"
    })
    if expected_e0_hash != actual_e0_hash:
        raise ValueError("E0 final report hash mismatch")
    if (
        e0_final.get("status") != "passed"
        or e0_final.get("formal_e0_passed") is not True
        or e0_final.get("task_accuracy_used") is not False
        or not e0_final.get("requirements")
        or not all(
            value is True for value in e0_final["requirements"].values()
        )
    ):
        raise ValueError("Online system requires a formally passed E0 report")
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    index = BM25MemoryIndex.from_dict(
        records=records,
        value=json.loads(args.bm25_index.read_text(encoding="utf-8")),
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    if side_manifest.get("manifest_sha256") != canonical_json_sha256({
        key: value
        for key, value in side_manifest.items()
        if key != "manifest_sha256"
    }):
        raise ValueError("Side-KV manifest hash mismatch")
    if int(side_manifest.get("layer_number", -1)) != profile.layer_number:
        raise ValueError("Side-KV layer differs from the system profile")
    side_entries = {
        str(entry["memory_id"]): entry for entry in side_manifest["records"]
    }
    if set(side_entries) != {record.memory_id for record in records}:
        raise ValueError("Text and side-KV banks cover different memory IDs")
    for record in records:
        entry = side_entries[record.memory_id]
        if (
            entry.get("payload_hash") != record.payload_hash
            or int(entry.get("kv_valid_slot_count", -1)) != record.token_count
        ):
            raise ValueError("Text and side-KV memory metadata differ")

    risk_artifact = torch.load(
        args.risk_artifact, map_location="cpu", weights_only=False
    )
    if risk_artifact.get("schema_version") != STEERING_VECTOR_ARTIFACT_SCHEMA:
        raise ValueError("Unexpected entropy-risk artifact schema")
    heldout = risk_artifact.get("risk_gate", {}).get("heldout_diagnostic", {})
    if float(heldout.get("heldout_roc_auc", 0.0)) < float(
        heldout.get("minimum_heldout_roc_auc", 1.0)
    ):
        raise ValueError("Entropy-risk artifact did not pass held-out qualification")
    gate = EntropyRiskGate.from_artifact(risk_artifact)
    if gate.config.layer_number != profile.layer_number:
        raise ValueError("Risk gate and system profile layers differ")

    reasoner = side_manifest["reasoner"]
    risk_reasoner = risk_artifact.get("reasoner", {})
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_reasoner.get(field) != reasoner.get(field):
            raise ValueError("Risk and side-KV reasoner provenance differ")
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
    resolved_model = str(
        getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]
    )
    resolved_tokenizer = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or reasoner["tokenizer_revision"]
    )
    if (
        resolved_model != reasoner["model_revision"]
        or resolved_tokenizer != reasoner["tokenizer_revision"]
    ):
        raise ValueError("Resolved model/tokenizer revision drifted")

    retriever = SemanticMemoryRetriever(
        index=index,
        query_builder=RetrievalQueryBuilder(
            tokenizer=tokenizer,
            analyzer=index.analyzer,
            config=RetrievalQueryConfig(
                partial_cot_window_tokens=profile.partial_cot_window_tokens
            ),
        ),
        kv_valid_slot_counts={
            memory_id: int(entry["kv_valid_slot_count"])
            for memory_id, entry in side_entries.items()
        },
        profile=profile,
    )
    loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=profile.layer_number,
        audit_canonical_rope=False,
        memory_score_normalization=profile.memory_score_normalization,
        memory_score_bias=profile.memory_score_bias,
    )
    system = OnlineExperienceMemorySystem(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        gate=gate,
        retriever=retriever,
        loader=loader,
        controller=controller,
        profile=profile,
    )
    prompt = tokenizer.apply_chat_template(
        build_gsm8k_messages(question),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = [
        int(token)
        for token in tokenizer.encode(prompt, add_special_tokens=False)
    ]
    try:
        result = system.generate(
            question=question, prompt_token_ids=prompt_ids
        )
    finally:
        controller.close()
    completion = tokenizer.decode(
        list(result.completion_token_ids), skip_special_tokens=True
    ).strip()
    output = {
        "schema_version": "experience-memory-online-generation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "answer_or_reward_used": False,
        "question_sha256": text_sha256(question),
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
        "completion": completion,
        "completion_token_ids_sha256": canonical_json_sha256(
            list(result.completion_token_ids)
        ),
        "system_profile": profile.to_dict(),
        "result": result.to_dict(),
        "inputs": {
            "memory_records_sha256": file_sha256(args.memory_records),
            "bm25_index_sha256": file_sha256(args.bm25_index),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "e0_final_report_sha256": file_sha256(args.e0_final_report),
            "risk_artifact_sha256": file_sha256(args.risk_artifact),
        },
    }
    _write_json(args.output, output)
    print(
        f"[experience-system] triggered={result.triggered} "
        f"side_kv_applied={result.side_kv_applied} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
