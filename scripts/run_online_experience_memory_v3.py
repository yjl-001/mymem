#!/usr/bin/env python3
"""Run one V3 full-prefix embedding→replaceable-side-KV generation."""

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

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.risk import ENTROPY_RISK_ARTIFACT_SCHEMA
from memgen.experience.v3 import (
    ExperienceMemoryV3Profile,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_selector import load_margin_selector_calibration
from memgen.experience.v3_artifacts import (
    authenticate_e0_inputs,
    load_formal_e0_report,
    load_v3_offline_report,
    validate_cross_bank_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--retrieval-key-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--v3-offline-report", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--risk-artifact", type=Path, required=True)
    parser.add_argument("--selector-calibration", type=Path)
    parser.add_argument(
        "--retrieval-embedding-transform",
        choices=(
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
        ),
        default=V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=GSM8K_PROMPT_CONTRACT.max_new_tokens,
    )
    parser.add_argument("--save-query-embeddings", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("V3 max-new-tokens must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.retrieval_keys import (
        EmbeddingMemoryRetriever,
        FullPrefixQueryEncoder,
        RetrievalKeyBankLoader,
    )
    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader
    from memgen.model.v3_runtime import (
        EntropyHysteresisGate,
        OnlineExperienceMemorySystemV3,
    )

    selector_calibration = None
    if args.selector_calibration is not None:
        selector_calibration = load_margin_selector_calibration(
            args.selector_calibration
        )
        if selector_calibration.get("source", {}).get(
            "retrieval_key_manifest_sha256"
        ) != file_sha256(args.retrieval_key_manifest):
            raise ValueError(
                "V3 selector calibration uses a different retrieval key bank"
            )
        calibration_transform = selector_calibration.get("source", {}).get(
            "retrieval_embedding_transform",
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
        )
        if calibration_transform != args.retrieval_embedding_transform:
            raise ValueError(
                "Selector calibration uses a different retrieval embedding transform"
            )
        profile = ExperienceMemoryV3Profile(
            retrieval_embedding_transform=args.retrieval_embedding_transform,
            retrieval_abstention_policy="top1_top2_margin",
            retrieval_min_top1_top2_margin=float(
                selector_calibration["calibration"][
                    "minimum_top1_top2_margin"
                ]
            ),
        )
    else:
        profile = ExperienceMemoryV3Profile(
            retrieval_embedding_transform=args.retrieval_embedding_transform
        )
    e0_report = load_formal_e0_report(args.e0_final_report)
    authenticate_e0_inputs(
        e0_report=e0_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
    )
    load_v3_offline_report(
        args.v3_offline_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
        retrieval_key_manifest_path=args.retrieval_key_manifest,
        e0_final_report_path=args.e0_final_report,
    )
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    key_manifest = json.loads(
        args.retrieval_key_manifest.read_text(encoding="utf-8")
    )
    validate_cross_bank_metadata(
        records=records,
        side_manifest=side_manifest,
        key_manifest=key_manifest,
    )
    reasoner = side_manifest["reasoner"]

    risk_artifact = torch.load(
        args.risk_artifact, map_location="cpu", weights_only=False
    )
    if risk_artifact.get("schema_version") != ENTROPY_RISK_ARTIFACT_SCHEMA:
        raise ValueError("V3 requires the canonical SDPA entropy-risk artifact")
    if risk_artifact.get("prompt_contract") != GSM8K_PROMPT_CONTRACT.metadata(
        chat_template=CONVERSATION_TEMPLATE
    ):
        raise ValueError("V3 risk artifact uses a different prompt contract")
    heldout = risk_artifact.get("risk_gate", {}).get("heldout_diagnostic", {})
    if float(heldout.get("heldout_roc_auc", 0.0)) < float(
        heldout.get("minimum_heldout_roc_auc", 1.0)
    ):
        raise ValueError("V3 risk artifact did not pass held-out diagnostics")
    risk_reasoner = risk_artifact.get("reasoner", {})
    for field_name in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_reasoner.get(field_name) != reasoner.get(field_name):
            raise ValueError("V3 risk and memory reasoner provenance differs")
    gate = EntropyHysteresisGate.from_artifact(risk_artifact)

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
        attn_implementation="sdpa",
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
        raise ValueError("Resolved V3 online reasoner revision drifted")

    key_bank = RetrievalKeyBankLoader(
        manifest_path=args.retrieval_key_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    side_loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    side_entries = {
        str(entry["memory_id"]): entry for entry in side_manifest["records"]
    }
    retriever = EmbeddingMemoryRetriever(
        key_bank=key_bank,
        records=records,
        kv_valid_slot_counts={
            memory_id: int(entry["kv_valid_slot_count"])
            for memory_id, entry in side_entries.items()
        },
        profile=profile,
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=profile.layer_number,
        audit_canonical_rope=False,
        memory_score_normalization=profile.memory_score_normalization,
        memory_score_bias=profile.memory_score_bias,
    )
    system = OnlineExperienceMemorySystemV3(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        gate=gate,
        query_encoder=FullPrefixQueryEncoder(
            model=model, device=args.device, layer_number=profile.layer_number
        ),
        retriever=retriever,
        loader=side_loader,
        controller=controller,
        profile=profile,
    )
    prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, args.question)
    started = time.perf_counter()
    try:
        result = system.generate(prompt_token_ids=prompt_ids)
    finally:
        controller.close()
    runtime_seconds = time.perf_counter() - started
    completion = tokenizer.decode(
        list(result.completion_token_ids), skip_special_tokens=True
    ).strip()

    query_sidecar: dict[str, Any] | None = None
    if args.save_query_embeddings and result.query_embeddings:
        from safetensors.torch import save_file

        sidecar_path = args.output.with_name(
            f"{args.output.stem}.query_embeddings.safetensors"
        )
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                f"attempt_{index:02d}": embedding.contiguous()
                for index, embedding in enumerate(result.query_embeddings, start=1)
            },
            str(sidecar_path),
            metadata={
                "schema_version": "experience-memory-v3-query-embeddings-v1",
                "representation": (
                    "raw_unit_before_retrieval_embedding_transform"
                ),
            },
        )
        query_sidecar = {
            "path": sidecar_path.name,
            "sha256": file_sha256(sidecar_path),
            "attempt_count": len(result.query_embeddings),
        }

    output = {
        "schema_version": "experience-memory-v3-online-generation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "answer_or_reward_used": False,
        "question_sha256": text_sha256(args.question),
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
        "generation_contract": system.decoding.config.to_dict() | {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "batch_size": 1,
            "implementation": "explicit_live_native_kv_cache",
        },
        "system_profile": profile.to_dict(),
        "retrieval_embedding_space": retriever.embedding_space_audit,
        "selector_calibration": (
            selector_calibration if selector_calibration is not None else None
        ),
        "hysteresis_gate": gate.config.to_dict(),
        "hysteresis_threshold_provenance": {
            key: risk_artifact.get("construction", {}).get(key)
            for key in (
                "high_entropy_quantile",
                "high_entropy_threshold",
                "low_entropy_quantile",
                "low_entropy_threshold",
                "sink_token_count",
            )
        },
        "risk_diagnostic_qualification": {
            "heldout_roc_auc": heldout.get("heldout_roc_auc"),
            "minimum_heldout_roc_auc": heldout.get("minimum_heldout_roc_auc"),
            "online_control_role": "diagnostic_only",
        },
        "completion": completion,
        "runtime_seconds": runtime_seconds,
        "result": result.to_dict(),
        "query_embedding_sidecar": query_sidecar,
        "inputs": {
            "memory_records_sha256": file_sha256(args.memory_records),
            "retrieval_key_manifest_sha256": file_sha256(
                args.retrieval_key_manifest
            ),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "v3_offline_report_sha256": file_sha256(args.v3_offline_report),
            "e0_final_report_sha256": file_sha256(args.e0_final_report),
            "risk_artifact_sha256": file_sha256(args.risk_artifact),
            "selector_calibration_sha256": (
                file_sha256(args.selector_calibration)
                if args.selector_calibration is not None
                else None
            ),
        },
    }
    output["output_sha256"] = canonical_json_sha256(output)
    write_json(args.output, output)
    print(
        f"[v3-online] attempts={result.retrieval_attempt_count} "
        f"rearms={result.rearm_count} replacements={result.replacement_count} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
