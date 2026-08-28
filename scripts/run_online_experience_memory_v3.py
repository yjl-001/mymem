#!/usr/bin/env python3
"""Run one V3 full-prefix embedding→replaceable-side-KV generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


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
from memgen.experience.risk import (
    ENTROPY_RISK_ARTIFACT_SCHEMA,
    TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA,
)
from memgen.experience.v3 import (
    ExperienceMemoryV3Profile,
    V34_QUERY_POOLING_CURRENT_TOKEN,
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_METHODS,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_selector import (
    load_margin_selector_calibration,
    selector_calibration_query_pooling,
)
from memgen.experience.v3_artifacts import (
    authenticate_e0_inputs,
    load_formal_e0_report,
    load_v3_offline_report,
    validate_cross_bank_metadata,
)


V35_QUERY_SIDECAR_REPRESENTATION = (
    "dynamic_query_l2_normalized_exact_audit"
)
LEGACY_QUERY_SIDECAR_REPRESENTATION = (
    "raw_unit_before_retrieval_embedding_transform"
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
    parser.add_argument(
        "--system-version", choices=("v3", "v3.4", "v3.5"), default="v3"
    )
    parser.add_argument("--selector-calibration", type=Path)
    parser.add_argument(
        "--dual-key-manifest",
        type=Path,
        help="Required V3.5 applicability/dynamic dual-key manifest.",
    )
    parser.add_argument(
        "--applicability-calibration",
        type=Path,
        help="Required V3.5 frozen static-shortlist calibration artifact.",
    )
    parser.add_argument(
        "--calibration-trace-only",
        action="store_true",
        help=(
            "Run the explicit V3.5 answer-blind first-attempt margin trace "
            "profile; a final selector artifact must not be supplied."
        ),
    )
    parser.add_argument(
        "--retrieval-embedding-transform",
        choices=(
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
        ),
        default=V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    parser.add_argument(
        "--query-pooling",
        choices=tuple(sorted(V3_QUERY_POOLING_METHODS)),
        default=None,
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


def _resolve_and_validate_versioned_args(args: argparse.Namespace) -> None:
    """Resolve defaults while keeping every legacy CLI contract unchanged."""

    continuous_version = args.system_version in {"v3.4", "v3.5"}
    if args.query_pooling is None:
        args.query_pooling = (
            V34_QUERY_POOLING_CURRENT_TOKEN
            if continuous_version
            else V3_QUERY_POOLING_BOUNDARY_LAST
        )
    if continuous_version and args.query_pooling != V34_QUERY_POOLING_CURRENT_TOKEN:
        raise ValueError(
            f"{args.system_version} requires current-generated-token query pooling"
        )
    if args.system_version == "v3.5":
        if getattr(args, "dtype", None) != "bfloat16":
            raise ValueError("V3.5 requires --dtype bfloat16")
        if args.dual_key_manifest is None:
            raise ValueError("V3.5 requires --dual-key-manifest")
        if args.applicability_calibration is None:
            raise ValueError("V3.5 requires --applicability-calibration")
        if args.retrieval_embedding_transform != V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE:
            raise ValueError("V3.5 does not permit a legacy retrieval transform")
        if args.calibration_trace_only:
            if args.selector_calibration is not None:
                raise ValueError(
                    "V3.5 calibration trace-only mode cannot use a final selector artifact"
                )
        elif args.selector_calibration is None:
            raise ValueError("Final V3.5 runs require --selector-calibration")
    elif (
        args.dual_key_manifest is not None
        or args.applicability_calibration is not None
        or args.calibration_trace_only
    ):
        raise ValueError("V3.5-only selector arguments require --system-version v3.5")


def _load_v35_profile_and_artifacts(
    args: argparse.Namespace,
) -> tuple[ExperienceMemoryV3Profile, dict[str, Any], dict[str, Any] | None]:
    """Authenticate V3.5 static/final selector artifacts and build its profile."""

    from memgen.experience.v3_5_selector import (
        load_v35_applicability_calibration,
        load_v35_selector_calibration,
    )

    assert args.dual_key_manifest is not None
    assert args.applicability_calibration is not None
    dual_manifest_sha256 = file_sha256(args.dual_key_manifest)
    applicability = load_v35_applicability_calibration(
        args.applicability_calibration,
        expected_input_hashes={
            "dual_key_manifest_sha256": dual_manifest_sha256,
        },
    )
    applicability_sha256 = file_sha256(args.applicability_calibration)
    applicability_values = applicability["calibration"]
    if args.calibration_trace_only:
        profile = ExperienceMemoryV3Profile.applicability_aware_continuous(
            applicability_shortlist_k=int(
                applicability_values["shortlist_k"]
            ),
            applicability_score_floor=float(
                applicability_values["minimum_applicability_score"]
            ),
            retrieval_min_top1_top2_margin=None,
            calibration_trace_only=True,
        )
        return profile, applicability, None

    assert args.selector_calibration is not None
    selector = load_v35_selector_calibration(
        args.selector_calibration,
        expected_input_hashes={
            "dual_key_manifest_sha256": dual_manifest_sha256,
            "applicability_calibration_sha256": applicability_sha256,
            "risk_artifact_sha256": file_sha256(args.risk_artifact),
        },
    )
    selector_values = selector["calibration"]
    if (
        int(selector_values["shortlist_k"])
        != int(applicability_values["shortlist_k"])
        or abs(
            float(selector_values["minimum_applicability_score"])
            - float(applicability_values["minimum_applicability_score"])
        )
        > 1e-12
    ):
        raise ValueError(
            "V3.5 final selector differs from its applicability calibration"
        )
    profile = ExperienceMemoryV3Profile.applicability_aware_continuous(
        applicability_shortlist_k=int(selector_values["shortlist_k"]),
        applicability_score_floor=float(
            selector_values["minimum_applicability_score"]
        ),
        retrieval_min_top1_top2_margin=float(
            selector_values["minimum_dynamic_top1_top2_margin"]
        ),
        calibration_trace_only=False,
    )
    return profile, applicability, selector


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _retrieval_embedding_audit(
    *, retriever: Any, key_bank: Any, dual_key_manifest: Path | None
) -> dict[str, Any]:
    """Return a JSON-safe retrieval-space identity for either bank version."""

    for owner in (retriever, key_bank):
        value = getattr(owner, "embedding_space_audit", None)
        if isinstance(value, Mapping):
            return dict(value)
    if dual_key_manifest is None:
        raise ValueError("V3 retriever did not expose its embedding-space audit")
    return {
        "schema_version": "experience-memory-v3.5-dual-retrieval-space-audit-v1",
        "dual_key_manifest_sha256": file_sha256(dual_key_manifest),
        "loader_authenticated": True,
    }


def _dual_key_artifact_identity(
    *, key_bank: Any, manifest_path: Path | None
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    manifest = getattr(key_bank, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise ValueError("V3.5 dual-key loader did not expose its manifest")
    tensor_artifact = manifest.get("tensor_artifact")
    input_artifacts = manifest.get("input_artifacts")
    if not isinstance(tensor_artifact, Mapping) or not isinstance(
        input_artifacts, Mapping
    ):
        raise ValueError("V3.5 dual-key artifact identity is incomplete")
    return {
        "schema_version": manifest.get("schema_version"),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_logical_sha256": manifest.get("manifest_sha256"),
        "tensor_artifact": dict(tensor_artifact),
        "input_artifacts": dict(input_artifacts),
    }


def prepare_v35_query_sidecar_embeddings(
    *,
    query_embeddings: Sequence[Any],
    runtime_trace: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Validate and return the exact unit vectors hashed by V3.5 retrieval."""

    import torch

    from memgen.model.retrieval_keys import tensor_sha256

    attempts = runtime_trace.get("retrieval_attempts")
    if not isinstance(attempts, list) or len(query_embeddings) != len(attempts):
        raise RuntimeError("V3.5 query embedding sidecar/attempt counts differ")
    canonical_embeddings: list[Any] = []
    for query_embedding, attempt_trace in zip(query_embeddings, attempts):
        if not isinstance(attempt_trace, Mapping):
            raise RuntimeError("V3.5 query embedding attempt trace is malformed")
        decision = attempt_trace.get("retrieval_decision")
        query_audit = (
            decision.get("query", {})
            if isinstance(decision, Mapping)
            else {}
        )
        if not isinstance(query_audit, Mapping):
            query_audit = {}
        canonical = (
            query_embedding.detach().float().cpu().reshape(-1).contiguous()
        )
        if (
            not bool(torch.isfinite(canonical).all().item())
            or query_audit.get("query_embedding_sha256")
            != tensor_sha256(canonical)
            or not math.isclose(
                float(canonical.norm().item()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            raise RuntimeError(
                "V3.5 canonical query sidecar differs from retrieval audit"
            )
        canonical_embeddings.append(canonical)
    return tuple(canonical_embeddings)


def main() -> None:
    args = parse_args()
    _resolve_and_validate_versioned_args(args)
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
    applicability_calibration = None
    if args.system_version == "v3.5":
        (
            profile,
            applicability_calibration,
            selector_calibration,
        ) = _load_v35_profile_and_artifacts(args)
    elif args.selector_calibration is not None:
        selector_calibration = load_margin_selector_calibration(
            args.selector_calibration
        )
        if selector_calibration.get("source", {}).get(
            "retrieval_key_manifest_sha256"
        ) != file_sha256(args.retrieval_key_manifest):
            raise ValueError(
                "V3 selector calibration uses a different retrieval key bank"
            )
        calibration_risk_sha256 = selector_calibration.get("source", {}).get(
            "risk_artifact_sha256"
        )
        if (
            args.system_version == "v3.4"
            and calibration_risk_sha256 != file_sha256(args.risk_artifact)
        ):
            raise ValueError(
                "V3.4 selector calibration uses a different token-risk artifact"
            )
        if selector_calibration.get("source", {}).get(
            "system_version", "v3"
        ) != args.system_version:
            raise ValueError(
                "Selector calibration uses a different system version"
            )
        calibration_transform = selector_calibration.get("source", {}).get(
            "retrieval_embedding_transform",
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
        )
        if calibration_transform != args.retrieval_embedding_transform:
            raise ValueError(
                "Selector calibration uses a different retrieval embedding transform"
            )
        if selector_calibration_query_pooling(
            selector_calibration
        ) != args.query_pooling:
            raise ValueError(
                "Selector calibration uses a different query pooling policy"
            )
        profile_kwargs = {
            "retrieval_embedding_transform": (
                args.retrieval_embedding_transform
            ),
            "retrieval_abstention_policy": "top1_top2_margin",
            "retrieval_min_top1_top2_margin": float(
                selector_calibration["calibration"][
                    "minimum_top1_top2_margin"
                ]
            ),
        }
        profile = (
            ExperienceMemoryV3Profile.continuous_token_joint(**profile_kwargs)
            if args.system_version == "v3.4"
            else ExperienceMemoryV3Profile(
                query_pooling=args.query_pooling, **profile_kwargs
            )
        )
    else:
        profile = (
            ExperienceMemoryV3Profile.continuous_token_joint(
                retrieval_embedding_transform=(
                    args.retrieval_embedding_transform
                )
            )
            if args.system_version == "v3.4"
            else ExperienceMemoryV3Profile(
                query_pooling=args.query_pooling,
                retrieval_embedding_transform=(
                    args.retrieval_embedding_transform
                ),
            )
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
    continuous_version = args.system_version in {"v3.4", "v3.5"}
    expected_risk_schema = (
        TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA
        if continuous_version
        else ENTROPY_RISK_ARTIFACT_SCHEMA
    )
    if risk_artifact.get("schema_version") != expected_risk_schema:
        raise ValueError(
            f"{args.system_version} requires its canonical risk artifact"
        )
    if risk_artifact.get("prompt_contract") != GSM8K_PROMPT_CONTRACT.metadata(
        chat_template=CONVERSATION_TEMPLATE
    ):
        raise ValueError("V3 risk artifact uses a different prompt contract")
    heldout = risk_artifact.get("risk_gate", {}).get("heldout_diagnostic", {})
    if (
        float(heldout.get("heldout_roc_auc", 0.0))
        < float(heldout.get("minimum_heldout_roc_auc", 1.0))
        or continuous_version
        and risk_artifact.get("qualification", {}).get("passed") is not True
    ):
        raise ValueError("V3 risk artifact did not pass held-out diagnostics")
    risk_reasoner = risk_artifact.get("reasoner", {})
    for field_name in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_reasoner.get(field_name) != reasoner.get(field_name):
            raise ValueError("V3 risk and memory reasoner provenance differs")
    gate = (
        EntropyHysteresisGate.from_token_artifact(risk_artifact)
        if continuous_version
        else EntropyHysteresisGate.from_artifact(risk_artifact)
    )

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

    side_loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    side_entries = {
        str(entry["memory_id"]): entry for entry in side_manifest["records"]
    }
    kv_valid_slot_counts = {
        memory_id: int(entry["kv_valid_slot_count"])
        for memory_id, entry in side_entries.items()
    }
    if args.system_version == "v3.5":
        from memgen.model.v3_5_retrieval import (
            ApplicabilityAwareMemoryRetriever,
            DualRetrievalKeyBankLoader,
            QuestionOnlyEncoder,
        )

        assert args.dual_key_manifest is not None
        # Re-open the legacy applicability source bank as an authenticated
        # artifact.  The dual bank is allowed to reproduce it, never merely
        # cite a path or an unchecked manifest hash.
        RetrievalKeyBankLoader(
            manifest_path=args.retrieval_key_manifest,
            expected_reasoner_name=reasoner["model_name"],
            expected_reasoner_revision=reasoner["model_revision"],
            expected_tokenizer_revision=reasoner["tokenizer_revision"],
        )
        legacy_tensor_sha256 = str(
            key_manifest.get("tensor_artifact", {}).get("sha256", "")
        )
        if not legacy_tensor_sha256:
            raise ValueError("V3 applicability source tensor hash is missing")
        key_bank = DualRetrievalKeyBankLoader(
            manifest_path=args.dual_key_manifest,
            expected_reasoner_name=reasoner["model_name"],
            expected_reasoner_revision=reasoner["model_revision"],
            expected_tokenizer_revision=reasoner["tokenizer_revision"],
            expected_input_hashes={
                "memory_records_sha256": file_sha256(args.memory_records),
                "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
                "e0_final_report_sha256": file_sha256(args.e0_final_report),
                "v3_retrieval_key_manifest_sha256": file_sha256(
                    args.retrieval_key_manifest
                ),
                "v3_retrieval_key_tensor_sha256": legacy_tensor_sha256,
                "v3_offline_report_sha256": file_sha256(
                    args.v3_offline_report
                ),
            },
        )
        retriever = ApplicabilityAwareMemoryRetriever(
            key_bank=key_bank,
            records=records,
            kv_valid_slot_counts=kv_valid_slot_counts,
            question_encoder=QuestionOnlyEncoder(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                layer_number=profile.layer_number,
            ),
            shortlist_k=int(profile.applicability_shortlist_k),
            applicability_score_floor=float(
                profile.applicability_score_floor
            ),
            dynamic_min_top1_top2_margin=(
                profile.retrieval_min_top1_top2_margin
            ),
            profile=profile,
        )
    else:
        key_bank = RetrievalKeyBankLoader(
            manifest_path=args.retrieval_key_manifest,
            expected_reasoner_name=reasoner["model_name"],
            expected_reasoner_revision=reasoner["model_revision"],
            expected_tokenizer_revision=reasoner["tokenizer_revision"],
        )
        retriever = EmbeddingMemoryRetriever(
            key_bank=key_bank,
            records=records,
            kv_valid_slot_counts=kv_valid_slot_counts,
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
            model=model,
            device=args.device,
            layer_number=profile.layer_number,
            query_pooling=profile.query_pooling,
        ),
        retriever=retriever,
        loader=side_loader,
        controller=controller,
        profile=profile,
    )
    prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, args.question)
    started = time.perf_counter()
    try:
        result = system.generate(
            prompt_token_ids=prompt_ids,
            question=(args.question.strip() if args.system_version == "v3.5" else None),
        )
    finally:
        controller.close()
    runtime_seconds = time.perf_counter() - started
    completion = tokenizer.decode(
        list(result.completion_token_ids), skip_special_tokens=True
    ).strip()

    result_payload = result.to_dict()
    query_sidecar: dict[str, Any] | None = None
    if args.save_query_embeddings and result.query_embeddings:
        from safetensors.torch import save_file

        sidecar_embeddings = tuple(result.query_embeddings)
        sidecar_representation = LEGACY_QUERY_SIDECAR_REPRESENTATION
        if args.system_version == "v3.5":
            sidecar_embeddings = prepare_v35_query_sidecar_embeddings(
                query_embeddings=result.query_embeddings,
                runtime_trace=result_payload,
            )
            sidecar_representation = V35_QUERY_SIDECAR_REPRESENTATION
        sidecar_path = args.output.with_name(
            f"{args.output.stem}.query_embeddings.safetensors"
        )
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                f"attempt_{index:02d}": embedding.contiguous()
                for index, embedding in enumerate(sidecar_embeddings, start=1)
            },
            str(sidecar_path),
            metadata={
                "schema_version": (
                    "experience-memory-v3.5-query-embeddings-v1"
                    if args.system_version == "v3.5"
                    else "experience-memory-v3-query-embeddings-v1"
                ),
                "representation": sidecar_representation,
            },
        )
        query_sidecar = {
            "path": sidecar_path.name,
            "sha256": file_sha256(sidecar_path),
            "attempt_count": len(sidecar_embeddings),
            "representation": sidecar_representation,
        }

    retrieval_space_audit = _retrieval_embedding_audit(
        retriever=retriever,
        key_bank=key_bank,
        dual_key_manifest=args.dual_key_manifest,
    )
    dual_key_artifact = _dual_key_artifact_identity(
        key_bank=key_bank,
        manifest_path=args.dual_key_manifest,
    )
    output = {
        "schema_version": (
            "experience-memory-v3.5-online-generation-v1"
            if args.system_version == "v3.5"
            else "experience-memory-v3-online-generation-v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "answer_or_reward_used": False,
        "task_accuracy_used": False,
        "task_results_used_for_selector_decision": False,
        "system_version": args.system_version,
        "runtime_dtype": args.dtype,
        "calibration_trace_only": bool(args.calibration_trace_only),
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
        "system_profile_sha256": canonical_json_sha256(profile.to_dict()),
        "retrieval_embedding_space": retrieval_space_audit,
        "dual_key_artifact": dual_key_artifact,
        "applicability_calibration": (
            applicability_calibration
            if applicability_calibration is not None
            else None
        ),
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
            "heldout_balanced_accuracy_at_train_threshold": heldout.get(
                "heldout_balanced_accuracy_at_train_threshold"
            ),
            "train_threshold_calibration": heldout.get(
                "train_threshold_calibration"
            ),
            "online_control_role": profile.risk_role,
        },
        "completion": completion,
        "runtime_seconds": runtime_seconds,
        "result": result_payload,
        "static_selector_trace": (
            result_payload.get("static_selector_trace")
            if args.system_version == "v3.5"
            else None
        ),
        "terminal_lifecycle_diagnostics": (
            {
                key: result_payload.get("summary", {}).get(key)
                for key in (
                    "terminal_abstain_count",
                    "clear_on_terminal_abstain_count",
                    "no_rearm_after_terminal_abstain",
                    "stale_memory_attention_after_terminal_clear_count",
                    "terminal_clear_attention_safe",
                    "final_gate_state",
                    "final_memory_id",
                )
            }
            if args.system_version == "v3.5"
            else None
        ),
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
            "dual_key_manifest_sha256": (
                file_sha256(args.dual_key_manifest)
                if args.dual_key_manifest is not None
                else None
            ),
            "applicability_calibration_sha256": (
                file_sha256(args.applicability_calibration)
                if args.applicability_calibration is not None
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
