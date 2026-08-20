#!/usr/bin/env python3
"""Audit E0 side-KV visibility, disabled parity, RoPE use, and cache safety."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl, write_jsonl


_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSONL rows: case_id, memory_id, prefix_token_ids including the trigger boundary.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compile-report",
        type=Path,
        help="Defaults to e0_report.json beside the side-KV manifest.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--parity-atol", type=float, default=1e-6)
    parser.add_argument(
        "--canonical-rope-rtol",
        type=float,
        default=2e-2,
        help="Fixed shared-phase score tolerance accounting for bfloat16 RoPE tables.",
    )
    parser.add_argument(
        "--min-active-logits-kl",
        type=float,
        default=1e-8,
        help=(
            "At least one case must exceed this fixed numerical-noise floor. "
            "This checks mechanism influence, not task correctness."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def validate_compile_report(
    *,
    report_path: Path,
    side_kv_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not report_path.is_file():
        raise ValueError(f"E0 compile report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "kv_compilation_passed_pending_runtime_audit":
        raise ValueError("E0 compile report is not ready for runtime-audit finalization")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("E0 compile report has no artifact map")

    verified: dict[str, str] = {}
    for name in (
        "memory_records",
        "compilation_trace",
        "payload_audit",
        "bm25_index",
    ):
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"E0 compile report is missing artifact {name}")
        artifact_path = report_path.parent / str(entry.get("path", ""))
        expected_sha256 = str(entry.get("sha256", ""))
        if not artifact_path.is_file() or file_sha256(artifact_path) != expected_sha256:
            raise ValueError(f"E0 compile artifact failed hash validation: {name}")
        verified[name] = expected_sha256

    side_kv = artifacts.get("side_kv")
    if not isinstance(side_kv, dict):
        raise ValueError("E0 compile report has no side-KV artifact")
    for name, path_field, hash_field in (
        ("side_kv_tensor", "tensor_path", "tensor_sha256"),
        ("side_kv_manifest", "manifest_path", "manifest_sha256"),
    ):
        artifact_path = report_path.parent / str(side_kv.get(path_field, ""))
        expected_sha256 = str(side_kv.get(hash_field, ""))
        if not artifact_path.is_file() or file_sha256(artifact_path) != expected_sha256:
            raise ValueError(f"E0 compile artifact failed hash validation: {name}")
        verified[name] = expected_sha256
        if name == "side_kv_manifest" and (
            artifact_path.resolve() != side_kv_manifest_path.resolve()
        ):
            raise ValueError("Audited side-KV manifest differs from the compile report")

    expected_set_hash = canonical_json_sha256(
        {
            "records": verified["memory_records"],
            "trace": verified["compilation_trace"],
            "audit": verified["payload_audit"],
            "index": verified["bm25_index"],
            "side_kv": side_kv,
        }
    )
    if report.get("artifact_set_sha256") != expected_set_hash:
        raise ValueError("E0 compile artifact-set hash mismatch")
    return report, verified


def clone_cache(cache: Any) -> Any:
    try:
        return copy.deepcopy(cache)
    except Exception:
        legacy = cache.to_legacy_cache()
        cloned = tuple(tuple(tensor.clone() for tensor in layer) for layer in legacy)
        constructor = getattr(type(cache), "from_legacy_cache", None)
        if not callable(constructor):
            raise RuntimeError("Unable to clone the native KV cache")
        return constructor(cloned)


def cache_layers(cache: Any) -> list[tuple[Any, Any]]:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [
            (layer.keys, layer.values)
            for layer in layers
            if getattr(layer, "keys", None) is not None
        ]
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is not None and values is not None:
        return list(zip(keys, values))
    legacy = cache.to_legacy_cache()
    return [(layer[0], layer[1]) for layer in legacy]


def cache_lengths(cache: Any) -> list[int]:
    return [int(keys.shape[-2]) for keys, _ in cache_layers(cache)]


def cache_prefix_preserved(before: Any, after: Any) -> bool:
    before_layers = cache_layers(before)
    after_layers = cache_layers(after)
    if len(before_layers) != len(after_layers):
        return False
    for (before_key, before_value), (after_key, after_value) in zip(
        before_layers, after_layers
    ):
        prefix_length = before_key.shape[-2]
        if not (
            after_key.shape[-2] == prefix_length + 1
            and after_value.shape[-2] == prefix_length + 1
            and before_key.equal(after_key[..., :prefix_length, :])
            and before_value.equal(after_value[..., :prefix_length, :])
        ):
            return False
    return True


def logits_kl(reference: Any, treatment: Any) -> float:
    import torch

    reference_log_probs = torch.log_softmax(reference.float(), dim=-1)
    treatment_log_probs = torch.log_softmax(treatment.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    value = (reference_probs * (reference_log_probs - treatment_log_probs)).sum(dim=-1)
    return float(value.mean().item())


def validate_case(record: dict[str, Any]) -> tuple[str, str, list[int], int]:
    case_id = str(record.get("case_id", ""))
    memory_id = str(record.get("memory_id", ""))
    prefix = record.get("prefix_token_ids")
    prompt_token_count = record.get("prompt_token_count")
    if record.get("schema_version") != "side-kv-mechanism-audit-case-input-v1":
        raise ValueError(f"Audit case {case_id} has an unexpected schema")
    if record.get("logical_split") != "calibration-val":
        raise ValueError(f"Audit case {case_id} is not from calibration-val")
    if record.get("answer_or_reward_used") is not False:
        raise ValueError(f"Audit case {case_id} is not marked answer-blind")
    if record.get("selection_policy") != "first_preanswer_reasoning_delimiter":
        raise ValueError(f"Audit case {case_id} has an unexpected selection policy")
    if not str(record.get("question_sha256", "")):
        raise ValueError(f"Audit case {case_id} has no question hash")
    if not case_id or not memory_id:
        raise ValueError("Every audit case requires case_id and memory_id")
    if not isinstance(prefix, list) or len(prefix) < 2:
        raise ValueError(f"Audit case {case_id} needs at least two prefix_token_ids")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in prefix):
        raise ValueError(f"Audit case {case_id} has invalid prefix_token_ids")
    if record.get("prefix_token_ids_sha256") != canonical_json_sha256(prefix):
        raise ValueError(f"Audit case {case_id} has a prefix hash mismatch")
    if (
        isinstance(prompt_token_count, bool)
        or not isinstance(prompt_token_count, int)
        or prompt_token_count <= 0
        or prompt_token_count >= len(prefix)
    ):
        raise ValueError(f"Audit case {case_id} has an invalid prompt token count")
    if record.get("generated_boundary_index") != len(prefix) - prompt_token_count - 1:
        raise ValueError(f"Audit case {case_id} has an inconsistent boundary index")
    if record.get("boundary_token_id") != prefix[-1]:
        raise ValueError(f"Audit case {case_id} has an inconsistent boundary token")
    return case_id, memory_id, prefix, prompt_token_count


def main() -> None:
    args = parse_args()
    if args.layer != 24:
        raise ValueError("E0-v1 is frozen to layer 24")
    if (
        args.max_cases < 0
        or args.parity_atol < 0
        or args.canonical_rope_rtol < 0
        or args.min_active_logits_kl < 0
    ):
        raise ValueError("Invalid audit limits or tolerance")

    compile_report_path = args.compile_report or args.side_kv_manifest.with_name(
        "e0_report.json"
    )
    compile_report, verified_compile_artifacts = validate_compile_report(
        report_path=compile_report_path,
        side_kv_manifest_path=args.side_kv_manifest,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader

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
    resolved_model_revision = str(
        getattr(model.config, "_commit_hash", None) or args.model_revision
    )
    loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=args.model,
        expected_reasoner_revision=resolved_model_revision,
        expected_tokenizer_revision=args.tokenizer_revision,
    )
    compiled_tokenizer_revision = str(
        loader.manifest.get("reasoner", {}).get("tokenizer_revision", "")
    )
    if not compiled_tokenizer_revision:
        raise ValueError("Side-KV manifest is missing its tokenizer revision")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=compiled_tokenizer_revision,
    )
    resolved_tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or compiled_tokenizer_revision
    )
    if resolved_tokenizer_revision != compiled_tokenizer_revision:
        raise ValueError("Runtime tokenizer revision differs from the side-KV compiler")
    cases = list(iter_jsonl(args.cases))
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("No side-KV mechanism audit cases were provided")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in cases:
        case_id, memory_id, prefix, prompt_token_count = validate_case(row)
        completion_prefix = tokenizer.decode(
            prefix[prompt_token_count:],
            skip_special_tokens=False,
        )
        boundary_text = tokenizer.decode(
            [prefix[-1]],
            skip_special_tokens=False,
        )
        if _ANSWER_MARKER_RE.search(completion_prefix):
            raise ValueError(f"Audit case {case_id} reaches an answer marker")
        if not boundary_text.rstrip(" \t").endswith((",", ".", "\n")):
            raise ValueError(f"Audit case {case_id} does not end at a reasoning delimiter")
        prefix_without_boundary = torch.tensor(
            [prefix[:-1]], dtype=torch.long, device=args.device
        )
        boundary_token = torch.tensor(
            [[prefix[-1]]], dtype=torch.long, device=args.device
        )
        prefill_mask = torch.ones_like(prefix_without_boundary)
        full_mask = torch.ones(
            (1, len(prefix)), dtype=prefill_mask.dtype, device=args.device
        )
        with torch.inference_mode():
            prefill = model(
                input_ids=prefix_without_boundary,
                attention_mask=prefill_mask,
                use_cache=True,
                return_dict=True,
            )
        prefix_cache = prefill.past_key_values

        baseline_cache = clone_cache(prefix_cache)
        with torch.inference_mode():
            baseline = model(
                input_ids=boundary_token,
                attention_mask=full_mask,
                past_key_values=baseline_cache,
                use_cache=True,
                return_dict=True,
            )

        controller = SideKVAttentionController(
            model=model,
            layer_number=args.layer,
            audit_canonical_rope=True,
        )
        try:
            inactive_cache = clone_cache(prefix_cache)
            with torch.inference_mode():
                inactive = model(
                    input_ids=boundary_token,
                    attention_mask=full_mask,
                    past_key_values=inactive_cache,
                    use_cache=True,
                    return_dict=True,
                )
            inactive_max_abs = float(
                (baseline.logits.float() - inactive.logits.float()).abs().max().item()
            )

            memory_cache = clone_cache(prefix_cache)
            memory = loader.get(
                memory_id,
                device=boundary_token.device,
                dtype=next(model.parameters()).dtype,
            )
            controller.clear_traces()
            with controller.use_memory(memory), torch.inference_mode():
                treatment = model(
                    input_ids=boundary_token,
                    attention_mask=full_mask,
                    past_key_values=memory_cache,
                    use_cache=True,
                    return_dict=True,
                )
            if len(controller.traces) != 1:
                raise RuntimeError(
                    f"Audit case {case_id} expected one side-KV attention trace"
                )
            attention_trace = controller.traces[0]
        finally:
            controller.close()

        prefix_lengths = cache_lengths(prefix_cache)
        baseline_lengths = cache_lengths(baseline.past_key_values)
        treatment_lengths = cache_lengths(treatment.past_key_values)
        length_invariant = bool(
            prefix_lengths
            and len(prefix_lengths) == len(baseline_lengths) == len(treatment_lengths)
            and all(after == before + 1 for before, after in zip(prefix_lengths, baseline_lengths))
            and treatment_lengths == baseline_lengths
        )
        prefix_preserved = cache_prefix_preserved(
            prefix_cache, treatment.past_key_values
        )
        kl = logits_kl(baseline.logits[:, -1, :], treatment.logits[:, -1, :])
        case_passed = bool(
            inactive_max_abs <= args.parity_atol
            and length_invariant
            and prefix_preserved
            and attention_trace.memory_attention_mass > 0
            and attention_trace.canonical_rope_score_relative_error is not None
            and attention_trace.canonical_rope_score_relative_error
            <= args.canonical_rope_rtol
            and math.isfinite(kl)
        )
        records.append(
            {
                "schema_version": "side-kv-mechanism-audit-case-v1",
                "case_id": case_id,
                "memory_id": memory_id,
                "prefix_token_ids_sha256": canonical_json_sha256(prefix),
                "prefix_length": len(prefix),
                "prompt_token_count": prompt_token_count,
                "case_source_audit_passed": True,
                "disabled_path_max_abs_logit_difference": inactive_max_abs,
                "disabled_path_parity_tolerance": args.parity_atol,
                "native_cache_lengths_before_boundary": prefix_lengths,
                "baseline_cache_lengths_after_boundary": baseline_lengths,
                "memory_cache_lengths_after_boundary": treatment_lengths,
                "native_cache_length_invariant": length_invariant,
                "native_cache_prefix_preserved": prefix_preserved,
                "memory_attention": attention_trace.to_dict(),
                "canonical_rope_relative_tolerance": args.canonical_rope_rtol,
                "first_step_logits_kl_baseline_to_memory": kl,
                "passed": case_passed,
            }
        )

    trace_path = output_dir / "mechanism_audit_trace.jsonl"
    report_path = output_dir / "mechanism_audit_report.json"
    final_report_path = output_dir / "e0_final_report.json"
    write_jsonl(trace_path, records)
    failed = [record["case_id"] for record in records if not record["passed"]]
    max_logits_kl = max(
        record["first_step_logits_kl_baseline_to_memory"] for record in records
    )
    active_logits_effect_detected = max_logits_kl > args.min_active_logits_kl
    mechanism_passed = not failed and active_logits_effect_detected
    report = {
        "schema_version": "side-kv-mechanism-audit-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if mechanism_passed else "failed",
        "formal_e0_mechanism_passed": mechanism_passed,
        "case_count": len(records),
        "passed_case_count": len(records) - len(failed),
        "failed_case_ids": failed,
        "mean_memory_attention_mass": sum(
            record["memory_attention"]["memory_attention_mass"] for record in records
        )
        / len(records),
        "mean_first_step_logits_kl": sum(
            record["first_step_logits_kl_baseline_to_memory"] for record in records
        )
        / len(records),
        "max_first_step_logits_kl": max_logits_kl,
        "min_active_logits_kl": args.min_active_logits_kl,
        "active_logits_effect_detected": active_logits_effect_detected,
        "max_canonical_rope_score_relative_error": max(
            record["memory_attention"]["canonical_rope_score_relative_error"]
            for record in records
        ),
        "canonical_rope_relative_tolerance": args.canonical_rope_rtol,
        "inputs": {
            "cases_path": str(args.cases.resolve()),
            "cases_sha256": file_sha256(args.cases),
            "side_kv_manifest_path": str(args.side_kv_manifest.resolve()),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "model": args.model,
            "model_revision": resolved_model_revision,
            "tokenizer_revision": resolved_tokenizer_revision,
            "layer": args.layer,
            "dtype": args.dtype,
        },
        "trace": {"path": trace_path.name, "sha256": file_sha256(trace_path)},
    }
    write_json(report_path, report)
    if failed:
        raise RuntimeError(f"Side-KV mechanism audit failed for cases: {failed}")
    if not active_logits_effect_detected:
        raise RuntimeError(
            "Side-KV attention was visible but did not change first-step logits "
            f"above the fixed KL floor {args.min_active_logits_kl}"
        )

    final_report = {
        "schema_version": "experience-memory-e0-final-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_e0_passed": True,
        "task_accuracy_used": False,
        "compile_report": {
            "path": str(compile_report_path.resolve()),
            "sha256": file_sha256(compile_report_path),
            "artifact_set_sha256": compile_report.get("artifact_set_sha256"),
            "verified_artifact_sha256": verified_compile_artifacts,
        },
        "mechanism_report": {
            "path": report_path.name,
            "sha256": file_sha256(report_path),
            "trace_sha256": file_sha256(trace_path),
        },
        "requirements": {
            "payload_audit_passed": True,
            "canonical_pre_rope_kv_compiled": True,
            "canonical_rope_shared_phase_identity": True,
            "disabled_path_logit_parity": True,
            "native_cache_prefix_preserved": True,
            "native_cache_length_excludes_memory_slots": True,
            "memory_attention_mass_recorded_and_positive": True,
            "active_memory_changes_logits_above_noise_floor": True,
            "calibration_cases_answer_blind_and_preanswer": True,
        },
    }
    final_report["final_report_sha256"] = canonical_json_sha256(final_report)
    write_json(final_report_path, final_report)
    print(
        f"[side-kv-audit] passed cases={len(records)} final={final_report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
