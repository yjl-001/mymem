#!/usr/bin/env python3
"""Evaluate the authenticated MemGen V4 runtime against a cache-greedy baseline."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.e1 import E1EvaluationScope
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    text_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--selector-anchor-manifest", type=Path, required=True)
    parser.add_argument("--token-risk-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--logical-split",
        choices=("calibration-val", "dev-test"),
        default="dev-test",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_hashed_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    logical = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    }
    if value.get("manifest_sha256") != canonical_json_sha256(logical):
        raise ValueError(f"Manifest hash mismatch: {path}")
    return value


def _processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def _score_completion(
    *, tokenizer: Any, token_ids: Sequence[int], ground_truth: str
) -> dict[str, Any]:
    ids = tuple(int(value) for value in token_ids)
    completion = tokenizer.decode(list(ids), skip_special_tokens=True).strip()
    verifier = diagnose_gsm8k_completion(completion, ground_truth)
    return {
        "completion": completion,
        "completion_token_ids": list(ids),
        "completion_token_ids_sha256": canonical_json_sha256(list(ids)),
        "generation_length": len(ids),
        "final_reward": float(verifier["reward"]),
        "format_valid": bool(verifier["format_valid"]),
        "verifier": verifier,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _repository_state() -> dict[str, Any]:
    paths = (
        "memgen/model/e1_runtime.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_runtime.py",
        "memgen/model/v4_runtime.py",
        "memgen/model/v4_selector.py",
        "memgen/model/v4_side_kv.py",
        "memgen/model/v4_online.py",
        "scripts/evaluate_v4_experience_memory.py",
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("V4 evaluation requires a git revision") from exc
    if not revision:
        raise RuntimeError("V4 evaluation resolved an empty git revision")
    return {
        "git_revision": revision,
        "implementation_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative) for relative in paths
        },
    }


def _direct_memory_spans(
    attention_traces: Sequence[Any], *, episode_start_indices: Sequence[int]
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    starts = {int(value) for value in episode_start_indices}
    for item in attention_traces:
        index = int(item.generated_input_index)
        memory_id = str(item.trace.memory_id)
        if (
            not spans
            or index in starts
            or spans[-1]["memory_id"] != memory_id
            or index != spans[-1]["end_generated_input_index"] + 1
        ):
            spans.append(
                {
                    "memory_id": memory_id,
                    "start_generated_input_index": index,
                    "end_generated_input_index": index,
                    "attention_step_count": 1,
                }
            )
        else:
            spans[-1]["end_generated_input_index"] = index
            spans[-1]["attention_step_count"] += 1
    return spans


def _v4_diagnostics(result: Any, *, selector_bank_ids: Sequence[str]) -> dict[str, Any]:
    payload = result.to_dict()
    summary = payload["summary"]
    attempts = tuple(result.selector_attempts)
    selected = [
        attempt for attempt in attempts if attempt.decision.selected_bank_id is not None
    ]
    spans = _direct_memory_spans(
        result.attention_traces,
        episode_start_indices=[item.generated_input_index for item in selected],
    )
    diagnostics = {
        "selector_attempt_count": len(attempts),
        "selection_count": len(selected),
        "abstain_count": len(attempts) - len(selected),
        "gate_observation_count": len(result.gate_traces),
        "joint_trigger_qualified_count": sum(
            trace.joint_trigger_qualified for trace in result.gate_traces
        ),
        "memory_attention_step_count": len(result.attention_traces),
        "memory_activation_spans": spans,
        "maximum_direct_memory_span": max(
            (span["attention_step_count"] for span in spans), default=0
        ),
        "attempt_budget_respected": len(attempts) <= 3,
        "direct_memory_window_respected": all(
            span["attention_step_count"] <= 32 for span in spans
        ),
        "selected_bank_in_selector_namespace": all(
            attempt.decision.selected_bank_id in selector_bank_ids
            for attempt in selected
        ),
        "reference_never_online": all(
            "::reference" not in trace.trace.memory_id
            for trace in result.attention_traces
        ),
        "native_cache_excludes_memory_slots": all(
            trace.trace.native_key_length == trace.processed_prefix_token_count
            for trace in result.attention_traces
        ),
        "memory_attention_mass_finite_and_positive": all(
            math.isfinite(float(trace.trace.memory_attention_mass))
            and float(trace.trace.memory_attention_mass) > 0.0
            for trace in result.attention_traces
        ),
        "selected_attempt_has_activation_counterfactual": all(
            attempt.activation_first_step_logits_kl is not None
            and math.isfinite(float(attempt.activation_first_step_logits_kl))
            and attempt.activation_baseline_first_token_id is not None
            and attempt.activation_target_first_token_id is not None
            for attempt in selected
        ),
        "lifecycle_attempt_count_matches": (
            int(result.lifecycle_summary["attempt_count"]) == len(attempts)
        ),
        "answer_marker_seen": result.answer_marker_seen,
        "final_state": result.final_state,
        "runtime_summary": summary,
    }
    diagnostics["passed"] = all(
        diagnostics[field]
        for field in (
            "attempt_budget_respected",
            "direct_memory_window_respected",
            "selected_bank_in_selector_namespace",
            "reference_never_online",
            "native_cache_excludes_memory_slots",
            "memory_attention_mass_finite_and_positive",
            "selected_attempt_has_activation_counterfactual",
            "lifecycle_attempt_count_matches",
        )
    )
    return diagnostics


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty V4 result set")
    baseline_rewards = [float(row["baseline"]["final_reward"]) for row in records]
    v4_rewards = [float(row["v4"]["final_reward"]) for row in records]
    failures = [index for index, reward in enumerate(baseline_rewards) if reward == 0.0]
    successes = [index for index, reward in enumerate(baseline_rewards) if reward == 1.0]
    selected = [
        int(row["diagnostics"]["selection_count"]) > 0 for row in records
    ]
    return {
        "sample_count": len(records),
        "baseline_accuracy": sum(baseline_rewards) / len(records),
        "v4_accuracy": sum(v4_rewards) / len(records),
        "absolute_uplift": (sum(v4_rewards) - sum(baseline_rewards)) / len(records),
        "baseline_failure_count": len(failures),
        "baseline_success_count": len(successes),
        "recovered_failure_count": sum(v4_rewards[index] == 1.0 for index in failures),
        "harmed_success_count": sum(v4_rewards[index] == 0.0 for index in successes),
        "question_with_selection_count": sum(selected),
        "question_with_selection_rate": sum(selected) / len(records),
        "total_selector_attempt_count": sum(
            int(row["diagnostics"]["selector_attempt_count"]) for row in records
        ),
        "total_abstain_count": sum(
            int(row["diagnostics"]["abstain_count"]) for row in records
        ),
        "total_memory_attention_step_count": sum(
            int(row["diagnostics"]["memory_attention_step_count"])
            for row in records
        ),
        "runtime_authentication_passed": all(
            row["diagnostics"]["passed"] for row in records
        ),
    }


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit < 0 or args.max_new_tokens < 0:
        raise ValueError("V4 offset, limit, and token budget must be non-negative")
    for path in (
        args.split_manifest,
        args.side_kv_manifest,
        args.selector_anchor_manifest,
        args.token_risk_artifact,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime
    from memgen.model.side_kv import SideKVAttentionController
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v4_online import OnlineExperienceMemorySystemV4
    from memgen.model.v4_selector import V4SelectorAnchorBankLoader
    from memgen.model.v4_side_kv import (
        V4_MEMORY_SCORE_BIAS,
        V4_MEMORY_SCORE_NORMALIZATION,
        V4SideKVBankLoader,
    )

    split_manifest = _load_hashed_manifest(args.split_manifest)
    if split_manifest.get("overlap_check", {}).get("passed") is not True:
        raise ValueError("V4 split manifest did not pass overlap checking")
    scope = E1EvaluationScope.from_logical_split(args.logical_split)
    selected_samples = [
        item
        for item in split_manifest["samples"]
        if item.get("logical_split") == scope.logical_split
    ][args.offset :]
    if args.limit:
        selected_samples = selected_samples[: args.limit]
    if not selected_samples:
        raise ValueError("V4 evaluation selected an empty logical split")

    side_loader = V4SideKVBankLoader(manifest_path=args.side_kv_manifest)
    reasoner = side_loader.manifest["reasoner"]
    anchor_loader = V4SelectorAnchorBankLoader(
        manifest_path=args.selector_anchor_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    selector_bank_ids = tuple(bank.bank_id for bank in anchor_loader.banks)
    if not set(selector_bank_ids).issubset(set(side_loader.bank_ids)):
        raise ValueError("V4 selector bank namespace is not a side-KV subset")
    anchor_inputs = anchor_loader.manifest["provenance"]["inputs"]
    if (
        anchor_inputs.get("side_kv_manifest_sha256")
        != file_sha256(args.side_kv_manifest)
        or anchor_inputs.get("split_manifest_sha256")
        != file_sha256(args.split_manifest)
        or anchor_inputs.get("token_risk_artifact_sha256")
        != file_sha256(args.token_risk_artifact)
    ):
        raise ValueError("V4 runtime artifacts were not compiled from these inputs")

    risk_artifact = torch.load(
        args.token_risk_artifact, map_location="cpu", weights_only=False
    )
    gate = EntropyHysteresisGate.from_token_artifact(risk_artifact)
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field) != reasoner.get(field):
            raise ValueError("V4 runtime risk artifact reasoner differs")

    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    if (
        str(getattr(model.config, "_commit_hash", None) or reasoner["model_revision"])
        != reasoner["model_revision"]
        or str(
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            or reasoner["tokenizer_revision"]
        )
        != reasoner["tokenizer_revision"]
    ):
        raise ValueError("V4 runtime model/tokenizer revision drifted")

    max_new_tokens = args.max_new_tokens or GSM8K_PROMPT_CONTRACT.max_new_tokens
    baseline_runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=max_new_tokens,
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=24,
        audit_canonical_rope=True,
        memory_score_normalization=V4_MEMORY_SCORE_NORMALIZATION,
        memory_score_bias=V4_MEMORY_SCORE_BIAS,
    )
    v4_runtime = OnlineExperienceMemorySystemV4(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=max_new_tokens,
        gate=gate,
        selector=anchor_loader.selector,
        loader=side_loader,
        controller=controller,
    )

    dataset_revision = str(split_manifest["dataset"]["revision"])
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=scope.dataset_split,
        revision=dataset_revision,
    )
    run_profile = {
        "schema_version": "memgen-v4-evaluation-profile-v1",
        "logical_split": scope.logical_split,
        "dataset_split": scope.dataset_split,
        "offset": args.offset,
        "limit": args.limit,
        "max_new_tokens": max_new_tokens,
        "dtype": args.dtype,
        "attention_implementation": "sdpa",
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
        "selector_anchor_manifest_sha256": file_sha256(
            args.selector_anchor_manifest
        ),
        "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
        "reasoner": dict(reasoner),
        "selector_config": anchor_loader.config.to_dict(),
        "repository": _repository_state(),
    }
    profile_sha256 = canonical_json_sha256(run_profile)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "v4_results.jsonl"
    existing = _read_jsonl(results_path) if args.resume else []
    existing_by_id: dict[str, dict[str, Any]] = {}
    selected_manifest_by_id = {
        str(item["sample_id"]): item for item in selected_samples
    }
    for row in existing:
        logical = {key: value for key, value in row.items() if key != "record_sha256"}
        if (
            row.get("record_sha256") != canonical_json_sha256(logical)
            or row.get("evaluation_profile_sha256") != profile_sha256
        ):
            raise ValueError("V4 resume record failed authentication")
        sample_id = str(row["sample_id"])
        if sample_id in existing_by_id:
            raise ValueError("V4 resume results contain duplicate sample IDs")
        sample_manifest = selected_manifest_by_id.get(sample_id)
        if (
            row.get("schema_version") != "memgen-v4-gsm8k-result-v1"
            or sample_manifest is None
            or row.get("logical_split") != scope.logical_split
            or row.get("dataset_split") != scope.dataset_split
            or row.get("source_index") != sample_manifest.get("source_index")
            or row.get("question_sha256")
            != sample_manifest.get("question_sha256")
            or row.get("answer_sha256") != sample_manifest.get("answer_sha256")
            or row.get("diagnostics", {}).get("passed") is not True
        ):
            raise ValueError("V4 resume record identity or runtime audit drifted")
        existing_by_id[sample_id] = row
    selected_ids = [str(item["sample_id"]) for item in selected_samples]
    if set(existing_by_id) - set(selected_ids):
        raise ValueError("V4 resume results contain samples outside the selected scope")

    records = [existing_by_id[sample_id] for sample_id in selected_ids if sample_id in existing_by_id]
    try:
        for position, sample_manifest in enumerate(selected_samples, start=1):
            sample_id = str(sample_manifest["sample_id"])
            if sample_id in existing_by_id:
                print(f"[v4-eval] reuse {position}/{len(selected_samples)} {sample_id}", flush=True)
                continue
            source = dataset[int(sample_manifest["source_index"])]
            question = str(source["question"]).strip()
            answer = str(source["answer"]).strip()
            if (
                text_sha256(question) != sample_manifest["question_sha256"]
                or text_sha256(answer) != sample_manifest["answer_sha256"]
            ):
                raise ValueError(f"V4 dataset row hash mismatch: {sample_id}")
            prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, question)
            baseline_started = time.perf_counter()
            baseline_ids = baseline_runtime.generate_cache_greedy(prompt_ids)
            baseline_seconds = time.perf_counter() - baseline_started
            v4_started = time.perf_counter()
            v4_result = v4_runtime.generate(prompt_token_ids=prompt_ids)
            v4_seconds = time.perf_counter() - v4_started
            ground_truth = _processed_solution(answer)
            record: dict[str, Any] = {
                "schema_version": "memgen-v4-gsm8k-result-v1",
                "evaluation_profile_sha256": profile_sha256,
                "sample_id": sample_id,
                "logical_split": scope.logical_split,
                "dataset_split": scope.dataset_split,
                "source_index": int(sample_manifest["source_index"]),
                "question_sha256": sample_manifest["question_sha256"],
                "answer_sha256": sample_manifest["answer_sha256"],
                "prompt_token_count": len(prompt_ids),
                "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
                "baseline": _score_completion(
                    tokenizer=tokenizer,
                    token_ids=baseline_ids,
                    ground_truth=ground_truth,
                ),
                "v4": _score_completion(
                    tokenizer=tokenizer,
                    token_ids=v4_result.completion_token_ids,
                    ground_truth=ground_truth,
                ),
                "v4_runtime": v4_result.to_dict(),
                "diagnostics": _v4_diagnostics(
                    v4_result, selector_bank_ids=selector_bank_ids
                ),
                "runtime_seconds": {
                    "baseline": baseline_seconds,
                    "v4": v4_seconds,
                },
            }
            record["record_sha256"] = canonical_json_sha256(record)
            existing_by_id[sample_id] = record
            records = [existing_by_id[item] for item in selected_ids if item in existing_by_id]
            _write_jsonl(results_path, records)
            print(
                f"[v4-eval] {position}/{len(selected_samples)} {sample_id} "
                f"base={record['baseline']['final_reward']:.0f} "
                f"v4={record['v4']['final_reward']:.0f} "
                f"attempts={record['diagnostics']['selector_attempt_count']} "
                f"selected={record['diagnostics']['selection_count']}",
                flush=True,
            )
    finally:
        controller.close()

    if len(records) != len(selected_samples):
        raise RuntimeError("V4 evaluation did not complete the selected sample set")
    aggregate = _aggregate(records)
    if not aggregate["runtime_authentication_passed"]:
        raise RuntimeError("V4 runtime diagnostics failed")
    selected_counter = Counter(
        attempt["decision"]["selected_bank_id"]
        for row in records
        for attempt in row["v4_runtime"]["selector_attempts"]
        if attempt["decision"]["selected_bank_id"] is not None
    )
    report = {
        "schema_version": "memgen-v4-gsm8k-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_final_test_claim": False,
        "evaluation_profile": run_profile,
        "evaluation_profile_sha256": profile_sha256,
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "aggregate": aggregate,
        "selection_counts_by_bank": dict(sorted(selected_counter.items())),
        "selector_bank_ids": list(selector_bank_ids),
        "inputs": {
            "split_manifest": {
                "path": str(args.split_manifest.resolve()),
                "sha256": file_sha256(args.split_manifest),
                "logical_sha256": split_manifest["manifest_sha256"],
            },
            "side_kv_manifest": {
                "path": str(args.side_kv_manifest.resolve()),
                "sha256": file_sha256(args.side_kv_manifest),
                "logical_sha256": side_loader.manifest["manifest_sha256"],
            },
            "selector_anchor_manifest": {
                "path": str(args.selector_anchor_manifest.resolve()),
                "sha256": file_sha256(args.selector_anchor_manifest),
                "logical_sha256": anchor_loader.manifest["manifest_sha256"],
            },
            "token_risk_artifact": {
                "path": str(args.token_risk_artifact.resolve()),
                "sha256": file_sha256(args.token_risk_artifact),
            },
        },
        "results": {
            "path": results_path.name,
            "sha256": file_sha256(results_path),
            "record_count": len(records),
            "record_order_sha256": canonical_json_sha256(selected_ids),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = output_dir / "v4_report.json"
    _write_json(report_path, report)
    print(
        "[v4-eval] complete "
        f"base={aggregate['baseline_accuracy']:.6f} "
        f"v4={aggregate['v4_accuracy']:.6f} "
        f"uplift={aggregate['absolute_uplift']:+.6f} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-eval] error: {exc}", file=sys.stderr)
        raise
