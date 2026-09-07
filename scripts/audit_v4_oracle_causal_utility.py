#!/usr/bin/env python3
"""Audit V4 target/reference causal utility with oracle bank selection.

Every case is reconstructed from an authenticated source-state gate event.
The baseline, oracle target, and offline-only reference branches replay the
same exact prefix once, clone the same native cache, and greedily continue for
at most 32 tokens under the current nonpersistent V4 memory lifecycle.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_oracle_audit import (
    V4_ORACLE_PROFILE_SCHEMA,
    aggregate_oracle_results,
    build_oracle_plan,
    finalize_result,
    validate_oracle_plan,
    validate_result_against_plan,
)
from memgen.experience.v4_source_state import load_source_state_cache
from scripts.build_v4_repair_bank import _validate_split_manifest, load_v4_experiences
from scripts.compile_v4_selector_anchors import _tokenize_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--bank-records", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--token-risk-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument(
        "--attempt-policy", choices=("all", "first"), default="all"
    )
    parser.add_argument(
        "--limit-per-kind",
        type=int,
        default=0,
        help="Smoke cap applied separately to failure and success cases; zero is full.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_revision() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("V4 oracle audit requires a git revision") from exc
    if not value:
        raise RuntimeError("V4 oracle audit resolved an empty git revision")
    return value


def _logical_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return canonical_json_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"created_at", hash_field}
        }
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    temporary.replace(path)
    return count


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


def _score_branch(
    *,
    tokenizer: Any,
    prompt_token_count: int,
    prefix_token_ids: Sequence[int],
    branch: Any,
    verified_success_completion: str,
) -> dict[str, Any]:
    prefix_completion = [int(value) for value in prefix_token_ids[prompt_token_count:]]
    continuation = [int(value) for value in branch.continuation_token_ids]
    full_completion_ids = prefix_completion + continuation
    generated_continuation = tokenizer.decode(
        continuation, skip_special_tokens=True
    ).strip()
    full_completion = tokenizer.decode(
        full_completion_ids, skip_special_tokens=True
    ).strip()
    verifier = diagnose_gsm8k_completion(
        full_completion, verified_success_completion
    )
    branch_payload = branch.to_dict()
    branch_payload.update(
        {
            "generated_continuation": generated_continuation,
            "full_completion": full_completion,
            "full_completion_token_ids": full_completion_ids,
            "full_completion_token_ids_sha256": canonical_json_sha256(
                full_completion_ids
            ),
            "strict_reward": float(verifier["reward"]),
            "task_success": bool(verifier["task_success"]),
            "format_valid": bool(verifier["format_valid"]),
            "diagnostic_answer_correct": verifier["diagnostic_answer_correct"],
            "failure_types": list(verifier["failure_types"]),
        }
    )
    return branch_payload


def _build_profile(
    *, args: argparse.Namespace, cache: Any, plan: Mapping[str, Any]
) -> dict[str, Any]:
    implementation_paths = (
        "data/gsm8k/prompt.py",
        "data/utils/math_utils.py",
        "memgen/experience/v4_oracle_audit.py",
        "memgen/experience/v4_source_state.py",
        "memgen/model/e1_runtime.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_runtime.py",
        "memgen/model/v4_oracle.py",
        "memgen/model/v4_runtime.py",
        "memgen/model/v4_side_kv.py",
        "scripts/audit_v4_oracle_causal_utility.py",
    )
    material = {
        "schema_version": V4_ORACLE_PROFILE_SCHEMA,
        "repository_revision": _git_revision(),
        "implementation_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
        },
        "cache_manifest_logical_sha256": cache.manifest["manifest_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "inputs": {
            "cache_manifest_path": str(args.cache_manifest.resolve()),
            "cache_manifest_sha256": file_sha256(args.cache_manifest),
            "experiences_path": str(args.experiences.resolve()),
            "experiences_sha256": file_sha256(args.experiences),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "bank_records_path": str(args.bank_records.resolve()),
            "bank_records_sha256": file_sha256(args.bank_records),
            "bank_manifest_path": str(args.bank_manifest.resolve()),
            "bank_manifest_sha256": file_sha256(args.bank_manifest),
            "side_kv_manifest_path": str(args.side_kv_manifest.resolve()),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "token_risk_artifact_path": str(args.token_risk_artifact.resolve()),
            "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
        },
        "reasoner": dict(cache.manifest["reasoner"]),
        "configuration": dict(plan["configuration"]),
        "branch_roles": ["baseline", "target", "reference"],
        "reference_role": "offline_directional_control_never_online_loadable",
        "answer_or_reward_access": "post_branch_scoring_only",
        "held_out_generalization_claim": False,
        "audit_interpretation": "optimistic_source_positive_control_and_mechanism_qualification",
        "qualified_for_online_use": False,
    }
    return {
        **material,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_sha256": canonical_json_sha256(material),
    }


def _validate_input_bindings(args: argparse.Namespace, cache: Any) -> None:
    if args.limit_per_kind < 0:
        raise ValueError("V4 oracle limit-per-kind must be non-negative")
    inputs = cache.manifest["provenance"]["inputs"]
    bindings = (
        (args.experiences, "experiences_sha256"),
        (args.split_manifest, "split_manifest_sha256"),
        (args.bank_records, "bank_records_sha256"),
        (args.bank_manifest, "bank_manifest_file_sha256"),
        (args.side_kv_manifest, "side_kv_manifest_file_sha256"),
        (args.token_risk_artifact, "token_risk_artifact_sha256"),
    )
    for path, field in bindings:
        if not path.is_file():
            raise FileNotFoundError(path)
        if inputs.get(field) != file_sha256(path):
            raise ValueError(f"V4 oracle input differs from source-state cache: {field}")


def main() -> None:
    args = parse_args()
    cache = load_source_state_cache(args.cache_manifest, load_tensors=False)
    _validate_input_bindings(args, cache)
    plan = build_oracle_plan(
        cache,
        attempt_policy=args.attempt_policy,
        limit_per_kind=args.limit_per_kind,
    )
    validate_oracle_plan(plan)
    profile = _build_profile(args=args, cache=cache, plan=plan)
    profile_sha256 = str(profile["profile_sha256"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "v4_oracle_profile.json"
    plan_path = output_dir / "v4_oracle_plan.json"
    results_path = output_dir / "v4_oracle_results.jsonl"
    report_path = output_dir / "v4_oracle_report.json"

    if profile_path.is_file():
        existing = json.loads(profile_path.read_text(encoding="utf-8"))
        if existing.get("profile_sha256") != profile_sha256:
            raise ValueError("V4 oracle output directory belongs to another profile")
    else:
        _write_json(profile_path, profile)
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_oracle_plan(existing_plan)
        if existing_plan.get("plan_sha256") != plan.get("plan_sha256"):
            raise ValueError("V4 oracle output directory belongs to another plan")
    else:
        _write_json(plan_path, plan)

    existing = _read_jsonl(results_path) if args.resume else []
    result_by_case: dict[str, dict[str, Any]] = {}
    for row in existing:
        validate_result_against_plan(
            row, plan=plan, profile_sha256=profile_sha256
        )
        case_id = str(row["case_id"])
        if case_id in result_by_case:
            raise ValueError("V4 oracle resume results contain duplicate cases")
        result_by_case[case_id] = row
    expected_case_ids = [str(case["case_id"]) for case in plan["cases"]]
    if set(result_by_case) - set(expected_case_ids):
        raise ValueError("V4 oracle resume results contain cases outside the plan")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.side_kv import SideKVAttentionController
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v4_oracle import (
        V4OfflineSideKVRoleBankLoader,
        V4OracleExactPrefixRuntime,
    )
    from memgen.model.v4_side_kv import (
        V4_MEMORY_SCORE_BIAS,
        V4_MEMORY_SCORE_NORMALIZATION,
    )

    split_manifest = _validate_split_manifest(
        args.split_manifest, dataset_revision=args.dataset_revision
    )
    experiences = load_v4_experiences(
        args.experiences, split_manifest=split_manifest
    )
    experience_by_id = {str(item["experience_id"]): item for item in experiences}
    role_loader = V4OfflineSideKVRoleBankLoader(
        manifest_path=args.side_kv_manifest
    )
    if (
        role_loader.manifest["manifest_sha256"]
        != plan["provenance"]["side_kv_manifest_logical_sha256"]
    ):
        raise ValueError("V4 oracle side-KV logical binding differs from plan")
    reasoner = role_loader.manifest["reasoner"]
    risk_artifact = torch.load(
        args.token_risk_artifact, map_location="cpu", weights_only=False
    )
    gate = EntropyHysteresisGate.from_token_artifact(risk_artifact)
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field) != reasoner.get(field):
            raise ValueError("V4 oracle gate and side-KV reasoner provenance differ")
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
        raise ValueError("V4 oracle model/tokenizer revision drifted")
    controller = SideKVAttentionController(
        model=model,
        layer_number=24,
        audit_canonical_rope=True,
        memory_score_normalization=V4_MEMORY_SCORE_NORMALIZATION,
        memory_score_bias=V4_MEMORY_SCORE_BIAS,
    )
    runtime = V4OracleExactPrefixRuntime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        gate=gate,
        controller=controller,
        maximum_continuation_tokens=32,
    )
    try:
        for position, case in enumerate(plan["cases"], start=1):
            case_id = str(case["case_id"])
            if case_id in result_by_case:
                print(
                    f"[v4-oracle] reuse {position}/{len(plan['cases'])} {case_id}",
                    flush=True,
                )
                continue
            experience = experience_by_id.get(str(case["experience_id"]))
            if experience is None or experience.get("sample_id") != case["sample_id"]:
                raise ValueError("V4 oracle lost a construction experience")
            prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(
                tokenizer, str(experience["context"])
            )
            completion = (
                str(experience["reference_trajectory"])
                if case["case_kind"] == "failure_oracle"
                else str(experience["trajectory"])
            )
            trajectory = _tokenize_trajectory(tokenizer, prompt_ids, completion)
            token_position = int(case["token_position"])
            prefix_ids = tuple(trajectory.ids[: token_position + 1])
            if (
                len(prefix_ids) != int(case["prefix_token_count"])
                or canonical_json_sha256(list(prefix_ids))
                != case["prefix_token_ids_sha256"]
            ):
                raise ValueError("V4 oracle reconstructed prefix differs from cache")
            bank_id = str(case["bank_id"])
            model_dtype = next(model.parameters()).dtype
            target = role_loader.get_target(
                bank_id, device=args.device, dtype=model_dtype
            )
            reference = role_loader.get_reference_offline(
                bank_id, device=args.device, dtype=model_dtype
            )
            started = time.perf_counter()
            branch_results, parity = runtime.run_three_branches(
                prefix_token_ids=prefix_ids,
                prompt_token_count=len(prompt_ids),
                target=target,
                reference=reference,
            )
            runtime_seconds = time.perf_counter() - started
            branches = {
                role: _score_branch(
                    tokenizer=tokenizer,
                    prompt_token_count=len(prompt_ids),
                    prefix_token_ids=prefix_ids,
                    branch=result,
                    verified_success_completion=str(experience["trajectory"]),
                )
                for role, result in branch_results.items()
            }
            baseline_reward = float(branches["baseline"]["strict_reward"])
            target_reward = float(branches["target"]["strict_reward"])
            reference_reward = float(branches["reference"]["strict_reward"])
            row = finalize_result(
                {
                    "profile_sha256": profile_sha256,
                    "plan_sha256": plan["plan_sha256"],
                    "case_id": case_id,
                    "case_sha256": case["case_sha256"],
                    "case_kind": case["case_kind"],
                    "source_event_id": case["source_event_id"],
                    "experience_id": case["experience_id"],
                    "sample_id": case["sample_id"],
                    "independent_sample_id": case["independent_sample_id"],
                    "bank_id": bank_id,
                    "is_medoid": case["is_medoid"],
                    "curation_tier": case["curation_tier"],
                    "gate_attempt_number": case["gate_attempt_number"],
                    "trajectory_side": case["trajectory_side"],
                    "prompt_token_count": len(prompt_ids),
                    "prefix_token_count": len(prefix_ids),
                    "prefix_token_ids_sha256": case["prefix_token_ids_sha256"],
                    "prefix_cache_parity": parity,
                    "branches": branches,
                    "contrasts": {
                        "baseline_wrong_to_target_correct": (
                            baseline_reward == 0.0 and target_reward == 1.0
                        ),
                        "baseline_wrong_to_reference_correct": (
                            baseline_reward == 0.0 and reference_reward == 1.0
                        ),
                        "target_minus_baseline_reward": target_reward - baseline_reward,
                        "reference_minus_baseline_reward": reference_reward - baseline_reward,
                        "target_minus_reference_reward": target_reward - reference_reward,
                        "target_harmed_baseline": baseline_reward == 1.0 and target_reward == 0.0,
                        "target_better_than_reference": target_reward > reference_reward,
                    },
                    "reference_online_injectable": False,
                    "runtime_seconds_three_branches": runtime_seconds,
                }
            )
            validate_result_against_plan(
                row, plan=plan, profile_sha256=profile_sha256
            )
            result_by_case[case_id] = row
            ordered_rows = [
                result_by_case[value]
                for value in expected_case_ids
                if value in result_by_case
            ]
            _write_jsonl(results_path, ordered_rows)
            progress = aggregate_oracle_results(
                plan=plan, rows=ordered_rows, profile_sha256=profile_sha256
            )
            _write_json(report_path, progress)
            print(
                f"[v4-oracle] {position}/{len(plan['cases'])} {case_id} "
                f"baseline={baseline_reward:.0f} target={target_reward:.0f} "
                f"reference={reference_reward:.0f}",
                flush=True,
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        controller.close()

    rows = [result_by_case[case_id] for case_id in expected_case_ids]
    report = aggregate_oracle_results(
        plan=plan, rows=rows, profile_sha256=profile_sha256
    )
    if not report["complete"]:
        raise RuntimeError("V4 oracle audit did not complete its case plan")
    report["artifacts"] = {
        "profile": {
            "path": profile_path.name,
            "sha256": file_sha256(profile_path),
            "logical_sha256": profile_sha256,
        },
        "plan": {
            "path": plan_path.name,
            "sha256": file_sha256(plan_path),
            "logical_sha256": plan["plan_sha256"],
        },
        "results": {
            "path": results_path.name,
            "sha256": file_sha256(results_path),
            "row_count": len(rows),
            "record_order_sha256": canonical_json_sha256(expected_case_ids),
        },
        "online_selector_tensor": None,
        "online_selector_manifest": None,
    }
    report["report_sha256"] = _logical_hash(report, "report_sha256")
    _write_json(report_path, report)
    print(
        f"[v4-oracle] complete cases={len(rows)} "
        f"unreachable={plan['gate_unreachable_failure_count']} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-oracle] error: {exc}", file=sys.stderr)
        raise
