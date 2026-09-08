#!/usr/bin/env python3
"""Four-layer unified-memory audit; immutable source binding and resumable cases."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.v4_3_artifacts import atomic_json, implementation_hashes, load_construction, read_json, read_jsonl
from memgen.experience.v4_3_bank import authenticate, authenticate_packets, canonical_hash, file_hash, seal
from memgen.experience.v4_3_audit import (
    CONFIGURATION, LAYERS, aggregate, bind_source_state, build_plan, build_wrong_controls, validate_result,
)
from memgen.experience.v4_source_state import load_source_state_cache

IMPLEMENTATION_PATHS = (
    "memgen/model/__init__.py",
    "data/gsm8k/prompt.py", "data/utils/math_utils.py", "memgen/chat_templates.py",
    "memgen/experience/v4_source_state.py", "memgen/experience/v4_3_artifacts.py",
    "memgen/experience/v4_3_bank.py", "memgen/experience/v4_3_audit.py",
    "memgen/model/e1_runtime.py", "memgen/model/side_kv.py", "memgen/model/v3_runtime.py",
    "memgen/model/v4_oracle.py", "memgen/model/v4_runtime.py", "memgen/model/v4_side_kv.py",
    "memgen/model/v4_3_runtime.py", "memgen/model/v4_3_side_kv.py",
    "scripts/audit_v4_3_unified_memory.py",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-dir", "side-kv-dir", "semantic-packets", "cache-manifest", "token-risk-artifact", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--smoke-report", type=Path, help="Required authenticated smoke report before full generation")
    p.add_argument("--all-bank-sweep", action="store_true", help="Append exhaustive primary-Bank failure-prefix ceiling audit")
    return p.parse_args()


def prepare(args):
    from memgen.model.v4_3_side_kv import V43SideKVBankLoader
    bank = load_construction(args.bank_dir)
    packets = read_jsonl(args.semantic_packets)
    authenticate_packets(packets)
    evidence = {e["evidence_id"]: e for packet in packets for e in packet["evidence"]}
    cache = load_source_state_cache(args.cache_manifest, load_tensors=False)
    binding = bind_source_state(cache=cache, bank=bank, evidence=evidence,
                                semantic_packets_path=args.semantic_packets, risk_path=args.token_risk_artifact)
    reasoner = {k: cache.manifest["reasoner"][k] for k in ("model_name", "model_revision", "tokenizer_revision")}
    records, loaders, manifests, slots = [], {}, {}, {}
    for tier in ("primary", "conditional"):
        source = bank[f"{tier}_bank_records.jsonl"]
        if not source:
            continue
        path = args.side_kv_dir / f"v4_3_{tier}_side_kv_manifest.json"
        loader = V43SideKVBankLoader(path, source_manifest=bank[f"{tier}_bank_manifest.json"],
                                    source_records=source, expected_reasoner=reasoner)
        if loader.manifest["dtype"] != "torch.bfloat16":
            raise ValueError("Production audit requires bfloat16 compiled tensors")
        manifests[tier] = loader.manifest["manifest_sha256"]
        records.extend(source)
        for bid in loader.bank_ids:
            loaders[bid] = loader
            slots[bid] = loader.entries[bid]["kv_valid_slot_count"]
    controls = build_wrong_controls(records, slots)
    plan = build_plan(candidates=bank["candidate_bank_records.jsonl"], events=cache.events,
                      controls=controls, binding=binding, mode=args.mode, all_bank_sweep=args.all_bank_sweep)
    experiment = {"configuration": CONFIGURATION, "binding_sha256": binding["binding_sha256"],
        "controls_sha256": controls["controls_sha256"], "compiled_manifest_sha256": manifests,
        "implementation_sha256": implementation_hashes(IMPLEMENTATION_PATHS),
        "reasoner": reasoner, "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(chat_template=CONVERSATION_TEMPLATE)}
    profile = seal({"schema_version": "memgen-v4.3-audit-profile-v1", "experiment": experiment,
        "experiment_identity_sha256": canonical_hash(experiment), "plan_sha256": plan["plan_sha256"],
        "answer_access": "post_branch_scoring_only", "offline_only": True, "qualified_for_online_use": False}, "profile_sha256")
    return bank, evidence, cache, records, loaders, binding, controls, plan, profile


def checked_prefix(case: Mapping[str, Any], evidence: Mapping[str, Any], tokenizer: Any) -> list[int]:
    prompt = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, evidence["question"])
    if len(prompt) != case["prompt_token_count"]:
        raise ValueError("Reconstructed prompt length differs from source cache")
    if case["audit_layer"] in {"visible_content", "prompt_end_latent"}:
        prefix = prompt
    else:
        completion = evidence["verified_success_trajectory"] if case["audit_layer"] == "success_safety" else evidence["verified_failure_trajectory"]
        tokens = prompt + list(tokenizer.encode(completion.strip(), add_special_tokens=False))
        prefix = tokens[:case["prefix_token_count"]]
    if len(prefix) != case["prefix_token_count"] or canonical_hash(prefix) != case["prefix_token_ids_sha256"]:
        raise ValueError("Exact prefix identity differs from authenticated source state")
    return prefix


def score_branch(tokenizer, prefix, prompt_count, branch, official_solution):
    result = dict(branch)
    prior = prefix[prompt_count:]
    full_ids = list(prior) + branch["continuation_token_ids"]
    local_ids = list(prior) + branch["local_continuation_token_ids"]
    completion = tokenizer.decode(full_ids, skip_special_tokens=True).strip()
    local_completion = tokenizer.decode(local_ids, skip_special_tokens=True).strip()
    # Packets preserve raw GSM8K solutions ending in '\n#### answer'; the
    # repository strict scorer requires boxed gold, just as source recovery did.
    parts = official_solution.strip().split("\n####")
    gold = parts[0] + "\\boxed{" + parts[-1].strip() + "}" if len(parts) > 1 else official_solution
    final = diagnose_gsm8k_completion(completion, gold)
    local = diagnose_gsm8k_completion(local_completion, gold)
    result.update(full_completion=completion, full_completion_token_ids_sha256=canonical_hash(full_ids),
                  strict_reward=float(final["reward"]), task_success=bool(final["task_success"]),
                  format_valid=bool(final["format_valid"]), failure_types=list(final["failure_types"]),
                  local_intervention_evaluation={"strict_reward": float(local["reward"]),
                      "format_valid": bool(local["format_valid"]), "full_completion": local_completion,
                      "observation_token_limit": 32})
    return result


def smoke_integrity(plan, rows):
    if plan["mode"] != "smoke" or len(rows) != len(plan["cases"]):
        return False
    for tier in ("primary", "conditional"):
        subset = [r for r in rows if r["quality_tier"] == tier]
        if not set(LAYERS) <= {r["audit_layer"] for r in subset}:
            return False
        latent = [r for r in subset if r["audit_layer"] != "visible_content"]
        if not any(b["first_step_logits_kl"] > 1e-12 or b["continuation_token_ids"] != r["branches"]["baseline"]["continuation_token_ids"]
                   for r in latent for name, b in r["branches"].items() if name != "baseline"):
            return False
    return True


def make_report(plan, rows, profile):
    report = aggregate(plan, rows, profile["profile_sha256"])
    report.pop("report_sha256")
    report.update(experiment_identity_sha256=profile["experiment_identity_sha256"],
                  smoke_integrity_passed=smoke_integrity(plan, rows),
                  external_api_calls_made=0, api_key_read=False,
                  results={r["case_id"]: r["record_sha256"] for r in rows})
    return seal(report, "report_sha256")


def write_report(args, plan, rows, profile):
    report = make_report(plan, rows, profile)
    atomic_json(args.output_dir / "v4_3_audit_report.json", report)
    core = {k: report[k] for k in ("status", "mode", "complete", "expected_case_count", "completed_case_count",
            "experiment_identity_sha256", "smoke_integrity_passed", "configuration", "source_sample_count",
            "excluded_samples", "gate_unreachable_failure_count", "gate_unreachable_counted_as_memory_ineffective",
            "offline_only", "qualified_for_online_use", "held_out_generalization_claim", "external_api_calls_made")}
    core["by_audit_layer"] = {layer: {k: values[k] for k in ("expected_case_count", "completed_case_count", "overall", "by_quality_tier")}
                              for layer, values in report["by_audit_layer"].items()}
    if "all_bank_sweep" in report["by_audit_layer"]:
        core["all_bank_ceiling"] = report["by_audit_layer"]["all_bank_sweep"].get("ceiling")
    atomic_json(args.output_dir / "v4_3_core_summary.json", seal(core, "summary_sha256"))
    return report


def authenticate_smoke(path: Path | None, experiment_id: str):
    if path is None or not path.is_file():
        raise ValueError("Full requires a completed smoke report from this exact experiment")
    report = read_json(path)
    authenticate(report, "report_sha256", "smoke report")
    if (report.get("mode") != "smoke" or report.get("complete") is not True
            or report.get("smoke_integrity_passed") is not True
            or report.get("experiment_identity_sha256") != experiment_id):
        raise ValueError("Smoke has not passed for these exact artifacts and implementation")
    profile = read_json(path.parent / "v4_3_audit_profile.json")
    authenticate(profile, "profile_sha256", "smoke profile")
    plan = read_json(path.parent / "v4_3_audit_plan.json")
    authenticate(plan, "plan_sha256", "smoke plan")
    if (profile["experiment_identity_sha256"] != canonical_hash(profile["experiment"])
            or report["profile_sha256"] != profile["profile_sha256"]
            or profile["plan_sha256"] != plan["plan_sha256"]
            or report["plan_sha256"] != plan["plan_sha256"]):
        raise ValueError("Smoke report/profile/plan identity mismatch")
    rows = [read_json(path.parent / "cases" / (c["case_id"] + ".json")) for c in plan["cases"]]
    reconstructed = make_report(plan, rows, profile)
    if (reconstructed != report or not smoke_integrity(plan, rows)
            or report["results"] != {r["case_id"]: r["record_sha256"] for r in rows}
            or profile["experiment_identity_sha256"] != experiment_id):
        raise ValueError("Smoke report does not agree with authenticated per-case results")


def main():
    args = parse_args()
    bank, evidence, cache, records, loaders, binding, controls, plan, profile = prepare(args)
    if args.mode == "full" and not args.plan_only:
        authenticate_smoke(args.smoke_report, profile["experiment_identity_sha256"])
    paths = {"v4_3_source_state_binding.json": binding, "v4_3_wrong_bank_controls.json": controls,
             "v4_3_audit_plan.json": plan, "v4_3_audit_profile.json": profile}
    if args.output_dir.is_symlink() or args.output_dir.resolve() in {args.bank_dir.resolve(), args.side_kv_dir.resolve(), args.cache_manifest.parent.resolve()}:
        raise ValueError("Audit output must be separate from immutable source directories")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.resume or args.validate_only):
        raise ValueError("Audit output exists; pass --resume")
    # Validate every existing immutable plan/profile before writing any new one.
    for name, value in paths.items():
        path = args.output_dir / name
        if path.exists() and read_json(path) != value:
            raise ValueError(f"Audit output identity drift: {path}")
        if args.validate_only and not path.exists():
            raise ValueError(f"Missing audit artifact: {path}")
    if not args.validate_only:
        for name, value in paths.items():
            atomic_json(args.output_dir / name, value, immutable=True)
    result_by_id = {}
    case_by_id = {c["case_id"]: c for c in plan["cases"]}
    directory = args.output_dir / "cases"
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.suffix != ".json" or path.stem not in case_by_id:
                raise ValueError(f"Unexpected case artifact: {path}")
            row = read_json(path)
            validate_result(row, case_by_id[path.stem], profile["profile_sha256"])
            result_by_id[path.stem] = row
    ordered = lambda: [result_by_id[c["case_id"]] for c in plan["cases"] if c["case_id"] in result_by_id]
    print(f"[v4.3-audit] mode={args.mode} planned={len(plan['cases'])} reused={len(result_by_id)} excluded_samples={len(plan['excluded_samples'])}", flush=True)
    if args.plan_only:
        return
    if args.validate_only:
        if len(result_by_id) != len(plan["cases"]):
            raise ValueError("Audit is incomplete")
        old = read_json(args.output_dir / "v4_3_audit_report.json")
        authenticate(old, "report_sha256", "audit report")
        reconstructed = make_report(plan, ordered(), profile)
        if old != reconstructed:
            raise ValueError("Report differs from case results")
        if args.mode == "smoke" and not smoke_integrity(plan, ordered()):
            raise ValueError("Smoke integrity failed")
        print("[v4.3-audit] authenticated complete result", flush=True)
        return
    if len(result_by_id) == len(plan["cases"]):
        report = write_report(args, plan, ordered(), profile)
        if args.mode == "smoke" and not report["smoke_integrity_passed"]:
            raise RuntimeError("Completed smoke has no qualifying causal observability; inspect report")
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from memgen.model.side_kv import SideKVAttentionController
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v4_3_runtime import V43UnifiedRuntime
    from memgen.model.v4_3_side_kv import MEMORY_SCORE_BIAS
    reasoner = profile["experiment"]["reasoner"]
    tokenizer = AutoTokenizer.from_pretrained(reasoner["model_name"], revision=reasoner["tokenizer_revision"])
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Validate every selected native prefix before any branch is generated.
    prefixes = {c["case_id"]: checked_prefix(c, evidence[c["experience_id"]], tokenizer) for c in plan["cases"]}
    risk = torch.load(args.token_risk_artifact, map_location="cpu", weights_only=False)
    if any(risk["reasoner"].get(k) != v for k, v in reasoner.items()):
        raise ValueError("Gate and compiled Memory reasoner identities differ")
    gate = EntropyHysteresisGate.from_token_artifact(risk)
    model = AutoModelForCausalLM.from_pretrained(reasoner["model_name"], revision=reasoner["model_revision"],
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    if (getattr(model.config, "_commit_hash", reasoner["model_revision"]) != reasoner["model_revision"]
            or tokenizer.init_kwargs.get("_commit_hash", reasoner["tokenizer_revision"]) != reasoner["tokenizer_revision"]):
        raise ValueError("Loaded model/tokenizer revision mismatch")
    controller = SideKVAttentionController(model=model, layer_number=24, audit_canonical_rope=True,
                                           memory_score_normalization="log_valid_slots", memory_score_bias=MEMORY_SCORE_BIAS)
    runtime = V43UnifiedRuntime(model=model, tokenizer=tokenizer, device=args.device, gate=gate, controller=controller)
    descriptors = {r["bank_id"]: r["descriptor"] for r in records}
    try:
        for i, case in enumerate(plan["cases"], 1):
            cid = case["case_id"]
            if cid in result_by_id:
                continue
            e, prefix = evidence[case["experience_id"]], prefixes[cid]
            memories = case["memories"]
            if case["audit_layer"] == "visible_content":
                branches, parity = runtime.run_visible(question=e["question"],
                    descriptors={name: None if bid is None else descriptors[bid] for name, bid in memories.items()}, memory_ids=memories)
            else:
                loaded = {name: None if bid is None else loaders[bid].get_memory(bid, device=args.device, dtype=torch.bfloat16) for name, bid in memories.items()}
                branches, parity = runtime.run_latent(prefix=prefix, prompt_count=case["prompt_token_count"], memories=loaded)
                del loaded
            # Gold solution access is confined to scoring after all branches.
            scored = {name: score_branch(tokenizer, prefix, case["prompt_token_count"], b, e["official_solution"]) for name, b in branches.items()}
            row = seal({"schema_version": "memgen-v4.3-audit-result-v1", "profile_sha256": profile["profile_sha256"],
                **{key: case[key] for key in ("case_id", "case_sha256", "audit_layer", "sample_id", "experience_id", "bank_id", "quality_tier", "semantic_category", "gate_attempt_index")},
                "prefix_cache_parity": parity, "branches": scored})
            validate_result(row, case, profile["profile_sha256"])
            atomic_json(directory / (cid + ".json"), row, immutable=True)
            result_by_id[cid] = row
            write_report(args, plan, ordered(), profile)
            outcomes = " ".join(f"{name}={b['strict_reward']:.0f}" for name, b in scored.items())
            print(f"[v4.3-audit] {i}/{len(plan['cases'])} layer={case['audit_layer']} tier={case['quality_tier']} {outcomes}", flush=True)
    finally:
        controller.close()
    report = write_report(args, plan, ordered(), profile)
    if args.mode == "smoke" and not report["smoke_integrity_passed"]:
        raise RuntimeError("Smoke complete but causal observability failed; full remains blocked")
    print(f"[v4.3-audit] complete report={args.output_dir / 'v4_3_audit_report.json'}", flush=True)


if __name__ == "__main__":
    main()
