#!/usr/bin/env python3
"""Frozen semantic routing with delayed visibility of all-layer native prefix KV."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import atomic_json, read_json, implementation_hashes
from memgen.experience.v4_3_bank import authenticate, canonical_hash, seal
from memgen.experience.v4_3_question_selector import NO_MEMORY, paired_metrics, checked_dataset_rows
from scripts import run_v4_3_memory_timing as timing
from scripts import run_v4_3_question_selector as source
from scripts.audit_v4_3_unified_memory import score_branch

BRANCHES = ("baseline", "native_prefix_kv", "gated_prefix_kv")
IMPLEMENTATION = ("scripts/run_v4_3_gated_prefix.py", "memgen/model/v4_3_gated_prefix.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-dir", "side-kv-dir", "semantic-packets", "cache-manifest", "token-risk-artifact",
                 "equivalence-dir", "split-manifest", "selector-dir", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    for flag in ("resume", "validate-only", "plan-only"):
        p.add_argument("--"+flag, action="store_true")
    return p.parse_args()


def prepare(args):
    # Read-only reuse of the complete selector provenance audit. No timing run required.
    previous, references, records, gate = timing.prepare(args)
    keys = ("source_profile_sha256", "selector_sha256", "source_report_sha256", "selector", "semantic_threshold",
            "samples", "bindings", "bank_ids", "dataset", "reasoner", "runtime_versions", "device",
            "risk_file_sha256", "gate_config", "fixed_bank_reference", "evaluation_role")
    eq = read_json(args.equivalence_dir / "profile.json")
    authenticate(eq, "profile_sha256", "frozen native prefix")
    profile = seal({**{k: previous[k] for k in keys}, "schema_version": "v43-gated-native-prefix-v1",
        "provenance_audit_sha256": previous["profile_sha256"], "source_equivalence_profile_sha256": eq["profile_sha256"],
        "branches": list(BRANCHES), "reference_branches_reused": ["baseline", "native_prefix_kv"],
        "memory_source": "unchanged_all_layer_native_prefix_including_system_wrapper",
        "memory_position_policy": "post_rope_keys_shifted_by_minus_memory_length_to_virtual_negative_prefix",
        "rope_support": "default_fixed_frequency_only", "memory_score_bias": 0., "memory_score_normalization": "none",
        "visibility": "first_forward_after_joint_gate_observation_until_generation_end",
        "gate_observation": "every_generated_pre_answer_token_until_first_joint_trigger",
        "max_activations": 1, "all_layers": True, "maximum_completion_tokens": 1024,
        "history_replay": False, "native_position_shift": False, "memory_tokens_inserted": False,
        "offline_prefix_kv_read": True, "official_test_used": False, "external_api_calls_made": 0,
        "implementation_sha256": implementation_hashes(IMPLEMENTATION)}, "profile_sha256")
    return profile, references, records, gate


def check_result(row, profile, entry, branch, reference):
    authenticate(row, "record_sha256", "gated prefix result")
    expected = {"profile_sha256": profile["profile_sha256"], "sample_id": entry["sample_id"],
                "question_sha256": entry["question_sha256"], "branch": branch,
                "selected_bank": profile["bindings"][entry["sample_id"]]["selected_bank"]}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Gated prefix result identity drift")
    r = row["result"]
    ids = r["continuation_token_ids"]
    if not 0 < len(ids) <= 1024 or canonical_hash(ids) != r["continuation_token_ids_sha256"] or r["strict_reward"] not in (0., 1.):
        raise ValueError("Invalid completion/reward")
    if branch != "gated_prefix_kv":
        if r != reference[branch]:
            raise ValueError("Frozen reference result drift")
        return
    if r["inserted_token_count"] != 0 or r["replayed_token_count"] != 0 or not r["offline_prefix_kv_read"]:
        raise ValueError("Gated prefix must not insert tokens or replay history")
    if r["final_cache_length"] != r["initial_cache_length"] + len(ids):
        raise ValueError("Memory enlarged native cache")
    active, activation, traces = r["activation_count"], r["activation"], r["gate_traces"]
    triggered = r["gate_trigger_count"]
    if active not in (0, 1) or triggered not in (0, 1) or active > triggered or bool(active) != (activation is not None):
        raise ValueError("Invalid gate/activation count")
    if sum(t["triggered"] for t in traces) != triggered or [t["generated_input_index"] for t in traces] != list(range(len(traces))):
        raise ValueError("Gate trace coverage mismatch")
    config = profile["gate_config"]
    for t in traces:
        if t["triggered"] != (t["entropy"] >= config["high_entropy_threshold"] and t["risk_score"] > config["risk_threshold"]):
            raise ValueError("Gate thresholds drifted")
    if expected["selected_bank"] == NO_MEMORY:
        if active or traces or ids != reference["baseline"]["continuation_token_ids"] or r["strict_reward"] != reference["baseline"]["strict_reward"]:
            raise ValueError("Selector abstention deviated from baseline")
    if active:
        n = activation["unchanged_generated_token_count"]
        if (not 2 <= n < len(ids) or activation["first_memory_query_generated_index"] != n-1
                or activation["trigger_generated_input_index"] != n-2
                or activation["native_cache_length_before"] != r["initial_cache_length"] + n
                or activation["unchanged_completion_sha256"] != canonical_hash(ids[:n])
                or not traces[-1]["triggered"] or traces[-1]["generated_input_index"] != n-2):
            raise ValueError("Activation must start after the completed native gate forward")
        steps = len(ids)-n
        layer_count = len(r["layer_read_counts"])
        if (r["active_forward_steps"] != steps or r["layer_read_counts"] != [steps]*layer_count
                or layer_count <= 0 or set(r["first_read_by_layer"]) != {str(i+1) for i in range(layer_count)}
                or r["history_kv_preserved"] is not True):
            raise ValueError("All-layer read/history preservation check failed")
        for observation in r["first_read_by_layer"].values():
            if (not 0 < observation["memory_attention_mass"] <= 1
                    or observation["memory_key_length"] != r["memory_token_count"]
                    or observation["native_key_length"] != activation["native_cache_length_before"]+1):
                raise ValueError("First memory read accounting mismatch")
    elif r["active_forward_steps"] or any(r["layer_read_counts"]) or r["first_read_by_layer"]:
        raise ValueError("Unactivated memory has read traces")


def summarize(profile, cases, references):
    items = list(cases.items())
    selected = [(sid, r) for sid, r in items if profile["bindings"][sid]["selected_bank"] != NO_MEMORY]
    activated = [(sid, r) for sid, r in items if r["gated_prefix_kv"]["activation_count"]]
    not_activated = [(sid, r) for sid, r in selected if not r["gated_prefix_kv"]["activation_count"]]
    def group(subset, reference="baseline"):
        if not subset:
            return {b: {"count": 0, "accuracy": None} for b in BRANCHES}
        rows = [{"rewards": {NO_MEMORY: references[sid][reference]["strict_reward"],
                             **{b: r[b]["strict_reward"] for b in BRANCHES}}} for sid, r in subset]
        result = {b: paired_metrics(rows, [b]*len(rows)) for b in BRANCHES}
        for b, metrics in result.items():
            metrics.pop("selection_counts")
            metrics["memory_use_count"] = sum((profile["bindings"][sid]["selected_bank"] != NO_MEMORY if b == "native_prefix_kv"
                else r[b]["activation_count"] if b == "gated_prefix_kv" else 0) for sid, r in subset)
        return result
    fixed_rows = [{"rewards": {NO_MEMORY: references[sid]["baseline"]["strict_reward"],
                             "fixed": references[sid]["fixed_from_train"]["strict_reward"]}} for sid, _ in items]
    fixed = paired_metrics(fixed_rows, ["fixed"]*len(items))
    fixed.pop("selection_counts")
    fixed["memory_use_count"] = len(items) if profile["fixed_bank_reference"] != NO_MEMORY else 0
    positions = [r["gated_prefix_kv"]["activation"]["unchanged_generated_token_count"] for _, r in activated]
    masses = [o["memory_attention_mass"] for _, r in activated for o in r["gated_prefix_kv"]["first_read_by_layer"].values()]
    return seal({"complete": len(items) == len(profile["samples"]), "profile_sha256": profile["profile_sha256"],
        "sample_count": len(items), "bank_count": len(profile["bank_ids"]), "selector": profile["selector"],
        "semantic_threshold": profile["semantic_threshold"], "evaluation_role": profile["evaluation_role"],
        "overall": group(items), "fixed_from_train_reference": fixed,
        "paired_against_native_prefix": group(items, "native_prefix_kv"), "paired_against_fixed": group(items, "fixed_from_train"),
        "subgroups": {"selected": group(selected), "activated": group(activated), "selected_but_not_activated": group(not_activated)},
        "gate": {"selected_count": len(selected), "joint_trigger_count": sum(r["gated_prefix_kv"]["gate_trigger_count"] for _, r in items),
                 "activation_count": len(activated), "selected_but_not_activated_count": len(not_activated),
                 "activation_rate_among_selected": len(activated)/len(selected) if selected else None,
                 "first_memory_read_after_generated_tokens": {"min": min(positions) if positions else None,
                     "mean": sum(positions)/len(positions) if positions else None, "max": max(positions) if positions else None},
                 "non_activation_reasons": dict(Counter(r["gated_prefix_kv"]["non_activation_reason"] for _, r in items
                                                          if not r["gated_prefix_kv"]["activation_count"]))},
        "attention": {"first_read_memory_mass_layer_mean": sum(masses)/len(masses) if masses else None,
                      "first_read_memory_mass_layer_min": min(masses) if masses else None,
                      "first_read_memory_mass_layer_max": max(masses) if masses else None},
        "integrity": {"history_kv_failure_count": sum(r["gated_prefix_kv"]["history_kv_preserved"] is not True for _, r in activated),
            "pre_activation_trajectory_mismatch_count": sum(
                r["gated_prefix_kv"]["continuation_token_ids"][:r["gated_prefix_kv"]["activation"]["unchanged_generated_token_count"]]
                != r["baseline"]["continuation_token_ids"][:r["gated_prefix_kv"]["activation"]["unchanged_generated_token_count"]] for _, r in activated),
            "no_activation_trajectory_mismatch_count": sum(r["gated_prefix_kv"]["continuation_token_ids"] != r["baseline"]["continuation_token_ids"] for _, r in not_activated),
            "inserted_token_count": sum(r["gated_prefix_kv"]["inserted_token_count"] for _, r in items),
            "replayed_token_count": sum(r["gated_prefix_kv"]["replayed_token_count"] for _, r in items)},
        "all_layers": True, "virtual_prefix_positions": True, "history_replay": False,
        "external_api_calls_made": 0, "offline_prefix_kv_read": True}, "summary_sha256")


def main():
    args = parse_args()
    profile, references, records, gate = prepare(args)
    out = args.output_dir
    sources = (args.selector_dir, args.equivalence_dir, args.bank_dir, args.side_kv_dir, args.cache_manifest.parent)
    if out.is_symlink() or any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents or out.resolve() in p.resolve().parents for p in sources):
        raise ValueError("Gated prefix output must be separate from source artifacts")
    if out.exists() and any(out.iterdir()) and not (args.resume or args.validate_only):
        raise ValueError("Output exists; pass --resume")
    pp = out / "profile.json"
    if pp.exists() and read_json(pp) != profile:
        raise ValueError("Gated prefix identity drift; use a new output directory")
    if args.plan_only:
        selected = sum(v["selected_bank"] != NO_MEMORY for v in profile["bindings"].values())
        print(f"[gated-prefix] samples={len(profile['samples'])} new_rollouts={selected} references=reused replay=false all_layers=true", flush=True)
        return
    if args.validate_only:
        if not pp.exists():
            raise ValueError("Missing gated prefix profile")
    else:
        atomic_json(pp, profile, immutable=True)
    cases, pending = {}, []
    for e in profile["samples"]:
        sid = e["sample_id"]
        root = source.sample_path(out, e)
        dp = root / "decision.json"
        decision = timing.decision_record(profile, e)
        if dp.exists():
            if read_json(dp) != decision:
                raise ValueError("Frozen decision drift")
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Outcomes lack a prior frozen decision")
        elif args.validate_only:
            raise ValueError("Missing frozen decision")
        else:
            atomic_json(dp, decision, immutable=True)
        cases[sid] = {}
        for branch in BRANCHES:
            path = root / (branch+".json")
            if path.exists():
                row = read_json(path)
                check_result(row, profile, e, branch, references[sid])
                cases[sid][branch] = row["result"]
            else:
                pending.append((e, branch))
    if args.validate_only and pending:
        raise ValueError("Incomplete gated prefix experiment")
    runtime = None
    try:
        need_model = any(b == "gated_prefix_kv" and profile["bindings"][e["sample_id"]]["selected_bank"] != NO_MEMORY for e, b in pending)
        from memgen.model.v4_3_gated_prefix import abstained_result
        if need_model:
            from datasets import load_dataset
            from memgen.model.v4_3_question_selector import load_runtime
            from memgen.model.v4_3_prefix_equivalence import prefix_bank
            from memgen.model.v4_3_gated_prefix import generate_gated, virtual_prefix
            questions = checked_dataset_rows(load_dataset("openai/gsm8k", "main", revision=profile["dataset"]["revision"], split="train"), profile["samples"])
            runtime = load_runtime(profile["reasoner"], args.device)
            runtime.gate = gate
            runtime.controller.close()
            memories = {}
            for r in records:
                ids, tensors = prefix_bank(args.equivalence_dir / "prefix_kv", r, runtime,
                                          profile["source_equivalence_profile_sha256"], validate_only=True)
                memories[r["bank_id"]] = virtual_prefix(runtime.model, ids, tensors)
            for e in profile["samples"]:
                bank = profile["bindings"][e["sample_id"]]["selected_bank"]
                if bank != NO_MEMORY:
                    q = questions[e["sample_id"]]["question"]
                    if len(runtime.visible_prefix(q, None))+memories[bank][0][0].shape[-2]+1024 > runtime.model.config.max_position_embeddings:
                        raise ValueError("Planned native+memory relative span exceeds context")
        for index, (e, branch) in enumerate(pending, 1):
            sid = e["sample_id"]
            bank = profile["bindings"][sid]["selected_bank"]
            if branch != "gated_prefix_kv":
                result = references[sid][branch]
            elif bank == NO_MEMORY:
                result = abstained_result(references[sid]["baseline"])
            else:
                print(f"[gated-prefix] pending={index}/{len(pending)} sample={sid} bank={bank}", flush=True)
                q = questions[sid]
                prefix, generated = generate_gated(runtime, q["question"], memories[bank])
                result = score_branch(runtime.tokenizer, prefix, len(prefix), generated, q["answer"])
                print(f"[gated-prefix] trigger={result['gate_trigger_count']} activated={result['activation_count']} history_preserved={result['history_kv_preserved']}", flush=True)
            row = seal({"profile_sha256": profile["profile_sha256"], "sample_id": sid,
                        "question_sha256": e["question_sha256"], "selected_bank": bank, "branch": branch, "result": result})
            check_result(row, profile, e, branch, references[sid])
            atomic_json(source.sample_path(out, e) / (branch+".json"), row, immutable=True)
            cases[sid][branch] = result
        value = summarize(profile, cases, references)
        if args.validate_only:
            if read_json(out / "report.json") != value:
                raise ValueError("Gated prefix report drift")
        else:
            atomic_json(out / "report.json", value)
            brief = {k: v for k, v in value.items() if k not in {"subgroups", "paired_against_fixed", "paired_against_native_prefix", "summary_sha256", "profile_sha256"}}
            atomic_json(out / "brief_summary.json", seal(brief, "summary_sha256"))
        print(f"[gated-prefix] complete summary={out / 'brief_summary.json'}", flush=True)
    finally:
        if runtime is not None:
            runtime.controller.close()


if __name__ == "__main__":
    main()
