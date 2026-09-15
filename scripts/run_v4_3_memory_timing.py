#!/usr/bin/env python3
"""Compare frozen semantic routing with prefix, prompt-end and gated memory."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import atomic_json, read_json, implementation_hashes
from memgen.experience.v4_3_bank import authenticate, canonical_hash, file_hash, seal
from memgen.experience.v4_3_question_selector import NO_MEMORY, paired_metrics, checked_dataset_rows, fit_selector
from scripts import run_v4_3_question_selector as source
from scripts.audit_v4_3_unified_memory import score_branch

BRANCHES = ("baseline", "native_prefix_kv", "prompt_end", "entropy_gate")
IMPLEMENTATION = ("scripts/run_v4_3_memory_timing.py", "memgen/model/v4_3_memory_timing.py")


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
    import torch
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v4_3_memory_timing import WRAPPER
    sp = read_json(args.selector_dir / "profile.json")
    authenticate(sp, "profile_sha256", "frozen selector profile")
    counts = Counter(e["selector_split"] for e in sp["samples"])
    old_args = SimpleNamespace(**vars(args), train_size=counts["train"], tune_size=counts["tune"],
                              eval_size=counts["eval"], seed=sp["seed"])
    prepared, eq, expected = source.prepare(old_args)
    if sp != expected:
        raise ValueError("Frozen selector source identity drift")
    selector = read_json(args.selector_dir / "selector.json")
    authenticate(selector, "selector_sha256", "frozen selector")
    cards = source.bound_read(args.selector_dir / "card_features.json", sp["profile_sha256"])["features"]
    rows = {s: [source.assemble_row(args.selector_dir, e, sp) for e in sp["samples"] if e["selector_split"] == s]
            for s in ("train", "tune")}
    if selector != fit_selector(rows["train"], rows["tune"], sp["bank_ids"], cards, sp["profile_sha256"]):
        raise ValueError("Source selector differs from its train/tune evidence")
    report = source.report(args.selector_dir, sp, selector)
    if read_json(args.selector_dir / "report.json") != report or not report["complete"]:
        raise ValueError("A complete authenticated selector experiment is required")
    entries = [e for e in sp["samples"] if e["selector_split"] == "eval"]
    references, bindings = {}, {}
    for e in entries:
        root = source.sample_path(args.selector_dir, e)
        common = dict(sample_id=e["sample_id"], question_sha256=e["question_sha256"])
        prediction = source.bound_read(root / "prediction.json", sp["profile_sha256"],
                                       selector_sha256=selector["selector_sha256"], **common)
        bank = prediction["decision"]["semantic_bank"]
        actions = {"baseline": NO_MEMORY, "native_prefix_kv": bank, "fixed_from_train": selector["fixed_bank_from_train"]}
        refs = {name: source.bound_read(root / (action+".json"), sp["profile_sha256"], action=action, **common)
                for name, action in actions.items()}
        references[e["sample_id"]] = {name: row["result"] for name, row in refs.items()}
        bindings[e["sample_id"]] = {"selected_bank": bank, "prediction_sha256": prediction["record_sha256"],
                                   "reference_sha256": {name: row["record_sha256"] for name, row in refs.items()}}
    risk = torch.load(args.token_risk_artifact, map_location="cpu", weights_only=False)
    if any(risk["reasoner"].get(k) != v for k, v in eq["source_experiment"]["source_reasoner"].items()):
        raise ValueError("Frozen gate reasoner mismatch")
    gate = EntropyHysteresisGate.from_token_artifact(risk)
    profile = seal({"schema_version": "v43-memory-timing-v1", "source_profile_sha256": sp["profile_sha256"],
        "selector_sha256": selector["selector_sha256"], "source_report_sha256": report["report_sha256"],
        "selector": "frozen_semantic_question_only", "semantic_threshold": selector["semantic_threshold"],
        "samples": entries, "bindings": bindings, "bank_ids": sp["bank_ids"], "dataset": sp["dataset"],
        "reasoner": sp["reasoner"], "runtime_versions": sp["runtime_versions"], "device": args.device,
        "risk_file_sha256": file_hash(args.token_risk_artifact), "gate_config": gate.config.to_dict(),
        "branches": list(BRANCHES), "reference_branches_reused": ["baseline", "native_prefix_kv"],
        "fixed_bank_reference": selector["fixed_bank_from_train"], "delayed_wrapper": WRAPPER,
        "maximum_completion_tokens": 1024, "memory_tokens_count_against_completion_budget": False,
        "max_injections": 1, "gate_observation": "every_generated_pre_answer_token_until_first_joint_trigger",
        "delayed_kv": "contextual_prefill_all_layers_native_positions_until_end_no_bias",
        "prefix_and_delayed_wrappers_differ": True, "official_test_used": False,
        "evaluation_role": "diagnostic_reuse_of_observed_selector_eval_not_fresh_test",
        "external_api_calls_made": 0, "implementation_sha256": implementation_hashes(IMPLEMENTATION)}, "profile_sha256")
    return profile, references, prepared[3], gate


def decision_record(profile, entry):
    return seal({"profile_sha256": profile["profile_sha256"], "sample_id": entry["sample_id"],
                 "question_sha256": entry["question_sha256"], **profile["bindings"][entry["sample_id"]]})


def check_result(row, profile, entry, branch):
    authenticate(row, "record_sha256", "timing branch")
    if any(row.get(k) != v for k, v in {"profile_sha256": profile["profile_sha256"],
           "sample_id": entry["sample_id"], "question_sha256": entry["question_sha256"], "branch": branch,
           "selected_bank": profile["bindings"][entry["sample_id"]]["selected_bank"]}.items()):
        raise ValueError("Timing branch identity drift")
    r = row["result"]
    ids = r["continuation_token_ids"]
    if not 0 < len(ids) <= 1024 or canonical_hash(ids) != r["continuation_token_ids_sha256"] or r["strict_reward"] not in (0., 1.):
        raise ValueError("Invalid timing completion/reward")
    if branch in ("prompt_end", "entropy_gate"):
        active, inj = r["activation_count"], r["injection"]
        if (r["timing"] != branch or r["final_cache_length"] != r["initial_cache_length"] + len(ids) + r["injected_token_count"]
                or r["injection_generated_token_count"] != (inj["generated_token_count"] if inj else None)):
            raise ValueError("Timing/cache accounting drift")
        if active not in (0, 1) or bool(active) != (inj is not None):
            raise ValueError("Invalid injection count")
        if row["selected_bank"] == NO_MEMORY and (active or r["gate_traces"]):
            raise ValueError("Selector abstention injected/probed memory")
        if branch == "prompt_end" and row["selected_bank"] != NO_MEMORY and not active:
            raise ValueError("Missing prompt-end injection")
        if sum(t["triggered"] for t in r["gate_traces"]) != (active if branch == "entropy_gate" else 0):
            raise ValueError("Gate trace/activation mismatch")
        if inj:
            n = inj["generated_token_count"]
            if not 0 <= n < len(ids) or (branch == "prompt_end" and n != 0) or (branch == "entropy_gate" and n < 1):
                raise ValueError("Invalid injection position")
            if (inj["cache_length_after"] - inj["cache_length_before"] != r["injected_token_count"]
                    or inj["memory_token_count"] != r["injected_token_count"]
                    or inj["pre_injection_completion_sha256"] != canonical_hash(ids[:n])):
                raise ValueError("Injection cache/token accounting mismatch")


def summarize(profile, cases, references):
    def group(items, reference="baseline"):
        if not items:
            return {b: {"count": 0, "accuracy": None} for b in BRANCHES}
        rows = [{"rewards": {NO_MEMORY: references[sid][reference]["strict_reward"],
                             **{b: r["strict_reward"] for b, r in branches.items()}}} for sid, branches in items]
        metrics = {b: paired_metrics(rows, [b]*len(rows)) for b in BRANCHES}
        for b, m in metrics.items():
            m.pop("selection_counts")
            m["memory_use_count"] = sum(
                (profile["bindings"][sid]["selected_bank"] != NO_MEMORY if b == "native_prefix_kv"
                 else bool(branches[b].get("activation_count", 0)) if b != "baseline" else False)
                for sid, branches in items)
        return metrics
    items = list(cases.items())
    gated = [(s, r) for s, r in items if r["entropy_gate"]["activation_count"]]
    selected = [(s, r) for s, r in items if profile["bindings"][s]["selected_bank"] != NO_MEMORY]
    no_trigger = [(s, r) for s, r in selected if not r["entropy_gate"]["activation_count"]]
    positions = [r["entropy_gate"]["injection_generated_token_count"] for _, r in gated]
    fixed_rows = [{"rewards": {NO_MEMORY: references[s]["baseline"]["strict_reward"],
                              "fixed": references[s]["fixed_from_train"]["strict_reward"]}} for s, _ in items]
    fixed = paired_metrics(fixed_rows, ["fixed"]*len(fixed_rows))
    fixed.pop("selection_counts")
    fixed["memory_use_count"] = len(items) if profile["fixed_bank_reference"] != NO_MEMORY else 0
    return seal({"complete": len(items) == len(profile["samples"]), "profile_sha256": profile["profile_sha256"],
        "sample_count": len(items), "bank_count": len(profile["bank_ids"]), "selector": profile["selector"],
        "semantic_threshold": profile["semantic_threshold"], "evaluation_role": profile["evaluation_role"],
        "overall": group(items), "fixed_from_train_reference": fixed,
        "paired_against_native_prefix": group(items, "native_prefix_kv"),
        "paired_against_fixed": group(items, "fixed_from_train"),
        "subgroups": {"selector_selected": group(selected), "gate_activated": group(gated),
                      "selected_but_no_trigger": group(no_trigger)},
        "gate": {"selected_count": len(selected), "activation_count": len(gated),
                 "selected_but_no_trigger_count": len(no_trigger),
                 "activation_rate_among_selected": len(gated)/len(selected) if selected else None,
                 "injection_generated_tokens": {"min": min(positions) if positions else None,
                     "mean": sum(positions)/len(positions) if positions else None, "max": max(positions) if positions else None},
                 "non_activation_reasons": dict(Counter(r["entropy_gate"]["non_activation_reason"] for _, r in items
                                                         if not r["entropy_gate"]["activation_count"]))},
        "integrity": {"pre_gate_prefix_mismatch_count": sum(
            r["entropy_gate"]["continuation_token_ids"][:r["entropy_gate"]["injection_generated_token_count"]]
            != r["baseline"]["continuation_token_ids"][:r["entropy_gate"]["injection_generated_token_count"]] for _, r in gated),
            "no_trigger_trajectory_mismatch_count": sum(r["entropy_gate"]["continuation_token_ids"] != r["baseline"]["continuation_token_ids"] for _, r in no_trigger)},
        "external_api_calls_made": 0, "offline_kv_splice": False}, "summary_sha256")


def main():
    args = parse_args()
    profile, references, records, gate = prepare(args)
    out = args.output_dir
    sources = (args.selector_dir, args.equivalence_dir, args.bank_dir, args.side_kv_dir, args.cache_manifest.parent)
    if out.is_symlink() or any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents or out.resolve() in p.resolve().parents for p in sources):
        raise ValueError("Timing output must be separate from source artifacts")
    if out.exists() and any(out.iterdir()) and not (args.resume or args.validate_only):
        raise ValueError("Output exists; pass --resume")
    pp = out / "profile.json"
    if pp.exists() and read_json(pp) != profile:
        raise ValueError("Timing experiment identity drift; use a new output directory")
    if args.plan_only:
        selected = sum(v["selected_bank"] != NO_MEMORY for v in profile["bindings"].values())
        print(f"[memory-timing] samples={len(profile['samples'])} selected={selected} new_rollouts={2*selected} references=reused gate={profile['gate_config']}", flush=True)
        return
    if args.validate_only:
        if not pp.exists():
            raise ValueError("Missing timing profile")
    else:
        atomic_json(pp, profile, immutable=True)
    cases, pending = {}, []
    for e in profile["samples"]:
        sid = e["sample_id"]
        root = source.sample_path(out, e)
        decision = decision_record(profile, e)
        dp = root / "decision.json"
        if dp.exists():
            if read_json(dp) != decision:
                raise ValueError("Frozen timing decision drift")
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Timing outcomes lack a prior frozen decision")
        elif args.validate_only:
            raise ValueError("Missing timing decision")
        else:
            atomic_json(dp, decision, immutable=True)
        cases[sid] = {}
        for branch in BRANCHES:
            path = root / (branch+".json")
            if path.exists():
                row = read_json(path)
                check_result(row, profile, e, branch)
                if branch in ("baseline", "native_prefix_kv") and row["result"] != references[sid][branch]:
                    raise ValueError("Reused reference result drift")
                cases[sid][branch] = row["result"]
            else:
                pending.append((e, branch))
    if args.validate_only and pending:
        raise ValueError("Incomplete timing experiment")
    runtime = None
    try:
        need_model = any(b in ("prompt_end", "entropy_gate") and profile["bindings"][e["sample_id"]]["selected_bank"] != NO_MEMORY for e, b in pending)
        if need_model:
            from datasets import load_dataset
            from memgen.model.v4_3_question_selector import load_runtime
            from memgen.model.v4_3_memory_timing import generate_delayed, memory_tokens
            questions = checked_dataset_rows(load_dataset("openai/gsm8k", "main", revision=profile["dataset"]["revision"], split="train"), profile["samples"])
            runtime = load_runtime(profile["reasoner"], args.device)
            runtime.gate = gate
            descriptors = {r["bank_id"]: r["descriptor"] for r in records}
            for e in profile["samples"]:
                bank = profile["bindings"][e["sample_id"]]["selected_bank"]
                if bank != NO_MEMORY:
                    q = questions[e["sample_id"]]["question"]
                    if len(runtime.visible_prefix(q, None)) + len(memory_tokens(runtime, descriptors[bank])) + 1024 > runtime.model.config.max_position_embeddings:
                        raise ValueError("Planned contextual memory exceeds context")
        for index, (e, branch) in enumerate(pending, 1):
            sid = e["sample_id"]
            bank = profile["bindings"][sid]["selected_bank"]
            if branch in ("baseline", "native_prefix_kv"):
                result = references[sid][branch]
            elif bank == NO_MEMORY:
                result = {**references[sid]["baseline"], "timing": branch, "injection": None,
                          "injected_token_count": 0, "injection_generated_token_count": None,
                          "gate_traces": [], "non_activation_reason": "selector_abstained"}
            else:
                print(f"[memory-timing] pending={index}/{len(pending)} sample={sid} branch={branch}", flush=True)
                q = questions[sid]
                prefix, generated = generate_delayed(runtime, q["question"], descriptors[bank], branch)
                result = score_branch(runtime.tokenizer, prefix, len(prefix), generated, q["answer"])
            row = seal({"profile_sha256": profile["profile_sha256"], "sample_id": sid,
                        "question_sha256": e["question_sha256"], "selected_bank": bank, "branch": branch, "result": result})
            check_result(row, profile, e, branch)
            atomic_json(source.sample_path(out, e) / (branch+".json"), row, immutable=True)
            cases[sid][branch] = result
        value = summarize(profile, cases, references)
        if args.validate_only:
            if read_json(out / "report.json") != value:
                raise ValueError("Timing report drift")
        else:
            atomic_json(out / "report.json", value)
            brief = {k: v for k, v in value.items() if k not in {"subgroups", "paired_against_fixed", "paired_against_native_prefix", "summary_sha256", "profile_sha256"}}
            atomic_json(out / "brief_summary.json", seal(brief, "summary_sha256"))
        print(f"[memory-timing] complete summary={out / 'brief_summary.json'}", flush=True)
    finally:
        if runtime is not None:
            runtime.controller.close()


if __name__ == "__main__":
    main()
