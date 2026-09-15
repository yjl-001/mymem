#!/usr/bin/env python3
"""Full official final-test: baseline, semantic native prefix, and gated prefix KV."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.phase1 import SPLIT_MANIFEST_SCHEMA, canonical_json_sha256
from memgen.experience.v4_3_artifacts import atomic_json, read_json, implementation_hashes
from memgen.experience.v4_3_bank import authenticate, canonical_hash, seal
from memgen.experience.v4_3_question_selector import NO_MEMORY, checked_dataset_rows, paired_metrics, predict
from scripts import run_v4_3_gated_prefix as gated
from scripts import run_v4_3_question_selector as source
from scripts.audit_v4_3_unified_memory import score_branch

BRANCHES = gated.BRANCHES
IMPLEMENTATION = ("scripts/run_v4_3_final_test.py",)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-dir", "side-kv-dir", "semantic-packets", "cache-manifest", "token-risk-artifact",
                 "equivalence-dir", "split-manifest", "selector-dir", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    for flag in ("resume", "validate-only", "plan-only"):
        p.add_argument("--"+flag, action="store_true")
    return p.parse_args()


def final_entries(manifest):
    logical = {k: v for k, v in manifest.items() if k not in {"created_at", "manifest_sha256"}}
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA or canonical_json_sha256(logical) != manifest.get("manifest_sha256"):
        raise ValueError("Invalid final-test split manifest")
    entries = manifest["samples"]
    selected = sorted((e for e in entries if e["logical_split"] == "final-test"), key=lambda e: e["source_index"])
    n = manifest["dataset"]["test_size"]
    other_hashes = {e["question_sha256"] for e in entries if e["logical_split"] != "final-test"}
    if (not selected or len(selected) != n or manifest["counts"]["final-test"] != n
            or [e["source_index"] for e in selected] != list(range(n))
            or len({e["sample_id"] for e in entries}) != len(entries)
            or any(e["dataset_split"] != "test" or e["question_sha256"] in other_hashes for e in selected)
            or any(e["dataset_split"] == "test" and e["logical_split"] != "final-test" for e in entries)):
        raise ValueError("Final-test must cover the entire official test with no cross-split overlap")
    return selected


def prepare(args):
    previous, _, records, gate = gated.prepare(args)
    manifest = read_json(args.split_manifest)
    entries = final_entries(manifest)
    selector = read_json(args.selector_dir / "selector.json")
    authenticate(selector, "selector_sha256", "frozen selector")
    if selector["selector_sha256"] != previous["selector_sha256"]:
        raise ValueError("Frozen selector identity changed")
    profile = {k: v for k, v in previous.items() if k not in {
        "profile_sha256", "samples", "bindings", "fixed_bank_reference", "reference_branches_reused", "implementation_sha256"}}
    profile.update(schema_version="v43-full-final-test-v1", samples=entries,
        source_gated_protocol_sha256=previous["profile_sha256"], split_manifest_sha256=manifest["manifest_sha256"],
        dataset=manifest["dataset"], evaluation_role="full_official_final_test_frozen_policy",
        official_test_used=True, final_test_tuning=False, reference_branches_reused=[],
        gold_access="scoring_only_after_all_three_branches_of_each_question",
        generated_token_policy="completion_token_ids_including_emitted_eos_excluding_question_and_memory",
        implementation_sha256=implementation_hashes(IMPLEMENTATION))
    return seal(profile, "profile_sha256"), selector, records, gate


def common(profile, entry):
    return dict(profile_sha256=profile["profile_sha256"], sample_id=entry["sample_id"], question_sha256=entry["question_sha256"])


def load_decision(path, profile, entry, selector):
    row = source.bound_read(path, profile["profile_sha256"], sample_id=entry["sample_id"],
                            question_sha256=entry["question_sha256"], selector_sha256=selector["selector_sha256"])
    if row["selected_bank"] != predict(selector, row["feature"])["semantic_bank"]:
        raise ValueError("Final-test decision differs from frozen question-only semantic selector")
    return row


def validate_raw(rows, profile, entry, decision):
    bank = decision["selected_bank"]
    bound_profile = {**profile, "bindings": {entry["sample_id"]: {"selected_bank": bank}}}
    refs = {b: {**row["result"], "strict_reward": 0.} for b, row in rows.items() if b != "gated_prefix_kv"}
    for branch, row in rows.items():
        authenticate(row, "record_sha256", "final-test raw generation")
        if "strict_reward" in row["result"] or "full_completion" in row["result"]:
            raise ValueError("Gold scoring must follow all raw branches")
        if row.get("decision_sha256") != decision["record_sha256"] or row.get("selected_bank") != bank:
            raise ValueError("Raw branch decision binding drift")
        proxy = seal({**common(profile, entry), "selected_bank": bank, "branch": branch,
                      "result": {**row["result"], "strict_reward": 0.}})
        if any(row.get(k) != v for k, v in common(profile, entry).items()) or row.get("branch") != branch:
            raise ValueError("Raw branch sample/profile drift")
        if branch == "gated_prefix_kv" and "baseline" not in refs:
            raise ValueError("Gated branch requires saved baseline")
        gated.check_result(proxy, bound_profile, entry, branch, refs)
    if bank == NO_MEMORY and "native_prefix_kv" in rows:
        if rows["native_prefix_kv"]["result"] != rows["baseline"]["result"]:
            raise ValueError("Abstained prefix must reuse the same-question baseline")


def load_scored(path, profile, entry, raw, decision):
    row = source.bound_read(path, profile["profile_sha256"], sample_id=entry["sample_id"],
                           question_sha256=entry["question_sha256"], decision_sha256=decision["record_sha256"],
                           selected_bank=decision["selected_bank"])
    if set(raw) != set(BRANCHES) or row["raw_record_sha256"] != {b: raw[b]["record_sha256"] for b in BRANCHES} or set(row["branches"]) != set(BRANCHES):
        raise ValueError("Scored case lacks its three original generations")
    for b, result in row["branches"].items():
        if result["strict_reward"] not in (0., 1.) or any(result.get(k) != v for k, v in raw[b]["result"].items()):
            raise ValueError("Scoring changed generation or contains invalid reward")
    return row


def token_stats(lengths):
    ordered = sorted(lengths)
    return {"total": sum(lengths), "mean": statistics.mean(lengths), "median": statistics.median(lengths),
            "min": min(lengths), "max": max(lengths), "p90": ordered[(9*len(ordered)+9)//10-1],
            "at_1024_token_limit_count": sum(n == 1024 for n in lengths)}


def summarize(profile, cases):
    rows = [{"rewards": {NO_MEMORY: c["branches"]["baseline"]["strict_reward"],
                         **{b: c["branches"][b]["strict_reward"] for b in BRANCHES}}} for c in cases]
    overall = {}
    for b in BRANCHES:
        m = paired_metrics(rows, [b]*len(rows))
        m.pop("selection_counts")
        m["memory_use_count"] = sum(c["selected_bank"] != NO_MEMORY if b == "native_prefix_kv"
            else c["branches"][b]["activation_count"] if b == "gated_prefix_kv" else 0 for c in cases)
        m["generated_tokens"] = token_stats([len(c["branches"][b]["continuation_token_ids"]) for c in cases])
        overall[b] = m
    gated_results = [c["branches"]["gated_prefix_kv"] for c in cases]
    active = [c for c in cases if c["branches"]["gated_prefix_kv"]["activation_count"]]
    positions = [c["branches"]["gated_prefix_kv"]["activation"]["unchanged_generated_token_count"] for c in active]
    paired_rows = [{"rewards": {NO_MEMORY: c["branches"]["native_prefix_kv"]["strict_reward"],
                                "gated": c["branches"]["gated_prefix_kv"]["strict_reward"]}} for c in cases]
    paired = paired_metrics(paired_rows, ["gated"]*len(cases))
    paired.pop("selection_counts")
    paired["memory_use_count"] = len(active)
    return seal({"complete": len(cases) == len(profile["samples"]), "sample_count": len(cases),
        "expected_sample_count": len(profile["samples"]), "profile_sha256": profile["profile_sha256"],
        "official_test_used": True, "evaluation_role": profile["evaluation_role"], "bank_count": len(profile["bank_ids"]),
        "selector": "frozen_semantic_question_only", "semantic_threshold": profile["semantic_threshold"],
        "overall": overall, "gated_vs_native_prefix": paired,
        "selection_counts": dict(Counter(c["selected_bank"] for c in cases)),
        "gate": {"selected_count": sum(c["selected_bank"] != NO_MEMORY for c in cases),
                 "joint_trigger_count": sum(r["gate_trigger_count"] for r in gated_results), "activation_count": len(active),
                 "first_read_after_generated_tokens_mean": statistics.mean(positions) if positions else None,
                 "non_activation_reasons": dict(Counter(r["non_activation_reason"] for r in gated_results if not r["activation_count"]))},
        "offline_memory_tokens_excluded_from_generation": {
            "native_prefix_input_tokens_total": sum(c["memory_token_count"] for c in cases if c["selected_bank"] != NO_MEMORY),
            "gated_memory_slots_exposed_total": sum(c["memory_token_count"] for c in active)},
        "integrity": {"history_kv_failure_count": sum(c["branches"]["gated_prefix_kv"]["history_kv_preserved"] is not True for c in active),
            "pre_activation_trajectory_mismatch_count": sum(
                c["branches"]["gated_prefix_kv"]["continuation_token_ids"][:c["branches"]["gated_prefix_kv"]["activation"]["unchanged_generated_token_count"]]
                != c["branches"]["baseline"]["continuation_token_ids"][:c["branches"]["gated_prefix_kv"]["activation"]["unchanged_generated_token_count"]] for c in active),
            "no_activation_trajectory_mismatch_count": sum(c["branches"]["gated_prefix_kv"]["continuation_token_ids"] != c["branches"]["baseline"]["continuation_token_ids"]
                                                          for c in cases if not c["branches"]["gated_prefix_kv"]["activation_count"])},
        "generated_token_policy": profile["generated_token_policy"], "final_test_tuning": False,
        "all_layers": True, "history_replay": False, "external_api_calls_made": 0}, "summary_sha256")


def main():
    args = parse_args()
    profile, selector, records, gate = prepare(args)
    out = args.output_dir
    sources = (args.selector_dir, args.equivalence_dir, args.bank_dir, args.side_kv_dir, args.cache_manifest.parent)
    if out.is_symlink() or any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents or out.resolve() in p.resolve().parents for p in sources):
        raise ValueError("Final-test output must be separate from source artifacts")
    if out.exists() and any(out.iterdir()) and not (args.resume or args.validate_only):
        raise ValueError("Output exists; pass --resume")
    pp = out / "profile.json"
    if pp.exists() and read_json(pp) != profile:
        raise ValueError("Final-test profile drift; frozen runs must not be mixed")
    if args.plan_only:
        print(f"[final-test] full_test_count={len(profile['samples'])} branches={list(BRANCHES)} all_policies_frozen=true", flush=True)
        return
    if args.validate_only:
        if not pp.exists():
            raise ValueError("Missing final-test profile")
    else:
        atomic_json(pp, profile, immutable=True)
    states, completed = {}, {}
    for e in profile["samples"]:
        sid = e["sample_id"]
        root = source.sample_path(out, e)
        raw = {}
        decision = None
        if (root / "decision.json").exists():
            decision = load_decision(root / "decision.json", profile, e, selector)
        elif root.exists() and any(root.iterdir()):
            raise ValueError("Final-test outcomes lack their prior question-only decision")
        for b in BRANCHES:
            if (root / (b+".json")).exists():
                raw[b] = read_json(root / (b+".json"))
        if raw:
            validate_raw(raw, profile, e, decision)
        if (root / "scored.json").exists():
            completed[sid] = load_scored(root / "scored.json", profile, e, raw, decision)
        states[sid] = (decision, raw)
    if args.validate_only and len(completed) != len(profile["samples"]):
        raise ValueError("Incomplete final-test")
    runtime = None
    try:
        if len(completed) != len(profile["samples"]):
            from datasets import load_dataset
            from memgen.model.v4_3_question_selector import load_runtime, encode_text, generate
            from memgen.model.v4_3_prefix_equivalence import prefix_bank, split_prefix
            from memgen.model.v4_3_gated_prefix import virtual_prefix, generate_gated, abstained_result
            dataset = load_dataset("openai/gsm8k", "main", revision=profile["dataset"]["revision"], split="test")
            if len(dataset) != len(profile["samples"]):
                raise ValueError("Official test length differs from full frozen manifest")
            questions = checked_dataset_rows(dataset, profile["samples"])
            del dataset
            runtime = load_runtime(profile["reasoner"], args.device)
            runtime.gate = gate
            runtime.controller.close()
            memories, virtual = {}, {}
            descriptors = {r["bank_id"]: r["descriptor"] for r in records}
            for r in records:
                bid = r["bank_id"]
                memories[bid] = prefix_bank(args.equivalence_dir / "prefix_kv", r, runtime,
                    profile["source_equivalence_profile_sha256"], validate_only=True)
                virtual[bid] = virtual_prefix(runtime.model, *memories[bid])
            # Check every possible selection before any final-test generation.
            for e in profile["samples"]:
                q = questions[e["sample_id"]]["question"]
                for bid, (ids, _) in memories.items():
                    n = max(len(split_prefix(runtime, q, descriptors[bid], ids)), len(runtime.visible_prefix(q, None))+len(ids))
                    if n+1024 > runtime.model.config.max_position_embeddings:
                        raise ValueError("Full final-test context budget exceeded; no truncation/exclusion")
            for index, e in enumerate(profile["samples"], 1):
                sid = e["sample_id"]
                if sid in completed:
                    continue
                root = source.sample_path(out, e)
                decision, raw = states[sid]
                q = questions[sid]["question"]
                if decision is None:
                    feature = encode_text(runtime, q)
                    bank = predict(selector, feature)["semantic_bank"]
                    decision = seal({**common(profile, e), "selector_sha256": selector["selector_sha256"],
                                     "feature": feature, "selected_bank": bank})
                    atomic_json(root / "decision.json", decision, immutable=True)
                bank = decision["selected_bank"]
                for b in BRANCHES:
                    if b in raw:
                        continue
                    print(f"[final-test] sample={index}/{len(profile['samples'])} branch={b}", flush=True)
                    if b == "baseline":
                        _, result = generate(runtime, q)
                    elif bank == NO_MEMORY:
                        result = raw["baseline"]["result"] if b == "native_prefix_kv" else abstained_result(raw["baseline"]["result"])
                    elif b == "native_prefix_kv":
                        _, result = generate(runtime, q, descriptors[bank], memories[bank])
                    else:
                        _, result = generate_gated(runtime, q, virtual[bank])
                    raw[b] = seal({**common(profile, e), "selected_bank": bank, "branch": b,
                                   "decision_sha256": decision["record_sha256"], "result": result})
                    validate_raw(raw, profile, e, decision)
                    atomic_json(root / (b+".json"), raw[b], immutable=True)
                # Gold is first consumed here, after all three branches are saved.
                scored = {b: score_branch(runtime.tokenizer, [], 0, raw[b]["result"], questions[sid]["answer"]) for b in BRANCHES}
                case = seal({**common(profile, e), "decision_sha256": decision["record_sha256"], "selected_bank": bank,
                    "memory_token_count": len(memories[bank][0]) if bank != NO_MEMORY else 0,
                    "raw_record_sha256": {b: raw[b]["record_sha256"] for b in BRANCHES}, "branches": scored})
                atomic_json(root / "scored.json", case, immutable=True)
                completed[sid] = case
                print(f"[final-test] completed={len(completed)}/{len(profile['samples'])}", flush=True)
        value = summarize(profile, [completed[e["sample_id"]] for e in profile["samples"]])
        if args.validate_only:
            if read_json(out / "report.json") != value:
                raise ValueError("Final-test report drift")
        else:
            atomic_json(out / "report.json", value)
            brief = {k: v for k, v in value.items() if k not in {"selection_counts", "profile_sha256", "summary_sha256"}}
            atomic_json(out / "brief_summary.json", seal(brief, "summary_sha256"))
        print(f"[final-test] complete summary={out / 'brief_summary.json'}", flush=True)
    finally:
        if runtime is not None:
            runtime.controller.close()


if __name__ == "__main__":
    main()
