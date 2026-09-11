#!/usr/bin/env python3
"""Primary-Bank visible text versus reusable native prefix KV equivalence audit."""
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import atomic_json, implementation_hashes, read_json
from memgen.experience.v4_3_bank import authenticate, canonical_hash, file_hash, seal
from scripts.audit_v4_3_unified_memory import prepare, checked_prefix, score_branch

BRANCHES = ("baseline", "visible_text", "native_prefix_kv", "frozen_side_kv")
IMPLEMENTATION = ("memgen/model/v4_3_prefix_equivalence.py", "scripts/audit_v4_3_prefix_equivalence.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-dir", "side-kv-dir", "semantic-packets", "cache-manifest", "token-risk-artifact", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--atol", type=float, default=0.05, help="Fixed absolute tolerance; stored before observing outcomes")
    p.add_argument("--rtol", type=float, default=0.02, help="Fixed relative tolerance; raw errors always reported")
    return p.parse_args()


def prepare_experiment(args):
    import math
    from memgen.model.v4_3_side_kv import runtime_versions
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    legacy = SimpleNamespace(**vars(args))
    legacy.mode, legacy.bank_scope, legacy.all_bank_sweep = "full", "primary", False
    prepared = prepare(legacy)
    bank, evidence, cache, records, loaders, binding, controls, plan, profile = prepared
    if len(records) != 11 or plan["source_sample_count"] != 76:
        raise ValueError("Equivalence audit requires all 11 primary Banks and 76 source samples")
    cases = [c for c in plan["cases"] if c["audit_layer"] == "visible_content"]
    if len(cases) != 76:
        raise ValueError("Incomplete primary prompt coverage")
    config = {"branches": list(BRANCHES), "bank_scope": "primary", "sample_count": 76, "bank_count": 11,
              "maximum_completion_tokens": 1024, "teacher_forced_horizon": "entire_visible_free_trajectory",
              "native_prefix": "all_layers_all_wrapper_tokens_native_positions_no_unloading_no_score_bias",
              "frozen_side_kv": "unchanged_v43_prompt_end", "atol": args.atol, "rtol": args.rtol,
              "dtype": "bfloat16", "attention_backend": "sdpa", "device": args.device,
              "runtime_versions": runtime_versions(), "gold_access": "scoring_after_all_branches",
              "held_out_generalization_claim": False, "external_api_calls_made": 0}
    experiment = seal({"schema_version": "v43-prefix-equivalence-profile-v1", "configuration": config,
                       "source_experiment": profile["experiment"], "cases": cases,
                       "implementation_sha256": implementation_hashes(IMPLEMENTATION)}, "profile_sha256")
    return prepared, cases, experiment


def validate_row(row, case, profile):
    authenticate(row, "record_sha256", "equivalence case")
    if row.get("profile_sha256") != profile["profile_sha256"] or any(row.get(k) != case[k] for k in ("case_id", "case_sha256", "sample_id", "bank_id")):
        raise ValueError("Equivalence case/profile binding drift")
    if set(row["branches"]) != set(BRANCHES):
        raise ValueError("Incomplete equivalence branch coverage")
    d = row["diagnostics"]
    a, b = [row["branches"][k] for k in ("visible_text", "native_prefix_kv")]
    if (d["step_count"] != len(a["continuation_token_ids"]) or d["step_count"] != len(d["steps"])
            or d["free_tokens_equal"] != (a["continuation_token_ids"] == b["continuation_token_ids"])
            or d["free_stop_equal"] != (a["stop_reason"] == b["stop_reason"])
            or d["atol"] != profile["configuration"]["atol"] or d["rtol"] != profile["configuration"]["rtol"]):
        raise ValueError("Inconsistent equivalence diagnostics")
    numerical = d["prefill_cache"]["within_tolerance"] and d["final_forced_cache"]["within_tolerance"] and all(s["within_tolerance"] for s in d["steps"])
    behavioral = d["free_tokens_equal"] and d["free_stop_equal"] and all(s["top1_equal"] for s in d["steps"])
    if d["numerical_pass"] != numerical or d["behavioral_pass"] != behavioral:
        raise ValueError("Equivalence pass flag disagrees with measurements")
    for branch in row["branches"].values():
        ids = branch["continuation_token_ids"]
        if not 0 < len(ids) <= 1024 or canonical_hash(ids) != branch["continuation_token_ids_sha256"]:
            raise ValueError("Continuation token identity/budget mismatch")


def summarize(rows, profile):
    def group(selected):
        return {name: {"correct": sum(r["branches"][name]["strict_reward"] for r in selected),
                       "count": len(selected),
                       "accuracy": sum(r["branches"][name]["strict_reward"] for r in selected)/len(selected) if selected else None,
                       "gain": sum(r["branches"][name]["strict_reward"] > r["branches"]["baseline"]["strict_reward"] for r in selected),
                       "harm": sum(r["branches"][name]["strict_reward"] < r["branches"]["baseline"]["strict_reward"] for r in selected)} for name in BRANCHES}
    complete = len(rows) == len(profile["cases"])
    passed = complete and all(r["diagnostics"]["numerical_pass"] and r["diagnostics"]["behavioral_pass"] for r in rows)
    return seal({"schema_version": "v43-prefix-equivalence-summary-v1", "profile_sha256": profile["profile_sha256"],
                 "complete": complete, "equivalence_passed": passed,
                 "status": ("equivalent_within_declared_tolerance" if passed else "completed_with_mismatches") if complete else "in_progress",
                 "expected_case_count": len(profile["cases"]), "completed_case_count": len(rows),
                 "configuration": profile["configuration"], "overall": group(rows),
                 "by_bank": {bid: group([r for r in rows if r["bank_id"] == bid]) for bid in sorted({r["bank_id"] for r in rows})},
                 "numerical_failure_count": sum(not r["diagnostics"]["numerical_pass"] for r in rows),
                 "free_trajectory_mismatch_count": sum(not r["diagnostics"]["free_tokens_equal"] for r in rows),
                 "forced_top1_mismatch_count": sum(r["diagnostics"]["top1_mismatch_count"] for r in rows),
                 "max_logits_abs": max((r["diagnostics"]["max_logits_abs"] for r in rows), default=None),
                 "max_kl": max((r["diagnostics"]["max_kl"] for r in rows), default=None),
                 "reward_mismatch_count": sum(r["branches"]["visible_text"]["strict_reward"] != r["branches"]["native_prefix_kv"]["strict_reward"] for r in rows),
                 "results": {r["case_id"]: r["record_sha256"] for r in rows}}, "summary_sha256")


def write_summary(out, rows, profile):
    summary = summarize(rows, profile)
    atomic_json(out / "core_summary.json", summary)
    brief = {k: v for k, v in summary.items() if k not in {"configuration", "by_bank", "results", "summary_sha256"}}
    brief["tolerance"] = {k: profile["configuration"][k] for k in ("atol", "rtol")}
    atomic_json(out / "brief_summary.json", seal(brief, "summary_sha256"))
    return summary


def main():
    args = parse_args()
    prepared, cases, profile = prepare_experiment(args)
    bank, evidence, cache, records, loaders = prepared[:5]
    out = args.output_dir
    source_dirs = (args.bank_dir, args.side_kv_dir, args.cache_manifest.parent)
    if any(out.resolve() == p.resolve() or out.resolve() in p.resolve().parents or p.resolve() in out.resolve().parents for p in source_dirs) or out.is_symlink():
        raise ValueError("Equivalence output must be separate from source artifacts")
    if out.exists() and any(out.iterdir()) and not (args.resume or args.validate_only):
        raise ValueError("Output exists; pass --resume")
    profile_path = out / "profile.json"
    if profile_path.exists() and read_json(profile_path) != profile:
        raise ValueError("Experiment identity drift; use a new output directory")
    if args.validate_only:
        if not profile_path.is_file():
            raise ValueError("Missing experiment profile")
    else:
        atomic_json(profile_path, profile, immutable=True)
    by_case = {c["case_id"]: c for c in cases}
    rows = {}
    directory = out / "cases"
    if directory.exists():
        for path in directory.iterdir():
            if path.is_symlink() or path.suffix != ".json" or path.stem not in by_case:
                raise ValueError("Unexpected equivalence result artifact")
            row = read_json(path)
            validate_row(row, by_case[path.stem], profile)
            rows[path.stem] = row
    # Cache hashes referenced by completed cases remain mandatory on resume.
    for row in rows.values():
        path = out / "prefix_kv" / (row["bank_id"] + ".json")
        m = read_json(path)
        authenticate(m, "manifest_sha256", "prefix KV")
        if (m["manifest_sha256"] != row["prefix_manifest_sha256"] or m["profile_sha256"] != profile["profile_sha256"]
                or file_hash(path.with_suffix(".safetensors")) != m["tensor_sha256"]):
            raise ValueError("Completed result prefix KV drift")
    ordered = lambda: [rows[c["case_id"]] for c in cases if c["case_id"] in rows]
    if args.plan_only:
        print(f"[prefix-equivalence] planned={len(cases)} banks={len(records)} reused={len(rows)}")
        return
    summary_path = out / "core_summary.json"
    if args.validate_only or len(rows) == len(cases):
        if len(rows) != len(cases):
            raise ValueError("Incomplete equivalence audit")
        summary = summarize(ordered(), profile)
        if args.validate_only:
            if read_json(summary_path) != summary:
                raise ValueError("Summary differs from authenticated results")
        else:
            write_summary(out, ordered(), profile)
        print(f"[prefix-equivalence] complete equivalence_passed={summary['equivalence_passed']} summary={summary_path}")
        if not summary["equivalence_passed"]:
            raise SystemExit(2)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from memgen.chat_templates import CONVERSATION_TEMPLATE
    from memgen.experience.v4_3_reasoner import validate_tokenizer_replay
    from memgen.model.side_kv import SideKVAttentionController
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v4_3_runtime import V43UnifiedRuntime
    from memgen.model.v4_3_side_kv import MEMORY_SCORE_BIAS
    from memgen.model.v4_3_prefix_equivalence import prefix_bank, run_equivalence, split_prefix, memory_prefix_ids
    source = profile["source_experiment"]
    reasoner = source["reasoner"]
    tokenizer = AutoTokenizer.from_pretrained(reasoner["model_name"], revision=reasoner["tokenizer_revision"])
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if source["source_reasoner"]["tokenizer_revision"] == "main":
        replay = validate_tokenizer_replay(tokenizer=tokenizer, cache=cache, evidence=evidence,
                    source_reasoner=source["source_reasoner"], packets_sha256=file_hash(args.semantic_packets))
        if replay != source["tokenizer_replay_validation"]:
            raise ValueError("Tokenizer source replay differs")
    risk = torch.load(args.token_risk_artifact, map_location="cpu", weights_only=False)
    if any(risk["reasoner"].get(k) != v for k, v in source["source_reasoner"].items()):
        raise ValueError("Frozen gate reasoner mismatch")
    model = AutoModelForCausalLM.from_pretrained(reasoner["model_name"], revision=reasoner["model_revision"],
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    if ((getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]) != reasoner["model_revision"]
            or (tokenizer.init_kwargs.get("_commit_hash") or reasoner["tokenizer_revision"]) != reasoner["tokenizer_revision"]):
        raise ValueError("Loaded model/tokenizer revision mismatch")
    controller = SideKVAttentionController(model=model, layer_number=24, audit_canonical_rope=True,
                            memory_score_normalization="log_valid_slots", memory_score_bias=MEMORY_SCORE_BIAS)
    runtime = V43UnifiedRuntime(model=model, tokenizer=tokenizer, device=args.device,
                               gate=EntropyHysteresisGate.from_token_artifact(risk), controller=controller)
    by_bank = {r["bank_id"]: r for r in records}
    try:
        native = {c["case_id"]: checked_prefix(c, evidence[c["experience_id"]], tokenizer) for c in cases}
        # Prove token boundaries and full horizon fit for ALL samples before generation.
        for c in cases:
            record = by_bank[c["bank_id"]]
            full = split_prefix(runtime, evidence[c["experience_id"]]["question"], record["descriptor"], memory_prefix_ids(tokenizer, record["descriptor"]))
            if len(full)+1024 > model.config.max_position_embeddings:
                raise ValueError("Memory plus question exceeds full completion horizon")
        memories = {}
        for i, r in enumerate(records, 1):
            print(f"[prefix-equivalence] cache bank={i}/{len(records)}", flush=True)
            memories[r["bank_id"]] = prefix_bank(out / "prefix_kv", r, runtime, profile["profile_sha256"])
        for i, c in enumerate(cases, 1):
            if c["case_id"] in rows:
                continue
            e, record = evidence[c["experience_id"]], by_bank[c["bank_id"]]
            print(f"[prefix-equivalence] case={i}/{len(cases)} bank={c['bank_id']}", flush=True)
            m_ids, tensors = memories[c["bank_id"]]
            full, text, kv, diagnostics = run_equivalence(runtime, e["question"], record["descriptor"], m_ids, tensors, atol=args.atol, rtol=args.rtol)
            memory = loaders[c["bank_id"]].get_memory(c["bank_id"], device=args.device, dtype=torch.bfloat16)
            side, _ = runtime.run_latent(prefix=native[c["case_id"]], prompt_count=len(native[c["case_id"]]),
                                        memories={"baseline": None, "matched": memory})
            branches = {"baseline": side["baseline"], "visible_text": text, "native_prefix_kv": kv, "frozen_side_kv": side["matched"]}
            for name, branch in branches.items():
                branch["memory_representation"] = name
                branch["condition_memory_id"] = None if name == "baseline" else c["bank_id"]
            scored = {name: score_branch(tokenizer, full if name in {"visible_text", "native_prefix_kv"} else native[c["case_id"]],
                            len(full) if name in {"visible_text", "native_prefix_kv"} else len(native[c["case_id"]]), b, e["official_solution"])
                      for name, b in branches.items()}
            manifest = read_json(out / "prefix_kv" / (c["bank_id"] + ".json"))
            row = seal({"profile_sha256": profile["profile_sha256"], **{k: c[k] for k in ("case_id", "case_sha256", "sample_id", "bank_id")},
                        "prefix_manifest_sha256": manifest["manifest_sha256"], "memory_prefix_token_count": len(m_ids),
                        "branches": scored, "diagnostics": diagnostics})
            validate_row(row, c, profile)
            atomic_json(directory / (c["case_id"] + ".json"), row, immutable=True)
            rows[c["case_id"]] = row
            write_summary(out, ordered(), profile)
            print(f"[prefix-equivalence] numerical={diagnostics['numerical_pass']} tokens_equal={diagnostics['free_tokens_equal']} "
                  + " ".join(f"{k}={v['strict_reward']:.0f}" for k, v in scored.items()), flush=True)
    finally:
        controller.close()
    summary = summarize(ordered(), profile)
    print(f"[prefix-equivalence] complete equivalence_passed={summary['equivalence_passed']} summary={summary_path}")
    if not summary["equivalence_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
