#!/usr/bin/env python3
"""Build utility tables, fit question-only selector, and evaluate held-out routing."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import atomic_json, read_json, read_jsonl, implementation_hashes
from memgen.experience.v4_3_bank import authenticate, canonical_hash, file_hash, seal
from memgen.experience.v4_3_question_selector import (
    NO_MEMORY, POLICY, SPLITS, partition_samples, checked_dataset_rows, fit_selector, predict, evaluate,
)
from scripts.audit_v4_3_prefix_equivalence import prepare_experiment
from scripts.audit_v4_3_unified_memory import score_branch

IMPLEMENTATION = ("memgen/experience/v4_3_question_selector.py", "memgen/model/v4_3_question_selector.py",
                  "scripts/run_v4_3_question_selector.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-dir", "side-kv-dir", "semantic-packets", "cache-manifest", "token-risk-artifact",
                 "equivalence-dir", "split-manifest", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--train-size", type=int, default=200)
    p.add_argument("--tune-size", type=int, default=100)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    return p.parse_args()


def prepare(args):
    import numpy as np
    eq = read_json(args.equivalence_dir / "profile.json")
    authenticate(eq, "profile_sha256", "native-prefix experiment")
    original_args = SimpleNamespace(**vars(args), atol=eq["configuration"]["atol"], rtol=eq["configuration"]["rtol"])
    prepared, _, expected_eq = prepare_experiment(original_args)
    if eq != expected_eq:
        raise ValueError("Fixed native-prefix experiment identity differs; do not silently change the consumer")
    # Equivalence trajectory/numerical pass is intentionally NOT an admission criterion.
    records = prepared[3]
    packets = read_jsonl(args.semantic_packets)
    split = read_json(args.split_manifest)
    entries = partition_samples(split, packets, counts=(args.train_size, args.tune_size, args.eval_size), seed=args.seed)
    prefixes = {}
    for r in records:
        m = read_json(args.equivalence_dir / "prefix_kv" / (r["bank_id"] + ".json"))
        authenticate(m, "manifest_sha256", "frozen prefix cache")
        tensor_path = args.equivalence_dir / "prefix_kv" / (r["bank_id"] + ".safetensors")
        if m["profile_sha256"] != eq["profile_sha256"] or m["source_record_sha256"] != r["record_sha256"] or file_hash(tensor_path) != m["tensor_sha256"]:
            raise ValueError("Frozen native prefix cache drift")
        prefixes[r["bank_id"]] = m["manifest_sha256"]
    profile = seal({"schema_version": "v43-question-selector-experiment-v1", "policy": POLICY,
                    "source_equivalence_profile_sha256": eq["profile_sha256"],
                    "reasoner": eq["source_experiment"]["reasoner"],
                    "runtime_versions": eq["configuration"]["runtime_versions"], "device": args.device,
                    "numpy_version": np.__version__,
                    "prefix_manifest_sha256": prefixes, "bank_ids": sorted(prefixes),
                    "split_manifest_sha256": split["manifest_sha256"], "split_file_sha256": file_hash(args.split_manifest),
                    "dataset": split["dataset"], "seed": args.seed, "samples": entries,
                    "maximum_completion_tokens": 1024, "primary_bank_count": 11,
                    "gate_enabled": False, "selector_input": "question_text_only",
                    "feature_policy": POLICY["features"], "implementation_sha256": implementation_hashes(IMPLEMENTATION),
                    "prior_pool_use": "calibration_pool_previously_used_for_other_components",
                    "official_test_used": False, "external_api_calls_made": 0}, "profile_sha256")
    return prepared, eq, profile


def bound_read(path, profile_sha, **expected):
    row = read_json(path)
    authenticate(row, "record_sha256", str(path))
    if row.get("profile_sha256") != profile_sha or any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Artifact identity drift: {path}")
    return row


def sample_path(out, entry):
    sid = entry["sample_id"]
    if Path(sid).name != sid:
        raise ValueError("Unsafe sample ID")
    path = out / "samples" / sid
    if (out / "samples").is_symlink() or path.is_symlink():
        raise ValueError("Refusing symlink sample directory")
    return path


def assemble_row(out, entry, profile):
    root = sample_path(out, entry)
    common = dict(sample_id=entry["sample_id"], question_sha256=entry["question_sha256"])
    feature = bound_read(root / "feature.json", profile["profile_sha256"], **common)
    rewards = {}
    for action in [NO_MEMORY, *profile["bank_ids"]]:
        row = bound_read(root / (action+".json"), profile["profile_sha256"], **common, action=action)
        r = row["result"]
        if r["strict_reward"] not in (0., 1.) or canonical_hash(r["continuation_token_ids"]) != r["continuation_token_ids_sha256"]:
            raise ValueError("Invalid utility-table reward/tokens")
        rewards[action] = r["strict_reward"]
    return {"sample_id": entry["sample_id"], "selector_split": entry["selector_split"],
            "feature": feature["feature"], "rewards": rewards}


def fit_and_save(out, profile, cards):
    if not (out / "selector.json").exists():
        for e in profile["samples"]:
            if e["selector_split"] == "eval" and sample_path(out, e).exists():
                if any(p.name != "feature.json" for p in sample_path(out, e).iterdir()):
                    raise ValueError("Evaluation artifacts exist without the original frozen selector")
    rows = {split: [assemble_row(out, e, profile) for e in profile["samples"] if e["selector_split"] == split]
            for split in ("train", "tune")}
    selector = fit_selector(rows["train"], rows["tune"], profile["bank_ids"], cards, profile["profile_sha256"])
    atomic_json(out / "selector.json", selector, immutable=True)
    return selector


def report(out, profile, selector):
    entries = [e for e in profile["samples"] if e["selector_split"] == "eval"]
    rows = [assemble_row(out, e, profile) for e in entries]
    predictions = {e["sample_id"]: bound_read(sample_path(out, e) / "prediction.json", profile["profile_sha256"],
                      sample_id=e["sample_id"], selector_sha256=selector["selector_sha256"]) for e in entries}
    result = evaluate(selector, rows, predictions)
    return seal({"complete": True, "profile_sha256": profile["profile_sha256"], "selector_sha256": selector["selector_sha256"],
                 "sample_counts": {split: sum(e["selector_split"] == split for e in profile["samples"]) for split in SPLITS},
                 "bank_count": len(profile["bank_ids"]), "native_prefix_kv_frozen": True, "gate_enabled": False,
                 "selected_ridge": selector["ridge"], "selected_threshold": selector["threshold"],
                 "tune_metrics": selector["selected_tune_metrics"], "evaluation": result}, "report_sha256")


def write_report(out, value):
    atomic_json(out / "report.json", value)
    brief = {k: v for k, v in value.items() if k not in {"profile_sha256", "selector_sha256", "report_sha256"}}
    brief["evaluation"] = {k: v for k, v in value["evaluation"].items() if k != "all_bank_accuracy"}
    atomic_json(out / "brief_summary.json", seal(brief, "summary_sha256"))


def main():
    args = parse_args()
    prepared, eq, profile = prepare(args)
    out = args.output_dir
    sources = (args.equivalence_dir, args.bank_dir, args.side_kv_dir, args.cache_manifest.parent)
    if out.is_symlink() or any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents or out.resolve() in p.resolve().parents for p in sources):
        raise ValueError("Selector outputs must be separate from frozen source directories")
    if out.exists() and any(out.iterdir()) and not (args.resume or args.validate_only):
        raise ValueError("Output exists; pass --resume")
    profile_path = out / "profile.json"
    if profile_path.exists() and read_json(profile_path) != profile:
        raise ValueError("Selector experiment identity drift; use a new output directory")
    if args.validate_only:
        if not profile_path.exists():
            raise ValueError("Missing selector profile")
    else:
        atomic_json(profile_path, profile, immutable=True)
    if args.plan_only:
        print(f"[question-selector] samples={len(profile['samples'])} branches={len(profile['samples'])*12} counts={args.train_size}/{args.tune_size}/{args.eval_size}")
        return
    # Authenticate existing work before any generation or training resumes.
    actions = [NO_MEMORY, *profile["bank_ids"]]
    expected_sample_ids = {e["sample_id"] for e in profile["samples"]}
    if (out / "samples").exists():
        if any(p.name not in expected_sample_ids or not p.is_dir() or p.is_symlink() for p in (out / "samples").iterdir()):
            raise ValueError("Unexpected sample artifact outside the selector plan")
    for e in profile["samples"]:
        root = sample_path(out, e)
        if not root.exists():
            continue
        allowed = {a+".json" for a in actions} | {"feature.json"}
        if e["selector_split"] == "eval":
            allowed.add("prediction.json")
            has_outcomes = any((root / (a+".json")).exists() for a in actions)
            if has_outcomes and (not (root / "prediction.json").exists() or not (out / "selector.json").exists()):
                raise ValueError("Evaluation outcomes lack a previously frozen selector/prediction")
        for path in root.iterdir():
            if path.name not in allowed or path.is_symlink() or not path.is_file():
                raise ValueError("Unexpected selector sample artifact")
            bound_read(path, profile["profile_sha256"], sample_id=e["sample_id"], question_sha256=e["question_sha256"])
    card_path = out / "card_features.json"
    selector_path = out / "selector.json"
    if args.validate_only or (out / "report.json").exists():
        cards = bound_read(card_path, profile["profile_sha256"])["features"]
        if args.validate_only:
            rows = {s: [assemble_row(out, e, profile) for e in profile["samples"] if e["selector_split"] == s] for s in ("train", "tune")}
            selector = fit_selector(rows["train"], rows["tune"], profile["bank_ids"], cards, profile["profile_sha256"])
            if read_json(selector_path) != selector:
                raise ValueError("Selector differs from authenticated train/tune rows")
        else:
            selector = fit_and_save(out, profile, cards)
        value = report(out, profile, selector)
        if args.validate_only:
            if read_json(out / "report.json") != value:
                raise ValueError("Evaluation report drift")
        else:
            write_report(out, value)
        print(f"[question-selector] validated complete report={out / 'brief_summary.json'}")
        return
    from datasets import load_dataset
    from memgen.model.v4_3_question_selector import load_runtime, encode_text, generate
    from memgen.model.v4_3_prefix_equivalence import prefix_bank, split_prefix
    dataset = load_dataset("openai/gsm8k", "main", revision=profile["dataset"]["revision"], split="train")
    questions = checked_dataset_rows(dataset, profile["samples"])
    del dataset
    runtime = load_runtime(profile["reasoner"], args.device)
    records = {r["bank_id"]: r for r in prepared[3]}
    try:
        memories = {bid: prefix_bank(args.equivalence_dir / "prefix_kv", r, runtime, eq["profile_sha256"], validate_only=True)
                    for bid, r in records.items()}
        # No silent truncation or outcome-dependent exclusions, including long eval questions.
        for e in profile["samples"]:
            q = questions[e["sample_id"]]["question"]
            for bid, r in records.items():
                ids = split_prefix(runtime, q, r["descriptor"], memories[bid][0])
                if len(ids)+1024 > runtime.model.config.max_position_embeddings:
                    raise ValueError("Question+memory exceeds frozen generation horizon")
        if card_path.exists():
            cards = bound_read(card_path, profile["profile_sha256"])["features"]
        else:
            cards = {bid: encode_text(runtime, r["descriptor"]) for bid, r in records.items()}
            atomic_json(card_path, seal({"profile_sha256": profile["profile_sha256"], "features": cards}), immutable=True)
        selector = None
        for split in SPLITS:
            if split == "eval":
                # This sealed artifact exists BEFORE any evaluation outcome generation.
                selector = fit_and_save(out, profile, cards)
            entries = [e for e in profile["samples"] if e["selector_split"] == split]
            for i, e in enumerate(entries, 1):
                root = sample_path(out, e)
                common = dict(profile_sha256=profile["profile_sha256"], sample_id=e["sample_id"], question_sha256=e["question_sha256"])
                q = questions[e["sample_id"]]["question"]
                fp = root / "feature.json"
                if fp.exists():
                    feature = bound_read(fp, profile["profile_sha256"], sample_id=e["sample_id"], question_sha256=e["question_sha256"])["feature"]
                else:
                    feature = encode_text(runtime, q)
                    atomic_json(fp, seal({**common, "feature": feature}), immutable=True)
                if split == "eval":
                    prediction = seal({**common, "selector_sha256": selector["selector_sha256"], "decision": predict(selector, feature)})
                    atomic_json(root / "prediction.json", prediction, immutable=True)
                for action in [NO_MEMORY, *profile["bank_ids"]]:
                    path = root / (action+".json")
                    if path.exists():
                        bound_read(path, profile["profile_sha256"], sample_id=e["sample_id"], question_sha256=e["question_sha256"], action=action)
                        continue
                    print(f"[question-selector] split={split} sample={i}/{len(entries)} action={action}", flush=True)
                    descriptor = None if action == NO_MEMORY else records[action]["descriptor"]
                    memory = None if action == NO_MEMORY else memories[action]
                    prefix, branch = generate(runtime, q, descriptor, memory)
                    # Gold enters only here, after this independent branch has generated.
                    scored = score_branch(runtime.tokenizer, prefix, len(prefix), branch, questions[e["sample_id"]]["answer"])
                    atomic_json(path, seal({**common, "action": action, "result": scored}), immutable=True)
                row = assemble_row(out, e, profile)
                print(f"[question-selector] split={split} sample={i}/{len(entries)} baseline={row['rewards'][NO_MEMORY]:.0f} oracle={max(row['rewards'].values()):.0f}", flush=True)
        write_report(out, report(out, profile, selector))
        print(f"[question-selector] complete report={out / 'brief_summary.json'}", flush=True)
    finally:
        runtime.controller.close()


if __name__ == "__main__":
    main()
