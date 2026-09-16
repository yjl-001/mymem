#!/usr/bin/env python3
"""Reuse train/tune utility tables to study similarity keys and abstention."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import atomic_json, implementation_hashes, read_json
from memgen.experience.v4_3_bank import authenticate, canonical_hash, seal, text_hash
from memgen.experience.v4_3_question_selector import NO_MEMORY, POLICY as SOURCE_POLICY
from memgen.experience.v4_3_similarity_study import (
    KEYS, POLICY, diagnostics, evaluate, fit_study, key_texts, score_features,
)
from scripts import run_v4_3_question_selector as source

IMPLEMENTATION = ("memgen/experience/v4_3_similarity_study.py", "scripts/run_v4_3_similarity_study.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-dir", "side-kv-dir", "semantic-packets", "cache-manifest", "token-risk-artifact",
                 "equivalence-dir", "split-manifest", "selector-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    for flag in ("resume", "plan-only", "validate-only"):
        p.add_argument("--" + flag, action="store_true")
    return p.parse_args()


def prepare(args):
    sp = read_json(args.selector_dir / "profile.json")
    authenticate(sp, "profile_sha256", "frozen selector source")
    counts = Counter(e["selector_split"] for e in sp["samples"])
    old_args = SimpleNamespace(**vars(args), train_size=counts["train"], tune_size=counts["tune"],
                              eval_size=counts["eval"], seed=sp["seed"])
    prepared, _, expected = source.prepare(old_args)
    if sp != expected:
        raise ValueError("Frozen selector source identity drift")
    old = read_json(args.selector_dir / "selector.json")
    authenticate(old, "selector_sha256", "original selector")
    if old["profile_sha256"] != sp["profile_sha256"] or old["policy"] != SOURCE_POLICY or old["bank_ids"] != sp["bank_ids"]:
        raise ValueError("Original selector binding drift")
    for split in ("train", "tune"):
        if old[split + "_sample_ids"] != [e["sample_id"] for e in sp["samples"] if e["selector_split"] == split]:
            raise ValueError("Original selector split membership drift")
    cards = source.bound_read(args.selector_dir / "card_features.json", sp["profile_sha256"])["features"]
    if cards != old["card_features"]:
        raise ValueError("Original card feature drift")
    texts = key_texts(prepared[3])
    if any(set(v) != set(sp["bank_ids"]) for v in texts.values()):
        raise ValueError("Key/Bank coverage mismatch")
    profile = seal({
        "schema_version": POLICY["version"], "policy": POLICY,
        "source_profile_sha256": sp["profile_sha256"], "legacy_selector_sha256": old["selector_sha256"],
        "bank_ids": sp["bank_ids"], "key_texts": texts, "reasoner": sp["reasoner"],
        "runtime_versions": sp["runtime_versions"], "numpy_version": sp["numpy_version"], "device": args.device,
        "samples": [e for e in sp["samples"] if e["selector_split"] in ("train", "tune")],
        "source_record_sha256": {r["bank_id"]: r["record_sha256"] for r in prepared[3]},
        "source_prefix_manifest_sha256": sp["prefix_manifest_sha256"],
        "implementation_sha256": implementation_hashes(IMPLEMENTATION),
        "native_prefix_kv_frozen": True, "gate_enabled": False, "external_api_calls_made": 0,
        "new_answer_generations": 0, "official_test_used": False, "selector_eval_outcomes_used": False,
        "evaluation_role": "reused_calibration_train_tune_not_pristine_test",
        "legacy_semantic_threshold_selected_on": "source_tune_reused_here_as_reference_only",
        "generated_token_policy": "completion_token_ids_including_emitted_eos_excluding_question_and_memory",
    }, "profile_sha256")
    return sp, old, cards, profile


def read_rows(args, sp, old, split):
    """Intentionally never opens eval/final-test sample outcomes."""
    if split not in ("train", "tune"):
        raise ValueError("Study may read only train/tune outcomes")
    base_rows, rows = [], []
    for e in sp["samples"]:
        if e["selector_split"] != split:
            continue
        root = source.sample_path(args.selector_dir, e)
        common = {"sample_id": e["sample_id"], "question_sha256": e["question_sha256"]}
        feature = source.bound_read(root / "feature.json", sp["profile_sha256"], **common)["feature"]
        rewards, counts = {}, {}
        for action in [NO_MEMORY, *sp["bank_ids"]]:
            record = source.bound_read(root / (action + ".json"), sp["profile_sha256"], action=action, **common)
            result = record["result"]
            tokens = result["continuation_token_ids"]
            if (not isinstance(tokens, list) or not 0 < len(tokens) <= 1024
                    or any(type(t) is not int or t < 0 for t in tokens)
                    or canonical_hash(tokens) != result["continuation_token_ids_sha256"]
                    or type(result["strict_reward"]) not in (float, int) or result["strict_reward"] not in (0., 1.)):
                raise ValueError("Invalid source utility-table reward/tokens")
            rewards[action], counts[action] = result["strict_reward"], len(tokens)
        base = {"sample_id": e["sample_id"], "selector_split": split, "feature": feature, "rewards": rewards}
        base_rows.append(base)
        rows.append({**base, "token_counts": counts})
    if canonical_hash(base_rows) != old[split + "_data_sha256"]:
        raise ValueError("Source rows differ from original fitted selector")
    return rows


def save_or_check(path, value, validate_only):
    if validate_only:
        if read_json(path) != value:
            raise ValueError(f"Study artifact drift: {path}")
    else:
        atomic_json(path, value, immutable=True)


def get_vectors(args, profile, cards):
    from memgen.model.v4_3_question_selector import load_runtime, encode_text
    vectors = {"full_card": cards}
    runtime = None
    try:
        for key in KEYS[1:]:
            vectors[key] = {}
            for bid in profile["bank_ids"]:
                path = args.output_dir / "key_features" / key / (bid + ".json")
                binding = {"profile_sha256": profile["profile_sha256"], "key": key, "bank_id": bid,
                           "text_sha256": text_hash(profile["key_texts"][key][bid])}
                if path.exists():
                    row = source.bound_read(path, profile["profile_sha256"], key=key, bank_id=bid, text_sha256=binding["text_sha256"])
                    vector = row["feature"]
                elif args.validate_only:
                    raise ValueError(f"Missing key feature: {path}")
                else:
                    if runtime is None:
                        runtime = load_runtime(profile["reasoner"], args.device)
                    print(f"[similarity-study] encoding key={key} bank={len(vectors[key])+1}/{len(cards)}", flush=True)
                    vector = encode_text(runtime, profile["key_texts"][key][bid])
                    atomic_json(path, seal({**binding, "feature": vector}), immutable=True)
                vectors[key][bid] = vector
        for features in vectors.values():
            score_features([next(iter(cards.values()))], profile["bank_ids"], features)
        return vectors
    finally:
        if runtime is not None:
            runtime.controller.close()


def brief_summary(report, study):
    def compact(m):
        return {"accuracy": m["accuracy"], "gain": m["gain"], "harm": m["harm"],
                "memory_use_count": m["memory_use_count"], "mean_tokens": m["generated_tokens"]["mean"],
                "token_delta_pct": 100 * m["generated_tokens"]["relative_delta"],
                "at_limit": m["generated_tokens"]["at_1024_token_limit_count"]}
    names = ["no_memory", "fixed_from_train", "legacy_semantic", *study["candidates"]]
    methods = {}
    for name in names:
        methods[name] = compact(report["tune"]["methods"][name])
        if name in study["candidates"]:
            fitted = study["candidates"][name]
            methods[name].update(threshold=fitted["threshold"], margin=fitted["margin"],
                                 train_oof_accuracy=fitted["train_oof"]["accuracy"])
    return seal({"complete": True, "sample_counts": report["sample_counts"], "bank_count": len(study["bank_ids"]),
                 "recommended_selected_on_train_only": study["recommended"],
                 "official_test_used": False, "native_prefix_kv_frozen": True, "new_answer_generations": 0,
                 "evaluation_role": report["evaluation_role"], "tune_methods": methods}, "summary_sha256")


def main():
    args = parse_args()
    sp, old, cards, profile = prepare(args)
    out = args.output_dir
    sources = (args.selector_dir, args.bank_dir, args.side_kv_dir, args.equivalence_dir,
               args.cache_manifest.parent, args.semantic_packets.parent, args.split_manifest.parent)
    if out.is_symlink() or any(out.resolve() == p.resolve() or out.resolve() in p.resolve().parents or p.resolve() in out.resolve().parents for p in sources):
        raise ValueError("Study output must be separate from frozen sources")
    if args.plan_only:
        print(f"[similarity-study] samples={dict(Counter(e['selector_split'] for e in profile['samples']))} "
              f"new_key_encodings={2*len(cards)} new_answer_generations=0 parameter_selection=train_only")
        return
    if out.exists() and any(out.iterdir()) and not (args.resume or args.validate_only):
        raise ValueError("Output exists; pass --resume")
    save_or_check(out / "profile.json", profile, args.validate_only)
    train = read_rows(args, sp, old, "train")
    vectors = get_vectors(args, profile, cards)
    print("[similarity-study] train-only five-fold key/rule selection", flush=True)
    study = fit_study(train, profile["bank_ids"], vectors, profile["profile_sha256"])
    # Materialize the final policy before even opening a tune reward file.
    save_or_check(out / "selector.json", study, args.validate_only)
    print(f"[similarity-study] frozen recommendation={study['recommended']}; evaluating reused tune", flush=True)
    tune = read_rows(args, sp, old, "tune")
    splits = {"train": train, "tune": tune}
    report = seal({
        "complete": True, "profile_sha256": profile["profile_sha256"], "selector_sha256": study["selector_sha256"],
        "evaluation_role": profile["evaluation_role"], "sample_counts": {s: len(rs) for s, rs in splits.items()},
        "tune_data_sha256": canonical_hash(tune), "official_test_used": False, "new_answer_generations": 0,
        "generated_token_policy": profile["generated_token_policy"],
        "cv_note": "OOF used to select family; selected-family OOF is not an unbiased final estimate",
        "legacy_note": "Legacy semantic threshold used this tune split historically; reference is not fresh validation",
        **{s: evaluate(study, rs, s, old["semantic_threshold"], old["fixed_bank_from_train"]) for s, rs in splits.items()},
        "diagnostics": {s: diagnostics(study, train, rs) for s, rs in splits.items()},
    }, "report_sha256")
    save_or_check(out / "report.json", report, args.validate_only)
    save_or_check(out / "brief_summary.json", brief_summary(report, study), args.validate_only)
    print(f"[similarity-study] complete summary={out / 'brief_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
