#!/usr/bin/env python3
"""Score memory applicability locally, freeze train thresholds, reuse reasoner outcomes."""
import argparse
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import atomic_json, implementation_hashes, read_json
from memgen.experience.v4_3_bank import authenticate, canonical_hash, seal, text_hash
from memgen.experience.v4_3_local_rerank import POLICY, POOLS, candidates, checked_scores, decisions, evaluate, fit
from scripts import run_v4_3_retrieval_coverage as source

IMPLEMENTATION = ("memgen/experience/v4_3_local_rerank.py", "memgen/model/v4_3_local_rerank.py",
                  "scripts/run_v4_3_local_rerank.py")


def parse_args():
    from memgen.model.v4_3_local_rerank import DEFAULT_MODEL, DEFAULT_REVISION
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("selector-dir", "study-dir", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--reranker-model", default=DEFAULT_MODEL)
    p.add_argument("--reranker-revision", default=DEFAULT_REVISION)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-length", type=int, default=8192)
    for flag in ("plan-only", "validate-only"):
        p.add_argument("--"+flag, action="store_true")
    return p.parse_args()


def load_questions(sp, entries):
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", revision=sp["dataset"]["revision"], split="train")
    questions = {}
    for e in entries:
        question = dataset[e["source_index"]]["question"].strip()
        if text_hash(question) != e["question_sha256"]:
            raise ValueError("Reranker dataset question content drift")
        questions[e["sample_id"]] = question
    return questions


def prepare(args):
    from memgen.model.v4_3_local_rerank import model_identity
    sp, old, previous, study, report = source.prepare(args)
    features = {}
    for e in previous["samples"]:
        fp = source.study_source.source.sample_path(args.selector_dir, e) / "feature.json"
        row = source.study_source.source.bound_read(fp, sp["profile_sha256"], sample_id=e["sample_id"], question_sha256=e["question_sha256"])
        features[e["sample_id"]] = row["feature"]
    pools = candidates(previous["samples"], features, sp["bank_ids"], study["key_features"]["problem_structure"])
    for e in previous["samples"]:
        prior = report[e["selector_split"]]["predictions"]["problem_structure/always"][e["sample_id"]]
        if pools[e["sample_id"]]["semantic_top3"][0] != prior:
            raise ValueError("Candidate top1 differs from the frozen similarity study")
    if args.max_length <= 0:
        raise ValueError("Positive reranker context limit required")
    identity = model_identity(args.reranker_model, args.reranker_revision)
    profile = seal({"policy": POLICY, "source_study_profile_sha256": previous["profile_sha256"],
        "source_selector_profile_sha256": sp["profile_sha256"], "source_study_report_sha256": report["report_sha256"],
        "bank_ids": sp["bank_ids"], "samples": previous["samples"], "candidate_pools": pools,
        "descriptors": previous["key_texts"]["full_card"], "reranker": identity,
        "device": args.device, "max_length": args.max_length, "dtype": "bfloat16", "attention_backend": "sdpa",
        "runtime_versions": {p: version(p) for p in ("torch", "transformers", "safetensors", "numpy")},
        "implementation_sha256": implementation_hashes(IMPLEMENTATION),
        "evaluation_role": previous["evaluation_role"], "new_reasoner_generations": 0,
        "external_inference_api_calls": 0, "official_test_used": False, "native_prefix_kv_frozen": True,
        "random_control": "one_fixed_seed_43_not_expectation_over_all_random_subsets",
        "score_cost": "all_banks_scored_once_for_three_pool_controls_pair_batch_size_one"}, "profile_sha256")
    return sp, old, study, report, profile


def frozen_predictions(rows, scores, pools, selector):
    result = {}
    for pool in POOLS:
        for suffix, threshold in (("forced", -1.), ("calibrated", selector["rules"][pool]["threshold"])):
            ds = decisions(rows, scores, pools, pool, threshold)
            result[pool+"/"+suffix] = dict(zip([r["sample_id"] for r in rows], ds))
    return result


def main():
    args = parse_args()
    out = args.output_dir
    if out.is_symlink() or any(out.resolve() == p.resolve() or out.resolve() in p.resolve().parents
                              or p.resolve() in out.resolve().parents for p in (args.selector_dir, args.study_dir)):
        raise ValueError("Local reranker output must be separate from sources")
    print("[local-rerank] authenticating sources and model identity", flush=True)
    sp, old, study, source_report, profile = prepare(args)
    if args.plan_only:
        print(f"[local-rerank] questions={len(profile['samples'])} pairs={len(profile['samples'])*len(profile['bank_ids'])} "
              f"model={profile['reranker']['source']} new_reasoner_generations=0")
        return
    source.study_source.save_or_check(out/"profile.json", profile, args.validate_only)
    qpath = out/"questions.json"
    if qpath.exists():
        cached = source.study_source.source.bound_read(qpath, profile["profile_sha256"])
        questions = cached["questions"]
    elif args.validate_only:
        raise ValueError("Missing reranker question cache")
    else:
        questions = load_questions(sp, profile["samples"])
        atomic_json(qpath, seal({"profile_sha256": profile["profile_sha256"], "questions": questions}), immutable=True)
    if set(questions) != {e["sample_id"] for e in profile["samples"]} or any(text_hash(questions[e["sample_id"]]) != e["question_sha256"] for e in profile["samples"]):
        raise ValueError("Reranker question cache drift")
    from memgen.model.v4_3_local_rerank import LocalReranker
    runtime = None
    all_scores, reports, selector = {}, {}, None
    try:
        for split in ("train", "tune"):
            entries = [e for e in profile["samples"] if e["selector_split"] == split]
            for i, e in enumerate(entries, 1):
                sid = e["sample_id"]
                all_scores[sid] = {}
                for bid in profile["bank_ids"]:
                    folder = source.study_source.source.sample_path(out, e)
                    path = folder/(bid+".json")
                    binding = {"sample_id": sid, "question_sha256": e["question_sha256"], "bank_id": bid,
                               "descriptor_sha256": text_hash(profile["descriptors"][bid])}
                    if path.exists():
                        row = source.study_source.source.bound_read(path, profile["profile_sha256"], **binding)
                        score = row["result"]
                    elif args.validate_only:
                        raise ValueError(f"Missing reranker pair score: {path}")
                    else:
                        if runtime is None:
                            print("[local-rerank] loading frozen local reranker", flush=True)
                            runtime = LocalReranker(profile["reranker"], args.device, args.max_length)
                        score = runtime.score(questions[sid], profile["descriptors"][bid])
                        atomic_json(path, seal({"profile_sha256": profile["profile_sha256"], **binding, "result": score}), immutable=True)
                    all_scores[sid][bid] = score
                checked_scores({"scores": all_scores[sid]}, profile["bank_ids"])
                print(f"[local-rerank] split={split} sample={i}/{len(entries)}", flush=True)
            if split == "tune":
                # Predictions have no access to these questions' rewards or reasoner completions.
                predictions = frozen_predictions(entries, all_scores, profile["candidate_pools"], selector)
                source.study_source.save_or_check(out/"tune_predictions.json", seal({"profile_sha256": profile["profile_sha256"],
                    "selector_sha256": selector["selector_sha256"], "predictions": predictions}), args.validate_only)
            rows = source.study_source.read_rows(args, sp, old, split)
            expected_hash = study["train_data_sha256"] if split == "train" else source_report["tune_data_sha256"]
            if canonical_hash(rows) != expected_hash:
                raise ValueError("Frozen native-prefix utility table drift")
            if split == "train":
                selector = fit(rows, profile["bank_ids"], all_scores, profile["candidate_pools"], profile["profile_sha256"])
                source.study_source.save_or_check(out/"selector.json", selector, args.validate_only)
            reports[split] = evaluate(selector, rows, all_scores, profile["candidate_pools"], split,
                                     source_report[split]["predictions"]["legacy_semantic"], old["fixed_bank_from_train"])
            if split == "tune" and any(reports[split]["predictions"][name] != ds for name, ds in predictions.items()):
                raise ValueError("Evaluation choices differ from pre-outcome predictions")
    finally:
        del runtime
    report = seal({"complete": True, "profile_sha256": profile["profile_sha256"], "selector_sha256": selector["selector_sha256"],
        "bank_count": len(profile["bank_ids"]), "sample_counts": {s: reports[s]["methods"]["no_memory"]["count"] for s in reports},
        "reranker": profile["reranker"], "evaluation_role": profile["evaluation_role"],
        "native_prefix_kv_frozen": True, "official_test_used": False, "new_reasoner_generations": 0,
        "external_inference_api_calls": 0, "scores_are_calibrated_reasoner_success_probabilities": False,
        "thresholds_selected_on_train": {p: r["threshold"] for p, r in selector["rules"].items()}, **reports}, "report_sha256")
    brief = {k: v for k, v in report.items() if k not in {"train", "tune", "profile_sha256", "selector_sha256", "report_sha256"}}
    brief["reranker"] = profile["reranker"]["source"]
    brief["tune_methods"] = {}
    for name, m in reports["tune"]["methods"].items():
        brief["tune_methods"][name] = {"accuracy": m["accuracy"], "gain": m["gain"], "harm": m["harm"],
            "memory_use_count": m["memory_use_count"], "mean_reasoner_tokens": m["generated_tokens"]["mean"],
            "at_limit": m["generated_tokens"]["at_1024_token_limit_count"]}
        if "reranker_cost" in m:
            brief["tune_methods"][name]["reranker_input_tokens"] = m["reranker_cost"]["input_tokens"]
    source.study_source.save_or_check(out/"report.json", report, args.validate_only)
    source.study_source.save_or_check(out/"brief_summary.json", seal(brief, "summary_sha256"), args.validate_only)
    print(f"[local-rerank] complete summary={out/'brief_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
