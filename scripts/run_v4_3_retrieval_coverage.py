#!/usr/bin/env python3
"""Find where effective Banks occur in existing similarity rankings, without generation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import implementation_hashes, read_json
from memgen.experience.v4_3_bank import authenticate, canonical_hash, seal, text_hash
from memgen.experience.v4_3_retrieval_coverage import POLICY, compact, coverage
from memgen.experience.v4_3_similarity_study import KEYS, POLICY as STUDY_POLICY
from scripts import run_v4_3_similarity_study as study_source

IMPLEMENTATION = ("memgen/experience/v4_3_retrieval_coverage.py", "scripts/run_v4_3_retrieval_coverage.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selector-dir", "study-dir", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def authenticated(path, field):
    value = read_json(path)
    authenticate(value, field, str(path))
    return value


def prepare(args):
    sp = authenticated(args.selector_dir / "profile.json", "profile_sha256")
    old = authenticated(args.selector_dir / "selector.json", "selector_sha256")
    profile = authenticated(args.study_dir / "profile.json", "profile_sha256")
    study = authenticated(args.study_dir / "selector.json", "selector_sha256")
    report = authenticated(args.study_dir / "report.json", "report_sha256")
    for p in (sp, profile):
        if p["implementation_sha256"] != implementation_hashes(tuple(p["implementation_sha256"])):
            raise ValueError("Source implementation drift")
    if (old["profile_sha256"] != sp["profile_sha256"]
            or profile["source_profile_sha256"] != sp["profile_sha256"]
            or profile["legacy_selector_sha256"] != old["selector_sha256"]
            or study["profile_sha256"] != profile["profile_sha256"]
            or study["policy"] != STUDY_POLICY or profile["policy"] != STUDY_POLICY
            or report["profile_sha256"] != profile["profile_sha256"]
            or report["selector_sha256"] != study["selector_sha256"] or not report["complete"]):
        raise ValueError("Similarity/source artifact binding mismatch")
    ids = sp["bank_ids"]
    if any(p["bank_ids"] != ids for p in (old, profile, study)):
        raise ValueError("Bank namespace drift")
    entries = [e for e in sp["samples"] if e["selector_split"] in ("train", "tune")]
    if entries != profile["samples"] or any(e.get("dataset_split") != "train" or e.get("logical_split") != "calibration-val" for e in entries):
        raise ValueError("Diagnostic split mismatch or official-test sample")
    if len({e["sample_id"] for e in entries}) != len(entries) or len({e["question_sha256"] for e in entries}) != len(entries):
        raise ValueError("Duplicate diagnostic samples/questions")
    cards = study_source.source.bound_read(args.selector_dir / "card_features.json", sp["profile_sha256"])["features"]
    if cards != old["card_features"] or cards != study["key_features"]["full_card"]:
        raise ValueError("Full-card features drift")
    for key in KEYS[1:]:
        for bid in ids:
            row = study_source.source.bound_read(args.study_dir / "key_features" / key / (bid+".json"),
                profile["profile_sha256"], key=key, bank_id=bid, text_sha256=text_hash(profile["key_texts"][key][bid]))
            if row["feature"] != study["key_features"][key][bid]:
                raise ValueError("Frozen key vector drift")
    return sp, old, profile, study, report


def main():
    args = parse_args()
    out = args.output_dir
    if out.is_symlink() or any(out.resolve() == p.resolve() or out.resolve() in p.resolve().parents
                              or p.resolve() in out.resolve().parents for p in (args.selector_dir, args.study_dir)):
        raise ValueError("Coverage output must be separate from sources")
    sp, old, profile, study, source_report = prepare(args)
    rows = {s: study_source.read_rows(args, sp, old, s) for s in ("train", "tune")}
    if (canonical_hash(rows["train"]) != study["train_data_sha256"]
            or canonical_hash(rows["tune"]) != source_report["tune_data_sha256"]):
        raise ValueError("Source utility data drift")
    results = {}
    for split, data in rows.items():
        print(f"[retrieval-coverage] split={split} samples={len(data)} banks={len(sp['bank_ids'])}", flush=True)
        result = coverage(data, sp["bank_ids"], study["key_features"], split)
        # Rankings must reproduce the exact always-top1 controls already reported.
        for key in KEYS:
            if result["keys"][key]["top1"] != source_report[split]["methods"][key+"/always"]:
                raise ValueError("Top1 differs from frozen similarity study")
        results[split] = result
    value = seal({"complete": True, "policy": POLICY, "source_profile_sha256": profile["profile_sha256"],
                  "source_selector_sha256": study["selector_sha256"], "source_report_sha256": source_report["report_sha256"],
                  "implementation_sha256": implementation_hashes(IMPLEMENTATION), "bank_count": len(sp["bank_ids"]),
                  "evaluation_role": profile["evaluation_role"], "native_prefix_kv_frozen": True,
                  "oracle_is_deployable": False, "new_answer_generations": 0, "external_api_calls_made": 0,
                  "official_test_used": False, **results}, "report_sha256")
    brief = seal({k: v for k, v in value.items() if k in (
        "complete", "bank_count", "evaluation_role", "native_prefix_kv_frozen", "oracle_is_deployable",
        "new_answer_generations", "external_api_calls_made", "official_test_used")}
        | {s: compact(r) for s, r in results.items()}, "summary_sha256")
    study_source.save_or_check(out / "report.json", value, args.validate_only)
    study_source.save_or_check(out / "brief_summary.json", brief, args.validate_only)
    print(f"[retrieval-coverage] complete summary={out / 'brief_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
