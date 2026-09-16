import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from memgen.experience.v4_3_artifacts import atomic_json, implementation_hashes, read_json
from memgen.experience.v4_3_bank import canonical_hash, seal, text_hash
from memgen.experience.v4_3_question_selector import NO_MEMORY, fit_selector
from memgen.experience.v4_3_retrieval_coverage import compact, coverage
from memgen.experience.v4_3_similarity_study import KEYS, POLICY as STUDY_POLICY, evaluate, fit_study

IDS = ["bank-"+b for b in "abcde"]
VECTORS = dict(zip(IDS, [[1., 0.], [.8, .6], [0., 1.], [-.6, .8], [-1., 0.]]))


def fixture(split):
    # Three repairable failures at ranks 1/3/5, one impossible failure, two harms,
    # and one baseline-correct question preserved by top1.
    base = [0, 0, 0, 0, 1, 1, 1]
    good = [{0}, {2}, {4}, set(), {2}, set(), {0}]
    return [{"sample_id": f"{split}-{i}", "selector_split": split, "feature": [1., 0.],
             "rewards": {NO_MEMORY: base[i], **{b: int(j in good[i]) for j, b in enumerate(IDS)}},
             "token_counts": {NO_MEMORY: 3, **{b: 4 for b in IDS}}} for i in range(7)]


class CoverageTests(unittest.TestCase):
    def test_rank_denominator_oracle_and_harm_decomposition(self):
        result = coverage(fixture("tune"), IDS, {k: VECTORS for k in KEYS}, "tune")
        self.assertEqual(result["repairable_count"], 3)
        self.assertEqual(result["unrepairable_baseline_wrong_count"], 1)
        self.assertEqual(result["all_bank_oracle_accuracy"], 6/7)
        key = result["keys"]["full_card"]
        self.assertEqual(key["repair_first_success_rank_histogram"], {"1": 1, "2": 0, "3": 1, "4": 0, "5": 1})
        self.assertEqual(key["top1"]["correct"], 2)
        self.assertEqual(key["top1"]["gain"], 1)
        self.assertEqual(key["top1_harm_count"], 2)
        self.assertEqual(key["top1_harm_all_banks_wrong_count"], 1)
        for k, hits in ((1, 1), (3, 2), (5, 3)):
            point = key["top_k"][str(k)]
            self.assertEqual(point["repair_hit_count"], hits)
            self.assertEqual(point["repair_recall"], hits/3)
            self.assertAlmostEqual(point["uniform_random_expected_recall"], k/5)
            self.assertEqual(point["oracle_accuracy_including_no_memory"], (3+hits)/7)
        self.assertEqual(key["top_k"]["3"]["top1_harm_with_safe_bank_in_top_k"], 1)
        self.assertEqual(key["top_k"]["3"]["fraction_of_top1_missed_repairs_recovered"], .5)
        self.assertEqual(compact(result)["keys"]["full_card"]["repair_hits_at_k"], {"1": 1, "3": 2, "5": 3})

    def test_random_without_replacement_multiple_good_banks_and_empty_denominator(self):
        rows = fixture("train")[:1]
        rows[0]["rewards"]["bank-b"] = 1
        result = coverage(rows, IDS, {k: VECTORS for k in KEYS}, "train")
        self.assertAlmostEqual(result["keys"]["full_card"]["top_k"]["2"]["uniform_random_expected_recall"], .7)
        for b in IDS:
            rows[0]["rewards"][b] = 0
        result = coverage(rows, IDS, {k: VECTORS for k in KEYS}, "train")
        self.assertEqual(result["repairable_count"], 0)
        self.assertIsNone(result["keys"]["full_card"]["top_k"]["1"]["repair_recall"])
        self.assertIsNone(result["keys"]["full_card"]["top_k"]["1"]["uniform_random_expected_recall"])

    def test_exact_ties_follow_bank_id_and_test_split_rejected(self):
        tied = {b: [1., 0.] for b in IDS}
        result = coverage(fixture("tune"), IDS, {k: tied for k in KEYS}, "tune")
        key = result["keys"]["problem_structure"]
        self.assertEqual(key["cases"][1]["first_correct_bank_rank"], 3)
        self.assertEqual(key["cases"][0]["ranked_bank_ids"], IDS)
        self.assertEqual(key["top_k"]["3"]["top_k_boundary_tie_count"], 7)
        with self.assertRaisesRegex(ValueError, "train/tune only"):
            coverage(fixture("eval"), IDS, {k: VECTORS for k in KEYS}, "eval")

    def test_authenticated_end_to_end_readonly_resume_and_source_drift(self):
        from scripts import run_v4_3_retrieval_coverage as driver
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(selector_dir=root/"source", study_dir=root/"study", output_dir=root/"out", validate_only=False)
            rows = {s: fixture(s) for s in ("train", "tune")}
            entries = [{"sample_id": r["sample_id"], "selector_split": s, "question_sha256": text_hash(r["sample_id"]),
                        "dataset_split": "train", "logical_split": "calibration-val"} for s, rs in rows.items() for r in rs]
            # Eval entry exists in the source plan, but no eval outcomes exist on disk.
            sp = seal({"bank_ids": IDS, "samples": entries + [{"sample_id": "eval-only", "selector_split": "eval"}],
                       "implementation_sha256": implementation_hashes(("memgen/experience/v4_3_question_selector.py",))}, "profile_sha256")
            base_rows = {s: [{k: v for k, v in r.items() if k != "token_counts"} for r in rs] for s, rs in rows.items()}
            old = fit_selector(base_rows["train"], base_rows["tune"], IDS, VECTORS, sp["profile_sha256"])
            for name, value in (("profile.json", sp), ("selector.json", old),
                                ("card_features.json", seal({"profile_sha256": sp["profile_sha256"], "features": VECTORS}))):
                atomic_json(args.selector_dir/name, value)
            profile = seal({"source_profile_sha256": sp["profile_sha256"], "legacy_selector_sha256": old["selector_sha256"],
                "bank_ids": IDS, "samples": entries, "policy": STUDY_POLICY, "evaluation_role": "fixture_reused_calibration",
                "key_texts": {k: {b: k+" "+b for b in IDS} for k in KEYS},
                "implementation_sha256": implementation_hashes(driver.study_source.IMPLEMENTATION)}, "profile_sha256")
            study = fit_study(rows["train"], IDS, {k: VECTORS for k in KEYS}, profile["profile_sha256"])
            report = seal({"complete": True, "profile_sha256": profile["profile_sha256"], "selector_sha256": study["selector_sha256"],
                "tune_data_sha256": canonical_hash(rows["tune"]),
                **{s: evaluate(study, rs, s, old["semantic_threshold"], old["fixed_bank_from_train"]) for s, rs in rows.items()}}, "report_sha256")
            for name, value in (("profile.json", profile), ("selector.json", study), ("report.json", report)):
                atomic_json(args.study_dir/name, value)
            for key in KEYS[1:]:
                for bid in IDS:
                    atomic_json(args.study_dir/"key_features"/key/(bid+".json"), seal({"profile_sha256": profile["profile_sha256"],
                        "key": key, "bank_id": bid, "text_sha256": text_hash(profile["key_texts"][key][bid]), "feature": VECTORS[bid]}))
            for s, rs in rows.items():
                for r in rs:
                    folder = args.selector_dir/"samples"/r["sample_id"]
                    binding = {"profile_sha256": sp["profile_sha256"], "sample_id": r["sample_id"], "question_sha256": text_hash(r["sample_id"])}
                    atomic_json(folder/"feature.json", seal({**binding, "feature": r["feature"]}))
                    for action, count in r["token_counts"].items():
                        tokens = [1]*count
                        atomic_json(folder/(action+".json"), seal({**binding, "action": action, "result": {
                            "strict_reward": r["rewards"][action], "continuation_token_ids": tokens,
                            "continuation_token_ids_sha256": canonical_hash(tokens)}}))
            sources = {p: p.stat().st_mtime_ns for directory in (args.selector_dir, args.study_dir) for p in directory.rglob("*.json")}
            with patch.object(driver, "parse_args", return_value=args):
                driver.main()
                result = read_json(args.output_dir/"brief_summary.json")
                self.assertEqual(result["tune"]["repairable_count"], 3)
                self.assertEqual(result["train"]["keys"]["problem_structure"]["repair_hits_at_k"]["3"], 2)
                before = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                driver.main()
                args.validate_only = True
                driver.main()
                self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
                self.assertEqual(sources, {p: p.stat().st_mtime_ns for p in sources})
                path = args.selector_dir/"samples"/"tune-0"/"bank-a.json"
                bad = read_json(path)
                bad["result"]["strict_reward"] = 0
                atomic_json(path, seal(bad))
                with self.assertRaisesRegex(ValueError, "Source rows differ"):
                    driver.main()

    def test_shell_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp)/"python"
            stub.write_text('#!/usr/bin/env bash\nset -eu\n'
                '[[ -z "${DEEPSEEK_API_KEY+x}${OPENAI_API_KEY+x}" ]]\n'
                'printf "%s\\n" "$@"\n')
            stub.chmod(0o755)
            root = Path(__file__).resolve().parents[1]
            for validate in ("0", "1"):
                result = subprocess.run(["bash", str(root/"test.sh"), "retrieval-coverage"], cwd=root, capture_output=True, text=True,
                    env={**os.environ, "MEMGEN_PYTHON_BIN": str(stub), "MEMGEN_V43_VALIDATE_ONLY": validate, "DEEPSEEK_API_KEY": "sentinel"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("scripts/run_v4_3_retrieval_coverage.py", result.stdout)
                self.assertEqual("--validate-only" in result.stdout, validate == "1")


if __name__ == "__main__":
    unittest.main()
