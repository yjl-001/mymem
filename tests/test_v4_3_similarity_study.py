from copy import deepcopy
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from memgen.experience.v4_3_artifacts import atomic_json, read_json
from memgen.experience.v4_3_bank import canonical_hash, seal, text_hash
from memgen.experience.v4_3_question_selector import NO_MEMORY, POLICY as OLD_POLICY, fit_selector
from memgen.experience.v4_3_similarity_study import (
    KEYS, diagnostics, evaluate, fit_study, key_texts, metrics, predict, route, score_features,
)
from tests import test_v4_3_question_selector as selector_tests
from tests.test_v4_3_side_kv import torch


def rows(split, harmful=False):
    result = selector_tests.utility_rows(split, n=20, harmful=harmful)
    for r in result:
        r["token_counts"] = {NO_MEMORY: 10, "bank-a": 20, "bank-b": 30}
    return result


def vectors():
    good = {"bank-a": [1., 0.], "bank-b": [-1., 0.]}
    bad = {"bank-a": [-1., 0.], "bank-b": [1., 0.]}
    return {"full_card": bad, "applicability": good, "problem_structure": bad}


class SimilarityStudyTests(unittest.TestCase):
    def test_train_cv_selects_applicability_tune_cannot_change_policy(self):
        train, tune = rows("train"), rows("tune")
        study = fit_study(train, ["bank-a", "bank-b"], vectors(), "profile")
        self.assertEqual(study["recommended"], "applicability/threshold")
        self.assertEqual(study["candidates"][study["recommended"]]["train_oof"]["accuracy"], 1.)
        self.assertEqual(predict(study, [1., 0.])["selected_bank"], "bank-a")
        original = deepcopy(study)
        good = evaluate(study, tune, "tune", .5, "bank-a")
        for r in tune:
            r["rewards"]["bank-a"], r["rewards"]["bank-b"] = r["rewards"]["bank-b"], r["rewards"]["bank-a"]
        bad = evaluate(study, tune, "tune", .5, "bank-a")
        self.assertEqual(good["methods"]["recommended"]["accuracy"], 1.)
        self.assertEqual(bad["methods"]["recommended"]["accuracy"], 0.)
        self.assertEqual(original, study)
        self.assertEqual(study, fit_study(train, ["bank-a", "bank-b"], vectors(), "profile"))
        tune[0]["sample_id"] = train[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate(study, tune, "tune", .5, "bank-a")
        with self.assertRaisesRegex(ValueError, "train/tune only"):
            evaluate(study, rows("eval"), "eval", .5, "bank-a")

    def test_harmful_memories_abstain_and_token_stats_are_exact(self):
        train = rows("train", harmful=True)
        study = fit_study(train, ["bank-a", "bank-b"], vectors(), "profile")
        self.assertEqual(predict(study, [1., 0.])["selected_bank"], NO_MEMORY)
        r = rows("tune")[:2]
        r[0]["token_counts"]["bank-a"] = 1024
        r[1]["token_counts"]["bank-b"] = 40
        m = metrics(r, ["bank-a", "bank-b"])
        self.assertEqual(m["gain"], 2)
        self.assertEqual(m["generated_tokens"]["total"], 1064)
        self.assertEqual(m["generated_tokens"]["mean"], 532)
        self.assertEqual(m["generated_tokens"]["at_1024_token_limit_count"], 1)
        self.assertEqual(m["generated_tokens"]["correct_at_limit_count"], 1)

    def test_threshold_boundary_margin_and_ties(self):
        ids = ["bank-a", "bank-b"]
        scores = np.asarray([[.5, .2], [.8, .8], [.9, .5]])
        self.assertEqual(route(scores, ids, {"threshold": .5, "margin": 0.}), [NO_MEMORY, "bank-a", "bank-a"])
        self.assertEqual(route(scores, ids, {"threshold": .5, "margin": .1}), [NO_MEMORY, NO_MEMORY, "bank-a"])
        with self.assertRaisesRegex(ValueError, "unit"):
            score_features([[2., 0.]], ids, vectors()["full_card"])
        with self.assertRaisesRegex(ValueError, "split isolation"):
            fit_study(rows("tune"), ids, vectors(), "profile")

    def test_diagnostics_count_each_top1_and_each_pair_once(self):
        train, tune = rows("train"), rows("tune")
        study = fit_study(train, ["bank-a", "bank-b"], vectors(), "profile")
        d = diagnostics(study, train, tune)["applicability"]
        self.assertEqual(sum(b["count"] for b in d["top1_similarity"]), 20)
        self.assertEqual(sum(b["gain"] for b in d["top1_margin"]), 20)
        self.assertEqual(sum(b["count"] for b in d["all_question_bank_pairs_similarity"]), 40)
        self.assertEqual(d["by_bank"]["bank-a"]["when_top1"]["accuracy"], 1.)
        self.assertEqual(d["by_bank"]["bank-a"]["fixed_on_all_questions"]["accuracy"], .5)

    def test_real_shell_dispatch_and_api_keys_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "python"
            stub.write_text('#!/usr/bin/env bash\nset -eu\n'
                            '[[ -z "${DEEPSEEK_API_KEY+x}${OPENAI_API_KEY+x}${GLM_API_KEY+x}${ANTHROPIC_API_KEY+x}" ]]\n'
                            'printf "%s\\n" "$@"\n')
            stub.chmod(0o755)
            env = {**os.environ, "MEMGEN_PYTHON_BIN": str(stub), "MEMGEN_V43_SIMILARITY_ROOT": tmp + "/out",
                   "MEMGEN_V43_VALIDATE_ONLY": "1", "DEEPSEEK_API_KEY": "sentinel"}
            root = Path(__file__).resolve().parents[1]
            completed = subprocess.run(["bash", str(root / "test.sh"), "similarity-study"], cwd=root,
                                       env=env, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("scripts/run_v4_3_similarity_study.py", completed.stdout)
            self.assertIn("--validate-only", completed.stdout)
            self.assertIn(tmp + "/out", completed.stdout)


@unittest.skipIf(torch is None, "Torch required for real frozen-encoder fixture")
class SimilarityStudyIntegrationTests(unittest.TestCase):
    def test_encode_reuse_freeze_before_tune_no_eval_access_resume_and_tamper(self):
        from memgen.model import v4_3_question_selector as runtime_module
        from scripts import run_v4_3_similarity_study as driver
        runtime = selector_tests.QuestionSelectorRuntimeTests.runtime(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(selector_dir=root / "source", output_dir=root / "study", bank_dir=root / "bank",
                                   side_kv_dir=root / "side", equivalence_dir=root / "eq", device="cpu",
                                   cache_manifest=root / "cache" / "manifest.json", semantic_packets=root / "packets" / "p.jsonl",
                                   split_manifest=root / "split" / "manifest.json", resume=True, validate_only=False, plan_only=False)
            records = [{"bank_id": "bank-" + b, "quality_tier": "primary", "descriptor": "Full card " + b,
                        "unified_process_card": {"applies_when": "Apply structure and decision " + b},
                        "clause_support": {"problem_structure": {"text": "Structure " + b}}, "record_sha256": b}
                       for b in ("a", "b")]
            original_records = deepcopy(records)
            cards = {r["bank_id"]: runtime_module.encode_text(runtime, r["descriptor"]) for r in records}
            entries, source_rows = [], {"train": [], "tune": []}
            for split, n in (("train", 10), ("tune", 4), ("eval", 1)):
                for i in range(n):
                    sid = split + "-" + str(i)
                    entries.append({"sample_id": sid, "selector_split": split, "question_sha256": text_hash(sid)})
                    if split != "eval":
                        source_rows[split].append({"sample_id": sid, "selector_split": split,
                            "feature": runtime_module.encode_text(runtime, "Question " + str(i)),
                            "rewards": {NO_MEMORY: i % 2, "bank-a": 1, "bank-b": 0}})
            sp = seal({"samples": entries, "seed": 43, "bank_ids": ["bank-a", "bank-b"], "policy": OLD_POLICY,
                       "reasoner": {}, "runtime_versions": {}, "numpy_version": np.__version__,
                       "prefix_manifest_sha256": {"bank-a": "a", "bank-b": "b"}}, "profile_sha256")
            old = fit_selector(source_rows["train"], source_rows["tune"], sp["bank_ids"], cards, sp["profile_sha256"])
            atomic_json(args.selector_dir / "profile.json", sp)
            atomic_json(args.selector_dir / "selector.json", old)
            atomic_json(args.selector_dir / "card_features.json", seal({"profile_sha256": sp["profile_sha256"], "features": cards}))
            for e in entries:
                if e["selector_split"] == "eval":
                    continue  # All eval outcomes are deliberately absent.
                r = next(r for r in source_rows[e["selector_split"]] if r["sample_id"] == e["sample_id"])
                folder = args.selector_dir / "samples" / e["sample_id"]
                binding = {"sample_id": e["sample_id"], "question_sha256": e["question_sha256"], "profile_sha256": sp["profile_sha256"]}
                atomic_json(folder / "feature.json", seal({**binding, "feature": r["feature"]}))
                for a in [NO_MEMORY, *sp["bank_ids"]]:
                    tokens = [1, 2, 3] if a == NO_MEMORY else [4, 5, 6, 7]
                    atomic_json(folder / (a + ".json"), seal({**binding, "action": a, "result": {
                        "strict_reward": r["rewards"][a], "continuation_token_ids": tokens,
                        "continuation_token_ids_sha256": canonical_hash(tokens)}}))
            source_mtimes = {p: p.stat().st_mtime_ns for p in args.selector_dir.rglob("*.json")}
            real_read_rows = driver.read_rows
            def guarded_read(*a):
                if a[-1] == "tune":
                    self.assertTrue((args.output_dir / "selector.json").exists())
                return real_read_rows(*a)
            with patch.object(driver, "parse_args", return_value=args), \
                 patch.object(driver.source, "prepare", return_value=((None, None, None, records), {}, sp)), \
                 patch.object(driver, "read_rows", side_effect=guarded_read), \
                 patch.object(runtime_module, "load_runtime", return_value=runtime) as loader, \
                 patch.object(runtime_module, "encode_text", wraps=runtime_module.encode_text) as encoder, \
                 patch.object(runtime_module, "generate", side_effect=AssertionError("No regeneration allowed")):
                driver.main()
                self.assertEqual(encoder.call_count, 4)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(records, original_records)
                self.assertEqual(key_texts(records)["full_card"], {r["bank_id"]: r["descriptor"] for r in records})
                report = read_json(args.output_dir / "report.json")
                self.assertEqual(report["sample_counts"], {"train": 10, "tune": 4})
                self.assertEqual(report["tune"]["methods"]["fixed_from_train"]["generated_tokens"]["total"], 16)
                self.assertFalse(report["official_test_used"])
                mtimes = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                driver.main()  # Resume also revalidates tables and creates no model.
                args.validate_only = True
                driver.main()
                self.assertEqual(encoder.call_count, 4)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
                self.assertEqual(source_mtimes, {p: p.stat().st_mtime_ns for p in source_mtimes})
                broken_path = args.output_dir / "key_features" / "applicability" / "bank-a.json"
                broken = read_json(broken_path)
                broken["feature"][0] += 1
                atomic_json(broken_path, broken)
                with self.assertRaises(ValueError):
                    driver.main()


if __name__ == "__main__":
    unittest.main()
