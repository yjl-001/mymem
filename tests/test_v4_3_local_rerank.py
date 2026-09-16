import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from memgen.experience.v4_3_artifacts import atomic_json, read_json
from memgen.experience.v4_3_bank import canonical_hash, seal, text_hash
from memgen.experience.v4_3_local_rerank import POOLS, candidates, checked_scores, decisions, evaluate, fit
from memgen.experience.v4_3_similarity_study import KEYS, evaluate as evaluate_study, fit_study
from memgen.experience.v4_3_question_selector import NO_MEMORY, fit_selector
from memgen.model.v4_3_local_rerank import LocalReranker, model_identity, DEFAULT_REVISION
from tests import test_v4_3_retrieval_coverage as coverage_tests

IDS, VECTORS = coverage_tests.IDS, coverage_tests.VECTORS


def make_scores(rows):
    result = {}
    for r in rows:
        result[r["sample_id"]] = {}
        for bid in IDS:
            score = .9 if r["rewards"][bid] else .1
            result[r["sample_id"]][bid] = {"score": score, "logit_margin": math.log(score/(1-score)),
                "input_tokens": 10, "elapsed_seconds": .01}
    return result


def make_pools(rows):
    entries = [{"sample_id": r["sample_id"], "question_sha256": text_hash(r["sample_id"])} for r in rows]
    return candidates(entries, {r["sample_id"]: r["feature"] for r in rows}, IDS, VECTORS)


class ToyTokenizer:
    init_kwargs = {}
    def encode(self, text, add_special_tokens=False):
        if text in ("yes", "no"):
            return [self.convert_tokens_to_ids(text)]
        return [3 + int(text_hash(word)[:8], 16) % 125 for word in text.split()]
    def convert_tokens_to_ids(self, word):
        return {"yes": 1, "no": 2}.get(word)


def tiny_reranker():
    torch.manual_seed(43)
    torch.set_num_threads(1)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=2048, attention_dropout=0.)).to(torch.bfloat16).eval()
    rt = LocalReranker.__new__(LocalReranker)
    rt.model, rt.tokenizer, rt.device, rt.max_length = model, ToyTokenizer(), torch.device("cpu"), 2048
    rt.configure_protocol()
    return rt


class LocalRerankTests(unittest.TestCase):
    def test_pools_reproducible_and_train_calibration_protects_all_harmful_cases(self):
        train, tune = coverage_tests.fixture("train"), coverage_tests.fixture("tune")
        rows = train+tune
        pools, scores = make_pools(rows), make_scores(rows)
        self.assertEqual(pools, make_pools(rows))
        self.assertEqual(pools[train[0]["sample_id"]]["semantic_top3"], IDS[:3])
        for r in rows:
            self.assertEqual(len(set(pools[r["sample_id"]]["random_top3"])), 3)
            checked_scores({"scores": scores[r["sample_id"]]}, IDS)
        selector = fit(train, IDS, scores, pools, "profile")
        self.assertEqual(selector["rules"]["all_banks"]["threshold"], .1)
        legacy = {r["sample_id"]: NO_MEMORY for r in tune}
        result = evaluate(selector, tune, scores, pools, "tune", legacy, IDS[0])
        self.assertEqual(result["methods"]["all_banks/calibrated"]["correct"], 6)
        self.assertEqual(result["methods"]["all_banks/calibrated"]["harm"], 0)
        self.assertEqual(result["methods"]["all_banks/calibrated"]["memory_use_count"], 5)
        self.assertEqual(result["methods"]["all_banks/forced"]["harm"], 1)
        self.assertEqual(result["methods"]["semantic_top3/forced"]["reranker_cost"]["pair_count"], 21)
        digest = selector["selector_sha256"]
        for r in tune:
            r["rewards"] = {a: 1-v for a, v in r["rewards"].items()}
        evaluate(selector, tune, scores, pools, "tune", legacy, IDS[0])
        self.assertEqual(selector["selector_sha256"], digest)
        self.assertEqual(selector, fit(train, IDS, scores, pools, "profile"))
        with self.assertRaisesRegex(ValueError, "split isolation"):
            fit(tune, IDS, scores, pools, "profile")

    def test_score_validation_and_tie_break(self):
        rows = coverage_tests.fixture("train")
        scores, pools = make_scores(rows), make_pools(rows)
        for bid in IDS:
            scores[rows[0]["sample_id"]][bid]["score"] = .5
        selected = decisions(rows[:1], scores, pools, "all_banks", -1.)
        self.assertEqual(selected, [IDS[0]])
        self.assertEqual(decisions(rows[:1], scores, pools, "all_banks", .5), [NO_MEMORY])
        with self.assertRaisesRegex(ValueError, "logit margin"):
            checked_scores({"scores": scores[rows[0]["sample_id"]]}, IDS)

    def test_real_qwen3_logits_scoring_no_generation_and_no_truncation(self):
        rt = tiny_reranker()
        with patch.object(rt.model, "generate", side_effect=AssertionError("Must only score logits")):
            result = rt.score("How many items remain?", "Subtract the used items from the initial total.")
        checked_scores({"scores": {"fixture": result}}, ["fixture"])
        self.assertGreater(result["input_tokens"], 0)
        self.assertFalse(rt.model.training)
        rt.max_length = 2
        with self.assertRaisesRegex(ValueError, "truncation"):
            rt.score("Question", "Memory")

    def test_model_identity_pins_local_weights_and_rejects_floating_hub_revision(self):
        with self.assertRaisesRegex(ValueError, "exact commit"):
            model_identity("Qwen/Qwen3-Reranker-8B", "main")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"config.json").write_text('{}')
            (root/"model.safetensors").write_bytes(b"fixture weights only")
            first = model_identity(tmp, None)
            (root/"model.safetensors").write_bytes(b"changed weights")
            self.assertNotEqual(first["snapshot_sha256"], model_identity(tmp, None)["snapshot_sha256"])

    def test_complete_driver_partial_resume_freeze_before_tune_and_readonly_validation(self):
        from scripts import run_v4_3_local_rerank as driver
        from memgen.model import v4_3_local_rerank as model_module
        rows = {s: coverage_tests.fixture(s) for s in ("train", "tune")}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(selector_dir=root/"source", study_dir=root/"study", output_dir=root/"out",
                reranker_model="Qwen/Qwen3-Reranker-8B", reranker_revision=DEFAULT_REVISION,
                device="cpu", max_length=2048, plan_only=False, validate_only=False)
            entries = [{"sample_id": r["sample_id"], "selector_split": s, "question_sha256": text_hash(r["sample_id"])}
                       for s, rs in rows.items() for r in rs]
            sp = seal({"samples": entries, "bank_ids": IDS}, "profile_sha256")
            base = {s: [{k: v for k, v in r.items() if k != "token_counts"} for r in rs] for s, rs in rows.items()}
            old = fit_selector(base["train"], base["tune"], IDS, VECTORS, sp["profile_sha256"])
            previous = seal({"samples": entries, "key_texts": {"full_card": {bid: "Memory for "+bid for bid in IDS}},
                            "evaluation_role": "fixture_reused_calibration"}, "profile_sha256")
            study = fit_study(rows["train"], IDS, {k: VECTORS for k in KEYS}, previous["profile_sha256"])
            original_report = seal({"tune_data_sha256": canonical_hash(rows["tune"]),
                **{s: evaluate_study(study, rs, s, old["semantic_threshold"], old["fixed_bank_from_train"]) for s, rs in rows.items()}}, "report_sha256")
            for e in entries:
                r = next(r for r in rows[e["selector_split"]] if r["sample_id"] == e["sample_id"])
                folder = args.selector_dir/"samples"/e["sample_id"]
                binding = {"profile_sha256": sp["profile_sha256"], "sample_id": e["sample_id"], "question_sha256": e["question_sha256"]}
                atomic_json(folder/"feature.json", seal({**binding, "feature": r["feature"]}))
                for action, count in r["token_counts"].items():
                    tokens = [1]*count
                    atomic_json(folder/(action+".json"), seal({**binding, "action": action, "result": {
                        "strict_reward": r["rewards"][action], "continuation_token_ids": tokens,
                        "continuation_token_ids_sha256": canonical_hash(tokens)}}))
            rt = tiny_reranker()
            real_score, real_read = rt.score, driver.source.study_source.read_rows
            calls = []
            def guarded_score(question, descriptor):
                if question.startswith("tune"):
                    self.assertTrue((args.output_dir/"selector.json").exists())
                if len(calls) == 1 and not (args.output_dir/"resume-marker").exists():
                    (args.output_dir/"resume-marker").touch()
                    raise RuntimeError("fixture interrupted")
                calls.append((question, descriptor))
                return real_score(question, descriptor)
            def guarded_read(*a):
                if a[-1] == "tune":
                    self.assertTrue((args.output_dir/"tune_predictions.json").exists())
                return real_read(*a)
            source_times = {p: p.stat().st_mtime_ns for p in args.selector_dir.rglob("*.json")}
            with patch.object(driver, "parse_args", return_value=args), \
                 patch.object(driver.source, "prepare", return_value=(sp, old, previous, study, original_report)), \
                 patch.object(driver, "load_questions", return_value={e["sample_id"]: e["sample_id"] for e in entries}) as questions, \
                 patch.object(driver.source.study_source, "read_rows", side_effect=guarded_read), \
                 patch.object(model_module, "LocalReranker", return_value=rt) as loader, \
                 patch.object(rt, "score", side_effect=guarded_score):
                with self.assertRaisesRegex(RuntimeError, "fixture interrupted"):
                    driver.main()
                driver.main()
                self.assertEqual(len(calls), 70)
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(questions.call_count, 1)
                report = read_json(args.output_dir/"report.json")
                self.assertEqual(report["sample_counts"], {"train": 7, "tune": 7})
                self.assertEqual(report["new_reasoner_generations"], 0)
                times = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                driver.main()
                args.validate_only = True
                driver.main()
                self.assertEqual(len(calls), 70)
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(times, {p: p.stat().st_mtime_ns for p in times})
                self.assertEqual(source_times, {p: p.stat().st_mtime_ns for p in source_times})
                path = args.output_dir/"samples"/"tune-0"/"bank-a.json"
                bad = read_json(path)
                bad["result"]["score"] = .123
                atomic_json(path, bad)
                with self.assertRaises(ValueError):
                    driver.main()

    def test_shell_dispatch_default_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp)/"python"
            stub.write_text('#!/usr/bin/env bash\nset -eu\n'
                '[[ -z "${DEEPSEEK_API_KEY+x}${OPENAI_API_KEY+x}" ]]\n'
                'printf "%s\\n" "$@"\n')
            stub.chmod(0o755)
            root = Path(__file__).resolve().parents[1]
            for validate in ("0", "1"):
                result = subprocess.run(["bash", str(root/"test.sh"), "local-rerank"], cwd=root, capture_output=True, text=True,
                    env={**os.environ, "MEMGEN_PYTHON_BIN": str(stub), "MEMGEN_V43_VALIDATE_ONLY": validate, "DEEPSEEK_API_KEY": "sentinel"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("scripts/run_v4_3_local_rerank.py", result.stdout)
                self.assertIn("Qwen/Qwen3-Reranker-8B", result.stdout)
                self.assertEqual("--validate-only" in result.stdout, validate == "1")


if __name__ == "__main__":
    unittest.main()
