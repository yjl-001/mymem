"""Construction invariants with deterministic teacher/reasoner fixtures (no downloads)."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.bank_construction.artifacts import Store, digest, atomic_json, read_json, run_lock
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.dataset import prepare_split
from memgen.experience.bank_construction.rollouts import rollout_plan, run_rollouts
from memgen.experience.bank_construction.review import classify, run_review
from memgen.experience.bank_construction.experiences import run_experiences
from memgen.experience.bank_construction.grouping import run_grouping
from memgen.experience.bank_construction.cards import run_cards
from memgen.experience.bank_construction.teacher import Teacher, parse_object
from memgen.experience.bank_construction.schemas import validate_review, validate_partition
from memgen.experience.bank_construction.compilation import select_bank
from memgen.experience.bank_construction.evaluation import metrics


class Rows(list):
    def add_column(self, name, values):
        return Rows(dict(r, **{name: v}) for r, v in zip(self, values))

    def train_test_split(self, *, test_size, shuffle, seed):
        import random
        values = list(self)
        random.Random(seed).shuffle(values)
        return {"train": Rows(values[test_size:]), "test": Rows(values[:test_size])}


def fixture_split(config):
    raw = {"train": Rows({"question": f"Train question {i}", "answer": "Compute.\n#### 2"} for i in range(6)),
           "test": Rows([{"question": "Official test only", "answer": "Compute.\n#### 2"}])}
    return prepare_split(raw, config, "dataset-commit")


def review(verdict="correct", final="correct"):
    return {"reasoning_verdict": verdict, "final_answer_verdict": final, "explanation": "Checked each step",
            "errors": [{"location": "first step", "error": "wrong quantity", "correction": "track remainder"}]
            if verdict == "incorrect" else []}


def signature():
    return {"transferable": True, "reason": "Grounded remaining-quantity repair", "experience_kind": "reasoning",
            "problem_structure": "Sequential changes in a remaining quantity", "decision_point": "Choose the ratio base",
            "failure_mechanism": "Uses the initial amount instead of the remainder", "repair_operator": "Update the remaining amount",
            "verification_operator": "Check conservation", "exclusion_conditions": ["Ratios explicitly share the original base"],
            "substitution_check": "Names and quantities may change", "near_miss": "Fixed original base",
            "source_grounding": "Failure uses the wrong ratio base"}


class FixtureTeacher:
    def __init__(self):
        self.calls = []

    def ask(self, task, payload, validate):
        self.calls.append((task, payload))
        if task == "review":
            value = review("incorrect" if "bad" in payload["trajectory"] else "correct",
                           "incorrect" if "bad" in payload["trajectory"] else "correct")
        elif task == "extract":
            value = signature()
        elif task == "partition":
            value = {"groups": [{"members": [r["evidence_id"] for r in payload["evidence"]],
                "method": "Update remaining quantity", "applies_when": "Sequential remainder changes",
                "exclusions": ["Original-base ratios"], "rationale": "Same operation and boundary"}]}
        elif task == "match":
            value = {"candidate_ids": [r["group_id"] for r in payload["candidates"]], "rationale": "Compatible method"}
        elif task == "merge":
            value = {"merge": True, "method": "Update remaining quantity", "applies_when": "Sequential remainder changes",
                     "exclusions": ["Original-base ratios"], "rationale": "Same operation"}
        elif task == "membership":
            value = {"incompatible_ids": [], "rationale": "All supplied members support this method"}
        elif task == "card":
            value = {"coherent": True, "reason": "Common repair", "support_ids": [r["evidence_id"] for r in payload["evidence"]],
                "card": {"problem_structure": "Sequential changes", "decision_point": "Select ratio base",
                    "applies_when": "The ratio refers to current remainder", "procedure": ["Update remainder before applying the next ratio"],
                    "avoid": ["Do not reuse initial amount"], "exclusions": ["Each ratio uses original amount"],
                    "verify": ["Check conservation"]}, "substitution_check": "Preserves relations", "near_miss": "Original-base ratios"}
        elif task == "card_review":
            value = {"quality_tier": "primary", "reason": "Grounded with boundaries", "issues": [],
                     "support_ids": [r["evidence_id"] for r in payload["evidence"]]}
        else:
            raise AssertionError(task)
        validate(value)
        return value


class FixtureReasoner:
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []
        self.closed = False

    def generate(self, prompt, **kwargs):
        self.calls.append(kwargs)
        correct = not kwargs["sampling"]
        text = "good \\boxed{2}" if correct else "bad \\boxed{3}"
        return {"text": text, "token_ids": [1, 2], "token_count": 2, "prompt_token_count": 10,
                "seed": kwargs["seed"], "stop_reason": "completed_boxed_answer", "truncated": False}

    def close(self):
        self.closed = True


class ConstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = replace(ConstructionConfig(), val_ratio=.34, group_batch_size=1, candidate_batch_size=1)
        self.store = Store(Path(self.temp.name), {"fixture": True})
        self.split = fixture_split(self.config)
        self.store.put("split", self.split, {})

    def test_fixed_sampling_and_per_sample_rng(self):
        row = self.split["splits"]["train"][0]
        plans = rollout_plan(row, self.config)
        self.assertEqual([p["sampling"] for p in plans], [False] + [True] * 7)
        self.assertEqual(len({p["seed"] for p in plans}), 8)
        self.assertEqual(plans, rollout_plan(row, self.config))
        self.assertNotEqual(plans[0]["seed"], rollout_plan(self.split["splits"]["train"][1], self.config)[0]["seed"])
        with self.assertRaises(ValueError):
            replace(self.config, max_new_tokens=768)

    def test_builder_split_counts_identity_and_duplicate_leakage(self):
        self.assertEqual(self.split["counts"], {"train": 4, "valid": 2, "test": 1})
        self.assertEqual(self.split, fixture_split(self.config))
        sets = [set(r["sample_id"] for r in rows) for rows in self.split["splits"].values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        raw = {"train": Rows([{"question": "duplicate", "answer": "#### 2"}] * 3),
               "test": Rows([{"question": "different", "answer": "#### 2"}])}
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            prepare_split(raw, self.config, "r")

    def test_success_requires_process_and_truncation_overrides_failures(self):
        generation = {"truncated": False}
        verifier = {"reward": 1., "format_valid": True}
        self.assertEqual(classify(generation, verifier, review())["outcome"], "success")
        result = classify(generation, verifier, review("incorrect"))
        self.assertEqual(result["outcome"], "failure")
        self.assertEqual(result["failure_types"], ["reasoning_error"])
        self.assertEqual(classify(generation, verifier, review("uncertain"))["outcome"], "uncertain")
        self.assertEqual(classify({"truncated": True}, verifier, None)["outcome"], "truncated")
        wrong = classify(generation, {"reward": 0., "format_valid": False}, review("incorrect", "incorrect"))
        self.assertEqual(set(wrong["failure_types"]), {"answer_error", "format_error", "reasoning_error"})
        conflict = classify(generation, verifier, review("correct", "incorrect"))
        self.assertEqual(conflict["outcome"], "uncertain")

    def test_format_diagnostic_last_number_is_not_answer_truth(self):
        verifier = {"reward": 0., "format_valid": False, "diagnostic_answer_correct": False}
        result = classify({"truncated": False}, verifier, review())
        self.assertEqual(result["failure_types"], ["format_error"])

    def test_store_tamper_and_input_drift(self):
        self.store.put("test", {"value": 1}, {"a": 1})
        with self.assertRaisesRegex(ValueError, "input drift"):
            self.store.get("test", {"a": 2})
        path = self.store.root / "test.json"
        row = read_json(path)
        row["payload"]["value"] = 2
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            self.store.get("test")

    def test_immutable_and_lock(self):
        with run_lock(self.store.root):
            with self.assertRaises(RuntimeError):
                with run_lock(self.store.root):
                    pass
        with self.assertRaises(ValueError):
            atomic_json(self.store.root / "profile.json", {"different": True})
        with self.assertRaises(ValueError):
            self.store.get("../profile")

    def test_profile_resumes_pinned_sources_and_rejects_code_drift(self):
        from memgen.experience.bank_construction.sources import make_profile, implementation_hashes, environment
        profile = {"configuration": self.config.to_dict(), "implementation": implementation_hashes(),
                   "environment": environment(), "reasoner": {"source": "r", "revision": "a" * 40},
                   "teacher": {"source": "t", "revision": "b" * 40}}
        # This branch must not contact the Hub to resolve main again.
        self.assertIs(make_profile(self.config, profile), profile)
        profile["implementation"] = {}
        with self.assertRaisesRegex(ValueError, "changed"):
            make_profile(self.config, profile)

    def test_partition_rejects_duplicate_and_fabricated_members(self):
        group = {"members": ["a", "a"], "method": "x", "applies_when": "y", "exclusions": [], "rationale": "z"}
        with self.assertRaises(ValueError):
            validate_partition({"groups": [group]}, ["a", "b"])
        group["members"] = ["a", "invented"]
        with self.assertRaises(ValueError):
            validate_partition({"groups": [group]}, ["a", "b"])

    def test_end_to_end_train_to_cards_and_resume(self):
        model = FixtureReasoner()
        teacher = FixtureTeacher()
        with redirect_stdout(io.StringIO()):
            index = run_rollouts(self.store, self.config, lambda: model)
            run_review(self.store, teacher)
            evidence = run_experiences(self.store, teacher)
            groups = run_grouping(self.store, teacher, self.config)
            cards = run_cards(self.store, teacher, self.config)
        self.assertEqual(index["rollout_count"], 32)
        self.assertEqual(sum(not c["sampling"] for c in model.calls), 4)
        self.assertTrue(all(c["max_new_tokens"] == 1024 for c in model.calls))
        self.assertTrue(model.closed)
        self.assertEqual(len(evidence["keys"]), 4)
        self.assertEqual(len(groups["groups"]), 1)
        self.assertEqual(groups["groups"][0]["distinct_sample_count"], 4)
        self.assertEqual(cards["tier_counts"], {"primary": 1})
        self.assertTrue(any(task == "match" for task, _ in teacher.calls))
        def unexpected():
            raise AssertionError("Resume must not load a model")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(index, run_rollouts(self.store, self.config, unexpected))
            run_review(self.store, teacher)
            run_experiences(self.store, teacher)
            run_grouping(self.store, teacher, self.config)
            run_cards(self.store, teacher, self.config)
        forbidden = [r["question"] for name in ("valid", "test") for r in self.split["splits"][name]]
        for question in forbidden:
            self.assertNotIn(question, json.dumps(teacher.calls))

    def test_teacher_receipts_retry_and_cached_validation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["enable_thinking"] is False
                return json.dumps(messages)
        class Model:
            tokenizer = Tokenizer()
            count = 0
            def generate(self, prompt, **kwargs):
                self.count += 1
                return {"text": "invalid" if self.count == 1 else json.dumps(review()), "truncated": False}
            def close(self):
                pass
        model = Model()
        teacher = Teacher(self.store, self.config, lambda: model)
        with redirect_stdout(io.StringIO()):
            a = teacher.ask("review", {"trajectory": "x"}, validate_review)
            b = teacher.ask("review", {"trajectory": "x"}, validate_review)
        self.assertEqual(a, b)
        self.assertEqual(model.count, 2)
        self.assertEqual(len(list((self.store.root / "teacher/review").glob("*-request.json"))), 1)
        with self.assertRaises(ValueError):
            parse_object('{"x":1,"x":2}')

    def test_resume_after_exhausted_teacher_attempts(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)
        class Model:
            tokenizer = Tokenizer()
            valid = False
            calls = 0
            def generate(self, *args, **kwargs):
                self.calls += 1
                return {"text": json.dumps(review()) if self.valid else "bad", "truncated": False}
            def close(self):
                pass
        model = Model()
        teacher = Teacher(self.store, replace(self.config, teacher_retries=0), lambda: model)
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                teacher.ask("review", {"trajectory": "x"}, validate_review)
            model.valid = True
            value = teacher.ask("review", {"trajectory": "x"}, validate_review)
        self.assertEqual(value["reasoning_verdict"], "correct")
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(list((self.store.root / "teacher/review").glob("*-attempt-*.json"))), 2)

    def test_grouping_refuses_teacher_rejected_merge_and_checks_all_members(self):
        with redirect_stdout(io.StringIO()):
            run_rollouts(self.store, self.config, FixtureReasoner)
            teacher = FixtureTeacher()
            run_review(self.store, teacher)
            run_experiences(self.store, teacher)
            class RefuseTeacher(FixtureTeacher):
                def ask(self, task, payload, validate):
                    if task == "membership":
                        self.calls.append((task, payload))
                        result = {"incompatible_ids": [payload["evidence"][0]["evidence_id"]], "rationale": "Different necessary base"}
                        validate(result)
                        return result
                    return super().ask(task, payload, validate)
            teacher = RefuseTeacher()
            groups = run_grouping(self.store, teacher, self.config)
        self.assertEqual(len(groups["groups"]), 4)
        self.assertTrue(any(task == "membership" for task, _ in teacher.calls))

    def test_truncated_rollout_never_reaches_process_teacher(self):
        with redirect_stdout(io.StringIO()):
            class Truncated(FixtureReasoner):
                def generate(self, *args, **kwargs):
                    row = super().generate(*args, **kwargs)
                    row.update(truncated=True, stop_reason="length")
                    return row
            run_rollouts(self.store, self.config, Truncated)
            teacher = FixtureTeacher()
            result = run_review(self.store, teacher)
            evidence = run_experiences(self.store, teacher)
        self.assertEqual(result["outcomes"], {"truncated": 32})
        self.assertEqual(teacher.calls, [])
        self.assertEqual(evidence["keys"], [])

    def test_metrics_and_tie_break(self):
        base = [{"reward": 0., "generated_token_count": 2}, {"reward": 1., "generated_token_count": 3}]
        result = metrics(list(reversed(base)), base)
        self.assertEqual((result["gain"], result["harm"]), (1, 1))
        self.assertEqual(result["generated_tokens"]["total"], 5)
        self.assertEqual(select_bank([1., 0.], [{"bank_id": "b", "key_vector": [1., 0.]},
                                               {"bank_id": "a", "key_vector": [1., 0.]}]), ("a", 1.))


if __name__ == "__main__":
    unittest.main()
