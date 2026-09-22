"""V5 universal construction and input-only selector contracts."""
from __future__ import annotations

from dataclasses import replace
import io
from contextlib import redirect_stdout
import json
import subprocess
from pathlib import Path
import tempfile
import unittest

from memgen.experience.bank_construction.artifacts import Store
from memgen.experience.v5.cards import render_card, run_cards, selector_text
from memgen.experience.v5.config import V5Config
from memgen.experience.v5.contrasts import choose_contrasts
from memgen.experience.v5.grouping import judge_pairs, run_grouping
from memgen.experience.v5.review import classify
from memgen.experience.v5.schemas import validate_atom, validate_card
from memgen.experience.v5.selector import (FEATURE_NAMES, candidate_features, fit_policy,
                                            reranker_document, select)


def atom(number):
    return {"atom_id": f"atom-{number}", "input_id": f"input-{number}",
        "contrast_id": f"contrast-{number}", "success_episode_id": f"good-{number}",
        "failure_episode_id": f"bad-{number}", "failure_types": ["process_error"],
        "teacher_assessment": {"transferable": True, "confidence": .9, "reason": "grounded"},
        "applicability": {"task_goal": "derive a quantity", "problem_structure": "shared ratio",
            "observable_cues": ["quantities share a scale"], "required_operation": "introduce a scale"},
        "experience": {"do": ["introduce one shared scale"], "avoid": ["scale terms independently"],
            "why": "the relation must remain invariant", "verify": ["substitute the result"],
            "runtime_limits": ["stop if the relation is additive"]},
        "exclusions_from_input": ["independent additive quantities"],
        "failure_mechanism": "independent scaling breaks the ratio"}


class FixtureTeacher:
    def ask(self, task, payload, validate):
        if task == "partition_atoms":
            value = {"groups": [{"members": [row["atom_id"] for row in payload["atoms"]],
                "shared_method": "introduce a shared scale", "applicability": "shared ratio",
                "exclusions": ["independent quantities"], "rationale": "same operation"}]}
        elif task == "summarize_group":
            value = {"shared_method": "introduce a shared scale", "applicability": "shared ratio",
                "exclusions": ["independent additive quantities"],
                "failure_mechanisms": ["independent scaling"],
                "recommended_actions": ["introduce one scale"], "avoid": ["scale independently"],
                "verification": ["substitute back"], "runtime_limits": ["requires a ratio"]}
        elif task == "card":
            value = {"selector_key": {"task_goal": "derive a quantity",
                "problem_structure": "several quantities share a multiplicative scale",
                "observable_cues": ["one quantity is a multiple of another"],
                "required_operation": "represent the common scale",
                "exclusions_from_input": ["quantities are independent"]},
                "memory_payload": {"applicability_summary": "Use for a shared multiplicative scale.",
                "method": ["introduce one scale variable", "substitute relations"],
                "avoid": ["assign independent scales"], "rationale": "one invariant links all terms",
                "verification": ["substitute into every relation"], "runtime_limits": ["not additive-only"]}}
        elif task == "card_review":
            value = {"quality_tier": "primary", "coherent": True, "executable": True,
                "input_observable_key": True, "contradiction_free": True,
                "reason": "supported by three inputs", "issues": []}
        else:
            raise AssertionError(task)
        validate(value)
        return value


class V5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name), {"fixture": "v5"})
        self.config = replace(V5Config(), teacher_backend="transformers", reranker_enabled=False,
                              atom_partition_batch_size=12, minimum_primary_inputs=3)

    def test_episode_classification_keeps_truncation_separate_and_process_strict(self):
        episode = {"outcome": {"truncated": False, "format_correct": True, "answer_correct": True}}
        correct = {"process_verdict": "correct", "answer_verdict": "correct",
                   "explanation": "valid", "errors": []}
        self.assertEqual(classify(episode, correct)["status"], "success")
        wrong = {**correct, "process_verdict": "incorrect",
                 "errors": [{"location": "step", "error": "bad", "correction": "fix"}]}
        result = classify(episode, wrong)
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["failure_types"], ["process_error"])
        truncated = {"outcome": {**episode["outcome"], "truncated": True}}
        self.assertEqual(classify(truncated, None)["status"], "truncated")

    def test_contrasts_prefer_greedy_failure_and_preserve_distinct_failure_modes(self):
        reviews = []
        for index, (status, failures) in enumerate((("failure", ["answer_error"]),
                ("success", []), ("failure", ["format_error"]), ("failure", ["process_error"]))):
            episode = {"episode_id": f"ep-{index}", "input_id": "input", "index": index,
                       "generation": {"token_count": 10 + index}}
            key = f"episodes/ep-{index}"
            self.store.put(key, episode, {"i": index})
            reviews.append({"episode_id": f"ep-{index}", "episode_key": key,
                            "status": status, "failure_types": failures,
                            "semantic_review": {"fixture": True}})
        pairs = choose_contrasts(reviews, self.store, 3)
        self.assertEqual([kind for _, _, kind in pairs],
                         ["answer_error", "process_error", "format_error"])
        self.assertEqual(pairs[0][1]["episode_id"], "ep-0")

    def test_atom_and_card_validation_are_structural_not_keyword_blacklists(self):
        value = atom(1)
        answer = {"transferable": True, "reason": "supported", "confidence": .9,
                  **{key: value[key] for key in ("applicability", "experience",
                                                 "exclusions_from_input", "failure_mechanism")}}
        validate_atom(answer)
        card = FixtureTeacher().ask("card", {}, validate_card)
        # Source-style numbers/entities do not trigger a static rejection in V5.
        card["memory_payload"]["rationale"] = "A ratio such as 2:3 preserves one scale."
        validate_card(card)

    def test_group_to_primary_card_owns_coverage_and_separates_selector_key(self):
        values = [atom(index) for index in range(3)]
        keys = []
        for value in values:
            key = "atoms/" + value["atom_id"]
            self.store.put(key, value, {"atom": value["atom_id"]})
            keys.append(key)
        self.store.put("stages/atoms", {"keys": keys, "counts": {"accepted": 3}}, {"fixture": 1})
        with redirect_stdout(io.StringIO()):
            groups = run_grouping(self.store, FixtureTeacher(), self.config,
                                  lambda value: [1., 0.], {"source": "fixture", "revision": "fixed"})
            cards = run_cards(self.store, FixtureTeacher(), self.config)
        self.assertEqual(groups["group_count"], 1)
        self.assertEqual(groups["groups"][0]["distinct_input_count"], 3)
        self.assertEqual(cards["tier_counts"], {"primary": 1})
        record = self.store.require(cards["keys"][0])
        self.assertTrue(record["bank_id"].startswith("v5-bank-"))
        self.assertNotIn("quantities are independent", record["selector_text"])
        self.assertIn("quantities are independent", record["descriptor"])
        self.assertIn("Recommended method", render_card({"selector_key": record["selector_key"],
                                                          "memory_payload": record["memory_payload"]}))

    def test_selector_uses_positive_key_then_separate_exclusions_and_utility_abstention(self):
        entry = {"bank_id": "v5-bank-a", "positive_selector_text": "shared ratio",
                 "selector_key": {"exclusions_from_input": ["independent quantities"]},
                 "distinct_input_count": 3}
        document = reranker_document(entry)
        self.assertTrue(document.startswith("shared ratio"))
        self.assertIn("Excluded when", document)
        rows = []
        for index in range(10):
            ranked = [(.9, "v5-bank-a", entry)]
            candidate = candidate_features(ranked, {"v5-bank-a": {"score": .9}})[0]
            candidate["utility"] = 1 if index < 8 else -1
            rows.append({"input_id": str(index), "baseline_reward": 0 if index < 8 else 1,
                         "candidates": [candidate]})
        policy = fit_policy(rows, ridge=1., folds=5, profile_sha256="fixture")
        self.assertEqual(policy["feature_names"], list(FEATURE_NAMES))
        decision = select(rows[0]["candidates"], policy)
        self.assertIn(decision["bank_id"], {"v5-bank-a", "no_memory"})
        self.assertIn("oof_metrics", policy)

    def test_primary_requires_three_inputs_and_no_protocol_fallback(self):
        values = [atom(index) for index in range(2)]
        keys = []
        for value in values:
            key = "atoms/" + value["atom_id"]
            self.store.put(key, value, {"atom": value["atom_id"]})
            keys.append(key)
        self.store.put("stages/atoms", {"keys": keys, "counts": {"accepted": 2}}, {"fixture": 2})
        with redirect_stdout(io.StringIO()):
            run_grouping(self.store, FixtureTeacher(), self.config,
                         lambda value: [1., 0.], {"source": "fixture", "revision": "fixed"})
            cards = run_cards(self.store, FixtureTeacher(), self.config)
        self.assertEqual(cards["tier_counts"], {"conditional": 1})

    def test_pair_judgments_are_mapped_by_short_id_not_response_order(self):
        groups = [{"group_id": f"g{i}", "shared_method": f"method {i}",
                   "applicability": "condition", "exclusions": []} for i in range(3)]
        class ReorderedTeacher:
            def ask(self, task, payload, validate):
                value = {"judgments": [
                    {"pair_id": "P02", "relation": "same_method",
                     "applicability_compatible": True, "exclusion_conflict": False,
                     "reason": "same"},
                    {"pair_id": "P01", "relation": "different",
                     "applicability_compatible": False, "exclusion_conflict": False,
                     "reason": "different"}]}
                validate(value)
                return value
        accepted, failures = judge_pairs(self.store, ReorderedTeacher(), groups,
                                         [(0, 1), (0, 2)], 2)
        self.assertEqual(accepted, [(0, 2)])
        self.assertEqual(failures, 0)

    def test_top_level_shell_dispatches_v5_plan_without_inference(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["bash", str(root / "test.sh"), "v5", "--plan-only",
            "--config", str(root / "configs/experiments/gsm8k/v5.json"),
            "--output-dir", str(Path(self.tmp.name) / "unused")], cwd=root,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["invariants"]["selector_query"], "input-only")
        self.assertFalse(plan["invariants"]["exclusions_embedded"])


if __name__ == "__main__":
    unittest.main()
