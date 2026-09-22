"""Short-ID compatibility recovery for interrupted card construction runs."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from memgen.experience.bank_construction.artifacts import Store
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.schemas import validate_card
from scripts.repair_local_bank_cards import (
    alias_evidence, construct_one, map_support_ids, run_alias_cards,
)


def signature():
    return {"transferable": True, "reason": "Grounded remaining-quantity repair",
            "experience_kind": "reasoning",
            "problem_structure": "Sequential changes in a remaining quantity",
            "decision_point": "Choose the ratio base",
            "failure_mechanism": "Uses the initial amount instead of the remainder",
            "repair_operator": "Update the remaining amount",
            "verification_operator": "Check conservation", "exclusion_conditions": [],
            "substitution_check": "Names and quantities may change", "near_miss": "Fixed base",
            "source_grounding": "Failure uses the wrong ratio base"}


class FixtureTeacher:
    def __init__(self):
        self.calls = []

    def ask(self, task, payload, validate):
        self.calls.append((task, payload))
        ids = [value["evidence_id"] for value in payload["evidence"]]
        if task == "card":
            value = {"coherent": True, "reason": "Common repair", "support_ids": ids,
                "card": {"problem_structure": "Sequential changes", "decision_point": "Select ratio base",
                    "applies_when": "The ratio refers to the current remainder",
                    "procedure": ["Update remainder before the next ratio"],
                    "avoid": ["Do not reuse the initial amount"],
                    "exclusions": ["Each ratio uses the original amount"],
                    "verify": ["Check conservation"]},
                "substitution_check": "Preserves relations", "near_miss": "Original-base ratios"}
        elif task == "card_review":
            value = {"quality_tier": "primary", "reason": "Grounded with boundaries",
                     "issues": [], "support_ids": ids}
        else:
            raise AssertionError(task)
        validate(value)
        return value


def group(group_id, members):
    return {"group_id": group_id, "members": members,
            "method": "Update the remaining quantity",
            "applies_when": "A later ratio applies to the current remainder",
            "exclusions": ["All ratios share the initial base"],
            "rationale": "Same executable repair", "distinct_sample_count": len(members)}


class CardRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name), {"fixture": "card-repair"})
        self.config = replace(ConstructionConfig(), group_batch_size=2)
        self.ids = ["evidence-" + char * 64 for char in "abc"]
        keys = []
        for index, evidence_id in enumerate(self.ids):
            key = f"evidence/{evidence_id}"
            self.store.put(key, {"evidence_id": evidence_id, "sample_id": f"sample-{index}",
                "signature": signature()}, {"index": index})
            keys.append(key)
        self.store.put("stages/evidence", {"keys": keys, "counts": {"accepted": 3}}, {"fixture": True})
        self.groups = {"groups": [group("group-a", self.ids[:2]), group("group-b", self.ids[2:])],
                       "evidence_count": 3}
        self.store.put("stages/groups", self.groups,
                       {"evidence": "fixture", "config": self.config.to_dict()})

    def test_alias_answer_maps_back_to_exact_original_ids(self):
        records = [self.store.require(f"evidence/{value}") for value in self.ids[:2]]
        short, aliases, mapping = alias_evidence(records)
        self.assertEqual(aliases, ["E00", "E01"])
        self.assertEqual([value["evidence_id"] for value in short], aliases)
        answer = {"coherent": False, "reason": "not one method", "support_ids": aliases}
        mapped = map_support_ids(answer, aliases, mapping, validate_card)
        self.assertEqual(mapped["support_ids"], self.ids[:2])

    def test_complete_stage_uses_short_ids_and_is_idempotent(self):
        teacher = FixtureTeacher()
        result = run_alias_cards(self.store, teacher, self.config)
        self.assertEqual(result["tier_counts"], {"primary": 2})
        for task, payload in teacher.calls:
            if task in {"card", "card_review"}:
                self.assertTrue(all(value["evidence_id"].startswith("E")
                                    for value in payload["evidence"]))
        calls = len(teacher.calls)
        self.assertEqual(run_alias_cards(self.store, teacher, self.config), result)
        self.assertEqual(len(teacher.calls), calls)

    def test_card_failure_becomes_reject_and_does_not_abort_other_groups(self):
        class FirstFails(FixtureTeacher):
            def ask(self, task, payload, validate):
                if task == "card" and payload["evidence"][0]["evidence_id"] == "E00" and not self.calls:
                    self.calls.append((task, payload))
                    raise RuntimeError("fixture exhausted retries")
                return super().ask(task, payload, validate)

        result = run_alias_cards(self.store, FirstFails(), self.config)
        self.assertEqual(result["tier_counts"], {"reject": 1, "primary": 1})
        record = self.store.require("cards/group-a")
        self.assertIsNone(record["card"])
        self.assertTrue(any((self.store.root / "cards/recovery/fallback/card").glob("*.json")))

    def test_review_failure_cannot_produce_primary(self):
        class ReviewFails(FixtureTeacher):
            def ask(self, task, payload, validate):
                if task == "card_review":
                    raise RuntimeError("fixture exhausted retries")
                return super().ask(task, payload, validate)

        source = {value: self.store.require(f"evidence/{value}") for value in self.ids}
        record, _ = construct_one(self.store, ReviewFails(), self.config,
                                  self.groups["groups"][0], source)
        self.assertEqual(record["quality_tier"], "conditional")
        self.assertIsNotNone(record["card"])


if __name__ == "__main__":
    unittest.main()
