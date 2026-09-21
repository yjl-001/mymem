"""Compatibility repair for partition responses that corrupt long IDs."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.bank_construction.artifacts import Store, digest
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.prompts import VERSION, messages
from memgen.experience.bank_construction.schemas import validate_partition
from memgen.experience.bank_construction.teacher import Teacher
from scripts.repair_local_bank_partitions import (
    alias_request, coverage_error, mapped_answer, repair_one, run_scalable_grouping, unresolved_requests,
)
from tests.test_local_bank_construction import FixtureTeacher, signature


class Adapter:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def chat(self, conversation, *, seed):
        self.calls.append((conversation, seed))
        return {"text": json.dumps(next(self.answers)), "truncated": False,
                "token_count": 1, "prompt_token_count": 1, "stop_reason": "stop", "seed": seed}


def group(members):
    return {"members": members, "method": "Update the remaining quantity",
            "applies_when": "A later ratio applies to the current remainder",
            "exclusions": ["All ratios share the initial base"], "rationale": "Same executable repair"}


class PartitionRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name), {"fixture": "partition-repair"})
        self.ids = ["evidence-" + char * 64 for char in "abc"]
        self.payload = {"context": "Initial within-batch method grouping", "evidence": [
            {"evidence_id": value, "signature": {"repair_operator": "update",
                "problem_structure": "sequential remainder", "exclusion_conditions": []}} for value in self.ids]}
        self.request = {"prompt_version": VERSION, "task": "partition", "messages": messages("partition", self.payload)}
        self.key = "teacher/partition/" + digest(self.request)
        self.store.put(self.key + "-request", self.request, self.request)

    def test_alias_mapping_and_detailed_coverage(self):
        repair, aliases, ids = alias_request(self.key, self.request)
        self.assertEqual(aliases, ["E00", "E01", "E02"])
        self.assertEqual(ids, self.ids)
        shown = json.loads(repair["messages"][-1]["content"])
        self.assertEqual([row["evidence_id"] for row in shown["evidence"]], aliases)
        answer = {"groups": [group(["E00", "E02"]), group(["E01"])]}
        mapped = mapped_answer(answer, aliases, ids)
        validate_partition(mapped, self.ids)
        self.assertIn("missing=['E02']", coverage_error({"groups": [group(["E00", "E01", "X"]) ]}, aliases))

    def test_repair_preserves_receipts_and_satisfies_original_request(self):
        bad = {"groups": [group(["E00", "E01", "E01"])]}
        good = {"groups": [group(["E00", "E02"]), group(["E01"])]}
        adapter = Adapter([bad, good])
        config = replace(ConstructionConfig(), teacher_backend="vllm")
        self.assertEqual(len(unresolved_requests(self.store)), 1)
        repair_one(self.store, config, adapter, self.key, self.request)
        accepted = self.store.get(self.key, self.request)
        validate_partition(accepted["answer"], self.ids)
        self.assertEqual(accepted["acceptance_mode"], "short_alias_partition_repair")
        self.assertIsNotNone(self.store.require(accepted["repair_key"]))
        self.assertIsNotNone(self.store.require(accepted["raw_key"]))
        self.assertEqual(unresolved_requests(self.store), [])
        cached = Teacher(self.store, config, lambda: self.fail("Mapped answer must satisfy the original cached request"))
        self.assertEqual(cached.ask("partition", self.payload,
                         lambda value: validate_partition(value, self.ids)), accepted["answer"])
        repair_one(self.store, config, adapter, self.key, self.request)
        self.assertEqual(len(adapter.calls), 2)

    def test_exhaustion_falls_back_to_audited_singletons(self):
        bad = {"groups": [group(["E00", "E00"])]}
        adapter = Adapter([bad])
        config = replace(ConstructionConfig(), teacher_backend="vllm", teacher_retries=0)
        repair_one(self.store, config, adapter, self.key, self.request)
        accepted = self.store.get(self.key, self.request)
        self.assertEqual(accepted["acceptance_mode"], "singleton_fallback")
        self.assertEqual([group["members"] for group in accepted["answer"]["groups"]],
                         [[value] for value in self.ids])
        fallback = self.store.require(accepted["raw_key"])
        self.assertEqual(fallback["failed_attempt_count"], 1)

    def test_scalable_grouping_uses_candidates_but_teacher_decides_merge(self):
        config = replace(ConstructionConfig(), group_batch_size=2, candidate_batch_size=1)
        keys = []
        for index in range(4):
            key = f"evidence/e{index}"
            self.store.put(key, {"evidence_id": f"e{index}", "sample_id": f"s{index}",
                "signature": signature()}, {"index": index})
            keys.append(key)
        self.store.put("stages/evidence", {"keys": keys, "counts": {"accepted": 4}}, {"fixture": True})
        teacher = FixtureTeacher()
        result = run_scalable_grouping(self.store, teacher, config, lambda text: [1., 0.],
                                       {"source": "fixture", "revision": "fixed"},
                                       candidate_top_k=2, consolidation_rounds=2)
        self.assertEqual(result["evidence_count"], 4)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(set(result["groups"][0]["members"]), set(keys_value.split("/")[-1] for keys_value in keys))
        self.assertFalse(result["grouping_strategy"]["candidate_generation_is_final_decision"])
        self.assertTrue(any(task == "match" for task, _ in teacher.calls))
        self.assertTrue(any(task == "membership" for task, _ in teacher.calls))
        cached = run_scalable_grouping(self.store, self.fail, config, self.fail,
                                       {"source": "fixture", "revision": "fixed"},
                                       candidate_top_k=2, consolidation_rounds=2)
        self.assertEqual(cached, result)


if __name__ == "__main__":
    unittest.main()
