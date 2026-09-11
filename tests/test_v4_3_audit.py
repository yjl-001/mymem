from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from memgen.experience import v4_3_bank as bank
from memgen.experience.v4_3_audit import (
    LAYERS, aggregate, bind_source_state, build_plan, build_wrong_controls, metrics, sweep_metrics, validate_result,
)
from scripts.audit_v4_3_unified_memory import score_branch
from tests.test_v4_3_bank import fixture


def audit_fixture():
    source = fixture()
    outputs = bank.build_outputs(**source)
    candidates = outputs["candidate_bank_records.jsonl"]
    events = []
    for r in candidates:
        for eid, sid in zip(r["construction"]["experience_ids"], r["construction"]["sample_ids"]):
            common = {"bank_id": r["source_v42_bank_id"], "experience_id": eid, "sample_id": sid}
            events.append(bank.seal({**common, "event_id": eid + "::prompt", "event_kind": "prompt_semantic",
                "prompt_token_count": 2, "prompt_token_ids_sha256": bank.canonical_hash([7, 8]),
                "failure_gate_eligible": True, "failure_gate_rejection_reason": None}))
            for kind in ("failure_gate_attempt", "success_gate_attempt"):
                events.append(bank.seal({**common, "event_id": eid + "::" + kind, "event_kind": kind,
                    "attempt_number": 1, "token_position": 3, "online_reachable_safety_negative": kind == "success_gate_attempt",
                    "prefix_alignment": {"prefix_token_ids_sha256": bank.canonical_hash([7, 8, 9, 10])}}))
    controls = build_wrong_controls(candidates, {r["bank_id"]: 100 for r in candidates})
    binding = bank.seal({"fixture": True}, "binding_sha256")
    return source, outputs, events, controls, binding


def result_for(case, profile="profile", gain=False):
    branches = {}
    for name, bid in case["memories"].items():
        visible = case["audit_layer"] == "visible_content"
        latent = bid is not None and not visible
        branches[name] = {"condition_memory_id": bid, "memory_id": bid if latent else None,
            "maximum_completion_tokens": 1024, "active_step_count": int(latent),
            "prefix_completion_token_count": case["prefix_token_count"] - case["prompt_token_count"],
            "continuation_token_ids": [4], "local_continuation_token_ids": [4],
            "continuation_token_ids_sha256": bank.canonical_hash([4]), "stop_reason": "eos",
            "strict_reward": float(gain and name == "matched"), "first_step_logits_kl": 0.1 if latent else 0,
            "format_valid": True, "local_intervention_evaluation": {"strict_reward": 0., "format_valid": False},
            "first_step_top1_changed": False,
            "attention_traces": [{"memory_id": bid, "memory_attention_mass": .1, "native_attention_mass": .9,
                "native_key_length": case["prefix_token_count"], "canonical_rope_score_relative_error": 0.0}] if latent else []}
    return bank.seal({"schema_version": "memgen-v4.3-audit-result-v1", "profile_sha256": profile,
        **{k: case[k] for k in ("case_id", "case_sha256", "audit_layer", "sample_id", "experience_id", "bank_id", "quality_tier", "semantic_category", "gate_attempt_index")},
        "branches": branches, "prefix_cache_parity": {"all_branches_share_exact_prefix": case["exact_native_prefix_cache"],
            "initial_cache_tensors_exactly_equal": True, "branch_cache_storage_is_independent": True,
            "cache_length_parity": True, "prefix_token_ids_sha256": case["prefix_token_ids_sha256"]}})


class V43AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.outputs, cls.events, cls.controls, cls.binding = audit_fixture()

    def plan(self, **kwargs):
        return build_plan(candidates=self.outputs["candidate_bank_records.jsonl"], events=self.events,
                          controls=self.controls, binding=self.binding, mode=kwargs.pop("mode", "smoke"), **kwargs)

    def test_wrong_banks_deterministic_distinct_and_sample_disjoint(self):
        records = self.outputs["candidate_bank_records.jsonl"]
        reverse = build_wrong_controls(list(reversed(records)), {r["bank_id"]: 100 for r in records})
        self.assertEqual(self.controls, reverse)
        by_id = {r["bank_id"]: r for r in records}
        for bid, control in self.controls["controls"].items():
            self.assertTrue(control["qualified"])
            self.assertEqual(len({control[name] for name in ("matched", "near_wrong", "far_wrong")}), 3)
            self.assertNotEqual(by_id[bid]["semantic_category"], by_id[control["far_wrong"]]["semantic_category"])
            for name in ("near_wrong", "far_wrong"):
                self.assertFalse(set(by_id[bid]["construction"]["sample_ids"]) & set(by_id[control[name]]["construction"]["sample_ids"]))

    def test_missing_wrong_control_is_explicit_failure(self):
        records = self.outputs["candidate_bank_records.jsonl"][:2]
        controls = build_wrong_controls(records, {r["bank_id"]: 5 for r in records})
        self.assertTrue(all(not c["qualified"] for c in controls["controls"].values()))

    def test_smoke_covers_two_tiers_four_layers_and_full_horizon(self):
        plan = self.plan()
        self.assertEqual(len(plan["cases"]), 16)
        self.assertEqual({(c["quality_tier"], c["audit_layer"]) for c in plan["cases"]},
                         {(t, layer) for t in ("primary", "conditional") for layer in LAYERS})
        self.assertEqual(plan["configuration"]["maximum_completion_tokens"], 1024)
        self.assertEqual(plan["configuration"]["maximum_active_steps"], 32)

    def test_full_covers_all_samples_at_prompt_end(self):
        plan = self.plan(mode="full")
        for layer in LAYERS:
            self.assertEqual(len({c["sample_id"] for c in plan["cases"] if c["audit_layer"] == layer}), 116)

    def test_primary_scope_filters_cases_controls_and_report_denominators(self):
        from scripts.audit_v4_3_unified_memory import smoke_integrity
        candidates = self.outputs["candidate_bank_records.jsonl"]
        records = [r for r in candidates if r["quality_tier"] == "primary"]
        ids = {r["bank_id"] for r in records}
        controls = build_wrong_controls(records, {bid: 100 for bid in ids})
        for mode in ("smoke", "full"):
            plan = build_plan(candidates=candidates, events=self.events, controls=controls,
                              binding=self.binding, mode=mode, bank_scope="primary", all_bank_sweep=True)
            self.assertEqual(plan["source_sample_count"], 76)
            self.assertEqual(plan["source_cache_sample_count"], 116)
            self.assertEqual(plan["out_of_scope_sample_count"], 40)
            self.assertEqual(plan["excluded_samples"], [])
            self.assertTrue(all(c["quality_tier"] == "primary" and
                                set(c["memories"].values()) - {None} <= ids for c in plan["cases"]))
            rows = [result_for(c) for c in plan["cases"]]
            report = aggregate(plan, rows, "profile")
            self.assertEqual(report["source_sample_count"], 76)
            self.assertEqual(report["bank_scope"], "primary")
            self.assertTrue(report["complete"])
            if mode == "smoke":
                self.assertTrue(smoke_integrity(plan, rows))
        with self.assertRaisesRegex(ValueError, "selected Bank scope"):
            self.plan(bank_scope="primary")  # A two-tier wrong pool cannot leak in.

    def test_audit_cli_defaults_to_primary(self):
        from unittest.mock import patch
        from scripts.audit_v4_3_unified_memory import parse_args
        argv = ["audit"]
        for flag in ("bank-dir", "side-kv-dir", "semantic-packets", "cache-manifest", "token-risk-artifact", "output-dir"):
            argv.extend(["--" + flag, "/tmp/fixture"])
        with patch("sys.argv", argv):
            self.assertEqual(parse_args().bank_scope, "primary")

    def test_unreachable_failure_not_in_gate_denominator(self):
        events = deepcopy(self.events)
        sid = events[0]["sample_id"]
        events = [e for e in events if not (e["sample_id"] == sid and e["event_kind"] == "failure_gate_attempt")]
        events[0]["failure_gate_eligible"] = False
        events[0]["failure_gate_rejection_reason"] = "failure_has_no_joint_gate"
        plan = build_plan(candidates=self.outputs["candidate_bank_records.jsonl"], events=events,
                          controls=self.controls, binding=self.binding, mode="full")
        self.assertEqual(len(plan["gate_unreachable_failures"]), 1)
        self.assertFalse(any(c["sample_id"] == sid and c["audit_layer"] == "exact_gate_failure" for c in plan["cases"]))
        self.assertTrue(any(c["sample_id"] == sid and c["audit_layer"] == "prompt_end_latent" for c in plan["cases"]))

    def test_artificial_success_state_rejected(self):
        events = deepcopy(self.events)
        next(e for e in events if e["event_kind"] == "success_gate_attempt")["online_reachable_safety_negative"] = False
        with self.assertRaisesRegex(ValueError, "actual gate"):
            build_plan(candidates=self.outputs["candidate_bank_records.jsonl"], events=events,
                       controls=self.controls, binding=self.binding, mode="full")

    def test_result_authentication_and_attention_mass_integrity(self):
        case = next(c for c in self.plan()["cases"] if c["audit_layer"] == "exact_gate_failure")
        row = result_for(case)
        validate_result(row, case, "profile")
        row["branches"]["matched"]["attention_traces"][0]["memory_attention_mass"] = 0
        with self.assertRaisesRegex(ValueError, "attention integrity"):
            validate_result(bank.seal(row), case, "profile")

    def test_visible_cannot_claim_exact_cache_parity(self):
        case = next(c for c in self.plan()["cases"] if c["audit_layer"] == "visible_content")
        row = result_for(case)
        row["prefix_cache_parity"]["all_branches_share_exact_prefix"] = True
        with self.assertRaisesRegex(ValueError, "Visible"):
            validate_result(bank.seal(row), case, "profile")

    def test_aggregation_keeps_layers_and_tiers_separate(self):
        plan = self.plan()
        rows = [result_for(c, gain=True) for c in plan["cases"]]
        report = aggregate(plan, rows, "profile")
        self.assertTrue(report["complete"])
        self.assertFalse(report["qualified_for_online_use"])
        for value in report["by_audit_layer"].values():
            self.assertEqual(set(value["by_quality_tier"]), {"primary", "conditional"})
            self.assertEqual(value["overall"]["branches"]["matched"]["accuracy"], 1.0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate(plan, rows + rows[:1], "profile")

    def test_sample_macro_does_not_count_gate_attempts_as_new_samples(self):
        case = self.plan()["cases"][0]
        good = result_for(case, gain=True)
        bad = deepcopy(result_for(case))
        bad["sample_id"] = "another-sample"
        summary = metrics([good, good, good, bad])
        self.assertEqual(summary["branches"]["matched"]["accuracy"], .75)
        self.assertEqual(summary["branches"]["matched"]["independent_sample_macro_accuracy"], .5)
        self.assertEqual(summary["branches"]["matched"]["gain_independent_sample_count"], 1)

    def test_all_bank_sweep_is_separate_outcome_informed_ceiling(self):
        plan = self.plan(all_bank_sweep=True)
        cases = [c for c in plan["cases"] if c["audit_layer"] == "all_bank_sweep"]
        self.assertTrue(cases)
        self.assertTrue(all(len(c["memories"]) == 12 for c in cases))
        summary = sweep_metrics([result_for(c) for c in cases])
        self.assertTrue(summary["outcome_information_used"])
        self.assertFalse(summary["online_accuracy_claim"])

    def test_raw_gsm8k_official_solution_is_converted_for_strict_scoring(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "The result is \\boxed{37}")
        branch = {"continuation_token_ids": [3], "local_continuation_token_ids": [3]}
        scored = score_branch(tokenizer, [1, 2], 2, branch, "Compute the amount.\n#### 37")
        self.assertEqual(scored["strict_reward"], 1.0)
        self.assertEqual(scored["local_intervention_evaluation"]["strict_reward"], 1.0)

    def test_binding_authenticates_real_independent_id_and_stripped_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet_path, risk_path, cache_path = root / "packets", root / "risk", root / "cache"
            packet_path.write_text("packet bytes")
            risk_path.write_bytes(b"risk bytes")
            cache_path.write_text("cache bytes")
            out = deepcopy(self.outputs)
            hashes = {"packets": bank.file_hash(packet_path), "records": "a" * 64, "manifest": "b" * 64}
            out["bundle"] = {"inputs": {"file_sha256": hashes}, "manifest_sha256": "c" * 64}
            evidence = {e["evidence_id"]: deepcopy(e) for p in self.source["packets"] for e in p["evidence"]}
            by_old = {r["source_v42_bank_id"]: r for r in out["candidate_bank_records.jsonl"]}
            events = deepcopy(self.events)
            for event in events:
                e = evidence[event["experience_id"]]
                event.update(independent_sample_id=bank.canonical_hash({"benchmark": "openai/gsm8k", "logical_split": "bank-source", "sample_id": e["sample_id"]}),
                    logical_split="bank-source", dataset_split="train", question_sha256=bank.text_hash(e["question"].strip()),
                    bank_record_sha256=by_old[event["bank_id"]]["curation_provenance"]["source_curated_record_sha256"],
                    completion_hashes={"verified_success_completion_sha256": bank.text_hash(e["verified_success_trajectory"].strip()),
                                       "verified_failure_completion_sha256": bank.text_hash(e["verified_failure_trajectory"].strip())})
            for e in evidence.values():
                e["verified_success_trajectory"] += "\n"
            cache = SimpleNamespace(manifest_path=cache_path, events=events, manifest={
                "manifest_sha256": "d" * 64, "reasoner": {}, "provenance": {"inputs": {
                    "bank_records_sha256": hashes["records"], "bank_manifest_file_sha256": hashes["manifest"],
                    "token_risk_artifact_sha256": bank.file_hash(risk_path)}}})
            binding = bind_source_state(cache=cache, bank=out, evidence=evidence, semantic_packets_path=packet_path, risk_path=risk_path)
            self.assertFalse(binding["source_manifest_rewritten"])
            self.assertEqual(len(binding["event_bindings"]), len(events))
            events[0]["independent_sample_id"] = events[0]["sample_id"]
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                bind_source_state(cache=cache, bank=out, evidence=evidence, semantic_packets_path=packet_path, risk_path=risk_path)


if __name__ == "__main__":
    unittest.main()
