from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from memgen.experience import v4_3_bank as bank


ROOT = Path(__file__).resolve().parents[1]


def fixture() -> dict:
    """Full-count synthetic input, using the actual 24-decision curation policy."""
    policy = json.loads((ROOT / "configs/experiments/gsm8k/v4_2_local_curation_policy.json").read_text())
    decisions = [d for d in policy["decisions"] if d["decision"] in {"primary", "conditional"}]
    records, packets = [], []
    for i, decision in enumerate(decisions):
        evidence = []
        for j in range(6 if i < 3 else 7):
            question = f"A warehouse tracks parcels for shipment batch {i * 10 + j}. What quantity is requested?"
            signature = {
                "problem_structure": "The problem relates a quantity to a rate over a stated duration.",
                "decision_point": "Determine the duration associated with the requested quantity.",
                "repair_operator": "Multiply the rate by the corresponding duration to obtain the quantity.",
                "failure_mechanism": "The computation uses a duration associated with a different quantity.",
                "verification_operator": "Verify that the rate and duration units combine into the requested quantity unit.",
            }
            evidence.append({
                "evidence_id": f"experience-{i:02d}-{j}",
                "sample_id": f"gsm8k-train-{i * 10 + j}-{bank.text_hash(question)[:12]}",
                "semantic_signature": signature, "source_experience_type": "reasoning_failure",
                "question": question, "official_solution": "Compute the shipment count. \\boxed{37}",
                "verified_success_trajectory": "The shipment contains the required parcels. \\boxed{37}",
                "verified_failure_trajectory": "The shipment count was miscomputed. \\boxed{38}",
                "source_signature_sha256": bank.canonical_hash({"signature": signature, "i": i, "j": j}),
                "source_provenance_sha256": bank.canonical_hash({"provenance": i * 10 + j}),
                "construction_input_sha256": bank.canonical_hash({"input": i * 10 + j}),
                "target_verifier": {"reward": 1}, "reference_verifier": {"reward": 0},
            })
        packet = bank.seal({"schema_version": bank.PACKET_SCHEMA, "candidate_id": f"candidate-{i}",
                            "evidence_count": len(evidence), "evidence": evidence}, "packet_sha256")
        packets.append(packet)
        record = bank.seal({
            "schema_version": "memgen-v4-bank-record-v1", "construction_version": "v4.2-local-curated",
            "bank_id": decision["bank_id"], "benchmark": "openai/gsm8k",
            "cluster": {"source_candidate_id": packet["candidate_id"]},
            "construction": {"experience_ids": [e["evidence_id"] for e in evidence],
                             "sample_ids": [e["sample_id"] for e in evidence], "distinct_sample_count": len(evidence),
                             "evidence_packet_sha256": packet["packet_sha256"],
                             "source_signature_sha256": {e["evidence_id"]: e["source_signature_sha256"] for e in evidence}},
            "curation": {k: decision[k] for k in ("decision", "reason", "semantic_category")}
                        | {"policy_sha256": bank.canonical_hash(policy), "source_record_sha256": "a" * 64},
            "process_card": {"target": "legacy content must never enter the descriptor", "reference": "legacy failure"},
        })
        records.append(record)
    inputs = {k: bank.text_hash(k) for k in ("records", "manifest", "packets", "policy")}
    ids = [r["bank_id"] for r in records]
    manifest = bank.seal({
        "schema_version": bank.CURATED_SCHEMA, "construction_version": "v4.2-local-curated",
        "record_count": 17, "evidence_count": 116, "bank_ids": ids,
        "record_order_sha256": bank.canonical_hash(ids), "record_sha256": {r["bank_id"]: r["record_sha256"] for r in records},
        "inputs": {"semantic_preflight": {"evidence_packet_file_sha256": inputs["packets"]}},
        "curation": {"policy_sha256": bank.canonical_hash(policy), "policy_file_sha256": inputs["policy"], "retained_bank_ids": ids},
        "source_local_direct": {"manifest_logical_sha256": policy["source_manifest_sha256"],
                                "profile_sha256": policy["source_profile_sha256"],
                                "record_order_sha256": policy["source_record_order_sha256"]},
    }, "manifest_sha256")
    return dict(records=records, manifest=manifest, packets=packets, policy=policy,
                input_hashes=inputs, implementation_hashes={"fixture": "a" * 64})


def rebind(data: dict, packet_index: int = 0) -> None:
    """Simulate an internally re-sealed upstream artifact for negative tests."""
    p = data["packets"][packet_index] = bank.seal(data["packets"][packet_index], "packet_sha256")
    r = data["records"][packet_index]
    r["construction"]["evidence_packet_sha256"] = p["packet_sha256"]
    r = data["records"][packet_index] = bank.seal(r)
    data["manifest"]["record_sha256"][r["bank_id"]] = r["record_sha256"]
    data["manifest"] = bank.seal(data["manifest"], "manifest_sha256")


class V43BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = fixture()
        cls.outputs = bank.build_outputs(**cls.source)

    def test_full_17_bank_116_evidence_contract_and_separation(self):
        out = self.outputs
        self.assertEqual(len(out["candidate_bank_records.jsonl"]), 17)
        self.assertEqual(out["construction_report.json"]["consumed_evidence_count"], 116)
        self.assertEqual(out["construction_report.json"]["qualified_tier_counts"], {"primary": 11, "conditional": 6})
        for tier in ("primary", "conditional"):
            bank.validate_manifest(out[f"{tier}_bank_manifest.json"], out[f"{tier}_bank_records.jsonl"])
            self.assertTrue(all(r["quality_tier"] == tier for r in out[f"{tier}_bank_records.jsonl"]))

    def test_semantic_packet_schema_and_hash(self):
        for mutation, error in ((lambda p: p.update(schema_version="wrong"), "schema"),
                                (lambda p: p["evidence"][0]["semantic_signature"].update(repair_operator="tampered"), "hash")):
            data = deepcopy(self.source)
            mutation(data["packets"][0])
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                bank.build_outputs(**data)

    def test_packet_signature_schema_and_sha_format(self):
        for key, value, error in (("semantic_signature", {}, "signature schema"),
                                  ("source_provenance_sha256", "broken", "SHA256")):
            data = deepcopy(self.source)
            data["packets"][0]["evidence"][0][key] = value
            rebind(data)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
                bank.build_outputs(**data)

    def test_packet_file_and_policy_authentication(self):
        for field, error in (("packets", "packet file hash"), ("policy", "policy hash")):
            data = deepcopy(self.source)
            data["input_hashes"][field] = "b" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                bank.build_outputs(**data)

    def test_duplicate_sample_rejected_before_support(self):
        data = deepcopy(self.source)
        evidence = data["packets"][0]["evidence"]
        evidence[1]["sample_id"] = evidence[0]["sample_id"]
        rebind(data)
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            bank.build_outputs(**data)

    def test_duplicate_evidence_rejected(self):
        data = deepcopy(self.source)
        evidence = data["packets"][0]["evidence"]
        evidence[1]["evidence_id"] = evidence[0]["evidence_id"]
        rebind(data)
        with self.assertRaisesRegex(ValueError, "duplicate evidence"):
            bank.build_outputs(**data)

    def test_sample_question_hash_identity(self):
        data = deepcopy(self.source)
        data["packets"][0]["evidence"][0]["question"] = "A different question."
        rebind(data)
        with self.assertRaisesRegex(ValueError, "sample/question identity"):
            bank.build_outputs(**data)

    def test_evidence_outside_bank_rejected(self):
        data = deepcopy(self.source)
        data["records"][0]["construction"]["experience_ids"][0] = "outside-bank"
        rebind(data)
        with self.assertRaisesRegex(ValueError, "outside Bank"):
            bank.build_outputs(**data)

    def test_source_signature_map_binding(self):
        data = deepcopy(self.source)
        data["packets"][0]["evidence"][0]["source_signature_sha256"] = "b" * 64
        rebind(data)
        with self.assertRaisesRegex(ValueError, "source signature hash"):
            bank.build_outputs(**data)

    def test_bank_coverage_and_record_manifest_hash(self):
        data = deepcopy(self.source)
        data["records"].pop()
        with self.assertRaisesRegex(ValueError, "17-bank"):
            bank.build_outputs(**data)
        data = deepcopy(self.source)
        data["records"][0]["benchmark"] = "other"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            bank.build_outputs(**data)

    def test_tier_cannot_drift(self):
        data = deepcopy(self.source)
        data["records"][0]["curation"]["decision"] = "primary"
        rebind(data)
        with self.assertRaisesRegex(ValueError, "policy decision"):
            bank.build_outputs(**data)

    def test_clause_support_is_distinct_and_complete(self):
        for r in self.outputs["candidate_bank_records.jsonl"]:
            ids = r["construction"]["experience_ids"]
            for clause in r["clause_support"].values():
                self.assertEqual(clause["support_count"], len(ids))
                self.assertEqual(clause["supporting_experience_ids"], ids)
                self.assertEqual(len(clause["supporting_sample_ids"]), len(set(clause["supporting_sample_ids"])))
                self.assertEqual(len(clause["candidate_audit"]), len(ids))
                self.assertEqual(len(clause["support_pair_similarities"]), len(ids) * (len(ids) - 1) // 2)

    def test_selection_and_tie_break_are_order_independent(self):
        evidence = self.source["packets"][0]["evidence"]
        a = bank.select_clause("repair_operator", evidence)
        b = bank.select_clause("repair_operator", list(reversed(evidence)))
        self.assertEqual(a, b)
        self.assertEqual(a["representative_experience_id"], min(e["evidence_id"] for e in evidence))

    def test_each_field_selects_independently_of_medoid(self):
        evidence = deepcopy(self.source["packets"][0]["evidence"])
        evidence[0]["semantic_signature"]["repair_operator"] = "Compute 37 parcels for the final answer."
        repair = bank.select_clause("repair_operator", evidence)
        scope = bank.select_clause("problem_structure", evidence)
        self.assertTrue(repair["qualified"])
        self.assertEqual(repair["support_count"], 5)
        self.assertNotEqual(repair["representative_experience_id"], scope["representative_experience_id"])

    def test_normalization_and_negation_direction_guard(self):
        self.assertEqual(bank.normalize_clause("  Verify   the rate unit ;  "), "Verify the rate unit.")
        self.assertEqual(bank.clause_similarity("Divide the total by each capacity.", "Multiply the total by each capacity."), 0)
        self.assertEqual(bank.clause_similarity("Use the remaining duration.", "Do not use the remaining duration."), 0)
        self.assertEqual(bank.clause_similarity("Use total before remaining.", "Use remaining before total."), 0)

    def test_complete_link_does_not_promote_transitive_similarity(self):
        evidence = deepcopy(self.source["packets"][0]["evidence"])
        words = ["carefully", "explicitly", "systematically", "locally", "independently", "directly"]
        for item, word in zip(evidence, words):
            item["semantic_signature"]["repair_operator"] = f"Compute the required quantity {word}."
        texts = [bank.normalize_clause(e["semantic_signature"]["repair_operator"]) for e in evidence]
        def similarity(a, b):
            i, j = texts.index(a), texts.index(b)
            return 1.0 if (i in {0, 1, 2, 3} and j in {0, 1, 2, 3}) or (i in {2, 3, 4, 5} and j in {2, 3, 4, 5}) else 0.0
        with patch.object(bank, "clause_similarity", side_effect=similarity):
            selected = bank.select_clause("repair_operator", evidence)
        self.assertEqual(selected["support_count"], 4)
        self.assertFalse(selected["qualified"])

    def test_non_executable_repair_is_not_process_qualified(self):
        evidence = deepcopy(self.source["packets"][0]["evidence"])
        for e in evidence:
            e["semantic_signature"]["repair_operator"] = "The situation involves different related quantities."
        result = bank.select_clause("repair_operator", evidence)
        self.assertFalse(result["qualified"])
        self.assertTrue(all("missing_executable_process_operator" in a["leakage_issues"] for a in result["candidate_audit"]))

    def test_render_order_and_absence_of_roles(self):
        r = self.outputs["primary_bank_records.jsonl"][0]
        descriptor = r["descriptor"]
        labels = ["Use when:", "Procedure:", "Avoid:", "Verify:", "Use only when:"]
        self.assertEqual(sorted(descriptor.index(label) for label in labels), [descriptor.index(label) for label in labels])
        self.assertEqual(set(r["unified_process_card"]), set(bank.CARD_FIELDS))
        for field in ("target", "reference", "roles", "process_card"):
            self.assertNotIn(field, r)
            self.assertNotIn(field, descriptor.lower())
        for field in ("question", "official_solution", "verified_success_trajectory", "target_verifier", "reference_verifier"):
            self.assertNotIn('"' + field + '"', json.dumps(r))

    def test_leakage_detection(self):
        e = deepcopy(self.source["packets"][0]["evidence"])
        e[0]["question"] = "Janet has parcels at the depot."
        cases = (("Multiply the rate by 37 units.", "numeric_constant"),
                 ("Multiply the rate by thirty five units.", "numeric_constant"),
                 ("Multiply the rate by ３７ units.", "numeric_constant"),
                 ("Track Janet and her requested parcels.", "source_entity"),
                 ("The final answer is \\boxed{37}.", "answer_or_role_or_reward_language"),
                 ("The final result equals 37.", "answer_fragment"),
                 ("The verified reward confirms this procedure.", "answer_or_role_or_reward_language"))
        for text, issue in cases:
            with self.subTest(text=text):
                self.assertIn(issue, bank.leakage_issues(text, e))

    def test_solution_trace_overlap_rejected(self):
        e = deepcopy(self.source["packets"][0]["evidence"])
        e[0]["verified_success_trajectory"] = "First combine red parcels from the warehouse with blue parcels from the depot."
        self.assertIn("source_text_or_solution_trace_overlap", bank.leakage_issues(e[0]["verified_success_trajectory"], e))

    def test_low_support_is_quarantined_with_no_payload(self):
        data = deepcopy(self.source)
        for item in data["packets"][0]["evidence"][:2]:
            item["semantic_signature"]["repair_operator"] = "Multiply the rate by 37 units."
        rebind(data)
        out = bank.build_outputs(**data)
        self.assertEqual(len(out["quarantined_bank_records.jsonl"]), 1)
        rejected = out["quarantined_bank_records.jsonl"][0]
        self.assertEqual(rejected["clause_support"]["repair_operator"]["support_count"], 4)
        self.assertIsNone(rejected["descriptor"])
        self.assertFalse(rejected["qualified_for_online_use"])
        self.assertNotIn(rejected["bank_id"], out["conditional_bank_manifest.json"]["bank_ids"])
        self.assertEqual(out["construction_report.json"]["consumed_evidence_count"], 116)

    def test_composed_scope_requires_five_joint_samples(self):
        data = deepcopy(self.source)
        evidence = data["packets"][0]["evidence"]
        evidence[0]["semantic_signature"]["problem_structure"] = "Compute 37 units using the source problem."
        evidence[1]["semantic_signature"]["decision_point"] = "Compute 38 units using the source problem."
        rebind(data)
        out = bank.build_outputs(**data)
        r = out["candidate_bank_records.jsonl"][0]
        self.assertTrue(all(s["qualified"] for s in r["clause_support"].values()))
        self.assertEqual(r["composed_field_support"]["applies_when"]["support_count"], 4)
        self.assertFalse(r["qualification"]["construction_qualified"])
        self.assertIsNone(r["descriptor"])

    def test_all_candidates_can_fail_without_relaxing_threshold(self):
        data = deepcopy(self.source)
        for i, packet in enumerate(data["packets"]):
            for e in packet["evidence"]:
                e["semantic_signature"]["repair_operator"] = "The answer is \\boxed{37}."
            rebind(data, i)
        out = bank.build_outputs(**data)
        self.assertEqual(out["construction_report.json"]["quarantined_count"], 17)
        self.assertEqual(out["primary_bank_manifest.json"]["status"], "no_qualified_banks")
        bank.validate_manifest(out["primary_bank_manifest.json"], [])

    def test_lineage_and_content_addressed_identity(self):
        out = self.outputs
        lineage = out["source_v42_to_v43_lineage.json"]
        bank.authenticate(lineage, "lineage_sha256", "lineage")
        self.assertEqual(len(lineage["source_v42_to_v43"]), 17)
        for source, record in zip(self.source["records"], out["candidate_bank_records.jsonl"]):
            self.assertEqual(lineage["source_v42_to_v43"][source["bank_id"]], record["bank_id"])
            self.assertEqual(set(source["construction"]["experience_ids"]), set(record["construction"]["experience_ids"]))
            changed = deepcopy(record)
            changed["unified_process_card"]["procedure"].append("Check the dimensional consistency of the computed result.")
            self.assertNotEqual(bank.content_bank_id(changed), record["bank_id"])

    def test_record_manifest_tampering_and_mixed_tier_rejection(self):
        records = deepcopy(self.outputs["primary_bank_records.jsonl"])
        manifest = deepcopy(self.outputs["primary_bank_manifest.json"])
        records[0]["descriptor"] += " tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            bank.validate_manifest(manifest, records)
        manifest["qualified_for_online_use"] = True
        manifest = bank.seal(manifest, "manifest_sha256")
        with self.assertRaisesRegex(ValueError, "offline/compiler"):
            bank.validate_manifest(manifest, self.outputs["primary_bank_records.jsonl"])

    def test_resealed_descriptor_cannot_bypass_clause_support(self):
        record = deepcopy(self.outputs["primary_bank_records.jsonl"][0])
        record["unified_process_card"]["procedure"] = ["Perform an unrelated calculation using a different procedure."]
        record["descriptor"] = bank.render_card(record["unified_process_card"])
        record["descriptor_sha256"] = bank.text_hash(record["descriptor"])
        record["bank_id"] = bank.content_bank_id(record)
        record = bank.seal(record)
        with self.assertRaisesRegex(ValueError, "supported clauses"):
            bank.validate_record(record)

    def test_resealed_support_threshold_cannot_be_relaxed(self):
        record = deepcopy(self.outputs["primary_bank_records.jsonl"][0])
        record["clause_support"]["repair_operator"]["minimum_pair_similarity"] = 0.1
        record = bank.seal(record)
        with self.assertRaisesRegex(ValueError, "support policy"):
            bank.validate_record(record)

    def test_mixed_tier_resealed_manifest_is_rejected(self):
        manifest = deepcopy(self.outputs["primary_bank_manifest.json"])
        manifest["quality_tier"] = "conditional"
        manifest = bank.seal(manifest, "manifest_sha256")
        with self.assertRaisesRegex(ValueError, "mixed-tier"):
            bank.validate_manifest(manifest, self.outputs["primary_bank_records.jsonl"])

    def test_byte_logical_determinism_and_no_input_mutation(self):
        before = deepcopy(self.source)
        self.assertEqual(self.outputs, bank.build_outputs(**self.source))
        self.assertEqual(self.source, before)

    def test_conditional_boundary_is_curation_not_fake_sample_support(self):
        for r in self.outputs["conditional_bank_records.jsonl"]:
            self.assertFalse(r["boundary_provenance"]["independent_support_claim"])
            self.assertIn(r["boundary_provenance"]["conditional_guard"], r["unified_process_card"]["only_use_when"])
        self.assertFalse(self.outputs["construction_report.json"]["held_out_generalization_claim"])


if __name__ == "__main__":
    unittest.main()
