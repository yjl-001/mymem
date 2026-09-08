"""Mock-provider tests; no paid API calls or real research outcomes."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memgen.experience import v4_3_bank as bank
from memgen.experience import v4_3_deepseek as semantic
from memgen.experience.v4_3_artifacts import load_construction
from scripts import build_v4_3_deepseek_bank as driver
from tests.test_v4_3_bank import fixture, rebind
from tests.test_v4_3_pipeline_contract import write_fixture

CLAUSES = fixture()["packets"][0]["evidence"][0]["semantic_signature"]


def response_for(packet):
    return {"considered_evidence_ids": [e["evidence_id"] for e in packet["evidence"]],
            "clauses": {field: {"text": text, "judgments": [
                {"evidence_id": e["evidence_id"], "supports": True,
                 "quote": e["semantic_signature"][field],
                 "rationale": "Synthetic test judgment: the rate-duration relation supports this clause."}
                for e in packet["evidence"]]} for field, text in CLAUSES.items()}}


class FakeClient:
    def __init__(self, *_):
        self.calls = []
        self.fail_at = None
        self.closed = False

    def generate(self, request, packet):
        if self.fail_at == len(self.calls):
            raise RuntimeError("simulated provider interruption")
        self.calls.append(request)
        self.assert_request(request, packet)
        wire = response_for(packet)
        if request["messages"][0]["content"] == semantic.SYSTEM_PROMPT:
            for clause in wire["clauses"].values():
                for j in clause["judgments"]:
                    del j["quote"]
        return semantic.parse_response(json.dumps(wire), packet), {"http_attempts": 1, "final_response_usage": {"total_tokens": 900}}

    def assert_request(self, request, packet):
        assert request["thinking"] == {"type": "disabled"}
        sent = json.loads(request["messages"][1]["content"])
        assert sent["evidence"] == packet["evidence"]
        assert len(sent["evidence"]) in {6, 7, 8}

    def close(self):
        self.closed = True


def kwargs_for(root, data=None):
    source, packets, policy = write_fixture(root, data)
    return dict(source_dir=source, packets_path=packets, policy_path=policy,
                output_dir=root / "cards", cache_dir=root / "responses")


class DeepSeekConstructionTests(unittest.TestCase):
    def entry(self, data, answer=None):
        source, packet = data["records"][0], data["packets"][0]
        request = semantic.request_spec(source, packet)
        return bank.seal({"request": request, "request_sha256": bank.canonical_hash(request),
                          "response": answer or response_for(packet)}, "cache_sha256")

    def test_numeric_diverse_sources_become_complete_authenticated_cards(self):
        data = fixture()
        for i, packet in enumerate(data["packets"]):
            for j, e in enumerate(packet["evidence"]):
                for field in bank.SIGNATURE_FIELDS:
                    e["semantic_signature"][field] = f"For the {13 + j} parcel example: " + e["semantic_signature"][field]
            rebind(data, i)
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = kwargs_for(Path(tmp), data)
            fake = FakeClient()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-sentinel-not-real"}):
                outputs = driver.construct(**kwargs, client_factory=lambda *_: fake)
            self.assertEqual(len(fake.calls), 17)
            self.assertTrue(fake.closed)
            report = outputs["construction_report.json"]
            self.assertEqual(report["qualified_tier_counts"], {"primary": 11, "conditional": 6})
            self.assertEqual(report["external_api_calls_made"], 17)
            loaded = load_construction(kwargs["output_dir"])
            self.assertEqual(len(loaded["candidate_bank_records.jsonl"]), 17)
            original = {p.name: p.read_bytes() for p in kwargs["output_dir"].iterdir()}
            for flag in ("resume", "validate_only"):
                with patch.object(driver.os.environ, "get", side_effect=AssertionError("Must not read key on cache hit")):
                    driver.construct(**kwargs, **{flag: True}, client_factory=lambda *_: self.fail("Unexpected provider"))
                self.assertEqual(original, {p.name: p.read_bytes() for p in kwargs["output_dir"].iterdir()})
            for path in list(kwargs["output_dir"].iterdir()) + list(kwargs["cache_dir"].iterdir()):
                self.assertNotIn("test-sentinel-not-real", path.read_text())

    def test_partial_resume_calls_only_missing_banks(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = kwargs_for(Path(tmp))
            first = FakeClient()
            first.fail_at = 3
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    driver.construct(**kwargs, client_factory=lambda *_: first)
                self.assertTrue(first.closed)
                second = FakeClient()
                driver.construct(**kwargs, resume=True, client_factory=lambda *_: second)
            self.assertEqual(len(second.calls), 14)

    def test_corrupt_response_blocks_all_key_access_and_future_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = kwargs_for(Path(tmp))
            fake = FakeClient()
            fake.fail_at = 2
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}), self.assertRaises(RuntimeError):
                driver.construct(**kwargs, client_factory=lambda *_: fake)
            path = next(p for p in kwargs["cache_dir"].glob("*.json") if p.name != "profile.json")
            entry = json.loads(path.read_text())
            entry["response"]["clauses"]["repair_operator"]["text"] = "tampered"
            path.write_text(json.dumps(entry))
            with patch.object(driver.os.environ, "get", side_effect=AssertionError("Must authenticate before key read")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    driver.construct(**kwargs, resume=True)

    def test_incomplete_validation_and_profile_drift_do_not_read_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = kwargs_for(Path(tmp))
            fake = FakeClient()
            fake.fail_at = 1
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}), self.assertRaises(RuntimeError):
                driver.construct(**kwargs, client_factory=lambda *_: fake)
            with patch.object(driver.os.environ, "get", side_effect=AssertionError("No key access allowed")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    driver.construct(**kwargs, validate_only=True)
                with self.assertRaisesRegex(ValueError, "profile drift"):
                    driver.construct(**kwargs, resume=True, max_tokens=4096)

    def test_insufficient_support_response_saved_without_regeneration(self):
        class InsufficientClient(FakeClient):
            def generate(self, request, packet):
                response, receipt = super().generate(request, packet)
                for j in response["clauses"]["repair_operator"]["judgments"][4:]:
                    j["supports"] = False
                return response, receipt
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = kwargs_for(Path(tmp))
            fake = InsufficientClient()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}):
                result = driver.construct(**kwargs, client_factory=lambda *_: fake)
            self.assertEqual(len(fake.calls), 17)
            self.assertEqual(result["construction_report.json"]["quarantined_count"], 17)
            with patch.object(driver.os.environ, "get", side_effect=AssertionError("No retry to force support")):
                driver.construct(**kwargs, resume=True)
            load_construction(kwargs["output_dir"])

    def test_schema_citations_and_membership_are_checked(self):
        packet = fixture()["packets"][0]
        for mutation in (
            lambda a: a["clauses"]["repair_operator"]["judgments"][0].update(quote="invented quote"),
            lambda a: a["clauses"]["repair_operator"]["judgments"][0].update(evidence_id="foreign"),
            lambda a: a["clauses"]["repair_operator"]["judgments"][0].update(supports="true"),
            lambda a: a["clauses"]["repair_operator"]["judgments"].pop(),
            lambda a: a["considered_evidence_ids"].append(a["considered_evidence_ids"][0]),
        ):
            answer = response_for(packet)
            mutation(answer)
            with self.assertRaises(ValueError):
                semantic.parse_response(json.dumps(answer), packet)
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            semantic.parse_response('{"clauses": {}, "clauses": {}}', packet)

    def test_support_shortfall_remains_but_numeric_text_is_not_screened(self):
        data = fixture()
        for kind in ("support", "leakage"):
            answer = response_for(data["packets"][0])
            if kind == "support":
                for j in answer["clauses"]["repair_operator"]["judgments"][4:]:
                    j["supports"] = False
            else:
                answer["clauses"]["repair_operator"]["text"] = "Multiply the rate by 37 to obtain the requested quantity."
            record = semantic.build_semantic_candidate(data["records"][0], data["packets"][0], self.entry(data, answer))
            bank.validate_record(record)
            if kind == "support":
                self.assertFalse(record["qualification"]["construction_qualified"])
                self.assertIsNone(record["descriptor"])
                self.assertEqual(record["clause_support"]["repair_operator"]["support_count"], 4)
            else:
                self.assertTrue(record["qualification"]["construction_qualified"])
                self.assertIn("37", record["descriptor"])

    def test_resealed_card_and_support_tampering_rejected(self):
        data = fixture()
        record = semantic.build_semantic_candidate(data["records"][0], data["packets"][0], self.entry(data))
        for mutate in (lambda r: r["clause_support"]["repair_operator"].update(support_count=99),
                       lambda r: r["unified_process_card"].update(procedure=["Use an unsupported shortcut."])):
            changed = deepcopy(record)
            mutate(changed)
            changed["bank_id"] = bank.content_bank_id(changed)
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                bank.validate_record(bank.seal(changed))

    def test_no_numeric_formula_entity_answer_phrase_or_operator_verb_screen(self):
        data = fixture()
        texts = ["The final answer is the requested quantity with its proper units.",
                 "Profit = total revenue - total cost, after aligning units.",
                 "The remaining fraction is (1 - discount rate).",
                 "The monetary unit relation is 1 dollar = 100 cents.",
                 "The time unit relation is 7 days per week.",
                 "Total contributions form the aggregate quantity."]
        with patch.object(bank, "leakage_issues", side_effect=AssertionError("Static screen must not run")):
            for text in texts:
                answer = response_for(data["packets"][0])
                answer["clauses"]["repair_operator"]["text"] = text
                answer["clauses"]["verification_operator"]["text"] = text
                record = semantic.build_semantic_candidate(data["records"][0], data["packets"][0], self.entry(data, answer))
                bank.validate_record(record)
                self.assertTrue(record["qualification"]["construction_qualified"])
                self.assertFalse(record["leakage_audit"]["static_content_screening_performed"])

    def test_transport_uses_json_mode_disables_redirects_and_sanitizes_receipt(self):
        data = fixture()
        request = semantic.request_spec(data["records"][0], data["packets"][0])
        import requests
        response = requests.Response()
        response.status_code = 200
        wire_response = response_for(data["packets"][0])
        for clause in wire_response["clauses"].values():
            for judgment in clause["judgments"]:
                del judgment["quote"]
        response._content = json.dumps({"choices": [{"message": {"content": json.dumps(wire_response)}}],
                                       "usage": {"total_tokens": 123, "untrusted_field": "ignored"}}).encode()
        with patch("requests.Session.post", return_value=response) as post:
            client = driver.DeepSeekClient("sentinel", request)
            try:
                answer, receipt = client.generate(request, data["packets"][0])
            finally:
                client.close()
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(post.call_args.kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(receipt, {"http_attempts": 1, "final_response_usage": {"total_tokens": 123}})
        self.assertEqual(answer, response_for(data["packets"][0]))

    def test_eight_member_source_citations_are_attached_without_model_transcription(self):
        data = fixture([8, 8, 6, 6, 6, 6, 6] + [7] * 10)
        packet = data["packets"][0]
        for e in packet["evidence"]:
            for field in bank.SIGNATURE_FIELDS:
                e["semantic_signature"][field] += '  Units: “coins” versus value.\nKeep punctuation—even this.'
        answer = response_for(packet)
        for clause in answer["clauses"].values():
            for j in clause["judgments"]:
                del j["quote"]
        answer["clauses"]["repair_operator"]["judgments"][0]["supports"] = False
        result = semantic.parse_response(json.dumps(answer), packet)
        for field, clause in result["clauses"].items():
            for j, e in zip(clause["judgments"], packet["evidence"]):
                self.assertEqual(j["quote"], e["semantic_signature"][field])
        self.assertFalse(result["clauses"]["repair_operator"]["judgments"][0]["supports"])
        self.assertIn("Do NOT output a quote", semantic.SYSTEM_PROMPT)
        for bad_id in ("outside-bank", [], packet["evidence"][1]["evidence_id"]):
            invalid = deepcopy(answer)
            invalid["clauses"]["repair_operator"]["judgments"][0]["evidence_id"] = bad_id
            with self.assertRaises(ValueError):
                semantic.parse_response(json.dumps(invalid), packet)

    def test_old_prompt_cache_is_rejected_without_key_access_or_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = kwargs_for(Path(tmp))
            fake = FakeClient()
            fake.fail_at = 1
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}), self.assertRaises(RuntimeError):
                driver.construct(**kwargs, client_factory=lambda *_: fake)
            path = kwargs["cache_dir"] / "profile.json"
            profile = json.loads(path.read_text())
            profile["policy_sha256"] = bank.canonical_hash({"historical_static_screen": True})
            path.write_text(json.dumps(bank.seal(profile, "profile_sha256")))
            snapshot = {p.name: p.read_bytes() for p in kwargs["cache_dir"].iterdir()}
            with patch.object(driver.os.environ, "get", side_effect=AssertionError("No API key read")):
                with self.assertRaisesRegex(ValueError, "old responses are not reused"):
                    driver.construct(**kwargs, resume=True)
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in kwargs["cache_dir"].iterdir()})

    def test_all_seventeen_requests_have_one_new_prompt_and_no_old_drafts(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = kwargs_for(Path(tmp))
            fake = FakeClient()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}):
                result = driver.construct(**kwargs, client_factory=lambda *_: fake)
            self.assertEqual(len(fake.calls), 17)
            self.assertEqual({r["messages"][0]["content"] for r in fake.calls}, {semantic.SYSTEM_PROMPT})
            for request in fake.calls:
                self.assertEqual(set(json.loads(request["messages"][1]["content"])), {"bank_id", "curation", "evidence"})
            report = result["construction_report.json"]
            self.assertEqual(report["screened_clause_count"], 0)
            self.assertFalse(report["static_content_screening_performed"])
            self.assertEqual(report["source_field_judgment_count"], 580)
            for record in result["candidate_bank_records.jsonl"]:
                self.assertEqual(record["leakage_audit"]["status"], "not_performed_prompt_guidance_only")
                self.assertFalse(record["leakage_audit"]["complete_leakage_freedom_claim"])

    def test_old_prompt_response_cannot_be_relabelled_as_current(self):
        data = fixture()
        entry = self.entry(data)
        entry["request"]["messages"][0]["content"] = "Historical teacher prompt"
        entry["request_sha256"] = bank.canonical_hash(entry["request"])
        entry = bank.seal(entry, "cache_sha256")
        with self.assertRaisesRegex(ValueError, "request binding"):
            semantic.build_semantic_candidate(data["records"][0], data["packets"][0], entry)


if __name__ == "__main__":
    unittest.main()
