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

    def test_real_support_shortfall_and_generated_leakage_remain_explicit(self):
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
            self.assertFalse(record["qualification"]["construction_qualified"])
            self.assertIsNone(record["descriptor"])
            if kind == "support":
                self.assertEqual(record["clause_support"]["repair_operator"]["support_count"], 4)

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

    def test_generic_relations_allowed_but_final_constants_screened(self):
        self.assertEqual(semantic.screen_card("Multiply the target quantity by twice the stated rate.", []), [])
        self.assertIn("numeric_constant", semantic.screen_card("Multiply the target quantity by 37.", []))

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

    def make_legacy_partial_cache(self, root):
        data = fixture([8, 8, 6, 6, 6, 6, 6] + [7] * 10)
        kwargs = kwargs_for(root, data)
        paths = {"records": kwargs["source_dir"] / "bank_records.jsonl",
                 "manifest": kwargs["source_dir"] / "bank_manifest.json",
                 "packets": kwargs["packets_path"], "policy": kwargs["policy_path"]}
        requests = [semantic.request_spec(r, p, legacy=True) for r, p in zip(data["records"], data["packets"])]
        profile = bank.seal({"schema_version": "memgen-v4.3-deepseek-cache-v1",
            "input_sha256": {k: bank.file_hash(p) for k, p in paths.items()},
            "implementation_sha256": driver.LEGACY_IMPLEMENTATION_HASHES,
            "policy_sha256": bank.canonical_hash(semantic.POLICY), "model": "deepseek-v4-flash", "max_tokens": 8192,
            "request_sha256": [bank.canonical_hash(r) for r in requests]}, "profile_sha256")
        cache = kwargs["cache_dir"]
        cache.mkdir()
        driver.atomic_json(cache / "profile.json", profile, immutable=True)
        entry = bank.seal({"profile_sha256": profile["profile_sha256"], "request": requests[0],
            "request_sha256": bank.canonical_hash(requests[0]), "response": response_for(data["packets"][0]),
            "receipt": {"http_attempts": 1, "final_response_usage": {"total_tokens": 900}}}, "cache_sha256")
        entry_path = cache / (entry["request_sha256"] + ".json")
        driver.atomic_json(entry_path, entry, immutable=True)
        return kwargs, entry_path

    def test_dbeab53_partial_cache_preserved_and_only_sixteen_banks_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, entry_path = self.make_legacy_partial_cache(Path(tmp))
            old_entry = entry_path.read_bytes()
            profile_path = kwargs["cache_dir"] / "profile.json"
            old_profile = profile_path.read_bytes()
            fake = FakeClient()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}):
                output = driver.construct(**kwargs, resume=True, client_factory=lambda *_: fake)
            self.assertEqual(len(fake.calls), 16)
            self.assertTrue(all(r["messages"][0]["content"] == semantic.SYSTEM_PROMPT for r in fake.calls))
            self.assertEqual(entry_path.read_bytes(), old_entry)
            self.assertEqual(profile_path.read_bytes(), old_profile)
            self.assertEqual(output["construction_report.json"]["external_api_calls_made"], 17)
            self.assertEqual(output["construction_report.json"]["qualified_tier_counts"], {"primary": 11, "conditional": 6})
            load_construction(kwargs["output_dir"])
            for flag in ("resume", "validate_only"):
                with patch.object(driver.os.environ, "get", side_effect=AssertionError("Must not read API key")):
                    driver.construct(**kwargs, **{flag: True})

    def test_legacy_cache_quote_and_profile_drift_still_rejected_before_api(self):
        for mutation in ("quote", "profile", "input"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                kwargs, entry_path = self.make_legacy_partial_cache(Path(tmp))
                if mutation == "quote":
                    entry = json.loads(entry_path.read_text())
                    entry["response"]["clauses"]["repair_operator"]["judgments"][0]["quote"] = "a paraphrase is not a legacy citation"
                    entry_path.write_text(json.dumps(bank.seal(entry, "cache_sha256")))
                elif mutation == "profile":
                    profile_path = kwargs["cache_dir"] / "profile.json"
                    profile = json.loads(profile_path.read_text())
                    profile["implementation_sha256"]["memgen/experience/v4_3_deepseek.py"] = "a" * 64
                    profile_path.write_text(json.dumps(bank.seal(profile, "profile_sha256")))
                else:
                    kwargs["source_dir"].joinpath("bank_records.jsonl").write_text(
                        kwargs["source_dir"].joinpath("bank_records.jsonl").read_text() + "\n")
                with patch.object(driver.os.environ, "get", side_effect=AssertionError("Must reject before key access")):
                    with self.assertRaises(ValueError):
                        driver.construct(**kwargs, resume=True)
                self.assertFalse((kwargs["cache_dir"] / "profile-citations-v2.json").exists())

    def test_migration_interruption_resumes_new_and_legacy_responses_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, entry_path = self.make_legacy_partial_cache(Path(tmp))
            original = entry_path.read_bytes()
            first = FakeClient()
            first.fail_at = 2
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sentinel"}):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    driver.construct(**kwargs, resume=True, client_factory=lambda *_: first)
                second = FakeClient()
                driver.construct(**kwargs, resume=True, client_factory=lambda *_: second)
            self.assertEqual(len(second.calls), 14)
            self.assertEqual(entry_path.read_bytes(), original)
            load_construction(kwargs["output_dir"])


if __name__ == "__main__":
    unittest.main()
