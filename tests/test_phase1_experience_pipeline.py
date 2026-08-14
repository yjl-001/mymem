from __future__ import annotations

import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import requests

from memgen.experience.phase1 import (
    EXPERIENCE_SCHEMA,
    ROLLOUT_SCHEMA,
    audit_teacher_record,
    build_verified_experiences,
    create_gsm8k_split_manifest,
    route_ai_adjudication,
    route_ai_review,
    split_audit_reasons,
    summarize_human_review,
)
from scripts.build_teacher_bank import TeacherClient, jsonl_examples
from scripts.build_teacher_bank import teacher_messages
from data.utils.math_utils import diagnose_gsm8k_completion
from scripts.review_experience_bank import (
    adjudicator_messages,
    load_human_resolutions,
    parse_review_payload,
    reviewer_messages,
)


VALID_TEACHER_PAYLOAD = {
    "experience_type": "answer_correctness",
    "failure_types": ["boxed_answer_mismatch"],
    "target": {
        "situation_signature": "supported situation",
        "transferable_decision": "supported decision",
        "verification_rule": "supported verification",
        "applicability_boundary": "supported boundary",
        "confidence": 0.9,
    },
    "reference": {
        "competing_pattern": "failed competing pattern",
        "failure_signal": "observed failure signal",
        "failure_mechanism": "observed failure mechanism",
        "non_reuse_boundary": "reference boundary",
        "confidence": 0.9,
    },
    "evidence": {
        "target_observation": "The successful response satisfies the task contract.",
        "reference_observation": "The failed response emits an incorrect final answer.",
    },
    "quality": {
        "target_supported": True,
        "reference_supported": True,
        "target_reference_distinct": True,
        "failure_type_aligned": True,
        "evidence_grounded": True,
        "contains_instance_specific_details": False,
        "reject_pair": False,
        "issues": [],
    },
}


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self.closed = False

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("missing fixture payload")
        return self._payload

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, outcomes: list[object]):
        self.outcomes = list(outcomes)
        self.headers: dict[str, str] = {}
        self.trust_env = False
        self.post_calls = 0
        self.closed = False

    def post(self, *_args, **_kwargs):
        self.post_calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def teacher_client(session: FakeSession, sleeps: list[float]) -> TeacherClient:
    return TeacherClient(
        base_url="https://api.example.test",
        api_key="top-secret-api-key",
        model="teacher-fixture",
        max_tokens=100,
        temperature=0.0,
        retries=3,
        proxy_retries=4,
        proxy_retry_initial_seconds=30.0,
        proxy_retry_max_seconds=300.0,
        connect_timeout_seconds=5.0,
        read_timeout_seconds=10.0,
        thinking="disabled",
        session=session,
        sleep=sleeps.append,
    )


def rollout(*, episode_id: str, reward: float, trajectory: str) -> dict:
    outcome = "verified_success" if reward == 1.0 else "verified_failure"
    return {
        "schema_version": ROLLOUT_SCHEMA,
        "episode_id": episode_id,
        "sample_id": "gsm8k-train-0-abc",
        "source": {
            "dataset": "openai/gsm8k",
            "dataset_revision": "fixture",
            "dataset_split": "train",
            "logical_split": "bank-source",
            "source_index": 0,
            "question_sha256": "abc",
            "split_manifest_sha256": "manifest",
        },
        "context": "A fixture question?",
        "trajectory": trajectory,
        "outcome": outcome,
        "reward": reward,
        "verifier": {
            "name": "fixture",
            "reward": reward,
            "expected_answer": "4",
            "feedback": f"fixture {outcome}",
        },
        "student": {"model_name": "fixture", "model_revision": "rev", "frozen": True},
        "rollout_configuration": {"sampling_seed": int(reward), "temperature": 0.8},
    }


def teacher_record(experience: dict) -> dict:
    return {
        "schema_version": "teacher-bank-record-v3",
        "experience_id": experience["experience_id"],
        "reference_evidence": "verified_failure",
        "source_episode_ids": {
            "target": experience["target_episode_id"],
            "reference": experience["reference_episode_id"],
        },
        "provenance_sha256": experience["provenance_sha256"],
        "source": copy.deepcopy(experience["source"]),
        "student": copy.deepcopy(experience["student"]),
        "rollout_configuration": copy.deepcopy(experience["rollout_configuration"]),
        "target_verifier": copy.deepcopy(experience["target_verifier"]),
        "reference_verifier": copy.deepcopy(experience["reference_verifier"]),
        "reference_failure_types": copy.deepcopy(experience["reference_failure_types"]),
        "experience_type": experience["experience_type"],
        "bank": {
            "experience_type": experience["experience_type"],
            "failure_types": copy.deepcopy(experience["reference_failure_types"]),
            "target": {
                "situation_signature": "A direct plan remains consistent with the task constraints.",
                "transferable_decision": "Continue the supported calculation without changing goals.",
                "verification_rule": "Check each derived relation against the stated conditions.",
                "applicability_boundary": "Use only while the current plan remains logically supported.",
                "confidence": 0.9,
            },
            "reference": {
                "competing_pattern": "Abandoning a supported path for an unrelated shortcut.",
                "failure_signal": "The new step no longer follows from the prior reasoning.",
                "failure_mechanism": "A goal shift introduces unsupported operations and a wrong result.",
                "non_reuse_boundary": "Do not reuse when new evidence genuinely invalidates the plan.",
                "confidence": 0.9,
            },
            "evidence": {
                "target_observation": "The accepted response follows a supported plan.",
                "reference_observation": "The failed response emits a wrong boxed answer.",
            },
            "quality": {
                "target_supported": True,
                "reference_supported": True,
                "target_reference_distinct": True,
                "failure_type_aligned": True,
                "evidence_grounded": True,
                "contains_instance_specific_details": False,
                "reject_pair": False,
                "issues": [],
            },
        },
    }


class SplitManifestTests(unittest.TestCase):
    def test_splits_are_stable_and_disjoint(self) -> None:
        train = [
            {"question": f"train question {index}", "answer": f"answer {index}"}
            for index in range(10)
        ]
        test = [
            {"question": f"test question {index}", "answer": f"test answer {index}"}
            for index in range(3)
        ]
        first = create_gsm8k_split_manifest(
            train,
            test,
            bank_source_size=6,
            calibration_val_size=2,
            seed=42,
            dataset_revision="fixture",
        )
        second = create_gsm8k_split_manifest(
            train,
            test,
            bank_source_size=6,
            calibration_val_size=2,
            seed=42,
            dataset_revision="fixture",
        )
        self.assertTrue(first["overlap_check"]["passed"])
        self.assertEqual(first["counts"], {
            "bank-source": 6,
            "calibration-val": 2,
            "dev-test": 2,
            "final-test": 3,
        })
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        first_assignments = [
            (item["sample_id"], item["logical_split"]) for item in first["samples"]
        ]
        second_assignments = [
            (item["sample_id"], item["logical_split"]) for item in second["samples"]
        ]
        self.assertEqual(first_assignments, second_assignments)


class VerifiedExperienceTests(unittest.TestCase):
    def test_pairs_only_success_with_failure(self) -> None:
        records = [
            rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
            rollout(episode_id="failure", reward=0.0, trajectory="invalid \\boxed{5}"),
        ]
        experiences, report = build_verified_experiences(records)
        self.assertEqual(len(experiences), 1)
        experience = experiences[0]
        self.assertEqual(experience["schema_version"], EXPERIENCE_SCHEMA)
        self.assertEqual(experience["target_episode_id"], "success")
        self.assertEqual(experience["reference_episode_id"], "failure")
        self.assertEqual(experience["reference_evidence"], "verified_failure")
        self.assertEqual(experience["experience_type"], "answer_correctness")
        self.assertEqual(
            experience["reference_failure_types"], ["boxed_answer_mismatch"]
        )
        self.assertEqual(report["verified_experience_count"], 1)

    def test_format_only_failure_remains_a_valid_reference(self) -> None:
        records = [
            rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
            rollout(episode_id="format-failure", reward=0.0, trajectory="The answer is 4."),
        ]
        experiences, _ = build_verified_experiences(records)
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0]["experience_type"], "format_compliance")
        self.assertEqual(experiences[0]["reference_failure_types"], ["missing_boxed"])
        self.assertTrue(
            experiences[0]["reference_verifier"]["diagnostic_answer_correct"]
        )

    def test_teacher_prompt_receives_reference_verifier_diagnosis(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
                rollout(episode_id="failure", reward=0.0, trajectory="The answer is 4."),
            ]
        )
        messages = teacher_messages(experiences[0])
        prompt = messages[1]["content"]
        self.assertIn("Reference verifier record", prompt)
        self.assertIn("missing_boxed", prompt)
        self.assertIn("format_compliance", prompt)

    def test_rejects_non_bank_source_rollout(self) -> None:
        record = rollout(episode_id="bad-split", reward=0.0, trajectory="wrong")
        record["source"]["logical_split"] = "calibration-val"
        with self.assertRaisesRegex(ValueError, "not from bank-source"):
            build_verified_experiences([record])

    def test_teacher_builder_requires_verifier_provenance(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
                rollout(episode_id="failure", reward=0.0, trajectory="invalid \\boxed{5}"),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiences.jsonl"
            path.write_text(json.dumps(experiences[0]) + "\n", encoding="utf-8")
            loaded = list(jsonl_examples(path, offset=0, limit=1))
            self.assertEqual(loaded[0]["reference_evidence"], "verified_failure")

            invalid = copy.deepcopy(experiences[0])
            invalid["reference_verifier"]["reward"] = 1.0
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero-reward verifier"):
                list(jsonl_examples(path, offset=0, limit=1))


class TeacherAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
                rollout(episode_id="failure", reward=0.0, trajectory="invalid \\boxed{5}"),
            ]
        )
        self.experience = experiences[0]

    def test_approved_record_has_no_reasons(self) -> None:
        self.assertEqual(audit_teacher_record(teacher_record(self.experience), self.experience), [])

    def test_teacher_inferred_reference_is_rejected(self) -> None:
        record = teacher_record(self.experience)
        record["reference_evidence"] = "teacher_inferred"
        reasons = audit_teacher_record(record, self.experience)
        self.assertIn("reference_not_verified_failure", reasons)

    def test_equivalent_or_instance_specific_records_are_rejected(self) -> None:
        record = copy.deepcopy(teacher_record(self.experience))
        record["bank"]["reference"] = copy.deepcopy(record["bank"]["target"])
        record["bank"]["reference"]["competing_pattern"] = "Repeat 42 from the target."
        record["bank"]["quality"]["target_reference_distinct"] = False
        reasons = audit_teacher_record(record, self.experience)
        self.assertIn("instance_specific_literal_detected", reasons)
        self.assertIn("teacher_marks_target_reference_equivalent", reasons)

    def test_mismatched_teacher_failure_type_is_rejected(self) -> None:
        record = teacher_record(self.experience)
        record["bank"]["failure_types"] = ["missing_boxed"]
        reasons = audit_teacher_record(record, self.experience)
        self.assertIn("teacher_failure_types_mismatch", reasons)

    def test_format_pair_requires_format_specific_abstraction(self) -> None:
        experiences, _ = build_verified_experiences(
            [
                rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
                rollout(episode_id="failure", reward=0.0, trajectory="The answer is 4."),
            ]
        )
        experience = experiences[0]
        record = teacher_record(experience)
        record["bank"]["reference"]["failure_signal"] = (
            "The reasoning follows an unsupported numerical relation."
        )
        record["bank"]["reference"]["failure_mechanism"] = (
            "An arithmetic mistake produces an incorrect result."
        )
        reasons = audit_teacher_record(record, experience)
        self.assertIn("format_reference_failure_signal_not_aligned", reasons)
        self.assertIn("format_reference_failure_mechanism_not_aligned", reasons)


class VerifierDiagnosticTests(unittest.TestCase):
    def test_missing_box_with_correct_answer_is_format_only_failure(self) -> None:
        diagnosis = diagnose_gsm8k_completion("Therefore the answer is 14.", "\\boxed{14}")
        self.assertEqual(diagnosis["reward"], 0.0)
        self.assertEqual(diagnosis["failure_types"], ["missing_boxed"])
        self.assertTrue(diagnosis["diagnostic_answer_correct"])

    def test_currency_inside_valid_box_is_normalized(self) -> None:
        diagnosis = diagnose_gsm8k_completion("Therefore \\boxed{$30}", "\\boxed{30}")
        self.assertEqual(diagnosis["reward"], 1.0)
        self.assertEqual(diagnosis["failure_types"], [])

    def test_fbox_is_parsed_without_crashing_legacy_upgrade(self) -> None:
        diagnosis = diagnose_gsm8k_completion("Therefore \\fbox{30}", "\\boxed{30}")
        self.assertEqual(diagnosis["reward"], 1.0)
        self.assertTrue(diagnosis["format_valid"])
        self.assertEqual(diagnosis["predicted_answer"], "30")

    def test_legacy_currency_false_negative_is_rescored_without_new_rollout(self) -> None:
        legacy_success = rollout(
            episode_id="legacy-currency", reward=0.0, trajectory="Therefore \\boxed{$4}"
        )
        legacy_success["verifier"]["version"] = "gsm8k-first-boxed-v1"
        actual_failure = rollout(
            episode_id="actual-failure", reward=0.0, trajectory="Therefore \\boxed{5}"
        )
        experiences, report = build_verified_experiences(
            [legacy_success, actual_failure]
        )
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0]["target_episode_id"], "legacy-currency")
        self.assertEqual(experiences[0]["reference_episode_id"], "actual-failure")
        self.assertEqual(report["verified_success_rollouts"], 1)


class HumanReviewTests(unittest.TestCase):
    def test_requires_complete_ninety_percent_agreement(self) -> None:
        records = []
        for index in range(10):
            records.append({
                "experience_id": f"experience-{index}",
                "human_review": {
                    "target_supported": True,
                    "reference_supported": True,
                    "target_reference_distinct": True,
                    "factually_consistent": index != 0,
                },
            })
        result = summarize_human_review(
            records,
            required_sample_size=10,
            required_agreement=0.9,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["agreement"], 0.9)

        records[1]["human_review"]["factually_consistent"] = None
        incomplete = summarize_human_review(
            records,
            required_sample_size=10,
            required_agreement=0.9,
        )
        self.assertFalse(incomplete["passed"])


class AIReviewRoutingTests(unittest.TestCase):
    def review(self, decision: str, confidence: float, *, supported: bool) -> dict:
        return {
            "decision": decision,
            "confidence": confidence,
            "criteria": {
                "target_supported": supported,
                "reference_supported": supported,
                "target_reference_distinct": supported,
                "factually_consistent": supported,
                "failure_type_aligned": supported,
                "transferable_without_instance_leakage": supported,
            },
        }

    def test_two_high_confidence_votes_are_automated(self) -> None:
        self.assertEqual(
            route_ai_review([], self.review("approve", 0.95, supported=True)),
            "ai_approved",
        )
        self.assertEqual(
            route_ai_review(
                ["teacher_marks_reference_unsupported"],
                self.review("reject", 0.95, supported=False),
            ),
            "ai_rejected",
        )

    def test_high_confidence_pro_overrides_soft_gate_and_low_confidence_is_adjudicated(self) -> None:
        self.assertEqual(
            route_ai_review([], self.review("reject", 0.95, supported=False)),
            "ai_rejected",
        )
        self.assertEqual(
            route_ai_review(
                ["format_failure_not_described"],
                self.review("approve", 0.95, supported=True),
            ),
            "ai_approved",
        )
        self.assertEqual(
            route_ai_review([], self.review("approve", 0.7, supported=True)),
            "ai_adjudication",
        )
        self.assertEqual(
            route_ai_review(
                ["provenance_hash_mismatch"],
                self.review("approve", 0.99, supported=True),
            ),
            "gate_rejected",
        )

    def test_only_unresolved_adjudication_goes_to_human(self) -> None:
        self.assertEqual(
            route_ai_adjudication(self.review("approve", 0.82, supported=True)),
            "ai_approved",
        )
        uncertain = self.review("uncertain", 0.9, supported=True)
        self.assertEqual(route_ai_adjudication(uncertain), "human_review")
        self.assertEqual(
            route_ai_adjudication(self.review("reject", 0.7, supported=False)),
            "human_review",
        )

    def test_audit_reasons_are_split_into_hard_and_soft_authority(self) -> None:
        hard, soft = split_audit_reasons(
            ["provenance_hash_mismatch", "format_failure_not_described"]
        )
        self.assertEqual(hard, ["provenance_hash_mismatch"])
        self.assertEqual(soft, ["format_failure_not_described"])

    def test_reviewer_payload_and_prompt_enforce_format_failure_grounding(self) -> None:
        payload = {
            "decision": "approve",
            "confidence": 0.9,
            "criteria": {
                "target_supported": True,
                "reference_supported": True,
                "target_reference_distinct": True,
                "factually_consistent": True,
                "failure_type_aligned": True,
                "transferable_without_instance_leakage": True,
            },
            "evidence": {"target": "supported", "reference": "supported"},
            "issues": [],
            "uncertainty_reason": "",
        }
        parsed = parse_review_payload(json.dumps(payload))
        self.assertEqual(parsed, payload)

        experiences, _ = build_verified_experiences(
            [
                rollout(episode_id="success", reward=1.0, trajectory="valid \\boxed{4}"),
                rollout(episode_id="failure", reward=0.0, trajectory="The answer is 4."),
            ]
        )
        experience = experiences[0]
        messages = reviewer_messages(experience, teacher_record(experience))
        self.assertIn("format-only reference", messages[0]["content"])
        self.assertIn("must not invent a reasoning error", messages[0]["content"])
        self.assertIn("intentionally hidden", messages[0]["content"])
        self.assertNotIn("automatic_gate_reasons", messages[1]["content"])
        self.assertNotIn('"quality"', messages[1]["content"])
        adjudication = adjudicator_messages(
            experience,
            teacher_record(experience),
            parsed,
            ["format_failure_not_described"],
        )
        self.assertIn("Soft gate warnings", adjudication[0]["content"])
        self.assertIn("first_review", adjudication[1]["content"])

    def test_dispute_finalizer_merges_only_completed_human_decisions(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "finalize_phase1_disputes.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ai_approved = root / "ai-approved.jsonl"
            ai_rejected = root / "ai-rejected.jsonl"
            disputes = root / "disputes.jsonl"
            final_approved = root / "final-approved.jsonl"
            final_rejected = root / "final-rejected.jsonl"
            report = root / "report.json"
            ai_approved.write_text('{"experience_id":"auto"}\n', encoding="utf-8")
            ai_rejected.write_text("", encoding="utf-8")
            disputes.write_text(
                json.dumps(
                    {
                        "experience_id": "disputed",
                        "automatic_gate": {"passed": True, "reasons": []},
                        "ai_review": {"decision": "reject"},
                        "teacher_record": {"experience_id": "disputed"},
                        "human_resolution": {
                            "decision": "approve",
                            "reviewer_notes": "resolved from source evidence",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--ai-approved",
                    str(ai_approved),
                    "--ai-rejected",
                    str(ai_rejected),
                    "--human-review",
                    str(disputes),
                    "--final-approved",
                    str(final_approved),
                    "--final-rejected",
                    str(final_rejected),
                    "--report-output",
                    str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            approved_records = [
                json.loads(line) for line in final_approved.read_text().splitlines()
            ]
            self.assertEqual(len(approved_records), 2)
            self.assertTrue(json.loads(report.read_text())["passed"])

    def test_completed_dispute_resolution_can_survive_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "disputes.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "experience_id": "fixture",
                        "review_provenance_sha256": "stable-review",
                        "human_resolution": {
                            "decision": "reject",
                            "reviewer_notes": "clear contradiction",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_human_resolutions(path)
        self.assertEqual(
            loaded["fixture"]["human_resolution"]["decision"], "reject"
        )


class TeacherClientTests(unittest.TestCase):
    def test_teacher_script_can_start_outside_repository(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "build_teacher_bank.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Construct inspectable target/reference", result.stdout)

    def test_reuses_session_and_recovers_from_proxy_errors(self) -> None:
        response_one = FakeResponse(
            200,
            {"choices": [{"message": {"content": json.dumps(VALID_TEACHER_PAYLOAD)}}]},
        )
        response_two = FakeResponse(
            200,
            {"choices": [{"message": {"content": json.dumps(VALID_TEACHER_PAYLOAD)}}]},
        )
        session = FakeSession(
            [
                requests.exceptions.ProxyError(
                    "407 from http://proxy-user:proxy-secret@proxy.invalid"
                ),
                requests.exceptions.ProxyError("407 Proxy Authentication Required"),
                response_one,
                response_two,
            ]
        )
        sleeps: list[float] = []
        stderr = io.StringIO()
        with redirect_stderr(stderr), teacher_client(session, sleeps) as client:
            self.assertEqual(client.call([]), VALID_TEACHER_PAYLOAD)
            self.assertEqual(client.call([]), VALID_TEACHER_PAYLOAD)

        self.assertEqual(session.post_calls, 4)
        self.assertEqual(sleeps, [30.0, 60.0])
        self.assertTrue(session.closed)
        self.assertTrue(response_one.closed)
        self.assertTrue(response_two.closed)
        log = stderr.getvalue()
        self.assertIn("proxy tunnel/authentication unavailable", log)
        self.assertNotIn("proxy-secret", log)
        self.assertNotIn("top-secret-api-key", log)

    def test_client_accepts_independent_reviewer_parser(self) -> None:
        response = FakeResponse(
            200,
            {"choices": [{"message": {"content": '{"decision":"approve"}'}}]},
        )
        session = FakeSession([response])
        with teacher_client(session, []) as client:
            parsed = client.call(
                [], response_parser=lambda content: {"raw": json.loads(content)}
            )
        self.assertEqual(parsed, {"raw": {"decision": "approve"}})

    def test_http_407_uses_long_proxy_backoff(self) -> None:
        proxy_response = FakeResponse(407)
        success_response = FakeResponse(
            200,
            {"choices": [{"message": {"content": json.dumps(VALID_TEACHER_PAYLOAD)}}]},
        )
        session = FakeSession([proxy_response, success_response])
        sleeps: list[float] = []
        with teacher_client(session, sleeps) as client:
            self.assertEqual(client.call([]), VALID_TEACHER_PAYLOAD)
        self.assertEqual(sleeps, [30.0])
        self.assertTrue(proxy_response.closed)

    def test_non_retryable_http_error_is_sanitized(self) -> None:
        session = FakeSession([FakeResponse(401)])
        sleeps: list[float] = []
        with teacher_client(session, sleeps) as client:
            with self.assertRaisesRegex(RuntimeError, "non-retryable HTTP 401") as raised:
                client.call([])
        self.assertNotIn("top-secret-api-key", str(raised.exception))
        self.assertEqual(sleeps, [])

    def test_exhausted_proxy_retry_suppresses_sensitive_exception_context(self) -> None:
        session = FakeSession(
            [
                requests.exceptions.ProxyError(
                    "407 from http://proxy-user:proxy-secret@proxy.invalid"
                ),
                requests.exceptions.ProxyError(
                    "407 from http://proxy-user:proxy-secret@proxy.invalid"
                ),
            ]
        )
        sleeps: list[float] = []
        stderr = io.StringIO()
        with redirect_stderr(stderr), teacher_client(session, sleeps) as client:
            client.proxy_retries = 1
            with self.assertRaisesRegex(RuntimeError, "configured long-retry window") as raised:
                client.call([])
        self.assertEqual(sleeps, [30.0])
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertNotIn("proxy-secret", stderr.getvalue())
        self.assertNotIn("proxy-secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
