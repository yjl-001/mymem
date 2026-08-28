from __future__ import annotations

import os
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    PROJECT_ROOT
    / "scripts"
    / "experiments"
    / "gsm8k"
    / "run_v3_5_applicability_selector_experiment.sh"
)


class V35RunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        offline_match = re.search(
            r"OFFLINE_RESUME_VALIDATOR='\n(?P<code>.*?)\n'\n\n# Keep",
            cls.source,
            flags=re.DOTALL,
        )
        assert offline_match is not None
        cls.offline_resume_validator = offline_match.group("code")
        match = re.search(
            r"RUN_RESUME_VALIDATOR='\n(?P<code>.*?)\n'\nDEVICE=",
            cls.source,
            flags=re.DOTALL,
        )
        assert match is not None
        cls.resume_validator = match.group("code")

    def build_completed_trace_fixture(
        self, root: Path, *, trace_only: bool = True
    ) -> tuple[list[str], dict[str, object]]:
        evaluator = runpy.run_path(
            str(PROJECT_ROOT / "scripts/evaluate_v3_experience_memory.py"),
            run_name="v35_runner_contract_fixture",
        )
        canonical = evaluator["canonical_json_sha256"]
        file_sha256 = evaluator["file_sha256"]
        profile_schema, row_schema, report_schema = evaluator[
            "evaluation_schemas"
        ]("v3.5")
        input_names = (
            "split.json",
            "memory.jsonl",
            "v3-key.json",
            "side-kv.json",
            "v3-offline.json",
            "e0-final.json",
            "risk.pt",
            "dual-key.json",
            "applicability.json",
        )
        input_paths = []
        for index, name in enumerate(input_names):
            path = root / name
            path.write_bytes(f"fixture-{index}".encode("utf-8"))
            input_paths.append(path)
        selector_path = root / "selector.json"
        if not trace_only:
            selector_path.write_text("selector-fixture\n", encoding="utf-8")
        sample_id = "sample-1"
        logical_split = "calibration-val" if trace_only else "dev-test"
        question_sha256 = "q" * 64
        answer_sha256 = "a" * 64
        split_manifest = {
            "schema_version": evaluator["SPLIT_MANIFEST_SCHEMA"],
            "dataset": {"revision": "fixture-revision"},
            "overlap_check": {"passed": True},
            "samples": [{
                "sample_id": sample_id,
                "logical_split": logical_split,
                "dataset_split": "train",
                "source_index": 7,
                "question_sha256": question_sha256,
                "answer_sha256": answer_sha256,
            }],
        }
        split_manifest["manifest_sha256"] = canonical({
            key: value
            for key, value in split_manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        })
        input_paths[0].write_text(
            json.dumps(split_manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        system_profile = {
            "schema_version": "experience-memory-system-profile-v3.5",
            "calibration_trace_only": trace_only,
        }
        inputs = {
            "split_manifest_sha256": file_sha256(input_paths[0]),
            "memory_records_sha256": file_sha256(input_paths[1]),
            "retrieval_key_manifest_sha256": file_sha256(input_paths[2]),
            "side_kv_manifest_sha256": file_sha256(input_paths[3]),
            "v3_offline_report_sha256": file_sha256(input_paths[4]),
            "e0_final_report_sha256": file_sha256(input_paths[5]),
            "risk_artifact_sha256": file_sha256(input_paths[6]),
            "dual_key_manifest_sha256": file_sha256(input_paths[7]),
            "applicability_calibration_sha256": file_sha256(input_paths[8]),
            "selector_calibration_sha256": (
                None if trace_only else file_sha256(selector_path)
            ),
        }
        profile = {
            "schema_version": profile_schema,
            "created_at": "2026-01-01T00:00:00+00:00",
            "repository": evaluator["repository_state"](),
            "logical_split": logical_split,
            "system_version": "v3.5",
            "calibration_trace_only": trace_only,
            "task_results_used_for_selector_decision": False,
            "selector_decision_data_contract": {
                "task_accuracy_used": False,
                "answer_or_reward_used": False,
                "first_attempt_dynamic_margins_only": trace_only,
            },
            "selected_sample_count": 1,
            "selected_sample_ids_sha256": canonical([sample_id]),
            "slice": {"offset": 0, "limit": 1 if trace_only else 0},
            "system_profile": system_profile,
            "system_profile_sha256": canonical(system_profile),
            "logging": {
                "query_embeddings_sidecar": trace_only,
                "query_embeddings_sidecar_required_for_calibration": (
                    trace_only
                ),
                "query_embedding_sidecar_representation": (
                    "dynamic_query_l2_normalized_exact_audit"
                ),
            },
            "reasoner": {"runtime_dtype": "bfloat16"},
            "inputs": inputs,
        }
        profile["profile_sha256"] = evaluator[
            "evaluation_profile_sha256"
        ](profile)

        def condition(*, strict: bool) -> dict[str, object]:
            token_ids = [1, 2]
            return {
                "completion_token_ids": token_ids,
                "completion_token_ids_sha256": canonical(token_ids),
                "generated_token_count": len(token_ids),
                "strict_correct": strict,
                "format_correct": strict,
            }

        row = {
            "schema_version": row_schema,
            "created_at": "2026-01-01T00:00:01+00:00",
            "profile_sha256": profile["profile_sha256"],
            "sample_id": sample_id,
            "logical_split": logical_split,
            "dataset_split": "train",
            "source_index": 7,
            "question_sha256": question_sha256,
            "answer_sha256": answer_sha256,
            "task_results_used_for_selector_decision": False,
            "calibration_trace_only": trace_only,
            "conditions": {
                "vanilla": condition(strict=False),
                "v3": condition(strict=False)
                | {
                    "online_diagnostics": {
                        "retrieval_attempt_count": 0,
                        "rearm_count": 0,
                        "activation_count": 0,
                        "replacement_count": 0,
                        "duplicate_count": 0,
                        "abstain_count": 0,
                        "memory_attention_step_count": 0,
                    },
                    "runtime_trace": {"retrieval_attempts": []},
                    "query_embedding_sidecar": None,
                },
            },
        }
        row["row_sha256"] = canonical({
            key: value
            for key, value in row.items()
            if key not in {"created_at", "row_sha256"}
        })
        profile_path = root / "run_profile.json"
        results_path = root / "results.jsonl"
        report_path = root / "run_report.json"
        profile_path.write_text(
            json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8"
        )
        results_path.write_text(
            json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = evaluator["progress_report"](
            status="completed",
            profile_sha256=str(profile["profile_sha256"]),
            selected_count=1,
            rows=[evaluator["summary_row"](row)],
            report_schema=report_schema,
        )
        report_path.write_text(
            json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            "-c",
            self.resume_validator,
            str(profile_path),
            str(report_path),
            logical_split,
            "1",
            "true" if trace_only else "false",
            *(str(path) for path in input_paths),
            str(selector_path),
            profile_schema,
            report_schema,
            "experience-memory-system-profile-v3.5",
            str(PROJECT_ROOT),
        ]
        return command, {
            "evaluator": evaluator,
            "profile_path": profile_path,
            "results_path": results_path,
            "report_path": report_path,
        }

    def run_resume_validator(
        self, command: list[str]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_runner_freezes_compressed_offline_calibration_dev_flow(self) -> None:
        required_fragments = (
            "compile_v3_5_dual_selector.py",
            "CALIBRATION_LIMIT=64",
            "--logical-split calibration-val",
            "--calibration-trace-only",
            "calibrate_v3_5_dynamic_selector.py",
            "--target-retained-fraction \"$TARGET_RETAINED_FRACTION\"",
            "--logical-split dev-test",
            "--limit 0",
            "analyze_v3_evaluation.py",
            "compare_v3_5_applicability_selector.py",
            "--v35-selector-calibration",
            "qualify_v3_5_dev.py",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_completed_run_resume_authenticates_profile_rows_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, _ = self.build_completed_trace_fixture(Path(directory))
            result = self.run_resume_validator(command)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_completed_dev_resume_authenticates_false_trace_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, _ = self.build_completed_trace_fixture(
                Path(directory), trace_only=False
            )
            result = self.run_resume_validator(command)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_completed_run_resume_rejects_profile_logical_hash_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, fixture = self.build_completed_trace_fixture(
                Path(directory)
            )
            profile_path = fixture["profile_path"]
            assert isinstance(profile_path, Path)
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["evaluation_interpretation"] = "tampered"
            profile_path.write_text(
                json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8"
            )
            result = self.run_resume_validator(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("logical profile_sha256 mismatch", result.stderr)

    def test_new_or_resumed_runs_are_reauthenticated_before_next_stage(
        self,
    ) -> None:
        self.assertEqual(
            self.source.count(
                '"$CALIBRATION_TRACE_DIR" calibration-val '
                '"$CALIBRATION_LIMIT" true'
            ),
            2,
        )
        self.assertEqual(
            self.source.count(
                'run_is_complete "$DEV_DIR" dev-test '
                '"$DEV_EXPECTED_COUNT" false'
            ),
            2,
        )
        self.assertIn(
            "calibration trace did not satisfy the completed-run "
            "authentication contract",
            self.source,
        )
        self.assertIn(
            "matched dev did not satisfy the completed-run "
            "authentication contract",
            self.source,
        )

    def test_completed_run_resume_rejects_result_row_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, fixture = self.build_completed_trace_fixture(
                Path(directory)
            )
            results_path = fixture["results_path"]
            assert isinstance(results_path, Path)
            row = json.loads(results_path.read_text(encoding="utf-8"))
            row["conditions"]["v3"]["strict_correct"] = True
            results_path.write_text(
                json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
            )
            result = self.run_resume_validator(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid or duplicate results row", result.stderr)

    def test_completed_run_resume_rejects_duplicate_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, fixture = self.build_completed_trace_fixture(
                Path(directory)
            )
            results_path = fixture["results_path"]
            assert isinstance(results_path, Path)
            row_text = results_path.read_text(encoding="utf-8")
            results_path.write_text(row_text + row_text, encoding="utf-8")
            result = self.run_resume_validator(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid or duplicate results row", result.stderr)

    def test_completed_trace_resume_rejects_missing_query_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, fixture = self.build_completed_trace_fixture(
                Path(directory)
            )
            evaluator = fixture["evaluator"]
            results_path = fixture["results_path"]
            assert isinstance(evaluator, dict)
            assert isinstance(results_path, Path)
            canonical = evaluator["canonical_json_sha256"]
            row = json.loads(results_path.read_text(encoding="utf-8"))
            v3 = row["conditions"]["v3"]
            v3["runtime_trace"]["retrieval_attempts"] = [{}]
            v3["query_embedding_sidecar"] = {
                "path": "query_embeddings/missing.safetensors",
                "sha256": "0" * 64,
                "attempt_count": 1,
                "representation": (
                    "dynamic_query_l2_normalized_exact_audit"
                ),
            }
            row["row_sha256"] = canonical({
                key: value
                for key, value in row.items()
                if key not in {"created_at", "row_sha256"}
            })
            results_path.write_text(
                json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
            )
            result = self.run_resume_validator(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid query embedding sidecar", result.stderr)

    def test_completed_run_resume_rejects_current_code_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, fixture = self.build_completed_trace_fixture(
                Path(directory)
            )
            evaluator = fixture["evaluator"]
            profile_path = fixture["profile_path"]
            results_path = fixture["results_path"]
            report_path = fixture["report_path"]
            assert isinstance(evaluator, dict)
            assert isinstance(profile_path, Path)
            assert isinstance(results_path, Path)
            assert isinstance(report_path, Path)
            canonical = evaluator["canonical_json_sha256"]
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["repository"]["implementation_set_sha256"] = "0" * 64
            profile["profile_sha256"] = evaluator[
                "evaluation_profile_sha256"
            ](profile)
            profile_path.write_text(
                json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8"
            )
            row = json.loads(results_path.read_text(encoding="utf-8"))
            row["profile_sha256"] = profile["profile_sha256"]
            row["row_sha256"] = canonical({
                key: value
                for key, value in row.items()
                if key not in {"created_at", "row_sha256"}
            })
            results_path.write_text(
                json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["profile_sha256"] = profile["profile_sha256"]
            report["report_sha256"] = canonical({
                key: value
                for key, value in report.items()
                if key != "report_sha256"
            })
            report_path.write_text(
                json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
            )
            result = self.run_resume_validator(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("repository/code implementation identity", result.stderr)

    def test_completed_run_resume_rejects_self_hashed_report_summary_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, fixture = self.build_completed_trace_fixture(
                Path(directory)
            )
            evaluator = fixture["evaluator"]
            report_path = fixture["report_path"]
            assert isinstance(evaluator, dict)
            assert isinstance(report_path, Path)
            canonical = evaluator["canonical_json_sha256"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["summary"]["sample_count"] = 99
            report["report_sha256"] = canonical({
                key: value
                for key, value in report.items()
                if key != "report_sha256"
            })
            report_path.write_text(
                json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
            )
            result = self.run_resume_validator(command)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run_report hash/count/summary", result.stderr)

    def test_offline_resume_rejects_same_head_with_implementation_drift(
        self,
    ) -> None:
        evaluator = runpy.run_path(
            str(PROJECT_ROOT / "scripts/evaluate_v3_experience_memory.py"),
            run_name="v35_runner_offline_identity_fixture",
        )
        repository = evaluator["repository_state"]()
        implementation_files = repository["implementation_files_sha256"]
        scoped_diff = subprocess.check_output(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                *implementation_files,
            ],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        current = {
            "git_revision": repository["git_revision"],
            "tracked_diff_sha256": evaluator["text_sha256"](scoped_diff),
            "implementation_files_sha256": implementation_files,
            "implementation_set_sha256": evaluator[
                "canonical_json_sha256"
            ](implementation_files),
        }
        stale_identity = {
            "compiler_git_revision": current["git_revision"],
            "compiler_tracked_diff_sha256": current["tracked_diff_sha256"],
            "compiler_implementation_files_sha256": current[
                "implementation_files_sha256"
            ],
            "compiler_implementation_set_sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "offline_report.json"
            calibration_path = root / "applicability_calibration.json"
            manifest_path = root / "dual_retrieval_key_manifest.json"
            report_path.write_text(
                json.dumps({
                    "compiler_git_revision": current["git_revision"],
                    "inputs": stale_identity,
                }),
                encoding="utf-8",
            )
            calibration_path.write_text(
                json.dumps({"source": stale_identity}), encoding="utf-8"
            )
            manifest_path.write_text(
                json.dumps({"input_artifacts": stale_identity}),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-c",
                self.offline_resume_validator,
                str(report_path),
                str(calibration_path),
                str(root / "dual.safetensors"),
                str(manifest_path),
                str(root / "memory.jsonl"),
                str(root / "side.json"),
                str(root / "e0.json"),
                str(root / "v3-key.json"),
                str(root / "v3-offline.json"),
                str(root / "approved.jsonl"),
                str(root / "verified.jsonl"),
                str(root / "split.json"),
                "experience-memory-v3.5-applicability-calibration-v1",
                "experience-memory-v3.5-dual-key-bank-v1",
                "experience-memory-v3.5-offline-report-v1",
                str(PROJECT_ROOT),
            ]
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "offline report compiler implementation identity differs",
            result.stderr,
        )

    def test_runner_requires_v35_artifacts_and_preserves_v3_provenance(self) -> None:
        for fragment in (
            "--v3-retrieval-key-manifest",
            "--v3-offline-report",
            "--dual-key-manifest",
            "--applicability-calibration",
            "--selector-calibration",
            "experience-memory-v3.5-applicability-calibration-v1",
            "experience-memory-v3.5-selector-calibration-v1",
            "task_accuracy_used",
            "answer_or_reward_used",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_offline_reuse_binds_compiler_code_and_split_identity(self) -> None:
        for fragment in (
            "compiler_git_revision",
            "compiler_tracked_diff_sha256",
            "compiler_implementation_files_sha256",
            "compiler_implementation_set_sha256",
            "split_manifest_logical_sha256",
            "dataset_revision",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_embedded_validator_argv_indices_match_shell_contracts(self) -> None:
        self.assertIn(
            "project_root = Path(sys.argv[16]).resolve()",
            self.offline_resume_validator,
        )
        self.assertIn(
            '"$V35_APPLICABILITY_CALIBRATION_SCHEMA" \\\n'
            '      "$V35_DUAL_KEY_BANK_SCHEMA" '
            '"$V35_OFFLINE_REPORT_SCHEMA" \\\n'
            '      "$REPO_ROOT"',
            self.source,
        )
        self.assertIn(
            "project_root = Path(sys.argv[19]).resolve()",
            self.resume_validator,
        )
        self.assertIn(
            '"$V35_EVALUATION_PROFILE_SCHEMA" '
            '"$V35_EVALUATION_REPORT_SCHEMA" \\\n'
            '      "$V35_SYSTEM_PROFILE_SCHEMA" "$REPO_ROOT"',
            self.source,
        )

    def test_query_embedding_sidecars_are_saved_only_for_stage_b(self) -> None:
        self.assertEqual(self.source.count("--save-query-embeddings"), 1)
        calibration_start = self.source.index(
            '"$CALIBRATION_TRACE_DIR" calibration-val'
        )
        calibration_end = self.source.index(
            "python scripts/calibrate_v3_5_dynamic_selector.py",
            calibration_start,
        )
        calibration_stage = self.source[calibration_start:calibration_end]
        self.assertIn("--calibration-trace-only", calibration_stage)
        self.assertIn("--save-query-embeddings", calibration_stage)
        self.assertIn("query_embeddings_sidecar", self.source)
        self.assertIn(
            "query_embeddings_sidecar_required_for_calibration", self.source
        )

        dev_start = self.source.index(
            'if ! run_is_complete "$DEV_DIR" dev-test'
        )
        dev_end = self.source.index(
            "python scripts/analyze_v3_evaluation.py", dev_start
        )
        self.assertNotIn(
            "--save-query-embeddings", self.source[dev_start:dev_end]
        )

    def test_runner_has_only_the_frozen_positional_and_baseline_interface(self) -> None:
        usage = next(
            line for line in self.source.splitlines() if "Usage: $0" in line
        )
        self.assertIn("PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT", usage)
        self.assertIn("--v3-bank-dir DIR", usage)
        self.assertIn("--v34-dev-dir DIR", usage)
        self.assertIn("--v31-dev-dir DIR", usage)

    def test_runner_cannot_start_final_test(self) -> None:
        self.assertIsNone(
            re.search(r"--logical-split[ \t]+(?:\"?\$?\{?FINAL|final-test)", self.source)
        )
        self.assertNotIn("--run-final)", self.source)
        result = subprocess.run(
            ["bash", str(RUNNER), "--run-final"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown option: --run-final", result.stderr)

    def test_runner_rejects_non_bfloat16_v35_execution(self) -> None:
        environment = os.environ.copy()
        environment["MEMGEN_V35_DTYPE"] = "float16"
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "/not-used/phase1",
                "/not-used/e0",
                "/not-used/risk.pt",
                "/not-used/output",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "V3.5 requires MEMGEN_V35_DTYPE=bfloat16", result.stderr
        )

    def test_output_layout_matches_the_v35_artifact_specification(self) -> None:
        expected_paths = (
            'V35_DIR="$OUTPUT_ROOT/v3_5_applicability_selector"',
            'DUAL_KEY_DIR="$V35_DIR/dual_key_bank"',
            'CALIBRATION_TRACE_DIR="$V35_DIR/calibration_trace"',
            'SELECTOR_CALIBRATION="$V35_DIR/selector_calibration.json"',
            'DEV_DIR="$V35_DIR/dev"',
            'V34_COMPARISON="$V35_DIR/dev_v35_minus_v34.json"',
            'V31_COMPARISON="$V35_DIR/dev_v35_minus_v31.json"',
            'QUALIFICATION="$V35_DIR/dev_qualification.json"',
        )
        for fragment in expected_paths:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)


if __name__ == "__main__":
    unittest.main()
