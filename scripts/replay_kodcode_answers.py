#!/usr/bin/env python3
"""Re-evaluate saved KodCode completions without generating new responses.

For every row in answer.json (or answers.json), this script reconstructs the
original KodCode test from the saved prompt and can execute either the exact
evaluator sequence used by KodCodeEnv.compute_reward:

    extract_python_code -> _rename_func -> PyExecutor.execute -> reward

or KodCode's public file-based pytest verifier:

    solution.py + test_solution.py -> timeout 30 pytest --cov=solution -> Pass/Fail

The output keeps both the saved reward and the replayed reward so extraction or
execution mismatches are observable instead of being hidden behind one metric.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.kodcode.env import KodCodeEnv
from data.utils.code_utils import PyExecutor, extract_python_code


DEFAULT_GLOBS = (
    ".cache/evaluate/kodcode/**/evaluate/answer.json",
    ".cache/evaluate/kodcode/**/evaluate/answers.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay every saved KodCode answer through the official evaluator path."
    )
    parser.add_argument(
        "--answers-json",
        "--answer-json",
        dest="answers_json",
        type=Path,
        default=None,
        help="Saved answer(s).json path. Defaults to the newest KodCode evaluation result.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <answers-json parent>/kodcode_replay.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Replay at most this many rows; 0 means every row (default: 0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used only with --limit.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "Replay every saved record, including duplicates introduced by distributed "
            "evaluation batch padding. By default, duplicate prompt+solution records are "
            "removed and the first occurrence is kept."
        ),
    )
    parser.add_argument("--timeout", type=int, default=5, help="Per-sample timeout in seconds.")
    parser.add_argument(
        "--official-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for official pytest modes (default: 30, KodCode public default).",
    )
    parser.add_argument(
        "--metric-mode",
        choices=(
            "all_pass",
            "check_code_report",
            "official_pytest",
            "official_pytest_compatible",
            "both",
            "all",
        ),
        default="all_pass",
        help=(
            "all_pass reproduces KodCodeEnv.compute_reward; check_code_report reproduces "
            "PyExecutor.check_code_report; official_pytest reproduces the public KodCode "
            "file-based pytest verifier exactly; official_pytest_compatible keeps MemGen's "
            "extraction but uses KodCode's pytest runner; both calculates the first two; "
            "all calculates the original three metrics (default: all_pass)."
        ),
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional HF revision for KodCode/KodCode-Light-RL-10K.",
    )
    return parser.parse_args()


def resolve_answers_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Answers file does not exist: {path}")
        return path
    candidates = [path for pattern in DEFAULT_GLOBS for path in REPO_ROOT.glob(pattern)]
    if not candidates:
        raise FileNotFoundError(
            "No KodCode answer.json/answers.json found below .cache/evaluate/kodcode. "
            "Pass --answers-json explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # StaticEvalRecorder appends a summary row that has no completion.
            if "completion" not in record:
                continue
            record["_answer_line"] = line_number
            records.append(record)
    if not records:
        raise ValueError(f"No per-sample completion records in {path}")
    return records


def record_identity(record: dict[str, Any]) -> str:
    """Return a stable identity for one saved evaluation example.

    ``answer.json`` does not retain the original dataset row id. The complete
    prompt together with the reference solution is nevertheless stable across
    the duplicated batches created by Accelerate's ``even_batches`` padding.
    Including the solution avoids treating two unusually similar prompts with
    distinct references as the same example.
    """
    return json.dumps(
        {"prompt": record.get("prompt"), "solution": record.get("solution")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first record for each saved evaluation example.

    Accelerate pads distributed evaluation to equal-sized shards by replaying
    items from the start of the dataset. The first occurrence is therefore the
    non-padding occurrence; later occurrences must not affect the metric.
    """
    seen: set[str] = set()
    unique_records: list[dict[str, Any]] = []
    for record in records:
        identity = record_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        unique_records.append(record)
    return unique_records, len(records) - len(unique_records)


def score_from_record(record: dict[str, Any]) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    try:
        return float(next(iter(metrics.values())))
    except (TypeError, ValueError):
        return None


def prompt_to_question(prompt: Any) -> str | None:
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        content = prompt[0].get("content")
    elif isinstance(prompt, dict):
        content = prompt.get("content")
    elif isinstance(prompt, str):
        content = prompt
    else:
        return None
    if not isinstance(content, str) or "Question:" not in content:
        return None
    # This deliberately ignores the instruction prefix, which may differ between forks.
    return content.split("Question:", maxsplit=1)[1].strip()


def build_question_index(revision: str | None) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    from datasets import load_dataset

    raw = load_dataset("KodCode/KodCode-Light-RL-10K", split="train", revision=revision)
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index, example in enumerate(raw):
        index[example["question"].strip()].append(
            {"source_index": source_index, "example": example}
        )
    return index, getattr(raw, "_fingerprint", None)


def choose_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    if limit == 0 or limit >= len(records):
        return records
    return random.Random(seed).sample(records, limit)


def extract_official_kodcode_solution(
    completion: str, function_name: str | None
) -> tuple[str | None, str | None, bool]:
    """Mirror KodCode's public candidate-to-``solution.py`` construction.

    The official SFT verification script keeps text after ``</think>``, extracts
    its first `````python`` block, and renames the function only when that block
    has exactly one function definition. Returning a reason instead of falling
    back to MemGen's extractor makes protocol mismatches visible.
    """
    think_match = re.search(r"</think>(.*)", completion, re.DOTALL)
    if think_match is None:
        return None, "official_missing_think_terminator", False

    code_match = re.search(r"```python(.*?)```", think_match.group(1), re.DOTALL)
    if code_match is None:
        return None, "official_missing_python_code_block", False

    solution_code = code_match.group(1).strip()
    function_names = re.findall(r"def\s+(\w+)\s*\(", solution_code)
    renamed = False
    if len(function_names) == 1 and function_name and function_names[0] != function_name:
        # This deliberately matches the public KodCode script's string replacement.
        solution_code = solution_code.replace(
            f"def {function_names[0]}", f"def {function_name}"
        )
        renamed = True
    return solution_code, None, renamed


def run_official_kodcode_pytest(
    completion: str,
    example: dict[str, Any],
    timeout: int,
    work_root: Path,
    runner_state: dict[str, bool | None],
    solution_code_override: str | None = None,
) -> dict[str, Any]:
    """Run a completion through the public KodCode file-based pytest protocol.

    This intentionally executes untrusted model output. Every case gets an
    isolated temporary directory, no shell interpolation is used, and GNU
    ``timeout --kill-after=5s`` bounds pytest just as in KodCode's public
    ``pipeline/run_test.sh``. It is still not a security sandbox; run only in
    an isolated environment.
    """
    if solution_code_override is None:
        test_info = example.get("test_info")
        function_name: str | None = None
        if isinstance(test_info, list) and len(test_info) == 1 and isinstance(test_info[0], dict):
            value = test_info[0].get("function_name")
            function_name = value if isinstance(value, str) else None

        solution_code, extraction_error, renamed = extract_official_kodcode_solution(
            completion, function_name
        )
        if extraction_error is not None:
            return {
                "score": 0.0,
                "status": extraction_error,
                "solution_code": None,
                "renamed_function": renamed,
                "log_tail": None,
                "coverage_enabled": None,
            }
    else:
        # Compatibility mode isolates execution semantics: use the exact code
        # that MemGen's evaluator would pass to PyExecutor, then run it through
        # KodCode's public file-based pytest runner.
        solution_code = solution_code_override
        renamed = None

    test_code = example.get("test")
    if not isinstance(test_code, str):
        return {
            "score": None,
            "status": "official_missing_test_code",
            "solution_code": solution_code,
            "renamed_function": renamed,
            "log_tail": None,
            "coverage_enabled": None,
        }

    work_root.mkdir(parents=True, exist_ok=True)
    command_prefix = ["timeout", "--kill-after=5s", f"{timeout}s", "pytest", "--rootdir=."]
    coverage_args = ["--cov=solution", "--cov-report", "term"]
    coverage_enabled = runner_state.get("coverage_available") is not False
    try:
        with tempfile.TemporaryDirectory(prefix="official_pytest_", dir=work_root) as temp_dir:
            case_dir = Path(temp_dir)
            (case_dir / "solution.py").write_text(solution_code, encoding="utf-8")
            (case_dir / "test_solution.py").write_text(
                "from solution import *\n\n" + test_code,
                encoding="utf-8",
            )
            completed = subprocess.run(
                command_prefix + (coverage_args if coverage_enabled else []),
                cwd=case_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout + 10,
                check=False,
            )
            output = completed.stdout or ""
            # ``pytest-cov`` is not a MemGen dependency, while KodCode's public
            # runner requests it only to log coverage. Coverage never affects the
            # Pass/Fail exit status, so retry without it when absent rather than
            # reporting every valid test as an unscorable runner error.
            if "unrecognized arguments: --cov" in output or "No module named 'pytest_cov'" in output:
                coverage_enabled = False
                runner_state["coverage_available"] = False
                coverage_error = output
                completed = subprocess.run(
                    command_prefix,
                    cwd=case_dir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout + 10,
                    check=False,
                )
                output = (
                    "[pytest-cov unavailable; retried without coverage because it does not affect "
                    "Pass/Fail]\n"
                    + coverage_error
                    + "\n\n[retry]\n"
                    + (completed.stdout or "")
                )
            elif coverage_enabled:
                runner_state["coverage_available"] = True
    except FileNotFoundError as error:
        return {
            "score": None,
            "status": "official_runner_missing",
            "solution_code": solution_code,
            "renamed_function": renamed,
            "log_tail": f"{type(error).__name__}: {error}",
            "coverage_enabled": None,
        }
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        return {
            "score": 0.0,
            "status": "official_timeout",
            "solution_code": solution_code,
            "renamed_function": renamed,
            "log_tail": str(output)[-4000:],
            "coverage_enabled": coverage_enabled,
        }

    if "No module named pytest" in output:
        status, score = "official_runner_error", None
    elif completed.returncode == 0:
        status, score = "official_pass", 1.0
    elif completed.returncode == 124:
        status, score = "official_timeout", 0.0
    else:
        status, score = "official_fail", 0.0
    return {
        "score": score,
        "status": status,
        "solution_code": solution_code,
        "renamed_function": renamed,
        "log_tail": output[-4000:] or None,
        "coverage_enabled": coverage_enabled,
    }


def replay_one(
    record: dict[str, Any],
    source_entry: dict[str, Any] | None,
    executor: PyExecutor,
    timeout: int,
    official_timeout: int,
    official_work_root: Path,
    official_runner_state: dict[str, bool | None],
    metric_mode: str,
) -> dict[str, Any]:
    completion = str(record.get("completion", ""))
    row: dict[str, Any] = {
        "answer_line": record["_answer_line"],
        "question": prompt_to_question(record.get("prompt")),
        "recorded_score": score_from_record(record),
        "completion": completion,
    }
    if source_entry is None:
        row.update(replay_score=None, category="dataset_join_missing")
        return row
    if source_entry.get("ambiguous"):
        row.update(replay_score=None, category="dataset_join_ambiguous")
        return row

    example = source_entry["example"]
    function_name = example["test_info"][0]["function_name"]
    extracted_blocks = extract_python_code(completion.strip())
    extracted_code = "\n".join(extracted_blocks)
    renamed_code = KodCodeEnv._rename_func(extracted_code, function_name)

    has_function = bool(re.search(r"(?m)^\s*def\s+\w+\s*\(", extracted_code))
    strict_score: float | None = None
    feedback: str | None = None
    if metric_mode in ("all_pass", "both", "all"):
        # These operations mirror KodCodeEnv.compute_reward exactly.
        _, feedback, results = executor.execute(renamed_code, [example["test"]], timeout=timeout)
        strict_score = sum(results) / len(results)

    check_score: float | None = None
    check_report: str | None = None
    check_error: str | None = None
    if metric_mode in ("check_code_report", "both", "all"):
        try:
            reports, scores = executor.check_code_report(
                [completion], [example["test"]], [example["test_info"]], timeout=timeout
            )
            check_report, check_score = reports[0], scores[0]
        except Exception as error:
            check_error = f"{type(error).__name__}: {error}"

    official_result: dict[str, Any] | None = None
    if metric_mode in ("official_pytest", "all"):
        official_result = run_official_kodcode_pytest(
            completion, example, official_timeout, official_work_root, official_runner_state
        )
    official_score = official_result["score"] if official_result is not None else None

    compatible_result: dict[str, Any] | None = None
    if metric_mode == "official_pytest_compatible":
        compatible_result = run_official_kodcode_pytest(
            completion,
            example,
            official_timeout,
            official_work_root,
            official_runner_state,
            solution_code_override=renamed_code,
        )
    compatible_score = compatible_result["score"] if compatible_result is not None else None

    reference_score = strict_score
    if reference_score is None:
        reference_score = check_score
    if reference_score is None:
        reference_score = official_score
    if reference_score is None:
        reference_score = compatible_score
    if metric_mode == "official_pytest" and official_result is not None:
        category = str(official_result["status"])
    elif metric_mode == "official_pytest_compatible" and compatible_result is not None:
        category = str(compatible_result["status"])
    elif reference_score is not None:
        category = (
            "pass" if reference_score == 1.0 else (
                "no_function_extracted" if not has_function else "execution_failed"
            )
        )
    else:
        category = "metric_not_run"
    recorded = row["recorded_score"]
    strict_matches_record: bool | None = None
    if recorded is not None and strict_score is not None:
        strict_matches_record = math.isclose(recorded, strict_score, abs_tol=1e-12)
    row.update(
        source_index=source_entry["source_index"],
        target_function_name=function_name,
        extracted_blocks=extracted_blocks,
        extracted_code=extracted_code,
        renamed_code=renamed_code,
        test=example["test"],
        # replay_score remains for compatibility with existing analysis commands.
        replay_score=reference_score,
        strict_all_pass_score=strict_score,
        check_code_report_score=check_score,
        score_matches_record=strict_matches_record,
        category=category,
        feedback=feedback,
        check_code_report=check_report,
        check_code_report_error=check_error,
        official_pytest_score=official_score,
        official_pytest_status=official_result["status"] if official_result is not None else None,
        official_solution_code=official_result["solution_code"] if official_result is not None else None,
        official_renamed_function=official_result["renamed_function"] if official_result is not None else None,
        official_pytest_log_tail=official_result["log_tail"] if official_result is not None else None,
        official_pytest_coverage_enabled=official_result["coverage_enabled"] if official_result is not None else None,
        official_pytest_compatible_score=compatible_score,
        official_pytest_compatible_status=compatible_result["status"] if compatible_result is not None else None,
        official_pytest_compatible_log_tail=compatible_result["log_tail"] if compatible_result is not None else None,
        official_pytest_compatible_coverage_enabled=(
            compatible_result["coverage_enabled"] if compatible_result is not None else None
        ),
    )
    return row


def main() -> int:
    args = parse_args()
    answers_path = resolve_answers_path(args.answers_json)
    output_dir = (args.output_dir or answers_path.parent / "kodcode_replay").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_records(answers_path)
    if args.keep_duplicates:
        deduplicated_records = raw_records
        duplicates_removed = 0
    else:
        deduplicated_records, duplicates_removed = deduplicate_records(raw_records)
    records = choose_records(deduplicated_records, args.limit, args.seed)
    print(
        "Loaded "
        f"{len(raw_records)} answer records; "
        f"using {len(deduplicated_records)} "
        f"({duplicates_removed} duplicate records removed).",
        flush=True,
    )
    question_index, dataset_fingerprint = build_question_index(args.dataset_revision)
    executor = PyExecutor()
    official_work_root = output_dir / "official_pytest_tmp"
    official_runner_state: dict[str, bool | None] = {"coverage_available": None}

    category_counts: Counter[str] = Counter()
    strict_scores: list[float] = []
    check_scores: list[float] = []
    official_scores: list[float] = []
    compatible_scores: list[float] = []
    mismatches = 0
    official_mismatches = 0
    compatible_mismatches = 0
    rows_path = output_dir / "replay_results.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for position, record in enumerate(records, start=1):
            question = prompt_to_question(record.get("prompt"))
            matches = question_index.get(question or "", [])
            if len(matches) == 1:
                source_entry: dict[str, Any] | None = matches[0]
            elif len(matches) > 1:
                source_entry = {"ambiguous": True}
            else:
                source_entry = None

            row = replay_one(
                record,
                source_entry,
                executor,
                args.timeout,
                args.official_timeout,
                official_work_root,
                official_runner_state,
                args.metric_mode,
            )
            category_counts[row["category"]] += 1
            if row.get("strict_all_pass_score") is not None:
                strict_scores.append(float(row["strict_all_pass_score"]))
            if row.get("check_code_report_score") is not None:
                check_scores.append(float(row["check_code_report_score"]))
            if row.get("official_pytest_score") is not None:
                official_scores.append(float(row["official_pytest_score"]))
            if row.get("official_pytest_compatible_score") is not None:
                compatible_scores.append(float(row["official_pytest_compatible_score"]))
            if row.get("score_matches_record") is False:
                mismatches += 1
            if (
                row.get("recorded_score") is not None
                and row.get("official_pytest_score") is not None
                and not math.isclose(
                    float(row["recorded_score"]),
                    float(row["official_pytest_score"]),
                    abs_tol=1e-12,
                )
            ):
                official_mismatches += 1
            if (
                row.get("recorded_score") is not None
                and row.get("official_pytest_compatible_score") is not None
                and not math.isclose(
                    float(row["recorded_score"]),
                    float(row["official_pytest_compatible_score"]),
                    abs_tol=1e-12,
                )
            ):
                compatible_mismatches += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{position}/{len(records)}] {row['category']}", flush=True)

    if args.metric_mode == "official_pytest":
        replayed_accuracy = (sum(official_scores) / len(official_scores)) if official_scores else None
    elif args.metric_mode == "official_pytest_compatible":
        replayed_accuracy = (sum(compatible_scores) / len(compatible_scores)) if compatible_scores else None
    elif args.metric_mode == "check_code_report":
        replayed_accuracy = (sum(check_scores) / len(check_scores)) if check_scores else None
    else:
        replayed_accuracy = (sum(strict_scores) / len(strict_scores)) if strict_scores else None

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "answers_json": str(answers_path),
        "dataset_name": "KodCode/KodCode-Light-RL-10K",
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "metric_mode": args.metric_mode,
        "answer_records_read": len(raw_records),
        "unique_records_before_limit": len(deduplicated_records),
        "duplicate_records_removed": duplicates_removed,
        "deduplication": "prompt_plus_solution_first_occurrence"
        if not args.keep_duplicates
        else "disabled_by_keep_duplicates",
        "replayed_records": len(records),
        "replayed_accuracy": replayed_accuracy,
        "strict_all_pass_accuracy": (sum(strict_scores) / len(strict_scores)) if strict_scores else None,
        "check_code_report_average": (sum(check_scores) / len(check_scores)) if check_scores else None,
        "official_pytest_accuracy": (sum(official_scores) / len(official_scores)) if official_scores else None,
        "official_pytest_compatible_accuracy": (
            sum(compatible_scores) / len(compatible_scores)
            if compatible_scores
            else None
        ),
        "category_counts": dict(sorted(category_counts.items())),
        "score_mismatches_against_saved_record": mismatches,
        "official_pytest_mismatches_against_saved_record": official_mismatches,
        "official_pytest_compatible_mismatches_against_saved_record": compatible_mismatches,
        "rows_file": str(rows_path),
        "warning": "Replay executes model-generated Python; use an isolated environment.",
    }
    summary_path = output_dir / "replay_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
