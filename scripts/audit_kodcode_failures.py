#!/usr/bin/env python3
"""Offline failure audit for a completed KodCode static evaluation.

The evaluator already saves one JSON object per evaluated sample to
``.cache/evaluate/kodcode/**/evaluate/answer.json``.  This script selects
failed rows from that file, locates their original KodCode tests using the
saved prompt, and replays the *same* extraction/rename/execution sequence as
``KodCodeEnv.compute_reward``.

It is deliberately separate from training and evaluation code.  Run it in an
isolated environment: replaying a completion executes model-generated Python.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable

from data.kodcode.env import KodCodeEnv
from data.utils.code_utils import PyExecutor, extract_python_code


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_GLOB = ".cache/evaluate/kodcode/**/evaluate/answer.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay failed KodCode evaluations and classify their failure modes."
    )
    parser.add_argument(
        "--answer-json",
        type=Path,
        default=None,
        help=(
            "Path to answer.json. If omitted, select the most recently modified file "
            f"matching {DEFAULT_ANSWER_GLOB!r}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <answer.json parent>/failure_audit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of recorded failures to replay; use 0 to replay all failures (default: 200).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument("--timeout", type=int, default=5, help="Per-sample execution timeout in seconds.")
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional Hugging Face revision passed to KodCode/KodCode-Light-RL-10K.",
    )
    return parser.parse_args()


def resolve_answer_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"answer.json does not exist: {path}")
        return path

    candidates = list(REPO_ROOT.glob(DEFAULT_ANSWER_GLOB))
    if not candidates:
        raise FileNotFoundError(
            "No answer.json found under "
            f"{REPO_ROOT / '.cache/evaluate/kodcode'}. Pass --answer-json explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def load_answer_records(answer_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with answer_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # StaticEvalRecorder appends one summary record without a completion.
            if "completion" not in item:
                continue
            item["_answer_line"] = line_number
            records.append(item)
    if not records:
        raise ValueError(f"No per-sample records found in {answer_path}")
    return records


def recorded_score(record: dict[str, Any]) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    value = next(iter(metrics.values()))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prompt_to_question(prompt: Any) -> str | None:
    """Recover the raw question without depending on the exact instruction prefix."""
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        content = first.get("content") if isinstance(first, dict) else None
    elif isinstance(prompt, dict):
        content = prompt.get("content")
    elif isinstance(prompt, str):
        content = prompt
    else:
        return None
    if not isinstance(content, str):
        return None

    marker = "Question:"
    if marker not in content:
        return None
    return content.split(marker, maxsplit=1)[1].strip()


def build_question_index(dataset_revision: str | None) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    # Import lazily so --help and py_compile do not require the ML environment.
    from datasets import load_dataset

    dataset = load_dataset(
        "KodCode/KodCode-Light-RL-10K",
        split="train",
        revision=dataset_revision,
    )
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index, example in enumerate(dataset):
        question = example["question"].strip()
        index[question].append({"source_index": source_index, "example": example})
    return index, getattr(dataset, "_fingerprint", None)


def get_target_function_name(example: dict[str, Any]) -> str:
    test_info = example["test_info"]
    return test_info[0]["function_name"]


def classify_exception(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (SyntaxError, IndentationError)):
        return "syntax_error"
    if isinstance(error, AssertionError):
        return "assertion_error"
    if isinstance(error, ImportError):
        return "import_error"
    if isinstance(error, NameError):
        return "name_error"
    if isinstance(error, TypeError):
        return "type_error"
    if isinstance(error, (KeyError, IndexError, AttributeError, ValueError)):
        return "runtime_api_error"
    return "runtime_other_error"


def replay_record(
    record: dict[str, Any],
    source_entry: dict[str, Any] | None,
    executor: PyExecutor,
    timeout: int,
    work_dir: Path,
) -> dict[str, Any]:
    completion = str(record.get("completion", ""))
    score = recorded_score(record)
    row: dict[str, Any] = {
        "answer_line": record["_answer_line"],
        "recorded_score": score,
        "question": prompt_to_question(record.get("prompt")),
        "completion": completion,
    }

    if source_entry is None:
        row.update(category="dataset_join_missing", exception=None)
        return row
    if source_entry.get("ambiguous"):
        row.update(category="dataset_join_ambiguous", exception=None)
        return row

    example = source_entry["example"]
    row["source_index"] = source_entry["source_index"]
    blocks = extract_python_code(completion.strip())
    extracted_code = "\n".join(blocks)
    row["extracted_code"] = extracted_code
    if not re.search(r"(?m)^\s*def\s+\w+\s*\(", extracted_code):
        row.update(category="no_function_extracted", exception=None)
        return row

    target_name = get_target_function_name(example)
    renamed_code = KodCodeEnv._rename_func(extracted_code, target_name)
    row["target_function_name"] = target_name
    row["renamed_code"] = renamed_code
    cleaned_test = re.sub(
        r"^\s*from\s+solution\s+import\s+\w+\s*",
        "",
        example["test"],
        flags=re.MULTILINE,
    )

    # This is the same worker method invoked by PyExecutor.execute in the real evaluator.
    try:
        executor._run_with_timeout(renamed_code + "\n" + cleaned_test, timeout, str(work_dir))
    except BaseException as error:  # Classify exactly what the evaluator normally hides.
        row.update(category=classify_exception(error), exception=f"{type(error).__name__}: {error}")
    else:
        row.update(category="pass", exception=None)
    return row


def choose_failures(records: Iterable[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    failures = [record for record in records if (recorded_score(record) or 0.0) == 0.0]
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    if limit == 0 or limit >= len(failures):
        return failures
    return random.Random(seed).sample(failures, limit)


def main() -> int:
    args = parse_args()
    answer_path = resolve_answer_path(args.answer_json)
    output_dir = (args.output_dir or answer_path.parent / "failure_audit").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "execution_workdir"

    records = load_answer_records(answer_path)
    selected = choose_failures(records, args.limit, args.seed)
    question_index, dataset_fingerprint = build_question_index(args.dataset_revision)

    executor = PyExecutor()
    counts: Counter[str] = Counter()
    mismatches = 0
    rows_path = output_dir / "failure_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for position, record in enumerate(selected, start=1):
            question = prompt_to_question(record.get("prompt"))
            matches = question_index.get(question or "", [])
            source_entry: dict[str, Any] | None
            if len(matches) == 1:
                source_entry = matches[0]
            elif len(matches) > 1:
                source_entry = {"ambiguous": True}
            else:
                source_entry = None

            row = replay_record(record, source_entry, executor, args.timeout, work_dir)
            counts[row["category"]] += 1
            if row["category"] == "pass":
                mismatches += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{position}/{len(selected)}] {row['category']}", flush=True)

    all_failures = sum((recorded_score(record) or 0.0) == 0.0 for record in records)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "answer_json": str(answer_path),
        "output_dir": str(output_dir),
        "dataset_name": "KodCode/KodCode-Light-RL-10K",
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "total_answer_records": len(records),
        "recorded_failures": all_failures,
        "replayed_failures": len(selected),
        "sampling_seed": args.seed,
        "timeout_seconds": args.timeout,
        "category_counts": dict(sorted(counts.items())),
        "replay_passes_with_recorded_zero": mismatches,
        "rows_file": str(rows_path),
        "warning": (
            "Replay executes model-generated Python. Use an isolated environment without "
            "credentials or sensitive files."
        ),
    }
    summary_path = output_dir / "failure_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
