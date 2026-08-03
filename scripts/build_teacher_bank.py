#!/usr/bin/env python3
"""Build preview target/reference experience-bank records with a teacher API.

This is an offline bank-construction utility.  It never participates in the
student model's inference path and never writes API credentials to disk.

For GSM8K preview mode, source records are verified demonstrations from the
training split.  The resulting reference record is explicitly labelled
``teacher_inferred`` because GSM8K itself supplies no failed rollout.  Do not
use those inferred references for a formal contrastive evaluation; provide
verified failed episodes through ``--input-jsonl`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable
from urllib import error, request


PROMPT_VERSION = "teacher-bank-v1"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct inspectable target/reference bank records with a teacher LLM."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        choices=("gsm8k",),
        help="Built-in preview source. GSM8K uses only its train split.",
    )
    source.add_argument(
        "--input-jsonl",
        type=Path,
        help=(
            "Generic episode source. Each line needs context and trajectory; optional "
            "id, outcome, reward, feedback, and reference_trajectory fields are preserved."
        ),
    )
    parser.add_argument("--split", default="train", help="Dataset split for built-in sources.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default="disabled",
        help=(
            "DeepSeek thinking mode. JSON-mode requests default to disabled because some "
            "providers occasionally return an empty final content field in thinking mode."
        ),
    )
    return parser.parse_args()


def gsm8k_examples(split: str, offset: int, limit: int) -> Iterable[dict[str, Any]]:
    if split != "train":
        raise ValueError("GSM8K teacher-bank preview is restricted to the train split.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required; install requirements.txt first.") from exc

    dataset = load_dataset("gsm8k", "main", split=split)
    upper = min(offset + limit, len(dataset))
    for index in range(offset, upper):
        item = dataset[index]
        yield {
            "id": f"gsm8k-train-{index}",
            "source": {"dataset": "gsm8k", "split": split, "index": index},
            "context": item["question"].strip(),
            "trajectory": item["answer"].strip(),
            "outcome": "verified_success",
            "reward": 1.0,
            "feedback": "Official GSM8K training demonstration.",
            "reference_evidence": "teacher_inferred",
        }


def jsonl_examples(path: Path, offset: int, limit: int) -> Iterable[dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for index, item in enumerate(records[offset : offset + limit], start=offset):
        missing = [key for key in ("context", "trajectory") if not item.get(key)]
        if missing:
            raise ValueError(f"Episode {index} missing required fields: {', '.join(missing)}")
        yield {
            "id": str(item.get("id", f"episode-{index}")),
            "source": item.get("source", {"input_jsonl": str(path.expanduser())}),
            "context": str(item["context"]),
            "trajectory": str(item["trajectory"]),
            "outcome": item.get("outcome", "unknown"),
            "reward": item.get("reward"),
            "feedback": item.get("feedback", ""),
            "reference_trajectory": item.get("reference_trajectory", ""),
            "reference_evidence": item.get(
                "reference_evidence",
                "verified_failure" if item.get("reference_trajectory") else "teacher_inferred",
            ),
        }


def teacher_messages(episode: dict[str, Any]) -> list[dict[str, str]]:
    reference = episode.get("reference_trajectory") or "No verified failed trajectory is available."
    system = """You are an offline experience-bank curator for a frozen LLM agent.
Return JSON only, with exactly the schema requested. Your task is abstraction,
not problem solving. Never copy names, numbers, final answers, code literals,
or instance-specific equations from the episode. Never invent a false fact.
Write concise English text that can transfer across math, coding, retrieval, or
tool-use tasks. The target record describes a reusable successful decision
pattern. The reference record describes a competing, inapplicable, or failed
decision pattern; it must not contain a detailed wrong solution."""
    user = f"""Create one target and one reference experience record.

Episode context:
{episode['context']}

Successful trajectory:
{episode['trajectory']}

Outcome: {episode['outcome']}
Verifier reward: {episode.get('reward')}
Verifier feedback: {episode.get('feedback', '')}

Optional verified failed trajectory:
{reference}

Return this JSON object exactly:
{{
  "target": {{
    "situation_signature": "...",
    "transferable_decision": "...",
    "verification_rule": "...",
    "applicability_boundary": "...",
    "confidence": 0.0
  }},
  "reference": {{
    "competing_pattern": "...",
    "failure_signal": "...",
    "failure_mechanism": "...",
    "non_reuse_boundary": "...",
    "confidence": 0.0
  }}
}}

Use confidence in [0, 1]. If no failed trajectory was supplied, infer only a
generic counter-pattern and do not claim that it was observed in this episode."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_payload(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Teacher returned an empty final content field")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    payload = json.loads(cleaned)
    for section in ("target", "reference"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"Teacher response missing object: {section}")
    return payload


def call_teacher(
    *, base_url: str, api_key: str, model: str, messages: list[dict[str, str]],
    max_tokens: int, temperature: float, retries: int, thinking: str,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": thinking},
        }
    ).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(1, retries + 1):
        try:
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return parse_json_payload(content)
        except (error.HTTPError, error.URLError, KeyError, IndexError, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Teacher API failed after {retries} attempts: {exc}") from exc
            print(
                f"[teacher-bank] API attempt {attempt}/{retries} failed: {exc}; retrying...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.offset < 0:
        raise ValueError("--limit must be positive and --offset must be non-negative.")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} in scripts/experiments/.server.env before running.")

    if args.dataset == "gsm8k":
        episodes = gsm8k_examples(args.split, args.offset, args.limit)
    else:
        episodes = jsonl_examples(args.input_jsonl, args.offset, args.limit)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            bank = call_teacher(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                messages=teacher_messages(episode),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                retries=args.retries,
                thinking=args.thinking,
            )
            record = {
                "schema_version": "teacher-bank-record-v1",
                "prompt_version": PROMPT_VERSION,
                "teacher": {"model": args.model, "base_url": args.base_url},
                "source": episode["source"],
                "episode_id": episode["id"],
                "outcome": episode["outcome"],
                "reward": episode["reward"],
                "reference_evidence": episode["reference_evidence"],
                "bank": bank,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records_written += 1
            print(f"[teacher-bank] wrote {episode['id']} ({records_written})", flush=True)
    print(f"[teacher-bank] complete: {output_path} ({records_written} records)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[teacher-bank] error: {exc}", file=sys.stderr)
        raise
