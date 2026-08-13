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
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import requests


PROMPT_VERSION = "teacher-bank-v2-verified-contrast"
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
        "--proxy-retries",
        type=int,
        default=20,
        help=(
            "Additional retries for proxy tunnel/authentication failures. The default "
            "long backoff covers roughly 90 minutes."
        ),
    )
    parser.add_argument("--proxy-retry-initial-seconds", type=float, default=30.0)
    parser.add_argument("--proxy-retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output and skip already written episode/experience IDs.",
    )
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
        reference_evidence = item.get(
            "reference_evidence",
            "verified_failure" if item.get("reference_trajectory") else "teacher_inferred",
        )
        if reference_evidence == "verified_failure":
            if item.get("outcome") != "verified_success" or item.get("reward") != 1.0:
                raise ValueError(
                    f"Episode {index} claims verified contrast but target is not verified_success"
                )
            target_verifier = item.get("target_verifier")
            reference_verifier = item.get("reference_verifier")
            if not isinstance(target_verifier, dict) or target_verifier.get("reward") != 1.0:
                raise ValueError(
                    f"Episode {index} claims verified target without a one-reward verifier record"
                )
            if not isinstance(reference_verifier, dict) or reference_verifier.get("reward") != 0.0:
                raise ValueError(
                    f"Episode {index} claims verified failure without a zero-reward verifier record"
                )
            if not item.get("target_episode_id") or not item.get("reference_episode_id"):
                raise ValueError(
                    f"Episode {index} claims verified contrast without source episode IDs"
                )
            if not item.get("reference_trajectory") or not item.get("provenance_sha256"):
                raise ValueError(
                    f"Episode {index} claims verified contrast without trajectory/provenance"
                )
            if item.get("source", {}).get("logical_split") != "bank-source":
                raise ValueError(
                    f"Episode {index} formal contrast must come from bank-source"
                )
        yield {
            "id": str(item.get("experience_id", item.get("id", f"episode-{index}"))),
            "experience_id": item.get("experience_id"),
            "source": item.get("source", {"input_jsonl": str(path.expanduser())}),
            "context": str(item["context"]),
            "trajectory": str(item["trajectory"]),
            "outcome": item.get("outcome", "unknown"),
            "reward": item.get("reward"),
            "feedback": item.get("feedback", ""),
            "reference_trajectory": item.get("reference_trajectory", ""),
            "reference_evidence": reference_evidence,
            "target_episode_id": item.get("target_episode_id"),
            "reference_episode_id": item.get("reference_episode_id"),
            "target_verifier": item.get("target_verifier"),
            "reference_verifier": item.get("reference_verifier"),
            "student": item.get("student"),
            "rollout_configuration": item.get("rollout_configuration"),
            "provenance_sha256": item.get("provenance_sha256"),
            "experience_created_at": item.get("created_at"),
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
  }},
  "quality": {{
    "target_supported": true,
    "reference_supported": true,
    "target_reference_distinct": true,
    "contains_instance_specific_details": false,
    "issues": []
  }}
}}

Use confidence in [0, 1]. If no failed trajectory was supplied, infer only a
generic counter-pattern and do not claim that it was observed in this episode.
The quality booleans are strict evidence checks. Mark an item unsupported or
not distinct instead of repairing it with invented details."""
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
    if not isinstance(payload.get("quality"), dict):
        raise ValueError("Teacher response missing object: quality")
    return payload


class TeacherClient:
    """Persistent and credential-safe teacher API client.

    One ``requests.Session`` is shared by all bank records, allowing HTTPS and
    proxy tunnels to be reused. Proxy failures use a separate long backoff
    because an enterprise gateway may take minutes to refresh authentication.
    Exception strings and proxy URLs are deliberately excluded from logs since
    they may contain credentials.
    """

    TRANSIENT_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        retries: int,
        proxy_retries: int,
        proxy_retry_initial_seconds: float,
        proxy_retry_max_seconds: float,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        thinking: str,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("Teacher API key must not be empty")
        if retries <= 0:
            raise ValueError("retries must be positive")
        if proxy_retries < 0:
            raise ValueError("proxy_retries must be non-negative")
        if proxy_retry_initial_seconds <= 0 or proxy_retry_max_seconds <= 0:
            raise ValueError("proxy retry delays must be positive")
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0:
            raise ValueError("HTTP timeouts must be positive")

        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.proxy_retries = proxy_retries
        self.proxy_retry_initial_seconds = proxy_retry_initial_seconds
        self.proxy_retry_max_seconds = proxy_retry_max_seconds
        self.timeout = (connect_timeout_seconds, read_timeout_seconds)
        self.thinking = thinking
        self.session = session or requests.Session()
        # requests reads HTTP(S)_PROXY/NO_PROXY via trust_env. Credentials stay
        # in memory and are never interpolated into our logs.
        self.session.trust_env = True
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        self._sleep = sleep

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "TeacherClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _proxy_wait_seconds(self, failure_count: int) -> float:
        return min(
            self.proxy_retry_initial_seconds * (2 ** (failure_count - 1)),
            self.proxy_retry_max_seconds,
        )

    def _wait_for_proxy(self, failure_count: int) -> None:
        if failure_count > self.proxy_retries:
            raise RuntimeError(
                "Teacher proxy remained unavailable after the configured long-retry window; "
                "rerun with the same output and --resume after proxy authentication recovers."
            ) from None
        delay = self._proxy_wait_seconds(failure_count)
        print(
            "[teacher-bank] proxy tunnel/authentication unavailable "
            f"(retry {failure_count}/{self.proxy_retries}); waiting {delay:.0f}s...",
            file=sys.stderr,
            flush=True,
        )
        self._sleep(delay)

    def call(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.thinking},
        }
        ordinary_failures = 0
        proxy_failures = 0

        while True:
            try:
                response = self.session.post(
                    self.endpoint,
                    json=body,
                    timeout=self.timeout,
                )
            except requests.exceptions.ProxyError:
                proxy_failures += 1
                self._wait_for_proxy(proxy_failures)
                continue
            except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                ordinary_failures += 1
                if ordinary_failures >= self.retries:
                    raise RuntimeError(
                        "Teacher API network request failed after short retries "
                        f"({type(exc).__name__}); no credentials were logged."
                    ) from None
                delay = 2 ** (ordinary_failures - 1)
                print(
                    "[teacher-bank] transient teacher network failure "
                    f"({type(exc).__name__}, retry {ordinary_failures}/{self.retries - 1}); "
                    f"waiting {delay}s...",
                    file=sys.stderr,
                    flush=True,
                )
                self._sleep(delay)
                continue
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    "Teacher HTTP client rejected the request configuration "
                    f"({type(exc).__name__}); no credentials were logged."
                ) from None

            if response.status_code == 407:
                response.close()
                proxy_failures += 1
                self._wait_for_proxy(proxy_failures)
                continue

            if response.status_code in self.TRANSIENT_HTTP_STATUS:
                status = response.status_code
                response.close()
                ordinary_failures += 1
                if ordinary_failures >= self.retries:
                    raise RuntimeError(
                        f"Teacher API returned transient HTTP {status} after short retries."
                    )
                delay = 2 ** (ordinary_failures - 1)
                print(
                    f"[teacher-bank] teacher API HTTP {status} "
                    f"(retry {ordinary_failures}/{self.retries - 1}); waiting {delay}s...",
                    file=sys.stderr,
                    flush=True,
                )
                self._sleep(delay)
                continue

            if not response.ok:
                status = response.status_code
                response.close()
                raise RuntimeError(
                    f"Teacher API returned non-retryable HTTP {status}; check API key, "
                    "balance, model name, and request parameters."
                )

            try:
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                return parse_json_payload(content)
            except (requests.exceptions.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
                ordinary_failures += 1
                if ordinary_failures >= self.retries:
                    raise RuntimeError(
                        "Teacher API returned an invalid JSON payload after short retries."
                    ) from None
                delay = 2 ** (ordinary_failures - 1)
                print(
                    "[teacher-bank] invalid teacher JSON payload "
                    f"(retry {ordinary_failures}/{self.retries - 1}); waiting {delay}s...",
                    file=sys.stderr,
                    flush=True,
                )
                self._sleep(delay)
            finally:
                response.close()


def call_teacher(
    *, base_url: str, api_key: str, model: str, messages: list[dict[str, str]],
    max_tokens: int, temperature: float, retries: int, thinking: str,
) -> dict[str, Any]:
    """Compatibility wrapper for one-off callers; bulk builds use TeacherClient."""

    with TeacherClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
        proxy_retries=20,
        proxy_retry_initial_seconds=30.0,
        proxy_retry_max_seconds=300.0,
        connect_timeout_seconds=30.0,
        read_timeout_seconds=180.0,
        thinking=thinking,
    ) as client:
        return client.call(messages)


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.offset < 0:
        raise ValueError("--limit must be positive and --offset must be non-negative.")
    if args.retries <= 0 or args.proxy_retries < 0:
        raise ValueError("--retries must be positive and --proxy-retries non-negative")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} in scripts/experiments/.server.env before running.")

    if args.dataset == "gsm8k":
        episodes = gsm8k_examples(args.split, args.offset, args.limit)
    else:
        episodes = jsonl_examples(args.input_jsonl, args.offset, args.limit)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    record = json.loads(line)
                    completed_ids.add(str(record.get("experience_id") or record["episode_id"]))
    records_written = 0
    mode = "a" if args.resume else "w"
    with TeacherClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        proxy_retries=args.proxy_retries,
        proxy_retry_initial_seconds=args.proxy_retry_initial_seconds,
        proxy_retry_max_seconds=args.proxy_retry_max_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        read_timeout_seconds=args.read_timeout_seconds,
        thinking=args.thinking,
    ) as client, output_path.open(mode, encoding="utf-8") as handle:
        for episode in episodes:
            if episode["id"] in completed_ids:
                print(f"[teacher-bank] skip completed {episode['id']}", flush=True)
                continue
            bank = client.call(teacher_messages(episode))
            record = {
                "schema_version": "teacher-bank-record-v2",
                "prompt_version": PROMPT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "teacher": {"model": args.model, "base_url": args.base_url},
                "source": episode["source"],
                "episode_id": episode["id"],
                "experience_id": episode.get("experience_id") or episode["id"],
                "outcome": episode["outcome"],
                "reward": episode["reward"],
                "reference_evidence": episode["reference_evidence"],
                "source_episode_ids": {
                    "target": episode.get("target_episode_id"),
                    "reference": episode.get("reference_episode_id"),
                },
                "target_verifier": episode.get("target_verifier"),
                "reference_verifier": episode.get("reference_verifier"),
                "student": episode.get("student"),
                "rollout_configuration": episode.get("rollout_configuration"),
                "provenance_sha256": episode.get("provenance_sha256"),
                "experience_created_at": episode.get("experience_created_at"),
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
