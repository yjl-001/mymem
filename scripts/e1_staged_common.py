"""Shared I/O and scoring helpers for E1-A/B/C command-line runners."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.experience.e1 import paired_binary_effect
from memgen.experience.e1_staged import build_memory_augmented_messages
from memgen.experience.phase1 import canonical_json_sha256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def logical_manifest_hash(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    })


def load_hashed_manifest(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None and value.get("schema_version") != schema:
        raise ValueError(f"Unexpected manifest schema: {path}")
    if value.get("manifest_sha256") != logical_manifest_hash(value):
        raise ValueError(f"Manifest hash mismatch: {path}")
    return value


def select_split_samples(
    split_manifest: Mapping[str, Any],
    *,
    logical_split: str,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    if logical_split not in {"calibration-val", "dev-test"}:
        raise ValueError("Staged E1 development cannot use final-test")
    if offset < 0 or limit <= 1:
        raise ValueError("offset must be non-negative and limit greater than one")
    if not split_manifest.get("overlap_check", {}).get("passed"):
        raise ValueError("GSM8K split manifest did not pass overlap audit")
    selected = [
        dict(item)
        for item in split_manifest["samples"]
        if item.get("logical_split") == logical_split
    ][offset : offset + limit]
    if len(selected) <= 1:
        raise ValueError("Selected split contains fewer than two samples")
    return selected


def prompt_token_ids(tokenizer: Any, *, question: str, memory_text: str | None) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        build_memory_augmented_messages(question=question, memory_text=memory_text),
        tokenize=False,
        add_generation_prompt=True,
    )
    return [
        int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)
    ]


def processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def score_completion(
    *,
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    ground_truth: str,
    runtime_seconds: float | None,
    prompt_token_count: int,
    memory_ids: Sequence[str] = (),
    side_kv: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completion_ids = tuple(int(item) for item in completion_token_ids)
    completion = tokenizer.decode(list(completion_ids), skip_special_tokens=True).strip()
    verifier = diagnose_gsm8k_completion(completion, ground_truth)
    return {
        "completion": completion,
        "completion_token_ids": list(completion_ids),
        "completion_token_ids_sha256": canonical_json_sha256(list(completion_ids)),
        "generation_length": len(completion_ids),
        "prompt_token_count": prompt_token_count,
        "memory_ids": list(memory_ids),
        "verifier": verifier,
        "final_reward": verifier["reward"],
        "format_valid": verifier["format_valid"],
        "runtime_seconds": runtime_seconds,
        "side_kv": dict(side_kv) if side_kv is not None else None,
    }


def summarize_conditions(
    records: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        rows = [record["conditions"][condition] for record in records]
        summary[condition] = {
            "sample_count": len(rows),
            "accuracy": sum(float(row["final_reward"]) for row in rows) / len(rows),
            "format_accuracy": sum(bool(row["format_valid"]) for row in rows) / len(rows),
            "mean_generation_length": sum(int(row["generation_length"]) for row in rows)
            / len(rows),
            "mean_prompt_token_count": sum(int(row["prompt_token_count"]) for row in rows)
            / len(rows),
        }
    return summary


def paired_condition_effect(
    records: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
    field: str,
    seed: int = 42,
    resamples: int = 10000,
) -> dict[str, Any]:
    return paired_binary_effect(
        {
            str(record["sample_id"]): record["conditions"][treatment][field]
            for record in records
        },
        {
            str(record["sample_id"]): record["conditions"][control][field]
            for record in records
        },
        seed=seed,
        resamples=resamples,
    )


def effect_is_positive(effect: Mapping[str, Any]) -> bool:
    interval = effect.get("bootstrap_95_ci")
    return bool(interval is not None and float(interval[0]) > 0.0)


def validate_resolved_revisions(
    *,
    model: Any,
    tokenizer: Any,
    reasoner: Mapping[str, Any],
    label: str,
) -> None:
    resolved_model = str(
        getattr(model.config, "_commit_hash", None)
        or reasoner["model_revision"]
    )
    resolved_tokenizer = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or reasoner["tokenizer_revision"]
    )
    if (
        resolved_model != str(reasoner["model_revision"])
        or resolved_tokenizer != str(reasoner["tokenizer_revision"])
    ):
        raise ValueError(f"{label} model/tokenizer revision drifted")
