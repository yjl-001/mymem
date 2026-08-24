"""Shared I/O and scoring helpers for E1-A/B/C command-line runners."""

from __future__ import annotations

from dataclasses import dataclass
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
        diagnostic_answers = [
            row.get("verifier", {}).get("diagnostic_answer_correct")
            for row in rows
        ]
        summary[condition] = {
            "sample_count": len(rows),
            "accuracy": sum(float(row["final_reward"]) for row in rows) / len(rows),
            "format_accuracy": sum(bool(row["format_valid"]) for row in rows) / len(rows),
            "diagnostic_answer_accuracy": sum(
                value is True for value in diagnostic_answers
            ) / len(rows),
            "diagnostic_answer_coverage": sum(
                value is not None for value in diagnostic_answers
            ) / len(rows),
            "mean_generation_length": sum(int(row["generation_length"]) for row in rows)
            / len(rows),
            "mean_prompt_token_count": sum(int(row["prompt_token_count"]) for row in rows)
            / len(rows),
        }
    return summary


def _binary_metric(row: Mapping[str, Any], field: str) -> bool | float:
    if field == "diagnostic_answer_correct":
        return row.get("verifier", {}).get("diagnostic_answer_correct") is True
    return row[field]


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
            str(record["sample_id"]): _binary_metric(
                record["conditions"][treatment], field
            )
            for record in records
        },
        {
            str(record["sample_id"]): _binary_metric(
                record["conditions"][control], field
            )
            for record in records
        },
        seed=seed,
        resamples=resamples,
    )


def completion_difference_summary(
    records: Sequence[Mapping[str, Any]], *, treatment: str, control: str
) -> dict[str, Any]:
    """Report token-level completion divergence for a paired condition."""

    different = 0
    for record in records:
        treatment_row = record["conditions"][treatment]
        control_row = record["conditions"][control]
        treatment_hash = treatment_row.get("completion_token_ids_sha256")
        control_hash = control_row.get("completion_token_ids_sha256")
        if treatment_hash is None or control_hash is None:
            changed = treatment_row.get("completion_token_ids") != control_row.get(
                "completion_token_ids"
            )
        else:
            changed = treatment_hash != control_hash
        different += bool(changed)
    count = len(records)
    return {
        "paired_sample_count": count,
        "different_completion_count": different,
        "different_completion_rate": different / count,
    }


def token_sequence_diagnostic(
    reference: Sequence[int], treatment: Sequence[int]
) -> dict[str, Any]:
    """Describe where two greedy token trajectories first diverge."""

    reference_ids = tuple(int(item) for item in reference)
    treatment_ids = tuple(int(item) for item in treatment)
    common_prefix_count = 0
    for reference_id, treatment_id in zip(reference_ids, treatment_ids):
        if reference_id != treatment_id:
            break
        common_prefix_count += 1
    exact_match = reference_ids == treatment_ids
    first_divergence_index = (
        None if exact_match else common_prefix_count
    )
    return {
        "reference_token_count": len(reference_ids),
        "treatment_token_count": len(treatment_ids),
        "exact_match": exact_match,
        "first_token_match": bool(
            reference_ids
            and treatment_ids
            and reference_ids[0] == treatment_ids[0]
        ),
        "common_prefix_token_count": common_prefix_count,
        "first_divergence_index": first_divergence_index,
    }


def strict_accuracy_transition_diagnostics(
    records: Sequence[Mapping[str, Any]], *, treatment: str, control: str
) -> dict[str, int]:
    """Partition strict reward flips into format and answer-content changes.

    ``diagnostic_answer_correct`` is best-effort and never replaces the formal
    GSM8K verifier.  The partition exists to reveal when a strict reward change
    is explained by adding or losing the required box around the same likely
    correct answer.
    """

    counts = {
        "paired_sample_count": len(records),
        "strict_treatment_gain_count": 0,
        "strict_treatment_loss_count": 0,
        "format_only_gain_count": 0,
        "format_only_loss_count": 0,
        "diagnostic_answer_gain_count": 0,
        "diagnostic_answer_loss_count": 0,
        "mixed_or_unclassified_gain_count": 0,
        "mixed_or_unclassified_loss_count": 0,
    }
    for record in records:
        treatment_row = record["conditions"][treatment]
        control_row = record["conditions"][control]
        treatment_reward = bool(treatment_row["final_reward"])
        control_reward = bool(control_row["final_reward"])
        treatment_answer = treatment_row.get("verifier", {}).get(
            "diagnostic_answer_correct"
        )
        control_answer = control_row.get("verifier", {}).get(
            "diagnostic_answer_correct"
        )
        treatment_format = bool(treatment_row["format_valid"])
        control_format = bool(control_row["format_valid"])

        if treatment_reward and not control_reward:
            counts["strict_treatment_gain_count"] += 1
            if (
                control_answer
                and treatment_answer
                and not control_format
                and treatment_format
            ):
                counts["format_only_gain_count"] += 1
            elif control_answer is False and treatment_answer is True:
                counts["diagnostic_answer_gain_count"] += 1
            else:
                counts["mixed_or_unclassified_gain_count"] += 1
        elif control_reward and not treatment_reward:
            counts["strict_treatment_loss_count"] += 1
            if (
                control_answer
                and treatment_answer
                and control_format
                and not treatment_format
            ):
                counts["format_only_loss_count"] += 1
            elif control_answer is True and treatment_answer is False:
                counts["diagnostic_answer_loss_count"] += 1
            else:
                counts["mixed_or_unclassified_loss_count"] += 1
    return counts


def format_transfer_diagnostic(
    *, text_effect: Mapping[str, Any], side_kv_effect: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare the direction of an observed text-format effect with side-KV."""

    text_delta = float(text_effect["mean_treatment_minus_control"])
    side_kv_delta = float(side_kv_effect["mean_treatment_minus_control"])
    positive_text_control = text_delta > 0.0
    if positive_text_control:
        transfer_observed: bool | None = side_kv_delta > 0.0
        status = "observed" if transfer_observed else "not_observed"
    else:
        transfer_observed = None
        status = "no_positive_text_control"
    return {
        "status": status,
        "positive_text_control_present": positive_text_control,
        "text_minus_no_memory": text_delta,
        "side_kv_minus_no_memory": side_kv_delta,
        "side_kv_minus_text": side_kv_delta - text_delta,
        "positive_direction_transferred": transfer_observed,
    }


@dataclass(frozen=True)
class PairedConditionComparison:
    """Named treatment/control pair shared by E1-B and E1-C reports."""

    name: str
    treatment: str
    control: str

    def __post_init__(self) -> None:
        if not self.name or not self.treatment or not self.control:
            raise ValueError("Paired condition comparison fields must be non-empty")
        if self.treatment == self.control:
            raise ValueError("Treatment and control conditions must differ")


class PairedConditionDiagnostics:
    """Build the standard task, format, content, and divergence diagnostics."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        bootstrap_resamples: int,
    ):
        if not records or bootstrap_resamples <= 0:
            raise ValueError("Paired diagnostics require records and resamples")
        self.records = tuple(records)
        self.bootstrap_resamples = bootstrap_resamples

    def summarize(
        self, comparisons: Sequence[PairedConditionComparison]
    ) -> dict[str, dict[str, Any]]:
        if not comparisons or len({item.name for item in comparisons}) != len(
            comparisons
        ):
            raise ValueError("Paired diagnostic names must be non-empty and unique")
        output: dict[str, dict[str, Any]] = {
            "accuracy_effects": {},
            "diagnostic_answer_effects": {},
            "format_effects": {},
            "strict_accuracy_transition_diagnostics": {},
            "completion_difference_diagnostics": {},
        }
        for comparison in comparisons:
            common = {
                "treatment": comparison.treatment,
                "control": comparison.control,
            }
            output["accuracy_effects"][comparison.name] = paired_condition_effect(
                self.records,
                field="final_reward",
                resamples=self.bootstrap_resamples,
                **common,
            )
            output["diagnostic_answer_effects"][comparison.name] = (
                paired_condition_effect(
                    self.records,
                    field="diagnostic_answer_correct",
                    resamples=self.bootstrap_resamples,
                    **common,
                )
            )
            output["format_effects"][comparison.name] = paired_condition_effect(
                self.records,
                field="format_valid",
                resamples=self.bootstrap_resamples,
                **common,
            )
            output["strict_accuracy_transition_diagnostics"][comparison.name] = (
                strict_accuracy_transition_diagnostics(self.records, **common)
            )
            output["completion_difference_diagnostics"][comparison.name] = (
                completion_difference_summary(self.records, **common)
            )
        return output


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
