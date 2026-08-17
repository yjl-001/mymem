"""Deterministic data contracts for Phase 2 steering experiments.

The GPU-heavy vector compiler and evaluator live in ``scripts/``.  Keeping the
selection, provenance, and calibration rules here makes the experiment
auditable and testable without loading a language model.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256


STEERING_VECTOR_ARTIFACT_SCHEMA = "steering-vector-artifact-v1"
STEERING_CALIBRATION_SCHEMA = "steering-calibration-report-v1"
PHASE2_EVIDENCE_ANCHOR_SCHEMA = "phase2-evidence-anchor-v1"
PHASE2_DELIMITERS = (",", ".", "\n")
PHASE2_PRIMARY_EXPERIENCE_TYPE = "answer_correctness"
PHASE2_ELIGIBLE_EXPERIENCE_TYPES = frozenset(
    {
        "answer_correctness",
        "format_compliance",
        "mixed_or_unclassified_task_failure",
    }
)
PHASE2_MECHANISM_CLUSTERS = frozenset(
    {
        "arithmetic_or_numeric",
        "unit_or_conversion",
        "counting_or_discreteness",
        "temporal_or_sequence",
        "relation_or_constraint",
        "other_task_reasoning",
    }
)

FORMAT_INSTRUCTION = (
    "Solve the math problem with proper reasoning, and make sure to put the "
    "FINAL ANSWER inside \\boxed{}."
)


def build_gsm8k_messages(question: str) -> list[dict[str, str]]:
    """Return exactly the user message used for the frozen Phase 1 student."""

    return [{
        "role": "user",
        "content": f"{FORMAT_INSTRUCTION}\nQuestion: {question.strip()}\n",
    }]


def approved_experiences(
    approved_records: Iterable[Mapping[str, Any]],
    experiences: Iterable[Mapping[str, Any]],
    *,
    allowed_experience_types: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join the formal Phase 1 bank to its raw verified trajectories.

    A Pro-approved teacher record is a quality decision, not the numerical
    evidence used by the vector compiler.  This function therefore returns the
    matching ``verified_experiences`` entries after checking the critical
    provenance fields on both sides.
    """

    allowed = set(allowed_experience_types or PHASE2_ELIGIBLE_EXPERIENCE_TYPES)
    if not allowed:
        raise ValueError("allowed_experience_types must not be empty")

    experience_by_id: dict[str, Mapping[str, Any]] = {}
    for experience in experiences:
        experience_id = str(experience.get("experience_id", ""))
        if not experience_id or experience_id in experience_by_id:
            raise ValueError(f"Missing or duplicate verified experience_id: {experience_id!r}")
        experience_by_id[experience_id] = experience

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    skipped_types: Counter[str] = Counter()
    for record in approved_records:
        experience_id = str(record.get("experience_id", ""))
        if not experience_id or experience_id in seen_ids:
            raise ValueError(f"Missing or duplicate approved experience_id: {experience_id!r}")
        seen_ids.add(experience_id)
        gate = record.get("ai_review_gate")
        if not isinstance(gate, Mapping) or gate.get("route") != "ai_approved":
            raise ValueError(f"{experience_id} is not an ai_approved bank record")
        if record.get("reference_evidence") != "verified_failure":
            raise ValueError(f"{experience_id} does not have verified_failure evidence")
        experience = experience_by_id.get(experience_id)
        if experience is None:
            raise ValueError(f"Approved record {experience_id} has no verified experience")
        for key in ("provenance_sha256", "source", "student", "experience_type"):
            if record.get(key) != experience.get(key):
                raise ValueError(f"{experience_id} has mismatched {key}")
        if record.get("source_episode_ids") != {
            "target": experience.get("target_episode_id"),
            "reference": experience.get("reference_episode_id"),
        }:
            raise ValueError(f"{experience_id} has mismatched source_episode_ids")
        if experience.get("reference_evidence") != "verified_failure":
            raise ValueError(f"{experience_id} verified record lost failure provenance")
        if experience.get("reference_verifier", {}).get("reward") != 0.0:
            raise ValueError(f"{experience_id} reference verifier is not a failure")
        experience_type = str(experience.get("experience_type", ""))
        if experience_type not in allowed:
            skipped_types[experience_type] += 1
            continue
        target = str(experience.get("trajectory", "")).strip()
        reference = str(experience.get("reference_trajectory", "")).strip()
        if not target or not reference:
            raise ValueError(f"{experience_id} has an empty target or reference trajectory")
        selected.append(dict(experience))

    if not selected:
        raise ValueError("No approved experiences remain after Phase 2 type selection")
    report = {
        "selected_count": len(selected),
        "selected_by_experience_type": dict(
            sorted(Counter(str(item["experience_type"]) for item in selected).items())
        ),
        "skipped_by_experience_type": dict(sorted(skipped_types.items())),
        "selection_provenance_sha256": canonical_json_sha256(
            {
                "experience_ids": [item["experience_id"] for item in selected],
                "allowed_experience_types": sorted(allowed),
            }
        ),
    }
    return selected, report


def parse_csv_strings(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_csv_numbers(value: str, *, integer: bool = False) -> list[int] | list[float]:
    values = parse_csv_strings(value)
    try:
        parsed: list[int] | list[float]
        if integer:
            parsed = [int(item) for item in values]
        else:
            parsed = [float(item) for item in values]
    except ValueError as exc:
        kind = "integer" if integer else "number"
        raise ValueError(f"Invalid {kind} CSV: {value!r}") from exc
    return parsed


def last_completion_boundary(
    token_ids: Sequence[int],
    *,
    completion_start: int,
    decode_token: Callable[[int], str],
    delimiters: Sequence[str] = PHASE2_DELIMITERS,
) -> int | None:
    """Return the last delimiter-ending token in a completion.

    This mirrors the online delimiter policy: first inspect the token text,
    including merged tokens such as ``,\n``.  The returned index identifies the
    hidden state that predicts the *following* token.
    """

    if completion_start < 0 or completion_start > len(token_ids):
        raise ValueError("completion_start is outside token_ids")
    delimiter_tuple = tuple(delimiters)
    candidate: int | None = None
    for index in range(completion_start, len(token_ids)):
        token_text = decode_token(int(token_ids[index]))
        if token_text.rstrip(" \t").endswith(delimiter_tuple):
            candidate = index
    return candidate


def stable_uniform(seed: int, *parts: str) -> float:
    """A platform-independent random number for deterministic controls."""

    material = "|".join([str(seed), *parts]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value / float(2**64)


def soft_entropy_gate(entropy: float, threshold: float, slope: float) -> float:
    """Numerically stable sigmoid gate used after the hard entropy trigger."""

    if not all(math.isfinite(value) for value in (entropy, threshold, slope)):
        raise ValueError("entropy, threshold, and slope must be finite")
    if slope <= 0:
        raise ValueError("slope must be positive")
    scaled = (entropy - threshold) / slope
    if scaled >= 40:
        return 1.0
    if scaled <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-scaled))


def validate_evidence_anchor(
    payload: Mapping[str, Any], experience: Mapping[str, Any]
) -> list[str]:
    """Validate Pro-provided exact quotes before they become vector evidence.

    The reviewer can decide that a pair is not safely anchorable.  A quote is
    accepted only when it appears exactly once in the original trajectory and
    ends at a real delimiter, so the compiler never guesses a semantic span.
    """

    reasons: list[str] = []
    if payload.get("decision") != "anchor":
        return ["reviewer_excluded"]
    cluster = payload.get("mechanism_cluster")
    if cluster not in PHASE2_MECHANISM_CLUSTERS:
        reasons.append("invalid_mechanism_cluster")
    for side, trajectory_field in (
        ("target", "trajectory"),
        ("reference", "reference_trajectory"),
    ):
        anchor = payload.get(f"{side}_anchor")
        if not isinstance(anchor, Mapping):
            reasons.append(f"missing_{side}_anchor")
            continue
        quote = anchor.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            reasons.append(f"missing_{side}_quote")
            continue
        trajectory = str(experience.get(trajectory_field, ""))
        if trajectory.count(quote) != 1:
            reasons.append(f"{side}_quote_not_unique_exact_match")
        if not quote.rstrip(" \t").endswith(PHASE2_DELIMITERS):
            reasons.append(f"{side}_quote_not_delimiter_terminated")
        if "\\boxed" in quote or "\\fbox" in quote:
            reasons.append(f"{side}_quote_is_final_answer_formatting")
        if len(quote) < 12 or len(quote) > 900:
            reasons.append(f"{side}_quote_length_out_of_range")
    return reasons


def select_calibration_winner(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select one config with explicit, deterministic safety tie-breaks."""

    eligible = [
        dict(row)
        for row in rows
        if row.get("condition") == "real_vector"
        and not bool(row.get("safety_failed", False))
        and float(row.get("format_accuracy", 0.0)) >= float(row.get("vanilla_format_accuracy", 0.0))
    ]
    if not eligible:
        raise ValueError("No real-vector calibration row satisfied the format/safety guard")

    # Higher reward wins.  Then prefer format preservation, fewer injections,
    # and finally a stable lexical config key so re-runs cannot choose a
    # different artifact on a tie.
    def sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
        return (
            -float(row.get("accuracy", 0.0)),
            -float(row.get("format_accuracy", 0.0)),
            float(row.get("mean_injections", math.inf)),
            canonical_json_sha256(dict(row.get("config", {}))),
        )

    return min(eligible, key=sort_key)
