"""Structural protocol validation for V5 teacher outputs.

Semantic quality belongs to the teacher/critic. These checks deliberately contain no
number, entity, answer-language, or overlap blacklist.
"""
from __future__ import annotations


def text(value, name="value"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def strings(value, name="value", *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be a list")
    for item in value:
        text(item, name)


def subset_ids(value, allowed, name):
    strings(value, name)
    if len(value) != len(set(value)) or not set(value) <= set(allowed):
        raise ValueError(f"{name} contains duplicate or unknown IDs")


def validate_review(value):
    for name in ("process_verdict", "answer_verdict"):
        if value.get(name) not in {"correct", "incorrect", "uncertain"}:
            raise ValueError(f"Invalid {name}")
    text(value.get("explanation"), "explanation")
    if not isinstance(value.get("errors"), list):
        raise ValueError("errors must be a list")
    for error in value["errors"]:
        for field in ("location", "error", "correction"):
            text(error.get(field), field)


def validate_atom(value):
    if type(value.get("transferable")) is not bool:
        raise ValueError("transferable must be boolean")
    text(value.get("reason"), "reason")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0,1]")
    if not value["transferable"]:
        return
    applicability = value.get("applicability")
    experience = value.get("experience")
    if not isinstance(applicability, dict) or not isinstance(experience, dict):
        raise ValueError("Missing applicability or experience")
    for field in ("task_goal", "problem_structure", "required_operation"):
        text(applicability.get(field), field)
    strings(applicability.get("observable_cues"), "observable_cues", nonempty=True)
    for field in ("do", "avoid", "verify"):
        strings(experience.get(field), field, nonempty=True)
    text(experience.get("why"), "why")
    strings(experience.get("runtime_limits"), "runtime_limits")
    strings(value.get("exclusions_from_input"), "exclusions_from_input")
    text(value.get("failure_mechanism"), "failure_mechanism")


def validate_partition(value, aliases):
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be nonempty")
    seen = []
    for group in groups:
        strings(group.get("members"), "members", nonempty=True)
        seen.extend(group["members"])
        for field in ("shared_method", "applicability", "rationale"):
            text(group.get(field), field)
        strings(group.get("exclusions"), "exclusions")
    if len(seen) != len(set(seen)) or set(seen) != set(aliases):
        raise ValueError("Partition must cover every supplied short ID exactly once")


def validate_pair_judgments(value, pair_ids):
    judgments = value.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("judgments must be a list")
    seen = []
    for row in judgments:
        pid = row.get("pair_id")
        if pid not in pair_ids or pid in seen:
            raise ValueError("Unknown or duplicate pair_id")
        seen.append(pid)
        if row.get("relation") not in {"same_method", "related_but_different", "different"}:
            raise ValueError("Invalid group relation")
        if type(row.get("applicability_compatible")) is not bool or type(row.get("exclusion_conflict")) is not bool:
            raise ValueError("Pair compatibility fields must be booleans")
        text(row.get("reason"), "reason")
    if set(seen) != set(pair_ids):
        raise ValueError("Pair judgments must cover the supplied pairs")


def validate_cluster_review(value, edge_ids):
    subset_ids(value.get("reject_edge_ids"), edge_ids, "reject_edge_ids")
    text(value.get("reason"), "reason")


SUMMARY_FIELDS = ("shared_method", "applicability", "exclusions", "failure_mechanisms",
                  "recommended_actions", "avoid", "verification", "runtime_limits")


def validate_summary(value):
    for field in ("shared_method", "applicability"):
        text(value.get(field), field)
    for field in SUMMARY_FIELDS[2:]:
        strings(value.get(field), field)


def validate_card(value):
    selector = value.get("selector_key")
    payload = value.get("memory_payload")
    if not isinstance(selector, dict) or not isinstance(payload, dict):
        raise ValueError("Missing selector_key or memory_payload")
    for field in ("task_goal", "problem_structure", "required_operation"):
        text(selector.get(field), field)
    strings(selector.get("observable_cues"), "observable_cues", nonempty=True)
    strings(selector.get("exclusions_from_input"), "exclusions_from_input")
    text(payload.get("applicability_summary"), "applicability_summary")
    for field in ("method", "avoid", "verification"):
        strings(payload.get(field), field, nonempty=True)
    text(payload.get("rationale"), "rationale")
    strings(payload.get("runtime_limits"), "runtime_limits")


def validate_card_review(value):
    if value.get("quality_tier") not in {"primary", "conditional", "reject"}:
        raise ValueError("Invalid quality_tier")
    for field in ("coherent", "executable", "input_observable_key", "contradiction_free"):
        if type(value.get(field)) is not bool:
            raise ValueError(f"{field} must be boolean")
    text(value.get("reason"), "reason")
    strings(value.get("issues"), "issues")
