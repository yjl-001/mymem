"""Structural validation only: no content keyword/number/entity blacklists."""
from __future__ import annotations

FIELDS = ("problem_structure", "decision_point", "failure_mechanism", "repair_operator", "verification_operator")


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty string")


def strings(value, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError("Expected a list")
    for item in value:
        text(item)


def exact_ids(value, ids):
    strings(value)
    if len(value) != len(set(value)) or set(value) != set(ids):
        raise ValueError("Evidence coverage must be exact, without duplicates or invented IDs")


def validate_review(value):
    for name in ("reasoning_verdict", "final_answer_verdict"):
        if value.get(name) not in {"correct", "incorrect", "uncertain"}:
            raise ValueError(f"Invalid {name}")
    text(value.get("explanation"))
    if not isinstance(value.get("errors"), list):
        raise ValueError("errors must be a list")
    for error in value["errors"]:
        for field in ("location", "error", "correction"):
            text(error.get(field))
    if value["reasoning_verdict"] == "incorrect" and not value["errors"]:
        raise ValueError("Incorrect reasoning requires a concrete error explanation")


def validate_evidence(value):
    if type(value.get("transferable")) is not bool:
        raise ValueError("transferable must be boolean")
    text(value.get("reason"))
    if not value["transferable"]:
        return
    if value.get("experience_kind") not in {"reasoning", "format", "mixed"}:
        raise ValueError("Invalid experience_kind")
    for field in (*FIELDS, "substitution_check", "near_miss", "source_grounding"):
        text(value.get(field))
    strings(value.get("exclusion_conditions"))


def validate_partition(value, ids):
    if not isinstance(value.get("groups"), list) or not value["groups"]:
        raise ValueError("Expected nonempty groups")
    members = []
    for group in value["groups"]:
        strings(group.get("members"), nonempty=True)
        members.extend(group["members"])
        for field in ("method", "applies_when", "rationale"):
            text(group.get(field))
        strings(group.get("exclusions"))
    exact_ids(members, ids)


def validate_match(value, ids):
    strings(value.get("candidate_ids"))
    selected = value["candidate_ids"]
    if len(selected) != len(set(selected)) or not set(selected) <= set(ids):
        raise ValueError("Unknown or duplicate candidate group IDs")
    text(value.get("rationale"))


def validate_merge(value):
    if type(value.get("merge")) is not bool:
        raise ValueError("merge must be boolean")
    text(value.get("rationale"))
    if value["merge"]:
        text(value.get("method"))
        text(value.get("applies_when"))
        strings(value.get("exclusions"))


def validate_membership(value, ids):
    validate_match({"candidate_ids": value.get("incompatible_ids"), "rationale": value.get("rationale")}, ids)


def validate_card(value, ids):
    if type(value.get("coherent")) is not bool:
        raise ValueError("coherent must be boolean")
    text(value.get("reason"))
    exact_ids(value.get("support_ids"), ids)
    if not value["coherent"]:
        return
    card = value.get("card")
    if not isinstance(card, dict):
        raise ValueError("Missing card")
    for field in ("problem_structure", "decision_point", "applies_when"):
        text(card.get(field))
    for field in ("procedure", "avoid", "exclusions", "verify"):
        strings(card.get(field), nonempty=True)
    for field in ("substitution_check", "near_miss"):
        text(value.get(field))


def validate_card_review(value, ids):
    if value.get("quality_tier") not in {"primary", "conditional", "reject"}:
        raise ValueError("Invalid card quality tier")
    text(value.get("reason"))
    strings(value.get("issues"))
    exact_ids(value.get("support_ids"), ids)
