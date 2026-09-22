"""Short-ID protocol for teacher tasks with mechanically exact coverage."""
from __future__ import annotations

from copy import deepcopy


def alias_records(records, *, field="evidence_id", prefix="E"):
    original = [record[field] for record in records]
    if any(not isinstance(value, str) or not value for value in original) or len(original) != len(set(original)):
        raise ValueError(f"Invalid or duplicate {field} values")
    aliases = [f"{prefix}{index:02d}" for index in range(len(records))]
    values = deepcopy(records)
    for value, alias in zip(values, aliases):
        value[field] = alias
    return values, aliases, dict(zip(aliases, original))


def map_exact_ids(value, *, response_field, aliases, mapping, validate):
    validate(value, aliases)
    mapped = deepcopy(value)
    mapped[response_field] = [mapping[item] for item in value[response_field]]
    validate(mapped, list(mapping.values()))
    return mapped


def without_support_ids(value):
    if value is None:
        return None
    return {key: deepcopy(item) for key, item in value.items() if key != "support_ids"}


def identifier_protocol(aliases, *, response_field):
    return {"response_field": response_field, "allowed_ids": list(aliases),
            "require_each_exactly_once": True}
