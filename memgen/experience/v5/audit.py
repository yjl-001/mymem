"""Read-only V5 lineage and qualification audits."""
from __future__ import annotations

from .compilation import validate_compiled


def audit_rollouts(store, config):
    split, episodes = store.require("split"), store.require("stages/episodes")
    expected = len(split["splits"]["train"][:config.train_limit or None]) * (
        config.greedy_rollouts + config.sampled_rollouts)
    if episodes["episode_count"] != expected or len(episodes["keys"]) != expected:
        raise ValueError("V5 Episode count mismatch")
    for key in episodes["keys"]:
        store.require(key)
    return {"complete": True, "phase": "rollouts", "episode_count": expected,
            "input_count": episodes["input_count"], "teacher_inference_used": False}


def audit_complete(store, config):
    rollout = audit_rollouts(store, config)
    for name in ("review", "contrasts", "atoms", "groups", "cards", "compile", "calibrate"):
        store.require("stages/" + name)
    cards = [store.require(key) for key in store.require("stages/cards")["keys"]]
    for card in cards:
        if card["quality_tier"] == "primary":
            if (card["distinct_input_count"] < config.minimum_primary_inputs
                    or card["protocol_failure"] or not card["qualification"]["semantic_passed"]):
                raise ValueError("Unqualified V5 card was promoted to Primary")
    bundle = validate_compiled(store, store.require("stages/compile"))
    if bundle["bank_count"] != sum(card["quality_tier"] == "primary" for card in cards):
        raise ValueError("V5 compiled Bank count differs from Primary cards")
    policy = store.require("stages/calibrate")["policy"]
    if policy["profile_sha256"] != store.profile_hash:
        raise ValueError("V5 selector policy profile drift")
    return {**rollout, "phase": "bank", "candidate_card_count": len(cards),
            "primary_bank_count": bundle["bank_count"], "selector_policy_sha256": policy["policy_sha256"]}
