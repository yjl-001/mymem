"""Pinned V4.3 tokenizer requests with explicit legacy source-state replay proof."""
from __future__ import annotations

import re
from typing import Any, Mapping

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.experience.v4_3_bank import canonical_hash, seal, text_hash

IDENTITY_FIELDS = ("model_name", "model_revision", "tokenizer_revision")


def resolved_reasoner(source: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(source)
    model_revision = source.get("model_revision")
    tokenizer_revision = source.get("tokenizer_revision")
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("Reasoner model revision must be an exact commit")
    if tokenizer_revision == "main":
        # This is a candidate pinned tokenizer, not a claim about what 'main'
        # pointed at historically. All cached native prefixes must be replayed.
        result["tokenizer_revision"] = model_revision
    elif not isinstance(tokenizer_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
        raise ValueError("Tokenizer revision must be an exact commit or authenticated legacy 'main'")
    return result


def identity(reasoner: Mapping[str, Any]) -> dict[str, Any]:
    return {key: reasoner[key] for key in IDENTITY_FIELDS}


def replay_contract(*, cache, evidence, source_reasoner, packets_sha256):
    if identity(cache.manifest["reasoner"]) != identity(source_reasoner):
        raise ValueError("Legacy source cache and reasoner identities differ")
    prompts = [e for e in cache.events if e["event_kind"] == "prompt_semantic"]
    if len(prompts) != 116 or len({e["sample_id"] for e in prompts}) != 116:
        raise ValueError("Tokenizer migration requires the full 116-sample source cache")
    for event in cache.events:
        e = evidence[event["experience_id"]]
        if (e["sample_id"] != event["sample_id"] or text_hash(e["question"].strip()) != event["question_sha256"]
                or text_hash(e["verified_success_trajectory"].strip()) != event["completion_hashes"]["verified_success_completion_sha256"]
                or text_hash(e["verified_failure_trajectory"].strip()) != event["completion_hashes"]["verified_failure_completion_sha256"]):
            raise ValueError("Tokenizer replay evidence/source-event identity mismatch")
    return seal({"schema_version": "memgen-v4.3-tokenizer-replay-v1",
        "source_reasoner": identity(source_reasoner), "effective_reasoner": identity(resolved_reasoner(source_reasoner)),
        "source_cache_manifest_sha256": cache.manifest["manifest_sha256"], "semantic_packets_sha256": packets_sha256,
        "event_order_sha256": canonical_hash([e["record_sha256"] for e in cache.events]),
        "prompt_count": len(prompts), "event_count": len(cache.events),
        "validation_scope": "all_cached_native_prompt_and_actual_gate_prefix_token_ids",
        "historical_tokenizer_file_equivalence_claim": False}, "replay_sha256")


def validate_tokenizer_replay(*, tokenizer, cache, evidence, source_reasoner, packets_sha256):
    report = replay_contract(cache=cache, evidence=evidence, source_reasoner=source_reasoner, packets_sha256=packets_sha256)
    prompts = {}
    for event in cache.events:
        if event["event_kind"] != "prompt_semantic":
            continue
        prompt = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, evidence[event["experience_id"]]["question"])
        if len(prompt) != event["prompt_token_count"] or canonical_hash(prompt) != event["prompt_token_ids_sha256"]:
            raise ValueError(f"Pinned tokenizer does not reproduce legacy prompt: {event['event_id']}")
        prompts[event["sample_id"]] = prompt
    completions = {}
    for event in cache.events:
        if event["event_kind"] == "prompt_semantic":
            continue
        field = {"failure_gate_attempt": "verified_failure_trajectory", "success_gate_attempt": "verified_success_trajectory"}.get(event["event_kind"])
        if field is None:
            raise ValueError("Unexpected event in tokenizer replay validation")
        key = (event["sample_id"], field)
        if key not in completions:
            completions[key] = list(tokenizer.encode(evidence[event["experience_id"]][field].strip(), add_special_tokens=False))
        count = event["token_position"] + 1
        prefix = (prompts[event["sample_id"]] + completions[key])[:count]
        if len(prefix) != count or canonical_hash(prefix) != event["prefix_alignment"]["prefix_token_ids_sha256"]:
            raise ValueError(f"Pinned tokenizer does not reproduce legacy gate prefix: {event['event_id']}")
    return report
