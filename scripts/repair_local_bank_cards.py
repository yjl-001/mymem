#!/usr/bin/env python3
"""Resume card construction with short, mechanically mapped evidence IDs.

This compatibility entry point leaves the implementation frozen by an existing
run profile unchanged.  Accepted cards already stored under ``cards/`` are
reused.  Missing cards use E00... aliases for both construction and review,
then map the validated answers back to the original evidence IDs before the
normal card checkpoint is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.bank_construction.artifacts import Store, digest, file_digest, read_json, run_lock
from memgen.experience.bank_construction.cards import render_card
from memgen.experience.bank_construction.config import ConstructionConfig, ModelConfig
from memgen.experience.bank_construction.grouping import chunks
from memgen.experience.bank_construction.parallel import ordered_map, teacher_workers
from memgen.experience.bank_construction.schemas import validate_card, validate_card_review
from memgen.experience.bank_construction.sources import environment, implementation_hashes
from memgen.experience.bank_construction.teacher import Teacher
from memgen.experience.bank_construction.vllm_teacher import VLLMTeacher

STRATEGY = "short-alias-card-recovery-v1"


def config_from_profile(profile):
    raw = deepcopy(profile["configuration"])
    raw["reasoner"] = ModelConfig(**raw["reasoner"])
    raw["teacher"] = ModelConfig(**raw["teacher"])
    return ConstructionConfig(**raw)


def alias_evidence(records):
    aliases = [f"E{i:02d}" for i in range(len(records))]
    values = deepcopy(records)
    original_ids = [value["evidence_id"] for value in records]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("Card batch contains duplicate evidence IDs")
    for value, alias in zip(values, aliases):
        value["evidence_id"] = alias
    return values, aliases, dict(zip(aliases, original_ids))


def map_support_ids(answer, aliases, mapping, validator):
    validator(answer, aliases)
    mapped = deepcopy(answer)
    mapped["support_ids"] = [mapping[value] for value in answer["support_ids"]]
    validator(mapped, list(mapping.values()))
    return mapped


def semantic_proposal(value):
    """Keep card semantics while hiding irrelevant IDs from another batch."""
    if value is None:
        return None
    return {key: deepcopy(item) for key, item in value.items() if key != "support_ids"}


def save_recovery(store, task, group_id, batch_index, payload, answer, mapped, mapping):
    inputs = {"strategy": STRATEGY, "task": task, "group_id": group_id,
              "batch_index": batch_index, "payload": payload}
    key = f"cards/recovery/{task}/{digest(inputs)}"
    body = {"schema_version": "memgen-card-alias-recovery-v1",
            "strategy": STRATEGY, "aliases": mapping, "alias_answer": answer,
            "mapped_answer": mapped}
    store.put(key, body, inputs)


def save_fallback(store, task, group_id, batch_index, payload, error, fallback):
    inputs = {"strategy": STRATEGY, "task": task, "group_id": group_id,
              "batch_index": batch_index, "payload": payload}
    key = f"cards/recovery/fallback/{task}/{digest(inputs)}"
    store.put(key, {"schema_version": "memgen-card-alias-fallback-v1",
        "strategy": STRATEGY, "error": str(error), "fallback": fallback}, inputs)


def ask_with_aliases(store, teacher, task, group_id, batch_index, batch, extra):
    short, aliases, mapping = alias_evidence(batch)
    payload = {"group": extra["group"], "evidence": short,
               "identifier_protocol": {"allowed_support_ids": aliases,
                                       "require_each_exactly_once": True}}
    if task == "card":
        payload["previous_proposal"] = semantic_proposal(extra.get("previous_proposal"))
        validator = validate_card
    elif task == "card_review":
        payload["proposal"] = semantic_proposal(extra["proposal"])
        validator = validate_card_review
    else:
        raise ValueError(f"Unsupported alias card task: {task}")
    try:
        answer = teacher.ask(task, payload, lambda value: validator(value, aliases))
    except RuntimeError as exc:
        original_ids = list(mapping.values())
        fallback = rejected_draft(original_ids) if task == "card" else uncertain_review(original_ids)
        save_fallback(store, task, group_id, batch_index, payload, exc, fallback)
        return fallback
    mapped = map_support_ids(answer, aliases, mapping, validator)
    save_recovery(store, task, group_id, batch_index, payload, answer, mapped, mapping)
    return mapped


def rejected_draft(ids):
    return {"coherent": False,
            "reason": "Card construction could not be validated after bounded short-ID retries.",
            "support_ids": list(ids)}


def uncertain_review(ids):
    return {"quality_tier": "conditional",
            "reason": "The card review could not be validated after bounded short-ID retries.",
            "issues": ["teacher_review_protocol_failure"],
            "support_ids": list(ids)}


def construct_one(store, teacher, config, group, source):
    members = [source[evidence_id] for evidence_id in group["members"]]
    inputs = {"group": group, "evidence": members}
    key = "cards/" + group["group_id"]
    record = store.get(key, inputs)
    if record is not None:
        return record, True

    batches = list(chunks(members, config.group_batch_size))
    summary = {name: group[name] for name in ("method", "applies_when", "exclusions")}
    draft = None
    for batch_index, batch in enumerate(batches):
        draft = ask_with_aliases(store, teacher, "card", group["group_id"], batch_index,
                                 batch, {"group": summary, "previous_proposal": draft})
        if not draft["coherent"]:
            break

    if draft["coherent"]:
        reviews = []
        for batch_index, batch in enumerate(batches):
            review = ask_with_aliases(store, teacher, "card_review", group["group_id"],
                                      batch_index, batch,
                                      {"group": summary, "proposal": draft})
            reviews.append(review)
        tiers = {value["quality_tier"] for value in reviews}
        review = {"quality_tier": "reject" if "reject" in tiers else
                  "conditional" if "conditional" in tiers else "primary",
                  "member_batch_reviews": reviews, "support_ids": group["members"]}
    else:
        review = {"quality_tier": "reject", "reason": draft["reason"],
                  "issues": ["incoherent_group"], "support_ids": group["members"]}

    descriptor = render_card(draft["card"]) if draft["coherent"] else None
    body = {"schema_version": "memgen-local-memory-card-v1", "group_id": group["group_id"],
            "evidence_ids": group["members"],
            "source_sample_ids": sorted({value["sample_id"] for value in members}),
            "card": draft.get("card") if draft["coherent"] else None,
            "descriptor": descriptor, "quality_tier": review["quality_tier"],
            "construction_review": review,
            "generalization": {name: draft.get(name) for name in ("substitution_check", "near_miss")},
            "quality_basis": "same_local_teacher_semantic_review",
            "downstream_effectiveness_proven": False, "profile_sha256": store.profile_hash}
    body["bank_id"] = "v43-bank-" + digest(body)
    body["record_sha256"] = digest(body)
    return store.put(key, body, inputs), False


def run_alias_cards(store, teacher, config):
    groups = store.require("stages/groups")
    source = {record["evidence_id"]: record for record in
              (store.require(key) for key in store.require("stages/evidence")["keys"])}
    work = list(enumerate(groups["groups"]))

    def process(item):
        index, group = item
        record, reused = construct_one(store, teacher, config, group, source)
        return index, "cards/" + group["group_id"], record, reused

    completed = []
    for item in ordered_map(process, work, teacher_workers(teacher)):
        completed.append(item)
        index, _, record, reused = item
        print(f"[card-recovery] cards={index + 1}/{len(work)} tier={record['quality_tier']} "
              f"source={'cached' if reused else 'short_alias'}", flush=True)
    completed.sort(key=lambda value: value[0])
    keys, counts = [], Counter()
    for _, key, record, _ in completed:
        keys.append(key)
        counts[record["quality_tier"]] += 1
    result = {"keys": keys, "tier_counts": dict(counts), "candidate_count": len(keys),
              "primary_only_for_runtime": True}
    # Match the core stage payload exactly so a later normal --resume is idempotent.
    store.put("stages/cards", result, {"groups": digest(groups)})
    manifest_inputs = {"strategy": STRATEGY, "groups": digest(groups)}
    store.put("cards/recovery/alias-v1-manifest", {
        "schema_version": "memgen-card-alias-recovery-manifest-v1",
        "strategy": STRATEGY, "script_sha256": file_digest(Path(__file__).resolve()),
        "candidate_count": len(keys), "tier_counts": dict(counts)}, manifest_inputs)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with run_lock(args.output_dir):
        profile = read_json(args.output_dir / "profile.json")
        if implementation_hashes() != profile["implementation"]:
            raise ValueError("Frozen core implementation differs from this run; use the exact checkout that created it")
        if environment() != profile["environment"]:
            raise ValueError("Runtime environment differs from this run")
        config = config_from_profile(profile)
        if config.teacher_backend != "vllm":
            raise ValueError("Card alias recovery requires the recorded local vLLM teacher")
        store = Store(args.output_dir, profile)
        adapter = VLLMTeacher(store, config, profile["teacher"])
        teacher = Teacher(store, config, lambda: adapter)
        try:
            result = run_alias_cards(store, teacher, config)
            print(f"[card-recovery] complete cards={result['candidate_count']} "
                  f"tiers={result['tier_counts']}", flush=True)
        finally:
            teacher.close()


if __name__ == "__main__":
    main()
