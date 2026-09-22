"""Alias-safe card construction and semantic review for every teacher group."""
from __future__ import annotations

from collections import Counter

from .artifacts import digest
from .grouping import chunks
from .identifiers import alias_records, identifier_protocol, map_exact_ids, without_support_ids
from .parallel import ordered_map, teacher_workers
from .schemas import validate_card, validate_card_review

STRATEGY = "short-alias-card-v1"


def render_card(card):
    parts = [("Problem structure", [card["problem_structure"]]),
             ("Use when", [card["applies_when"]]), ("Decision point", [card["decision_point"]]),
             ("Procedure", card["procedure"]), ("Avoid", card["avoid"]),
             ("Do not use when", card["exclusions"]), ("Verify", card["verify"])]
    return "\n".join(f"{title}:\n" + "\n".join("- " + value for value in values)
                     for title, values in parts)


def save_fallback(store, task, group_id, batch_index, payload, error, fallback):
    inputs = {"strategy": STRATEGY, "task": task, "group_id": group_id,
              "batch_index": batch_index, "payload": payload}
    return store.put("cards/fallback/" + task + "/" + digest(inputs),
                     {"error": str(error), "fallback": fallback}, inputs)


def ask_batch(store, teacher, task, group_id, batch_index, batch, summary, proposal):
    short, aliases, mapping = alias_records(batch)
    payload = {"group": summary, "evidence": short,
               "identifier_protocol": identifier_protocol(aliases, response_field="support_ids")}
    if task == "card":
        payload["previous_proposal"] = without_support_ids(proposal)
        validator = validate_card
        fallback = {"coherent": False,
                    "reason": "Card construction exhausted bounded short-ID retries.",
                    "support_ids": list(mapping.values())}
    elif task == "card_review":
        payload["proposal"] = without_support_ids(proposal)
        validator = validate_card_review
        fallback = {"quality_tier": "conditional",
                    "reason": "Card review exhausted bounded short-ID retries.",
                    "issues": ["teacher_review_protocol_failure"],
                    "support_ids": list(mapping.values())}
    else:
        raise ValueError(f"Unsupported card task: {task}")
    try:
        answer = teacher.ask(task, payload, lambda value: validator(value, aliases))
    except RuntimeError as exc:
        save_fallback(store, task, group_id, batch_index, payload, exc, fallback)
        return fallback
    mapped = map_exact_ids(answer, response_field="support_ids", aliases=aliases,
                           mapping=mapping, validate=validator)
    inputs = {"strategy": STRATEGY, "task": task, "group_id": group_id,
              "batch_index": batch_index, "payload": payload}
    store.put("cards/mappings/" + task + "/" + digest(inputs),
              {"aliases": mapping, "alias_answer": answer, "mapped_answer": mapped}, inputs)
    return mapped


def construct_card(store, teacher, config, group, source):
    members = [source[evidence_id] for evidence_id in group["members"]]
    inputs = {"group": group, "evidence": members}
    key = "cards/" + group["group_id"]
    cached = store.get(key, inputs)
    if cached is not None:
        return key, cached
    batches = list(chunks(members, config.group_batch_size))
    summary = {name: group[name] for name in ("method", "applies_when", "exclusions")}
    draft = None
    for batch_index, batch in enumerate(batches):
        draft = ask_batch(store, teacher, "card", group["group_id"], batch_index,
                          batch, summary, draft)
        if not draft["coherent"]:
            break
    if draft["coherent"]:
        reviews = [ask_batch(store, teacher, "card_review", group["group_id"], batch_index,
                             batch, summary, draft)
                   for batch_index, batch in enumerate(batches)]
        tiers = {review["quality_tier"] for review in reviews}
        review = {"quality_tier": "reject" if "reject" in tiers else
                  "conditional" if "conditional" in tiers else "primary",
                  "member_batch_reviews": reviews, "support_ids": group["members"]}
    else:
        review = {"quality_tier": "reject", "reason": draft["reason"],
                  "issues": ["incoherent_group"], "support_ids": group["members"]}
    descriptor = render_card(draft["card"]) if draft["coherent"] else None
    body = {"schema_version": "memgen-local-memory-card-v1", "group_id": group["group_id"],
            "evidence_ids": group["members"],
            "source_sample_ids": sorted({record["sample_id"] for record in members}),
            "card": draft.get("card") if draft["coherent"] else None,
            "descriptor": descriptor, "quality_tier": review["quality_tier"],
            "construction_review": review,
            "generalization": {name: draft.get(name) for name in ("substitution_check", "near_miss")},
            "quality_basis": "same_local_teacher_semantic_review",
            "downstream_effectiveness_proven": False, "profile_sha256": store.profile_hash}
    body["bank_id"] = "v43-bank-" + digest(body)
    body["record_sha256"] = digest(body)
    return key, store.put(key, body, inputs)


def run_cards(store, teacher, config):
    groups = store.require("stages/groups")
    source = {record["evidence_id"]: record for record in
              (store.require(key) for key in store.require("stages/evidence")["keys"])}
    work = list(enumerate(groups["groups"]))

    def process(item):
        index, group = item
        key, record = construct_card(store, teacher, config, group, source)
        return index, key, record

    completed = []
    for item in ordered_map(process, work, teacher_workers(teacher)):
        completed.append(item)
        index, _, record = item
        print(f"[local-bank] cards={index + 1}/{len(work)} tier={record['quality_tier']}", flush=True)
    completed.sort(key=lambda value: value[0])
    keys = [key for _, key, _ in completed]
    counts = Counter(record["quality_tier"] for _, _, record in completed)
    return store.put("stages/cards", {"keys": keys, "tier_counts": dict(counts),
        "candidate_count": len(keys), "primary_only_for_runtime": True}, {"groups": digest(groups)})
