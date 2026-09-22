"""Balanced, provenance-owned V5 Memory Card construction and qualification."""
from __future__ import annotations

from collections import Counter

from memgen.experience.bank_construction.artifacts import digest
from memgen.experience.bank_construction.parallel import ordered_map, teacher_workers
from .grouping import atom_view, chunks
from .schemas import validate_card, validate_card_review, validate_summary


def render_card(card):
    selector, payload = card["selector_key"], card["memory_payload"]
    sections = [
        ("Applicable when", [payload["applicability_summary"]]),
        ("Recommended method", payload["method"]),
        ("Avoid", payload["avoid"]),
        ("Why", [payload["rationale"]]),
        ("Verify", payload["verification"]),
        ("Limits", [*selector["exclusions_from_input"], *payload["runtime_limits"]]),
    ]
    return "\n".join(f"{title}:\n" + "\n".join("- " + item for item in values)
                     for title, values in sections if values)


def selector_text(selector_key):
    return "\n".join(("Task goal: " + selector_key["task_goal"],
                      "Problem structure: " + selector_key["problem_structure"],
                      "Observable cues: " + "; ".join(selector_key["observable_cues"]),
                      "Required operation: " + selector_key["required_operation"]))


def safe_ask(store, teacher, task, payload, validate, key, inputs):
    cached = store.get(key, inputs)
    if cached is not None:
        return cached
    try:
        answer = teacher.ask(task, payload, validate)
        result = {"ok": True, "answer": answer}
    except RuntimeError as exc:
        result = {"ok": False, "error": str(exc)}
    return store.put(key, result, inputs)


def summarize_group(store, teacher, config, group, atoms):
    level = []
    for index, batch in enumerate(chunks(atoms, config.card_leaf_size)):
        payload = {"kind": "experience_atoms", "items": [atom_view(atom) for atom in batch]}
        inputs = {"group_id": group["group_id"], "level": 0, "index": index, "payload": payload}
        result = safe_ask(store, teacher, "summarize_group", payload, validate_summary,
                          f"card_summaries/{group['group_id']}/level-00/{index:06d}", inputs)
        if not result["ok"]:
            return None, result["error"]
        level.append(result["answer"])
    depth = 1
    while len(level) > 1:
        next_level = []
        for index, batch in enumerate(chunks(level, 2)):
            if len(batch) == 1:
                next_level.append(batch[0])
                continue
            payload = {"kind": "child_summaries", "items": batch}
            inputs = {"group_id": group["group_id"], "level": depth,
                      "index": index, "payload": payload}
            result = safe_ask(store, teacher, "summarize_group", payload, validate_summary,
                              f"card_summaries/{group['group_id']}/level-{depth:02d}/{index:06d}", inputs)
            if not result["ok"]:
                return None, result["error"]
            next_level.append(result["answer"])
        level, depth = next_level, depth + 1
    return level[0], None


def construct_card(store, teacher, config, group, atom_by_id):
    atoms = [atom_by_id[atom_id] for atom_id in group["members"]]
    inputs = {"group": group, "atoms_sha256": digest(atoms),
              "minimum_primary_inputs": config.minimum_primary_inputs}
    key = "cards/" + group["group_id"]
    cached = store.get(key, inputs)
    if cached is not None:
        return key, cached
    summary, error = summarize_group(store, teacher, config, group, atoms)
    card, review, protocol_failure = None, None, error is not None
    if not protocol_failure:
        card_result = safe_ask(store, teacher, "card", {"group_summary": summary}, validate_card,
            f"card_generation/{group['group_id']}", {"group": group, "summary": summary})
        protocol_failure = not card_result["ok"]
        error = card_result.get("error")
        if card_result["ok"]:
            card = card_result["answer"]
            review_result = safe_ask(store, teacher, "card_review",
                {"group_summary": summary, "candidate_card": card}, validate_card_review,
                f"card_reviews/{group['group_id']}", {"group": group, "summary": summary, "card": card})
            protocol_failure = not review_result["ok"]
            error = review_result.get("error")
            review = review_result.get("answer")
    support_ok = group["distinct_input_count"] >= config.minimum_primary_inputs
    semantic_ok = bool(review and review["quality_tier"] == "primary" and review["coherent"]
                       and review["executable"] and review["input_observable_key"]
                       and review["contradiction_free"])
    if protocol_failure or group["protocol_fallback"] or not card:
        tier = "reject"
    elif support_ok and semantic_ok:
        tier = "primary"
    elif review and review["quality_tier"] != "reject":
        tier = "conditional"
    else:
        tier = "reject"
    body = {"schema_version": "memgen-v5-memory-card-v1", "group_id": group["group_id"],
        "atom_ids": group["members"], "source_input_ids": sorted({atom["input_id"] for atom in atoms}),
        "distinct_input_count": group["distinct_input_count"], "selector_key": card["selector_key"] if card else None,
        "memory_payload": card["memory_payload"] if card else None,
        "selector_text": selector_text(card["selector_key"]) if card else None,
        "descriptor": render_card(card) if card else None, "quality_tier": tier,
        "teacher_review": review, "protocol_failure": protocol_failure or group["protocol_fallback"],
        "protocol_error": error, "qualification": {"minimum_distinct_inputs": config.minimum_primary_inputs,
            "support_passed": support_ok, "semantic_passed": semantic_ok,
            "fallback_free": not (protocol_failure or group["protocol_fallback"])},
        "downstream_effectiveness_proven": False, "profile_sha256": store.profile_hash}
    body["bank_id"] = "v5-bank-" + digest(body)
    body["record_sha256"] = digest(body)
    return key, store.put(key, body, inputs)


def run_cards(store, teacher, config):
    groups = store.require("stages/groups")
    atoms = {atom["atom_id"]: atom for atom in
             (store.require(key) for key in store.require("stages/atoms")["keys"])}
    work = list(enumerate(groups["groups"]))
    completed = []
    def process(item):
        index, group = item
        return index, *construct_card(store, teacher, config, group, atoms)
    for item in ordered_map(process, work, teacher_workers(teacher)):
        completed.append(item)
        print(f"[v5] cards={item[0] + 1}/{len(work)} tier={item[2]['quality_tier']}", flush=True)
    completed.sort()
    counts = Counter(record["quality_tier"] for _, _, record in completed)
    return store.put("stages/cards", {"keys": [key for _, key, _ in completed],
        "candidate_count": len(completed), "tier_counts": dict(counts),
        "primary_only_for_runtime": True}, {"groups": digest(groups)})
