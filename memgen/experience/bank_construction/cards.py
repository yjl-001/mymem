"""One evidence-grounded card per teacher group; semantic review, no content blacklist."""
from __future__ import annotations

from collections import Counter
from .artifacts import digest
from .schemas import validate_card, validate_card_review
from .grouping import chunks


def render_card(card):
    parts = [("Problem structure", [card["problem_structure"]]),
             ("Use when", [card["applies_when"]]), ("Decision point", [card["decision_point"]]),
             ("Procedure", card["procedure"]), ("Avoid", card["avoid"]),
             ("Do not use when", card["exclusions"]), ("Verify", card["verify"])]
    return "\n".join(f"{title}:\n" + "\n".join("- " + v for v in values) for title, values in parts)


def run_cards(store, teacher, config):
    groups = store.require("stages/groups")
    source = {r["evidence_id"]: r for r in
              (store.require(k) for k in store.require("stages/evidence")["keys"])}
    keys, counts = [], Counter()
    for i, group in enumerate(groups["groups"]):
        members = [source[eid] for eid in group["members"]]
        inputs = {"group": group, "evidence": members}
        key = "cards/" + group["group_id"]
        record = store.get(key, inputs)
        if record is None:
            draft = None
            batches = list(chunks(members, config.group_batch_size))
            summary = {k: group[k] for k in ("method", "applies_when", "exclusions")}
            for batch in batches:
                ids = [e["evidence_id"] for e in batch]
                draft = teacher.ask("card", {"group": summary, "evidence": batch, "previous_proposal": draft},
                                    lambda r: validate_card(r, ids))
                if not draft["coherent"]:
                    break
            if draft["coherent"]:
                reviews = []
                for batch in batches:
                    ids = [e["evidence_id"] for e in batch]
                    reviews.append(teacher.ask("card_review", {"group": summary, "evidence": batch, "proposal": draft},
                                               lambda r: validate_card_review(r, ids)))
                tiers = {r["quality_tier"] for r in reviews}
                review = {"quality_tier": "reject" if "reject" in tiers else "conditional" if "conditional" in tiers else "primary",
                          "member_batch_reviews": reviews, "support_ids": group["members"]}
            else:
                review = {"quality_tier": "reject", "reason": draft["reason"], "issues": ["incoherent_group"],
                          "support_ids": group["members"]}
            descriptor = render_card(draft["card"]) if draft["coherent"] else None
            body = {"schema_version": "memgen-local-memory-card-v1", "group_id": group["group_id"],
                    "evidence_ids": group["members"], "source_sample_ids": sorted({r["sample_id"] for r in members}),
                    "card": draft.get("card") if draft["coherent"] else None,
                    "descriptor": descriptor, "quality_tier": review["quality_tier"],
                    "construction_review": review, "generalization": {k: draft.get(k) for k in
                        ("substitution_check", "near_miss")},
                    "quality_basis": "same_local_teacher_semantic_review",
                    "downstream_effectiveness_proven": False, "profile_sha256": store.profile_hash}
            body["bank_id"] = "v43-bank-" + digest(body)
            body["record_sha256"] = digest(body)
            record = store.put(key, body, inputs)
        keys.append(key)
        counts[record["quality_tier"]] += 1
        print(f"[local-bank] cards={i + 1}/{len(groups['groups'])} tier={record['quality_tier']}", flush=True)
    return store.put("stages/cards", {"keys": keys, "tier_counts": dict(counts),
        "candidate_count": len(keys), "primary_only_for_runtime": True}, {"groups": digest(groups)})
