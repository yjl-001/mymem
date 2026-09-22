"""Frozen V5 online selector and one-Bank native-prefix consumer."""
from __future__ import annotations

from .compilation import primary_records, validate_compiled
from .selector import candidate_features, reranker_document, retrieve, select


def prepare_bank(store):
    bundle = validate_compiled(store, store.require("stages/compile"))
    policy = store.require("stages/calibrate")["policy"]
    records = {record["bank_id"]: record for record in primary_records(store)}
    if set(records) != {entry["bank_id"] for entry in bundle["entries"]}:
        raise ValueError("V5 runtime card/compiled namespaces differ")
    return bundle, policy, records


def choose_memory(question, runtime, reranker, bundle, policy, top_k):
    from memgen.model.v4_3_question_selector import encode_text
    if policy.get("selector_top_k") not in {None, top_k}:
        raise ValueError("V5 runtime Top-K differs from the frozen selector policy")
    if policy.get("reranker_enabled") not in {None, reranker is not None}:
        raise ValueError("V5 runtime reranker mode differs from the frozen selector policy")
    vector = encode_text(runtime, question)
    ranked = retrieve(vector, bundle["entries"], top_k)
    scores = ({entry["bank_id"]: reranker.score(question, reranker_document(entry))
               for _, _, entry in ranked} if reranker is not None else {})
    candidates = candidate_features(ranked, scores)
    return select(candidates, policy), candidates, scores


def answer(question, store, reasoner, reranker, bundle, policy, records, top_k):
    from .prefix import prefix_bank
    from memgen.model.v4_3_question_selector import generate
    decision, candidates, scores = choose_memory(question, reasoner.runtime, reranker,
                                                  bundle, policy, top_k)
    record = records.get(decision["bank_id"])
    memory = (prefix_bank(store.root / "prefix_kv", record, reasoner.runtime,
                          store.profile_hash, validate_only=True) if record else None)
    _, generated = generate(reasoner.runtime, question, record["descriptor"] if record else None, memory)
    text = reasoner.tokenizer.decode(generated["continuation_token_ids"], skip_special_tokens=True)
    return {"decision": decision, "candidates": candidates, "reranker_scores": scores,
            "output": text, "continuation_token_ids": generated["continuation_token_ids"],
            "generated_token_count": len(generated["continuation_token_ids"]),
            "stop_reason": generated["stop_reason"]}
