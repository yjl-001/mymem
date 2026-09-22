"""Valid-only candidate execution and expected-utility selector calibration."""
from __future__ import annotations

from collections import Counter
import math
import statistics

from memgen.experience.bank_construction.artifacts import digest
from .compilation import primary_records, validate_compiled
from .selector import (NO_MEMORY, candidate_features, fit_policy, metrics, reranker_document,
                       retrieve, select)
from .task import selected_rows


def token_statistics(values, limit=1024):
    values = sorted(values)
    if not values:
        return {"total": 0, "mean": None, "median": None, "min": None, "max": None,
                "p90": None, "at_limit": 0}
    return {"total": sum(values), "mean": statistics.mean(values), "median": statistics.median(values),
        "min": values[0], "max": values[-1], "p90": values[math.ceil(.9*len(values))-1],
        "at_limit": sum(value == limit for value in values)}


def run_calibration(store, config, task, model_factory, reranker_factory=None):
    split = store.require("split")
    bundle = validate_compiled(store, store.require("stages/compile"))
    rows = selected_rows(split, "valid", config)
    inputs = {"split_sha256": digest(split), "bundle_sha256": digest(bundle),
              "input_ids": [row["input_id"] for row in rows], "selector_top_k": config.selector_top_k,
              "reranker_enabled": config.reranker_enabled, "utility_ridge": config.utility_ridge,
              "utility_folds": config.utility_folds}
    cached = store.get("stages/calibrate", inputs)
    if cached is not None:
        return cached
    from .prefix import prefix_bank
    from memgen.model.v4_3_question_selector import encode_text, generate
    records = {record["bank_id"]: record for record in primary_records(store)}
    reasoner, reranker = model_factory(), None
    observations, generation_tokens, reranker_cost = [], [], {"pairs": 0, "input_tokens": 0,
                                                               "seconds": 0.}
    try:
        # Freeze question-only candidate features before any outcome is scored.
        frozen = {}
        for index, row in enumerate(rows, start=1):
            key = "calibration/candidates/" + row["input_id"]
            binding = {"input": row["input"], "bundle_sha256": digest(bundle),
                       "top_k": config.selector_top_k, "reranker": config.reranker_enabled}
            value = store.get(key, binding)
            if value is None:
                vector = encode_text(reasoner.runtime, row["input"])
                ranked = retrieve(vector, bundle["entries"], config.selector_top_k)
                scores = {}
                if config.reranker_enabled and ranked:
                    if reranker_factory is None:
                        raise ValueError("V5 reranker is enabled but no factory was provided")
                    if reranker is None:
                        reranker = reranker_factory()
                    for _, bank_id, entry in ranked:
                        scores[bank_id] = reranker.score(row["input"], reranker_document(entry))
                candidates = candidate_features(ranked, scores)
                value = store.put(key, {"query_vector": vector, "candidates": candidates,
                    "reranker_scores": scores}, binding)
            frozen[row["input_id"]] = value
            for score in value["reranker_scores"].values():
                reranker_cost["pairs"] += 1
                reranker_cost["input_tokens"] += score["input_tokens"]
                reranker_cost["seconds"] += score["elapsed_seconds"]
            print(f"[v5] selector_features={index}/{len(rows)}", flush=True)

        def generate_action(row, bank_id):
            action = bank_id or NO_MEMORY
            record = records.get(bank_id)
            binding = {"input": row, "action": action,
                       "record_sha256": record["record_sha256"] if record else None,
                       "bundle_sha256": digest(bundle)}
            key = "calibration/results/" + row["input_id"] + "/" + action
            result = store.get(key, binding)
            if result is None:
                memory = (prefix_bank(store.root / "prefix_kv", record, reasoner.runtime,
                                      store.profile_hash, validate_only=True) if record else None)
                _, generated = generate(reasoner.runtime, row["input"],
                                        record["descriptor"] if record else None, memory)
                text = reasoner.tokenizer.decode(generated["continuation_token_ids"], skip_special_tokens=True)
                outcome = task.verify(text, row)
                result = store.put(key, {"input_id": row["input_id"], "action": action, "output": text,
                    "reward": int(outcome["reward"]), "verifier": outcome,
                    "generated_token_count": len(generated["continuation_token_ids"]),
                    "stop_reason": generated["stop_reason"]}, binding)
            return result

        for index, row in enumerate(rows, start=1):
            baseline = generate_action(row, None)
            generation_tokens.append(baseline["generated_token_count"])
            candidates = []
            for features in frozen[row["input_id"]]["candidates"]:
                result = generate_action(row, features["bank_id"])
                generation_tokens.append(result["generated_token_count"])
                utility = int(result["reward"] == 1 and baseline["reward"] == 0) - int(
                    result["reward"] == 0 and baseline["reward"] == 1)
                candidates.append({**features, "utility": utility, "reward": result["reward"],
                                   "generated_token_count": result["generated_token_count"]})
            observations.append({"input_id": row["input_id"], "baseline_reward": baseline["reward"],
                                 "baseline_generated_token_count": baseline["generated_token_count"],
                                 "candidates": candidates})
            print(f"[v5] calibration_generation={index}/{len(rows)} candidates={len(candidates)}", flush=True)
    finally:
        reasoner.close()
        if reranker is not None:
            reranker.model = None
            reranker.tokenizer = None
            try:
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    policy = fit_policy(observations, config.utility_ridge, config.utility_folds, store.profile_hash,
                        top_k=config.selector_top_k, reranker_enabled=config.reranker_enabled)
    predictions = [select(row["candidates"], policy) for row in observations]
    baseline_correct = sum(row["baseline_reward"] for row in observations)
    baseline = {"count": len(rows), "correct": baseline_correct,
                "accuracy": baseline_correct/len(rows) if rows else None, "gain": 0, "harm": 0,
                "net_gain": 0, "memory_use_count": 0,
                "generated_tokens": token_statistics([row["baseline_generated_token_count"] for row in observations])}
    semantic_predictions = []
    rerank_predictions = []
    for row in observations:
        semantic = sorted(row["candidates"], key=lambda candidate: (-candidate["semantic"], candidate["bank_id"]))[0] if row["candidates"] else None
        reranked = sorted(row["candidates"], key=lambda candidate: (-candidate["reranker"], candidate["bank_id"]))[0] if row["candidates"] else None
        semantic_predictions.append({"bank_id": semantic["bank_id"] if semantic else NO_MEMORY})
        rerank_predictions.append({"bank_id": reranked["bank_id"] if reranked else NO_MEMORY})
    report = {"evaluation_role": "valid_calibration_not_official_test", "sample_count": len(rows),
        "bank_count": len(bundle["entries"]), "official_test_used": False,
        "policy": policy, "observations": observations, "baseline": baseline,
        "semantic_top1_forced": metrics(observations, semantic_predictions),
        "reranker_top1_forced": metrics(observations, rerank_predictions),
        "v5_selector_oof": policy["oof_metrics"],
        "v5_selector_in_sample": metrics(observations, predictions),
        "v5_utility_selector": policy["oof_metrics"],
        "reranker_cost": reranker_cost, "all_generated_tokens": token_statistics(generation_tokens),
        "candidate_generation_count": sum(len(row["candidates"]) for row in observations),
        "gold_access": "only_after_question_only_candidates_were_frozen"}
    return store.put("stages/calibrate", report, inputs)
