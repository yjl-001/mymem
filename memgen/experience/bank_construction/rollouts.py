"""Exactly one greedy and seven sampled trajectories per train question."""
from __future__ import annotations

from time import perf_counter
from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import diagnose_gsm8k_completion
from .artifacts import digest
from .dataset import selected_rows


def rollout_plan(row, config):
    return [{"rollout_id": f"{row['sample_id']}-{i}", "sample_id": row["sample_id"],
             "index": i, "sampling": i > 0,
             "seed": int(digest([config.sampling_seed, row["sample_id"], i])[:8], 16)}
            for i in range(config.greedy_rollouts + config.sampled_rollouts)]


def validate_rollout_record(value, row, plan, config):
    if any(value.get(name) != item for name, item in plan.items()):
        raise ValueError("Rollout plan mismatch")
    generation = value["generation"]
    if (generation["seed"] != plan["seed"]
            or generation["token_count"] != len(generation["token_ids"])
            or not 0 < generation["token_count"] <= config.max_new_tokens
            or generation["stop_reason"] not in {"length", "eos", "completed_boxed_answer"}
            or generation["truncated"] != (generation["stop_reason"] == "length")
            or (generation["truncated"] and generation["token_count"] != config.max_new_tokens)):
        raise ValueError("Rollout generation contract mismatch")
    expected = diagnose_gsm8k_completion(generation["text"], row["scoring_solution"])
    if value["verifier"] != expected:
        raise ValueError("Rollout verifier mismatch")
    return value


def run_rollouts(store, config, model_factory):
    split = store.require("split")
    rows = selected_rows(split, "train", config)
    index, pending, model = [], [], None
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Only train rows may produce construction evidence")
        for plan in rollout_plan(row, config):
            key = "rollouts/" + plan["rollout_id"]
            inputs = {"sample": row, "plan": plan}
            index.append(key)
            if store.get(key, inputs) is None:
                pending.append((row, plan, key, inputs))
    completed = len(index) - len(pending)
    print(f"[local-bank] rollouts cached={completed}/{len(index)} batch_size={config.rollout_batch_size}", flush=True)
    try:
        for offset in range(0, len(pending), config.rollout_batch_size):
            batch = pending[offset:offset + config.rollout_batch_size]
            if model is None:
                model = model_factory()
            requests = [{"prompt": GSM8K_PROMPT_CONTRACT.render(model.tokenizer, row["question"]),
                         "seed": plan["seed"], "sampling": plan["sampling"]} for row, plan, _, _ in batch]
            kwargs = dict(max_new_tokens=config.max_new_tokens, temperature=config.temperature,
                          top_p=config.top_p, top_k=config.top_k, stop_on_box=True)
            started = perf_counter()
            if hasattr(model, "generate_batch"):
                results = model.generate_batch(requests, **kwargs)
            else:  # Lightweight fixture/custom adapters may only implement single generation.
                results = [model.generate(**r, **kwargs) for r in requests]
            if len(results) != len(batch):
                raise ValueError("Batch returned the wrong number of trajectories")
            elapsed = perf_counter() - started
            tokens = sum(r["token_count"] for r in results)
            for (row, plan, key, inputs), result in zip(batch, results):
                store.put(key, {**plan, "generation": result,
                    "verifier": diagnose_gsm8k_completion(result["text"], row["scoring_solution"])}, inputs)
            completed += len(batch)
            print(f"[local-bank] rollouts completed={completed}/{len(index)} batch={len(batch)} "
                  f"generation_seconds={elapsed:.2f} generated_tokens_per_second={tokens / max(elapsed, 1e-9):.2f}", flush=True)
    finally:
        if model is not None:
            model.close()
    return store.put("stages/rollouts", {"keys": index, "question_count": len(rows), "rollout_count": len(index)},
                     {"split": digest(split), "configuration": config.to_dict()})
