"""Exactly one greedy and seven sampled trajectories per train question."""
from __future__ import annotations

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import diagnose_gsm8k_completion
from .artifacts import digest
from .dataset import selected_rows


def rollout_plan(row, config):
    return [{"rollout_id": f"{row['sample_id']}-{i}", "sample_id": row["sample_id"],
             "index": i, "sampling": i > 0,
             "seed": int(digest([config.sampling_seed, row["sample_id"], i])[:8], 16)} for i in range(8)]


def run_rollouts(store, config, model_factory):
    split = store.require("split")
    rows = selected_rows(split, "train", config)
    index, model = [], None
    try:
        for position, row in enumerate(rows):
            if row["split"] != "train":
                raise ValueError("Only train rows may produce construction evidence")
            for plan in rollout_plan(row, config):
                key = "rollouts/" + plan["rollout_id"]
                inputs = {"sample": row, "plan": plan}
                record = store.get(key, inputs)
                if record is None:
                    if model is None:
                        model = model_factory()
                    result = model.generate(GSM8K_PROMPT_CONTRACT.render(model.tokenizer, row["question"]),
                        seed=plan["seed"], max_new_tokens=config.max_new_tokens, sampling=plan["sampling"],
                        temperature=config.temperature, top_p=config.top_p, top_k=config.top_k, stop_on_box=True)
                    record = store.put(key, {**plan, "generation": result,
                        "verifier": diagnose_gsm8k_completion(result["text"], row["scoring_solution"])}, inputs)
                index.append(key)
            print(f"[local-bank] rollouts question={position + 1}/{len(rows)}", flush=True)
    finally:
        if model is not None:
            model.close()
    return store.put("stages/rollouts", {"keys": index, "question_count": len(rows), "rollout_count": len(index)},
                     {"split": digest(split), "configuration": config.to_dict()})
