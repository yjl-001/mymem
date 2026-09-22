"""Task execution protocol. Memory construction consumes only its generic records."""
from __future__ import annotations

from dataclasses import dataclass

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.gsm8k.splits import split_gsm8k
from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.experience.bank_construction.artifacts import digest


@dataclass(frozen=True)
class GSM8KTask:
    name: str = "gsm8k"

    def prepare_split(self, raw, config, revision):
        indexed = {name: ds.add_column("source_index", list(range(len(ds)))) for name, ds in raw.items()}
        splits = split_gsm8k(indexed, val_ratio=config.val_ratio, seed=config.split_seed)
        result = {"task": self.name, "dataset": config.dataset, "revision": revision,
                  "seed": config.split_seed, "val_ratio": config.val_ratio, "splits": {}}
        owners = {}
        for name, rows in splits.items():
            values = []
            for row in rows:
                question, answer = row["question"].strip(), row["answer"].strip()
                if not question or "####" not in answer:
                    raise ValueError("Malformed GSM8K source row")
                ihash = digest(question)
                if ihash in owners and owners[ihash] != name:
                    raise ValueError("Duplicate input crosses splits")
                owners[ihash] = name
                source_split = "test" if name == "test" else "train"
                rationale, final = answer.rsplit("####", 1)
                values.append({"input_id": f"gsm8k-{source_split}-{row['source_index']}",
                    "split": name, "source_index": row["source_index"], "input": question,
                    "input_sha256": ihash, "reference": answer,
                    "scoring_reference": rationale.strip() + "\\boxed{" + final.strip() + "}",
                    "reference_sha256": digest(answer)})
            result["splits"][name] = values
        result["counts"] = {name: len(rows) for name, rows in result["splits"].items()}
        return result

    def prompt(self, tokenizer, row):
        return GSM8K_PROMPT_CONTRACT.render(tokenizer, row["input"])

    def verify(self, output, row):
        result = diagnose_gsm8k_completion(output, row["scoring_reference"])
        return {"reward": result["reward"], "answer_correct": result["reward"] == 1.,
                "format_correct": result["format_valid"], "details": result}

    @property
    def stop_on_box(self):
        return True


def task_for(config):
    if config.task == "gsm8k":
        return GSM8KTask()
    raise ValueError("Unregistered V5 task protocol: " + config.task)


def selected_rows(split, name, config):
    limit = config.train_limit if name == "train" else config.valid_limit
    rows = split["splits"][name]
    return rows[:limit] if limit else rows


def run_split(store, config, profile, task):
    inputs = {"dataset": profile["dataset"], "task": config.task,
              "val_ratio": config.val_ratio, "seed": config.split_seed}
    cached = store.get("split", inputs)
    if cached is not None:
        return cached
    from datasets import load_dataset
    raw = load_dataset(config.dataset, "main", revision=profile["dataset"]["revision"])
    return store.put("split", task.prepare_split(raw, config, profile["dataset"]["revision"]), inputs)
