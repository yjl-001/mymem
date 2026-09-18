"""Builder-aligned data lineage. No held-out row enters teacher construction."""
from __future__ import annotations

from data.gsm8k.splits import split_gsm8k
from .artifacts import digest


def prepare_split(raw, config, revision):
    indexed = {name: ds.add_column("source_index", list(range(len(ds)))) for name, ds in raw.items()}
    splits = split_gsm8k(indexed, val_ratio=config.val_ratio, seed=config.split_seed)
    result = {"dataset": config.dataset, "revision": revision, "seed": config.split_seed,
              "val_ratio": config.val_ratio, "splits": {}}
    owners = {}
    for name, rows in splits.items():
        values = []
        for row in rows:
            question, answer = row["question"].strip(), row["answer"].strip()
            if not question or "####" not in answer:
                raise ValueError("Malformed GSM8K source row")
            qhash = digest(question)
            if qhash in owners and owners[qhash] != name:
                raise ValueError(f"Duplicate question crosses splits: {name}/{owners[qhash]}")
            owners[qhash] = name
            source_split = "test" if name == "test" else "train"
            rationale, final = answer.rsplit("####", 1)
            values.append({"sample_id": f"gsm8k-{source_split}-{row['source_index']}",
                           "split": name, "source_index": row["source_index"],
                           "question": question, "question_sha256": qhash,
                           "official_solution": answer,
                           "scoring_solution": rationale.strip() + "\\boxed{" + final.strip() + "}",
                           "answer_sha256": digest(answer)})
        result["splits"][name] = values
    result["counts"] = {k: len(v) for k, v in result["splits"].items()}
    return result


def selected_rows(split, name, config):
    limit = config.train_limit if name == "train" else config.valid_limit
    rows = split["splits"][name]
    return rows[:limit] if limit else rows


def run_split(store, config, profile):
    inputs = {"dataset": profile["dataset"], "val_ratio": config.val_ratio, "seed": config.split_seed}
    existing = store.get("split", inputs)
    if existing is not None:
        return existing
    from datasets import load_dataset
    raw = load_dataset(config.dataset, "main", revision=profile["dataset"]["revision"])
    return store.put("split", prepare_split(raw, config, profile["dataset"]["revision"]), inputs)
