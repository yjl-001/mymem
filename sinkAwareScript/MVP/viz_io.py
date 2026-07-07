import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strategy_family(strategy: str) -> str:
    if strategy.startswith("random_"):
        return "random"
    if strategy.startswith("same_bucket_random_"):
        return "same_bucket_random"
    return strategy


def family_means(summary: dict, key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for strategy, value in summary.get(key, {}).items():
        if value is None:
            continue
        grouped.setdefault(strategy_family(strategy), []).append(float(value))
    return {name: sum(values) / len(values) for name, values in grouped.items()}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
