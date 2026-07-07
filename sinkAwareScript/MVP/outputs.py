import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from records import StrategyRecord


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """追加写 JSONL。

    长实验可能跑很久，所以每个样本结束就落盘，避免中途失败丢掉所有结果。
    """

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def append_strategy_csv(path: Path, rows: list[StrategyRecord]) -> None:
    """额外写一个轻量 CSV，方便快速看 strategy reward。"""

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "reference_mode", "strategy", "reward", "planned_steps", "inserted_steps"],
        )
        if f.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_idx": row.sample_idx,
                    "reference_mode": row.reference_mode,
                    "strategy": row.strategy,
                    "reward": row.reward,
                    "planned_steps": json.dumps(row.planned_steps),
                    "inserted_steps": json.dumps(row.inserted_steps),
                }
            )


def write_dataclass_jsonl(path: Path, rows) -> None:
    """dataclass 列表 -> dict -> JSONL。"""

    write_jsonl(path, [asdict(row) for row in rows])


def write_summary(path: Path, summary: dict) -> None:
    """覆盖写当前累计 summary。"""

    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
