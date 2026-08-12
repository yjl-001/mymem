#!/usr/bin/env python3
"""Build the frozen, leak-checked GSM8K split manifest used by Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import create_gsm8k_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-source-size", type=int, default=6000)
    parser.add_argument("--calibration-val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required; install requirements.txt first") from exc

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        revision=args.dataset_revision,
    )
    manifest = create_gsm8k_split_manifest(
        dataset["train"],
        dataset["test"],
        bank_source_size=args.bank_source_size,
        calibration_val_size=args.calibration_val_size,
        seed=args.seed,
        dataset_revision=args.dataset_revision,
        train_fingerprint=getattr(dataset["train"], "_fingerprint", None),
        test_fingerprint=getattr(dataset["test"], "_fingerprint", None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[split-manifest] {args.output} sha256={manifest['manifest_sha256']} "
        f"counts={manifest['counts']}"
    )


if __name__ == "__main__":
    main()
