#!/usr/bin/env python3
"""Answer one question using the constructed question-only semantic top-1 Bank."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from memgen.experience.bank_construction.artifacts import Store, read_json
    from memgen.experience.bank_construction.config import ModelConfig
    from memgen.experience.bank_construction.compilation import primary_records, select_bank, validate_compiled
    from memgen.model.local_bank import LocalModel
    from memgen.model.v4_3_question_selector import encode_text, generate
    from memgen.model.v4_3_prefix_equivalence import prefix_bank
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", help="Override only the reasoner device, not dtype/model/consumer")
    args = parser.parse_args()
    if not args.question.strip():
        raise ValueError("Question cannot be empty")
    profile = read_json(args.bank_dir / "profile.json")
    store = Store(args.bank_dir, profile)
    bundle = validate_compiled(store, store.require("stages/compile"))
    settings = ModelConfig(**profile["configuration"]["reasoner"])
    if args.device:
        settings = replace(settings, device=args.device)
    model = LocalModel(profile["reasoner"], settings, reasoner=True)
    try:
        bid, score = select_bank(encode_text(model.runtime, args.question), bundle["entries"])
        record = next((r for r in primary_records(store) if r["bank_id"] == bid), None)
        memory = None if record is None else prefix_bank(store.root / "prefix_kv", record, model.runtime, store.profile_hash, validate_only=True)
        _, output = generate(model.runtime, args.question, None if record is None else record["descriptor"], memory)
        print(json.dumps({"bank_id": bid, "cosine": score, "selector": "semantic_top1_no_abstention_reference",
            "answer": model.tokenizer.decode(output["continuation_token_ids"], skip_special_tokens=True),
            "generated_token_count": len(output["continuation_token_ids"]), "stop_reason": output["stop_reason"]}, ensure_ascii=False, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
