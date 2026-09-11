#!/usr/bin/env python3
"""Route a new question with the frozen selector; no dataset or answer access."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from memgen.experience.v4_3_artifacts import read_json, implementation_hashes
from memgen.experience.v4_3_bank import authenticate
from memgen.experience.v4_3_question_selector import predict
from scripts.run_v4_3_question_selector import IMPLEMENTATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    profile = read_json(args.selector_dir / "profile.json")
    selector = read_json(args.selector_dir / "selector.json")
    authenticate(profile, "profile_sha256", "selector experiment")
    authenticate(selector, "selector_sha256", "selector")
    if selector["profile_sha256"] != profile["profile_sha256"] or profile["implementation_sha256"] != implementation_hashes(IMPLEMENTATION):
        raise ValueError("Selector code/profile binding drift")
    from memgen.model.v4_3_side_kv import runtime_versions
    import numpy as np
    if runtime_versions() != profile["runtime_versions"] or np.__version__ != profile["numpy_version"]:
        raise ValueError("Question encoder runtime versions differ")
    from memgen.model.v4_3_question_selector import load_runtime, encode_text
    runtime = load_runtime(profile["reasoner"], args.device)
    try:
        decision = predict(selector, encode_text(runtime, args.question))
        print(json.dumps(decision, ensure_ascii=False, indent=2))
    finally:
        runtime.controller.close()


if __name__ == "__main__":
    main()
