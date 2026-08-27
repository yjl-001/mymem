#!/usr/bin/env python3
"""Audit V3.4 current-token first-attempt retrieval geometry answer-blindly."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3 import (
    V34_QUERY_POOLING_CURRENT_TOKEN,
    V34_SYSTEM_PROFILE_SCHEMA,
)
from memgen.experience.v3_selector import numeric_summary, selection_concentration
from scripts.calibrate_v3_margin_selector import (
    EVALUATION_PROFILE_SCHEMA,
    EVALUATION_ROW_SCHEMA,
    evaluation_profile_sha256,
)


REPORT_SCHEMA = "experience-memory-v3.4-current-token-geometry-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-profile", type=Path, required=True)
    parser.add_argument("--retrieval-key-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-first-attempts", type=int, default=32)
    return parser.parse_args()


def load_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    system = value.get("system_profile", {})
    if (
        value.get("schema_version") != EVALUATION_PROFILE_SCHEMA
        or value.get("profile_sha256") != evaluation_profile_sha256(value)
        or value.get("logical_split") != "calibration-val"
        or value.get("system_version") != "v3.4"
        or system.get("schema_version") != V34_SYSTEM_PROFILE_SCHEMA
        or system.get("query_pooling") != V34_QUERY_POOLING_CURRENT_TOKEN
        or system.get("risk_role") != "online_joint_control"
        or system.get("boundary_policy")
        != "none_pre_answer_every_generated_token"
        or system.get("retrieval_abstention_policy") != "disabled"
    ):
        raise ValueError("Geometry audit requires a raw V3.4 calibration run")
    return value


def load_memory_ids(path: Path, *, expected_sha256: str) -> list[str]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("Retrieval key bank differs from the source run")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("manifest_sha256")
    actual = canonical_json_sha256({
        key: item for key, item in value.items() if key != "manifest_sha256"
    })
    memory_ids = [str(item.get("memory_id", "")) for item in value.get("records", [])]
    if (
        expected != actual
        or not memory_ids
        or any(not memory_id for memory_id in memory_ids)
        or len(set(memory_ids)) != len(memory_ids)
    ):
        raise ValueError("Invalid retrieval key manifest")
    return memory_ids


def collect(
    path: Path, *, profile_sha256: str
) -> tuple[int, list[float], list[str], int, int]:
    row_count = 0
    margins: list[float] = []
    memory_ids: list[str] = []
    included_count = 0
    top1_reproduction_count = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            actual_hash = canonical_json_sha256({
                key: item
                for key, item in row.items()
                if key not in {"created_at", "row_sha256"}
            })
            if (
                row.get("schema_version") != EVALUATION_ROW_SCHEMA
                or row.get("profile_sha256") != profile_sha256
                or row.get("row_sha256") != actual_hash
                or not sample_id
                or sample_id in seen
            ):
                raise ValueError(f"Invalid source row at line {line_number}")
            seen.add(sample_id)
            row_count += 1
            attempts = row.get("conditions", {}).get("v3", {}).get(
                "runtime_trace", {}
            ).get("retrieval_attempts", [])
            if not attempts:
                continue
            first = attempts[0]
            decision = first.get("retrieval_decision", {})
            query = decision.get("query", {})
            hits = list(decision.get("hits", []))
            margin = query.get("top1_top2_margin")
            selected_id = first.get("selected_memory_id")
            if (
                decision.get("status") != "selected"
                or query.get("pooling") != V34_QUERY_POOLING_CURRENT_TOKEN
                or len(hits) < 2
                or selected_id is None
                or margin is None
                or not math.isfinite(float(margin))
                or float(margin) < 0.0
            ):
                raise ValueError(
                    f"Invalid raw V3.4 first attempt for {sample_id}"
                )
            margins.append(float(margin))
            memory_ids.append(str(selected_id))
            included_count += int(
                query.get("trigger_observation_included_in_pooling") is True
                and int(query.get("query_embedding_token_index", -1))
                == int(query.get("query_token_count", -1)) - 1
            )
            top1_reproduction_count += int(
                str(hits[0].get("memory_id", "")) == str(selected_id)
            )
    return (
        row_count,
        margins,
        memory_ids,
        included_count,
        top1_reproduction_count,
    )


def markdown(value: Mapping[str, Any]) -> str:
    geometry = value["geometry"]
    concentration = geometry["selection_concentration"]
    return "\n".join([
        "# MemGen V3.4 current-token retrieval geometry audit",
        "",
        f"- Status: `{value['status']}`",
        f"- Completed calibration samples: {value['source']['completed_sample_count']}",
        f"- First attempts: {geometry['first_attempt_count']}",
        f"- Current-token pooling reproduction: {geometry['current_token_pooling_reproduction_count']} / {geometry['first_attempt_count']}",
        f"- Top-1 reproduction: {geometry['top1_reproduction_count']} / {geometry['first_attempt_count']}",
        f"- First-memory top-1 share: {concentration['top1_share']}",
        f"- Selection Gini: {concentration['gini']}",
        f"- Selected memories: {concentration['selected_memory_count']}",
        f"- Median margin: {geometry['top1_top2_margin']['median']}",
        "",
        "This audit is answer-blind and reads retrieval traces but not task accuracy, answers, or rewards.",
        "",
    ])


def main() -> None:
    args = parse_args()
    if args.minimum_first_attempts <= 0:
        raise ValueError("minimum-first-attempts must be positive")
    profile = load_profile(args.run_profile)
    complete_memory_ids = load_memory_ids(
        args.retrieval_key_manifest,
        expected_sha256=str(
            profile.get("inputs", {}).get("retrieval_key_manifest_sha256", "")
        ),
    )
    row_count, margins, memory_ids, included, top1_reproduced = collect(
        args.results, profile_sha256=str(profile["profile_sha256"])
    )
    first_attempt_count = len(margins)
    passed = (
        row_count == int(profile.get("selected_sample_count", -1))
        and first_attempt_count >= args.minimum_first_attempts
        and included == first_attempt_count
        and top1_reproduced == first_attempt_count
    )
    value: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "not_qualified",
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "source": {
            "logical_split": "calibration-val",
            "completed_sample_count": row_count,
            "run_profile_sha256": profile["profile_sha256"],
            "run_profile_file_sha256": file_sha256(args.run_profile),
            "results_file_sha256": file_sha256(args.results),
            "retrieval_key_manifest_sha256": file_sha256(
                args.retrieval_key_manifest
            ),
        },
        "geometry": {
            "query_pooling": V34_QUERY_POOLING_CURRENT_TOKEN,
            "first_attempt_count": first_attempt_count,
            "minimum_first_attempts": args.minimum_first_attempts,
            "current_token_pooling_reproduction_count": included,
            "top1_reproduction_count": top1_reproduced,
            "top1_top2_margin": numeric_summary(margins) if margins else None,
            "selection_concentration": selection_concentration(
                memory_ids, complete_memory_ids=complete_memory_ids
            ),
        },
        "requirements": {
            "source_profile_authenticated": True,
            "source_rows_authenticated_and_complete": (
                row_count == int(profile.get("selected_sample_count", -1))
            ),
            "minimum_first_attempts_met": (
                first_attempt_count >= args.minimum_first_attempts
            ),
            "current_token_pooling_reproduced": included == first_attempt_count,
            "top1_reproduced": top1_reproduced == first_attempt_count,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
        },
    }
    value["report_sha256"] = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "report_sha256"}
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(markdown(value), encoding="utf-8")
    if not passed:
        raise RuntimeError("V3.4 current-token retrieval geometry did not qualify")
    print(
        f"[v3.4-geometry] passed first_attempts={first_attempt_count} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
