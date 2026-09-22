"""Read-only integrity audit of a complete construction and validation run."""
from __future__ import annotations

from .artifacts import digest, read_json
from .compilation import select_bank, validate_compiled
from .dataset import selected_rows
from .evaluation import metrics
from .pipeline import STAGES
from .rollouts import rollout_plan, validate_rollout_record


def audit_rollouts(store, config):
    split = store.require("split")
    index = store.require("stages/rollouts")
    expected = []
    for row in selected_rows(split, "train", config):
        for plan in rollout_plan(row, config):
            key = "rollouts/" + plan["rollout_id"]
            expected.append(key)
            value = store.require(key)
            validate_rollout_record(value, row, plan, config)
    if index["keys"] != expected or index["rollout_count"] != len(expected):
        raise ValueError("Rollout phase coverage mismatch")
    later = ("review", "evidence", "groups", "cards", "compile", "evaluate")
    if any(store.get("stages/" + name) is not None for name in later):
        raise ValueError("Rollout-only run contains Bank-phase stages")
    if (store.root / "teacher").exists() and any((store.root / "teacher").rglob("*.json")):
        raise ValueError("Rollout-only run contains teacher receipts")
    summary = read_json(store.root / "rollout_summary.json")
    unsigned = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if (summary.get("summary_sha256") != digest(unsigned)
            or summary.get("profile_sha256") != store.profile_hash
            or summary.get("rollout_count") != len(expected)
            or summary.get("teacher_inference_used") is not False):
        raise ValueError("Rollout summary mismatch")
    return {"complete": True, "phase": "rollouts", "question_count": index["question_count"],
            "rollout_count": len(expected), "teacher_inference_used": False}


def audit_complete(store, config, *, check_summary=True):
    for name in STAGES:
        store.require("split" if name == "split" else "stages/" + name)
    # Check every receipt, including rejected outputs, not only accepted stage manifests.
    checked = 0
    for folder in ("stages", "rollouts", "reviews", "evidence", "teacher", "cards", "retrieval", "valid_choices", "valid_results", "imports"):
        for path in sorted((store.root / folder).rglob("*.json")):
            store.require(str(path.relative_to(store.root))[:-5])
            checked += 1
    split = store.require("split")
    expected = {"rollouts/" + p["rollout_id"] for r in selected_rows(split, "train", config) for p in rollout_plan(r, config)}
    if expected != set(store.require("stages/rollouts")["keys"]):
        raise ValueError("Rollout plan coverage mismatch")
    for key in expected:
        store.require(key)
        store.require("reviews/" + key.split("/")[1])
    for stage in ("review", "evidence", "cards"):
        for key in store.require("stages/" + stage)["keys"]:
            store.require(key)
    bundle = validate_compiled(store, store.require("stages/compile"))
    actions = {"no_memory", *(e["bank_id"] for e in bundle["entries"])}
    rows = selected_rows(split, "valid", config)
    baseline, selected_results, choices = [], [], []
    report = store.require("stages/evaluate")
    mode = report.get("evaluation_mode", "exhaustive")
    exhaustive = mode == "exhaustive"
    if mode not in {"selector_only", "exhaustive"}:
        raise ValueError("Unknown saved evaluation mode")
    per_bank_results = {action: [] for action in actions - {"no_memory"}} if exhaustive else {}
    for row in rows:
        choice = store.require("valid_choices/" + row["sample_id"])
        if choice["bank_id"] not in actions:
            raise ValueError("Invalid saved selector action")
        selected, score = select_bank(choice["query_vector"], bundle["entries"])
        if choice["bank_id"] != (selected or "no_memory") or choice["cosine"] != score:
            raise ValueError("Saved selector choice differs from the frozen key vectors")
        choices.append(choice["bank_id"])
        base = store.require("valid_results/" + row["sample_id"] + "/no_memory")
        baseline.append(base)
        selected_results.append(base if choice["bank_id"] == "no_memory" else
                                store.require("valid_results/" + row["sample_id"] + "/" + choice["bank_id"]))
        if exhaustive:
            for action in per_bank_results:
                per_bank_results[action].append(
                    store.require("valid_results/" + row["sample_id"] + "/" + action))
    if report["baseline"] != metrics(baseline, baseline):
        raise ValueError("Baseline aggregate mismatch")
    for action, values in per_bank_results.items():
        if report["per_bank"][action] != metrics(values, baseline):
            raise ValueError("Per-bank aggregate mismatch")
    if bool(report.get("per_bank_complete")) != exhaustive or (not exhaustive and report["per_bank"]):
        raise ValueError("Per-bank evaluation completeness mismatch")
    expected_generations = len(rows) * len(actions) if exhaustive else len(rows) + sum(
        choice != "no_memory" for choice in choices)
    if report.get("generation_count") != expected_generations:
        raise ValueError("Evaluation generation count mismatch")
    selected_metrics = metrics(selected_results, baseline)
    if any(report["semantic_top1"][k] != v for k, v in selected_metrics.items()):
        raise ValueError("Selector aggregate mismatch")
    if check_summary:
        brief = read_json(store.root / "brief_summary.json")
        if (brief["summary_sha256"] != digest({k: v for k, v in brief.items() if k != "summary_sha256"})
                or brief["profile_sha256"] != store.profile_hash or brief["validation"] != store.require("stages/evaluate")):
            raise ValueError("Summary identity mismatch")
    return {"complete": True, "checked_checkpoints": checked, "bank_count": bundle["bank_count"]}
