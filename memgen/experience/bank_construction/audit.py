"""Read-only integrity audit of a complete construction and validation run."""
from __future__ import annotations

from .artifacts import digest, read_json
from .compilation import select_bank, validate_compiled
from .dataset import selected_rows
from .evaluation import metrics
from .pipeline import STAGES
from .rollouts import rollout_plan


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
    results = {action: [] for action in actions}
    choices = []
    for row in selected_rows(split, "valid", config):
        choice = store.require("valid_choices/" + row["sample_id"])
        if choice["bank_id"] not in actions:
            raise ValueError("Invalid saved selector action")
        selected, score = select_bank(choice["query_vector"], bundle["entries"])
        if choice["bank_id"] != (selected or "no_memory") or choice["cosine"] != score:
            raise ValueError("Saved selector choice differs from the frozen key vectors")
        choices.append(choice["bank_id"])
        for action in actions:
            results[action].append(store.require("valid_results/" + row["sample_id"] + "/" + action))
    report = store.require("stages/evaluate")
    if report["baseline"] != metrics(results["no_memory"], results["no_memory"]):
        raise ValueError("Baseline aggregate mismatch")
    for action in actions - {"no_memory"}:
        if report["per_bank"][action] != metrics(results[action], results["no_memory"]):
            raise ValueError("Per-bank aggregate mismatch")
    selected_metrics = metrics([results[action][i] for i, action in enumerate(choices)], results["no_memory"])
    if any(report["semantic_top1"][k] != v for k, v in selected_metrics.items()):
        raise ValueError("Selector aggregate mismatch")
    if check_summary:
        brief = read_json(store.root / "brief_summary.json")
        if (brief["summary_sha256"] != digest({k: v for k, v in brief.items() if k != "summary_sha256"})
                or brief["profile_sha256"] != store.profile_hash or brief["validation"] != store.require("stages/evaluate")):
            raise ValueError("Summary identity mismatch")
    return {"complete": True, "checked_checkpoints": checked, "bank_count": bundle["bank_count"]}
