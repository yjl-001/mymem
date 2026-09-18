"""Stage orchestration with lazy model lifetimes and complete-run reporting."""
from __future__ import annotations

from .artifacts import atomic_json, digest
from .cards import run_cards
from .compilation import run_compile
from .dataset import run_split
from .evaluation import run_evaluate
from .experiences import run_experiences
from .grouping import run_grouping
from .review import run_review
from .rollouts import run_rollouts
from .teacher import Teacher

STAGES = ("split", "rollouts", "review", "evidence", "groups", "cards", "compile", "evaluate")


def run(store, config, profile, *, stage="all", reasoner_factory=None, teacher_factory=None, reuse_from=None):
    if stage not in (*STAGES, "all"):
        raise ValueError("Unknown stage")
    if reasoner_factory is None or teacher_factory is None:
        from memgen.model.local_bank import LocalModel
        reasoner_factory = reasoner_factory or (lambda: LocalModel(profile["reasoner"], config.reasoner, reasoner=True))
        if teacher_factory is None:
            if config.teacher_backend == "vllm":
                from .vllm_teacher import VLLMTeacher
                teacher_factory = lambda: VLLMTeacher(store, config, profile["teacher"])
            else:
                teacher_factory = lambda: LocalModel(profile["teacher"], config.teacher)
    teacher = Teacher(store, config, teacher_factory)
    selected = STAGES if stage == "all" else (stage,)
    try:
        if config.teacher_backend == "vllm" and stage == "all" and any(
                store.get("stages/" + name) is None for name in ("review", "evidence", "groups", "cards")):
            print("[local-bank] checking local vLLM teacher identity and version", flush=True)
            teacher.prepare()  # Fail before expensive rollout collection if the service is misconfigured.
        for name in selected:
            print(f"[local-bank] stage={name}", flush=True)
            if name in {"compile", "evaluate"}:
                teacher.close()  # Release an in-process teacher; vLLM remains operator-owned.
            if name == "split":
                run_split(store, config, profile)
            elif name == "rollouts":
                if reuse_from is not None:
                    from .reuse import reuse_rollouts
                    reuse_rollouts(store, config, profile, reuse_from)
                run_rollouts(store, config, reasoner_factory)
            elif name == "review":
                run_review(store, teacher)
            elif name == "evidence":
                run_experiences(store, teacher)
            elif name == "groups":
                run_grouping(store, teacher, config)
            elif name == "cards":
                run_cards(store, teacher, config)
            elif name == "compile":
                run_compile(store, config, reasoner_factory)
            elif name == "evaluate":
                run_evaluate(store, config, reasoner_factory)
    finally:
        teacher.close()
    if all(store.get("split" if s == "split" else "stages/" + s) is not None for s in STAGES):
        cards, compiled = store.require("stages/cards"), store.require("stages/compile")
        report = {"complete": True, "profile_sha256": store.profile_hash,
            "status": "complete" if compiled["bank_count"] else "complete_without_primary_banks",
            "split_counts": store.require("split")["counts"],
            "rollouts": {k: v for k, v in store.require("stages/rollouts").items() if k != "keys"},
            "reused_rollout_count": (store.get("imports/rollouts") or {}).get("rollout_count", 0),
            "review_outcomes": store.require("stages/review")["outcomes"],
            "evidence_counts": store.require("stages/evidence")["counts"],
            "candidate_bank_count": cards["candidate_count"], "tier_counts": cards["tier_counts"],
            "compiled_bank_count": compiled["bank_count"],
            "teacher": {k: v for k, v in profile["teacher"].items() if k != "files"},
            "external_inference_api_calls": 0, "native_prefix_kv_frozen": True,
            "validation": store.require("stages/evaluate")}
        from .audit import audit_complete
        audit_complete(store, config, check_summary=False)
        atomic_json(store.root / "brief_summary.json", {**report, "summary_sha256": digest(report)})
        print(f"[local-bank] complete summary={store.root / 'brief_summary.json'}", flush=True)
