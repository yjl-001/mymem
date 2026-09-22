"""Two-phase orchestration with lazy, phase-bounded model lifetimes."""
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

STAGES = ("split", "rollouts", "review", "evidence", "groups", "cards", "compile", "evaluate")
PHASES = {
    "rollouts": ("split", "rollouts"),
    "bank": ("split", "review", "evidence", "groups", "cards", "compile", "evaluate"),
    "all": STAGES,
}
TEACHER_STAGES = {"review", "evidence", "groups", "cards"}


def stage_complete(store, name):
    return store.get("split" if name == "split" else "stages/" + name) is not None


def write_rollout_summary(store, config):
    if not all(stage_complete(store, name) for name in PHASES["rollouts"]):
        return
    index = store.require("stages/rollouts")
    report = {"complete": True, "phase": "rollouts", "profile_sha256": store.profile_hash,
              "split_counts": store.require("split")["counts"],
              "question_count": index["question_count"], "rollout_count": index["rollout_count"],
              "rollouts_per_question": config.greedy_rollouts + config.sampled_rollouts,
              "sampling": {"greedy": config.greedy_rollouts, "sampled": config.sampled_rollouts,
                           "temperature": config.temperature, "top_p": config.top_p,
                           "top_k": config.top_k, "max_new_tokens": config.max_new_tokens},
              "teacher_inference_used": False}
    atomic_json(store.root / "rollout_summary.json", {**report, "summary_sha256": digest(report)})


def write_complete_summary(store, config, profile):
    if not all(stage_complete(store, name) for name in STAGES):
        return
    cards, compiled = store.require("stages/cards"), store.require("stages/compile")
    report = {"complete": True, "profile_sha256": store.profile_hash,
        "status": "complete" if compiled["bank_count"] else "complete_without_primary_banks",
        "split_counts": store.require("split")["counts"],
        "rollouts": {key: value for key, value in store.require("stages/rollouts").items() if key != "keys"},
        "reused_rollout_count": (store.get("imports/rollouts") or {}).get("rollout_count", 0),
        "review_outcomes": store.require("stages/review")["outcomes"],
        "evidence_counts": store.require("stages/evidence")["counts"],
        "candidate_bank_count": cards["candidate_count"], "tier_counts": cards["tier_counts"],
        "compiled_bank_count": compiled["bank_count"],
        "teacher": {key: value for key, value in profile["teacher"].items() if key != "files"},
        "external_inference_api_calls": 0, "native_prefix_kv_frozen": True,
        "validation": store.require("stages/evaluate")}
    from .audit import audit_complete
    audit_complete(store, config, check_summary=False)
    atomic_json(store.root / "brief_summary.json", {**report, "summary_sha256": digest(report)})
    print(f"[local-bank] complete summary={store.root / 'brief_summary.json'}", flush=True)


def run(store, config, profile, *, phase="all", stage=None, reasoner_factory=None,
        teacher_factory=None, rollout_source=None):
    if phase not in PHASES or stage not in (None, "all", *STAGES):
        raise ValueError("Unknown phase or stage")
    if stage not in (None, "all") and phase != "all":
        raise ValueError("Choose a phase or one recovery stage, not both")
    selected = PHASES[phase] if stage in (None, "all") else (stage,)
    needs_reasoner = any(name in {"rollouts", "groups", "compile", "evaluate"} for name in selected)
    needs_teacher = any(name in TEACHER_STAGES for name in selected)
    if (reasoner_factory is None and needs_reasoner) or (teacher_factory is None and needs_teacher):
        from memgen.model.local_bank import LocalModel
        if reasoner_factory is None and needs_reasoner:
            reasoner_factory = lambda: LocalModel(profile["reasoner"], config.reasoner, reasoner=True)
        if teacher_factory is None and needs_teacher:
            if config.teacher_backend == "vllm":
                from .vllm_teacher import VLLMTeacher
                teacher_factory = lambda: VLLMTeacher(store, config, profile["teacher"])
            else:
                teacher_factory = lambda: LocalModel(profile["teacher"], config.teacher)

    teacher = None
    def get_teacher():
        nonlocal teacher
        if teacher is None:
            if teacher_factory is None:
                raise RuntimeError("Teacher factory is unavailable outside a teacher stage")
            from .teacher import Teacher
            teacher = Teacher(store, config, teacher_factory)
        return teacher

    def ensure_rollout_input():
        if stage_complete(store, "rollouts"):
            return
        if rollout_source is None:
            raise ValueError("Bank phase requires completed rollouts; pass --rollout-source or run --phase rollouts")
        from .reuse import reuse_rollouts
        reuse_rollouts(store, config, profile, rollout_source)
        def incomplete_source():
            raise ValueError("Rollout source is incomplete; finish the rollouts phase before Bank construction")
        run_rollouts(store, config, incomplete_source)

    try:
        for name in selected:
            print(f"[local-bank] phase={phase} stage={name}", flush=True)
            if name == "split":
                run_split(store, config, profile)
            elif name == "rollouts":
                if rollout_source is not None:
                    from .reuse import reuse_rollouts
                    reuse_rollouts(store, config, profile, rollout_source)
                run_rollouts(store, config, reasoner_factory)
            elif name in TEACHER_STAGES:
                ensure_rollout_input()
                current_teacher = get_teacher()
                if name == "review":
                    run_review(store, current_teacher)
                elif name == "evidence":
                    run_experiences(store, current_teacher)
                elif name == "groups":
                    model = None
                    def encode(text):
                        nonlocal model
                        if model is None:
                            model = reasoner_factory()
                        from memgen.model.v4_3_question_selector import encode_text
                        return encode_text(model.runtime, text)
                    try:
                        run_grouping(store, current_teacher, config, encode, profile["reasoner"])
                    finally:
                        if model is not None:
                            model.close()
                elif name == "cards":
                    run_cards(store, current_teacher, config)
            elif name == "compile":
                ensure_rollout_input()
                if teacher is not None:
                    teacher.close()
                    teacher = None
                run_compile(store, config, reasoner_factory)
            elif name == "evaluate":
                ensure_rollout_input()
                if teacher is not None:
                    teacher.close()
                    teacher = None
                run_evaluate(store, config, reasoner_factory)
    finally:
        if teacher is not None:
            teacher.close()
    if profile.get("implementation_scope") == "rollouts":
        write_rollout_summary(store, config)
    write_complete_summary(store, config, profile)
