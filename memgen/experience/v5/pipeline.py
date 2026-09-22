"""Production V5 orchestration with phase-bounded model lifetimes."""
from __future__ import annotations

from memgen.experience.bank_construction.artifacts import atomic_json, digest
from .atoms import run_atoms
from .audit import audit_complete, audit_rollouts
from .calibration import run_calibration
from .cards import run_cards
from .compilation import run_compile
from .contrasts import run_contrasts
from .grouping import run_grouping
from .review import run_review
from .rollouts import run_rollouts
from .task import run_split

STAGES = ("split", "episodes", "review", "contrasts", "atoms", "groups", "cards", "compile", "calibrate")
PHASES = {"rollouts": ("split", "episodes"),
          "bank": ("split", "review", "contrasts", "atoms", "groups", "cards", "compile", "calibrate"),
          "all": STAGES}
TEACHER_STAGES = {"review", "atoms", "groups", "cards"}


def stage_complete(store, name):
    return store.get("split" if name == "split" else "stages/" + name) is not None


def write_summary(store, config, profile):
    if profile["implementation_scope"] == "rollouts":
        if all(stage_complete(store, name) for name in PHASES["rollouts"]):
            report = {**audit_rollouts(store, config), "schema_version": "memgen-v5-rollout-summary-v1",
                      "profile_sha256": store.profile_hash,
                      "sampling": {"greedy": 1, "sampled": 7, "temperature": .8,
                                   "top_p": .95, "max_new_tokens": 1024}}
            atomic_json(store.root / "rollout_summary.json", {**report, "summary_sha256": digest(report)})
        return
    if all(stage_complete(store, name) for name in STAGES):
        audit = audit_complete(store, config)
        cards, groups = store.require("stages/cards"), store.require("stages/groups")
        calibration = store.require("stages/calibrate")
        report = {"schema_version": "memgen-v5-summary-v1", "complete": True,
            "profile_sha256": store.profile_hash, "status": "complete" if audit["primary_bank_count"]
            else "complete_without_primary_banks", "split_counts": store.require("split")["counts"],
            "episode_count": store.require("stages/episodes")["episode_count"],
            "contrast_count": store.require("stages/contrasts")["contrast_count"],
            "atom_counts": store.require("stages/atoms")["counts"], "group_count": groups["group_count"],
            "card_tier_counts": cards["tier_counts"], "compiled_bank_count": audit["primary_bank_count"],
            "selector_policy_sha256": audit["selector_policy_sha256"],
            "valid_baseline": calibration["baseline"],
            "valid_semantic_top1": calibration["semantic_top1_forced"],
            "valid_reranker_top1": calibration["reranker_top1_forced"],
            "valid_v5_selector": calibration["v5_utility_selector"],
            "native_prefix_kv_frozen": True, "one_bank_per_input": True,
            "selector_query": "input-only", "official_test_used": False,
            "external_inference_api_calls": 0}
        atomic_json(store.root / "brief_summary.json", {**report, "summary_sha256": digest(report)})
        print(f"[v5] complete summary={store.root / 'brief_summary.json'}", flush=True)


def run(store, config, profile, task, *, phase="all", stage=None, rollout_source=None,
        reasoner_factory=None, teacher_factory=None, reranker_factory=None):
    if phase not in PHASES or stage not in (None, "all", *STAGES):
        raise ValueError("Unknown V5 phase or stage")
    if stage not in (None, "all") and phase != "all":
        raise ValueError("Choose a V5 phase or one recovery stage, not both")
    selected = PHASES[phase] if stage in (None, "all") else (stage,)
    needs_reasoner = any(name in {"episodes", "groups", "compile", "calibrate"} for name in selected)
    needs_teacher = any(name in TEACHER_STAGES for name in selected)
    if needs_reasoner and reasoner_factory is None:
        from memgen.model.local_bank import LocalModel
        reasoner_factory = lambda: LocalModel(profile["reasoner"], config.reasoner, reasoner=True)
    if needs_teacher and teacher_factory is None:
        if config.teacher_backend == "vllm":
            from memgen.experience.bank_construction.vllm_teacher import VLLMTeacher
            model_factory = lambda: VLLMTeacher(store, config, profile["teacher"])
        else:
            from memgen.model.local_bank import LocalModel
            model_factory = lambda: LocalModel(profile["teacher"], config.teacher)
        teacher_factory = model_factory
    if config.reranker_enabled and reranker_factory is None and "calibrate" in selected:
        from memgen.model.v5_local_rerank import V5LocalReranker
        reranker_factory = lambda: V5LocalReranker(profile["reranker"], config.reranker.device,
                                                    config.reranker_max_length)
    teacher = None
    def get_teacher():
        nonlocal teacher
        if teacher is None:
            from .teacher import V5Teacher
            teacher = V5Teacher(store, config, teacher_factory)
        return teacher
    def ensure_episodes():
        if stage_complete(store, "episodes"):
            return
        if rollout_source is None:
            raise ValueError("V5 Bank phase requires --rollout-source or an existing Episode stage")
        from .reuse import reuse_episodes
        reuse_episodes(store, config, profile, task, rollout_source)
    try:
        for name in selected:
            print(f"[v5] phase={phase} stage={name}", flush=True)
            if name == "split":
                run_split(store, config, profile, task)
            elif name == "episodes":
                if rollout_source is not None:
                    from .reuse import reuse_episodes
                    reuse_episodes(store, config, profile, task, rollout_source)
                else:
                    run_rollouts(store, config, task, reasoner_factory)
            elif name == "review":
                ensure_episodes(); run_review(store, get_teacher())
            elif name == "contrasts":
                ensure_episodes(); run_contrasts(store, config)
            elif name == "atoms":
                run_atoms(store, get_teacher())
            elif name == "groups":
                model = reasoner_factory()
                try:
                    from memgen.model.v4_3_question_selector import encode_text
                    run_grouping(store, get_teacher(), config,
                                 lambda text: encode_text(model.runtime, text), profile["reasoner"])
                finally:
                    model.close()
            elif name == "cards":
                run_cards(store, get_teacher(), config)
            elif name == "compile":
                if teacher is not None:
                    teacher.close(); teacher = None
                run_compile(store, reasoner_factory)
            elif name == "calibrate":
                if teacher is not None:
                    teacher.close(); teacher = None
                run_calibration(store, config, task, reasoner_factory, reranker_factory)
    finally:
        if teacher is not None:
            teacher.close()
    write_summary(store, config, profile)
