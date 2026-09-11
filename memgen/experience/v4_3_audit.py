"""Answer-blind V4.3 plans, fixed wrong-Bank controls, and outcome aggregation."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from memgen.experience.v4_3_bank import authenticate, canonical_hash, file_hash, seal, text_hash

BRANCHES = ("baseline", "matched", "near_wrong", "far_wrong")
LAYERS = ("visible_content", "prompt_end_latent", "exact_gate_failure", "success_safety")
CONFIGURATION = {
    "maximum_completion_tokens": 1024, "maximum_active_steps": 32,
    "local_observation_tokens": 32, "recovery_low_token_count": 2,
    "post_memory_native_continuation": True,
    "generation_stop_policy": "completed_boxed_answer_or_eos_or_completion_budget",
    "maximum_active_memories": 1, "layer_number": 24, "attention_backend": "sdpa",
    "decoding": "greedy", "dtype": "bfloat16", "relative_phase_delta": 0,
}
CONTROL_POLICY = {
    "version": "v43-wrong-bank-controls-v1", "representation": "local_word_and_bigram_tfidf_cosine",
    "fit_scope": "frozen_qualified_descriptors_only", "no_shared_construction_sample": True,
    "near_order": ["same_category_first", "similarity_desc", "same_tier_first", "slot_distance_asc", "bank_id_asc"],
    "far_filter": "different_category_and_distinct_from_near",
    "far_order": ["similarity_asc", "same_tier_first", "slot_distance_asc", "bank_id_asc"],
    "answer_or_reward_access": False, "random_sampling": False,
}


def descriptor_vectors(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    counts = {}
    df = Counter()
    for r in sorted(records, key=lambda r: r["bank_id"]):
        tokens = re.findall(r"[a-z]+", r["descriptor"].lower())
        terms = tokens + [" ".join(pair) for pair in zip(tokens, tokens[1:])]
        counts[r["bank_id"]] = Counter(terms)
        df.update(set(terms))
    vectors = {}
    for bank_id, terms in counts.items():
        values = {t: (1 + math.log(c)) * (1 + math.log((len(records) + 1) / (df[t] + 1))) for t, c in sorted(terms.items())}
        norm = math.sqrt(sum(v * v for v in values.values()))
        vectors[bank_id] = {t: v / norm for t, v in values.items()}
    return vectors


def cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return round(sum(a[k] * b.get(k, 0) for k in sorted(a)), 12)


def build_wrong_controls(records: Sequence[Mapping[str, Any]], slot_counts: Mapping[str, int]) -> dict[str, Any]:
    ids = [r["bank_id"] for r in records]
    if len(ids) != len(set(ids)) or set(slot_counts) != set(ids) or any(c <= 0 for c in slot_counts.values()):
        raise ValueError("Wrong-bank source namespace/slot counts invalid")
    vectors = descriptor_vectors(records)
    result = {}
    for r in sorted(records, key=lambda r: r["bank_id"]):
        bank_id = r["bank_id"]
        pool = [other for other in records if other["bank_id"] != bank_id
                and not set(other["construction"]["sample_ids"]) & set(r["construction"]["sample_ids"])]
        diagnostics = {}
        for other in pool:
            oid = other["bank_id"]
            diagnostics[oid] = {"similarity": cosine(vectors[bank_id], vectors[oid]),
                "same_category": other["semantic_category"] == r["semantic_category"],
                "same_tier": other["quality_tier"] == r["quality_tier"],
                "slot_distance": abs(slot_counts[oid] - slot_counts[bank_id])}
        if len(pool) < 2:
            result[bank_id] = {"qualified": False, "reason": "insufficient_nonoverlapping_wrong_banks"}
            continue
        near = min(diagnostics, key=lambda oid: (not diagnostics[oid]["same_category"], -diagnostics[oid]["similarity"],
                    not diagnostics[oid]["same_tier"], diagnostics[oid]["slot_distance"], oid))
        far_pool = [oid for oid, d in diagnostics.items() if oid != near and not d["same_category"]]
        if not far_pool:
            result[bank_id] = {"qualified": False, "reason": "no_distinct_different_category_far_bank"}
            continue
        far = min(far_pool, key=lambda oid: (diagnostics[oid]["similarity"], not diagnostics[oid]["same_tier"],
                                           diagnostics[oid]["slot_distance"], oid))
        result[bank_id] = {"qualified": True, "matched": bank_id, "near_wrong": near, "far_wrong": far,
                          "selected_diagnostics": {"near_wrong": diagnostics[near], "far_wrong": diagnostics[far]},
                          "eligible_pool": dict(sorted(diagnostics.items()))}
    return seal({"schema_version": "memgen-v4.3-wrong-bank-plan-v1", "policy": CONTROL_POLICY,
                 "descriptor_sha256": {r["bank_id"]: r["descriptor_sha256"] for r in records},
                 "slot_counts": dict(slot_counts), "controls": result}, "controls_sha256")


def bind_source_state(*, cache: Any, bank: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]],
                      semantic_packets_path: Path, risk_path: Path) -> dict[str, Any]:
    """Create a new V4.3 binding; never rewrite the V4.2 cache manifest."""
    inputs = bank["bundle"]["inputs"]["file_sha256"]
    old_inputs = cache.manifest["provenance"]["inputs"]
    if (file_hash(semantic_packets_path) != inputs["packets"]
            or old_inputs["bank_records_sha256"] != inputs["records"]
            or old_inputs["bank_manifest_file_sha256"] != inputs["manifest"]
            or old_inputs["token_risk_artifact_sha256"] != file_hash(risk_path)):
        raise ValueError("V4.3 source-state lineage/file identity mismatch")
    candidates = {r["source_v42_bank_id"]: r for r in bank["candidate_bank_records.jsonl"]}
    bindings = []
    seen_prompts = set()
    for event in cache.events:
        r = candidates.get(event["bank_id"])
        e = evidence.get(event["experience_id"])
        if r is None or e is None:
            raise ValueError("Cache event outside V4.3 construction membership")
        membership = dict(zip(r["construction"]["experience_ids"], r["construction"]["sample_ids"]))
        if (membership.get(event["experience_id"]) != event["sample_id"] or e["sample_id"] != event["sample_id"]
                or event["independent_sample_id"] != canonical_hash({"benchmark": "openai/gsm8k", "logical_split": "bank-source", "sample_id": e["sample_id"]})
                or event.get("logical_split") != "bank-source" or event.get("dataset_split") != "train"
                or event["question_sha256"] != text_hash(e["question"].strip())
                or event["bank_record_sha256"] != r["curation_provenance"]["source_curated_record_sha256"]
                or event["completion_hashes"]["verified_success_completion_sha256"] != text_hash(e["verified_success_trajectory"].strip())
                or event["completion_hashes"]["verified_failure_completion_sha256"] != text_hash(e["verified_failure_trajectory"].strip())):
            raise ValueError("Cache event sample/question/trajectory/Bank identity mismatch")
        if event["event_kind"] == "prompt_semantic":
            seen_prompts.add(event["sample_id"])
        bindings.append({"event_id": event["event_id"], "event_sha256": event["record_sha256"],
                         "source_v42_bank_id": event["bank_id"], "v43_bank_id": r["bank_id"],
                         "sample_id": e["sample_id"], "experience_id": event["experience_id"]})
    expected_samples = {sid for r in candidates.values() for sid in r["construction"]["sample_ids"]}
    if seen_prompts != expected_samples or len(seen_prompts) != 116:
        raise ValueError("V4.3 requires the complete 116-sample source cache, even for smoke")
    return seal({"schema_version": "memgen-v4.3-source-state-binding-v1",
        "source_cache_manifest_sha256": cache.manifest["manifest_sha256"],
        "source_cache_manifest_file_sha256": file_hash(cache.manifest_path),
        "construction_bundle_sha256": bank["bundle"]["manifest_sha256"],
        "lineage_sha256": bank["source_v42_to_v43_lineage.json"]["lineage_sha256"],
        "risk_artifact_sha256": file_hash(risk_path), "reasoner": cache.manifest["reasoner"],
        "event_bindings": bindings, "source_manifest_rewritten": False,
        "prefix_authentication": "all_selected_prefixes_retokenized_before_generation",
        "offline_only": True, "qualified_for_online_use": False}, "binding_sha256")


def build_plan(*, candidates: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]],
               controls: Mapping[str, Any], binding: Mapping[str, Any], mode: str,
               all_bank_sweep: bool = False, bank_scope: str = "all") -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError("Unknown V4.3 audit mode")
    if bank_scope not in {"primary", "all"}:
        raise ValueError("Unknown V4.3 Bank scope")
    tiers = ("primary",) if bank_scope == "primary" else ("primary", "conditional")
    authenticate(controls, "controls_sha256", "wrong-bank controls")
    authenticate(binding, "binding_sha256", "source binding")
    by_old = {r["source_v42_bank_id"]: r for r in candidates}
    prompts = sorted((e for e in events if e["event_kind"] == "prompt_semantic"), key=lambda e: e["sample_id"])
    if len(prompts) != 116:
        raise ValueError("Plan requires full source cache coverage")
    source_cache_sample_count = len(prompts)
    scoped_candidates = [r for r in candidates if r["quality_tier"] in tiers]
    eligible_ids = {r["bank_id"] for r in scoped_candidates if r["qualification"]["construction_qualified"]}
    if set(controls["controls"]) != eligible_ids:
        raise ValueError("Wrong-bank controls must use exactly the selected Bank scope")
    for control in controls["controls"].values():
        if control.get("qualified") and any(control[name] not in eligible_ids for name in BRANCHES[1:]):
            raise ValueError("Wrong-bank memory outside selected Bank scope")
    prompts = [p for p in prompts if by_old[p["bank_id"]]["quality_tier"] in tiers]
    excluded, unreachable, cases = [], [], []
    selected_banks = set()
    if mode == "smoke":
        # Require selected tiers and actual failure/success gates. Keep this sample
        # rule fixed and independent of generated answers or repair outcomes.
        for tier in tiers:
            available = []
            for r in candidates:
                if r["quality_tier"] != tier or not controls["controls"].get(r["bank_id"], {}).get("qualified"):
                    continue
                kinds = {e["event_kind"] for e in events if e["bank_id"] == r["source_v42_bank_id"]}
                if {"failure_gate_attempt", "success_gate_attempt"} <= kinds:
                    available.append(r["bank_id"])
            if not available:
                raise ValueError(f"Smoke lacks a {tier} Bank with qualified controls and actual failure/success gates")
            selected_banks.add(min(available))
    for prompt in prompts:
        r = by_old[prompt["bank_id"]]
        control = controls["controls"].get(r["bank_id"], {})
        if not prompt["failure_gate_eligible"]:
            unreachable.append({"sample_id": prompt["sample_id"], "bank_id": r["bank_id"],
                                "reason": prompt["failure_gate_rejection_reason"], "counted_as_memory_ineffective": False})
        if not r["qualification"]["construction_qualified"] or not control.get("qualified"):
            excluded.append({"sample_id": prompt["sample_id"], "bank_id": r["bank_id"],
                             "reason": "construction_quarantine" if not r["qualification"]["construction_qualified"] else control["reason"]})
            continue
        if mode == "smoke" and r["bank_id"] not in selected_banks:
            continue
        for layer in LAYERS:
            if layer in {"visible_content", "prompt_end_latent"}:
                source_events = [prompt]
            else:
                kind = "failure_gate_attempt" if layer == "exact_gate_failure" else "success_gate_attempt"
                source_events = sorted((e for e in events if e["sample_id"] == prompt["sample_id"] and e["event_kind"] == kind),
                                       key=lambda e: e["attempt_number"])
            for event in source_events:
                if layer == "success_safety" and event.get("online_reachable_safety_negative") is not True:
                    raise ValueError("Success safety must use the successful trajectory's actual gate")
                prefix_sha = prompt["prompt_token_ids_sha256"] if layer in {"visible_content", "prompt_end_latent"} else event["prefix_alignment"]["prefix_token_ids_sha256"]
                payload = {"audit_layer": layer, "sample_id": prompt["sample_id"], "experience_id": prompt["experience_id"],
                    "bank_id": r["bank_id"], "source_v42_bank_id": r["source_v42_bank_id"],
                    "quality_tier": r["quality_tier"], "semantic_category": r["semantic_category"],
                    "source_event_id": event["event_id"], "source_event_sha256": event["record_sha256"],
                    "gate_attempt_index": event.get("attempt_number", 0), "prompt_token_count": prompt["prompt_token_count"],
                    "prefix_token_count": prompt["prompt_token_count"] if layer in {"visible_content", "prompt_end_latent"} else event["token_position"] + 1,
                    "prefix_token_ids_sha256": prefix_sha,
                    "memories": {"baseline": None, **{name: control[name] for name in BRANCHES[1:]}},
                    "exact_native_prefix_cache": layer != "visible_content"}
                payload["case_id"] = "v43-case-" + canonical_hash(payload)
                cases.append(seal(payload, "case_sha256"))
    if mode == "smoke":
        retained = []
        for tier in tiers:
            for layer in LAYERS:
                subset = [c for c in cases if c["quality_tier"] == tier and c["audit_layer"] == layer]
                retained.extend(subset[:2])
        cases = retained
        if any(not any(c["quality_tier"] == t and c["audit_layer"] == l for c in cases) for t in tiers for l in LAYERS):
            raise ValueError("Smoke does not cover selected tiers and all four layers")
    if all_bank_sweep:
        primaries = sorted(r["bank_id"] for r in candidates if r["quality_tier"] == "primary" and r["qualification"]["construction_qualified"])
        for base in list(cases):
            if base["audit_layer"] != "exact_gate_failure":
                continue
            row = {k: v for k, v in base.items() if k not in {"case_id", "case_sha256"}}
            row["audit_layer"] = "all_bank_sweep"
            row["memories"] = {"baseline": None, **{"memory:" + bid: bid for bid in primaries}}
            row["case_id"] = "v43-case-" + canonical_hash(row)
            cases.append(seal(row, "case_sha256"))
    if not cases:
        raise ValueError("No auditable V4.3 cases; inspect construction/control qualification")
    return seal({"schema_version": "memgen-v4.3-audit-plan-v1", "mode": mode,
        "configuration": CONFIGURATION, "binding_sha256": binding["binding_sha256"],
        "controls_sha256": controls["controls_sha256"], "cases": cases, "case_count": len(cases),
        "case_order_sha256": canonical_hash([c["case_id"] for c in cases]),
        "bank_scope": bank_scope, "selected_quality_tiers": list(tiers),
        "source_sample_count": len(prompts), "source_cache_sample_count": source_cache_sample_count,
        "out_of_scope_sample_count": source_cache_sample_count - len(prompts),
        "excluded_samples": excluded,
        "bank_categories": {r["bank_id"]: r["semantic_category"] for r in scoped_candidates},
        "gate_unreachable_failures": unreachable, "gate_unreachable_counted_as_memory_ineffective": False,
        "all_bank_sweep": all_bank_sweep, "offline_only": True, "qualified_for_online_use": False,
        "held_out_generalization_claim": False, "selector_artifact": None,
        "audit_interpretation": "construction_mechanism_qualification"}, "plan_sha256")


def validate_result(row: Mapping[str, Any], case: Mapping[str, Any], profile_sha256: str) -> None:
    authenticate(row, "record_sha256", "V4.3 result")
    if row.get("schema_version") != "memgen-v4.3-audit-result-v1" or row.get("profile_sha256") != profile_sha256 or row.get("case_sha256") != case["case_sha256"] or row.get("case_id") != case["case_id"]:
        raise ValueError("Result profile/case binding mismatch")
    if any(row.get(k) != case[k] for k in ("audit_layer", "sample_id", "experience_id", "bank_id", "quality_tier", "semantic_category", "gate_attempt_index")):
        raise ValueError("Result reporting identity differs from its case")
    branches = row["branches"]
    if set(branches) != set(case["memories"]):
        raise ValueError("Result branch coverage mismatch")
    if case["exact_native_prefix_cache"]:
        p = row["prefix_cache_parity"]
        for field in ("all_branches_share_exact_prefix", "initial_cache_tensors_exactly_equal", "branch_cache_storage_is_independent", "cache_length_parity"):
            if p.get(field) is not True:
                raise ValueError("Exact native prefix/cache parity failed")
        if p.get("prefix_token_ids_sha256") != case["prefix_token_ids_sha256"]:
            raise ValueError("Result prefix identity mismatch")
    elif row["prefix_cache_parity"].get("all_branches_share_exact_prefix") is not False:
        raise ValueError("Visible audit cannot claim exact shared prefix")
    for name, b in branches.items():
        visible = case["audit_layer"] == "visible_content"
        if b["condition_memory_id"] != case["memories"][name] or b["memory_id"] != (None if visible else case["memories"][name]):
            raise ValueError("Result unified memory identity mismatch")
        if (b["maximum_completion_tokens"] != 1024 or b["active_step_count"] > 32
                or b["prefix_completion_token_count"] != (0 if visible else case["prefix_token_count"] - case["prompt_token_count"])
                or b["active_step_count"] < 0 or b["active_step_count"] > len(b["continuation_token_ids"])
                or b["prefix_completion_token_count"] + len(b["continuation_token_ids"]) > 1024
                or len(b["local_continuation_token_ids"]) > 32
                or b["strict_reward"] not in (0.0, 1.0)
                or b["local_intervention_evaluation"]["strict_reward"] not in (0.0, 1.0)
                or b["stop_reason"] not in {"completed_boxed_answer", "eos", "maximum_completion_tokens"}):
            raise ValueError("Result horizon/outcome contract mismatch")
        if b["continuation_token_ids_sha256"] != canonical_hash(b["continuation_token_ids"]) or b["local_continuation_token_ids"] != b["continuation_token_ids"][:32]:
            raise ValueError("Result continuation hash/local window mismatch")
        for metric in ("first_step_logits_kl",):
            if not math.isfinite(b[metric]) or b[metric] < -1e-7:
                raise ValueError("Result contains invalid diagnostics")
        traces = b["attention_traces"]
        if not visible and name != "baseline":
            if not traces or len(traces) != b["active_step_count"]:
                raise ValueError("Memory trace/active count mismatch")
            for i, trace in enumerate(traces):
                mass, native = trace["memory_attention_mass"], trace["native_attention_mass"]
                if (trace["memory_id"] != b["memory_id"] or not 0 < mass <= 1 or not 0 <= native <= 1
                        or not math.isclose(mass + native, 1.0, abs_tol=0.003)
                        or trace["native_key_length"] != case["prefix_token_count"] + i
                        or trace["canonical_rope_score_relative_error"] is None
                        or not math.isfinite(trace["canonical_rope_score_relative_error"])):
                    raise ValueError("Memory attention integrity failed")
        elif traces or b["active_step_count"]:
            raise ValueError("Baseline/visible branch cannot have latent attention traces")


def _mean(values):
    return sum(values) / len(values) if values else None


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_rows = defaultdict(list)
    for r in rows:
        sample_rows[r["sample_id"]].append(r)
    branch_names = sorted({name for r in rows for name in r["branches"]})
    summary = {"case_count": len(rows), "independent_sample_count": len(sample_rows), "branches": {}}
    for name in branch_names:
        applicable = [r for r in rows if name in r["branches"]]
        grouped = defaultdict(list)
        for r in applicable:
            grouped[r["sample_id"]].append(r)
        branches = [r["branches"][name] for r in applicable]
        gains = [r for r in applicable if r["branches"]["baseline"]["strict_reward"] == 0 and r["branches"][name]["strict_reward"] == 1]
        harms = [r for r in applicable if r["branches"]["baseline"]["strict_reward"] == 1 and r["branches"][name]["strict_reward"] == 0]
        summary["branches"][name] = {
            "case_count": len(applicable), "correct_count": sum(b["strict_reward"] for b in branches),
            "accuracy": _mean([b["strict_reward"] for b in branches]),
            "local_window_accuracy": _mean([b["local_intervention_evaluation"]["strict_reward"] for b in branches]),
            "final_format_valid_rate": _mean([float(b["format_valid"]) for b in branches]),
            "local_window_format_valid_rate": _mean([float(b["local_intervention_evaluation"]["format_valid"]) for b in branches]),
            "independent_sample_macro_accuracy": _mean([_mean([r["branches"][name]["strict_reward"] for r in group]) for group in grouped.values()]),
            "gain_count": len(gains), "harm_count": len(harms),
            "gain_independent_sample_count": len({r["sample_id"] for r in gains}),
            "harm_independent_sample_count": len({r["sample_id"] for r in harms}),
            "independent_sample_macro_gain": _mean([_mean([float(r in gains) for r in group]) for group in grouped.values()]),
            "independent_sample_macro_harm": _mean([_mean([float(r in harms) for r in group]) for group in grouped.values()]),
            "mean_memory_attention_mass": _mean([_mean([t["memory_attention_mass"] for t in b["attention_traces"]]) for b in branches if b["attention_traces"]]),
            "mean_first_step_kl": _mean([b["first_step_logits_kl"] for b in branches]),
            "first_step_top1_change_rate": _mean([float(b["first_step_top1_changed"]) for b in branches]),
            "local_continuation_divergence_rate": _mean([float(r["branches"][name]["local_continuation_token_ids"] != r["branches"]["baseline"]["local_continuation_token_ids"]) for r in applicable]),
            "final_trajectory_divergence_rate": _mean([float(r["branches"][name]["continuation_token_ids"] != r["branches"]["baseline"]["continuation_token_ids"]) for r in applicable]),
        }
    for wrong in ("near_wrong", "far_wrong"):
        pairs = [r for r in rows if "matched" in r["branches"] and wrong in r["branches"]]
        by_sample = defaultdict(list)
        for r in pairs:
            by_sample[r["sample_id"]].append(r["branches"]["matched"]["strict_reward"] - r["branches"][wrong]["strict_reward"])
        summary["matched_minus_" + wrong] = {"mean_reward_difference": _mean([v for vs in by_sample.values() for v in vs]),
            "independent_sample_macro_reward_difference": _mean([_mean(vs) for vs in by_sample.values()]),
            "matched_better_count": sum(v > 0 for vs in by_sample.values() for v in vs),
            "matched_worse_count": sum(v < 0 for vs in by_sample.values() for v in vs)}
    return summary


def sweep_metrics(rows: Sequence[Mapping[str, Any]], categories: Mapping[str, str] | None = None) -> dict[str, Any]:
    helpful, harmful, best, ranks = defaultdict(set), defaultdict(set), Counter(), []
    sample_best, sample_gain = defaultdict(list), defaultdict(list)
    pair_cases = defaultdict(list)
    for row in rows:
        scores = {name[7:]: b["strict_reward"] for name, b in row["branches"].items() if name.startswith("memory:")}
        baseline = row["branches"]["baseline"]["strict_reward"]
        highest = max(scores.values())
        winners = [bid for bid, score in scores.items() if score == highest]
        for bid, score in scores.items():
            if score > baseline:
                helpful[bid].add(row["sample_id"])
            if score < baseline:
                harmful[bid].add(row["sample_id"])
        for bid in winners:
            best[bid] += 1 / len(winners)
        sample_best[row["sample_id"]].append(max(baseline, highest))
        sample_gain[row["sample_id"]].append(float(highest > baseline))
        matched = scores.get(row["bank_id"])
        if matched is not None and categories is not None:
            for bid, score in scores.items():
                if bid != row["bank_id"] and categories[bid] == categories[row["bank_id"]]:
                    pair_cases[(row["bank_id"], bid)].append((row["sample_id"], baseline, matched, score))
        ranks.append({"case_id": row["case_id"], "matched_bank_rank": None if matched is None else 1 + sum(v > matched for v in scores.values()),
                      "rank_basis": "strict_correctness_competition_rank_ties_preserved"})
    interchangeability = []
    for (matched_id, alternative_id), observations in sorted(pair_cases.items()):
        per_sample = defaultdict(list)
        for sid, baseline, matched, alternative in observations:
            per_sample[sid].append(alternative - matched)
        interchangeability.append({"matched_bank_id": matched_id, "alternative_bank_id": alternative_id,
            "semantic_category": categories[matched_id], "case_count": len(observations),
            "independent_sample_count": len(per_sample),
            "alternative_minus_matched_sample_macro": _mean([_mean(v) for v in per_sample.values()]),
            "both_helpful_independent_sample_count": len({sid for sid, b, m, a in observations if m > b and a > b}),
            "alternative_harm_independent_sample_count": len({sid for sid, b, m, a in observations if a < b})})
    return {"outcome_information_used": True, "online_accuracy_claim": False,
        "same_category_interchangeability": interchangeability,
        "interchangeability_interpretation": "paired_construction_prefix_diagnostic_not_generalization",
        "oracle_best_includes_baseline": True,
        "oracle_best_ceiling_sample_macro": _mean([_mean(v) for v in sample_best.values()]),
        "any_helpful_memory_rate_sample_macro": _mean([_mean(v) for v in sample_gain.values()]),
        "reusable_memory_count": sum(len(s) >= 2 for s in helpful.values()),
        "reuse_definition": "helps_at_least_two_independent_construction_samples",
        "harmful_memory_count": len(harmful), "helpful_samples_per_memory": {k: len(v) for k, v in helpful.items()},
        "harmful_samples_per_memory": {k: len(v) for k, v in harmful.items()},
        "best_memory_hubness_tie_fractional_case_counts": dict(best), "matched_bank_ranks": ranks}


def aggregate(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], profile_sha256: str) -> dict[str, Any]:
    authenticate(plan, "plan_sha256", "V4.3 audit plan")
    if plan.get("configuration") != CONFIGURATION or plan.get("qualified_for_online_use") is not False:
        raise ValueError("Audit plan frozen configuration drifted")
    cases = {c["case_id"]: c for c in plan["cases"]}
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate audit result")
    for row in rows:
        if row["case_id"] not in cases:
            raise ValueError("Result outside audit plan")
        validate_result(row, cases[row["case_id"]], profile_sha256)
    layers = {}
    for layer in (*LAYERS, "all_bank_sweep"):
        subset = [r for r in rows if r["audit_layer"] == layer]
        expected = [c for c in plan["cases"] if c["audit_layer"] == layer]
        if not expected:
            continue
        item = {"expected_case_count": len(expected), "completed_case_count": len(subset), "overall": metrics(subset)}
        for dimension in ("quality_tier", "bank_id", "gate_attempt_index"):
            item["by_" + dimension] = {str(key): metrics([r for r in subset if r[dimension] == key]) for key in sorted({c[dimension] for c in expected})}
        if layer == "all_bank_sweep" and subset:
            item["ceiling"] = sweep_metrics(subset, plan["bank_categories"])
        layers[layer] = item
    return seal({"schema_version": "memgen-v4.3-audit-report-v1", "profile_sha256": profile_sha256,
        "plan_sha256": plan["plan_sha256"], "mode": plan["mode"], "configuration": CONFIGURATION,
        "complete": len(rows) == len(cases), "expected_case_count": len(cases), "completed_case_count": len(rows),
        "status": "completed_mechanism_diagnostic" if len(rows) == len(cases) else "in_progress",
        "by_audit_layer": layers, "source_sample_count": plan["source_sample_count"],
        "bank_scope": plan.get("bank_scope", "all"),
        "source_cache_sample_count": plan.get("source_cache_sample_count", 116),
        "out_of_scope_sample_count": plan.get("out_of_scope_sample_count", 0),
        "excluded_samples": plan["excluded_samples"],
        "gate_unreachable_failure_count": len(plan["gate_unreachable_failures"]),
        "gate_unreachable_counted_as_memory_ineffective": False,
        "offline_only": True, "qualified_for_online_use": False, "held_out_generalization_claim": False,
        "local_and_final_metrics_separated": True, "selector_artifact": None,
        "all_bank_oracle_best_is_online_accuracy": False}, "report_sha256")
