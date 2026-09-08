"""Deterministic, tensor-free V4.3 construction from authenticated V4.2 packets.

This is extractive lexical consensus, not an LLM semantic-entailment audit.
All five signature fields vote independently. Raw solutions are audit inputs
only; no verifier score, trajectory, or legacy role card enters the descriptor.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Sequence


VERSION = "v4.3-unified-heuristic"
RECORD_SCHEMA = "memgen-v4.3-unified-bank-record-v1"
MANIFEST_SCHEMA = "memgen-v4.3-unified-bank-manifest-v1"
PACKET_SCHEMA = "memgen-v4.2-semantic-evidence-packet-v1"
CURATED_SCHEMA = "memgen-v4.2-local-curated-bank-manifest-v1"
SIGNATURE_FIELDS = (
    "problem_structure", "decision_point", "repair_operator",
    "failure_mechanism", "verification_operator",
)
CARD_FIELDS = ("applies_when", "procedure", "avoid", "verify", "only_use_when")
COMPILER_CONTRACT = {
    "layer_number": 24, "all_kv_groups": True, "canonical_pre_rope": True,
    "relative_phase_delta": 0, "attention_backend": "sdpa",
    "normalization": "native_input_layernorm", "projection": "native_k_proj_v_proj",
    "server_dtype": "bfloat16", "memory_count_per_bank": 1,
    "maximum_active_memories": 1,
    "descriptor_variants": ["raw_descriptor", "internal_principle", "hidden_note"],
    "retention_policy": "content_positions_only",
    "variant_combination": "concatenate_slot_dimension",
}
# Scope restrictions are derived from the six existing curation decisions, not
# claimed to have five independent evidence votes. They only narrow admission.
CONDITIONAL_GUARDS = {
    "requested-entity state update": "The problem explicitly asks for a future age after a stated time interval.",
    "complete aggregation": "The problem explicitly defines the categories or full groups and any remainder to aggregate.",
    "unit-valued quantity": "The problem assigns a value to each coin denomination and distinguishes coin count from monetary value.",
    "requested magnitude": "The requested difference is explicitly an absolute magnitude rather than a signed change.",
    "requested quantity interpretation": "The requested result is explicitly the remaining quantity already represented by the computation.",
}
CONSTRUCTION_POLICY = {
    "schema_version": "memgen-v4.3-extractive-consensus-policy-v1",
    "minimum_distinct_support": 5,
    "similarity_rule": "normalized_token_sequence_match_with_ordered_protected_tokens",
    "minimum_pair_similarity": 0.80,
    "support_rule": "maximum_complete_link_subset_containing_representative",
    "ranking": ["distinct_sample_support_desc", "group_centrality_desc",
                "complete_clause_character_count_asc", "experience_id_asc"],
    "support_subset_tie_break": "experience_ids_lexicographic",
    "representation": "stdlib_cpu_token_sequence_no_embedding_model",
    "normalization": "NFKC_whitespace_typographic_punctuation_v1",
    "leakage_policy": "numeric_entity_source_overlap_answer_marker_v1",
    "core_fields": list(SIGNATURE_FIELDS),
    "composed_applies_when_support": "intersection_of_structure_and_decision_support_at_least_five",
    "conditional_guards": CONDITIONAL_GUARDS,
    "thresholds_frozen_before_outcomes": True,
    "semantic_entailment_claim": False,
}
_PROTECTED = frozenset("no not never without before after first last current original remaining total each per numerator denominator multiply divide add subtract increase decrease larger smaller absolute signed twice half double".split())
_NUMBER_WORDS = re.compile(r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|half|quarter|twice|double)\b", re.I)
_FORBIDDEN = re.compile(r"\\boxed|####|\b(?:target|reference|reward|verifier|sample[_ -]?id|evidence[_ -]?id)\b|final answer\s*(?:is|:)|<\|", re.I)
_ENTITY_EXCEPTIONS = frozenset("a an the how what when where why which if in on at after before each every there this that it he she they his her their its to for of and but then given suppose let find calculate determine use check verify avoid do only procedure".split())
_PROCEDURE_VERBS = frozenset("identify determine compute calculate multiply divide add subtract sum aggregate combine list enumerate account include exclude remove convert scale apply update track carry preserve use choose select solve isolate express write form set equate substitute distribute factor distinguish separate compare align match assign label count evaluate recompute interpret report derive obtain restore retain calculate".split())
_VERIFY_VERBS = frozenset("verify check confirm compare ensure validate substitute recompute reconstruct reverse multiply divide sum assess test inspect reconcile evaluate match recover".split())


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(value: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    result = deepcopy(dict(value))
    result.pop(field, None)
    result[field] = canonical_hash(result)
    return result


def authenticate(value: Mapping[str, Any], field: str, owner: str) -> None:
    if value.get(field) != canonical_hash({k: v for k, v in value.items() if k != field}):
        raise ValueError(f"{owner} {field} hash mismatch")


def _unique(values: Any, owner: str) -> list[str]:
    if (not isinstance(values, list) or not values
            or any(not isinstance(v, str) or not v.strip() for v in values)
            or len(values) != len(set(values))):
        raise ValueError(f"{owner}: missing or duplicate identities")
    return values


def _sha(value: Any, owner: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{owner}: invalid SHA256")


def normalize_clause(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value.rstrip(".; ") + "." if value else ""


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+(?:'[a-z]+)?", value.casefold()))


def clause_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if tuple(t for t in a if t in _PROTECTED or t.endswith("n't")) != tuple(t for t in b if t in _PROTECTED or t.endswith("n't")):
        return 0.0
    # min makes SequenceMatcher's occasionally directional score symmetric.
    return min(SequenceMatcher(None, a, b, autojunk=False).ratio(),
               SequenceMatcher(None, b, a, autojunk=False).ratio())


def leakage_issues(text: str, evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    """Conservative static screening; never claims complete semantic leakage QA."""
    normalized = unicodedata.normalize("NFKC", text)
    issues: set[str] = set()
    if any(c.isnumeric() for c in normalized) or _NUMBER_WORDS.search(normalized):
        issues.add("numeric_constant")
    if _FORBIDDEN.search(normalized):
        issues.add("answer_or_role_or_reward_language")
    if re.search(r"[=$€£]|\\(?:frac|begin)|https?://|\b[a-z]\s*[-+*/^]\s*[a-z]\b", normalized, re.I):
        issues.add("equation_or_nonprocess_payload")
    words = _tokens(normalized)
    if len(words) < 4 or len(normalized) > 1200:
        issues.add("incomplete_or_unbounded_process_clause")
    lower = normalized.casefold()
    for item in evidence:
        for field in ("sample_id", "evidence_id"):
            if item.get(field) and str(item[field]).casefold() in lower:
                issues.add("source_identity")
        question = str(item.get("question", ""))
        entities = {w.casefold() for w in re.findall(r"\b[A-Z][a-z]+(?:'[a-z]+)?\b", question)} - _ENTITY_EXCEPTIONS
        if entities.intersection(words):
            issues.add("source_entity")
        for field in ("question", "official_solution", "verified_success_trajectory", "verified_failure_trajectory"):
            raw_words = _tokens(str(item.get(field, "")))
            spans = {raw_words[i:i + 8] for i in range(len(raw_words) - 7)}
            if any(words[i:i + 8] in spans for i in range(len(words) - 7)):
                issues.add("source_text_or_solution_trace_overlap")
        solution = str(item.get("official_solution", ""))
        for answer in re.findall(r"\\boxed\{([^{}]+)\}|####\s*([^\n]+)", solution):
            fragment = next((s.strip().casefold() for s in answer if s.strip()), "")
            if fragment and re.search(r"(?<!\w)" + re.escape(fragment) + r"(?!\w)", lower):
                issues.add("answer_fragment")
    return sorted(issues)


def authenticate_packets(packets: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_candidate: dict[str, Mapping[str, Any]] = {}
    seen_evidence: set[str] = set()
    seen_samples: set[str] = set()
    for packet in packets:
        if packet.get("schema_version") != PACKET_SCHEMA:
            raise ValueError("semantic packet schema mismatch")
        authenticate(packet, "packet_sha256", "semantic packet")
        candidate_id = packet.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in by_candidate:
            raise ValueError("duplicate or missing packet candidate")
        evidence = packet.get("evidence")
        if not isinstance(evidence, list) or not 5 <= len(evidence) <= 8 or packet.get("evidence_count") != len(evidence):
            raise ValueError("semantic packet evidence count mismatch")
        for item in evidence:
            if not isinstance(item, Mapping):
                raise ValueError("semantic evidence must be an object")
            eid, sid = item.get("evidence_id"), item.get("sample_id")
            if not isinstance(eid, str) or not eid or eid in seen_evidence:
                raise ValueError("missing or duplicate evidence identity")
            if not isinstance(sid, str) or not sid or sid in seen_samples:
                raise ValueError("missing or duplicate sample identity")
            for field in ("question", "official_solution", "verified_success_trajectory", "verified_failure_trajectory", "source_experience_type"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f"semantic evidence missing {field}")
            match = re.fullmatch(r"gsm8k-train-[0-9]+-([0-9a-f]{12})", sid)
            if not match or match[1] != text_hash(item["question"].strip())[:12]:
                raise ValueError("construction sample/question identity mismatch")
            signature = item.get("semantic_signature")
            if not isinstance(signature, Mapping) or set(signature) != set(SIGNATURE_FIELDS):
                raise ValueError("semantic signature schema mismatch")
            if any(not isinstance(signature[f], str) or not signature[f].strip() for f in SIGNATURE_FIELDS):
                raise ValueError("empty semantic signature field")
            for field in ("source_signature_sha256", "source_provenance_sha256", "construction_input_sha256"):
                _sha(item.get(field), field)
            for field in ("target_verifier", "reference_verifier"):
                if not isinstance(item.get(field), Mapping):
                    raise ValueError(f"semantic evidence missing {field}")
            seen_evidence.add(eid)
            seen_samples.add(sid)
        by_candidate[candidate_id] = packet
    if not by_candidate:
        raise ValueError("empty semantic packet input")
    return by_candidate


def authenticate_sources(*, records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
                         packets: Sequence[Mapping[str, Any]], policy: Mapping[str, Any],
                         input_hashes: Mapping[str, str]) -> dict[str, Mapping[str, Any]]:
    """Authenticate surviving sources without requiring deleted Phase-1 files."""
    if manifest.get("schema_version") != CURATED_SCHEMA or manifest.get("construction_version") != "v4.2-local-curated":
        raise ValueError("curated manifest schema/version mismatch")
    authenticate(manifest, "manifest_sha256", "curated manifest")
    for field in ("records", "manifest", "packets", "policy"):
        _sha(input_hashes.get(field), f"input {field}")
    if manifest.get("inputs", {}).get("semantic_preflight", {}).get("evidence_packet_file_sha256") != input_hashes["packets"]:
        raise ValueError("semantic packet file hash binding mismatch")
    curation = manifest.get("curation", {})
    if curation.get("policy_sha256") != canonical_hash(policy) or curation.get("policy_file_sha256") != input_hashes["policy"]:
        raise ValueError("curation policy hash binding mismatch")
    if policy.get("schema_version") != "memgen-v4.2-local-curation-policy-v1":
        raise ValueError("curation policy schema mismatch")
    expected_counts = {"primary": 11, "conditional": 6, "quarantine": 3, "hard_reject": 4}
    for key, value in {"expected_retained_record_count": 17, "expected_retained_evidence_count": 116,
                       "expected_source_record_count": 24, "expected_source_evidence_count": 167}.items():
        if policy.get(key) != value:
            raise ValueError(f"curation frozen {key} mismatch")
    if policy.get("retained_decisions") != ["primary", "conditional"]:
        raise ValueError("curation retention rule mismatch")
    decisions = policy.get("decisions", [])
    _unique([d["bank_id"] for d in decisions], "curation decisions")
    if dict(Counter(d["decision"] for d in decisions)) != expected_counts or policy.get("expected_decision_counts") != expected_counts:
        raise ValueError("curation tier counts mismatch")
    old_source = manifest.get("source_local_direct", {})
    for policy_key, source_key in (("source_manifest_sha256", "manifest_logical_sha256"),
                                   ("source_profile_sha256", "profile_sha256"),
                                   ("source_record_order_sha256", "record_order_sha256")):
        if not policy.get(policy_key) or policy[policy_key] != old_source.get(source_key):
            raise ValueError(f"curation lineage {policy_key} mismatch")
    if policy["source_record_order_sha256"] != canonical_hash([d["bank_id"] for d in decisions]):
        raise ValueError("curation source decision order mismatch")
    retained = [d for d in decisions if d["decision"] in {"primary", "conditional"}]
    bank_ids = _unique([r.get("bank_id") for r in records], "curated banks")
    if (len(records) != 17 or manifest.get("record_count") != 17 or manifest.get("evidence_count") != 116
            or bank_ids != [d["bank_id"] for d in retained] or bank_ids != manifest.get("bank_ids")
            or manifest.get("record_order_sha256") != canonical_hash(bank_ids)
            or set(manifest.get("record_sha256", {})) != set(bank_ids)
            or curation.get("retained_bank_ids") != bank_ids):
        raise ValueError("17-bank retained membership/order mismatch")
    packet_map = authenticate_packets(packets)
    seen_samples: set[str] = set()
    seen_evidence: set[str] = set()
    for record, decision in zip(records, retained):
        authenticate(record, "record_sha256", "curated record")
        if (record.get("schema_version") != "memgen-v4-bank-record-v1"
                or record.get("construction_version") != "v4.2-local-curated"
                or record.get("benchmark") != "openai/gsm8k"
                or record["record_sha256"] != manifest["record_sha256"][record["bank_id"]]):
            raise ValueError("curated record manifest binding mismatch")
        if any(record.get("curation", {}).get(k) != decision[k] for k in ("decision", "reason", "semantic_category")) or record["curation"].get("policy_sha256") != canonical_hash(policy):
            raise ValueError("curated record policy decision mismatch")
        construction = record["construction"]
        eids = _unique(construction.get("experience_ids"), "bank evidence")
        sids = _unique(construction.get("sample_ids"), "bank samples")
        if not 5 <= len(eids) <= 8 or len(eids) != len(sids) or construction.get("distinct_sample_count") != len(sids):
            raise ValueError("bank independent support must be five to eight")
        candidate = record.get("cluster", {}).get("source_candidate_id")
        packet = packet_map.get(candidate)
        if packet is None or construction.get("evidence_packet_sha256") != packet["packet_sha256"]:
            raise ValueError("bank packet identity/hash mismatch")
        evidence = packet["evidence"]
        if {e["evidence_id"]: e["sample_id"] for e in evidence} != dict(zip(eids, sids)):
            raise ValueError("evidence outside Bank or incomplete membership")
        if construction.get("source_signature_sha256") != {e["evidence_id"]: e["source_signature_sha256"] for e in evidence}:
            raise ValueError("source signature hash binding mismatch")
        if seen_samples.intersection(sids) or seen_evidence.intersection(eids):
            raise ValueError("duplicate sample/evidence across retained Banks")
        seen_samples.update(sids)
        seen_evidence.update(eids)
    if len(seen_samples) != 116 or len(seen_evidence) != 116:
        raise ValueError("116 independent construction evidence required")
    return packet_map


def select_clause(field: str, evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exhaustive complete-link support is cheap for at most eight examples."""
    _unique([e["evidence_id"] for e in evidence], "clause evidence")
    _unique([e["sample_id"] for e in evidence], "clause samples")
    candidates = []
    for item in sorted(evidence, key=lambda e: e["evidence_id"]):
        text = normalize_clause(item["semantic_signature"][field])
        issues = leakage_issues(text, evidence)
        required_verbs = {"repair_operator": _PROCEDURE_VERBS, "verification_operator": _VERIFY_VERBS}.get(field)
        if required_verbs is not None and not required_verbs.intersection(_tokens(text)):
            issues.append("missing_executable_process_operator")
        candidates.append({"experience_id": item["evidence_id"], "sample_id": item["sample_id"],
                           "text": text, "issues": issues})
    valid = [i for i, c in enumerate(candidates) if not c["issues"]]
    scores = {(i, j): clause_similarity(candidates[i]["text"], candidates[j]["text"])
              for i in valid for j in valid}
    threshold = CONSTRUCTION_POLICY["minimum_pair_similarity"]
    ranked = []
    for i in valid:
        subsets = [subset for n in range(1, len(valid) + 1) for subset in combinations(valid, n)
                   if i in subset and all(scores[a, b] >= threshold for a, b in combinations(subset, 2))]
        support = min(subsets, key=lambda s: (-len(s), tuple(candidates[j]["experience_id"] for j in s)))
        centrality = round(sum(scores[i, j] for j in valid) / len(valid), 12)
        ranked.append((-len(support), -centrality, len(candidates[i]["text"]), candidates[i]["experience_id"], i, support))
    audit = [{"experience_id": c["experience_id"], "sample_id": c["sample_id"],
              "normalized_clause_sha256": text_hash(c["text"]), "leakage_issues": c["issues"]} for c in candidates]
    if not ranked:
        return {"field": field, "qualified": False, "failure": "no_process_only_candidate",
                "support_count": 0, "candidate_audit": audit}
    ranking = min(ranked)
    chosen = candidates[ranking[4]]
    support = [candidates[i] for i in ranking[5]]
    support_ids = [c["experience_id"] for c in support]
    by_id = {e["evidence_id"]: e for e in evidence}
    return {
        "field": field, "qualified": len(support) >= 5,
        "failure": None if len(support) >= 5 else "insufficient_distinct_sample_support",
        "text": chosen["text"], "representative_experience_id": chosen["experience_id"],
        "supporting_experience_ids": support_ids,
        "supporting_sample_ids": [c["sample_id"] for c in support], "support_count": len(support),
        "support_rule": CONSTRUCTION_POLICY["support_rule"], "minimum_pair_similarity": threshold,
        "group_centrality": -ranking[1],
        "source_signature_sha256": {eid: by_id[eid]["source_signature_sha256"] for eid in support_ids},
        "support_pair_similarities": [{"experience_ids": [a["experience_id"], b["experience_id"]],
                                       "similarity": clause_similarity(a["text"], b["text"])} for a, b in combinations(support, 2)],
        "candidate_ranking": [{"experience_id": r[3], "support_count": -r[0], "group_centrality": -r[1],
                               "character_count": r[2]} for r in sorted(ranked)],
        "candidate_audit": audit,
    }


def render_card(card: Mapping[str, Any]) -> str:
    if set(card) != set(CARD_FIELDS):
        raise ValueError("unified process card schema mismatch")
    for field in ("applies_when", "only_use_when"):
        if not isinstance(card[field], str) or not card[field].strip():
            raise ValueError(f"empty card {field}")
    for field in ("procedure", "avoid", "verify"):
        _unique(card[field], f"card {field}")
    def numbered(field: str) -> str:
        return "\n".join(f"{i}. {text}" for i, text in enumerate(card[field], 1))
    return (f"Use when:\n{card['applies_when']}\n\nProcedure:\n{numbered('procedure')}"
            + "\n\nAvoid:\n" + "\n".join(card["avoid"])
            + f"\n\nVerify:\n{numbered('verify')}\n\nUse only when:\n{card['only_use_when']}")


def content_bank_id(record: Mapping[str, Any]) -> str:
    identity = {k: record[k] for k in ("construction_version", "source_v42_bank_id", "candidate_id",
                                     "unified_process_card", "construction", "construction_policy_sha256")}
    return "v43-bank-" + canonical_hash(identity)


def _assemble_card(support: Mapping[str, Any], tier: str, category: str) -> dict[str, Any]:
    boundary = "Use this procedure only when both the stated problem structure and decision point apply; otherwise reassess the method."
    if tier == "conditional":
        if category not in CONDITIONAL_GUARDS:
            raise ValueError("missing_conditional_scope_guard")
        boundary += " " + CONDITIONAL_GUARDS[category]
    return {
        "applies_when": support["problem_structure"]["text"] + " " + support["decision_point"]["text"],
        "procedure": [support["repair_operator"]["text"]],
        "avoid": ["Avoid this failure: " + support["failure_mechanism"]["text"]],
        "verify": [support["verification_operator"]["text"]], "only_use_when": boundary,
    }


def _scope_support(support: Mapping[str, Any], sample_map: Mapping[str, str]) -> dict[str, Any]:
    structure = set(support["problem_structure"].get("supporting_experience_ids", []))
    decision = set(support["decision_point"].get("supporting_experience_ids", []))
    ids = sorted(structure & decision)
    return {"source_fields": ["problem_structure", "decision_point"],
            "support_rule": "intersection_of_independent_clause_support",
            "supporting_experience_ids": ids, "supporting_sample_ids": [sample_map[e] for e in ids],
            "support_count": len(ids), "qualified": len(ids) >= 5}


def build_candidate(source: Mapping[str, Any], packet: Mapping[str, Any], *,
                    support_override=None, construction_policy=None, screen=None) -> dict[str, Any]:
    evidence = sorted(packet["evidence"], key=lambda e: e["evidence_id"])
    support = support_override if support_override is not None else {f: select_clause(f, evidence) for f in SIGNATURE_FIELDS}
    construction_policy = construction_policy or CONSTRUCTION_POLICY
    screen = screen or leakage_issues
    failures = [f + ":" + s["failure"] for f, s in support.items() if not s["qualified"]]
    scope_support = _scope_support(support, {e["evidence_id"]: e["sample_id"] for e in evidence})
    if not scope_support["qualified"]:
        failures.append("applies_when:insufficient_joint_distinct_sample_support")
    tier = source["curation"]["decision"]
    category = source["curation"]["semantic_category"]
    guard = None
    if tier == "conditional":
        guard = CONDITIONAL_GUARDS.get(category)
        if guard is None:
            failures.append("missing_conditional_scope_guard")
    card = descriptor = None
    leakage = []
    # Quarantined records preserve membership and diagnostic hashes, but expose
    # no partial runtime payload that a later compiler could accidentally use.
    if not failures:
        card = _assemble_card(support, tier, category)
        for field in CARD_FIELDS:
            texts = card[field] if isinstance(card[field], list) else [card[field]]
            for text in texts:
                leakage.extend(f"{field}:{issue}" for issue in screen(text, evidence))
        if leakage:
            failures.append("assembled_card_leakage")
            card = None
        else:
            descriptor = render_card(card)
    record = {
        "schema_version": RECORD_SCHEMA, "construction_version": VERSION,
        "source_v42_bank_id": source["bank_id"], "candidate_id": packet["candidate_id"],
        "benchmark": "openai/gsm8k", "semantic_category": category, "quality_tier": tier,
        "unified_process_card": card, "descriptor": descriptor,
        "descriptor_sha256": text_hash(descriptor) if descriptor is not None else None,
        "construction": {
            "experience_ids": [e["evidence_id"] for e in evidence],
            "sample_ids": [e["sample_id"] for e in evidence], "distinct_sample_count": len(evidence),
            "evidence_packet_sha256": packet["packet_sha256"],
            "source_signature_sha256": {e["evidence_id"]: e["source_signature_sha256"] for e in evidence},
            "semantic_signature_content_sha256": {e["evidence_id"]: canonical_hash(e["semantic_signature"]) for e in evidence},
            "source_provenance_sha256": {e["evidence_id"]: e["source_provenance_sha256"] for e in evidence},
            "construction_input_sha256": {e["evidence_id"]: e["construction_input_sha256"] for e in evidence},
            "consumption_rule": "all_members_all_five_signature_fields_no_sampling",
        },
        "clause_support": support,
        "composed_field_support": {"applies_when": scope_support},
        "card_field_sources": {"applies_when": ["problem_structure", "decision_point"],
                               "procedure": ["repair_operator"], "avoid": ["failure_mechanism"],
                               "verify": ["verification_operator"],
                               "only_use_when": ["inherited_applies_when_scope", "curation_scope_restriction"]},
        "boundary_provenance": {"basis": "scope_restriction_not_an_independent_evidence_vote",
                                "conditional_guard": guard, "independent_support_claim": False},
        "curation_provenance": {"source_curated_record_sha256": source["record_sha256"], **source["curation"]},
        "construction_policy_sha256": canonical_hash(construction_policy),
        "qualification": {"construction_qualified": not failures, "failures": failures,
                          "status": "qualified_for_offline_compilation" if not failures else "quarantine"},
        "leakage_audit": {"status": "passed_static_screen" if not failures else "not_qualified",
                          "assembled_card_issues": sorted(set(leakage)),
                          "semantic_factual_consistency_review_performed": False,
                          "complete_leakage_freedom_claim": False},
        "compiler_contract": deepcopy(COMPILER_CONTRACT),
        "offline_only": True, "qualified_for_online_use": False,
        "contains_runtime_answer_or_reward_signal": False,
    }
    record["bank_id"] = content_bank_id(record)
    return seal(record)


def validate_record(record: Mapping[str, Any]) -> None:
    if "semantic_construction" in record:
        from memgen.experience.v4_3_deepseek import validate_semantic_record
        validate_semantic_record(record)
        return
    authenticate(record, "record_sha256", "V4.3 record")
    if record.get("schema_version") != RECORD_SCHEMA or record.get("construction_version") != VERSION:
        raise ValueError("V4.3 record schema/version mismatch")
    if record.get("bank_id") != content_bank_id(record):
        raise ValueError("V4.3 content-addressed Bank ID mismatch")
    if record.get("compiler_contract") != COMPILER_CONTRACT or record.get("construction_policy_sha256") != canonical_hash(CONSTRUCTION_POLICY):
        raise ValueError("V4.3 frozen construction/compiler contract mismatch")
    if record.get("offline_only") is not True or record.get("qualified_for_online_use") is not False or record.get("contains_runtime_answer_or_reward_signal") is not False:
        raise ValueError("V4.3 offline-only contract mismatch")
    if any(k in record for k in ("roles", "target", "reference", "process_card")):
        raise ValueError("V4.3 legacy role fields are forbidden")
    cons = record["construction"]
    eids = _unique(cons["experience_ids"], "V4.3 evidence")
    sids = _unique(cons["sample_ids"], "V4.3 samples")
    if not 5 <= len(eids) <= 8 or len(eids) != len(sids) or cons["distinct_sample_count"] != len(sids):
        raise ValueError("V4.3 construction support mismatch")
    if record["quality_tier"] not in {"primary", "conditional"} or set(record["clause_support"]) != set(SIGNATURE_FIELDS):
        raise ValueError("V4.3 tier/clause schema mismatch")
    for field in ("source_signature_sha256", "semantic_signature_content_sha256", "source_provenance_sha256", "construction_input_sha256"):
        if set(cons[field]) != set(eids):
            raise ValueError("V4.3 evidence hash coverage mismatch")
        for value in cons[field].values():
            _sha(value, field)
    qualified = record["qualification"]["construction_qualified"]
    if not isinstance(qualified, bool):
        raise ValueError("V4.3 qualification must be a boolean")
    if record["curation_provenance"]["decision"] != record["quality_tier"] or record["curation_provenance"]["semantic_category"] != record["semantic_category"]:
        raise ValueError("V4.3 curation provenance mismatch")
    if qualified:
        if record["qualification"]["failures"] or not record["unified_process_card"]:
            raise ValueError("V4.3 qualified record has failures/no card")
        sample_map = dict(zip(eids, sids))
        expected_scope = _scope_support(record["clause_support"], sample_map)
        if not expected_scope["qualified"] or record.get("composed_field_support") != {"applies_when": expected_scope}:
            raise ValueError("V4.3 composed applicability independent support mismatch")
        for field, support in record["clause_support"].items():
            ids = _unique(support["supporting_experience_ids"], "clause support")
            samples = _unique(support["supporting_sample_ids"], "clause support samples")
            if (support["qualified"] is not True or support["support_count"] != len(ids) or len(ids) < 5
                    or not set(ids) <= set(eids) or samples != [sample_map[e] for e in ids]
                    or support["representative_experience_id"] not in ids
                    or support["source_signature_sha256"] != {e: cons["source_signature_sha256"][e] for e in ids}):
                raise ValueError(f"V4.3 {field} independent support mismatch")
            if (support["support_rule"] != CONSTRUCTION_POLICY["support_rule"]
                    or support["minimum_pair_similarity"] != CONSTRUCTION_POLICY["minimum_pair_similarity"]):
                raise ValueError("V4.3 clause support policy mismatch")
            pairs = support["support_pair_similarities"]
            if (len(pairs) != len(ids) * (len(ids) - 1) // 2
                    or {tuple(p["experience_ids"]) for p in pairs} != set(combinations(ids, 2))
                    or any(not CONSTRUCTION_POLICY["minimum_pair_similarity"] <= p["similarity"] <= 1 for p in pairs)):
                raise ValueError("V4.3 clause complete-link evidence mismatch")
            audits = support["candidate_audit"]
            if len(audits) != len(eids) or {a["experience_id"] for a in audits} != set(eids):
                raise ValueError("V4.3 clause did not audit all construction evidence")
            audit_by_id = {a["experience_id"]: a for a in audits}
            representative = support["representative_experience_id"]
            if (audit_by_id[representative]["normalized_clause_sha256"] != text_hash(support["text"])
                    or any(audit_by_id[e]["leakage_issues"] for e in ids)):
                raise ValueError("V4.3 representative/support leakage binding mismatch")
        if record["unified_process_card"] != _assemble_card(record["clause_support"], record["quality_tier"], record["semantic_category"]):
            raise ValueError("V4.3 card is not derived from the supported clauses")
        if record["descriptor"] != render_card(record["unified_process_card"]) or record["descriptor_sha256"] != text_hash(record["descriptor"]):
            raise ValueError("V4.3 descriptor binding mismatch")
    elif not record["qualification"]["failures"] or any(record[k] is not None for k in ("descriptor", "descriptor_sha256", "unified_process_card")):
        raise ValueError("V4.3 quarantine must have failures and no runtime payload")


def build_outputs(*, records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
                  packets: Sequence[Mapping[str, Any]], policy: Mapping[str, Any],
                  input_hashes: Mapping[str, str], implementation_hashes: Mapping[str, str],
                  candidates=None, construction_policy=None, report_metadata=None) -> dict[str, Any]:
    packet_map = authenticate_sources(records=records, manifest=manifest, packets=packets,
                                      policy=policy, input_hashes=input_hashes)
    construction_policy = construction_policy or CONSTRUCTION_POLICY
    if candidates is None:
        candidates = [build_candidate(r, packet_map[r["cluster"]["source_candidate_id"]]) for r in records]
    if len(candidates) != len(records) or any(
            candidate["source_v42_bank_id"] != source["bank_id"]
            or candidate["curation_provenance"]["source_curated_record_sha256"] != source["record_sha256"]
            or candidate["construction"]["evidence_packet_sha256"] != packet_map[source["cluster"]["source_candidate_id"]]["packet_sha256"]
            for candidate, source in zip(candidates, records)):
        raise ValueError("Construction candidate/source binding mismatch")
    for record in candidates:
        validate_record(record)
    bindings = {"file_sha256": dict(input_hashes), "source_manifest_sha256": manifest["manifest_sha256"],
                "curation_policy_sha256": canonical_hash(policy),
                "construction_policy_sha256": canonical_hash(construction_policy),
                "implementation_sha256": dict(implementation_hashes)}
    lineage = seal({"schema_version": "memgen-v4.3-bank-lineage-v1", "inputs": bindings,
                    "source_v42_to_v43": {r["source_v42_bank_id"]: r["bank_id"] for r in candidates},
                    "record_sha256": {r["bank_id"]: r["record_sha256"] for r in candidates},
                    "membership_preserved": True}, "lineage_sha256")
    outputs: dict[str, Any] = {"candidate_bank_records.jsonl": candidates,
                               "source_v42_to_v43_lineage.json": lineage,
                               "construction_policy.json": construction_policy}
    for tier in ("primary", "conditional"):
        admitted = [r for r in candidates if r["quality_tier"] == tier and r["qualification"]["construction_qualified"]]
        bank_ids = [r["bank_id"] for r in admitted]
        outputs[f"{tier}_bank_records.jsonl"] = admitted
        outputs[f"{tier}_bank_manifest.json"] = seal({
            "schema_version": MANIFEST_SCHEMA, "construction_version": VERSION,
            "quality_tier": tier, "benchmark": "openai/gsm8k", "bank_ids": bank_ids,
            "bank_count": len(admitted), "record_count": len(admitted),
            "record_order_sha256": canonical_hash(bank_ids),
            "record_sha256": {r["bank_id"]: r["record_sha256"] for r in admitted},
            "source_candidate_count": sum(r["quality_tier"] == tier for r in candidates),
            "evidence_count": sum(r["construction"]["distinct_sample_count"] for r in admitted),
            "inputs": bindings, "lineage_sha256": lineage["lineage_sha256"],
            "compiler_contract": COMPILER_CONTRACT, "offline_only": True,
            "qualified_for_online_use": False, "selector_artifact": None,
            "contains_runtime_answer_or_reward_signal": False,
            "status": "constructed_not_tensor_compiled" if admitted else "no_qualified_banks",
        }, "manifest_sha256")
    rejected = [r for r in candidates if not r["qualification"]["construction_qualified"]]
    outputs["quarantined_bank_records.jsonl"] = rejected
    outputs["clause_support_report.json"] = seal({"schema_version": "memgen-v4.3-clause-support-report-v1",
        "inputs": bindings, "minimum_distinct_support": 5, "semantic_entailment_claim": False,
        "banks": {r["bank_id"]: {"clause_support": r["clause_support"], "qualification": r["qualification"],
                                  "composed_field_support": r["composed_field_support"],
                                  "boundary_provenance": r["boundary_provenance"]} for r in candidates}}, "report_sha256")
    outputs["leakage_audit_report.json"] = seal({"schema_version": "memgen-v4.3-leakage-audit-report-v1",
        "inputs": bindings, "method": construction_policy["leakage_policy"],
        "semantic_factual_consistency_review_performed": False,
        "banks": {r["bank_id"]: {"card_audit": r["leakage_audit"],
                    "fields": {f: s["candidate_audit"] for f, s in r["clause_support"].items()}} for r in candidates}}, "report_sha256")
    outputs["construction_report.json"] = seal({"schema_version": "memgen-v4.3-construction-report-v1",
        "status": "construction_complete_with_quarantine" if rejected else "construction_complete",
        "inputs": bindings, "candidate_count": 17, "consumed_evidence_count": 116,
        "packet_evidence_count": sum(len(p["evidence"]) for p in packets),
        "unused_packet_candidate_ids": sorted(set(packet_map) - {r["candidate_id"] for r in candidates}),
        "source_tier_counts": {"primary": 11, "conditional": 6},
        "qualified_tier_counts": {t: len(outputs[f"{t}_bank_records.jsonl"]) for t in ("primary", "conditional")},
        "quarantined_count": len(rejected), "lineage_sha256": lineage["lineage_sha256"],
        "screened_clause_count": sum(len(s["candidate_audit"]) for r in candidates for s in r["clause_support"].values()),
        "flagged_candidate_clause_count": sum(bool(a["leakage_issues"]) for r in candidates for s in r["clause_support"].values() for a in s["candidate_audit"]),
        "consumption_rule": "all_members_all_five_signature_fields_no_sampling",
        "api_key_read": False, "external_api_calls_made": 0, "model_loaded": False,
        "embedding_artifact_used": False, "embedding_artifact_reason": "fieldwise_deterministic_CPU_sequence_representation",
        "original_signature_recomputed": False,
        "original_signature_authentication": "packet_file_and_packet_logical_hash_then_curated_signature_map",
        "offline_only": True, "qualified_for_online_use": False,
        "held_out_generalization_claim": False, "construction_evaluation_overlap": "construction_mechanism_audits_only",
        "selector_artifact": None, "tensor_artifact": None,
    }, "report_sha256")
    if report_metadata:
        outputs["construction_report.json"] = seal({**outputs["construction_report.json"], **report_metadata}, "report_sha256")
    return outputs


def validate_manifest(manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    authenticate(manifest, "manifest_sha256", "V4.3 manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("construction_version") != VERSION:
        raise ValueError("V4.3 manifest schema/version mismatch")
    if (manifest.get("offline_only") is not True or manifest.get("qualified_for_online_use") is not False
            or manifest.get("selector_artifact") is not None or manifest.get("contains_runtime_answer_or_reward_signal") is not False
            or manifest.get("compiler_contract") != COMPILER_CONTRACT):
        raise ValueError("V4.3 manifest offline/compiler contract mismatch")
    ids = [r["bank_id"] for r in records]
    if (len(ids) != len(set(ids)) or ids != manifest.get("bank_ids") or len(ids) != manifest.get("record_count")
            or len(ids) != manifest.get("bank_count") or manifest.get("record_order_sha256") != canonical_hash(ids)
            or manifest.get("record_sha256") != {r["bank_id"]: r["record_sha256"] for r in records}):
        raise ValueError("V4.3 manifest record binding mismatch")
    for record in records:
        validate_record(record)
        if record["quality_tier"] != manifest["quality_tier"] or not record["qualification"]["construction_qualified"]:
            raise ValueError("V4.3 manifest cannot admit quarantined or mixed-tier records")
    if manifest.get("evidence_count") != sum(r["construction"]["distinct_sample_count"] for r in records):
        raise ValueError("V4.3 manifest evidence count mismatch")
