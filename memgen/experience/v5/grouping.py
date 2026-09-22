"""Order-stable semantic grouping: local partitions, candidate graph, and cluster edge audit."""
from __future__ import annotations

import math

from memgen.experience.bank_construction.artifacts import digest
from memgen.experience.bank_construction.parallel import ordered_map, teacher_workers
from .schemas import validate_cluster_review, validate_pair_judgments, validate_partition

STRATEGY = "v5-partition-candidate-graph-cluster-audit-v1"


def chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def atom_view(atom, alias=None):
    result = {"applicability": atom["applicability"], "experience": atom["experience"],
              "exclusions_from_input": atom["exclusions_from_input"],
              "failure_mechanism": atom["failure_mechanism"]}
    if alias is not None:
        result["atom_id"] = alias
    return result


def make_group(members, shared_method, applicability, exclusions, rationale, *, protocol_fallback=False):
    body = {"members": sorted(members), "shared_method": shared_method.strip(),
            "applicability": applicability.strip(), "exclusions": list(exclusions),
            "rationale": rationale.strip(), "protocol_fallback": bool(protocol_fallback)}
    return {**body, "group_id": "v5-group-" + digest(body)}


def initial_partition(store, teacher, atoms, batch_index):
    aliases = [f"A{index + 1:02d}" for index in range(len(atoms))]
    mapping = dict(zip(aliases, (atom["atom_id"] for atom in atoms)))
    payload = {"atoms": [atom_view(atom, alias) for atom, alias in zip(atoms, aliases)]}
    inputs = {"strategy": STRATEGY, "batch_index": batch_index, "payload": payload}
    key = f"grouping/initial/{batch_index:06d}"
    cached = store.get(key, inputs)
    if cached is not None:
        return cached
    try:
        answer = teacher.ask("partition_atoms", payload,
                             lambda value: validate_partition(value, aliases))
        groups = [make_group([mapping[alias] for alias in row["members"]], row["shared_method"],
                             row["applicability"], row["exclusions"], row["rationale"])
                  for row in answer["groups"]]
        failed = False
    except RuntimeError as exc:
        # Preserve every Atom, but mark these conservative singletons ineligible for Primary.
        groups = [make_group([atom["atom_id"]], atom["experience"]["do"][0],
                    atom["applicability"]["problem_structure"], atom["exclusions_from_input"],
                    "Protocol failure preserved this Atom as an auditable singleton.",
                    protocol_fallback=True) for atom in atoms]
        store.put(f"grouping/protocol_failures/initial/{batch_index:06d}",
                  {"error": str(exc), "atom_ids": list(mapping.values())}, inputs)
        failed = True
    result = {"groups": groups, "protocol_failure": failed}
    return store.put(key, result, inputs)


def group_text(group):
    return "\n".join(("Method: " + group["shared_method"],
                      "Applicability: " + group["applicability"],
                      "Exclusions: " + "; ".join(group["exclusions"])))


def group_vector(store, group, encoder, identity):
    inputs = {"group": group, "encoder": identity, "strategy": STRATEGY}
    key = "grouping/vectors/" + group["group_id"]
    cached = store.get(key, inputs)
    if cached is None:
        cached = store.put(key, {"vector": encoder(group_text(group))}, inputs)
    vector = cached["vector"]
    if not isinstance(vector, list) or not vector or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                                               for x in vector):
        raise ValueError("Invalid V5 grouping vector")
    return vector


def candidate_pairs(groups, vectors, top_k):
    if len(groups) < 2:
        return []
    try:
        import numpy as np
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Grouping vector matrix must be rank two")
        scores = matrix @ matrix.T
        pairs = set()
        for index, row in enumerate(scores):
            order = np.argsort(-row, kind="stable")
            added = 0
            for other in (int(value) for value in order if int(value) != index):
                pairs.add(tuple(sorted((index, other))))
                added += 1
                if added >= min(top_k, len(groups) - 1):
                    break
        return sorted(pairs)
    except ImportError:
        if any(len(vector) != len(vectors[0]) for vector in vectors):
            raise ValueError("Grouping vector dimensions differ")
        pairs = set()
        for index, vector in enumerate(vectors):
            ranked = sorted(((sum(a*b for a, b in zip(vector, other)), j)
                             for j, other in enumerate(vectors) if j != index),
                            key=lambda item: (-item[0], groups[item[1]]["group_id"]))
            pairs.update(tuple(sorted((index, j))) for _, j in ranked[:top_k])
        return sorted(pairs)


def judge_pairs(store, teacher, groups, pairs, batch_size):
    accepted, protocol_failures = [], 0
    work = list(enumerate(chunks(pairs, batch_size)))

    def process(item):
        batch_index, batch = item
        ids = [f"P{index + 1:02d}" for index in range(len(batch))]
        payload = {"pairs": [{"pair_id": pid,
                    "left": {k: groups[left][k] for k in ("shared_method", "applicability", "exclusions")},
                    "right": {k: groups[right][k] for k in ("shared_method", "applicability", "exclusions")}}
                   for pid, (left, right) in zip(ids, batch)]}
        inputs = {"strategy": STRATEGY, "batch_index": batch_index, "payload": payload}
        key = f"grouping/pair_judgments/{batch_index:06d}"
        cached = store.get(key, inputs)
        if cached is not None:
            return batch_index, cached
        try:
            answer = teacher.ask("judge_group_pairs", payload,
                                 lambda value: validate_pair_judgments(value, ids))
            by_pair_id = {row["pair_id"]: row for row in answer["judgments"]}
            same = [pair for pair_id, pair in zip(ids, batch) for row in [by_pair_id[pair_id]]
                    if row["relation"] == "same_method" and row["applicability_compatible"]
                    and not row["exclusion_conflict"]]
            result = {"accepted_pairs": same, "protocol_failure": False, "answer": answer}
        except RuntimeError as exc:
            result = {"accepted_pairs": [], "protocol_failure": True, "error": str(exc)}
        return batch_index, store.put(key, result, inputs)

    results = sorted(ordered_map(process, work, teacher_workers(teacher)))
    for _, result in results:
        accepted.extend(tuple(pair) for pair in result["accepted_pairs"])
        protocol_failures += int(result["protocol_failure"])
    return sorted(set(accepted)), protocol_failures


def components(count, edges):
    parent = list(range(count))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    for left, right in edges:
        union(left, right)
    result = {}
    for index in range(count):
        result.setdefault(find(index), []).append(index)
    return [indices for _, indices in sorted(result.items())]


def review_components(store, teacher, groups, edges, batch_size):
    rejected, failures = set(), 0
    for component_index, indices in enumerate(components(len(groups), edges)):
        inside = [edge for edge in edges if edge[0] in indices and edge[1] in indices]
        if len(indices) < 3 or not inside:
            continue
        for batch_index, batch in enumerate(chunks(inside, batch_size)):
            member_indices = sorted({value for edge in batch for value in edge})
            aliases = {index: f"G{position + 1:02d}" for position, index in enumerate(member_indices)}
            edge_aliases = {edge: f"E{position + 1:02d}" for position, edge in enumerate(batch)}
            payload = {"groups": [{"group_id": aliases[index], **{k: groups[index][k]
                        for k in ("shared_method", "applicability", "exclusions")}} for index in member_indices],
                       "proposed_edges": [{"edge_id": edge_aliases[edge], "left": aliases[edge[0]],
                                           "right": aliases[edge[1]]} for edge in batch]}
            inputs = {"strategy": STRATEGY, "component": component_index,
                      "batch_index": batch_index, "payload": payload}
            key = f"grouping/cluster_reviews/{component_index:06d}/{batch_index:06d}"
            cached = store.get(key, inputs)
            if cached is None:
                try:
                    answer = teacher.ask("review_cluster", payload,
                                         lambda value: validate_cluster_review(value, list(edge_aliases.values())))
                    inverse = {alias: edge for edge, alias in edge_aliases.items()}
                    result = {"rejected_pairs": [inverse[alias] for alias in answer["reject_edge_ids"]],
                              "protocol_failure": False, "answer": answer}
                except RuntimeError as exc:
                    # A failed audit cannot certify these semantic edges.
                    result = {"rejected_pairs": batch, "protocol_failure": True, "error": str(exc)}
                cached = store.put(key, result, inputs)
            rejected.update(tuple(pair) for pair in cached["rejected_pairs"])
            failures += int(cached["protocol_failure"])
    return [edge for edge in edges if edge not in rejected], failures


def run_grouping(store, teacher, config, encoder, encoder_identity):
    atoms_index = store.require("stages/atoms")
    atoms = sorted((store.require(key) for key in atoms_index["keys"]), key=lambda atom: atom["atom_id"])
    inputs = {"atoms": digest(atoms), "config": config.to_dict(), "strategy": STRATEGY,
              "encoder": encoder_identity}
    cached = store.get("stages/groups", inputs)
    if cached is not None:
        return cached
    batches = list(enumerate(chunks(atoms, config.atom_partition_batch_size)))
    initial = sorted(ordered_map(lambda item: (item[0], initial_partition(store, teacher, item[1], item[0])),
                                 batches, teacher_workers(teacher)))
    groups = sorted((group for _, result in initial for group in result["groups"]),
                    key=lambda group: group["group_id"])
    vectors = [group_vector(store, group, encoder, encoder_identity) for group in groups]
    proposed = candidate_pairs(groups, vectors, config.group_candidate_top_k)
    accepted, pair_failures = judge_pairs(store, teacher, groups, proposed, config.group_pair_batch_size)
    accepted, cluster_failures = review_components(
        store, teacher, groups, accepted, config.group_pair_batch_size)
    atom_by_id = {atom["atom_id"]: atom for atom in atoms}
    final = []
    for indices in components(len(groups), accepted):
        member_groups = [groups[index] for index in indices]
        members = sorted(atom for group in member_groups for atom in group["members"])
        if len(members) != len(set(members)):
            raise ValueError("V5 semantic groups overlap")
        body = {"member_group_ids": sorted(group["group_id"] for group in member_groups),
                "members": members,
                "protocol_fallback": any(group["protocol_fallback"] for group in member_groups),
                "distinct_input_count": len({atom_by_id[atom]["input_id"] for atom in members})}
        final.append({**body, "group_id": "v5-cluster-" + digest(body)})
    covered = [atom for group in final for atom in group["members"]]
    if len(covered) != len(set(covered)) or set(covered) != set(atom_by_id):
        raise ValueError("V5 grouping must preserve every accepted Atom exactly once")
    result = {"groups": sorted(final, key=lambda group: group["group_id"]),
        "atom_count": len(atoms), "initial_group_count": len(groups), "candidate_pair_count": len(proposed),
        "accepted_edge_count": len(accepted), "group_count": len(final),
        "protocol_failure_count": sum(result["protocol_failure"] for _, result in initial)
                                  + pair_failures + cluster_failures,
        "strategy": STRATEGY, "embedding_is_final_decision": False,
        "teacher_tasks": ["partition_atoms", "judge_group_pairs", "review_cluster"]}
    return store.put("stages/groups", result, inputs)
