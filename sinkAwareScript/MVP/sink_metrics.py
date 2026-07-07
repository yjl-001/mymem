import math

import torch

from records import CandidateRecord


def entropy_from_logits(logits: torch.Tensor) -> float:
    """计算 next-token 分布熵，作为 uncertainty signal。"""

    # logits: [B=1, vocab_size]
    # probs/log_probs: [B=1, vocab_size]
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return float(-(probs * log_probs).sum().item())


def sink_mass_from_attentions(attentions, attention_mask, *, sink_key_count: int, sink_layer_window: int) -> float:
    """计算当前 query 对 sink key 的 attention mass。

    MVP 暂时把“前 sink_key_count 个有效 key position”当作 sink key 集合。
    这不是最终定义，而是为了先验证 online signal 是否有用；如果有效，
    后面再接入离线校准出的 S_sink / H_sink。
    """

    if not attentions:
        return float("nan")

    # attention_mask: [B=1, L_key]，非 0 的位置是真实 token/latent。
    # valid_positions: [L_valid]
    valid_positions = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return float("nan")
    # sink_positions: [min(sink_key_count, L_valid)]
    sink_positions = valid_positions[:sink_key_count]

    # sink_layer_window=0 表示所有层都平均；否则只看最后 N 层。
    selected_layers = attentions[-sink_layer_window:] if sink_layer_window > 0 else attentions
    masses = []
    for layer_attn in selected_layers:
        # layer_attn: [B=1, num_heads, L_query, L_key]
        # 在 use_cache=True 时 L_query 通常是 1；cache 为空时可能是完整 L_current。
        # 这里只看最后一个 query 对所有 key 的 attention。
        query_to_keys = layer_attn[0, :, -1, :]
        # query_to_keys: [num_heads, L_key]
        key_limit = query_to_keys.size(-1)
        usable_positions = sink_positions[sink_positions < key_limit]
        if usable_positions.numel() == 0:
            continue
        # query_to_keys[:, usable_positions]: [num_heads, num_sink_keys]
        # per-head 先对 sink keys 求和，再对 heads/layers 平均。
        masses.append(query_to_keys[:, usable_positions].sum(dim=-1).float().mean())

    if not masses:
        return float("nan")
    return float(torch.stack(masses).mean().item())


def normalize_candidate_scores(candidates: list[CandidateRecord]) -> None:
    """对单个样本内部的 sink_mass/entropy 做 z-score。

    这样比较的是“这个样本自己的 delimiter 候选点里，哪个更异常”，而不是
    跨样本直接比较原始 attention 数值。
    """

    for attr, z_attr in [("sink_mass", "sink_mass_z"), ("entropy", "entropy_z")]:
        values = [getattr(c, attr) for c in candidates if math.isfinite(getattr(c, attr))]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        std = math.sqrt(variance)
        if std < 1e-8:
            std = 1.0
        for candidate in candidates:
            value = getattr(candidate, attr)
            setattr(candidate, z_attr, (value - mean) / std if math.isfinite(value) else 0.0)


def add_candidate(candidates, model, current_input_ids, current_attention_mask, outputs, logits, sample_idx, step, args):
    """把当前 delimiter 后候选点加入 trace。

    注意：这个函数在“生成第 step 个 token 之前”调用。此时 current_input_ids
    的最后一个真实 token 已经是 delimiter，所以 step 是可插入 inference latent
    的位置。
    """

    tokenizer = model.tokenizer
    # current_input_ids: [B=1, L_real_tokens]，只包含真实 token，不包含 latent。
    # current_attention_mask: [B=1, L_real_tokens + L_latents_so_far]
    # logits: [B=1, vocab_size]
    delimiter_id = int(current_input_ids[0, -1].item())
    candidates.append(
        CandidateRecord(
            sample_idx=sample_idx,
            reference_mode=getattr(args, "current_reference_mode", "unknown"),
            reference_reward=float("nan"),
            candidate_rank=len(candidates),
            step=step,
            generated_so_far=step,
            # 这里先放临时值；generation.py 会在拿到 actual_generated_len 后重算。
            rel_pos=step / max(args.max_new_tokens, 1),
            pos_bucket=min(int((step / max(args.max_new_tokens, 1)) * 4), 3),
            delimiter_token_id=delimiter_id,
            delimiter_text=tokenizer.decode([delimiter_id], skip_special_tokens=False),
            sink_mass=sink_mass_from_attentions(
                outputs.attentions,
                current_attention_mask,
                sink_key_count=args.sink_key_count,
                sink_layer_window=args.sink_layer_window,
            ),
            sink_mass_z=0.0,
            entropy=entropy_from_logits(logits),
            entropy_z=0.0,
        )
    )
