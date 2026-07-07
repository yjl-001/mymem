import torch

from model_setup import decode_completion
from records import CandidateRecord, GenerationTrace
from sink_metrics import add_candidate, normalize_candidate_scores


def insert_latent(model, current_inputs_embeds, current_attention_mask, current_position_ids, *, is_prompt: bool):
    """在当前 embedding 序列末尾插入一段 latent memory。

    这段逻辑对应 MemGenModel.generate 里的 augmentation 分支：
    1. reasoner embedding -> weaver hidden space；
    2. Weaver 追加 query latent 并生成 latent hidden states；
    3. weaver hidden -> reasoner embedding；
    4. 拼回当前 reasoner 输入序列。
    """

    # current_inputs_embeds: [B=1, L_current, H_reasoner]
    # weaver_inputs_embeds: [B=1, L_current, H_weaver]
    weaver_inputs_embeds = model.reasoner_to_weaver(current_inputs_embeds)
    if is_prompt:
        # weaver_hidden_states: [B=1, prompt_latents_len, H_weaver]
        # latent_mask: [B=1, prompt_latents_len]
        weaver_hidden_states, latent_mask, _ = model.weaver.augment_prompt(
            weaver_inputs_embeds, current_attention_mask, current_position_ids
        )
    else:
        # weaver_hidden_states: [B=1, inference_latents_len, H_weaver]
        # latent_mask: [B=1, inference_latents_len]
        weaver_hidden_states, latent_mask, _ = model.weaver.augment_inference(
            weaver_inputs_embeds, current_attention_mask, current_position_ids
        )
    # latent_inputs_embeds: [B=1, latent_len, H_reasoner]
    latent_inputs_embeds = model.weaver_to_reasoner(weaver_hidden_states)
    # 拼接后:
    # current_inputs_embeds: [B=1, L_current + latent_len, H_reasoner]
    # current_attention_mask/current_position_ids: [B=1, L_current + latent_len]
    current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
    current_attention_mask = torch.cat([current_attention_mask, latent_mask], dim=1)
    current_position_ids = model._generate_position_ids(current_attention_mask)
    return current_inputs_embeds, current_attention_mask, current_position_ids


@torch.no_grad()
def generate_with_forced_steps(model, prompt_ids, prompt_mask, *, sample_idx: int, forced_steps: set[int], args,
                               prompt_augment: bool = True, collect_candidates: bool = False) -> GenerationTrace:
    """单样本生成循环，支持强制在指定 delimiter step 插入 latent。

    这是本 MVP 的核心：它不是训练用 generate，而是一个可控实验循环。
    - collect_candidates=True：prompt-only baseline，记录 delimiter 候选点和 SinkMass。
    - forced_steps={j}：在候选点 j 做 single-insertion branch。
    - forced_steps={j1,j2,...}：模拟 first-K/random/sink_top_b 等策略。
    """

    tokenizer = model.tokenizer
    reasoner = model.reasoner
    # prompt_ids/prompt_mask: [B=1, L_prompt]
    current_input_ids = prompt_ids
    # current_inputs_embeds: [B=1, L_prompt, H_reasoner]
    current_inputs_embeds = reasoner.get_input_embeddings()(prompt_ids)
    current_attention_mask = prompt_mask
    # current_position_ids: [B=1, L_prompt]
    current_position_ids = model._generate_position_ids(current_attention_mask)
    current_cache = None
    candidates: list[CandidateRecord] = []
    forced_steps_used: list[int] = []

    for step in range(args.max_new_tokens):
        # 只有“当前真实 token 前缀以 delimiter 结尾”时，inference latent 才允许插入。
        # 如果 earlier insertion 让后续轨迹偏离 baseline，即使 forced_steps 里有该 step，
        # 这里也会跳过，避免在非边界位置插入。
        prefix_ends_with_delimiter = (
            step > 0 and model._check_ends_with_delimiter(current_input_ids, tokenizer, model.delimiters).item()
        )
        should_insert = (step == 0 and prompt_augment) or (step in forced_steps and prefix_ends_with_delimiter)
        if should_insert:
            current_inputs_embeds, current_attention_mask, current_position_ids = insert_latent(
                model, current_inputs_embeds, current_attention_mask, current_position_ids, is_prompt=(step == 0)
            )
            current_cache = None
            if step in forced_steps:
                forced_steps_used.append(step)

        # 插入 latent 会改变 embedding 序列长度，所以必须清空 cache；
        # 没有插入时可以沿用 KV cache，只喂最后一个 token embedding。
        if current_cache is not None:
            reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
            reasoner_position_ids = current_position_ids[:, -1:]
        else:
            reasoner_inputs_embeds = current_inputs_embeds
            reasoner_position_ids = current_position_ids

        outputs = reasoner(
            # reasoner_inputs_embeds:
            # - cache 为空: [B=1, L_current, H_reasoner]
            # - cache 可用: [B=1, 1, H_reasoner]
            inputs_embeds=reasoner_inputs_embeds,
            # attention_mask 总是完整上下文 mask: [B=1, L_current]
            attention_mask=current_attention_mask,
            # position_ids 与本次输入 embeds 对齐: [B=1, L_current] 或 [B=1, 1]
            position_ids=reasoner_position_ids,
            output_attentions=collect_candidates,
            output_hidden_states=False,
            use_cache=True,
            past_key_values=current_cache,
        )
        # outputs.logits: [B=1, L_query, vocab_size]；只取最后一个 query 的 next-token logits。
        logits = outputs.logits[:, -1]
        # baseline 轨迹才记录候选点。策略/branch 轨迹不重复记录，避免混入
        # “插入后新轨迹”的候选分布。
        if collect_candidates and prefix_ends_with_delimiter:
            add_candidate(candidates, model, current_input_ids, current_attention_mask, outputs, logits, sample_idx, step, args)

        # 这里复用项目原有的 greedy/sample 逻辑，保证和 MemGen.generate 取 token 一致。
        # next_token_ids: [B=1, 1]
        next_token_ids = model._get_next_token(logits, do_sample=args.do_sample, temperature=args.temperature)
        current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)
        # next_token_embeds: [B=1, 1, H_reasoner]
        next_token_embeds = reasoner.get_input_embeddings()(next_token_ids)
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embeds], dim=1)
        # next_mask: [B=1, 1]
        next_mask = torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
        current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)
        current_position_ids = model._generate_position_ids(current_attention_mask)
        current_cache = outputs.past_key_values

        if int(next_token_ids.item()) == tokenizer.eos_token_id:
            break

    # current_input_ids 只记录真实 token；latent 没有 token id，所以 completion ids
    # 可以直接从 prompt_len 之后切出来。
    prompt_len = prompt_ids.size(1)
    # current_input_ids: [B=1, L_prompt + L_generated]，latent 不在这里出现。
    completion_ids = current_input_ids[0, prompt_len:].tolist()
    actual_generated_len = max(len(completion_ids), 1)
    # 候选点是在生成过程中记录的；等整条 baseline 完成后，才能用真实生成长度
    # 重新计算相对位置，避免 step / max_new_tokens 把所有短生成都压到 bucket 0。
    for candidate in candidates:
        candidate.rel_pos = candidate.step / actual_generated_len
        candidate.pos_bucket = min(int(candidate.rel_pos * 4), 3)
    normalize_candidate_scores(candidates)
    return GenerationTrace(
        completion_ids=completion_ids,
        completion=decode_completion(model, completion_ids),
        candidates=candidates,
        forced_steps_used=forced_steps_used,
    )
