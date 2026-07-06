from dataclasses import dataclass
import logging
import os
from typing import Optional, Literal, Set

from peft import PeftModel, LoraConfig
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    open_model_parameters
)

@dataclass
class MemGenOutputWithPast(CausalLMOutputWithPast):
    """在标准 CausalLMOutputWithPast 上额外带回训练时真正监督的位置。"""

    supervised_labels: Optional[torch.LongTensor] = None

class MemGenLoraSwitchMixin:
    """管理 Weaver/Trigger 两套 LoRA 的插入、冻结和打开。

    MemGen 有三份模型，但训练时通常只打开一个模块：
    - 训练 Weaver: 打开 Weaver LoRA、query latents 和双向 projection，冻结 Trigger；
    - 训练 Trigger: 打开 Trigger LoRA 和分类头，冻结 Weaver/projection；
    - Reasoner 始终冻结。
    """

    def _insert_lora_adapters(
        self, 
        weaver_model: PreTrainedModel, 
        weaver_lora_config: dict, 
        trigger_model: PreTrainedModel, 
        trigger_lora_config: dict
    ) -> tuple[PeftModel, PeftModel]:
        # insert lora adapters into weaver and trigger.
        # adapter_name 固定为 "weaver"/"trigger"，后续保存和恢复 checkpoint 时会按名字查找。
        weaver_lora_config = LoraConfig(**weaver_lora_config)
        trigger_lora_config = LoraConfig(**trigger_lora_config)
    
        weaver_model_with_lora = PeftModel(
            weaver_model, weaver_lora_config, adapter_name=MemGenWeaver.adapter_name
        )
        trigger_model_with_lora = PeftModel(
            trigger_model, trigger_lora_config, adapter_name=MemGenTrigger.adapter_name
        )

        return weaver_model_with_lora, trigger_model_with_lora
    
    def fix_component(self, name: Literal["weaver", "trigger"]):
        # frozen parameters of weaver or trigger.
        # 冻结 Weaver 时还要冻结 reasoner<->weaver projection，因为 projection 属于 latent 生成路径。
        component = getattr(self, name)
        fix_model_parameters(component)
        if name == "weaver":
            fix_model_parameters(self.weaver_to_reasoner)
            fix_model_parameters(self.reasoner_to_weaver)
    
    def open_component(self, name: Literal["weaver", "trigger"]):
        # open parameters of weaver or trigger.
        # 先整体打开 wrapper，再把 base_model 冻结回去，最后只保留目标 LoRA adapter 可训练。
        component = getattr(self, name)  
        open_model_parameters(component)
        if name == "weaver":
            open_model_parameters(self.weaver_to_reasoner)
            open_model_parameters(self.reasoner_to_weaver)        
        
        fix_model_parameters(component.model.base_model) # only finetune the lora adapters of the specific component
        
        for n, p in component.model.named_parameters():
            # PEFT 模型中可能同时挂着多个 adapter；这里确保只训练当前组件自己的 LoRA。
            if "lora_A" in n or "lora_B" in n:
                if name in n:
                    assert p.requires_grad, f"{n} should be trainable"
                else:
                    assert not p.requires_grad, f"{n} should be frozen"


class MemGenGenerationMixin(GenerationMixin):
    """MemGenModel 复用的生成/增强辅助函数。

    这些函数不持有独立状态，但默认 self 上存在 tokenizer、reasoner、trigger、
    delimiters 和 config 等属性，因此作为 mixin 被 MemGenModel 继承。
    """

    def _get_next_token(
        self, 
        next_token_logits: torch.Tensor, 
        do_sample: bool, 
        temperature: Optional[float] = 0.0
    ) -> torch.Tensor:
        """从最后一步 logits 选出下一个 token，支持 greedy 或温度采样。"""
        if len(next_token_logits.shape) != 2:
            raise ValueError("Input logits must be a 2D tensor [batch_size, vocab_size]")
        
        if do_sample and temperature != 0:  # Apply temperature scaling and sample from the resulting probability distribution    
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:  # Greedy decoding: pick the token with the highest probability
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    def _generate_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """根据左 padding attention_mask 生成连续 position_ids。"""
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _is_conversation(self, input_ids: torch.Tensor, tokenizer) -> bool:
        # if the input_ids has more than one <|im_start|>assistant\n, then it will be considered as a conversation.
        # 只检查 batch 中第一条样本；外层假设同一数据集的格式一致。
        if len(input_ids.shape) != 2:
            raise ValueError("input_ids must be a 2D tensor of shape (batch_size, seq_len)")
        
        seq = input_ids[0].tolist()

        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        target_seq = im_start_ids + assistant_ids

        count = 0
        for i in range(len(seq) - len(target_seq) + 1):
            if seq[i:i+len(target_seq)] == target_seq:
                count += 1

        return count > 1


    def _postprocess_assistant_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer
    ) -> torch.Tensor:
        """屏蔽 ChatML assistant header，避免特殊模板 token 参与 loss。

        数据集通常只想监督 assistant 的自然语言/代码答案，不想让
        `<|im_start|>assistant\n` 这类结构化标记影响 loss 或 prompt 边界判断。
        """
        if tokenizer.chat_template != CONVERSATION_TEMPLATE:
            raise ValueError(
                "Invalid tokenizer.chat_template detected.\n"
                f"Expected:\n{CONVERSATION_TEMPLATE}\n\n"
                f"Got:\n{tokenizer.chat_template}\n\n"
                "Please ensure that you are using the correct conversation template."
            )
        
        # Encode the token sequence for "<|im_start|>assistant\n".
        # 这里用 token id 精确匹配，避免 decode 后因空格/特殊 token 表示变化造成误判。
        pattern_ids: list[int] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        batch_size, seq_len = input_ids.shape
        new_labels = labels.clone()

        for b in range(batch_size):
            seq = input_ids[b].tolist()
            for i in range(len(seq) - len(pattern_ids) + 1):
                # Mask positions matching the pattern
                if seq[i : i + len(pattern_ids)] == pattern_ids:
                    new_labels[b, i : i + len(pattern_ids)] = -100

        return new_labels

    def _get_delimiter_token_ids(self, tokenizer, delimiters: list[str]) -> Set[int]:
        """预计算 delimiter 对应的 token ids，供生成/训练时快速判断边界。"""
        delimiter_token_ids = set()
        for d in delimiters:
            ids = tokenizer.encode(d, add_special_tokens=False)
            delimiter_token_ids.update(ids)
        return delimiter_token_ids

    def _check_ends_with_delimiter(
        self, input_ids: torch.Tensor, tokenizer, delimiters: list[str]
    ) -> torch.Tensor:
        """检查每个序列的最后一个有效 token 是否以 delimiter 结尾。

        返回形状为 (B, 1) 的 bool tensor。先走 token id 快速路径；如果最后一个
        token 是 `},\n` / `,\n` / `\n    ` 这类合并 token，再 decode 单个 token
        做文本后缀兜底。
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 获取最后一个有效 token (跳过 padding)。
        # 左 padding 时有效 token 在序列右侧，不能用 mask.sum()-1 推最后位置。
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        mask = input_ids != pad_token_id
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        masked_positions = torch.where(mask, positions, torch.full_like(positions, -1))
        last_positions = masked_positions.max(dim=1).values.clamp(min=0)
        last_tokens = input_ids[torch.arange(batch_size, device=device), last_positions]

        # 预计算并缓存 delimiter token ids tensor (只执行一次)。
        # 注意缓存 tensor 绑定当前 device；如果后续跨 device 复用模型，需要重新确认。
        cache_key = '_delimiter_token_tensor'
        delimiter_tensor = getattr(self, cache_key, None)
        if delimiter_tensor is None or delimiter_tensor.device != device:
            token_ids = self._get_delimiter_token_ids(tokenizer, delimiters)
            delimiter_tensor = torch.tensor(list(token_ids), device=device)
            setattr(self, cache_key, delimiter_tensor)

        is_delimiter = (last_tokens.unsqueeze(1) == delimiter_tensor).any(dim=1)

        # Some tokenizers merge punctuation/newlines with adjacent text. In that case
        # the token id is not equal to the standalone delimiter id, but the decoded
        # token text still ends with a delimiter.
        fallback_indices = (~is_delimiter).nonzero(as_tuple=True)[0]
        if fallback_indices.numel() > 0:
            delimiter_tuple = tuple(delimiters)
            text_cache_key = '_delimiter_text_match_cache'
            cached_delimiters, text_match_cache = getattr(self, text_cache_key, (None, {}))
            if cached_delimiters != delimiter_tuple:
                text_match_cache = {}
                setattr(self, text_cache_key, (delimiter_tuple, text_match_cache))

            for idx in fallback_indices.tolist():
                token_id = int(last_tokens[idx].item())
                text_match = text_match_cache.get(token_id)
                if text_match is None:
                    token_text = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                    )
                    text_match = token_text.rstrip(" \t").endswith(delimiter_tuple)
                    text_match_cache[token_id] = text_match

                if text_match:
                    is_delimiter[idx] = True

        return is_delimiter.unsqueeze(1)
    
    def _select_augment_points_after_delimiter(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        delimiters: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_num: int = 10,
    ) -> list[int]:
        """训练时选择 latent 插入点。

        返回的 index 都是“在原始 input_ids 的该位置之前插入 latent”：
        - 第一个 index: prompt augmentation 点，即第一个被监督 token 的位置；
        - 后续 index: completion 内 delimiter 后的 inference augmentation 点。

        这里要求 batch 内同一列都被监督，保证一次插入可以对整批样本对齐。
        """

        assert input_ids.shape == labels.shape
        B, seq_len = input_ids.size(0), input_ids.size(1)

        inference_augment_idx = []

        # 仍然用原版语义寻找 prompt 边界：labels 从 -100 切换到 supervised 的位置。
        # 但 completion 内部可能包含模型生成的 ChatML assistant header，随后会被
        # _postprocess_assistant_labels mask 掉，形成额外的 -100 -> supervised 跳变。
        # 这种额外跳变不是真实 prompt 边界，所以这里只取第一个边界作为 prompt augmentation 点。
        prompt_boundary_candidates = []
        for i in range(1, seq_len):
            if (labels[:, i] != -100).all() and (labels[:, i - 1] == -100).all():
                prompt_boundary_candidates.append(i)

        if len(prompt_boundary_candidates) == 0:
            logging.error("❌ No prompt augment boundary found for augmentation point selection")
            logging.error("Batch size = %d, seq_len = %d", B, seq_len)
            for b in range(B):
                logging.error("---- Sample %d ----", b)
                logging.error("Decoded text:\n%s", tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=False))
            raise ValueError("Single-turn forward requires at least one prompt augment boundary")

        prompt_augment_idx = prompt_boundary_candidates[0]

        for i in range(prompt_augment_idx + 1, seq_len):
            # Detect valid label regions for inference augmentation.
            # Masked gaps inside completions can occur when generated ChatML markers
            # are hidden from loss; they must not create extra prompt boundaries.
            if (labels[:, i] != -100).all() and (labels[:, i - 1] != -100).all():
                batch_tokens_before_i = input_ids[:, :i]
                # Fast token-level check (no decode)
                if self._check_ends_with_delimiter(batch_tokens_before_i, tokenizer, delimiters).any():
                    inference_augment_idx.append(i)

        final_points = [prompt_augment_idx]

        # Limit the number of inference augmentation points to max_num
        if len(inference_augment_idx) > max_num: 
            inference_augment_idx = inference_augment_idx[:max_num]

        final_points.extend(inference_augment_idx)
        
        if len(final_points) == 0:
            raise RuntimeError("No valid augmentation points found")
        
        final_points.sort()
        return final_points

    @torch.no_grad()
    def _should_augment(
        self, 
        input_ids: torch.LongTensor, 
        sentence_augment_count: torch.LongTensor, 
        do_sample: bool,
        temperature: float,
        is_prompt: bool = False
    ) -> torch.LongTensor:
        """生成时决定当前位置是否插入 latent。

        返回 aug_vector，长度为 batch_size：
        - -100: 当前位置不是候选点，不调用 trigger；
        - 0: 当前位置是候选点，但 trigger 选择不插入；
        - 1: 当前位置是候选点，且 trigger 选择插入 latent。
        """
            
        tokenizer = self.tokenizer
        delimiters = self.delimiters
        trigger = self.trigger
        max_augment_num = self.config.max_inference_aug_num

        batch_size = input_ids.size(0)
        
        if is_prompt:  
            # 第 0 步是 prompt augmentation 候选点。先把所有样本标记为候选，
            # 再交给 trigger 或 active=False 默认策略决定是否插入。
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            position_ids = self._generate_position_ids(attention_mask)
            aug_vector = torch.zeros((batch_size,), dtype=torch.long, device=input_ids.device)
            trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]

        else:  
            # inference augmentation 只在 delimiter 后触发，并受每条样本的增强次数上限约束。
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            position_ids = self._generate_position_ids(attention_mask)
            aug_vector = torch.full((batch_size,), -100, dtype=torch.long, device=input_ids.device)
            ends_with_delimiters = self._check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)
            aug_vector[ends_with_delimiters] = 0
            over_limit = (sentence_augment_count >= max_augment_num)
            aug_vector[over_limit] = -100
            trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]

        if trigger_indices.numel() > 0:
            # 只对候选样本跑 trigger，避免在非 delimiter 位置浪费前向计算。
            trigger_logits = trigger(
                input_ids=input_ids[trigger_indices],
                attention_mask=attention_mask[trigger_indices],
                position_ids=position_ids[trigger_indices]
            )
            last_token_logits = trigger_logits[:, -1]  # [batch, 2]

            next_tokens = self._get_next_token(
                last_token_logits,
                do_sample=do_sample,
                temperature=temperature
            ).view(-1)

            aug_vector[trigger_indices] = next_tokens

        return aug_vector


    @torch.no_grad()
    def _append_one_step(
        self,
        reasoner_outputs, 
        current_inputs_embeds: torch.Tensor,
        current_attention_mask: torch.Tensor,
        current_position_ids: torch.Tensor,
        current_input_ids: torch.Tensor,
        do_sample: bool,
        temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 reasoner 刚生成的一个真实 token 同步追加到四份生成状态中。"""
        B = current_inputs_embeds.size(0)
        
        # Append next token id.
        # current_input_ids 只包含真实 token，不包含 latent。
        next_token_logits = reasoner_outputs.logits[:, -1]
        next_token_ids = self._get_next_token(next_token_logits, do_sample=do_sample, temperature=temperature)
        current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)
        
        # Append next token embeds.
        # 下一轮 reasoner 可以直接复用 embedding 序列，无需重新嵌入整段 input_ids。
        next_token_embeds = self.reasoner.get_input_embeddings()(next_token_ids)
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embeds], dim=1)
        
        # Append attention mask.
        # 新生成 token 总是有效 token，mask 为 1。
        attn_mask = torch.ones((B, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
        current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
        
        # Append position ids.
        # 这里假设当前 batch 已通过左 padding 对齐，最后一列都是最新有效位置。
        next_position_id = current_position_ids[:, -1:] + 1
        current_position_ids = torch.cat([current_position_ids, next_position_id], dim=1)

        return current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids
    

    @torch.no_grad()
    def _left_pad(
        self,
        input_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_num: int
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """在左侧补 pad_num 个位置，用于 batch 内“有 latent/无 latent”长度对齐。"""
        
        if input_embeds is not None:
            B, L, D = input_embeds.shape
            pad_embeds = torch.zeros((B, pad_num, D), dtype=input_embeds.dtype, device=input_embeds.device)
            input_embeds = torch.cat([pad_embeds, input_embeds], dim=1)  # [B, pad_num + L, D]
        
        if attention_mask is not None:
            B = attention_mask.size(0)
            pad_mask = torch.zeros((B, pad_num), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)  # [B, pad_num + L]
        
        if position_ids is not None:
            B = position_ids.size(0)
            pad_pos = torch.zeros((B, pad_num), dtype=position_ids.dtype, device=position_ids.device)
            position_ids = torch.cat([pad_pos, position_ids], dim=1)  # [B, pad_num + L]

        return input_embeds, attention_mask, position_ids
    
    @torch.no_grad()
    def _left_clip_pad_tokens(
        self, inputs_embeds: torch.FloatTensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """裁掉 batch 中所有样本共有的左 padding，减少无效计算。"""

        B, L, D = inputs_embeds.shape

        # Find the index of the first non-padding token in each sequence
        first_nonpad_idx = []
        for b in range(B):
            nonzero = (attention_mask[b] != 0).nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                # Entire row is padding; can potentially trim the whole sequence
                first_nonpad_idx.append(L)
            else:
                first_nonpad_idx.append(nonzero[0].item())
        
        # Determine the minimum number of left-padding tokens across the batch
        min_pad = min(first_nonpad_idx)

        # If no padding on the left, return original tensors
        if min_pad == 0:
            return inputs_embeds, attention_mask, position_ids

        # Trim the left-padding from all sequences in the batch
        inputs_embeds = inputs_embeds[:, min_pad:, :]
        attention_mask = attention_mask[:, min_pad:]
        position_ids = position_ids[:, min_pad:]

        return inputs_embeds, attention_mask, position_ids
    
    @torch.no_grad()
    def _check_generate(self, input_ids: torch.LongTensor, augmentation_pos: torch.LongTensor):
        """检查 augmentation_pos[b][i] == 1 的位置, input_ids[b][:i] (不包括第 i 位) 对应的字符串是否以 delimiters 结尾
        仅在 DEBUG_MODE 下启用，避免训练时的性能开销
        """
        # 仅在 DEBUG 模式下执行验证，避免训练时的大量 decode 开销
        if os.environ.get('DEBUG_MODE', '').lower() != 'true':
            return

        delimiters = self.delimiters
        tokenizer = self.tokenizer

        B, L = input_ids.shape
        assert augmentation_pos.shape == input_ids.shape

        for b in range(B):
            for i in range(1, L):
                is_augment_point = augmentation_pos[b, i].item()

                if is_augment_point == -100:
                    continue

                if is_augment_point == 1 or is_augment_point == 0:
                    prefix_input_ids = input_ids[b, :i].unsqueeze(0)

                    ends_with_delimiter = self._check_ends_with_delimiter(
                        prefix_input_ids, tokenizer, delimiters
                    ).item()

                    if not ends_with_delimiter:
                        decoded_prefix = tokenizer.decode(prefix_input_ids.squeeze(0), skip_special_tokens=False)

                        raise ValueError(
                            f"Augmentation position error at batch {b}, index {i}. "
                            f"augmentation_pos is {is_augment_point}, but the prefix does NOT end with a delimiter.\n"
                            f"Prefix tail: {decoded_prefix[-80:]!r}\n"
                            f"Delimiters: {delimiters}"
                        )
                else:
                    raise ValueError(
                        f"Invalid value in augmentation_pos at batch {b}, index {i}: {is_augment_point}. "
                        "Expected 1, 0, or -100."
                    )
