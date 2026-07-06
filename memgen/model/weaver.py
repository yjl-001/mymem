from peft import PeftModel
import torch
import torch.nn as nn


class MemGenWeaver(nn.Module):
    """Latent memory 生成器。

    Weaver 本质上是一份带 LoRA 的 causal LM。它不会直接输出 token id，
    而是把一组可学习 query latent 拼到当前上下文后面，让模型最后几位
    hidden states 变成“记忆向量”。这些向量随后会被投影回 reasoner 的
    embedding 空间，作为不可见的 latent token 插入推理流。
    """

    adapter_name = "weaver"

    def __init__(
        self, 
        model: PeftModel,
        prompt_latents_len: int,
        inference_latents_len: int,
    ):
        super().__init__()

        self.model = model
        hidden_size = model.base_model.config.hidden_size

        # prompt augmentation:
        # 用在 generation 第 0 步或训练时 prompt -> completion 边界处。
        # 这组 latent 学的是“读完整个题目后，先写入哪些隐式记忆”。
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_len, hidden_size), 
            requires_grad=True
        )

        # inference augmentation:
        # 用在生成过程中遇到 delimiter 后。它和 prompt latent 分开训练，
        # 因为“题目前置记忆”和“中途续写记忆”的分布通常不同。
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_len, hidden_size), 
            requires_grad=True
        )

        # latent normalization + scale:
        # query latent 是直接优化的参数，LayerNorm + 可学习 scale 可以让
        # latent 幅度更稳定，避免把 weaver hidden state 推到异常范围。
        self.prompt_latent_ln = nn.LayerNorm(hidden_size)
        self.inference_latent_ln = nn.LayerNorm(hidden_size)
        self.prompt_latent_scale = nn.Parameter(torch.ones(1))
        self.inference_latent_scale = nn.Parameter(torch.ones(1))

    @property
    def prompt_latents_num(self) -> int:
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        assert self.prompt_query_latents.device == self.inference_query_latents.device
        return self.prompt_query_latents.device

    def _augment(
        self, 
        latents: torch.Tensor,
        latent_ln: nn.LayerNorm,
        latent_scale: torch.Tensor,
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 query latent 拼到上下文后，并返回这些 latent 位置的 hidden states。

        Args:
            latents: 形状为 (latent_len, H_weaver) 的可学习 query。
            inputs_embeds: 已经从 reasoner 空间投影到 weaver 空间的上下文，
                形状为 (B, L, H_weaver)。
            attention_mask/position_ids: 与 inputs_embeds 对齐的 mask 和位置。

        Returns:
            latents_hidden_states: (B, latent_len, H_weaver)，只取新拼接 latent
                位置的最后层 hidden states，作为本次生成的 latent memory。
            latents_mask: (B, latent_len)，给外层拼接 attention_mask 使用。
            latents_position_ids: (B, latent_len)，主要用于调试/形状对齐。
        """

        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)

        # normalize + scale
        latents = latent_ln(latents) * latent_scale
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 将同一组 query latent 复制到 batch 内每个样本后面。
        # 注意这里拼的是 embedding，不是 token id；这些 latent 对 tokenizer 不可见。
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # attention_mask: (B, L_total) padding mask, not causal mask
        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)
        
        # 让 latent 的 position id 接在真实上下文最后一个有效位置之后。
        # 对左 padding batch 来说，position_ids.max(dim=1) 是每行的最后有效位置。
        last_position_ids = position_ids.max(dim=1)[0] # 取每行的最大的位置值 [B]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device) # [latent_num]
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1 # [B, 1] + [latent_num] = [B, latent_num]
        # 拼接真实上下文和 latent 位置
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1) # [B, L_total]

        # weaver 只负责“加工” latent hidden states；真正的 loss/generation
        # 仍然发生在冻结的 reasoner 上。
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,  
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1] # 取最后一层 hidden states [B, L_total, D]
        latents_hidden_states = hidden_states[:, -latents_num:, :] # 取最后 latents_num 个位置的 hidden states [B, latent_num, D]

        return latents_hidden_states, latents_mask, latents_position_ids

    def augment_prompt(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.prompt_query_latents,
            latent_ln=self.prompt_latent_ln,
            latent_scale=self.prompt_latent_scale,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )


    def augment_inference(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.inference_query_latents,
            latent_ln=self.inference_latent_ln,
            latent_scale=self.inference_latent_scale,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
