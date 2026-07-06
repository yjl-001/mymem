from peft import PeftModel
import torch
import torch.nn as nn


class MemGenTrigger(nn.Module):
    """Memory Trigger：在候选位置上做二分类，决定是否插入 latent memory。

    输出最后一维大小为 2：
    - class 0: 不插入 latent
    - class 1: 插入 latent

    训练 Trigger 时，这个模块是一份独立 LoRA 模型加一个线性分类头；
    不训练 Trigger 时，active=False 会退化成“默认总是插入”，方便只研究 Weaver。
    """

    adapter_name = "trigger"

    def __init__(
        self,
        model: PeftModel,
        active: bool,
    ):
        super().__init__()
        
        self.active = active
        self.model = model
        # 直接用 base LM 的最后层 hidden state 做 token-level 二分类。
        # 外层只会读取最后一个 token 的 logits，因为是否增强取决于当前前缀状态。
        self.output_layer = nn.Linear(model.base_model.config.hidden_size, 2)

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.LongTensor,
        position_ids: torch.Tensor
    ) -> torch.FloatTensor:
        
        if self.active:
            # active=True: 正常跑 trigger backbone，并在每个 token 位置输出二分类 logits。
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,  
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            logits = self.output_layer(hidden_states)
        
        else:
            # active=False: 关闭真实 trigger。这里人为让 class 1 的 logit 更大，
            # 因此外层 greedy 时会把所有候选点都判为“插入 latent”。
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)  # logits: [batch_size, seq_len, 2]
            logits[..., 1] = 1.0  

        return logits


