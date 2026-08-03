from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Optional

from transformers import GenerationConfig

from interactions.tensor_utils import TensorHelper, TensorConfig


@dataclass
class InteractionConfig:
    """一次 agent rollout 的长度、采样和输出配置。"""

    max_turns: int = 1
    max_start_length: int = 1024   
    max_prompt_length: int = 4096   
    max_response_length: int = 512
    max_obs_length: int = 512
    # do_sample: bool = False
    temperature: float = 1.0  
    batch_size: int = 8
    output_dir: Optional[str] = None
    weaver_do_sample: bool = False
    trigger_do_sample: bool = False
    sink_aware_generation: bool = False
    record_entropy_gate_trace: bool = False
    evaluation_split: str = "test"

@dataclass
class InteractionDataProto:
    """InteractionManager 内部传递的 batch 容器。

    batch: 放 tensor，例如 input_ids、attention_mask、responses；
    no_tensor_batch: 放 Python 对象，例如原始 prompt、env 实例、交互历史。
    """

    batch: dict = field(default_factory=dict)
    no_tensor_batch: dict = field(default_factory=dict)

class InteractionManager(ABC):
    """把 MemGenModel.generate 包装成静态/动态任务都能复用的 agent loop。"""
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" 
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        assert tokenizer.pad_token_id is not None
        # TensorHelper 统一处理左/右 padding、截断、attention mask 等张量整理逻辑。
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        # generation configs for agent.
        # weaver_do_sample / trigger_do_sample 是项目自定义字段，MemGenModel.generate 会读取。
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        self.generation_config.weaver_do_sample = self.config.weaver_do_sample
        self.generation_config.trigger_do_sample = self.config.trigger_do_sample
        self.generation_config.sink_aware_generation = self.config.sink_aware_generation

        logging.info(
            "Weaver do sample: %s, Trigger do sample: %s, Sink-aware generation: %s",
            self.generation_config.weaver_do_sample,
            self.generation_config.trigger_do_sample,
            self.generation_config.sink_aware_generation,
        )
    
    @abstractmethod
    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        ...
