import torch
from typing import Dict, List, Tuple
import copy

from interactions.base_interaction import (
    InteractionDataProto,
    InteractionConfig, 
    InteractionManager
)


class MultiTurnInteractionManager(InteractionManager):
    """DynamicEnv 使用的多轮 agent loop。

    每轮流程是：构造当前 chat history -> 模型生成 action -> env.step(action)
    -> 把 observation 作为下一轮 user message 追加回历史。
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        super().__init__(
            tokenizer, actor_rollout_wg, config, is_validation
        )

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(   
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _build_chat_history(self, rollings: Dict) -> List[Dict]:
        """把初始 system/user prompt 和已发生的交互历史拼成完整 messages。"""

        init_prompts = rollings.get("init_prompts")
        if init_prompts is None:
            raise ValueError("")
        
        inter_histories = rollings.get("inter_histories")
        if inter_histories is None:
            raise ValueError("")
        
        chat_histories: List[List[Dict]] = []
        for init_prompt, inter_history in zip(init_prompts, inter_histories):
            chat_histories.append(init_prompt + inter_history)

        return chat_histories
    
    def _update_interaction_history(self, rollings: InteractionDataProto, responses: List[str], observations: List[str]) -> List[List[Dict]]:
        """把本轮 assistant response 和 env observation 写回每条样本的历史。"""

        inter_histories = copy.deepcopy(rollings.no_tensor_batch.get("inter_histories"))
        assert len(inter_histories) == len(responses) == len(observations)
        for inter_history, response, observation in zip(inter_histories, responses, observations):
            assistant_info = {"role": "assistant", "content": response}
            user_info = {"role": "user", "content": observation}
            
            inter_history.append(assistant_info)
            inter_history.append(user_info)
        
        return inter_histories
    
    def _postprocess_responses(self, responses: torch.Tensor, envs: List) -> torch.Tensor:
        """把模型原始输出交给 env 清洗成合法 action，再重新 tokenize。"""

        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        processed_responses_str = []
        for r, env in zip(responses_str, envs):
            processed_r = env.preprocess_action(r)
            processed_responses_str.append(processed_r)

        responses = self._batch_tokenize(processed_responses_str)
        return responses, processed_responses_str


    def _example_level_pad(
        self, responses_ids: torch.Tensor, responses_str: List[str], active_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        """把只包含 active 样本的 response 填回原 batch 位置。"""

        assert active_mask.sum() == responses_ids.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses_ids.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.tokenizer.pad_token_id,
            dtype=responses_ids.dtype, device=responses_ids.device
        )
        padded_responses[active_mask] = responses_ids   
        
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str

    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """Run main LLM generation loop (conversation format)."""
        assert "init_prompts" in gen_batch.no_tensor_batch
        assert "envs" in gen_batch.no_tensor_batch
        batch_size = len(gen_batch.no_tensor_batch["init_prompts"])

        rollings = gen_batch   
        rollings.no_tensor_batch["inter_histories"] = [[] for _ in range(batch_size)]

        # active_mask 标记哪些样本还没 done；已完成样本后续不再送进模型生成。
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        all_augmentation_pos = []

        for step in range(self.config.max_turns):
            if not active_mask.sum():   
                break            

            mask_list = active_mask.tolist()  
            # 只保留 active 样本进入本轮生成，减少无效计算。
            rollings_active = {
                k: [item for item, keep in zip(v, mask_list) if keep]
                for k, v in rollings.no_tensor_batch.items()
            }
            # use tokenizer to add chat template and encode text to tokens: input_ids, attention_mask
            messages = self._build_chat_history(rollings_active)
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, 
                add_generation_prompt=True, 
                padding=True, return_tensors="pt", return_dict=True
            )

            # agent rollout
            result = self.actor_rollout_wg.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config,
                return_augmentation_mask=True,
            )
            gen_output = result[0].to("cpu")
            all_augmentation_pos.append(result[1].to("cpu"))

            # postprocess.
            # 模型输出先截掉 prompt，再截到 EOS；env 还可以进一步清洗 action 格式。
            prompt_len = inputs["input_ids"].size(1)
            responses = gen_output[:, prompt_len:]
            responses = self.tensor_fn.erase_after_first_eos(responses, self.tokenizer.eos_token_id)
            responses_ids, responses_str = self._postprocess_responses(responses, rollings_active["envs"])
            all_responses_ids, all_responses_str = self._example_level_pad(responses_ids, responses_str, active_mask)

            # env.step 返回 observation/done；observation 会作为下一轮 user message。
            next_obs, dones = self._execute_predictions(rollings, all_responses_str, active_mask)
            processed_obs = self._postprocess_observations(next_obs)
            
            # post process interaction states.
            # 一旦 done，样本会在后续轮次被 active_mask 排除。
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            interaction_histories = self._update_interaction_history(rollings, all_responses_str, processed_obs)
            rollings.no_tensor_batch["inter_histories"] = interaction_histories
  
        # build final outputs
        final_outputs = self._build_final_outputs(rollings)
        final_outputs.batch["augmentation_pos_list"] = all_augmentation_pos
        return final_outputs

    def _execute_predictions(self, rollings: InteractionDataProto, responses: List[str], active_mask: torch.Tensor) -> Tuple[List[str], List[str]]:
        """对 active 样本执行 env.step；inactive 样本填空 observation。"""
        observations = []
        dones = []
        for response, env, is_active in zip(responses, rollings.no_tensor_batch["envs"], active_mask):
            if is_active:
                observation, _, done = env.step(response)
            else:   
                observation = ""
                done = True
            observations.append(observation)
            dones.append(done)

        return observations, dones

    
    def _postprocess_observations(self, observations: List[str]) -> List[str]:
        """限制 observation token 长度，避免工具返回内容撑爆上下文。"""
        self.tokenizer.padding_side = "right" 
        next_obs_ids = self._batch_tokenize(observations)

        max_len = self.config.max_obs_length
        if next_obs_ids.shape[1] > max_len:
            extra_text = "..."
            extra_ids = self.tokenizer.encode(
                extra_text, add_special_tokens=False, return_tensors="pt"
            ).to(next_obs_ids.device)
            extra_len = extra_ids.shape[1]

            new_obs_ids = []
            for row in next_obs_ids:
                valid_len = (row != self.tokenizer.pad_token_id).sum().item()

                if valid_len > max_len:  
                    truncated = row[: max_len - extra_len]
                    new_row = torch.cat([truncated, extra_ids.squeeze(0)], dim=0)
                else:
                    new_row = row[:max_len]

                new_obs_ids.append(new_row.unsqueeze(0))

            next_obs_ids = torch.cat(new_obs_ids, dim=0)
            observations = self.tokenizer.batch_decode(next_obs_ids, skip_special_tokens=True)

        return observations

    def _build_final_outputs(self, rollings: InteractionDataProto) -> InteractionDataProto:
        """把多轮历史重新编码成 Trainer 需要的统一输出字段。"""

        init_prompts: List[List[Dict]] = rollings.no_tensor_batch["init_prompts"]
        inter_histories: List[List[Dict]] = rollings.no_tensor_batch["inter_histories"]
        
        output = InteractionDataProto()

        output.no_tensor_batch["inter_histories"] = [
            prompt + inter for prompt, inter in zip(init_prompts, inter_histories)
        ]
        
        # ---------- prompts ----------
        # prompts 只包含初始 system/user，不含 add_generation_prompt。
        self.tokenizer.padding_side = "left"
        prompt_ids = self.tokenizer.apply_chat_template(                
            init_prompts, tokenize=True, 
            add_generation_prompt=False,  
            padding=True, return_tensors="pt", return_dict=True
        )
        output.batch["prompts"] = prompt_ids["input_ids"]
        prompt_attn_mask = prompt_ids["attention_mask"]
        
        # ---------- responses ----------
        # responses 是交互历史；assistant_masks 用来标记哪些 token 属于 agent 输出。
        self.tokenizer.padding_side = "right"
        response_ids = self.tokenizer.apply_chat_template(                
            inter_histories, 
            tokenize=True, 
            padding=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_tensors="pt", return_dict=True
        )
        output.batch["responses"] = response_ids["input_ids"]
        response_attn_mask = response_ids["attention_mask"]

        completion_info_mask = response_ids["assistant_masks"]

        # ---------- input_ids ----------
        # Trainer 统一消费 input_ids = prompt + response/history。
        output.batch["input_ids"] = torch.cat(
            [prompt_ids["input_ids"], response_ids["input_ids"]], dim=1
        )
        output.batch["attention_mask"] = torch.cat(
            [prompt_attn_mask, response_attn_mask], dim=1
        )

        # ---------- info_mask ----------
        # prompt_info_mask 全 0，completion_info_mask 只监督 assistant token。
        prompt_info_mask = torch.zeros(
            prompt_ids["input_ids"].shape, 
            dtype=completion_info_mask.dtype, 
            device=completion_info_mask.device
        )

        output.batch["info_mask"] = torch.cat(
            [prompt_info_mask, completion_info_mask], dim=1
        )
        
        self.tokenizer.padding_side = "left"

        return output
