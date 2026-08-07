import csv
import os
import random

from accelerate import Accelerator
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig, GRPOConfig
from trl.models import unwrap_model_for_generation

from data import (
    BaseBuilder,
)
from interactions.base_interaction import (
    InteractionConfig,   
    InteractionManager, 
    InteractionDataProto
)
from interactions.singleturn_interaction import SingleTurnInteractionManager
from interactions.multiturn_interaction import MultiTurnInteractionManager

from memgen.model.modeling_memgen import MemGenModel
from memgen.trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from memgen.trainer.trigger_grpo_trainer import TriggerGRPOTrainer
from memgen.utils import (
    StaticEvalRecorder,
    DynamicEvalRecorder,
    create_tensorboard,
    log_trainable_params,
    gather_objects
)


class MemGenRunner:
    """训练和评测的编排层。

    Runner 不实现模型细节，而是负责把以下对象接起来：
    - data_builder: 构造 train/valid/test 数据集和 Env 类型；
    - env: 提供 reward/feedback；
    - interaction_manager: 把模型生成包装成单轮或多轮 agent loop；
    - trainer: 根据配置选择 Weaver SFT、Weaver GRPO 或 Trigger GRPO。
    """

    def __init__(
        self,
        model: MemGenModel,
        data_builder: BaseBuilder,
        config: dict,
        working_dir: str,
    ):  
        # parse configs.
        # _parse_configs 会把 run 配置拆成 TRL TrainingArguments 和 InteractionConfig。
        self.config = config
        self.working_dir = working_dir

        self._parse_configs(config.get("run"))  
        
        # parse model.
        # processing_class 是 tokenizer，沿用 TRL 命名，后面传给 SFT/GRPO Trainer。
        self.processing_class = model.tokenizer
        self.model = model

        # initialize envs and generation managers.
        # Env 只保存数据集相关的 reward/交互逻辑；静态任务通常一次生成即可，
        # 动态任务会在 InteractionManager 里多轮调用 env。
        self.dataset_dict = data_builder.get_dataset_dict()
        self.env_cls = data_builder.get_env_cls()
        self.env = self.env_cls(config.get("dataset"))

        # partition datasets.
        # Weaver 和 Trigger 使用不同训练子集：Trigger 默认只从训练/验证集中采样一部分，
        # 降低二阶段 GRPO 的成本。
        self.weaver_train_dataset, self.trigger_train_dataset = self._parse_train_dataset(self.dataset_dict["train"])
        self.weaver_valid_dataset, self.trigger_valid_dataset = self._parse_valid_dataset(self.dataset_dict["valid"])
        self.test_dataset = self.dataset_dict["test"]
        
        self.weaver_train_dataset = self._filter_dataset(self.weaver_train_dataset)
        self.trigger_train_dataset = self._filter_dataset(self.trigger_train_dataset)
        self.weaver_valid_dataset = self._filter_dataset(self.weaver_valid_dataset)
        self.trigger_valid_dataset = self._filter_dataset(self.trigger_valid_dataset)
        if self.interaction_config.evaluation_split == "valid":
            self.evaluation_dataset = self.weaver_valid_dataset
        elif self.interaction_config.evaluation_split == "test":
            self.evaluation_dataset = self.test_dataset
        else:
            raise ValueError("run.interaction.evaluation_split must be 'valid' or 'test'.")
        
        # initialize generation manager.
        # StaticEnv -> SingleTurnInteractionManager；DynamicEnv -> MultiTurnInteractionManager。
        if self.env_cls.ENV_CARD == "STATIC":
            self.inter_cls = SingleTurnInteractionManager
        elif self.env_cls.ENV_CARD == "DYNAMIC":
            self.inter_cls = MultiTurnInteractionManager
        else: 
            raise ValueError("Unsupported environment type.")
        
        self.generation_manager: InteractionManager = self.inter_cls(
            self.processing_class, self.model, self.interaction_config
        )
    
    def _parse_train_dataset(self, train_dataset: Dataset) -> tuple[Dataset, Dataset]:
        # use part of the dataset to train the trigger.
        # 这里实际取 1/3；Weaver 仍使用完整 train_dataset。
        trigger_trainset_size = min(len(train_dataset) // 3, len(train_dataset))
        rand_indices = random.sample(range(len(train_dataset)), trigger_trainset_size)
        return train_dataset, train_dataset.select(rand_indices)
    
    def _parse_valid_dataset(self, valid_dataset: Dataset) -> tuple[Dataset, Dataset]:
        # 验证集也按同样规则给 Trigger 抽子集，避免二阶段评估成本过高。
        trigger_validset_size = min(len(valid_dataset) // 3, len(valid_dataset))
        rand_indices = random.sample(range(len(valid_dataset)), trigger_validset_size)
        return valid_dataset, valid_dataset.select(rand_indices)

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        """过滤 prompt 过长样本，避免进入 trainer 后才因为长度超限失败。"""
        tokenizer = self.processing_class

        # Determine max length based on training mode.
        # SFT 用 max_length；GRPO 用 max_prompt_length，因为 completion 由 rollout 生成。
        max_len = 1024
        if self.run_mode == "evaluate":
            # Evaluation has no trainer stage. Filter only by the initial
            # context budget configured for the interaction loop.
            max_len = self.interaction_config.max_start_length
        elif self.train_weaver and self.train_weaver_method == "sft":
            max_len = self.weaver_sft_training_args.max_length
        elif self.train_weaver and self.train_weaver_method == "grpo":
            max_len = self.weaver_grpo_training_args.max_prompt_length
        elif self.train_trigger and self.train_trigger_method == "grpo":
            max_len = self.trigger_grpo_training_args.max_prompt_length
        else:
            raise ValueError("Wrong training mode.")

        # Function to filter out samples exceeding max length.
        # Static 数据通常有 prompt 字段；Dynamic/多轮数据通常有 messages 字段。
        def filter_func(sample):
            if "prompt" in sample and sample["prompt"] is not None:
                prompt = sample["prompt"]
                # Static builders historically support both raw strings (GPQA)
                # and conversational prompts (GSM8K/KodCode).  A raw string is
                # not valid input to apply_chat_template, so encode it directly.
                if isinstance(prompt, str):
                    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                else:
                    prompt_ids = tokenizer.apply_chat_template(
                        prompt, tokenize=True, add_generation_prompt=True
                    )
                return len(prompt_ids) < max_len
            elif "messages" in sample and sample["messages"] is not None:
                conversation = tokenizer.apply_chat_template(sample["messages"][:2], tokenize=True)
                return len(conversation) < max_len
            return True 

        # Apply filtering
        dataset = dataset.filter(filter_func)

        return dataset

    def _tokenize_static_prompts(self, prompts: list):
        """Encode static-evaluation prompts using the same convention as GRPO.

        Static datasets can contain either raw strings or ChatML message lists.
        Keep the two formats separate instead of passing raw strings to
        ``apply_chat_template``.  The chat template already emits structural
        special tokens, so neither path asks the tokenizer to add another set.
        """
        if not prompts:
            raise ValueError("Cannot generate from an empty static-evaluation batch.")

        are_strings = [isinstance(prompt, str) for prompt in prompts]
        if any(are_strings) and not all(are_strings):
            raise TypeError(
                "A static-evaluation batch must contain either only string prompts "
                "or only chat-message prompts."
            )

        self.processing_class.padding_side = "left"
        if all(are_strings):
            return self.processing_class(
                text=prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

        return self.processing_class.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        )
    
    # ===== train weaver =====
    def _create_weaver_trainer(self):
        """根据 train_weaver_method 创建 Weaver 的 SFT 或 GRPO trainer。"""

        # SFT Trainer
        if self.train_weaver_method == "sft":

            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_sft_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
            )
        
        # GRPO Trainer
        elif self.train_weaver_method == 'grpo':
            # GRPO 阶段的评估由外部 evaluate 路径完成，这里关闭 TRL 内置 eval。
            self.weaver_grpo_training_args.do_eval = False
            self.weaver_grpo_training_args.eval_strategy = 'no'
            # Weaver GRPO 训练时：
            # - weaver_do_sample=True，让 latent/response 产生可探索的 rollout；
            # - trigger_do_sample=False，固定触发策略，避免同时优化两个模块。
            self.generation_manager.generation_config.weaver_do_sample = True
            self.generation_manager.generation_config.trigger_do_sample = False
            self.generation_manager.generation_config.temperature = self.weaver_grpo_training_args.temperature
            self.generation_manager.generation_config.max_new_tokens = self.weaver_grpo_training_args.max_completion_length
            
            # self.weaver_train_dataset = self.weaver_train_dataset.select(range(1600))
            
            weaver_trainer = WeaverGRPOTrainer(
                model=self.model,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.weaver_grpo_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
                # --- add env into trainer ---
                # Trainer 需要 env 和 generation_manager 来执行 rollout 并计算 reward。
                env_class=self.env_cls,
                env_main_config=self.config.get("dataset"),
                generation_manager=self.generation_manager,
            )
        else:
            raise ValueError("Unsupported weaver training method.")

        return weaver_trainer
    
    # ===== train trigger =====
    def _create_trigger_trainer(self):
        """创建 Trigger 的 GRPO trainer。Trigger 当前只支持 GRPO。"""
        
        if self.train_trigger_method == "grpo":
            self.trigger_grpo_training_args.do_eval = False
            self.trigger_grpo_training_args.eval_strategy = 'no'

            # Trigger GRPO 训练时固定 Weaver 的生成采样，只让 Trigger 决策探索。
            self.generation_manager.generation_config.trigger_do_sample = True
            self.generation_manager.generation_config.weaver_do_sample = False
            self.generation_manager.generation_config.temperature = self.weaver_grpo_training_args.temperature
            self.generation_manager.generation_config.max_new_tokens = self.weaver_grpo_training_args.max_completion_length
            
            trigger_trainer = TriggerGRPOTrainer(
                model=self.model, 
                processing_class=self.processing_class, 
                train_dataset=self.trigger_train_dataset, 
                eval_dataset=self.trigger_valid_dataset, 
                reward_funcs=[self.env_cls.compute_reward],
                args=self.trigger_grpo_training_args
            )
        else:
            raise ValueError("Unsupported trigger training method.")

        return trigger_trainer
    
    # ===== train weaver/trigger =====
    def train(self):
        """按配置执行单阶段训练：要么训练 Weaver，要么训练 Trigger。"""

        if self.train_weaver:
            trainer = self._create_weaver_trainer()
            # 训练 Weaver 时冻结 Trigger；projection 层会随 Weaver 一起打开/冻结。
            self.model.fix_component('trigger')
            
        if self.train_trigger:
            trainer = self._create_trigger_trainer()
            # 训练 Trigger 时冻结 Weaver，避免 reward 信号同时改动两套策略。
            self.model.fix_component('weaver')

        log_trainable_params(self.model)

        try:
            trainer.train()
            trainer.save_model()
        except RuntimeError as e:
            # 检查是否是 OOM 相关的错误
            if "OOM" in str(e) or "out of memory" in str(e).lower():
                logging.error(f"[Runner] Training stopped due to OOM: {e}")
                # 尝试最后一次保存
                try:
                    oom_dir = os.path.join(self.working_dir, "model_oom_final")
                    logging.info(f"[Runner] Attempting to save final checkpoint to {oom_dir}")
                    trainer.save_model(oom_dir)
                    logging.info(f"[Runner] Final checkpoint saved successfully")
                except Exception as save_e:
                    logging.error(f"[Runner] Failed to save final checkpoint: {save_e}")
                raise
            else:
                # 非 OOM 错误，直接抛出
                raise
        

    # ===== evaluate =====
    def evaluate(self):
        """根据 Env 类型选择静态或动态评测流程。"""
        self.model = self.model.to(torch.bfloat16)

        evaluate_func_mapping = {
            "STATIC": self._static_evaluate,
            "DYNAMIC": self._dynamic_evaluate
        }
        evaluate_func = evaluate_func_mapping.get(self.env.ENV_CARD)
        if evaluate_func is None:
            raise ValueError("The env has unrecogonized ENV_CARD attribute")
        
        return evaluate_func()
    
    def _static_evaluate(self):
        """静态任务评测：prompt -> completion -> reward。"""

        accelerator = Accelerator()

        if accelerator.is_main_process:
            writer = create_tensorboard(save_dir=self.working_dir)
            save_file = os.path.join(self.interaction_config.output_dir, "answer.json")
            recorder = StaticEvalRecorder(
                compute_metrics=[self.env_cls.compute_reward],
                writer=writer,
                log_file=save_file
            )
        else:
            writer = None
            recorder = None

        batch_size = self.interaction_config.batch_size
        csv_path = os.path.join(self.interaction_config.output_dir, "augmentation_positions.csv")
        entropy_csv_path = os.path.join(self.interaction_config.output_dir, "entropy_gate_trace.csv")
        all_aug_rows = []
        all_entropy_rows = []
        global_sample_offset = 0
        entropy_sample_offset = 0

        # Accelerator 只切分 dataloader；真实 generation 在 unwrap_model_for_generation 中执行，
        # 这样可以拿到未被 DDP 包裹的模型对象。
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.evaluation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch
        ))

        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        for test_batch in tqdm(test_dataloader, disable=not accelerator.is_main_process):
            with unwrap_model_for_generation(model_wrapped, accelerator) as unwrapped_model:
                prompts = [x["prompt"] for x in test_batch]
                prompt_inputs = self._tokenize_static_prompts(prompts)
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                gen_batch = InteractionDataProto()
                # InteractionManager 统一消费 InteractionDataProto：
                # batch 放 input_ids/attention_mask。单轮 manager 不消费原始 prompt，
                # 因此不要在 no_tensor_batch 中留下看似会被使用的字段。
                gen_batch.batch["input_ids"] = prompt_ids.to(accelerator.device)
                gen_batch.batch["attention_mask"] = prompt_mask.to(accelerator.device)

                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)
                augmentation_pos = gen_output.batch.get("augmentation_pos")
                aug_pos_cpu = augmentation_pos.cpu() if augmentation_pos is not None else None
                entropy_gate_trace = gen_output.no_tensor_batch.get("entropy_gate_trace", [])

                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                pad_id = self.processing_class.pad_token_id
                eos_id = self.processing_class.eos_token_id
                valid_mask = completion_ids.ne(pad_id)
                is_eos = completion_ids.eq(eos_id) & valid_mask
                has_eos = is_eos.any(dim=-1)
                # argmax returns 0 for rows without EOS; torch.where selects
                # non_pad_counts for those rows, so no Python-side correction is needed.
                first_eos = is_eos.int().argmax(dim=-1)
                non_pad_counts = valid_mask.sum(dim=-1)
                generated_token_counts = torch.where(
                    has_eos,
                    first_eos + 1,  # Count the EOS generation step.
                    non_pad_counts,
                )

            # only main rank can write the json
            local_completions = completions
            local_batches = test_batch
            local_aug_pos = aug_pos_cpu
            # all_gather_object serializes Python objects; stage this small
            # tensor on CPU but keep it as a tensor until recorder output.
            local_token_counts = generated_token_counts.cpu()
            local_entropy_gate_trace = {
                "batch_size": len(test_batch),
                "events": entropy_gate_trace,
            }

            all_completions = gather_objects(local_completions)
            all_batches = gather_objects(local_batches)
            all_aug_pos_list = gather_objects(local_aug_pos) if local_aug_pos is not None else []
            all_token_counts = gather_objects(local_token_counts)
            all_entropy_payloads = gather_objects(local_entropy_gate_trace)

            if accelerator.is_main_process:
                for comps, batch, token_counts in zip(all_completions, all_batches, all_token_counts):
                    recorder.record_batch(
                        comps,
                        batch,
                        generated_token_counts=token_counts.tolist(),
                    )

                for rank_aug_pos in all_aug_pos_list:
                    if rank_aug_pos is not None:
                        for b in range(rank_aug_pos.size(0)):
                            gen_len = rank_aug_pos.size(1)
                            for i in range(gen_len):
                                if rank_aug_pos[b, i].item() == 1:
                                    aug_type = "prompt" if i == 0 else "inference"
                                    relative_pos = i / max(gen_len, 1)
                                    all_aug_rows.append({
                                        "sample_idx": global_sample_offset + b,
                                        "aug_type": aug_type,
                                        "step_in_generation": i,
                                        "total_generated_len": gen_len,
                                        "relative_position": round(relative_pos, 4),
                                    })
                    if rank_aug_pos is not None:
                        global_sample_offset += rank_aug_pos.size(0)

                if isinstance(all_entropy_payloads, dict):
                    all_entropy_payloads = [all_entropy_payloads]
                for payload in all_entropy_payloads:
                    for event in payload["events"]:
                        row = {"sample_idx": entropy_sample_offset, **event}
                        all_entropy_rows.append(row)
                    entropy_sample_offset += int(payload["batch_size"])

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            recorder.finalize()
            writer.close()

            if all_aug_rows:
                with open(csv_path, "w", newline="") as f:
                    csv_writer = csv.DictWriter(f, fieldnames=["sample_idx", "aug_type", "step_in_generation", "total_generated_len", "relative_position"])
                    csv_writer.writeheader()
                    csv_writer.writerows(all_aug_rows)

            if all_entropy_rows:
                fieldnames = [
                    "sample_idx",
                    "generation_step",
                    "entropy",
                    "entropy_head_std",
                    "entropy_threshold",
                    "mean_sink_attention_mass",
                    "valid_key_count",
                    "sink_token_count",
                    "non_sink_key_count",
                    "inference_augmentation_count_before",
                    "budget_exhausted",
                    "insert",
                ]
                with open(entropy_csv_path, "w", newline="") as f:
                    csv_writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    csv_writer.writeheader()
                    csv_writer.writerows(all_entropy_rows)

    def _dynamic_evaluate(self):
        """动态任务评测：每条样本先创建 Env，再由 agent loop 多轮交互。"""
        
        def _set_batch_envs(batch: list) -> tuple[list[str], list[str], list]:  # batch set envs
            # 每个样本需要独立 env 实例保存检索/反馈状态。
            system_prompts, init_user_prompts, envs = [], [], []
            for task_config in batch:
                env = self.env_cls(self.config.get("dataset"))
                system_prompt, init_user_prompt = env.set_env(task_config)

                system_prompts.append(system_prompt)
                init_user_prompts.append(init_user_prompt)
                envs.append(env)
            
            return system_prompts, init_user_prompts, envs
        
        def _build_data_proto(
            system_prompts: list[str], init_user_prompts: list[str], envs: list
        ) -> InteractionDataProto:
            # Dynamic interaction 的初始输入是 messages，而不是已经 tokenize 的 tensor。
            # 后续 tokenization/截断由 InteractionManager 根据 InteractionConfig 处理。
            messages = []
            for system_prmopt, init_user_prompt in zip(system_prompts, init_user_prompts):
                system_message = {"role": "system", "content": system_prmopt}
                user_message = {"role": "user", "content": init_user_prompt}
                init_messages = [system_message, user_message]
                messages.append(init_messages)

            data_proto = InteractionDataProto()
            data_proto.no_tensor_batch["init_prompts"] = messages
            data_proto.no_tensor_batch["envs"] = envs

            return data_proto
        
        # ===== body =====
        accelerator = Accelerator()

        if accelerator.is_main_process:
            writer = create_tensorboard(save_dir=self.working_dir) 
            save_file = os.path.join(self.interaction_config.output_dir, "conversations.txt")
            recorder = DynamicEvalRecorder(writer=writer, log_file=save_file)
        else:
            writer = None
            recorder = None

        batch_size = self.interaction_config.batch_size
        
        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.evaluation_dataset,
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()
        
        # batch generate.
        # Dynamic 任务的 reward 通常来自 env.feedback()，而不是一次性答案匹配。
        for step, test_batch in tqdm(enumerate(test_dataloader), desc="Evaluation"):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                system_prompts, init_user_prompts, envs = _set_batch_envs(test_batch) 
                input_data_proto = _build_data_proto(system_prompts, init_user_prompts, envs)
                
                self.generation_manager.actor_rollout_wg = unwrapped_model
                outputs: InteractionDataProto = self.generation_manager.run_agent_loop(input_data_proto)
                
                inter_histories = outputs.no_tensor_batch["inter_histories"]
                inter_context = self.processing_class.apply_chat_template(inter_histories, tokenize=False)

            # calculate batch rewards
            rewards = []
            for env in input_data_proto.no_tensor_batch["envs"]:
                reward = env.feedback()
                rewards.append(reward)

            all_contexts = gather_objects(inter_context)
            all_rewards = gather_objects(rewards)

            if accelerator.is_main_process:
                for conts, rs in zip(all_contexts, all_rewards):
                    recorder.record_batch(conts, rs)
        
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            recorder.finalize()
            writer.close()
    
    def _parse_configs(self, configs):
        """把 run YAML 转成 Runner 内部状态。

        这一层会同时构造三套 TrainingArguments：
        - weaver_sft_training_args
        - weaver_grpo_training_args
        - trigger_grpo_training_args

        即使某一套本轮不用，也会先建出来，方便后续代码根据训练模式统一取字段。
        """
        
        self.run_mode = configs.get("mode", "train")
        if self.run_mode not in {"train", "evaluate"}:
            raise ValueError("run.mode must be 'train' or 'evaluate'.")

        self.train_weaver = configs.get("train_weaver", True)
        self.train_trigger = configs.get("train_trigger", False)
        if self.run_mode == "train" and self.train_weaver == self.train_trigger:
            raise ValueError(
                "Training requires exactly one of run.train_weaver and run.train_trigger."
            )
        
        # --- Parse weaver training args ---
        self.train_weaver_method = configs.get("train_weaver_method", "sft")
        if self.train_weaver_method not in ["sft", "grpo"]:
            raise ValueError("Unsupported weaver training method.")
        
        # parse weaver sft training args
        weaver_config = configs.get("weaver", dict())
        weaver_sft_config = weaver_config.get("sft", dict())
        self.weaver_sft_training_args = SFTConfig(**weaver_sft_config)

        # parse weaver grpo training args
        weaver_grpo_config = weaver_config.get("grpo", dict())
        self.weaver_grpo_training_args = GRPOConfig(**weaver_grpo_config)

        # --- Parse trigger training args ---
        trigger_config = configs.get("trigger", dict()) 
        self.train_trigger_method = configs.get("train_trigger_method", "grpo")
        if self.train_trigger_method not in ["grpo"]:
            raise ValueError("Unsupported trigger training method.")
        
        trigger_grpo_config = trigger_config.get("grpo", dict())
        self.trigger_grpo_training_args = GRPOConfig(**trigger_grpo_config)

        # --- update training args ---
        # output/log 目录由 main.py 生成的 working_dir 统一管理，覆盖 YAML 里的默认路径。
        updated_args = {
            "output_dir": os.path.join(self.working_dir, "model"),
            "logging_dir": os.path.join(self.working_dir, "run"),
            # A stage produces one explicit final checkpoint at model/. Do
            # not claim to reload a nonexistent "best" intermediate model.
            "save_strategy": "no",
            "load_best_model_at_end": False,
        }
        for k, v in updated_args.items():
            setattr(self.weaver_sft_training_args, k, v)
            setattr(self.weaver_grpo_training_args, k, v)
            setattr(self.trigger_grpo_training_args, k, v)

        # --- parse interaction args ---
        # InteractionConfig 控制 rollout 时的截断长度、采样温度、batch_size 和输出目录。
        interaction_configs = configs.get("interaction", {})
        self.interaction_config = InteractionConfig(
            max_turns=interaction_configs.get("max_turns", 30),
            max_start_length=interaction_configs.get("max_start_length", 1024),
            max_prompt_length=interaction_configs.get("max_prompt_length", 4096),
            max_response_length=interaction_configs.get("max_response_length", 512),
            max_obs_length=interaction_configs.get("max_obs_length", 512),
            temperature=interaction_configs.get("temperature", 0.0),
            batch_size=interaction_configs.get("batch_size", 32),
            output_dir=os.path.join(self.working_dir, "evaluate"),
            weaver_do_sample=interaction_configs.get("weaver_do_sample", False),
            trigger_do_sample=interaction_configs.get("trigger_do_sample", False),
            sink_aware_generation=interaction_configs.get(
                "sink_aware_generation",
                False,
            ),
            record_entropy_gate_trace=interaction_configs.get(
                "record_entropy_gate_trace",
                False,
            ),
            evaluation_split=interaction_configs.get("evaluation_split", "test"),
        )
        if (
            self.interaction_config.sink_aware_generation
            and self.interaction_config.batch_size != 1
        ):
            raise ValueError(
                "Sink-aware generation requires run.interaction.batch_size=1"
            )
