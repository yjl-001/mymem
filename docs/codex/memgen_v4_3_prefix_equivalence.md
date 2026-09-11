# V4.3 primary 文本／原生前缀 KV 等价实验

本实验使用已生成的 11 张 primary 卡片及 76 个来源样本。它验证把完整记忆前缀预先
计算成全层原生 KV，是否保留可见文本分支的计算和输出；不是 held-out 泛化实验。
不重新调用 DeepSeek，不调整原有 side-KV 的编译、位置、注意力偏置或关闭策略。

## 一条命令运行

沿用上一轮服务器的 `MEMGEN_V4_OUTPUT_ROOT`、Bank、side-KV、source-cache、risk 路径配置：

```bash
bash test.sh equivalence
```

没有额外 smoke/full 手动切换。先认证全量来源和 primary 工件，检查全部 76 个输入的
分词边界与上下文预算，编译或复用 11 份原生前缀缓存，然后完成所有成对实验。
同一命令可断点续跑，已完成 case 不再生成。缺少输入不会触发上游重建。

默认新输出：`$MEMGEN_V4_OUTPUT_ROOT/offline/v4_3_prefix_equivalence`。
若未设置 V4 root，则沿用仓库默认 `/data/memgen-runs/v4`。
可用 `MEMGEN_V43_EQUIVALENCE_ROOT` 指定独立目录，不复用旧 audit 输出目录。

```bash
bash test.sh equivalence --plan-only
MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh equivalence
```

## 四个分支

| 分支 | 计算方式 |
|---|---|
| baseline | 原生无记忆 prompt |
| visible_text | 与旧 visible 分支相同的 system 卡片及完整题目，从头 prefill |
| native_prefix_kv | 载入 system 卡片的全层原生缓存，再处理完整题目后续 token |
| frozen_side_kv | 复用旧 primary 第 24 层 side-KV，从原生 prompt-end 激活 |

每个分支都使用相同的 greedy 策略、最多 1024 个 completion token，以及原有 boxed/EOS
停止条件。后两个记忆分支含义不同：native_prefix_kv 的记忆永久保留在原生上下文中；
frozen_side_kv 继续使用 delta=0、原分数偏置、最多 32 步及低熵/答案标记关闭条件。
只有 frozen_side_kv 使用 attention controller。原生前缀分支的 active_step_count=0
表示没有激活这个 side controller，不表示没有记忆。

原生缓存包括 system 角色标记、指导语、卡片及结束标记，保存每层 post-RoPE K 和 V。
存储按 Bank 分开并绑定卡片、模型/tokenizer revision、运行库版本、实验配置与代码哈希。
消费前复制缓存；题目和生成不会污染 Bank 中的共享前缀。分词必须满足：
`memory_prefix_ids + suffix_ids == visible_prompt_ids`，否则报错，不做静默近似。

## 等价检查

1. 先比较两条路径 prefill 后、最后一个 live query token 之前的每层缓存。
2. 文本分支和原生前缀 KV 分支各自自由生成。
3. 以文本分支的完整自由续写为固定序列，两条路径逐 token teacher-force，比较每一步
   raw logits、KL、top-1、文本 top-1 margin，最后再比较完整缓存。
4. 分别记录自由生成 token、停止原因和最终奖励是否一致。gold 仅在四分支完成后评分。

默认 `atol=0.05`、`rtol=0.02` 用于 bfloat16 元素级 allclose，并同时报告原始最大误差
和相对 L2。阈值在运行前绑定，可显式通过 CLI 修改，但会形成不同实验身份，不能覆盖
已有目录；程序不根据观察结果自动放宽阈值。数值通过并不要求 bitwise 完全一致。
最终 equivalence_passed 要求全部 case 数值通过、固定序列每一步 top-1 相同，并且
自由生成 token 和停止原因相同。只有准确率相同不能判为等价。

发现不等价仍完成余下 case 并保留诊断，最终退出码为 2。正常通过为 0；运行错误非零。
这避免第一处浮点偏差使整个机制实验中断，也避免将完成但不等价标记为成功。

## 结果

- `brief_summary.json`：最小反馈文件，包含四分支正确数、修复/伤害、等价判定和误差计数。
- `core_summary.json`：另含逐 Bank 统计和逐 case 内容哈希。
- `cases/*.json`：完整回答、每一步比较、每层缓存误差，以及冻结 side-KV attention trace。
- `prefix_kv/*.{json,safetensors}`：11 份可复用的全层原生记忆缓存。
- `profile.json`：来源、配置、代码版本和样本计划。

本地验证使用随机初始化小型 24 层 Qwen，覆盖 float32/bfloat16、缓存磁盘复用、多个
问题独立消费、错误 KV 检出、完整四分支运行、结果认证和只读断点验证。
真实 Qwen-1.5B/CUDA 数值误差与准确率需服务器运行，不能以本地测试替代。
全层前缀缓存仍占用原生上下文长度和全层 KV 空间，尚不具备单层 side-KV 的存储优势。

缓存接口依据 [Transformers cache 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/cache_explanation.md)
核对，并在本地 Transformers 4.55.4 上运行验证；未升级项目依赖。
