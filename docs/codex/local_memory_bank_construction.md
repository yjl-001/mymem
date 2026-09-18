# 本地教师 Bank 构造流水线

本入口从 GSM8K 原始题目构造新 Bank，不依赖旧 17-bank/116-evidence、DeepSeek、risk gate 或旧 source-state cache。
Qwen3-32B 在服务器本地通过 Transformers 加载；`device=auto` 可利用可见 GPU 分配权重。
没有外部推理 API。首次运行仍可能从 Hugging Face 下载公开数据、tokenizer 和权重。

## 运行

在已有 CUDA PyTorch 环境中安装构造依赖（保留已有 torch/CUDA 安装）：

```bash
python -m pip install -r requirements-bank-construction.txt
bash scripts/experiments/gsm8k/run_local_memory_bank.sh \
  --config configs/experiments/gsm8k/local_bank.json \
  --output-dir output/experiments/banks/gsm8k-local-qwen32b-r1
```

`all` 是默认模式：split → rollouts → review → evidence → groups → cards → compile → evaluate。
最后一步只在 builder valid 上运行 baseline 和每个 primary Bank 的完整答案，不运行 test。
运行规模是 train 的 8 条轨迹/题，加上 valid 的 `1 + primary Bank 数` 个解题分支/题；教师调用另计。

原命令加 `--resume` 可继续同一配置和代码的运行。每条轨迹、审核、教师请求、响应和 Bank 独立原子落盘。
模型、提示词、配置或软件环境变更需要新输出目录；禁止把不同构造版本的数据静默混合。
`--stage split|rollouts|review|evidence|groups|cards|compile|evaluate` 是运维恢复入口，不需要逐阶段手工运行。

```bash
# 不下载、不加载模型的计划检查
python scripts/build_local_memory_bank.py --output-dir /tmp/bank-plan --plan-only
# 已完成工件的完整性检查，不运行推理
python scripts/build_local_memory_bank.py \
  --output-dir output/experiments/banks/gsm8k-local-qwen32b-r1 --validate-only
```

配置中的模型 `source` 可以是完整的本地 safetensors 模型目录。远程 `main` 首次解析为精确 commit，续跑
复用该 commit；本地模型按配置/tokenizer/权重文件 SHA 绑定。teacher 和 reasoner 分阶段加载，进入编译前
释放 teacher。实际可用显存、CPU offload 和推理速度取决于服务器；不会自动量化或修改模型。

## 模块与数据合同

| 模块 | 职责 |
|---|---|
| `data/gsm8k/splits.py` | builder 与构造共享的 train/valid/test 划分 |
| `bank_construction/config.py`、`sources.py`、`artifacts.py` | 配置、精确版本、原子工件、写锁 |
| `rollouts.py` | 1 greedy + 7 sampled、独立随机种子、停止原因 |
| `review.py`、`teacher.py`、`schemas.py` | 本地过程审核、轨迹分类、结构校验和重试 |
| `experiences.py` | 同题对照与单题经验提取 |
| `grouping.py` | 教师分组、跨批次匹配、归并及成员核验 |
| `cards.py`、`prompts.py` | 可迁移方法和使用边界、逐成员批次复核 |
| `compilation.py`、`memgen/model/local_bank.py` | 检索表示和既有 native prefix KV 编译/消费 |
| `evaluation.py`、`audit.py`、`pipeline.py` | valid 结果表、完整性审核和调度 |

### 数据划分

官方 train 按 `int(len(train) * val_ratio)` 划出 valid，`val_ratio=.1`，显式 `split_seed=42`。
官方 test 保持不变。默认 6726/747/1319。builder 也使用该显式 seed；这不承诺恢复某次旧的隐式 RNG 划分。
保存原始 row index、sample ID、题目/答案 hash 和完整清单；发现跨划分的相同题目会停止报告。
教师只收到 train 数据。valid/test 保存在 split 工件中，但不会发给构造教师。

小规模运维验证可复制配置并设置 `train_limit` 和 `valid_limit`，不改变原始划分；之后全量运行必须使用
新目录。默认两个 limit 均为 0，表示全部。构造 limit 会改变证据集合，不能与旧工件混用。

### 采样和标签

固定每题 1 条 greedy + 7 条随机采样；随机部分 temperature=.8、top_p=.95、top_k=0，上限 1024。
复用 GSM8K prompt 与 MemGen ChatML。停止条件为 EOS、完整 boxed answer、或长度上限；前两个优先于
长度判定，因此恰好在第 1024 token 完成不被错误标记为截断。生成 token 包含实际发出的 EOS，排除输入。
单序列推理使 RNG 不依赖 batch 顺序；seed 来自全局 seed、sample ID 和 rollout index。

成功要求格式及答案验证通过，且教师认为推理正确。失败标签可重叠：answer_error、format_error、reasoning_error。
最终答案 reward 与构造 outcome 独立：碰巧答对但推理错误的轨迹仍为构造失败。
长度截断不进入成功/失败集合，也不调用过程教师；不确定及 verifier/教师答案判定冲突单列 uncertain。
无 box 时不把“最后一个数字”直接认定为最终答案，使用教师对明确最终答案的判断。

每题最多一对对照，沿用原来单题单 evidence 的支持统计口径。分别在 success/failure 中优先选择 greedy，
再按 rollout index 选择，全部 8 条轨迹和审核仍保留。没有合格成功/失败对的题目明确归档。

### 本地教师与归并

教师使用其原生 Qwen3 chat template、`enable_thinking=False`，默认 temperature=.7、top_p=.8、top_k=20，
输出预算 8192。这些是教师参数，和上述 reasoner 的采样参数分开保存。
每次调用存储任务、提示词版本、完整请求、原始响应、种子及解析结果。无效 JSON/schema 或教师截断会进行
有界重试；失败不生成成功检查点。续跑会保留旧失败响应并允许一轮新的有界尝试。不会用 regex 修补语义。

教师执行：过程审核 → 单题结构化经验 → 分批初始分组 → 跨批次候选匹配 → 两组归并提案 →
原始成员分批核验 → 卡片逐批汇总 → 最终卡片回查全部成员批次。
跨批次候选窗口会检查所有现存组；没有 embedding 阈值或主题规则决定归并。模型可以保留单例、拒绝
归并；初始分组允许拆分。大组不一次塞入全部 evidence，而是逐批复核，避免只信任摘要。
批次大小只限制上下文计算，不设置 Bank 的语义大小阈值。prompt 超出上下文时明确报错，不静默截断；
异常长的单条经验可在新配置/目录中调整教师预算或批次大小。

每条经验和卡片都要求替换检验及近似反例，保留必要的数学关系、公式与常数，不使用数字/实体/关键词
黑名单。成员 ID、覆盖、来源等可机械检查的内容仍严格验证。
primary/conditional/reject 来自同一个教师的第二次语义审阅，所有成员批次均 primary 才列为 primary；
这不是独立审阅，也不证明下游有效。没有硬编码“至少 5 条才能构造”，实际独立题目支持数保留在工件中。
不一致组/卡片被明确拒绝，不能靠模糊化描述强行补齐 Bank 数量。

### KV、selector 与 valid

只编译 primary 卡片。KV 编译及 greedy 消费直接复用 V4.3 native-prefix 实现：全部层、全部记忆前缀
和包装 token，原生位置，从题目开始使用至结束；没有 entropy gate、延迟注入、32 步卸载或 side-attention。
检索默认使用卡片 problem_structure，经冻结 reasoner 最后一层均值池化和 L2 归一化；完整卡片仍是 KV value。

valid 先保存所有 question-only top-1 选择，再生成 baseline/逐 Bank 结果表。当前 top-1 是无拒用阈值的
参考策略，不自动宣称为最优 selector；Qwen3-Reranker 不参与本构造流程。报告每个 Bank 以及该参考 selector
的准确率、gain/harm、生成 token 总量/均值/中位数/p90/上限命中数。记忆输入和 KV 存储成本不计入生成 token。
valid 是开发数据，不自动用该结果把卡片宣传为测试集有效，也不自动调阈值或升级经验质量标签。
若没有 primary，仍完整输出 baseline 与 `complete_without_primary_banks`，绝不宣称得到了可用记忆。

构造出的库支持单题在线读取：

```bash
python scripts/use_local_memory_bank.py \
  --bank-dir output/experiments/banks/gsm8k-local-qwen32b-r1 \
  --question 'A tank contains 20 liters. Half is removed. How much remains?'
```

只输入题目，选择至多一个 Bank，复制前缀缓存并生成；不需要教师、标准答案或旧实验目录。
新库使用独立 schema，不伪装成旧的固定 17 个 Bank 的认证 bundle。

## 产物

`profile.json`、`split.json` 保存版本与数据清单；`rollouts/`、`reviews/`、`evidence/`、`teacher/`、`cards/`
保存逐条工件；`prefix_kv/` 保存 safetensors 和 manifest；`retrieval/` 保存 key 与向量；`valid_choices/`、
`valid_results/` 保存选择与答案；`stages/` 是完成索引；`brief_summary.json` 是最终简报。
不要编辑 sealed 工件来绕过检查，修改构造策略应创建新的 run。

## 验证

```bash
python -m unittest tests.test_local_bank_construction tests.test_local_bank_runtime
```

纯逻辑测试使用可控教师响应；runtime 测试使用随机初始化的 tiny Qwen，实际执行 attention、原生 KV 保存/
复用及 valid 解码，不下载模型。这些测试验证实现合同，不代替服务器 Qwen3-32B 语义质量与全量实验。

接口参考：[Qwen3-32B 官方模型说明](https://huggingface.co/Qwen/Qwen3-32B)，
[Transformers 4.57.3 生成文档](https://github.com/huggingface/transformers/blob/v4.57.3/docs/source/en/llm_tutorial.md)。
