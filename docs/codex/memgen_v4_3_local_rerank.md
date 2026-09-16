# V4.3 本地 Qwen3 重排实验

默认使用 **Qwen/Qwen3-Reranker-8B**，通过本地 Transformers 前向计算为题目与完整记忆卡
的适用性打分。它是专用重排模型，区别于通用对话模型 Qwen/Qwen3-8B；这个入口实现的是
专用 reranker 的 yes/no 打分协议，不是通用模型生成 JSON 的 listwise judge。

官方参考：[模型卡与 Transformers 示例](https://huggingface.co/Qwen/Qwen3-Reranker-8B)。
代码按官方 prefix/body/suffix 协议及 yes/no logits 计算方式实现，使用自定义数学经验
适用性指令。仓库中原 reasoner 使用的 ChatML 模板不应用到 reranker。

## 运行

沿用旧实验路径变量，在已安装原实验依赖的服务器环境执行：

```bash
bash test.sh local-rerank
```

默认 Hub 模型固定在已核对的提交：
`5fa94080caafeaa45a15d11f969d7978e087a3db`。
首次需要可下载该版本的权重和 tokenizer；随后使用 Hugging Face 缓存。
这是下载公开模型文件，不是调用外部推理 API。DeepSeek 等 API key 在启动时移除。

已有完整的本地 Transformers 模型目录时：

```bash
MEMGEN_V43_RERANK_MODEL=/path/to/Qwen3-Reranker-8B \
  bash test.sh local-rerank
```

本地目录应包含完整 config、tokenizer 和 safetensors 权重。程序计算这些文件的内容哈希，
防止替换权重后混用旧结果；本地模型身份检查需要顺序读取权重文件。Hub 模式只接受精确
commit，不接受浮动 main。不同模型、版本、参数或路径实验请指定新的输出目录。

默认 BF16、SDPA、单题单卡片前向、全部模型在一张 GPU 上，不使用量化或多卡分片。
8B BF16 权重本身约 16GB，实际显存还包括激活和运行开销。这轮不同时加载原 reasoner。
本地测试环境为 Transformers 4.57.6；Qwen 官方说明架构支持要求 Transformers >=4.51.0。
原实验环境已有 4.57.6 时无需为此升级；保持来源实验环境不变。

可用参数及变量：

- `MEMGEN_V43_SELECTOR_ROOT`：已有全动作答案表。
- `MEMGEN_V43_SIMILARITY_ROOT`：已有 key 向量及相似度研究结果。
- `MEMGEN_V43_RERANK_ROOT`：新输出，默认 `$MEMGEN_V4_OUTPUT_ROOT/offline/v4_3_local_rerank`。
- `MEMGEN_V43_RERANK_MODEL`：默认专用 8B Hub 名称，也支持本地目录。
- `MEMGEN_V43_RERANK_REVISION`：Hub exact commit；本地目录改为文件内容哈希绑定。
- `MEMGEN_V43_CUDA_VISIBLE_DEVICES`、`MEMGEN_V43_DEVICE`：GPU 选择，与旧脚本一致。
- `--max-length`：默认 8192。超长输入直接报错，保留已有结果，不静默截断题目或卡片。

```bash
bash test.sh local-rerank --plan-only
MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh local-rerank
```

默认自动复用已认证的逐题逐 Bank 分数；中断后重跑原命令即可。
完整结果验证时不加载模型、不加载数据集，也不覆盖文件。
仅计划模式只认证来源、模型身份并打印规模，不下载/加载权重，不创建输出。

## 打分依据

输入仅包括数学题原文、已有完整记忆卡和固定适用性指令。关注：目标量、数学关系、
约束、决策点、操作的适用条件，以及是否会引入无依据假设或无关操作。
不把词汇主题相似等同于操作适用，也不要求记忆包含数值答案。

每个题目—卡片对独立打分，模型看不到 bank_id、原相似度、历史正确率、标准答案或
reasoner 的推理轨迹。读取冻结模型最后位置的 yes/no logits：

`score = sigmoid(logit_yes - logit_no)`。

不生成思维链、解释或 JSON，不调用 model.generate；因此 reranker 生成 token 数为 0，
但输入 prefill/前向计算并非免费，需要单独统计成本。
这个分数是模型在给定指令下的 yes/no 偏好，**不是校准后的 reasoner 答对概率或收益概率**。

## 对照与弃用

保持 11 个 primary Bank、完整卡片 value 和 native prefix KV 消费方式冻结。
题目特征及问题结构 key 直接复用现有结果，建立三个候选集合：

1. `semantic_top3`：问题结构余弦相似度前三名，不预先套用旧弃用阈值。
2. `random_top3`：固定 seed=43，根据题目哈希和 bank_id 取三个不同候选。
3. `all_banks`：全部 11 个 Bank。

随机对照是一组固定候选，不是上一轮覆盖率的精确随机期望，也不是多随机种子均值。
它便于复现，但单次随机差异需要谨慎解释。

每个集合报告两种选择：

- `/forced`：总是选 reranker 分数最高的 Bank。
- `/calibrated`：最高分严格超过该集合训练出来的阈值才使用，否则 no_memory。

每个集合的阈值只在 train 上，从固定集合 `None, -1, .1, .25, .5, .75, .9, .95, .99`
中选择。None 表示总是弃用，-1 表示全部放行。依次按正确题数更多、harm 更少、reasoner
生成 token 更少、记忆使用更少、候选顺序决定。阈值在 train 保存后才处理 tune 分数；
tune 的最终选择保存后才读取 tune utility 表评分。不会根据 tune 自动替换模型或指令。

另保留无记忆、旧训练集固定 Bank、旧相似度策略和问题结构始终 top-1 对照。
旧相似度阈值历史上使用过当前 tune；这些数据已是开发数据，不是独立最终测试。

## 执行量与成本

默认 train=200、tune=100，读取同样的 300 道题，**总共 3300 次独立题目—Bank 打分**。
为公平比较三个候选策略，本轮一次性打分所有 Bank，再从中取对应候选的分数。
所以实际实验工作量是每题 11 次；未来仅部署 top-3 时可只做每题 3 次。

全部 reasoner 回答、评分与生成长度直接复用已保存的全动作表，新增 reasoner 生成次数为 0。
原始问题从相同 GSM8K train revision 读取并核对 question_sha256，随后单独缓存。
输入到打分函数的只有 question/descriptor 两个字符串，不读取 answer 字段。

完整报告分别记录：

- 各策略准确率、gain/harm、使用次数、配对检验和 Bank 分布。
- 原 reasoner 的 total/mean/median/p90 生成 token、上限次数等既有统计。
- 对应候选的 reranker 输入 token、pair 数和逐 pair 前向计时之和。

耗时统计不包含模型加载、编码/检索、文件 I/O 或 reasoner 生成，因此不称为端到端延迟。
三个集合和 forced/calibrated 会复用分数，不能把各方法成本简单相加当成实验总成本。

## 产物

- `brief_summary.json`：贴回这一份即可开始分析。
- `report.json`：完整 train/tune 指标和逐题选择。
- `profile.json`：固定输入、模型身份、版本、提示词和候选集合。
- `questions.json`：哈希验证过的原始题目，不含标准答案。
- `samples/<sample_id>/<bank_id>.json`：打分、输入 token、前向耗时与身份绑定。
- `selector.json`：仅根据 train 确定的三个弃用阈值。
- `tune_predictions.json`：评分前冻结的 tune 选择。

本地验证采用人工 utility 表和真实 tiny Qwen3 的 logits 前向，包括输入协议、分数计算、
候选/阈值、断点复用、冻结顺序、只读验证和原始文件不被修改。完整 8B GPU 打分和真实
收益需在保存原始 artifacts 的服务器上运行；本地测试不代表已取得 8B 实验效果。
