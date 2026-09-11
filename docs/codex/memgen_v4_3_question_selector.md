# V4.3 固定 native_prefix_kv 的题目级 selector

研究范围固定为 11 个 primary Bank，沿用已完成等价实验中的原生前缀 KV。
selector 在读题前选择最多一张记忆或 no_memory；本轮不接 entropy gate、不改注入层数、
位置、包装、消费窗口或卡片内容。轨迹等价未通过不阻止本轮，主要指标为准确率与修复/伤害。

## 完整运行

沿用前轮路径环境变量，执行：

```bash
bash test.sh selector
```

默认从恢复版 `recovery/split_manifest.json` 的 1000 道 calibration-val 中，按固定哈希
排序选取 400 道：训练 200、调参 100、评估 100。三组在生成前确定，不按答案或效果筛选。
所有构造 packet 的样本都必须在同一 manifest 的 bank-source 中，且题目哈希匹配。
官方 test 不加载、不生成；dev-test 不使用。此前 calibration-val 可能用于其他组件，
本轮评估只声称与当前卡片构造、selector 训练/调参分离，不称为从未使用的最终测试集。

程序一次完成：
1. 认证冻结卡片、原生缓存、来源划分和实验配置；检查所有题目/答案内容哈希、全部
   题目×记忆的 1024 completion 上下文预算，不静默截断或跳过长题。
2. 对训练、调参题运行 no_memory + 11 个 Bank，建立完整二元 reward 表。
3. 只使用训练集拟合收益模型，调参集选择正则化和弃用阈值；保存不可覆盖的 selector。
4. 每道评估题先保存只看题目特征的选择，再运行全部 12 分支，用于评价实际选择与上限。
5. 输出最终报告。逐题逐分支断点保存；相同命令重跑会认证并复用完成结果。

400 题对应 4800 条独立生成分支，每分支最多 1024 completion tokens，不包含编码/prefill
计算。实际耗时取决于服务器和平均生成长度；程序不发起 DeepSeek 或其他付费模型 API。

仅检查计划或只读验证完成结果：

```bash
bash test.sh selector --plan-only
MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh selector
```

完整 1000 题可使用独立目录运行：

```bash
MEMGEN_V43_SELECTOR_ROOT=/path/to/selector_1000 \
  bash test.sh selector --train-size 600 --tune-size 200 --eval-size 200
```

同目录不能更改规模、划分、代码、库版本、模型或卡片后强制续跑。

## 题目特征与收益模型

冻结的原 reasoner 编码原始题目文本，取最终归一化隐藏状态的 token 平均并做 L2 归一化。
不读取标准答案、生成轨迹、source Bank ID 或这道题的实验结果。新增小型多输出 ridge
回归器，每个 Bank 一个输出，学习：

`该 Bank 的 strict_reward − no_memory 的 strict_reward`，目标值为 -1、0、+1。

使用所有 Bank 的收益，不制造唯一的“正确 Bank”分类标签。ridge 只在训练题拟合，
固定网格在调参题上选择正则化及收益阈值。最大预测收益超过阈值才注入；否则 no_memory。
包含始终弃用的候选，允许实验得出“不使用记忆在调参集上更好”。不在评估后重新拟合，
不把模型分数声称为校准后的成功概率。这是轻量的题目级 selector 首版，不预设其有效。

Bank 的新选择依据存储在 `selector.json`，按 bank_id 对应回归权重；不会改写冻结 Bank。
value 仍使用原 `prefix_kv/*.safetensors`。运行中不激活 side-KV controller，也不调用 gate。

## 评估对照

- no_memory：无记忆。
- selector：训练得到的收益预测选择器，包括弃用。
- semantic：题目与卡片在同一冻结编码器中的余弦相似度，弃用阈值仅在调参集选择。
- fixed_from_train：在训练集上选出的最佳固定动作（允许 no_memory），评估时不再改变。
- oracle_best：每道题事后挑最好结果的诊断上限，包括 no_memory；不是可部署准确率。

报告准确率、相对无记忆的 gain/harm、净收益、使用频率、Bank 选择分布、配对精确二项
检验，以及捕获多少可用收益。评估全 Bank 表仅用于这些诊断，不反馈给 selector。

## 输出与新题选择

默认输出为 `$MEMGEN_V4_OUTPUT_ROOT/offline/v4_3_question_selector`；没有 V4 root 时
沿用 `/data/memgen-runs/v4`。`MEMGEN_V43_SELECTOR_ROOT` 可覆盖。

- `brief_summary.json`：用于贴回分析的精简汇总。
- `report.json`：包含逐 Bank 准确率的完整汇总。
- `selector.json`：冻结权重、正则化、阈值、训练/调参样本身份和调参记录。
- `profile.json`：冻结输入、版本、数据划分和消费合同。
- `samples/<sample_id>/feature.json`：只从题目得到的特征。
- `samples/<sample_id>/<action>.json`：各动作的独立完整回答和评分。
- `samples/<sample_id>/prediction.json`：评估结果生成前保存的选择。

给出一条新题，只运行 selector（不加载数据集或标准答案）：

```bash
python scripts/select_v4_3_question_bank.py \
  --selector-dir /path/to/v4_3_question_selector \
  --question 'Your new question here' --device cuda
```

其他路径覆盖沿用上一轮，同时新增：
`MEMGEN_V43_SELECTOR_SPLIT_MANIFEST`、`MEMGEN_V43_SELECTOR_TRAIN_SIZE`、
`MEMGEN_V43_SELECTOR_TUNE_SIZE`、`MEMGEN_V43_SELECTOR_EVAL_SIZE`。
`MEMGEN_V43_EQUIVALENCE_ROOT` 必须指向已完成的原生前缀实验目录。只认证并读取其前缀
缓存，不重写该目录，也不需要 equivalence_passed=true。

数据加载接口按 [Datasets 文档](https://github.com/huggingface/datasets/blob/main/docs/source/loading.mdx)
核对；只请求 GSM8K train，并逐条验证选中题目的 question/answer 哈希。即使旧 manifest
记录的 revision 是 main，也不能接受内容漂移。
