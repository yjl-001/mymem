# V4.3 相似度 selector 与 native prefix KV 研究

本轮固定 11 个 primary Bank 的完整记忆卡、离线 prefix KV、包装、全部层、原生位置、
生成策略及 1024 completion token 预算。只研究检索 key 和题目级弃用规则。

完整 final-test 已观察到：无记忆 644/1319，原生前缀 660/1319，净增 16 题，
gain=255、harm=239，配对 p=0.4998；平均生成长度从 280.81 增至 336.34。
这支持继续诊断匹配和误用，尚不能声称原生前缀带来稳定提升。
用户已决定暂停 entropy gate 与延迟 KV 消费方向。

## 一次运行

沿用之前的路径变量，在服务器仓库中执行：

```bash
bash test.sh similarity-study
```

源目录 `MEMGEN_V43_SELECTOR_ROOT` 默认是
`$MEMGEN_V4_OUTPUT_ROOT/offline/v4_3_question_selector`。
输出 `MEMGEN_V43_SIMILARITY_ROOT` 默认是
`$MEMGEN_V4_OUTPUT_ROOT/offline/v4_3_similarity_study`。
`MEMGEN_V4_OUTPUT_ROOT` 未设置时沿用 `$MEMGEN_OUTPUT_ROOT/v4`，
后者未设置时为 `/data/memgen-runs/v4`。
Bank、equivalence、source cache、split manifest 等路径变量与旧实验相同。
源目录必须是已有逐题 × 全 Bank 结果的 selector 实验目录，不能是 final-test 目录。

程序自动继承源实验的划分（默认 train=200、tune=100），只读这两组的全部 12 动作结果。
不会读取旧 selector 的 eval 结果文件，也不会加载 GSM8K 官方 test、重新生成答案、
编译 KV 或调用 DeepSeek。来源认证仍复用现有 artifact/profile 检查。

完整卡片向量和题目向量直接复用。新增两种 key 每个 Bank 各编码一次，默认共 **22 次
短文本编码**，使用相同冻结 reasoner、dtype、SDPA 和均值/L2 表示。需要可加载原模型的
相同运行环境；`--device` 必须与原实验一致。首次运行后无需重新加载模型。

默认可断点续跑；已有 source 和 study artifacts 不覆盖。改动策略、模型、版本或输入后
需使用新的 study 输出目录，不在旧目录混合结果。

```bash
bash test.sh similarity-study --plan-only
MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh similarity-study
```

`--plan-only` 只认证来源并打印计划，不创建输出、不编码。
`--validate-only` 重算并核对报告和策略，不加载模型、不写入文件。

## 三种检索 key

| 名称 | 用于相似度的文本 | 消费的 value |
|---|---|---|
| `full_card` | 旧完整 descriptor：适用、操作、避免、验证和边界 | 原完整卡片 prefix KV |
| `applicability` | `unified_process_card.applies_when`：问题结构 + 决策点 | 同上 |
| `problem_structure` | `clause_support.problem_structure.text` | 同上 |

只提取已有抽象字段，不生成新卡，不加入 evidence 原题或答案。当前 primary 的
`only_use_when` 是通用边界句，`applies_when` 已包含结构与决策点，因此不把二者当成
两种不同 key 重复比较。新的 key 不改 bank_id，也不修改原 descriptor 或 KV 张量。

问题和 key 均来自冻结 reasoner 的末层 token 平均表示；它不是专门训练的检索编码器。
这轮检验该表示是否足够有用，不把余弦相似度解释为成功概率。

## 参数如何确定

每种 key 比较两类策略，共六个候选家族：

1. `threshold`：top-1 相似度严格大于阈值才使用其 Bank。
2. `threshold_margin`：满足上面条件，且 top-1 与 top-2 的相似度差不小于 margin。

分数相同按 bank_id 升序选择。未通过则使用 `no_memory`。
阈值候选为无条件放行，以及拟合集 top-1 分数的 25%、50%、75%、90% 分位点；
margin 候选为 0，以及拟合集分差的同样四个分位点。另有“全部弃用”的候选。
这些分位点都是**当前拟合分区**计算的，不从 tune 或 final-test 推导。

在 train 内按 sample_id 的固定哈希分五折。对每个 key/规则家族：

1. 在四折上确定阈值/margin，在剩余一折上记录选择；重复得到所有训练题的 OOF 结果。
2. 比较六个家族的 OOF 表现，确定推荐家族。
3. 各家族在完整 train 上重新拟合阈值；保存推荐家族和全部候选的冻结 `selector.json`。
4. **保存策略后才读取 tune 的奖励和生成结果**，报告各候选表现，不再修改推荐策略。

各处排名均依次优先：正确题数更多、harm 更少、生成 token 总数更少、使用记忆更少、
候选顺序。生成成本只在准确率和 harm 相同时用于决胜，没有人为设置“准确率换 token”
的权重。允许推荐策略最终全部弃用，不强迫产生正收益。

OOF 用于选家族，因此被选家族的 OOF 分数不能作为无偏最终成绩。tune 本身也是历史
calibration pool 的一部分，此前已经用于旧 selector 调参；本轮只能称开发验证。
最终策略不会因为看到 tune 排名而自动切换；下一次研究迭代也不能把 tune 称为新测试集。

## 比较和诊断

精简汇总包含无记忆、旧训练集最佳固定动作、旧相似度策略及六个新候选。
旧固定动作直接继承原 `fixed_bank_from_train`；旧相似度阈值直接继承原 artifact，
它曾在当前 tune 上选过参数，是历史参考。完整报告还包含各 key 始终选 top-1 的对照。

所有准确率和 token 数来自已保存的同一题、同一动作生成。选择规则只看题目特征，
评估再查全动作表；不是逐题查看结果后选最优 Bank。
报告正确数、accuracy、gain/harm、净收益、记忆使用数、选择分布及配对检验；生成统计
包括 total、mean、median、p90、min/max、相对基线变化、达到 1024 上限的题数和其中
正确数。计数包含已发出的 EOS，不包含题目和离线记忆 token；不等同于整体延迟或 FLOPs。
`brief_summary.json` 只保留常用指标，完整统计在 `report.json`。

诊断按 train 分位点给 train/tune 分别分桶，观察：

- top-1 相似度与 gain/harm、token 增量的关系。
- top-1/top-2 分差与同样指标的关系。
- 全部题目 × Bank 相似度与效果的关系。
- 各 Bank 在全部题目上固定使用的效果，以及仅在其为 top-1 时的效果。

全题目 × Bank 的配对并非独立样本，这些桶只作描述性诊断，不据此制造显著性结论。
同样不假定提高阈值或分差必然改善质量，可能只减少有效覆盖。

## 产物与贴回结果

- `brief_summary.json`：只需贴回这一份即可开始分析。
- `report.json`：完整统计、分桶、逐 Bank 诊断和逐题选择。
- `selector.json`：冻结的六套策略、推荐家族、key 向量、训练数据哈希、折划分及 OOF 结果。
- `profile.json`：冻结来源、key 原文、消费合同、版本和实现哈希。
- `key_features/<key>/<bank_id>.json`：可断点复用的新增 key 向量。

推理入口为 `memgen.experience.v4_3_similarity_study.predict(study, question_feature)`，
只接受冻结 artifact 和题目向量，返回 bank_id/弃用及 top-1 分数和分差。选中的 bank_id
仍对应原 `prefix_kv` 的 value。研究脚本不会自动替换生产/旧评估 selector；是否采用
新规则要结合服务器实际输出判断。

本地测试使用人工 utility 表验证选择/弃用和 token 统计，并以真实 tiny Qwen 编码器
验证脚本、缓存、断点续跑、只读校验和先冻结再读 tune 的顺序。真实 11-Bank 结果需在
保存原实验 artifacts 的服务器运行上述命令取得。
