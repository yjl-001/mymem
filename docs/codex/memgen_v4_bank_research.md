# MemGen V4 bank 研究台账

本文只记录已经完成的 V4 bank 探索、可复用结论、已关闭路线和当前阻塞点。当前系统合同与运行入口见
[MemGen V4 系统](memgen_v4_system.md)。两份文档分工如下：

- 本文回答“为什么走到当前设计、哪些路线已经试过、实验说明了什么”；
- 系统文档回答“当前仍然有效的组件、接口、工件和约束是什么”。

不得因为代码仍为旧工件兼容而把历史路线重新视为当前推荐方案，也不得把 provisional bank 或失败的
selector qualification 表述为在线有效系统。

## 1. 一直保持不变的研究约束

V4 延续 V3 的经验记忆研究方向，不重新定义总体目标：

```text
memory = (key, value)

key：当前推理状态是否具有某类可修复缺陷
value：选中后应向模型提供什么修复过程
```

当前固定条件：

- benchmark 只有 GSM8K，因此 selector 是一次性对所有 bank 排分的一层结构，没有 task/domain 第一层；
- reasoner 与 tokenizer 使用 V3 已认证的精确 revision；
- 只研究 layer 24 单点、全部 KV groups、SDPA、canonical pre-RoPE、relative phase delta 为零；
- target 与 reference 离线共同构造，线上只允许注入 target；
- auxiliary 只保留 schema 扩展位；
- bank 至少由五个不同 construction sample 支持；
- layer/head calibration 在 bank、selector 和生命周期稳定以后再研究；
- 不修改 V3.5 冻结实现，也不重复 V3 已完成的 final-test 或 full-bank treatment。

## 2. Bank 构造路线与实验结果

### 2.1 V4.0：逐条 signature + bounded map-reduce

第一版将 MI 的 bank 思路直接实现为教师生成 repair signature、map shard 聚类和多轮 reduce。已经完成
5318 条 signature，并完成 111 个 map shard。map 后约有 2651 个 prototype；reduce 运行三轮后仍有
2282 个 prototype，减少幅度只有约 13.9%。

这条路线暴露了四个问题：

1. 自然语言 prototype 仍携带大量实例化表达，同一 repair mechanism 很难在 reduce 中稳定合并；
2. 每轮只能局部比较小批 prototype，跨 shard 的等价关系收敛极慢；
3. 大 JSON 响应频繁出现截断、重复 assignment 和 schema 漂移；
4. 即使 checkpoint 可恢复，API 成本仍随 prototype 数和 reduce 轮数持续增长。

结论：bounded map-reduce 不适合作为当前 bank membership 的主算法。已有 signature 仍是有价值且已付费的
上游语义资产，但旧 map/reduce checkpoint 不再续跑。

### 2.2 V4.1：受控 repair atom + embedding 候选 + 教师判边

V4.1 尝试先把 signature 规范化为受控 repair atom，再用固定 BGE 产生近邻候选，由教师判断 pair 是否具有
相同 failure mechanism、repair action 和 applicability，最后用 complete-link 与 cluster audit 构造 bank。

这一版确认了若干后来保留的设计：

- `source_experience_type` 只是 provenance，不能作为 reasoning cluster 的硬边界；
- answer serialization 必须单独归档，不能重新解释成 reasoning memory；
- candidate merge 必须是 complete-link，不能用连通分量传递合并；
- 每个正式 bank 必须有至少五个不同 sample；
- target/reference 必须是 process-only，并有明确的 applicability 与禁用边界。

但逐 experience canonicalization、逐 pair 判断和逐 cluster audit 仍把教师放在热路径中。运行到约一半时已经
出现大量空 `exclusion_reason`、角色字段不一致和重试；即使 parser 可以确定性修复冗余字段，调用规模和成本
仍不适合当前研究节奏。

结论：保留受控角色、complete-link、五样本支持和审计思想；停止 V4.1 付费热路径。V4.1 的底层 schema 与
loader 仍为历史工件兼容保留，但不再提供推荐运行入口。

### 2.3 V4.2：本地三视图 repair graph

V4.2 复用 5318 条已认证 signature，确定性地将其划为：

| 类别 | 数量 | 处理 |
| --- | ---: | --- |
| answer serialization | 2181 | 归档，不进入 reasoning bank |
| source non-applicable | 59 | 归档，不强制分配 |
| reasoning atom | 3078 | 进入本地三视图图构造 |

每个 reasoning atom 使用 mechanism、repair、applicability 三个文本视图。固定 BGE 只负责产生 mutual-kNN
候选；三个视图必须分别超过 `0.82/0.82/0.70` 才形成正边，再用确定性 greedy complete-link 分组。

本地构造结果：

- mutual-kNN pairs：18631；
- positive edges：17911；
- complete-link groups：1485；
- 至少五个 distinct sample 的 candidates：95，共 615 条 evidence；
- support 不足的 groups：1390，共 2463 条 evidence；
- 全过程外部 API 调用：0。

这一结果说明本地三视图可以高精度地找到一批重复 repair mechanism，但不应追求覆盖全部数据。大量
singleton 和小组是当前粒度下的长尾，不需要为了“把所有数据放进 bank”而强行合并。

### 2.4 高质量 shortlist

95 个 candidate 是 discovery pool，不是 runtime bank。shortlist 使用簇内三视图最弱余量、support tier、
candidate-centroid NMS 和 farthest-first 多样性选择，将其缩小到 25 个候选。人工 semantic policy 又排除
1 个候选，得到 24 个 candidate、167 条 bounded evidence。

有效结论：

- bank 应是小而高质量的 repair basis，不是源数据全集；
- support=5 的边界候选必须同时有较强 cohesion；
- 去冗余必须在三个视图分别成立，不能只看一个 joint average；
- shortlist 仍只是 synthesis 输入，不能直接声称在线有效。

### 2.5 付费 semantic synthesis 预检

针对 24 个候选完成了零 API 的完整 preflight：

- evidence：167 条；
- combined synthesis requests：6；
- nominal review requests：3；
- nominal paid requests：9；
- exact combined request characters：824243；
- estimated input tokens（按三字符估计）：274748；
- 实际 API 调用：0。

用户最终决定先不运行 DeepSeek/GLM semantic synthesis，而用 deterministic local-direct 验证完整系统接口。
因此付费 semantic 路径只是保留的回退方案，不是当前结果的一部分，也不能把 preflight 当成语义审核通过。

### 2.6 Local-direct bank

local-direct 从每个 candidate 的五至八条 evidence 中选择三视图 joint medoid，直接将既有 signature 映射为
target/reference process card。结果为 24 个 bank、167 条 evidence，零 API。

它验证了以下工程合同：

- target/reference schema 可以确定性构造；
- target 与 reference 可以共同编译，而 loader 只公开 target；
- bank、evidence、medoid、profile 与上游 shortlist 可以完整 hash 绑定；
- 无需教师也能构造一个可供下游消融的 provisional bank。

它没有验证逐 evidence 的事实一致性、cluster-level semantic coherence 或独立 card review，因此质量层级
始终是 `provisional_local_direct`。

### 2.7 静态 curated bank

对 24 张 local-direct process card 做 hash-bound 静态检查：

| 决策 | 数量 | 当前处理 |
| --- | ---: | --- |
| primary | 11 | 保留 |
| conditional | 6 | 保留 |
| quarantine | 3 | 排除 |
| hard reject | 4 | 排除 |

当前主实验由此得到 17 个 bank、116 条 distinct-sample evidence。原始 24-bank 工件保留为 ablation；curation
不会原地修改或覆盖它。该步骤仍不是完整 construction-evidence semantic audit，所以输出层级为
`provisional_local_curated`。

### 2.8 MI 风格 Side-KV

17 个 target 和 17 个 reference 均已成功编译为 layer-24 side-KV：

- raw descriptor、internal principle、hidden note 三种固定 descriptor 形式；
- canonical pre-RoPE；
- target online only；
- reference 只供离线 provenance/contrast；
- side-KV compilation passed，17 个 target 均可加载。

因此当前阻塞不在 tensor-free bank 到 Side-KV 的编译通道。

## 3. Selector qualification 暴露的核心问题

当前 dynamic-only selector 对 116 条 construction evidence 的结果为：

```text
116 construction experiences
  ├─ 50 failure_has_no_joint_gate
  └─ 66 gate-eligible anchors
       ├─ 31 anchors 分布在 11 个不足五条的 bank：拒绝
       └─ 35 anchors 分布在 6 个 bank：进入 LOO calibration
            ├─ 2 correct selected
            ├─ 0 wrong selected
            ├─ 1 success false selected
            └─ 33 failure abstained
```

关键比例：

- gate coverage：66/116，56.9%；
- bank survival after five-gate-anchor rule：6/17，35.3%；
- thresholded correct failure coverage：2/35，5.7%；
- success false-selection：1/35，2.9%；
- failure wrong-routing：0/35。

冻结阈值是 `absolute=0.0611734390`、`margin=0.0`。五个 bank 的
`correct_selected_count=0`，但它们并非从未正确 Top-1：其中 `2e4c0a...` 的六条 failure query 全部正确
Top-1，只是所有 score 都低于全局 absolute threshold。五个 zero-selected bank 合计仍有 16/28 的原始正确
Top-1；加上唯一通过阈值的 bank，六分类原始 Top-1 下限为 18/35，即 51.4%。

这说明失败不能归因成“这五个 bank 无效”。当前核心问题分为四层：

1. **Bank–Gate 不对齐**：43.1% 的 construction failure 没有 V3.4 joint gate；
2. **Bank–Selector 表示不对齐**：文本 repair cluster 的区别没有充分反映在纯 local-16 hidden state 中；
3. **跨 bank score 标尺不一致**：100% raw Top-1 的 bank 仍被全局 absolute threshold 全部拒绝；
4. **校准分布错位**：matched success state 即使不会通过上游 gate，也被当作 selector 可达的 false-selection
   query，可能人为推高阈值。

另外，`bank >= 5 construction samples` 与 `bank >= 5 joint-gate anchors` 是两个不同条件。后三个 5-support
bank 各有 4/5 gate anchors，仍被当前硬门槛整张拒绝。这个门槛是 selector 额外引入的实验变量，不是原始
bank 构造合同本身。

## 4. 当前结论与关闭项

已经确认：

- 本地三视图 + complete-link 足以产生小规模高一致性候选，不再恢复 API map-reduce；
- 不需要把全部源数据放入 bank；serialization、non-applicable 与低支持长尾应继续归档；
- 17-bank target/reference Side-KV 编译链有效；
- 当前纯 dynamic local-16 selector 在安全阈值下不可上线；
- 不能通过直接降低阈值、强制输出 tensor 或删除所有 zero-selected bank 绕过失败；
- 当前没有证据要求重新进行 DeepSeek semantic synthesis，因为 selector 失败首先是状态/校准接口问题。

暂时保留但不作为当前入口：

- V4.0 signature 生成代码：已有 5318 条 signature 的可复现来源；
- V4.1 schema/parser：历史工件兼容和 V4.2 上游认证仍有直接依赖；
- V4.2 paid semantic builder：只有在 local-direct bank 的语义质量被独立证明是瓶颈后才考虑；
- online evaluator：等待 selector 真正通过 qualification 后再使用。

已经删除或应删除：

- 旧 map-reduce progress inspector：只服务已经停止的 reduce checkpoint；
- 指向 V4.1 付费目录的旧总 runner：不能消费当前 curated bank，容易误运行过时流程；
- 假装可调、实际上被强制等于冻结常量的 selector CLI 选项。

## 5. 下一步 selector 研究边界

下一步不重新构造 bank，也不重新编译已有 Side-KV。先把 selector 编译改成一次前向后保存 offline-only source
state cache，至少包含 failure local state、matched success state、prompt-end state、gate reachability、bank/sample
identity 和 provenance。后续所有 selector ablation 都从同一份缓存运行，避免反复执行 116-example 前向。

优先比较：

1. 当前 dynamic-only baseline；
2. 只在真实 gate-reachable success states 上计算 false-selection；
3. prompt semantic state；
4. prompt + dynamic 的单层联合 score；
5. global raw score 与 bank-normalized score；
6. 原始五 construction support 下，四个与五个 gate-anchor 的诊断差异。

无论输入包含多少特征，线上仍对所有 bank 一次评分，不恢复 task/domain 第一层。只有在 answer-blind、
leave-one-problem-out 审计中同时得到可接受的 macro routing、错误路由、gate-conditioned false selection 和
per-bank coverage，才生成 online selector artifact。

在继续上述 selector 比较前，当前代码先提供两个尚待服务器 GPU 执行的离线资格组件：

- 一次性 layer-24 source-state cache：保存 prompt semantic states、failure 全部最多三次 gate raw32 window、
  matched-success aligned controls 与 success actual-gate safety states；
- oracle target/reference causal audit：绕过 selector，在同一 exact prefix/native cache 上比较 no-memory、正确
  target 和同 bank reference，并执行当前 bounded nonpersistent lifecycle；分支前强制验证 cache tensor 完全
  一致且底层 storage 独立。

CPU auditor 已为后续 selector 研究预留 raw 与 bank empirical-tail normalization、sample-level LOO、sample-macro
routing、hubness、bank-size bias 和 threshold diagnostics，但这些都只是离线诊断，不能直接生成在线 selector。

这两个组件的 smoke/full runner 是
`scripts/experiments/gsm8k/run_v4_source_oracle_audit.sh`。实现和 schema 的存在不等于 oracle 已通过；在服务器
报告生成前，17 个 bank 仍只具有 Side-KV 编译资格，没有因果有效性资格。

## 6. 活跃代码边界

当前主线：

- `scripts/build_v4_2_local_clusters.py`
- `scripts/select_v4_2_bank_candidates.py`
- `scripts/build_v4_2_local_direct_bank.py`
- `scripts/curate_v4_2_local_direct_bank.py`
- `scripts/compile_v4_side_kv.py`
- `scripts/compile_v4_selector_anchors.py`
- `scripts/extract_v4_source_state_cache.py`
- `scripts/audit_v4_source_state_cache.py`
- `scripts/audit_v4_oracle_causal_utility.py`
- `scripts/experiments/gsm8k/run_v4_2_curated_offline.sh`
- `scripts/experiments/gsm8k/run_v4_source_oracle_audit.sh`
- `memgen/experience/v4_2_bank.py`
- `memgen/experience/v4_2_local_direct.py`
- `memgen/experience/v4_2_curated.py`
- `memgen/model/v4_side_kv.py`
- `memgen/model/v4_selector.py`
- `memgen/model/v4_online.py`
- `memgen/model/v4_runtime.py`
- `memgen/experience/v4_source_state.py`
- `memgen/experience/v4_oracle_audit.py`
- `memgen/model/v4_oracle.py`

兼容/回退代码不得误当成当前推荐入口。删除这类底层文件前必须先解除 active artifact 的 implementation hash、
schema import 和 source-signature authentication 依赖，不能为了减少行数破坏已有工件。
