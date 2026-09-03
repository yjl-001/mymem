# MemGen V4.2：低成本 MI 风格 repair bank 与单层在线 selector

V4 延续 V3 已确认的研究方向，不修改 V3.5 的冻结实现。它针对 V3.8 暴露的首要瓶颈——大多数失败 query
在旧 bank、旧 value 和 persistent-to-EOS 注入组合下没有 helpful memory——重新构造 bank/value，并把
selector 收敛成固定 benchmark 内的一层 state-to-bank routing。

V4 当前只研究 GSM8K，固定：

- reasoner、tokenizer 与 V3 已认证工件使用相同精确 revision；
- layer 24 单点、全部 KV groups、SDPA、canonical pre-RoPE、relative phase delta 为零；
- 一层 selector，不再引入 task/domain selector；
- target 与 reference 离线共同构造、共同编译，线上只允许 target；
- auxiliary 只保留 schema 扩展位，不构造内容；
- layer/head calibration 留到 bank、selector 和生命周期稳定之后。

## 1. 两个阶段

```text
阶段一：离线构造 bank

已认证 V4 repair signatures（不重新生成）
  └─ 本地确定性角色边界
       ├─ source non-applicable：归档，不强制分配
       ├─ verified format_compliance：serialization 归档
       └─ 其余 applicable signatures：直接形成 local repair atoms
            └─ 固定 revision 的 BGE，一次编码三个独立视角
                 ├─ failure mechanism + decision point
                 ├─ repair + verification operator
                 └─ problem structure + decision point
                      └─ joint mutual-kNN 召回
                           └─ 三个视角分别通过绝对阈值才产生正边
                                └─ 本地 complete-link：簇内每一对都必须有正边
                                     └─ 至少五个不同 sample 才成为 synthesis candidate
                                          └─ 停止并输出零 API preflight

显式批准后的低频步骤（不属于本地聚类命令）
  └─ DeepSeek 按 candidate 批量做 coherence audit + target/reference synthesis
       └─ 独立 card review
            └─ target/reference 编译 layer-24 side-KV

同一批 construction pairs
  └─ failure trajectory 第一次 entropy+risk joint gate
       └─ layer-24 latest-up-to-sixteen reasoning-state mean + L2
            ├─ 本 cluster failure states：positive anchors
            ├─ matched-success 同进度 states：negative anchors
            └─ 其他 cluster failure states：negative anchors
                 └─ leave-one-problem-out calibration 冻结 absolute + margin thresholds

阶段二：在线 selector + gate + 注入

每个 pre-answer generated token
  └─ 无 active memory 且 entropy/risk joint gate 为正
       └─ 当前 layer-24 local-sixteen state 对全部 qualified banks 一次评分
            ├─ absolute 与 top-one/top-two margin 都通过：注入唯一 target bank
            └─ 否则 abstain；消耗 attempt，但不是整题 terminal
```

阶段划分只有“离线 bank”和“在线使用”两段。side-KV 编译、anchor 构造和 answer-blind 阈值冻结都是离线
bank 工件的组成部分，不单独定义新的研究阶段。

## 2. Construction evidence 与聚类

### 2.1 当前 V4.2 本地聚类

V4.2 将 DeepSeek 从逐 experience canonicalization、逐 pair 判断和逐 candidate audit 的热路径中移除。它复用
已经付费完成并认证的 V4 `repair_signatures.jsonl`；`build_v4_2_local_clusters.py` 不读取
`DEEPSEEK_API_KEY`、不构造 teacher client，也不会自动进入 card synthesis。source profile 中保留的 teacher
字段只用于验证历史 signature 的 provenance，不代表本轮发生 API 调用。

每个 applicable、非 `format_compliance` signature 形成一个 immutable atom，并生成三个独立文本视角：
mechanism、repair、applicability。三个视角由同一个固定 revision 的
`BAAI/bge-small-en-v1.5` 分别编码和 L2 normalize。加权 joint cosine 只决定 mutual top-k 邻居；一条边最终
还必须同时满足 mechanism、repair、applicability 三个绝对阈值。默认参数为 `k=32`、阈值
`0.82/0.82/0.70`、权重 `0.45/0.45/0.10`。与 V4.1 不同，这里的本地相似度是候选 membership 的直接证据；
因此阈值进入 authenticated profile，不能在同一输出目录静默改变。

正边图使用确定性的 greedy complete-link partition。节点只有与组内所有已有成员都有正边时才能加入，所以
A-B 与 B-C 不会在缺少 A-C 时被传递合并。`source_experience_type` 只记录 provenance，既不切断邻居搜索，也
不作为分组边界。每个 group 至少包含五个不同 `sample_id` 才成为 candidate；同题多个 contrast 不重复计算
support。每个 candidate 固定选五个不同 sample 的代表：先选 medoid，再做 farthest-first 覆盖。其余 group
明确归入 unsupported archive，不会被强制合并。

本地聚类结束只产生“可送去合成”的 candidate，不产生可在线加载的 bank。`local_cluster_plan.json` 明确设置
`qualified_for_online_use: false`；`api_preflight_report.json` 记录 candidate 数、未来批量 synthesis 的请求数
上界、代表证据字符数与粗略 token 估计、候选数量 guardrail，以及本轮 `external_api_calls_made: 0`。若 candidate 数
超过 guardrail，后续付费步骤必须保持阻塞，先本地检查阈值和最大簇。

### 2.2 冻结的 V4.1 路径（仅作历史对照）

V4.1 不再续跑旧的 bounded map-reduce，也不重新调用已经完成的 5318 条 signature。输入仍不是 V3 teacher
bank 或旧 memory，而是 Phase-1 原始 verifier-backed contrast pairs、完整的
`repair_signatures.jsonl` 及其 `construction_profile.json`。加载时逐条验证 ID、sample、原始 provenance、
signature hash、teacher/prompt/profile 绑定，并要求 signature checkpoint 精确覆盖全部 construction pairs。
任何漂移都会失败关闭。官方 GSM8K solution 只在最终 card 阶段通过 source index、question hash、answer hash
和 dataset fingerprint 重新 join；前面的规范化、候选图和 cluster audit 不需要重复下载题目内容。

DeepSeek 模型固定为 `deepseek-v4-flash`、temperature 为零、thinking disabled、JSON mode。每次调用记录
model、base URL、prompt version、输入 hash、输出 hash 与 record hash，但绝不落盘 API key。V4.1 新构造分
为五个逻辑步骤：

1. 将旧的 applicable signature 批量规范化为受控 repair atom；
2. 用冻结 BGE revision 产生候选边，再由 DeepSeek 对候选 pair 做语义判定；
3. 本地形成 complete-link candidate，并用五到十个不同 sample 做 coherence audit；
4. 对通过 audit 的 cluster 合成 target/reference process card；
5. 对 card 做只审核、不改写的独立 semantic review。

repair atom 明确分离两个概念：`source_experience_type` 只是 verifier 对最终失败的记录，`memory_role` 才决定
该经验能否成为 reasoning memory。规范化器必须从有限词表选择 state scope、mechanism family、repair family
与 applicability family，并输出 canonical failure transition、repair action、applicability condition 和
verification action。所有 boxing、标签、货币符号、末尾单位、显示精度等只改变最终表达而不改变推理状态的经验
统一标为 `answer_serialization`；它们保留在审计计数中，但不会进入 target bank。无法获得单一、可复用、可验证
transition 的条目标为 `unusable`，也不会被迫塞入某个簇。对 verifier 已确认答案内容正确、只有表达失败的
`format_compliance`，本地 parser 还设置了确定性防线：教师不能把它重新解释成 reasoning memory；这项防线只
决定隔离角色，不把 `experience_type` 用作 reasoning cluster 边界。

reasoning atoms 先按四个受控类别组成的 canonical tuple 做确定性 exact seed；此处允许不同
`source_experience_type` 合并。自然语言 transition/action 保留为 seed 内证据，不作为脆弱的逐字硬边界；若同一
类别组合实际包含不同过程，后续五到十例 coherence audit 必须拒绝，不能因为类别相同而自动进入 bank。
每个 seed 用固定模型 `BAAI/bge-small-en-v1.5` 的固定 commit、CLS pooling 与 L2 normalization 得到 centroid。
embedding 只从全局近邻、同 repair family 近邻和同 mechanism family 近邻召回候选，不参与最终 merge，也不是
在线 selector 的 key。DeepSeek 对每个候选 pair 必须分别确认相同 failure mechanism、相同 repair action、
兼容 applicability 和 process-only；四项全真才产生正边。这样 `experience_type` 不再人为切断同一机制，向量
相似也不能越权成为 bank membership。

正边图在本地用确定性 complete-link 分组：一个 seed 只有与现有组内每个 seed 都有显式正边才能加入。因此
即使 A-B、B-C 都相似，缺少 A-C 正边时也不会通过传递闭包把三者链式合并。每个至少含五个不同 sample 的
candidate 再接受一次 DeepSeek coherence audit；audit 同时检查单一机制、单一修复、适用性、process-only、
serialization-free 和 leakage-free。多 seed candidate 被拒绝后，只允许各 exact seed 独立回退审计，不会再
以较弱阈值强行合并。支持不足、audit 拒绝、serialization、unusable 与旧 signature non-applicable 五类都在
`cluster_plan.json` 中显式归档，所有源 signature 必须恰好归入一个终态。

`exclusion_reason` 和 excluded-role 的类别字段属于冗余控制元数据，不作为 repair 语义证据。不同兼容 API
provider 可能对 reasoning atom 返回空字符串而非 JSON null，或给 excluded atom 返回 null。V4.1 parser 会在
本地确定性规范化这些字段：reasoning 清空 exclusion reason，answer serialization 固定到专用隔离类别，
unusable 固定到 `other` 类别；规范化字段列表写入 canonical checkpoint。failure transition、repair action、
applicability、verification、受控角色和实例泄漏检查仍保持严格，不能通过规范化绕过。

canonical、pair 和 audit API 单元分别追加到独立 JSONL checkpoint；`--resume` 只复用 prompt、teacher、输入
和 payload hash 全部一致的单元。一个批次若反复返回截断或 schema 错误，会递归二分；仍失败的 singleton 被
确定性归档，其他条目继续。HTTP 402/403、代理耗尽或网络错误仍失败关闭，但此前 checkpoint 保留。BGE 向量
及其 atom 顺序、输入文本、model revision、dtype、shape 和文件 hash 也独立认证，恢复时不会重复编码。默认
canonical batch、pair batch 和近邻数分别为二十四、二十四和十二，可通过
`MEMGEN_V4_1_CANONICAL_BATCH_SIZE`、`MEMGEN_V4_1_PAIR_BATCH_SIZE` 与
`MEMGEN_V4_1_NEIGHBOR_COUNT` 设置；这些值进入 profile，不能与旧 checkpoint 混用。

每个正式 runtime bank 仍至少由五个不同 sample ID 支持。用 atom embedding 的确定性 farthest-first 选择五到
十个代表，coherence audit、card synthesis 和独立 review 都只读取这批有界证据。card 的 target 只保留适用
范围、诊断、修复动作、验证和禁用边界；reference 只描述重复失败过程及与 target 的边界。线上仍只能加载
target，reference 只用于离线对照和 provenance。

所有 signature/card 字段必须是 process-only 文本，确定性检查禁止数字、boxed answer、分式、显式方程和
GSM8K answer marker。target 卡只保留适用范围、诊断、修复动作、验证和禁用边界；reference 卡只描述重复
出现的失败过程、信号、机制和与 target 的边界，不能写成“如何犯错”的在线指令。

## 3. MI 风格 side-KV bank

每张 target/reference descriptor 都用三种固定形式编译：

- raw descriptor；
- internal reasoning principle 上下文；
- hidden steering note 上下文。

每种形式只保留 descriptor 自身 token 位置在 layer 24 输入处经 input layer norm、`k_proj`、`v_proj` 得到的
canonical pre-RoPE K/V；上下文 prefix 的 token 不进入 memory slot。三种形式的 slots 合并为一个 role
memory。target ID 等于 `bank_id`，reference ID 固定为 `bank_id::reference`。

三种形式增加 slot 数后，memory attention 仍使用 `log_valid_slots` normalization，并冻结总先验质量为十，
避免更长 descriptor 或更多 variant 仅因 slot 数量获得更大的注意力质量。编译工件必须同时含 target 和
reference role 以支持 provenance/contrast 审计，但 loader 只公开 `get_target()`；没有 online reference
读取接口。

## 4. 一层 source-state selector

V4 不再把自然语言 card embedding 当作 runtime key。每个 construction pair 都 teacher-force verified
failure 与 success trajectory，并复用已认证 V3.4 entropy+risk gate：

- positive：failure trajectory 第一次 counterfactual joint-gate token，取 layer-24 输出 states 中从推理
  开始到当前为止最近最多十六个 token 的 raw mean，再 L2 normalize；
- matched negative：按 failure anchor 的归一化推理进度，在 paired success trajectory 选同进度 token，
  同样做 local-sixteen mean + L2；
- cross-bank negative：其他 qualified cluster 的 failure anchors。

一个 bank 的 online anchor 资格仍要求至少五个不同 problem 的 positive gate states。没有 joint gate 的
construction pair 保留在 process-card provenance 中，但不能充当 selector positive；因此某张 card 可能
通过语义构造，却因不足五个 state anchors 而不能进入 online selector。online selector 至少需要两个
qualified banks。

对一个 normalized query `q` 和 bank `b`：

```text
positive_evidence(b, q) = logmeanexp(cos(q, positive anchors of b))
negative_evidence(b, q) = logmeanexp(cos(q, negative anchors of b))
score(b, q) = positive_evidence - negative_evidence
```

`logmeanexp` 消除 anchor 数量带来的直接先验。所有 bank 一次评分，以 `score desc, bank_id asc` 稳定排序。
只有 top-one score 达到 absolute threshold 且 top-one/top-two margin 达到 margin threshold 才选择；否则
abstain。

两个阈值只在 bank-source state 上做 leave-one-problem-out calibration。校准一个 failure query 时，从所有
bank 的 evidence 中去掉该问题对应的 failure anchor；校准 success query 时去掉其 matched-success anchor。
固定目标是在 success false selection rate 与 failure wrong-routing rate 都不超过五个百分点的候选中，最大化
correct failure selection coverage；并以总 unsafe count、较低 absolute threshold、较低 margin threshold 做
确定性 tie-break。每个 qualified bank 还必须至少有一个 leave-one-problem-out failure state 被正确选择，
否则不生成 online-qualified selector artifact。该过程不读取 evaluation answer、task reward、target K/V 或
reference K/V。

## 5. Gate-scoped memory episode

每题最多三次 selector attempt，状态为 `ARMED / ACTIVE / COOLDOWN / EXHAUSTED / CLOSED`：

- `ARMED`：无 active memory；joint gate 为正才运行 selector；
- selection：回滚处理当前 token 前的 native cache，激活 target，重算同一 token；这一步计为 direct-memory
  episode 的第一步；
- `ACTIVE`：不允许重选；每个实际 memory-conditioned token 更新低熵 streak 与 active-step count；
- 连续两个 `entropy <= low`：当前 token 后卸载 direct side-KV，并直接回到 `ARMED`；
- 满三十二步仍未恢复：当前 token 后卸载，进入 `COOLDOWN`；连续两个 native low token 后 re-arm；
- abstain：消耗 attempt，未到第三次时进入 `COOLDOWN`，两个 low token 后可再次 gate；第三次后
  `EXHAUSTED`；
- later episode 可以选择相同或不同 bank；同一 episode 内不 replacement；
- answer marker、EOS 或 generation budget 结束都进入 `CLOSED` 并卸载 memory。

direct side-KV 卸载后，过去 token 已形成的 model cache 仍然保留其正常因果历史；“卸载”严格指未来 attention
不再看到额外 memory K/V slots。side-KV slots 从不写入 Hugging Face native cache。

## 6. 工件与执行

默认目录：

```text
OUTPUT_ROOT/
  offline/construction/                 # 已完成的旧 V4 source signature 工件，只读
    repair_signatures.jsonl
    construction_profile.json
  offline/construction_v4_1/
    canonicalization_units.jsonl
    canonical_atoms.jsonl
    canonical_embeddings.npy
    canonical_embeddings_manifest.json
    exact_seeds.json
    candidate_pairs.json
    pair_judgment_units.jsonl
    pair_judgments.jsonl
    clique_candidates.json
    cluster_audit_units.jsonl
    cluster_plan.json
    process_cards.jsonl
    card_reviews.jsonl
    rejected_clusters.jsonl
    bank_records.jsonl
    bank_manifest.json
  offline/construction_v4_2_local/
    construction_profile.json
    local_atoms.jsonl
    mechanism_embeddings.npy
    repair_embeddings.npy
    applicability_embeddings.npy
    multiview_embeddings_manifest.json
    positive_edges.jsonl
    positive_edge_manifest.json
    local_clusters.jsonl
    cluster_review_packets.jsonl
    local_cluster_plan.json
    api_preflight_report.json
  offline/side_kv/
    v4_side_kv.safetensors
    v4_side_kv_manifest.json
    v4_side_kv_compile_report.json
  offline/selector/
    v4_selector_anchors.safetensors
    v4_selector_anchor_manifest.json
    v4_selector_calibration_rows.jsonl
    v4_selector_calibration_report.json
    v4_selector_anchor_compile_report.json
  evaluation/dev-test/
    v4_results.jsonl
    v4_report.json
```

当前建议先完成 V4.2 本地聚类。它复用 `offline/construction` 中已经生成的 signature，不读取旧 map/reduce，
也不需要设置 DeepSeek key：

```bash
python scripts/build_v4_2_local_clusters.py \
  --experiences "$PHASE1_DIR/verified_experiences.jsonl" \
  --split-manifest "$PHASE1_DIR/split_manifest.json" \
  --source-signatures "$V4_SOURCE_DIR/repair_signatures.jsonl" \
  --source-construction-profile "$V4_SOURCE_DIR/construction_profile.json" \
  --output-dir "$OUTPUT_ROOT/offline/construction_v4_2_local" \
  --dataset-revision main \
  --embedding-device cuda \
  --resume \
  2>&1 | tee "$OUTPUT_ROOT/v4_2_local_cluster.log"
```

完成后先检查 compact preflight，不能把 candidate 当成正式 bank：

```bash
jq '{status, external_api_calls_made, api_key_read,
     qualified_candidate_count, planned_initial_synthesis_requests,
     max_api_candidates, within_candidate_guardrail,
     local_graph_diagnostics}' \
  "$OUTPUT_ROOT/offline/construction_v4_2_local/api_preflight_report.json"
```

只有检查 candidate 规模和语义样本之后，才应另行实现并显式启动 candidate-level synthesis。现有
`run_v4_system.sh --stage construct-cards` 消费的是 V4.1 `cluster_plan.json`，不能读取 V4.2 local plan。

以下 V4.1 付费命令保留为历史复现实验，不是当前推荐路径。检查
`construction_v4_1/cluster_plan.json` 后，再构造 card：

```bash
bash scripts/experiments/gsm8k/run_v4_system.sh \
  --stage construct-cards \
  --resume \
  "$PHASE1_DIR" \
  "$E0_DIR" \
  "$RISK_ARTIFACT" \
  "$OUTPUT_ROOT/v4"
```

也可以从 V4.1 构造到 dev-test 一次完整运行：

```bash
export DEEPSEEK_API_KEY='...'
export MEMGEN_V4_CUDA_VISIBLE_DEVICES=0

bash scripts/experiments/gsm8k/run_v4_system.sh \
  --stage all \
  --logical-split dev-test \
  --limit 8 \
  --resume \
  "$PHASE1_DIR" \
  "$E0_DIR" \
  "$RISK_ARTIFACT" \
  "$OUTPUT_ROOT/v4"
```

已有离线工件后，只跑或恢复 evaluation：

```bash
bash scripts/experiments/gsm8k/run_v4_system.sh \
  --stage eval \
  --logical-split dev-test \
  --limit 0 \
  --resume \
  "$PHASE1_DIR" \
  "$E0_DIR" \
  "$RISK_ARTIFACT" \
  "$OUTPUT_ROOT/v4"
```

V4 初始 runner 只允许 `calibration-val` 和 `dev-test`，没有 `final-test` 分支。每题同时记录 cache-greedy
baseline 与 V4 completion。正式报告给出 strict accuracy/uplift、recovered failures、harmed successes、selection
与 abstain 数；runtime audit 还要求每次 selected activation 有 native-vs-target 首步 KL、每个 active step 的
memory attention mass 为正、native cache 长度不含 memory slots、每个 episode 不超过三十二步，以及所有
online memory ID 都不是 reference。

当前 V4 输出仍是 research evaluation，不是 final-test 或论文结论。只有服务器完成真实 DeepSeek 构造、GPU
side-KV/anchor 编译和 dev-test 因果评估后，才能判断新 bank 是否缓解 V3.8 的 coverage gap，以及一层
source-state selector 是否能在控制 harmful selection 的同时找到这些 repair banks。
