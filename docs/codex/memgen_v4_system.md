# MemGen V4.2：当前系统合同

本文只描述当前仍有效的 V4 系统边界。完整的 bank 构造路线、已关闭实验、具体计数与 selector 失败分析见
[V4 bank 研究台账](memgen_v4_bank_research.md)。

V4 延续 V3 的研究目标：用 source reasoning state 判断当前状态是否具有相同的可修复缺陷，再把对应 repair
memory 通过 side-KV 插入推理流。它不修改 V3.5 冻结实现，也不恢复自然语言 key 与 hidden-state query 的直接
cosine 路线。

## 1. 冻结边界

当前只研究 GSM8K，并固定：

- selector 对所有 bank 一次评分，不设置 task/domain 第一层；
- reasoner、tokenizer 与 V3 已认证工件使用相同精确 revision；
- layer 24 单点、全部 KV groups、SDPA、canonical pre-RoPE、relative phase delta 为零；
- target 与 reference 离线共同构造、共同编译，线上只允许 target；
- auxiliary 只保留 schema 扩展位，不构造内容；
- 每个 bank 至少由五个不同 construction sample 支持；
- 每题最多三次 selector attempt；memory 恢复后可卸载和重新选择，不持续到回答结束；
- layer/head calibration 留到 bank、selector 和生命周期稳定以后。

当前状态不是 online ready：17 个 curated bank 的 target/reference Side-KV 已编译通过，但 dynamic-only
selector qualification 失败。116 条 construction evidence 中只有 66 条产生 joint-gate anchor；11 个 bank
不足五条 anchor，剩余 6 个 bank 的安全校准只能正确选择 2/35 条 failure query。因此当前不会生成 selector
tensor，也不得启动 dev-test evaluation。

## 2. 两阶段结构

```text
阶段一：离线 bank

已认证 repair signatures
  └─ serialization / non-applicable 归档
  └─ reasoning atoms
       └─ mechanism / repair / applicability 三视图 embedding
            └─ mutual-kNN + 三阈值正边
                 └─ deterministic complete-link
                      └─ >=5 distinct samples
                           └─ high-quality shortlist + centroid NMS
                                └─ deterministic local-direct cards
                                     └─ hash-bound static curation
                                          └─ target/reference layer-24 side-KV
                                          └─ answer-blind selector audit

阶段二：在线使用

pre-answer token
  └─ frozen entropy+risk joint gate
       └─ one-stage state-to-bank selector
            ├─ qualified：激活唯一 target side-KV
            └─ abstain：不注入并进入 cooldown/exhaustion 生命周期
```

side-KV 编译、selector source-state 构造和阈值冻结都是离线 bank 工件的一部分，不额外划分研究阶段。

## 3. 当前 bank

### 3.1 本地三视图构造

V4.2 复用已经生成并认证的 5318 条 repair signature，不重新调用教师。确定性角色边界将 answer
serialization 与 source non-applicable 归档，其余 3078 条 reasoning atom 分别编码：

- mechanism：failure mechanism + decision point；
- repair：repair operator + verification operator；
- applicability：problem structure + decision point。

固定 `BAAI/bge-small-en-v1.5` revision 对三个视图分别 L2 normalize。joint cosine 只召回 mutual-kNN；一条
正边还必须分别通过 `0.82/0.82/0.70`。greedy complete-link 要求新成员与组内所有成员都有正边，禁止
A-B、B-C 在缺少 A-C 时传递合并。

每组至少五个不同 sample 才成为 candidate；其余长尾明确归档，不为了覆盖全部数据而强行合并。本地构造得到
95 个 candidate，随后按最弱视图余量、support tier、三视图 centroid NMS 与多样性缩小为 25 个。人工 policy
排除 1 个后，local-direct 使用 24 个 candidate、167 条 evidence。

### 3.2 Local-direct 与 curated

local-direct 不做新的模型语义审核。它在每个 candidate 内按 mechanism/repair/applicability 的冻结权重
`0.45/0.45/0.10` 选择 joint medoid，再确定性映射 process card：

- target：scope、diagnosis、action、verification、do-not-use boundary；
- reference：failure signal、failure mechanism、undesired pattern、contrast boundary；
- auxiliary：`null`。

它明确记录 `semantic_audit_performed=false`、`independent_review_performed=false` 与
`quality_tier=provisional_local_direct`。静态 curation 对 24 张 card 做 hash-bound 检查，保留 11 张 primary
与 6 张 conditional，排除 3 张 quarantine 与 4 张 hard reject，得到当前 17-bank/116-evidence 主实验。

curated 工件仍是 `provisional_local_curated`；静态 card 检查不能被表述成完整 construction-evidence semantic
audit。

## 4. Side-KV 合同

每张 target/reference descriptor 编译三种固定形式：

- raw descriptor；
- internal reasoning principle；
- hidden steering note。

只保留 descriptor token 在 layer 24 输入处经 input layer norm、`k_proj`、`v_proj` 得到的 canonical pre-RoPE
K/V；context prefix 不进入 memory slot。三种形式的 slots 合并为一个 role memory。target ID 等于 bank ID，
reference ID 为 `bank_id::reference`。

memory attention 使用 `log_valid_slots` normalization，并冻结总先验质量为 10，避免 descriptor 长度或 variant
数量形成额外先验。工件同时保存 target/reference 以支持 provenance 和离线 contrast，但 loader 只提供
`get_target()`，没有 online reference 接口。

当前 17 target + 17 reference 已通过编译。这只证明工件、角色和注入通道成立，不证明 selector 可用或 memory
能改善任务结果。

## 5. Selector 基线与当前阻塞

当前已失败的 baseline 使用 failure trajectory 第一次 V3.4 entropy+risk joint gate，在 layer 24 对最近最多
16 个 reasoning states 做 mean + L2：

- 本 bank failure states 是 positive；
- matched success 同进度 states 是 negative；
- 其他 bank failure states 是 cross-bank negative。

对 query `q` 和 bank `b`：

```text
positive(b,q) = logmeanexp(cos(q, positive anchors of b))
negative(b,q) = logmeanexp(cos(q, negative anchors of b))
score(b,q) = positive(b,q) - negative(b,q)
```

LOO 校准要求 success false-selection 和 failure wrong-routing 都不超过 5%，再最大化正确 failure coverage。
实际冻结到 `absolute=0.0611734390`、`margin=0.0`，只有 2/35 个 failure 被正确选择。五个 bank 虽然各自至少
有一次 raw correct Top-1，却全部低于全局 absolute threshold；其中一个 bank 的 raw Top-1 为 6/6。

因此该 baseline 只保留为诊断，不再视为待放宽阈值的候选。下一步必须先建立 offline-only source-state cache，
再在同一份状态上比较 gate-conditioned success calibration、prompt semantic state、prompt+dynamic 单层联合分数
和跨-bank score normalization。无论组合多少特征，线上仍是一次性对全部 bank 排分，不恢复两层 selector。

校准失败时必须：

- 写出 calibration rows/report、skipped experiences、rejected banks 和总报告；
- 设置 `qualified_for_online_use=false`；
- 不生成 selector tensor/manifest；
- 清除同一输出目录可能残留的旧 selector runtime artifact。

## 6. Memory episode 生命周期

每题最多三次 selector attempt，状态为 `ARMED / ACTIVE / COOLDOWN / EXHAUSTED / CLOSED`：

- `ARMED`：无 active memory；joint gate 为正才运行 selector；
- selection：回滚处理当前 token 前的 native cache，激活 target 并重算同一 token；
- `ACTIVE`：不允许重选，每个 memory-conditioned token 更新低熵 streak 和 active-step count；
- 连续两个 `entropy <= low`：当前 token 后卸载 memory，回到 `ARMED`；
- 满 32 步仍未恢复：卸载并进入 `COOLDOWN`，连续两个 native low token 后 re-arm；
- abstain：消耗 attempt；第三次后进入 `EXHAUSTED`；
- answer marker、EOS 或 generation budget 结束：进入 `CLOSED` 并卸载。

卸载只表示未来 attention 不再看到额外 side-KV slots；已经形成的原生 causal cache 保持不变，side-KV slots
从不写入 Hugging Face native cache。

## 7. 当前工件与入口

```text
OUTPUT_ROOT/offline/
  construction/                       # 5318 条已认证 signature，只读来源
  construction_v4_2_local/            # 三视图 atoms、embedding、graph、95 candidates
  construction_v4_2_shortlist/        # 25-candidate high-quality basis
  construction_v4_2_semantic/         # 零 API evidence preflight；paid 路径未执行
  construction_v4_2_local_direct/     # 24-bank/167-evidence provisional ablation
  construction_v4_2_local_curated/    # 17-bank/116-evidence 当前主实验
  side_kv_v4_2_local_curated/          # 已通过的 17 target + 17 reference
  selector_v4_2_local_curated/         # 当前只有失败诊断，不存在合格 runtime tensor
  v4_oracle_full/
    source_state_cache/                # offline-only layer-24 raw source states
    source_state_audit/                # CPU-only window/normalization/LOO 诊断
    oracle_audit/                      # exact-prefix target/reference causal audit
```

当前唯一推荐的 curated offline 入口：

```bash
bash scripts/experiments/gsm8k/run_v4_2_curated_offline.sh \
  --stage curate|side-kv|selector|all \
  "$PHASE1_DIR" \
  "$E0_DIR" \
  "$TOKEN_RISK_ARTIFACT" \
  "$OUTPUT_ROOT"
```

在 selector 新方案通过 qualification 前，不运行 `scripts/evaluate_v4_experience_memory.py`。旧 V4.1 付费总 runner
和 map-reduce progress inspector 已删除；V4.0/V4.1 的底层 schema/parser 暂时保留，只用于 source-signature
认证、历史工件兼容和当前 V4.2 的直接 imports。

### 7.1 Source-state cache 与 oracle audit

selector 改造前先运行 source-side mechanism qualification：

```bash
bash scripts/experiments/gsm8k/run_v4_source_oracle_audit.sh \
  --mode smoke|full \
  --stage cache|state-audit|oracle|all \
  "$PHASE1_DIR" \
  "$CURATED_BANK_DIR" \
  "$SIDE_KV_DIR" \
  "$TOKEN_RISK_ARTIFACT" \
  "$OUTPUT_ROOT"
```

cache 对每个 construction sample 保存 prompt-end、question mean/boundary/local-window，以及 failure 最多三次
实际 counterfactual gate window。failure attempt 的 matched-success aligned window 只作为 repair-direction
对照；success 轨迹自身实际触发的 gate window 才是 safety negative。大张量写入 safetensors，逐 event metadata
写入 JSONL，manifest 与 gate reachability report 绑定全部文件哈希。cache 固定
`offline_only=true`、`qualified_for_online_use=false`、`contains_reward_or_answer_signal=false`，线上 loader 不得消费。

CPU auditor 从 raw32 派生 local1/4/8/16/32，按 sample 做 leave-one-out，分别报告 construction support 与 gate
support，并比较 raw score、bank-specific empirical tail normalization、hubness、bank-size bias 和安全阈值诊断。
阈值错误率与 coverage 使用独立 sample；event-level 指标只保留为状态观察诊断，另报 sample-macro 路由指标。
多个 attempts 可以是多个状态观察，但不会增加独立 support。即使某个诊断配置表现较好，该步骤也不会生成
online selector tensor。

oracle audit 对每个 cache 中实际可达的 failure gate event 使用其已知 bank ID，从同一 token prefix 和一次 replay
得到的 native cache 克隆 baseline、target、reference 三支，固定 greedy、32-step bounded continuation 与当前
非持续 lifecycle。分支前除 cache length 外，还逐 tensor 检查 shape、dtype、value 完全相同，并验证三份 cache
不共享底层 mutable storage；任一 parity 条件失败即停止该 case。reference 只能由独立 offline loader 读取。
报告保留全部 17 个 bank，单列 gate-unreachable failure，并把本结果限定为 construction source 上的 optimistic
positive control / mechanism qualification，不能表述为 held-out 泛化。

## 8. 代码边界

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
- `memgen/experience/v4_source_state.py`
- `memgen/experience/v4_oracle_audit.py`
- `memgen/model/v4_oracle.py`
- `memgen/model/v4_side_kv.py`
- `memgen/model/v4_selector.py`
- `memgen/model/v4_online.py`
- `memgen/model/v4_runtime.py`

保留但不作为默认入口：

- V4.0 signature generator 与基础 schema；
- V4.1 schema/parser；
- V4.2 paid semantic builder；
- online evaluator。

删除兼容代码前，必须先解除 active artifact 的 implementation hash、schema import 和 source-signature
authentication 依赖。不能为了减少代码行数使已认证工件在同一 revision 下失效。
