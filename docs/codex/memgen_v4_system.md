# MemGen V4：MI 风格 repair bank 与单层在线 selector

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

raw verifier-backed bank-source pairs + official GSM8K solution
  └─ DeepSeek V4 Flash：逐 pair 抽象 repair signature
       └─ 按 failure mechanism + repair operator 聚类
            ├─ 至少五个独立 construction problem
            ├─ target process card：official solution + verified success
            └─ reference process card：paired verified failure
                 └─ 独立 DeepSeek semantic review
                      └─ target/reference 三种上下文化形式分别编译 layer-24 side-KV

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

输入不是 V3 teacher bank 或旧的 104/161 条 memory，而是 Phase-1 的原始、verifier-backed contrast
pairs。每个 pair 必须来自 `bank-source`，并同时满足：target reward 为一、reference reward 为零、source
provenance 与 split manifest 完整一致。官方 GSM8K solution 通过 source index、question hash、answer hash
和 dataset fingerprint 重新 join，不信任自由文本路径。

DeepSeek 模型固定为 `deepseek-v4-flash`、temperature 为零、thinking disabled、JSON mode。每次调用记录
model、base URL、prompt version、construction input hash 与输出 hash，但绝不落盘 API key。构造分四次
独立调用：

1. 每个 contrast pair 抽象一个 instance-free repair signature；
2. 全局按 `failure_mechanism + repair_operator` 聚类；
3. 每个 cluster 合成 target/reference process card；
4. 对已经生成的 card 做只审核、不改写的 semantic review。

Signature 构造逐条持久化。若 DeepSeek 请求成功但返回的 JSON 未通过 V4 schema 或实例泄漏检查，后续短重试会
携带上一条输出和本地校验原因，要求模型定向修正。若短重试仍失败，该 construction example 会被写成
`applicable=false` 的确定性拒绝审计记录并从聚类输入中排除，批处理继续；系统不会据此编造 repair
signature。网络故障、代理重试耗尽、鉴权、余额或不可重试 HTTP 错误仍然失败关闭。
`repair_signatures.jsonl` 的 `generation_status` 区分正常教师输出与受控拒绝，最终 manifest 的
`teacher_invalid_signature_ids` 汇总后者。

cluster 不按题目故事、对象或宽泛数学主题划分。一个正式 runtime bank 至少含五个不同 sample ID；DeepSeek
从中选择五到十个独立 representative examples 覆盖结构变化。少于五个、机制混杂、无法落到单一 repair
operator 的组被拒绝，不通过合并弱相关 singleton 凑数。

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
  offline/construction/
    repair_signatures.jsonl
    cluster_plan.json
    process_cards.jsonl
    card_reviews.jsonl
    rejected_clusters.jsonl
    bank_records.jsonl
    bank_manifest.json
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

完整 smoke run：

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
