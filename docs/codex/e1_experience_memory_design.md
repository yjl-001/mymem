# E1：匹配经验内容的配对因果实验

## 1. 目标与边界

E1 检验 BM25 选出的 Phase 1 经验内容是否比错配经验或仅观察 gate 更能改善 GSM8K，且不损害
严格 `\\boxed{}` 格式。E1 不搜索 layer、注入强度、memory 数量或 residual vector；不调用新的
Teacher/Pro，也不在 `final-test` 上选择配置。

首版固定：

- layer 24 canonical pre-RoPE side-KV；
- sink-masked entropy 与冻结 persistence-risk score 联合 gate；
- 首个同时通过两阈值的 pre-answer delimiter；
- 每题最多一次、每次一条 memory；
- 96-token partial-CoT query window；
- BM25 top-1；
- greedy decoding，最多 768 completion tokens。

## 2. 两阶段执行

### 2.1 Answer-blind assignment prepass

`build_e1_assignment_manifest.py` 只读取 question、冻结 split、risk artifact、E0 MemoryRecords、
BM25 index 和 side-KV metadata，不读取 gold answer、reward 或 verifier 结果。它运行完整
`gate-observation-only` completion，并冻结：

```text
sample/question hash
prompt hash
first joint entropy-risk boundary
exact prefix token IDs/hash
retrieval query hash/terms
matched memory ID/score/payload hash/KV slots
shuffled memory ID/payload hash/KV slots
gate-only completion token IDs/hash
```

若没有联合 trigger 或 BM25 正分 hit，样本以稳定 reason abstain。检索返回的是 `memory_id`；运行时
用相同 ID 从 E0 side-KV bank 取 K/V，不做第二次 hidden/KV nearest-neighbour 检索。

### 2.2 Frozen four-condition execution

`evaluate_e1_experience_memory.py` 读取不可变 manifest：

- `vanilla`：重新 greedy 生成；
- `gate-observation-only`：assignment prepass 已产生的 control completion；
- `matched-memory`：从固定 prefix 在 boundary token 的 layer-24 attention 附加 BM25 top-1 KV；
- `shuffled-memory`：相同 prefix/boundary/runtime，仅改 memory ID。

vanilla 与 gate-only 必须 token-level 完全一致。matched/shuffled 都从 prefix 重新建立独立 native
cache；memory slot 不写入 cache，也不改变真实 token position。

## 3. Shuffled control

对全部 matched IDs 作确定性 bijective derangement：

1. 每个 sample 的 shuffled ID 必须不同于 matched ID/payload；
2. 两个 arm 的 memory-ID 多重集合完全相同；
3. 因而全局 payload/KV-slot 分布完全相同；
4. 在所有合法循环置换中选择 paired KV-slot 绝对差最小者；
5. 额外报告 matched/shuffled exact-slot 子集。

若某个 matched ID 占比超过一半而无法无自配对，assignment 明确失败，不改用另一套回退对照。

## 4. 判定

主要集合是 manifest 中 `triggered + retrieved + shuffled` 的 assigned subset；同时报告全样本
intention-to-treat：

1. paired bootstrap 和 McNemar：matched accuracy − shuffled accuracy；
2. paired bootstrap 和 McNemar：matched accuracy − gate-only accuracy；
3. matched format accuracy − vanilla format accuracy；
4. memory attention mass、first-step logits KL、BM25 score、长度作为诊断。

正式 E1 通过要求 assignment/pairing 完整、vanilla/gate token parity、至少固定数量配对样本、两项
accuracy difference 的 paired 95% bootstrap CI 下界大于零，以及 matched 格式准确率不低于
vanilla。若未通过，结论仅限当前 payload/BM25/side-KV 组合。

## 5. 服务器入口

```bash
cp scripts/experiments/gsm8k/e1.server.env.example \
  scripts/experiments/gsm8k/.e1.server.env

bash scripts/experiments/gsm8k/run_e1_experience_memory.sh \
  --limit 100 \
  /absolute/path/to/phase1-run \
  /absolute/path/to/formal-e0-run \
  /absolute/path/to/phase2-entropy-risk-state-delta-answer_correctness.pt
```

`.e1.server.env` 只包含 output root 和可选 GPU 编号。模型、tokenizer、dataset revision、layer、
检索器和预算均由冻结 artifact 或脚本常量解析。
