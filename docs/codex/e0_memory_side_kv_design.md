# E0-v1：Experience Memory 与 canonical side-KV 设计契约

## 1. 范围

E0-v1 只验证三件事：Phase 1 审核抽象能否安全形成在线 payload、冻结 reasoner 能否把
payload 编译为 layer-24 canonical pre-RoPE K/V、以及运行时能否在不改写 native KV cache
的情况下实际读取这些 slots。

E0 不调用新的 Teacher/Pro，不使用任务正确率，不实现或恢复 residual-vector action。

## 2. 对象边界

```text
ApprovedMemorySourceSelector
  └─ 验证 ai_approved + answer_correctness + verified_failure + provenance
      并按冻结 schema 严格验证 Pro review v1/v2

PayloadSanitizer
  └─ 只做规范化和 fail-closed 泄漏检查，不改写语义

MemoryRecordCompiler
  └─ 按固定 lineage 渲染三行 payload，并计算 token/hash/provenance

MemoryBankBuilder
  └─ 编排选择、编译、去重和审计 trace

BM25MemoryIndex / RetrievalQueryBuilder
  └─ 透明的 sanitized-key 检索与 question + partial-CoT query；零分时 abstain

CanonicalSideKVCompiler
  └─ 使用冻结 reasoner 的 block-24 input norm 和 K/V projection 编译 slots

SideKVBankLoader
  └─ 校验 revision/hash 并按 memory ID 加载只读 tensor

SideKVAttentionController
  └─ 管理单样本 memory 生命周期、联合 attention 和机制 trace
```

记录构造对象位于 `memgen/experience/memory.py`，检索对象位于
`memgen/experience/retrieval.py`；两者都不依赖 Torch/Transformers。模型相关对象位于
`memgen/model/side_kv.py`，不修改原 Weaver/Trigger 类。

Phase 1 已存在两种不可互换的 Pro-review artifact。早期冻结 bank 使用
`phase1-ai-review-record-v1`：E0 要求 reviewer decision 为 `approve`、六项整体 criteria 全为
`true`、confidence 不低于该记录的 `routing_confidence_threshold`，并重新检查当时的自动完整性
审计。新 bank 使用 `phase1-ai-review-record-v2`：E0 要求八个字段和五个 pair assessment 全为
`supported` 且各自 evidence 非空。只按精确 schema/prompt version 分派验证策略；未知版本不能
依据相似字段自动推断。两种策略都不新增 Reviewer 调用，最终仍只允许
`ai_approved + answer_correctness` 进入 memory pool。审计报告分别记录 selected/accepted review
profile 数量。

## 3. Payload 契约

固定字段映射：

```text
When facing = target.situation_signature + target.applicability_boundary
Prefer      = target.transferable_decision + target.verification_rule
Avoid       = reference.competing_pattern + reference.failure_signal
              + reference.failure_mechanism
```

`reference.non_reuse_boundary` 仅保留于 Phase 1 provenance；E0-v1 不把它拼入 payload，避免把
“何时不应复用失败模式”的例外语义错误渲染为在线 Avoid 指令。

编译器只允许 NFKC、控制字符删除、空白折叠和完全重复句去重。任一必需字段包含具体数字/公式、
`\\boxed`/`\\fbox` 答案容器，或完整复制原题、轨迹、Teacher evidence、Pro evidence 时，整条
record 被拒绝。泛指的 “final answer” 和普通英文数量词本身不携带实例答案，不能仅凭这些词
判为泄漏。局部 n-gram overlap 只写入 `payload_diagnostics` 和汇总报告，不再重复执行 Phase 1
的抽象质量准入。编译器不得通过删除局部 literal 后继续使用剩余句子。

## 4. Layer 与 KV 契约

```text
研究层号                         24
HF decoder block index           23
KV compiler input hidden tuple   hidden_states[23]
风险 state hidden tuple          hidden_states[24]
relative phase delta             0
```

对 payload token 的 block-24 输入状态 `h`：

```text
x     = block24.input_layernorm(h)
K_pre = block24.self_attn.k_proj(x)
V     = block24.self_attn.v_proj(x)
```

K 在任何 rotary operation 之前保存，形状为 `[num_kv_heads, slots, head_dim]`。GQA 展开只发生
在 runtime attention 中。

## 5. Runtime attention

E0-v1 固定 eager attention、batch size 1、layer 24 全部 KV groups、每样本一条 memory，首次
激活后持续到 EOS。没有 selector、gain 或 alpha。

native score 使用正常 rotated Q/K；memory score 使用 live pre-RoPE Q 与 canonical K：

```text
S_native = Q_rotated @ K_native_rotated.T / sqrt(d) + native_mask
S_memory = Q_pre_rope @ K_memory_pre_rope.T / sqrt(d) + memory_mask
weights  = softmax(concat(S_native, S_memory))
output   = W_native @ V_native + W_memory @ V_memory
```

只有 native K/V 调用 Hugging Face cache update。memory slots 不进入 DynamicCache、不改变
attention mask 的真实 token 长度，也不消耗 position。

触发 boundary 必须按以下方式重放，避免 one-token delay：

```text
prefill prefix[:-1] -> native cache
activate memory
forward prefix[-1]  -> boundary query 首次读取 memory
```

## 6. E0 artifacts 与状态

`compile_experience_memory_bank.py` 输出：

- `memory_records.v2.jsonl`
- `memory_compilation_trace.jsonl`
- `payload_audit_report.json`
- `bm25_index.v1.json`
- `side_kv_bank.safetensors`
- `side_kv_manifest.json`
- `e0_report.json`

其中 `e0_report.json` 在 KV 编译后仍标为 `pending_runtime_audit`。只有
`audit_side_kv_mechanism.py` 在 calibration prefix 上证明 disabled parity、历史 cache 保持、
cache 长度不含 memory slots、shared-phase RoPE score identity、memory attention mass 非零，
并要求至少一个 answer-blind case 的 next-token logits KL 超过冻结的数值噪声阈值后，才输出
`e0_final_report.json` 并把 `formal_e0_passed` 标为 true。

shared-phase identity 的相对误差阈值固定为 `2e-2`，用于覆盖 bfloat16 RoPE table 的量化误差；
它不是依据任务正确率选择的超参数。

正式 audit case 只允许来自 `calibration-val`，必须声明 `answer_or_reward_used=false` 并携带
prompt span 与 prefix hash。审计端会重新解码 completion span，拒绝已到达答案标记或未停在
推理 delimiter 的 case。

一键入口：

```bash
bash scripts/experiments/gsm8k/run_e0_experience_memory.sh \
  /absolute/path/to/phase1-run-dir
```

第二个参数可选；未提供时会从冻结 split manifest 的 `calibration-val` 自动生成 answer-blind
audit cases。外部 case 文件也必须满足同一 schema 和来源审计。

E0 不设置人为 payload token budget；它编译通过硬约束的完整 payload，记录
`min/median/p95/max`，只以 reasoner/tokenizer 的真实序列上限作为技术边界。E1 使用的全局容量
必须在读取 E0 长度、KV footprint 和系统成本后冻结，不能依据 E1 或 final-test accuracy 选择。
`.e0.server.env` 因而只包含输出根目录和可选 GPU 编号；模型、序列上限及数据 revision 均从
Phase 1 artifacts 和冻结 reasoner 配置自动读取。
