## EVAL DEFINITION: v3-3-pooling-audit

### Capability evals

- [ ] Reconstruct every calibration first-attempt prefix from authenticated prompt and completion token IDs.
- [ ] Re-encode each prefix once at layer 24 and derive boundary-last, pre-boundary, partial-mean, and full-prefix-mean query vectors.
- [ ] Re-encode each `when_facing` key once and derive last-token and mean-pooled key vectors.
- [ ] Compare exactly four frozen raw-cosine candidates without task answers, rewards, strict accuracy, or format accuracy.
- [ ] Save authenticated per-sample top-2 traces and reusable query/key embedding tensors.

### Regression evals

- [ ] Reconstructed prefix token hashes match the source retrieval traces for every first attempt.
- [ ] Re-encoded last-token key hashes match the authenticated V3 key bank for every memory.
- [ ] The current key-last/query-boundary-last baseline reproduces 100% of logged top-1 memory IDs.
- [ ] Baseline concentration metrics reproduce the frozen V3.1 calibration artifact.
- [ ] Layer 24, full semantic partial-CoT context, raw cosine, memory/KV artifacts, trigger, re-arm, and injection remain unchanged.
- [ ] No dev-test or final-test is invoked by the audit runner.

### Frozen candidates

1. `key_last__query_boundary_last` — current baseline.
2. `key_last__query_pre_boundary` — remove only the mechanical trigger delimiter from query pooling.
3. `key_mean__query_partial_mean` — symmetric semantic mean pooling over key text and partial CoT.
4. `key_mean__query_full_mean` — symmetric key mean plus full-prefix mean excluding the trigger delimiter.

### Answer-blind qualification

A non-baseline candidate qualifies only when, relative to the reproduced raw baseline:

- first-memory top-1 share decreases;
- selection Gini decreases;
- selected-memory count is not lower;
- normalized selection entropy increases.

Qualified candidates are ranked by lower Gini, then lower top-1 share, then higher selected-memory count. Task metrics are not used.
