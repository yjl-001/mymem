## EVAL DEFINITION: v3-3-pre-boundary-online

### Capability evals

- [ ] The V3 profile exposes an explicit query-pooling contract and supports both the frozen V3.1 boundary-last baseline and the qualified V3.3 pre-boundary candidate.
- [ ] V3.3 re-encodes `question + full partial CoT`, including the trigger boundary, but selects the layer-24 hidden state immediately before that boundary.
- [ ] Every retrieval attempt records the boundary token, selected query token/index, full-prefix token hash, query embedding hash, scores, margin, and selected memory.
- [ ] Answer-blind margin calibration is bound to both retrieval-embedding transform and query pooling; a V3.1 threshold cannot be loaded by V3.3.
- [ ] The V3.3 selector is calibrated directly from all authenticated `key_last__query_pre_boundary` first-attempt margins in the answer-blind calibration-val pooling audit, with a 50% target retained fraction.
- [ ] Before dev-test, the selector builder exactly reproduces the qualified candidate's sample set, per-sample top-2 traces, and concentration summary.
- [ ] A matched V3.1-margin versus V3.3-pre-boundary-margin dev comparison reports paired strict/format effects, generated-token deltas, mechanism counts, retrieval concentration, margins, and boundary-token strata.

### Regression evals

- [ ] Layer 24, raw key-last memory embeddings, no centering, exact cosine, top-2 diagnostics, and one active memory remain frozen.
- [ ] The three-attempt budget, entropy re-arm, replace-current-memory, duplicate/abstain semantics, and persistence-risk diagnostic-only role remain unchanged.
- [ ] The V3.1 boundary-last mode remains reproducible and legacy selector artifacts without an explicit pooling field are interpreted only as boundary-last.
- [ ] Vanilla completions match exactly across the paired dev runs.
- [ ] No task answer, reward, strict accuracy, or format accuracy is read during pooling qualification or selector calibration.
- [ ] The experiment stops before dev-test when the pooling audit is not qualified, recommends another candidate, or online calibration geometry differs from the audit.
- [ ] The experiment never runs final-test automatically.

### Success metrics

- Code graders: targeted tests, full unit-test discovery, Python compilation, shell syntax, and `git diff --check` pass.
- Calibration qualification: all first-attempt sample IDs, top-2 traces, and selected memory IDs come from the authenticated offline audit candidate; concentration metrics match exactly.
- Matched dev report: integrity passes and the only system differences are query pooling plus pooling-specific answer-blind margin thresholds.
- Task decision: promote to a later final-test only if paired dev strict accuracy is non-degrading; format accuracy and token deltas are reported as guardrails, not substituted for strict accuracy.

### Human review

- Review the sentence-ending-period stratum separately because the offline audit showed higher concentration for pre-boundary queries at `.` boundaries.
- Treat retrieval geometry as mechanism evidence, not proof of task improvement.
