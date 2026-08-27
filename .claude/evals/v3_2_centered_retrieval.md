## EVAL DEFINITION: v3-2-centered-retrieval

### Capability evals

- [ ] The centered retrieval mode computes one centroid from the authenticated raw key bank.
- [ ] The same centroid is subtracted from every key and online query before L2 normalization.
- [ ] Centered top-2 scores, margins, centroid identity, and transformed embedding identities are logged.
- [ ] Margin calibration is answer-blind, transform-bound, and still targets 50% first-attempt retention.
- [ ] A matched V3.1 raw-margin versus V3.2 centered-margin comparison reports task, token, mechanism, and calibration-hubness deltas.

### Regression evals

- [ ] The default raw exact-cosine retrieval path preserves bank-order tie breaking and prior audit fields.
- [ ] Layer 24, full partial-CoT queries, three attempts, entropy re-arm, replacement, side-KV, and injection strength remain frozen.
- [ ] Existing V3/V3.1 tests pass.
- [ ] Existing V3.1 calibration artifacts that predate the transform field remain valid as raw-space artifacts.
- [ ] The V3.2 experiment runner never invokes final-test.

### Experiment success metrics

Offline/calibration qualification:

- Integrity checks pass.
- V3.2 first-memory top-1 share is below the V3.1 calibration value.
- V3.2 selection Gini is below the V3.1 calibration value.
- The frozen threshold retains approximately 50% of calibration first attempts, with deterministic tie handling.

Matched dev-test decision:

- Primary: strict accuracy delta, paired bootstrap CI, and McNemar exact p-value.
- Secondary: format accuracy, generated-token delta, activations, replacements, duplicates, abstains, and memory-attention steps.
- No final-test rerun is authorized by this eval.
