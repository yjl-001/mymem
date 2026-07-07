# Strategy-Level Follow-Up TODO

The current MVP is now a candidate-ranking experiment. Strategy-level reward is
exploratory and should not be used as the main claim yet.

## Why Strategy Rollouts Are Deferred

- Strategy rollouts select `planned_steps` from a reference trajectory.
- Earlier insertions can change later tokens, so later planned steps may no
  longer be delimiter positions.
- Different strategies can have different `inserted_steps` counts, so reward is
  not strictly same-budget.
- `sink_top_b` is an offline selector over the full reference trajectory. A real
  policy must decide online without seeing future delimiter scores.

## Required Before Using Strategy Reward As Evidence

1. Build true online policies:
   - `first_k_online`: insert at the first K delimiter positions encountered.
   - `random_online`: sample online at delimiter positions under a calibrated
     probability until budget is exhausted.
   - `sink_threshold_online`: insert when online SinkMass crosses a threshold,
     with cooldown and budget.
2. Define online normalization for SinkMass:
   - running mean/std,
   - or calibration-set mean/std,
   - not full-trajectory z-score.
3. Report both planned and inserted positions, plus actual insertion counts.
4. Compare under matched actual insertion budgets.
5. Keep `no_memory` and `prompt_only` references separate.

## Current Safe Claim Boundary

Use candidate-level metrics first:

```text
Does SinkMassZ rank fixed-reference delimiter candidates by single-insertion
utility better than first-K/random/same-bucket random?
```

Only after that is stable should strategy-level reward be used as a main claim.
