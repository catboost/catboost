# S29-LG-SPIKE-T2 — DEC-034 mechanism verdict

**Verdict: Outcome A (shared compounding mechanism). Confidence: moderate.**

LG+Cosine drift in a post-S28 codebase is structurally the same float32
joint-Cosine denominator compounding as ST+Cosine, just smaller in
magnitude at the spike cell. A single Kahan/Neumaier fix applied to the
shared joint-denominator accumulator is expected to close both paths.

## Evidence summary

| Signal                         | LG+Cosine (spike, post-S28) | ST+Cosine anchor (S28)   |
|--------------------------------|-----------------------------|--------------------------|
| iter-1 mean drift (%)          | 0.0024 (mean)               | 0.77                     |
| iter-1 per-seed drift (%)      | 0.0046 / 0.0015 / 0.0010    | —                        |
| 50-iter drift (%)              | 0.0029 – 0.1970 per seed    | ~47 (aggregate)          |
| iter=1 tree, seed=0            | root + BFS feature seq      | n/a                      |
|                                | `[0,0,0,1,1,1,1]` identical |                          |
|                                | CPU vs MLX                  |                          |
| Classifier thresholds (README) | A ≤ 1.5%, B ≥ 5%, C in-band | spike lands deep in A    |

Cell: `N=1000, features=10, depth=3, max_leaves=8, bins=128, lr=0.03,
bootstrap=no, rs=0.0, seeds={0,1,2}`. Sources:
`data/iter1_drift.json`, `data/iter_curve.csv`,
`data/tree_structure_iter1.json`.

## Why outcome A

1. **Drift is ~300× smaller, not structurally different.** Spike iter-1
   mean 0.0024% vs ST+Cosine anchor 0.77% differs in magnitude, not in
   kind. Both are sub-1% at iter-1 and both grow with iterations. This
   is the signature of a shared compounding mechanism driven by the
   float32 joint-Cosine denominator accumulator — the same accumulator
   both paths now share after S28 commit `0ea86bde21`.
2. **Priority-queue ordering did not diverge at this cell.** At iter=1,
   seed=0, MLX and CPU produce bit-identical BFS split sequences
   `[0,0,0,1,1,1,1]` and identical root feature. If LG-specific
   priority-queue ordering were the primary drift driver (outcome B),
   we would expect the split sequence or feature choice to flip even
   at iter=1. It did not.
3. **Curve shape matches compounding, not a structural break.** Per-seed
   drift grows from <0.005% at iter=1 to 0.003–0.197% at iter=50 (see
   `iter_curve.csv`). ST+Cosine grew from 0.77% to ~47% over the same
   iteration span on its own cell. Both curves are monotone-ish growth
   in the same direction, which is what a float-accumulator compounding
   mechanism produces. A priority-queue ordering flip would look like a
   step discontinuity, not a smooth ramp.

## Cell-mismatch disclosure (mandatory context)

The t5-gate-report
(`docs/sprint28/fu-fu3-revalidate/t5-gate-report.md`) recorded
LG+Cosine ratio ≈ 1.14 (~14% MLX-worse-than-CPU) at
`N=1000, depth=6, 50 iter`. That measurement is **not comparable** to
the 0.0024% number above and does **not** indicate a larger
float-precision gap on LG. From t5 lines 112–114:

> "MLX `FindBestSplitPerPartition` in the Lossguide path also hardcodes
> L2 Newton gain. When CPU is given `score_function='Cosine'`, the two
> sides are computing different gain functions — exactly the DEC-032
> pattern."

That 14% was **algorithmic divergence** (MLX computing L2 Newton gain
while CPU computed Cosine gain), not float-precision drift. S28
`0ea86bde21` shipped the Cosine dispatch to MLX LG. The spike's
post-S28 0.0024% is therefore the first honest measurement of
LG+Cosine float-precision drift with matching gain functions on both
sides.

## Limitations

- **3 seeds.** Variance estimate weak; iter-50 per-seed spread
  (0.003–0.197%) already suggests seed sensitivity.
- **One shallow cell.** `depth=3, max_leaves=8`. Deeper LG cells and
  higher `max_leaves` were not measured — and the priority-queue
  divergence surface grows with leaf count, so ordering-sensitivity
  cannot be ruled out for large-leaf regimes on this data alone.
- **Priority-queue ordering surface likely under-exercised.** With only
  8 leaves the queue makes few contested choices; any latent ordering
  sensitivity would be more visible at 64+ leaves.
- **No cross-check at the ST+Cosine cell geometry.** The ST anchor was
  measured on a different config; a direct head-to-head at matched
  `N/depth/iter` was not attempted in this spike.

Implication: confidence in outcome A is moderate, not high. The spike
rules out outcome B *at this cell* but cannot rule it out at deep/wide
LG configurations. A Kahan fix that addresses the shared joint-denominator
accumulator should nevertheless be applied first; if residual LG-only
drift persists after Kahan lands, re-open outcome B on a deep cell.

## Recommendation

Close S29 with **CLI-GUARD + this verdict**. Keep the LG+Cosine guard
(Python `_validate_params` and C++ `TrainConfigToInternal`) in place
pending the shared fix. Merge `S29-ST-COSINE-KAHAN` carry and the
`S29-LG-COSINE-RCA` follow-up into a single **S30-COSINE-KAHAN** task:
apply Kahan/Neumaier summation to the joint-Cosine denominator
accumulator once, gate both ST+Cosine and LG+Cosine behind the same
post-fix parity check, and remove both guards together. If a residual
LG-specific drift remains after the shared fix ships — particularly on
deep/wide LG cells not covered by this spike — re-open outcome B in
S31.
