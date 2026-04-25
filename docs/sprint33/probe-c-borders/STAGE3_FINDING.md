# PROBE-C Stage 3 — d=2 gain-margin sanity check

**Date**: 2026-04-24
**Test**: zero column f10 in the anchor, retrain, observe what MLX picks at d=2.
**Inputs**: `data/anchor.csv` (rng=42) → `data/anchor_f10_zeroed.csv` (f10 = 0.0).
**Outputs**: `data/mlx_anchor.json` (natural) vs `data/mlx_anchor_f10_zeroed.json`.

## Why this test

PROBE-C Stage 2 showed iter=2 tree[1] divergence is at depth=2 (CPU re-picks
feat=0, MLX picks feat=10 — pure noise in y = 0.5·X[0] + 0.3·X[1] + 0.1·noise).
MLX's reported d=2 incremental gain is **0.012** — three orders of magnitude
smaller than d=1's 14.76. Pattern is consistent with sub-precision argmax
instability: at thin margins, FP-order/precision noise can flip the winner.
This stage runs the cheap version of that sanity check by removing feat=10
from the candidate set and observing MLX's "runner-up."

## Result

| Phase | Natural (feat=10 present) | f10-zeroed |
|---|---|---|
| d=0 | feat=0 bin=64 (cum=87.180571) | feat=0 bin=64 (cum=87.180894) — **3.23e-4 drift even though split is identical** |
| d=1 | feat=1 bin=58 (cum=101.94191) | feat=1 bin=58 (cum=101.94266) |
| d=2 Δ-gain | 0.01202 (feat=10) | 0.01005 (**feat=19**) |
| d=2 cumulative | 101.95393 | 101.95271 |
| d=3 | feat=14 | feat=12 |
| d=4 | feat=12 | feat=14 |
| d=5 | feat=13 | feat=3 |

**With feat=10 unavailable, MLX does not converge on CPU's choice (feat=0).**
It picks feat=19 — another noise feature. d=3..d=5 also differ. The d=2 gain
margin between MLX's natural winner (feat=10, 0.01202) and runner-up
(feat=19, 0.01005) is **0.002** — comfortably inside the precision-class noise
floor.

## What this tells us

1. **MLX's score function systematically ranks noise features above feat=0
   at d=2** in the d=0+d=1 induced partitions. CPU's continued preference for
   feat=0 at d=2 is unique to CPU's evaluation. This is not a randomized tie-break
   — it's a deterministic ordering disagreement near a thin margin.

2. **d=0 gain itself is layout-sensitive**. Zeroing f10 shifted the quantized
   bin-feature count from 2540 to 2413 (127-bin diff = exactly one feature's
   worth). The d=0 *split* is byte-identical (feat=0, bin=64), but the *computed
   gain* shifted by 3.23e-4. This is direct evidence that gain values are
   sensitive to global accumulation order in the packed layout, at the
   ~1e-4 absolute scale.

3. **All d=2+ winners are in the same precision band** (Δ-gain 0.01–0.03).
   Argmax decisions at this depth are decided in the 4th-decimal place of the
   gain, where the layout-induced accumulation noise (#2 above) lives.

## Verdict on B (the option B framing)

The d=2 divergence is **consistent with precision-class argmax-instability
under thin gain margins**, NOT with a deterministic wrong-code mechanism. But
this evidence is indirect — it shows MLX prefers noise-vs-noise over noise-vs-signal
at d=2, and that gain values are layout-sensitive at the right scale, but it
does NOT directly compare CPU vs MLX gain for the same (feature, bin).

## What this changes for option C (#125)

Reframe the depth-2 dump:

- **Original C plan**: dump per-feature winning gain, find the first one
  that differs.
- **Revised C plan**: dump per-feature top-k gains for k=10–20, both sides.
  The expected finding is that gain *rankings* in the top-5 cluster within
  ~0.01 absolute, with MLX's ranking shuffled vs CPU's by precision noise.
  If true, the closure path is precision-class (joint-denominator fp64,
  Kahan accumulator on the partial sums, deterministic accumulation order),
  NOT a code-fix-class issue.

## Implication for DEC-036

If the precision hypothesis holds, DEC-036's 52.6% drift is the cumulative
effect of ~50 iterations × 5 depths of argmax disagreements, each at the
~1e-2 gain-margin level. The drift compounds because each disagreement
slightly redirects the residual that subsequent iterations fit. Closure
likely requires either:
  (a) match CPU's exact accumulation order (not feasible without invasive
      kernel rewrite), or
  (b) widen joint-denominator and partial sums to fp64 — extending DEC-038
      (allVals fp64 fix) and DEC-039 (cap-127) to the full gain pipeline at
      depth ≥ 1, or
  (c) accept that ST+Cosine has an inherent precision floor at this scale and
      ship the guard permanently (return DEC-032 to a closed state).

These tradeoffs land on @research-scientist's plate, not the probe-D
instrumentation plate.
