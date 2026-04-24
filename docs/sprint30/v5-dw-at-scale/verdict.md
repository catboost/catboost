# S30-V5-DW-AT-SCALE — Verdict

**Top-line verdict: MIXED**

DW+Cosine at N=50000 produces mean drift of **6.33%** — above the "contained" threshold
(5%) but far below the "catastrophic" threshold (20%). L0 cancellation error does scale
with N in the DW path, confirming that N is a lever. However, DW's N-scaled drift (6.33%)
is 8.4x smaller than ST's drift (53.30%) at the same N. The L0 hypothesis accounts for
part of ST's failure but cannot be its sole driver.

---

## Per-seed drift table

| seed | MLX_RMSE   | CPU_RMSE   | ratio    | drift_pct |
|------|------------|------------|----------|-----------|
| 42   | 0.18017600 | 0.16962773 | 1.062185 | 6.2185%   |
| 43   | 0.18069500 | 0.16959735 | 1.065435 | 6.5435%   |
| 44   | 0.18010900 | 0.16955519 | 1.062244 | 6.2244%   |
| **mean** | — | — | 1.063288 | **6.3288%** |

Binary: `csv_train_t3` (K4 fp64 active; DW+Cosine unguarded).
CPU: catboost pip, `grow_policy='Depthwise'`, `score_function='Cosine'`,
`random_strength=0.0`, `bootstrap_type='No'`, `l2_leaf_reg=3.0`, `thread_count=1`.

---

## Cross-measurement comparison

| Cell | grow_policy | N | mean drift | source |
|------|------------|---|-----------|--------|
| DW@N=1000 (S28 baseline) | Depthwise | 1,000 | 0.94% | S28 G5a (t5-gate-report.md) — 5 seeds, ratios [0.9950, 1.0160] |
| DW@N=50000 (this run) | Depthwise | 50,000 | **6.33%** | S30-V5 (this document) — 3 seeds |
| ST@N=50000 (T3 G3a) | SymmetricTree | 50,000 | 53.30% | S30-T3 (t3-measure/verdict.md) — 5 seeds |

N-scaling in the DW path: 1000 → 50000 (50x), drift 0.94% → 6.33% (+5.39 pp, ~6.7x).
Grow-policy gap at N=50000: DW 6.33% vs ST 53.30% — an 8.4x ratio in drift magnitude.

---

## Interpretation: what this tells us about the L0 hypothesis

The L0-dominant hypothesis predicted that DW's S28 pass was an N-scale artifact — that
at N=50000 DW would also break catastrophically (>= 20%), matching ST's ~53% failure and
attributing the entire drift difference to DW running at smaller N.

This prediction is **partially confirmed and partially falsified**:

**Confirmed**: N does matter for DW. Scaling from 1k to 50k increases drift 6.7x (0.94%
to 6.33%), consistent with L0 cancellation error accumulating proportionally to N (more
data -> more partial sums -> more float32 rounding). L0 is real and scales with N in the
DW path.

**Falsified**: DW at N=50k is still 8.4x better than ST at N=50k (6.33% vs 53.30%). If
L0 were the sole driver, both grow policies should exhibit roughly the same per-bin
quantization error and produce similar compounding drift at equal N. The 8.4x gap
requires a grow-policy-specific explanation.

**Most likely mechanism**: The DW and ST Cosine score formulas differ in how the
denominator is aggregated. In DW's `FindBestSplitPerPartition`, the Cosine denominator
is computed **per leaf** (`cosDen_d` reset for each partition `p`), so rounding errors
stay local to one leaf. In ST's `FindBestSplit`, the denominator accumulates **across all
2^d = 64 partitions jointly** (`Σ_p cosDen_p`), amplifying rounding errors 64x before
taking the `sqrt`. This structural difference was already flagged in the S28-OBLIV-DISPATCH
REFLECT ("denominator `sqrt(Σden)` accumulates across 2^d = 64 partitions at depth=6,
amplifying rounding errors in the denominator"). The present measurement provides
quantitative confirmation: per-leaf accumulation (DW) produces ~6% drift; joint
accumulation across 64 partitions (ST) produces ~53% drift at the same N.

**K4's role**: K4 replaced float32 accumulators with fp64 in the gain formula (`cosNum_d`,
`cosDen_d`). T3 showed this reduced the iter=1 gain residual from 4.75e-5 to 3.81e-6 (12.5x)
but left 53% final-iter drift intact. The DW measurement confirms K4's fp64 fix is active
(same binary) — DW's 6.33% drift is the K4-fixed DW floor at N=50k, not a pre-K4 number.
The remaining DW drift after K4 is most likely the float32 histogram input quantization
floor (already identified in T3 verdict): histograms are stored as float32 before entering
the fp64 gain computation.

---

## Recommendation for S31

**Do NOT narrow S31 to a pure L0 histogram-float32 fix.** The 8.4x DW/ST gap points to
the ST joint-denominator aggregation as a major independent source of drift. A plan that
fixes only L0 (e.g., fp64 histograms) while leaving the joint-denominator structure intact
will likely bring ST drift from 53% down to something comparable to DW's 6.33% — still
well above the 2% gate.

**Recommended S31 scope (two-track)**:

1. **ST denominator isolation** (higher priority): Investigate and fix the joint
   `sqrt(Σ_p cosDen_p)` accumulation across partitions in `FindBestSplit`. One direct
   approach: compute the Cosine score per-partition and sum scores (not sum denominators),
   matching DW's per-leaf pattern. This is an algorithmic change and must be validated
   against CPU CatBoost's ST Cosine reference to ensure the formula matches CPU semantics
   before measuring drift reduction.

2. **L0 histogram precision** (secondary): At DW's N=50k floor of 6.33%, even after
   fixing the ST denominator, a residual ~6% DW drift remains. If the target gate is <2%
   for both ST and DW, the histogram float32 floor must also be addressed (e.g., fp64
   accumulation in the histogram kernel, or a compensated-summation pass). However, this
   is only needed if DW's 6.33% floor is confirmed post-ST-fix as the binding constraint.

**Decision gate for S31**: After the ST denominator fix, re-run G3a (ST@N=50k) AND a DW
re-baseline at N=50k. If ST comes down to the DW floor (~6%) and DW also remains ~6%,
both grow policies share the same root cause (L0 histogram floor) and a single fix
resolves both. If ST comes down further than DW, additional ST-specific interactions remain.

---

## Data artifacts

All raw files under `docs/sprint30/v5-dw-at-scale/data/`:

| File | Contents |
|------|----------|
| `dw_at_scale_seed42.csv` | seed=42 single-row RMSE + drift |
| `dw_at_scale_seed43.csv` | seed=43 single-row RMSE + drift |
| `dw_at_scale_seed44.csv` | seed=44 single-row RMSE + drift |
| `dw_at_scale_summary.csv` | All 3 seeds aggregated |

Runner script: `docs/sprint30/v5-dw-at-scale/run_dw_at_scale.py`
