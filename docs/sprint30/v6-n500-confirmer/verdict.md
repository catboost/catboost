# S30-V6-N500-CONFIRMER — Verdict

**Verdict: FALSIFIED**

The L1 fp32 histogram hypothesis is **not** the dominant mechanism for the 53% G3a drift.
G3a FAILS at N=500 with aggregate drift of **50.72%**, compared to 53.30% at N=50000.
The actual reduction factor is **1.1x**; the L1-linear-in-N prediction required **100x**.

---

## Baseline Table (N=50000, from Fix 2 verdict)

Config: loss=RMSE, grow_policy=SymmetricTree, score_function=Cosine,
        depth=6, bins=128, iterations=50.
Binary: csv_train_t3 (K4 fp64 Kahan fix + Fix2 fp64 gain widening, both active).

### ST+Cosine at N=50000 (T3/Fix2 baseline, seeds 42–46)

| seed | MLX_RMSE   | CPU_RMSE   | ratio    | drift_pct |
|------|------------|------------|----------|-----------|
| 42   | 0.29659500 | 0.19362645 | 1.531790 | 53.18%    |
| 43   | 0.29593600 | 0.19357118 | 1.528823 | 52.88%    |
| 44   | 0.29566200 | 0.19320460 | 1.530305 | 53.03%    |
| 45   | 0.29550900 | 0.19250458 | 1.535075 | 53.51%    |
| 46   | 0.29712900 | 0.19305704 | 1.539074 | 53.91%    |

**Aggregate drift: 53.30%** — G3a FAIL (threshold < 2.0%).

### DW+Cosine at N=50000 (V5 baseline, seeds 42–44)

| seed | MLX_RMSE   | CPU_RMSE   | ratio    | drift_pct |
|------|------------|------------|----------|-----------|
| 42   | 0.18017600 | 0.16962773 | 1.062185 | 6.22%     |
| 43   | 0.18069500 | 0.16959735 | 1.065435 | 6.54%     |
| 44   | 0.18010900 | 0.16955519 | 1.062244 | 6.22%     |

**Mean drift: 6.33%** — fails DW parity threshold but far below ST.

---

## V6 Measurement Table

### ST+Cosine at N=500 (V6 primary cell, seeds 42–46)

| seed | MLX_RMSE   | CPU_RMSE   | ratio    | drift_pct |
|------|------------|------------|----------|-----------|
| 42   | 0.31550900 | 0.20830069 | 1.514681 | 51.47%    |
| 43   | 0.31359600 | 0.20694834 | 1.515335 | 51.53%    |
| 44   | 0.31825000 | 0.21071684 | 1.510321 | 51.03%    |
| 45   | 0.33126800 | 0.22215999 | 1.491124 | 49.11%    |
| 46   | 0.31487800 | 0.20928458 | 1.504545 | 50.45%    |

**Aggregate drift: 50.72%** — G3a FAIL (threshold < 2.0%).

### ST+Cosine at N=5000 (V6 sanity-check cell, seeds 42–46)

| seed | MLX_RMSE   | CPU_RMSE   | ratio    | drift_pct |
|------|------------|------------|----------|-----------|
| 42   | 0.30231400 | 0.19724270 | 1.532701 | 53.27%    |
| 43   | 0.29880800 | 0.19578614 | 1.526196 | 52.62%    |
| 44   | 0.29942800 | 0.19445515 | 1.539831 | 53.98%    |
| 45   | 0.30276900 | 0.19678837 | 1.538551 | 53.86%    |
| 46   | 0.30256400 | 0.19445347 | 1.555971 | 55.60%    |

**Aggregate drift: 53.86%** — G3a FAIL.

---

## Scaling Observation

| N      | mean drift | source                             |
|--------|------------|-------------------------------------|
| 500    | 50.72%     | V6 this run (seeds 42–46, 5 seeds)  |
| 1,000  | 52.66%     | V1-DRIFT-VS-N (seeds 42–44, 3 seeds)|
| 5,000  | 53.86%     | V6 this run (seeds 42–46, 5 seeds)  |
| 5,000  | 53.29%     | V1-DRIFT-VS-N (seeds 42–44, 3 seeds)|
| 10,000 | 53.43%     | V1-DRIFT-VS-N                       |
| 25,000 | 53.13%     | V1-DRIFT-VS-N                       |
| 50,000 | 53.30%     | T3/Fix2 (seeds 42–46, 5 seeds)      |

Range across 100x N span (500 → 50000): 50.72% to 53.86%. Maximum deviation from the
grand mean of 52.86%: **2.1%** relative. The N=5000 cross-check (V6 53.86% vs V1 53.29%
for the same N, different seed subsets) confirms within-method reproducibility.

**L1 prediction check:**

The L1 hypothesis predicts drift ∝ N (per-bin error scales linearly with number of
summands). Moving from N=50000 to N=500 should reduce drift by 100x, collapsing ~53%
to ~0.5%. The measured outcome:

| Metric | L1 prediction | Measured |
|--------|--------------|----------|
| Expected drift at N=500 | ~0.5% (PASS) | 50.72% (FAIL) |
| Reduction factor vs N=50k | 100x | 1.1x |
| Scaling exponent b (drift ~ N^b) | b ≈ 1.0 | b ≈ 0.0 (see V1) |

The reduction factor of 1.1x against an expected 100x eliminates L1 fp32 histogram
accumulation as the dominant mechanism, with high confidence.

---

## G3b, G3c, G4, G5 at N=500

These gates were not re-run at N=500. The task spec required them only "for ST+Cosine".
The reasoning is that the primary confirmation question is whether ST+Cosine G3a passes at
N=500. Since G3a FAILS at 50.72% (a FALSIFIED verdict), running G3b/G3c at N=500 would
not change the S31 scope recommendation.

For the record: G3b (LG-Mid, N=1000) and G3c (LG-Stress, N=2000) were already at small-N
configs in T3 and failed at 27–31% and 44–45% respectively. Those cells are not N-scaled
down from N=50k — they are already at low N. The flat V6/V1 result implies LG drift at
those N values is equally not L1-driven.

---

## DW+Cosine at N=500

Not run. V5 showed DW scales with N (0.94% at N=1000 → 6.33% at N=50000, b > 0).
DW's scaling behavior is structurally different from ST: DW computes the Cosine
denominator per-leaf (local accumulation), ST accumulates across all 2^d = 64
partitions jointly (global accumulation). V6's scope is ST+Cosine falsification only.

---

## Verdict: FALSIFIED

G3a FAILS at N=500 with aggregate drift 50.72%. Combined with V1's measurement at
N=1000 (52.66%) and the complete N=500–50000 curve showing b ≈ 0.0, the L1 hypothesis
is conclusively falsified:

- L1 (Metal fp32 histogram accumulation error ∝ N × ε_f32 / bins) is **not** the
  dominant driver of the 53% RMSE drift in ST+Cosine.
- The Fix 2 verdict (docs/sprint30/fix2-fp64-gain/verdict.md §7) cited L1 as the
  "binding constraint" based on the per-bin error magnitude estimate (4.7e-5 at N=50k).
  That estimate was arithmetically correct, but V6/V1 show the error does not manifest
  as split-selection flips proportional to N — the drift exists equally at N=500.
- The 53% drift is N-independent: it is a systematic, structural divergence between
  MLX and CPU split decisions, not a floating-point accumulation error.

---

## Root Cause Implication

The flat N-scaling points to an algorithmic discrepancy in one of:

1. **The Cosine gain formula itself** — the formula MLX computes may differ from
   CatBoost CPU's formula by a constant factor or structural difference (e.g., the
   joint-denominator `sqrt(Σ_p cosDen_p)` vs CPU's formula structure). V5 already
   identified that the ST denominator aggregates across 2^d = 64 partitions jointly
   while DW does so per-leaf; this is an algorithmic difference, not a precision
   difference, and would produce N-independent drift.

2. **Split tie-breaking** — when two candidates have very similar gains, the ordering
   rule (whether MLX and CPU use the same tie-breaking convention) would produce a fixed
   divergence regardless of N.

3. **Quantization boundary alignment** — the `max_bin=128` quantile-based border
   computation might differ between MLX's implementation and CPU CatBoost, causing
   systematically different bin assignments independent of N.

The V1 recommendation (split-selection audit at iter=1) remains the correct S31 first
step: dump the winning (feature_idx, split_bin) for each of the 6 tree layers at iter=1
from both MLX and CPU and compare. If they diverge at layer 0, the Cosine gain formula
or tie-breaking is the bug. If they agree, the divergence is in leaf value estimation.

---

## Recommended S31 Scope

**Primary task: iter=1 split-selection audit (ST+Cosine).**

1. Instrument `csv_train_t3` to print winning split (feature_idx, bin_idx, gain) for
   each layer of the first tree.
2. Run same config against CPU CatBoost with split logging (or a CPU-mode run of
   `csv_train_t3` with a forced-CPU histogram path to isolate the split search).
3. Identify the first layer where MLX and CPU diverge.
4. If divergence at layer 0: examine the Cosine gain formula and quantization borders.
   The joint-denominator `sqrt(Σ_p cosDen_p)` vs per-partition structure (V5 finding)
   is the primary suspect — verify that MLX's ST Cosine formula matches CPU CatBoost's
   ST Cosine formula exactly (not DW semantics applied to ST).
5. Do NOT invest in Metal kernel fp64 accumulation until the split-selection audit
   confirms a histogram-precision-dependent divergence (this is now very unlikely
   given the flat N-scaling).

**Secondary: LG+Cosine (K2 obligation, DEC-035).**

S31-LG-DEEP-RESIDUAL remains mandated by K2. The LG drift at max_leaves=31 (27–31%)
and max_leaves=64 (44–45%) is also N-independent in character — the V6 result
strengthens the case that LG's divergence is algorithmic rather than precision-driven.
The LG iter=1 split-audit should be run in parallel with the ST audit.

---

## Data Artifacts

| File | Contents | Rows |
|------|----------|------|
| `data/v6_n500.csv`  | N=500, seeds 42–46: mlx_rmse, cpu_rmse, ratio, drift_pct, wall_secs | 5 |
| `data/v6_n5k.csv`   | N=5000, seeds 42–46: same fields | 5 |

Runner: `docs/sprint30/v6-n500-confirmer/run_v6_n500_confirmer.py`

Binary: `csv_train_t3` (K4 fp64 Kahan fix + Fix2 gain widening both active).
Wall time per seed: ~0.4s (N=500), ~0.5s (N=5000).
Total sweep time: ~9 seconds (10 cells × ~0.9s each including CPU catboost run).
