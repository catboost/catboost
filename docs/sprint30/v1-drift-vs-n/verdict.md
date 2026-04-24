# S30-V1-DRIFT-VS-N — Verdict

## Verdict

**L0 histogram hypothesis FALSIFIED — drift is flat across N (b = 0.0017, all N within 1.5% of 53%); the dominant drift mechanism is not summand-count-driven fp32 atomic accumulation.**

---

## Per-N Drift Table

Config: loss=RMSE, grow_policy=SymmetricTree, score_function=Cosine, depth=6, bins=128,
        iterations=50. Binary: csv_train_t3 (K4 fp64 Kahan fix active).

| N | seed=42 | seed=43 | seed=44 | mean |
|---|---------|---------|---------|------|
| 1 000 | 52.2866% | 54.4482% | 51.2466% | **52.66%** |
| 5 000 | 53.2701% | 52.6196% | 53.9831% | **53.29%** |
| 10 000 | 53.8564% | 52.6840% | 53.7398% | **53.43%** |
| 25 000 | 53.3933% | 52.7534% | 53.2572% | **53.13%** |
| 50 000 | 53.1790% | 52.8823% | 53.0305% | **53.03%** |

Grand mean across all cells: 53.11%. Max relative deviation from grand mean: 0.89%.

---

## Log-Log Scaling Fit

Fit: `drift_pct = a * N^b` (OLS on log-log)

| Parameter | Value |
|-----------|-------|
| a | 52.31 |
| b | **0.0017** |

A b-value of 0.0017 is indistinguishable from zero (compare to the b > 0.3 threshold for
hypothesis support). Drift is flat to within measurement noise across a 50x range of N.

---

## Plain-Language Interpretation

The L0 hypothesis predicted that drift should grow with N because more documents per bin means
more summands in the fp32 atomic accumulation, which should produce more cancellation error in
the histogram gradient sums. The data show the opposite: drift is a rock-steady ~53% from
N=1000 to N=50000. The scaling exponent b = 0.0017 is effectively zero. This eliminates the
"more summands → more cancellation → wrong split" mechanism as the dominant source of the
observed 53% RMSE divergence. The drift pattern is more consistent with a systematic, N-independent
error — such as a fixed algorithmic difference in the Cosine gain formula, a split-tie-breaking
policy discrepancy, or a constant per-tree bias that compounds identically regardless of dataset
size. The identical ~53% ratio across five N-values spanning 50x also confirms that the
divergence is not a noise-floor artifact and that both MLX and CPU converge to their own
consistent solutions — they just converge to different solutions.

---

## Recommendation for Next Step

The flat-drift profile rules out the entire class of precision fixes that reduce histogram
accumulation error (K4 and its successors). The problem is not that MLX accumulates histograms
imprecisely — it is that MLX and CPU make systematically different split decisions that are
independent of N. The correct next diagnostic is a **split-selection audit at iter=1**:

1. Instrument `csv_train_t3` to dump the winning (feature_idx, split_bin) for each of the 6
   tree layers at iter=1, for a single seed.
2. Run the same dump against CatBoost CPU with a debug/custom build that logs split decisions.
3. Identify the first layer where MLX and CPU diverge (if any).
4. If they diverge at layer 0 (root split), the Cosine gain formula itself produces a different
   ranking — the bug is in the formula, not the precision. If they agree at all layers, the
   divergence is in leaf value estimation (`CalcLeafValues`), not tree structure.

This audit is the correct S31 first step for ST+Cosine. It avoids committing another sprint to
precision micro-optimizations that cannot close the gap.

---

## Data Artifacts

| File | Contents |
|------|----------|
| `data/v1_drift_vs_n.csv` | 15 rows (N × seed): mlx_rmse, cpu_rmse, drift_pct, wall_secs |

Runner: `docs/sprint30/v1-drift-vs-n/run_v1_drift_vs_n.py`

Raw N=50000 drift values (seeds 42/43/44: 53.18%/52.88%/53.03%) are consistent with
T3/G3a seeds 42/43/44 (53.18%/52.88%/53.03%) — confirming the harness is correctly
reproducing the T3 cell at N=50000.
