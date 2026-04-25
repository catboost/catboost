# S33-L3-ITER2 Verdict

**Date**: 2026-04-24
**Branch**: mlx/sprint-33-iter2-scaffold
**Tip at run**: 785ce28bac (instrumentation committed separately)
**Anchor config**: N=50000, depth=6, bins=127, seed=42, SymmetricTree/Cosine/RMSE, rs=0.0
**Kernel md5**: 9edaef45b99b9db3e2717da93800e76f (verified pre- and post-build)

---

## Graft Mechanism

Same mechanism as L2-GRAFT (Option b, snapshot resume).
CPU iter=1 cursor (50000 float32) injected via `graft_snapshot_seed42.json`.
MLX binary resumes from `iteration=0, num_trees=1`, reusing the CPU-produced cursor
as its starting state for iter=2. This ensures S1-GRADIENT comparison is
conditioned on a bit-identical cursor between CPU and MLX.

---

## Per-Stage Diff Table

| Stage         | Status     | max_abs_diff   | max_rel_diff   | frac_diverging |
|---------------|------------|----------------|----------------|----------------|
| S1-GRADIENT   | CLEAN      | 0.000e+00      | 0.000e+00      | 0.0000%        |
| S2-SPLIT      | DIVERGENT  | —              | —              | —              |
| S3-LEAF       | DIVERGENT  | 2.548e-02      | 2.548e+10      | 100.0000%      |
| S4-APPROX     | DIVERGENT  | 2.185e-02      | 1.008e+01      | 100.0000%      |

S1 note: both `max_abs_diff` and `max_rel_diff` are exactly 0.000e+00; the gradient
arrays are bit-identical (50000 float32, confirmed by element-wise subtraction).
This is expected: RMSE gradient = cursor - target, and the grafted cursor is
bit-for-bit the CPU iter=1 output.

S2 note: the divergence is categorical — different split bin, not a numeric threshold
comparison. There is no meaningful max_rel_diff to report for a discrete (feat, bin)
selection.

S3/S4 note: these diverge entirely as a cascade from the wrong S2 split. The leaf
values and per-doc approximations are computed from the wrong partition boundary
(bin=64 vs bin=3), so 100% of elements differ.

---

## S2 Split Detail

| Side | feat | bin | gain          |
|------|------|-----|---------------|
| CPU  | 0    | 3   | (not dumped)  |
| MLX  | 0    | 64  | 87.059881     |

Both sides select feature 0 as the best feature. The divergence is **within-feature
bin selection**: CPU picks bin=3, MLX picks bin=64. This is the direct consequence
of wrong per-bin gradient values in the MLX histogram (see Histogram Anomaly below).

---

## Histogram Anomaly (evidence for L4)

The MLX raw histogram dump (`mlx_hist_d0_iter2.bin`, 5080 float32 = 2 blocks of 2540)
shows a structural inconsistency with the S1 gradient dump:

| Quantity                          | Observed         | Expected              |
|-----------------------------------|------------------|-----------------------|
| S1 gradient total (50000 docs)    | +0.011414        | +0.011414 (correct)   |
| Histogram grad block total        | -738.9923        | ~+0.228 (20 feat * sum_g) |
| Histogram hess block total        | 992200           | ~1000000 (20 feat * N) |
| Mean hess per bin (2540 bins)     | 390.6 docs/bin   | 19.7 docs/bin          |

The hessian total is consistent with the full-depth histogram: for RMSE h_i=1.0,
summing across all bins of all 20 features gives N_features * N_docs = 20 * 50000 =
1,000,000. The observed 992200 is plausible (boundary docs unassigned in ~7800 cases).

The gradient total of -738.99 is NOT consistent with iter=2 gradients (sum=0.011).
The magnitude and sign pattern (bin[0]=+429, bin[1]=+402, ...) match a state where
the gradient per doc correlates strongly with the feature value — i.e. the raw
residuals before any tree was applied, or after a single large-magnitude tree.
The iteration-1 (0-indexed) gradients, if the cursor was near the initial basePred
and the target is strongly correlated with X[:,0], would produce exactly this kind
of monotone per-bin gradient pattern.

**Hypothesis for L4**: The histogram Metal kernel receives a `statsK` array whose
underlying GPU buffer contains stale gradient data from a prior iteration. Although
`dimGrads[0]` is re-assigned fresh each iteration (line 4010) and `mx::eval()` is
called at line 4113/4229 before the histogram call, the MLX lazy evaluation graph may
be reconstructing `statsK` from a compute graph node that resolves against an old
buffer. The `mx::concatenate` at line 4534 references `dimGrads[k]` — if the
array handle is reused in-place across the graph boundary, the Metal kernel may
receive the prior iteration's materialized buffer.

**Suspected location**: `csv_train.cpp:4534` — `statsK` construction inside the
SymmetricTree histogram build loop (`for (ui32 k = 0; k < approxDim; ++k)`), and the
`DispatchHistogram` call at line 4540. The specific mechanism (lazy-graph alias,
missing `mx::eval()` before reshape/concatenate, or buffer reuse in `mx::fast::metal_kernel`)
is for L4 to confirm.

---

## Class Call

**SPLIT**

The first divergent stage at iter=2 is S2 (histogram build + best-split selection).
Gradient computation (S1) is clean given an identical cursor, confirming the
divergence is introduced in the histogram kernel path, not the loss function.

---

## File:Line Pointer

Primary suspect: `catboost/mlx/tests/csv_train.cpp:4534`
(`statsK` construction for SymmetricTree depth=0 histogram dispatch)

Secondary: `catboost/mlx/tests/csv_train.cpp:4540`
(`DispatchHistogram` call — `stats` argument receives `statsK`)

Both are within the SymmetricTree branch of Step 2 (Greedy tree structure search),
inside the outer depth loop at iter=2.

---

## Decision for L4

DEC-012 atomicity: L3 stops at the first divergent stage (SPLIT). L4 targets the
histogram data integrity issue — specifically whether `statsK` passed to
`DispatchHistogram` at iter>=2 carries the correct iter=2 gradients or stale data
from a prior iteration's compute graph.

L4 fix scope: one structural change — ensure `dimGrads[k]` is fully materialized
(or graph-isolated) before being embedded in the `statsK` concatenate that feeds
the histogram Metal kernel.

**L4 entry point**: task #123 L4-FIX.

---

## Reproducibility

Raw dumps in `docs/sprint33/l3-iter2/data/`.
Reproducible diff: `python docs/sprint33/l3-iter2/scripts/diff_l3.py`
Binary build: `docs/sprint33/l3-iter2/scripts/build_l3.sh`
Full experiment: `python docs/sprint33/l3-iter2/run_l3_iter2.py`
