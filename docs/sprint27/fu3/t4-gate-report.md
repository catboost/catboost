# S27-FU-3-T4 Gate Report: G3-FU3 Evidence

**Branch**: `mlx/sprint-27-correctness-closeout`
**Date**: 2026-04-22
**Authored by**: ML Engineer (@ml-engineer)
**Commit**: `fc44bfc936`

---

## Gate: G3-FU3

**Definition**: DW N=1000 ratios ∈ [0.98, 1.02] with `score_function='L2'` on both
sides (CPU explicit, MLX hardcoded).

**Path covered**: `FindBestSplitPerPartition` gain-scope equivalence **conditional on
L2 score function matching**. Unconditional algorithm parity (Cosine port) is S28 scope.

---

## Method

The 5 failing cells from the S27-FU-3-T1 triage (seeds 1337, 42, 7 × rs ∈ {0.0, 1.0},
minus seed=1337 rs=1.0 which already passed the segmented gate) were re-evaluated with
`score_function='L2'` explicitly set on the CPU CatBoostRegressor side.

MLX RMSE values are from the T1 step1 sweep (`step1_step2_results.json`, commit `0931ad6e9c`).
CPU L2 RMSE values were computed fresh in this task run.

**Gate criterion**: `ratio = MLX_RMSE / CPU_L2_RMSE ∈ [0.98, 1.02]` (symmetric, applies
to both rs=0 and rs=1 cells, because L2-on-L2 matching eliminates the PRNG-divergence
argument used in the segmented gate).

---

## Per-Cell Results

| seed | rs  | MLX RMSE   | CPU RMSE (L2) | ratio  | verdict |
|------|-----|-----------|--------------|--------|---------|
| 1337 | 0.0 | 0.179724  | 0.179521     | 1.0011 | PASS    |
| 42   | 0.0 | 0.181591  | 0.181673     | 0.9995 | PASS    |
| 42   | 1.0 | 0.198501  | 0.199380     | 0.9956 | PASS    |
| 7    | 0.0 | 0.179449  | 0.179432     | 1.0001 | PASS    |
| 7    | 1.0 | 0.195260  | 0.195392     | 0.9993 | PASS    |

---

## Overall Gate Verdict

**5/5 PASS**

Ratio distribution: [0.9956, 0.9993, 0.9995, 1.0001, 1.0011]

All ratios within [0.98, 1.02]. Maximum deviation: 0.44% (seed=42 rs=1.0 at 0.9956).

Prior to this fix (CPU using Cosine default), the same 5 cells produced ratios in
[0.8232, 0.8620] — a 14–17% systematic gap driven by the score function mismatch,
not a training correctness bug.

---

## Reference: Pre-Fix Ratios (CPU Cosine, from T1 triage)

| seed | rs  | ratio (CPU Cosine) | pred_std_R | T1 verdict |
|------|-----|-------------------|------------|------------|
| 1337 | 0.0 | 0.8315            | 1.1004     | FAIL       |
| 42   | 0.0 | 0.8619            | 1.0759     | FAIL       |
| 42   | 1.0 | 0.8232            | 1.1028     | FAIL       |
| 7    | 0.0 | 0.8620            | 1.0832     | FAIL       |
| 7    | 1.0 | 0.8276            | 1.1011     | FAIL       |

---

## Footnote

Gate G3-FU3 measures parity **conditional on matching score functions** (both sides
use L2 Newton gain: CPU explicit via `score_function='L2'`, MLX hardcoded in
`FindBestSplitPerPartition`). Unconditional DW parity — meaning MLX correctly
implements Cosine normalization and produces results matching CPU's default — is
**S28 scope**. See DEC-032 for full rationale.

The `score_function='L2'` parameter is added to the CPU side only. The MLX side
is not modified: adding the parameter would falsely imply it is honored at the
`CatBoostMLXRegressor` API level, which S28-AUDIT has not confirmed.
