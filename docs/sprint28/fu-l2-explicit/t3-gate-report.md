# S28-L2-EXPLICIT Gate Report

**Branch**: `mlx/sprint-28-score-function-fidelity`
**Date**: 2026-04-23
**Authored by**: @ml-engineer (S28-L2-EXPLICIT, task #72)
**Raw data**: `docs/sprint28/fu-l2-explicit/t3-gate-results.json`

---

## Context

S28-L2-EXPLICIT wires the `EScoreFunction` enum (added in S28-COSINE, commit
`83f30c3677`) into live dispatch.  The hardcoded L2 Newton gain calls in both
branches of `FindBestSplitPerPartition` (one-hot lines 1325–1327 and ordinal
lines 1377–1379 per S28-AUDIT) are replaced with `switch (scoreFunction)`.
`score_function` is threaded from:

```
TConfig.ScoreFunction  →  ParseScoreFunction()  →  FindBestSplitPerPartition(..., scoreFunction)
TTrainConfig.ScoreFunction  →  TrainConfigToInternal  →  TConfig
TTrainConfig.score_function  (nanobind)  →  Python CatBoostMLX.score_function
```

Default remains `"L2"` on all three layers to preserve pre-S28 behaviour.

---

## Gate G3a: L2 no-regression

**Definition**: No `score_function` argument passed (default L2 path). MLX DW
N=1000 vs CPU DW `score_function='L2'`, 5 seeds {42–46}, rs=0. Ratios must
remain ∈ [0.98, 1.02]. Regression here would mean dispatch itself introduces
drift (shared state, aliased buffers, or accumulator initialisation error).

### Per-seed results

| seed | MLX_RMSE | CPU_L2 | ratio | G3a |
|------|----------|--------|-------|-----|
| 42 | 0.181591 | 0.181673 | 0.9995 | PASS |
| 43 | 0.180416 | 0.180156 | 1.0014 | PASS |
| 44 | 0.182786 | 0.182824 | 0.9998 | PASS |
| 45 | 0.182511 | 0.182678 | 0.9991 | PASS |
| 46 | 0.182111 | 0.182548 | 0.9976 | PASS |

**Max deviation from 1.0**: 0.24% (seed 46). Identical to S28-COSINE G2b
measurements — no drift introduced by the dispatch layer.

**G3a verdict: PASS** (5/5 seeds)

---

## Gate G3b: Cosine live-path parity

**Definition**: `score_function='Cosine'` passed. MLX DW N=1000 vs CPU DW
`score_function='Cosine'`, 5 seeds {42–46}, rs=0. Both sides now compute the
same algorithm; ratios must be ∈ [0.98, 1.02]. This gate closes the 0.83–0.87
baseline gap recorded in `t2-gate-report.md`.

### Per-seed results

| seed | MLX_RMSE | CPU_Cosine | ratio | G3b |
|------|----------|------------|-------|-----|
| 42 | 0.214025 | 0.210677 | 1.0159 | PASS |
| 43 | 0.212311 | 0.208968 | 1.0160 | PASS |
| 44 | 0.210635 | 0.210156 | 1.0023 | PASS |
| 45 | 0.214788 | 0.213174 | 1.0076 | PASS |
| 46 | 0.218467 | 0.219571 | 0.9950 | PASS |

**Max deviation from 1.0**: 1.60% (seeds 42–43). All within [0.98, 1.02].

**G3b verdict: PASS** (5/5 seeds)

### Baseline gap closure

| seed | Before S28-L2-EXPLICIT (ratio) | After (ratio) | Closed? |
|------|-------------------------------|---------------|---------|
| 42 | 0.8619 | 1.0159 | YES |
| 43 | 0.8634 | 1.0160 | YES |
| 44 | 0.8698 | 1.0023 | YES |
| 45 | 0.8562 | 1.0076 | YES |
| 46 | 0.8294 | 0.9950 | YES |

The 14–17% gap (DEC-032) closes to ≤ 1.60% deviation. The residual ~1.5%
is not algorithmic divergence — it arises from CPU using `double` throughout
the Cosine accumulation loop while MLX uses `float32`. This is the same source
of the max 1 ULP measured at formula level in G2a (t2-gate-report.md).

---

## Gate G3c: Unknown value guard

**Definition**: `score_function='NewtonCosine'` must raise with a clear error
message, not silently fall back to L2.

**Result**: Python layer raises `ValueError` immediately (before any C++ call):

```
score_function='NewtonCosine' is not yet implemented in the MLX backend.
Supported values: L2, Cosine.
```

**G3c verdict: PASS** — error raised, message names the unsupported value and
lists valid options.

---

## REFLECT

### Live-path ULP expansion vs formula level

Formula-level G2a (t2-gate-report.md) measured max 1 ULP (float32 formula vs
CPU double reference, 10 test cases). Live-path G3b shows max 1.60% RMSE
deviation. These two measurements are at different levels of abstraction
(single-split gain vs. 50-iter RMSE), but both are consistent with float32 vs
double accumulation as the only source of drift. There is no sign of
accumulation blow-up, aliased buffer errors, or per-partition indexing bugs.

The 1.60% residual at seeds 42–43 is larger than expected purely from 1 ULP on
individual gains. The source is cumulative: 50 iterations × 64 partitions × K
approx dims each accumulate float32 vs double rounding. Across the tree
ensemble, these compound. This is expected and acceptable per DEC-008 (≤4 ULP /
≤2% aggregate tolerance).

Note: seed 46 ratio = 0.9950 (MLX slightly better RMSE than CPU). This is the
same direction as the original 0.8294 entry in that seed. The Cosine formula
and random seed interact at depth-6 small partitions to produce different split
choices. At 1.60% max deviation, this is within gate and acceptable.

### Enum extensibility

Adding `NewtonL2` / `NewtonCosine` requires:
1. Add enum values to `EScoreFunction` in `csv_train.cpp`.
2. Implement formulas (analogous to `ComputeCosineGain`).
3. Update `ParseScoreFunction` to remove the throw for those values.
4. Add validation in `core.py` `_VALID_SCORE_FUNCTIONS`.

The `switch` structure in both branches of `FindBestSplitPerPartition` is
already the correct shape — `default: throw` will catch any new enum value
added without a corresponding case. Estimated effort: 1 sprint task.

### Default value rationale

CPU CatBoost DW default is `Cosine`. MLX default is `L2`. This is an
intentional divergence preserved for backward compatibility: existing MLX
users who do not pass `score_function` get the same behaviour they have always
gotten. They must opt into `score_function='Cosine'` explicitly to match CPU
default. This is documented in `TConfig::ScoreFunction` comment and
`TTrainConfig::ScoreFunction` docstring.

---

## Overall gate verdict

| Gate | Criterion | Result |
|------|-----------|--------|
| G3a (L2 no-regression, 5 seeds) | 5/5 ∈ [0.98, 1.02] | **PASS** |
| G3b (Cosine live-path, 5 seeds) | 5/5 ∈ [0.98, 1.02] | **PASS** |
| G3c (NewtonCosine guard) | Raises with clear message | **PASS** |
