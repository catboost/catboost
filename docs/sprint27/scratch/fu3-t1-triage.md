# S27-FU-3-T1 Triage: DW N=1000 Parity Asymmetry

**Branch**: `mlx/sprint-27-correctness-closeout`  
**Binary rebuilt**: `fb7eb59b5f` (post-FU-1 tip, csv_train rebuilt 2026-04-22 23:xx)  
**Date**: 2026-04-22  
**Authored by**: QA Engineer (@qa-engineer)

---

## Step 1 — Asymmetry still present post-FU-1?

**Result: ASYMMETRY PERSISTS. Not superseded.**

Re-run of 6 DW N=1000 cells (3 seeds × 2 rs) against the post-FU-1 binary:

| seed | rs  | CPU RMSE | MLX RMSE | ratio  | pred_std_R | seg  | vs pre-FU-1 ratio |
|------|-----|----------|----------|--------|------------|------|-------------------|
| 1337 | 0.0 | 0.216145 | 0.179724 | 0.8315 | 1.1004     | FAIL | delta = -0.0000   |
| 1337 | 1.0 | 0.237086 | 0.196611 | 0.8293 | 1.0959     | pass | delta = -0.0000   |
| 42   | 0.0 | 0.210677 | 0.181591 | 0.8619 | 1.0759     | FAIL | delta = +0.0000   |
| 42   | 1.0 | 0.241135 | 0.198501 | 0.8232 | 1.1028     | FAIL | delta = -0.0000   |
| 7    | 0.0 | 0.208184 | 0.179449 | 0.8620 | 1.0832     | FAIL | delta = -0.0000   |
| 7    | 1.0 | 0.235937 | 0.195260 | 0.8276 | 1.1011     | FAIL | delta = -0.0000   |

Ratios are bit-identical to the pre-FU-1 values recorded in
`benchmarks/sprint26/fu2/g1-results.md`. FU-1 fix (`ComputeLeafIndicesDepthwise`)
is orthogonal to `FindBestSplitPerPartition` — no interaction confirmed. 5 failing
cells remain (seed=1337 rs=1.0 passes segmented gate; pred_std_R=1.0959 < 1.10).

---

## Step 2 — ST control at N=1000

**Result: ST IS CLEAN (rs=0). Asymmetry IS DW-specific.**

SymmetricTree at N=1000 with matched config (same seeds, d=6, 128 bins, LR=0.03, 50 iters):

| seed | rs  | CPU RMSE | MLX RMSE | ratio  | pred_std_R | seg  |
|------|-----|----------|----------|--------|------------|------|
| 1337 | 0.0 | 0.201483 | 0.201873 | 1.0019 | 1.0036     | PASS |
| 1337 | 1.0 | 0.239023 | 0.203424 | 0.8511 | 1.0870     | PASS |
| 42   | 0.0 | 0.204238 | 0.204776 | 1.0026 | 1.0054     | PASS |
| 42   | 1.0 | 0.241850 | 0.206402 | 0.8534 | 1.0834     | PASS |
| 7    | 0.0 | 0.201687 | 0.202547 | 1.0043 | 1.0050     | PASS |
| 7    | 1.0 | 0.238456 | 0.204530 | 0.8577 | 1.0797     | PASS |

ST rs=0 ratios: 1.0019, 1.0026, 1.0043 — within the ±2% gate.
ST rs=1 ratios: 0.85–0.86 (MLX better by 14-16%) — this is PRNG realization divergence,
same pattern present at all N values, passes the segmented one-sided gate.

**Diagnostic conclusion**: Since ST rs=0 at N=1000 is tight (< 0.5% delta) and DW rs=0
shows 14-17% gap, the asymmetry is **DW-path-specific**. This rules out hypothesis (b)
as the primary driver. The mechanism must be in `FindBestSplitPerPartition` specifically.

---

## Step 3 — Score function root cause

**Result: MECHANISM CONFIRMED. CPU score_function='Cosine' ≠ MLX L2 Newton gain.**

### CPU source audit finding

`catboost/private/libs/algo/score_calcers.h` and `score_calcers.cpp` reveal two score
functions:
- **Cosine** (CPU default): `score = sum(leafApprox × gradSum) / sqrt(sum(leafApprox² × hessSum))`
- **L2**: `score = sum(leafApprox × gradSum)` (no normalization)

The Cosine denominator `sqrt(Σ leafApprox² × hessSum)` normalizes by effective partition
weight, acting as an implicit regularizer against tiny partitions.

CPU `get_all_params()` confirms: `score_function: Cosine` for both Depthwise and
SymmetricTree (verified on seed=42 rs=0 N=1000).

MLX `FindBestSplitPerPartition` implements the Newton L2 gain formula:
```
gain += sumLeft²/(weightLeft+λ) + sumRight²/(weightRight+λ) - totalSum²/(totalWeight+λ)
```
This is the **L2 score function**, NOT Cosine.

### Empirical confirmation

CPU DW at N=1000 forced to `score_function='L2'` reproduces MLX RMSE to within ULP noise:

| seed | CPU Cosine | CPU L2   | MLX      | MLX/CPU_L2 | MLX/CPU_Cosine |
|------|-----------|---------|----------|-----------|----------------|
| 1337 | 0.216145  | 0.179521 | 0.179724 | 1.0011    | 0.8315         |
| 42   | 0.210677  | 0.181673 | 0.181591 | 0.9995    | 0.8619         |
| 7    | 0.208184  | 0.179432 | 0.179449 | 1.0001    | 0.8620         |

`MLX/CPU_L2` ratios: 1.0011, 0.9995, 1.0001 — within ±0.11%, explaining the full
14-17% divergence from CPU Cosine as a score function choice, not a bug.

### Why DW is amplified vs ST

ST `Cosine vs L2` ratio at N=1000 rs=0:
- seed=1337: 1.0031, seed=42: 1.0046, seed=7: 1.0044

ST shows only 0.3–0.5% sensitivity to score function. DW shows 14-17%. The amplifier
is **partition fragmentation**: at depth=6 with N=1000, there are 64 partitions each
containing ~15 docs on average. Per-partition L2 gain can overfit a 15-doc micro-leaf by
picking a split that perfectly separates its gradient pattern. Cosine normalization
suppresses this by dividing by `sqrt(partition_hessian_magnitude)`. When ST evaluates
splits, it sums gains across all `2^d` partitions jointly — the per-partition variance
averages out, making both score functions converge to nearly identical rankings at the
tree level. DW's independent per-partition optimization has no such averaging.

---

## Tentative verdict: **(c) ACCEPTED — score function mismatch, not a training bug**

The 14-17% RMSE improvement for MLX DW at N=1000 is the natural consequence of using
L2 gain (more aggressive split selection) vs CPU's Cosine (normalized, regularized).
MLX is not wrong — it is implementing the correct Newton gain formula. CPU is not wrong
either — it uses Cosine as a deliberate choice to regularize non-oblivious trees.

**This is not a correctness bug in FindBestSplitPerPartition.** The function computes
L2 Newton gain correctly. The divergence from CPU stems from a score function choice
that was never documented as a deliberate decision in the MLX port.

**Why not (a) BUG**: No algorithmic error found. MLX/CPU_L2 = 1.0001 across all seeds.
**Why not (b) NOISE**: The mechanism is the score function, not gradient-RMS noise. It
is deterministic (rs=0 shows identical divergence), reproducible, and explained by a
concrete formula difference in the CPU source code.

### Two sub-options under (c)

The T3 decision is whether the accepted divergence should trigger:

**(c1) Scope restriction — gate narrows to N≥10k**
DEC-032 rationale: MLX uses L2 gain (Newton), which is a valid score function that
differs from CPU's Cosine. At N≥10k the two functions produce similar results (DW
N=10k shows ratio 0.992–0.995 rs=0, within ±1%). The N=1000 regime is not the
target deployment size for depth=6 Depthwise models. Gate scope: N≥10k only.

**(c2) Implement Cosine score function — close the fidelity gap**
DEC-032 rationale: implement `score_function=Cosine` in `FindBestSplitPerPartition`
to match CPU's default behavior. The Cosine formula is simple:
`score = sumLeft×avgLeft + sumRight×avgRight` where
`avg = sumGrad / (sumHess + λ)`. Or equivalently the normalized form above.
This would bring DW at all N values to ±2% parity with CPU.

**Recommendation for T3**: Start with **(c1)** — it is the minimal-risk change and
consistent with the S26-FU-2 FU-1 gate which already established N≥10k as the
viable range for DW (12/12 N≥10k cells PASS in FU-2). If Cosine fidelity is a
project priority, **(c2)** is a clean follow-up tracked as a separate DEC entry.

---

## Recommended T3 gate action

**Tighten the DW gate scope to N≥10k with DEC-032 rationale.**

DEC-032 should document:
1. MLX uses L2 Newton gain in `FindBestSplitPerPartition`.
2. CPU uses Cosine score function by default for Depthwise.
3. At N≥10k, both produce results within ±1% (empirically confirmed in FU-2).
4. At N=1000 with depth=6, per-partition fragmentation (~15 docs/partition) amplifies
   the score function difference to 14-17%. This is not a bug.
5. Decision: gate scope is N≥10k. If exact Cosine parity is needed, open a follow-up
   task to implement `score_function=Cosine` in `FindBestSplitPerPartition`.

Kill-switch: none fires. DW N=1000 is below the gate scope after DEC-032.

---

## Artifacts

- `docs/sprint27/scratch/fu3-t1-instrumentation/step1_step2_sweep.py` — Steps 1+2 script
- `docs/sprint27/scratch/fu3-t1-instrumentation/step1_step2_results.json` — Steps 1+2 data
- `docs/sprint27/scratch/fu3-t1-instrumentation/step3_score_function_analysis.py` — Step 3 script
- `docs/sprint27/scratch/fu3-t1-instrumentation/step3_score_function_results.json` — Step 3 data
