# S27-FU-1-T4 Gate Report: G1-FU1

**Date**: 2026-04-22
**Branch**: `mlx/sprint-27-correctness-closeout`
**Fix commit**: `fb7eb59b5f` (S27-FU-1-T3 — ComputeLeafIndicesDepthwise BFS fix)
**Harness**: `docs/sprint27/scratch/fu1_t4_gate.py`

## Overall Verdict

**G1-FU1: PASS** — 6/6 DW validation cells within [0.98, 1.02]

## Path Coverage

**What this gate covers**: `ComputeLeafIndicesDepthwise` called from the C++ training
loop when `valDocs > 0` (i.e., when an explicit `eval_set` is provided). This is the
specific function fixed by FU-1: it previously returned `nodeIdx - numNodes` (BFS-array
leaf offset, wrong encoding for Bug A) and indexed splits by BFS position into a
partition-ordered array (wrong split descriptor at depth >= 3, Bug B).

**What this gate does NOT cover**: histogram kernel, `FindBestSplitPerPartition`,
quantization / bin border logic, nanobind orchestration, leaf Newton step, or training
cursor updates (training path was already correct — Bug A/B were validation-path only).

## Gate Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| grow_policy | Depthwise | FU-1 scope; LG has its own ComputeLeafIndicesLossguide |
| depth | 4 | >= 2 hits Bug A; >= 3 hits Bug B; odd avoids symmetric shape artifact |
| N | 10k, 50k | Gate sizes per HANDOFF.md G1-FU1 spec |
| seeds | 0, 7, 42 | 3 seeds per spec |
| random_strength | 0.0 | rs=0 tight symmetric: no PRNG divergence to explain away |
| iters | 40 | Enough trees for val_loss to settle |
| val_fraction | 0.2 | 20% holdout; same split applied to CPU and MLX via eval_set |
| loss | RMSE | Simplest gate surface (regression) |
| metric | min(eval_loss_history) | Best val RMSE; mirrors CPU use_best_model semantics |

## 6-Cell Gate Results

| grow_policy | N | seed | rs | CPU best val RMSE | MLX best val RMSE | ratio | Verdict |
|-------------|---|------|----|-------------------|-------------------|-------|---------|
| Depthwise | 10,000 | 0 | 0.0 | 0.23464793 | 0.23435919 | 0.9988 | PASS |
| Depthwise | 10,000 | 7 | 0.0 | 0.23120663 | 0.23149855 | 1.0013 | PASS |
| Depthwise | 10,000 | 42 | 0.0 | 0.23301618 | 0.23363554 | 1.0027 | PASS |
| Depthwise | 50,000 | 0 | 0.0 | 0.23070309 | 0.23096776 | 1.0011 | PASS |
| Depthwise | 50,000 | 7 | 0.0 | 0.23250358 | 0.23244457 | 0.9997 | PASS |
| Depthwise | 50,000 | 42 | 0.0 | 0.22998090 | 0.22990301 | 0.9997 | PASS |

**Ratio distribution**: min=0.9988 / median=1.0004 / max=1.0027

**G1-FU1 gate criterion**: ratio in [0.98, 1.02] for all 6 cells.
**G1-FU1 verdict**: **PASS** (6/6)

## Non-Regression Checks

### SymmetricTree smoke test (1 cell: N=10k, seed=0, rs=0, depth=4)

| grow_policy | N | seed | rs | CPU best val RMSE | MLX best val RMSE | ratio | Verdict |
|-------------|---|------|----|-------------------|-------------------|-------|---------|
| SymmetricTree | 10,000 | 0 | 0.0 | 0.26076676 | 0.26126093 | 1.0019 | PASS |

FU-1 fix is DW-specific (ComputeLeafIndicesDepthwise). SymmetricTree uses ComputeLeafIndices
(untouched). This cell confirms no accidental regression in the ST path.

### DW training RMSE non-regression (N=10k, seed=0, rs=0, depth=4)

| Metric | Value |
|--------|-------|
| MLX train RMSE (post-fix) | 0.23278134 |
| CPU train RMSE | 0.23259824 |
| MLX/CPU train RMSE ratio | 1.0008 |
| Pre-fix baseline | N/A (T1 used depth=3 not depth=4) |

Training path non-regression: MLX/CPU train RMSE ratio = 1.0008 (within [0.98, 1.02] — training path unaffected by FU-1 fix).

## Kill-switch Status

**No kill-switch fires.** All gate cells within criterion.

## Collateral Findings

### AN-017 re-capture

AN-017 anchor: `benchmarks/sprint26/fu2/fu2-gate-report.md:101`.
Original value: `0.17222003` (mean DW RMSE over 100 determinism runs, N=10k, seed=1337,
rs=0, grow_policy=Depthwise, d=6, 128 bins, LR=0.03, 50 iters, no eval_set).

AN-017 was captured by the G5 determinism harness at
`benchmarks/sprint26/fu2/g4_determinism.py` which trains WITHOUT validation data
(no `eval_set`, no `eval_fraction`). Therefore `valDocs = 0` in the C++ training loop
and `ComputeLeafIndicesDepthwise` is NEVER CALLED during those runs.
FU-1 fixes only the validation path (line 4054 in csv_train.cpp, inside `if (valDocs > 0)`).
AN-017 is a training-RMSE anchor, not a validation-RMSE anchor. It is NOT FU-1-affected.

### AN-017 live re-run results (5-run mini-capture, post-FU-1 binary)

| Metric | Original (S26-FU-2) | Post-FU-1 | Delta |
|--------|---------------------|-----------|-------|
| Mean RMSE | 0.17222003 | 0.17222002 | 5.53e-09 (abs) |
| Median RMSE | 0.17222002 | 0.17222002 | 0 |
| drift_rel | — | — | 3.21e-08 |
| max - min | 1.49e-08 | 1.49e-08 | 0 |

**Verdict: NOT FU-1-AFFECTED.** Relative drift 3.21e-08 is well below the 1e-4 threshold.
`valDocs=0` in the generating harness — `ComputeLeafIndicesDepthwise` was never on the code
path for this anchor. No anchor update to `fu2-gate-report.md` is needed.

**Anchor update**: SKIPPED (no drift exceeding threshold). No separate AN-017 commit needed.

## Timing

| grow_policy | N | seed | CPU_t | MLX_t |
|-------------|---|------|-------|-------|
| Depthwise | 10,000 | 0 | 0.16s | 0.46s |
| Depthwise | 10,000 | 7 | 0.09s | 0.32s |
| Depthwise | 10,000 | 42 | 0.09s | 0.30s |
| Depthwise | 50,000 | 0 | 0.25s | 0.40s |
| Depthwise | 50,000 | 7 | 0.24s | 0.41s |
| Depthwise | 50,000 | 42 | 0.25s | 0.41s |

## Files

- `docs/sprint27/scratch/fu1_t4_gate.py` — this harness
- `docs/sprint27/fu1/t4-gate-report.md` — this report
- `benchmarks/sprint26/fu2/fu2-gate-report.md` — AN-017 source
- `benchmarks/sprint26/fu2/g4_determinism.py` — AN-017 generating harness
