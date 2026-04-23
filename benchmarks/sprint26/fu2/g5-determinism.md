# S26-FU-2 G5 Determinism Check

**Branch**: `mlx/sprint-26-fu2-noise-dwlg`  
**Config**: N=10,000, seed=1337, rs=0.0, Depthwise, d=6, 128 bins, LR=0.03, 50 iters  
**Runs**: 100  
**Threshold**: max − min ≤ 1e-06  

## Why Depthwise at gate

D0 used SymmetricTree at the G4 gate (range 1.49e-08). FU-2's change is in
`FindBestSplitPerPartition`, the non-oblivious code path. Depthwise exercises
this path directly. `rs=0` because at `rs=1` the RNG dominates and determinism
is trivially met by seed-fixing; `rs=0` stresses GPU float32 accumulation order
— the actual source of any non-determinism that FU-2 could introduce.

## Results

| Metric | Value |
|--------|-------|
| Mean RMSE | 0.17222003 |
| Median RMSE | 0.17222002 |
| max − min | 1.49e-08 |
| Std dev | 7.01e-09 |
| Wall time | 71.2s (0.71s/run) |

**Verdict**: DETERMINISTIC (range 1.49e-08 ≤ 1e-6)

**KS-4 status**: CLEAR

## Comparison with D0 baseline

| Gate | Config | max − min | Verdict |
|------|--------|-----------|---------|
| D0 G4 | SymmetricTree, N=10k, rs=0 | 1.49e-08 | DETERMINISTIC |
| FU-2 G5 | Depthwise, N=10k, rs=0 | 1.49e-08 | DETERMINISTIC |

## Notes

- `rs=0.0` disables RandomStrength noise injection, making split selection
  deterministic given identical inputs and seed.
- The FU-2 change adds a noise path in `FindBestSplitPerPartition`. At `rs=0`,
  the noise scale is zero so the new code path is dormant — the only source of
  variation is Metal GPU float32 accumulation order (same as pre-FU-2).
- Expected: `max − min < 1e-6`. If > 1e-6, the RNG plumbing introduces
  non-determinism even at rs=0 (e.g., an uninitialized generator, a stale
  branch on a Metal buffer state). That would be KS-4.
- Threshold is hard: ≤ 1e-6. Do NOT loosen.
