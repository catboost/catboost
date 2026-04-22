# S26-D0-7 G4 Determinism Check

**Branch**: `mlx/sprint-26-python-parity`  
**Config**: N=10k, seed=1337, rs=0.0, SymmetricTree, d=6, 128 bins, LR=0.03, 50 iters  
**Runs**: 100

## Results

| Metric | Value |
|--------|-------|
| Mean RMSE | 0.19457837 |
| Median RMSE | 0.19457836 |
| max − min | 1.49e-08 |
| Std dev | 6.17e-09 |
| Wall time | 48.1s (0.48s/run) |

**Verdict**: DETERMINISTIC (range < 1e-6)

## Notes

- `rs=0.0` disables RandomStrength noise injection, making split selection
  deterministic given identical inputs and seed.
- Expected behavior: `max − min < 1e-6` for a fully deterministic implementation.
- Float32 Metal accumulation noise (Metal GPU reduction order not guaranteed)
  can cause run-to-run variation up to ~1e-5; this is documented acceptable behavior.
- If `max − min > 1e-5`, the DEC-028 fix or subsequent changes have introduced
  a new source of non-determinism and must be investigated before sprint close.
- This check is NOT a gate criterion — it is a sanity check that the fix did not
  introduce unexpected non-determinism.
