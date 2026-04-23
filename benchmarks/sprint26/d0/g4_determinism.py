"""S26-D0-7 G4 determinism sanity check.

Config: N=10k, seed=1337, rs=0 (deterministic), SymmetricTree, d=6, 128 bins,
        LR=0.03, 50 iters, RMSE.

Run MLX training 100 times. Collect final RMSE each run.
Report: max − min, std dev.

Expected: < 1e-6 if fully deterministic (same binary, same seed, same rs=0).
If non-zero, document as float32 accumulation noise (acceptable up to ~1e-5).
Results written to benchmarks/sprint26/d0/g4-determinism.md.
"""

import os
import time
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

N = 10_000
SEED = 1337
RS = 0.0
FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03
RUNS = 100

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, FEATURES)).astype(np.float32)
noise = rng.standard_normal(N).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)

from catboost_mlx import CatBoostMLXRegressor

rmses = []
t0_wall = time.perf_counter()

for i in range(RUNS):
    m = CatBoostMLXRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss="rmse", grow_policy="SymmetricTree", bins=BINS,
        random_seed=SEED, random_strength=RS,
        bootstrap_type="no",
        verbose=False,
    )
    m.fit(X, y)
    if m._train_loss_history:
        rmse = float(m._train_loss_history[-1])
    else:
        preds = m.predict(X)
        rmse = float(np.sqrt(((np.asarray(preds, dtype=np.float64) - y) ** 2).mean()))
    rmses.append(rmse)
    if (i + 1) % 10 == 0:
        print(f"  run {i+1:3d}/100  rmse={rmse:.8f}")

wall = time.perf_counter() - t0_wall
rmses = np.array(rmses, dtype=np.float64)

range_val = float(rmses.max() - rmses.min())
std_val = float(rmses.std())
mean_val = float(rmses.mean())
median_val = float(np.median(rmses))

print(f"\n--- G4 Determinism Results ---")
print(f"  runs       : {RUNS}")
print(f"  mean RMSE  : {mean_val:.8f}")
print(f"  median     : {median_val:.8f}")
print(f"  max - min  : {range_val:.2e}")
print(f"  std dev    : {std_val:.2e}")
print(f"  wall time  : {wall:.1f}s  ({wall/RUNS:.2f}s/run)")

if range_val < 1e-6:
    verdict = "DETERMINISTIC (range < 1e-6)"
elif range_val < 1e-5:
    verdict = "NEAR-DETERMINISTIC (float32 accumulation noise, range < 1e-5, acceptable)"
else:
    verdict = f"NON-DETERMINISTIC (range {range_val:.2e} > 1e-5 — investigate)"

print(f"  verdict    : {verdict}")

# ── Write markdown ────────────────────────────────────────────────────────────

md_lines = [
    "# S26-D0-7 G4 Determinism Check",
    "",
    "**Branch**: `mlx/sprint-26-python-parity`  ",
    "**Config**: N=10k, seed=1337, rs=0.0, SymmetricTree, d=6, 128 bins, LR=0.03, 50 iters  ",
    f"**Runs**: {RUNS}",
    "",
    "## Results",
    "",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Mean RMSE | {mean_val:.8f} |",
    f"| Median RMSE | {median_val:.8f} |",
    f"| max − min | {range_val:.2e} |",
    f"| Std dev | {std_val:.2e} |",
    f"| Wall time | {wall:.1f}s ({wall/RUNS:.2f}s/run) |",
    "",
    f"**Verdict**: {verdict}",
    "",
    "## Notes",
    "",
    "- `rs=0.0` disables RandomStrength noise injection, making split selection",
    "  deterministic given identical inputs and seed.",
    "- Expected behavior: `max − min < 1e-6` for a fully deterministic implementation.",
    "- Float32 Metal accumulation noise (Metal GPU reduction order not guaranteed)",
    "  can cause run-to-run variation up to ~1e-5; this is documented acceptable behavior.",
    "- If `max − min > 1e-5`, the DEC-028 fix or subsequent changes have introduced",
    "  a new source of non-determinism and must be investigated before sprint close.",
    "- This check is NOT a gate criterion — it is a sanity check that the fix did not",
    "  introduce unexpected non-determinism.",
]

results_path = (
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/"
    "benchmarks/sprint26/d0/g4-determinism.md"
)
with open(results_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")

print(f"\nResults written to benchmarks/sprint26/d0/g4-determinism.md")
