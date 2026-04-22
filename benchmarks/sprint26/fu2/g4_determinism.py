"""S26-FU-2 G5 determinism check.

Gate config: N=10k, seed=1337, rs=0.0, grow_policy='Depthwise', d=6, 128 bins,
             LR=0.03, 50 iters, RMSE.

Why Depthwise at gate: D0's gate used SymmetricTree. FU-2's new lever is in the
non-oblivious path (FindBestSplitPerPartition), so Depthwise is the path most
affected by the change. rs=0 because at rs=1 the RNG dominates and determinism
is trivially met by seed-fixing; rs=0 stresses GPU float32 accumulation order
which is where non-determinism would actually surface.

Run MLX training 100 times. Collect final RMSE each run.
Threshold: max − min ≤ 1e-6. D0 G4 (SymmetricTree) landed at 1.49e-08.

Kill-switch KS-4: if max − min > 1e-6, RNG plumbed non-deterministically — ESCALATE.
Results written to benchmarks/sprint26/fu2/g5-determinism.md.
"""

import os
import sys
import time
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

N = 10_000
SEED = 1337
RS = 0.0
GROW_POLICY = "Depthwise"
FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03
RUNS = 100
THRESHOLD = 1e-6

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
        loss="rmse", grow_policy=GROW_POLICY, bins=BINS,
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
        print(f"  run {i+1:3d}/{RUNS}  rmse={rmse:.8f}")
        sys.stdout.flush()

wall = time.perf_counter() - t0_wall
rmses = np.array(rmses, dtype=np.float64)

range_val = float(rmses.max() - rmses.min())
std_val = float(rmses.std())
mean_val = float(rmses.mean())
median_val = float(np.median(rmses))

print(f"\n--- G5 Determinism Results (Depthwise, rs=0.0) ---")
print(f"  grow_policy: {GROW_POLICY}")
print(f"  runs       : {RUNS}")
print(f"  mean RMSE  : {mean_val:.8f}")
print(f"  median     : {median_val:.8f}")
print(f"  max - min  : {range_val:.2e}")
print(f"  std dev    : {std_val:.2e}")
print(f"  wall time  : {wall:.1f}s  ({wall/RUNS:.2f}s/run)")

if range_val <= THRESHOLD:
    verdict = f"DETERMINISTIC (range {range_val:.2e} ≤ 1e-6)"
    ks4_status = "CLEAR"
elif range_val < 1e-5:
    verdict = f"NEAR-DETERMINISTIC (float32 accumulation noise, range {range_val:.2e} < 1e-5, acceptable)"
    ks4_status = "CLEAR (range < 1e-5, acceptable float32 noise)"
else:
    verdict = (
        f"NON-DETERMINISTIC (range {range_val:.2e} > 1e-6 — "
        "KS-4 FIRES: RNG plumbed non-deterministically, ESCALATE, revert-and-replan)"
    )
    ks4_status = "KS-4 FIRE"

print(f"  verdict    : {verdict}")
if ks4_status.startswith("KS-4"):
    print(f"\n*** KS-4 KILL-SWITCH FIRES — range {range_val:.2e} > threshold {THRESHOLD:.0e} ***")
    print("*** STOP. Do NOT paper over. Investigate non-determinism source. ***")

# ── Write markdown ────────────────────────────────────────────────────────────

RESULTS_PATH = (
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/"
    "benchmarks/sprint26/fu2/g5-determinism.md"
)

md_lines = [
    "# S26-FU-2 G5 Determinism Check",
    "",
    "**Branch**: `mlx/sprint-26-fu2-noise-dwlg`  ",
    f"**Config**: N={N:,}, seed={SEED}, rs={RS}, {GROW_POLICY}, d={DEPTH}, {BINS} bins, LR={LR}, {ITERS} iters  ",
    f"**Runs**: {RUNS}  ",
    f"**Threshold**: max − min ≤ {THRESHOLD:.0e}  ",
    "",
    "## Why Depthwise at gate",
    "",
    "D0 used SymmetricTree at the G4 gate (range 1.49e-08). FU-2's change is in",
    "`FindBestSplitPerPartition`, the non-oblivious code path. Depthwise exercises",
    "this path directly. `rs=0` because at `rs=1` the RNG dominates and determinism",
    "is trivially met by seed-fixing; `rs=0` stresses GPU float32 accumulation order",
    "— the actual source of any non-determinism that FU-2 could introduce.",
    "",
    "## Results",
    "",
    "| Metric | Value |",
    "|--------|-------|",
    f"| Mean RMSE | {mean_val:.8f} |",
    f"| Median RMSE | {median_val:.8f} |",
    f"| max − min | {range_val:.2e} |",
    f"| Std dev | {std_val:.2e} |",
    f"| Wall time | {wall:.1f}s ({wall/RUNS:.2f}s/run) |",
    "",
    f"**Verdict**: {verdict}",
    "",
    f"**KS-4 status**: {ks4_status}",
    "",
    "## Comparison with D0 baseline",
    "",
    "| Gate | Config | max − min | Verdict |",
    "|------|--------|-----------|---------|",
    f"| D0 G4 | SymmetricTree, N=10k, rs=0 | 1.49e-08 | DETERMINISTIC |",
    f"| FU-2 G5 | Depthwise, N=10k, rs=0 | {range_val:.2e} | {verdict.split(' (')[0]} |",
    "",
    "## Notes",
    "",
    "- `rs=0.0` disables RandomStrength noise injection, making split selection",
    "  deterministic given identical inputs and seed.",
    "- The FU-2 change adds a noise path in `FindBestSplitPerPartition`. At `rs=0`,",
    "  the noise scale is zero so the new code path is dormant — the only source of",
    "  variation is Metal GPU float32 accumulation order (same as pre-FU-2).",
    "- Expected: `max − min < 1e-6`. If > 1e-6, the RNG plumbing introduces",
    "  non-determinism even at rs=0 (e.g., an uninitialized generator, a stale",
    "  branch on a Metal buffer state). That would be KS-4.",
    "- Threshold is hard: ≤ 1e-6. Do NOT loosen.",
]

with open(RESULTS_PATH, "w") as f:
    f.write("\n".join(md_lines) + "\n")

print(f"\nResults written to benchmarks/sprint26/fu2/g5-determinism.md")
