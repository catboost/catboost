"""S27-FU-1-T4 AN-017 re-capture post-FU-1.

AN-017 is the mean DW RMSE from the S26-FU-2 G5 determinism run:
  benchmarks/sprint26/fu2/fu2-gate-report.md line ~101
  Original value: 0.17222003 (mean over 100 runs)
  Config: N=10k, seed=1337, rs=0.0, Depthwise, d=6, 128 bins, LR=0.03, 50 iters.
  Harness: benchmarks/sprint26/fu2/g4_determinism.py

This anchor does NOT use eval_set — there is no validation data.  valDocs=0 in the
C++ training loop, so ComputeLeafIndicesDepthwise is never called.  FU-1 only fixes
the code path inside `if (valDocs > 0)` at csv_train.cpp:4047.

Therefore AN-017 is expected to be unchanged by FU-1.  This script re-runs the same
3-run mini-check (not 100 runs — 100 runs is a gate-level check, not re-capture)
to confirm the value is stable, then compares to the original anchor.

Tolerance: |drift_rel| < 1e-4 (relative) → anchor not affected → note and skip update.
           |drift_rel| >= 1e-4 → some other change moved it → update anchor + footnote.

Commit message if no update needed: none (skip commit).
Commit message if update needed:
  [mlx] sprint-27: S27-FU-1-T4 AN-017 re-capture post-FU-1
"""

import os
import sys
import time
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# AN-017 original anchor values (from fu2-gate-report.md §4)
AN017_ORIGINAL_MEAN = 0.17222003
AN017_ORIGINAL_MEDIAN = 0.17222002
AN017_ORIGINAL_MAX_MIN = 1.49e-08
AN017_ORIGINAL_STD = 7.01e-09

# Config matching g4_determinism.py exactly
N = 10_000
SEED = 1337
RS = 0.0
GROW_POLICY = "Depthwise"
FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03
RUNS = 5       # mini re-capture: 5 runs sufficient to confirm stability; full 100-run is gate-level
THRESHOLD_REL = 1e-4

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, FEATURES)).astype(np.float32)
noise = rng.standard_normal(N).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)

from catboost_mlx import CatBoostMLXRegressor

print("AN-017 re-capture (S27-FU-1-T4)")
print(f"  Config: N={N}, seed={SEED}, rs={RS}, {GROW_POLICY}, d={DEPTH}, {BINS} bins, "
      f"LR={LR}, {ITERS} iters (no eval_set)")
print(f"  Original anchor (fu2-gate-report.md): mean={AN017_ORIGINAL_MEAN}")
print(f"  Running {RUNS} mini-capture runs...")
sys.stdout.flush()

rmses = []
t0 = time.perf_counter()
for i in range(RUNS):
    m = CatBoostMLXRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss="rmse", grow_policy=GROW_POLICY, bins=BINS,
        random_seed=SEED, random_strength=RS,
        bootstrap_type="no",
        verbose=False,
    )
    m.fit(X, y)   # NO eval_set — valDocs=0, ComputeLeafIndicesDepthwise not called
    if m._train_loss_history:
        rmse = float(m._train_loss_history[-1])
    else:
        preds = m.predict(X)
        rmse = float(np.sqrt(((np.asarray(preds, dtype=np.float64) - y) ** 2).mean()))
    rmses.append(rmse)
    print(f"  run {i+1}/{RUNS}: rmse={rmse:.8f}")
    sys.stdout.flush()

wall = time.perf_counter() - t0

rmses_arr = np.array(rmses, dtype=np.float64)
mean_new = float(rmses_arr.mean())
median_new = float(np.median(rmses_arr))
range_new = float(rmses_arr.max() - rmses_arr.min())
std_new = float(rmses_arr.std())

drift_abs = abs(mean_new - AN017_ORIGINAL_MEAN)
drift_rel = drift_abs / AN017_ORIGINAL_MEAN if AN017_ORIGINAL_MEAN != 0 else float("nan")

print(f"\n--- AN-017 Re-Capture Results ---")
print(f"  Original (S26-FU-2)  : mean={AN017_ORIGINAL_MEAN:.8f}  median={AN017_ORIGINAL_MEDIAN:.8f}")
print(f"  Post-FU-1 ({RUNS} runs): mean={mean_new:.8f}  median={median_new:.8f}")
print(f"  |drift_abs|          : {drift_abs:.2e}")
print(f"  drift_rel            : {drift_rel:.2e}")
print(f"  max-min (new)        : {range_new:.2e}")
print(f"  wall                 : {wall:.1f}s ({wall/RUNS:.2f}s/run)")
sys.stdout.flush()

# Verdict
if drift_rel < THRESHOLD_REL:
    verdict = "NOT FU-1-AFFECTED"
    detail = (
        f"Anchor stable (|drift_rel|={drift_rel:.2e} < {THRESHOLD_REL:.0e}). "
        f"AN-017 uses no eval_set (valDocs=0); ComputeLeafIndicesDepthwise not on the code path. "
        f"No anchor update needed."
    )
    update_needed = False
else:
    verdict = "DRIFTED — UPDATE REQUIRED"
    detail = (
        f"Anchor drifted (|drift_rel|={drift_rel:.2e} >= {THRESHOLD_REL:.0e}). "
        f"Some other change (not FU-1 validation path) moved the training RMSE. "
        f"Investigate: diff all commits on branch since AN-017 was captured (S26-FU-2 tip 715b15b613). "
        f"Update anchor in fu2-gate-report.md with post-FU-1 value and footnote."
    )
    update_needed = True

print(f"\n  Verdict: {verdict}")
print(f"  {detail}")

# ── Write inline finding for gate report ──────────────────────────────────────

AN017_FINDING = {
    "original_mean": AN017_ORIGINAL_MEAN,
    "original_median": AN017_ORIGINAL_MEDIAN,
    "new_mean": mean_new,
    "new_median": median_new,
    "drift_abs": drift_abs,
    "drift_rel": drift_rel,
    "update_needed": update_needed,
    "verdict": verdict,
    "detail": detail,
    "runs": RUNS,
}

print(f"\nAN-017 finding ready for gate report (update_needed={update_needed}).")
