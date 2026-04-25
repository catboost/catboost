#!/usr/bin/env python3
"""DEC-042 G4a/G4b: ST+Cosine drift measurement post-fix.

G4a: iter=1, ST+Cosine, drift <= 0.1% vs CPU CatBoost Cosine
G4b: iter=50, ST+Cosine, drift <= 2% vs CPU CatBoost Cosine

Anchor: np.random.default_rng(42), N=50000, 20 features,
y = 0.5*X[0] + 0.3*X[1] + 0.1*noise, ST/Cosine/RMSE, d=6, bins=128, l2=3, lr=0.03
Pre-fix baseline (DEC-036): ~52.6% drift at iter=50 (ratio ~1.526)
"""

import csv as csv_mod
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx")
BINARY = REPO / "csv_train_g4_cosine"
OUT_DIR = REPO / "docs/sprint33/commit2-gates/data"
ANCHOR_CSV = REPO / "docs/sprint33/probe-c-borders/data/anchor.csv"

FEATURES = 20
BINS = 128
DEPTH = 6
LR = 0.03
L2 = 3.0
SEED = 42
RS = 0.0

# Pre-fix baselines (from DEC-036 / PROBE-E / sprint32 close)
PREFIX_ITER1_RATIO = None   # Not previously measured under current anchor
PREFIX_ITER50_RATIO = 1.526  # 52.6% drift at iter=50


def parse_final_loss(stdout: str) -> float:
    last_loss = None
    for line in stdout.split("\n"):
        if "loss=" in line and "iter=" in line:
            for tok in line.split():
                if tok.startswith("loss="):
                    try:
                        last_loss = float(tok.split("=", 1)[1])
                    except ValueError:
                        pass
    if last_loss is None:
        raise ValueError(f"Could not parse final loss:\n{stdout[:1000]}")
    return last_loss


def run_mlx(iters: int) -> float:
    cmd = [
        str(BINARY),
        str(ANCHOR_CSV),
        "--iterations", str(iters),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--loss", "rmse",
        "--grow-policy", "SymmetricTree",
        "--score-function", "Cosine",
        "--seed", str(SEED),
        "--random-strength", str(RS),
        "--l2", str(L2),
        "--verbose",
    ]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"csv_train_g4_cosine failed:\n{result.stderr[:500]}\n{result.stdout[:200]}")
    return parse_final_loss(result.stdout)


def run_cpu_cosine(iters: int) -> float:
    from catboost import CatBoostRegressor
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((50000, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(50000).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    m = CatBoostRegressor(
        iterations=iters,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        score_function="Cosine",
        max_bin=BINS,
        random_seed=SEED,
        random_strength=RS,
        bootstrap_type="No",
        l2_leaf_reg=L2,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


print("=" * 70)
print("DEC-042 G4a/G4b: ST+Cosine drift (post-fix)")
print(f"  Binary: {BINARY}")
print(f"  Anchor: {ANCHOR_CSV}")
print("=" * 70)

if not BINARY.exists():
    print(f"ERROR: Binary not found: {BINARY}")
    sys.exit(1)

results = {}

# G4a: iter=1
print("\n--- G4a: iter=1 ---")
try:
    mlx_1 = run_mlx(1)
    cpu_1 = run_cpu_cosine(1)
    ratio_1 = mlx_1 / cpu_1
    drift_pct_1 = abs(ratio_1 - 1.0) * 100.0
    g4a_pass = drift_pct_1 <= 0.1
    print(f"  MLX RMSE (iter=1): {mlx_1:.8f}")
    print(f"  CPU RMSE (iter=1): {cpu_1:.8f}")
    print(f"  ratio:             {ratio_1:.6f}")
    print(f"  drift:             {drift_pct_1:.4f}%  (threshold: <=0.1%)")
    print(f"  G4a: {'PASS' if g4a_pass else 'FAIL'}")
    results["g4a"] = {
        "mlx_rmse": mlx_1, "cpu_rmse": cpu_1, "ratio": ratio_1,
        "drift_pct": drift_pct_1, "threshold_pct": 0.1, "pass": g4a_pass
    }
except Exception as e:
    print(f"  G4a ERROR: {e}")
    results["g4a"] = {"error": str(e), "pass": False}

# G4b: iter=50
print("\n--- G4b: iter=50 ---")
try:
    mlx_50 = run_mlx(50)
    cpu_50 = run_cpu_cosine(50)
    ratio_50 = mlx_50 / cpu_50
    drift_pct_50 = abs(ratio_50 - 1.0) * 100.0
    g4b_pass = drift_pct_50 <= 2.0
    prefix_drift = abs(PREFIX_ITER50_RATIO - 1.0) * 100.0
    print(f"  MLX RMSE (iter=50):   {mlx_50:.8f}")
    print(f"  CPU RMSE (iter=50):   {cpu_50:.8f}")
    print(f"  ratio:                {ratio_50:.6f}")
    print(f"  drift:                {drift_pct_50:.4f}%  (threshold: <=2%)")
    print(f"  pre-fix drift:        {prefix_drift:.1f}% (DEC-036 baseline)")
    print(f"  G4b: {'PASS' if g4b_pass else 'FAIL'}")
    results["g4b"] = {
        "mlx_rmse": mlx_50, "cpu_rmse": cpu_50, "ratio": ratio_50,
        "drift_pct": drift_pct_50, "threshold_pct": 2.0, "pass": g4b_pass,
        "prefix_drift_pct": prefix_drift
    }
except Exception as e:
    print(f"  G4b ERROR: {e}")
    results["g4b"] = {"error": str(e), "pass": False}

# Save
out_json = OUT_DIR / "g4a_g4b_results.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to: {out_json}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
g4a_v = results.get("g4a", {}).get("pass", False)
g4b_v = results.get("g4b", {}).get("pass", False)
print(f"G4a (iter=1, drift<=0.1%): {'PASS' if g4a_v else 'FAIL'}")
print(f"G4b (iter=50, drift<=2%):  {'PASS' if g4b_v else 'FAIL'}")
if not (g4a_v and g4b_v):
    sys.exit(1)
