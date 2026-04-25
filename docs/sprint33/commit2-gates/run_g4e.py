#!/usr/bin/env python3
"""DEC-042 G4e: DW+Cosine sanity unchanged at S28 anchor.

Verifies the DEC-042 per-side mask fix does NOT break the DW+Cosine path.
The fix is in FindBestSplit (SymmetricTree path); DW uses FindBestSplitPerPartition.
This gate confirms the DW Cosine path is unaffected.

S28 baseline (from docs/sprint28/fu-l2-explicit/t3-gate-results.json):
  MLX(DW+Cosine) vs CPU(DW+Cosine), N=1000, 5 seeds {42..46}
  All 5 seeds PASS [0.98, 1.02]
  Ratios: [1.0159, 1.0160, 1.0023, 1.0076, 0.9950]

Post-fix expectation: ratios unchanged (DW path not touched by fix).

Config: N=1000, DW, Cosine, 50 iters, d=6, bins=128, lr=0.03, l2=3, rs=0, seeds 42-46.
Uses Python API (catboost_mlx), same as S28 harness.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx")
OUT_DIR = REPO / "docs/sprint33/commit2-gates/data"

# Add python path
sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03
LAMBDA = 3.0
N = 1000
SEEDS = [42, 43, 44, 45, 46]

# S28 baseline from t3-gate-results.json (DW+Cosine G3b)
S28_BASELINE = {
    42: {"mlx": 0.2140252912302744, "cpu": 0.21067719079049668, "ratio": 1.0158920879247302},
    43: {"mlx": 0.21231093615709665, "cpu": 0.2089676034007807, "ratio": 1.0159992874584667},
    44: {"mlx": 0.21063518594448244, "cpu": 0.21015601844597923, "ratio": 1.002280056036684},
    45: {"mlx": 0.21478827717872967, "cpu": 0.2131736587102339, "ratio": 1.0075741931637554},
    46: {"mlx": 0.21846702798989448, "cpu": 0.21957083711876269, "ratio": 0.9949728791703282},
}


def make_xy(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    return X, y


try:
    from catboost_mlx import CatBoostMLXRegressor
    MLX_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: catboost_mlx not available: {e}")
    MLX_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CB_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: catboost not available: {e}")
    CB_AVAILABLE = False

print("=" * 70)
print("DEC-042 G4e: DW+Cosine sanity (S28 anchor)")
print(f"  Config: N={N}, DW, Cosine, {ITERS} iters, d={DEPTH}, bins={BINS}")
print(f"  Seeds: {SEEDS}")
print(f"  MLX available: {MLX_AVAILABLE}, CatBoost available: {CB_AVAILABLE}")
print("=" * 70)

results = []
pass_count = 0

print(f"\n{'seed':>5} | {'MLX_RMSE':>11} | {'CPU_Cos':>11} | {'ratio':>8} | {'S28_ratio':>9} | {'delta':>8} | gate")
print("-" * 70)

for seed in SEEDS:
    X, y = make_xy(N, seed)

    mlx_rmse = None
    if MLX_AVAILABLE:
        m_mlx = CatBoostMLXRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LR,
            l2_reg_lambda=LAMBDA, bins=BINS,
            random_seed=seed, random_strength=0.0,
            grow_policy='Depthwise',
            score_function='Cosine',
        )
        m_mlx.fit(X, y)
        preds = m_mlx.predict(X)
        mlx_rmse = float(np.sqrt(np.mean((preds - y) ** 2)))

    cpu_cosine_rmse = None
    if CB_AVAILABLE:
        m_cos = CatBoostRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LR,
            loss_function="RMSE", grow_policy="Depthwise",
            max_bin=BINS, random_seed=seed, random_strength=0.0,
            bootstrap_type="No", l2_leaf_reg=LAMBDA,
            score_function="Cosine", verbose=0, thread_count=1,
        )
        m_cos.fit(X, y)
        cpu_cosine_rmse = float(m_cos.evals_result_["learn"]["RMSE"][-1])

    if mlx_rmse is not None and cpu_cosine_rmse is not None:
        ratio = mlx_rmse / cpu_cosine_rmse
        cell_pass = 0.98 <= ratio <= 1.02
        s28_ratio = S28_BASELINE[seed]["ratio"]
        delta = ratio - s28_ratio
        if cell_pass:
            pass_count += 1
        print(f"{seed:>5} | {mlx_rmse:>11.8f} | {cpu_cosine_rmse:>11.8f} | {ratio:>8.4f} | {s28_ratio:>9.4f} | {delta:>+8.4f} | {'PASS' if cell_pass else 'FAIL'}")
        results.append({
            "seed": seed,
            "mlx_rmse": mlx_rmse,
            "cpu_cosine_rmse": cpu_cosine_rmse,
            "ratio": ratio,
            "s28_baseline_ratio": s28_ratio,
            "delta_from_s28": delta,
            "pass": cell_pass,
        })
    else:
        print(f"{seed:>5} | {'N/A':>11} | {'N/A':>11} | {'N/A':>8} | {'N/A':>9} | {'N/A':>8} | SKIP")
        results.append({"seed": seed, "pass": None})

print("-" * 70)

g4e_pass = pass_count == len(SEEDS) and len(results) == len(SEEDS)
print(f"\nG4e: {pass_count}/{len(SEEDS)} PASS — verdict: {'PASS' if g4e_pass else 'FAIL'}")

if results:
    ratios = [r["ratio"] for r in results if r.get("ratio") is not None]
    deltas = [r["delta_from_s28"] for r in results if r.get("delta_from_s28") is not None]
    if ratios:
        print(f"Ratio range: [{min(ratios):.4f}, {max(ratios):.4f}]")
    if deltas:
        print(f"Delta from S28 baseline range: [{min(deltas):+.4f}, {max(deltas):+.4f}]")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_json = OUT_DIR / "g4e_results.json"
with open(out_json, "w") as f:
    json.dump({
        "gate": "G4e",
        "description": "DW+Cosine sanity at S28 anchor, post DEC-042 fix",
        "pass": g4e_pass,
        "pass_count": pass_count,
        "total": len(SEEDS),
        "results": results,
    }, f, indent=2)
print(f"Results written to: {out_json}")

if not g4e_pass:
    sys.exit(1)
