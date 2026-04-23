"""S28-L2-EXPLICIT Gate Harness — G3a (L2 no-regression) + G3b (Cosine live-path)
+ G3c (unknown-value guard).

Gate G3a (L2 no-regression):
  No score_function passed (defaults to L2). MLX DW N=1000 vs CPU DW L2,
  5 seeds {42-46}, rs=0. Ratios must be 5/5 PASS in [0.98, 1.02].
  Any regression means enum dispatch itself introduced drift.

Gate G3b (Cosine live-path parity):
  score_function='Cosine' passed. MLX DW N=1000 vs CPU DW Cosine,
  5 seeds {42-46}, rs=0. At formula level S28-COSINE measured max 1 ULP;
  live-path must also be tight. Ratios expected [0.98, 1.02] (closing
  the 0.83-0.87 gap recorded in t2-gate-report.md).

Gate G3c (unknown-value guard):
  score_function='NewtonCosine' must throw std::invalid_argument with
  a clear message, not silently fall back to L2.

Dataset: N=1000, 20 features, depth=6, 128 bins, 50 iters, rs=0, Depthwise.
Seeds: {42, 43, 44, 45, 46}
"""

import os
import sys
import json
import math
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/python"
)))
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

OUT_DIR = Path(
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/"
    "docs/sprint28/fu-l2-explicit"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03
N = 1000
SEEDS = [42, 43, 44, 45, 46]
LAMBDA = 3.0

# =============================================================================
# ULP utility
# =============================================================================

def float32_ulp_distance(a: float, b: float) -> int:
    a32 = float(np.float32(a))
    b32 = float(np.float32(b))
    ia = struct.unpack('I', struct.pack('f', a32))[0]
    ib = struct.unpack('I', struct.pack('f', b32))[0]
    return abs(int(ia) - int(ib))


# =============================================================================
# Imports
# =============================================================================

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


def make_xy(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    return X, y


# =============================================================================
# Gate G3a: L2 no-regression (default score_function, no arg passed)
# =============================================================================

print("=" * 80)
print("Gate G3a: L2 no-regression — no score_function arg, default=L2")
print("=" * 80)
print(f"\n{'seed':>5s} | {'MLX_RMSE':>10s} | {'CPU_L2':>10s} | {'ratio':>8s} | G3a")
print("-" * 50)

g3a_results = []
g3a_pass_count = 0

for seed in SEEDS:
    X, y = make_xy(N, seed)

    mlx_rmse = None
    if MLX_AVAILABLE:
        # No score_function kwarg — exercises default path
        m_mlx = CatBoostMLXRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LR,
            l2_reg_lambda=LAMBDA, bins=BINS,
            random_seed=seed, random_strength=0.0,
            grow_policy='Depthwise',
        )
        m_mlx.fit(X, y)
        preds = m_mlx.predict(X)
        mlx_rmse = float(np.sqrt(np.mean((preds - y) ** 2)))

    cpu_l2_rmse = None
    if CB_AVAILABLE:
        m_l2 = CatBoostRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LR,
            loss_function="RMSE", grow_policy="Depthwise",
            max_bin=BINS, random_seed=seed, random_strength=0.0,
            bootstrap_type="No", l2_leaf_reg=LAMBDA,
            score_function="L2", verbose=0, thread_count=1,
        )
        m_l2.fit(X, y)
        cpu_l2_rmse = float(m_l2.evals_result_["learn"]["RMSE"][-1])

    if mlx_rmse is not None and cpu_l2_rmse is not None:
        ratio = mlx_rmse / cpu_l2_rmse
        passes = 0.98 <= ratio <= 1.02
        if passes:
            g3a_pass_count += 1
        print(f"{seed:>5d} | {mlx_rmse:>10.6f} | {cpu_l2_rmse:>10.6f} | {ratio:>8.4f} | {'PASS' if passes else 'FAIL'}")
        g3a_results.append({
            "seed": seed, "mlx_rmse": mlx_rmse, "cpu_l2_rmse": cpu_l2_rmse,
            "ratio": ratio, "pass": passes
        })
    else:
        print(f"{seed:>5d} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>8s} | SKIP")

g3a_pass = g3a_pass_count == len(SEEDS) and len(g3a_results) == len(SEEDS)
print(f"\nG3a: {g3a_pass_count}/{len(SEEDS)} PASS — verdict: {'PASS' if g3a_pass else 'FAIL'}")


# =============================================================================
# Gate G3b: Cosine live-path parity
# =============================================================================

print("\n" + "=" * 80)
print("Gate G3b: Cosine live-path — score_function='Cosine', MLX vs CPU(Cosine)")
print("=" * 80)
print(f"\n{'seed':>5s} | {'MLX_RMSE':>10s} | {'CPU_Cos':>10s} | {'ratio':>8s} | G3b")
print("-" * 50)

g3b_results = []
g3b_pass_count = 0

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
        # Parity gate: same algorithm on both sides should converge to [0.98, 1.02]
        passes = 0.98 <= ratio <= 1.02
        if passes:
            g3b_pass_count += 1
        print(f"{seed:>5d} | {mlx_rmse:>10.6f} | {cpu_cosine_rmse:>10.6f} | {ratio:>8.4f} | {'PASS' if passes else 'FAIL'}")
        g3b_results.append({
            "seed": seed, "mlx_rmse": mlx_rmse, "cpu_cosine_rmse": cpu_cosine_rmse,
            "ratio": ratio, "pass": passes
        })
    else:
        print(f"{seed:>5d} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>8s} | SKIP")

g3b_pass = g3b_pass_count == len(SEEDS) and len(g3b_results) == len(SEEDS)
print(f"\nG3b: {g3b_pass_count}/{len(SEEDS)} PASS — verdict: {'PASS' if g3b_pass else 'FAIL'}")


# =============================================================================
# Gate G3c: unknown-value guard
# =============================================================================

print("\n" + "=" * 80)
print("Gate G3c: Unknown score_function='NewtonCosine' must raise, not silently use L2")
print("=" * 80)

g3c_pass = False
g3c_msg = ""

if MLX_AVAILABLE:
    X, y = make_xy(N, 42)
    try:
        m_bad = CatBoostMLXRegressor(
            iterations=5, depth=3, learning_rate=LR,
            l2_reg_lambda=LAMBDA, bins=BINS,
            random_seed=42, random_strength=0.0,
            grow_policy='Depthwise',
            score_function='NewtonCosine',
        )
        m_bad.fit(X, y)
        g3c_msg = "ERROR: no exception raised — silent fallback to L2 (bug)"
        g3c_pass = False
    except Exception as exc:
        g3c_msg = str(exc)
        # Must contain a helpful message, not an empty exception
        g3c_pass = "NewtonCosine" in g3c_msg or "not yet implemented" in g3c_msg or "Unknown" in g3c_msg
    print(f"\nException raised: {g3c_pass}")
    print(f"Message: {g3c_msg[:200]}")
    print(f"G3c: {'PASS' if g3c_pass else 'FAIL'}")
else:
    print("SKIP — catboost_mlx not available")


# =============================================================================
# REFLECT: ULP expansion from formula level to live-path level
# =============================================================================

print("\n" + "=" * 80)
print("REFLECT: live-path ULP expansion vs formula level")
print("=" * 80)

if g3b_results and g3a_results:
    # Formula level (G2a): max 1 ULP  (from t2-gate-report.md)
    # Live-path parity (G3b): measured as RMSE ratio, not ULP directly.
    # Translate RMSE ratio to expected ULP order:
    #   ratio=1.000 → 0 ULP; ratio=1.001 → O(1) ULP in RMSE space
    g3b_ratios = [r["ratio"] for r in g3b_results]
    max_dev = max(abs(r - 1.0) for r in g3b_ratios)
    print(f"\nFormula-level max ULP (t2-gate): 1")
    print(f"Live-path RMSE ratios: {[round(r, 4) for r in g3b_ratios]}")
    print(f"Live-path max deviation from 1.0: {max_dev:.4f} ({max_dev*100:.2f}%)")
    if max_dev <= 0.02:
        print("Live-path deviation ≤ 2% — consistent with formula-level 1 ULP (no accumulation blow-up)")
    else:
        print("WARNING: live-path deviation > 2% — investigate accumulation path for dtype issues")

    # Enum extensibility note
    print("\nEnum extensibility note:")
    print("  Adding NewtonL2/NewtonCosine requires: (a) add enum values, (b) add formulas,")
    print("  (c) update ParseScoreFunction to remove the throw for those values.")
    print("  The switch structure in FindBestSplitPerPartition is already the correct shape.")
    print("  Estimated effort: 1 sprint task, no structural changes needed.")


# =============================================================================
# Save results
# =============================================================================

results = {
    "date": "2026-04-23",
    "branch": "mlx/sprint-28-score-function-fidelity",
    "task": "S28-L2-EXPLICIT",
    "gate": "G3a+G3b+G3c",
    "seeds": SEEDS,
    "config": {"N": N, "depth": DEPTH, "iters": ITERS, "bins": BINS, "lr": LR, "lambda": LAMBDA},
    "g3a_l2_no_regression": {
        "pass_count": g3a_pass_count,
        "total": len(SEEDS),
        "pass": g3a_pass,
        "results": g3a_results,
        "ratios": [r["ratio"] for r in g3a_results],
    },
    "g3b_cosine_live_path": {
        "pass_count": g3b_pass_count,
        "total": len(SEEDS),
        "pass": g3b_pass,
        "results": g3b_results,
        "ratios": [r["ratio"] for r in g3b_results],
    },
    "g3c_unknown_guard": {
        "pass": g3c_pass,
        "exception_message": g3c_msg,
    },
}

out_json = OUT_DIR / "t3-gate-results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to: {out_json}")

# =============================================================================
# Final summary
# =============================================================================

print("\n" + "=" * 80)
print("GATE SUMMARY")
print("=" * 80)
print(f"G3a (L2 no-regression, 5 seeds):  {'PASS' if g3a_pass else 'FAIL'}")
print(f"G3b (Cosine live-path, 5 seeds):  {'PASS' if g3b_pass else 'FAIL'}")
print(f"G3c (NewtonCosine guard):          {'PASS' if g3c_pass else 'FAIL'}")
overall = g3a_pass and g3b_pass and g3c_pass
print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
