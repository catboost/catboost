"""S27-FU-3-T1 Step 3: Score function analysis for DW N=1000 asymmetry.

Hypothesis (from CPU source audit): CPU uses Cosine score function by default
(score_function=Cosine from get_all_params()), while MLX implements L2 Newton gain.
The Cosine score normalizes by sqrt(sum(leafApprox^2 * weight)), which acts as an
implicit regularizer that prevents overfitting to tiny partitions.

At large N, Cosine and L2 produce the same split ranking (both converge to same optima).
At N=1000 with depth=6 (64 partitions, ~15 docs each), the normalization matters:
L2 gains favor high-variance splits that overfit individual docs; Cosine suppresses these.

This script:
1. Trains CPU Depthwise with Cosine (default) and with L2 score function at N=1000
2. Trains CPU ST with Cosine (default) at N=1000 as control
3. Compares RMSE to isolate the score function as the causal mechanism
4. Computes a per-partition score-function-difference metric at iter=0 to quantify effect

Key question: Does forcing CPU DW to use L2 score function reproduce MLX's lower RMSE?
If yes, the mechanism is confirmed: MLX's L2 gain = CPU's L2 score function.
"""

import os
import sys
import json
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03
N = 1000

SEEDS = [1337, 42, 7]


def make_xy(N, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(N).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    return X, y


from catboost import CatBoostRegressor

results_step3 = []

print("=" * 90)
print("STEP 3 — Score function analysis: Cosine vs L2 for DW N=1000")
print("=" * 90)
print(f"\nHypothesis: CPU uses score_function=Cosine; MLX uses L2 Newton gain.")
print(f"If CPU DW with L2 score function ≈ MLX DW RMSE, mechanism is confirmed.\n")

print(f"{'Policy':>13s} {'score_fn':>8s} {'seed':>5s} {'rs':>4s} | {'RMSE':>10s} | {'vs_CPU_cosine':>14s}")
print("-" * 70)

for seed in SEEDS:
    X, y = make_xy(N, seed)

    for rs in [0.0]:  # rs=0 for deterministic comparison
        # CPU Depthwise with Cosine (default)
        m_cos = CatBoostRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LR,
            loss_function="RMSE", grow_policy="Depthwise", max_bin=BINS,
            random_seed=seed, random_strength=rs,
            bootstrap_type="No",
            score_function="Cosine",
            verbose=0, thread_count=1,
        )
        m_cos.fit(X, y)
        rmse_cosine = float(m_cos.evals_result_["learn"]["RMSE"][-1])

        # CPU Depthwise with L2 score function (what MLX implements)
        m_l2 = CatBoostRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LR,
            loss_function="RMSE", grow_policy="Depthwise", max_bin=BINS,
            random_seed=seed, random_strength=rs,
            bootstrap_type="No",
            score_function="L2",
            verbose=0, thread_count=1,
        )
        m_l2.fit(X, y)
        rmse_l2 = float(m_l2.evals_result_["learn"]["RMSE"][-1])

        ratio_l2_vs_cos = rmse_l2 / rmse_cosine
        print(f"{'Depthwise':>13s} {'Cosine':>8s} {seed:>5d} {rs:>4.1f} | {rmse_cosine:>10.6f} | {'(baseline)':>14s}")
        print(f"{'Depthwise':>13s} {'L2':>8s} {seed:>5d} {rs:>4.1f} | {rmse_l2:>10.6f} | {ratio_l2_vs_cos:>14.4f}")
        print()
        sys.stdout.flush()

        results_step3.append({
            "policy": "Depthwise", "score_fn": "Cosine", "seed": seed, "rs": rs,
            "rmse": rmse_cosine,
        })
        results_step3.append({
            "policy": "Depthwise", "score_fn": "L2", "seed": seed, "rs": rs,
            "rmse": rmse_l2, "ratio_vs_cosine": ratio_l2_vs_cos,
        })

print("-" * 70)
print("\nSummary (DW rs=0, score function effect):")
for seed in SEEDS:
    cosine_rmse = next(r["rmse"] for r in results_step3
                       if r["score_fn"] == "Cosine" and r["seed"] == seed and r["rs"] == 0.0)
    l2_rmse = next(r["rmse"] for r in results_step3
                   if r["score_fn"] == "L2" and r["seed"] == seed and r["rs"] == 0.0)
    ratio = l2_rmse / cosine_rmse
    print(f"  seed={seed}: CPU_Cosine={cosine_rmse:.6f} CPU_L2={l2_rmse:.6f} ratio={ratio:.4f}")

# Comparison: CPU L2 should match MLX DW RMSE if mechanism is score function
# Pre-collected MLX values from g1-results.md (post-FU-1 = identical to pre-FU-1)
mlx_values = {1337: 0.179724, 42: 0.181591, 7: 0.179449}
print()
print("Score function mechanism test (MLX vs CPU-L2 vs CPU-Cosine):")
print(f"  {'seed':>5s} | {'CPU_Cosine':>11s} | {'CPU_L2':>11s} | {'MLX':>11s} | "
      f"{'MLX/CPU_L2':>12s} | {'MLX/CPU_Cos':>13s}")
print("  " + "-" * 80)
for seed in SEEDS:
    cpu_cos = next(r["rmse"] for r in results_step3
                   if r["score_fn"] == "Cosine" and r["seed"] == seed and r["rs"] == 0.0)
    cpu_l2 = next(r["rmse"] for r in results_step3
                  if r["score_fn"] == "L2" and r["seed"] == seed and r["rs"] == 0.0)
    mlx = mlx_values[seed]
    print(f"  {seed:>5d} | {cpu_cos:>11.6f} | {cpu_l2:>11.6f} | {mlx:>11.6f} | "
          f"{mlx/cpu_l2:>12.4f} | {mlx/cpu_cos:>13.4f}")

# Also test ST with L2 vs Cosine to confirm ST is less sensitive
print()
print("Control: ST N=1000 score function effect (shows insensitivity vs DW)")
print(f"{'Policy':>13s} {'score_fn':>8s} {'seed':>5s} | {'RMSE':>10s}")
print("-" * 50)
for seed in SEEDS:
    X, y = make_xy(N, seed)
    for sfn in ["Cosine", "L2"]:
        m = CatBoostRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LR,
            loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
            random_seed=seed, random_strength=0.0,
            bootstrap_type="No",
            score_function=sfn,
            verbose=0, thread_count=1,
        )
        m.fit(X, y)
        rmse = float(m.evals_result_["learn"]["RMSE"][-1])
        results_step3.append({
            "policy": "SymmetricTree", "score_fn": sfn, "seed": seed, "rs": 0.0,
            "rmse": rmse,
        })
        print(f"{'SymmetricTree':>13s} {sfn:>8s} {seed:>5d} | {rmse:>10.6f}")
    print()

print("-" * 50)
print("ST Cosine vs L2 ratios (should be near 1.0 — ST insensitive to score_fn):")
for seed in SEEDS:
    st_cos = next(r["rmse"] for r in results_step3
                  if r["policy"] == "SymmetricTree" and r["score_fn"] == "Cosine"
                  and r["seed"] == seed)
    st_l2 = next(r["rmse"] for r in results_step3
                 if r["policy"] == "SymmetricTree" and r["score_fn"] == "L2"
                 and r["seed"] == seed)
    print(f"  seed={seed}: Cosine={st_cos:.6f} L2={st_l2:.6f} ratio={st_l2/st_cos:.4f}")

# Save
out_path = (
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/"
    "docs/sprint27/scratch/fu3-t1-instrumentation/step3_score_function_results.json"
)
with open(out_path, "w") as f:
    json.dump(results_step3, f, indent=2)
print(f"\nResults written to: {out_path}")
