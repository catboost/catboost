"""DEC-029 single-tree Depthwise/Lossguide diagnostic.

Fit a 1-iteration tree with LR=1, rs=0 (deterministic) for each grow policy.
At LR=1, 1-tree, rs=0: any prediction divergence is 100% attributable to tree
structure or leaf assignment — gradient/noise/model-export paths, not iteration
compounding.

Usage:
    python benchmarks/sprint26/d0/one_tree_depthwise.py

Expected output (post-fix):
    SymmetricTree:  MLX pred_std / CPU pred_std ≈ 1.0  (control)
    Depthwise:      MLX pred_std / CPU pred_std ≈ 1.0
    Lossguide:      MLX pred_std / CPU pred_std ≈ 1.0
    RMSE (1-iter, depth=6): all policies within 5% of CPU
"""

import numpy as np
import json
from catboost import CatBoostRegressor
from catboost_mlx import CatBoostMLXRegressor

N, FEATURES, BINS, SEED = 10_000, 20, 128, 1337
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, FEATURES)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)

LR = 1.0
RS = 0.0   # deterministic
DEPTH_SMALL = 2   # 4 leaves — easy to manually verify
DEPTH_LARGE = 6   # matches localize.py


def run_comparison(grow_policy: str, depth: int, label: str = ""):
    tag = f"{grow_policy} depth={depth}{(' ' + label) if label else ''}"

    # CPU CatBoost
    cpu_kw = dict(iterations=1, depth=depth, learning_rate=LR,
                  loss_function="RMSE", grow_policy=grow_policy,
                  max_bin=BINS, random_seed=SEED, random_strength=RS,
                  verbose=0, thread_count=1)
    cpu_m = CatBoostRegressor(**cpu_kw)
    cpu_m.fit(X, y)
    cpu_pred = cpu_m.predict(X)

    # MLX CatBoost
    mlx_m = CatBoostMLXRegressor(
        iterations=1, depth=depth, learning_rate=LR,
        loss="rmse", grow_policy=grow_policy,
        bins=BINS, random_seed=SEED, random_strength=RS,
        verbose=False,
    )
    mlx_m.fit(X, y)
    mlx_pred = mlx_m.predict(X)

    cpu_std = float(np.std(cpu_pred))
    mlx_std = float(np.std(mlx_pred))
    ratio = mlx_std / cpu_std if cpu_std > 1e-9 else float("nan")

    cpu_rmse = float(np.sqrt(np.mean((cpu_pred - y) ** 2)))
    mlx_rmse = float(np.sqrt(np.mean((mlx_pred - y) ** 2)))
    delta_pct = abs(mlx_rmse - cpu_rmse) / max(cpu_rmse, 1e-9) * 100

    print(f"\n--- {tag} ---")
    print(f"  CPU pred: mean={np.mean(cpu_pred):.4f}  std={cpu_std:.4f}  "
          f"min={np.min(cpu_pred):.4f}  max={np.max(cpu_pred):.4f}")
    print(f"  MLX pred: mean={np.mean(mlx_pred):.4f}  std={mlx_std:.4f}  "
          f"min={np.min(mlx_pred):.4f}  max={np.max(mlx_pred):.4f}")
    print(f"  std ratio (MLX/CPU) = {ratio:.4f}  (1.0 = identical spread)")
    print(f"  CPU RMSE = {cpu_rmse:.6f}  MLX RMSE = {mlx_rmse:.6f}  delta = {delta_pct:.2f}%")

    # Sample-by-sample comparison (first 10 docs)
    print(f"  Sample predictions (doc 0..9):")
    for i in range(min(10, N)):
        diff = mlx_pred[i] - cpu_pred[i]
        print(f"    doc[{i:3d}]: cpu={cpu_pred[i]:.6f}  mlx={mlx_pred[i]:.6f}  diff={diff:+.6f}")

    # Leaf values from model JSON
    model_data = mlx_m._model_data  # already a dict after nanobind path
    tree0 = model_data["trees"][0]
    gp = tree0.get("grow_policy", "SymmetricTree")
    splits = tree0.get("splits", [])
    leaf_values = tree0.get("leaf_values", [])
    leaf_bfs_ids = tree0.get("leaf_bfs_ids", [])

    print(f"\n  MLX tree JSON summary:")
    print(f"    grow_policy = {gp!r}")
    print(f"    depth = {tree0.get('depth', '?')}")
    print(f"    #splits = {len(splits)}")
    print(f"    #leaf_values = {len(leaf_values)}")
    if leaf_bfs_ids:
        print(f"    leaf_bfs_ids = {leaf_bfs_ids}")
    if splits:
        print(f"    splits[0] = {splits[0]}")
    if leaf_values:
        print(f"    leaf_values (first 8) = {[f'{v:.6f}' for v in leaf_values[:8]]}")

    # Leaf values from CPU
    if grow_policy == "SymmetricTree" and depth <= 6:
        try:
            cpu_leaves = cpu_m.get_leaf_values(0)  # CatBoost API
            print(f"  CPU leaf_values (first 8) = {[f'{v:.6f}' for v in list(cpu_leaves)[:8]]}")
        except Exception:
            pass

    return ratio, delta_pct


print("=" * 70)
print("DEC-029 single-tree Depthwise/Lossguide diagnostic")
print("=" * 70)
print(f"N={N}  FEATURES={FEATURES}  BINS={BINS}  LR={LR}  RS={RS}  SEED={SEED}")

results = {}
for policy in ("SymmetricTree", "Depthwise", "Lossguide"):
    for depth in (DEPTH_SMALL, DEPTH_LARGE):
        ratio, dpct = run_comparison(policy, depth)
        results[f"{policy}_d{depth}"] = (ratio, dpct)

print("\n" + "=" * 70)
print("Summary: std ratio (MLX/CPU) and RMSE delta %")
print("  policy         depth  std_ratio  rmse_delta%")
for k, (r, d) in results.items():
    print(f"  {k:20s}  {r:.4f}     {d:.2f}%")

print("\n[PASS] if all std_ratio ≈ 1.0 and rmse_delta < 5%")
print("[FAIL] if std_ratio << 1.0 (all docs same leaf) or >> 1.0 (overfit)")
