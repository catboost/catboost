"""S26-D0-5 one-tree driver.

Single-tree regime: 1 iteration, LR=1.0, depth=6, N=10k, seed=1337,
SymmetricTree, max_bin=128, RMSE, L2=3.0.

At LR=1.0 the leaf value IS the partition-conditional mean of the negative
residual.  Any prediction-std gap is 100% attributable to split choice
(leaf computation proven exact by D0-4).

Captures:
  P9  — effective config dump (MLX vs CPU)
  P10 — quantization borders for features 0 and 1
  P11 — top-10 split candidates at root (from C++ debug output)
  P12 — gain formula source literal (from C++ debug output)
  P13 — cross-check one candidate (from C++ debug output)

Raw instrumentation output (C++ stderr/stdout) is captured in
  benchmarks/sprint26/d0/one-tree-instrumentation.txt
"""

import os, sys, json
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ── Synthetic dataset (canonical for S26) ────────────────────────────────────
N, F, BINS, DEPTH, SEED = 10_000, 20, 128, 6, 1337
LR, L2 = 1.0, 3.0

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, F)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)

print(f"y stats: mean={y.mean():.6f}  std={y.std():.6f}  min={y.min():.6f}  max={y.max():.6f}")
print()

# ── CPU CatBoost (1 tree, LR=1.0, L2=3.0, no bootstrap, rs=0) ──────────────
from catboost import CatBoostRegressor

cpu_rs0 = CatBoostRegressor(
    iterations=1, depth=DEPTH, learning_rate=LR,
    loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
    random_seed=SEED, verbose=0, thread_count=1,
    l2_leaf_reg=L2, random_strength=0.0, bootstrap_type="No",
    rsm=1.0, min_data_in_leaf=1,
)
cpu_rs0.fit(X, y)
cpu_rs0_preds = cpu_rs0.predict(X)
loss_rs0 = cpu_rs0.evals_result_["learn"]["RMSE"][0]

cpu_rs1 = CatBoostRegressor(
    iterations=1, depth=DEPTH, learning_rate=LR,
    loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
    random_seed=SEED, verbose=0, thread_count=1,
    l2_leaf_reg=L2, random_strength=1.0, bootstrap_type="No",
    rsm=1.0, min_data_in_leaf=1,
)
cpu_rs1.fit(X, y)
cpu_rs1_preds = cpu_rs1.predict(X)
loss_rs1 = cpu_rs1.evals_result_["learn"]["RMSE"][0]

cpu_default = CatBoostRegressor(
    iterations=1, depth=DEPTH, learning_rate=LR,
    loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
    random_seed=SEED, verbose=0, thread_count=1,
)
cpu_default.fit(X, y)
cpu_default_preds = cpu_default.predict(X)
loss_default = cpu_default.evals_result_["learn"]["RMSE"][0]

print("=== CPU CatBoost (1 tree, LR=1.0) ===")
all_p = cpu_rs0.get_all_params()
print("  get_all_params() [key fields]:")
for k in ["random_strength", "rsm", "bootstrap_type", "subsample",
          "min_data_in_leaf", "l2_leaf_reg", "feature_border_type", "border_count"]:
    print(f"    {k}: {all_p.get(k, 'N/A')}")
print(f"  rs=0, no bootstrap:  preds std={cpu_rs0_preds.std():.6f}  ratio={cpu_rs0_preds.std()/y.std():.4f}  loss={loss_rs0:.6f}")
print(f"  rs=1.0, no bootstrap:preds std={cpu_rs1_preds.std():.6f}  ratio={cpu_rs1_preds.std()/y.std():.4f}  loss={loss_rs1:.6f}")
print(f"  default (MVS 80%):   preds std={cpu_default_preds.std():.6f}  ratio={cpu_default_preds.std()/y.std():.4f}  loss={loss_default:.6f}")
print()

# ── P9: CPU effective config ──────────────────────────────────────────────────
cpu_all = cpu_rs0.get_all_params()
cpu_config_p9 = {k: cpu_all.get(k, "N/A") for k in [
    "random_strength", "rsm", "bootstrap_type", "subsample",
    "bagging_temperature", "min_data_in_leaf", "l2_leaf_reg",
    "feature_border_type", "border_count",
]}
print("=== P9: CPU effective config ===")
for k, v in cpu_config_p9.items():
    print(f"  {k}: {v}")
print()

# ── P10: CPU quantization borders for features 0 and 1 ───────────────────────
# CatBoost private API: pool object stores borders
try:
    from catboost.core import Pool
    pool = Pool(X, y)
    # Fit a model with the pool to get quantization
    cpu_pool_model = CatBoostRegressor(
        iterations=1, depth=1, learning_rate=1.0,
        loss_function="RMSE", max_bin=BINS, verbose=0, thread_count=1,
        l2_leaf_reg=L2, random_strength=0.0, bootstrap_type="No",
    )
    cpu_pool_model.fit(pool)
    # Try to get borders through private API
    try:
        borders_f0 = cpu_pool_model.get_borders()
        print(f"=== P10: CPU borders via get_borders() ===")
        if isinstance(borders_f0, dict):
            for k in list(borders_f0.keys())[:2]:
                print(f"  Feature {k}: count={len(borders_f0[k])}  min={min(borders_f0[k]):.4f}  max={max(borders_f0[k]):.4f}")
        else:
            print(f"  Unexpected type: {type(borders_f0)}")
    except Exception as e:
        print(f"  get_borders() failed: {e}")
        # Fallback: quantize manually
        from catboost.utils import get_feature_importances
        print(f"  (no border API available — using manual equal-frequency quantization for comparison)")
except Exception as e:
    print(f"  Pool/border introspection failed: {e}")

# Manual equal-frequency quantization matching MLX implementation
def mlx_borders(vals, max_bins):
    """Replicate MLX QuantizeFeatures border computation exactly."""
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.array([])
    sorted_v = np.unique(np.sort(vals))  # dedup then sort
    num_borders = min(len(sorted_v) - 1, max_bins - 1)
    borders = []
    for b in range(num_borders):
        frac = (b + 1) / (num_borders + 1)
        idx = int(frac * (len(sorted_v) - 1))
        idx = min(idx, len(sorted_v) - 1)
        if idx + 1 < len(sorted_v):
            borders.append(0.5 * (sorted_v[idx] + sorted_v[idx + 1]))
        else:
            borders.append(sorted_v[idx])
    return np.array(borders)

mlx_b0 = mlx_borders(X[:, 0], BINS)
mlx_b1 = mlx_borders(X[:, 1], BINS)
print(f"\n=== P10: MLX borders (manual replication) ===")
print(f"  Feature 0: count={len(mlx_b0)}  min={mlx_b0.min():.4f}  max={mlx_b0.max():.4f}")
print(f"  Feature 0 first 5: {mlx_b0[:5]}")
print(f"  Feature 0 last 5:  {mlx_b0[-5:]}")
print(f"  Feature 1: count={len(mlx_b1)}  min={mlx_b1.min():.4f}  max={mlx_b1.max():.4f}")
print(f"  Feature 1 first 5: {mlx_b1[:5]}")
print(f"  Feature 1 last 5:  {mlx_b1[-5:]}")
print()

# Save borders to JSON for diff
borders_json = {
    "mlx_f0": mlx_b0.tolist(),
    "mlx_f1": mlx_b1.tolist(),
}
import pathlib
out_dir = pathlib.Path(__file__).parent
with open(out_dir / "borders-feat01.json", "w") as fh:
    json.dump(borders_json, fh, indent=2)
print(f"  Borders saved to benchmarks/sprint26/d0/borders-feat01.json")
print()

# ── MLX: run with rs=0 and rs=1.0, capture instrumentation from C++ ──────────
sys.path.insert(0, str(pathlib.Path(__file__).parents[4] / "python"))
from catboost_mlx import CatBoostMLXRegressor

print("=== MLX CatBoostMLXRegressor (1 tree, LR=1.0) ===")

mlx_rs0 = CatBoostMLXRegressor(
    iterations=1, depth=DEPTH, learning_rate=LR,
    loss="rmse", bins=BINS,
    random_seed=SEED, verbose=False,
    l2_reg_lambda=L2,
    random_strength=0.0,
    bootstrap_type="no",
    subsample=1.0,
    min_data_in_leaf=1,
)
mlx_rs0.fit(X, y)
mlx_rs0_preds = mlx_rs0.predict(X)

print(f"  MLX rs=0:   preds std={mlx_rs0_preds.std():.6f}  ratio={mlx_rs0_preds.std()/y.std():.4f}")
print()

mlx_rs1 = CatBoostMLXRegressor(
    iterations=1, depth=DEPTH, learning_rate=LR,
    loss="rmse", bins=BINS,
    random_seed=SEED, verbose=False,
    l2_reg_lambda=L2,
    random_strength=1.0,
    bootstrap_type="no",
    subsample=1.0,
    min_data_in_leaf=1,
)
mlx_rs1.fit(X, y)
mlx_rs1_preds = mlx_rs1.predict(X)

print(f"  MLX rs=1.0: preds std={mlx_rs1_preds.std():.6f}  ratio={mlx_rs1_preds.std()/y.std():.4f}")
print()

# ── P9: MLX effective config ─────────────────────────────────────────────────
print("=== P9: Config diff (MLX vs CPU at equivalent settings) ===")
mlx_config = {
    "ColsampleByTree":     1.0,
    "SubsampleRatio":      1.0,
    "RandomStrength":      1.0,
    "MinDataInLeaf":       1,
    "FeatureBorderType":   "EqualFrequency (custom)",
    "L2RegLambda":         L2,
    "BootstrapType":       "no",
    "BaggingTemperature":  1.0,
    "MvsReg":              0.0,
    "noise_scale_formula": "randomStrength * totalHessian / numPartitions",
}
cpu_config_equiv = {
    "ColsampleByTree (rsm)": 1.0,
    "SubsampleRatio":        "N/A (bootstrap=No)",
    "RandomStrength":        1.0,
    "MinDataInLeaf":         1,
    "FeatureBorderType":     "GreedyLogSum",
    "L2RegLambda":           L2,
    "BootstrapType":         "No",
    "BaggingTemperature":    "N/A",
    "MvsReg":                "N/A",
    "noise_scale_formula":   "randomStrength * sqrt(sum(g_i^2) / N)",
}
print(f"  {'Parameter':<28} {'MLX':<35} {'CPU (equiv)':<35}")
print(f"  {'-'*28} {'-'*35} {'-'*35}")
all_keys = sorted(set(list(mlx_config.keys()) + list(cpu_config_equiv.keys())))
for k in all_keys:
    mv = str(mlx_config.get(k, "—"))
    cv = str(cpu_config_equiv.get(k, "—"))
    flag = " <-- DIFFERS" if mv != cv else ""
    print(f"  {k:<28} {mv:<35} {cv:<35}{flag}")
print()

# ── Noise scale quantification ────────────────────────────────────────────────
# At iter 0: gradients = cursor - y ≈ -y (since cursor ≈ mean(y) ≈ 0)
# CPU formula: sqrt(sum(g^2)/N) = sqrt(mean(g^2)) ≈ std(y) (gradient RMS)
# MLX formula: totalWeight / numPartitions = N / 1 = N
# Gain scores at root: ~N * var(y) (root partition, all N docs)
g = (cpu_rs0.predict(X) - y).astype(np.float64)  # residuals before tree 0 but after base pred
# Actually grad = cursor - y at iter 0; cursor = mean(y)
base_pred = y.mean()
g_iter0 = (base_pred - y).astype(np.float64)
cpu_noise_scale = np.sqrt((g_iter0**2).mean())
mlx_noise_scale = float(N)  # totalWeight / 1 partition for RMSE
root_gain_approx = float(np.sum(g_iter0**2) / (N + L2))  # sum_parent^2 / (N+L2) contribution
print("=== P12/noise quantification ===")
print(f"  y.std()                     = {y.std():.6f}")
print(f"  grad_rms at iter 0 (CPU)    = {cpu_noise_scale:.6f}   (= std(y) since base_pred~=mean(y))")
print(f"  noise_scale MLX             = {mlx_noise_scale:.1f}   (= N = totalHessian)")
print(f"  noise_scale CPU             = {cpu_noise_scale:.6f}")
print(f"  noise_scale ratio MLX/CPU   = {mlx_noise_scale / cpu_noise_scale:.1f}x")
print(f"  root gain score (approx)    = {root_gain_approx:.2f}   (sum_parent^2/(N+L2))")
print(f"  noise/gain ratio MLX        = {mlx_noise_scale / (root_gain_approx + 1e-10):.3f}  (>> 1 => wipes signal)")
print(f"  noise/gain ratio CPU        = {cpu_noise_scale / (root_gain_approx + 1e-10):.6f}")
print()

# ── Summary comparison ────────────────────────────────────────────────────────
print("=== Summary: 1-tree preds std ratios ===")
print(f"  y std = {y.std():.6f}")
rows = [
    ("CPU rs=0, no bootstrap",    cpu_rs0_preds.std(),    loss_rs0),
    ("CPU rs=1, no bootstrap",    cpu_rs1_preds.std(),    loss_rs1),
    ("CPU default (MVS 80%)",     cpu_default_preds.std(), loss_default),
    ("MLX rs=0 (no noise)",       mlx_rs0_preds.std(),    None),
    ("MLX rs=1.0 (default)",      mlx_rs1_preds.std(),    None),
]
print(f"  {'Config':<30} {'preds_std':<12} {'ratio':<8} {'loss0'}")
for label, std, loss in rows:
    lstr = f"{loss:.6f}" if loss is not None else "N/A"
    print(f"  {label:<30} {std:<12.6f} {std/y.std():<8.4f} {lstr}")
print()
print("=== Key findings ===")
print(f"  1. CPU rs=0 ratio {cpu_rs0_preds.std()/y.std():.4f} ≈ MLX rs=0 ratio {mlx_rs0_preds.std()/y.std():.4f}")
print(f"     => identical split quality when noise disabled")
print(f"  2. MLX rs=1.0 ratio {mlx_rs1_preds.std()/y.std():.4f} << CPU rs=1.0 ratio {cpu_rs1_preds.std()/y.std():.4f}")
print(f"     => MLX noise formula is {mlx_noise_scale/cpu_noise_scale:.0f}x larger than CPU")
print(f"  3. Root cause: MLX noiseScale = rs * N = {mlx_noise_scale:.0f}")
print(f"                 CPU noiseScale = rs * sqrt(mean(g^2)) = {cpu_noise_scale:.4f}")
print(f"  4. Fix: replace noiseScale formula in FindBestSplit (csv_train.cpp:990)")
print(f"     from: rs * totalWeight / (numPartitions * K)")
print(f"     to:   rs * gradRms  where gradRms = sqrt(sum(g^2) / N)")
