"""S26 pre-sprint dissection: does MLX training converge internally,
or does predict() fail? Also check prediction statistics.

- MLX train loss history falls => predict() is broken (leaf eval / offset)
- MLX train loss history stuck => training itself doesn't converge
- MLX predictions constant => leaf collapse
- MLX predictions varying, wrong scale => learning rate / base pred issue
"""
import os, struct
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

N, FEATURES, BINS, ITERS, DEPTH, SEED, LR = 10_000, 20, 128, 50, 6, 1337, 0.03
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, FEATURES)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)

print(f"y stats: mean={y.mean():.4f} std={y.std():.4f} min={y.min():.4f} max={y.max():.4f}")
print()

from catboost import CatBoostRegressor
from catboost_mlx import CatBoostMLXRegressor

# CPU
print("=== CPU CatBoost SymmetricTree ===")
cpu = CatBoostRegressor(
    iterations=ITERS, depth=DEPTH, learning_rate=LR,
    loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
    random_seed=SEED, verbose=0, thread_count=1,
)
cpu.fit(X, y)
cpu_preds = cpu.predict(X)
cpu_losses = cpu.evals_result_["learn"]["RMSE"]
print(f"  train loss history: [0]={cpu_losses[0]:.4f} [10]={cpu_losses[10]:.4f} "
      f"[25]={cpu_losses[25]:.4f} [-1]={cpu_losses[-1]:.4f}")
print(f"  preds stats: mean={cpu_preds.mean():.4f} std={cpu_preds.std():.4f} "
      f"min={cpu_preds.min():.4f} max={cpu_preds.max():.4f}")
print(f"  pred-based RMSE: {np.sqrt(((cpu_preds - y)**2).mean()):.4f}")
print()

# MLX
print("=== MLX CatBoost SymmetricTree ===")
mlx = CatBoostMLXRegressor(
    iterations=ITERS, depth=DEPTH, learning_rate=LR,
    loss="rmse", grow_policy="SymmetricTree", bins=BINS,
    random_seed=SEED, verbose=False,
)
mlx.fit(X, y)
mlx_preds = mlx.predict(X)
mlx_losses = mlx._train_loss_history
print(f"  train loss history length: {len(mlx_losses)}")
if mlx_losses:
    idxs = [0, min(10, len(mlx_losses)-1), min(25, len(mlx_losses)-1), -1]
    s = " ".join(f"[{i}]={mlx_losses[i]:.4f}" for i in idxs)
    print(f"  train loss: {s}")
print(f"  preds stats: mean={mlx_preds.mean():.4f} std={mlx_preds.std():.4f} "
      f"min={mlx_preds.min():.4f} max={mlx_preds.max():.4f}")
print(f"  pred-based RMSE: {np.sqrt(((mlx_preds - y)**2).mean()):.4f}")
print()

# Prediction distribution sanity
print("=== Prediction correlation (CPU vs MLX) ===")
corr = np.corrcoef(cpu_preds, mlx_preds)[0, 1]
print(f"  Pearson(CPU_pred, MLX_pred): {corr:.4f}")
print(f"  Pearson(CPU_pred, y):        {np.corrcoef(cpu_preds, y)[0,1]:.4f}")
print(f"  Pearson(MLX_pred, y):        {np.corrcoef(mlx_preds, y)[0,1]:.4f}")
print(f"  ratio MLX_std / CPU_std:     {mlx_preds.std()/cpu_preds.std():.4f}")
print(f"  bias (MLX_mean - CPU_mean):  {mlx_preds.mean() - cpu_preds.mean():.4f}")
