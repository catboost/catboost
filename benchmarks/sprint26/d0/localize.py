"""Pre-sprint localization: does MLX oblivious match CPU at this config?

If YES (MLX oblivious ≈ CPU): the harness is sound; Depthwise is the anomaly → sprint scope expands to include a correctness check as Day 0-e before timing work.

If NO (MLX oblivious also diverges): the subprocess harness itself has an issue (data encoding, binary path, param passing) → sprint Day 0 must debug the harness before anything else.
"""
import os, time, struct
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

N, FEATURES, BINS, ITERS, DEPTH, SEED, LR = 10_000, 20, 128, 50, 6, 1337, 0.03
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, FEATURES)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)

def fhex(x): return struct.pack(">f", float(x)).hex()

from catboost import CatBoostRegressor
from catboost_mlx import CatBoostMLXRegressor

def cpu_loss(grow):
    m = CatBoostRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss_function="RMSE", grow_policy=grow, max_bin=BINS,
        random_seed=SEED, verbose=0, thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])

def mlx_loss(grow):
    m = CatBoostMLXRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss="rmse", grow_policy=grow, bins=BINS,
        random_seed=SEED, verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(X, y)
    t1 = time.perf_counter()
    preds = m.predict(X)
    return float(np.sqrt(((preds - y) ** 2).mean())), t1 - t0

print("policy         | CPU RMSE          | MLX pred RMSE     | delta       | delta %   | MLX time(s)")
print("-" * 100)
for grow in ("SymmetricTree", "Depthwise", "Lossguide"):
    cpu = cpu_loss(grow)
    mlx, t = mlx_loss(grow)
    delta = abs(mlx - cpu)
    pct = delta / cpu * 100
    print(f"{grow:14s} | {cpu:.10f}  hex={fhex(cpu)} | {mlx:.10f}  hex={fhex(mlx)} | {delta:.3e} | {pct:7.2f}% | {t:.2f}")
