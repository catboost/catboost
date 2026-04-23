"""S26 pre-sprint seed/size sweep: is the SymmetricTree MLX-vs-CPU delta
constant across seeds/sizes? Narrows 'parameter defaults mismatch' vs
'real correctness bug'.

- Constant delta % across seeds/sizes -> systematic offset (parameter mapping)
- Seed-variant  -> randomness handling differs (bootstrap/subsample)
- Size-variant  -> binning/precision issue that scales with N
"""
import os, struct, time
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

FEATURES, BINS, ITERS, DEPTH, LR = 20, 128, 50, 6, 0.03
SEEDS = [1337, 42, 7, 99]
SIZES = [1000, 10000, 50000]

from catboost import CatBoostRegressor
from catboost_mlx import CatBoostMLXRegressor


def fhex(x):
    return struct.pack(">f", float(x)).hex()


def make_xy(N, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, FEATURES)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)
    return X, y


def cpu_loss(X, y, seed):
    m = CatBoostRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
        random_seed=seed, verbose=0, thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def mlx_loss(X, y, seed):
    m = CatBoostMLXRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss="rmse", grow_policy="SymmetricTree", bins=BINS,
        random_seed=seed, verbose=False,
    )
    m.fit(X, y)
    preds = m.predict(X)
    return float(np.sqrt(((preds - y) ** 2).mean()))


print(f"{'N':>7s} {'seed':>5s} {'CPU RMSE':>12s} {'MLX RMSE':>12s} {'delta':>10s} {'delta%':>8s}")
print("-" * 62)
t_start = time.perf_counter()
for N in SIZES:
    for seed in SEEDS:
        X, y = make_xy(N, seed)
        cpu = cpu_loss(X, y, seed)
        mlx = mlx_loss(X, y, seed)
        delta = abs(mlx - cpu)
        pct = delta / cpu * 100 if cpu > 0 else 0.0
        print(f"{N:>7d} {seed:>5d} {cpu:>12.6f} {mlx:>12.6f} {delta:>10.3e} {pct:>7.2f}%")
print(f"\nelapsed: {time.perf_counter() - t_start:.1f}s")
