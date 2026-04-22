"""Rule out Bayesian bootstrap mismatch as the cause of MLX shrinkage.

MLX defaults to bootstrap_type='no'; CPU CatBoostRegressor with no explicit
bootstrap_type defaults to Bayesian (per CatBoost docs). If CPU with
bootstrap_type='No' still converges to ~0.20 vs MLX's 0.34, we've ruled out
sampling as the cause and confirmed a leaf-magnitude bug.
"""
import os
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

N, FEATURES, BINS, ITERS, DEPTH, SEED, LR = 10_000, 20, 128, 50, 6, 1337, 0.03
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, FEATURES)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)

from catboost import CatBoostRegressor

for bt in ("Bayesian", "No"):
    m = CatBoostRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
        random_seed=SEED, verbose=0, thread_count=1,
        bootstrap_type=bt,
    )
    m.fit(X, y)
    preds = m.predict(X)
    loss = float(m.evals_result_["learn"]["RMSE"][-1])
    print(f"CPU bootstrap_type={bt:10s} final RMSE={loss:.4f}  pred std={preds.std():.4f}")
