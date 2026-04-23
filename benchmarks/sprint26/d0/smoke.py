"""S26 pre-sprint smoke: D0-b (CPU Depthwise determinism) + D0-c (MLX Depthwise feasibility).

Small scale — feasibility proof only. Not a gate measurement.
"""
import os
import sys
import time
import struct
import numpy as np

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

N = 10_000
FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
SEED = 1337
LR = 0.03

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, FEATURES)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)

common = dict(
    iterations=ITERS,
    depth=DEPTH,
    learning_rate=LR,
    loss_function="RMSE",
    grow_policy="Depthwise",
    max_bin=BINS,
    random_seed=SEED,
    verbose=0,
    thread_count=1,
)


def fhex(x):
    return struct.pack(">f", float(x)).hex()


print("=== D0-b: CPU CatBoost Depthwise determinism (3x same seed) ===")
from catboost import CatBoostRegressor

cpu_losses = []
cpu_times = []
for r in range(3):
    t0 = time.perf_counter()
    m = CatBoostRegressor(**common)
    m.fit(X, y)
    t1 = time.perf_counter()
    loss = float(m.evals_result_["learn"]["RMSE"][-1])
    cpu_losses.append(loss)
    cpu_times.append(t1 - t0)
    print(f"  run{r}: loss={loss:.10f}  hex={fhex(loss)}  t={t1-t0:.3f}s")

cpu_hexes = {fhex(l) for l in cpu_losses}
print(f"  deterministic (ULP=0): {'YES' if len(cpu_hexes)==1 else 'NO'}")
print(f"  max(loss) - min(loss) = {max(cpu_losses) - min(cpu_losses):.3e}")
print()

print("=== D0-c: CatBoost-MLX Depthwise feasibility (1x) ===")
mlx_kw = dict(
    iterations=ITERS,
    depth=DEPTH,
    learning_rate=LR,
    loss="rmse",
    grow_policy="Depthwise",
    bins=BINS,
    random_seed=SEED,
    verbose=False,
)
try:
    from catboost_mlx import CatBoostMLXRegressor
    t0 = time.perf_counter()
    m = CatBoostMLXRegressor(**mlx_kw)
    m.fit(X, y)
    t1 = time.perf_counter()
    preds = m.predict(X)
    mlx_loss = float(np.sqrt(((preds - y) ** 2).mean()))
    print(f"  run0: pred-based RMSE={mlx_loss:.10f}  hex={fhex(mlx_loss)}  t={t1-t0:.3f}s")
    print(f"  MLX vs CPU(run0) delta: {abs(mlx_loss - cpu_losses[0]):.3e}  ({abs(mlx_loss - cpu_losses[0])/cpu_losses[0]*100:.2f}%)")
    print(f"  feasibility: END-TO-END RAN (no crash)")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    raise

print()
print("=== Summary ===")
print(f"  CPU Depthwise det (3 runs): {'ULP=0 YES' if len(cpu_hexes)==1 else f'NO — {len(cpu_hexes)} distinct values'}")
print(f"  MLX Depthwise feasibility : PASS")
print(f"  CPU mean iter time (10k)  : {(sum(cpu_times)/len(cpu_times))/ITERS*1000:.2f} ms/iter")
