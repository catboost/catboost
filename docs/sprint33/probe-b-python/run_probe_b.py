#!/usr/bin/env python3
"""
PROBE-B: Verify whether the nanobind Python path exhibits the same
52.6% iter=50 ST+Cosine drift as csv_train.cpp CLI, OR whether (per L4
verdict Option 3) it uses a different "production" quantization path
that gives clean parity.

Anchor dataset matches docs/sprint33/l4-fix/run_phase1.py exactly:
  N=50000, 20 features, X ~ N(0,1), y = 0.5 X[:,0] + 0.3 X[:,1] + 0.1 N(0,1).
  Config: iters=50 (and 100 for ratio table), depth=6, bins=127,
  lr=0.03, Cosine, RMSE, ST, bootstrap=No, RS=0, has_time=True, seed=42.

Both Python guards (Python core.py + C++ train_api.cpp) are bypassed:
  - Python via monkey-patching `_validate_params`.
  - C++ via a probe-mode rebuild with `-DCATBOOST_MLX_PROBE_BYPASS`
    (see docs/sprint33/probe-b-python/verdict.md § Build Procedure).

Expected outcomes:
  - PRODUCTION-CLEAN: Python-path drift ≤ 2% (path uses a separate
    quantization that handles Cosine correctly).
  - PRODUCTION-BROKEN: drift ≫ 2% — Python path delegates to the same
    csv_train.cpp::QuantizeFeatures, so Option 3 is invalid.
"""
import json, os, sys, time, hashlib
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── 0. Verify .so freshness and md5 ───────────────────────────────────────────
import catboost_mlx
SO_PATH = Path(catboost_mlx.__file__).parent / "_core.cpython-313-darwin.so"
so_md5 = hashlib.md5(open(SO_PATH, "rb").read()).hexdigest()
print(f"[probe-b] _core.so: {SO_PATH}")
print(f"[probe-b] _core.so md5: {so_md5}")
print(f"[probe-b] _core.so mtime: {time.ctime(SO_PATH.stat().st_mtime)}")

# ─── 1. Build anchor dataset (matches L4 run_phase1.py exactly) ────────────────
N, NF, SEED = 50_000, 20, 42
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, NF)).astype(np.float32)
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)
print(f"[probe-b] dataset: N={N}, F={NF}, seed={SEED}, "
      f"y_mean={y.mean():.6f}, y_std={y.std():.6f}")

ITERS_LIST = [50, 100]   # match L4 verdict drift table
DEPTH, BINS, LR = 6, 127, 0.03

# ─── 2. CPU baseline via stock catboost.CatBoostRegressor ──────────────────────
import catboost
print(f"[probe-b] catboost (CPU): version={catboost.__version__}")
from catboost import CatBoostRegressor

cpu_results = {}
for it in ITERS_LIST:
    m_cpu = CatBoostRegressor(
        iterations=it, depth=DEPTH, border_count=BINS, learning_rate=LR,
        grow_policy="SymmetricTree", score_function="Cosine",
        loss_function="RMSE", bootstrap_type="No",
        random_strength=0.0, has_time=True, random_seed=SEED,
        task_type="CPU", verbose=False)
    t0 = time.time()
    m_cpu.fit(X, y)
    pred = m_cpu.predict(X)
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    t1 = time.time() - t0
    cpu_results[it] = rmse
    print(f"[probe-b] CPU iter={it:3d}: RMSE={rmse:.6f}  ({t1:.1f}s)")

# ─── 3. MLX via nanobind Python path ───────────────────────────────────────────
# Monkey-patch the Python ST+Cosine guard.
from catboost_mlx import CatBoostMLXRegressor
_orig_validate = CatBoostMLXRegressor._validate_params
def _patched_validate(self):
    sf = self.score_function
    self.score_function = "L2"
    try:
        _orig_validate(self)
    finally:
        self.score_function = sf
CatBoostMLXRegressor._validate_params = _patched_validate

mlx_results = {}
for it in ITERS_LIST:
    m_mlx = CatBoostMLXRegressor(
        iterations=it, depth=DEPTH, bins=BINS, learning_rate=LR,
        grow_policy="SymmetricTree", score_function="Cosine",
        loss="rmse", bootstrap_type="no",
        random_strength=0.0, random_seed=SEED, verbose=False)
    t0 = time.time()
    m_mlx.fit(X, y)
    pred = np.asarray(m_mlx.predict(X), dtype=np.float64).ravel()
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    t1 = time.time() - t0
    mlx_results[it] = rmse
    print(f"[probe-b] MLX iter={it:3d}: RMSE={rmse:.6f}  ({t1:.1f}s)")

# ─── 4. Drift table ────────────────────────────────────────────────────────────
out = {
    "_core_so_md5": so_md5,
    "_core_so_path": str(SO_PATH),
    "_core_so_mtime": time.ctime(SO_PATH.stat().st_mtime),
    "anchor": dict(N=N, F=NF, seed=SEED, depth=DEPTH, bins=BINS, lr=LR,
                   grow_policy="SymmetricTree", score_function="Cosine",
                   loss="RMSE", bootstrap_type="No", random_strength=0.0,
                   has_time=True),
    "cpu_rmse": cpu_results,
    "mlx_rmse": mlx_results,
}
print()
print("=== PROBE-B drift table (Python path) ===")
print(f"{'iters':>6}  {'CPU RMSE':>12}  {'MLX RMSE':>12}  {'ratio':>8}  {'drift%':>8}")
for it in ITERS_LIST:
    cpu, mlx = cpu_results[it], mlx_results[it]
    ratio = mlx / cpu
    drift = (mlx - cpu) / cpu * 100
    print(f"{it:>6}  {cpu:>12.6f}  {mlx:>12.6f}  {ratio:>8.4f}  {drift:>7.2f}%")
    out[f"drift_pct_iter{it}"] = drift
    out[f"ratio_iter{it}"] = ratio

(DATA_DIR / "probe_b_results.json").write_text(json.dumps(out, indent=2))
print(f"\n[probe-b] results -> {DATA_DIR/'probe_b_results.json'}")

# Class call
drift50 = out["drift_pct_iter50"]
if drift50 <= 2.0:
    klass = "PRODUCTION-CLEAN"
elif drift50 > 10.0:
    klass = "PRODUCTION-BROKEN"
else:
    klass = "INCONCLUSIVE"
print(f"\n[probe-b] CLASS CALL: {klass}  (iter=50 drift = {drift50:.2f}%)")
