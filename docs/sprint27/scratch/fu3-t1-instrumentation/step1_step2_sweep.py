"""S27-FU-3-T1 Steps 1 and 2: re-run the 5 failing DW N=1000 cells post-FU-1,
then run ST control at N=1000 with matched config.

Step 1: confirm asymmetry still present post-FU-1.
Step 2: run SymmetricTree control at same seeds to determine if ST also shows
        MLX-better-than-CPU asymmetry (small-N noise) or if it's DW-specific.

Config from benchmarks/sprint26/fu2/g1_sweep.py:
  d=6, 128 bins, LR=0.03, 50 iters, RMSE, 20 features,
  bootstrap_type='No'/'no', rs=0.0 and rs=1.0.

5 failing cells (from g1-results.md segmented gate failures):
  Depthwise, N=1000, seed=1337, rs=0.0  (ratio=0.8315)
  Depthwise, N=1000, seed=42,   rs=0.0  (ratio=0.8619)
  Depthwise, N=1000, seed=42,   rs=1.0  (ratio=0.8232)
  Depthwise, N=1000, seed=7,    rs=0.0  (ratio=0.8620)
  Depthwise, N=1000, seed=7,    rs=1.0  (ratio=0.8276)

Note: seed=1337 rs=1.0 passed the segmented gate (pred_std_R=1.0959 < 1.10)
so it is not in the 5 failing cells per the formal gate spec.
"""

import os
import sys
import json
import time
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
RS_VALUES = [0.0, 1.0]

# S26-FU-2 reference values for comparison (pre-FU-1, from g1-results.md)
PREFU1_DW_N1000 = {
    (1337, 0.0): {"cpu": 0.216145, "mlx": 0.179724, "ratio": 0.8315, "pred_std_R": 1.1004},
    (1337, 1.0): {"cpu": 0.237086, "mlx": 0.196611, "ratio": 0.8293, "pred_std_R": 1.0959},
    (42,   0.0): {"cpu": 0.210677, "mlx": 0.181591, "ratio": 0.8619, "pred_std_R": 1.0759},
    (42,   1.0): {"cpu": 0.241135, "mlx": 0.198501, "ratio": 0.8232, "pred_std_R": 1.1028},
    (7,    0.0): {"cpu": 0.208184, "mlx": 0.179449, "ratio": 0.8620, "pred_std_R": 1.0832},
    (7,    1.0): {"cpu": 0.235937, "mlx": 0.195260, "ratio": 0.8276, "pred_std_R": 1.1011},
}


def make_xy(N, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(N).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    return X, y


def run_cpu(X, y, seed, rs, grow_policy):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss_function="RMSE", grow_policy=grow_policy, max_bin=BINS,
        random_seed=seed, random_strength=rs,
        bootstrap_type="No",
        verbose=0, thread_count=1,
    )
    t0 = time.perf_counter()
    m.fit(X, y)
    t1 = time.perf_counter()
    rmse = float(m.evals_result_["learn"]["RMSE"][-1])
    preds = m.predict(X)
    return rmse, np.asarray(preds, dtype=np.float64), t1 - t0


def run_mlx(X, y, seed, rs, grow_policy):
    from catboost_mlx import CatBoostMLXRegressor
    m = CatBoostMLXRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss="rmse", grow_policy=grow_policy, bins=BINS,
        random_seed=seed, random_strength=rs,
        bootstrap_type="no",
        verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(X, y)
    t1 = time.perf_counter()
    if m._train_loss_history:
        rmse = float(m._train_loss_history[-1])
    else:
        preds_for_rmse = m.predict(X)
        rmse = float(np.sqrt(((np.asarray(preds_for_rmse, dtype=np.float64) - y) ** 2).mean()))
    preds = m.predict(X)
    return rmse, np.asarray(preds, dtype=np.float64), t1 - t0


def cell_passes_segmented(ratio, pred_std_r, rs):
    if rs == 0.0:
        return (0.98 <= ratio <= 1.02) and (0.90 <= pred_std_r <= 1.10)
    else:
        return (ratio <= 1.02) and (0.90 <= pred_std_r <= 1.10)


results = []

print("=" * 90)
print("STEP 1 — DW N=1000 asymmetry re-check post-FU-1")
print("=" * 90)
print(f"{'Policy':>13s} {'N':>5s} {'seed':>5s} {'rs':>4s} | "
      f"{'CPU_RMSE':>11s} {'MLX_RMSE':>11s} {'ratio':>7s} {'pred_std_R':>10s} | "
      f"{'seg':>4s} | {'vs_prefu1':>12s}")
print("-" * 90)

for seed in SEEDS:
    X, y = make_xy(N, seed)
    for rs in RS_VALUES:
        cpu_rmse, cpu_preds, cpu_t = run_cpu(X, y, seed, rs, "Depthwise")
        mlx_rmse, mlx_preds, mlx_t = run_mlx(X, y, seed, rs, "Depthwise")

        ratio = mlx_rmse / cpu_rmse
        pred_std_r = float(np.std(mlx_preds) / np.std(cpu_preds))
        seg = cell_passes_segmented(ratio, pred_std_r, rs)

        # Compare to pre-FU-1 reference
        prefu1 = PREFU1_DW_N1000.get((seed, rs), {})
        pre_ratio = prefu1.get("ratio", float("nan"))
        ratio_delta = ratio - pre_ratio
        status = f"{ratio_delta:+.4f} vs pre"

        print(f"{'Depthwise':>13s} {N:>5d} {seed:>5d} {rs:>4.1f} | "
              f"{cpu_rmse:>11.6f} {mlx_rmse:>11.6f} {ratio:>7.4f} {pred_std_r:>10.4f} | "
              f"{'PASS' if seg else 'FAIL':>4s} | {status:>12s}")
        sys.stdout.flush()

        results.append({
            "step": 1, "policy": "Depthwise", "N": N, "seed": seed, "rs": rs,
            "cpu_rmse": cpu_rmse, "mlx_rmse": mlx_rmse, "ratio": ratio,
            "pred_std_r": pred_std_r, "seg_pass": seg,
            "prefu1_ratio": pre_ratio,
        })

print()
print("=" * 90)
print("STEP 2 — ST control at N=1000 (matched config, same seeds)")
print("=" * 90)
print(f"{'Policy':>13s} {'N':>5s} {'seed':>5s} {'rs':>4s} | "
      f"{'CPU_RMSE':>11s} {'MLX_RMSE':>11s} {'ratio':>7s} {'pred_std_R':>10s} | "
      f"{'seg':>4s}")
print("-" * 90)

for seed in SEEDS:
    X, y = make_xy(N, seed)
    for rs in RS_VALUES:
        cpu_rmse, cpu_preds, cpu_t = run_cpu(X, y, seed, rs, "SymmetricTree")
        mlx_rmse, mlx_preds, mlx_t = run_mlx(X, y, seed, rs, "SymmetricTree")

        ratio = mlx_rmse / cpu_rmse
        pred_std_r = float(np.std(mlx_preds) / np.std(cpu_preds))
        seg = cell_passes_segmented(ratio, pred_std_r, rs)

        print(f"{'SymmetricTree':>13s} {N:>5d} {seed:>5d} {rs:>4.1f} | "
              f"{cpu_rmse:>11.6f} {mlx_rmse:>11.6f} {ratio:>7.4f} {pred_std_r:>10.4f} | "
              f"{'PASS' if seg else 'FAIL':>4s}")
        sys.stdout.flush()

        results.append({
            "step": 2, "policy": "SymmetricTree", "N": N, "seed": seed, "rs": rs,
            "cpu_rmse": cpu_rmse, "mlx_rmse": mlx_rmse, "ratio": ratio,
            "pred_std_r": pred_std_r, "seg_pass": seg,
        })

# Save JSON
out_path = (
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/"
    "docs/sprint27/scratch/fu3-t1-instrumentation/step1_step2_results.json"
)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to: {out_path}")

# Summary
print()
print("--- Step 1 summary (DW N=1000 post-FU-1) ---")
dw_results = [r for r in results if r["step"] == 1]
n_fail = sum(1 for r in dw_results if not r["seg_pass"])
print(f"  Fail count: {n_fail}/6 cells")
for r in dw_results:
    tag = "FAIL" if not r["seg_pass"] else "pass"
    ratio_change = r["ratio"] - r["prefu1_ratio"]
    print(f"  seed={r['seed']} rs={r['rs']:.1f}: ratio={r['ratio']:.4f} "
          f"(prefu1={r['prefu1_ratio']:.4f}, delta={ratio_change:+.4f}) [{tag}]")

print()
print("--- Step 2 summary (ST control N=1000) ---")
st_results = [r for r in results if r["step"] == 2]
n_st_fail = sum(1 for r in st_results if not r["seg_pass"])
n_st_mlx_better_rs0 = sum(1 for r in st_results if r["rs"] == 0.0 and r["ratio"] < 0.98)
n_st_mlx_better_rs1 = sum(1 for r in st_results if r["rs"] == 1.0 and r["ratio"] < 0.98)
print(f"  Total failures: {n_st_fail}/6")
print(f"  rs=0 MLX-better-than-CPU by >2%: {n_st_mlx_better_rs0}/3")
print(f"  rs=1 MLX-better-than-CPU by >2%: {n_st_mlx_better_rs1}/3")
for r in st_results:
    tag = "FAIL" if not r["seg_pass"] else "pass"
    print(f"  seed={r['seed']} rs={r['rs']:.1f}: ratio={r['ratio']:.4f} [{tag}]")

print()
if n_st_mlx_better_rs0 > 0:
    print("DIAGNOSTIC: ST rs=0 also shows MLX-better asymmetry at N=1000 => NOT DW-specific")
    print("  => Supports hypothesis (b): small-N noise / quantization mechanism")
else:
    print("DIAGNOSTIC: ST rs=0 is tight at N=1000 => asymmetry IS DW-specific")
    print("  => Supports hypothesis (a/c): FindBestSplitPerPartition-specific divergence")
