#!/usr/bin/env python3
"""
S33-L1-DETERMINISM: Drift measurement under maximally deterministic config.

Anchor: N=50000, SymmetricTree, Cosine, RMSE, depth=6, bins=127, iter=50,
        seeds={42,43,44}, random_seed=0.

Determinism stack applied on BOTH sides:
  bootstrap_type = 'No'
  subsample      = 1.0
  random_strength = 0.0
  has_time       = True   (CPU only; MLX processes in file order — equivalent)
  sampling_unit  = 'Object' (default; explicit on CPU)
  random_seed    = 0  (CatBoost global); seed = 42/43/44 (data + training seed)

Binary: csv_train_t4 (built 2026-04-24 17:55; DEC-038 + DEC-039 fixes active).

Outputs:
  data/cpu_rmse_seed{42,43,44}.txt
  data/mlx_rmse_seed{42,43,44}.txt
"""

import csv as csv_mod
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[3]
MLX_BINARY = REPO_ROOT / "csv_train_t4"
DATA_DIR   = Path(__file__).resolve().parent / "data"

# Anchor
N        = 50_000
DEPTH    = 6
BINS_MLX = 127        # DEC-039 cap: std::min(128, 127u) = 127 borders
BINS_CPU = 127        # border_count = 127 (= max_bin - 1 for max_bin=128)
LR       = 0.03
LOSS     = "rmse"
GROW     = "SymmetricTree"
SCORE_FN = "Cosine"
ITERS    = 50
SEEDS    = [42, 43, 44]
L2       = 3.0

# Determinism stack
BOOTSTRAP_TYPE  = "No"
SUBSAMPLE       = 1.0
RANDOM_STRENGTH = 0.0
HAS_TIME        = True   # CPU-only; MLX processes in file order (equivalent)
RANDOM_SEED_CB  = 0      # CatBoost global random_seed


def make_data(n: int, seed: int):
    """Canonical S26+ data: 20 features, signal in f0/f1, 10% noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    n_feat = X.shape[1]
    header = [f"f{i}" for i in range(n_feat)] + ["target"]
    with open(path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(header)
        for i in range(len(y)):
            writer.writerow(list(X[i]) + [y[i]])


def parse_final_loss(stdout: str) -> float:
    """Extract last iter train loss from csv_train stdout."""
    last_loss = None
    for line in stdout.split("\n"):
        if "loss=" in line and "iter=" in line:
            for tok in line.split():
                if tok.startswith("loss="):
                    try:
                        last_loss = float(tok.split("=", 1)[1])
                    except ValueError:
                        pass
    if last_loss is None:
        raise ValueError(f"Could not parse final loss:\n{stdout[:2000]}")
    return last_loss


def run_mlx(data_path: Path, seed: int) -> float:
    """Run csv_train_t4 with determinism stack; return final RMSE."""
    cmd = [
        str(MLX_BINARY),
        str(data_path),
        "--iterations",      str(ITERS),
        "--depth",           str(DEPTH),
        "--lr",              str(LR),
        "--bins",            str(BINS_MLX),
        "--loss",            LOSS,
        "--grow-policy",     GROW,
        "--score-function",  SCORE_FN,
        "--seed",            str(seed),
        "--random-strength", str(RANDOM_STRENGTH),
        "--bootstrap-type",  "no",
        "--subsample",       str(SUBSAMPLE),
        "--verbose",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"csv_train_t4 failed (seed={seed}):\n"
            f"STDERR: {result.stderr[:500]}"
        )
    return parse_final_loss(result.stdout)


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Run CatBoost CPU with determinism stack; return final train RMSE."""
    from catboost import CatBoostRegressor
    # Note: CatBoost rejects subsample when bootstrap_type='No'.
    # subsample=1.0 is the implicit default when bootstrap is disabled.
    m = CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy=GROW,
        score_function=SCORE_FN,
        max_bin=BINS_CPU + 1,    # CatBoost max_bin = border_count + 1; border_count=127 → max_bin=128
        random_seed=seed,
        random_strength=RANDOM_STRENGTH,
        bootstrap_type=BOOTSTRAP_TYPE,
        has_time=HAS_TIME,
        sampling_unit="Object",
        l2_leaf_reg=L2,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def main():
    if not MLX_BINARY.exists():
        print(f"ERROR: {MLX_BINARY} not found.", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("S33-L1-DETERMINISM: iter=50 drift under maximally deterministic config")
    print(f"  Anchor: N={N}, d={DEPTH}, bins_mlx={BINS_MLX}, bins_cpu={BINS_CPU}, iters={ITERS}")
    print(f"  Config: {LOSS}/{GROW}/Cosine, rs={RANDOM_STRENGTH}, bootstrap={BOOTSTRAP_TYPE}")
    print(f"  CPU extras: has_time={HAS_TIME}, sampling_unit=Object, random_seed={RANDOM_SEED_CB}")
    print(f"  Seeds: {SEEDS}")
    print()

    hdr = f"{'seed':>5} | {'CPU_RMSE':>12} {'MLX_RMSE':>12} {'drift_%':>9} | class"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    rows = []
    drifts = []

    for seed in SEEDS:
        print(f"  Running seed={seed}...", flush=True)
        X, y = make_data(N, seed)

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            data_path = Path(tf.name)
        try:
            write_csv(data_path, X, y)
            t0 = time.perf_counter()
            mlx_rmse = run_mlx(data_path, seed)
            mlx_wall = time.perf_counter() - t0

            t0 = time.perf_counter()
            cpu_rmse = run_cpu(X, y, seed)
            cpu_wall = time.perf_counter() - t0
        finally:
            os.unlink(data_path)

        drift_pct = (mlx_rmse - cpu_rmse) / cpu_rmse * 100.0
        abs_drift = abs(drift_pct)

        if abs_drift <= 2.0:
            cls = "K6-FIRES"
        elif abs_drift < 20.0:
            cls = "SOFT-PARTIAL"
        else:
            cls = "FALSIFIED"

        print(f"{seed:>5} | {cpu_rmse:>12.8f} {mlx_rmse:>12.8f} {drift_pct:>+9.3f}% | {cls}  "
              f"(mlx_wall={mlx_wall:.1f}s, cpu_wall={cpu_wall:.1f}s)")

        # Write individual seed files
        (DATA_DIR / f"cpu_rmse_seed{seed}.txt").write_text(f"{cpu_rmse:.10f}\n")
        (DATA_DIR / f"mlx_rmse_seed{seed}.txt").write_text(f"{mlx_rmse:.10f}\n")

        drifts.append(drift_pct)
        rows.append({
            "seed": seed,
            "cpu_rmse": cpu_rmse,
            "mlx_rmse": mlx_rmse,
            "drift_pct": drift_pct,
            "class": cls,
        })

    print(sep)

    median_drift = sorted(drifts)[len(drifts) // 2]
    abs_drifts = [abs(d) for d in drifts]
    max_abs = max(abs_drifts)
    min_abs = min(abs_drifts)

    # Overall class
    if max_abs <= 2.0:
        overall = "K6-FIRES"
    elif max_abs < 20.0:
        overall = "SOFT-PARTIAL"
    else:
        overall = "FALSIFIED"

    print()
    print(f"  Median drift:   {median_drift:+.3f}%")
    print(f"  Max |drift|:    {max_abs:.3f}%")
    print(f"  Min |drift|:    {min_abs:.3f}%")
    print(f"  S32 baseline:   ~52.6%")
    print()
    print(f"=== L1-DETERMINISM CLASS: {overall} ===")

    # Write summary CSV
    out_path = DATA_DIR / "l1_drift_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f,
            fieldnames=["seed", "cpu_rmse", "mlx_rmse", "drift_pct", "class"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Raw data: {out_path}")

    return overall


if __name__ == "__main__":
    main()
