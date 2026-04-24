#!/usr/bin/env python3
"""
S30-V1-DRIFT-VS-N: Stress-test the L0-histogram-dominance hypothesis.

Sweeps N ∈ {1000, 5000, 10000, 25000, 50000} at seeds {42, 43, 44}.
Config: loss=RMSE, grow_policy=SymmetricTree, score_function='Cosine',
        depth=6, bins=128, iterations=50.

Binary: csv_train_t3 (K4 fp64 fix active, same as T3/G3a).
CPU reference: catboost pip package, score_function='Cosine', same params as T3.

Output:
  data/v1_drift_vs_n.csv  — rows: N, seed, mlx_rmse, cpu_rmse, drift_pct
  stdout — per-cell results + scaling verdict
"""

import csv as csv_mod
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
T3_BINARY = REPO_ROOT / "csv_train_t3"
DATA_DIR  = Path(__file__).resolve().parent / "data"

# Sweep parameters
N_VALUES = [1000, 5000, 10000, 25000, 50000]
SEEDS    = [42, 43, 44]

# Fixed cell params — mirror T3/G3a exactly
DEPTH    = 6
BINS     = 128
LR       = 0.03
LOSS     = "rmse"
GROW     = "SymmetricTree"
SCORE_FN = "Cosine"
ITERS    = 50


def make_data(n: int, seed: int):
    """Canonical S26 data: 20 features, signal in f0 and f1, 10% noise.
    Identical generator to T3/run_t3_st_anchor.py.
    """
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
    """Extract final iter train loss from csv_train stdout.
    csv_train prints: iter=N  trees=N  depth=N  loss=XXXX  time=Yms
    """
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
        raise ValueError(f"Could not parse final loss from stdout:\n{stdout[:2000]}")
    return last_loss


def run_mlx(data_path: Path, n: int, seed: int) -> tuple[float, float]:
    """Run csv_train_t3; return (final_rmse, wall_clock_secs)."""
    cmd = [
        str(T3_BINARY),
        str(data_path),
        "--iterations", str(ITERS),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--loss", LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed", str(seed),
        "--verbose",
    ]
    env = os.environ.copy()
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  ERROR: csv_train_t3 exited {result.returncode}", file=sys.stderr)
        print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"csv_train_t3 failed for N={n}, seed={seed}")
    rmse = parse_final_loss(result.stdout)
    return rmse, elapsed


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Run CPU CatBoost with same params; return final train RMSE.
    Mirrors T3/run_t3_st_anchor.py run_cpu() exactly.
    """
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy=GROW,
        score_function=SCORE_FN,
        max_bin=BINS,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="No",
        l2_leaf_reg=3.0,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def fit_power_law(n_vals: list[int], drift_means: list[float]) -> tuple[float, float]:
    """Fit drift_pct = a * N^b via log-log OLS.
    Returns (a, b). b > 0 means drift grows with N.
    Only uses entries where drift_mean > 0.
    """
    xs = []
    ys = []
    for n, d in zip(n_vals, drift_means):
        if d > 0:
            xs.append(math.log(n))
            ys.append(math.log(d))
    if len(xs) < 2:
        return float("nan"), float("nan")
    n_pts = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n_pts * sxx - sx * sx
    if abs(denom) < 1e-12:
        return float("nan"), float("nan")
    b = (n_pts * sxy - sx * sy) / denom
    a_log = (sy - b * sx) / n_pts
    a = math.exp(a_log)
    return a, b


def main():
    if not T3_BINARY.exists():
        print(f"ERROR: {T3_BINARY} not found. Build with:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DCOSINE_T3_MEASURE \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_t3", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("S30-V1-DRIFT-VS-N — L0 histogram hypothesis scaling check")
    print(f"  Binary:  csv_train_t3 (K4 fp64 fix active)")
    print(f"  N sweep: {N_VALUES}")
    print(f"  Seeds:   {SEEDS}")
    print(f"  Config:  {LOSS}/{GROW}/Cosine, depth={DEPTH}, bins={BINS}, iters={ITERS}")
    print()

    all_rows = []
    # drift_by_n[N] = list of drift_pct across seeds
    drift_by_n: dict[int, list[float]] = {n: [] for n in N_VALUES}

    for n in N_VALUES:
        print(f"=== N={n} ===")
        for seed in SEEDS:
            print(f"  seed={seed} ...", end="", flush=True)
            X, y = make_data(n, seed)

            # Write temp CSV for MLX binary
            with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
                data_path = Path(tf.name)
            try:
                write_csv(data_path, X, y)
                mlx_rmse, wall_secs = run_mlx(data_path, n, seed)
                cpu_rmse = run_cpu(X, y, seed)
            finally:
                try:
                    os.unlink(data_path)
                except FileNotFoundError:
                    pass

            drift_pct = abs(mlx_rmse - cpu_rmse) / cpu_rmse * 100.0
            ratio = mlx_rmse / cpu_rmse
            print(f"  MLX={mlx_rmse:.8f}  CPU={cpu_rmse:.8f}  ratio={ratio:.6f}"
                  f"  drift={drift_pct:.4f}%  wall={wall_secs:.1f}s")

            drift_by_n[n].append(drift_pct)
            all_rows.append({
                "N": n,
                "seed": seed,
                "mlx_rmse": mlx_rmse,
                "cpu_rmse": cpu_rmse,
                "drift_pct": drift_pct,
                "wall_secs": wall_secs,
            })
        print()

    # Write raw CSV
    out_csv = DATA_DIR / "v1_drift_vs_n.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=["N", "seed", "mlx_rmse", "cpu_rmse", "drift_pct", "wall_secs"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Raw data written: {out_csv}")
    print()

    # Summary table
    n_means = []
    print("=== Per-N drift summary ===")
    print(f"  {'N':>7}  {'s42':>8}  {'s43':>8}  {'s44':>8}  {'mean':>8}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for n in N_VALUES:
        drifts = drift_by_n[n]
        mean_d = sum(drifts) / len(drifts)
        n_means.append(mean_d)
        per_seed_str = "  ".join(f"{d:8.4f}" for d in drifts)
        print(f"  {n:>7}  {per_seed_str}  {mean_d:8.4f}%")
    print()

    # Power-law fit on means
    a, b = fit_power_law(N_VALUES, n_means)
    print(f"Log-log fit (drift ~ a * N^b):")
    if not math.isnan(b):
        print(f"  a = {a:.6g}")
        print(f"  b = {b:.4f}  (positive → drift grows with N)")
    else:
        print("  Fit failed (insufficient valid data points)")
    print()

    # Monotonicity check
    monotone_up = all(n_means[i] <= n_means[i + 1] for i in range(len(n_means) - 1))
    monotone_dn = all(n_means[i] >= n_means[i + 1] for i in range(len(n_means) - 1))

    # Variance check (all values within 20% of mean — "suspiciously flat")
    grand_mean = sum(n_means) / len(n_means)
    max_rel_dev = max(abs(d - grand_mean) / grand_mean for d in n_means) if grand_mean > 0 else float("inf")

    print("=== Scaling verdict ===")
    if max_rel_dev < 0.20:
        verdict = "FLAT"
        print("  Pattern: FLAT (all N within 20% of grand mean)")
        print("  Interpretation: L0 hypothesis FALSIFIED — drift is not summand-count-driven.")
    elif monotone_up and not math.isnan(b) and b > 0.3:
        verdict = "SUPPORTED"
        print(f"  Pattern: MONOTONE INCREASING, b={b:.4f} > 0.3")
        print("  Interpretation: L0 hypothesis SUPPORTED — drift grows with N.")
    elif monotone_up and not math.isnan(b):
        verdict = "WEAK"
        print(f"  Pattern: MONOTONE INCREASING, b={b:.4f} <= 0.3 (weak scaling)")
        print("  Interpretation: Mixed — drift increases with N but slowly; L0 partial contributor.")
    elif monotone_dn:
        verdict = "ANTI-MONOTONE"
        print("  Pattern: MONOTONE DECREASING")
        print("  Interpretation: L0 hypothesis FALSIFIED — drift decreases as N grows.")
    else:
        verdict = "MIXED"
        print("  Pattern: NON-MONOTONIC")
        print("  Interpretation: Mixed signal — multiple mechanisms; further disambiguation needed.")

    print(f"\n  Final verdict: {verdict}")
    print(f"  Mean drifts by N: {dict(zip(N_VALUES, [round(d, 4) for d in n_means]))}")
    if not math.isnan(b):
        print(f"  Scaling exponent b = {b:.4f}")

    return 0


if __name__ == "__main__":
    main()
