#!/usr/bin/env python3
"""
S30-T3-MEASURE: Gate G3a — ST anchor post-Kahan parity.

Cell: N=50000, depth=6, bins=128, iterations=50, seeds={42,43,44,45,46}
Config: loss=RMSE, grow_policy=SymmetricTree, score_function='Cosine'
Binary: csv_train_t3 (built with -DCOSINE_T3_MEASURE; K4 fp64 Kahan fix active)

PASS criterion: aggregate drift < 2.0%
  aggregate_drift = mean( |MLX_RMSE - CPU_RMSE| / CPU_RMSE * 100 ) across seeds

CPU reference: catboost pip package, same parameters, score_function='Cosine'.

Output:
  data/g3a_seed{seed}.csv  — per-seed raw numbers
  stdout — gate verdict
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
DATA_DIR  = REPO_ROOT / "docs" / "sprint30" / "t3-measure" / "data"

# G3a cell parameters (mirror S28 t7-gate-report.md G6a exactly)
N         = 50_000
DEPTH     = 6
BINS      = 128
LR        = 0.03
LOSS      = "rmse"
GROW      = "SymmetricTree"
SCORE_FN  = "Cosine"
ITERS     = 50
SEEDS     = [42, 43, 44, 45, 46]

G3a_PASS_THRESHOLD = 2.0  # percent aggregate drift


def make_data(n: int, seed: int):
    """Canonical S26 data: 20 features, signal in f0 and f1, 10% noise."""
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
    We want the last such line's loss value.
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


def run_mlx(data_path: Path, seed: int) -> tuple[float, float]:
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
        raise RuntimeError(f"csv_train_t3 failed for seed={seed}")
    rmse = parse_final_loss(result.stdout)
    return rmse, elapsed


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Run CPU CatBoost with same params; return final train RMSE."""
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

    print(f"G3a — ST anchor post-Kahan parity")
    print(f"  Cell: N={N}, depth={DEPTH}, bins={BINS}, iters={ITERS}, seeds={SEEDS}")
    print(f"  Config: {LOSS}/{GROW}/Cosine  (K4 fp64 fix active in csv_train_t3)")
    print(f"  PASS criterion: aggregate drift < {G3a_PASS_THRESHOLD}%\n")

    results = []
    raw_rows = []

    for seed in SEEDS:
        print(f"--- seed={seed} ---")
        X, y = make_data(N, seed)

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            data_path = Path(tf.name)
        try:
            write_csv(data_path, X, y)
            mlx_rmse, wall_secs = run_mlx(data_path, seed)
            cpu_rmse = run_cpu(X, y, seed)
        finally:
            os.unlink(data_path)

        drift_pct = abs(mlx_rmse - cpu_rmse) / cpu_rmse * 100.0
        ratio = mlx_rmse / cpu_rmse
        print(f"  MLX={mlx_rmse:.8f}  CPU={cpu_rmse:.8f}  ratio={ratio:.6f}  drift={drift_pct:.4f}%  wall={wall_secs:.1f}s")
        results.append(drift_pct)
        raw_rows.append({
            "seed": seed,
            "mlx_rmse": mlx_rmse,
            "cpu_rmse": cpu_rmse,
            "ratio": ratio,
            "drift_pct": drift_pct,
            "wall_secs": wall_secs,
        })

    # Write raw CSV
    out_path = DATA_DIR / "g3a_st_anchor.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["seed","mlx_rmse","cpu_rmse","ratio","drift_pct","wall_secs"])
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\nRaw data: {out_path}")

    # Gate check
    agg_drift = sum(results) / len(results)
    max_drift = max(results)
    min_drift = min(results)
    gate_pass = agg_drift < G3a_PASS_THRESHOLD

    print(f"\n=== G3a RESULT ===")
    print(f"  Seeds:          {SEEDS}")
    print(f"  Per-seed drift: {[f'{d:.4f}%' for d in results]}")
    print(f"  Aggregate drift (mean): {agg_drift:.4f}%")
    print(f"  Max drift:      {max_drift:.4f}%")
    print(f"  Min drift:      {min_drift:.4f}%")
    print(f"  Threshold:      {G3a_PASS_THRESHOLD}%")
    print(f"  G3a: {'PASS' if gate_pass else 'FAIL'}")

    if not gate_pass:
        print(f"\nG3a FAIL — aggregate drift {agg_drift:.4f}% >= {G3a_PASS_THRESHOLD}% threshold", file=sys.stderr)
        sys.exit(1)

    print("\nG3a PASS")
    return 0


if __name__ == "__main__":
    main()
