#!/usr/bin/env python3
# S30-V5-DW-AT-SCALE commit 1/2: harness + raw data
"""
S30-V5-DW-AT-SCALE: DW+Cosine parity at ST-anchor cell size (N=50000).

Hypothesis under test (L0-dominant): The S28 DW+Cosine cell passed at 1.6%
drift because its N=1000 was too small for L0 histogram cancellation error to
accumulate.  If DW also breaks catastrophically at N=50000, the L0 hypothesis
is CONSISTENT and S31's histogram-fix plan proceeds.  If DW stays contained
(<= 5%), a grow-policy-specific mechanism must be driving ST's 53% failure.

Cell: N=50000, depth=6, bins=128, iterations=50, seeds={42,43,44}
Config: loss=RMSE, grow_policy=Depthwise, score_function=Cosine
Binary: csv_train_t3 (COSINE_T3_MEASURE active; K4 fp64 gain active; DW+Cosine
        has NO guard, so the T3 bypass is not needed, but using the T3 binary
        ensures K4 is active and the binary is current-branch tip.)

CPU reference: catboost pip package, same parameters, score_function='Cosine'.

Output:
  data/dw_at_scale_seed{seed}.csv  -- per-seed raw timing + RMSE
  data/dw_at_scale_summary.csv     -- all seeds aggregated
  stdout                           -- per-seed results + verdict
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
DATA_DIR   = Path(__file__).resolve().parent / "data"

# ST-anchor cell parameters (mirror T3 run_t3_st_anchor.py exactly except GROW)
N          = 50_000
DEPTH      = 6
BINS       = 128
LR         = 0.03
LOSS       = "rmse"
GROW       = "Depthwise"   # <-- the only change from T3 ST anchor
SCORE_FN   = "Cosine"
ITERS      = 50
SEEDS      = [42, 43, 44]

# S28 DW+Cosine baseline at N=1000 (5-seed mean from t5-gate-report.md G5a)
S28_DW_N1000_MEAN_DRIFT = 1.104   # percent: (|1.0159-1|+|1.0160-1|+|1.0023-1|+|1.0076-1|+|0.9950-1|) / 5 * 100
# T3 ST+Cosine at N=50000 (5-seed mean from t3-measure/verdict.md G3a)
T3_ST_N50K_MEAN_DRIFT   = 53.30   # percent

# Interpretation thresholds (from task spec)
CATASTROPHIC_THRESHOLD  = 20.0   # >= 20% -> L0 CONSISTENT
CONTAINED_THRESHOLD     =  5.0   # <=  5% -> FALSIFIES L0-only explanation


def make_data(n: int, seed: int):
    """Canonical S26 data generator: 20 features, signal in f0+f1, 10% noise.
    Identical to run_t3_st_anchor.py make_data().
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
    Looks for last line matching: iter=N  trees=N  depth=N  loss=XXXX  time=Yms
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
    """Run csv_train_t3 with DW+Cosine; return (final_rmse, wall_clock_secs)."""
    cmd = [
        str(T3_BINARY),
        str(data_path),
        "--iterations", str(ITERS),
        "--depth",      str(DEPTH),
        "--lr",         str(LR),
        "--bins",       str(BINS),
        "--loss",       LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed",       str(seed),
        "--verbose",
    ]
    env = os.environ.copy()
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  ERROR: {T3_BINARY.name} exited {result.returncode}", file=sys.stderr)
        print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
        print(f"  STDOUT: {result.stdout[:500]}", file=sys.stderr)
        raise RuntimeError(f"{T3_BINARY.name} failed for seed={seed}")
    rmse = parse_final_loss(result.stdout)
    return rmse, elapsed


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Run CPU CatBoost with DW+Cosine same params; return final train RMSE.
    Matches T3 run_cpu() exactly except grow_policy.
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


def interpret(mean_drift: float) -> str:
    if mean_drift >= CATASTROPHIC_THRESHOLD:
        return "CONSISTENT"    # L0 hypothesis stands
    elif mean_drift <= CONTAINED_THRESHOLD:
        return "FALSIFIES"     # L0 alone can't explain ST failure
    else:
        return "MIXED"         # L0 is a contributor but not sole driver


def main():
    if not T3_BINARY.exists():
        print(f"ERROR: {T3_BINARY} not found.  Build with:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DCOSINE_T3_MEASURE \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_t3", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("S30-V5-DW-AT-SCALE — DW+Cosine at N=50000")
    print(f"  Cell: N={N}, depth={DEPTH}, bins={BINS}, iters={ITERS}, seeds={SEEDS}")
    print(f"  Config: {LOSS}/{GROW}/{SCORE_FN}  (K4 fp64 fix active via csv_train_t3)")
    print(f"  Binary: {T3_BINARY}")
    print(f"  Baseline references:")
    print(f"    S28 DW@N=1000 mean drift: {S28_DW_N1000_MEAN_DRIFT:.3f}%")
    print(f"    T3  ST@N=50k  mean drift: {T3_ST_N50K_MEAN_DRIFT:.2f}%")
    print()

    results    = []
    raw_rows   = []

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
        ratio     = mlx_rmse / cpu_rmse
        print(f"  MLX={mlx_rmse:.8f}  CPU={cpu_rmse:.8f}  "
              f"ratio={ratio:.6f}  drift={drift_pct:.4f}%  wall={wall_secs:.1f}s")
        results.append(drift_pct)
        raw_rows.append({
            "seed":      seed,
            "mlx_rmse":  mlx_rmse,
            "cpu_rmse":  cpu_rmse,
            "ratio":     ratio,
            "drift_pct": drift_pct,
            "wall_secs": wall_secs,
        })

        # Write per-seed file immediately (so partial results survive a crash)
        per_seed_path = DATA_DIR / f"dw_at_scale_seed{seed}.csv"
        with open(per_seed_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(
                f, fieldnames=["seed","mlx_rmse","cpu_rmse","ratio","drift_pct","wall_secs"]
            )
            writer.writeheader()
            writer.writerow(raw_rows[-1])

    # Write aggregated summary CSV
    summary_path = DATA_DIR / "dw_at_scale_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=["seed","mlx_rmse","cpu_rmse","ratio","drift_pct","wall_secs"]
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\nRaw data: {summary_path}")

    # Aggregate
    mean_drift = sum(results) / len(results)
    max_drift  = max(results)
    min_drift  = min(results)
    verdict    = interpret(mean_drift)

    print(f"\n=== S30-V5-DW-AT-SCALE RESULT ===")
    print(f"  Seeds:          {SEEDS}")
    print(f"  Per-seed drift: {[f'{d:.4f}%' for d in results]}")
    print(f"  Mean drift:     {mean_drift:.4f}%")
    print(f"  Max drift:      {max_drift:.4f}%")
    print(f"  Min drift:      {min_drift:.4f}%")
    print()
    print(f"  Comparison:")
    print(f"    DW@N=1000  (S28 baseline): {S28_DW_N1000_MEAN_DRIFT:.3f}%")
    print(f"    DW@N=50000 (this run):     {mean_drift:.4f}%")
    print(f"    ST@N=50000 (T3 G3a):       {T3_ST_N50K_MEAN_DRIFT:.2f}%")
    print()
    print(f"  Thresholds: catastrophic >= {CATASTROPHIC_THRESHOLD}%  |  "
          f"contained <= {CONTAINED_THRESHOLD}%")
    print(f"  L0 hypothesis verdict: {verdict}")

    if verdict == "CONSISTENT":
        print("\nInterpretation: DW also breaks catastrophically at N=50k. "
              "L0 cancellation error accumulates with N regardless of grow policy. "
              "S31 histogram-fix plan is on solid ground.")
    elif verdict == "FALSIFIES":
        print("\nInterpretation: DW stays contained at N=50k. "
              "L0 scale effect cannot explain why ST fails at 53%. "
              "A grow-policy-specific mechanism (joint vs per-leaf denominator) "
              "must be the dominant driver. S31 must widen scope.")
    else:
        print(f"\nInterpretation: Mixed signal ({mean_drift:.1f}% in 5-20% zone). "
              "L0 is a contributor but is not the sole driver of ST's 53% failure. "
              "S31 should address both L0 precision and ST-specific denominator aggregation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
