#!/usr/bin/env python3
"""
S30-V6-N500-CONFIRMER: Cheap falsification check for the L1 fp32 histogram hypothesis.

The L1 hypothesis predicts that per-bin histogram accumulation error scales linearly
with N:
    per_bin_error ≈ N × ε_f32 / bins

At N=50000: ≈ 50000 × 1.2e-7 / 128 ≈ 4.7e-5  (~10× the gain boundary → drift expected)
At N=500:   ≈ 500 × 1.2e-7 / 128 ≈ 4.7e-7   (~10000× smaller → drift should collapse)

Prediction: if L1 is the binding mechanism, G3a should PASS at N=500 (aggregate drift
< 2%), because the per-bin error is far below the gain boundary.

Reference: V1 (docs/sprint30/v1-drift-vs-n/verdict.md) already measured N=1000 at
52.66% drift. This V6 run extends down to N=500 (and optionally N=5000 as midpoint) to
complete the falsification record.

Config: loss=RMSE, grow_policy=SymmetricTree, score_function=Cosine,
        depth=6, bins=128, iterations=50, seeds={42,43,44,45,46}.
Binary: csv_train_t3 (K4 fp64 Kahan fix + Fix2 fp64 gain widening active).

Output:
  data/v6_n500.csv         -- per-seed results at N=500
  data/v6_n5000.csv        -- per-seed results at N=5000 (if run)
  stdout                   -- gate verdict
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

# V6 sweep parameters
# Primary: N=500 (mandatory).  N=5000 included for scaling sanity-check.
N_VALUES  = [500, 5000]
SEEDS     = [42, 43, 44, 45, 46]   # Full G3a seed set for direct comparison

# Fixed cell params — mirror T3/G3a exactly (only N changes)
DEPTH    = 6
BINS     = 128
LR       = 0.03
LOSS     = "rmse"
GROW     = "SymmetricTree"
SCORE_FN = "Cosine"
ITERS    = 50

G3a_PASS_THRESHOLD = 2.0  # percent aggregate drift — same as T3 G3a


def make_data(n: int, seed: int):
    """Canonical S26 data generator: 20 features, signal in f0/f1, 10% noise.
    Identical to T3/G3a and V1 — changing only n.
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
        "--depth",       str(DEPTH),
        "--lr",          str(LR),
        "--bins",        str(BINS),
        "--loss",        LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed",        str(seed),
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


def run_for_n(n: int) -> list[dict]:
    """Run all seeds for a given N; return list of result dicts."""
    rows = []
    print(f"\n=== N={n} ===")
    for seed in SEEDS:
        print(f"  seed={seed} ...", end="", flush=True)
        X, y = make_data(n, seed)

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            data_path = Path(tf.name)
        try:
            write_csv(data_path, X, y)
            mlx_rmse, wall_secs = run_mlx(data_path, seed)
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
        rows.append({
            "N": n,
            "seed": seed,
            "mlx_rmse": mlx_rmse,
            "cpu_rmse": cpu_rmse,
            "ratio": ratio,
            "drift_pct": drift_pct,
            "wall_secs": wall_secs,
        })
    return rows


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

    print("S30-V6-N500-CONFIRMER — L1 fp32 histogram cheap falsification check")
    print(f"  Binary:  csv_train_t3 (K4 fp64 fix + Fix2 gain widening active)")
    print(f"  N sweep: {N_VALUES}")
    print(f"  Seeds:   {SEEDS}")
    print(f"  Config:  {LOSS}/{GROW}/Cosine, depth={DEPTH}, bins={BINS}, iters={ITERS}")
    print(f"  G3a PASS threshold: aggregate drift < {G3a_PASS_THRESHOLD}%")
    print()
    print("  L1 hypothesis prediction: drift ~ N (linear scaling)")
    print(f"  At N=500:  expected per-bin error ≈ 4.7e-7 → drift should be << 2% (PASS)")
    print(f"  At N=50k:  measured per-bin error ≈ 4.7e-5 → drift = 53.30% (FAIL)")

    all_rows_by_n: dict[int, list[dict]] = {}

    for n in N_VALUES:
        rows = run_for_n(n)
        all_rows_by_n[n] = rows

        # Write per-N CSV
        n_tag = str(n).replace("000", "k") if n >= 1000 else str(n)
        out_csv = DATA_DIR / f"v6_n{n_tag}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv_mod.DictWriter(
                f, fieldnames=["N", "seed", "mlx_rmse", "cpu_rmse", "ratio", "drift_pct", "wall_secs"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> written: {out_csv}")

    # Summary table
    print("\n=== V6 drift summary (ST+Cosine, per N) ===")
    print(f"  {'N':>7}  " + "  ".join(f"s{s:02d}" for s in SEEDS) + "  {'mean':>8}")
    print(f"  {'-'*7}  " + "  ".join(["-"*6] * len(SEEDS)) + "  {'-'*8}")

    means_by_n = {}
    for n in N_VALUES:
        rows = all_rows_by_n[n]
        drifts = [r["drift_pct"] for r in rows]
        mean_d = sum(drifts) / len(drifts)
        means_by_n[n] = mean_d
        seed_str = "  ".join(f"{d:6.2f}%" for d in drifts)
        print(f"  {n:>7}  {seed_str}  {mean_d:7.2f}%")

    # Reference rows from prior measurements (V1 + T3/Fix2)
    print()
    print("  --- reference rows (prior measurements) ---")
    ref_table = [
        (500,    None,  "V6 this run"),
        (1000,   52.66, "V1-DRIFT-VS-N (seeds 42/43/44 mean)"),
        (5000,   53.29, "V1-DRIFT-VS-N (seeds 42/43/44 mean)"),
        (10000,  53.43, "V1-DRIFT-VS-N"),
        (25000,  53.13, "V1-DRIFT-VS-N"),
        (50000,  53.30, "T3/Fix2 baseline (seeds 42-46 mean)"),
    ]
    for n_ref, d_ref, src in ref_table:
        if d_ref is not None:
            print(f"  {n_ref:>7}  {d_ref:7.2f}%  [{src}]")

    # G3a gate check at N=500
    n500_rows = all_rows_by_n[500]
    n500_drifts = [r["drift_pct"] for r in n500_rows]
    n500_agg = sum(n500_drifts) / len(n500_drifts)
    n500_pass = n500_agg < G3a_PASS_THRESHOLD

    print(f"\n=== G3a gate check at N=500 ===")
    for r in n500_rows:
        print(f"  seed={r['seed']}:  MLX={r['mlx_rmse']:.8f}  CPU={r['cpu_rmse']:.8f}"
              f"  ratio={r['ratio']:.6f}  drift={r['drift_pct']:.4f}%")
    print(f"  Aggregate drift: {n500_agg:.4f}%")
    print(f"  Threshold:       {G3a_PASS_THRESHOLD}%")
    print(f"  G3a at N=500:    {'PASS' if n500_pass else 'FAIL'}")

    # Scaling verdict
    # Compare N=500 to N=50000 (reference 53.30%)
    n50k_ref = 53.30
    reduction_factor = n50k_ref / n500_agg if n500_agg > 0 else float("inf")
    expected_linear_reduction = 50000 / 500  # = 100x if L1 is linear in N

    print(f"\n=== L1 hypothesis scaling check ===")
    print(f"  N=50000 baseline:          {n50k_ref:.2f}% (T3/Fix2)")
    print(f"  N=500 measured:            {n500_agg:.2f}%")
    print(f"  Actual reduction factor:   {reduction_factor:.1f}x")
    print(f"  Expected if L1 dominant:   {expected_linear_reduction:.0f}x (linear in N)")

    if n500_pass:
        print()
        print("  VERDICT: CONFIRMED")
        print("  G3a PASSES at N=500. Drift dropped from 53.30% to < 2.0%.")
        print("  Reduction factor is consistent with L1 linear-in-N scaling.")
        print("  L1 (Metal fp32 histogram accumulation) is binding at N=50k.")
        print("  S31 scope: Metal histogram kernel fp64 accumulation.")
    elif n500_agg < 20.0:
        print()
        print("  VERDICT: PARTIAL")
        print(f"  G3a does not pass (drift={n500_agg:.2f}% >= 2%), but drift is reduced from 53.30%.")
        print("  L1 is contributing but another mechanism persists at small N.")
        print("  S31 scope: needs expanded analysis; L1 is not sole driver.")
    else:
        print()
        print("  VERDICT: FALSIFIED")
        print(f"  G3a FAILS at N=500 (drift={n500_agg:.2f}% ≈ N=50k level).")
        print("  Drift is N-independent — L1 fp32 histogram accumulation is NOT the dominant")
        print("  mechanism. The 53% drift has a systematic, N-independent cause.")
        print("  This is consistent with V1 (N=1000 → 52.66%) and confirms V1's finding.")
        print("  S31 scope: split-selection algorithmic audit (see V1 recommendation).")

    return 0 if n500_pass else 1


if __name__ == "__main__":
    sys.exit(main())
