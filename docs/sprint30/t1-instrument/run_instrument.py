#!/usr/bin/env python3
"""
S30-T1-INSTRUMENT runner script.

Generates the ST anchor cell data (N=50000, depth=6, bins=128,
score_function='Cosine', grow_policy='SymmetricTree', iters=1)
and runs csv_train_instrument to dump per-accumulator fp32/fp64 residuals.

Seeds: 42, 43, 44 (3 seeds per DEC-035 T1 spec minimum)

Output: docs/sprint30/t1-instrument/data/  (CSV files per seed per accumulator)
"""

import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BINARY    = REPO_ROOT / "csv_train_instrument"
DATA_DIR  = REPO_ROOT / "docs" / "sprint30" / "t1-instrument" / "data"

# ST anchor cell parameters (from docs/sprint28/fu-obliv-dispatch/t7-gate-report.md G6a)
N         = 50_000
DEPTH     = 6
BINS      = 128
LR        = 0.03
LOSS      = "rmse"
GROW      = "SymmetricTree"
SCORE_FN  = "Cosine"
ITERS     = 1
SEEDS     = [42, 43, 44]


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Canonical S26 data: 20 features, signal in f0 and f1, 10% noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    """Write X, y to a CSV with header row: f0,f1,...,fN,target."""
    n_feat = X.shape[1]
    header = [f"f{i}" for i in range(n_feat)] + ["target"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(y)):
            writer.writerow(list(X[i]) + [y[i]])


def run_seed(seed: int) -> bool:
    """Run instrumented binary for one seed. Returns True on success."""
    print(f"\n--- seed={seed} ---")
    X, y = make_data(N, seed)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
        csv_path = tf.name

    try:
        write_csv(Path(csv_path), X, y)

        cmd = [
            str(BINARY),
            csv_path,
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
        env["COSINE_RESIDUAL_OUTDIR"] = str(DATA_DIR)

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False, env=env)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})", file=sys.stderr)
            return False
        return True
    finally:
        os.unlink(csv_path)


def summarise() -> None:
    """Read all generated CSVs and print per-accumulator residual table."""
    print("\n\n=== RESIDUAL SUMMARY ===\n")
    print(f"{'Accumulator':<35} {'seed':<6} {'n':<8} {'max_res':<14} {'mean_res':<14} {'p99_res':<14}")
    print("-" * 91)

    for seed in SEEDS:
        # cosDen / cosNum residuals (multiple depths)
        for depth in range(DEPTH + 1):
            cos_path = DATA_DIR / f"cos_accum_seed{seed}_depth{depth}.csv"
            if cos_path.exists():
                with open(cos_path) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    for col, label in [
                        ("cosNum_abs_residual", f"cosNum (depth={depth})"),
                        ("cosDen_abs_residual", f"cosDen (depth={depth})"),
                        ("gain_abs_residual",   f"gain   (depth={depth})"),
                    ]:
                        vals = [float(r[col]) for r in rows]
                        vals_sorted = sorted(vals)
                        p99 = vals_sorted[int(0.99 * (len(vals_sorted) - 1))]
                        print(f"  {label:<33} {seed:<6} {len(vals):<8} "
                              f"{max(vals):<14.4e} {sum(vals)/len(vals):<14.4e} {p99:<14.4e}")

        # leaf-sum residuals
        leaf_path = DATA_DIR / f"leaf_sum_seed{seed}.csv"
        if leaf_path.exists():
            with open(leaf_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                for col, label in [
                    ("gSum_residual",     "gSum (leaf scatter_add)"),
                    ("hSum_residual",     "hSum (leaf scatter_add)"),
                    ("leafVal_residual",  "leafVal (Newton step)"),
                ]:
                    vals = [float(r[col]) for r in rows]
                    vals_sorted = sorted(vals)
                    p99 = vals_sorted[int(0.99 * (len(vals_sorted) - 1))]
                    print(f"  {label:<33} {seed:<6} {len(vals):<8} "
                          f"{max(vals):<14.4e} {sum(vals)/len(vals):<14.4e} {p99:<14.4e}")

        # approx-update residuals
        approx_path = DATA_DIR / f"approx_update_seed{seed}.csv"
        if approx_path.exists():
            with open(approx_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                vals = [float(r["inc_residual"]) for r in rows]
                vals_sorted = sorted(vals)
                p99 = vals_sorted[int(0.99 * (len(vals_sorted) - 1))]
                print(f"  {'approxUpdate (cursor inc)':<33} {seed:<6} {len(vals):<8} "
                      f"{max(vals):<14.4e} {sum(vals)/len(vals):<14.4e} {p99:<14.4e}")

    print()


def main():
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DCOSINE_RESIDUAL_INSTRUMENT \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_instrument", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for seed in SEEDS:
        ok = run_seed(seed)
        if not ok:
            all_ok = False

    summarise()

    if not all_ok:
        print("SOME SEEDS FAILED", file=sys.stderr)
        sys.exit(1)

    print("All seeds complete. Check docs/sprint30/t1-instrument/data/ for CSV artifacts.")


if __name__ == "__main__":
    main()
