#!/usr/bin/env python3
"""
S31-T3b-T1-AUDIT: run the MLX-side instrumented iter=1 dump.

Runs csv_train_t3b (compiled with -DITER1_AUDIT) on the canonical S26 data
(N=50000, 20 features, seeds 42/43/44) at the S28 anchor configuration:
  - depth=6, bins=128, iters=1, loss=rmse, grow_policy=SymmetricTree,
    score_function=Cosine, random_strength=0, bootstrap=no

For each seed, produces:
  docs/sprint31/t3b-audit/data/mlx_splits_seed<N>.json

The JSON contains per-layer records with parent partition stats, top-K=5
candidates, and the winning split tuple.

Usage:
  python docs/sprint31/t3b-audit/run_mlx_audit.py [--seed SEED]

If --seed is omitted, runs all three seeds (42, 43, 44).

Note: build csv_train_t3b first:
  ./docs/sprint31/t3b-audit/build_mlx_audit.sh
"""

import argparse
import csv as csv_mod
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BINARY    = REPO_ROOT / "csv_train_t3b"
DATA_DIR  = REPO_ROOT / "docs" / "sprint31" / "t3b-audit" / "data"

# S28 anchor parameters
N         = 50_000
DEPTH     = 6
BINS      = 128
LR        = 0.03
LOSS      = "rmse"
GROW      = "SymmetricTree"
SCORE_FN  = "Cosine"
ITERS     = 1
ALL_SEEDS = [42, 43, 44]


def make_data(n: int, seed: int):
    """Canonical S26 data: 20 features, signal in f0/f1, 10% noise."""
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


def run_seed(seed: int) -> bool:
    """Run ITER1_AUDIT binary for one seed. Returns True on success."""
    print(f"\n--- seed={seed} ---")
    X, y = make_data(N, seed)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
        csv_path = Path(tf.name)

    try:
        write_csv(csv_path, X, y)

        cmd = [
            str(BINARY),
            str(csv_path),
            "--iterations", str(ITERS),
            "--depth",      str(DEPTH),
            "--lr",         str(LR),
            "--bins",       str(BINS),
            "--loss",       LOSS,
            "--grow-policy", GROW,
            "--score-function", SCORE_FN,
            "--seed",       str(seed),
            "--random-strength", "0",
            "--bootstrap-type", "no",
        ]

        env = os.environ.copy()
        env["ITER1_AUDIT_OUTDIR"] = str(DATA_DIR)

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        # Print stderr (contains [AUDIT] lines) to the console
        if result.stderr:
            for line in result.stderr.splitlines():
                if "[AUDIT]" in line or "ERROR" in line.upper():
                    print(f"  {line}", file=sys.stderr)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})", file=sys.stderr)
            print(f"  STDERR: {result.stderr[-800:]}", file=sys.stderr)
            return False

        out_path = DATA_DIR / f"mlx_splits_seed{seed}.json"
        if out_path.exists():
            print(f"  Output: {out_path}")
        else:
            print(f"  WARNING: expected output not found: {out_path}", file=sys.stderr)
            return False
        return True

    finally:
        csv_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="S31-T3b MLX audit runner")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to run (default: all 42/43/44)")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:", file=sys.stderr)
        print("  ./docs/sprint31/t3b-audit/build_mlx_audit.sh", file=sys.stderr)
        print("Which runs:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DITER1_AUDIT -DCOSINE_T3_MEASURE \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_t3b", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else ALL_SEEDS

    all_ok = True
    for seed in seeds:
        ok = run_seed(seed)
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nSOME SEEDS FAILED", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. MLX audit JSON written to: {DATA_DIR}")
    print("Next: python docs/sprint31/t3b-audit/compare_splits.py")


if __name__ == "__main__":
    main()
