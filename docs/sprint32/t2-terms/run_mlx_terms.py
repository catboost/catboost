#!/usr/bin/env python3
"""
S32-T2-INSTRUMENT: run the MLX-side Cosine term audit binary.

Runs csv_train_t2_terms (compiled with -DCOSINE_TERM_AUDIT) on the canonical
S26 data (N=50000, 20 features, seed=42) at the S28 anchor configuration:
  - depth=6, bins=128, iters=1, loss=rmse, grow=SymmetricTree, Cosine, rs=0

Produces:
  docs/sprint32/t2-terms/data/mlx_terms_seed42_depth0.csv

Schema: feat,bin,sumLeft,sumRight,weightLeft,weightRight,lambda,
        cosNum_term,cosDen_term,gain

At depth=0 N=50k RMSE ST: numPartitions=1, K=1.
Each row = one (feat, bin) candidate; all per-(p,k) contributions = single term.

Usage:
  python docs/sprint32/t2-terms/run_mlx_terms.py [--seed SEED]
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
BINARY    = REPO_ROOT / "csv_train_t2_terms"
DATA_DIR  = REPO_ROOT / "docs" / "sprint32" / "t2-terms" / "data"

# S28 anchor parameters
N         = 50_000
DEPTH     = 6
BINS      = 128
LR        = 0.03
LOSS      = "rmse"
GROW      = "SymmetricTree"
SCORE_FN  = "Cosine"
ITERS     = 1
ALL_SEEDS = [42]   # T2 focuses on seed=42 as the primary comparison seed


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
    """Run COSINE_TERM_AUDIT binary for one seed. Returns True on success."""
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
        env["COSINE_TERM_AUDIT_OUTDIR"] = str(DATA_DIR)

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.stderr:
            for line in result.stderr.splitlines():
                if "[TERM_AUDIT]" in line or "ERROR" in line.upper():
                    print(f"  {line}", file=sys.stderr)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})", file=sys.stderr)
            print(f"  STDERR: {result.stderr[-800:]}", file=sys.stderr)
            return False

        out_path = DATA_DIR / f"mlx_terms_seed{seed}_depth0.csv"
        if out_path.exists():
            import csv as _csv
            with open(out_path) as fh:
                rows = list(_csv.DictReader(fh))
            print(f"  Output: {out_path} ({len(rows)} candidate rows)")
            # Quick sanity print: top-3 by gain
            rows_sorted = sorted(rows, key=lambda r: float(r["gain"]), reverse=True)
            print("  Top-3 candidates by MLX gain:")
            for r in rows_sorted[:3]:
                print(f"    feat={r['feat']} bin={r['bin']} "
                      f"wL={float(r['weightLeft']):.1f} wR={float(r['weightRight']):.1f} "
                      f"gL={float(r['sumLeft']):.6f} gR={float(r['sumRight']):.6f} "
                      f"gain={float(r['gain']):.8f}")
        else:
            print(f"  WARNING: expected output not found: {out_path}", file=sys.stderr)
            return False
        return True

    finally:
        csv_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="S32-T2 MLX term audit runner")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to run (default: seed=42)")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:", file=sys.stderr)
        print("  ./docs/sprint32/t2-terms/build_mlx_terms.sh", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else ALL_SEEDS
    ok = all(run_seed(s) for s in seeds)
    if ok:
        print(f"\nMLX term dump complete. Data in {DATA_DIR}/")
    else:
        print("\nOne or more seeds FAILED.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
