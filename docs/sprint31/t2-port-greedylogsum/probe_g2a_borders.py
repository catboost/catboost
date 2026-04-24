#!/usr/bin/env python3
"""
S31-T2 G2a: Borders byte-match probe.

Compares borders produced by:
  1. CPU CatBoost (Pool.quantize with border_count=127 — GreedyLogSum default)
  2. MLX csv_train_dump_borders binary (C++ GreedyLogSum port, -DCATBOOST_MLX_DUMP_BORDERS)

The dump-borders binary outputs "BORDER\t{feature_idx}\t{border_value}" lines then exits.
Both use the same N=5000 Gaussian float32 datasets, max_bins=128.

Target: 100% float32 exact match (np.array_equal) across all 10x10 = 100 pairs.
"""

import collections
import csv as csv_mod
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DUMP_BINARY = REPO_ROOT / "csv_train_dump_borders"

N_DATASETS = 10
N_FEATURES_PER_DATASET = 10
N_DOCS = 5000
MAX_BINS = 128
SEEDS_DATA = list(range(101, 101 + N_DATASETS))


def make_dataset(n_docs: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_docs, n_features)).astype(np.float32)
    rng_y = np.random.default_rng(seed + 1000)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng_y.standard_normal(n_docs) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    n_feat = X.shape[1]
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow([f"f{i}" for i in range(n_feat)] + ["target"])
        for i in range(len(y)):
            w.writerow(list(X[i]) + [float(y[i])])


def get_mlx_borders(data_csv: Path) -> dict:
    """Run csv_train_dump_borders; parse BORDER lines."""
    r = subprocess.run(
        [str(DUMP_BINARY), str(data_csv), "--bins", str(MAX_BINS)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"dump_borders failed: {r.stderr[:300]}")
    borders = collections.defaultdict(list)
    for line in r.stdout.split("\n"):
        if not line.startswith("BORDER"):
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            fi = int(parts[1])
            bv = np.float32(float(parts[2]))
            borders[fi].append(bv)
    return {k: np.array(sorted(v), dtype=np.float32) for k, v in borders.items()}


def get_cpu_catboost_borders(X: np.ndarray, y: np.ndarray) -> dict:
    """CPU CatBoost Pool.quantize + save_quantization_borders."""
    try:
        from catboost import Pool
    except ImportError:
        return None
    pool = Pool(X, y)
    pool.quantize(border_count=MAX_BINS - 1)
    with tempfile.NamedTemporaryFile(suffix=".borders", delete=False) as f:
        bpath = f.name
    try:
        pool.save_quantization_borders(bpath)
        cpu = collections.defaultdict(list)
        with open(bpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    cpu[int(parts[0])].append(np.float32(float(parts[1])))
    finally:
        os.unlink(bpath)
    return {k: np.array(sorted(v), dtype=np.float32) for k, v in cpu.items()}


def main():
    print("G2a — Borders byte-match: C++ MLX port vs CPU CatBoost GreedyLogSum")
    print(f"  Binary: {DUMP_BINARY.name}")
    print(f"  Datasets: {N_DATASETS} x {N_FEATURES_PER_DATASET} features, N={N_DOCS}, MAX_BINS={MAX_BINS}\n")

    if not DUMP_BINARY.exists():
        print(f"ERROR: {DUMP_BINARY} not found. Build with:")
        print("  clang++ -std=c++17 -O2 -DCATBOOST_MLX_DUMP_BORDERS -I. \\")
        print("    -I/opt/homebrew/Cellar/mlx/0.31.1/include \\")
        print("    -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \\")
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\")
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_dump_borders")
        return 1

    total_pairs = 0
    mismatches = 0
    details = []

    for ds_idx, seed in enumerate(SEEDS_DATA):
        X, y = make_dataset(N_DOCS, N_FEATURES_PER_DATASET, seed)

        # Write temp CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            data_path = Path(tf.name)
        try:
            write_csv(data_path, X, y)
            mlx_all = get_mlx_borders(data_path)
        finally:
            os.unlink(data_path)

        cpu_all = get_cpu_catboost_borders(X, y)
        if cpu_all is None:
            print("SKIP — catboost not available")
            return 0

        for feat_idx in range(N_FEATURES_PER_DATASET):
            total_pairs += 1
            mlx_b = mlx_all.get(feat_idx, np.array([], dtype=np.float32))
            cpu_b = cpu_all.get(feat_idx, np.array([], dtype=np.float32))

            len_ok = len(mlx_b) == len(cpu_b)
            val_ok = bool(np.array_equal(mlx_b, cpu_b)) if len_ok else False

            if len_ok and val_ok:
                print(f"  ds={ds_idx} f={feat_idx}: MATCH ({len(mlx_b)} borders)")
            else:
                mismatches += 1
                max_diff = float(np.max(np.abs(mlx_b - cpu_b))) if len_ok and len(mlx_b) > 0 else None
                details.append({
                    "ds": ds_idx, "feat": feat_idx, "seed": seed,
                    "mlx_len": len(mlx_b), "cpu_len": len(cpu_b),
                    "mlx_first5": list(mlx_b[:5]),
                    "cpu_first5": list(cpu_b[:5]),
                    "max_diff": max_diff,
                })

    print(f"\nTotal feature-dataset pairs: {total_pairs}")
    print(f"Exact matches:   {total_pairs - mismatches}")
    print(f"Mismatches:      {mismatches}")

    if mismatches == 0:
        print("\nG2a: PASS — 100% byte-match (C++ MLX == CPU CatBoost GreedyLogSum)")
        return 0
    else:
        print(f"\nG2a: FAIL — {mismatches}/{total_pairs} pairs differ")
        for d in details[:10]:
            print(f"  ds={d['ds']} feat={d['feat']} seed={d['seed']}: "
                  f"mlx_len={d['mlx_len']} cpu_len={d['cpu_len']} "
                  f"max_diff={d['max_diff']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
