#!/usr/bin/env python3
"""
S30-D2-REDUX runner and post-processor.

Identical cell parameters to D2 (N=50000, depth=6, bins=128, seeds={42,43,44},
RMSE, SymmetricTree, Cosine, 1 iteration).

Key difference from D2: the csv_train_d2_redux binary has corrected gain_f32
instrumentation.  gain_f32 is now the TRUE pre-K4 fp32 path:
    float(cosNum_f32_shadow / sqrtf(cosDen_f32_shadow))
NOT float(cosNum_d / sqrt(cosDen_d)) as D2 mistakenly used.

Measurements:
  M1 — L3 gain residual: |gain_f32_path - gain_f64|
       This now measures fp32 accumulation path divergence, not cast ULP.
       Expected: ~1e-3 to 1e-4 (from V2 audit prediction) vs D2's floor 3.81e-6.

  M2 — L5 leaf-value residual (unchanged from D2 — already correct).

  M3 — L4 argmax flip count: independently argmax gain_f32 and gain_f64 columns.
       Now measures: "would the pre-K4 fp32 path have chosen a different split
       than the fp64 path?" NOT "does float(x) == x?" as D2 measured.

Build:
  clang++ -std=c++17 -O2 -DCOSINE_D2_INSTRUMENT \\
    -I. -I/opt/homebrew/opt/mlx/include \\
    -L/opt/homebrew/opt/mlx/lib -lmlx \\
    -framework Metal -framework Foundation -Wno-c++20-extensions \\
    catboost/mlx/tests/csv_train.cpp -o csv_train_d2_redux

Output: docs/sprint30/d2-redux/data/
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BINARY    = REPO_ROOT / "csv_train_d2_redux"
DATA_DIR  = REPO_ROOT / "docs" / "sprint30" / "d2-redux" / "data"

# ST anchor cell — identical to D2
N        = 50_000
DEPTH    = 6
BINS     = 128
LR       = 0.03
LOSS     = "rmse"
GROW     = "SymmetricTree"
SCORE_FN = "Cosine"
ITERS    = 1
SEEDS    = [42, 43, 44]


def make_data(n: int, seed: int) -> tuple:
    """Canonical S26 data: 20 features, signal in f0 and f1, 10% noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    n_feat = X.shape[1]
    header = [f"f{i}" for i in range(n_feat)] + ["target"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(y)):
            writer.writerow(list(X[i]) + [y[i]])


def run_seed(seed: int) -> bool:
    """Run the D2-redux binary for one seed. Returns True on success."""
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


def compute_argmax_flips(seed: int) -> list:
    """
    Post-process gain_scalar CSVs to compute argmax flip count per depth.

    For each depth level d, read gain_scalar_seedN_depthD.csv and find:
      - fp32 argmax: (feat, bin) with maximum gain_f32  [TRUE pre-K4 fp32 path]
      - fp64 argmax: (feat, bin) with maximum gain_f64  [TRUE fp64 path]
      - flipped: True if the two argmaxes differ

    This is now a meaningful test: gain_f32 comes from float accumulation,
    not from a cast of the fp64 result.
    """
    results = []
    for depth in range(DEPTH):
        path = DATA_DIR / f"gain_scalar_seed{seed}_depth{depth}.csv"
        if not path.exists():
            print(f"  [M3] WARNING: {path} not found", file=sys.stderr)
            continue

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        best_f32_row = max(rows, key=lambda r: float(r["gain_f32"]))
        feat_mlx = int(best_f32_row["feat_idx"])
        bin_mlx  = int(best_f32_row["bin"])

        best_f64_row = max(rows, key=lambda r: float(r["gain_f64"]))
        feat_fp64 = int(best_f64_row["feat_idx"])
        bin_fp64  = int(best_f64_row["bin"])

        flipped = (feat_mlx != feat_fp64) or (bin_mlx != bin_fp64)
        results.append({
            "depth": depth,
            "chosen_feat_f32":  feat_mlx,
            "chosen_bin_f32":   bin_mlx,
            "chosen_feat_fp64": feat_fp64,
            "chosen_bin_fp64":  bin_fp64,
            "flipped":          flipped,
        })

    return results


def write_argmax_flip_csv(seed: int, flips: list) -> None:
    path = DATA_DIR / f"argmax_flip_seed{seed}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["depth", "chosen_feat_f32", "chosen_bin_f32",
                           "chosen_feat_fp64", "chosen_bin_fp64", "flipped"])
        writer.writeheader()
        writer.writerows(flips)
    print(f"[D2-redux] wrote {len(flips)} rows -> {path}")


def read_leaf_residuals(seed: int) -> dict:
    path = DATA_DIR / f"leaf_sum_seed{seed}.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    gsum_res    = [float(r["gSum_residual"])   for r in rows]
    leafval_res = [float(r["leafVal_residual"]) for r in rows]
    return {
        "n_leaves":    len(rows),
        "gSum_max":    max(gsum_res),
        "gSum_mean":   sum(gsum_res) / len(gsum_res),
        "leafVal_max": max(leafval_res),
        "leafVal_mean":sum(leafval_res) / len(leafval_res),
    }


def summarise() -> dict:
    """Print the full M1/M2/M3 measurement table. Returns summary dict for verdict."""
    print("\n\n=== D2-REDUX FULL-STACK RESIDUAL SUMMARY ===\n")
    print("NOTE: gain_f32 is now the TRUE pre-K4 fp32 path (V2 corrected methodology)")
    print("      gain_f64 is the true fp64 path\n")

    # M1: L3 gain residual
    print("M1 — L3 gain residual: |fp32-path - fp64-path|  (CORRECTED — not cast ULP)")
    print(f"  {'seed':<6} {'depth':<7} {'n':<8} {'max_res':<14} {'mean_res':<14} {'p99_res':<14}")
    print("  " + "-" * 63)
    all_m1_max  = []
    all_m1_mean = []
    per_cell    = {}  # (seed, depth) -> (max, mean, p99)
    for seed in SEEDS:
        for depth in range(DEPTH):
            path = DATA_DIR / f"gain_scalar_seed{seed}_depth{depth}.csv"
            if not path.exists():
                continue
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue
            vals = [float(r["gain_abs_residual"]) for r in rows]
            vals_sorted = sorted(vals)
            p99  = vals_sorted[int(0.99 * (len(vals_sorted) - 1))]
            maxv = max(vals)
            meanv = sum(vals) / len(vals)
            all_m1_max.append(maxv)
            all_m1_mean.append(meanv)
            per_cell[(seed, depth)] = (maxv, meanv, p99)
            print(f"  {seed:<6} {depth:<7} {len(vals):<8} "
                  f"{maxv:<14.4e} {meanv:<14.4e} {p99:<14.4e}")

    global_m1_max = max(all_m1_max) if all_m1_max else float("nan")
    global_m1_mean = sum(all_m1_mean) / len(all_m1_mean) if all_m1_mean else float("nan")
    print(f"\n  Global M1 max: {global_m1_max:.4e}   Global M1 mean-of-means: {global_m1_mean:.4e}")
    print(f"  (D2 prior — biased cast-ULP floor: 3.81e-6)")

    # M2: L5 leaf-value residual
    print("\nM2 — L5 leaf-value residual (methodology unchanged from D2 — correct)")
    print(f"  {'seed':<6} {'n_leaves':<10} {'gSum_max':<14} {'gSum_mean':<14} "
          f"{'leafVal_max':<14} {'leafVal_mean':<14}")
    print("  " + "-" * 72)
    all_m2_leafval_max = []
    for seed in SEEDS:
        stats = read_leaf_residuals(seed)
        if not stats:
            print(f"  {seed:<6} (no leaf_sum CSV)")
            continue
        lv_max = stats["leafVal_max"]
        all_m2_leafval_max.append(lv_max)
        print(f"  {seed:<6} {stats['n_leaves']:<10} {stats['gSum_max']:<14.4e} "
              f"{stats['gSum_mean']:<14.4e} {lv_max:<14.4e} {stats['leafVal_mean']:<14.4e}")
    global_m2_max = max(all_m2_leafval_max) if all_m2_leafval_max else float("nan")
    print(f"\n  Global M2 max leafVal residual: {global_m2_max:.4e}")

    # M3: L4 argmax flips
    print("\nM3 — L4 argmax flip count (pre-K4 fp32 path vs fp64 path)")
    print("     NOW MEANINGFUL: gain_f32 uses float accumulators, not a cast of fp64")
    print(f"  {'seed':<6} {'depth':<7} {'feat_f32':<10} {'bin_f32':<9} "
          f"{'feat_f64':<10} {'bin_f64':<9} {'flipped':<8}")
    print("  " + "-" * 60)
    total_flips     = 0
    total_decisions = 0
    flip_detail     = []
    for seed in SEEDS:
        flips = compute_argmax_flips(seed)
        write_argmax_flip_csv(seed, flips)
        for rec in flips:
            total_decisions += 1
            flipped_mark = "YES" if rec["flipped"] else "no"
            if rec["flipped"]:
                total_flips += 1
            flip_detail.append((seed, rec["depth"], rec["flipped"]))
            print(f"  {seed:<6} {rec['depth']:<7} {rec['chosen_feat_f32']:<10} "
                  f"{rec['chosen_bin_f32']:<9} {rec['chosen_feat_fp64']:<10} "
                  f"{rec['chosen_bin_fp64']:<9} {flipped_mark:<8}")

    flip_rate = total_flips / total_decisions if total_decisions > 0 else 0.0
    print(f"\n  Total flips: {total_flips} / {total_decisions} decisions "
          f"across {len(SEEDS)} seeds  (rate = {flip_rate:.1%})")
    print(f"  (D2 prior — biased: 0 / 18 = 0.0%)")

    return {
        "global_m1_max":  global_m1_max,
        "global_m1_mean": global_m1_mean,
        "global_m2_max":  global_m2_max,
        "total_flips":    total_flips,
        "total_decisions":total_decisions,
        "flip_rate":      flip_rate,
        "per_cell_m1":    per_cell,
    }


def main():
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DCOSINE_D2_INSTRUMENT \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_d2_redux", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for seed in SEEDS:
        ok = run_seed(seed)
        if not ok:
            all_ok = False

    if all_ok:
        summarise()
        print("\nAll seeds complete. Check docs/sprint30/d2-redux/data/ for artifacts.")
    else:
        print("SOME SEEDS FAILED — partial results in data/", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
