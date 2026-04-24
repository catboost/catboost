#!/usr/bin/env python3
"""
S30-D2-INSTRUMENT runner and post-processor.

Measures three quantities at iter=1 on the ST anchor cell
(N=50000, depth=6, bins=128, seeds={42,43,44}, RMSE, SymmetricTree, Cosine):

  M1 — Gain scalar residual post-cast (L3)
       gain_scalar_seedN_depthD.csv: per (feat, bin)
       gain_f32, gain_f64, gain_abs_residual
       Written by csv_train_d2 binary.

  M2 — Leaf-value sum residual (L5)
       leaf_sum_seedN.csv: per leaf
       Written by csv_train_d2 binary (same as T1/T2 dumpLeaf block, new outDir).

  M3 — Argmax flip count
       argmax_flip_seedN.csv: per depth level
       chosen_feat_mlx, chosen_bin_mlx, chosen_feat_fp64, chosen_bin_fp64, flipped
       Computed here by post-processing the gain_scalar CSVs.

       The argmax shadow is a pure re-argmax on the already-dumped gain_f32 and
       gain_f64 columns — no second pass through FindBestSplit. This is exact
       because the winning split in FindBestSplit is determined solely by argmax
       over all (feat, bin) tuples (noise is zero per DEC-028 for SymmetricTree
       when randomStrength=0.0).

Build command for csv_train_d2 (expected at repo root):
  clang++ -std=c++17 -O2 -DCOSINE_D2_INSTRUMENT \\
    -I. -I/opt/homebrew/opt/mlx/include \\
    -L/opt/homebrew/opt/mlx/lib -lmlx \\
    -framework Metal -framework Foundation -Wno-c++20-extensions \\
    catboost/mlx/tests/csv_train.cpp -o csv_train_d2

Output: docs/sprint30/d2-stack-instrument/data/
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BINARY    = REPO_ROOT / "csv_train_d2"
DATA_DIR  = REPO_ROOT / "docs" / "sprint30" / "d2-stack-instrument" / "data"

# ST anchor cell parameters — identical to T1/T2/T3
N         = 50_000
DEPTH     = 6
BINS      = 128
LR        = 0.03
LOSS      = "rmse"
GROW      = "SymmetricTree"
SCORE_FN  = "Cosine"
ITERS     = 1
SEEDS     = [42, 43, 44]


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
    """Run the D2-instrumented binary for one seed. Returns True on success."""
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
      - fp32 argmax: (feat, bin) with maximum gain_f32
      - fp64 argmax: (feat, bin) with maximum gain_f64
      - flipped: True if the two argmaxes differ

    Returns list of dicts, one per depth:
      {depth, chosen_feat_mlx, chosen_bin_mlx, chosen_feat_fp64, chosen_bin_fp64, flipped}
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

        # Find fp32 argmax
        best_f32_row = max(rows, key=lambda r: float(r["gain_f32"]))
        feat_mlx = int(best_f32_row["feat_idx"])
        bin_mlx  = int(best_f32_row["bin"])

        # Find fp64 argmax
        best_f64_row = max(rows, key=lambda r: float(r["gain_f64"]))
        feat_fp64 = int(best_f64_row["feat_idx"])
        bin_fp64  = int(best_f64_row["bin"])

        flipped = (feat_mlx != feat_fp64) or (bin_mlx != bin_fp64)
        results.append({
            "depth": depth,
            "chosen_feat_mlx":  feat_mlx,
            "chosen_bin_mlx":   bin_mlx,
            "chosen_feat_fp64": feat_fp64,
            "chosen_bin_fp64":  bin_fp64,
            "flipped":          flipped,
        })

    return results


def write_argmax_flip_csv(seed: int, flips: list) -> None:
    path = DATA_DIR / f"argmax_flip_seed{seed}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["depth", "chosen_feat_mlx", "chosen_bin_mlx",
                           "chosen_feat_fp64", "chosen_bin_fp64", "flipped"])
        writer.writeheader()
        writer.writerows(flips)
    print(f"[D2] wrote {len(flips)} rows -> {path}")


def read_leaf_residuals(seed: int) -> dict:
    """Read leaf_sum CSV written by the D2 binary. Returns summary stats."""
    path = DATA_DIR / f"leaf_sum_seed{seed}.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    gsum_res   = [float(r["gSum_residual"])   for r in rows]
    leafval_res = [float(r["leafVal_residual"]) for r in rows]
    return {
        "n_leaves":         len(rows),
        "gSum_max":         max(gsum_res),
        "gSum_mean":        sum(gsum_res) / len(gsum_res),
        "leafVal_max":      max(leafval_res),
        "leafVal_mean":     sum(leafval_res) / len(leafval_res),
    }


def summarise() -> None:
    """Print the full M1/M2/M3 measurement table."""
    print("\n\n=== D2 FULL-STACK RESIDUAL SUMMARY ===\n")

    # ── M1: Gain scalar residual (L3) ──
    print("M1 — Gain scalar residual post-cast (L3)")
    print(f"  {'seed':<6} {'depth':<7} {'n':<8} {'max_res':<14} {'mean_res':<14} {'p99_res':<14}")
    print("  " + "-" * 63)
    all_m1_max = []
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
            p99 = vals_sorted[int(0.99 * (len(vals_sorted) - 1))]
            max_v = max(vals)
            all_m1_max.append(max_v)
            print(f"  {seed:<6} {depth:<7} {len(vals):<8} "
                  f"{max_v:<14.4e} {sum(vals)/len(vals):<14.4e} {p99:<14.4e}")
    if all_m1_max:
        print(f"\n  Global M1 max gain residual: {max(all_m1_max):.4e}")

    # ── M2: Leaf-value sum residual (L5) ──
    print("\nM2 — Leaf-value sum residual (L5)")
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
    if all_m2_leafval_max:
        print(f"\n  Global M2 max leafVal residual: {max(all_m2_leafval_max):.4e}")

    # ── M3: Argmax flip count ──
    print("\nM3 — Argmax flip count (fp32 vs fp64 best-split per depth)")
    print(f"  {'seed':<6} {'depth':<7} {'feat_mlx':<10} {'bin_mlx':<9} "
          f"{'feat_f64':<10} {'bin_f64':<9} {'flipped':<8}")
    print("  " + "-" * 60)
    total_flips = 0
    total_decisions = 0
    for seed in SEEDS:
        flips = compute_argmax_flips(seed)
        write_argmax_flip_csv(seed, flips)
        for rec in flips:
            total_decisions += 1
            flipped_mark = "YES" if rec["flipped"] else "no"
            if rec["flipped"]:
                total_flips += 1
            print(f"  {seed:<6} {rec['depth']:<7} {rec['chosen_feat_mlx']:<10} "
                  f"{rec['chosen_bin_mlx']:<9} {rec['chosen_feat_fp64']:<10} "
                  f"{rec['chosen_bin_fp64']:<9} {flipped_mark:<8}")

    flip_rate = total_flips / total_decisions if total_decisions > 0 else 0.0
    print(f"\n  Total flips: {total_flips} / {total_decisions} decisions "
          f"across {len(SEEDS)} seeds  (rate = {flip_rate:.1%})")

    # ── Layer ranking ──
    print("\n=== LAYER RANKING ===")
    if all_m1_max:
        m1_max = max(all_m1_max)
    else:
        m1_max = float("nan")
    if all_m2_leafval_max:
        m2_max = max(all_m2_leafval_max)
    else:
        m2_max = float("nan")

    print(f"  L3 (gain_scalar post-cast)  max residual: {m1_max:.4e}  "
          f"(prior T2: 3.81e-6 — verify consistency)")
    print(f"  L5 (leaf_val Newton step)   max residual: {m2_max:.4e}  "
          f"(prior T1: ~4e-8)")
    print(f"  L1/L2 (cosNum/cosDen accum) max residual: ~4.07e-3  "
          f"(T1 measured; suppressed by K4 to L3 floor)")
    print(f"  M3 flip rate:               {flip_rate:.1%}  "
          f"({total_flips} flips / {total_decisions} decisions)")

    print("\n=== INTERPRETATION ===")
    if total_flips == 0:
        print("  M3: ZERO argmax flips. L3 residual (~3.81e-6) is NOT flipping split choices.")
        print("  Implication: The 53% trajectory drift cannot originate from L3 gain-cast.")
        print("  The 12.5x gain reduction from K4 did not change split ordering.")
        print("  The drift must compound from a source other than L3 split-selection flips.")
    else:
        print(f"  M3: {total_flips} argmax flips detected.")
        print(f"  Implication: L3 gain residual IS causing split-selection flips at rate {flip_rate:.1%}.")
    print()


def main():
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found. Build with:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DCOSINE_D2_INSTRUMENT \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_d2", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for seed in SEEDS:
        ok = run_seed(seed)
        if not ok:
            all_ok = False

    if all_ok:
        summarise()
        print("All seeds complete. Check docs/sprint30/d2-stack-instrument/data/ for artifacts.")
    else:
        print("SOME SEEDS FAILED — partial results in data/", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
