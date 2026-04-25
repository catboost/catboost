#!/usr/bin/env python3
"""
S33-PROBE-C Stage 1: per-feature, per-index border-VALUE comparison
between csv_train.cpp's GreedyLogSum quantization and CatBoost CPU's upfront
quantization grid (Probe-A artifact).

The L4 verdict claimed both pick "border=0.014169" at iter=2 depth=0 for
feature 0, but Probe-A's upfront-grid TSV was never directly compared against
the live csv_train.cpp output. Probe-A explicitly flagged this as an open
question (caveat 3, lines 249-252):

    "The probe does not examine the actual *border values* — only counts. If
     csv_train.cpp's per-feature borders differ in value from CatBoost CPU's
     (e.g., one rounds float32 differently, or uses a different sample to
     compute GreedyLogSum), that would be a separate, also-open question."

This script answers that open question.

Anchor: identical to Probe-A and L3 (np.random.default_rng(42), N=50000,
20 features, target = 0.5 X[0] + 0.3 X[1] + 0.1 noise, ST/Cosine/RMSE,
depth=6, bins=128, l2=3.0, lr=0.03, random_strength=0).

Outputs (data/):
  mlx_borders.tsv        csv_train.cpp's borders (BORDER\\t<feat>\\t<value>)
  cpu_borders.tsv        copy of probe-a-borders/data/upfront_quantization_borders.tsv
  border_diff.csv        per-(feat, idx) absolute & ULP differences
  border_count.csv       per-feature counts side-by-side
  summary.txt            top-level findings
"""

import csv
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BIN = REPO_ROOT / "csv_train_probe_c"
PROBE_A_BORDERS = REPO_ROOT / "docs/sprint33/probe-a-borders/data/upfront_quantization_borders.tsv"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Anchor — must match Probe-A exactly.
N = 50_000
NUM_FEATURES = 20
SEED = 42
DEPTH = 6
BINS = 128  # Probe-A uses 128; csv_train.cpp will cap internally per DEC-039.
LR = 0.03
LOSS = "rmse"
GROW = "SymmetricTree"
SCORE_FN = "Cosine"
ITERS = 2
L2 = 3.0
RS = 0.0
BOOTSTRAP = "No"


def make_data():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, NUM_FEATURES)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(N) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"f{i}" for i in range(X.shape[1])] + ["target"])
        for i in range(len(y)):
            w.writerow(list(map(float, X[i])) + [float(y[i])])


def run_mlx_borders(csv_path: Path) -> dict:
    """Run csv_train_probe_c, parse BORDER\\t<feat>\\t<value> lines."""
    cmd = [
        str(BIN), str(csv_path),
        "--iterations", str(ITERS),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--l2", str(L2),
        "--loss", LOSS,
        "--bins", str(BINS),
        "--seed", str(SEED),
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--bootstrap-type", BOOTSTRAP,
        "--random-strength", str(RS),
    ]
    print(f"  [mlx] cmd: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        print("  [mlx] STDERR:")
        print(res.stderr)
        raise SystemExit(f"csv_train_probe_c exited non-zero: {res.returncode}")

    by_feat: dict[int, list[float]] = {}
    n_lines = 0
    for line in res.stdout.splitlines():
        if not line.startswith("BORDER\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        feat = int(parts[1])
        val = float(parts[2])
        by_feat.setdefault(feat, []).append(val)
        n_lines += 1
    print(f"  [mlx] parsed {n_lines} BORDER lines across {len(by_feat)} features")
    return by_feat


def read_cpu_borders() -> dict:
    by_feat: dict[int, list[float]] = {}
    with open(PROBE_A_BORDERS) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            feat = int(parts[0])
            val = float(parts[1])
            by_feat.setdefault(feat, []).append(val)
    return by_feat


def fp32_ulp_diff(a: float, b: float) -> int:
    """Distance in float32 ULPs between two scalars (sign-aware monotone mapping)."""
    ai = int(np.array([a], dtype=np.float32).view(np.int32)[0])
    bi = int(np.array([b], dtype=np.float32).view(np.int32)[0])
    # Map to monotone unsigned ordering.
    if ai < 0:
        ai = 0x80000000 - ai
    if bi < 0:
        bi = 0x80000000 - bi
    return abs(ai - bi)


def main():
    if not BIN.exists():
        print(f"ERROR: {BIN} not found. Run scripts/build_dump_borders.sh first.", file=sys.stderr)
        sys.exit(2)
    if not PROBE_A_BORDERS.exists():
        print(f"ERROR: {PROBE_A_BORDERS} not found.", file=sys.stderr)
        sys.exit(2)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] generating anchor dataset...")
    X, y = make_data()
    print(f"  X shape={X.shape}  y mean={y.mean():.6f}  y std={y.std():.6f}")

    print("[2/4] writing CSV...")
    csv_path = DATA_DIR / "anchor.csv"
    write_csv(csv_path, X, y)
    print(f"  {csv_path}  size={csv_path.stat().st_size} bytes")

    print("[3/4] running csv_train_probe_c (DUMP_BORDERS)...")
    mlx = run_mlx_borders(csv_path)

    print("[4/4] reading CPU upfront borders + diffing...")
    cpu = read_cpu_borders()

    # Mirror MLX borders into our data dir for self-contained record.
    mlx_path = DATA_DIR / "mlx_borders.tsv"
    with open(mlx_path, "w") as f:
        for feat in sorted(mlx):
            for v in mlx[feat]:
                f.write(f"{feat}\t{v:.9g}\n")
    cpu_path = DATA_DIR / "cpu_borders.tsv"
    cpu_path.write_bytes(PROBE_A_BORDERS.read_bytes())

    # Per-feature counts.
    count_path = DATA_DIR / "border_count.csv"
    with open(count_path, "w") as f:
        f.write("feature_idx,mlx_count,cpu_count,delta\n")
        for feat in range(NUM_FEATURES):
            mc = len(mlx.get(feat, []))
            cc = len(cpu.get(feat, []))
            f.write(f"{feat},{mc},{cc},{mc-cc}\n")

    # Per-(feat, idx) value diff at the OVERLAP (min count across the two lists).
    # If counts differ, mismatch is also a structural observation.
    diff_path = DATA_DIR / "border_diff.csv"
    n_exact = 0
    n_total = 0
    n_within_ulp1 = 0
    max_abs_diff = 0.0
    max_ulp_diff = 0
    max_abs_at = None
    max_ulp_at = None
    with open(diff_path, "w") as f:
        f.write("feature_idx,idx,mlx_val,cpu_val,abs_diff,ulp_diff\n")
        for feat in range(NUM_FEATURES):
            mlist = mlx.get(feat, [])
            clist = cpu.get(feat, [])
            n = min(len(mlist), len(clist))
            for i in range(n):
                m = float(mlist[i])
                c = float(clist[i])
                ad = abs(m - c)
                ud = fp32_ulp_diff(m, c)
                f.write(f"{feat},{i},{m:.9g},{c:.9g},{ad:.6e},{ud}\n")
                n_total += 1
                if ad == 0.0:
                    n_exact += 1
                if ud <= 1:
                    n_within_ulp1 += 1
                if ad > max_abs_diff:
                    max_abs_diff = ad
                    max_abs_at = (feat, i, m, c)
                if ud > max_ulp_diff:
                    max_ulp_diff = ud
                    max_ulp_at = (feat, i, m, c)

    summary = []
    summary.append("=" * 78)
    summary.append("S33-PROBE-C Stage 1 — border-VALUE byte-match across MLX vs CatBoost CPU")
    summary.append("=" * 78)
    summary.append("")
    summary.append("Anchor: N=50000, ST/Cosine/RMSE, depth=6, bins=128, seed=42, l2=3, lr=0.03")
    summary.append("Sources:")
    summary.append("  MLX  = csv_train.cpp QuantizeFeatures (GreedyLogSum, cap 127 per DEC-039)")
    summary.append("  CPU  = CatBoost Pool.quantize(border_count=128) [Probe-A artifact]")
    summary.append("")
    summary.append("Per-feature counts:")
    summary.append("  feat   mlx   cpu  Δ")
    sum_mlx = 0
    sum_cpu = 0
    for feat in range(NUM_FEATURES):
        mc = len(mlx.get(feat, []))
        cc = len(cpu.get(feat, []))
        sum_mlx += mc
        sum_cpu += cc
        marker = "" if mc == cc else "  ← MISMATCH"
        summary.append(f"    {feat:>2}   {mc:>3}   {cc:>3}  {mc-cc:+}{marker}")
    summary.append(f"  total  {sum_mlx:>3}   {sum_cpu:>3}  {sum_mlx-sum_cpu:+}")
    summary.append("")
    summary.append("Per-(feature, idx) value comparison (overlap range):")
    summary.append(f"  pairs compared : {n_total}")
    summary.append(f"  byte-identical : {n_exact}  ({100*n_exact/max(n_total,1):.2f}%)")
    summary.append(f"  ≤ 1 ulp        : {n_within_ulp1}  ({100*n_within_ulp1/max(n_total,1):.2f}%)")
    summary.append(f"  max abs diff   : {max_abs_diff:.6e}  at {max_abs_at}")
    summary.append(f"  max ulp diff   : {max_ulp_diff}  at {max_ulp_at}")
    summary.append("")
    if max_abs_diff == 0.0 and sum_mlx == sum_cpu:
        summary.append("VERDICT: borders byte-identical across all overlap pairs.")
        summary.append("  → Stage 1 PASS — border value mismatch is not the iter≥2 mechanism.")
        summary.append("  → Proceed to Stage 2 (per-bin gain dump at iter=2 depth=0).")
    elif max_abs_diff == 0.0 and sum_mlx != sum_cpu:
        summary.append("VERDICT: per-pair values byte-identical BUT counts differ.")
        summary.append("  → Investigate the count delta — last-border truncation?")
    else:
        summary.append("VERDICT: borders DIVERGE.")
        summary.append(f"  → max abs={max_abs_diff:.6e}, max ulp={max_ulp_diff}.")
        summary.append("  → Stage 1 LOCATES iter≥2 mechanism: divergent quantization grid.")
        summary.append("  → Open: which side's borders are 'right' (in CatBoost-parity sense)?")

    txt = "\n".join(summary) + "\n"
    (DATA_DIR / "summary.txt").write_text(txt)
    print()
    print(txt)


if __name__ == "__main__":
    main()
