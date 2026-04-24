#!/usr/bin/env python3
"""
S32-T2-INSTRUMENT: align and compare MLX vs CPU per-bin term data at depth=0.

Input:
  data/mlx_terms_seed42_depth0.csv
  data/cpu_terms_seed42_depth0.csv

Both have schema:
  feat, bin, sumLeft, sumRight, weightLeft, weightRight, lambda,
  cosNum_term, cosDen_term, gain

Alignment: join on (feat, bin). At depth=0 both sides produce the same
candidate set (post-DEC-037 border alignment).

For each aligned (feat, bin), compute relative differences:
  rdiff_X = |X_mlx - X_cpu| / (|X_cpu| + 1e-30)

Report:
  1. Max rdiff for each column (gL, gR, wL, wR, lambda, cosNum, cosDen, gain).
  2. First diverging column (first column where max rdiff > 1e-6).
  3. Per-bin detail for (feat=0, bin=59..67) — the T3b winner neighbourhood.
  4. Top-10 rows by |gain_mlx - gain_cpu| / gain_cpu.

Usage:
  python docs/sprint32/t2-terms/compare_terms.py [--data-dir PATH]
"""

import argparse
import csv
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR  = REPO_ROOT / "docs" / "sprint32" / "t2-terms" / "data"

COLS_TO_COMPARE = [
    ("sumLeft",     "gL"),
    ("sumRight",    "gR"),
    ("weightLeft",  "wL"),
    ("weightRight", "wR"),
    ("lambda",      "lambda"),
    ("cosNum_term", "cosNum"),
    ("cosDen_term", "cosDen"),
    ("gain",        "gain"),
]
DIVERGENCE_THRESHOLD = 1e-6


def load_csv(path: Path) -> dict[tuple[int, int], dict]:
    rows = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = (int(row["feat"]), int(row["bin"]))
            rows[key] = {k: float(v) for k, v in row.items() if k not in ("feat", "bin")}
            rows[key]["feat"] = int(row["feat"])
            rows[key]["bin"]  = int(row["bin"])
    return rows


def rdiff(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-30)


def main():
    parser = argparse.ArgumentParser(description="S32-T2 term comparison")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mlx_path = args.data_dir / f"mlx_terms_seed{args.seed}_depth0.csv"
    cpu_path = args.data_dir / f"cpu_terms_seed{args.seed}_depth0.csv"

    for p in (mlx_path, cpu_path):
        if not p.exists():
            print(f"ERROR: {p} not found.", flush=True)
            import sys; import sys; sys.exit(1)

    mlx = load_csv(mlx_path)
    cpu = load_csv(cpu_path)

    keys_mlx = set(mlx.keys())
    keys_cpu = set(cpu.keys())
    common   = keys_mlx & keys_cpu
    only_mlx = keys_mlx - keys_cpu
    only_cpu = keys_cpu - keys_mlx

    print(f"S32-T2 Term Comparison — seed={args.seed} depth=0")
    print(f"  MLX candidates:   {len(mlx)}")
    print(f"  CPU candidates:   {len(cpu)}")
    print(f"  Aligned (common): {len(common)}")
    print(f"  Only in MLX:      {len(only_mlx)}")
    print(f"  Only in CPU:      {len(only_cpu)}")
    print()

    if len(common) == 0:
        print("ERROR: no common (feat,bin) pairs — cannot compare.")
        import sys; sys.exit(1)

    # Compute max rdiff per column
    col_max_rdiff = {col: 0.0 for col, _ in COLS_TO_COMPARE}
    col_max_key   = {col: None for col, _ in COLS_TO_COMPARE}

    for key in common:
        m = mlx[key]
        c = cpu[key]
        for col, _ in COLS_TO_COMPARE:
            r = rdiff(m[col], c[col])
            if r > col_max_rdiff[col]:
                col_max_rdiff[col] = r
                col_max_key[col]   = key

    print("Column max rdiff summary (descending):")
    sorted_cols = sorted(COLS_TO_COMPARE, key=lambda x: col_max_rdiff[x[0]], reverse=True)
    first_diverging = None
    for col, label in COLS_TO_COMPARE:
        flag = ""
        if col_max_rdiff[col] > DIVERGENCE_THRESHOLD:
            flag = "  <-- DIVERGES"
            if first_diverging is None:
                # first_diverging in COLS_TO_COMPARE order (not sorted by rdiff)
                pass
        print(f"  {label:12s}: max_rdiff = {col_max_rdiff[col]:.3e}  "
              f"at (feat={col_max_key[col][0]}, bin={col_max_key[col][1]}){flag}")

    # Identify first diverging column in causal order
    for col, label in COLS_TO_COMPARE:
        if col_max_rdiff[col] > DIVERGENCE_THRESHOLD:
            first_diverging = (col, label)
            break

    print()
    if first_diverging:
        col, label = first_diverging
        key = col_max_key[col]
        m = mlx[key]
        c = cpu[key]
        print(f"FIRST DIVERGING COLUMN: {label} (rdiff={col_max_rdiff[col]:.4e})")
        print(f"  at (feat={key[0]}, bin={key[1]}):")
        print(f"    MLX {label}: {m[col]:.15g}")
        print(f"    CPU {label}: {c[col]:.15g}")
        print(f"    ratio MLX/CPU: {m[col]/c[col]:.8f}" if c[col] != 0 else "    ratio: undef (CPU=0)")
    else:
        print("NO DIVERGENCE FOUND across all columns (all rdiff <= 1e-6).")
        print("This would be unexpected given T3b G1 PASS finding.")

    # Per-bin detail around f0 winner neighbourhood (bins 59-67 from T3b data)
    print()
    print("Per-bin detail for feat=0, bins 59..67 (T3b winner neighbourhood):")
    print(f"  {'bin':>4}  {'gL_MLX':>14} {'gL_CPU':>14} {'wL_MLX':>10} {'wL_CPU':>10}"
          f"  {'gain_MLX':>14} {'gain_CPU':>14}  {'g_rdiff':>9} {'gain_rdiff':>10}")
    for b in range(59, 68):
        key = (0, b)
        if key not in mlx and key not in cpu:
            print(f"  {b:>4}  (not in either)")
            continue
        m = mlx.get(key)
        c = cpu.get(key)
        if m is None:
            print(f"  {b:>4}  (only in CPU)")
            continue
        if c is None:
            print(f"  {b:>4}  (only in MLX)")
            continue
        gL_rd   = rdiff(m["sumLeft"],  c["sumLeft"])
        gain_rd = rdiff(m["gain"],     c["gain"])
        print(f"  {b:>4}  {m['sumLeft']:>14.8f} {c['sumLeft']:>14.8f} "
              f"{m['weightLeft']:>10.1f} {c['weightLeft']:>10.1f}  "
              f"{m['gain']:>14.8f} {c['gain']:>14.8f}  "
              f"{gL_rd:>9.3e} {gain_rd:>10.3e}")

    # Top-10 by gain rdiff
    print()
    print("Top-10 rows by gain rdiff:")
    gain_rdiffs = []
    for key in common:
        m = mlx[key]
        c = cpu[key]
        r = rdiff(m["gain"], c["gain"])
        gain_rdiffs.append((r, key, m["gain"], c["gain"],
                            m["sumLeft"], c["sumLeft"],
                            m["weightLeft"], c["weightLeft"]))
    gain_rdiffs.sort(reverse=True)
    print(f"  {'feat':>4} {'bin':>4}  {'gain_MLX':>14} {'gain_CPU':>14}  "
          f"{'gain_rdiff':>10}  {'gL_rdiff':>9}  {'wL_rdiff':>9}")
    for r, key, gm, gc, gLm, gLc, wLm, wLc in gain_rdiffs[:10]:
        gL_rd = rdiff(gLm, gLc)
        wL_rd = rdiff(wLm, wLc)
        print(f"  {key[0]:>4} {key[1]:>4}  {gm:>14.8f} {gc:>14.8f}  "
              f"{r:>10.3e}  {gL_rd:>9.3e}  {wL_rd:>9.3e}")

    # Gain ratio summary
    print()
    gain_ratios = []
    for key in common:
        gc = cpu[key]["gain"]
        if abs(gc) > 1e-10:
            gain_ratios.append(mlx[key]["gain"] / gc)
    if gain_ratios:
        median_ratio = sorted(gain_ratios)[len(gain_ratios) // 2]
        mean_ratio   = sum(gain_ratios) / len(gain_ratios)
        print(f"Gain ratio summary (MLX/CPU) across {len(gain_ratios)} aligned candidates:")
        print(f"  Median: {median_ratio:.6f}   Mean: {mean_ratio:.6f}")
        print(f"  (T3b reported ~0.946 — this should confirm or refine that estimate)")

    # Decision point
    print()
    print("VERDICT:")
    if first_diverging:
        col, label = first_diverging
        print(f"  First diverging quantity: {label}  (rdiff={col_max_rdiff[col]:.4e})")
        if col in ("sumLeft", "sumRight"):
            print("  => BUG LAYER: GRADIENT (gL or gR differs)")
            print("     Check: histogram kernel, gradient initialization, partition stats.")
        elif col in ("weightLeft", "weightRight"):
            print("  => BUG LAYER: WEIGHT (wL or wR differs)")
            print("     Possible: P11 hessian-vs-sampleWeight, partition stats error.")
        elif col == "lambda":
            print("  => BUG LAYER: LAMBDA (scaled L2 regularizer differs)")
            print("     Check: scaledL2RegLambda computation vs CPU's scaledL2Regularizer.")
        elif col in ("cosNum_term", "cosDen_term"):
            print("  => BUG LAYER: FORMULA (numerator or denominator term computation)")
            print("     gL/gR/wL/wR are identical but cosNum or cosDen diverges.")
            print("     This contradicts T1-CODEPATH (algebraic identity) — investigate!")
        elif col == "gain":
            print("  => BUG LAYER: GAIN FINALIZATION (cosNum/sqrt(cosDen) step)")
            print("     cosNum and cosDen are identical but gain diverges.")
    else:
        print("  No divergence detected. All per-bin terms match to < 1e-6.")
        print("  If T3b gain ratio was 0.946, the divergence may be at depth>0")
        print("  or in a comparison artifact. Check multi-partition composition (K6).")


if __name__ == "__main__":
    main()
