#!/usr/bin/env python3
"""
S32-T2-INSTRUMENT: CPU-side per-bin term dump at depth=0 seed=42.

Produces a CSV with the same schema as the MLX term dump:
  feat, bin, sumLeft, sumRight, weightLeft, weightRight, lambda,
  cosNum_term, cosDen_term, gain

Method:
  1. Build the same canonical N=50000 dataset (seed=42) as csv_train.cpp.
  2. Quantize with CatBoost GreedyLogSum, border_count=128 — same borders as
     post-DEC-037 MLX (maxBordersCount = maxBins = 128).
  3. Compute gradients: RMSE -> grad = approx - target at approx = mean(y).
     (CatBoost RMSE gradient convention: g = approx - target, h = 1).
  4. At depth=0 (single partition, all docs), for each (feat, bin) candidate:
       gR = sum of grad for docs with quantized_bin > bin  (suffix sum)
       gL = totalG - gR
       wR = count of docs with quantized_bin > bin
       wL = totalW - wR
       cosNum = gL^2/(wL+lambda) + gR^2/(wR+lambda)
       cosDen = gL^2*wL/(wL+lambda)^2 + gR^2*wR/(wR+lambda)^2  + 1e-20
       gain   = cosNum / sqrt(cosDen)
  5. Write to data/cpu_terms_seed42_depth0.csv

Note on suffix-sum convention:
  CPU CatBoost uses upper_bound convention: bin_value = #borders strictly less
  than feature_value. A split at bin threshold b means: go-right if
  bin_value > b (equivalently: feature_value > borders[b]).
  So sumRight = sum over docs where bin_value > b = suffix_sum starting at b+1.
  This is the same as what MLX's suffGrad[base + b] computes:
    suffGrad[b] = sum over hist[b..folds-1] where hist[k] = stats for bin_value = k+1
    (1-indexed: hist[k] covers docs with bin_value in {k+1}).
  After T1-CODEPATH verification, both sides use the same effective split set.

Output: data/cpu_terms_seed42_depth0.csv
"""

import csv
import math
import subprocess
import sys
import tempfile
import json
from pathlib import Path

import numpy as np

try:
    import catboost
    from catboost import Pool
except ImportError:
    print("ERROR: catboost not installed. pip install catboost", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR  = REPO_ROOT / "docs" / "sprint32" / "t2-terms" / "data"

# Anchor parameters — must match csv_train.cpp exactly
ANCHOR_N      = 50_000
ANCHOR_SEED   = 42
ANCHOR_BINS   = 128
ANCHOR_L2     = 3.0
ANCHOR_LR     = 0.03
ANCHOR_LOSS   = "RMSE"
ANCHOR_GROW   = "SymmetricTree"
ANCHOR_SCORE  = "Cosine"
ANCHOR_RS     = 0
ANCHOR_DEPTH  = 6


def make_data(n: int, seed: int):
    """Canonical S26 data: 20 features, signal in f0/f1, 10% noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def get_borders_and_binned(X: np.ndarray, bins: int, seed: int) -> tuple[dict, np.ndarray]:
    """
    Run CatBoost GreedyLogSum quantization and return the full border grid
    and the quantized bin values (0-indexed: bin_value = #borders < value).
    """
    script = f"""
import numpy as np, catboost, json, sys, tempfile, os
rng = np.random.default_rng({seed})
N = {len(X)}
X = rng.standard_normal((N, 20)).astype('float32')
y = (X[:,0]*0.5 + X[:,1]*0.3 + rng.standard_normal(N)*0.1).astype('float32')
pool = catboost.Pool(X, y)
pool.quantize(border_count={bins}, feature_border_type='GreedyLogSum')
td = tempfile.mkdtemp()
bfile = os.path.join(td, 'borders.txt')
pool.save_quantization_borders(bfile)
borders_raw = {{}}
with open(bfile) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            fi = int(parts[0]); val = float(parts[1])
            borders_raw.setdefault(fi, []).append(val)
print("BORDERS_JSON:" + json.dumps({{str(k): v for k, v in borders_raw.items()}}))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    full_borders: dict[int, list[float]] = {}
    for line in result.stdout.splitlines():
        if line.startswith("BORDERS_JSON:"):
            raw = json.loads(line[len("BORDERS_JSON:"):])
            full_borders = {int(k): v for k, v in raw.items()}

    N_docs  = X.shape[0]
    N_feats = X.shape[1]
    binned  = np.zeros((N_feats, N_docs), dtype=np.int32)
    for f in range(N_feats):
        b_list = full_borders.get(f, [])
        if b_list:
            b_arr = np.array(b_list, dtype=np.float64)
            # upper_bound convention: bin_value = number of borders strictly < value
            binned[f] = np.searchsorted(
                b_arr, X[:, f].astype(np.float64), side='right'
            ).astype(np.int32)
    return full_borders, binned


def compute_cpu_terms(
    X: np.ndarray,
    y: np.ndarray,
    full_borders: dict,
    binned: np.ndarray,
    l2: float,
    seed: int,
    out_path: Path,
) -> list[dict]:
    """
    Compute per-bin Cosine term dump at depth=0.

    Returns list of dicts with keys:
      feat, bin, sumLeft, sumRight, weightLeft, weightRight, lambda,
      cosNum_term, cosDen_term, gain
    """
    N_docs  = X.shape[0]
    N_feats = X.shape[1]

    # RMSE gradients: CatBoost convention g = approx - target at approx = mean(y)
    # NOTE: csv_train.cpp also uses approx - target (initial approx = mean(y))
    base_pred = float(np.mean(y))
    grads = (base_pred - y).astype(np.float64)  # approx - target
    hess  = np.ones(N_docs, dtype=np.float64)

    # At depth=0: single partition covering all docs
    total_g = float(grads.sum())
    total_w = float(hess.sum())   # = N_docs

    # scaledL2 = L2RegLambda * sumAllWeights / docCount  (P5 fix from S31-T2)
    scaled_l2 = l2 * total_w / float(N_docs)

    print(f"  totalG={total_g:.6f}  totalW={total_w:.0f}  "
          f"l2={l2}  scaledL2={scaled_l2:.6f}")
    print(f"  basePred={base_pred:.6f}")

    records = []

    for f in range(N_feats):
        b_list = full_borders.get(f, [])
        M = len(b_list)
        if M == 0:
            continue

        bins_f = np.clip(binned[f], 0, M).astype(np.int32)

        # Build histogram: hist_g[b] = sum of grad for docs with bin_value == b
        hist_g = np.zeros(M + 1, dtype=np.float64)
        hist_h = np.zeros(M + 1, dtype=np.float64)
        np.add.at(hist_g, bins_f, grads)
        np.add.at(hist_h, bins_f, hess)

        # Suffix sums: suffix_g[b] = sum(hist_g[b+1 .. M]) = sumRight for split at b
        # This is gR = sum of grads for docs with bin_value > b
        # (i.e., docs on the right side of split at bin threshold b)
        suffix_g = np.zeros(M + 1, dtype=np.float64)
        suffix_h = np.zeros(M + 1, dtype=np.float64)
        for b in range(M - 1, -1, -1):
            suffix_g[b] = suffix_g[b + 1] + hist_g[b + 1]
            suffix_h[b] = suffix_h[b + 1] + hist_h[b + 1]

        for b in range(M):
            gR = float(suffix_g[b])
            wR = float(suffix_h[b])
            gL = total_g - gR
            wL = total_w - wR

            if wL < 1e-15 or wR < 1e-15:
                continue

            inv_l = 1.0 / (wL + scaled_l2)
            inv_r = 1.0 / (wR + scaled_l2)

            cos_num = gL * gL * inv_l + gR * gR * inv_r
            cos_den = gL * gL * wL * inv_l * inv_l + gR * gR * wR * inv_r * inv_r + 1e-20
            gain    = cos_num / math.sqrt(cos_den)

            records.append({
                "feat":         f,
                "bin":          b,
                "sumLeft":      gL,
                "sumRight":     gR,
                "weightLeft":   wL,
                "weightRight":  wR,
                "lambda":       scaled_l2,
                "cosNum_term":  cos_num,
                "cosDen_term":  cos_den,
                "gain":         gain,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["feat", "bin", "sumLeft", "sumRight", "weightLeft", "weightRight",
                  "lambda", "cosNum_term", "cosDen_term", "gain"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"  Written: {out_path} ({len(records)} rows)")

    # Quick sanity: top-3 by gain
    top3 = sorted(records, key=lambda r: r["gain"], reverse=True)[:3]
    print("  Top-3 candidates by CPU gain:")
    for r in top3:
        print(f"    feat={r['feat']} bin={r['bin']} "
              f"wL={r['weightLeft']:.1f} wR={r['weightRight']:.1f} "
              f"gL={r['sumLeft']:.6f} gR={r['sumRight']:.6f} "
              f"gain={r['gain']:.8f}")
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description="S32-T2 CPU term dump")
    parser.add_argument("--seed", type=int, default=ANCHOR_SEED)
    parser.add_argument("--n", type=int, default=ANCHOR_N)
    parser.add_argument("--bins", type=int, default=ANCHOR_BINS)
    parser.add_argument("--l2", type=float, default=ANCHOR_L2)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    print(f"S32-T2 CPU term dump")
    print(f"  Anchor: N={args.n}, depth=0, bins={args.bins}, RMSE, ST, Cosine, rs=0")
    print(f"  Seed: {args.seed}  L2={args.l2}")

    X, y = make_data(args.n, args.seed)
    print(f"  Data shape: {X.shape}")

    print(f"  Quantizing with GreedyLogSum border_count={args.bins} ...")
    full_borders, binned = get_borders_and_binned(X, args.bins, args.seed)
    print(f"  Border grid: {len(full_borders)} features")
    for f in range(min(3, len(full_borders))):
        print(f"    f{f}: {len(full_borders.get(f, []))} borders")

    out_path = args.out_dir / f"cpu_terms_seed{args.seed}_depth0.csv"
    compute_cpu_terms(X, y, full_borders, binned, args.l2, args.seed, out_path)

    print(f"\nCPU term dump complete: {out_path}")


if __name__ == "__main__":
    main()
