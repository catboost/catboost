#!/usr/bin/env python3
"""
S31-T3b-T1-AUDIT — CPU-side split dump harness.

For each depth layer of iteration 1, dumps from CPU CatBoost:
  - Per-leaf parent aggregates: (leaf_idx, sumG, sumH, W, leaf_count)
  - Top-K=5 split candidates: (feat_idx, bin_idx, gain) sorted by gain desc
  - Winning split: (feat_idx, bin_idx, gain, bin_value)

Method:
  1. Pre-quantize the Pool with border_count=128, feature_border_type=GreedyLogSum.
     Save the full quantization grid with pool.save_quantization_borders().
  2. Train CatBoost on the pre-quantized pool with logging_level='Debug'.
     The Debug log line format is:
       "<feat_idx>, bin=<bin_idx_0based> score <score>"
     where bin_idx is 0-indexed into the full quantization grid (verified empirically).
  3. Re-implement Cosine gain from raw partition stats and the full border grid
     to compute ALL candidate gains and rank top-K=5 per layer.
  4. Replay the SymmetricTree split sequence layer-by-layer to advance
     partition assignments for per-layer parent stats.

Anchor cell: N=50000, depth=6, bins=128, RMSE, SymmetricTree, Cosine,
             rs=0, seeds 42/43/44.

Output: data/cpu_splits_seed<seed>.json
Schema: {"seed": int, "N": int, "layers": [
  {
    "depth": int,
    "partitions": [{"leaf_idx": int, "sumG": float, "sumH": float,
                    "W": float, "leaf_count": int}],
    "top5": [{"feat_idx": int, "bin_idx": int, "gain": float}],
    "winner": {"feat_idx": int, "bin_idx": int, "gain": float, "bin_value": float}
  }
]}

Key bin_idx convention: 0-indexed into the full quantization border grid.
  split at bin_idx b means: go-right if quantized_bin_value > b
  (CatBoost uses upper_bound convention: quantized_bin = #borders < value)
"""

import json
import math
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import numpy as np

try:
    import catboost
    from catboost import CatBoostRegressor, Pool
except ImportError:
    print("ERROR: catboost not installed. pip install catboost==1.2.10", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Anchor cell parameters
# ---------------------------------------------------------------------------
ANCHOR_N = 50_000
ANCHOR_DEPTH = 6
ANCHOR_BINS = 128
ANCHOR_LR = 0.03
ANCHOR_LOSS = "RMSE"
ANCHOR_GROW = "SymmetricTree"
ANCHOR_SCORE_FN = "Cosine"
ANCHOR_RS = 0
ANCHOR_L2 = 3.0
ANCHOR_SEEDS = [42, 43, 44]
TOP_K = 5

DEFAULT_OUT_DIR = Path(__file__).parent / "data"


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Canonical S26/S28 anchor data. Must match csv_train.cpp make_data() exactly.
    20 float features, signal in f0+f1, 10% noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def cosine_gain_fp64(
    sum_g_parts: list[tuple[float, float]],
    sum_w_parts: list[tuple[float, float]],
    l2: float,
) -> float:
    """
    Cosine gain (double precision) across all partitions.
    T1-PRE §2 verified formula — algebraically equivalent to CPU CatBoost:
      num = Σ_p [ gL²/(wL+λ) + gR²/(wR+λ) ]
      den = Σ_p [ gL²*wL/(wL+λ)² + gR²*wR/(wR+λ)² ] + 1e-20
      gain = num / sqrt(den)
    """
    cos_num = 0.0
    cos_den = 1e-20
    for (gL, gR), (wL, wR) in zip(sum_g_parts, sum_w_parts):
        if wL < 1e-15 or wR < 1e-15:
            continue
        inv_l = 1.0 / (wL + l2)
        inv_r = 1.0 / (wR + l2)
        cos_num += gL * gL * inv_l + gR * gR * inv_r
        cos_den += gL * gL * wL * inv_l * inv_l + gR * gR * wR * inv_r * inv_r
    return cos_num / math.sqrt(cos_den)


def parse_debug_winner(line: str) -> dict | None:
    """
    Parse Debug log line: "<feat_idx>, bin=<int> score <float>"
    Returns {"feat_idx": int, "bin_idx": int, "score": float} or None.
    bin_idx is 0-indexed into the full quantization grid.
    """
    m = re.match(r"^\s*(\d+),\s*bin=(\d+)\s+score\s+([\d.e+\-]+)\s*$", line)
    if m:
        return {
            "feat_idx": int(m.group(1)),
            "bin_idx": int(m.group(2)),  # 0-indexed into full quant grid
            "score": float(m.group(3)),
        }
    return None


def run_catboost_subprocess(
    n: int,
    seed: int,
    depth: int,
    bins: int,
    l2: float,
) -> tuple[list[dict], dict[int, list[float]]]:
    """
    Run CatBoost in a subprocess with Debug logging.
    Returns (winners, full_borders_grid):
      winners: list of {"feat_idx", "bin_idx", "score"} per depth layer
      full_borders_grid: {feat_idx: [sorted float borders]} (128 entries/feature)
    """
    script = f"""
import numpy as np, catboost, json, sys, tempfile, os

rng = np.random.default_rng({seed})
N = {n}
X = rng.standard_normal((N, 20)).astype('float32')
y = (X[:,0]*0.5 + X[:,1]*0.3 + rng.standard_normal(N)*0.1).astype('float32')

pool = catboost.Pool(X, y)
# Pre-quantize to get the full border grid
pool.quantize(border_count={bins}, feature_border_type='GreedyLogSum')

# Save full quantization borders
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

cb = catboost.CatBoostRegressor(
    iterations=1,
    depth={depth},
    learning_rate={ANCHOR_LR},
    loss_function='{ANCHOR_LOSS}',
    grow_policy='{ANCHOR_GROW}',
    score_function='{ANCHOR_SCORE_FN}',
    random_seed={seed},
    random_strength={ANCHOR_RS},
    l2_leaf_reg={l2},
    bootstrap_type='No',  # MLX uses no bootstrapping; match exactly
    logging_level='Debug',
)
cb.fit(pool)  # train on pre-quantized pool — same borders used internally
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )

    stdout = result.stdout
    if result.returncode != 0:
        print(f"  subprocess stderr: {result.stderr[:300]}", file=sys.stderr)

    winners = []
    full_borders: dict[int, list[float]] = {}
    for line in stdout.splitlines():
        w = parse_debug_winner(line)
        if w is not None:
            winners.append(w)
        elif line.startswith("BORDERS_JSON:"):
            raw = json.loads(line[len("BORDERS_JSON:"):])
            full_borders = {int(k): v for k, v in raw.items()}

    return winners, full_borders


def compute_layers(
    X: np.ndarray,
    y: np.ndarray,
    winners: list[dict],
    full_borders: dict[int, list[float]],
    l2: float = ANCHOR_L2,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Replay SymmetricTree split sequence to compute per-layer stats.
    Uses the full quantization grid (all 128 borders per feature).
    """
    N_docs = len(y)
    N_feats = X.shape[1]

    # Quantize X using the full borders from CatBoost's GreedyLogSum
    # bin[f][d] = number of borders in full_borders[f] that are < X[d,f]
    # (i.e. upper_bound convention, 0-indexed)
    binned = np.zeros((N_feats, N_docs), dtype=np.int32)
    for f in range(N_feats):
        b_list = full_borders.get(f, [])
        if b_list:
            b_arr = np.array(b_list, dtype=np.float64)
            # 'right' side: finds first position where b_arr[pos] >= X → #values < X
            # This matches CatBoost's upper_bound(borders, value) which gives
            # number of borders strictly less than value
            binned[f] = np.searchsorted(
                b_arr, X[:, f].astype(np.float64), side='right'
            ).astype(np.int32)

    # RMSE gradients: CPU convention = target - approx (at approx = mean(y))
    base_pred = float(np.mean(y))
    grads = y.astype(np.float64) - base_pred
    hess = np.ones(N_docs, dtype=np.float64)

    partitions = np.zeros(N_docs, dtype=np.int32)
    layers = []

    for depth_idx, winner in enumerate(winners):
        num_parts = 1 << depth_idx

        # ---- Parent aggregates ----
        part_sums_g = np.zeros(num_parts, dtype=np.float64)
        part_sums_h = np.zeros(num_parts, dtype=np.float64)
        part_counts = np.zeros(num_parts, dtype=np.int64)
        for d in range(N_docs):
            p = partitions[d]
            part_sums_g[p] += grads[d]
            part_sums_h[p] += hess[d]
            part_counts[p] += 1

        part_stats = [
            {
                "leaf_idx": int(p),
                "sumG": float(part_sums_g[p]),
                "sumH": float(part_sums_h[p]),
                "W": float(part_sums_h[p]),
                "leaf_count": int(part_counts[p]),
            }
            for p in range(num_parts)
        ]

        # ---- Candidate gains: all (feat, bin) pairs ----
        all_candidates = []

        for f in range(N_feats):
            b_list = full_borders.get(f, [])
            M = len(b_list)
            if M == 0:
                continue

            # Per-partition histogram over bins 0..M (M+1 bin values)
            # Efficiently using numpy bincount
            hist_g = np.zeros((num_parts, M + 1), dtype=np.float64)
            hist_h = np.zeros((num_parts, M + 1), dtype=np.float64)
            bins_f = np.clip(binned[f], 0, M).astype(np.int32)

            for p in range(num_parts):
                mask_p = partitions == p
                if not np.any(mask_p):
                    continue
                bp = bins_f[mask_p]
                gp = grads[mask_p]
                hp = hess[mask_p]
                # Accumulate into histogram bins
                np.add.at(hist_g[p], bp, gp)
                np.add.at(hist_h[p], bp, hp)

            # Suffix sums: suffix[p,b] = sum(hist[p, b+1..M]) = sumRight for split at b
            # Vectorized: suffix[:, b] = sum(hist[:, b+1..M]) for each b
            suffix_g = np.zeros((num_parts, M + 1), dtype=np.float64)
            suffix_h = np.zeros((num_parts, M + 1), dtype=np.float64)
            for b in range(M - 1, -1, -1):
                suffix_g[:, b] = suffix_g[:, b + 1] + hist_g[:, b + 1]
                suffix_h[:, b] = suffix_h[:, b + 1] + hist_h[:, b + 1]

            for b in range(M):
                # sumRight[p] = suffix_g[p, b], sumLeft[p] = part_total[p] - sumRight[p]
                gR = suffix_g[:, b]  # shape [num_parts]
                wR = suffix_h[:, b]
                gL = part_sums_g - gR
                wL = part_sums_h - wR

                # Build per-partition pairs
                sum_g_pairs = list(zip(gL.tolist(), gR.tolist()))
                sum_w_pairs = list(zip(wL.tolist(), wR.tolist()))

                gain = cosine_gain_fp64(sum_g_pairs, sum_w_pairs, l2)
                all_candidates.append((f, b, gain))

        all_candidates.sort(key=lambda x: x[2], reverse=True)
        top5 = [
            {"feat_idx": int(f), "bin_idx": int(b), "gain": float(g)}
            for f, b, g in all_candidates[:top_k]
        ]

        # ---- Winner ----
        w_feat = winner["feat_idx"]
        w_bin = winner["bin_idx"]
        w_borders = full_borders.get(w_feat, [])
        w_bval = w_borders[w_bin] if 0 <= w_bin < len(w_borders) else float("nan")

        layer_winner = {
            "feat_idx": w_feat,
            "bin_idx": w_bin,
            "gain": winner["score"],
            "bin_value": w_bval,
        }

        layers.append({
            "depth": depth_idx,
            "partitions": part_stats,
            "top5": top5,
            "winner": layer_winner,
        })

        # ---- Advance partitions: go-right if binned[w_feat][d] > w_bin ----
        w_bins_f = binned[w_feat]
        for d in range(N_docs):
            if w_bins_f[d] > w_bin:
                partitions[d] |= (1 << depth_idx)

    return layers


def run_seed(seed: int, n: int, out_dir: Path, depth: int, bins: int) -> bool:
    print(f"\n=== CPU dump: seed={seed} N={n} ===")
    X, y = make_data(n, seed)

    print(f"  Training CatBoost (logging_level=Debug) ...")
    winners, full_borders = run_catboost_subprocess(n, seed, depth, bins, ANCHOR_L2)

    print(f"  Parsed {len(winners)} winner splits from Debug log")
    if len(winners) < depth:
        print(f"  WARNING: expected {depth} depth layers, got {len(winners)}", file=sys.stderr)

    for d, w in enumerate(winners):
        b_list = full_borders.get(w["feat_idx"], [])
        bval = b_list[w["bin_idx"]] if 0 <= w["bin_idx"] < len(b_list) else float("nan")
        print(f"    depth={d}: feat={w['feat_idx']} bin_idx={w['bin_idx']} "
              f"bin_value={bval:.6f} score={w['score']:.8f}")

    print(f"  Full border grid: {len(full_borders)} features with borders")
    print(f"    feature 0: {len(full_borders.get(0,[]))} borders")
    print(f"    feature 1: {len(full_borders.get(1,[]))} borders")

    print(f"  Computing top-K candidates per layer ...")
    layers = compute_layers(X, y, winners, full_borders, top_k=TOP_K)

    out_path = out_dir / f"cpu_splits_seed{seed}.json"
    with open(out_path, "w") as fh:
        json.dump({"seed": seed, "N": n, "layers": layers}, fh, indent=2)
    print(f"  Written: {out_path}")

    print(f"  Summary:")
    for layer in layers:
        d = layer["depth"]
        w = layer["winner"]
        t5 = layer["top5"]
        top3_str = ", ".join(f"{c['gain']:.6f}" for c in t5[:3])
        print(f"    depth={d}: winner=(feat={w['feat_idx']}, bin={w['bin_idx']}, "
              f"gain={w['gain']:.8f})  top3=[{top3_str}...]")

    return True


def main(seeds: list[int], n: int, out_dir: Path, depth: int, bins: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"S31-T3b CPU dump harness")
    print(f"  Anchor: N={n}, depth={depth}, bins={bins}, RMSE, ST, Cosine, rs=0")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {out_dir}")

    for seed in seeds:
        run_seed(seed, n, out_dir, depth, bins)

    print(f"\nAll seeds complete. CPU dumps in {out_dir}/cpu_splits_seed*.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S31-T3b CPU split dump")
    parser.add_argument("--seeds", nargs="+", type=int, default=ANCHOR_SEEDS)
    parser.add_argument("--n", type=int, default=ANCHOR_N)
    parser.add_argument("--depth", type=int, default=ANCHOR_DEPTH)
    parser.add_argument("--bins", type=int, default=ANCHOR_BINS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.n != ANCHOR_N:
        print(f"NOTE: N={args.n} overrides anchor N={ANCHOR_N}. "
              f"G1 verdict must cite N={ANCHOR_N} measurements.")
    main(args.seeds, args.n, args.out_dir, args.depth, args.bins)
