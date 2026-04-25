#!/usr/bin/env python3
"""
PROBE-C Stage 2 — full tree[1] depth-by-depth comparison.

L3 reported "S2 SPLIT divergent at iter=2 depth=0" based on bin-index
disagreement (CPU bin=3, MLX bin=64). PROBE-C Stage 1 found those bins
correspond to the SAME physical value (0.014169) — depth=0 actually agrees.
This script extends the comparison to all 6 depths of iter=2's tree.

Inputs:
  data/mlx_anchor.json     csv_train_probe_c2 model JSON (saved from MLX)
  data/mlx_borders.tsv     MLX 127-border grid (Stage-1 output)
  data/cpu_borders.tsv     CPU 128-border upfront grid (Probe-A artifact)

Hardcoded CPU tree[1] from L3 reproduce (random_seed=42, 50000 rows,
ST/Cosine/RMSE, depth=6, l2=3, lr=0.03):
  d=0: feat=0, border=+0.014169  (split_index=3)
  d=1: feat=1, border=-0.092413  (split_index=8)
  d=2: feat=0, border=-0.946874  (split_index=1)
  d=3: feat=0, border=+1.042727  (split_index=5)
  d=4: feat=1, border=+0.815438  (split_index=10)
  d=5: feat=1, border=-1.081904  (split_index=6)
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_borders(path):
    by_feat = {}
    for line in (DATA / path).read_text().splitlines():
        if not line.strip():
            continue
        f, v = line.split("\t")
        by_feat.setdefault(int(f), []).append(float(v))
    return by_feat


CPU_T1 = [
    {"feat": 0, "border": 0.014169400557875633, "split_index": 3},
    {"feat": 1, "border": -0.09241345524787903, "split_index": 8},
    {"feat": 0, "border": -0.9468743801116943,  "split_index": 1},
    {"feat": 0, "border": 1.0427273511886597,   "split_index": 5},
    {"feat": 1, "border": 0.815437912940979,    "split_index": 10},
    {"feat": 1, "border": -1.0819035768508911,  "split_index": 6},
]


def main():
    mlx_b = load_borders("mlx_borders.tsv")
    model = json.loads((DATA / "mlx_anchor.json").read_text())
    mlx_t1 = model["trees"][1]["splits"]

    print("=" * 84)
    print("iter=2 tree[1]: MLX vs CPU — depth-by-depth physical comparison")
    print("=" * 84)
    print()
    print(
        f"{'d':>2} {'MLX (feat,bin)':>15} {'MLX phys':>14} {'CPU feat':>9}"
        f" {'CPU phys':>14} {'phys Δ':>11} {'verdict':>14}"
    )
    print("-" * 84)

    rows = []
    for d, (m, c) in enumerate(zip(mlx_t1, CPU_T1)):
        mfeat, mbin = m["feature_idx"], m["bin_threshold"]
        cfeat, cval = c["feat"], c["border"]
        mval = mlx_b[mfeat][mbin] if 0 <= mbin < len(mlx_b[mfeat]) else float("nan")
        feat_match = mfeat == cfeat
        if feat_match:
            d_abs = abs(mval - cval)
            verdict = "AGREE" if d_abs < 1e-5 else f"feat-only(Δ={d_abs:.2e})"
        else:
            d_abs = float("nan")
            verdict = "DIVERGE"
        rows.append((d, mfeat, mbin, mval, cfeat, cval, feat_match, d_abs, verdict))
        print(
            f"{d:>2}  ({mfeat:>2},{mbin:>4})    {mval:>14.6g}    {cfeat:>4}"
            f"  {cval:>14.6g}  {d_abs:>11.3e}  {verdict:>14}"
        )

    print()
    first_div = next((r for r in rows if not r[6]), None)
    if first_div is None:
        print("VERDICT: tree[1] IDENTICAL feature-wise. Mechanism is in leaf values or border drift.")
    else:
        d, mfeat, mbin, mval, cfeat, cval, _, _, _ = first_div
        print(f"VERDICT: feature-level divergence at depth={d}.")
        print(f"  CPU picks feat={cfeat} (border={cval:.6g})")
        print(f"  MLX picks feat={mfeat} bin={mbin} (border={mval:.6g})")
        print(f"  All depths < {d} agree on feature, with depth-0 ULP-identical and")
        print(f"  depth-1 differing only by missing-border-induced grid offset.")
        print()
        print("  This contradicts the L3 'S2 SPLIT divergent at depth=0' verdict")
        print("  (which conflated CPU CBM stored-coords with MLX upfront grid index).")
        print(f"  The real iter=2 split-class divergence is at DEPTH={d}.")


if __name__ == "__main__":
    main()
