#!/usr/bin/env python3
"""
S31-T3b-T1-AUDIT: compare CPU vs MLX per-layer split decisions.

Reads:
  docs/sprint31/t3b-audit/data/cpu_splits_seed<N>.json
  docs/sprint31/t3b-audit/data/mlx_splits_seed<N>.json

For each seed, for each depth layer:
  1. Compare winner (feat_idx, bin_idx) — MATCH or DIVERGE
  2. If winner diverges, classify mechanism per DEC-036 table
  3. Compare partition aggregate stats (sumG, sumH, W)
  4. Compare top5 gain rankings

Reports the first diverging layer with mechanism class and file:line pointers.

DEC-036 Mechanism Classes:
  Layer 0   feat/bin differ          → Layer-0 formula issue (formula itself)
  Layer L≥1, same feat diff bin      → Enumeration divergence (bin enumeration order)
  Layer L≥1, diff feat               → Tie-break divergence (gain ranking / noise)
  Layer L≥1, same (feat,bin) diff    → Normalization divergence (gain formula)
  No divergence at any layer         → K1: expand to iter=2

Gate G1: divergence localized with mechanism class named.

Usage:
  python docs/sprint31/t3b-audit/compare_splits.py [--seed SEED] [--verbose]
"""

import argparse
import json
import math
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Gain comparison tolerance (relative)
# Two gain values are considered matching if |a - b| / max(|a|, |b|, 1e-10) < GAIN_RTOL
GAIN_RTOL = 1e-4  # 0.01% relative tolerance — same formula should be within FP rounding


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_layers(cpu_layers: list, mlx_layers: list, seed: int, verbose: bool) -> dict:
    """
    Compare CPU and MLX layer records.

    Returns a dict with keys:
      first_diverging_depth: int or None
      mechanism: str or None
      details: str
      per_layer: list of per-layer comparison dicts
    """
    result = {
        "seed": seed,
        "first_diverging_depth": None,
        "mechanism": None,
        "details": "",
        "per_layer": [],
    }

    n_cpu = len(cpu_layers)
    n_mlx = len(mlx_layers)
    n_layers = min(n_cpu, n_mlx)

    if n_cpu != n_mlx:
        print(f"  WARNING: CPU has {n_cpu} layers, MLX has {n_mlx} layers — comparing first {n_layers}")

    for li in range(n_layers):
        cpu_lay = cpu_layers[li]
        mlx_lay = mlx_layers[li]

        depth = cpu_lay["depth"]
        assert depth == mlx_lay["depth"], (
            f"depth mismatch at position {li}: CPU={depth}, MLX={mlx_lay['depth']}"
        )

        cpu_w = cpu_lay["winner"]
        mlx_w = mlx_lay["winner"]

        cpu_feat = cpu_w["feat_idx"]
        mlx_feat = mlx_w["feat_idx"]
        cpu_bin  = cpu_w["bin_idx"]
        mlx_bin  = mlx_w["bin_idx"]
        cpu_gain = cpu_w["gain"]
        mlx_gain = mlx_w["gain"]

        # Gain relative difference
        max_gain = max(abs(cpu_gain), abs(mlx_gain), 1e-10)
        gain_rdiff = abs(cpu_gain - mlx_gain) / max_gain

        # Partition stats comparison
        cpu_parts = {p["leaf_idx"]: p for p in cpu_lay["partitions"]}
        mlx_parts = {p["leaf_idx"]: p for p in mlx_lay["partitions"]}
        part_keys = set(cpu_parts.keys()) & set(mlx_parts.keys())
        max_sumG_err = 0.0
        max_sumH_err = 0.0
        for pk in part_keys:
            cp = cpu_parts[pk]
            mp = mlx_parts[pk]
            # Only compare non-zero partitions
            if cp["leaf_count"] == 0:
                continue
            # Guard: skip sumG comparison when both are near-zero (RMSE basepred ≈ mean(y))
            # because relative error is meaningless for near-zero absolute values.
            sumG_abs_scale = max(abs(cp["sumG"]), abs(mp["sumG"]))
            if sumG_abs_scale > 1e-3:
                sumG_err = abs(cp["sumG"] - mp["sumG"]) / sumG_abs_scale
                max_sumG_err = max(max_sumG_err, sumG_err)
            sumH_err = abs(cp["sumH"] - mp["sumH"]) / max(abs(cp["sumH"]), 1e-10)
            max_sumH_err = max(max_sumH_err, sumH_err)

        winner_match = (cpu_feat == mlx_feat) and (cpu_bin == mlx_bin)

        layer_record = {
            "depth": depth,
            "cpu_winner": {"feat": cpu_feat, "bin": cpu_bin, "gain": cpu_gain},
            "mlx_winner": {"feat": mlx_feat, "bin": mlx_bin, "gain": mlx_gain},
            "winner_match": winner_match,
            "gain_rdiff": gain_rdiff,
            "max_sumG_relerr": max_sumG_err,
            "max_sumH_relerr": max_sumH_err,
        }
        result["per_layer"].append(layer_record)

        status = "MATCH" if winner_match else "DIVERGE"
        gain_str = f"gain_rdiff={gain_rdiff:.2e}"
        stats_str = f"sumG_relerr={max_sumG_err:.2e} sumH_relerr={max_sumH_err:.2e}"

        if verbose or not winner_match:
            print(f"  depth={depth}: [{status}]  "
                  f"CPU=f{cpu_feat}/b{cpu_bin}(g={cpu_gain:.6f})  "
                  f"MLX=f{mlx_feat}/b{mlx_bin}(g={mlx_gain:.6f})  "
                  f"{gain_str}  {stats_str}")

        if not winner_match and result["first_diverging_depth"] is None:
            result["first_diverging_depth"] = depth

            # Classify mechanism (DEC-036 table).
            # Post-DEC-037: border count is now 128 in both CPU and MLX.
            # Remaining divergences are attributed to gain formula, not border enumeration.
            if cpu_feat == mlx_feat and gain_rdiff < GAIN_RTOL:
                # Same feature, very similar gain but different bin — border placement tie.
                bin_diff = cpu_bin - mlx_bin
                mech = "BIN-ENUMERATION"
                detail = (
                    f"Same feature f{cpu_feat}, nearly identical gain (rdiff={gain_rdiff:.2e}): "
                    f"CPU=b{cpu_bin} vs MLX=b{mlx_bin} (offset={bin_diff:+d}). "
                    f"Border placement tie — two adjacent bins have near-equal gain; "
                    f"CPU and MLX greedy split order breaks the tie differently. "
                    f"File: catboost/mlx/tests/csv_train.cpp GreedyLogSumBestSplit."
                )
            elif cpu_feat == mlx_feat and gain_rdiff >= GAIN_RTOL:
                # Same feature, different bin, large gain difference — formula causes wrong argmax.
                bin_diff = cpu_bin - mlx_bin
                mech = "GAIN-FORMULA"
                detail = (
                    f"Same feature f{cpu_feat} but different bin: CPU=b{cpu_bin} vs MLX=b{mlx_bin} "
                    f"(offset={bin_diff:+d}). Gain rdiff={gain_rdiff:.2e} (>={GAIN_RTOL}) — "
                    f"Cosine score formula produces wrong gain magnitudes in MLX, shifting argmax. "
                    f"CPU gain for same bin=b{mlx_bin}: check cpu_top5. "
                    f"Root: Cosine accumulation (cosNum/cosDen) in FindBestSplit diverges from "
                    f"CosineScoreCalcer in catboost/private/libs/algo/score_calcers.cpp. "
                    f"File: catboost/mlx/tests/csv_train.cpp (FindBestSplit, S28-OBLIV-DISPATCH)."
                )
            elif depth == 0 and gain_rdiff >= GAIN_RTOL:
                mech = "LAYER-0-FORMULA"
                detail = (
                    f"First split diverges at depth=0: CPU=f{cpu_feat}/b{cpu_bin} vs "
                    f"MLX=f{mlx_feat}/b{mlx_bin}. Gain rdiff={gain_rdiff:.2e} — formula issue. "
                    f"Inspect csv_train.cpp FindBestSplit Cosine accumulation vs "
                    f"catboost/private/libs/algo/score_calcers.cpp."
                )
            elif gain_rdiff < GAIN_RTOL:
                mech = "GAIN-TIE-BREAK"
                detail = (
                    f"Different feature (CPU=f{cpu_feat}/b{cpu_bin}, MLX=f{mlx_feat}/b{mlx_bin}) "
                    f"but gain difference is small ({gain_rdiff:.2e} < {GAIN_RTOL}). "
                    f"Likely a tie-break / noise perturbation difference. "
                    f"Inspect random_strength / noise schedule in FindBestSplit."
                )
            else:
                mech = "GAIN-RANKING"
                detail = (
                    f"Different feature (CPU=f{cpu_feat}/b{cpu_bin} gain={cpu_gain:.6f}, "
                    f"MLX=f{mlx_feat}/b{mlx_bin} gain={mlx_gain:.6f}). "
                    f"Gain ranking divergence: gain_rdiff={gain_rdiff:.2e}. "
                    f"Inspect Cosine gain formula in csv_train.cpp FindBestSplit vs "
                    f"catboost/private/libs/algo/score_calcers.cpp CosineScoreCalcer."
                )

            result["mechanism"] = mech
            result["details"] = detail

            # Show top5 comparison at first divergence
            print(f"\n  *** FIRST DIVERGENCE at depth={depth} ***")
            print(f"  Mechanism: {mech}")
            print(f"  {detail}")

            print(f"\n  CPU top5 at depth={depth}:")
            for i, c in enumerate(cpu_lay.get("top5", [])[:5]):
                marker = " <-- winner" if (c["feat_idx"] == cpu_feat and c["bin_idx"] == cpu_bin) else ""
                print(f"    [{i}] f{c['feat_idx']}/b{c['bin_idx']} gain={c['gain']:.8f}{marker}")
            print(f"  CPU winner: f{cpu_feat}/b{cpu_bin} gain={cpu_gain:.8f}")

            print(f"\n  MLX top5 at depth={depth}:")
            for i, c in enumerate(mlx_lay.get("top5", [])[:5]):
                marker = " <-- winner" if (c["feat_idx"] == mlx_feat and c["bin_idx"] == mlx_bin) else ""
                print(f"    [{i}] f{c['feat_idx']}/b{c['bin_idx']} gain={c['gain']:.8f}{marker}")
            print(f"  MLX winner: f{mlx_feat}/b{mlx_bin} gain={mlx_gain:.8f}")

            # Partition stats at first divergence
            print(f"\n  Partition stats at depth={depth}:")
            print(f"    max_sumG_relerr={max_sumG_err:.2e}  max_sumH_relerr={max_sumH_err:.2e}")
            if max_sumH_err > 1e-6:
                print("    WARNING: sumH divergence > 1e-6 — histogram inputs are different!")
            elif max_sumG_err > 1e-6:
                print("    WARNING: sumG divergence > 1e-6 — gradient inputs are different!")
            print()

    return result


def classify_no_divergence(result: dict) -> str:
    """If no divergence found, return K1 (expand to iter=2) note."""
    n = len(result["per_layer"])
    if n == 0:
        return "NO LAYERS COMPARED — check build and run."
    matches = sum(1 for l in result["per_layer"] if l["winner_match"])
    return (
        f"All {n} layers MATCH. K1 fires: no divergence at iter=1. "
        f"Expand audit to iter=2 (see DEC-036 K1 clause)."
    )


def main():
    parser = argparse.ArgumentParser(description="S31-T3b split comparison")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: all available)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all layers, not just diverging ones")
    args = parser.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    else:
        # Auto-discover available seeds (need both cpu and mlx files)
        seeds = []
        for f in sorted(DATA_DIR.glob("cpu_splits_seed*.json")):
            seed = int(f.stem.replace("cpu_splits_seed", ""))
            mlx_path = DATA_DIR / f"mlx_splits_seed{seed}.json"
            if mlx_path.exists():
                seeds.append(seed)
        if not seeds:
            print("No paired cpu/mlx JSON files found in:", DATA_DIR)
            print("Run cpu_dump.py first, then run_mlx_audit.py")
            sys.exit(1)

    print(f"S31-T3b split comparison — seeds: {seeds}")
    print(f"Data dir: {DATA_DIR}\n")

    all_diverged = []
    no_diverge_seeds = []

    for seed in seeds:
        cpu_path = DATA_DIR / f"cpu_splits_seed{seed}.json"
        mlx_path = DATA_DIR / f"mlx_splits_seed{seed}.json"

        if not cpu_path.exists():
            print(f"MISSING: {cpu_path}")
            continue
        if not mlx_path.exists():
            print(f"MISSING: {mlx_path}")
            continue

        print(f"=== seed={seed} ===")
        cpu_data = load_json(cpu_path)
        mlx_data = load_json(mlx_path)

        cpu_n = cpu_data.get("N", "?")
        mlx_n = mlx_data.get("N", mlx_data.get("seed", "?"))
        print(f"  CPU: N={cpu_n}, {len(cpu_data['layers'])} layers")
        print(f"  MLX: seed={mlx_data['seed']}, {len(mlx_data['layers'])} layers")

        r = compare_layers(cpu_data["layers"], mlx_data["layers"], seed, args.verbose)

        if r["first_diverging_depth"] is not None:
            all_diverged.append(r)
        else:
            no_diverge_seeds.append(seed)
            print(f"  {classify_no_divergence(r)}\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_diverged:
        # Find the minimum diverging depth across seeds
        min_depth = min(r["first_diverging_depth"] for r in all_diverged)
        # Collect mechanisms at the min diverging depth
        mechs = [r["mechanism"] for r in all_diverged
                 if r["first_diverging_depth"] == min_depth]
        dominant_mech = max(set(mechs), key=mechs.count)

        print(f"First diverging layer: depth={min_depth}")
        print(f"Mechanism: {dominant_mech}")
        print(f"Diverging seeds: {[r['seed'] for r in all_diverged]}")
        if no_diverge_seeds:
            print(f"Non-diverging seeds: {no_diverge_seeds}")
        print()
        print("DEC-036 classification:")
        for r in all_diverged:
            print(f"  seed={r['seed']}: depth={r['first_diverging_depth']} [{r['mechanism']}]")
            print(f"    {r['details']}")
            print()

        print("Fix direction:")
        if dominant_mech == "GAIN-FORMULA":
            print("  Root cause: Cosine score formula in MLX FindBestSplit produces gains ~5%")
            print("  lower than CPU (94.6% of CPU value). This shifts argmax to wrong bin.")
            print("  Likely: cosNum or cosDen accumulation is different across partitions.")
            print("  Inspect: catboost/mlx/tests/csv_train.cpp (FindBestSplit, label S28-OBLIV-DISPATCH)")
            print("  vs:      catboost/private/libs/algo/score_calcers.cpp (CosineScoreCalcer)")
            print("  Key: CPU sums [gL^2/(wL+l) + gR^2/(wR+l)] over partitions;")
            print("       MLX must accumulate cosNum/cosDen identically.")
        elif dominant_mech == "LAYER-0-FORMULA":
            print("  Review Cosine gain formula in csv_train.cpp FindBestSplit")
            print("  vs catboost/private/libs/algo/score_calcers.cpp CosineScoreCalcer.")
            print("  File: catboost/mlx/tests/csv_train.cpp (FindBestSplit, ~line 1690)")
            print("  Ref:  catboost/private/libs/algo/score_calcers.cpp (CosineScoreCalcer)")
        elif dominant_mech == "BIN-ENUMERATION":
            print("  Border placement tie: adjacent bins have nearly equal gain.")
            print("  CPU and MLX break the tie differently due to greedy split order.")
            print("  File: catboost/mlx/tests/csv_train.cpp (GreedyLogSumBestSplit)")
            print("  Note: DEC-037 border count fix (maxBins) is already applied.")
        elif dominant_mech == "GAIN-TIE-BREAK":
            print("  Gain values are equal (tie); different feature chosen.")
            print("  Investigate random_strength / noise perturbation differences.")
            print("  File: catboost/mlx/tests/csv_train.cpp (noiseScale, DEC-028)")
            print("  Ref:  catboost/private/libs/algo/greedy_tensor_search.cpp")
        elif dominant_mech == "GAIN-RANKING":
            print("  Gain values differ significantly — scoring formula divergence.")
            print("  Review Cosine gain accumulation across partitions in FindBestSplit.")
            print("  File: catboost/mlx/tests/csv_train.cpp (FindBestSplit, S28-OBLIV-DISPATCH)")
            print("  Ref:  catboost/private/libs/algo/score_calcers.cpp CosineScoreCalcer")

        print()
        print("G1 PASS: mechanism class identified.")
    else:
        print("No divergence found at iter=1 across all tested seeds.")
        print(classify_no_divergence({"per_layer": sum((r["per_layer"] for r in []), [])}))
        print("K1 fires: expand to iter=2.")

    print("=" * 60)


if __name__ == "__main__":
    main()
