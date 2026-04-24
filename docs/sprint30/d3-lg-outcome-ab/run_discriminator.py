#!/usr/bin/env python3
"""S30-D3-LG-OUTCOME-AB — DEC-034 outcome A vs B discriminator (task #102).

At iter=1 for each (max_leaves, seed) pair in
max_leaves ∈ {8, 16, 31, 64} × seeds ∈ {0, 1, 2}, run MLX (csv_train_t3,
LG+Cosine) and CPU (catboost pip, LG+Cosine) and extract three observables
per DEC-034 mechanism claim:

  Obs-1  Tree structure identity:
         Does MLX's BFS feature sequence equal CPU's BFS feature sequence?
         Divergence starting at some max_leaves threshold => outcome B.
  Obs-2  MLX split-gain summary at iter-1:
         max split_gain, mean split_gain, and the gain of the last-popped
         split.  Gain scale is a proxy for residual magnitude growth with
         max_leaves (the per-gain fp64 residual is 3.81e-6 post-K4 per T2).
  Obs-3  Priority-queue flip rate:
         For each position k in MLX's pop order, does the (feature, quantized
         threshold) tuple equal the CPU split at the corresponding BFS node?
         Rate grows with max_leaves => outcome B.

Config (task spec): N=1000, depth=7, features=20, bins=128, iterations=1,
loss=RMSE, grow_policy=Lossguide, score_function=Cosine, rs=0.0, bootstrap=no.

Notes
-----
* csv_train_t3 (built with -DCOSINE_T3_MEASURE) bypasses the LG+Cosine
  guard on the MLX side.  CPU CatBoost does not guard LG+Cosine — no bypass
  needed on that side.
* Output JSONs live under data/ as mlx_ml{L}_s{S}.json and cpu_ml{L}_s{S}.json.
  A summary row per cell goes into data/summary.csv.
* Verdict authority lives in verdict.md; this script produces data only.
"""

from __future__ import annotations

import csv as csv_mod
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
)
T3_BINARY = REPO_ROOT / "csv_train_t3"
DATA_DIR = REPO_ROOT / "docs" / "sprint30" / "d3-lg-outcome-ab" / "data"

# Task-spec cell
N = 1000
FEATURES = 20
DEPTH = 7
BINS = 128
LR = 0.03
LOSS = "rmse"
GROW = "Lossguide"
SCORE_FN = "Cosine"
ITERS = 1

MAX_LEAVES_VALUES = [8, 16, 31, 64]
SEEDS = [0, 1, 2]

# S29 control cell (depth=3, max_leaves=8 on the SAME 20-feature data shape).
# S29's baseline used 10 features and bit-identical BFS was reported.  Here we
# redo the control on the 20-feature shape used by T3 and this discriminator,
# so "bit-identical BFS at shallow cell" is confirmed *on the same data*.
S29_CONTROL_N = 1000
S29_CONTROL_FEATURES = 20  # task spec data shape; not the 10-feature S29 data
S29_CONTROL_DEPTH = 3
S29_CONTROL_MAX_LEAVES = 8


def make_data(n: int, seed: int):
    """Canonical S26/T3 data: 20 features, signal in f0+f1, 10% noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATURES)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(
        np.float32
    )
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    header = [f"f{i}" for i in range(X.shape[1])] + ["target"]
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(header)
        for i in range(len(y)):
            w.writerow(list(X[i]) + [y[i]])


# ---------------------------------------------------------------------------
# MLX runner — csv_train_t3 → JSON
# ---------------------------------------------------------------------------


def run_mlx(data_path: Path, seed: int, max_leaves: int, out_json: Path) -> float:
    cmd = [
        str(T3_BINARY),
        str(data_path),
        "--iterations", str(ITERS),
        "--depth", str(DEPTH),
        "--max-leaves", str(max_leaves),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--loss", LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed", str(seed),
        "--output", str(out_json),
    ]
    env = os.environ.copy()
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"csv_train_t3 failed (ml={max_leaves}, seed={seed}):\n"
            f"stderr={result.stderr[:500]}\nstdout={result.stdout[:500]}"
        )
    # parse final RMSE from stdout
    rmse = None
    for line in result.stdout.split("\n"):
        if "loss=" in line and "iter=" in line:
            for tok in line.split():
                if tok.startswith("loss="):
                    try:
                        rmse = float(tok.split("=", 1)[1])
                    except ValueError:
                        pass
    if rmse is None:
        raise RuntimeError(f"could not parse MLX RMSE for ml={max_leaves}, seed={seed}")
    _ = elapsed  # kept for future perf use
    return rmse


# ---------------------------------------------------------------------------
# CPU runner — catboost pip
# ---------------------------------------------------------------------------


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int, max_leaves: int, out_json: Path) -> float:
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        max_leaves=max_leaves,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy=GROW,
        score_function=SCORE_FN,
        max_bin=BINS,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="No",
        l2_leaf_reg=3.0,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    m.save_model(str(out_json), format="json")
    return float(m.evals_result_["learn"]["RMSE"][-1])


# ---------------------------------------------------------------------------
# Tree parsing and BFS sequence extraction
# ---------------------------------------------------------------------------


def mlx_splits_by_bfs(mlx_json_path: Path) -> list[dict]:
    """Return MLX splits sorted by bfs_node_index (level-order). Each dict
    carries feature_idx, bin_threshold, bfs_node_index, gain, pop_order."""
    with open(mlx_json_path) as f:
        d = json.load(f)
    tree = d["trees"][0]
    splits = tree.get("splits") or []
    gains = tree.get("split_gains") or []
    # MLX's splits array is ordered by POP order — index 0 = root (first pop).
    # Each split carries bfs_node_index; we attach both the pop order and the
    # gain at pop time.
    enriched = []
    for pop_idx, s in enumerate(splits):
        enriched.append({
            "feature_idx": s["feature_idx"],
            "bin_threshold": s["bin_threshold"],
            "bfs_node_index": s["bfs_node_index"],
            "gain": float(gains[pop_idx]) if pop_idx < len(gains) else None,
            "pop_order": pop_idx,
        })
    return sorted(enriched, key=lambda x: x["bfs_node_index"])


def mlx_pop_order(mlx_json_path: Path) -> list[dict]:
    """Return MLX splits in pop order (raw splits array order)."""
    with open(mlx_json_path) as f:
        d = json.load(f)
    tree = d["trees"][0]
    splits = tree.get("splits") or []
    gains = tree.get("split_gains") or []
    return [
        {
            "feature_idx": s["feature_idx"],
            "bin_threshold": s["bin_threshold"],
            "bfs_node_index": s["bfs_node_index"],
            "gain": float(gains[i]) if i < len(gains) else None,
            "pop_order": i,
        }
        for i, s in enumerate(splits)
    ]


def cpu_splits_by_bfs(cpu_json_path: Path) -> list[dict]:
    """Walk CPU nested tree in BFS order; return split records in BFS node
    order, assigning sequential bfs_node_index starting at 0 for root.

    CPU tree JSON: {left, right, split: {float_feature_index, border, ...}}
    Leaves are {value, weight} — no 'split' key.
    """
    with open(cpu_json_path) as f:
        d = json.load(f)
    trees = d.get("oblivious_trees") or d.get("trees") or []
    if not trees:
        return []
    root = trees[0]
    out = []
    # BFS traversal — each visited internal node gets sequential bfs_node_index
    frontier = [(root, 0)]
    nxt_bfs_idx = 0  # we also keep a running counter in case nodes aren't dense
    while frontier:
        new_frontier = []
        for node, bfs_idx in frontier:
            if not isinstance(node, dict):
                continue
            sp = node.get("split")
            if not isinstance(sp, dict):
                continue  # leaf
            out.append({
                "feature_idx": sp.get("float_feature_index"),
                "border": sp.get("border"),
                "bfs_node_index": bfs_idx,
            })
            # left-child bfs_idx = 2*bfs_idx + 1, right = 2*bfs_idx + 2
            # (standard complete-binary-tree indexing).  This is used only to
            # *attempt* to map CPU splits back to MLX bfs_node_index indices.
            new_frontier.append((node.get("left"), 2 * bfs_idx + 1))
            new_frontier.append((node.get("right"), 2 * bfs_idx + 2))
            nxt_bfs_idx += 1
        frontier = new_frontier
    return out


def cpu_bfs_feature_sequence(cpu_json_path: Path) -> list[int]:
    """Pure BFS level-order feature sequence on CPU tree.  Matches the S29
    spike method: walk left-before-right, layer by layer, collecting
    float_feature_index of each internal node."""
    with open(cpu_json_path) as f:
        d = json.load(f)
    trees = d.get("oblivious_trees") or d.get("trees") or []
    if not trees:
        return []
    root = trees[0]
    seq = []
    frontier = [root]
    while frontier:
        nxt = []
        for n in frontier:
            if not isinstance(n, dict):
                continue
            sp = n.get("split")
            if isinstance(sp, dict):
                seq.append(sp.get("float_feature_index"))
                if "left" in n:
                    nxt.append(n["left"])
                if "right" in n:
                    nxt.append(n["right"])
        frontier = nxt
    return seq


def mlx_bfs_feature_sequence(mlx_json_path: Path) -> list[int]:
    """BFS level-order MLX feature sequence: sort MLX splits by bfs_node_index,
    then walk in bfs_node_index order.  Matches S29 method."""
    by_bfs = mlx_splits_by_bfs(mlx_json_path)
    # Produce feature sequence in bfs_node_index order.  S29 method simply
    # reads feature_idx off each split sorted by bfs_node_index.
    return [s["feature_idx"] for s in by_bfs]


# ---------------------------------------------------------------------------
# Discriminator core — per-cell observables
# ---------------------------------------------------------------------------


def measure_cell(max_leaves: int, seed: int, tmpdir: Path) -> dict:
    """Run one (max_leaves, seed) cell and return observables."""
    X, y = make_data(N, seed)
    data_csv = tmpdir / f"data_ml{max_leaves}_s{seed}.csv"
    write_csv(data_csv, X, y)

    mlx_json = DATA_DIR / f"mlx_ml{max_leaves}_s{seed}.json"
    cpu_json = DATA_DIR / f"cpu_ml{max_leaves}_s{seed}.json"

    mlx_rmse = run_mlx(data_csv, seed, max_leaves, mlx_json)
    cpu_rmse = run_cpu(X, y, seed, max_leaves, cpu_json)

    mlx_bfs = mlx_bfs_feature_sequence(mlx_json)
    cpu_bfs = cpu_bfs_feature_sequence(cpu_json)
    mlx_pop = mlx_pop_order(mlx_json)
    cpu_by_bfs = cpu_splits_by_bfs(cpu_json)

    # Observable 1: tree-structure identity
    root_feature_match = bool(
        mlx_bfs and cpu_bfs and mlx_bfs[0] == cpu_bfs[0]
    )
    # bfs_feature_sequence_match — exact sequence equality
    bfs_seq_match = mlx_bfs == cpu_bfs
    # first divergent BFS position
    first_div = None
    for i in range(min(len(mlx_bfs), len(cpu_bfs))):
        if mlx_bfs[i] != cpu_bfs[i]:
            first_div = i
            break
    if first_div is None and len(mlx_bfs) != len(cpu_bfs):
        first_div = min(len(mlx_bfs), len(cpu_bfs))
    num_internal_splits = len(mlx_bfs)
    num_feature_matches = sum(
        1 for a, b in zip(mlx_bfs, cpu_bfs) if a == b
    )

    # Observable 2: MLX gain summary
    gains = [s["gain"] for s in mlx_pop if s["gain"] is not None]
    gain_max = float(max(gains)) if gains else None
    gain_min = float(min(gains)) if gains else None
    gain_mean = float(sum(gains) / len(gains)) if gains else None
    # Gain of last-popped split is the lowest-gain one (queue exhausts best first)
    last_pop_gain = float(mlx_pop[-1]["gain"]) if mlx_pop else None

    # Observable 3: priority-queue flip rate.
    # Compare MLX splits-by-bfs to CPU splits-by-bfs at each matched bfs_node_index.
    # A "flip" is a position where both sides have an internal split at that bfs
    # index but disagree on (feature).  Border comparison is skipped (different
    # representations: MLX bin_threshold vs CPU float border).
    mlx_by_bfs = mlx_splits_by_bfs(mlx_json)
    mlx_map = {s["bfs_node_index"]: s for s in mlx_by_bfs}
    cpu_map = {s["bfs_node_index"]: s for s in cpu_by_bfs}
    common_idx = set(mlx_map.keys()) & set(cpu_map.keys())
    flips = 0
    mismatched_detail = []
    for idx in sorted(common_idx):
        if mlx_map[idx]["feature_idx"] != cpu_map[idx]["feature_idx"]:
            flips += 1
            mismatched_detail.append({
                "bfs_node_index": idx,
                "mlx_feature": mlx_map[idx]["feature_idx"],
                "cpu_feature": cpu_map[idx]["feature_idx"],
            })
    # Also count splits that exist in one tree but not the other (structural asymmetry)
    mlx_only = [idx for idx in mlx_map if idx not in cpu_map]
    cpu_only = [idx for idx in cpu_map if idx not in mlx_map]
    flip_rate = flips / max(1, len(common_idx))
    # Structural divergence rate: fraction of BFS-indices where MLX and CPU
    # don't agree on presence-or-feature
    total_positions = len(set(mlx_map.keys()) | set(cpu_map.keys()))
    structural_divergence = (
        flips + len(mlx_only) + len(cpu_only)
    ) / max(1, total_positions)

    drift_pct = 100.0 * abs(mlx_rmse - cpu_rmse) / cpu_rmse

    return {
        "max_leaves": max_leaves,
        "seed": seed,
        "num_internal_splits_mlx": len(mlx_bfs),
        "num_internal_splits_cpu": len(cpu_bfs),
        "mlx_bfs_feature_seq": mlx_bfs,
        "cpu_bfs_feature_seq": cpu_bfs,
        "root_feature_match": root_feature_match,
        "bfs_seq_match": bfs_seq_match,
        "first_divergent_bfs_pos": first_div,
        "num_feature_matches_in_zip": num_feature_matches,
        "gain_max": gain_max,
        "gain_min": gain_min,
        "gain_mean": gain_mean,
        "last_pop_gain": last_pop_gain,
        "queue_flips": flips,
        "mlx_only_bfs": len(mlx_only),
        "cpu_only_bfs": len(cpu_only),
        "common_bfs_positions": len(common_idx),
        "flip_rate": flip_rate,
        "structural_divergence_rate": structural_divergence,
        "mismatched_detail": mismatched_detail,
        "mlx_rmse": mlx_rmse,
        "cpu_rmse": cpu_rmse,
        "drift_pct": drift_pct,
    }


def main() -> int:
    if not T3_BINARY.exists():
        print(f"ERROR: {T3_BINARY} not found", file=sys.stderr)
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("S30-D3-LG-OUTCOME-AB — DEC-034 outcome A vs B discriminator")
    print(f"  Cell: N={N}, features={FEATURES}, depth={DEPTH}, bins={BINS}")
    print(f"  max_leaves ∈ {MAX_LEAVES_VALUES}, seeds ∈ {SEEDS}")
    print(f"  iters=1, loss=RMSE, grow=Lossguide, score=Cosine")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        rows = []
        for ml in MAX_LEAVES_VALUES:
            print(f"\n--- max_leaves={ml} ---")
            for seed in SEEDS:
                rec = measure_cell(ml, seed, tmpdir)
                rows.append(rec)
                print(
                    f"  seed={seed}  "
                    f"mlx_splits={rec['num_internal_splits_mlx']}/"
                    f"cpu_splits={rec['num_internal_splits_cpu']}  "
                    f"bfs_match={rec['bfs_seq_match']}  "
                    f"first_div={rec['first_divergent_bfs_pos']}  "
                    f"flips={rec['queue_flips']}/{rec['common_bfs_positions']} "
                    f"(rate={rec['flip_rate']:.3f}, structdiv={rec['structural_divergence_rate']:.3f})  "
                    f"gain_max={rec['gain_max']:.4g}  "
                    f"drift={rec['drift_pct']:.4f}%"
                )

    # Write raw detail
    full_path = DATA_DIR / "per_cell_detail.json"
    with open(full_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nDetail dumped: {full_path}")

    # Write summary CSV
    csv_path = DATA_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow([
            "max_leaves", "seed",
            "num_splits_mlx", "num_splits_cpu",
            "bfs_seq_match", "first_divergent_bfs_pos",
            "queue_flips", "common_bfs_positions", "flip_rate",
            "mlx_only_bfs", "cpu_only_bfs", "structural_divergence_rate",
            "gain_max", "gain_min", "gain_mean", "last_pop_gain",
            "mlx_rmse", "cpu_rmse", "drift_pct",
        ])
        for r in rows:
            w.writerow([
                r["max_leaves"], r["seed"],
                r["num_internal_splits_mlx"], r["num_internal_splits_cpu"],
                int(r["bfs_seq_match"]), r["first_divergent_bfs_pos"],
                r["queue_flips"], r["common_bfs_positions"],
                round(r["flip_rate"], 4),
                r["mlx_only_bfs"], r["cpu_only_bfs"],
                round(r["structural_divergence_rate"], 4),
                f"{r['gain_max']:.8g}" if r["gain_max"] is not None else "",
                f"{r['gain_min']:.8g}" if r["gain_min"] is not None else "",
                f"{r['gain_mean']:.8g}" if r["gain_mean"] is not None else "",
                f"{r['last_pop_gain']:.8g}" if r["last_pop_gain"] is not None else "",
                f"{r['mlx_rmse']:.8f}", f"{r['cpu_rmse']:.8f}",
                f"{r['drift_pct']:.4f}",
            ])
    print(f"Summary CSV: {csv_path}")

    # Per-max_leaves aggregates for verdict
    print("\n" + "=" * 72)
    print("PER-MAX_LEAVES AGGREGATES")
    print("=" * 72)
    print(f"{'max_leaves':>10} {'bfs_match (%)':>14} {'flip_rate':>10} {'struct_div':>11} "
          f"{'gain_max':>12} {'drift (%)':>10}")
    for ml in MAX_LEAVES_VALUES:
        cells = [r for r in rows if r["max_leaves"] == ml]
        n = len(cells)
        bfs_match_pct = 100.0 * sum(1 for r in cells if r["bfs_seq_match"]) / n
        mean_flip = sum(r["flip_rate"] for r in cells) / n
        mean_structdiv = sum(r["structural_divergence_rate"] for r in cells) / n
        mean_gain_max = sum(r["gain_max"] for r in cells if r["gain_max"] is not None) / n
        mean_drift = sum(r["drift_pct"] for r in cells) / n
        print(f"{ml:>10} {bfs_match_pct:>13.1f}% {mean_flip:>10.4f} {mean_structdiv:>11.4f} "
              f"{mean_gain_max:>12.4g} {mean_drift:>10.4f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
