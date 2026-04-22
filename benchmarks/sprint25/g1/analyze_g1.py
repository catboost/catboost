#!/usr/bin/env python3
"""DEC-026-G1 ε-calibration analyzer.

Reads per-run Option A dump traces from the G1 sweep and computes:

Tail 1 — FLIP EVENTS
    For every (config_id, run_id, iter, depth_level) node, compare the rank-0
    winner (feat, bin) in the T1 trace vs. the Path 5 trace.  A disagreement is
    a "flip". To honour Ramos's earliest-node-per-iter rule, only the smallest
    depth_level in each (config, run, iter) with a disagreement is counted as
    a first-flip row; later depth_levels that disagree in the same iter are
    treated as downstream cascade and ignored.

    flip_gap_t1 is measured in T1's gain space: gain_T1(T1_winner) − gain_T1(
    Path5_winner).  If the Path 5 winner is not present in T1's top-5, we fall
    back to gain_T1(T1_winner) − gain_T1(T1.rank==4) (the worst top-k entry)
    and flag path5_winner_outside_t1_topk = true — this is a LOWER BOUND on
    the real flip gap in T1's gain space (the true gap is at least this big).

Tail 2 — LEGITIMATE-GAP FLOOR
    For every non-flipping node in T1's trace, compute
    legit_gap = gain(rank=0) − gain(rank=1).  A node is treated as non-flipping
    relative to the earliest-flip rule above: if the (config, run, iter) has
    a flip at some depth d*, nodes at depth_level < d* are non-flipping; the
    flip row itself and deeper cascade rows are excluded.  min_gap_per_config
    is the min across remaining rows per config_id.

ε THREADING
    epsilon_min  = max flip_gap_t1 over first-flip rows (must be < ε)
    epsilon_max  = min legit_gap over non-flipping rows  (must be > ε)
    safety_ratio = epsilon_max / epsilon_min

    PASS when safety_ratio >= 2.0. Recommended ε = sqrt(ε_min · ε_max) with
    tolerance [ε_min × 1.1, ε_max / 1.1].
    FALSIFIED otherwise — identify the (config, run, iter, depth_level) that
    sets ε_max (the killer).

Outputs:
    results/analysis/flip_events.csv
    results/analysis/legit_gap_floor.csv
    results/analysis/epsilon_threading.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict

G1_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(G1_DIR, "results")
TRACES_DIR = os.path.join(RESULTS_DIR, "traces")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

CONFIGS = list(range(1, 19))  # 1..18
RUNS = list(range(1, 6))      # 1..5


def load_trace(config_id: int, run_id: int, kernel: str) -> list[dict]:
    fname = f"c{config_id}_r{run_id}_{kernel}.csv"
    path = os.path.join(TRACES_DIR, fname)
    with open(path) as f:
        return list(csv.DictReader(f))


def index_by_node(rows: list[dict]) -> dict:
    """Return {(iter, depth_level): {rank: row}}.

    Handles the rare case where rank=255 shares (feat, bin) with a ranked row
    — we just overwrite, but rank=255 is only used for reference, not for
    flip detection.
    """
    out: dict = defaultdict(dict)
    for r in rows:
        key = (int(r["iter"]), int(r["depth_level"]))
        rank = int(r["rank"])
        out[key][rank] = {
            "feat": int(r["feat"]),
            "bin": int(r["bin"]),
            "gain": float(r["gain"]),
            "is_winner": int(r["is_winner"]),
        }
    return out


def analyze_config_run(
    config_id: int,
    run_id: int,
    flip_rows: list,
    legit_rows: list,
) -> None:
    t1 = index_by_node(load_trace(config_id, run_id, "t1"))
    t2 = index_by_node(load_trace(config_id, run_id, "t2_path5"))

    # Group nodes by iter to apply earliest-depth-per-iter rule.
    iters = sorted({k[0] for k in t1.keys()})

    for it in iters:
        # Find depth_levels with a winner disagreement, ordered ascending.
        depths = sorted({k[1] for k in t1.keys() if k[0] == it})
        first_flip_depth = None
        for d in depths:
            t1_node = t1.get((it, d), {})
            t2_node = t2.get((it, d), {})
            if 0 not in t1_node or 0 not in t2_node:
                # Malformed row — skip.
                continue
            t1_win = (t1_node[0]["feat"], t1_node[0]["bin"])
            t2_win = (t2_node[0]["feat"], t2_node[0]["bin"])
            if t1_win != t2_win:
                first_flip_depth = d
                break

        # Emit flip row for the earliest-flip depth (if any).
        if first_flip_depth is not None:
            d = first_flip_depth
            t1_node = t1[(it, d)]
            t2_node = t2[(it, d)]
            t1_win = t1_node[0]
            t2_win = t2_node[0]

            # Look up the Path-5 winner inside T1's top-5 (ranks 1..5).
            t2_in_t1 = None
            for r in range(1, 6):
                if r in t1_node and (
                    t1_node[r]["feat"] == t2_win["feat"]
                    and t1_node[r]["bin"] == t2_win["bin"]
                ):
                    t2_in_t1 = t1_node[r]
                    break

            outside = t2_in_t1 is None
            if outside:
                # Fallback: gap to T1.rank==4 (worst present top-5 entry).
                # This is a LOWER BOUND because the true Path-5 gain in T1
                # space is strictly below rank-4's gain.
                worst = None
                for r in (4, 3, 2, 1):
                    if r in t1_node:
                        worst = t1_node[r]
                        break
                if worst is None:
                    # No top-5 data at all — emit with NaN gap.
                    flip_gap = float("nan")
                else:
                    flip_gap = t1_win["gain"] - worst["gain"]
            else:
                flip_gap = t1_win["gain"] - t2_in_t1["gain"]

            flip_rows.append({
                "config_id": config_id,
                "run_id": run_id,
                "iter": it,
                "depth_level": d,
                "t1_winner_feat": t1_win["feat"],
                "t1_winner_bin": t1_win["bin"],
                "t1_winner_gain": t1_win["gain"],
                "t2_winner_feat": t2_win["feat"],
                "t2_winner_bin": t2_win["bin"],
                "t2_winner_gain": t2_win["gain"],
                "flip_gap_t1": flip_gap,
                "path5_winner_outside_t1_topk": outside,
            })

        # Emit legit-gap rows for depths BEFORE first_flip_depth (or all if
        # no flip). Rows at first_flip_depth itself and beyond are excluded.
        cap = first_flip_depth if first_flip_depth is not None else max(depths) + 1
        for d in depths:
            if d >= cap:
                break
            t1_node = t1.get((it, d), {})
            if 0 not in t1_node or 1 not in t1_node:
                continue
            gap = t1_node[0]["gain"] - t1_node[1]["gain"]
            legit_rows.append({
                "config_id": config_id,
                "run_id": run_id,
                "iter": it,
                "depth_level": d,
                "rank0_feat": t1_node[0]["feat"],
                "rank0_bin": t1_node[0]["bin"],
                "rank0_gain": t1_node[0]["gain"],
                "rank1_gain": t1_node[1]["gain"],
                "legit_gap": gap,
            })


def main() -> int:
    flip_rows: list = []
    legit_rows: list = []

    for cid in CONFIGS:
        for run in RUNS:
            analyze_config_run(cid, run, flip_rows, legit_rows)

    # ---- Write flip_events.csv ----
    flip_path = os.path.join(ANALYSIS_DIR, "flip_events.csv")
    with open(flip_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config_id", "run_id", "iter", "depth_level",
                "t1_winner_feat", "t1_winner_bin", "t1_winner_gain",
                "t2_winner_feat", "t2_winner_bin", "t2_winner_gain",
                "flip_gap_t1", "path5_winner_outside_t1_topk",
            ],
        )
        w.writeheader()
        for r in flip_rows:
            r = dict(r)
            r["path5_winner_outside_t1_topk"] = (
                "true" if r["path5_winner_outside_t1_topk"] else "false"
            )
            w.writerow(r)

    # ---- legit_gap_floor.csv (one row per config) ----
    # Aggregate min legit_gap per config_id. Configs where EVERY node flipped
    # would have no rows — emit NaN floor in that case.
    by_config: dict = defaultdict(list)
    for r in legit_rows:
        by_config[r["config_id"]].append(r)

    floor_path = os.path.join(ANALYSIS_DIR, "legit_gap_floor.csv")
    with open(floor_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config_id", "n_nonflip_nodes", "n_zero_gain_ties",
                "min_legit_gap", "min_positive_legit_gap",
                "argmin_run", "argmin_iter", "argmin_depth",
                "argmin_rank0_feat", "argmin_rank0_bin",
                "argmin_pos_run", "argmin_pos_iter", "argmin_pos_depth",
            ],
        )
        w.writeheader()
        for cid in CONFIGS:
            rows = by_config.get(cid, [])
            if not rows:
                w.writerow({"config_id": cid, "n_nonflip_nodes": 0,
                            "n_zero_gain_ties": 0, "min_legit_gap": "",
                            "min_positive_legit_gap": "",
                            "argmin_run": "", "argmin_iter": "", "argmin_depth": "",
                            "argmin_rank0_feat": "", "argmin_rank0_bin": "",
                            "argmin_pos_run": "", "argmin_pos_iter": "",
                            "argmin_pos_depth": ""})
                continue
            argmin = min(rows, key=lambda r: r["legit_gap"])
            pos_rows = [r for r in rows if r["legit_gap"] > 0.0]
            ties = len(rows) - len(pos_rows)
            if pos_rows:
                argmin_pos = min(pos_rows, key=lambda r: r["legit_gap"])
                mpos = argmin_pos["legit_gap"]
                mpos_run = argmin_pos["run_id"]
                mpos_iter = argmin_pos["iter"]
                mpos_depth = argmin_pos["depth_level"]
            else:
                mpos = ""
                mpos_run = mpos_iter = mpos_depth = ""
            w.writerow({
                "config_id": cid,
                "n_nonflip_nodes": len(rows),
                "n_zero_gain_ties": ties,
                "min_legit_gap": argmin["legit_gap"],
                "min_positive_legit_gap": mpos,
                "argmin_run": argmin["run_id"],
                "argmin_iter": argmin["iter"],
                "argmin_depth": argmin["depth_level"],
                "argmin_rank0_feat": argmin["rank0_feat"],
                "argmin_rank0_bin": argmin["rank0_bin"],
                "argmin_pos_run": mpos_run,
                "argmin_pos_iter": mpos_iter,
                "argmin_pos_depth": mpos_depth,
            })

    # ---- ε threading ----
    # ε_min = max flip_gap_t1 over earliest-flip rows (finite only).
    finite_flips = [r for r in flip_rows if math.isfinite(r["flip_gap_t1"])]
    if finite_flips:
        eps_min_row = max(finite_flips, key=lambda r: r["flip_gap_t1"])
        eps_min = eps_min_row["flip_gap_t1"]
    else:
        eps_min_row = None
        eps_min = 0.0  # no flips anywhere — ε can be anything > 0

    # ε_max = min legit_gap over non-flipping rows (all configs pooled).
    if legit_rows:
        eps_max_row = min(legit_rows, key=lambda r: r["legit_gap"])
        eps_max = eps_max_row["legit_gap"]
    else:
        eps_max_row = None
        eps_max = float("nan")

    # ε_max (positive-gap floor) — the smallest STRICTLY POSITIVE legit gap.
    # This measures the "best case" if zero-gain ties were handled by another
    # tiebreaker (they are already deterministic across T1/Path5 in practice,
    # but ε-threading provably cannot cover them).
    positive_legit = [r for r in legit_rows if r["legit_gap"] > 0.0]
    if positive_legit:
        eps_max_pos_row = min(positive_legit, key=lambda r: r["legit_gap"])
        eps_max_positive = eps_max_pos_row["legit_gap"]
    else:
        eps_max_pos_row = None
        eps_max_positive = float("nan")

    if eps_min > 0.0 and eps_max > 0.0:
        safety_ratio = eps_max / eps_min
    else:
        safety_ratio = float("inf") if eps_min == 0.0 else 0.0

    if eps_min > 0.0 and eps_max_positive > 0.0:
        safety_ratio_positive = eps_max_positive / eps_min
    else:
        safety_ratio_positive = (
            float("inf") if eps_min == 0.0 else 0.0
        )

    verdict_pass = safety_ratio >= 2.0 and eps_max > 0.0
    if verdict_pass and eps_min > 0.0:
        recommended_eps = math.sqrt(eps_min * eps_max)
        tolerance = [eps_min * 1.1, eps_max / 1.1]
    elif verdict_pass:
        # No flips observed; pick a small value well below ε_max.
        recommended_eps = eps_max / 10.0
        tolerance = [eps_max / 1000.0, eps_max / 1.1]
    else:
        recommended_eps = None
        tolerance = None

    # Killer = the (config, run, iter, depth) that sets ε_max when FALSIFIED.
    killer = None
    if not verdict_pass and eps_max_row is not None:
        killer = {
            "config_id": eps_max_row["config_id"],
            "run_id": eps_max_row["run_id"],
            "iter": eps_max_row["iter"],
            "depth_level": eps_max_row["depth_level"],
            "legit_gap": eps_max_row["legit_gap"],
        }

    eps_min_detail = None
    if eps_min_row is not None:
        eps_min_detail = {
            "config_id": eps_min_row["config_id"],
            "run_id": eps_min_row["run_id"],
            "iter": eps_min_row["iter"],
            "depth_level": eps_min_row["depth_level"],
            "flip_gap_t1": eps_min_row["flip_gap_t1"],
            "path5_winner_outside_t1_topk": bool(eps_min_row["path5_winner_outside_t1_topk"]),
        }

    # Per-config flip counts for the report.
    flip_count_by_config: dict = defaultdict(int)
    for r in flip_rows:
        flip_count_by_config[r["config_id"]] += 1

    result = {
        "epsilon_min": eps_min,
        "epsilon_max": eps_max,
        "epsilon_max_positive": eps_max_positive,
        "safety_ratio": safety_ratio,
        "safety_ratio_positive": safety_ratio_positive,
        "verdict": "PASS" if verdict_pass else "FALSIFIED",
        "recommended_epsilon": recommended_eps,
        "tolerance": tolerance,
        "epsilon_min_row": eps_min_detail,
        "epsilon_max_row": (
            {
                "config_id": eps_max_row["config_id"],
                "run_id": eps_max_row["run_id"],
                "iter": eps_max_row["iter"],
                "depth_level": eps_max_row["depth_level"],
                "legit_gap": eps_max_row["legit_gap"],
            }
            if eps_max_row is not None else None
        ),
        "epsilon_max_positive_row": (
            {
                "config_id": eps_max_pos_row["config_id"],
                "run_id": eps_max_pos_row["run_id"],
                "iter": eps_max_pos_row["iter"],
                "depth_level": eps_max_pos_row["depth_level"],
                "legit_gap": eps_max_pos_row["legit_gap"],
            }
            if eps_max_pos_row is not None else None
        ),
        "killer": killer,
        "flip_count_total": len(flip_rows),
        "flip_count_by_config": {str(k): v for k, v in sorted(flip_count_by_config.items())},
        "nonflip_node_count_total": len(legit_rows),
    }

    out_path = os.path.join(ANALYSIS_DIR, "epsilon_threading.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Stdout summary.
    print("=" * 60)
    print("DEC-026-G1 ε-calibration analyzer")
    print("=" * 60)
    print(f"flip events (earliest-per-iter): {len(flip_rows)}")
    print(f"non-flipping nodes:              {len(legit_rows)}")
    print()
    print("flips per config:")
    for cid in CONFIGS:
        n = flip_count_by_config.get(cid, 0)
        if n:
            print(f"  config #{cid:2d}: {n} flip(s)")
    if not flip_rows:
        print("  (no flips in any config)")
    print()
    print(f"ε_min                = {eps_min:.8e}  (max flip_gap_t1)")
    print(f"ε_max  (incl. ties)  = {eps_max:.8e}  (min legit_gap)")
    print(f"ε_max  (positive)    = {eps_max_positive:.8e}  (min positive legit_gap)")
    print(f"safety ratio (incl)  = {safety_ratio:.3e}")
    print(f"safety ratio (pos)   = {safety_ratio_positive:.3e}  (target >= 2.0)")
    print(f"verdict              = {result['verdict']}")
    if recommended_eps is not None:
        print(f"recommended ε = {recommended_eps:.8e}")
        print(f"tolerance     = [{tolerance[0]:.8e}, {tolerance[1]:.8e}]")
    if killer is not None:
        print(
            f"killer       = config #{killer['config_id']} "
            f"run {killer['run_id']} iter {killer['iter']} "
            f"depth {killer['depth_level']} legit_gap {killer['legit_gap']:.8e}"
        )
    print()
    print(f"wrote: {flip_path}")
    print(f"wrote: {floor_path}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
