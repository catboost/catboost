"""S29-LG-SPIKE-T1 harness — DEC-034 mechanism spike (task #84).

Measures iter-1 aggregate RMSE drift for Lossguide + Cosine at N=1000,
depth=3, max_leaves=8, 3 seeds (0, 1, 2).  Compares to the ST+Cosine
iter-1 anchor (0.77%) from docs/sprint28/fu-obliv-dispatch/t7-gate-report.md
to discriminate between:

  Outcome A (shared mechanism): LG iter-1 drift ~= ST+Cosine iter-1 anchor
    (drift ~< 1.5%).  Float32 leaf-value precision is the sole driver.
  Outcome B (decoupled mechanism): LG iter-1 drift >= 5%.  Priority-queue
    leaf ordering amplifies float32 gain noise into structurally different
    trees.
  Outcome C (ambiguous): 1.5% <= drift < 5%.  No clean classification.

This harness is data + harness ONLY.  Verdict lives in #85 (S29-LG-SPIKE-T2).

REPRODUCTION REQUIREMENT — local guard bypass (NOT committed):
  To run this harness, the Python-layer guard in
    python/catboost_mlx/core.py::_validate_params
  and the C++-layer guard in
    catboost/mlx/train_api.cpp::TrainConfigToInternal
  that raise on Cosine + Lossguide MUST be temporarily bypassed in the
  working copy.  The bypass is LOCAL ONLY; the bypassed state is not part
  of this commit.  After bypassing, rebuild `_core.so`:

    cd python && python setup.py build_ext --inplace

  The rebuilt `.so` is copied to python/catboost_mlx/ by setup.py
  automatically (no nanobind cache trap).  See this directory's README.md.

Artifacts written:
  data/iter1_drift.json                # PRIMARY — 3 seeds iter=1 drift
  data/iter_curve.csv (if produced)    # SECONDARY — drift vs iter curve
  data/tree_structure_iter1.json (tert.) # TERTIARY — root-split comparison
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the in-process catboost_mlx is the locally-built one.
_REPO = Path(
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
)
sys.path.insert(0, str(_REPO / "python"))

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Measurement cell — matches the S28 precedent tied to DEC-034 scope.
# N=1000, depth=3, max_leaves=8 per task spec.  Seeds 0, 1, 2 per task spec.
# ---------------------------------------------------------------------------

OUT_DIR = _REPO / "docs" / "sprint29" / "lg-mechanism-spike"
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N = 1000
FEATURES = 10
DEPTH = 3
MAX_LEAVES = 8
LR = 0.03
BINS = 128
SEEDS = [0, 1, 2]

# ST+Cosine iter-1 anchor from docs/sprint28/fu-obliv-dispatch/t7-gate-report.md
ST_COSINE_ITER1_ANCHOR_PCT = 0.77

# Outcome thresholds (per task spec #84).
OUTCOME_A_MAX_PCT = 1.5   # mean drift <= this -> shared mechanism
OUTCOME_B_MIN_PCT = 5.0   # mean drift >= this -> decoupled mechanism


def make_data(n: int, seed: int):
    """Synthetic regression data. 10 features, signal in f0/f1, 10% noise.

    Matches the shape family used by test_python_path_parity.py and
    t5-gate-harness.py (which use 20 features); here FEATURES=10 per the
    task spec.  Noise scale and signal shape are unchanged.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    return X, y


def cpu_rmse_lg_cosine(X, y, seed: int, iterations: int) -> float:
    """CPU CatBoost Lossguide + Cosine reference train RMSE.

    CPU is the authoritative reference.  score_function='Cosine' is CPU's
    natural Lossguide default; stated explicitly per DEC-031 Rule 3.
    max_leaves=MAX_LEAVES matches MLX.  bootstrap_type='No' avoids
    sample-subsampling randomness that would diverge between backends.
    """
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=iterations,
        depth=DEPTH,
        max_leaves=MAX_LEAVES,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy="Lossguide",
        score_function="Cosine",
        max_bin=BINS,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="No",
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def mlx_rmse_lg_cosine(X, y, seed: int, iterations: int) -> float:
    """MLX catboost-mlx Lossguide + Cosine train RMSE (requires guard bypass).

    Uses the nanobind in-process path via CatBoostMLXRegressor; RMSE pulled
    from _train_loss_history[-1] (populated by the nanobind path).
    """
    from catboost_mlx import CatBoostMLXRegressor

    m = CatBoostMLXRegressor(
        iterations=iterations,
        depth=DEPTH,
        max_leaves=MAX_LEAVES,
        learning_rate=LR,
        loss="rmse",
        grow_policy="Lossguide",
        score_function="Cosine",
        bins=BINS,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="no",
        verbose=False,
    )
    m.fit(X, y)
    if m._train_loss_history:
        return float(m._train_loss_history[-1])
    preds = m.predict(X)
    return float(
        np.sqrt(((np.asarray(preds, dtype=np.float64) - y.astype(np.float64)) ** 2).mean())
    )


# ---------------------------------------------------------------------------
# PRIMARY — iter-1 drift (3 seeds)
# ---------------------------------------------------------------------------


def run_primary() -> dict:
    """Compute iter-1 RMSE drift per seed and overall mean."""
    per_seed = []
    t0 = time.time()
    for seed in SEEDS:
        X, y = make_data(N, seed)
        cpu = cpu_rmse_lg_cosine(X, y, seed, iterations=1)
        mlx = mlx_rmse_lg_cosine(X, y, seed, iterations=1)
        drift_pct = 100.0 * abs(mlx - cpu) / cpu
        per_seed.append({
            "seed": seed,
            "cpu_rmse": round(cpu, 6),
            "mlx_rmse": round(mlx, 6),
            "drift_pct": round(drift_pct, 4),
        })
        print(f"  [iter=1] seed={seed}  CPU={cpu:.6f}  MLX={mlx:.6f}  "
              f"drift={drift_pct:.3f}%")

    drifts = [r["drift_pct"] for r in per_seed]
    mean_pct = float(np.mean(drifts))
    std_pct = float(np.std(drifts))
    elapsed = time.time() - t0

    summary = {
        "measurement": "LG+Cosine iter-1 aggregate RMSE drift vs CPU",
        "config": {
            "N": N,
            "features": FEATURES,
            "depth": DEPTH,
            "max_leaves": MAX_LEAVES,
            "learning_rate": LR,
            "bins": BINS,
            "iterations": 1,
            "score_function": "Cosine",
            "grow_policy": "Lossguide",
            "bootstrap_type": "no",
            "random_strength": 0.0,
            "seeds": SEEDS,
        },
        "per_seed": per_seed,
        "mean_drift_pct": round(mean_pct, 4),
        "std_drift_pct": round(std_pct, 4),
        "st_cosine_iter1_anchor_pct": ST_COSINE_ITER1_ANCHOR_PCT,
        "outcome_thresholds": {
            "A_shared_max_pct": OUTCOME_A_MAX_PCT,
            "B_decoupled_min_pct": OUTCOME_B_MIN_PCT,
            "C_ambiguous_range": [OUTCOME_A_MAX_PCT, OUTCOME_B_MIN_PCT],
        },
        "elapsed_sec": round(elapsed, 2),
        "notes": (
            "Authoritative verdict lives in task #85 (S29-LG-SPIKE-T2). "
            "This harness writes data only; the outcome classification "
            "is intentionally not encoded here to keep data and verdict "
            "separate artefacts."
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# SECONDARY — drift-vs-iter curve (iter in {1,2,5,10,25,50})
# ---------------------------------------------------------------------------


ITER_CURVE_POINTS = [1, 2, 5, 10, 25, 50]


def run_secondary(rows: list) -> None:
    """Extend PRIMARY to multiple iteration counts. Appends to `rows`."""
    for iters in ITER_CURVE_POINTS:
        for seed in SEEDS:
            X, y = make_data(N, seed)
            cpu = cpu_rmse_lg_cosine(X, y, seed, iterations=iters)
            mlx = mlx_rmse_lg_cosine(X, y, seed, iterations=iters)
            drift_pct = 100.0 * abs(mlx - cpu) / cpu
            print(f"  [iter={iters:>2}] seed={seed}  CPU={cpu:.6f}  "
                  f"MLX={mlx:.6f}  drift={drift_pct:.3f}%")
            rows.append({
                "iterations": iters,
                "seed": seed,
                "cpu_rmse": round(cpu, 6),
                "mlx_rmse": round(mlx, 6),
                "drift_pct": round(drift_pct, 4),
            })


def write_iter_curve_csv(rows: list, path: Path) -> None:
    """Write the drift-vs-iter rows to CSV."""
    import csv
    fieldnames = ["iterations", "seed", "cpu_rmse", "mlx_rmse", "drift_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# TERTIARY — root-split structural comparison at iter-1, seed=0
# ---------------------------------------------------------------------------


def _cpu_tree_first_split(cpu_tree_json) -> dict:
    """Walk the CPU CatBoost nested tree JSON and return the root split.

    CPU tree JSON shape (from CatBoost save_model(format='json')) is a
    nested dict: {left, right, split:{border, float_feature_index, ...}}.
    The top-level dict's 'split' field is the root.
    """
    if not isinstance(cpu_tree_json, dict):
        return {}
    split = cpu_tree_json.get("split")
    if not isinstance(split, dict):
        return {}
    return {
        "float_feature_index": split.get("float_feature_index"),
        "border": split.get("border"),
        "split_type": split.get("split_type"),
    }


def run_tertiary() -> dict | None:
    """Compare the root split (feature) of MLX vs CPU at iter-1, seed=0.

    MLX stores quantized bin_threshold (int); CPU stores dequantized border
    (float).  An exact numeric threshold match isn't possible without
    dequantization, so we compare feature_idx — the structural signal.
    Same feature -> priority-queue chose the same first split; split-border
    magnitude drift would then be attributable to float32 bin-quantization
    rather than gain ordering.
    """
    import tempfile

    from catboost import CatBoostRegressor
    from catboost_mlx import CatBoostMLXRegressor

    seed = SEEDS[0]
    X, y = make_data(N, seed)

    cpu_m = CatBoostRegressor(
        iterations=1, depth=DEPTH, max_leaves=MAX_LEAVES,
        learning_rate=LR, loss_function="RMSE",
        grow_policy="Lossguide", score_function="Cosine",
        max_bin=BINS, random_seed=seed, random_strength=0.0,
        bootstrap_type="No", verbose=0, thread_count=1,
    )
    cpu_m.fit(X, y)

    mlx_m = CatBoostMLXRegressor(
        iterations=1, depth=DEPTH, max_leaves=MAX_LEAVES,
        learning_rate=LR, loss="rmse",
        grow_policy="Lossguide", score_function="Cosine",
        bins=BINS, random_seed=seed, random_strength=0.0,
        bootstrap_type="no", verbose=False,
    )
    mlx_m.fit(X, y)

    out: dict = {"seed": seed, "iterations": 1}

    # CPU side — save_model(format='json'), nested {left,right,split} shape
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            cpu_json_path = f.name
        cpu_m.save_model(cpu_json_path, format="json")
        with open(cpu_json_path) as f:
            cpu_json = json.load(f)
        cpu_trees = cpu_json.get("oblivious_trees") or cpu_json.get("trees") or []
        if cpu_trees:
            first_tree = cpu_trees[0]
            root = _cpu_tree_first_split(first_tree)
            out["cpu_root_feature_idx"] = root.get("float_feature_index")
            out["cpu_root_border"] = root.get("border")
            out["cpu_first_tree_raw"] = first_tree
    except Exception as exc:
        out["cpu_tree_error"] = repr(exc)

    # MLX side — _model_data["trees"][0] directly (no file I/O needed).
    # Shape: {"splits":[{feature_idx, bin_threshold, bfs_node_index,...}],
    #         "leaf_values":[...]}.  Root is the split with bfs_node_index==0.
    try:
        mlx_tree = mlx_m._model_data["trees"][0]
        out["mlx_tree"] = mlx_tree
        splits = mlx_tree.get("splits") or []
        root_split = next(
            (s for s in splits if s.get("bfs_node_index") == 0),
            splits[0] if splits else None,
        )
        if root_split is not None:
            out["mlx_root_feature_idx"] = root_split.get("feature_idx")
            out["mlx_root_bin_threshold"] = root_split.get("bin_threshold")
    except Exception as exc:
        out["mlx_tree_error"] = repr(exc)

    # Root-split match flag (feature only — thresholds are in different domains)
    if "cpu_root_feature_idx" in out and "mlx_root_feature_idx" in out:
        out["root_feature_match"] = (
            out["cpu_root_feature_idx"] == out["mlx_root_feature_idx"]
        )

    # Structural summary: how many MLX splits vs CPU splits, and
    # the BFS-ordered feature index sequence on both sides.
    try:
        mlx_tree = mlx_m._model_data["trees"][0]
        mlx_splits_by_bfs = sorted(
            mlx_tree.get("splits") or [],
            key=lambda s: s.get("bfs_node_index", 0),
        )
        out["mlx_splits_bfs_feature_sequence"] = [
            s.get("feature_idx") for s in mlx_splits_by_bfs
        ]
        out["mlx_num_leaves"] = len(mlx_tree.get("leaf_values") or [])
    except Exception as exc:
        out["mlx_bfs_error"] = repr(exc)

    try:
        # Walk CPU tree in BFS order to produce the same feature-index sequence.
        def bfs_feature_sequence(node):
            seq = []
            frontier = [node]
            while frontier:
                next_frontier = []
                for n in frontier:
                    if isinstance(n, dict) and "split" in n:
                        sp = n.get("split", {})
                        seq.append(sp.get("float_feature_index"))
                        if "left" in n:
                            next_frontier.append(n["left"])
                        if "right" in n:
                            next_frontier.append(n["right"])
                frontier = next_frontier
            return seq

        if "cpu_first_tree_raw" in out:
            out["cpu_splits_bfs_feature_sequence"] = bfs_feature_sequence(
                out["cpu_first_tree_raw"]
            )
    except Exception as exc:
        out["cpu_bfs_error"] = repr(exc)

    if (
        "cpu_splits_bfs_feature_sequence" in out
        and "mlx_splits_bfs_feature_sequence" in out
    ):
        out["bfs_feature_sequence_match"] = (
            out["cpu_splits_bfs_feature_sequence"]
            == out["mlx_splits_bfs_feature_sequence"]
        )

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("S29-LG-SPIKE-T1 — DEC-034 mechanism spike (task #84)")
    print(f"  N={N}, features={FEATURES}, depth={DEPTH}, max_leaves={MAX_LEAVES}")
    print(f"  score_function=Cosine, grow_policy=Lossguide")
    print(f"  seeds={SEEDS}, rs=0.0, bins={BINS}")
    print(f"  ST+Cosine iter-1 anchor: {ST_COSINE_ITER1_ANCHOR_PCT}%")
    print("=" * 72)

    # PRIMARY
    print("\n[PRIMARY] iter-1 drift measurement")
    primary = run_primary()
    iter1_path = DATA_DIR / "iter1_drift.json"
    with open(iter1_path, "w") as f:
        json.dump(primary, f, indent=2)
    print(f"  -> {iter1_path}")
    print(f"  mean drift: {primary['mean_drift_pct']:.4f}%  "
          f"(std {primary['std_drift_pct']:.4f}%)")

    # SECONDARY
    run_sec = os.environ.get("S29_SPIKE_SECONDARY", "1") == "1"
    if run_sec:
        print("\n[SECONDARY] drift-vs-iter curve "
              f"(iter in {ITER_CURVE_POINTS})")
        rows: list = []
        # Seed the rows with the primary iter=1 data so the CSV is a
        # single coherent artifact, not a subset. Re-use primary results.
        for r in primary["per_seed"]:
            rows.append({
                "iterations": 1,
                "seed": r["seed"],
                "cpu_rmse": r["cpu_rmse"],
                "mlx_rmse": r["mlx_rmse"],
                "drift_pct": r["drift_pct"],
            })
        # Run the remaining iteration points.
        for iters in [i for i in ITER_CURVE_POINTS if i != 1]:
            for seed in SEEDS:
                X, y = make_data(N, seed)
                cpu = cpu_rmse_lg_cosine(X, y, seed, iterations=iters)
                mlx = mlx_rmse_lg_cosine(X, y, seed, iterations=iters)
                drift_pct = 100.0 * abs(mlx - cpu) / cpu
                print(f"  [iter={iters:>2}] seed={seed}  CPU={cpu:.6f}  "
                      f"MLX={mlx:.6f}  drift={drift_pct:.3f}%")
                rows.append({
                    "iterations": iters,
                    "seed": seed,
                    "cpu_rmse": round(cpu, 6),
                    "mlx_rmse": round(mlx, 6),
                    "drift_pct": round(drift_pct, 4),
                })
        curve_path = DATA_DIR / "iter_curve.csv"
        write_iter_curve_csv(rows, curve_path)
        print(f"  -> {curve_path}")

    # TERTIARY
    run_tert = os.environ.get("S29_SPIKE_TERTIARY", "1") == "1"
    if run_tert:
        print("\n[TERTIARY] root-split structural comparison (iter=1, seed=0)")
        tert = run_tertiary()
        if tert is not None:
            tert_path = DATA_DIR / "tree_structure_iter1.json"
            with open(tert_path, "w") as f:
                json.dump(tert, f, indent=2, default=str)
            print(f"  -> {tert_path}")
            print(f"  keys: {sorted(tert.keys())}")

    print("\nDone.")


if __name__ == "__main__":
    main()
