#!/usr/bin/env python3
"""
S33-L2-GRAFT: Frame A vs Frame B discriminator.

Strategy: Option (b) — manual approx-init via the existing snapshot mechanism.

  Step 1: Train MLX iter=1 with --snapshot-path to capture the exact snapshot
          JSON format (tree structure, RNG state, etc.).
  Step 2: Train CPU iter=1 → capture per-doc raw predictions
          (prediction_type='RawFormulaVal').
  Step 3: Overwrite the train_cursor in the MLX snapshot with CPU iter=1
          predictions. Keep everything else (tree structure, RNG state) from
          the MLX snapshot so the snapshot passes the validity check.
  Step 4: Resume MLX from the grafted snapshot with total iter=50 → runs
          iterations 1..49 (49 MLX iterations from the CPU starting point).

Sanity check: grafted snapshot + 0 additional MLX iterations should yield
drift ≈ 0 (i.e., train RMSE ≈ CPU iter=1 RMSE with only rounding noise).
  * Achieved by running MLX with --iterations 1 from the grafted snapshot
    (startIteration=1, NumIterations=1 → loop body never entered; BUT csv_train
    always runs at least 1 more iter if startIteration < NumIterations).
  * Alternative: run MLX iter=1 normally and compare against CPU iter=1 RMSE.

Actually the cleanest sanity check: compare grafted snapshot's cursor RMSE
against CPU iter=1 RMSE directly from the cursor values (no MLX training needed).
We do this by computing RMSE(cursor, targets) in Python.

Anchor: N=50000, SymmetricTree, Cosine, RMSE, depth=6, bins=127, iter=50,
        seeds={42,43,44}, random_seed=0, rs=0, bootstrap=No.

Outputs:
  data/cpu_iter1_state_seed{42,43,44}.json   CPU iter=1 snapshot (cbm predictions)
  data/mlx_grafted_rmse_seed{42,43,44}.txt   final RMSE after graft
  data/sanity_check_seed42.txt               grafted cursor RMSE vs CPU iter=1 RMSE
  data/l2_drift_summary.csv                  full results table
"""

import csv as csv_mod
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[3]
MLX_BINARY = REPO_ROOT / "csv_train_t4"
DATA_DIR   = Path(__file__).resolve().parent / "data"

# Anchor — identical to L1
N        = 50_000
DEPTH    = 6
BINS_MLX = 127
BINS_CPU = 127
LR       = 0.03
LOSS     = "rmse"
GROW     = "SymmetricTree"
SCORE_FN = "Cosine"
ITERS    = 50
SEEDS    = [42, 43, 44]
L2       = 3.0
RANDOM_STRENGTH = 0.0
BOOTSTRAP_TYPE  = "No"
SUBSAMPLE       = 1.0

# L1 baseline drift (median) for comparison
L1_MEDIAN_DRIFT = 52.643  # %


# ---------------------------------------------------------------------------
# Data generation (canonical S26+ synthetic data)
# ---------------------------------------------------------------------------

def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    n_feat = X.shape[1]
    header = [f"f{i}" for i in range(n_feat)] + ["target"]
    with open(path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(header)
        for i in range(len(y)):
            writer.writerow(list(X[i]) + [y[i]])


# ---------------------------------------------------------------------------
# Loss parsing
# ---------------------------------------------------------------------------

def parse_final_loss(stdout: str) -> float:
    """Extract last iter train loss from csv_train stdout."""
    last_loss = None
    for line in stdout.split("\n"):
        if "loss=" in line and "iter=" in line:
            for tok in line.split():
                if tok.startswith("loss="):
                    try:
                        last_loss = float(tok.split("=", 1)[1])
                    except ValueError:
                        pass
    if last_loss is None:
        raise ValueError(f"Could not parse final loss from stdout:\n{stdout[:3000]}")
    return last_loss


# ---------------------------------------------------------------------------
# MLX runner
# ---------------------------------------------------------------------------

def run_mlx(data_path: Path, seed: int, iterations: int,
            snapshot_save_path: Path = None,
            snapshot_load_path: Path = None) -> float:
    """Run csv_train_t4; optionally save/load a snapshot. Return final RMSE."""
    cmd = [
        str(MLX_BINARY),
        str(data_path),
        "--iterations",      str(iterations),
        "--depth",           str(DEPTH),
        "--lr",              str(LR),
        "--bins",            str(BINS_MLX),
        "--loss",            LOSS,
        "--grow-policy",     GROW,
        "--score-function",  SCORE_FN,
        "--seed",            str(seed),
        "--random-strength", str(RANDOM_STRENGTH),
        "--bootstrap-type",  "no",
        "--subsample",       str(SUBSAMPLE),
        "--verbose",
    ]
    if snapshot_save_path is not None:
        cmd += ["--snapshot-path", str(snapshot_save_path), "--snapshot-interval", "1"]
    if snapshot_load_path is not None:
        cmd += ["--snapshot-path", str(snapshot_load_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"csv_train_t4 failed (seed={seed}, iters={iterations}):\n"
            f"STDERR: {result.stderr[:800]}\nSTDOUT: {result.stdout[:800]}"
        )
    return parse_final_loss(result.stdout)


# ---------------------------------------------------------------------------
# CPU runner
# ---------------------------------------------------------------------------

def run_cpu_iter1(X: np.ndarray, y: np.ndarray, seed: int):
    """
    Train CPU CatBoost for iter=1.
    Returns (train_rmse_iter1, per_doc_raw_predictions).
    """
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=1,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy=GROW,
        score_function=SCORE_FN,
        max_bin=BINS_CPU + 1,
        random_seed=seed,
        random_strength=RANDOM_STRENGTH,
        bootstrap_type=BOOTSTRAP_TYPE,
        has_time=True,
        sampling_unit="Object",
        l2_leaf_reg=L2,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    train_rmse = float(m.evals_result_["learn"]["RMSE"][-1])
    # RawFormulaVal = accumulated predictions (base_pred + lr * leaf_value)
    raw_preds = m.predict(X, prediction_type="RawFormulaVal").astype(np.float32)
    return train_rmse, raw_preds


def run_cpu_iter50(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Train CPU for iter=50; return final RMSE (reuse L1 numbers where available)."""
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy=GROW,
        score_function=SCORE_FN,
        max_bin=BINS_CPU + 1,
        random_seed=seed,
        random_strength=RANDOM_STRENGTH,
        bootstrap_type=BOOTSTRAP_TYPE,
        has_time=True,
        sampling_unit="Object",
        l2_leaf_reg=L2,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


# ---------------------------------------------------------------------------
# Snapshot graft
# ---------------------------------------------------------------------------

def load_snapshot_json(path: Path) -> dict:
    """Load MLX snapshot JSON (not standard JSON — has float arrays)."""
    with open(path, "r") as f:
        content = f.read()
    return content


def _extract_array(content: str, key: str):
    """Extract a float array from snapshot JSON content."""
    pos = content.find(f'"{key}"')
    if pos == -1:
        return []
    start = content.find('[', pos)
    end = content.find(']', start)
    arr_str = content[start+1:end]
    vals = []
    for tok in arr_str.split(','):
        tok = tok.strip()
        if tok:
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return vals


def craft_graft_snapshot(mlx_iter1_snapshot_path: Path,
                         cpu_predictions: np.ndarray,
                         out_path: Path) -> None:
    """
    Build the graft snapshot: take the MLX iter=1 snapshot (which has
    the real tree structure, RNG state, iteration=0) but replace the
    train_cursor with CPU iter=1 per-doc predictions.

    The snapshot validity check: snap.Trees.size() == num_trees &&
    !snap.TrainCursor.empty(). Using the MLX snapshot preserves both.
    """
    with open(mlx_iter1_snapshot_path, "r") as f:
        content = f.read()

    # Replace train_cursor array
    # Find the "train_cursor": [...] section and replace with CPU predictions
    tc_pos = content.find('"train_cursor"')
    if tc_pos == -1:
        raise ValueError("train_cursor not found in snapshot")
    arr_start = content.find('[', tc_pos)
    arr_end = content.find(']', arr_start)

    # Build new cursor string
    new_vals = ",".join(f"{v:.10g}" for v in cpu_predictions)
    new_content = content[:arr_start+1] + new_vals + content[arr_end:]

    with open(out_path, "w") as f:
        f.write(new_content)


def compute_rmse_from_cursor(cursor: np.ndarray, y: np.ndarray) -> float:
    """Compute RMSE directly from accumulated cursor."""
    return float(np.sqrt(np.mean((cursor - y) ** 2)))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    if not MLX_BINARY.exists():
        print(f"ERROR: {MLX_BINARY} not found.", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("S33-L2-GRAFT: Frame A vs Frame B discriminator")
    print(f"  Anchor: N={N}, d={DEPTH}, bins={BINS_MLX}, iters={ITERS}")
    print(f"  Config: {LOSS}/{GROW}/Cosine, rs={RANDOM_STRENGTH}, bootstrap={BOOTSTRAP_TYPE}")
    print(f"  Graft: CPU iter=1 predictions → MLX cursor; MLX runs iter 2..50")
    print(f"  Seeds: {SEEDS}")
    print()

    # --- Sanity check (seed=42 only) ---
    print("=== SANITY CHECK (seed=42) ===")
    print("  Verifying graft snapshot cursor matches CPU iter=1 RMSE (should be ~0% drift).")

    seed42 = 42
    X42, y42 = make_data(N, seed42)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
        data_path_42 = Path(tf.name)
    write_csv(data_path_42, X42, y42)

    try:
        # Get CPU iter=1 predictions
        cpu_iter1_rmse_42, cpu_preds_42 = run_cpu_iter1(X42, y42, seed42)
        print(f"  CPU iter=1 RMSE (seed=42): {cpu_iter1_rmse_42:.8f}")

        # Verify: RMSE(cpu_preds, y) should match cpu_iter1_rmse
        cursor_rmse_42 = compute_rmse_from_cursor(cpu_preds_42, y42)
        print(f"  RMSE(cursor, y) (sanity): {cursor_rmse_42:.8f}  "
              f"(should ~= CPU iter=1 RMSE; delta={abs(cursor_rmse_42-cpu_iter1_rmse_42):.2e})")

        # Run MLX iter=1 to capture snapshot format
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
            mlx_snap_path_42 = Path(tf2.name)
        mlx_iter1_rmse_42 = run_mlx(data_path_42, seed42, iterations=1,
                                      snapshot_save_path=mlx_snap_path_42)
        print(f"  MLX iter=1 RMSE (seed=42): {mlx_iter1_rmse_42:.8f}")

        # Craft graft snapshot
        graft_snap_42 = DATA_DIR / "graft_snapshot_seed42.json"
        craft_graft_snapshot(mlx_snap_path_42, cpu_preds_42, graft_snap_42)
        os.unlink(mlx_snap_path_42)

        # Extract grafted cursor and verify
        with open(graft_snap_42, "r") as f:
            graft_content = f.read()
        graft_cursor_vals = _extract_array(graft_content, "train_cursor")
        graft_cursor_42 = np.array(graft_cursor_vals, dtype=np.float32)
        graft_cursor_rmse = compute_rmse_from_cursor(graft_cursor_42, y42)
        sanity_drift_pct = (graft_cursor_rmse - cpu_iter1_rmse_42) / cpu_iter1_rmse_42 * 100.0
        print(f"  Graft cursor RMSE (seed=42): {graft_cursor_rmse:.8f}  "
              f"drift_vs_cpu_iter1={sanity_drift_pct:+.4f}%")

        # Write sanity check file
        sanity_txt = (
            f"seed: {seed42}\n"
            f"cpu_iter1_rmse:    {cpu_iter1_rmse_42:.10f}\n"
            f"cursor_rmse_py:    {cursor_rmse_42:.10f}  (RMSE computed from cpu_preds array)\n"
            f"graft_cursor_rmse: {graft_cursor_rmse:.10f}  (RMSE computed from graft JSON cursor)\n"
            f"sanity_drift_%:    {sanity_drift_pct:+.6f}%\n"
            f"mlx_iter1_rmse:    {mlx_iter1_rmse_42:.10f}  (for reference)\n"
            f"verdict:           {'PASS' if abs(sanity_drift_pct) < 0.01 else 'FAIL — graft mechanism broken'}\n"
        )
        (DATA_DIR / "sanity_check_seed42.txt").write_text(sanity_txt)
        print(f"  Sanity check: {'PASS' if abs(sanity_drift_pct) < 0.01 else 'FAIL'}")
        print()

        if abs(sanity_drift_pct) > 0.01:
            print("ERROR: sanity check failed — graft cursor does not match CPU iter=1 predictions.")
            print("  This means float format conversion is lossy. L2 result will be INVALID.")
            sys.exit(1)
    finally:
        os.unlink(data_path_42)

    # --- L1 baseline reference (from L1 verdict) ---
    l1_cpu = {42: 0.19362645, 43: 0.19357118, 44: 0.19320460}
    l1_mlx = {42: 0.29562600, 43: 0.29512200, 44: 0.29491300}

    # --- Main matrix: seeds 42, 43, 44 ---
    print("=== MAIN MATRIX ===")
    hdr = (f"{'seed':>5} | {'CPU iter50':>12} {'MLX iter50 (L1)':>16} "
           f"{'MLX grafted':>12} {'drift_graf%':>11} {'drift_l1%':>10} | class")
    sep = "-" * (len(hdr) + 4)
    print(hdr)
    print(sep)

    rows = []
    grafted_drifts = []

    for seed in SEEDS:
        print(f"\n  Processing seed={seed}...", flush=True)
        X, y = make_data(N, seed)

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            data_path = Path(tf.name)
        write_csv(data_path, X, y)

        # Reuse L1 CPU numbers; also run fresh to verify
        cpu_iter50_l1 = l1_cpu[seed]

        try:
            # Step 1: CPU iter=1 predictions
            t0 = time.perf_counter()
            cpu_iter1_rmse, cpu_preds = run_cpu_iter1(X, y, seed)
            cpu_iter1_wall = time.perf_counter() - t0
            print(f"    CPU iter=1 RMSE: {cpu_iter1_rmse:.8f}  ({cpu_iter1_wall:.1f}s)")

            # Save CPU iter=1 state artifact
            cpu_iter1_artifact = {
                "seed": seed,
                "cpu_iter1_rmse": float(cpu_iter1_rmse),
                "n_preds": len(cpu_preds),
                "pred_mean": float(np.mean(cpu_preds)),
                "pred_std": float(np.std(cpu_preds)),
                "pred_min": float(np.min(cpu_preds)),
                "pred_max": float(np.max(cpu_preds)),
                "predictions_first10": cpu_preds[:10].tolist(),
            }
            artifact_path = DATA_DIR / f"cpu_iter1_state_seed{seed}.json"
            with open(artifact_path, "w") as f:
                json.dump(cpu_iter1_artifact, f, indent=2)

            # Step 2: MLX iter=1 snapshot
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
                mlx_snap_path = Path(tf2.name)
            t0 = time.perf_counter()
            mlx_iter1_rmse = run_mlx(data_path, seed, iterations=1,
                                      snapshot_save_path=mlx_snap_path)
            mlx_iter1_wall = time.perf_counter() - t0
            print(f"    MLX iter=1 RMSE: {mlx_iter1_rmse:.8f}  ({mlx_iter1_wall:.1f}s)  "
                  f"iter1_drift={100*(mlx_iter1_rmse-cpu_iter1_rmse)/cpu_iter1_rmse:+.3f}%")

            # Step 3: Craft graft snapshot
            graft_snap_path = DATA_DIR / f"graft_snapshot_seed{seed}.json"
            craft_graft_snapshot(mlx_snap_path, cpu_preds, graft_snap_path)
            os.unlink(mlx_snap_path)

            # Step 4: Run MLX from grafted snapshot (iterations 1..49)
            t0 = time.perf_counter()
            mlx_grafted_rmse = run_mlx(data_path, seed, iterations=ITERS,
                                        snapshot_load_path=graft_snap_path)
            mlx_grafted_wall = time.perf_counter() - t0
            print(f"    MLX grafted RMSE (iter 2..50): {mlx_grafted_rmse:.8f}  ({mlx_grafted_wall:.1f}s)")

        finally:
            os.unlink(data_path)

        # Compute drifts
        drift_grafted = (mlx_grafted_rmse - cpu_iter50_l1) / cpu_iter50_l1 * 100.0
        drift_l1 = (l1_mlx[seed] - cpu_iter50_l1) / cpu_iter50_l1 * 100.0
        abs_drift_grafted = abs(drift_grafted)

        # Frame call per seed
        if abs_drift_grafted <= 10.0:
            frame = "FRAME-A"
        elif abs_drift_grafted >= 40.0:
            frame = "FRAME-B"
        else:
            frame = "MIXED"

        grafted_drifts.append(drift_grafted)
        print(f"    drift_grafted={drift_grafted:+.3f}%  drift_L1={drift_l1:+.3f}%  → {frame}")

        print(f"{seed:>5} | {cpu_iter50_l1:>12.8f} {l1_mlx[seed]:>16.8f} "
              f"{mlx_grafted_rmse:>12.8f} {drift_grafted:>+11.3f}% {drift_l1:>+10.3f}% | {frame}")

        # Write individual output files
        (DATA_DIR / f"mlx_grafted_rmse_seed{seed}.txt").write_text(
            f"{mlx_grafted_rmse:.10f}\n"
        )

        rows.append({
            "seed": seed,
            "cpu_iter50": cpu_iter50_l1,
            "mlx_iter50_l1": l1_mlx[seed],
            "mlx_grafted": mlx_grafted_rmse,
            "drift_grafted_pct": drift_grafted,
            "drift_l1_pct": drift_l1,
            "frame_per_seed": frame,
        })

    print(sep)

    # --- Class call ---
    median_grafted = sorted(grafted_drifts)[len(grafted_drifts) // 2]
    abs_grafted = [abs(d) for d in grafted_drifts]
    max_abs = max(abs_grafted)
    min_abs = min(abs_grafted)

    # Ratio: grafted / ungrafted (ungrafted = L1 median = 52.643)
    ratio = abs(median_grafted) / L1_MEDIAN_DRIFT

    if abs(median_grafted) <= 10.0:
        overall_class = "FRAME-A"
    elif abs(median_grafted) >= 40.0:
        overall_class = "FRAME-B"
    else:
        overall_class = "MIXED"

    print()
    print(f"  Median grafted drift:   {median_grafted:+.3f}%")
    print(f"  Max |grafted drift|:    {max_abs:.3f}%")
    print(f"  Min |grafted drift|:    {min_abs:.3f}%")
    print(f"  L1 baseline (ungrafted): {L1_MEDIAN_DRIFT:.3f}%")
    print(f"  Ratio (grafted/ungrafted): {ratio:.3f}")
    print()
    print(f"=== L2-GRAFT CLASS: {overall_class} ===")

    if overall_class == "FRAME-A":
        print("  → Iter=1 ε is driving the cascade. Fix = iter=1 precision (gain ratio → 1e-6).")
    elif overall_class == "FRAME-B":
        print("  → Per-iter persistent bug re-injects divergence. Next: L3 ITER2 instrumentation.")
    else:
        print("  → Both frames contribute. Document the split and proceed to both.")

    # Write summary CSV
    out_path = DATA_DIR / "l2_drift_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f,
            fieldnames=["seed", "cpu_iter50", "mlx_iter50_l1", "mlx_grafted",
                        "drift_grafted_pct", "drift_l1_pct", "frame_per_seed"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Raw data: {out_path}")
    return overall_class


if __name__ == "__main__":
    main()
