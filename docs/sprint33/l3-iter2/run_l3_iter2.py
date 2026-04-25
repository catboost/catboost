#!/usr/bin/env python3
"""
S33-L3-ITER2: Per-stage iter=2 instrumentation.

Strategy: reuse the L2 graft mechanism (Option b, snapshot resume).

  Phase 1: Run CPU iter=2 from scratch; collect all four stage outputs.
  Phase 2: Run MLX iter=2 starting from CPU iter=1 grafted state (bit-identical
           cursor, same as L2 sanity verified at -0.000003%).
           Both sides therefore have identical g_i, h_i at start of iter=2.
  Phase 3: Diff S1 -> S2 -> S3 -> S4 in order. First non-noise divergence is
           the bug class.

Stage definitions:
  S1 GRADIENT  : g_i = approx_i - target_i, h_i = 1  (RMSE)
  S2 SPLIT     : depth=0 histogram (sum_g, sum_h per bin) + best split
  S3 LEAF      : leaf values after Newton step + leaf assignment per doc
  S4 APPROX    : per-doc cursor after applying the second tree

Noise threshold:
  max relative diff > 1e-4 OR fraction of elements > 0.1% diverging
  is classified as non-noise.

Anchor: N=50000, SymmetricTree, Cosine, RMSE, depth=6, bins=127, iter=2,
        seed=42, random_seed=0, rs=0, bootstrap=No, has_time=True.

Outputs (in data/):
  {cpu,mlx}_grad_iter2.bin        S1: per-doc g_i (float32, N values)
  {cpu,mlx}_hess_iter2.bin        S1: per-doc h_i (float32, N values)
  {cpu,mlx}_hist_d0_iter2.bin     S2: histogram depth=0 (float32)
  {cpu,mlx}_partstats_d0_iter2.bin S2: partition stats (sumG, sumH per partition)
  {cpu,mlx}_bestsplit_d0_iter2.json S2: best split (feat, bin, gain)
  {cpu,mlx}_leafvalues_iter2.json  S3: leaf values
  {cpu,mlx}_partitions_iter2.bin   S3: per-doc leaf indices (uint32)
  {cpu,mlx}_approx_iter2.bin       S4: per-doc cursor after iter=2 (float32)
  diff_summary.txt                 Tabular Δ stats per stage
"""

import csv as csv_mod
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT  = Path(__file__).resolve().parents[3]
MLX_BINARY = REPO_ROOT / "csv_train_l3"
DATA_DIR   = Path(__file__).resolve().parent / "data"

# Anchor
N        = 50_000
DEPTH    = 6
BINS_MLX = 127
BINS_CPU = 127
LR       = 0.03
LOSS     = "rmse"
GROW     = "SymmetricTree"
SCORE_FN = "Cosine"
ITERS    = 2
SEED     = 42
L2       = 3.0
RANDOM_STRENGTH = 0.0
BOOTSTRAP_TYPE  = "No"
SUBSAMPLE       = 1.0

# Non-noise threshold
REL_DIFF_THRESHOLD  = 1e-4
FRAC_DIFF_THRESHOLD = 0.001   # 0.1%


# ---------------------------------------------------------------------------
# Data generation (canonical S26+ synthetic data — same as L2)
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
# CPU side: extract all four stage outputs
# ---------------------------------------------------------------------------

def cpu_stage_dumps(X: np.ndarray, y: np.ndarray, seed: int, out_dir: Path) -> dict:
    """
    Train CPU CatBoost for 2 iterations and extract all stage outputs.

    Returns dict with keys:
      grad_iter2, hess_iter2 : np.ndarray float32 [N]
      hist_d0_iter2          : np.ndarray float32 — raw sumG/sumH histogram depth=0
      bestsplit_d0_feat, _bin, _gain : int, int, float
      leaf_values_iter2      : np.ndarray float32 [numLeaves]
      partitions_iter2       : np.ndarray uint32  [N]
      approx_iter2           : np.ndarray float32 [N]
    """
    from catboost import CatBoostRegressor

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- iter=1: needed for gradient at start of iter=2 ----
    m1 = CatBoostRegressor(
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
    m1.fit(X, y)
    approx_iter1 = m1.predict(X, prediction_type="RawFormulaVal").astype(np.float32)

    # S1 GRADIENT at start of iter=2: g = approx_iter1 - y, h = 1
    grad_iter2 = (approx_iter1 - y).astype(np.float32)
    hess_iter2 = np.ones(N, dtype=np.float32)

    grad_iter2.tofile(str(out_dir / "cpu_grad_iter2.bin"))
    hess_iter2.tofile(str(out_dir / "cpu_hess_iter2.bin"))
    print(f"  [CPU-S1] grad mean={grad_iter2.mean():.8f} absmax={np.abs(grad_iter2).max():.8f}")

    # ---- iter=2: get tree structure, leaf values, approx ----
    m2 = CatBoostRegressor(
        iterations=2,
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
    m2.fit(X, y)

    # S4 APPROX: per-doc accumulated predictions after 2 iters
    approx_iter2 = m2.predict(X, prediction_type="RawFormulaVal").astype(np.float32)
    approx_iter2.tofile(str(out_dir / "cpu_approx_iter2.bin"))
    print(f"  [CPU-S4] approx mean={approx_iter2.mean():.8f} absmax={np.abs(approx_iter2).max():.8f}")

    # S3 LEAF VALUES: extract from the second tree (index 1, 0-based)
    # CatBoost leaf values are stored per-tree in model internals.
    import json as json_mod
    try:
        model_json = json_mod.loads(m2.get_metadata()["params"])
    except Exception:
        model_json = {}

    # Use get_leaf_values / model internals
    try:
        # CatBoost Python: model._object._get_leaf_values() returns flat list
        # for all trees. For a depth=6 tree, numLeaves=64 per tree.
        all_leaf_values = m2._object._get_leaf_values()
        num_leaves_per_tree = 2 ** DEPTH  # 64 for depth=6
        # Second tree: indices [num_leaves_per_tree ... 2*num_leaves_per_tree)
        leaf_values_iter2 = np.array(all_leaf_values[num_leaves_per_tree:2*num_leaves_per_tree],
                                      dtype=np.float32)
        print(f"  [CPU-S3] leaf_values: {len(leaf_values_iter2)} leaves  "
              f"mean={leaf_values_iter2.mean():.8f} absmax={np.abs(leaf_values_iter2).max():.8f}")
    except Exception as e:
        print(f"  [CPU-S3] WARNING: could not extract leaf values via _object._get_leaf_values(): {e}")
        # Fallback: estimate from approx delta
        approx_delta = approx_iter2 - approx_iter1
        # Can't reconstruct leaf values reliably without partitions; just save zeros as placeholder
        leaf_values_iter2 = np.zeros(2 ** DEPTH, dtype=np.float32)

    # Save leaf values JSON
    leaf_json = [{"leaf": int(i), "leaf_value": float(v)}
                 for i, v in enumerate(leaf_values_iter2)]
    with open(out_dir / "cpu_leafvalues_iter2.json", "w") as f:
        json_mod.dump(leaf_json, f, indent=2)

    # S3 LEAF PARTITIONS (which leaf each doc goes to in tree 2)
    # Reconstruct: delta approx = LR * leaf_value[partition]
    # For RMSE with LR=0.03, the increment = approx_iter2 - approx_iter1.
    approx_delta = approx_iter2 - approx_iter1
    partitions_iter2 = np.zeros(N, dtype=np.uint32)
    if leaf_values_iter2.max() != 0:
        for i in range(N):
            # Find which leaf value matches the delta (within fp32 precision)
            # delta = LR * leaf_value[leaf_idx]
            expected_delta = LR * leaf_values_iter2
            diffs = np.abs(approx_delta[i] - expected_delta)
            partitions_iter2[i] = int(np.argmin(diffs))
    partitions_iter2.tofile(str(out_dir / "cpu_partitions_iter2.bin"))

    # S2 SPLIT: tree structure of the second tree
    # CatBoost's SymmetricTree splits are stored as list of (feature, bin) per depth level.
    # Extract via model JSON dump.
    try:
        import tempfile as tf_mod
        with tf_mod.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        m2.save_model(tmp_path, format="json")
        with open(tmp_path) as f:
            full_json = json_mod.load(f)
        os.unlink(tmp_path)

        # Navigate to second tree splits
        oblivious_trees = full_json.get("oblivious_trees", [])
        if len(oblivious_trees) >= 2:
            tree2 = oblivious_trees[1]  # 0-indexed second tree
            splits = tree2.get("splits", [])
            # splits[0] is depth=0 (the root split)
            if splits:
                d0_split = splits[0]
                feat_id = int(d0_split.get("float_feature_index",
                              d0_split.get("feature_index", -1)))
                bin_id  = int(d0_split.get("border_id",
                              d0_split.get("split_index", -1)))
                # Gain not stored in model JSON; record -1 as placeholder
                bestsplit = {"feat": feat_id, "bin": bin_id, "gain": -1.0,
                             "note": "CPU gain not extractable from model JSON"}
                print(f"  [CPU-S2] bestsplit d=0: feat={feat_id} bin={bin_id}"
                      f"  (gain=n/a from model JSON)")
            else:
                bestsplit = {"feat": -1, "bin": -1, "gain": -1.0, "note": "no splits in tree"}
        else:
            bestsplit = {"feat": -1, "bin": -1, "gain": -1.0, "note": "model has < 2 trees"}
    except Exception as e:
        bestsplit = {"feat": -1, "bin": -1, "gain": -1.0, "note": f"extraction error: {e}"}
        print(f"  [CPU-S2] WARNING: could not extract tree structure: {e}")

    with open(out_dir / "cpu_bestsplit_d0_iter2.json", "w") as f:
        json_mod.dump(bestsplit, f, indent=2)

    # S2 HISTOGRAM: CatBoost does not expose raw histograms directly.
    # We record the partition stats (sumG, sumH at root = depth=0 partition 0)
    # which are computable: sumG = sum(grad), sumH = sum(hess) = N for RMSE.
    sumG_root = float(grad_iter2.sum())
    sumH_root = float(N)  # h_i = 1 for RMSE
    partstats = np.array([sumG_root, sumH_root], dtype=np.float32)
    partstats.tofile(str(out_dir / "cpu_partstats_d0_iter2.bin"))
    # Histogram raw bins not available from Python API; write placeholder note
    hist_note = {
        "note": "CPU raw histogram not extractable from Python CatBoost API",
        "root_sumG": sumG_root,
        "root_sumH": sumH_root,
        "comparison": "S2 diff uses partstats + bestsplit feat/bin; histogram bin-level diff unavailable on CPU side"
    }
    with open(out_dir / "cpu_hist_note.json", "w") as f:
        json_mod.dump(hist_note, f, indent=2)
    print(f"  [CPU-S2] root partition: sumG={sumG_root:.6f} sumH={sumH_root:.1f}")

    return {
        "grad_iter2": grad_iter2,
        "hess_iter2": hess_iter2,
        "approx_iter1": approx_iter1,
        "approx_iter2": approx_iter2,
        "leaf_values_iter2": leaf_values_iter2,
        "partitions_iter2": partitions_iter2,
        "bestsplit_d0": bestsplit,
        "sumG_root": sumG_root,
    }


# ---------------------------------------------------------------------------
# MLX side: run with L3 dump enabled (graft from CPU iter=1)
# ---------------------------------------------------------------------------

def read_snapshot_cursor(snap_path: Path) -> np.ndarray:
    """Extract train_cursor from MLX snapshot JSON (non-standard format)."""
    with open(snap_path, "r") as f:
        content = f.read()
    pos = content.find('"train_cursor"')
    if pos == -1:
        raise ValueError("train_cursor not found in snapshot")
    arr_start = content.find('[', pos)
    arr_end = content.find(']', arr_start)
    vals = []
    for tok in content[arr_start+1:arr_end].split(','):
        tok = tok.strip()
        if tok:
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return np.array(vals, dtype=np.float32)


def craft_graft_snapshot(mlx_snap_path: Path, cpu_preds: np.ndarray, out_path: Path) -> None:
    """Replace train_cursor in MLX snapshot with CPU iter=1 predictions."""
    with open(mlx_snap_path, "r") as f:
        content = f.read()
    tc_pos = content.find('"train_cursor"')
    if tc_pos == -1:
        raise ValueError("train_cursor not found in snapshot")
    arr_start = content.find('[', tc_pos)
    arr_end = content.find(']', arr_start)
    new_vals = ",".join(f"{v:.10g}" for v in cpu_preds)
    new_content = content[:arr_start+1] + new_vals + content[arr_end:]
    with open(out_path, "w") as f:
        f.write(new_content)


def mlx_stage_dumps(data_path: Path, seed: int, cpu_preds_iter1: np.ndarray,
                    out_dir: Path) -> dict:
    """
    Run MLX iter=2 starting from CPU iter=1 grafted cursor.
    Env vars route each dump to out_dir.
    Returns paths to dumped files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: MLX iter=1 to get snapshot format
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
        mlx_snap_path = Path(tf2.name)

    cmd1 = [
        str(MLX_BINARY),
        str(data_path),
        "--iterations", "1",
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS_MLX),
        "--loss", LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed", str(seed),
        "--random-strength", str(RANDOM_STRENGTH),
        "--bootstrap-type", "no",
        "--subsample", str(SUBSAMPLE),
        "--snapshot-path", str(mlx_snap_path),
        "--snapshot-interval", "1",
        "--verbose",
    ]
    r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
    if r1.returncode != 0:
        raise RuntimeError(f"MLX iter=1 failed:\nSTDERR: {r1.stderr[:600]}")

    # Step 2: graft CPU iter=1 cursor into snapshot
    graft_path = out_dir / f"graft_snapshot_seed{seed}.json"
    craft_graft_snapshot(mlx_snap_path, cpu_preds_iter1, graft_path)
    os.unlink(mlx_snap_path)

    # Verify graft: cursor RMSE should match CPU iter=1 RMSE
    graft_cursor = read_snapshot_cursor(graft_path)
    print(f"  [MLX-graft] cursor samples: first5={graft_cursor[:5]}")

    # Step 3: run MLX iter=2 from grafted snapshot with all L3 dumps enabled
    env = os.environ.copy()
    env["CATBOOST_MLX_DUMP_ITER2_GRAD"]   = str(out_dir)
    env["CATBOOST_MLX_DUMP_ITER2_HIST"]   = str(out_dir)
    env["CATBOOST_MLX_DUMP_ITER2_TREE"]   = str(out_dir)
    env["CATBOOST_MLX_DUMP_ITER2_LEAVES"] = str(out_dir)
    env["CATBOOST_MLX_DUMP_ITER2_APPROX"] = str(out_dir)

    cmd2 = [
        str(MLX_BINARY),
        str(data_path),
        "--iterations", "2",
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS_MLX),
        "--loss", LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed", str(seed),
        "--random-strength", str(RANDOM_STRENGTH),
        "--bootstrap-type", "no",
        "--subsample", str(SUBSAMPLE),
        "--snapshot-path", str(graft_path),
        "--verbose",
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=300, env=env)
    if r2.returncode != 0:
        raise RuntimeError(
            f"MLX iter=2 (grafted) failed:\nSTDERR: {r2.stderr[:600]}\n"
            f"STDOUT: {r2.stdout[:600]}"
        )
    print(f"  [MLX-run] stderr (L3 dump lines):")
    for line in r2.stderr.split("\n"):
        if "[L3" in line or "[build]" in line:
            print(f"    {line}")
    print(f"  [MLX-run] stdout snippet: {r2.stdout[:400]}")

    return {}


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------

def read_f32_bin(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32)


def read_u32_bin(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint32)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def diff_arrays_f32(label: str, cpu: np.ndarray, mlx: np.ndarray,
                    rtol: float = REL_DIFF_THRESHOLD) -> dict:
    """Compute element-wise diff statistics. Returns summary dict."""
    if len(cpu) != len(mlx):
        return {"label": label, "status": "SIZE_MISMATCH",
                "cpu_len": len(cpu), "mlx_len": len(mlx)}
    abs_diff = np.abs(cpu.astype(np.float64) - mlx.astype(np.float64))
    denom = np.abs(cpu.astype(np.float64))
    denom_safe = np.where(denom < 1e-12, 1e-12, denom)
    rel_diff = abs_diff / denom_safe

    max_abs  = float(abs_diff.max())
    max_rel  = float(rel_diff.max())
    mean_abs = float(abs_diff.mean())
    frac_diverging = float((rel_diff > rtol).sum()) / len(cpu)

    is_noise = (max_rel <= rtol) and (frac_diverging <= FRAC_DIFF_THRESHOLD)
    status = "CLEAN" if is_noise else "DIVERGENT"

    print(f"  [{label}] max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
          f"frac>{rtol:.0e}={frac_diverging:.4%}  -> {status}")

    return {
        "label": label,
        "status": status,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mean_abs_diff": mean_abs,
        "frac_diverging": frac_diverging,
        "n_elements": len(cpu),
    }


def diff_best_split(cpu_json: dict, mlx_json: dict) -> dict:
    """Diff best split feat/bin; gain comparison is informational only."""
    cpu_feat = cpu_json.get("feat", -1)
    mlx_feat = mlx_json.get("feat", -1)
    cpu_bin  = cpu_json.get("bin",  -1)
    mlx_bin  = mlx_json.get("bin",  -1)
    cpu_gain = float(cpu_json.get("gain", -1))
    mlx_gain = float(mlx_json.get("gain", -1))

    feat_match = (cpu_feat == mlx_feat)
    bin_match  = (cpu_bin  == mlx_bin)

    gain_rel = abs(cpu_gain - mlx_gain) / max(abs(cpu_gain), 1e-12) if cpu_gain > 0 else float("nan")

    status = "CLEAN" if (feat_match and bin_match) else "DIVERGENT"
    print(f"  [S2-SPLIT] CPU=(feat={cpu_feat},bin={cpu_bin},gain={cpu_gain:.6f})  "
          f"MLX=(feat={mlx_feat},bin={mlx_bin},gain={mlx_gain:.6f})  "
          f"feat_match={feat_match}  bin_match={bin_match}  gain_rel={gain_rel:.3e}  -> {status}")

    return {
        "label": "S2-SPLIT",
        "status": status,
        "cpu_feat": cpu_feat, "mlx_feat": mlx_feat, "feat_match": feat_match,
        "cpu_bin":  cpu_bin,  "mlx_bin":  mlx_bin,  "bin_match":  bin_match,
        "cpu_gain": cpu_gain, "mlx_gain": mlx_gain, "gain_rel_diff": gain_rel,
    }


def classify_stage(diffs: list) -> str:
    """Return first divergent stage label, or 'CLEAN' if all stages pass."""
    for d in diffs:
        if d.get("status") == "DIVERGENT":
            return d["label"]
    return "CLEAN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not MLX_BINARY.exists():
        print(f"ERROR: {MLX_BINARY} not found.", file=sys.stderr)
        print(f"  Run: ./docs/sprint33/l3-iter2/scripts/build_l3.sh", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("S33-L3-ITER2: per-stage iter=2 instrumentation")
    print(f"  Anchor: N={N}, d={DEPTH}, bins={BINS_MLX}, seed={SEED}, iters={ITERS}")
    print(f"  Config: {LOSS}/{GROW}/{SCORE_FN}, rs={RANDOM_STRENGTH}, bootstrap={BOOTSTRAP_TYPE}")
    print()

    # Generate data
    X, y = make_data(N, SEED)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
        data_path = Path(tf.name)
    write_csv(data_path, X, y)

    try:
        # ---- CPU side ----
        print("=== CPU SIDE ===")
        cpu = cpu_stage_dumps(X, y, SEED, DATA_DIR)
        print()

        # ---- MLX side ----
        print("=== MLX SIDE ===")
        mlx_stage_dumps(data_path, SEED, cpu["approx_iter1"], DATA_DIR)
        print()

        # ---- Load MLX dumps ----
        print("=== LOADING MLX DUMPS ===")
        mlx_grad    = read_f32_bin(DATA_DIR / "mlx_grad_iter2.bin")
        mlx_hess    = read_f32_bin(DATA_DIR / "mlx_hess_iter2.bin")
        mlx_approx  = read_f32_bin(DATA_DIR / "mlx_approx_iter2.bin")
        mlx_parts   = read_u32_bin(DATA_DIR / "mlx_partitions_iter2.bin")
        mlx_bsplit  = read_json(DATA_DIR / "mlx_bestsplit_d0_iter2.json")

        # Leaf values from JSON
        mlx_leaf_json = read_json(DATA_DIR / "mlx_leafvalues_iter2.json")
        mlx_leaf_vals = np.array([e["leaf_value"] for e in mlx_leaf_json], dtype=np.float32)

        print(f"  MLX grad:   {len(mlx_grad)} elements  mean={mlx_grad.mean():.8f}")
        print(f"  MLX hess:   {len(mlx_hess)} elements  mean={mlx_hess.mean():.8f}")
        print(f"  MLX approx: {len(mlx_approx)} elements  mean={mlx_approx.mean():.8f}")
        print(f"  MLX leaves: {len(mlx_leaf_vals)} leaves  mean={mlx_leaf_vals.mean():.8f}")
        print(f"  MLX split:  feat={mlx_bsplit.get('feat')} bin={mlx_bsplit.get('bin')} "
              f"gain={mlx_bsplit.get('gain'):.6f}")
        print()

        # ---- Diff each stage ----
        print("=== STAGE DIFFS ===")
        diffs = []

        # S1 GRADIENT
        d_grad = diff_arrays_f32("S1-GRADIENT-g", cpu["grad_iter2"], mlx_grad)
        d_hess = diff_arrays_f32("S1-GRADIENT-h", cpu["hess_iter2"], mlx_hess)
        # Combined S1 status: any divergence in either
        s1_status = "DIVERGENT" if (d_grad["status"] == "DIVERGENT" or
                                     d_hess["status"] == "DIVERGENT") else "CLEAN"
        diffs.append({"label": "S1-GRADIENT", "status": s1_status,
                      "grad": d_grad, "hess": d_hess})

        # S2 SPLIT (feat/bin agreement)
        d_split = diff_best_split(cpu["bestsplit_d0"], mlx_bsplit)
        diffs.append(d_split)

        # S3 LEAF VALUES
        if len(cpu["leaf_values_iter2"]) == len(mlx_leaf_vals):
            d_leaf = diff_arrays_f32("S3-LEAF-VALUES",
                                      cpu["leaf_values_iter2"], mlx_leaf_vals)
        else:
            d_leaf = {"label": "S3-LEAF-VALUES", "status": "SIZE_MISMATCH",
                      "cpu_len": len(cpu["leaf_values_iter2"]),
                      "mlx_len": len(mlx_leaf_vals)}
            print(f"  [S3-LEAF-VALUES] SIZE_MISMATCH cpu={d_leaf['cpu_len']} mlx={d_leaf['mlx_len']}")
        diffs.append(d_leaf)

        # S4 APPROX
        d_approx = diff_arrays_f32("S4-APPROX", cpu["approx_iter2"], mlx_approx)
        diffs.append(d_approx)

        print()

        # ---- Class call ----
        first_divergent = classify_stage(diffs)
        print("=== CLASS CALL ===")
        print(f"  First divergent stage: {first_divergent}")
        if first_divergent == "CLEAN":
            print("  All stages CLEAN — no divergence found at iter=2. Unexpected.")
        elif first_divergent.startswith("S1"):
            print("  CLASS: GRADIENT")
            print("  The gradient computation itself diverges at iter=2.")
            print("  For RMSE, g_i = cursor_i - target_i. If cursor is bit-identical")
            print("  (FRAME-B graft sanity), this should not happen. Investigate cursor state.")
        elif first_divergent.startswith("S2"):
            print("  CLASS: SPLIT")
            print("  Histogram build or best-split selection diverges at iter=2.")
            print("  S1 gradients are clean — the bug is in FindBestSplit or DispatchHistogram.")
            print("  Likely the Cosine gain formula at iter>=2 diverges (DEC-036 root cause).")
        elif first_divergent.startswith("S3"):
            print("  CLASS: LEAF")
            print("  Leaf value estimation diverges. S1+S2 clean.")
            print("  Newton step or scatter_add_axis precision issue at iter>=2.")
        elif first_divergent.startswith("S4"):
            print("  CLASS: APPROX")
            print("  Approx update diverges. S1+S2+S3 clean.")
            print("  mx::take or mx::add accumulation issue.")

        # ---- Write diff_summary.txt ----
        summary_lines = [
            "S33-L3-ITER2 diff summary",
            f"Date: 2026-04-24",
            f"Anchor: N={N}, depth={DEPTH}, bins={BINS_MLX}, seed={SEED}",
            f"Config: {GROW}/{SCORE_FN}/{LOSS}, rs={RANDOM_STRENGTH}",
            "",
            f"{'stage':<22} {'status':<12} {'max_rel_diff':<16} {'frac_diverging':<18} {'max_abs_diff':<16}",
            "-" * 88,
        ]
        for d in diffs:
            lbl = d["label"]
            st  = d.get("status", "?")
            mr  = d.get("max_rel_diff", d.get("gain_rel_diff", float("nan")))
            fd  = d.get("frac_diverging", float("nan"))
            ma  = d.get("max_abs_diff", float("nan"))
            summary_lines.append(
                f"{lbl:<22} {st:<12} {mr:<16.4e} {fd:<18.4%} {ma:<16.4e}"
            )
        summary_lines.append("")
        summary_lines.append(f"CLASS CALL: {first_divergent}")
        summary_lines.append("")

        summary_path = DATA_DIR / "diff_summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n")
        print(f"\n  Diff summary -> {summary_path}")

    finally:
        os.unlink(data_path)

    return first_divergent


if __name__ == "__main__":
    result = main()
    print(f"\nL3 class: {result}")
