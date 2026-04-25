#!/usr/bin/env python3
"""
S33-L4 Phase 1: Confirm lazy-eval hypothesis.

Re-runs MLX graft iter=2 (0-indexed iter=1) with csv_train_l4_phase1 binary
(which has mx::eval(dimGrads[k], dimHess[k]) before line 4534) and checks
whether the histogram grad block sum changes from -738.99 to ~+0.228.

Creates a fresh graft snapshot (iter=0 base) using the CPU iter=1 cursor
from the L3 data directory.
"""

import json
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[3]
L4_BINARY   = REPO_ROOT / "csv_train_l4_phase1"
L3_DATA_DIR = REPO_ROOT / "docs/sprint33/l3-iter2/data"
L4_DATA_DIR = REPO_ROOT / "docs/sprint33/l4-fix/data"
L4_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Anchor config (must match L3 experiment exactly)
N        = 50_000
DEPTH    = 6
BINS     = 127
LR       = 0.03
LOSS     = "rmse"
GROW     = "SymmetricTree"
SCORE_FN = "Cosine"
SEED     = 42
RS       = 0.0
BOOTSTRAP = "no"
SUBSAMPLE = 1.0


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def write_csv(path, X, y):
    with open(path, "w") as f:
        header = ",".join([f"f{i}" for i in range(X.shape[1])] + ["target"])
        f.write(header + "\n")
        for i in range(len(y)):
            row = ",".join([f"{v:.8g}" for v in X[i]] + [f"{y[i]:.8g}"])
            f.write(row + "\n")


def craft_graft_snapshot(snap_path: Path, cpu_preds: np.ndarray, out_path: Path) -> None:
    """Replace train_cursor in MLX snapshot with CPU iter=1 predictions."""
    with open(snap_path, "r") as f:
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


print("=== S33-L4 Phase 1 Confirmation ===")
print(f"Binary: {L4_BINARY}")
print()

# Make data (same as L3)
X, y = make_data(N, SEED)
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as tmp:
    csv_path = tmp.name
write_csv(Path(csv_path), X, y)
print(f"Data: {N} rows, 20 features -> {csv_path}")

# --- Step 1: get iter=0 snapshot using Phase 1 binary ---
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
    snap_path = Path(tf.name)

cmd_iter0 = [
    str(L4_BINARY),
    csv_path,
    "--iterations", "1",
    "--depth", str(DEPTH),
    "--lr", str(LR),
    "--bins", str(BINS),
    "--loss", LOSS,
    "--grow-policy", GROW,
    "--score-function", SCORE_FN,
    "--seed", str(SEED),
    "--random-strength", str(RS),
    "--bootstrap-type", BOOTSTRAP,
    "--subsample", str(SUBSAMPLE),
    "--snapshot-path", str(snap_path),
    "--snapshot-interval", "1",
]
print("Step 1: Running MLX iter=0 to get fresh snapshot...")
r0 = subprocess.run(cmd_iter0, capture_output=True, text=True, timeout=120)
if r0.returncode != 0:
    print(f"ERROR: iter=0 run failed: {r0.stderr[:300]}")
    import sys; sys.exit(1)
print(f"  Done. Snapshot: {snap_path}")

# Read snapshot to verify iteration=0
with open(snap_path) as f:
    snap_data = f.read()
iter_in_snap = int(snap_data.split('"iteration":')[1].split(',')[0].strip())
num_trees_in_snap = int(snap_data.split('"num_trees":')[1].split(',')[0].strip())
print(f"  Snapshot: iteration={iter_in_snap}, num_trees={num_trees_in_snap}")

# --- Step 2: graft CPU iter=1 cursor from L3 data ---
# The CPU iter=1 gradients are cpu_grad_iter2.bin: these are grad = cursor_1 - target
# We need the actual cursor (predictions after iter=1), not the gradients.
# The graft_snapshot_seed42.json already has the CPU iter=1 cursor.
# Extract it from the existing L3 graft snapshot:
with open(L3_DATA_DIR / "graft_snapshot_seed42.json") as f:
    existing_graft = f.read()
tc_pos = existing_graft.find('"train_cursor"')
arr_start = existing_graft.find('[', tc_pos)
arr_end = existing_graft.find(']', arr_start)
cpu_cursor_vals = [float(v.strip()) for v in existing_graft[arr_start+1:arr_end].split(',') if v.strip()]
cpu_cursor = np.array(cpu_cursor_vals, dtype=np.float32)
print(f"  CPU iter=1 cursor: N={len(cpu_cursor)}, mean={cpu_cursor.mean():.6f}, "
      f"absmax={np.abs(cpu_cursor).max():.6f}")

# Verify: cpu_grad_iter2 = cpu_cursor - target => cpu_cursor ≈ cpu_grad + target
cpu_grad = np.fromfile(str(L3_DATA_DIR / "cpu_grad_iter2.bin"), dtype=np.float32)
grad_from_cursor = cpu_cursor - y
print(f"  Sanity check: max|cursor-target - cpu_grad| = "
      f"{np.abs(grad_from_cursor - cpu_grad).max():.6e}  (should be ~0)")

# Graft
graft_snap_path = L4_DATA_DIR / "graft_snapshot_phase1.json"
craft_graft_snapshot(snap_path, cpu_cursor, graft_snap_path)
os.unlink(snap_path)
print(f"  Grafted snapshot -> {graft_snap_path}")

# Verify graft content
with open(graft_snap_path) as f:
    graft_data = f.read()
graft_iter = int(graft_data.split('"iteration":')[1].split(',')[0].strip())
print(f"  Graft snapshot: iteration={graft_iter} (should be 0)")

# --- Step 3: run MLX iter=1 (0-indexed) from grafted snapshot with Phase 1 binary ---
env = os.environ.copy()
env["CATBOOST_MLX_DUMP_ITER2_GRAD"]   = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_HIST"]   = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_TREE"]   = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_LEAVES"] = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_APPROX"] = str(L4_DATA_DIR)

cmd_iter1 = [
    str(L4_BINARY),
    csv_path,
    "--iterations", "2",    # run until iter=1 (0-indexed); startIteration=1
    "--depth", str(DEPTH),
    "--lr", str(LR),
    "--bins", str(BINS),
    "--loss", LOSS,
    "--grow-policy", GROW,
    "--score-function", SCORE_FN,
    "--seed", str(SEED),
    "--random-strength", str(RS),
    "--bootstrap-type", BOOTSTRAP,
    "--subsample", str(SUBSAMPLE),
    "--snapshot-path", str(graft_snap_path),
]
print("\nStep 3: Running MLX iter=1 (0-indexed) with Phase 1 binary + L3 dump...")
r1 = subprocess.run(cmd_iter1, capture_output=True, text=True, timeout=180, env=env)
print(f"  stdout: {r1.stdout[-400:]}")
print(f"  stderr: {r1.stderr[-300:]}")
if r1.returncode != 0:
    print(f"ERROR: iter=1 run failed (return={r1.returncode})")
    import sys; sys.exit(1)

# --- Step 4: check histogram ---
hist_l4_path = L4_DATA_DIR / "mlx_hist_d0_iter2.bin"
if not hist_l4_path.exists():
    print("ERROR: histogram dump not found — check dump env vars and binary")
    import sys; sys.exit(1)

hist_l4 = np.fromfile(str(hist_l4_path), dtype=np.float32)
hist_l3 = np.fromfile(str(L3_DATA_DIR / "mlx_hist_d0_iter2.bin"), dtype=np.float32)

n_bins = len(hist_l3) // 2
grad_l3 = hist_l3[:n_bins]
hess_l3 = hist_l3[n_bins:]
grad_l4 = hist_l4[:n_bins]
hess_l4 = hist_l4[n_bins:]

print()
print("=== Histogram Grad Block Comparison ===")
print(f"  L3 (pre-fix) grad sum:  {grad_l3.sum():.6f}")
print(f"  L4 (post-fix) grad sum: {grad_l4.sum():.6f}")
print(f"  Expected (~20 * S1 grad sum): {cpu_grad.sum() * 20:.6f}")
print(f"  L3 hess sum:  {hess_l3.sum():.1f}")
print(f"  L4 hess sum:  {hess_l4.sum():.1f}")
print(f"  Expected hess (~20*N=1000000): {20 * N:.1f}")
print()

# Also check best split
split_l4_path = L4_DATA_DIR / "mlx_bestsplit_d0_iter2.json"
split_cpu_path = L3_DATA_DIR / "cpu_bestsplit_d0_iter2.json"
split_l3_path  = L3_DATA_DIR / "mlx_bestsplit_d0_iter2.json"

print("=== Best Split Comparison ===")
if split_l4_path.exists():
    with open(split_l4_path) as f:
        s4 = json.load(f)
    with open(split_cpu_path) as f:
        sc = json.load(f)
    with open(split_l3_path) as f:
        s3 = json.load(f)
    print(f"  L3 (pre-fix):  feat={s3.get('feat')}, bin={s3.get('bin')}, gain={s3.get('gain'):.4f}")
    print(f"  L4 (post-fix): feat={s4.get('feat')}, bin={s4.get('bin')}, gain={s4.get('gain'):.4f}")
    print(f"  CPU reference: feat={sc.get('feat')}, bin={sc.get('bin')}")
print()

# Phase 1 verdict
expected_grad_sum = cpu_grad.sum() * 20
l3_err = abs(grad_l3.sum() - expected_grad_sum)
l4_err = abs(grad_l4.sum() - expected_grad_sum)
print("=== Phase 1 Verdict ===")
if l4_err < 1.0:
    print("  CONFIRMED: mx::eval barrier fixes histogram grad block sum")
    print(f"  L3 error: {l3_err:.4f} ({grad_l3.sum():.4f} vs expected {expected_grad_sum:.4f})")
    print(f"  L4 error: {l4_err:.6f} ({grad_l4.sum():.6f} vs expected {expected_grad_sum:.6f})")
    print("  Hypothesis: LAZY_EVAL_CONFIRMED")
else:
    print("  NOT CONFIRMED: histogram grad sum still wrong")
    print(f"  L3 sum: {grad_l3.sum():.6f}, L4 sum: {grad_l4.sum():.6f}, Expected: {expected_grad_sum:.6f}")
    print("  Hypothesis: LAZY_EVAL_NOT_CONFIRMED — needs alternative investigation")

os.unlink(csv_path)
