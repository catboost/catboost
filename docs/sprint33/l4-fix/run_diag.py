#!/usr/bin/env python3
"""Run diagnostic binary to check dimGrads vs statsK at iter=1 (0-indexed) depth=0.

Creates a fresh graft snapshot each run to avoid overwrite issues.
"""
import json
import os
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[3]
DIAG_BINARY = REPO_ROOT / "csv_train_l4_diag"
L3_DATA_DIR = REPO_ROOT / "docs/sprint33/l3-iter2/data"
L4_DATA_DIR = REPO_ROOT / "docs/sprint33/l4-fix/data"

# Anchor
N=50000; DEPTH=6; BINS=127; LR=0.03
LOSS="rmse"; GROW="SymmetricTree"; SCORE_FN="Cosine"
SEED=42; RS=0.0; BOOTSTRAP="no"; SUBSAMPLE=1.0

def make_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y

def write_csv(path, X, y):
    with open(path, "w") as f:
        f.write(",".join([f"f{i}" for i in range(X.shape[1])] + ["target"]) + "\n")
        for i in range(len(y)):
            f.write(",".join([f"{v:.8g}" for v in X[i]] + [f"{y[i]:.8g}"]) + "\n")

def craft_graft_snapshot(snap_path, cpu_preds, out_path):
    with open(snap_path) as f:
        content = f.read()
    tc_pos = content.find('"train_cursor"')
    arr_start = content.find('[', tc_pos)
    arr_end = content.find(']', arr_start)
    new_vals = ",".join(f"{v:.10g}" for v in cpu_preds)
    new_content = content[:arr_start+1] + new_vals + content[arr_end:]
    with open(out_path, "w") as f:
        f.write(new_content)

X, y = make_data(N, SEED)
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as tmp:
    csv_path = tmp.name
write_csv(Path(csv_path), X, y)
print(f"Data written: {csv_path}")

# Step 1: run iter=0 to get base snapshot (use DIAG binary to ensure same binary)
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
    base_snap_path = Path(tf.name)

r0 = subprocess.run([
    str(DIAG_BINARY), csv_path,
    "--iterations", "1", "--depth", str(DEPTH), "--lr", str(LR),
    "--bins", str(BINS), "--loss", LOSS, "--grow-policy", GROW,
    "--score-function", SCORE_FN, "--seed", str(SEED),
    "--random-strength", str(RS), "--bootstrap-type", BOOTSTRAP,
    "--subsample", str(SUBSAMPLE),
    "--snapshot-path", str(base_snap_path),
    "--snapshot-interval", "1",
], capture_output=True, text=True, timeout=120)
if r0.returncode != 0:
    print(f"ERROR iter=0: {r0.stderr[:200]}")
    import sys; sys.exit(1)

with open(base_snap_path) as f:
    base_data = f.read()
base_iter = int(base_data.split('"iteration":')[1].split(',')[0].strip())
print(f"Base snapshot: iteration={base_iter} (should be 0)")

# Extract CPU iter=1 cursor from the ORIGINAL L3 graft snapshot
# (before it was overwritten by the phase1 run)
# The CPU cursor should be in cpu_grad_iter2.bin: cpu_grad = cpu_cursor - y
# so cpu_cursor = cpu_grad + y
cpu_grad = np.fromfile(str(L3_DATA_DIR / "cpu_grad_iter2.bin"), dtype=np.float32)
cpu_cursor_from_grad = (cpu_grad + y).astype(np.float32)
print(f"CPU cursor (from grad+y): mean={cpu_cursor_from_grad.mean():.6f}, "
      f"absmax={np.abs(cpu_cursor_from_grad).max():.6f}")

# Step 2: graft into temp file
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
    graft_path = Path(tf2.name)
craft_graft_snapshot(base_snap_path, cpu_cursor_from_grad, graft_path)
os.unlink(base_snap_path)
print(f"Graft snapshot created at {graft_path}")

# Verify iteration
with open(graft_path) as f:
    graft_data = f.read()
graft_iter = int(graft_data.split('"iteration":')[1].split(',')[0].strip())
print(f"Graft iteration={graft_iter} (should be 0)")

# Step 3: run diagnostic with fresh graft
# Clear L4 data outputs
for p in L4_DATA_DIR.glob("mlx_*.bin"):
    p.unlink()
for p in L4_DATA_DIR.glob("mlx_*.json"):
    p.unlink()

env = os.environ.copy()
env["CATBOOST_MLX_DUMP_ITER2_GRAD"]   = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_HIST"]   = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_TREE"]   = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_LEAVES"] = str(L4_DATA_DIR)
env["CATBOOST_MLX_DUMP_ITER2_APPROX"] = str(L4_DATA_DIR)

cmd = [
    str(DIAG_BINARY), csv_path,
    "--iterations", "2", "--depth", str(DEPTH), "--lr", str(LR),
    "--bins", str(BINS), "--loss", LOSS, "--grow-policy", GROW,
    "--score-function", SCORE_FN, "--seed", str(SEED),
    "--random-strength", str(RS), "--bootstrap-type", BOOTSTRAP,
    "--subsample", str(SUBSAMPLE), "--snapshot-path", str(graft_path),
]
print(f"\nRunning diagnostic...")
r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
print("=== STDERR ===")
print(r.stderr)
print("=== STDOUT ===")
print(r.stdout[-400:])
if r.returncode != 0:
    print(f"ERROR: return={r.returncode}")

os.unlink(graft_path)

# Check results
hist_path = L4_DATA_DIR / "mlx_hist_d0_iter2.bin"
grad_path = L4_DATA_DIR / "mlx_grad_iter2.bin"
if hist_path.exists():
    hist = np.fromfile(str(hist_path), dtype=np.float32)
    n = len(hist) // 2
    print(f"\nHistogram grad block sum: {hist[:n].sum():.6f}  (expected ~{cpu_grad.sum()*20:.4f})")
    print(f"First 5 grad bins: {hist[:5]}")
if grad_path.exists():
    g = np.fromfile(str(grad_path), dtype=np.float32)
    print(f"mlx_grad sum: {g.sum():.8f}  cpu_grad sum: {cpu_grad.sum():.8f}")
    print(f"mlx_grad vs cpu_grad max_diff: {np.abs(g - cpu_grad).max():.3e}")

os.unlink(csv_path)
