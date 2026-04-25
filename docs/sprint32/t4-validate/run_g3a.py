#!/usr/bin/env python3
"""
S32-T4-VALIDATE G3a: depth=0 gain ratio = 1.000 +/- 1e-4 across seeds 42, 43, 44.

Method:
  1. Run csv_train_t4_audit (compiled with -DITER1_AUDIT -DCOSINE_T3_MEASURE) on
     N=50000, depth=6, bins=127 (DEC-039 cap), loss=RMSE, ST+Cosine, rs=0.
  2. Run CatBoost CPU (catboost pip) with the same data and border_count=127.
  3. Extract depth=0 winner gain from each side.
  4. Compute ratio = MLX_gain / CPU_gain.
  5. PASS criterion: |ratio - 1.000| <= 1e-4 for all three seeds.

Note: DEC-039 caps MLX at 127 borders. CPU uses border_count=127 explicitly.
Both sides use 127 borders; max_bin=128 in CSV/catboost pip maps to 127 borders
(border_count = max_bin - 1 in CatBoost pip API).
"""

import csv as csv_mod
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT    = Path(__file__).resolve().parents[3]
AUDIT_BINARY = REPO_ROOT / "csv_train_t4_audit"
DATA_DIR     = REPO_ROOT / "docs" / "sprint32" / "t4-validate" / "data"

# Anchor parameters
N         = 50_000
DEPTH     = 6
BINS_MLX  = 127   # DEC-039 cap
BINS_CPU  = 127   # border_count=127 (= max_bin - 1 for max_bin=128)
LR        = 0.03
LOSS      = "rmse"
GROW      = "SymmetricTree"
SCORE_FN  = "Cosine"
ITERS     = 1
SEEDS     = [42, 43, 44]
L2        = 3.0

G3A_TOLERANCE = 1e-4


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


def run_mlx_audit(data_path: Path, seed: int) -> float:
    """Run ITER1_AUDIT binary; return depth=0 winner gain."""
    tmp_dir = Path(tempfile.mkdtemp())
    env = os.environ.copy()
    env["ITER1_AUDIT_OUTDIR"] = str(tmp_dir)

    cmd = [
        str(AUDIT_BINARY),
        str(data_path),
        "--iterations", str(ITERS),
        "--depth",      str(DEPTH),
        "--lr",         str(LR),
        "--bins",       str(BINS_MLX),
        "--loss",       LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed",       str(seed),
        "--random-strength", "0",
        "--bootstrap-type", "no",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            f"csv_train_t4_audit failed (seed={seed}):\n"
            f"STDERR: {result.stderr[:500]}"
        )

    json_path = tmp_dir / f"mlx_splits_seed{seed}.json"
    if json_path.exists():
        with open(json_path) as fh:
            data = json.load(fh)
        layers = data.get("layers", [])
        if layers:
            return float(layers[0]["winner"]["gain"])

    # Fallback: parse [AUDIT] depth=0 gain from stderr
    for line in result.stderr.splitlines():
        if "[AUDIT]" in line and "depth=0" in line and "gain=" in line:
            m = re.search(r"gain=([\d.e+\-]+)", line)
            if m:
                return float(m.group(1))

    raise RuntimeError(
        f"Cannot extract depth=0 gain from ITER1_AUDIT output (seed={seed}).\n"
        f"STDERR last 500 chars: {result.stderr[-500:]}"
    )


def run_cpu_gain(csv_path: Path, seed: int) -> float:
    """Run CatBoost CPU with Debug log; return depth=0 winner score."""
    script = f"""
import re, sys
from catboost import CatBoostRegressor, Pool
import pandas as pd, numpy as np

df = pd.read_csv('{csv_path}')
X = df.drop('target', axis=1).values.astype('float32')
y = df['target'].values.astype('float32')

pool = Pool(X, y)
pool.quantize(border_count={BINS_CPU}, feature_border_type='GreedyLogSum')

cb = CatBoostRegressor(
    iterations=1,
    depth={DEPTH},
    learning_rate={LR},
    loss_function='RMSE',
    grow_policy='SymmetricTree',
    score_function='Cosine',
    random_seed={seed},
    random_strength=0.0,
    l2_leaf_reg={L2},
    bootstrap_type='No',
    logging_level='Debug',
)
cb.fit(pool)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"CatBoost CPU subprocess failed (seed={seed}):\n"
            f"STDERR: {result.stderr[:400]}"
        )
    # First Debug winner line = depth=0: "<feat_idx>, bin=<int> score <float>"
    for line in result.stdout.splitlines():
        m = re.match(r"^\s*\d+,\s*bin=\d+\s+score\s+([\d.e+\-]+)", line)
        if m:
            return float(m.group(1))
    raise RuntimeError(
        f"No depth=0 winner score in CatBoost Debug output (seed={seed}).\n"
        f"STDOUT: {result.stdout[:1000]}"
    )


def main():
    if not AUDIT_BINARY.exists():
        print(f"ERROR: {AUDIT_BINARY} not found.", file=sys.stderr)
        print("Build with:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DITER1_AUDIT -DCOSINE_T3_MEASURE \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_t4_audit", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("G3a — depth=0 gain ratio = 1.000 +/- 1e-4 (3 seeds)")
    print(f"  Anchor: N={N}, depth={DEPTH}, bins_mlx={BINS_MLX}, bins_cpu={BINS_CPU}")
    print(f"  Config: {LOSS}/{GROW}/Cosine, rs=0, L2={L2}, LR={LR}")
    print(f"  Tolerance: +/- {G3A_TOLERANCE}")
    print()

    hdr = f"{'seed':>5} | {'MLX_gain':>13} {'CPU_gain':>13} {'ratio':>9} {'|r-1|':>9} | gate"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    rows = []
    overall_pass = True

    for seed in SEEDS:
        X, y = make_data(N, seed)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            data_path = Path(tf.name)
        try:
            write_csv(data_path, X, y)
            mlx_gain = run_mlx_audit(data_path, seed)
            cpu_gain = run_cpu_gain(data_path, seed)
        finally:
            os.unlink(data_path)

        ratio = mlx_gain / cpu_gain if cpu_gain != 0 else float("nan")
        delta = abs(ratio - 1.0)
        cell_pass = math.isfinite(delta) and delta <= G3A_TOLERANCE
        gate_str = "PASS" if cell_pass else f"FAIL (delta={delta:.2e})"
        if not cell_pass:
            overall_pass = False

        print(f"{seed:>5} | {mlx_gain:>13.6f} {cpu_gain:>13.6f} {ratio:>9.6f} {delta:>9.2e} | {gate_str}")
        rows.append({
            "seed": seed,
            "mlx_gain": mlx_gain,
            "cpu_gain": cpu_gain,
            "ratio": ratio,
            "delta": delta,
            "cell_pass": cell_pass,
        })

    print(sep)

    out_path = DATA_DIR / "g3a_gain_ratio.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=["seed", "mlx_gain", "cpu_gain", "ratio", "delta", "cell_pass"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRaw data: {out_path}")

    verdict = "PASS" if overall_pass else "FAIL"
    print(f"\n=== G3a RESULT: {verdict} ===")
    for r in rows:
        print(f"  seed={r['seed']}: ratio={r['ratio']:.6f}  delta={r['delta']:.2e}  "
              f"{'PASS' if r['cell_pass'] else 'FAIL'}")
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
