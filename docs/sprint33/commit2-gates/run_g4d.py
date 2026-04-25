#!/usr/bin/env python3
"""DEC-042 G4d: 18-config L2 SymmetricTree parity non-regression.

Verifies the DEC-042 per-side mask fix did NOT break the L2 training path.
Reuses the same sweep as S31-T2 G2d / S32-T4 G3d.

Grid (3 x 3 x 2 = 18 cells):
  N     in {1000, 10000, 50000}
  seed  in {1337, 42, 7}
  rs    in {0.0, 1.0}

Fixed: SymmetricTree, d=6, 128 bins, LR=0.03, 50 iters, RMSE/L2, 20 features.

PASS criterion (segmented, per DEC-031 Rule 5):
  rs=0: ratio MLX_RMSE / CPU_RMSE in [0.98, 1.02]
  rs=1: MLX_RMSE <= CPU_RMSE * 1.02

Binary: csv_train_g4 (DEC-042 per-side mask fix, no -DCOSINE_T3_MEASURE needed for L2)
"""

import csv as csv_mod
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx")
BINARY = REPO / "csv_train_g4"
OUT_DIR = REPO / "docs/sprint33/commit2-gates/data"

FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03

SIZES = [1000, 10_000, 50_000]
SEEDS = [1337, 42, 7]
RS_VALUES = [0.0, 1.0]


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATURES)).astype(np.float32)
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


def parse_final_loss(stdout: str) -> float:
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
        raise ValueError(f"Could not parse final loss:\n{stdout[:1000]}")
    return last_loss


def run_mlx(data_path: Path, seed: int, rs: float) -> float:
    cmd = [
        str(BINARY),
        str(data_path),
        "--iterations", str(ITERS),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--bins", str(BINS),
        "--loss", "rmse",
        "--grow-policy", "SymmetricTree",
        "--score-function", "L2",
        "--seed", str(seed),
        "--random-strength", str(rs),
        "--verbose",
    ]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/mlx/lib"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"csv_train_g4 failed (seed={seed}, rs={rs}):\n{result.stderr[:300]}"
        )
    return parse_final_loss(result.stdout)


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int, rs: float) -> float:
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        score_function="L2",
        max_bin=BINS,
        random_seed=seed,
        random_strength=rs,
        bootstrap_type="No",
        l2_leaf_reg=3.0,
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def main():
    if not BINARY.exists():
        print(f"ERROR: {BINARY} not found.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("G4d — 18-config L2 SymmetricTree parity non-regression (DEC-042 fix)")
    print(f"  Grid: N={SIZES}, seeds={SEEDS}, rs={RS_VALUES}")
    print(f"  Fixed: SymmetricTree, d={DEPTH}, bins={BINS}, LR={LR}, iters={ITERS}, RMSE/L2")
    print(f"  Pass: rs=0 ratio in [0.98,1.02]; rs=1 MLX<=CPU*1.02\n")

    hdr = f"{'N':>7} {'seed':>5} {'rs':>4} | {'CPU_RMSE':>11} {'MLX_RMSE':>11} {'ratio':>7} | gate"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    rows = []
    overall_pass = True
    fail_count = 0

    for N in SIZES:
        for seed in SEEDS:
            X, y = make_data(N, seed)
            for rs in RS_VALUES:
                with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
                    data_path = Path(tf.name)
                try:
                    write_csv(data_path, X, y)
                    mlx_rmse = run_mlx(data_path, seed, rs)
                    cpu_rmse = run_cpu(X, y, seed, rs)
                finally:
                    os.unlink(data_path)

                ratio = mlx_rmse / cpu_rmse if cpu_rmse > 0 else float("nan")

                if rs == 0.0:
                    cell_pass = 0.98 <= ratio <= 1.02
                else:
                    cell_pass = mlx_rmse <= cpu_rmse * 1.02

                gate_str = "PASS" if cell_pass else "FAIL"
                if not cell_pass:
                    overall_pass = False
                    fail_count += 1

                print(f"{N:>7} {seed:>5} {rs:>4.1f} | "
                      f"{cpu_rmse:>11.6f} {mlx_rmse:>11.6f} {ratio:>7.4f} | {gate_str}")

                rows.append({
                    "N": N, "seed": seed, "rs": rs,
                    "cpu_rmse": cpu_rmse, "mlx_rmse": mlx_rmse,
                    "ratio": ratio, "cell_pass": cell_pass,
                })

    print(sep)

    out_csv = OUT_DIR / "g4d_l2_parity.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=["N", "seed", "rs", "cpu_rmse", "mlx_rmse", "ratio", "cell_pass"]
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json = OUT_DIR / "g4d_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "gate": "G4d",
            "pass": overall_pass,
            "pass_count": len(rows) - fail_count,
            "total": len(rows),
            "rows": rows,
        }, f, indent=2)

    print(f"\nRaw data: {out_csv}")
    pass_count = len(rows) - fail_count
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"\n=== G4d RESULT: {verdict} ===")
    print(f"  Cells: {pass_count}/{len(rows)} PASS")
    ratios = [r["ratio"] for r in rows]
    print(f"  Ratio range: [{min(ratios):.4f}, {max(ratios):.4f}]")
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
