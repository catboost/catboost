#!/usr/bin/env python3
"""
S30-T3-MEASURE: Gate G3c — LG-Stress post-Kahan parity (K2 gate, priority-queue stress).

Cell: N=2000, depth=7, max_leaves=64, iterations=100, seeds={0,1,2}
Config: loss=RMSE, grow_policy=Lossguide, score_function='Cosine'
Binary: csv_train_t3 (built with -DCOSINE_T3_MEASURE; K4 fp64 Kahan fix active)

PASS criterion: all 3 seeds have drift ratio MLX_RMSE / CPU_RMSE in [0.98, 1.02].
If ANY seed fails: K2 fires — T4b (LG+Cosine guard removal) is skipped this sprint.

Rationale: This cell intentionally stresses the priority-queue divergence surface.
max_leaves=64 provides 8× the contested-split density of the S29 spike (max_leaves=8).
Per DEC-034 limitations: "With only 8 leaves the queue makes few contested choices;
any latent ordering sensitivity would be more visible at 64+ leaves."
This is the K2 gate per DEC-035.

CPU reference: catboost pip, score_function='Cosine', max_leaves=64.
"""

import csv as csv_mod
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
T3_BINARY = REPO_ROOT / "csv_train_t3"
DATA_DIR  = REPO_ROOT / "docs" / "sprint30" / "t3-measure" / "data"

# G3c cell parameters (K2 gate)
N          = 2_000
DEPTH      = 7
MAX_LEAVES = 64
BINS       = 128
LR         = 0.03
LOSS       = "rmse"
GROW       = "Lossguide"
SCORE_FN   = "Cosine"
ITERS      = 100
SEEDS      = [0, 1, 2]

G3c_PASS_LO = 0.98
G3c_PASS_HI = 1.02


def make_data(n: int, seed: int):
    """Canonical S26 data: 20 features, signal in f0 and f1, 10% noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    header = [f"f{i}" for i in range(X.shape[1])] + ["target"]
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
        raise ValueError(f"Could not parse final loss from stdout:\n{stdout[:2000]}")
    return last_loss


def run_mlx(data_path: Path, seed: int) -> tuple[float, float]:
    cmd = [
        str(T3_BINARY),
        str(data_path),
        "--iterations", str(ITERS),
        "--depth",      str(DEPTH),
        "--max-leaves", str(MAX_LEAVES),
        "--lr",         str(LR),
        "--bins",       str(BINS),
        "--loss",       LOSS,
        "--grow-policy", GROW,
        "--score-function", SCORE_FN,
        "--seed",       str(seed),
        "--verbose",
    ]
    env = os.environ.copy()
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  ERROR: csv_train_t3 exited {result.returncode}", file=sys.stderr)
        print(f"  STDERR: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"csv_train_t3 failed for seed={seed}")
    rmse = parse_final_loss(result.stdout)
    return rmse, elapsed


def run_cpu(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(
        iterations=ITERS,
        depth=DEPTH,
        max_leaves=MAX_LEAVES,
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
    return float(m.evals_result_["learn"]["RMSE"][-1])


def main():
    if not T3_BINARY.exists():
        print(f"ERROR: {T3_BINARY} not found. Build with:", file=sys.stderr)
        print("  clang++ -std=c++17 -O2 -DCOSINE_T3_MEASURE \\", file=sys.stderr)
        print("    -I. -I/opt/homebrew/opt/mlx/include \\", file=sys.stderr)
        print("    -L/opt/homebrew/opt/mlx/lib -lmlx \\", file=sys.stderr)
        print("    -framework Metal -framework Foundation -Wno-c++20-extensions \\", file=sys.stderr)
        print("    catboost/mlx/tests/csv_train.cpp -o csv_train_t3", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"G3c — LG-Stress post-Kahan parity (K2 gate)")
    print(f"  Cell: N={N}, depth={DEPTH}, max_leaves={MAX_LEAVES}, iters={ITERS}, seeds={SEEDS}")
    print(f"  Config: {LOSS}/{GROW}/Cosine  (K4 fp64 fix active)")
    print(f"  PASS criterion: all seeds in [{G3c_PASS_LO}, {G3c_PASS_HI}]")
    print(f"  K2: fires if ANY seed fails (T4b skipped, S31-LG-DEEP-RESIDUAL filed)\n")

    results = []
    raw_rows = []

    for seed in SEEDS:
        print(f"--- seed={seed} ---")
        X, y = make_data(N, seed)

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tf:
            data_path = Path(tf.name)
        try:
            write_csv(data_path, X, y)
            mlx_rmse, wall_secs = run_mlx(data_path, seed)
            cpu_rmse = run_cpu(X, y, seed)
        finally:
            os.unlink(data_path)

        ratio = mlx_rmse / cpu_rmse
        drift_pct = abs(mlx_rmse - cpu_rmse) / cpu_rmse * 100.0
        seed_pass = G3c_PASS_LO <= ratio <= G3c_PASS_HI
        print(f"  MLX={mlx_rmse:.8f}  CPU={cpu_rmse:.8f}  ratio={ratio:.6f}  drift={drift_pct:.4f}%  "
              f"wall={wall_secs:.1f}s  [{'PASS' if seed_pass else 'FAIL'}]")
        results.append({"seed": seed, "ratio": ratio, "drift_pct": drift_pct, "pass": seed_pass})
        raw_rows.append({
            "seed": seed,
            "mlx_rmse": mlx_rmse,
            "cpu_rmse": cpu_rmse,
            "ratio": ratio,
            "drift_pct": drift_pct,
            "wall_secs": wall_secs,
            "pass": seed_pass,
        })

    # Write raw CSV
    out_path = DATA_DIR / "g3c_lg_stress.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["seed","mlx_rmse","cpu_rmse","ratio","drift_pct","wall_secs","pass"])
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\nRaw data: {out_path}")

    # Gate check + K2 decision
    all_pass = all(r["pass"] for r in results)
    print(f"\n=== G3c RESULT ===")
    for r in results:
        print(f"  seed={r['seed']}: ratio={r['ratio']:.6f}  drift={r['drift_pct']:.4f}%  "
              f"[{'PASS' if r['pass'] else 'FAIL'}]")
    print(f"  Threshold: [{G3c_PASS_LO}, {G3c_PASS_HI}]")
    print(f"  G3c: {'PASS' if all_pass else 'FAIL'}")

    if all_pass:
        print("\nK2 NO-FIRE — T4b proceeds (LG+Cosine guard removal authorized)")
        print("G3c PASS")
    else:
        failed = [r["seed"] for r in results if not r["pass"]]
        print(f"\nK2 FIRED — seeds {failed} outside [{G3c_PASS_LO}, {G3c_PASS_HI}]")
        print("T4b SKIPPED — LG+Cosine guard stays this sprint")
        print("Action: file S31-LG-DEEP-RESIDUAL per DEC-035 K2 clause")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    main()
