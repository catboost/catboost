"""S28-FU3-REVALIDATE Gate Harness — G5a (DW Cosine both sides) + G5c (LG Cosine both sides).

Gate G5a (DW-NATIVE-COSINE):
  score_function='Cosine' passed to BOTH CPU and MLX. DW N=1000, 5 seeds {42-46},
  rs=0. Ratios must be 5/5 PASS in [0.98, 1.02]. This is the structural proof that
  the DW force-L2 conditional in _cpu_fit_nonoblivious can be safely removed.

Gate G5c (LG-OUTCOME):
  score_function='Cosine' passed to BOTH CPU and MLX. LG N=1000, 5 seeds {42-46},
  rs=0. Ratios recorded regardless; outcome used to decide whether LG force-L2
  conditional is also removable or needs a documented followup (S29 candidate).

Dataset: N=1000, 20 features, depth=6, 128 bins, 50 iters, rs=0.
Seeds: {42, 43, 44, 45, 46}
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/python"
)))
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

OUT_DIR = Path(
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/"
    "docs/sprint28/fu-fu3-revalidate"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03
N = 1000
SEEDS = [42, 43, 44, 45, 46]

G5a_GATE = (0.98, 1.02)
G5c_GATE = (0.98, 1.02)


def make_data(n: int, seed: int):
    """Canonical S26 data generator — must match _make_data in test_python_path_parity.py."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    return X, y


def cpu_fit(X, y, seed: int, grow_policy: str, score_function: str,
            max_leaves: int = 31):
    from catboost import CatBoostRegressor
    kwargs = dict(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        loss_function="RMSE",
        grow_policy=grow_policy,
        score_function=score_function,
        max_bin=BINS,
        random_seed=seed,
        random_strength=0.0,
        bootstrap_type="No",
        verbose=0,
        thread_count=1,
    )
    if grow_policy == "Lossguide":
        kwargs["max_leaves"] = max_leaves
    m = CatBoostRegressor(**kwargs)
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def mlx_fit(X, y, seed: int, grow_policy: str, score_function: str):
    from catboost_mlx import CatBoostMLXRegressor
    m = CatBoostMLXRegressor(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LR,
        loss="rmse",
        grow_policy=grow_policy,
        score_function=score_function,
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
    return float(np.sqrt(
        ((np.asarray(preds, dtype=np.float64) - y.astype(np.float64)) ** 2).mean()
    ))


def run_gate(grow_policy: str, score_function: str, gate_bounds: tuple):
    print(f"\n--- {grow_policy} score_function={score_function} ---")
    results = []
    all_pass = True
    for seed in SEEDS:
        X, y = make_data(N, seed)
        mlx_rmse = mlx_fit(X, y, seed, grow_policy, score_function)
        cpu_rmse = cpu_fit(X, y, seed, grow_policy, score_function)
        ratio = mlx_rmse / cpu_rmse
        lo, hi = gate_bounds
        passed = lo <= ratio <= hi
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        results.append({
            "seed": seed,
            "mlx_rmse": round(mlx_rmse, 6),
            "cpu_rmse": round(cpu_rmse, 6),
            "ratio": round(ratio, 4),
            "pass": passed,
        })
        print(f"  seed={seed}  MLX={mlx_rmse:.6f}  CPU={cpu_rmse:.6f}  "
              f"ratio={ratio:.4f}  {status}")
    verdict = "PASS" if all_pass else "FAIL"
    print(f"  --> {grow_policy}/{score_function} overall: {verdict} "
          f"({sum(r['pass'] for r in results)}/{len(results)})")
    return results, all_pass


def main():
    print("=" * 60)
    print("S28-FU3-REVALIDATE Gate Harness")
    print(f"N={N}, depth={DEPTH}, iters={ITERS}, rs=0, seeds={SEEDS}")
    print("=" * 60)

    # G5a: DW Cosine both sides
    dw_results, dw_pass = run_gate("Depthwise", "Cosine", G5a_GATE)

    # G5c: LG Cosine both sides
    lg_results, lg_pass = run_gate("Lossguide", "Cosine", G5c_GATE)

    output = {
        "G5a_DW_Cosine": {
            "gate": "G5a-DW-NATIVE-COSINE",
            "bounds": list(G5a_GATE),
            "N": N,
            "seeds": SEEDS,
            "grow_policy": "Depthwise",
            "score_function": "Cosine",
            "results": dw_results,
            "verdict": "PASS" if dw_pass else "FAIL",
        },
        "G5c_LG_Cosine": {
            "gate": "G5c-LG-OUTCOME",
            "bounds": list(G5c_GATE),
            "N": N,
            "seeds": SEEDS,
            "grow_policy": "Lossguide",
            "score_function": "Cosine",
            "results": lg_results,
            "verdict": "PASS" if lg_pass else "FAIL",
        },
    }

    out_path = OUT_DIR / "t5-gate-results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  G5a DW Cosine both sides: {'PASS' if dw_pass else 'FAIL'}")
    print(f"  G5c LG Cosine both sides: {'PASS' if lg_pass else 'FAIL'}")
    print("=" * 60)

    return dw_pass, lg_pass


if __name__ == "__main__":
    dw_pass, lg_pass = main()
    sys.exit(0 if dw_pass else 1)
