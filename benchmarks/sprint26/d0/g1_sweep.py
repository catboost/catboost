"""S26-D0-7 G1 gate sweep: 18-config SymmetricTree parity after DEC-028 fix.

Grid (3 × 3 × 2 = 18 cells):
  N           ∈ {1000, 10000, 50000}
  seed        ∈ {1337, 42, 7}
  random_str  ∈ {0.0, 1.0}

Fixed: SymmetricTree, d=6, 128 bins, LR=0.03, 50 iters, RMSE, 20 features,
       bootstrap_type='No' on CPU / 'no' on MLX (bootstrap ruled out as confound
       in benchmarks/sprint26/d0/bootstrap.py).

G1 acceptance: every cell has MLX_RMSE / CPU_RMSE ∈ [0.98, 1.02] (±2%).
Results written to benchmarks/sprint26/d0/g1-results.md.
"""

import os
import time
import numpy as np
from scipy.stats import pearsonr  # optional; falls back to np.corrcoef if missing

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

FEATURES = 20
BINS = 128
ITERS = 50
DEPTH = 6
LR = 0.03

SIZES = [1000, 10_000, 50_000]
SEEDS = [1337, 42, 7]
RS_VALUES = [0.0, 1.0]


def make_xy(N: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, FEATURES)).astype(np.float32)
    noise = rng.standard_normal(N).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + noise * 0.1).astype(np.float32)
    return X, y


def run_cpu(X, y, seed: int, rs: float):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss_function="RMSE", grow_policy="SymmetricTree", max_bin=BINS,
        random_seed=seed, random_strength=rs,
        bootstrap_type="No",
        verbose=0, thread_count=1,
    )
    t0 = time.perf_counter()
    m.fit(X, y)
    t1 = time.perf_counter()
    rmse = float(m.evals_result_["learn"]["RMSE"][-1])
    preds = m.predict(X)
    return rmse, np.asarray(preds, dtype=np.float64), t1 - t0


def run_mlx(X, y, seed: int, rs: float):
    from catboost_mlx import CatBoostMLXRegressor
    m = CatBoostMLXRegressor(
        iterations=ITERS, depth=DEPTH, learning_rate=LR,
        loss="rmse", grow_policy="SymmetricTree", bins=BINS,
        random_seed=seed, random_strength=rs,
        bootstrap_type="no",
        verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(X, y)
    t1 = time.perf_counter()
    # Use internal RMSE history if populated; fall back to pred-based RMSE
    if m._train_loss_history:
        rmse = float(m._train_loss_history[-1])
    else:
        preds_for_rmse = m.predict(X)
        rmse = float(np.sqrt(((np.asarray(preds_for_rmse, dtype=np.float64) - y) ** 2).mean()))
    preds = m.predict(X)
    return rmse, np.asarray(preds, dtype=np.float64), t1 - t0


def compute_pearson(a, b):
    try:
        r, _ = pearsonr(a, b)
        return float(r)
    except Exception:
        return float(np.corrcoef(a, b)[0, 1])


# ── Run sweep ────────────────────────────────────────────────────────────────

header = (
    f"{'N':>7s} {'seed':>5s} {'rs':>4s} | "
    f"{'CPU_RMSE':>11s} {'MLX_RMSE':>11s} {'delta%':>7s} "
    f"{'ratio':>7s} {'pred_std_R':>10s} {'pearson':>8s} "
    f"{'CPU_t':>7s} {'MLX_t':>7s} | {'gate':>4s}"
)
sep = "-" * len(header)

print(header)
print(sep)

rows = []
overall_pass = True

for N in SIZES:
    for seed in SEEDS:
        X, y = make_xy(N, seed)
        for rs in RS_VALUES:
            cpu_rmse, cpu_preds, cpu_t = run_cpu(X, y, seed, rs)
            mlx_rmse, mlx_preds, mlx_t = run_mlx(X, y, seed, rs)

            ratio = mlx_rmse / cpu_rmse if cpu_rmse > 0 else float("nan")
            delta_pct = (mlx_rmse - cpu_rmse) / cpu_rmse * 100 if cpu_rmse > 0 else float("nan")
            pred_std_ratio = float(np.std(mlx_preds) / np.std(cpu_preds)) if np.std(cpu_preds) > 0 else float("nan")
            pearson_r = compute_pearson(cpu_preds, mlx_preds)
            cell_pass = 0.98 <= ratio <= 1.02

            if not cell_pass:
                overall_pass = False

            gate_str = "PASS" if cell_pass else "FAIL"
            row_str = (
                f"{N:>7d} {seed:>5d} {rs:>4.1f} | "
                f"{cpu_rmse:>11.6f} {mlx_rmse:>11.6f} {delta_pct:>+7.2f}% "
                f"{ratio:>7.4f} {pred_std_ratio:>10.4f} {pearson_r:>8.4f} "
                f"{cpu_t:>7.2f} {mlx_t:>7.2f} | {gate_str}"
            )
            print(row_str)
            rows.append({
                "N": N, "seed": seed, "rs": rs,
                "cpu_rmse": cpu_rmse, "mlx_rmse": mlx_rmse,
                "delta_pct": delta_pct, "ratio": ratio,
                "pred_std_ratio": pred_std_ratio, "pearson": pearson_r,
                "cpu_t": cpu_t, "mlx_t": mlx_t,
                "pass": cell_pass,
            })

print(sep)
gate_verdict = "G1 GATE: PASS — all 18 cells within ±2%" if overall_pass else "G1 GATE: FAIL — see FAIL cells above"
print(gate_verdict)

# ── Write markdown results file ───────────────────────────────────────────────

md_lines = [
    "# S26-D0-7 G1 Parity Sweep Results",
    "",
    "**Branch**: `mlx/sprint-26-python-parity`  ",
    "**Fix**: DEC-028 RandomStrength noise formula (commit 24162e1006)  ",
    "**Gate criterion**: `MLX_RMSE / CPU_RMSE ∈ [0.98, 1.02]` for all 18 cells",
    "",
    "## 18-Cell Results",
    "",
    "| N | seed | rs | CPU RMSE | MLX RMSE | delta% | ratio | pred_std_R | Pearson | CPU_t | MLX_t | Gate |",
    "|---|------|----|----------|----------|--------|-------|-----------|---------|-------|-------|------|",
]

fail_cells = []
for r in rows:
    gate_str = "PASS" if r["pass"] else "**FAIL**"
    md_lines.append(
        f"| {r['N']:,} | {r['seed']} | {r['rs']:.1f} | "
        f"{r['cpu_rmse']:.6f} | {r['mlx_rmse']:.6f} | "
        f"{r['delta_pct']:+.2f}% | {r['ratio']:.4f} | "
        f"{r['pred_std_ratio']:.4f} | {r['pearson']:.4f} | "
        f"{r['cpu_t']:.2f}s | {r['mlx_t']:.2f}s | {gate_str} |"
    )
    if not r["pass"]:
        fail_cells.append(r)

md_lines += [
    "",
    "## Summary",
    "",
]

n_pass = sum(1 for r in rows if r["pass"])
n_fail = len(rows) - n_pass
md_lines.append(f"- **Cells passed**: {n_pass} / 18")
md_lines.append(f"- **Cells failed**: {n_fail} / 18")
md_lines.append("")

if fail_cells:
    md_lines.append("### Failing cells")
    md_lines.append("")
    for r in fail_cells:
        md_lines.append(
            f"- N={r['N']:,}, seed={r['seed']}, rs={r['rs']:.1f}: "
            f"ratio={r['ratio']:.4f} (MLX={r['mlx_rmse']:.6f}, CPU={r['cpu_rmse']:.6f}, delta={r['delta_pct']:+.2f}%)"
        )
    md_lines.append("")

if overall_pass:
    md_lines.append("**G1 GATE: PASS** — all 18 cells within ±2% RMSE ratio.")
else:
    md_lines.append(
        f"**G1 GATE: FAIL** — {n_fail} of 18 cells outside ±2% ratio. "
        "Sprint does not pass G1 as stated."
    )

md_lines += [
    "",
    "## Notes",
    "",
    "- `ratio = MLX_RMSE / CPU_RMSE`; gate window [0.98, 1.02].",
    "- `pred_std_R = std(MLX_preds) / std(CPU_preds)` — measures leaf-magnitude preservation.",
    "  A ratio < 1.0 indicates residual shrinkage (the DEC-028 symptom).",
    "- `Pearson` is Pearson correlation between CPU and MLX predictions on train set.",
    "- CPU uses `bootstrap_type='No'`; MLX uses `bootstrap_type='no'` (same semantic,",
    "  different case convention). Bootstrap ruled out as confound in bootstrap.py.",
    "- `rs=0.0` cells are deterministic within each engine; any delta is parameter or",
    "  binning divergence, not stochastic noise.",
    "- `rs=1.0` cells include RNG divergence (CPU and MLX use different random sequences)",
    "  so larger deltas are expected and acceptable if ratio stays within [0.98, 1.02].",
]

results_path = (
    "/Users/ramos/Library/Mobile Documents/"
    "com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/"
    "benchmarks/sprint26/d0/g1-results.md"
)
with open(results_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")

print(f"\nResults written to benchmarks/sprint26/d0/g1-results.md")
