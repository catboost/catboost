#!/usr/bin/env python3
"""
S36-LATENT-P11 T1 — empirical drift probe.

Tests the math prediction (math-derivation.md §9) that MLX's score-denominator
divergence vs CPU produces measurable training-loss drift on Logloss/Poisson
(non-trivial-hessian losses), with predicted iter=50 Logloss drift in [5%, 50%]
and iter=1 drift ~ 0%.

Methodology
-----------
For each (anchor, loss, seed):
  1. Run csv_train (MLX) for 100 iterations with --verbose; parse iter=N loss.
  2. Run CatBoostClassifier/Regressor (CPU) for 100 iterations; read
     evals_result_['learn'][metric].
  3. At iter ∈ {1, 10, 50, 100} (1-indexed; zero-indexed positions 0, 9, 49, 99),
     compute drift = |MLX_loss - CPU_loss| / CPU_loss.

Configuration matched between builds:
  depth=6, bins=128, lr=0.03, l2=3, no sample weights,
  bootstrap=No, boost_from_average=True, score_function=L2,
  grow_policy=SymmetricTree, border_type=GreedyLogSum.

Anchors
-------
  1. synthetic-logloss : N=20k, 10 features, y ~ Bernoulli(σ(0.5 x0 + 0.3 x1 + 0.1 ε))
  2. synthetic-poisson : N=20k, 10 features, y ~ Poisson(exp(0.3 x0 + 0.2 x1 - 0.5))
  3. adult              : UCI Adult Income, Logloss only

Usage
-----
  python3 docs/sprint36/p11/scripts/run_probe_f1.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DATA_DIR = REPO / "docs" / "sprint36" / "p11" / "data"
RUNS_DIR = REPO / "docs" / "sprint36" / "p11" / "runs"
MLX_BIN = REPO / "csv_train_p11"  # built from current master via build_csv_train_p11.sh

ITERS_TOTAL = 100
DEPTH = 6
LR = 0.03
L2 = 3
BINS = 128
SEEDS = [42, 1337, 7, 17, 9999]
ITER_CHECKPOINTS = [1, 10, 50, 100]  # 1-indexed (loss after iter N completed)


# ---------------------------------------------------------------------------
# Anchor dataset generators
# ---------------------------------------------------------------------------

def gen_synthetic_logloss(seed: int, out: Path) -> Path:
    """N=20k, 10 features, y ~ Bernoulli(σ(0.5*X[0] + 0.3*X[1] + 0.1*noise))."""
    rng = np.random.default_rng(seed)
    N, D = 20_000, 10
    X = rng.standard_normal((N, D)).astype(np.float32)
    noise = rng.standard_normal(N).astype(np.float32)
    logits = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * noise
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(N) < p).astype(np.int32)
    # csv_train wants target as col 0
    arr = np.column_stack([y.astype(np.float32), X])
    header = ",".join(["y"] + [f"f{i}" for i in range(D)])
    np.savetxt(out, arr, delimiter=",", header=header, comments="", fmt="%.6f")
    return out


def gen_synthetic_poisson(seed: int, out: Path) -> Path:
    """N=20k, 10 features, y ~ Poisson(exp(0.3*X[0] + 0.2*X[1] - 0.5))."""
    rng = np.random.default_rng(seed)
    N, D = 20_000, 10
    X = rng.standard_normal((N, D)).astype(np.float32)
    rate = np.exp(0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.5)
    y = rng.poisson(rate).astype(np.float32)
    arr = np.column_stack([y, X])
    header = ",".join(["y"] + [f"f{i}" for i in range(D)])
    np.savetxt(out, arr, delimiter=",", header=header, comments="", fmt="%.6f")
    return out


def gen_adult(out: Path) -> Path:
    """UCI Adult Income — drop categoricals for fair MLX-vs-CPU comparison
    (csv_train's CTR path differs structurally; numeric-only keeps the test
    isolated to the score-denominator question).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    raw = out.with_suffix(".raw")
    if not raw.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(raw, "wb") as f:
            f.write(r.read())
    rows = []
    with open(raw) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != 15:
                continue
            # numeric cols: 0,2,4,10,11,12 ; target = parts[14]
            try:
                age = float(parts[0])
                fnlwgt = float(parts[2])
                edu_num = float(parts[4])
                cap_gain = float(parts[10])
                cap_loss = float(parts[11])
                hours = float(parts[12])
                y = 1.0 if parts[14] == ">50K" else 0.0
                rows.append([y, age, fnlwgt, edu_num, cap_gain, cap_loss, hours])
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=np.float32)
    header = "y,age,fnlwgt,edu_num,cap_gain,cap_loss,hours"
    np.savetxt(out, arr, delimiter=",", header=header, comments="", fmt="%.6f")
    return out


# ---------------------------------------------------------------------------
# MLX runner (csv_train binary)
# ---------------------------------------------------------------------------

ITER_LINE_RE = re.compile(
    r"iter=(\d+)\s+trees=\d+\s+\S+=\d+\s+loss=([\d.eE+-]+)"
)


def run_mlx(csv: Path, loss: str, seed: int, log_path: Path) -> list[float]:
    """Run csv_train with --verbose and return per-iteration train loss list of length ITERS_TOTAL."""
    if not MLX_BIN.exists():
        raise FileNotFoundError(f"MLX binary not found at {MLX_BIN}")
    cmd = [
        str(MLX_BIN), str(csv),
        "--loss", loss,
        "--iterations", str(ITERS_TOTAL),
        "--depth", str(DEPTH),
        "--lr", str(LR),
        "--l2", str(L2),
        "--bins", str(BINS),
        "--target-col", "0",
        "--seed", str(seed),
        "--score-function", "L2",
        "--verbose",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log_path.write_text(
        "# cmd: " + " ".join(cmd) + "\n"
        + "# returncode: " + str(res.returncode) + "\n"
        + "# stdout:\n" + res.stdout
        + "\n# stderr:\n" + res.stderr
    )
    if res.returncode != 0:
        raise RuntimeError(f"csv_train failed (rc={res.returncode}): {res.stderr[:400]}")
    losses = [None] * ITERS_TOTAL
    for line in res.stdout.splitlines():
        m = ITER_LINE_RE.match(line.strip())
        if m:
            i = int(m.group(1))
            v = float(m.group(2))
            if 0 <= i < ITERS_TOTAL:
                losses[i] = v
    if any(v is None for v in losses):
        missing = [i for i, v in enumerate(losses) if v is None]
        raise RuntimeError(f"MLX output missing iter losses for: {missing[:10]}...")
    return losses


# ---------------------------------------------------------------------------
# CPU runner (catboost package)
# ---------------------------------------------------------------------------

def run_cpu(csv: Path, loss: str, seed: int, log_path: Path) -> list[float]:
    """Train CatBoost CPU with matching hyperparameters; return per-iter learn loss."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    import io
    import contextlib

    arr = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float32)
    y = arr[:, 0]
    X = arr[:, 1:]

    if loss == "logloss":
        y_int = y.astype(int)
        model = CatBoostClassifier(
            iterations=ITERS_TOTAL,
            depth=DEPTH,
            learning_rate=LR,
            l2_leaf_reg=L2,
            border_count=BINS,
            feature_border_type="GreedyLogSum",
            bootstrap_type="No",
            boost_from_average=True,
            loss_function="Logloss",
            score_function="L2",
            grow_policy="SymmetricTree",
            random_seed=seed,
            has_time=True,  # disable random shuffle to match deterministic MLX path
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        target = y_int
    elif loss == "poisson":
        # Poisson does not accept boost_from_average; emulate via explicit
        # baseline = log(mean(y)) to match csv_train's CalcBasePrediction.
        model = CatBoostRegressor(
            iterations=ITERS_TOTAL,
            depth=DEPTH,
            learning_rate=LR,
            l2_leaf_reg=L2,
            border_count=BINS,
            feature_border_type="GreedyLogSum",
            bootstrap_type="No",
            loss_function="Poisson",
            score_function="L2",
            grow_policy="SymmetricTree",
            random_seed=seed,
            has_time=True,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        target = y
    else:
        raise ValueError(f"Unsupported loss for CPU: {loss}")

    buf = io.StringIO()
    fit_kwargs = dict(eval_set=(X, target), use_best_model=False)
    if loss == "poisson":
        avgT = max(float(y.mean()), 1e-6)
        baseline = np.full(len(y), float(np.log(avgT)), dtype=np.float64)
        fit_kwargs["baseline"] = baseline
        # Also need baseline for eval_set
        fit_kwargs["eval_set"] = Pool(X, target, baseline=baseline)
    with contextlib.redirect_stdout(buf):
        model.fit(X, target, **fit_kwargs)
    log_path.write_text(buf.getvalue())
    history = model.evals_result_["learn"]
    metric_key = "Logloss" if loss == "logloss" else "Poisson"
    losses = list(history[metric_key])
    if len(losses) != ITERS_TOTAL:
        raise RuntimeError(
            f"CPU history length {len(losses)} != {ITERS_TOTAL} (early stop?)")
    return losses


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def drift(mlx_v: float, cpu_v: float) -> float:
    """Relative drift |MLX - CPU| / CPU."""
    if cpu_v == 0:
        return float("inf")
    return abs(mlx_v - cpu_v) / abs(cpu_v)


def run_anchor(name: str, csv: Path, loss: str, seed: int) -> dict:
    log_dir = RUNS_DIR / f"{name}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [run] anchor={name} loss={loss} seed={seed}", flush=True)
    mlx_losses = run_mlx(csv, loss, seed, log_dir / "mlx.log")
    cpu_losses = run_cpu(csv, loss, seed, log_dir / "cpu.log")
    rec = {
        "anchor": name,
        "loss": loss,
        "seed": seed,
        "csv": str(csv.name),
        "mlx_curve": mlx_losses,
        "cpu_curve": cpu_losses,
        "checkpoints": {},
    }
    for it in ITER_CHECKPOINTS:
        i = it - 1
        m = mlx_losses[i]
        c = cpu_losses[i]
        rec["checkpoints"][str(it)] = {
            "mlx": m,
            "cpu": c,
            "drift_rel": drift(m, c),
        }
    return rec


def cpu_consistency_check(records: list[dict]) -> dict:
    """Verify CPU does not disagree with itself across seeds (>2% iter=50 spread on synthetic)."""
    out = {}
    for loss in ("logloss", "poisson"):
        synth_recs = [
            r for r in records
            if r["anchor"] == f"synthetic-{loss}"
        ]
        if len(synth_recs) < 2:
            continue
        cpu50 = [r["checkpoints"]["50"]["cpu"] for r in synth_recs]
        mean = sum(cpu50) / len(cpu50)
        spread = (max(cpu50) - min(cpu50)) / abs(mean) if mean else 0.0
        out[loss] = {"cpu_iter50_values": cpu50, "rel_spread": spread}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="run only seed=42 + skip Adult (smoke)")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [42] if args.quick else SEEDS

    print(f"[probe-f1] preparing anchor datasets", flush=True)
    records = []

    print(f"[probe-f1] anchor: synthetic-logloss (seeds {seeds})", flush=True)
    for s in seeds:
        csv = DATA_DIR / f"synthetic-logloss-seed{s}.csv"
        if not csv.exists():
            gen_synthetic_logloss(s, csv)
        records.append(run_anchor("synthetic-logloss", csv, "logloss", s))

    print(f"[probe-f1] anchor: synthetic-poisson (seeds {seeds})", flush=True)
    for s in seeds:
        csv = DATA_DIR / f"synthetic-poisson-seed{s}.csv"
        if not csv.exists():
            gen_synthetic_poisson(s, csv)
        records.append(run_anchor("synthetic-poisson", csv, "poisson", s))

    if not args.quick:
        print(f"[probe-f1] anchor: adult (seed=42)", flush=True)
        csv = DATA_DIR / "adult.csv"
        if not csv.exists():
            gen_adult(csv)
        records.append(run_anchor("adult", csv, "logloss", 42))

    consistency = cpu_consistency_check(records)
    print(f"[probe-f1] CPU self-consistency: {consistency}", flush=True)

    out_json = REPO / "docs" / "sprint36" / "p11" / "probe-f1-results.json"
    out_json.write_text(json.dumps(
        {"records": records, "cpu_consistency": consistency},
        indent=2,
    ))
    print(f"[probe-f1] wrote {out_json}", flush=True)

    # Print compact summary table
    print("\n=== Drift table (MLX vs CPU CatBoost) ===")
    print(f"{'anchor':<24} {'loss':<10} {'seed':>5} "
          f"{'iter=1':>10} {'iter=10':>10} {'iter=50':>10} {'iter=100':>10}")
    for r in records:
        cps = r["checkpoints"]
        print(f"{r['anchor']:<24} {r['loss']:<10} {r['seed']:>5} "
              f"{cps['1']['drift_rel']*100:>9.3f}% "
              f"{cps['10']['drift_rel']*100:>9.3f}% "
              f"{cps['50']['drift_rel']*100:>9.3f}% "
              f"{cps['100']['drift_rel']*100:>9.3f}%")


if __name__ == "__main__":
    main()
