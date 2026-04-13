#!/usr/bin/env python3
"""
bench_mlx_vs_cpu.py — CatBoost-MLX (Metal GPU) vs CatBoost CPU benchmark.

Trains both backends across three dataset scales and three loss functions,
measuring wall-clock training time, iterations/second, and final train loss.
Outputs a markdown table to stdout (or --output file).

Usage
-----
    python benchmarks/bench_mlx_vs_cpu.py
    python benchmarks/bench_mlx_vs_cpu.py --hardware "M2 Pro 16GB" --iterations 200
    python benchmarks/bench_mlx_vs_cpu.py --scales 10000,100000 --output results.md
"""

import argparse
import datetime
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ── Backend availability ──────────────────────────────────────────────────────

try:
    from catboost import CatBoostClassifier as CBClassifier
    from catboost import CatBoostRegressor as CBRegressor
    _HAS_CATBOOST_CPU = True
except ImportError:
    _HAS_CATBOOST_CPU = False
    print(
        "WARNING: catboost is not installed — CPU benchmarks will be skipped.\n"
        "         Install with: pip install catboost",
        file=sys.stderr,
    )

try:
    from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor
    _HAS_CATBOOST_MLX = True
except ImportError:
    _HAS_CATBOOST_MLX = False
    print(
        "WARNING: catboost_mlx is not installed — MLX benchmarks will be skipped.\n"
        "         Build from source: see catboost/mlx/README.md",
        file=sys.stderr,
    )

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    dataset_label: str    # e.g. "10k x 50"
    loss: str             # "RMSE", "Logloss", "MultiClass"
    backend: str          # "CPU" or "MLX"
    elapsed_s: float      # wall-clock seconds
    iter_per_s: float     # iterations / elapsed_s
    train_loss: float     # final train loss value (last iteration)


# ── Dataset generation ────────────────────────────────────────────────────────

def _make_dataset(
    n_rows: int,
    n_cols: int,
    loss: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic (X, y) pair appropriate for the given loss function.

    RMSE       — continuous targets drawn from a linear model + Gaussian noise.
    Logloss    — binary {0, 1} targets; class balance is ~50 / 50.
    MultiClass — 3-class integer targets {0, 1, 2}; balanced across classes.

    All feature columns are i.i.d. standard normal.
    """
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    coeff = rng.standard_normal(n_cols).astype(np.float32)

    if loss == "RMSE":
        y = X @ coeff + rng.standard_normal(n_rows).astype(np.float32)

    elif loss == "Logloss":
        logit = X @ coeff
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = rng.binomial(1, prob).astype(np.float32)

    elif loss == "MultiClass":
        # Produce 3 class scores and assign argmax label
        W = rng.standard_normal((n_cols, 3)).astype(np.float32)
        scores = X @ W
        y = np.argmax(scores, axis=1).astype(np.float32)

    else:
        raise ValueError(f"Unknown loss: {loss}")

    return X, y


# ── Warm-up ───────────────────────────────────────────────────────────────────

def _warmup(backend: str, loss: str, iterations: int, rng: np.random.Generator) -> None:
    """Run a tiny training job to absorb JIT / Metal compilation overhead.

    Both backends have a first-call penalty:
    - CatBoost CPU: shared library initialisation and thread pool spin-up.
    - CatBoost-MLX: Metal kernel compilation (cached after first call within
      a process, but not across processes on a cold cache).

    Using 200 rows x 10 features keeps warm-up under a few seconds even on
    a cold Metal cache.
    """
    X_warm, y_warm = _make_dataset(200, 10, loss, rng)
    kwargs_common = dict(iterations=min(iterations, 20), depth=4, learning_rate=0.1)

    if backend == "CPU" and _HAS_CATBOOST_CPU:
        if loss == "RMSE":
            m = CBRegressor(**kwargs_common, verbose=0, random_seed=42)
        elif loss == "Logloss":
            m = CBClassifier(
                **kwargs_common, loss_function="Logloss", verbose=0, random_seed=42
            )
        else:
            m = CBClassifier(
                **kwargs_common, loss_function="MultiClass", verbose=0, random_seed=42
            )
        m.fit(X_warm, y_warm)

    elif backend == "MLX" and _HAS_CATBOOST_MLX:
        mlx_loss = {"RMSE": "rmse", "Logloss": "logloss", "MultiClass": "multiclass"}[loss]
        if loss == "RMSE":
            m = CatBoostMLXRegressor(
                **kwargs_common, loss=mlx_loss, random_seed=42, verbose=False
            )
        else:
            m = CatBoostMLXClassifier(
                **kwargs_common, loss=mlx_loss, random_seed=42, verbose=False
            )
        m.fit(X_warm, y_warm)


# ── Final-loss extraction helpers ─────────────────────────────────────────────

def _cpu_final_loss(model, loss: str) -> float:
    """Extract final train-loss value from a fitted CatBoost CPU model.

    CatBoost CPU exposes per-iteration metrics via evals_result_ once the
    model is fitted (no eval_set needed — learn metrics are always recorded).
    The dict structure is: {'learn': {'MetricName': [v0, v1, ..., vN]}}.
    """
    evals = model.evals_result_
    learn = evals.get("learn", {})
    if not learn:
        return float("nan")
    # Take the last value of whichever metric is present
    for val_list in learn.values():
        if val_list:
            return float(val_list[-1])
    return float("nan")


def _mlx_final_loss(model) -> float:
    """Extract final train-loss from a fitted CatBoostMLX model.

    CatBoostMLX exposes the per-iteration train loss via the
    `train_loss_history` property (list of floats, one per iteration).
    """
    history = model.train_loss_history
    if history:
        return float(history[-1])
    return float("nan")


# ── Per-benchmark runner ──────────────────────────────────────────────────────

def _run_one(
    n_rows: int,
    n_cols: int,
    loss: str,
    backend: str,
    iterations: int,
    rng: np.random.Generator,
) -> Optional[BenchResult]:
    """Train one (scale, loss, backend) combination and return a BenchResult.

    Returns None if the required backend is unavailable.
    """
    dataset_label = f"{n_rows // 1000}k x {n_cols}"

    if backend == "CPU" and not _HAS_CATBOOST_CPU:
        return None
    if backend == "MLX" and not _HAS_CATBOOST_MLX:
        return None

    X, y = _make_dataset(n_rows, n_cols, loss, rng)

    # ---- Build model ----
    common = dict(iterations=iterations, depth=6, learning_rate=0.1)

    if backend == "CPU":
        if loss == "RMSE":
            model = CBRegressor(
                **common,
                l2_leaf_reg=3.0,
                verbose=0,
                random_seed=42,
            )
        elif loss == "Logloss":
            model = CBClassifier(
                **common,
                loss_function="Logloss",
                l2_leaf_reg=3.0,
                verbose=0,
                random_seed=42,
            )
        else:  # MultiClass
            model = CBClassifier(
                **common,
                loss_function="MultiClass",
                l2_leaf_reg=3.0,
                verbose=0,
                random_seed=42,
            )

    else:  # MLX
        mlx_loss = {"RMSE": "rmse", "Logloss": "logloss", "MultiClass": "multiclass"}[loss]
        if loss == "RMSE":
            model = CatBoostMLXRegressor(
                **common,
                loss=mlx_loss,
                l2_reg_lambda=3.0,
                random_seed=42,
                verbose=False,
            )
        else:
            model = CatBoostMLXClassifier(
                **common,
                loss=mlx_loss,
                l2_reg_lambda=3.0,
                random_seed=42,
                verbose=False,
            )

    # ---- Time the fit ----
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0

    # ---- Collect metrics ----
    train_loss = _cpu_final_loss(model, loss) if backend == "CPU" else _mlx_final_loss(model)

    return BenchResult(
        dataset_label=dataset_label,
        loss=loss,
        backend=backend,
        elapsed_s=elapsed,
        iter_per_s=iterations / elapsed,
        train_loss=train_loss,
    )


# ── Markdown table formatting ─────────────────────────────────────────────────

def _format_table(results: List[BenchResult]) -> str:
    """Render benchmark results as a GitHub-flavoured markdown table."""
    header = (
        "| Dataset   | Loss       | Backend | Time (s) | Iter/s  | Train Loss |\n"
        "|-----------|------------|---------|----------|---------|------------|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r.dataset_label:<9} "
            f"| {r.loss:<10} "
            f"| {r.backend:<7} "
            f"| {r.elapsed_s:>8.2f} "
            f"| {r.iter_per_s:>7.1f} "
            f"| {r.train_loss:>10.4f} |"
        )
    return header + "\n".join(rows) + "\n"


def _format_header(hardware: str, iterations: int) -> str:
    """Format a human-readable header section for the benchmark output."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_version = sys.version.split()[0]

    cb_version = "n/a"
    if _HAS_CATBOOST_CPU:
        try:
            import catboost
            cb_version = catboost.__version__
        except Exception:
            pass

    mlx_version = "n/a"
    if _HAS_CATBOOST_MLX:
        try:
            import catboost_mlx
            mlx_version = catboost_mlx.__version__
        except Exception:
            pass

    lines = [
        "# CatBoost-MLX vs CatBoost CPU Benchmark",
        "",
        f"- **Date**: {now}",
        f"- **Hardware**: {hardware}",
        f"- **Python**: {py_version}",
        f"- **catboost**: {cb_version}",
        f"- **catboost_mlx**: {mlx_version}",
        f"- **Iterations**: {iterations}",
        f"- **Hyperparameters**: depth=6, learning_rate=0.1, l2_leaf_reg=3.0, random_seed=42",
        "",
    ]
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CatBoost-MLX (Metal GPU) vs CatBoost CPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hardware",
        default="unknown",
        help='Hardware tag included in the output header (e.g. "M2 Pro 16GB").',
    )
    parser.add_argument(
        "--scales",
        default="10000,100000,500000",
        help=(
            "Comma-separated list of row counts to benchmark. "
            "Default: 10000,100000,500000"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of boosting iterations for every model. Default: 100",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write output to this file path instead of stdout.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Parse --scales
    try:
        scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    except ValueError as exc:
        print(f"ERROR: --scales must be comma-separated integers: {exc}", file=sys.stderr)
        sys.exit(1)

    if not scales:
        print("ERROR: --scales produced an empty list.", file=sys.stderr)
        sys.exit(1)

    n_features = 50
    losses = ["RMSE", "Logloss", "MultiClass"]
    backends = ["CPU", "MLX"]
    iterations = args.iterations

    if not _HAS_CATBOOST_CPU and not _HAS_CATBOOST_MLX:
        print(
            "ERROR: Neither catboost nor catboost_mlx is installed. "
            "Nothing to benchmark.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Deterministic RNG for all dataset generation — same X/y for every run.
    rng = np.random.default_rng(42)

    # ---- Warm-up pass ----
    print("Warming up backends...", file=sys.stderr)
    for backend in backends:
        for loss in losses:
            _warmup(backend, loss, iterations, np.random.default_rng(0))
    print("Warm-up complete.\n", file=sys.stderr)

    # ---- Benchmark runs ----
    results: List[BenchResult] = []
    total_runs = len(scales) * len(losses) * len(backends)
    run_num = 0

    for n_rows in scales:
        for loss in losses:
            for backend in backends:
                run_num += 1
                label = f"{n_rows // 1000}k x {n_features}"
                print(
                    f"[{run_num}/{total_runs}] {label}  loss={loss}  backend={backend} ...",
                    end="  ",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    result = _run_one(
                        n_rows=n_rows,
                        n_cols=n_features,
                        loss=loss,
                        backend=backend,
                        iterations=iterations,
                        rng=rng,
                    )
                except Exception as exc:
                    print(f"FAILED: {exc}", file=sys.stderr)
                    continue

                if result is None:
                    print("SKIPPED (backend unavailable)", file=sys.stderr)
                    continue

                results.append(result)
                print(f"{result.elapsed_s:.2f}s", file=sys.stderr)

    if not results:
        print("No benchmark results collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ---- Format output ----
    output_text = _format_header(args.hardware, iterations) + _format_table(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"\nResults written to: {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
