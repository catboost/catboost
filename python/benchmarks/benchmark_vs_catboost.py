#!/usr/bin/env python3
"""
benchmark_vs_catboost.py -- Head-to-head: CatBoost-MLX vs CatBoost CPU.

What this file does:
    Trains CatBoost-MLX and the official CatBoost library on identical
    synthetic regression datasets, then prints a comparison table of
    training time, final training loss, and prediction time.  If the
    ``catboost`` package is not installed the comparison column is omitted
    and only CatBoost-MLX results are shown.

How it fits into the project:
    Standalone script. Run directly::

        python python/benchmarks/benchmark_vs_catboost.py
        python python/benchmarks/benchmark_vs_catboost.py --rows 1000 10000 100000
        python python/benchmarks/benchmark_vs_catboost.py --rows 50000 --features 100 --iterations 300 --depth 8

Key concepts:
    - RMSE: Root Mean Squared Error on a held-out 20 % test split.
    - Training loss: RMSE on the *training* set — matches what both
      libraries minimise during boosting, so it is a direct measure of
      how well each converged.
    - Synthetic data: Reproducibly generated so the script needs no
      external files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Resolve catboost_mlx from the ``python/`` directory regardless of how the
# script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_regression(
    n_samples: int,
    n_features: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) for a linear regression task."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features).astype(np.float32)
    coefs = rng.randn(n_features).astype(np.float32)
    noise = rng.normal(0.0, 0.1, n_samples).astype(np.float32)
    y = (X @ coefs + noise).astype(np.float32)
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions.astype(float) - targets.astype(float)) ** 2)))


def run_catboost_mlx(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    iterations: int,
    depth: int,
) -> Dict[str, float]:
    """Train and evaluate CatBoost-MLX; return timing and loss dict."""
    from catboost_mlx import CatBoostMLXRegressor  # type: ignore[import]

    model = CatBoostMLXRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=0.1,
        l2_reg_lambda=3.0,
        verbose=False,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    train_preds = model.predict(X_train)
    _ = time.perf_counter() - t0  # warm up

    t0 = time.perf_counter()
    test_preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    return {
        "train_time": round(train_time, 3),
        "predict_time": round(predict_time, 4),
        "train_loss": round(_rmse(train_preds, y_train), 5),
        "test_rmse": round(_rmse(test_preds, y_test), 5),
    }


def run_catboost_cpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    iterations: int,
    depth: int,
) -> Optional[Dict[str, float]]:
    """Train and evaluate official CatBoost CPU.

    Returns None if the ``catboost`` package is not installed.
    """
    try:
        import catboost as cb  # type: ignore[import]
    except ImportError:
        return None

    model = cb.CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        verbose=0,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    train_preds = model.predict(X_train)
    _ = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    return {
        "train_time": round(train_time, 3),
        "predict_time": round(predict_time, 4),
        "train_loss": round(_rmse(train_preds, y_train), 5),
        "test_rmse": round(_rmse(test_preds, y_test), 5),
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _divider(col_widths: List[int], char: str = "-") -> str:
    return "+-" + "-+-".join(char * w for w in col_widths) + "-+"


def _row(values: List[str], col_widths: List[int]) -> str:
    cells = [v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(values, col_widths))]
    return "| " + " | ".join(cells) + " |"


def print_table(
    rows: int,
    features: int,
    results_mlx: Dict[str, float],
    results_cpu: Optional[Dict[str, float]],
) -> None:
    """Print a formatted comparison table for one dataset size."""
    headers = ["Metric", "CatBoost-MLX"]
    if results_cpu is not None:
        headers.append("CatBoost CPU")
        headers.append("Speedup")

    metrics = [
        ("Train time (s)",    "train_time",   "{:.3f}"),
        ("Predict time (s)",  "predict_time", "{:.4f}"),
        ("Train loss (RMSE)", "train_loss",   "{:.5f}"),
        ("Test RMSE",         "test_rmse",    "{:.5f}"),
    ]

    data_rows: List[List[str]] = []
    for label, key, fmt in metrics:
        row: List[str] = [label, fmt.format(results_mlx[key])]
        if results_cpu is not None:
            row.append(fmt.format(results_cpu[key]))
            # Speedup only makes sense for time metrics (higher is better for MLX)
            if key in ("train_time", "predict_time"):
                mlx_val = results_mlx[key]
                cpu_val = results_cpu[key]
                if mlx_val > 0:
                    speedup = cpu_val / mlx_val
                    row.append(f"{speedup:.2f}x")
                else:
                    row.append("N/A")
            else:
                row.append("")
        data_rows.append(row)

    # Compute column widths
    all_rows = [headers] + data_rows
    n_cols = len(headers)
    col_widths = [max(len(r[i]) for r in all_rows) for i in range(n_cols)]

    title = f"  {rows:,} rows x {features} features"
    print()
    print(title)
    print(_divider(col_widths))
    print(_row(headers, col_widths))
    print(_divider(col_widths, "="))
    for r in data_rows:
        print(_row(r, col_widths))
    print(_divider(col_widths))


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def run_benchmark(
    row_sizes: List[int],
    n_features: int,
    iterations: int,
    depth: int,
) -> None:
    """Run benchmarks across all requested dataset sizes."""
    catboost_available: Optional[bool] = None  # determined on first iteration

    print()
    print("=" * 60)
    print("  CatBoost-MLX vs CatBoost CPU — Regression Benchmark")
    print("=" * 60)
    print(f"  Iterations : {iterations}")
    print(f"  Depth      : {depth}")
    print(f"  Features   : {n_features}")
    print(f"  Sizes      : {', '.join(f'{n:,}' for n in row_sizes)}")

    for n_rows in row_sizes:
        X_train, y_train, X_test, y_test = make_regression(n_rows, n_features)

        # --- CatBoost-MLX ---
        try:
            results_mlx = run_catboost_mlx(
                X_train, y_train, X_test, y_test, iterations, depth
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ERROR running CatBoost-MLX on {n_rows:,} rows: {exc}")
            continue

        # --- CatBoost CPU ---
        try:
            results_cpu = run_catboost_cpu(
                X_train, y_train, X_test, y_test, iterations, depth
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n  WARNING: CatBoost CPU failed on {n_rows:,} rows: {exc}")
            results_cpu = None

        if catboost_available is None:
            catboost_available = results_cpu is not None
            if not catboost_available:
                print("\n  Note: `catboost` package not found — skipping CPU comparison.")
                print("  Install with: pip install catboost")

        print_table(n_rows, n_features, results_mlx, results_cpu)

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CatBoost-MLX vs CatBoost CPU on synthetic regression data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1_000, 10_000, 100_000],
        metavar="N",
        help="Dataset row counts to benchmark (space-separated).",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=50,
        metavar="F",
        help="Number of input features.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        metavar="T",
        help="Number of boosting iterations.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        metavar="D",
        help="Maximum tree depth.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        row_sizes=args.rows,
        n_features=args.features,
        iterations=args.iterations,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
