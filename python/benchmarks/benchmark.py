#!/usr/bin/env python3
"""
benchmark.py -- Speed and accuracy comparison against other GBDT frameworks.

What this file does:
    Runs a fair competition between CatBoost-MLX and other popular gradient
    boosting libraries (XGBoost, LightGBM, CatBoost-official). It generates
    synthetic datasets at various sizes, trains each framework with comparable
    hyperparameters, and reports training time, prediction time, and accuracy.
    Frameworks not installed are automatically skipped.

How it fits into the project:
    Standalone script. Imports catboost_mlx and optionally xgboost, lightgbm,
    catboost. Run directly: ``python benchmark.py``.

Key concepts:
    - RMSE: Root Mean Squared Error -- measures prediction error for regression.
    - Accuracy: Fraction of correct predictions for classification.
    - Synthetic data: Randomly generated datasets so benchmarks are reproducible
      without needing external data files.

Usage:
  python benchmark.py                    # run all benchmarks
  python benchmark.py --sizes 1000 10000 # custom dataset sizes
  python benchmark.py --output results.json  # save results to JSON
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path so we can import catboost_mlx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Dataset Generation ─────────────────────────────────────────────────────

def make_regression(n_samples, n_features=50, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    coefs = rng.randn(n_features)
    y = X @ coefs + rng.normal(0, 0.1, n_samples)
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def make_binary(n_samples, n_features=50, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    logits = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.3, n_samples)
    y = (logits > 0).astype(float)
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def make_multiclass(n_samples, n_features=50, n_classes=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    y = (X[:, :3].sum(axis=1) * n_classes / 3).astype(int).clip(0, n_classes - 1).astype(float)
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


# ── Framework Runners ──────────────────────────────────────────────────────

def run_catboost_mlx(X_train, y_train, X_test, y_test, task, params):
    from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor

    if task == "regression":
        model = CatBoostMLXRegressor(**params)
    else:
        model = CatBoostMLXClassifier(**params)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    if task == "regression":
        metric = np.sqrt(np.mean((preds - y_test) ** 2))
        metric_name = "RMSE"
    else:
        metric = np.mean(preds == y_test)
        metric_name = "Accuracy"

    return {
        "train_time": round(train_time, 3),
        "predict_time": round(predict_time, 3),
        "metric": round(float(metric), 4),
        "metric_name": metric_name,
    }


def run_xgboost(X_train, y_train, X_test, y_test, task, params):
    try:
        import xgboost as xgb
    except ImportError:
        return None

    xgb_params = {
        "n_estimators": params.get("iterations", 100),
        "max_depth": params.get("depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "reg_lambda": params.get("l2_reg_lambda", 3.0),
        "tree_method": "hist",
        "verbosity": 0,
    }

    if task == "regression":
        model = xgb.XGBRegressor(**xgb_params)
    elif task == "binary":
        model = xgb.XGBClassifier(**xgb_params, eval_metric="logloss")
    else:
        model = xgb.XGBClassifier(**xgb_params, eval_metric="mlogloss")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    if task == "regression":
        metric = np.sqrt(np.mean((preds - y_test) ** 2))
        metric_name = "RMSE"
    else:
        metric = np.mean(preds == y_test)
        metric_name = "Accuracy"

    return {
        "train_time": round(train_time, 3),
        "predict_time": round(predict_time, 3),
        "metric": round(float(metric), 4),
        "metric_name": metric_name,
    }


def run_lightgbm(X_train, y_train, X_test, y_test, task, params):
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    lgb_params = {
        "n_estimators": params.get("iterations", 100),
        "max_depth": params.get("depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "reg_lambda": params.get("l2_reg_lambda", 3.0),
        "verbose": -1,
    }

    if task == "regression":
        model = lgb.LGBMRegressor(**lgb_params)
    else:
        model = lgb.LGBMClassifier(**lgb_params)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    if task == "regression":
        metric = np.sqrt(np.mean((preds - y_test) ** 2))
        metric_name = "RMSE"
    else:
        metric = np.mean(preds == y_test)
        metric_name = "Accuracy"

    return {
        "train_time": round(train_time, 3),
        "predict_time": round(predict_time, 3),
        "metric": round(float(metric), 4),
        "metric_name": metric_name,
    }


def run_catboost_gpu(X_train, y_train, X_test, y_test, task, params):
    try:
        import catboost as cb
    except ImportError:
        return None

    cb_params = {
        "iterations": params.get("iterations", 100),
        "depth": params.get("depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "l2_leaf_reg": params.get("l2_reg_lambda", 3.0),
        "verbose": 0,
    }

    if task == "regression":
        model = cb.CatBoostRegressor(**cb_params)
    else:
        model = cb.CatBoostClassifier(**cb_params)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    if task == "regression":
        metric = np.sqrt(np.mean((preds - y_test) ** 2))
        metric_name = "RMSE"
    else:
        preds = preds.astype(float) if task == "binary" else preds
        metric = np.mean(preds == y_test)
        metric_name = "Accuracy"

    return {
        "train_time": round(train_time, 3),
        "predict_time": round(predict_time, 3),
        "metric": round(float(metric), 4),
        "metric_name": metric_name,
    }


# ── Main Benchmark Runner ─────────────────────────────────────────────────

FRAMEWORKS = [
    ("CatBoost-MLX", run_catboost_mlx),
    ("XGBoost", run_xgboost),
    ("LightGBM", run_lightgbm),
    ("CatBoost", run_catboost_gpu),
]

TASKS = [
    ("regression", make_regression),
    ("binary", make_binary),
    ("multiclass", make_multiclass),
]


def run_benchmark(sizes, iterations=200, depth=6, learning_rate=0.1):
    params = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_reg_lambda": 3.0,
    }

    all_results = []

    for task_name, data_fn in TASKS:
        for n in sizes:
            print(f"\n{'=' * 60}")
            print(f"  {task_name.upper()} — {n:,} samples")
            print(f"{'=' * 60}")

            X_train, y_train, X_test, y_test = data_fn(n)

            header = f"  {'Framework':<16} {'Train(s)':>9} {'Pred(s)':>9} {'Metric':>10}"
            print(header)
            print(f"  {'-' * 50}")

            for fw_name, fw_fn in FRAMEWORKS:
                try:
                    result = fw_fn(X_train, y_train, X_test, y_test, task_name, params)
                except Exception as e:
                    result = None
                    print(f"  {fw_name:<16} ERROR: {e}")

                if result is None:
                    print(f"  {fw_name:<16} (not installed)")
                    continue

                print(f"  {fw_name:<16} {result['train_time']:>9.3f} "
                      f"{result['predict_time']:>9.3f} "
                      f"{result['metric']:>10.4f}")

                all_results.append({
                    "task": task_name,
                    "n_samples": n,
                    "framework": fw_name,
                    **result,
                })

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CatBoost-MLX Benchmarks")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000],
                        help="Dataset sizes to benchmark")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of boosting iterations")
    parser.add_argument("--depth", type=int, default=6,
                        help="Tree depth")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("CatBoost-MLX Benchmark Suite")
    print(f"Iterations: {args.iterations}, Depth: {args.depth}")
    print(f"Dataset sizes: {args.sizes}")

    results = run_benchmark(
        sizes=args.sizes,
        iterations=args.iterations,
        depth=args.depth,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"\nTotal benchmarks: {len(results)}")


if __name__ == "__main__":
    main()
