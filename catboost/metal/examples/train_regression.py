"""Exercise translated CUDA training on Metal and save a standard CatBoost model."""

import argparse
import json
from pathlib import Path

import numpy as np

from catboost_metal import CatBoostMetalRegressor, device_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--score-function", choices=("L2", "Cosine"), default="Cosine")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".build" / "example")
    args = parser.parse_args()
    if args.rows < 32:
        parser.error("--rows must be at least 32")
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(args.rows, 8)).astype(np.float32)
    y = (2 * X[:, 0] - X[:, 1] + 0.75 * (X[:, 2] > 0)
         + X[:, 3] * X[:, 3] + rng.normal(scale=0.05, size=args.rows)).astype(np.float32)
    split = 3 * args.rows // 4
    # Build the host library before timing. First-fit timing still includes
    # Metal shader/pipeline initialization as well as training setup/execution.
    device = device_info()
    model = CatBoostMetalRegressor(iterations=args.iterations, depth=4, border_count=32,
                                  learning_rate=0.1, score_function=args.score_function)
    model.fit(X[:split], y[:split])
    prediction = model.predict(X[split:])
    initial_rmse = float(np.sqrt(np.mean((y[split:] - model.bias_) ** 2)))
    test_rmse = float(np.sqrt(np.mean((y[split:] - prediction) ** 2)))
    gpu_model_difference = float(np.max(np.abs(
        model.predict(X[:split]) - model.training_predictions_)))
    report = {
        "device": device["name"], "score_function": args.score_function,
        "train_rows": split, "test_rows": args.rows - split,
        "features": X.shape[1], "trees": model.tree_count_,
        "initial_train_rmse": model.loss_history_[0],
        "final_train_rmse": model.loss_history_[-1],
        "initial_test_rmse": initial_rmse, "final_test_rmse": test_rmse,
        "gpu_vs_exported_predictions_max_abs_error": gpu_model_difference,
        **model.training_stats_,
        "note": "Synthetic smoke experiment; no CUDA device or speedup comparison.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output_dir / "model.cbm")
    model.save_model(args.output_dir / "model.json", format="json")
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved model and results to {args.output_dir}")
    if test_rmse >= initial_rmse or gpu_model_difference > 1e-3:
        raise SystemExit("Metal example failed the learning/model interoperability check")


if __name__ == "__main__":
    main()
