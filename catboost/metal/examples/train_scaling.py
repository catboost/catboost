"""Exercise Metal training on larger numeric datasets without CPU fitting.

Run from the checkout with PYTHONPATH=catboost/metal/python. These timings are
local observations, not NVIDIA comparisons. Data generation and held-out model
checks are outside fit_wall_seconds; quantization and export are inside it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from catboost_metal import CatBoostMetalRegressor


def dataset(rows, features, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, features), dtype=np.float32)
    y = (2 * x[:, 0] - x[:, 1] + 1.5 * (x[:, 2] > 0)
         + x[:, 3] * x[:, 4] + .1 * rng.standard_normal(rows)).astype(np.float32)
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".build" / "scaling")
    args = parser.parse_args()
    if args.features < 5 or min(args.rows) < 1:
        parser.error("Use at least five features and positive row counts.")
    from catboost import CatBoostClassifier, CatBoostRegressor

    def forbid_cpu_fit(*_args, **_kwargs):
        raise AssertionError("The scaling example must train exclusively through Metal.")

    CatBoostClassifier.fit = CatBoostRegressor.fit = forbid_cpu_fit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_x, test_y = dataset(16_384, args.features, 901)
    results = []
    for rows in args.rows:
        x, y = dataset(rows, args.features, 900)
        model = CatBoostMetalRegressor(
            iterations=args.iterations, depth=args.depth, learning_rate=.15,
            border_count=32, score_function="Cosine").fit(x, y)
        cpu = model.predict(test_x)
        gpu = model.predict(test_x, task_type="METAL")
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12, atol=1e-12)
        baseline = float(np.sqrt(np.mean((test_y - model.bias_) ** 2)))
        rmse = float(np.sqrt(np.mean((test_y - gpu) ** 2)))
        if not rmse < baseline or not model.loss_history_[-1] < model.loss_history_[0]:
            raise AssertionError("Training failed to improve both learn and held-out loss.")
        record = {
            "rows": rows, "features": args.features, "iterations": args.iterations,
            "depth": args.depth, "test_rows": len(test_y), "test_rmse": rmse,
            "constant_test_rmse": baseline,
            "gpu_vs_standard_max_abs": float(np.max(np.abs(cpu - gpu))),
            "initial_train_rmse": model.loss_history_[0], "final_train_rmse": model.loss_history_[-1],
            "stats": model.training_stats_,
            "timing_scope": "One local M3 run; no NVIDIA comparison; development may run concurrently.",
        }
        results.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        del model, x, y


if __name__ == "__main__":
    main()
