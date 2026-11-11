"""Train on the complete UCI Adult split with Metal, keeping its test set held out.

Dataset: Becker and Kohavi (1996), https://doi.org/10.24432/C5XW20,
UCI Machine Learning Repository, CC BY 4.0. Expected checksums are the ones used
by CatBoost's own datasets.py. No CatBoost CPU trainer is invoked.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd

from catboost_metal import CatBoostMetalClassifier


NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
         "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
         "hours-per-week", "native-country", "income"]
CATEGORICAL = [1, 3, 5, 6, 7, 8, 9, 13]
CHECKSUMS = {"adult.data": "5d7c39d7b8804f071cdd1f2a7c460872",
             "adult.test": "35238206dfdf7f1fe215bbb874adecdc"}


def load_data(directory):
    directory.mkdir(parents=True, exist_ok=True)
    frames, sources = [], {}
    for name, expected in CHECKSUMS.items():
        path = directory / name
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/" + name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != expected:
                raise ValueError(f"Unexpected dataset checksum: {name}")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != expected:
            raise ValueError(f"Cached dataset checksum mismatch: {path}")
        sources[name] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
        frame = pd.read_csv(path, names=NAMES, header=None, sep=r",\s*", engine="python",
                            na_values=["?"], skiprows=1 if name == "adult.test" else 0)
        frame["income"] = frame.income.str.rstrip(".")
        for index in CATEGORICAL:
            frame[NAMES[index]] = frame[NAMES[index]].fillna("__MISSING__")
        frames.append(frame)
    return *frames, sources


def run(output, *, iterations=500, bootstrap_type="No"):
    from catboost import CatBoostClassifier, CatBoostRegressor
    from catboost.utils import eval_metric
    # Any accidental CPU fit is an error, including through helper modules.
    def forbidden(*args, **kwargs):
        raise RuntimeError("This experiment must train exclusively through Metal.")
    CatBoostClassifier.fit = forbidden
    CatBoostRegressor.fit = forbidden
    output.mkdir(parents=True, exist_ok=True)
    train, test, sources = load_data(output / "data")
    y = (train.income == ">50K").astype(np.float32).to_numpy()
    test_y = (test.income == ">50K").astype(np.float32).to_numpy()
    rng = np.random.default_rng(20260912)
    learn, validation = [], []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        count = len(indices) // 5
        validation.extend(indices[:count])
        learn.extend(indices[count:])
    learn, validation = np.sort(learn), np.sort(validation)
    X, test_X = train.drop(columns="income"), test.drop(columns="income")
    parameters = dict(iterations=iterations, depth=6, learning_rate=.07, border_count=64,
                      leaf_estimation_iterations=5, cat_features=CATEGORICAL,
                      one_hot_max_size=2, ctr_type="Borders", random_seed=2026,
                      bootstrap_type=bootstrap_type, nan_mode="Min")
    model = CatBoostMetalClassifier(**parameters)
    started = time.perf_counter()
    model.fit(X.iloc[learn], y[learn], eval_set=(X.iloc[validation], y[validation]),
              early_stopping_rounds=40, use_best_model=True)
    wall = time.perf_counter() - started
    raw = model.predict(test_X, prediction_type="RawFormulaVal", task_type="METAL")
    standard = model.predict(test_X, prediction_type="RawFormulaVal")
    prediction_error = float(np.max(np.abs(raw - standard)))
    prior = float(np.mean(y[learn]))
    prior_loss = float(-np.mean(test_y * np.log(prior) + (1 - test_y) * np.log1p(-prior)))
    loss = float(np.mean(np.logaddexp(0, raw) - test_y * raw))
    model.save_model(output / "model.cbm")
    model.save_model(output / "model.json", format="json")
    restored = CatBoostClassifier().load_model(str(output / "model.cbm"))
    restored_error = float(np.max(np.abs(restored.predict(test_X, prediction_type="RawFormulaVal") - raw)))
    report = {"dataset": "UCI Adult", "citation": "https://doi.org/10.24432/C5XW20",
              "sources": sources, "learn_rows": len(learn), "validation_rows": len(validation),
              "test_rows": len(test_y), "parameters": parameters, "trees": model.tree_count_,
              "best_iteration": model.best_iteration_, "test_logloss": loss,
              "test_accuracy": float(np.mean((raw > 0) == test_y)),
              "test_auc": float(eval_metric(test_y, raw, "AUC")[0]),
              "training_prior_test_logloss": prior_loss, "fit_wall_seconds": wall,
              "gpu_vs_standard_max_abs_error": prediction_error,
              "cbm_roundtrip_max_abs_error": restored_error, "training_stats": model.training_stats_,
              "scope": "One seeded Metal run; official test split unused for selection; no NVIDIA comparison."}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if loss >= prior_loss or max(prediction_error, restored_error) > 1e-5:
        raise RuntimeError("Adult learning or model-interoperability validation failed.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--bootstrap-type", choices=["No", "Bayesian", "Bernoulli", "Poisson", "MVS"], default="No")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".build" / "adult-full")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, iterations=args.iterations,
                         bootstrap_type=args.bootstrap_type), indent=2))
