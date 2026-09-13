"""Train on repository datasets using Metal and verify standard-model inference.

The validation split comes from the training fixture. The separate test fixture
is used only after fitting. These small fixtures are interoperability and
learning checks; they do not establish CUDA quality or performance parity.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "pytest" / "data"


def _load_fixture(directory, filename, description):
    """Read the fixture's actual column description, excluding Auxiliary data."""
    source = DATA / directory / filename
    cd = DATA / directory / description
    roles = {}
    for line in cd.read_text().splitlines():
        if line.strip():
            column, role, *_ = line.split("\t")
            roles[int(column)] = role
    rows = np.loadtxt(source, dtype=str, delimiter="\t", ndmin=2)
    target_columns = [index for index, role in roles.items() if role == "Target"]
    if len(target_columns) != 1:
        raise ValueError("Dataset example requires exactly one target column.")
    columns = [index for index in range(rows.shape[1])
               if roles.get(index, "Num") in ("Num", "Categ")]
    cats = [feature for feature, column in enumerate(columns) if roles.get(column) == "Categ"]
    features = rows[:, columns].astype(object) if cats else rows[:, columns].astype(np.float32)
    if cats:
        for feature in range(len(columns)):
            if feature not in cats:
                features[:, feature] = features[:, feature].astype(np.float32)
    targets = rows[:, target_columns[0]].astype(np.float32)
    return features, targets, cats, {
        "data": str(source.relative_to(ROOT.parent)),
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "column_description": str(cd.relative_to(ROOT.parent)),
        "column_description_sha256": hashlib.sha256(cd.read_bytes()).hexdigest(),
    }


def _split(targets, classifier):
    rng = np.random.default_rng(20260912)
    groups = [np.flatnonzero(targets == value) for value in np.unique(targets)] if classifier else [np.arange(len(targets))]
    train, validation = [], []
    for group in groups:
        shuffled = rng.permutation(group)
        count = max(1, len(group) // 5)
        validation.extend(shuffled[:count])
        train.extend(shuffled[count:])
    return np.asarray(sorted(train)), np.asarray(sorted(validation))


def _metrics(targets, raw, classifier):
    if classifier:
        return {"logloss": float(np.mean(np.logaddexp(0.0, raw) - targets * raw)),
                "accuracy": float(np.mean((raw > 0.0) == targets))}
    return {"rmse": float(np.sqrt(np.mean((targets - raw) ** 2)))}


def _run_case(name, classifier, directory, train_file, test_file, cd, output, iterations, **options):
    from catboost import CatBoostClassifier, CatBoostRegressor

    X, y, cats, train_source = _load_fixture(directory, train_file, cd)
    test_X, test_y, test_cats, test_source = _load_fixture(directory, test_file, cd)
    if cats != test_cats:
        raise ValueError("Training and test categorical columns do not match.")
    train, validation = _split(y, classifier)
    estimator_type = CatBoostMetalClassifier if classifier else CatBoostMetalRegressor
    options = {"iterations": iterations, "depth": 3, "learning_rate": .1,
               "border_count": 16, "l2_leaf_reg": 3.0, "cat_features": cats,
               "leaf_estimation_iterations": 3 if classifier else 1, **options}
    model = estimator_type(**options)
    start = time.perf_counter()
    model.fit(X[train], y[train], eval_set=(X[validation], y[validation]),
              early_stopping_rounds=20, use_best_model=True)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    raw = model.predict(test_X, prediction_type="RawFormulaVal", task_type="METAL")
    predict_seconds = time.perf_counter() - start
    standard_raw = model.predict(test_X, prediction_type="RawFormulaVal", task_type="CPU")
    model_difference = float(np.max(np.abs(raw - standard_raw)))
    case_output = output / name
    case_output.mkdir(parents=True, exist_ok=True)
    roundtrip_difference = {}
    standard_type = CatBoostClassifier if classifier else CatBoostRegressor
    for format in ("cbm", "json"):
        path = case_output / f"model.{format}"
        model.save_model(path, format=format)
        restored = standard_type().load_model(str(path), format=format)
        restored_raw = restored.predict(test_X, prediction_type="RawFormulaVal")
        roundtrip_difference[format] = float(np.max(np.abs(restored_raw - raw)))
    json_model = json.loads((case_output / "model.json").read_text())
    from catboost_metal._inference import predict_model_json
    independent_raw = predict_model_json(json_model, test_X)
    independent_difference = float(np.max(np.abs(independent_raw - raw)))
    split_types = {}
    for tree in json_model["oblivious_trees"]:
        for split in tree.get("splits") or []:
            kind = split["split_type"]
            split_types[kind] = split_types.get(kind, 0) + 1
    mean = float(np.mean(y[train], dtype=np.float64))
    prior = np.log(mean / (1 - mean)) if classifier else mean
    heldout = _metrics(test_y, raw, classifier)
    prior_metrics = _metrics(test_y, np.full(test_y.shape, prior), classifier)
    metric = "logloss" if classifier else "rmse"
    if heldout[metric] >= prior_metrics[metric]:
        raise RuntimeError(f"{name} did not improve held-out {metric} over its training-target prior.")
    if max(model_difference, independent_difference, *roundtrip_difference.values()) > 1e-5:
        raise RuntimeError(f"{name} failed standard-model prediction interoperability.")
    report = {
        "name": name, "training_source": train_source, "test_source": test_source,
        "source_train_rows": len(y), "learn_rows": len(train),
        "validation_rows": len(validation), "test_rows": len(test_y),
        "features": X.shape[1], "categorical_features": cats,
        "parameters": options, "tree_count": model.tree_count_,
        "best_iteration": model.get_best_iteration(), "split_types": split_types,
        "heldout": heldout, "training_prior_on_heldout": prior_metrics,
        "fit_wall_seconds": fit_seconds, "heldout_metal_predict_wall_seconds": predict_seconds,
        "gpu_vs_standard_prediction_max_abs_error": model_difference,
        "gpu_vs_independent_json_prediction_max_abs_error": independent_difference,
        "roundtrip_prediction_max_abs_error": roundtrip_difference,
        "training_stats": model.training_stats_,
    }
    (case_output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_datasets(output_dir, iterations=150):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _run_case("adult_onehot", True, "adult", "train_small", "test_small", "train.cd",
                  output_dir, iterations, one_hot_max_size=255),
        _run_case("adult_borders_ctr", True, "adult", "train_small", "test_small", "train.cd",
                  output_dir, iterations, one_hot_max_size=2, ctr_type="Borders", random_seed=2026),
        _run_case("numeric_regression", False, "multiregression", "train", "test", "train_single.cd",
                  output_dir, iterations),
    ]
    report = {"cases": cases,
        "validation": "Seeded 20 percent split of training fixtures; test fixtures are not used for early stopping.",
        "timing": "Fit wall time includes preparation and first-use compilation; inference timing includes quantization and transfers.",
        "scope": "Small repository fixture checks, with no CatBoost CPU training or NVIDIA CUDA comparison."}
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".build" / "datasets")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10000:
        parser.error("--iterations must be in [1, 10000]")
    print(json.dumps(run_datasets(args.output_dir, args.iterations), indent=2))


if __name__ == "__main__":
    main()
