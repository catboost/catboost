"""Fit both Metal multiclass objectives on the repository's Iris fixture."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from catboost_metal import CatBoostMetalClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".build" / "iris")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / "dotnet/CatBoostNetTests/testbed/iris/iris.data"
    rows = [row.split(",") for row in source.read_text().splitlines() if row.strip()]
    features = np.asarray([row[:4] for row in rows], np.float32)
    labels = np.asarray([row[4] for row in rows])
    rng = np.random.default_rng(2026)
    learn, validation, test = [], [], []
    for label in np.unique(labels):
        order = rng.permutation(np.flatnonzero(labels == label))
        test.extend(order[:10]); validation.extend(order[10:20]); learn.extend(order[20:])
    learn, validation, test = map(np.asarray, (learn, validation, test))

    def forbid_cpu_fit(*_args, **_kwargs):
        raise AssertionError("The Iris example must train exclusively through Metal.")

    CatBoostClassifier.fit = CatBoostRegressor.fit = forbid_cpu_fit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for objective in ("MultiClass", "MultiClassOneVsAll"):
        model = CatBoostMetalClassifier(
            loss_function=objective, iterations=200, depth=3, learning_rate=.1,
            border_count=16).fit(features[learn], labels[learn],
                eval_set=(features[validation], labels[validation]), early_stopping_rounds=20)
        probabilities = model.predict_proba(features[test], task_type="METAL")
        np.testing.assert_allclose(probabilities, model.predict_proba(features[test]), atol=1e-12, rtol=1e-12)
        predictions = model.predict(features[test], task_type="METAL").reshape(-1)
        accuracy = float(np.mean(predictions == labels[test]))
        if accuracy < .8:
            raise AssertionError(f"Unexpectedly low held-out Iris accuracy: {accuracy}")
        for format in ("cbm", "json"):
            filename = args.output_dir / f"{objective}.{format}"
            model.save_model(filename, format=format)
            restored = CatBoostClassifier().load_model(str(filename), format=format)
            np.testing.assert_allclose(restored.predict_proba(features[test]), probabilities, atol=1e-12, rtol=1e-12)
        record = {
            "objective": objective, "learn_rows": len(learn), "validation_rows": len(validation),
            "test_rows": len(test), "classes": model.classes_.tolist(),
            "test_accuracy": accuracy, "trees": model.tree_count_,
            "best_iteration": model.best_iteration_, "stats": model.training_stats_,
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "scope": "A small repository fixture; no NVIDIA performance or quality comparison.",
        }
        results.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
