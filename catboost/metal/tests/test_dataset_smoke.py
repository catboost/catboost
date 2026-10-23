"""End-to-end actual-data check; CPU CatBoost is used solely for model reading."""

import importlib.util
from pathlib import Path
import platform

import pytest
from catboost import CatBoostClassifier, CatBoostRegressor


def test_repository_datasets_learn_and_export_on_metal(monkeypatch, tmp_path):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires Apple Silicon Metal")
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset validation must not invoke CPU training")
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    path = Path(__file__).resolve().parents[1] / "examples" / "train_datasets.py"
    spec = importlib.util.spec_from_file_location("metal_dataset_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    report = example.run_datasets(tmp_path, iterations=100)
    assert len(report["cases"]) == 3
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["adult_onehot"]["split_types"].get("OneHotFeature", 0) > 0
    assert cases["adult_borders_ctr"]["split_types"].get("OnlineCtr", 0) > 0
    for case in report["cases"]:
        assert case["validation_rows"] > 0 and case["test_rows"] > 0
        assert 1 <= case["tree_count"] <= 100
        assert case["training_stats"]["device"].startswith("Apple")
        assert case["training_stats"]["kernel_dispatches"] > 0
        assert case["gpu_vs_standard_prediction_max_abs_error"] < 1e-5
        assert case["gpu_vs_independent_json_prediction_max_abs_error"] < 1e-5
        assert (tmp_path / case["name"] / "model.cbm").is_file()
        assert (tmp_path / case["name"] / "model.json").is_file()
