"""Model option inspection and serialization after actual Metal training."""

import json
import platform

import numpy as np
import pytest
from catboost import CatBoostRegressor

from catboost_metal import CatBoostMetalRegressor


@pytest.fixture(params=["L2", "Cosine"])
def trained_model(request, monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal GPU checks require macOS on Apple Silicon")

    def forbid_cpu_training(*args, **kwargs):
        raise AssertionError("Metadata checks must not invoke CatBoost CPU training")

    monkeypatch.setattr(CatBoostRegressor, "fit", forbid_cpu_training)
    features = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=np.float32)
    targets = np.array([-4, -4, -4, 6, 6, 6], dtype=np.float32)
    model = CatBoostMetalRegressor(
        iterations=1, depth=2, learning_rate=0.25, l2_leaf_reg=2,
        border_count=3, score_function=request.param,
    ).fit(features, targets)
    assert model.training_stats_["kernel_dispatches"] > 0
    return model, features


def _check_options(model, score_function):
    expected = {
        "loss_function": "RMSE", "iterations": 1, "depth": 2,
        "learning_rate": 0.25, "l2_leaf_reg": 2, "border_count": 3,
        "score_function": score_function, "boosting_type": "Plain",
        "data_partition": "DocParallel", "grow_policy": "SymmetricTree",
        "bootstrap_type": "No", "random_strength": 0,
        "leaf_estimation_method": "Newton", "leaf_estimation_iterations": 1,
        "leaf_estimation_backtracking": "No", "task_type": "GPU",
        "permutation_count": 1,
        "boost_from_average": True, "feature_border_type": "GreedyLogSum",
        "nan_mode": "Forbidden", "use_best_model": False,
        "fold_size_loss_normalization": False,
        "add_ridge_penalty_to_loss_function": False,
    }
    # GPU describes where training ran; the explicit backend metadata below
    # distinguishes Metal from NVIDIA. It also makes GPU-only options parse.
    assert model.get_params() == expected
    # Upstream's effective-options view omits the permutation count when the
    # exported model has no categorical features (plain_options_helper.cpp).
    assert model.get_all_params() == {key: value for key, value in expected.items()
                                     if key != "permutation_count"}
    assert model.learning_rate_ == 0.25
    metadata = model.get_metadata()
    assert metadata["metal_backend"] == "METAL"
    nested = json.loads(metadata["params"])
    assert nested["task_type"] == "GPU"
    assert nested["flat_params"]["task_type"] == "GPU"
    assert json.loads(metadata["output_options"]) == {"use_best_model": False}


def test_option_inspection_and_roundtrip_after_metal_fit(trained_model, tmp_path, capfd):
    wrapper, features = trained_model
    model = wrapper.to_catboost()
    _check_options(model, wrapper.score_function)
    predictions = wrapper.predict(features)
    np.testing.assert_allclose(predictions, wrapper.training_predictions_, atol=1e-6)
    np.testing.assert_array_equal(model.predict(features), predictions)

    for model_format in ("json", "cbm"):
        path = tmp_path / f"metal-metadata.{model_format}"
        wrapper.save_model(path, format=model_format)
        restored = CatBoostRegressor().load_model(str(path), format=model_format)
        _check_options(restored, wrapper.score_function)
        np.testing.assert_array_equal(restored.predict(features), predictions)
    assert "invalid params" not in capfd.readouterr().err
