"""Host quantization and real Metal training with missing/categorical inputs."""

import json

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor
from catboost_metal._data import quantize_features


@pytest.fixture(autouse=True)
def no_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Metal tests must not call CatBoost CPU training")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_nan_borders_match_upstream_pool_quantization(nan_mode, tmp_path):
    X = np.array([[np.nan], [-4], [-2], [0], [1], [3], [9]], np.float32)
    borders, bins, _, _ = quantize_features(X, 4, nan_mode)
    pool = Pool(X)
    pool.quantize(border_count=4, feature_border_type="GreedyLogSum", nan_mode=nan_mode)
    filename = tmp_path / "borders.tsv"
    pool.save_quantization_borders(str(filename))
    expected = [np.float32(line.split()[1]) for line in filename.read_text().splitlines()]
    np.testing.assert_array_equal(borders[0], expected)
    assert bins[0, 0] == (0 if nan_mode == "Min" else len(expected))


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_missing_values_train_predict_and_export(nan_mode, tmp_path):
    rng = np.random.default_rng(281)
    X = rng.normal(size=(127, 3)).astype(np.float32)
    X[::5, 1] = np.nan
    y = np.where(np.isnan(X[:, 1]), 7.0, 2.0 * X[:, 0]).astype(np.float32)
    model = CatBoostMetalRegressor(iterations=5, depth=3, nan_mode=nan_mode).fit(X, y)
    np.testing.assert_allclose(model.predict(X), model.training_predictions_, atol=2e-5)
    np.testing.assert_allclose(model.predict(X, task_type="METAL"), model.predict(X), atol=2e-5)
    path = tmp_path / "missing.json"
    model.save_model(path, format="json")
    info = json.loads(path.read_text())["features_info"]["float_features"][1]
    assert info["has_nans"] is True
    assert info["nan_value_treatment"] == ("AsFalse" if nan_mode == "Min" else "AsTrue")
    restored = CatBoostRegressor().load_model(str(path), format="json")
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_one_hot_training_retains_original_categorical_model(tmp_path):
    X = np.array([["red", -2], ["blue", -1], ["green", 1], ["red", 2],
                  ["blue", 3], ["green", 4]] * 11, dtype=object)
    y = np.array([8, -5, 1, 8, -5, 1] * 11, np.float32)
    model = CatBoostMetalRegressor(iterations=4, depth=2, learning_rate=0.4,
                                  cat_features=[0]).fit(X, y)
    assert model.to_catboost().get_cat_feature_indices() == [0]
    assert any(model._result.split_types.ravel() == 1)
    probes = np.array([["red", 0], ["blue", 0], ["unseen", 0], ["green", 0]], dtype=object)
    np.testing.assert_allclose(model.predict(X), model.training_predictions_, atol=1e-5)
    np.testing.assert_allclose(model.predict(probes, task_type="METAL"), model.predict(probes), atol=1e-5)
    path = tmp_path / "categorical.cbm"
    model.save_model(path)
    restored = CatBoostRegressor().load_model(str(path))
    np.testing.assert_array_equal(restored.predict(probes), model.predict(probes))


def test_numeric_pool_reuses_labels_weights_and_names():
    X = np.array([[-2], [-1], [0], [1], [2]], np.float32)
    y = np.array([-4, -4, 0, 6, 6], np.float32)
    weights = np.array([1, 2, 0, 3, 4], np.float32)
    options = dict(iterations=2, depth=2)
    direct = CatBoostMetalRegressor(**options).fit(X, y, sample_weight=weights)
    pooled = CatBoostMetalRegressor(**options).fit(Pool(X, y, weight=weights, feature_names=["measurement"]))
    np.testing.assert_allclose(direct.training_predictions_, pooled.training_predictions_, atol=1e-6)
    assert pooled.to_catboost().feature_names_ == ["measurement"]


def test_string_classifier_labels_survive_standard_model_roundtrip(tmp_path):
    X = np.arange(-8, 8, dtype=np.float32)[:, None]
    y = np.where(X[:, 0] > 0, "accepted", "rejected")
    model = CatBoostMetalClassifier(iterations=3, depth=2).fit(X, y)
    np.testing.assert_array_equal(model.classes_, ["accepted", "rejected"])
    np.testing.assert_allclose(model.predict_proba(X, task_type="METAL"), model.predict_proba(X), atol=1e-6)
    np.testing.assert_array_equal(model.predict(X, task_type="METAL"), model.predict(X))
    path = tmp_path / "labels.cbm"
    model.save_model(path)
    restored = CatBoostClassifier().load_model(str(path))
    np.testing.assert_array_equal(restored.classes_, model.classes_)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_default_classifier_selects_loss_and_resets_classes_on_refit():
    rng = np.random.default_rng(293)
    features = rng.normal(size=(180, 4)).astype(np.float32)
    labels = np.asarray(["red", "green", "blue"])[np.argmax(features[:, :3], axis=1)]
    model = CatBoostMetalClassifier(iterations=4, depth=2).fit(features, labels)
    assert model.loss_function == "MultiClass"
    assert model.predict_proba(features).shape == (180, 3)
    np.testing.assert_allclose(model.predict_proba(features, task_type="METAL"),
                               model.predict_proba(features), atol=1e-12)
    model.fit(features, np.where(features[:, 0] > 0, "yes", "no"))
    assert model.loss_function == "Logloss"
    np.testing.assert_array_equal(model.classes_, ["no", "yes"])
    assert model.predict_proba(features).shape == (180, 2)
    with pytest.raises(ValueError, match="exactly two"):
        CatBoostMetalClassifier(loss_function="Logloss", iterations=1).fit(features, labels)


@pytest.mark.parametrize("weights", [[0, 0], [1, -1], [1], [1, np.inf]])
def test_invalid_weights_rejected_before_runtime(weights, monkeypatch):
    from catboost_metal import _native
    monkeypatch.setattr(_native, "train", lambda *a, **kw: pytest.fail("Unexpected GPU call"))
    with pytest.raises(ValueError):
        CatBoostMetalRegressor().fit([[0], [1]], [0, 1], sample_weight=weights)
