"""Existing public estimator interfaces for GPU vector objectives."""
import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor


OBJECTIVES = ("MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")


@pytest.fixture(autouse=True)
def prohibit_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("These tests must not fit CPU models")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)


def dataset(objective, rows=193):
    rng = np.random.default_rng(915)
    X = rng.normal(size=(rows, 4)).astype(np.float32)
    y = np.column_stack([X[:, 0] + .1 * X[:, 2], X[:, 1] - .2 * X[:, 3], X[:, 2]])
    if objective == "RMSEWithUncertainty":
        y = y[:, 0] + rng.normal(size=rows).astype(np.float32) * (.1 + .2 * abs(X[:, 1]))
    elif objective == "MultiLogloss":
        y = (y > 0).astype(np.float32)
    elif objective == "MultiCrossEntropy":
        y = (1 / (1 + np.exp(-y))).astype(np.float32)
    return X, y


def estimator(objective, **kwargs):
    cls = CatBoostMetalClassifier if objective in ("MultiLogloss", "MultiCrossEntropy") else CatBoostMetalRegressor
    return cls(loss_function=objective, iterations=10, depth=3, learning_rate=.2,
               border_count=16, leaf_estimation_iterations=2, **kwargs)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_existing_estimator_prediction_types_and_roundtrip(objective, tmp_path):
    X, y = dataset(objective)
    model = estimator(objective).fit(X, y)
    kinds = [None, "RawFormulaVal", "Probability", "LogProbability", "Exponent"]
    if model._classifier:
        kinds.append("Class")
    if objective == "RMSEWithUncertainty":
        kinds.append("RMSEWithUncertainty")
    for kind in kinds:
        for start, end in [(0, 0), (2, 7)]:
            cpu = model.predict(X[:29], kind, ntree_start=start, ntree_end=end)
            metal = model.predict(X[:29], kind, task_type="GPU", ntree_start=start, ntree_end=end)
            np.testing.assert_allclose(metal, cpu, rtol=2e-11, atol=2e-11)
            assert model.predict(X[:0], kind, task_type="GPU").shape == (0, *np.shape(cpu)[1:])
    for fmt in ("json", "cbm"):
        path = tmp_path / f"vector.{fmt}"
        model.save_model(path, format=fmt)
        native_cls = CatBoostClassifier if model._classifier else CatBoostRegressor
        restored = native_cls().load_model(path, format=fmt)
        np.testing.assert_allclose(restored.predict(X[:29]), model.predict(X[:29], task_type="GPU"), rtol=2e-11, atol=2e-11)
    assert model.n_outputs_ == (2 if objective == "RMSEWithUncertainty" else 3)
    assert "Apple" in model.training_stats_["device"]


def test_weighted_multi_rmse_bias_and_pool():
    X, y = dataset("MultiRMSE")
    y += np.array([3, -5, 7], np.float32)
    weight = np.linspace(.1, 2, len(X), dtype=np.float32)
    model = estimator("MultiRMSE", boost_from_average=True).fit(Pool(X, y, weight=weight))
    expected = np.average(y.astype(float), axis=0, weights=weight.astype(float)).astype(np.float32)
    np.testing.assert_array_equal(model.bias_, expected)
    np.testing.assert_allclose(model.predict(Pool(X[:17]), task_type="METAL"), model.to_catboost().predict(X[:17]), atol=2e-12)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_validation_callback_snapshot_continuation(objective, tmp_path):
    X, y = dataset(objective)
    parameters = dict(bootstrap_type="Bayesian", random_seed=8, random_strength=.2)
    expected = estimator(objective, **parameters).fit(X[:130], y[:130],
        eval_set=(X[130:], y[130:]), use_best_model=False)
    snapshot = tmp_path / "vector.npz"
    first = estimator(objective, **parameters).fit(X[:130], y[:130],
        eval_set=(X[130:], y[130:]), use_best_model=False, snapshot_file=snapshot,
        callback=lambda info: info.iteration < 4)
    assert first.tree_count_ == 4
    resumed = estimator(objective, **parameters).fit(X[:130], y[:130],
        eval_set=(X[130:], y[130:]), use_best_model=False, snapshot_file=snapshot)
    np.testing.assert_array_equal(resumed._result.leaf_values, expected._result.leaf_values)
    np.testing.assert_array_equal(resumed.training_predictions_, expected.training_predictions_)
    assert resumed.get_evals_result() == expected.get_evals_result()


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_failed_refit_invalid_target_clears_model(objective):
    X, y = dataset(objective)
    model = estimator(objective).fit(X, y)
    invalid = y.copy()
    invalid.flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, invalid)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_classifier_infers_multilabel_loss_then_refits_binary():
    X, binary = dataset("MultiLogloss")
    _, soft = dataset("MultiCrossEntropy")
    model = CatBoostMetalClassifier(iterations=3, depth=2)
    model.fit(X, binary)
    assert model.loss_function == "MultiLogloss"
    assert model.predict_proba(X[:5], task_type="GPU").shape == (5, 3)
    model.fit(X, soft)
    assert model.loss_function == "MultiCrossEntropy"
    np.testing.assert_allclose(model.predict_proba(X[:5], task_type="GPU"), model.to_catboost().predict_proba(X[:5]), atol=2e-12)
    model.fit(X, binary[:, 0])
    assert model.loss_function == "Logloss"
    assert model.predict(X[:5], task_type="GPU").shape == (5,)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_validation_early_stopping_retains_consistent_vector_model(objective):
    X, y = dataset(objective)
    model = estimator(objective).fit(X[:110], y[:110], eval_set=(X[110:], y[110:]),
        early_stopping_rounds=2, use_best_model=True)
    assert model.tree_count_ == model.get_best_iteration() + 1
    np.testing.assert_allclose(model.predict(X[110:], "RawFormulaVal", task_type="GPU"),
                               model.to_catboost().predict(X[110:], "RawFormulaVal"), atol=2e-12)
