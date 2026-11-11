"""Acceptance tests for the rebuilt native CatBoost extension, with real Metal training.

CATBOOST_NATIVE_METAL_TESTS=1 PYTHONPATH=/path/to/native/package python -m pytest ...
The opt-in keeps an unrelated installed wheel from being mistaken for this build.
"""

import numpy as np
import pytest
import os

from catboost import CatBoostClassifier, CatBoostError, CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal CatBoost package",
)


def data():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(120, 4)).astype(np.float32)
    y = (1.5 * x[:, 0] - 0.7 * x[:, 1] + 0.1 * x[:, 2]).astype(np.float32)
    return x, y, np.linspace(0.4, 2.0, len(y), dtype=np.float32)


def params(**kwargs):
    result = dict(task_type="GPU", iterations=16, depth=3, learning_rate=0.2,
                  bootstrap_type="No", random_strength=0, random_seed=23,
                  border_count=24, verbose=False, allow_writing_files=False)
    result.update(kwargs)
    return result


def test_native_model_and_quantized_pool_roundtrip(tmp_path):
    x, y, w = data()
    pool = Pool(x, y, weight=w, feature_names=["a", "b", "c", "d"])
    pool.quantize(border_count=24)
    model = CatBoostRegressor(**params()).fit(pool)
    assert get_gpu_device_count() == 1
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_all_params()["task_type"] == "GPU"
    assert model.feature_names_ == ["a", "b", "c", "d"]
    prediction = model.predict(x)
    assert np.average((prediction - y) ** 2, weights=w) < np.average(y ** 2, weights=w) * 0.3
    for format_, suffix in [("cbm", ".cbm"), ("json", ".json")]:
        path = tmp_path / ("model" + suffix)
        model.save_model(path, format=format_)
        loaded = CatBoostRegressor().load_model(path, format=format_)
        if format_ == "cbm":
            np.testing.assert_array_equal(loaded.predict(x), prediction)
        else:
            np.testing.assert_array_max_ulp(loaded.predict(x), prediction, maxulp=2)


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_native_numeric_missing_values_follow_pool_quantization(nan_mode):
    x, y, _ = data()
    x[::7, 0] = np.nan
    model = CatBoostRegressor(**params(nan_mode=nan_mode)).fit(x, y)
    prediction = model.predict(x)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), prediction, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("objective", ["RMSE", "Poisson", "Huber:delta=1.2", "Expectile:alpha=0.7"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_native_weighted_scalar_objectives(objective, method):
    x, y, w = data()
    if objective == "Poisson":
        y = np.exp(y / 3).astype(np.float32)
    model = CatBoostRegressor(**params(loss_function=objective, leaf_estimation_method=method,
                                      leaf_estimation_iterations=2)).fit(x, y, sample_weight=w)
    history = next(iter(model.get_evals_result()["learn"].values()))
    assert np.isfinite(model.predict(x)).all()
    assert history[-1] < history[0]


@pytest.mark.parametrize("objective,method", [
    ("Lq:q=2.5", "Newton"), ("Lq:q=1.5", "Gradient"),
    ("Tweedie:variance_power=1.5", "Newton"), ("Tweedie:variance_power=1.5", "Gradient"),
    ("LogLinQuantile:alpha=0.7", "Gradient"),
    ("Quantile:alpha=0.7", "Gradient"), ("Quantile:alpha=0.7", "Exact"),
    ("MAE", "Gradient"), ("MAE", "Exact"), ("MAPE", "Gradient"), ("MAPE", "Exact"),
])
def test_native_extended_scalar_objectives(objective, method):
    x, y, w = data()
    if objective.startswith(("Tweedie", "LogLinQuantile")):
        y = np.exp(y / 3).astype(np.float32)
    model = CatBoostRegressor(**params(loss_function=objective, leaf_estimation_method=method,
                                      leaf_estimation_iterations=1 if method == "Exact" else 2))
    model.fit(x, y, sample_weight=w, eval_set=Pool(x, y, weight=w))
    history = next(iter(model.get_evals_result()["learn"].values()))
    assert np.isfinite(model.predict(x)).all()
    assert history[-1] < history[0]
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), model.predict(x), rtol=2e-6, atol=2e-6)
    assert model.get_all_params()["leaf_estimation_method"] == method


@pytest.mark.parametrize("objective,alpha", [("Quantile:alpha=0.7", 0.7), ("MAE", 0.5), ("MAPE", 0.5)])
def test_native_exact_leaf_is_weighted_residual_quantile(objective, alpha):
    targets = np.array([-5, -2, 1, 3, 8, 13], dtype=np.float32)
    weights = np.array([0, 1, 2, 1, 4, 2], dtype=np.float32)
    x = np.arange(len(targets), dtype=np.float32).reshape(-1, 1)
    baseline = np.full(len(targets), 0.75, dtype=np.float32)
    model = CatBoostRegressor(**params(loss_function=objective, leaf_estimation_method="Exact",
                                      iterations=1, depth=0, learning_rate=0.25, l2_leaf_reg=8))
    model.fit(Pool(x, targets, weight=weights, baseline=baseline))
    effective_weights = weights / np.maximum(1, np.abs(targets)) if objective == "MAPE" else weights
    selected = np.searchsorted(np.cumsum(effective_weights), alpha * effective_weights.sum(), side="left")
    expected = 0.25 * (targets[selected] - baseline[0])
    np.testing.assert_allclose(model.get_leaf_values(), [expected], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(model.predict(x), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), [weights.sum()])


@pytest.mark.parametrize("objective,expected", [
    ("MAE", 1e-6), ("Quantile:alpha=0.5;delta=0.005", 0.005),
    ("Quantile:alpha=0.5;delta=0", 0), ("MAPE", 0),
])
def test_native_quantile_bias_selects_observed_target_at_extreme_range(objective, expected):
    x = np.arange(100, dtype=np.float32).reshape(-1, 1)
    target = np.zeros(100, dtype=np.float32)
    target[-1] = np.float32(1e38)
    model = CatBoostRegressor(**params(loss_function=objective, iterations=1, depth=0,
                                      boost_from_average=True, leaf_estimation_method="Exact"))
    model.fit(x, target)
    assert model.get_scale_and_bias()[1] == float(np.float32(expected))
    assert np.isfinite(model.predict(x)).all()


def test_native_quantile_bias_preserves_tiny_positive_weight_threshold():
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    target = np.array([-10, 0, 5, 9], dtype=np.float32)
    weights = np.array([0, 1, 4, 1], dtype=np.float32) * np.float32(1e-30)
    model = CatBoostRegressor(**params(loss_function="Quantile:alpha=0.5;delta=0", iterations=1,
                                      depth=0, boost_from_average=True, leaf_estimation_method="Exact"))
    model.fit(x, target, sample_weight=weights)
    assert model.get_scale_and_bias()[1] == 5


def test_native_parameterized_objective_is_configured_before_initial_loss():
    # q=1.5 has a finite float32 loss here. An accidental default q=2 during
    # session creation overflows before the requested objective is configured.
    x = np.arange(8, dtype=np.float32).reshape(-1, 1)
    target = np.linspace(0.5e20, 1e20, len(x), dtype=np.float32)
    model = CatBoostRegressor(**params(loss_function="Lq:q=1.5", iterations=1, depth=0,
                                      leaf_estimation_method="Gradient", learning_rate=0.01))
    model.fit(x, target)
    assert np.isfinite(model.predict(x)).all()
    assert np.isfinite(model.get_evals_result()["learn"]["Lq:q=1.5"]).all()


def test_native_depth16_reaches_all_levels_and_roundtrips(tmp_path):
    # Each unused bit partitions every leaf evenly. Distinct coefficients and
    # zero regularization force all sixteen splits, yielding 65536 leaf values.
    rows = np.arange(1 << 16, dtype=np.uint32)
    x = ((rows[:, None] >> np.arange(16, dtype=np.uint32)) & 1).astype(np.float32)
    targets = ((2 * x - 1) * np.arange(1, 17, dtype=np.float32)).sum(axis=1)
    model = CatBoostRegressor(**params(iterations=1, depth=16, learning_rate=0.25,
                                      l2_leaf_reg=0, score_function="L2", boost_from_average=False))
    model.fit(x, targets)
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [1 << 16])
    np.testing.assert_allclose(model.predict(x), targets * 0.25, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(model.get_leaf_weights(), np.ones(1 << 16))
    path = tmp_path / "deep.cbm"
    model.save_model(path)
    restored = CatBoostRegressor().load_model(path)
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), model.predict(x), rtol=3e-6, atol=3e-6)


def test_native_binary_labels_and_metrics():
    x, y, w = data()
    labels = np.where(y > 0, "high", "low")
    model = CatBoostClassifier(**params(eval_metric="AUC", custom_metric=["Accuracy", "Logloss:use_weights=false"]))
    model.fit(x, labels, sample_weight=w, eval_set=Pool(x, labels, weight=w))
    assert set(model.classes_) == {"low", "high"}
    np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1.0)
    assert model.get_evals_result()["validation"]["AUC"][-1] > 0.9


def test_native_cross_entropy_soft_targets():
    x, y, w = data()
    soft_targets = (1.0 / (1.0 + np.exp(-y))).astype(np.float32)
    model = CatBoostClassifier(**params(loss_function="CrossEntropy"))
    model.fit(x, soft_targets, sample_weight=w)
    loss = model.get_evals_result()["learn"]["CrossEntropy"]
    assert loss[-1] < loss[0]
    assert model.classes_.tolist() == [0, 1]
    np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1.0)


def test_native_multiple_eval_sets_early_stop_and_unlabeled_pool():
    x, y, w = data()
    model = CatBoostRegressor(**params(iterations=40, early_stopping_rounds=3, use_best_model=True))
    model.fit(Pool(x, y, weight=w), eval_set=[Pool(x[:20]), Pool(x, -y, weight=w)])
    assert 0 < model.tree_count_ < 40
    assert model.get_best_iteration() >= 0
    assert len(model.get_test_evals()) == 2
    np.testing.assert_allclose(model.get_test_evals()[1][0], model.predict(x), rtol=2e-6, atol=2e-6)


class StopAfter:
    def __init__(self, iterations):
        self.iterations = iterations

    def after_iteration(self, info):
        return info.iteration < self.iterations


class CustomRMSE:
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        error = 0.0
        total = 0.0
        for row in range(len(target)):
            value = 1.0 if weight is None else weight[row]
            error += value * (approxes[0][row] - target[row]) ** 2
            total += value
        return error, total

    def get_final_error(self, error, weight):
        return (error / weight) ** 0.5


def test_native_custom_evaluation_metric_uses_shared_host_api():
    x, y, w = data()
    model = CatBoostRegressor(**params(eval_metric=CustomRMSE()))
    model.fit(Pool(x, y, weight=w), eval_set=Pool(x, y, weight=w))
    metrics = model.get_evals_result()["validation"]
    np.testing.assert_allclose(metrics["CustomRMSE"], metrics["RMSE"], rtol=2e-6, atol=2e-6)


def test_native_callbacks_baseline_and_initial_model():
    x, y, _ = data()
    first = CatBoostRegressor(**params()).fit(x, y, callbacks=[StopAfter(4)])
    assert first.tree_count_ == 4
    second = CatBoostRegressor(**params(iterations=5)).fit(x, y, init_model=first)
    assert second.tree_count_ == 9
    assert np.mean((second.predict(x) - y) ** 2) < np.mean((first.predict(x) - y) ** 2)
    baseline = CatBoostRegressor(**params()).fit(Pool(x, y, baseline=y))
    np.testing.assert_array_equal(baseline.predict(x), np.zeros(len(y)))


@pytest.mark.parametrize("backtracking", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("score", ["Cosine", "L2"])
def test_native_score_noise_and_backtracking_are_seeded(backtracking, score):
    x, y, w = data()
    options = params(iterations=8, random_strength=1.3, score_function=score,
                     leaf_estimation_iterations=4, leaf_estimation_backtracking=backtracking)
    first = CatBoostRegressor(**options).fit(x, y, sample_weight=w)
    second = CatBoostRegressor(**options).fit(x, y, sample_weight=w)
    assert first.get_all_params()["random_strength"] == pytest.approx(1.3)
    assert first.get_all_params()["leaf_estimation_backtracking"] == backtracking
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert first.get_evals_result()["learn"]["RMSE"][-1] < first.get_evals_result()["learn"]["RMSE"][0]


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"])
@pytest.mark.parametrize("objective", ["RMSE", "Logloss"])
def test_native_all_scalar_scores(score, objective):
    x, y, weights = data()
    estimator = CatBoostClassifier if objective == "Logloss" else CatBoostRegressor
    target = (y > 0).astype(np.float32) if objective == "Logloss" else y
    model = estimator(**params(score_function=score, loss_function=objective)).fit(x, target, sample_weight=weights)
    assert model.get_all_params()["score_function"] == score
    loss = model.get_evals_result()["learn"][objective]
    assert loss[-1] < loss[0]
    raw = model.predict(x, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(model.predict(x, prediction_type="RawFormulaVal", task_type="GPU"), raw,
                               atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson", "MVS"])
def test_native_snapshot_restores_exact_gpu_cursor(tmp_path, bootstrap):
    x, y, w = data()
    extra = dict(bootstrap_type=bootstrap)
    if bootstrap in {"Bernoulli", "Poisson", "MVS"}:
        extra["subsample"] = 0.8
    common = params(**extra, iterations=14, metric_period=3, allow_writing_files=True,
                    train_dir=str(tmp_path), save_snapshot=True, snapshot_interval=0,
                    snapshot_file="metal.snapshot")
    partial = CatBoostRegressor(**common).fit(x, y, sample_weight=w, callbacks=[StopAfter(5)])
    assert partial.tree_count_ == 5
    restored = CatBoostRegressor(**common).fit(x, y, sample_weight=w)
    direct_params = params(**extra, iterations=14, metric_period=3)
    direct = CatBoostRegressor(**direct_params).fit(x, y, sample_weight=w)
    np.testing.assert_allclose(restored.get_leaf_values(), direct.get_leaf_values(), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(restored.get_tree_leaf_counts(), direct.get_tree_leaf_counts())
    np.testing.assert_allclose(restored.predict(x), direct.predict(x), rtol=2e-6, atol=2e-6)
    # Loading a completed checkpoint does not add more trees.
    again = CatBoostRegressor(**common).fit(x, y, sample_weight=w)
    np.testing.assert_array_equal(again.predict(x), restored.predict(x))
    with pytest.raises(CatBoostError, match="snapshot.*data|data.*differ"):
        CatBoostRegressor(**common).fit(x, y + 1, sample_weight=w)


def test_native_unsupported_options_fail_clearly():
    x, y, _ = data()
    with pytest.raises(CatBoostError, match="Metal.*objective|Metal supports"):
        CatBoostRegressor(**params(loss_function="LogCosh")).fit(x, y)
