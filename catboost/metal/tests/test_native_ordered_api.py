"""Native numeric Ordered acceptance. Every fit uses the standard GPU API.

Run with CATBOOST_NATIVE_METAL_TESTS=1 and PYTHONPATH pointing at the rebuilt
native package. These checks cover the initial single-permutation adapter;
runtime equation and independent prefix-state tests live in test_ordered_training.py.
"""

import os

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostError, CatBoostRegressor, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal CatBoost package",
)


def dataset(rows=128):
    rng = np.random.default_rng(419)
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    target = (1.5 * x[:, 0] - 0.7 * x[:, 1] + 0.1 * x[:, 2]).astype(np.float32)
    weights = np.linspace(0.4, 2, rows, dtype=np.float32)
    return x, target, weights


def options(**extra):
    result = dict(task_type="GPU", boosting_type="Ordered", iterations=12,
                  depth=3, learning_rate=0.15, random_seed=47, border_count=20,
                  bootstrap_type="No", random_strength=0, score_function="Cosine",
                  leaf_estimation_backtracking="No", boost_from_average=False,
                  verbose=False, allow_writing_files=False)
    result.update(extra)
    return result


def regressor(**config):
    return CatBoostRegressor(**config).set_params(permutation_count=1)


def classifier(**config):
    return CatBoostClassifier(**config).set_params(permutation_count=1)


class StopAfter:
    def __init__(self, count):
        self.count = count

    def after_iteration(self, info):
        return info.iteration < self.count


@pytest.mark.parametrize("objective,method", [
    ("RMSE", "Newton"),
    ("Logloss", "Newton"),
    ("CrossEntropy", "Newton"),
    ("Poisson", "Newton"),
    ("Huber:delta=1.2", "Newton"),
    ("Expectile:alpha=0.7", "Gradient"),
    ("Lq:q=2.5", "Newton"),
    ("Tweedie:variance_power=1.5", "Gradient"),
    ("LogLinQuantile:alpha=0.7", "Gradient"),
    ("Quantile:alpha=0.7", "Gradient"),
    ("MAE", "Gradient"),
    ("MAPE", "Gradient"),
])
def test_native_ordered_weighted_scalar_objectives(objective, method):
    x, target, weights = dataset()
    estimator = regressor
    if objective in {"Logloss", "CrossEntropy"}:
        estimator = classifier
        target = ((target > 0).astype(np.float32) if objective == "Logloss" else
                  (1 / (1 + np.exp(-target))).astype(np.float32))
    elif objective.startswith(("Poisson", "Tweedie", "LogLinQuantile", "MAPE")):
        target = np.exp(target / 3).astype(np.float32)
    pool = Pool(x, target, weight=weights)
    model = estimator(**options(loss_function=objective, leaf_estimation_method=method,
                                leaf_estimation_iterations=2))
    model.fit(pool, eval_set=pool, use_best_model=False)
    params = model.get_all_params()
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert params["task_type"] == "GPU"
    assert params["boosting_type"] == "Ordered"
    assert params["data_partition"] == "FeatureParallel"
    assert params["permutation_count"] == 1
    assert params["leaf_estimation_method"] == method
    history = next(iter(model.get_evals_result()["learn"].values()))
    assert np.isfinite(history).all()
    assert history[-1] < history[0]
    raw = model.predict(x, prediction_type="RawFormulaVal")
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(model.predict(x, prediction_type="RawFormulaVal", task_type="GPU"),
                               raw, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model.get_test_eval(), raw, rtol=3e-6, atol=3e-6)
    if objective == "RMSE":
        expected = np.sqrt(np.average((raw - target) ** 2, weights=weights))
        assert model.get_evals_result()["validation"]["RMSE"][-1] == pytest.approx(expected, rel=3e-6)


@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("score", ["Cosine", "NewtonCosine"])
def test_native_ordered_score_and_leaf_method(method, score):
    x, target, weights = dataset()
    model = regressor(**options(leaf_estimation_method=method, score_function=score,
                                       leaf_estimation_iterations=3)).fit(x, target, sample_weight=weights)
    assert model.get_all_params()["score_function"] == score
    assert model.get_all_params()["leaf_estimation_method"] == method
    assert model.get_evals_result()["learn"]["RMSE"][-1] < model.get_evals_result()["learn"]["RMSE"][0]


@pytest.mark.parametrize("objective", ["MAE", "Quantile:alpha=0.7", "MAPE"])
def test_native_ordered_nonsmooth_losses_default_to_gradient(objective):
    x, target, weights = dataset()
    config = options(loss_function=objective)
    config.pop("boost_from_average")
    model = regressor(**config).fit(x, target, sample_weight=weights)
    assert model.get_all_params()["leaf_estimation_method"] == "Gradient"
    assert np.isfinite(model.predict(x)).all()
    history = next(iter(model.get_evals_result()["learn"].values()))
    assert history[-1] < history[0]


@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_native_ordered_backtracking_and_score_noise_are_seeded(backtracking):
    x, target, weights = dataset()
    config = options(iterations=7, random_strength=1.3, leaf_estimation_iterations=4,
                     leaf_estimation_backtracking=backtracking)
    first = regressor(**config).fit(x, target, sample_weight=weights)
    second = regressor(**config).fit(x, target, sample_weight=weights)
    assert first.get_all_params()["leaf_estimation_backtracking"] == backtracking
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert first.get_evals_result()["learn"]["RMSE"][-1] < first.get_evals_result()["learn"]["RMSE"][0]


@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson", "MVS"])
def test_native_ordered_snapshot_preserves_all_prefix_cursors(tmp_path, bootstrap):
    x, target, weights = dataset()
    bootstrap_options = dict(bootstrap_type=bootstrap)
    if bootstrap == "Bayesian":
        bootstrap_options["bagging_temperature"] = 0.7
    elif bootstrap in {"Bernoulli", "Poisson", "MVS"}:
        bootstrap_options["subsample"] = 0.8
    common = options(**bootstrap_options, iterations=9, metric_period=3,
                     random_strength=0.6, allow_writing_files=True,
                     train_dir=str(tmp_path), save_snapshot=True, snapshot_interval=0,
                     snapshot_file="ordered.snapshot")
    pool = Pool(x, target, weight=weights)
    partial = regressor(**common).fit(pool, eval_set=pool, use_best_model=False,
                                             callbacks=[StopAfter(4)])
    assert partial.tree_count_ == 4
    assert (tmp_path / "ordered.snapshot").is_file()
    resumed = regressor(**common).fit(pool, eval_set=pool, use_best_model=False)
    direct = regressor(**options(**bootstrap_options, iterations=9, metric_period=3,
                                         random_strength=0.6)).fit(pool, eval_set=pool, use_best_model=False)
    assert resumed.tree_count_ == direct.tree_count_ == 9
    np.testing.assert_array_equal(resumed.get_tree_leaf_counts(), direct.get_tree_leaf_counts())
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_leaf_weights(), direct.get_leaf_weights())
    np.testing.assert_array_equal(resumed.predict(x), direct.predict(x))
    assert resumed.get_evals_result() == direct.get_evals_result()
    completed = regressor(**common).fit(pool, eval_set=pool, use_best_model=False)
    np.testing.assert_array_equal(completed.predict(x), resumed.predict(x))
    assert completed.tree_count_ == 9


@pytest.mark.parametrize("quantized", [False, True])
def test_native_ordered_quantized_pool_model_roundtrip_and_gpu_inference(tmp_path, quantized):
    x, target, weights = dataset()
    pool = Pool(x, target, weight=weights, feature_names=["a", "b", "c", "d"])
    if quantized:
        pool.quantize(border_count=20)
        path = tmp_path / "ordered.pool"
        pool.save(path)
        pool = Pool("quantized://" + str(path))
    model = regressor(**options()).fit(pool)
    assert model.feature_names_ == ["a", "b", "c", "d"]
    raw = model.predict(x)
    np.testing.assert_allclose(model.predict(pool), raw, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), raw, atol=3e-6, rtol=3e-6)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("ordered." + format_)
        model.save_model(path, format=format_)
        restored = regressor().load_model(path, format=format_)
        if format_ == "cbm":
            np.testing.assert_array_equal(restored.predict(x), raw)
        else:
            np.testing.assert_allclose(restored.predict(x), raw, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(restored.predict(x, task_type="GPU"), raw, atol=3e-6, rtol=3e-6)


def test_native_ordered_callbacks_initial_model_and_baseline():
    x, target, weights = dataset()
    first = regressor(**options()).fit(x, target, sample_weight=weights, callbacks=[StopAfter(4)])
    assert first.tree_count_ == 4
    initial = first.predict(x)
    continued = regressor(**options(iterations=5)).fit(x, target, sample_weight=weights, init_model=first)
    assert continued.tree_count_ == 9
    assert np.average((continued.predict(x) - target) ** 2, weights=weights) < np.average((initial - target) ** 2, weights=weights)
    baseline_pool = Pool(x, target, weight=weights, baseline=initial)
    baseline = regressor(**options(iterations=5)).fit(baseline_pool, eval_set=baseline_pool,
                                                           use_best_model=False)
    np.testing.assert_allclose(baseline.predict(x) + initial, continued.predict(x), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(baseline.get_test_eval(), baseline.predict(x) + initial, rtol=3e-6, atol=3e-6)


def test_native_ordered_multiple_eval_sets_early_stop_and_unlabeled_pool():
    x, target, weights = dataset()
    model = regressor(**options(iterations=40, early_stopping_rounds=4, metric_period=3,
                                       use_best_model=True, best_model_min_trees=2))
    model.fit(Pool(x, target, weight=weights), eval_set=[Pool(x[:20]), Pool(x, -target, weight=weights)])
    assert 2 <= model.tree_count_ < 40
    assert model.get_best_iteration() >= 0
    evals = model.get_test_evals()
    assert len(evals) == 2
    np.testing.assert_allclose(evals[0][0], model.predict(x[:20]), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(evals[1][0], model.predict(x), rtol=3e-6, atol=3e-6)


def test_native_ordered_fold_options_and_bootstrap_observations():
    x, target, weights = dataset(600)
    for normalization, observations in [(False, "TestOnly"), (True, "LearnAndTest")]:
        model = regressor(**options(iterations=5, depth=2, bootstrap_type="Bernoulli", subsample=0.8))
        model.set_params(min_fold_size=32, fold_len_multiplier=1.7,
                         fold_size_loss_normalization=normalization, observations_to_bootstrap=observations)
        model.fit(x, target, sample_weight=weights)
        params = model.get_all_params()
        assert params["min_fold_size"] == 32
        assert params["fold_len_multiplier"] == pytest.approx(1.7)
        assert params["fold_size_loss_normalization"] == normalization
        assert params["observations_to_bootstrap"] == observations
        assert model.get_evals_result()["learn"]["RMSE"][-1] < model.get_evals_result()["learn"]["RMSE"][0]


@pytest.mark.parametrize("case,match", [
    ("permutations", "Permutation count should be positive"),
    ("groups", "at least four groups"),
    ("categorical_full", "learn-only CTR"),
    ("multiclass", "(?i)Ordered.*scalar|multiclass.*Ordered"),
    ("exact", "(?i)Ordered.*Exact|Exact.*Ordered"),
])
def test_native_ordered_unsupported_combinations_fail_clearly(case, match):
    x, target, weights = dataset()
    model = regressor(**options(iterations=2))
    pool = Pool(x, target, weight=weights)
    if case == "permutations":
        model.set_params(permutation_count=0)
    elif case == "groups":
        pool = Pool(x, target, weight=weights, group_id=np.repeat(np.arange(2), len(x) // 2))
    elif case == "categorical_full":
        model.set_params(one_hot_max_size=1, max_ctr_complexity=2, counter_calc_method="Full")
        mixed = np.empty((len(x), 2), dtype=object)
        mixed[:, 0] = x[:, 0]
        mixed[:, 1] = np.where(x[:, 1] > 0, "right", "left")
        pool = Pool(mixed, target, weight=weights, cat_features=[1])
    elif case == "multiclass":
        model = classifier(**options(iterations=2, loss_function="MultiClass"))
        pool = Pool(x, np.arange(len(x)) % 3, weight=weights)
    elif case == "exact":
        model.set_params(loss_function="MAE", leaf_estimation_method="Exact", leaf_estimation_iterations=1)
    with pytest.raises(CatBoostError, match=match):
        model.fit(pool)


@pytest.mark.parametrize("changed", ["target", "weight", "fold"])
def test_native_ordered_snapshot_rejects_incompatible_data_or_folds(tmp_path, changed):
    x, target, weights = dataset()
    common = options(iterations=7, allow_writing_files=True, train_dir=str(tmp_path),
                     save_snapshot=True, snapshot_interval=0, snapshot_file="ordered.snapshot")
    regressor(**common).fit(x, target, sample_weight=weights, callbacks=[StopAfter(3)])
    model = regressor(**common)
    if changed == "target":
        target = target + 1
    elif changed == "weight":
        weights = weights.copy()
        weights[0] *= 2
    else:
        model.set_params(min_fold_size=16)
    with pytest.raises(CatBoostError, match="snapshot|Snapshot"):
        model.fit(x, target, sample_weight=weights)
