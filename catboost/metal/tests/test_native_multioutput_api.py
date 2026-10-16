"""Standard CatBoost GPU API acceptance for the native vector objective adapter.

Opt in separately while the adapter is being validated:
CATBOOST_NATIVE_METAL_VECTOR_TESTS=1 PYTHONPATH=/path/to/native/package pytest ...
Every fit requests GPU; CPU model application is used only as an inference check.
"""

import os

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostError, CatBoostRegressor, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_VECTOR_TESTS") != "1",
    reason="requires the rebuilt native Metal vector adapter",
)

OBJECTIVES = ["MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy"]


def dataset(objective, rows=128, dimensions=3):
    rng = np.random.default_rng(921)
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    coefficients = rng.normal(size=(4, dimensions)).astype(np.float32) * 0.4
    signal = x @ coefficients
    if objective == "MultiRMSE":
        target = signal + np.linspace(2, 5, dimensions, dtype=np.float32) * np.where(np.arange(dimensions) % 2, -1, 1)
    elif objective == "RMSEWithUncertainty":
        target = 0.7 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(size=rows) * (0.1 + 0.15 * np.abs(x[:, 2]))
    elif objective == "MultiLogloss":
        target = signal > 0
    else:
        target = 1 / (1 + np.exp(-signal))
    return x, np.asarray(target, dtype=np.float32), np.linspace(0.2, 2, rows, dtype=np.float32)


def options(objective, **extra):
    result = dict(task_type="GPU", loss_function=objective, iterations=8, depth=3,
                  learning_rate=0.12, border_count=16, random_seed=39,
                  bootstrap_type="No", random_strength=0, score_function="Cosine",
                  leaf_estimation_iterations=2, leaf_estimation_backtracking="No",
                  verbose=False, allow_writing_files=False)
    result.update(extra)
    return result


def estimator(objective, **extra):
    cls = CatBoostClassifier if objective in {"MultiLogloss", "MultiCrossEntropy"} else CatBoostRegressor
    return cls(**options(objective, **extra))


def raw_prediction(model, data, **kwargs):
    return model.predict(data, prediction_type="RawFormulaVal", **kwargs)


def weighted_metric(objective, raw, target, weights):
    # Independent formulas for CatBoost's existing metric normalizations.
    if objective == "MultiRMSE":
        return np.sqrt(np.average(np.sum((raw - target) ** 2, axis=1), weights=weights))
    if objective == "RMSEWithUncertainty":
        loss = 0.5 * np.log(2 * np.pi) + raw[:, 1] + 0.5 * np.exp(-2 * raw[:, 1]) * (raw[:, 0] - target) ** 2
    else:
        loss = np.mean(np.logaddexp(0, raw) - target * raw, axis=1)
    return np.average(loss, weights=weights)


class StopAfter:
    def __init__(self, count):
        self.count = count

    def after_iteration(self, info):
        return info.iteration < self.count


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_native_vector_weighted_metrics_match_every_exported_iteration(objective, method):
    x, target, weights = dataset(objective)
    pool = Pool(x, target, weight=weights)
    model = estimator(objective, leaf_estimation_method=method).fit(pool, eval_set=pool, use_best_model=False)
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_all_params()["task_type"] == "GPU"
    assert model.get_all_params()["leaf_estimation_method"] == method
    history = model.get_evals_result()
    expected = [weighted_metric(objective, raw, target, weights)
                for raw in model.staged_predict(x, prediction_type="RawFormulaVal")]
    for token in ("learn", "validation"):
        np.testing.assert_allclose(history[token][objective], expected, rtol=4e-6, atol=4e-6)
    assert expected[-1] < expected[0]
    raw = raw_prediction(model, x)
    assert raw.shape == (len(x), 2 if objective == "RMSEWithUncertainty" else target.shape[1])
    np.testing.assert_allclose(np.asarray(model.get_test_evals()[0]).T, raw, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(raw_prediction(model, x, task_type="GPU"), raw, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("score", ["L2", "Cosine", "SolarL2", "LOOL2", "SatL2"])
def test_native_vector_all_supported_scores(score):
    x, target, weights = dataset("MultiRMSE")
    model = estimator("MultiRMSE", score_function=score, random_strength=0.5).fit(x, target, sample_weight=weights)
    assert model.get_all_params()["score_function"] == score
    history = model.get_evals_result()["learn"]["MultiRMSE"]
    assert np.isfinite(history).all()
    assert history[-1] < history[0]


@pytest.mark.parametrize("objective,bootstrap", list(zip(OBJECTIVES, ["No", "Bayesian", "Bernoulli", "Poisson"])))
def test_native_vector_snapshot_restores_exact_all_output_state(tmp_path, objective, bootstrap):
    x, target, weights = dataset(objective)
    extra = dict(iterations=9, bootstrap_type=bootstrap, random_strength=0.7,
                 leaf_estimation_backtracking="AnyImprovement")
    if bootstrap == "Bayesian":
        extra["bagging_temperature"] = 0.7
    elif bootstrap in {"Bernoulli", "Poisson"}:
        extra["subsample"] = 0.8
    saved = dict(extra, allow_writing_files=True, train_dir=str(tmp_path),
                 save_snapshot=True, snapshot_interval=0, snapshot_file="vector.snapshot")
    pool = Pool(x, target, weight=weights)
    partial = estimator(objective, **saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(4)])
    assert partial.tree_count_ == 4
    resumed = estimator(objective, **saved).fit(pool, eval_set=pool, use_best_model=False)
    direct = estimator(objective, **extra).fit(pool, eval_set=pool, use_best_model=False)
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_leaf_weights(), direct.get_leaf_weights())
    np.testing.assert_array_equal(raw_prediction(resumed, x), raw_prediction(direct, x))
    assert resumed.get_evals_result() == direct.get_evals_result()
    completed = estimator(objective, **saved).fit(pool, eval_set=pool, use_best_model=False)
    assert completed.tree_count_ == 9
    np.testing.assert_array_equal(raw_prediction(completed, x), raw_prediction(direct, x))
    changed = target.copy()
    changed.flat[0] = 1 - changed.flat[0] if objective.startswith("Multi") and objective != "MultiRMSE" else changed.flat[0] + 0.5
    with pytest.raises(CatBoostError, match="snapshot|Snapshot"):
        estimator(objective, **saved).fit(Pool(x, changed, weight=weights), eval_set=pool, use_best_model=False)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_native_vector_initial_model_and_per_output_baseline(objective):
    x, target, weights = dataset(objective)
    first = estimator(objective, iterations=3, boost_from_average=False).fit(x, target, sample_weight=weights)
    initial = raw_prediction(first, x)
    continued = estimator(objective, iterations=5, boost_from_average=False).fit(x, target, sample_weight=weights, init_model=first)
    assert continued.tree_count_ == 8
    pool = Pool(x, target, weight=weights, baseline=initial)
    baseline = estimator(objective, iterations=5, boost_from_average=False).fit(pool, eval_set=pool, use_best_model=False)
    np.testing.assert_allclose(initial + raw_prediction(baseline, x), raw_prediction(continued, x), rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(np.asarray(baseline.get_test_evals()[0]).T,
                               initial + raw_prediction(baseline, x), rtol=4e-6, atol=4e-6)
    expected = weighted_metric(objective, initial + raw_prediction(baseline, x), target, weights)
    for token in ("learn", "validation"):
        assert baseline.get_evals_result()[token][objective][-1] == pytest.approx(expected, rel=4e-6, abs=4e-6)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_native_vector_prediction_views_and_model_roundtrip(tmp_path, objective):
    x, target, weights = dataset(objective)
    model = estimator(objective).fit(x, target, sample_weight=weights)
    raw = raw_prediction(model, x)
    kinds = ["RawFormulaVal"]
    if objective in {"MultiLogloss", "MultiCrossEntropy"}:
        kinds += ["Probability", "LogProbability", "Class"]
        probability = 1 / (1 + np.exp(-raw))
        np.testing.assert_allclose(model.predict_proba(x), probability, rtol=3e-6, atol=3e-6)
        np.testing.assert_array_equal(model.predict(x), raw > 0)
    elif objective == "RMSEWithUncertainty":
        kinds += ["RMSEWithUncertainty"]
        expected = np.column_stack([raw[:, 0], np.exp(2 * raw[:, 1])])
        np.testing.assert_allclose(model.predict(x), expected, rtol=3e-6, atol=3e-6)
    else:
        np.testing.assert_array_equal(model.predict(x), raw)
    for kind in kinds:
        for begin, end in [(0, 0), (2, 6)]:
            kwargs = dict(prediction_type=kind, ntree_start=begin, ntree_end=end)
            np.testing.assert_allclose(model.predict(x[:23], task_type="GPU", **kwargs),
                                       model.predict(x[:23], **kwargs), rtol=4e-6, atol=4e-6)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("vector." + format_)
        model.save_model(path, format=format_)
        restored = estimator(objective).load_model(path, format=format_)
        if format_ == "cbm":
            np.testing.assert_array_equal(raw_prediction(restored, x), raw)
        else:
            np.testing.assert_allclose(raw_prediction(restored, x), raw, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(raw_prediction(restored, x, task_type="GPU"), raw, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("dimensions", [2, 64])
def test_native_multi_rmse_default_weighted_bias_and_quantized_outputs(tmp_path, dimensions):
    x, target, weights = dataset("MultiRMSE", rows=80, dimensions=dimensions)
    pool = Pool(x, target, weight=weights)
    pool.quantize(border_count=8)
    # CatBoost's on-disk quantized format does not support multidimensional
    # targets on either backend. Exercise the supported in-memory Pool path.
    restored_pool = pool
    model = estimator("MultiRMSE", iterations=3, depth=1, border_count=8).fit(restored_pool, eval_set=restored_pool, use_best_model=False)
    assert model.get_all_params()["boost_from_average"]
    expected_bias = np.average(target.astype(np.float64), axis=0, weights=weights.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(model.get_scale_and_bias()[1], expected_bias)
    raw = raw_prediction(model, x)
    assert raw.shape == (len(x), dimensions)
    np.testing.assert_allclose(raw_prediction(model, restored_pool), raw, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(raw_prediction(model, x, task_type="GPU"), raw, rtol=4e-6, atol=4e-6)
    expected = weighted_metric("MultiRMSE", raw, target, weights)
    for token in ("learn", "validation"):
        assert model.get_evals_result()[token]["MultiRMSE"][-1] == pytest.approx(expected, rel=4e-6, abs=4e-6)


@pytest.mark.parametrize("case,match", [
    ("one_output", "2.*64|at least.*2|multidimensional"),
    ("too_many_outputs", "2.*64|64.*output"),
    ("mvs", "MVS"),
    ("ordered", "(?i)Ordered"),
    ("categorical", "(?i)target|categorical|numeric"),
])
def test_native_vector_unsupported_combinations_fail_clearly(case, match):
    dimensions = 1 if case == "one_output" else 65 if case == "too_many_outputs" else 3
    x, target, weights = dataset("MultiRMSE", dimensions=dimensions)
    model = estimator("MultiRMSE", iterations=2)
    if case == "mvs":
        model.set_params(bootstrap_type="MVS", subsample=0.8)
    elif case == "ordered":
        model.set_params(boosting_type="Ordered")
    pool = Pool(x, target, weight=weights)
    if case == "categorical":
        mixed = np.empty((len(x), 2), dtype=object)
        mixed[:, 0] = x[:, 0]
        mixed[:, 1] = np.where(x[:, 1] > 0, "right", "left")
        pool = Pool(mixed, target, weight=weights, cat_features=[1])
    with pytest.raises(CatBoostError, match=match):
        model.fit(pool)


def test_native_uncertainty_accepts_scalar_target_and_returns_mean_variance():
    x, target, weights = dataset("RMSEWithUncertainty")
    assert target.ndim == 1
    model = estimator("RMSEWithUncertainty").fit(x, target, sample_weight=weights)
    prediction = model.predict(x, prediction_type="RMSEWithUncertainty", task_type="GPU")
    assert prediction.shape == (len(x), 2)
    assert np.isfinite(prediction).all()
    assert (prediction[:, 1] > 0).all()
