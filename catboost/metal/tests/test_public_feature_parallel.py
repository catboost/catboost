"""Public FeatureParallel acceptance. Every fit must execute the Metal backend.

Numeric online cursors are checked in original input order. Categorical models
use the existing independent original-row CTR oracle, including unseen tuples.
CPU model readers are checked, but no CPU model is fitted.
"""
import json
import os
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostError, CatBoostRanker, CatBoostRegressor, Pool
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRanker, CatBoostMetalRegressor
from catboost_metal._training import _shared_metric
from test_native_compound_ctrs import categorical_problem, check_final_tables, independent_prediction, projections
from test_native_greedy_api import OBJECTIVES, dataset
from test_ordered_query_public import fit_arguments, problem as query_problem


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1" or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires the rebuilt native Metal frontend and online cursor accessor")


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit
    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU", "Public acceptance must not fit CPU models"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(**extra):
    return dict(data_partition="FeatureParallel", iterations=3, depth=3, learning_rate=.17,
                l2_leaf_reg=2, border_count=12, permutation_count=4, min_fold_size=8,
                fold_len_multiplier=1.7, fold_permutation_block=3, score_function="Cosine",
                random_seed=619, random_strength=0, boost_from_average=False,
                bootstrap_type="No", leaf_estimation_iterations=2) | extra


def assert_numeric_cursor(model, x):
    raw = model.predict(x, prediction_type="RawFormulaVal", task_type="GPU")
    np.testing.assert_allclose(model.training_predictions_, raw, rtol=8e-6, atol=3e-6)
    assert np.isfinite(model.loss_history_).all()
    assert len(model.loss_history_) == model.training_stats_["completed_iterations"] + 1
    assert model.training_stats_["backend"] == "METAL"
    assert model.training_stats_["device"] and model.training_stats_["kernel_dispatches"] > 0
    assert model.training_stats_["gpu_seconds"] > 0


@pytest.mark.parametrize("objective,method", OBJECTIVES + (("Lq:q=3", "Newton"),))
@pytest.mark.parametrize("count", (1, 4))
def test_plain_fp_registered_scalar_objectives_have_original_order_cursors(objective, method, count):
    x, y, weights = dataset(97)
    kind = CatBoostMetalRegressor
    if objective in ("Logloss", "CrossEntropy"):
        kind = CatBoostMetalClassifier
        y = (y > 0).astype(np.float32) if objective == "Logloss" else (1 / (1 + np.exp(-y))).astype(np.float32)
    elif objective.startswith(("Poisson", "Tweedie", "LogLinQuantile", "MAPE")):
        y = np.exp(y / 3).astype(np.float32)
    model = kind(**options(loss_function=objective, leaf_estimation_method=method, permutation_count=count)).fit(x, y, weights)
    assert_numeric_cursor(model, x)
    assert model.feature_names_ == ["0", "1", "2", "3"]
    params = model.to_catboost().get_all_params()
    assert params["task_type"] == "GPU" and params["data_partition"] == "FeatureParallel"
    assert model.training_stats_["permutation_count"] == count


@pytest.mark.parametrize("objective", ("QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank"))
@pytest.mark.parametrize("count", (1, 4))
def test_plain_fp_query_objectives_preserve_groups_metrics_and_cursor(objective, count):
    data = query_problem(objective)
    x, y = data[:2]
    description = "YetiRank:permutations=4;decay=0.8" if objective == "YetiRank" else objective
    model = CatBoostMetalRanker(**options(loss_function=description, permutation_count=count,
        leaf_estimation_method="Newton", min_fold_size=2)).fit(x, None if objective == "PairLogit" else y,
                                                              **fit_arguments(data, objective))
    assert_numeric_cursor(model, x)
    np.testing.assert_array_equal(model.group_offsets_, data[-1])
    assert isinstance(model.to_catboost(), CatBoostRanker)
    if objective == "PairLogit":
        np.testing.assert_array_equal(model.pairs_, data[6])
        np.testing.assert_array_equal(model.pairs_weight_, data[7])
    if objective == "YetiRank":
        assert "PFound" in model.get_evals_result()["learn"]
    metric_name = "PFound" if objective == "YetiRank" else objective
    pair_data = (data[6][:, 0], data[6][:, 1], data[7]) if objective == "PairLogit" else None
    expected = _shared_metric(metric_name, model.training_predictions_, y,
        None if objective == "PairLogit" else data[3] * data[4], data[-1], pairs=pair_data,
        subgroup_hashes=data[5])
    assert model.loss_history_[-1] == pytest.approx(expected, rel=8e-6, abs=2e-6)


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("ctr", ("Borders", "Buckets", "FloatTargetMeanValue", "FeatureFreq"))
def test_compound_public_models_match_independent_ctr_tables_and_readers(tmp_path, boosting, count, ctr):
    x, y, future, pool_options = categorical_problem()
    model = CatBoostMetalRegressor(**options(boosting_type=boosting, permutation_count=count,
        max_ctr_complexity=2, one_hot_max_size=2, ctr_type=ctr, cat_features=[0, 1],
        depth=4, iterations=8, model_size_reg=0, learning_rate=.2, min_fold_size=16,
        border_count=16)).fit(x, y, pool_options["weight"])
    assert model.training_stats_["tree_ctr_features"] > 0
    path = tmp_path / "compound.json"
    model.save_model(path, format="json")
    document = json.loads(path.read_text())
    assert any(len(projection) > 1 for projection in projections(document))
    check_final_tables(document, x, y)
    expected = independent_prediction(document, x, y, future)
    np.testing.assert_allclose(model.predict(future, task_type="GPU"), expected, rtol=8e-6, atol=2e-6)
    for file_format in ("cbm", "json"):
        path = tmp_path / ("compound." + file_format)
        model.save_model(path, format=file_format)
        loaded = CatBoostRegressor().load_model(path, format=file_format)
        for task in ("CPU", "GPU"):
            np.testing.assert_allclose(loaded.predict(future, task_type=task), expected, rtol=8e-6, atol=2e-6)
            np.testing.assert_allclose(loaded.predict(future, task_type=task, ntree_start=1, ntree_end=3),
                model.predict(future, task_type="GPU", ntree_start=1, ntree_end=3), rtol=8e-6, atol=2e-6)


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_three_category_dynamic_complexity_is_forwarded(tmp_path, boosting):
    x, y, future, pool_options = categorical_problem(3)
    model = CatBoostMetalRegressor(**options(boosting_type=boosting, max_ctr_complexity=3, model_size_reg=0,
        iterations=12, depth=5, cat_features=[0, 1, 2], one_hot_max_size=2,
        learning_rate=.2, min_fold_size=16, border_count=16)).fit(x, y, pool_options["weight"])
    path = tmp_path / "complexity3.json"
    model.save_model(path, format="json")
    document = json.loads(path.read_text())
    assert any(len(projection) == 3 for projection in projections(document))
    np.testing.assert_allclose(model.predict(future, task_type="GPU"), independent_prediction(document, x, y, future),
                               rtol=8e-6, atol=2e-6)


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_compound_callbacks_snapshots_and_completed_resume_are_exact(tmp_path, boosting):
    x, y, future, pool_options = categorical_problem()
    config = options(boosting_type=boosting, iterations=5, max_ctr_complexity=2,
                     cat_features=[0, 1], one_hot_max_size=2, model_size_reg=0)
    kwargs = dict(eval_set=(future[:12], np.resize(y, 12)), use_best_model=False)
    full = CatBoostMetalRegressor(**config).fit(x, y, pool_options["weight"], **kwargs)
    seen = []
    def callback(info):
        seen.append(info.iteration)
        return info.iteration < 2
    saved = dict(snapshot_file=tmp_path / "fit.cbsnapshot", snapshot_interval=0)
    partial = CatBoostMetalRegressor(**config).fit(x, y, pool_options["weight"], **kwargs, **saved, callback=callback)
    assert partial.tree_count_ == 2 and seen == [1, 2]
    resumed = CatBoostMetalRegressor(**config).fit(x, y, pool_options["weight"], **kwargs, **saved)
    assert resumed.training_stats_["resumed_iterations"] == 2
    np.testing.assert_array_equal(resumed.training_predictions_, full.training_predictions_)
    np.testing.assert_array_equal(resumed.to_catboost().get_leaf_values(), full.to_catboost().get_leaf_values())
    assert resumed.get_evals_result() == full.get_evals_result()
    complete = CatBoostMetalRegressor(**config).fit(x, y, pool_options["weight"], **kwargs, **saved)
    assert complete.training_stats_["resumed_iterations"] == 5
    np.testing.assert_array_equal(complete.training_predictions_, full.training_predictions_)
    with pytest.raises(CatBoostError, match="snapshot|Snapshot|different|checksum|parameters"):
        CatBoostMetalRegressor(**(config | {"model_size_reg": 1})).fit(x, y, pool_options["weight"], **kwargs, **saved)
    fresh = CatBoostMetalRegressor(**config).fit(x, y, pool_options["weight"], **kwargs, **saved, resume=False)
    assert fresh.training_stats_["resumed_iterations"] == 0


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_retained_best_cursor_and_snapshot_match_retained_numeric_model(tmp_path, boosting):
    x, y, weights = dataset(129)
    config = options(boosting_type=boosting, max_ctr_complexity=2, iterations=8)
    kwargs = dict(eval_set=(x, -y, weights), use_best_model=True, early_stopping_rounds=2)
    saved = dict(snapshot_file=tmp_path / "best.cbsnapshot", snapshot_interval=0)
    fitted = CatBoostMetalRegressor(**config).fit(x, y, weights, **kwargs, **saved)
    assert fitted.tree_count_ == fitted.get_best_iteration() + 1 < config["iterations"]
    assert_numeric_cursor(fitted, x)
    resumed = CatBoostMetalRegressor(**config).fit(x, y, weights, **kwargs, **saved)
    np.testing.assert_array_equal(resumed.training_predictions_, fitted.training_predictions_)
    assert resumed.tree_count_ == fitted.tree_count_


def test_string_class_order_and_named_feature_weights_are_preserved():
    x, y, weights = dataset(97)
    labels = np.where(y > 0, "yes", "no")
    pool = Pool(x, labels, weight=weights, feature_names=["first", "second", "third", "fourth"])
    model = CatBoostMetalClassifier(**options(class_weights={"yes": 2, "no": 1},
                                              feature_weights={"first": 2, 3: .2})).fit(pool)
    np.testing.assert_array_equal(model.classes_, ["no", "yes"])
    native = model.to_catboost()
    assert isinstance(native, CatBoostClassifier)
    np.testing.assert_array_equal(native.classes_, model.classes_)
    probability = model.predict(pool, prediction_type="Probability", task_type="GPU")
    np.testing.assert_array_equal(model.predict(pool), model.classes_[probability.argmax(axis=1)])
    assert model.feature_names_ == pool.get_feature_names()
    params = json.loads(native.get_metadata()["params"])
    feature_weights = params["tree_learner_options"]["penalties"]["feature_weights"]
    assert feature_weights["0"] == 2
    assert feature_weights["3"] == pytest.approx(.2)


def test_native_init_model_and_pool_baseline_preserve_default_intercept_semantics():
    x, y, weights = dataset(97)
    config = options(iterations=2)
    config.pop("boost_from_average")
    initial = CatBoostMetalRegressor(**config).fit(x, y, weights)
    continued = CatBoostMetalRegressor(**config).fit(x, y, weights, init_model=initial)
    assert continued.tree_count_ == 4
    assert_numeric_cursor(continued, x)
    baseline = np.linspace(-.3, .5, len(x), dtype=np.float32)
    with_baseline = CatBoostMetalRegressor(**config).fit(Pool(x, y, weight=weights, baseline=baseline))
    np.testing.assert_allclose(with_baseline.training_predictions_,
        with_baseline.predict(x, task_type="GPU") + baseline, rtol=8e-6, atol=2e-6)


def test_objective_history_is_distinct_from_same_family_eval_metric():
    x, y, weights = dataset(97)
    model = CatBoostMetalRegressor(**options(loss_function="Quantile:alpha=0.7",
        eval_metric="Quantile:alpha=0.2", leaf_estimation_method="Gradient")).fit(x, y, weights)
    history = model.get_evals_result()["learn"]
    assert history["Quantile:alpha=0.7"] != history["Quantile:alpha=0.2"]
    assert model.loss_history_[1:] == history["Quantile:alpha=0.7"]
    expected_initial = _shared_metric("Quantile:alpha=0.7", np.zeros(len(x)), y, weights)
    assert model.loss_history_[0] == pytest.approx(expected_initial, rel=3e-6)


@pytest.mark.parametrize("quantized", (False, True))
@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_pool_timestamps_restore_original_cursor_order_and_keep_feature_names(quantized, boosting):
    x, y, weights = dataset(97)
    names = ["signal", "threshold", "interaction", "nuisance"]
    pool = Pool(x, y, weight=weights, feature_names=names,
                timestamp=np.random.default_rng(971).permutation(len(x)).astype(np.uint64))
    if quantized:
        pool.quantize(border_count=12)
    model = CatBoostMetalRegressor(**options(boosting_type=boosting, max_ctr_complexity=2,
                                              permutation_count=1)).fit(pool)
    assert model.feature_names_ == names
    assert_numeric_cursor(model, x)


@pytest.mark.parametrize("objective", ("RMSE", "QueryRMSE", "QuerySoftMax", "PairLogit"))
@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_public_simple_leaves_follow_registered_fp_gradient_single_step(objective, boosting):
    config = options(boosting_type=boosting, loss_function=objective, leaf_estimation_iterations=1, max_ctr_complexity=2)
    if objective == "RMSE":
        x, y, weights = dataset(97)
        kind, args = CatBoostMetalRegressor, dict(sample_weight=weights)
    else:
        data = query_problem(objective)
        x, y = data[:2]
        y = None if objective == "PairLogit" else y
        kind, args = CatBoostMetalRanker, fit_arguments(data, objective)
    simple = kind(**(config | {"leaf_estimation_method": "Simple"})).fit(x, y, **args)
    gradient = kind(**(config | {"leaf_estimation_method": "Gradient"})).fit(x, y, **args)
    np.testing.assert_array_equal(simple.to_catboost().get_leaf_values(), gradient.to_catboost().get_leaf_values())
    np.testing.assert_array_equal(simple.training_predictions_, gradient.training_predictions_)
