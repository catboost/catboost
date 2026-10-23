"""Native numeric greedy acceptance. Every fit uses standard task_type='GPU'.

Opt in with CATBOOST_NATIVE_METAL_TESTS=1 and PYTHONPATH pointing at the rebuilt
native CatBoost package. This suite targets the initial Plain, single-permutation
adapter. Independent objective, split, sampling and leaf equations live in the
private greedy runtime tests; no CPU model is fitted as an acceptance baseline.
"""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostError, CatBoostRegressor, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal CatBoost package with greedy integration",
)

POLICIES = ("Depthwise", "Lossguide", "Region")
SCORES = ("L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2", "SatL2")
OBJECTIVES = (
    ("RMSE", "Newton"), ("Logloss", "Newton"), ("CrossEntropy", "Newton"),
    ("Poisson", "Newton"), ("Huber:delta=1.2", "Newton"),
    ("Expectile:alpha=0.7", "Gradient"), ("Tweedie:variance_power=1.5", "Newton"),
    ("LogLinQuantile:alpha=0.7", "Gradient"), ("Quantile:alpha=0.7", "Gradient"),
    ("MAE", "Gradient"), ("MAPE", "Gradient"),
)


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def gpu_fit(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU", "Native greedy acceptance must not fit a CPU model"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", gpu_fit)


def dataset(rows=193):
    rng = np.random.default_rng(7184)
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    target = (1.5 * x[:, 0] + (x[:, 1] > .2) * x[:, 2] - .7 * x[:, 3]).astype(np.float32)
    weights = rng.uniform(.4, 2.3, rows).astype(np.float32)
    weights[::17] = 0
    return x, target, weights


def options(policy="Lossguide", **extra):
    result = dict(task_type="GPU", boosting_type="Plain", grow_policy=policy,
                  iterations=12, depth=4, learning_rate=.15, random_seed=47,
                  border_count=20, bootstrap_type="No", random_strength=0,
                  score_function="Cosine", leaf_estimation_backtracking="No",
                  boost_from_average=False, metric_period=1, verbose=False, allow_writing_files=False)
    if policy == "Lossguide":
        result["max_leaves"] = 7
    result.update(extra)
    return result


def regressor(**config):
    return CatBoostRegressor(**config).set_params(permutation_count=1)


def classifier(**config):
    return CatBoostClassifier(**config).set_params(permutation_count=1)


def sampler(kind):
    result = {"bootstrap_type": kind}
    if kind == "Bayesian":
        result["bagging_temperature"] = .7
    elif kind in ("Bernoulli", "Poisson"):
        result["subsample"] = .8
    return result


def check_model_metadata_and_weights(model, policy, weights):
    params = model.get_all_params()
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert params["task_type"] == "GPU"
    assert params["boosting_type"] == "Plain"
    assert params["grow_policy"] == policy
    # Shared CleanPlainJsonParams intentionally omits this option for numeric
    # Plain models. Check the executed runtime count and serialized input too.
    assert model.get_metadata()["metal_permutations"] == "1"
    assert json.loads(model.get_metadata()["params"])["boosting_options"]["permutation_count"] == 1
    leaf_counts = model.get_tree_leaf_counts()
    maximum = params["depth"] + 1 if policy == "Region" else 1 << params["depth"]
    if policy == "Lossguide":
        maximum = min(maximum, params["max_leaves"])
    assert len(leaf_counts) == model.tree_count_ and np.all((leaf_counts >= 1) & (leaf_counts <= maximum))
    leaf_weights = model.get_leaf_weights()
    assert len(leaf_weights) == int(leaf_counts.sum())
    assert np.isfinite(leaf_weights).all() and np.all(leaf_weights >= 0)
    offset = 0
    for count in leaf_counts:
        count = int(count)
        assert leaf_weights[offset:offset + count].sum() == pytest.approx(weights.sum(dtype=float), rel=4e-6, abs=1e-5)
        offset += count


class StopAfter:
    def __init__(self, count):
        self.count = count
        self.iterations = []

    def after_iteration(self, info):
        self.iterations.append(info.iteration)
        return info.iteration < self.count


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("objective,method", OBJECTIVES)
def test_native_greedy_weighted_cuda_registered_scalar_losses(policy, objective, method):
    x, target, weights = dataset()
    estimator = regressor
    if objective in ("Logloss", "CrossEntropy"):
        estimator = classifier
        target = ((target > 0).astype(np.float32) if objective == "Logloss" else
                  (1 / (1 + np.exp(-target))).astype(np.float32))
    elif objective.startswith(("Poisson", "Tweedie", "LogLinQuantile", "MAPE")):
        target = np.exp(target / 3).astype(np.float32)
    learn = Pool(x[:145], target[:145], weight=weights[:145])
    heldout = Pool(x[145:], target[145:], weight=weights[145:])
    model = estimator(**options(policy, loss_function=objective,
                                leaf_estimation_method=method, leaf_estimation_iterations=2))
    model.fit(learn, eval_set=heldout, use_best_model=False)
    check_model_metadata_and_weights(model, policy, weights[:145])
    assert model.tree_count_ == 12
    assert model.get_all_params()["leaf_estimation_method"] == method
    history = next(iter(model.get_evals_result()["learn"].values()))
    assert len(history) == 12 and np.isfinite(history).all() and history[-1] < history[0]
    raw = model.predict(x[145:], prediction_type="RawFormulaVal")
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(model.get_test_eval(), raw, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(model.predict(x[145:], prediction_type="RawFormulaVal", task_type="GPU"),
                               raw, rtol=4e-6, atol=4e-6)
    if objective == "RMSE":
        expected = np.sqrt(np.average((raw - target[145:])**2, weights=weights[145:]))
        assert model.get_evals_result()["validation"]["RMSE"][-1] == pytest.approx(expected, rel=4e-6)
    elif objective in ("Logloss", "CrossEntropy"):
        expected = np.average(np.logaddexp(0, raw) - target[145:] * raw, weights=weights[145:])
        assert model.get_evals_result()["validation"][objective][-1] == pytest.approx(expected, rel=4e-6)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("score", SCORES)
def test_native_greedy_all_seven_cuda_scores(policy, score):
    x, target, weights = dataset()
    model = regressor(**options(policy, score_function=score, leaf_estimation_method="Gradient",
                                 leaf_estimation_iterations=2, iterations=7)).fit(x, target, sample_weight=weights)
    check_model_metadata_and_weights(model, policy, weights)
    assert model.get_all_params()["score_function"] == score
    history = model.get_evals_result()["learn"]["RMSE"]
    assert np.isfinite(history).all() and history[-1] < history[0]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("kind", ("No", "Bayesian", "Bernoulli", "Poisson"))
def test_native_greedy_samplers_are_seeded_and_keep_original_leaf_weights(policy, kind):
    x, target, weights = dataset()
    config = options(policy, iterations=7, random_strength=.6, **sampler(kind))
    first = regressor(**config).fit(x, target, sample_weight=weights)
    second = regressor(**config).fit(x, target, sample_weight=weights)
    check_model_metadata_and_weights(first, policy, weights)
    assert first.get_all_params()["bootstrap_type"] == kind
    np.testing.assert_array_equal(first.get_tree_leaf_counts(), second.get_tree_leaf_counts())
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    np.testing.assert_array_equal(first.predict(x), second.predict(x))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("backtracking", ("AnyImprovement", "Armijo"))
def test_native_greedy_binary_backtracking(policy, backtracking):
    x, target, weights = dataset()
    labels = (target > 0).astype(np.float32)
    model = classifier(**options(policy, loss_function="Logloss", iterations=7,
        leaf_estimation_method="Newton", leaf_estimation_iterations=5,
        leaf_estimation_backtracking=backtracking, learning_rate=.3)).fit(x, labels, sample_weight=weights)
    check_model_metadata_and_weights(model, policy, weights)
    assert model.get_all_params()["leaf_estimation_backtracking"] == backtracking
    history = model.get_evals_result()["learn"]["Logloss"]
    assert np.isfinite(history).all() and history[-1] < history[0]
    np.testing.assert_allclose(model.predict_proba(x, task_type="GPU"), model.predict_proba(x), rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("objective,alpha", (("MAE", .5), ("Quantile:alpha=0.7", .7), ("MAPE", .5)))
def test_native_greedy_exact_weighted_residual_quantiles(policy, objective, alpha):
    target = np.array([-5, -2, 1, 3, 8, 13], np.float32)
    weights = np.array([0, 1, 2, 1, 4, 2], np.float32)
    x = np.arange(len(target), dtype=np.float32).reshape(-1, 1)
    initial = np.full(len(target), .75, np.float32)
    model = regressor(**options(policy, loss_function=objective, leaf_estimation_method="Exact",
        leaf_estimation_iterations=1, iterations=1, depth=0, learning_rate=.25, l2_leaf_reg=8))
    model.fit(Pool(x, target, weight=weights, baseline=initial))
    effective = weights / np.maximum(1, np.abs(target)) if objective == "MAPE" else weights
    selected = np.searchsorted(np.cumsum(effective), alpha * effective.sum(), side="left")
    expected = .25 * (target[selected] - .75)
    np.testing.assert_allclose(model.get_leaf_values(), [expected], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model.predict(x), expected, rtol=3e-6, atol=3e-6)
    check_model_metadata_and_weights(model, policy, weights)
    assert model.get_all_params()["leaf_estimation_method"] == "Exact"
    x, target, weights = dataset()
    fitted = regressor(**options(policy, loss_function=objective, leaf_estimation_method="Exact",
        leaf_estimation_iterations=1, iterations=7)).fit(x, target, sample_weight=weights)
    check_model_metadata_and_weights(fitted, policy, weights)
    history = next(iter(fitted.get_evals_result()["learn"].values()))
    assert np.isfinite(history).all() and history[-1] < history[0]


@pytest.mark.parametrize("policy,kind", (("Depthwise", "No"), ("Lossguide", "Bayesian"),
                                         ("Region", "Bernoulli"), ("Lossguide", "Poisson")))
def test_native_greedy_callback_snapshot_resume_preserves_sampling(tmp_path, policy, kind):
    x, target, weights = dataset()
    learn = Pool(x[:145], target[:145], weight=weights[:145])
    heldout = Pool(x[145:], target[145:], weight=weights[145:])
    common = options(policy, iterations=9, metric_period=3, random_strength=.6, **sampler(kind))
    saved = dict(common, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_interval=0, snapshot_file="greedy.snapshot")
    callback = StopAfter(4)
    partial = regressor(**saved).fit(learn, eval_set=heldout, use_best_model=False, callbacks=[callback])
    assert partial.tree_count_ == 4 and callback.iterations == [1, 2, 3, 4]
    assert (tmp_path / "greedy.snapshot").is_file()
    resumed = regressor(**saved).fit(learn, eval_set=heldout, use_best_model=False)
    direct = regressor(**common).fit(learn, eval_set=heldout, use_best_model=False)
    assert resumed.tree_count_ == direct.tree_count_ == 9
    np.testing.assert_array_equal(resumed.get_tree_leaf_counts(), direct.get_tree_leaf_counts())
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_leaf_weights(), direct.get_leaf_weights())
    np.testing.assert_array_equal(resumed.predict(x), direct.predict(x))
    np.testing.assert_array_equal(resumed.get_test_eval(), direct.get_test_eval())
    assert resumed.get_evals_result() == direct.get_evals_result()
    completed = regressor(**saved).fit(learn, eval_set=heldout, use_best_model=False)
    assert completed.tree_count_ == 9
    np.testing.assert_array_equal(completed.predict(x), resumed.predict(x))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("quantized", (False, True))
def test_native_greedy_quantized_pool_json_cbm_and_gpu_readers(tmp_path, policy, quantized):
    x, target, weights = dataset()
    names = ["a", "b", "c", "d"]
    pool = Pool(x, target, weight=weights, feature_names=names)
    if quantized:
        pool.quantize(border_count=20)
        path = tmp_path / "greedy.pool"
        pool.save(path)
        pool = Pool("quantized://" + str(path))
    model = regressor(**options(policy)).fit(pool)
    check_model_metadata_and_weights(model, policy, weights)
    assert model.feature_names_ == names
    prediction = model.predict(x, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(model.predict(pool), prediction, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(model.predict(pool, task_type="GPU"), prediction, rtol=4e-6, atol=4e-6)
    heldout = np.random.default_rng(43).normal(size=(37, 4)).astype(np.float32)
    expected = model.predict(heldout)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("greedy." + format_)
        model.save_model(path, format=format_)
        restored = regressor().load_model(path, format=format_)
        np.testing.assert_allclose(restored.predict(heldout), expected, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(restored.predict(heldout, task_type="GPU"), expected, rtol=4e-6, atol=4e-6)
        np.testing.assert_allclose(restored.predict(heldout, task_type="GPU", ntree_start=2, ntree_end=8),
            model.predict(heldout, ntree_start=2, ntree_end=8), rtol=4e-6, atol=4e-6)
        assert restored.feature_names_ == names
        assert restored.get_metadata()["metal_backend"] == "METAL"
        np.testing.assert_array_equal(restored.get_tree_leaf_counts(), model.get_tree_leaf_counts())
        if format_ == "json":
            document = json.loads(path.read_text())
            assert "trees" in document and "oblivious_trees" not in document
            assert len(document["trees"]) == model.tree_count_


@pytest.mark.parametrize("policy", POLICIES)
def test_native_greedy_string_labels_probability_and_saved_classes(tmp_path, policy):
    x, target, weights = dataset()
    labels = np.where(target > 0, "positive", "negative")
    model = classifier(**options(policy, loss_function="Logloss", eval_metric="Accuracy"))
    model.fit(x[:145], labels[:145], sample_weight=weights[:145],
              eval_set=Pool(x[145:], labels[145:], weight=weights[145:]), use_best_model=False)
    assert model.classes_.tolist() == ["negative", "positive"]
    probability = model.predict_proba(x[145:])
    np.testing.assert_allclose(probability.sum(axis=1), 1., atol=1e-7)
    np.testing.assert_allclose(model.predict_proba(x[145:], task_type="GPU"), probability, rtol=4e-6, atol=4e-6)
    predicted = np.asarray(model.predict(x[145:])).reshape(-1)
    expected_accuracy = np.average(predicted == labels[145:], weights=weights[145:])
    assert model.get_evals_result()["validation"]["Accuracy"][-1] == pytest.approx(expected_accuracy, rel=4e-6)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("binary." + format_)
        model.save_model(path, format=format_)
        restored = classifier().load_model(path, format=format_)
        np.testing.assert_array_equal(restored.classes_, model.classes_)
        np.testing.assert_allclose(restored.predict_proba(x[145:], task_type="GPU"), probability, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("policy", POLICIES)
def test_native_greedy_callback_initial_model_and_baseline(policy):
    x, target, weights = dataset()
    callback = StopAfter(4)
    first = regressor(**options(policy)).fit(x, target, sample_weight=weights, callbacks=[callback])
    assert first.tree_count_ == 4 and callback.iterations == [1, 2, 3, 4]
    initial = first.predict(x)
    continued = regressor(**options(policy, iterations=5)).fit(x, target, sample_weight=weights, init_model=first)
    assert continued.tree_count_ == 9
    assert np.average((continued.predict(x) - target)**2, weights=weights) < np.average((initial - target)**2, weights=weights)
    baseline_pool = Pool(x, target, weight=weights, baseline=initial)
    baseline = regressor(**options(policy, iterations=5)).fit(baseline_pool, eval_set=baseline_pool,
                                                            use_best_model=False)
    np.testing.assert_allclose(baseline.predict(x) + initial, continued.predict(x), rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(baseline.get_test_eval(), baseline.predict(x) + initial, rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("policy", POLICIES)
def test_native_greedy_multiple_eval_sets_early_stop_and_unlabeled_pool(policy):
    x, target, weights = dataset()
    model = regressor(**options(policy, iterations=40, early_stopping_rounds=4, metric_period=3,
                                use_best_model=True, best_model_min_trees=2))
    model.fit(Pool(x, target, weight=weights), eval_set=[Pool(x[:20]), Pool(x, -target, weight=weights)])
    assert 2 <= model.tree_count_ < 40
    assert model.get_best_iteration() >= 0
    evals = model.get_test_evals()
    assert len(evals) == 2
    np.testing.assert_allclose(evals[0][0], model.predict(x[:20]), rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(evals[1][0], model.predict(x), rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("nan_mode", ("Min", "Max"))
def test_native_greedy_nan_training_and_heldout_routing(policy, nan_mode):
    x, target, weights = dataset()
    x[::11, 0] = np.nan
    model = regressor(**options(policy, nan_mode=nan_mode, iterations=7)).fit(x[:145], target[:145], sample_weight=weights[:145])
    raw = model.predict(x[145:])
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(model.predict(x[145:], task_type="GPU"), raw, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("changed", ("target", "weight", "policy"))
def test_native_greedy_snapshot_rejects_incompatible_resume(tmp_path, changed):
    x, target, weights = dataset()
    common = options("Lossguide", iterations=7, allow_writing_files=True, train_dir=str(tmp_path),
                     save_snapshot=True, snapshot_interval=0, snapshot_file="greedy.snapshot")
    regressor(**common).fit(x, target, sample_weight=weights, callbacks=[StopAfter(3)])
    if changed == "target":
        target = target + 1
    elif changed == "weight":
        weights = weights.copy()
        weights[1] *= 2
    else:
        common["grow_policy"] = "Depthwise"
        common.pop("max_leaves")
    with pytest.raises(CatBoostError, match="snapshot|Snapshot|parameters|Parameters"):
        regressor(**common).fit(x, target, sample_weight=weights)


@pytest.mark.parametrize("extra,match", (({"loss_function": "Lq:q=2.5"}, "Lq"),
    ({"bootstrap_type": "MVS", "subsample": .8}, "MVS"),
    ({"boosting_type": "Ordered"}, "[Oo]rdered|Plain")))
def test_native_greedy_rejects_unregistered_or_unsupported_modes(extra, match):
    x, target, weights = dataset()
    with pytest.raises(CatBoostError, match=match):
        regressor(**options("Depthwise", iterations=2, **extra)).fit(x, target, sample_weight=weights)


@pytest.mark.parametrize("depth,positive", [(30, 40), (100, 100), (65535, 20)])
def test_native_region_deep_paths_and_snapshot_resume(tmp_path, depth, positive):
    x = np.zeros((2 * positive, positive), np.float32)
    x[np.arange(positive), np.arange(positive)] = 1
    y = np.r_[np.ones(positive), -np.ones(positive)].astype(np.float32)
    config = options("Region", iterations=2, depth=depth, border_count=1,
        l2_leaf_reg=0, learning_rate=.25, score_function="L2", allow_writing_files=True,
        train_dir=str(tmp_path), save_snapshot=True, snapshot_interval=0, snapshot_file="deep.snapshot")
    first = regressor(**config).fit(x, y, callbacks=[StopAfter(1)])
    assert first.get_tree_leaf_counts().tolist() == [min(depth, positive) + 1]
    resumed = regressor(**config).fit(x, y)
    full = regressor(**{**config, "save_snapshot": False, "allow_writing_files": False}).fit(x, y)
    np.testing.assert_array_equal(resumed.get_leaf_values(), full.get_leaf_values())
    np.testing.assert_array_equal(resumed.predict(x), full.predict(x))
    np.testing.assert_allclose(resumed.predict(x, task_type="GPU"), full.predict(x), atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("depth,capacity", [(100, 101), (2**32 - 1, 33), (2**32 - 1, 1)])
def test_native_lossguide_requested_depth_is_bounded_by_leaf_capacity(tmp_path, depth, capacity):
    # User-supplied quantization retains a real candidate with no occupied
    # right child. CUDA Lossguide's defined zero-gain winner extends a chain.
    borders = tmp_path / "borders.tsv"
    borders.write_text("0\t0.5\n")
    x = np.zeros((32, 1), np.float32)
    pool = Pool(x, np.ones(32, np.float32))
    pool.quantize(input_borders=str(borders))
    model = regressor(**options("Lossguide", depth=depth, max_leaves=capacity, iterations=2,
        score_function="NewtonL2", allow_const_label=True)).fit(pool)
    assert model.get_tree_leaf_counts().tolist() == [capacity, capacity]
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), model.predict(x), atol=2e-7, rtol=2e-7)
