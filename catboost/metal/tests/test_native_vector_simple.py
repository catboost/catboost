"""Native symmetric vector Simple leaves, without any CPU model fitting.

CUDA's greedy search helper handles all six symmetric vector objectives. Simple
exports sampled weak G/(W+L2), then copies the searched tree into every history.
The equations below are independent of the Metal leaf solver and model reader.
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from test_greedy_sampling import draws
from test_native_greedy_api import StopAfter


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal symmetric vector Simple adapter",
)

OBJECTIVES = (
    "MultiClass", "MultiClassOneVsAll", "MultiRMSE", "RMSEWithUncertainty",
    "MultiLogloss", "MultiCrossEntropy",
)


@pytest.fixture(autouse=True)
def only_gpu_fits(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(objective, **extra):
    return dict(task_type="GPU", loss_function=objective, boosting_type="Plain",
        data_partition="DocParallel", grow_policy="SymmetricTree", iterations=1,
        depth=1, learning_rate=.17, l2_leaf_reg=2.3, random_seed=857,
        random_strength=0., border_count=8, bootstrap_type="No", score_function="L2",
        boost_from_average=False, leaf_estimation_method="Simple",
        leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
        permutation_count=1, has_time=True, metric_period=1, verbose=False,
        allow_writing_files=False) | extra


def numeric_problem(objective):
    rng = np.random.default_rng(87351)
    x = rng.normal(size=(129, 4)).astype(np.float32)
    signal = np.column_stack((1.1 * x[:, 0] - .5 * x[:, 1],
                              .8 * x[:, 1] + .3 * x[:, 2],
                              -.7 * x[:, 0] - .4 * x[:, 2],
                              .6 * x[:, 3] + .15)).astype(np.float32)
    if objective in ("MultiClass", "MultiClassOneVsAll"):
        target = np.argmax(signal, axis=1)
    elif objective == "MultiRMSE":
        target = signal + np.array([1.3, -.7, .4, -.2], np.float32)
    elif objective == "RMSEWithUncertainty":
        target = signal[:, 0] + .2 * x[:, 3]
    elif objective == "MultiLogloss":
        target = (signal > np.array([.1, -.2, .3, .0])).astype(np.float32)
    else:
        target = (.07 + .86 / (1 + np.exp(-signal))).astype(np.float32)
    dimensions = 2 if objective == "RMSEWithUncertainty" else signal.shape[1]
    baseline = (.2 * rng.normal(size=(len(x), dimensions))).astype(np.float32)
    # Keep the public baseline in MultiClass's source C-1 gauge so its anchor
    # does not obscure the independently checked tree values and predictions.
    if objective == "MultiClass":
        baseline[:, -1] = 0
    weights = (.3 + np.arange(len(x)) % 11 / 5).astype(np.float32)
    weights[::13] = 0
    return x, target, weights, baseline


def gradients(objective, target, baseline, weights):
    """CUDA vector first derivatives, including the uncertainty natural gradient."""
    if objective == "MultiClass":
        exponential = np.exp(baseline - baseline.max(axis=1, keepdims=True))
        probability = exponential / exponential.sum(axis=1, keepdims=True)
        residual = np.eye(baseline.shape[1], dtype=np.float32)[target] - probability
        residual = residual[:, :-1]
    elif objective == "MultiClassOneVsAll":
        probability = np.clip(1 / (1 + np.exp(-baseline)), np.float32(1e-7),
                              np.float32(1) - np.float32(1e-7))
        residual = np.eye(baseline.shape[1], dtype=np.float32)[target] - probability
    elif objective == "MultiRMSE":
        residual = target - baseline
    elif objective == "RMSEWithUncertainty":
        error = np.float32(target - baseline[:, 0])
        variance = np.exp(np.minimum(np.float32(-2) * baseline[:, 1], np.float32(70)))
        residual = np.column_stack((error, np.float32(error * error * variance - 1)))
    else:
        residual = target - 1 / (1 + np.exp(-baseline))
    return np.float32(residual * weights[:, None])


def leaf_equations(objective, gradient, weights, ids, count, l2, learning_rate):
    masses = np.bincount(ids, weights=weights, minlength=count)
    sums = np.column_stack([
        np.bincount(ids, weights=gradient[:, dim], minlength=count)
        for dim in range(gradient.shape[1])
    ])
    values = np.float32(sums / (masses[:, None] + np.float32(l2)))
    values[masses <= 1e-20] = 0
    if objective == "MultiClass":
        # greedy_search_helper.cpp first stores float quotients, accumulates
        # those rounded C-1 coordinates into double totalSum, then rounds each
        # coordinate + totalSum back to float. The last public coordinate is 0.
        total = values.astype(np.float64).sum(axis=1, keepdims=True)
        values = np.float32(values.astype(np.float64) + total)
        values = np.column_stack((values, np.zeros(count, np.float32)))
    return np.float32(values * np.float32(learning_rate)), masses


def metric(objective, target, raw, weights):
    raw = np.asarray(raw, dtype=np.float64)
    if objective == "MultiClass":
        maximum = raw.max(axis=1)
        losses = maximum + np.log(np.exp(raw - maximum[:, None]).sum(axis=1))
        losses -= raw[np.arange(len(raw)), target]
    elif objective == "MultiClassOneVsAll":
        encoded = np.eye(raw.shape[1])[target]
        losses = (np.logaddexp(0, raw) - encoded * raw).mean(axis=1)
    elif objective == "MultiRMSE":
        return np.sqrt(np.average(np.square(target - raw).sum(axis=1), weights=weights))
    elif objective == "RMSEWithUncertainty":
        losses = .5 * np.log(2 * np.pi) + raw[:, 1]
        losses += .5 * np.exp(-2 * raw[:, 1]) * np.square(target - raw[:, 0])
    else:
        losses = (np.logaddexp(0, raw) - target * raw).mean(axis=1)
    return np.average(losses, weights=weights)


def exported(model, path):
    model.save_model(str(path), format="json")
    return json.loads(path.read_text())


def readers(model, data, tmp_path):
    expected = model.predict(data, prediction_type="RawFormulaVal", task_type="GPU")
    np.testing.assert_allclose(model.predict(data, prediction_type="RawFormulaVal"),
                               expected, rtol=5e-6, atol=8e-7)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("simple." + format_)
        model.save_model(str(path), format=format_)
        restored = CatBoost().load_model(str(path), format=format_)
        for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights"):
            actual, original = getattr(restored, method)(), getattr(model, method)()
            if format_ == "json" and method != "get_tree_leaf_counts":
                # The JSON writer preserves every double (also checked using
                # Python's parser); CatBoost's decimal reader can round the
                # same token to an adjacent float64 value. CBM and snapshot
                # assertions retain exact equality.
                field = "leaf_values" if method == "get_leaf_values" else "leaf_weights"
                document = json.loads(path.read_text())
                encoded = [value for tree in document["oblivious_trees"] for value in tree[field]]
                np.testing.assert_array_equal(encoded, original)
                np.testing.assert_array_max_ulp(actual, original, maxulp=1)
            else:
                np.testing.assert_array_equal(actual, original)
        np.testing.assert_allclose(
            restored.predict(data, prediction_type="RawFormulaVal", task_type="GPU"),
            expected, rtol=5e-6, atol=8e-7)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("depth,score,sampling", [(0, "L2", "No"), (1, "Cosine", "Bernoulli")])
def test_native_vector_simple_exports_sampled_gradient_leaf_equations(
        tmp_path, objective, depth, score, sampling):
    x, target, weights, baseline = numeric_problem(objective)
    config = options(objective, depth=depth, score_function=score, bootstrap_type=sampling)
    if sampling == "Bernoulli":
        config["subsample"] = .43
    pool = Pool(x, target, weight=weights, baseline=baseline)
    model = CatBoost(config).fit(pool, eval_set=pool, use_best_model=False)
    assert model.get_metadata()["metal_backend"] == "METAL"
    for key in ("task_type", "boosting_type", "data_partition", "grow_policy",
                "leaf_estimation_method", "leaf_estimation_iterations", "score_function"):
        assert model.get_all_params()[key] == config[key]
    tree, = exported(model, tmp_path / "equations.json")["oblivious_trees"]
    ids = np.zeros(len(x), np.uint32)
    assert len(tree.get("splits") or []) == depth
    for bit, split in enumerate(tree.get("splits") or []):
        assert split["split_type"] == "FloatFeature"
        ids |= (x[:, split["float_feature_index"]] > split["border"]).astype(np.uint32) << bit
    factors = draws(sampling, len(x), seed=config["random_seed"], subsample=.43)
    sampled_weights = np.float32(weights * factors)
    sampled_gradient = np.float32(gradients(objective, target, baseline, weights) * factors[:, None])
    expected, masses = leaf_equations(objective, sampled_gradient, sampled_weights, ids,
                                      1 << depth, config["l2_leaf_reg"], config["learning_rate"])
    np.testing.assert_allclose(tree["leaf_values"], expected.ravel(), rtol=7e-5, atol=5e-6)
    np.testing.assert_allclose(tree["leaf_weights"], masses, rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(model.get_leaf_values(), expected.ravel(), rtol=7e-5, atol=5e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), masses, rtol=5e-6, atol=5e-6)
    raw = np.float32(baseline + expected[ids])
    np.testing.assert_allclose(np.asarray(model.get_test_evals()[0]).T, raw, rtol=7e-5, atol=6e-6)
    np.testing.assert_allclose(
        model.predict(x, prediction_type="RawFormulaVal", task_type="GPU") + baseline,
        raw, rtol=7e-5, atol=6e-6)
    expected_metric = metric(objective, target, raw, weights)
    for token in ("learn", "validation"):
        assert model.get_evals_result()[token][objective][-1] == pytest.approx(
            expected_metric, rel=7e-6, abs=7e-6)
    if sampling != "No":
        assert not np.isclose(masses.sum(), weights.sum(dtype=np.float64), rtol=1e-3)
        original, _ = leaf_equations(objective, gradients(objective, target, baseline, weights),
                                     weights, ids, 1 << depth, config["l2_leaf_reg"], config["learning_rate"])
        assert np.max(np.abs(expected - original)) > 1e-4
    readers(model, x, tmp_path)


def categorical_problem(objective):
    rng = np.random.default_rng(648291)
    category = rng.permutation(np.tile(np.arange(12), 16))
    x = np.array([[f"kind-{value}"] for value in category], object)
    target = category % 3
    if objective == "RMSEWithUncertainty":
        target = (category.astype(np.float32) - 5.5) * .3 + .05 * rng.normal(size=len(x))
    weights = np.linspace(.2, 1.8, len(x), dtype=np.float32)
    weights[::17] = 0
    dimensions = 2 if objective == "RMSEWithUncertainty" else 3
    baseline = (.12 * rng.normal(size=(len(x), dimensions))).astype(np.float32)
    if objective == "MultiClass":
        baseline[:, -1] = 0
    return x, target, weights, baseline


def exact_forest(actual, expected):
    assert actual.tree_count_ == expected.tree_count_
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_evals"):
        np.testing.assert_array_equal(getattr(actual, method)(), getattr(expected, method)())
    assert actual.get_evals_result() == expected.get_evals_result()


@pytest.mark.parametrize("objective", ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty"))
@pytest.mark.parametrize("histories", (1, 4))
def test_native_vector_simple_categorical_histories_resume_exactly(tmp_path, objective, histories):
    x, target, weights, baseline = categorical_problem(objective)
    config = options(objective, iterations=5, depth=2, score_function="Cosine",
        bootstrap_type="Bayesian", bagging_temperature=1.3, random_strength=.35,
        permutation_count=histories, has_time=histories == 1, one_hot_max_size=1,
        max_ctr_complexity=1, simple_ctr=[
            ("FloatTargetMeanValue" if objective == "RMSEWithUncertainty" else "Borders")
            + ":CtrBorderCount=7:Prior=0.5"])
    pool = Pool(x, target, weight=weights, baseline=baseline, cat_features=[0])
    pool.quantize()
    heldout = x.copy()
    heldout[::19, 0] = "unseen-simple-category"
    evaluation = Pool(heldout, target, weight=weights, baseline=baseline, cat_features=[0])

    def trained(parameters, **extra):
        return CatBoost(parameters).fit(pool, eval_set=evaluation, use_best_model=False, **extra)

    direct = trained(config)
    assert direct.get_metadata()["metal_backend"] == "METAL"
    assert direct.get_metadata()["metal_permutations"] == str(histories)
    assert direct.get_all_params()["leaf_estimation_method"] == "Simple"
    document = exported(direct, tmp_path / "categorical.json")
    assert any(split["split_type"] == "OnlineCtr"
               for tree in document["oblivious_trees"] for split in tree.get("splits") or [])
    assert np.ptp(direct.get_leaf_values()) > 1e-5
    saved = config | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="simple.snapshot",
                          allow_writing_files=True, train_dir=str(tmp_path))
    callback = StopAfter(2)
    partial = trained(saved, callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    assert (tmp_path / "simple.snapshot").is_file()
    resumed = trained(saved)
    exact_forest(resumed, direct)
    exact_forest(trained(saved), direct)
    exact_forest(trained(saved | dict(iterations=7)), trained(config | dict(iterations=7)))
    np.testing.assert_allclose(np.asarray(resumed.get_test_evals()[0]).T,
        resumed.predict(heldout, prediction_type="RawFormulaVal", task_type="GPU") + baseline,
        rtol=7e-6, atol=7e-6)
    readers(resumed, heldout, tmp_path)


@pytest.mark.parametrize("objective,extra,diagnostic", [
    ("MultiClass", {"score_function": "NewtonL2"}, "(?i)(Newton|score)"),
    ("MultiRMSE", {"score_function": "NewtonCosine"}, "(?i)(Newton|score)"),
    ("MultiClassOneVsAll", {"leaf_estimation_iterations": 0}, "(?i)(Simple|estimation.*iteration)"),
    ("MultiCrossEntropy", {"leaf_estimation_iterations": 2}, "(?i)(Simple|estimation.*iteration)"),
])
def test_native_vector_simple_keeps_source_option_restrictions(objective, extra, diagnostic):
    x, target, weights, baseline = numeric_problem(objective)
    with pytest.raises(CatBoostError, match=diagnostic):
        CatBoost(options(objective, **extra)).fit(Pool(x, target, weight=weights, baseline=baseline))
