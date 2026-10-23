"""Learned simple Borders priors, independent likelihood oracle and lifecycle.

Every CatBoost fit uses Metal. The independent oracle maximizes the beta-binomial
likelihood with SciPy, rather than reproducing CUDA's 50-step Newton routine.
Category counts use all original learn rows, including zero-weight observations.
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRegressor, Pool
from scipy.optimize import minimize
from scipy.special import betaln, digamma

from test_native_compound_ctrs import check_final_tables, independent_prediction


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires rebuilt native Metal automatic CTR priors",
)

PATHS = ("plain-doc", "plain-feature", "ordered-feature")
AUTO_CTR = "Borders:Prior=0.125/0.5:PriorEstimation=BetaPrior:CtrBorderType=Uniform:CtrBorderCount=15"


@pytest.fixture(autouse=True)
def require_gpu_training(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def problem():
    counts = np.array([3, 8, 13, 21, 29, 35])
    x = np.array([[f"category-{category}"] for category in range(len(counts)) for _ in range(40)], object)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(40 - k)] for k in counts]).astype(np.float32)
    order = np.random.default_rng(1103).permutation(len(y))
    x, y = x[order], y[order]
    weights = np.linspace(.3, 2., len(y), dtype=np.float32)
    weights[::13] = 0
    future = np.array([[f"category-{i}"] for i in range(len(counts))] + [["unseen"]], object)
    return x, y, weights, future


def beta_binomial_prior(x, y):
    _, inverse = np.unique(x[:, 0], return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    positives = np.bincount(inverse, weights=y).astype(float)

    def objective(log_parameters):
        alpha, beta = np.exp(log_parameters)
        # Binomial coefficients are constant with respect to the prior.
        value = -np.sum(betaln(positives + alpha, counts - positives + beta) - betaln(alpha, beta))
        common = digamma(counts + alpha + beta)
        da = np.sum(digamma(positives + alpha) - common + digamma(alpha + beta) - digamma(alpha))
        db = np.sum(digamma(counts - positives + beta) - common + digamma(alpha + beta) - digamma(beta))
        return value, -np.array([alpha * da, beta * db])

    optimum = minimize(objective, np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-10})
    assert optimum.success and np.linalg.norm(optimum.jac) < 1e-7, optimum
    alpha, beta = np.exp(optimum.x)
    assert alpha > 0 and beta > 0
    return np.array([alpha, alpha + beta], np.float32)


def options(path="plain-doc", count=1, **extra):
    result = dict(
        task_type="GPU", loss_function="RMSE", boosting_type="Ordered" if path == "ordered-feature" else "Plain",
        data_partition="DocParallel" if path == "plain-doc" else "FeatureParallel",
        iterations=7, depth=3, learning_rate=.2, l2_leaf_reg=2, boost_from_average=False,
        score_function="Cosine", leaf_estimation_method="Newton", leaf_estimation_iterations=1,
        leaf_estimation_backtracking="No", bootstrap_type="No", random_strength=0, random_seed=83,
        one_hot_max_size=2, max_ctr_complexity=1, simple_ctr=[AUTO_CTR], ctr_target_border_count=1,
        ctr_history_unit="Sample", counter_calc_method="SkipTest", model_size_reg=0,
        permutation_count=count, has_time=count == 1, border_count=16,
        allow_writing_files=False, verbose=False, metric_period=1,
    )
    if path == "ordered-feature":
        result.update(min_fold_size=16, fold_len_multiplier=1.7, fold_permutation_block=3)
    return result | extra


def fit(config, pool, **kwargs):
    return CatBoostRegressor().set_params(**config).fit(pool, **kwargs)


def categorical_params(model):
    assert model.get_metadata()["metal_backend"] == "METAL"
    return json.loads(model.get_metadata()["params"])["cat_feature_params"]


def learned_prior(model, feature=0, description=0):
    item = categorical_params(model)["per_feature_ctrs"][str(feature)][description]
    assert item["prior_estimation"] == "BetaPrior"
    assert len(item["priors"]) == 1
    return np.asarray(item["priors"][0], dtype=np.float32)


def explicit_ctr(prior, border_count=15):
    numerator, denominator = map(float, prior)
    return (f"Borders:Prior={numerator:.17g}/{denominator:.17g}:PriorEstimation=No:"
            f"CtrBorderType=Uniform:CtrBorderCount={border_count}")


def raw(model, data, task="GPU"):
    return model.predict(data, prediction_type="RawFormulaVal", task_type=task)


def compare_forests(actual, expected, future, exact=False):
    np.testing.assert_array_equal(actual.get_tree_leaf_counts(), expected.get_tree_leaf_counts())
    check = np.testing.assert_array_equal if exact else lambda a, b: np.testing.assert_allclose(a, b, atol=4e-6, rtol=5e-6)
    for method in ("get_leaf_values", "get_leaf_weights"):
        check(getattr(actual, method)(), getattr(expected, method)())
    check(raw(actual, future), raw(expected, future))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("quantized", (False, True))
def test_learned_prior_matches_independent_likelihood_and_final_tables(tmp_path, path, count, quantized):
    x, y, weights, future = problem()
    pool = Pool(x, y, cat_features=[0], weight=weights)
    if quantized:
        pool.quantize(border_count=16)
    evaluation = Pool(future, np.resize(y, len(future)), cat_features=[0])
    config = options(path, count)
    model = fit(config, pool, eval_set=evaluation, use_best_model=False)
    expected = beta_binomial_prior(x, y)
    actual = learned_prior(model)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert not np.allclose(actual, [.125, .5])
    explicit = fit(config | dict(simple_ctr=[explicit_ctr(actual)]), pool,
                   eval_set=evaluation, use_best_model=False)
    compare_forests(model, explicit, future)
    for fmt in ("json", "cbm"):
        output = tmp_path / ("prior." + fmt)
        model.save_model(output, format=fmt)
        restored = CatBoostRegressor().load_model(output, format=fmt)
        np.testing.assert_array_equal(learned_prior(restored), actual)
        if fmt == "json":
            document = json.loads(output.read_text())
            ctrs = document["features_info"]["ctrs"]
            assert ctrs
            for ctr in ctrs:
                np.testing.assert_allclose([ctr["prior_numerator"], ctr["prior_denomerator"]],
                                           actual, atol=2e-6, rtol=2e-6)
            check_final_tables(document, x, y)
            prediction = independent_prediction(document, x, y, future)
        else:
            prediction = raw(model, future)
        for task in ("GPU", "CPU"):
            np.testing.assert_allclose(raw(restored, future, task), prediction, atol=4e-6, rtol=5e-6)


@pytest.mark.parametrize("history", ("Sample", "Group"))
@pytest.mark.parametrize("count", (1, 4))
def test_prior_uses_full_unweighted_learn_rows_independently_of_history(history, count):
    x, y, weights, _ = problem()
    group_id = np.repeat(np.arange(len(y) // 4), 4)
    group_weights = np.repeat(np.linspace(.2, 3., len(y) // 4), 4).astype(np.float32)
    expected = beta_binomial_prior(x, y)
    for pool in (
        Pool(x, y, cat_features=[0], weight=weights),
        Pool(x, y, cat_features=[0], group_id=group_id, group_weight=group_weights),
    ):
        model = fit(options("ordered-feature", count, ctr_history_unit=history, iterations=2), pool)
        np.testing.assert_allclose(learned_prior(model), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("global_auto", (False, True))
def test_per_feature_overrides_use_flat_feature_indices(global_auto):
    x, y, weights, _ = problem()
    second = np.array([[f"pair-{int(value.rsplit('-', 1)[1]) // 2}"] for value in x[:, 0]], object)
    data = np.column_stack((np.arange(len(y)) % 5, x[:, 0], second[:, 0])).astype(object)
    common = explicit_ctr([2., 3.])
    config = options(iterations=2, simple_ctr=[AUTO_CTR if global_auto else common],
                     per_feature_ctr=["2:" + (common if global_auto else AUTO_CTR)])
    model = fit(config, Pool(data, y, cat_features=[1, 2], weight=weights))
    descriptions = categorical_params(model)["per_feature_ctrs"]
    if global_auto:
        np.testing.assert_allclose(learned_prior(model, 1), beta_binomial_prior(x, y), atol=2e-6, rtol=2e-6)
        assert descriptions["2"][0]["prior_estimation"] == "No"
        np.testing.assert_array_equal(descriptions["2"][0]["priors"], [[2., 3.]])
    else:
        assert "1" not in descriptions
        np.testing.assert_allclose(learned_prior(model, 2), beta_binomial_prior(second, y), atol=2e-6, rtol=2e-6)


def test_one_auto_description_replaces_all_borders_priors_on_that_feature():
    x, y, weights, _ = problem()
    model = fit(options(iterations=2, simple_ctr=[AUTO_CTR, explicit_ctr([2., 3.], border_count=7)]),
                Pool(x, y, cat_features=[0], weight=weights))
    descriptions = categorical_params(model)["per_feature_ctrs"]["0"]
    assert [item["prior_estimation"] for item in descriptions] == ["BetaPrior", "No"]
    for item in descriptions:
        np.testing.assert_allclose(item["priors"], [beta_binomial_prior(x, y)], atol=2e-6, rtol=2e-6)


def test_multiple_actual_target_borders_keep_configured_prior_like_cuda():
    x, _, weights, future = problem()
    y = (np.arange(len(x)) % 5).astype(np.float32)
    config = options(iterations=3, ctr_target_border_count=2)
    pool = Pool(x, y, cat_features=[0], weight=weights)
    model = fit(config, pool)
    explicit = fit(config | dict(simple_ctr=[explicit_ctr([.125, .5])]), pool)
    assert not categorical_params(model).get("per_feature_ctrs")
    compare_forests(model, explicit, future)


def test_configured_multiple_borders_with_binary_targets_rejects_auto_estimation():
    x, y, weights, _ = problem()
    with pytest.raises(CatBoostError, match="(?i)prior.*(border|target)|ctr_target_border_count"):
        fit(options(iterations=1, ctr_target_border_count=2), Pool(x, y, cat_features=[0], weight=weights))


@pytest.mark.parametrize("kind", ("Buckets", "FloatTargetMeanValue", "FeatureFreq"))
def test_other_ctr_types_reject_prior_estimation(kind):
    x, y, _, _ = problem()
    with pytest.raises(CatBoostError, match="(?i)prior estimation.*(not available|type)|prior.*unsupported"):
        fit(options(iterations=1, simple_ctr=[f"{kind}:PriorEstimation=BetaPrior"]), Pool(x, y, cat_features=[0]))


def test_compound_ctr_prior_estimation_remains_unsupported():
    x, y, _, _ = problem()
    with pytest.raises(CatBoostError, match="(?i)prior estimation.*combinations|prior.*unsupported"):
        fit(options("plain-feature", iterations=1, max_ctr_complexity=2,
                    combinations_ctr=["Borders:PriorEstimation=BetaPrior"]), Pool(x, y, cat_features=[0]))


def test_degenerate_target_rejects_undefined_beta_prior():
    x, y, _, _ = problem()
    with pytest.raises(CatBoostError, match="(?i)point|prior|target.*class"):
        fit(options(iterations=1, allow_const_label=True), Pool(x, np.zeros_like(y), cat_features=[0]))


class StopAfter:
    def __init__(self, stop=None):
        self.stop = stop
        self.iterations = []

    def after_iteration(self, info):
        self.iterations.append(info.iteration)
        return self.stop is None or info.iteration < self.stop


@pytest.mark.parametrize("path", PATHS)
def test_automatic_priors_snapshot_exact_continuation_and_target_identity(tmp_path, path):
    x, y, weights, future = problem()
    pool = Pool(x, y, cat_features=[0], weight=weights)
    config = options(path, count=4)
    snapshot = config | dict(save_snapshot=True, snapshot_file="priors.snapshot", snapshot_interval=0,
                             train_dir=str(tmp_path), allow_writing_files=True)
    stopped = fit(snapshot, pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(3)])
    assert stopped.tree_count_ == 3
    recorder = StopAfter()
    resumed = fit(snapshot, pool, eval_set=pool, use_best_model=False, callbacks=[recorder])
    direct = fit(config, pool, eval_set=pool, use_best_model=False)
    assert recorder.iterations == [4, 5, 6, 7]
    compare_forests(resumed, direct, future, exact=True)
    assert resumed.get_evals_result() == direct.get_evals_result()
    np.testing.assert_array_equal(learned_prior(resumed), learned_prior(direct))
    complete = fit(snapshot, pool, eval_set=pool, use_best_model=False)
    compare_forests(complete, direct, future, exact=True)
    saved = (tmp_path / "priors.snapshot").read_bytes()
    changed = y.copy()
    changed[0] = 1 - changed[0]
    with pytest.raises(CatBoostError, match="(?i)snapshot|match|parameter|checksum"):
        fit(snapshot, Pool(x, changed, cat_features=[0], weight=weights), eval_set=pool, use_best_model=False)
    assert (tmp_path / "priors.snapshot").read_bytes() == saved


@pytest.mark.parametrize("path", PATHS)
def test_baseline_and_initial_model_do_not_replace_targets_used_for_estimation(path):
    x, y, weights, future = problem()
    config = options(path, iterations=3)
    pool = Pool(x, y, cat_features=[0], weight=weights)
    initial = fit(config, pool)
    continued = fit(config, pool, init_model=initial)
    baseline = raw(initial, x)
    baseline_pool = Pool(x, y, cat_features=[0], weight=weights, baseline=baseline)
    from_baseline = fit(config, baseline_pool)
    expected = beta_binomial_prior(x, y)
    for model in (initial, continued, from_baseline):
        np.testing.assert_allclose(learned_prior(model), expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(raw(from_baseline, future) + raw(initial, future),
                               raw(continued, future), atol=4e-6, rtol=5e-6)


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
def test_greedy_training_reuses_resolved_simple_priors(policy):
    x, y, weights, future = problem()
    config = options(iterations=3, grow_policy=policy)
    pool = Pool(x, y, cat_features=[0], weight=weights)
    model = fit(config, pool)
    prior = learned_prior(model)
    np.testing.assert_allclose(prior, beta_binomial_prior(x, y), atol=2e-6, rtol=2e-6)
    explicit = fit(config | dict(simple_ctr=[explicit_ctr(prior)]), pool)
    compare_forests(model, explicit, future)
