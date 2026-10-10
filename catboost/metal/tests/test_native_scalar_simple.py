"""Native symmetric Simple leaves, checked against scalar derivative equations.

DocParallel exports the sampled weak score statistics. FeatureParallel Plain
and Ordered refit one Gradient step on each history's original observations.
Every fit uses the rebuilt GPU API; no CPU fit supplies an oracle.
"""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from cuda_scalar_reference import objective_terms
from test_greedy_sampling import draws
from test_native_greedy_api import StopAfter


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal scalar Simple adapter",
)

MODES = ("PlainDP", "PlainFP", "OrderedFP")
OBJECTIVES = (
    ("RMSE", None), ("Logloss", None), ("CrossEntropy", None),
    ("Poisson", None), ("Huber:delta=1.2", 1.2),
    ("Expectile:alpha=0.7", .7), ("Lq:q=2.5", 2.5),
    ("Tweedie:variance_power=1.5", 1.5),
    ("LogLinQuantile:alpha=0.7", .7), ("Quantile:alpha=0.7", .7),
    ("MAE", None), ("MAPE", None),
)


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(mode, objective="RMSE", **extra):
    config = dict(task_type="GPU", loss_function=objective,
        boosting_type="Ordered" if mode == "OrderedFP" else "Plain",
        data_partition="DocParallel" if mode == "PlainDP" else "FeatureParallel",
        grow_policy="SymmetricTree", iterations=2, depth=2, learning_rate=.15,
        l2_leaf_reg=2.3, random_seed=857, random_strength=0., border_count=8,
        bootstrap_type="No", score_function="Cosine", boost_from_average=False,
        leaf_estimation_method="Simple", leaf_estimation_iterations=1,
        leaf_estimation_backtracking="No", permutation_count=1, has_time=True,
        metric_period=1, verbose=False, allow_writing_files=False)
    if mode == "OrderedFP":
        config.update(min_fold_size=8, fold_len_multiplier=1.7, fold_permutation_block=3)
    return config | extra


def numeric_problem(objective):
    rng = np.random.default_rng(614)
    x = rng.normal(size=(97, 3)).astype(np.float32)
    signal = (1.3 * x[:, 0] - .5 * x[:, 1] + .25 * x[:, 2]).astype(np.float32)
    target = signal.copy()
    if objective == "Logloss":
        target = (signal > .1).astype(np.float32)
    elif objective == "CrossEntropy":
        target = (.05 + .9 / (1 + np.exp(-signal))).astype(np.float32)
    elif objective.startswith(("Poisson", "Tweedie", "LogLinQuantile", "MAPE")):
        target = np.exp(signal / 3).astype(np.float32)
    weights = (.3 + np.arange(len(x)) % 11 / 5).astype(np.float32)
    weights[::13] = 0
    baseline = np.linspace(-.7, .4, len(x), dtype=np.float32)
    return x, target, weights, baseline


def fit(config, pool, **kwargs):
    return CatBoost().set_params(**config).fit(pool, **kwargs)


def exported(model, path):
    model.save_model(str(path), format="json")
    return json.loads(path.read_text())


def numeric_leaves(tree, x):
    ids = np.zeros(len(x), np.uint32)
    for bit, split in enumerate(tree.get("splits") or []):
        assert split["split_type"] == "FloatFeature"
        ids |= (x[:, split["float_feature_index"]] > split["border"]).astype(np.uint32) << bit
    return ids


def check_scalar_equations(model, document, x, target, weights, baseline, config, parameter=None):
    cursor = baseline.copy()
    expected_values, expected_weights = [], []
    objective = config["loss_function"].partition(":")[0]
    for iteration, tree in enumerate(document["oblivious_trees"]):
        ids = numeric_leaves(tree, x)
        leaves = len(tree["leaf_values"])
        _, gradient, hessian = objective_terms(target, cursor, objective, parameter)
        gradient = np.float32(gradient * weights)
        if config["data_partition"] == "DocParallel":
            mass = np.float32(hessian * weights) if config["score_function"].startswith("Newton") else weights
            factors = draws(config["bootstrap_type"], len(x), seed=config["random_seed"],
                iteration=iteration, subsample=config.get("subsample", .66),
                temperature=config.get("bagging_temperature", 1.))
            gradient = np.float32(gradient * factors)
            mass = np.float32(mass * factors)
        else:
            mass = weights
        sums = np.bincount(ids, weights=gradient, minlength=leaves)
        masses = np.bincount(ids, weights=mass, minlength=leaves)
        values = np.float32(sums / (masses + np.float32(config["l2_leaf_reg"])))
        values = np.float32(values * np.float32(config["learning_rate"]))
        np.testing.assert_allclose(tree["leaf_values"], values, rtol=7e-5, atol=4e-6)
        np.testing.assert_allclose(tree["leaf_weights"], masses, rtol=5e-5, atol=5e-6)
        cursor = np.float32(cursor + values[ids])
        expected_values.extend(values)
        expected_weights.extend(masses)
    np.testing.assert_allclose(model.get_leaf_values(), expected_values, rtol=7e-5, atol=4e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), expected_weights, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(model.get_test_eval(), cursor, rtol=7e-5, atol=5e-6)
    np.testing.assert_allclose(model.predict(x, prediction_type="RawFormulaVal", task_type="GPU") + baseline,
                               cursor, rtol=7e-5, atol=5e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("objective,parameter", OBJECTIVES)
def test_native_scalar_simple_registered_losses_follow_exported_leaf_equations(tmp_path, mode, objective, parameter):
    x, target, weights, baseline = numeric_problem(objective)
    pool = Pool(x, target, weight=weights, baseline=baseline)
    config = options(mode, objective)
    model = fit(config, pool, eval_set=pool, use_best_model=False)
    params = model.get_all_params()
    assert model.get_metadata()["metal_backend"] == "METAL"
    for key in ("data_partition", "boosting_type", "leaf_estimation_method", "leaf_estimation_iterations"):
        assert params[key] == config[key]
    check_scalar_equations(model, exported(model, tmp_path / "simple.json"),
                           x, target, weights, baseline, config, parameter)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sampling", ("No", "Bayesian", "Bernoulli"))
def test_native_scalar_simple_weak_score_mass_versus_original_gradient_mass(tmp_path, mode, sampling):
    x, target, weights, baseline = numeric_problem("Logloss")
    pool = Pool(x, target, weight=weights, baseline=baseline)
    config = options(mode, "Logloss", score_function="NewtonCosine", bootstrap_type=sampling)
    if sampling == "Bayesian":
        config["bagging_temperature"] = 1.3
    elif sampling == "Bernoulli":
        config["subsample"] = .43
    model = fit(config, pool, eval_set=pool, use_best_model=False)
    document = exported(model, tmp_path / "weak.json")
    check_scalar_equations(model, document, x, target, weights, baseline, config)
    first_mass = np.sum(document["oblivious_trees"][0]["leaf_weights"])
    if mode == "PlainDP":
        assert not np.isclose(first_mass, weights.sum(dtype=float), rtol=1e-3)
    else:
        assert first_mass == pytest.approx(weights.sum(dtype=float), rel=5e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backtracking", ("AnyImprovement", "Armijo"))
def test_native_scalar_simple_single_step_ignores_backtracking(mode, backtracking):
    x, target, weights, baseline = numeric_problem("Logloss")
    pool = Pool(x, target, weight=weights, baseline=baseline)
    config = options(mode, "Logloss", score_function="NewtonCosine", bootstrap_type="Bayesian",
                     bagging_temperature=1.3)
    reference = fit(config, pool, eval_set=pool, use_best_model=False)
    actual = fit(config | dict(leaf_estimation_backtracking=backtracking),
                 pool, eval_set=pool, use_best_model=False)
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(actual, method)(), getattr(reference, method)())
    assert actual.get_evals_result() == reference.get_evals_result()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("iterations", (0, 2))
def test_native_scalar_simple_requires_exactly_one_estimation_iteration(mode, iterations):
    x, target, weights, baseline = numeric_problem("RMSE")
    with pytest.raises(CatBoostError, match="(?i)(Simple|estimation.*iteration)"):
        fit(options(mode, leaf_estimation_iterations=iterations),
            Pool(x, target, weight=weights, baseline=baseline))


def check_exact(actual, expected):
    assert actual.tree_count_ == expected.tree_count_
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(actual, method)(), getattr(expected, method)())
    assert actual.get_evals_result() == expected.get_evals_result()
    assert actual.get_best_iteration() == expected.get_best_iteration()
    assert actual.get_best_score() == expected.get_best_score()


@pytest.mark.parametrize("mode", ("PlainFP", "OrderedFP"))
@pytest.mark.parametrize("sampling", ("Bayesian", "Bernoulli"))
def test_native_scalar_simple_categorical_p4_snapshots_preserve_histories(tmp_path, mode, sampling):
    from test_native_ordered_ctrs import inputs

    cls, x, target, po, config = inputs("Logloss", "Simple", sampling=sampling, count=4)
    config.update(boosting_type="Ordered" if mode == "OrderedFP" else "Plain",
        data_partition="FeatureParallel", iterations=6, leaf_estimation_iterations=1,
        random_strength=.35, has_time=False, metric_period=1)
    pool = Pool(x, target, **po)
    pool.quantize()
    heldout = x.copy()
    heldout[::19, 0] = "unseen-simple-category"
    evaluation = Pool(heldout, target, **po)

    def trained(parameters, **extra):
        return cls().set_params(**parameters).fit(pool, eval_set=evaluation, use_best_model=False, **extra)

    direct = trained(config)
    assert direct.get_metadata()["metal_backend"] == "METAL"
    assert direct.get_metadata()["metal_permutations"] == "4"
    for key in ("data_partition", "boosting_type", "leaf_estimation_method", "leaf_estimation_iterations"):
        assert direct.get_all_params()[key] == config[key]
    document = exported(direct, tmp_path / "categorical.json")
    assert any(split["split_type"] == "OnlineCtr"
               for tree in document["oblivious_trees"] for split in tree.get("splits") or [])
    assert np.ptp(direct.get_test_eval()) > 1e-5

    saved = config | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="simple.snapshot",
                          allow_writing_files=True, train_dir=str(tmp_path))
    callback = StopAfter(2)
    partial = trained(saved, callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    assert (tmp_path / "simple.snapshot").is_file()
    resumed = trained(saved)
    check_exact(resumed, direct)
    check_exact(trained(saved), direct)
    extended = trained(saved | dict(iterations=8))
    check_exact(extended, trained(config | dict(iterations=8)))

    expected = resumed.predict(heldout, prediction_type="RawFormulaVal", task_type="GPU")
    for format_ in ("cbm", "json"):
        path = tmp_path / ("restored." + format_)
        resumed.save_model(str(path), format=format_)
        loaded = cls().load_model(str(path), format=format_)
        np.testing.assert_array_equal(loaded.get_leaf_values(), resumed.get_leaf_values())
        np.testing.assert_array_equal(loaded.get_leaf_weights(), resumed.get_leaf_weights())
        np.testing.assert_allclose(loaded.predict(heldout, prediction_type="RawFormulaVal", task_type="GPU"),
                                   expected, rtol=5e-6, atol=7e-7)
