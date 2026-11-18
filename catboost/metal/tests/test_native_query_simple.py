"""Native symmetric query Simple leaves against independent CUDA equations.

DocParallel retains the sampled search target: Newton scores use curvature,
other scores use original query or incident-pair weights. Its Simple count
guard also permits signed curvature mass. FeatureParallel Plain and Ordered
refit one Gradient step on original weights. PairLogit retains Metal's existing
zero-average correction; CUDA's current zero-initialized bias loop is a no-op.
"""

import os

import numpy as np
import pytest
from catboost import CatBoost, Pool

from cuda_querywise_reference import query_terms
from test_greedy_sampling import draws
from test_native_greedy_api import StopAfter
from test_native_pair_api import data as pair_data
from test_native_query_api import pool_for
from test_native_scalar_simple import check_exact, exported, fit, numeric_leaves, options
from test_pairwise_training import training_terms


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal query Simple adapter",
)

MODES = ("PlainDP", "PlainFP", "OrderedFP")
OBJECTIVES = ("QueryRMSE", "QuerySoftMax:beta=0.7;lambda=0.03", "PairLogit")


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def problem(objective):
    if objective == "PairLogit":
        x, target, groups, pairs, pair_weights = pair_data()
        pool_options = dict(group_id=groups, pairs=pairs, pairs_weight=pair_weights)
        weights = np.ones(len(x), np.float32)
    else:
        _, x, target, groups, weights = pool_for(objective)
        pairs = pair_weights = None
        pool_options = dict(group_id=groups, weight=weights)
    baseline = np.linspace(-.7, .4, len(x), dtype=np.float32)
    offsets = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(x)].astype(np.uint32)
    return dict(pool=Pool(x, target, baseline=baseline, **pool_options), x=x,
                target=target, weights=weights, offsets=offsets, baseline=baseline,
                pairs=pairs, pair_weights=pair_weights)


def row_terms(data, cursor, objective):
    loss, _, parameters = objective.partition(":")
    if loss == "PairLogit":
        terms = training_terms(cursor, data["pairs"][:, 0], data["pairs"][:, 1], data["pair_weights"])
        return tuple(np.asarray(terms[key], np.float32)
                     for key in ("gradients", "curvature", "incident_weights"))
    parameters = dict(item.split("=", 1) for item in parameters.split(";") if item)
    gradient, hessian, _, _ = query_terms(
        data["target"], cursor, data["weights"], data["offsets"], loss,
        beta=float(parameters.get("beta", 1)), query_lambda=float(parameters.get("lambda", .01)))
    return np.float32(gradient), np.float32(hessian), data["weights"]


def check_leaf_equations(model, document, data, config):
    cursor = data["baseline"].copy()
    all_values, all_weights = [], []
    for iteration, tree in enumerate(document["oblivious_trees"]):
        ids = numeric_leaves(tree, data["x"])
        leaf_count = len(tree["leaf_values"])
        gradient, hessian, original_mass = row_terms(data, cursor, config["loss_function"])
        if config["data_partition"] == "DocParallel":
            mass = hessian if config["score_function"].startswith("Newton") else original_mass
            factors = draws(config["bootstrap_type"], len(cursor), seed=config["random_seed"],
                iteration=iteration, subsample=config.get("subsample", .66),
                temperature=config.get("bagging_temperature", 1.))
            gradient = np.float32(gradient * factors)
            mass = np.float32(mass * factors)
            # CUDA drops zero Bernoulli/Poisson draws before forming Count.
            retained = (factors != 0 if config["bootstrap_type"] in ("Bernoulli", "Poisson")
                        else np.ones(len(cursor), bool))
            populated = np.bincount(ids[retained], minlength=leaf_count) > 0
        else:
            mass = original_mass
            populated = np.bincount(ids, weights=mass, minlength=leaf_count) >= 1e-20
        sums = np.bincount(ids, weights=gradient, minlength=leaf_count)
        masses = np.bincount(ids, weights=mass, minlength=leaf_count)
        regularization = float(np.float32(config["l2_leaf_reg"])) or float(np.float32(1e-20))
        values = np.divide(sums, masses + regularization, out=np.zeros(leaf_count), where=populated)
        values = np.float32(values)
        if config["loss_function"] == "PairLogit" and config["data_partition"] == "FeatureParallel":
            # Preserve the existing Metal zero-average correction. The literal
            # CUDA bias loop reads zero-initialized leaves before copying values.
            values = np.float32(values - np.float32(values.mean(dtype=float)))
        values = np.float32(values * np.float32(config["learning_rate"]))
        np.testing.assert_allclose(tree["leaf_values"], values, rtol=9e-5, atol=5e-6)
        np.testing.assert_allclose(tree["leaf_weights"], masses, rtol=5e-5, atol=8e-6)
        cursor = np.float32(cursor + values[ids])
        all_values.extend(values)
        all_weights.extend(masses)
    np.testing.assert_allclose(model.get_leaf_values(), all_values, rtol=9e-5, atol=5e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), all_weights, rtol=5e-5, atol=8e-6)
    np.testing.assert_allclose(model.get_test_eval(), cursor, rtol=9e-5, atol=8e-6)
    np.testing.assert_allclose(
        model.predict(data["x"], prediction_type="RawFormulaVal", task_type="GPU") + data["baseline"],
        cursor, rtol=9e-5, atol=8e-6)


def check_options(model, config):
    assert model.get_metadata()["metal_backend"] == "METAL"
    params = model.get_all_params()
    for key in ("boosting_type", "data_partition", "leaf_estimation_method", "leaf_estimation_iterations"):
        assert params[key] == config[key]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("score", ("Cosine", "NewtonCosine"))
def test_native_query_simple_original_and_curvature_score_mass(tmp_path, mode, objective, score):
    data = problem(objective)
    config = options(mode, objective, score_function=score)
    model = fit(config, data["pool"], eval_set=data["pool"], use_best_model=False)
    check_options(model, config)
    check_leaf_equations(model, exported(model, tmp_path / "simple.json"), data, config)


SAMPLED_CASES = [
    (objective, "PlainDP", sampling)
    for objective in OBJECTIVES for sampling in ("Bayesian", "Bernoulli")
] + [
    (objective, mode, ("Bayesian", "Bernoulli")[(objective_index + mode_index) % 2])
    for objective_index, objective in enumerate(OBJECTIVES)
    for mode_index, mode in enumerate(("PlainFP", "OrderedFP"))
]


@pytest.mark.parametrize("objective,mode,sampling", SAMPLED_CASES)
def test_native_query_simple_bootstrap_is_retained_only_by_docparallel(tmp_path, objective, mode, sampling):
    data = problem(objective)
    sampling_options = (dict(bagging_temperature=1.3) if sampling == "Bayesian" else dict(subsample=.43))
    config = options(mode, objective, score_function="NewtonCosine", bootstrap_type=sampling, **sampling_options)
    model = fit(config, data["pool"], eval_set=data["pool"], use_best_model=False)
    document = exported(model, tmp_path / "sampled.json")
    check_leaf_equations(model, document, data, config)
    _, _, original_mass = row_terms(data, data["baseline"], objective)
    exported_mass = np.sum(document["oblivious_trees"][0]["leaf_weights"])
    if mode == "PlainDP":
        assert not np.isclose(exported_mass, original_mass.sum(dtype=float), rtol=1e-3)
    else:
        assert exported_mass == pytest.approx(original_mass.sum(dtype=float), rel=5e-6)


@pytest.mark.parametrize("objective,mode", list(zip(OBJECTIVES, MODES)))
@pytest.mark.parametrize("backtracking", ("AnyImprovement", "Armijo"))
def test_native_query_simple_one_step_ignores_backtracking(objective, mode, backtracking):
    data = problem(objective)
    config = options(mode, objective, score_function="NewtonCosine", bootstrap_type="Bayesian",
                     bagging_temperature=1.3)
    reference = fit(config, data["pool"], eval_set=data["pool"], use_best_model=False)
    actual = fit(config | dict(leaf_estimation_backtracking=backtracking),
                 data["pool"], eval_set=data["pool"], use_best_model=False)
    check_exact(actual, reference)


@pytest.mark.parametrize("mode,score", (
    ("PlainDP", "NewtonL2"), ("PlainFP", "Cosine"), ("OrderedFP", "Cosine"),
    ("PlainFP", "NewtonL2"), ("OrderedFP", "NewtonCosine"),
))
def test_native_query_simple_negative_lambda_export_and_snapshot_weight_semantics(tmp_path, mode, score):
    # The DocParallel fixture has initial g=(-.5,+.5), h=(-.75,-.75)
    # per query: Count=8 and signed mass=-6 per leaf. Its Simple denominator
    # remains positive. Ordered needs some positive learning/quality curvature
    # to produce a finite dynamic Cosine score (the allnegative case is below).
    x = np.tile(np.array([[0.], [1.]], np.float32), (8, 1))
    target = x[:, 0].copy()
    baseline = np.zeros(len(x), np.float32)
    query_lambda = -1.
    if mode == "OrderedFP" and score == "NewtonCosine":
        query_lambda = -.15
        baseline[8:10] = [-2., 2.]
    weights = np.ones(len(x), np.float32)
    pool = Pool(x, target, group_id=np.repeat(np.arange(8), 2), baseline=baseline, weight=weights)
    data = dict(pool=pool, x=x, target=target, weights=weights, baseline=baseline,
                offsets=np.arange(0, len(x) + 1, 2, dtype=np.uint32), pairs=None, pair_weights=None)
    config = options(mode, f"QuerySoftMax:beta=1;lambda={query_lambda}", iterations=4, depth=1,
                     l2_leaf_reg=12., score_function=score)
    initial_curvature = row_terms(data, baseline, config["loss_function"])[1]
    assert np.any(initial_curvature < 0)
    if mode == "OrderedFP" and score == "NewtonCosine":
        assert np.any(initial_curvature > 0)

    def trained(parameters, **extra):
        return fit(parameters, pool, eval_set=pool, use_best_model=False, **extra)

    direct = trained(config)
    check_options(direct, config)
    document = exported(direct, tmp_path / "negative-lambda.json")
    check_leaf_equations(direct, document, data, config)
    assert np.all(direct.get_tree_leaf_counts() == 2)
    assert np.max(np.abs(direct.get_leaf_values())) > .01
    if mode == "PlainDP":
        assert np.all(direct.get_leaf_weights() < 0)
        np.testing.assert_allclose(document["oblivious_trees"][0]["leaf_weights"], [-6., -6.], atol=1e-6)
    else:
        np.testing.assert_array_equal(direct.get_leaf_weights(), np.full(8, 8.))

    saved = config | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="query-simple.snapshot",
                          allow_writing_files=True, train_dir=str(tmp_path))
    callback = StopAfter(2)
    partial = trained(saved, callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    assert (tmp_path / "query-simple.snapshot").is_file()
    resumed = trained(saved)
    check_exact(resumed, direct)
    check_exact(trained(saved), direct)
    check_exact(trained(saved | dict(iterations=6)), trained(config | dict(iterations=6)))

    expected = resumed.predict(x, prediction_type="RawFormulaVal", task_type="GPU")
    for format_ in ("cbm", "json"):
        path = tmp_path / ("restored-query-simple." + format_)
        resumed.save_model(str(path), format=format_)
        restored = CatBoost().load_model(str(path), format=format_)
        if format_ == "cbm":
            np.testing.assert_array_equal(restored.get_leaf_values(), resumed.get_leaf_values())
            np.testing.assert_array_equal(restored.get_leaf_weights(), resumed.get_leaf_weights())
        else:
            # JSON's decimal serialization may move a stored double by one ULP.
            # Preserve every native float32 value and bound the double change.
            for accessor in ("get_leaf_values", "get_leaf_weights"):
                actual, expected_values = getattr(restored, accessor)(), getattr(resumed, accessor)()
                np.testing.assert_array_max_ulp(actual, expected_values, maxulp=1)
                np.testing.assert_array_equal(actual.astype(np.float32), expected_values.astype(np.float32))
        np.testing.assert_allclose(restored.predict(x, prediction_type="RawFormulaVal", task_type="GPU"),
                                   expected, rtol=5e-6, atol=7e-7)


def test_native_ordered_simple_allnegative_curvature_has_no_valid_cosine_split(tmp_path):
    x = np.tile(np.array([[0.], [1.]], np.float32), (8, 1))
    target = x[:, 0].copy()
    weights = np.ones(len(x), np.float32)
    baseline = np.zeros(len(x), np.float32)
    offsets = np.arange(0, len(x) + 1, 2, dtype=np.uint32)
    pool = Pool(x, target, group_id=np.repeat(np.arange(8), 2), baseline=baseline, weight=weights)
    data = dict(pool=pool, x=x, target=target, weights=weights, baseline=baseline,
                offsets=offsets, pairs=None, pair_weights=None)
    config = options("OrderedFP", "QuerySoftMax:beta=1;lambda=-1", iterations=2,
                     depth=1, l2_leaf_reg=12., score_function="NewtonCosine")
    gradient, curvature, _ = row_terms(data, baseline, config["loss_function"])
    assert np.all(curvature < 0) and abs(gradient.sum(dtype=float)) < 1e-12
    # CUDA dynamic scores use zero slopes for nonpositive estimate weights.
    # Their norm remains1e-20, below the1e-15 validity threshold, so the sole
    # candidate is invalid. Gradient1 refitting has a zero total gradient.
    model = fit(config, pool, eval_set=pool, use_best_model=False)
    check_leaf_equations(model, exported(model, tmp_path / "allnegative.json"), data, config)
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [1, 1])
    np.testing.assert_array_equal(model.get_leaf_values(), [0., 0.])
    np.testing.assert_array_equal(model.get_leaf_weights(), [16., 16.])
