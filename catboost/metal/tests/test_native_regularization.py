"""Public Metal regularization wiring, equations, and saved host RNG lifecycle.

The leaf oracle evaluates the original weighted rows. The split oracle uses
an independent MT19937-64 interpreter and the checked-in CUDA policy launch
arithmetic; it does not call a Metal score helper or train a CPU model. These
are source-contract checks, not comparisons with execution on NVIDIA hardware.
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRegressor, Pool

from test_native_compound_ctrs import categorical_problem, options as compound_options
from test_ordered_rng import ReferenceMt64


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal regularization adapter",
)
PATHS = ("plain-doc", "plain-feature", "ordered-feature")


class StopAfter:
    def __init__(self, stop=None):
        self.stop, self.iterations = stop, []

    def after_iteration(self, info):
        self.iterations.append(info.iteration)
        return self.stop is None or info.iteration < self.stop


@pytest.fixture(autouse=True)
def only_gpu_fits(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(path="plain-doc", **extra):
    result = dict(
        task_type="GPU", loss_function="RMSE", eval_metric="RMSE",
        grow_policy="SymmetricTree",
        boosting_type="Ordered" if path == "ordered-feature" else "Plain",
        data_partition="DocParallel" if path == "plain-doc" else "FeatureParallel",
        iterations=6, depth=3, learning_rate=.2, l2_leaf_reg=.7,
        random_seed=731, bootstrap_type="No", random_strength=0,
        score_function="Cosine", leaf_estimation_method="Newton",
        leaf_estimation_iterations=4, leaf_estimation_backtracking="No",
        boost_from_average=False, permutation_count=1, has_time=True,
        one_hot_max_size=2, max_ctr_complexity=1, model_size_reg=0,
        counter_calc_method="SkipTest", border_count=16,
        verbose=False, allow_writing_files=False, metric_period=1,
    )
    if path == "ordered-feature":
        result.update(min_fold_size=2, fold_len_multiplier=1.7, fold_permutation_block=3)
    return result | extra


def fit(config, pool, **kwargs):
    result = CatBoostRegressor().set_params(**config).fit(pool, **kwargs)
    assert result.get_metadata()["metal_backend"] == "METAL"
    return result


def raw(model, data):
    return model.predict(data, prediction_type="RawFormulaVal", task_type="GPU")


def root_problem():
    y = np.array([.15, .5, 1.1, 1.8, 2.2, .8, 10.], np.float32)
    weights = np.array([.5, 1, 2, 3, 1, .25, 0], np.float32)
    x = np.column_stack((np.arange(len(y)), np.arange(len(y)) ** 2)).astype(np.float32)
    return x, y, weights


def poisson_leaf(y, weights, cursor, method, normalize, ridge, l2, steps, mode="No"):
    """The scalar source walker, widened equations with float model points."""
    mass = weights.sum(dtype=np.float64)
    l2 = float(np.float32(l2))

    def evaluate(point):
        prediction = np.float32(cursor + point)
        exponential = np.exp(float(prediction))
        value = np.dot(weights.astype(float), y.astype(float) * prediction - exponential)
        gradient = np.dot(weights.astype(float), y.astype(float) - exponential)
        diagonal = mass * exponential if method == "Newton" else mass
        if normalize:
            value /= mass
            gradient /= mass
            diagonal /= mass
        if ridge:
            value -= .5 * l2 * float(point) ** 2
            gradient -= l2 * float(point)
        return value, gradient, diagonal + l2

    point = np.float32(0)
    value, gradient, diagonal = evaluate(point)
    fresh, updated, step = True, False, 1.
    for attempt in range(max(steps, 100)):
        if attempt >= steps and (updated or mode == "No"):
            break
        if fresh:
            direction = np.float32(gradient / (diagonal + 1e-20))
            dot = gradient * float(direction)
        trial = np.float32(float(point) + step * float(direction))
        trial_value, trial_gradient, trial_diagonal = evaluate(trial)
        threshold = value + (1e-5 * step * dot if mode == "Armijo" else 0)
        if mode == "No" or (np.isfinite(trial_value) and trial_value >= threshold):
            point, value, gradient, diagonal = trial, trial_value, trial_gradient, trial_diagonal
            fresh, updated, step = True, True, 1.
        else:
            fresh, step = False, step / 2
    return point


def root_forest(y, weights, config, normalize, ridge):
    cursor = np.float32(0)
    leaves = []
    for _ in range(config["iterations"]):
        point = poisson_leaf(y, weights, cursor, config["leaf_estimation_method"], normalize,
            ridge, config["l2_leaf_reg"], config["leaf_estimation_iterations"],
            config["leaf_estimation_backtracking"])
        leaf = np.float32(np.float32(config["learning_rate"]) * point)
        leaves.append(leaf)
        cursor = np.float32(cursor + leaf)
    return np.asarray(leaves), cursor


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
@pytest.mark.parametrize("normalize,ridge", ((False, False), (False, True), (True, False), (True, True)))
def test_public_scalar_leaf_flags_match_original_weighted_rows(path, method, normalize, ridge):
    x, y, weights = root_problem()
    config = options(path, loss_function="Poisson", iterations=3, depth=0,
        leaf_estimation_method=method, fold_size_loss_normalization=normalize,
        add_ridge_penalty_to_loss_function=ridge)
    model = fit(config, Pool(x, y, weight=weights))
    leaves, prediction = root_forest(y, weights, config, normalize, ridge)
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), np.ones(3, np.uint32))
    np.testing.assert_allclose(model.get_leaf_values(), leaves, rtol=1e-5, atol=4e-6)
    np.testing.assert_allclose(raw(model, x), prediction, rtol=1e-5, atol=4e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), weights.sum(dtype=np.float64), atol=4e-6)
    if normalize or ridge:
        control, _ = root_forest(y, weights, config, False, False)
        assert np.max(np.abs(leaves - control)) > 1e-5


@pytest.mark.parametrize("mode", ("AnyImprovement", "Armijo"))
@pytest.mark.parametrize("normalize", (False, True))
def test_public_ridge_changes_the_backtracking_objective(mode, normalize):
    y = np.array([0, 2, 1, 8, 3, 1, 2], np.float32)
    weights = np.array([.5, 1, 2, 3, 1, .25, 4], np.float32)
    x = np.arange(len(y), dtype=np.float32).reshape(-1, 1)
    config = options(loss_function="Poisson", iterations=1, depth=0, learning_rate=1,
        l2_leaf_reg=.4, leaf_estimation_backtracking=mode,
        fold_size_loss_normalization=normalize, add_ridge_penalty_to_loss_function=True)
    model = fit(config, Pool(x, y, weight=weights))
    expected, _ = root_forest(y, weights, config, normalize, True)
    np.testing.assert_allclose(model.get_leaf_values(), expected, rtol=1e-5, atol=4e-6)


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_public_greedy_ridge_keeps_source_leaf_normalization_noop(policy, method):
    x, y, weights = root_problem()
    config = options(loss_function="Poisson", grow_policy=policy, iterations=3, depth=0,
        leaf_estimation_method=method, fold_size_loss_normalization=True,
        add_ridge_penalty_to_loss_function=True)
    model = fit(config, Pool(x, y, weight=weights))
    expected, prediction = root_forest(y, weights, config, False, True)
    np.testing.assert_allclose(model.get_leaf_values(), expected, rtol=1e-5, atol=4e-6)
    np.testing.assert_allclose(raw(model, x), prediction, rtol=1e-5, atol=4e-6)


def numeric_problem(tmp_path):
    """Each source policy is present; quantization exposes every border."""
    rng = np.random.default_rng(2478)
    counts = [1, 7, 31] * 3
    rows = 192
    x = np.column_stack([1 + rng.permutation(np.arange(rows) % (count + 1))
                         for count in counts]).astype(np.float32)
    y = (rng.normal(size=rows) + .2 * x[:, 0] - .1 * x[:, 3]).astype(np.float32)
    weights = rng.integers(1, 8, rows).astype(np.float32)
    weights[::23] = 0
    pool = Pool(x, y, weight=weights)
    pool.quantize(per_float_feature_quantization=[f"{i}:border_count={count}"
                                                  for i, count in enumerate(counts)])
    filename = tmp_path / "regularization-borders.tsv"
    pool.save_quantization_borders(str(filename))
    borders = [[] for _ in counts]
    for line in filename.read_text().splitlines():
        feature, value, *_ = line.split("\t")
        borders[int(feature)].append(np.float32(value))
    assert list(map(len, borders)) == counts
    return pool, x, y, weights, borders


def policy_exponents(path, seed, exponent, frequency):
    # Plain Doc draws BaseIterationSeed even for P=1. FeatureParallel instead
    # creates a shared bootstrap seed cache, including bootstrap_type=No.
    shared = ReferenceMt64(seed)
    shared.advance(1 if path == "plain-doc" else 65537)
    policies = ReferenceMt64(shared.next())
    result = []
    for _ in range(3):
        source = policies.next()
        high = 36969 * ((source >> 32) & 65535) + (source >> 48)
        low = 18000 * (source & 65535) + ((source >> 16) & 65535)
        uniform = ((high * 65536 + low) & 0xffffffff) / 4294967295
        result.append(1. if uniform >= frequency else float(np.float32(exponent)))
    return result


def root_split_scores(x, y, weights, borders, exponents, l2):
    gradients = np.float32(y * weights)
    candidates = []
    total_sum = np.float32(gradients.sum(dtype=np.float64))
    total_mass = np.float32(weights.sum(dtype=np.float64))
    l2 = float(np.float32(l2))
    for feature, thresholds in enumerate(borders):
        exponent = exponents[0 if len(thresholds) == 1 else 1 if len(thresholds) <= 15 else 2]
        for border in thresholds:
            left = x[:, feature] <= border
            left_sum = np.float32(gradients[left].sum(dtype=np.float64))
            left_mass = np.float32(weights[left].sum(dtype=np.float64))
            score = np.float32(0)
            for value, mass in ((left_sum, left_mass),
                    (np.float32(total_sum - left_sum), np.float32(total_mass - left_mass))):
                term = -float(value) ** 2 / (float(mass) + l2) if mass > 1e-20 else 0.
                if exponent != 1 and term:
                    term = -(abs(term) / float(mass)) ** exponent * float(mass)
                score = np.float32(float(score) + term)
            candidates.append((float(score), feature, float(border)))
    return sorted(candidates)


@pytest.mark.parametrize("path", ("plain-doc", "plain-feature"))
@pytest.mark.parametrize("score", ("L2", "NewtonL2"))
@pytest.mark.parametrize("seed", (27, 731))
def test_public_meta_l2_selects_independent_source_policy_score(tmp_path, path, score, seed):
    pool, x, y, weights, borders = numeric_problem(tmp_path)
    config = options(path, iterations=1, depth=1, score_function=score, random_seed=seed,
        meta_l2_exponent=.35, meta_l2_frequency=.47)
    exponents = policy_exponents(path, seed, .35, .47)
    assert 1. in exponents and float(np.float32(.35)) in exponents
    expected = root_split_scores(x, y, weights, borders, exponents, config["l2_leaf_reg"])
    assert expected[1][0] - expected[0][0] > 1e-4  # Unambiguous winner.
    if (path, seed) in (("plain-doc", 27), ("plain-feature", 731)):
        ordinary = root_split_scores(x, y, weights, borders, [1., 1., 1.], config["l2_leaf_reg"])
        assert expected[0][1:] != ordinary[0][1:]
    model = fit(config, pool)
    filename = tmp_path / "meta.json"
    model.save_model(filename, format="json")
    split = json.loads(filename.read_text())["oblivious_trees"][0]["splits"][0]
    assert split["split_type"] == "FloatFeature"
    assert split["float_feature_index"] == expected[0][1]
    assert split["border"] == pytest.approx(expected[0][2], abs=1e-6)


LIFECYCLE_CASES = (("plain-doc", "numeric"), ("plain-doc", "simple"),
                   ("plain-feature", "numeric"), ("plain-feature", "simple"),
                   ("plain-feature", "compound"))


def lifecycle_problem(tmp_path, path, kind):
    if kind == "numeric":
        pool, x, *_ = numeric_problem(tmp_path)
        return pool, x, options(path)
    x, y, future, pool_options = categorical_problem()
    config = compound_options(boosting="Plain", complexity=2 if kind == "compound" else 1)
    config.update(data_partition="DocParallel" if path == "plain-doc" else "FeatureParallel")
    # Distinct declared CTR grid policies exercise policy-local seed expansion
    # within both the dependent simple bank and active tree-CTR packs.
    ctrs = [f"Borders:CtrBorderType=Uniform:CtrBorderCount={count}:Prior=0.5"
            for count in (7, 31)]
    config.update(simple_ctr=ctrs, combinations_ctr=ctrs)
    return Pool(x, y, **pool_options), future, config


def assert_exact_model(actual, expected, future):
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(actual, method)(), getattr(expected, method)())
    np.testing.assert_array_equal(raw(actual, future), raw(expected, future))
    assert actual.get_evals_result() == expected.get_evals_result()


@pytest.mark.parametrize("path,kind", LIFECYCLE_CASES)
@pytest.mark.parametrize("bootstrap", ("No", "Bernoulli"))
def test_public_fractional_meta_l2_exact_snapshot_recovery(tmp_path, path, kind, bootstrap):
    pool, future, config = lifecycle_problem(tmp_path, path, kind)
    config.update(iterations=7, permutation_count=4, has_time=False, random_seed=731,
        score_function="L2", bootstrap_type=bootstrap, meta_l2_exponent=.65, meta_l2_frequency=.47,
        fold_size_loss_normalization=True, add_ridge_penalty_to_loss_function=True)
    if bootstrap == "Bernoulli":
        config["subsample"] = .73
    saved = config | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="meta.snapshot",
        train_dir=str(tmp_path), allow_writing_files=True)
    stopped = fit(saved, pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(3)])
    assert stopped.tree_count_ == 3
    assert (tmp_path / "meta.snapshot").is_file()
    recorder = StopAfter()
    resumed = fit(saved, pool, eval_set=pool, use_best_model=False, callbacks=[recorder])
    direct = fit(config, pool, eval_set=pool, use_best_model=False)
    assert recorder.iterations == [4, 5, 6, 7]
    assert_exact_model(resumed, direct, future)
    completed = fit(saved, pool, eval_set=pool, use_best_model=False)
    assert_exact_model(completed, direct, future)
    if kind == "compound":
        assert int(direct.get_metadata()["metal_tree_ctr_features"]) > 0


@pytest.mark.parametrize("completed", (False, True))
@pytest.mark.parametrize("field,value", (("fold_size_loss_normalization", False),
    ("add_ridge_penalty_to_loss_function", False), ("meta_l2_exponent", .8), ("meta_l2_frequency", .61)))
def test_regularization_snapshot_rejects_changed_options_without_writing(tmp_path, completed, field, value):
    pool, *_ = numeric_problem(tmp_path)
    config = options(iterations=4, score_function="L2", meta_l2_exponent=.65, meta_l2_frequency=.47,
        fold_size_loss_normalization=True, add_ridge_penalty_to_loss_function=True,
        save_snapshot=True, snapshot_interval=0, snapshot_file="options.snapshot",
        train_dir=str(tmp_path), allow_writing_files=True)
    fit(config, pool, callbacks=[] if completed else [StopAfter(2)])
    snapshot = tmp_path / "options.snapshot"
    original = snapshot.read_bytes()
    with pytest.raises(CatBoostError, match="(?i)snapshot|incompat|match|parameter|option"):
        fit(config | {field: value}, pool)
    assert snapshot.read_bytes() == original


@pytest.mark.parametrize("path,policy", (("plain-doc", "SymmetricTree"),
    ("ordered-feature", "SymmetricTree"), ("plain-doc", "Depthwise")))
def test_meta_l2_is_a_source_noop_for_cosine_scores(tmp_path, path, policy):
    pool, x, *_ = numeric_problem(tmp_path)
    config = options(path, grow_policy=policy, score_function="Cosine", iterations=3)
    ordinary = fit(config, pool, eval_set=pool, use_best_model=False)
    configured = fit(config | dict(meta_l2_exponent=-.5, meta_l2_frequency=.47),
                     pool, eval_set=pool, use_best_model=False)
    assert_exact_model(configured, ordinary, x)
