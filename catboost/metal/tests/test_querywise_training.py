"""Connected querywise Metal training checked without any CPU CatBoost fit."""

import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import _native
from cuda_querywise_reference import query_terms, query_loss, structure_reference, leaf_reference, train_reference
from test_bootstrap import uniforms


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Querywise tests must not train CPU CatBoost")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal checks require Apple Silicon")
    return _native.device_info()


def options(objective="QueryRMSE", **overrides):
    return dict(iterations=3, depth=2, learning_rate=0.2, l2_leaf_reg=2, bias=0,
                score_function="Cosine", objective=objective, query_beta=1, query_lambda=0.01,
                leaf_estimation_method="Newton", leaf_estimation_iterations=4,
                leaf_estimation_backtracking="No") | overrides


def problem(objective, seed=492, rows=521):
    rng = np.random.default_rng(seed)
    bins = rng.integers(0, 6, (3, rows), dtype=np.uint8)
    signal = 1.2 * (bins[0] > 2) - 0.7 * (bins[2] > 3) + rng.normal(0, 0.3, rows)
    targets = np.exp(signal / 2) if objective == "QuerySoftMax" else signal
    weights = rng.uniform(0.25, 3, rows).astype(np.float32)
    weights[::17] = 0
    offsets = np.array([0, 1, 4, 7, 12, 33, 64, 321, rows], np.uint32)
    initial = rng.normal(0, 0.1, rows).astype(np.float32)
    features = np.repeat(np.arange(3, dtype=np.uint32), 5)
    borders = np.tile(np.arange(5, dtype=np.uint32), 3)
    return bins, targets.astype(np.float32), weights, offsets, initial, features, borders


def compare(bins, targets, features, borders, **params):
    expected = train_reference(bins, targets, features, borders, **params)
    actual = _native.train(bins, targets, features, borders, **params)
    for name in ("depths", "split_features", "split_bins"):
        np.testing.assert_array_equal(getattr(actual, name), expected[name], err_msg=name)
    for name in ("leaf_values", "leaf_weights", "predictions", "loss"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=3e-4, atol=7e-5, err_msg=name)
        assert np.isfinite(getattr(actual, name)).all()
    assert actual.stats["kernel_dispatches"] > 0
    return actual, expected


def test_query_rmse_reference_matches_weighted_hand_calculation():
    g, h, numerator, denominator = query_terms(
        [0, 2, 4, 6, -1, 3], [0, 1, 0, -1, 0, 2], [1, 2, 1, 2, 1, 3],
        [0, 3, 6], "QueryRMSE")
    np.testing.assert_allclose(g, [-1.5, -1, 2.5, 26 / 3, -11 / 3, -5])
    np.testing.assert_array_equal(h, [1, 2, 1, 2, 1, 3])
    assert numerator == pytest.approx(205 / 3)
    assert denominator == 10


def test_query_softmax_reference_matches_weighted_hand_calculation():
    g, h, numerator, denominator = query_terms(
        [0, 1, 2, 1], [0, 0, 0, 0], [1, 3, 2, 1], [0, 2, 4], "QuerySoftMax")
    np.testing.assert_allclose(g, [-0.75, 0.75, 2 / 3, -2 / 3])
    np.testing.assert_allclose(h, [0.5925, 0.5925, 5 * (2 / 9 + 0.01), 5 * (2 / 9 + 0.01)], rtol=1e-8)
    assert numerator == pytest.approx(-3 * np.log(0.75) - 4 * np.log(2 / 3) - np.log(1 / 3))
    assert denominator == 8


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("steps", [1, 4])
def test_queries_crossing_leaves_recompute_joint_query_statistics(metal_device, objective, method, steps):
    if objective == "QueryRMSE":
        targets = np.array([0, 2, 4, 6, -1, 3], np.float32)
        weights = np.array([1, 2, 1, 2, 1, 3], np.float32)
        initial = np.array([0, 1, 0, -1, 0, 2], np.float32)
        groups = np.array([0, 3, 6], np.uint32)
        bins = np.array([[0, 1, 0, 1, 0, 1]], np.uint8)
    else:
        targets, weights = np.array([0, 1, 2, 1], np.float32), np.array([1, 3, 2, 1], np.float32)
        initial, groups = np.zeros(4, np.float32), np.array([0, 2, 4], np.uint32)
        bins = np.array([[0, 1, 1, 0]], np.uint8)
    candidates = np.array([0], np.uint32)
    result, _ = compare(bins, targets, candidates, candidates,
        **options(objective, iterations=2, depth=1, l2_leaf_reg=1,
                  sample_weight=weights, initial_predictions=initial, group_offsets=groups,
                  leaf_estimation_method=method, leaf_estimation_iterations=steps))
    np.testing.assert_array_equal(result.depths, [1, 1])


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine"])
@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
def test_weighted_query_training_matches_independent_full_oracle(metal_device, objective, method, score, mode):
    bins, targets, weights, groups, initial, features, borders = problem(objective)
    compare(bins, targets, features, borders,
        **options(objective, score_function=score, sample_weight=weights, group_offsets=groups,
                  initial_predictions=initial, leaf_estimation_method=method, leaf_estimation_backtracking=mode))


def test_query_rmse_is_invariant_to_per_query_target_offsets(metal_device):
    bins, _, weights, groups, initial, features, borders = problem("QueryRMSE")
    targets = np.arange(bins.shape[1], dtype=np.float32) % 7
    shifted = targets.copy()
    for group, (start, end) in enumerate(zip(groups[:-1], groups[1:])):
        shifted[start:end] += (group - 3) * 11
    params = options(group_offsets=groups, sample_weight=weights, initial_predictions=initial)
    original = _native.train(bins, targets, features, borders, **params)
    changed = _native.train(bins, shifted, features, borders, **params)
    np.testing.assert_array_equal(original.depths, changed.depths)
    np.testing.assert_array_equal(original.split_features, changed.split_features)
    np.testing.assert_array_equal(original.split_bins, changed.split_bins)
    np.testing.assert_allclose(original.leaf_values, changed.leaf_values, rtol=3e-4, atol=1e-5)
    np.testing.assert_allclose(original.loss, changed.loss, rtol=3e-5, atol=1e-5)


def test_query_rmse_constructor_centers_large_targets_before_evaluating_initial_loss(metal_device):
    empty = np.array([], np.uint32)
    with _native.Session(np.zeros((1, 2), np.uint8), np.full(2, 1e38, np.float32), empty, empty,
            **options("QueryRMSE", iterations=2, depth=0, group_offsets=np.array([0, 2], np.uint32))) as session:
        np.testing.assert_array_equal(session.result().loss, [0])
        session.step()
        session.step()
        result = session.result()
    np.testing.assert_array_equal(result.loss, 0)
    np.testing.assert_array_equal(result.leaf_values, 0)
    np.testing.assert_array_equal(result.predictions, 0)


@pytest.mark.parametrize("beta,lam", [(-1, 0.01), (0, -100), (1, -0.1), (-1, -0.3)])
def test_finite_signed_query_softmax_parameters_are_preserved(metal_device, beta, lam):
    bins = np.array([[0, 1, 0, 1]], np.uint8)
    targets = np.array([0, 1, 2, 1], np.float32)
    candidates = np.array([0], np.uint32)
    result, _ = compare(bins, targets, candidates, candidates,
        **options("QuerySoftMax", iterations=2, depth=1, leaf_estimation_method="Gradient",
                  group_offsets=np.array([0, 2, 4], np.uint32), query_beta=beta, query_lambda=lam))
    if beta == 0:
        np.testing.assert_array_equal(result.leaf_values, 0)


def signed_curvature_options(**extra):
    return options("QuerySoftMax", iterations=1, depth=1, learning_rate=1, l2_leaf_reg=1,
                   leaf_estimation_iterations=1, group_offsets=np.array([0, 2], np.uint32),
                   query_lambda=-1, **extra)


def test_signed_newton_curvature_uses_the_positive_regularized_diagonal(metal_device):
    bins, targets, candidates = np.array([[0, 1]], np.uint8), np.array([0, 1], np.float32), np.array([0], np.uint32)
    result, _ = compare(bins, targets, candidates, candidates, **signed_curvature_options())
    # Each row has g=+/-0.5, h=-0.75. Adding leaf L2=1 gives direction +/-2.
    np.testing.assert_array_equal(result.leaf_values, [[-2, 2]])


@pytest.mark.parametrize("l2", [0.5, 0.75])
def test_nonpositive_query_newton_diagonal_is_an_explicit_error(metal_device, l2):
    candidates = np.array([0], np.uint32)
    with pytest.raises(RuntimeError, match="(?i)(finite|curvature|diagonal|leaf)"):
        _native.train(np.array([[0, 1]], np.uint8), np.array([0, 1], np.float32), candidates, candidates,
                       **(signed_curvature_options() | {"l2_leaf_reg": l2}))


@pytest.mark.parametrize("score", ["NewtonL2", "NewtonCosine"])
@pytest.mark.parametrize("beta,lam", [(1, -1), (1e30, 1e30)])
def test_newton_structure_rejects_invalid_query_row_curvature(metal_device, score, beta, lam):
    candidates = np.array([0], np.uint32)
    with pytest.raises(RuntimeError, match="(?i)(finite|curvature|weight|score|derivative)"):
        _native.train(np.array([[0, 1]], np.uint8), np.array([0, 1], np.float32), candidates, candidates,
                       **(signed_curvature_options(score_function=score)
                          | {"query_beta": beta, "query_lambda": lam}))


@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
def test_gradient_does_not_inspect_unused_overflowing_query_hessians(metal_device, mode):
    candidates = np.array([0], np.uint32)
    result = _native.train(np.array([[0, 1]], np.uint8), np.ones(2, np.float32), candidates, candidates,
        **options("QuerySoftMax", iterations=2, depth=1, group_offsets=np.array([0, 2], np.uint32),
                  leaf_estimation_method="Gradient", leaf_estimation_backtracking=mode,
                  query_beta=1e30, query_lambda=1e30))
    np.testing.assert_array_equal(result.leaf_values, 0)
    np.testing.assert_allclose(result.loss, np.log(2), rtol=2e-6)


@pytest.mark.parametrize("mode", ["AnyImprovement", "Armijo"])
@pytest.mark.parametrize("steps", [1, 2])
def test_query_backtracking_halves_until_first_acceptance_even_beyond_iteration_budget(metal_device, mode, steps):
    candidates = np.array([0], np.uint32)
    targets = np.ones(2, np.float32)
    initial = np.array([0, -10], np.float32)
    point, _, trace = leaf_reference(targets, initial, np.ones(2), [0, 2], np.array([0, 1]), 2,
        objective="QuerySoftMax", l2_leaf_reg=0.01, query_lambda=0,
        leaf_estimation_iterations=steps, leaf_estimation_backtracking=mode)
    if steps == 2:
        assert trace == [(1.0, False), (0.5, False), (0.25, False), (0.125, False), (0.0625, True)]
    else:
        assert trace == [(1.0, True)]  # A single iteration bypasses the acceptance test.
    result, _ = compare(np.array([[0, 1]], np.uint8), targets, candidates, candidates,
        **options("QuerySoftMax", iterations=1, depth=1, learning_rate=1, l2_leaf_reg=0.01,
                  initial_predictions=initial, group_offsets=np.array([0, 2], np.uint32), query_lambda=0,
                  leaf_estimation_iterations=steps, leaf_estimation_backtracking=mode))
    np.testing.assert_allclose(result.leaf_values[0], point, rtol=2e-5, atol=1e-5)
    assert (result.loss[1] < result.loss[0]) == (steps == 2)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("bootstrap", ["Bernoulli", "Bayesian"])
def test_queries_are_normalized_before_structure_bootstrap(metal_device, objective, bootstrap):
    # Frozen noisy fixture deliberately changes its winning structure if query
    # statistics are incorrectly recomputed from the bootstrap weights.
    rng = np.random.default_rng(0)
    bins = rng.integers(0, 4, (4, 32), dtype=np.uint8)
    targets = rng.normal(size=32).astype(np.float32)
    if objective == "QuerySoftMax":
        targets = np.exp(targets).astype(np.float32)
    weights = rng.uniform(0.25, 3, 32).astype(np.float32)
    initial = rng.normal(0, 0.3, 32).astype(np.float32)
    groups = np.arange(0, 33, 4, dtype=np.uint32)
    features = np.repeat(np.arange(4, dtype=np.uint32), 3)
    borders = np.tile(np.arange(3, dtype=np.uint32), 4)
    seed, absolute = 238, 11
    draws = uniforms(targets.size, seed=seed, iteration=absolute, stream=0)
    factors = ((draws < np.float32(0.55)).astype(np.float32) if bootstrap == "Bernoulli"
               else np.power(-np.log(draws.astype(np.float64) + 1e-20), 1.5).astype(np.float32))
    reference_params = options(objective, iterations=1, sample_weight=weights, initial_predictions=initial,
                               group_offsets=groups, leaf_estimation_iterations=2)
    expected = train_reference(bins, targets, features, borders,
                                **reference_params, bootstrap_factors=[factors])
    correct, _ = structure_reference(bins, targets, initial, weights, groups, features, borders,
        objective=objective, depth=2, l2_leaf_reg=2, score_function="Cosine", bootstrap_factors=factors)
    wrong, _ = structure_reference(bins, targets, initial, weights * factors, groups, features, borders,
        objective=objective, depth=2, l2_leaf_reg=2, score_function="Cosine")
    assert correct != wrong, "Fixture must distinguish bootstrap-before-query-normalization"
    sampled = _native.train(bins, targets, features, borders, **reference_params,
                            bootstrap_type=bootstrap, random_seed=seed, iteration_offset=absolute,
                            **({"subsample": 0.55} if bootstrap == "Bernoulli" else {"bagging_temperature": 1.5}))
    for name in ("depths", "split_features", "split_bins"):
        np.testing.assert_array_equal(getattr(sampled, name), expected[name])
    for name in ("leaf_values", "leaf_weights", "predictions", "loss"):
        np.testing.assert_allclose(getattr(sampled, name), expected[name], rtol=3e-4, atol=5e-5)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_zero_weight_extreme_rows_do_not_poison_query_statistics(metal_device, objective, method):
    bins = np.array([[0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0]], np.uint8)
    targets = np.array([0, 3e38, 1, 2, 3e38, 0], np.float32)
    initial = np.array([0.2, -3e38, -0.3, 0.1, 3e38, 0.4], np.float32)
    weights = np.array([1, 0, 1, 2, 0, 1], np.float32)
    compare(bins, targets, np.array([0, 1], np.uint32), np.array([0, 0], np.uint32),
        **options(objective, sample_weight=weights, initial_predictions=initial,
                  group_offsets=np.array([0, 3, 6], np.uint32), query_beta=3,
                  leaf_estimation_method=method))


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("mode", ["No", "Armijo"])
def test_each_permutation_uses_its_full_queries_for_shared_split_leaf_estimation(metal_device, objective, method, mode):
    bins, targets, weights, groups, initial, features, borders = problem(objective)
    matrices = [np.roll(bins, p * 29, axis=1).copy() for p in range(4)]
    if mode == "Armijo":
        # Preserve signal so accepted objective improvements exceed float32 loss
        # resolution. Fully shuffled weak signals can converge within one ULP.
        for p, shuffled in enumerate(matrices):
            matrix = bins.copy()
            matrix[:, p::5] = shuffled[:, p::5]
            matrices[p] = matrix
    cursors = np.stack([initial + np.float32(p * 0.03) for p in range(4)])
    params = options(objective, iterations=3, group_offsets=groups, sample_weight=weights,
                     leaf_estimation_method=method, leaf_estimation_backtracking=mode)
    expected_losses = [query_loss(targets, cursors[-1], weights, groups, objective)]
    expected_values, expected_masses, expected_splits = [], [], []
    with _native.Session(bins, targets, features, borders, **params) as session:
        session.configure_permutations(matrices, initial_predictions=cursors)
        for chosen in [2, 0, 1]:
            splits, _ = structure_reference(matrices[chosen], targets, cursors[chosen], weights,
                groups, features, borders, objective=objective, depth=2, l2_leaf_reg=2, score_function="Cosine")
            for p, matrix in enumerate(matrices):
                ids = np.zeros(targets.size, np.int64)
                for level, (feature, border) in enumerate(splits):
                    ids |= (matrix[feature] > border).astype(np.int64) << level
                point, masses, _ = leaf_reference(targets, cursors[p], weights, groups, ids,
                    1 << len(splits), objective=objective, l2_leaf_reg=2,
                    leaf_estimation_method=method, leaf_estimation_iterations=4,
                    leaf_estimation_backtracking=mode)
                values = point * np.float32(0.2)
                cursors[p] += values[ids]
            expected_values.append(values)
            expected_masses.append(masses)
            expected_splits.append(splits)
            expected_losses.append(query_loss(targets, cursors[-1], weights, groups, objective))
            session.select_permutation(chosen)
            session.step()
            np.testing.assert_allclose(session.permutation_state["predictions"], cursors, rtol=4e-4, atol=7e-5)
        result = session.result()
        final_state = session.permutation_state
    for tree, splits in enumerate(expected_splits):
        assert result.depths[tree] == len(splits)
        np.testing.assert_array_equal(result.split_features[tree, :len(splits)], [s[0] for s in splits])
        np.testing.assert_array_equal(result.split_bins[tree, :len(splits)], [s[1] for s in splits])
        np.testing.assert_allclose(result.leaf_values[tree, :len(expected_values[tree])],
                                   expected_values[tree], rtol=4e-4, atol=7e-5)
        np.testing.assert_allclose(result.leaf_weights[tree, :len(expected_masses[tree])], expected_masses[tree])
    np.testing.assert_allclose(result.loss, expected_losses, rtol=3e-4, atol=7e-5)
    np.testing.assert_array_equal(result.predictions, final_state["predictions"][-1])


def test_float32_query_backtracking_boundary_keeps_a_valid_adjacent_oracle_iterate(metal_device):
    bins, targets, weights, groups, initial, features, borders = problem("QuerySoftMax")
    matrices = [np.roll(bins, p * 29, axis=1).copy() for p in range(4)]
    cursors = np.stack([initial + np.float32(p * 0.03) for p in range(4)])
    splits, _ = structure_reference(matrices[2], targets, cursors[2], weights, groups, features, borders,
        objective="QuerySoftMax", depth=2, l2_leaf_reg=2, score_function="Cosine")
    with _native.Session(bins, targets, features, borders,
            **options("QuerySoftMax", iterations=1, group_offsets=groups, sample_weight=weights,
                      leaf_estimation_method="Gradient", leaf_estimation_backtracking="Armijo")) as session:
        session.configure_permutations(matrices, initial_predictions=cursors)
        session.select_permutation(2)
        session.step()
        actual = session.permutation_state["predictions"]
    for p in (1, 3):
        ids = np.zeros(targets.size, np.int64)
        for level, (feature, border) in enumerate(splits):
            ids |= (matrices[p][feature] > border).astype(np.int64) << level
        adjacent_cursors, numerators = [], []
        for steps in (3, 4):
            point, _, _ = leaf_reference(targets, cursors[p], weights, groups, ids, 1 << len(splits),
                objective="QuerySoftMax", l2_leaf_reg=2, leaf_estimation_method="Gradient",
                leaf_estimation_iterations=steps, leaf_estimation_backtracking="No")
            adjacent_cursors.append(cursors[p] + np.float32(0.2) * point[ids])
            numerators.append(query_terms(targets, cursors[p] + point[ids], weights,
                                          groups, "QuerySoftMax")[2])
        # At F approximately 4950, the remaining improvements are 3.3e-5 and
        # 9.5e-5, each below the 4.9e-4 ULP of the CUDA float objective.
        improvement = numerators[0] - numerators[1]
        assert 0 < improvement < float(np.spacing(np.float32(numerators[0])))
        assert any(np.allclose(actual[p], expected, rtol=2e-5, atol=2e-7)
                   for expected in adjacent_cursors)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("bootstrap", [{"bootstrap_type": "Bernoulli", "subsample": 0.6},
                                      {"bootstrap_type": "MVS", "subsample": 0.6}])
def test_four_query_permutation_cursors_resume_with_absolute_sampling_state(metal_device, tmp_path, objective, bootstrap):
    bins, targets, weights, groups, initial, features, borders = problem(objective)
    matrices = [np.roll(bins, permutation * 29, axis=1).copy() for permutation in range(4)]
    initial_cursors = np.stack([initial + np.float32(permutation * 0.03) for permutation in range(4)])
    params = options(objective, iterations=5, group_offsets=groups, sample_weight=weights,
                     leaf_estimation_backtracking="Armijo", random_strength=0.7,
                     random_seed=239, **bootstrap)
    choices = [2, 0, 3, 1, 2]

    def run(config, selected, state=None):
        with _native.Session(bins, targets, features, borders, **config) as session:
            session.configure_permutations(matrices,
                initial_predictions=initial_cursors if state is None else state["predictions"],
                mvs_lambdas=None if state is None else state["mvs_lambdas"],
                mvs_valid=None if state is None else state["mvs_valid"])
            for chosen in selected:
                session.select_permutation(chosen)
                session.step()
            return session.result(), session.permutation_state

    full, full_state = run(params, choices)
    first, state = run(params, choices[:2])
    path = tmp_path / "query-cursors.npz"
    np.savez(path, **state)
    with np.load(path, allow_pickle=False) as saved:
        restored = {key: saved[key].copy() for key in state}
    rest, rest_state = run(params | {"iterations": 3, "iteration_offset": 2}, choices[2:], restored)
    for name in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(np.concatenate([getattr(first, name), getattr(rest, name)]), getattr(full, name))
    np.testing.assert_array_equal(np.concatenate([first.loss, rest.loss[1:]]), full.loss)
    np.testing.assert_array_equal(rest.predictions, full.predictions)
    for name in full_state:
        np.testing.assert_array_equal(rest_state[name], full_state[name])


@pytest.mark.parametrize("change", [
    {"group_offsets": None}, {"group_offsets": []}, {"group_offsets": [0]},
    {"group_offsets": [1, 6]}, {"group_offsets": [0, 5]}, {"group_offsets": [0, 7]},
    {"group_offsets": [0, 3, 3, 6]}, {"group_offsets": [0, 4, 2, 6]},
    {"group_offsets": [0.0, 3.0, 6.0]}, {"group_offsets": [0, -1, 6]},
    {"query_beta": np.nan}, {"query_beta": np.inf}, {"query_lambda": np.nan},
    {"query_lambda": -np.inf}, {"query_beta": 1e100},
])
def test_invalid_query_configuration_fails_before_loading_metal(monkeypatch, change):
    def forbidden():
        pytest.fail("Invalid query configuration reached Metal library loading")
    monkeypatch.setattr(_native, "build_library", forbidden)
    empty = np.array([], np.uint32)
    with pytest.raises((ValueError, TypeError)):
        _native.train(np.zeros((1, 6), np.uint8), np.ones(6, np.float32), empty, empty,
            **(options("QuerySoftMax", depth=0, group_offsets=np.array([0, 3, 6], np.uint32)) | change))


@pytest.mark.parametrize("targets", [[-1, 1], [0, 0]])
def test_query_softmax_rejects_invalid_target_mass_before_loading_metal(monkeypatch, targets):
    monkeypatch.setattr(_native, "build_library", lambda: pytest.fail("Invalid query targets reached GPU"))
    empty = np.array([], np.uint32)
    with pytest.raises(ValueError):
        _native.train(np.zeros((1, 2), np.uint8), np.array(targets, np.float32), empty, empty,
            **options("QuerySoftMax", depth=0, group_offsets=np.array([0, 2], np.uint32)))
