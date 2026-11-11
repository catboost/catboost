"""Deep symmetric-tree correctness without invoking CPU model training."""

import json
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRegressor

from catboost_metal import _native
from catboost_metal.regressor import _model_json
from cuda_extended_reference import train_reference
from cuda_scalar_reference import exact_leaf_value, weighted_loss
from test_backtracking import cuda_leaf_walker
from test_bootstrap import uniforms


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Deep Metal training requires Apple Silicon",
)


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Deep training checks must execute on Metal")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def _options(depth, **overrides):
    options = dict(iterations=1, depth=depth, learning_rate=0.25, l2_leaf_reg=2,
                   bias=0.125, score_function="L2", objective="RMSE",
                   leaf_estimation_iterations=1, leaf_estimation_backtracking="No")
    options.update(overrides)
    return options


def _leaf_ids(bins, result, tree):
    """Evaluate the exported split predicates directly on original row order."""
    ids = np.zeros(bins.shape[1], dtype=np.uint32)
    for level in range(int(result.depths[tree])):
        feature = int(result.split_features[tree, level])
        border = int(result.split_bins[tree, level])
        one_hot = result.split_types is not None and result.split_types[tree, level] != 0
        right = bins[feature] == border if one_hot else bins[feature] > border
        ids |= right.astype(np.uint32) << level
    return ids


def _check_leaf_counts_and_predictions(bins, result, bias, weights=None):
    rows = bins.shape[1]
    weights = np.ones(rows) if weights is None else np.asarray(weights, dtype=np.float64)
    prediction = np.full(rows, float(np.float32(bias)), dtype=np.float64)
    for tree, depth in enumerate(result.depths):
        count = 1 << int(depth)
        ids = _leaf_ids(bins, result, tree)
        expected_weights = np.bincount(ids, weights=weights, minlength=count)
        np.testing.assert_allclose(result.leaf_weights[tree, :count], expected_weights,
                                   rtol=2e-6, atol=2e-6)
        assert np.all(result.leaf_values[tree, :count][expected_weights == 0] == 0)
        assert np.all(result.leaf_values[tree, count:] == 0)
        assert np.all(result.leaf_weights[tree, count:] == 0)
        prediction += result.leaf_values[tree, ids]
    np.testing.assert_allclose(result.predictions, prediction, rtol=3e-5, atol=3e-5)
    assert result.stats["kernel_dispatches"] > 0
    assert np.isfinite(result.loss).all()


def _check_numeric_export(tmp_path, bins, borders, result, bias):
    payload = _model_json(borders, result, bias, "L2")
    path = tmp_path / "deep.json"
    path.write_text(json.dumps(payload))
    model = CatBoostRegressor().load_model(str(path), format="json")
    raw = bins.T.astype(np.float32)
    np.testing.assert_allclose(model.predict(raw), result.predictions, rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), 1 << result.depths)
    np.testing.assert_array_equal(model.calc_leaf_indexes(raw),
                                  np.column_stack([_leaf_ids(bins, result, tree)
                                                   for tree in range(len(result.depths))]))
    restored = CatBoostRegressor()
    binary_path = tmp_path / "deep.cbm"
    model.save_model(str(binary_path))
    restored.load_model(str(binary_path))
    np.testing.assert_array_equal(restored.predict(raw), model.predict(raw))


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_deep_weighted_training_matches_scalar_oracle_and_export(tmp_path, depth):
    rng = np.random.default_rng(71981)
    bins = rng.integers(0, 17, size=(4, 1031), dtype=np.uint8)
    targets = (0.75 * bins[0] + 0.3125 * bins[1] - 0.1875 * bins[2]
               + rng.normal(0, 0.025, bins.shape[1])).astype(np.float32)
    weights = (2.0 ** rng.integers(-2, 3, bins.shape[1])).astype(np.float32)
    weights[::31] = 0
    features = np.repeat(np.arange(4, dtype=np.uint32), 16)
    borders = np.tile(np.arange(16, dtype=np.uint32), 4)
    options = _options(depth, iterations=2, sample_weight=weights)
    expected = train_reference(bins, targets, features, borders, **options)
    actual = _native.train(bins, targets, features, borders, **options)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    for tree, actual_depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[tree, :actual_depth],
                                      expected["split_features"][tree, :actual_depth])
        np.testing.assert_array_equal(actual.split_bins[tree, :actual_depth],
                                      expected["split_bins"][tree, :actual_depth])
    for name in ("leaf_values", "leaf_weights", "predictions"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=7e-5,
                                   atol=7e-5, err_msg=name)
    np.testing.assert_allclose(actual.loss, expected["loss"], rtol=7e-5, atol=7e-6)
    _check_leaf_counts_and_predictions(bins, actual, options["bias"], weights)
    _check_numeric_export(tmp_path, bins,
                          [np.arange(16, dtype=np.float32) + 0.5 for _ in range(4)],
                          actual, options["bias"])


@pytest.fixture(scope="module")
def binary_hypercube():
    rows = np.arange(1 << 16, dtype=np.uint32)
    bins = ((rows[None, :] >> np.arange(16, dtype=np.uint32)[:, None]) & 1).astype(np.uint8)
    coefficients = np.arange(1, 17, dtype=np.float32)
    terms = (2 * bins.astype(np.float32) - 1) * coefficients[:, None]
    return bins, terms


@pytest.mark.parametrize("depth", [10, 12, 16])
def test_requested_deep_levels_are_reached_on_full_binary_hypercube(tmp_path, binary_hypercube, depth):
    bins, terms = binary_hypercube
    features, borders = np.arange(16, dtype=np.uint32), np.zeros(16, dtype=np.uint32)
    targets = terms.sum(axis=0)
    options = _options(depth, bias=0, l2_leaf_reg=0)
    actual = _native.train(bins, targets, features, borders, **options)
    # Every unsplit feature remains balanced in every parent. With zero L2,
    # splitting feature j improves the score by rows*(j+1)^2; already used
    # features improve it by zero. Distinct coefficients force all requested
    # levels in descending feature order, including the sixteenth level.
    np.testing.assert_array_equal(actual.depths, [depth])
    np.testing.assert_array_equal(actual.split_features[0, :depth], np.arange(15, 15 - depth, -1))
    np.testing.assert_array_equal(actual.split_bins[0, :depth], 0)
    np.testing.assert_array_equal(actual.leaf_weights[0, :1 << depth], 1 << (16 - depth))
    expected = options["learning_rate"] * terms[16 - depth:].sum(axis=0)
    np.testing.assert_allclose(actual.predictions, expected, rtol=3e-6, atol=3e-6)
    _check_leaf_counts_and_predictions(bins, actual, 0)
    _check_numeric_export(tmp_path, bins, [np.array([0.5], np.float32) for _ in range(16)],
                          actual, 0)


def test_deep_compact_histograms_keep_empty_numeric_and_one_hot_bins():
    rows = np.arange(1031)
    numeric = ((rows % 5) == 0).astype(np.uint8) * 16
    categories = (rows % 3).astype(np.uint8)
    bins = np.stack([numeric, categories])
    targets = (np.array([-3, 1, 6], np.float32)[categories] + 4 * (numeric > 0)).astype(np.float32)
    features = np.concatenate([np.zeros(16, np.uint32), np.ones(4, np.uint32)])
    borders = np.concatenate([np.arange(16, dtype=np.uint32), np.arange(4, dtype=np.uint32)])
    kinds = np.concatenate([np.zeros(16, np.uint8), np.ones(4, np.uint8)])
    options = _options(16, bias=0, l2_leaf_reg=0)
    actual = _native.train(bins, targets, features, borders, candidate_types=kinds, **options)
    depth = int(actual.depths[0])
    assert depth >= 3
    assert set(actual.split_types[0, :depth]) == {0, 1}
    _check_leaf_counts_and_predictions(bins, actual, 0)
    ids = _leaf_ids(bins, actual, 0)
    counts = np.bincount(ids, minlength=1 << depth)
    sums = np.bincount(ids, weights=targets, minlength=1 << depth)
    expected = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0) * 0.25
    assert np.any(counts == 0)
    np.testing.assert_allclose(actual.leaf_values[0, :1 << depth], expected, rtol=3e-6, atol=3e-6)
    # Numeric bins 1..15 and categorical bin 3 never occur, but their candidate
    # entries still occupy the compact feature metadata and scoring layout.
    np.testing.assert_allclose(actual.predictions, targets * 0.25, rtol=3e-6, atol=3e-6)


def test_long_deep_session_allocates_storage_for_completed_trees():
    rows = 1031
    bins = np.zeros((1, rows), dtype=np.uint8)
    targets = (np.arange(rows) % 17).astype(np.float32) / 8
    empty = np.array([], dtype=np.uint32)
    options = _options(16, iterations=10000, bias=0, l2_leaf_reg=1)
    # A padded allocation for every requested tree would exceed 5 GiB before
    # the first step. The actual trees here each contain a single leaf.
    with _native.Session(bins, targets, empty, empty, **options) as session:
        for completed in range(1, 4):
            step = session.step()
            assert step.completed_iterations == completed
            assert not step.finished
            assert step.depth == 0
            assert step.leaf_values.shape == step.leaf_weights.shape == (1,)
            np.testing.assert_array_equal(step.leaf_weights, [rows])
        result = session.result()
        assert result.completed_iterations == 3
        np.testing.assert_array_equal(result.depths, [0, 0, 0])
        expected = targets.mean(dtype=np.float64) * (1 - (1 - 0.25 * rows / (rows + 1)) ** 3)
        np.testing.assert_allclose(result.predictions, expected, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(session.predictions(), result.predictions, atol=0)
        _check_leaf_counts_and_predictions(bins, result, 0)
    assert session.closed


def test_deep_bootstrap_structure_uses_seeded_sampling_and_leaves_use_original_weights():
    rng = np.random.default_rng(23718)
    bins = rng.integers(0, 17, size=(4, 1031), dtype=np.uint8)
    targets = (0.5 * bins[0] - 0.25 * bins[2] + rng.normal(size=1031)).astype(np.float32)
    weights = (2.0 ** rng.integers(-2, 3, 1031)).astype(np.float32)
    weights[::29] = 0
    features = np.repeat(np.arange(4, dtype=np.uint32), 16)
    borders = np.tile(np.arange(16, dtype=np.uint32), 4)
    seed, fraction = 79247, np.float32(0.55)
    options = _options(16, sample_weight=weights, l2_leaf_reg=0.125)
    selected = uniforms(1031, seed=seed, iteration=0, stream=0) < fraction
    expected_structure = train_reference(
        bins, targets, features, borders, **(options | {"sample_weight": weights * selected}))
    actual = _native.train(bins, targets, features, borders, **options,
                           bootstrap_type="Bernoulli", subsample=float(fraction), random_seed=seed)
    np.testing.assert_array_equal(actual.depths, expected_structure["depths"])
    depth = int(actual.depths[0])
    assert depth > 8
    np.testing.assert_array_equal(actual.split_features[0, :depth],
                                  expected_structure["split_features"][0, :depth])
    np.testing.assert_array_equal(actual.split_bins[0, :depth],
                                  expected_structure["split_bins"][0, :depth])
    _check_leaf_counts_and_predictions(bins, actual, options["bias"], weights)
    ids = _leaf_ids(bins, actual, 0)
    original_weights = np.bincount(ids, weights=weights, minlength=1 << depth)
    original_gradients = np.bincount(
        ids, weights=weights.astype(np.float64) * (targets.astype(np.float64) - options["bias"]),
        minlength=1 << depth)
    expected_values = original_gradients / (original_weights + options["l2_leaf_reg"]) * 0.25
    np.testing.assert_allclose(actual.leaf_values[0, :1 << depth], expected_values, rtol=7e-5, atol=7e-5)
    assert np.any(~selected & (weights > 0))
    assert original_weights.sum() > np.sum(weights * selected, dtype=np.float64)


@pytest.mark.parametrize("mode", ["AnyImprovement", "Armijo"])
def test_deep_poisson_backtracking_matches_independent_leaf_walker(mode):
    rng = np.random.default_rng(7213)
    bins = rng.integers(0, 17, size=(4, 1031), dtype=np.uint8)
    targets = (np.exp((bins[0].astype(np.float32) - 8) / 8)
               + 0.125 * bins[2]).astype(np.float32)
    weights = (2.0 ** rng.integers(-2, 3, 1031)).astype(np.float32)
    weights[::23] = 0
    features = np.repeat(np.arange(4, dtype=np.uint32), 16)
    borders = np.tile(np.arange(16, dtype=np.uint32), 4)
    options = _options(16, objective="Poisson", bias=-3, l2_leaf_reg=0.1,
                       sample_weight=weights, leaf_estimation_iterations=4,
                       leaf_estimation_backtracking=mode)
    actual = _native.train(bins, targets, features, borders, **options)
    depth = int(actual.depths[0])
    assert depth > 8
    ids = _leaf_ids(bins, actual, 0)
    baseline = np.full(1031, options["bias"], dtype=np.float32)
    point, leaf_weights, trace = cuda_leaf_walker(
        targets, baseline, weights, ids, 1 << depth, objective="Poisson",
        l2_leaf_reg=options["l2_leaf_reg"], leaf_estimation_method="Newton",
        leaf_estimation_iterations=options["leaf_estimation_iterations"],
        leaf_estimation_backtracking=mode)
    assert any(not accepted for _, accepted, _ in trace)
    expected_values = point * np.float32(options["learning_rate"])
    np.testing.assert_allclose(actual.leaf_values[0, :1 << depth], expected_values, rtol=7e-5, atol=7e-5)
    np.testing.assert_allclose(actual.leaf_weights[0, :1 << depth], leaf_weights, rtol=2e-6, atol=2e-6)
    expected_predictions = baseline + expected_values[ids]
    np.testing.assert_allclose(actual.predictions, expected_predictions, rtol=7e-5, atol=7e-5)
    expected_loss = weighted_loss(targets, expected_predictions, weights, "Poisson", None)
    np.testing.assert_allclose(actual.loss[-1], expected_loss, rtol=7e-5, atol=7e-5)
    _check_leaf_counts_and_predictions(bins, actual, options["bias"], weights)


@pytest.mark.parametrize("weighted", [False, True])
def test_constant_deep_session_preserves_small_sum_amid_cancellation(weighted):
    rows = 1031
    targets = np.zeros(rows, dtype=np.float32)
    positions = [0, 511, 1030]
    if weighted:
        targets[positions] = [1, 1, -1]
        weights = np.zeros(rows, dtype=np.float32)
        weights[positions] = [2 ** 24, 1, 2 ** 24]
    else:
        targets[positions] = [2 ** 24, 1, -(2 ** 24)]
        weights = np.ones(rows, dtype=np.float32)
    empty = np.array([], dtype=np.uint32)
    actual = _native.train(np.zeros((1, rows), dtype=np.uint8), targets, empty, empty,
                           **_options(16, bias=0, l2_leaf_reg=0, learning_rate=1, sample_weight=weights))
    expected = 1 / weights.sum(dtype=np.float64)
    np.testing.assert_array_equal(actual.depths, [0])
    np.testing.assert_allclose(actual.leaf_values[0, 0], expected, rtol=1e-5, atol=1e-13)
    np.testing.assert_allclose(actual.predictions, expected, rtol=1e-5, atol=1e-13)


def test_deep_exact_weighted_quantiles_match_independent_order_statistics():
    rng = np.random.default_rng(51602)
    bins = rng.integers(0, 17, size=(4, 1031), dtype=np.uint8)
    targets = (rng.normal(0, 3, 1031) + 0.125 * bins[0]).astype(np.float32)
    weights = (2.0 ** rng.integers(-2, 3, 1031)).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(4, dtype=np.uint32), 16)
    borders = np.tile(np.arange(16, dtype=np.uint32), 4)
    options = _options(16, objective="Quantile", objective_param=0.3,
                       leaf_estimation_method="Exact", l2_leaf_reg=0.01, sample_weight=weights)
    actual = _native.train(bins, targets, features, borders, **options)
    depth = int(actual.depths[0])
    assert depth > 8
    ids = _leaf_ids(bins, actual, 0)
    residuals = targets - np.float32(options["bias"])
    expected = np.zeros(1 << depth)
    for leaf in np.unique(ids):
        members = ids == leaf
        expected[leaf] = exact_leaf_value(residuals[members], weights[members], "Quantile", 0.3) * 0.25
    np.testing.assert_allclose(actual.leaf_values[0, :1 << depth], expected, rtol=3e-6, atol=3e-6)
    _check_leaf_counts_and_predictions(bins, actual, options["bias"], weights)
