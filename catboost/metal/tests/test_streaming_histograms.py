"""Bounded feature-tile training checked against all-candidate row predicates.

The oracle never constructs feature tiles or calls a model trainer. Dyadic
targets and observation weights make histogram sums exact in the main fixture;
ordered float32 score accumulation retains the CUDA candidate tie semantics.
Only occupied parents are represented, keeping the depth-16 oracle small.
"""

import json
import math
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRegressor

from catboost_metal import _native
from catboost_metal.regressor import _model_json
from cuda_scalar_reference import objective_terms, weighted_loss
from cuda_auxiliary_score_reference import score_children
from test_bootstrap import uniforms
from test_score_noise import _normal


HISTOGRAM_BUDGET = 256 * 1024 * 1024
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Streaming histogram training requires Apple Silicon",
)


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Streaming histogram checks must execute training on Metal")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def _options(**changes):
    return dict(iterations=1, depth=16, learning_rate=0.25, l2_leaf_reg=0.5,
                bias=0, score_function="L2", objective="RMSE",
                leaf_estimation_iterations=1, leaf_estimation_backtracking="No") | changes


def _problem(objective="RMSE"):
    rng = np.random.default_rng(93581)
    bins = rng.integers(0, 65, size=(9, 259), dtype=np.uint8)
    feature_types = np.zeros(9, np.uint8)
    feature_types[[2, 8]] = 1
    for feature in (2, 8):
        bins[feature] = rng.permutation(np.arange(259) % 64).astype(np.uint8)
        bins[feature, ::37] = 255  # Unknown categories have no equality candidate.
    signal = (bins[0].astype(np.float32) / 8 - bins[4].astype(np.float32) / 16
              + 5 * (bins[8] == 11) + 3 * (bins[2] == 31)
              + rng.integers(-16, 17, 259).astype(np.float32) / 16)
    targets = signal if objective == "RMSE" else (signal > 2).astype(np.float32)
    weights = (2.0 ** rng.integers(-2, 3, 259)).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(9, dtype=np.uint32), 64)
    borders = np.tile(np.arange(64, dtype=np.uint32), 9)
    order = rng.permutation(features.size)
    return (bins, targets.astype(np.float32), weights, features[order], borders[order],
            feature_types[features[order]])


def _ordered_score(sums, weights, regularization, cosine):
    """CUDA score operation order, independent of histogram representation."""
    if not cosine:
        contributions = np.zeros(sums.size, np.float32)
        nonempty = weights > np.float32(1e-20)
        contributions[nonempty] = ((-sums[nonempty] * sums[nonempty])
                                   / (weights[nonempty] + regularization))
        return contributions.cumsum(dtype=np.float32)[-1]
    means = np.divide(sums, weights + regularization, out=np.zeros_like(sums), where=weights > 0)
    numerator = (sums * means).cumsum(dtype=np.float32)[-1]
    squared_terms = weights * means * means
    denominator = np.concatenate((np.array([1e-10], np.float32), squared_terms)).cumsum(dtype=np.float32)[-1]
    return np.float32(-numerator / np.sqrt(denominator))


def _reference(bins, targets, weights, features, borders, types, options):
    """One full candidate search, followed by original-weight leaf estimation."""
    rows = len(targets)
    prediction = np.full(rows, np.float32(options["bias"]), np.float32)
    _, derivative, curvature = objective_terms(targets, prediction, options["objective"])
    gradient = weights * derivative.astype(np.float32)
    hessian = weights * curvature.astype(np.float32)
    score_name = options["score_function"]
    denominators = hessian if score_name.startswith("Newton") else weights.copy()
    cosine = score_name.endswith("Cosine")
    regularization = np.float32(options["l2_leaf_reg"]) or np.float32(1e-20)
    strength = np.float32(options.get("random_strength", 0))
    seed = options.get("random_seed", 0)
    scale = np.float32(0)
    if cosine and strength:
        weak = np.where(np.abs(gradient) < np.float32(1e-15), np.float32(0),
                        gradient / (denominators + np.float32(1e-15)))
        variance = np.sum(denominators.astype(np.float64) * weak.astype(np.float64) ** 2) / rows
        scale = np.float32(math.sqrt(variance) * float(strength) * rows / (rows + 1))
    bootstrap = options.get("bootstrap_type", "No")
    if bootstrap == "Bernoulli":
        selected = uniforms(rows, seed=seed) < np.float32(options["subsample"])
        gradient = gradient * selected
        denominators = denominators * selected
    else:
        assert bootstrap == "No", "This independent oracle implements No/Bernoulli sampling"
    leaf_ids = np.zeros(rows, np.uint32)
    splits, chosen = [], set()
    for level in range(options["depth"]):
        occupied, compact_ids = np.unique(leaf_ids, return_inverse=True)
        count = len(occupied)
        parent_sums = np.bincount(compact_ids, weights=gradient, minlength=count).astype(np.float32)
        parent_weights = np.bincount(compact_ids, weights=denominators, minlength=count).astype(np.float32)
        noise = (np.array([np.float32(_normal(f, seed=seed, stream=level + 1)) * scale
                           for f in range(bins.shape[0])], np.float32)
                 if cosine and strength else np.zeros(bins.shape[0], np.float32))
        scores = np.empty(features.size, np.float32)
        for candidate, (feature, border, kind) in enumerate(zip(features, borders, types)):
            selected = bins[feature] == border if kind else bins[feature] <= border
            selected_sums = np.bincount(compact_ids[selected], weights=gradient[selected],
                                        minlength=count).astype(np.float32)
            selected_weights = np.bincount(compact_ids[selected], weights=denominators[selected],
                                           minlength=count).astype(np.float32)
            other_sums = parent_sums - selected_sums
            other_weights = np.maximum(parent_weights - selected_weights, np.float32(0))
            child_sums, child_weights = np.empty(2 * count, np.float32), np.empty(2 * count, np.float32)
            child_sums[0::2], child_sums[1::2] = ((other_sums, selected_sums) if kind
                                                else (selected_sums, other_sums))
            child_weights[0::2], child_weights[1::2] = ((other_weights, selected_weights) if kind
                                                       else (selected_weights, other_weights))
            scores[candidate] = (score_children(child_sums, child_weights, score_name)
                                 if score_name in ("SolarL2", "LOOL2")
                                 else _ordered_score(child_sums, child_weights, regularization, cosine))
            if cosine:
                scores[candidate] += noise[feature]
        winner = int(np.argmin(scores))
        split = tuple(map(int, (features[winner], borders[winner], types[winner])))
        if split in chosen:
            break
        chosen.add(split)
        splits.append(split)
        feature, border, kind = split
        right = bins[feature] == border if kind else bins[feature] > border
        leaf_ids |= right.astype(np.uint32) << level
    leaves = 1 << len(splits)
    leaf_weights = np.bincount(leaf_ids, weights=weights, minlength=leaves)
    # Leaf fitting uses the original objective weights, even after bootstrap
    # or a Newton structure score selected a different denominator.
    _, original_derivative, original_curvature = objective_terms(targets, prediction, options["objective"])
    leaf_gradients = np.bincount(leaf_ids, weights=weights * original_derivative, minlength=leaves)
    leaf_curvature = np.bincount(leaf_ids, weights=weights * original_curvature, minlength=leaves)
    leaf_values = leaf_gradients / (leaf_curvature + float(regularization) + 1e-20)
    leaf_values[leaf_weights < 1e-20] = 0
    leaf_values *= np.float32(options["learning_rate"])
    predictions = prediction.astype(np.float64) + leaf_values[leaf_ids]
    loss = [weighted_loss(targets, prediction, weights, options["objective"]),
            weighted_loss(targets, predictions, weights, options["objective"])]
    return dict(splits=splits, leaf_ids=leaf_ids, leaf_values=leaf_values,
                leaf_weights=leaf_weights, predictions=predictions, loss=loss)


def _assert_streamed(stats, expected_tiles):
    assert stats["histogram_tiles"] == expected_tiles
    assert 0 < stats["histogram_bytes"] <= HISTOGRAM_BUDGET
    assert stats["kernel_dispatches"] > 0


def _assert_reference(actual, expected):
    depth = len(expected["splits"])
    np.testing.assert_array_equal(actual.depths, [depth])
    actual_splits = list(zip(actual.split_features[0, :depth], actual.split_bins[0, :depth],
                            actual.split_types[0, :depth]))
    assert actual_splits == expected["splits"]
    leaves = 1 << depth
    for name in ("leaf_values", "leaf_weights"):
        np.testing.assert_allclose(getattr(actual, name)[0, :leaves], expected[name], rtol=7e-5, atol=7e-5)
        np.testing.assert_array_equal(getattr(actual, name)[0, leaves:], 0)
    np.testing.assert_allclose(actual.predictions, expected["predictions"], rtol=7e-5, atol=7e-5)
    np.testing.assert_allclose(actual.loss, expected["loss"], rtol=7e-5, atol=7e-6)
    assert np.isfinite(actual.predictions).all()


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"])
def test_streamed_weighted_numeric_and_one_hot_scores_match_full_oracle(score):
    objective = "Logloss" if score.startswith("Newton") else "RMSE"
    bins, targets, weights, features, borders, types = _problem(objective)
    assert 9 * 64 * (1 << 16) * 8 > HISTOGRAM_BUDGET
    assert np.any(features[1:] < features[:-1])  # Original order crosses tile boundaries.
    options = _options(score_function=score, objective=objective)
    expected = _reference(bins, targets, weights, features, borders, types, options)
    actual = _native.train(bins, targets, features, borders, candidate_types=types,
                           sample_weight=weights, **options)
    _assert_streamed(actual.stats, 2)
    _assert_reference(actual, expected)


@pytest.mark.parametrize("score", ["Cosine", "NewtonCosine"])
def test_streamed_bootstrap_and_feature_noise_match_full_oracle(score):
    objective = "Logloss" if score.startswith("Newton") else "RMSE"
    bins, targets, weights, features, borders, types = _problem(objective)
    options = _options(score_function=score, objective=objective, bootstrap_type="Bernoulli",
                       subsample=0.625, random_seed=81927, random_strength=0.75)
    expected = _reference(bins, targets, weights, features, borders, types, options)
    actual = _native.train(bins, targets, features, borders, candidate_types=types,
                           sample_weight=weights, **options)
    _assert_streamed(actual.stats, 2)
    _assert_reference(actual, expected)
    assert actual.leaf_weights.sum(dtype=np.float64) == weights.sum(dtype=np.float64)


def test_tile_merge_keeps_original_candidate_ties_and_zero_span_features(tmp_path):
    rows = np.arange(259)
    bins = np.tile((rows % 2).astype(np.uint8) * 64, (10, 1))
    targets = np.where(rows % 2, 5, 1).astype(np.float32)
    active = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9], np.uint32)
    features = np.repeat(active, 64)[::-1].copy()
    borders = np.tile(np.arange(64, dtype=np.uint32), len(active))[::-1].copy()
    # Every candidate has the same partition. Index0 belongs to the last tile;
    # feature5 has no candidates and a zero span inside the first tile.
    actual = _native.train(bins, targets, features, borders, **_options())
    _assert_streamed(actual.stats, 2)
    np.testing.assert_array_equal(actual.depths, [1])
    assert actual.split_features[0, 0] == 9
    assert actual.split_bins[0, 0] == 63
    expected_values = np.array([130, 129 * 5], np.float64) / (np.array([130, 129]) + 0.5) * 0.25
    np.testing.assert_allclose(actual.predictions, expected_values[rows % 2], rtol=3e-6, atol=3e-6)
    payload = _model_json([np.arange(64, dtype=np.float32) + 0.5 for _ in range(10)], actual, 0, "L2")
    path = tmp_path / "streamed.json"
    path.write_text(json.dumps(payload))
    model = CatBoostRegressor().load_model(str(path), format="json")
    np.testing.assert_allclose(model.predict(bins.T.astype(np.float32)), actual.predictions,
                               rtol=3e-6, atol=3e-6)


def test_large_untiled_histogram_and_long_session_fit_bounded_workspace():
    bins = np.zeros((16, 7), np.uint8)
    targets = np.arange(7, dtype=np.float32) / 8
    features = np.repeat(np.arange(16, dtype=np.uint32), 255)
    borders = np.tile(np.arange(255, dtype=np.uint32), 16)
    assert 16 * 255 * (1 << 16) * 8 > 1024 * 1024 * 1024
    options = _options(iterations=10000)
    with _native.Session(bins, targets, features, borders, **options) as session:
        step = session.step()
        assert step.completed_iterations == 1 and not step.finished
        assert step.depth == 1
        _assert_streamed(step.stats, 8)
        result = session.result()
        assert result.completed_iterations == 1
        _assert_streamed(result.stats, 8)
        expected = 0.25 * targets.sum(dtype=np.float64) / (len(targets) + 0.5)
        np.testing.assert_allclose(result.predictions, expected, rtol=3e-6, atol=3e-6)
        np.testing.assert_array_equal(step.leaf_weights, [7, 0])
        np.testing.assert_array_equal(result.leaf_weights[0, :2], step.leaf_weights)
        np.testing.assert_array_equal(result.leaf_weights[0, 2:], 0)
        np.testing.assert_array_equal(session.predictions(), result.predictions)
    assert session.closed
