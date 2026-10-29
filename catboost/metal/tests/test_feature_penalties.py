"""Independent CUDA model-size penalties and weighted split-gain regression tests.

Formula sources: cuda/methods/update_feature_weights.cpp and
cuda/methods/kernel/pointwise_scores.cu::FindOptimalSplitSingleFoldImpl.
"""

import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import _native
from cuda_reference import _score_children
from cuda_scalar_reference import auxiliary_score_children, objective_terms, weighted_loss
from test_backtracking import cuda_leaf_walker
from test_permutation_session import assert_result_equal, problem as permutation_problem


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Feature-penalty tests must not invoke CPU CatBoost training")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal checks require Apple Silicon")
    return _native.device_info()


def options(**overrides):
    return dict(iterations=4, depth=3, learning_rate=0.2, l2_leaf_reg=2, bias=0,
                objective="RMSE", objective_param=None, score_function="Cosine",
                leaf_estimation_method="Newton", leaf_estimation_iterations=3,
                leaf_estimation_backtracking="No", random_seed=713) | overrides


def ctr_penalties(counts, used, strength):
    counts = np.asarray(counts, np.uint32)
    unused = (counts > 0) & ~np.asarray(used, bool)
    maximum = max(1, int(counts[unused].max(initial=0)))
    values = np.ones(counts.shape, np.float32)
    # CUDA computes 1+c/max in float before host pow(double,float strength).
    bases = np.float32(1) + counts[unused].astype(np.float32) / np.float32(maximum)
    values[unused] = np.power(bases.astype(np.float64), -float(np.float32(strength))).astype(np.float32)
    return values


def structure_reference(bins, targets, cursor, sample_weights, features, borders,
                        counts, initial_used, strength, feature_weights, params):
    _, derivative, hessian = objective_terms(targets, cursor, params["objective"], params["objective_param"])
    gradients = (sample_weights.astype(np.float64) * derivative).astype(np.float32).astype(np.float64)
    structure_weights = sample_weights.astype(np.float64)
    if params["score_function"].startswith("Newton"):
        structure_weights = (structure_weights * hessian).astype(np.float32).astype(np.float64)
    regularization = float(np.float32(params["l2_leaf_reg"])) or float(np.float32(1e-20))
    family = params["score_function"].removeprefix("Newton")
    ids = np.zeros(targets.size, np.int64)
    used = np.asarray(initial_used, np.uint8).copy()
    before = np.float32(0)
    splits, trace = [], []
    for level in range(params["depth"]):
        penalties = ctr_penalties(counts, used, strength)
        weighted_scores, gains = [], []
        for feature, border in zip(features, borders):
            child_sums, child_weights = [], []
            for leaf in range(1 << level):
                parent = ids == leaf
                left = parent & (bins[feature] <= border)
                right = parent & ~left
                child_sums.extend([gradients[left].sum(), gradients[right].sum()])
                child_weights.extend([structure_weights[left].sum(), structure_weights[right].sum()])
            if family in ("SolarL2", "LOOL2"):
                raw = auxiliary_score_children(child_sums, child_weights, family)
            else:
                raw = _score_children(np.asarray(child_sums), np.asarray(child_weights), regularization, family)
            weighted = np.float32(np.float32(raw) * penalties[feature])
            gain = np.float32(np.float32(weighted - before) * np.float32(feature_weights[feature]))
            weighted_scores.append(weighted)
            gains.append(gain)
        if not gains:
            break
        winner = int(np.argmin(gains))
        feature, border = int(features[winner]), int(borders[winner])
        trace.append(dict(winner=winner, before=float(before), penalties=penalties.copy(),
                          weighted_scores=np.asarray(weighted_scores), gains=np.asarray(gains)))
        before = weighted_scores[winner]
        if counts[feature] > 0:
            used[feature] = 1
        if (feature, border) in splits:
            break
        splits.append((feature, border))
        ids |= (bins[feature] > border).astype(np.int64) << level
    return splits, ids, used, trace


def data(score="Cosine"):
    rng = np.random.default_rng(828)
    bins = rng.integers(0, 6, (4, 263), dtype=np.uint8)
    signal = (0.8 * (bins[0] > 2) - 1.4 * (bins[1] > 3)
              + 1.1 * (bins[2] > 1) + rng.normal(0, 0.2, bins.shape[1]))
    objective = "CrossEntropy" if score.startswith("Newton") else "RMSE"
    targets = 1 / (1 + np.exp(-signal)) if objective == "CrossEntropy" else signal
    weights = rng.uniform(0.2, 3, targets.size).astype(np.float32)
    weights[::17] = 0
    return (bins, targets.astype(np.float32), weights,
            np.repeat(np.arange(4, dtype=np.uint32), 5),
            np.tile(np.arange(5, dtype=np.uint32), 4), objective)


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"])
@pytest.mark.parametrize("strength", [0, 0.5, 2])
@pytest.mark.parametrize("restored", [False, True])
def test_dynamic_penalties_and_weighted_gains_match_independent_forest_oracle(
        metal_device, score, strength, restored):
    bins, targets, weights, features, borders, objective = data(score)
    counts = np.array([0, 7, 31, 200], np.uint32)
    user_weights = np.array([0.7, 2, 0.35, 0], np.float32)
    used = np.array([0, 0, int(restored), 0], np.uint8)
    params = options(score_function=score, objective=objective)
    cursor = np.zeros(targets.size, np.float32)
    with _native.Session(bins, targets, features, borders, **params, sample_weight=weights) as session:
        session.configure_feature_penalties(counts, model_size_reg=strength,
                                            feature_weights=user_weights, used_features=used)
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], used)
        for _ in range(params["iterations"]):
            splits, leaf_ids, used, _ = structure_reference(
                bins, targets, cursor, weights, features, borders, counts, used, strength, user_weights, params)
            step = session.step()
            assert list(zip(step.split_features.tolist(), step.split_bins.tolist())) == splits
            np.testing.assert_array_equal(session.feature_penalty_state["used_features"], used)
            point, leaf_weights, _ = cuda_leaf_walker(
                targets, cursor, weights, leaf_ids, 1 << len(splits), objective=objective,
                objective_param=None, l2_leaf_reg=params["l2_leaf_reg"],
                leaf_estimation_method="Newton", leaf_estimation_iterations=params["leaf_estimation_iterations"],
                leaf_estimation_backtracking="No")
            values = (point * np.float32(params["learning_rate"])).astype(np.float32)
            np.testing.assert_allclose(step.leaf_values, values, rtol=2e-4, atol=2e-5)
            np.testing.assert_allclose(step.leaf_weights, leaf_weights, rtol=2e-6)
            cursor = (cursor + values[leaf_ids]).astype(np.float32)
            np.testing.assert_allclose(session.predictions(), cursor, rtol=2e-4, atol=3e-5)
            np.testing.assert_allclose(step.loss, weighted_loss(targets, cursor, weights, objective), rtol=2e-5)


@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_default_half_strength_changes_equal_score_ctr_selection(metal_device, score):
    # Identical raw scores: the default .5 model-size penalty makes the
    # high-cardinality CTR less attractive than the low-cardinality CTR.
    bins = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], np.uint8)
    targets = np.array([-2, -2, 2, 2], np.float32)
    features, borders = np.array([0, 1], np.uint32), np.zeros(2, np.uint32)
    params = options(iterations=1, depth=2, score_function=score)
    unpenalized = _native.train(bins, targets, features, borders, **params)
    assert unpenalized.split_features[0, 0] == 0
    with _native.Session(bins, targets, features, borders, **params) as session:
        session.configure_feature_penalties(np.array([100, 1], np.uint32))
        step = session.step()
        assert step.depth == 1 and step.split_features[0] == 1
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], [0, 1])


@pytest.mark.parametrize("score", ["L2", "Cosine", "SolarL2", "LOOL2"])
def test_zero_user_weights_preserve_candidate_index_ties_and_mark_used_ctr(metal_device, score):
    bins = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], np.uint8)
    targets = np.array([-1, -1, 2, 2], np.float32)
    # Candidate order, rather than feature order, controls a zero-gain tie.
    features, borders = np.array([1, 0], np.uint32), np.zeros(2, np.uint32)
    with _native.Session(bins, targets, features, borders,
                         **options(iterations=1, depth=3, score_function=score)) as session:
        session.configure_feature_penalties(np.array([3, 9], np.uint32), feature_weights=[0, 0])
        step = session.step()
        assert step.depth == 1 and step.split_features[0] == 1
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], [0, 1])


def test_model_size_penalty_recomputes_maximum_from_unused_ctrs():
    counts = np.array([0, 10, 100], np.uint32)
    before = ctr_penalties(counts, [0, 0, 0], 0.5)
    after = ctr_penalties(counts, [0, 0, 1], 0.5)
    np.testing.assert_allclose(before, [1, 1 / np.sqrt(1.1), 1 / np.sqrt(2)], rtol=1e-7)
    np.testing.assert_allclose(after, [1, 1 / np.sqrt(2), 1], rtol=1e-7)


def test_user_feature_weights_apply_to_gain_after_subtracting_previous_score(metal_device):
    bins = np.array([[0, 1, 1, 1, 1, 1, 0, 1],
                     [0, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 1, 0, 0, 0]], np.uint8)
    targets = np.array([-3, -3, -2, 3, 2, 4, 1, 1], np.float32)
    features, borders = np.arange(3, dtype=np.uint32), np.zeros(3, np.uint32)
    counts, feature_weights = np.zeros(3, np.uint32), np.array([0.25, 1, 3], np.float32)
    params = options(iterations=1, depth=3, score_function="L2", l2_leaf_reg=1)
    splits, _, _, trace = structure_reference(
        bins, targets, np.zeros(8, np.float32), np.ones(8, np.float32), features, borders,
        counts, np.zeros(3, np.uint8), 0.5, feature_weights, params)
    assert splits == [(2, 0), (1, 0)]
    # At depth one, previous score=-23.4. Weighted gains are approximately
    # [.0375,-1.0167,0], whereas weighting scores directly selects duplicate 2.
    np.testing.assert_allclose(trace[1]["gains"], [0.0375, -1.0166667, 0], atol=1e-6)
    assert int(np.argmin(trace[1]["weighted_scores"] * feature_weights)) == 2
    with _native.Session(bins, targets, features, borders, **params) as session:
        session.configure_feature_penalties(counts, feature_weights=feature_weights)
        step = session.step()
        assert step.depth == 2
        assert list(zip(step.split_features.tolist(), step.split_bins.tolist())) == splits


def test_using_the_largest_ctr_changes_the_next_depths_penalty_denominator(metal_device):
    bins = np.array([[1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                     [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0]], np.uint8)
    targets = np.array([7, -5, 5, 4, -8, 7, -5, -3, -4, 8, 8, -3], np.float32)
    features, borders = np.arange(3, dtype=np.uint32), np.zeros(3, np.uint32)
    counts = np.array([100, 10, 0], np.uint32)
    params = options(iterations=1, depth=3, score_function="L2", l2_leaf_reg=1)
    splits, _, used, trace = structure_reference(
        bins, targets, np.zeros(12, np.float32), np.ones(12, np.float32), features, borders,
        counts, np.zeros(3, np.uint8), 0.5, np.ones(3, np.float32), params)
    assert splits == [(0, 0), (2, 0), (1, 0)]
    np.testing.assert_allclose(trace[1]["penalties"], [1, 1 / np.sqrt(2), 1], rtol=1e-7)
    # If maxUnique stays at 100, feature 1 would win at the second depth.
    stale_penalty = np.array([1, 1 / np.sqrt(1.1), 1], np.float32)
    stale_scores = trace[1]["weighted_scores"] / trace[1]["penalties"] * stale_penalty
    assert int(np.argmin(stale_scores - trace[1]["before"])) == 1
    with _native.Session(bins, targets, features, borders, **params) as session:
        session.configure_feature_penalties(counts)
        step = session.step()
        assert list(zip(step.split_features.tolist(), step.split_bins.tolist())) == splits
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], used)
        np.testing.assert_array_equal(used, [1, 1, 0])


@pytest.mark.parametrize("bootstrap", [
    {"bootstrap_type": "No"},
    {"bootstrap_type": "Bayesian", "bagging_temperature": 1.5},
    {"bootstrap_type": "MVS", "subsample": 0.6},
])
def test_snapshot_restores_used_flags_with_all_permutation_and_sampling_state(metal_device, tmp_path, bootstrap):
    original, matrices, targets, weights, features, borders, initial = permutation_problem(4)
    counts = np.array([300, 7, 60], np.uint32)
    user_weights = np.array([1, 0.7, 1.8], np.float32)
    params = options(iterations=6, depth=2, random_strength=1.3, **bootstrap)
    choices = [2, 0, 3, 1, 2, 0]

    def run(run_options, selected, state=None):
        with _native.Session(original, targets, features, borders, **run_options, sample_weight=weights) as session:
            session.configure_permutations(matrices,
                initial_predictions=initial if state is None else state["predictions"],
                mvs_lambdas=None if state is None else state["mvs_lambdas"],
                mvs_valid=None if state is None else state["mvs_valid"])
            session.configure_feature_penalties(counts, feature_weights=user_weights,
                used_features=None if state is None else state["used_features"])
            for chosen in selected:
                session.select_permutation(chosen)
                session.step()
            return session.result(), session.permutation_state | session.feature_penalty_state

    full, full_state = run(params, choices)
    repeat, repeat_state = run(params, choices)
    assert_result_equal(repeat, full)
    for key in full_state:
        np.testing.assert_array_equal(repeat_state[key], full_state[key])
    first, state = run(params, choices[:3])
    assert state["used_features"].any()
    path = tmp_path / "feature-penalty-state.npz"
    np.savez(path, **state)
    with np.load(path, allow_pickle=False) as archive:
        restored = {key: archive[key].copy() for key in state}
    resumed, resumed_state = run(params | {"iterations": 3, "iteration_offset": 3}, choices[3:], restored)
    for key in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(np.concatenate([getattr(first, key), getattr(resumed, key)]), getattr(full, key))
    np.testing.assert_array_equal(np.concatenate([first.loss, resumed.loss[1:]]), full.loss)
    np.testing.assert_array_equal(resumed.predictions, full.predictions)
    for key in full_state:
        np.testing.assert_array_equal(resumed_state[key], full_state[key])


@pytest.mark.parametrize("case", [
    {"ctr_unique_values": [1, 2]}, {"ctr_unique_values": [1.0] * 4},
    {"ctr_unique_values": [-1, 0, 0, 0]}, {"ctr_unique_values": [2**32, 0, 0, 0]},
    {"model_size_reg": -1}, {"model_size_reg": np.nan}, {"model_size_reg": np.inf},
    {"model_size_reg": 1e100}, {"feature_weights": [1, 2]},
    {"feature_weights": [1, -1, 1, 1]}, {"feature_weights": [1, np.nan, 1, 1]},
    {"feature_weights": [1, np.inf, 1, 1]}, {"used_features": [0, 1]},
    {"used_features": [0, 2, 0, 0]}, {"used_features": [0.0] * 4},
])
def test_invalid_penalties_fail_before_native_mutation(metal_device, monkeypatch, case):
    bins, targets, weights, features, borders, _ = data()

    class ForbiddenNativeCalls:
        def __getattr__(self, name):
            pytest.fail(f"Invalid feature penalties reached native call {name}")

    with _native.Session(bins, targets, features, borders, **options(sample_weight=weights)) as session:
        with monkeypatch.context() as patch:
            patch.setattr(session, "_lib", ForbiddenNativeCalls())
            with pytest.raises((ValueError, TypeError)):
                session.configure_feature_penalties(**({"ctr_unique_values": [0, 10, 30, 100]} | case))


def test_penalty_configuration_cannot_replace_live_forest_state(metal_device):
    bins, targets, weights, features, borders, _ = data()
    with _native.Session(bins, targets, features, borders, **options(sample_weight=weights)) as session:
        session.configure_feature_penalties([0, 10, 30, 100])
        with pytest.raises(ValueError):
            session.configure_feature_penalties([0, 0, 0, 0])
        session.step()
        used = session.feature_penalty_state["used_features"].copy()
        with pytest.raises(ValueError):
            session.configure_feature_penalties([0, 0, 0, 0])
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], used)
