"""Seeded GPU greedy sampling, continuation, validation, and original-weight leaves."""
import ctypes as ct
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from catboost_metal import _greedy
from catboost_metal._greedy_training import run_training
from catboost_metal._native import BootstrapOptions, ScoreNoiseOptions
from test_greedy_objectives import problem
from test_greedy_training import route


@pytest.fixture(autouse=True)
def actual_gpu_without_cpu_fitting(monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU CatBoost fitting is forbidden in GPU tests")
    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


def options(bootstrap, policy, **extra):
    return dict(iterations=5, depth=3, max_leaves=5 if policy == "Lossguide" else None,
        learning_rate=.17, l2_leaf_reg=.8, bias=.1, score_function="Cosine",
        grow_policy=policy, bootstrap_type=bootstrap, subsample=.37,
        bagging_temperature=2.4, random_seed=2**63 + 215, iteration_offset=19,
        random_strength=.6, **extra)


def same_forest(left, right):
    assert len(left.trees) == len(right.trees)
    for a, b in zip(left.trees, right.trees):
        np.testing.assert_array_equal(a.nodes, b.nodes)
        np.testing.assert_array_equal(a.leaf_values, b.leaf_values)
        np.testing.assert_array_equal(a.leaf_weights, b.leaf_weights)
    np.testing.assert_array_equal(left.predictions, right.predictions)
    np.testing.assert_array_equal(left.loss, right.loss)


@pytest.mark.parametrize("bootstrap", _greedy.BOOTSTRAPS)
@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
def test_bootstrap_noise_snapshot_resume_preserves_absolute_seeded_iteration(tmp_path, bootstrap, policy):
    bins, y, w, cf, cb = problem("RMSE")
    params = options(bootstrap, policy)
    full = run_training(bins, y, cf, cb, sample_weight=w, **params)
    assert full.stats["bootstrap_state"] == {"iteration_offset": 24, "mvs_lambda": None}
    path = tmp_path / "greedy-bootstrap.npz"
    partial = run_training(bins, y, cf, cb, sample_weight=w, **params,
        save_snapshot=True, snapshot_file=path, snapshot_interval=0,
        callback=lambda info: info.iteration < 2)
    assert partial.trained_iterations == 2
    assert partial.stats["bootstrap_state"]["iteration_offset"] == 21
    resumed = run_training(bins, y, cf, cb, sample_weight=w, **params,
        save_snapshot=True, snapshot_file=path, snapshot_interval=0)
    assert resumed.resumed_iterations == 2
    assert resumed.stats["bootstrap_state"] == full.stats["bootstrap_state"]
    same_forest(full, resumed)


@pytest.mark.parametrize("bootstrap", _greedy.BOOTSTRAPS)
def test_manual_session_continuation_preserves_seed_and_original_weight_leaf_equations(bootstrap):
    bins, y, w, cf, cb = problem("RMSE")
    params = options(bootstrap, "Lossguide")
    params.update(iterations=4, depth=1, max_leaves=2)
    full = _greedy.train(bins, y, cf, cb, sample_weight=w, **params)
    first_params = {**params, "iterations": 2}
    with _greedy.TrainingSession(bins, y, cf, cb, sample_weight=w, **first_params) as session:
        assert session.random_seed == params["random_seed"]
        assert session.iteration_offset == params["iteration_offset"]
        assert session._bootstrap_options.random_seed_low == 215
        assert session._bootstrap_options.random_seed_high == 2**31
        assert session.bootstrap_state["iteration_offset"] == 19
        session.step(); session.step()
        assert session.bootstrap_state["iteration_offset"] == 21
        first = session.result()
    second_params = {**params, "iterations": 2, "iteration_offset": params["iteration_offset"] + 2,
        "initial_predictions": first.predictions}
    second = _greedy.train(bins, y, cf, cb, sample_weight=w, **second_params)
    np.testing.assert_array_equal(full.predictions, second.predictions)
    for expected, actual in zip(full.trees[2:], second.trees):
        np.testing.assert_array_equal(expected.nodes, actual.nodes)
        np.testing.assert_array_equal(expected.leaf_values, actual.leaf_values)
    previous = np.full(len(y), params["bias"], np.float32)
    for tree in full.trees:
        assignments = route(tree, bins)
        for leaf in range(len(tree.leaf_values)):
            keep = assignments == leaf
            weight = w[keep].sum(dtype=np.float64)
            numerator = (w[keep].astype(float) * (y[keep].astype(float) - previous[keep])).sum()
            expected = params["learning_rate"] * numerator / (weight + params["l2_leaf_reg"])
            np.testing.assert_allclose(tree.leaf_weights[leaf], weight, rtol=2e-6)
            np.testing.assert_allclose(tree.leaf_values[leaf], expected, rtol=2e-5, atol=2e-7)
        previous += tree.leaf_values[assignments]


@pytest.mark.parametrize("change", [{"random_seed": 2**63 + 216}, {"iteration_offset": 20},
    {"subsample": .6}, {"bagging_temperature": 1.3}, {"random_strength": .5}])
def test_sampling_snapshot_configuration_changes_are_rejected(tmp_path, change):
    bins, y, w, cf, cb = problem("RMSE")
    params = options("Bernoulli", "Lossguide")
    path = tmp_path / "sampling.npz"
    run_training(bins, y, cf, cb, sample_weight=w, **params,
        save_snapshot=True, snapshot_file=path, callback=lambda _: False)
    with pytest.raises(ValueError, match="match|fingerprint|configuration"):
        run_training(bins, y, cf, cb, sample_weight=w, **{**params, **change},
            save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize("bad,match", [({"bootstrap_type": "MVS"}, "MVS"),
    ({"random_seed": -1}, "random_seed"), ({"random_seed": 2**64}, "random_seed"),
    ({"random_seed": True}, "random_seed"), ({"iteration_offset": 2**32-1}, "iteration_offset"),
    ({"bagging_temperature": -.1}, "bagging_temperature"), ({"bagging_temperature": float("inf")}, "bagging_temperature"),
    ({"subsample": 0}, "subsample"), ({"subsample": 1.1}, "subsample"),
    ({"bootstrap_type": "Poisson", "subsample": 1}, "subsample"),
    ({"random_strength": -.01}, "random_strength"), ({"random_strength": float("nan")}, "random_strength")])
def test_sampling_python_validation(bad, match):
    bins, y, w, cf, cb = problem("RMSE", rows=31)
    with pytest.raises(ValueError, match=match):
        _greedy.TrainingSession(bins, y, cf, cb, sample_weight=w,
            **{**options("No", "Lossguide"), **bad})


def test_sampling_setters_reject_post_step_mutation_and_reserved_fields():
    bins, y, w, cf, cb = problem("RMSE", rows=31)
    with _greedy.TrainingSession(bins, y, cf, cb, sample_weight=w, **options("No", "Lossguide")) as session:
        bootstrap = BootstrapOptions(2, 215, 2**31, 19, 1., .37, 0, 0, 0, 0, 1, 0)
        noise = ScoreNoiseOptions(.5, 1, 0, 0)
        error = ct.create_string_buffer(2048)
        for name, value in (("bootstrap", bootstrap), ("score_noise", noise)):
            function = getattr(session._lib, "cbm_greedy_session_set_" + name)
            assert function(session._handle, ct.byref(value), error, len(error)) != 0
        session.step()
        bootstrap.reserved0 = noise.reserved0 = 0
        for name, value in (("bootstrap", bootstrap), ("score_noise", noise)):
            function = getattr(session._lib, "cbm_greedy_session_set_" + name)
            assert function(session._handle, ct.byref(value), error, len(error)) != 0
            assert "before the first tree" in error.value.decode()


@pytest.mark.parametrize("bootstrap", ["Bayesian", "Bernoulli", "Poisson"])
def test_seed_high_bits_and_iteration_offset_change_sampled_forest(bootstrap):
    rng = np.random.default_rng(819)
    bins = rng.integers(0, 7, (4, 127), dtype=np.uint8)
    y = rng.normal(size=127).astype(np.float32)
    cf, cb = np.repeat(np.arange(4), 6), np.tile(np.arange(6), 4)
    params = options(bootstrap, "Lossguide")
    params.update(iterations=2, random_strength=0, subsample=.12)
    reference = _greedy.train(bins, y, cf, cb, **params)
    replay = _greedy.train(bins, y, cf, cb, **params)
    same_forest(reference, replay)
    for changed in ({"random_seed": 215}, {"iteration_offset": 20}):
        alternate = _greedy.train(bins, y, cf, cb, **{**params, **changed})
        assert not np.array_equal(reference.predictions, alternate.predictions)
