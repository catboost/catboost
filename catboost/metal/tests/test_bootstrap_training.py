"""Bootstrap behavior through the public Metal estimator and resumable trainer."""

import json
import platform

import numpy as np
import pytest

from catboost_metal import CatBoostMetalRegressor, _native


BOOTSTRAPS = [
    pytest.param({"bootstrap_type": "Bayesian", "bagging_temperature": 1.5}, id="Bayesian"),
    pytest.param({"bootstrap_type": "Bernoulli", "subsample": .45}, id="Bernoulli"),
    pytest.param({"bootstrap_type": "Poisson", "subsample": .55}, id="Poisson"),
    pytest.param({"bootstrap_type": "MVS", "subsample": .5}, id="MVS-auto"),
    pytest.param({"bootstrap_type": "MVS", "subsample": .5, "mvs_reg": .75}, id="MVS-fixed"),
]


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError("Bootstrap tests must not invoke CPU CatBoost training")

    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    assert _native.device_info()["backend"] == "Metal"


def _problem(rows=257):
    rng = np.random.default_rng(26471)
    features = rng.normal(size=(rows, 6)).astype(np.float32)
    # Several weak competing signals make the fixed different-seed fixture
    # exercise structure sampling without asserting a quality improvement.
    target = (rng.normal(size=rows) + .2 * features[:, 0] - .2 * features[:, 1]).astype(np.float32)
    weights = rng.uniform(.2, 3, rows).astype(np.float32)
    weights[::13] = 0
    return features, target, weights


def _options(**overrides):
    return dict(iterations=5, depth=3, learning_rate=.2, l2_leaf_reg=1,
                border_count=15, score_function="L2", boost_from_average=False,
                random_seed=79247) | overrides


def _assert_same_model(left, right):
    for name in ("depths", "split_features", "split_bins", "split_types"):
        np.testing.assert_array_equal(getattr(left._result, name), getattr(right._result, name))
    for name in ("leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_allclose(getattr(left._result, name), getattr(right._result, name),
                                   rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS)
def test_seeded_training_repeats_and_different_seeds_change_sampling(metal_device, bootstrap):
    features, target, weights = _problem()
    options = _options(**bootstrap)
    first = CatBoostMetalRegressor(**options).fit(features, target, sample_weight=weights)
    repeated = CatBoostMetalRegressor(**options).fit(features, target, sample_weight=weights)
    changed = CatBoostMetalRegressor(**(options | {"random_seed": 853})).fit(
        features, target, sample_weight=weights)
    _assert_same_model(first, repeated)
    assert np.max(np.abs(first.training_predictions_ - changed.training_predictions_)) > 1e-5
    assert first.training_stats_["kernel_dispatches"] > 0
    assert first.training_stats_["device"].startswith("Apple")


@pytest.mark.parametrize("bootstrap", [
    {"bootstrap_type": "Bayesian", "bagging_temperature": 0},
    {"bootstrap_type": "Bernoulli", "subsample": 1},
    {"bootstrap_type": "MVS", "subsample": 1},
])
def test_bootstrap_identity_boundaries_match_disabled_sampling(metal_device, bootstrap):
    features, target, weights = _problem()
    baseline = CatBoostMetalRegressor(**_options(bootstrap_type="No")).fit(
        features, target, sample_weight=weights)
    sampled = CatBoostMetalRegressor(**_options(**bootstrap)).fit(
        features, target, sample_weight=weights)
    _assert_same_model(sampled, baseline)


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS)
def test_final_leaf_estimation_uses_original_observation_weights(metal_device, bootstrap):
    features, target, weights = _problem()
    model = CatBoostMetalRegressor(**_options(iterations=1, l2_leaf_reg=0, **bootstrap)).fit(
        features, target, sample_weight=weights)
    result = model._result
    bins = model._layout.transform(features)
    depth = int(result.depths[0])
    leaves = np.zeros(len(target), np.uint32)
    for level in range(depth):
        feature = int(result.split_features[0, level])
        border = int(result.split_bins[0, level])
        leaves |= (bins[feature] > border).astype(np.uint32) << level
    leaf_weights = np.bincount(leaves, weights=weights, minlength=1 << depth)
    gradients = np.bincount(leaves, weights=weights.astype(np.float64) * target, minlength=1 << depth)
    values = np.divide(gradients, leaf_weights, out=np.zeros(1 << depth), where=leaf_weights > 0) * .2
    np.testing.assert_allclose(result.leaf_weights[0, :1 << depth], leaf_weights, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.leaf_values[0, :1 << depth], values, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS)
def test_saved_bootstrap_state_resumes_the_uninterrupted_sequence(metal_device, tmp_path, bootstrap):
    features, target, weights = _problem()
    options = _options(iterations=7, **bootstrap)
    path = tmp_path / "bootstrap.snapshot"
    # Supplying validation covers both persisted prediction cursors. Keep all
    # trees so early-stopping selection cannot mask a resumption mismatch.
    fit_options = dict(sample_weight=weights, eval_set=(features, target, weights), use_best_model=False)
    first = CatBoostMetalRegressor(**(options | {"iterations": 3})).fit(
        features, target, save_snapshot=True, snapshot_file=path, **fit_options)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["metadata"].item()))
        assert header["completed_iterations"] == 3
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    resumed = CatBoostMetalRegressor(**options).fit(
        features, target, save_snapshot=True, snapshot_file=path, **fit_options)
    uninterrupted = CatBoostMetalRegressor(**options).fit(features, target, **fit_options)
    assert first.tree_count_ == 3 and resumed.tree_count_ == 7
    assert resumed.training_stats_["resumed_iterations"] == 3
    _assert_same_model(resumed, uninterrupted)
    np.testing.assert_allclose(resumed.get_evals_result()["validation"]["RMSE"],
                               uninterrupted.get_evals_result()["validation"]["RMSE"], atol=3e-6)
    state = resumed.training_stats_["bootstrap_state"]
    assert state["iteration_offset"] == 7
    if bootstrap["bootstrap_type"] == "MVS" and "mvs_reg" not in bootstrap:
        assert state["mvs_lambda"] is not None and state["mvs_lambda"] >= 0


def test_snapshot_rejects_different_random_stream_before_gpu(metal_device, monkeypatch, tmp_path):
    features, target, weights = _problem()
    path = tmp_path / "bootstrap.snapshot"
    options = _options(iterations=2, bootstrap_type="Bernoulli", subsample=.5)
    CatBoostMetalRegressor(**options).fit(features, target, sample_weight=weights,
                                         save_snapshot=True, snapshot_file=path)
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Wrong-seed snapshot reached GPU"))
    with pytest.raises(ValueError, match="does not match"):
        CatBoostMetalRegressor(**(options | {"iterations": 4, "random_seed": 18})).fit(
            features, target, sample_weight=weights, save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize("options", [
    {"bootstrap_type": "Invalid"}, {"bootstrap_type": "Bayesian", "subsample": .5},
    {"bootstrap_type": "Bernoulli", "bagging_temperature": 1},
    {"bootstrap_type": "No", "subsample": 1},
    {"bootstrap_type": "MVS", "mvs_reg": -1},
    {"bootstrap_type": "MVS", "mvs_reg": np.inf},
    {"bootstrap_type": "Bernoulli", "subsample": 0},
    {"bootstrap_type": "Bernoulli", "subsample": np.nan},
    {"bootstrap_type": "Poisson", "subsample": 1},
    {"bootstrap_type": "Poisson", "subsample": 1 - 1e-10},
    {"bootstrap_type": "Bayesian", "bagging_temperature": -1},
    {"random_seed": -1}, {"random_seed": 2**64}, {"random_seed": True},
])
def test_invalid_sampling_options_are_rejected_before_gpu(monkeypatch, options):
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Invalid sampling reached GPU"))
    with pytest.raises((ValueError, TypeError)):
        CatBoostMetalRegressor(**_options(**options))
