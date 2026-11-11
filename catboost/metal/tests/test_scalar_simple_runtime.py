"""Private Simple leaves preserve CUDA's sampled Count and signed weak mass.

Expected bootstrap support comes from the independent integer RNG reference;
expected Combination leaves come from component equations, without CPU fitting.
"""

import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _native
from test_bootstrap import uniforms
from test_combination_runtime import (
    component, install_seed_callback, no_cpu_fit, problem, session, terms,
)


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Simple leaves require actual Apple Silicon Metal",
)


def _set_simple(model, bootstrap=0, seed=0):
    error = ct.create_string_buffer(2048)
    objective = _native.ObjectiveOptions(19, 3, 0, 0)
    model._check(model._lib.cbm_session_set_objective(
        model._handle, ct.byref(objective), error, len(error)), error)
    options = _native.BootstrapOptions(bootstrap, seed, 0, 0, 1, .25, 0, 0, 0, 0, 0, 0)
    model._check(model._lib.cbm_session_set_bootstrap(
        model._handle, ct.byref(options), error, len(error)), error)


def _support_fixture(bootstrap, *, mixed):
    for seed in range(4096):
        u = uniforms(4, seed=seed)
        # Poisson's first log-product test yields zero iff U <= 1-subsample.
        # Keep clear of float log/comparison boundaries in both algorithms.
        threshold = .25 if bootstrap == 2 else .75
        if np.min(np.abs(u - threshold)) < .01:
            continue
        retained = u < threshold if bootstrap == 2 else u > threshold
        matching = retained.any() and not retained.all() if mixed else not retained.any()
        if matching:
            return seed, retained
    raise AssertionError("No bounded bootstrap support fixture found")


def _root_data(weights):
    return dict(bins=np.zeros((3, 4), np.uint8),
                targets=np.array([.1, .3, .6, .9], np.float32),
                weights=np.asarray(weights, np.float32), cursor=np.zeros(4, np.float32),
                offsets=np.array([0, 4], np.uint32), l2=0)


@pytest.mark.parametrize("bootstrap", [2, 3], ids=["Bernoulli", "Poisson"])
def test_zero_bootstrap_count_returns_zero_for_occupied_original_root(bootstrap):
    seed, retained = _support_fixture(bootstrap, mixed=False)
    assert not retained.any()
    data = _root_data(np.ones(4))
    with session(data, [component("RMSE", 1)], depth=0, iterations=1,
                 leaf_iterations=1, score="L2") as model:
        _set_simple(model, bootstrap, seed)
        step = model.step()
        assert step.depth == 0
        np.testing.assert_array_equal(step.leaf_values, [0])
        np.testing.assert_array_equal(step.leaf_weights, [0])
        np.testing.assert_array_equal(model.result().predictions, data["cursor"])


@pytest.mark.parametrize("bootstrap", [2, 3], ids=["Bernoulli", "Poisson"])
def test_retained_zero_weight_rows_use_normalized_zero_regularization(bootstrap):
    seed, retained = _support_fixture(bootstrap, mixed=True)
    # Kept rows have original weight zero; positive-weight rows are dropped.
    # CUDA Count remains positive, but shared catboost_options.cpp:357-358
    # normalizes exactly-zero L2 to 1e-20 before any Simple leaf estimation.
    # Consequently this valid zero-mass root is 0/(0+1e-20), not 0/0.
    data = _root_data(~retained)
    assert retained.any() and data["weights"].sum() > 0
    with session(data, [component("RMSE", 1)], depth=0, iterations=1,
                 leaf_iterations=1, score="L2") as model:
        _set_simple(model, bootstrap, seed)
        step = model.step()
        assert step.depth == 0
        np.testing.assert_array_equal(step.leaf_values, [0])
        np.testing.assert_array_equal(step.leaf_weights, [0])
        np.testing.assert_array_equal(model.result().predictions, data["cursor"])


@pytest.mark.parametrize("depth", [0, 1])
def test_combination_simple_exports_finite_negative_weak_masses(depth):
    data = problem()
    for key in ("targets", "weights", "cursor"):
        data[key] = data[key][:67].copy()
    data["targets"] = np.clip(data["targets"], 0, 1)
    data["bins"] = data["bins"][:, :67].copy()
    data["offsets"] = np.array([0, 19, 40, 67], np.uint32)
    components = [component("YetiRank", 10000, permutations=3, decay=.8),
                  component("RMSE", .001)]
    with session(data, components, depth=depth, iterations=1,
                 leaf_iterations=1, score="L2") as model:
        _set_simple(model)
        callback, consumed = install_seed_callback(model)
        step = model.step()
        assert step.depth == depth
        assert len(consumed) == 1, "Simple must reuse its weak target without leaf oracle draws"
        ids = np.zeros(67, np.uint32)
        for level, (feature, border) in enumerate(zip(step.split_features, step.split_bins)):
            ids |= np.uint32(data["bins"][feature] > border) << level
        gradient, _, weak_mass, _, _ = terms(
            data, components, data["cursor"], iter(consumed))
        leaves = 1 << depth
        sums = np.bincount(ids, weights=gradient, minlength=leaves)
        masses = np.bincount(ids, weights=weak_mass, minlength=leaves)
        counts = np.bincount(ids, minlength=leaves)
        expected = np.zeros(leaves, np.float32)
        occupied = counts > 0
        expected[occupied] = sums[occupied] / (masses[occupied] + data.get("l2", 2))
        expected *= np.float32(.2)
        assert np.any(masses < 0), "Fixture must exercise signed weak masses"
        np.testing.assert_allclose(step.leaf_weights, masses, rtol=5e-5, atol=1e-4)
        np.testing.assert_allclose(step.leaf_values, expected, rtol=8e-5, atol=3e-6)
        np.testing.assert_allclose(model.result().predictions,
            np.float32(data["cursor"] + expected[ids]), rtol=8e-5, atol=3e-6)
        assert callback is not None  # retain callback storage through every GPU call
