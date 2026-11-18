"""Signed Combination weights enter CUDA's dot product before its square root."""

import ctypes as ct
import math
import platform

import numpy as np
import pytest

from catboost_metal import _native
from cuda_reference import _score_children
from test_bootstrap import probe
from test_combination_runtime import component, install_seed_callback, problem, session, terms
from test_score_noise import _normal


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Signed Combination noise requires actual Apple Silicon Metal",
)


@pytest.mark.parametrize("rows", [17, 513])
def test_noise_reduction_preserves_signed_weight_contributions(probe, rows):
    weights = np.full(rows, 4, np.float32)
    weights[:rows // 2] = -1
    gradient = np.where(weights > 0, 8, 1).astype(np.float32)
    gradient[-1] = weights[-1] = 0
    weak = np.divide(gradient, weights + np.float32(1e-15))
    expected = np.dot(weights.astype(float), weak.astype(float) ** 2) / rows
    result = probe(8, rows, derivatives=gradient, weights=weights)
    partials = result["values"][:min((rows + 255) // 256, 4096)]
    assert expected > 0
    assert partials.sum(dtype=float) == pytest.approx(expected, rel=2e-6, abs=2e-7)
    if rows == 513:
        assert partials[0] < 0, "A valid positive total can contain a negative partial"
    clipped = np.dot(np.maximum(weights, 0).astype(float), weak.astype(float) ** 2) / rows
    assert not math.isclose(expected, clipped, rel_tol=.01)


@pytest.mark.parametrize("feature_parallel", [False, True])
def test_combination_noise_with_negative_rows_selects_independent_score_winner(feature_parallel):
    data = problem()
    for key in ("targets", "weights", "cursor"):
        data[key] = data[key][:67].copy()
    data["targets"] = np.clip(data["targets"], 0, 1)
    data["bins"] = data["bins"][:, :67].copy()
    data["offsets"] = np.array([0, 19, 40, 67], np.uint32)
    components = [component("YetiRank", .03, permutations=3, decay=.8),
                  component("RMSE", 2), component("YetiRank", .05, permutations=5, decay=.9)]
    seed, strength = 612, 15
    with session(data, components, iterations=1, leaf_iterations=1, depth=1) as model:
        if feature_parallel:
            model.set_feature_activity(np.ones(3, np.uint8))
        error = ct.create_string_buffer(2048)
        bootstrap = _native.BootstrapOptions(0, seed, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0)
        model._check(model._lib.cbm_session_set_bootstrap(model._handle, ct.byref(bootstrap), error, len(error)), error)
        noise = _native.ScoreNoiseOptions(strength, 0, 0, 0)
        model._check(model._lib.cbm_session_set_score_noise(model._handle, ct.byref(noise), error, len(error)), error)
        callback, consumed = install_seed_callback(model)
        step = model.step()
        assert len(consumed) == 4, "Noise statistics reuse the same two-component weak target"
        gradient, _, weights, _, _ = terms(data, components, data["cursor"], iter(consumed[:2]))
        assert np.any(weights < 0), "Original zero-weight rows must retain negative Yeti incident mass"
        weak = np.where(np.abs(gradient) < np.float32(1e-15), 0,
                        gradient / (weights + float(np.float32(1e-15))))
        variance = np.dot(weights, weak ** 2) / weights.size
        assert variance > 0
        scale = np.float32(math.sqrt(variance) * strength * weights.size / (weights.size + 1))
        perturbation = np.array([np.float32(_normal(feature, seed=seed, iteration=0, stream=1)) * scale
                                for feature in range(3)], np.float32)
        scores = []
        for feature in range(3):
            for border in range(3):
                right = data["bins"][feature] > border
                sums = [gradient[~right].sum(), gradient[right].sum()]
                masses = [weights[~right].sum(), weights[right].sum()]
                assert min(masses) > 0
                scores.append(np.float32(_score_children(sums, masses, 2, "Cosine")) + perturbation[feature])
        expected = divmod(int(np.argmin(scores)), 3)
        assert step.depth == 1
        assert (step.split_features[0], step.split_bins[0]) == expected
        assert callback is not None
