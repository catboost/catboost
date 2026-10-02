"""Independent CUDA SatL2 arithmetic, pole/sign and actual Metal score checks.

score_calcers.cuh::TSatL2ScoreCalcer computes its rational adjustment from
widened statistics, rounds that adjustment to float, and rounds Score after
each child. Sample weights, not Hessians or row counts, select its w>2 branch.
No CPU model trainer is used.
"""

import math
import platform

import numpy as np
import pytest

from catboost_metal import _native
from cuda_auxiliary_score_reference import score_children
from cuda_scalar_reference import train_reference
from test_scalar_histogram_precision import build_probe, score_histograms


@pytest.fixture(scope="module")
def shader():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("SatL2 shader tests require Apple Silicon")
    return build_probe()


POLE = (3 + math.sqrt(5)) / 2
NEAREST = np.float32(POLE)
WEIGHTS = [0, 0.5, 1, 2, np.nextafter(np.float32(2), np.float32(np.inf)), 2.25, 2.5,
           np.nextafter(NEAREST, np.float32(0)), NEAREST,
           np.nextafter(NEAREST, np.float32(np.inf)), 3, 4, 10,
           2**24, np.nextafter(np.float32(2**25), np.float32(0)), 2**25, 1e30]


@pytest.mark.parametrize("weight", WEIGHTS)
def test_actual_sat_leaf_matches_cuda_widening_and_float_rounds(shader, weight):
    weight = np.float32(weight)
    gradient = np.float32(1.25)
    actual = score_histograms([[[gradient]]], [[[weight]]], [gradient], [weight],
                              [0], [0], [0], rows=1, score_function=6, library_path=shader)
    expected = score_children([gradient], [weight], "SatL2")
    np.testing.assert_array_equal(actual, [expected])
    if 2 < float(weight) < POLE:
        assert actual[0] > 0
    elif float(weight) > POLE:
        assert actual[0] < 0
    else:
        assert actual[0] == 0


@pytest.mark.parametrize("weight,gradient", [(NEAREST, 1e-20),
    (np.nextafter(NEAREST, np.float32(0)), -1e-20), (1e30, 1e30)])
def test_sat_representable_term_survives_intermediate_square_range(shader, weight, gradient):
    weight, gradient = np.float32(weight), np.float32(gradient)
    actual = score_histograms([[[gradient]]], [[[weight]]], [gradient], [weight],
                              [0], [0], [0], rows=1, score_function=6, library_path=shader)
    expected = score_children([gradient], [weight], "SatL2")
    assert np.isfinite(expected) and expected != 0
    np.testing.assert_array_equal(actual, [expected])


def test_sat_rounds_after_each_leaf_and_ignores_lambda(shader):
    # At w=3 the Sat adjustment is3, giving child contribution exactly -G².
    # Unit terms vanish when added one by one to a runningfloat of -1e8.
    gradients = np.array([10000] + [1] * 8, np.float32)
    weights = np.full(9, 3, np.float32)
    expected = score_children(gradients, weights, "SatL2")
    assert expected == -1e8 and expected != float(np.float32(-1e8 - 8))
    for l2 in [0, 3, 1e20]:
        actual = score_histograms(gradients[:, None, None], weights[:, None, None],
            gradients, weights, [0], [0], [0], rows=9, l2=l2, score_function=6, library_path=shader)
        np.testing.assert_array_equal(actual, [expected])


def _options(**changes):
    return dict(iterations=1, depth=1, learning_rate=.2, l2_leaf_reg=3, bias=0,
                score_function="SatL2", objective="RMSE", leaf_estimation_iterations=1,
                leaf_estimation_backtracking="No") | changes


@pytest.mark.parametrize("mass", [2, 2.25, np.nextafter(NEAREST, np.float32(0)), NEAREST, 3, 4])
def test_native_sat_pole_changes_selected_partition_without_gain_filter(shader, mass):
    bins = np.array([[0, 0], [0, 1]], np.uint8)
    targets, weights = np.array([-1, 1], np.float32), np.full(2, mass, np.float32)
    features, borders = np.arange(2, dtype=np.uint32), np.zeros(2, np.uint32)
    expected = train_reference(bins, targets, features, borders, **_options(sample_weight=weights))
    actual = _native.train(bins, targets, features, borders, **_options(sample_weight=weights))
    np.testing.assert_array_equal(actual.depths, expected['depths'])
    np.testing.assert_array_equal(actual.split_features, expected['split_features'])
    np.testing.assert_allclose(actual.predictions, expected['predictions'], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("objective,parameter,bias", [('RMSE',None,0),('Logloss',None,-1),('Huber',.4,0)])
def test_native_sat_weighted_multitree_reference(shader, objective, parameter, bias):
    rng = np.random.default_rng(19477)
    bins = rng.integers(0, 8, (4, 267), dtype=np.uint8)
    signal = (bins[1] > 3) - .5 * (bins[3] > 4) + rng.normal(0, .1, 267)
    targets = ((signal > .4) if objective == 'Logloss' else signal).astype(np.float32)
    weights = rng.uniform(.5, 2, 267).astype(np.float32);weights[::17] = 0
    features, borders = np.repeat(np.arange(4, dtype=np.uint32), 7), np.tile(np.arange(7,dtype=np.uint32),4)
    options = _options(iterations=3,depth=3,objective=objective,objective_param=parameter,bias=bias,sample_weight=weights)
    expected = train_reference(bins,targets,features,borders,**options)
    actual = _native.train(bins,targets,features,borders,**options)
    np.testing.assert_array_equal(actual.depths,expected['depths'])
    np.testing.assert_array_equal(actual.split_features,expected['split_features'])
    np.testing.assert_array_equal(actual.split_bins,expected['split_bins'])
    np.testing.assert_allclose(actual.predictions,expected['predictions'],rtol=1e-4,atol=3e-5)
