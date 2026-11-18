"""CUDA generic symmetric vector search retains repeated negative winners."""
import platform

import numpy as np
import pytest

from catboost_metal import _multiclass
from test_vector_langevin import Noise, no_cpu_fit

pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Actual Apple Silicon Metal required")


@pytest.mark.parametrize("method,langevin", [("Simple", False), ("Simple", True), ("Gradient", True), ("Newton", True)])
@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_repeated_negative_winner_keeps_depth_and_source_search_events(method, langevin, objective):
    bins = np.array([[0, 0, 0, 0, 1, 1, 1, 1]], np.uint8)
    targets = bins[0].astype(np.uint32)
    noise = Noise(zero=True)
    with _multiclass.Session(bins, targets, np.array([0], np.uint32), np.array([0], np.uint32),
            classes=2, objective=objective, iterations=1, depth=3, learning_rate=.2,
            leaf_estimation_method=method, l2_leaf_reg=1., score_function="L2") as session:
        if langevin:
            noise.install(session)
        tree = session.step()
        assert tree.depth == 3
        np.testing.assert_array_equal(tree.split_features, [0, 0, 0])
        np.testing.assert_array_equal(tree.split_bins, [0, 0, 0])
        np.testing.assert_array_equal(tree.leaf_weights, [4, 0, 0, 0, 0, 0, 0, 4])
        np.testing.assert_array_equal(tree.leaf_values[1:7], 0)
        if langevin:
            assert noise.seeds == [7, 7, 7]
            assert noise.events == ([] if method == "Simple" else
                [(1, 16), (2, 32 if method == "Newton" and objective == "MultiClass" else 16)])
        else:
            assert not noise.events and not noise.seeds


@pytest.mark.parametrize("method", ["Gradient", "Newton"])
def test_published_disabled_vector_stop_rule_remains(method):
    bins = np.array([[0, 0, 0, 0, 1, 1, 1, 1]], np.uint8)
    with _multiclass.Session(bins, bins[0].astype(np.uint32), np.array([0], np.uint32), np.array([0], np.uint32),
            classes=2, iterations=1, depth=3, learning_rate=.2,
            leaf_estimation_method=method, l2_leaf_reg=1., score_function="L2") as session:
        assert session.step().depth == 1
