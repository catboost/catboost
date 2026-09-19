"""Reject malformed NumPy buffers before the C ABI can read beyond them."""

import numpy as np
import pytest

from catboost_metal import _native


@pytest.mark.parametrize("replacement", [
    {"targets": np.array([1], np.float32)},
    {"targets": np.array([[1, 2]], np.float32)},
    {"targets": np.array([1, np.inf], np.float32)},
    {"candidate_bins": np.array([], np.uint32)},
    {"candidate_features": np.array([[0]], np.uint32)},
    {"candidate_features": np.array([1], np.uint32)},
    {"candidate_features": np.array([-1])},
    {"candidate_bins": np.array([0.5])},
    {"candidate_bins": np.array([255], np.uint32)},
    {"bins": np.array([[0, -1]])},
    {"bins": np.array([[0, 256]])},
    {"bins": np.empty((1, 0), np.uint8)},
    {"iterations": -1}, {"iterations": 2**32 + 1}, {"depth": 17},
    {"learning_rate": 2}, {"bias": float("inf")}, {"l2_leaf_reg": -1},
])
def test_malformed_buffers_cannot_reach_ctypes(monkeypatch, replacement):
    monkeypatch.setattr(_native, "build_library", lambda: pytest.fail("Reached native runtime"))
    args = dict(bins=np.array([[0, 1]], np.uint8), targets=np.array([1, 2], np.float32),
                candidate_features=np.array([0], np.uint32), candidate_bins=np.array([0], np.uint32),
                iterations=1, depth=1, learning_rate=0.1, l2_leaf_reg=3, bias=0,
                score_function="Cosine")
    args.update(replacement)
    with pytest.raises(ValueError):
        _native.train(**args)
