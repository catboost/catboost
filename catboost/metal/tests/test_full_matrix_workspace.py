"""Full-matrix search must not allocate unused dense scalar histograms."""
import platform

import numpy as np
import pytest

from catboost_metal import _pair_matrix, _query_cross_entropy

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple GPU required")


@pytest.mark.parametrize("target", ["pair", "qce"])
@pytest.mark.parametrize("bootstrap", ["No", "Bernoulli"])
@pytest.mark.parametrize("backtracking", ["No", "AnyImprovement", "Armijo"])
def test_wide_feature_bank_fits_and_preserves_complete_forests(target, bootstrap, backtracking, monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU CatBoost training is forbidden")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)
    rows = 33
    active = (np.arange(rows) % 2).astype(np.uint8)
    weights = np.linspace(.4, 1.6, rows, dtype=np.float32)
    common = dict(candidate_features=np.array([0], np.uint32), candidate_bins=np.array([0], np.uint32),
                  iterations=2, depth=8, sample_weight=weights, learning_rate=.1,
                  leaf_estimation_iterations=3, leaf_estimation_backtracking=backtracking,
                  bootstrap_type=bootstrap, subsample=.8, random_seed=718)
    if target == "pair":
        factory = _pair_matrix.Session
        common.update(pair_winners=np.arange(1, rows, 2, dtype=np.uint32),
                      pair_losers=np.arange(0, rows - 1, 2, dtype=np.uint32),
                      pair_weights=np.linspace(.5, 1.5, rows // 2, dtype=np.float32))
    else:
        factory = _query_cross_entropy.Session
        common.update(targets=active.astype(np.float32), group_offsets=np.array([0, 16, 32, 33], np.uint32))
    outputs = []
    for features in (1, 5000):
        # Dense scalar histograms for this second bank require 2.62 GB, although
        # only one candidate is evaluated. Its full-matrix workspace is small.
        bins = np.full((features, rows), 255, np.uint8)
        bins[0] = active
        with factory(bins=bins, **common) as session:
            before = session.workspace
            assert before["histogram_bytes"] == 8
            assert before["estimated_peak_gpu_bytes"] < 16 * 1024**2
            session.configure_permutations([bins])
            assert session.workspace["estimated_peak_gpu_bytes"] == before["estimated_peak_gpu_bytes"] + 32 * 8
            before = session.workspace
            trees = [session.step(), session.step()]
            assert session.workspace == before
            assert all(tree.depth == 8 for tree in trees)
            for tree in trees:
                assert np.isfinite(tree.leaf_values).all()
                assert tree.leaf_weights.sum(dtype=float) == pytest.approx(weights.sum(dtype=float), rel=2e-6)
            outputs.append((trees, session.predictions(), before))
    # The topology/cursor footprint increases only by the stored bins and the
    # 17 bytes/feature metadata, independent of leaf count and bin capacity.
    assert outputs[1][2]["estimated_peak_gpu_bytes"] - outputs[0][2]["estimated_peak_gpu_bytes"] == 4999 * (rows + 17)
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    for narrow, wide in zip(outputs[0][0], outputs[1][0]):
        for attr in ("split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
            np.testing.assert_array_equal(getattr(narrow, attr), getattr(wide, attr))
