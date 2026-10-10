"""Region path growth can exceed the symmetric depth bound on Metal."""
import platform
import numpy as np
import pytest
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier

from catboost_metal._greedy_training import run_training
from catboost_metal._greedy_inference import predict_bins, tree_depth
from catboost_metal._greedy_model import model_json, dumps_model_json


@pytest.fixture(autouse=True)
def metal_and_no_cpu_fit(monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    def forbidden(*args, **kwargs): raise AssertionError("CPU fitting is forbidden")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier): monkeypatch.setattr(cls, "fit", forbidden)


@pytest.mark.parametrize("depth,positive_rows", [(30, 40), (100, 64), (65535, 20)])
def test_deep_region_training_gpu_prediction_reader_and_resume(tmp_path, depth, positive_rows):
    rows = 2 * positive_rows
    bins = np.zeros((positive_rows, rows), np.uint8)
    bins[np.arange(positive_rows), np.arange(positive_rows)] = 1
    targets = np.r_[np.ones(positive_rows), -np.ones(positive_rows)].astype(np.float32)
    cf, cb = np.arange(positive_rows, dtype=np.uint32), np.zeros(positive_rows, np.uint32)
    options = dict(iterations=2, depth=depth, learning_rate=.25, l2_leaf_reg=0., bias=0.,
        score_function="L2", grow_policy="Region", eval_bins=bins, eval_targets=targets,
        use_best_model=False, save_snapshot=True, snapshot_file=tmp_path / "region.npz")
    expected_depth = min(depth, positive_rows)
    result = run_training(bins, targets, cf, cb, **options, callback=lambda info: info.iteration < 1)
    assert tree_depth(result.trees[0]) == expected_depth
    assert len(result.trees[0].leaf_values) == expected_depth + 1
    np.testing.assert_array_equal(result.predictions, predict_bins(bins, result.trees))
    # Every split removes one positive observation. The remaining path keeps
    # decreasing squared-error loss, with no appeal to zero-gain splitting.
    first = result.trees[0]
    np.testing.assert_array_equal(first.leaf_values[1:], .25)
    assert first.leaf_values[0] == pytest.approx(-.25 * expected_depth / (rows - expected_depth), rel=2e-6)
    resumed = run_training(bins, targets, cf, cb, **options)
    full = run_training(bins, targets, cf, cb, **dict(options, save_snapshot=False))
    np.testing.assert_array_equal(resumed.predictions, full.predictions)
    np.testing.assert_array_equal(resumed.eval_predictions, full.eval_predictions)
    for left, right in zip(resumed.trees, full.trees):
        np.testing.assert_array_equal(left.nodes, right.nodes)
        np.testing.assert_array_equal(left.leaf_values, right.leaf_values)
    document = model_json(full, [np.array([.5], np.float32)] * positive_rows, grow_policy="Region")
    path = tmp_path / "region.json"; path.write_text(dumps_model_json(document))
    model = CatBoost().load_model(path, format="json")
    np.testing.assert_allclose(model.predict(bins.T.astype(float)), full.predictions, rtol=3e-6, atol=3e-6)
    cbm = tmp_path / "region.cbm"; model.save_model(cbm)
    np.testing.assert_allclose(CatBoost().load_model(cbm).predict(bins.T.astype(float)), full.predictions,
                               rtol=3e-6, atol=3e-6)
