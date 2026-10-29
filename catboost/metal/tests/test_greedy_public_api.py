"""Public non-symmetric training, GPU prediction, and saved model interoperability."""

import json
import platform

import numpy as np
import pytest

from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires Apple Silicon Metal",
)


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("Greedy public tests must not invoke a CatBoost CPU trainer")

    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


def problem():
    rng = np.random.default_rng(7789)
    x = rng.normal(size=(257, 5)).astype(np.float32)
    y = (1.3 * x[:, 0] + (x[:, 1] > .4) * x[:, 2] - .3 * x[:, 3]).astype(np.float32)
    weights = rng.uniform(.2, 1.8, len(y)).astype(np.float32)
    weights[::17] = 0
    return x, y, weights


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_greedy_public_weighted_models_and_gpu_prediction(tmp_path, policy, classification, method):
    x, y, weights = problem()
    x[::13, 1] = np.nan
    if classification:
        y = np.where(y > 0, "positive", "negative")
    cls = CatBoostMetalClassifier if classification else CatBoostMetalRegressor
    model = cls(iterations=16, depth=4, border_count=16, grow_policy=policy,
                max_leaves=7 if policy == "Lossguide" else None,
                leaf_estimation_method=method, leaf_estimation_iterations=2,
                learning_rate=.2, nan_mode="Max", min_data_in_leaf=3)
    model.fit(x[:193], y[:193], sample_weight=weights[:193],
              eval_set=(x[193:], y[193:], weights[193:]), use_best_model=False)
    assert model.tree_count_ == 16
    assert np.max(model.tree_depths_) <= 4
    assert model.loss_history_[-1] < model.loss_history_[0]
    for start, end in ((0, 0), (3, 11)):
        actual = model.predict(x, task_type="GPU", prediction_type="RawFormulaVal",
                               ntree_start=start, ntree_end=end)
        expected = model.predict(x, prediction_type="RawFormulaVal", ntree_start=start, ntree_end=end)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    if classification:
        np.testing.assert_allclose(model.predict_proba(x, task_type="GPU"), model.predict_proba(x),
                                   rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(model.predict(x, task_type="GPU"), model.predict(x))
    if policy == "Lossguide":
        assert all(len(tree.leaf_values) <= 7 for tree in model._result.trees)
    metadata = json.loads(model.to_catboost().get_metadata()["params"])
    assert metadata["tree_learner_options"]["grow_policy"] == policy
    assert metadata["flat_params"]["task_type"] == "GPU"
    for format_ in ("cbm", "json"):
        from catboost import CatBoost
        path = tmp_path / ("model." + format_)
        model.save_model(path, format=format_)
        loaded = CatBoost().load_model(str(path), format=format_)
        np.testing.assert_allclose(loaded.predict(x, prediction_type="RawFormulaVal"),
                                   model.predict(x, prediction_type="RawFormulaVal"),
                                   rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
def test_greedy_public_snapshot_resume_is_exact(tmp_path, policy):
    x, y, weights = problem()
    options = dict(depth=4, border_count=16, grow_policy=policy, learning_rate=.2,
                   leaf_estimation_iterations=2, random_seed=31)
    expected = CatBoostMetalRegressor(iterations=9, **options).fit(
        x, y, sample_weight=weights, eval_set=(x, y), use_best_model=False)
    snapshot = tmp_path / "greedy.snapshot"
    CatBoostMetalRegressor(iterations=3, **options).fit(
        x, y, sample_weight=weights, eval_set=(x, y), use_best_model=False,
        save_snapshot=True, snapshot_file=snapshot, snapshot_interval=0)
    resumed = CatBoostMetalRegressor(iterations=9, **options).fit(
        x, y, sample_weight=weights, eval_set=(x, y), use_best_model=False,
        save_snapshot=True, snapshot_file=snapshot, snapshot_interval=0)
    np.testing.assert_array_equal(resumed.training_predictions_, expected.training_predictions_)
    np.testing.assert_array_equal(resumed.loss_history_, expected.loss_history_)
    np.testing.assert_array_equal(resumed.predict(x, task_type="GPU"), expected.predict(x, task_type="GPU"))
    assert resumed.get_evals_result() == expected.get_evals_result()


def test_greedy_lossguide_defaults_and_depth_zero():
    x, y, _ = problem()
    model = CatBoostMetalRegressor(grow_policy="Lossguide", depth=0, iterations=2).fit(x, y)
    assert model.max_leaves == 31
    assert model.score_function == "NewtonL2"
    np.testing.assert_array_equal(model.tree_depths_, [0, 0])
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), model.predict(x), atol=1e-12)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy", "Poisson", "Huber:delta=.7",
    "Expectile:alpha=.7", "Tweedie:variance_power=1.6", "LogLinQuantile:alpha=.7",
    "Quantile:alpha=.7", "MAE", "MAPE"])
def test_greedy_extended_public_objectives(policy, objective):
    x, y, weights = problem()
    cls = CatBoostMetalRegressor
    if objective == "Logloss":
        y, cls = (y > 0).astype(np.float32), CatBoostMetalClassifier
    elif objective == "CrossEntropy":
        y, cls = (1 / (1 + np.exp(-y))).astype(np.float32), CatBoostMetalClassifier
    elif objective.startswith(("Poisson", "Tweedie", "LogLinQuantile")) or objective == "MAPE":
        y = np.exp(y / 3).astype(np.float32)
    model = cls(iterations=6, depth=3, grow_policy=policy, loss_function=objective,
                score_function="Cosine", border_count=16, leaf_estimation_iterations=3,
                leaf_estimation_backtracking="AnyImprovement", learning_rate=.15).fit(x, y, sample_weight=weights)
    np.testing.assert_allclose(model.predict(x, prediction_type="RawFormulaVal", task_type="GPU"),
                               model.predict(x, prediction_type="RawFormulaVal"), rtol=2e-6, atol=2e-6)
    assert np.isfinite(model.loss_history_).all()
    assert model.loss_history_[-1] <= model.loss_history_[0]
    if objective.startswith("Quantile") or objective in ("MAE", "MAPE"):
        assert model.leaf_estimation_method == "Exact"


@pytest.mark.parametrize("options", [
    {"boosting_type": "Ordered"}, {"bootstrap_type": "MVS"},
    {"random_strength": -1}, {"leaf_estimation_backtracking": "Unknown"},
    {"loss_function": "Lq:q=2"}, {"score_function": "Unknown"},
])
def test_greedy_public_rejects_unconnected_training_modes(options):
    with pytest.raises(ValueError):
        CatBoostMetalRegressor(grow_policy="Depthwise", **options)
