"""Standard native multiclass API acceptance; all training executes on Metal."""

import os

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostError, Pool

pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
                               reason="requires the rebuilt native Metal extension")


def dataset():
    rng = np.random.default_rng(873)
    x = rng.normal(size=(150, 4)).astype(np.float32)
    labels = np.argmax(np.stack([2 * x[:, 0] - x[:, 1], x[:, 1] + x[:, 2],
                                -x[:, 0] - x[:, 2]], axis=1), axis=1)
    return x, labels, np.linspace(0.5, 2, len(x), dtype=np.float32)


def options(**extra):
    result = dict(task_type="GPU", iterations=12, depth=3, learning_rate=0.2,
                  random_seed=31, border_count=20, verbose=False, allow_writing_files=False,
                  bootstrap_type="No", random_strength=0, leaf_estimation_backtracking="No")
    result.update(extra)
    return result


class StopAfter:
    def __init__(self, count):
        self.count = count

    def after_iteration(self, info):
        return info.iteration < self.count


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_native_multiclass_labels_weights_metrics_and_export(tmp_path, objective, method):
    x, labels, weights = dataset()
    names = np.array(["apple", "blue", "green"])[labels]
    model = CatBoostClassifier(**options(loss_function=objective, leaf_estimation_method=method,
                                        class_weights=[1, 1.4, 0.8], custom_metric="Accuracy"))
    pool = Pool(x, names, weight=weights)
    model.fit(pool, eval_set=pool, use_best_model=False)
    assert model.classes_.tolist() == ["apple", "blue", "green"]
    assert model.get_metadata()["metal_backend"] == "METAL"
    accuracy = next(values for name, values in model.get_evals_result()["validation"].items()
                    if name.startswith("Accuracy"))
    assert accuracy[-1] > 0.8
    loss = model.get_evals_result()["learn"][objective]
    assert loss[-1] < loss[0]
    raw = model.predict(x, prediction_type="RawFormulaVal")
    assert raw.shape == (len(x), 3)
    np.testing.assert_allclose(model.predict(x, prediction_type="RawFormulaVal", task_type="GPU"),
                               raw, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(model.predict_proba(x, task_type="GPU"), model.predict_proba(x),
                               atol=3e-6, rtol=3e-6)
    if objective == "MultiClass":
        np.testing.assert_array_equal(raw[:, -1], np.zeros(len(x)))
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("multiclass." + format_)
        model.save_model(path, format=format_)
        restored = CatBoostClassifier().load_model(path, format=format_)
        np.testing.assert_allclose(restored.predict(x, prediction_type="RawFormulaVal"), raw,
                                   atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(restored.classes_, model.classes_)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_native_multiclass_initial_model_and_baseline(objective):
    x, labels, weights = dataset()
    first = CatBoostClassifier(**options(loss_function=objective, iterations=4)).fit(x, labels, sample_weight=weights)
    continued = CatBoostClassifier(**options(loss_function=objective, iterations=5)).fit(
        x, labels, sample_weight=weights, init_model=first)
    direct = CatBoostClassifier(**options(loss_function=objective, iterations=9)).fit(x, labels, sample_weight=weights)
    assert continued.tree_count_ == 9
    np.testing.assert_allclose(continued.predict(x, prediction_type="RawFormulaVal"),
                               direct.predict(x, prediction_type="RawFormulaVal"), atol=2e-5, rtol=2e-5)
    initial = first.predict(x, prediction_type="RawFormulaVal")
    baseline = CatBoostClassifier(**options(loss_function=objective, iterations=5)).fit(
        Pool(x, labels, weight=weights, baseline=initial))
    np.testing.assert_allclose(baseline.predict(x, prediction_type="RawFormulaVal") + initial,
                               continued.predict(x, prediction_type="RawFormulaVal"), atol=2e-5, rtol=2e-5)


def test_native_multiclass_missing_classes_and_public_baseline_columns():
    x, labels, weights = dataset()
    keep = labels != 2
    x, labels, weights = x[keep], np.where(labels[keep] == 0, 1, 3), weights[keep]
    baseline = np.zeros((len(x), 5), dtype=np.float32)
    baseline[:, 1] = 0.25
    baseline[:, 3] = -0.5
    model = CatBoostClassifier(**options(loss_function="MultiClass", classes_count=5))
    model.fit(Pool(x, labels, weight=weights, baseline=baseline),
              eval_set=Pool(x, labels, weight=weights, baseline=baseline), use_best_model=False)
    raw = model.predict(x, prediction_type="RawFormulaVal")
    assert raw.shape == (len(x), 5)
    assert np.isneginf(raw[:, [0, 2, 4]]).all()
    assert np.isfinite(raw[:, [1, 3]]).all()
    np.testing.assert_allclose(model.predict(x, prediction_type="RawFormulaVal", task_type="GPU"),
                               raw, atol=3e-6, rtol=3e-6)
    assert model.get_leaf_values().size == model.get_tree_leaf_counts().sum() * 2
    continued = CatBoostClassifier(**options(loss_function="MultiClass", classes_count=5, iterations=2))
    continued.fit(x, labels, init_model=model)
    assert np.isfinite(continued.predict(x, prediction_type="RawFormulaVal")[:, [1, 3]]).all()


def test_native_multiclass_initial_model_rejects_changed_label_order():
    x, labels, _ = dataset()
    names = np.array(["apple", "blue", "green"])[labels]
    first = CatBoostClassifier(**options(loss_function="MultiClass", iterations=2,
                                        class_names=["apple", "blue", "green"])).fit(x, names)
    with pytest.raises(CatBoostError, match="class labels|class order"):
        CatBoostClassifier(**options(loss_function="MultiClass", iterations=2,
                                    class_names=["green", "blue", "apple"])).fit(x, names, init_model=first)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson"])
def test_native_multiclass_snapshot_restores_cursor_and_sampling(tmp_path, objective, bootstrap):
    x, labels, weights = dataset()
    extra = dict(loss_function=objective, iterations=7, bootstrap_type=bootstrap,
                 random_strength=0.8, leaf_estimation_iterations=3, leaf_estimation_backtracking="AnyImprovement")
    if bootstrap in ("Bernoulli", "Poisson"):
        extra["subsample"] = 0.8
    direct = CatBoostClassifier(**options(**extra)).fit(x, labels, sample_weight=weights)
    saved = options(**extra, allow_writing_files=True, train_dir=str(tmp_path),
                    save_snapshot=True, snapshot_file="mc.snapshot", snapshot_interval=0)
    partial = CatBoostClassifier(**saved).fit(x, labels, sample_weight=weights, callbacks=[StopAfter(3)])
    assert partial.tree_count_ == 3
    restored = CatBoostClassifier(**saved).fit(x, labels, sample_weight=weights)
    np.testing.assert_allclose(restored.get_leaf_values(), direct.get_leaf_values(), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(restored.predict(x, prediction_type="RawFormulaVal"),
                               direct.predict(x, prediction_type="RawFormulaVal"), atol=2e-6, rtol=2e-6)
    again = CatBoostClassifier(**saved).fit(x, labels, sample_weight=weights)
    np.testing.assert_array_equal(again.predict_proba(x), restored.predict_proba(x))


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_native_multiclass_snapshot_keeps_exact_optimizer_cursor_with_baseline(tmp_path, objective):
    x, labels, weights = dataset()
    baseline = np.random.default_rng(1581).normal(size=(len(x), 3)).astype(np.float32)
    pool = Pool(x, labels, weight=weights, baseline=baseline)
    extra = dict(loss_function=objective, iterations=8, bootstrap_type="Bernoulli", subsample=0.8,
                 random_strength=0.7, leaf_estimation_iterations=3, leaf_estimation_backtracking="AnyImprovement")
    direct = CatBoostClassifier(**options(**extra)).fit(pool)
    saved = options(**extra, allow_writing_files=True, train_dir=str(tmp_path),
                    save_snapshot=True, snapshot_file="gauge.snapshot", snapshot_interval=0)
    CatBoostClassifier(**saved).fit(pool, callbacks=[StopAfter(3)])
    restored = CatBoostClassifier(**saved).fit(pool)
    np.testing.assert_array_equal(restored.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(restored.predict(x, prediction_type="RawFormulaVal"),
                                  direct.predict(x, prediction_type="RawFormulaVal"))
    np.testing.assert_array_equal(restored.get_evals_result()["learn"][objective],
                                  direct.get_evals_result()["learn"][objective])


def test_native_multiclass_multiple_eval_sets_and_early_stop():
    x, labels, _ = dataset()
    model = CatBoostClassifier(**options(loss_function="MultiClass", iterations=30, early_stopping_rounds=3))
    model.fit(x, labels, eval_set=[Pool(x[:10]), Pool(x, (labels + 1) % 3)])
    assert 0 < model.tree_count_ < 30
    assert len(model.get_test_evals()) == 2
    np.testing.assert_allclose(np.asarray(model.get_test_evals()[1]).T,
                               model.predict(x, prediction_type="RawFormulaVal"), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_native_multiclass_selects_best_feature_across_candidate_groups(objective, score):
    rng = np.random.default_rng(13905)
    x = rng.normal(size=(400, 20)).astype(np.float32)
    labels = np.digitize(x[:, -1], [-0.4, 0.5])
    # 400 candidates span more than one 256-candidate reduction group. Only
    # the final feature determines labels; its winner must beat the first group.
    model = CatBoostClassifier(**options(loss_function=objective, score_function=score,
                                        iterations=1, depth=1, border_count=20)).fit(x, labels)
    assert np.argmax(model.feature_importances_) == x.shape[1] - 1
    assert model.get_tree_leaf_counts().tolist() == [2]
