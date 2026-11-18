"""Actual M-series coupled multiclass kernels and connected training.

Reference calculations are objective algebra and model traversal, never a CPU
training baseline. Standard CatBoost is used only to read/evaluate GPU models.
"""
import json
import ctypes as ct
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier

from catboost_metal import _multiclass


pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                                reason="Actual Apple Silicon Metal GPU required")


@pytest.fixture(autouse=True)
def no_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Multiclass Metal tests must not train CPU models")
    monkeypatch.setattr(CatBoost, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


def reference(logits, labels, weights, classes, objective, leaf_ids, leaves, l2, method):
    values = np.asarray(logits, np.float64)
    dimensions, rows = values.shape
    weights = np.asarray(weights, np.float64)
    if objective == "MultiClass":
        full = np.concatenate([values, np.zeros((1, rows))])
        maximum = full.max(axis=0)
        exponential = np.exp(full - maximum)
        probabilities = exponential / exponential.sum(axis=0)
        loss = (maximum + np.log(exponential.sum(axis=0)) - full[labels, np.arange(rows)]) * weights
    else:
        exponential = np.exp(-np.abs(values))
        probabilities = np.clip(np.where(values >= 0, 1 / (1 + exponential), exponential / (1 + exponential)),
                                1e-7, 1 - 1e-7)
        onehot = np.arange(classes)[:, None] == labels
        loss = (np.maximum(values, 0) - values * onehot + np.log1p(exponential)).mean(axis=0) * weights
    onehot = np.arange(classes)[:, None] == labels
    gradients = weights * (onehot - probabilities)
    directions = np.zeros((leaves, dimensions))
    for leaf in range(leaves):
        selected = leaf_ids == leaf
        w = weights[selected].sum()
        if w < 1e-20:
            continue
        g = gradients[:, selected].sum(axis=1)
        if method == "Gradient":
            solution = g / (w + l2)
        elif objective == "MultiClassOneVsAll":
            hessian = (weights[selected] * probabilities[:, selected] * (1 - probabilities[:, selected])).sum(axis=1)
            solution = g / (hessian + l2 + 1e-20)
        else:
            prob = probabilities[:, selected]
            hessian = np.diag((prob * weights[selected]).sum(axis=1)) - (prob * weights[selected]) @ prob.T
            if l2:
                solution = np.linalg.solve(hessian + l2 * np.eye(classes), g)
            else:
                solution = np.append(np.linalg.solve(hessian[:-1, :-1], g[:-1]), 0)
        directions[leaf] = (solution[:-1] - solution[-1]) if objective == "MultiClass" else solution
    # CUDA masks zero-weight rows, including probabilities.
    probabilities[:, weights == 0] = 0
    return gradients[:dimensions], probabilities, loss, directions


@pytest.mark.parametrize("classes", [2, 3, 8, 32, 64])
@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_gpu_derivatives_and_coupled_leaf_solver(classes, objective, method):
    rng = np.random.default_rng(473 + classes)
    rows, leaves = 259, 5
    dimensions = classes - int(objective == "MultiClass")
    logits = rng.normal(0, .7, (dimensions, rows)).astype(np.float32)
    labels = rng.integers(0, classes, rows, dtype=np.uint32)
    weights = rng.uniform(.1, 2, rows).astype(np.float32); weights[::11] = 0
    leaf_ids = rng.integers(0, leaves - 1, rows, dtype=np.uint32)
    result = _multiclass.objective_and_leaf_directions(logits, labels, classes=classes,
        objective=objective, sample_weight=weights, leaf_ids=leaf_ids, leaves=leaves,
        l2_leaf_reg=3, leaf_estimation_method=method)
    expected = reference(logits, labels, weights, classes, objective, leaf_ids, leaves, 3, method)
    for key, array in zip(("gradients", "probabilities", "weighted_losses", "directions"), expected):
        np.testing.assert_allclose(result[key], array, rtol=2e-5, atol=2e-6)
    assert not result["directions"][-1].any()
    assert result["stats"]["kernel_dispatches"] == 3


@pytest.mark.parametrize("l2", [0, 1e-10, 1e-7, .01])
def test_coupled_ridge_is_full_class_regularization(l2):
    # Solving H_top + lambda*I would be a different regularized objective.
    labels = np.array([0] * 27 + [1] * 14 + [2] * 5, np.uint32)
    logits = np.tile(np.array([[.7], [-.2]], np.float32), (1, len(labels)))
    weights = np.ones(len(labels), np.float32)
    result = _multiclass.objective_and_leaf_directions(logits, labels, classes=3, l2_leaf_reg=l2)
    expected = reference(logits, labels, weights, 3, "MultiClass", np.zeros(len(labels)), 1, l2, "Newton")
    np.testing.assert_allclose(result["directions"], expected[-1], rtol=3e-5, atol=2e-5)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_large_weighted_reduction_and_extreme_logits(objective):
    rows, classes = 100003, 3
    rng = np.random.default_rng(901)
    logits = rng.uniform(-80, 80, (classes - int(objective == "MultiClass"), rows)).astype(np.float32)
    labels = rng.integers(0, classes, rows, dtype=np.uint32)
    weights = rng.uniform(0, 3, rows).astype(np.float32)
    result = _multiclass.objective_and_leaf_directions(logits, labels, classes=classes,
                                                       objective=objective, sample_weight=weights)
    expected = reference(logits, labels, weights, classes, objective, np.zeros(rows), 1, 3, "Newton")
    np.testing.assert_allclose(result["directions"], expected[-1], rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(result["weighted_losses"], expected[2], rtol=2e-5, atol=2e-5)


def _arguments(objective="MultiClass", **overrides):
    bins = np.tile(np.array([[0, 0, 1, 1, 2, 2]], np.uint8), (1, 17))
    result = dict(bins=bins, targets=np.tile([0, 0, 1, 1, 2, 2], 17),
                  candidate_features=np.array([0, 0]), candidate_bins=np.array([0, 1]),
                  classes=3, objective=objective, iterations=8, depth=2, learning_rate=.2)
    result.update(overrides)
    return result


def _traverse(result, bins, bias=0):
    prediction = np.broadcast_to(np.asarray(bias), (bins.shape[1], result.leaf_values.shape[-1])).copy()
    prediction = prediction.astype(np.float64)
    for tree, depth in enumerate(result.depths):
        leaf = np.zeros(bins.shape[1], np.uint32)
        for level in range(int(depth)):
            feature, border = result.split_features[tree, level], result.split_bins[tree, level]
            right = bins[feature] == border if result.split_types[tree, level] else bins[feature] > border
            leaf |= right.astype(np.uint32) << level
        prediction += result.leaf_values[tree, leaf]
    return prediction


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_connected_training_learns_shared_trees(objective, method, score):
    arguments = _arguments(objective, leaf_estimation_method=method, score_function=score,
                           leaf_estimation_iterations=3)
    result = _multiclass.train(**arguments)
    assert result.loss[-1] < result.loss[0] * .75
    assert np.mean(result.predictions.argmax(axis=1) == arguments["targets"]) == 1
    assert result.leaf_values.shape == (8, 4, 3)
    np.testing.assert_allclose(result.predictions, _traverse(result, arguments["bins"]), atol=2e-6)
    np.testing.assert_allclose(result.leaf_weights.sum(axis=1), arguments["bins"].shape[1])
    if objective == "MultiClass":
        assert not result.leaf_values[:, :, -1].any()


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_iterative_leaf_values_match_coupled_objective_reference(objective):
    rng = np.random.default_rng(810)
    arguments = _arguments(objective, iterations=1, leaf_estimation_iterations=4)
    weights = rng.uniform(.1, 2, arguments["bins"].shape[1]).astype(np.float32)
    arguments["sample_weight"] = weights
    result = _multiclass.train(**arguments)
    leaf_ids = np.zeros(len(weights), np.uint32)
    for level in range(int(result.depths[0])):
        leaf_ids |= (arguments["bins"][result.split_features[0, level]] > result.split_bins[0, level]).astype(np.uint32) << level
    dimensions = 3 - int(objective == "MultiClass")
    leaf_values = np.zeros((4, dimensions))
    for _ in range(4):
        logits = leaf_values[leaf_ids].T
        values = reference(logits, arguments["targets"], weights, 3, objective, leaf_ids, 4, 3, "Newton")
        leaf_values += values[-1]
    np.testing.assert_allclose(result.leaf_values[0, :, :dimensions], leaf_values * .2, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_onehot_maximum_bin_and_session_resume(objective):
    arguments = _arguments(objective, bins=np.tile(np.array([[1, 1, 7, 7, 255, 255]], np.uint8), (1, 17)),
                           candidate_bins=np.array([1, 7]), candidate_types=np.ones(2, np.uint8))
    complete = _multiclass.train(**arguments)
    with _multiclass.Session(**dict(arguments, iterations=3)) as session:
        assert session.result().leaf_values.shape == (0, 4, 3)
        for _ in range(3):
            step = session.step()
        cursor = session.predictions()
        partial = session.result()
        np.testing.assert_array_equal(cursor, partial.predictions)
        assert step.split_types.all()
    assert session.closed
    with pytest.raises(RuntimeError, match="closed"):
        session.step()
    continued = _multiclass.train(**dict(arguments, iterations=5, initial_predictions=cursor))
    np.testing.assert_allclose(continued.predictions, complete.predictions, atol=2e-6)
    np.testing.assert_array_equal(continued.leaf_values, complete.leaf_values[3:])


def test_multiclass_initial_prediction_gauge_and_depth_zero():
    rng = np.random.default_rng(791)
    arguments = _arguments(iterations=2, depth=0)
    initial = rng.normal(size=(arguments["bins"].shape[1], 3)).astype(np.float32)
    result = _multiclass.train(**dict(arguments, initial_predictions=initial))
    np.testing.assert_array_equal(result.predictions[:, -1], initial[:, -1])
    np.testing.assert_allclose(result.predictions, _traverse(result, arguments["bins"], initial), atol=4e-7)
    assert (result.depths == 0).all()
    np.testing.assert_allclose(result.leaf_weights[:, 0], initial.shape[0])


def test_failed_newton_step_restores_last_completed_cursor():
    initial = np.tile(np.array([-20., 0., 0.], np.float32), (3, 1))
    with _multiclass.Session(np.array([[0, 1, 2]], np.uint8), [0, 1, 2], [0, 0], [0, 1],
            classes=3, iterations=2, depth=0, initial_predictions=initial,
            l2_leaf_reg=0, leaf_estimation_iterations=2) as session:
        before = session.predictions()
        loss = session.result().loss.copy()
        for _ in range(2):
            with pytest.raises(RuntimeError, match="leaf solve failed"):
                session.step()
            np.testing.assert_array_equal(session.predictions(), before)
            np.testing.assert_array_equal(session.result().loss, loss)
            assert session.completed_iterations == 0


def test_initial_large_class_gauge_preserves_small_raw_values():
    initial = np.tile(np.array([-1., 1e8], np.float32), (3, 1))
    with _multiclass.Session(np.array([[0, 1, 2]], np.uint8), [0, 1, 0], [0], [0],
                            classes=2, depth=0, iterations=1, initial_predictions=initial) as session:
        np.testing.assert_array_equal(session.predictions(), initial)
        step = session.step()
        expected = initial.copy()
        expected += step.leaf_values[0]
        np.testing.assert_array_equal(session.predictions(), expected)


def test_large_weights_do_not_overflow_finite_l2_score():
    result = _multiclass.train(np.array([[0, 1, 2]], np.uint8), [0, 1, 2], [0, 0], [0, 1],
        classes=3, iterations=1, depth=2, score_function="L2", sample_weight=np.full(3, 1e20))
    assert result.loss[-1] < result.loss[0]
    assert np.isfinite(result.leaf_values).all()


def test_actual_depth_twelve_uses_deep_partition_and_vector_histograms():
    rng = np.random.default_rng(2719)
    bins = rng.integers(0, 2, (12, 4097), dtype=np.uint8)
    result = _multiclass.train(bins, np.arange(4097) % 3, np.arange(12), np.zeros(12, np.uint32),
                              classes=3, iterations=1, depth=12, score_function="L2")
    assert result.depths[0] == 12
    assert result.leaf_weights.sum() == 4097
    np.testing.assert_allclose(result.predictions, _traverse(result, bins), atol=1e-7)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_public_classifier_string_labels_class_weights_and_gpu_inference(objective, tmp_path):
    from catboost_metal import CatBoostMetalClassifier
    rng = np.random.default_rng(984)
    x = rng.normal(size=(307, 3))
    labels = np.array(["amber", "blue", "cyan"])[x.argmax(axis=1)]
    model = CatBoostMetalClassifier(iterations=20, depth=3, learning_rate=.2, loss_function=objective,
        leaf_estimation_iterations=2, class_weights=[1., 1.5, .8]).fit(x, labels)
    raw_cpu = model.predict(x, prediction_type="RawFormulaVal", task_type="CPU")
    raw_gpu = model.predict(x, prediction_type="RawFormulaVal", task_type="GPU")
    np.testing.assert_allclose(raw_gpu, raw_cpu, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.predict_proba(x, task_type="GPU"), model.predict_proba(x), atol=2e-15)
    assert np.mean(model.predict(x, task_type="GPU").reshape(-1) == labels) > .9
    path = tmp_path / "public-multiclass.cbm"; model.save_model(str(path))
    standard = CatBoostClassifier().load_model(str(path))
    np.testing.assert_array_equal(standard.classes_, ["amber", "blue", "cyan"])
    np.testing.assert_allclose(standard.predict(x, prediction_type="RawFormulaVal"), raw_gpu, atol=1e-12)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_standard_catboost_reader_evaluates_metal_multiclass_model(objective, tmp_path):
    arguments = _arguments(objective, bias=np.array([.2, -.4, .8], np.float32))
    result = _multiclass.train(**arguments)
    model = {"features_info": {"float_features": [{"feature_index": 0, "flat_feature_index": 0,
              "borders": [.5, 1.5], "has_nans": False, "nan_value_treatment": "AsIs"}]},
             "scale_and_bias": [1., arguments["bias"].tolist()], "oblivious_trees": [],
             "model_info": {"params": json.dumps({"loss_function": {"type": objective, "params": {}}}),
                "class_params": json.dumps({"class_label_type": "Integer", "class_names": [0, 1, 2],
                                             "class_to_label": [0, 1, 2], "classes_count": 0})}}
    for index, depth in enumerate(result.depths):
        splits = [{"split_type": "FloatFeature", "float_feature_index": 0,
                   "border": float(result.split_bins[index, level]) + .5,
                   "split_index": int(result.split_bins[index, level])} for level in range(int(depth))]
        model["oblivious_trees"].append({"splits": splits,
            "leaf_values": result.leaf_values[index, :1 << depth].reshape(-1).tolist(),
            "leaf_weights": result.leaf_weights[index, :1 << depth].tolist()})
    path = tmp_path / "multiclass.json"
    path.write_text(json.dumps(model))
    loaded = CatBoostClassifier().load_model(str(path), format="json")
    raw = loaded.predict(arguments["bins"].T, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(raw, result.predictions, rtol=0, atol=2e-6)
    cbm = tmp_path / "multiclass.cbm"; loaded.save_model(str(cbm))
    restored = CatBoostClassifier().load_model(str(cbm))
    np.testing.assert_array_equal(restored.predict(arguments["bins"].T, prediction_type="RawFormulaVal"), raw)


@pytest.mark.parametrize("change", [
    {"classes": 1}, {"classes": 65}, {"depth": 17}, {"classes": 3.5}, {"iterations": False},
    {"targets": [0.5] * 102}, {"targets": [3] * 102}, {"sample_weight": [-1] * 102},
    {"sample_weight": [0] * 102}, {"bias": [0, 1]}, {"initial_predictions": np.zeros((102, 2))},
    {"candidate_bins": [0, 255]}, {"candidate_types": [0, 2]}, {"candidate_types": [0, 1]},
    {"objective": "Logloss"}, {"leaf_estimation_method": "Exact"}, {"l2_leaf_reg": -1},
    {"bootstrap_type": "MVS"}, {"random_strength": -1}, {"leaf_estimation_backtracking": "Invalid"},
    {"bootstrap_type": "Poisson", "subsample": 1}, {"bagging_temperature": -1},
])
def test_reject_invalid_inputs_before_gpu_loading(change, monkeypatch):
    def forbidden():
        raise AssertionError("Invalid input reached native loader")
    monkeypatch.setattr(_multiclass, "build_library", forbidden)
    with pytest.raises(ValueError):
        _multiclass.Session(**_arguments(**change))


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("bootstrap", ["Bayesian", "Bernoulli", "Poisson"])
def test_multiclass_bootstrap_repeat_resume_and_original_leaf_weights(objective, bootstrap):
    rng = np.random.default_rng(301)
    bins = rng.integers(0, 8, (5, 521), dtype=np.uint8)
    target = (bins[0] + 2 * bins[1]) % 3
    features = np.repeat(np.arange(5), 7); borders = np.tile(np.arange(7), 5)
    weights = rng.uniform(.3, 2, 521).astype(np.float32)
    arguments = dict(bins=bins, targets=target, candidate_features=features, candidate_bins=borders,
        classes=3, objective=objective, iterations=6, depth=3, bootstrap_type=bootstrap,
        subsample=.6, bagging_temperature=1.5, random_seed=134, random_strength=.8,
        sample_weight=weights, leaf_estimation_iterations=2)
    complete = _multiclass.train(**arguments)
    repeat = _multiclass.train(**arguments)
    np.testing.assert_array_equal(repeat.split_features, complete.split_features)
    np.testing.assert_array_equal(repeat.split_bins, complete.split_bins)
    np.testing.assert_allclose(repeat.leaf_values, complete.leaf_values, atol=1e-6)
    np.testing.assert_allclose(complete.leaf_weights.sum(axis=1), weights.sum(dtype=np.float64), rtol=1e-6)
    first = _multiclass.train(**dict(arguments, iterations=2))
    second = _multiclass.train(**dict(arguments, iterations=4, initial_predictions=first.predictions,
                                      iteration_offset=2))
    np.testing.assert_array_equal(second.split_features, complete.split_features[2:])
    np.testing.assert_allclose(second.predictions, complete.predictions, atol=2e-6)
    assert second.stats["bootstrap_state"] == {"iteration_offset": 6, "mvs_lambda": None}


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_multiclass_backtracking_halves_overshooting_newton_direction(objective, backtracking):
    initial = np.tile(np.array([-12., 0., 0.], np.float32), (12, 1))
    arguments = dict(bins=np.zeros((1, 12), np.uint8), targets=np.tile([0, 1, 2], 4),
        candidate_features=[], candidate_bins=[], classes=3, objective=objective,
        iterations=1, depth=0, learning_rate=1, l2_leaf_reg=.001,
        initial_predictions=initial, leaf_estimation_iterations=4)
    result = _multiclass.train(**dict(arguments, leaf_estimation_backtracking=backtracking))
    assert result.loss[-1] < result.loss[0]
    assert np.max(np.abs(result.leaf_values)) < 30
    np.testing.assert_allclose(result.predictions, initial + result.leaf_values[0, 0], atol=1e-6)
    # This fixture requires rejected trial steps, each adding GPU objective work.
    assert result.stats["kernel_dispatches"] > 25


@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_one_leaf_iteration_bypasses_multiclass_backtracking(backtracking):
    arguments = _arguments(iterations=2, leaf_estimation_iterations=1)
    plain = _multiclass.train(**arguments)
    result = _multiclass.train(**dict(arguments, leaf_estimation_backtracking=backtracking))
    np.testing.assert_array_equal(result.leaf_values, plain.leaf_values)
    np.testing.assert_array_equal(result.predictions, plain.predictions)


def test_multiclass_l2_ignores_random_strength_like_cuda():
    arguments = _arguments(score_function="L2")
    plain = _multiclass.train(**arguments)
    noisy = _multiclass.train(**dict(arguments, random_strength=1e6))
    np.testing.assert_array_equal(noisy.leaf_values, plain.leaf_values)


def test_armijo_dot_product_wider_than_float32_is_scaled_before_reduction():
    result = _multiclass.train(np.zeros((1, 3), np.uint8), [0, 1, 2], [], [], classes=3,
        iterations=1, depth=0, initial_predictions=np.tile([-50, 0, 0], (3, 1)),
        sample_weight=np.full(3, 1e20), l2_leaf_reg=0, learning_rate=1,
        leaf_estimation_iterations=2, leaf_estimation_backtracking="Armijo")
    assert result.loss[-1] < result.loss[0] / 5
    assert np.isfinite(result.leaf_values).all()
    assert result.stats["kernel_dispatches"] > 100


def test_multiclass_winner_reduction_compares_all_candidate_groups():
    targets = np.tile([0, 1, 2], 43)
    bins = np.vstack([np.zeros((2, len(targets)), np.uint8), targets.astype(np.uint8)])
    features = np.concatenate([np.repeat([0, 1], 255), [2, 2]])
    borders = np.concatenate([np.tile(np.arange(255), 2), [0, 1]])
    result = _multiclass.train(bins, targets, features, borders, classes=3,
                              iterations=1, depth=2, score_function="Cosine")
    assert result.split_features[0, 0] == 2
    assert np.mean(result.predictions.argmax(axis=1) == targets) == 1


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_ctr_size_penalty_prefers_smaller_table_and_restores_used_flags(objective, score):
    targets = np.tile([0, 1, 2], 57)
    column = targets.astype(np.uint8)
    bins = np.vstack([column, column])
    args = dict(bins=bins, targets=targets, candidate_features=[0, 0, 1, 1], candidate_bins=[0, 1, 0, 1],
                classes=3, objective=objective, score_function=score, depth=1, iterations=1)
    with _multiclass.Session(**args) as session:
        session.configure_feature_penalties([100, 1], model_size_reg=.5)
        step = session.step()
        assert step.split_features[0] == 1
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], [0, 1])
    with _multiclass.Session(**args) as session:
        session.configure_feature_penalties([100, 1], model_size_reg=.5, used_features=[1, 0])
        assert session.step().split_features[0] == 0
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], [1, 0])
    unpenalized = _multiclass.train(**args, ctr_unique_values=[100, 1], model_size_reg=0)
    assert unpenalized.split_features[0, 0] == 0


def test_greedy_ctr_weight_replaces_ctr_custom_factor_like_cuda():
    targets = np.tile([0, 1, 2], 17)
    bins = np.vstack([targets, targets]).astype(np.uint8)
    result = _multiclass.train(bins, targets, [0, 1], [0, 0], classes=3, iterations=1, depth=1,
        ctr_unique_values=[100, 1], model_size_reg=.5, feature_weights=[1000, 1])
    assert result.split_features[0, 0] == 1


@pytest.mark.parametrize("bad", [
    {"ctr_unique_values": [-1, 0]}, {"ctr_unique_values": [0, 2**32]},
    {"ctr_unique_values": [0.]}, {"ctr_unique_values": [0, 0], "model_size_reg": -1},
    {"ctr_unique_values": [0, 0], "used_features": [0, 2]},
    {"ctr_unique_values": [0, 0], "feature_weights": [-1, 1]},
])
def test_feature_penalty_input_validation(bad):
    with _multiclass.Session(np.zeros((2, 3), np.uint8), [0, 1, 2], [], [], classes=3, depth=0) as session:
        with pytest.raises(ValueError):
            session.configure_feature_penalties(**bad)


def test_native_failed_permutation_configuration_keeps_session_reusable():
    bins = np.zeros((1, 3), np.uint8)
    with _multiclass.Session(bins, [0, 1, 2], [], [], classes=3, depth=0,
                            sample_weight=np.full(3, 1e20)) as session:
        before = session.predictions(); loss = session.result().loss.copy()
        initial = np.array([[[-3e30, 0, 0]] * 3, [[0, 0, 0]] * 3], np.float32)
        u8, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_float)
        matrices = (u8 * 2)(_multiclass._u8(bins), _multiclass._u8(bins))
        cursors = (f32 * 2)(*[_multiclass._f32(item) for item in initial])
        error = ct.create_string_buffer(2048)
        code = session._lib.cbm_multiclass_session_set_permutations(session._handle, 2, matrices,
                                                                   cursors, None, None, error, len(error))
        assert code != 0
        assert b"Nonfinite" in error.value
        np.testing.assert_array_equal(session.predictions(), before)
        np.testing.assert_array_equal(session.result().loss, loss)
        session.configure_permutations([bins, bins])
        np.testing.assert_array_equal(session.permutation_state["predictions"], [before, before])


def test_native_failed_optimizer_restore_keeps_original_cursor():
    with _multiclass.Session(np.zeros((1, 3), np.uint8), [0, 1, 2], [], [], classes=3, depth=0,
                            sample_weight=np.full(3, 1e20)) as session:
        before = session.optimization_predictions(); loss = session.result().loss.copy()
        invalid = np.full((2, 3), -3e30, np.float32)
        error = ct.create_string_buffer(2048)
        code = session._lib.cbm_multiclass_session_restore_optimization_state(session._handle, 1,
            _multiclass._f32(invalid), error, len(error))
        assert code != 0
        np.testing.assert_array_equal(session.optimization_predictions(), before)
        np.testing.assert_array_equal(session.result().loss, loss)
