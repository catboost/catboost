"""Standalone vector greedy training, full optimizer recovery and standard models."""
import json
from dataclasses import replace

import numpy as np
import pytest
from catboost import CatBoost
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor, _multiclass
from catboost_metal._data import cuda_search_permutation
from catboost_metal._greedy_training import run_training
from catboost_metal._greedy_inference import predict_bins
from catboost_metal._greedy_model import model_json
from catboost_metal._training import _fingerprint, _json
from test_greedy_vector_training import problem as private_problem, OBJECTIVES, SCORES, POLICIES, route
from test_native_greedy_vector import inputs


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU CatBoost fitting is forbidden")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def problem(objective="MultiClass", policy="Depthwise", category="numeric", **extra):
    _, x, y, w, cats, config, _, _ = inputs(objective, policy, category)
    for key in ("task_type", "has_time", "simple_ctr", "verbose", "allow_writing_files"):
        config.pop(key)
    config.update(cat_features=cats, **extra)
    cls = CatBoostMetalRegressor if objective == "RMSEWithUncertainty" else CatBoostMetalClassifier
    return cls, x, y, w, config


def resident(model, y, weights, config):
    layout = model._layout
    cf, cb, ct = [], [], []
    for feature, borders in enumerate(layout.borders):
        choices = layout.categorical[feature].candidate_bins if feature in layout.categorical else range(len(borders))
        cf.extend([feature] * len(choices)); cb.extend(choices)
        ct.extend([int(feature in layout.categorical)] * len(choices))
    options = {key: config[key] for key in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "grow_policy",
        "leaf_estimation_method", "leaf_estimation_iterations", "leaf_estimation_backtracking", "score_function",
        "random_seed", "random_strength", "bootstrap_type")}
    for key in ("subsample", "bagging_temperature", "max_leaves"):
        if key in config: options[key] = config[key]
    with _multiclass.Session(layout.permutation_bins[0], y, cf, cb, candidate_types=ct,
            classes=model.n_outputs_, objective=config["loss_function"], sample_weight=weights,
            bias=model.bias_, **options) as session:
        if layout.permutation_count > 1: session.configure_permutations(layout.permutation_bins)
        for iteration, tree in enumerate(model._result.trees):
            if layout.permutation_count > 1:
                session.select_permutation(cuda_search_permutation(config["random_seed"], iteration, layout.permutation_count))
            expected = session.step()
            for key in ("nodes", "leaf_values", "leaf_weights"):
                np.testing.assert_array_equal(getattr(tree, key), getattr(expected, key))
        np.testing.assert_array_equal(model.training_predictions_, session.predictions())
        return session.permutation_state


def readers(model, x, path):
    expected = model.predict(x, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(model.predict(x, prediction_type="RawFormulaVal", task_type="GPU"), expected,
                               rtol=6e-6, atol=1e-6)
    for fmt in ("cbm", "json"):
        output = path / ("vector-public." + fmt)
        model.save_model(output, format=fmt)
        restored = CatBoost().load_model(output, format=fmt)
        np.testing.assert_allclose(restored.predict(x, prediction_type="RawFormulaVal"), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(restored.predict(x, prediction_type="RawFormulaVal", task_type="GPU"), expected,
                                   rtol=6e-6, atol=1e-6)
    for start, end in ((0, 1), (1, model.tree_count_), (model.tree_count_, model.tree_count_)):
        actual = model.predict(x, prediction_type="RawFormulaVal", task_type="GPU", ntree_start=start, ntree_end=end)
        raw = np.zeros_like(actual)
        if start == 0: raw[:] = model.bias_
        bins = model._layout.transform(x)
        for tree in model._result.trees[start:end]: raw += tree.leaf_values[route(tree, bins)]
        np.testing.assert_array_equal(actual, raw)
    predictions = ("Class", "Probability", "LogProbability", "Exponent") if model._classifier else ("RMSEWithUncertainty", "Exponent")
    for prediction in predictions:
        a = model.predict(x, prediction_type=prediction, task_type="GPU")
        b = model.predict(x, prediction_type=prediction)
        if prediction == "Class": np.testing.assert_array_equal(a, b)
        else: np.testing.assert_allclose(a, b, rtol=6e-6, atol=1e-6)
    empty = model.predict(x[:0], prediction_type="RawFormulaVal", task_type="GPU")
    assert empty.shape == (0, model.n_outputs_)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_public_numeric_forests_match_vector_session_and_readers(tmp_path, objective, policy, score, method):
    cls, x, y, w, config = problem(objective, policy, score_function=score, leaf_estimation_method=method)
    model = cls(**config).fit(x, y, sample_weight=w)
    resident(model, y, w, config)
    readers(model, x, tmp_path)
    assert model.to_catboost().get_all_params()["grow_policy"] == policy


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("category", ["numeric", "onehot", "ctr"])
@pytest.mark.parametrize("sampler", ["No", "Bayesian", "Bernoulli", "Poisson"])
def test_all_samplers_labels_weights_and_categories_resume_bitwise(tmp_path, objective, policy, category, sampler):
    extra = dict(bootstrap_type=sampler, random_strength=.35, score_function="Cosine")
    if sampler in ("Bernoulli", "Poisson"): extra["subsample"] = .71
    cls, x, y, w, config = problem(objective, policy, category, **extra)
    labels, effective = y, w
    if objective != "RMSEWithUncertainty":
        labels = np.array(["amber", "blue", "cyan"])[y]
        config["class_weights"] = {"amber": .4, "blue": 1.7, "cyan": 2.1}
        effective = w * np.array([.4, 1.7, 2.1], np.float32)[y]
    evaluation = x.copy()
    if category != "numeric": evaluation[::13, 2] = "unseen"
    fit = dict(sample_weight=w, eval_set=(evaluation, labels, w), use_best_model=False)
    direct = cls(**config).fit(x, labels, **fit)
    expected = resident(direct, y, effective, config)
    snapshot = tmp_path / "state.npz"
    saved = dict(save_snapshot=True, snapshot_file=snapshot, snapshot_interval=0)
    partial = cls(**config).fit(x, labels, **fit, **saved, callback=lambda i: i.iteration < 2)
    assert partial.tree_count_ == 2
    resumed = cls(**config).fit(x, labels, **fit, **saved)
    for key in ("training_predictions_", "loss_history_", "tree_depths_"):
        np.testing.assert_array_equal(getattr(resumed, key), getattr(direct, key))
    assert resumed.get_evals_result() == direct.get_evals_result()
    for a, b in zip(resumed._result.trees, direct._result.trees):
        for key in ("nodes", "leaf_values", "leaf_weights"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    with np.load(snapshot, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["permutation_predictions"], expected["predictions"])
        np.testing.assert_array_equal(archive["optimization_predictions"], expected["optimization_predictions"])
        assert archive["leaf_values"].shape[1] == direct.n_outputs_
    readers(resumed, evaluation, tmp_path)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("backtracking", ["No", "AnyImprovement", "Armijo"])
def test_best_model_retains_full_state_and_extends_exactly(tmp_path, objective, policy, backtracking):
    cls, x, y, w, config = problem(objective, policy, "ctr", iterations=7,
        leaf_estimation_backtracking=backtracking, bootstrap_type="Bernoulli", subsample=.71,
        eval_metric=None if objective == "RMSEWithUncertainty" else "Accuracy")
    selected = config["eval_metric"] or objective
    fit = dict(sample_weight=w, eval_set=(x, np.roll(y, 67), w), use_best_model=True)
    saved = dict(save_snapshot=True, snapshot_file=tmp_path / "best.npz", snapshot_interval=0)
    model = cls(**config).fit(x, y, **fit, **saved)
    history = model.get_evals_result()["validation"][selected]
    best = int(np.argmin(history) if objective == "RMSEWithUncertainty" else np.argmax(history))
    assert model.tree_count_ == best + 1
    np.testing.assert_array_equal(model.training_predictions_, predict_bins(
        model._layout.permutation_bins[-1], model._result.trees, model.bias_))
    with np.load(saved["snapshot_file"], allow_pickle=False) as archive:
        assert len(archive["loss"]) == 8 and archive["optimization_predictions"].shape[0] == 4
    extended = config | dict(iterations=9)
    resumed = cls(**extended).fit(x, y, **fit, **saved)
    direct = cls(**extended).fit(x, y, **fit)
    np.testing.assert_array_equal(resumed.training_predictions_, direct.training_predictions_)
    assert resumed.get_evals_result() == direct.get_evals_result()
    readers(resumed, x, tmp_path)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("count", [1, 2, 7, 64])
def test_public_ctr_all_dataset_counts_preserve_optimization(tmp_path, objective, policy, count):
    cls, x, y, w, config = problem(objective, policy, "ctr", iterations=2, permutation_count=count)
    fit = dict(sample_weight=w, save_snapshot=True, snapshot_file=tmp_path / "banks.npz")
    model = cls(**config).fit(x, y, **fit)
    expected = resident(model, y, w, config)
    assert model._layout.permutation_count == count
    with np.load(fit["snapshot_file"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["optimization_predictions"], expected["optimization_predictions"])
    same = cls(**(config | dict(model_size_reg=8))).fit(x, y, sample_weight=w)
    np.testing.assert_array_equal(model.training_predictions_, same.training_predictions_)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("ctr", ["Borders", "FeatureFreq"])
def test_unseen_categories_and_onall_cardinalities(tmp_path, objective, policy, ctr):
    cls, x, y, w, config = problem(objective, policy, "onehot", one_hot_max_size=8, ctr_type=ctr)
    evaluation = x.copy(); evaluation[::11, 2] = "only-in-validation"
    model = cls(**config).fit(x, y, sample_weight=w, eval_set=(evaluation, y, w), use_best_model=False)
    assert model._layout.ctrs and len(model._layout.categorical[2].candidate_bins) == 0
    resident(model, y, w, config); readers(model, evaluation, tmp_path)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("count", [1, 4])
def test_baseline_and_optimizer_coordinates_survive_resume_and_trim(tmp_path, objective, policy, count):
    options, banks, _ = private_problem(objective, policy, count=count, iterations=6, bias=np.zeros(2 if objective == "RMSEWithUncertainty" else 3),
                                      leaf_estimation_backtracking="Armijo", bootstrap_type="Bernoulli", random_strength=.4)
    rows = options["bins"].shape[1]
    evaluation = dict(eval_bins=options["bins"], eval_targets=np.roll(options["targets"], 57),
                      eval_weight=options["sample_weight"], use_best_model=True)
    full = run_training(**options, **evaluation, permutation_bins=banks)
    saved = dict(save_snapshot=True, snapshot_file=tmp_path / "baseline.npz", snapshot_interval=0)
    run_training(**options, **evaluation, permutation_bins=banks, **saved, callback=lambda i: i.iteration < 2)
    resumed = run_training(**options, **evaluation, permutation_bins=banks, **saved)
    np.testing.assert_array_equal(resumed.predictions, full.predictions)
    np.testing.assert_array_equal(resumed.eval_predictions, full.eval_predictions)
    np.testing.assert_array_equal(resumed.loss, full.loss)
    assert resumed.evals_result == full.evals_result
    expected = options["initial_predictions"].copy()
    for tree in full.trees: expected += tree.leaf_values[route(tree, banks[-1])]
    # Training publishes a fused raw-value * rate + cursor update. Model
    # evaluation adds already-rounded float32 leaf values. Recovery above is
    # exact; these two representations can differ by one rounding per tree.
    np.testing.assert_allclose(full.predictions, expected, rtol=3e-6, atol=3e-7)
    with np.load(saved["snapshot_file"], allow_pickle=False) as archive:
        assert archive["optimization_predictions"].shape == (count, options["classes"] - (objective == "MultiClass"), rows)
    assert full.stats["validation"]["dataset_uploads"] == 1
    assert full.stats["validation"]["bins_upload_bytes"] == options["bins"].size


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("corruption", ["missing", "shape", "nan", "last_cursor", "leaf_shape", "eval_shape", "graph", "checksum"])
def test_vector_snapshot_corruption_fails_before_gpu(tmp_path, monkeypatch, objective, corruption):
    options, banks, _ = private_problem(objective, count=4, bias=0.)
    saved = dict(save_snapshot=True, snapshot_file=tmp_path / "corrupt.npz", snapshot_interval=0)
    fit = dict(permutation_bins=banks, eval_bins=options["bins"], eval_targets=options["targets"], use_best_model=False)
    run_training(**options, **fit, **saved, callback=lambda i: i.iteration < 2)
    with np.load(saved["snapshot_file"], allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(arrays.pop("metadata").item()); header.pop("checksum")
    if corruption == "missing": arrays.pop("optimization_predictions")
    if corruption == "shape": arrays["optimization_predictions"] = arrays["optimization_predictions"][:, :, :-1]
    if corruption == "nan": arrays["optimization_predictions"][0, 0, 0] = np.nan
    if corruption == "last_cursor": arrays["permutation_predictions"][-1, 0, 0] += 1
    if corruption == "leaf_shape": arrays["leaf_values"] = arrays["leaf_values"][:, :-1]
    if corruption == "eval_shape": arrays["eval_predictions"] = arrays["eval_predictions"][:, :-1]
    if corruption == "graph": arrays["nodes"][0, 3] = 0
    header["checksum"] = _fingerprint(arrays, header)
    if corruption == "checksum": arrays["optimization_predictions"][0, 0, 0] += 1
    np.savez(saved["snapshot_file"], **arrays, metadata=np.asarray(_json(header)))
    monkeypatch.setattr(_multiclass, "_load", lambda *args: (_ for _ in ()).throw(AssertionError("Invalid snapshot reached GPU")))
    with pytest.raises(ValueError, match="[Ss]napshot|[Tt]ree"):
        run_training(**options, **fit, **saved)


@pytest.mark.parametrize("policy", POLICIES)
def test_automatic_multiclass_default_and_binary_refit_follow_cuda(policy):
    _, x, y, w, config = problem(policy=policy)
    config.pop("loss_function"); config.pop("score_function")
    estimator = CatBoostMetalClassifier(**config).fit(x, y, sample_weight=w)
    assert estimator._objective == "MultiClass"
    assert estimator.score_function == ("L2" if policy == "Lossguide" else "Cosine")
    binary = estimator.fit(x, y == 1, sample_weight=w)
    assert binary._objective == "Logloss"
    assert binary.score_function == ("NewtonL2" if policy == "Lossguide" else "Cosine")


@pytest.mark.parametrize("classes", [2, 7, 64])
@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_root_only_large_output_counts(tmp_path, classes, objective):
    options, _, _ = private_problem(objective, classes=classes, iterations=3, depth=0, bias=0.)
    options["candidate_features"] = options["candidate_bins"] = np.empty(0, np.uint32)
    options["candidate_types"] = None
    fit = dict(eval_bins=options["bins"], eval_targets=options["targets"], use_best_model=False,
               save_snapshot=True, snapshot_file=tmp_path / "root.npz")
    a = run_training(**options, **fit, callback=lambda i: i.iteration < 1)
    b = run_training(**options, **fit)
    assert a.predictions.shape == b.predictions.shape == (263, classes)
    assert all(len(t.leaf_values) == 1 for t in b.trees)
    direct = _multiclass.train(**options)
    np.testing.assert_array_equal(b.predictions, direct.predictions)
    document = model_json(b, [np.arange(7)] * 4, objective=objective, bias=np.zeros(classes))
    assert len(document["scale_and_bias"][1]) == classes
    assert len(document["trees"][0]["value"]) == classes


def test_uncertainty_negative_likelihood_snapshot_remains_valid(tmp_path):
    bins = np.zeros((1, 17), np.uint8)
    options = dict(bins=bins, targets=np.zeros(17, np.float32), candidate_features=np.empty(0, np.uint32), candidate_bins=np.empty(0, np.uint32),
                   objective="RMSEWithUncertainty", iterations=3, depth=0, learning_rate=.1, l2_leaf_reg=3.,
                   bias=np.array([0., -4.], np.float32), score_function="L2")
    path = tmp_path / "negative.npz"
    run_training(**options, save_snapshot=True, snapshot_file=path, callback=lambda i: i.iteration < 1)
    result = run_training(**options, save_snapshot=True, snapshot_file=path)
    assert (result.loss < 0).all()
    np.testing.assert_array_equal(result.predictions, run_training(**options).predictions)


@pytest.mark.parametrize("corruption", ["dimensions", "scalar", "bias", "nonfinite", "weights"])
def test_vector_model_shape_checks(corruption):
    options, _, _ = private_problem(iterations=2)
    result = _multiclass.train(**options)
    bias = np.zeros(3)
    if corruption == "dimensions": result.trees = (result.trees[0], replace(result.trees[1], leaf_values=result.trees[1].leaf_values[:, :2]))
    if corruption == "scalar": result.trees = (replace(result.trees[0], leaf_values=result.trees[0].leaf_values[:, 0]),)
    if corruption == "bias": bias = np.zeros(2)
    if corruption == "nonfinite": result.trees[0].leaf_values[0, 0] = np.nan
    if corruption == "weights": result.trees[0].leaf_weights[0] = -1
    with pytest.raises(ValueError): model_json(result, [np.arange(7)] * 4, objective="MultiClass", bias=bias)
