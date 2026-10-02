"""Public native-option acceptance, with GPU fits and standard model readers.

These cases exercise the bridge selected by the newly forwarded options. Native
equivalence checks preserve input-row cursors; categorical export additionally
uses independent frequency and last-one-hot-bin oracles. CPU is prediction only.
"""
from collections import Counter
import json
import os
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRanker, CatBoostMetalRegressor
from catboost_metal._categorical import cat_feature_hashes
from catboost_metal._feature_parallel_frontend import native_parameters
from test_greedy_sampling import draws as bootstrap_draws
from test_native_compound_ctrs import check_final_tables, independent_prediction
from test_native_full_counters import pools as frequency_pools, problem as frequency_problem
from test_native_query_simple import check_leaf_equations as check_query_simple, problem as query_simple_problem
from test_native_rsm import problem as rsm_problem, reference_masks
from test_native_scalar_simple import check_scalar_equations, numeric_leaves
from test_native_vector_simple import gradients as vector_gradients, leaf_equations as vector_simple_leaves


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1"
    or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires the rebuilt native Metal option bridge")


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU", "Acceptance must not fit CPU models"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(**extra):
    return dict(iterations=3, depth=2, learning_rate=.17, l2_leaf_reg=2,
                border_count=12, permutation_count=1, min_fold_size=4,
                fold_len_multiplier=1.7, fold_permutation_block=3, score_function="Cosine",
                random_seed=619, random_strength=0, boost_from_average=False,
                bootstrap_type="No", leaf_estimation_method="Newton",
                leaf_estimation_iterations=2, leaf_estimation_backtracking="No") | extra


def numeric_problem():
    x = np.random.default_rng(591).normal(size=(97, 4)).astype(np.float32)
    y = (1.7 * x[:, 0] - .8 * x[:, 1] + .3 * x[:, 2] ** 2).astype(np.float32)
    weights = (.4 + np.arange(len(x)) % 9 / 5).astype(np.float32)
    return x, y, weights


def assert_equivalent(model, native, learn, tmp_path, *, evaluation=None,
                      probes=(), numeric_cursor=True):
    assert model._native_bridge_fitted
    assert model.training_stats_["backend"] == "METAL"
    assert native.get_metadata()["metal_backend"] == "METAL"
    assert model.training_stats_["device"] and model.training_stats_["kernel_dispatches"] > 0
    assert model.training_stats_["gpu_seconds"] > 0
    assert model.training_stats_["data_partition"] == model.data_partition
    assert model.training_stats_["grow_policy"] == model.grow_policy
    for method in ("get_leaf_values", "get_leaf_weights", "get_tree_leaf_counts"):
        np.testing.assert_array_equal(getattr(model.to_catboost(), method)(), getattr(native, method)())
    np.testing.assert_array_equal(model.training_predictions_,
                                  np.asarray(native._object._get_metal_training_cursor(), np.float32))
    assert model.get_evals_result() == native.get_evals_result()
    assert np.isfinite(model.loss_history_).all()
    assert len(model.loss_history_) == model.training_stats_["completed_iterations"] + 1
    if numeric_cursor:
        # Every row has its original position, including P=4/Ordered fixtures.
        np.testing.assert_allclose(model.training_predictions_,
            native.predict(learn, prediction_type="RawFormulaVal", task_type="GPU"),
            rtol=8e-6, atol=3e-6)
    if evaluation is not None:
        np.testing.assert_array_equal(model.to_catboost().get_test_eval(), native.get_test_eval())
    inputs = [learn, *([evaluation] if evaluation is not None else []), *probes]
    expected = [native.predict(pool, prediction_type="RawFormulaVal", task_type="GPU") for pool in inputs]
    for pool, prediction in zip(inputs, expected):
        for task in ("CPU", "GPU"):
            np.testing.assert_allclose(model.predict(pool, prediction_type="RawFormulaVal", task_type=task),
                                       prediction, rtol=8e-6, atol=3e-6)
    for fmt in ("cbm", "json"):
        path = tmp_path / ("public-options." + fmt)
        model.save_model(path, format=fmt)
        restored = CatBoost().load_model(path, format=fmt)
        for pool, prediction in zip(inputs, expected):
            for task in ("CPU", "GPU"):
                np.testing.assert_allclose(restored.predict(pool, prediction_type="RawFormulaVal", task_type=task),
                                           prediction, rtol=8e-6, atol=3e-6)
    return json.loads((tmp_path / "public-options.json").read_text())


def fit_and_compare(model, learn, tmp_path, *, evaluation=None, probes=(), numeric_cursor=True):
    assert model._native_adapter
    model.fit(learn, eval_set=evaluation, use_best_model=False)
    # Generic CatBoost accepts native-only settings absent from subclass
    # constructor signatures. Fit may infer a multilabel objective first.
    native = CatBoost().set_params(**native_parameters(model))
    native.fit(learn, eval_set=evaluation, use_best_model=False)
    document = assert_equivalent(model, native, learn, tmp_path, evaluation=evaluation,
                                 probes=probes, numeric_cursor=numeric_cursor)
    return native, document


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
def test_scalar_fixed_binary_root_uses_native_greedy_policy(policy, tmp_path):
    row = np.arange(96)
    x = np.column_stack([row % 2, (row // 2) % 2]).astype(np.float32)
    learn = Pool(x, 20 * x[:, 1] + .01 * x[:, 0])
    model = CatBoostMetalRegressor(**options(grow_policy=policy, fixed_binary_splits=[0]))
    _, document = fit_and_compare(model, learn, tmp_path)
    assert model.data_partition == "DocParallel"
    assert len(document["trees"]) == model.iterations
    for tree in document["trees"]:
        assert tree["split"]["split_type"] == "FloatFeature"
        assert tree["split"]["float_feature_index"] == 0
        assert tree["split"]["border"] == .5


def test_pairlogit_fixed_binary_root_uses_native_greedy_ranker(tmp_path):
    row = np.arange(96)
    x = np.column_stack([row % 2, (row // 2) % 2]).astype(np.float32)
    pairs = [(first + 2, first) for first in range(0, len(row), 4)]
    learn = Pool(x, x[:, 1], group_id=row // 8, pairs=pairs)
    model = CatBoostMetalRanker(**options(loss_function="PairLogit", grow_policy="Depthwise",
                                         fixed_binary_splits=[0]))
    _, document = fit_and_compare(model, learn, tmp_path)
    np.testing.assert_array_equal(model.pairs_, pairs)
    assert all(tree["split"]["float_feature_index"] == 0 for tree in document["trees"])


@pytest.mark.parametrize("objective", ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise"))
def test_full_matrix_rsm_matches_native_and_independent_feature_masks(objective, tmp_path):
    learn, folds = rsm_problem(tmp_path)
    if objective != "PairLogitPairwise":
        learn.set_pairs([])
    learn.set_weight(np.linspace(.5, 1.5, learn.num_row()))
    model = CatBoostMetalRanker(**options(loss_function=objective, rsm=.5, random_seed=761,
        score_function="NewtonL2", leaf_estimation_iterations=1, one_hot_max_size=2,
        ctr_border_count=3, model_size_reg=0))
    native, document = fit_and_compare(model, learn, tmp_path, numeric_cursor=False)
    masks, draws = reference_masks(folds, loss=objective, rsm=.5, trees=model.iterations)
    assert len({tuple(sorted(mask)) for mask in masks}) > 1
    assert any(len(mask) < len(folds) for mask in masks)
    for tree, mask in zip(document["oblivious_trees"], masks):
        for split in tree["splits"]:
            feature = split["float_feature_index"] if split["split_type"] == "FloatFeature" else len(folds)
            assert feature in mask
    assert native.get_metadata()["metal_rsm_rng"] == "cuda_single_device_host_shadow_v1"
    assert int(native.get_metadata()["metal_rsm_host_draw_count"]) == draws


PROFILES = (
    pytest.param(dict(fold_size_loss_normalization=True), id="normalize"),
    pytest.param(dict(add_ridge_penalty_to_loss_function=True), id="ridge"),
    pytest.param(dict(meta_l2_exponent=2, meta_l2_frequency=1), id="meta"),
    pytest.param(dict(langevin=True, diffusion_temperature=.1), id="langevin"),
)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("path", ("PlainDP", "PlainFP", "OrderedFP", "Depthwise"))
def test_regularization_profiles_preserve_native_partition_and_cursor(profile, path, tmp_path):
    config = options(**profile)
    if path in ("PlainFP", "OrderedFP"):
        config.update(data_partition="FeatureParallel", permutation_count=4)
    if path == "OrderedFP":
        # Normalization alone is already supported by the private Ordered path.
        config.update(boosting_type="Ordered", max_ctr_complexity=2)
    if path == "Depthwise":
        config["grow_policy"] = "Depthwise"
    x, y, weights = numeric_problem()
    model = CatBoostMetalRegressor(**config)
    native, _ = fit_and_compare(model, Pool(x, y, weight=weights), tmp_path)
    params = json.loads(native.get_metadata()["params"])["flat_params"]
    for key, value in profile.items():
        assert params[key] == pytest.approx(value)


def test_one_hot_256_can_select_last_known_bin_and_preserve_unseen_branch(tmp_path):
    categories = np.array([f"category-{i}" for i in range(256)], object)
    hashes = cat_feature_hashes(categories)
    assert len(np.unique(hashes)) == 256
    # Native dense bins sort unsigned hashes: this is runtime bin 255.
    special = int(np.argmax(hashes))
    x = np.tile(categories, 4).reshape(-1, 1)
    y = np.tile((np.arange(256) == special).astype(np.float32), 4)
    future = [[categories[special]], [categories[(special + 1) % 256]], ["previously-unseen-category"]]
    model = CatBoostMetalRegressor(**options(one_hot_max_size=256, iterations=1, depth=1,
        score_function="L2", leaf_estimation_method="Gradient", leaf_estimation_iterations=1,
        learning_rate=1, border_count=1, random_seed=13))
    _, document = fit_and_compare(model, Pool(x, y, cat_features=[0]), tmp_path, probes=[future])
    assert model.data_partition == "DocParallel"
    split, = document["oblivious_trees"][0]["splits"]
    assert split["split_type"] == "OneHotFeature"
    assert split["value"] & 0xffffffff == int(hashes[special])
    prediction = model.predict(future, task_type="GPU")
    assert prediction[0] > prediction[1]
    assert prediction[1] == prediction[2]


def test_full_frequency_counter_uses_eval_counts_then_exports_learn_only_tables(tmp_path):
    x, y, weights, evaluation, eval_y = frequency_problem()
    learn, test = frequency_pools()
    model = CatBoostMetalRegressor(**options(counter_calc_method="Full", ctr_type="FeatureFreq",
        one_hot_max_size=2, model_size_reg=0, iterations=1, depth=1, learning_rate=1,
        l2_leaf_reg=0, leaf_estimation_iterations=1))
    native, document = fit_and_compare(model, learn, tmp_path, evaluation=test, numeric_cursor=False)
    assert model.data_partition == "DocParallel"
    ctr, = document["features_info"]["ctrs"]
    assert ctr["ctr_type"] == "FeatureFreq"
    split, = document["oblivious_trees"][0]["splits"]
    assert split["split_type"] == "OnlineCtr"
    counts = Counter(np.concatenate((x[:, 0], evaluation[:, 0])))
    encoded = np.array([(counts[value] + ctr["prior_numerator"]) /
                       (len(x) + len(evaluation) + ctr["prior_denomerator"]) for value in x[:, 0]], np.float32)
    encoded = (encoded + np.float32(ctr["shift"])) * np.float32(ctr["scale"])
    leaf = encoded > np.float32(split["border"])
    assert leaf.any() and not leaf.all()
    expected_weights = [weights[leaf == index].sum(dtype=np.float64) for index in (0, 1)]
    expected_leaves = [(weights[leaf == index] * y[leaf == index]).sum(dtype=np.float64) / expected_weights[index]
                       for index in (0, 1)]
    np.testing.assert_allclose(native.get_leaf_weights(), expected_weights, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(native.get_leaf_values(), expected_leaves, rtol=3e-6, atol=3e-6)
    full_eval = independent_prediction(document, np.concatenate((x, evaluation)), np.r_[y, eval_y], evaluation)
    exported_eval = independent_prediction(document, x, y, evaluation)
    np.testing.assert_allclose(native.get_test_eval(), full_eval, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model.predict(test, task_type="GPU"), exported_eval, rtol=3e-6, atol=3e-6)
    assert np.max(np.abs(full_eval - exported_eval)) > .1
    check_final_tables(document, x, y)


def test_multirmse_ridge_preserves_vector_cursor(tmp_path):
    x, y, weights = numeric_problem()
    targets = np.column_stack([y, -.7 * y + x[:, 3]]).astype(np.float32)
    model = CatBoostMetalRegressor(**options(loss_function="MultiRMSE", add_ridge_penalty_to_loss_function=True))
    fit_and_compare(model, Pool(x, targets, weight=weights), tmp_path)
    assert model.n_outputs_ == 2 and model.training_predictions_.shape == targets.shape


@pytest.mark.parametrize("explicit", (False, True))
def test_multilogloss_explicit_and_inferred_labels_preserve_vector_cursor(explicit, tmp_path):
    x, y, weights = numeric_problem()
    targets = np.column_stack([y > 0, x[:, 3] > 0]).astype(np.float32)
    config = options(add_ridge_penalty_to_loss_function=True)
    if explicit:
        config["loss_function"] = "MultiLogloss"
    model = CatBoostMetalClassifier(**config)
    native, _ = fit_and_compare(model, Pool(x, targets, weight=weights), tmp_path)
    assert model.loss_function == "MultiLogloss"
    assert model.n_outputs_ == 2 and model.training_predictions_.shape == targets.shape
    for task in ("CPU", "GPU"):
        np.testing.assert_allclose(model.predict_proba(x, task_type=task),
            native.predict(x, prediction_type="Probability", task_type=task), rtol=8e-6, atol=3e-6)


def test_uncertainty_default_prediction_and_output_count_match_native(tmp_path):
    x, y, weights = numeric_problem()
    model = CatBoostMetalRegressor(**options(loss_function="RMSEWithUncertainty", score_function="L2",
                                            add_ridge_penalty_to_loss_function=True))
    native, _ = fit_and_compare(model, Pool(x, y, weight=weights), tmp_path)
    assert model.n_outputs_ == 2 and model.training_predictions_.shape == (len(x), 2)
    for task in ("CPU", "GPU"):
        expected = native.predict(x, prediction_type="RMSEWithUncertainty", task_type=task)
        np.testing.assert_allclose(model.predict(x, task_type=task), expected, rtol=8e-6, atol=3e-6)


def test_self_init_matches_separate_initial_model_copy(tmp_path):
    x, y, weights = numeric_problem()
    learn = Pool(x, y, weight=weights)
    config = options(add_ridge_penalty_to_loss_function=True)
    model = CatBoostMetalRegressor(**config).fit(learn)
    initial = model.to_catboost()
    copied = CatBoostMetalRegressor(**config).fit(learn, init_model=initial)
    model.fit(learn, init_model=model)
    assert model.tree_count_ == 2 * config["iterations"]
    np.testing.assert_array_equal(model.training_predictions_, copied.training_predictions_)
    native = CatBoost().set_params(**native_parameters(model)).fit(learn, init_model=initial)
    assert_equivalent(model, native, learn, tmp_path)


def test_weighted_yeti_custom_eval_uses_weighted_pfound_objective_history(tmp_path):
    row = np.arange(96)
    x = np.random.default_rng(755).normal(size=(len(row), 4)).astype(np.float32)
    labels = ((row % 8) / 7).astype(np.float32)
    weights = (.5 + row % 7 / 5).astype(np.float32)
    learn = Pool(x, labels, weight=weights, group_id=row // 8)
    model = CatBoostMetalRanker(**options(loss_function="YetiRank:permutations=4;decay=0.8",
        eval_metric="NDCG", add_ridge_penalty_to_loss_function=True))
    native, _ = fit_and_compare(model, learn, tmp_path, evaluation=learn)
    history = native.get_evals_result()["learn"]
    assert "PFound:use_weights=true" in history
    np.testing.assert_array_equal(model.loss_history_[1:], history["PFound:use_weights=true"])


def test_unsupported_scalar_rsm_raises_native_catboost_error():
    x, y, weights = numeric_problem()
    model = CatBoostMetalRegressor(**options(rsm=.5))
    assert model._native_adapter
    with pytest.raises(CatBoostError, match="(?i)rsm|feature subsampling"):
        model.fit(Pool(x, y, weight=weights))


def test_unsupported_full_matrix_langevin_raises_native_catboost_error():
    row = np.arange(64)
    x = np.column_stack([row % 2, row % 7]).astype(np.float32)
    learn = Pool(x, (row % 4) / 3, group_id=row // 8)
    model = CatBoostMetalRanker(**options(loss_function="QueryCrossEntropy", langevin=True,
        diffusion_temperature=.1, leaf_estimation_iterations=1, score_function="NewtonL2"))
    assert model._native_adapter
    with pytest.raises(CatBoostError, match="(?i)langevin"):
        model.fit(learn)


@pytest.mark.parametrize("objective", ("RMSE", "Logloss", "MultiRMSE", "MultiLogloss"))
def test_explicit_simple_scalar_and_vector_dp_keep_sampled_weak_leaf_equations(objective, tmp_path):
    x, y, weights = numeric_problem()
    config = options(loss_function=objective, leaf_estimation_method="Simple", leaf_estimation_iterations=1)
    if objective == "Logloss":
        y = (y > 0).astype(np.float32)
        # Curvature and sampled mass distinguish Simple from an ordinary
        # Gradient refit even when both use a single leaf step.
        config.update(score_function="NewtonCosine", bootstrap_type="Bernoulli", subsample=.43)
    elif objective.startswith("Multi"):
        y = np.column_stack([y, -.7 * y + x[:, 3]]).astype(np.float32)
        if objective == "MultiLogloss":
            y = (y > 0).astype(np.float32)
        config.update(bootstrap_type="Bernoulli", subsample=.43)
    kind = CatBoostMetalClassifier if objective in ("Logloss", "MultiLogloss") else CatBoostMetalRegressor
    model = kind(**config)
    learn = Pool(x, y, weight=weights)
    native, document = fit_and_compare(model, learn, tmp_path, evaluation=learn)
    assert model.data_partition == "DocParallel" and model.grow_policy == "SymmetricTree"
    assert native.get_all_params()["leaf_estimation_method"] == "Simple"
    config = native_parameters(model)
    if y.ndim == 1:
        check_scalar_equations(native, document, x, y, weights, np.zeros(len(x), np.float32), config)
    else:
        assert model.n_outputs_ == y.shape[1]
        cursor = np.zeros_like(y)
        for iteration, tree in enumerate(document["oblivious_trees"]):
            ids = numeric_leaves(tree, x)
            factors = bootstrap_draws(config["bootstrap_type"], len(x), seed=config["random_seed"],
                                      iteration=iteration, subsample=config["subsample"])
            sampled_weights = np.float32(weights * factors)
            gradient = np.float32(vector_gradients(objective, y, cursor, weights) * factors[:, None])
            expected, masses = vector_simple_leaves(objective, gradient, sampled_weights, ids,
                len(tree["leaf_weights"]), config["l2_leaf_reg"], config["learning_rate"])
            np.testing.assert_allclose(tree["leaf_values"], expected.ravel(), rtol=8e-5, atol=6e-6)
            np.testing.assert_allclose(tree["leaf_weights"], masses, rtol=5e-6, atol=5e-6)
            cursor = np.float32(cursor + expected[ids])
        np.testing.assert_allclose(model.training_predictions_, cursor, rtol=8e-5, atol=8e-6)
    if config["bootstrap_type"] != "No":
        first_mass = np.sum(document["oblivious_trees"][0]["leaf_weights"])
        assert not np.isclose(first_mass, weights.sum(dtype=np.float64), rtol=1e-3)


@pytest.mark.parametrize("objective", ("QueryRMSE", "QuerySoftMax:beta=0.7;lambda=0.03", "PairLogit"))
def test_explicit_simple_query_dp_matches_independent_sampled_score_mass(objective, tmp_path):
    data = query_simple_problem(objective)
    data["baseline"].fill(0)
    data["pool"].set_baseline(data["baseline"])
    model = CatBoostMetalRanker(**options(loss_function=objective, leaf_estimation_method="Simple",
        leaf_estimation_iterations=1, score_function="NewtonCosine", bootstrap_type="Bernoulli", subsample=.43))
    native, document = fit_and_compare(model, data["pool"], tmp_path, evaluation=data["pool"])
    assert model.data_partition == "DocParallel" and model.grow_policy == "SymmetricTree"
    assert native.get_all_params()["leaf_estimation_method"] == "Simple"
    check_query_simple(native, document, data, native_parameters(model))


@pytest.mark.parametrize("objective", ("PairLogitPairwise", "YetiRankPairwise"))
def test_full_matrix_simple_with_rsm_uses_native_dp_bridge(objective, tmp_path):
    row = np.arange(96)
    x = np.random.default_rng(755).normal(size=(len(row), 4)).astype(np.float32)
    labels = ((row % 8) / 7).astype(np.float32)
    pairs = [(first + 3, first) for first in range(0, len(row), 4)] if objective == "PairLogitPairwise" else None
    learn = Pool(x, labels, weight=(.5 + row % 7 / 5).astype(np.float32), group_id=row // 8, pairs=pairs)
    model = CatBoostMetalRanker(**options(loss_function=objective, leaf_estimation_method="Simple",
        leaf_estimation_iterations=1, score_function="NewtonL2", rsm=.5))
    native, _ = fit_and_compare(model, learn, tmp_path, evaluation=learn)
    assert model.data_partition == "DocParallel" and model.grow_policy == "SymmetricTree"
    assert native.get_all_params()["leaf_estimation_method"] == "Simple"
    assert native.get_all_params()["rsm"] == .5
