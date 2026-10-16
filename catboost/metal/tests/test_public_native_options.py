"""Host-only option routing and native translation; no model fit or GPU access."""

from types import SimpleNamespace
import pickle

import numpy as np
import pytest
from catboost import CatBoost
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRanker, CatBoostMetalRegressor, _native
from catboost_metal import _feature_parallel_frontend as frontend
from catboost_metal._feature_parallel_frontend import _tree_depths, native_parameters


# Retain only the public dispatch functions for tests that replace their native
# bridge with a host stub. The fixture still forbids all actual fitting paths.
_REGRESSOR_DISPATCH = CatBoostMetalRegressor.fit
_RANKER_DISPATCH = CatBoostMetalRanker.fit


DEFAULTS = dict(
    fixed_binary_splits=None,
    rsm=1.,
    add_ridge_penalty_to_loss_function=False,
    meta_l2_exponent=1.,
    meta_l2_frequency=0.,
    langevin=False,
    diffusion_temperature=0.,
    fold_size_loss_normalization=False,
)
REGULARIZATION = dict(
    fold_size_loss_normalization=True,
    add_ridge_penalty_to_loss_function=True,
    meta_l2_exponent=1.25,
    meta_l2_frequency=.5,
    langevin=True,
    diffusion_temperature=4.,
)


@pytest.fixture(autouse=True)
def no_training_or_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Host option tests must not fit models, build libraries, or load Metal")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)
    monkeypatch.setattr(CatBoostMetalRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostMetalRanker, "fit", forbidden)
    monkeypatch.setattr(_native, "build_library", forbidden)
    monkeypatch.setattr(_native, "_load", forbidden)


@pytest.mark.parametrize("kind", (CatBoostMetalRegressor, CatBoostMetalClassifier, CatBoostMetalRanker))
@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_default_options_preserve_existing_runtime(kind, boosting):
    implicit = kind(boosting_type=boosting)
    explicit = kind(boosting_type=boosting, **DEFAULTS)
    for model in (implicit, explicit):
        assert model.data_partition == ("FeatureParallel" if boosting == "Ordered" else "DocParallel")
        assert not model._native_feature_parallel
        assert not model._native_adapter
        for name, value in DEFAULTS.items():
            assert getattr(model, name) == value


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
def test_default_greedy_options_preserve_existing_runtime(policy):
    model = CatBoostMetalRegressor(grow_policy=policy, **DEFAULTS)
    assert model.data_partition == "DocParallel"
    assert not model._native_feature_parallel
    assert not model._native_adapter


@pytest.mark.parametrize("options", ({"data_partition": "FeatureParallel"}, {"max_ctr_complexity": 2},
                                      {"boosting_type": "Ordered", "max_ctr_complexity": 2}))
def test_existing_native_feature_parallel_routes_remain_enabled(options):
    model = CatBoostMetalRegressor(**options)
    assert model._native_feature_parallel
    assert model._native_adapter


@pytest.mark.parametrize("kind,objective", (
    (CatBoostMetalRegressor, "RMSE"),
    (CatBoostMetalRegressor, "MultiRMSE"),
    (CatBoostMetalRegressor, "RMSEWithUncertainty"),
    (CatBoostMetalClassifier, "MultiClass"),
    (CatBoostMetalRanker, "QueryRMSE"),
    (CatBoostMetalRanker, "QuerySoftMax"),
    (CatBoostMetalRanker, "PairLogit"),
))
def test_explicit_simple_symmetric_leaves_select_native_doc_parallel(kind, objective):
    model = kind(loss_function=objective, leaf_estimation_method="Simple")
    assert model._native_adapter
    assert not model._native_feature_parallel
    params = native_parameters(model)
    assert params["loss_function"] == objective
    assert params["leaf_estimation_method"] == "Simple"
    assert params["leaf_estimation_iterations"] == 1
    assert params["data_partition"] == "DocParallel"
    assert params["grow_policy"] == "SymmetricTree"


@pytest.mark.parametrize("objective", ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise"))
@pytest.mark.parametrize("rsm", (1., .5))
def test_full_matrix_simple_keeps_existing_runtime_until_native_option_is_requested(objective, rsm):
    model = CatBoostMetalRanker(loss_function=objective, leaf_estimation_method="Simple", rsm=rsm)
    assert model.leaf_estimation_method == "Simple"
    assert model.leaf_estimation_iterations == 1
    assert model._native_adapter == (rsm < 1.)
    assert not model._native_feature_parallel
    assert model.data_partition == "DocParallel"
    if rsm < 1.:
        params = native_parameters(model)
        assert params["loss_function"] == objective
        assert params["leaf_estimation_method"] == "Simple"
        assert params["leaf_estimation_iterations"] == 1
        assert params["rsm"] == rsm
        assert params["data_partition"] == "DocParallel"


def test_implicit_yeti_pair_simple_keeps_its_existing_runtime():
    model = CatBoostMetalRanker(loss_function="YetiRankPairwise")
    assert model.leaf_estimation_method == "Simple"
    assert model.leaf_estimation_iterations == 1
    assert not model._native_adapter
    assert not model._native_feature_parallel


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
def test_greedy_simple_keeps_its_existing_runtime(policy):
    model = CatBoostMetalRegressor(grow_policy=policy, leaf_estimation_method="Simple")
    assert model.leaf_estimation_iterations == 1
    assert not model._native_adapter
    assert not model._native_feature_parallel


@pytest.mark.parametrize("kind,options,match", (
    (CatBoostMetalRegressor, {"leaf_estimation_iterations": 2}, "one estimation iteration"),
    (CatBoostMetalRanker, {"loss_function": "QueryRMSE", "leaf_estimation_iterations": 2}, "one estimation iteration"),
    (CatBoostMetalRanker, {"loss_function": "YetiRank"}, "requires Newton"),
))
def test_simple_keeps_iteration_and_classic_yeti_restrictions(kind, options, match):
    with pytest.raises(ValueError, match=match):
        kind(leaf_estimation_method="Simple", **options)


@pytest.mark.parametrize("name,value", (
    ("fixed_binary_splits", [0, 3]),
    ("rsm", .5),
    ("add_ridge_penalty_to_loss_function", True),
    ("meta_l2_exponent", 1.25),
    ("meta_l2_frequency", .5),
    ("langevin", True),
    ("diffusion_temperature", 4.),
    ("fold_size_loss_normalization", True),
))
def test_nondefault_options_select_native_adapter_without_changing_partition(name, value):
    model = CatBoostMetalRegressor(**{name: value})
    assert model._native_adapter
    assert not model._native_feature_parallel
    params = native_parameters(model)
    assert params["task_type"] == "GPU"
    assert params["data_partition"] == "DocParallel"
    assert params["grow_policy"] == "SymmetricTree"
    assert params[name] == value


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
def test_fixed_splits_keep_greedy_doc_parallel_options(policy):
    options = dict(grow_policy=policy, depth=4, min_data_in_leaf=3, fixed_binary_splits=[2, 0, 7])
    if policy == "Lossguide":
        options["max_leaves"] = 7
    model = CatBoostMetalRegressor(**options)
    assert model._native_adapter
    assert not model._native_feature_parallel
    params = native_parameters(model)
    assert params["task_type"] == "GPU"
    assert params["data_partition"] == "DocParallel"
    assert params["grow_policy"] == policy
    assert params["fixed_binary_splits"] == [2, 0, 7]
    assert params["min_data_in_leaf"] == 3
    if policy == "Lossguide":
        assert params["max_leaves"] == 7
    else:
        assert "max_leaves" not in params


@pytest.mark.parametrize("objective", ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise"))
def test_full_matrix_ranker_rsm_stays_doc_parallel(objective):
    model = CatBoostMetalRanker(loss_function=objective, rsm=.5, bayesian_matrix_reg=.375)
    assert model._native_adapter
    assert not model._native_feature_parallel
    params = native_parameters(model)
    assert params["loss_function"] == objective
    assert params["data_partition"] == "DocParallel"
    assert params["grow_policy"] == "SymmetricTree"
    assert params["rsm"] == .5
    assert params["bayesian_matrix_reg"] == .375
    assert params["sampling_unit"] == "Object"
    assert "max_leaves" not in params
    assert "min_data_in_leaf" not in params


def test_group_sampling_and_matrix_regularization_reach_native_ranker():
    model = CatBoostMetalRanker(loss_function="YetiRankPairwise:permutations=3;decay=0.75",
                               sampling_unit="Group", bayesian_matrix_reg=.25, rsm=.5)
    params = native_parameters(model)
    assert params["loss_function"] == "YetiRankPairwise:permutations=3;decay=0.75"
    assert params["sampling_unit"] == "Group"
    assert params["bayesian_matrix_reg"] == .25
    assert params["data_partition"] == "DocParallel"


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
@pytest.mark.parametrize("kind", (CatBoostMetalRegressor, CatBoostMetalClassifier, CatBoostMetalRanker))
def test_regularization_reaches_scalar_feature_parallel_and_ordered(kind, boosting):
    model = kind(boosting_type=boosting, data_partition="FeatureParallel", **REGULARIZATION)
    assert model._native_adapter
    assert model._native_feature_parallel == (boosting == "Plain")
    params = native_parameters(model)
    assert params["task_type"] == "GPU"
    assert params["data_partition"] == "FeatureParallel"
    assert params["boosting_type"] == boosting
    assert params["grow_policy"] == "SymmetricTree"
    for name, value in REGULARIZATION.items():
        assert params[name] == value
    assert "max_leaves" not in params
    assert "min_data_in_leaf" not in params


def test_ordered_normalization_alone_keeps_its_existing_runtime():
    model = CatBoostMetalRegressor(boosting_type="Ordered", fold_size_loss_normalization=True)
    assert not model._native_feature_parallel
    assert not model._native_adapter
    assert model.fold_size_loss_normalization


@pytest.mark.parametrize("maximum", (1, 2, 255))
def test_existing_one_hot_limits_keep_the_original_runtime(maximum):
    model = CatBoostMetalRegressor(one_hot_max_size=maximum)
    assert not model._native_adapter
    assert not model._native_feature_parallel


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_one_hot_256_uses_native_adapter_without_changing_partition(boosting):
    model = CatBoostMetalRegressor(boosting_type=boosting, one_hot_max_size=256)
    assert model._native_adapter
    assert not model._native_feature_parallel
    params = native_parameters(model)
    assert params["one_hot_max_size"] == 256
    assert params["data_partition"] == ("FeatureParallel" if boosting == "Ordered" else "DocParallel")


@pytest.mark.parametrize("maximum", (257, 65536))
def test_one_hot_limits_above_256_are_rejected(maximum):
    with pytest.raises(ValueError, match="one_hot_max_size"):
        CatBoostMetalRegressor(one_hot_max_size=maximum)


@pytest.mark.parametrize("kind", (CatBoostMetalRegressor, CatBoostMetalClassifier, CatBoostMetalRanker))
def test_full_counter_calculation_routes_default_doc_parallel_to_native(kind):
    model = kind(counter_calc_method="Full")
    assert model._native_adapter
    assert not model._native_feature_parallel
    params = native_parameters(model)
    assert params["counter_calc_method"] == "Full"
    assert params["data_partition"] == "DocParallel"
    assert params["grow_policy"] == "SymmetricTree"


def test_symmetric_depths_follow_leaf_counts_without_reading_greedy_nodes():
    model = SimpleNamespace(get_tree_leaf_counts=lambda: np.array([1, 2, 8], np.uint32))
    np.testing.assert_array_equal(_tree_depths(model, "SymmetricTree"), [0, 1, 3])


def test_greedy_depths_distinguish_balanced_and_chain_trees_with_four_leaves():
    # Both trees have four leaves. Their longest root-to-leaf paths have
    # lengths two and three, so log2(leaf_count) cannot recover both depths.
    steps = [
        [(1, 4), (1, 2), (0, 0), (0, 0), (1, 2), (0, 0), (0, 0)],
        [(0, 1), (1, 0), (0, 1), (0, 0)],
    ]
    model = SimpleNamespace(tree_count_=2, _get_tree_step_nodes=steps.__getitem__)
    np.testing.assert_array_equal(_tree_depths(model, "Lossguide"), [2, 3])


def test_greedy_depths_follow_relative_child_steps_and_handle_constant_tree():
    # Unreachable entries between relative child jumps do not add depth.
    steps = [
        [(2, 5), (0, 6), (0, 1), (0, 0), (0, 3), (0, 1), (0, 0), (0, 0)],
        [(0, 0)],
    ]
    model = SimpleNamespace(tree_count_=2, _get_tree_step_nodes=steps.__getitem__)
    np.testing.assert_array_equal(_tree_depths(model, "Region"), [2, 0])


@pytest.mark.parametrize("method,iterations", (("Newton", 10), ("Gradient", 40)))
@pytest.mark.parametrize("labels,objective", (
    ([[0, 1], [1, 0], [1, 1]], "MultiLogloss"),
    ([[.25, .75], [1, .5], [0, 1]], "MultiCrossEntropy"),
))
def test_implicit_multilabel_loss_is_selected_before_native_dispatch(monkeypatch, method, iterations,
                                                                    labels, objective):
    model = CatBoostMetalClassifier(one_hot_max_size=256, leaf_estimation_method=method)
    features = np.zeros((3, 1), np.float32)
    labels = np.asarray(labels, np.float32)
    calls = []

    def bridge(estimator, x, y, weight, **options):
        assert estimator is model and x is features and y is labels and weight is None
        assert estimator._objective == estimator.loss_function == objective
        assert estimator.leaf_estimation_iterations == iterations
        calls.append(native_parameters(estimator))
        return estimator

    monkeypatch.setattr(frontend, "fit_feature_parallel", bridge)
    assert _REGRESSOR_DISPATCH(model, features, labels) is model
    assert len(calls) == 1
    assert calls[0]["loss_function"] == objective
    assert calls[0]["data_partition"] == "DocParallel"


@pytest.mark.parametrize("requested,expected", ((None, "RMSEWithUncertainty"),
                                                ("RawFormulaVal", "RawFormulaVal")))
def test_native_uncertainty_prediction_preserves_its_default_transform(requested, expected):
    calls = []
    result = np.array([[1.5, 4.]])

    def predict(data, **options):
        calls.append(options)
        return result

    model = SimpleNamespace(_classifier=False, _objective="RMSEWithUncertainty",
                            _model=SimpleNamespace(predict=predict))
    assert frontend.predict_feature_parallel(model, [[0.]], prediction_type=requested,
                                              ntree_start=1, ntree_end=3) is result
    assert calls == [dict(task_type="CPU", prediction_type=expected, ntree_start=1, ntree_end=3)]


@pytest.mark.parametrize("kind,dispatch", ((CatBoostMetalRegressor, _REGRESSOR_DISPATCH),
                                           (CatBoostMetalRanker, _RANKER_DISPATCH)))
def test_self_continuation_copies_the_initial_model_before_reset(monkeypatch, kind, dispatch):
    estimator = kind(one_hot_max_size=256)
    initial_model = object()
    calls = []

    def copy():
        assert estimator._model is fitted
        calls.append("copy")
        return initial_model

    fitted = SimpleNamespace(copy=copy)
    estimator._model = fitted

    def bridge(model, *args, **options):
        assert model is estimator
        assert options["init_model"] is initial_model
        assert model._model is None
        calls.append("bridge")
        return model

    monkeypatch.setattr(frontend, "fit_feature_parallel", bridge)
    assert dispatch(estimator, [[0.], [1.]], [0., 1.], init_model=estimator) is estimator
    assert calls == ["copy", "bridge"]


def test_shared_eval_setter_preserves_scalar_values():
    model = CatBoost()
    values = [[[1., -.25, 3.5]]]
    model._set_test_evals(values)
    assert model.get_test_evals() == values
    assert model.get_test_eval() == [1., -.25, 3.5]
    values[0][0][0] = 99.
    assert model.get_test_eval() == [1., -.25, 3.5]


def test_shared_eval_setter_preserves_dimensions_and_multiple_eval_sets():
    model = CatBoost()
    values = [
        [[1., 2., 3.], [-1., -.5, 0.]],
        [[4., 5.], [.25, .75]],
    ]
    model._set_test_evals(values)
    assert model.get_test_evals() == values
    single = CatBoost()
    single._set_test_evals([values[0]])
    assert single.get_test_eval() == values[0]


def test_shared_eval_setter_replaces_prior_sets_dimensions_and_rows():
    model = CatBoost()
    model._set_test_evals([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    replacement = [[[9.]]]
    model._set_test_evals(replacement)
    assert model.get_test_evals() == replacement
    assert model.get_test_eval() == [9.]
    replacement = [[[10., 11., 12.]], [[13., 14.]]]
    model._set_test_evals(replacement)
    assert model.get_test_evals() == replacement


def test_shared_eval_values_survive_copy_and_pickle_without_aliasing():
    source = CatBoost()
    expected = [[[1., 2.], [-1., -2.]], [[3.], [-3.]]]
    source._set_test_evals(expected)
    clones = (source.copy(), pickle.loads(pickle.dumps(source)))
    for clone in clones:
        assert clone.get_test_evals() == expected
        changed = clone.get_test_evals()
        changed[0][0][0] = 42.
        assert clone.get_test_evals() == expected
        clone._set_test_evals(changed)
        assert clone.get_test_evals() == changed
        assert source.get_test_evals() == expected


@pytest.mark.parametrize("name", ("fold_size_loss_normalization", "add_ridge_penalty_to_loss_function", "langevin"))
@pytest.mark.parametrize("value", (0, 1, "true", None))
def test_boolean_options_reject_non_boolean_values(name, value):
    with pytest.raises(ValueError, match=name):
        CatBoostMetalRegressor(**{name: value})


@pytest.mark.parametrize("name", ("rsm", "meta_l2_exponent", "meta_l2_frequency", "diffusion_temperature"))
@pytest.mark.parametrize("value", (True, False, "0.5", 1j, None, [], np.nan, np.inf, -np.inf))
def test_numeric_options_reject_invalid_types_and_nonfinite_values(name, value):
    with pytest.raises(ValueError, match=name):
        CatBoostMetalRegressor(**{name: value})


@pytest.mark.parametrize("value", (0., -.5, 1.5))
def test_rsm_requires_a_positive_fraction(value):
    with pytest.raises(ValueError, match="rsm"):
        CatBoostMetalRegressor(rsm=value)


@pytest.mark.parametrize("value", (True, 1, "0", [True], [-1], [1.], [1.5], ["1"], [None], [[1]], [2**32]))
def test_fixed_splits_require_unsigned_integer_indices(value):
    with pytest.raises(ValueError, match="fixed_binary_splits"):
        CatBoostMetalRegressor(fixed_binary_splits=value)
