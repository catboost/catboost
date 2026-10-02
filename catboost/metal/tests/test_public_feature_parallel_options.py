"""Host-only public option contracts; no trainer or Metal library is loaded."""
from types import SimpleNamespace

import numpy as np
import pytest
from catboost import CatBoost, Pool
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRanker, CatBoostMetalRegressor
from catboost_metal._feature_parallel_frontend import _Callback, _evaluation_pool, _feature_names, _pool, native_parameters


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Host option tests cannot fit models")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


@pytest.mark.parametrize("kind", (CatBoostMetalRegressor, CatBoostMetalClassifier, CatBoostMetalRanker))
@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
@pytest.mark.parametrize("count", (1, 4))
def test_native_ctr_options_reach_shared_adapter(kind, boosting, count):
    ctr = "Borders:Prior=0.2:Prior=0.7:CtrBorderCount=11"
    model = kind(data_partition="FeatureParallel", boosting_type=boosting, permutation_count=count,
                 max_ctr_complexity=3, simple_ctr=ctr, combinations_ctr=[ctr],
                 ctr_target_border_count=2, counter_calc_method="Full", one_hot_max_size=2,
                 fold_permutation_block=3, min_fold_size=2, fold_size_loss_normalization=True)
    params = native_parameters(model)
    assert model._native_feature_parallel
    assert params["task_type"] == "GPU" and params["boosting_type"] == boosting
    assert params["max_ctr_complexity"] == 3 and params["simple_ctr"] == params["combinations_ctr"] == [ctr]
    assert params["permutation_count"] == count and params["has_time"] == (count == 1)
    assert params["counter_calc_method"] == "Full" and params["ctr_target_border_count"] == 2
    assert params["fold_permutation_block"] == 3 and params["fold_size_loss_normalization"]


@pytest.mark.parametrize("kind", ("Borders", "Buckets", "FloatTargetMeanValue", "FeatureFreq"))
def test_ctr_shortcut_preserves_prior_and_border_count(kind):
    params = native_parameters(CatBoostMetalRegressor(data_partition="FeatureParallel", ctr_type=kind,
                               ctr_prior=.25, ctr_border_count=11))
    assert params["simple_ctr"] == params["combinations_ctr"] == [
        f"{kind}:CtrBorderType=Uniform:CtrBorderCount=11:Prior=0.25"]


@pytest.mark.parametrize("kwargs,match", [
    ({"max_ctr_complexity": 2, "data_partition": "DocParallel"}, "Compound CTRs"),
    ({"data_partition": "FeatureParallel", "grow_policy": "Depthwise"}, "Non-symmetric"),
    ({"data_partition": "FeatureParallel", "loss_function": "MultiRMSE"}, "scalar"),
    ({"simple_ctr": "Borders"}, "require FeatureParallel"),
    ({"data_partition": "FeatureParallel", "leaf_estimation_method": "Simple", "leaf_estimation_iterations": 2}, "one estimation"),
])
def test_unsupported_combinations_are_explicit(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CatBoostMetalRegressor(**kwargs)


@pytest.mark.parametrize("kwargs,match", [
    ({"ctr_target_border": .4}, "ctr_target_border"),
    ({"simple_ctr": []}, "nonempty"),
    ({"combinations_ctr": [None]}, "nonempty"),
])
def test_native_conversion_rejects_unsupported_or_empty_ctr_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        native_parameters(CatBoostMetalRegressor(data_partition="FeatureParallel", **kwargs))


def test_existing_default_modes_keep_the_original_runtime():
    assert not CatBoostMetalRegressor()._native_feature_parallel
    assert not CatBoostMetalRegressor(boosting_type="Ordered")._native_feature_parallel
    assert not CatBoostMetalRanker(boosting_type="Ordered")._native_feature_parallel
    assert CatBoostMetalRegressor(max_ctr_complexity=2).data_partition == "FeatureParallel"


@pytest.mark.parametrize("loss", ("RMSE", "Poisson", "Expectile:alpha=0.7"))
def test_simple_uses_one_iteration_without_objective_default_leakage(loss):
    model = CatBoostMetalRegressor(data_partition="FeatureParallel", loss_function=loss,
                                  leaf_estimation_method="Simple")
    assert native_parameters(model)["leaf_estimation_iterations"] == 1


def test_feature_weights_resolve_original_names_then_expand_ctr_sources():
    model = CatBoostMetalRegressor(feature_weights={"category": .2, 0: 3})
    layout = SimpleNamespace(borders=[[], [], [], []], ctrs={2: SimpleNamespace(source_feature=1),
                                                          3: SimpleNamespace(source_feature=1)})
    np.testing.assert_array_equal(model._feature_weights(["numeric", "category"], layout),
                                  np.asarray([3, .2, .2, .2], np.float32))
    with pytest.raises(ValueError, match="duplicate"):
        CatBoostMetalRegressor(feature_weights={"numeric": 1, 0: 2})._feature_weights(["numeric"])
    for invalid in ([-1], [np.nan], [np.inf], [1, 2]):
        with pytest.raises(ValueError, match="feature_weights"):
            CatBoostMetalRegressor(feature_weights=invalid)._feature_weights(["numeric"])


def test_pool_preserves_group_weights_pairs_and_names_without_reencoding():
    rows = [["a", 0.], ["b", 1.], ["a", 2.], ["c", 3.]]
    pool = Pool(rows, [0, 1, 0, 1], cat_features=[0], feature_names=["category", "numeric"],
                group_id=[4, 4, 8, 8], group_weight=[2., 2., 3., 3.],
                pairs=[[1, 0], [3, 2]], pairs_weight=[.2, .7])
    assert _pool(pool, cats=["category"]) is pool
    assert pool.get_group_weight() == [2, 2, 3, 3]
    np.testing.assert_allclose(pool.get_pairs_weight(), [.2, .7])
    with pytest.raises(ValueError, match="inside the Pool"):
        _pool(pool, weight=[1] * 4)
    with pytest.raises(ValueError, match="disagrees"):
        _pool(pool, cats=[1])
    combined = _pool(rows, [0, 1, 0, 1], cats=[0], weight=[1, 2, 3, 4],
                     groups=[4, 4, 8, 8], group_weight=[2., 2., 3., 3.])
    assert combined.get_weight() == [1, 2, 3, 4]
    assert combined.get_group_weight() == [2, 2, 3, 3]


def test_evaluation_tuple_does_not_drop_duplicate_or_orphan_metadata():
    options = dict(ranker=False, cats=None, groups=None, weight=None, group_weight=None,
                   subgroups=None, pairs=None, pairs_weight=None)
    result = _evaluation_pool(([[0], [1]], [0, 1], [.2, .7]), **options)
    np.testing.assert_allclose(result.get_weight(), [.2, .7])
    with pytest.raises(ValueError, match="not both"):
        _evaluation_pool(([[0], [1]], [0, 1], [.2, .7]), **(options | {"weight": [1, 1]}))
    with pytest.raises(ValueError, match="require eval_set"):
        _evaluation_pool(None, **(options | {"pairs": [[1, 0]]}))


def test_callback_info_is_isolated_and_false_stops():
    info = SimpleNamespace(iteration=2, metrics={"learn": {"RMSE": [1.]}})
    def callback(value):
        value.metrics.clear()
        return False
    adapter = _Callback(callback)
    assert adapter.after_iteration(info) is False
    assert info.metrics and adapter.stopped and adapter.iterations == [2]


def test_anonymous_pool_names_avoid_named_column_collisions():
    pool = Pool([[0, 1], [1, 0]], feature_names=["", "0"])
    assert _feature_names(pool) == ["feature_0", "0"]
    assert pool.get_feature_names() == ["", "0"]
    model = CatBoostMetalRegressor(feature_weights={"feature_0": .5})
    np.testing.assert_array_equal(model._feature_weights(_feature_names(pool)), [.5, 1])


def test_old_extension_fails_before_fit_with_a_specific_requirement(monkeypatch):
    import catboost
    import catboost_metal._feature_parallel_frontend as frontend
    class OldModel:
        _object = object()
        def set_params(self, **params):
            return self
    monkeypatch.setattr(catboost, "CatBoostRegressor", OldModel)
    monkeypatch.setattr(frontend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(frontend.platform, "machine", lambda: "arm64")
    with pytest.raises(RuntimeError, match="rebuilt CatBoost Metal extension with online-cursor support"):
        CatBoostMetalRegressor(data_partition="FeatureParallel").fit([[0], [1]], [0, 1])


def test_yeti_requests_learn_metric_without_changing_its_default_metric():
    params = native_parameters(CatBoostMetalRanker(data_partition="FeatureParallel", loss_function="YetiRank"))
    assert params["eval_metric"] == "PFound:hints=skip_train~false"
    assert params["custom_metric"] == ["PFound:use_weights=true;hints=skip_train~false"]
