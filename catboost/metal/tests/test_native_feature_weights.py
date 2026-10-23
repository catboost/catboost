"""Native feature weights: independent gains, source IDs and saved state."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the native Metal feature-weight adapter",
)


def problem():
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), (32, 1))
    y = 4 * x[:, 0] + x[:, 1]
    return x, y


def options(path="PlainDP", **extra):
    result = dict(task_type="GPU", loss_function="RMSE", iterations=1, depth=1,
                  learning_rate=1, l2_leaf_reg=2, bootstrap_type="No", random_strength=0,
                  score_function="L2", leaf_estimation_method="Gradient", leaf_estimation_iterations=1,
                  leaf_estimation_backtracking="No", boost_from_average=False,
                  permutation_count=1, has_time=True, border_count=1, random_seed=13,
                  verbose=False, allow_writing_files=False)
    if path in ("PlainFP", "OrderedFP"):
        result.update(data_partition="FeatureParallel", score_function="Cosine")
    if path == "OrderedFP":
        result.update(boosting_type="Ordered", min_fold_size=8)
    if path in ("Depthwise", "Lossguide", "Region"):
        result.update(grow_policy=path)
        if path == "Lossguide":
            result["max_leaves"] = 2
    return result | extra


def root_feature(model, tmp_path, name):
    path = tmp_path / (name + ".json")
    model.save_model(path, format="json")
    document = json.loads(path.read_text())
    if "oblivious_trees" in document:
        return document["oblivious_trees"][0]["splits"][0]["float_feature_index"]
    return document["trees"][0]["split"]["float_feature_index"]


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP", "Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("quantized", [False, True])
def test_source_gain_weight_changes_selected_feature(path, quantized, tmp_path):
    x, y = problem()
    pool = Pool(x, y)
    if quantized:
        pool.quantize(border_count=1)
    # Independent depth-one L2 gains from original rows, including L2.
    gains = []
    for feature in range(2):
        groups = [y[x[:, feature] == value] for value in (0, 1)]
        gains.append(sum(float(v.sum()) ** 2 / (len(v) + 2) for v in groups))
    assert gains[0] > gains[1] and gains[0] * .001 < gains[1]
    reference = CatBoost(options(path)).fit(pool)
    weighted = CatBoost(options(path, feature_weights=[.001, 1])).fit(pool)
    assert root_feature(reference, tmp_path, "reference") == 0
    assert root_feature(weighted, tmp_path, "weighted") == 1
    assert weighted.get_metadata()["metal_backend"] == "METAL"
    np.testing.assert_allclose(weighted.predict(Pool(x), task_type="GPU"), weighted.predict(Pool(x)),
                               rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("form", ["indices", "names", "list"])
def test_ignored_original_columns_keep_manager_ids(form, tmp_path):
    x, y = problem()
    expanded = np.column_stack((np.zeros(len(y), np.float32), x))
    pool = Pool(expanded, y, feature_names=["ignored", "strong", "weak"])
    weights = {"indices": {1: .001}, "names": {"strong": .001}, "list": [1, .001, 1]}[form]
    model = CatBoost(options(ignored_features=[0], feature_weights=weights)).fit(pool)
    # Model float-feature indices include the ignored original column.
    assert root_feature(model, tmp_path, form) == 2


class StopAfter:
    def __init__(self, stop=None):
        self.stop = stop
        self.seen = []

    def after_iteration(self, info):
        self.seen.append(info.iteration)
        return self.stop is None or info.iteration < self.stop


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP", "Depthwise", "Lossguide", "Region"])
def test_feature_weights_snapshot_and_model_roundtrip(path, tmp_path):
    x, y = problem()
    pool = Pool(x, y)
    config = options(path, iterations=5, learning_rate=.2, feature_weights={0: .1, 1: 2})
    full = CatBoost(config).fit(pool, eval_set=pool, use_best_model=False)
    saved = config | dict(save_snapshot=True, snapshot_interval=0,
                          snapshot_file=str(tmp_path / "state"), train_dir=str(tmp_path),
                          allow_writing_files=True)
    CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(2)])
    callback = StopAfter()
    resumed = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[callback])
    assert callback.seen == [3, 4, 5]
    for name in ("get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(full, name)(), getattr(resumed, name)())
    assert full.get_evals_result() == resumed.get_evals_result()
    expected = full.predict(pool, task_type="GPU")
    for fmt in ("cbm", "json"):
        file = tmp_path / ("weighted." + fmt)
        resumed.save_model(file, format=fmt)
        restored = CatBoost().load_model(file, format=fmt)
        np.testing.assert_array_equal(restored.predict(pool, task_type="GPU"), expected)


@pytest.mark.parametrize("weights", [{0: -1}, {99: 2}])
def test_invalid_feature_weight_is_rejected(weights):
    x, y = problem()
    with pytest.raises(CatBoostError):
        CatBoost(options(feature_weights=weights)).fit(x, y)


@pytest.fixture(autouse=True)
def require_metal_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def duplicated_feature_pool(loss):
    """Equal columns have equal candidate statistics for every objective.

    Opposite feature weights must change the chosen column while preserving the
    leaves. This independently distinguishes gain weighting from accidentally
    weighting targets, derivatives, curvature, or exported leaf masses.
    """
    row = np.arange(128)
    z = (row % 2).astype(np.float32)
    x = np.column_stack((z, z))
    weight = (.5 + (row % 8) * .2).astype(np.float32)
    if loss in ("MultiClass", "MultiClassOneVsAll"):
        y = (2 * z + (row // 2) % 2).astype(np.int32)
    elif loss == "MultiRMSE":
        y = np.column_stack((2 * z - .7, .4 - 1.2 * z, z + .1))
    elif loss == "MultiLogloss":
        y = np.column_stack((z, 1 - z))
    elif loss == "MultiCrossEntropy":
        y = np.column_stack((.15 + .7 * z, .85 - .7 * z))
    elif loss == "RMSEWithUncertainty":
        y = 1 + z + .05 * ((row // 2) % 3)
    else:
        y = .1 + .7 * z + .05 * ((row // 2) % 2)
    extra = {}
    if loss.startswith("Query") or loss.startswith("Pair") or loss.startswith("Yeti"):
        extra["group_id"] = row // 8
    if loss in ("PairLogit", "PairLogitPairwise"):
        extra["pairs"] = [(int(i + 1), int(i)) for i in row[::2]]
        extra["pairs_weight"] = [1 + (i % 5) / 3 for i in range(len(row) // 2)]
    return Pool(x, y, weight=weight, **extra)


VECTOR_PATHS = [(loss, "PlainDP") for loss in
                ("MultiClass", "MultiClassOneVsAll", "MultiRMSE", "RMSEWithUncertainty",
                 "MultiLogloss", "MultiCrossEntropy")]
VECTOR_PATHS += [(loss, path) for loss in ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")
                 for path in ("Depthwise", "Lossguide", "Region")]
QUERY_PATHS = [(loss, path) for loss in ("QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank")
              for path in ("PlainDP", "PlainFP", "OrderedFP", "Depthwise", "Lossguide", "Region")]
FULL_MATRIX_PATHS = [(loss, method) for loss in
                     ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise")
                     for method in ("Newton", "Simple")]


def check_gain_only_weights(pool, config, tmp_path):
    first = CatBoost(config | dict(feature_weights=[1, .001])).fit(pool)
    second = CatBoost(config | dict(feature_weights=[.001, 1])).fit(pool)
    assert root_feature(first, tmp_path, "first") == 0
    assert root_feature(second, tmp_path, "second") == 1
    for method in ("get_leaf_values", "get_leaf_weights", "get_tree_leaf_counts"):
        np.testing.assert_array_equal(getattr(first, method)(), getattr(second, method)())
    assert first.get_evals_result() == second.get_evals_result()
    raw = first.predict(pool, prediction_type="RawFormulaVal", task_type="GPU")
    np.testing.assert_array_equal(second.predict(pool, prediction_type="RawFormulaVal", task_type="GPU"), raw)
    np.testing.assert_allclose(second.predict(pool, prediction_type="RawFormulaVal"), raw,
                               rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("loss,path", VECTOR_PATHS)
def test_vector_and_greedy_vector_weights_change_gain_without_changing_leaves(loss, path, tmp_path):
    check_gain_only_weights(duplicated_feature_pool(loss), options(path, loss_function=loss), tmp_path)


@pytest.mark.parametrize("loss,path", QUERY_PATHS)
def test_query_weights_change_gain_without_changing_group_or_edge_statistics(loss, path, tmp_path):
    check_gain_only_weights(duplicated_feature_pool(loss),
        options(path, loss_function=loss, leaf_estimation_method="Newton" if loss == "YetiRank" else "Gradient"), tmp_path)


@pytest.mark.parametrize("loss,method", FULL_MATRIX_PATHS)
def test_full_matrix_weights_preserve_coupled_solution_and_simple_statistics(loss, method, tmp_path):
    check_gain_only_weights(duplicated_feature_pool(loss),
        options(loss_function=loss, score_function="NewtonL2", leaf_estimation_method=method), tmp_path)


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP"])
def test_onehot_original_manager_id_weights_only_its_candidate(path, tmp_path):
    x, y = problem()
    x = np.column_stack((np.where(x[:, 0] == 0, "left", "right"), x[:, 0].astype(object)))
    pool = Pool(x, y, cat_features=[0])
    first = CatBoost(options(path, one_hot_max_size=2, feature_weights=[1, .001])).fit(pool)
    second = CatBoost(options(path, one_hot_max_size=2, feature_weights=[.001, 1])).fit(pool)
    documents = []
    for name, model in (("category", first), ("numeric", second)):
        file = tmp_path / (name + ".json")
        model.save_model(file, format="json")
        documents.append(json.loads(file.read_text()))
    assert documents[0]["oblivious_trees"][0]["splits"][0]["split_type"] == "OneHotFeature"
    assert documents[1]["oblivious_trees"][0]["splits"][0]["split_type"] == "FloatFeature"
    np.testing.assert_allclose(first.predict(pool), second.predict(pool), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP"])
def test_simple_ctr_owns_an_independent_appended_manager_id(path, tmp_path):
    row = np.arange(480)
    category, numeric = row % 6, (row // 6) % 2
    x = np.column_stack(([f"c{value}" for value in category], numeric.astype(object)))
    y = (4 * (category >= 3) + .2 * numeric).astype(np.float32)
    pool = Pool(x, y, cat_features=[0])
    config = options(path, one_hot_max_size=2, max_ctr_complexity=1,
                     simple_ctr=["Borders:CtrBorderCount=7:Prior=0.5"], model_size_reg=0)
    ordinary = CatBoost(config).fit(pool)
    category_weight = CatBoost(config | dict(feature_weights={0: 0})).fit(pool)
    # Original category is manager 0, numeric is manager 1, and the simple CTR
    # is independently appended as manager 2, despite having no input column 2.
    ctr_weight = CatBoost(config | dict(feature_weights={2: 0})).fit(pool)
    for model, name, kind in ((ordinary, "ordinary", "OnlineCtr"),
                              (category_weight, "category", "OnlineCtr"),
                              (ctr_weight, "ctr", "FloatFeature")):
        file = tmp_path / (name + ".json")
        model.save_model(file, format="json")
        document = json.loads(file.read_text())
        assert document["oblivious_trees"][0]["splits"][0]["split_type"] == kind
    for method in ("get_leaf_values", "get_leaf_weights"):
        np.testing.assert_array_equal(getattr(ordinary, method)(), getattr(category_weight, method)())
    np.testing.assert_array_equal(ordinary.predict(pool), category_weight.predict(pool))


@pytest.mark.parametrize("boosting", ["Plain", "Ordered"])
@pytest.mark.parametrize("histories", [1, 4])
def test_compound_ctrs_do_not_alias_original_category_weights(boosting, histories, tmp_path):
    from test_native_compound_ctrs import (
        categorical_problem, check_readers_and_oracle, options as compound_options,
    )
    x, y, future, pool_options = categorical_problem()
    pool = Pool(x, y, **pool_options)
    config = compound_options(boosting=boosting, count=histories)
    ordinary = CatBoost(config).fit(pool)
    # The two original categorical slots have no one-hot candidates. Their
    # simple CTRs own separate IDs; dynamic CTRs deliberately use user weight 1
    # instead of CUDA's unrelated local-pack-index alias into slots 0 and 1.
    weighted = CatBoost(config | dict(feature_weights={0: 0, 1: 19})).fit(pool)
    check_readers_and_oracle(weighted, x, y, future, tmp_path)
    for method in ("get_leaf_values", "get_leaf_weights", "get_tree_leaf_counts"):
        np.testing.assert_array_equal(getattr(ordinary, method)(), getattr(weighted, method)())
    np.testing.assert_array_equal(ordinary.predict(future, task_type="GPU"),
                                  weighted.predict(future, task_type="GPU"))
    assert ordinary.get_evals_result() == weighted.get_evals_result()


@pytest.mark.parametrize("boosting", ["Plain", "Ordered"])
def test_weighted_simple_and_dynamic_ctr_snapshot_restores_exact_scores(boosting, tmp_path):
    from test_native_compound_ctrs import (
        categorical_problem, check_readers_and_oracle, options as compound_options, snapshot_options,
    )
    x, y, future, pool_options = categorical_problem()
    pool = Pool(x, y, **pool_options)
    config = compound_options(boosting=boosting, count=4, iterations=6,
                              feature_weights={2: .6, 3: 1.4})
    direct = CatBoost(config).fit(pool, eval_set=pool, use_best_model=False)
    saved = snapshot_options(config, tmp_path)
    partial = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(2)])
    assert partial.tree_count_ == 2
    callback = StopAfter()
    resumed = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[callback])
    assert callback.seen == [3, 4, 5, 6]
    for method in ("get_leaf_values", "get_leaf_weights", "get_tree_leaf_counts", "get_test_eval"):
        np.testing.assert_array_equal(getattr(direct, method)(), getattr(resumed, method)())
    assert direct.get_evals_result() == resumed.get_evals_result()
    check_readers_and_oracle(resumed, x, y, future, tmp_path)
    snapshot = tmp_path / "compound.snapshot"
    accepted = snapshot.read_bytes()
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|parameters.*differ"):
        CatBoost(saved | dict(feature_weights={2: .7, 3: 1.4})).fit(
            pool, eval_set=pool, use_best_model=False)
    assert snapshot.read_bytes() == accepted
