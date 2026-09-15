"""Native GPU greedy PairLogit acceptance with literal, supplied pair edges.

The private greedy suite checks the split and optimizer equations independently.
These tests cover CatBoost's Pool, grouped permutation, lifecycle and model-reader
adapters. Every fit uses GPU; CPU prediction is only a standard model reader.
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRanker, Pool

from test_native_greedy_api import POLICIES, SCORES, StopAfter, sampler
from test_native_pair_api import data, params


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_QUERY_TESTS") != "1",
    reason="requires the rebuilt native Metal greedy PairLogit adapter",
)

CTR_KINDS = ("Borders", "Buckets", "FloatTargetMeanValue", "FeatureFreq")


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def config(policy, **extra):
    result = params(
        boosting_type="Plain", grow_policy=policy, iterations=5,
        leaf_estimation_method="Newton", leaf_estimation_iterations=4,
        metric_period=1, eval_metric="PairLogit", custom_metric=["PairAccuracy"],
    )
    if policy == "Lossguide":
        result["max_leaves"] = 6
    return result | extra


def literal_data():
    x, y, groups, edges, weights = data()
    # Repeated edges retain their original masses, and zero-mass reversed edges
    # must not contribute gradients, leaf weights, or evaluation metrics.
    edges = np.concatenate((edges, edges[::7], edges[::9, ::-1]))
    weights = np.concatenate((weights, np.linspace(.2, .9, len(weights[::7]), dtype=np.float32),
                              np.zeros(len(weights[::9]), np.float32)))
    weights[0] = 0
    return x, y, groups, edges, weights


def pair_metrics(prediction, edges, weights):
    raw = np.asarray(prediction, dtype=np.float64)
    margins = raw[edges[:, 0]] - raw[edges[:, 1]]
    return {
        "PairLogit": np.average(np.logaddexp(0, -margins), weights=weights),
        "PairAccuracy": np.average(margins > 0, weights=weights),
    }


def check_metrics(model, edges, weights):
    expected = pair_metrics(model.get_test_eval(), edges, weights)
    for name, value in expected.items():
        recorded = weighted_history(model, "validation", name)[-1]
        assert recorded == pytest.approx(value, rel=4e-6, abs=4e-8)


def weighted_history(model, dataset, name):
    history = model.get_evals_result()[dataset]
    return history[name] if name in history else history[name + ":use_weights=true"]


def check_leaf_weights(model, policy, weights):
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_all_params()["grow_policy"] == policy
    counts = model.get_tree_leaf_counts()
    maximum = 4 if policy == "Region" else (6 if policy == "Lossguide" else 8)
    assert len(counts) == model.tree_count_
    assert np.all((counts >= 1) & (counts <= maximum))
    values = model.get_leaf_values()
    leaf_weights = model.get_leaf_weights()
    assert np.isfinite(values).all() and np.isfinite(leaf_weights).all()
    assert np.all(leaf_weights >= 0)
    offset = 0
    for count in counts:
        count = int(count)
        assert leaf_weights[offset:offset + count].sum() == pytest.approx(
            2 * weights.sum(dtype=float), rel=4e-6, abs=4e-6,
        )
        assert values[offset:offset + count].sum() == pytest.approx(0, abs=4e-6)
        offset += count


def check_readers(model, x, tmp_path):
    raw = model.predict(x, task_type="GPU")
    np.testing.assert_allclose(model.predict(x), raw, rtol=4e-6, atol=8e-7)
    for fmt in ("cbm", "json"):
        path = tmp_path / ("greedy-pair." + fmt)
        model.save_model(path, format=fmt)
        restored = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(restored.predict(x, task_type="GPU"), raw, rtol=4e-6, atol=8e-7)
        np.testing.assert_allclose(restored.predict(x), raw, rtol=4e-6, atol=8e-7)
        if fmt == "json":
            assert "trees" in json.loads(path.read_text())


def check_exact_model(actual, expected, x):
    assert actual.tree_count_ == expected.tree_count_
    for name in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(actual, name)(), getattr(expected, name)())
    np.testing.assert_array_equal(actual.predict(x, task_type="GPU"), expected.predict(x, task_type="GPU"))
    assert actual.get_evals_result() == expected.get_evals_result()


def snapshot_config(options, tmp_path):
    return options | dict(save_snapshot=True, snapshot_interval=0,
                          snapshot_file="greedy-pair.snapshot", allow_writing_files=True,
                          train_dir=str(tmp_path))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_native_greedy_pair_methods_scores_metrics_weights_and_readers(tmp_path, policy, score, method):
    x, y, groups, edges, weights = literal_data()
    pool = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=weights)
    options = config(policy, score_function=score, leaf_estimation_method=method)
    model = CatBoostRanker(**options).fit(pool, eval_set=pool, use_best_model=False)
    check_leaf_weights(model, policy, weights)
    check_metrics(model, edges, weights)
    history = model.get_evals_result()["learn"]["PairLogit"]
    assert history[-1] < history[0]
    np.testing.assert_allclose(model.get_test_eval(), model.predict(x, task_type="GPU"), rtol=4e-6, atol=8e-7)
    check_readers(model, x, tmp_path)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("quantized", (False, True))
@pytest.mark.parametrize("metric", ("PairLogit", "PairAccuracy"))
def test_native_greedy_pair_literal_weights_zero_edges_and_unlabeled_pools(policy, quantized, metric):
    x, y, groups, edges, weights = literal_data()
    plain = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=weights)
    unlabeled = Pool(x, group_id=groups, pairs=edges, pairs_weight=weights,
                     group_weight=np.repeat(np.linspace(.4, 2.7, 8), 6))
    unlabeled.set_weight(np.linspace(.3, 2.2, len(x)))
    positive = weights > 0
    no_zero = Pool(x, y, group_id=groups, pairs=edges[positive], pairs_weight=weights[positive])
    if quantized:
        for pool in (plain, unlabeled, no_zero):
            pool.quantize(border_count=16)
    options = config(policy, iterations=3, eval_metric=metric)
    direct = CatBoostRanker(**options).fit(plain, eval_set=plain, use_best_model=False)
    extra_weights = CatBoostRanker(**options).fit(unlabeled, eval_set=unlabeled, use_best_model=False)
    for name in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(extra_weights, name)(), getattr(direct, name)())
    np.testing.assert_array_equal(extra_weights.predict(x, task_type="GPU"), direct.predict(x, task_type="GPU"))
    # CatBoost expands custom metrics to weighted/unweighted variants when
    # object weights are present; compare the corresponding weighted history.
    for dataset in ("learn", "validation"):
        for name in ("PairLogit", "PairAccuracy"):
            assert weighted_history(extra_weights, dataset, name) == weighted_history(direct, dataset, name)
    removed = CatBoostRanker(**options).fit(no_zero, eval_set=no_zero, use_best_model=False)
    for name in ("get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_allclose(getattr(removed, name)(), getattr(direct, name)(), rtol=4e-6, atol=8e-7)
    check_metrics(extra_weights, edges, weights)
    check_leaf_weights(extra_weights, policy, weights)


@pytest.mark.parametrize("policy", POLICIES)
def test_native_greedy_pair_generated_edges_without_supplied_pairs(policy):
    x, y, groups, _, _ = data()
    pool = Pool(x, y, group_id=groups)
    model = CatBoostRanker(**config(policy)).fit(pool, eval_set=pool, use_best_model=False)
    assert model.tree_count_ == 5
    assert np.isfinite(model.predict(x, task_type="GPU")).all()
    history = model.get_evals_result()["learn"]["PairLogit"]
    assert history[-1] < history[0]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("sampling,mode", (("No", "No"), ("Bayesian", "Armijo"),
                                          ("Bernoulli", "Armijo"), ("Poisson", "AnyImprovement")))
def test_native_greedy_pair_exact_callback_snapshot_completed_and_extended_continuation(
        tmp_path, policy, sampling, mode):
    x, y, groups, edges, weights = literal_data()
    pool = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=weights)
    options = config(policy, random_strength=.4, leaf_estimation_backtracking=mode, **sampler(sampling))
    saved = snapshot_config(options, tmp_path)
    callback = StopAfter(2)
    partial = CatBoostRanker(**saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    assert (tmp_path / "greedy-pair.snapshot").is_file()
    resumed = CatBoostRanker(**saved).fit(pool, eval_set=pool, use_best_model=False)
    direct = CatBoostRanker(**options).fit(pool, eval_set=pool, use_best_model=False)
    check_exact_model(resumed, direct, x)
    completed = CatBoostRanker(**saved).fit(pool, eval_set=pool, use_best_model=False)
    check_exact_model(completed, direct, x)
    extended = CatBoostRanker(**(saved | dict(iterations=7))).fit(pool, eval_set=pool, use_best_model=False)
    longer = CatBoostRanker(**(options | dict(iterations=7))).fit(pool, eval_set=pool, use_best_model=False)
    assert extended.tree_count_ == 7
    check_exact_model(extended, longer, x)
    changed = weights.copy()
    changed[1] *= 1.5
    changed_edges = edges.copy()
    changed_edges[1] = changed_edges[1, ::-1]
    for changed_pairs, changed_weights in ((edges, changed), (changed_edges, weights)):
        altered = Pool(x, y, group_id=groups, pairs=changed_pairs, pairs_weight=changed_weights)
        for learn, evaluation in ((altered, pool), (pool, altered)):
            with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
                CatBoostRanker(**(saved | dict(iterations=7))).fit(learn, eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("mode", ("No", "AnyImprovement", "Armijo"))
def test_native_greedy_pair_initial_model_baseline_and_exact_snapshot(tmp_path, policy, mode):
    x, y, groups, edges, weights = literal_data()
    pool = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=weights)
    options = config(policy, leaf_estimation_backtracking=mode)
    initial = CatBoostRanker(**(options | dict(iterations=2))).fit(pool)
    baseline = np.linspace(-.2, .3, len(x), dtype=np.float32)
    pool.set_baseline(baseline)
    direct = CatBoostRanker(**options).fit(pool, eval_set=pool, use_best_model=False, init_model=initial)
    saved = snapshot_config(options, tmp_path)
    partial = CatBoostRanker(**saved).fit(pool, eval_set=pool, use_best_model=False,
                                        init_model=initial, callbacks=[StopAfter(2)])
    assert partial.tree_count_ == 4
    resumed = CatBoostRanker(**saved).fit(pool, eval_set=pool, use_best_model=False, init_model=initial)
    assert resumed.tree_count_ == 7
    check_exact_model(resumed, direct, x)
    np.testing.assert_allclose(resumed.get_test_eval(), resumed.predict(x, task_type="GPU") + baseline,
                               rtol=4e-6, atol=8e-7)
    check_metrics(resumed, edges, weights)
    check_readers(resumed, x, tmp_path)


def categorical_data(kind):
    rng = np.random.default_rng(34721)
    groups = np.repeat(np.arange(16), 8)
    category = rng.choice(12, len(groups), p=np.arange(1, 13) / 78)
    x = np.array([[f"category-{c}"] for c in category], dtype=object)
    y = (.125 + .75 * (category % 2)).astype(np.float32)
    edges = np.asarray([(winner, loser) for start in range(0, len(x), 8)
                        for winner in range(start, start + 8) for loser in range(start, start + 8)
                        if y[winner] > y[loser]], dtype=np.uint32)
    duplicates, zero_edges = edges[::19], edges[::23, ::-1]
    weights = np.concatenate((np.linspace(.25, 1.75, len(edges) + len(duplicates), dtype=np.float32),
                              np.zeros(len(zero_edges), np.float32)))
    edges = np.concatenate((edges, duplicates, zero_edges))
    actual = "Borders" if kind == "OneHot" else kind
    options = dict(one_hot_max_size=255 if kind == "OneHot" else 1,
                   max_ctr_complexity=1, model_size_reg=0, ctr_target_border_count=1,
                   simple_ctr=[f"{actual}:CtrBorderType=Uniform:CtrBorderCount=3:Prior=0.5"])
    pool_options = dict(cat_features=[0], group_id=groups, pairs=edges, pairs_weight=weights)
    return x, y, edges, weights, pool_options, options


def categorical_acceptance(tmp_path, policy, kind, count, sampling="No", quantized=False):
    x, y, edges, weights, pool_options, ctr_options = categorical_data(kind)
    pool = Pool(x, y, **pool_options)
    if quantized:
        pool.quantize()
    heldout = x.copy()
    heldout[::17, 0] = "unseen-category"
    evaluation = Pool(heldout, y, **pool_options)
    options = config(policy, **ctr_options, permutation_count=count, has_time=count == 1,
                     ctr_history_unit="Sample" if count == 1 else "Group",
                     random_strength=.3, leaf_estimation_backtracking="Armijo", **sampler(sampling))
    direct = CatBoostRanker().set_params(**options).fit(pool, eval_set=evaluation, use_best_model=False)
    if kind != "OneHot":
        assert direct.get_metadata()["metal_permutations"] == str(count)
    check_leaf_weights(direct, policy, weights)
    check_metrics(direct, edges, weights)
    np.testing.assert_allclose(direct.get_test_eval(), direct.predict(heldout, task_type="GPU"), rtol=4e-6, atol=8e-7)
    saved = snapshot_config(options, tmp_path)
    callback = StopAfter(2)
    partial = CatBoostRanker().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False,
                                                     callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    resumed = CatBoostRanker().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False)
    check_exact_model(resumed, direct, heldout)
    check_readers(resumed, heldout, tmp_path)
    doc = json.loads((tmp_path / "greedy-pair.json").read_text())
    split_type = "OneHotFeature" if kind == "OneHot" else "OnlineCtr"
    assert split_type in json.dumps(doc["trees"])
    changed = x.copy()
    changed[0, 0] = "changed-original-category"
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
        CatBoostRanker().set_params(**saved).fit(Pool(changed, y, **pool_options),
                                                eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("kind", ("OneHot", *CTR_KINDS))
@pytest.mark.parametrize("count,quantized", ((1, False), (4, True)))
def test_native_greedy_pair_categories_raw_quantized_permutations_and_snapshot(tmp_path, policy, kind, count, quantized):
    categorical_acceptance(tmp_path, policy, kind, count, "Bernoulli" if count == 4 else "No", quantized)


@pytest.mark.parametrize("policy,sampling,kind", (("Depthwise", "Bernoulli", "Borders"),
                                                ("Region", "Bayesian", "FloatTargetMeanValue"),
                                                ("Region", "Bernoulli", "Borders"),
                                                ("Lossguide", "Poisson", "FeatureFreq")))
def test_native_greedy_pair_seven_permutation_armijo_recovery(tmp_path, policy, sampling, kind):
    categorical_acceptance(tmp_path, policy, kind, 7, sampling, quantized=True)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("metric", ("PairLogit:use_weights=false", "PairAccuracy", "PairAccuracy:use_weights=false"))
def test_native_greedy_pair_selection_metrics_use_supplied_validation_edges(policy, metric):
    x, y, groups, edges, weights = literal_data()
    # Shared CatBoost option validation permits explicit use_weights only with
    # non-default object weights. Pair metrics still use the supplied edge mass.
    objects = np.linspace(.4, 2.1, len(x), dtype=np.float32)
    pool = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=weights, weight=objects)
    validation_edges = edges[:, ::-1].copy()
    evaluation = Pool(x, y + 500, group_id=groups, pairs=validation_edges, pairs_weight=weights, weight=objects)
    model = CatBoostRanker(**config(policy, iterations=3, eval_metric=metric)).fit(
        pool, eval_set=evaluation, use_best_model=False,
    )
    raw = np.asarray(model.get_test_eval())
    margins = raw[validation_edges[:, 0]] - raw[validation_edges[:, 1]]
    terms = margins > 0 if metric.startswith("PairAccuracy") else np.logaddexp(0, -margins)
    expected = np.average(terms, weights=None if "use_weights=false" in metric else weights)
    assert model.get_evals_result()["validation"][metric][-1] == pytest.approx(expected, rel=4e-6, abs=4e-8)


@pytest.mark.parametrize("policy", POLICIES)
def test_native_greedy_pair_best_model_and_early_stop_preserve_snapshot_cursor(tmp_path, policy):
    x = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.float32)
    y = x.ravel()
    groups = np.repeat([0, 1, 2], 2)
    edges = np.array([[1, 0], [3, 2], [5, 4]], dtype=np.uint32)
    pool = Pool(x, y, group_id=groups, pairs=edges)
    evaluation = Pool(x, y, group_id=groups, pairs=edges[:, ::-1].copy())
    options = config(policy, iterations=9, depth=1, border_count=1, learning_rate=.4)
    direct = CatBoostRanker(**options).fit(pool, eval_set=evaluation, use_best_model=True)
    saved = snapshot_config(options, tmp_path)
    callback = StopAfter(3)
    partial = CatBoostRanker(**saved).fit(pool, eval_set=evaluation, use_best_model=True, callbacks=[callback])
    assert callback.iterations == [1, 2, 3]
    assert partial.tree_count_ == 1
    resumed = CatBoostRanker(**saved).fit(pool, eval_set=evaluation, use_best_model=True)
    assert direct.best_iteration_ == resumed.best_iteration_ == 0
    assert direct.tree_count_ == resumed.tree_count_ == 1
    assert len(resumed.get_evals_result()["validation"]["PairLogit"]) == 9
    check_exact_model(resumed, direct, x)
    early = CatBoostRanker(**options).fit(pool, eval_set=evaluation, use_best_model=True, early_stopping_rounds=2)
    assert early.tree_count_ == 1 and early.best_iteration_ == 0
    assert len(early.get_evals_result()["validation"]["PairLogit"]) == 3
    np.testing.assert_array_equal(early.predict(x, task_type="GPU"), direct.predict(x, task_type="GPU"))
    check_readers(resumed, x, tmp_path)


@pytest.mark.parametrize("loss", ("RMSE", "Logloss"))
def test_native_pair_accuracy_target_exemption_preserves_scalar_constant_label_validation(loss):
    x, _, _, _, _ = data()
    options = config("Depthwise", iterations=2, loss_function=loss, eval_metric=loss, custom_metric=[])
    with pytest.raises(CatBoostError, match="(?i)all train targets are equal|target contains only one unique value"):
        CatBoost(options).fit(Pool(x, np.zeros(len(x), dtype=np.float32)))
