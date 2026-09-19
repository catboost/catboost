"""Classic YetiRank native greedy acceptance; every fit executes on Metal.

Opt in with CATBOOST_NATIVE_METAL_QUERY_TESTS=1 and a rebuilt native package.
Per-query shared metric evaluation is an independent host calculation, never
a CPU fit. Runtime tests separately cover generated targets and split equations.
"""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRanker, Pool
from catboost.utils import eval_metric

from test_native_greedy_api import POLICIES, SCORES, StopAfter, sampler
from test_native_ranking_ctr_p1 import problem as ctr_problem
from test_yeti_rank_lifecycle import data


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_QUERY_TESTS") != "1",
    reason="requires the rebuilt native Metal greedy YetiRank adapter",
)
CTR_KINDS = ("Borders", "Buckets", "FloatTargetMeanValue", "FeatureFreq")
METRICS = ("PFound:top=5;decay=0.7", "NDCG:top=5;type=Exp")


@pytest.fixture(autouse=True)
def require_gpu_for_every_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def config(policy, **extra):
    result = dict(
        task_type="GPU", loss_function="YetiRank:permutations=7;decay=0.85;mode=Classic",
        boosting_type="Plain", grow_policy=policy, iterations=5, depth=3,
        learning_rate=.13, random_seed=817, bootstrap_type="No", random_strength=0,
        leaf_estimation_method="Newton", leaf_estimation_iterations=3,
        leaf_estimation_backtracking="No", score_function="NewtonL2",
        has_time=True, permutation_count=1, metric_period=1,
        eval_metric=METRICS[0], custom_metric=[METRICS[1]],
        verbose=False, allow_writing_files=False,
    )
    if policy == "Lossguide":
        result["max_leaves"] = 6
    return result | extra


def ranker(options):
    # permutation_count is an accepted native option exposed via set_params.
    return CatBoostRanker().set_params(**options)


def numeric_data():
    x, y, groups, objects = data()
    objects = objects.copy()
    objects[::17] = 0
    group_weights = np.linspace(.4, 2.3, len(np.unique(groups)), dtype=np.float32)[groups]
    combined = np.float32(objects * group_weights)
    separate = Pool(x, y, group_id=groups, group_weight=group_weights)
    separate.set_weight(objects)
    return x, y, groups, objects, group_weights, combined, separate


def metric_history(model, metric, dataset="validation"):
    base, _, suffix = metric.partition(":")
    wanted = dict(item.split("=", 1) for item in suffix.split(";") if item)
    for name, values in model.get_evals_result()[dataset].items():
        actual_base, _, actual_suffix = name.partition(":")
        actual = dict(item.split("=", 1) for item in actual_suffix.split(";") if item)
        if actual_base == base and all(actual.get(key) == value for key, value in wanted.items()):
            if "use_weights" not in wanted and actual.get("use_weights", "true") != "true":
                continue
            return values
    raise AssertionError(f"Missing metric {metric}: {model.get_evals_result()[dataset]}")


def independent_metric(raw, labels, groups, weights, metric):
    offsets = np.r_[np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]]), len(groups)]
    values = [eval_metric(labels[a:b], raw[a:b], metric,
                          group_id=np.zeros(b - a, np.uint32), thread_count=1)[0]
              for a, b in zip(offsets[:-1], offsets[1:])]
    # CUDA's ranking metric controller uses the first original effective
    # document weight as query mass. Shared evaluation supplies tie semantics.
    mass = None if "use_weights=false" in metric else weights[offsets[:-1]]
    return float(np.average(values, weights=mass))


def check_metrics(model, labels, groups, weights, metrics=METRICS, index=-1):
    raw = np.asarray(model.get_test_eval(), dtype=np.float64)
    for metric in metrics:
        expected = independent_metric(raw, labels, groups, weights, metric)
        assert metric_history(model, metric)[index] == pytest.approx(expected, abs=2e-8)


def check_weights(model, policy, original_weights):
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_all_params()["grow_policy"] == policy
    assert model.get_all_params()["leaf_estimation_method"] == "Newton"
    counts = model.get_tree_leaf_counts()
    maximum = 4 if policy == "Region" else (6 if policy == "Lossguide" else 8)
    assert len(counts) == model.tree_count_
    assert np.all((counts >= 1) & (counts <= maximum))
    values, weights = model.get_leaf_values(), model.get_leaf_weights()
    assert np.isfinite(values).all() and np.isfinite(weights).all()
    assert np.all(weights >= 0)
    offset = 0
    for count in counts:
        count = int(count)
        assert weights[offset:offset + count].sum() == pytest.approx(
            original_weights.sum(dtype=float), rel=4e-6, abs=4e-6,
        )
        offset += count


def check_exact(actual, expected, x):
    assert actual.tree_count_ == expected.tree_count_
    for name in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(actual, name)(), getattr(expected, name)())
    np.testing.assert_array_equal(actual.predict(x, task_type="GPU"), expected.predict(x, task_type="GPU"))
    assert actual.get_evals_result() == expected.get_evals_result()
    assert actual.best_iteration_ == expected.best_iteration_


def check_readers(model, x, tmp_path):
    raw = model.predict(x, task_type="GPU")
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(model.predict(x), raw, rtol=4e-6, atol=8e-7)
    document = None
    for fmt in ("cbm", "json"):
        path = tmp_path / ("greedy-yeti." + fmt)
        model.save_model(path, format=fmt)
        restored = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(restored.predict(x, task_type="GPU"), raw, rtol=4e-6, atol=8e-7)
        np.testing.assert_allclose(restored.predict(x), raw, rtol=4e-6, atol=8e-7)
        np.testing.assert_array_equal(restored.get_tree_leaf_counts(), model.get_tree_leaf_counts())
        if fmt == "json":
            document = json.loads(path.read_text())
            assert "trees" in document and "oblivious_trees" not in document
            assert len(document["trees"]) == model.tree_count_
    return document


def saved_config(options, tmp_path):
    return options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="greedy-yeti.snapshot",
                          allow_writing_files=True, train_dir=str(tmp_path))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("leaf_iterations", (1, 3))
def test_native_greedy_yeti_scores_newton_original_weights_metrics_and_readers(
        tmp_path, policy, score, leaf_iterations):
    x, y, groups, _, _, weights, pool = numeric_data()
    options = config(policy, score_function=score, leaf_estimation_iterations=leaf_iterations)
    model = ranker(options).fit(pool, eval_set=pool, use_best_model=False)
    assert model.tree_count_ == options["iterations"]
    assert model.get_all_params()["score_function"] == score
    assert model.get_all_params()["leaf_estimation_iterations"] == leaf_iterations
    assert max(model.get_tree_leaf_counts()) > 1
    assert np.ptp(model.predict(x, task_type="GPU")) > 0
    check_weights(model, policy, weights)
    check_metrics(model, y, groups, weights)
    np.testing.assert_allclose(model.get_test_eval(), model.predict(x, task_type="GPU"), rtol=4e-6, atol=8e-7)
    check_readers(model, x, tmp_path)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("sampling", ("No", "Bayesian", "Bernoulli", "Poisson"))
def test_native_greedy_yeti_seeded_sampling_combined_weights_and_full_snapshot_lifecycle(tmp_path, policy, sampling):
    x, y, groups, _, _, weights, pool = numeric_data()
    options = config(policy, random_strength=.4, **sampler(sampling))
    direct = ranker(options).fit(pool, eval_set=pool, use_best_model=False)
    combined = Pool(x, y, group_id=groups, weight=weights)
    repeated = ranker(options).fit(combined, eval_set=combined, use_best_model=False)
    check_exact(repeated, direct, x)
    check_weights(direct, policy, weights)
    saved = saved_config(options, tmp_path)
    callback = StopAfter(2)
    partial = ranker(saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    assert (tmp_path / "greedy-yeti.snapshot").is_file()
    restored = ranker(saved).fit(pool, eval_set=pool, use_best_model=False)
    check_exact(restored, direct, x)
    completed = ranker(saved).fit(pool, eval_set=pool, use_best_model=False)
    check_exact(completed, direct, x)
    extended = ranker(saved | dict(iterations=7)).fit(pool, eval_set=pool, use_best_model=False)
    longer = ranker(options | dict(iterations=7)).fit(pool, eval_set=pool, use_best_model=False)
    assert extended.tree_count_ == 7
    check_exact(extended, longer, x)
    check_metrics(extended, y, groups, weights)


@pytest.mark.parametrize("policy", POLICIES)
def test_native_greedy_yeti_baseline_initial_model_and_exact_snapshot(tmp_path, policy):
    x, y, groups, _, _, weights, pool = numeric_data()
    options = config(policy, random_strength=.4, **sampler("Bernoulli"))
    initial = ranker(options | dict(iterations=2)).fit(pool)
    baseline = np.linspace(-.2, .3, len(x), dtype=np.float32)
    pool.set_baseline(baseline)
    fit = dict(eval_set=pool, use_best_model=False, init_model=initial)
    direct = ranker(options).fit(pool, **fit)
    saved = saved_config(options, tmp_path)
    callback = StopAfter(2)
    partial = ranker(saved).fit(pool, callbacks=[callback], **fit)
    assert partial.tree_count_ == 4 and callback.iterations == [1, 2]
    restored = ranker(saved).fit(pool, **fit)
    assert restored.tree_count_ == 7
    check_exact(restored, direct, x)
    np.testing.assert_allclose(restored.get_test_eval(), restored.predict(x, task_type="GPU") + baseline,
                               rtol=4e-6, atol=8e-7)
    check_metrics(restored, y, groups, weights)
    check_readers(restored, x, tmp_path)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("kind", ("OneHot", *CTR_KINDS))
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("quantized", (False, True))
def test_native_greedy_yeti_selected_categories_permutations_quantization_and_exact_recovery(
        tmp_path, policy, kind, count, quantized):
    history = "Group" if quantized else "Sample"
    x, y, groups, pool_options, ctr_options, _ = ctr_problem(
        "YetiRank", "Newton", "Borders" if kind == "OneHot" else kind, history=history,
    )
    options = ctr_options | config(policy, iterations=4, permutation_count=count, has_time=count == 1,
                                   random_strength=.3, **sampler("Bernoulli" if count == 4 else "No"))
    options.update(ctr_history_unit=history, one_hot_max_size=255 if kind == "OneHot" else 1)
    pool = Pool(x, y, **pool_options)
    if quantized:
        pool.quantize()
    heldout = x.copy()
    heldout[::17, 0] = "unseen-category"
    evaluation = Pool(heldout, y, **pool_options)
    fit = dict(eval_set=evaluation, use_best_model=False)
    direct = ranker(options).fit(pool, **fit)
    if kind != "OneHot":
        assert direct.get_metadata()["metal_permutations"] == str(count)
    check_weights(direct, policy, pool_options["weight"])
    check_metrics(direct, y, groups, pool_options["weight"])
    np.testing.assert_allclose(direct.get_test_eval(), direct.predict(heldout, task_type="GPU"),
                               rtol=4e-6, atol=8e-7)
    saved = saved_config(options, tmp_path)
    callback = StopAfter(2)
    partial = ranker(saved).fit(pool, callbacks=[callback], **fit)
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    restored = ranker(saved).fit(pool, **fit)
    check_exact(restored, direct, heldout)
    completed = ranker(saved).fit(pool, **fit)
    check_exact(completed, direct, heldout)
    extended = ranker(saved | dict(iterations=6)).fit(pool, **fit)
    longer = ranker(options | dict(iterations=6)).fit(pool, **fit)
    check_exact(extended, longer, heldout)
    document = check_readers(extended, heldout, tmp_path)
    split_type = "OneHotFeature" if kind == "OneHot" else "OnlineCtr"
    assert split_type in json.dumps(document["trees"])
    assert split_type in json.dumps(document["trees"][options["iterations"]:])
    if kind != "OneHot":
        assert {ctr["ctr_type"] for ctr in document["features_info"]["ctrs"]} == {kind}
    changed = x.copy()
    changed[0, 0] = "changed-original-category"
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
        ranker(saved | dict(iterations=6)).fit(Pool(changed, y, **pool_options), **fit)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("weighted", (False, True))
def test_native_greedy_yeti_relevance_labels_weighted_selection_metrics_and_best_model(
        tmp_path, policy, metric, weighted):
    x, y, groups, _, _, weights, pool = numeric_data()
    metric += ";use_weights=" + str(weighted).lower()
    # The validation labels deliberately differ from learn labels; metrics
    # must use the supplied heldout relevance and the original query mass.
    validation_y = np.float32(1 - y)
    evaluation = Pool(x, validation_y, group_id=groups, weight=weights)
    model = ranker(config(policy, eval_metric=metric, custom_metric=[])).fit(
        pool, eval_set=evaluation, use_best_model=True,
    )
    history = metric_history(model, metric)
    assert model.best_iteration_ == int(np.argmax(history))
    assert model.tree_count_ == model.best_iteration_ + 1
    check_metrics(model, validation_y, groups, weights, metrics=[metric], index=model.best_iteration_)
    check_readers(model, x, tmp_path)


@pytest.mark.parametrize("policy", POLICIES)
def test_native_greedy_yeti_early_stopped_snapshot_retains_terminal_cursor(tmp_path, policy):
    x = np.array([[0], [1]] * 4, dtype=np.float32)
    groups = np.repeat(np.arange(4), 2)
    pool = Pool(x, x.ravel(), group_id=groups)
    evaluation = Pool(x, 1 - x.ravel(), group_id=groups)
    options = config(policy, iterations=9, depth=1, border_count=1, eval_metric="PFound", custom_metric=[])
    fit = dict(eval_set=evaluation, use_best_model=True, early_stopping_rounds=2)
    direct = ranker(options).fit(pool, **fit)
    assert direct.best_iteration_ == 0 and direct.tree_count_ == 1
    assert len(metric_history(direct, "PFound")) == 3
    saved = saved_config(options, tmp_path)
    callback = StopAfter(2)
    ranker(saved).fit(pool, callbacks=[callback], **fit)
    assert callback.iterations == [1, 2]
    restored = ranker(saved).fit(pool, **fit)
    check_exact(restored, direct, x)
    completed = ranker(saved).fit(pool, **fit)
    check_exact(completed, direct, x)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("change", ("label", "weight", "groups"))
def test_native_greedy_yeti_snapshot_rejects_changed_learn_and_validation_data(tmp_path, policy, change):
    x, y, groups, _, _, weights, _ = numeric_data()
    pool = Pool(x, y, group_id=groups, weight=weights)
    saved = saved_config(config(policy), tmp_path)
    ranker(saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(2)])
    changed_y, changed_weights, changed_groups = y.copy(), weights.copy(), groups.copy()
    if change == "label":
        changed_y[1] = 1 - changed_y[1]
    elif change == "weight":
        changed_weights[1] *= 1.5
    else:
        changed_groups[9] = changed_groups[10]
    altered = Pool(x, changed_y, group_id=changed_groups, weight=changed_weights)
    for learn, evaluation in ((altered, pool), (pool, altered)):
        with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
            ranker(saved).fit(learn, eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("extra", (
    dict(loss_function="YetiRank:mode=NDCG"),
    dict(loss_function="YetiRank:permutations=0"),
    dict(loss_function="YetiRank:decay=-1"),
    dict(bootstrap_type="MVS", subsample=.8),
    dict(leaf_estimation_method="Gradient"),
    dict(leaf_estimation_backtracking="AnyImprovement"),
    dict(leaf_estimation_backtracking="Armijo"),
    dict(boosting_type="Ordered"),
    dict(loss_function="YetiRankPairwise"),
    dict(loss_function="PairLogitPairwise"),
    dict(loss_function="QueryCrossEntropy"),
))
def test_native_greedy_yeti_unsupported_modes_and_full_matrix_objectives_rejected(policy, extra):
    x, y, groups, _ = data()
    with pytest.raises(CatBoostError):
        ranker(config(policy, **extra)).fit(Pool(x, y, group_id=groups))
