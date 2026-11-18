"""Standalone Ordered query objectives, continuation and ordinary model readers."""

import json
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker
from catboost_metal import CatBoostMetalRanker, _ordered
from catboost_metal._training import _shared_metric


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple GPU required")

OBJECTIVES = ("QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank")
SIZES = np.array([3, 5, 4, 2, 7, 1, 6, 4, 8, 3, 5, 2], np.uint32)


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Standalone Ordered acceptance must not fit CatBoost on the CPU")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def problem(objective, categorical=False):
    rows = int(SIZES.sum())
    rng = np.random.default_rng(1541)
    x = rng.normal(size=(rows, 3)).astype(np.float32)
    y = np.float32(1.3 * x[:, 0] - .4 * x[:, 1])
    if objective == "QuerySoftMax":
        y = np.exp(y / 3).astype(np.float32)
    elif objective == "YetiRank":
        y = np.float32(np.clip(.5 + y / 5, 0, 1))
    offsets = np.r_[np.uint32(0), np.cumsum(SIZES, dtype=np.uint32)]
    groups = np.repeat(np.arange(len(SIZES), dtype=np.uint64) + 2**34, SIZES)
    weights = rng.uniform(.3, 1.7, rows).astype(np.float32)
    weights[::17] = 0
    group_weights = np.repeat(np.linspace(.6, 1.4, len(SIZES), dtype=np.float32), SIZES)
    subgroups = np.arange(rows, dtype=np.uint32) % 3
    pairs = np.array([(int(a), int(b - 1)) for a, b in zip(offsets[:-1], offsets[1:]) if b - a > 1], np.uint32)
    pair_weights = np.linspace(.2, 1.6, len(pairs), dtype=np.float32)
    if categorical:
        x = np.column_stack((x.astype(object), np.array(["a", "b", "c"])[np.arange(rows) % 3]))
    return x, y, groups, weights, group_weights, subgroups, pairs, pair_weights, offsets


def options(objective, count=4, categorical=False, **extra):
    description = ("QuerySoftMax:beta=0.7;lambda=0.03" if objective == "QuerySoftMax" else
                   "YetiRank:permutations=5;decay=0.8" if objective == "YetiRank" else objective)
    result = dict(loss_function=description, boosting_type="Ordered", permutation_count=count,
                  iterations=4, depth=3, border_count=8, learning_rate=.17,
                  l2_leaf_reg=.3 if objective == "YetiRank" else 2., score_function="Cosine",
                  leaf_estimation_method="Newton", leaf_estimation_iterations=3,
                  leaf_estimation_backtracking="No", random_seed=51, random_strength=.4,
                  min_fold_size=2, fold_len_multiplier=1.7)
    if categorical:
        result.update(cat_features=[3], one_hot_max_size=5)
    return result | extra


def fit_arguments(data, objective, heldout=None):
    x, y, groups, weights, group_weights, subgroups, pairs, pair_weights, _ = data
    result = dict(group_id=groups, subgroup_id=subgroups)
    if objective == "PairLogit":
        result.update(pairs=pairs, pairs_weight=pair_weights)
    else:
        result.update(sample_weight=weights, group_weight=group_weights)
    if heldout is not None:
        if objective == "PairLogit":
            result.update(eval_set=(heldout, None, groups), eval_pairs=pairs,
                          eval_pairs_weight=pair_weights)
        else:
            result.update(eval_set=(heldout, y, groups, weights, group_weights, subgroups))
        result["use_best_model"] = False
    return result


def check_same(actual, expected):
    for field in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(actual._result, field), getattr(expected._result, field))
    assert actual.get_evals_result() == expected.get_evals_result()


def check_readers(model, x, path):
    expected = model.predict(x, task_type="GPU")
    np.testing.assert_allclose(model.predict(x), expected, rtol=6e-6, atol=1e-6)
    for kind in ("cbm", "json"):
        output = path / ("ordered-query." + kind)
        model.save_model(output, format=kind)
        loaded = CatBoostRanker().load_model(output, format=kind)
        assert loaded.get_all_params()["boosting_type"] == "Ordered"
        assert loaded.get_all_params()["data_partition"] == "FeatureParallel"
        for task in ("CPU", "GPU"):
            np.testing.assert_allclose(loaded.predict(x, task_type=task), expected, rtol=6e-6, atol=1e-6)
            np.testing.assert_allclose(loaded.predict(x, task_type=task, ntree_start=1, ntree_end=3),
                                       model.predict(x, task_type="GPU", ntree_start=1, ntree_end=3),
                                       rtol=6e-6, atol=1e-6)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("categorical", (False, True))
def test_query_prefix_snapshots_metrics_and_readers(tmp_path, objective, count, categorical):
    data = problem(objective, categorical)
    x, y, groups, weights, group_weights, subgroups, pairs, pair_weights, offsets = data
    heldout = x.copy()
    if categorical:
        heldout[::13, 3] = "unknown"
    args = fit_arguments(data, objective, heldout)
    labels = None if objective == "PairLogit" else y
    config = options(objective, count, categorical, bootstrap_type="Bayesian", bagging_temperature=.6)
    full = CatBoostMetalRanker(**config).fit(x, labels, **args)
    saved = dict(save_snapshot=True, snapshot_file=tmp_path / "prefix.npz", snapshot_interval=0)
    seen = []
    def callback(info):
        seen.append(info.iteration)
        return info.iteration < 2
    partial = CatBoostMetalRanker(**config).fit(x, labels, **args, **saved, callback=callback)
    assert partial.tree_count_ == 2 and seen == [1, 2]
    resumed = CatBoostMetalRanker(**config).fit(x, labels, **args, **saved)
    assert resumed.training_stats_["resumed_iterations"] == 2
    check_same(resumed, full)
    with np.load(saved["snapshot_file"], allow_pickle=False) as archive:
        assert archive["ordered_selection_rng_words"].shape == (312,)
        assert archive["ordered_descriptors"].shape[1] == 4
    metric = next(iter(full.get_evals_result()["validation"]))
    mass = None if objective == "PairLogit" else weights * group_weights
    edges = (pairs[:, 0], pairs[:, 1], pair_weights) if objective == "PairLogit" else None
    expected = _shared_metric(metric, full.predict(heldout, task_type="GPU"), y, mass, offsets,
                              pairs=edges, subgroup_hashes=subgroups)
    assert full.get_evals_result()["validation"][metric][-1] == pytest.approx(expected, rel=7e-6, abs=1e-6)
    check_readers(resumed, heldout, tmp_path)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("bootstrap", ("No", "Bayesian", "Bernoulli", "Poisson", "MVS"))
def test_ordered_query_samplers_have_finite_complete_forests(objective, bootstrap):
    data = problem(objective)
    settings = dict(bootstrap_type=bootstrap, iterations=2)
    if bootstrap == "Bayesian":
        settings["bagging_temperature"] = .7
    elif bootstrap != "No":
        settings["subsample"] = .7
    if bootstrap == "MVS":
        settings["mvs_reg"] = .8
    model = CatBoostMetalRanker(**options(objective, **settings)).fit(
        data[0], None if objective == "PairLogit" else data[1], **fit_arguments(data, objective))
    assert model.tree_count_ == 2
    assert np.isfinite(model.training_predictions_).all()
    assert np.isfinite(model.loss_history_).all()
    assert model.training_stats_["boosting_type"] == "Ordered"


@pytest.mark.parametrize("objective", OBJECTIVES[:3])
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
@pytest.mark.parametrize("backtracking", ("No", "AnyImprovement", "Armijo"))
def test_query_leaf_modes_are_forwarded(objective, method, backtracking):
    data = problem(objective)
    model = CatBoostMetalRanker(**options(objective, iterations=2, leaf_estimation_method=method,
        leaf_estimation_backtracking=backtracking, score_function="NewtonCosine")).fit(
            data[0], None if objective == "PairLogit" else data[1], **fit_arguments(data, objective))
    assert model.tree_count_ == 2 and np.isfinite(model.training_predictions_).all()


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_completed_snapshot_can_be_read_without_reopening_gpu(tmp_path, monkeypatch, objective):
    data = problem(objective)
    settings = options(objective, count=1, iterations=3)
    saved = dict(save_snapshot=True, snapshot_file=tmp_path / "complete.npz", snapshot_interval=0)
    args = fit_arguments(data, objective, data[0])
    labels = None if objective == "PairLogit" else data[1]
    full = CatBoostMetalRanker(**settings).fit(data[0], labels, **args, **saved)
    def forbidden(*args, **kwargs):
        pytest.fail("Completed snapshot must not reopen the Ordered training session")
    monkeypatch.setattr(_ordered, "Session", forbidden)
    restored = CatBoostMetalRanker(**settings).fit(data[0], labels, **args, **saved)
    check_same(restored, full)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_early_stop_retains_full_prefix_state_and_recovers(tmp_path, objective):
    data = problem(objective)
    labels = np.float32(np.clip(.5 + data[1] / 5, 0, 1))
    args = fit_arguments(data, objective, data[0])
    args.update(eval_set=(data[0], labels, data[2]), early_stopping_rounds=2, use_best_model=True)
    settings = options(objective, iterations=8, depth=0, eval_metric="NDCG")
    saved = dict(save_snapshot=True, snapshot_file=tmp_path / "early.npz", snapshot_interval=0)
    full = CatBoostMetalRanker(**settings).fit(data[0], labels, **args, **saved)
    assert full.tree_count_ == 1
    assert full.training_stats_["stop_reason"] == "early_stopping"
    with np.load(saved["snapshot_file"], allow_pickle=False) as archive:
        assert len(archive["depths"]) == 3
        assert json.loads(archive["metadata"].item())["ordered_state"]["iteration_offset"] == 3
    restored = CatBoostMetalRanker(**settings).fit(data[0], labels, **args, **saved)
    check_same(restored, full)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_changed_query_partition_rejects_snapshot_before_session(tmp_path, monkeypatch, objective):
    data = problem(objective)
    settings = options(objective)
    saved = dict(save_snapshot=True, snapshot_file=tmp_path / "identity.npz", snapshot_interval=0)
    args = fit_arguments(data, objective)
    labels = None if objective == "PairLogit" else data[1]
    CatBoostMetalRanker(**settings).fit(data[0], labels, **args, **saved, callback=lambda info: False)
    groups = data[2].copy()
    # Merge the first two queries. Existing pairs stay within the merged query.
    groups[:int(SIZES[:2].sum())] = groups[0]
    changed = args | {"group_id": groups}
    if objective != "PairLogit":
        changed["group_weight"] = np.ones(len(groups), np.float32)
    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched snapshot must fail before constructing Ordered session")
    monkeypatch.setattr(_ordered, "Session", forbidden)
    with pytest.raises(ValueError, match="Snapshot does not match"):
        CatBoostMetalRanker(**settings).fit(data[0], labels, **changed, **saved)


@pytest.mark.parametrize("settings", (
    dict(loss_function="PairLogitPairwise"), dict(loss_function="YetiRankPairwise"),
    dict(loss_function="QueryCrossEntropy"), dict(loss_function="YetiRank", leaf_estimation_method="Gradient"),
    dict(loss_function="YetiRank", leaf_estimation_backtracking="Armijo"),
    dict(loss_function="QueryRMSE", grow_policy="Depthwise"),
    dict(loss_function="QueryRMSE", score_function="L2"),
))
def test_unsupported_ordered_query_modes_fail_before_training(settings):
    with pytest.raises(ValueError):
        CatBoostMetalRanker(boosting_type="Ordered", **settings)
