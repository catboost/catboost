"""Supplied-edge PairLogit lifecycle on all three Metal greedy tree policies.

No CatBoost model is fitted as an oracle. Readers use the ordinary CatBoost
model format, metrics use the supplied edges directly, and recovery is exact.
"""

import json

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker
from catboost_metal import CatBoostMetalRanker, _greedy
from catboost_metal._greedy_training import run_training
from test_native_greedy_api import POLICIES, sampler
from test_pair_lifecycle import problem as public_problem
from test_pairwise_training import problem as training_problem


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Standalone greedy PairLogit acceptance must not fit CPU CatBoost")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def data(categorical=False):
    bins, y, _, _, offsets, groups, pairs, weights = public_problem()
    x = bins.T.astype(np.float32)
    if categorical:
        x = np.column_stack((x.astype(object), np.array(["a", "b", "c"])[np.arange(len(y)) % 3]))
    return x, y, offsets, groups, pairs, weights


def config(policy, method="Newton", sampling="No", categorical=False, **extra):
    result = dict(loss_function="PairLogit", grow_policy=policy, iterations=6,
                  depth=3, learning_rate=.2, l2_leaf_reg=2, border_count=3,
                  score_function="Cosine", random_strength=.3, random_seed=22,
                  leaf_estimation_method=method, leaf_estimation_iterations=4,
                  leaf_estimation_backtracking="Armijo", **sampler(sampling))
    if policy == "Lossguide":
        result["max_leaves"] = 6
    if categorical:
        result.update(cat_features=[2], one_hot_max_size=8)
    return result | extra


def assert_same_result(actual, expected):
    assert len(actual.trees) == len(expected.trees)
    for first, second in zip(actual.trees, expected.trees):
        for field in ("nodes", "leaf_values", "leaf_weights"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
    np.testing.assert_array_equal(actual.loss, expected.loss)
    np.testing.assert_array_equal(actual.eval_predictions, expected.eval_predictions)
    assert actual.evals_result == expected.evals_result


def assert_readers(model, x, tmp_path):
    raw = model.predict(x, task_type="GPU")
    np.testing.assert_allclose(model.predict(x), raw, rtol=5e-6, atol=1e-6)
    for fmt in ("cbm", "json"):
        path = tmp_path / ("greedy-pair." + fmt)
        model.save_model(path, format=fmt)
        reader = CatBoostRanker().load_model(path, format=fmt)
        assert reader.get_all_params()["loss_function"] == "PairLogit"
        for task in ("CPU", "GPU"):
            np.testing.assert_allclose(reader.predict(x, task_type=task), raw, rtol=5e-6, atol=1e-6)
            np.testing.assert_allclose(
                reader.predict(x, task_type=task, ntree_start=1, ntree_end=3),
                model.predict(x, task_type="GPU", ntree_start=1, ntree_end=3),
                rtol=5e-6, atol=1e-6,
            )
        if fmt == "json":
            assert "trees" in json.loads(path.read_text())


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
@pytest.mark.parametrize("sampling", ("No", "Bayesian", "Bernoulli", "Poisson"))
@pytest.mark.parametrize("categorical", (False, True))
def test_supplied_pair_ranker_snapshot_metrics_and_readers(tmp_path, policy, method, sampling, categorical):
    x, y, _, groups, pairs, weights = data(categorical)
    heldout = x.copy()
    if categorical:
        heldout[::19, 2] = "unknown"
    fit = dict(group_id=groups, pairs=pairs, pairs_weight=weights,
               eval_set=(heldout, y, groups), eval_pairs=pairs,
               eval_pairs_weight=weights, use_best_model=False)
    options = config(policy, method, sampling, categorical)
    direct = CatBoostMetalRanker(**options).fit(x, **fit)
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path / "pair.npz")
    seen = []

    def callback(info):
        seen.append(info.iteration)
        return info.iteration < 2

    partial = CatBoostMetalRanker(**options).fit(x, **fit, **saved, callback=callback)
    assert partial.tree_count_ == 2 and seen == [1, 2]
    resumed = CatBoostMetalRanker(**options).fit(x, **fit, **saved)
    assert resumed._result.resumed_iterations == 2
    assert_same_result(resumed._result, direct._result)
    np.testing.assert_array_equal(resumed.predict(heldout, task_type="GPU"), direct.predict(heldout, task_type="GPU"))
    np.testing.assert_array_equal(direct.pairs_, pairs)
    np.testing.assert_array_equal(direct.pairs_weight_, weights)
    raw = direct.predict(heldout, task_type="GPU")
    expected = np.average(np.logaddexp(0, raw[pairs[:, 1]] - raw[pairs[:, 0]]), weights=weights)
    assert direct.evals_result_["validation"]["PairLogit"][-1] == pytest.approx(expected, rel=5e-6, abs=1e-6)
    for tree in direct._result.trees:
        assert tree.leaf_weights.sum(dtype=float) == pytest.approx(2 * weights.sum(dtype=float), rel=4e-6)
        assert tree.leaf_values.mean(dtype=float) == pytest.approx(0, abs=5e-8)
    assert_readers(resumed, heldout, tmp_path)


@pytest.mark.parametrize("policy", POLICIES)
def test_literal_weights_ignore_labels_object_and_group_weights(policy):
    x, y, offsets, groups, pairs, weights = data()
    options = config(policy, iterations=3, random_strength=0)
    original = CatBoostMetalRanker(**options).fit(x, group_id=groups, pairs=pairs, pairs_weight=weights)
    changed = CatBoostMetalRanker(**options).fit(
        x, y + np.linspace(-100, 100, len(y)), group_id=groups,
        sample_weight=np.linspace(.25, 7, len(y)),
        group_weight=np.repeat([2., .3, 11.], np.diff(offsets)), pairs=pairs, pairs_weight=weights,
    )
    assert_same_result(changed._result, original._result)
    assert np.count_nonzero(weights == 0) > 0
    assert (pairs[-2] == pairs[0]).all()  # Fixture retains duplicate and opposing edges.


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("count", (1, 4, 7))
@pytest.mark.parametrize("sampling", ("No", "Bayesian", "Bernoulli", "Poisson"))
def test_pair_permutation_banks_snapshot_all_arrays_and_extended_continuation(tmp_path, policy, count, sampling):
    options = training_problem(grow_policy=policy, depth=3, max_leaves=6, iterations=4,
                               random_strength=.3, random_seed=22,
                               leaf_estimation_backtracking="Armijo", **sampler(sampling))
    banks = np.stack([options["bins"].copy() for _ in range(count)])
    for index in range(1, count):
        banks[index, 0] = np.roll(banks[index, 0], 3 * index)
    options.update(permutation_bins=banks, eval_bins=options["bins"], eval_targets=options["targets"],
                   eval_group_offsets=options["group_offsets"],
                   eval_pair_winners=options["pair_winners"], eval_pair_losers=options["pair_losers"],
                   eval_pair_weights=options["pair_weights"], use_best_model=False)
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path / "resumed.npz")
    run_training(**options, **saved, callback=lambda info: info.iteration < 2)
    resumed = run_training(**options, **saved)
    assert resumed.resumed_iterations == 2
    direct = run_training(**options, **(saved | dict(snapshot_file=tmp_path / "direct.npz")))
    assert_same_result(resumed, direct)
    with np.load(tmp_path / "resumed.npz", allow_pickle=False) as actual, np.load(tmp_path / "direct.npz", allow_pickle=False) as expected:
        assert set(actual.files) == set(expected.files)
        for key in actual.files:
            assert actual[key].dtype.kind != "O"
            if key != "metadata":
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        if count > 1:
            assert actual["permutation_predictions"].shape == (count, len(options["targets"]))
    extended = run_training(**(options | dict(iterations=6)), **saved)
    fresh = run_training(**(options | dict(iterations=6)))
    assert extended.resumed_iterations == 4
    assert_same_result(extended, fresh)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("field", ("pairs", "pairs_weight", "eval_pairs", "eval_pairs_weight"))
def test_pair_snapshot_identity_rejects_changed_training_and_evaluation_edges(tmp_path, monkeypatch, policy, field):
    x, y, _, groups, pairs, weights = data()
    fit = dict(group_id=groups, pairs=pairs, pairs_weight=weights,
               eval_set=(x, y, groups), eval_pairs=pairs, eval_pairs_weight=weights,
               use_best_model=False, save_snapshot=True, snapshot_interval=0,
               snapshot_file=tmp_path / "identity.npz")
    options = config(policy, iterations=2)
    CatBoostMetalRanker(**options).fit(x, **fit)
    fit[field] = fit[field].copy()
    if field.endswith("weight"):
        fit[field][1] *= 2
    else:
        fit[field][1] = fit[field][1, ::-1]
    monkeypatch.setattr(_greedy, "build_library", lambda: pytest.fail("Changed snapshot identity reached GPU loading"))
    with pytest.raises(ValueError, match="(?i)snapshot.*(differ|match)|(differ|match).*snapshot"):
        CatBoostMetalRanker(**options).fit(x, **fit)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("metric", ("PairLogit:use_weights=false", "PairAccuracy", "PairAccuracy:use_weights=false"))
def test_selection_metrics_evaluate_literal_validation_edges(policy, metric):
    x, y, _, groups, pairs, weights = data()
    validation_pairs = pairs[:, ::-1].copy()
    model = CatBoostMetalRanker(**config(policy, iterations=3, eval_metric=metric)).fit(
        x, group_id=groups, pairs=pairs, pairs_weight=weights,
        eval_set=(x, y + 500, groups), eval_pairs=validation_pairs,
        eval_pairs_weight=weights, use_best_model=False,
    )
    raw = model.predict(x, task_type="GPU")
    difference = raw[validation_pairs[:, 0]] - raw[validation_pairs[:, 1]]
    terms = difference > 0 if metric.startswith("PairAccuracy") else np.logaddexp(0, -difference)
    expected = np.average(terms, weights=np.ones(len(weights)) if "use_weights=false" in metric else weights)
    assert model.evals_result_["validation"][metric][-1] == pytest.approx(expected, abs=2e-6)
    assert model.training_stats_["metric_maximized"] == metric.startswith("PairAccuracy")


@pytest.mark.parametrize("policy", POLICIES)
def test_best_model_trims_predictions_but_snapshot_retains_full_cursor(tmp_path, policy):
    x = np.array([[0], [1], [0], [1], [0], [1]], np.float32)
    groups = np.repeat([0, 1, 2], 2)
    pairs = np.array([[1, 0], [3, 2], [5, 4]], np.uint32)
    options = config(policy, iterations=9, depth=1, border_count=1, learning_rate=.4, random_strength=0)
    fit = dict(group_id=groups, pairs=pairs, eval_set=(x, np.zeros(len(x)), groups),
               eval_pairs=pairs[:, ::-1].copy(), use_best_model=True)
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path / "best.npz")
    direct = CatBoostMetalRanker(**options).fit(x, **fit)
    CatBoostMetalRanker(**options).fit(x, **fit, **saved, callback=lambda info: info.iteration < 3)
    resumed = CatBoostMetalRanker(**options).fit(x, **fit, **saved)
    assert direct.best_iteration_ == 0 and direct.tree_count_ == 1
    assert resumed._result.resumed_iterations == 3
    assert resumed.training_stats_["iterations_trained"] == 9
    assert_same_result(resumed._result, direct._result)
    assert len(resumed.evals_result_["validation"]["PairLogit"]) == 9
    with np.load(tmp_path / "best.npz", allow_pickle=False) as snapshot:
        assert len(json.loads(str(snapshot["metadata"].item()))["history"]["validation"]["PairLogit"]) == 9
    early = CatBoostMetalRanker(**options).fit(x, **fit, early_stopping_rounds=2)
    assert early.tree_count_ == 1 and early.training_stats_["iterations_trained"] == 3
    np.testing.assert_array_equal(early.predict(x, task_type="GPU"), direct.predict(x, task_type="GPU"))
