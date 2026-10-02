"""Classic YetiRank's standalone greedy lifecycle on the Apple GPU.

CPU CatBoost is used only to read exported models and evaluate ranking
metrics. No fitted CPU model serves as a training oracle.
"""

import json
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker
from catboost.utils import eval_metric
from catboost_metal import CatBoostMetalRanker, _greedy

from test_native_greedy_api import POLICIES, sampler
from test_subgroup_metadata import independent_pfound, problem as subgroup_problem
from test_yeti_rank_lifecycle import data


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple GPU required",
)


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Standalone greedy YetiRank acceptance must not fit CPU CatBoost")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def config(policy, **extra):
    options = dict(
        loss_function="YetiRank:permutations=7;decay=0.8;mode=Classic",
        grow_policy=policy, iterations=5, depth=3, border_count=8,
        learning_rate=.13, score_function="NewtonL2", random_seed=791,
        random_strength=.4, bootstrap_type="No", leaf_estimation_method="Newton",
        leaf_estimation_iterations=3, leaf_estimation_backtracking="No",
    )
    if policy == "Lossguide":
        options["max_leaves"] = 6
    return options | extra


def assert_same_result(actual, expected):
    assert actual.tree_count_ == expected.tree_count_
    for first, second in zip(actual._result.trees, expected._result.trees):
        for field in ("nodes", "leaf_values", "leaf_weights"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    for field in ("predictions", "loss", "eval_predictions"):
        np.testing.assert_array_equal(getattr(actual._result, field), getattr(expected._result, field))
    assert actual.evals_result_ == expected.evals_result_
    assert actual.best_iteration_ == expected.best_iteration_
    assert actual.best_score_ == expected.best_score_
    assert actual.training_stats_["yeti_rng"] == expected.training_stats_["yeti_rng"]


def assert_readers(model, x, tmp_path):
    raw = model.predict(x, task_type="GPU")
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(model.predict(x), raw, rtol=5e-6, atol=1e-6)
    for fmt in ("cbm", "json"):
        path = tmp_path / ("greedy-yeti." + fmt)
        model.save_model(path, format=fmt)
        reader = CatBoostRanker().load_model(path, format=fmt)
        params = reader.get_all_params()
        assert params["loss_function"].partition(":")[0] == "YetiRank"
        assert params["grow_policy"] == model.grow_policy
        np.testing.assert_array_equal(reader.get_tree_leaf_counts(),
                                      [len(tree.leaf_values) for tree in model._result.trees])
        for task in ("CPU", "GPU"):
            np.testing.assert_allclose(reader.predict(x, task_type=task), raw, rtol=5e-6, atol=1e-6)
            np.testing.assert_allclose(
                reader.predict(x, task_type=task, ntree_start=1, ntree_end=3),
                model.predict(x, task_type="GPU", ntree_start=1, ntree_end=3),
                rtol=5e-6, atol=1e-6,
            )
        if fmt == "json":
            document = json.loads(path.read_text())
            assert "trees" in document and "oblivious_trees" not in document
            assert len(document["trees"]) == model.tree_count_


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("sampling,categorical", [
    ("No", False), ("Bayesian", True), ("Bernoulli", False), ("Poisson", True),
])
def test_seeded_snapshot_callback_exact_resume_and_model_readers(tmp_path, policy, sampling, categorical):
    x, y, groups, weights = data()
    options = config(policy, **sampler(sampling))
    if categorical:
        codes = np.arange(len(y)) % 3
        x = np.column_stack((x.astype(object), np.array(["a", "b", "c"])[codes]))
        y = np.float32(.6 * y + .2 * codes)
        options.update(cat_features=[4], one_hot_max_size=8)
    heldout = x[::-1].copy()
    if categorical:
        heldout[::19, 4] = "unknown"
    fit = dict(group_id=groups, sample_weight=weights,
               eval_set=(heldout, y[::-1], groups[::-1], weights[::-1]), use_best_model=False)
    direct = CatBoostMetalRanker(**options).fit(x, y, **fit)
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path / "yeti.npz")
    seen = []

    def callback(info):
        seen.append(info.iteration)
        assert len(info.metrics["learn"]["PFound"]) == info.iteration
        return info.iteration < 2

    partial = CatBoostMetalRanker(**options).fit(x, y, **fit, **saved, callback=callback)
    assert partial.tree_count_ == 2 and seen == [1, 2]
    resumed = CatBoostMetalRanker(**options).fit(x, y, **fit, **saved)
    assert resumed._result.resumed_iterations == 2
    assert_same_result(resumed, direct)
    np.testing.assert_array_equal(resumed.predict(heldout, task_type="GPU"),
                                  direct.predict(heldout, task_type="GPU"))
    completed = CatBoostMetalRanker(**options).fit(x, y, **fit, **saved)
    assert completed._result.resumed_iterations == options["iterations"]
    assert_same_result(completed, direct)
    extended_options = options | dict(iterations=7)
    extended = CatBoostMetalRanker(**extended_options).fit(x, y, **fit, **saved)
    fresh = CatBoostMetalRanker(**extended_options).fit(x, y, **fit)
    assert extended._result.resumed_iterations == options["iterations"]
    assert_same_result(extended, fresh)
    assert extended.training_stats_["yeti_rng"]["completed_iterations"] == 7
    assert extended.training_stats_["yeti_rng"]["learner"] == "greedy_v1"
    for tree in extended._result.trees:
        assert np.isfinite(tree.leaf_values).all()
        assert (tree.leaf_weights >= 0).all()
        assert tree.leaf_weights.sum(dtype=float) == pytest.approx(weights.sum(dtype=float), rel=4e-6)
    assert_readers(extended, heldout, tmp_path)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("metric", (
    "PFound:top=4;decay=0.7;use_weights=true",
    "PFound:top=4;decay=0.7;use_weights=false",
    "NDCG:top=4;type=Exp;denominator=LogPosition",
))
def test_subgroup_metrics_use_validation_relevance_and_original_query_mass(policy, metric):
    x, y, groups, objects, subgroups, group_weights = subgroup_problem()
    validation_y = np.float32(1 - y)
    effective = np.float32(objects * group_weights)
    model = CatBoostMetalRanker(**config(policy, iterations=4, eval_metric=metric)).fit(
        x, y, group_id=groups, sample_weight=objects, group_weight=group_weights,
        subgroup_id=subgroups,
        eval_set=(x, validation_y, groups, objects, group_weights, subgroups), use_best_model=False,
    )
    history = model.evals_result_["validation"][metric]
    for iteration, actual in enumerate(history, 1):
        raw = model.predict(x, task_type="GPU", ntree_end=iteration)
        if metric.startswith("PFound"):
            expected = independent_pfound(raw, validation_y, groups, effective, subgroups,
                                          4, .7, weighted="use_weights=false" not in metric)
        else:
            per_query = [eval_metric(validation_y[groups == group], raw[groups == group], metric,
                                     group_id=np.zeros(np.count_nonzero(groups == group), np.uint32),
                                     thread_count=1)[0] for group in np.unique(groups)]
            expected = np.average(per_query, weights=effective[model.group_offsets_[:-1]])
        assert actual == pytest.approx(expected, abs=2e-7)
        learn = independent_pfound(raw, y, groups, effective, subgroups, 6, .85)
        assert model.evals_result_["learn"]["PFound"][iteration - 1] == pytest.approx(learn, abs=2e-7)
    assert model.training_stats_["metric_maximized"]
    assert model.best_iteration_ == int(np.argmax(history))
    assert model.best_score_["validation"][metric] == max(history)
    np.testing.assert_allclose(model.loss_history_[1:], model.evals_result_["learn"]["PFound"], rtol=1e-7)


@pytest.mark.parametrize("policy", POLICIES)
def test_best_model_and_early_stop_preserve_full_snapshot_cursor(tmp_path, policy):
    # Every query has one relevant document. Reversed validation relevance
    # gives the same ordering metric after each tree, fixing the first optimum.
    x = np.array([[0], [1]] * 4, np.float32)
    y = x.ravel().copy()
    groups = np.repeat(np.arange(4), 2)
    options = config(policy, iterations=9, depth=1, border_count=1,
                     learning_rate=.13, random_strength=0, eval_metric="PFound")
    fit = dict(group_id=groups, eval_set=(x, 1 - y, groups), use_best_model=True)
    direct = CatBoostMetalRanker(**options).fit(x, y, **fit)
    assert direct.best_iteration_ == 0 and direct.tree_count_ == 1
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path / "best.npz")
    CatBoostMetalRanker(**options).fit(x, y, **fit, **saved, callback=lambda info: info.iteration < 3)
    resumed = CatBoostMetalRanker(**options).fit(x, y, **fit, **saved)
    assert resumed._result.resumed_iterations == 3
    assert resumed.training_stats_["iterations_trained"] == 9
    assert_same_result(resumed, direct)
    assert len(resumed.evals_result_["validation"]["PFound"]) == 9
    with np.load(tmp_path / "best.npz", allow_pickle=False) as snapshot:
        header = json.loads(str(snapshot["metadata"].item()))
        assert header["completed_iterations"] == 9
        assert header["stats"]["yeti_rng"]["completed_iterations"] == 9
        assert len(header["history"]["validation"]["PFound"]) == 9
    early_saved = saved | dict(snapshot_file=tmp_path / "early.npz")
    early = CatBoostMetalRanker(**options).fit(x, y, **fit, **early_saved, early_stopping_rounds=2)
    assert early.tree_count_ == 1 and early.training_stats_["iterations_trained"] == 3
    assert early.training_stats_["stop_reason"] == "early_stopping"
    np.testing.assert_array_equal(early.predict(x, task_type="GPU"), direct.predict(x, task_type="GPU"))
    terminal = CatBoostMetalRanker(**options).fit(x, y, **fit, **early_saved, early_stopping_rounds=2)
    assert terminal._result.resumed_iterations == 3
    assert_same_result(terminal, early)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("dataset", ("learn", "validation"))
def test_snapshot_rejects_changed_subgroups_before_gpu_loading(tmp_path, monkeypatch, policy, dataset):
    x, y, groups, objects, subgroups, group_weights = subgroup_problem()
    fit = dict(group_id=groups, sample_weight=objects, group_weight=group_weights, subgroup_id=subgroups,
               eval_set=(x, y, groups, objects, group_weights, subgroups), use_best_model=False,
               save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path / "subgroups.npz")
    options = config(policy, iterations=2)
    CatBoostMetalRanker(**options).fit(x, y, **fit)
    changed = subgroups.copy()
    changed[0] = "z"
    if dataset == "learn":
        fit["subgroup_id"] = changed
    else:
        fit["eval_set"] = (x, y, groups, objects, group_weights, changed)
    monkeypatch.setattr(_greedy, "build_library", lambda: pytest.fail("Changed subgroup snapshot reached GPU loading"))
    with pytest.raises(ValueError, match="(?i)snapshot.*(differ|match)|(differ|match).*snapshot"):
        CatBoostMetalRanker(**options).fit(x, y, **fit)


@pytest.mark.parametrize("policy", POLICIES)
def test_classic_defaults_and_explicit_centering_are_recorded(policy):
    x, y, groups, _ = data()
    options = config(policy, iterations=2)
    for key in ("leaf_estimation_method", "leaf_estimation_iterations", "leaf_estimation_backtracking"):
        del options[key]
    model = CatBoostMetalRanker(**options, yeti_legacy_prefix_centering=True)
    assert model.l2_leaf_reg == 0
    assert model.leaf_estimation_method == "Newton"
    assert model.leaf_estimation_iterations == 1
    assert model.leaf_estimation_backtracking == "No"
    assert model.eval_metric == "PFound"
    model.fit(x, y, group_id=groups)
    assert model.training_stats_["yeti_centering"] == "legacy_prefix"
    assert model.training_stats_["yeti_rng"]["leaf_iterations"] == 1


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("extra", (
    dict(loss_function="YetiRank:mode=NDCG"),
    dict(loss_function="YetiRank:permutations=1.5"),
    dict(loss_function="YetiRank:decay=-1"),
    dict(leaf_estimation_method="Gradient"),
    dict(leaf_estimation_backtracking="Armijo"),
))
def test_classic_yeti_option_validation(policy, extra):
    with pytest.raises(ValueError):
        CatBoostMetalRanker(**config(policy, **extra))
