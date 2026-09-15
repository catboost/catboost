"""CUDA query-metric weight conventions and actual Metal selection lifecycle."""
import numpy as np
import pytest
from catboost import CatBoost
from catboost.utils import eval_metric
from catboost_metal import CatBoostMetalRanker
from catboost_metal._training import _shared_metric


METRICS = ("NDCG:top=4;type=Exp", "MAP:top=3;border=0.4", "PFound:top=4;decay=0.7")


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Ranking metric tests must train only through Metal")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def problem():
    rng = np.random.default_rng(986)
    sizes = np.array([3, 5, 4, 7, 6, 4])
    offsets = np.r_[0, sizes.cumsum()].astype(np.uint32)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    x = rng.normal(size=(len(groups), 3)).astype(np.float32)
    y = np.clip(.5 + .2 * x[:, 0] - .1 * x[:, 1], 0, 1).astype(np.float32)
    weights = rng.uniform(.3, 2, len(groups)).astype(np.float32)
    weights[offsets[:-1]] = [0, 4, .2, 2, .7, 1.5]
    pair_rows = np.array([(int(a) + 1, int(a)) for a in offsets[:-1]], np.uint32)
    pair_weights = np.array([.1, 1, 9, 3, .2, 7], np.float32)
    return x, y, offsets, groups, weights, pair_rows, pair_weights


def expected_metric(raw, labels, offsets, weights, name, paired=False):
    # Independent per-query shared calculations, then CUDA's query aggregation.
    # The pinned TMAPKMetric increments its denominator once per query and
    # ignores UseWeights/QueryInfo::Weight. CUDA invokes that same host metric.
    unweighted = name.endswith("use_weights=false") or name.startswith("MAP:")
    mass = np.ones(len(offsets) - 1) if unweighted or paired else weights[offsets[:-1]]
    values = [eval_metric(labels[a:b], raw[a:b], name,
                          group_id=np.zeros(b-a, np.uint32), thread_count=1)[0]
              for a, b in zip(offsets[:-1], offsets[1:])]
    return float(np.average(values, weights=mass))


@pytest.mark.parametrize("name", METRICS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("paired", [False, True])
def test_group_metrics_use_first_row_mass_or_pair_target_units(name, weighted, paired):
    x, labels, offsets, groups, weights, pairs, pair_weights = problem()
    name += ";use_weights=" + str(weighted).lower()
    raw = x[:, 2].astype(float)
    raw[3:6] = 0.25  # Keep shared CatBoost tie rules in the comparison.
    edges = (pairs[:, 0], pairs[:, 1], pair_weights) if paired else None
    actual = _shared_metric(name, raw, labels, weights, offsets, edges)
    assert actual == pytest.approx(expected_metric(raw, labels, offsets, weights, name, paired), abs=1e-14)
    # Changing non-first object mass must not alter the query aggregate.
    changed = weights * 23
    changed[offsets[:-1]] = weights[offsets[:-1]]
    assert _shared_metric(name, raw, labels, changed, offsets, edges) == actual


@pytest.mark.parametrize("name", METRICS)
def test_ranking_metric_requires_groups(name):
    with pytest.raises(ValueError, match="group boundaries"):
        _shared_metric(name, [0, 1], [0, 1], None)


@pytest.mark.parametrize("name", METRICS)
def test_pair_ranking_evaluation_requires_real_relevance_labels(name):
    x, y, offsets, groups, weights, pairs, pair_weights = problem()
    with pytest.raises(ValueError, match="relevance labels"):
        CatBoostMetalRanker(loss_function="PairLogit", eval_metric=name).fit(
            x, group_id=groups, pairs=pairs, pairs_weight=pair_weights)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax", "PairLogit"])
@pytest.mark.parametrize("name", METRICS)
def test_gpu_ranking_selection_metric_and_snapshot_resume(tmp_path, objective, name):
    x, y, offsets, groups, weights, pairs, pair_weights = problem()
    config = dict(loss_function=objective, eval_metric=name, iterations=8, depth=3,
                  learning_rate=.12, random_seed=41, bootstrap_type="Bernoulli", subsample=.8,
                  random_strength=0., leaf_estimation_iterations=2)
    paired = objective == "PairLogit"
    fit = dict(group_id=groups, sample_weight=weights,
               eval_set=(x, y, groups, weights), use_best_model=False)
    if paired:
        fit.update(pairs=pairs, pairs_weight=pair_weights, eval_pairs=pairs, eval_pairs_weight=pair_weights)
    full = CatBoostMetalRanker(**config).fit(x, y, **fit)
    actual = full.evals_result_["validation"][name][-1]
    raw = full.predict(x)
    assert actual == pytest.approx(expected_metric(raw, y, offsets, weights, name, paired), abs=2e-7)
    snapshot = tmp_path / "rank.snapshot"
    CatBoostMetalRanker(**{**config, "iterations": 3}).fit(x, y, **fit, save_snapshot=True,
                                                       snapshot_file=snapshot, snapshot_interval=0)
    resumed = CatBoostMetalRanker(**config).fit(x, y, **fit, save_snapshot=True,
                                              snapshot_file=snapshot, snapshot_interval=0)
    np.testing.assert_array_equal(resumed.predict(x), raw)
    assert resumed.evals_result_ == full.evals_result_
    assert full._result.stats["metric_maximized"] is True
    selected = CatBoostMetalRanker(**config).fit(x, y, **{**fit, "use_best_model": True},
                                               early_stopping_rounds=3)
    history = selected.evals_result_["validation"][name]
    assert selected.best_iteration_ == int(np.argmax(history))
    assert selected.tree_count_ == selected.best_iteration_ + 1
