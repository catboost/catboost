"""Native query targets in CUDA's FeatureParallel Plain and Ordered modes.

All fits use Metal. Metrics are recomputed from their mathematical definitions;
compound CTR buckets and predictions use the independent original-row oracle
from card 2. No CPU model is trained as a reference.
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRanker, Pool

from test_native_compound_ctrs import (
    CTR_KINDS, categorical_problem, check_final_tables, exported, independent_prediction,
    options as compound_options, projections, snapshot_options,
)
from test_native_greedy_api import SCORES, StopAfter, sampler


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal FeatureParallel query adapter",
)

LOSSES = ("QueryRMSE", "QuerySoftMax:beta=0.7;lambda=0.03", "PairLogit",
          "YetiRank:permutations=7;decay=0.85")
METHODS = [(loss, method) for loss in LOSSES
           for method in (("Newton",) if loss.startswith("YetiRank") else ("Newton", "Gradient"))]
BOOSTING = ("Plain", "Ordered")
SAMPLERS = ("No", "Bayesian", "Bernoulli", "Poisson", "MVS")


@pytest.fixture(autouse=True)
def require_gpu_training(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(loss="QueryRMSE", boosting="Ordered", count=4, **extra):
    config = dict(
        task_type="GPU", loss_function=loss, boosting_type=boosting,
        data_partition="FeatureParallel", grow_policy="SymmetricTree",
        iterations=7, depth=3, learning_rate=.15, random_seed=713,
        border_count=16, bootstrap_type="No", random_strength=0,
        score_function="Cosine", l2_leaf_reg=2, leaf_estimation_method="Newton",
        leaf_estimation_iterations=3, leaf_estimation_backtracking="No",
        permutation_count=count, has_time=count == 1,
        metric_period=1, verbose=False, allow_writing_files=False,
    )
    if boosting == "Ordered":
        config.update(min_fold_size=8, fold_len_multiplier=1.7, fold_permutation_block=3)
    return config | extra


def literal_pairs(targets, groups):
    edges = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        ranked = rows[np.argsort(targets[rows], kind="stable")]
        edges.extend((int(a), int(b)) for a, b in zip(ranked[1:], ranked[:-1]) if targets[a] > targets[b])
    edges = np.asarray(edges, dtype=np.uint32).reshape(-1, 2)
    assert len(edges)
    weights = (.25 + (np.arange(len(edges)) % 11) / 7).astype(np.float32)
    weights[::13] = 0
    # Repeated weighted observations are retained, and zero reversed edges
    # must not contribute either derivatives or metric mass.
    return (np.concatenate((edges, edges[::7], edges[::9, ::-1])),
            np.concatenate((weights, weights[::7] * .4, np.zeros(len(edges[::9]), np.float32))))


def numeric_problem(loss):
    rng = np.random.default_rng(4812)
    sizes = np.tile([3, 5, 7, 4, 6, 8], 4)
    groups = np.repeat(np.arange(len(sizes), dtype=np.uint64), sizes)
    x = rng.normal(size=(len(groups), 4)).astype(np.float32)
    signal = 1.3 * x[:, 0] - .6 * x[:, 1] + .4 * x[:, 0] * x[:, 2]
    y = (signal + (groups % 5) * .75).astype(np.float32)
    if not loss.startswith("QueryRMSE"):
        y = (.125 + .75 / (1 + np.exp(-signal))).astype(np.float32)
    weights = (.4 + (np.arange(len(x)) % 9) / 5).astype(np.float32)
    weights[::17] = 0
    po = dict(group_id=groups, weight=weights)
    if loss == "PairLogit":
        po["pairs"], po["pairs_weight"] = literal_pairs(y, groups)
    return x, y, po


def compound_problem(loss, complexity=2):
    x, y, future, po = categorical_problem(complexity)
    y = (.125 + .75 * y).astype(np.float32)
    groups = np.arange(len(x), dtype=np.uint64) // 6
    po["group_id"] = groups
    if loss == "PairLogit":
        po["pairs"], po["pairs_weight"] = literal_pairs(y, groups)
    future_y = np.resize(y, len(future))
    future_po = dict(cat_features=po["cat_features"],
                     group_id=np.arange(len(future), dtype=np.uint64) // 6,
                     weight=(.6 + np.arange(len(future)) % 5 / 5).astype(np.float32))
    if loss == "PairLogit":
        future_po["pairs"], future_po["pairs_weight"] = literal_pairs(future_y, future_po["group_id"])
    return x, y, po, future, future_y, future_po


def compound_config(loss, boosting="Ordered", count=4, kind="Borders", history="Group", **extra):
    # Explicit query modes reuse the same dynamic categorical contract as the
    # already accepted scalar modes; only the objective/derivative routing differs.
    config = compound_options(kind, boosting, count)
    config.pop("boost_from_average", None)
    config.update(options(loss, boosting, count), max_ctr_complexity=2,
                  depth=4, iterations=8, ctr_history_unit=history)
    return config | extra


def fit(config, pool, **kwargs):
    return CatBoostRanker().set_params(**config).fit(pool, **kwargs)


def metric_name(loss):
    return "PFound" if loss.startswith("YetiRank") else loss.partition(":")[0]


def metric_history(model, dataset, metric):
    matches = [(key, value) for key, value in model.get_evals_result()[dataset].items()
               if key.partition(":")[0] == metric and "use_weights=false" not in key]
    assert len(matches) == 1, matches
    return matches[0][1]


def independent_metric(loss, raw, target, po):
    raw = np.asarray(raw, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups = po["group_id"]
    weights = np.asarray(po.get("weight", np.ones(len(raw))), dtype=np.float64)
    base = loss.partition(":")[0]
    if base == "PairLogit":
        edges = po["pairs"]
        return np.average(np.logaddexp(0, raw[edges[:, 1]] - raw[edges[:, 0]]), weights=po["pairs_weight"])
    if base == "QueryRMSE":
        residual = target - raw
        for group in np.unique(groups):
            rows = groups == group
            total = weights[rows].sum()
            if total:
                residual[rows] -= np.dot(residual[rows], weights[rows]) / total
        return np.sqrt(np.average(residual ** 2, weights=weights))
    if base == "QuerySoftMax":
        parameters = dict(part.split("=", 1) for part in loss.partition(":")[2].split(";") if "=" in part)
        beta = float(parameters.get("beta", 1))
        numerator = denominator = 0.
        for group in np.unique(groups):
            rows = (groups == group) & (weights > 0)
            if not rows.any():
                continue
            score = beta * raw[rows]
            mass = target[rows] * weights[rows]
            log_normalizer = np.log(np.dot(np.exp(score - score.max()), weights[rows])) + score.max()
            numerator -= np.dot(mass, score + np.log(weights[rows]) - log_normalizer)
            denominator += mass.sum()
        return numerator / denominator
    assert base == "YetiRank"
    scores, query_weights = [], []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        # Shared PFound uses pessimistic relevance order for tied predictions.
        ranked = rows[np.lexsort((target[rows], -raw[rows]))]
        look, found = 1., 0.
        for row in ranked:
            found += look * target[row]
            look *= (1 - target[row]) * .85
        scores.append(found)
        # CUDA caches the first effective document weight as its query weight.
        query_weights.append(weights[rows[0]])
    return np.average(scores, weights=query_weights)


def check_metrics(model, loss, y, po):
    expected = independent_metric(loss, model.get_test_eval(), y, po)
    actual = metric_history(model, "validation", metric_name(loss))[-1]
    assert actual == pytest.approx(expected, rel=5e-6, abs=2e-7)


def readers(model, x, tmp_path):
    expected = model.predict(x, task_type="GPU")
    assert np.isfinite(expected).all() and np.ptp(expected) > 1e-6
    for fmt in ("cbm", "json"):
        path = tmp_path / ("query-mode." + fmt)
        model.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        for reader in (model, loaded):
            for task in ("CPU", "GPU"):
                np.testing.assert_allclose(reader.predict(x, task_type=task),
                                           expected, atol=2e-6, rtol=5e-6)


def check_readers_and_oracle(model, x, y, future, tmp_path, minimum_complexity=2):
    # Ranker.predict already returns RawFormulaVal and has no prediction_type
    # parameter. Reuse the independent CTR equations with its public signature.
    document = exported(model, tmp_path / "compound.json")
    selected = projections(document)
    assert any(len(projection) >= minimum_complexity for projection in selected), selected
    check_final_tables(document, x, y)
    expected = independent_prediction(document, x, y, future)
    assert np.isfinite(expected).all() and np.ptp(expected) > 1e-5
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert int(model.get_metadata()["metal_tree_ctr_features"]) > 0
    for fmt in ("json", "cbm"):
        path = tmp_path / ("compound." + fmt)
        model.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        for reader in (model, loaded):
            for task in ("CPU", "GPU"):
                np.testing.assert_allclose(reader.predict(future, task_type=task),
                                           expected, atol=2e-6, rtol=5e-6)
    return document


def check_exact(actual, expected, x):
    assert actual.tree_count_ == expected.tree_count_
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(actual, method)(), getattr(expected, method)())
    np.testing.assert_array_equal(actual.predict(x, task_type="GPU"), expected.predict(x, task_type="GPU"))
    assert actual.get_evals_result() == expected.get_evals_result()
    assert actual.get_best_iteration() == expected.get_best_iteration()
    assert actual.get_best_score() == expected.get_best_score()


def check_original_leaf_mass(model, loss, po):
    expected = 2 * po["pairs_weight"].sum(dtype=float) if loss == "PairLogit" else po["weight"].sum(dtype=float)
    at = 0
    for leaves in model.get_tree_leaf_counts():
        assert model.get_leaf_weights()[at:at + leaves].sum() == pytest.approx(expected, rel=5e-6, abs=5e-6)
        at += leaves


@pytest.mark.parametrize("loss,method", METHODS)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("score", ("Cosine", "NewtonCosine"))
@pytest.mark.parametrize("count", (1, 4))
def test_native_query_feature_parallel_objectives_methods_and_independent_metrics(tmp_path, loss, method, boosting, score, count):
    x, y, po = numeric_problem(loss)
    pool = Pool(x, y, **po)
    config = options(loss, boosting, count, score_function=score, leaf_estimation_method=method)
    model = fit(config, pool, eval_set=pool, use_best_model=False)
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_all_params()["data_partition"] == "FeatureParallel"
    assert model.get_all_params()["boosting_type"] == boosting
    assert model.get_all_params()["leaf_estimation_method"] == method
    check_metrics(model, loss, y, po)
    check_original_leaf_mass(model, loss, po)
    np.testing.assert_allclose(model.get_test_eval(), model.predict(x, task_type="GPU"), atol=2e-6, rtol=5e-6)
    readers(model, x, tmp_path)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("score", SCORES)
def test_native_plain_feature_parallel_keeps_all_cuda_scalar_scores(tmp_path, loss, score):
    x, y, po = numeric_problem(loss)
    pool = Pool(x, y, **po)
    model = fit(options(loss, "Plain", score_function=score), pool, eval_set=pool, use_best_model=False)
    assert model.get_all_params()["score_function"] == score
    check_metrics(model, loss, y, po)
    readers(model, x, tmp_path)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("sampling", SAMPLERS)
def test_native_query_feature_parallel_snapshots_replay_all_stochastic_cursors(tmp_path, loss, boosting, sampling):
    x, y, po = numeric_problem(loss)
    pool = Pool(x, y, **po)
    config = options(loss, boosting, random_strength=.5, **sampler(sampling))
    direct = fit(config, pool, eval_set=pool, use_best_model=False)
    saved = snapshot_options(config, tmp_path)
    callback = StopAfter(3)
    partial = fit(saved, pool, eval_set=pool, use_best_model=False, callbacks=[callback])
    assert partial.tree_count_ == 3 and callback.iterations == [1, 2, 3]
    resumed = fit(saved, pool, eval_set=pool, use_best_model=False)
    check_exact(resumed, direct, x)
    check_exact(fit(saved, pool, eval_set=pool, use_best_model=False), direct, x)
    extended = fit(saved | dict(iterations=9), pool, eval_set=pool, use_best_model=False)
    longer = fit(config | dict(iterations=9), pool, eval_set=pool, use_best_model=False)
    check_exact(extended, longer, x)
    check_metrics(extended, loss, y, po)
    readers(extended, x, tmp_path)
    changed = y.copy()
    changed[0] += .03125
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
        fit(saved | dict(iterations=9), Pool(x, changed, **po), eval_set=pool, use_best_model=False)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("kind", CTR_KINDS)
@pytest.mark.parametrize("history", ("Sample", "Group"))
@pytest.mark.parametrize("count", (1, 4))
def test_native_query_compounds_have_exact_learn_tables_and_unseen_prediction(tmp_path, loss, boosting, kind, history, count):
    x, y, po, future, future_y, future_po = compound_problem(loss)
    pool = Pool(x, y, **po)
    if count == 4:
        pool.quantize(border_count=16)
    evaluation = Pool(future, future_y, **future_po)
    config = compound_config(loss, boosting, count, kind, history)
    model = fit(config, pool, eval_set=evaluation, use_best_model=False)
    assert model.get_metadata()["metal_permutations"] == str(count)
    check_readers_and_oracle(model, x, y, future, tmp_path)
    check_metrics(model, loss, future_y, future_po)
    check_original_leaf_mass(model, loss, po)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("history", ("Sample", "Group"))
def test_native_query_compound_histories_survive_callback_complete_and_extended_snapshots(tmp_path, loss, boosting, history):
    x, y, po, future, future_y, future_po = compound_problem(loss)
    pool = Pool(x, y, **po)
    pool.quantize(border_count=16)
    evaluation = Pool(future, future_y, **future_po)
    config = compound_config(loss, boosting, kind="FloatTargetMeanValue", history=history,
                             bootstrap_type="Bernoulli", subsample=.8, random_strength=.4)
    saved = snapshot_options(config, tmp_path)
    partial = fit(saved, pool, eval_set=evaluation, use_best_model=False, callbacks=[StopAfter(3)])
    assert partial.tree_count_ == 3
    assert any(len(p) > 1 for p in projections(exported(partial, tmp_path / "partial.json")))
    direct = fit(config, pool, eval_set=evaluation, use_best_model=False)
    resumed = fit(saved, pool, eval_set=evaluation, use_best_model=False)
    check_exact(resumed, direct, future)
    check_exact(fit(saved, pool, eval_set=evaluation, use_best_model=False), direct, future)
    extended = fit(saved | dict(iterations=10), pool, eval_set=evaluation, use_best_model=False)
    longer = fit(config | dict(iterations=10), pool, eval_set=evaluation, use_best_model=False)
    check_exact(extended, longer, future)
    check_readers_and_oracle(extended, x, y, future, tmp_path)
    check_metrics(extended, loss, future_y, future_po)
    changed = x.copy()
    changed[0, 1] = "different-compound-key"
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
        fit(saved, Pool(changed, y, **po), eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
def test_native_query_compound_complexity_three_is_selected(tmp_path, loss, boosting):
    x, y, po, future, future_y, future_po = compound_problem(loss, 3)
    model = fit(compound_config(loss, boosting, max_ctr_complexity=3, iterations=12, depth=5),
                Pool(x, y, **po), eval_set=Pool(future, future_y, **future_po), use_best_model=False)
    check_readers_and_oracle(model, x, y, future, tmp_path, minimum_complexity=3)
    check_metrics(model, loss, future_y, future_po)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
def test_native_query_initial_compound_models_baselines_and_resume(tmp_path, loss, boosting):
    x, y, po, future, _, _ = compound_problem(loss)
    pool = Pool(x, y, **po)
    config = compound_config(loss, boosting, iterations=5)
    initial = fit(config | dict(iterations=3), pool)
    assert any(len(p) > 1 for p in projections(exported(initial, tmp_path / "initial.json")))
    baseline = np.linspace(-.2, .3, len(x), dtype=np.float32)
    pool.set_baseline(baseline)
    direct = fit(config, pool, init_model=initial, eval_set=pool, use_best_model=False)
    saved = snapshot_options(config, tmp_path)
    partial = fit(saved, pool, init_model=initial, eval_set=pool, use_best_model=False, callbacks=[StopAfter(2)])
    assert partial.tree_count_ == initial.tree_count_ + 2
    resumed = fit(saved, pool, init_model=initial, eval_set=pool, use_best_model=False)
    check_exact(resumed, direct, future)
    np.testing.assert_allclose(resumed.get_test_eval(), resumed.predict(x, task_type="GPU") + baseline,
                               atol=2e-6, rtol=5e-6)
    check_metrics(resumed, loss, y, po)
    document = check_readers_and_oracle(resumed, x, y, future, tmp_path)
    info = document["features_info"]
    offset = sum(len(f["borders"]) for f in info.get("float_features", []))
    offset += sum(len(f.get("values", [])) for f in info.get("categorical_features", []))
    indices = set()
    for ctr in info.get("ctrs", []):
        if len(ctr["elements"]) > 1:
            indices.update(range(offset, offset + len(ctr["borders"])))
        offset += len(ctr["borders"])
    assert any(s["split_index"] in indices for tree in document["oblivious_trees"][initial.tree_count_:]
               for s in tree["splits"] or [])


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
def test_native_query_best_model_and_terminal_early_stop_recover_exactly(tmp_path, loss, boosting):
    x, y, po = numeric_problem(loss)
    train = Pool(x, y, **po)
    reversed_y = y.max() + y.min() - y
    reversed_po = dict(po)
    if loss == "PairLogit":
        reversed_po["pairs"] = po["pairs"][:, ::-1]
    evaluation = Pool(x, reversed_y, **reversed_po)
    config = options(loss, boosting, iterations=15, learning_rate=.4)
    direct = fit(config, train, eval_set=evaluation, use_best_model=True)
    history = metric_history(direct, "validation", metric_name(loss))
    best = int(np.argmax(history) if loss.startswith("YetiRank") else np.argmin(history))
    assert direct.get_best_iteration() == best
    assert direct.tree_count_ == best + 1 < config["iterations"]
    saved = snapshot_options(config, tmp_path)
    fit(saved, train, eval_set=evaluation, use_best_model=True, callbacks=[StopAfter(2)])
    resumed = fit(saved, train, eval_set=evaluation, use_best_model=True)
    check_exact(resumed, direct, x)
    early = fit(config, train, eval_set=evaluation, use_best_model=True, early_stopping_rounds=2)
    assert len(metric_history(early, "validation", metric_name(loss))) < config["iterations"]
    early_saved = snapshot_options(config, tmp_path / "early")
    fit(early_saved, train, eval_set=evaluation, use_best_model=True,
        early_stopping_rounds=2, callbacks=[StopAfter(1)])
    check_exact(fit(early_saved, train, eval_set=evaluation, use_best_model=True, early_stopping_rounds=2), early, x)
    check_exact(fit(early_saved, train, eval_set=evaluation, use_best_model=True, early_stopping_rounds=2), early, x)
    readers(resumed, x, tmp_path)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
def test_native_query_effective_weights_are_combined_once_and_literal_pairs_remain_authoritative(loss, boosting):
    x, y, po = numeric_problem(loss)
    groups = po["group_id"]
    group_weights = (.5 + np.arange(int(groups.max()) + 1) / 16).astype(np.float32)[groups]
    separate_po = {key: value for key, value in po.items() if key != "weight"}
    separate = Pool(x, y, group_weight=group_weights, **separate_po)
    separate.set_weight(po["weight"])
    combined_po = po | dict(weight=np.float32(po["weight"] * group_weights))
    config = options(loss, boosting, iterations=4)
    direct = fit(config, Pool(x, y, **combined_po), eval_set=Pool(x, y, **combined_po), use_best_model=False)
    split = fit(config, separate, eval_set=separate, use_best_model=False)
    check_exact(split, direct, x)
    if loss == "PairLogit":
        plain_po = {key: value for key, value in po.items() if key != "weight"}
        unweighted = fit(config, Pool(x, y, **plain_po), eval_set=Pool(x, y, **plain_po), use_best_model=False)
        for method in ("get_leaf_values", "get_leaf_weights", "get_test_eval"):
            np.testing.assert_array_equal(getattr(direct, method)(), getattr(unweighted, method)())


@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("quantized", (False, True))
def test_native_feature_parallel_literal_pair_pools_without_targets(boosting, quantized):
    x, y, po = numeric_problem("PairLogit")
    unlabeled = Pool(x, **po)
    labeled = Pool(x, y, **po)
    if quantized:
        unlabeled.quantize(border_count=16)
        labeled.quantize(border_count=16)
    config = options("PairLogit", boosting, custom_metric=["PairAccuracy"], iterations=4)
    direct = fit(config, labeled, eval_set=labeled, use_best_model=False)
    without_labels = fit(config, unlabeled, eval_set=unlabeled, use_best_model=False)
    check_exact(without_labels, direct, x)
    raw = np.asarray(without_labels.get_test_eval())
    edges = po["pairs"]
    expected = np.average(raw[edges[:, 0]] > raw[edges[:, 1]], weights=po["pairs_weight"])
    assert metric_history(without_labels, "validation", "PairAccuracy")[-1] == pytest.approx(expected, abs=2e-7)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("backtracking", ("AnyImprovement", "Armijo"))
def test_native_feature_parallel_query_backtracking_contract(tmp_path, loss, boosting, backtracking):
    x, y, po = numeric_problem(loss)
    pool = Pool(x, y, **po)
    config = options(loss, boosting, leaf_estimation_backtracking=backtracking, leaf_estimation_iterations=4)
    if loss.startswith("YetiRank"):
        with pytest.raises(CatBoostError, match="(?i)yeti.*backtrack|backtrack.*yeti"):
            fit(config, pool)
    else:
        direct = fit(config, pool, eval_set=pool, use_best_model=False)
        check_metrics(direct, loss, y, po)
        readers(direct, x, tmp_path)


@pytest.mark.parametrize("objective", ("QueryCrossEntropy", "PairLogitPairwise", "YetiRankPairwise"))
@pytest.mark.parametrize("boosting", BOOSTING)
def test_native_full_matrix_modes_do_not_silently_enter_feature_parallel(objective, boosting):
    x, y, po = numeric_problem("PairLogit")
    config = options(objective, boosting, score_function="NewtonCosine")
    with pytest.raises(CatBoostError, match="(?i)DocParallel|doc.parallel|Plain|FeatureParallel"):
        fit(config, Pool(x, y, **po))


@pytest.mark.parametrize("loss", LOSSES)
def test_native_ordered_query_exact_leaves_remain_unsupported(loss):
    x, y, po = numeric_problem(loss)
    with pytest.raises(CatBoostError, match="(?i)Exact|leaf[ _]estimation"):
        fit(options(loss, leaf_estimation_method="Exact"), Pool(x, y, **po))


@pytest.mark.parametrize("score", ("L2", "NewtonL2", "SolarL2", "LOOL2", "SatL2"))
def test_native_ordered_query_score_boundary_is_explicit(score):
    x, y, po = numeric_problem("QueryRMSE")
    with pytest.raises(CatBoostError, match="(?i)Ordered.*(Cosine|score)|score.*Ordered"):
        fit(options(score_function=score), Pool(x, y, **po))


@pytest.mark.parametrize("loss", LOSSES[:3])
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
@pytest.mark.parametrize("count", (1, 4))
def test_native_feature_parallel_first_tree_matches_independent_group_derivatives(loss, boosting, method, count):
    # All queries have the same balanced split. This removes structure-search
    # ambiguity while retaining group offsets and Ordered estimation prefixes.
    x = np.tile([0., 0., 1., 1.], 4).astype(np.float32)[:, None]
    groups = np.repeat(np.arange(4, dtype=np.uint64), 4)
    baseline = (groups * .25).astype(np.float32)
    weights = np.ones(len(x), dtype=np.float32)
    if loss == "QueryRMSE":
        y = np.float32(2 * x.ravel() - 1 + groups * 5)
        # Query centering removes every group-specific target and baseline
        # offset. CUDA's QueryRMSE diagonal is the original unit weight.
        gradient, curvature = 1., 1.
    elif loss.startswith("QuerySoftMax"):
        y = np.float32(.25 + .5 * x.ravel())
        # At a constant query point, each document has probability 1/4,
        # target mass2, and beta=.7. Lambda regularizes curvature only.
        gradient = .7 * (.75 - 2 * .25)
        curvature = .7 * 2 * (.7 * .25 * .75 + .03)
    else:
        y = x.ravel().copy()
        gradient, curvature = .5, .25
    po = dict(group_id=groups, weight=weights, baseline=baseline)
    if loss == "PairLogit":
        po["pairs"] = np.array([(a + 2, a) for a in range(0, 16, 4)] +
                               [(a + 3, a + 1) for a in range(0, 16, 4)], dtype=np.uint32)
        po["pairs_weight"] = np.ones(8, dtype=np.float32)
    pool = Pool(x, y, **po)
    config = options(loss, boosting, count, iterations=1, depth=1, learning_rate=.2,
                     l2_leaf_reg=0, leaf_estimation_method=method, leaf_estimation_iterations=1,
                     border_count=1, fold_size_loss_normalization=False)
    model = fit(config, pool, eval_set=pool, use_best_model=False)
    value = .2 * gradient / (curvature if method == "Newton" else 1.)
    assert model.get_tree_leaf_counts().tolist() == [2]
    np.testing.assert_allclose(model.get_leaf_values(), [-value, value], atol=2e-7, rtol=5e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), [8., 8.], atol=1e-7, rtol=0)
    np.testing.assert_allclose(model.get_test_eval(), (2 * x.ravel() - 1) * value + baseline,
                               atol=2e-7, rtol=5e-6)
    check_metrics(model, loss, y, po)
