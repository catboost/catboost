"""Native Metal CV acceptance, with independent held-out metric equations.

Enable CATBOOST_NATIVE_METAL_CV_TESTS=1 with the rebuilt native package. Every
training request uses task_type='GPU'; CPU application only checks model readers.
Metrics-only and returned-model runs must produce exactly the same histories.
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool, cv
from catboost import core as catboost_core


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_CV_TESTS") != "1",
    reason="requires the rebuilt native Metal cross-validation adapter",
)

SYMMETRIC = ("PlainDP", "PlainFP", "OrderedFP")
GREEDY = ("Depthwise", "Lossguide", "Region")
SCALARS = (
    ("RMSE", "Newton"), ("Logloss", "Newton"), ("CrossEntropy", "Newton"),
    ("Poisson", "Newton"), ("Huber:delta=0.8", "Newton"),
    ("Expectile:alpha=0.7", "Gradient"), ("Lq:q=2.5", "Newton"),
    ("Tweedie:variance_power=1.5", "Newton"),
    ("LogLinQuantile:alpha=0.7", "Gradient"),
    ("Quantile:alpha=0.7", "Gradient"), ("MAE", "Gradient"), ("MAPE", "Gradient"),
)
QUERIES = ("QueryRMSE", "QuerySoftMax:beta=0.7;lambda=0.03", "PairLogit",
           "YetiRank:permutations=5;decay=0.85")
VECTORS = ("MultiClass", "MultiClassOneVsAll", "MultiRMSE",
           "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")
COMBINATIONS = {
    "point": (("RMSE", 1.25), ("Huber:delta=0.8", .75)),
    "query": (("RMSE", 2.), ("QueryRMSE", .6)),
    "pair": (("PairLogit", .75), ("RMSE", 1.25)),
    "yeti": (("YetiRank:permutations=5", .03), ("RMSE", 2.)),
}


class MetalSquaredError:
    def calc_ders_range_metal(self):
        return "const float r = target - approx; return float3(-weight*r*r, weight*r, weight);"

    def calc_ders_range(self, *args):
        raise AssertionError("CV must not evaluate a CPU objective callback")

    def calc_ders_range_gpu(self, *args):
        raise AssertionError("Metal CV must not evaluate a CUDA objective callback")


@pytest.fixture(autouse=True)
def require_gpu_training(monkeypatch):
    original_cv = catboost_core._cv
    original_fit = CatBoost._fit

    def checked_cv(*args, **kwargs):
        params = kwargs.get("params", args[0] if args else None)
        assert params["task_type"] == "GPU", "Native CV acceptance must train on Metal"
        return original_cv(*args, **kwargs)

    def checked_fit(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(catboost_core, "_cv", checked_cv)
    monkeypatch.setattr(CatBoost, "_fit", checked_fit)


@dataclass
class Problem:
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    pool_options: dict
    baseline: np.ndarray = None

    def pool(self, quantized=False):
        result = Pool(self.x, self.y, weight=self.weights, **self.pool_options)
        if self.baseline is not None:
            result.set_baseline(self.baseline)
        if quantized:
            result.quantize(border_count=12)
        return result


def problem(loss="RMSE", category=None, grouped=False, rows=96):
    rng = np.random.default_rng(47219)
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    signal = .8 * x[:, 0] - .4 * x[:, 1] + .2 * x[:, 2] ** 2
    base = loss.partition(":")[0]
    po = {}
    if category:
        ids = np.arange(rows)
        a, b = ids % 4, (ids // 4) % 4
        signal = .15 * signal + (a == b).astype(np.float32) + .12 * (a == 2)
        if category == "onehot":
            a %= 2
        x = np.column_stack(([f"a{v}" for v in a], [f"b{v}" for v in b], x[:, 2:])).astype(object)
        po["cat_features"] = [0, 1]
    y = signal.astype(np.float32)
    if base in ("Logloss", "MultiClass", "MultiClassOneVsAll"):
        y = ((signal > np.median(signal)).astype(np.float32) if base == "Logloss"
             else (np.arange(rows) % 3).astype(np.float32))
    elif base in ("CrossEntropy", "QuerySoftMax", "YetiRank", "YetiRankPairwise",
                  "QueryCrossEntropy", "PairLogit", "PairLogitPairwise"):
        y = (.1 + .8 / (1 + np.exp(-signal))).astype(np.float32)
    elif base in ("Poisson", "Tweedie", "LogLinQuantile", "MAPE"):
        y = np.exp(signal / 3).astype(np.float32)
    elif base in ("MultiRMSE", "MultiLogloss", "MultiCrossEntropy"):
        y = np.column_stack((signal + .3, -.6 * signal + x[:, -1].astype(float))).astype(np.float32)
        if base == "MultiLogloss":
            y = (y > 0).astype(np.float32)
        elif base == "MultiCrossEntropy":
            y = (1 / (1 + np.exp(-y))).astype(np.float32)
    weights = (.4 + (np.arange(rows) % 11) / 7).astype(np.float32)
    if grouped or base.startswith(("Query", "PairLogit", "YetiRank")):
        groups = np.arange(rows, dtype=np.uint64) // 4
        po["group_id"] = groups
        if base.startswith("PairLogit"):
            edges = []
            for group in np.unique(groups):
                indices = np.flatnonzero(groups == group)
                ranked = indices[np.argsort(y[indices], kind="stable")]
                edges.extend(zip(ranked[1:], ranked[:-1]))
            po["pairs"] = np.asarray(edges, dtype=np.uint32)
            po["pairs_weight"] = (.3 + (np.arange(len(edges)) % 7) / 5).astype(np.float32)
    return Problem(x, y, weights, po)


def options(loss="RMSE", mode="PlainDP", method="Newton", **extra):
    result = dict(
        task_type="GPU", loss_function=loss, iterations=4, depth=2,
        learning_rate=.15, random_seed=813, bootstrap_type="No", random_strength=0,
        score_function="Cosine", l2_leaf_reg=2, border_count=12,
        leaf_estimation_method=method, leaf_estimation_iterations=2,
        leaf_estimation_backtracking="No", boost_from_average=False,
        permutation_count=1, has_time=True, max_ctr_complexity=1,
        grow_policy=mode if mode in GREEDY else "SymmetricTree",
        boosting_type="Ordered" if mode == "OrderedFP" else "Plain",
        data_partition="FeatureParallel" if mode in ("PlainFP", "OrderedFP") else "DocParallel",
        metric_period=1, verbose=False, allow_writing_files=True,
    )
    if mode == "OrderedFP":
        result.update(min_fold_size=4, fold_len_multiplier=1.7, fold_permutation_block=3)
    if mode == "Lossguide":
        result["max_leaves"] = 4
    return result | extra


def explicit_folds(data):
    # Unequal, non-contiguous held-out sets distinguish arithmetic fold means
    # from a pooled/size-weighted metric and exercise quantized subset indexing.
    ids = data.pool_options.get("group_id", np.arange(len(data.y)))
    mask = ids % 5 < 2
    a, b = np.flatnonzero(mask), np.flatnonzero(~mask)
    return [(b, a), (a, b)]


def metric_column(history, base, suffix="mean", dataset="test"):
    keys = [key for key in history if key.startswith(dataset + "-") and key.endswith("-" + suffix)
            and key[len(dataset) + 1:].rsplit("-", 1)[0].partition(":")[0] == base
            and "use_weights=false" not in key]
    assert len(keys) == 1, (base, keys, list(history))
    return history[keys[0]]


def fold_options(data, indices):
    po = {}
    for name in ("group_id", "subgroup_id", "group_weight"):
        if name in data.pool_options:
            po[name] = data.pool_options[name][indices]
    po["weight"] = data.weights[indices]
    if "pairs" in data.pool_options:
        remap = np.full(len(data.y), -1, dtype=np.int64)
        remap[indices] = np.arange(len(indices))
        edges = remap[data.pool_options["pairs"]]
        keep = np.all(edges >= 0, axis=1)
        po["pairs"] = edges[keep]
        po["pairs_weight"] = data.pool_options["pairs_weight"][keep]
    return po


def independent_metric(description, raw, target, po):
    raw, target = np.asarray(raw, dtype=float), np.asarray(target, dtype=float)
    weights = np.asarray(po["weight"], dtype=float)
    base = description.partition(":")[0]
    residual = target - raw if raw.ndim == target.ndim else None
    if base == "RMSE":
        return np.sqrt(np.average(residual ** 2, weights=weights))
    if base in ("Logloss", "CrossEntropy"):
        value = np.logaddexp(0, raw) - target * raw
    elif base == "Poisson":
        value = np.exp(raw) - target * raw
    elif base == "Huber":
        value = np.where(np.abs(residual) <= .8, .5 * residual ** 2, .8 * (np.abs(residual) - .4))
    elif base == "Expectile":
        value = np.where(residual > 0, .7, .3) * residual ** 2
    elif base == "Lq":
        value = np.abs(residual) ** 2.5
    elif base == "Tweedie":
        value = 2 * (target * np.exp(-.5 * raw) + np.exp(.5 * raw))
    elif base == "LogLinQuantile":
        residual = target - np.exp(raw)
        value = np.where(residual > 0, .7, -.3) * residual
    elif base in ("Quantile", "MAE"):
        alpha = .5 if base == "MAE" else .7
        value = np.maximum(np.abs(residual) - 1e-6, 0) * np.where(residual > 0, alpha, 1 - alpha)
        if base == "MAE":
            value *= 2
    elif base == "MAPE":
        value = np.abs(residual) / np.maximum(1, np.abs(target))
    elif base == "MultiRMSE":
        return np.sqrt(np.average(np.sum(residual ** 2, axis=1), weights=weights))
    elif base == "RMSEWithUncertainty":
        value = .5 * np.log(2 * np.pi) + raw[:, 1] + .5 * np.exp(-2 * raw[:, 1]) * (target - raw[:, 0]) ** 2
    elif base in ("MultiLogloss", "MultiCrossEntropy"):
        value = np.mean(np.logaddexp(0, raw) - target * raw, axis=1)
    elif base in ("MultiClass", "MultiClassOneVsAll"):
        if base == "MultiClass":
            value = np.logaddexp.reduce(raw, axis=1) - raw[np.arange(len(target)), target.astype(int)]
        else:
            value = (np.logaddexp(0, raw).sum(axis=1) - raw[np.arange(len(target)), target.astype(int)]) / raw.shape[1]
    elif base == "PairLogit":
        pairs = po["pairs"]
        return np.average(np.logaddexp(0, raw[pairs[:, 1]] - raw[pairs[:, 0]]), weights=po["pairs_weight"])
    elif base == "QueryRMSE":
        for group in np.unique(po["group_id"]):
            rows = po["group_id"] == group
            residual[rows] -= np.average(residual[rows], weights=weights[rows])
        return np.sqrt(np.average(residual ** 2, weights=weights))
    elif base == "QuerySoftMax":
        numerator = denominator = 0.
        for group in np.unique(po["group_id"]):
            rows = po["group_id"] == group
            scores = .7 * raw[rows]
            log_z = np.log(np.dot(np.exp(scores - scores.max()), weights[rows])) + scores.max()
            mass = target[rows] * weights[rows]
            numerator -= np.dot(mass, scores + np.log(weights[rows]) - log_z)
            denominator += mass.sum()
        return numerator / denominator
    elif base == "PFound":
        scores, group_weights = [], []
        for group in np.unique(po["group_id"]):
            rows = np.flatnonzero(po["group_id"] == group)
            ranked = rows[np.lexsort((target[rows], -raw[rows]))]
            look, found = 1., 0.
            for row in ranked:
                found += look * target[row]
                look *= (1 - target[row]) * .85
            scores.append(found)
            group_weights.append(weights[rows[0]])
        return np.average(scores, weights=group_weights)
    else:
        raise AssertionError("Missing independent CV metric: " + description)
    return np.average(value, weights=weights)


def metric_for_loss(loss):
    base = loss.partition(":")[0]
    if base in ("YetiRank", "YetiRankPairwise", "QueryCrossEntropy"):
        return "PFound"
    return "PairLogit" if base == "PairLogitPairwise" else loss


def assert_fold_oracle(history, models, data, folds, metric, components=None):
    expected = []
    for model, (_, test) in zip(models, folds):
        values = []
        target = data.y[test]
        if target.dtype.kind in "OUS":
            mapping = {label: idx for idx, label in enumerate(model.classes_)}
            target = np.asarray([mapping[label] for label in target])
        last_sampled_tree_count = 1
        for iteration in history["iterations"]:
            # Shared CV advances only at emitted metric rows. A stopped fold
            # keeps its last sampled metric, even if it trained more trees
            # between that sample and its early-stop iteration.
            if int(iteration) < model.tree_count_:
                last_sampled_tree_count = int(iteration) + 1
            raw = model.predict(data.x[test], prediction_type="RawFormulaVal",
                                ntree_end=last_sampled_tree_count, task_type="GPU")
            if data.baseline is not None:
                raw = raw + data.baseline[test]
            po = fold_options(data, test)
            if components:
                value = sum((-1 if loss.startswith("YetiRank") else 1) * float(np.float32(weight))
                            * independent_metric(metric_for_loss(loss), raw, target, po)
                            for loss, weight in components)
            else:
                value = independent_metric(metric, raw, target, po)
            values.append(value)
        expected.append(values)
    expected = np.asarray(expected)
    base = "Combination" if components else metric.partition(":")[0]
    np.testing.assert_allclose(metric_column(history, base), expected.mean(axis=0), rtol=8e-6, atol=2e-6)
    np.testing.assert_allclose(metric_column(history, base, "std"), expected.std(axis=0, ddof=1), rtol=2e-5, atol=3e-6)


def run_both(tmp_path, data, config, quantized=False, folds=None, cv_options=None, metric=None,
             components=None, check_oracle=True):
    folds = explicit_folds(data) if folds is None else folds
    kwargs = dict(as_pandas=False, folds=folds, shuffle=False, stratified=False)
    kwargs.update(cv_options or {})
    metrics_dir, models_dir = tmp_path / "metrics", tmp_path / "models"
    # The shared CV writer always saves returned fold models. Give it regular
    # temporary train directories rather than changing that shared behavior.
    metrics = cv(data.pool(quantized), config | dict(train_dir=str(metrics_dir)), **kwargs)
    returned, models = cv(data.pool(quantized), config | dict(train_dir=str(models_dir)), return_models=True, **kwargs)
    assert metrics.keys() == returned.keys()
    for key in metrics:
        np.testing.assert_array_equal(metrics[key], returned[key], err_msg=key)
        if key != "iterations" and check_oracle:
            assert np.isfinite(metrics[key]).all(), key
    assert len(models) == (kwargs.get("fold_count", len(folds)))
    for model in models:
        assert isinstance(model, CatBoost) and model.is_fitted()
        assert model.get_metadata()["metal_backend"] == "METAL"
        assert model.get_all_params()["task_type"] == "GPU"
        assert model.get_all_params()["use_best_model"] is False
        assert model.random_seed_ == config["random_seed"]
        assert model.tree_count_ > 0
        if not kwargs.get("early_stopping_rounds"):
            assert model.tree_count_ == config["iterations"]
        assert np.isfinite(model.get_leaf_values()).all()
    # Match actual shared fold model basenames; logs remain permitted.
    model_files = [path for path in models_dir.rglob("*") if path.is_file() and path.name in ("model", "model.cbm")]
    assert len(model_files) == len(models)
    assert not [path for path in metrics_dir.rglob("*") if path.is_file() and path.name in ("model", "model.cbm")]
    if check_oracle:
        assert_fold_oracle(metrics, models, data, folds, metric or metric_for_loss(config["loss_function"]), components)
    return metrics, models


@pytest.mark.parametrize("loss,method", SCALARS)
@pytest.mark.parametrize("mode", SYMMETRIC)
@pytest.mark.parametrize("quantized", (False, True))
def test_scalar_symmetric_cv_matches_returned_fold_metrics(tmp_path, loss, method, mode, quantized):
    run_both(tmp_path, problem(loss), options(loss, mode, method), quantized)


@pytest.mark.parametrize("loss,method", [item for item in SCALARS if not item[0].startswith("Lq")])
@pytest.mark.parametrize("mode", GREEDY)
def test_greedy_scalar_cv_matches_returned_fold_metrics(tmp_path, loss, method, mode):
    run_both(tmp_path, problem(loss), options(loss, mode, method))


@pytest.mark.parametrize("loss", QUERIES)
@pytest.mark.parametrize("mode", SYMMETRIC)
@pytest.mark.parametrize("quantized", (False, True))
def test_query_cv_preserves_groups_and_explicit_pair_weights(tmp_path, loss, mode, quantized):
    run_both(tmp_path, problem(loss), options(loss, mode), quantized)


@pytest.mark.parametrize("loss", QUERIES)
@pytest.mark.parametrize("mode", GREEDY)
def test_greedy_query_cv_preserves_groups_and_explicit_pairs(tmp_path, loss, mode):
    run_both(tmp_path, problem(loss), options(loss, mode))


@pytest.mark.parametrize("loss", ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise:permutations=5"))
@pytest.mark.parametrize("quantized", (False, True))
def test_full_matrix_cv_returns_usable_models_and_histories(tmp_path, loss, quantized):
    metric = metric_for_loss(loss)
    run_both(tmp_path, problem(loss), options(loss, eval_metric=metric, score_function="NewtonL2"), quantized)


@pytest.mark.parametrize("loss,mode", [(loss, "PlainDP") for loss in VECTORS] +
                         [(loss, mode) for loss in ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")
                          for mode in GREEDY])
def test_vector_cv_dimensions_and_fold_metric_normalization(tmp_path, loss, mode):
    run_both(tmp_path, problem(loss), options(loss, mode))


@pytest.mark.parametrize("mode", SYMMETRIC)
@pytest.mark.parametrize("quantized", (False, True))
def test_custom_cv_executes_metal_objective_and_preserves_histories(tmp_path, mode, quantized):
    run_both(tmp_path, problem(), options(MetalSquaredError(), mode, eval_metric="RMSE"),
             quantized, metric="RMSE")


@pytest.mark.parametrize("profile", tuple(COMBINATIONS))
@pytest.mark.parametrize("mode", SYMMETRIC)
def test_combination_cv_retains_component_metrics_and_stochastic_state(tmp_path, profile, mode):
    components = COMBINATIONS[profile]
    loss = "Combination:" + ";".join(f"loss{i}={name};weight{i}={weight}" for i, (name, weight) in enumerate(components))
    data = problem("PairLogit" if profile == "pair" else "QuerySoftMax")
    run_both(tmp_path, data, options(loss, mode), components=components)


@pytest.mark.parametrize("loss", ("RMSE", "PairLogit"))
@pytest.mark.parametrize("category", ("onehot", "simple", "compound"))
@pytest.mark.parametrize("mode", SYMMETRIC)
@pytest.mark.parametrize("quantized", (False, True))
def test_categorical_cv_subsets_and_final_ctr_models(tmp_path, loss, category, mode, quantized):
    ctr = "Borders:CtrBorderType=Uniform:CtrBorderCount=7:Prior=0.5"
    config = options(loss, mode, one_hot_max_size=4 if category == "onehot" else 1,
                     simple_ctr=[ctr], combinations_ctr=[ctr], ctr_target_border_count=1,
                     max_ctr_complexity=2 if category == "compound" else 1,
                     ctr_history_unit="Group", model_size_reg=0, counter_calc_method="SkipTest")
    if category == "compound":
        # Compound CTRs require FeatureParallel; PlainDP's ordinary case is
        # exercised with one-hot and simple CTRs above.
        config["data_partition"] = "FeatureParallel"
    run_both(tmp_path, problem(loss, category=category, grouped=True), config, quantized)


@pytest.mark.parametrize("cv_type", ("Classical", "Inverted", "TimeSeries"))
@pytest.mark.parametrize("quantized", (False, True))
def test_builtin_split_types_match_explicit_fold_oracles(tmp_path, cv_type, quantized):
    data = problem(rows=96)
    blocks = np.array_split(np.arange(len(data.y)), 3 if cv_type == "TimeSeries" else 2)
    if cv_type == "TimeSeries":
        folds = [(np.concatenate(blocks[:i]), blocks[i]) for i in (1, 2)]
    else:
        folds = [(blocks[1], blocks[0]), (blocks[0], blocks[1])]
        if cv_type == "Inverted":
            folds = [(test, train) for train, test in folds]
    kwargs = dict(folds=None, fold_count=2, type=cv_type, shuffle=False, stratified=False)
    history, models = run_both(tmp_path, data, options(), quantized, folds=folds, cv_options=kwargs,
                               check_oracle=False)
    assert_fold_oracle(history, models, data, folds, "RMSE")


@pytest.mark.parametrize("quantized", (False, True))
def test_stratified_cv_preserves_string_class_labels(tmp_path, quantized):
    data = problem("MultiClass")
    labels = np.array(["zebra", "ant", "owl"])
    data.y = labels[data.y.astype(int)]
    config = options("MultiClass")
    history, models = run_both(tmp_path, data, config, quantized,
                               cv_options=dict(folds=None, fold_count=3, stratified=True), check_oracle=False)
    assert np.isfinite(metric_column(history, "MultiClass")).all()
    for model in models:
        np.testing.assert_array_equal(model.classes_, np.sort(labels))
        raw = model.predict(data.x, prediction_type="RawFormulaVal", task_type="GPU")
        assert raw.shape == (len(data.y), 3)


@pytest.mark.parametrize("loss", ("RMSE", "Logloss", "MultiClass", "MultiRMSE"))
@pytest.mark.parametrize("quantized", (False, True))
def test_fold_baselines_and_class_metadata_survive_cv(tmp_path, loss, quantized):
    data = problem(loss)
    columns = 3 if loss == "MultiClass" else 2 if loss == "MultiRMSE" else 1
    baseline = np.linspace(-.15, .2, len(data.y) * columns, dtype=np.float32).reshape(len(data.y), columns)
    data.baseline = baseline[:, 0] if columns == 1 else baseline
    config = options(loss)
    if loss in ("Logloss", "MultiClass"):
        labels = np.array(["second", "first", "third"][:columns if columns > 1 else 2])
        data.y = labels[data.y.astype(int)]
    run_both(tmp_path, data, config, quantized)


@pytest.mark.parametrize("period", (1, 3))
def test_cv_early_stopping_tracks_each_fold_and_carries_last_values(tmp_path, period):
    # Incompatible train/test signs ensure the same simple model gets worse
    # on held-out rows. Other folds learn their target, giving unequal lengths.
    data = problem(rows=96)
    data.y = np.where(np.arange(96) < 32, -data.x[:, 0], data.x[:, 0]).astype(np.float32)
    folds = [(np.arange(32, 96), np.arange(32)), (np.arange(32, 64), np.arange(64, 96))]
    config = options(iterations=18, metric_period=period, use_best_model=True)
    history, models = run_both(tmp_path, data, config, folds=folds,
                               cv_options=dict(early_stopping_rounds=2), check_oracle=False)
    counts = [model.tree_count_ for model in models]
    assert min(counts) < max(counts), counts
    assert min(counts) < config["iterations"]
    assert history["iterations"][-1] == max(counts) - 1
    assert_fold_oracle(history, models, data, folds, "RMSE")


@pytest.mark.parametrize("period", (1, 3, 5))
@pytest.mark.parametrize("skip_train", (False, True))
def test_cv_metric_period_extra_metrics_and_skip_train_hints(tmp_path, period, skip_train):
    loss = "Logloss:hints=skip_train~true" if skip_train else "Logloss"
    history, _ = run_both(tmp_path, problem("Logloss"),
                          options(loss, iterations=7, metric_period=period,
                                  custom_metric=["AUC", "BrierScore"]))
    expected_iterations = sorted(set(range(0, 7, period)) | {6})
    np.testing.assert_array_equal(history["iterations"], expected_iterations)
    assert not any(key.startswith("train-AUC-") for key in history)
    assert any(key.startswith("train-Logloss-") for key in history) is not skip_train
    for key in history:
        assert len(history[key]) == len(expected_iterations)
    assert np.isfinite(metric_column(history, "AUC")).all()
    assert np.isfinite(metric_column(history, "BrierScore")).all()


@pytest.mark.parametrize("quantized", (False, True))
def test_cv_model_readers_and_saved_fold_files(tmp_path, quantized):
    data = problem("RMSE", category="simple")
    _, models = run_both(tmp_path, data, options(one_hot_max_size=1), quantized)
    for index, model in enumerate(models):
        expected = model.predict(data.x, prediction_type="RawFormulaVal", task_type="GPU")
        for fmt in ("cbm", "json"):
            path = tmp_path / (f"fold-{index}." + fmt)
            model.save_model(path, format=fmt)
            loaded = CatBoost().load_model(path, format=fmt)
            assert loaded.get_metadata()["metal_backend"] == "METAL"
            for task in ("CPU", "GPU"):
                np.testing.assert_allclose(loaded.predict(data.x, prediction_type="RawFormulaVal", task_type=task),
                                           expected, rtol=5e-6, atol=5e-7)


def test_metrics_only_cv_can_disable_all_file_writing(tmp_path):
    data = problem()
    directory = tmp_path / "disabled"
    history = cv(data.pool(), options(allow_writing_files=False, train_dir=str(directory)),
                 folds=explicit_folds(data), as_pandas=False)
    assert len(history["iterations"]) == 4
    # core.cv creates the directories for plotting even when logging is off.
    assert not [path for path in directory.rglob("*") if path.is_file()]


@pytest.mark.parametrize("restriction", ("snapshot", "ordered", "group_overlap", "group_stratified"))
def test_cv_retains_shared_input_restrictions(tmp_path, restriction):
    data = problem("QueryRMSE" if restriction.startswith("group") else "RMSE")
    config = options("QueryRMSE" if restriction.startswith("group") else "RMSE", train_dir=str(tmp_path))
    pool = data.pool()
    kwargs = dict(fold_count=2, shuffle=False, as_pandas=False)
    if restriction == "snapshot":
        config["save_snapshot"] = True
        match = "Saving snapshots in Cross-validation is not supported"
    elif restriction == "ordered":
        pool = Pool(data.x, data.y, weight=data.weights, timestamp=np.arange(len(data.y), dtype=np.uint64))
        match = "Cross-validation for Ordered objects data"
    elif restriction == "group_overlap":
        kwargs.pop("fold_count")
        kwargs["folds"] = [(np.arange(48), np.arange(46, 96)), (np.arange(48, 96), np.arange(48))]
        match = "same group id must be in the same fold"
    else:
        kwargs["stratified"] = True
        match = "Stratified split is incompatible with groupwise metrics"
    with pytest.raises(CatBoostError, match=match):
        cv(pool, config, **kwargs)
