"""Native compound CTR acceptance, including an independent final-table oracle.

Every fit uses Metal. The oracle groups original feature values directly and
recomputes learn-only statistics; it does not call the compound scheduler,
consume exported CTR tables, or fit a CPU reference. Standard CPU prediction is
used only to check the public model reader alongside Metal prediction.
"""

from collections import defaultdict
from itertools import product
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostError, CatBoostRegressor, Pool
from catboost_metal._categorical import cat_feature_hashes

from test_native_greedy_api import OBJECTIVES, StopAfter


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal compound CTR adapter",
)

CTR_KINDS = ("Borders", "Buckets", "FloatTargetMeanValue", "FeatureFreq")
BOOSTING = ("Plain", "Ordered")


@pytest.fixture(autouse=True)
def require_gpu_training(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(kind="Borders", boosting="Plain", count=4, complexity=2, **extra):
    ctr = f"{kind}:CtrBorderType=Uniform:CtrBorderCount=15:Prior=0.5"
    result = dict(
        task_type="GPU", data_partition="FeatureParallel", boosting_type=boosting,
        grow_policy="SymmetricTree", loss_function="RMSE", iterations=8, depth=4,
        learning_rate=.2, l2_leaf_reg=2, random_seed=619, bootstrap_type="No",
        random_strength=0, score_function="Cosine", leaf_estimation_method="Newton",
        leaf_estimation_iterations=2, leaf_estimation_backtracking="No",
        boost_from_average=False, metric_period=1, verbose=False, allow_writing_files=False,
        one_hot_max_size=2, max_ctr_complexity=complexity, model_size_reg=0,
        simple_ctr=[ctr], combinations_ctr=[ctr], ctr_target_border_count=1,
        ctr_history_unit="Sample", counter_calc_method="SkipTest", border_count=16,
        permutation_count=count, has_time=count == 1,
    )
    if boosting == "Ordered":
        result.update(min_fold_size=16, fold_len_multiplier=1.7, fold_permutation_block=3)
    return result | extra


def categorical_problem(complexity=2):
    """Marginal signals expose a stronger interaction to the dynamic scheduler.

    Unequal joint frequencies also give FeatureFreq an interaction signal.
    One joint tuple is absent although each constituent value occurs in learn.
    """
    rows, labels = [], []
    cardinality = 6 if complexity == 2 else 4
    for values in product(range(cardinality), repeat=complexity):
        if values == (0,) * (complexity - 1) + (1,):
            continue
        parity = sum(values) % 2
        count = 8 + 2 * values[0] + 10 * parity
        if complexity == 3:
            count += 4 * ((values[0] + values[1]) % 2)
        for _ in range(count):
            rows.append([f"cat-{column}-{value}" for column, value in enumerate(values)])
            labels.append(parity)
    order = np.random.default_rng(831).permutation(len(rows))
    x = np.asarray(rows, dtype=object)[order]
    y = np.asarray(labels, dtype=np.float32)[order]
    future = [[f"cat-{column}-{value}" for column, value in enumerate(values)]
              for values in product(range(cardinality), repeat=complexity)]
    # A completely new tuple, new constituents, and the absent tuple of known
    # values exercise both category and joint-key fallback to the CTR prior.
    future.extend([["never-seen"] * complexity,
                   ["never-seen"] + [f"cat-{column}-0" for column in range(1, complexity)]])
    weights = (.4 + (np.arange(len(x)) % 9) / 5).astype(np.float32)
    weights[::29] = 0
    return x, y, np.asarray(future, dtype=object), dict(cat_features=list(range(complexity)), weight=weights)


def mixed_problem(kind):
    rows, labels = [], []
    for category, side in product(range(8), range(2)):
        count = 24 + 3 * category + 9 * ((category + side) % 2)
        for repeat in range(count):
            value = ("left", "right")[side] if kind == "onehot" else float(side)
            rows.append([f"cat-{category}", value])
            # A leading side effect selects the numeric / one-hot split before
            # the conditional high-cardinality category is made available.
            positive_count = count * (.15 + .30 * side + .40 * ((category + side) % 2))
            labels.append(float(repeat < positive_count))
    order = np.random.default_rng(231).permutation(len(rows))
    x = np.asarray(rows, dtype=object)[order]
    y = np.asarray(labels, dtype=np.float32)[order]
    future = [[f"cat-{category}", ("left", "right")[side] if kind == "onehot" else float(side)]
              for category, side in product(range(8), range(2))]
    future.extend([["new-category", value] for value in (("left", "right") if kind == "onehot" else (-1., 2.))])
    return x, y, np.asarray(future, dtype=object), dict(cat_features=[0, 1] if kind == "onehot" else [0])


def exported(model, path):
    model.save_model(path, format="json")
    return json.loads(path.read_text())


def projections(document):
    return [ctr["elements"] for ctr in document["features_info"].get("ctrs", [])]


def _columns(document):
    info = document["features_info"]
    return ({f["feature_index"]: f["flat_feature_index"] for f in info.get("categorical_features", [])},
            {f["feature_index"]: f["flat_feature_index"] for f in info.get("float_features", [])})


def _project(rows, elements, cat_columns, float_columns):
    columns = []
    for element in elements:
        kind = element["combination_element"]
        if kind == "cat_feature_value":
            columns.append(rows[:, cat_columns[element["cat_feature_index"]]].tolist())
        elif kind == "float_feature":
            columns.append((rows[:, float_columns[element["float_feature_index"]]].astype(np.float32)
                            > np.float32(element["border"])).tolist())
        else:
            assert kind == "cat_feature_exact_value"
            hashes = cat_feature_hashes(rows[:, cat_columns[element["cat_feature_index"]]])
            columns.append((hashes == np.uint32(element["value"] & 0xffffffff)).tolist())
    return list(zip(*columns))


def independent_prediction(document, train_x, targets, future_x):
    """Recompute each final CTR from original rows, then traverse JSON leaves."""
    cat_columns, float_columns = _columns(document)
    info = document["features_info"]
    decisions = []
    for feature in info.get("float_features", []):
        values = future_x[:, feature["flat_feature_index"]].astype(np.float32)
        decisions.extend(values > np.float32(border) for border in feature["borders"])
    for feature in info.get("categorical_features", []):
        hashes = cat_feature_hashes(future_x[:, feature["flat_feature_index"]])
        decisions.extend(hashes == np.uint32(value & 0xffffffff) for value in feature.get("values", []))
    unique_targets = np.unique(targets)
    assert len(unique_targets) == 2, "oracle fixture requires one unambiguous target border"
    classes = (targets > (float(unique_targets[0]) + float(unique_targets[1])) / 2).astype(np.uint8)
    for ctr in info.get("ctrs", []):
        keys = _project(train_x, ctr["elements"], cat_columns, float_columns)
        statistics = defaultdict(lambda: [0, 0., 0, 0])
        for key, target, label in zip(keys, targets, classes):
            values = statistics[key]
            values[0] += 1
            values[1] += float(target)
            values[2 + int(label)] += 1
        values = []
        for key in _project(future_x, ctr["elements"], cat_columns, float_columns):
            count, total, negative, positive = statistics.get(key, [0, 0., 0, 0])
            kind = ctr["ctr_type"]
            if kind == "FeatureFreq":
                numerator, denominator = count, len(train_x)
            elif kind == "FloatTargetMeanValue":
                numerator, denominator = total, count
            elif kind == "Borders":
                assert ctr["target_border_idx"] == 0
                numerator, denominator = positive, count
            else:
                assert kind == "Buckets" and ctr["target_border_idx"] in (0, 1)
                numerator, denominator = (negative, positive)[ctr["target_border_idx"]], count
            # Match the public model's float32 CTR arithmetic, but obtain its
            # sufficient statistics independently from the original samples.
            ratio = ((np.float32(numerator) + np.float32(ctr["prior_numerator"])) /
                     (np.float32(denominator) + np.float32(ctr["prior_denomerator"])))
            values.append((ratio + np.float32(ctr["shift"])) * np.float32(ctr["scale"]))
        values = np.asarray(values, dtype=np.float32)
        decisions.extend(values > np.float32(border) for border in ctr["borders"])
    result = np.zeros(len(future_x), dtype=np.float64)
    for tree in document["oblivious_trees"]:
        leaf = np.zeros(len(future_x), dtype=np.int64)
        for depth, split in enumerate(tree["splits"] or []):
            leaf |= decisions[split["split_index"]].astype(np.int64) << depth
        result += np.asarray(tree["leaf_values"])[leaf]
    scale, bias = document["scale_and_bias"]
    return result * scale + bias[0]


def check_final_tables(document, train_x, targets):
    """Check every exported bucket against independent full-learn statistics."""
    cat_columns, float_columns = _columns(document)
    unique_targets = np.unique(targets)
    assert len(unique_targets) == 2
    target_border = (float(unique_targets[0]) + float(unique_targets[1])) / 2
    hash_lookup = {}
    for index, column in cat_columns.items():
        names = sorted(set(train_x[:, column]))
        hash_lookup[index] = dict(zip(names, map(int, cat_feature_hashes(names))))
    checked = set()
    for ctr in document["features_info"].get("ctrs", []):
        identifier = ctr["identifier"]
        if identifier in checked:
            continue
        checked.add(identifier)
        expected = defaultdict(lambda: [0, 0., 0, 0])
        keys = _project(train_x, ctr["elements"], cat_columns, float_columns)
        for key, target in zip(keys, targets):
            combined = 0
            for element, value in zip(ctr["elements"], key):
                if element["combination_element"] == "cat_feature_value":
                    value = hash_lookup[element["cat_feature_index"]][value]
                    if value >= 1 << 31:
                        value -= 1 << 32  # Model hashes sign-extend CityHash32.
                combined = (0x4906ba494954cb65 * (combined + 0x4906ba494954cb65 * int(value))) % (1 << 64)
            statistics = expected[combined]
            statistics[0] += 1
            statistics[1] += float(target)
            statistics[2 + int(target > target_border)] += 1
        table = document["ctr_data"][identifier]
        stride = table["hash_stride"]
        flat = table["hash_map"]
        # The upstream JSON exporter also emits one empty hash-bucket sentinel
        # with an arbitrary bucket-zero payload. It is not an inference key.
        actual = {int(flat[i]): flat[i + 1:i + stride] for i in range(0, len(flat), stride)
                  if int(flat[i]) != (1 << 64) - 1}
        assert actual.keys() == expected.keys()
        kind = ctr["ctr_type"]
        if kind == "FeatureFreq":
            assert table["counter_denominator"] == len(train_x)
        for key, (count, total, negative, positive) in expected.items():
            if kind == "FeatureFreq":
                assert actual[key] == [count]
            elif kind == "FloatTargetMeanValue":
                np.testing.assert_allclose(actual[key], [total, count], atol=2e-6, rtol=3e-6)
            else:
                assert actual[key] == [negative, positive]


def check_readers_and_oracle(model, x, y, future, tmp_path, minimum_complexity=2, element=None):
    document = exported(model, tmp_path / "compound.json")
    selected = projections(document)
    assert any(len(projection) >= minimum_complexity for projection in selected), selected
    if element:
        assert any(len(projection) > 1 and any(part["combination_element"] == element for part in projection)
                   for projection in selected), selected
    check_final_tables(document, x, y)
    expected = independent_prediction(document, x, y, future)
    assert np.isfinite(expected).all() and np.ptp(expected) > 1e-5
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert int(model.get_metadata()["metal_tree_ctr_features"]) > 0
    for fmt in ("json", "cbm"):
        path = tmp_path / ("compound." + fmt)
        model.save_model(path, format=fmt)
        loaded = model.__class__().load_model(path, format=fmt)
        for reader in (model, loaded):
            for task in ("CPU", "GPU"):
                np.testing.assert_allclose(
                    reader.predict(future, prediction_type="RawFormulaVal", task_type=task),
                    expected, atol=2e-6, rtol=5e-6,
                )
    return document


def fit(config, pool, **kwargs):
    cls = CatBoostClassifier if config["loss_function"] in ("Logloss", "CrossEntropy") else CatBoostRegressor
    return cls().set_params(**config).fit(pool, **kwargs)


def check_exact(actual, expected, future):
    assert actual.tree_count_ == expected.tree_count_
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(actual, method)(), getattr(expected, method)())
    np.testing.assert_array_equal(actual.predict(future, prediction_type="RawFormulaVal", task_type="GPU"),
                                  expected.predict(future, prediction_type="RawFormulaVal", task_type="GPU"))
    assert actual.get_evals_result() == expected.get_evals_result()


def snapshot_options(config, tmp_path):
    return config | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="compound.snapshot",
                         train_dir=str(tmp_path), allow_writing_files=True)


@pytest.mark.parametrize("kind", CTR_KINDS)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("quantized", (False, True))
def test_native_compounds_select_joint_projections_and_match_independent_inference(
        tmp_path, kind, boosting, count, quantized):
    x, y, future, pool_options = categorical_problem()
    if kind == "FloatTargetMeanValue":
        y = .125 + .75 * y
    pool = Pool(x, y, **pool_options)
    if quantized:
        pool.quantize(border_count=16)
    evaluation = Pool(future, np.resize(y, len(future)), cat_features=pool_options["cat_features"])
    config = options(kind, boosting, count)
    model = fit(config, pool, eval_set=evaluation, use_best_model=False)
    assert model.get_metadata()["metal_permutations"] == str(count)
    assert model.get_all_params()["max_ctr_complexity"] == 2
    document = check_readers_and_oracle(model, x, y, future, tmp_path)
    assert {ctr["ctr_type"] for ctr in document["features_info"]["ctrs"]} == {kind}
    np.testing.assert_allclose(model.get_test_eval(), independent_prediction(document, x, y, future),
                               atol=2e-6, rtol=5e-6)
    at = 0
    for leaves in model.get_tree_leaf_counts():
        assert model.get_leaf_weights()[at:at + leaves].sum() == pytest.approx(
            pool_options["weight"].sum(dtype=float), rel=5e-6,
        )
        at += leaves


@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
def test_native_complexity_three_is_used_and_complexity_one_stays_simple(tmp_path, boosting, count):
    x, y, future, pool_options = categorical_problem(3)
    pool = Pool(x, y, **pool_options)
    config = options(boosting=boosting, count=count, complexity=3, iterations=12, depth=5)
    compound = fit(config, pool)
    document = check_readers_and_oracle(compound, x, y, future, tmp_path, minimum_complexity=3)
    assert max(map(len, projections(document))) == 3
    simple = fit(config | dict(max_ctr_complexity=1,
                               data_partition="DocParallel" if boosting == "Plain" else "FeatureParallel"), pool)
    simple_doc = exported(simple, tmp_path / "simple.json")
    assert projections(simple_doc) and all(len(projection) == 1 for projection in projections(simple_doc))
    assert not np.array_equal(simple.predict(future), compound.predict(future))


@pytest.mark.parametrize("kind,element", (("numeric", "float_feature"), ("onehot", "cat_feature_exact_value")))
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
def test_native_tree_dependent_numeric_and_onehot_projections(tmp_path, kind, element, boosting, count):
    x, y, future, pool_options = mixed_problem(kind)
    config = options(boosting=boosting, count=count, iterations=12, depth=4)
    model = fit(config, Pool(x, y, **pool_options))
    check_readers_and_oracle(model, x, y, future, tmp_path, element=element)


@pytest.mark.parametrize("loss,method", list(OBJECTIVES) + [("Lq:q=2.7", "Newton")])
@pytest.mark.parametrize("boosting", BOOSTING)
def test_native_scalar_objectives_train_selected_compounds(tmp_path, loss, method, boosting):
    x, y, future, pool_options = categorical_problem()
    if loss != "Logloss":
        y = .125 + .75 * y
    if loss == "MAPE":
        y = 2 * y - 1  # Opposite residual signs with equal MAPE denominators.
    if loss.startswith("LogLinQuantile"):
        # exp(0)=1: straddle the initial prediction so quantile gradients
        # expose the interaction immediately instead of a constant first step.
        y = .5 + y
    config = options(boosting=boosting, loss_function=loss, leaf_estimation_method=method)
    model = fit(config, Pool(x, y, **pool_options))
    check_readers_and_oracle(model, x, y, future, tmp_path)
    history = next(iter(model.get_evals_result()["learn"].values()))
    assert np.isfinite(history).all()


@pytest.mark.parametrize("kind", CTR_KINDS)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("history", ("Sample", "Group"))
def test_native_compound_snapshots_restore_retained_grids_and_all_histories(tmp_path, kind, boosting, count, history):
    x, y, future, pool_options = categorical_problem()
    if history == "Group":
        pool_options["group_id"] = np.arange(len(x), dtype=np.uint64) // 6
    pool = Pool(x, y, **pool_options)
    if count == 4:
        pool.quantize(border_count=16)
    evaluation = Pool(future, np.resize(y, len(future)), cat_features=pool_options["cat_features"])
    config = options(kind, boosting, count, ctr_history_unit=history,
                     random_strength=.3, bootstrap_type="Bernoulli", subsample=.8)
    saved = snapshot_options(config, tmp_path)
    partial = fit(saved, pool, eval_set=evaluation, use_best_model=False, callbacks=[StopAfter(3)])
    assert partial.tree_count_ == 3 and (tmp_path / "compound.snapshot").is_file()
    assert any(len(projection) > 1 for projection in projections(exported(partial, tmp_path / "partial.json")))
    direct = fit(config, pool, eval_set=evaluation, use_best_model=False)
    resumed = fit(saved, pool, eval_set=evaluation, use_best_model=False)
    check_exact(resumed, direct, future)
    check_exact(fit(saved, pool, eval_set=evaluation, use_best_model=False), direct, future)
    extended = fit(saved | dict(iterations=10), pool, eval_set=evaluation, use_best_model=False)
    longer = fit(config | dict(iterations=10), pool, eval_set=evaluation, use_best_model=False)
    check_exact(extended, longer, future)
    check_readers_and_oracle(extended, x, y, future, tmp_path)
    changed = x.copy()
    changed[0, 1] = "changed-category"
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
        fit(saved, Pool(changed, y, **pool_options), eval_set=evaluation, use_best_model=False)
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot|parameters.*differ"):
        fit(saved | dict(max_ctr_complexity=3), pool, eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
def test_native_compound_initial_models_baselines_and_exact_recovery(tmp_path, boosting, count):
    x, y, future, pool_options = categorical_problem()
    config = options(boosting=boosting, count=count, loss_function="Logloss", iterations=5)
    pool = Pool(x, y, **pool_options)
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
    np.testing.assert_allclose(resumed.get_test_eval(),
                               resumed.predict(x, prediction_type="RawFormulaVal", task_type="GPU") + baseline,
                               atol=2e-6, rtol=5e-6)
    document = check_readers_and_oracle(resumed, x, y, future, tmp_path)
    info = document["features_info"]
    offset = sum(len(feature["borders"]) for feature in info.get("float_features", []))
    offset += sum(len(feature.get("values", [])) for feature in info.get("categorical_features", []))
    compound_splits = set()
    for ctr in info.get("ctrs", []):
        if len(ctr["elements"]) > 1:
            compound_splits.update(range(offset, offset + len(ctr["borders"])))
        offset += len(ctr["borders"])
    assert any(split["split_index"] in compound_splits
               for tree in document["oblivious_trees"][initial.tree_count_:]
               for split in tree["splits"] or [])


@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
def test_native_compound_best_model_and_early_stop_snapshot_cursor(tmp_path, boosting, count):
    x, y, future, pool_options = categorical_problem()
    train = Pool(x, y, **pool_options)
    evaluation = Pool(x, 1 - y, **pool_options)
    config = options(boosting=boosting, count=count, iterations=12, learning_rate=.4)
    direct = fit(config, train, eval_set=evaluation, use_best_model=True)
    assert direct.tree_count_ == direct.get_best_iteration() + 1 < config["iterations"]
    saved = snapshot_options(config, tmp_path)
    fit(saved, train, eval_set=evaluation, use_best_model=True, callbacks=[StopAfter(3)])
    resumed = fit(saved, train, eval_set=evaluation, use_best_model=True)
    check_exact(resumed, direct, future)
    early = fit(config, train, eval_set=evaluation, use_best_model=True, early_stopping_rounds=2)
    assert len(early.get_evals_result()["learn"]["RMSE"]) < config["iterations"]
    assert early.tree_count_ == early.get_best_iteration() + 1
    early_saved = snapshot_options(config, tmp_path / "early")
    fit(early_saved, train, eval_set=evaluation, use_best_model=True,
        early_stopping_rounds=2, callbacks=[StopAfter(1)])
    early_resumed = fit(early_saved, train, eval_set=evaluation, use_best_model=True, early_stopping_rounds=2)
    check_exact(early_resumed, early, future)
    early_completed = fit(early_saved, train, eval_set=evaluation, use_best_model=True, early_stopping_rounds=2)
    check_exact(early_completed, early, future)
    check_readers_and_oracle(resumed, x, y, future, tmp_path)


@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
def test_native_default_complexity_one_keeps_existing_simple_ctr_path(tmp_path, boosting, count):
    x, y, future, pool_options = categorical_problem()
    pool = Pool(x, y, **pool_options)
    config = options(boosting=boosting, count=count, complexity=1, iterations=4,
                     data_partition="DocParallel" if boosting == "Plain" else "FeatureParallel")
    explicit = fit(config, pool, eval_set=pool, use_best_model=False)
    default = dict(config)
    default.pop("max_ctr_complexity")
    # Different dormant combination options must not influence simple CTR
    # training or accidentally activate dynamic compound feature generation.
    default["combinations_ctr"] = ["FeatureFreq:Prior=0"]
    implicit = fit(default, pool, eval_set=pool, use_best_model=False)
    assert implicit.get_all_params()["max_ctr_complexity"] == 1
    check_exact(implicit, explicit, future)
    document = exported(implicit, tmp_path / "default-simple.json")
    assert projections(document) and all(len(projection) == 1 for projection in projections(document))


@pytest.mark.parametrize("loss", ("Quantile:alpha=0.7", "MAE", "MAPE"))
@pytest.mark.parametrize("count", (1, 4))
def test_native_plain_compounds_preserve_exact_leaf_estimation(tmp_path, loss, count):
    x, y, future, pool_options = categorical_problem()
    y = .125 + .75 * y
    config = options(count=count, loss_function=loss, leaf_estimation_method="Exact")
    model = fit(config, Pool(x, y, **pool_options))
    assert model.get_all_params()["leaf_estimation_method"] == "Exact"
    check_readers_and_oracle(model, x, y, future, tmp_path)


@pytest.mark.parametrize("overrides,error", (
    ({"data_partition": "DocParallel"}, "compound CTRs require.*FeatureParallel"),
    ({"grow_policy": "Depthwise", "data_partition": "DocParallel"}, "compound CTRs support scalar pointwise.*symmetric"),
    ({"grow_policy": "Lossguide", "max_leaves": 5, "data_partition": "DocParallel"}, "compound CTRs support scalar pointwise.*symmetric"),
    ({"grow_policy": "Region", "data_partition": "DocParallel"}, "compound CTRs support scalar pointwise.*symmetric"),
    ({"counter_calc_method": "Full"}, "learn-only CTR|SkipTest"),
    ({"max_ctr_complexity": 32}, "max ctr complexity"),
    ({"combinations_ctr": ["Borders:PriorEstimation=BetaPrior"]}, "(?i)prior estimation|prior.*unsupported|unsupported.*prior"),
    ({"combinations_ctr": ["Counter:Prior=0.5"]}, "(?i)CTR.*(support|implement)|support.*CTR"),
    ({"boosting_type": "Ordered", "loss_function": "MAE", "leaf_estimation_method": "Exact"},
     "(?i)Ordered.*(Newton or Gradient|Exact)|Exact.*ordered"),
))
def test_native_unsupported_compound_options_reject_explicitly(overrides, error):
    x, y, _, pool_options = categorical_problem()
    with pytest.raises(CatBoostError, match=error):
        fit(options() | overrides, Pool(x, y, **pool_options))


@pytest.mark.parametrize("objective", ("MultiClass", "MultiRMSE", "QueryRMSE", "PairLogit", "PairLogitPairwise"))
def test_native_unsupported_compound_objective_families_reject_explicitly(objective):
    x, y, _, pool_options = categorical_problem()
    if objective == "MultiRMSE":
        y = np.column_stack((y, 1 - y))
    elif objective in ("QueryRMSE", "PairLogit", "PairLogitPairwise"):
        pool_options["group_id"] = np.zeros(len(x), dtype=np.uint64)
    config = options(loss_function=objective, data_partition="DocParallel")
    with pytest.raises(CatBoostError, match="compound CTRs support scalar pointwise.*symmetric"):
        CatBoost(config).fit(Pool(x, y, **pool_options))


@pytest.mark.parametrize("kind", CTR_KINDS)
@pytest.mark.parametrize("boosting", BOOSTING)
@pytest.mark.parametrize("count", (1, 4))
def test_native_group_histories_train_compounds_with_original_grouped_pools(tmp_path, kind, boosting, count):
    x, y, future, pool_options = categorical_problem()
    pool_options["group_id"] = np.arange(len(x), dtype=np.uint64) // 6
    if kind == "FloatTargetMeanValue":
        y = .125 + .75 * y
    config = options(kind, boosting, count, ctr_history_unit="Group")
    model = fit(config, Pool(x, y, **pool_options))
    assert model.get_metadata()["metal_permutations"] == str(count)
    assert model.get_all_params()["ctr_history_unit"] == "Group"
    check_readers_and_oracle(model, x, y, future, tmp_path)
