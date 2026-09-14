"""Inventoried card 4 CLI and public API release checks; every fit uses Metal."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


MODES = ("PlainDP", "PlainFP", "OrderedFP")
GREEDY = ("Depthwise", "Lossguide", "Region")
VECTORS = ("MultiClass", "MultiClassOneVsAll", "MultiRMSE", "RMSEWithUncertainty",
           "MultiLogloss", "MultiCrossEntropy")
FULL_MATRIX = ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise")


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def configurations():
    def case(family, mode, loss="RMSE", **extra):
        return dict(name=f"{family}-{mode}-{loss}", family=family, mode=mode, loss=loss, **extra)
    for mode in MODES:
        for loss in ("RMSE", "Logloss"):
            yield case("simple", mode, loss)
    for mode in GREEDY:
        for loss in ("RMSE", "MultiClass"):
            yield case("simple", mode, loss)
    for loss in VECTORS:
        yield case("simple-vector", "PlainDP", loss)
    for loss in FULL_MATRIX:
        yield case("simple-matrix", "PlainDP", loss)
        yield case("rsm", "PlainDP", loss)
    for mode in GREEDY:
        yield case("fixed", mode)
    for mode in (*MODES, *GREEDY):
        yield case("weights", mode)
    for mode in MODES:
        yield case("prior", mode, "Logloss")
        yield case("full-counter", mode)
        yield case("onehot256", mode)
        yield case("text", mode, "Logloss")
        yield case("embedding", mode, "Logloss")
        yield case("normalization", mode)
        yield case("ridge", mode)
    yield case("meta", "PlainFP")
    yield case("ridge", "PlainDP", "QueryCrossEntropy")
    for mode in (*MODES, "Depthwise"):
        yield case("langevin", mode)
    yield case("langevin-noop", "Lossguide")
    for mode in ("PlainFP", "OrderedFP"):
        for kind, loss in (("regressor", "RMSE"), ("classifier", "Logloss"), ("ranker", "QueryRMSE")):
            yield case("public-" + kind, mode, loss, api_only=True)
    for mode in ("PlainDP", "OrderedFP"):
        yield case("cv", mode, api_only=True)


def inventory():
    cases = list(configurations())
    assert len({item["name"] for item in cases}) == len(cases)
    cli = [item for item in cases if not item.get("api_only")]
    return dict(schema=1, expected_cases_cli=len(cli), expected_cases_smoke=len(cases),
                families_cli=dict(Counter(item["family"] for item in cli)),
                families_smoke=dict(Counter(item["family"] for item in cases)), cases=cases)


def options(case):
    mode, family, loss = case["mode"], case["family"], case["loss"]
    result = dict(task_type="GPU", loss_function=loss, iterations=3, depth=2, learning_rate=.2,
        l2_leaf_reg=2, random_seed=713, bootstrap_type="No", random_strength=0,
        score_function="Cosine", leaf_estimation_method="Newton", leaf_estimation_iterations=2,
        leaf_estimation_backtracking="No", boost_from_average=False, border_count=7,
        data_partition="FeatureParallel" if mode in ("PlainFP", "OrderedFP") else "DocParallel",
        boosting_type="Ordered" if mode == "OrderedFP" else "Plain",
        grow_policy=mode if mode in GREEDY else "SymmetricTree", max_ctr_complexity=1,
        permutation_count=1, has_time=True, one_hot_max_size=2,
        metric_period=1, logging_level="Silent", allow_writing_files=False)
    if mode == "OrderedFP":
        result.update(min_fold_size=8, fold_len_multiplier=1.7)
    if mode == "Lossguide":
        result["max_leaves"] = 4
    if family.startswith("simple"):
        result.update(leaf_estimation_method="Simple", leaf_estimation_iterations=1)
    if family in ("simple-matrix", "rsm"):
        result["score_function"] = "NewtonL2"
    if family == "rsm":
        result["rsm"] = .3
    if family == "fixed":
        result.update(fixed_binary_splits=[0], depth=3)
    if family == "weights":
        result.update(feature_weights={0: .001, 1: 1}, depth=1)
    if family == "prior":
        result.update(simple_ctr=["Borders:CtrBorderCount=7:Prior=0.5:PriorEstimation=BetaPrior"],
                      ctr_target_border_count=1, model_size_reg=0)
    if family == "full-counter":
        result.update(simple_ctr=["FeatureFreq:CtrBorderType=Uniform:CtrBorderCount=15:Prior=0.5"],
                      counter_calc_method="Full", iterations=1, depth=1, learning_rate=1,
                      l2_leaf_reg=0, leaf_estimation_iterations=1, model_size_reg=0)
    if family == "onehot256":
        result.update(one_hot_max_size=256, iterations=1, depth=1)
    if family == "text":
        result["text_processing"] = dict(
            tokenizers=[dict(tokenizer_id="Space", delimiter=" ")],
            dictionaries=[dict(dictionary_id="Word", token_level_type="Word", occurrence_lower_bound="1")],
            feature_processing={"default": [dict(tokenizers_names=["Space"], dictionaries_names=["Word"],
                feature_calcers=["BoW", "NaiveBayes", "BM25"])]})
    if family == "embedding":
        result["embedding_calcers"] = ["LDA", "KNN:k=3"]
    if family == "normalization":
        result["fold_size_loss_normalization"] = True
    if family == "ridge":
        result.update(add_ridge_penalty_to_loss_function=True, leaf_estimation_backtracking="AnyImprovement")
        if loss in FULL_MATRIX:
            result["score_function"] = "NewtonL2"
    if family == "meta":
        result.update(score_function="L2", meta_l2_exponent=.7, meta_l2_frequency=.43)
    if family.startswith("langevin"):
        result.update(langevin=True, diffusion_temperature=10.)
        if mode in GREEDY:
            result.update(leaf_estimation_method="Gradient")
        if family == "langevin-noop":
            result.update(leaf_estimation_method="Simple", leaf_estimation_iterations=1)
    return result


def data(case):
    import numpy as np
    rng = np.random.default_rng(491)
    row = np.arange(128)
    x = rng.normal(size=(len(row), 4)).astype(np.float32)
    signal = 1.3 * x[:, 0] - .7 * x[:, 1] + .2 * x[:, 2]
    family, loss = case["family"], case["loss"]
    pool_options = {}
    if family in ("weights", "fixed"):
        x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), (32, 1))
        signal = (4 * x[:, 0] + x[:, 1]) if family == "weights" else (x[:, 0] + 20 * x[:, 1])
    if family == "prior":
        cats = np.repeat(np.arange(6), 40)
        positive = (3, 8, 13, 21, 29, 35)
        y = np.concatenate([np.r_[np.ones(count), np.zeros(40 - count)] for count in positive]).astype(np.float32)
        order = np.random.default_rng(318).permutation(len(cats))
        x, y = np.array([[f"cat{value}"] for value in cats], object)[order], y[order]
        pool_options["cat_features"] = [0]
    elif family == "full-counter":
        x = np.repeat(np.array(["a", "b", "c", "d"], object), [12, 24, 36, 48])[:, None]
        x = x[np.random.default_rng(831).permutation(len(x))]
        y = (x[:, 0] == "a").astype(np.float32)
        pool_options["cat_features"] = [0]
    elif family == "onehot256":
        category = np.tile(np.arange(256), 2)
        x = np.array([[f"cat{value}"] for value in category], object)
        y = (category == 255).astype(np.float32)
        pool_options["cat_features"] = [0]
    elif family == "text":
        y = (np.arange(120) * 17 % 11 >= 5).astype(np.float32)
        x = np.array([["excellent bright pleasant common" if label else "poor dark awful common"]
                      for label in y], object)
        pool_options["text_features"] = [0]
    elif family == "embedding":
        x = np.empty((len(row), 1), object)
        vectors = rng.normal(size=(len(row), 4)).astype(np.float32)
        x[:, 0] = list(vectors)
        y = (vectors[:, 0] - .7 * vectors[:, 1] > 0).astype(np.float32)
        pool_options["embedding_features"] = [0]
    elif loss in ("MultiClass", "MultiClassOneVsAll"):
        y = np.digitize(signal, [-.4, .5]).astype(np.float32)
    elif loss in ("MultiRMSE", "MultiLogloss", "MultiCrossEntropy"):
        y = np.column_stack((signal + .2, -.6 * signal + x[:, 2])).astype(np.float32)
        if loss == "MultiLogloss": y = (y > 0).astype(np.float32)
        if loss == "MultiCrossEntropy": y = (1 / (1 + np.exp(-y))).astype(np.float32)
    elif loss == "Logloss":
        y = (signal > 0).astype(np.float32)
    elif loss in FULL_MATRIX:
        y = (.1 + .8 / (1 + np.exp(-signal))).astype(np.float32)
    else:
        y = signal.astype(np.float32)
    pool_options["weight"] = (.5 + np.arange(len(y)) % 7 / 5).astype(np.float32)
    if loss.startswith(("Query", "Pair", "Yeti")):
        pool_options["group_id"] = np.arange(len(y)) // 8
    if loss == "PairLogitPairwise":
        edges = []
        for start in range(0, len(y), 8):
            ranked = np.argsort(y[start:start + 8]) + start
            edges.extend((int(a), int(b)) for a, b in zip(ranked[1:], ranked[:-1]))
        pool_options["pairs"] = np.asarray(edges, np.uint32)
        pool_options["pairs_weight"] = np.linspace(.5, 1.5, len(edges), dtype=np.float32)
    return x, y, pool_options


def evaluation_data(case, learn):
    import numpy as np
    if case["family"] != "full-counter":
        return learn
    x = np.repeat(np.array(["a", "b", "c", "d", "eval-only"], object), [60, 2, 2, 2, 18])[:, None]
    y = (x[:, 0] == "a").astype(np.float32)
    return x, y, dict(cat_features=[0], weight=np.ones(len(x), np.float32))


def write_cli_data(folder, x, y, pool_options, prefix="learn"):
    import numpy as np
    def literal(value):
        return repr(value.item()) if isinstance(value, (np.integer, np.floating)) else str(value)
    targets = np.asarray(y).reshape(len(y), -1)
    columns = [f"{i}\tTarget" for i in range(targets.shape[1])]
    columns.append(f"{len(columns)}\tWeight")
    if "group_id" in pool_options:
        columns.append(f"{len(columns)}\tGroupId")
    offset = len(columns)
    types = {}
    for key, kind in (("cat_features", "Categ"), ("text_features", "Text"), ("embedding_features", "NumVector")):
        types.update({i: kind for i in pool_options.get(key, [])})
    columns.extend(f"{offset + i}\t{types.get(i, 'Num')}\tfeature{i}" for i in range(x.shape[1]))
    lines = []
    for row in range(len(y)):
        values = [*targets[row], pool_options["weight"][row]]
        if "group_id" in pool_options: values.append(pool_options["group_id"][row])
        values.extend(";".join(map(literal, value)) if types.get(i) == "NumVector" else value
                      for i, value in enumerate(x[row]))
        assert all("\t" not in str(value) and "\n" not in str(value) for value in values)
        lines.append("\t".join(map(literal, values)))
    (folder / (prefix + ".tsv")).write_text("\n".join(lines) + "\n")
    description = "\n".join(columns) + "\n"
    if (folder / "columns.cd").exists():
        assert (folder / "columns.cd").read_text() == description, "learn/eval column schemas differ"
    (folder / "columns.cd").write_text(description)
    if "pairs" in pool_options:
        pair_file = "pairs.tsv" if prefix == "learn" else prefix + "-pairs.tsv"
        (folder / pair_file).write_text("".join(f"{a}\t{b}\t{literal(weight)}\n" for (a, b), weight in
            zip(pool_options["pairs"], pool_options["pairs_weight"])))


def json_reader_check(model, restored, pool, expected):
    """Bound decimal-parser leaf rounding and both binary64 tree summations."""
    import numpy as np
    from catboost import CatBoost
    # A preceding GPU prediction selects that evaluator on the model. Route
    # leaf inspection explicitly through the shared CPU model reader.
    np.testing.assert_array_equal(CatBoost.predict(model, pool, prediction_type="RawFormulaVal", task_type="CPU"), expected)
    actual = np.asarray(restored.predict(pool, prediction_type="RawFormulaVal", task_type="CPU"))
    counts = model.get_tree_leaf_counts()
    np.testing.assert_array_equal(restored.get_tree_leaf_counts(), counts)
    leaves, roundtrip = model.get_leaf_values(), restored.get_leaf_values()
    assert np.isfinite(leaves).all() and np.isfinite(roundtrip).all()
    indexes, roundtrip_indexes = model.calc_leaf_indexes(pool), restored.calc_leaf_indexes(pool)
    assert model._object._is_oblivious() == restored._object._is_oblivious()
    def canonical_paths(reader, tree, count):
        if reader._object._is_oblivious():
            return {leaf: leaf for leaf in range(count)}
        steps, node_leaves = reader._get_tree_step_nodes(tree), reader._get_tree_node_to_leaf(tree)
        paths = {}
        def visit(node, path):
            if steps[node] == (0, 0):
                paths[path] = node_leaves[node]
            else:
                for side, step in enumerate(steps[node]):
                    if step == 0:
                        paths[path + (side,)] = node_leaves[node]
                    else:
                        visit(node + step, path + (side,))
        visit(0, ())
        assert sorted(paths.values()) == list(range(count))
        return paths
    scale, bias = model.get_scale_and_bias()
    roundtrip_scale, roundtrip_bias = restored.get_scale_and_bias()
    assert roundtrip_scale == scale
    np.testing.assert_array_equal(roundtrip_bias, bias)
    dimension = len(leaves) // int(np.sum(counts))
    leaves, roundtrip = leaves.reshape(-1, dimension), roundtrip.reshape(-1, dimension)
    delta = np.zeros((len(indexes), dimension), np.float64)
    magnitude = np.zeros_like(delta)
    offset, maximum_leaf_delta = 0, 0.
    for tree, count in enumerate(counts):
        # JSON expands parent-packed greedy leaves and can reorder storage.
        # Compare values and routing by branch path, not by storage position.
        original_paths = canonical_paths(model, tree, int(count))
        restored_paths = canonical_paths(restored, tree, int(count))
        assert original_paths.keys() == restored_paths.keys()
        original_order = np.asarray([original_paths[path] for path in sorted(original_paths)])
        restored_order = np.asarray([restored_paths[path] for path in sorted(original_paths)])
        original_values, restored_values = leaves[offset + original_order], roundtrip[offset + restored_order]
        # The shared decimal parser can round a leaf by one binary64 ULP.
        np.testing.assert_array_max_ulp(restored_values, original_values, maxulp=1)
        maximum_leaf_delta = max(maximum_leaf_delta, float(np.max(np.abs(restored_values - original_values))))
        original_inverse, restored_inverse = np.argsort(original_order), np.argsort(restored_order)
        np.testing.assert_array_equal(original_inverse[indexes[:, tree]], restored_inverse[roundtrip_indexes[:, tree]])
        selected, selected_roundtrip = offset + indexes[:, tree], offset + roundtrip_indexes[:, tree]
        delta += np.abs(roundtrip[selected_roundtrip] - leaves[selected])
        magnitude += np.abs(leaves[selected]) + np.abs(roundtrip[selected_roundtrip])
        offset += int(count)
    # gamma_n bounds each sum plus scale/bias application. Include magnitudes
    # from both readers, so near-zero predictions need no broad absolute floor.
    n_epsilon = (len(counts) + 2) * np.finfo(np.float64).eps
    gamma = n_epsilon / (1 - n_epsilon)
    bound = abs(scale) * delta + gamma * (abs(scale) * magnitude + 2 * np.abs(bias))
    error = np.abs(actual.reshape(-1, dimension) - np.asarray(expected).reshape(-1, dimension))
    assert np.isfinite(actual).all() and np.all(error <= bound), \
        f"JSON prediction error {error.max()} exceeds its leaf/summation bound {bound.max()}"
    return dict(leaf_max_ulp_allowed=1, leaf_max_abs_delta=maximum_leaf_delta,
                prediction_max_abs_delta=float(error.max()), prediction_max_bound=float(bound.max()))


def read_checks(model, pool, folder, case):
    import numpy as np
    from catboost import CatBoost, CatBoostError
    assert model.get_metadata()["metal_backend"] == "METAL"
    raw = dict(prediction_type="RawFormulaVal")
    # Typed rankers omit prediction_type from their convenience signature;
    # the shared base reader exposes the raw-value contract for every model.
    expected = CatBoost.predict(model, pool, **raw)
    assert np.isfinite(expected).all()
    estimated = case["family"] in ("text", "embedding")
    if estimated:
        assert np.ptp(expected) > 1e-7
        try:
            CatBoost.predict(model, pool, task_type="GPU", **raw)
        except CatBoostError as error:
            assert any(token in str(error).lower() for token in ("numeric/categorical", "estimated", "text", "embedding")), str(error)
        else:
            raise AssertionError("GPU prediction must retain CUDA's explicit estimated-feature boundary")
    else:
        np.testing.assert_allclose(CatBoost.predict(model, pool, task_type="GPU", **raw), expected, rtol=8e-6, atol=3e-6)
    json_roundtrip = None
    for fmt in (("cbm",) if estimated else ("cbm", "json")):
        path = folder / ("export." + fmt)
        model.save_model(path, format=fmt)
        restored = CatBoost().load_model(path, format=fmt)
        if fmt == "cbm":
            np.testing.assert_array_equal(restored.predict(pool, **raw), expected)
        else:
            json_roundtrip = json_reader_check(model, restored, pool, expected)
        if not estimated:
            np.testing.assert_allclose(restored.predict(pool, task_type="GPU", **raw), expected, rtol=8e-6, atol=3e-6)
    np.savez(folder / "outputs.npz", predictions=expected, leaf_values=model.get_leaf_values(),
             leaf_weights=model.get_leaf_weights())
    family = case["family"]
    if family.startswith("simple"):
        assert model.get_all_params()["leaf_estimation_method"] == "Simple"
    if family in ("fixed", "weights"):
        document = json.loads((folder / "export.json").read_text())
        trees = document.get("oblivious_trees", document.get("trees"))
        for tree in trees:
            split = tree["splits"][0] if "splits" in tree else tree["split"]
            assert split["float_feature_index"] == (0 if family == "fixed" else 1)
    if family == "prior":
        priors = json.loads(model.get_metadata()["params"])["cat_feature_params"]["per_feature_ctrs"]["0"][0]["priors"]
        np.testing.assert_allclose(priors, [[1.2275301, 2.697511]], rtol=3e-6, atol=3e-6)
    if family == "rsm":
        assert model.get_metadata()["metal_rsm_rng"] == "cuda_single_device_host_shadow_v1"
    if family == "onehot256":
        document = json.loads((folder / "export.json").read_text())
        assert not document["features_info"].get("ctrs")
        assert document["oblivious_trees"][0]["splits"][0]["split_type"] == "OneHotFeature"
        future = [["cat255"], ["cat0"], ["never-seen"]]
        known_and_unseen = model.predict(future, task_type="GPU", **raw)
        assert known_and_unseen[0] > known_and_unseen[1]
        assert known_and_unseen[1] == known_and_unseen[2]
    return dict(trees=model.tree_count_, inference="shared-cpu-reader" if estimated else "cpu-and-metal",
                estimated_gpu_rejection_checked=estimated, json_roundtrip=json_roundtrip)


def cli_parameters(config):
    """Exercise the named CLI options, retaining other settings in JSON."""
    params, flags = dict(config), []
    scalar = {"leaf_estimation_method": "--leaf-estimation-method", "rsm": "--rsm",
              "meta_l2_exponent": "--meta-l2-leaf-exponent", "meta_l2_frequency": "--meta-l2-leaf-frequency",
              "counter_calc_method": "--counter-calc-method", "one_hot_max_size": "--one-hot-max-size",
              "diffusion_temperature": "--diffusion-temperature"}
    for name, flag in scalar.items():
        if name in params: flags.extend((flag, str(params.pop(name))))
    if "langevin" in params:
        flags.extend(("--langevin", str(params.pop("langevin")).lower()))
    for name, flag in (("fold_size_loss_normalization", "--fold-size-loss-normalization"),
                       ("add_ridge_penalty_to_loss_function", "--add-ridge-penalty-for-loss-function")):
        if params.get(name):
            params.pop(name)
            flags.append(flag)
    if "fixed_binary_splits" in params:
        flags.extend(("--fixed-binary-splits", ":".join(map(str, params.pop("fixed_binary_splits")))))
    if "feature_weights" in params:
        flags.extend(("--feature-weights", ",".join(f"{key}:{value}" for key, value in params.pop("feature_weights").items())))
    if "simple_ctr" in params:
        flags.extend(("--simple-ctr", ",".join(params.pop("simple_ctr"))))
    return params, flags


def fit_cli(args, folder, config, learn, evaluation):
    from catboost import CatBoost
    write_cli_data(folder, *learn)
    write_cli_data(folder, *evaluation, prefix="test")
    params, flags = cli_parameters(config)
    (folder / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    command = [str(args.cli), "fit", "--task-type", "GPU", "--params-file", str(folder / "params.json"),
        "--learn-set", str(folder / "learn.tsv"), "--test-set", str(folder / "test.tsv"),
        "--column-description", str(folder / "columns.cd"), "--use-best-model", "false",
        "--allow-writing-files", "false", "--model-file", str(folder / "model.cbm"),
        "--eval-file", str(folder / "eval.tsv"), "--output-columns", "RawFormulaVal", *flags]
    if "pairs" in learn[2]:
        command += ["--learn-pairs", str(folder / "pairs.tsv"), "--test-pairs", str(folder / "test-pairs.tsv")]
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (folder / "fit.log").open("w") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    return CatBoost().load_model(folder / "model.cbm")


def public_fit(case, folder, x, y, pool_options):
    import numpy as np
    from catboost import Pool
    from catboost_metal import CatBoostMetalRegressor, CatBoostMetalClassifier, CatBoostMetalRanker
    classes = {"public-regressor": CatBoostMetalRegressor, "public-classifier": CatBoostMetalClassifier,
               "public-ranker": CatBoostMetalRanker}
    config = dict(iterations=3, depth=2, learning_rate=.2, random_seed=713, border_count=7,
        data_partition="FeatureParallel", boosting_type="Ordered" if case["mode"] == "OrderedFP" else "Plain",
        bootstrap_type="No", random_strength=0, score_function="Cosine", boost_from_average=False,
        permutation_count=4, min_fold_size=8, feature_weights={0: .25},
        simple_ctr=["Borders:CtrBorderCount=7:Prior=0.5"], leaf_estimation_method="Simple",
        leaf_estimation_iterations=1, leaf_estimation_backtracking="No")
    if case["family"] == "public-ranker":
        config.pop("boost_from_average")  # The ranking constructor fixes this itself.
    (folder / "constructor.json").write_text(json.dumps(config, indent=2) + "\n")
    model = classes[case["family"]](**config)
    pool = Pool(x, y, **pool_options)
    model.fit(pool, eval_set=pool, use_best_model=False)
    assert model._native_bridge_fitted and model.training_stats_["backend"] == "METAL"
    assert model.training_predictions_.shape == (len(y),) and np.isfinite(model.training_predictions_).all()
    return model.to_catboost()


def full_counter_checks(model, folder, learn, evaluation, command):
    import numpy as np
    document = json.loads((folder / "export.json").read_text())
    ctr, = document["features_info"]["ctrs"]
    assert ctr["ctr_type"] == "FeatureFreq"
    tree, = document["oblivious_trees"]
    split, = tree["splits"]
    assert split["split_type"] == "OnlineCtr"
    x, y, pool_options = learn
    eval_x = evaluation[0]
    def leaf_ids(source, values):
        counts = Counter(source[:, 0])
        frequency = np.asarray([(counts[value] + ctr["prior_numerator"]) /
            (len(source) + ctr["prior_denomerator"]) for value in values[:, 0]], np.float32)
        scaled = np.float32((frequency + np.float32(ctr["shift"])) * np.float32(ctr["scale"]))
        return (scaled > np.float32(split["border"])).astype(np.uint32)
    full_rows = np.concatenate((x, eval_x))
    train_leaf = leaf_ids(full_rows, x)
    assert set(train_leaf) == {0, 1}
    masses = [pool_options["weight"][train_leaf == i].sum(dtype=float) for i in (0, 1)]
    values = [np.sum(pool_options["weight"][train_leaf == i] * y[train_leaf == i], dtype=float) / masses[i]
              for i in (0, 1)]
    np.testing.assert_allclose(model.get_leaf_values(), values, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), masses, rtol=3e-6, atol=3e-6)
    final_cursor = np.asarray(values)[leaf_ids(x, eval_x)]
    full_cursor = np.asarray(values)[leaf_ids(full_rows, eval_x)]
    actual_eval = (np.loadtxt(folder / "eval.tsv", skiprows=1, ndmin=1) if command == "cli"
                   else np.asarray(model.get_test_eval()))
    np.testing.assert_allclose(actual_eval, full_cursor, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(model.predict(eval_x, task_type="GPU"), final_cursor, rtol=4e-6, atol=4e-6)
    assert np.max(np.abs(full_cursor - final_cursor)) > .1
    return dict(full_eval_counter_oracle=True, final_learn_only_counter_oracle=True,
                eval_cursor_source="cli-eval-file" if command == "cli" else "native-get_test_eval")


def cv_case(case, folder, pool, config):
    import numpy as np
    from catboost import cv
    rows = np.arange(pool.num_row())
    a, b = rows[rows % 5 < 2], rows[rows % 5 >= 2]
    folds = [(b, a), (a, b)]
    common = dict(as_pandas=False, folds=folds, shuffle=False, stratified=False)
    config = config | dict(allow_writing_files=True)
    metrics_params, model_params = [config | dict(train_dir=str(folder / name)) for name in ("metrics", "models")]
    (folder / "metrics-params.json").write_text(json.dumps(metrics_params, indent=2) + "\n")
    (folder / "models-params.json").write_text(json.dumps(model_params, indent=2) + "\n")
    (folder / "folds.json").write_text(json.dumps([[a.tolist(), b.tolist()] for a, b in folds], indent=2) + "\n")
    metrics = cv(pool, metrics_params, **common)
    returned, models = cv(pool, model_params, return_models=True, **common)
    assert len(models) == 2 and metrics.keys() == returned.keys()
    for name in metrics:
        np.testing.assert_array_equal(metrics[name], returned[name])
        assert len(metrics[name]) == config["iterations"] and np.isfinite(metrics[name]).all()
    errors = []
    for model, (_, heldout) in zip(models, folds):
        assert model.get_metadata()["metal_backend"] == "METAL"
        subset = pool.slice(heldout)
        raw = model.predict(subset)
        errors.append(np.sqrt(np.average((raw - np.asarray(subset.get_label(), float)) ** 2,
                                         weights=subset.get_weight())))
    np.testing.assert_allclose(metrics["test-RMSE-mean"][-1], np.mean(errors), rtol=5e-6, atol=5e-6)
    (folder / "history.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return dict(folds=2, metrics_only_equal=True, independent_fold_metric=True, trees=config["iterations"],
                metrics_parameters_sha256=sha256(folder / "metrics-params.json"),
                model_parameters_sha256=sha256(folder / "models-params.json"), folds_sha256=sha256(folder / "folds.json"))


def execute(args, report, cases):
    from catboost import CatBoost, Pool
    for case in cases:
        folder = args.output_dir / case["name"]
        folder.mkdir()
        config = options(case)
        (folder / "params.json").write_text(json.dumps(config, indent=2) + "\n")
        learn = data(case)
        x, y, pool_options = learn
        evaluation = evaluation_data(case, learn)
        pool = Pool(x, y, **pool_options)
        evaluation_pool = Pool(evaluation[0], evaluation[1], **evaluation[2])
        if case["family"] == "cv":
            result = cv_case(case, folder, pool, config)
        else:
            if args.command == "cli":
                model = fit_cli(args, folder, config, learn, evaluation)
            elif case["family"].startswith("public-"):
                model = public_fit(case, folder, x, y, pool_options)
            else:
                model = CatBoost(config).fit(pool, eval_set=evaluation_pool, use_best_model=False)
            assert model.tree_count_ == config["iterations"]
            result = read_checks(model, evaluation_pool, folder, case)
            if case["family"] == "full-counter":
                result.update(full_counter_checks(model, folder, learn, evaluation, args.command))
            if case["family"].startswith("langevin"):
                assert model.get_all_params()["langevin"] is True
                assert model.get_all_params()["diffusion_temperature"] == 10.
            if case["family"] == "langevin-noop":
                import numpy as np
                disabled = folder / "disabled"
                disabled.mkdir()
                baseline_config = config | dict(langevin=False)
                if args.command == "cli":
                    baseline = fit_cli(args, disabled, baseline_config, learn, evaluation)
                else:
                    (disabled / "params.json").write_text(json.dumps(baseline_config, indent=2) + "\n")
                    baseline = CatBoost(baseline_config).fit(pool, eval_set=evaluation_pool, use_best_model=False)
                assert baseline.get_all_params().get("langevin", False) is False
                for method in ("get_leaf_values", "get_leaf_weights"):
                    np.testing.assert_array_equal(getattr(baseline, method)(), getattr(model, method)())
                np.testing.assert_array_equal(baseline.predict(evaluation_pool, task_type="GPU"),
                                              model.predict(evaluation_pool, task_type="GPU"))
                result.update(source_simple_noop_equal=True, positive_temperature_did_not_enable_langevin=True)
            if case["family"].startswith("public-"):
                result["constructor_sha256"] = sha256(folder / "constructor.json")
        report["cases"].append(case | result | dict(parameters_sha256=sha256(folder / "params.json")))
        print("PASS " + args.command + " Metal: " + case["name"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "cli", "smoke"))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--standalone", type=Path, default=Path(__file__).resolve().parents[2] / "python")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    listing = inventory()
    if args.command == "list":
        print(json.dumps(listing, indent=2))
        return
    if args.package is None or args.output_dir is None:
        parser.error("--package and --output-dir are required")
    if args.command == "cli" and (args.cli is None or not args.cli.is_file()):
        parser.error("--cli must name the frozen executable")
    for name, value in vars(args).items():
        if isinstance(value, Path): setattr(args, name, value.resolve())
    sys.path[:0] = [str(args.standalone), str(args.package)]
    import catboost
    from catboost import core
    assert Path(catboost.__file__).resolve().is_relative_to(args.package), "wrong native package imported"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = [item for item in listing["cases"] if args.command == "smoke" or not item.get("api_only")]
    report = dict(schema=1, command=args.command, status="running", cases=[], skipped_cases=0,
        expected_cases=listing["expected_cases_" + args.command], inventory=listing,
        package=str(Path(catboost.__file__).resolve()), python=sys.executable, started_unix=time.time(),
        source_sha256=sha256(__file__), extension_sha256=sha256(Path(catboost.__file__).parent / "_catboost.so"))
    if args.command == "smoke":
        import catboost_metal
        assert Path(catboost_metal.__file__).resolve().is_relative_to(args.standalone), "wrong standalone package imported"
        report["standalone"] = str(Path(catboost_metal.__file__).resolve())
        report["standalone_sources_sha256"] = {str(path.relative_to(args.standalone)): sha256(path)
            for path in sorted((args.standalone / "catboost_metal").rglob("*.py"))}
    if args.command == "cli": report.update(cli=str(args.cli), cli_sha256=sha256(args.cli))
    original_fit, original_cv = catboost.CatBoost._fit, core._cv
    def checked_fit(self, *values, **kwargs):
        assert args.command == "smoke", "CLI acceptance must not fit through Python"
        assert self.get_params().get("task_type") == "GPU"
        return original_fit(self, *values, **kwargs)
    def checked_cv(*values, **kwargs):
        assert args.command == "smoke"
        assert kwargs.get("params", values[0] if values else {})["task_type"] == "GPU"
        return original_cv(*values, **kwargs)
    catboost.CatBoost._fit, core._cv = checked_fit, checked_cv
    try:
        execute(args, report, cases)
        assert len(report["cases"]) == report["expected_cases"] and report["skipped_cases"] == 0
        assert sha256(__file__) == report["source_sha256"], "runner changed during acceptance"
        for name, digest in report.get("standalone_sources_sha256", {}).items():
            assert sha256(args.standalone / name) == digest, "standalone source changed during acceptance: " + name
        report["completed_cases"] = len(report["cases"])
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        catboost.CatBoost._fit, core._cv = original_fit, original_cv
        report["elapsed_seconds"] = time.time() - report["started_unix"]
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['status']}: {len(report['cases'])}/{report['expected_cases']}; skipped={report['skipped_cases']}", flush=True)


if __name__ == "__main__":
    main()
