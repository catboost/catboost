"""Native Metal text/embedding training, final calcers and online histories.

Every fit executes task_type='GPU'. References are direct metrics and the shared
CBM inference reader; no CPU model is trained. GPU prediction for estimated
models retains CUDA's explicit unsupported boundary.
"""
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires native Metal estimated-feature integration",
)
PATHS = ("plain-doc", "plain-feature", "ordered-feature")


@pytest.fixture(autouse=True)
def require_metal_fit(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


class StopAfter:
    def __init__(self, count):
        self.count = count

    def after_iteration(self, info):
        return info.iteration < self.count


def options(path="plain-doc", count=4, **extra):
    config = dict(
        task_type="GPU", loss_function="Logloss", iterations=7, depth=3,
        boosting_type="Ordered" if path == "ordered-feature" else "Plain",
        data_partition="DocParallel" if path == "plain-doc" else "FeatureParallel",
        grow_policy="SymmetricTree", learning_rate=.2, random_seed=718,
        bootstrap_type="No", random_strength=0, score_function="Cosine",
        leaf_estimation_method="Newton", leaf_estimation_iterations=2,
        leaf_estimation_backtracking="No", l2_leaf_reg=2, boost_from_average=False,
        permutation_count=count, has_time=count == 1, border_count=16,
        max_ctr_complexity=1, one_hot_max_size=2, model_size_reg=0,
        counter_calc_method="SkipTest", verbose=False, metric_period=1,
        allow_writing_files=False,
    )
    if path == "ordered-feature":
        config.update(min_fold_size=8, fold_len_multiplier=1.7)
    return config | extra


def text_processing(calcers):
    return dict(
        tokenizers=[dict(tokenizer_id="Space", delimiter=" ")],
        dictionaries=[dict(dictionary_id="Word", token_level_type="Word", occurrence_lower_bound="1")],
        feature_processing={"default": [dict(tokenizers_names=["Space"], dictionaries_names=["Word"],
                                             feature_calcers=calcers)]},
    )


def text_problem(loss="Logloss", mixed=False):
    n = 120
    labels = (np.arange(n) * 17 % 11 >= 5).astype(np.float32)
    train = [["excellent bright pleasant common" if label else "poor dark awful common"] for label in labels]
    test_labels = labels[7:43].copy()
    # Retain the training vocabulary (including the constant common token).
    # CUDA retains constant estimated grids; missing that token can legitimately
    # send every evaluation object into an unseen zero-weight leaf.
    test = [["excellent bright pleasant common unknown" if label else "poor dark awful common unseen"] for label in test_labels]
    if mixed:
        for i, row in enumerate(train):
            row.extend([f"cat{i % 13}", float(i % 7)])
        for i, row in enumerate(test):
            row.extend([f"cat{i % 17}", float(i % 7)])
    if loss == "RMSE":
        labels = labels * 2.5 - .75
        test_labels = test_labels * 2.5 - .75
    params = dict(text_features=[0])
    if mixed:
        params["cat_features"] = [1]
    return Pool(train, labels, **params), Pool(test, test_labels, **params), test_labels


def embedding_problem(loss="Logloss"):
    rng = np.random.default_rng(733)
    vectors = rng.normal(size=(156, 4)).astype(np.float32)
    signal = vectors[:, 0] - .7 * vectors[:, 1] + .15 * vectors[:, 2]
    labels = (signal > 0).astype(np.float32) if loss == "Logloss" else signal.astype(np.float32)
    data = [[vector] for vector in vectors]
    return (Pool(data[:120], labels[:120], embedding_features=[0]),
            Pool(data[120:], labels[120:], embedding_features=[0]), labels[120:])


def fit(config, learn, test, **kwargs):
    return CatBoost(config).fit(learn, eval_set=test, use_best_model=False, **kwargs)


def check_eval(model, test, labels, loss):
    raw = np.asarray(model.predict(test, prediction_type="RawFormulaVal"))
    assert np.isfinite(raw).all()
    assert np.ptp(raw) > 1e-7, "the estimated feature must influence learned trees"
    history = model.get_evals_result()["validation"][loss]
    # The public reader applies the saved final calcers. Training evaluation
    # must use those same final features, rather than online learn features.
    for tree_count, recorded in enumerate(history, 1):
        prediction = np.asarray(model.predict(test, prediction_type="RawFormulaVal", ntree_end=tree_count))
        expected = (np.mean(np.logaddexp(0, prediction) - labels * prediction)
                    if loss == "Logloss" else np.sqrt(np.mean((prediction - labels) ** 2)))
        assert recorded == pytest.approx(expected, rel=3e-6, abs=3e-7)
    assert model.get_metadata()["metal_backend"] == "METAL"
    return raw


@pytest.mark.parametrize("calcer", ["BoW", "NaiveBayes", "BM25", "all"])
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("count", [1, 4])
def test_text_calcers_train_with_each_history_and_shared_final_model(tmp_path, calcer, path, count):
    learn, test, labels = text_problem()
    calcers = ["BoW", "NaiveBayes", "BM25"] if calcer == "all" else [calcer]
    model = fit(options(path, count, text_processing=text_processing(calcers)), learn, test)
    raw = check_eval(model, test, labels, "Logloss")
    assert model.get_text_feature_indices() == [0]
    if calcer != "BoW":
        assert int(model.get_metadata()["metal_permutations"]) == count
    saved = tmp_path / "text.cbm"
    model.save_model(saved)
    restored = CatBoost().load_model(saved)
    np.testing.assert_array_equal(restored.predict(test, prediction_type="RawFormulaVal"), raw)


@pytest.mark.parametrize("calcer", ["LDA", "KNN:k=3", "all"])
@pytest.mark.parametrize("loss", ["Logloss", "RMSE"])
@pytest.mark.parametrize("path", PATHS)
def test_embedding_calcers_train_and_export_with_final_calcers(tmp_path, calcer, loss, path):
    learn, test, labels = embedding_problem(loss)
    calcers = ["LDA", "KNN:k=3"] if calcer == "all" else [calcer]
    model = fit(options(path, embedding_calcers=calcers, loss_function=loss), learn, test)
    raw = check_eval(model, test, labels, loss)
    assert model.get_embedding_feature_indices() == [0]
    assert int(model.get_metadata()["metal_permutations"]) == 4
    saved = tmp_path / "embedding.cbm"
    model.save_model(saved)
    np.testing.assert_array_equal(CatBoost().load_model(saved).predict(test, prediction_type="RawFormulaVal"), raw)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("kind", ["text", "embedding"])
def test_greedy_estimated_features_keep_native_tree_layout(kind, policy):
    learn, test, labels = text_problem() if kind == "text" else embedding_problem()
    extra = dict(text_processing=text_processing(["BoW", "NaiveBayes"])) if kind == "text" else dict(embedding_calcers=["LDA", "KNN:k=3"])
    config = options(grow_policy=policy, **extra)
    if policy == "Lossguide":
        config["max_leaves"] = 6
    model = fit(config, learn, test)
    check_eval(model, test, labels, "Logloss")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", ["text", "embedding"])
def test_estimated_snapshot_restores_online_histories_exactly(tmp_path, path, kind):
    learn, test, labels = text_problem() if kind == "text" else embedding_problem()
    extra = dict(text_processing=text_processing(["BoW", "NaiveBayes"])) if kind == "text" else dict(embedding_calcers=["LDA", "KNN:k=3"])
    config = options(path, **extra)
    full = fit(config, learn, test)
    snapshot = config | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="estimated.snapshot",
                             train_dir=str(tmp_path), allow_writing_files=True)
    stopped = fit(snapshot, learn, test, callbacks=[StopAfter(3)])
    assert stopped.tree_count_ == 3
    resumed = fit(snapshot, learn, test)
    np.testing.assert_array_equal(resumed.predict(test, prediction_type="RawFormulaVal"),
                                  full.predict(test, prediction_type="RawFormulaVal"))
    assert resumed.get_evals_result() == full.get_evals_result()
    check_eval(resumed, test, labels, "Logloss")


@pytest.mark.parametrize("path", ["plain-feature", "ordered-feature"])
def test_text_estimated_splits_coexist_with_compound_ctr_tensors(path):
    learn, test, labels = text_problem(mixed=True)
    model = fit(options(path, text_processing=text_processing(["BoW", "NaiveBayes"]),
                        max_ctr_complexity=2, depth=4), learn, test)
    check_eval(model, test, labels, "Logloss")
    assert model.get_cat_feature_indices() == [1]
    assert model.get_text_feature_indices() == [0]


@pytest.mark.parametrize("kind", ["text", "embedding"])
def test_estimated_models_keep_cuda_gpu_prediction_boundary(kind):
    learn, test, _ = text_problem() if kind == "text" else embedding_problem()
    extra = dict(text_processing=text_processing(["BoW"])) if kind == "text" else dict(embedding_calcers=["LDA"])
    model = fit(options(iterations=2, **extra), learn, test)
    with pytest.raises(CatBoostError, match="(?i)numeric/categorical|text|embedding"):
        model.predict(test, task_type="GPU")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("loss", ["Logloss", "RMSE"])
def test_bow_matches_literal_token_presence_with_both_models_trained_on_metal(path, loss):
    # A single most-frequent token produces exactly its boolean membership;
    # use the independent numeric representation to check training bin layout.
    membership = (np.arange(120) % 4 != 0).astype(np.float32)
    labels = membership if loss == "Logloss" else 2 * membership - .5
    text = Pool([["bright" if value else "dark"] for value in membership], labels, text_features=[0])
    numeric = Pool(membership[:, None], labels)
    config = options(path, count=1, loss_function=loss, iterations=4, depth=1)
    actual = fit(config | dict(text_processing=text_processing(["BoW:top_tokens_count=1"])), text, text)
    reference = fit(config, numeric, numeric)
    np.testing.assert_allclose(actual.predict(text, prediction_type="RawFormulaVal"),
                               reference.predict(numeric, prediction_type="RawFormulaVal"), rtol=0, atol=2e-7)


def test_multiple_text_sources_dictionaries_embeddings_and_eval_subsets():
    rng = np.random.default_rng(88)
    n = 120
    labels = (np.arange(n) % 3 != 0).astype(np.float32)
    data = [["bright sky" if y else "dark cloud", "warm day" if y else "cold night",
             rng.normal(size=4).astype(np.float32) + y] for y in labels]
    params = dict(text_features=[0, 1], embedding_features=[2], feature_names=["review", "weather", "vector"])
    learn = Pool(data, labels, **params)
    test = Pool(data[:48], labels[:48], **params).slice(list(range(3, 45, 2)))
    processing = text_processing(["BoW", "NaiveBayes"])
    processing["dictionaries"].append(dict(dictionary_id="Bigram", token_level_type="Word", gram_order="2",
                                            occurrence_lower_bound="1"))
    processing["feature_processing"]["default"][0]["dictionaries_names"] = ["Word", "Bigram"]
    model = fit(options(text_processing=processing, embedding_calcers=["LDA", "KNN:k=3"]), learn, test)
    check_eval(model, test, labels[3:45:2], "Logloss")
    assert model.feature_names_ == ["review", "weather", "vector"]
    assert model.get_text_feature_indices() == [0, 1]
    assert model.get_embedding_feature_indices() == [2]


@pytest.mark.parametrize("kind", ["text", "embedding"])
def test_multiclass_estimated_calcers_keep_public_output_dimensions(kind):
    rng = np.random.default_rng(19)
    labels = np.arange(120) % 3
    if kind == "text":
        words = ["red apple warm", "blue ocean cold", "green forest mild"]
        data = [[words[y]] for y in labels]
        params = dict(text_features=[0])
        extra = dict(text_processing=text_processing(["BoW", "NaiveBayes", "BM25"]))
    else:
        data = [[rng.normal(size=4).astype(np.float32) + np.eye(3, 4)[y] * 2] for y in labels]
        params = dict(embedding_features=[0])
        extra = dict(embedding_calcers=["LDA", "KNN:k=3"])
    learn, test = Pool(data, labels, **params), Pool(data[:36], labels[:36], **params)
    model = fit(options(loss_function="MultiClass", **extra), learn, test)
    raw = np.asarray(model.predict(test, prediction_type="RawFormulaVal"))
    assert raw.shape == (36, 3) and np.isfinite(raw).all()
    for iteration, recorded in enumerate(model.get_evals_result()["validation"]["MultiClass"], 1):
        pred = np.asarray(model.predict(test, prediction_type="RawFormulaVal", ntree_end=iteration))
        maximum = pred.max(axis=1)
        log_sum = maximum + np.log(np.exp(pred - maximum[:, None]).sum(axis=1))
        expected = np.mean(log_sum - pred[np.arange(36), labels[:36]])
        assert recorded == pytest.approx(expected, rel=3e-6, abs=3e-7)


@pytest.mark.parametrize("kind", ["text", "embedding"])
def test_estimated_snapshot_rejects_changed_source_rows(tmp_path, kind):
    learn, test, _ = text_problem() if kind == "text" else embedding_problem()
    extra = dict(text_processing=text_processing(["BoW", "NaiveBayes"])) if kind == "text" else dict(embedding_calcers=["LDA", "KNN:k=3"])
    config = options(iterations=3, save_snapshot=True, snapshot_interval=0,
                     snapshot_file="source.snapshot", train_dir=str(tmp_path), allow_writing_files=True, **extra)
    fit(config, learn, test)
    snapshot = (tmp_path / "source.snapshot").read_bytes()
    if kind == "text":
        labels = np.asarray(learn.get_label(), dtype=np.float32)
        rows = [["excellent bright pleasant common" if y else "poor dark awful common"] for y in labels]
        rows[0] = ["excellent bright pleasant common"]
        changed = Pool(rows, labels, text_features=[0])
    else:
        rng = np.random.default_rng(733)
        vectors = rng.normal(size=(156, 4)).astype(np.float32)
        vectors[0, 0] += .75
        changed = Pool([[row] for row in vectors[:120]], learn.get_label(), embedding_features=[0])
    with pytest.raises(CatBoostError, match="(?i)snapshot|checksum|incompat|different"):
        fit(config, changed, test)
    assert (tmp_path / "source.snapshot").read_bytes() == snapshot


@pytest.mark.parametrize("kind", ["text", "embedding"])
def test_public_prequantized_pool_cannot_silently_omit_estimated_buckets(kind):
    learn, test, _ = text_problem() if kind == "text" else embedding_problem()
    extra = dict(text_processing=text_processing(["BoW"])) if kind == "text" else dict(embedding_calcers=["LDA"])
    model = fit(options(iterations=2, **extra), learn, test)
    # Pool quantization is shared preprocessing, not a CPU model fit.
    learn.quantize()
    assert learn.is_quantized()
    with pytest.raises(CatBoostError, match="(?i)unquantized Pool"):
        model.predict(learn, prediction_type="RawFormulaVal")


@pytest.mark.parametrize("kind", ["offline-text", "online-text", "embedding"])
@pytest.mark.parametrize("path", ["plain-feature", "ordered-feature"])
@pytest.mark.parametrize("count", [1, 4])
def test_estimated_feature_parallel_snapshot_counts_cuda_host_draws(tmp_path, kind, path, count):
    import struct

    # CUDA's FeatureParallel chooser draws once per iteration when P>2. The
    # first MirrorMapping bootstrap allocation consumes 1+65536 host draws even
    # for bootstrap No. A depth-one search always has exactly one attempt:
    # one independent-dataset draw, and another for online estimators at P>1.
    # This inspects native serialized host state, not a mirror of its helper.
    learn, test, _ = embedding_problem() if kind == "embedding" else text_problem()
    extra = (dict(embedding_calcers=["LDA"]) if kind == "embedding" else
             dict(text_processing=text_processing(["BoW" if kind == "offline-text" else "NaiveBayes"])))
    config = options(path, count, depth=1, iterations=4, save_snapshot=True,
                     snapshot_interval=0, snapshot_file="draws.snapshot", train_dir=str(tmp_path),
                     allow_writing_files=True, **extra)
    stopped = fit(config, learn, test, callbacks=[StopAfter(2)])
    online = kind != "offline-text"

    def check_state(model):
        # use_best_model=False, symmetric trees and no compound CTRs/stochastic
        # objective leave the v6 payload ending in its common Q/I/bool state.
        raw = (tmp_path / "draws.snapshot").read_bytes()
        draws, completed, initialized = struct.unpack_from("<QIB", raw, len(raw) - 13)
        permutations = int(model.get_metadata()["metal_permutations"])
        per_iteration = int(permutations > 2) + 1 + int(online and permutations > 1)
        assert completed == model.tree_count_
        assert initialized == 1
        assert draws == 65537 + completed * per_iteration

    check_state(stopped)
    resumed = fit(config, learn, test)
    check_state(resumed)
    full = fit(options(path, count, depth=1, iterations=4, **extra), learn, test)
    np.testing.assert_array_equal(resumed.predict(test, prediction_type="RawFormulaVal"),
                                  full.predict(test, prediction_type="RawFormulaVal"))


def test_estimated_buckets_shift_onehot_predicates_in_compound_ctr_tables(tmp_path):
    from test_native_compound_ctrs import mixed_problem

    x, labels, _, _ = mixed_problem("onehot")
    tone = (np.arange(len(x)) * 31 % 7 >= 3).astype(np.float32)
    labels = labels + .15 * tone
    rows = [[*row, "bright" if value else "dark"] for row, value in zip(x, tone)]
    pool = Pool(rows, labels, text_features=[2], cat_features=[0, 1],
                feature_names=["category", "side", "tone"])
    config = options("plain-feature", loss_function="RMSE", iterations=10, depth=4,
                     max_ctr_complexity=2, learning_rate=.3, text_processing=text_processing(["BoW"]))
    model = fit(config, pool, pool)
    descriptions = [split for tree in range(model.tree_count_) for split in model._get_tree_splits(tree, pool)]
    assert any("src_feature_id=" in split for split in descriptions)
    assert any("counter_type=" in split and " val = " in split for split in descriptions)
    # Numeric -> estimated -> one-hot -> CTR is the evaluator bucket order.
    # The old shared provider skipped the estimated buckets when addressing a
    # one-hot predicate inside a CTR; per-tree and full-model metrics diverged.
    raw = check_eval(model, pool, labels, "RMSE")
    path = tmp_path / "mixed.cbm"
    model.save_model(path)
    restored = CatBoost().load_model(path)
    np.testing.assert_array_equal(restored.predict(pool, prediction_type="RawFormulaVal"), raw)
