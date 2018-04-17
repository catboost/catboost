import pytest
import math

import numpy as np
from pandas import read_table, DataFrame, Series

from catboost import Pool, CatBoost, CatBoostClassifier, CatBoostRegressor, CatboostError, cv, train

from catboost_pytest_lib import data_file, local_canonical_file, remove_time_from_json
import yatest.common

EPS = 1e-5

TRAIN_FILE = data_file('adult', 'train_small')
TEST_FILE = data_file('adult', 'test_small')
CD_FILE = data_file('adult', 'train.cd')

NAN_TRAIN_FILE = data_file('adult_nan', 'train_small')
NAN_TEST_FILE = data_file('adult_nan', 'test_small')
NAN_CD_FILE = data_file('adult_nan', 'train.cd')

CLOUDNESS_TRAIN_FILE = data_file('cloudness_small', 'train_small')
CLOUDNESS_TEST_FILE = data_file('cloudness_small', 'test_small')
CLOUDNESS_CD_FILE = data_file('cloudness_small', 'train.cd')

QUERYWISE_TRAIN_FILE = data_file('querywise', 'train')
QUERYWISE_TEST_FILE = data_file('querywise', 'test')
QUERYWISE_CD_FILE = data_file('querywise', 'train.cd')
QUERYWISE_TRAIN_PAIRS_FILE = data_file('querywise', 'train.pairs')

OUTPUT_MODEL_PATH = 'model.bin'
OUTPUT_COREML_MODEL_PATH = 'model.mlmodel'
OUTPUT_CPP_MODEL_PATH = 'model.cpp'
OUTPUT_PYTHON_MODEL_PATH = 'model.py'
PREDS_PATH = 'predictions.npy'
FIMP_PATH = 'feature_importance.npy'
OIMP_PATH = 'object_importances.txt'
JSON_LOG_PATH = 'catboost_info/catboost_training.json'
TARGET_IDX = 1
CAT_FEATURES = [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 16]


model_diff_tool = yatest.common.binary_path("catboost/tools/model_comparator/model_comparator")


class LogStdout:
    def __init__(self, file):
        self.log_file = file

    def __enter__(self):
        import sys
        self.saved_stdout = sys.stdout
        sys.stdout = self.log_file
        return self.saved_stdout

    def __exit__(self, exc_type, exc_value, exc_traceback):
        import sys
        sys.stdout = self.saved_stdout
        self.log_file.close()


def compare_canonical_models(*args, **kwargs):
    return local_canonical_file(*args, diff_tool=model_diff_tool, **kwargs)


def map_cat_features(data, cat_features):
    for i in range(len(data)):
        for j in cat_features:
            data[i][j] = str(data[i][j])
    return data


def _check_shape(pool):
    return np.shape(pool.get_features()) == (101, 17)


def _check_data(data1, data2):
    return np.all(np.isclose(data1, data2, rtol=0.001, equal_nan=True))


def test_load_file():
    assert _check_shape(Pool(TRAIN_FILE, column_description=CD_FILE))


def test_load_list():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    cat_features = pool.get_cat_feature_indices()
    data = map_cat_features(pool.get_features(), cat_features)
    label = pool.get_label()
    assert _check_shape(Pool(data, label, cat_features))


def test_load_ndarray():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    cat_features = pool.get_cat_feature_indices()
    data = np.array(map_cat_features(pool.get_features(), cat_features))
    label = np.array(pool.get_label())
    assert _check_shape(Pool(data, label, cat_features))


def test_load_df():
    pool = Pool(NAN_TRAIN_FILE, column_description=NAN_CD_FILE)
    data = read_table(NAN_TRAIN_FILE, header=None)
    label = DataFrame(data.iloc[:, TARGET_IDX])
    data.drop([TARGET_IDX], axis=1, inplace=True)
    cat_features = pool.get_cat_feature_indices()
    pool2 = Pool(data, label, cat_features)
    assert _check_data(pool.get_features(), pool2.get_features())
    assert _check_data(pool.get_label(), pool2.get_label())


def test_load_df_vs_load_from_file():
    pool1 = Pool(TRAIN_FILE, column_description=CD_FILE)
    data = read_table(TRAIN_FILE, header=None, dtype=str)
    label = DataFrame(data.iloc[:, TARGET_IDX])
    data.drop([TARGET_IDX], axis=1, inplace=True)
    cat_features = pool1.get_cat_feature_indices()
    pool2 = Pool(np.array(data), label, cat_features)
    assert pool1 == pool2


def test_load_series():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    data = read_table(TRAIN_FILE, header=None)
    label = Series(data.iloc[:, TARGET_IDX])
    data.drop([TARGET_IDX], axis=1, inplace=True)
    data = Series(list(data.values))
    cat_features = pool.get_cat_feature_indices()
    pool2 = Pool(data, label, cat_features)
    assert _check_data(pool.get_features(), pool2.get_features())
    assert _check_data(pool.get_label(), pool2.get_label())


def test_pool_cat_features():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    assert np.all(pool.get_cat_feature_indices() == CAT_FEATURES)


def test_load_generated():
    pool_size = (100, 10)
    data = np.round(np.random.normal(size=pool_size), decimals=3)
    label = np.random.randint(2, size=pool_size[0])
    pool = Pool(data, label)
    assert _check_data(pool.get_features(), data)
    assert _check_data(pool.get_label(), label)


def test_load_dumps():
    pool_size = (100, 10)
    data = np.random.randint(10, size=pool_size)
    label = np.random.randint(2, size=pool_size[0])
    pool1 = Pool(data, label)
    lines = []
    for i in range(len(data)):
        line = [str(label[i])] + [str(x) for x in data[i]]
        lines.append('\t'.join(line))
    text = '\n'.join(lines)
    with open('test_data_dumps', 'w') as f:
        f.write(text)
    pool2 = Pool('test_data_dumps')
    assert _check_data(pool1.get_features(), pool2.get_features())
    assert _check_data(pool1.get_label(), pool2.get_label())


def test_predict_regress():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE'})
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_predict_sklearn_regress():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=2, random_seed=0)
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_predict_sklearn_class():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0, loss_function='Logloss:border=0.5')
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_predict_class_raw():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(train_pool)
    pred = model.predict(test_pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_raw_predict_equals_to_model_predict():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, random_seed=0)
    model.fit(train_pool, eval_set=test_pool)
    pred = model.predict(test_pool, prediction_type='RawFormulaVal')
    assert all(model.get_test_eval() == pred)


def test_fit_from_file():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE'})
    model.fit(train_pool)
    predictions1 = model.predict(train_pool)

    model.fit(TRAIN_FILE, column_description=CD_FILE)
    predictions2 = model.predict(train_pool)
    assert all(predictions1 == predictions2)


def test_coreml_import_export():
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost(params={'loss_function': 'QueryRMSE', 'random_seed': 0, 'iterations': 20, 'thread_count': 8})
    model.fit(train_pool)
    model.save_model(OUTPUT_COREML_MODEL_PATH, format="coreml")
    canon_pred = model.predict(test_pool)
    coreml_loaded_model = CatBoostRegressor()
    coreml_loaded_model.load_model(OUTPUT_COREML_MODEL_PATH, format="coreml")
    assert all(canon_pred == coreml_loaded_model.predict(test_pool))
    return local_canonical_file(OUTPUT_COREML_MODEL_PATH)


def test_cpp_export_no_cat_features():
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE'})
    model.fit(train_pool)
    model.save_model(OUTPUT_CPP_MODEL_PATH, format="cpp")
    return local_canonical_file(OUTPUT_CPP_MODEL_PATH)


def test_python_export_no_cat_features():
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE'})
    model.fit(train_pool)
    model.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python")
    return local_canonical_file(OUTPUT_PYTHON_MODEL_PATH)


def test_python_export_with_cat_features():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 20, 'random_seed': 0})
    model.fit(train_pool)
    model.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python")
    return local_canonical_file(OUTPUT_PYTHON_MODEL_PATH)


def test_predict_class():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(train_pool)
    pred = model.predict(test_pool, prediction_type="Class")
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_predict_class_proba():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(train_pool)
    pred = model.predict_proba(test_pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_no_cat_in_predict():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(train_pool)
    pred1 = model.predict(map_cat_features(test_pool.get_features(), train_pool.get_cat_feature_indices()))
    pred2 = model.predict(Pool(map_cat_features(test_pool.get_features(), train_pool.get_cat_feature_indices()), cat_features=train_pool.get_cat_feature_indices()))
    assert _check_data(pred1, pred2)


def test_save_model():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoost()
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    model2 = CatBoost(model_file=OUTPUT_MODEL_PATH)
    pred1 = model.predict(test_pool)
    pred2 = model2.predict(test_pool)
    assert _check_data(pred1, pred2)


def test_multiclass():
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=2, random_seed=0, loss_function='MultiClass', thread_count=8)
    classifier.fit(pool)
    classifier.save_model(OUTPUT_MODEL_PATH)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(OUTPUT_MODEL_PATH)
    pred = new_classifier.predict_proba(pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_querywise():
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost(params={'loss_function': 'QueryRMSE', 'random_seed': 0, 'iterations': 2, 'thread_count': 8})
    model.fit(train_pool)
    pred1 = model.predict(test_pool)

    df = read_table(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_query_id = df.loc[:, 0]
    train_target = df.loc[:, 1]
    train_data = df.drop([0, 1, 2, 3], axis=1).astype(str)

    df = read_table(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = df.drop([0, 1, 2, 3], axis=1).astype(str)

    model.fit(train_data, train_target, group_id=train_query_id)
    pred2 = model.predict(test_data)
    assert _check_data(pred1, pred2)


def test_zero_baseline():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    baseline = np.zeros(pool.num_row())
    pool.set_baseline(baseline)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_ones_weight():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    weight = np.ones(pool.num_row())
    pool.set_weight(weight)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_non_ones_weight():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    weight = np.arange(1, pool.num_row()+1)
    pool.set_weight(weight)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_fit_data():
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    eval_pool = Pool(CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE)
    base_model = CatBoostClassifier(iterations=2, random_seed=0, loss_function="MultiClass")
    base_model.fit(pool)
    baseline = np.array(base_model.predict(pool, prediction_type='RawFormulaVal'))
    eval_baseline = np.array(base_model.predict(eval_pool, prediction_type='RawFormulaVal'))
    eval_pool.set_baseline(eval_baseline)
    model = CatBoostClassifier(iterations=2, random_seed=0, loss_function="MultiClass")
    data = map_cat_features(pool.get_features(), pool.get_cat_feature_indices())
    model.fit(data, pool.get_label(), pool.get_cat_feature_indices(), sample_weight=np.arange(1, pool.num_row()+1), baseline=baseline, use_best_model=True, eval_set=eval_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_ntree_limit():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=100, random_seed=0)
    model.fit(train_pool)
    pred = model.predict_proba(test_pool, ntree_end=10)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_staged_predict():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, random_seed=0)
    model.fit(train_pool)
    preds = []
    for pred in model.staged_predict(test_pool):
        preds.append(pred)
    np.save(PREDS_PATH, np.array(preds))
    return local_canonical_file(PREDS_PATH)


def test_invalid_loss_base():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoost({"loss_function": "abcdef"})
        model.fit(pool)


def test_invalid_loss_classifier():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostClassifier(loss_function="abcdef")
        model.fit(pool)


def test_invalid_loss_regressor():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostRegressor(loss_function="fee")
        model.fit(pool)


def test_fit_no_label():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostClassifier()
        model.fit(pool.get_features())


def test_predict_without_fit():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostClassifier()
        model.predict(pool)


def test_real_numbers_cat_features():
    with pytest.raises(CatboostError):
        data = np.random.rand(100, 10)
        label = np.random.randint(2, size=100)
        Pool(data, label, [1, 2])


def test_wrong_ctr_for_classification():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostClassifier(ctr_description=['Borders:TargetBorderCount=5:TargetBorderType=Uniform'])
        model.fit(pool)


def test_wrong_feature_count():
    with pytest.raises(CatboostError):
        data = np.random.rand(100, 10)
        label = np.random.randint(2, size=100)
        model = CatBoostClassifier()
        model.fit(data, label)
        model.predict(data[:, :-1])


def test_feature_importance_off():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostClassifier(iterations=5, random_seed=0, calc_feature_importance=False)
        model.fit(pool)
        model.feature_importances_


def test_wrong_params_classifier():
    with pytest.raises(CatboostError):
        CatBoostClassifier(wrong_param=1)


def test_wrong_params_base():
    with pytest.raises(CatboostError):
        CatBoost({'wrong_param': 1})


def test_wrong_params_regressor():
    with pytest.raises(CatboostError):
        CatBoostRegressor(wrong_param=1)


def test_wrong_kwargs_base():
    with pytest.raises(CatboostError):
        CatBoost({'kwargs': {'wrong_param': 1}})


def test_custom_eval():
    class LoglossMetric(object):
        def get_final_error(self, error, weight):
            return error / (weight + 1e-38)

        def is_max_optimal(self):
            return True

        def evaluate(self, approxes, target, weight):
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])

            approx = approxes[0]

            error_sum = 0.0
            weight_sum = 0.0

            for i in xrange(len(approx)):
                w = 1.0 if weight is None else weight[i]
                weight_sum += w
                error_sum += w * (target[i] * approx[i] - math.log(1 + math.exp(approx[i])))

            return error_sum, weight_sum

    train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)

    model = CatBoostClassifier(iterations=5, random_seed=0, use_best_model=True, eval_metric=LoglossMetric())
    model.fit(train_pool, eval_set=test_pool)
    pred1 = model.predict(test_pool)

    model2 = CatBoostClassifier(iterations=5, random_seed=0, use_best_model=True, eval_metric="Logloss")
    model2.fit(train_pool, eval_set=test_pool)
    pred2 = model2.predict(test_pool)

    for p1, p2 in zip(pred1, pred2):
        assert abs(p1 - p2) < EPS


def test_custom_objective():
    class LoglossObjective(object):
        def calc_ders_range(self, approxes, targets, weights):
            assert len(approxes) == len(targets)
            if weights is not None:
                assert len(weights) == len(approxes)

            exponents = []
            for index in xrange(len(approxes)):
                exponents.append(math.exp(approxes[index]))

            result = []
            for index in xrange(len(targets)):
                p = exponents[index] / (1 + exponents[index])
                der1 = (1 - p) if targets[index] > 0.0 else -p
                der2 = -p * (1 - p)

                if weights is not None:
                    der1 *= weights[index]
                    der2 *= weights[index]

                result.append((der1, der2))

            return result

    train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)

    model = CatBoostClassifier(iterations=5, random_seed=0, use_best_model=True,
                               loss_function=LoglossObjective(), eval_metric="Logloss",
                               # Leaf estimation method and gradient iteration are set to match
                               # defaults for Logloss.
                               leaf_estimation_method="Newton", leaf_estimation_iterations=10)
    model.fit(train_pool, eval_set=test_pool)
    pred1 = model.predict(test_pool, prediction_type='RawFormulaVal')

    model2 = CatBoostClassifier(iterations=5, random_seed=0, use_best_model=True, loss_function="Logloss")
    model2.fit(train_pool, eval_set=test_pool)
    pred2 = model2.predict(test_pool, prediction_type='RawFormulaVal')

    for p1, p2 in zip(pred1, pred2):
        assert abs(p1 - p2) < EPS


def test_pool_after_fit():
    pool1 = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool2 = Pool(TRAIN_FILE, column_description=CD_FILE)
    assert _check_data(pool1.get_features(), pool2.get_features())
    model = CatBoostClassifier(iterations=5, random_seed=0)
    model.fit(pool2)
    assert _check_data(pool1.get_features(), pool2.get_features())


def test_priors():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0, has_time=True, ctr_description=["Borders:Prior=0:Prior=0.6:Prior=1:Prior=5", "Counter:Prior=0:Prior=0.6:Prior=1:Prior=5"])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_ignored_features():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model1 = CatBoostClassifier(iterations=5, random_seed=0, ignored_features=[1, 2, 3])
    model2 = CatBoostClassifier(iterations=5, random_seed=0)
    model1.fit(train_pool)
    model2.fit(train_pool)
    predictions1 = model1.predict(test_pool)
    predictions2 = model2.predict(test_pool)
    assert not _check_data(predictions1, predictions2)
    model1.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_class_weights():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0, class_weights=[1, 2])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_classification_ctr():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0, ctr_description=['Borders', 'Counter'])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_regression_ctr():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=5, random_seed=0, ctr_description=['Borders:TargetBorderCount=5:TargetBorderType=Uniform', 'Counter'])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_copy_model():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model1 = CatBoostRegressor(iterations=5, random_seed=0)
    model1.fit(pool)
    model2 = model1.copy()
    predictions1 = model1.predict(pool)
    predictions2 = model2.predict(pool)
    assert _check_data(predictions1, predictions2)
    model2.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_cv():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    results = cv(pool, {"iterations": 5, "random_seed": 0, "loss_function": "Logloss"})
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:]:
        assert value < prev_value
        prev_value = value


def test_cv_query():
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    results = cv(pool, {"iterations": 5, "random_seed": 0, "loss_function": "QueryRMSE"})
    assert "train-QueryRMSE-mean" in results

    prev_value = results["train-QueryRMSE-mean"][0]
    for value in results["train-QueryRMSE-mean"][1:]:
        assert value < prev_value
        prev_value = value


def test_cv_pairs():
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE)
    results = cv(pool, {"iterations": 5, "random_seed": 8, "loss_function": "PairLogit"})
    assert "train-PairLogit-mean" in results

    prev_value = results["train-PairLogit-mean"][0]
    for value in results["train-PairLogit-mean"][1:]:
        assert value < prev_value
        prev_value = value


def test_feature_importance():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0)
    model.fit(pool)
    np.save(FIMP_PATH, np.array(model.feature_importances_))
    return local_canonical_file(FIMP_PATH)


def test_interaction_feature_importance():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0)
    model.fit(pool)
    np.save(FIMP_PATH, np.array(model.get_feature_importance(pool, fstr_type='Interaction')))
    return local_canonical_file(FIMP_PATH)


def test_doc_feature_importance():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0)
    model.fit(pool)
    np.save(FIMP_PATH, np.array(model.get_feature_importance(pool, fstr_type='Doc')))
    return local_canonical_file(FIMP_PATH)


def test_shap_feature_importance():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0, max_ctr_complexity=1)
    model.fit(pool)
    np.save(FIMP_PATH, np.array(model.get_feature_importance(pool, fstr_type='ShapValues')))
    return local_canonical_file(FIMP_PATH)


def test_od():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(od_type='Iter', od_wait=20, random_seed=42)
    model.fit(train_pool, eval_set=test_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_clone():
    estimator = CatBoostClassifier(
        custom_metric="Accuracy",
        loss_function="MultiClass",
        iterations=400)

    # This is important for sklearn.base.clone since
    # it uses get_params for cloning estimator.
    params = estimator.get_params()
    new_estimator = CatBoostClassifier(**params)
    new_params = new_estimator.get_params()

    for param in params:
        assert param in new_params
        assert new_params[param] == params[param]


def test_different_cat_features_order():
    dataset = np.array([[2, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    labels = [1.2, 3.4, 9.5, 24.5]

    pool1 = Pool(dataset, labels, cat_features=[0, 1])
    pool2 = Pool(dataset, labels, cat_features=[1, 0])

    model = CatBoost({'learning_rate': 1, 'loss_function': 'RMSE', 'iterations': 2, 'random_seed': 42})
    model.fit(pool1)
    assert (model.predict(pool1) == model.predict(pool2)).all()


def test_full_history():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(od_type='Iter', od_wait=20, random_seed=42, approx_on_full_history=True)
    model.fit(train_pool, eval_set=test_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_bad_params_in_cv():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    with pytest.warns(UserWarning):
        cv(pool, {"iterations": 5, "random_seed": 0, "loss_function": "Logloss", "use_best_model": True})


def test_cv_logging():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    cv(pool, {"iterations": 5, "random_seed": 0, "loss_function": "Logloss"})
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_cv_with_not_binarized_target():
    train_file = data_file('adult_not_binarized', 'train_small')
    cd = data_file('adult_not_binarized', 'train.cd')
    pool = Pool(train_file, column_description=cd)
    cv(pool, {"iterations": 5, "random_seed": 0, "loss_function": "Logloss"})
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@pytest.mark.parametrize('loss_function', ['Logloss', 'RMSE', 'QueryRMSE'])
def test_eval_metrics(loss_function):
    train, test, cd, metric = TRAIN_FILE, TEST_FILE, CD_FILE, loss_function
    if loss_function == 'QueryRMSE':
        train, test, cd, metric = QUERYWISE_TRAIN_FILE, QUERYWISE_TEST_FILE, QUERYWISE_CD_FILE, 'PFound'
    if loss_function == 'Logloss':
        metric = 'AUC'

    train_pool = Pool(train, column_description=cd)
    test_pool = Pool(test, column_description=cd)
    model = CatBoost(params={'loss_function': loss_function, 'random_seed': 0, 'iterations': 20, 'thread_count': 8, 'eval_metric': metric})

    model.fit(train_pool, eval_set=test_pool, use_best_model=False)
    first_metrics = np.round(np.loadtxt('catboost_info/test_error.tsv', skiprows=1)[:, 1], 10)
    second_metrics = np.round(model.eval_metrics(test_pool, [metric])[metric], 10)
    assert np.all(first_metrics == second_metrics)


@pytest.mark.parametrize('verbose', [5, False, True])
def test_verbose_int(verbose):
    expected_line_count = {5: 2, False: 0, True: 10}
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile = 'test_data_dumps'

    with LogStdout(open(tmpfile, 'w')):
        cv(pool, {"iterations": 10, "random_seed": 0, "loss_function": "Logloss"}, verbose=verbose)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == expected_line_count[verbose])

    with LogStdout(open(tmpfile, 'w')):
        train(pool, {"iterations": 10, "random_seed": 0, "loss_function": "Logloss"}, verbose=verbose)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == expected_line_count[verbose])

    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_eval_set():
    dataset = [(1, 2, 3, 4), (2, 2, 3, 4), (3, 2, 3, 4), (4, 2, 3, 4)]
    labels = [1, 2, 3, 4]
    train_pool = Pool(dataset, labels, cat_features=[0, 3, 2])

    model = CatBoost({'learning_rate': 1, 'loss_function': 'RMSE', 'iterations': 2, 'random_seed': 0})

    eval_dataset = [(5, 6, 6, 6), (6, 6, 6, 6)]
    eval_labels = [5, 6]
    eval_pool = (eval_dataset, eval_labels)

    model.fit(train_pool, eval_set=eval_pool)

    eval_pools = [eval_pool]

    model.fit(train_pool, eval_set=eval_pools)

    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_object_importances():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool = Pool(TEST_FILE, column_description=CD_FILE)

    model = CatBoost({'loss_function': 'RMSE', 'iterations': 10, 'random_seed': 0})
    model.fit(train_pool)
    indices, scores = model.get_object_importance(pool, train_pool, top_size=10)
    np.savetxt(OIMP_PATH, scores)

    return local_canonical_file(OIMP_PATH)


def test_shap():
    train_pool = Pool([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 5, 8], cat_features=[])
    test_pool = Pool([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = CatBoostRegressor(iterations=1, random_seed=0, max_ctr_complexity=1, depth=2)
    model.fit(train_pool)
    shap_values = model.get_feature_importance(test_pool, fstr_type='ShapValues')

    dataset = [(0.5, 1.2), (1.6, 0.5), (1.8, 1.0), (0.4, 0.6), (0.3, 1.6), (1.5, 0.2)]
    labels = [1.1, 1.85, 2.3, 0.7, 1.1, 1.6]
    train_pool = Pool(dataset, labels, cat_features=[])

    model = CatBoost({'iterations': 10, 'random_seed': 0, 'max_ctr_complexity': 1})
    model.fit(train_pool)

    testset = [(0.6, 1.2), (1.4, 0.3), (1.5, 0.8), (1.4, 0.6)]
    predictions = model.predict(testset)
    shap_values = model.get_feature_importance(Pool(testset), fstr_type='ShapValues')
    assert(len(predictions) == len(shap_values))
    for pred_idx in range(len(predictions)):
        assert(abs(sum(shap_values[pred_idx]) - predictions[pred_idx]) < 1e-9)

    with open(FIMP_PATH, 'w') as out:
        out.write(shap_values)

    local_canonical_file(FIMP_PATH)
