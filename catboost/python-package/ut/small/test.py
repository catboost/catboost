import pytest
import math

import numpy as np
from pandas import read_table, DataFrame, Series

from catboost import Pool, CatBoost, CatBoostClassifier, CatBoostRegressor, CatboostError, cv

from catboost_pytest_lib import data_file, local_canonical_file

EPS = 1e-5

TRAIN_FILE = data_file('adult', 'train_small')
TEST_FILE = data_file('adult', 'test_small')
CD_FILE = data_file('adult', 'train.cd')

CLOUDNESS_TRAIN_FILE = data_file('cloudness_small', 'train_small')
CLOUDNESS_TEST_FILE = data_file('cloudness_small', 'test_small')
CLOUDNESS_CD_FILE = data_file('cloudness_small', 'train.cd')

OUTPUT_MODEL_PATH = 'model.bin'
PREDS_PATH = 'predictions.npy'
TARGET_IDX = 1
CAT_FEATURES = [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 16]


def map_cat_features(data, cat_features):
    for i in range(len(data)):
        for j in cat_features:
            data[i][j] = str(data[i][j])
    return data


def _check_shape(pool):
    return np.shape(pool.get_features()) == (101, 17)


def _check_data(data1, data2):
    return np.all(np.isclose(data1, data2, rtol=0.001))


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
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    data = read_table(TRAIN_FILE, header=None)
    label = DataFrame(data.iloc[:, TARGET_IDX])
    data.drop([TARGET_IDX], axis=1, inplace=True)
    cat_features = pool.get_cat_feature_indices()
    pool2 = Pool(data, label, cat_features)
    assert _check_data(pool.get_features(), pool2.get_features())
    assert _check_data(pool.get_label(), pool2.get_label())


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
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_predict_sklearn_regress():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=2, random_seed=0)
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_predict_sklearn_class():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_predict_class_raw():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(train_pool)
    pred = model.predict(test_pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


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
    model = CatBoostClassifier(iterations=2, random_seed=0, loss_function='MultiClass', thread_count=8)
    model.fit(pool)
    pred = model.predict_proba(pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_zero_baseline():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    baseline = np.zeros((pool.num_row(), 2))
    pool = Pool(pool.get_features(), pool.get_label(), baseline=baseline)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_non_zero_bazeline():
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    base_model = CatBoostClassifier(iterations=2, random_seed=0, loss_function="MultiClass")
    base_model.fit(pool)
    baseline = np.array(base_model.predict(pool))
    pool2 = Pool(pool.get_features(), pool.get_label(), baseline=baseline)
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(pool2)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_ones_weight():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool2 = Pool(pool.get_features(), pool.get_label(), weight=np.ones(pool.num_row()))
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(pool2)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_non_ones_weight():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool2 = Pool(pool.get_features(), pool.get_label(), weight=np.arange(1, pool.num_row()+1))
    model = CatBoostClassifier(iterations=2, random_seed=0)
    model.fit(pool2)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_fit_data():
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    eval_pool = Pool(CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE)
    base_model = CatBoostClassifier(iterations=2, random_seed=0, loss_function="MultiClass")
    base_model.fit(pool)
    baseline = np.array(base_model.predict(pool, prediction_type='RawFormulaVal'))
    eval_baseline = np.array(base_model.predict(eval_pool, prediction_type='RawFormulaVal'))
    eval_pool._set_baseline(eval_baseline)
    model = CatBoostClassifier(iterations=2, random_seed=0, loss_function="MultiClass")
    data = map_cat_features(pool.get_features(), pool.get_cat_feature_indices())
    model.fit(data, pool.get_label(), pool.get_cat_feature_indices(), sample_weight=np.arange(1, pool.num_row()+1), baseline=baseline, use_best_model=True, eval_set=eval_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_ntree_limit():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=100, random_seed=0)
    model.fit(train_pool)
    pred = model.predict_proba(test_pool, ntree_limit=10)
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


def test_invalid_loss():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoost({"loss_function": "abcdef"})
        model.fit(pool)


def test_no_eval_set():
    with pytest.raises(CatboostError):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostClassifier()
        model.fit(pool, use_best_model=True)


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
        model = CatBoostClassifier(ctr_description=['Borders:5:Uniform'])
        model.fit(pool)


def test_wrong_feature_count():
    with pytest.raises(CatboostError):
        data = np.random.rand(100, 10)
        label = np.random.randint(2, size=100)
        model = CatBoostClassifier()
        model.fit(data, label)
        model.predict(data[:, :-1])


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
                               leaf_estimation_method="Newton", gradient_iterations=10)
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
    model = CatBoostClassifier(iterations=5, random_seed=0, has_time=True, priors=[0, 0.6, 1, 5])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


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
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_class_weights():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0, class_weights=[1, 2])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_classification_ctr():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, random_seed=0, ctr_description=['Borders', 'Counter'])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_regression_ctr():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=5, random_seed=0, ctr_description=['Borders:5:Uniform', 'Counter:10:MinEntropy'])
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_copy_model():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model1 = CatBoostRegressor(iterations=5, random_seed=0)
    model1.fit(pool)
    model2 = model1.copy()
    predictions1 = model1.predict(pool)
    predictions2 = model2.predict(pool)
    assert _check_data(predictions1, predictions2)
    model2.save_model(OUTPUT_MODEL_PATH)
    return local_canonical_file(OUTPUT_MODEL_PATH)


def test_cv():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    results = cv({"iterations": 5, "random_seed": 0, "loss_function": "Logloss"}, pool)
    assert isinstance(results, dict)
    assert "Logloss_train_avg" in results

    prev_value = results["Logloss_train_avg"][0]
    for value in results["Logloss_train_avg"][1:]:
        assert value < prev_value
        prev_value = value
