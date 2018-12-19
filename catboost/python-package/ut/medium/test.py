import sys
import hashlib
import math
import re
import subprocess
import pytest
import tempfile
import time

import numpy as np
import random
from pandas import read_table, DataFrame, Series
from six.moves import xrange
from catboost import FeaturesData, EFstrType, Pool, CatBoost, CatBoostClassifier, CatBoostRegressor, CatboostError, cv, train
from catboost.utils import eval_metric, create_cd, get_roc_curve, select_threshold
from catboost.eval.catboost_evaluation import CatboostEvaluation

from catboost_pytest_lib import (
    data_file,
    local_canonical_file,
    remove_time_from_json,
    binary_path,
    test_output_path,
    DelayedTee
)

if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import _pickle as pickle
    pytest_plugins = "list_plugin",

fails_on_gpu = pytest.mark.fails_on_gpu

EPS = 1e-5

BOOSTING_TYPE = ['Ordered', 'Plain']
OVERFITTING_DETECTOR_TYPE = ['IncToDec', 'Iter']

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
QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT = data_file('querywise', 'train.cd.group_weight')
QUERYWISE_CD_FILE_WITH_GROUP_ID = data_file('querywise', 'train.cd.query_id')
QUERYWISE_CD_FILE_WITH_SUBGROUP_ID = data_file('querywise', 'train.cd.subgroup_id')
QUERYWISE_TRAIN_PAIRS_FILE = data_file('querywise', 'train.pairs')
QUERYWISE_TRAIN_PAIRS_FILE_WITH_PAIR_WEIGHT = data_file('querywise', 'train.pairs.weighted')
QUERYWISE_TEST_PAIRS_FILE = data_file('querywise', 'test.pairs')

AIRLINES_5K_TRAIN_FILE = data_file('airlines_5K', 'train')
AIRLINES_5K_TEST_FILE = data_file('airlines_5K', 'test')
AIRLINES_5K_CD_FILE = data_file('airlines_5K', 'cd')

OUTPUT_MODEL_PATH = 'model.bin'
OUTPUT_COREML_MODEL_PATH = 'model.mlmodel'
OUTPUT_CPP_MODEL_PATH = 'model.cpp'
OUTPUT_PYTHON_MODEL_PATH = 'model.py'
OUTPUT_JSON_MODEL_PATH = 'model.json'
PREDS_PATH = 'predictions.npy'
FIMP_NPY_PATH = 'feature_importance.npy'
FIMP_TXT_PATH = 'feature_importance.txt'
OIMP_PATH = 'object_importances.txt'
JSON_LOG_PATH = 'catboost_info/catboost_training.json'
TARGET_IDX = 1
CAT_FEATURES = [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 16]


model_diff_tool = binary_path("catboost/tools/model_comparator/model_comparator")


class LogStdout:
    def __init__(self, file):
        self.log_file = file

    def __enter__(self):
        self.saved_stdout = sys.stdout
        sys.stdout = self.log_file
        return self.saved_stdout

    def __exit__(self, exc_type, exc_value, exc_traceback):
        sys.stdout = self.saved_stdout
        self.log_file.close()


def compare_canonical_models(model, diff_limit=0):
    return local_canonical_file(model, diff_tool=[model_diff_tool, '--diff-limit', str(diff_limit)])


def map_cat_features(data, cat_features):
    for i in range(len(data)):
        for j in cat_features:
            data[i][j] = str(data[i][j])
    return data


def _check_shape(pool, object_count, features_count):
    return np.shape(pool.get_features()) == (object_count, features_count)


def _check_data(data1, data2):
    return np.all(np.isclose(data1, data2, rtol=0.001, equal_nan=True))


def set_random_weight(pool):
    pool.set_weight(np.random.random(pool.num_row()))
    if pool.num_pairs() > 0:
        pool.set_pairs_weight(np.random.random(pool.num_pairs()))


def set_random_target(pool):
    pool._set_label(np.random.random((pool.num_row(), )))


def set_random_target_01(pool):
    pool._set_label(np.random.randint(2, size=(pool.num_row(), )))


def verify_finite(result):
    inf = float('inf')
    for r in result:
        assert(r == r)
        assert(abs(r) < inf)


def append_param(metric_name, param):
    return metric_name + (':' if ':' not in metric_name else ';') + param

# Test cases begin here ########################################################


def test_load_file():
    assert _check_shape(Pool(TRAIN_FILE, column_description=CD_FILE), 101, 17)


def test_load_list():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    cat_features = pool.get_cat_feature_indices()
    data = map_cat_features(pool.get_features(), cat_features)
    label = pool.get_label()
    assert _check_shape(Pool(data, label, cat_features), 101, 17)


def test_load_ndarray():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    cat_features = pool.get_cat_feature_indices()
    data = np.array(map_cat_features(pool.get_features(), cat_features))
    label = np.array(pool.get_label())
    assert _check_shape(Pool(data, label, cat_features), 101, 17)


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


# feature_matrix is (doc_count x feature_count)
def get_features_data_from_matrix(feature_matrix, cat_feature_indices):
    object_count = len(feature_matrix)
    feature_count = len(feature_matrix[0])
    cat_feature_count = len(cat_feature_indices)
    num_feature_count = feature_count - cat_feature_count

    result_num = np.empty((object_count, num_feature_count), dtype=np.float32)
    result_cat = np.empty((object_count, cat_feature_count), dtype=object)

    for object_idx in xrange(object_count):
        num_feature_idx = 0
        cat_feature_idx = 0
        for feature_idx in xrange(len(feature_matrix[object_idx])):
            if (cat_feature_idx < cat_feature_count) and (cat_feature_indices[cat_feature_idx] == feature_idx):
                # simplified handling of transformation to bytes for tests
                result_cat[object_idx, cat_feature_idx] = (
                    feature_matrix[object_idx, feature_idx]
                    if isinstance(feature_matrix[object_idx, feature_idx], bytes)
                    else str(feature_matrix[object_idx, feature_idx]).encode('utf-8')
                )
                cat_feature_idx += 1
            else:
                result_num[object_idx, num_feature_idx] = float(feature_matrix[object_idx, feature_idx])
                num_feature_idx += 1

    return FeaturesData(num_feature_data=result_num, cat_feature_data=result_cat)


def get_features_data_from_file(data_file, drop_columns, cat_feature_indices):
    data_matrix_from_file = read_table(data_file, header=None, dtype=str)
    data_matrix_from_file.drop(drop_columns, axis=1, inplace=True)
    return get_features_data_from_matrix(np.array(data_matrix_from_file), cat_feature_indices)


def compare_flat_index_and_features_data_pools(flat_index_pool, features_data_pool):
    assert flat_index_pool.shape == features_data_pool.shape

    cat_feature_indices = flat_index_pool.get_cat_feature_indices()
    num_feature_count = flat_index_pool.shape[1] - len(cat_feature_indices)

    flat_index_pool_features = flat_index_pool.get_features()
    features_data_pool_features = features_data_pool.get_features()

    for object_idx in xrange(flat_index_pool.shape[0]):
        num_feature_idx = 0
        cat_feature_idx = 0
        for flat_feature_idx in xrange(flat_index_pool.shape[1]):
            if (
                (cat_feature_idx < len(cat_feature_indices))
                and (cat_feature_indices[cat_feature_idx] == flat_feature_idx)
            ):

                # simplified handling of transformation to bytes for tests
                assert (flat_index_pool_features[object_idx][flat_feature_idx] ==
                        features_data_pool_features[object_idx][num_feature_count + cat_feature_idx])
                cat_feature_idx += 1
            else:
                assert np.isclose(
                    flat_index_pool_features[object_idx][flat_feature_idx],
                    features_data_pool_features[object_idx][num_feature_idx],
                    rtol=0.001,
                    equal_nan=True
                )
                num_feature_idx += 1


def test_from_features_data_vs_load_from_files():
    pool_from_files = Pool(TRAIN_FILE, column_description=CD_FILE)

    features_data = get_features_data_from_file(
        data_file=TRAIN_FILE,
        drop_columns=[TARGET_IDX],
        cat_feature_indices=pool_from_files.get_cat_feature_indices()
    )
    pool_from_features_data = Pool(data=features_data)

    compare_flat_index_and_features_data_pools(pool_from_files, pool_from_features_data)


def test_features_data_with_empty_objects():
    fd = FeaturesData(
        cat_feature_data=np.empty((0, 4), dtype=object)
    )
    assert fd.get_object_count() == 0
    assert fd.get_feature_count() == 4
    assert fd.get_num_feature_count() == 0
    assert fd.get_cat_feature_count() == 4
    assert fd.get_feature_names() == [''] * 4

    fd = FeaturesData(
        num_feature_data=np.empty((0, 2), dtype=np.float32),
        num_feature_names=['f0', 'f1']
    )
    assert fd.get_object_count() == 0
    assert fd.get_feature_count() == 2
    assert fd.get_num_feature_count() == 2
    assert fd.get_cat_feature_count() == 0
    assert fd.get_feature_names() == ['f0', 'f1']

    fd = FeaturesData(
        cat_feature_data=np.empty((0, 2), dtype=object),
        num_feature_data=np.empty((0, 3), dtype=np.float32)
    )
    assert fd.get_object_count() == 0
    assert fd.get_feature_count() == 5
    assert fd.get_num_feature_count() == 3
    assert fd.get_cat_feature_count() == 2
    assert fd.get_feature_names() == [''] * 5


def test_features_data_names():
    # empty specification of names
    fd = FeaturesData(
        cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object),
        num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32),
    )
    assert fd.get_feature_names() == [''] * 5

    # full specification of names
    fd = FeaturesData(
        cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object),
        cat_feature_names=['shop', 'search'],
        num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32),
        num_feature_names=['weight', 'price', 'volume']
    )
    assert fd.get_feature_names() == ['weight', 'price', 'volume', 'shop', 'search']

    # partial specification of names
    fd = FeaturesData(
        cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object),
        num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32),
        num_feature_names=['weight', 'price', 'volume']
    )
    assert fd.get_feature_names() == ['weight', 'price', 'volume', '', '']

    # partial specification of names
    fd = FeaturesData(
        cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object),
        cat_feature_names=['shop', 'search'],
        num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32),
    )
    assert fd.get_feature_names() == ['', '', '', 'shop', 'search']


def compare_pools_from_features_data_and_generic_matrix(
    features_data,
    generic_matrix,
    cat_features_indices,
    feature_names=None
):
    pool1 = Pool(data=features_data)
    pool2 = Pool(data=generic_matrix, cat_features=cat_features_indices, feature_names=feature_names)
    assert _check_data(pool1.get_features(), pool2.get_features())
    assert pool1.get_cat_feature_indices() == pool2.get_cat_feature_indices()
    assert pool1.get_feature_names() == pool2.get_feature_names()


def test_features_data_good():
    # 0 objects
    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(cat_feature_data=np.empty((0, 4), dtype=object)),
        np.empty((0, 4), dtype=object),
        cat_features_indices=[0, 1, 2, 3]
    )

    # 0 objects
    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.empty((0, 2), dtype=object),
            cat_feature_names=['cat0', 'cat1'],
            num_feature_data=np.empty((0, 3), dtype=np.float32),
        ),
        np.empty((0, 5), dtype=object),
        cat_features_indices=[3, 4],
        feature_names=['', '', '', 'cat0', 'cat1']
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object)),
        [[b'amazon', b'bing'], [b'ebay', b'google']],
        cat_features_indices=[0, 1]
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32)),
        [[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]],
        cat_features_indices=[]
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object),
            num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32)
        ),
        [[1.0, 2.0, 3.0, b'amazon', b'bing'], [22.0, 7.1, 10.2, b'ebay', b'google']],
        cat_features_indices=[3, 4]
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object),
            cat_feature_names=['shop', 'search']
        ),
        [[b'amazon', b'bing'], [b'ebay', b'google']],
        cat_features_indices=[0, 1],
        feature_names=['shop', 'search']
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32),
            num_feature_names=['weight', 'price', 'volume']
        ),
        [[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]],
        cat_features_indices=[],
        feature_names=['weight', 'price', 'volume']
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object),
            cat_feature_names=['shop', 'search'],
            num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32),
            num_feature_names=['weight', 'price', 'volume']
        ),
        [[1.0, 2.0, 3.0, b'amazon', b'bing'], [22.0, 7.1, 10.2, b'ebay', b'google']],
        cat_features_indices=[3, 4],
        feature_names=['weight', 'price', 'volume', 'shop', 'search']
    )


def test_features_data_bad():
    # empty
    with pytest.raises(CatboostError):
        FeaturesData()

    # names w/o data
    with pytest.raises(CatboostError):
        FeaturesData(cat_feature_data=[[b'amazon', b'bing']], num_feature_names=['price'])

    # bad matrix type
    with pytest.raises(CatboostError):
        FeaturesData(
            cat_feature_data=[[b'amazon', b'bing']],
            num_feature_data=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

    # bad matrix shape
    with pytest.raises(CatboostError):
        FeaturesData(num_feature_data=np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32))

    # bad element type
    with pytest.raises(CatboostError):
        FeaturesData(
            cat_feature_data=np.array([b'amazon', b'bing'], dtype=object),
            num_feature_data=np.array([1.0, 2.0, 3.0], dtype=np.float64)
        )

    # bad element type
    with pytest.raises(CatboostError):
        FeaturesData(cat_feature_data=np.array(['amazon', 'bing']))

    # bad names type
    with pytest.raises(CatboostError):
        FeaturesData(
            cat_feature_data=np.array([[b'google'], [b'reddit']], dtype=object),
            cat_feature_names=[None, 'news_aggregator']
        )

    # bad names length
    with pytest.raises(CatboostError):
        FeaturesData(
            cat_feature_data=np.array([[b'google'], [b'bing']], dtype=object),
            cat_feature_names=['search_engine', 'news_aggregator']
        )

    # no features
    with pytest.raises(CatboostError):
        FeaturesData(
            cat_feature_data=np.array([[], [], []], dtype=object),
            num_feature_data=np.array([[], [], []], dtype=np.float32)
        )

    # number of objects is different
    with pytest.raises(CatboostError):
        FeaturesData(
            cat_feature_data=np.array([[b'google'], [b'bing']], dtype=object),
            num_feature_data=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )


def test_predict_regress(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    assert(model.is_fitted())
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_predict_sklearn_regress(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    assert(model.is_fitted())
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_predict_sklearn_class(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, loss_function='Logloss:border=0.5', task_type=task_type, devices='0')
    model.fit(train_pool)
    assert(model.is_fitted())
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_predict_class_raw(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict(test_pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


@fails_on_gpu(how='model.get_test_eval() returns `bool`')
def test_raw_predict_equals_to_model_predict(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    assert(model.is_fitted())
    pred = model.predict(test_pool, prediction_type='RawFormulaVal')
    assert all(model.get_test_eval() == pred)


def test_model_pickling(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    pred = model.predict(test_pool, prediction_type='RawFormulaVal')
    model_unpickled = pickle.loads(pickle.dumps(model))
    pred_new = model_unpickled.predict(test_pool, prediction_type='RawFormulaVal')
    assert all(pred_new == pred)


def test_fit_from_file(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    predictions1 = model.predict(train_pool)

    model.fit(TRAIN_FILE, column_description=CD_FILE)
    predictions2 = model.predict(train_pool)
    assert all(predictions1 == predictions2)


@fails_on_gpu(how='assert 0.019921323750168085 < EPS, where 0.019921323750168085 = abs((0.03378972364589572 - 0.053711047396063805))')
def test_fit_from_features_data(task_type):
    pool_from_files = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(pool_from_files)
    assert(model.is_fitted())
    predictions_from_files = model.predict(pool_from_files)

    features_data = get_features_data_from_file(
        data_file=TRAIN_FILE,
        drop_columns=[TARGET_IDX],
        cat_feature_indices=pool_from_files.get_cat_feature_indices()
    )
    model.fit(X=features_data, y=pool_from_files.get_label())
    predictions_from_features_data = model.predict(Pool(features_data))

    for prediction1, prediction2 in zip(predictions_from_files, predictions_from_features_data):
        assert abs(prediction1 - prediction2) < EPS


def test_fit_from_empty_features_data(task_type):
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    with pytest.raises(CatboostError):
        model.fit(
            X=FeaturesData(num_feature_data=np.empty((0, 2), dtype=np.float32)),
            y=np.empty((0), dtype=np.int32)
        )


def test_coreml_import_export(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost(params={'loss_function': 'RMSE', 'random_seed': 0, 'iterations': 20, 'thread_count': 8, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    model.save_model(OUTPUT_COREML_MODEL_PATH, format="coreml")
    canon_pred = model.predict(test_pool)
    coreml_loaded_model = CatBoostRegressor()
    coreml_loaded_model.load_model(OUTPUT_COREML_MODEL_PATH, format="coreml")
    assert all(canon_pred == coreml_loaded_model.predict(test_pool))
    return compare_canonical_models(OUTPUT_COREML_MODEL_PATH)


@pytest.mark.parametrize('pool', ['adult', 'higgs'])
def test_convert_model_to_json(task_type, pool):
    train_pool = Pool(data_file(pool, 'train_small'), column_description=data_file(pool, 'train.cd'))
    test_pool = Pool(data_file(pool, 'test_small'), column_description=data_file(pool, 'train.cd'))
    converted_model_path = "converted_model.bin"
    model = CatBoost({'random_seed': 0, 'iterations': 20, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    model.save_model(OUTPUT_JSON_MODEL_PATH, format="json")
    model2 = CatBoost()
    model2.load_model(OUTPUT_JSON_MODEL_PATH, format="json")
    model2.save_model(converted_model_path)
    pred1 = model.predict(test_pool)
    pred2 = model2.predict(test_pool)
    assert _check_data(pred1, pred2)
    subprocess.check_call((model_diff_tool, OUTPUT_MODEL_PATH, converted_model_path, '--diff-limit', '0.000001'))
    return compare_canonical_models(converted_model_path)


def test_coreml_cbm_import_export(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost(params={'loss_function': 'RMSE', 'random_seed': 0, 'iterations': 20, 'thread_count': 8, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    canon_pred = model.predict(test_pool)
    model.save_model(OUTPUT_COREML_MODEL_PATH, format="coreml")

    coreml_loaded_model = CatBoost()
    coreml_loaded_model.load_model(OUTPUT_COREML_MODEL_PATH, format="coreml")
    coreml_loaded_model.save_model(OUTPUT_MODEL_PATH)

    cbm_loaded_model = CatBoost()
    cbm_loaded_model.load_model(OUTPUT_MODEL_PATH)
    assert all(canon_pred == cbm_loaded_model.predict(test_pool))
    return compare_canonical_models(OUTPUT_COREML_MODEL_PATH)


def test_cpp_export_no_cat_features(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    model.save_model(OUTPUT_CPP_MODEL_PATH, format="cpp")
    return local_canonical_file(OUTPUT_CPP_MODEL_PATH)


def test_cpp_export_with_cat_features(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 20, 'random_seed': 0, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    model.save_model(OUTPUT_CPP_MODEL_PATH, format="cpp")
    return local_canonical_file(OUTPUT_CPP_MODEL_PATH)


@pytest.mark.parametrize('iterations', [2, 40])
def test_export_to_python_no_cat_features(task_type, iterations):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost({'iterations': iterations, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    model.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python")
    return local_canonical_file(OUTPUT_PYTHON_MODEL_PATH)


@pytest.mark.parametrize('iterations', [2, 40])
def test_export_to_python_with_cat_features(task_type, iterations):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': iterations, 'random_seed': 0, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    model.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python", pool=train_pool)
    return local_canonical_file(OUTPUT_PYTHON_MODEL_PATH)


def test_predict_class(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict(test_pool, prediction_type="Class")
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_predict_class_proba(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict_proba(test_pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


@fails_on_gpu(how='assert 0.031045619651137835 < EPS, where 0.031045619651137835 = <function amax at ...')
@pytest.mark.parametrize('function_name', ['predict', 'predict_proba'])
def test_predict_funcs_from_features_data(function_name, task_type):
    function = getattr(CatBoostClassifier, function_name)

    train_pool_from_files = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool_from_files)

    test_pool_from_files = Pool(TEST_FILE, column_description=CD_FILE)
    predictions_from_files = function(model, test_pool_from_files)

    train_features_data, test_features_data = [
        get_features_data_from_file(
            data_file=data_file,
            drop_columns=[TARGET_IDX],
            cat_feature_indices=train_pool_from_files.get_cat_feature_indices()
        )
        for data_file in [TRAIN_FILE, TEST_FILE]
    ]
    model.fit(X=train_features_data, y=train_pool_from_files.get_label())
    predictions_from_features_data = function(model, test_features_data)

    for prediction1, prediction2 in zip(predictions_from_files, predictions_from_features_data):
        assert np.max(np.abs(prediction1 - prediction2)) < EPS

    # empty
    empty_test_features_data = FeaturesData(
        num_feature_data=np.empty((0, test_features_data.get_num_feature_count()), dtype=np.float32),
        cat_feature_data=np.empty((0, test_features_data.get_cat_feature_count()), dtype=object)
    )
    empty_predictions = function(model, empty_test_features_data)
    assert len(empty_predictions) == 0


def test_no_cat_in_predict(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred1 = model.predict(map_cat_features(test_pool.get_features(), train_pool.get_cat_feature_indices()))
    pred2 = model.predict(Pool(map_cat_features(test_pool.get_features(), train_pool.get_cat_feature_indices()), cat_features=train_pool.get_cat_feature_indices()))
    assert _check_data(pred1, pred2)


def test_save_model(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoost({'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    model2 = CatBoost()
    model2.load_model(OUTPUT_MODEL_PATH)
    pred1 = model.predict(test_pool)
    pred2 = model2.predict(test_pool)
    assert _check_data(pred1, pred2)


@fails_on_gpu(how='cuda/train_lib/train.cpp:283: Error: loss function is not supported for GPU learning MultiClass')
def test_multiclass(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=2, random_seed=0, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(pool)
    classifier.save_model(OUTPUT_MODEL_PATH)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(OUTPUT_MODEL_PATH)
    pred = new_classifier.predict_proba(pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


@fails_on_gpu(how='cuda/train_lib/train.cpp:283: Error: loss function is not supported for GPU learning MultiClass')
def test_multiclass_classes_count_missed_classes(task_type):
    np.random.seed(0)
    pool = Pool(np.random.random(size=(100, 10)), label=np.random.choice([1, 3], size=100))
    classifier = CatBoostClassifier(classes_count=4, iterations=2, random_seed=0, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(pool)
    classifier.save_model(OUTPUT_MODEL_PATH)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(OUTPUT_MODEL_PATH)
    pred = new_classifier.predict_proba(pool)
    classes = new_classifier.predict(pool)
    assert pred.shape == (100, 4)
    assert np.array(classes).all() in [1, 3]
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_querywise(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost(params={'loss_function': 'QueryRMSE', 'random_seed': 0, 'iterations': 2, 'thread_count': 8, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    pred1 = model.predict(test_pool)

    df = read_table(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_query_id = df.loc[:, 1]
    train_target = df.loc[:, 2]
    train_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    df = read_table(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    model.fit(train_data, train_target, group_id=train_query_id)
    pred2 = model.predict(test_data)
    assert _check_data(pred1, pred2)


def test_group_weight(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
    model = CatBoost(params={'loss_function': 'YetiRank', 'random_seed': 0, 'iterations': 10, 'thread_count': 8, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    pred1 = model.predict(test_pool)

    df = read_table(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_query_weight = df.loc[:, 0]
    train_query_id = df.loc[:, 1]
    train_target = df.loc[:, 2]
    train_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    df = read_table(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_query_weight = df.loc[:, 0]
    test_data = Pool(df.drop([0, 1, 2, 3, 4], axis=1).astype(str), group_weight=test_query_weight)

    model.fit(train_data, train_target, group_id=train_query_id, group_weight=train_query_weight)
    pred2 = model.predict(test_data)
    assert _check_data(pred1, pred2)


def test_zero_baseline(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    baseline = np.zeros(pool.num_row())
    pool.set_baseline(baseline)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_ones_weight(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    weight = np.ones(pool.num_row())
    pool.set_weight(weight)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_non_ones_weight(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    weight = np.arange(1, pool.num_row() + 1)
    pool.set_weight(weight)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_ones_weight_equal_to_nonspecified_weight(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')

    predictions = []

    for set_weights in [False, True]:
        if set_weights:
            weight = np.ones(train_pool.num_row())
            train_pool.set_weight(weight)
        model.fit(train_pool)
        predictions.append(model.predict(test_pool))

    assert _check_data(predictions[0], predictions[1])


def test_py_data_group_id(task_type):
    train_pool_from_files = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_ID)
    test_pool_from_files = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_ID)
    model = CatBoost(
        params={'loss_function': 'QueryRMSE', 'random_seed': 0, 'iterations': 2, 'thread_count': 4, 'task_type': task_type, 'devices': '0'}
    )
    model.fit(train_pool_from_files)
    predictions_from_files = model.predict(test_pool_from_files)

    train_df = read_table(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_target = train_df.loc[:, 2]
    raw_train_group_id = train_df.loc[:, 1]
    train_data = train_df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    test_df = read_table(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = Pool(test_df.drop([0, 1, 2, 3, 4], axis=1).astype(str))

    for group_id_func in (int, str, lambda id: 'myid_' + str(id)):
        train_group_id = [group_id_func(group_id) for group_id in raw_train_group_id]
        model.fit(train_data, train_target, group_id=train_group_id)
        predictions_from_py_data = model.predict(test_data)
        assert _check_data(predictions_from_files, predictions_from_py_data)


def test_py_data_subgroup_id(task_type):
    train_pool_from_files = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE_WITH_SUBGROUP_ID)
    test_pool_from_files = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE_WITH_SUBGROUP_ID)
    model = CatBoost(
        params={'loss_function': 'QueryRMSE', 'random_seed': 0, 'iterations': 2, 'thread_count': 4, 'task_type': task_type, 'devices': '0'}
    )
    model.fit(train_pool_from_files)
    predictions_from_files = model.predict(test_pool_from_files)

    train_df = read_table(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_group_id = train_df.loc[:, 1]
    raw_train_subgroup_id = train_df.loc[:, 4]
    train_target = train_df.loc[:, 2]
    train_data = train_df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    test_df = read_table(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = Pool(test_df.drop([0, 1, 2, 3, 4], axis=1).astype(str))

    for subgroup_id_func in (int, str, lambda id: 'myid_' + str(id)):
        train_subgroup_id = [subgroup_id_func(subgroup_id) for subgroup_id in raw_train_subgroup_id]
        model.fit(train_data, train_target, group_id=train_group_id, subgroup_id=train_subgroup_id)
        predictions_from_py_data = model.predict(test_data)
        assert _check_data(predictions_from_files, predictions_from_py_data)


@fails_on_gpu(how='cuda/train_lib/train.cpp:283: Error: loss function is not supported for GPU learning MultiClass')
def test_fit_data(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    eval_pool = Pool(CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE)
    base_model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, loss_function="MultiClass", task_type=task_type, devices='0')
    base_model.fit(pool)
    baseline = np.array(base_model.predict(pool, prediction_type='RawFormulaVal'))
    eval_baseline = np.array(base_model.predict(eval_pool, prediction_type='RawFormulaVal'))
    eval_pool.set_baseline(eval_baseline)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, loss_function="MultiClass")
    data = map_cat_features(pool.get_features(), pool.get_cat_feature_indices())
    model.fit(data, pool.get_label(), pool.get_cat_feature_indices(), sample_weight=np.arange(1, pool.num_row() + 1), baseline=baseline, use_best_model=True, eval_set=eval_pool)
    pred = model.predict_proba(eval_pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_ntree_limit(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=100, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict_proba(test_pool, ntree_end=10)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_staged_predict(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    preds = []
    for pred in model.staged_predict(test_pool):
        preds.append(pred)
    np.save(PREDS_PATH, np.array(preds))
    return local_canonical_file(PREDS_PATH)


@fails_on_gpu(how='assert 1.0 < EPS')
@pytest.mark.parametrize('staged_function_name', ['staged_predict', 'staged_predict_proba'])
def test_staged_predict_funcs_from_features_data(staged_function_name, task_type):
    staged_function = getattr(CatBoostClassifier, staged_function_name)
    fit_iterations = 10

    train_pool_from_files = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=fit_iterations, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool_from_files)

    test_pool_from_files = Pool(TEST_FILE, column_description=CD_FILE)
    predictions_from_files = []
    for prediction in staged_function(model, test_pool_from_files):
        predictions_from_files.append(prediction)

    train_features_data, test_features_data = [
        get_features_data_from_file(
            data_file=data_file,
            drop_columns=[TARGET_IDX],
            cat_feature_indices=train_pool_from_files.get_cat_feature_indices()
        )
        for data_file in [TRAIN_FILE, TEST_FILE]
    ]
    model.fit(X=train_features_data, y=train_pool_from_files.get_label())
    predictions_from_features_data = []
    for prediction in staged_function(model, test_features_data):
        predictions_from_features_data.append(prediction)

    for prediction1, prediction2 in zip(predictions_from_files, predictions_from_features_data):
        assert np.max(np.abs(prediction1 - prediction2)) < EPS

    # empty
    empty_test_features_data = FeaturesData(
        num_feature_data=np.empty((0, test_features_data.get_num_feature_count()), dtype=np.float32),
        cat_feature_data=np.empty((0, test_features_data.get_cat_feature_count()), dtype=object)
    )
    empty_predictions = []
    for prediction in staged_function(model, empty_test_features_data):
        assert np.shape(prediction) == ((0, 2) if staged_function_name == 'staged_predict_proba' else (0, ))
        empty_predictions.append(prediction)
    assert len(empty_predictions) == fit_iterations


def test_invalid_loss_base(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({"loss_function": "abcdef", 'task_type': task_type, 'devices': '0'})
    with pytest.raises(CatboostError):
        model.fit(pool)


def test_invalid_loss_classifier(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(loss_function="abcdef", task_type=task_type, devices='0')
    with pytest.raises(CatboostError):
        model.fit(pool)


def test_invalid_loss_regressor(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(loss_function="fee", task_type=task_type, devices='0')
    with pytest.raises(CatboostError):
        model.fit(pool)


def test_fit_no_label(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(task_type=task_type, devices='0')
    with pytest.raises(CatboostError):
        model.fit(pool.get_features())


def test_predict_without_fit(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(task_type=task_type, devices='0')
    with pytest.raises(CatboostError):
        model.predict(pool)


def test_real_numbers_cat_features():
    data = np.random.rand(100, 10)
    label = np.random.randint(2, size=100)
    with pytest.raises(CatboostError):
        Pool(data, label, [1, 2])


def test_wrong_ctr_for_classification(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(ctr_description=['Borders:TargetBorderCount=5:TargetBorderType=Uniform'], task_type=task_type, devices='0')
    with pytest.raises(CatboostError):
        model.fit(pool)


def test_wrong_feature_count(task_type):
    data = np.random.rand(100, 10)
    label = np.random.randint(2, size=100)
    model = CatBoostClassifier(task_type=task_type, devices='0')
    model.fit(data, label)
    with pytest.raises(CatboostError):
        model.predict(data[:, :-1])


def test_wrong_params_classifier():
    with pytest.raises(TypeError):
        CatBoostClassifier(wrong_param=1)


def test_wrong_params_base():
    data = np.random.rand(100, 10)
    label = np.random.randint(2, size=100)
    model = CatBoost({'wrong_param': 1})
    with pytest.raises(CatboostError):
        model.fit(data, label)


def test_wrong_params_regressor():
    with pytest.raises(TypeError):
        CatBoostRegressor(wrong_param=1)


def test_wrong_kwargs_base():
    data = np.random.rand(100, 10)
    label = np.random.randint(2, size=100)
    model = CatBoost({'kwargs': {'wrong_param': 1}})
    with pytest.raises(CatboostError):
        model.fit(data, label)


def test_duplicate_params_base():
    data = np.random.rand(100, 10)
    label = np.random.randint(2, size=100)
    model = CatBoost({'iterations': 100, 'n_estimators': 50})
    with pytest.raises(CatboostError):
        model.fit(data, label)


def test_duplicate_params_classifier():
    data = np.random.rand(100, 10)
    label = np.random.randint(2, size=100)
    model = CatBoostClassifier(depth=3, max_depth=4, random_seed=42, random_state=12)
    with pytest.raises(CatboostError):
        model.fit(data, label)


def test_duplicate_params_regressor():
    data = np.random.rand(100, 10)
    label = np.random.randint(2, size=100)
    model = CatBoostRegressor(learning_rate=0.1, eta=0.03, border_count=10, max_bin=12)
    with pytest.raises(CatboostError):
        model.fit(data, label)


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


@fails_on_gpu(how='cuda/train_lib/train.cpp:283: Error: loss function is not supported for GPU learning Custom')
def test_custom_objective(task_type):
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

    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, use_best_model=True,
                               loss_function=LoglossObjective(), eval_metric="Logloss",
                               # Leaf estimation method and gradient iteration are set to match
                               # defaults for Logloss.
                               leaf_estimation_method="Newton", leaf_estimation_iterations=10, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    pred1 = model.predict(test_pool, prediction_type='RawFormulaVal')

    model2 = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, use_best_model=True, loss_function="Logloss")
    model2.fit(train_pool, eval_set=test_pool)
    pred2 = model2.predict(test_pool, prediction_type='RawFormulaVal')

    for p1, p2 in zip(pred1, pred2):
        assert abs(p1 - p2) < EPS


def test_pool_after_fit(task_type):
    pool1 = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool2 = Pool(TRAIN_FILE, column_description=CD_FILE)
    assert _check_data(pool1.get_features(), pool2.get_features())
    model = CatBoostClassifier(iterations=5, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool2)
    assert _check_data(pool1.get_features(), pool2.get_features())


def test_priors(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        random_seed=0,
        has_time=True,
        ctr_description=["Borders:Prior=0:Prior=0.6:Prior=1:Prior=5",
                         ("FeatureFreq" if task_type == 'GPU' else "Counter") + ":Prior=0:Prior=0.6:Prior=1:Prior=5"],
        task_type=task_type, devices='0',
    )
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_ignored_features(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model1 = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0', max_ctr_complexity=1, ignored_features=[1, 2, 3])
    model2 = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0', max_ctr_complexity=1)
    model1.fit(train_pool)
    model2.fit(train_pool)
    predictions1 = model1.predict_proba(test_pool)
    predictions2 = model2.predict_proba(test_pool)
    assert not _check_data(predictions1, predictions2)
    model1.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_class_weights(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, class_weights=[1, 2], task_type=task_type, devices='0')
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_classification_ctr(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0,
                               ctr_description=['Borders', 'FeatureFreq' if task_type == 'GPU' else 'Counter'],
                               task_type=task_type, devices='0')
    model.fit(pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


@fails_on_gpu(how="libs/options/catboost_options.cpp:280: Error: GPU doesn't not support target binarization per CTR description currently. Please use target_borders option instead")
def test_regression_ctr(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=5, learning_rate=0.03, random_seed=0, ctr_description=['Borders:TargetBorderCount=5:TargetBorderType=Uniform', 'Counter'], task_type=task_type, devices='0')
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


@fails_on_gpu(how="libs/algo/learn_context.h:110: Error: except learn on CPU task type, got GPU")
def test_cv(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    results = cv(pool, {
        "iterations": 5,
        "learning_rate": 0.03,
        "random_seed": 0,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": task_type,
    })
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@fails_on_gpu(how="libs/algo/learn_context.h:110: Error: except learn on CPU task type, got GPU")
def test_cv_query(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    results = cv(pool, {"iterations": 5, "learning_rate": 0.03, "random_seed": 0, "loss_function": "QueryRMSE", "task_type": task_type})
    assert "train-QueryRMSE-mean" in results

    prev_value = results["train-QueryRMSE-mean"][0]
    for value in results["train-QueryRMSE-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@fails_on_gpu(how="libs/algo/learn_context.h:110: Error: except learn on CPU task type, got GPU")
def test_cv_pairs(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE)
    results = cv(pool, {"iterations": 5, "learning_rate": 0.03, "random_seed": 8, "loss_function": "PairLogit", "task_type": task_type})
    assert "train-PairLogit-mean" in results

    prev_value = results["train-PairLogit-mean"][0]
    for value in results["train-PairLogit-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool)
    np.save(FIMP_NPY_PATH, np.array(model.feature_importances_))
    return local_canonical_file(FIMP_NPY_PATH)


def test_feature_importance_explicit(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool)
    np.save(FIMP_NPY_PATH, np.array(model.get_feature_importance(fstr_type=EFstrType.FeatureImportance)))
    return local_canonical_file(FIMP_NPY_PATH)


def test_feature_importance_prettified(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool)

    feature_importances = model.get_feature_importance(fstr_type=EFstrType.FeatureImportance, prettified=True)
    with open(FIMP_TXT_PATH, 'w') as ofile:
        for f_id, f_imp in feature_importances:
            ofile.write('{}\t{}\n'.format(f_id, f_imp))
    return local_canonical_file(FIMP_TXT_PATH)


def test_interaction_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(pool)
    np.save(FIMP_NPY_PATH, np.array(model.get_feature_importance(fstr_type=EFstrType.Interaction)))
    return local_canonical_file(FIMP_NPY_PATH)


def test_shap_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, random_seed=0, max_ctr_complexity=1, task_type=task_type, devices='0')
    model.fit(pool)
    np.save(FIMP_NPY_PATH, np.array(model.get_feature_importance(fstr_type=EFstrType.ShapValues, data=pool)))
    return local_canonical_file(FIMP_NPY_PATH)


def test_od(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.03, od_type='Iter', od_wait=20, random_seed=42, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


def test_clone(task_type):
    estimator = CatBoostClassifier(
        custom_metric="Accuracy",
        loss_function="MultiClass",
        iterations=400,
        learning_rate=0.03,
        task_type=task_type, devices='0')

    # This is important for sklearn.base.clone since
    # it uses get_params for cloning estimator.
    params = estimator.get_params()
    new_estimator = CatBoostClassifier(**params)
    new_params = new_estimator.get_params()

    for param in params:
        assert param in new_params
        assert new_params[param] == params[param]


def test_different_cat_features_order(task_type):
    dataset = np.array([[2, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    labels = [1.2, 3.4, 9.5, 24.5]

    pool1 = Pool(dataset, labels, cat_features=[0, 1])
    pool2 = Pool(dataset, labels, cat_features=[1, 0])

    model = CatBoost({'learning_rate': 1, 'loss_function': 'RMSE', 'iterations': 2, 'random_seed': 42, 'task_type': task_type, 'devices': '0'})
    model.fit(pool1)
    assert (model.predict(pool1) == model.predict(pool2)).all()


@fails_on_gpu(how='libs/options/json_helper.h:198: Error: change of option approx_on_full_history is unimplemented for task type GPU and was not default in previous run')
def test_full_history(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.03, od_type='Iter', od_wait=20, random_seed=42, approx_on_full_history=True, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    model.save_model(OUTPUT_MODEL_PATH)
    return compare_canonical_models(OUTPUT_MODEL_PATH)


@fails_on_gpu(how='libs/algo/learn_context.h:110: Error: except learn on CPU task type, got GPU')
def test_cv_logging(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    cv(pool, {"iterations": 5, "learning_rate": 0.03, "random_seed": 0, "loss_function": "Logloss", "task_type": task_type})
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@fails_on_gpu(how='libs/algo/learn_context.h:110: Error: except learn on CPU task type, got GPU')
def test_cv_with_not_binarized_target(task_type):
    train_file = data_file('adult_not_binarized', 'train_small')
    cd = data_file('adult_not_binarized', 'train.cd')
    pool = Pool(train_file, column_description=cd)
    cv(pool, {"iterations": 5, "learning_rate": 0.03, "random_seed": 0, "loss_function": "Logloss", "task_type": task_type})
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@pytest.mark.parametrize('loss_function', ['Logloss', 'RMSE', 'QueryRMSE'])
def test_eval_metrics(loss_function, task_type):
    train, test, cd, metric = TRAIN_FILE, TEST_FILE, CD_FILE, loss_function
    if loss_function == 'QueryRMSE':
        train, test, cd, metric = QUERYWISE_TRAIN_FILE, QUERYWISE_TEST_FILE, QUERYWISE_CD_FILE, 'PFound'
    if loss_function == 'Logloss':
        metric = 'AUC'

    train_pool = Pool(train, column_description=cd)
    test_pool = Pool(test, column_description=cd)
    model = CatBoost(params={'loss_function': loss_function, 'random_seed': 0, 'iterations': 20, 'thread_count': 8, 'eval_metric': metric,
                             'task_type': task_type, 'devices': '0', 'counter_calc_method': 'SkipTest'})

    model.fit(train_pool, eval_set=test_pool, use_best_model=False)
    first_metrics = np.loadtxt('catboost_info/test_error.tsv', skiprows=1)[:, 1]
    second_metrics = model.eval_metrics(test_pool, [metric])[metric]
    elemwise_reldiff = np.abs(first_metrics - second_metrics) / np.max((np.abs(first_metrics), np.abs(second_metrics)), 0)
    elemwise_absdiff = np.abs(first_metrics - second_metrics)
    elemwise_mindiff = np.min((elemwise_reldiff, elemwise_absdiff), 0)
    if task_type == 'GPU':
        assert np.all(abs(elemwise_mindiff) < 1e-7)
    else:
        assert np.all(abs(elemwise_mindiff) < 1e-9)


@pytest.mark.parametrize('loss_function', ['Logloss', 'RMSE', 'QueryRMSE'])
def test_eval_metrics_batch_calcer(loss_function, task_type):
    metric = loss_function
    if loss_function == 'QueryRMSE':
        train, test, cd = QUERYWISE_TRAIN_FILE, QUERYWISE_TEST_FILE, QUERYWISE_CD_FILE
        metric = 'PFound'
    else:
        train, test, cd = TRAIN_FILE, TEST_FILE, CD_FILE

    train_pool = Pool(train, column_description=cd)
    test_pool = Pool(test, column_description=cd)
    model = CatBoost(params={'loss_function': loss_function, 'random_seed': 0, 'iterations': 100, 'thread_count': 8,
                             'eval_metric': metric, 'task_type': task_type, 'devices': '0', 'counter_calc_method': 'SkipTest'})

    model.fit(train_pool, eval_set=test_pool, use_best_model=False)
    first_metrics = np.loadtxt('catboost_info/test_error.tsv', skiprows=1)[:, 1]

    calcer = model.create_metric_calcer([metric])
    calcer.add(test_pool)

    second_metrics = calcer.eval_metrics().get_result(metric)

    elemwise_reldiff = np.abs(first_metrics - second_metrics) / np.max((np.abs(first_metrics), np.abs(second_metrics)), 0)
    elemwise_absdiff = np.abs(first_metrics - second_metrics)
    elemwise_mindiff = np.min((elemwise_reldiff, elemwise_absdiff), 0)
    if task_type == 'GPU':
        assert np.all(abs(elemwise_mindiff) < 1e-6)
    else:
        assert np.all(abs(elemwise_mindiff) < 1e-9)


@fails_on_gpu(how='assert 0.001453466387789204 < EPS, where 0.001453466387789204 = abs((0.8572555206815472 - 0.8587089870693364))')
@pytest.mark.parametrize('catboost_class', [CatBoostClassifier, CatBoostRegressor])
def test_score_from_features_data(catboost_class, task_type):
    train_pool_from_files = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool_from_files = Pool(TEST_FILE, column_description=CD_FILE)
    model = catboost_class(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool_from_files)
    score_from_files = model.score(test_pool_from_files)

    train_features_data, test_features_data = [
        get_features_data_from_file(
            data_file=data_file,
            drop_columns=[TARGET_IDX],
            cat_feature_indices=train_pool_from_files.get_cat_feature_indices()
        )
        for data_file in [TRAIN_FILE, TEST_FILE]
    ]
    model.fit(X=train_features_data, y=train_pool_from_files.get_label())
    score_from_features_data = model.score(test_features_data, test_pool_from_files.get_label())

    assert abs(score_from_files - score_from_features_data) < EPS

    # empty
    empty_test_features_data = FeaturesData(
        num_feature_data=np.empty((0, test_features_data.get_num_feature_count()), dtype=np.float32),
        cat_feature_data=np.empty((0, test_features_data.get_cat_feature_count()), dtype=object)
    )
    score_from_features_data = model.score(empty_test_features_data, [])
    assert np.isnan(score_from_features_data)


@pytest.mark.parametrize('catboost_class', [CatBoostClassifier, CatBoostRegressor])
def test_call_score_with_pool_and_y(catboost_class):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = catboost_class(iterations=2)

    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    train_features, test_features = [
        get_features_data_from_file(
            data_file=data_file,
            drop_columns=[TARGET_IDX],
            cat_feature_indices=train_pool.get_cat_feature_indices()
        )
        for data_file in [TRAIN_FILE, TEST_FILE]
    ]
    train_target = train_pool.get_label()
    test_target = test_pool.get_label()
    test_pool_without_label = Pool(test_features)

    model.fit(train_pool)
    model.score(test_pool)

    with pytest.raises(CatboostError, message="Label in X has not initialized."):
        model.score(test_pool_without_label, test_target)

    with pytest.raises(CatboostError, message="Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool."):
        model.score(test_pool, test_target)

    with pytest.raises(CatboostError, message="Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool."):
        model.score(test_pool_without_label, test_target)

    model.fit(train_features, train_target)
    model.score(test_features, test_target)

    with pytest.raises(CatboostError, message="y should be specified."):
        model.score(test_features)


@fails_on_gpu(how="libs/algo/learn_context.h:110: Error: except learn on CPU task type, got GPU")
@pytest.mark.parametrize('verbose', [5, False, True])
def test_verbose_int(verbose, task_type):
    expected_line_count = {5: 3, False: 0, True: 10}
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile = 'test_data_dumps'

    with LogStdout(open(tmpfile, 'w')):
        cv(pool, {"iterations": 10, "learning_rate": 0.03, "random_seed": 0, "loss_function": "Logloss", "task_type": task_type}, verbose=verbose)
    with open(tmpfile, 'r') as output:
        line_conut = sum(1 for line in output)
        assert(line_conut == expected_line_count[verbose])

    with LogStdout(open(tmpfile, 'w')):
        train(pool, {"iterations": 10, "learning_rate": 0.03, "random_seed": 0, "loss_function": "Logloss", "task_type": task_type}, verbose=verbose)
    with open(tmpfile, 'r') as output:
        line_conut = sum(1 for line in output)
        assert(line_conut == expected_line_count[verbose])

    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_eval_set(task_type):
    dataset = [(1, 2, 3, 4), (2, 2, 3, 4), (3, 2, 3, 4), (4, 2, 3, 4)]
    labels = [1, 2, 3, 4]
    train_pool = Pool(dataset, labels, cat_features=[0, 3, 2])

    model = CatBoost({'learning_rate': 1, 'loss_function': 'RMSE', 'iterations': 2, 'random_seed': 0, 'task_type': task_type, 'devices': '0'})

    eval_dataset = [(5, 6, 6, 6), (6, 6, 6, 6)]
    eval_labels = [5, 6]
    eval_pool = (eval_dataset, eval_labels)

    model.fit(train_pool, eval_set=eval_pool)

    eval_pools = [eval_pool]

    model.fit(train_pool, eval_set=eval_pools)

    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_object_importances(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool = Pool(TEST_FILE, column_description=CD_FILE)

    model = CatBoost({'loss_function': 'RMSE', 'iterations': 10, 'random_seed': 0, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    indices, scores = model.get_object_importance(pool, train_pool, top_size=10)
    np.savetxt(OIMP_PATH, scores)

    return local_canonical_file(OIMP_PATH)


def test_shap(task_type):
    train_pool = Pool([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 5, 8], cat_features=[])
    test_pool = Pool([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = CatBoostRegressor(iterations=1, random_seed=0, max_ctr_complexity=1, depth=2, task_type=task_type, devices='0')
    model.fit(train_pool)
    shap_values = model.get_feature_importance(fstr_type=EFstrType.ShapValues, data=test_pool)

    dataset = [(0.5, 1.2), (1.6, 0.5), (1.8, 1.0), (0.4, 0.6), (0.3, 1.6), (1.5, 0.2)]
    labels = [1.1, 1.85, 2.3, 0.7, 1.1, 1.6]
    train_pool = Pool(dataset, labels, cat_features=[])

    model = CatBoost({'iterations': 10, 'random_seed': 0, 'max_ctr_complexity': 1, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)

    testset = [(0.6, 1.2), (1.4, 0.3), (1.5, 0.8), (1.4, 0.6)]
    predictions = model.predict(testset)
    shap_values = model.get_feature_importance(fstr_type=EFstrType.ShapValues, data=Pool(testset))
    assert(len(predictions) == len(shap_values))
    for pred_idx in range(len(predictions)):
        assert(abs(sum(shap_values[pred_idx]) - predictions[pred_idx]) < 1e-9)

    np.savetxt(FIMP_TXT_PATH, shap_values)
    return local_canonical_file(FIMP_TXT_PATH)


def test_shap_complex_ctr(task_type):
    pool = Pool([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 2]], [0, 0, 5, 8], cat_features=[0, 1, 2])
    model = train(pool, {'random_seed': 12302113, 'iterations': 100, 'task_type': task_type, 'devices': '0'})
    shap_values = model.get_feature_importance(fstr_type=EFstrType.ShapValues, data=pool)
    predictions = model.predict(pool)
    assert(len(predictions) == len(shap_values))
    for pred_idx in range(len(predictions)):
        assert(abs(sum(shap_values[pred_idx]) - predictions[pred_idx]) < 1e-9)

    np.savetxt(FIMP_TXT_PATH, shap_values)
    return local_canonical_file(FIMP_TXT_PATH)


def random_xy(num_rows, num_cols_x):
    x = np.random.randint(100, 104, size=(num_rows, num_cols_x))  # three cat values
    y = np.random.randint(0, 2, size=(num_rows))  # 0/1 labels
    return (x, y)


def save_and_give_path(y, x, filename):
    file = test_output_path(filename)
    np.savetxt(file, np.hstack((np.transpose([y]), x)), delimiter='\t', fmt='%i')
    return file


def test_multiple_eval_sets_no_empty():
    cat_features = [0, 3, 2]
    cd_file = test_output_path('cd.txt')
    with open(cd_file, 'wt') as cd:
        cd.write('0\tTarget\n')
        for feature_no in sorted(cat_features):
            cd.write('{}\tCateg\n'.format(1 + feature_no))

    x, y = random_xy(6, 4)
    train_pool = Pool(x, y, cat_features=cat_features)

    x0, y0 = random_xy(0, 4)  # empty tuple eval set
    x1, y1 = random_xy(3, 4)
    test0_file = save_and_give_path(y0, x0, 'test0.txt')  # empty file eval set

    with pytest.raises(CatboostError, message="Do not create Pool for empty data"):
        Pool(x0, y0, cat_features=cat_features)

    model = CatBoost({'learning_rate': 1, 'loss_function': 'RMSE', 'iterations': 2, 'random_seed': 0,
                      'allow_const_label': True})

    with pytest.raises(CatboostError, message="Do not fit with empty tuple in multiple eval sets"):
        model.fit(train_pool, eval_set=[(x1, y1), (x0, y0)], column_description=cd_file)

    with pytest.raises(CatboostError, message="Do not fit with empty file in multiple eval sets"):
        model.fit(train_pool, eval_set=[(x1, y1), test0_file], column_description=cd_file)

    with pytest.raises(CatboostError, message="Do not fit with None in multiple eval sets"):
        model.fit(train_pool, eval_set=[(x1, y1), None], column_description=cd_file)

    model.fit(train_pool, eval_set=[None], column_description=cd_file)


def test_multiple_eval_sets():
    # Know the seed to report it if assertion below fails
    seed = int(1000 * time.time()) % 0xffffffff
    np.random.seed(seed)

    def model_fit_with(train_set, test_sets, cd_file):
        model = CatBoost({'use_best_model': False, 'loss_function': 'RMSE', 'iterations': 12, 'random_seed': 0})
        model.fit(train_set, eval_set=list(reversed(test_sets)), column_description=cd_file)
        return model

    num_features = 11
    cat_features = range(num_features)
    cd_file = test_output_path('cd.txt')
    with open(cd_file, 'wt') as cd:
        cd.write('0\tTarget\n')
        for feature_no in sorted(cat_features):
            cd.write('{}\tCateg\n'.format(1 + feature_no))

    x, y = random_xy(12, num_features)
    train_pool = Pool(x, y, cat_features=cat_features)

    x1, y1 = random_xy(13, num_features)
    x2, y2 = random_xy(14, num_features)
    y2 = np.zeros_like(y2)

    test1_file = save_and_give_path(y1, x1, 'test1.txt')
    test2_pool = Pool(x2, y2, cat_features=cat_features)

    model0 = model_fit_with(train_pool, [test1_file, test2_pool], cd_file)
    model1 = model_fit_with(train_pool, [test2_pool, (x1, y1)], cd_file)
    model2 = model_fit_with(train_pool, [(x2, y2), test1_file], cd_file)

    # The three models above shall predict identically on a test set
    # (make sure they are trained with 'use_best_model': False)
    xt, yt = random_xy(7, num_features)
    test_pool = Pool(xt, yt, cat_features=cat_features)

    pred0 = model0.predict(test_pool)
    pred1 = model1.predict(test_pool)
    pred2 = model2.predict(test_pool)

    hash0 = hashlib.md5(pred0).hexdigest()
    hash1 = hashlib.md5(pred1).hexdigest()
    hash2 = hashlib.md5(pred2).hexdigest()

    assert hash0 == hash1 and hash1 == hash2, 'seed: ' + str(seed)


def test_get_metadata_notrain():
    model = CatBoost()
    with pytest.raises(CatboostError, message='Only string keys should be allowed'):
        model.get_metadata()[1] = '1'
    with pytest.raises(CatboostError, message='Only string values should be allowed'):
        model.get_metadata()['1'] = 1
    model.get_metadata()['1'] = '1'
    assert model.get_metadata().get('1', 'EMPTY') == '1'
    assert model.get_metadata().get('2', 'EMPTY') == 'EMPTY'
    for i in xrange(100):
        model.get_metadata()[str(i)] = str(i)
    del model.get_metadata()['98']
    with pytest.raises(KeyError):
        i = model.get_metadata()['98']
    for i in xrange(0, 98, 2):
        assert str(i) in model.get_metadata()
        del model.get_metadata()[str(i)]
    for i in xrange(0, 98, 2):
        assert str(i) not in model.get_metadata()
        assert str(i + 1) in model.get_metadata()


# tests for single objects

def get_single_object_from_dataset(pool):
    test_list = open(data_file(pool, 'test_small')).read().split('\n')[:-1]
    for i in range(len(test_list)):
        test_list[i] = test_list[i].split('\t')
        for j in range(len(test_list[i])):
            try:
                test_list[i][j] = float(test_list[i][j])
            except:
                test_list[i][j] = test_list[i][j]
    return random.choice(test_list)


@pytest.mark.parametrize('pool', ['higgs', 'adult'])
def test_single_object_features():
    train_pool = Pool(data_file(pool, 'train_small'), column_description=data_file(pool, 'train.cd'))
    test_object = get_single_object_from_dataset(pool)
    model = CatBoost({'iterations': 20, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    assert model.predict(test_object) == model.predict([test_object])[0]
    assert list(model.staged_predict(test_object)) == list(list(model.staged_predict([test_object]))[0])


@pytest.mark.parametrize('pool', 'adult')
def test_single_object_regressor():
    train_pool = Pool(data_file(pool, 'train_small'), column_description=data_file(pool, 'train.cd'))
    model = CatBoostRegressor(iterations=2, learning_rate=0.03, random_seed=0, task_type=task_type, devices='0')
    model.fit(train_pool)
    test_object = get_single_object_from_dataset(pool)
    assert model.predict(test_object) == model.predict([test_object])[0]
    assert list(model.staged_predict(test_object)) == list(list(model.staged_predict([test_object]))[0])


@pytest.mark.parametrize('pool', 'adult')
def test_single_object_classifier():
    train_pool = Pool(data_file(pool, 'train_small'), column_description=data_file(pool, 'train.cd'))
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, random_seed=0, loss_function='Logloss:border=0.5', task_type=task_type, devices='0')
    model.fit(train_pool)
    test_object = get_single_object_from_dataset(pool)
    assert model.predict(test_object) == model.predict([test_object])[0]
    assert list(model.staged_predict(test_object)) == list(list(model.staged_predict([test_object]))[0])
    assert model.predict_proba(test_object) == model.predict_proba([test_object])[0]
    assert list(model.staged_predict_proba(test_object)) == list(list(model.staged_predict_proba([test_object]))[0])


def test_metadata():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=2,
        learning_rate=0.03,
        random_seed=0,
        loss_function='Logloss:border=0.5',
        metadata={"type": "AAA", "postprocess": "BBB"}
    )
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH)

    model2 = CatBoost()
    model2.load_model(OUTPUT_MODEL_PATH)
    assert 'type' in model2.get_metadata()
    assert model2.get_metadata()['type'] == 'AAA'
    assert 'postprocess' in model2.get_metadata()
    assert model2.get_metadata()['postprocess'] == 'BBB'
    return compare_canonical_models(OUTPUT_MODEL_PATH)


@pytest.mark.parametrize('metric', ['Logloss', 'RMSE'])
def test_util_eval_metric(metric):
    metric_results = eval_metric([1, 0], [0.88, 0.22], metric)
    np.savetxt(PREDS_PATH, np.array(metric_results))
    return local_canonical_file(PREDS_PATH)


@pytest.mark.parametrize('metric', ['MultiClass', 'AUC'])
def test_util_eval_metric_multiclass(metric):
    metric_results = eval_metric([1, 0, 2], [[0.88, 0.22, 0.3], [0.21, 0.45, 0.1], [0.12, 0.32, 0.9]], metric)
    np.savetxt(PREDS_PATH, np.array(metric_results))
    return local_canonical_file(PREDS_PATH)


def test_option_used_ram_limit():
    for limit in [1000, 1234.56, 0, 0.0, 0.5,
                  '100', '34.56', '0', '0.0', '0.5',
                  '1.2mB', '1000b', '', None, 'none', 'inf']:
        CatBoost({'used_ram_limit': limit})

    for limit in [-1000, 'any', '-0.5', 'nolimit', 'oo']:
        try:
            CatBoost({'used_ram_limit': limit})
            assert False, "Shall not allow used_ram_limit={!r}".format(limit)
        except:
            assert True


def get_values_that_json_dumps_breaks_on():
    name_dtype = {name: np.__dict__[name] for name in dir(np) if (
        isinstance(np.__dict__[name], type)
        and re.match('(int|uint|float|bool).*', name)
    )}
    name_value = {}
    for name, dtype in name_dtype.items():
        try:
            value = dtype(1)
            if str(value).startswith('<'):
                continue
            name_value[name] = value
            name_value['array of ' + name] = np.array([[1, 0], [0, 1]], dtype=dtype)
        except:
            pass
    return name_value


def test_serialization_of_numpy_objects_internal():
    from catboost._catboost import _PreprocessParams
    _PreprocessParams(get_values_that_json_dumps_breaks_on())


def test_serialization_of_numpy_objects_save_model():
    train_pool = Pool(*random_xy(10, 5))
    model = CatBoostClassifier(
        iterations=np.int64(2),
        random_seed=np.int32(0),
        loss_function='Logloss'
    )
    model.fit(train_pool)
    model.save_model(OUTPUT_MODEL_PATH, format='coreml',
                     export_parameters=get_values_that_json_dumps_breaks_on())


def test_serialization_of_numpy_objects_execution_case():
    from catboost.eval.execution_case import ExecutionCase
    ExecutionCase(get_values_that_json_dumps_breaks_on())


@fails_on_gpu(how='assert 0 == 4')
def test_metric_period_redefinition(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile = 'test_data_dumps'
    model = CatBoost(dict(iterations=10, metric_period=3, task_type=task_type, devices='0'))

    with LogStdout(open(tmpfile, 'w')):
        model.fit(pool)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 4)

    with LogStdout(open(tmpfile, 'w')):
        model.fit(pool, metric_period=2)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 6)


@fails_on_gpu(how='AssertionError: (assertion failed, but when it was re-run for printing intermediate values, it did not fail.  Suggestions: compute assert expression before the assert or use --assert=plain)')  # noqa
def test_verbose_redefinition(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile = 'test_data_dumps'
    model = CatBoost(dict(iterations=10, verbose=False, task_type=task_type, devices='0'))

    with LogStdout(open(tmpfile, 'w')):
        model.fit(pool)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 0)

    with LogStdout(open(tmpfile, 'w')):
        model.fit(pool, verbose=True)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 10)


class TestInvalidCustomLossAndMetric(object):
    class GoodCustomLoss(object):
        def calc_ders_range(self, approxes, targets, weights):
            assert len(approxes) == len(targets)
            der1 = 2.0 * (np.array(approxes) - np.array(targets))
            der2 = -2.0
            if weights is not None:
                assert len(weights) == len(targets)
                der1 *= np.array(weights)
                der2 *= np.array(weights)
            return zip(der1, der2)

    class BadCustomLoss(object):
        def calc_ders_range(self, approxes, targets, weights):
            raise Exception('BadCustomLoss calc_ders_range')

        def calc_ders_multi(self, approxes, targets, weights):
            raise Exception('BadCustomLoss calc_ders_multi')

    class IncompleteCustomLoss(object):
        pass

    class GoodCustomMetric(object):
        def get_final_error(self, error, weight):
            return 0.0

        def is_max_optimal(self):
            return True

        def evaluate(self, approxes, target, weight):
            return (0.0, 0.0)

    class IncompleteCustomMetric(object):
        pass

    def test_loss_good_metric_none(self):
        with pytest.raises(CatboostError, match='metric is not defined|No metrics specified'):
            model = CatBoost({"loss_function": self.GoodCustomLoss(), "iterations": 2, "random_seed": 0})
            pool = Pool(*random_xy(10, 5))
            model.fit(pool)

    def test_loss_bad_metric_logloss(self):
        with pytest.raises(Exception, match='BadCustomLoss calc_ders_range'):
            model = CatBoost({"loss_function": self.BadCustomLoss(), "eval_metric": "Logloss", "iterations": 2, "random_seed": 0})
            pool = Pool(*random_xy(10, 5))
            model.fit(pool)

    def test_loss_bad_metric_multiclass(self):
        with pytest.raises(Exception, match='BadCustomLoss calc_ders_multi'):
            model = CatBoost({"loss_function": self.BadCustomLoss(), "eval_metric": "MultiClass", "iterations": 2, "random_seed": 0})
            pool = Pool(*random_xy(10, 5))
            model.fit(pool)

    def test_loss_incomplete_metric_logloss(self):
        with pytest.raises(Exception, match='has no.*calc_ders_range'):
            model = CatBoost({"loss_function": self.IncompleteCustomLoss(), "eval_metric": "Logloss", "iterations": 2, "random_seed": 0})
            pool = Pool(*random_xy(10, 5))
            model.fit(pool)

    def test_loss_incomplete_metric_multiclass(self):
        with pytest.raises(Exception, match='has no.*calc_ders_multi'):
            model = CatBoost({"loss_function": self.IncompleteCustomLoss(), "eval_metric": "MultiClass", "iterations": 2, "random_seed": 0})
            pool = Pool(*random_xy(10, 5))
            model.fit(pool)

    def test_custom_metric_object(self):
        with pytest.raises(CatboostError, match='custom_metric.*must be string'):
            model = CatBoost({"custom_metric": self.GoodCustomMetric(), "iterations": 2, "random_seed": 0})
            pool = Pool(*random_xy(10, 5))
            model.fit(pool)

    def test_loss_none_metric_good(self):
        model = CatBoost({"eval_metric": self.GoodCustomMetric(), "iterations": 2, "random_seed": 0})
        pool = Pool(*random_xy(10, 5))
        model.fit(pool)

    def test_loss_none_metric_incomplete(self):
        with pytest.raises(CatboostError, match='evaluate.*returned incorrect value'):
            model = CatBoost({"eval_metric": self.IncompleteCustomMetric(), "iterations": 2, "random_seed": 0})
            pool = Pool(*random_xy(10, 5))
            model.fit(pool)


def test_silent():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile = 'test_data_dumps'

    with LogStdout(open(tmpfile, 'w')):
        model = CatBoost(dict(iterations=10, silent=True))
        model.fit(pool)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 0)

    with LogStdout(open(tmpfile, 'w')):
        model = CatBoost(dict(iterations=10, silent=True))
        model.fit(pool, silent=False)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 10)

    with LogStdout(open(tmpfile, 'w')):
        train(pool, {'silent': True})
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 0)

    with LogStdout(open(tmpfile, 'w')):
        model = CatBoost(dict(iterations=10, silent=False))
        model.fit(pool, silent=True)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 0)

    with LogStdout(open(tmpfile, 'w')):
        model = CatBoost(dict(iterations=10, verbose=5))
        model.fit(pool, silent=True)
    with open(tmpfile, 'r') as output:
        assert(sum(1 for line in output) == 0)


def test_set_params_with_synonyms(task_type):
    params = {'num_trees': 20,
              'max_depth': 5,
              'learning_rate': 0.001,
              'logging_level': 'Silent',
              'loss_function': 'RMSE',
              'eval_metric': 'RMSE',
              'od_wait': 150,
              'random_seed': 8888,
              'task_type': task_type,
              'devices': '0'
              }

    model1 = CatBoostRegressor(**params)
    params_after_setting = model1.get_params()
    assert(params == params_after_setting)

    data = np.random.randint(10, size=(20, 20))
    label = np.random.randint(2, size=20)
    train_pool = Pool(data, label, cat_features=[1, 2])
    model1.fit(train_pool)
    model1.save_model('model.cb')

    model2 = CatBoost()
    model2.load_model('model.cb')
    params_after_save_model = model2.get_params()
    assert(params.keys() != params_after_save_model.keys())

    model2 = CatBoost()
    model2.set_params(**model1.get_params())
    assert(model1.get_params() == model2.get_params())

    state = model1.__getstate__()
    model2 = CatBoost()
    model2.__setstate__(state)
    assert(model1.get_params() == model2.get_params())


def test_feature_names_from_model():
    input_file = 'pool'
    with open(input_file, 'w') as inp:
        inp.write('0\t1\t2\t0\n1\t2\t3\t1\n')

    column_description1 = 'description1.cd'
    create_cd(
        label=3,
        cat_features=[0, 1],
        feature_names={0: 'a', 1: 'b', 2: 'ab'},
        output_path=column_description1
    )

    column_description2 = 'description2.cd'
    create_cd(
        label=3,
        cat_features=[0, 1],
        output_path=column_description2
    )

    column_description3 = 'description3.cd'
    create_cd(
        label=3,
        cat_features=[0, 1],
        feature_names={0: 'a', 2: 'ab'},
        output_path=column_description3
    )

    pools = [
        Pool(input_file, column_description=column_description1),
        Pool(input_file, column_description=column_description2),
        Pool(input_file, column_description=column_description3)
    ]

    output_file = 'feature_names'
    with open(output_file, 'w') as output:
        for i in range(len(pools)):
            pool = pools[i]
            model = CatBoost(dict(iterations=10))
            try:
                print(model.feature_names_)
            except CatboostError:
                pass
            else:
                assert False
            model.fit(pool)
            output.write(str(model.feature_names_) + '\n')

    return local_canonical_file(output_file)


Value_AcceptableAsEmpty = [
    ('', True),
    ('nan', True),
    ('NaN', True),
    ('NAN', True),
    ('NA', True),
    ('Na', True),
    ('na', True),
    ('-', False),
    ('n/a', False),
    ('junk', False),
    ('None', False),
]


class TestMissingValues(object):

    def assert_expected(self, pool):
        assert str(pool.get_features()) == str([[1.0], [float('nan')]])

    @pytest.mark.parametrize('value,value_acceptable_as_empty', [(None, True)] + Value_AcceptableAsEmpty)
    @pytest.mark.parametrize('object', [list, np.array, DataFrame, Series])
    def test_create_pool_from_object(self, value, value_acceptable_as_empty, object):
        if value_acceptable_as_empty:
            self.assert_expected(Pool(object([[1], [value]])))
            self.assert_expected(Pool(object([1, value])))
        else:
            with pytest.raises(TypeError):
                Pool(object([1, value]))

    @pytest.mark.parametrize('value,value_acceptable_as_empty', Value_AcceptableAsEmpty)
    def test_create_pool_from_file(self, value, value_acceptable_as_empty):
        pool_path = test_output_path('pool')
        open(pool_path, 'wt').write('1\t1\n0\t{}\n'.format(value))
        if value_acceptable_as_empty:
            self.assert_expected(Pool(pool_path))
        else:
            with pytest.raises(CatboostError):
                Pool(pool_path)


def test_model_and_pool_compatibility():
    pool1 = Pool([[0, 0, 0], [1, 1, 1]], [0, 1], cat_features=[0, 1])
    pool2 = Pool([[0, 0, 0], [1, 1, 1]], [0, 1], cat_features=[1, 2])
    model = CatBoostRegressor(iterations=1)
    model.fit(pool1)
    with pytest.raises(CatboostError):
        model.get_feature_importance(fstr_type=EFstrType.ShapValues, data=pool2)


def test_shap_verbose():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)

    model = CatBoost(dict(iterations=250))
    model.fit(pool)

    tmpfile = 'test_data_dumps'
    with LogStdout(open(tmpfile, 'w')):
        model.get_feature_importance(fstr_type=EFstrType.ShapValues, data=pool, verbose=12)
    with open(tmpfile, 'r') as output:
        line_count = sum(1 for line in output)
        assert(line_count == 5)


def test_eval_set_with_nans(task_type):
    features = np.random.random((10, 200))
    labels = np.random.random((10,))
    features_with_nans = features.copy()
    np.putmask(features_with_nans, features_with_nans < 0.5, np.nan)
    model = CatBoost({'iterations': 2, 'random_seed': 0, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    train_pool = Pool(features, label=labels)
    test_pool = Pool(features_with_nans, label=labels)
    model.fit(train_pool, eval_set=test_pool)


def test_learning_rate_auto_set(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model1 = CatBoostClassifier(iterations=10, random_seed=0, task_type=task_type, devices='0')
    model1.fit(train_pool)
    predictions1 = model1.predict_proba(test_pool)

    model2 = CatBoostClassifier(iterations=10, learning_rate=model1.learning_rate_, random_seed=0, task_type=task_type, devices='0')
    model2.fit(train_pool)
    predictions2 = model2.predict_proba(test_pool)
    assert _check_data(predictions1, predictions2)
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@fails_on_gpu(how='libs/algo/learn_context.h:110: Error: except learn on CPU task type, got GPU')
def test_learning_rate_auto_set_in_cv(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    results = cv(pool, {"iterations": 5, "random_seed": 0, "loss_function": "Logloss", "task_type": task_type})
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@fails_on_gpu(how='cuda/train_lib/train.cpp:283: Error: loss function is not supported for GPU learning MultiClass')
def test_shap_multiclass(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=10, random_seed=0, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(pool)
    pred = classifier.predict(pool, prediction_type='RawFormulaVal')
    shap_values = classifier.get_feature_importance(
        fstr_type=EFstrType.ShapValues,
        data=pool,
        thread_count=8
    )
    features_count = pool.num_col()
    assert len(pred) == len(shap_values)
    result = []
    for i in range(len(pred)):
        result_for_doc = []
        for j in range(len(pred[i])):
            result_for_doc = result_for_doc + list(shap_values[i][j])
            assert len(shap_values[i][j]) == features_count + 1
            s = sum(shap_values[i][j])
            assert abs(s - pred[i][j]) < EPS
        result.append(result_for_doc)
    result = np.array([np.array([value for value in doc]) for doc in result])
    np.savetxt(FIMP_TXT_PATH, result)
    return local_canonical_file(FIMP_TXT_PATH)


def test_loading_pool_with_numpy_int():
    assert _check_shape(Pool(np.array([[2, 2], [1, 2]]), [1.2, 3.4], cat_features=[0]), object_count=2, features_count=2)


def test_loading_pool_with_numpy_str():
    assert _check_shape(Pool(np.array([['abc', '2'], ['1', '2']]), np.array([1, 3]), cat_features=[0]), object_count=2, features_count=2)


def test_loading_pool_with_lists():
    assert _check_shape(Pool([['abc', 2], ['1', 2]], [1, 3], cat_features=[0]), object_count=2, features_count=2)


def test_pairs_generation(task_type):
    model = CatBoost({"loss_function": "PairLogit", "iterations": 2, "random_seed": 0, "task_type": task_type})
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model.fit(pool)
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


@fails_on_gpu(how="cuda/methods/dynamic_boosting.h:169: Error: pool has just 3 groups or docs, can't use #1 GPUs to learn on such small pool")
def test_pairs_generation_generated(task_type):
    model = CatBoost(params={'loss_function': 'PairLogit', 'random_seed': 0, 'iterations': 10, 'thread_count': 8, 'task_type': task_type, 'devices': '0'})

    df = read_table(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    df = df.loc[:10, :]
    train_target = df.loc[:, 2]
    train_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    df = read_table(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    train_group_id = np.sort(np.random.randint(len(train_target) // 3, size=len(train_target)) + 1)
    pairs = []
    for idx1 in range(len(train_group_id)):
        idx2 = idx1 + 1
        while idx2 < len(train_group_id) and train_group_id[idx1] == train_group_id[idx2]:
            if train_target[idx1] > train_target[idx2]:
                pairs.append((idx1, idx2))
            if train_target[idx1] < train_target[idx2]:
                pairs.append((idx2, idx1))
            idx2 += 1
    model.fit(train_data, train_target, group_id=train_group_id, pairs=pairs)
    predictions1 = model.predict(train_data)
    predictions_on_test1 = model.predict(test_data)
    model.fit(train_data, train_target, group_id=train_group_id)
    predictions2 = model.predict(train_data)
    predictions_on_test2 = model.predict(test_data)
    assert all(predictions1 == predictions2)
    assert all(predictions_on_test1 == predictions_on_test2)


def test_pairs_generation_with_max_pairs(task_type):
    model = CatBoost({"loss_function": "PairLogit:max_pairs=30", "iterations": 2, "random_seed": 0, "task_type": task_type})
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model.fit(pool)
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_early_stopping_rounds(task_type):
    train_pool = Pool([[0], [1]], [0, 1])
    test_pool = Pool([[0], [1]], [1, 0])

    model = CatBoostRegressor(od_type='Iter', od_pval=2)
    with pytest.raises(CatboostError):
        model.fit(train_pool, eval_set=test_pool)

    model = CatBoost(params={'od_pval': 0.001, 'early_stopping_rounds': 2})
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=1)

    model = CatBoostClassifier(loss_function='Logloss:hints=skip_train~true', iterations=1000,
                               learning_rate=0.03, od_type='Iter', od_wait=10, random_seed=0)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=1)

    model = train(pool=train_pool, eval_set=test_pool, early_stopping_rounds=2,
                  params={'loss_function': 'Logloss:hints=skip_train~true',
                          'json_log': 'json_log_train.json', 'random_seed': 0})

    return [local_canonical_file(remove_time_from_json(JSON_LOG_PATH)),
            local_canonical_file(remove_time_from_json('catboost_info/json_log_train.json'))]


def test_slice_pool():
    pool = Pool([[0], [1], [2]], [0, 1, 2], pairs=[(0, 1), (0, 2), (1, 2)])
    with pytest.raises(CatboostError):
        pool.slice([0, 3, 1])
    rindexes = [
        [0],
        [2],
        [0, 0, 0],
        [0, 2, 1, 0],
        np.array([2, 2])
    ]
    for rindex in rindexes:
        sliced_pool = pool.slice(rindex)
        assert sliced_pool.get_label() == list(rindex)


def test_str_metrics_in_eval_metrics(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=40, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    first_metrics = model.eval_metrics(data=train_pool, metrics='Logloss')
    second_metrics = model.eval_metrics(data=train_pool, metrics=['Logloss'])
    assert np.all(np.array(first_metrics['Logloss']) == np.array(second_metrics['Logloss']))


def test_str_eval_metrics_in_eval_features():
    learn_params = {
        'iterations': 20, 'learning_rate': 0.5, 'random_seed': 0,
        'logging_level': 'Silent', 'loss_function': 'RMSE', 'boosting_type': 'Plain', 'allow_const_label': True}
    evaluator = CatboostEvaluation(
        TRAIN_FILE, fold_size=2, fold_count=2,
        column_description=CD_FILE, partition_random_seed=0)
    first_result = evaluator.eval_features(learn_config=learn_params, eval_metrics='MAE', features_to_eval=[6, 7, 8])
    second_result = evaluator.eval_features(learn_config=learn_params, eval_metrics=['MAE'], features_to_eval=[6, 7, 8])
    assert first_result.get_results()['MAE'] == second_result.get_results()['MAE']


# check different sizes as well as passing as int as well as str
@pytest.mark.parametrize('used_ram_limit', ['1024', '2Gb'])
def test_allow_writing_files_and_used_ram_limit(used_ram_limit, task_type):
    train_pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    test_pool = Pool(AIRLINES_5K_TEST_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    model = CatBoostClassifier(
        use_best_model=False,
        allow_writing_files=False,
        used_ram_limit=int(used_ram_limit) if used_ram_limit.isdigit() else used_ram_limit,
        max_ctr_complexity=8,
        depth=10,
        boosting_type='Plain',
        iterations=20,
        learning_rate=0.03,
        thread_count=4,
        random_seed=0,
        task_type=task_type, devices='0',
    )
    model.fit(train_pool, eval_set=test_pool)
    pred = model.predict(test_pool)
    np.save(PREDS_PATH, np.array(pred))
    return local_canonical_file(PREDS_PATH)


def test_roc():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    cv(
        train_pool,
        params={
            'loss_function': 'Logloss',
            'iterations': 10,
            'roc_file': 'out_cv',
            'random_seed': 0,
            'thread_count': 4
        }
    )

    model = CatBoostClassifier(loss_function='Logloss', iterations=20, random_seed=0)
    model.fit(train_pool)

    curve = get_roc_curve(model, test_pool, thread_count=4)
    table = np.array(zip(curve[2], [1 - x for x in curve[1]], curve[0]))
    np.savetxt('out_model', table)

    try:
        select_threshold(model, data=test_pool, FNR=0.5, FPR=0.5)
        assert False, 'Only one of FNR, FPR must be defined.'
    except CatboostError:
        pass

    with open('bounds', 'w') as f:
        fnr_boundary = select_threshold(model, data=test_pool, FNR=0.4)
        fpr_boundary = select_threshold(model, data=test_pool, FPR=0.2)
        inter_boundary = select_threshold(model, data=test_pool)

        try:
            select_threshold(model, data=test_pool, curve=curve)
            assert False, 'Only one of data and curve parameters must be defined.'
        except CatboostError:
            pass

        assert fnr_boundary == select_threshold(model, curve=curve, FNR=0.4)
        assert fpr_boundary == select_threshold(model, curve=curve, FPR=0.2)
        assert inter_boundary == select_threshold(model, curve=curve)

        f.write('by FNR=0.4: ' + str(fnr_boundary) + '\n')
        f.write('by FPR=0.2: ' + str(fpr_boundary) + '\n')
        f.write('by intersection: ' + str(inter_boundary) + '\n')

    return [
        local_canonical_file('catboost_info/out_cv'),
        local_canonical_file('out_model'),
        local_canonical_file('bounds')
    ]


@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
@pytest.mark.parametrize('overfitting_detector_type', OVERFITTING_DETECTOR_TYPE)
def test_overfit_detector_with_resume_from_snapshot_and_metric_period(boosting_type, overfitting_detector_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    FIRST_ITERATIONS = 8
    FINAL_ITERATIONS = 100
    # Overfitting must occur between the FIRST_ITERATIONS and FINAL_ITERATIONS.

    models = []

    for metric_period in [1, 5]:
        final_training_stdout_len_wo_snapshot = None

        # must be always run first without snapshot then with it to properly test assertions
        for with_resume_from_snapshot in [False, True]:
            model = CatBoostClassifier(
                use_best_model=False,
                boosting_type=boosting_type,
                thread_count=4,
                random_seed=0,
                learning_rate=0.2,
                od_type=overfitting_detector_type,
                metric_period=metric_period
            )
            if overfitting_detector_type == 'IncToDec':
                model.set_params(od_wait=10, od_pval=0.5)
            elif overfitting_detector_type == 'Iter':
                model.set_params(od_wait=10)
            if with_resume_from_snapshot:
                model.set_params(
                    save_snapshot=True,
                    snapshot_file=test_output_path(
                        'snapshot_with_metric_period={}_od_type={}'.format(
                            metric_period, overfitting_detector_type
                        )
                    )
                )
                model.set_params(iterations=FIRST_ITERATIONS)
                with tempfile.TemporaryFile('w+') as stdout_part:
                    with DelayedTee(sys.stdout, stdout_part):
                        model.fit(train_pool, eval_set=test_pool)
                    first_training_stdout_len = sum(1 for line in stdout_part)
                # overfitting detector has not stopped learning yet
                assert model.tree_count_ == FIRST_ITERATIONS
            else:
                model.set_params(save_snapshot=False)

            model.set_params(iterations=FINAL_ITERATIONS)
            with tempfile.TemporaryFile('w+') as stdout_part:
                with DelayedTee(sys.stdout, stdout_part):
                    model.fit(train_pool, eval_set=test_pool)
                final_training_stdout_len = sum(1 for line in stdout_part)

            if with_resume_from_snapshot:
                final_training_stdout_len_with_snapshot = final_training_stdout_len
                assert (
                    (final_training_stdout_len_with_snapshot - first_training_stdout_len)
                    ==
                    (
                        (models[0].tree_count_ / metric_period - FIRST_ITERATIONS / metric_period)
                        - (FIRST_ITERATIONS - 2) / metric_period
                        - 1
                    )
                )
                assert (
                    (
                        final_training_stdout_len_with_snapshot
                        + first_training_stdout_len
                        - 2 * final_training_stdout_len_wo_snapshot
                    )
                    ==
                    (
                        (FIRST_ITERATIONS - 2) / metric_period
                        + (models[0].tree_count_ / metric_period - FIRST_ITERATIONS / metric_period)
                        - 2 * ((models[0].tree_count_ - 1) / metric_period)
                        - 1
                    )
                )
            else:
                final_training_stdout_len_wo_snapshot = final_training_stdout_len

            models.append(model)

    canon_model_output = test_output_path('model.bin')
    models[0].save_model(canon_model_output)

    # overfitting detector stopped learning
    assert models[0].tree_count_ < FINAL_ITERATIONS

    for model2 in models[1:]:
        model_output = test_output_path('model2.bin')
        model2.save_model(model_output)
        subprocess.check_call((model_diff_tool, canon_model_output, model_output))


def test_use_loss_if_no_eval_metric():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    args = {
        'iterations': 100,
        'loss_function': 'Logloss',
        'use_best_model': True,
        'random_seed': 0
    }

    model_1 = CatBoostClassifier(**args)
    model_1.fit(train_pool, eval_set=test_pool)

    args['custom_metric'] = ['AUC', 'Precision']
    model_2 = CatBoostClassifier(**args)
    model_2.fit(train_pool, eval_set=test_pool)

    assert model_1.tree_count_ == model_2.tree_count_


def test_use_loss_if_no_eval_metric_cv():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    params = {
        'iterations': 50,
        'loss_function': 'Logloss',
        'random_seed': 0,
        'logging_level': 'Silent'
    }

    cv_params = {
        'params': params,
        'seed': 0,
        'nfold': 3,
        'early_stopping_rounds': 5
    }

    results_1 = cv(train_pool, **cv_params)

    cv_params['params']['custom_metric'] = ['AUC']
    results_2 = cv(train_pool, **cv_params)

    cv_params['params']['custom_metric'] = []
    cv_params['params']['eval_metric'] = 'AUC'
    results_3 = cv(train_pool, **cv_params)

    assert results_1.shape[0] == results_2.shape[0] and results_2.shape[0] != results_3.shape[0]


@pytest.mark.parametrize('metrics', [
    {'custom_metric': ['Accuracy', 'Logloss'], 'eval_metric': None},
    {'custom_metric': ['Accuracy', 'Accuracy'], 'eval_metric': None},
    {'custom_metric': ['Accuracy'], 'eval_metric': 'Logloss'},
    {'custom_metric': ['Accuracy'], 'eval_metric': 'Accuracy'},
    {'custom_metric': ['Accuracy', 'Logloss'], 'eval_metric': 'Logloss'}])
def test_no_fail_if_metric_is_repeated_cv(metrics):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    params = {
        'iterations': 10,
        'loss_function': 'Logloss',
        'custom_metric': metrics['custom_metric'],
        'logging_level': 'Silent'
    }
    if metrics['eval_metric'] is not None:
        params['eval_metric'] = metrics['eval_metric']

    cv_params = {
        'params': params,
        'nfold': 2,
        'as_pandas': True
    }

    cv(train_pool, **cv_params)


def test_use_last_testset_for_best_iteration():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    test_size = test_pool.num_row()
    half_size = test_size // 2
    test_pool_1 = test_pool.slice(range(half_size))
    test_pool_2 = test_pool.slice(range(half_size, test_size))
    metric = 'Logloss'

    args = {
        'iterations': 100,
        'loss_function': metric,
        'random_seed': 0
    }

    model = CatBoostClassifier(**args)
    model.fit(train_pool, eval_set=[test_pool_1, test_pool_2])
    pool_1_best_iter = np.argmin(model.eval_metrics(test_pool_1, metrics=[metric])[metric])
    pool_2_best_iter = np.argmin(model.eval_metrics(test_pool_2, metrics=[metric])[metric])
    assert pool_1_best_iter != pool_2_best_iter

    args['use_best_model'] = True
    best_model = CatBoostClassifier(**args)
    best_model.fit(train_pool, eval_set=[test_pool_1, test_pool_2])

    assert best_model.tree_count_ == pool_2_best_iter + 1


def test_best_model_min_trees(task_type):
    train_pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    test_pool = Pool(AIRLINES_5K_TEST_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    learn_params = {
        'iterations': 200,
        'random_seed': 0,
        'use_best_model': True,
        'task_type': task_type,
        'learning_rate': 0.2
    }
    model_1 = CatBoostClassifier(**learn_params)
    model_1.fit(train_pool, eval_set=test_pool)

    learn_params['best_model_min_trees'] = 100
    model_2 = CatBoostClassifier(**learn_params)
    model_2.fit(train_pool, eval_set=test_pool)

    assert model_1.tree_count_ < learn_params['best_model_min_trees']
    assert model_2.tree_count_ >= learn_params['best_model_min_trees']


class Metrics(object):

    @staticmethod
    def filter_regression(names):
        supported_by = {
            'MAE',
            'MAPE',
            'Poisson',
            'Quantile',
            'RMSE',
            'LogLinQuantile',
            'SMAPE',
            'R2',
            'MSLE',
            'MedianAbsoluteError',
        }
        return filter(lambda name: name in supported_by, names)

    @staticmethod
    def filter_binclass(cases):
        supported_by = {
            'Logloss',
            'CrossEntropy',
            'Precision',
            'Recall',
            'F1',
            'BalancedAccuracy',
            'BalancedErrorRate',
            'MCC',
            'Accuracy',
            'CtrFactor',
            'AUC',
            'BrierScore',
            'HingeLoss',
            'HammingLoss',
            'ZeroOneLoss',
            'Kappa',
            'WKappa',
            'LogLikelihoodOfPrediction',
        }
        good = re.compile(r'^({})(\W|$)'.format('|'.join(supported_by)))
        return filter(lambda case: good.match(case), cases)

    @staticmethod
    def filter_multiclass(cases):
        supported_by = {
            'MultiClass',
            'MultiClassOneVsAll',
            'Precision',
            'Recall',
            'F1',
            'TotalF1',
            'MCC',
            'Accuracy',
            'AUC',
            'HingeLoss',
            'HammingLoss',
            'ZeroOneLoss',
            'Kappa',
            'WKappa',
        }
        good = re.compile(r'^({})(\W|$)'.format('|'.join(supported_by)))
        return filter(lambda case: good.match(case), cases)

    @staticmethod
    def filter_ranking(cases):
        supported_by = {
            'PairLogit',
            'PairLogitPairwise',
            'PairAccuracy',
            'YetiRank',
            'YetiRankPairwise',
            'QueryRMSE',
            'QuerySoftMax',
            'PFound',
            'NDCG',
            'AverageGain'
        }
        good = re.compile(r'^({})(\W|$)'.format('|'.join(supported_by)))
        return filter(lambda case: good.match(case), cases)

    @staticmethod
    def filter_use_weights(cases):
        not_supported_by = {
            'BrierScore',
            'Kappa',
            'MedianAbsoluteError',
            'UserDefinedPerObject',
            'UserDefinedQuerywise',
            'WKappa',
        }
        bad = re.compile(r'^({})(\W|$)'.format('|'.join(not_supported_by)))
        return filter(lambda case: not bad.match(case), cases)

    def __init__(self, query):
        cases = {
            'Accuracy',
            'AUC',
            'BalancedAccuracy',
            'BalancedErrorRate',
            'BrierScore',
            'CrossEntropy',
            'CtrFactor',
            'Custom',
            'F1',
            'HammingLoss',
            'HingeLoss',
            'Kappa',
            'LogLinQuantile',
            'Logloss',
            'MAE',
            'MAPE',
            'MCC',
            'MedianAbsoluteError',
            'MSLE',
            'MultiClass',
            'MultiClassOneVsAll',
            'NDCG',
            'PairAccuracy',
            'PairLogit',
            'PairLogitPairwise',
            'PFound',
            'Poisson',
            'Precision',
            'Quantile',
            'QueryAverage:top=5',
            'QueryCrossEntropy',
            'QueryRMSE',
            'QuerySoftMax',
            'R2',
            'Recall',
            'RMSE',
            'SMAPE',
            'TotalF1',
            'UserDefinedPerObject',
            'UserDefinedQuerywise',
            'WKappa',
            'YetiRank',
            'YetiRankPairwise',
            'ZeroOneLoss',
            'LogLikelihoodOfPrediction',
        }
        for attr in query.split():
            if attr.startswith('-'):
                cases.difference_update(getattr(self, 'filter_' + attr[1:])(cases))
            else:
                cases.intersection_update(getattr(self, 'filter_' + attr)(cases))
        self.cases = cases

    def get_cases(self):
        return self.cases


class TestUseWeights(object):

    @pytest.fixture
    def a_regression_learner(self, task_type):
        train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        test_pool = Pool(TEST_FILE, column_description=CD_FILE)
        map(set_random_weight, (train_pool, test_pool))
        map(set_random_target, (train_pool, test_pool))
        cb = CatBoostRegressor(loss_function='RMSE', iterations=3, random_seed=0, task_type=task_type, devices='0')
        cb.fit(train_pool)
        return (cb, test_pool)

    @pytest.fixture
    def a_classification_learner(self, task_type):
        train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        test_pool = Pool(TEST_FILE, column_description=CD_FILE)
        map(set_random_weight, (train_pool, test_pool))
        map(set_random_target_01, (train_pool, test_pool))
        cb = CatBoostClassifier(loss_function='Logloss', iterations=3, random_seed=0, task_type=task_type, devices='0')
        cb.fit(train_pool)
        return (cb, test_pool)

    @pytest.fixture
    def a_multiclass_learner(self, task_type):
        train_pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
        test_pool = Pool(CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE)
        map(set_random_weight, (train_pool, test_pool))
        cb = CatBoostClassifier(loss_function='MultiClass', iterations=3, random_seed=0, use_best_model=False, task_type=task_type, devices='0')
        cb.fit(train_pool)
        return (cb, test_pool)

    def a_ranking_learner(self, task_type, metric):
        train_pool = Pool(QUERYWISE_TRAIN_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE_WITH_PAIR_WEIGHT, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
        test_pool = Pool(QUERYWISE_TEST_FILE, pairs=QUERYWISE_TEST_PAIRS_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
        if metric == 'QueryRMSE':
            loss_function = 'QueryRMSE'
        else:
            loss_function = 'PairLogit'

        cb = CatBoost({"loss_function": loss_function, "iterations": 3, "random_seed": 0, 'task_type': task_type, 'devices': '0'})
        cb.fit(train_pool)
        return (cb, test_pool)

    @pytest.mark.parametrize('metric_name', Metrics('use_weights regression').get_cases())
    def test_regression_metric(self, a_regression_learner, metric_name):
        cb, test_pool = a_regression_learner
        self.conclude(cb, test_pool, metric_name)

    @pytest.mark.parametrize('metric_name', Metrics('use_weights binclass').get_cases())
    def test_classification_metric(self, a_classification_learner, metric_name):
        cb, test_pool = a_classification_learner
        self.conclude(cb, test_pool, metric_name)

    @fails_on_gpu(how='cuda/train_lib/train.cpp:283: Error: loss function is not supported for GPU learning MultiClass')
    @pytest.mark.parametrize('metric_name', Metrics('use_weights multiclass').get_cases())
    def test_multiclass_metric(self, a_multiclass_learner, metric_name):
        cb, test_pool = a_multiclass_learner
        self.conclude(cb, test_pool, metric_name)

    @pytest.mark.parametrize('metric_name', Metrics('use_weights ranking').get_cases())
    def test_ranking_metric(self, task_type, metric_name):
        cb, test_pool = self.a_ranking_learner(task_type, metric_name)
        self.conclude(cb, test_pool, metric_name)

    def conclude(self, learner, test_pool, metric_name):
        metrics = [append_param(metric_name, 'use_weights=false'),
                   append_param(metric_name, 'use_weights=true')]
        eval_metrics = learner.eval_metrics(test_pool, metrics=metrics)
        assert eval_metrics != [], 'empty eval_metrics for metric `{}`'.format(metric_name)
        values = dict()
        use_weights_has_effect = False
        for name, value in eval_metrics.items():
            m = re.match(r'(.*)(use_weights=\w+)(.*)', name)
            assert m, "missed mark `use_weights` in eval_metric name `{}`\n\teval_metrics={}".format(name, eval_metrics)
            name_class = m.group(1) + m.group(3)
            if name_class not in values:
                values[name_class] = (name, value)
            else:
                value_a = values[name_class][1]
                value_b = value
                map(verify_finite, (value_a, value_b))
                use_weights_has_effect = value_a != value_b
                del values[name_class]
            if use_weights_has_effect:
                break
        assert use_weights_has_effect, "param `use_weights` has no effect\n\teval_metrics={}".format(eval_metrics)


def test_set_cat_features_in_init():
    data = np.random.randint(10, size=(20, 20))
    label = np.random.randint(2, size=20)
    train_pool = Pool(data, label, cat_features=[1, 2])
    test_pool = Pool(data, label, cat_features=[1, 2])

    params = {
        'logging_level': 'Silent',
        'loss_function': 'Logloss',
        'iterations': 10,
        'random_seed': 20
    }

    model1 = CatBoost(params)
    model1.fit(train_pool)

    params_with_cat_features = params.copy()
    params_with_cat_features['cat_features'] = model1.get_cat_feature_indices()

    model2 = CatBoost(params_with_cat_features)
    model2.fit(train_pool)
    assert(model1.get_cat_feature_indices() == model2.get_cat_feature_indices())

    model1 = CatBoost(params)
    model1.fit(train_pool)
    params_with_wrong_cat_features = params.copy()
    params_with_wrong_cat_features['cat_features'] = [0, 2]
    model2 = CatBoost(params_with_wrong_cat_features)
    with pytest.raises(CatboostError):
        model2.fit(train_pool)

    model1 = CatBoost(params_with_cat_features)
    model1.fit(X=data, y=label)
    model2 = model1
    model2.fit(X=data, y=label)
    assert(np.array_equal(model1.predict(test_pool), model2.predict(test_pool)))

    model1 = CatBoost(params_with_cat_features)
    with pytest.raises(CatboostError):
        model1.fit(X=data, y=label, cat_features=[1, 3])

    model1 = CatBoost(params_with_cat_features)
    model1.fit(X=data, y=label, eval_set=(data, label))
    assert(model1.get_cat_feature_indices() == [1, 2])

    model1 = CatBoost(params_with_wrong_cat_features)
    with pytest.raises(CatboostError):
        model1.fit(X=data, y=label, eval_set=test_pool)

    model1 = CatBoost(params_with_cat_features)
    state = model1.__getstate__()
    model2 = CatBoost()
    model2.__setstate__(state)
    model1.fit(X=data, y=label)
    model2.fit(X=data, y=label)
    assert(np.array_equal(model1.predict(test_pool), model2.predict(test_pool)))
    assert(model2.get_cat_feature_indices() == [1, 2])

    model1 = CatBoost(params_with_cat_features)
    model2 = CatBoost()
    model2.set_params(**model1.get_params())
    model1.fit(X=data, y=label)
    model2.fit(X=data, y=label)
    assert(np.array_equal(model1.predict(test_pool), model2.predict(test_pool)))
    assert(model2.get_cat_feature_indices() == [1, 2])

    model1 = CatBoost(params_with_cat_features)
    state = model1.__getstate__()
    model2 = CatBoostClassifier()
    model2.__setstate__(state)
    model1.fit(X=data, y=label)
    model2.fit(X=data, y=label)
    assert(np.array_equal(model1.predict(test_pool), model2.predict(test_pool, prediction_type='RawFormulaVal')))
    assert(model2.get_cat_feature_indices() == [1, 2])

    model1 = CatBoost(params_with_cat_features)
    model2 = CatBoostClassifier()
    model2.set_params(**model1.get_params())
    model1.fit(X=data, y=label)
    model2.fit(X=data, y=label)
    assert(np.array_equal(model1.predict(test_pool), model2.predict(test_pool, prediction_type='RawFormulaVal')))
    assert(model2.get_cat_feature_indices() == [1, 2])


def test_deprecated_behavoir():
    data = np.random.randint(10, size=(20, 20))
    label = np.random.randint(2, size=20)
    train_pool = Pool(data, label, cat_features=[1, 2])

    params = {
        'logging_level': 'Silent',
        'loss_function': 'Logloss',
        'iterations': 10,
        'random_seed': 20,
    }

    model = CatBoost(params)
    with pytest.raises(CatboostError):
        model.metadata_

    with pytest.raises(CatboostError):
        model.is_fitted_

    model.fit(train_pool)

    with pytest.raises(CatboostError):
        model.metadata_

    with pytest.raises(CatboostError):
        model.is_fitted_


def test_no_yatest_common():
    assert "yatest" not in globals()
