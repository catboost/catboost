from collections import OrderedDict, Counter
import filecmp
import hashlib
import math
import numpy as np
import pprint
import pytest
import random
import operator
import re
import subprocess
import sys
import tempfile
import json
from catboost import (
    CatBoost,
    CatBoostClassifier,
    CatBoostRegressor,
    CatBoostRanker,
    CatBoostError,
    EFeaturesSelectionAlgorithm,
    EFeaturesSelectionGrouping,
    EFstrType,
    FeaturesData,
    Pool,
    cv,
    metrics,
    sum_models,
    train,
    _have_equal_features,
    to_regressor,
    to_classifier,
    to_ranker,
    MultiTargetCustomMetric,
    MultiTargetCustomObjective,)
from catboost.core import is_maximizable_metric, is_minimizable_metric
from catboost.eval.catboost_evaluation import CatboostEvaluation, EvalType
from catboost.utils import eval_metric, create_cd, read_cd, get_roc_curve, select_threshold, quantize
from catboost.utils import DataMetaInfo, TargetStats, compute_training_options
import os.path
import os
import pandas as pd
from pandas import read_csv, DataFrame, Series, Categorical
from pandas.arrays import SparseArray
import scipy.sparse
import scipy.special

from catboost_pytest_lib import (
    DelayedTee,
    binary_path,
    data_file,
    get_limited_precision_dsv_diff_tool,
    get_limited_precision_json_diff_tool,
    get_limited_precision_numpy_diff_tool,
    local_canonical_file,
    permute_dataset_columns,
    remove_time_from_json,
    test_output_path,
    generate_concatenated_random_labeled_dataset,
    generate_random_labeled_dataset,
    generate_dataset_with_num_and_cat_features,
    load_dataset_as_dataframe,
    load_pool_features_as_df
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
NONSYMMETRIC = ['Lossguide', 'Depthwise']

TRAIN_FILE = data_file('adult', 'train_small')
TEST_FILE = data_file('adult', 'test_small')
CD_FILE = data_file('adult', 'train.cd')

NAN_TRAIN_FILE = data_file('adult_nan', 'train_small')
NAN_TEST_FILE = data_file('adult_nan', 'test_small')
NAN_CD_FILE = data_file('adult_nan', 'train.cd')

FLOAT_WITH_NAN_FILE = data_file('float_with_nan.tsv')

CLOUDNESS_TRAIN_FILE = data_file('cloudness_small', 'train_small')
CLOUDNESS_TEST_FILE = data_file('cloudness_small', 'test_small')
CLOUDNESS_CD_FILE = data_file('cloudness_small', 'train.cd')
CLOUDNESS_ONLY_NUM_CD_FILE = data_file('cloudness_small', 'train_float.cd')

MULTIREGRESSION_TRAIN_FILE = data_file('multiregression', 'train')
MULTIREGRESSION_TEST_FILE = data_file('multiregression', 'test')
MULTIREGRESSION_CD_FILE = data_file('multiregression', 'train.cd')

SCENE_TRAIN_FILE = data_file('scene', 'train')
SCENE_TEST_FILE = data_file('scene', 'test')
SCENE_CD_FILE = data_file('scene', 'train.cd')

QUERYWISE_TRAIN_FILE = data_file('querywise', 'train')
QUERYWISE_TEST_FILE = data_file('querywise', 'test')
QUERYWISE_CD_FILE = data_file('querywise', 'train.cd')
QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT = data_file('querywise', 'train.cd.group_weight')
QUERYWISE_CD_FILE_WITH_GROUP_ID = data_file('querywise', 'train.cd.query_id')
QUERYWISE_CD_FILE_WITH_SUBGROUP_ID = data_file('querywise', 'train.cd.subgroup_id')
QUERYWISE_TRAIN_PAIRS_FILE = data_file('querywise', 'train.pairs')
QUERYWISE_TRAIN_PAIRS_FILE_WITH_PAIR_WEIGHT = data_file('querywise', 'train.pairs.weighted')
QUERYWISE_TEST_PAIRS_FILE = data_file('querywise', 'test.pairs')
QUERYWISE_FEATURE_NAMES_FILE = data_file('querywise', 'train.feature_names')
QUERYWISE_QUANTIZATION_BORDERS_EXAMPLE = data_file('querywise', 'train.quantization_borders_example')

QUANTIZED_TRAIN_FILE = data_file('quantized_adult', 'train.qbin')
QUANTIZED_TEST_FILE = data_file('quantized_adult', 'test.qbin')
QUANTIZED_CD_FILE = data_file('quantized_adult', 'pool.cd')

AIRLINES_5K_TRAIN_FILE = data_file('airlines_5K', 'train')
AIRLINES_5K_TEST_FILE = data_file('airlines_5K', 'test')
AIRLINES_5K_CD_FILE = data_file('airlines_5K', 'cd')

SMALL_CATEGORIAL_FILE = data_file('small_categorial', 'train')
SMALL_CATEGORIAL_CD_FILE = data_file('small_categorial', 'train.cd')

BLACK_FRIDAY_TRAIN_FILE = data_file('black_friday', 'train')
BLACK_FRIDAY_TEST_FILE = data_file('black_friday', 'test')
BLACK_FRIDAY_CD_FILE = data_file('black_friday', 'cd')

HIGGS_TRAIN_FILE = data_file('higgs', 'train_small')
HIGGS_5K_TRAIN_FILE = data_file('higgs', 'train_5K')
HIGGS_TEST_FILE = data_file('higgs', 'test_small')
HIGGS_CD_FILE = data_file('higgs', 'train.cd')

ROTTEN_TOMATOES_TRAIN_FILE = data_file('rotten_tomatoes', 'train')
ROTTEN_TOMATOES_TRAIN_SMALL_NO_QUOTES_FILE = data_file('rotten_tomatoes', 'train_small_no_quotes')
ROTTEN_TOMATOES_TEST_FILE = data_file('rotten_tomatoes', 'test')
ROTTEN_TOMATOES_CD_FILE = data_file('rotten_tomatoes', 'cd')
ROTTEN_TOMATOES_CD_BINCLASS_FILE = data_file('rotten_tomatoes', 'cd_binclass')

ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE = data_file('rotten_tomatoes_small_with_embeddings', 'train')
ROTTEN_TOMATOES_WITH_EMBEDDINGS_CD_BINCLASS_FILE = data_file(
    'rotten_tomatoes_small_with_embeddings',
    'cd_binclass'
)

AIRLINES_ONEHOT_TRAIN_FILE = data_file('airlines_onehot_250', 'train_small')
AIRLINES_ONEHOT_TEST_FILE = data_file('airlines_onehot_250', 'test_small')
AIRLINES_ONEHOT_CD_FILE = data_file('airlines_onehot_250', 'train.cd')

CONVERT_LIGHT_GBM_PREDICTIONS = data_file('convertions_models', 'predict')
CONVERT_RANDOM_GENERATED_TEST = data_file('convertions_models', 'test')
CONVERT_MODEL_ONNX = data_file('convertions_models', 'model_gbm.onnx')

OUTPUT_MODEL_PATH = 'model.bin'
OUTPUT_COREML_MODEL_PATH = 'model.mlmodel'
OUTPUT_CPP_MODEL_PATH = 'model.cpp'
OUTPUT_PYTHON_MODEL_PATH = 'model.py'
OUTPUT_JSON_MODEL_PATH = 'model.json'
OUTPUT_ONNX_MODEL_PATH = 'model.onnx'
OUTPUT_PMML_MODEL_PATH = 'model.pmml'
PREDS_PATH = 'predictions.npy'
PREDS_TXT_PATH = 'predictions.txt'
CV_CSV_PATH = 'result.csv'
FIMP_NPY_PATH = 'feature_importance.npy'
FIMP_TXT_PATH = 'feature_importance.txt'
OIMP_PATH = 'object_importances.txt'
OUTPUT_QUANTIZED_POOL_PATH = 'quantized_pool.bin'
TARGET_IDX = 1
CAT_FEATURES = [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 16]
CAT_COLUMNS = [0, 2, 3, 5, 7, 9, 10, 11, 12, 13, 17]

JSON_LOG_PATH = 'catboost_info/catboost_training.json'


def JSON_LOG_CV_PATH(foldIdx):
    return 'catboost_info/fold-' + str(foldIdx) + '/catboost_training.json'


numpy_num_data_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64
]

sparse_matrix_types = [
    scipy.sparse.csr_matrix,
    scipy.sparse.bsr_matrix,
    scipy.sparse.coo_matrix,
    scipy.sparse.csc_matrix,
    scipy.sparse.dok_matrix,
    scipy.sparse.lil_matrix
]

label_types = ['consecutive_integers', 'nonconsecutive_integers', 'string', 'float']


model_diff_tool = binary_path("catboost/tools/model_comparator/model_comparator")

np.set_printoptions(legacy='1.13')


def compare_canonical_models(model, diff_limit=0):
    return local_canonical_file(model, diff_tool=[model_diff_tool, '--diff-limit', str(diff_limit)])


def _check_shape(pool, object_count, features_count):
    return pool.shape == (object_count, features_count)


def _check_data(data1, data2):
    return np.all(np.isclose(data1, data2, rtol=0.001, equal_nan=True))


def _count_lines(afile):
    with open(afile, 'r') as f:
        num_lines = sum(1 for line in f)
    return num_lines


def _generate_nontrivial_binary_target(num, seed=20181219, prng=None):
    '''
    Generate binary vector with non zero variance
    :param num:
    :return:
    '''
    if prng is None:
        prng = np.random.RandomState(seed=seed)

    def gen():
        return prng.randint(0, 2, size=num)
    if num <= 1:
        return gen()

    y = gen()  # 0/1 labels
    while y.min() == y.max():
        y = gen()
    return y


def _generate_random_target(num, seed=20181219, prng=None):
    if prng is None:
        prng = np.random.RandomState(seed=seed)
    return prng.random_sample((num,))


def set_random_weight(pool, seed=20181219, prng=None):
    if prng is None:
        prng = np.random.RandomState(seed=seed)
    pool.set_weight(prng.random_sample(pool.num_row()))
    if pool.num_pairs() > 0:
        pool.set_pairs_weight(prng.random_sample(pool.num_pairs()))


def verify_finite(result):
    inf = float('inf')
    for r in result:
        assert (r == r)
        assert (abs(r) < inf)


def append_param(metric_name, param):
    return metric_name + (':' if ':' not in metric_name else ';') + param


# returns (features_data, labels)
def load_simple_dataset_as_lists(is_test):
    features_data = []
    labels = []
    with open(TEST_FILE if is_test else TRAIN_FILE) as data_file:
        for line in data_file:
            elements = line[:-1].split('\t')
            features_data.append([])
            for column_idx, element in enumerate(elements):
                if column_idx == TARGET_IDX:
                    labels.append(element)
                else:
                    features_data[-1].append(element if column_idx in CAT_COLUMNS else float(element))

    return features_data, labels


# Test cases begin here ########################################################

@pytest.mark.parametrize('niter', [100, 500])
def test_multiregression_custom_eval(niter, n=10):
    class MultiRMSE(MultiTargetCustomMetric):
        def get_final_error(self, error, weight):
            if (weight == 0):
                return 0
            else:
                return (error / weight) ** 0.5

        def is_max_optimal(self):
            return False

        def evaluate(self, approxes, target, weight):
            assert len(target) == len(approxes)
            assert len(target[0]) == len(approxes[0])

            error_sum = 0.0
            weight_sum = 0.0

            for dim in range(len(target)):
                for i in range(len(target[0])):
                    w = 1.0 if weight is None else weight[i]
                    error_sum += w * (approxes[dim][i] - target[dim][i]) ** 2

            for i in range(len(target[0])):
                weight_sum += 1.0 if weight is None else weight[i]

            return error_sum, weight_sum

    xs = np.arange(n).reshape((-1, 1)).astype(np.float32)
    ys = np.hstack([
        (xs > 0.5 * n),
        (xs < 0.5 * n)
    ]).astype(np.float32)

    train_pool = Pool(data=xs,
                      label=ys)

    test_pool = Pool(data=xs,
                     label=ys)

    model1 = CatBoostRegressor(loss_function='MultiRMSE', iterations=niter, use_best_model=True, eval_metric=MultiRMSE())
    model1.fit(train_pool, eval_set=test_pool)
    pred1 = model1.predict(test_pool)

    model2 = CatBoostRegressor(loss_function='MultiRMSE', iterations=niter, use_best_model=True, eval_metric="MultiRMSE")
    model2.fit(train_pool, eval_set=test_pool)
    pred2 = model2.predict(test_pool)

    assert np.allclose(pred1, pred2)
    assert np.allclose(model1.evals_result_['validation']['MultiRMSE'], model2.evals_result_['validation']['MultiRMSE'])


@pytest.mark.parametrize('niter', [100, 500])
def test_multiregression(niter, n=10):
    xs = np.arange(n).reshape((-1, 1)).astype(np.float32)
    ys = np.hstack([
        (xs > 0.5 * n),
        (xs < 0.5 * n)
    ]).astype(np.float32)

    model = CatBoostRegressor(loss_function='MultiRMSE', iterations=niter)
    model.fit(xs, ys)
    ys_pred = model.predict(xs)
    model.score(xs, ys)

    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(ys_pred), fmt='%.8f')

    assert ys_pred.shape == ys.shape
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('niter', [1, 100, 500])
def test_save_model_multiregression(niter):
    train_file = MULTIREGRESSION_TRAIN_FILE
    cd_file = MULTIREGRESSION_CD_FILE
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)

    train_pool = Pool(train_file, column_description=cd_file)
    test_pool = Pool(train_file, column_description=cd_file)

    model = CatBoost(dict(loss_function='MultiRMSE', iterations=niter))
    model.fit(train_pool)
    model.save_model(output_model_path)

    model2 = CatBoost()
    model2.load_model(output_model_path)

    pred1 = model.predict(test_pool)
    pred2 = model2.predict(test_pool)
    assert _check_data(pred1, pred2)


def test_load_file():
    assert _check_shape(Pool(TRAIN_FILE, column_description=CD_FILE), 101, 17)


def test_load_list():
    features_data, labels = load_simple_dataset_as_lists(is_test=False)
    assert _check_shape(Pool(features_data, labels, CAT_FEATURES), 101, 17)


datasets_for_test_ndarray = [
    'adult',
    'cloudness_small',
    'higgs',
    'rotten_tomatoes',
    'rotten_tomatoes_small_with_embeddings'
]


def get_only_features_names(columns_metadata):
    column_names = columns_metadata['column_names']
    non_feature_column_indices = set(columns_metadata['non_feature_column_indices'])
    feature_names = []
    for i, column_name in enumerate(column_names):
        if i not in non_feature_column_indices:
            feature_names.append(column_name)
    return feature_names


@pytest.mark.parametrize(
    'dataset',
    datasets_for_test_ndarray,
    ids=['dataset=%s' % dataset for dataset in datasets_for_test_ndarray]
)
@pytest.mark.parametrize('order', ['C', 'F'], ids=['order=C', 'order=F'])
def test_load_ndarray_vs_load_from_file(dataset, order):
    n_objects = 101
    if dataset == 'adult':  # mixed numeric and categorical features data, cat data is strings
        train_file = TRAIN_FILE
        cd_file = CD_FILE
        dtypes = [object]
        float_dtype_is_ok = False
    elif dataset == 'cloudness_small':  # mixed numeric and categorical features data, cat data is integers
        train_file = CLOUDNESS_TRAIN_FILE
        cd_file = CLOUDNESS_CD_FILE
        dtypes = [np.float32, np.float64, object]
        float_dtype_is_ok = False
    elif dataset == 'higgs':  # mixed numeric and categorical features data, cat data is strings
        train_file = HIGGS_TRAIN_FILE
        cd_file = HIGGS_CD_FILE
        dtypes = [np.float32, np.float64, object]
        float_dtype_is_ok = True
    elif dataset == 'rotten_tomatoes':  # mixed numeric, categorical and text features data
        train_file = ROTTEN_TOMATOES_TRAIN_SMALL_NO_QUOTES_FILE
        cd_file = ROTTEN_TOMATOES_CD_BINCLASS_FILE
        dtypes = [object]
        float_dtype_is_ok = False
    elif dataset == 'rotten_tomatoes_small_with_embeddings':
        # mixed numeric, categorical and embedding features data
        train_file = ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE
        cd_file = ROTTEN_TOMATOES_WITH_EMBEDDINGS_CD_BINCLASS_FILE
        dtypes = [object]
        float_dtype_is_ok = False

    columns_metadata = read_cd(cd_file, data_file=train_file, canonize_column_types=True)
    target_column_idx = columns_metadata['column_type_to_indices']['Label'][0]
    cat_column_indices = columns_metadata['column_type_to_indices'].get('Categ', [])
    text_column_indices = columns_metadata['column_type_to_indices'].get('Text', [])
    num_vector_column_indices = columns_metadata['column_type_to_indices'].get('NumVector', [])
    n_features = (
        + len(cat_column_indices)
        + len(text_column_indices)
        + len(num_vector_column_indices)
        + len(columns_metadata['column_type_to_indices'].get('Num', []))
    )
    feature_names = get_only_features_names(columns_metadata)

    pool_from_file = Pool(train_file, column_description=cd_file, ignore_csv_quoting=True)
    pool_from_file.set_feature_names(feature_names)

    for dtype in dtypes:
        features_data = np.empty((n_objects, n_features), dtype=dtype, order=order)
        labels = np.empty(n_objects, dtype=float)
        with open(train_file) as train_input:
            for line_idx, l in enumerate(train_input.readlines()):
                elements = l[:-1].split('\t')
                feature_idx = 0
                for column_idx, element in enumerate(elements):
                    if column_idx == target_column_idx:
                        labels[line_idx] = float(element)
                    else:
                        if feature_idx in num_vector_column_indices:
                            features_data[line_idx, feature_idx] = (
                                np.array([float(e) for e in element.split(';')])
                            )
                        else:
                            features_data[line_idx, feature_idx] = (
                                element
                                if (dtype is object) or (column_idx in cat_column_indices) or (column_idx in text_column_indices)
                                else dtype(element)
                            )
                        feature_idx += 1

        if (dtype in [np.float32, np.float64]) and (not float_dtype_is_ok):
            with pytest.raises(CatBoostError):
                pool_from_ndarray = Pool(features_data, labels,
                                         cat_features=columns_metadata['cat_feature_indices'],
                                         text_features=columns_metadata['text_feature_indices'],
                                         embedding_features=columns_metadata['embedding_feature_indices'],
                                         feature_names=feature_names)
        else:
            pool_from_ndarray = Pool(features_data, labels,
                                     cat_features=columns_metadata['cat_feature_indices'],
                                     text_features=columns_metadata['text_feature_indices'],
                                     embedding_features=columns_metadata['embedding_feature_indices'],
                                     feature_names=feature_names)

            assert _have_equal_features(pool_from_file, pool_from_ndarray)
            assert _check_data([float(label) for label in pool_from_file.get_label()], pool_from_ndarray.get_label())


@pytest.mark.parametrize('order', ['C', 'F'], ids=['order=C', 'order=F'])
def test_load_ndarray_vs_load_from_file_multitarget(order):
    train_file = MULTIREGRESSION_TRAIN_FILE
    cd_file = MULTIREGRESSION_CD_FILE
    dtypes = [np.float32, np.float64]

    n_objects = np.loadtxt(train_file, delimiter='\t').shape[0]
    columns_metadata = read_cd(cd_file, data_file=train_file, canonize_column_types=True)
    target_indices = columns_metadata['column_type_to_indices'].get('Label', [])
    feature_indices = columns_metadata['column_type_to_indices'].get('Num', [])

    pool_from_file = Pool(train_file, column_description=cd_file)

    for dtype in dtypes:
        features_data = np.empty((n_objects, len(feature_indices)), dtype=dtype, order=order)
        labels = np.empty((n_objects, len(target_indices)), dtype=dtype, order=order)
        with open(train_file) as train_input:
            for line_idx, l in enumerate(train_input.readlines()):
                elements = l[:-1].split('\t')
                feature_idx = 0
                target_idx = 0
                for column_idx, element in enumerate(elements):
                    if column_idx in target_indices:
                        labels[line_idx, target_idx] = float(element)
                        target_idx += 1
                    else:
                        features_data[line_idx, feature_idx] = dtype(element)
                        feature_idx += 1

        pool_from_ndarray = Pool(features_data, labels)
        assert _have_equal_features(pool_from_file, pool_from_ndarray)
        assert np.allclose(
            [[float(elem) for elem in row] for row in pool_from_file.get_label()],
            pool_from_ndarray.get_label(),
            atol=0
        )


@pytest.mark.parametrize(
    'features_dtype',
    numpy_num_data_types,
    ids=['features_dtype=%s' % np.dtype(dtype).name for dtype in numpy_num_data_types]
)
def test_fit_on_ndarray(features_dtype):
    if np.dtype(features_dtype).kind == 'f':
        cat_features = []
        lower_bound = -1.0
        upper_bound = 1.0
    else:
        cat_features = [0, 7, 11]
        lower_bound = max(np.iinfo(features_dtype).min, -32767)
        upper_bound = min(np.iinfo(features_dtype).max, 32767)

    n_features = 20
    order_to_pool = {}
    for order in ('C', 'F'):
        features, labels = generate_random_labeled_dataset(
            n_samples=100,
            n_features=n_features,
            labels=[0, 1],
            features_dtype=features_dtype,
            features_range=(lower_bound, upper_bound),
            features_order=order
        )
        order_to_pool[order] = Pool(features, label=labels, cat_features=cat_features)

    assert _have_equal_features(order_to_pool['C'], order_to_pool['F'])

    model = CatBoostClassifier(iterations=5)
    model.fit(order_to_pool['F'])  # order is irrelevant here - they are equal

    assert model.n_features_in_ == n_features

    preds = model.predict(order_to_pool['F'])

    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(preds))
    return local_canonical_file(preds_path)


@pytest.mark.parametrize(
    'dataset',
    ['adult', 'adult_nan', 'querywise', 'rotten_tomatoes', 'rotten_tomatoes_small_with_embeddings']
)
def test_load_df_vs_load_from_file(dataset):
    train_file, cd_file, target_idx, group_id_idx, other_non_feature_columns = {
        'adult': (TRAIN_FILE, CD_FILE, TARGET_IDX, None, []),
        'adult_nan': (NAN_TRAIN_FILE, NAN_CD_FILE, TARGET_IDX, None, []),
        'querywise': (QUERYWISE_TRAIN_FILE, QUERYWISE_CD_FILE, 2, 1, [0, 3, 4]),
        'rotten_tomatoes': (ROTTEN_TOMATOES_TRAIN_FILE, ROTTEN_TOMATOES_CD_BINCLASS_FILE, 11, None, []),
        'rotten_tomatoes_small_with_embeddings': (
            ROTTEN_TOMATOES_WITH_EMBEDDINGS_TRAIN_FILE,
            ROTTEN_TOMATOES_WITH_EMBEDDINGS_CD_BINCLASS_FILE,
            14,
            None,
            []
        )
    }[dataset]

    pool1 = Pool(train_file, column_description=cd_file)
    cat_features = pool1.get_cat_feature_indices()
    text_features = pool1.get_text_feature_indices()
    embedding_features = pool1.get_embedding_feature_indices()

    data = read_csv(train_file, header=None, delimiter='\t', na_filter=False)

    labels = data.iloc[:, target_idx]
    group_ids = None
    if group_id_idx:
        group_ids = [int(group_id) for group_id in data.iloc[:, group_id_idx]]

    data.drop(
        [target_idx] + ([group_id_idx] if group_id_idx else []) + other_non_feature_columns,
        axis=1,
        inplace=True
    )

    pool1.set_feature_names(list(data.columns))

    def check(pool1, data2):
        pool2 = Pool(
            data2,
            labels,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
            group_id=group_ids
        )
        assert _have_equal_features(pool1, pool2)
        assert _check_data([float(label) for label in pool1.get_label()], pool2.get_label())

    if len(embedding_features):
        for embedding_array_ctor in [np.array, list]:
            for embedding_elements_dtype in [np.float32, float]:
                data2 = data.copy()
                for embedding_feature in embedding_features:
                    data2[embedding_feature] = [
                        embedding_array_ctor(
                            [embedding_elements_dtype(element) for element in object_embedding_str.split(';')]
                        )
                        for object_embedding_str in data[embedding_feature]
                    ]
                check(pool1, data2)
    else:
        check(pool1, data)


def test_load_df_vs_load_from_file_multitarget():
    train_file = MULTIREGRESSION_TRAIN_FILE
    cd_file = MULTIREGRESSION_CD_FILE
    target_idx = [0, 1]

    pool1 = Pool(train_file, column_description=cd_file)
    data = read_csv(train_file, header=None, delimiter='\t')

    labels = data.iloc[:, target_idx]

    data.drop(
        target_idx,
        axis=1,
        inplace=True
    )

    cat_features = pool1.get_cat_feature_indices()

    pool1.set_feature_names(list(data.columns))

    pool2 = Pool(data, labels, cat_features, group_id=None)
    assert _have_equal_features(pool1, pool2)
    assert _check_data([[float(label) for label in sublabel] for sublabel in pool1.get_label()], pool2.get_label())


def test_load_series():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    data = read_csv(TRAIN_FILE, header=None, delimiter='\t')
    labels = Series(data.iloc[:, TARGET_IDX])
    data.drop([TARGET_IDX], axis=1, inplace=True)
    data = Series(list(data.values))
    cat_features = pool.get_cat_feature_indices()
    pool2 = Pool(data, labels, cat_features)
    assert _have_equal_features(pool, pool2)
    assert _check_data([int(label) for label in pool.get_label()], pool2.get_label())


def test_pool_cat_features():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    assert np.all(pool.get_cat_feature_indices() == CAT_FEATURES)


def test_pool_cat_features_as_strings():
    df = DataFrame(data=[[1, 2], [3, 4]], columns=['col1', 'col2'])
    pool = Pool(df, cat_features=['col2'])
    assert np.all(pool.get_cat_feature_indices() == [1])

    data = [[1, 2, 3], [4, 5, 6]]
    pool = Pool(data, feature_names=['col1', 'col2', 'col3'], cat_features=['col2', 'col3'])
    assert np.all(pool.get_cat_feature_indices() == [1, 2])

    data = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(CatBoostError):
        Pool(data, cat_features=['col2', 'col3'])


def test_load_generated():
    pool_size = (100, 10)
    prng = np.random.RandomState(seed=20181219)
    data = np.round(prng.normal(size=pool_size), decimals=3)
    label = _generate_nontrivial_binary_target(pool_size[0], prng=prng)
    pool = Pool(data, label)
    assert _check_data(pool.get_features(), data)
    assert _check_data(pool.get_label(), label)


def test_load_dumps():
    pool_size = (100, 10)
    prng = np.random.RandomState(seed=20181219)
    data = prng.randint(10, size=pool_size)
    labels = _generate_nontrivial_binary_target(pool_size[0], prng=prng)
    pool1 = Pool(data, labels)
    lines = []
    for i in range(len(data)):
        line = [str(labels[i])] + [str(x) for x in data[i]]
        lines.append('\t'.join(line))
    text = '\n'.join(lines)
    with open('test_data_dumps', 'w') as f:
        f.write(text)
    pool2 = Pool('test_data_dumps')
    assert _check_data(pool1.get_features(), pool2.get_features())
    assert _check_data(pool1.get_label(), [int(label) for label in pool2.get_label()])


@pytest.mark.parametrize(
    'features_type',
    ['numpy.ndarray', 'pandas.DataFrame'],
    ids=['features_type=numpy.ndarray', 'features_type=pandas.DataFrame']
)
def test_pool_from_slices(features_type):
    full_size = (100, 30)
    subset_size = (20, 17)

    prng = np.random.RandomState(seed=20191120)

    for start_offsets in ((0, 0), (5, 3)):
        full_features_data = np.round(prng.normal(size=full_size), decimals=3)
        full_label = _generate_nontrivial_binary_target(full_size[0], prng=prng)

        subset_features_data = full_features_data[start_offsets[0]:subset_size[0], start_offsets[1]:subset_size[1]]
        subset_label = full_label[start_offsets[0]:subset_size[0]]

        if features_type == 'numpy.ndarray':
            pool = Pool(subset_features_data, subset_label)
        else:
            pool = Pool(DataFrame(subset_features_data), subset_label)
        assert _check_data(pool.get_features(), subset_features_data)
        assert _check_data([float(value) for value in pool.get_label()], subset_label)


@pytest.mark.parametrize(
    'cat_features_specified',
    [False, True],
    ids=['cat_features_specified=False', 'cat_features_specified=True']
)
def test_dataframe_with_pandas_categorical_columns(cat_features_specified):
    df = DataFrame()
    df['num_feat_0'] = [0, 1, 0, 2, 3, 1, 2]
    df['num_feat_1'] = [0.12, 0.8, 0.33, 0.11, 0.0, 1.0, 0.0]
    df['cat_feat_2'] = Series(['A', 'B', 'A', 'C', 'A', 'A', 'A'], dtype='category')
    df['cat_feat_3'] = Series(['x', 'x', 'y', 'y', 'y', 'x', 'x'])
    df['cat_feat_4'] = Categorical(
        ['large', 'small', 'medium', 'large', 'small', 'small', 'medium'],
        categories=['small', 'medium', 'large'],
        ordered=True
    )
    df['cat_feat_5'] = [0, 1, 0, 2, 3, 1, 2]

    labels = [0, 1, 1, 0, 1, 0, 1]

    model = CatBoostClassifier(iterations=2)
    if cat_features_specified:
        model.fit(X=df, y=labels, cat_features=[2, 3, 4, 5])
        pred = model.predict(df)

        preds_path = test_output_path(PREDS_TXT_PATH)
        np.savetxt(preds_path, np.array(pred), fmt='%.8f')
        return local_canonical_file(preds_path)
    else:
        with pytest.raises(CatBoostError):
            model.fit(X=df, y=labels)


def test_equivalence_of_pools_from_pandas_dataframe_with_different_cat_features_column_types():
    df = DataFrame()
    df['num_feat_0'] = [0, 1, 0, 2, 3, 1, 2]
    df['num_feat_1'] = [0.12, 0.8, 0.33, 0.11, 0.0, 1.0, 0.0]
    df['cat_feat_2'] = ['A', 'B', 'A', 'C', 'A', 'A', 'A']
    df['cat_feat_3'] = ['x', 'x', 'y', 'y', 'y', 'x', 'x']
    df['cat_feat_4'] = ['large', 'small', 'medium', 'large', 'small', 'small', 'medium']
    df['cat_feat_5'] = [0, 1, 0, 2, 3, 1, 2]

    labels = [0, 1, 1, 0, 1, 0, 1]

    cat_features = ['cat_feat_%i' % i for i in range(2, 6)]

    pool_from_df = Pool(df, labels, cat_features=cat_features)

    for cat_features_dtype in ['object', 'category']:
        columns_for_new_df = OrderedDict()
        for column_name, column_data in df.iteritems():
            if column_name in cat_features:
                column_data = column_data.astype(cat_features_dtype)
            columns_for_new_df.setdefault(column_name, column_data)

        new_df = DataFrame(columns_for_new_df)

        pool_from_new_df = Pool(new_df, labels, cat_features=cat_features)

        assert _have_equal_features(pool_from_df, pool_from_new_df)


def test_pool_with_external_feature_names():
    for cd_has_feature_names in [False, True]:
        if cd_has_feature_names:
            cd_file = data_file('adult', 'train_with_id.cd')
        else:
            cd_file = data_file('adult', 'train.cd')
        train_pool = Pool(
            TRAIN_FILE,
            column_description=cd_file,
            feature_names=data_file('adult', 'feature_names')
        )
        model = CatBoostClassifier(iterations=2)
        model.fit(train_pool)
        if cd_has_feature_names:
            model_cd_with_feature_names = model
        else:
            model_cd_without_feature_names = model

    assert model_cd_with_feature_names == model_cd_without_feature_names
    assert model.feature_names_ == [
        'C0',
        'C1',
        'C2',
        'F0',
        'C3',
        'F1',
        'C4',
        'F2',
        'C5',
        'C6',
        'C7',
        'C8',
        'C9',
        'F3',
        'F4',
        'F5',
        'C10'
    ]


# feature_matrix is (doc_count x feature_count)
def get_features_data_from_matrix(feature_matrix, cat_feature_indices, order='C'):
    object_count = len(feature_matrix)
    feature_count = len(feature_matrix[0])
    cat_feature_count = len(cat_feature_indices)
    num_feature_count = feature_count - cat_feature_count

    result_num = np.empty((object_count, num_feature_count), dtype=np.float32, order=order)
    result_cat = np.empty((object_count, cat_feature_count), dtype=object, order=order)

    for object_idx in range(object_count):
        num_feature_idx = 0
        cat_feature_idx = 0
        for feature_idx in range(len(feature_matrix[object_idx])):
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


def get_features_data_from_file(data_file, drop_columns, cat_feature_indices, order='C'):
    data_matrix_from_file = read_csv(data_file, header=None, dtype=str, delimiter='\t')
    data_matrix_from_file.drop(drop_columns, axis=1, inplace=True)
    return get_features_data_from_matrix(np.array(data_matrix_from_file), cat_feature_indices, order)


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
    assert _have_equal_features(pool1, pool2)


@pytest.mark.parametrize('order', ['C', 'F'], ids=['order=C', 'order=F'])
def test_features_data_good(order):
    # 0 objects
    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(cat_feature_data=np.empty((0, 4), dtype=object, order=order)),
        np.empty((0, 4), dtype=object),
        cat_features_indices=[0, 1, 2, 3]
    )

    # 0 objects
    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.empty((0, 2), dtype=object, order=order),
            cat_feature_names=['cat0', 'cat1'],
            num_feature_data=np.empty((0, 3), dtype=np.float32, order=order),
        ),
        np.empty((0, 5), dtype=object),
        cat_features_indices=[3, 4],
        feature_names=['', '', '', 'cat0', 'cat1']
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object, order=order)
        ),
        [[b'amazon', b'bing'], [b'ebay', b'google']],
        cat_features_indices=[0, 1]
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32, order=order)
        ),
        [[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]],
        cat_features_indices=[]
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object, order=order),
            num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32, order=order)
        ),
        [[1.0, 2.0, 3.0, b'amazon', b'bing'], [22.0, 7.1, 10.2, b'ebay', b'google']],
        cat_features_indices=[3, 4]
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object, order=order),
            cat_feature_names=['shop', 'search']
        ),
        [[b'amazon', b'bing'], [b'ebay', b'google']],
        cat_features_indices=[0, 1],
        feature_names=['shop', 'search']
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32, order=order),
            num_feature_names=['weight', 'price', 'volume']
        ),
        [[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]],
        cat_features_indices=[],
        feature_names=['weight', 'price', 'volume']
    )

    compare_pools_from_features_data_and_generic_matrix(
        FeaturesData(
            cat_feature_data=np.array([[b'amazon', b'bing'], [b'ebay', b'google']], dtype=object, order=order),
            cat_feature_names=['shop', 'search'],
            num_feature_data=np.array([[1.0, 2.0, 3.0], [22.0, 7.1, 10.2]], dtype=np.float32, order=order),
            num_feature_names=['weight', 'price', 'volume']
        ),
        [[1.0, 2.0, 3.0, b'amazon', b'bing'], [22.0, 7.1, 10.2, b'ebay', b'google']],
        cat_features_indices=[3, 4],
        feature_names=['weight', 'price', 'volume', 'shop', 'search']
    )


def test_features_data_bad():
    # empty
    with pytest.raises(CatBoostError):
        FeaturesData()

    # names w/o data
    with pytest.raises(CatBoostError):
        FeaturesData(cat_feature_data=[[b'amazon', b'bing']], num_feature_names=['price'])

    # bad matrix type
    with pytest.raises(CatBoostError):
        FeaturesData(
            cat_feature_data=[[b'amazon', b'bing']],
            num_feature_data=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

    # bad matrix shape
    with pytest.raises(CatBoostError):
        FeaturesData(num_feature_data=np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32))

    # bad element type
    with pytest.raises(CatBoostError):
        FeaturesData(
            cat_feature_data=np.array([b'amazon', b'bing'], dtype=object),
            num_feature_data=np.array([1.0, 2.0, 3.0], dtype=np.float64)
        )

    # bad element type
    with pytest.raises(CatBoostError):
        FeaturesData(cat_feature_data=np.array(['amazon', 'bing']))

    # bad names type
    with pytest.raises(CatBoostError):
        FeaturesData(
            cat_feature_data=np.array([[b'google'], [b'reddit']], dtype=object),
            cat_feature_names=[None, 'news_aggregator']
        )

    # bad names length
    with pytest.raises(CatBoostError):
        FeaturesData(
            cat_feature_data=np.array([[b'google'], [b'bing']], dtype=object),
            cat_feature_names=['search_engine', 'news_aggregator']
        )

    # no features
    with pytest.raises(CatBoostError):
        FeaturesData(
            cat_feature_data=np.array([[], [], []], dtype=object),
            num_feature_data=np.array([[], [], []], dtype=np.float32)
        )

    # number of objects is different
    with pytest.raises(CatBoostError):
        FeaturesData(
            cat_feature_data=np.array([[b'google'], [b'bing']], dtype=object),
            num_feature_data=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )


def test_predict_regress(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 2, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    assert (model.is_fitted())
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_predict_sklearn_regress(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(train_pool)
    assert (model.is_fitted())
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_predict_sklearn_class(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, loss_function='Logloss', task_type=task_type, devices='0')
    model.fit(train_pool)
    assert (model.is_fitted())
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_predict_class_raw(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict(test_pool)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


def test_raw_predict_equals_to_model_predict(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    assert (model.is_fitted())
    pred = model.predict(test_pool, prediction_type='RawFormulaVal')
    assert np.all(np.isclose(model.get_test_eval(), pred, rtol=1.e-6))


@pytest.mark.parametrize('problem', ['Classifier', 'Regressor'])
def test_predict_and_predict_proba_on_single_object(problem):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    if problem == 'Classifier':
        model = CatBoostClassifier(iterations=2)
    else:
        model = CatBoostRegressor(iterations=2)

    model.fit(train_pool)

    test_data = read_csv(TEST_FILE, header=None, delimiter='\t')
    test_data.drop([TARGET_IDX], axis=1, inplace=True)

    pred = model.predict(test_data)
    if problem == 'Classifier':
        pred_probabilities = model.predict_proba(test_data)

    random.seed(0)
    for i in range(3):  # just some indices
        test_object_idx = random.randrange(test_data.shape[0])
        assert pred[test_object_idx] == model.predict(test_data.values[test_object_idx])

        if problem == 'Classifier':
            assert np.array_equal(pred_probabilities[test_object_idx], model.predict_proba(test_data.values[test_object_idx]))


def test_predict_on_gpu(task_type):
    if task_type == 'CPU':
        return
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=10, task_type='GPU', devices='0')
    model.fit(pool)
    cpu_prediction = np.array(model.predict(pool, task_type='CPU'))
    gpu_prediction = np.array(model.predict(pool, task_type='GPU'))
    assert np.abs(cpu_prediction - gpu_prediction).max() < 1e-9


def test_model_pickling(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    pred = model.predict(test_pool, prediction_type='RawFormulaVal')
    model_unpickled = pickle.loads(pickle.dumps(model))
    pred_new = model_unpickled.predict(test_pool, prediction_type='RawFormulaVal')
    assert all(pred_new == pred)


def test_save_load_equality(task_type):
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)

    def check_load_from_stream(model):
        cb_stream = CatBoost()
        with open(output_model_path, 'rb') as stream:
            cb_stream.load_model(stream=stream)
        assert model == cb_stream

    def check_load_from_string(model):
        cb_blob = CatBoost()
        cb_blob.load_model(blob=open(output_model_path, 'rb').read())
        assert model == cb_blob

    def fill_check_model(params, train_file, test_file, cd_file):
        model, _ = fit_from_file(params, train_file, test_file, cd_file)
        model.save_model(fname=output_model_path)
        check_load_from_string(model)
        check_load_from_stream(model)

    fill_check_model({'iterations': 10, 'task_type': task_type, 'devices': '0'}, TRAIN_FILE, TEST_FILE, CD_FILE)
    fill_check_model({'loss_function': 'RMSE', 'iterations': 10}, HIGGS_TRAIN_FILE, HIGGS_TEST_FILE, HIGGS_CD_FILE)

    params = {
        'dictionaries': [
            {'dictionary_id': 'UniGram', 'token_level_type': 'Letter', 'occurrence_lower_bound': '1'},
            {'dictionary_id': 'BiGram', 'token_level_type': 'Letter', 'occurrence_lower_bound': '1', 'gram_order': '2'},
            {'dictionary_id': 'Word', 'occurrence_lower_bound': '1'},
        ],
        'feature_calcers': ['NaiveBayes', 'BoW:top_tokens_count=10'],
        'iterations': 10,
        'loss_function': 'MultiClass',
        'task_type': task_type,
        'devices': '0'
    }
    fill_check_model(params, ROTTEN_TOMATOES_TRAIN_FILE, ROTTEN_TOMATOES_TEST_FILE, ROTTEN_TOMATOES_CD_FILE)


def test_load_model_incorrect_argument(task_type):
    with pytest.raises(CatBoostError):
        cb = CatBoost()
        cb.load_model()

    with pytest.raises(AssertionError):
        cb = CatBoost()
        cb.load_model(blob=42)

    with pytest.raises(AssertionError):
        cb = CatBoost()
        cb.load_model(blob=u"✓ means check")

    with pytest.raises(CatBoostError):
        cb = CatBoost()
        cb.load_model(stream=24)


def test_fit_from_file(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 2, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    predictions1 = model.predict(train_pool)

    model.fit(TRAIN_FILE, column_description=CD_FILE)
    predictions2 = model.predict(train_pool)
    assert all(predictions1 == predictions2)
    assert 'train_finish_time' in model.get_metadata()


def test_fit_from_empty_features_data(task_type):
    model = CatBoost({'iterations': 2, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    with pytest.raises(CatBoostError):
        model.fit(
            X=FeaturesData(num_feature_data=np.empty((0, 2), dtype=np.float32)),
            y=np.empty((0), dtype=np.int32)
        )


def fit_from_df(params, learn_file, test_file, cd_file):
    learn_df = read_csv(learn_file, header=None, sep='\t', na_filter=False)
    test_df = read_csv(test_file, header=None, sep='\t', na_filter=False)
    columns_metadata = read_cd(cd_file, data_file=learn_file)

    target_column_idx = columns_metadata['column_type_to_indices']['Label'][0]
    cat_feature_indices = columns_metadata['cat_feature_indices']
    text_feature_indices = columns_metadata['text_feature_indices']

    def get_split_on_features_and_label(df, label_idx):
        y = df[label_idx]
        X = df.drop(label_idx, axis=1)
        return X, y

    X_train, y_train = get_split_on_features_and_label(learn_df, target_column_idx)
    X_test, _ = get_split_on_features_and_label(test_df, target_column_idx)

    model = CatBoost(params)
    model.fit(X_train, y_train, cat_features=cat_feature_indices, text_features=text_feature_indices)

    return model.predict(X_test)


def fit_from_file(params, learn_file, test_file, cd_file):
    learn_pool = Pool(learn_file, column_description=cd_file)
    test_pool = Pool(test_file, column_description=cd_file)

    model = CatBoost(params)
    model.fit(learn_pool)

    return model, model.predict(test_pool)


@pytest.mark.parametrize('problem_type', ['binclass', 'multiclass', 'regression'])
def test_fit_with_texts(task_type, problem_type):
    params = {
        'dictionaries': [
            {'dictionary_id': 'UniGram', 'token_level_type': 'Letter', 'occurrence_lower_bound': '1'},
            {'dictionary_id': 'BiGram', 'token_level_type': 'Letter', 'occurrence_lower_bound': '1', 'gram_order': '2'},
            {'dictionary_id': 'Word', 'occurrence_lower_bound': '1'},
        ],
        'feature_calcers': ['BoW:top_tokens_count=10'] if problem_type == 'regression' else ['NaiveBayes', 'BoW:top_tokens_count=10'],
        'iterations': 100,
        'loss_function': {
            'binclass': 'Logloss',
            'multiclass': 'MultiClass',
            'regression': 'RMSE'
        }[problem_type],
        'task_type': task_type,
        'devices': '0'
    }

    learn = ROTTEN_TOMATOES_TRAIN_FILE
    test = ROTTEN_TOMATOES_TEST_FILE
    cd = ROTTEN_TOMATOES_CD_FILE if problem_type == 'multiclass' else ROTTEN_TOMATOES_CD_BINCLASS_FILE

    preds1 = fit_from_df(params, learn, test, cd)
    _, preds2 = fit_from_file(params, learn, test, cd)

    assert np.all(preds1 == preds2)


def test_coreml_import_export(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost(params={'loss_function': 'RMSE', 'iterations': 20, 'thread_count': 8, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    output_coreml_model_path = test_output_path(OUTPUT_COREML_MODEL_PATH)
    model.save_model(output_coreml_model_path, format="coreml")
    canon_pred = model.predict(test_pool)
    coreml_loaded_model = CatBoostRegressor()
    coreml_loaded_model.load_model(output_coreml_model_path, format="coreml")
    assert all(canon_pred == coreml_loaded_model.predict(test_pool))
    return compare_canonical_models(output_coreml_model_path)


def test_coreml_import_export_one_hot_features(task_type):
    train_pool = Pool(SMALL_CATEGORIAL_FILE, column_description=SMALL_CATEGORIAL_CD_FILE)
    model = CatBoost(params={'loss_function': 'RMSE', 'iterations': 2, 'task_type': task_type, 'devices': '0', 'one_hot_max_size': 4})
    model.fit(train_pool)
    output_coreml_model_path = test_output_path(OUTPUT_COREML_MODEL_PATH)
    model.save_model(output_coreml_model_path, format="coreml", pool=train_pool)
    pred = model.predict(train_pool)
    coreml_loaded_model = CatBoostRegressor()
    coreml_loaded_model.load_model(output_coreml_model_path, format="coreml")
    assert all(pred == coreml_loaded_model.predict(train_pool))
    return compare_canonical_models(output_coreml_model_path)


@pytest.mark.parametrize('pool,parameters', [('adult', {}), ('adult', {'one_hot_max_size': 100}), ('higgs', {})])
def test_convert_model_to_json(task_type, pool, parameters):
    train_pool = Pool(data_file(pool, 'train_small'), column_description=data_file(pool, 'train.cd'))
    test_pool = Pool(data_file(pool, 'test_small'), column_description=data_file(pool, 'train.cd'))
    converted_model_path = test_output_path("converted_model.bin")
    parameters.update({'iterations': 20, 'task_type': task_type, 'devices': '0'})
    model = CatBoost(parameters)
    model.fit(train_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    output_json_model_path = test_output_path(OUTPUT_JSON_MODEL_PATH)
    model.save_model(output_model_path)
    model.save_model(output_json_model_path, format="json")
    model2 = CatBoost()
    model2.load_model(output_json_model_path, format="json")
    model2.save_model(converted_model_path)
    pred1 = model.predict(test_pool)
    pred2 = model2.predict(test_pool)
    assert _check_data(pred1, pred2)
    subprocess.check_call((
        model_diff_tool, output_model_path, converted_model_path,
        '--diff-limit', '0.000001',
        '--ignore-keys', '.*TargetBorderClassifierIdx',
    ))
    return compare_canonical_models(converted_model_path)


def test_coreml_cbm_import_export(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost(params={'loss_function': 'RMSE', 'iterations': 20, 'thread_count': 8, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    canon_pred = model.predict(test_pool)
    output_coreml_model_path = test_output_path(OUTPUT_COREML_MODEL_PATH)
    model.save_model(output_coreml_model_path, format="coreml")

    coreml_loaded_model = CatBoost()
    coreml_loaded_model.load_model(output_coreml_model_path, format="coreml")
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    coreml_loaded_model.save_model(output_model_path)

    cbm_loaded_model = CatBoost()
    cbm_loaded_model.load_model(output_model_path)
    assert all(canon_pred == cbm_loaded_model.predict(test_pool))
    return compare_canonical_models(output_coreml_model_path)


def test_cpp_export_no_cat_features(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost({'iterations': 2, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    output_cpp_model_path = test_output_path(OUTPUT_CPP_MODEL_PATH)
    model.save_model(output_cpp_model_path, format="cpp")
    return local_canonical_file(output_cpp_model_path)


def test_cpp_export_with_cat_features(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 20, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    output_cpp_model_path = test_output_path(OUTPUT_CPP_MODEL_PATH)
    model.save_model(output_cpp_model_path, format="cpp", pool=train_pool)
    return local_canonical_file(output_cpp_model_path)


@pytest.mark.parametrize('iterations', [2, 40])
def test_export_to_python_no_cat_features(task_type, iterations):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoost({'iterations': iterations, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    output_python_model_path = test_output_path(OUTPUT_PYTHON_MODEL_PATH)
    model.save_model(output_python_model_path, format="python")
    return local_canonical_file(output_python_model_path)


@pytest.mark.parametrize('iterations', [2, 40])
def test_export_to_python_with_cat_features(task_type, iterations):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': iterations, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    output_python_model_path = test_output_path(OUTPUT_PYTHON_MODEL_PATH)
    model.save_model(output_python_model_path, format="python", pool=train_pool)
    return local_canonical_file(output_python_model_path)


def test_export_to_python_with_cat_features_from_pandas(task_type):
    model = CatBoost({'iterations': 5, 'task_type': task_type, 'devices': '0'})
    X = DataFrame([[1, 2], [3, 4]], columns=['Num', 'Categ'])
    y = [1, 0]
    cat_features = [1]
    model.fit(X, y, cat_features)
    output_python_model_path = test_output_path(OUTPUT_PYTHON_MODEL_PATH)
    model.save_model(output_python_model_path, format="python", pool=X)
    return local_canonical_file(output_python_model_path)


ONNX_TEST_PARAMETERS = [
    ('binclass', False),
    ('binclass', True),
    ('multiclass', False),
    ('regression', False),
    ('regression', True)
]


@pytest.mark.parametrize(
    'problem_type,boost_from_average',
    ONNX_TEST_PARAMETERS,
    ids=[
        'problem_type=%s-boost_from_average=%s' % (problem_type, boost_from_average)
        for problem_type, boost_from_average in ONNX_TEST_PARAMETERS
    ]
)
def test_onnx_export(problem_type, boost_from_average):
    if problem_type == 'binclass':
        loss_function = 'Logloss'
        train_path = TRAIN_FILE
        cd_path = CD_FILE
    elif problem_type == 'multiclass':
        loss_function = 'MultiClass'
        train_path = CLOUDNESS_TRAIN_FILE
        cd_path = CLOUDNESS_CD_FILE
    elif problem_type == 'regression':
        loss_function = 'RMSE'
        train_path = TRAIN_FILE
        cd_path = CD_FILE
    else:
        raise Exception('Unsupported problem_type: %s' % problem_type)

    train_pool = Pool(train_path, column_description=cd_path)

    model = CatBoost(
        {
            'task_type': 'CPU',  # TODO(akhropov): GPU results are unstable, difficult to compare models
            'loss_function': loss_function,
            'iterations': 5,
            'depth': 4,

            # onnx format export does not yet support categorical features so ignore them
            'ignored_features': train_pool.get_cat_feature_indices(),
            'boost_from_average': boost_from_average
        }
    )

    model.fit(train_pool)

    output_onnx_model_path = test_output_path(OUTPUT_ONNX_MODEL_PATH)
    model.save_model(
        output_onnx_model_path,
        format="onnx",
        export_parameters={
            'onnx_domain': 'ai.catboost',
            'onnx_model_version': 1,
            'onnx_doc_string': 'test model for problem_type %s' % problem_type,
            'onnx_graph_name': 'CatBoostModel_for_%s' % problem_type
        }
    )
    return compare_canonical_models(output_onnx_model_path, diff_limit=1e-18)


@pytest.mark.parametrize(
    'problem_type,boost_from_average',
    ONNX_TEST_PARAMETERS,
    ids=[
        'problem_type=%s-boost_from_average=%s' % (problem_type, boost_from_average)
        for problem_type, boost_from_average in ONNX_TEST_PARAMETERS
    ]
)
def test_onnx_import(problem_type, boost_from_average):
    if (problem_type == 'binclass') and boost_from_average:
        pytest.xfail('CatBoost does not support importing classification onnx models with bias')

    if problem_type == 'binclass':
        loss_function = 'Logloss'
        train_path = TRAIN_FILE
        test_path = TEST_FILE
        cd_path = CD_FILE
    elif problem_type == 'multiclass':
        loss_function = 'MultiClass'
        train_path = CLOUDNESS_TRAIN_FILE
        test_path = CLOUDNESS_TEST_FILE
        cd_path = CLOUDNESS_CD_FILE
    elif problem_type == 'regression':
        loss_function = 'RMSE'
        train_path = TRAIN_FILE
        test_path = TEST_FILE
        cd_path = CD_FILE
    else:
        raise Exception('Unsupported problem_type: %s' % problem_type)

    train_pool = Pool(train_path, column_description=cd_path)
    test_pool = Pool(test_path, column_description=cd_path)

    model = CatBoost(
        {
            'task_type': 'CPU',
            'loss_function': loss_function,
            'iterations': 5,
            'depth': 4,
            'ignored_features': train_pool.get_cat_feature_indices(),
            'boost_from_average': boost_from_average
        }
    )

    model.fit(train_pool)

    output_onnx_model_path = test_output_path(OUTPUT_ONNX_MODEL_PATH)
    model.save_model(
        output_onnx_model_path,
        format="onnx",
        export_parameters={
            'onnx_domain': 'ai.catboost',
            'onnx_model_version': 1,
            'onnx_doc_string': 'test model for problem_type %s' % problem_type,
            'onnx_graph_name': 'CatBoostModel_for_%s' % problem_type
        }
    )
    model.save_model(output_onnx_model_path, format="onnx")

    prediction_type = 'RawFormulaVal' if problem_type == 'regression' else 'Class'
    canon_pred = model.predict(test_pool, prediction_type=prediction_type)

    onnx_loaded_model = CatBoost(
        {
            'task_type': 'CPU',
            'loss_function': loss_function,
            'iterations': 5,
            'depth': 4,
            'ignored_features': train_pool.get_cat_feature_indices(),
            'boost_from_average': boost_from_average
        }
    )

    onnx_loaded_model.load_model(output_onnx_model_path, format="onnx")
    if problem_type == 'regression':
        assert (
            np.allclose(
                canon_pred,
                onnx_loaded_model.predict(test_pool, prediction_type=prediction_type),
                atol=1e-4
            )
        )
    else:
        loaded_pred = onnx_loaded_model.predict(test_pool, prediction_type=prediction_type)
        if problem_type == 'binclass':
            # TODO(akhropov): remove when MLTOOLS-4924 is fixed
            loaded_pred = np.array([value[0] for value in loaded_pred])
        assert np.all(canon_pred == loaded_pred)


def test_onnx_export_lightgbm_import_catboost():
    lightgbm_predict = np.loadtxt(CONVERT_LIGHT_GBM_PREDICTIONS)
    test = np.loadtxt(CONVERT_RANDOM_GENERATED_TEST)
    model = CatBoostRegressor()
    model.load_model(CONVERT_MODEL_ONNX, format='onnx')
    catboost_predict = model.predict(test.astype(np.float32))
    assert (np.allclose(lightgbm_predict, catboost_predict, atol=1e-4))


@pytest.mark.parametrize('problem_type', ['binclass', 'multiclass', 'regression'])
def test_pmml_export(problem_type):
    if problem_type == 'binclass':
        loss_function = 'Logloss'
        train_path = TRAIN_FILE
        cd_path = CD_FILE
    elif problem_type == 'multiclass':
        loss_function = 'MultiClass'
        train_path = CLOUDNESS_TRAIN_FILE
        cd_path = CLOUDNESS_CD_FILE
    elif problem_type == 'regression':
        loss_function = 'RMSE'
        train_path = TRAIN_FILE
        cd_path = CD_FILE
    else:
        raise Exception('Unsupported problem_type: %s' % problem_type)

    train_pool = Pool(train_path, column_description=cd_path)

    model = CatBoost(
        {
            'task_type': 'CPU',  # TODO(akhropov): GPU results are unstable, difficult to compare models
            'loss_function': loss_function,
            'iterations': 5,
            'depth': 4,
            'one_hot_max_size': 255
        }
    )

    model.fit(train_pool)

    output_pmml_model_path = test_output_path(OUTPUT_PMML_MODEL_PATH)

    if problem_type == "multiclass":
        with pytest.raises(CatBoostError):
            model.save_model(output_pmml_model_path, format="pmml")
    else:
        model.save_model(
            output_pmml_model_path,
            format="pmml",
            export_parameters={
                'pmml_copyright': '(c) catboost team',
                'pmml_description': 'CatBoostModel_for_%s' % problem_type,
                'pmml_model_version': '1'
            },
            pool=train_pool
        )
        return compare_canonical_models(output_pmml_model_path)


def test_predict_class(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict(test_pool, prediction_type="Class")
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


def test_zero_learning_rate(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0, task_type=task_type, devices='0')
    with pytest.raises(CatBoostError):
        model.fit(train_pool)


def test_predict_class_proba(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict_proba(test_pool)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


def test_no_cat_in_predict(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(train_pool)

    test_features_data, _ = load_simple_dataset_as_lists(is_test=True)
    pred1 = model.predict(test_features_data)
    pred2 = model.predict(Pool(test_features_data, cat_features=CAT_FEATURES))
    assert np.all(pred1 == pred2)


def test_save_model(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoost({'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    model2 = CatBoost()
    model2.load_model(output_model_path)
    pred1 = model.predict(test_pool)
    pred2 = model2.predict(test_pool)
    assert _check_data(pred1, pred2)


def test_multiclass(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=2, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    classifier.save_model(output_model_path)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(output_model_path)
    pred = new_classifier.predict_proba(pool)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.round(np.array(pred)), 9)
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('missed_classes', [False, True], ids=['missed_classes=False', 'missed_classes=True'])
def test_multiclass_classes_count(task_type, missed_classes):
    object_count = 100
    feature_count = 10
    classes_count = 4
    if missed_classes:
        unique_labels = [1, 3]
    else:
        unique_labels = [0, 1, 2, 3]
    expected_classes_attr = [i for i in range(4)]

    prng = np.random.RandomState(seed=0)
    pool = Pool(
        prng.random_sample(size=(object_count, feature_count)),
        label=prng.choice(unique_labels, size=object_count)
    )

    # returns pred probabilities
    def check_classifier(classifier):
        assert np.all(classifier.classes_ == expected_classes_attr)

        pred_classes = classifier.predict(pool)
        assert all([(pred_class in unique_labels) for pred_class in pred_classes])

        pred_probabilities = classifier.predict_proba(pool)
        assert pred_probabilities.shape == (object_count, classes_count)
        return pred_probabilities

    classifier = CatBoostClassifier(
        classes_count=classes_count,
        iterations=2,
        loss_function='MultiClass',
        thread_count=8,
        task_type=task_type,
        devices='0'
    )
    classifier.fit(pool)

    check_classifier(classifier)

    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    classifier.save_model(output_model_path)

    new_classifier = CatBoostClassifier()
    new_classifier.load_model(output_model_path)

    pred_probabilities = check_classifier(new_classifier)

    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(pred_probabilities), delimiter='\t')
    return [
        local_canonical_file(preds_path, diff_tool=get_limited_precision_dsv_diff_tool(1e-6, False)),
        compare_canonical_models(output_model_path, diff_limit=1.e-6)
    ]


@pytest.mark.parametrize('label_type', ['string', 'int'])
def test_multiclass_custom_class_labels(label_type, task_type):
    if label_type == 'int':
        train_labels = [1, 2]
    elif label_type == 'string':
        train_labels = ['Class1', 'Class2']
    prng = np.random.RandomState(seed=0)
    train_pool = Pool(prng.random_sample(size=(100, 10)), label=prng.choice(train_labels, size=100))
    test_pool = Pool(prng.random_sample(size=(50, 10)))
    classifier = CatBoostClassifier(iterations=2, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(train_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    classifier.save_model(output_model_path)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(output_model_path)
    pred = new_classifier.predict_proba(test_pool)
    classes = new_classifier.predict(test_pool)
    assert pred.shape == (50, 2)
    assert all(((class1 in train_labels) for class1 in classes))
    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(pred), fmt='%.8f')
    return local_canonical_file(preds_path)


def test_multiclass_custom_class_labels_from_files(task_type):
    labels = ['a', 'b', 'c', 'd']

    cd_path = test_output_path('cd.txt')
    np.savetxt(cd_path, [[0, 'Target']], fmt='%s', delimiter='\t')

    prng = np.random.RandomState(seed=0)

    train_path = test_output_path('train.txt')
    np.savetxt(train_path, generate_concatenated_random_labeled_dataset(100, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    test_path = test_output_path('test.txt')
    np.savetxt(test_path, generate_concatenated_random_labeled_dataset(25, 10, labels, prng=prng), fmt='%s', delimiter='\t')

    train_pool = Pool(train_path, column_description=cd_path)
    test_pool = Pool(test_path, column_description=cd_path)
    classifier = CatBoostClassifier(iterations=2, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(train_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    classifier.save_model(output_model_path)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(output_model_path)
    pred = new_classifier.predict_proba(test_pool)
    classes = new_classifier.predict(test_pool)
    assert pred.shape == (25, 4)
    assert all(((class1 in labels) for class1 in classes))
    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(pred), fmt='%.8f')
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('prediction_type', ['RawFormulaVal', 'Probability', 'Class'])
def test_multilabel(prediction_type):
    pool = Pool(SCENE_TRAIN_FILE, column_description=SCENE_CD_FILE)
    classifier = CatBoostClassifier(iterations=2, loss_function='MultiLogloss', thread_count=8)
    classifier.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    classifier.save_model(output_model_path)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(output_model_path)
    pred = new_classifier.predict(pool, prediction_type=prediction_type)
    preds_path = test_output_path(PREDS_PATH)
    if prediction_type == 'Class':
        np.savetxt(preds_path, np.array(pred), fmt='%.0f')
    else:
        np.savetxt(preds_path, np.array(pred), fmt='%.8f')
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('target_count', [1, 2, 3])
def test_compare_multilogloss_with_logloss(target_count):
    object_count = 100
    feature_count = 10
    iterations = 100
    np.random.seed(42)
    X = np.random.random((object_count, feature_count))
    y = np.random.randint(low=0, high=2, size=object_count)
    Y = np.vstack([y for _ in range(target_count)]).T
    logloss_clf = CatBoostClassifier(iterations=iterations, loss_function='Logloss')
    logloss_clf.fit(X, y)
    multilogloss_clf = CatBoostClassifier(iterations=iterations, loss_function='MultiLogloss')
    multilogloss_clf.fit(X, Y)
    diff = logloss_clf.best_score_['learn']['Logloss'] / multilogloss_clf.best_score_['learn']['MultiLogloss']
    assert 0.9 < diff < 1.1


def test_multilabel_class_names():
    pool = Pool(SCENE_TRAIN_FILE, column_description=SCENE_CD_FILE)
    classifier = CatBoostClassifier(
        iterations=100,
        loss_function='MultiLogloss',
        class_names=['a', 'b', 'c', 'd', 'e', 'f']
    )
    classifier.fit(pool)
    pred = classifier.predict(pool, prediction_type='Class')
    assert np.array_equal(classifier.classes_, ['a', 'b', 'c', 'd', 'e', 'f'])
    assert set(pred.flatten()) == {0, 1}


def test_multilabel_eval_metric():
    train_pool = Pool(SCENE_TRAIN_FILE, column_description=SCENE_CD_FILE)
    test_pool = Pool(SCENE_TEST_FILE, column_description=SCENE_CD_FILE)
    classifier = CatBoostClassifier(iterations=100, loss_function='MultiLogloss')
    classifier.fit(train_pool, eval_set=test_pool)
    target = test_pool.get_label().astype(int)
    num_classes = target.shape[1]
    approx = classifier.predict(test_pool, prediction_type='RawFormulaVal')
    prediction = classifier.predict(test_pool, prediction_type='Class')
    accuracy = eval_metric(target, approx, 'Accuracy')[0]
    assert accuracy == np.mean([np.array_equal(t, p) for t, p in zip(target, prediction)])
    accuracy_per_class = eval_metric(target, approx, 'Accuracy:type=PerClass')
    assert len(accuracy_per_class) == num_classes
    for j in range(num_classes):
        assert accuracy_per_class[j] == eval_metric(target[:, j], approx[:, j], 'Accuracy')[0]
    for metric in ('Precision', 'Recall', 'F1'):
        metric_per_class = eval_metric(target, approx, metric)
        assert len(metric_per_class) == num_classes
        for j in range(num_classes):
            assert metric_per_class[j] == eval_metric(target[:, j], approx[:, j], metric)[0]


def test_class_names(task_type):
    class_names = ['Small', 'Medium', 'Large']

    prng = np.random.RandomState(seed=0)
    train_pool = Pool(prng.random_sample(size=(100, 10)), label=prng.choice(class_names, size=100))
    test_pool = Pool(prng.random_sample(size=(25, 10)))

    classifier = CatBoostClassifier(
        iterations=2,
        loss_function='MultiClass',
        class_names=class_names,
        thread_count=8,
        task_type=task_type,
        devices='0'
    )
    classifier.fit(train_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    classifier.save_model(output_model_path)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(output_model_path)
    pred = new_classifier.predict_proba(test_pool)
    classes = new_classifier.predict(test_pool)
    assert pred.shape == (25, 3)
    assert all(((class1 in class_names) for class1 in classes))
    assert sorted(classifier.classes_) == sorted(class_names)
    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(pred), fmt='%.8f')
    return local_canonical_file(preds_path)


def test_inconsistent_labels_and_class_names():
    class_names = ['Small', 'Medium', 'Large']

    prng = np.random.RandomState(seed=0)
    train_pool = Pool(prng.random_sample(size=(100, 10)), label=prng.choice([0, 1, 2], size=100))

    classifier = CatBoostClassifier(
        iterations=2,
        loss_function='MultiClass',
        class_names=class_names,
    )
    with pytest.raises(CatBoostError):
        classifier.fit(train_pool)


def test_inconsistent_class_labels_count():
    classifier = CatBoostClassifier(
        iterations=2,
        loss_function='Logloss',
        class_names=[0, 1, 2],
    )
    with pytest.raises(CatBoostError):
        classifier.fit([[0], [1], [2]], [0, 1, 2])

    classifier = CatBoostClassifier(
        iterations=2,
        loss_function='MultiClass',
        class_names=[0, 1],
        classes_count=3
    )
    with pytest.raises(CatBoostError):
        classifier.fit([[0], [1], [2]], [0, 1, 2])


def test_unknown_class_labels_in_eval_dataset():
    classifier = CatBoostClassifier(
        iterations=2,
        loss_function='Logloss',
        class_names=[0, 1, 2],
    )
    with pytest.raises(CatBoostError):
        classifier.fit([[0], [1], [2]], [0, 1, 2])

    classifier = CatBoostClassifier(
        iterations=2,
        loss_function='MultiClass',
        class_names=[0, 1],
        classes_count=3
    )
    with pytest.raises(CatBoostError):
        classifier.fit([[0], [1], [2]], [0, 1, 2])


@pytest.mark.parametrize(
    'features_dtype',
    ['str', 'np.float32'],
    ids=['features_dtype=str', 'features_dtype=np.float32']
)
def test_querywise(features_dtype, task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    model = CatBoostRanker(loss_function='QueryRMSE', iterations=2, thread_count=8, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred1 = model.predict(test_pool)

    df = read_csv(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_query_id = df.loc[:, 1]
    train_target = df.loc[:, 2]
    train_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(eval(features_dtype))

    df = read_csv(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(eval(features_dtype))

    model.fit(train_data, train_target, group_id=train_query_id)
    pred2 = model.predict(test_data)
    assert _check_data(pred1, pred2)


def test_group_weight(task_type):
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
    model = CatBoostRanker(loss_function='YetiRank', iterations=10, thread_count=8, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred1 = model.predict(test_pool)

    df = read_csv(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_query_weight = df.loc[:, 0]
    train_query_id = df.loc[:, 1]
    train_target = df.loc[:, 2]
    train_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(str)

    df = read_csv(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_query_weight = df.loc[:, 0]
    test_query_id = df.loc[:, 1]
    test_data = Pool(df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32), group_id=test_query_id, group_weight=test_query_weight)

    model.fit(train_data, train_target, group_id=train_query_id, group_weight=train_query_weight)
    pred2 = model.predict(test_data)
    assert _check_data(pred1, pred2)


def test_zero_baseline(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    baseline = np.zeros(pool.num_row())
    pool.set_baseline(baseline)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_ones_weight(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    weight = np.ones(pool.num_row())
    pool.set_weight(weight)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_non_ones_weight(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    weight = np.arange(1, pool.num_row() + 1)
    pool.set_weight(weight)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_ones_weight_equal_to_nonspecified_weight(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=0.03, task_type=task_type, devices='0')

    predictions = []

    for set_weights in [False, True]:
        if set_weights:
            weight = np.ones(train_pool.num_row())
            train_pool.set_weight(weight)
        model.fit(train_pool)
        predictions.append(model.predict(test_pool))

    assert np.all(predictions[0] == predictions[1])


def test_py_data_group_id(task_type):
    train_pool_from_files = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_ID)
    test_pool_from_files = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_ID)
    model = CatBoostRanker(loss_function='QueryRMSE', iterations=2, thread_count=4, task_type=task_type, devices='0')
    model.fit(train_pool_from_files)
    predictions_from_files = model.predict(test_pool_from_files)

    train_df = read_csv(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_target = train_df.loc[:, 2]
    raw_train_group_id = train_df.loc[:, 1]
    train_data = train_df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32)

    test_df = read_csv(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = Pool(test_df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32))

    for group_id_func in (int, str, lambda id: 'myid_' + str(id)):
        train_group_id = [group_id_func(group_id) for group_id in raw_train_group_id]
        model.fit(train_data, train_target, group_id=train_group_id)
        predictions_from_py_data = model.predict(test_data)
        assert _check_data(predictions_from_files, predictions_from_py_data)


def test_py_data_subgroup_id(task_type):
    train_pool_from_files = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE_WITH_SUBGROUP_ID)
    test_pool_from_files = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE_WITH_SUBGROUP_ID)
    model = CatBoostRanker(loss_function='QueryRMSE', iterations=2, thread_count=4, task_type=task_type, devices='0')
    model.fit(train_pool_from_files)
    predictions_from_files = model.predict(test_pool_from_files)

    train_df = read_csv(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_group_id = train_df.loc[:, 1]
    raw_train_subgroup_id = train_df.loc[:, 4]
    train_target = train_df.loc[:, 2]
    train_data = train_df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32)

    test_df = read_csv(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = Pool(test_df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32))

    for subgroup_id_func in (int, str, lambda id: 'myid_' + str(id)):
        train_subgroup_id = [subgroup_id_func(subgroup_id) for subgroup_id in raw_train_subgroup_id]
        model.fit(train_data, train_target, group_id=train_group_id, subgroup_id=train_subgroup_id)
        predictions_from_py_data = model.predict(test_data)
        assert _check_data(predictions_from_files, predictions_from_py_data)


def test_fit_data(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    eval_pool = Pool(CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE)
    base_model = CatBoostClassifier(iterations=10, learning_rate=0.05, loss_function="MultiClass", task_type=task_type, devices='0')
    base_model.fit(pool)
    baseline = np.array(base_model.predict(pool, prediction_type='RawFormulaVal'))
    eval_baseline = np.array(base_model.predict(eval_pool, prediction_type='RawFormulaVal'))
    eval_pool.set_baseline(eval_baseline)
    model = CatBoostClassifier(iterations=90, learning_rate=0.05, loss_function="MultiClass")
    data = get_features_data_from_file(
        CLOUDNESS_TRAIN_FILE,
        drop_columns=[0],
        cat_feature_indices=pool.get_cat_feature_indices()
    )
    model.fit(data, pool.get_label(), sample_weight=np.arange(1, pool.num_row() + 1), baseline=baseline, use_best_model=True, eval_set=eval_pool)
    pred = model.predict_proba(eval_pool)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path, diff_tool=get_limited_precision_numpy_diff_tool())


def test_fit_predict_baseline(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_pool.set_baseline(np.arange(1, train_pool.num_row() + 1))
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    test_baseline = np.arange(0, test_pool.num_row())
    test_pool.set_baseline(test_baseline)
    test_pool_without_baseline = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=100, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict(test_pool)
    pred_no_baseline = model.predict(test_pool_without_baseline)
    preds_path = test_output_path(PREDS_PATH)
    assert np.allclose(pred, pred_no_baseline + test_baseline)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


def test_ntree_limit(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=100, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(train_pool)
    pred = model.predict_proba(test_pool, ntree_end=10)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


def test_ntree_invalid_range():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=100, learning_rate=0.03)
    model.fit(train_pool)
    with pytest.raises(CatBoostError):
        model.predict_proba(test_pool, ntree_end=10000000)
        model.predict_proba(test_pool, ntree_start=10000000)
        model.predict_proba(test_pool, ntree_start=20, ntree_end=10)


def test_staged_predict(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(train_pool)
    preds = []
    for pred in model.staged_predict(test_pool):
        preds.append(pred)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(preds))
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('problem', ['Classifier', 'Regressor'])
def test_staged_predict_and_predict_proba_on_single_object(problem):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    if problem == 'Classifier':
        model = CatBoostClassifier(iterations=10)
    else:
        model = CatBoostRegressor(iterations=10)

    model.fit(train_pool)

    test_data = read_csv(TEST_FILE, header=None, delimiter='\t')
    test_data.drop([TARGET_IDX], axis=1, inplace=True)

    preds = []
    for pred in model.staged_predict(test_data):
        preds.append(pred)

    if problem == 'Classifier':
        pred_probabilities = []
        for pred_probabilities_for_iteration in model.staged_predict_proba(test_data):
            pred_probabilities.append(pred_probabilities_for_iteration)

    random.seed(0)
    for i in range(3):  # just some indices
        test_object_idx = random.randrange(test_data.shape[0])

        single_object_preds = []
        for pred in model.staged_predict(test_data.values[test_object_idx]):
            single_object_preds.append(pred)

        assert len(preds) == len(single_object_preds)
        for iteration in range(len(preds)):
            assert preds[iteration][test_object_idx] == single_object_preds[iteration]

        if problem == 'Classifier':
            single_object_pred_probabilities = []
            for pred_probabilities_for_iteration in model.staged_predict_proba(test_data.values[test_object_idx]):
                single_object_pred_probabilities.append(pred_probabilities_for_iteration)

            assert len(pred_probabilities) == len(single_object_pred_probabilities)
            for iteration in range(len(pred_probabilities)):
                assert np.array_equal(pred_probabilities[iteration][test_object_idx], single_object_pred_probabilities[iteration])


def test_invalid_loss_base(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost({"loss_function": "abcdef", 'task_type': task_type, 'devices': '0'})
    with pytest.raises(CatBoostError):
        model.fit(pool)


def test_invalid_loss_classifier(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(loss_function="abcdef", task_type=task_type, devices='0')
    with pytest.raises(CatBoostError):
        model.fit(pool)


def test_invalid_loss_regressor(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(loss_function="fee", task_type=task_type, devices='0')
    with pytest.raises(CatBoostError):
        model.fit(pool)


def test_invalid_loss_ranker(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(loss_function="MultiClass", task_type=task_type, devices='0')
    with pytest.raises(CatBoostError):
        model.fit(pool)


def test_fit_no_label(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(task_type=task_type, devices='0')
    with pytest.raises(CatBoostError):
        model.fit(pool.get_features())


def test_predict_without_fit(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(task_type=task_type, devices='0')
    with pytest.raises(CatBoostError):
        model.predict(pool)


def test_real_numbers_cat_features():
    prng = np.random.RandomState(seed=20181219)
    data = prng.rand(100, 10)
    label = _generate_nontrivial_binary_target(100, prng=prng)
    with pytest.raises(CatBoostError):
        Pool(data, label, [1, 2])


def test_wrong_ctr_for_classification(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(ctr_description=['Borders:TargetBorderCount=5:TargetBorderType=Uniform'], task_type=task_type, devices='0')
    with pytest.raises(CatBoostError):
        model.fit(pool)


def test_wrong_feature_count(task_type):
    prng = np.random.RandomState(seed=20181219)
    data = prng.rand(100, 10)
    label = _generate_nontrivial_binary_target(100, prng=prng)
    model = CatBoostClassifier(task_type=task_type, devices='0')
    model.fit(data, label)
    with pytest.raises(CatBoostError):
        model.predict(data[:, :-1])


def test_wrong_params_classifier():
    with pytest.raises(TypeError):
        CatBoostClassifier(wrong_param=1)


def test_wrong_params_base():
    prng = np.random.RandomState(seed=20181219)
    data = prng.rand(100, 10)
    label = _generate_nontrivial_binary_target(100, prng=prng)
    model = CatBoost({'wrong_param': 1})
    with pytest.raises(CatBoostError):
        model.fit(data, label)


def test_wrong_params_regressor():
    with pytest.raises(TypeError):
        CatBoostRegressor(wrong_param=1)


def test_wrong_params_ranker():
    with pytest.raises(TypeError):
        CatBoostRegressor(wrong_param=1)


def test_wrong_kwargs_base():
    prng = np.random.RandomState(seed=20181219)
    data = prng.rand(100, 10)
    label = _generate_nontrivial_binary_target(100, prng=prng)
    model = CatBoost({'kwargs': {'wrong_param': 1}})
    with pytest.raises(CatBoostError):
        model.fit(data, label)


def test_duplicate_params_base():
    prng = np.random.RandomState(seed=20181219)
    data = prng.rand(100, 10)
    label = _generate_nontrivial_binary_target(100, prng=prng)
    model = CatBoost({'iterations': 100, 'n_estimators': 50})
    with pytest.raises(CatBoostError):
        model.fit(data, label)


def test_duplicate_params_classifier():
    prng = np.random.RandomState(seed=20181219)
    data = prng.rand(100, 10)
    label = _generate_nontrivial_binary_target(100, prng=prng)
    model = CatBoostClassifier(depth=3, max_depth=4, random_seed=42, random_state=12)
    with pytest.raises(CatBoostError):
        model.fit(data, label)


def test_duplicate_params_regressor():
    prng = np.random.RandomState(seed=20181219)
    data = prng.rand(100, 10)
    label = _generate_nontrivial_binary_target(100, prng=prng)
    model = CatBoostRegressor(learning_rate=0.1, eta=0.03, border_count=10, max_bin=12)
    with pytest.raises(CatBoostError):
        model.fit(data, label)


def test_generated_metrics_default_params():
    metrics_without_default_params = (
        metrics.TotalF1, metrics.AUC, metrics.NDCG, metrics.CtrFactor, metrics.RecallAt, metrics.QueryCrossEntropy,
        metrics.Logloss, metrics.LogLikelihoodOfPrediction, metrics.MultiClassOneVsAll, metrics.R2, metrics.MAPE,
        metrics.UserQuerywiseMetric, metrics.MSLE, metrics.Expectile, metrics.FilteredDCG, metrics.DCG, metrics.MCC,
        metrics.PrecisionAt, metrics.CrossEntropy, metrics.ERR, metrics.SMAPE, metrics.PairAccuracy, metrics.FairLoss,
        metrics.QuerySoftMax, metrics.PRAUC, metrics.MAP, metrics.MedianAbsoluteError, metrics.F1,
        metrics.NormalizedGini, metrics.BrierScore, metrics.Precision, metrics.HingeLoss, metrics.ZeroOneLoss,
        metrics.WKappa, metrics.Combination, metrics.MAE, metrics.PairLogit, metrics.Kappa, metrics.MRR, metrics.RMSE,
        metrics.Poisson, metrics.BalancedAccuracy, metrics.Accuracy, metrics.MultiClass, metrics.HammingLoss,
        metrics.QueryRMSE, metrics.RMSEWithUncertainty, metrics.QueryAUC, metrics.LogLinQuantile, metrics.Recall,
        metrics.BalancedErrorRate, metrics.MultiRMSE, metrics.Quantile, metrics.PFound
    )
    for metric in metrics_without_default_params:
        assert str(metric()) == metric.__name__
    metrics_with_mandatory_params = (
        metrics.NumErrors, metrics.Huber, metrics.AverageGain, metrics.QueryAverage, metrics.Lq, metrics.Tweedie
    )
    for metric in metrics_with_mandatory_params:
        with pytest.raises(ValueError):
            metric()


def _test_generated_metrics(loss_function, train_pool, test_pool, metrics, task):
    model = CatBoost({'loss_function': loss_function, 'iterations': 50})
    model.fit(train_pool, eval_set=test_pool)

    # text description metrics
    td_results = model.eval_metrics(test_pool, list(metrics.keys()))
    # instances of autogenerated metric classes
    ag_results = model.eval_metrics(test_pool, list(metrics.values()))

    for metric in metrics:
        assert np.allclose(td_results[metric], ag_results[metric]), "Different results for {}".format(metric)

    if task not in ("ranking", "multiclassification", "multiregression"):
        preds = model.predict(test_pool, prediction_type='RawFormulaVal')
        for metric, metric_obj in metrics.items():
            score = metric_obj.eval(test_pool.get_label(), preds)[0]
            assert np.abs(td_results[metric][-1] - score) < 1e-6, "Eval metric results differ for {}".format(metric)


def test_f1_vs_fbeta():
    model = CatBoost({'loss_function': metrics.CrossEntropy(), 'iterations': 50})
    train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)
    model.fit(train_pool, eval_set=test_pool)

    results = model.eval_metrics(test_pool, ['F1', 'F:beta=1'])
    assert np.allclose(results['F1'], results['F:beta=1']), "Different results for F1 and F:beta=1"


def test_generated_classification_metrics():
    _test_generated_metrics(
        'Logloss',
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        {
            'Logloss': metrics.Logloss(), 'CrossEntropy': metrics.CrossEntropy(), 'Precision': metrics.Precision(),
            'Recall': metrics.Recall(), 'F1': metrics.F1(), 'BalancedAccuracy': metrics.BalancedAccuracy(),
            'BalancedErrorRate': metrics.BalancedErrorRate(), 'MCC': metrics.MCC(), 'Accuracy': metrics.Accuracy(),
            'CtrFactor': metrics.CtrFactor(), 'AUC': metrics.AUC(), 'AUC:type=Ranking': metrics.AUC(type='Ranking'),
            'NormalizedGini': metrics.NormalizedGini(), 'BrierScore': metrics.BrierScore(),
            'HingeLoss': metrics.HingeLoss(), 'HammingLoss': metrics.HammingLoss(),
            'ZeroOneLoss': metrics.ZeroOneLoss(), 'Kappa': metrics.Kappa(), 'WKappa': metrics.WKappa(),
            'LogLikelihoodOfPrediction': metrics.LogLikelihoodOfPrediction(), 'F:beta=2': metrics.F(beta=2)
        },
        "classification"
    )


def test_generated_regression_metrics_with_default_params():
    _test_generated_metrics(
        'RMSE',
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        {
            'MAE': metrics.MAE(), 'MAPE': metrics.MAPE(), 'Poisson': metrics.Poisson(),
            # Metric description should be 'Quantile' instead of 'Quantile:alpha=0.5'
            'Quantile:alpha=0.5': metrics.Quantile(), 'RMSE': metrics.RMSE(),
            'LogLinQuantile': metrics.LogLinQuantile(), 'Expectile': metrics.Expectile(),
            'FairLoss': metrics.FairLoss(), 'SMAPE': metrics.SMAPE(),
            'R2': metrics.R2(), 'MSLE': metrics.MSLE(), 'MedianAbsoluteError': metrics.MedianAbsoluteError()
        },
        "regression"
    )


def test_generated_regression_metrics_with_specified_params():
    _test_generated_metrics(
        'RMSE',
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        {
            'Quantile:alpha=0.2': metrics.Quantile(alpha=0.2), 'Quantile:alpha=0.8': metrics.Quantile(alpha=0.8),
            'LogLinQuantile:alpha=0.2': metrics.LogLinQuantile(alpha=0.2),
            'LogLinQuantile:alpha=0.8': metrics.LogLinQuantile(alpha=0.8),
            'Lq:q=1.0': metrics.Lq(q=1.0), 'Lq:q=2.0': metrics.Lq(q=2.0), 'Lq:q=10.0': metrics.Lq(q=10.0),
            'Huber:delta=1.0': metrics.Huber(delta=1.0), 'Huber:delta=0.5': metrics.Huber(delta=0.5),
            'Huber:delta=2.0': metrics.Huber(delta=2.0),
            'Expectile:alpha=0.2': metrics.Expectile(alpha=0.2), 'Expectile:alpha=0.8': metrics.Expectile(alpha=0.8),
            'Tweedie:variance_power=1.2': metrics.Tweedie(variance_power=1.2),
            'Tweedie:variance_power=1.5': metrics.Tweedie(variance_power=1.5),
            'Tweedie:variance_power=1.8': metrics.Tweedie(variance_power=1.8),
            'FairLoss:smoothness=0.5': metrics.FairLoss(smoothness=0.5),
            'FairLoss:smoothness=2.0': metrics.FairLoss(smoothness=2.0),
            'NumErrors:greater_than=10': metrics.NumErrors(greater_than=10),
            'NumErrors:greater_than=20': metrics.NumErrors(greater_than=20),
        },
        "regression"
    )


def test_generated_multiclassification_aggregate_metrics():
    _test_generated_metrics(
        'MultiClass',
        Pool(data=CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE),
        Pool(data=CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE),
        {
            'MultiClass': metrics.MultiClass(), 'MultiClassOneVsAll': metrics.MultiClassOneVsAll(),
            'TotalF1': metrics.TotalF1(), 'TotalF1:average=Macro': metrics.TotalF1(average='Macro'),
            'TotalF1:average=Micro': metrics.TotalF1(average='Micro'),
            'MCC': metrics.MCC(),
            'Accuracy': metrics.Accuracy(), 'HingeLoss': metrics.HingeLoss(), 'HammingLoss': metrics.HammingLoss(),
            'ZeroOneLoss': metrics.ZeroOneLoss(), 'Kappa': metrics.Kappa(), 'WKappa': metrics.WKappa(),
            'AUC:type=Mu': metrics.AUCMulticlass(),
            'AUC:type=Mu;misclass_cost_matrix=0/0.5/2/1/0/1/0/0.5/0':
                metrics.AUCMulticlass(misclass_cost_matrix=np.array([[0.0, 0.5, 2.0], [1.0, 0.0, 1.0], [0.0, 0.5, 0.0]]))
        },
        "multiclassification"
    )


def test_generated_multiclassification_class_metrics():
    train_pool = Pool(data=CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    test_pool = Pool(data=CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE)
    n_classes = len(set(train_pool.get_label()))

    model = CatBoost({'loss_function': 'MultiClass', 'iterations': 50})
    model.fit(train_pool, eval_set=test_pool)

    class_metrics = (
        {'name': 'Precision', 'text': 'Precision', 'instance': metrics.Precision()},
        {'name': 'Recall', 'text': 'Recall', 'instance': metrics.Recall()},
        {'name': 'F1', 'text': 'F1', 'instance': metrics.F1()},
        {'name': 'AUC', 'text': 'AUC:type=OneVsAll', 'instance': metrics.AUC(type='OneVsAll')},
    )

    # text description metrics
    td_results = model.eval_metrics(test_pool, list(map(operator.itemgetter('text'), class_metrics)))
    # instances of autogenerated metric classes
    ag_results = model.eval_metrics(test_pool, list(map(operator.itemgetter('instance'), class_metrics)))
    for metric in class_metrics:
        for i_class in range(n_classes):
            metric_key = '{}:class={}'.format(metric['name'], i_class)
            assert np.allclose(td_results[metric_key], ag_results[metric_key]),\
                "Different results for {}".format(metric_key)


def test_generated_ranking_pairwise_metric():
    _test_generated_metrics(
        'PairLogit',
        Pool(data=QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE),
        Pool(data=QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE),
        {
            'PairLogit': metrics.PairLogit(), 'PairAccuracy': metrics.PairAccuracy(),
            'AUC': metrics.AUC(), 'AUC:type=Ranking': metrics.AUC(type='Ranking')
        },
        "ranking"
    )


def test_generated_ranking_groupwise_metric():
    _test_generated_metrics(
        'QueryRMSE',
        Pool(data=QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE),
        Pool(data=QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE),
        {
            'QueryRMSE': metrics.QueryRMSE(), 'PFound': metrics.PFound(),
            # Metric descriptions should be 'NDCG' & 'DCG' instead of 'NDCG:type=Base' & 'DCG:type=Base' respectively
            'NDCG:type=Base': metrics.NDCG(), 'DCG:type=Base': metrics.DCG(),
            'FilteredDCG': metrics.FilteredDCG(), 'AverageGain:top=5': metrics.AverageGain(top=5),
            'PrecisionAt': metrics.PrecisionAt(), 'RecallAt': metrics.RecallAt(), 'MAP': metrics.MAP(),
            'AUC': metrics.AUC()
        },
        "ranking"
    )


def test_generated_multiregression_metric_and_loss():
    _test_generated_metrics(
        metrics.MultiRMSE(),
        Pool(data=MULTIREGRESSION_TRAIN_FILE, column_description=MULTIREGRESSION_CD_FILE),
        Pool(data=MULTIREGRESSION_TEST_FILE, column_description=MULTIREGRESSION_CD_FILE),
        {
            'MultiRMSE': metrics.MultiRMSE()
        },
        "multiregression"
    )


def _test_generated_losses(train_pool, test_pool, losses, result_dtype=float):
    catboost_params = {'iterations': 50}
    for loss_text, loss_instance in losses.items():
        # text-description metric/loss
        td_params = catboost_params.copy()
        td_params['loss_function'] = loss_text
        td_model = CatBoost(td_params)
        td_model.fit(train_pool, eval_set=test_pool)
        td_pred = td_model.predict(test_pool)
        # instances of autogenerated metric/loss classes
        ag_params = catboost_params.copy()
        ag_params['loss_function'] = loss_instance
        ag_model = CatBoost(ag_params)
        ag_model.fit(train_pool, eval_set=test_pool)
        ag_pred = ag_model.predict(test_pool)
        np.allclose(td_pred.astype(result_dtype), ag_pred.astype(result_dtype)),\
            "Different results for {}".format(loss_text)


def test_generated_classification_losses():
    _test_generated_losses(
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        {'Logloss': metrics.Logloss(), 'CrossEntropy': metrics.CrossEntropy()},
        int
    )


def test_generated_regression_losses_with_default_params():
    _test_generated_losses(
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        {
            'MAE': metrics.MAE(), 'MAPE': metrics.MAPE(), 'Poisson': metrics.Poisson(),
            'Quantile:alpha=0.5': metrics.Quantile(), 'RMSE': metrics.RMSE(),
            'RMSEWithUncertainty': metrics.RMSEWithUncertainty(), 'LogLinQuantile': metrics.LogLinQuantile(),
            # BUG: Expectile incorrectly expects alpha in catboost.core.Catboost({'loss_function': 'Expectile'})
            # 'Expectile': metrics.Expectile()
        }
    )


def test_generated_regression_losses_with_specified_params():
    _test_generated_losses(
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        {
            'Quantile:alpha=0.2': metrics.Quantile(alpha=0.2), 'Quantile:alpha=0.8': metrics.Quantile(alpha=0.8),
            'LogLinQuantile:alpha=0.2': metrics.LogLinQuantile(alpha=0.2),
            'LogLinQuantile:alpha=0.8': metrics.LogLinQuantile(alpha=0.8),
            'Lq:q=1.0': metrics.Lq(q=1.0), 'Lq:q=2.0': metrics.Lq(q=2.0), 'Lq:q=10.0': metrics.Lq(q=10.0),
            'Huber:delta=1.0': metrics.Huber(delta=1.0), 'Huber:delta=0.5': metrics.Huber(delta=0.5),
            'Huber:delta=2.0': metrics.Huber(delta=2.0),
            'Expectile:alpha=0.2': metrics.Expectile(alpha=0.2), 'Expectile:alpha=0.8': metrics.Expectile(alpha=0.8),
            'MultiQuantile:alpha=0.2,0.4': metrics.MultiQuantile(alpha="0.2,0.4"),
            'Tweedie:variance_power=1.2': metrics.Tweedie(variance_power=1.2),
            'Tweedie:variance_power=1.5': metrics.Tweedie(variance_power=1.5),
            'Tweedie:variance_power=1.8': metrics.Tweedie(variance_power=1.8)
        }
    )


def test_generated_multiclassification_losses():
    _test_generated_losses(
        Pool(data=CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE),
        Pool(data=CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE),
        {'MultiClass': metrics.MultiClass(), 'MultiClassOneVsAll': metrics.MultiClassOneVsAll()}
    )


def test_generated_pairwise_ranking_losses():
    _test_generated_losses(
        Pool(data=QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE),
        Pool(data=QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE),
        {
            'PairLogit': metrics.PairLogit(), 'PairLogit:max_pairs=2': metrics.PairLogit(max_pairs=2),
            'PairLogit:max_pairs=5': metrics.PairLogit(max_pairs=5)
        }
    )


def test_generated_groupwise_ranking_losses():
    _test_generated_losses(
        Pool(data=QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE),
        Pool(data=QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE),
        {
            'QueryRMSE': metrics.QueryRMSE(), 'QuerySoftMax': metrics.QuerySoftMax(),
            'QuerySoftMax:beta=0.5': metrics.QuerySoftMax(beta=0.5),
            'QuerySoftMax:beta=2.0': metrics.QuerySoftMax(beta=2.0)
        }
    )


def _test_metric_with_default_params_description(loss_function, train_pool, test_pool, metric):
    model = CatBoost({'loss_function': loss_function, 'iterations': 10})
    model.fit(train_pool, eval_set=test_pool)
    results = model.eval_metrics(test_pool, metric)
    assert metric in results


@pytest.mark.xfail
def test_quantile_with_default_params_description():
    _test_metric_with_default_params_description(
        'RMSE',
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        'Quantile'
    )


@pytest.mark.xfail
def test_dcg_with_default_params_description():
    _test_metric_with_default_params_description(
        'QueryRMSE',
        Pool(data=TRAIN_FILE, column_description=CD_FILE),
        Pool(data=TEST_FILE, column_description=CD_FILE),
        'DCG'
    )


@pytest.mark.xfail
def test_expectile_loss_with_default_params():
    train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)
    model = CatBoost({'iterations': 50, 'loss_function': 'Expectile'})
    model.fit(train_pool, eval_set=test_pool)


def test_generated_metrics():
    # Test equivalence of the generated metric classes to string representations.
    train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)

    model = CatBoostRegressor()
    model.fit(train_pool)
    eval_metrics = [metrics.Lq(q=3)]
    metrics_evals = model.eval_metrics(test_pool, eval_metrics)
    for metric in eval_metrics:
        assert str(metric) in metrics_evals

    # Test setting hints and default values.
    results = cv(
        train_pool,
        {
            'iterations': 20,
            'learning_rate': 0.03,
            'loss_function': metrics.Logloss().set_hints(skip_train=True),
            'eval_metric': metrics.AUC()
        },
    )
    for non_existing_field in ('train-Logloss-mean', 'train-Logloss-std', 'train-AUC-mean', 'train-AUC-std'):
        assert non_existing_field not in results


def test_metrics_is_min_max_optimal():
    logloss = metrics.Logloss()
    assert logloss.is_min_optimal() and not logloss.is_max_optimal()
    assert is_minimizable_metric('Logloss') and not is_maximizable_metric('Logloss')
    auc = metrics.AUC()
    assert auc.is_max_optimal() and not auc.is_min_optimal()
    assert is_maximizable_metric('AUC') and not is_minimizable_metric('AUC')


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

            for i in range(len(approx)):
                w = 1.0 if weight is None else weight[i]
                weight_sum += w
                error_sum += w * (target[i] * approx[i] - math.log(1 + math.exp(approx[i])))

            return error_sum, weight_sum

    train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)

    model = CatBoostClassifier(iterations=5, use_best_model=True, eval_metric=LoglossMetric())
    model.fit(train_pool, eval_set=test_pool)
    pred1 = model.predict(test_pool)

    model2 = CatBoostClassifier(iterations=5, use_best_model=True, eval_metric='Logloss')
    model2.fit(train_pool, eval_set=test_pool)
    pred2 = model2.predict(test_pool)

    assert np.all(pred1 == pred2)


class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = []
        for index in range(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in range(len(targets)):
            p = exponents[index] / (1 + exponents[index])
            der1 = (1 - p) if targets[index] > 0.0 else -p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result


class LoglossObjectiveNumpy(object):
    def __init__(self):
        self._objective = LoglossObjective()

    def calc_ders_range(self, approxes, targets, weights):
        return np.array(self._objective.calc_ders_range(approxes, targets, weights))


class LoglossObjectiveNumpy32(object):
    def __init__(self):
        self._objective = LoglossObjective()

    def calc_ders_range(self, approxes, targets, weights):
        return np.array(self._objective.calc_ders_range(approxes, targets, weights), dtype=np.float32)


@pytest.mark.parametrize('loss_objective', [LoglossObjective, LoglossObjectiveNumpy, LoglossObjectiveNumpy32])
@fails_on_gpu(how='User defined loss functions, metrics and callbacks are not supported for GPU')
def test_custom_objective(task_type, loss_objective):

    train_pool = Pool(data=TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(data=TEST_FILE, column_description=CD_FILE)

    model = CatBoostClassifier(iterations=5, learning_rate=0.03, use_best_model=True,
                               loss_function=loss_objective(), eval_metric="Logloss",
                               # Leaf estimation method and gradient iteration are set to match
                               # defaults for Logloss.
                               leaf_estimation_method="Newton", leaf_estimation_iterations=1, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    pred1 = model.predict(test_pool, prediction_type='RawFormulaVal')

    model2 = CatBoostClassifier(iterations=5, learning_rate=0.03, use_best_model=True, loss_function="Logloss", leaf_estimation_method="Newton", leaf_estimation_iterations=1)
    model2.fit(train_pool, eval_set=test_pool)
    pred2 = model2.predict(test_pool, prediction_type='RawFormulaVal')

    for p1, p2 in zip(pred1, pred2):
        assert abs(p1 - p2) < EPS


@fails_on_gpu(how='User defined loss functions, metrics and callbacks are not supported for GPU')
def test_multilabel_custom_objective(task_type, n=10):
    class MultiRMSEObjective(MultiTargetCustomObjective):
        def calc_ders_multi(self, approxes, targets, weight):
            assert len(approxes) == len(targets)

            grad = []
            hess = [[0 for j in range(len(targets))] for i in range(len(targets))]

            for index in range(len(targets)):
                der1 = (targets[index] - approxes[index]) * weight
                der2 = -weight

                grad.append(der1)
                hess[index][index] = der2

            return (grad, hess)

    xs = np.arange(n).reshape((-1, 1)).astype(np.float32)
    ys = np.hstack([
        (xs > 0.5 * n),
        (xs < 0.5 * n)
    ]).astype(np.float32)

    train_pool = Pool(data=xs,
                      label=ys)

    test_pool = Pool(data=xs,
                     label=ys)

    model = CatBoostRegressor(
        iterations=5,
        learning_rate=0.03,
        use_best_model=True,
        loss_function=MultiRMSEObjective(),
        eval_metric="MultiRMSE",
        # Leaf estimation method and gradient iteration are set to match
        # defaults for Logloss.
        leaf_estimation_method="Newton",
        leaf_estimation_iterations=1,
        task_type=task_type,
        devices='0'
    )

    model.fit(train_pool, eval_set=test_pool)
    pred1 = model.predict(test_pool, prediction_type='RawFormulaVal')

    model2 = CatBoostRegressor(
        iterations=5,
        learning_rate=0.03,
        boost_from_average=False,
        use_best_model=True,
        loss_function="MultiRMSE",
        leaf_estimation_method="Newton",
        leaf_estimation_iterations=1
    )
    model2.fit(train_pool, eval_set=test_pool)
    pred2 = model2.predict(test_pool, prediction_type='RawFormulaVal')

    for p1, p2 in zip(pred1, pred2):
        assert (abs(p1 - p2) < EPS).all()


def test_pool_after_fit(task_type):
    pool1 = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool2 = Pool(TRAIN_FILE, column_description=CD_FILE)
    assert _have_equal_features(pool1, pool2)
    model = CatBoostClassifier(iterations=5, task_type=task_type, devices='0')
    model.fit(pool2)
    assert _have_equal_features(pool1, pool2)


def test_priors(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        has_time=True,
        ctr_description=["Borders:Prior=0:Prior=0.6:Prior=1:Prior=5",
                         ("FeatureFreq" if task_type == 'GPU' else "Counter") + ":Prior=0:Prior=0.6:Prior=1:Prior=5"],
        task_type=task_type, devices='0',
    )
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_ignored_features(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model1 = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', max_ctr_complexity=1)
    model1.fit(train_pool)
    fstr = model1.get_feature_importance()
    model2 = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', max_ctr_complexity=1, ignored_features=np.argsort(fstr)[-2:])
    model2.fit(train_pool)
    predictions1 = model1.predict_proba(test_pool)
    predictions2 = model2.predict_proba(test_pool)
    assert not _check_data(predictions1, predictions2)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model2.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_multi_reload_model(task_type):
    friday_train_pool = Pool(data=BLACK_FRIDAY_TRAIN_FILE, column_description=BLACK_FRIDAY_CD_FILE, has_header=True)
    friday_params = dict(
        iterations=20,
        learning_rate=0.5,
        task_type=task_type,
        devices='0',
        max_ctr_complexity=1,
        target_border=5000,
    )
    friday_model = CatBoostClassifier(**friday_params)
    friday_model.fit(friday_train_pool)
    friday_model_path = test_output_path('friday_' + OUTPUT_MODEL_PATH)
    friday_model.save_model(friday_model_path)

    adult_train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    adult_model = CatBoost({'iterations': 2, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    adult_model.fit(adult_train_pool)
    adult_model_path = test_output_path('adult_' + OUTPUT_MODEL_PATH)
    adult_model.save_model(adult_model_path)

    model = CatBoost()
    model.load_model(friday_model_path)
    model.predict(friday_train_pool)
    model.load_model(adult_model_path)
    model.predict(adult_train_pool)
    model.load_model(friday_model_path)
    model.predict(friday_train_pool)
    model.load_model(adult_model_path)
    model.predict(adult_train_pool)


def test_ignored_features_names(task_type):
    train_pool = Pool(data=BLACK_FRIDAY_TRAIN_FILE, column_description=BLACK_FRIDAY_CD_FILE, has_header=True)
    test_pool = Pool(data=BLACK_FRIDAY_TEST_FILE, column_description=BLACK_FRIDAY_CD_FILE, has_header=True)
    params = dict(
        iterations=20,
        learning_rate=0.5,
        task_type=task_type,
        devices='0',
        max_ctr_complexity=1,
        target_border=5000,
    )
    model2 = CatBoostClassifier(**params)
    params.update(dict(ignored_features=['Stay_In_Current_City_Years', 'Gender']))
    model1 = CatBoostClassifier(**params)
    model1.fit(train_pool)
    model2.fit(train_pool)
    predictions1 = model1.predict_proba(test_pool)
    predictions2 = model2.predict_proba(test_pool)
    assert not _check_data(predictions1, predictions2)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model1.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_class_weights_list_binclass(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, class_weights=[1, 2], task_type=task_type, devices='0')
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_auto_class_weights():
    rnd = np.random.RandomState(2020)
    data = rnd.rand(10, 2)
    target = rnd.randint(2, size=10)

    counts = Counter(target)
    max_class_count = max(counts.values())
    expected_balanced = np.array([max_class_count / counts[class_id] if counts[class_id] > 0 else 1.
                                  for class_id in range(2)])
    expected_sqrt_balanced = np.sqrt(expected_balanced)

    balanced_model = CatBoostClassifier(iterations=5, auto_class_weights='Balanced')
    balanced_model.fit(data, target)

    balanced_params = balanced_model.get_all_params()
    assert 'class_weights' in balanced_params and _check_data(balanced_params['class_weights'],
                                                              expected_balanced)
    assert 'class_weights' not in balanced_model.get_params()

    sqrt_balanced_model = CatBoostClassifier(iterations=5, auto_class_weights='SqrtBalanced')
    sqrt_balanced_model.fit(data, target)
    sqrt_balanced_params = sqrt_balanced_model.get_all_params()
    assert 'class_weights' in sqrt_balanced_params and _check_data(sqrt_balanced_params['class_weights'],
                                                                   expected_sqrt_balanced)
    assert 'class_weights' not in sqrt_balanced_model.get_params()


@pytest.mark.parametrize(
    'dict_type',
    [dict, OrderedDict],
    ids=['label_type=' + val for val in ['dict', 'OrderedDict']]
)
@pytest.mark.parametrize(
    'label_type',
    [int, str],
    ids=['label_type=' + val for val in ['int', 'str']]
)
def test_class_weights_dict_binclass(dict_type, label_type):
    if label_type is int:
        uniq_labels = [0, 1]
    elif label_type is str:
        uniq_labels = ['a', 'b']

    features, labels = generate_random_labeled_dataset(100, 5, uniq_labels)

    weights = [0.5, 2.0]
    class_weights_dict = dict_type(zip(uniq_labels, weights))

    model_with_class_weights_as_list = CatBoostClassifier(
        iterations=5,
        class_names=uniq_labels,
        class_weights=weights
    )
    model_with_class_weights_as_list.fit(features, labels)

    model_with_class_weights_as_dict = CatBoostClassifier(
        iterations=5,
        class_weights=class_weights_dict
    )
    model_with_class_weights_as_dict.fit(features, labels)

    assert model_with_class_weights_as_list == model_with_class_weights_as_dict


@pytest.mark.parametrize(
    'label_type',
    [int, str],
    ids=['label_type=' + val for val in ['int', 'str']]
)
def test_bad_class_weights_dict_binclass(label_type):
    if label_type is int:
        uniq_labels = [0, 1, 2]
    elif label_type is str:
        uniq_labels = ['a', 'b', 'c']

    features, labels = generate_random_labeled_dataset(100, 5, uniq_labels[:2])

    model = CatBoostClassifier(
        class_weights={uniq_labels[0]: 0.5, uniq_labels[1]: 2.0, uniq_labels[2]: 0.3}
    )
    with pytest.raises(CatBoostError):
        model.fit(features, labels)


@pytest.mark.parametrize(
    'dict_type',
    [dict, OrderedDict],
    ids=['label_type=' + val for val in ['dict', 'OrderedDict']]
)
@pytest.mark.parametrize(
    'label_type',
    [int, str],
    ids=['label_type=' + val for val in ['int', 'str']]
)
def test_class_weights_dict_multiclass(dict_type, label_type):
    if label_type is int:
        uniq_labels = [0, 1, 2, 3]
    elif label_type is str:
        uniq_labels = ['a', 'b', 'c', 'd']

    features, labels = generate_random_labeled_dataset(100, 5, uniq_labels)

    weights = [0.1, 2.0, 3.0, 0.4]
    class_weights_dict = dict_type(zip(uniq_labels, weights))

    model_with_class_weights_as_list = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=5,
        class_names=uniq_labels,
        class_weights=weights
    )
    model_with_class_weights_as_list.fit(features, labels)

    model_with_class_weights_as_dict = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=5,
        class_weights=class_weights_dict
    )
    model_with_class_weights_as_dict.fit(features, labels)

    assert model_with_class_weights_as_list == model_with_class_weights_as_dict


@pytest.mark.parametrize(
    'label_type',
    [int, str],
    ids=['label_type=' + val for val in ['int', 'str']]
)
def test_bad_class_weights_dict_multiclass(label_type):
    if label_type is int:
        uniq_labels = [0, 1, 2, 3]
    elif label_type is str:
        uniq_labels = ['a', 'b', 'c', 'd']

    features, labels = generate_random_labeled_dataset(100, 5, uniq_labels)

    model = CatBoostClassifier(
        loss_function='MultiClass',
        class_weights={uniq_labels[0]: 0.5, uniq_labels[1]: 2.0, uniq_labels[2]: 0.3}
    )
    with pytest.raises(CatBoostError):
        model.fit(features, labels)


def test_classification_ctr(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03,
                               ctr_description=['Borders', 'FeatureFreq' if task_type == 'GPU' else 'Counter'],
                               task_type=task_type, devices='0')
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


@fails_on_gpu(how="private/libs/options/catboost_options.cpp:280: Error: GPU doesn't not support target binarization per CTR description currently. Please use ctr_target_border_count option instead")
def test_regression_ctr(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=5, learning_rate=0.03, ctr_description=['Borders:TargetBorderCount=5:TargetBorderType=Uniform', 'Counter'], task_type=task_type, devices='0')
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_ctr_target_border_count(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=5, learning_rate=0.03, ctr_target_border_count=5, task_type=task_type, devices='0')
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_copy_model():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model1 = CatBoostRegressor(iterations=5)
    model1.fit(pool)
    model2 = model1.copy()
    predictions1 = model1.predict(pool)
    predictions2 = model2.predict(pool)
    assert _check_data(predictions1, predictions2)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model2.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_cv(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
        },
    )
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_query(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {"iterations": 20, "learning_rate": 0.03, "loss_function": "QueryRMSE", "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')},
    )
    assert "train-QueryRMSE-mean" in results

    prev_value = results["train-QueryRMSE-mean"][0]
    for value in results["train-QueryRMSE-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_pairs(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "random_seed": 8,
            "loss_function": "PairLogit",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
        },
    )
    assert "train-PairLogit-mean" in results

    prev_value = results["train-PairLogit-mean"][0]
    for value in results["train-PairLogit-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_pairs_generated(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 10,
            "learning_rate": 0.03,
            "random_seed": 8,
            "loss_function": "PairLogit",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
        },
    )
    assert "train-PairLogit-mean" in results

    prev_value = results["train-PairLogit-mean"][0]
    for value in results["train-PairLogit-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_custom_loss(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 5,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "custom_loss": "AUC",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
        }
    )
    assert "test-AUC-mean" in results
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_skip_train(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss:hints=skip_train~true",
            "eval_metric": "AUC",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
        },
    )
    assert "train-Logloss-mean" not in results
    assert "train-Logloss-std" not in results
    assert "train-AUC-mean" not in results
    assert "train-AUC-std" not in results

    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_skip_train_default(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "custom_loss": "AUC",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
        },
    )
    assert "train-AUC-mean" not in results
    assert "train-AUC-std" not in results

    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_metric_period(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
        },
        metric_period=5,
    )
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


@pytest.mark.parametrize(
    'with_metric_period',
    [False, True],
    ids=['with_metric_period=' + val for val in ['False', 'True']]
)
def test_cv_overfitting_detector(with_metric_period, task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
        },
        metric_period=5 if with_metric_period else None,
        early_stopping_rounds=7,
    )
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:-1]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


@pytest.mark.parametrize('param_type', ['indices', 'strings'])
def test_cv_with_cat_features_param(param_type):
    if param_type == 'indices':
        cat_features_param = [1, 2]
        feature_names_param = None
    else:
        cat_features_param = ['feat1', 'feat2']
        feature_names_param = ['feat' + str(i) for i in range(20)]

    prng = np.random.RandomState(seed=20181219)
    data = prng.randint(10, size=(20, 20))
    label = _generate_nontrivial_binary_target(20, prng=prng)
    pool = Pool(data, label, cat_features=cat_features_param, feature_names=feature_names_param)

    train_dir_prefix = test_output_path('')
    params = {
        'loss_function': 'Logloss',
        'iterations': 10,
        'train_dir': os.path.join(train_dir_prefix, 'catboost_info')
    }

    results1 = cv(pool, params, as_pandas=False)
    params_with_cat_features = params.copy()
    params_with_cat_features['cat_features'] = cat_features_param
    results2 = cv(pool, params_with_cat_features, as_pandas=False)
    assert results1 == results2

    params_with_wrong_cat_features = params.copy()
    params_with_wrong_cat_features['cat_features'] = [0, 2] if param_type == 'indices' else ['feat0', 'feat2']
    with pytest.raises(CatBoostError):
        cv(pool, params_with_wrong_cat_features)


@pytest.mark.parametrize('problem_type', ['binclass', 'multiclass', 'regression'])
def test_cv_with_text(problem_type):
    cats_words = ['Meow', 'Kitten', 'Paw', 'Tail', 'Purring', 'Crouch', 'Whisker']
    dogs_words = ['Puppy', 'Whelp', 'Woof', 'Tail', 'Paw', 'Snarl', 'Barking']
    words = [cats_words, dogs_words]
    np.random.seed(1)
    labels = np.random.choice(2, 1000)
    texts = [[' '.join(np.random.choice(words[label], 3, replace=False))] for label in labels]
    data_pool = Pool(data=texts, label=labels, text_features=[0])
    train_dir_prefix = test_output_path('')
    params = {
        'loss_function':  {
            'binclass': 'Logloss',
            'multiclass': 'MultiClass',
            'regression': 'RMSE'
        }[problem_type],
        'iterations': 10,
        'random_seed': 42,
        'learning_rate': 0.5,
        'train_dir': os.path.join(train_dir_prefix, 'catboost_info')
    }
    result = cv(iterations=10, pool=data_pool, params=params, fold_count=3).round(decimals=3)

    preds_path = test_output_path(CV_CSV_PATH)
    result.to_csv(preds_path)
    return local_canonical_file(preds_path, diff_tool=get_limited_precision_json_diff_tool(1.e-6))


def test_cv_with_save_snapshot(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    with pytest.raises(CatBoostError):
        cv(
            pool,
            {
                "iterations": 20,
                "learning_rate": 0.03,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "task_type": task_type,
                "save_snapshot": True,
                "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
            },
        )


@pytest.mark.parametrize("params", [
    {"loss_function": "Logloss", "eval_metric": "AUC"},
    {"loss_function": "RMSE", "eval_metric": "R2"},
])
def test_cv_return_models(params):
    train_dir_prefix = test_output_path('')
    params.update(
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
        }
    )
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    fold_count = 5
    results, cv_models = cv(
        train_pool,
        params,
        fold_count=fold_count,
        return_models=True,
    )

    assert len(cv_models) == fold_count

    single_model = CatBoost(params).fit(train_pool)
    single_model_prediction = single_model.predict(test_pool, prediction_type="RawFormulaVal")
    single_model_metric = eval_metric(test_pool.get_label(), single_model_prediction, params["eval_metric"])

    for model in cv_models:
        assert isinstance(model, CatBoost)
        assert model.random_seed_ == single_model.random_seed_
        assert model.tree_count_ == params["iterations"]

        model_prediction = model.predict(test_pool, prediction_type="RawFormulaVal")
        cv_model_metric = eval_metric(test_pool.get_label(), model_prediction, params["eval_metric"])
        assert np.allclose(
            cv_model_metric,
            single_model_metric,
            atol=results["test-{}-std".format(params["eval_metric"])][0]
        )

        # Other methods
        model.calc_leaf_indexes(test_pool)
        assert model.get_all_params().keys() == single_model.get_all_params().keys()
        assert not np.isnan(model.get_feature_importance(test_pool)).any()
        assert not np.isnan(model.get_scale_and_bias()).any()
        model = model.copy()
        try:
            model.compare(single_model, test_pool, [params["eval_metric"]])
        except ImportError as ie:
            pytest.xfail(str(ie)) if str(ie) == "No module named widget" \
                else pytest.fail(str(ie))
        del model


def test_cv_small_data():
    cv_data = [["France", 1924, 44],
               ["USA", 1932, 37],
               ["Switzerland", 1928, 25],
               ["Norway", 1952, 30],
               ["Japan", 1972, 35],
               ["Mexico", 1968, 112]]
    labels = [1, 1, 0, 0, 0, 1]
    pool = Pool(data=cv_data,
                label=labels,
                cat_features=[0])
    train_dir_prefix = test_output_path('')
    params = {
        "iterations": 100,
        "depth": 2,
        "loss_function": "Logloss",
        "verbose": False,
        "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
    }
    cv(pool, params, fold_count=2)


def test_cv_without_loss_function():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    with pytest.raises(CatBoostError):
        cv(pool, {})
        pytest.fail("Parameter loss_function should be specified for cross-validation")


def test_tune_hyperparams_small_data():
    train_data = [[1, 4, 5, 6],
                  [4, 5, 6, 7],
                  [30, 40, 50, 60],
                  [20, 30, 70, 60],
                  [10, 80, 40, 30],
                  [10, 10, 20, 30]]
    train_labels = [10, 20, 30, 15, 10, 25]
    grid = {
        'learning_rate': [0.03, 0.1],
        'depth': [4, 6],
        'l2_leaf_reg': [1, 9],
        'iterations': [10]
    }
    CatBoost().randomized_search(grid, X=train_data, y=train_labels, search_by_train_test_split=True)
    CatBoost().randomized_search(grid, X=train_data, y=train_labels, search_by_train_test_split=False)
    CatBoost().grid_search(grid, X=train_data, y=train_labels, search_by_train_test_split=True)
    CatBoost().grid_search(grid, X=train_data, y=train_labels, search_by_train_test_split=False)


def test_grid_search_aliases(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "task_type": task_type,
            "iterations": 10,
        }
    )
    grid = {
        "num_boost_round": [5, 10],
        "eta": [0.03, 0.1],
        "random_state": [21, 42],
        "reg_lambda": [3.0, 6.6],
        "max_depth": [4, 8],
        "colsample_bylevel": [0.5, 1],
        "max_bin": [8, 28]
    }
    results = model.grid_search(
        grid,
        pool
    )
    for key, value in results["params"].items():
        assert value in grid[key]


def test_grid_search_and_get_best_result(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    for refit in [True, False]:
        for search_by_train_test_split in [True, False]:
            model = CatBoost(
                {
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "task_type": task_type,
                    "custom_metric": ["CrossEntropy", "F1", "F:beta=2"]
                }
            )
            feature_border_type_list = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']
            one_hot_max_size_list = [4, 7, 10]
            iterations_list = [5, 7, 10]
            border_count_list = [4, 10, 50, 100]
            model.grid_search(
                {
                    'feature_border_type': feature_border_type_list,
                    'one_hot_max_size': one_hot_max_size_list,
                    'iterations': iterations_list,
                    'border_count': border_count_list
                },
                pool,
                refit=refit,
                search_by_train_test_split=search_by_train_test_split
            )
            best_scores = model.get_best_score()
            if refit:
                assert 'validation' not in best_scores, 'validation results found for refit=True'
                assert 'learn' in best_scores, 'no train results found for refit=True'
            elif search_by_train_test_split:
                assert 'validation' in best_scores, 'no validation results found for refit=False, search_by_train_test_split=True'
                assert 'learn' in best_scores, 'no train results found for refit=False, search_by_train_test_split=True'
            else:
                assert 'validation' not in best_scores, 'validation results found for refit=False, search_by_train_test_split=False'
                assert 'learn' not in best_scores, 'train results found for refit=False, search_by_train_test_split=False'
            if 'validation' in best_scores:
                for metric in ["AUC", "Logloss", "CrossEntropy", "F1", "F:beta=2"]:
                    assert metric in best_scores['validation'], 'no validation ' + metric + ' results found'
            if 'learn' in best_scores:
                for metric in ["Logloss", "CrossEntropy", "F1", "F:beta=2"]:
                    assert metric in best_scores['learn'], 'no train ' + metric + ' results found'
                assert "AUC" not in best_scores['learn'], 'train AUC results found'


def test_grid_search(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
        }
    )
    feature_border_type_list = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']
    one_hot_max_size_list = [4, 7, 10]
    iterations_list = [5, 7, 10]
    border_count_list = [4, 10, 50, 100]
    results = model.grid_search(
        {
            'feature_border_type': feature_border_type_list,
            'one_hot_max_size': one_hot_max_size_list,
            'iterations': iterations_list,
            'border_count': border_count_list
        },
        pool
    )
    assert "train-Logloss-mean" in results['cv_results'], '"train-Logloss-mean" not in results'

    prev_value = results['cv_results']["train-Logloss-mean"][0]
    for value in results['cv_results']["train-Logloss-mean"][1:]:
        assert value < prev_value, 'not monotonous Logloss-mean'
        prev_value = value

    assert 'feature_border_type' in results['params'], '"feature_border_type" not in results'
    assert results['params']['feature_border_type'] in feature_border_type_list, "wrong 'feature_border_type_list' value"
    assert 'one_hot_max_size' in results['params'], '"one_hot_max_size" not in results'
    assert results['params']['one_hot_max_size'] in one_hot_max_size_list, "wrong 'one_hot_max_size_list' value"
    assert 'iterations' in results['params'], '"iterations" not in results'
    assert results['params']['iterations'] in iterations_list, "wrong 'iterations_list' value"
    assert 'border_count' in results['params'], '"border_count" not in results'
    assert results['params']['border_count'] in border_count_list, "wrong 'border_count_list' value"


def test_grid_search_for_multiclass():
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    model = CatBoostClassifier(iterations=10)
    grid = {
        'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }

    results = model.grid_search(grid, pool, shuffle=False, verbose=False)
    for key, value in results["params"].items():
        assert value in grid[key]


def test_randomized_search(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type
        }
    )
    feature_border_type_list = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']
    one_hot_max_size_list = [4, 7, 10]
    iterations_list = [5, 7, 10]
    border_count_list = [4, 10, 50, 100]
    results = model.randomized_search(
        {
            'feature_border_type': feature_border_type_list,
            'one_hot_max_size': one_hot_max_size_list,
            'iterations': iterations_list,
            'border_count': border_count_list
        },
        pool
    )
    assert "train-Logloss-mean" in results['cv_results'], '"train-Logloss-mean" not in results'

    prev_value = results['cv_results']["train-Logloss-mean"][0]
    for value in results['cv_results']["train-Logloss-mean"][1:]:
        assert value < prev_value, 'not monotonic Logloss-mean'
        prev_value = value

    assert results['params'].get('feature_border_type') in feature_border_type_list
    assert results['params'].get('one_hot_max_size') in one_hot_max_size_list
    assert results['params'].get('iterations') in iterations_list
    assert results['params'].get('border_count') in border_count_list


def test_randomized_search_only_dist(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type
        }
    )

    class UniformChoice:

        def __init__(self, values):
            self.values = values

        def rvs(self):
            return np.random.choice(self.values)

    feature_border_type_list = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']
    results = model.randomized_search(
        {
            'feature_border_type': feature_border_type_list,
            'one_hot_max_size': UniformChoice(list(range(20))),
            'iterations': UniformChoice([4, 7, 9]),
            'border_count': UniformChoice([10, 6, 20, 4])
        },
        pool
    )
    assert results['params']['feature_border_type'] in feature_border_type_list
    assert results['params']['one_hot_max_size'] in range(20)
    assert results['params']['border_count'] in [10, 6, 20, 4]
    assert results['params']['iterations'] in [4, 7, 9]


def test_randomized_search_refit_model(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type
        }
    )

    class UniformChoice:

        def __init__(self, values):
            self.values = values

        def rvs(self):
            return np.random.choice(self.values)

    feature_border_type_list = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']
    model.randomized_search(
        {
            'feature_border_type': feature_border_type_list,
            'one_hot_max_size': UniformChoice(list(range(20))),
            'iterations': UniformChoice([4, 7, 9]),
            'border_count': UniformChoice([10, 6, 20, 4])
        },
        pool,
        refit=True
    )

    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model.predict(test_pool)


def test_randomized_search_cv(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type
        }
    )

    class UniformChoice:

        def __init__(self, values):
            self.values = values

        def rvs(self):
            return np.random.choice(self.values)

    feature_border_type_list = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']
    results = model.randomized_search(
        {
            'feature_border_type': feature_border_type_list,
            'one_hot_max_size': UniformChoice(list(range(20))),
            'iterations': UniformChoice([1, 2, 3]),
            'border_count': UniformChoice([10, 6, 20, 4])
        },
        pool,
        n_iter=2,
        search_by_train_test_split=False
    )
    assert results['params']['feature_border_type'] in feature_border_type_list
    assert results['params']['one_hot_max_size'] in range(20)
    assert results['params']['border_count'] in [10, 6, 20, 4]
    assert results['params']['iterations'] in [1, 2, 3]


def test_grid_search_with_class_weights_lists():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10)
    grid = {
        'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 7],
        'class_weights': [[1, 2], [1, 3]]
    }

    results = model.grid_search(grid, pool, shuffle=False, verbose=False)
    for key, value in results["params"].items():
        assert value in grid[key]


def test_grid_search_wrong_param_type(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
        }
    )
    feature_border_type_list = ['Median', 12, 'UniformAndQuantiles', 'MaxLogSum']
    one_hot_max_size_list = [4, 7, '10']
    try:
        model.grid_search(
            {
                'feature_border_type': feature_border_type_list,
                'one_hot_max_size': one_hot_max_size_list
            },
            pool
        )
    except CatBoostError:
        return
    assert False


def test_grid_search_trivial(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
        }
    )
    feature_border_type_list = ['Median']
    one_hot_max_size_list = [4]
    iterations_list = [30]
    results = model.grid_search(
        {
            'feature_border_type': feature_border_type_list,
            'one_hot_max_size': one_hot_max_size_list,
            'iterations': iterations_list
        },
        pool
    )
    assert 'feature_border_type' in results['params']
    assert results['params']['feature_border_type'] == 'Median'
    assert 'one_hot_max_size' in results['params']
    assert results['params']['one_hot_max_size'] == 4
    assert 'iterations' in results['params']
    assert results['params']['iterations'] == 30


def test_grid_search_several_grids(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "iterations": 10,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
        }
    )
    grids = []
    grids.append(
        {
            'feature_border_type': ['Median', 'Uniform'],
            'one_hot_max_size': [4, 6],
            'iterations': [10],
            'border_count': [4, 10]
        }
    )
    grids.append(
        {
            'feature_border_type': ['UniformAndQuantiles', 'MaxLogSum'],
            'one_hot_max_size': [8, 10],
            'iterations': [20],
            'border_count': [50, 100]
        }
    )
    results = model.grid_search(
        grids,
        pool
    )
    grid_num = 0 if results['params']['feature_border_type'] in grids[0]['feature_border_type'] else 1
    assert results['params']['feature_border_type'] in grids[grid_num]['feature_border_type']
    assert results['params']['one_hot_max_size'] in grids[grid_num]['one_hot_max_size']
    assert results['params']['iterations'] in grids[grid_num]['iterations']
    assert results['params']['border_count'] in grids[grid_num]['border_count']


def test_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool_querywise = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)

    model = CatBoostRanker(iterations=5, learning_rate=0.03, task_type=task_type, devices="0", loss_function="QueryRMSE")
    model.fit(pool_querywise)

    assert len(model.feature_importances_.shape) == 0
    model.get_feature_importance(type=EFstrType.LossFunctionChange, data=pool_querywise)

    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(pool)
    assert (model.get_feature_importance() == model.get_feature_importance(type=EFstrType.PredictionValuesChange)).all()
    failed = False
    try:
        model.get_feature_importance(type=EFstrType.LossFunctionChange)
    except CatBoostError:
        failed = True
    assert failed
    np.save(fimp_npy_path, np.array(model.feature_importances_))
    assert len(model.feature_importances_.shape)
    return local_canonical_file(fimp_npy_path)


def test_feature_importance_interaction_asymmetric_grow_policy():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        max_ctr_complexity=3)
    model.fit(pool)

    fstr_symm = np.array(model.get_feature_importance(type='Interaction', data=pool))
    model._convert_to_asymmetric_representation()
    fstr_asymm = np.array(model.get_feature_importance(type='Interaction', data=pool))

    assert np.all(fstr_symm - fstr_asymm < 1e-8)


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_feature_importance_asymmetric_prediction_value_change(task_type, grow_policy):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool_querywise = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)

    params = {
        "iterations": 5,
        "learning_rate": 0.03,
        "task_type": task_type,
        "devices": "0",
        "loss_function": "QueryRMSE",
        "grow_policy": grow_policy
    }
    model = CatBoostRanker(**params)
    model.fit(pool_querywise)
    assert len(model.feature_importances_.shape) == 0
    params["loss_function"] = "RMSE"
    model = CatBoostRegressor(**params)
    model.fit(pool)
    assert (model.get_feature_importance() == model.get_feature_importance(type=EFstrType.PredictionValuesChange)).all()
    np.save(fimp_npy_path, np.array(model.feature_importances_))
    assert len(model.feature_importances_.shape)
    return local_canonical_file(fimp_npy_path)


def test_feature_importance_explicit(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.array(model.get_feature_importance(type=EFstrType.PredictionValuesChange)))
    return local_canonical_file(fimp_npy_path)


def test_feature_importance_prettified(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(pool)

    feature_importances = model.get_feature_importance(type=EFstrType.PredictionValuesChange, prettified=True)
    fimp_txt_path = test_output_path(FIMP_TXT_PATH)
    with open(fimp_txt_path, 'w') as ofile:
        for f_id, f_imp in feature_importances.values:
            ofile.write('{}\t{}\n'.format(f_id, f_imp))
    return local_canonical_file(fimp_txt_path)


def test_interaction_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.array(model.get_feature_importance(type=EFstrType.Interaction)))
    return local_canonical_file(fimp_npy_path)


def make_reference_data(pool, calc_shap_mode):
    reference_data = None
    if calc_shap_mode == "IndependentTreeSHAP":
        quarter_size = pool.num_row() // 4
        reference_data = pool.slice(list(range(quarter_size)))
    return reference_data


@pytest.mark.parametrize('calc_shap_mode', ['TreeSHAP', 'IndependentTreeSHAP'])
def test_shap_feature_importance(task_type, calc_shap_mode):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    reference_data = make_reference_data(pool, calc_shap_mode)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, max_ctr_complexity=1, task_type=task_type, devices='0')
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data)
    assert np.allclose(model.predict(pool, prediction_type='RawFormulaVal'), np.sum(shaps, axis=1))

    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(shaps), 9))
    return local_canonical_file(fimp_npy_path)


def test_approximate_shap_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, max_ctr_complexity=1, task_type=task_type, devices='0')
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, shap_calc_type="Approximate")
    assert np.allclose(model.predict(pool, prediction_type='RawFormulaVal'), np.sum(shaps, axis=1))

    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(shaps), 9))
    return local_canonical_file(fimp_npy_path)


def test_exact_shap_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, max_ctr_complexity=1, task_type=task_type, devices='0')
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, shap_calc_type="Exact")
    assert np.allclose(model.predict(pool, prediction_type='RawFormulaVal'), np.sum(shaps, axis=1))

    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(shaps), 9))
    return local_canonical_file(fimp_npy_path)


@pytest.mark.parametrize('calc_shap_mode', ['TreeSHAP', 'IndependentTreeSHAP'])
def test_shap_feature_importance_multiclass(task_type, calc_shap_mode):
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    reference_data = make_reference_data(pool, calc_shap_mode)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', loss_function='MultiClass')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data)), 9))
    return local_canonical_file(fimp_npy_path)


def test_approximate_shap_feature_importance_multiclass(task_type):
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', loss_function='MultiClass')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool,
                                                                           shap_calc_type="Approximate")), 9))
    return local_canonical_file(fimp_npy_path)


def test_exact_shap_feature_importance_multiclass(task_type):
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', loss_function='MultiClass')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool,
                                                                           shap_calc_type="Exact")), 9))
    return local_canonical_file(fimp_npy_path)


@pytest.mark.parametrize('calc_shap_mode', ['TreeSHAP', 'IndependentTreeSHAP'])
def test_shap_feature_importance_multirmse(task_type, calc_shap_mode):
    train_file = MULTIREGRESSION_TRAIN_FILE
    cd_file = MULTIREGRESSION_CD_FILE
    pool = Pool(train_file, column_description=cd_file)
    reference_data = make_reference_data(pool, calc_shap_mode)
    model = CatBoostRegressor(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', loss_function='MultiRMSE')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data)), 9))
    return local_canonical_file(fimp_npy_path)


def test_approximate_shap_feature_importance_multirmse(task_type):
    train_file = MULTIREGRESSION_TRAIN_FILE
    cd_file = MULTIREGRESSION_CD_FILE
    pool = Pool(train_file, column_description=cd_file)
    model = CatBoostRegressor(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', loss_function='MultiRMSE')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool,
                                                                           shap_calc_type="Approximate")), 9))
    return local_canonical_file(fimp_npy_path)


def test_exact_shap_feature_importance_multirmse(task_type):
    train_file = MULTIREGRESSION_TRAIN_FILE
    cd_file = MULTIREGRESSION_CD_FILE
    pool = Pool(train_file, column_description=cd_file)
    model = CatBoostRegressor(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', loss_function='MultiRMSE')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool,
                                                                           shap_calc_type="Exact")), 9))
    return local_canonical_file(fimp_npy_path)


def test_shap_feature_importance_ranking(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE)
    model = CatBoostRanker(iterations=20, learning_rate=0.03, task_type=task_type, devices='0', loss_function='PairLogit')
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool)
    assert np.allclose(model.predict(pool), np.sum(shaps, axis=1))

    if task_type == 'GPU':
        return pytest.xfail(reason="On GPU models with loss Pairlogit are too unstable. MLTOOLS-4722")
    else:
        fimp_npy_path = test_output_path(FIMP_NPY_PATH)
        np.save(fimp_npy_path, np.around(np.array(shaps), 9))
        return local_canonical_file(fimp_npy_path)


def test_approximate_shap_feature_importance_ranking(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE)
    model = CatBoostRanker(iterations=20, learning_rate=0.03, task_type=task_type, devices='0', loss_function='PairLogit')
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, shap_calc_type="Approximate")
    assert np.allclose(model.predict(pool), np.sum(shaps, axis=1))

    if task_type == 'GPU':
        return pytest.xfail(reason="On GPU models with loss Pairlogit are too unstable. MLTOOLS-4722")
    else:
        fimp_npy_path = test_output_path(FIMP_NPY_PATH)
        np.save(fimp_npy_path, np.around(np.array(shaps), 9))
        return local_canonical_file(fimp_npy_path)


def test_exact_shap_feature_importance_ranking(task_type):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE)
    model = CatBoostRanker(iterations=20, learning_rate=0.03, task_type=task_type, devices='0', loss_function='PairLogit')
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, shap_calc_type="Exact")
    assert np.allclose(model.predict(pool), np.sum(shaps, axis=1))

    if task_type == 'GPU':
        return pytest.xfail(reason="On GPU models with loss Pairlogit are too unstable. MLTOOLS-4722")
    else:
        fimp_npy_path = test_output_path(FIMP_NPY_PATH)
        np.save(fimp_npy_path, np.around(np.array(shaps), 9))
        return local_canonical_file(fimp_npy_path)


def test_shap_feature_importance_asymmetric_and_symmetric(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        max_ctr_complexity=1,
        task_type=task_type,
        devices='0')
    model.fit(pool)
    shap_symm = np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool))
    model._convert_to_asymmetric_representation()
    shap_asymm = np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool))
    assert np.all(shap_symm - shap_asymm < 1e-8)


def test_approximate_shap_feature_importance_asymmetric_and_symmetric(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        max_ctr_complexity=1,
        task_type=task_type,
        devices='0')
    model.fit(pool)
    shap_symm = np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool,
                                                      shap_calc_type="Approximate"))
    model._convert_to_asymmetric_representation()
    shap_asymm = np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool,
                                                       shap_calc_type="Approximate"))
    assert np.all(shap_symm - shap_asymm < 1e-8)


@pytest.mark.parametrize('calc_shap_mode', ['TreeSHAP', 'IndependentTreeSHAP'])
def test_shap_feature_importance_with_langevin(calc_shap_mode):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    reference_data = make_reference_data(pool, calc_shap_mode)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, depth=10, langevin=True, diffusion_temperature=1000)
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data)
    assert np.allclose(model.predict(pool, prediction_type='RawFormulaVal'), np.sum(shaps, axis=1))

    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(shaps), 9))
    return local_canonical_file(fimp_npy_path)


def test_approximate_shap_feature_importance_with_langevin():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, depth=10, langevin=True, diffusion_temperature=1000)
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, shap_calc_type="Approximate")
    assert np.allclose(model.predict(pool, prediction_type='RawFormulaVal'), np.sum(shaps, axis=1))

    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(shaps), 9))
    return local_canonical_file(fimp_npy_path)


def test_exact_shap_feature_importance_with_langevin():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, depth=10, langevin=True, diffusion_temperature=1000)
    model.fit(pool)
    shaps = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, shap_calc_type="Exact")
    assert np.allclose(model.predict(pool, prediction_type='RawFormulaVal'), np.sum(shaps, axis=1))

    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(shaps), 9))
    return local_canonical_file(fimp_npy_path)


def test_loss_function_change_asymmetric_and_symmetric(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        max_ctr_complexity=1,
        task_type=task_type,
        devices='0')
    model.fit(pool)
    shap_symm = np.array(model.get_feature_importance(type=EFstrType.LossFunctionChange, data=pool))
    model._convert_to_asymmetric_representation()
    shap_asymm = np.array(model.get_feature_importance(type=EFstrType.LossFunctionChange, data=pool))
    assert np.all(shap_symm - shap_asymm < 1e-8)


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_shap_feature_importance_asymmetric(task_type, grow_policy):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        max_ctr_complexity=1,
        task_type=task_type,
        grow_policy=grow_policy,
        devices='0')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(type=EFstrType.ShapValues, data=pool)), 9))
    return local_canonical_file(fimp_npy_path)


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_loss_function_change_asymmetric(task_type, grow_policy):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        max_ctr_complexity=1,
        task_type=task_type,
        grow_policy=grow_policy,
        devices='0')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.array(model.get_feature_importance(type=EFstrType.LossFunctionChange, data=pool)))
    return local_canonical_file(fimp_npy_path)


@pytest.mark.parametrize('calc_shap_mode', ['TreeSHAP', 'IndependentTreeSHAP'])
def test_shap_feature_importance_modes(task_type, calc_shap_mode):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    reference_data = make_reference_data(pool, calc_shap_mode)
    model = CatBoostClassifier(iterations=5, task_type=task_type)
    model.fit(pool)
    modes = ["Auto", "UsePreCalc", "NoPreCalc"]
    shaps_for_modes = []
    for mode in modes:
        shaps_for_modes.append(model.get_feature_importance(type=EFstrType.ShapValues, data=pool, shap_mode=mode, reference_data=reference_data))
    for i in range(len(modes) - 1):
        assert np.all(np.abs(shaps_for_modes[i] - shaps_for_modes[i - 1]) < 1e-9)


def test_shap_feature_probability(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    reference_data = make_reference_data(pool, "IndependentTreeSHAP")
    model = CatBoostClassifier(iterations=50, task_type=task_type)
    model.fit(pool)
    shap_values = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data, model_output="Probability")
    predictions = model.predict(pool, "Probability")
    for doc_idx in range(len(shap_values)):
        assert abs(sum(shap_values[doc_idx]) - predictions[doc_idx][1]) < 1e-6


def test_shap_feature_multiclass_probability(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    reference_data = make_reference_data(pool, "IndependentTreeSHAP")
    model = CatBoostClassifier(iterations=50, loss_function='MultiClass', task_type=task_type)
    classes_count = 3
    model.fit(pool)
    shap_values = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data, model_output="Probability")
    predictions = model.predict(pool, "Probability")
    for doc_idx in range(len(shap_values)):
        for class_idx in range(classes_count):
            assert abs(sum(shap_values[doc_idx][class_idx]) - predictions[doc_idx][class_idx]) < 1e-6


def test_shap_feature_log_loss(task_type):
    def log_loss(yt, yp):
        return (-(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))

    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    reference_data = make_reference_data(pool, "IndependentTreeSHAP")
    model = CatBoostClassifier(iterations=50, loss_function='Logloss', task_type=task_type)
    model.fit(pool)
    label = pool.get_label()
    shap_values = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data, model_output="LossFunction")
    predictions = model.predict(pool, "Probability")
    for doc_idx in range(len(shap_values)):
        assert abs(sum(shap_values[doc_idx]) - log_loss(float(label[doc_idx]), float(predictions[doc_idx][1]))) < 1e-6


def test_shap_feature_rmse(task_type):
    def rmse(yt, yp):
        return np.absolute(yt - yp)

    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    reference_data = make_reference_data(train_pool, "IndependentTreeSHAP")
    model = CatBoostRegressor(iterations=10, loss_function='RMSE')
    model.fit(train_pool)
    label = test_pool.get_label()
    shap_values = model.get_feature_importance(type=EFstrType.ShapValues, data=test_pool, reference_data=reference_data, model_output="LossFunction")
    predictions = model.predict(test_pool)
    for doc_idx in range(len(shap_values)):
        assert abs(sum(shap_values[doc_idx]) - rmse(float(label[doc_idx]), float(predictions[doc_idx]))) < 1e-6


def test_feature_importance_sage_basic():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, devices='0')
    model.fit(pool)

    feature_importances = model.get_feature_importance(pool, type=EFstrType.SageValues, sage_n_samples=32,
                                                       sage_batch_size=512)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(feature_importances), 2))
    return local_canonical_file(fimp_npy_path)


def test_feature_importance_sage_querywise():
    pool_querywise = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)

    params = {
        "iterations": 5,
        "learning_rate": 0.03,
        "devices": "0",
        "loss_function": "QueryRMSE"
    }
    model = CatBoostRanker(**params)
    model.fit(pool_querywise)

    feature_importances = model.get_feature_importance(pool_querywise, type=EFstrType.SageValues)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(feature_importances), 2))
    return local_canonical_file(fimp_npy_path)


def test_feature_importance_sage_all_feature_types():
    pool = Pool(ROTTEN_TOMATOES_TRAIN_FILE, column_description=ROTTEN_TOMATOES_CD_BINCLASS_FILE)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)

    params = {
        "iterations": 5,
        "learning_rate": 0.03,
        "devices": "0"
    }
    model = CatBoostClassifier(**params)
    model.fit(pool)

    feature_importances = model.get_feature_importance(pool, type=EFstrType.SageValues)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(feature_importances), 2))
    return local_canonical_file(fimp_npy_path)


def test_prediction_diff_feature_importance(task_type):
    pool_file = 'higgs'
    pool = Pool(data_file(pool_file, 'train_small'), column_description=data_file(pool_file, 'train.cd'))
    model = CatBoostClassifier(iterations=110, task_type=task_type, learning_rate=0.03, max_ctr_complexity=1, devices='0')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.around(np.array(model.get_feature_importance(
        type=EFstrType.PredictionDiff,
        data=pool.get_features()[:2]
    )), 9))
    return local_canonical_file(fimp_npy_path)


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_prediction_diff_nonsym_feature_importance(task_type, grow_policy):
    pool_file = 'higgs'
    pool = Pool(data_file(pool_file, 'train_small'), column_description=data_file(pool_file, 'train.cd'))
    model = CatBoostClassifier(iterations=110, task_type=task_type, grow_policy=grow_policy, learning_rate=0.03, max_ctr_complexity=1, devices='0')
    model.fit(pool)
    fimp_txt_path = test_output_path(FIMP_TXT_PATH)
    np.savetxt(fimp_txt_path, np.around(np.array(model.get_feature_importance(
        type=EFstrType.PredictionDiff,
        data=pool.get_features()[:2]
    )), 9))
    return local_canonical_file(fimp_txt_path)


def test_od(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.03, od_type='Iter', od_wait=20, random_seed=42, task_type=task_type, devices='0')
    model.fit(train_pool, eval_set=test_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


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


@fails_on_gpu(how='private/libs/options/json_helper.h:198: Error: change of option approx_on_full_history is unimplemented for task type GPU and was not default in previous run')
def test_full_history(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, od_type='Iter', od_wait=20, random_seed=42,
        approx_on_full_history=True, task_type=task_type, devices='0', boosting_type='Ordered'
    )
    model.fit(train_pool, eval_set=test_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_cv_logging(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    cv(
        pool,
        {
            "iterations": 14,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
        },
    )
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_cv_with_not_binarized_target(task_type):
    train_file = data_file('adult_not_binarized', 'train_small')
    cd = data_file('adult_not_binarized', 'train.cd')
    pool = Pool(train_file, column_description=cd)
    train_dir_prefix = test_output_path('')
    cv(
        pool,
        {
            "iterations": 10,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "task_type": task_type,
            "target_border": 0.5,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
        },
    )
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


@pytest.mark.parametrize('loss_function', ['Logloss', 'RMSE', 'QueryRMSE'])
def test_eval_metrics(loss_function, task_type):
    train, test, cd, metric = TRAIN_FILE, TEST_FILE, CD_FILE, loss_function
    if loss_function == 'QueryRMSE':
        train, test, cd, metric = QUERYWISE_TRAIN_FILE, QUERYWISE_TEST_FILE, QUERYWISE_CD_FILE, 'PFound'
    if loss_function == 'Logloss':
        metric = 'AUC'

    train_pool = Pool(train, column_description=cd)
    test_pool = Pool(test, column_description=cd)
    model = CatBoost(params={'loss_function': loss_function, 'iterations': 20, 'thread_count': 8, 'eval_metric': metric,
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
    model = CatBoost(params={'loss_function': loss_function, 'iterations': 100, 'thread_count': 8,
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


@pytest.mark.parametrize('verbose', [5, False, True])
def test_verbose_int(verbose, task_type):
    expected_line_count = {5: 3, False: 0, True: 10}
    expected_cv_line_count = {5: 24, False: 15, True: 45}
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile = 'test_data_dumps'

    with open(tmpfile, 'w') as cout:
        cv(
            pool,
            {"iterations": 10, "learning_rate": 0.03, "loss_function": "Logloss", "task_type": task_type},
            verbose=verbose,
            log_cout=cout,
        )
    assert (_count_lines(tmpfile) == expected_cv_line_count[verbose])

    with open(tmpfile, 'w') as cout:
        train(pool, {"iterations": 10, "learning_rate": 0.03, "loss_function": "Logloss", "task_type": task_type, "devices": '0'},
              verbose=verbose, log_cout=cout)
    assert (_count_lines(tmpfile) == expected_line_count[verbose])

    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_eval_set(task_type):
    dataset = [(1, 2, 3, 4), (2, 2, 3, 4), (3, 2, 3, 4), (4, 2, 3, 4)]
    labels = [1, 2, 3, 4]
    train_pool = Pool(dataset, labels, cat_features=[0, 3, 2])

    model = CatBoost({'learning_rate': 1, 'loss_function': 'RMSE', 'iterations': 2, 'task_type': task_type, 'devices': '0'})

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

    model = CatBoost({'loss_function': 'RMSE', 'iterations': 10, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)
    indices, scores = model.get_object_importance(pool, train_pool, top_size=10)
    oimp_path = test_output_path(OIMP_PATH)
    np.savetxt(oimp_path, scores)

    return local_canonical_file(oimp_path)


def test_positive_object_importance_per_object():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    pool = Pool(TEST_FILE, column_description=CD_FILE)

    model = CatBoost({'loss_function': 'RMSE', 'iterations': 10})
    model.fit(train_pool)
    indices, scores = model.get_object_importance(pool, train_pool, type='PerObject', importance_values_sign='Positive')


def test_shap(task_type):
    train_pool = Pool([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 5, 8], cat_features=[])
    test_pool = Pool([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = CatBoostRegressor(iterations=1, max_ctr_complexity=1, depth=2, task_type=task_type, devices='0')
    model.fit(train_pool)
    shap_values = model.get_feature_importance(type=EFstrType.ShapValues, data=test_pool)

    dataset = [(0.5, 1.2), (1.6, 0.5), (1.8, 1.0), (0.4, 0.6), (0.3, 1.6), (1.5, 0.2)]
    labels = [1.1, 1.85, 2.3, 0.7, 1.1, 1.6]
    train_pool = Pool(dataset, labels, cat_features=[])

    model = CatBoost({'iterations': 10, 'max_ctr_complexity': 1, 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool)

    testset = [(0.6, 1.2), (1.4, 0.3), (1.5, 0.8), (1.4, 0.6)]
    predictions = model.predict(testset)
    shap_values = model.get_feature_importance(type=EFstrType.ShapValues, data=Pool(testset))
    assert (len(predictions) == len(shap_values))
    for pred_idx in range(len(predictions)):
        assert (abs(sum(shap_values[pred_idx]) - predictions[pred_idx]) < 1e-9)
    fimp_txt_path = test_output_path(FIMP_TXT_PATH)
    np.savetxt(fimp_txt_path, shap_values)
    return local_canonical_file(fimp_txt_path)


@pytest.mark.parametrize('calc_shap_mode', ['TreeSHAP', 'IndependentTreeSHAP'])
def test_shap_complex_ctr(task_type, calc_shap_mode):
    pool = Pool([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 2]], [0, 0, 5, 8], cat_features=[0, 1, 2])
    reference_data = make_reference_data(pool, calc_shap_mode)
    model = train(pool, {'random_seed': 12302113, 'iterations': 100, 'task_type': task_type, 'devices': '0'})
    shap_values = model.get_feature_importance(type=EFstrType.ShapValues, data=pool, reference_data=reference_data)
    predictions = model.predict(pool)
    assert (len(predictions) == len(shap_values))
    for pred_idx in range(len(predictions)):
        assert (abs(sum(shap_values[pred_idx]) - predictions[pred_idx]) < 1e-9)
    fimp_txt_path = test_output_path(FIMP_TXT_PATH)
    np.savetxt(fimp_txt_path, np.around(np.array(shap_values), 9))
    return local_canonical_file(fimp_txt_path)


def test_shap_interaction_feature_importance(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, max_ctr_complexity=1, task_type=task_type, devices='0')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.array(model.get_feature_importance(type=EFstrType.ShapInteractionValues, data=pool)))
    return local_canonical_file(fimp_npy_path)


def test_shap_interaction_feature_importance_multiclass(task_type):
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    model = CatBoostClassifier(iterations=5, learning_rate=0.03, task_type=task_type, devices='0', loss_function='MultiClass')
    model.fit(pool)
    fimp_npy_path = test_output_path(FIMP_NPY_PATH)
    np.save(fimp_npy_path, np.array(model.get_feature_importance(type=EFstrType.ShapInteractionValues, data=pool)))
    return local_canonical_file(fimp_npy_path)


def test_shap_interaction_feature_on_symmetric(task_type):
    pool = Pool(SMALL_CATEGORIAL_FILE, column_description=SMALL_CATEGORIAL_CD_FILE)
    model = CatBoost(params={'loss_function': 'RMSE', 'iterations': 2, 'task_type': task_type, 'devices': '0', 'one_hot_max_size': 4})
    model.fit(pool)
    shap_interaction_values = model.get_feature_importance(
        type=EFstrType.ShapInteractionValues,
        data=pool,
        thread_count=8
    )

    doc_count = pool.num_row()
    shap_interaction_values = shap_interaction_values[:, :-1, :-1]
    for doc_idx in range(doc_count):
        assert np.all(np.abs(shap_interaction_values[doc_idx] - shap_interaction_values[doc_idx].T) < 1e-8)


def test_shap_interaction_feature_importance_asymmetric_and_symmetric(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.03,
        max_ctr_complexity=1,
        task_type=task_type,
        devices='0')
    model.fit(pool)
    shap_interaction_symm = np.array(model.get_feature_importance(type=EFstrType.ShapInteractionValues, data=pool))
    model._convert_to_asymmetric_representation()
    shap_interaction_asymm = np.array(model.get_feature_importance(type=EFstrType.ShapInteractionValues, data=pool))
    assert np.all(shap_interaction_symm - shap_interaction_asymm < 1e-8)


def test_properties_shap_interaction_values(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=50, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(pool)
    shap_values = classifier.get_feature_importance(
        type=EFstrType.ShapValues,
        data=pool,
        thread_count=8
    )

    shap_interaction_values = classifier.get_feature_importance(
        type=EFstrType.ShapInteractionValues,
        data=pool,
        thread_count=8
    )

    features_count = pool.num_col()
    doc_count = pool.num_row()
    classes_count = 3
    assert shap_interaction_values.shape == (doc_count, classes_count, features_count + 1, features_count + 1)

    shap_values = shap_values[:, :, :-1]
    shap_interaction_values = shap_interaction_values[:, :, :-1, :-1]

    for doc_idx in range(doc_count):
        for class_idx in range(classes_count):
            for feature_idx_1 in range(features_count):
                # check that sum(Ф(i, j)) = ф(i)
                sum_current_interaction_row = np.sum(shap_interaction_values[doc_idx][class_idx][feature_idx_1])
                assert np.allclose(
                    shap_values[doc_idx][class_idx][feature_idx_1],
                    sum_current_interaction_row
                )


def test_shap_interaction_value_between_pair():
    pool = Pool(SMALL_CATEGORIAL_FILE, column_description=SMALL_CATEGORIAL_CD_FILE)
    model = CatBoost(params={'loss_function': 'RMSE', 'iterations': 2, 'devices': '0', 'one_hot_max_size': 4})
    model.fit(pool)
    shap_interaction_values = model.get_feature_importance(
        type=EFstrType.ShapInteractionValues,
        data=pool,
        thread_count=8
    )
    features_count = pool.num_col()
    doc_count = pool.num_row()

    for feature_idx_1 in range(features_count):
        for feature_idx_2 in range(features_count):
            interaction_value = model.get_feature_importance(
                type=EFstrType.ShapInteractionValues,
                data=pool,
                thread_count=8,
                interaction_indices=[feature_idx_1, feature_idx_2]
            )
            if feature_idx_1 == feature_idx_2:
                assert interaction_value.shape == (doc_count, 2, 2)
            else:
                assert interaction_value.shape == (doc_count, 3, 3)
            for doc_idx in range(doc_count):
                if feature_idx_1 == feature_idx_2:
                    assert abs(interaction_value[doc_idx][0][0] - shap_interaction_values[doc_idx][feature_idx_1][feature_idx_2]) < 1e-6
                else:
                    assert abs(interaction_value[doc_idx][0][1] - shap_interaction_values[doc_idx][feature_idx_1][feature_idx_2]) < 1e-6


def test_shap_interaction_value_between_pair_multi():
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=10, loss_function='MultiClass', thread_count=8, devices='0')
    classifier.fit(pool)

    shap_interaction_values = classifier.get_feature_importance(
        type=EFstrType.ShapInteractionValues,
        data=pool,
        thread_count=8
    )
    features_count = pool.num_col()
    doc_count = pool.num_row()
    checked_doc_count = doc_count // 3
    classes_count = 3

    for feature_idx_1 in range(features_count):
        for feature_idx_2 in range(features_count):
            interaction_value = classifier.get_feature_importance(
                type=EFstrType.ShapInteractionValues,
                data=pool,
                thread_count=8,
                interaction_indices=[feature_idx_1, feature_idx_2]
            )
            if feature_idx_1 == feature_idx_2:
                assert interaction_value.shape == (doc_count, classes_count, 2, 2)
            else:
                assert interaction_value.shape == (doc_count, classes_count, 3, 3)

            for doc_idx in range(checked_doc_count):
                for class_idx in range(classes_count):
                    if feature_idx_1 == feature_idx_2:
                        assert abs(interaction_value[doc_idx][class_idx][0][0] - shap_interaction_values[doc_idx][class_idx][feature_idx_1][feature_idx_2]) < 1e-6
                    else:
                        assert abs(interaction_value[doc_idx][class_idx][0][1] - shap_interaction_values[doc_idx][class_idx][feature_idx_1][feature_idx_2]) < 1e-6


def random_xy(num_rows, num_cols_x, seed=20181219, prng=None):
    if prng is None:
        prng = np.random.RandomState(seed=20181219)
    x = prng.randint(100, 104, size=(num_rows, num_cols_x))  # three cat values
    y = _generate_nontrivial_binary_target(num_rows, prng=prng)
    return x, y


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

    prng = np.random.RandomState(seed=20181219)
    x, y = random_xy(6, 4, prng=prng)
    train_pool = Pool(x, y, cat_features=cat_features)

    x0, y0 = random_xy(0, 4, prng=prng)  # empty tuple eval set
    x1, y1 = random_xy(3, 4, prng=prng)
    test0_file = save_and_give_path(y0, x0, 'test0.txt')  # empty file eval set

    with pytest.raises(CatBoostError):
        Pool(x0, y0, cat_features=cat_features)
        pytest.fail("Do not create Pool for empty data")

    model = CatBoost({'learning_rate': 1, 'loss_function': 'RMSE', 'iterations': 2,
                      'allow_const_label': True})

    with pytest.raises(CatBoostError):
        model.fit(train_pool, eval_set=[(x1, y1), (x0, y0)], column_description=cd_file)
        pytest.fail("Do not fit with empty tuple in multiple eval sets")

    with pytest.raises(CatBoostError):
        model.fit(train_pool, eval_set=[(x1, y1), test0_file], column_description=cd_file)
        pytest.fail("Do not fit with empty file in multiple eval sets")

    with pytest.raises(CatBoostError):
        model.fit(train_pool, eval_set=[(x1, y1), None], column_description=cd_file)
        pytest.fail("Do not fit with None in multiple eval sets")

    model.fit(train_pool, eval_set=[None], column_description=cd_file)


def test_multiple_eval_sets():
    # Know the seed to report it if assertion below fails
    seed = 20181219
    prng = np.random.RandomState(seed=seed)

    def model_fit_with(train_set, test_sets, cd_file):
        model = CatBoost({'use_best_model': False, 'loss_function': 'RMSE', 'iterations': 12})
        model.fit(train_set, eval_set=list(reversed(test_sets)), column_description=cd_file)
        return model

    num_features = 11
    cat_features = list(range(num_features))
    cd_file = test_output_path('cd.txt')
    with open(cd_file, 'wt') as cd:
        cd.write('0\tTarget\n')
        for feature_no in sorted(cat_features):
            cd.write('{}\tCateg\n'.format(1 + feature_no))

    x, y = random_xy(12, num_features, prng=prng)
    train_pool = Pool(x, y, cat_features=cat_features)

    x1, y1 = random_xy(13, num_features, prng=prng)
    x2, y2 = random_xy(14, num_features, prng=prng)
    y2 = np.zeros_like(y2)

    test1_file = save_and_give_path(y1, x1, 'test1.txt')
    test2_pool = Pool(x2, y2, cat_features=cat_features)

    model0 = model_fit_with(train_pool, [test1_file, test2_pool], cd_file)
    model1 = model_fit_with(train_pool, [test2_pool, (x1, y1)], cd_file)
    model2 = model_fit_with(train_pool, [(x2, y2), test1_file], cd_file)

    # The three models above shall predict identically on a test set
    # (make sure they are trained with 'use_best_model': False)
    xt, yt = random_xy(7, num_features, prng=prng)
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
    with pytest.raises(CatBoostError):
        model.get_metadata()[1] = '1'
        pytest.fail('Only string keys should be allowed')
    with pytest.raises(CatBoostError):
        model.get_metadata()['1'] = 1
        pytest.fail('Only string values should be allowed')
    model.get_metadata()['1'] = '1'
    assert model.get_metadata().get('1', 'EMPTY') == '1'
    assert model.get_metadata().get('2', 'EMPTY') == 'EMPTY'
    for i in range(100):
        model.get_metadata()[str(i)] = str(i)
    del model.get_metadata()['98']
    with pytest.raises(KeyError):
        i = model.get_metadata()['98']
    for i in range(0, 98, 2):
        assert str(i) in model.get_metadata()
        del model.get_metadata()[str(i)]
    for i in range(0, 98, 2):
        assert str(i) not in model.get_metadata()
        assert str(i + 1) in model.get_metadata()


def test_metadata():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=2,
        learning_rate=0.03,
        loss_function='Logloss',
        metadata={"type": "AAA", "postprocess": "BBB"}
    )
    model.fit(train_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)

    model2 = CatBoost()
    model2.load_model(output_model_path)
    assert 'type' in model2.get_metadata()
    assert model2.get_metadata()['type'] == 'AAA'
    assert 'postprocess' in model2.get_metadata()
    assert model2.get_metadata()['postprocess'] == 'BBB'
    return compare_canonical_models(output_model_path)


@pytest.mark.parametrize('metric', ['Logloss', 'RMSE', 'PRAUC'])
def test_util_eval_metric(metric):
    metric_results = eval_metric([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0.88, 0.22, 1.0, 0.6, 0.12, 0.1, 0.0, 0.5, 0.95, 0.8], metric)
    preds_path = test_output_path(PREDS_PATH)
    np.savetxt(preds_path, np.array(metric_results))
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('metric', ['MultiClass', 'AUC', 'AUC:type=OneVsAll', 'AUC:misclass_cost_matrix=0/1/0.33/0/0/0.239/-1/1.2/0', 'PRAUC', 'PRAUC:type=OneVsAll'])
def test_util_eval_metric_multiclass(metric):
    metric_results = eval_metric([1, 0, 2], [[0.88, 0.22, 0.3], [0.21, 0.45, 0.1], [0.12, 0.32, 0.9]], metric)
    preds_path = test_output_path(PREDS_PATH)
    np.savetxt(preds_path, np.array(metric_results))
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('metric', ['PairLogit', 'PairAccuracy'])
def test_util_eval_metric_pairwise(metric):
    metric_results = eval_metric(
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0.1, 0.1, 0, -1, 0.1, 1, 2],
        metric,
        group_id=[1, 1, 1, 1, 1, 1, 1, 1],
        pairs=[[0, 1], [1, 4], [7, 6], [7, 4], [1, 6], [2, 3], [3, 6], [5, 4]]
    )
    preds_path = test_output_path(PREDS_PATH)
    np.savetxt(preds_path, np.array(metric_results))
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('metric', ['PFound'])
def test_util_eval_metric_subgroups(metric):
    metric_results = eval_metric(
        [-1, 0, 0, 0.5, 0.2, 0.1, -2.39, 1.9],
        [0, 1, 0, -1, 0.7, 0, -2, 2],
        metric,
        group_id=[1, 1, 1, 1, 1, 2, 2, 2],
        subgroup_id=['r', 'r', 'g', 'b', 'g', 'r', 'r', 'g']
    )
    preds_path = test_output_path(PREDS_PATH)
    np.savetxt(preds_path, np.array(metric_results))
    return local_canonical_file(preds_path)


def test_option_used_ram_limit():
    for limit in [1000, 1234.56, 0, 0.0, 0.5,
                  '100', '34.56', '0', '0.0', '0.5',
                  '1.2mB', '1000b', '', None, 'none', 'inf']:
        CatBoost({'used_ram_limit': limit}).fit([[0, 1], [2, 3]], [0, 1])

    for limit in [-1000, 'any', '-0.5', 'nolimit', 'oo']:
        with pytest.raises(Exception):
            CatBoost({'used_ram_limit': limit, 'iterations': 1}).fit([[0, 1], [2, 3]], [0, 1])


def get_values_that_json_dumps_breaks_on():
    name_dtype = {name: value for name, value in np.__dict__.items() if (
        isinstance(value, type) and
        re.match('(int|uint|float|bool).*', name)
    )}
    name_value = {}
    for name, dtype in name_dtype.items():
        try:
            value = dtype(1)
            if str(value).startswith('<'):
                continue
            name_value[name] = value
            name_value['array of ' + name] = np.array([[1, 0], [0, 1]], dtype=dtype)
        except Exception:
            pass
    return name_value


def test_serialization_of_numpy_objects_internal():
    from catboost._catboost import _PreprocessParams
    _PreprocessParams(get_values_that_json_dumps_breaks_on())


def test_serialization_of_numpy_objects_save_model():
    prng = np.random.RandomState(seed=20181219)
    train_pool = Pool(*random_xy(10, 5, prng=prng))
    model = CatBoostClassifier(
        iterations=np.int64(2),
        random_seed=np.int32(0),
        loss_function='Logloss'
    )
    model.fit(train_pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path, format='coreml',
                     export_parameters=get_values_that_json_dumps_breaks_on())


def test_serialization_of_numpy_objects_execution_case():
    from catboost.eval.execution_case import ExecutionCase
    ExecutionCase(get_values_that_json_dumps_breaks_on())


def test_metric_period_redefinition(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile1 = test_output_path('tmpfile1')
    tmpfile2 = test_output_path('tmpfile2')
    model = CatBoost(dict(iterations=10, metric_period=3, task_type=task_type, devices='0'))

    with open(tmpfile1, 'w') as cout:
        model.fit(pool, log_cout=cout)
    with open(tmpfile2, 'w') as cout:
        model.fit(pool, metric_period=2, log_cout=cout)

    assert (_count_lines(tmpfile1) == 5)
    assert (_count_lines(tmpfile2) == 7)


def test_verbose_redefinition(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile1 = test_output_path('tmpfile1')
    tmpfile2 = test_output_path('tmpfile2')
    model = CatBoost(dict(iterations=10, verbose=False, task_type=task_type, devices='0'))

    with open(tmpfile1, 'w') as cout:
        model.fit(pool, log_cout=cout)
    with open(tmpfile2, 'w') as cout:
        model.fit(pool, verbose=True, log_cout=cout)

    assert (_count_lines(tmpfile1) == 0)
    assert (_count_lines(tmpfile2) == 11)


class TestInvalidCustomLossAndMetric(object):
    class GoodCustomLoss(object):
        def calc_ders_range(self, approxes, targets, weights):
            assert len(approxes) == len(targets)
            der1 = 2.0 * (np.array(approxes) - np.array(targets))
            der2 = np.full(len(approxes), -2.0)
            if weights is not None:
                assert len(weights) == len(targets)
                der1 *= np.array(weights)
                der2 *= np.array(weights)
            return list(zip(der1, der2))

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
        with pytest.raises(CatBoostError, match='If loss function is a user defined object, then the eval metric must be specified'):
            model = CatBoost({"loss_function": self.GoodCustomLoss(), "iterations": 2})
            prng = np.random.RandomState(seed=20181219)
            pool = Pool(*random_xy(10, 5, prng=prng))
            model.fit(pool)

    def test_loss_bad_metric_logloss(self):
        with pytest.raises(Exception, match='BadCustomLoss calc_ders_range'):
            model = CatBoost({"loss_function": self.BadCustomLoss(), "eval_metric": "Logloss", "iterations": 2})
            prng = np.random.RandomState(seed=20181219)
            pool = Pool(*random_xy(10, 5, prng=prng))
            model.fit(pool)

    def test_loss_bad_metric_multiclass(self):
        with pytest.raises(Exception, match='BadCustomLoss calc_ders_multi'):
            model = CatBoost({"loss_function": self.BadCustomLoss(), "eval_metric": "MultiClass", "iterations": 2})
            prng = np.random.RandomState(seed=20181219)
            pool = Pool(*random_xy(10, 5, prng=prng))
            model.fit(pool)

    def test_loss_incomplete_metric_logloss(self):
        with pytest.raises(Exception, match='has no.*calc_ders_range'):
            model = CatBoost({"loss_function": self.IncompleteCustomLoss(), "eval_metric": "Logloss", "iterations": 2})
            prng = np.random.RandomState(seed=20181219)
            pool = Pool(*random_xy(10, 5, prng=prng))
            model.fit(pool)

    def test_loss_incomplete_metric_multiclass(self):
        with pytest.raises(Exception, match='has no.*calc_ders_multi'):
            model = CatBoost({"loss_function": self.IncompleteCustomLoss(), "eval_metric": "MultiClass", "iterations": 2})
            prng = np.random.RandomState(seed=20181219)
            pool = Pool(*random_xy(10, 5, prng=prng))
            model.fit(pool)

    def test_custom_metric_object(self):
        with pytest.raises(CatBoostError, match='custom_metric.*must be string'):
            model = CatBoost({"custom_metric": self.GoodCustomMetric(), "iterations": 2})
            prng = np.random.RandomState(seed=20181219)
            pool = Pool(*random_xy(10, 5, prng=prng))
            model.fit(pool)

    def test_loss_none_metric_good(self):
        model = CatBoost({"eval_metric": self.GoodCustomMetric(), "iterations": 2})
        prng = np.random.RandomState(seed=20181219)
        pool = Pool(*random_xy(10, 5, prng=prng))
        model.fit(pool)

    def test_loss_none_metric_incomplete(self):
        with pytest.raises(CatBoostError, match='has no.*evaluate'):
            model = CatBoost({"eval_metric": self.IncompleteCustomMetric(), "iterations": 2})
            prng = np.random.RandomState(seed=20181219)
            pool = Pool(*random_xy(10, 5, prng=prng))
            model.fit(pool)

    def test_custom_loss_and_metric(self):
        model = CatBoost(
            {"loss_function": self.GoodCustomLoss(), "eval_metric": self.GoodCustomMetric(), "iterations": 2}
        )
        prng = np.random.RandomState(seed=20181219)
        pool = Pool(*random_xy(10, 5, prng=prng))
        model.fit(pool)


def test_silent():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    tmpfile1 = test_output_path('tmpfile1')
    tmpfile2 = test_output_path('tmpfile2')
    tmpfile3 = test_output_path('tmpfile3')
    tmpfile4 = test_output_path('tmpfile4')
    tmpfile5 = test_output_path('tmpfile5')

    with open(tmpfile1, 'w') as cout:
        model = CatBoost(dict(iterations=10, silent=True))
        model.fit(pool, log_cout=cout)
    with open(tmpfile2, 'w') as cout:
        model = CatBoost(dict(iterations=10, silent=True))
        model.fit(pool, silent=False, log_cout=cout)
    with open(tmpfile3, 'w') as cout:
        train(pool, {'silent': True}, log_cout=cout)
    with open(tmpfile4, 'w') as cout:
        model = CatBoost(dict(iterations=10, silent=False))
        model.fit(pool, silent=True, log_cout=cout)
    with open(tmpfile5, 'w') as cout:
        model = CatBoost(dict(iterations=10, verbose=5))
        model.fit(pool, silent=True, log_cout=cout)

    assert (_count_lines(tmpfile1) == 0)
    assert (_count_lines(tmpfile2) == 11)
    assert (_count_lines(tmpfile3) == 0)
    assert (_count_lines(tmpfile4) == 0)
    assert (_count_lines(tmpfile5) == 0)


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
    assert (params == params_after_setting)

    prng = np.random.RandomState(seed=20181219)
    data = prng.randint(10, size=(20, 20))
    label = _generate_nontrivial_binary_target(20, prng=prng)
    train_pool = Pool(data, label, cat_features=[1, 2])
    model1.fit(train_pool)
    model_path = test_output_path('model.cb')
    model1.save_model(model_path)

    model2 = CatBoost()
    model2.load_model(model_path)
    params_after_save_model = model2.get_params()
    assert (params.keys() != params_after_save_model.keys())

    model2 = CatBoost()
    model2.set_params(**model1.get_params())
    assert (model1.get_params() == model2.get_params())

    state = model1.__getstate__()
    model2 = CatBoost()
    model2.__setstate__(state)
    assert (model1.get_params() == model2.get_params())


def test_feature_names_from_model():
    input_file = test_output_path('pool')
    with open(input_file, 'w') as inp:
        inp.write('0\t1\t2\t0\n1\t2\t3\t1\n')

    column_description1 = test_output_path('description1.cd')
    create_cd(
        label=3,
        cat_features=[0, 1],
        feature_names={0: 'a', 1: 'b', 2: 'ab'},
        output_path=column_description1
    )

    column_description2 = test_output_path('description2.cd')
    create_cd(
        label=3,
        cat_features=[0, 1],
        output_path=column_description2
    )

    column_description3 = test_output_path('description3.cd')
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

    output_file = test_output_path('feature_names')
    with open(output_file, 'w') as output:
        for i in range(len(pools)):
            pool = pools[i]
            model = CatBoost(dict(iterations=10))
            assert model.feature_names_ is None
            model.fit(pool)
            output.write(str(model.feature_names_) + '\n')

    return local_canonical_file(output_file)


@pytest.mark.parametrize('format', ['cbm', 'json'])
def test_feature_names_from_loaded_model(format):
    df = DataFrame({
        'a': np.random.choice(['X', 'Y', 'Z'], 100),
        'b': np.random.randint(0, 10, 100),
        'c': np.random.randint(0, 10, 100),
        'target': np.random.randint(0, 10, 100),
    })

    feature_names = ['a', 'b', 'c']
    cat_features = ['a']
    pool = Pool(data=df[feature_names], label=df['target'], cat_features=cat_features, feature_names=feature_names)

    model = CatBoostRegressor(iterations=10)
    model.fit(pool)
    assert model.feature_names_ == feature_names

    model_file = test_output_path('model')
    model.save_model(model_file, format=format, pool=pool)
    loaded_model = CatBoostRegressor().load_model(model_file, format=format)
    assert loaded_model.feature_names_ == feature_names


Value_AcceptableAsEmpty = [
    ('', True),
    ('nan', True),
    ('NaN', True),
    ('NAN', True),
    ('NA', True),
    ('Na', True),
    ('na', True),
    ("#N/A", True),
    ("#N/A N/A", True),
    ("#NA", True),
    ("-1.#IND", True),
    ("-1.#QNAN", True),
    ("-NaN", True),
    ("-nan", True),
    ("1.#IND", True),
    ("1.#QNAN", True),
    ("N/A", True),
    ("NULL", True),
    ("n/a", True),
    ("null", True),
    ("Null", True),
    ("none", True),
    ("None", True),
    ('-', True),
    ('junk', False)
]


class TestMissingValues(object):

    def assert_expected(self, pool):
        assert str(pool.get_features()) == str(np.array([[1.0], [float('nan')]]))

    @pytest.mark.parametrize('value,value_acceptable_as_empty', [(None, True)] + Value_AcceptableAsEmpty)
    @pytest.mark.parametrize('object', [list, np.array, DataFrame, Series])
    def test_create_pool_from_object(self, value, value_acceptable_as_empty, object):
        if value_acceptable_as_empty:
            self.assert_expected(Pool(object([[1], [value]])))
            self.assert_expected(Pool(object([1, value])))
        else:
            with pytest.raises(CatBoostError):
                Pool(object([1, value]))

    @pytest.mark.parametrize('value,value_acceptable_as_empty', Value_AcceptableAsEmpty)
    def test_create_pool_from_file(self, value, value_acceptable_as_empty):
        pool_path = test_output_path('pool')
        open(pool_path, 'wt').write('1\t1\n0\t{}\n'.format(value))
        if value_acceptable_as_empty:
            self.assert_expected(Pool(pool_path))
        else:
            with pytest.raises(CatBoostError):
                Pool(pool_path)


def test_model_and_pool_compatibility():
    features = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]
    targets = [(f[0] ^ f[1]) & f[2] for f in features]
    pool1 = Pool(features, targets, cat_features=[0, 1])
    pool2 = Pool(features, targets, cat_features=[1, 2])
    model = CatBoostRegressor(iterations=4)
    model.fit(pool1)
    with pytest.raises(CatBoostError):
        model.predict(pool2)
    with pytest.raises(CatBoostError):
        model.get_feature_importance(type=EFstrType.ShapValues, data=pool2)


@pytest.mark.parametrize('calc_shap_mode', ['TreeSHAP', 'IndependentTreeSHAP'])
def test_shap_verbose(calc_shap_mode):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    reference_data = make_reference_data(pool, calc_shap_mode)

    model = CatBoost(dict(iterations=250))
    model.fit(pool)

    tmpfile = test_output_path('test_data_dumps')
    with open(tmpfile, 'w') as cout:
        model.get_feature_importance(type=EFstrType.ShapValues, data=pool, verbose=12, reference_data=reference_data, log_cout=cout)
    if calc_shap_mode == "TreeSHAP":
        assert (_count_lines(tmpfile) == 5)
    else:
        assert (_count_lines(tmpfile) == 6)


def test_eval_set_with_nans(task_type):
    prng = np.random.RandomState(seed=20181219)
    features = prng.random_sample((10, 200))
    labels = prng.random_sample((10,))
    features_with_nans = features.copy()
    np.putmask(features_with_nans, features_with_nans < 0.5, np.nan)
    model = CatBoost({'iterations': 2, 'loss_function': 'RMSE', 'task_type': task_type, 'devices': '0'})
    train_pool = Pool(features, label=labels)
    test_pool = Pool(features_with_nans, label=labels)
    model.fit(train_pool, eval_set=test_pool)


def test_model_sum_and_init_with_differing_nan_processing_strategy(task_type):
    prng = np.random.RandomState(seed=20200803)
    features = prng.random_sample((10, 200))
    labels = prng.random_sample((10,))
    features_with_nans = features.copy()
    np.putmask(features_with_nans, features_with_nans < 0.5, np.nan)
    no_nan_pool = Pool(features, label=labels)
    nan_pool = Pool(features_with_nans, label=labels)

    def make_model(nan_mode=None, init_model=None):
        model = CatBoostRegressor(
            iterations=10,
            task_type=task_type,
            devices='0',
            nan_mode=nan_mode
        )
        if nan_mode == 'Forbidden':
            model.fit(no_nan_pool, init_model=init_model)
        else:
            model.fit(nan_pool, init_model=init_model)
        return model

    model_as_is = make_model()
    model_as_false = make_model('Min')
    model_as_true = make_model('Max')
    _ = sum_models([model_as_is, model_as_false])
    _ = sum_models([model_as_false, model_as_false])

    make_model('Min', init_model=model_as_is)
    make_model('Forbidden', init_model=model_as_false)

    # TODO(kirillovs): this could be relaxed after properly fixing binary features storage in model
    with pytest.raises(CatBoostError):
        sum_models([model_as_is, model_as_true])
    with pytest.raises(CatBoostError):
        sum_models([model_as_false, model_as_true])
    with pytest.raises(CatBoostError):
        make_model('AsIs', init_model=model_as_true)
    with pytest.raises(CatBoostError):
        make_model('AsTrue', init_model=model_as_false)


def test_learning_rate_auto_set(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model1 = CatBoostClassifier(iterations=10, task_type=task_type, devices='0')
    model1.fit(train_pool)
    predictions1 = model1.predict_proba(test_pool)

    model2 = CatBoostClassifier(iterations=10, learning_rate=model1.learning_rate_, task_type=task_type, devices='0')
    model2.fit(train_pool)
    predictions2 = model2.predict_proba(test_pool)
    assert _check_data(predictions1, predictions2)
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_learning_rate_auto_set_in_cv(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {"iterations": 14, "loss_function": "Logloss", "task_type": task_type,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')},
    )
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:]:
        assert value < prev_value
        prev_value = value
    return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_shap_multiclass(task_type):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=50, loss_function='MultiClass', thread_count=8, task_type=task_type, devices='0')
    classifier.fit(pool)
    pred = classifier.predict(pool, prediction_type='Probability')

    shap_values = classifier.get_feature_importance(
        type=EFstrType.ShapValues,
        data=pool,
        thread_count=8
    )
    features_count = pool.num_col()
    classes_count = 3
    assert pred.shape == (len(pred), classes_count)
    assert shap_values.shape == (len(pred), classes_count, features_count + 1)
    fimp_txt_path = test_output_path(FIMP_TXT_PATH)
    np.savetxt(fimp_txt_path, shap_values.reshape(len(pred), -1), fmt='%.9f')
    shap_values = np.sum(shap_values, axis=2)
    for doc_id in range(len(pred)):
        shap_probas = np.exp(shap_values[doc_id]) / np.sum(np.exp(shap_values[doc_id]))
        assert np.allclose(shap_probas, pred[doc_id])
    return local_canonical_file(fimp_txt_path)


def test_loading_pool_with_numpy_int():
    assert _check_shape(Pool(np.array([[2, 2], [1, 2]]), [1.2, 3.4], cat_features=[0]), object_count=2, features_count=2)


def test_loading_pool_with_numpy_str():
    assert _check_shape(
        Pool(
            np.array([['abc', '2', 'the cat'], ['1', '2', 'on the road']]),
            np.array([1, 3]),
            cat_features=[0],
            text_features=[2]
        ),
        object_count=2,
        features_count=3
    )


def test_loading_pool_with_lists():
    assert _check_shape(
        Pool(
            [['abc', 2, 'the cat'], ['1', 2, 'on the road']],
            [1, 3],
            cat_features=[0],
            text_features=[2]
        ),
        object_count=2,
        features_count=3
    )


def test_pool_group_id_hash():
    def get_group_sizes(group_id):
        assert group_id is not None and len(group_id) > 0
        neighbors_diff = abs(np.diff(group_id))
        group_end_ids = [-1] + list(np.where(neighbors_diff > 0)[0]) + [len(group_id) - 1]
        return np.diff(group_end_ids)

    np.random.seed(42)

    num_rows = 10000
    cnt_groups = 100
    group_id = sorted(np.random.randint(0, cnt_groups, num_rows))
    pool = Pool(data=np.random.randint(0, 1000, (num_rows, 2)),
                label=np.random.choice([-1, 1], num_rows),
                group_id=group_id
                )
    group_id_hash = pool.get_group_id_hash()
    assert len(np.unique(group_id)) == len(np.unique(group_id_hash))
    assert np.all(get_group_sizes(group_id) == get_group_sizes(group_id_hash))


def test_pairs_generation(task_type):
    model = CatBoost({"loss_function": "PairLogit", "iterations": 2, "task_type": task_type})
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model.fit(pool)
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_pairs_generation_generated(task_type):
    model = CatBoostRanker(loss_function='PairLogit', iterations=10, thread_count=8, task_type=task_type, devices='0')

    df = read_csv(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    df = df.loc[:10, :]
    train_target = df.loc[:, 2]
    train_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32)

    df = read_csv(QUERYWISE_TEST_FILE, delimiter='\t', header=None)
    test_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32)

    prng = np.random.RandomState(seed=20181219)
    train_group_id = np.sort(prng.randint(len(train_target) // 3, size=len(train_target)) + 1)
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

    assert np.all(np.isclose(predictions1, predictions2, rtol=1.e-8, equal_nan=True))
    assert np.all(np.isclose(predictions_on_test1, predictions_on_test2, rtol=1.e-8, equal_nan=True))


def test_pairs_generation_with_max_pairs(task_type):
    model = CatBoost({"loss_function": "PairLogit:max_pairs=30", "iterations": 2, "task_type": task_type})
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    model.fit(pool)
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_early_stopping_rounds(task_type):
    train_pool = Pool([[0], [1]], [0, 1])
    test_pool = Pool([[0], [1]], [1, 0])

    model = CatBoostRegressor(od_type='Iter', od_pval=2)
    with pytest.raises(CatBoostError):
        model.fit(train_pool, eval_set=test_pool)

    model = CatBoost(params={'od_pval': 0.001, 'early_stopping_rounds': 2})
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=1)

    model = CatBoostClassifier(loss_function='Logloss:hints=skip_train~true', iterations=1000,
                               learning_rate=0.03, od_type='Iter', od_wait=10)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=1)

    model = train(pool=train_pool, eval_set=test_pool, early_stopping_rounds=2,
                  params={'loss_function': 'Logloss:hints=skip_train~true',
                          'json_log': 'json_log_train.json'})

    return [local_canonical_file(remove_time_from_json(JSON_LOG_PATH)),
            local_canonical_file(remove_time_from_json('catboost_info/json_log_train.json'))]


def test_slice_pool():
    pool = Pool(
        [[0], [1], [2], [3], [4], [5]],
        label=[0, 1, 2, 3, 4, 5],
        group_id=[0, 0, 0, 1, 1, 2],
        pairs=[(0, 1), (0, 2), (1, 2), (3, 4)])

    for bad_indices in [[0], [2], [0, 0, 0]]:
        with pytest.raises(CatBoostError):
            pool.slice(bad_indices)
    rindexes = [
        [0, 1, 2],
        [3, 4],
        np.array([3, 4]),
        [5]
    ]
    for rindex in rindexes:
        sliced_pool = pool.slice(rindex)
        assert _check_data(sliced_pool.get_label(), list(rindex))


def test_fit_and_predict_on_sliced_pools(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    np.random.seed(42)

    train_subset_indices = np.random.choice(
        train_pool.num_row(),
        size=int(train_pool.num_row() * 0.6),
        replace=False
    )

    train_subset_pool = train_pool.slice(train_subset_indices)

    test_subset_indices = np.random.choice(
        test_pool.num_row(),
        size=int(test_pool.num_row() * 0.4),
        replace=False
    )

    test_subset_pool = train_pool.slice(test_subset_indices)

    args = {
        'iterations': 10,
        'loss_function': 'Logloss',
        'task_type': task_type
    }

    model = CatBoostClassifier(**args)
    model.fit(train_subset_pool, eval_set=test_subset_pool)

    pred = model.predict(test_subset_pool)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


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
        'iterations': 20, 'learning_rate': 0.5, 'logging_level': 'Silent', 'loss_function': 'RMSE',
        'boosting_type': 'Plain', 'allow_const_label': True, 'bootstrap_type': 'No',
    }
    evaluator = CatboostEvaluation(
        TRAIN_FILE, fold_size=2, fold_count=2,
        column_description=CD_FILE, partition_random_seed=0)
    first_result = evaluator.eval_features(learn_config=learn_params, eval_metrics='MAE', features_to_eval=[6, 7, 8])
    second_result = evaluator.eval_features(learn_config=learn_params, eval_metrics=['MAE'], features_to_eval=[6, 7, 8])
    assert first_result.get_results()['MAE'] == second_result.get_results()['MAE']


def test_compare():
    dataset = np.array([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60], [20, 15, 85, 60]])
    train_labels = [1.2, 3.4, 9.5, 24.5]
    model = CatBoostRegressor(learning_rate=1, depth=6, loss_function='RMSE', bootstrap_type='No')
    model.fit(dataset, train_labels)
    model2 = CatBoostRegressor(learning_rate=0.1, depth=1, loss_function='MAE', bootstrap_type='No')
    model2.fit(dataset, train_labels)

    try:
        model.compare(model2, Pool(dataset, label=train_labels), ["RMSE"])
    except ImportError as ie:
        pytest.xfail(str(ie)) if str(ie) == "No module named widget" \
            else pytest.fail(str(ie))


def test_cv_fold_count_alias(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_dir_prefix = test_output_path('')
    results_fold_count = cv(pool=pool, params={
        "iterations": 5,
        "learning_rate": 0.03,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": task_type,
        "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
    }, fold_count=4)
    results_nfold = cv(pool=pool, params={
        "iterations": 5,
        "learning_rate": 0.03,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": task_type,
        "train_dir": os.path.join(train_dir_prefix, 'catboost_info'),
    }, nfold=4)
    assert results_fold_count.equals(results_nfold)


def test_predict_loss_function_alias(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test = Pool(TEST_FILE, column_description=CD_FILE)
    booster = train(params={'loss_function': 'MultiClassOneVsAll', 'num_trees': 5}, pool=pool)
    shape_if_loss_function = booster.predict(test).shape

    booster = train(params={'objective': 'MultiClassOneVsAll', 'num_trees': 5}, pool=pool)
    shape_if_objective = booster.predict(test).shape

    assert shape_if_loss_function == shape_if_objective


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
        task_type=task_type, devices='0',
    )
    model.fit(train_pool, eval_set=test_pool)
    pred = model.predict(test_pool)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


def test_permuted_columns_dataset():
    permuted_test, permuted_cd = permute_dataset_columns(AIRLINES_5K_TEST_FILE, AIRLINES_5K_CD_FILE)
    train_pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    test_pool = Pool(AIRLINES_5K_TEST_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    permuted_test_pool = Pool(permuted_test, column_description=permuted_cd, has_header=True)
    model = CatBoostClassifier(
        use_best_model=False,
        max_ctr_complexity=8,
        depth=10,
        boosting_type='Plain',
        iterations=20,
        learning_rate=0.03,
        thread_count=4,
    )
    model.fit(train_pool, eval_set=test_pool)
    pred = model.predict(test_pool)
    permuted_pred = model.predict(permuted_test_pool)
    assert all(pred == permuted_pred)


def non_decreasing(sequence):
    for i in range(1, len(sequence)):
        if sequence[i] < sequence[i - 1]:
            return False
    return True


@pytest.mark.parametrize('iterations', [5, 20, 110], ids=['iterations=5', 'iterations=20', 'iterations=110'])
def do_test_roc(task_type, pool, iterations, additional_train_params={}):
    train_pool = Pool(data_file(pool, 'train_small'), column_description=data_file(pool, 'train.cd'))
    test_pool = Pool(data_file(pool, 'test_small'), column_description=data_file(pool, 'train.cd'))

    model = CatBoostClassifier(loss_function='Logloss', iterations=iterations, **additional_train_params)
    model.fit(train_pool)

    curve = get_roc_curve(model, test_pool, thread_count=4)
    (fpr, tpr, thresholds) = curve
    assert non_decreasing(fpr)
    assert non_decreasing(tpr)

    table = np.array(list(zip(curve[2], [1 - x for x in curve[1]], curve[0])))
    out_roc = test_output_path('roc')
    np.savetxt(out_roc, table)

    try:
        select_threshold(model, data=test_pool, FNR=0.5, FPR=0.5)
        assert False, 'Only one of FNR, FPR must be defined.'
    except CatBoostError:
        pass

    out_bounds = test_output_path('bounds')
    with open(out_bounds, 'w') as f:
        fnr_boundary = select_threshold(model, data=test_pool, FNR=0.4)
        fpr_boundary = select_threshold(model, data=test_pool, FPR=0.2)
        inter_boundary = select_threshold(model, data=test_pool)

        try:
            select_threshold(model, data=test_pool, curve=curve)
            assert False, 'Only one of data and curve parameters must be defined.'
        except CatBoostError:
            pass

        assert fnr_boundary == select_threshold(model, curve=curve, FNR=0.4)
        assert fpr_boundary == select_threshold(model, curve=curve, FPR=0.2)
        assert inter_boundary == select_threshold(model, curve=curve)

        f.write('by FNR=0.4: ' + str(fnr_boundary) + '\n')
        f.write('by FPR=0.2: ' + str(fpr_boundary) + '\n')
        f.write('by intersection: ' + str(inter_boundary) + '\n')

    return [
        local_canonical_file(out_roc),
        local_canonical_file(out_bounds)
    ]


# different iteration parameters needed to check less and more accurate models
@pytest.mark.parametrize('iterations', [5, 20, 110], ids=['iterations=5', 'iterations=20', 'iterations=110'])
def test_roc(task_type, iterations):
    return do_test_roc(task_type, pool='adult', iterations=iterations)


def test_roc_with_target_border():
    return do_test_roc(
        task_type='CPU',
        pool='adult_not_binarized',
        iterations=20,
        additional_train_params={'target_border': 0.4}
    )


def test_roc_cv(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    cv(
        train_pool,
        params={
            'loss_function': 'Logloss',
            'iterations': 10,
            'roc_file': 'out_roc',
            'thread_count': 4,
            'task_type': task_type
        },
    )

    return [
        local_canonical_file('catboost_info/out_roc')
    ]


@pytest.mark.parametrize('boosting_type', ['Ordered'])
@pytest.mark.parametrize('overfitting_detector_type', OVERFITTING_DETECTOR_TYPE)
def test_overfit_detector_with_resume_from_snapshot_and_metric_period(boosting_type, overfitting_detector_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    FIRST_ITERATIONS = 8
    FINAL_ITERATIONS = 100
    OD_WAIT = 10
    # Overfitting must occur between the FIRST_ITERATIONS and FINAL_ITERATIONS.

    models = []

    for metric_period in [1, 5]:
        final_training_stdout_len_wo_snapshot = None

        # must be always run first without snapshot then with it to properly test assertions
        for with_resume_from_snapshot in [False, True]:
            params = {
                'use_best_model': False,
                'boosting_type': boosting_type,
                'thread_count': 4,
                'learning_rate': 0.2,
                'od_type': overfitting_detector_type,
                'metric_period': metric_period,
                'leaf_estimation_iterations': 10,
                'max_ctr_complexity': 4
            }
            if overfitting_detector_type == 'IncToDec':
                params['od_wait'] = OD_WAIT
                params['od_pval'] = 0.5
            elif overfitting_detector_type == 'Iter':
                params['od_wait'] = OD_WAIT
            if with_resume_from_snapshot:
                params['save_snapshot'] = True
                params['snapshot_file'] = test_output_path(
                    'snapshot_with_metric_period={}_od_type={}'.format(
                        metric_period, overfitting_detector_type
                    )
                )
                params['iterations'] = FIRST_ITERATIONS
                small_model = CatBoostClassifier(**params)
                with tempfile.TemporaryFile('w+') as stdout_part:
                    with DelayedTee(sys.stdout, stdout_part):
                        small_model.fit(train_pool, eval_set=test_pool)
                    first_training_stdout_len = sum(1 for line in stdout_part)
                # overfitting detector has not stopped learning yet
                assert small_model.tree_count_ == FIRST_ITERATIONS
            else:
                params['save_snapshot'] = False

            params['iterations'] = FINAL_ITERATIONS
            model = CatBoostClassifier(**params)
            with tempfile.TemporaryFile('w+') as stdout_part:
                with DelayedTee(sys.stdout, stdout_part):
                    model.fit(train_pool, eval_set=test_pool)
                final_training_stdout_len = sum(1 for line in stdout_part)

            def expected_metric_lines(start, finish, period, overfitted=False):
                assert finish > start
                if period == 1:
                    return finish - start
                start = start + (period - start % period) % period
                result = (finish - 1 - start) // period + 1
                if not overfitted and ((finish - 1) % period) != 0:
                    result += 1
                return result

            if with_resume_from_snapshot:
                final_training_stdout_len_with_snapshot = final_training_stdout_len

                assert first_training_stdout_len == expected_metric_lines(0, FIRST_ITERATIONS, metric_period, False) + 4
                assert final_training_stdout_len_wo_snapshot == expected_metric_lines(0, models[0].tree_count_, metric_period, True) + 5
                assert final_training_stdout_len_with_snapshot == expected_metric_lines(FIRST_ITERATIONS, models[0].tree_count_, metric_period, True) + 5

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
    }

    model_1 = CatBoostClassifier(**args)
    model_1.fit(train_pool, eval_set=test_pool)

    args['custom_metric'] = ['AUC', 'Precision']
    model_2 = CatBoostClassifier(**args)
    model_2.fit(train_pool, eval_set=test_pool)

    assert model_1.tree_count_ == model_2.tree_count_


def test_use_loss_if_no_eval_metric_cv(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    params = {
        'iterations': 50,
        'loss_function': 'Logloss',
        'logging_level': 'Silent',
        'task_type': task_type
    }

    cv_params = {
        'params': params,
        'seed': 0,
        'nfold': 3,
        'early_stopping_rounds': 5,
    }

    results_1 = cv(train_pool, **cv_params)

    cv_params['params']['custom_metric'] = ['AUC']
    results_2 = cv(train_pool, **cv_params)

    cv_params['params']['custom_metric'] = []
    cv_params['params']['eval_metric'] = 'AUC'
    results_3 = cv(train_pool, **cv_params)

    assert results_1.shape[0] == results_2.shape[0] and results_2.shape[0] != results_3.shape[0]


def test_target_with_file():
    with pytest.raises(CatBoostError):
        Pool(TRAIN_FILE, label=[1, 2, 3, 4, 5], column_description=CD_FILE)


@pytest.mark.parametrize('metrics', [
    {'custom_metric': ['Accuracy', 'Logloss'], 'eval_metric': None},
    {'custom_metric': ['Accuracy', 'Accuracy'], 'eval_metric': None},
    {'custom_metric': ['Accuracy'], 'eval_metric': 'Logloss'},
    {'custom_metric': ['Accuracy'], 'eval_metric': 'Accuracy'},
    {'custom_metric': ['Accuracy', 'Logloss'], 'eval_metric': 'Logloss'}])
def test_no_fail_if_metric_is_repeated_cv(task_type, metrics):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    params = {
        'iterations': 10,
        'loss_function': 'Logloss',
        'custom_metric': metrics['custom_metric'],
        'logging_level': 'Silent',
        'task_type': task_type
    }
    if metrics['eval_metric'] is not None:
        params['eval_metric'] = metrics['eval_metric']

    cv_params = {
        'params': params,
        'nfold': 2,
        'as_pandas': True,
    }

    cv(train_pool, **cv_params)


IGNORED_FEATURES_DATA_TYPES = ['integer', 'string']


@pytest.mark.parametrize(
    'data_type',
    IGNORED_FEATURES_DATA_TYPES,
    ids=['data_type=' + data_type for data_type in IGNORED_FEATURES_DATA_TYPES]
)
@pytest.mark.parametrize(
    'has_missing',
    [False, True],
    ids=['has_missing=%s' % has_missing for has_missing in [False, True]]
)
def test_cv_with_ignored_features(task_type, data_type, has_missing):
    pool = Pool(TRAIN_FILE, column_description=data_file('adult', 'train_with_id.cd'))

    if (data_type, has_missing) == ('integer', False):
        ignored_features = [0, 2, 5]
    elif (data_type, has_missing) == ('integer', True):
        ignored_features = [0, 2, 5, 23, 100]
    elif (data_type, has_missing) == ('string', False):
        ignored_features = ['C6', 'C9', 'F4', 'F5']
    elif (data_type, has_missing) == ('string', True):
        return pytest.xfail(reason="Not working at the moment. TODO(akhropov): MLTOOLS-4783")
        # ignored_features = ['C6', 'C9', 'F4', 'F5', 'F17', 'F20']
    else:
        raise Exception('bad params: data_type=%s, has_missing=%s' % (data_type, has_missing))

    train_dir_prefix = test_output_path('')
    results = cv(
        pool,
        {
            "iterations": 20,
            "learning_rate": 0.03,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": task_type,
            "ignored_features": ignored_features,
            "train_dir": os.path.join(train_dir_prefix, 'catboost_info')
        },
    )
    assert "train-Logloss-mean" in results

    prev_value = results["train-Logloss-mean"][0]
    for value in results["train-Logloss-mean"][1:]:
        assert value < prev_value
        prev_value = value

    # Unfortunately, for GPU results differ too much between different GPU models.
    if task_type != 'GPU':
        return local_canonical_file(remove_time_from_json(os.path.join(train_dir_prefix, JSON_LOG_CV_PATH(0))))


def test_use_last_testset_for_best_iteration():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    test_size = test_pool.num_row()
    half_size = test_size // 2
    test_pool_1 = test_pool.slice(list(range(half_size)))
    test_pool_2 = test_pool.slice(list(range(half_size, test_size)))
    metric = 'Logloss'

    args = {
        'iterations': 100,
        'loss_function': metric,
        'random_seed': 6,
        'leaf_estimation_iterations': 10,
        'max_ctr_complexity': 4,
        'boosting_type': 'Ordered',
        'bootstrap_type': 'Bayesian',
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
        'use_best_model': True,
        'task_type': task_type,
        'learning_rate': 0.3
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
            'RMSEWithUncertainty'
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
            'F:beta=2',
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
            'F:beta=2',
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
            'PairAccuracy',
            'QueryRMSE',
            'QuerySoftMax',
            'PFound',
            'NDCG',
            'AverageGain',
            'MRR',
            'ERR',
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
            'PythonUserDefinedPerObject',
            'F:beta=2',
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
            'RMSEWithUncertainty',
            'SMAPE',
            'TotalF1',
            'UserDefinedPerObject',
            'UserDefinedQuerywise',
            'WKappa',
            'YetiRank',
            'YetiRankPairwise',
            'ZeroOneLoss',
            'LogLikelihoodOfPrediction',
            'MRR',
            'ERR',
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
        train_features_df, cat_features = load_pool_features_as_df(TRAIN_FILE, CD_FILE)
        test_features_df, _ = load_pool_features_as_df(TEST_FILE, CD_FILE)

        prng = np.random.RandomState(seed=20181219)
        train_pool = Pool(
            data=train_features_df,
            label=_generate_random_target(train_features_df.shape[0], prng=prng),
            cat_features=cat_features
        )
        test_pool = Pool(
            data=test_features_df,
            label=_generate_random_target(test_features_df.shape[0], prng=prng),
            cat_features=cat_features
        )
        set_random_weight(train_pool, prng=prng)
        set_random_weight(test_pool, prng=prng)

        cb = CatBoostRegressor(loss_function='RMSE', iterations=3, task_type=task_type, devices='0')
        cb.fit(train_pool)
        return (cb, test_pool)

    @pytest.fixture
    def a_classification_learner(self, task_type):
        train_features_df, cat_features = load_pool_features_as_df(TRAIN_FILE, CD_FILE)
        test_features_df, _ = load_pool_features_as_df(TEST_FILE, CD_FILE)

        prng = np.random.RandomState(seed=20181219)
        train_pool = Pool(
            data=train_features_df,
            label=_generate_nontrivial_binary_target(train_features_df.shape[0], prng=prng),
            cat_features=cat_features
        )
        test_pool = Pool(
            data=test_features_df,
            label=_generate_nontrivial_binary_target(test_features_df.shape[0], prng=prng),
            cat_features=cat_features
        )
        set_random_weight(train_pool, prng=prng)
        set_random_weight(test_pool, prng=prng)

        cb = CatBoostClassifier(loss_function='Logloss', iterations=3, task_type=task_type, devices='0')
        cb.fit(train_pool)
        return (cb, test_pool)

    @pytest.fixture
    def a_multiclass_learner(self, task_type):
        train_pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
        test_pool = Pool(CLOUDNESS_TEST_FILE, column_description=CLOUDNESS_CD_FILE)
        prng = np.random.RandomState(seed=20181219)
        set_random_weight(train_pool, prng=prng)
        set_random_weight(test_pool, prng=prng)
        cb = CatBoostClassifier(loss_function='MultiClass', iterations=3, use_best_model=False, task_type=task_type, devices='0')
        cb.fit(train_pool)
        return (cb, test_pool)

    def a_ranking_learner(self, task_type, metric):
        train_pool = Pool(QUERYWISE_TRAIN_FILE, pairs=QUERYWISE_TRAIN_PAIRS_FILE_WITH_PAIR_WEIGHT, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
        test_pool = Pool(QUERYWISE_TEST_FILE, pairs=QUERYWISE_TEST_PAIRS_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_WEIGHT)
        prng = np.random.RandomState(seed=20181219)
        set_random_weight(train_pool, prng=prng)
        set_random_weight(test_pool, prng=prng)

        if metric == 'QueryRMSE':
            loss_function = 'QueryRMSE'
        else:
            loss_function = 'PairLogit'

        cb = CatBoostRanker(loss_function=loss_function, iterations=3, task_type=task_type, devices='0')
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
                list(map(verify_finite, (value_a, value_b)))
                use_weights_has_effect = value_a != value_b
                del values[name_class]
            if use_weights_has_effect:
                break
        assert use_weights_has_effect, "param `use_weights` has no effect\n\teval_metrics={}".format(eval_metrics)


def test_lambda_mart():
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    cb = CatBoostRanker(loss_function='LambdaMart', iterations=3, devices='0')
    cb.fit(pool)


@pytest.mark.parametrize('param_type', ['indices', 'strings'])
def test_set_cat_features_in_init(param_type):
    if param_type == 'indices':
        cat_features_param = [1, 2]
        feature_names_param = None
    else:
        cat_features_param = ['feat1', 'feat2']
        feature_names_param = ['feat' + str(i) for i in range(20)]

    prng = np.random.RandomState(seed=20181219)
    data = prng.randint(10, size=(20, 20))
    label = _generate_nontrivial_binary_target(20, prng=prng)
    train_pool = Pool(data, label, cat_features=cat_features_param, feature_names=feature_names_param)
    test_pool = Pool(data, label, cat_features=cat_features_param, feature_names=feature_names_param)

    params = {
        'logging_level': 'Silent',
        'loss_function': 'Logloss',
        'iterations': 10,
        'thread_count': -1,
        'random_seed': 20
    }

    model1 = CatBoost(params)
    model1.fit(train_pool)

    params_with_cat_features = params.copy()
    params_with_cat_features['cat_features'] = cat_features_param

    model2 = CatBoost(params_with_cat_features)
    model2.fit(train_pool)
    assert (model1.get_cat_feature_indices() == model2.get_cat_feature_indices())
    assert (np.array_equal(model1.predict(test_pool), model2.predict(test_pool)))

    model1 = CatBoost(params)
    model1.fit(train_pool)
    params_with_wrong_cat_features = params.copy()
    params_with_wrong_cat_features['cat_features'] = [0, 2] if param_type == 'indices' else ['feat0', 'feat2']
    model2 = CatBoost(params_with_wrong_cat_features)
    with pytest.raises(CatBoostError):
        model2.fit(train_pool)

    # the following tests won't work for param_type == 'strings' because it requires X param in fit to be Pool
    if param_type == 'indices':
        model1 = CatBoost(params_with_cat_features)
        model1.fit(X=data, y=label)
        model2 = model1
        model2.fit(X=data, y=label)
        assert (np.array_equal(model1.predict(test_pool), model2.predict(test_pool)))

        model1 = CatBoost(params_with_cat_features)
        with pytest.raises(CatBoostError):
            model1.fit(X=data, y=label, cat_features=[1, 3])

        model1 = CatBoost(params_with_cat_features)
        model1.fit(X=data, y=label, eval_set=(data, label))
        assert (model1.get_cat_feature_indices() == [1, 2])

        model1 = CatBoost(params_with_wrong_cat_features)
        with pytest.raises(CatBoostError):
            model1.fit(X=data, y=label, eval_set=test_pool)

        model1 = CatBoost(params_with_cat_features)
        state = model1.__getstate__()
        model2 = CatBoost()
        model2.__setstate__(state)
        model1.fit(X=data, y=label)
        model2.fit(X=data, y=label)
        assert (np.array_equal(model1.predict(test_pool), model2.predict(test_pool)))
        assert (model2.get_cat_feature_indices() == [1, 2])

        model1 = CatBoost(params_with_cat_features)
        model2 = CatBoost()
        model2.set_params(**model1.get_params())
        model1.fit(X=data, y=label)
        model2.fit(X=data, y=label)
        assert (np.array_equal(model1.predict(test_pool), model2.predict(test_pool)))
        assert (model2.get_cat_feature_indices() == [1, 2])

        model1 = CatBoost(params_with_cat_features)
        state = model1.__getstate__()
        model2 = CatBoostClassifier()
        model2.__setstate__(state)
        model1.fit(X=data, y=label)
        model2.fit(X=data, y=label)
        assert (np.array_equal(model1.predict(test_pool), model2.predict(test_pool, prediction_type='RawFormulaVal')))
        assert (model2.get_cat_feature_indices() == [1, 2])

        model1 = CatBoost(params_with_cat_features)
        model2 = CatBoostClassifier()
        model2.set_params(**model1.get_params())
        model1.fit(X=data, y=label)
        model2.fit(X=data, y=label)
        assert (np.array_equal(model1.predict(test_pool), model2.predict(test_pool, prediction_type='RawFormulaVal')))
        assert (model2.get_cat_feature_indices() == [1, 2])


def test_no_yatest_common():
    assert "yatest" not in globals()


def test_keep_metric_params_precision():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=10)
    model.fit(train_pool)
    metrics = ['Quantile:alpha=0.6']
    metrics_evals = model.eval_metrics(test_pool, metrics)
    for metric in metrics:
        assert metric in metrics_evals


def test_shrink():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    args = {
        'iterations': 30,
        'loss_function': 'Logloss',
        'use_best_model': False,
        'learning_rate': 0.3
    }
    model = CatBoostClassifier(**args)
    args['iterations'] = 9
    model2 = CatBoostClassifier(**args)

    model.fit(train_pool, eval_set=test_pool)
    model2.fit(train_pool, eval_set=test_pool)
    assert model.tree_count_ == 30
    model.shrink(9)
    assert model.tree_count_ == 9
    pred1 = model.predict(test_pool)
    pred2 = model2.predict(test_pool)
    assert np.all(pred1 == pred2)
    model.shrink(8, ntree_start=1)
    assert model.tree_count_ == 7


def test_shrink_with_bias():
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE_WITH_GROUP_ID)
    args = {
        'iterations': 30,
        'loss_function': 'RMSE',
        'use_best_model': False,
        'learning_rate': 0.3
    }
    model = CatBoostRegressor(**args)
    model.fit(train_pool)
    scale, bias = model.get_scale_and_bias()
    model.shrink(9)
    assert (scale, bias) == model.get_scale_and_bias()
    model.shrink(8, ntree_start=1)
    assert (scale, 0) == model.get_scale_and_bias()


def test_set_scale_and_bias():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    args = {
        'iterations': 30,
        'loss_function': 'Logloss',
        'use_best_model': False,
        'learning_rate': 0.3
    }
    model = CatBoostClassifier(**args)
    model.fit(train_pool)
    pred1 = model.predict(test_pool, prediction_type='RawFormulaVal')
    assert (1., 0.) == model.get_scale_and_bias()
    model.set_scale_and_bias(3.14, 15.)
    assert (3.14, 15.) == model.get_scale_and_bias()
    pred2 = model.predict(test_pool, prediction_type='RawFormulaVal')
    assert np.all(abs(pred1 * 3.14 + 15 - pred2) < 1e-15)


def test_get_metric_evals(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, eval_metric='Accuracy', task_type=task_type)
    model.fit(train_pool, eval_set=test_pool)
    evals_path = test_output_path('evals.txt')
    with open(evals_path, 'w') as f:
        pprint.PrettyPrinter(stream=f).pprint(model.evals_result_)
    return local_canonical_file(evals_path)


def test_get_evals_result_without_eval_set():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10, custom_metric=['AUC', 'Accuracy'], random_state=0)
    model.fit(train_pool)
    evals_path = test_output_path('evals.txt')
    with open(evals_path, 'w') as f:
        pprint.PrettyPrinter(stream=f).pprint(model.get_evals_result())
    return local_canonical_file(evals_path)


def test_best_score(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'eval_metric': 'ZeroOneLoss',
        'custom_metric': ['Precision', 'CtrFactor'],
        'task_type': task_type,
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool)
    evals_result = model.evals_result_
    best_score = model.best_score_
    assert best_score.keys() == evals_result.keys()
    for pool_name in best_score:
        assert set(best_score[pool_name].keys()) == set(evals_result[pool_name].keys())
        for metric_name in best_score[pool_name]:
            if metric_name == 'CtrFactor':
                assert abs(best_score[pool_name][metric_name] - 1) == min(abs(value - 1) for value in evals_result[pool_name][metric_name])
            elif metric_name in ['ZeroOneLoss', 'Logloss']:
                assert best_score[pool_name][metric_name] == min(evals_result[pool_name][metric_name])
            else:
                assert best_score[pool_name][metric_name] == max(evals_result[pool_name][metric_name])


def test_best_iteration(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'eval_metric': 'ZeroOneLoss',
        'custom_metric': ['Precision', 'Recall'],
        'task_type': task_type,
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool)
    log_path = test_output_path('log.txt')
    with open(log_path, 'w') as cout:
        model.fit(train_pool, eval_set=test_pool, log_cout=cout)
    with open(log_path, 'r') as log_file:
        content = log_file.read()
        best_iteration_from_log = re.search(r'(?<=bestIteration = )\d+', content).group(0)
    assert str(model.best_iteration_) == best_iteration_from_log


def assert_sum_models_equal_sliced_copies(train, test, cd):
    train_pool = Pool(train, column_description=cd)
    test_pool = Pool(test, column_description=cd)
    iter_step = 10
    model_count = 2
    model = CatBoostClassifier(iterations=iter_step * model_count)
    model.fit(train_pool)

    truncated_copies = [model.copy() for _ in range(model_count)]
    for i, truncated_model in enumerate(truncated_copies):
        truncated_model.shrink(ntree_start=i * iter_step, ntree_end=(i + 1) * iter_step)

        pred_local_shrinked = truncated_model.predict(test_pool, prediction_type='RawFormulaVal')
        pred_local = model.predict(
            test_pool,
            prediction_type='RawFormulaVal',
            ntree_start=i * iter_step,
            ntree_end=(i + 1) * iter_step)

        assert np.all(pred_local_shrinked == pred_local)
        assert np.all(model.classes_ == truncated_model.classes_)

    weights = [1.0] * model_count
    merged_model = sum_models(truncated_copies, weights)
    pred = model.predict(test_pool, prediction_type='RawFormulaVal')
    merged_pred = merged_model.predict(test_pool, prediction_type='RawFormulaVal')

    assert np.all(pred == merged_pred)
    assert np.all(model.classes_ == merged_model.classes_)


def test_model_merging():
    assert_sum_models_equal_sliced_copies(TRAIN_FILE, TEST_FILE, CD_FILE)
    # multiclass
    assert_sum_models_equal_sliced_copies(CLOUDNESS_TRAIN_FILE, CLOUDNESS_TEST_FILE, CLOUDNESS_ONLY_NUM_CD_FILE)


def test_model_sum_labels():
    n_samples = 100
    n_features = 10

    """
        list of (expected_sum_classes, lists of (loss_function, class_names, label_set))
        if expected_sum_classes is False it means sum should fail
    """
    params_list = [
        (
            False,
            [
                ('Logloss', ['0', '1'], [0, 1]),
                ('Logloss', ['1', '0'], [0, 1]),
                ('Logloss', ['0', '1'], [0, 1])
            ]
        ),
        (False, [('Logloss', None, [0, 1]), ('Logloss', None, [1, 2])]),
        (False, [('Logloss', None, [0, 1]), ('MultiClass', None, [1, 2])]),
        (False, [('RMSE', None, [0.1, 0.2, 1.0]), ('MultiClass', None, [1, 2, 4])]),
        (False, [('MultiClass', None, [0, 1, 2]), ('MultiClass', None, [1, 2, 3])]),
        (
            False,
            [
                ('MultiClass', None, ['class0', 'class1', 'class2']),
                ('MultiClass', None, ['Class0', 'Class1', 'Class2', 'Class3'])
            ]
        ),
        ([0, 1], [('Logloss', None, [0, 1]), ('Logloss', None, [0, 1]), ('Logloss', None, [0, 1])]),
        (
            ['class0', 'class1', 'class2'],
            [
                ('MultiClass', None, ['class0', 'class1', 'class2']),
                ('MultiClass', None, ['class0', 'class1', 'class2'])
            ]
        ),
        ([1, 2], [('RMSE', None, [0.1, 0.2, 1.0]), ('Logloss', None, [1, 2])]),
        ([], [('RMSE', None, [0.1, 0.2, 1.0]), ('RMSE', None, [0.22, 0.7, 1.3, 1.7])]),
    ]

    for expected_classes, train_specs_list in params_list:
        models = []
        for loss_function, class_names, label_set in train_specs_list:
            features, labels = generate_random_labeled_dataset(
                n_samples=n_samples,
                n_features=n_features,
                labels=label_set
            )
            params = {'loss_function': loss_function, 'iterations': 5}
            if class_names:
                params['class_names'] = class_names
            model = CatBoost(params)
            model.fit(features, labels)
            models.append(model)

        if expected_classes is False:
            with pytest.raises(CatBoostError):
                sum_models(models)
        else:
            model_sum = sum_models(models)
            assert np.all(model_sum.classes_ == expected_classes)


def test_tree_depth_pairwise(task_type):
    if task_type == 'GPU':
        with pytest.raises(CatBoostError):
            CatBoost({'iterations': 2, 'loss_function': 'PairLogitPairwise', 'task_type': task_type, 'devices': '0', 'depth': 9})
        CatBoost({'iterations': 2, 'loss_function': 'PairLogitPairwise', 'task_type': task_type, 'devices': '0', 'depth': 8})


def test_eval_set_with_no_target(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    eval_set_pool = Pool(TEST_FILE, column_description=data_file('train_notarget.cd'))
    model = CatBoost({'iterations': 2, 'loss_function': 'Logloss', 'task_type': task_type, 'devices': '0'})
    model.fit(train_pool, eval_set=eval_set_pool)

    evals_path = test_output_path('evals.txt')
    with open(evals_path, 'w') as f:
        pprint.PrettyPrinter(stream=f).pprint(model.get_evals_result())
    return local_canonical_file(evals_path)


def test_eval_set_with_no_target_with_eval_metric(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    eval_set_pool = Pool(TEST_FILE, column_description=data_file('train_notarget.cd'))
    model = CatBoost(
        {
            'iterations': 2,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'task_type': task_type,
            'devices': '0'
        }
    )
    with pytest.raises(CatBoostError):
        model.fit(train_pool, eval_set=eval_set_pool)


def test_eval_period_size():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=10)
    model.fit(train_pool, eval_set=test_pool)

    eval_metrics_all_trees_path = test_output_path('eval_metrics_all_trees.txt')
    with open(eval_metrics_all_trees_path, 'w') as f:
        pprint.PrettyPrinter(stream=f).pprint(
            model.eval_metrics(test_pool, ['AUC', 'Recall'], eval_period=20)
        )

    eval_metrics_begin_end_path = test_output_path('eval_metrics_begin_end.txt')
    with open(eval_metrics_begin_end_path, 'w') as f:
        pprint.PrettyPrinter(stream=f).pprint(
            model.eval_metrics(
                test_pool,
                metrics=['AUC', 'Recall'],
                ntree_start=3,
                ntree_end=5,
                eval_period=20)
        )

    return [
        local_canonical_file(eval_metrics_all_trees_path),
        local_canonical_file(eval_metrics_begin_end_path)
    ]


def test_output_border_file(task_type):
    OUTPUT_BORDERS_FILE = 'output_border_file.dat'

    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    args = {
        'iterations': 30,
        'loss_function': 'Logloss',
        'use_best_model': False,
        'learning_rate': 0.3
    }
    model1 = CatBoostClassifier(border_count=32,
                                output_borders=OUTPUT_BORDERS_FILE,
                                **args)
    model2 = CatBoostClassifier(input_borders=os.path.join('catboost_info', OUTPUT_BORDERS_FILE),
                                **args)

    model3 = CatBoostClassifier(**args)
    model4 = CatBoostClassifier(border_count=2, **args)

    model1.fit(train_pool)
    model2.fit(train_pool)
    model3.fit(train_pool)
    model4.fit(train_pool)
    pred1 = model1.predict(test_pool)
    pred2 = model2.predict(test_pool)
    pred3 = model3.predict(test_pool)
    pred4 = model4.predict(test_pool)
    assert np.all(pred1 == pred2)
    assert not np.all(pred1 == pred3)
    assert not np.all(pred1 == pred4)


def test_output_border_file_regressor(task_type):
    OUTPUT_BORDERS_FILE = 'output_border_file.dat'

    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    args = {
        'iterations': 30,
        'loss_function': 'RMSE',
        'use_best_model': False,
        'learning_rate': 0.3
    }
    model1 = CatBoostRegressor(border_count=32,
                               output_borders=OUTPUT_BORDERS_FILE,
                               **args)
    model2 = CatBoostRegressor(input_borders=os.path.join('catboost_info', OUTPUT_BORDERS_FILE),
                               **args)

    model3 = CatBoostRegressor(**args)
    model4 = CatBoostRegressor(border_count=2, **args)

    model1.fit(train_pool)
    model2.fit(train_pool)
    model3.fit(train_pool)
    model4.fit(train_pool)
    pred1 = model1.predict(test_pool)
    pred2 = model2.predict(test_pool)
    pred3 = model3.predict(test_pool)
    pred4 = model4.predict(test_pool)
    assert _check_data(pred1, pred2)
    assert not _check_data(pred1, pred3)
    assert not _check_data(pred1, pred4)


def test_output_border_file_ranker(task_type):
    OUTPUT_BORDERS_FILE = 'output_border_file.dat'

    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
    args = {
        'iterations': 30,
        'loss_function': 'YetiRank',
        'use_best_model': False,
        'learning_rate': 0.3
    }
    model1 = CatBoostRanker(border_count=32, output_borders=OUTPUT_BORDERS_FILE, **args)
    model2 = CatBoostRanker(input_borders=os.path.join('catboost_info', OUTPUT_BORDERS_FILE), **args)

    model3 = CatBoostRanker(**args)
    model4 = CatBoostRanker(border_count=2, **args)

    model1.fit(train_pool)
    model2.fit(train_pool)
    model3.fit(train_pool)
    model4.fit(train_pool)
    pred1 = model1.predict(test_pool)
    pred2 = model2.predict(test_pool)
    pred3 = model3.predict(test_pool)
    pred4 = model4.predict(test_pool)
    assert _check_data(pred1, pred2)
    assert not _check_data(pred1, pred3)
    assert not _check_data(pred1, pred4)


def test_save_border_file():
    output_borders_file = test_output_path('output_borders_file.dat')
    save_borders_file = test_output_path('save_borders_file.dat')

    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    args = {
        'iterations': 30,
        'loss_function': 'Logloss',
        'use_best_model': False,
        'learning_rate': 0.3
    }
    model = CatBoostClassifier(border_count=32,
                               output_borders=output_borders_file,
                               **args)

    model.fit(train_pool)
    model.save_borders(save_borders_file)
    return [local_canonical_file(save_borders_file), local_canonical_file(output_borders_file)]


def test_set_feature_names():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)

    model = CatBoostClassifier(loss_function='Logloss', iterations=10)
    model.fit(train_pool)

    names = ["feature_{}".format(x) for x in range(train_pool.num_col())]
    model.set_feature_names(names)
    assert names == model.feature_names_


def test_model_comparison():
    def fit_model(iterations):
        pool = Pool(TRAIN_FILE, column_description=CD_FILE)
        model = CatBoostClassifier(iterations=iterations)
        model.fit(pool)
        return model

    model0 = CatBoostClassifier()
    model1 = fit_model(42)
    model2 = fit_model(5)

    # Test checks that model is fitted.
    with pytest.raises(CatBoostError):
        model1 == model0

    with pytest.raises(CatBoostError):
        model0 == model1

    # Trained model must not equal to object of other type.
    assert model1 != 42
    assert not (model1 == 'hello')

    # Check identity.
    assert model1 == model1
    assert not (model1 != model1)
    assert model1 == model1.copy()

    # Check equality to other model.
    assert not (model1 == model2)
    assert (model1 != model2)


def test_param_synonyms(task_type):
    # list of  ([synonyms list], value)
    synonym_params = [
        (['loss_function', 'objective'], 'CrossEntropy'),
        (['iterations', 'num_boost_round', 'n_estimators', 'num_trees'], 5),
        (['learning_rate', 'eta'], 0.04),
        (['random_seed', 'random_state'], 1),
        (['l2_leaf_reg', 'reg_lambda'], 4),
        (['depth', 'max_depth'], 7),
        (['rsm', 'colsample_bylevel'], 0.5),
        (['border_count', 'max_bin'], 32),
        # (['verbose', 'verbose_eval'], True), # TODO(akhropov): support 'verbose_eval' in CatBoostClassifier ?
    ]

    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)

    """
      don't test every possible synonym name combination - it's too computationally expensive
      process iteratively and just update all synonyms for which synonym names lists are non exhausted yet
    """
    variants_count = max((len(synonym_names) for synonym_names, value in synonym_params))

    canonical_predictions = None

    for variant_idx in range(variants_count):
        params = {'task_type': task_type, 'devices': '0'}
        for synonym_names, value in synonym_params:
            synonym_name = synonym_names[variant_idx] if variant_idx < len(synonym_names) else synonym_names[0]
            params[synonym_name] = value

        for model in [CatBoost(params), CatBoostClassifier(**params)]:
            with tempfile.TemporaryFile('w+') as stdout_part:
                with DelayedTee(sys.stdout, stdout_part):
                    model.fit(train_pool)
                training_stdout_len = sum(1 for line in stdout_part)

            predictions = model.predict(train_pool, prediction_type='RawFormulaVal')
            if canonical_predictions is None:
                canonical_predictions = predictions
                canonical_training_stdout_len = training_stdout_len
            else:
                assert all(predictions == canonical_predictions)
                assert training_stdout_len == canonical_training_stdout_len


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_grow_policy_fails(task_type, grow_policy):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    args = {
        'iterations': 30,
        'loss_function': 'Logloss',
        'use_best_model': False,
        'learning_rate': 0.3,
        'grow_policy': grow_policy,
        'boosting_type': 'Plain',
        'task_type': task_type,
        'devices': '0'
    }
    model = CatBoostClassifier(**args)

    model.fit(train_pool)

    with pytest.raises(CatBoostError):
        model.shrink(9)
    with pytest.raises(CatBoostError):
        model.get_object_importance(test_pool, train_pool)

    model_output = test_output_path('model')
    for format in ['AppleCoreML', 'cpp', 'python', 'onnx']:
        with pytest.raises(CatBoostError):
            model.save_model(model_output, format=format)
    _ = sum_models([model, model.copy()])


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_multiclass_grow_policy(task_type, grow_policy):
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    # MultiClass
    classifier = CatBoostClassifier(
        iterations=2,
        loss_function='MultiClass',
        thread_count=8,
        task_type=task_type,
        devices='0',
        boosting_type='Plain',
        grow_policy=grow_policy
    )
    classifier.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    classifier.save_model(output_model_path)
    new_classifier = CatBoostClassifier()
    new_classifier.load_model(output_model_path)
    pred1 = classifier.predict_proba(pool)
    pred2 = new_classifier.predict_proba(pool)
    assert np.array_equal(pred1, pred2)
    learn_error_path = test_output_path("learn_error.txt")
    np.savetxt(learn_error_path, np.array(classifier.evals_result_['learn']['MultiClass']), fmt='%.8f')
    return local_canonical_file(learn_error_path)


@pytest.mark.parametrize('grow_policy', NONSYMMETRIC)
def test_grow_policy_restriction(task_type, grow_policy):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    params = {
        'iterations': 2,
        'thread_count': 8,
        'task_type': task_type,
        'devices': '0',
        'grow_policy': grow_policy
    }

    with pytest.raises(Exception):
        if grow_policy == 'Lossguide':
            params['max_leaves'] = 65537
        else:
            params['max_depth'] = 17
        classifier = CatBoostClassifier(**params)
        classifier.fit(pool)

    if grow_policy == 'Lossguide':
        params['max_leaves'] = 65536
    else:
        params['max_depth'] = 16
    classifier = CatBoostClassifier(**params)
    classifier.fit(pool)
    pred = classifier.predict_proba(pool)
    preds_path = test_output_path(PREDS_PATH)
    np.save(preds_path, np.array(pred))
    return local_canonical_file(preds_path)


def test_use_all_cpus(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE, thread_count=-1)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE, thread_count=-1)
    model = CatBoostClassifier(iterations=10, task_type=task_type, thread_count=-1, devices='0')
    model.fit(train_pool)
    model.predict(test_pool, thread_count=-1)
    model.predict_proba(test_pool, thread_count=-1)
    model.staged_predict(test_pool, thread_count=-1)
    model.staged_predict_proba(test_pool, thread_count=-1)


def test_baseline():
    input_pool = Pool(np.ones((5, 4)))
    baseline = np.array([[1, 3, 2, 1, 2]], dtype=np.float32).reshape(5, 1)
    input_pool.set_baseline(baseline)
    assert (input_pool.get_baseline() == baseline).all()


EVAL_TYPES = ['All', 'SeqRem', 'SeqAdd', 'SeqAddAndAll']
EVAL_PROBLEMS = ['binclass', 'multiclass', 'regression', 'ranking']


@pytest.mark.parametrize('eval_type', EVAL_TYPES, ids=['eval_type=%s' % eval_type for eval_type in EVAL_TYPES])
@pytest.mark.parametrize('problem', EVAL_PROBLEMS, ids=['problem=%s' % problem for problem in EVAL_PROBLEMS])
def test_eval_features(task_type, eval_type, problem):
    if problem == 'binclass':
        loss_function = 'Logloss'
        eval_metrics = ['AUC']
        train_file = TRAIN_FILE
        cd_file = CD_FILE
        features_to_eval = [6, 7, 8]
        group_column = None
    elif problem == 'multiclass':
        loss_function = 'MultiClass'
        eval_metrics = ['Accuracy']
        train_file = CLOUDNESS_TRAIN_FILE
        cd_file = CLOUDNESS_CD_FILE
        features_to_eval = [101, 102, 105, 106]
        group_column = None
    elif problem == 'regression':
        loss_function = 'RMSE'
        eval_metrics = ['RMSE']
        train_file = TRAIN_FILE
        cd_file = CD_FILE
        features_to_eval = [6, 7, 8]
        group_column = None
    elif problem == 'ranking':
        loss_function = 'QueryRMSE'
        eval_metrics = ['NDCG']
        train_file = QUERYWISE_TRAIN_FILE
        cd_file = QUERYWISE_CD_FILE
        features_to_eval = [20, 22, 23, 25, 26]
        group_column = 1

    learn_params = {
        'task_type': task_type,
        'devices': '0',
        'iterations': 20,
        'learning_rate': 0.5,
        'logging_level': 'Silent',
        'loss_function': loss_function,
        'boosting_type': 'Plain',
        'allow_const_label': True
    }
    evaluator = CatboostEvaluation(
        train_file,
        fold_size=50,
        fold_count=2,
        column_description=cd_file,
        group_column=group_column,
        partition_random_seed=0
    )
    results = evaluator.eval_features(
        learn_config=learn_params,
        features_to_eval=features_to_eval,
        eval_type=EvalType(eval_type),
        eval_metrics=eval_metrics
    )

    canonical_files = []
    for metric_name, metric_results in results.get_results().items():
        name = 'eval_results_{}.csv'.format(metric_name.replace(':', '_'))
        eval_results_file_name = test_output_path(name)
        metric_results.get_baseline_comparison().to_csv(eval_results_file_name)
        canonical_files.append(local_canonical_file(eval_results_file_name))

    return canonical_files


def test_metric_period_with_verbose_true():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(dict(iterations=16, metric_period=4))

    tmpfile = test_output_path('tmpfile')
    with open(tmpfile, 'w') as cout:
        model.fit(pool, verbose=True, log_cout=cout)

    assert (_count_lines(tmpfile) == 6)


def test_eval_features_with_file_header():
    learn_params = {
        'iterations': 20,
        'learning_rate': 0.5,
        'logging_level': 'Silent',
        'loss_function': 'RMSE',
        'boosting_type': 'Plain',
        'allow_const_label': True
    }

    evaluator = CatboostEvaluation(
        BLACK_FRIDAY_TRAIN_FILE,
        fold_size=50,
        fold_count=2,
        column_description=BLACK_FRIDAY_CD_FILE,
        has_header=True,
        partition_random_seed=0
    )

    results = evaluator.eval_features(
        learn_config=learn_params,
        features_to_eval=[6, 7, 8],
        eval_type=EvalType('SeqAdd'),
        eval_metrics=['RMSE']
    )

    eval_results_file_name = test_output_path('eval_results.csv')
    logloss_result = results.get_metric_results('RMSE')
    comparison_results = logloss_result.get_baseline_comparison()
    comparison_results.to_csv(eval_results_file_name)

    return local_canonical_file(eval_results_file_name)


def test_compute_options():
    data_meta_info = DataMetaInfo(
        object_count=100000,
        feature_count=10,
        max_cat_features_uniq_values_on_learn=0,
        target_stats=TargetStats(min_value=0, max_value=1),
        has_pairs=False
    )

    options = compute_training_options(
        options={'thread_count': 1},
        train_meta_info=data_meta_info,
        test_meta_info=data_meta_info,
    )

    options_file_name = test_output_path('options.json')
    with open(options_file_name, 'w') as f:
        json.dump(options, f, indent=4, sort_keys=True)

    return local_canonical_file(options_file_name)


COMBINATION_TYPES = [
    'no',
    'init_model_same_loss',
    'init_model_different_losses',
    'sum_models_same_loss',
    'sum_models_different_losses'
]


@pytest.mark.parametrize(
    'combination',
    COMBINATION_TYPES,
    ids=['combination=%s' % combination for combination in COMBINATION_TYPES]
)
def test_feature_statistics(combination):
    n_features = 3
    n_samples = 500
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    if combination == 'no':
        model = CatBoostRegressor(iterations=200)
        model.fit(X, y, silent=True)
    elif combination == 'init_model_same_loss':
        init_model = CatBoostRegressor(iterations=100)
        init_model.fit(X, y, silent=True)
        model = CatBoostRegressor(iterations=100)
        model.fit(X, y, silent=True, init_model=init_model)
    elif combination == 'init_model_different_losses':
        init_model = CatBoostRegressor(iterations=50, loss_function='Lq:q=3')
        init_model.fit(X, y, silent=True)
        model = CatBoostRegressor(iterations=120)
        model.fit(X, y, silent=True, init_model=init_model)
    elif combination == 'sum_models_same_loss':
        model0 = CatBoostRegressor(iterations=100)
        model0.fit(X, y, silent=True)
        model1 = CatBoostRegressor(iterations=100)
        model1.fit(X, y, silent=True)
        model = sum_models([model0, model1])
    else:  # combination == 'sum_models_different_losses'
        model0 = CatBoostRegressor(iterations=50, loss_function='Lq:q=3')
        model0.fit(X, y, silent=True)
        model1 = CatBoostRegressor(iterations=120)
        model1.fit(X, y, silent=True)
        model = sum_models([model0, model1])

    feature_num = 0
    model.calc_feature_statistics({'learn': X, 'test': X}, {'learn': y, 'test': y}, feature_num, plot=False)
    res = model.calc_feature_statistics(X, y, feature_num, plot=False)

    def mean_per_bin(res, feature_num, data):
        return np.array([data[np.digitize(X[:, feature_num], res['borders']) == bin_num].mean()
                         for bin_num in range(len(res['borders']) + 1)])

    def vary_feature(res, feature_num, bucket_num, X):
        if bucket_num == 0:
            bucket_value = np.min(X[:, feature_num])
        elif bucket_num == len(res['borders']):
            bucket_value = np.max(X[:, feature_num])
        else:
            bucket_value = np.mean(res['borders'][bucket_num-1:bucket_num+1])
        return np.hstack((X[:, :feature_num], np.tile(bucket_value, (n_samples, 1)), X[:, feature_num + 1:]))

    assert (np.alltrue(np.array(res['binarized_feature']) == np.digitize(X[:, feature_num], res['borders'])))
    assert (res['objects_per_bin'].sum() == X.shape[0])
    assert (
        np.alltrue(
            np.unique(np.digitize(X[:, feature_num], res['borders']), return_counts=True)[1] == res['objects_per_bin']
        )
    )
    assert (np.allclose(res['mean_prediction'], mean_per_bin(res, feature_num, model.predict(X)), atol=1e-4))
    assert (np.allclose(res['mean_target'], mean_per_bin(res, feature_num, y), atol=1e-4))
    assert (
        np.allclose(
            res['predictions_on_varying_feature'],
            list(
                np.mean(model.predict(vary_feature(res, feature_num, bucket_num, X)))
                for bucket_num in range(len(res['borders']) + 1)
            ),
            atol=1e-4
        )
    )


def test_prediction_plot():
    preds_path = test_output_path('predictions.json')
    n_features = 3
    n_samples = 500
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    model = CatBoostRegressor(iterations=200)
    model.fit(X, y, silent=True)

    res, _ = model.plot_predictions(data=X[:2, ], features_to_change=[0, 1], plot=False)
    json.dump(res, open(preds_path, 'w'))
    return local_canonical_file(preds_path)


def test_binclass_with_nontrivial_classes():
    catboost_training_path = test_output_path('catboost_training.json')
    model = CatBoostClassifier(iterations=10, loss_function='Logloss')
    model.set_params(json_log=catboost_training_path)
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    y = [1, 2, 1]
    model.fit(X, y)
    json_log = json.load(open(remove_time_from_json(catboost_training_path)))
    json_log['iterations'] = [
        dict({'iteration': row['iteration'], 'learn': [np.round(row['learn'][0], decimals=4)]})
        for row in json_log['iterations']]
    json.dump(json_log, open(catboost_training_path, 'w'))
    return local_canonical_file(catboost_training_path)


def test_loss_function_auto_set():
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    bin_y = [1, 2, 1]
    multi_y = [1, 2, 3]

    def test_one_case(params, X, y, expected_loss):
        model = CatBoostClassifier(**params).fit(X, y)
        assert model.get_all_params()['loss_function'] == expected_loss

        model = CatBoostClassifier(**params).fit(Pool(X, y))
        assert model.get_all_params()['loss_function'] == expected_loss

    test_one_case({'iterations': 10}, X, bin_y, 'Logloss')
    test_one_case({'iterations': 10}, X, multi_y, 'MultiClass')
    test_one_case({'iterations': 10, 'target_border': 1.5}, X, multi_y, 'Logloss')


DATASET_TARGET_TYPES = ['binarized', 'not_binarized', 'multiclass']


@pytest.mark.parametrize(
    'dataset_target_type',
    DATASET_TARGET_TYPES,
    ids=['dataset_target_type=%s' % dtt for dtt in DATASET_TARGET_TYPES]
)
def test_loss_function_auto_set_from_file(dataset_target_type):
    dataset_path = {
        'binarized': 'adult',
        'not_binarized': 'adult_not_binarized',
        'multiclass': 'cloudness_small'
    }[dataset_target_type]

    params = {'iterations': 3}
    if dataset_target_type == 'not_binarized':
        params['target_border'] = 0.5

    model = CatBoostClassifier(**params).fit(
        data_file(dataset_path, 'train_small'),
        column_description=data_file(dataset_path, 'train.cd')
    )

    expected_loss = {
        'binarized': 'Logloss',
        'not_binarized': 'Logloss',
        'multiclass': 'MultiClass'
    }[dataset_target_type]

    assert model.get_all_params()['loss_function'] == expected_loss


PROBLEM_TYPES = ['binclass', 'multiclass', 'regression', 'ranking']


def get_params_for_problem_type(problem_type):
    if problem_type == 'binclass':
        return {
            'loss_function': 'Logloss',
            'train_path': TRAIN_FILE,
            'test_path': TEST_FILE,
            'cd_path': CD_FILE,
            'boosting_types': BOOSTING_TYPE
        }
    elif problem_type == 'multiclass':
        return {
            'loss_function': 'MultiClass',
            'train_path': CLOUDNESS_TRAIN_FILE,
            'test_path': CLOUDNESS_TEST_FILE,
            'cd_path': CLOUDNESS_CD_FILE,
            'boosting_types': BOOSTING_TYPE
        }
    elif problem_type == 'regression':
        return {
            'loss_function': 'RMSE',
            'train_path': TRAIN_FILE,
            'test_path': TEST_FILE,
            'cd_path': CD_FILE,
            'boosting_types': BOOSTING_TYPE
        }
    elif problem_type == 'ranking':
        return {
            'loss_function': 'YetiRankPairwise',
            'train_path': QUERYWISE_TRAIN_FILE,
            'test_path': QUERYWISE_TEST_FILE,
            'cd_path': QUERYWISE_CD_FILE,
            'boosting_types': ['Plain']
        }
    else:
        raise Exception('Unsupported problem_type: %s' % problem_type)


@pytest.mark.parametrize('problem_type', PROBLEM_TYPES, ids=['problem_type=%s' % pt for pt in PROBLEM_TYPES])
def test_continue_learning_with_same_params(problem_type):
    params = get_params_for_problem_type(problem_type)

    train_pool = Pool(params['train_path'], column_description=params['cd_path'])

    for boosting_type in params['boosting_types']:
        train_params = {
            'task_type': 'CPU',  # TODO(akhropov): GPU support
            'loss_function': params['loss_function'],
            'boosting_type': boosting_type,
            'learning_rate': 0.3  # fixed, because automatic value depends on number of iterations
        }

        iterations_list = [5, 7, 10]
        total_iterations = sum(iterations_list)

        def train_model(iterations, init_model=None):
            local_params = train_params
            local_params['iterations'] = iterations
            model = CatBoost(local_params)
            model.fit(train_pool, init_model=init_model)
            return model

        total_model = train_model(total_iterations)

        incremental_model = None
        for iterations in iterations_list:
            incremental_model = train_model(iterations, incremental_model)

        assert total_model == incremental_model


PARAM_SETS = ['iterations,learning_rate', 'iterations,depth,rsm']


@pytest.mark.parametrize('problem_type', PROBLEM_TYPES, ids=['problem_type=%s' % pt for pt in PROBLEM_TYPES])
@pytest.mark.parametrize('param_set', PARAM_SETS, ids=['param_set=%s' % pt for pt in PARAM_SETS])
def test_continue_learning_with_changing_params(problem_type, param_set):
    params = get_params_for_problem_type(problem_type)

    train_pool = Pool(params['train_path'], column_description=params['cd_path'])
    test_pool = Pool(params['test_path'], column_description=params['cd_path'])

    if param_set == 'iterations,learning_rate':
        updated_params_list = [
            {'iterations': 5, 'learning_rate': 0.3},
            {'iterations': 2, 'learning_rate': 0.1},
            {'iterations': 3, 'learning_rate': 0.2},
        ]
    elif param_set == 'iterations,depth,rsm':
        updated_params_list = [
            {'iterations': 2, 'depth': 3, 'rsm': 1.0},
            {'iterations': 4, 'depth': 7, 'rsm': 0.2},
            {'iterations': 3, 'depth': 6, 'rsm': 0.5},
        ]

    canonical_files = []

    for boosting_type in params['boosting_types']:
        train_params = {
            'task_type': 'CPU',  # TODO(akhropov): GPU support
            'loss_function': params['loss_function'],
            'boosting_type': boosting_type,
        }

        def train_model(updated_params, init_model=None):
            local_params = train_params
            local_params.update(updated_params)
            model = CatBoost(local_params)
            model.fit(train_pool, init_model=init_model)
            return model

        model = None
        for updated_params in updated_params_list:
            model = train_model(updated_params, model)

        pred = model.predict(test_pool)
        preds_path = test_output_path('predictions_for_boosting_type_%s.txt' % boosting_type)
        np.savetxt(preds_path, np.array(pred), fmt='%.8f')
        canonical_files.append(local_canonical_file(preds_path))

    return canonical_files


SAMPLES_AND_FEATURES_FOR_CONTINUATION = [
    ('same', 'more'),
    ('more', 'same'),
    ('new', 'same'),
    ('more', 'more'),
    ('new', 'more'),
]


@pytest.mark.parametrize(
    'samples,features',
    SAMPLES_AND_FEATURES_FOR_CONTINUATION,
    ids=[
        'samples=%s,features=%s' % (samples, features)
        for (samples, features) in SAMPLES_AND_FEATURES_FOR_CONTINUATION
    ]
)
def test_continue_learning_with_changing_dataset(samples, features):
    all_df = read_csv(TRAIN_FILE, header=None, delimiter='\t')
    all_labels = Series(all_df.iloc[:, TARGET_IDX])
    all_df.drop([TARGET_IDX], axis=1, inplace=True)
    all_features_df = all_df

    if samples == 'same':
        features_df_1 = all_features_df.copy()
        labels_1 = all_labels.copy()
        features_df_2 = all_features_df
        labels_2 = all_labels
    elif samples == 'more':
        features_df_1 = all_features_df.head(70).copy()
        labels_1 = all_labels.head(70).copy()
        features_df_2 = all_features_df
        labels_2 = all_labels
    elif samples == 'new':
        features_df_1 = all_features_df.head(60).copy()
        labels_1 = all_labels.head(60).copy()
        features_df_2 = all_features_df.tail(len(all_features_df) - 60).copy()
        labels_2 = all_labels.tail(len(all_features_df) - 60).copy()

    if features == 'more':
        features_df_1.drop(features_df_1.columns[-5:], axis=1, inplace=True)
        cat_features_1 = list(filter(lambda i: i < len(features_df_1.columns), CAT_FEATURES))
    else:
        cat_features_1 = CAT_FEATURES
    cat_features_2 = CAT_FEATURES

    train_params = {
        'task_type': 'CPU',  # TODO(akhropov): GPU support
        'loss_function': 'Logloss',
        'boosting_type': 'Plain',
        'learning_rate': 0.3  # fixed, because automatic value depends on number of iterations
    }

    def train_model(train_features_df, train_labels, cat_features, iterations, init_model=None):
        local_params = train_params
        local_params['iterations'] = iterations
        model = CatBoost(local_params)
        print('cat_features', cat_features)
        model.fit(X=train_features_df, y=train_labels, cat_features=cat_features, init_model=init_model)
        return model

    model1 = train_model(features_df_1, labels_1, cat_features_1, iterations=5)
    model2 = train_model(features_df_2, labels_2, cat_features_2, iterations=4, init_model=model1)

    pred = model2.predict(Pool(TEST_FILE, column_description=CD_FILE))
    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(pred), fmt='%.8f')

    return local_canonical_file(preds_path)


def test_equal_feature_names():
    train_data = [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]
    with pytest.raises(CatBoostError):
        Pool(train_data, feature_names=['first', 'second', 'third', 'fourth', 'second', 'sixth'])


@pytest.mark.parametrize('feature_names', [0, 'text'])
def test_not_sequence_feature_names(feature_names):
    train_data = DataFrame({'text_f': ['у попа была собака',
                                       'он ее любил',
                                       'she ate a piece of meat',
                                       'he killed her...']})
    with pytest.raises(CatBoostError):
        Pool(train_data, feature_names=feature_names)


@pytest.mark.parametrize('variance_power', [1.001, 1.42, 1.8, 1.999])
def test_tweedie_loss_on_gpu(task_type, variance_power):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    model = CatBoost(
        {
            'iterations': 10,
            'loss_function': 'Tweedie:variance_power=' + str(variance_power),
            'task_type': task_type,
            'devices': '0-7'
        }
    )

    model.fit(train_pool)
    model.predict(test_pool)


@pytest.mark.parametrize('delta', [0.01, 0.5, 1, 1.5])
def test_huber_loss_on_gpu(task_type, delta):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    model = CatBoost(
        {
            'iterations': 10,
            'loss_function': 'Huber:delta=' + str(delta),
            'task_type': task_type,
            'devices': '0-7'
        }
    )

    model.fit(train_pool)
    model.predict(test_pool)


class TestModelWithoutParams(object):

    @pytest.fixture(
        params=[
            ('cut-info', 'RMSE'),
            ('cut-params', 'RMSE'),
            ('cut-info', 'QueryRMSE'),
            ('cut-params', 'QueryRMSE'),
        ],
        ids=lambda param: '-'.join(param),
    )
    def model_etc(self, request):
        cut, loss = request.param
        model_json = test_output_path('model.json')
        train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
        test_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
        model = CatBoost(dict(iterations=16, loss_function=loss))
        model.fit(train_pool, verbose=True)
        model.save_model(model_json, format='json')
        data = json.load(open(model_json))
        if cut == 'cut-info':
            data.pop('model_info')
        if cut == 'cut-params':
            data['model_info'].pop('params')
        json.dump(data, open(model_json, 'wt'))
        model.load_model(model_json, format='json')
        return model, train_pool, test_pool

    def test_ostr(self, model_etc):
        model, train_pool, test_pool = model_etc
        with pytest.raises(CatBoostError):
            model.get_object_importance(test_pool, train_pool, top_size=10)

    @pytest.mark.parametrize('should_fail,fstr_type', [
        (False, 'FeatureImportance'),
        (False, 'PredictionValuesChange'),
        (True, 'LossFunctionChange'),
        (False, 'ShapValues'),
    ])
    def test_fstr(self, model_etc, fstr_type, should_fail):
        model, train_pool, test_pool = model_etc
        if should_fail:
            with pytest.raises(CatBoostError):
                model.get_feature_importance(type=fstr_type, data=train_pool)
        else:
            model.get_feature_importance(type=fstr_type, data=train_pool)

    def test_prediction_values_fstr_change_on_different_pools(self):
        train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
        test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)
        model = CatBoost(dict(iterations=16, loss_function='RMSE'))
        model.fit(train_pool)
        train_fstr = model.get_feature_importance(type='PredictionValuesChange', data=train_pool)
        test_fstr = model.get_feature_importance(type='PredictionValuesChange', data=test_pool)
        assert not np.array_equal(train_fstr, test_fstr)

    @pytest.fixture(params=['not-trained', 'collapsed'])
    def model_no_trees(self, request):
        train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
        test_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
        model = CatBoost(dict(iterations=16, loss_function='RMSE'))
        if request.param == 'collapsed':
            model.fit(train_pool, verbose=True)
            model.shrink(0, 0)
        else:
            pass  # not-trained
        return model, train_pool, test_pool

    @pytest.mark.parametrize('fstr_type', ['FeatureImportance', 'PredictionValuesChange'])
    def test_fstr_no_trees(self, model_no_trees, fstr_type):
        model, train_pool, test_pool = model_no_trees
        with pytest.raises(CatBoostError):
            model.get_feature_importance(type=fstr_type, data=train_pool)

    def test_ostr_no_trees(self, model_no_trees):
        model, train_pool, test_pool = model_no_trees
        with pytest.raises(CatBoostError):
            model.get_object_importance(test_pool, train_pool, top_size=10)


def test_get_all_params():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(dict(iterations=16, thread_count=4))
    model.fit(pool, verbose=True)

    options = model.get_all_params()

    model2 = CatBoost(options)
    model2.fit(pool, verbose=True)

    assert all(model.predict(pool) == model2.predict(pool))

    options_file = test_output_path('options.json')
    with open(options_file, 'w') as f:
        json.dump(options, f, indent=4, sort_keys=True)

    return local_canonical_file(options_file)


@pytest.mark.parametrize('metric', ['MAE', 'RMSE', 'CrossEntropy', 'AUC'])
def test_weights_in_eval_metric(metric):
    predictions = [1, 1, 2, 3, 1, 4, 1, 2, 3, 4]
    label = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0]
    weights = [1, 0.75, 2.39, 0.5, 1, 1.3, 0.7, 1, 1.1, 0.67]
    result_with_no_weights = eval_metric(label, predictions, metric)
    result_with_weights = eval_metric(label, predictions, metric, weights)
    assert not np.isclose(result_with_no_weights, result_with_weights)


@pytest.mark.parametrize('metric_name', ['Accuracy', 'Precision', 'Recall', 'F1'])
@pytest.mark.parametrize('proba_border', [0.25, 0.55, 0.75])
def test_prediction_border_in_eval_metric(metric_name, proba_border):
    metric_with_border = "{metric_name}:proba_border={proba_border}".format(
        metric_name=metric_name,
        proba_border=proba_border
    )
    metric_no_params = metric_name
    prediction_probas = np.array([0.06314072, 0.77672081, 0.00885847, 0.87585758, 0.70030717,
                                  0.42210464, 0.00698532, 0.08357631, 0.87840924, 0.71889332,
                                  0.27881971, 0.03881581, 0.12708005, 0.04321602, 0.46778848,
                                  0.34862325, 0.95195515, 0.08093261, 0.79914953, 0.50639467])
    label = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    predictions = scipy.special.logit(prediction_probas)

    # We test that the metrics are correctly rounding the probability. Thus, the results with a custom border should
    # be identical if we just round the predictions before evaluating the metric with the given threshold.
    binarized_predictions = np.where(prediction_probas >= proba_border, 1.0, -1.0)

    result_with_border = eval_metric(label, predictions, metric_with_border)
    result_expected = eval_metric(label, binarized_predictions, metric_no_params)

    assert result_with_border == result_expected


def test_dataframe_with_custom_index():
    np.random.seed(0)
    X = DataFrame(np.random.randint(0, 9, (3, 2)), index=[55, 675, 34])
    X[0] = X[0].astype('category')
    y = X[1]

    model = CatBoost(dict(iterations=10))
    model.fit(X, y, cat_features=[0])


@pytest.mark.parametrize('features_type', [
    'numerical_only',
    'numerical_and_categorical',
])
def test_load_model_from_snapshot(features_type):
    model = CatBoost(dict(iterations=16, thread_count=4))
    filename = test_output_path('snapshot.bak')
    try:
        os.remove(filename)
    except OSError:
        pass
    if features_type == 'numerical_only':
        pool = Pool(data=[[0, 1, 2, 3],
                          [1, 2, 3, 4],
                          [5, 6, 7, 8]],
                    label=[1, 1, -1])
    else:
        df = DataFrame(data={'col1': ['a', 'b', 'c', 'd'], 'col2': [1, 1, 1, 1], 'col3': [2, 3, 4, 5]})
        pool = Pool(data=df,
                    label=[1, 1, -1, -1],
                    cat_features=['col1'])
    model.fit(pool, verbose=True, save_snapshot=True, snapshot_file=filename)
    if features_type == 'numerical_only':
        model_from_snapshot = CatBoost(dict(iterations=16, thread_count=4))
        model_from_snapshot.load_model(filename, format='CpuSnapshot')
        assert model == model_from_snapshot
    else:
        with pytest.raises(CatBoostError):
            model.load_model(filename, format='CpuSnapshot')


def test_regress_with_per_float_feature_binarization_param(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    per_float_feature_quantization_list = ['0:nan_mode=Forbidden,border_count=2,border_type=GreedyLogSum',
                                           '1:nan_mode=Forbidden,border_count=3,border_type=GreedyLogSum',
                                           '2:nan_mode=Forbidden,border_count=8,border_type=GreedyLogSum',
                                           '3:nan_mode=Forbidden,border_count=14,border_type=GreedyLogSum',
                                           '4:nan_mode=Forbidden,border_count=31,border_type=GreedyLogSum',
                                           '5:nan_mode=Forbidden,border_count=32,border_type=GreedyLogSum']
    model = CatBoostRegressor(iterations=2,
                              learning_rate=0.03,
                              task_type=task_type,
                              devices='0',
                              per_float_feature_quantization=per_float_feature_quantization_list)
    model.fit(train_pool)
    assert (model.is_fitted())
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_pairs_without_groupid():
    model = CatBoost(params={'loss_function': 'PairLogit', 'iterations': 10, 'thread_count': 8})
    pairs = read_csv(QUERYWISE_TRAIN_PAIRS_FILE, delimiter='\t', header=None)
    df = read_csv(QUERYWISE_TRAIN_FILE, delimiter='\t', header=None)
    train_target = df.loc[:, 2]
    train_data = df.drop([0, 1, 2, 3, 4], axis=1).astype(np.float32)
    model.fit(train_data, train_target, pairs=pairs)
    return local_canonical_file(remove_time_from_json(JSON_LOG_PATH))


def test_groupwise_sampling_without_groups(task_type):
    params = {
        'task_type': task_type,
        'iterations': 10,
        'thread_count': 4,
        'bootstrap_type': 'Bernoulli',
        'sampling_unit': 'Group',
        'subsample': 0.5
    }
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(params=params)
    try:
        model.fit(train_pool)
    except CatBoostError:
        return

    assert False


def test_convert_to_asymmetric(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    train_params = {
        'iterations': 10,
        'learning_rate': 0.03,
        'task_type': task_type
    }
    model = CatBoost(train_params)
    model.fit(train_pool)
    model._convert_to_asymmetric_representation()


def get_quantized_path(fname):
    return 'quantized://' + fname


def test_pool_is_quantized():

    quantized_pool = Pool(data=get_quantized_path(QUANTIZED_TRAIN_FILE), column_description=QUANTIZED_CD_FILE)
    assert quantized_pool.is_quantized()

    raw_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    assert not raw_pool.is_quantized()


def test_quantized_pool_with_all_features_ignored():
    quantized_pool = Pool(data=get_quantized_path(QUANTIZED_TRAIN_FILE), column_description=QUANTIZED_CD_FILE)
    with pytest.raises(CatBoostError):
        CatBoostClassifier(ignored_features=list(range(100))).fit(quantized_pool)


@pytest.fixture(params=[
    ('adult', TRAIN_FILE, TEST_FILE, CD_FILE, 'Logloss'),
    ('higgs', HIGGS_TRAIN_FILE, HIGGS_TEST_FILE, HIGGS_CD_FILE, 'Logloss'),
    ('querywise', QUERYWISE_TRAIN_FILE, QUERYWISE_TEST_FILE, QUERYWISE_CD_FILE, 'RMSE'),
], ids=[
    'adult',
    'higgs',
    'querywise'
])
def train_on_raw_and_quantized_data_params_fixture(request):
    pool_name, train_pool_file, test_pool_file, column_description, loss_function = request.param
    borders_file = test_output_path('{}_borders.dat'.format(pool_name))
    train_pool = Pool(train_pool_file, column_description=column_description)
    test_pool = Pool(test_pool_file, column_description=column_description)
    train_quantized_pool = Pool(train_pool_file, column_description=column_description)
    test_quantized_pool = Pool(test_pool_file, column_description=column_description)
    params = {
        'task_type': 'CPU',
        'loss_function': loss_function,
        'iterations': 5,
        'depth': 4,
    }

    train_quantized_pool.quantize()
    train_quantized_pool.save_quantization_borders(borders_file)
    test_quantized_pool.quantize(input_borders=borders_file)

    assert (train_quantized_pool.is_quantized())
    assert (test_quantized_pool.is_quantized())

    return {
        'pool_name': pool_name,
        'train_pool': train_pool,
        'test_pool': test_pool,
        'train_quantized_pool': train_quantized_pool,
        'test_quantized_pool': test_quantized_pool,
        'loss_function': loss_function,
        'params': params,
    }


@pytest.fixture()
def models_trained_on_raw_and_quantized_data_fixture(train_on_raw_and_quantized_data_params_fixture):
    train_pool = train_on_raw_and_quantized_data_params_fixture['train_pool']
    test_pool = train_on_raw_and_quantized_data_params_fixture['test_pool']
    train_quantized_pool = train_on_raw_and_quantized_data_params_fixture['train_quantized_pool']
    test_quantized_pool = train_on_raw_and_quantized_data_params_fixture['test_quantized_pool']
    loss_function = train_on_raw_and_quantized_data_params_fixture['loss_function']
    params = train_on_raw_and_quantized_data_params_fixture['params']

    model = CatBoost(params=params)
    model_fitted_with_quantized_pool = CatBoost(params=params)

    model.fit(train_pool)
    model_fitted_with_quantized_pool.fit(train_quantized_pool)

    return {
        'pool_name': train_on_raw_and_quantized_data_params_fixture['pool_name'],
        'train_pool': train_pool,
        'test_pool': test_pool,
        'train_quantized_pool': train_quantized_pool,
        'test_quantized_pool': test_quantized_pool,
        'model': model,
        'model_fitted_with_quantized_pool': model_fitted_with_quantized_pool,
        'loss_function': loss_function,
    }


def test_quantized_pool_cv(train_on_raw_and_quantized_data_params_fixture):
    train_quantized_pool = train_on_raw_and_quantized_data_params_fixture['train_quantized_pool']
    params = train_on_raw_and_quantized_data_params_fixture['params']
    results_with_quantized_pool = cv(train_quantized_pool, params).round(decimals=4)
    preds_path = test_output_path(CV_CSV_PATH)
    results_with_quantized_pool.to_csv(preds_path)
    return local_canonical_file(preds_path)


def test_quantized_pool_train_predict(train_on_raw_and_quantized_data_params_fixture):
    train_pool = train_on_raw_and_quantized_data_params_fixture['train_pool']
    test_pool = train_on_raw_and_quantized_data_params_fixture['test_pool']
    train_quantized_pool = train_on_raw_and_quantized_data_params_fixture['train_quantized_pool']
    test_quantized_pool = train_on_raw_and_quantized_data_params_fixture['test_quantized_pool']
    params = train_on_raw_and_quantized_data_params_fixture['params']

    model = train(train_pool, params=params)
    model_fitted_with_quantized_pool = train(train_quantized_pool, params=params)

    predictions = model.predict(test_pool)
    predictions_with_quantized_pool = model_fitted_with_quantized_pool.predict(test_quantized_pool)
    assert np.all(predictions == predictions_with_quantized_pool)


def test_quantized_pool_predict(models_trained_on_raw_and_quantized_data_fixture):
    test_pool = models_trained_on_raw_and_quantized_data_fixture['test_pool']
    test_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['test_quantized_pool']
    model = models_trained_on_raw_and_quantized_data_fixture['model']
    model_fitted_with_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['model_fitted_with_quantized_pool']

    predictions = model.predict(test_pool)
    predictions_with_quantized_pool = model_fitted_with_quantized_pool.predict(test_quantized_pool)
    assert np.all(predictions == predictions_with_quantized_pool)


@pytest.mark.parametrize('pool_type', ['train', 'test'])
def test_quantized_pool_calc_feature_statistics(models_trained_on_raw_and_quantized_data_fixture, pool_type):
    pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_pool']
    quantized_pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_quantized_pool']
    model = models_trained_on_raw_and_quantized_data_fixture['model']
    model_fitted_with_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['model_fitted_with_quantized_pool']

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def serialize_feature_statistics(stats):
        return json.dumps(stats, cls=NumpyEncoder, sort_keys=True)

    if models_trained_on_raw_and_quantized_data_fixture['pool_name'] == 'adult':
        # TODO(akhropov): calc_feature_statistics does not support CTR features
        with pytest.raises(CatBoostError):
            model.calc_feature_statistics(pool, plot=False)
    else:
        stats = model.calc_feature_statistics(pool, plot=False)
        stats_with_quantized_pool = model_fitted_with_quantized_pool.calc_feature_statistics(
            quantized_pool,
            plot=False
        )
        assert serialize_feature_statistics(stats) == serialize_feature_statistics(stats_with_quantized_pool)
        model.calc_feature_statistics({'learn': pool, 'test': pool}, plot=False)


@pytest.mark.parametrize('pool_type', ['train', 'test'])
def test_quantized_pool_eval_metrics(models_trained_on_raw_and_quantized_data_fixture, pool_type):
    pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_pool']
    quantized_pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_quantized_pool']
    model = models_trained_on_raw_and_quantized_data_fixture['model']
    model_fitted_with_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['model_fitted_with_quantized_pool']
    loss_function = models_trained_on_raw_and_quantized_data_fixture['loss_function']

    metrics = model.eval_metrics(pool, metrics=[loss_function])
    metrics_with_quantized_pool = model_fitted_with_quantized_pool.eval_metrics(quantized_pool, metrics=[loss_function])
    assert metrics == metrics_with_quantized_pool


@pytest.mark.parametrize('pool_type', ['train', 'test'])
@pytest.mark.parametrize('fstr_type', [
    EFstrType.PredictionValuesChange,
    EFstrType.FeatureImportance,
    EFstrType.ShapValues,
    EFstrType.ShapInteractionValues,
    EFstrType.Interaction
])
def test_quantized_pool_get_feature_importance(models_trained_on_raw_and_quantized_data_fixture, pool_type, fstr_type):
    pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_pool']
    quantized_pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_quantized_pool']
    model = models_trained_on_raw_and_quantized_data_fixture['model']
    model_fitted_with_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['model_fitted_with_quantized_pool']

    fstr = model.get_feature_importance(data=pool, type=fstr_type)
    fstr_with_quantized_pool = \
        model_fitted_with_quantized_pool.get_feature_importance(data=quantized_pool, type=fstr_type)
    assert np.all(fstr == fstr_with_quantized_pool)


def test_quantized_pool_get_object_importance(models_trained_on_raw_and_quantized_data_fixture):
    train_pool = models_trained_on_raw_and_quantized_data_fixture['train_pool']
    test_pool = models_trained_on_raw_and_quantized_data_fixture['test_pool']
    train_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['train_quantized_pool']
    test_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['test_quantized_pool']
    model = models_trained_on_raw_and_quantized_data_fixture['model']
    model_fitted_with_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['model_fitted_with_quantized_pool']

    importance = model.get_object_importance(test_pool, train_pool)
    importance_with_quantized_pool = \
        model_fitted_with_quantized_pool.get_object_importance(test_quantized_pool, train_quantized_pool)
    assert np.all(np.array(importance) == np.array(importance_with_quantized_pool))


@pytest.mark.parametrize('pool_type', ['train', 'test'])
def test_quantized_pool_select_threshold(models_trained_on_raw_and_quantized_data_fixture, pool_type):
    pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_pool']
    quantized_pool = models_trained_on_raw_and_quantized_data_fixture[pool_type + '_quantized_pool']
    model = models_trained_on_raw_and_quantized_data_fixture['model']
    model_fitted_with_quantized_pool = models_trained_on_raw_and_quantized_data_fixture['model_fitted_with_quantized_pool']

    threshold = select_threshold(model, pool)
    threshold_with_quantized_pool = select_threshold(model_fitted_with_quantized_pool, quantized_pool)
    assert threshold == threshold_with_quantized_pool


def check_save_quantized_pool(train_pool, train_pool_copy, test_pool, params):
    feature_names = train_pool.get_feature_names()

    train_quantized_pool = train_pool_copy
    train_quantized_pool.quantize()

    assert (train_quantized_pool.is_quantized())

    train_quantized_pool.save(OUTPUT_QUANTIZED_POOL_PATH)

    train_quantized_load_pool = Pool(get_quantized_path(OUTPUT_QUANTIZED_POOL_PATH))

    model = CatBoost(params=params)
    model_fitted_with_load_quantized_pool = CatBoost(params=params)

    model.fit(train_pool)
    predictions1 = model.predict(test_pool)

    model_fitted_with_load_quantized_pool.fit(train_quantized_load_pool)
    predictions2 = model_fitted_with_load_quantized_pool.predict(test_pool)

    loaded_feature_names = train_quantized_load_pool.get_feature_names()
    assert (feature_names == loaded_feature_names)
    assert all(predictions1 == predictions2)


FEATURES_TYPES = ['num', 'cat', 'num_and_cat']


@pytest.mark.parametrize(
    'features_types',
    FEATURES_TYPES,
    ids=['features_types=%s' % ft for ft in FEATURES_TYPES]
)
def test_save_quantized_pool_with_source_pool_from_dataframe(features_types):
    n_train_samples = 200
    n_test_samples = 50

    if features_types == 'num':
        n_num_features = 5
        n_cat_features = 0
    elif features_types == 'cat':
        n_num_features = 0
        n_cat_features = 5
    elif features_types == 'num_and_cat':
        n_num_features = 4
        n_cat_features = 5

    train_data = generate_dataset_with_num_and_cat_features(
        n_samples=n_train_samples,
        n_num_features=n_num_features,
        n_cat_features=n_cat_features,
        labels=[0.0, 0.12, 1.0],
    )
    test_data = generate_dataset_with_num_and_cat_features(
        n_samples=n_test_samples,
        n_num_features=n_num_features,
        n_cat_features=n_cat_features,
        labels=[0.0, 0.12, 1.0],
    )

    check_save_quantized_pool(
        train_pool=Pool(train_data[0], train_data[1]),
        test_pool=Pool(test_data[0], test_data[1]),
        train_pool_copy=Pool(train_data[0], train_data[1]),
        params={
            'task_type': 'CPU',
            'loss_function': 'RMSE',
            'iterations': 5,
            'depth': 4,
            'dev_efb_max_buckets': 0
        }
    )


@pytest.mark.parametrize(
    'features_types',
    ['num', 'num_and_cat'],
    ids=['features_types=%s' % ft for ft in ['num', 'num_and_cat']]
)
def test_save_quantized_pool_with_source_pool_from_files(features_types):
    if features_types == 'num':
        dataset_name = 'higgs'
    elif features_types == 'num_and_cat':
        dataset_name = 'adult'

    train_path = data_file(dataset_name, 'train_small')
    test_path = data_file(dataset_name, 'test_small')
    cd_path = data_file(dataset_name, 'train.cd')

    check_save_quantized_pool(
        train_pool=Pool(train_path, column_description=cd_path),
        test_pool=Pool(test_path, column_description=cd_path),
        train_pool_copy=Pool(train_path, column_description=cd_path),
        params={
            'task_type': 'CPU',
            'loss_function': 'Logloss',
            'iterations': 5,
            'depth': 4,
            'dev_efb_max_buckets': 0
        }
    )


# returns dict with 'train_file', 'test_file', 'data_files_have_header', 'cd_file', 'loss_function' keys
def get_dataset_specification_for_sparse_input_tests():
    return {
        'adult': {
            'train_file': TRAIN_FILE,
            'test_file': TEST_FILE,
            'data_files_have_header': False,
            'cd_file': CD_FILE,
            'loss_function': 'Logloss'
        },
        'airlines_5k': {
            'train_file': AIRLINES_5K_TRAIN_FILE,
            'test_file': AIRLINES_5K_TEST_FILE,
            'data_files_have_header': True,
            'cd_file': AIRLINES_5K_CD_FILE,
            'loss_function': 'Logloss'
        },
        'airlines_onehot_250': {
            'train_file': AIRLINES_ONEHOT_TRAIN_FILE,
            'test_file': AIRLINES_ONEHOT_TEST_FILE,
            'data_files_have_header': False,
            'cd_file': AIRLINES_ONEHOT_CD_FILE,
            'loss_function': 'Logloss'
        },
        'black_friday': {
            'train_file': BLACK_FRIDAY_TRAIN_FILE,
            'test_file': BLACK_FRIDAY_TEST_FILE,
            'data_files_have_header': True,
            'cd_file': BLACK_FRIDAY_CD_FILE,
            'loss_function': 'RMSE'
        },
        'cloudness_small': {
            'train_file': CLOUDNESS_TRAIN_FILE,
            'test_file': CLOUDNESS_TEST_FILE,
            'data_files_have_header': False,
            'cd_file': CLOUDNESS_CD_FILE,
            'loss_function': 'MultiClass'
        },
        'higgs': {
            'train_file': HIGGS_TRAIN_FILE,
            'test_file': HIGGS_TEST_FILE,
            'data_files_have_header': False,
            'cd_file': HIGGS_CD_FILE,
            'loss_function': 'Logloss'
        },
        'querywise': {
            'train_file': QUERYWISE_TRAIN_FILE,
            'test_file': QUERYWISE_TEST_FILE,
            'data_files_have_header': False,
            'cd_file': QUERYWISE_CD_FILE,
            'loss_function': 'RMSE'
        },
    }


# this is needed because scipy.sparse matrix types do not support non-numerical data
def convert_cat_columns_to_hashed(src_features_dataframe):
    def create_hashed_categorical_column(src_column):
        hashed_column = []
        for value in src_column:
            hashed_column.append(np.uint32(hash(value)))
        return hashed_column

    new_columns_data = OrderedDict()
    for column_name, column_data in src_features_dataframe.iteritems():
        if column_data.dtype.name == 'category':
            new_columns_data[column_name] = create_hashed_categorical_column(column_data)
        else:
            new_columns_data[column_name] = column_data

    return DataFrame(new_columns_data)


@pytest.mark.parametrize('dataset', list(get_dataset_specification_for_sparse_input_tests().keys()))
def test_pools_equal_on_dense_and_scipy_sparse_input(dataset):
    metadata = get_dataset_specification_for_sparse_input_tests()[dataset]

    columns_metadata = read_cd(
        metadata['cd_file'],
        data_file=metadata['train_file'],
        canonize_column_types=True
    )

    data = load_dataset_as_dataframe(
        metadata['train_file'],
        columns_metadata,
        has_header=metadata['data_files_have_header']
    )

    data['features'] = convert_cat_columns_to_hashed(data['features'])

    dense_pool = Pool(
        data['features'],
        label=data['target'],
        cat_features=columns_metadata['cat_feature_indices']
    )

    canon_sparse_pool = None

    for sparse_matrix_type in sparse_matrix_types:
        sparse_features = sparse_matrix_type(data['features'])

        if columns_metadata['cat_feature_indices'] and (sparse_features.dtype.kind == 'f'):
            with pytest.raises(CatBoostError):
                Pool(
                    sparse_features,
                    label=data['target'],
                    feature_names=list(data['features'].columns),
                    cat_features=columns_metadata['cat_feature_indices']
                )
        else:
            sparse_pool = Pool(
                sparse_features,
                label=data['target'],
                feature_names=list(data['features'].columns),
                cat_features=columns_metadata['cat_feature_indices']
            )

            if canon_sparse_pool is None:
                canon_sparse_pool = sparse_pool
                assert _have_equal_features(dense_pool, sparse_pool, True)
            else:
                assert _have_equal_features(sparse_pool, canon_sparse_pool, False)


@pytest.mark.parametrize(
    'features_dtype',
    numpy_num_data_types,
    ids=['features_dtype=%s' % np.dtype(dtype).name for dtype in numpy_num_data_types]
)
@pytest.mark.parametrize(
    'features_density',
    [0.1, 0.2, 0.8],
    ids=['features_density=%s' % density for density in [0.1, 0.2, 0.8]]
)
def test_fit_on_scipy_sparse_spmatrix(features_dtype, features_density):
    if np.dtype(features_dtype).kind == 'f':
        cat_features = []
        lower_bound = -1.0
        upper_bound = 1.0
    else:
        cat_features = [0, 7, 11]
        lower_bound = max(np.iinfo(features_dtype).min, -32767)
        upper_bound = min(np.iinfo(features_dtype).max, 32767)

    features, labels = generate_random_labeled_dataset(
        n_samples=100,
        n_features=20,
        labels=[0, 1],
        features_density=features_density,
        features_dtype=features_dtype,
        features_range=(lower_bound, upper_bound),
    )

    dense_pool = Pool(features, label=labels, cat_features=cat_features)

    canon_sparse_pool = None

    for sparse_matrix_type in sparse_matrix_types:
        sparse_pool = Pool(sparse_matrix_type(features), label=labels, cat_features=cat_features)

        if canon_sparse_pool is None:
            canon_sparse_pool = sparse_pool
            assert _have_equal_features(dense_pool, sparse_pool, True)
        else:
            assert _have_equal_features(sparse_pool, canon_sparse_pool, False)

    model = CatBoostClassifier(iterations=5)
    model.fit(canon_sparse_pool)
    preds = model.predict(canon_sparse_pool)

    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(preds))
    return local_canonical_file(preds_path)


# pandas has NaN value indicating missing values by default,
# NaNs in categorical values are not supported by CatBoost
def make_catboost_compatible_categorical_missing_values(src_features_dataframe):
    new_columns_data = OrderedDict()
    for column_name, column_data in src_features_dataframe.iteritems():
        if column_data.dtype.name == 'category':
            column_data = column_data.cat.rename_categories([str(x) for x in column_data.cat.categories])
            column_data = column_data.cat.add_categories('').fillna('')

        new_columns_data[column_name] = column_data

    return DataFrame(new_columns_data)


def convert_to_sparse(src_features_dataframe, indexing_kind):
    new_columns_data = OrderedDict()
    for column_name, column_data in src_features_dataframe.iteritems():
        if column_data.dtype.name == 'category':
            fill_value = ''
        else:
            fill_value = 0.0

        new_columns_data[column_name] = SparseArray(column_data, fill_value=fill_value, kind=indexing_kind)

    return DataFrame(new_columns_data)


@pytest.mark.parametrize('dataset', list(get_dataset_specification_for_sparse_input_tests().keys()))
@pytest.mark.parametrize('indexing_kind', ['integer', 'block'])
def test_pools_equal_on_pandas_dense_and_sparse_input(dataset, indexing_kind):
    metadata = get_dataset_specification_for_sparse_input_tests()[dataset]

    columns_metadata = read_cd(
        metadata['cd_file'],
        data_file=metadata['train_file'],
        canonize_column_types=True
    )

    data = load_dataset_as_dataframe(
        metadata['train_file'],
        columns_metadata,
        has_header=metadata['data_files_have_header']
    )

    data['features'] = make_catboost_compatible_categorical_missing_values(data['features'])

    dense_pool = Pool(
        data['features'],
        label=data['target'],
        cat_features=columns_metadata['cat_feature_indices']
    )

    sparse_pool = Pool(
        convert_to_sparse(data['features'], indexing_kind),
        label=data['target'],
        cat_features=columns_metadata['cat_feature_indices']
    )

    assert _have_equal_features(dense_pool, sparse_pool, True)


@pytest.mark.parametrize('dataset', list(get_dataset_specification_for_sparse_input_tests().keys()))
@pytest.mark.parametrize('indexing_kind', ['integer', 'block'])
@pytest.mark.parametrize('boosting_type', BOOSTING_TYPE)
def test_training_and_prediction_equal_on_pandas_dense_and_sparse_input(task_type, dataset, indexing_kind, boosting_type):
    metadata = get_dataset_specification_for_sparse_input_tests()[dataset]

    if (task_type == 'GPU') and (boosting_type == 'Ordered') and (metadata['loss_function'] == 'MultiClass'):
        return pytest.xfail(reason="On GPU loss MultiClass can't be used with ordered boosting")

    columns_metadata = read_cd(
        metadata['cd_file'],
        data_file=metadata['train_file'],
        canonize_column_types=True
    )

    def get_dense_and_sparse_pools(data_file):
        data = load_dataset_as_dataframe(
            data_file,
            columns_metadata,
            has_header=metadata['data_files_have_header']
        )

        data['features'] = make_catboost_compatible_categorical_missing_values(data['features'])

        dense_pool = Pool(
            data['features'],
            label=data['target'],
            cat_features=columns_metadata['cat_feature_indices']
        )

        sparse_pool = Pool(
            convert_to_sparse(data['features'], indexing_kind),
            label=data['target'],
            feature_names=list(data['features'].columns),
            cat_features=columns_metadata['cat_feature_indices']
        )

        return dense_pool, sparse_pool

    dense_train_pool, sparse_train_pool = get_dense_and_sparse_pools(metadata['train_file'])
    dense_test_pool, sparse_test_pool = get_dense_and_sparse_pools(metadata['test_file'])

    params = {
        'task_type': task_type,
        'loss_function': metadata['loss_function'],
        'iterations': 5,
        'boosting_type': boosting_type
    }

    model_on_dense = CatBoost(params=params)
    model_on_dense.fit(dense_train_pool, eval_set=dense_test_pool)
    predictions_on_dense = model_on_dense.predict(dense_test_pool)

    model_on_sparse = CatBoost(params=params)
    model_on_sparse.fit(sparse_train_pool, eval_set=sparse_test_pool)
    predictions_model_on_sparse_on_dense_pool = model_on_sparse.predict(dense_test_pool)

    assert _check_data(predictions_on_dense, predictions_model_on_sparse_on_dense_pool)

    predictions_model_on_sparse_on_sparse_pool = model_on_sparse.predict(sparse_test_pool)

    assert _check_data(predictions_on_dense, predictions_model_on_sparse_on_sparse_pool)


def test_sparse_input_with_categorical_features_with_default_value_present_only_in_eval():
    cat_features = [1, 2]

    """
        1 x 2
        x x 3
        4 5 6
        1 x 3
    """
    row = np.array([0, 0, 1, 2, 2, 2, 3, 3])
    col = np.array([0, 2, 2, 0, 1, 2, 0, 2])
    data = np.array([1, 2, 3, 4, 5, 6, 1, 3])
    X_train = scipy.sparse.csr_matrix((data, (row, col)), shape=(4, 3))
    y_train = np.array([0, 1, 1, 0])

    """
        3 4 2
        x 1 6
        5 x x
        7 x 8
        1 x 1
    """
    row = np.array([0, 0, 0, 1, 1, 2, 3, 3, 4, 4])
    col = np.array([0, 1, 2, 1, 2, 0, 0, 2, 0, 2])
    data = np.array([3, 4, 2, 1, 6, 5, 7, 8, 1, 1])
    X_validation = scipy.sparse.csr_matrix((data, (row, col)), shape=(5, 3))
    y_validation = np.array([1, 0, 0, 1, 0])

    model = CatBoostClassifier(iterations=5)

    model.fit(X_train, y_train, cat_features, eval_set=(X_validation, y_validation))

    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, model.predict(X_validation), fmt='%.8f')
    return local_canonical_file(preds_path)


@pytest.mark.parametrize('model_shrink_rate', [None, 0, 0.2])
def test_different_formats_of_monotone_constraints(model_shrink_rate):
    from catboost.datasets import monotonic2, set_cache_path
    set_cache_path(test_output_path())  # specify cache dir to fix potential data race while downloading dataset
    monotonic2_train, monotonic2_test = monotonic2()
    X_train, y_train = monotonic2_train.drop(columns=['Target']), monotonic2_train['Target']
    X_test, y_test = monotonic2_test.drop(columns=['Target']), monotonic2_test['Target']
    feature_names = list(X_train.columns)
    train_pool = Pool(data=X_train, label=y_train, feature_names=feature_names)
    test_pool = Pool(data=X_test, label=y_test, feature_names=feature_names)

    monotone_constraints_array = np.array([-1, 1, 1, -1])
    monotone_constraints_list = [-1, 1, 1, -1]
    monotone_constraints_dict_1 = {0: -1, 1: 1, 2: 1, 3: -1}
    monotone_constraints_dict_2 = {'MonotonicNeg0': -1, 'MonotonicPos0': 1, 'MonotonicPos1': 1, 'MonotonicNeg1': -1}
    monotone_constraints_string_1 = "(-1,1,1,-1)"
    monotone_constraints_string_2 = "0:-1,1:1,2:1,3:-1"
    monotone_constraints_string_3 = "MonotonicNeg0:-1,MonotonicPos0:1,MonotonicPos1:1,MonotonicNeg1:-1"

    common_options = dict(iterations=50, model_shrink_rate=model_shrink_rate)
    model1 = CatBoostRegressor(monotone_constraints=monotone_constraints_array, **common_options)
    model1.fit(train_pool)
    predictions1 = model1.predict(test_pool)
    assert abs(model1.evals_result_['learn']['RMSE'][-1] - model1.eval_metrics(train_pool, 'RMSE')['RMSE'][-1]) < 1e-9

    for monotone_constraints in [monotone_constraints_list, monotone_constraints_dict_1,
                                 monotone_constraints_dict_2, monotone_constraints_string_1,
                                 monotone_constraints_string_2, monotone_constraints_string_3]:
        model2 = CatBoostRegressor(monotone_constraints=monotone_constraints, **common_options)
        model2.fit(train_pool)
        predictions2 = model2.predict(test_pool)
        assert all(predictions1 == predictions2)


def test_different_formats_of_monotone_constraints_in_different_modes():
    from catboost.datasets import monotonic2, set_cache_path
    set_cache_path(test_output_path())  # specify cache dir to fix potential data race while downloading dataset
    monotonic2_train, monotonic2_test = monotonic2()
    X_train, y_train = monotonic2_train.drop(columns=['Target']), monotonic2_train['Target']
    feature_names = list(X_train.columns)
    train_pool = Pool(data=X_train, label=y_train, feature_names=feature_names)

    monotone_constraints_array = np.array([-1, 1, 1, -1])
    monotone_constraints_list = [-1, 1, 1, -1]
    monotone_constraints_dict_1 = {0: -1, 1: 1, 2: 1, 3: -1}
    monotone_constraints_dict_2 = {'MonotonicNeg0': -1, 'MonotonicPos0': 1, 'MonotonicPos1': 1, 'MonotonicNeg1': -1}
    monotone_constraints_string_1 = "(-1,1,1,-1)"
    monotone_constraints_string_2 = "0:-1,1:1,2:1,3:-1"
    monotone_constraints_string_3 = "MonotonicNeg0:-1,MonotonicPos0:1,MonotonicPos1:1,MonotonicNeg1:-1"

    common_options = dict(iterations=5, loss_function='RMSE')

    for monotone_constraints in [
        monotone_constraints_array,
        monotone_constraints_list,
        monotone_constraints_dict_1,
        monotone_constraints_dict_2,
        monotone_constraints_string_1,
        monotone_constraints_string_2,
        monotone_constraints_string_3
    ]:
        # grid search
        model = CatBoost(dict(common_options, **{'monotone_constraints': monotone_constraints}))
        max_depth_list = [4, 5]
        model.grid_search(
            {
                'max_depth': max_depth_list,
            },
            train_pool,
        )
        # random search
        model = CatBoost(dict(common_options, **{'monotone_constraints': monotone_constraints}))
        model.randomized_search(
            {
                'max_depth': max_depth_list,
            },
            train_pool,
            n_iter=1,
        )
        # cv
        cv(train_pool, params=dict(common_options, **{'monotone_constraints': monotone_constraints}))

    wrong_monotone_constraints_array = np.array([-1, 2, 1, -1])
    wrong_monotone_constraints_list = [-1, 2, 1, -1]
    wrong_monotone_constraints_dict_1 = {0: -1, 1: 2, 2: 1, 3: -1}
    wrong_monotone_constraints_dict_2 = {'MonotonicNeg0': -1, 'MonotonicPos0': 2, 'MonotonicPos1': 1, 'MonotonicNeg1': -1}
    wrong_monotone_constraints_string_1 = "(-1,2,1,-1)"
    wrong_monotone_constraints_string_2 = "0:-1,1:2,2:1,3:-1"
    wrong_monotone_constraints_string_3 = "MonotonicNeg0:-1,MonotonicPos0:2,MonotonicPos1:1,MonotonicNeg1:-1"

    for monotone_constraints in [
        wrong_monotone_constraints_array,
        wrong_monotone_constraints_list,
        wrong_monotone_constraints_dict_1,
        wrong_monotone_constraints_dict_2,
        wrong_monotone_constraints_string_1,
        wrong_monotone_constraints_string_2,
        wrong_monotone_constraints_string_3
    ]:
        with pytest.raises(CatBoostError):
            model = CatBoostRegressor(monotone_constraints=monotone_constraints, **common_options)
            model.fit(train_pool)
        with pytest.raises(CatBoostError):
            model = CatBoost({"monotone_constraints": monotone_constraints})
            iterations_list = [5, 10]
            model.grid_search(
                {
                    'iterations': iterations_list,
                },
                train_pool,
            )
        with pytest.raises(CatBoostError):
            model = CatBoost({"monotone_constraints": monotone_constraints})
            iterations_list = [5, 10]
            model.randomized_search(
                {
                    'iterations': iterations_list,
                },
                train_pool,
                n_iter=1,
            )
        with pytest.raises(CatBoostError):
            cv(train_pool, params=dict(common_options, **{'monotone_constraints': monotone_constraints}))


def test_same_values_with_different_types(task_type):
    # take integers from [0, 127] because they can be represented by any of this types

    canon_predictions = None

    n_features = 20
    n_objects = 500

    params = {
        'task_type': task_type,
        'loss_function': 'Logloss',
        'iterations': 5
    }

    labels = np.random.randint(0, 2, size=n_objects)

    canon_features = np.random.randint(0, 127, size=(n_objects, n_features), dtype=np.int8)

    for data_type in numpy_num_data_types:
        features_df = DataFrame()

        for feature_idx in range(n_features):
            features_df['feature_%i' % feature_idx] = canon_features[:, feature_idx].astype(data_type)

        model = CatBoost(params)
        model.fit(features_df, labels)
        predictions = model.predict(features_df)

        if canon_predictions is None:
            canon_predictions = predictions
        else:
            _check_data(canon_predictions, predictions)


@pytest.mark.parametrize('loss_function', ['Logloss', 'MultiClass'])
def test_default_eval_metric(loss_function):
    X = np.array([[1, 2, 3], [5, 4, 23], [8954, 4, 22]])
    y = np.array([1, 0, 1])
    p = Pool(X, y)
    model = CatBoostClassifier(task_type='CPU', loss_function=loss_function, iterations=15, metric_period=100)
    model.fit(p)
    assert model.get_all_params()["eval_metric"] == loss_function


def test_multiclass_train_on_constant_data(task_type):
    features = np.asarray(
        [[0.27290749, 0.63002519, 0., 0.38624339, 0.],
         [0.27290748, 0.63002519, 0., 0.38624339, 0.],
         [0.27290747, 0.63002519, 0., 0.38624339, 0.]]
    )
    classes = ['0', '1', '2']
    labels = np.asarray(classes)

    clf = CatBoostClassifier(
        iterations=2,
        loss_function='MultiClass'
    )

    model = clf.fit(features, labels)

    assert np.all(model.classes_ == classes)
    model.predict(features)


@pytest.mark.parametrize(
    'label_type',
    label_types,
    ids=['label_type=%s' % s for s in label_types]
)
@pytest.mark.parametrize(
    'loss_function',
    ['Logloss', 'CrossEntropy'],
    ids=['loss_function=%s' % s for s in ['Logloss', 'CrossEntropy']]
)
def test_classes_attribute_binclass(label_type, loss_function):
    params = {'loss_function': loss_function, 'iterations': 2}

    if label_type == 'consecutive_integers':
        unique_labels = [0, 1]
    elif label_type == 'nonconsecutive_integers':
        unique_labels = [2, 5]
    elif label_type == 'string':
        unique_labels = ['class0', 'class1']
    elif label_type == 'float':
        unique_labels = [0.0, 0.1, 0.2, 0.5, 1.0]
        if loss_function == 'Logloss':
            params['target_border'] = 0.5

    features, labels = generate_random_labeled_dataset(n_samples=20, n_features=5, labels=unique_labels)

    model = CatBoostClassifier(**params)
    if (loss_function == 'CrossEntropy') and (label_type in ['nonconsecutive_integers', 'string']):
        # 'nonconsecutive_integers' because it is out of [0.0, 1.0] bounds
        # 'string' because CrossEntropy requires numerical target
        with pytest.raises(CatBoostError):
            model.fit(features, labels)
    else:
        model.fit(features, labels)

        if label_type == 'consecutive_integers':
            assert np.all(model.classes_ == [0, 1])
        elif label_type == 'nonconsecutive_integers':
            assert np.all(sorted(model.classes_) == [2, 5])
        elif label_type == 'string':
            assert np.all(model.classes_ == ['class0', 'class1'])
        elif label_type == 'float':
            assert np.all(model.classes_ == [0, 1])


@pytest.mark.parametrize(
    'label_type',
    label_types,
    ids=['label_type=%s' % s for s in label_types]
)
def test_classes_attribute_multiclass(label_type):
    params = {'loss_function': 'MultiClass', 'iterations': 2}

    if label_type == 'consecutive_integers':
        unique_labels = [0, 1, 2, 3]
    elif label_type == 'nonconsecutive_integers':
        unique_labels = [2, 5, 9, 11]
    elif label_type == 'string':
        unique_labels = ['class0', 'class1', 'class2', 'class3']
    elif label_type == 'float':
        unique_labels = [0.0, 0.1, 0.2, 0.5, 1.0]

    features, labels = generate_random_labeled_dataset(n_samples=20, n_features=5, labels=unique_labels)

    model = CatBoostClassifier(**params)
    model.fit(features, labels)

    if label_type == 'consecutive_integers':
        assert np.all(model.classes_ == [0, 1, 2, 3])
    elif label_type == 'nonconsecutive_integers':
        assert np.all(model.classes_ == [2, 5, 9, 11])
    elif label_type == 'string':
        assert np.all(sorted(model.classes_) == ['class0', 'class1', 'class2', 'class3'])
    elif label_type == 'float':
        assert np.allclose(sorted(model.classes_), [0, 0.1, 0.2, 0.5, 1])


def test_multiclass_non_positive_definite_from_github():
    train = np.array([
        [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ],
        [
            0.00473934, 0.05, 0., 0., 0., 0., 0., 0., 0., 0.
        ],
        [
            0.04739336, 0.1, 0., 0., 0., 0.03191489, 0., 0., 0., 0.
        ],
        [
            0., 0., 0., 0., 0.00298507, 0.09574468, 0.0195122, 0.01492537, 0.00787402, 0.
        ],
        [
            0.0521327, 0.15, 0.00480769, 0.07692308, 0., 0., 0., 0., 0., 0.
        ]
    ])
    cat_params = {
        'iterations': 500,
        'random_state': 42,
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'early_stopping_rounds': 30,
        'thread_count': 16,
        'l2_leaf_reg': 0
    }
    y_train = np.array([1, 2, 4, 2, 1])
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X=np.array(train)[:5, :10], y=np.array(y_train)[:5], verbose=0)


def test_snapshot_checksum(task_type):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    model = CatBoostClassifier(
        task_type=task_type,
        iterations=15,
        save_snapshot=True,
        snapshot_file='snapshot',
    )
    model.fit(train_pool, eval_set=test_pool)

    model_next = CatBoostClassifier(
        task_type=task_type,
        iterations=30,
        save_snapshot=True,
        snapshot_file='snapshot',
    )
    model_next.fit(train_pool, eval_set=test_pool)

    with pytest.raises(CatBoostError):
        model_next.fit(test_pool, eval_set=train_pool)


def test_bootstrap_defaults():
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)

    model = CatBoostClassifier(iterations=1)
    model.fit(pool)
    params = model.get_all_params()
    assert params['bootstrap_type'] == 'MVS'
    assert abs(params['subsample'] - 0.8) < EPS

    model = CatBoostClassifier(iterations=1, bootstrap_type='Bernoulli')
    model.fit(pool)
    params = model.get_all_params()
    assert params['bootstrap_type'] == 'Bernoulli'
    assert abs(params['subsample'] - 0.66) < EPS

    model = CatBoostClassifier(iterations=1, loss_function='MultiClass')
    model.fit(pool)
    params = model.get_all_params()
    assert params['bootstrap_type'] == 'Bayesian'
    assert 'subsample' not in params


def test_monoforest_regression():
    train_pool = Pool(HIGGS_TRAIN_FILE, column_description=HIGGS_CD_FILE)
    model = CatBoostRegressor(loss_function='RMSE', iterations=10)
    model.fit(train_pool)
    from catboost.monoforest import to_polynom_string, plot_features_strength
    poly = to_polynom_string(model)
    assert poly, "Unexpected empty poly"
    plot = plot_features_strength(model)
    assert plot, "Unexpected empty plot"


def test_text_processing_tokenizer():
    from catboost.text_processing import Tokenizer
    assert Tokenizer(lowercasing=True).tokenize('Aba caba') == ['aba', 'caba']


def test_text_processing_dictionary():
    from catboost.text_processing import Dictionary

    dictionary = Dictionary(occurence_lower_bound=0).fit([
        ['aba', 'caba'],
        ['ala', 'caba'],
        ['caba', 'aba']
    ])

    assert dictionary.size == 3
    assert dictionary.get_top_tokens(2) == ['caba', 'aba']
    assert dictionary.apply([['ala', 'caba'], ['aba']]) == [[2, 0], [1]]

    dictionary_path = test_output_path('dictionary.tsv')
    dictionary.save(dictionary_path)
    return compare_canonical_models(dictionary_path)


def test_log_proba():
    # multiclass with softmax
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=50, thread_count=8, devices='0')
    classifier.fit(pool)
    pred = classifier.predict(pool, prediction_type='Probability')
    log_pred = classifier.predict(pool, prediction_type='LogProbability')
    assert np.allclose(log_pred, np.log(pred))

    # multiclass OneVsAll
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=50, thread_count=8, loss_function='MultiClassOneVsAll', devices='0')
    classifier.fit(pool)
    pred = classifier.predict(pool, prediction_type='Probability')
    log_pred = classifier.predict(pool, prediction_type='LogProbability')
    assert np.allclose(log_pred, np.log(pred))

    # binary classification
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    classifier = CatBoostClassifier(iterations=2)
    classifier.fit(pool)
    pred = classifier.predict(pool, prediction_type='Probability')
    log_pred_1 = classifier.predict(pool, prediction_type='LogProbability')
    log_pred_2 = classifier.predict_log_proba(pool)
    assert np.allclose(log_pred_1, np.log(pred))
    assert np.allclose(log_pred_1, log_pred_2)


def test_binclass_probability_threshold():
    # binary classification
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)

    classifier = CatBoostClassifier(iterations=2)
    with pytest.raises(CatBoostError):
        classifier.set_probability_threshold(0.6)  # can't set before training
    with pytest.raises(CatBoostError):
        classifier.get_probability_threshold()  # can't get before training

    classifier.fit(pool)
    assert classifier.get_probability_threshold() == 0.5  # check default value

    classifier.set_probability_threshold(0.6)  # check set
    assert classifier.get_probability_threshold() == 0.6  # check get

    def check_predictions_with_threshold(classifier, threshold):
        proba = classifier.predict_proba(pool)[:, 1]
        pred = classifier.predict(pool).astype(int)
        if threshold < 0.5:
            assert np.any((proba > threshold) & (proba < 0.5))  # check for non-dummy test case
        elif threshold > 0.5:
            assert np.any((proba > 0.5) & (proba < threshold))  # check for non-dummy test case

        assert np.all(pred == (proba > threshold).astype(int))  # check if threshold is working

    check_predictions_with_threshold(classifier, 0.6)

    accuracy = classifier.score(pool)  # check if score function is working too
    pred = classifier.predict(pool).astype(int)
    assert np.allclose(accuracy, np.mean(pred == np.array(pool.get_label()).astype(int)))

    # change threshold
    classifier.set_probability_threshold(0.4)
    assert classifier.get_probability_threshold() == 0.4
    check_predictions_with_threshold(classifier, 0.4)

    # nullify it
    classifier.set_probability_threshold(None)
    assert classifier.get_probability_threshold() == 0.5
    check_predictions_with_threshold(classifier, 0.5)


def test_exponent_prediction_type():
    # poisson regression
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    regressor = CatBoostRegressor(iterations=2, objective='Poisson')
    regressor.fit(pool)
    pred = regressor.predict(pool, prediction_type='RawFormulaVal')
    exp_pred = regressor.predict(pool)
    assert np.allclose(exp_pred, np.exp(pred))


def test_rmse_with_uncertainty_prediction_type():
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=CD_FILE)
    regressor = CatBoostRegressor(iterations=2, objective='RMSEWithUncertainty')
    regressor.fit(pool)
    pred = np.transpose(regressor.predict(pool, prediction_type='RawFormulaVal'))
    exp_pred = np.transpose(regressor.predict(pool))
    assert np.allclose(exp_pred[0], pred[0])
    assert np.allclose(exp_pred[1], np.exp(pred[1] * 2))


def test_bad_uncertainty_prediction_types_usage():
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=CD_FILE)
    regressor = CatBoostRegressor(iterations=100, posterior_sampling=True)
    regressor.fit(pool)
    for prediction_type in ['TotalUncertainty', 'VirtEnsembles']:
        with pytest.raises(Exception):
            regressor.predict(pool, prediction_type=prediction_type)


@pytest.mark.parametrize('virtual_ensembles_count', [1, 5])
@pytest.mark.parametrize('prediction_type', ['TotalUncertainty', 'VirtEnsembles'])
@pytest.mark.parametrize('loss_function', ['RMSE', 'RMSEWithUncertainty', 'Logloss', 'MultiClass'])
def test_uncertainty_prediction_types(virtual_ensembles_count, prediction_type, loss_function):
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE) \
        if loss_function != 'Logloss' \
        else Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoost(
        {
            "iterations": 100,
            "loss_function": loss_function,
            "posterior_sampling": True,
        }
    )

    model.fit(pool)

    preds = model.virtual_ensembles_predict(pool, prediction_type=prediction_type, virtual_ensembles_count=virtual_ensembles_count)
    preds_path = test_output_path(PREDS_TXT_PATH)
    if prediction_type == 'VirtEnsembles':
        shape = preds.shape
        assert len(shape) == 3 and shape[1] == virtual_ensembles_count
        preds = preds.reshape(shape[0], shape[1] * shape[2])
    np.savetxt(preds_path, preds, fmt='%.9f', delimiter='\t')
    return local_canonical_file(preds_path)


def test_staged_log_proba():
    # multiclass with softmax
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=50, thread_count=8, devices='0')
    classifier.fit(pool)
    pred_it = classifier.staged_predict(pool, prediction_type='Probability')
    log_pred_it = classifier.staged_predict(pool, prediction_type='LogProbability')
    for pred, log_pred in zip(pred_it, log_pred_it):
        assert np.allclose(log_pred, np.log(pred))

    # multiclass OneVsAll
    pool = Pool(CLOUDNESS_TRAIN_FILE, column_description=CLOUDNESS_CD_FILE)
    classifier = CatBoostClassifier(iterations=50, thread_count=8, loss_function='MultiClassOneVsAll', devices='0')
    classifier.fit(pool)
    pred_it = classifier.staged_predict(pool, prediction_type='Probability')
    log_pred_it = classifier.staged_predict(pool, prediction_type='LogProbability')
    for pred, log_pred in zip(pred_it, log_pred_it):
        assert np.allclose(log_pred, np.log(pred))

    # binary classification
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    classifier = CatBoostClassifier(iterations=2)
    classifier.fit(pool)
    pred_it = classifier.staged_predict(pool, prediction_type='Probability')
    log_pred_it_1 = classifier.staged_predict(pool, prediction_type='LogProbability')
    log_pred_it_2 = classifier.staged_predict_log_proba(pool)
    for pred, log_pred_1, log_pred_2 in zip(pred_it, log_pred_it_1, log_pred_it_2):
        assert np.allclose(log_pred_1, np.log(pred))
        assert np.allclose(log_pred_1, log_pred_2)
        assert np.allclose(log_pred_1, log_pred_2)


def test_shap_assert():
    model_path = test_output_path('model.json')
    pool = Pool([[0, ], [1, ], ], [0, 1])
    model = train(pool, {'iterations': 1, 'task_type': 'CPU', 'devices': '0'})
    model.save_model(model_path, format='json')

    json_model = json.load(open(model_path))
    json_model['scale_and_bias'] = [1, [1]]
    json.dump(json_model, open(model_path, 'w'))
    model = CatBoost().load_model(model_path, format='json')
    shap_values = model.get_feature_importance(type='ShapValues', data=pool)
    predictions = model.predict(pool)
    assert (len(predictions) == len(shap_values))
    for i, pred_idx in enumerate(range(len(predictions))):
        assert (abs(sum(shap_values[pred_idx]) - predictions[pred_idx]) < 1e-9), (sum(shap_values[pred_idx]) - predictions[pred_idx])

    json_model['oblivious_trees'] = [{
        'leaf_values': [1, 2],
        'leaf_weights': [1, 0],
        'splits': [{'border': 0.5, 'float_feature_index': 1, 'split_index': 0, 'split_type': 'FloatFeature'}]
    }]
    json_model['features_info'] = {
        'float_features': [{'borders': [0.5], 'feature_index': 0, 'flat_feature_index': 0, 'has_nans': False, 'nan_value_treatment': 'AsIs'}]
    }

    json.dump(json_model, open(model_path, 'w'))
    model = CatBoost().load_model(model_path, format='json')
    model.get_feature_importance(type='ShapValues', data=pool)


@pytest.mark.parametrize('shrink_mode', ['Constant', 'Decreasing'])
@pytest.mark.parametrize('shrink_rate', [0, 0.2])
@pytest.mark.parametrize('diffusion', [0, 1000])
def test_diffusion_temperature_with_shrink_mode(shrink_mode, shrink_rate, diffusion):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)
    params = {
        'iterations': 50,
        'learning_rate': 0.03,
        'model_shrink_mode': shrink_mode,
        'model_shrink_rate': shrink_rate,
        'diffusion_temperature': diffusion
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool)
    pred = model.predict_proba(test_pool)
    preds_path = test_output_path('predictions.tsv')
    np.savetxt(preds_path, np.array(pred), fmt='%.15f', delimiter='\t')
    return local_canonical_file(preds_path)


def test_langevin_with_empty_leafs():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    params = {
        'iterations': 10,
        'depth': 10,
        'learning_rate': 0.03,
        'langevin': True,
        'diffusion_temperature': 1000
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool)
    for value, weight in zip(model.get_leaf_values(), model.get_leaf_weights()):
        if weight == 0:
            assert value == 0


def test_to_classifier():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    models = []
    for learning_rate in [.03, .05]:
        models.append(CatBoostClassifier(iterations=2, learning_rate=learning_rate))
        models[-1].fit(train_pool, eval_set=test_pool)

    merged_model = sum_models(models)
    prediction = merged_model.predict(test_pool, prediction_type='Probability')
    assert type(merged_model) is CatBoost

    merged_model = to_classifier(merged_model)

    assert isinstance(merged_model, CatBoostClassifier)
    assert _check_data(merged_model.predict_proba(test_pool), prediction)


def test_to_classifier_wrong_type():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostRegressor(iterations=2, learning_rate=.05, objective='RMSE')
    model.fit(train_pool)
    with pytest.raises(CatBoostError):
        to_classifier(model)


def test_to_regressor():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    test_pool = Pool(TEST_FILE, column_description=CD_FILE)

    models = []
    for learning_rate in [.03, .05]:
        models.append(CatBoostRegressor(iterations=2, learning_rate=learning_rate))
        models[-1].fit(train_pool, eval_set=test_pool)

    merged_model = sum_models(models)
    prediction = merged_model.predict(test_pool)
    assert type(merged_model) is CatBoost

    merged_model = to_regressor(merged_model)

    assert isinstance(merged_model, CatBoostRegressor)
    assert _check_data(merged_model.predict(test_pool), prediction)


def test_to_regressor_wrong_type():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=.05, objective='Logloss')
    model.fit(train_pool)
    with pytest.raises(CatBoostError):
        to_regressor(model)


def test_to_ranker():
    train_pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    test_pool = Pool(QUERYWISE_TEST_FILE, column_description=QUERYWISE_CD_FILE)

    models = []
    for learning_rate in [.03, .05]:
        models.append(CatBoostRanker(iterations=2, learning_rate=learning_rate))
        models[-1].fit(train_pool, eval_set=test_pool)

    merged_model = sum_models(models)
    prediction = merged_model.predict(test_pool)
    assert type(merged_model) is CatBoost

    merged_model = to_ranker(merged_model)

    assert isinstance(merged_model, CatBoostRanker)
    assert _check_data(merged_model.predict(test_pool), prediction)


def test_to_ranker_wrong_type():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=.05, objective='Logloss')
    model.fit(train_pool)
    with pytest.raises(CatBoostError):
        to_ranker(model)


def test_load_and_save_quantization_borders():
    borders_32_file = test_output_path('borders_32.dat')
    borders_10_file = test_output_path('borders_10.dat')
    borders_from_input_borders_file = test_output_path('borders_from_input_borders.dat')

    pool_border_count_32 = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    pool_border_count_32.quantize(border_count=32)
    pool_border_count_32.save_quantization_borders(borders_32_file)

    pool_border_count_10 = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    pool_border_count_10.quantize(border_count=10)
    pool_border_count_10.save_quantization_borders(borders_10_file)

    pool_from_input_borders = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    pool_from_input_borders.quantize(input_borders=borders_10_file)
    pool_from_input_borders.save_quantization_borders(borders_from_input_borders_file)

    assert filecmp.cmp(borders_10_file, borders_from_input_borders_file)
    assert not filecmp.cmp(borders_32_file, borders_10_file)

    return [local_canonical_file(borders_32_file), local_canonical_file(borders_10_file)]


def test_feature_weights_work():
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    most_important_feature_index = 3
    classifier_params = {
        'iterations': 5,
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'devices': '0',
        'loss_function': 'MultiClass',
    }

    model_without_feature_weights = CatBoostClassifier(**classifier_params)
    model_without_feature_weights.fit(pool)
    importance_without_feature_weights = model_without_feature_weights.get_feature_importance(
        type=EFstrType.PredictionValuesChange,
        data=pool
    )[most_important_feature_index]

    model_with_feature_weights = CatBoostClassifier(
        feature_weights={most_important_feature_index: 0.1},
        **classifier_params
    )
    model_with_feature_weights.fit(pool)
    importance_with_feature_weights = model_with_feature_weights.get_feature_importance(
        type=EFstrType.PredictionValuesChange,
        data=pool
    )[most_important_feature_index]

    assert importance_with_feature_weights < importance_without_feature_weights


def test_different_formats_of_feature_weights():
    train_pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    test_pool = Pool(AIRLINES_5K_TEST_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)

    feature_weights_array = np.array([1, 1, 1, 0.1, 1, 1, 1, 2])
    feature_weights_list = [1, 1, 1, 0.1, 1, 1, 1, 2]
    feature_weights_dict_1 = {3: 0.1, 7: 2}
    feature_weights_dict_2 = {'DepTime': 0.1, 'Distance': 2}
    feature_weights_string_1 = "(1,1,1,0.1,1,1,1,2)"
    feature_weights_string_2 = "3:0.1,7:2"
    feature_weights_string_3 = "DepTime:0.1,Distance:2"

    common_options = dict(iterations=50, loss_function='Logloss')
    model1 = CatBoostClassifier(feature_weights=feature_weights_array, **common_options)
    model1.fit(train_pool)
    predictions1 = model1.predict(test_pool)
    for feature_weights in [
        feature_weights_list,
        feature_weights_dict_1,
        feature_weights_dict_2,
        feature_weights_string_1,
        feature_weights_string_2,
        feature_weights_string_3
    ]:
        model2 = CatBoostClassifier(feature_weights=feature_weights, **common_options)
        model2.fit(train_pool)
        predictions2 = model2.predict(test_pool)
        assert all(predictions1 == predictions2)

    # check that it works not only in fit mode
    for first_feature_use_penalties in [
        feature_weights_array,
        feature_weights_list,
        feature_weights_dict_1,
        feature_weights_dict_2,
        feature_weights_string_1,
        feature_weights_string_2,
        feature_weights_string_3
    ]:
        # grid search
        model = CatBoost({"feature_weights": feature_weights})
        iterations_list = [5, 10]
        model.grid_search(
            {
                'iterations': iterations_list,
            },
            train_pool,
        )
        # random search
        model = CatBoost({"feature_weights": feature_weights})
        iterations_list = [5, 10]
        model.randomized_search(
            {
                'iterations': iterations_list,
            },
            train_pool,
            n_iter=1,
        )
        # cv
        cv(train_pool, params=dict(common_options, **{'feature_weights': feature_weights}))

    # check that negative values are not accepted
    wrong_feature_weights_array = np.array([1, 1, 1, 0.1, 1, 1, 1, -2])
    wrong_feature_weights_list = [1, 1, 1, 0.1, 1, 1, 1, -2]
    wrong_feature_weights_dict_1 = {3: 0.1, 7: -2}
    wrong_feature_weights_dict_2 = {'DepTime': 0.1, 'Distance': -2}
    wrong_feature_weights_string_1 = "(1,1,1,0.1,1,1,1,-2)"
    wrong_feature_weights_string_2 = "3:0.1,7:-2"
    wrong_feature_weights_string_3 = "DepTime:0.1,Distance:-2"

    for feature_weights in [
        wrong_feature_weights_array,
        wrong_feature_weights_list,
        wrong_feature_weights_dict_1,
        wrong_feature_weights_dict_2,
        wrong_feature_weights_string_1,
        wrong_feature_weights_string_2,
        wrong_feature_weights_string_3
    ]:
        with pytest.raises(CatBoostError):
            model = CatBoostClassifier(feature_weights=feature_weights, **common_options)
            model.fit(train_pool)
        with pytest.raises(CatBoostError):
            model = CatBoost({"feature_weights": feature_weights})
            iterations_list = [5, 10]
            model.grid_search(
                {
                    'iterations': iterations_list,
                },
                train_pool,
            )
        with pytest.raises(CatBoostError):
            model = CatBoost({"feature_weights": feature_weights})
            iterations_list = [5, 10]
            model.randomized_search(
                {
                    'iterations': iterations_list,
                },
                train_pool,
                n_iter=1,
            )
        with pytest.raises(CatBoostError):
            cv(train_pool, params=dict(common_options, **{'feature_weights': feature_weights}))


def test_feature_tags_interface():
    data = np.arange(80).reshape((10, 8))
    label = np.arange(10) % 2
    pool = Pool(
        data,
        label=label,
        feature_names=["c_0", "c_1", "c_2", "num_3", "c_4", "c_5", "c_6", "num_7"],
        feature_tags={
            "tag1" : {
                "features": np.arange(4),
                "cost": 5
            },
            "tag2": {"features": ["num_3", "num_7"]},
            "tag3" : {"features": [4, 5]}
        }
    )
    cat = CatBoostClassifier(ignored_features=["#tag1", "#tag2"])
    cat.fit(pool)
    assert np.array_equal(np.where(cat.feature_importances_ == 0)[0], [0, 1, 2, 3, 7])
    cat = CatBoostClassifier()
    result = cat.select_features(
        pool,
        features_for_select=["#tag1", "#tag2"],
        num_features_to_select=3
    )
    assert all(x in [0, 1, 2, 3, 7] for x in result["selected_features"])
    assert all(x in [0, 1, 2, 3, 7] for x in result["eliminated_features"])
    with pytest.raises(CatBoostError):
        Pool(data, feature_tags={"features": [4, 5]})
    with pytest.raises(CatBoostError):
        Pool(data, feature_tags={"tag2": {"features": ["num_3", "num_4"]}})


def test_first_feature_use_penalties_work():
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    most_important_feature_index = 3
    classifier_params = {
        'iterations': 5,
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'devices': '0',
        'loss_function': 'MultiClass',
    }

    model_without_feature_penalties = CatBoostClassifier(**classifier_params)
    model_without_feature_penalties.fit(pool)
    importance_without_feature_penalties = model_without_feature_penalties.get_feature_importance(
        type=EFstrType.PredictionValuesChange,
        data=pool
    )[most_important_feature_index]

    model_with_feature_penalties = CatBoostClassifier(
        first_feature_use_penalties={most_important_feature_index: 100},
        **classifier_params
    )
    model_with_feature_penalties.fit(pool)
    importance_with_feature_penalties = model_with_feature_penalties.get_feature_importance(
        type=EFstrType.PredictionValuesChange,
        data=pool
    )[most_important_feature_index]

    assert importance_with_feature_penalties < importance_without_feature_penalties


def test_different_formats_of_first_feature_use_penalties():
    train_pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    test_pool = Pool(AIRLINES_5K_TEST_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)

    first_feature_use_penalties_array = np.array([0, 0, 0, 10, 0, 0, 0, 2])
    first_feature_use_penalties_list = [0, 0, 0, 10, 0, 0, 0, 2]
    first_feature_use_penalties_dict_1 = {3: 10, 7: 2}
    first_feature_use_penalties_dict_2 = {'DepTime': 10, 'Distance': 2}
    first_feature_use_penalties_string_1 = "(0,0,0,10,0,0,0,2)"
    first_feature_use_penalties_string_2 = "3:10,7:2"
    first_feature_use_penalties_string_3 = "DepTime:10,Distance:2"

    common_options = dict(iterations=50, loss_function='Logloss')
    model1 = CatBoostClassifier(first_feature_use_penalties=first_feature_use_penalties_array, **common_options)
    model1.fit(train_pool)
    predictions1 = model1.predict(test_pool)
    for first_feature_use_penalties in [
        first_feature_use_penalties_list,
        first_feature_use_penalties_dict_1,
        first_feature_use_penalties_dict_2,
        first_feature_use_penalties_string_1,
        first_feature_use_penalties_string_2,
        first_feature_use_penalties_string_3
    ]:
        model2 = CatBoostClassifier(first_feature_use_penalties=first_feature_use_penalties, **common_options)
        model2.fit(train_pool)
        predictions2 = model2.predict(test_pool)
        assert all(predictions1 == predictions2)

    # check that it works not only in fit mode
    for first_feature_use_penalties in [
        first_feature_use_penalties_array,
        first_feature_use_penalties_list,
        first_feature_use_penalties_dict_1,
        first_feature_use_penalties_dict_2,
        first_feature_use_penalties_string_1,
        first_feature_use_penalties_string_2,
        first_feature_use_penalties_string_3
    ]:
        # grid search
        model = CatBoost({"first_feature_use_penalties": first_feature_use_penalties})
        iterations_list = [5, 10]
        model.grid_search(
            {
                'iterations': iterations_list,
            },
            train_pool,
        )
        # random search
        model = CatBoost({"first_feature_use_penalties": first_feature_use_penalties})
        iterations_list = [5, 10]
        model.randomized_search(
            {
                'iterations': iterations_list,
            },
            train_pool,
            n_iter=1,
        )
        # cv
        cv(train_pool, params=dict(common_options, **{'first_feature_use_penalties': first_feature_use_penalties}))

    # check that negative values are not accepted
    wrong_first_feature_use_penalties_array = np.array([0, 0, 0, 10, 0, 0, 0, -2])
    wrong_first_feature_use_penalties_list = [0, 0, 0, 10, 0, 0, 0, -2]
    wrong_first_feature_use_penalties_dict_1 = {3: 10, 7: -2}
    wrong_first_feature_use_penalties_dict_2 = {'DepTime': 10, 'Distance': -2}
    wrong_first_feature_use_penalties_string_1 = "(0,0,0,10,0,0,0,-2)"
    wrong_first_feature_use_penalties_string_2 = "3:10,7:-2"
    wrong_first_feature_use_penalties_string_3 = "DepTime:10,Distance:-2"
    for first_feature_use_penalties in [
        wrong_first_feature_use_penalties_array,
        wrong_first_feature_use_penalties_list,
        wrong_first_feature_use_penalties_dict_1,
        wrong_first_feature_use_penalties_dict_2,
        wrong_first_feature_use_penalties_string_1,
        wrong_first_feature_use_penalties_string_2,
        wrong_first_feature_use_penalties_string_3
    ]:
        with pytest.raises(CatBoostError):
            model = CatBoostClassifier(first_feature_use_penalties=first_feature_use_penalties, **common_options)
            model.fit(train_pool)
        with pytest.raises(CatBoostError):
            model = CatBoost({"first_feature_use_penalties": first_feature_use_penalties})
            iterations_list = [5, 10]
            model.grid_search(
                {
                    'iterations': iterations_list,
                },
                train_pool,
            )
        with pytest.raises(CatBoostError):
            model = CatBoost({"first_feature_use_penalties": first_feature_use_penalties})
            iterations_list = [5, 10]
            model.randomized_search(
                {
                    'iterations': iterations_list,
                },
                train_pool,
                n_iter=1,
            )
        with pytest.raises(CatBoostError):
            cv(train_pool, params=dict(common_options, **{'first_feature_use_penalties': first_feature_use_penalties}))


def test_penalties_coefficient_work():
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    most_important_feature_index = 3
    classifier_params = {
        'iterations': 5,
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'devices': '0',
        'loss_function': 'MultiClass',
    }

    model_without_feature_penalties = CatBoostClassifier(**classifier_params)
    model_without_feature_penalties.fit(pool)

    model_with_feature_penalties = CatBoostClassifier(
        first_feature_use_penalties={most_important_feature_index: 100},
        **classifier_params
    )
    model_with_feature_penalties.fit(pool)

    model_with_zero_feature_penalties = CatBoostClassifier(
        first_feature_use_penalties={most_important_feature_index: 100},
        penalties_coefficient=0,
        **classifier_params
    )
    model_with_zero_feature_penalties.fit(pool)

    assert any(model_without_feature_penalties.predict_proba(pool)[0] == model_with_zero_feature_penalties.predict_proba(pool)[0])
    assert any(model_without_feature_penalties.predict_proba(pool)[0] != model_with_feature_penalties.predict_proba(pool)[0])


@pytest.mark.parametrize('features', [[5, 7], [5]])
def test_partial_dependence(features):
    pool_file = 'higgs'
    pool = Pool(data_file(pool_file, 'train_small'), column_description=data_file(pool_file, 'train.cd'))
    model = CatBoostClassifier(iterations=100, learning_rate=0.03, max_ctr_complexity=1, devices='0')
    model.fit(pool)
    fimp_txt_path = test_output_path(FIMP_TXT_PATH)
    np.savetxt(fimp_txt_path, np.around(np.array(model.plot_partial_dependence(pool, features, plot=False)[0]).flatten(), 8))
    return local_canonical_file(fimp_txt_path)


@pytest.mark.parametrize('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
def test_per_object_feature_penalties_work(grow_policy):
    pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    most_important_feature_index = 3
    classifier_params = {
        'iterations': 5,
        'learning_rate': 0.03,
        'task_type': 'CPU',
        'loss_function': 'MultiClass',
        'grow_policy': grow_policy,
    }

    model_without_feature_penalties = CatBoostClassifier(**classifier_params)
    model_without_feature_penalties.fit(pool)
    importance_without_feature_penalties = model_without_feature_penalties.get_feature_importance(
        type=EFstrType.PredictionValuesChange,
        data=pool
    )[most_important_feature_index]

    model_with_feature_penalties = CatBoostClassifier(
        per_object_feature_penalties={most_important_feature_index: 100},
        **classifier_params
    )
    model_with_feature_penalties.fit(pool)
    importance_with_feature_penalties = model_with_feature_penalties.get_feature_importance(
        type=EFstrType.PredictionValuesChange,
        data=pool
    )[most_important_feature_index]

    assert importance_with_feature_penalties < importance_without_feature_penalties


def test_different_formats_of_per_object_feature_penalties():
    train_pool = Pool(AIRLINES_5K_TRAIN_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)
    test_pool = Pool(AIRLINES_5K_TEST_FILE, column_description=AIRLINES_5K_CD_FILE, has_header=True)

    per_row_feature_penalties_array = np.array([0, 0, 0, 10, 0, 0, 0, 2])
    per_row_feature_penalties_list = [0, 0, 0, 10, 0, 0, 0, 2]
    per_row_feature_penalties_dict_1 = {3: 10, 7: 2}
    per_row_feature_penalties_dict_2 = {'DepTime': 10, 'Distance': 2}
    per_row_feature_penalties_string_1 = "(0,0,0,10,0,0,0,2)"
    per_row_feature_penalties_string_2 = "3:10,7:2"
    per_row_feature_penalties_string_3 = "DepTime:10,Distance:2"

    common_options = dict(iterations=10)
    model1 = CatBoostClassifier(per_object_feature_penalties=per_row_feature_penalties_array, **common_options)
    model1.fit(train_pool)
    predictions1 = model1.predict(test_pool)
    for per_row_feature_penalties in [
        per_row_feature_penalties_list,
        per_row_feature_penalties_dict_1,
        per_row_feature_penalties_dict_2,
        per_row_feature_penalties_string_1,
        per_row_feature_penalties_string_2,
        per_row_feature_penalties_string_3
    ]:
        model2 = CatBoostClassifier(per_object_feature_penalties=per_row_feature_penalties, **common_options)
        model2.fit(train_pool)
        predictions2 = model2.predict(test_pool)
        assert all(predictions1 == predictions2)


LOAD_AND_QUANTIZE_TEST_PARAMS = {
    'querywise_without_params': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {},  # load_params
        {},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'querywise_pairs': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {'pairs': QUERYWISE_TRAIN_PAIRS_FILE},  # load_params
        {},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'querywise_feature_names': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {'feature_names': QUERYWISE_FEATURE_NAMES_FILE},  # load_params
        {},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'querywise_ignored_features': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {},  # load_params
        {'ignored_features': [4, 8, 15]},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'querywise_per_float_feature_quantization': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {},  # load_params
        {'per_float_feature_quantization': ['1:border_count=70']},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'querywise_border_count': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {},  # load_params
        {'border_count': 500},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'querywise_feature_border_type': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {},  # load_params
        {'feature_border_type': 'Median'},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'querywise_input_borders': (
        QUERYWISE_TRAIN_FILE,
        QUERYWISE_CD_FILE,
        {},  # load_params
        {'input_borders': QUERYWISE_QUANTIZATION_BORDERS_EXAMPLE},  # quantize_params
        False,  # subset_quantization_differs
    ),
    'float_with_nan_mode_max': (
        FLOAT_WITH_NAN_FILE,
        None,  # column_description
        {},  # load_params
        {'nan_mode': 'Max'},  # quantize_params
        True,  # subset_quantization_differs
    ),
    'higgs_5K_without_params': (
        HIGGS_5K_TRAIN_FILE,
        HIGGS_CD_FILE,
        {},  # load_params
        {},  # quantize_params
        True,  # subset_quantization_differs
    ),
}


@pytest.mark.parametrize(('pool_file', 'column_description', 'load_params', 'quantize_params',
                          'subset_quantization_differs'),
                         argvalues=list(LOAD_AND_QUANTIZE_TEST_PARAMS.values()), ids=list(LOAD_AND_QUANTIZE_TEST_PARAMS.keys()))
def test_pool_load_and_quantize(pool_file, column_description, load_params, quantize_params,
                                subset_quantization_differs):
    SMALL_BLOCK_SIZE = 500
    SMALL_SUBSET_SIZE_FOR_BUILD_BORDERS = 100
    quantized_pool = Pool(pool_file, column_description=column_description, **load_params)
    quantized_pool.quantize(**quantize_params)

    quantized_pool_small_subset = Pool(pool_file, column_description=column_description, **load_params)
    quantized_pool_small_subset.quantize(
        dev_max_subset_size_for_build_borders=SMALL_SUBSET_SIZE_FOR_BUILD_BORDERS,
        **quantize_params)

    quantize_on_load_params = quantize_params.copy()
    quantize_on_load_params.update(load_params)

    quantized_on_load_pool = quantize(pool_file, column_description=column_description, **quantize_on_load_params)
    quantized_on_load_pool_small_blocks = quantize(
        pool_file,
        column_description=column_description,
        dev_block_size=SMALL_BLOCK_SIZE,
        **quantize_on_load_params)
    quantized_on_load_pool_small_subset = quantize(
        pool_file,
        column_description=column_description,
        dev_max_subset_size_for_build_borders=SMALL_SUBSET_SIZE_FOR_BUILD_BORDERS,
        **quantize_on_load_params)
    quantized_on_load_pool_small_blocks_and_subset = quantize(
        pool_file,
        column_description=column_description,
        dev_block_size=SMALL_BLOCK_SIZE,
        dev_max_subset_size_for_build_borders=SMALL_SUBSET_SIZE_FOR_BUILD_BORDERS,
        **quantize_on_load_params)

    assert quantized_on_load_pool.is_quantized()
    assert quantized_pool == quantized_on_load_pool
    assert quantized_pool == quantized_on_load_pool_small_blocks

    assert (quantized_pool != quantized_pool_small_subset) == subset_quantization_differs
    assert quantized_pool_small_subset == quantized_on_load_pool_small_subset
    assert quantized_pool_small_subset == quantized_on_load_pool_small_blocks_and_subset

    if load_params or quantize_params:
        quantized_without_params_pool = Pool(pool_file, column_description=column_description)
        quantized_without_params_pool.quantize()
        assert quantized_pool != quantized_without_params_pool


def test_pool_load_and_quantize_unknown_param():
    with pytest.raises(CatBoostError):
        quantize(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE, this_param_is_unknown=123)


def test_quantize_unknown_param():
    pool = Pool(QUERYWISE_TRAIN_FILE, column_description=QUERYWISE_CD_FILE)
    with pytest.raises(CatBoostError):
        pool.quantize(this_param_is_unknown=123)


def test_iterate_leaf_indexes():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=.05, objective='Logloss')
    model.fit(train_pool)
    for _ in model.iterate_leaf_indexes(train_pool):
        pass


def test_plot_tree():
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(iterations=2, learning_rate=.05, objective='Logloss')
    model.fit(train_pool)
    model.plot_tree(0)


@pytest.mark.parametrize('params', [{},
                                    {'verbose': True},
                                    {'verbose': False},
                                    {'logging_level': 'Silent'},
                                    {'logging_level': 'Verbose'},
                                    {'metric_period': 5}])
def test_same_params(params):
    train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    model_path = test_output_path('model.bin')
    params['iterations'] = 10
    params['loss_function'] = 'Logloss'
    CatBoost(params).fit(train_pool).save_model(model_path)
    assert CatBoost().load_model(model_path).get_params() == params


@pytest.mark.parametrize('task', ['binclass', 'multiclass'])
@pytest.mark.parametrize('metric', ['TotalF1', 'MCC', 'F1', 'Precision', 'Recall'])
@pytest.mark.parametrize('use_weights', [True, False])
def test_eval_metric_with_weights(task_type, task, metric, use_weights):
    X = np.random.random()
    if task == 'multiclass' and metric in ('F1', 'Precision', 'Recall'):
        pytest.skip('Metric with multiple values is not allowed to use as eval_metric')
    np.random.seed(0)
    X = np.random.random(size=(100, 10))
    if task == 'binclass':
        y = [0] * 30 + [1] * 70
    else:
        y = [0] * 20 + [1] * 30 + [2] * 50
    np.random.shuffle(y)
    weight = np.random.random(size=100)
    pool = Pool(X, y, weight=weight)

    full_metric_name = '{}:use_weights={}'.format(metric, str(use_weights).lower())
    full_metric_name_inverse = '{}:use_weights={}'.format(metric, str(not use_weights).lower())
    model = CatBoostClassifier(
        loss_function='Logloss' if task == 'binclass' else 'MultiClass',
        iterations=1,
        eval_metric=full_metric_name,
        task_type=task_type
    )
    model.fit(pool, eval_set=pool)
    fit_metric = model.evals_result_['validation'][full_metric_name]
    eval_metric = model.eval_metrics(pool, full_metric_name)[full_metric_name]
    eval_metric_inverse = model.eval_metrics(pool, full_metric_name_inverse)[full_metric_name_inverse]
    assert not np.array_equal(eval_metric, eval_metric_inverse)
    assert np.array_equal(fit_metric, eval_metric)


@pytest.mark.parametrize('problem', ['Classifier', 'Regressor'])
def test_callbacks_early_stop(problem):
    np.random.seed(0)
    train_data = np.random.randint(0, 100, size=(100, 5))
    train_labels = np.random.randint(0, 2, size=100)
    if problem == 'Classifier':
        loss = 'Logloss'
        model = CatBoostClassifier(iterations=10, loss_function=loss)
    else:
        loss = 'RMSE'
        model = CatBoostRegressor(iterations=10, loss_function=loss)

    class EarlyStopCallback:
        def __init__(self, stop_iteration):
            self._stop_iteration = stop_iteration

        def after_iteration(self, info):
            return info.iteration != self._stop_iteration

    model.fit(train_data, train_labels, callbacks=[
        EarlyStopCallback(7),
        EarlyStopCallback(5),
        EarlyStopCallback(6)
    ])

    evals_result = model.get_evals_result()
    assert 'learn' in evals_result
    assert loss in evals_result['learn']
    assert len(evals_result['learn'][loss]) == 5


def test_callbacks_metrics():
    np.random.seed(0)

    def generate_dataset():
        return np.random.randint(0, 100, size=(100, 5)), np.random.randint(0, 2, size=100)

    train_data, train_labels = generate_dataset()
    validation_0 = generate_dataset()
    validation_1 = generate_dataset()
    metric_names = ['Logloss', 'CrossEntropy']
    model = CatBoostClassifier(iterations=10, custom_metric=metric_names)

    class MetricsCheckerCallback:
        def after_iteration(self, info):
            for dataset_name in ['learn', 'validation_0', 'validation_1']:
                assert dataset_name in info.metrics
                for metric_name in metric_names:
                    assert metric_name in info.metrics[dataset_name]
                    assert len(info.metrics[dataset_name][metric_name]) == info.iteration
            return True

    model.fit(train_data, train_labels,
              callbacks=[MetricsCheckerCallback()],
              eval_set=[validation_0, validation_1])


def test_fit_cat_features_type():
    X = DataFrame(
        data=np.random.randint(0, 100, size=(100, 5)),
        columns=['feature{}'.format(i) for i in range(5)]
    )
    y = np.random.randint(0, 2, size=100)

    model = CatBoostClassifier(iterations=2,
                               depth=2,
                               learning_rate=1,
                               loss_function='CrossEntropy',
                               logging_level='Silent')

    model.fit(X, y, cat_features=np.arange(3))
    model.fit(X, y, cat_features=[0, 1, 2])


def test_sklearn_meta_algo():
    from sklearn.calibration import CalibratedClassifierCV

    X_train = [[1, 2, 3, 4], [2, 3, 4, 5]]
    y_train = [1, 0]

    model = CatBoostClassifier()
    model.fit(X_train, y_train)

    cc_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    model = cc_model.fit(X_train, y_train)


def test_pool_with_timestamp(task_type):
    features, labels = generate_random_labeled_dataset(n_samples=20, n_features=5, labels=[0, 1])
    np.random.seed(42)
    timestamp = np.arange(20)
    np.random.shuffle(timestamp)

    pool = Pool(features, label=labels, timestamp=timestamp)
    model = CatBoostClassifier(
        has_time=True,
        iterations=2,
        learning_rate=0.03,
        task_type=task_type,
        devices='0'
    )
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


def test_pool_set_timestamp(task_type):
    pool = Pool(TRAIN_FILE, column_description=CD_FILE)
    np.random.seed(42)
    timestamp = np.arange(pool.num_row())
    np.random.shuffle(timestamp)
    pool.set_timestamp(timestamp)
    model = CatBoostClassifier(
        has_time=True,
        iterations=2,
        learning_rate=0.03,
        task_type=task_type,
        devices='0'
    )
    model.fit(pool)
    output_model_path = test_output_path(OUTPUT_MODEL_PATH)
    model.save_model(output_model_path)
    return compare_canonical_models(output_model_path)


@pytest.mark.parametrize('train_final_model', [True, False])
def test_select_features(task_type, train_final_model):
    learn = Pool(TRAIN_FILE, column_description=CD_FILE)
    test = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=10,
        learning_rate=0.03,
        task_type=task_type,
        devices='0'
    )
    summary = model.select_features(
        learn,
        eval_set=test,
        steps=1,
        train_final_model=train_final_model,
        features_for_select='0-16',
        num_features_to_select=10
    )
    assert len(summary['selected_features']) == 10
    assert len(summary['eliminated_features']) == 7
    if train_final_model:
        assert model.is_fitted()
        assert model.best_score_ != {}
        assert model.evals_result_ != {}
    else:
        assert not model.is_fitted()
        assert model.best_score_ == {}
        assert model.evals_result_ == {}


def test_select_features_with_custom_eval_metric():
    class CustomMetric(object):
        def get_final_error(self, error, weight):
            return 0.35

        def is_max_optimal(self):
            return True

        def evaluate(self, approxes, target, weight):
            return (0.0, 0.0)

    learn = Pool(TRAIN_FILE, column_description=CD_FILE)
    test = Pool(TEST_FILE, column_description=CD_FILE)
    model = CatBoostClassifier(
        iterations=10,
        learning_rate=0.03,
        use_best_model=False,
        eval_metric=CustomMetric()
    )
    model.select_features(
        learn,
        eval_set=test,
        steps=1,
        train_final_model=True,
        features_for_select='0-16',
        num_features_to_select=10
    )
    assert model.best_score_['validation']['CustomMetric'] == 0.35


@pytest.mark.parametrize(
    'train_final_model',
    [True, False],
    ids=['train_final_model=True', 'train_final_model=False']
)
@pytest.mark.parametrize(
    'algorithm',
    [
        EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange,
        EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
        EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    ],
    ids=[
        'algorithm=%s' % algorithm for algorithm in [
            'RecursiveByPredictionValuesChange',
            'RecursiveByLossFunctionChange',
            'RecursiveByShapValues'
        ]
    ]
)
def test_select_features_by_single_feature_tags(task_type, train_final_model, algorithm):
    # use querywise dataset because it contains more features to select from
    columns_metadata = read_cd(
        QUERYWISE_CD_FILE,
        data_file=QUERYWISE_TRAIN_FILE,
        canonize_column_types=True
    )

    learn_data = load_dataset_as_dataframe(QUERYWISE_TRAIN_FILE, columns_metadata)
    test_data = load_dataset_as_dataframe(QUERYWISE_TEST_FILE, columns_metadata)

    n_features = learn_data['features'].columns.size
    feature_names = [str(feature_idx) for feature_idx in range(n_features)]
    feature_tags = {}
    for feature_idx in range(n_features):
        feature_tags[str(feature_idx)] = {"features": [feature_idx], "cost": 1}

    learn = Pool(
        learn_data['features'],
        learn_data['target'],
        cat_features=columns_metadata['cat_feature_indices'],
        feature_names=feature_names,
        feature_tags=feature_tags
    )
    test = Pool(
        test_data['features'],
        test_data['target'],
        cat_features=columns_metadata['cat_feature_indices'],
        feature_names=feature_names,
        feature_tags=feature_tags
    )
    model_for_features = CatBoostRegressor(
        iterations=100,
        learning_rate=0.03,
        task_type=task_type,
        devices='0',
        logging_level='Debug'
    )
    summary_for_features = model_for_features.select_features(
        learn,
        eval_set=test,
        steps=2,
        train_final_model=train_final_model,
        algorithm=algorithm,
        features_for_select=list(range(20)),
        num_features_to_select=10
    )
    model_for_features_tags = CatBoostRegressor(
        iterations=100,
        learning_rate=0.03,
        task_type=task_type,
        devices='0',
        logging_level='Debug'
    )
    summary_for_features_tags = model_for_features_tags.select_features(
        learn,
        eval_set=test,
        steps=2,
        train_final_model=train_final_model,
        algorithm=algorithm,
        grouping=EFeaturesSelectionGrouping.ByTags,
        features_tags_for_select=[str(i) for i in range(20)],
        num_features_tags_to_select=10
    )
    assert set(summary_for_features['selected_features_names']) == set(summary_for_features_tags['selected_features_names'])
    assert set(summary_for_features['selected_features_names']) == set(summary_for_features_tags['selected_features_tags'])
    assert summary_for_features['eliminated_features_names'] == summary_for_features_tags['eliminated_features_names']
    assert summary_for_features['eliminated_features_names'] == summary_for_features_tags['eliminated_features_tags']


@pytest.mark.parametrize(
    'train_final_model',
    [True, False],
    ids=['train_final_model=True', 'train_final_model=False']
)
@pytest.mark.parametrize(
    'algorithm',
    [
        EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange,
        EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
        EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    ],
    ids=[
        'algorithm=%s' % algorithm for algorithm in [
            'RecursiveByPredictionValuesChange',
            'RecursiveByLossFunctionChange',
            'RecursiveByShapValues'
        ]
    ]
)
def test_select_features_by_multi_feature_tags(task_type, train_final_model, algorithm):
    n_features = 30
    learn_features, learn_labels = generate_random_labeled_dataset(
        n_samples=4000,
        n_features=n_features,
        labels=[0, 1],
        seed=1
    )
    test_features, test_labels = generate_random_labeled_dataset(
        n_samples=1500,
        n_features=n_features,
        labels=[0, 1],
        seed=2
    )

    feature_names = [str(feature_idx) for feature_idx in range(n_features)]
    feature_tags = {
        "tag0": {"features": [0, 1, 2, 10, 12, 15], "cost": 1},
        "tag1": {"features": [3, 7, 11, 13, 26, 27, 28], "cost": 2},
        "tag2": {"features": [4, 5], "cost": 1},
        "tag3": {"features": [6, 8, 9, 14], "cost": 3},
        "tag4": {"features": [16], "cost": 1},
        "tag5": {"features": [17, 18, 19, 22, 23, 25, 29], "cost": 4},
        "tag6": {"features": [20, 21, 24], "cost": 1}
    }

    learn = Pool(learn_features, learn_labels, feature_names=feature_names, feature_tags=feature_tags)
    test = Pool(test_features, test_labels, feature_names=feature_names, feature_tags=feature_tags)
    model = CatBoostClassifier(
        iterations=20,
        learning_rate=0.03,
        task_type=task_type,
        devices='0',
        logging_level='Debug'
    )
    summary = model.select_features(
        learn,
        eval_set=test,
        steps=2,
        train_final_model=train_final_model,
        algorithm=algorithm,
        grouping=EFeaturesSelectionGrouping.ByTags,
        features_tags_for_select=["tag0", "tag1", "tag2", "tag4", "tag6"],
        num_features_tags_to_select=3
    )

    summary_file_name = test_output_path('summary.json')
    with open(summary_file_name, 'w') as f:
        json.dump(summary, f, indent=4, sort_keys=True)

    return local_canonical_file(summary_file_name, diff_tool=get_limited_precision_json_diff_tool(1.e-6))


def test_embedding_features_data_list_with_data_with_features_order():
    pool1 = Pool(
        data=DataFrame(
            {
                'f0': [0, 1, 2],
                'f1': [3, 4, 5],
                '_embedding_feature_2': [[0.1, 0.2], [0.12, 0.3], [1.0, 0.0]],
                '_embedding_feature_3': [[0.8, 0.3, 0.1], [0.0, 0.2, 0.8], [1.0, 0.0, 0.5]]
            }
        ),
        embedding_features=[2, 3]
    )
    pool2 = Pool(
        data=DataFrame(
            {
                'f0': [0, 1, 2],
                'f1': [3, 4, 5]
            }
        ),
        embedding_features_data=[
            np.array([[0.1, 0.2], [0.12, 0.3], [1.0, 0.0]]),
            np.array([[0.8, 0.3, 0.1], [0.0, 0.2, 0.8], [1.0, 0.0, 0.5]])
        ],
        embedding_features=[2, 3]
    )
    assert _have_equal_features(pool1, pool2)


def test_embedding_features_data_dict_with_data_with_features_order():
    pool1 = Pool(
        data=DataFrame(
            {
                'f0': [0, 1, 2],
                'f1': [3, 4, 5],
                'f2': [[0.1, 0.2], [0.12, 0.3], [1.0, 0.0]],
                'f3': [[0.8, 0.3, 0.1], [0.0, 0.2, 0.8], [1.0, 0.0, 0.5]]
            }
        ),
        embedding_features=['f2', 'f3']
    )
    pool2 = Pool(
        data=DataFrame(
            {
                'f0': [0, 1, 2],
                'f1': [3, 4, 5]
            }
        ),
        embedding_features_data={
            'f2': np.array([[0.1, 0.2], [0.12, 0.3], [1.0, 0.0]]),
            'f3': np.array([[0.8, 0.3, 0.1], [0.0, 0.2, 0.8], [1.0, 0.0, 0.5]])
        },
        embedding_features=['f2', 'f3']
    )
    assert _have_equal_features(pool1, pool2)


def test_embedding_features_data_list_with_data_with_objects_order():
    pool1 = Pool(
        data=[
            [0, 3, [0.1, 0.2], [0.8, 0.3, 0.1], 6],
            [1, 4, [0.12, 0.3], [0.0, 0.2, 0.8], 7],
            [2, 5, [1.0, 0.0], [1.0, 0.0, 0.5], 8]
        ],
        embedding_features=[2, 3]
    )
    pool2 = Pool(
        data=[
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]
        ],
        embedding_features_data=[
            np.array([[0.1, 0.2], [0.12, 0.3], [1.0, 0.0]]),
            np.array([[0.8, 0.3, 0.1], [0.0, 0.2, 0.8], [1.0, 0.0, 0.5]])
        ],
        embedding_features=[2, 3]
    )

    assert _have_equal_features(pool1, pool2)


def test_embedding_features_data_dict_with_data_with_objects_order():
    pool1 = Pool(
        data=[
            [0, 3, [0.1, 0.2], [0.8, 0.3, 0.1], 6],
            [1, 4, [0.12, 0.3], [0.0, 0.2, 0.8], 7],
            [2, 5, [1.0, 0.0], [1.0, 0.0, 0.5], 8]
        ],
        embedding_features=[2, 3]
    )
    pool2 = Pool(
        data=[
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]
        ],
        embedding_features_data={
            2 : np.array([[0.1, 0.2], [0.12, 0.3], [1.0, 0.0]]),
            3 : np.array([[0.8, 0.3, 0.1], [0.0, 0.2, 0.8], [1.0, 0.0, 0.5]])
        },
        embedding_features=[2, 3]
    )

    assert _have_equal_features(pool1, pool2)


def test_pandas_integer_array():
    X = DataFrame({'feature': list(range(10))}, dtype=pd.Int64Dtype())
    y = list(range(10))
    cb = CatBoostRegressor(iterations=1)
    cb.fit(X, y)


@pytest.mark.parametrize('problem_type', ['classification', 'regression', 'ranking'])
def test_train_with_embedding_features(problem_type):
    train_pool = Pool(
        data=[
            [0.1, 3, [0.1, 0.2], [0.8, 0.3, 0.1], 0.12],
            [0.2, 4, [0.12, 0.3], [0.0, 0.2, 0.8], 0.0],
            [0.1, 5, [1.0, 0.0], [1.0, 0.0, 0.5], 0.2],
            [0.8, 2, [0.3, 0.1], [0.8, 0.2, 0.9], 0.3],
            [0.4, 1, [0.2, 0.0], [1.0, 0.92, 0.1], 0.4],
            [0.3, 2, [0.1, 0.22], [0.1, 0.22, 0.2], 0.0],
            [0.5, 3, [0.21, 0.3], [0.4, 0.6, 0.33], 0.25]
        ],
        label=[0, 1, 1, 0, 1, 0, 1] if problem_type == 'classification' else [0.0, 0.1, 0.2, 0.12, 0.3, 0.1, 0.13],
        group_id=[0, 0, 0, 1, 1, 1, 2],
        embedding_features=[2, 3]
    )
    test_pool = Pool(
        data=[
            [0.3, 2, [0.0, 0.14], [0.9, 0.0, 0.12], 0.22],
            [0.4, 1, [0.2, 0.8], [0.0, 0.21, 0.7], 0.04],
            [0.2, 7, [0.3, 0.11], [0.0, 0.2, 0.4], 0.7],
            [0.0, 3, [0.5, 0.3], [1.0, 0.93, 0.1], 0.8]
        ],
        label=[1, 0, 0, 1] if problem_type == 'classification' else [0.1, 0.0, 0.3, 0.2],
        group_id=[0, 0, 1, 1],
        embedding_features=[2, 3]
    )

    model = CatBoost(
        {
            'loss_function' : {
                'classification': 'Logloss',
                'regression': 'RMSE',
                'ranking': 'YetiRank'
            }[problem_type],
            'iterations' : 5
        }
    )
    model.fit(train_pool, eval_set=test_pool)

    preds = model.predict(test_pool)
    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(preds))
    return local_canonical_file(preds_path)


@pytest.mark.parametrize(
    'estimator_type',
    ['classifier', 'regressor'],
    ids=['estimator_type=classifier', 'estimator_type=regressor']
)
@pytest.mark.parametrize(
    'with_cat_features',
    [False, True],
    ids=['with_cat_features=False', 'with_cat_features=True']
)
def test_eval_fraction_on_ndarray(estimator_type, with_cat_features):
    if estimator_type == 'classifier':
        loss_function = 'Logloss'
    else:
        loss_function = 'RMSE'

    if with_cat_features:
        features_dtype = np.int32
        cat_features = [0, 7, 11]
        lower_bound = -32000
        upper_bound = 32000
    else:
        features_dtype = np.float32
        cat_features = []
        lower_bound = -1.0
        upper_bound = 1.0

    n_features = 20

    for order in ('C', 'F'):
        all_features_data, all_labels = generate_random_labeled_dataset(
            n_samples=100,
            n_features=n_features,
            labels=[0, 1],
            features_dtype=features_dtype,
            features_range=(lower_bound, upper_bound),
            features_order=order
        )

        train_features_data = all_features_data[:75]
        train_labels = all_labels[:75]
        eval_features_data = all_features_data[75:]
        eval_labels = all_labels[75:]

        params = {'iterations': 10, 'has_time': True, 'cat_features': cat_features, 'loss_function': loss_function}

        model_on_splitted = CatBoost(params)
        model_on_splitted.fit(train_features_data, train_labels, eval_set=(eval_features_data, eval_labels))

        params['eval_fraction'] = 0.25
        model_on_all = CatBoost(params)
        model_on_all.fit(all_features_data, all_labels)

        assert model_on_splitted == model_on_all

    # canonized result with shuffling

    all_features_data, all_labels = generate_random_labeled_dataset(
        n_samples=100,
        n_features=n_features,
        labels=[0, 1],
        features_dtype=features_dtype,
        features_range=(lower_bound, upper_bound)
    )

    params = {'iterations': 10, 'cat_features': cat_features, 'loss_function': loss_function, 'eval_fraction': 0.25}
    model_on_all = CatBoost(params)
    model_on_all.fit(all_features_data, all_labels)

    preds = model_on_all.predict(all_features_data)

    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(preds))
    return local_canonical_file(preds_path)


@pytest.mark.parametrize(
    'estimator_type',
    ['classifier', 'regressor'],
    ids=['estimator_type=classifier', 'estimator_type=regressor']
)
@pytest.mark.parametrize(
    'with_cat_features',
    [False, True],
    ids=['with_cat_features=False', 'with_cat_features=True']
)
def test_eval_fraction_on_pool(estimator_type, with_cat_features):
    if estimator_type == 'classifier':
        loss_function = 'Logloss'
    else:
        loss_function = 'RMSE'

    if with_cat_features:
        features_dtype = np.int32
        cat_features = [0, 7, 11]
        lower_bound = -32000
        upper_bound = 32000
    else:
        features_dtype = np.float32
        cat_features = []
        lower_bound = -1.0
        upper_bound = 1.0

    n_features = 20
    for order in ('C', 'F'):
        all_features_data, all_labels = generate_random_labeled_dataset(
            n_samples=100,
            n_features=n_features,
            labels=[0, 1],
            features_dtype=features_dtype,
            features_range=(lower_bound, upper_bound),
            features_order=order
        )

        all_pool = Pool(all_features_data, label=all_labels, cat_features=cat_features)
        train_pool = Pool(all_features_data[:75], label=all_labels[:75], cat_features=cat_features)
        eval_pool = Pool(all_features_data[75:], label=all_labels[75:], cat_features=cat_features)

        params = {'iterations': 10, 'has_time': True, 'cat_features': cat_features, 'loss_function': loss_function}

        model_on_splitted = CatBoost(params)
        model_on_splitted.fit(train_pool, eval_set=eval_pool)

        params['eval_fraction'] = 0.25
        model_on_all = CatBoost(params)
        model_on_all.fit(all_pool)

        assert model_on_splitted == model_on_all

    # canonized result with shuffling

    all_features_data, all_labels = generate_random_labeled_dataset(
        n_samples=100,
        n_features=n_features,
        labels=[0, 1],
        features_dtype=features_dtype,
        features_range=(lower_bound, upper_bound)
    )
    all_pool = Pool(all_features_data, label=all_labels, cat_features=cat_features)

    params = {'iterations': 10, 'cat_features': cat_features, 'loss_function': loss_function, 'eval_fraction': 0.25}
    model_on_all = CatBoost(params)
    model_on_all.fit(all_pool)

    preds = model_on_all.predict(all_features_data)

    preds_path = test_output_path(PREDS_TXT_PATH)
    np.savetxt(preds_path, np.array(preds))
    return local_canonical_file(preds_path)
