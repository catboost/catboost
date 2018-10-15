import numpy as np
import os
import pytest
import re
import yatest

from catboost import Pool, CatBoost, CatBoostClassifier
from catboost_pytest_lib import data_file, local_canonical_file

CATBOOST_APP_PATH = yatest.common.binary_path('catboost')


def model_diff_tool():
    return yatest.common.binary_path("catboost/tools/model_comparator/model_comparator")


def _get_train_test_cd_path(dataset):
    train_path = data_file(dataset, 'train_small')
    test_path = data_file(dataset, 'test_small')
    cd_path = data_file(dataset, 'train.cd')
    return (train_path, test_path, cd_path)


def _get_train_test_pool(dataset):
    train_path, test_path, cd_path = _get_train_test_cd_path(dataset)
    train_pool = Pool(train_path, column_description=cd_path)
    test_pool = Pool(test_path, column_description=cd_path)
    return (train_pool, test_pool)


def _get_cpp_py_cbm_model(dataset):
    train_path, _, cd_path = _get_train_test_cd_path(dataset)
    basename = yatest.common.test_output_path('model')
    cmd = [CATBOOST_APP_PATH, 'fit',
           '-f', train_path,
           '--cd', cd_path,
           '-i', '100',
           '-r', '1234',
           '-m', basename,
           '--model-format', 'CPP',
           '--model-format', 'Python',
           '--model-format', 'CatboostBinary',
           ]
    yatest.common.execute(cmd)
    assert os.path.exists(basename + '.cpp')
    assert os.path.exists(basename + '.py')
    assert os.path.exists(basename + '.bin')
    return (basename + '.cpp', basename + '.py', basename + '.bin')


def _split_features(features, cat_features_indices, hash_to_string):
    float_features = []
    cat_features = []
    for i in range(len(features)):
        if i in cat_features_indices:
            cat_features.append(hash_to_string[features[i]])
        else:
            float_features.append(features[i])
    return float_features, cat_features


def _check_data(data1, data2, rtol=0.001):
    return np.all(np.isclose(data1, data2, rtol=rtol, equal_nan=True))


@pytest.mark.parametrize('dataset', ['adult', 'higgs'])
def test_cpp_export(dataset):
    model_cpp, _, model_cbm = _get_cpp_py_cbm_model(dataset)

    # check that the .cpp file compiles

    if os.name == 'posix':
        compiler = ['g++', '-std=c++14']
    else:
        compiler = ['cl.exe']

    dot = yatest.common.test_output_path('.')
    util = os.path.join(dot, 'util')
    util_digest = os.path.join(dot, 'util', 'digest')
    os.mkdir(util)
    os.mkdir(util_digest)
    city_h = os.path.join(util_digest, 'city.h')
    open(city_h, 'w').write("""/* dummy city.h */
        unsigned long long CityHash64(const char*, size_t);
        """)
    compile_cmd = compiler + [
        '-c',
        '-I', yatest.common.test_output_path('.'),
        model_cpp
    ]

    try:
        yatest.common.execute(compile_cmd)
    except OSError as e:
        if re.search(r"No such file or directory.*'{}'".format(re.escape(compiler[0])), str(e)):
            print('We ignore `compiler not found` error: {}'.format(str(e)))
        else:
            raise

    # TODO(dbakshee): build an applicator and check that the .cpp model applies correctly

    return [local_canonical_file(model_cpp),
            local_canonical_file(model_cbm, diff_tool=model_diff_tool())]


def _predict_python(test_pool, apply_catboost_model):
    pred_python = []
    cat_feature_indices = test_pool.get_cat_feature_indices()
    cat_feature_hash = test_pool.get_cat_feature_hash_to_string()
    for test_line in test_pool.get_features():
        float_features, cat_features = _split_features(test_line, cat_feature_indices, cat_feature_hash)
        pred_python.append(apply_catboost_model(float_features, cat_features))
    return pred_python


@pytest.mark.parametrize('dataset', ['adult', 'higgs'])
def test_python_export_from_app(dataset):
    _, test_pool = _get_train_test_pool(dataset)
    _, model_py, model_cbm = _get_cpp_py_cbm_model(dataset)

    model = CatBoost()
    model.load_model(model_cbm)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')

    scope = {}
    execfile(model_py, scope)
    pred_python = _predict_python(test_pool, scope['apply_catboost_model'])

    assert _check_data(pred_model, pred_python)


@pytest.mark.parametrize('iterations', [2, 40])
@pytest.mark.parametrize('dataset', ['adult', 'higgs'])
def test_python_export_from_python(dataset, iterations):
    train_pool, test_pool = _get_train_test_pool(dataset)

    model = CatBoost({'iterations': iterations, 'random_seed': 0})
    model.fit(train_pool)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')

    model_py = yatest.common.test_output_path('model.py')
    model.save_model(model_py, format="python", pool=train_pool)

    scope = {}
    execfile(model_py, scope)
    pred_python = _predict_python(test_pool, scope['apply_catboost_model'])

    assert _check_data(pred_model, pred_python)


@pytest.mark.parametrize('dataset', ['adult', 'higgs'])
def test_python_after_load(dataset):
    train_pool, test_pool = _get_train_test_pool(dataset)
    model = CatBoostClassifier(iterations=40, random_seed=0)
    model.fit(train_pool)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')

    model_cbm = yatest.common.test_output_path('model.cbm')
    model.save_model(model_cbm)

    model_loaded = CatBoostClassifier()
    model_loaded.load_model(model_cbm)

    model_py = yatest.common.test_output_path('model.py')
    model_loaded.save_model(model_py, format="python", pool=train_pool)
    pred_model_loaded = model_loaded.predict(test_pool, prediction_type='RawFormulaVal')

    scope = {}
    execfile(model_py, scope)
    pred_python = _predict_python(test_pool, scope['apply_catboost_model'])

    assert _check_data(pred_model, pred_python)
    assert _check_data(pred_model_loaded, pred_python)
