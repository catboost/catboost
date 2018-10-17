import numpy as np
import os
import pytest
import re
import yatest

from catboost import Pool, CatBoost, CatBoostClassifier
from catboost_pytest_lib import data_file

CATBOOST_APP_PATH = yatest.common.binary_path('catboost')
APPROXIMATE_DIFF_PATH = yatest.common.binary_path('limited_precision_dsv_diff')


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
    _, test_path, cd_path = _get_train_test_cd_path(dataset)

    # form the commands we are going to run

    applicator_cpp = yatest.common.source_path('catboost/libs/model/model_export/ut/applicator.cpp')
    applicator_exe = yatest.common.test_output_path('applicator.exe')
    predictions_by_catboost_path = yatest.common.test_output_path('predictions_by_catboost.txt')
    predictions_path = yatest.common.test_output_path('predictions.txt')

    if os.name == 'posix':
        compile_cmd = ['g++', '-std=c++14', '-o', applicator_exe]
    else:
        compile_cmd = ['cl.exe', '-Fe' + applicator_exe]
    compile_cmd += [applicator_cpp, model_cpp]
    apply_cmd = [applicator_exe, test_path, cd_path, predictions_path]
    calc_cmd = [CATBOOST_APP_PATH, 'calc',
                '-m', model_cbm,
                '--input-path', test_path,
                '--cd', cd_path,
                '--output-path', predictions_by_catboost_path,
                ]
    compare_cmd = [APPROXIMATE_DIFF_PATH,
                   '--have-header',
                   '--diff-limit', '1e-6',
                   predictions_path,
                   predictions_by_catboost_path,
                   ]

    try:
        yatest.common.execute(compile_cmd)
        yatest.common.execute(apply_cmd)
        yatest.common.execute(calc_cmd)
        yatest.common.execute(compare_cmd)
    except OSError as e:
        if re.search(r"No such file or directory.*'{}'".format(re.escape(compile_cmd[0])), str(e)):
            pytest.xfail(reason='We ignore `compiler not found` error: {}\n'.format(str(e)))
        else:
            raise


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
