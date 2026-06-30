import numpy as np
import os
import pytest
import re
import yatest

from catboost import Pool, CatBoost, CatBoostClassifier
from catboost_pytest_lib import data_file, load_pool_features_as_df

CATBOOST_APP_PATH = yatest.common.binary_path('catboost/app/catboost')
APPROXIMATE_DIFF_PATH = yatest.common.binary_path('catboost/tools/limited_precision_dsv_diff/limited_precision_dsv_diff')


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


def __get_train_loss_function(dataset):
    return {
        'adult': 'Logloss',
        'covertype': 'MultiClass'
    }.get(dataset, 'RMSE')


def _get_cpp_py_cbm_model(dataset, parameters=[]):
    train_path, _, cd_path = _get_train_test_cd_path(dataset)
    basename = yatest.common.test_output_path('model')
    cmd = [CATBOOST_APP_PATH, 'fit',
           '-f', train_path,
           '--cd', cd_path,
           '--loss-function', __get_train_loss_function(dataset),
           '-i', '100',
           '-r', '1234',
           '-m', basename,
           '--model-format', 'CPP',
           '--model-format', 'Python',
           '--model-format', 'CatboostBinary',
           ] + parameters
    yatest.common.execute(cmd)
    assert os.path.exists(basename + '.cpp')
    assert os.path.exists(basename + '.py')
    assert os.path.exists(basename + '.bin')
    return (basename + '.cpp', basename + '.py', basename + '.bin')


def _check_data(data1, data2, rtol=0.001):
    return np.all(np.isclose(data1, data2, rtol=rtol, equal_nan=True))


@pytest.mark.parametrize(
    'dataset,parameters',
    [
        ('adult', []),
        ('adult', ['-I', '3']),
        ('higgs', []),
        ('covertype', [])
    ]
)
def test_cpp_export(dataset, parameters):
    model_cpp, _, model_cbm = _get_cpp_py_cbm_model(dataset, parameters)
    _, test_path, cd_path = _get_train_test_cd_path(dataset)

    # form the commands we are going to run

    applicator_cpp = yatest.common.source_path('catboost/libs/model/model_export/ut/applicator.cpp')
    applicator_exe = yatest.common.test_output_path('applicator.exe')
    predictions_by_catboost_path = yatest.common.test_output_path('predictions_by_catboost.txt')
    predictions_path = yatest.common.test_output_path('predictions.txt')

    is_multiclass_model = __get_train_loss_function(dataset) == 'MultiClass'

    if os.name == 'posix':
        compile_cmd = ['g++', '-std=c++14', '-o', applicator_exe]
    else:
        compile_cmd = ['cl.exe', '-Fe' + applicator_exe]
    compile_cmd += [applicator_cpp, model_cpp]
    apply_cmd = [applicator_exe, test_path, cd_path, predictions_path]
    if is_multiclass_model:
        model = CatBoost()
        model.load_model(model_cbm)
        apply_cmd.append(','.join(model.classes_))

    calc_cmd = [CATBOOST_APP_PATH, 'calc',
                '-m', model_cbm,
                '--input-path', test_path,
                '--cd', cd_path,
                '--output-path', predictions_by_catboost_path,
                '--prediction-type', 'RawFormulaVal'
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


def test_read_model_after_train():
    train_path, test_path, cd_path = _get_train_test_cd_path('adult')
    eval_file = yatest.common.test_output_path('eval-file')
    cmd = [
        CATBOOST_APP_PATH, 'fit',
        '-f', train_path,
        '--cd', cd_path,
        '-t', test_path,
        '-i', '100',
        '--eval-file', eval_file
    ]
    try:
        yatest.common.execute(
            cmd + [
                '--model-format', 'CPP',
                '--model-format', 'Python',
            ]
        )
    except Exception as e:
        if 'All chosen model formats not supported deserialization' not in str(e):
            raise
        yatest.common.execute(
            cmd + [
                '--model-format', 'CPP',
                '--model-format', 'CatboostBinary',
            ]
        )

        yatest.common.execute(
            cmd + [
                '--model-format', 'CatboostBinary',
                '--model-format', 'Python',
            ]
        )
        return
    assert False


def _predict_python_on_test(dataset, apply_catboost_model_function):
    features_data, cat_feature_indices = load_pool_features_as_df(
        data_file(dataset, 'test_small'),
        data_file(dataset, 'train.cd')
    )
    float_feature_indices = [i for i in range(features_data.shape[1]) if i not in cat_feature_indices]

    pred_python = []

    for row_idx in range(features_data.shape[0]):
        # can't pass *_feature_indices to .iloc directly because it will lose the proper type
        float_features = [features_data.iloc[row_idx, col_idx] for col_idx in float_feature_indices]
        cat_features = [features_data.iloc[row_idx, col_idx] for col_idx in cat_feature_indices]
        pred_python.append(apply_catboost_model_function(float_features, cat_features))

    return pred_python


@pytest.mark.parametrize('dataset', ['adult', 'higgs', 'covertype'])
def test_python_export_from_app(dataset):
    _, test_pool = _get_train_test_pool(dataset)
    _, model_py, model_cbm = _get_cpp_py_cbm_model(dataset)

    model = CatBoost()
    model.load_model(model_cbm)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')

    is_multiclass_model = __get_train_loss_function(dataset) == 'MultiClass'

    scope = {}
    exec(open(model_py).read(), scope)  # noqa
    pred_python = _predict_python_on_test(
        dataset,
        scope['apply_catboost_model_multi' if is_multiclass_model else 'apply_catboost_model']
    )

    assert _check_data(pred_model, pred_python)


@pytest.mark.parametrize('iterations', [2, 40])
@pytest.mark.parametrize('dataset', ['adult', 'higgs', 'covertype'])
def test_python_export_from_python(dataset, iterations):
    train_pool, test_pool = _get_train_test_pool(dataset)

    model = CatBoost(
        {'iterations': iterations, 'random_seed': 0, 'loss_function': __get_train_loss_function(dataset)}
    )
    model.fit(train_pool)
    pred_model = model.predict(test_pool, prediction_type='RawFormulaVal')

    model_py = yatest.common.test_output_path('model.py')
    model.save_model(model_py, format="python", pool=train_pool)

    is_multiclass_model = __get_train_loss_function(dataset) == 'MultiClass'

    scope = {}
    exec(open(model_py).read(), scope)  # noqa
    pred_python = _predict_python_on_test(
        dataset,
        scope['apply_catboost_model_multi' if is_multiclass_model else 'apply_catboost_model']
    )

    assert _check_data(pred_model, pred_python)


@pytest.mark.parametrize('dataset', ['adult', 'higgs', 'covertype'])
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

    is_multiclass_model = __get_train_loss_function(dataset) == 'MultiClass'

    scope = {}
    exec(open(model_py).read(), scope)  # noqa
    pred_python = _predict_python_on_test(
        dataset,
        scope['apply_catboost_model_multi' if is_multiclass_model else 'apply_catboost_model']
    )

    assert _check_data(pred_model, pred_python)
    assert _check_data(pred_model_loaded, pred_python)
